import numpy as np
import torch
from torch.utils.data import DataLoader
from model import GRUPointProcess, PointProcessDataset, train_model
import argparse
import torch.nn.functional as F
import pandas as pd

def simulate_hawkes(n, mu, alpha, beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * (1 - np.exp(-beta[0]*step)) / beta[0]
        l_trg_Int2 += l_trg2 * (1 - np.exp(-beta[1]*step)) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l:  # accept
            T.append(x)
            LL.append(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
    
    return [np.array(T), np.array(LL)]

def generate_hawkes1():
    [T, LL] = simulate_hawkes(100000, 0.2, [0.8, 0.0], [1.0, 20.0])
    score = -LL[80000:].mean()
    return [T, score]

def generate_hawkes2():
    [T, LL] = simulate_hawkes(100000, 0.2, [0.4, 0.4], [1.0, 20.0])
    score = -LL[80000:].mean()
    return [T, score]

def generate_stationary_poisson_process(num_samples, lambda_):
    intervals = np.random.exponential(scale=1/lambda_, size=num_samples)
    times = np.cumsum(intervals)
    return times

def evaluate_model(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            history = batch['history'].to(device)
            elapsed = batch['elapsed'].to(device)
            l, Int_l = model(history, elapsed)
            loss = model.compute_loss(l, Int_l)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches

def compute_true_intensity_hawkes(history, elapsed, mu, alpha, beta):
    intensity = mu
    for t in history:
        dt = elapsed - t
        if dt > 0:
            intensity += alpha[0] * beta[0] * np.exp(-beta[0] * dt)
            intensity += alpha[1] * beta[1] * np.exp(-beta[1] * dt)
    return intensity

def compute_true_cumulative_intensity_hawkes(history, elapsed, mu, alpha, beta):
    """计算真实的Hawkes累积强度函数值"""
    cum_intensity = mu * elapsed
    for t in history:
        dt = elapsed - t
        if dt > 0:  # 只考虑历史事件的影响
            cum_intensity += alpha[0] * (1 - np.exp(-beta[0] * dt))
            cum_intensity += alpha[1] * (1 - np.exp(-beta[1] * dt))
    return cum_intensity

def compute_true_intensity_poisson(history, elapsed, lambda_):
    """计算真实的泊松强度函数值"""
    return lambda_

def compute_true_cumulative_intensity_poisson(history, elapsed, lambda_):
    """计算真实的泊松累积强度函数值"""
    return lambda_ * elapsed

def evaluate_model(model, test_data, time_step, batch_size=1024):
    device = next(model.parameters()).device
    model.eval()
    dT = np.diff(test_data)
    n = len(dT)
    if n <= time_step:
        print("Warning: Not enough test data points")
        return float('inf')
    total_nll = 0
    count = 0
    with torch.no_grad():
        for i in range(0, n - time_step, batch_size):
            end_idx = min(i + batch_size, n - time_step)
            batch_size_actual = end_idx - i
            histories = []
            elapsed_times = []
            for j in range(i, end_idx):
                history = test_data[j:j+time_step]
                next_time = test_data[j+time_step]
                elapsed = next_time - history[-1]
                
                histories.append(history.reshape(-1, 1))
                elapsed_times.append([elapsed])
        
            history_tensor = torch.FloatTensor(histories).to(device)
            elapsed_tensor = torch.FloatTensor(elapsed_times).to(device)
            
            try:
                gru_out, _ = model.gru(history_tensor)
                gru_out = gru_out[:, -1, :]
                hidden_tau = model.elapsed_time_linear(elapsed_tensor)
                hidden_gru = model.gru_linear(gru_out)
                hidden = torch.tanh(hidden_tau + hidden_gru)
                for layer in model.hidden_layers:
                    hidden = torch.tanh(layer(hidden))
                Int_l = F.softplus(model.output_layer(hidden)) + 1e-6
                epsilon = 1e-6
                elapsed_plus = elapsed_tensor + epsilon
                hidden_tau_plus = model.elapsed_time_linear(elapsed_plus)
                hidden_plus = torch.tanh(hidden_tau_plus + hidden_gru)
                for layer in model.hidden_layers:
                    hidden_plus = torch.tanh(layer(hidden_plus))
                Int_l_plus = F.softplus(model.output_layer(hidden_plus)) + 1e-6
                l = (Int_l_plus - Int_l) / epsilon
                l = torch.clamp(l, min=1e-6, max=1e6)
                nll = -(torch.log(l + 1e-10) - Int_l)
                total_nll += nll.sum().item()
                count += batch_size_actual
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                continue
    if count == 0:
        return float('inf')
    
    return total_nll / count


def predict_median_intervals(model, test_data, time_step, batch_size=1024):
    device = next(model.parameters()).device
    model.eval()
    dT_test = np.diff(test_data)
    n = len(dT_test)
    
    if n <= time_step:
        print("Warning: Not enough test data points")
        return None, None, float('inf')
    histories = []
    actual_intervals = []
    
    for i in range(n - time_step):
        history = dT_test[i:i+time_step]
        next_interval = dT_test[i+time_step]
        
        histories.append(history.reshape(-1, 1))
        actual_intervals.append(next_interval)
    
    histories = np.array(histories)
    actual_intervals = np.array(actual_intervals)
    
    # 初始化二分查找的边界
    mean_interval = np.mean(dT_test)
    x_left = 1e-4 * mean_interval * np.ones_like(actual_intervals)
    x_right = 100 * mean_interval * np.ones_like(actual_intervals)
    
    with torch.no_grad():
        for iteration in range(13):  # 13次迭代
            x_center = (x_left + x_right) / 2
            
            # 分批处理预测
            all_cumulative_intensities = []
            
            for j in range(0, len(histories), batch_size):
                end_idx = min(j + batch_size, len(histories))
                
                # 确保输入维度正确
                batch_histories = torch.FloatTensor(histories[j:end_idx]).to(device)  # [batch_size, time_step, 1]
                batch_x_center = torch.FloatTensor(x_center[j:end_idx]).reshape(-1, 1).to(device)  # [batch_size, 1]
                
                # GRU处理历史序列
                gru_out, _ = model.gru(batch_histories)  # [batch_size, time_step, size_gru]
                gru_out = gru_out[:, -1, :]  # [batch_size, size_gru]
                
                # 计算隐藏状态
                hidden_tau = model.elapsed_time_linear(batch_x_center)  # [batch_size, size_nn]
                hidden_gru = model.gru_linear(gru_out)  # [batch_size, size_nn]
                hidden = torch.tanh(hidden_tau + hidden_gru)  # [batch_size, size_nn]
                
                # 通过隐藏层
                for layer in model.hidden_layers:
                    hidden = torch.tanh(layer(hidden))
                
                # 计算累积强度
                Int_l = F.softplus(model.output_layer(hidden)) + 1e-6  # [batch_size, 1]
                all_cumulative_intensities.extend(Int_l.cpu().numpy().flatten())
            
            cumulative_intensities = np.array(all_cumulative_intensities)
            
            # 更新二分查找的边界
            x_left = np.where(cumulative_intensities < np.log(2), x_center, x_left)
            x_right = np.where(cumulative_intensities >= np.log(2), x_center, x_right)
    
    # 计算最终预测和误差
    predicted_intervals = (x_left + x_right) / 2
    absolute_errors = np.abs(actual_intervals - predicted_intervals)
    mean_absolute_error = absolute_errors.mean()
    
    return predicted_intervals, actual_intervals, mean_absolute_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, choices=['hawkes1',"hawkes" ,'stationary_poisson'], default='hawkes', 
                        help="Choose the data generation process: 'hawkes1' or 'stationary_poisson'")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if args.process == 'hawkes1':
        print("Generating Hawkes1 process data...")
        T_train, score_train = generate_hawkes1()
        T_test, score_test = generate_hawkes2()
        
        train_params = {
            'mu': 0.2, 
            'alpha': [0.8, 0.0], 
            'beta': [1.0, 20.0]
        }
        test_params = {
            'mu': 0.2, 
            'alpha': [0.4, 0.4], 
            'beta': [1.0, 20.0]
        }
    if args.process == "hawkes":
        df_train=pd.read_csv("/mnt/lia/scratch/wenqliu/ee556-2/master-project/point_process/mu_0.9_alpha_0.8_beta_1.0_T_10000_cluster.csv")
        df_test=pd.read_csv("/mnt/lia/scratch/wenqliu/ee556-2/master-project/point_process/mu_0.9_alpha_0.8_beta_1.0_T_1000_cluster.csv")
        T_train = df_train["time"].values
        T_test = df_test["time"].values

    else: 
        print("Generating Stationary Poisson process data...")
        lambda_ = 1.0
        T_train = generate_stationary_poisson_process(80000, lambda_)
        T_test = generate_stationary_poisson_process(20000, lambda_)
        score_train = score_test = -np.log(lambda_) + lambda_
        train_params = test_params = {'lambda_': lambda_}

    # print(f"Training data theoretical score: {score_train}")
    # print(f"Testing data theoretical score: {score_test}")
    
    time_step = 20
    batch_size = 256
    train_dataset = PointProcessDataset(T_train, time_step)
    test_dataset = PointProcessDataset(T_test, time_step)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    model = GRUPointProcess(
        time_step=time_step,
        size_gru=64,
        size_nn=64,
        size_layer_chfn=2
    ).to(device)
    
    print("\nStarting model training...")
    train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=20,
        learning_rate=0.001
    )
    # torch.save(model.state_dict(), "SPP_model.pth")
    # print("\nEvaluating on test set...")
    test_nll = evaluate_model(model, T_test, time_step)
    print(f"Test set negative log-likelihood: {test_nll:.4f}")
    
    # lambda_ = 1.0
    # theoretical_nll = -np.log(lambda_) + lambda_
    # print(f"Theoretical negative log-likelihood: {theoretical_nll:.4f}")
    # print(f"Difference from theoretical: {abs(test_nll - theoretical_nll):.4f}")

    print("\nPredicting median intervals...")
    predicted_intervals, actual_intervals, mae = predict_median_intervals(model, T_test, time_step)

    if mae != float('inf'):
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # # 计算其他统计量
        # relative_error = np.abs(predicted_intervals - actual_intervals) / actual_intervals
        # print(f"Mean Relative Error: {relative_error.mean():.4f}")
        # print(f"Median Relative Error: {np.median(relative_error):.4f}")
    else:
        print("Prediction failed")
if __name__ == "__main__":
    main()
