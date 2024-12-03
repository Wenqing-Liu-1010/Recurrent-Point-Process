import numpy as np
import torch
from torch.utils.data import DataLoader
from model import GRUPointProcess, PointProcessDataset, train_model

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

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成Hawkes过程数据
    print("Generating Hawkes process data...")
    T_train, score_train = generate_hawkes1()
    T_test, score_test = generate_hawkes2()
    
    print(f"Training data score: {score_train}")
    print(f"Testing data score: {score_test}")
    
    time_step = 20
    batch_size = 32
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
    
    # 创建模型
    model = GRUPointProcess(
        time_step=time_step,
        size_gru=32,
        size_nn=32,
        size_layer_chfn=2
    )
    
    # 训练模型
    train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=20,
        learning_rate=0.001
    )
