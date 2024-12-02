import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from generation import MarkedIntensityHomogenuosPoisson, generate_samples_marked
from torch.utils.data import Dataset, DataLoader
import sys

# 参数设置
BATCH_SIZE = 256
MAX_STEPS = 300
EPOCHS = 100
REG = 0.1
LR = 1e-4
TYPE = sys.argv[1] if len(sys.argv) > 1 else 'joint'  # 模型类型: joint/event/timeseries
NUM_STEPS_TIMESERIES = 7
TIMESERIES_FEATURE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)

class PointProcessDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.max_len = max(len(seq) for seq in sequences)
        print(f"Dataset max length: {self.max_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded_seq = np.zeros((self.max_len, 2))
        padded_seq[:len(seq)] = np.array(seq)
        seq_len = len(seq)
        
        # 为每个时间点创建一个时间序列特征
        time_series = torch.ones(self.max_len, TIMESERIES_FEATURE)
        
        return {
            'sequence': torch.FloatTensor(padded_seq),
            'length': torch.LongTensor([seq_len]),
            'time_series': time_series
        }

class RNNPP(nn.Module):
    def __init__(self, 
                 num_classes=7,
                 state_size_event=16,
                 state_size_timeseries=32,
                 loss_type='mse',
                 model_type='joint',
                 reg=REG):
        super(RNNPP, self).__init__()
        
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.model_type = model_type
        self.reg = reg
        
        if model_type in ['joint', 'event']:
            self.event_embedding = nn.Embedding(num_classes, num_classes)  # One-hot like embedding
            event_input_size = num_classes + 1  # +1 for time
            self.event_gru = nn.GRU(event_input_size, state_size_event, batch_first=True)
            self.event_projection = nn.Linear(state_size_event, state_size_timeseries)
        if model_type in ['joint', 'timeseries']:
            self.timeseries_gru = nn.GRU(TIMESERIES_FEATURE, state_size_timeseries, batch_first=True)
        if model_type == 'joint':
            combined_size = state_size_timeseries * 2  # event_output 和 timeseries_output 都是 state_size_timeseries 维度
        elif model_type == 'event':
            combined_size = state_size_event
        else:  # timeseries
            combined_size = state_size_timeseries
        self.time_layer = nn.Linear(combined_size, 1)
        self.mark_layer = nn.Linear(combined_size, num_classes)
        self.w_t = nn.Parameter(torch.ones(1))
        self.epsilon = 1e-3
        
    def forward(self, event_sequence, time_series, seq_lengths):
        batch_size = event_sequence.size(0)
        if self.model_type in ['joint', 'event']:
            # Split marks and times
            marks = event_sequence[..., 0].long()
            times = event_sequence[..., 1].unsqueeze(-1)
            
            # Embed marks
            mark_embedded = self.event_embedding(marks)
            
            # Concatenate with times
            event_input = torch.cat([mark_embedded, times], dim=-1)
            
            # Pack sequence for GRU
            packed_event = nn.utils.rnn.pack_padded_sequence(
                event_input, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            event_output, _ = self.event_gru(packed_event)
            
            # Unpack sequence
            event_output, _ = nn.utils.rnn.pad_packed_sequence(event_output, batch_first=True)
            print("Event output shape:", event_output.shape)

        if self.model_type in ['joint', 'timeseries']:
            batch_size = time_series.size(0)
            seq_len = event_output.size(1)
            print("Original time_series shape:", time_series.shape)
            timeseries_output, _ = self.timeseries_gru(time_series)
            timeseries_output = timeseries_output[:, -1:, :].expand(-1, seq_len, -1)
            print("Timeseries output shape:", timeseries_output.shape)
        if self.model_type == 'joint':
            event_output = self.event_projection(event_output)
            print("Projected event output shape:", event_output.shape)
            combined_output = torch.cat([event_output, timeseries_output], dim=-1)
        elif self.model_type == 'event':
            combined_output = event_output
        else:
            combined_output = timeseries_output
        time_pred = self.time_layer(combined_output)
        mark_logits = self.mark_layer(combined_output)
        return time_pred, mark_logits
    
    def compute_loss(self, time_pred, mark_logits, targets, seq_lengths):
        max_len = time_pred.size(1)
        mask = torch.arange(max_len, device=seq_lengths.device).expand(len(seq_lengths), max_len)
        mask = mask < seq_lengths.unsqueeze(1)
        time_pred = time_pred.squeeze(-1)
        time_targets = targets[..., 1]
        print(time_targets)

        if self.loss_type == 'mse':
            time_loss = torch.abs(time_pred - time_targets)
            time_loss = (time_loss * mask).sum() / mask.sum()
        else:
            w_t = torch.where(torch.abs(self.w_t) < self.epsilon,
                            torch.sign(self.w_t) * self.epsilon,
                            self.w_t)
            part1 = time_pred
            part2 = w_t * time_targets.unsqueeze(-1)
            time_loglike = part1 + part2 + (torch.exp(part1) - torch.exp(part1 + part2)) / w_t
            time_loss = -(time_loglike * mask.unsqueeze(-1)).sum() / mask.sum()

        mark_loss = nn.functional.cross_entropy(
            mark_logits.view(-1, self.num_classes),
            targets[..., 0].long().view(-1),
            reduction='none'
        )
        mark_loss = (mark_loss.view_as(mask) * mask).sum() / mask.sum()

        total_loss = mark_loss + self.reg * time_loss
        return total_loss, mark_loss, time_loss

def main():
    DIM_SIZE = 7
    mi = MarkedIntensityHomogenuosPoisson(DIM_SIZE)
    for u in range(DIM_SIZE):
        mi.initialize(1.0, u)
    simulated_sequences = generate_samples_marked(mi, 15.0, 1000)
    dataset = PointProcessDataset(simulated_sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = RNNPP(num_classes=DIM_SIZE).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        total_loss_avg = 0
        mark_loss_avg = 0
        time_loss_avg = 0
        num_batches = 0
        for batch in dataloader:
            event_sequence = batch['sequence'].to(DEVICE)
            time_series = batch['time_series'].to(DEVICE)
            seq_lengths = batch['length'].squeeze(-1).to(DEVICE)
            
            time_pred, mark_logits = model(event_sequence, time_series, seq_lengths)
            
            total_loss, mark_loss, time_loss = model.compute_loss(
                time_pred, mark_logits, event_sequence, seq_lengths
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss_avg += total_loss.item()
            mark_loss_avg += mark_loss.item()
            time_loss_avg += time_loss.item()
            num_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}:')
            print(f'Total Loss: {total_loss_avg/num_batches:.4f}')
            print(f'Mark Loss: {mark_loss_avg/num_batches:.4f}')
            print(f'Time Loss: {time_loss_avg/num_batches:.4f}\n')

if __name__ == '__main__':
    main()
