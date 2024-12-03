import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PointProcessDataset(Dataset):
    def __init__(self, data, time_step=20):
        self.data = data
        self.time_step = time_step
        
    def __len__(self):
        return len(self.data) - self.time_step
        
    def __getitem__(self, idx):
        history = self.data[idx:idx+self.time_step].reshape(-1, 1)
        next_time = self.data[idx+self.time_step]
        elapsed = next_time - history[-1, 0]
        return {
            'history': torch.FloatTensor(history),
            'elapsed': torch.FloatTensor([elapsed])
        }

class GRUPointProcess(nn.Module):
    def __init__(self, time_step=20, size_gru=64, size_nn=64, size_layer_chfn=2):
        super(GRUPointProcess, self).__init__()
        self.time_step = time_step
        self.size_gru = size_gru
        self.size_nn = size_nn
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=size_gru,
            num_layers=5,
            batch_first=True
        )
        self.elapsed_time_linear = nn.Linear(1, size_nn, bias=False)
        self.gru_linear = nn.Linear(size_gru, size_nn)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(size_nn, size_nn)
            for _ in range(size_layer_chfn-1)
        ])
        self.output_layer = nn.Linear(size_nn, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, 0, 0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, event_history, elapsed_time):
        event_history = torch.clamp(event_history, min=1e-6)
        elapsed_time = torch.clamp(elapsed_time, min=1e-6)
        elapsed_time = elapsed_time.requires_grad_(True)
        
        gru_out, _ = self.gru(event_history)
        gru_out = gru_out[:, -1, :]
        
        hidden_tau = self.elapsed_time_linear(elapsed_time)
        hidden_gru = self.gru_linear(gru_out)
        hidden = torch.tanh(hidden_tau + hidden_gru)
        
        for layer in self.hidden_layers:
            hidden = torch.tanh(layer(hidden))
        
        Int_l = F.softplus(self.output_layer(hidden)) + 1e-6
        
        try:
            # l = torch.autograd.grad(Int_l.sum(), elapsed_time, create_graph=True)[0]
            l = torch.autograd.grad(Int_l.sum(), elapsed_time, create_graph=True)[0]
            l = torch.clamp(l, min=1e-6, max=1e6)
        except RuntimeError:
            print("Gradient computation failed")
            l = torch.ones_like(elapsed_time) * 1e-6
        
        return l, Int_l
    
    def compute_loss(self, l, Int_l):
        l = torch.clamp(l, min=1e-6)
        Int_l = torch.clamp(Int_l, min=0.0)
        
        loss = -torch.mean(torch.log(l) - Int_l)
        
        if torch.isnan(loss):
            print("Warning: Loss is NaN!")
            loss = torch.tensor(0.0, requires_grad=True, device=l.device)
            
        return loss

def train_model(model, train_loader, test_loader=None, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_bar:
            history = batch['history'].to(device)
            elapsed = batch['elapsed'].to(device)
            
            optimizer.zero_grad()
            l, Int_l = model(history, elapsed)
            loss = model.compute_loss(l, Int_l)
            
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / num_batches
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        
        scheduler.step(avg_train_loss)
        
        # if (epoch + 1) % 5 == 0:
        #     torch.save(model.state_dict(), f'gru_point_process_epoch_{epoch+1}.pt')
