import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import argparse
import os
import json
import matplotlib.pyplot as plt

# --- Global Config Placeholder ---
CFG = {}

# --- 1. Data & Utils ---
class MolecularDynamicsDataset(Dataset):
    def __init__(self, density_file, field_file, seq_len, rollout_steps=1):
        print(f"Loading density data from {density_file}...")
        data_dict = np.load(density_file, allow_pickle=True).item()
        raw_rho = torch.tensor(data_dict['density'], dtype=CFG['dtype'])
        self.rho = raw_rho
        
        try:
            field_data = np.loadtxt(field_file)
            if field_data.shape[1] == 4: field_data = field_data[:, 1:] 
            self.field = torch.tensor(field_data, dtype=torch.float64)
        except Exception:
            print("Warning: Could not load field.dat, using zeros.")
            self.field = torch.zeros((len(self.rho), 3), dtype=torch.float64)

        min_len = min(len(self.rho), len(self.field))
        self.rho = self.rho[:min_len]
        self.field = self.field[:min_len]
        
        self.seq_len = seq_len
        self.rollout_steps = rollout_steps

    def __len__(self):
        return len(self.rho) - self.seq_len - self.rollout_steps

    def __getitem__(self, idx):
        # 1. Input History
        rho_seq = self.rho[idx : idx + self.seq_len]
        # 2. Future Targets (Sequence)
        rho_targets = self.rho[idx + self.seq_len : idx + self.seq_len + self.rollout_steps]
        # 3. Future Fields
        field_targets = self.field[idx + self.seq_len : idx + self.seq_len + self.rollout_steps]
        return rho_seq, field_targets, rho_targets

# [Insert apply_mcweeny and enforce_physical_constraints functions here - omitted for brevity]
# ...

# --- 2. Model ---
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class DensityMatrixLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_basis = CFG['n_basis']
        input_dim = (2 * 2 * self.n_basis**2) + 3
        
        self.lstm = nn.LSTM(input_dim, CFG['hidden_dim'], batch_first=True)
        self.attention = SelfAttention(CFG['hidden_dim'])
        self.head = nn.Linear(CFG['hidden_dim'], 2 * 2 * self.n_basis**2)

    def forward(self, rho_seq, field):
        # rho_seq: (Batch, Seq, 2, N, N)
        B, S, _, _, _ = rho_seq.shape
        rho_flat = rho_seq.view(B, S, -1)
        rho_input = torch.cat([rho_flat.real, rho_flat.imag], dim=-1)
        
        # Expand field to match sequence length
        if field.dim() == 2: # (Batch, 3) -> Single step prediction
             field_expanded = field.unsqueeze(1).expand(-1, S, -1)
        else: # (Batch, Seq, 3) -> If passing full sequence field (not used in standard inference)
             field_expanded = field
             
        # For prediction, we usually only care about the *next* field acting on the *last* state.
        # However, standard LSTM takes (Input_t) -> Output_t+1.
        # We align the field such that we append the *next* field to the *current* input.
        # Note: Ideally, field input should be time-aligned. 
        # Here we simplify: assume field matches temporal index.
        field_expanded = field.unsqueeze(1).expand(-1, S, -1)

        full_input = torch.cat([rho_input, field_expanded], dim=-1).float()
        
        lstm_out, _ = self.lstm(full_input)
        attn_out = self.attention(lstm_out)
        
        delta_raw = self.head(attn_out[:, -1, :])
        split = delta_raw.shape[-1] // 2
        delta_rho = torch.complex(delta_raw[:, :split], delta_raw[:, split:]) \
                        .to(torch.complex128) \
                        .view(B, 2, self.n_basis, self.n_basis)
        
        last_rho = rho_seq[:, -1]
        rho_pred = last_rho + delta_rho
        
        rho_pred = 0.5 * (rho_pred + rho_pred.transpose(-1, -2).conj())
        return rho_pred

class PhysicsLoss(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S.to(CFG['device'])
    
    def forward(self, rho_pred, rho_target):
        # MSE
        loss = F.mse_loss(rho_pred.real, rho_target.real) + \
               F.mse_loss(rho_pred.imag, rho_target.imag)
        return loss

# --- 3. Training Loop (With Rollout) ---
def train():
    try:
        S = torch.tensor(np.load(CFG['overlap_file']), dtype=CFG['dtype'])
    except:
        S = torch.eye(CFG['n_basis'], dtype=CFG['dtype'])

    # Initialize Dataset with Rollout Steps
    rollout = CFG.get('rollout_steps', 5)
    data = MolecularDynamicsDataset(CFG['density_file'], CFG['field_file'], CFG['seq_len'], rollout_steps=rollout)
    loader = DataLoader(data, batch_size=CFG['batch_size'], shuffle=True)
    
    model = DensityMatrixLSTM().to(CFG['device'])
    opt = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    crit = PhysicsLoss(S)
    
    print(f"Training with Rollout={rollout} for {CFG['epochs']} epochs...")
    
    for epoch in range(CFG['epochs']):
        model.train()
        ep_loss = 0
        
        for x_seq, f_seq, y_seq in loader:
            x_seq, f_seq, y_seq = x_seq.to(CFG['device']), f_seq.to(CFG['device']), y_seq.to(CFG['device'])
            
            # Noise Injection (Drift Simulation)
            # if model.training:
            #     x_seq = x_seq + (torch.randn_like(x_seq) * 1e-8)

            opt.zero_grad()
            total_loss = 0
            curr_input = x_seq
            
            # Rollout Loop
            for k in range(rollout):
                next_field = f_seq[:, k, :] # (Batch, 3)
                
                # Predict
                pred_rho = model(curr_input, next_field)
                
                # Loss
                total_loss += crit(pred_rho, y_seq[:, k])
                
                # Autoregressive Feed
                if k < rollout - 1:
                    new_step = pred_rho.unsqueeze(1) # (Batch, 1, 2, N, N)
                    curr_input = torch.cat([curr_input[:, 1:], new_step], dim=1)

            total_loss = total_loss / rollout
            total_loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            ep_loss += total_loss.item()
        
        if (epoch+1)%10 == 0: print(f"Epoch {epoch+1}: {ep_loss/len(loader):.7f}")
        
    torch.save(model.state_dict(), CFG['model_save_path'])
    return model, data

def load_config(json_path, args):
    with open(json_path, 'r') as f:
        config = json.load(f)
    flat = {}
    for k in config: flat.update(config[k])
    
    # CLI Overrides
    if args.predict: flat['predict_only'] = True
    
    flat['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flat['dtype'] = torch.complex128
    return flat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='inputs/train.json')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    
    CFG = load_config(args.config, args)
    
    if not CFG.get('predict_only', False):
        train()
