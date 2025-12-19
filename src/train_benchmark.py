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
    """
    LSTM-based density matrix predictor with optional Verlet integration.

    Integration modes:
    - First-order (Euler): ρ(t+Δt) = ρ(t) + Δρ
      Model predicts velocity-like term Δρ

    - Second-order (Verlet): ρ(t+Δt) = 2ρ(t) - ρ(t-Δt) + Δ²ρ
      Model predicts acceleration-like term Δ²ρ
      Better for oscillatory dynamics (quantum coherences)

    Stabilized Verlet includes:
    - Damping term to prevent runaway oscillations
    - Adaptive Euler blending for early training stability
    - Trace projection to enforce electron count conservation
    """
    def __init__(self):
        super().__init__()
        self.n_basis = CFG['n_basis']
        self.use_verlet = CFG.get('use_verlet', False)

        # Verlet stabilization parameters
        self.verlet_damping = CFG.get('verlet_damping', 0.1)  # Friction coefficient
        self.verlet_blend = CFG.get('verlet_blend', 0.5)  # Blend with Euler (0=pure Verlet, 1=pure Euler)
        self.trace_projection = CFG.get('trace_projection', True)  # Project to conserve trace
        self.n_alpha = CFG.get('n_alpha', 1.0)
        self.n_beta = CFG.get('n_beta', 0.0)

        # Load overlap matrix for trace projection (Tr(ρS) = N_e in non-orthonormal AO basis)
        try:
            import numpy as np
            S = np.load(CFG['overlap_file'])
            self.register_buffer('S', torch.tensor(S, dtype=torch.complex128))
        except:
            # Fallback to identity (orthonormal basis)
            self.register_buffer('S', torch.eye(self.n_basis, dtype=torch.complex128))

        input_dim = (2 * 2 * self.n_basis**2) + 3

        self.lstm = nn.LSTM(input_dim, CFG['hidden_dim'], batch_first=True)
        self.attention = SelfAttention(CFG['hidden_dim'])
        self.head = nn.Linear(CFG['hidden_dim'], 2 * 2 * self.n_basis**2)

    def forward(self, rho_seq, field, prev_rho=None):
        """
        Forward pass with optional Verlet integration.

        Args:
            rho_seq: (Batch, Seq, 2, N, N) - density matrix history
            field: (Batch, 3) - external field at next timestep
            prev_rho: (Batch, 2, N, N) - ρ(t-1) for Verlet in rollout
                      Only needed when use_verlet=True and doing autoregressive rollout

        Returns:
            rho_pred: (Batch, 2, N, N) - predicted density matrix
        """
        # rho_seq: (Batch, Seq, 2, N, N)
        B, S, _, _, _ = rho_seq.shape
        rho_flat = rho_seq.view(B, S, -1)
        rho_input = torch.cat([rho_flat.real, rho_flat.imag], dim=-1)

        # Expand field to match sequence length
        field_expanded = field.unsqueeze(1).expand(-1, S, -1)

        full_input = torch.cat([rho_input, field_expanded], dim=-1).float()

        lstm_out, _ = self.lstm(full_input)
        attn_out = self.attention(lstm_out)

        delta_raw = self.head(attn_out[:, -1, :])
        split = delta_raw.shape[-1] // 2
        delta_rho = torch.complex(delta_raw[:, :split], delta_raw[:, split:]) \
                        .to(torch.complex128) \
                        .view(B, 2, self.n_basis, self.n_basis)

        last_rho = rho_seq[:, -1]  # ρ(t)

        if self.use_verlet:
            # Get previous state
            if prev_rho is not None:
                prev = prev_rho
            else:
                prev = rho_seq[:, -2]  # ρ(t-Δt)

            # Velocity estimate from finite difference
            velocity = last_rho - prev  # v ≈ ρ(t) - ρ(t-Δt)

            # === Stabilized Verlet Integration ===
            # Standard Verlet: ρ(t+Δt) = 2ρ(t) - ρ(t-Δt) + Δ²ρ
            # With damping:    ρ(t+Δt) = 2ρ(t) - ρ(t-Δt) + Δ²ρ - γ*v
            # The damping term -γ*v acts like friction to prevent runaway

            rho_verlet = 2 * last_rho - prev + delta_rho - self.verlet_damping * velocity

            # Euler prediction for blending
            rho_euler = last_rho + delta_rho

            # Blend Verlet with Euler for stability
            # blend=0 -> pure Verlet, blend=1 -> pure Euler
            rho_pred = (1 - self.verlet_blend) * rho_verlet + self.verlet_blend * rho_euler
        else:
            # First-order (Euler): ρ(t+Δt) = ρ(t) + Δρ
            rho_pred = last_rho + delta_rho

        # Enforce Hermiticity
        rho_pred = 0.5 * (rho_pred + rho_pred.transpose(-1, -2).conj())

        # Trace projection: scale to conserve electron count (works for both Euler and Verlet)
        if self.trace_projection:
            rho_pred = self._project_trace(rho_pred)

        return rho_pred

    def _project_trace(self, rho):
        """
        Project density matrix to conserve electron count.

        In non-orthonormal AO basis: Tr(ρS) = N_electrons
        Scale ρ so that Tr(ρS) equals target electron count.

        Args:
            rho: (B, 2, N, N) density matrix (2 spin channels)

        Returns:
            Scaled density matrix with correct Tr(ρS)
        """
        # rho: (B, 2, N, N) - 2 spin channels
        # S: (N, N) overlap matrix
        S = self.S.to(rho.device)

        # Compute Tr(ρS) = Tr(ρ @ S) for each batch and spin channel
        # For complex matrices: trace of product
        rho_alpha = rho[:, 0]  # (B, N, N)
        rho_beta = rho[:, 1]   # (B, N, N)

        # Tr(ρS) = sum_ij ρ_ij * S_ji = sum_ij ρ_ij * S_ij^T
        # More efficiently: Tr(A @ B) = sum(A * B^T) element-wise
        trace_alpha = torch.sum(rho_alpha * S.T, dim=(-2, -1)).real  # (B,)
        trace_beta = torch.sum(rho_beta * S.T, dim=(-2, -1)).real    # (B,)

        # Compute scale factors (avoid division by zero)
        scale_alpha = self.n_alpha / (trace_alpha.abs() + 1e-10)  # (B,)
        scale_beta = self.n_beta / (trace_beta.abs() + 1e-10)     # (B,)

        # Handle zero target beta (no scaling needed for empty channel)
        if self.n_beta < 1e-10:
            scale_beta = torch.ones_like(scale_beta)

        # Apply scaling (broadcast over spatial dimensions)
        rho_alpha_scaled = rho_alpha * scale_alpha.view(-1, 1, 1)  # (B, N, N)
        rho_beta_scaled = rho_beta * scale_beta.view(-1, 1, 1)     # (B, N, N)

        # Stack back together
        return torch.stack([rho_alpha_scaled, rho_beta_scaled], dim=1)

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
            # For Verlet, track previous state explicitly
            use_verlet = CFG.get('use_verlet', False)
            prev_rho = None  # Will be set after first prediction

            for k in range(rollout):
                next_field = f_seq[:, k, :]  # (Batch, 3)

                # Predict (pass prev_rho for Verlet rollout)
                pred_rho = model(curr_input, next_field, prev_rho=prev_rho)

                # Loss
                total_loss += crit(pred_rho, y_seq[:, k])

                # Autoregressive Feed
                if k < rollout - 1:
                    if use_verlet:
                        # For Verlet: track ρ(t-1) explicitly
                        # After k=0: prev_rho = curr_input[:, -1], curr = pred_rho
                        # After k=1: prev_rho = old_curr, curr = pred_rho
                        prev_rho = curr_input[:, -1].clone()

                    new_step = pred_rho.unsqueeze(1)  # (Batch, 1, 2, N, N)
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
    parser.add_argument('--verlet', action='store_true',
                        help='Use Verlet (second-order) integration instead of Euler')
    args = parser.parse_args()

    CFG = load_config(args.config, args)

    # CLI override for Verlet
    if args.verlet:
        CFG['use_verlet'] = True
        print("Using Verlet (second-order) integration")

    if not CFG.get('predict_only', False):
        train()
