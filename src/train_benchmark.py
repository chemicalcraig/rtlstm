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

from timestep_sync import parse_field_timestamps, store_model_dt

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

def compute_variance_bins(dataset, n_bins=2):
    """
    Partition density matrix elements into variance-based bins.

    Args:
        dataset: MolecularDynamicsDataset with .rho attribute
        n_bins: Number of bins (sub-models)

    Returns:
        masks: List of (2, N, N) boolean tensors, one per bin
        thresholds: Variance thresholds used for binning
    """
    rho = dataset.rho  # (T, 2, N, N)
    n_spins, n_basis = rho.shape[1], rho.shape[2]

    # Compute variance for each element
    var_real = rho.real.var(dim=0)
    var_imag = rho.imag.var(dim=0)
    var_combined = var_real + var_imag + 1e-12

    # Flatten to get all element variances
    var_flat = var_combined.flatten()

    # Use only non-trivial elements (ignore empty spin channels)
    meaningful_mask = var_flat > 1e-10
    meaningful_vars = var_flat[meaningful_mask]

    if len(meaningful_vars) == 0:
        # All elements trivial - single bin
        masks = [torch.ones(n_spins, n_basis, n_basis, dtype=torch.bool)]
        return masks, []

    # Compute quantile thresholds
    thresholds = []
    for i in range(1, n_bins):
        q = i / n_bins
        thresh = torch.quantile(meaningful_vars.float(), q).item()
        thresholds.append(thresh)

    # Create masks for each bin
    masks = []
    for bin_idx in range(n_bins):
        mask = torch.zeros(n_spins, n_basis, n_basis, dtype=torch.bool)

        for s in range(n_spins):
            for i in range(n_basis):
                for j in range(n_basis):
                    v = var_combined[s, i, j].item()

                    # Assign to bin based on variance
                    if v < 1e-10:
                        # Trivial element - assign to highest variance bin (least sensitive)
                        assigned_bin = n_bins - 1
                    elif bin_idx == 0:
                        # First bin: variance <= first threshold (LOW variance)
                        assigned_bin = 0 if v <= thresholds[0] else -1
                    elif bin_idx == n_bins - 1:
                        # Last bin: variance > last threshold (HIGH variance)
                        assigned_bin = n_bins - 1 if v > thresholds[-1] else -1
                    else:
                        # Middle bins
                        assigned_bin = bin_idx if thresholds[bin_idx-1] < v <= thresholds[bin_idx] else -1

                    if assigned_bin == bin_idx:
                        mask[s, i, j] = True

        masks.append(mask)

    # Print bin statistics
    print(f"[Ensemble] Created {n_bins} variance bins:")
    for i, mask in enumerate(masks):
        count = mask.sum().item()
        if i == 0:
            print(f"  Bin {i} (LOW var, ≤{thresholds[0]:.2e}): {count} elements")
        elif i == n_bins - 1:
            print(f"  Bin {i} (HIGH var, >{thresholds[-1]:.2e}): {count} elements")
        else:
            print(f"  Bin {i} (var in ({thresholds[i-1]:.2e}, {thresholds[i]:.2e}]): {count} elements")

    return masks, thresholds


class SubModelLSTM(nn.Module):
    """
    A sub-model for the ensemble - predicts delta for assigned elements only.
    Sees full input context but outputs are masked.
    """
    def __init__(self, n_basis, hidden_dim, mask):
        super().__init__()
        self.n_basis = n_basis
        self.register_buffer('mask', mask)  # (2, N, N)

        input_dim = (2 * 2 * n_basis**2) + 3

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2 * 2 * n_basis**2)

    def forward(self, rho_input_flat, field_expanded):
        """
        Args:
            rho_input_flat: (B, S, 2*2*N*N) flattened real+imag density
            field_expanded: (B, S, 3) field repeated over sequence

        Returns:
            delta_rho: (B, 2, N, N) masked delta prediction
        """
        B = rho_input_flat.shape[0]
        full_input = torch.cat([rho_input_flat, field_expanded], dim=-1).float()

        lstm_out, _ = self.lstm(full_input)
        attn_out = self.attention(lstm_out)

        delta_raw = self.head(attn_out[:, -1, :])
        split = delta_raw.shape[-1] // 2
        delta_rho = torch.complex(delta_raw[:, :split], delta_raw[:, split:]) \
                        .to(torch.complex128) \
                        .view(B, 2, self.n_basis, self.n_basis)

        # Apply mask - only output for assigned elements
        mask = self.mask.to(delta_rho.device)
        delta_rho = delta_rho * mask.unsqueeze(0)

        return delta_rho


class EnsembleDensityMatrixLSTM(nn.Module):
    """
    Ensemble of LSTM sub-models, each specializing in different variance bins.

    Architecture:
    - Partition elements by variance into n_bins groups
    - Each bin gets a dedicated sub-model
    - All sub-models see full input context
    - Outputs are masked and aggregated

    This prevents high-variance elements from dominating gradient updates
    for low-variance elements during training.
    """
    def __init__(self, masks):
        super().__init__()
        self.n_basis = CFG['n_basis']
        self.n_bins = len(masks)
        self.use_verlet = CFG.get('use_verlet', False)

        # Verlet stabilization parameters
        self.verlet_damping = CFG.get('verlet_damping', 0.1)
        self.verlet_blend = CFG.get('verlet_blend', 0.5)
        self.trace_projection = CFG.get('trace_projection', True)
        self.n_alpha = CFG.get('n_alpha', 1.0)
        self.n_beta = CFG.get('n_beta', 0.0)

        # Load overlap matrix
        try:
            S = np.load(CFG['overlap_file'])
            self.register_buffer('S', torch.tensor(S, dtype=torch.complex128))
        except:
            self.register_buffer('S', torch.eye(self.n_basis, dtype=torch.complex128))

        # Store masks as buffers
        for i, mask in enumerate(masks):
            self.register_buffer(f'mask_{i}', mask)

        # Create sub-models
        self.sub_models = nn.ModuleList([
            SubModelLSTM(self.n_basis, CFG['hidden_dim'], masks[i])
            for i in range(self.n_bins)
        ])

        print(f"[Ensemble] Created {self.n_bins} sub-models")

    def get_masks(self):
        """Return list of masks for saving in checkpoint."""
        return [getattr(self, f'mask_{i}') for i in range(self.n_bins)]

    def forward(self, rho_seq, field, prev_rho=None):
        """
        Forward pass aggregating all sub-model predictions.

        Args:
            rho_seq: (B, S, 2, N, N) density matrix history
            field: (B, 3) external field at next timestep
            prev_rho: (B, 2, N, N) for Verlet rollout

        Returns:
            rho_pred: (B, 2, N, N) predicted density matrix
        """
        B, S, _, _, _ = rho_seq.shape

        # Prepare shared input
        rho_flat = rho_seq.view(B, S, -1)
        rho_input = torch.cat([rho_flat.real, rho_flat.imag], dim=-1)
        field_expanded = field.unsqueeze(1).expand(-1, S, -1)

        # Aggregate delta predictions from all sub-models
        delta_rho = torch.zeros(B, 2, self.n_basis, self.n_basis,
                                dtype=torch.complex128, device=rho_seq.device)

        for sub_model in self.sub_models:
            delta_rho = delta_rho + sub_model(rho_input, field_expanded)

        # Apply integration scheme
        last_rho = rho_seq[:, -1]

        if self.use_verlet:
            if prev_rho is not None:
                prev = prev_rho
            else:
                prev = rho_seq[:, -2]

            velocity = last_rho - prev
            rho_verlet = 2 * last_rho - prev + delta_rho - self.verlet_damping * velocity
            rho_euler = last_rho + delta_rho
            rho_pred = (1 - self.verlet_blend) * rho_verlet + self.verlet_blend * rho_euler
        else:
            rho_pred = last_rho + delta_rho

        # Enforce Hermiticity
        rho_pred = 0.5 * (rho_pred + rho_pred.transpose(-1, -2).conj())

        # Trace projection
        if self.trace_projection:
            rho_pred = self._project_trace(rho_pred)

        return rho_pred

    def _project_trace(self, rho):
        """Project density matrix to conserve electron count."""
        S = self.S.to(rho.device)

        rho_alpha = rho[:, 0]
        rho_beta = rho[:, 1]

        trace_alpha = torch.sum(rho_alpha * S.T, dim=(-2, -1)).real
        trace_beta = torch.sum(rho_beta * S.T, dim=(-2, -1)).real

        scale_alpha = self.n_alpha / (trace_alpha.abs() + 1e-10)
        scale_beta = self.n_beta / (trace_beta.abs() + 1e-10)

        if self.n_beta < 1e-10:
            scale_beta = torch.ones_like(scale_beta)

        rho_alpha_scaled = rho_alpha * scale_alpha.view(-1, 1, 1)
        rho_beta_scaled = rho_beta * scale_beta.view(-1, 1, 1)

        return torch.stack([rho_alpha_scaled, rho_beta_scaled], dim=1)


class PhysicsLoss(nn.Module):
    def __init__(self, S, element_weights=None):
        """
        Physics-informed loss with optional element-wise weighting.

        Args:
            S: Overlap matrix
            element_weights: (2, N, N) tensor of weights per element.
                           If None, uniform weighting is used.
                           Use compute_variance_weights() to create these.
        """
        super().__init__()
        self.S = S.to(CFG['device'])
        if element_weights is not None:
            self.register_buffer('element_weights', element_weights.to(CFG['device']))
        else:
            self.register_buffer('element_weights', None)

    def forward(self, rho_pred, rho_target):
        if self.element_weights is not None:
            # Weighted MSE: weight by inverse variance per element
            # This gives equal importance to all elements regardless of their amplitude
            diff_real = (rho_pred.real - rho_target.real) ** 2
            diff_imag = (rho_pred.imag - rho_target.imag) ** 2

            # Expand weights for batch dimension: (2, N, N) -> (1, 2, N, N)
            w = self.element_weights.unsqueeze(0)

            # Weighted mean
            loss = (w * diff_real).mean() + (w * diff_imag).mean()
        else:
            # Standard MSE
            loss = F.mse_loss(rho_pred.real, rho_target.real) + \
                   F.mse_loss(rho_pred.imag, rho_target.imag)
        return loss


def compute_variance_weights(dataset, min_weight=0.1, max_weight=100.0):
    """
    Compute element-wise weights based on inverse variance.

    Elements with small variance (small dynamics) get higher weights
    so the model pays more attention to them.

    Args:
        dataset: MolecularDynamicsDataset with .rho attribute
        min_weight: Minimum weight (prevents zero weights)
        max_weight: Maximum weight (prevents exploding weights for near-constant elements)

    Returns:
        weights: (2, N, N) tensor of weights
    """
    # Get all density matrices: (T, 2, N, N)
    rho = dataset.rho

    # Compute variance across time dimension for each element
    # var: (2, N, N)
    var_real = rho.real.var(dim=0)
    var_imag = rho.imag.var(dim=0)

    # Combined variance (real + imag)
    var_combined = var_real + var_imag + 1e-12  # Avoid division by zero

    # Debug: print raw variances
    print(f"[VarianceWeights] rho shape: {rho.shape}")
    print(f"[VarianceWeights] var_real:\n{var_real}")
    print(f"[VarianceWeights] var_imag:\n{var_imag}")
    print(f"[VarianceWeights] var_combined:\n{var_combined}")

    # Inverse variance weighting
    weights = 1.0 / var_combined

    # Normalize PER SPIN CHANNEL so mean weight = 1 within each channel
    # This prevents empty/zero spin channels from dominating the normalization
    for spin in range(weights.shape[0]):
        spin_weights = weights[spin]
        # Only normalize if this channel has meaningful variance
        if var_combined[spin].max() > 1e-10:
            weights[spin] = spin_weights / spin_weights.mean()
        else:
            # Empty spin channel (e.g., beta in H2+) - set uniform weight
            weights[spin] = 1.0

    # Clamp to reasonable range
    weights = weights.clamp(min=min_weight, max=max_weight)

    print(f"[VarianceWeights] Weight range: {weights.min():.3f} - {weights.max():.3f}")
    print(f"[VarianceWeights] Weight stats: mean={weights.mean():.3f}, std={weights.std():.3f}")
    print("weights:", weights)
    return weights

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

    # Choose model type: Ensemble or Single
    use_ensemble = CFG.get('use_ensemble', False)
    n_bins = CFG.get('n_bins', 2)
    masks = None

    if use_ensemble:
        masks, thresholds = compute_variance_bins(data, n_bins=n_bins)
        model = EnsembleDensityMatrixLSTM(masks).to(CFG['device'])
    else:
        model = DensityMatrixLSTM().to(CFG['device'])

    opt = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])

    # Compute variance-based element weights for loss (optional, only for non-ensemble)
    use_variance_weights = CFG.get('use_variance_weights', False) and not use_ensemble
    if use_variance_weights:
        element_weights = compute_variance_weights(
            data,
            min_weight=CFG.get('var_weight_min', 0.1),
            max_weight=CFG.get('var_weight_max', 100.0)
        )
        crit = PhysicsLoss(S, element_weights=element_weights)
    else:
        crit = PhysicsLoss(S)

    # Build info string
    mode_str = f"Ensemble ({n_bins} bins)" if use_ensemble else "Single model"
    weight_str = " with variance-weighted loss" if use_variance_weights else ""
    print(f"Training {mode_str} with Rollout={rollout} for {CFG['epochs']} epochs{weight_str}...")
    
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

    # Compute training dt from field file
    try:
        _, _, training_dt = parse_field_timestamps(CFG['field_file'])
    except:
        training_dt = CFG.get('dt', 0.4)  # Fallback to config or default
        print(f"Warning: Could not parse field file for dt, using {training_dt}")

    # Save model with dt metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'n_basis': CFG['n_basis'],
            'hidden_dim': CFG['hidden_dim'],
            'seq_len': CFG['seq_len'],
            'use_verlet': CFG.get('use_verlet', False),
            'verlet_damping': CFG.get('verlet_damping', 0.1),
            'verlet_blend': CFG.get('verlet_blend', 0.5),
            'trace_projection': CFG.get('trace_projection', True),
            'n_alpha': CFG.get('n_alpha', 1.0),
            'n_beta': CFG.get('n_beta', 0.0),
            'use_variance_weights': use_variance_weights,
            'use_ensemble': use_ensemble,
            'n_bins': n_bins if use_ensemble else None,
        }
    }

    # Save masks for ensemble model (needed for inference)
    if use_ensemble and masks is not None:
        checkpoint['masks'] = [m.cpu() for m in masks]

    checkpoint = store_model_dt(checkpoint, training_dt)
    torch.save(checkpoint, CFG['model_save_path'])
    print(f"Model saved to {CFG['model_save_path']} (training_dt={training_dt:.6f})")

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
    parser.add_argument('--variance-weights', action='store_true',
                        help='Use inverse-variance element weighting for loss')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble model with variance-based binning')
    parser.add_argument('--n-bins', type=int, default=2,
                        help='Number of variance bins for ensemble (default: 2)')
    args = parser.parse_args()

    CFG = load_config(args.config, args)

    # CLI override for Verlet
    if args.verlet:
        CFG['use_verlet'] = True
        print("Using Verlet (second-order) integration")

    # CLI override for variance weighting
    if args.variance_weights:
        CFG['use_variance_weights'] = True
        print("Using inverse-variance element weighting")

    # CLI override for ensemble
    if args.ensemble:
        CFG['use_ensemble'] = True
        CFG['n_bins'] = args.n_bins
        print(f"Using ensemble model with {args.n_bins} variance bins")

    if not CFG.get('predict_only', False):
        train()
