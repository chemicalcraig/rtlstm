#!/usr/bin/env python3
"""
Quick comparison test: Euler vs Verlet integration.

Runs both methods on the same data for a short training and compares:
- Training loss convergence
- Final prediction error
- Stability during rollout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_benchmark import MolecularDynamicsDataset, DensityMatrixLSTM, PhysicsLoss

# Configuration
SKIP_FACTOR = 1  # Back to k=1 for fair rollout comparison
ROLLOUT_TEST = 50  # Long rollout to test stability

CFG = {
    'density_file': 'test_train/densities/density_series.npy',
    'overlap_file': 'test_train/data/h2_plus_rttddft_overlap.npy',
    'field_file': 'test_train/data/field.dat',
    'n_basis': 4,
    'seq_len': 20,
    'hidden_dim': 64,  # Smaller for quick test
    'batch_size': 16,
    'learning_rate': 1e-3,
    'learning_rate_verlet': 5e-3,  # Higher LR for Verlet
    'epochs': 50,  # Quicker test
    'rollout_steps': 5,  # Training rollout
    'rollout_test': ROLLOUT_TEST,  # Testing rollout (longer)
    'skip_factor': SKIP_FACTOR,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.complex128,
    # Verlet stabilization
    'verlet_damping': 0.02,     # Light damping
    'verlet_blend': 0.0,        # Pure Verlet
    'trace_projection': False,  # OFF during training
    'trace_projection_inference': True,  # ON during inference
    'n_alpha': 1.0,
    'n_beta': 0.0
}


class SkipStepDataset(torch.utils.data.Dataset):
    """Dataset that returns targets at k-step intervals."""

    def __init__(self, density_file, field_file, seq_len, rollout_steps, skip_factor, dtype):
        print(f"Loading density data from {density_file}...")
        data_dict = np.load(density_file, allow_pickle=True).item()
        self.rho = torch.tensor(data_dict['density'], dtype=dtype)

        try:
            field_data = np.loadtxt(field_file)
            if field_data.shape[1] == 4:
                field_data = field_data[:, 1:]
            self.field = torch.tensor(field_data, dtype=torch.float64)
        except:
            print("Warning: Could not load field.dat, using zeros.")
            self.field = torch.zeros((len(self.rho), 3), dtype=torch.float64)

        min_len = min(len(self.rho), len(self.field))
        self.rho = self.rho[:min_len]
        self.field = self.field[:min_len]

        self.seq_len = seq_len
        self.rollout_steps = rollout_steps
        self.skip_factor = skip_factor

    def __len__(self):
        # Account for skip factor in available samples
        return len(self.rho) - self.seq_len - self.rollout_steps * self.skip_factor

    def __getitem__(self, idx):
        # Input sequence (consecutive frames)
        rho_seq = self.rho[idx : idx + self.seq_len]

        # Targets at k-step intervals
        target_indices = [idx + self.seq_len + (i+1) * self.skip_factor - 1
                         for i in range(self.rollout_steps)]
        rho_targets = torch.stack([self.rho[i] for i in target_indices])

        # Fields at target times
        field_targets = torch.stack([self.field[i] for i in target_indices])

        return rho_seq, field_targets, rho_targets


def train_model(use_verlet: bool, cfg: dict):
    """Train model with specified integration method."""
    cfg = cfg.copy()
    cfg['use_verlet'] = use_verlet

    # Inject into global CFG for the model
    import train_benchmark
    train_benchmark.CFG = cfg

    # Load overlap
    try:
        S = torch.tensor(np.load(cfg['overlap_file']), dtype=cfg['dtype'])
    except:
        S = torch.eye(cfg['n_basis'], dtype=cfg['dtype'])

    # Dataset with skip-step
    rollout = cfg['rollout_steps']
    skip_factor = cfg.get('skip_factor', 1)
    data = SkipStepDataset(
        cfg['density_file'], cfg['field_file'],
        cfg['seq_len'], rollout_steps=rollout,
        skip_factor=skip_factor, dtype=cfg['dtype']
    )
    loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True)

    # Model
    model = DensityMatrixLSTM().to(cfg['device'])

    # Use higher learning rate for Verlet
    lr = cfg.get('learning_rate_verlet', cfg['learning_rate']) if use_verlet else cfg['learning_rate']
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"  Learning rate: {lr}")
    crit = PhysicsLoss(S)

    method = "Verlet" if use_verlet else "Euler"
    print(f"\n{'='*50}")
    print(f"Training with {method} integration")
    print(f"{'='*50}")

    history = []
    start_time = time.time()

    for epoch in range(cfg['epochs']):
        model.train()
        ep_loss = 0

        for x_seq, f_seq, y_seq in loader:
            x_seq = x_seq.to(cfg['device'])
            f_seq = f_seq.to(cfg['device'])
            y_seq = y_seq.to(cfg['device'])

            opt.zero_grad()
            total_loss = 0
            curr_input = x_seq
            prev_rho = None

            for k in range(rollout):
                next_field = f_seq[:, k, :]
                pred_rho = model(curr_input, next_field, prev_rho=prev_rho)
                total_loss += crit(pred_rho, y_seq[:, k])

                if k < rollout - 1:
                    if use_verlet:
                        prev_rho = curr_input[:, -1].clone()
                    new_step = pred_rho.unsqueeze(1)
                    curr_input = torch.cat([curr_input[:, 1:], new_step], dim=1)

            total_loss = total_loss / rollout
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += total_loss.item()

        avg_loss = ep_loss / len(loader)
        history.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.7f}")

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f}s")

    # Test rollout stability with ground truth comparison
    model.eval()
    rollout_test = cfg.get('rollout_test', 50)

    # Enable trace projection for inference if configured
    if use_verlet and cfg.get('trace_projection_inference', False):
        model.trace_projection = True
        print(f"  [Inference] Trace projection enabled")

    with torch.no_grad():
        # Get a batch and the full ground truth sequence
        x_seq, f_seq, y_seq = next(iter(loader))
        x_seq = x_seq.to(cfg['device'])
        f_seq = f_seq.to(cfg['device'])
        y_seq = y_seq.to(cfg['device'])

        # We need more ground truth data - reload with longer rollout
        test_data = SkipStepDataset(
            cfg['density_file'], cfg['field_file'],
            cfg['seq_len'], rollout_steps=rollout_test,
            skip_factor=cfg.get('skip_factor', 1), dtype=cfg['dtype']
        )
        if len(test_data) > 0:
            x_seq_full, f_seq_full, y_seq_full = test_data[0]
            x_seq_full = x_seq_full.unsqueeze(0).to(cfg['device'])
            f_seq_full = f_seq_full.unsqueeze(0).to(cfg['device'])
            y_seq_full = y_seq_full.unsqueeze(0).to(cfg['device'])
        else:
            # Fallback - just use what we have
            x_seq_full, f_seq_full, y_seq_full = x_seq, f_seq, y_seq
            rollout_test = min(rollout_test, f_seq_full.shape[1])

        curr_input = x_seq_full
        prev_rho = None
        rollout_errors = []  # Error vs ground truth
        trace_errors = []    # Trace conservation error

        for step in range(min(rollout_test, y_seq_full.shape[1])):
            next_field = f_seq_full[:, step, :]
            pred_rho = model(curr_input, next_field, prev_rho=prev_rho)
            target_rho = y_seq_full[:, step]

            # Error vs ground truth (Frobenius norm)
            error = torch.norm(pred_rho - target_rho) / (torch.norm(target_rho) + 1e-10)
            rollout_errors.append(error.item())

            # Trace error (should be ~1 for alpha, ~0 for beta)
            trace_alpha = torch.real(torch.diagonal(pred_rho[:, 0], dim1=-2, dim2=-1).sum(-1))
            trace_error = torch.abs(trace_alpha - 1.0).mean()
            trace_errors.append(trace_error.item())

            if use_verlet:
                prev_rho = curr_input[:, -1].clone()
            new_step = pred_rho.unsqueeze(1)
            curr_input = torch.cat([curr_input[:, 1:], new_step], dim=1)

    return {
        'method': method,
        'final_loss': history[-1],
        'history': history,
        'time': elapsed,
        'rollout_errors': rollout_errors,
        'trace_errors': trace_errors,
        'model': model
    }


def main():
    print("=" * 60)
    print("VERLET vs EULER INTEGRATION COMPARISON")
    print("=" * 60)
    print(f"Device: {CFG['device']}")
    print(f"Skip Factor: k={CFG['skip_factor']}")
    print(f"Epochs: {CFG['epochs']}, Rollout Test: {CFG['rollout_test']} steps")
    print(f"\nVerlet Stabilization:")
    print(f"  Damping: {CFG['verlet_damping']} (friction coefficient)")
    print(f"  Blend:   {CFG['verlet_blend']} (0=pure Verlet, 1=pure Euler)")
    print(f"  Trace Projection: {CFG['trace_projection']}")

    # Train both models
    euler_results = train_model(use_verlet=False, cfg=CFG)
    verlet_results = train_model(use_verlet=True, cfg=CFG)

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Euler':>15} {'Verlet':>15} {'Winner':>10}")
    print("-" * 65)

    # Final loss
    euler_loss = euler_results['final_loss']
    verlet_loss = verlet_results['final_loss']
    winner = "Verlet" if verlet_loss < euler_loss else "Euler"
    print(f"{'Final Loss':<25} {euler_loss:>15.7f} {verlet_loss:>15.7f} {winner:>10}")

    # Training time
    euler_time = euler_results['time']
    verlet_time = verlet_results['time']
    winner = "Euler" if euler_time < verlet_time else "Verlet"
    print(f"{'Training Time (s)':<25} {euler_time:>15.1f} {verlet_time:>15.1f} {winner:>10}")

    # Rollout stability - early (steps 1-10)
    euler_early = np.mean(euler_results['rollout_errors'][:10])
    verlet_early = np.mean(verlet_results['rollout_errors'][:10])
    winner = "Verlet" if verlet_early < euler_early else "Euler"
    print(f"{'Early Rollout (1-10)':<25} {euler_early:>15.2e} {verlet_early:>15.2e} {winner:>10}")

    # Rollout stability - late (last 10 steps)
    euler_late = np.mean(euler_results['rollout_errors'][-10:])
    verlet_late = np.mean(verlet_results['rollout_errors'][-10:])
    winner = "Verlet" if verlet_late < euler_late else "Euler"
    print(f"{'Late Rollout (last 10)':<25} {euler_late:>15.2e} {verlet_late:>15.2e} {winner:>10}")

    # Error growth rate (late / early)
    euler_growth = euler_late / (euler_early + 1e-10)
    verlet_growth = verlet_late / (verlet_early + 1e-10)
    winner = "Verlet" if verlet_growth < euler_growth else "Euler"
    print(f"{'Error Growth Rate':<25} {euler_growth:>15.1f}x {verlet_growth:>15.1f}x {winner:>10}")

    # Trace conservation
    euler_trace = np.mean(euler_results['trace_errors'][-10:])
    verlet_trace = np.mean(verlet_results['trace_errors'][-10:])
    winner = "Verlet" if verlet_trace < euler_trace else "Euler"
    print(f"{'Trace Error (late)':<25} {euler_trace:>15.2e} {verlet_trace:>15.2e} {winner:>10}")

    # Loss improvement over epochs
    euler_improve = (euler_results['history'][0] - euler_results['history'][-1]) / euler_results['history'][0]
    verlet_improve = (verlet_results['history'][0] - verlet_results['history'][-1]) / verlet_results['history'][0]
    winner = "Verlet" if verlet_improve > euler_improve else "Euler"
    print(f"{'Loss Improvement (%)':<25} {euler_improve*100:>15.1f} {verlet_improve*100:>15.1f} {winner:>10}")

    # Convergence plot (text-based)
    print("\n" + "=" * 60)
    print("LOSS CONVERGENCE (scaled)")
    print("=" * 60)

    max_loss = max(max(euler_results['history']), max(verlet_results['history']))
    width = 40

    for i in range(0, len(euler_results['history']), 5):
        euler_bar = int(euler_results['history'][i] / max_loss * width)
        verlet_bar = int(verlet_results['history'][i] / max_loss * width)

        print(f"Ep {i+1:2d} E: {'█' * euler_bar}{' ' * (width - euler_bar)} | "
              f"V: {'█' * verlet_bar}")

    print("\n" + "=" * 60)
    print("ROLLOUT ERROR vs GROUND TRUTH (50 steps)")
    print("=" * 60)

    steps_to_show = [0, 4, 9, 19, 29, 39, 49]
    print(f"{'Step':>6} {'Euler':>12} {'Verlet':>12} {'Winner':>10}")
    print("-" * 42)
    for i in steps_to_show:
        if i < len(euler_results['rollout_errors']) and i < len(verlet_results['rollout_errors']):
            e_err = euler_results['rollout_errors'][i]
            v_err = verlet_results['rollout_errors'][i]
            winner = "Verlet" if v_err < e_err else "Euler"
            print(f"{i+1:>6} {e_err:>12.4f} {v_err:>12.4f} {winner:>10}")

    print("\n" + "=" * 60)
    print("TRACE CONSERVATION (Tr(ρ) should = 1)")
    print("=" * 60)

    print(f"{'Step':>6} {'Euler':>12} {'Verlet':>12} {'Winner':>10}")
    print("-" * 42)
    for i in steps_to_show:
        if i < len(euler_results['trace_errors']) and i < len(verlet_results['trace_errors']):
            e_tr = euler_results['trace_errors'][i]
            v_tr = verlet_results['trace_errors'][i]
            winner = "Verlet" if v_tr < e_tr else "Euler"
            print(f"{i+1:>6} {e_tr:>12.4f} {v_tr:>12.4f} {winner:>10}")

    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
