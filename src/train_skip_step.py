"""
Training script for Skip-Step Propagation.

This script trains models to predict t → t + k·Δt mappings where k > 1,
enabling significant speedups in density matrix propagation.

Key Features:
    - Curriculum learning: Gradually increase k during training
    - Multiple supervision modes: endpoints, intermediate, multi-scale
    - Adaptive regularization: Stronger constraints for larger k
    - Works with both LSTM and GNN models

Usage:
    python train_skip_step.py --config inputs/train_skip.json --model lstm
    python train_skip_step.py --config inputs/train_skip.json --model gnn

Example workflow:
    1. Train with k=1 (standard) for baseline
    2. Fine-tune with curriculum k=1→20 for acceleration
    3. Evaluate speedup vs accuracy trade-off
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Local imports
from skip_step import (
    SkipStepConfig,
    SkipStepDatasetLSTM,
    SkipStepCurriculum,
    SkipStepLoss,
    AdaptiveSkipLoss,
    SupervisionMode,
    compute_effective_dt
)

# Model imports
from train_benchmark import (
    MolecularDynamicsDataset,
    DensityMatrixLSTM,
    PhysicsLoss,
    CFG as LSTM_CFG
)

# Try to import GNN components
try:
    from graph_data import MolecularGraphDataset, collate_graph_sequences
    from graph_model import DensityGraphNet, GraphPhysicsLoss, count_parameters
    HAS_GNN = True
except ImportError:
    HAS_GNN = False


# Global config
CFG: Dict = {}


def load_data_for_skip_step(model_type: str) -> Tuple:
    """
    Load data and create skip-step dataset.

    Args:
        model_type: 'lstm' or 'gnn'

    Returns:
        dataset: Skip-step dataset
        S: Overlap matrix
    """
    # Load raw data
    print(f"Loading data from {CFG['density_file']}...")
    data_dict = np.load(CFG['density_file'], allow_pickle=True)
    if isinstance(data_dict, np.ndarray) and data_dict.dtype == object:
        data_dict = data_dict.item()

    rho = torch.tensor(data_dict['density'], dtype=torch.complex128)

    # Time reversal correction
    if CFG.get('time_reversal_correction', True):
        rho = torch.conj(rho)

    # Load field
    try:
        field_data = np.loadtxt(CFG['field_file'])
        if field_data.ndim == 1:
            field_data = field_data.reshape(-1, 3)
        elif field_data.shape[1] == 4:
            field_data = field_data[:, 1:]
        field = torch.tensor(field_data, dtype=torch.float64)
    except Exception:
        print("Warning: Could not load field, using zeros")
        field = torch.zeros(len(rho), 3, dtype=torch.float64)

    # Align lengths
    min_len = min(len(rho), len(field))
    rho = rho[:min_len]
    field = field[:min_len]

    print(f"Loaded {len(rho)} timesteps, shape: {rho.shape}")

    # Load overlap matrix
    try:
        S = torch.tensor(np.load(CFG['overlap_file']), dtype=torch.float64)
    except Exception:
        print("Warning: Could not load overlap matrix, using identity")
        n_basis = rho.shape[-1]
        S = torch.eye(n_basis, dtype=torch.float64)

    # Create skip-step dataset
    skip_config = SkipStepConfig(
        skip_factor=CFG.get('initial_skip', 1),  # Start with curriculum initial
        supervision_mode=CFG.get('supervision_mode', 'endpoints'),
        intermediate_samples=CFG.get('intermediate_samples', 2),
        use_curriculum=CFG.get('use_curriculum', True)
    )

    if model_type == 'lstm':
        dataset = SkipStepDatasetLSTM(
            density_data=rho,
            field_data=field,
            seq_len=CFG['seq_len'],
            skip_factor=skip_config.skip_factor,
            rollout_steps=CFG.get('rollout_steps', 5),
            supervision_mode=skip_config.supervision_mode,
            intermediate_samples=skip_config.intermediate_samples
        )
    else:
        # For GNN, use the base dataset with skip-step wrapper
        # (simplified - in practice would need proper graph construction)
        dataset = SkipStepDatasetLSTM(
            density_data=rho,
            field_data=field,
            seq_len=CFG['seq_len'],
            skip_factor=skip_config.skip_factor,
            rollout_steps=CFG.get('rollout_steps', 5),
            supervision_mode=skip_config.supervision_mode
        )

    return dataset, S, skip_config


def create_model(model_type: str) -> nn.Module:
    """Create model based on type."""
    if model_type == 'lstm':
        # Update global config for LSTM model
        global LSTM_CFG
        LSTM_CFG.update({
            'n_basis': CFG['n_basis'],
            'hidden_dim': CFG['hidden_dim'],
            'device': CFG['device'],
            'dtype': torch.complex128
        })
        model = DensityMatrixLSTM()

    elif model_type == 'gnn':
        if not HAS_GNN:
            raise ImportError("GNN components not available. Install torch_geometric.")

        config = {
            'node_dim': CFG.get('node_dim', 24),
            'edge_dim': CFG.get('edge_dim', 6),
            'hidden_dim': CFG['hidden_dim'],
            'n_gnn_layers': CFG.get('n_gnn_layers', 3),
            'n_lstm_layers': CFG.get('n_lstm_layers', 2),
            'n_attention_heads': CFG.get('n_attention_heads', 4),
            'dropout': CFG.get('dropout', 0.1)
        }
        model = DensityGraphNet(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_epoch_skip_step(
    model: nn.Module,
    dataset: SkipStepDatasetLSTM,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    k: int,
    device: torch.device,
    S: torch.Tensor,
    batch_size: int = 32
) -> Tuple[float, Dict[str, float]]:
    """
    Train one epoch with skip-step propagation.

    Args:
        model: Model to train
        dataset: Skip-step dataset
        optimizer: Optimizer
        criterion: Loss function
        k: Current skip factor
        device: Training device
        S: Overlap matrix
        batch_size: Batch size

    Returns:
        avg_loss: Average loss
        metrics: Dictionary of metrics
    """
    model.train()

    # Update dataset skip factor
    dataset.set_skip_factor(k)

    # Create dataloader (need custom collate for skip-step)
    def collate_fn(batch):
        rho_seqs = torch.stack([b[0] for b in batch])
        field_targets = torch.stack([b[1] for b in batch])
        rho_targets = torch.stack([b[2] for b in batch])
        # Skip supervision dict for simplicity
        return rho_seqs, field_targets, rho_targets

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0
    rollout = CFG.get('rollout_steps', 5)

    for x_seq, f_seq, y_seq in loader:
        x_seq = x_seq.to(device)
        f_seq = f_seq.to(device)
        y_seq = y_seq.to(device)

        # Noise injection
        if CFG.get('noise_scale', 0) > 0:
            x_seq = x_seq + torch.randn_like(x_seq) * CFG['noise_scale']

        optimizer.zero_grad()
        batch_loss = 0.0
        curr_input = x_seq
        prev_pred = None

        # Rollout loop
        for step in range(rollout):
            next_field = f_seq[:, step, :]

            # Forward pass
            pred_rho = model(curr_input, next_field)

            # Compute loss
            target_rho = y_seq[:, step]

            if isinstance(criterion, (SkipStepLoss, AdaptiveSkipLoss)):
                if isinstance(criterion, AdaptiveSkipLoss):
                    loss, loss_dict = criterion(pred_rho, target_rho, k=k)
                else:
                    loss, loss_dict = criterion(
                        pred_rho, target_rho,
                        prev_pred=prev_pred
                    )
            else:
                loss = criterion(pred_rho, target_rho)
                loss_dict = {'mse': loss.detach()}

            batch_loss = batch_loss + loss

            # Track metrics
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + val.item()

            # Autoregressive feed
            prev_pred = pred_rho.detach()
            if step < rollout - 1:
                new_step = pred_rho.unsqueeze(1).detach()
                curr_input = torch.cat([curr_input[:, 1:], new_step], dim=1)

        batch_loss = batch_loss / rollout
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_metrics = {key: val / max(1, n_batches * rollout)
                   for key, val in metrics_sum.items()}
    avg_metrics['k'] = k

    return avg_loss, avg_metrics


def validate_skip_step(
    model: nn.Module,
    dataset: SkipStepDatasetLSTM,
    criterion: nn.Module,
    k: int,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[float, Dict[str, float]]:
    """Validate with current skip factor."""
    model.eval()
    dataset.set_skip_factor(k)

    def collate_fn(batch):
        rho_seqs = torch.stack([b[0] for b in batch])
        field_targets = torch.stack([b[1] for b in batch])
        rho_targets = torch.stack([b[2] for b in batch])
        return rho_seqs, field_targets, rho_targets

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    n_batches = 0
    rollout = CFG.get('rollout_steps', 5)

    with torch.no_grad():
        for x_seq, f_seq, y_seq in loader:
            x_seq = x_seq.to(device)
            f_seq = f_seq.to(device)
            y_seq = y_seq.to(device)

            batch_loss = 0.0
            curr_input = x_seq

            for step in range(rollout):
                next_field = f_seq[:, step, :]
                pred_rho = model(curr_input, next_field)
                target_rho = y_seq[:, step]

                if hasattr(criterion, 'forward'):
                    result = criterion(pred_rho, target_rho)
                    loss = result[0] if isinstance(result, tuple) else result
                else:
                    loss = F.mse_loss(pred_rho.real, target_rho.real) + \
                           F.mse_loss(pred_rho.imag, target_rho.imag)

                batch_loss += loss.item()

                if step < rollout - 1:
                    new_step = pred_rho.unsqueeze(1)
                    curr_input = torch.cat([curr_input[:, 1:], new_step], dim=1)

            total_loss += batch_loss / rollout
            n_batches += 1

    return total_loss / max(1, n_batches), {'k': k}


def train():
    """Main training function with skip-step and curriculum learning."""
    print("=" * 60)
    print("Skip-Step Training")
    print("=" * 60)

    model_type = CFG.get('model_type', 'lstm')
    print(f"Model type: {model_type}")

    # Load data
    dataset, S, skip_config = load_data_for_skip_step(model_type)
    S = S.to(CFG['device'])
    print(f"Dataset samples: {len(dataset)}")

    # Train/val split
    val_split = CFG.get('val_split', 0.1)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    # We'll handle this differently since skip-step dataset needs k updates
    # For simplicity, use same dataset and just validate on later samples
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))

    # Create subsets manually
    class SubsetWrapper:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def set_skip_factor(self, k):
            self.dataset.set_skip_factor(k)
            # Recompute valid indices
            self.indices = [i for i in self.indices if i < len(self.dataset)]

    train_data = SubsetWrapper(dataset, train_indices)
    val_data = SubsetWrapper(dataset, val_indices)

    # Create model
    model = create_model(model_type).to(CFG['device'])

    if hasattr(model, 'parameters'):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

    # Load pretrained weights if specified
    if CFG.get('pretrained_path'):
        print(f"Loading pretrained weights from {CFG['pretrained_path']}")
        checkpoint = torch.load(CFG['pretrained_path'], map_location=CFG['device'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.get('learning_rate', 1e-3),
        weight_decay=CFG.get('weight_decay', 1e-4)
    )

    # Create curriculum scheduler
    curriculum = SkipStepCurriculum(
        initial_k=CFG.get('initial_skip', 1),
        final_k=CFG.get('final_skip', 20),
        warmup_epochs=CFG.get('curriculum_warmup', 20),
        total_epochs=CFG.get('epochs', 200),
        schedule=CFG.get('curriculum_schedule', 'exponential')
    )

    # Create loss function
    base_loss = PhysicsLoss(S)
    criterion = AdaptiveSkipLoss(
        SkipStepLoss(
            base_loss,
            lambda_smoothness=CFG.get('lambda_smoothness', 1e-4),
            lambda_energy=CFG.get('lambda_energy', 1e-4),
            lambda_phase=CFG.get('lambda_phase', 1e-5),
            S=S
        ),
        k_ref=1,
        smoothness_scale=CFG.get('smoothness_scale', 0.1),
        energy_scale=CFG.get('energy_scale', 0.1)
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    # Training loop
    epochs = CFG.get('epochs', 200)
    best_val_loss = float('inf')
    history = []

    print(f"\nTraining for {epochs} epochs with curriculum k: {curriculum.initial_k} → {curriculum.final_k}")
    print("-" * 60)

    for epoch in range(epochs):
        start_time = time.time()

        # Get current skip factor from curriculum
        k = curriculum.step(epoch)

        # Train
        train_loss, train_metrics = train_epoch_skip_step(
            model, dataset, optimizer, criterion, k,
            CFG['device'], S, CFG.get('batch_size', 32)
        )

        # Validate
        val_loss, val_metrics = validate_skip_step(
            model, dataset, criterion, k,
            CFG['device'], CFG.get('batch_size', 32)
        )

        lr_scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CFG.get('model_save_path', 'skip_step_model.pt')
            torch.save({
                'epoch': epoch,
                'k': k,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curriculum_state': curriculum.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': dict(CFG)
            }, save_path)

        elapsed = time.time() - start_time

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0 or k != curriculum.step(max(0, epoch - 1)):
            dt_eff = compute_effective_dt(CFG.get('dt_fine', 0.4), k)
            print(f"Epoch {epoch+1:4d} | k={k:2d} (Δt_eff={dt_eff:.1f}) | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | Time: {elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'k': k,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    print("-" * 60)
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")

    # Save training history
    history_path = CFG.get('history_path', 'skip_step_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")

    return model


def evaluate_speedup(
    model: nn.Module,
    dataset,
    k_values: List[int],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Evaluate model at different skip factors.

    Returns speedup vs accuracy trade-off data.
    """
    results = {'k': [], 'error': [], 'speedup': []}

    model.eval()
    reference_k = 1

    for k in k_values:
        dataset.set_skip_factor(k)

        errors = []
        with torch.no_grad():
            for i in range(min(100, len(dataset))):
                x_seq, f_seq, y_seq, _ = dataset[i]
                x_seq = x_seq.unsqueeze(0).to(device)
                f_seq = f_seq.unsqueeze(0).to(device)
                y_seq = y_seq.unsqueeze(0).to(device)

                # Single step prediction
                pred = model(x_seq, f_seq[:, 0, :])
                target = y_seq[:, 0]

                error = torch.norm(pred - target) / torch.norm(target)
                errors.append(error.item())

        avg_error = np.mean(errors)
        speedup = k / reference_k

        results['k'].append(k)
        results['error'].append(avg_error)
        results['speedup'].append(speedup)

        print(f"k={k:2d}: Error={avg_error:.4f}, Speedup={speedup:.1f}x")

    return results


def load_config(json_path: str, args) -> Dict:
    """Load configuration."""
    with open(json_path, 'r') as f:
        config = json.load(f)

    flat = {}
    for section in config:
        if isinstance(config[section], dict):
            flat.update(config[section])
        else:
            flat[section] = config[section]

    # CLI overrides
    if hasattr(args, 'model') and args.model:
        flat['model_type'] = args.model

    flat['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return flat


def create_default_config() -> Dict:
    """Create default skip-step configuration."""
    return {
        "io": {
            "density_file": "test_train/densities/density_series.npy",
            "overlap_file": "test_train/data/h2_plus_rttddft_overlap.npy",
            "field_file": "test_train/data/field.dat",
            "model_save_path": "skip_step_model.pt",
            "history_path": "skip_step_history.json",
            "pretrained_path": None
        },
        "system": {
            "n_basis": 4,
            "dt_fine": 0.4
        },
        "model": {
            "model_type": "lstm",
            "seq_len": 20,
            "hidden_dim": 128
        },
        "skip_step": {
            "initial_skip": 1,
            "final_skip": 20,
            "supervision_mode": "endpoints",
            "intermediate_samples": 2
        },
        "curriculum": {
            "use_curriculum": True,
            "curriculum_warmup": 20,
            "curriculum_schedule": "exponential"
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 200,
            "rollout_steps": 5,
            "val_split": 0.1,
            "noise_scale": 1e-3
        },
        "regularization": {
            "lambda_smoothness": 1e-4,
            "lambda_energy": 1e-4,
            "lambda_phase": 1e-5,
            "smoothness_scale": 0.1,
            "energy_scale": 0.1
        },
        "physics": {
            "time_reversal_correction": True
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with skip-step propagation")
    parser.add_argument('--config', type=str, default='inputs/train_skip.json')
    parser.add_argument('--model', type=str, choices=['lstm', 'gnn'], default='lstm')
    parser.add_argument('--create-config', action='store_true')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate speedup vs accuracy trade-off')
    args = parser.parse_args()

    if args.create_config:
        config = create_default_config()
        config_path = Path('inputs/train_skip.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created config at {config_path}")
    else:
        if not os.path.exists(args.config):
            print(f"Config not found: {args.config}")
            print("Creating default config...")
            config = create_default_config()
            config_path = Path(args.config)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

        CFG = load_config(args.config, args)

        if args.evaluate:
            # Load trained model and evaluate
            print("Evaluation mode - loading model...")
            model = create_model(CFG.get('model_type', 'lstm')).to(CFG['device'])
            checkpoint = torch.load(CFG['model_save_path'], map_location=CFG['device'])
            model.load_state_dict(checkpoint['model_state_dict'])

            dataset, S, _ = load_data_for_skip_step(CFG.get('model_type', 'lstm'))
            k_values = [1, 2, 5, 10, 15, 20]
            results = evaluate_speedup(model, dataset, k_values, CFG['device'])
        else:
            train()
