"""
Training script for DensityGraphNet.

This script trains the graph neural network model for density matrix
propagation using rollout training similar to train_benchmark.py.

Key differences from LSTM training:
    - Uses MolecularGraphDataset instead of MolecularDynamicsDataset
    - Operates on graph sequences instead of tensor sequences
    - Physics constraints applied at edge level

Usage:
    python train_graph.py --config inputs/train_graph.json
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
from graph_data import (
    MolecularGraphDataset,
    MolecularGraph,
    density_to_edge_features,
    edge_features_to_density,
    collate_graph_sequences
)
from graph_model import (
    DensityGraphNet,
    GraphPhysicsLoss,
    count_parameters,
    create_model_from_dataset_info
)


# --- Global Config ---
CFG: Dict = {}


def train_epoch(
    model: DensityGraphNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: GraphPhysicsLoss,
    rollout: int,
    device: torch.device,
    S: torch.Tensor,
    noise_scale: float = 1e-3
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: DensityGraphNet model
        loader: DataLoader with MolecularGraphDataset
        optimizer: Optimizer
        criterion: GraphPhysicsLoss
        rollout: Number of rollout steps
        device: Training device
        S: Overlap matrix
        noise_scale: Noise injection scale for regularization

    Returns:
        avg_loss: Average loss for epoch
        avg_metrics: Dictionary of average metric values
    """
    model.train()
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0

    for batch in loader:
        input_seqs, field_targets, target_seqs = batch

        # Process each sequence in batch independently
        # (Graph batching is more complex, we process sequentially for now)
        batch_loss = 0.0
        batch_metrics = {}

        for seq_idx in range(len(input_seqs)):
            input_graphs = input_seqs[seq_idx]
            fields = field_targets[seq_idx]  # (rollout, 3)
            target_graphs = target_seqs[seq_idx]

            # Move graphs to device
            input_graphs = [move_graph_to_device(g, device) for g in input_graphs]
            target_graphs = [move_graph_to_device(g, device) for g in target_graphs]
            fields = fields.to(device)

            # Noise injection on input (drift simulation)
            if noise_scale > 0:
                for g in input_graphs:
                    g.edge_density = g.edge_density + torch.randn_like(g.edge_density) * noise_scale

            optimizer.zero_grad()
            seq_loss = 0.0
            curr_seq = list(input_graphs)

            # Rollout loop
            for k in range(rollout):
                next_field = fields[k]  # (3,)
                target_graph = target_graphs[k]

                # Compute target delta
                last_density = curr_seq[-1].edge_density
                target_density = target_graph.edge_density
                delta_target = target_density - last_density

                # Predict delta
                delta_pred = model(curr_seq, next_field)

                # Compute loss
                loss, loss_dict = criterion(
                    delta_pred,
                    delta_target,
                    curr_seq[-1].edge_index,
                    last_density,
                    S=S,
                    n_basis=curr_seq[-1].num_nodes
                )

                seq_loss = seq_loss + loss

                # Accumulate metrics
                for key, val in loss_dict.items():
                    batch_metrics[key] = batch_metrics.get(key, 0.0) + val.item()

                # Autoregressive feed: create new graph with predicted density
                if k < rollout - 1:
                    new_graph = curr_seq[-1].clone()
                    new_graph.edge_density = last_density + delta_pred.detach()
                    curr_seq = curr_seq[1:] + [new_graph]

            # Average over rollout steps
            seq_loss = seq_loss / rollout
            batch_loss += seq_loss

        # Average over batch
        batch_loss = batch_loss / len(input_seqs)
        batch_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += batch_loss.item()
        for key, val in batch_metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + val / (len(input_seqs) * rollout)
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def validate(
    model: DensityGraphNet,
    loader: DataLoader,
    criterion: GraphPhysicsLoss,
    rollout: int,
    device: torch.device,
    S: torch.Tensor
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model.

    Args:
        model: DensityGraphNet model
        loader: Validation DataLoader
        criterion: GraphPhysicsLoss
        rollout: Number of rollout steps
        device: Device
        S: Overlap matrix

    Returns:
        avg_loss: Average validation loss
        avg_metrics: Dictionary of average metrics
    """
    model.eval()
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_seqs, field_targets, target_seqs = batch

            batch_loss = 0.0
            batch_metrics = {}

            for seq_idx in range(len(input_seqs)):
                input_graphs = input_seqs[seq_idx]
                fields = field_targets[seq_idx]
                target_graphs = target_seqs[seq_idx]

                input_graphs = [move_graph_to_device(g, device) for g in input_graphs]
                target_graphs = [move_graph_to_device(g, device) for g in target_graphs]
                fields = fields.to(device)

                seq_loss = 0.0
                curr_seq = list(input_graphs)

                for k in range(rollout):
                    next_field = fields[k]
                    target_graph = target_graphs[k]

                    last_density = curr_seq[-1].edge_density
                    target_density = target_graph.edge_density
                    delta_target = target_density - last_density

                    delta_pred = model(curr_seq, next_field)

                    loss, loss_dict = criterion(
                        delta_pred,
                        delta_target,
                        curr_seq[-1].edge_index,
                        last_density,
                        S=S,
                        n_basis=curr_seq[-1].num_nodes
                    )

                    seq_loss = seq_loss + loss

                    for key, val in loss_dict.items():
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + val.item()

                    if k < rollout - 1:
                        new_graph = curr_seq[-1].clone()
                        new_graph.edge_density = last_density + delta_pred
                        curr_seq = curr_seq[1:] + [new_graph]

                seq_loss = seq_loss / rollout
                batch_loss += seq_loss.item()

            batch_loss = batch_loss / len(input_seqs)
            total_loss += batch_loss
            for key, val in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + val / (len(input_seqs) * rollout)
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def move_graph_to_device(graph: MolecularGraph, device: torch.device) -> MolecularGraph:
    """Move all graph tensors to specified device."""
    return MolecularGraph(
        edge_index=graph.edge_index.to(device),
        num_nodes=graph.num_nodes,
        node_features=graph.node_features.to(device),
        edge_static=graph.edge_static.to(device),
        edge_density=graph.edge_density.to(device),
        atomic_numbers=graph.atomic_numbers.to(device),
        positions=graph.positions.to(device)
    )


def train():
    """Main training function."""
    print("=" * 60)
    print("DensityGraphNet Training")
    print("=" * 60)

    # Load overlap matrix
    try:
        S = torch.tensor(np.load(CFG['overlap_file']), dtype=torch.float64)
        print(f"Loaded overlap matrix: {S.shape}")
    except Exception as e:
        print(f"Warning: Could not load overlap matrix: {e}")
        S = torch.eye(CFG.get('n_basis', 4), dtype=torch.float64)

    S = S.to(CFG['device'])

    # Create dataset
    rollout = CFG.get('rollout_steps', 5)
    seq_len = CFG.get('seq_len', 20)

    print(f"\nLoading dataset with seq_len={seq_len}, rollout={rollout}...")

    dataset = MolecularGraphDataset(
        density_file=CFG['density_file'],
        field_file=CFG['field_file'],
        overlap_file=CFG['overlap_file'],
        positions_file=CFG.get('positions_file'),
        atomic_numbers_file=CFG.get('atomic_numbers_file'),
        seq_len=seq_len,
        rollout_steps=rollout,
        overlap_threshold=CFG.get('overlap_threshold', 1e-6),
        time_reversal_correction=CFG.get('time_reversal_correction', True)
    )

    # Get dataset info for model creation
    dataset_info = dataset.get_graph_info()
    print(f"Dataset info: {dataset_info}")
    print(f"Dataset size: {len(dataset)} samples")

    # Split into train/val
    val_split = CFG.get('val_split', 0.1)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.get('batch_size', 8),
        shuffle=True,
        collate_fn=collate_graph_sequences
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.get('batch_size', 8),
        shuffle=False,
        collate_fn=collate_graph_sequences
    )

    # Create model
    print("\nCreating model...")
    model_config = {
        'node_dim': dataset_info['node_feature_dim'] + 1,  # +1 for field coupling
        'edge_dim': dataset_info['edge_feature_dim'],
        'hidden_dim': CFG.get('hidden_dim', 128),
        'n_gnn_layers': CFG.get('n_gnn_layers', 3),
        'n_lstm_layers': CFG.get('n_lstm_layers', 2),
        'n_attention_heads': CFG.get('n_attention_heads', 4),
        'dropout': CFG.get('dropout', 0.1)
    }

    model = DensityGraphNet(model_config).to(CFG['device'])
    print(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.get('learning_rate', 1e-3),
        weight_decay=CFG.get('weight_decay', 1e-4)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    # Create loss function
    criterion = GraphPhysicsLoss(
        lambda_trace=CFG.get('lambda_trace', 1e-4),
        lambda_idem=CFG.get('lambda_idem', 1e-5),
        n_electrons_alpha=CFG.get('n_alpha', 1.0),
        n_electrons_beta=CFG.get('n_beta', 0.0)
    )

    # Training loop
    epochs = CFG.get('epochs', 200)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = CFG.get('patience', 50)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            rollout, CFG['device'], S,
            noise_scale=CFG.get('noise_scale', 1e-3)
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion,
            rollout, CFG['device'], S
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_path = CFG.get('model_save_path', 'density_graph.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)
        else:
            patience_counter += 1

        # Logging
        elapsed = time.time() - start_time
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

            # Print physics metrics
            if train_metrics.get('trace'):
                print(f"         | Trace: {train_metrics['trace']:.6f} | "
                      f"Idem: {train_metrics.get('idem', 0):.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("-" * 60)
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {CFG.get('model_save_path', 'density_graph.pt')}")

    return model, dataset


def load_config(json_path: str, args) -> Dict:
    """Load and flatten configuration."""
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Flatten nested structure
    flat = {}
    for section in config:
        if isinstance(config[section], dict):
            flat.update(config[section])
        else:
            flat[section] = config[section]

    # CLI overrides
    if hasattr(args, 'predict') and args.predict:
        flat['predict_only'] = True

    # Set device
    flat['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flat['dtype'] = torch.complex128

    return flat


def create_default_config():
    """Create default configuration file."""
    config = {
        "io": {
            "density_file": "test_train/densities/density_series.npy",
            "overlap_file": "test_train/data/h2_plus_rttddft_overlap.npy",
            "field_file": "test_train/data/field.dat",
            "positions_file": None,
            "atomic_numbers_file": None,
            "model_save_path": "density_graph.pt"
        },
        "system": {
            "n_basis": 4,
            "n_alpha": 1.0,
            "n_beta": 0.0
        },
        "model": {
            "seq_len": 20,
            "hidden_dim": 128,
            "n_gnn_layers": 3,
            "n_lstm_layers": 2,
            "n_attention_heads": 4,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 200,
            "rollout_steps": 5,
            "val_split": 0.1,
            "patience": 50,
            "noise_scale": 1e-3
        },
        "physics": {
            "lambda_trace": 1e-4,
            "lambda_idem": 1e-5,
            "overlap_threshold": 1e-6,
            "time_reversal_correction": True
        }
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DensityGraphNet")
    parser.add_argument('--config', type=str, default='inputs/train_graph.json',
                       help='Path to configuration JSON')
    parser.add_argument('--predict', action='store_true',
                       help='Run prediction only')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default config file and exit')
    args = parser.parse_args()

    if args.create_config:
        config = create_default_config()
        config_path = Path('inputs/train_graph.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created default config at {config_path}")
    else:
        # Check if config exists
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            print("Creating default config...")
            config = create_default_config()
            config_path = Path(args.config)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Created config at {config_path}")

        CFG = load_config(args.config, args)

        if not CFG.get('predict_only', False):
            train()
