"""
Unified training script for DensityGraphNet with Skip-Step propagation.

This script provides a complete training pipeline that combines:
    - Graph Neural Network architecture for size-independent learning
    - Skip-step propagation for computational acceleration
    - Curriculum learning for stable training
    - Physics-constrained loss functions

Usage:
    # Standard training (k=1)
    python train_graph_skip.py --config inputs/train_graph_skip.json

    # Skip-step training with curriculum
    python train_graph_skip.py --config inputs/train_graph_skip.json --skip-step

    # Evaluate speedup/accuracy trade-off
    python train_graph_skip.py --config inputs/train_graph_skip.json --evaluate
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
    SkipStepGraphDataset,
    MolecularGraph,
    collate_skip_step_graphs,
    edge_features_to_density
)
from graph_model import (
    DensityGraphNet,
    SkipStepGraphLoss,
    count_parameters
)
from skip_step import SkipStepCurriculum


# Global config
CFG: Dict = {}


def move_graph_to_device(graph: MolecularGraph, device: torch.device) -> MolecularGraph:
    """Move all graph tensors to device."""
    return MolecularGraph(
        edge_index=graph.edge_index.to(device),
        num_nodes=graph.num_nodes,
        node_features=graph.node_features.to(device),
        edge_static=graph.edge_static.to(device),
        edge_density=graph.edge_density.to(device),
        atomic_numbers=graph.atomic_numbers.to(device),
        positions=graph.positions.to(device)
    )


def train_epoch(
    model: DensityGraphNet,
    dataset: SkipStepGraphDataset,
    optimizer: torch.optim.Optimizer,
    criterion: SkipStepGraphLoss,
    k: int,
    device: torch.device,
    S: torch.Tensor,
    batch_size: int = 8,
    rollout: int = 5,
    noise_scale: float = 1e-3
) -> Tuple[float, Dict[str, float]]:
    """
    Train one epoch with skip-step propagation.

    Args:
        model: DensityGraphNet model
        dataset: SkipStepGraphDataset
        optimizer: Optimizer
        criterion: Loss function
        k: Current skip factor
        device: Training device
        S: Overlap matrix
        batch_size: Batch size
        rollout: Rollout steps
        noise_scale: Noise for regularization

    Returns:
        avg_loss: Average epoch loss
        avg_metrics: Average metrics
    """
    model.train()
    dataset.set_skip_factor(k)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_skip_step_graphs
    )

    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0

    for input_seqs, field_targets, target_seqs, supervision_list in loader:
        batch_loss = 0.0
        batch_metrics = {}

        # Process each sequence in batch
        for seq_idx in range(len(input_seqs)):
            input_graphs = input_seqs[seq_idx]
            fields = field_targets[seq_idx]
            target_graphs = target_seqs[seq_idx]
            supervision = supervision_list[seq_idx]

            # Move to device
            input_graphs = [move_graph_to_device(g, device) for g in input_graphs]
            target_graphs = [move_graph_to_device(g, device) for g in target_graphs]
            fields = fields.to(device)

            # Noise injection
            if noise_scale > 0:
                for g in input_graphs:
                    g.edge_density = g.edge_density + torch.randn_like(g.edge_density) * noise_scale

            optimizer.zero_grad()
            seq_loss = 0.0
            curr_seq = list(input_graphs)
            prev_delta = None

            # For Verlet: track previous edge density
            use_verlet = model.use_verlet
            prev_edge_density = None

            # Rollout loop
            for step in range(min(rollout, len(target_graphs))):
                next_field = fields[step]
                target_graph = target_graphs[step]

                # Get current and target density
                last_density = curr_seq[-1].edge_density
                target_density = target_graph.edge_density

                # Predict new density with skip-factor conditioning
                # Model now returns full new density (applies Euler or Verlet internally)
                new_density_pred = model(
                    curr_seq, next_field, skip_factor=k,
                    prev_edge_density=prev_edge_density
                )

                # Convert to delta for loss computation (both methods evaluated same way)
                delta_pred = new_density_pred - last_density
                delta_target = target_density - last_density

                # Compute loss
                loss, loss_dict = criterion(
                    delta_pred,
                    delta_target,
                    curr_seq[-1].edge_index,
                    last_density,
                    S=S,
                    n_basis=curr_seq[-1].num_nodes,
                    skip_factor=k,
                    prev_delta=prev_delta
                )

                seq_loss = seq_loss + loss

                # Accumulate metrics
                for key, val in loss_dict.items():
                    if isinstance(val, torch.Tensor):
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + val.item()
                    else:
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + val

                # Autoregressive feed
                prev_delta = delta_pred.detach()
                if step < rollout - 1:
                    # Track previous density for Verlet
                    if use_verlet:
                        prev_edge_density = last_density.clone()

                    new_graph = curr_seq[-1].clone()
                    new_graph.edge_density = new_density_pred.detach()
                    curr_seq = curr_seq[1:] + [new_graph]

            seq_loss = seq_loss / rollout
            batch_loss += seq_loss

        # Average over batch and backprop
        batch_loss = batch_loss / len(input_seqs)
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        for key, val in batch_metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + val / (len(input_seqs) * rollout)
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_metrics = {k: v / max(1, n_batches) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def validate(
    model: DensityGraphNet,
    dataset: SkipStepGraphDataset,
    criterion: SkipStepGraphLoss,
    k: int,
    device: torch.device,
    S: torch.Tensor,
    batch_size: int = 8,
    rollout: int = 5
) -> Tuple[float, Dict[str, float]]:
    """Validate model."""
    model.eval()
    dataset.set_skip_factor(k)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_skip_step_graphs
    )

    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0

    with torch.no_grad():
        for input_seqs, field_targets, target_seqs, supervision_list in loader:
            batch_loss = 0.0

            for seq_idx in range(len(input_seqs)):
                input_graphs = [move_graph_to_device(g, device) for g in input_seqs[seq_idx]]
                target_graphs = [move_graph_to_device(g, device) for g in target_seqs[seq_idx]]
                fields = field_targets[seq_idx].to(device)

                seq_loss = 0.0
                curr_seq = list(input_graphs)

                # For Verlet: track previous edge density
                use_verlet = model.use_verlet
                prev_edge_density = None

                for step in range(min(rollout, len(target_graphs))):
                    next_field = fields[step]
                    target_graph = target_graphs[step]

                    last_density = curr_seq[-1].edge_density
                    target_density = target_graph.edge_density

                    # Model returns full new density
                    new_density_pred = model(
                        curr_seq, next_field, skip_factor=k,
                        prev_edge_density=prev_edge_density
                    )

                    # Convert to delta for loss
                    delta_pred = new_density_pred - last_density
                    delta_target = target_density - last_density

                    loss, loss_dict = criterion(
                        delta_pred, delta_target,
                        curr_seq[-1].edge_index, last_density,
                        S=S, n_basis=curr_seq[-1].num_nodes, skip_factor=k
                    )

                    seq_loss += loss.item()

                    if step < rollout - 1:
                        # Track previous density for Verlet
                        if use_verlet:
                            prev_edge_density = last_density.clone()

                        new_graph = curr_seq[-1].clone()
                        new_graph.edge_density = new_density_pred
                        curr_seq = curr_seq[1:] + [new_graph]

                batch_loss += seq_loss / rollout

            total_loss += batch_loss / len(input_seqs)
            n_batches += 1

    return total_loss / max(1, n_batches), {'k': k}


def train():
    """Main training function."""
    print("=" * 70)
    print("DensityGraphNet Training with Skip-Step Propagation")
    print("=" * 70)

    # Load overlap matrix
    try:
        S = torch.tensor(np.load(CFG['overlap_file']), dtype=torch.float64)
    except Exception:
        print("Warning: Could not load overlap, using identity")
        S = torch.eye(CFG.get('n_basis', 4), dtype=torch.float64)
    S = S.to(CFG['device'])

    # Create dataset
    print(f"\nLoading dataset...")
    dataset = SkipStepGraphDataset(
        density_file=CFG['density_file'],
        field_file=CFG['field_file'],
        overlap_file=CFG['overlap_file'],
        positions_file=CFG.get('positions_file'),
        atomic_numbers_file=CFG.get('atomic_numbers_file'),
        seq_len=CFG.get('seq_len', 20),
        rollout_steps=CFG.get('rollout_steps', 5),
        skip_factor=CFG.get('initial_skip', 1),
        supervision_mode=CFG.get('supervision_mode', 'endpoints'),
        time_reversal_correction=CFG.get('time_reversal_correction', True)
    )

    dataset_info = dataset.get_graph_info()
    print(f"Dataset: {len(dataset)} samples, {dataset_info['n_nodes']} nodes, "
          f"{dataset_info['n_edges']} edges")

    # Create model
    print(f"\nCreating model...")
    model_config = {
        'node_dim': dataset_info['node_feature_dim'] + 1,
        'edge_dim': dataset_info['edge_feature_dim'],
        'hidden_dim': CFG.get('hidden_dim', 128),
        'n_gnn_layers': CFG.get('n_gnn_layers', 3),
        'n_lstm_layers': CFG.get('n_lstm_layers', 2),
        'n_attention_heads': CFG.get('n_attention_heads', 4),
        'dropout': CFG.get('dropout', 0.1),
        'use_skip_conditioning': CFG.get('use_skip_conditioning', True),
        'max_skip_factor': CFG.get('final_skip', 20),
        'k_embedding_dim': CFG.get('k_embedding_dim', 32),
        'use_verlet': CFG.get('use_verlet', False)
    }

    model = DensityGraphNet(model_config).to(CFG['device'])
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Skip conditioning: {model.use_skip_conditioning}")
    print(f"Integration: {'Verlet (2nd order)' if model.use_verlet else 'Euler (1st order)'}")

    # Load pretrained if specified
    if CFG.get('pretrained_path') and os.path.exists(CFG['pretrained_path']):
        print(f"Loading pretrained weights from {CFG['pretrained_path']}")
        checkpoint = torch.load(CFG['pretrained_path'], map_location=CFG['device'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.get('learning_rate', 1e-3),
        weight_decay=CFG.get('weight_decay', 1e-4)
    )

    # Curriculum scheduler
    use_curriculum = CFG.get('use_curriculum', True)
    if use_curriculum:
        curriculum = SkipStepCurriculum(
            initial_k=CFG.get('initial_skip', 1),
            final_k=CFG.get('final_skip', 20),
            warmup_epochs=CFG.get('curriculum_warmup', 20),
            total_epochs=CFG.get('epochs', 200),
            schedule=CFG.get('curriculum_schedule', 'exponential')
        )
        print(f"Curriculum: k={curriculum.initial_k}→{curriculum.final_k}, "
              f"schedule={CFG.get('curriculum_schedule', 'exponential')}")
    else:
        curriculum = None
        print(f"No curriculum, fixed k={CFG.get('initial_skip', 1)}")

    # Loss function
    criterion = SkipStepGraphLoss(
        lambda_trace=CFG.get('lambda_trace', 1e-4),
        lambda_idem=CFG.get('lambda_idem', 1e-5),
        lambda_smooth=CFG.get('lambda_smooth', 1e-4),
        lambda_phase=CFG.get('lambda_phase', 1e-5),
        n_electrons_alpha=CFG.get('n_alpha', 1.0),
        n_electrons_beta=CFG.get('n_beta', 0.0),
        adaptive_scaling=CFG.get('adaptive_scaling', True)
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    # Training loop
    epochs = CFG.get('epochs', 200)
    best_val_loss = float('inf')
    history = []
    rollout = CFG.get('rollout_steps', 5)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        start_time = time.time()

        # Get current k
        if curriculum:
            k = curriculum.step(epoch)
        else:
            k = CFG.get('initial_skip', 1)

        # Train
        train_loss, train_metrics = train_epoch(
            model, dataset, optimizer, criterion, k,
            CFG['device'], S,
            batch_size=CFG.get('batch_size', 8),
            rollout=rollout,
            noise_scale=CFG.get('noise_scale', 1e-3)
        )

        # Validate
        val_loss, val_metrics = validate(
            model, dataset, criterion, k,
            CFG['device'], S,
            batch_size=CFG.get('batch_size', 8),
            rollout=rollout
        )

        lr_scheduler.step(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CFG.get('model_save_path', 'gnn_skip_step.pt')
            torch.save({
                'epoch': epoch,
                'k': k,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)

        elapsed = time.time() - start_time

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            dt_eff = k * CFG.get('dt_fine', 0.4)
            print(f"Epoch {epoch+1:4d} | k={k:2d} (Δt={dt_eff:.1f}) | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | {elapsed:.1f}s")

        history.append({
            'epoch': epoch, 'k': k,
            'train_loss': train_loss, 'val_loss': val_loss
        })

    print("-" * 70)
    print(f"Training complete. Best: {best_val_loss:.6f}")
    print(f"Model saved to: {CFG.get('model_save_path', 'gnn_skip_step.pt')}")

    # Save history
    with open(CFG.get('history_path', 'gnn_skip_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model


def evaluate_speedup():
    """Evaluate speedup vs accuracy trade-off."""
    print("=" * 70)
    print("Evaluating Skip-Step Speedup")
    print("=" * 70)

    # Load model
    checkpoint = torch.load(CFG['model_save_path'], map_location=CFG['device'])
    model_config = checkpoint['config']
    model = DensityGraphNet(model_config).to(CFG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    dataset = SkipStepGraphDataset(
        density_file=CFG['density_file'],
        field_file=CFG['field_file'],
        overlap_file=CFG['overlap_file'],
        seq_len=CFG.get('seq_len', 20),
        rollout_steps=1
    )

    # Test different k values
    k_values = [1, 2, 5, 10, 15, 20]
    results = {'k': [], 'error': [], 'speedup': []}

    print(f"\n{'k':>4} {'Error':>12} {'Speedup':>10}")
    print("-" * 30)

    for k in k_values:
        if k > len(dataset.rho) // (dataset.seq_len + 1):
            continue

        dataset.set_skip_factor(k)
        errors = []

        with torch.no_grad():
            for i in range(min(50, len(dataset))):
                input_graphs, fields, target_graphs, _ = dataset[i]
                input_graphs = [move_graph_to_device(g, CFG['device']) for g in input_graphs]
                target_graphs = [move_graph_to_device(g, CFG['device']) for g in target_graphs]
                field = fields[0].to(CFG['device'])

                # Predict
                delta_pred = model(input_graphs, field, skip_factor=k)

                # Compare to target
                last_density = input_graphs[-1].edge_density
                target_density = target_graphs[0].edge_density
                delta_target = target_density - last_density

                error = torch.norm(delta_pred - delta_target) / (torch.norm(delta_target) + 1e-8)
                errors.append(error.item())

        avg_error = np.mean(errors)
        speedup = k

        results['k'].append(k)
        results['error'].append(avg_error)
        results['speedup'].append(speedup)

        print(f"{k:4d} {avg_error:12.6f} {speedup:10.1f}x")

    # Save results
    with open('speedup_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to speedup_results.json")
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

    flat['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return flat


def create_default_config() -> Dict:
    """Create default configuration."""
    return {
        "io": {
            "density_file": "test_train/densities/density_series.npy",
            "overlap_file": "test_train/data/h2_plus_rttddft_overlap.npy",
            "field_file": "test_train/data/field.dat",
            "positions_file": None,
            "atomic_numbers_file": None,
            "model_save_path": "gnn_skip_step.pt",
            "history_path": "gnn_skip_history.json",
            "pretrained_path": None
        },
        "system": {
            "n_basis": 4,
            "n_alpha": 1.0,
            "n_beta": 0.0,
            "dt_fine": 0.4
        },
        "model": {
            "seq_len": 20,
            "hidden_dim": 128,
            "n_gnn_layers": 3,
            "n_lstm_layers": 2,
            "n_attention_heads": 4,
            "dropout": 0.1,
            "use_skip_conditioning": True,
            "k_embedding_dim": 32
        },
        "skip_step": {
            "initial_skip": 1,
            "final_skip": 20,
            "supervision_mode": "endpoints"
        },
        "curriculum": {
            "use_curriculum": True,
            "curriculum_warmup": 20,
            "curriculum_schedule": "exponential"
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 200,
            "rollout_steps": 5,
            "noise_scale": 1e-3
        },
        "physics": {
            "lambda_trace": 1e-4,
            "lambda_idem": 1e-5,
            "lambda_smooth": 1e-4,
            "lambda_phase": 1e-5,
            "adaptive_scaling": True,
            "time_reversal_correction": True
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN with skip-step")
    parser.add_argument('--config', type=str, default='inputs/train_graph_skip.json')
    parser.add_argument('--create-config', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--verlet', action='store_true',
                        help='Use Verlet (second-order) integration instead of Euler')
    args = parser.parse_args()

    if args.create_config:
        config = create_default_config()
        config_path = Path('inputs/train_graph_skip.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created config at {config_path}")
    else:
        if not os.path.exists(args.config):
            print(f"Config not found, creating default...")
            config = create_default_config()
            config_path = Path(args.config)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

        CFG = load_config(args.config, args)

        # CLI override for Verlet
        if args.verlet:
            CFG['use_verlet'] = True
            print("Using Verlet (second-order) integration")

        if args.evaluate:
            evaluate_speedup()
        else:
            train()
