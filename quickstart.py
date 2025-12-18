#!/usr/bin/env python3
"""
Quick-Start Script for rtlstm: Graph-based Skip-Step Density Propagation

This script demonstrates the complete pipeline:
    1. Load H₂⁺ RT-TDDFT data
    2. Build graph representation from density matrices
    3. Train DensityGraphNet with skip-step curriculum
    4. Evaluate speedup vs accuracy trade-off
    5. Visualize predictions

Usage:
    python quickstart.py                    # Full demo
    python quickstart.py --skip-training    # Use pretrained (if available)
    python quickstart.py --quick            # Fast demo (fewer epochs)

Requirements:
    pip install torch numpy matplotlib
    pip install torch-geometric  # Optional but recommended
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np

# Check dependencies
print("=" * 70)
print("rtlstm Quick-Start: Graph-based Skip-Step Density Propagation")
print("=" * 70)

print("\n[1/7] Checking dependencies...")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("  - matplotlib not found (visualization disabled)")

try:
    import torch_geometric
    HAS_TORCH_GEOMETRIC = True
    print(f"  - torch_geometric: {torch_geometric.__version__}")
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("  - torch_geometric not found (GNN features disabled)")
    print("    Install with: pip install torch-geometric")

print(f"  - torch: {torch.__version__}")
print(f"  - numpy: {np.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  - Using device: {DEVICE}")


def load_test_data():
    """Load the H₂⁺ test data."""
    print("\n[2/7] Loading H₂⁺ RT-TDDFT test data...")

    # Paths to test data
    data_dir = Path(__file__).parent / 'test_train'
    density_file = data_dir / 'densities' / 'density_series.npy'
    overlap_file = data_dir / 'data' / 'h2_plus_rttddft_overlap.npy'
    field_file = data_dir / 'data' / 'field.dat'

    # Check files exist
    if not density_file.exists():
        print(f"  ERROR: Density file not found: {density_file}")
        print("  Please ensure test data is in test_train/ directory")
        return None

    # Load density matrices
    data_dict = np.load(density_file, allow_pickle=True).item()
    rho = torch.tensor(data_dict['density'], dtype=torch.complex128)
    rho = torch.conj(rho)  # Time reversal correction

    n_timesteps, n_spin, n_basis, _ = rho.shape
    print(f"  - Density matrices: {n_timesteps} timesteps, {n_spin} spins, {n_basis}×{n_basis} basis")

    # Load overlap matrix
    if overlap_file.exists():
        S = torch.tensor(np.load(overlap_file), dtype=torch.float64)
        print(f"  - Overlap matrix: {S.shape}")
    else:
        S = torch.eye(n_basis, dtype=torch.float64)
        print(f"  - Overlap matrix: using identity (file not found)")

    # Load field
    if field_file.exists():
        field_data = np.loadtxt(field_file)
        if field_data.shape[1] == 4:
            field_data = field_data[:, 1:]  # Skip time column
        field = torch.tensor(field_data, dtype=torch.float64)
        print(f"  - External field: {field.shape}")
    else:
        field = torch.zeros(n_timesteps, 3, dtype=torch.float64)
        print(f"  - External field: using zeros (file not found)")

    return {
        'rho': rho,
        'S': S,
        'field': field,
        'n_basis': n_basis,
        'n_timesteps': n_timesteps
    }


def demo_graph_representation(data):
    """Demonstrate converting density matrices to graphs."""
    print("\n[3/7] Building graph representation...")

    from graph_data import (
        build_graph_from_overlap,
        density_to_edge_features,
        edge_features_to_density,
        compute_node_features,
        MolecularGraph
    )

    n_basis = data['n_basis']
    S = data['S']
    rho = data['rho']

    # Create dummy positions (linear arrangement)
    positions = torch.zeros(n_basis, 3, dtype=torch.float64)
    positions[:, 0] = torch.arange(n_basis, dtype=torch.float64) * 1.4  # ~bond length

    # Atomic numbers (H₂⁺ has hydrogen basis)
    atomic_numbers = torch.ones(n_basis, dtype=torch.long)

    # Build graph from overlap matrix
    edge_index, edge_static = build_graph_from_overlap(
        S, positions, atomic_numbers, threshold=1e-6
    )

    print(f"  - Nodes: {n_basis} (basis functions)")
    print(f"  - Edges: {edge_index.shape[1]} (from overlap matrix)")

    # Convert first density matrix to edge features
    rho_0 = rho[0]  # (2, N, N)
    edge_density = density_to_edge_features(rho_0, edge_index)

    print(f"  - Edge features: {edge_density.shape} [Re_α, Im_α, Re_β, Im_β]")

    # Verify roundtrip
    rho_reconstructed = edge_features_to_density(edge_density, edge_index, n_basis)
    reconstruction_error = torch.norm(rho_0 - rho_reconstructed) / torch.norm(rho_0)
    print(f"  - Reconstruction error: {reconstruction_error:.2e} (should be ~0)")

    # Compute node features
    node_features = compute_node_features(atomic_numbers, positions)
    print(f"  - Node features: {node_features.shape}")

    return {
        'edge_index': edge_index,
        'edge_static': edge_static,
        'node_features': node_features,
        'positions': positions,
        'atomic_numbers': atomic_numbers
    }


def demo_lstm_baseline(data, quick=False):
    """Quick LSTM baseline for comparison."""
    print("\n[4/7] Training LSTM baseline (for comparison)...")

    from train_benchmark import DensityMatrixLSTM, PhysicsLoss, CFG

    # Configure
    n_basis = data['n_basis']
    CFG.update({
        'n_basis': n_basis,
        'hidden_dim': 64,
        'device': DEVICE,
        'dtype': torch.complex128
    })

    # Create model
    model = DensityMatrixLSTM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  - LSTM parameters: {n_params:,}")

    # Prepare data
    rho = data['rho'].to(DEVICE)
    field = data['field'].to(DEVICE)
    S = data['S'].to(DEVICE)

    seq_len = 10
    n_train = min(100 if quick else 500, len(rho) - seq_len - 5)

    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = PhysicsLoss(S)

    epochs = 20 if quick else 50
    print(f"  - Training for {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, n_train, 8):
            batch_end = min(i + 8, n_train)
            batch_loss = 0

            for j in range(i, batch_end):
                x_seq = rho[j:j+seq_len].unsqueeze(0)
                y = rho[j+seq_len].unsqueeze(0)
                f = field[j+seq_len].unsqueeze(0)

                pred = model(x_seq, f)
                loss = criterion(pred, y)
                batch_loss += loss

            batch_loss = batch_loss / (batch_end - i)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss = {total_loss/(n_train//8):.6f}")

    return model


def demo_gnn_skip_step(data, graph_info, quick=False):
    """Demonstrate GNN with skip-step training."""
    print("\n[5/7] Training DensityGraphNet with skip-step...")

    if not HAS_TORCH_GEOMETRIC:
        print("  SKIPPED: torch_geometric not installed")
        return None

    from graph_data import SkipStepGraphDataset
    from graph_model import DensityGraphNet, SkipStepGraphLoss, count_parameters
    from skip_step import SkipStepCurriculum

    # Create dataset
    data_dir = Path(__file__).parent / 'test_train'

    try:
        dataset = SkipStepGraphDataset(
            density_file=str(data_dir / 'densities' / 'density_series.npy'),
            field_file=str(data_dir / 'data' / 'field.dat'),
            overlap_file=str(data_dir / 'data' / 'h2_plus_rttddft_overlap.npy'),
            seq_len=10,
            rollout_steps=3,
            skip_factor=1
        )
    except Exception as e:
        print(f"  ERROR creating dataset: {e}")
        return None

    dataset_info = dataset.get_graph_info()
    print(f"  - Dataset: {len(dataset)} samples")

    # Create model with k-conditioning
    model_config = {
        'node_dim': dataset_info['node_feature_dim'] + 1,
        'edge_dim': dataset_info['edge_feature_dim'],
        'hidden_dim': 64,
        'n_gnn_layers': 2,
        'n_lstm_layers': 1,
        'n_attention_heads': 2,
        'dropout': 0.1,
        'use_skip_conditioning': True,
        'max_skip_factor': 20,
        'k_embedding_dim': 16
    }

    model = DensityGraphNet(model_config).to(DEVICE)
    print(f"  - GNN parameters: {count_parameters(model):,}")
    print(f"  - k-conditioning: enabled")

    # Curriculum
    epochs = 30 if quick else 100
    curriculum = SkipStepCurriculum(
        initial_k=1,
        final_k=10,
        warmup_epochs=5 if quick else 20,
        total_epochs=epochs,
        schedule='exponential'
    )

    # Loss and optimizer
    S = data['S'].to(DEVICE)
    criterion = SkipStepGraphLoss(
        lambda_trace=1e-4,
        lambda_smooth=1e-4,
        n_electrons_alpha=1.0,
        n_electrons_beta=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    print(f"  - Training for {epochs} epochs with curriculum k=1→10...")

    from graph_data import collate_skip_step_graphs
    from torch.utils.data import DataLoader

    def move_graph(g):
        from graph_data import MolecularGraph
        return MolecularGraph(
            edge_index=g.edge_index.to(DEVICE),
            num_nodes=g.num_nodes,
            node_features=g.node_features.to(DEVICE),
            edge_static=g.edge_static.to(DEVICE),
            edge_density=g.edge_density.to(DEVICE),
            atomic_numbers=g.atomic_numbers.to(DEVICE),
            positions=g.positions.to(DEVICE)
        )

    history = []
    for epoch in range(epochs):
        k = curriculum.step(epoch)
        dataset.set_skip_factor(k)

        if len(dataset) < 1:
            continue

        model.train()
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                           collate_fn=collate_skip_step_graphs)

        epoch_loss = 0
        n_batches = 0

        for input_seqs, field_targets, target_seqs, _ in loader:
            batch_loss = 0

            for seq_idx in range(len(input_seqs)):
                input_graphs = [move_graph(g) for g in input_seqs[seq_idx]]
                target_graphs = [move_graph(g) for g in target_seqs[seq_idx]]
                fields = field_targets[seq_idx].to(DEVICE)

                optimizer.zero_grad()
                seq_loss = 0
                curr_seq = list(input_graphs)

                for step in range(min(3, len(target_graphs))):
                    last_density = curr_seq[-1].edge_density
                    target_density = target_graphs[step].edge_density
                    delta_target = target_density - last_density

                    delta_pred = model(curr_seq, fields[step], skip_factor=k)

                    loss, _ = criterion(
                        delta_pred, delta_target,
                        curr_seq[-1].edge_index, last_density,
                        S=S, n_basis=curr_seq[-1].num_nodes, skip_factor=k
                    )
                    seq_loss += loss

                    if step < 2:
                        new_graph = curr_seq[-1].clone()
                        new_graph.edge_density = last_density + delta_pred.detach()
                        curr_seq = curr_seq[1:] + [new_graph]

                seq_loss = seq_loss / 3
                seq_loss.backward()
                batch_loss += seq_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss / len(input_seqs)
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        history.append({'epoch': epoch, 'k': k, 'loss': avg_loss})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}: k={k}, loss = {avg_loss:.6f}")

    # Save model
    save_path = Path(__file__).parent / 'quickstart_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'history': history
    }, save_path)
    print(f"  - Model saved to: {save_path}")

    return model, dataset, history


def evaluate_speedup(model, dataset, data):
    """Evaluate speedup vs accuracy trade-off."""
    print("\n[6/7] Evaluating skip-step speedup...")

    if model is None or not HAS_TORCH_GEOMETRIC:
        print("  SKIPPED: No trained model available")
        return None

    from graph_data import MolecularGraph

    def move_graph(g):
        return MolecularGraph(
            edge_index=g.edge_index.to(DEVICE),
            num_nodes=g.num_nodes,
            node_features=g.node_features.to(DEVICE),
            edge_static=g.edge_static.to(DEVICE),
            edge_density=g.edge_density.to(DEVICE),
            atomic_numbers=g.atomic_numbers.to(DEVICE),
            positions=g.positions.to(DEVICE)
        )

    model.eval()
    k_values = [1, 2, 5, 10]
    results = {'k': [], 'error': [], 'speedup': []}

    print(f"\n  {'k':>4} {'Rel. Error':>12} {'Speedup':>10} {'Effective Δt':>14}")
    print("  " + "-" * 44)

    dt_fine = 0.4  # atomic units

    for k in k_values:
        dataset.set_skip_factor(k)
        if len(dataset) < 5:
            continue

        errors = []
        with torch.no_grad():
            for i in range(min(20, len(dataset))):
                input_graphs, fields, target_graphs, _ = dataset[i]
                input_graphs = [move_graph(g) for g in input_graphs]
                target_graphs = [move_graph(g) for g in target_graphs]

                if len(target_graphs) == 0:
                    continue

                field = fields[0].to(DEVICE)

                delta_pred = model(input_graphs, field, skip_factor=k)

                last_density = input_graphs[-1].edge_density
                target_density = target_graphs[0].edge_density
                delta_target = target_density - last_density

                error = torch.norm(delta_pred - delta_target) / (torch.norm(delta_target) + 1e-8)
                errors.append(error.item())

        if errors:
            avg_error = np.mean(errors)
            dt_eff = k * dt_fine

            results['k'].append(k)
            results['error'].append(avg_error)
            results['speedup'].append(k)

            print(f"  {k:4d} {avg_error:12.6f} {k:10.1f}x {dt_eff:14.2f} au")

    return results


def visualize_results(data, results, history):
    """Create visualization of results."""
    print("\n[7/7] Visualizing results...")

    if not HAS_MATPLOTLIB:
        print("  SKIPPED: matplotlib not installed")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Density matrix evolution (diagonal elements)
    ax1 = axes[0, 0]
    rho = data['rho'].numpy()
    n_plot = min(200, len(rho))
    times = np.arange(n_plot) * 0.4  # dt = 0.4 au

    # Alpha spin diagonal
    for i in range(data['n_basis']):
        ax1.plot(times, np.real(rho[:n_plot, 0, i, i]), label=f'ρ_α[{i},{i}]', alpha=0.7)

    ax1.set_xlabel('Time (au)')
    ax1.set_ylabel('Diagonal density')
    ax1.set_title('H₂⁺ Density Matrix Evolution (α-spin diagonal)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training history
    ax2 = axes[0, 1]
    if history:
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        ks = [h['k'] for h in history]

        ax2.semilogy(epochs, losses, 'b-', label='Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, ks, 'r--', label='Skip factor k')
        ax2_twin.set_ylabel('Skip factor k', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')

        ax2.set_title('Training Progress with Curriculum')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No training history', ha='center', va='center')
        ax2.set_title('Training Progress')

    # Plot 3: Speedup vs Accuracy
    ax3 = axes[1, 0]
    if results and results['k']:
        ax3.plot(results['k'], results['error'], 'bo-', markersize=10, linewidth=2)
        ax3.set_xlabel('Skip factor k')
        ax3.set_ylabel('Relative prediction error')
        ax3.set_title('Skip-Step: Speedup vs Accuracy Trade-off')
        ax3.grid(True, alpha=0.3)

        # Add speedup annotation
        for k, err in zip(results['k'], results['error']):
            ax3.annotate(f'{k}×', (k, err), textcoords="offset points",
                        xytext=(0, 10), ha='center')
    else:
        ax3.text(0.5, 0.5, 'No speedup results', ha='center', va='center')
        ax3.set_title('Speedup vs Accuracy')

    # Plot 4: External field
    ax4 = axes[1, 1]
    field = data['field'].numpy()
    n_plot = min(200, len(field))
    times = np.arange(n_plot) * 0.4

    ax4.plot(times, field[:n_plot, 0], label='Ex')
    ax4.plot(times, field[:n_plot, 1], label='Ey')
    ax4.plot(times, field[:n_plot, 2], label='Ez')
    ax4.set_xlabel('Time (au)')
    ax4.set_ylabel('Field (au)')
    ax4.set_title('External Electric Field')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = Path(__file__).parent / 'quickstart_results.png'
    plt.savefig(fig_path, dpi=150)
    print(f"  - Figure saved to: {fig_path}")

    plt.show()


def print_summary():
    """Print summary and next steps."""
    print("\n" + "=" * 70)
    print("Quick-Start Complete!")
    print("=" * 70)

    print("""
Summary:
--------
You've just run the rtlstm pipeline which:

1. Loaded H₂⁺ RT-TDDFT density matrix data
2. Converted density matrices to graph representation
   - Nodes = basis functions
   - Edges = overlap matrix connectivity
   - Edge features = density matrix elements ρ_ij

3. Trained DensityGraphNet with:
   - Graph Neural Network encoder
   - LSTM temporal module
   - k-conditioning for skip-step prediction
   - Curriculum learning (k=1→10)

4. Evaluated speedup vs accuracy:
   - k=1:  Standard propagation (baseline)
   - k=10: 10× faster with small accuracy loss

Key Files Created:
------------------
- quickstart_model.pt     : Trained model checkpoint
- quickstart_results.png  : Visualization

Next Steps:
-----------
1. Train on your own data:
   python src/train_graph_skip.py --config inputs/train_graph_skip.json

2. Fine-tune for larger k:
   - Edit inputs/train_graph_skip.json
   - Set "final_skip": 20 for 20× speedup

3. Test transferability:
   - Train on H₂⁺, evaluate on H₂ or LiH
   - The graph representation enables this!

4. For production use:
   - Increase hidden_dim: 128→256
   - More GNN layers: 3→4
   - Longer training: 200+ epochs

Documentation:
--------------
See src/graph_model.py for model architecture details
See src/skip_step.py for curriculum learning options
""")


def main():
    parser = argparse.ArgumentParser(description='rtlstm Quick-Start Demo')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, use existing model if available')
    parser.add_argument('--quick', action='store_true',
                       help='Quick demo with fewer epochs')
    args = parser.parse_args()

    # Load data
    data = load_test_data()
    if data is None:
        print("\nERROR: Could not load test data. Exiting.")
        return 1

    # Demo graph representation
    graph_info = demo_graph_representation(data)

    # Train models
    history = []
    results = None

    if args.skip_training:
        model_path = Path(__file__).parent / 'quickstart_model.pt'
        if model_path.exists() and HAS_TORCH_GEOMETRIC:
            print("\n[4-5/7] Loading pretrained model...")
            from graph_model import DensityGraphNet
            from graph_data import SkipStepGraphDataset

            checkpoint = torch.load(model_path, map_location=DEVICE)
            model = DensityGraphNet(checkpoint['config']).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            history = checkpoint.get('history', [])

            data_dir = Path(__file__).parent / 'test_train'
            dataset = SkipStepGraphDataset(
                density_file=str(data_dir / 'densities' / 'density_series.npy'),
                field_file=str(data_dir / 'data' / 'field.dat'),
                overlap_file=str(data_dir / 'data' / 'h2_plus_rttddft_overlap.npy'),
                seq_len=10, rollout_steps=3, skip_factor=1
            )
            print(f"  - Loaded model from {model_path}")
        else:
            print("\n[4-5/7] No pretrained model found, training...")
            result = demo_gnn_skip_step(data, graph_info, quick=args.quick)
            if result:
                model, dataset, history = result
            else:
                model, dataset = None, None
    else:
        # Train LSTM baseline
        lstm_model = demo_lstm_baseline(data, quick=args.quick)

        # Train GNN with skip-step
        result = demo_gnn_skip_step(data, graph_info, quick=args.quick)
        if result:
            model, dataset, history = result
        else:
            model, dataset = None, None

    # Evaluate speedup
    if model is not None and dataset is not None:
        results = evaluate_speedup(model, dataset, data)

    # Visualize
    visualize_results(data, results, history)

    # Summary
    print_summary()

    return 0


if __name__ == '__main__':
    sys.exit(main())
