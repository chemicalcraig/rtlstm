"""
Unit tests for the graph data pipeline.

Tests the conversion of density matrices to graph representations
and verifies physics constraints are preserved.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pytest
from pathlib import Path

from graph_data import (
    build_graph_from_overlap,
    density_to_edge_features,
    edge_features_to_density,
    compute_node_features,
    enforce_hermiticity_edges,
    MolecularGraph,
    MolecularGraphDataset,
    create_test_graph,
    MAX_ATOMIC_NUM
)


# --- Test data paths ---
TEST_DATA_DIR = Path(__file__).parent.parent / "test_train"
DENSITY_FILE = TEST_DATA_DIR / "densities" / "density_series.npy"
OVERLAP_FILE = TEST_DATA_DIR / "data" / "h2_plus_rttddft_overlap.npy"
FIELD_FILE = TEST_DATA_DIR / "data" / "field.dat"


class TestGraphConstruction:
    """Tests for graph construction from overlap matrix."""

    def test_build_graph_identity_overlap(self):
        """Identity overlap should give fully connected graph (with threshold=0)."""
        N = 4
        S = torch.eye(N, dtype=torch.float64)
        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, edge_static = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        # Should have N self-loops
        assert edge_index.shape[1] == N
        # All edges should be diagonal
        assert torch.all(edge_index[0] == edge_index[1])

    def test_build_graph_full_overlap(self):
        """Full overlap matrix should give fully connected graph."""
        N = 4
        S = torch.ones(N, N, dtype=torch.float64)
        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, edge_static = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        # Should have N^2 edges (fully connected)
        assert edge_index.shape[1] == N * N

    def test_build_graph_threshold(self):
        """Thresholding should remove weak connections."""
        N = 4
        S = torch.eye(N, dtype=torch.float64)
        # Add small off-diagonal
        S[0, 1] = S[1, 0] = 0.01
        S[2, 3] = S[3, 2] = 0.1

        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, edge_static = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.05
        )

        # Should have: 4 self-loops + 2 edges (2,3) and (3,2)
        assert edge_index.shape[1] == 6

    def test_edge_static_features(self):
        """Edge static features should contain S_ij and distance."""
        N = 3
        S = torch.ones(N, N, dtype=torch.float64) * 0.5
        torch.diagonal(S)[:] = 1.0

        positions = torch.zeros(N, 3, dtype=torch.float64)
        positions[0, 0] = 0.0
        positions[1, 0] = 1.0
        positions[2, 0] = 3.0

        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, edge_static = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        # Check that edge features have 2 columns: [S_ij, distance]
        assert edge_static.shape[1] == 2

        # Find edge (0,1) and check distance = 1.0
        for k in range(edge_index.shape[1]):
            if edge_index[0, k] == 0 and edge_index[1, k] == 1:
                assert torch.isclose(edge_static[k, 1], torch.tensor(1.0))
            # Edge (0,2) should have distance 3.0
            if edge_index[0, k] == 0 and edge_index[1, k] == 2:
                assert torch.isclose(edge_static[k, 1], torch.tensor(3.0))


class TestDensityEdgeConversion:
    """Tests for density matrix <-> edge feature conversion."""

    def test_density_to_edge_shape(self):
        """Edge features should have correct shape."""
        N = 4
        graph = create_test_graph(n_basis=N)
        E = graph.edge_index.shape[1]

        assert graph.edge_density.shape == (E, 4)

    def test_roundtrip_conversion(self):
        """Converting density to edges and back should preserve values."""
        N = 4
        # Create Hermitian density matrix
        rho = torch.randn(2, N, N, dtype=torch.complex128)
        rho = 0.5 * (rho + rho.transpose(-1, -2).conj())

        # Build full graph (all edges)
        S = torch.ones(N, N, dtype=torch.float64)
        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, _ = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        # Convert to edges and back
        edge_density = density_to_edge_features(rho, edge_index)
        rho_reconstructed = edge_features_to_density(
            edge_density, edge_index, N, symmetrize=False
        )

        # Should be identical
        assert torch.allclose(rho, rho_reconstructed, atol=1e-10)

    def test_hermiticity_preserved(self):
        """Reconstructed matrix should be Hermitian."""
        N = 4
        rho = torch.randn(2, N, N, dtype=torch.complex128)
        rho = 0.5 * (rho + rho.transpose(-1, -2).conj())

        S = torch.ones(N, N, dtype=torch.float64)
        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, _ = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        edge_density = density_to_edge_features(rho, edge_index)
        rho_reconstructed = edge_features_to_density(
            edge_density, edge_index, N, symmetrize=True
        )

        # Check Hermitian: rho = rho^dagger
        assert torch.allclose(
            rho_reconstructed,
            rho_reconstructed.transpose(-1, -2).conj(),
            atol=1e-10
        )


class TestHermiticityEnforcement:
    """Tests for edge-level Hermiticity enforcement."""

    def test_diagonal_imaginary_zero(self):
        """Diagonal elements should have zero imaginary part."""
        N = 4
        graph = create_test_graph(n_basis=N)

        # Create random delta
        delta = torch.randn(graph.edge_index.shape[1], 4)
        delta_sym = enforce_hermiticity_edges(delta, graph.edge_index)

        # Find diagonal edges and check imaginary parts
        i, j = graph.edge_index
        for k in range(len(i)):
            if i[k] == j[k]:
                assert delta_sym[k, 1] == 0.0  # Im(alpha)
                assert delta_sym[k, 3] == 0.0  # Im(beta)

    def test_conjugate_symmetry(self):
        """Off-diagonal elements should satisfy conjugate symmetry."""
        N = 4
        S = torch.ones(N, N, dtype=torch.float64)
        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, _ = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        delta = torch.randn(edge_index.shape[1], 4)
        delta_sym = enforce_hermiticity_edges(delta, edge_index)

        # Build lookup
        i, j = edge_index
        edge_to_idx = {(i[k].item(), j[k].item()): k for k in range(len(i))}

        # Check pairs
        for k in range(len(i)):
            ii, jj = i[k].item(), j[k].item()
            if ii < jj and (jj, ii) in edge_to_idx:
                rev_k = edge_to_idx[(jj, ii)]
                # Real parts should be equal
                assert torch.isclose(delta_sym[k, 0], delta_sym[rev_k, 0])
                assert torch.isclose(delta_sym[k, 2], delta_sym[rev_k, 2])
                # Imaginary parts should be negated
                assert torch.isclose(delta_sym[k, 1], -delta_sym[rev_k, 1])
                assert torch.isclose(delta_sym[k, 3], -delta_sym[rev_k, 3])


class TestNodeFeatures:
    """Tests for node feature computation."""

    def test_one_hot_encoding(self):
        """One-hot encoding should have correct dimension."""
        N = 4
        atomic_numbers = torch.tensor([1, 1, 3, 6], dtype=torch.long)  # H, H, Li, C
        positions = torch.randn(N, 3)

        features = compute_node_features(atomic_numbers, positions, use_one_hot=True)

        # Should have MAX_ATOMIC_NUM + 3 (positions) dimensions
        assert features.shape == (N, MAX_ATOMIC_NUM + 3)

    def test_one_hot_values(self):
        """One-hot encoding should be correct."""
        N = 2
        atomic_numbers = torch.tensor([1, 6], dtype=torch.long)  # H, C
        positions = torch.zeros(N, 3)

        features = compute_node_features(atomic_numbers, positions, use_one_hot=True)

        # Check one-hot part (first MAX_ATOMIC_NUM dims)
        assert features[0, 1] == 1.0  # Hydrogen at index 1
        assert features[1, 6] == 1.0  # Carbon at index 6


class TestMolecularGraph:
    """Tests for MolecularGraph dataclass."""

    def test_create_test_graph(self):
        """Test graph creation utility."""
        graph = create_test_graph(n_basis=4)

        assert graph.num_nodes == 4
        assert graph.edge_index.shape[0] == 2
        assert graph.node_features.shape[0] == 4
        assert graph.edge_density.shape[1] == 4
        assert graph.edge_static.shape[1] == 2

    def test_clone(self):
        """Clone should create independent copy."""
        graph = create_test_graph(n_basis=4)
        clone = graph.clone()

        # Modify original
        graph.edge_density[0, 0] = 999.0

        # Clone should be unchanged
        assert clone.edge_density[0, 0] != 999.0


class TestMolecularGraphDataset:
    """Tests for the dataset class using real test data."""

    @pytest.fixture
    def dataset(self):
        """Create dataset from test data if available."""
        if not DENSITY_FILE.exists():
            pytest.skip(f"Test data not found: {DENSITY_FILE}")
        if not OVERLAP_FILE.exists():
            pytest.skip(f"Overlap file not found: {OVERLAP_FILE}")
        if not FIELD_FILE.exists():
            pytest.skip(f"Field file not found: {FIELD_FILE}")

        return MolecularGraphDataset(
            density_file=str(DENSITY_FILE),
            field_file=str(FIELD_FILE),
            overlap_file=str(OVERLAP_FILE),
            seq_len=5,
            rollout_steps=2
        )

    def test_dataset_creation(self, dataset):
        """Dataset should load without errors."""
        assert len(dataset) > 0

    def test_dataset_getitem(self, dataset):
        """Dataset should return correct structure."""
        input_graphs, field_targets, target_graphs = dataset[0]

        assert len(input_graphs) == 5  # seq_len
        assert len(target_graphs) == 2  # rollout_steps
        assert field_targets.shape == (2, 3)  # (rollout_steps, 3)

    def test_graph_structure_consistent(self, dataset):
        """All graphs should have same structure."""
        input_graphs, _, target_graphs = dataset[0]

        ref_edges = input_graphs[0].edge_index.shape[1]
        ref_nodes = input_graphs[0].num_nodes

        for g in input_graphs + target_graphs:
            assert g.edge_index.shape[1] == ref_edges
            assert g.num_nodes == ref_nodes

    def test_graph_info(self, dataset):
        """Graph info should return correct dimensions."""
        info = dataset.get_graph_info()

        assert 'n_nodes' in info
        assert 'n_edges' in info
        assert 'node_feature_dim' in info
        assert 'edge_feature_dim' in info
        assert info['edge_density_dim'] == 4


class TestPhysicsConstraints:
    """Tests for physics constraint preservation."""

    def test_trace_preservation(self):
        """Trace should be preserved through conversion."""
        N = 4
        rho = torch.randn(2, N, N, dtype=torch.complex128)
        rho = 0.5 * (rho + rho.transpose(-1, -2).conj())

        # Compute original trace
        trace_orig = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)

        # Convert to edges and back
        S = torch.ones(N, N, dtype=torch.float64)
        positions = torch.randn(N, 3)
        atomic_numbers = torch.ones(N, dtype=torch.long)

        edge_index, _ = build_graph_from_overlap(
            S, positions, atomic_numbers, threshold=0.0
        )

        edge_density = density_to_edge_features(rho, edge_index)
        rho_reconstructed = edge_features_to_density(
            edge_density, edge_index, N, symmetrize=True
        )

        trace_recon = torch.diagonal(rho_reconstructed, dim1=-2, dim2=-1).sum(dim=-1)

        assert torch.allclose(trace_orig, trace_recon, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
