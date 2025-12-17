"""
Unit tests for the graph neural network model.

Tests the DensityGraphNet architecture and its components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pytest
from pathlib import Path

# Check for torch_geometric
try:
    import torch_geometric
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

if HAS_TORCH_GEOMETRIC:
    from graph_model import (
        MLP,
        EdgeGatedGCN,
        DensityMessagePassing,
        EdgeAttentionPooling,
        GraphLSTM,
        DensityDecoder,
        DensityGraphNet,
        GraphPhysicsLoss,
        count_parameters,
        create_model_from_dataset_info
    )
    from graph_data import create_test_graph, MolecularGraph


# Skip all tests if torch_geometric not installed
pytestmark = pytest.mark.skipif(
    not HAS_TORCH_GEOMETRIC,
    reason="torch_geometric not installed"
)


class TestMLP:
    """Tests for MLP building block."""

    def test_mlp_shapes(self):
        """MLP should transform to correct output dimension."""
        mlp = MLP(input_dim=16, hidden_dim=32, output_dim=8, num_layers=3)
        x = torch.randn(10, 16)
        y = mlp(x)
        assert y.shape == (10, 8)

    def test_mlp_single_layer(self):
        """Single layer MLP should work."""
        mlp = MLP(input_dim=16, hidden_dim=32, output_dim=8, num_layers=1)
        x = torch.randn(10, 16)
        y = mlp(x)
        assert y.shape == (10, 8)


class TestEdgeGatedGCN:
    """Tests for edge-gated GCN layer."""

    def test_gcn_shapes(self):
        """GCN should preserve node/edge dimensions."""
        node_dim = 16
        edge_dim = 8
        hidden_dim = 32
        n_nodes = 10
        n_edges = 30

        gcn = EdgeGatedGCN(node_dim, edge_dim, hidden_dim)

        x = torch.randn(n_nodes, node_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, edge_dim)

        x_new, edge_new = gcn(x, edge_index, edge_attr)

        assert x_new.shape == (n_nodes, node_dim)
        assert edge_new.shape == (n_edges, edge_dim)

    def test_gcn_gradient_flow(self):
        """Gradients should flow through GCN."""
        gcn = EdgeGatedGCN(16, 8, 32)

        x = torch.randn(10, 16, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 8, requires_grad=True)

        x_new, edge_new = gcn(x, edge_index, edge_attr)
        loss = x_new.sum() + edge_new.sum()
        loss.backward()

        assert x.grad is not None
        assert edge_attr.grad is not None


class TestDensityMessagePassing:
    """Tests for full message passing stack."""

    def test_message_passing_shapes(self):
        """Message passing should preserve dimensions."""
        node_dim = 24
        edge_dim = 6
        hidden_dim = 64
        n_nodes = 4
        n_edges = 16

        mp = DensityMessagePassing(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )

        x = torch.randn(n_nodes, node_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, edge_dim)

        x_out, edge_out = mp(x, edge_index, edge_attr)

        assert x_out.shape == (n_nodes, node_dim)
        assert edge_out.shape == (n_edges, edge_dim)


class TestGraphLSTM:
    """Tests for temporal module."""

    def test_graph_lstm_output_shape(self):
        """GraphLSTM should produce correct context shape."""
        edge_dim = 6
        hidden_dim = 64
        seq_len = 5
        n_edges = 16

        lstm = GraphLSTM(
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=2,
            use_attention_pooling=False  # Use simpler pooling for test
        )

        # Create sequence of edge features
        edge_attr_seq = [torch.randn(n_edges, edge_dim) for _ in range(seq_len)]
        edge_batch_seq = [torch.zeros(n_edges, dtype=torch.long) for _ in range(seq_len)]

        context = lstm(edge_attr_seq, edge_batch_seq)

        assert context.shape == (1, hidden_dim)  # 1 graph in batch

    def test_graph_lstm_batched(self):
        """GraphLSTM should handle multiple graphs."""
        edge_dim = 6
        hidden_dim = 64
        seq_len = 3
        batch_size = 4
        edges_per_graph = 10

        lstm = GraphLSTM(
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            use_attention_pooling=False
        )

        # Create batched sequence
        edge_attr_seq = []
        edge_batch_seq = []

        for t in range(seq_len):
            edge_attr = torch.randn(batch_size * edges_per_graph, edge_dim)
            edge_batch = torch.repeat_interleave(
                torch.arange(batch_size), edges_per_graph
            )
            edge_attr_seq.append(edge_attr)
            edge_batch_seq.append(edge_batch)

        context = lstm(edge_attr_seq, edge_batch_seq)

        assert context.shape == (batch_size, hidden_dim)


class TestDensityDecoder:
    """Tests for decoder module."""

    def test_decoder_output_shape(self):
        """Decoder should output 4 values per edge."""
        node_dim = 24
        edge_dim = 6
        hidden_dim = 64
        n_nodes = 4
        n_edges = 16

        decoder = DensityDecoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim
        )

        x = torch.randn(n_nodes, node_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, edge_dim)
        context = torch.randn(1, hidden_dim)
        edge_batch = torch.zeros(n_edges, dtype=torch.long)

        delta = decoder(x, edge_index, edge_attr, context, edge_batch)

        # 4 outputs: [Δre_α, Δim_α, Δre_β, Δim_β]
        assert delta.shape == (n_edges, 4)


class TestDensityGraphNet:
    """Tests for the full model."""

    @pytest.fixture
    def model_config(self):
        return {
            'node_dim': 24,
            'edge_dim': 6,
            'hidden_dim': 32,
            'n_gnn_layers': 2,
            'n_lstm_layers': 1,
            'n_attention_heads': 2,
            'dropout': 0.0
        }

    @pytest.fixture
    def model(self, model_config):
        return DensityGraphNet(model_config)

    @pytest.fixture
    def graph_seq(self):
        """Create sequence of test graphs."""
        seq_len = 5
        graphs = [create_test_graph(n_basis=4) for _ in range(seq_len)]

        # Adjust node features to have dimension 23 (before field coupling)
        for g in graphs:
            pad_size = 23 - g.node_features.shape[-1]
            if pad_size > 0:
                g.node_features = torch.cat([
                    g.node_features,
                    torch.zeros(g.num_nodes, pad_size)
                ], dim=-1)

        return graphs

    def test_forward_edge_output(self, model, graph_seq):
        """Forward pass should return edge predictions."""
        next_field = torch.randn(3)
        delta = model(graph_seq, next_field, return_matrix=False)

        n_edges = graph_seq[-1].edge_index.shape[1]
        assert delta.shape == (n_edges, 4)

    def test_forward_matrix_output(self, model, graph_seq):
        """Forward pass can return full matrix."""
        next_field = torch.randn(3)
        rho = model(graph_seq, next_field, return_matrix=True)

        n_basis = graph_seq[-1].num_nodes
        assert rho.shape == (2, n_basis, n_basis)
        assert rho.dtype == torch.complex128

    def test_hermiticity_enforced(self, model, graph_seq):
        """Output matrix should be Hermitian."""
        next_field = torch.randn(3)
        rho = model(graph_seq, next_field, return_matrix=True)

        # Check ρ = ρ†
        rho_dagger = rho.transpose(-1, -2).conj()
        assert torch.allclose(rho, rho_dagger, atol=1e-6)

    def test_predict_step(self, model, graph_seq):
        """predict_step should return new graph."""
        next_field = torch.randn(3)
        new_graph = model.predict_step(graph_seq, next_field)

        assert isinstance(new_graph, MolecularGraph)
        assert new_graph.num_nodes == graph_seq[-1].num_nodes
        assert new_graph.edge_index.shape == graph_seq[-1].edge_index.shape

    def test_gradient_flow(self, model, graph_seq):
        """Gradients should flow through model."""
        next_field = torch.randn(3)

        # Make edge_density require gradients
        for g in graph_seq:
            g.edge_density = g.edge_density.clone().requires_grad_(True)

        delta = model(graph_seq, next_field)
        loss = delta.sum()
        loss.backward()

        # Check gradients exist for model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestGraphPhysicsLoss:
    """Tests for physics-aware loss function."""

    def test_mse_loss(self):
        """MSE component should compute correctly."""
        loss_fn = GraphPhysicsLoss(lambda_trace=0.0, lambda_idem=0.0)

        delta_pred = torch.randn(16, 4)
        delta_target = delta_pred.clone()  # Same = zero loss

        loss, loss_dict = loss_fn(
            delta_pred, delta_target,
            edge_index=torch.randint(0, 4, (2, 16)),
            edge_density_current=torch.randn(16, 4)
        )

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
        assert 'mse' in loss_dict

    def test_trace_constraint(self):
        """Trace constraint should penalize deviations."""
        loss_fn = GraphPhysicsLoss(
            lambda_trace=1.0,
            lambda_idem=0.0,
            n_electrons_alpha=1.0,
            n_electrons_beta=0.0
        )

        n_basis = 4
        # Create edge features that give wrong trace
        edge_index = torch.tensor([[i, j] for i in range(n_basis) for j in range(n_basis)]).T
        edge_density = torch.zeros(n_basis**2, 4)
        delta_pred = torch.zeros(n_basis**2, 4)
        delta_target = torch.zeros(n_basis**2, 4)

        loss, loss_dict = loss_fn(
            delta_pred, delta_target,
            edge_index=edge_index,
            edge_density_current=edge_density,
            S=torch.eye(n_basis),
            n_basis=n_basis
        )

        # With zero density, trace = 0 but target = 1, so trace loss > 0
        assert loss > 0
        assert 'trace' in loss_dict


class TestModelCreation:
    """Tests for model creation utilities."""

    def test_create_from_dataset_info(self):
        """Should create model with correct dimensions."""
        dataset_info = {
            'node_feature_dim': 23,
            'edge_feature_dim': 6,
            'n_nodes': 4,
            'n_edges': 16
        }

        model = create_model_from_dataset_info(
            dataset_info,
            hidden_dim=64,
            n_gnn_layers=2
        )

        assert isinstance(model, DensityGraphNet)
        assert model.hidden_dim == 64
        assert model.n_gnn_layers == 2

    def test_count_parameters(self):
        """Should count parameters correctly."""
        config = {
            'node_dim': 24,
            'edge_dim': 6,
            'hidden_dim': 32,
            'n_gnn_layers': 1,
            'n_lstm_layers': 1,
            'n_attention_heads': 2,
            'dropout': 0.0
        }
        model = DensityGraphNet(config)
        n_params = count_parameters(model)

        assert n_params > 0
        assert isinstance(n_params, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
