"""
Graph Neural Network model for density matrix propagation.

This module implements DensityGraphNet, a GNN-based architecture for
learning the time evolution of quantum density matrices. The model
operates on graph representations where edges carry density matrix
elements, enabling size-agnostic learning and transferability across
different molecular systems.

Architecture Overview:
    1. GNN Encoder: Message passing to learn spatial correlations
    2. Temporal Module: Graph-level LSTM for sequence modeling
    3. Decoder: Predict Δρ residuals on edges
    4. Symmetrization: Enforce Hermiticity constraint

Key Features:
    - Size-independent: Works with any number of basis functions
    - Residual learning: Predicts change Δρ, not absolute values
    - Physics-aware: Hermiticity enforced by construction
    - Transferable: Same model can process different molecules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math

# Import graph data utilities
from graph_data import (
    MolecularGraph,
    enforce_hermiticity_edges,
    edge_features_to_density,
    density_to_edge_features
)

# Conditional PyG import
try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import softmax as pyg_softmax
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    MessagePassing = nn.Module  # Fallback for type hints


# =============================================================================
# Building Blocks
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = 'silu',
        dropout: float = 0.0,
        layer_norm: bool = True
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:  # No activation/norm on last layer
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))

                if activation == 'silu':
                    layers.append(nn.SiLU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeGatedGCN(MessagePassing):
    """
    Edge-Gated Graph Convolutional Layer.

    Updates both node and edge features using gated message passing.
    Particularly suited for density matrix propagation where edge
    features (ρ_ij) are the primary quantities of interest.

    Message: m_ij = σ(gate_ij) * φ(x_i, x_j, e_ij)
    Node update: x_i' = ψ(x_i, Σ_j m_ij)
    Edge update: e_ij' = χ(e_ij, x_i', x_j')
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__(aggr='add', flow='source_to_target')

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Message network: combines source, target nodes and edge
        self.message_mlp = MLP(
            2 * node_dim + edge_dim,
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # Gate network: scalar gate for each message
        self.gate_mlp = MLP(
            2 * node_dim + edge_dim,
            hidden_dim,
            1,
            num_layers=2,
            dropout=0.0  # No dropout on gates
        )

        # Node update network
        self.node_mlp = MLP(
            node_dim + hidden_dim,
            hidden_dim,
            node_dim,
            num_layers=2,
            dropout=dropout
        )

        # Edge update network
        self.edge_mlp = MLP(
            edge_dim + 2 * node_dim,
            hidden_dim,
            edge_dim,
            num_layers=2,
            dropout=dropout
        )

        # Residual connections
        self.node_residual = nn.Linear(node_dim, node_dim) if node_dim != node_dim else nn.Identity()
        self.edge_residual = nn.Linear(edge_dim, edge_dim) if edge_dim != edge_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (N, node_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features

        Returns:
            x_new: (N, node_dim) updated node features
            edge_attr_new: (E, edge_dim) updated edge features
        """
        # Store for residual
        x_res = x
        edge_res = edge_attr

        # Message passing for node updates
        aggr_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update nodes
        x_new = self.node_mlp(torch.cat([x, aggr_msg], dim=-1))
        x_new = x_new + self.node_residual(x_res)

        # Update edges using new node features
        i, j = edge_index
        edge_input = torch.cat([edge_attr, x_new[i], x_new[j]], dim=-1)
        edge_attr_new = self.edge_mlp(edge_input)
        edge_attr_new = edge_attr_new + self.edge_residual(edge_res)

        return x_new, edge_attr_new

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute gated messages."""
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute message and gate
        msg = self.message_mlp(msg_input)
        gate = torch.sigmoid(self.gate_mlp(msg_input))

        return gate * msg


class DensityMessagePassing(nn.Module):
    """
    Stack of message passing layers for density matrix graphs.

    This module processes the spatial structure of the density matrix,
    learning correlations between different matrix elements through
    the graph structure defined by the overlap matrix.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers

        # Initial projections
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Stack of GNN layers
        self.gnn_layers = nn.ModuleList([
            EdgeGatedGCN(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projections
        self.node_decoder = nn.Linear(hidden_dim, node_dim)
        self.edge_decoder = nn.Linear(hidden_dim, edge_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process graph through GNN layers.

        Args:
            x: (N, node_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features

        Returns:
            x_out: (N, node_dim) processed node features
            edge_out: (E, edge_dim) processed edge features
        """
        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.gnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        # Decode
        x_out = self.node_decoder(x)
        edge_out = self.edge_decoder(edge_attr)

        return x_out, edge_out


# =============================================================================
# Temporal Module
# =============================================================================

class EdgeAttentionPooling(nn.Module):
    """
    Attention-based pooling of edge features to graph-level representation.

    Uses attention over edges weighted by importance for the prediction task.
    """

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.transform = nn.Linear(edge_dim, hidden_dim)

    def forward(
        self,
        edge_attr: torch.Tensor,
        edge_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool edge features to graph-level.

        Args:
            edge_attr: (E, edge_dim) edge features
            edge_batch: (E,) batch assignment for each edge

        Returns:
            graph_embed: (B, hidden_dim) graph embeddings
        """
        # Compute attention scores
        attn_scores = self.attention(edge_attr)  # (E, 1)

        # Softmax over edges within each graph
        attn_weights = pyg_softmax(attn_scores, edge_batch)  # (E, 1)

        # Transform and weight
        edge_transformed = self.transform(edge_attr)  # (E, hidden_dim)
        weighted = attn_weights * edge_transformed  # (E, hidden_dim)

        # Sum within each graph
        num_graphs = edge_batch.max().item() + 1
        graph_embed = torch.zeros(
            num_graphs, weighted.shape[-1],
            device=weighted.device, dtype=weighted.dtype
        )
        graph_embed.scatter_add_(0, edge_batch.unsqueeze(-1).expand_as(weighted), weighted)

        return graph_embed


class GraphLSTM(nn.Module):
    """
    Temporal module for processing sequences of graphs.

    Pools each graph to a fixed-size embedding, then processes
    the sequence with an LSTM to capture temporal dynamics.
    Includes self-attention over the sequence for long-range dependencies.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        super().__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_attention_pooling = use_attention_pooling

        # Graph pooling
        if use_attention_pooling and HAS_TORCH_GEOMETRIC:
            self.pooling = EdgeAttentionPooling(edge_dim, hidden_dim)
            lstm_input_dim = hidden_dim
        else:
            # Fallback to mean pooling with projection
            self.pooling = None
            self.pool_projection = nn.Linear(edge_dim, hidden_dim)
            lstm_input_dim = hidden_dim

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False
        )

        # Self-attention over sequence
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        edge_attr_seq: List[torch.Tensor],
        edge_batch_seq: List[torch.Tensor],
        field_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process sequence of graphs.

        Args:
            edge_attr_seq: List of (E_t, edge_dim) edge features per timestep
            edge_batch_seq: List of (E_t,) batch assignments per timestep
            field_seq: Optional (B, Seq, 3) field values (not used currently)

        Returns:
            context: (B, hidden_dim) temporal context vector
        """
        # Pool each graph to fixed-size representation
        pooled = []
        for edge_attr, edge_batch in zip(edge_attr_seq, edge_batch_seq):
            if self.pooling is not None:
                g_embed = self.pooling(edge_attr, edge_batch)
            else:
                # Simple mean pooling fallback
                num_graphs = edge_batch.max().item() + 1
                g_embed = torch.zeros(
                    num_graphs, self.edge_dim,
                    device=edge_attr.device, dtype=edge_attr.dtype
                )
                counts = torch.zeros(num_graphs, device=edge_attr.device)
                g_embed.scatter_add_(0, edge_batch.unsqueeze(-1).expand_as(edge_attr), edge_attr)
                counts.scatter_add_(0, edge_batch, torch.ones_like(edge_batch, dtype=edge_attr.dtype))
                g_embed = g_embed / counts.unsqueeze(-1).clamp(min=1)
                g_embed = self.pool_projection(g_embed)

            pooled.append(g_embed)

        # Stack into sequence tensor: (B, Seq, hidden_dim)
        seq_tensor = torch.stack(pooled, dim=1)

        # LSTM processing
        lstm_out, _ = self.lstm(seq_tensor)  # (B, Seq, hidden_dim)

        # Self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)

        # Residual + LayerNorm
        seq_out = self.layer_norm(lstm_out + attn_out)

        # Return last timestep context
        context = seq_out[:, -1, :]  # (B, hidden_dim)

        return context


# =============================================================================
# Decoder
# =============================================================================

class DensityDecoder(nn.Module):
    """
    Decoder for predicting density matrix changes.

    Takes the temporal context from GraphLSTM and predicts Δρ
    for each edge, maintaining the residual learning approach.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        context_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # Context projection to edge level
        self.context_projection = nn.Linear(context_dim, hidden_dim)

        # Delta prediction network
        # Input: [edge_attr, x_i, x_j, context]
        self.delta_mlp = MLP(
            edge_dim + 2 * node_dim + hidden_dim,
            hidden_dim,
            4,  # Output: [Δre_α, Δim_α, Δre_β, Δim_β]
            num_layers=num_layers,
            dropout=dropout
        )

        # Scale output to appropriate magnitude
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        context: torch.Tensor,
        edge_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict Δρ for each edge.

        Args:
            x: (N, node_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
            context: (B, context_dim) temporal context
            edge_batch: (E,) batch assignment for edges

        Returns:
            delta_rho: (E, 4) predicted density changes
        """
        i, j = edge_index

        # Expand context to edge level
        context_proj = self.context_projection(context)  # (B, hidden_dim)
        context_edge = context_proj[edge_batch]  # (E, hidden_dim)

        # Concatenate all features
        delta_input = torch.cat([
            edge_attr,
            x[i],
            x[j],
            context_edge
        ], dim=-1)

        # Predict delta
        delta_rho = self.delta_mlp(delta_input)  # (E, 4)

        # Scale output
        delta_rho = delta_rho * self.output_scale

        return delta_rho


# =============================================================================
# Main Model
# =============================================================================

class DensityGraphNet(nn.Module):
    """
    Graph Neural Network for density matrix time evolution.

    This model learns to propagate quantum density matrices by:
    1. Encoding the spatial structure through message passing
    2. Capturing temporal dynamics with LSTM
    3. Predicting residual changes Δρ
    4. Enforcing Hermiticity by construction

    The graph representation enables:
    - Size independence (works with any basis set size)
    - Transferability (train on H2+, infer on H2)
    - Physics-aware learning (respects matrix structure)
    """

    def __init__(self, config: Dict):
        """
        Initialize DensityGraphNet.

        Args:
            config: Dictionary with model configuration:
                - node_dim: Input node feature dimension
                - edge_dim: Input edge feature dimension (typically 6)
                - hidden_dim: Hidden layer dimension
                - n_gnn_layers: Number of GNN layers
                - n_lstm_layers: Number of LSTM layers
                - n_attention_heads: Number of attention heads
                - dropout: Dropout rate
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric required for DensityGraphNet. "
                "Install with: pip install torch-geometric"
            )

        # Store config
        self.config = config
        self.node_dim = config.get('node_dim', 23)  # MAX_ATOMIC_NUM + 3
        self.edge_dim = config.get('edge_dim', 6)   # [re_α, im_α, re_β, im_β, S, dist]
        self.hidden_dim = config.get('hidden_dim', 128)
        self.n_gnn_layers = config.get('n_gnn_layers', 3)
        self.n_lstm_layers = config.get('n_lstm_layers', 2)
        self.n_attention_heads = config.get('n_attention_heads', 4)
        self.dropout = config.get('dropout', 0.1)

        # GNN encoder
        self.gnn_encoder = DensityMessagePassing(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.n_gnn_layers,
            dropout=self.dropout
        )

        # Temporal module
        self.temporal = GraphLSTM(
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_lstm_layers=self.n_lstm_layers,
            num_attention_heads=self.n_attention_heads,
            dropout=self.dropout
        )

        # Decoder
        self.decoder = DensityDecoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            num_layers=3,
            dropout=self.dropout
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def encode_sequence(
        self,
        graph_seq: List[MolecularGraph],
        field_seq: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode a sequence of graphs through the GNN.

        Args:
            graph_seq: List of MolecularGraph objects
            field_seq: Optional (Seq, 3) field values

        Returns:
            Tuple of lists: (node_features, edge_features, edge_indices, edge_batches)
        """
        encoded_nodes = []
        encoded_edges = []
        edge_indices = []
        edge_batches = []

        for t, graph in enumerate(graph_seq):
            # Get field for this timestep
            if field_seq is not None and t < len(field_seq):
                field = field_seq[t]
                # Add field coupling to node features
                field_coupling = torch.einsum('j,ij->i', field.float(), graph.positions.float())
                node_x = torch.cat([
                    graph.node_features.float(),
                    field_coupling.unsqueeze(-1)
                ], dim=-1)
            else:
                # Pad with zero field coupling
                node_x = torch.cat([
                    graph.node_features.float(),
                    torch.zeros(graph.num_nodes, 1, device=graph.node_features.device)
                ], dim=-1)

            # Combine edge features: [density (4), static (2)]
            edge_attr = torch.cat([
                graph.edge_density.float(),
                graph.edge_static.float()
            ], dim=-1)

            # Encode through GNN
            node_out, edge_out = self.gnn_encoder(
                node_x, graph.edge_index, edge_attr
            )

            encoded_nodes.append(node_out)
            encoded_edges.append(edge_out)
            edge_indices.append(graph.edge_index)

            # Create batch assignment (all edges belong to graph 0 for single graph)
            edge_batch = torch.zeros(graph.edge_index.shape[1], dtype=torch.long,
                                    device=graph.edge_index.device)
            edge_batches.append(edge_batch)

        return encoded_nodes, encoded_edges, edge_indices, edge_batches

    def forward(
        self,
        graph_seq: List[MolecularGraph],
        next_field: torch.Tensor,
        return_matrix: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: predict next density state.

        Args:
            graph_seq: List of seq_len MolecularGraph objects (history)
            next_field: (3,) or (B, 3) field at next timestep
            return_matrix: If True, return full density matrix instead of edges

        Returns:
            If return_matrix:
                rho_pred: (2, N, N) complex density matrix
            Else:
                delta_rho: (E, 4) predicted edge changes
        """
        # Handle batched vs single field
        if next_field.dim() == 1:
            next_field = next_field.unsqueeze(0)  # (1, 3)

        # Build field sequence (use same field for all history steps for simplicity)
        # In practice, you might want to pass the actual historical fields
        field_seq = next_field.expand(len(graph_seq), -1)  # (Seq, 3)

        # Encode sequence through GNN
        encoded_nodes, encoded_edges, edge_indices, edge_batches = self.encode_sequence(
            graph_seq, field_seq
        )

        # Temporal processing
        context = self.temporal(encoded_edges, edge_batches)  # (B, hidden_dim)

        # Get last graph for decoding
        last_graph = graph_seq[-1]
        last_nodes = encoded_nodes[-1]
        last_edges = encoded_edges[-1]
        last_edge_index = edge_indices[-1]
        last_edge_batch = edge_batches[-1]

        # Decode delta
        delta_rho = self.decoder(
            last_nodes,
            last_edge_index,
            last_edges,
            context,
            last_edge_batch
        )  # (E, 4)

        # Enforce Hermiticity
        delta_rho = enforce_hermiticity_edges(delta_rho, last_edge_index)

        if return_matrix:
            # Reconstruct full density matrix
            # Get current density from last graph
            current_edges = last_graph.edge_density
            new_edges = current_edges + delta_rho

            rho_pred = edge_features_to_density(
                new_edges,
                last_edge_index,
                last_graph.num_nodes,
                symmetrize=True
            )
            return rho_pred

        return delta_rho

    def predict_step(
        self,
        graph_seq: List[MolecularGraph],
        next_field: torch.Tensor
    ) -> MolecularGraph:
        """
        Predict next graph state (for autoregressive inference).

        Args:
            graph_seq: List of MolecularGraph objects (history)
            next_field: (3,) field at next timestep

        Returns:
            next_graph: MolecularGraph with predicted density
        """
        # Get delta prediction
        delta_rho = self.forward(graph_seq, next_field, return_matrix=False)

        # Create new graph with updated density
        last_graph = graph_seq[-1]
        new_graph = last_graph.clone()
        new_graph.edge_density = last_graph.edge_density + delta_rho

        return new_graph


# =============================================================================
# Loss Functions
# =============================================================================

class GraphPhysicsLoss(nn.Module):
    """
    Physics-aware loss function for graph-based density prediction.

    Combines:
    1. MSE loss on edge features (density elements)
    2. Trace conservation penalty
    3. Idempotency penalty (for pure states)
    """

    def __init__(
        self,
        lambda_trace: float = 1e-4,
        lambda_idem: float = 1e-5,
        n_electrons_alpha: float = 1.0,
        n_electrons_beta: float = 0.0
    ):
        super().__init__()

        self.lambda_trace = lambda_trace
        self.lambda_idem = lambda_idem
        self.n_electrons_alpha = n_electrons_alpha
        self.n_electrons_beta = n_electrons_beta

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
        edge_index: torch.Tensor,
        edge_density_current: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        n_basis: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss.

        Args:
            delta_pred: (E, 4) predicted delta
            delta_target: (E, 4) target delta
            edge_index: (2, E) edge indices
            edge_density_current: (E, 4) current density (to compute new state)
            S: Optional (N, N) overlap matrix for trace constraint
            n_basis: Number of basis functions

        Returns:
            total_loss: Scalar loss value
            loss_dict: Dictionary of individual loss components
        """
        # MSE on delta prediction
        mse_loss = F.mse_loss(delta_pred, delta_target)

        loss_dict = {'mse': mse_loss.detach()}
        total_loss = mse_loss

        # Physics constraints (if S provided)
        if S is not None and n_basis is not None and self.lambda_trace > 0:
            # Reconstruct predicted density matrix
            new_edges = edge_density_current + delta_pred
            rho_pred = edge_features_to_density(new_edges, edge_index, n_basis, symmetrize=True)

            # Trace constraint: Tr(ρS) = N_electrons
            # For open-shell: check alpha and beta separately
            rho_alpha = rho_pred[0]  # (N, N) complex
            rho_beta = rho_pred[1]

            S_real = S.real if torch.is_complex(S) else S

            trace_alpha = torch.real(torch.trace(rho_alpha @ S_real.to(rho_alpha.dtype)))
            trace_beta = torch.real(torch.trace(rho_beta @ S_real.to(rho_beta.dtype)))

            trace_loss_alpha = (trace_alpha - self.n_electrons_alpha) ** 2
            trace_loss_beta = (trace_beta - self.n_electrons_beta) ** 2
            trace_loss = trace_loss_alpha + trace_loss_beta

            total_loss = total_loss + self.lambda_trace * trace_loss
            loss_dict['trace'] = trace_loss.detach()

            # Idempotency: ||ρSρ - ρ||² (for pure states)
            if self.lambda_idem > 0:
                S_complex = S_real.to(rho_alpha.dtype)
                rhoS_alpha = rho_alpha @ S_complex
                idem_alpha = torch.norm(rhoS_alpha @ rho_alpha - rho_alpha) ** 2

                rhoS_beta = rho_beta @ S_complex
                idem_beta = torch.norm(rhoS_beta @ rho_beta - rho_beta) ** 2

                idem_loss = idem_alpha + idem_beta
                total_loss = total_loss + self.lambda_idem * idem_loss
                loss_dict['idem'] = idem_loss.detach()

        return total_loss, loss_dict


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_dataset_info(
    dataset_info: Dict,
    hidden_dim: int = 128,
    n_gnn_layers: int = 3,
    **kwargs
) -> DensityGraphNet:
    """
    Create model with dimensions matching dataset.

    Args:
        dataset_info: Output of MolecularGraphDataset.get_graph_info()
        hidden_dim: Hidden layer dimension
        n_gnn_layers: Number of GNN layers
        **kwargs: Additional config options

    Returns:
        Configured DensityGraphNet model
    """
    config = {
        'node_dim': dataset_info['node_feature_dim'] + 1,  # +1 for field coupling
        'edge_dim': dataset_info['edge_feature_dim'],
        'hidden_dim': hidden_dim,
        'n_gnn_layers': n_gnn_layers,
        **kwargs
    }

    return DensityGraphNet(config)


if __name__ == "__main__":
    # Basic functionality test
    print("Testing graph_model module...")

    if not HAS_TORCH_GEOMETRIC:
        print("torch_geometric not installed, skipping model tests")
    else:
        from graph_data import create_test_graph

        # Create test configuration
        config = {
            'node_dim': 24,  # 20 (one-hot) + 3 (pos) + 1 (field coupling)
            'edge_dim': 6,
            'hidden_dim': 64,
            'n_gnn_layers': 2,
            'n_lstm_layers': 1,
            'n_attention_heads': 2,
            'dropout': 0.0
        }

        # Create model
        model = DensityGraphNet(config)
        print(f"Model created with {count_parameters(model):,} parameters")

        # Create test sequence
        seq_len = 5
        graph_seq = [create_test_graph(n_basis=4) for _ in range(seq_len)]

        # Adjust node features to match expected dimension
        for g in graph_seq:
            # Pad node features to match config
            pad_size = config['node_dim'] - 1 - g.node_features.shape[-1]
            if pad_size > 0:
                g.node_features = torch.cat([
                    g.node_features,
                    torch.zeros(g.num_nodes, pad_size)
                ], dim=-1)

        # Test forward pass
        next_field = torch.randn(3)
        delta_pred = model(graph_seq, next_field)
        print(f"Delta prediction shape: {delta_pred.shape}")

        # Test with matrix output
        rho_pred = model(graph_seq, next_field, return_matrix=True)
        print(f"Density matrix shape: {rho_pred.shape}")

        # Test loss
        loss_fn = GraphPhysicsLoss(lambda_trace=1e-4, lambda_idem=1e-5)
        delta_target = torch.randn_like(delta_pred)
        loss, loss_dict = loss_fn(
            delta_pred, delta_target,
            graph_seq[-1].edge_index,
            graph_seq[-1].edge_density,
            S=torch.eye(4),
            n_basis=4
        )
        print(f"Loss: {loss.item():.6f}")
        print(f"Loss components: {loss_dict}")

        print("\nAll tests passed!")
