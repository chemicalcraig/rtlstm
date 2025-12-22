"""
Graph-based data pipeline for density matrix propagation.

This module provides utilities for representing density matrices as graphs
where nodes are basis function centers and edges carry the density matrix
elements. This enables size-agnostic learning for transferability across
different molecular systems.

Architecture:
    - Nodes: Atomic orbital basis function centers
    - Edges: Defined by overlap matrix S_ij > threshold
    - Edge features: [Re(ρ_α), Im(ρ_α), Re(ρ_β), Im(ρ_β), S_ij, |r_ij|]
    - Node features: [Z_embedding, field·r_i]
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict, NamedTuple
from dataclasses import dataclass

# Conditional import for torch_geometric
try:
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    Data = None
    Batch = None


# --- Constants ---
MAX_ATOMIC_NUM = 20  # Support up to Calcium for now

# Periodic table for element symbol to atomic number conversion
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
}


def parse_xyz_file(xyz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parse an XYZ file to extract atomic positions and atomic numbers.

    XYZ format:
        N_atoms
        Comment line
        Element1  x1  y1  z1
        Element2  x2  y2  z2
        ...

    Args:
        xyz_path: Path to .xyz file

    Returns:
        positions: (N_atoms, 3) array of atomic positions
        atomic_numbers: (N_atoms,) array of atomic numbers (Z)
        elements: List of element symbols
    """
    positions = []
    atomic_numbers = []
    elements = []

    with open(xyz_path, 'r') as f:
        lines = f.readlines()

    # First line: number of atoms
    n_atoms = int(lines[0].strip())

    # Skip comment line (line 1), parse atom lines (lines 2 onwards)
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        element = parts[0].capitalize()  # Handle case variations
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

        elements.append(element)
        positions.append([x, y, z])

        # Convert element symbol to atomic number
        if element in ELEMENT_TO_Z:
            atomic_numbers.append(ELEMENT_TO_Z[element])
        else:
            raise ValueError(f"Unknown element: {element}")

    return np.array(positions), np.array(atomic_numbers), elements


def expand_atom_to_basis_positions(
    atom_positions: np.ndarray,
    atom_atomic_numbers: np.ndarray,
    n_basis: int,
    basis_per_atom: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand atom-centered positions to per-basis-function positions.

    For atom-centered basis sets, each basis function is located at an
    atomic position. Multiple basis functions can be on the same atom.

    Args:
        atom_positions: (N_atoms, 3) atomic positions
        atom_atomic_numbers: (N_atoms,) atomic numbers
        n_basis: Total number of basis functions
        basis_per_atom: List of basis functions per atom. If None, assumes
                       equal distribution (n_basis // n_atoms per atom).

    Returns:
        basis_positions: (n_basis, 3) position for each basis function
        basis_atomic_numbers: (n_basis,) atomic number for each basis function
    """
    n_atoms = len(atom_positions)

    if basis_per_atom is None:
        # Assume equal distribution of basis functions per atom
        if n_basis % n_atoms != 0:
            # Try to distribute as evenly as possible
            base = n_basis // n_atoms
            remainder = n_basis % n_atoms
            basis_per_atom = [base + (1 if i < remainder else 0) for i in range(n_atoms)]
        else:
            basis_per_atom = [n_basis // n_atoms] * n_atoms

    if sum(basis_per_atom) != n_basis:
        raise ValueError(
            f"basis_per_atom sum ({sum(basis_per_atom)}) != n_basis ({n_basis})"
        )

    # Expand positions and atomic numbers
    basis_positions = []
    basis_atomic_numbers = []

    for atom_idx, n_funcs in enumerate(basis_per_atom):
        for _ in range(n_funcs):
            basis_positions.append(atom_positions[atom_idx])
            basis_atomic_numbers.append(atom_atomic_numbers[atom_idx])

    return np.array(basis_positions), np.array(basis_atomic_numbers)


def load_geometry_from_xyz(
    xyz_path: str,
    n_basis: int,
    basis_per_atom: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load molecular geometry from XYZ file and expand to basis function positions.

    This is the main entry point for loading geometry from XYZ files.

    Args:
        xyz_path: Path to .xyz file
        n_basis: Total number of basis functions (from overlap matrix)
        basis_per_atom: Optional list specifying basis functions per atom.
                       If None, distributes evenly.

    Returns:
        positions: (n_basis, 3) tensor of basis function positions
        atomic_numbers: (n_basis,) tensor of atomic numbers
    """
    atom_pos, atom_z, elements = parse_xyz_file(xyz_path)

    print(f"[XYZ] Parsed {len(elements)} atoms: {', '.join(elements)}")
    print(f"[XYZ] Expanding to {n_basis} basis functions...")

    basis_pos, basis_z = expand_atom_to_basis_positions(
        atom_pos, atom_z, n_basis, basis_per_atom
    )

    return (
        torch.tensor(basis_pos, dtype=torch.float64),
        torch.tensor(basis_z, dtype=torch.long)
    )


@dataclass
class MolecularGraph:
    """
    Container for molecular graph data without torch_geometric dependency.
    Can be converted to PyG Data object when needed.
    """
    # Graph structure
    edge_index: torch.Tensor      # (2, E) COO format
    num_nodes: int

    # Static features (constant across time)
    node_features: torch.Tensor   # (N, d_node) atom embeddings + positions
    edge_static: torch.Tensor     # (E, 2) [S_ij, |r_ij|]

    # Dynamic features (change with time)
    edge_density: torch.Tensor    # (E, 4) [Re(ρ_α), Im(ρ_α), Re(ρ_β), Im(ρ_β)]

    # Metadata
    atomic_numbers: torch.Tensor  # (N,) for node type encoding
    positions: torch.Tensor       # (N, 3) basis center coordinates

    def to_pyg_data(self, field: Optional[torch.Tensor] = None) -> 'Data':
        """Convert to PyTorch Geometric Data object."""
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required. Install with: "
                            "pip install torch-geometric")

        # Combine static and dynamic edge features
        edge_attr = torch.cat([self.edge_density, self.edge_static], dim=-1)

        # Add field coupling to node features if provided
        if field is not None:
            field_coupling = torch.einsum('j,ij->i', field, self.positions)
            node_x = torch.cat([self.node_features,
                               field_coupling.unsqueeze(-1)], dim=-1)
        else:
            node_x = self.node_features

        return Data(
            x=node_x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            pos=self.positions,
            z=self.atomic_numbers,
            num_nodes=self.num_nodes
        )

    def clone(self) -> 'MolecularGraph':
        """Create a deep copy of the graph."""
        return MolecularGraph(
            edge_index=self.edge_index.clone(),
            num_nodes=self.num_nodes,
            node_features=self.node_features.clone(),
            edge_static=self.edge_static.clone(),
            edge_density=self.edge_density.clone(),
            atomic_numbers=self.atomic_numbers.clone(),
            positions=self.positions.clone()
        )


def build_graph_from_overlap(
    S: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    threshold: float = 1e-6,
    include_self_loops: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph connectivity from overlap matrix.

    The overlap matrix S_ij measures the spatial overlap between basis
    functions i and j. Non-zero overlap indicates the basis functions
    interact and should be connected in the graph.

    Args:
        S: (N, N) overlap matrix (real, symmetric)
        positions: (N, 3) basis function center coordinates
        atomic_numbers: (N,) atomic numbers for each basis function
        threshold: S_ij values below this are not edges (sparsification)
        include_self_loops: Whether to include diagonal (i=i) edges

    Returns:
        edge_index: (2, E) COO format edge indices
        edge_static: (E, 2) static edge features [S_ij, |r_ij|]
    """
    N = S.shape[0]
    device = S.device
    dtype = S.dtype if S.dtype in [torch.float32, torch.float64] else torch.float64

    edge_list = []
    edge_features = []

    for i in range(N):
        for j in range(N):
            # Skip sub-threshold off-diagonal elements
            if i != j and abs(S[i, j]) < threshold:
                continue
            # Skip self-loops if not wanted
            if i == j and not include_self_loops:
                continue

            edge_list.append([i, j])

            # Compute edge features
            S_ij = S[i, j].real if torch.is_complex(S) else S[i, j]
            r_ij = positions[j] - positions[i]
            dist = torch.norm(r_ij.float())

            edge_features.append([float(S_ij), float(dist)])

    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
    edge_static = torch.tensor(edge_features, dtype=torch.float64, device=device)

    return edge_index, edge_static


def density_to_edge_features(
    rho: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Encode complex density matrix elements as edge features.

    For each edge (i,j), extracts ρ_ij for both spin channels and
    separates into real and imaginary parts.

    Args:
        rho: (2, N, N) complex density matrix [alpha, beta spin]
        edge_index: (2, E) edge indices in COO format

    Returns:
        edge_density: (E, 4) features [Re(ρ_α), Im(ρ_α), Re(ρ_β), Im(ρ_β)]
    """
    i, j = edge_index[0], edge_index[1]

    # Extract density matrix elements for each edge
    rho_alpha = rho[0, i, j]  # (E,) complex
    rho_beta = rho[1, i, j]   # (E,) complex

    # Stack as [Re_α, Im_α, Re_β, Im_β]
    edge_density = torch.stack([
        rho_alpha.real,
        rho_alpha.imag,
        rho_beta.real,
        rho_beta.imag
    ], dim=-1)  # (E, 4)

    return edge_density.to(torch.float64)


def edge_features_to_density(
    edge_density: torch.Tensor,
    edge_index: torch.Tensor,
    n_basis: int,
    symmetrize: bool = True
) -> torch.Tensor:
    """
    Reconstruct density matrix from edge features.

    Inverse of density_to_edge_features. Used after GNN prediction
    to convert edge predictions back to matrix form.

    Args:
        edge_density: (E, 4) features [Re(ρ_α), Im(ρ_α), Re(ρ_β), Im(ρ_β)]
        edge_index: (2, E) edge indices
        n_basis: Number of basis functions (matrix dimension)
        symmetrize: Whether to enforce Hermiticity

    Returns:
        rho: (2, N, N) complex density matrix
    """
    device = edge_density.device

    # Initialize empty density matrices
    rho = torch.zeros(2, n_basis, n_basis, dtype=torch.complex128, device=device)

    i, j = edge_index[0], edge_index[1]

    # Reconstruct complex values
    rho_alpha = torch.complex(edge_density[:, 0], edge_density[:, 1])
    rho_beta = torch.complex(edge_density[:, 2], edge_density[:, 3])

    # Place in matrix
    rho[0, i, j] = rho_alpha
    rho[1, i, j] = rho_beta

    # Enforce Hermiticity: ρ = 0.5 * (ρ + ρ†)
    if symmetrize:
        rho = 0.5 * (rho + rho.transpose(-1, -2).conj())

    return rho


def compute_node_features(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    use_one_hot: bool = True
) -> torch.Tensor:
    """
    Compute static node features from atomic information.

    Args:
        atomic_numbers: (N,) atomic numbers
        positions: (N, 3) atom/basis positions
        use_one_hot: Use one-hot encoding for atomic numbers

    Returns:
        node_features: (N, d) node feature matrix
    """
    N = atomic_numbers.shape[0]
    device = atomic_numbers.device

    features = []

    if use_one_hot:
        # One-hot encode atomic numbers
        z_one_hot = F.one_hot(
            atomic_numbers.long().clamp(0, MAX_ATOMIC_NUM - 1),
            num_classes=MAX_ATOMIC_NUM
        ).float()  # (N, MAX_ATOMIC_NUM)
        features.append(z_one_hot)
    else:
        # Just use atomic number as feature
        features.append(atomic_numbers.float().unsqueeze(-1))

    # Add position information (can help with spatial reasoning)
    features.append(positions.float())

    return torch.cat(features, dim=-1).to(device)


def enforce_hermiticity_edges(
    delta_rho: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Enforce Hermiticity constraint at the edge level.

    For a Hermitian matrix: ρ_ij = ρ_ji*
    This means:
        Re(ρ_ij) = Re(ρ_ji)
        Im(ρ_ij) = -Im(ρ_ji)

    We average conjugate pairs to enforce this symmetry.

    Args:
        delta_rho: (E, 4) edge density changes [Re_α, Im_α, Re_β, Im_β]
        edge_index: (2, E) edge indices

    Returns:
        delta_sym: (E, 4) symmetrized edge density changes
    """
    i, j = edge_index[0], edge_index[1]
    E = edge_index.shape[1]
    device = delta_rho.device

    # Build reverse edge lookup: (j, i) -> edge_idx
    edge_to_idx = {}
    for k in range(E):
        edge_to_idx[(i[k].item(), j[k].item())] = k

    delta_sym = delta_rho.clone()
    processed = set()

    for k in range(E):
        if k in processed:
            continue

        ii, jj = i[k].item(), j[k].item()

        if ii == jj:
            # Diagonal: imaginary part must be zero for Hermitian
            delta_sym[k, 1] = 0.0  # Im(ρ_α)
            delta_sym[k, 3] = 0.0  # Im(ρ_β)
            processed.add(k)
        elif (jj, ii) in edge_to_idx:
            rev_k = edge_to_idx[(jj, ii)]

            # Average real parts
            avg_re_alpha = 0.5 * (delta_rho[k, 0] + delta_rho[rev_k, 0])
            avg_re_beta = 0.5 * (delta_rho[k, 2] + delta_rho[rev_k, 2])

            # Anti-average imaginary parts (ρ_ij = ρ_ji*)
            avg_im_alpha = 0.5 * (delta_rho[k, 1] - delta_rho[rev_k, 1])
            avg_im_beta = 0.5 * (delta_rho[k, 3] - delta_rho[rev_k, 3])

            delta_sym[k, 0] = avg_re_alpha
            delta_sym[k, 1] = avg_im_alpha
            delta_sym[k, 2] = avg_re_beta
            delta_sym[k, 3] = avg_im_beta

            delta_sym[rev_k, 0] = avg_re_alpha
            delta_sym[rev_k, 1] = -avg_im_alpha  # Conjugate
            delta_sym[rev_k, 2] = avg_re_beta
            delta_sym[rev_k, 3] = -avg_im_beta   # Conjugate

            processed.add(k)
            processed.add(rev_k)

    return delta_sym


class MolecularGraphDataset(Dataset):
    """
    Dataset for graph-based density matrix propagation.

    Converts density matrix time series into sequences of molecular graphs
    for training graph neural networks that can generalize across different
    molecular systems.

    The graph structure is defined by the overlap matrix S, which remains
    constant throughout the dynamics. Only the edge features (density matrix
    elements) change with time.
    """

    def __init__(
        self,
        density_file: str,
        field_file: str,
        overlap_file: str,
        positions_file: Optional[str] = None,
        atomic_numbers_file: Optional[str] = None,
        xyz_file: Optional[str] = None,
        basis_per_atom: Optional[List[int]] = None,
        seq_len: int = 20,
        rollout_steps: int = 5,
        overlap_threshold: float = 1e-6,
        n_basis: Optional[int] = None,
        dtype: torch.dtype = torch.complex128,
        time_reversal_correction: bool = True
    ):
        """
        Initialize the graph dataset.

        Args:
            density_file: Path to .npy file with density matrices
            field_file: Path to field.dat with external field data
            overlap_file: Path to overlap matrix .npy file
            positions_file: Path to basis center positions .npy (optional)
            atomic_numbers_file: Path to atomic numbers .npy array (optional)
            xyz_file: Path to .xyz geometry file (alternative to positions_file)
                     XYZ takes precedence if both are provided
            basis_per_atom: List of basis functions per atom (for xyz_file)
                           If None, distributes n_basis evenly across atoms
            seq_len: Number of history steps for input sequence
            rollout_steps: Number of future steps for training targets
            overlap_threshold: S_ij threshold for graph connectivity
            n_basis: Number of basis functions (inferred if None)
            dtype: Data type for complex tensors
            time_reversal_correction: Apply conjugation to fix time arrow
        """
        print(f"[GraphDataset] Loading density data from {density_file}...")

        # Load density matrices
        data_dict = np.load(density_file, allow_pickle=True)
        if isinstance(data_dict, np.ndarray) and data_dict.dtype == object:
            data_dict = data_dict.item()

        raw_rho = torch.tensor(data_dict['density'], dtype=dtype)

        # Time reversal correction (same as original dataset)
        if time_reversal_correction:
            self.rho = torch.conj(raw_rho)
        else:
            self.rho = raw_rho

        self.n_timesteps, self.n_spin, self.n_basis, _ = self.rho.shape
        if n_basis is not None:
            assert n_basis == self.n_basis, f"n_basis mismatch: {n_basis} vs {self.n_basis}"

        print(f"[GraphDataset] Loaded {self.n_timesteps} timesteps, "
              f"{self.n_spin} spins, {self.n_basis} basis functions")

        # Load external field
        try:
            field_data = np.loadtxt(field_file)
            if field_data.ndim == 1:
                field_data = field_data.reshape(-1, 3)
            elif field_data.shape[1] == 4:
                field_data = field_data[:, 1:]  # Skip time column
            self.field = torch.tensor(field_data, dtype=torch.float64)
        except Exception as e:
            print(f"[GraphDataset] Warning: Could not load field file: {e}")
            print("[GraphDataset] Using zero field.")
            self.field = torch.zeros((self.n_timesteps, 3), dtype=torch.float64)

        # Align lengths
        min_len = min(len(self.rho), len(self.field))
        self.rho = self.rho[:min_len]
        self.field = self.field[:min_len]
        self.n_timesteps = min_len

        # Load overlap matrix
        print(f"[GraphDataset] Loading overlap matrix from {overlap_file}...")
        self.S = torch.tensor(np.load(overlap_file), dtype=torch.float64)
        assert self.S.shape == (self.n_basis, self.n_basis), \
            f"Overlap shape mismatch: {self.S.shape} vs ({self.n_basis}, {self.n_basis})"

        # Load positions and atomic numbers
        # Priority: xyz_file > positions_file/atomic_numbers_file > dummy
        if xyz_file is not None:
            # Load from XYZ geometry file
            self.positions, self.atomic_numbers = load_geometry_from_xyz(
                xyz_file, self.n_basis, basis_per_atom
            )
        elif positions_file is not None:
            self.positions = torch.tensor(np.load(positions_file), dtype=torch.float64)
            if atomic_numbers_file is not None:
                self.atomic_numbers = torch.tensor(np.load(atomic_numbers_file), dtype=torch.long)
            else:
                print("[GraphDataset] Warning: No atomic numbers file, assuming hydrogen")
                self.atomic_numbers = torch.ones(self.n_basis, dtype=torch.long)
        else:
            # Default: place basis functions at origin (not ideal but functional)
            print("[GraphDataset] Warning: No geometry file, using dummy positions")
            self.positions = self._generate_dummy_positions()
            self.atomic_numbers = torch.ones(self.n_basis, dtype=torch.long)

        # Build graph structure (constant for all timesteps)
        print(f"[GraphDataset] Building graph with threshold={overlap_threshold}...")
        self.edge_index, self.edge_static = build_graph_from_overlap(
            self.S, self.positions, self.atomic_numbers,
            threshold=overlap_threshold,
            include_self_loops=True
        )
        print(f"[GraphDataset] Graph has {self.n_basis} nodes, "
              f"{self.edge_index.shape[1]} edges")

        # Compute static node features
        self.node_features = compute_node_features(
            self.atomic_numbers, self.positions, use_one_hot=True
        )

        # Store configuration
        self.seq_len = seq_len
        self.rollout_steps = rollout_steps
        self.dtype = dtype

    def _generate_dummy_positions(self) -> torch.Tensor:
        """Generate placeholder positions based on overlap structure."""
        # Use overlap matrix to infer approximate distances
        # This is a crude approximation - real positions should come from input
        N = self.n_basis
        positions = torch.zeros(N, 3, dtype=torch.float64)

        # Simple linear arrangement as fallback
        for i in range(N):
            positions[i, 0] = i * 1.0  # 1 Bohr spacing

        return positions

    def __len__(self) -> int:
        return self.n_timesteps - self.seq_len - self.rollout_steps

    def __getitem__(self, idx: int) -> Tuple[List[MolecularGraph], torch.Tensor,
                                             List[MolecularGraph]]:
        """
        Get a training sample.

        Returns:
            input_graphs: List of seq_len MolecularGraph objects (history)
            field_targets: (rollout_steps, 3) future field values
            target_graphs: List of rollout_steps MolecularGraph objects (targets)
        """
        # Build input sequence of graphs
        input_graphs = []
        for t in range(idx, idx + self.seq_len):
            rho_t = self.rho[t]  # (2, N, N) complex
            edge_density = density_to_edge_features(rho_t, self.edge_index)

            graph = MolecularGraph(
                edge_index=self.edge_index.clone(),
                num_nodes=self.n_basis,
                node_features=self.node_features.clone(),
                edge_static=self.edge_static.clone(),
                edge_density=edge_density,
                atomic_numbers=self.atomic_numbers.clone(),
                positions=self.positions.clone()
            )
            input_graphs.append(graph)

        # Build target sequence of graphs
        target_graphs = []
        for t in range(idx + self.seq_len, idx + self.seq_len + self.rollout_steps):
            rho_t = self.rho[t]
            edge_density = density_to_edge_features(rho_t, self.edge_index)

            graph = MolecularGraph(
                edge_index=self.edge_index.clone(),
                num_nodes=self.n_basis,
                node_features=self.node_features.clone(),
                edge_static=self.edge_static.clone(),
                edge_density=edge_density,
                atomic_numbers=self.atomic_numbers.clone(),
                positions=self.positions.clone()
            )
            target_graphs.append(graph)

        # Get target field values
        field_targets = self.field[idx + self.seq_len : idx + self.seq_len + self.rollout_steps]

        return input_graphs, field_targets, target_graphs

    def get_graph_info(self) -> Dict:
        """Return graph structure information for model initialization."""
        return {
            'n_nodes': self.n_basis,
            'n_edges': self.edge_index.shape[1],
            'node_feature_dim': self.node_features.shape[-1],
            'edge_static_dim': self.edge_static.shape[-1],
            'edge_density_dim': 4,  # [Re_α, Im_α, Re_β, Im_β]
            'edge_feature_dim': self.edge_static.shape[-1] + 4,
        }


def collate_graph_sequences(
    batch: List[Tuple[List[MolecularGraph], torch.Tensor, List[MolecularGraph]]]
) -> Tuple[List[List[MolecularGraph]], torch.Tensor, List[List[MolecularGraph]]]:
    """
    Custom collate function for graph sequence batches.

    For now, returns lists of graphs. For actual batching with PyG,
    use Batch.from_data_list on the converted Data objects.

    Args:
        batch: List of (input_graphs, field_targets, target_graphs) tuples

    Returns:
        Batched input graphs, stacked field targets, batched target graphs
    """
    input_seqs = [item[0] for item in batch]
    field_targets = torch.stack([item[1] for item in batch], dim=0)
    target_seqs = [item[2] for item in batch]

    return input_seqs, field_targets, target_seqs


# =============================================================================
# Skip-Step Graph Dataset
# =============================================================================

class SkipStepGraphDataset(Dataset):
    """
    Skip-step dataset for graph-based density matrix propagation.

    Enables learning mappings t → t + k·Δt where k > 1, providing
    computational speedup by skipping intermediate time steps.

    Supports:
        - Dynamic skip factor adjustment (for curriculum learning)
        - Multiple supervision modes (endpoints, intermediate, multi-scale)
        - Efficient graph construction with shared static features
    """

    def __init__(
        self,
        density_file: str,
        field_file: str,
        overlap_file: str,
        positions_file: Optional[str] = None,
        atomic_numbers_file: Optional[str] = None,
        xyz_file: Optional[str] = None,
        basis_per_atom: Optional[List[int]] = None,
        seq_len: int = 20,
        rollout_steps: int = 5,
        skip_factor: int = 1,
        overlap_threshold: float = 1e-6,
        time_reversal_correction: bool = True,
        supervision_mode: str = "endpoints",
        intermediate_samples: int = 2
    ):
        """
        Initialize skip-step graph dataset.

        Args:
            density_file: Path to density matrix time series
            field_file: Path to field data
            overlap_file: Path to overlap matrix
            positions_file: Optional path to basis positions .npy
            atomic_numbers_file: Optional path to atomic numbers .npy
            xyz_file: Path to .xyz geometry file (alternative to positions_file)
            basis_per_atom: List of basis functions per atom (for xyz_file)
            seq_len: Sequence length (at coarse resolution)
            rollout_steps: Number of prediction steps
            skip_factor: k - number of fine steps per coarse step
            overlap_threshold: Graph sparsification threshold
            time_reversal_correction: Apply conjugation fix
            supervision_mode: 'endpoints', 'intermediate', or 'multi_scale'
            intermediate_samples: Points between endpoints for intermediate mode
        """
        print(f"[SkipStepGraphDataset] Loading data...")

        # Load density matrices
        data_dict = np.load(density_file, allow_pickle=True)
        if isinstance(data_dict, np.ndarray) and data_dict.dtype == object:
            data_dict = data_dict.item()

        raw_rho = torch.tensor(data_dict['density'], dtype=torch.complex128)

        if time_reversal_correction:
            self.rho = torch.conj(raw_rho)
        else:
            self.rho = raw_rho

        self.n_timesteps, self.n_spin, self.n_basis, _ = self.rho.shape
        print(f"[SkipStepGraphDataset] {self.n_timesteps} timesteps, "
              f"{self.n_spin} spins, {self.n_basis} basis")

        # Load field
        try:
            field_data = np.loadtxt(field_file)
            if field_data.ndim == 1:
                field_data = field_data.reshape(-1, 3)
            elif field_data.shape[1] == 4:
                field_data = field_data[:, 1:]
            self.field = torch.tensor(field_data, dtype=torch.float64)
        except Exception:
            print("[SkipStepGraphDataset] Warning: Could not load field, using zeros")
            self.field = torch.zeros((self.n_timesteps, 3), dtype=torch.float64)

        # Align lengths
        min_len = min(len(self.rho), len(self.field))
        self.rho = self.rho[:min_len]
        self.field = self.field[:min_len]
        self.n_timesteps = min_len

        # Load overlap matrix
        self.S = torch.tensor(np.load(overlap_file), dtype=torch.float64)

        # Load positions and atomic numbers
        # Priority: xyz_file > positions_file/atomic_numbers_file > dummy
        if xyz_file is not None:
            # Load from XYZ geometry file
            self.positions, self.atomic_numbers = load_geometry_from_xyz(
                xyz_file, self.n_basis, basis_per_atom
            )
        elif positions_file is not None:
            self.positions = torch.tensor(np.load(positions_file), dtype=torch.float64)
            if atomic_numbers_file is not None:
                self.atomic_numbers = torch.tensor(np.load(atomic_numbers_file), dtype=torch.long)
            else:
                self.atomic_numbers = torch.ones(self.n_basis, dtype=torch.long)
        else:
            # Default: linear arrangement
            print("[SkipStepGraphDataset] Warning: No geometry file, using dummy positions")
            self.positions = torch.zeros(self.n_basis, 3, dtype=torch.float64)
            self.positions[:, 0] = torch.arange(self.n_basis, dtype=torch.float64)
            self.atomic_numbers = torch.ones(self.n_basis, dtype=torch.long)

        # Build graph structure (shared across all timesteps)
        self.edge_index, self.edge_static = build_graph_from_overlap(
            self.S, self.positions, self.atomic_numbers,
            threshold=overlap_threshold, include_self_loops=True
        )

        self.node_features = compute_node_features(
            self.atomic_numbers, self.positions, use_one_hot=True
        )

        print(f"[SkipStepGraphDataset] Graph: {self.n_basis} nodes, "
              f"{self.edge_index.shape[1]} edges")

        # Skip-step parameters
        self.seq_len = seq_len
        self.rollout_steps = rollout_steps
        self.k = skip_factor
        self.supervision_mode = supervision_mode
        self.intermediate_samples = intermediate_samples

        # Compute valid sample range
        self._update_sample_count()

    def _update_sample_count(self):
        """Update number of valid samples based on current k."""
        required_span = (self.seq_len + self.rollout_steps) * self.k
        self.n_samples = max(0, self.n_timesteps - required_span)

    def set_skip_factor(self, k: int):
        """Update skip factor (for curriculum learning)."""
        self.k = k
        self._update_sample_count()

    def __len__(self) -> int:
        return self.n_samples

    def _build_graph_at_time(self, t: int) -> MolecularGraph:
        """Build graph for a specific timestep."""
        rho_t = self.rho[t]
        edge_density = density_to_edge_features(rho_t, self.edge_index)

        return MolecularGraph(
            edge_index=self.edge_index.clone(),
            num_nodes=self.n_basis,
            node_features=self.node_features.clone(),
            edge_static=self.edge_static.clone(),
            edge_density=edge_density,
            atomic_numbers=self.atomic_numbers.clone(),
            positions=self.positions.clone()
        )

    def __getitem__(self, idx: int) -> Tuple[List[MolecularGraph], torch.Tensor,
                                             List[MolecularGraph], Dict]:
        """
        Get a skip-step training sample.

        Returns:
            input_graphs: List of seq_len graphs at coarse resolution
            field_targets: (rollout_steps, 3) target fields
            target_graphs: List of rollout_steps target graphs
            supervision: Dict with skip info and optional intermediate targets
        """
        # Build input sequence at coarse resolution (stride k)
        input_graphs = []
        for i in range(self.seq_len):
            t = idx + i * self.k
            input_graphs.append(self._build_graph_at_time(t))

        # Build target sequence
        target_start = idx + self.seq_len * self.k
        target_graphs = []
        target_indices = []

        for i in range(self.rollout_steps):
            t = target_start + i * self.k
            target_graphs.append(self._build_graph_at_time(t))
            target_indices.append(t)

        # Get target fields
        field_targets = self.field[target_indices]

        # Build supervision dict
        supervision = {
            'skip_factor': self.k,
            'mode': self.supervision_mode,
            'target_indices': target_indices
        }

        # Add intermediate supervision if requested
        if self.supervision_mode == 'intermediate' and self.k > 1:
            intermediate_graphs = []
            intermediate_times = []

            for step in range(self.rollout_steps):
                step_start = target_start + step * self.k
                step_end = target_start + (step + 1) * self.k

                if self.intermediate_samples > 0:
                    sample_indices = np.linspace(
                        step_start, step_end,
                        self.intermediate_samples + 2,
                        dtype=int
                    )[1:-1]

                    for si in sample_indices:
                        if si < self.n_timesteps:
                            intermediate_graphs.append(self._build_graph_at_time(si))
                            intermediate_times.append(si - target_start)

            if intermediate_graphs:
                supervision['intermediate_graphs'] = intermediate_graphs
                supervision['intermediate_times'] = intermediate_times

        elif self.supervision_mode == 'multi_scale' and self.k > 1:
            # Provide targets at multiple scales
            multi_scale = {}
            for scale_k in [1, 2, 4, 8]:
                if scale_k < self.k:
                    scale_graphs = []
                    n_scale_steps = min(self.rollout_steps * (self.k // scale_k),
                                       self.n_timesteps - target_start)

                    for i in range(n_scale_steps):
                        t = target_start + i * scale_k
                        if t < self.n_timesteps:
                            scale_graphs.append(self._build_graph_at_time(t))

                    if scale_graphs:
                        multi_scale[scale_k] = scale_graphs

            supervision['multi_scale'] = multi_scale

        return input_graphs, field_targets, target_graphs, supervision

    def get_graph_info(self) -> Dict:
        """Return graph structure information."""
        return {
            'n_nodes': self.n_basis,
            'n_edges': self.edge_index.shape[1],
            'node_feature_dim': self.node_features.shape[-1],
            'edge_static_dim': self.edge_static.shape[-1],
            'edge_density_dim': 4,
            'edge_feature_dim': self.edge_static.shape[-1] + 4,
            'skip_factor': self.k
        }


def collate_skip_step_graphs(
    batch: List[Tuple[List[MolecularGraph], torch.Tensor, List[MolecularGraph], Dict]]
) -> Tuple[List[List[MolecularGraph]], torch.Tensor, List[List[MolecularGraph]], List[Dict]]:
    """
    Collate function for skip-step graph batches.

    Args:
        batch: List of (input_graphs, field_targets, target_graphs, supervision) tuples

    Returns:
        Batched inputs, stacked fields, batched targets, list of supervision dicts
    """
    input_seqs = [item[0] for item in batch]
    field_targets = torch.stack([item[1] for item in batch], dim=0)
    target_seqs = [item[2] for item in batch]
    supervision_list = [item[3] for item in batch]

    return input_seqs, field_targets, target_seqs, supervision_list


# --- Utility functions for testing ---

def create_test_graph(n_basis: int = 4) -> MolecularGraph:
    """Create a simple test graph for debugging."""
    # Simple overlap matrix (identity + small off-diagonal)
    S = torch.eye(n_basis, dtype=torch.float64)
    for i in range(n_basis - 1):
        S[i, i+1] = 0.1
        S[i+1, i] = 0.1

    # Linear positions
    positions = torch.zeros(n_basis, 3, dtype=torch.float64)
    positions[:, 0] = torch.arange(n_basis, dtype=torch.float64)

    # All hydrogen
    atomic_numbers = torch.ones(n_basis, dtype=torch.long)

    # Build graph
    edge_index, edge_static = build_graph_from_overlap(
        S, positions, atomic_numbers, threshold=0.05
    )

    # Random density (Hermitian)
    rho = torch.randn(2, n_basis, n_basis, dtype=torch.complex128)
    rho = 0.5 * (rho + rho.transpose(-1, -2).conj())  # Make Hermitian

    edge_density = density_to_edge_features(rho, edge_index)
    node_features = compute_node_features(atomic_numbers, positions)

    return MolecularGraph(
        edge_index=edge_index,
        num_nodes=n_basis,
        node_features=node_features,
        edge_static=edge_static,
        edge_density=edge_density,
        atomic_numbers=atomic_numbers,
        positions=positions
    )


if __name__ == "__main__":
    # Basic sanity checks
    print("Testing graph_data module...")

    # Test graph creation
    graph = create_test_graph(n_basis=4)
    print(f"Created test graph with {graph.num_nodes} nodes, "
          f"{graph.edge_index.shape[1]} edges")
    print(f"Node features shape: {graph.node_features.shape}")
    print(f"Edge density shape: {graph.edge_density.shape}")
    print(f"Edge static shape: {graph.edge_static.shape}")

    # Test density reconstruction
    rho_reconstructed = edge_features_to_density(
        graph.edge_density, graph.edge_index, graph.num_nodes
    )
    print(f"Reconstructed density shape: {rho_reconstructed.shape}")

    # Test Hermiticity enforcement
    delta = torch.randn(graph.edge_index.shape[1], 4)
    delta_sym = enforce_hermiticity_edges(delta, graph.edge_index)
    print(f"Hermiticity enforcement: input shape {delta.shape}, "
          f"output shape {delta_sym.shape}")

    # Test PyG conversion if available
    if HAS_TORCH_GEOMETRIC:
        field = torch.randn(3)
        pyg_data = graph.to_pyg_data(field=field)
        print(f"PyG Data: {pyg_data}")
    else:
        print("torch_geometric not installed, skipping PyG conversion test")

    print("\nAll tests passed!")
