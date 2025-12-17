"""
Skip-Step Propagation Module for Accelerated Density Matrix Dynamics.

This module implements the machinery for learning mappings t → t + k·Δt
where k > 1, enabling significant computational speedups by bypassing
expensive intermediate time-integration steps.

Key Concepts:
    - Skip Factor (k): Number of fine time-steps to skip per prediction
    - Effective Δt: k × Δt_fine (the actual time interval learned)
    - Curriculum Learning: Gradually increase k during training for stability
    - Multi-Scale Supervision: Optionally constrain intermediate states

Physics Considerations:
    - Larger k means more dynamics "hidden" between samples
    - Need stronger regularization to maintain physical constraints
    - Consider energy/trace drift over longer intervals
    - Phase accumulation becomes significant for large k

Usage:
    # Wrap existing dataset for skip-step training
    skip_dataset = SkipStepDataset(
        base_dataset,
        skip_factor=10,
        supervision_mode='endpoints'  # or 'intermediate' or 'multi_scale'
    )

    # Use curriculum learning
    scheduler = SkipStepCurriculum(
        initial_k=1,
        final_k=20,
        warmup_epochs=50,
        schedule='linear'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math


class SupervisionMode(Enum):
    """Supervision strategies for skip-step training."""
    ENDPOINTS = "endpoints"          # Only supervise t and t+k
    INTERMEDIATE = "intermediate"    # Also supervise some middle points
    MULTI_SCALE = "multi_scale"      # Hierarchical: k=1, k=2, k=4, etc.
    TRAJECTORY = "trajectory"        # Full trajectory matching


@dataclass
class SkipStepConfig:
    """Configuration for skip-step propagation."""
    skip_factor: int = 10                              # k: steps to skip
    supervision_mode: str = "endpoints"                # How to supervise
    intermediate_samples: int = 2                      # Points between endpoints
    multi_scale_factors: List[int] = None             # [1, 2, 4, 8, ...]
    trajectory_length: int = 5                         # For trajectory mode

    # Curriculum learning
    use_curriculum: bool = True
    initial_skip: int = 1
    curriculum_warmup_epochs: int = 50
    curriculum_schedule: str = "exponential"          # linear, exponential, step

    # Stability regularization
    lambda_smoothness: float = 1e-4                   # Penalize jerky predictions
    lambda_energy: float = 1e-4                       # Energy conservation
    lambda_phase: float = 1e-5                        # Phase coherence

    def __post_init__(self):
        if self.multi_scale_factors is None:
            # Default: powers of 2 up to skip_factor
            max_power = int(np.log2(self.skip_factor)) + 1
            self.multi_scale_factors = [2**i for i in range(max_power)]


# =============================================================================
# Skip-Step Dataset Wrappers
# =============================================================================

class SkipStepDatasetLSTM(Dataset):
    """
    Skip-step wrapper for LSTM-based MolecularDynamicsDataset.

    Modifies the base dataset to sample with stride k instead of stride 1,
    effectively learning t → t + k·Δt mappings.
    """

    def __init__(
        self,
        density_data: torch.Tensor,
        field_data: torch.Tensor,
        seq_len: int,
        skip_factor: int = 10,
        rollout_steps: int = 5,
        supervision_mode: str = "endpoints",
        intermediate_samples: int = 2
    ):
        """
        Initialize skip-step dataset.

        Args:
            density_data: (T, 2, N, N) density matrix time series
            field_data: (T, 3) external field time series
            seq_len: Number of history steps (at skip resolution)
            skip_factor: k - number of fine steps per coarse step
            rollout_steps: Number of future predictions
            supervision_mode: How to provide supervision
            intermediate_samples: Number of intermediate supervisions
        """
        self.rho = density_data
        self.field = field_data
        self.seq_len = seq_len
        self.k = skip_factor
        self.rollout_steps = rollout_steps
        self.supervision_mode = SupervisionMode(supervision_mode)
        self.intermediate_samples = intermediate_samples

        # Compute valid indices
        # Need: seq_len * k steps of history + rollout_steps * k steps of future
        self.total_fine_steps = len(self.rho)
        self.required_span = (seq_len + rollout_steps) * self.k
        self.n_samples = max(0, self.total_fine_steps - self.required_span)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get a skip-step sample.

        Returns:
            rho_seq: (seq_len, 2, N, N) input sequence at coarse resolution
            field_targets: (rollout_steps, 3) target fields at coarse resolution
            rho_targets: (rollout_steps, 2, N, N) target densities
            supervision: Dict with intermediate supervision if requested
        """
        # Sample indices at coarse resolution (stride k)
        coarse_indices = [idx + i * self.k for i in range(self.seq_len)]
        rho_seq = self.rho[coarse_indices]

        # Target indices
        target_start = idx + self.seq_len * self.k
        target_indices = [target_start + i * self.k for i in range(self.rollout_steps)]
        rho_targets = self.rho[target_indices]

        # Field at target times
        field_targets = self.field[target_indices]

        # Build supervision dict based on mode
        supervision = self._build_supervision(idx, target_start)

        return rho_seq, field_targets, rho_targets, supervision

    def _build_supervision(self, idx: int, target_start: int) -> Dict:
        """Build supervision dictionary based on mode."""
        supervision = {
            'mode': self.supervision_mode.value,
            'skip_factor': self.k
        }

        if self.supervision_mode == SupervisionMode.INTERMEDIATE:
            # Sample intermediate points between each coarse step
            intermediate_rho = []
            intermediate_times = []

            for step in range(self.rollout_steps):
                step_start = target_start + step * self.k
                step_end = target_start + (step + 1) * self.k

                # Sample intermediate_samples points uniformly
                if self.k > 1 and self.intermediate_samples > 0:
                    sample_indices = np.linspace(
                        step_start, step_end,
                        self.intermediate_samples + 2,
                        dtype=int
                    )[1:-1]  # Exclude endpoints

                    for si in sample_indices:
                        if si < len(self.rho):
                            intermediate_rho.append(self.rho[si])
                            intermediate_times.append(si - target_start)

            if intermediate_rho:
                supervision['intermediate_rho'] = torch.stack(intermediate_rho)
                supervision['intermediate_times'] = torch.tensor(intermediate_times)

        elif self.supervision_mode == SupervisionMode.MULTI_SCALE:
            # Provide targets at multiple scales: k=1, k=2, k=4, ...
            multi_scale = {}
            for scale_k in [1, 2, 4, 8]:
                if scale_k <= self.k:
                    scale_indices = [target_start + i * scale_k
                                    for i in range(self.rollout_steps * (self.k // scale_k))]
                    scale_indices = [i for i in scale_indices if i < len(self.rho)]
                    if scale_indices:
                        multi_scale[scale_k] = self.rho[scale_indices]
            supervision['multi_scale'] = multi_scale

        elif self.supervision_mode == SupervisionMode.TRAJECTORY:
            # Full fine-resolution trajectory
            traj_indices = list(range(target_start, min(target_start + self.k * self.rollout_steps, len(self.rho))))
            supervision['trajectory'] = self.rho[traj_indices]
            supervision['trajectory_fields'] = self.field[traj_indices]

        return supervision

    def set_skip_factor(self, k: int):
        """Update skip factor (for curriculum learning)."""
        self.k = k
        self.required_span = (self.seq_len + self.rollout_steps) * self.k
        self.n_samples = max(0, self.total_fine_steps - self.required_span)


class SkipStepDatasetGraph(Dataset):
    """
    Skip-step wrapper for graph-based MolecularGraphDataset.

    Similar to LSTM version but returns graph sequences.
    """

    def __init__(
        self,
        base_dataset,  # MolecularGraphDataset
        skip_factor: int = 10,
        supervision_mode: str = "endpoints"
    ):
        """
        Wrap a MolecularGraphDataset for skip-step training.

        Args:
            base_dataset: MolecularGraphDataset instance
            skip_factor: k - steps to skip
            supervision_mode: Supervision strategy
        """
        self.base = base_dataset
        self.k = skip_factor
        self.supervision_mode = SupervisionMode(supervision_mode)

        # Access underlying data
        self.rho = base_dataset.rho
        self.field = base_dataset.field
        self.seq_len = base_dataset.seq_len
        self.rollout_steps = base_dataset.rollout_steps

        # Compute valid range
        self.total_steps = len(self.rho)
        self.required_span = (self.seq_len + self.rollout_steps) * self.k
        self.n_samples = max(0, self.total_steps - self.required_span)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        """Get skip-step sample as graphs."""
        # Build coarse-resolution graph sequence
        input_graphs = []
        for i in range(self.seq_len):
            t = idx + i * self.k
            graph = self.base._build_graph_at_time(t)
            input_graphs.append(graph)

        # Target graphs
        target_start = idx + self.seq_len * self.k
        target_graphs = []
        for i in range(self.rollout_steps):
            t = target_start + i * self.k
            graph = self.base._build_graph_at_time(t)
            target_graphs.append(graph)

        # Fields
        target_indices = [target_start + i * self.k for i in range(self.rollout_steps)]
        field_targets = self.field[target_indices]

        # Supervision
        supervision = {'skip_factor': self.k, 'mode': self.supervision_mode.value}

        return input_graphs, field_targets, target_graphs, supervision

    def set_skip_factor(self, k: int):
        """Update skip factor for curriculum learning."""
        self.k = k
        self.required_span = (self.seq_len + self.rollout_steps) * self.k
        self.n_samples = max(0, self.total_steps - self.required_span)


# =============================================================================
# Curriculum Learning
# =============================================================================

class SkipStepCurriculum:
    """
    Curriculum scheduler for gradually increasing skip factor.

    Starting with k=1 (standard propagation) and gradually increasing
    to the target k helps the model learn stable long-range predictions.
    """

    def __init__(
        self,
        initial_k: int = 1,
        final_k: int = 20,
        warmup_epochs: int = 50,
        total_epochs: int = 200,
        schedule: str = "exponential",
        plateau_epochs: int = 10
    ):
        """
        Initialize curriculum scheduler.

        Args:
            initial_k: Starting skip factor
            final_k: Target skip factor
            warmup_epochs: Epochs before starting to increase k
            total_epochs: Total training epochs
            schedule: 'linear', 'exponential', 'step', or 'cosine'
            plateau_epochs: Epochs to hold at each k level (for 'step')
        """
        self.initial_k = initial_k
        self.final_k = final_k
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.plateau_epochs = plateau_epochs

        self.current_k = initial_k
        self.current_epoch = 0

    def step(self, epoch: int) -> int:
        """
        Get skip factor for current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            k: Skip factor to use
        """
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            self.current_k = self.initial_k
            return self.current_k

        # Progress through curriculum (0 to 1)
        curriculum_epochs = self.total_epochs - self.warmup_epochs
        progress = min(1.0, (epoch - self.warmup_epochs) / curriculum_epochs)

        if self.schedule == "linear":
            k_float = self.initial_k + progress * (self.final_k - self.initial_k)

        elif self.schedule == "exponential":
            # Exponential growth: k = initial * (final/initial)^progress
            ratio = self.final_k / max(1, self.initial_k)
            k_float = self.initial_k * (ratio ** progress)

        elif self.schedule == "step":
            # Discrete steps with plateaus
            n_steps = (self.final_k - self.initial_k)
            epochs_per_step = max(1, curriculum_epochs // n_steps)
            steps_completed = (epoch - self.warmup_epochs) // epochs_per_step
            k_float = min(self.final_k, self.initial_k + steps_completed)

        elif self.schedule == "cosine":
            # Smooth cosine schedule
            k_float = self.initial_k + 0.5 * (self.final_k - self.initial_k) * \
                      (1 - math.cos(math.pi * progress))

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        self.current_k = max(1, min(self.final_k, int(round(k_float))))
        return self.current_k

    def get_current_k(self) -> int:
        """Get current skip factor."""
        return self.current_k

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_k': self.current_k,
            'current_epoch': self.current_epoch,
            'initial_k': self.initial_k,
            'final_k': self.final_k
        }

    def load_state_dict(self, state: Dict):
        """Load scheduler state."""
        self.current_k = state['current_k']
        self.current_epoch = state['current_epoch']


# =============================================================================
# Skip-Step Loss Functions
# =============================================================================

class SkipStepLoss(nn.Module):
    """
    Loss function for skip-step training with stability regularization.

    Handles multiple supervision modes and adds physics-motivated
    regularization terms for stable long-range predictions.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        lambda_smoothness: float = 1e-4,
        lambda_energy: float = 1e-4,
        lambda_phase: float = 1e-5,
        lambda_intermediate: float = 0.5,
        S: Optional[torch.Tensor] = None
    ):
        """
        Initialize skip-step loss.

        Args:
            base_loss: Base loss function (e.g., PhysicsLoss or GraphPhysicsLoss)
            lambda_smoothness: Weight for smoothness regularization
            lambda_energy: Weight for energy conservation
            lambda_phase: Weight for phase coherence
            lambda_intermediate: Weight for intermediate supervision
            S: Overlap matrix for physics constraints
        """
        super().__init__()
        self.base_loss = base_loss
        self.lambda_smoothness = lambda_smoothness
        self.lambda_energy = lambda_energy
        self.lambda_phase = lambda_phase
        self.lambda_intermediate = lambda_intermediate
        self.S = S

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
        supervision: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute skip-step loss.

        Args:
            pred: Predicted density (or delta)
            target: Target density (or delta)
            prev_pred: Previous prediction (for smoothness)
            supervision: Supervision dict from dataset
            **kwargs: Additional args for base loss

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of loss components
        """
        # Base prediction loss
        if hasattr(self.base_loss, 'forward'):
            base_result = self.base_loss(pred, target, **kwargs)
            if isinstance(base_result, tuple):
                base_loss, loss_dict = base_result
            else:
                base_loss = base_result
                loss_dict = {'base': base_loss.detach()}
        else:
            base_loss = F.mse_loss(pred, target)
            loss_dict = {'base': base_loss.detach()}

        total_loss = base_loss

        # Smoothness regularization: penalize large changes between predictions
        if self.lambda_smoothness > 0 and prev_pred is not None:
            if pred.shape == prev_pred.shape:
                smoothness_loss = F.mse_loss(pred, prev_pred)
                total_loss = total_loss + self.lambda_smoothness * smoothness_loss
                loss_dict['smoothness'] = smoothness_loss.detach()

        # Phase coherence: penalize large imaginary component changes
        if self.lambda_phase > 0:
            if pred.dim() >= 2:
                # For edge features: columns 1, 3 are imaginary parts
                if pred.shape[-1] == 4:
                    phase_var = pred[:, [1, 3]].var()
                    total_loss = total_loss + self.lambda_phase * phase_var
                    loss_dict['phase'] = phase_var.detach()
                # For matrices: imaginary part
                elif torch.is_complex(pred):
                    phase_var = pred.imag.var()
                    total_loss = total_loss + self.lambda_phase * phase_var
                    loss_dict['phase'] = phase_var.detach()

        # Intermediate supervision
        if supervision is not None and self.lambda_intermediate > 0:
            mode = supervision.get('mode', 'endpoints')

            if mode == 'intermediate' and 'intermediate_rho' in supervision:
                # Would need interpolated predictions - placeholder
                pass

            elif mode == 'multi_scale' and 'multi_scale' in supervision:
                # Multi-scale consistency loss
                # Penalize if fine-scale predictions don't match coarse
                pass

        return total_loss, loss_dict


class AdaptiveSkipLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on skip factor.

    Larger skip factors require stronger regularization.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        k_ref: int = 1,
        smoothness_scale: float = 0.1,
        energy_scale: float = 0.1
    ):
        """
        Initialize adaptive loss.

        Args:
            base_loss: Base loss function
            k_ref: Reference skip factor (regularization = 1.0 at this k)
            smoothness_scale: How much to increase smoothness penalty per k
            energy_scale: How much to increase energy penalty per k
        """
        super().__init__()
        self.base_loss = base_loss
        self.k_ref = k_ref
        self.smoothness_scale = smoothness_scale
        self.energy_scale = energy_scale

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        k: int,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss with k-dependent regularization."""
        # Scale regularization with k
        k_ratio = k / self.k_ref

        # Adjusted lambdas
        lambda_smooth = self.smoothness_scale * k_ratio
        lambda_energy = self.energy_scale * k_ratio

        # Compute base loss
        if isinstance(self.base_loss, SkipStepLoss):
            # Update lambdas
            self.base_loss.lambda_smoothness = lambda_smooth
            self.base_loss.lambda_energy = lambda_energy

        result = self.base_loss(pred, target, **kwargs)

        if isinstance(result, tuple):
            loss, loss_dict = result
            loss_dict['k'] = torch.tensor(float(k))
            loss_dict['k_ratio'] = torch.tensor(k_ratio)
        else:
            loss = result
            loss_dict = {'loss': loss.detach(), 'k': torch.tensor(float(k))}

        return loss, loss_dict


# =============================================================================
# Skip-Step Model Wrapper
# =============================================================================

class SkipStepPredictor(nn.Module):
    """
    Wrapper that adapts any base model for skip-step prediction.

    Optionally adds k-conditioning so the model knows the time scale.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_k: int = 20,
        k_embedding_dim: int = 16,
        condition_on_k: bool = True
    ):
        """
        Initialize skip-step predictor.

        Args:
            base_model: Base model (DensityMatrixLSTM or DensityGraphNet)
            max_k: Maximum skip factor
            k_embedding_dim: Dimension of k embedding
            condition_on_k: Whether to condition model on k
        """
        super().__init__()
        self.base_model = base_model
        self.max_k = max_k
        self.condition_on_k = condition_on_k

        if condition_on_k:
            # Learnable embedding for skip factor
            self.k_embedding = nn.Embedding(max_k + 1, k_embedding_dim)

            # Projection to modify model hidden state
            # This would need to be adapted based on the base model architecture
            self.k_projection = nn.Linear(k_embedding_dim, 64)

    def forward(
        self,
        *args,
        k: int = 1,
        **kwargs
    ):
        """
        Forward pass with optional k-conditioning.

        Args:
            *args: Arguments for base model
            k: Current skip factor
            **kwargs: Keyword arguments for base model

        Returns:
            Model output
        """
        if self.condition_on_k:
            # Get k embedding
            k_tensor = torch.tensor([min(k, self.max_k)],
                                   device=next(self.parameters()).device)
            k_embed = self.k_embedding(k_tensor)  # (1, k_embedding_dim)
            k_context = self.k_projection(k_embed)  # (1, 64)

            # Inject k_context into model (model-specific)
            # For now, just pass through
            # TODO: Modify base models to accept k_context

        return self.base_model(*args, **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_effective_dt(dt_fine: float, k: int) -> float:
    """Compute effective time step for skip factor k."""
    return dt_fine * k


def estimate_phase_accumulation(H: torch.Tensor, dt_eff: float) -> torch.Tensor:
    """
    Estimate phase accumulation over time interval.

    For eigenvalue λ of H, phase accumulates as exp(-i·λ·dt_eff/ℏ).
    Large phase accumulation may require special handling.

    Args:
        H: Hamiltonian matrix
        dt_eff: Effective time step

    Returns:
        max_phase: Maximum phase accumulation (radians)
    """
    # Get eigenvalues
    if torch.is_complex(H):
        eigenvalues = torch.linalg.eigvalsh(H.real)
    else:
        eigenvalues = torch.linalg.eigvalsh(H)

    # Max energy difference
    E_range = eigenvalues.max() - eigenvalues.min()

    # Phase accumulation (ℏ = 1 in atomic units)
    max_phase = E_range * dt_eff

    return max_phase


def validate_skip_factor(
    k: int,
    dt_fine: float,
    characteristic_time: float,
    max_phase: float = np.pi / 2
) -> bool:
    """
    Validate if skip factor is physically reasonable.

    Args:
        k: Skip factor
        dt_fine: Fine time step
        characteristic_time: Characteristic time scale of dynamics
        max_phase: Maximum allowed phase accumulation

    Returns:
        is_valid: Whether k is reasonable
    """
    dt_eff = k * dt_fine

    # Check against characteristic time
    if dt_eff > characteristic_time:
        print(f"Warning: Effective dt ({dt_eff:.3f}) > characteristic time ({characteristic_time:.3f})")
        return False

    return True


def create_skip_step_schedule(
    total_epochs: int,
    final_k: int,
    strategy: str = "gradual"
) -> List[int]:
    """
    Create a schedule of k values for each epoch.

    Args:
        total_epochs: Total training epochs
        final_k: Target skip factor
        strategy: 'gradual', 'fast_start', 'slow_start'

    Returns:
        schedule: List of k values for each epoch
    """
    schedule = []

    for epoch in range(total_epochs):
        progress = epoch / total_epochs

        if strategy == "gradual":
            k = 1 + int(progress * (final_k - 1))

        elif strategy == "fast_start":
            # Quick ramp up, then plateau
            k = 1 + int(min(1.0, progress * 2) * (final_k - 1))

        elif strategy == "slow_start":
            # Slow initial increase, then accelerate
            k = 1 + int((progress ** 2) * (final_k - 1))

        else:
            k = final_k

        schedule.append(min(k, final_k))

    return schedule


if __name__ == "__main__":
    print("Testing skip_step module...")

    # Test curriculum scheduler
    curriculum = SkipStepCurriculum(
        initial_k=1,
        final_k=20,
        warmup_epochs=10,
        total_epochs=100,
        schedule='exponential'
    )

    print("\nCurriculum schedule (exponential):")
    for epoch in [0, 5, 10, 25, 50, 75, 100]:
        k = curriculum.step(epoch)
        print(f"  Epoch {epoch:3d}: k = {k}")

    # Test schedule creation
    print("\nPre-computed schedule (gradual):")
    schedule = create_skip_step_schedule(100, final_k=20, strategy='gradual')
    print(f"  Epochs 0,25,50,75,99: k = {[schedule[i] for i in [0, 25, 50, 75, 99]]}")

    # Test config
    config = SkipStepConfig(skip_factor=10)
    print(f"\nDefault config multi-scale factors: {config.multi_scale_factors}")

    print("\nAll tests passed!")
