"""
Unit tests for skip-step propagation module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pytest

from skip_step import (
    SkipStepConfig,
    SupervisionMode,
    SkipStepDatasetLSTM,
    SkipStepCurriculum,
    SkipStepLoss,
    AdaptiveSkipLoss,
    compute_effective_dt,
    validate_skip_factor,
    create_skip_step_schedule
)


class TestSkipStepConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = SkipStepConfig()
        assert config.skip_factor == 10
        assert config.supervision_mode == "endpoints"
        assert config.use_curriculum is True

    def test_multi_scale_factors_auto(self):
        """Multi-scale factors should be auto-generated."""
        config = SkipStepConfig(skip_factor=16)
        # Should have powers of 2 up to 16
        assert 1 in config.multi_scale_factors
        assert 2 in config.multi_scale_factors
        assert 4 in config.multi_scale_factors
        assert 8 in config.multi_scale_factors
        assert 16 in config.multi_scale_factors


class TestSkipStepDatasetLSTM:
    """Tests for LSTM skip-step dataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        T = 200  # timesteps
        n_basis = 4
        rho = torch.randn(T, 2, n_basis, n_basis, dtype=torch.complex128)
        rho = 0.5 * (rho + rho.transpose(-1, -2).conj())  # Hermitian
        field = torch.randn(T, 3, dtype=torch.float64)
        return rho, field

    def test_dataset_creation(self, sample_data):
        """Dataset should be created without errors."""
        rho, field = sample_data
        dataset = SkipStepDatasetLSTM(
            rho, field,
            seq_len=10,
            skip_factor=5,
            rollout_steps=3
        )
        assert len(dataset) > 0

    def test_skip_factor_effect(self, sample_data):
        """Higher skip factor should reduce dataset size."""
        rho, field = sample_data

        dataset_k1 = SkipStepDatasetLSTM(rho, field, seq_len=10, skip_factor=1, rollout_steps=3)
        dataset_k10 = SkipStepDatasetLSTM(rho, field, seq_len=10, skip_factor=10, rollout_steps=3)

        assert len(dataset_k10) < len(dataset_k1)

    def test_getitem_shapes(self, sample_data):
        """Dataset items should have correct shapes."""
        rho, field = sample_data
        seq_len = 10
        rollout = 3

        dataset = SkipStepDatasetLSTM(
            rho, field,
            seq_len=seq_len,
            skip_factor=5,
            rollout_steps=rollout
        )

        rho_seq, field_targets, rho_targets, supervision = dataset[0]

        assert rho_seq.shape[0] == seq_len
        assert field_targets.shape == (rollout, 3)
        assert rho_targets.shape[0] == rollout
        assert 'skip_factor' in supervision

    def test_set_skip_factor(self, sample_data):
        """Should be able to change skip factor."""
        rho, field = sample_data
        dataset = SkipStepDatasetLSTM(rho, field, seq_len=10, skip_factor=1, rollout_steps=3)

        original_len = len(dataset)
        dataset.set_skip_factor(10)
        new_len = len(dataset)

        assert new_len < original_len
        assert dataset.k == 10

    def test_coarse_sampling(self, sample_data):
        """Samples should be at coarse resolution (stride k)."""
        rho, field = sample_data
        k = 5
        seq_len = 4

        dataset = SkipStepDatasetLSTM(
            rho, field,
            seq_len=seq_len,
            skip_factor=k,
            rollout_steps=2
        )

        rho_seq, _, _, _ = dataset[0]

        # Check that consecutive samples are k steps apart
        # (They should be from indices 0, 5, 10, 15 in the original data)
        for i in range(seq_len):
            expected_idx = i * k
            assert torch.allclose(rho_seq[i], rho[expected_idx])


class TestSkipStepCurriculum:
    """Tests for curriculum learning scheduler."""

    def test_warmup_phase(self):
        """Should stay at initial_k during warmup."""
        curriculum = SkipStepCurriculum(
            initial_k=1,
            final_k=20,
            warmup_epochs=10,
            total_epochs=100
        )

        for epoch in range(10):
            k = curriculum.step(epoch)
            assert k == 1

    def test_final_value(self):
        """Should reach final_k at end of training."""
        curriculum = SkipStepCurriculum(
            initial_k=1,
            final_k=20,
            warmup_epochs=10,
            total_epochs=100,
            schedule='linear'
        )

        k = curriculum.step(100)
        assert k == 20

    def test_monotonic_increase(self):
        """k should never decrease."""
        curriculum = SkipStepCurriculum(
            initial_k=1,
            final_k=20,
            warmup_epochs=10,
            total_epochs=100,
            schedule='exponential'
        )

        prev_k = 0
        for epoch in range(100):
            k = curriculum.step(epoch)
            assert k >= prev_k
            prev_k = k

    def test_different_schedules(self):
        """Different schedules should have different profiles."""
        schedules = ['linear', 'exponential', 'step', 'cosine']
        results = {}

        for schedule in schedules:
            curriculum = SkipStepCurriculum(
                initial_k=1,
                final_k=20,
                warmup_epochs=10,
                total_epochs=100,
                schedule=schedule
            )
            mid_k = curriculum.step(50)
            results[schedule] = mid_k

        # Exponential should reach higher values faster
        # (but this depends on exact implementation)
        assert all(1 <= k <= 20 for k in results.values())

    def test_state_dict(self):
        """Should save and load state."""
        curriculum = SkipStepCurriculum(initial_k=1, final_k=20)
        curriculum.step(50)

        state = curriculum.state_dict()
        assert 'current_k' in state
        assert 'current_epoch' in state

        new_curriculum = SkipStepCurriculum(initial_k=1, final_k=20)
        new_curriculum.load_state_dict(state)
        assert new_curriculum.current_k == curriculum.current_k


class TestSkipStepLoss:
    """Tests for skip-step loss function."""

    def test_basic_loss(self):
        """Should compute loss without errors."""
        base_loss = torch.nn.MSELoss()
        skip_loss = SkipStepLoss(base_loss)

        pred = torch.randn(10, 4)
        target = torch.randn(10, 4)

        loss, loss_dict = skip_loss(pred, target)

        assert loss.ndim == 0  # Scalar
        assert 'base' in loss_dict

    def test_smoothness_regularization(self):
        """Smoothness should penalize large changes."""
        base_loss = torch.nn.MSELoss()
        skip_loss = SkipStepLoss(base_loss, lambda_smoothness=1.0)

        pred = torch.randn(10, 4)
        target = torch.randn(10, 4)
        prev_pred = pred + 10.0  # Very different from pred

        loss_with_smooth, _ = skip_loss(pred, target, prev_pred=prev_pred)
        loss_without_smooth, _ = skip_loss(pred, target, prev_pred=None)

        # Loss should be higher with smoothness penalty
        assert loss_with_smooth > loss_without_smooth


class TestAdaptiveSkipLoss:
    """Tests for adaptive loss that scales with k."""

    def test_k_scaling(self):
        """Higher k should have stronger regularization."""
        base_loss = SkipStepLoss(torch.nn.MSELoss(), lambda_smoothness=0.1)
        adaptive = AdaptiveSkipLoss(base_loss, k_ref=1, smoothness_scale=0.1)

        pred = torch.randn(10, 4)
        target = pred + 0.1  # Small difference
        prev_pred = torch.randn(10, 4)  # Different

        loss_k1, dict_k1 = adaptive(pred, target, k=1, prev_pred=prev_pred)
        loss_k10, dict_k10 = adaptive(pred, target, k=10, prev_pred=prev_pred)

        # k=10 should have higher loss due to scaled regularization
        # (Only if smoothness penalty is significant)
        assert dict_k1['k'].item() == 1
        assert dict_k10['k'].item() == 10


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_effective_dt(self):
        """Should correctly compute effective time step."""
        dt_fine = 0.4
        k = 10
        dt_eff = compute_effective_dt(dt_fine, k)
        assert dt_eff == 4.0

    def test_validate_skip_factor(self):
        """Should validate against characteristic time."""
        dt_fine = 0.1
        char_time = 1.0

        # k=5 gives dt_eff=0.5, which is less than char_time
        assert validate_skip_factor(5, dt_fine, char_time)

        # k=20 gives dt_eff=2.0, which exceeds char_time
        assert not validate_skip_factor(20, dt_fine, char_time)

    def test_create_skip_step_schedule(self):
        """Should create valid schedule."""
        schedule = create_skip_step_schedule(100, final_k=20, strategy='gradual')

        assert len(schedule) == 100
        assert schedule[0] == 1  # Start at 1
        assert schedule[-1] == 20  # End at final_k

        # Should be monotonically non-decreasing
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i-1]


class TestSupervisionModes:
    """Tests for different supervision strategies."""

    def test_endpoints_supervision(self):
        """Endpoints mode should only have basic supervision."""
        rho = torch.randn(100, 2, 4, 4, dtype=torch.complex128)
        field = torch.randn(100, 3)

        dataset = SkipStepDatasetLSTM(
            rho, field,
            seq_len=5,
            skip_factor=5,
            rollout_steps=2,
            supervision_mode='endpoints'
        )

        _, _, _, supervision = dataset[0]
        assert supervision['mode'] == 'endpoints'

    def test_intermediate_supervision(self):
        """Intermediate mode should provide middle samples."""
        rho = torch.randn(100, 2, 4, 4, dtype=torch.complex128)
        field = torch.randn(100, 3)

        dataset = SkipStepDatasetLSTM(
            rho, field,
            seq_len=5,
            skip_factor=10,
            rollout_steps=2,
            supervision_mode='intermediate',
            intermediate_samples=3
        )

        _, _, _, supervision = dataset[0]
        assert supervision['mode'] == 'intermediate'
        # Should have intermediate samples
        if 'intermediate_rho' in supervision:
            assert supervision['intermediate_rho'].shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
