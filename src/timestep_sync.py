"""
Timestep synchronization utilities for prediction with out-of-sample data.

When using a trained model for inference, bootstrap densities and field files
may have different timesteps than the training data. This module provides:
1. Parsing timestamps from field files and density file names
2. Validation that input data matches model's expected dt
3. Automatic resampling/syncing when dt doesn't match
"""

import numpy as np
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import warnings


def parse_field_timestamps(field_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Parse timestamps and field values from a field.dat file.

    Args:
        field_path: Path to field.dat file (format: time Ex Ey Ez)

    Returns:
        timestamps: Array of timestamps
        field_values: Array of field values (N, 3)
        dt: Inferred timestep (median difference between consecutive steps)
    """
    # Handle header line
    with open(field_path, 'r') as f:
        first_line = f.readline().strip()

    # Check if first line is header
    skip_header = first_line.startswith('#') or not first_line[0].isdigit()

    data = np.loadtxt(field_path, skiprows=1 if skip_header else 0)

    if data.ndim == 1:
        # Single row - no timestamps column
        timestamps = np.array([0.0])
        field_values = data.reshape(1, -1)
        if field_values.shape[1] == 4:
            field_values = field_values[:, 1:]
        dt = 0.0
    elif data.shape[1] == 4:
        # Format: time Ex Ey Ez
        timestamps = data[:, 0]
        field_values = data[:, 1:]
        # Compute dt from consecutive differences (use median for robustness)
        if len(timestamps) > 1:
            dts = np.diff(timestamps)
            dt = np.median(dts)
        else:
            dt = 0.0
    elif data.shape[1] == 3:
        # Format: Ex Ey Ez (no timestamps)
        field_values = data
        timestamps = np.arange(len(data), dtype=np.float64)
        dt = 1.0  # Assume unit timestep
        warnings.warn(f"Field file has no timestamp column, assuming dt=1.0")
    else:
        raise ValueError(f"Unexpected field file format: shape {data.shape}")

    return timestamps, field_values, dt


def parse_density_files(density_dir: str, pattern: str = "rho_*.npy") -> Tuple[List[str], List[int]]:
    """
    Parse density files and extract step indices from filenames.

    Args:
        density_dir: Directory containing rho_*.npy files
        pattern: Glob pattern for density files

    Returns:
        sorted_files: List of file paths sorted by step index
        step_indices: List of step indices
    """
    density_path = Path(density_dir)
    files = list(density_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No density files found matching {pattern} in {density_dir}")

    # Extract step indices from filenames (rho_N.npy -> N)
    file_steps = []
    for f in files:
        match = re.search(r'rho_(\d+)\.npy$', f.name)
        if match:
            file_steps.append((int(match.group(1)), str(f)))

    # Sort by step index
    file_steps.sort(key=lambda x: x[0])

    step_indices = [fs[0] for fs in file_steps]
    sorted_files = [fs[1] for fs in file_steps]

    return sorted_files, step_indices


def infer_density_timestamps(
    step_indices: List[int],
    field_timestamps: np.ndarray
) -> np.ndarray:
    """
    Infer timestamps for density files based on step indices and field timestamps.

    Assumes rho_N.npy corresponds to field_timestamps[N].

    Args:
        step_indices: Step indices from density filenames
        field_timestamps: Timestamps from field file

    Returns:
        density_timestamps: Timestamps for each density file
    """
    max_idx = max(step_indices)
    if max_idx >= len(field_timestamps):
        warnings.warn(
            f"Density step indices (max={max_idx}) exceed field timestamps "
            f"(len={len(field_timestamps)}). Using extrapolated timestamps."
        )
        # Extrapolate timestamps
        dt = np.median(np.diff(field_timestamps)) if len(field_timestamps) > 1 else 1.0
        extended_timestamps = np.concatenate([
            field_timestamps,
            field_timestamps[-1] + dt * np.arange(1, max_idx - len(field_timestamps) + 2)
        ])
        return np.array([extended_timestamps[i] for i in step_indices])

    return np.array([field_timestamps[i] for i in step_indices])


def validate_timesteps(
    model_dt: float,
    input_dt: float,
    tolerance: float = 0.01
) -> Tuple[bool, str]:
    """
    Validate that input timestep matches model's expected timestep.

    Args:
        model_dt: Timestep used during model training
        input_dt: Timestep of input data
        tolerance: Relative tolerance for dt matching

    Returns:
        is_valid: True if timesteps match within tolerance
        message: Descriptive message about the validation result
    """
    if model_dt <= 0 or input_dt <= 0:
        return False, f"Invalid timestep: model_dt={model_dt}, input_dt={input_dt}"

    rel_diff = abs(model_dt - input_dt) / model_dt

    if rel_diff <= tolerance:
        return True, f"Timesteps match: model_dt={model_dt:.6f}, input_dt={input_dt:.6f}"
    else:
        return False, (
            f"Timestep mismatch: model expects dt={model_dt:.6f}, "
            f"input has dt={input_dt:.6f} (diff={rel_diff*100:.1f}%)"
        )


def compute_subsample_indices(
    input_timestamps: np.ndarray,
    target_dt: float,
    start_time: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices to subsample input data to match target dt.

    Args:
        input_timestamps: Timestamps of input data
        target_dt: Target timestep to match
        start_time: Starting time for resampling

    Returns:
        indices: Indices into input data that best match target dt grid
        target_times: Target timestamp grid
    """
    # Create target time grid
    end_time = input_timestamps[-1]
    n_target = int((end_time - start_time) / target_dt) + 1
    target_times = start_time + np.arange(n_target) * target_dt

    # Find closest input index for each target time
    indices = []
    for t in target_times:
        if t > input_timestamps[-1]:
            break
        idx = np.argmin(np.abs(input_timestamps - t))
        indices.append(idx)

    return np.array(indices), target_times[:len(indices)]


def sync_bootstrap_to_model(
    bootstrap_files: List[str],
    field_path: str,
    model_dt: float,
    density_dir: Optional[str] = None,
    tolerance: float = 0.01
) -> Dict[str, Union[List[str], np.ndarray, bool, str]]:
    """
    Synchronize bootstrap files and field to match model's expected timestep.

    This is the main entry point for timestep synchronization.

    Args:
        bootstrap_files: List of bootstrap density file paths
        field_path: Path to field.dat file
        model_dt: Timestep the model was trained with
        density_dir: Directory containing all density files (for resampling)
        tolerance: Relative tolerance for dt matching

    Returns:
        dict with keys:
            - 'synced_files': List of synced bootstrap file paths
            - 'synced_field': Synced field values (N, 3)
            - 'synced_timestamps': Timestamps for synced data
            - 'was_resampled': Whether resampling was performed
            - 'message': Status message
    """
    # Parse field file
    field_timestamps, field_values, input_dt = parse_field_timestamps(field_path)

    # Validate timesteps
    is_valid, message = validate_timesteps(model_dt, input_dt, tolerance)

    if is_valid:
        # No resampling needed
        return {
            'synced_files': bootstrap_files,
            'synced_field': field_values,
            'synced_timestamps': field_timestamps,
            'was_resampled': False,
            'message': message
        }

    print(f"WARNING: {message}")
    print(f"Attempting to resample input data to match model dt={model_dt}...")

    # Need to resample - find density directory
    if density_dir is None:
        # Infer from bootstrap files
        if bootstrap_files:
            density_dir = str(Path(bootstrap_files[0]).parent)
        else:
            raise ValueError("No density_dir provided and cannot infer from empty bootstrap_files")

    # Get all available density files
    all_files, all_indices = parse_density_files(density_dir)
    all_timestamps = infer_density_timestamps(all_indices, field_timestamps)

    # Compute resampling indices
    resample_indices, target_times = compute_subsample_indices(
        all_timestamps, model_dt, start_time=all_timestamps[0]
    )

    if len(resample_indices) == 0:
        raise ValueError("No valid indices found for resampling")

    # Select resampled files
    synced_files = [all_files[i] for i in resample_indices]

    # Resample field to match
    field_resample_indices, _ = compute_subsample_indices(
        field_timestamps, model_dt, start_time=field_timestamps[0]
    )
    synced_field = field_values[field_resample_indices]
    synced_timestamps = field_timestamps[field_resample_indices]

    return {
        'synced_files': synced_files,
        'synced_field': synced_field,
        'synced_timestamps': synced_timestamps,
        'was_resampled': True,
        'message': f"Resampled from dt={input_dt:.6f} to dt={model_dt:.6f}: "
                   f"{len(all_files)} -> {len(synced_files)} density files"
    }


def get_model_dt(checkpoint_path: str) -> Optional[float]:
    """
    Extract training dt from model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file

    Returns:
        dt: Training timestep if stored, None otherwise
    """
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        warnings.warn(f"Could not load checkpoint: {e}")
        return None

    # Check various possible locations for dt
    if isinstance(checkpoint, dict):
        # Check top-level
        if 'dt' in checkpoint:
            return checkpoint['dt']
        if 'training_dt' in checkpoint:
            return checkpoint['training_dt']

        # Check config dict
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            cfg = checkpoint['config']
            if 'dt' in cfg:
                return cfg['dt']
            if 'dt_fine' in cfg:
                return cfg['dt_fine']
            if 'training_dt' in cfg:
                return cfg['training_dt']

    return None


def store_model_dt(checkpoint: dict, dt: float) -> dict:
    """
    Add training dt to checkpoint before saving.

    Args:
        checkpoint: Model checkpoint dict
        dt: Training timestep

    Returns:
        Updated checkpoint dict with dt included
    """
    checkpoint['training_dt'] = dt

    # Also store in config if present
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        checkpoint['config']['training_dt'] = dt

    return checkpoint


# Convenience function for command-line testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python timestep_sync.py <field.dat> [density_dir]")
        print("\nParses field file and density directory to report timestep information.")
        sys.exit(1)

    field_path = sys.argv[1]
    density_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Parsing field file: {field_path}")
    timestamps, field_values, dt = parse_field_timestamps(field_path)
    print(f"  Timestamps: {len(timestamps)} entries")
    print(f"  Time range: {timestamps[0]:.4f} to {timestamps[-1]:.4f}")
    print(f"  Inferred dt: {dt:.6f}")

    if density_dir:
        print(f"\nParsing density directory: {density_dir}")
        files, indices = parse_density_files(density_dir)
        print(f"  Found {len(files)} density files")
        print(f"  Step indices: {indices[0]} to {indices[-1]}")

        density_timestamps = infer_density_timestamps(indices, timestamps)
        print(f"  Density time range: {density_timestamps[0]:.4f} to {density_timestamps[-1]:.4f}")
