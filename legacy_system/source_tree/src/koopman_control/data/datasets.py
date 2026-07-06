"""Dimension-aware dataset loading, validation, and splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def validate_dataset(
    arrays: Dict[str, np.ndarray],
    source: str | Path = "<memory>",
    *,
    state_dim: int | None = None,
    control_dim: int | None = None,
) -> None:
    """Validate the trajectory relationship between states and inputs."""
    if "states" not in arrays or "inputs" not in arrays:
        raise ValueError(f"{source} must contain states and inputs")
    states = np.asarray(arrays["states"])
    inputs = np.asarray(arrays["inputs"])
    if states.ndim != 3:
        raise ValueError(f"{source}: states must have shape (traj, steps+1, dim)")
    if inputs.ndim != 3:
        raise ValueError(f"{source}: inputs must have shape (traj, steps, dim)")
    if state_dim is not None and states.shape[2] != state_dim:
        raise ValueError(
            f"{source}: states last dimension must be {state_dim}, "
            f"got {states.shape[2]}"
        )
    if control_dim is not None and inputs.shape[2] != control_dim:
        raise ValueError(
            f"{source}: inputs last dimension must be {control_dim}, "
            f"got {inputs.shape[2]}"
        )
    if states.shape[0] != inputs.shape[0]:
        raise ValueError(f"{source}: trajectory counts do not match")
    if states.shape[1] != inputs.shape[1] + 1:
        raise ValueError(f"{source}: states must contain one more step than inputs")


def load_dataset(
    path: str | Path,
    *,
    state_dim: int | None = None,
    control_dim: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load a compressed trajectory dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    validate_dataset(
        arrays,
        dataset_path,
        state_dim=state_dim,
        control_dim=control_dim,
    )
    return arrays


def save_dataset(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save a compressed trajectory dataset."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **{key: np.asarray(value) for key, value in arrays.items()},
    )


def split_train_val(
    arrays: Dict[str, np.ndarray],
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    """Split a dataset by complete trajectories."""
    validate_dataset(arrays)
    n_traj = arrays["states"].shape[0]
    if n_traj < 2:
        raise ValueError("At least two trajectories are required")
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n_traj)
    n_val = min(max(1, int(round(n_traj * val_ratio))), n_traj - 1)
    val_indices = np.sort(permutation[:n_val])
    train_indices = np.sort(permutation[n_val:])

    def take(indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            key: np.asarray(value)[indices].copy()
            for key, value in arrays.items()
        }

    metadata = {
        "mode": "split_single_dataset",
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
    }
    return take(train_indices), take(val_indices), metadata
