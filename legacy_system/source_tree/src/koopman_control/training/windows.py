"""Trajectory-window construction and sampling."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_windows(
    states: np.ndarray,
    inputs: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build overlapping multi-step windows from complete trajectories."""
    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(inputs, dtype=np.float64)
    n_traj, n_times, _ = x.shape
    n_steps = u.shape[1]
    if n_steps != n_times - 1:
        raise ValueError("inputs must contain one fewer step than states")
    if window < 1 or window > n_steps:
        raise ValueError(
            f"window must be in [1, {n_steps}], got {window}"
        )
    count_per_trajectory = n_steps - window + 1
    x_windows = np.empty(
        (n_traj * count_per_trajectory, window + 1, x.shape[2]),
        dtype=np.float64,
    )
    u_windows = np.empty(
        (n_traj * count_per_trajectory, window, u.shape[2]),
        dtype=np.float64,
    )
    cursor = 0
    for trajectory in range(n_traj):
        for start in range(count_per_trajectory):
            x_windows[cursor] = x[
                trajectory,
                start : start + window + 1,
            ]
            u_windows[cursor] = u[
                trajectory,
                start : start + window,
            ]
            cursor += 1
    return x_windows, u_windows
