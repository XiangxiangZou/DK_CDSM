"""Cable-command safety limits independent of a specific robot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensionLimits:
    minimum: float | np.ndarray
    maximum: float | np.ndarray
    max_rate: float | np.ndarray | None = None


def apply_tension_limits(
    command: np.ndarray,
    limits: TensionLimits,
    *,
    previous: np.ndarray | None = None,
    dt: float | None = None,
) -> np.ndarray:
    """Apply absolute and optional rate limits to cable tensions."""
    clipped = np.clip(
        np.asarray(command, dtype=np.float64),
        np.asarray(limits.minimum, dtype=np.float64),
        np.asarray(limits.maximum, dtype=np.float64),
    )
    if limits.max_rate is None:
        return clipped
    if previous is None or dt is None or dt <= 0:
        raise ValueError(
            "previous and positive dt are required for rate limiting"
        )
    delta_limit = np.asarray(limits.max_rate, dtype=np.float64) * dt
    previous_arr = np.asarray(previous, dtype=np.float64)
    return previous_arr + np.clip(
        clipped - previous_arr,
        -delta_limit,
        delta_limit,
    )


def validate_tensions(
    tensions: np.ndarray,
    limits: TensionLimits,
) -> None:
    """Raise when a tension command violates configured absolute limits."""
    values = np.asarray(tensions, dtype=np.float64)
    if np.any(values < np.asarray(limits.minimum)):
        raise ValueError("cable tension below configured minimum")
    if np.any(values > np.asarray(limits.maximum)):
        raise ValueError("cable tension above configured maximum")
