"""Metrics for cable tension commands."""

from __future__ import annotations

import numpy as np


def tension_metrics(
    tensions: np.ndarray,
    *,
    minimum: float | np.ndarray | None = None,
    maximum: float | np.ndarray | None = None,
    dt: float | None = None,
) -> dict[str, object]:
    values = np.asarray(tensions, dtype=np.float64)
    result: dict[str, object] = {
        "peak_tension": float(np.max(values)),
        "minimum_tension": float(np.min(values)),
        "mean_tension": float(np.mean(values)),
        "rms_tension": float(np.sqrt(np.mean(values * values))),
    }
    if values.shape[0] > 1:
        differences = np.diff(values, axis=0)
        result["rms_tension_change"] = float(
            np.sqrt(np.mean(differences * differences))
        )
        if dt is not None and dt > 0:
            result["peak_tension_rate"] = float(
                np.max(np.abs(differences)) / dt
            )
    if minimum is not None:
        result["below_minimum_fraction"] = float(
            np.mean(values < np.asarray(minimum))
        )
    if maximum is not None:
        result["at_maximum_fraction"] = float(
            np.mean(values >= np.asarray(maximum))
        )
    return result
