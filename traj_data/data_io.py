"""Small data-saving helpers for trajectory collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def validate_dataset(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    states = np.asarray(arrays["states"])
    inputs = np.asarray(arrays["inputs"])
    if states.ndim != 3 or states.shape[2] != 4:
        raise ValueError("states must have shape (traj, steps+1, 4)")
    if inputs.ndim != 3 or inputs.shape[2] != 2:
        raise ValueError("inputs must have shape (traj, steps, 2)")
    if states.shape[0] != inputs.shape[0] or states.shape[1] != inputs.shape[1] + 1:
        raise ValueError("states and inputs trajectory/step dimensions do not match")

    finite = {name: bool(np.all(np.isfinite(value))) for name, value in arrays.items()}
    finite_for_required = bool(finite["states"] and finite["inputs"] and finite["cable_ctrl"])
    return {
        "finite_required_arrays": finite_for_required,
        "finite_by_array": finite,
        "state_min": np.nanmin(states, axis=(0, 1)),
        "state_max": np.nanmax(states, axis=(0, 1)),
        "peak_abs_tau": float(np.nanmax(np.abs(inputs))),
        "peak_cable_tension": float(np.nanmax(np.asarray(arrays["cable_ctrl"]))),
        "trajectory_count": int(states.shape[0]),
        "steps": int(inputs.shape[1]),
    }


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = validate_dataset(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: np.asarray(value) for key, value in arrays.items()})
    return summary
