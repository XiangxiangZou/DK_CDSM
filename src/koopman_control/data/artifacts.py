"""Small helpers for portable experiment artifacts."""

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
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(
            jsonable(payload),
            handle,
            indent=2,
            ensure_ascii=False,
        )


def save_normalizers(path: str | Path, x_normer, u_normer) -> None:
    save_json(
        path,
        {"x": x_normer.to_json(), "u": u_normer.to_json()},
    )


def save_runtime_matrices(
    path: str | Path,
    *,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    control_mode: str,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        A=np.asarray(A, dtype=np.float64),
        B=np.asarray(B, dtype=np.float64),
        C=np.asarray(C, dtype=np.float64),
        control_mode=np.array([control_mode]),
    )
