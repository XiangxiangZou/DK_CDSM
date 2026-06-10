"""Central model artifact registry."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .runtime import DKACModel, DKNModel, DKUCModel, EDMDModel

PREDICTION_MODELS = ("edmd", "dkuc", "dkac", "dkn")
CONTROL_MODELS = ("edmd", "dkuc", "dkac")


def normalize_model_list(
    raw_models: Iterable[str],
    *,
    control_only: bool = False,
) -> list[str]:
    allowed = CONTROL_MODELS if control_only else PREDICTION_MODELS
    values = [item.lower() for item in raw_models]
    if "all" in values:
        return list(allowed)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported models {unknown}; allowed={allowed}")
    return values


def load_prediction_model(
    artifact_root: str | Path,
    model_name: str,
    device: str = "cpu",
):
    root = Path(artifact_root)
    name = model_name.lower()
    if name == "edmd":
        return EDMDModel(root / "edmd")
    if name == "dkuc":
        return DKUCModel(root / "dkuc", root, device)
    if name == "dkac":
        return DKACModel(root / "dkac", root, device)
    if name == "dkn":
        return DKNModel(root / "dkn", root, device)
    raise ValueError(f"Unsupported model: {model_name}")


def load_control_model(
    artifact_root: str | Path,
    model_name: str,
    device: str = "cpu",
):
    name = model_name.lower()
    if name not in CONTROL_MODELS:
        raise ValueError(f"{model_name} is not a linear-control model")
    return load_prediction_model(artifact_root, name, device)
