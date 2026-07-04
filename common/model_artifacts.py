"""Load control-ready models from prediction output folders."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.io_utils import PROJECT_ROOT

PREDICTION_ROOT = PROJECT_ROOT / "prediction"
if str(PREDICTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICTION_ROOT))

DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "control" / "model_selections.json"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _state_readout(latent_dim: int, state_dim: int) -> np.ndarray:
    C = np.zeros((state_dim, latent_dim), dtype=np.float64)
    C[:, :state_dim] = np.eye(state_dim)
    return C


class ControlModelAdapter:
    """Add LQR/MPC control-facing methods to prediction model artifacts."""

    def __init__(self, model, model_name: str) -> None:
        self.model = model
        self.name = model_name
        self.A = np.asarray(model.A, dtype=np.float64)
        self.B = np.asarray(model.B, dtype=np.float64)
        self.x_normer = model.x_normer
        self.u_normer = model.u_normer
        self.state_dim = int(model.state_dim)
        self.control_dim = int(getattr(model, "control_dim", self.u_normer.mean.size))
        self.C = _state_readout(self.A.shape[0], self.state_dim)

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        return self.model.lift(x_phys)

    def recover_control(self, x_phys: np.ndarray, internal_control: np.ndarray) -> np.ndarray:
        internal = np.asarray(internal_control, dtype=np.float64).reshape(-1)
        if self.name in {"edmd", "dkuc"}:
            return self.u_normer.inverse(internal.reshape(1, -1))[0]
        if self.name == "dkac":
            u_norm = np.linalg.pinv(self.control_matrix(x_phys), rcond=1e-6) @ internal
            return self.u_normer.inverse(u_norm.reshape(1, -1))[0]
        raise ValueError(f"{self.name} is not a supported control model")

    def control_matrix(self, x_phys: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "control_matrix"):
            raise ValueError(f"{self.name} has no state-dependent control matrix")
        return self.model.control_matrix(x_phys)


def infer_model_name(artifact_dir: str | Path) -> str:
    path = Path(artifact_dir)
    if (path / "edmd_model.npz").exists():
        return "edmd"
    if (path / "best_dkuc.pt").exists():
        return "dkuc"
    if (path / "best_dkac.pt").exists():
        return "dkac"
    raise ValueError(f"Cannot infer prediction model type from {path}")


def load_prediction_control_model(
    artifact_dir: str | Path,
    model_name: str = "auto",
    device: str = "cuda",
) -> ControlModelAdapter:
    """Load EDMD/DKUC/DKAC directly from one prediction output directory."""
    path = Path(artifact_dir)
    name = infer_model_name(path) if model_name == "auto" else model_name.lower()
    if name == "edmd":
        from edmd_prediction import EDMDModel

        return ControlModelAdapter(EDMDModel(path), "edmd")
    if name == "dkuc":
        from dkuc_prediction import DKUCModel

        return ControlModelAdapter(DKUCModel(path, device), "dkuc")
    if name == "dkac":
        from dkac_prediction import DKACModel

        return ControlModelAdapter(DKACModel(path, device), "dkac")
    raise ValueError("Control supports prediction artifacts for edmd, dkuc, and dkac")


def resolve_model_selection(
    *,
    controller: str,
    artifact_dir: str | Path = "",
    model_name: str = "auto",
    model_key: str = "",
    model_config: str | Path = DEFAULT_MODEL_CONFIG,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve which prediction model artifact a controller should use."""
    if str(artifact_dir):
        resolved = str(_project_path(artifact_dir))
        resolved_name = infer_model_name(resolved) if model_name == "auto" else model_name
        return (
            resolved,
            resolved_name,
            {
                "source": "cli",
                "controller": controller,
                "model_key": "",
                "artifact_dir": str(artifact_dir),
                "method": resolved_name,
            },
        )

    config_path = _project_path(model_config)
    payload = _load_json(config_path)
    key = model_key or payload.get("controller_defaults", {}).get(controller, "")
    if not key:
        raise ValueError(
            f"No model_key provided and no default configured for controller '{controller}'"
        )
    models = payload.get("models", {})
    if key not in models:
        raise ValueError(f"Model key '{key}' is not defined in {config_path}")
    entry = dict(models[key])
    method = str(entry.get("method", "")).lower()
    if not method:
        raise ValueError(f"Model key '{key}' has no method")
    if model_name != "auto" and model_name.lower() != method:
        raise ValueError(
            f"CLI model '{model_name}' does not match selection method '{method}'"
        )
    artifact = entry.get("artifact_dir", "")
    if not artifact:
        raise ValueError(f"Model key '{key}' has no artifact_dir")
    return (
        str(_project_path(artifact)),
        method,
        {
            "source": "model_config",
            "controller": controller,
            "model_config": str(config_path),
            "model_key": key,
            "entry": entry,
        },
    )
