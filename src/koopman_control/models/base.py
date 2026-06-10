"""Interfaces shared by prediction and control-ready Koopman models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Protocol

import numpy as np


class PredictiveModel(Protocol):
    """Minimum interface required by prediction evaluation."""

    name: str
    artifact_dir: Path

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        ...

    def step_latent(
        self,
        z: np.ndarray,
        u_phys: np.ndarray,
        x_phys: np.ndarray | None = None,
    ) -> np.ndarray:
        ...

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        ...

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        ...


class ControlReadyModel(PredictiveModel, Protocol):
    """Additional contract required by linear Koopman tracking control."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    control_mode: str
    x_normer: object
    u_normer: object

    def recover_control(
        self,
        x_phys: np.ndarray,
        internal_control: np.ndarray,
    ) -> np.ndarray:
        ...


class BaseRolloutMixin:
    """Default open-loop rollout implementation."""

    state_dim: int = 4

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        controls = np.asarray(u_seq, dtype=np.float64)
        pred = np.zeros(
            (controls.shape[0] + 1, self.state_dim),
            dtype=np.float64,
        )
        pred[0] = np.asarray(x0, dtype=np.float64).reshape(self.state_dim)
        z = self.lift(pred[0])
        for k, control in enumerate(controls):
            z = self.step_latent(z, control, pred[k])
            pred[k + 1] = self.recover_state(z)
        return pred


def load_json(path: str | Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON document."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def torch_load_state(path: str | Path, device):
    """Load a state dictionary across supported PyTorch versions."""
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
