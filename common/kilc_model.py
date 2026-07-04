"""Continuous-DKUC runtime used by the KILC controller."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from common.prediction_utils import Normalizer, load_json

STATE_DIM = 4
CONTROL_DIM = 2


def _torch_load_state(path: str | Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _activation_layer(name: str):
    import torch.nn as nn

    values = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
    if name not in values:
        raise ValueError(f"Unsupported activation: {name}")
    return values[name]


class MLP:
    """Factory wrapper to delay torch imports until a KILC model is loaded."""

    def __new__(cls, widths: Sequence[int], activation: str = "elu"):
        import torch.nn as nn

        layers: list[nn.Module] = []
        act = _activation_layer(activation)
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index != len(widths) - 2:
                layers.append(act())
        return nn.Sequential(*layers)


def _continuous_dkuc_network_class():
    import torch
    import torch.nn as nn

    class ContinuousDKUCNetwork(nn.Module):
        """State-embedded continuous-time DKUC network."""

        def __init__(
            self,
            *,
            lift_dim: int,
            hidden: Sequence[int],
            activation: str,
            bound_lift: float,
            state_dim: int = STATE_DIM,
            control_dim: int = CONTROL_DIM,
        ) -> None:
            super().__init__()
            self.state_dim = int(state_dim)
            self.control_dim = int(control_dim)
            self.lift_dim = int(lift_dim)
            self.latent_dim = self.state_dim + self.lift_dim
            self.encoder = MLP(
                (self.state_dim, *tuple(hidden), self.lift_dim),
                activation,
            )
            self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)
            self.bound_lift = float(bound_lift)
            with torch.no_grad():
                self.A.weight.zero_()
                self.B.weight.zero_()
                rows = min(self.state_dim, self.control_dim)
                self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

        def lift(self, x_norm):
            features = self.encoder(x_norm)
            if self.bound_lift > 0.0:
                features = self.bound_lift * torch.tanh(features / self.bound_lift)
            return torch.cat([x_norm, features], dim=-1)

        def state_from_latent(self, z):
            return z[..., : self.state_dim]

        def derivative(self, z, u_norm):
            return self.A(z) + self.B(u_norm)

    return ContinuousDKUCNetwork


class ContinuousDKUCModel:
    """Artifact-backed continuous-time DKUC runtime for KILC."""

    name = "dkuc"
    variant = "continuous"
    control_mode = "zdot=A_c z+B_c u_norm"

    def __init__(
        self,
        artifact_dir: str | Path,
        normalizer_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        import torch

        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        config = load_json(self.artifact_dir / "model_config.json")["config"]
        normalizers = load_json(Path(normalizer_dir) / "normalizers.json")
        self.x_normer = Normalizer.from_json(normalizers["x"])
        self.u_normer = Normalizer.from_json(normalizers["u"])
        self.state_dim = int(self.x_normer.mean.size)
        self.control_dim = int(self.u_normer.mean.size)
        Network = _continuous_dkuc_network_class()
        self.model = Network(
            lift_dim=int(config["lift_dim"]),
            hidden=tuple(config["hidden"]),
            activation=str(config["activation"]),
            bound_lift=float(config["bound_lift"]),
            state_dim=self.state_dim,
            control_dim=self.control_dim,
        ).to(self.device)
        state = _torch_load_state(
            self.artifact_dir / "best_dkuc_continuous.pt",
            self.device,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.A = self.model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.weight.detach().cpu().numpy().astype(np.float64)
        self.C = np.zeros((self.state_dim, self.model.latent_dim), dtype=np.float64)
        self.C[:, : self.state_dim] = np.eye(self.state_dim)
        self.latent_dim = int(self.model.latent_dim)
        self._torch = torch

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(
            np.asarray(x_phys).reshape(1, -1)
        ).astype(np.float32)
        with self._torch.no_grad():
            z = self.model.lift(self._torch.from_numpy(x_norm).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def derivative_latent(self, z: np.ndarray, u_phys: np.ndarray) -> np.ndarray:
        u_norm = self.u_normer.transform(np.asarray(u_phys).reshape(1, -1))[0]
        return self.A @ np.asarray(z).reshape(-1) + self.B @ u_norm

    def recover_control_delta(self, u_norm_delta: np.ndarray) -> np.ndarray:
        return np.asarray(u_norm_delta, dtype=np.float64).reshape(-1) * self.u_normer.std

    def recover_control(self, x_phys: np.ndarray, internal_control: np.ndarray) -> np.ndarray:
        del x_phys
        return self.recover_control_delta(internal_control)


def load_continuous_dkuc_model(
    artifact_root: str | Path,
    device: str = "cuda",
) -> ContinuousDKUCModel:
    root = Path(artifact_root)
    artifact_dir = root / "dkuc_continuous"
    if not artifact_dir.exists() and (root / "best_dkuc_continuous.pt").exists():
        artifact_dir = root
    normalizer_dir = root if (root / "normalizers.json").exists() else root.parent
    return ContinuousDKUCModel(artifact_dir, normalizer_dir, device)
