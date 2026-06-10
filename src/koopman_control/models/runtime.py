"""Artifact-backed runtime adapters for EDMD, DKUC, DKAC, and DKN."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from koopman_control.data.normalization import Normalizer
from koopman_control.models.base import (
    BaseRolloutMixin,
    load_json,
    torch_load_state,
)


class EDMDModel(BaseRolloutMixin):
    """RBF-dictionary EDMD runtime."""

    name = "edmd"
    control_mode = "z_next=A z+B u_norm"

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        model_path = self.artifact_dir / "edmd_model.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"EDMD artifact not found: {model_path}")
        with np.load(model_path, allow_pickle=False) as data:
            self.centers = np.asarray(data["centers"], dtype=np.float64)
            self.sigma = float(np.asarray(data["sigma"]).reshape(-1)[0])
            self.A = np.asarray(data["A"], dtype=np.float64)
            self.B = np.asarray(data["B"], dtype=np.float64)
            self.cond_number = float(
                np.asarray(data["cond_number"]).reshape(-1)[0]
            )
            self.x_normer = Normalizer(
                np.asarray(data["x_mean"]),
                np.asarray(data["x_std"]),
            )
            self.u_normer = Normalizer(
                np.asarray(data["u_mean"]),
                np.asarray(data["u_std"]),
            )
        self.state_dim = int(self.x_normer.mean.size)
        self.control_dim = int(self.u_normer.mean.size)
        self.C = np.zeros(
            (self.state_dim, self.A.shape[0]),
            dtype=np.float64,
        )
        self.C[:, : self.state_dim] = np.eye(self.state_dim)
        self.latent_dim = int(self.A.shape[0])

    def _lift_norm(self, x_norm: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(x_norm, dtype=np.float64))
        diff = values[:, None, :] - self.centers[None, :, :]
        sqdist = np.sum(diff * diff, axis=2)
        rbf = np.exp(-0.5 * sqdist / (self.sigma * self.sigma))
        return np.hstack([values, rbf])

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(
            np.asarray(x_phys, dtype=np.float64).reshape(1, -1)
        )
        return self._lift_norm(x_norm).reshape(-1)

    def step_latent(
        self,
        z: np.ndarray,
        u_phys: np.ndarray,
        x_phys: np.ndarray | None = None,
    ) -> np.ndarray:
        del x_phys
        u_norm = self.u_normer.transform(
            np.asarray(u_phys, dtype=np.float64).reshape(1, -1)
        )[0]
        return self.A @ np.asarray(z).reshape(-1) + self.B @ u_norm

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[
            : self.state_dim
        ]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]

    def recover_control(
        self,
        x_phys: np.ndarray,
        internal_control: np.ndarray,
    ) -> np.ndarray:
        del x_phys
        return self.u_normer.inverse(
            np.asarray(internal_control, dtype=np.float64).reshape(1, -1)
        )[0]


class _TorchRuntimeBase(BaseRolloutMixin):
    """Shared artifact loading behavior for neural models."""

    checkpoint_name: str

    def _load_common(
        self,
        artifact_dir: str | Path,
        normalizer_dir: str | Path,
        device: str,
    ) -> dict:
        import torch

        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        config_payload = load_json(self.artifact_dir / "model_config.json")
        normalizers = load_json(Path(normalizer_dir) / "normalizers.json")
        self.x_normer = Normalizer.from_json(normalizers["x"])
        self.u_normer = Normalizer.from_json(normalizers["u"])
        self.state_dim = int(self.x_normer.mean.size)
        self.control_dim = int(self.u_normer.mean.size)
        return config_payload["config"]

    def _finish_common(self) -> None:
        self.A = (
            self.model.A.weight.detach().cpu().numpy().astype(np.float64)
        )
        self.B = (
            self.model.B.weight.detach().cpu().numpy().astype(np.float64)
        )
        self.C = np.zeros(
            (self.state_dim, self.model.latent_dim),
            dtype=np.float64,
        )
        self.C[:, : self.state_dim] = np.eye(self.state_dim)
        self.latent_dim = int(self.model.latent_dim)

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[
            : self.state_dim
        ]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]


class DKUCModel(_TorchRuntimeBase):
    """DKUC artifact runtime."""

    name = "dkuc"
    control_mode = "z_next=A z+B u_norm"

    def __init__(
        self,
        artifact_dir: str | Path,
        normalizer_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        import torch

        from koopman_control.models.networks import DKUCNetwork

        config = self._load_common(artifact_dir, normalizer_dir, device)
        self.model = DKUCNetwork(
            lift_dim=int(config["lift_dim"]),
            hidden=tuple(config["hidden"]),
            activation=str(config["activation"]),
            bound_lift=float(config["bound_lift"]),
            state_dim=self.state_dim,
            control_dim=self.control_dim,
        ).to(self.device)
        state = torch_load_state(
            self.artifact_dir / "best_dkuc.pt",
            self.device,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self._finish_common()
        self._torch = torch

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(
            np.asarray(x_phys).reshape(1, -1)
        ).astype(np.float32)
        with self._torch.no_grad():
            z = self.model.lift(
                self._torch.from_numpy(x_norm).to(self.device)
            )
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def step_latent(
        self,
        z: np.ndarray,
        u_phys: np.ndarray,
        x_phys: np.ndarray | None = None,
    ) -> np.ndarray:
        del x_phys
        u_norm = self.u_normer.transform(
            np.asarray(u_phys).reshape(1, -1)
        )[0]
        return self.A @ np.asarray(z).reshape(-1) + self.B @ u_norm

    def recover_control(
        self,
        x_phys: np.ndarray,
        internal_control: np.ndarray,
    ) -> np.ndarray:
        del x_phys
        return self.u_normer.inverse(
            np.asarray(internal_control).reshape(1, -1)
        )[0]


class DKACModel(_TorchRuntimeBase):
    """DKAC artifact runtime."""

    name = "dkac"
    control_mode = "z_next=A z+B v, v=G(x_norm)u_norm"

    def __init__(
        self,
        artifact_dir: str | Path,
        normalizer_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        import torch

        from koopman_control.models.networks import DKACNetwork

        config = self._load_common(artifact_dir, normalizer_dir, device)
        self.model = DKACNetwork(
            lift_dim=int(config["lift_dim"]),
            hidden=tuple(config["hidden"]),
            control_hidden=tuple(config["control_hidden"]),
            control_dim_hat=int(config["control_dim_hat"]),
            activation=str(config["activation"]),
            bound_lift=float(config["bound_lift"]),
            identity_control_bias=bool(config["identity_control_bias"]),
            state_dim=self.state_dim,
            control_dim=self.control_dim,
        ).to(self.device)
        state = torch_load_state(
            self.artifact_dir / "best_dkac.pt",
            self.device,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self._finish_common()
        self.control_dim_hat = int(self.model.control_dim_hat)
        self._torch = torch

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(
            np.asarray(x_phys).reshape(1, -1)
        ).astype(np.float32)
        with self._torch.no_grad():
            z = self.model.lift(
                self._torch.from_numpy(x_norm).to(self.device)
            )
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def control_matrix(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(
            np.asarray(x_phys).reshape(1, -1)
        ).astype(np.float32)
        with self._torch.no_grad():
            matrix = self.model.control_matrix(
                self._torch.from_numpy(x_norm).to(self.device)
            )
        return matrix.cpu().numpy()[0].astype(np.float64)

    def step_latent(
        self,
        z: np.ndarray,
        u_phys: np.ndarray,
        x_phys: np.ndarray | None = None,
    ) -> np.ndarray:
        if x_phys is None:
            x_phys = self.recover_state(z)
        u_norm = self.u_normer.transform(
            np.asarray(u_phys).reshape(1, -1)
        )[0]
        internal_control = self.control_matrix(x_phys) @ u_norm
        return (
            self.A @ np.asarray(z).reshape(-1)
            + self.B @ internal_control
        )

    def recover_control(
        self,
        x_phys: np.ndarray,
        internal_control: np.ndarray,
    ) -> np.ndarray:
        matrix = self.control_matrix(x_phys)
        u_norm = np.linalg.pinv(matrix, rcond=1e-6) @ np.asarray(
            internal_control
        ).reshape(-1)
        return self.u_normer.inverse(u_norm.reshape(1, -1))[0]


class DKNModel(_TorchRuntimeBase):
    """Prediction-only DKN artifact runtime."""

    name = "dkn"
    control_mode = "prediction_only_state_dependent_control_encoder"

    def __init__(
        self,
        artifact_dir: str | Path,
        normalizer_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        import torch

        from koopman_control.models.networks import DKNNetwork

        config = self._load_common(artifact_dir, normalizer_dir, device)
        self.model = DKNNetwork(
            lift_dim=int(config["lift_dim"]),
            hidden=tuple(config["hidden"]),
            control_hidden=tuple(config["control_hidden"]),
            control_dim_hat=int(config["control_dim_hat"]),
            activation=str(config["activation"]),
            bound_lift=bool(config["bound_lift"]),
            state_dim=self.state_dim,
            control_dim=self.control_dim,
        ).to(self.device)
        try:
            checkpoint = torch.load(
                self.artifact_dir / "best_dkn.pt",
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.artifact_dir / "best_dkn.pt",
                map_location=self.device,
            )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self._finish_common()
        self.control_dim_hat = int(self.model.control_dim_hat)
        self._torch = torch

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(
            np.asarray(x_phys).reshape(1, -1)
        ).astype(np.float32)
        with self._torch.no_grad():
            z = self.model.encode(
                self._torch.from_numpy(x_norm).to(self.device)
            )
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def step_latent(
        self,
        z: np.ndarray,
        u_phys: np.ndarray,
        x_phys: np.ndarray | None = None,
    ) -> np.ndarray:
        del x_phys
        u_norm = self.u_normer.transform(
            np.asarray(u_phys).reshape(1, -1)
        )[0]
        z_tensor = self._torch.from_numpy(
            np.asarray(z, dtype=np.float32).reshape(1, -1)
        ).to(self.device)
        u_tensor = self._torch.from_numpy(
            u_norm.astype(np.float32).reshape(1, -1)
        ).to(self.device)
        with self._torch.no_grad():
            z_next = self.model.koopman_step(z_tensor, u_tensor)
        return z_next.cpu().numpy().reshape(-1).astype(np.float64)
