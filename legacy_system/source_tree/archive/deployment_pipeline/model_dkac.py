"""DKAC 模型 artifact 加载与预测/控制适配器。

DKAC 形式：
    `z=[x_n, phi_x(x_n)]`
    `v=G(x_n)u_n`
    `z_next=A z+B v`

闭环控制时 LQR 求解的是内部控制 `v`，执行前必须通过 `pinv(G(x))` 还原为
标准化物理力矩 `u_n`，再反标准化为 `tau_a,tau_b`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .model_base import BaseRolloutMixin, load_json, torch_load_state
    from .normalizers import Normalizer
except ImportError:  # pragma: no cover
    from model_base import BaseRolloutMixin, load_json, torch_load_state
    from normalizers import Normalizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdsm_dkac_vs_edmd_tracking_control import DKACModel as _RootDKACModel  # noqa: E402


class DKACModel(BaseRolloutMixin):
    """DKAC 运行期模型。"""

    name = "dkac"
    control_mode = "z_next=A z+B v, v=G(x_norm)u_norm"

    def __init__(self, artifact_dir: str | Path, normalizer_dir: str | Path, device: str = "cpu") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg_payload = load_json(self.artifact_dir / "model_config.json")
        cfg = cfg_payload["config"]
        normalizers = load_json(Path(normalizer_dir) / "normalizers.json")
        self.x_normer = Normalizer.from_json(normalizers["x"])
        self.u_normer = Normalizer.from_json(normalizers["u"])

        self.model = _RootDKACModel(
            lift_dim=int(cfg["lift_dim"]),
            hidden=tuple(cfg["hidden"]),
            control_hidden=tuple(cfg["control_hidden"]),
            control_dim_hat=int(cfg["control_dim_hat"]),
            activation=str(cfg["activation"]),
            bound_lift=float(cfg["bound_lift"]),
            identity_control_bias=bool(cfg["identity_control_bias"]),
        ).to(self.device)
        state = torch_load_state(self.artifact_dir / "best_dkac.pt", self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.A = self.model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.weight.detach().cpu().numpy().astype(np.float64)
        self.C = np.zeros((4, self.model.latent_dim), dtype=np.float64)
        self.C[:, :4] = np.eye(4)
        self.latent_dim = int(self.model.latent_dim)
        self.control_dim_hat = int(self.model.control_dim_hat)

    @torch.no_grad()
    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        """物理状态升维为 DKAC 潜状态。"""
        x_norm = self.x_normer.transform(np.asarray(x_phys, dtype=np.float64).reshape(1, -1)).astype(np.float32)
        z = self.model.lift(torch.from_numpy(x_norm).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    @torch.no_grad()
    def control_matrix(self, x_phys: np.ndarray) -> np.ndarray:
        """计算当前状态下的 `G(x_n)` 控制编码矩阵。"""
        x_norm = self.x_normer.transform(np.asarray(x_phys, dtype=np.float64).reshape(1, -1)).astype(np.float32)
        G = self.model.control_matrix(torch.from_numpy(x_norm).to(self.device))
        return G.cpu().numpy()[0].astype(np.float64)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        """用物理力矩先计算 `v=G(x)u_n`，再执行 `A z+B v`。"""
        if x_phys is None:
            x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:4]
            x_phys = self.x_normer.inverse(x_norm.reshape(1, -1))[0]
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        v = self.control_matrix(x_phys) @ u_norm
        return self.A @ np.asarray(z, dtype=np.float64).reshape(-1) + self.B @ v

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        """从潜状态前 4 维恢复物理状态。"""
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:4]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]

    def recover_control(self, x_phys: np.ndarray, internal_control: np.ndarray) -> np.ndarray:
        """把 LQR 求出的内部控制 `v` 反解为物理关节力矩。"""
        G = self.control_matrix(x_phys)
        u_norm = np.linalg.pinv(G, rcond=1e-6) @ np.asarray(internal_control, dtype=np.float64).reshape(-1)
        return self.u_normer.inverse(u_norm.reshape(1, -1))[0]


def load_model(artifact_dir: str | Path, normalizer_dir: str | Path, device: str = "cpu") -> DKACModel:
    """统一加载入口。"""
    return DKACModel(artifact_dir, normalizer_dir, device)
