"""DKN 模型 artifact 加载与预测适配器。

DKN 形式：
    `z=[x_n, phi_x(x_n)]`
    `u_hat=g(x_n,u_n)`
    `z_next=A z+B u_hat`

当前 DKN 只进入 `run_03` 的预测评估，不进入 `run_04` 的统一线性 LQR 跟踪控制。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .model_base import BaseRolloutMixin, load_json
    from .normalizers import Normalizer
except ImportError:  # pragma: no cover
    from model_base import BaseRolloutMixin, load_json
    from normalizers import Normalizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdsm_dkn_vs_edmd_prediction_compare import DKNModel as _RootDKNModel  # noqa: E402


class DKNModel(BaseRolloutMixin):
    """DKN 运行期预测模型。"""

    name = "dkn"
    control_mode = "prediction_only_state_dependent_control_encoder"

    def __init__(self, artifact_dir: str | Path, normalizer_dir: str | Path, device: str = "cpu") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg_payload = load_json(self.artifact_dir / "model_config.json")
        cfg = cfg_payload["config"]
        normalizers = load_json(Path(normalizer_dir) / "normalizers.json")
        self.x_normer = Normalizer.from_json(normalizers["x"])
        self.u_normer = Normalizer.from_json(normalizers["u"])

        self.model = _RootDKNModel(
            lift_dim=int(cfg["lift_dim"]),
            hidden=tuple(cfg["hidden"]),
            control_hidden=tuple(cfg["control_hidden"]),
            control_dim_hat=int(cfg["control_dim_hat"]),
            activation=str(cfg["activation"]),
            bound_lift=bool(cfg["bound_lift"]),
        ).to(self.device)
        ckpt = torch.load(self.artifact_dir / "best_dkn.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.A = self.model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.weight.detach().cpu().numpy().astype(np.float64)
        self.C = np.zeros((4, self.model.latent_dim), dtype=np.float64)
        self.C[:, :4] = np.eye(4)
        self.latent_dim = int(self.model.latent_dim)

    @torch.no_grad()
    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        """物理状态升维为 DKN 潜状态。"""
        x_norm = self.x_normer.transform(np.asarray(x_phys, dtype=np.float64).reshape(1, -1)).astype(np.float32)
        z = self.model.encode(torch.from_numpy(x_norm).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    @torch.no_grad()
    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        """用状态相关非线性控制编码推进 DKN 潜状态。"""
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32).reshape(1, -1)).to(self.device)
        u_t = torch.from_numpy(u_norm.astype(np.float32).reshape(1, -1)).to(self.device)
        z_next = self.model.koopman_step(z_t, u_t)
        return z_next.cpu().numpy().reshape(-1).astype(np.float64)

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        """从潜状态前 4 维恢复物理状态。"""
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:4]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]


def load_model(artifact_dir: str | Path, normalizer_dir: str | Path, device: str = "cpu") -> DKNModel:
    """统一加载入口。"""
    return DKNModel(artifact_dir, normalizer_dir, device)
