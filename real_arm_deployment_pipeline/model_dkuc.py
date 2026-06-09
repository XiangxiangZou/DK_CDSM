"""DKUC 模型 artifact 加载与预测/控制适配器。

DKUC 形式：
    `z=[x_n, phi_x(x_n)]`
    `z_next=A z+B u_n`

DKUC 可直接进入统一 Koopman LQR/MPC，因为 B 矩阵对应标准化物理控制 `u_n`。
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdsm_dkuc_vs_dkac_tracking_control import DKUCModel as _RootDKUCModel  # noqa: E402


class DKUCModel(BaseRolloutMixin):
    """DKUC 运行期模型。

    参数:
        artifact_dir: `run_02` 生成的 `artifacts/<run>/dkuc` 目录。
        normalizer_dir: 包含共享 `normalizers.json` 的本次训练根目录。
        device: PyTorch 推理设备。
    """

    name = "dkuc"
    control_mode = "z_next=A z+B u_norm"

    def __init__(self, artifact_dir: str | Path, normalizer_dir: str | Path, device: str = "cpu") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg_payload = load_json(self.artifact_dir / "model_config.json")
        cfg = cfg_payload["config"]
        normalizers = load_json(Path(normalizer_dir) / "normalizers.json")
        self.x_normer = Normalizer.from_json(normalizers["x"])
        self.u_normer = Normalizer.from_json(normalizers["u"])

        self.model = _RootDKUCModel(
            lift_dim=int(cfg["lift_dim"]),
            hidden=tuple(cfg["hidden"]),
            activation=str(cfg["activation"]),
            bound_lift=float(cfg["bound_lift"]),
        ).to(self.device)
        state = torch_load_state(self.artifact_dir / "best_dkuc.pt", self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.A = self.model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.weight.detach().cpu().numpy().astype(np.float64)
        self.C = np.zeros((4, self.model.latent_dim), dtype=np.float64)
        self.C[:, :4] = np.eye(4)
        self.latent_dim = int(self.model.latent_dim)

    @torch.no_grad()
    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        """物理状态升维为 DKUC 潜状态。"""
        x_norm = self.x_normer.transform(np.asarray(x_phys, dtype=np.float64).reshape(1, -1)).astype(np.float32)
        z = self.model.lift(torch.from_numpy(x_norm).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        """执行 `z_next=A z+B u_n`。"""
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        return self.A @ np.asarray(z, dtype=np.float64).reshape(-1) + self.B @ u_norm

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        """从潜状态前 4 维恢复物理状态。"""
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:4]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]

    def recover_control(self, x_phys: np.ndarray, internal_control: np.ndarray) -> np.ndarray:
        """DKUC 的 LQR 内部控制就是标准化关节力矩 `u_n`。"""
        return self.u_normer.inverse(np.asarray(internal_control, dtype=np.float64).reshape(1, -1))[0]


def load_model(artifact_dir: str | Path, normalizer_dir: str | Path, device: str = "cpu") -> DKUCModel:
    """统一加载入口。"""
    return DKUCModel(artifact_dir, normalizer_dir, device)
