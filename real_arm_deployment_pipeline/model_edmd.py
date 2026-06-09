"""EDMD 模型 artifact 加载与预测适配器。

EDMD 是固定 RBF 字典 Koopman baseline：
    `z=[x_n, rbf(x_n)]`
    `z_next=A z+B u_n`

该模型既可用于 `run_03` 的预测评估，也可用于 `run_04` 的 Koopman LQR 跟踪控制。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .model_base import BaseRolloutMixin
    from .normalizers import Normalizer
except ImportError:  # pragma: no cover
    from model_base import BaseRolloutMixin
    from normalizers import Normalizer


class EDMDModel(BaseRolloutMixin):
    """EDMD 运行期模型。

    参数:
        artifact_dir: `run_02` 生成的 `artifacts/<run>/edmd` 目录。

    必需文件:
        `edmd_model.npz`，其中包含 RBF centers、sigma、A、B 和标准化参数。
    """

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
            self.cond_number = float(np.asarray(data["cond_number"]).reshape(-1)[0])
            self.x_normer = Normalizer(np.asarray(data["x_mean"]), np.asarray(data["x_std"]))
            self.u_normer = Normalizer(np.asarray(data["u_mean"]), np.asarray(data["u_std"]))
        self.C = np.zeros((4, self.A.shape[0]), dtype=np.float64)
        self.C[:, :4] = np.eye(4)
        self.latent_dim = int(self.A.shape[0])

    def _lift_norm(self, x_norm: np.ndarray) -> np.ndarray:
        """对标准化状态计算 RBF 字典升维。"""
        x_norm = np.atleast_2d(np.asarray(x_norm, dtype=np.float64))
        diff = x_norm[:, None, :] - self.centers[None, :, :]
        sqdist = np.sum(diff * diff, axis=2)
        rbf = np.exp(-0.5 * sqdist / (self.sigma * self.sigma))
        return np.hstack([x_norm, rbf])

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        """物理状态升维为 EDMD 字典状态。"""
        x_norm = self.x_normer.transform(np.asarray(x_phys, dtype=np.float64).reshape(1, -1))
        return self._lift_norm(x_norm).reshape(-1)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        """执行 `z_next=A z+B u_n`。"""
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        return self.A @ np.asarray(z, dtype=np.float64).reshape(-1) + self.B @ u_norm

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        """从 `z` 的前 4 维标准化状态恢复物理状态。"""
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:4]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]

    def recover_control(self, x_phys: np.ndarray, internal_control: np.ndarray) -> np.ndarray:
        """EDMD 的内部控制就是标准化关节力矩 `u_n`。"""
        return self.u_normer.inverse(np.asarray(internal_control, dtype=np.float64).reshape(1, -1))[0]


def load_model(artifact_dir: str | Path, device: str = "cpu") -> EDMDModel:
    """统一加载入口；`device` 参数为兼容其它模型，EDMD 不使用。"""
    return EDMDModel(artifact_dir)
