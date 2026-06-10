"""模型预测/控制统一接口。

四类模型的内部实现不同，但 `run_03_validate_prediction.py` 和
`run_04_tracking_compare.py` 只依赖这里定义的接口。这样后续替换模型实现或
添加新模型时，主流程不需要改动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol

import numpy as np


class PredictiveModel(Protocol):
    """所有预测模型都必须实现的接口。"""

    name: str
    artifact_dir: Path

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        """物理状态 `x=[qa,qb,dqa,dqb]` 升维为模型内部状态 `z`。"""
        ...

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        """用物理控制输入 `u=[tau_a,tau_b]` 推进一步潜空间状态。"""
        ...

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        """从潜空间状态恢复物理状态。"""
        ...

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """从初值 `x0` 和控制序列 `u_seq` 做开环 rollout。"""
        ...


class ControlReadyModel(PredictiveModel, Protocol):
    """可进入 Koopman LQR/MPC 跟踪控制的模型接口。"""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    control_mode: str
    x_normer: object
    u_normer: object

    def recover_control(self, x_phys: np.ndarray, internal_control: np.ndarray) -> np.ndarray:
        """把 LQR/MPC 求得的内部控制量还原为物理关节力矩 `[tau_a,tau_b]`。"""
        ...


class BaseRolloutMixin:
    """通用 rollout 实现，模型适配器只需提供 lift/step/recover。"""

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """执行开环 rollout。

        参数:
            x0: 初始物理状态，形状 `(4,)`。
            u_seq: 物理控制序列，形状 `(T,2)`。

        返回:
            预测状态轨迹，形状 `(T+1,4)`，包含初始状态。
        """
        controls = np.asarray(u_seq, dtype=np.float64)
        pred = np.zeros((controls.shape[0] + 1, 4), dtype=np.float64)
        pred[0] = np.asarray(x0, dtype=np.float64).reshape(4)
        z = self.lift(pred[0])
        for k in range(controls.shape[0]):
            z = self.step_latent(z, controls[k], pred[k])
            pred[k + 1] = self.recover_state(z)
        return pred


def load_json(path: str | Path) -> Dict[str, object]:
    """读取 JSON 文件为字典。"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_load_state(path: str | Path, device):
    """兼容不同 PyTorch 版本的 state_dict 加载。"""
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
