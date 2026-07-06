"""真实机械臂接口占位实现。

当前项目阶段把 MuJoCo 绳驱模型当作实验室真实机械臂使用，因此可运行 plant 是
`MujocoCablePlant`。后续接入实物时，在本文件中实现通信、传感器读取、张力下发
和周期同步，并保持 `plant_interface.CableDrivenPlant` 的方法语义不变。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class RealArmPlant:
    """真实绳驱机械臂适配器模板。

    参数:
        config_path: 实物通信和标定配置文件路径。建议包含关节零位、张力通道映射、
            控制周期、张力上下限和安全急停策略。
        dt: 控制周期，单位 s。

    状态:
        当前只是占位类，所有方法都会抛出 `NotImplementedError`。
    """

    def __init__(self, config_path: str | Path, dt: float) -> None:
        self.config_path = Path(config_path)
        self.dt = float(dt)
        raise NotImplementedError(
            "RealArmPlant 还未接入真实硬件；当前请继续使用 MujocoCablePlant。"
        )

    def set_state(self, q: np.ndarray, dq: np.ndarray) -> None:
        """设置初始状态。实物系统中通常应替换为回零、使能和同步初始反馈。"""
        raise NotImplementedError

    def read_state(self) -> np.ndarray:
        """读取真实编码器/速度估计，返回 `[qa,qb,dqa,dqb]`。"""
        raise NotImplementedError

    def compute_tendon_jacobian(self, eps: float = 1e-6) -> np.ndarray:
        """读取或计算当前 tendon 雅可比。实物可用标定模型或在线几何模型。"""
        raise NotImplementedError

    def apply_cable_tensions(self, tensions: np.ndarray) -> None:
        """下发 8 根绳张力命令，必须包含硬件安全限幅。"""
        raise NotImplementedError

    def step(self) -> None:
        """等待或同步一个控制周期。"""
        raise NotImplementedError

    def torque_dofs(self) -> Tuple[int, int, int, int]:
        """返回与 MuJoCo/几何模型一致的 joint1..joint4 DOF 索引。"""
        raise NotImplementedError
