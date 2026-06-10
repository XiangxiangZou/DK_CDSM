"""真实机械臂/MuJoCo 被控对象统一接口定义。

本文件只定义上层流程需要的最小接口，不绑定具体硬件。当前可运行实现是
`mujoco_plant.MujocoCablePlant`；后续接入实验室真实机械臂时，应实现同样
的方法语义，尽量不改采数、预测评估和控制器主流程。
"""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np


class CableDrivenPlant(Protocol):
    """绳驱空间机械臂被控对象接口。

    接口约定:
        状态统一为 `x=[qa, qb, dqa, dqb]`。
        关节力矩统一为 `u=[tau_a, tau_b]`。
        绳张力统一为 8 维数组，顺序与 `cable_mapping.CABLE_NAMES` 一致。
    """

    dt: float
    q_limits: np.ndarray

    def set_state(self, q: np.ndarray, dq: np.ndarray) -> None:
        """设置或初始化主动关节状态。实物机械臂可实现为回零/同步初始状态。"""
        ...

    def read_state(self) -> np.ndarray:
        """读取当前真实反馈状态 `[qa, qb, dqa, dqb]`。"""
        ...

    def compute_tendon_jacobian(self, eps: float = 1e-6) -> np.ndarray:
        """计算或读取当前构型的 tendon 雅可比，形状 `(8,nv)`。"""
        ...

    def apply_cable_tensions(self, tensions: np.ndarray) -> None:
        """下发 8 根绳张力，单位 N。"""
        ...

    def step(self) -> None:
        """执行一个控制周期。MuJoCo 中是 `mj_step`，实物中是等待下个周期。"""
        ...

    def torque_dofs(self) -> Tuple[int, int, int, int]:
        """返回 joint1..joint4 的 DOF 索引，供力矩到绳张力映射使用。"""
        ...
