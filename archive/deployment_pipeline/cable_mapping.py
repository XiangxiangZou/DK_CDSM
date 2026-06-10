"""绳驱关节力矩到 8 根绳张力的映射模块。

模块职责：
1. 维护 MuJoCo XML 中 8 根绳和 8 个 motor actuator 的固定顺序。
2. 将控制器输出的关节力矩 `tau=[tau_a, tau_b]` 转换为可下发的绳张力。
3. 保持接口与后续真实机械臂一致：上层只关心 `tau -> cable_tensions`。

注意：
- 当前默认不启用绳张力上限，仅保留预紧张力 `F_PRELOAD`。
- 如果后续实物系统有明确张力上限，只需要设置 `f_max` 或替换本模块。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# 8 根绳在 XML、数据集 `cable_ctrl` 和控制下发中的统一顺序。
CABLE_NAMES = (
    "cable11",
    "cable12",
    "cable13",
    "cable14",
    "cable21",
    "cable22",
    "cable23",
    "cable24",
)
# 与 CABLE_NAMES 一一对应的 MuJoCo actuator 名称。
ACTUATOR_NAMES = tuple("winch_c" + name[len("cable") :] for name in CABLE_NAMES)

# 每个主动关节由两组正向/反向拮抗绳共同产生等效关节力矩。
IDX_F1P = (0, 2)
IDX_F1M = (1, 3)
IDX_F2P = (4, 6)
IDX_F2M = (5, 7)

# 每根绳的预紧力，单位 N。真实机械臂部署时应与硬件预紧策略一致。
F_PRELOAD = 20.0

# 绳张力上限，单位 N。None 表示不裁剪上限；实物部署前应按硬件能力设置。
F_MAX_CABLE: Optional[float] = None


def solve_antagonistic_pair(
    m_p: float,
    m_m: float,
    tau_des: float,
    f_pre: float,
    f_max: Optional[float],
) -> Tuple[float, float, float]:
    """把单个关节的期望力矩分配到一对拮抗绳。

    参数:
        m_p: 正向绳组对该关节的力矩臂系数。
        m_m: 反向绳组对该关节的力矩臂系数。
        tau_des: 该关节的期望力矩，单位 Nm。
        f_pre: 每根绳保留的预紧力，单位 N。
        f_max: 单根绳允许的最大张力，单位 N；None 或 inf 表示不裁剪。

    返回:
        `(f_pos, f_neg, residual)`，分别是正向绳张力、反向绳张力和映射残差。
    """
    tau_base = (m_p + m_m) * f_pre
    tau_eff = tau_des - tau_base
    eps = 1e-12

    # u_max 是在预紧力基础上允许增加的张力增量。
    if f_max is None or not np.isfinite(f_max):
        u_max = float("inf")
    else:
        u_max = max(float(f_max) - float(f_pre), 0.0)

    # 在“只拉正向绳”“只拉反向绳”“只保留预紧力”中选残差最小者。
    candidates = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > eps:
        u = min(max(tau_eff / m_p, 0.0), u_max)
        candidates.append((u, 0.0, abs(tau_eff - m_p * u), u))
    if abs(m_m) > eps:
        u = min(max(tau_eff / m_m, 0.0), u_max)
        candidates.append((0.0, u, abs(tau_eff - m_m * u), u))

    u_p, u_m, residual, _ = min(candidates, key=lambda item: (item[2], item[3]))
    return f_pre + u_p, f_pre + u_m, residual


def cable_antagonistic_map(
    tau_a_des: float,
    tau_b_des: float,
    tendon_jacobian: np.ndarray,
    dof_j1: int,
    dof_j2: int,
    dof_j3: int,
    dof_j4: int,
    f_pre: float = F_PRELOAD,
    f_max: Optional[float] = F_MAX_CABLE,
) -> np.ndarray:
    """把两个主动关节力矩映射为 8 根绳张力。

    参数:
        tau_a_des: 第一主动关节 qa 的期望力矩，单位 Nm。
        tau_b_des: 第二主动关节 qb 的期望力矩，单位 Nm。
        tendon_jacobian: tendon length 对所有 qpos 的有限差分雅可比，形状 `(8, nv)`。
        dof_j1: joint1 在 MuJoCo `qvel/qfrc` 中的自由度索引。
        dof_j2: joint2 在 MuJoCo `qvel/qfrc` 中的自由度索引。
        dof_j3: joint3 在 MuJoCo `qvel/qfrc` 中的自由度索引。
        dof_j4: joint4 在 MuJoCo `qvel/qfrc` 中的自由度索引。
        f_pre: 预紧张力，单位 N。
        f_max: 单根绳张力上限，单位 N；None 或 inf 表示不裁剪。

    返回:
        形状 `(8,)` 的张力数组，顺序与 `CABLE_NAMES` 完全一致。
    """
    # joint1 与 joint2 同步驱动 qa，joint3 与 joint4 同步驱动 qb。
    a = tendon_jacobian[:, dof_j1] + tendon_jacobian[:, dof_j2]
    b = tendon_jacobian[:, dof_j3] + tendon_jacobian[:, dof_j4]

    m_p1 = a[IDX_F1P[0]] + a[IDX_F1P[1]]
    m_m1 = a[IDX_F1M[0]] + a[IDX_F1M[1]]
    m_p2 = b[IDX_F2P[0]] + b[IDX_F2P[1]]
    m_m2 = b[IDX_F2M[0]] + b[IDX_F2M[1]]

    f1p, f1m, _ = solve_antagonistic_pair(m_p1, m_m1, tau_a_des, f_pre, f_max)
    f2p, f2m, _ = solve_antagonistic_pair(m_p2, m_m2, tau_b_des, f_pre, f_max)

    tensions = np.empty(8, dtype=np.float64)
    tensions[list(IDX_F1P)] = f1p
    tensions[list(IDX_F1M)] = f1m
    tensions[list(IDX_F2P)] = f2p
    tensions[list(IDX_F2M)] = f2m
    return tensions
