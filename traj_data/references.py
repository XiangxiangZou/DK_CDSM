"""受 XML 物理限制约束的随机关节空间参考轨迹生成模块。

提供在机械臂关节空间内生成随机、平滑参考轨迹的工具函数，
包括关节限位收缩、初始状态采样、轨迹生成以及软限位保护力矩计算。
"""

from __future__ import annotations

import numpy as np

# 软限位保护区的边界余量比例（在安全限位内侧各收缩 8%）
SOFT_LIMIT_MARGIN: float = 0.08


def shrink_limits(q_limits: np.ndarray, ratio: float) -> np.ndarray:
    """将关节限位范围按比例向内收缩。

    以关节限位的中心为基准，按给定比例缩小其半宽，返回一个更窄的安全范围。
    常用于在物理限位内部定义一个保守的"软限位"工作区间。

    参数
    ----------
    q_limits : np.ndarray
        形状为 (2, 2) 的关节限位数组，每行对应一个关节的 [下限, 上限]。
    ratio : float
        收缩比例，取值范围 (0, 1]。1.0 表示不收缩（保持原始限位），
        0.5 表示将范围收缩到原始的一半。

    返回
    -------
    np.ndarray
        形状为 (2, 2) 的收缩后限位数组，每行 [下限, 上限]。
    """
    limits = np.asarray(q_limits, dtype=np.float64).reshape(2, 2)
    if not np.all(np.isfinite(limits)):
        raise ValueError("关节限位必须是有限值")
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("限位收缩比例必须在 (0, 1] 范围内")
    center, half_width = _limits_stats(limits, ratio)
    return np.column_stack([center - half_width, center + half_width])


def _limits_stats(limits: np.ndarray, ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """计算关节限位的中心位置和半宽（按比例缩放）。

    参数
    ----------
    limits : np.ndarray
        形状为 (2, 2) 的限位数组，[下限, 上限]。
    ratio : float
        半宽缩放比例，默认 1.0。

    返回
    -------
    tuple[np.ndarray, np.ndarray]
        (center, half_width)：各关节中心位置和缩放后的半宽，均为形状 (2,) 的数组。
    """
    center = limits.mean(axis=1)
    half_width = 0.5 * (limits[:, 1] - limits[:, 0]) * float(ratio)
    return center, half_width


def sample_initial_state(
    rng: np.random.Generator,
    safe_limits: np.ndarray,
    *,
    q_init_ratio: float,
    dq_init_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在安全限位内随机采样初始关节位置和速度。

    参数
    ----------
    rng : np.random.Generator
        NumPy 随机数生成器。
    safe_limits : np.ndarray
        形状为 (2, 2) 的安全限位数组，[下限, 上限]。
    q_init_ratio : float
        初始位置采样范围比例。在安全限位中心附近按此比例缩小的区域内均匀采样。
    dq_init_range : float
        初始速度的采样范围，速度在 [-dq_init_range, +dq_init_range] 内均匀采样。

    返回
    -------
    tuple[np.ndarray, np.ndarray]
        (q0, dq0)：初始关节位置和速度，均为形状 (2,) 的 float64 数组。
    """
    # 利用收缩后的限位计算采样中心与半宽，避免重复计算
    init_limits = shrink_limits(safe_limits, q_init_ratio)
    center = init_limits.mean(axis=1)
    half_width = 0.5 * (init_limits[:, 1] - init_limits[:, 0])
    q0 = rng.uniform(center - half_width, center + half_width)
    # 在 [-dq_init_range, +dq_init_range] 内均匀采样初始速度
    dq0 = rng.uniform(-dq_init_range, dq_init_range, size=2)
    return q0.astype(np.float64), dq0.astype(np.float64)


def random_joint_reference(
    rng: np.random.Generator,
    safe_limits: np.ndarray,
    *,
    steps: int,
    dt: float,
    q_start: np.ndarray,
    waypoint_count: int,
    range_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在安全关节限位内生成平滑、大范围的随机参考轨迹。

    通过在限位范围内随机采样若干路径点，并使用 Hermite 三次插值
    （smoothstep 平滑函数）连接各点，生成连续平滑的关节参考轨迹。

    参数
    ----------
    rng : np.random.Generator
        NumPy 随机数生成器。
    safe_limits : np.ndarray
        形状为 (2, 2) 的安全限位，[下限, 上限]。
    steps : int
        轨迹的总时间步数，必须为正整数。
    dt : float
        每步的时间间隔（秒）。
    q_start : np.ndarray
        轨迹的起始关节位置，形状 (2,)。会被裁剪到安全限位内。
    waypoint_count : int
        路径点数量（包含起点），至少为 2。
    range_ratio : float
        路径点采样范围比例，取值 (0, 1]。控制路径点可以在多大比例的
        安全限位范围内随机分布。

    返回
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (q_ref, dq_ref, t)：参考位置 (steps, 2)、参考速度 (steps, 2)
        和时间向量 (steps,)。
    """
    if steps <= 0:
        raise ValueError("步数必须为正整数")
    # 路径点数量限制在 [2, steps] 之间
    waypoint_count = min(max(2, int(waypoint_count)), steps)
    range_ratio = float(range_ratio)
    if not 0.0 < range_ratio <= 1.0:
        raise ValueError("参考轨迹范围比例必须在 (0, 1] 内")

    # 复用 shrink_limits 计算路径点采样范围
    ref_limits = shrink_limits(safe_limits, range_ratio)

    # 在时间轴上均匀分布路径点的步索引
    waypoint_steps = np.linspace(0, steps - 1, waypoint_count, dtype=np.float64)
    # 在采样范围内随机生成路径点
    waypoints = rng.uniform(ref_limits[:, 0], ref_limits[:, 1], size=(waypoint_count, 2))
    # 将第一个路径点固定为起始位置（裁剪到范围内）
    waypoints[0] = np.clip(
        np.asarray(q_start, dtype=np.float64).reshape(2), ref_limits[:, 0], ref_limits[:, 1]
    )

    q_ref = np.zeros((steps, 2), dtype=np.float64)
    last_stop = 0  # 显式追踪最后一段的终点，避免依赖循环变量的隐式作用域

    # 逐段使用 smoothstep 函数进行平滑插值
    # smoothstep: s(t) = 3t² - 2t³，一阶导数在端点为零，保证速度连续
    for segment in range(waypoint_count - 1):
        start = int(round(waypoint_steps[segment]))
        stop = int(round(waypoint_steps[segment + 1]))
        stop = max(stop, start + 1)  # 确保每段至少有两个点
        last_stop = stop
        alpha = np.linspace(0.0, 1.0, stop - start + 1)  # 归一化时间 [0, 1]
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep 插值权重
        q_ref[start : stop + 1] = (
            (1.0 - smooth[:, None]) * waypoints[segment]
            + smooth[:, None] * waypoints[segment + 1]
        )

    # 填充最后一段之后的剩余步数
    if last_stop + 1 < steps:
        q_ref[last_stop + 1 :] = waypoints[-1]

    # 确保生成的轨迹在安全限位内（smoothstep 凸组合理论上保证在 ref_limits 内，
    # 此处作为防御性裁剪）
    q_ref = np.clip(q_ref, safe_limits[:, 0], safe_limits[:, 1])
    # 通过数值微分计算参考速度
    dq_ref = np.gradient(q_ref, float(dt), axis=0)
    # 生成时间向量
    t = np.arange(steps, dtype=np.float64) * float(dt)
    return q_ref, dq_ref, t


def soft_limit_guard(
    q: np.ndarray,
    dq: np.ndarray,
    safe_limits: np.ndarray,
    *,
    kp: float,
    kd: float,
    margin: float = SOFT_LIMIT_MARGIN,
) -> np.ndarray:
    """计算软限位保护力矩，将关节推离软限位边界。

    当关节位置进入边界区域时，施加虚拟弹簧-阻尼力矩将其推回安全区间。
    此力矩不包含在 PD 控制量中，作为独立的保护项使用。

    参数
    ----------
    q : np.ndarray
        当前关节位置，形状 (2,)。
    dq : np.ndarray
        当前关节速度，形状 (2,)。
    safe_limits : np.ndarray
        形状为 (2, 2) 的安全限位，[下限, 上限]。
    kp : float
        保护力矩的比例增益（弹簧刚度），正值。
    kd : float
        保护力矩的微分增益（阻尼系数），正值。
    margin : float
        边界余量比例，默认 0.08（即 8%）。在安全限位内侧各收缩该比例
        作为缓冲区。

    返回
    -------
    np.ndarray
        形状为 (2,) 的保护力矩向量。对安全区间内的关节，对应分量为零。
    """
    q = np.asarray(q, dtype=np.float64).reshape(2)
    dq = np.asarray(dq, dtype=np.float64).reshape(2)
    # 计算软限位边界：在安全限位内侧各收缩 margin 比例作为缓冲区
    span = safe_limits[:, 1] - safe_limits[:, 0]
    low = safe_limits[:, 0] + margin * span   # 下界留余量
    high = safe_limits[:, 1] - margin * span  # 上界留余量

    # 向量化计算保护力矩，避免逐关节循环
    tau = np.zeros(2, dtype=np.float64)
    # 低于下界的关节：施加正向力矩（弹簧拉力）+ 阻尼
    below = q < low
    tau = np.where(below, kp * (low - q) - kd * dq, tau)
    # 超过上界的关节：施加负向力矩（弹簧推力）+ 阻尼
    above = q > high
    tau = np.where(above, -kp * (q - high) - kd * dq, tau)
    return tau
