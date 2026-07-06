"""离线数据采集协议。

本模块只负责“如何产生采集动作并记录数据”，不关心后续训练的是 DKUC、
DKAC、EDMD 还是 DKN。所有模型后续都读取同一份 `dataset.npz`，从而保证
预测误差和跟踪控制对比使用一致的数据来源。

输出数据约定：
- `states`: `(traj, steps+1, 4)`，状态顺序 `[qa, qb, dqa, dqb]`。
- `inputs`: `(traj, steps, 2)`，控制输入 `[tau_a, tau_b]`，单位 Nm。
- `q_ref`: `(traj, steps, 2)`，PD 模式下的参考角度；random 模式填 NaN。
- `dq_ref`: `(traj, steps, 2)`，PD 模式下的参考角速度；random 模式填 NaN。
- `cable_ctrl`: `(traj, steps, 8)`，实际下发的 8 根绳张力，单位 N。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from cable_robotics.tension_allocator import (
    F_MAX_CABLE,
    F_PRELOAD,
    allocate_antagonistic_tensions,
)
from cdsm.constants import (
    CABLE_NAMES,
    CONTROL_DIM,
    STATE_DIM,
    make_tension_layout,
)
from cdsm.plants.base import CDSMPlant


@dataclass(frozen=True)
class CollectionConfig:
    """采集参数集合。

    参数:
        traj_count: 采集轨迹条数。
        steps: 每条轨迹的控制步数。
        dt: 控制周期/仿真步长，单位 s。
        seed: 随机种子，用于初始状态、随机激励和参考轨迹复现。
        q_limit_ratio: 使用 XML 关节限位的比例。例如 0.90 表示只使用中心 90% 范围。
        q_init_ratio: 初始关节角采样范围占安全关节范围的比例。
        dq_init_range: 初始角速度采样范围，`[-dq_init_range, dq_init_range]`，单位 rad/s。
        random_tau: random 模式下分段常值随机力矩幅值上限，单位 Nm。
        random_hold_steps: random 模式下随机力矩保持的步数。
        random_damping: random 模式下附加阻尼项系数，力矩中加入 `-random_damping*dq`。
        boundary_kp: 接近关节安全边界时的回拉比例增益。
        boundary_kd: 接近关节安全边界时的回拉阻尼增益。
        amp_min: PDCtrl 模式随机多正弦参考的最小幅值，单位 rad。
        amp_max: PDCtrl 模式随机多正弦参考的最大幅值，单位 rad。
        omega_min: PDCtrl 模式随机多正弦参考的最小角频率，单位 rad/s。
        omega_max: PDCtrl 模式随机多正弦参考的最大角频率，单位 rad/s。
        kp_a: PDCtrl 模式 qa 关节 PD 比例增益。
        kp_b: PDCtrl 模式 qb 关节 PD 比例增益。
        kd_a: PDCtrl 模式 qa 关节 PD 微分增益。
        kd_b: PDCtrl 模式 qb 关节 PD 微分增益。
        tau_max: 两种模式最终关节力矩裁剪上限，单位 Nm。
        f_preload: 每根绳的预紧张力，单位 N。
        f_max_cable: 单根绳张力上限，单位 N；None 表示不裁剪上限。
    """

    traj_count: int = 120
    steps: int = 300
    dt: float = 0.01
    seed: int = 42
    q_limit_ratio: float = 0.90
    q_init_ratio: float = 0.65
    dq_init_range: float = 0.4
    random_tau: float = 35.0
    random_hold_steps: int = 8
    random_damping: float = 0.8
    boundary_kp: float = 80.0
    boundary_kd: float = 6.0
    amp_min: float = 0.15
    amp_max: float = 0.55
    omega_min: float = 0.7
    omega_max: float = 2.3
    kp_a: float = 80.0
    kp_b: float = 70.0
    kd_a: float = 8.0
    kd_b: float = 7.0
    tau_max: float = 80.0
    f_preload: float = F_PRELOAD
    f_max_cable: float | None = F_MAX_CABLE


@dataclass(frozen=True)
class SineRefParams:
    """单个关节的双正弦参考轨迹参数。"""

    a1: float
    a2: float
    w1: float
    w2: float
    phi2: float


def _safe_q_limits(plant: CDSMPlant, ratio: float) -> np.ndarray:
    """根据 XML 关节限位收缩出安全采集范围。

    参数:
        plant: MuJoCo 被控对象，提供原始主动关节限位。
        ratio: 使用原始限位范围的比例，避免采集时频繁触碰硬限位。

    返回:
        形状 `(2,2)` 的安全角度范围，单位 rad。
    """
    q_limits = np.asarray(plant.q_limits, dtype=np.float64).copy()
    if not np.all(np.isfinite(q_limits)):
        raise ValueError("Active joints must have finite limits for real-arm-safe collection.")
    center = np.mean(q_limits, axis=1)
    half_width = 0.5 * (q_limits[:, 1] - q_limits[:, 0]) * float(ratio)
    return np.column_stack([center - half_width, center + half_width])


def _sample_initial_state(
    rng: np.random.RandomState,
    q_limits: np.ndarray,
    q_init_ratio: float,
    dq_init_range: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """采样一条轨迹的初始状态。

    参数:
        rng: 随机数发生器。
        q_limits: 安全关节角范围，形状 `(2,2)`。
        q_init_ratio: 初始角度采样范围占安全范围的比例。
        dq_init_range: 初始角速度最大绝对值，单位 rad/s。

    返回:
        `(q0, dq0)`，分别为初始角度和角速度。
    """
    center = np.mean(q_limits, axis=1)
    half_width = 0.5 * (q_limits[:, 1] - q_limits[:, 0]) * float(q_init_ratio)
    q0 = rng.uniform(center - half_width, center + half_width)
    dq0 = rng.uniform(-dq_init_range, dq_init_range, size=CONTROL_DIM)
    return q0.astype(np.float64), dq0.astype(np.float64)


def _sample_sine_params(rng: np.random.RandomState, cfg: CollectionConfig) -> SineRefParams:
    """为一个关节随机生成双正弦参考轨迹参数。"""
    return SineRefParams(
        a1=float(rng.uniform(cfg.amp_min, cfg.amp_max)),
        a2=float(rng.uniform(0.5 * cfg.amp_min, 0.8 * cfg.amp_max)),
        w1=float(rng.uniform(cfg.omega_min, cfg.omega_max)),
        w2=float(rng.uniform(1.3 * cfg.omega_min, 1.8 * cfg.omega_max)),
        phi2=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def _eval_sine_ref(params: SineRefParams, t: float) -> Tuple[float, float]:
    """计算双正弦参考在时刻 t 的角度和角速度。"""
    q = params.a1 * np.sin(params.w1 * t) + params.a2 * np.sin(params.w2 * t + params.phi2)
    dq = params.a1 * params.w1 * np.cos(params.w1 * t) + params.a2 * params.w2 * np.cos(
        params.w2 * t + params.phi2
    )
    return float(q), float(dq)


def _limit_repel_tau(
    q: np.ndarray,
    dq: np.ndarray,
    q_limits: np.ndarray,
    tau: np.ndarray,
    boundary_kp: float,
    boundary_kd: float,
) -> np.ndarray:
    """在接近安全边界时覆盖原始力矩，使关节回到安全范围内。

    参数:
        q: 当前主动关节角度 `[qa, qb]`，单位 rad。
        dq: 当前主动关节角速度 `[dqa, dqb]`，单位 rad/s。
        q_limits: 安全关节角范围，形状 `(2,2)`。
        tau: 原始期望关节力矩，单位 Nm。
        boundary_kp: 边界回拉比例增益。
        boundary_kd: 边界回拉阻尼增益。

    返回:
        处理后的关节力矩，单位 Nm。
    """
    out = np.asarray(tau, dtype=np.float64).reshape(CONTROL_DIM).copy()
    span = q_limits[:, 1] - q_limits[:, 0]
    soft_lo = q_limits[:, 0] + 0.08 * span
    soft_hi = q_limits[:, 1] - 0.08 * span
    for j in range(CONTROL_DIM):
        if q[j] < soft_lo[j]:
            out[j] = boundary_kp * (soft_lo[j] - q[j]) - boundary_kd * dq[j]
        elif q[j] > soft_hi[j]:
            out[j] = -boundary_kp * (q[j] - soft_hi[j]) - boundary_kd * dq[j]
    return out


def _tau_to_cable(plant: CDSMPlant, tau: np.ndarray, cfg: CollectionConfig) -> np.ndarray:
    """把关节力矩转换为当前构型下的 8 根绳张力。"""
    jac = plant.compute_tendon_jacobian()
    dof_j1, dof_j2, dof_j3, dof_j4 = plant.torque_dofs()
    layout = make_tension_layout(
        dof_j1,
        dof_j2,
        dof_j3,
        dof_j4,
    )
    tensions, _ = allocate_antagonistic_tensions(
        np.asarray(tau, dtype=np.float64),
        jac,
        layout,
        f_pre=cfg.f_preload,
        f_max=cfg.f_max_cable,
    )
    return tensions


def collect_random_excitation(
    plant: CDSMPlant,
    cfg: CollectionConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """用随机激励采集离线数据。

    采集方式:
        在关节安全范围内施加分段常值随机力矩，并叠加速度阻尼和边界回拉。
        这种模式不依赖参考轨迹，适合提高状态-控制空间覆盖度。

    参数:
        plant: MuJoCo 被控对象。
        cfg: 采集参数，random 相关参数会在本模式中生效。

    返回:
        `(arrays, meta)`。arrays 可直接保存为 `dataset.npz`，meta 可保存为 JSON。
    """
    rng = np.random.RandomState(cfg.seed)
    safe_limits = _safe_q_limits(plant, cfg.q_limit_ratio)
    n, steps = cfg.traj_count, cfg.steps

    states = np.zeros((n, steps + 1, STATE_DIM), dtype=np.float64)
    inputs = np.zeros((n, steps, CONTROL_DIM), dtype=np.float64)
    cable_ctrl = np.zeros((n, steps, len(CABLE_NAMES)), dtype=np.float64)
    q_ref = np.full((n, steps, CONTROL_DIM), np.nan, dtype=np.float64)
    dq_ref = np.full((n, steps, CONTROL_DIM), np.nan, dtype=np.float64)

    for i in range(n):
        # 每条轨迹从安全范围内随机初始状态开始。
        q0, dq0 = _sample_initial_state(rng, safe_limits, cfg.q_init_ratio, cfg.dq_init_range)
        plant.set_state(q0, dq0)
        states[i, 0] = plant.read_state()
        tau_hold = rng.uniform(-cfg.random_tau, cfg.random_tau, size=CONTROL_DIM)

        for k in range(steps):
            # 分段常值随机力矩，减少高频抖动，模拟可实现的激励输入。
            if k % max(1, cfg.random_hold_steps) == 0:
                tau_hold = rng.uniform(-cfg.random_tau, cfg.random_tau, size=CONTROL_DIM)
            x = plant.read_state()
            q, dq = x[:2], x[2:]
            tau = tau_hold - cfg.random_damping * dq
            tau = _limit_repel_tau(q, dq, safe_limits, tau, cfg.boundary_kp, cfg.boundary_kd)
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)

            tensions = _tau_to_cable(plant, tau, cfg)
            plant.apply_cable_tensions(tensions)
            plant.step()

            states[i, k + 1] = plant.read_state()
            inputs[i, k] = tau
            cable_ctrl[i, k] = tensions

    meta = {
        "mode": "random",
        "description": "piecewise-random joint torque excitation inside active joint limits",
        "config": asdict(cfg),
        "safe_q_limits": safe_limits.tolist(),
        "state_order": ["qa", "qb", "dqa", "dqb"],
        "input_order": ["tau_a", "tau_b"],
        "cable_order": list(CABLE_NAMES),
    }
    return {"states": states, "inputs": inputs, "q_ref": q_ref, "dq_ref": dq_ref, "cable_ctrl": cable_ctrl}, meta


def collect_pd_control(
    plant: CDSMPlant,
    cfg: CollectionConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """用 PD 控制器跟踪随机参考轨迹采集离线数据。

    采集方式:
        每条轨迹随机生成两个关节的多正弦参考，在关节安全范围内裁剪参考角度，
        再通过 PD 控制器生成 `tau=[tau_a,tau_b]`。

    参数:
        plant: MuJoCo 被控对象。
        cfg: 采集参数，PDCtrl 相关参数会在本模式中生效。

    返回:
        `(arrays, meta)`。arrays 可直接保存为 `dataset.npz`，meta 可保存为 JSON。
    """
    rng = np.random.RandomState(cfg.seed)
    safe_limits = _safe_q_limits(plant, cfg.q_limit_ratio)
    n, steps = cfg.traj_count, cfg.steps

    states = np.zeros((n, steps + 1, STATE_DIM), dtype=np.float64)
    inputs = np.zeros((n, steps, CONTROL_DIM), dtype=np.float64)
    cable_ctrl = np.zeros((n, steps, len(CABLE_NAMES)), dtype=np.float64)
    q_ref_hist = np.zeros((n, steps, CONTROL_DIM), dtype=np.float64)
    dq_ref_hist = np.zeros((n, steps, CONTROL_DIM), dtype=np.float64)

    kp = np.array([cfg.kp_a, cfg.kp_b], dtype=np.float64)
    kd = np.array([cfg.kd_a, cfg.kd_b], dtype=np.float64)

    for i in range(n):
        # 每条轨迹使用新的随机初始状态和新的随机参考轨迹。
        q0, dq0 = _sample_initial_state(rng, safe_limits, cfg.q_init_ratio, cfg.dq_init_range)
        plant.set_state(q0, dq0)
        states[i, 0] = plant.read_state()

        ref_a = _sample_sine_params(rng, cfg)
        ref_b = _sample_sine_params(rng, cfg)
        for k in range(steps):
            t = k * cfg.dt
            qa_ref, dqa_ref = _eval_sine_ref(ref_a, t)
            qb_ref, dqb_ref = _eval_sine_ref(ref_b, t)
            q_ref = np.array([qa_ref, qb_ref], dtype=np.float64)
            dq_ref = np.array([dqa_ref, dqb_ref], dtype=np.float64)
            q_ref = np.clip(q_ref, safe_limits[:, 0], safe_limits[:, 1])

            x = plant.read_state()
            q, dq = x[:2], x[2:]
            tau = kp * (q_ref - q) + kd * (dq_ref - dq)
            tau = _limit_repel_tau(q, dq, safe_limits, tau, cfg.boundary_kp, cfg.boundary_kd)
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)

            tensions = _tau_to_cable(plant, tau, cfg)
            plant.apply_cable_tensions(tensions)
            plant.step()

            states[i, k + 1] = plant.read_state()
            inputs[i, k] = tau
            q_ref_hist[i, k] = q_ref
            dq_ref_hist[i, k] = dq_ref
            cable_ctrl[i, k] = tensions

    meta = {
        "mode": "PDCtrl",
        "description": "PD tracking of random multi-sine joint references inside active joint limits",
        "config": asdict(cfg),
        "safe_q_limits": safe_limits.tolist(),
        "state_order": ["qa", "qb", "dqa", "dqb"],
        "input_order": ["tau_a", "tau_b"],
        "cable_order": list(CABLE_NAMES),
    }
    return {"states": states, "inputs": inputs, "q_ref": q_ref_hist, "dq_ref": dq_ref_hist, "cable_ctrl": cable_ctrl}, meta
