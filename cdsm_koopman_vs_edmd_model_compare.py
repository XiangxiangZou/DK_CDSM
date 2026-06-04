"""
绳驱空间机械臂 (CDSM) 模型预测对比: DeepKoopman vs Koopman-EDMD
===============================================================

本脚本用 **两种** Koopman 类方法分别对绳驱空间机械臂做 *完整* 动力学建模与
预测, 并把两者与 MuJoCo 真值对比, 画出各自的模型误差:

  方法 A: DeepKoopman (深度神经网络观测量)
      z = [x_n ; psi_theta(x_n)],  z_{k+1} = A z_k + B u_n,  x_n = C z
      其中 psi_theta 是可训练 MLP, A/B 与编码器联合端到端训练。

  方法 B: Koopman-EDMD (固定 RBF 字典 + 最小二乘)
      z = [x_n ; rbf(x_n)],        z_{k+1} = A z_k + B u_n,  x_n = C z
      其中 rbf 中心由 k-means 选取, A/B 由 (岭) 最小二乘一次闭式拟合。

两者共用同一套数据、同一套标准化器、同一读出 C=[I_4|0], 因此对比公平: 唯一
区别是"观测量字典是学出来的 (DeepKoopman) 还是固定的 (EDMD)"。

预测模式 (可选参数 --pred_mode)
-------------------------------
  * one_step : 一步预测。每步都从 MuJoCo 真值 x_k 重新升维, 预测 x_{k+1};
               反映"模型局部一步精度", 不累积误差。
  * rollout  : 多步回滚预测。仅在 t=0 升维一次, 之后纯潜空间线性递推
               z_{k+1}=A z_k+B u_k, x_k=C z_k; 误差会随时间累积, 反映
               "开环长程预测"能力 (Koopman 线性预测的真实表现)。
  * both     : 两种都评并分别出图 (默认)。

输出图 (一式三份 PNG/SVG/PDF, 前缀 one_step_ / rollout_)
-------------------------------------------------------
  *_dynamic_response : 某条验证轨迹 4 个状态: MuJoCo vs DeepKoopman vs EDMD
  *_error_curve      : 两方法的瞬时状态误差范数随时间 (含 RMSE 标注)
  *_rmse_growth      : 两方法逐步 RMSE 随时间 (验证集平均)
  *_rmse_by_state    : 两方法各状态分量 RMSE 柱状对比

运行
----
    python cdsm_koopman_vs_edmd_model_compare.py                 # both 模式, 完整规模
    python cdsm_koopman_vs_edmd_model_compare.py --pred_mode rollout
    python cdsm_koopman_vs_edmd_model_compare.py --pred_mode one_step
    # 冒烟:
    python cdsm_koopman_vs_edmd_model_compare.py --train_traj 8 --val_traj 3 \
        --steps 120 --epochs 12 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires MuJoCo in the configured environment.") from exc

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires scikit-learn in the configured environment.") from exc

# ----------------------------------------------------------------------------
# 复用完整 DeepKoopman 实现 (方法 A) 及其训练/标准化基础设施
# ----------------------------------------------------------------------------
from cdsm_full_deepkoopman_lqr_mpc import (
    CONTROL_DIM,
    STATE_DIM,
    STATE_LABELS,
    DeepKoopmanConfig,
    KoopmanRuntime,
    Normalizer,
    build_windows,
    make_device,
    set_seed,
    train_deepkoopman,
)

from utils_plot import get_save_dir, save_figure

XML_DEFAULT = "multi_joint_cable_dirven_space_robot.xml"

# 与 multi_joint_cable_dirven_space_robot.xml 一致。本脚本本地维护采数与绳索
# 映射逻辑，避免依赖 cdsm_hybrid_residual_edmd.py 的实现细节。
ACTIVE_JOINTS = ("joint1", "joint3")
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}
CABLE_NAMES = [
    "cable11",
    "cable12",
    "cable13",
    "cable14",
    "cable21",
    "cable22",
    "cable23",
    "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]
IDX_F1P = [0, 2]
IDX_F1M = [1, 3]
IDX_F2P = [4, 6]
IDX_F2M = [5, 7]
F_PRELOAD = 20.0
# Cable max-tension clipping is intentionally disabled in this comparison script.
# F_MAX_CABLE = 2000.0
F_MAX_CABLE: Optional[float] = None


# ============================================================================
# MuJoCo cable data collection (local copy, decoupled from residual EDMD script)
# ============================================================================
@dataclass
class PDCollectConfig:
    traj_count: int
    steps: int
    dt: float
    seed: int
    q_init_range: float
    dq_init_range: float
    amp_range: Tuple[float, float]
    omega_range: Tuple[float, float]
    kp: Tuple[float, float]
    kd: Tuple[float, float]
    tau_max: float


@dataclass
class SineRefParams:
    A1: float
    A2: float
    w1: float
    w2: float
    phi1: float
    phi2: float


def name_to_joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in XML: {name}")
    return int(jid)


def name_to_actuator_id(model: mujoco.MjModel, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found in XML: {name}")
    return int(aid)


def name_to_tendon_id(model: mujoco.MjModel, name: str) -> int:
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if tid < 0:
        raise ValueError(f"Tendon not found in XML: {name}")
    return int(tid)


def load_cable_model(
    xml_path: str, dt: float
) -> Tuple[mujoco.MjModel, mujoco.MjData, mujoco.MjData, Dict[str, np.ndarray]]:
    """Load the cable-driven MuJoCo model and cache the indices used here."""
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)

    active_joint_ids = np.array([name_to_joint_id(model, n) for n in ACTIVE_JOINTS], dtype=int)
    active_qpos = np.array([model.jnt_qposadr[j] for j in active_joint_ids], dtype=int)
    active_dof = np.array([model.jnt_dofadr[j] for j in active_joint_ids], dtype=int)

    jnt_id = {n: name_to_joint_id(model, n) for n in ("joint1", "joint2", "joint3", "joint4")}
    dof_all = {n: int(model.jnt_dofadr[jnt_id[n]]) for n in jnt_id}

    mimic_pairs = []
    for mimic, source in MIMIC_JOINTS.items():
        mimic_jid = name_to_joint_id(model, mimic)
        source_jid = name_to_joint_id(model, source)
        mimic_pairs.append((model.jnt_qposadr[mimic_jid], model.jnt_qposadr[source_jid]))

    actuator_ids = np.array([name_to_actuator_id(model, n) for n in ACTUATOR_NAMES], dtype=int)
    tendon_ids = np.array([name_to_tendon_id(model, n) for n in CABLE_NAMES], dtype=int)

    indices = {
        "active_qpos": active_qpos,
        "active_dof": active_dof,
        "mimic_pairs": np.array(mimic_pairs, dtype=int),
        "actuator_ids": actuator_ids,
        "tendon_ids": tendon_ids,
        "dof_j1": dof_all["joint1"],
        "dof_j2": dof_all["joint2"],
        "dof_j3": dof_all["joint3"],
        "dof_j4": dof_all["joint4"],
    }
    return model, data, scratch, indices


def set_active_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    q: np.ndarray,
    dq: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[indices["active_qpos"]] = q
    data.qvel[indices["active_dof"]] = dq
    for mimic_qpos, source_qpos in indices["mimic_pairs"]:
        data.qpos[mimic_qpos] = data.qpos[source_qpos]
    mujoco.mj_forward(model, data)


def get_active_state(data: mujoco.MjData, indices: Dict[str, np.ndarray]) -> np.ndarray:
    q = data.qpos[indices["active_qpos"]]
    dq = data.qvel[indices["active_dof"]]
    return np.array([q[0], q[1], dq[0], dq[1]], dtype=np.float64)


def compute_tendon_jacobian_fd(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    q_ref: np.ndarray,
    tendon_ids_ordered: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Finite-difference tendon length Jacobian, shape (8, nv)."""
    nv = model.nv
    J = np.zeros((len(tendon_ids_ordered), nv), dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64).copy()
    for j in range(nv):
        scratch.qpos[:] = q_ref
        scratch.qpos[j] = q_ref[j] + eps
        mujoco.mj_fwdPosition(model, scratch)
        L_plus = np.array(scratch.ten_length, dtype=np.float64)[tendon_ids_ordered]

        scratch.qpos[:] = q_ref
        scratch.qpos[j] = q_ref[j] - eps
        mujoco.mj_fwdPosition(model, scratch)
        L_minus = np.array(scratch.ten_length, dtype=np.float64)[tendon_ids_ordered]
        J[:, j] = (L_plus - L_minus) / (2.0 * eps)
    return J


def _solve_antagonistic_pair(
    m_p: float, m_m: float, tau_des: float, f_pre: float, f_max: Optional[float]
) -> Tuple[float, float, float]:
    """Map one desired joint torque to a preload-plus-antagonistic cable pair."""
    # Max cable tension clipping is disabled:
    # u_max = f_max - f_pre
    tau_base = (m_p + m_m) * f_pre
    tau_eff = tau_des - tau_base
    eps = 1e-12
    candidates: List[Tuple[float, float, float, float]] = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > eps:
        u = max(tau_eff / m_p, 0.0)
        # u_clip = min(u, u_max)
        u_unlimited = u
        candidates.append((u_unlimited, 0.0, abs(tau_eff - m_p * u_unlimited), u_unlimited))
    if abs(m_m) > eps:
        u = max(tau_eff / m_m, 0.0)
        # u_clip = min(u, u_max)
        u_unlimited = u
        candidates.append((0.0, u_unlimited, abs(tau_eff - m_m * u_unlimited), u_unlimited))
    u_p, u_m, res, _ = min(candidates, key=lambda c: (c[2], c[3]))
    return f_pre + u_p, f_pre + u_m, res


def cable_antagonistic_map(
    tau_a_des: float,
    tau_b_des: float,
    J: np.ndarray,
    dof_j1: int,
    dof_j2: int,
    dof_j3: int,
    dof_j4: int,
    f_pre: float = F_PRELOAD,
    f_max: Optional[float] = F_MAX_CABLE,
) -> np.ndarray:
    """Desired joint torque -> 8 cable tensions in CABLE_NAMES order."""
    a = J[:, dof_j1] + J[:, dof_j2]
    b = J[:, dof_j3] + J[:, dof_j4]
    m_p1 = a[IDX_F1P[0]] + a[IDX_F1P[1]]
    m_m1 = a[IDX_F1M[0]] + a[IDX_F1M[1]]
    m_p2 = b[IDX_F2P[0]] + b[IDX_F2P[1]]
    m_m2 = b[IDX_F2M[0]] + b[IDX_F2M[1]]
    f1p, f1m, _ = _solve_antagonistic_pair(m_p1, m_m1, tau_a_des, f_pre, f_max)
    f2p, f2m, _ = _solve_antagonistic_pair(m_p2, m_m2, tau_b_des, f_pre, f_max)
    F = np.empty(8, dtype=np.float64)
    F[IDX_F1P] = f1p
    F[IDX_F1M] = f1m
    F[IDX_F2P] = f2p
    F[IDX_F2M] = f2m
    return F


def _sample_sine_params(rng: np.random.RandomState, cfg: PDCollectConfig) -> SineRefParams:
    amp_lo, amp_hi = cfg.amp_range
    w_lo, w_hi = cfg.omega_range
    return SineRefParams(
        A1=rng.uniform(amp_lo, amp_hi),
        A2=rng.uniform(amp_lo * 0.5, amp_hi * 0.8),
        w1=rng.uniform(w_lo, w_hi),
        w2=rng.uniform(w_lo * 1.3, w_hi * 1.8),
        phi1=rng.uniform(0.0, 2.0 * np.pi),
        phi2=rng.uniform(0.0, 2.0 * np.pi),
    )


def eval_sine_ref(params: SineRefParams, t: float) -> Tuple[float, float]:
    q = params.A1 * np.sin(params.w1 * t) + params.A2 * np.sin(params.w2 * t + params.phi2)
    dq = (
        params.A1 * params.w1 * np.cos(params.w1 * t)
        + params.A2 * params.w2 * np.cos(params.w2 * t + params.phi2)
    )
    return float(q), float(dq)


def pd_torque(
    q: np.ndarray,
    dq: np.ndarray,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_max: float,
) -> np.ndarray:
    tau = kp * (q_ref - q) + kd * (dq_ref - dq)
    # Kept disabled to preserve prior experiment behavior.
    # return np.clip(tau, -tau_max, tau_max)
    return tau


def collect_pd_trajectories(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scratch: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    cfg: PDCollectConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    rng = np.random.RandomState(cfg.seed)
    n = cfg.traj_count
    T = cfg.steps
    states = np.zeros((n, T + 1, STATE_DIM), dtype=np.float64)
    inputs = np.zeros((n, T, CONTROL_DIM), dtype=np.float64)
    cable_ctrl = np.zeros((n, T, len(CABLE_NAMES)), dtype=np.float64)
    q_ref_hist = np.zeros((n, T, 2), dtype=np.float64)

    kp = np.array(cfg.kp, dtype=np.float64)
    kd = np.array(cfg.kd, dtype=np.float64)
    cable_tension_limit_enabled = F_MAX_CABLE is not None
    cable_sat_count = 0
    total_cable_values = n * T * len(CABLE_NAMES)

    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])

    for i in range(n):
        q0 = rng.uniform(-cfg.q_init_range, cfg.q_init_range, size=2)
        dq0 = rng.uniform(-cfg.dq_init_range, cfg.dq_init_range, size=2)
        set_active_state(model, data, indices, q0, dq0)
        states[i, 0] = get_active_state(data, indices)

        ref_a = _sample_sine_params(rng, cfg)
        ref_b = _sample_sine_params(rng, cfg)

        for k in range(T):
            t = k * cfg.dt
            qa_ref, dqa_ref = eval_sine_ref(ref_a, t)
            qb_ref, dqb_ref = eval_sine_ref(ref_b, t)
            q_ref = np.array([qa_ref, qb_ref])
            dq_ref = np.array([dqa_ref, dqb_ref])
            q_ref_hist[i, k] = q_ref

            q = data.qpos[indices["active_qpos"]]
            dq = data.qvel[indices["active_dof"]]
            tau = pd_torque(q, dq, q_ref, dq_ref, kp, kd, cfg.tau_max)

            J = compute_tendon_jacobian_fd(
                model, scratch, data.qpos.copy(), indices["tendon_ids"]
            )
            F_cable = cable_antagonistic_map(
                float(tau[0]),
                float(tau[1]),
                J,
                dof_j1,
                dof_j2,
                dof_j3,
                dof_j4,
            )
            if cable_tension_limit_enabled:
                cable_sat_count += int(np.sum(F_cable >= F_MAX_CABLE - 1e-9))
            data.ctrl[indices["actuator_ids"]] = F_cable
            mujoco.mj_step(model, data)
            states[i, k + 1] = get_active_state(data, indices)
            inputs[i, k] = tau
            cable_ctrl[i, k] = F_cable

    meta = {
        "kp": list(cfg.kp),
        "kd": list(cfg.kd),
        "tau_max": cfg.tau_max,
        "f_preload": F_PRELOAD,
        "f_max_cable": F_MAX_CABLE,
        "cable_tension_limit_enabled": cable_tension_limit_enabled,
        "cable_saturation_count": cable_sat_count,
        "cable_saturation_ratio": cable_sat_count / max(total_cable_values, 1),
        "control_mode": "pd_joint_torque_via_local_cable_map",
    }
    return {
        "states": states,
        "inputs": inputs,
        "q_ref": q_ref_hist,
        "cable_ctrl": cable_ctrl,
    }, meta


# ============================================================================
# 统一线性 Koopman 预测器接口
#   lift(x_phys) -> z       : 物理状态 -> 潜变量 (含状态内嵌, 前 4 维为标准化状态)
#   step(z, u_n) -> z       : 潜空间线性推进 z+ = A z + B u_n (u_n 为标准化控制)
#   recover(z) -> x_phys    : 潜变量 -> 物理状态 (反标准化 C z)
# 两种方法都实现该接口, 故预测/评估代码完全共用, 保证对比公平。
# ============================================================================
class LinearKoopmanPredictor:
    """统一接口基类 (DeepKoopman / EDMD 各自实现)。"""

    u_normer: Normalizer
    latent_dim: int
    name: str

    def lift(self, x_phys: np.ndarray) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def step(self, z: np.ndarray, u_n: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def recover(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class DeepKoopmanPredictor(LinearKoopmanPredictor):
    """方法 A: 把训练好的 DeepKoopman 运行期封装成统一预测接口。"""

    def __init__(self, runtime: KoopmanRuntime) -> None:
        self.rt = runtime
        self.u_normer = runtime.u_normer
        self.latent_dim = runtime.A_d.shape[0]
        self.name = "DeepKoopman"

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        return self.rt.encode(x_phys)

    def step(self, z: np.ndarray, u_n: np.ndarray) -> np.ndarray:
        return self.rt.A_d @ z + self.rt.B_d @ u_n

    def recover(self, z: np.ndarray) -> np.ndarray:
        return self.rt.predict_state(z)


# ============================================================================
# 方法 B: 完整 Koopman-EDMD (固定 RBF 字典 + 岭最小二乘)
# ============================================================================
@dataclass
class EdmdConfig:
    """Koopman-EDMD 字典与拟合超参数。"""

    n_centers: int            # RBF 中心个数 (k-means)
    rbf_sigma: Optional[float]  # RBF 带宽; None 用中心间中位距估计
    ridge: float              # 岭回归正则
    kmeans_seed: int          # k-means 随机种子


def estimate_rbf_sigma(centers: np.ndarray) -> float:
    """用中心两两距离的中位数作为 RBF 带宽初值。"""
    n = centers.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.RandomState(0)
    m = min(2000, n)
    i = rng.randint(0, n, size=m)
    j = rng.randint(0, n, size=m)
    mask = i != j
    if not np.any(mask):
        return 1.0
    d = np.linalg.norm(centers[i[mask]] - centers[j[mask]], axis=1)
    return max(float(np.median(d)), 1e-6)


class EdmdPredictor(LinearKoopmanPredictor):
    """
    完整 Koopman-EDMD 预测器。

    字典: z = [x_n, rbf(x_n; centers, sigma)], 因此读出 C=[I_4|0] 精确。
    动力学: z_{k+1} = A z_k + B u_n, 由 (岭) 最小二乘一次拟合得到 A,B。
    """

    def __init__(
        self,
        x_normer: Normalizer,
        u_normer: Normalizer,
        centers: np.ndarray,
        sigma: float,
        A: np.ndarray,
        B: np.ndarray,
        cond_number: float = 0.0,
    ) -> None:
        self.x_normer = x_normer
        self.u_normer = u_normer
        self.centers = np.asarray(centers, dtype=np.float64)  # (n_centers, 4)
        self.sigma = float(sigma)
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.cond_number = float(cond_number)
        self.latent_dim = A.shape[0]
        self.name = "Koopman-EDMD"

    def _lift_norm(self, x_n: np.ndarray) -> np.ndarray:
        """对已标准化状态 (..., 4) 升维, 返回 (..., 4+n_centers)。"""
        x_n = np.atleast_2d(np.asarray(x_n, dtype=np.float64))
        diff = x_n[:, None, :] - self.centers[None, :, :]
        sqdist = np.sum(diff * diff, axis=2)
        rbf = np.exp(-0.5 * sqdist / (self.sigma * self.sigma))
        return np.hstack([x_n, rbf])

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_n = self.x_normer.transform(np.atleast_2d(x_phys))
        return self._lift_norm(x_n).reshape(-1)

    def step(self, z: np.ndarray, u_n: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(z, dtype=np.float64).reshape(-1) + self.B @ np.asarray(
            u_n, dtype=np.float64
        ).reshape(-1)

    def recover(self, z: np.ndarray) -> np.ndarray:
        x_n = np.asarray(z, dtype=np.float64).reshape(-1)[:STATE_DIM]
        return self.x_normer.inverse(x_n)


def fit_full_edmd(
    states: np.ndarray,
    inputs: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    cfg: EdmdConfig,
) -> EdmdPredictor:
    """
    用训练轨迹拟合完整 Koopman-EDMD 的 (A,B)。

    步骤:
      1) 摊平为转移对 (x_k, u_k, x_{k+1}) 并标准化;
      2) k-means 在标准化状态空间选 RBF 中心, 估计带宽 sigma;
      3) 升维 Z=lift(x_n), Zp=lift(x'_n); 构造 Omega=[Z, U_n];
      4) 岭最小二乘 Zp ≈ Omega G, 拆出 A=G[:D].T, B=G[D:].T。
    """
    X = states[:, :-1, :].reshape(-1, STATE_DIM)
    Xp = states[:, 1:, :].reshape(-1, STATE_DIM)
    U = inputs.reshape(-1, CONTROL_DIM)
    Xn = x_normer.transform(X)
    Xpn = x_normer.transform(Xp)
    Un = u_normer.transform(U)

    n_clusters = min(cfg.n_centers, Xn.shape[0])
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=cfg.kmeans_seed, batch_size=256, n_init=3
    )
    kmeans.fit(Xn)
    centers = kmeans.cluster_centers_.astype(np.float64)
    sigma = cfg.rbf_sigma if cfg.rbf_sigma is not None else estimate_rbf_sigma(centers)

    # 临时预测器仅用来复用 _lift_norm 升维
    tmp = EdmdPredictor(
        x_normer, u_normer, centers, sigma,
        A=np.zeros((STATE_DIM + n_clusters, STATE_DIM + n_clusters)),
        B=np.zeros((STATE_DIM + n_clusters, CONTROL_DIM)),
    )
    Z = tmp._lift_norm(Xn)     # (M, D)
    Zp = tmp._lift_norm(Xpn)   # (M, D)
    Omega = np.hstack([Z, Un])  # (M, D+nu)

    # 用样本平均的二阶矩 (Gram/M) 做岭回归, 使 ridge 与数据量无关 (尺度不变),
    # 否则固定 ridge 在大数据集上会被淹没, 导致 RBF 字典严重病态 -> rollout 发散。
    M = Omega.shape[0]
    gram = (Omega.T @ Omega) / M
    rhs = (Omega.T @ Zp) / M
    reg = cfg.ridge * np.eye(gram.shape[0])
    G = np.linalg.solve(gram + reg, rhs)  # (D+nu, D)
    cond_number = float(np.linalg.cond(gram))

    D = Z.shape[1]
    A = G[:D, :].T.copy()        # (D, D)
    B = G[D:, :].T.copy()        # (D, nu)
    return EdmdPredictor(x_normer, u_normer, centers, sigma, A, B, cond_number)


# ============================================================================
# 预测 (one-step / rollout) 与评估
# ============================================================================
def predict_trajectory(
    pred: LinearKoopmanPredictor,
    states: np.ndarray,
    inputs: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    用给定预测器对单条轨迹做预测, 返回 (T+1, 4) 物理状态轨迹 (含真值初值)。

    one_step : 每步从真值 x_k 升维, 预测 x_{k+1} (teacher forcing, 不累积误差);
    rollout  : 仅 t=0 升维, 之后纯潜空间递推 (开环, 误差累积)。
    """
    T = inputs.shape[0]
    u_n = pred.u_normer.transform(inputs)
    out = np.zeros((T + 1, STATE_DIM), dtype=np.float64)
    out[0] = np.asarray(states[0], dtype=np.float64)
    if mode == "rollout":
        z = pred.lift(states[0])
        for k in range(T):
            z = pred.step(z, u_n[k])
            out[k + 1] = pred.recover(z)
    elif mode == "one_step":
        for k in range(T):
            z = pred.lift(states[k])
            z = pred.step(z, u_n[k])
            out[k + 1] = pred.recover(z)
    else:
        raise ValueError(f"Unknown pred mode: {mode}")
    return out


def evaluate_predictor(
    pred: LinearKoopmanPredictor, val_raw: Dict[str, np.ndarray], mode: str
) -> Dict[str, object]:
    """在验证集上评估某预测器, 返回预测轨迹与误差统计。"""
    states = val_raw["states"]
    inputs = val_raw["inputs"]
    n_traj = states.shape[0]
    preds = np.zeros_like(states)
    for i in range(n_traj):
        preds[i] = predict_trajectory(pred, states[i], inputs[i], mode)
    err = preds - states
    return {
        "preds": preds,
        "states_true": states,
        "rmse_by_state": np.sqrt(np.mean(err * err, axis=(0, 1))),
        "step_rmse": np.sqrt(np.mean(err * err, axis=(0, 2))),  # 逐时刻 (含 t=0)
        "total_rmse": float(np.sqrt(np.mean(err * err))),
    }


# ============================================================================
# 出图
# ============================================================================
def plot_dynamic_response(
    res_dk: Dict[str, object],
    res_ed: Dict[str, object],
    dt: float,
    traj_idx: int,
    mode: str,
    prefix: str,
) -> None:
    """某条验证轨迹 4 状态: MuJoCo vs DeepKoopman vs EDMD。"""
    true = res_dk["states_true"][traj_idx]
    dk = res_dk["preds"][traj_idx]
    ed = res_ed["preds"][traj_idx]
    t = np.arange(true.shape[0]) * dt
    units = ["rad", "rad", "rad/s", "rad/s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for i, label in enumerate(STATE_LABELS):
        ax = axes[i // 2, i % 2]
        ax.plot(t, true[:, i], "k-", lw=1.8, label="MuJoCo")
        ax.plot(t, dk[:, i], "--", lw=1.4, color="C0", label="DeepKoopman")
        ax.plot(t, ed[:, i], "-.", lw=1.4, color="C1", label="Koopman-EDMD")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{label} ({units[i]})")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{mode} prediction: MuJoCo vs DeepKoopman vs Koopman-EDMD")
    fig.tight_layout()
    save_figure(f"{prefix}_dynamic_response")
    plt.close(fig)


def plot_error_curve(
    res_dk: Dict[str, object],
    res_ed: Dict[str, object],
    dt: float,
    traj_idx: int,
    mode: str,
    prefix: str,
) -> None:
    """某条验证轨迹的瞬时状态误差范数随时间 (DeepKoopman vs EDMD)。"""
    true = res_dk["states_true"][traj_idx]
    err_dk = res_dk["preds"][traj_idx] - true
    err_ed = res_ed["preds"][traj_idx] - true
    t = np.arange(true.shape[0]) * dt
    e_dk = np.sqrt(np.mean(err_dk * err_dk, axis=1))
    e_ed = np.sqrt(np.mean(err_ed * err_ed, axis=1))
    rmse_dk = float(np.sqrt(np.mean(err_dk * err_dk)))
    rmse_ed = float(np.sqrt(np.mean(err_ed * err_ed)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, e_dk, lw=1.6, color="C0", label=f"DeepKoopman (RMSE={rmse_dk:.3g})")
    ax.plot(t, e_ed, lw=1.6, color="C1", label=f"Koopman-EDMD (RMSE={rmse_ed:.3g})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous state error (RMS over states)")
    ax.set_title(f"{mode} model error vs MuJoCo, trajectory {traj_idx}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(f"{prefix}_error_curve")
    plt.close(fig)


def plot_rmse_growth(
    res_dk: Dict[str, object],
    res_ed: Dict[str, object],
    dt: float,
    mode: str,
    prefix: str,
) -> None:
    """验证集平均逐时刻 RMSE (DeepKoopman vs EDMD)。"""
    sd = res_dk["step_rmse"]
    se = res_ed["step_rmse"]
    t = np.arange(sd.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, sd, lw=1.8, color="C0", label="DeepKoopman")
    ax.plot(t, se, lw=1.8, color="C1", label="Koopman-EDMD")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State RMSE (validation mean)")
    title = "one-step" if mode == "one_step" else "open-loop rollout"
    ax.set_title(f"{title} prediction RMSE vs MuJoCo")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(f"{prefix}_rmse_growth")
    plt.close(fig)


def plot_rmse_by_state(
    res_dk: Dict[str, object],
    res_ed: Dict[str, object],
    mode: str,
    prefix: str,
) -> None:
    """各状态分量 RMSE 柱状对比 (DeepKoopman vs EDMD)。"""
    rd = res_dk["rmse_by_state"]
    re = res_ed["rmse_by_state"]
    x = np.arange(STATE_DIM)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, rd, width, color="C0", label="DeepKoopman")
    b2 = ax.bar(x + width / 2, re, width, color="C1", label="Koopman-EDMD")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2g}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS)
    ax.set_ylabel("RMSE")
    ax.set_title(f"{mode} prediction RMSE by state")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(f"{prefix}_rmse_by_state")
    plt.close(fig)


def run_one_mode(
    mode: str,
    dk_pred: DeepKoopmanPredictor,
    ed_pred: EdmdPredictor,
    val_raw: Dict[str, np.ndarray],
    dt: float,
    demo_idx: int,
) -> Dict[str, object]:
    """评估并出图一个预测模式, 返回该模式两方法的 RMSE 摘要。"""
    print(f"  [{mode}] evaluating DeepKoopman & Koopman-EDMD...")
    res_dk = evaluate_predictor(dk_pred, val_raw, mode)
    res_ed = evaluate_predictor(ed_pred, val_raw, mode)
    prefix = mode
    plot_dynamic_response(res_dk, res_ed, dt, demo_idx, mode, prefix)
    plot_error_curve(res_dk, res_ed, dt, demo_idx, mode, prefix)
    plot_rmse_growth(res_dk, res_ed, dt, mode, prefix)
    plot_rmse_by_state(res_dk, res_ed, mode, prefix)
    print(
        f"  [{mode}] total RMSE: DeepKoopman={res_dk['total_rmse']:.6g}  "
        f"Koopman-EDMD={res_ed['total_rmse']:.6g}"
    )
    return {
        "deepkoopman": {
            "total_rmse": res_dk["total_rmse"],
            "rmse_by_state": res_dk["rmse_by_state"].tolist(),
        },
        "koopman_edmd": {
            "total_rmse": res_ed["total_rmse"],
            "rmse_by_state": res_ed["rmse_by_state"].tolist(),
        },
    }


# ============================================================================
# 命令行参数
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CDSM model prediction comparison: DeepKoopman vs Koopman-EDMD."
    )
    # --- 全局 ---
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    # --- 预测模式 (可选: 一步 / 多步回滚 / 两者) ---
    p.add_argument(
        "--pred_mode",
        choices=["one_step", "rollout", "both"],
        default="both",
        help="one_step=一步预测; rollout=多步回滚预测; both=两者都评。",
    )

    # --- 数据采集 (PD 多正弦) ---
    p.add_argument("--train_traj", type=int, default=120)
    p.add_argument("--val_traj", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--q_init_range", type=float, default=1.0)
    p.add_argument("--dq_init_range", type=float, default=0.8)
    p.add_argument("--amp_min", type=float, default=-1.0)
    p.add_argument("--amp_max", type=float, default=1.0)
    p.add_argument("--omega_min", type=float, default=0.3)
    p.add_argument("--omega_max", type=float, default=1.5)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)
    p.add_argument("--tau_max", type=float, default=float("inf"))

    # --- DeepKoopman 网络与训练 (方法 A) ---
    p.add_argument("--lift_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--window", type=int, default=40, help="多步窗口上限 (curriculum 终值)")
    p.add_argument("--window_start", type=int, default=4, help="curriculum 多步窗口起始长度")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--steps_per_epoch", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--w_pred", type=float, default=1.0)
    p.add_argument("--w_linear", type=float, default=0.5)
    p.add_argument("--w_l2", type=float, default=1e-8)
    # 长程稳定性: 谱范数惩罚 + 权重衰减 + 有界升维
    p.add_argument("--w_stab", type=float, default=10.0, help="||A||_2>rho_target 的惩罚权重")
    p.add_argument("--rho_target", type=float, default=1.0, help="谱范数目标上限")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--bound_lift", type=int, default=1, help="1=升维末层加 tanh 有界, 0=不加")

    # --- Koopman-EDMD 字典与拟合 (方法 B) ---
    p.add_argument("--edmd_centers", type=int, default=200)
    p.add_argument("--edmd_sigma", type=float, default=None)
    # 注: ridge 作用在 *样本平均* 的二阶矩 (Gram/M) 上, 故与数据量无关;
    # RBF 字典易病态, 取 1e-4 量级既能稳住 rollout 又基本不损一步精度。
    p.add_argument("--edmd_ridge", type=float, default=1e-4)
    p.add_argument("--edmd_seed", type=int, default=2007)

    # --- 评估出图 ---
    p.add_argument("--demo_traj", type=int, default=0)
    return p


def main() -> None:
    """采数 -> 训练 DeepKoopman & 拟合 EDMD -> 预测对比 -> 出图 + 汇总。"""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== CDSM model prediction compare: DeepKoopman vs Koopman-EDMD ===")
    print(f"device={device}, pred_mode={args.pred_mode}, output={out_dir}")

    # ---- [1/5] PD 采数 ----
    pd_train = PDCollectConfig(
        traj_count=args.train_traj, steps=args.steps, dt=args.dt, seed=args.seed,
        q_init_range=args.q_init_range, dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max), omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b), kd=(args.kd_a, args.kd_b), tau_max=args.tau_max,
    )
    pd_val = PDCollectConfig(
        traj_count=args.val_traj, steps=args.steps, dt=args.dt, seed=args.seed + 1000,
        q_init_range=args.q_init_range, dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max), omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b), kd=(args.kd_a, args.kd_b), tau_max=args.tau_max,
    )
    print("[1/5] Collecting PD MuJoCo trajectories...")
    mj_model, mj_data, scratch, indices = load_cable_model(args.xml, args.dt)
    train_raw, train_meta = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_train)
    val_raw, val_meta = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_val)
    np.savez(out_dir / "dataset_train.npz", **train_raw)
    np.savez(out_dir / "dataset_val.npz", **val_raw)
    print(f"      train={train_raw['states'].shape}, val={val_raw['states'].shape}")

    # ---- [2/5] 标准化器 (两方法共用) ----
    print("[2/5] Fitting shared normalizers...")
    x_all = train_raw["states"][:, :-1, :].reshape(-1, STATE_DIM)
    u_all = train_raw["inputs"].reshape(-1, CONTROL_DIM)
    x_normer = Normalizer.fit(x_all)
    u_normer = Normalizer.fit(u_all)

    # ---- [3/5] 训练 DeepKoopman (方法 A) ----
    print("[3/5] Training DeepKoopman...")
    Xw_tr, Uw_tr = build_windows(train_raw["states"], train_raw["inputs"], args.window)
    Xw_va, Uw_va = build_windows(val_raw["states"], val_raw["inputs"], args.window)
    Xw_tr_n = (Xw_tr - x_normer.mean) / x_normer.std
    Xw_va_n = (Xw_va - x_normer.mean) / x_normer.std
    Uw_tr_n = (Uw_tr - u_normer.mean) / u_normer.std
    Uw_va_n = (Uw_va - u_normer.mean) / u_normer.std
    dk_cfg = DeepKoopmanConfig(
        lift_dim=args.lift_dim, hidden=tuple(args.hidden), activation=args.activation,
        window=args.window, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, lr=args.lr, grad_clip=args.grad_clip,
        w_pred=args.w_pred, w_linear=args.w_linear, w_l2=args.w_l2,
        window_start=args.window_start, w_stab=args.w_stab, rho_target=args.rho_target,
        weight_decay=args.weight_decay, bound_lift=bool(args.bound_lift),
    )
    dk_model, history, dk_stats = train_deepkoopman(
        (Xw_tr_n, Uw_tr_n), (Xw_va_n, Uw_va_n), dk_cfg, device, out_dir
    )
    np.savetxt(
        out_dir / "training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_pred,train_linear,val_full_total,val_full_pred,val_full_linear",
        comments="",
    )
    dk_pred = DeepKoopmanPredictor(KoopmanRuntime(dk_model, x_normer, u_normer, device))

    # ---- [4/5] 拟合 Koopman-EDMD (方法 B) ----
    print("[4/5] Fitting Koopman-EDMD...")
    edmd_cfg = EdmdConfig(
        n_centers=args.edmd_centers, rbf_sigma=args.edmd_sigma,
        ridge=args.edmd_ridge, kmeans_seed=args.edmd_seed,
    )
    ed_pred = fit_full_edmd(train_raw["states"], train_raw["inputs"], x_normer, u_normer, edmd_cfg)
    print(
        f"      EDMD latent_dim={ed_pred.latent_dim}, sigma={ed_pred.sigma:.4g}, "
        f"cond(G)={ed_pred.cond_number:.3e}"
    )

    # ---- [5/5] 预测对比 + 出图 ----
    print("[5/5] Predicting and comparing...")
    modes = ["one_step", "rollout"] if args.pred_mode == "both" else [args.pred_mode]
    demo_idx = min(max(args.demo_traj, 0), max(args.val_traj - 1, 0))
    results: Dict[str, object] = {}
    for mode in modes:
        results[mode] = run_one_mode(mode, dk_pred, ed_pred, val_raw, args.dt, demo_idx)

    summary = {
        "xml": args.xml,
        "dt": args.dt,
        "pred_mode": args.pred_mode,
        "deepkoopman_config": asdict(dk_cfg),
        "deepkoopman_latent_dim": dk_model.latent_dim,
        "deepkoopman_train_stats": dk_stats,
        "edmd_config": asdict(edmd_cfg),
        "edmd_latent_dim": ed_pred.latent_dim,
        "edmd_sigma": ed_pred.sigma,
        "edmd_cond": ed_pred.cond_number,
        "normalization": {"x": x_normer.to_json(), "u": u_normer.to_json()},
        "results": results,
        "collection": {
            "train": {**asdict(pd_train), "meta": train_meta},
            "val": {**asdict(pd_val), "meta": val_meta},
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
