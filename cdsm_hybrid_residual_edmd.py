"""
cdsm_hybrid_residual_edmd.py
============================
绳驱空间机械臂：名义模型 + EDMD 残差 混合建模（一步预测评估）。

流程:
    1. MuJoCo (multi_joint_cable_dirven_space_robot.xml) + 多正弦 PD + 拮抗绳驱映射 采集轨迹;
    2. 名义模型 f_nom 计算一步残差 r_k = x_{k+1}^{mj} - f_nom(x_k, u_k);
       其中 u_k = [tau_a, tau_b] 为 PD 指令关节力矩 (与名义模型接口一致);
    3. EDMD 拟合 r_hat(x_k, u_k);
    4. 验证集评估 (可选): 一步预测 或 开环多步 rollout.

运行:
    python cdsm_hybrid_residual_edmd.py
    python cdsm_hybrid_residual_edmd.py --eval_mode one_step
    python cdsm_hybrid_residual_edmd.py --eval_mode rollout
    python cdsm_hybrid_residual_edmd.py --eval_mode both
    python cdsm_hybrid_residual_edmd.py --train_traj 5 --val_traj 2 --steps 50
    python cdsm_hybrid_residual_edmd.py --dictionary rbf
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
import mujoco
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from cdsm_rigid_nominal_model import CdsmRigidNominalModel, make_nominal_model
from utils_plot import get_save_dir, save_figure

XML_PATH = "multi_joint_cable_dirven_space_robot.xml"
STATE_LABELS = ["q_a", "q_b", "dq_a", "dq_b"]

# 与 multi_joint_cable_dirven_space_robot.xml 一致
ACTIVE_JOINTS = ("joint1", "joint3")
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}
CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]
IDX_F1P = [0, 2]
IDX_F1M = [1, 3]
IDX_F2P = [4, 6]
IDX_F2M = [5, 7]
F_PRELOAD = 20.0
F_MAX_CABLE = 2000.0


# ===================================================================
# MuJoCo 辅助 (自包含, 不依赖 edmd_mujoco_cdsm_joint_torque.py)
# ===================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def name_to_joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in XML: {name}")
    return jid


def name_to_actuator_id(model: mujoco.MjModel, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found in XML: {name}")
    return aid


def name_to_tendon_id(model: mujoco.MjModel, name: str) -> int:
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if tid < 0:
        raise ValueError(f"Tendon not found in XML: {name}")
    return tid


def load_cable_model(
    xml_path: str, dt: float
) -> Tuple[mujoco.MjModel, mujoco.MjData, mujoco.MjData, Dict[str, np.ndarray]]:
    """
    加载绳驱 MuJoCo 模型并缓存索引.

    返回 (model, data, scratch, indices).
    scratch 专用于 tendon Jacobian 有限差分, 不污染主仿真 data.
    """
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
    """绳索长度对 q 的 Jacobian (8, nv); 使用 mj_fwdPosition 避免控制回调递归."""
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
    m_p: float, m_m: float, tau_des: float, f_pre: float, f_max: float
) -> Tuple[float, float, float]:
    u_max = f_max - f_pre
    tau_base = (m_p + m_m) * f_pre
    tau_eff = tau_des - tau_base
    eps = 1e-12
    candidates: List[Tuple[float, float, float, float]] = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > eps:
        u = max(tau_eff / m_p, 0.0)
        u_clip = min(u, u_max)
        candidates.append((u_clip, 0.0, abs(tau_eff - m_p * u_clip), u_clip))
    if abs(m_m) > eps:
        u = max(tau_eff / m_m, 0.0)
        u_clip = min(u, u_max)
        candidates.append((0.0, u_clip, abs(tau_eff - m_m * u_clip), u_clip))
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
    f_max: float = F_MAX_CABLE,
) -> np.ndarray:
    """期望关节力矩 -> 8 根绳张力 (按 CABLE_NAMES 顺序)."""
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


# ===================================================================
# 配置
# ===================================================================
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
class ResidualEdmdConfig:
    dictionary: str
    ridge: float
    rbf_centers: int
    rbf_sigma: Optional[float]
    rbf_seed: int


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.std + self.mean


# ===================================================================
# 特征: 原始角度 sin/cos + dq + tau
# ===================================================================
def build_feature_vector(x_raw: np.ndarray, u_raw: np.ndarray) -> np.ndarray:
    """z = [sin(qa), cos(qa), sin(qb), cos(qb), dqa, dqb, tau_a, tau_b]."""
    qa = x_raw[:, 0:1]
    qb = x_raw[:, 1:2]
    return np.hstack(
        [
            np.sin(qa),
            np.cos(qa),
            np.sin(qb),
            np.cos(qb),
            x_raw[:, 2:4],
            u_raw,
        ]
    )


def hermite_dictionary(z_norm: np.ndarray) -> np.ndarray:
    cols = [np.ones((z_norm.shape[0], 1), dtype=np.float64), z_norm, z_norm * z_norm - 1.0]
    pairs = []
    for i in range(z_norm.shape[1]):
        for j in range(i + 1, z_norm.shape[1]):
            pairs.append((z_norm[:, i] * z_norm[:, j])[:, None])
    if pairs:
        cols.append(np.hstack(pairs))
    return np.hstack(cols)


def estimate_rbf_sigma(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return 1.0
    rng = np.random.RandomState(0)
    n = min(1000, centers.shape[0])
    i = rng.randint(0, centers.shape[0], size=n)
    j = rng.randint(0, centers.shape[0], size=n)
    mask = i != j
    if not np.any(mask):
        return 1.0
    d = np.linalg.norm(centers[i[mask]] - centers[j[mask]], axis=1)
    return max(float(np.median(d)), 1e-6)


def rbf_dictionary(z_norm: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    diff = z_norm[:, None, :] - centers[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    rbf = np.exp(-0.5 * sqdist / (sigma * sigma))
    return np.hstack([np.ones((z_norm.shape[0], 1)), z_norm, rbf])


# ===================================================================
# Step 1: PD 多正弦数据采集
# ===================================================================
@dataclass
class SineRefParams:
    A1: float
    A2: float
    w1: float
    w2: float
    phi1: float
    phi2: float


def _sample_sine_params(rng: np.random.RandomState, cfg: PDCollectConfig, joint_idx: int) -> SineRefParams:
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
    # 力矩限幅暂时关闭 (按需求注释)
    # return np.clip(tau, -tau_max, tau_max)
    return tau


def collect_pd_trajectories(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scratch: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    cfg: PDCollectConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    rng = np.random.RandomState(cfg.seed)
    n = cfg.traj_count
    T = cfg.steps
    states = np.zeros((n, T + 1, 4), dtype=np.float64)
    inputs = np.zeros((n, T, 2), dtype=np.float64)
    cable_ctrl = np.zeros((n, T, 8), dtype=np.float64)
    q_ref_hist = np.zeros((n, T, 2), dtype=np.float64)

    kp = np.array(cfg.kp, dtype=np.float64)
    kd = np.array(cfg.kd, dtype=np.float64)
    # sat_count = 0
    # total_steps = n * T

    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])

    for i in range(n):
        q0 = rng.uniform(-cfg.q_init_range, cfg.q_init_range, size=2)
        dq0 = rng.uniform(-cfg.dq_init_range, cfg.dq_init_range, size=2)
        set_active_state(model, data, indices, q0, dq0)
        states[i, 0] = get_active_state(data, indices)

        ref_a = _sample_sine_params(rng, cfg, 0)
        ref_b = _sample_sine_params(rng, cfg, 1)

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
            # if np.any(np.abs(tau) >= cfg.tau_max - 1e-9):
            #     sat_count += 1

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
            data.ctrl[indices["actuator_ids"]] = F_cable
            mujoco.mj_step(model, data)
            states[i, k + 1] = get_active_state(data, indices)
            inputs[i, k] = tau
            cable_ctrl[i, k] = F_cable

    meta = {
        # "saturation_ratio": sat_count / max(total_steps, 1),
        "kp": list(cfg.kp),
        "kd": list(cfg.kd),
        "tau_max": cfg.tau_max,
        "f_preload": F_PRELOAD,
        "f_max_cable": F_MAX_CABLE,
        "control_mode": "pd_joint_torque_via_cable_map",
    }
    return {
        "states": states,
        "inputs": inputs,
        "q_ref": q_ref_hist,
        "cable_ctrl": cable_ctrl,
    }, meta


# ===================================================================
# Step 2: 名义一步 + 残差
# ===================================================================
def compute_nominal_next(
    nominal: CdsmRigidNominalModel, x: np.ndarray, u: np.ndarray, dt: float
) -> np.ndarray:
    return nominal.step(x, u, dt=dt, apply_joint_limits=True)


def build_residual_dataset(
    dataset: Dict[str, np.ndarray],
    nominal: CdsmRigidNominalModel,
    dt: float,
) -> Dict[str, np.ndarray]:
    states = dataset["states"]
    inputs = dataset["inputs"]
    n_traj, n_step, _ = inputs.shape
    x_nom_next = np.zeros_like(states[:, 1:, :])
    residuals = np.zeros_like(states[:, 1:, :])

    for i in range(n_traj):
        for k in range(n_step):
            x_k = states[i, k]
            u_k = inputs[i, k]
            x_np_nom = compute_nominal_next(nominal, x_k, u_k, dt)
            x_nom_next[i, k] = x_np_nom
            residuals[i, k] = states[i, k + 1] - x_np_nom

    return {
        "states": states,
        "inputs": inputs,
        "x_nom_next": x_nom_next,
        "residuals": residuals,
    }


def flatten_residual_data(res_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = res_data["states"]
    inputs = res_data["inputs"]
    x = states[:, :-1, :].reshape(-1, 4)
    u = inputs.reshape(-1, 2)
    xp_true = states[:, 1:, :].reshape(-1, 4)
    r = res_data["residuals"].reshape(-1, 4)
    return x, u, xp_true, r


# ===================================================================
# Step 3: 残差 EDMD
# ===================================================================
class ResidualEdmdModel:
    def __init__(
        self,
        weights: np.ndarray,
        z_norm: Normalizer,
        r_norm: Normalizer,
        dictionary: str,
        centers: Optional[np.ndarray] = None,
        sigma: Optional[float] = None,
        cond_number: float = 0.0,
        feature_dim: int = 0,
    ) -> None:
        self.weights = weights
        self.z_norm = z_norm
        self.r_norm = r_norm
        self.dictionary = dictionary
        self.centers = centers
        self.sigma = sigma
        self.cond_number = cond_number
        self.feature_dim = feature_dim

    def phi(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x2 = np.atleast_2d(x)
        u2 = np.atleast_2d(u)
        z = build_feature_vector(x2, u2)
        zn = self.z_norm.transform(z)
        if self.dictionary == "hermite":
            return hermite_dictionary(zn)
        if self.dictionary == "rbf":
            if self.centers is None or self.sigma is None:
                raise RuntimeError("RBF model missing centers or sigma.")
            return rbf_dictionary(zn, self.centers, self.sigma)
        raise ValueError(f"Unknown dictionary: {self.dictionary}")

    def predict_residual(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        r_norm = self.phi(x, u) @ self.weights
        return self.r_norm.inverse(r_norm)[0]

    def predict_hybrid_next(self, nominal: CdsmRigidNominalModel, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        x_nom = compute_nominal_next(nominal, x, u, dt)
        return x_nom + self.predict_residual(x, u)

    def rollout_hybrid(
        self,
        nominal: CdsmRigidNominalModel,
        x0: np.ndarray,
        u_seq: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """开环多步: x_{k+1} = f_nom(x_hat_k,u_k) + r_hat(x_hat_k,u_k), 使用预测状态递推."""
        u_seq = np.asarray(u_seq, dtype=np.float64)
        traj = np.zeros((u_seq.shape[0] + 1, 4), dtype=np.float64)
        traj[0] = np.asarray(x0, dtype=np.float64).reshape(4)
        x = traj[0].copy()
        for k, u in enumerate(u_seq):
            x = self.predict_hybrid_next(nominal, x, u, dt)
            traj[k + 1] = x
        return traj


def fit_residual_edmd(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    cfg: ResidualEdmdConfig,
) -> ResidualEdmdModel:
    z = build_feature_vector(x, u)
    z_norm = Normalizer.fit(z)
    zn = z_norm.transform(z)
    r_norm = Normalizer.fit(r)
    rn = r_norm.transform(r)

    centers = None
    sigma = None
    if cfg.dictionary == "hermite":
        phi = hermite_dictionary(zn)
    elif cfg.dictionary == "rbf":
        n_clusters = min(cfg.rbf_centers, zn.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=cfg.rbf_seed, batch_size=256, n_init=3
        )
        kmeans.fit(zn)
        centers = kmeans.cluster_centers_
        sigma = cfg.rbf_sigma if cfg.rbf_sigma is not None else estimate_rbf_sigma(centers)
        phi = rbf_dictionary(zn, centers, sigma)
    else:
        raise ValueError(f"Unsupported dictionary: {cfg.dictionary}")

    gram = phi.T @ phi
    rhs = phi.T @ rn
    reg = cfg.ridge * np.eye(gram.shape[0])
    weights = np.linalg.solve(gram + reg, rhs)
    cond_number = float(np.linalg.cond(gram))

    return ResidualEdmdModel(
        weights=weights,
        z_norm=z_norm,
        r_norm=r_norm,
        dictionary=cfg.dictionary,
        centers=centers,
        sigma=sigma,
        cond_number=cond_number,
        feature_dim=phi.shape[1],
    )


def save_residual_model(path: Path, model: ResidualEdmdModel, cfg: ResidualEdmdConfig) -> None:
    payload = {
        "weights": model.weights,
        "z_mean": model.z_norm.mean,
        "z_std": model.z_norm.std,
        "r_mean": model.r_norm.mean,
        "r_std": model.r_norm.std,
        "dictionary": np.array([cfg.dictionary]),
        "ridge": np.array([cfg.ridge]),
        "rbf_centers": np.array([cfg.rbf_centers]),
        "rbf_seed": np.array([cfg.rbf_seed]),
        "cond_number": np.array([model.cond_number]),
        "feature_dim": np.array([model.feature_dim]),
    }
    if model.centers is not None:
        payload["centers"] = model.centers
    if model.sigma is not None:
        payload["sigma"] = np.array([model.sigma])
    np.savez(path, **payload)


# ===================================================================
# Step 4: 评估 (一步 / 多步 rollout)
# ===================================================================
def rollout_nominal(
    nominal: CdsmRigidNominalModel,
    x0: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
) -> np.ndarray:
    """开环多步名义模型: 每步用 x_hat_k 递推, 不用真值状态校正."""
    u_seq = np.asarray(u_seq, dtype=np.float64)
    traj = np.zeros((u_seq.shape[0] + 1, 4), dtype=np.float64)
    traj[0] = np.asarray(x0, dtype=np.float64).reshape(4)
    x = traj[0].copy()
    for k, u in enumerate(u_seq):
        x = compute_nominal_next(nominal, x, u, dt)
        traj[k + 1] = x
    return traj


def _metrics_from_errors(err: np.ndarray) -> Dict[str, np.ndarray]:
    """err: (..., 4)  支持 (N,4) 一步 或 (N,T+1,4) rollout."""
    rmse_by_state = np.sqrt(np.mean(err * err, axis=tuple(range(err.ndim - 1))))
    mae_by_state = np.mean(np.abs(err), axis=tuple(range(err.ndim - 1)))
    total_rmse = float(np.sqrt(np.mean(err * err)))
    total_mae = float(np.mean(np.abs(err)))
    out: Dict[str, np.ndarray] = {
        "rmse_by_state": rmse_by_state,
        "mae_by_state": mae_by_state,
        "total_rmse": np.array([total_rmse]),
        "total_mae": np.array([total_mae]),
    }
    if err.ndim == 3:
        # 逐步 RMSE: 对轨迹条数与状态维聚合, 保留时间轴
        out["step_rmse"] = np.sqrt(np.mean(err * err, axis=(0, 2)))
    return out


def _metrics_to_json(metrics: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for model_name, m in metrics.items():
        entry: Dict[str, object] = {
            "total_rmse": float(m["total_rmse"][0]),
            "total_mae": float(m["total_mae"][0]),
            "rmse_by_state": m["rmse_by_state"].tolist(),
            "mae_by_state": m["mae_by_state"].tolist(),
        }
        if "step_rmse" in m:
            entry["step_rmse"] = m["step_rmse"].tolist()
        payload[model_name] = entry
    nom_rmse = float(metrics["nominal"]["total_rmse"][0])
    hyb_rmse = float(metrics["hybrid"]["total_rmse"][0])
    payload["improvement_ratio"] = (nom_rmse - hyb_rmse) / nom_rmse if nom_rmse > 0 else 0.0
    return payload


def evaluate_one_step(
    x: np.ndarray,
    u: np.ndarray,
    xp_true: np.ndarray,
    nominal: CdsmRigidNominalModel,
    edmd: ResidualEdmdModel,
    dt: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    n = x.shape[0]
    pred_nom = np.zeros_like(xp_true)
    pred_hyb = np.zeros_like(xp_true)
    for i in range(n):
        pred_nom[i] = compute_nominal_next(nominal, x[i], u[i], dt)
        pred_hyb[i] = edmd.predict_hybrid_next(nominal, x[i], u[i], dt)

    err_nom = pred_nom - xp_true
    err_hyb = pred_hyb - xp_true
    return {
        "nominal": _metrics_from_errors(err_nom),
        "hybrid": _metrics_from_errors(err_hyb),
    }


def evaluate_rollout(
    res_data: Dict[str, np.ndarray],
    nominal: CdsmRigidNominalModel,
    edmd: ResidualEdmdModel,
    dt: float,
) -> Dict[str, object]:
    """
    验证集开环多步 rollout.

    每条轨迹: x_hat_0 = x_0^{mj}, 对 k=0..T-1 用记录的控制 u_k 与 x_hat_k 递推.
    与一步评估的区别: 一步始终用真值 x_k; rollout 用 x_hat_k, 误差会累积.
    """
    states = res_data["states"]
    inputs = res_data["inputs"]
    n_traj = states.shape[0]
    pred_nom = np.zeros_like(states)
    pred_hyb = np.zeros_like(states)

    for i in range(n_traj):
        x0 = states[i, 0]
        u_seq = inputs[i]
        pred_nom[i] = rollout_nominal(nominal, x0, u_seq, dt)
        pred_hyb[i] = edmd.rollout_hybrid(nominal, x0, u_seq, dt)

    err_nom = pred_nom - states
    err_hyb = pred_hyb - states
    metrics = {
        "nominal": _metrics_from_errors(err_nom),
        "hybrid": _metrics_from_errors(err_hyb),
    }
    return {
        "metrics": metrics,
        "pred_nominal": pred_nom,
        "pred_hybrid": pred_hyb,
        "states_true": states,
    }


def plot_one_step_rmse_compare(metrics: Dict[str, Dict[str, np.ndarray]], out_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = STATE_LABELS
    x_pos = np.arange(len(labels))
    width = 0.35

    for ax, key, title in zip(axes, ["rmse_by_state", "mae_by_state"], ["RMSE", "MAE"]):
        nom = metrics["nominal"][key]
        hyb = metrics["hybrid"][key]
        ax.bar(x_pos - width / 2, nom, width, label="Nominal")
        ax.bar(x_pos + width / 2, hyb, width, label="Hybrid")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def plot_error_histogram(
    x: np.ndarray,
    u: np.ndarray,
    xp_true: np.ndarray,
    nominal: CdsmRigidNominalModel,
    edmd: ResidualEdmdModel,
    dt: float,
    out_name: str,
) -> None:
    n = x.shape[0]
    err_nom = np.zeros(n)
    err_hyb = np.zeros(n)
    for i in range(n):
        pn = compute_nominal_next(nominal, x[i], u[i], dt)
        ph = edmd.predict_hybrid_next(nominal, x[i], u[i], dt)
        err_nom[i] = np.linalg.norm(pn - xp_true[i])
        err_hyb[i] = np.linalg.norm(ph - xp_true[i])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err_nom, bins=40, alpha=0.6, label="Nominal", density=True)
    ax.hist(err_hyb, bins=40, alpha=0.6, label="Hybrid", density=True)
    ax.set_xlabel("L2 error per transition")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def plot_demo_trajectory_one_step(
    res_val: Dict[str, np.ndarray],
    nominal: CdsmRigidNominalModel,
    edmd: ResidualEdmdModel,
    dt: float,
    traj_idx: int,
    out_name: str,
) -> None:
    """演示: 每步从 MuJoCo 真值 x_k 出发做一步预测 (非误差累积评估协议)."""
    states = res_val["states"][traj_idx]
    inputs = res_val["inputs"][traj_idx]
    T = inputs.shape[0]
    t = np.arange(T + 1) * dt

    pred_true = states
    pred_nom = np.zeros_like(states)
    pred_hyb = np.zeros_like(states)
    pred_nom[0] = states[0]
    pred_hyb[0] = states[0]

    for k in range(T):
        x_k = states[k]
        u_k = inputs[k]
        pred_nom[k + 1] = compute_nominal_next(nominal, x_k, u_k, dt)
        pred_hyb[k + 1] = edmd.predict_hybrid_next(nominal, x_k, u_k, dt)

    _plot_trajectory_quad(t, pred_true, pred_nom, pred_hyb,
                          "Nominal (1-step from true x_k)", "Hybrid (1-step from true x_k)",
                          f"Trajectory {traj_idx}: one-step from true state each step",
                          out_name)


def plot_rollout_error_growth(metrics: Dict[str, Dict[str, np.ndarray]], dt: float, out_name: str) -> None:
    """多步 rollout: RMSE 随预测步数增长."""
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(metrics["nominal"]["step_rmse"].shape[0])
    t = steps * dt
    ax.plot(t, metrics["nominal"]["step_rmse"], lw=1.8, label="Nominal rollout")
    ax.plot(t, metrics["hybrid"]["step_rmse"], lw=1.8, label="Hybrid rollout")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMSE over states")
    ax.set_title("Open-loop rollout error growth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def plot_demo_trajectory_rollout(
    rollout_result: Dict[str, object],
    dt: float,
    traj_idx: int,
    out_name: str,
) -> None:
    """演示: 从真值初值 x_0 开环 rollout, 使用 x_hat_k 递推."""
    states = rollout_result["states_true"][traj_idx]
    pred_nom = rollout_result["pred_nominal"][traj_idx]
    pred_hyb = rollout_result["pred_hybrid"][traj_idx]
    T = states.shape[0] - 1
    t = np.arange(T + 1) * dt
    _plot_trajectory_quad(
        t, states, pred_nom, pred_hyb,
        "Nominal open-loop rollout",
        "Hybrid open-loop rollout",
        f"Trajectory {traj_idx}: open-loop rollout (x_hat_0 = MuJoCo x_0)",
        out_name,
    )


def _plot_trajectory_quad(
    t: np.ndarray,
    true_traj: np.ndarray,
    pred_nom: np.ndarray,
    pred_hyb: np.ndarray,
    label_nom: str,
    label_hyb: str,
    title: str,
    out_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for col in range(4):
        ax = axes[col // 2, col % 2]
        ax.plot(t, true_traj[:, col], "k-", lw=1.8, label="MuJoCo")
        ax.plot(t, pred_nom[:, col], "--", lw=1.2, label=label_nom)
        ax.plot(t, pred_hyb[:, col], "-.", lw=1.2, label=label_hyb)
        ax.set_title(STATE_LABELS[col])
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def print_eval_metrics(mode_label: str, metrics: Dict[str, Dict[str, np.ndarray]]) -> None:
    nom_rmse = float(metrics["nominal"]["total_rmse"][0])
    hyb_rmse = float(metrics["hybrid"]["total_rmse"][0])
    improve = (nom_rmse - hyb_rmse) / nom_rmse if nom_rmse > 0 else 0.0
    print(f"  [{mode_label}] RMSE  nominal : {nom_rmse:.6g}")
    print(f"  [{mode_label}] RMSE  hybrid  : {hyb_rmse:.6g}")
    print(f"  [{mode_label}] Improvement   : {100.0 * improve:.1f}%")
    for i, lab in enumerate(STATE_LABELS):
        rn = metrics["nominal"]["rmse_by_state"][i]
        rh = metrics["hybrid"]["rmse_by_state"][i]
        print(f"    {lab:5s}  nom={rn:.6g}  hyb={rh:.6g}")


# ===================================================================
# Main
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CDSM hybrid residual EDMD (PD collect + fit + one-step eval).")
    p.add_argument("--xml", default=XML_PATH)
    p.add_argument("--train_traj", type=int, default=150)
    p.add_argument("--val_traj", type=int, default=30)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--q_init_range", type=float, default=0.3)
    p.add_argument("--dq_init_range", type=float, default=0.2)
    p.add_argument("--amp_min", type=float, default=0.25)
    p.add_argument("--amp_max", type=float, default=0.65)
    p.add_argument("--omega_min", type=float, default=0.4)
    p.add_argument("--omega_max", type=float, default=1.2)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)
    p.add_argument("--tau_max", type=float, default=45.0)
    p.add_argument("--dictionary", choices=["hermite", "rbf"], default="hermite")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--rbf_centers", type=int, default=200)
    p.add_argument("--rbf_sigma", type=float, default=None)
    p.add_argument("--rbf_seed", type=int, default=2007)
    p.add_argument("--demo_traj", type=int, default=0, help="Validation trajectory index for demo plot.")
    p.add_argument(
        "--eval_mode",
        choices=["one_step", "rollout", "both"],
        default="one_step",
        help=(
            "one_step: 每步用真值 x_k 做一步预测; "
            "rollout: 开环多步, 用 x_hat_k 递推; "
            "both: 两种都跑."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(get_save_dir())
    t0 = time.time()

    pd_cfg_train = PDCollectConfig(
        traj_count=args.train_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        q_init_range=args.q_init_range,
        dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max),
        omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b),
        kd=(args.kd_a, args.kd_b),
        tau_max=args.tau_max,
    )
    pd_cfg_val = PDCollectConfig(
        traj_count=args.val_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed + 1000,
        q_init_range=args.q_init_range,
        dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max),
        omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b),
        kd=(args.kd_a, args.kd_b),
        tau_max=args.tau_max,
    )

    print("[1/4] Loading MuJoCo (cable-driven) and collecting PD trajectories...")
    model, data, scratch, indices = load_cable_model(args.xml, args.dt)

    train_raw, meta_train = collect_pd_trajectories(model, data, scratch, indices, pd_cfg_train)
    val_raw, meta_val = collect_pd_trajectories(model, data, scratch, indices, pd_cfg_val)
    print(f"      train {train_raw['states'].shape}, val {val_raw['states'].shape}")
    print(f"      cable ctrl shape {train_raw['cable_ctrl'].shape}")

    np.savez(out_dir / "dataset_train.npz", **train_raw)
    np.savez(out_dir / "dataset_val.npz", **val_raw)
    collection_meta = {
        "xml": args.xml,
        "pd_train": {**asdict(pd_cfg_train), "meta": meta_train},
        "pd_val": {**asdict(pd_cfg_val), "meta": meta_val},
    }
    with open(out_dir / "collection_meta.json", "w", encoding="utf-8") as f:
        json.dump(collection_meta, f, indent=2, ensure_ascii=False)

    print("[2/4] Computing nominal one-step residuals...")
    nominal = make_nominal_model(dt=args.dt)
    res_train = build_residual_dataset(train_raw, nominal, args.dt)
    res_val = build_residual_dataset(val_raw, nominal, args.dt)

    x_tr, u_tr, xp_tr, r_tr = flatten_residual_data(res_train)
    r_norm_mean = float(np.linalg.norm(r_tr, axis=1).mean())
    x_step_norm_mean = float(np.linalg.norm(xp_tr - x_tr, axis=1).mean())
    print(f"      mean |r|={r_norm_mean:.6g}, mean |x_next-x_k|={x_step_norm_mean:.6g}")

    np.savez(
        out_dir / "residual_train.npz",
        x=x_tr,
        u=u_tr,
        xp_true=xp_tr,
        residuals=r_tr,
    )

    print("[3/4] Fitting residual EDMD...")
    edmd_cfg = ResidualEdmdConfig(
        dictionary=args.dictionary,
        ridge=args.ridge,
        rbf_centers=args.rbf_centers,
        rbf_sigma=args.rbf_sigma,
        rbf_seed=args.rbf_seed,
    )
    edmd_model = fit_residual_edmd(x_tr, u_tr, r_tr, edmd_cfg)
    print(f"      dictionary={edmd_cfg.dictionary}, feature_dim={edmd_model.feature_dim}, cond={edmd_model.cond_number:.2e}")
    save_residual_model(out_dir / "residual_edmd_model.npz", edmd_model, edmd_cfg)

    eval_modes = ["one_step", "rollout"] if args.eval_mode == "both" else [args.eval_mode]
    print(f"[4/4] Validation evaluation (eval_mode={args.eval_mode})...")

    x_va, u_va, xp_va, _r_va = flatten_residual_data(res_val)
    demo_idx = min(max(args.demo_traj, 0), max(args.val_traj - 1, 0))

    summary: Dict[str, object] = {
        "xml": args.xml,
        "eval_mode": args.eval_mode,
        "edmd_config": asdict(edmd_cfg),
        "residual_train_mean_l2": r_norm_mean,
        "collection_meta": collection_meta,
    }

    print("-" * 60)
    if "one_step" in eval_modes:
        metrics_os = evaluate_one_step(x_va, u_va, xp_va, nominal, edmd_model, args.dt)
        summary["one_step"] = _metrics_to_json(metrics_os)
        plot_one_step_rmse_compare(metrics_os, "one_step_rmse_compare")
        plot_error_histogram(x_va, u_va, xp_va, nominal, edmd_model, args.dt, "one_step_error_hist")
        plot_demo_trajectory_one_step(res_val, nominal, edmd_model, args.dt, demo_idx, "one_step_demo_traj")
        print_eval_metrics("one-step", metrics_os)

    if "rollout" in eval_modes:
        rollout_res = evaluate_rollout(res_val, nominal, edmd_model, args.dt)
        metrics_ro = rollout_res["metrics"]
        summary["rollout"] = _metrics_to_json(metrics_ro)
        plot_one_step_rmse_compare(metrics_ro, "rollout_rmse_compare")
        plot_rollout_error_growth(metrics_ro, args.dt, "rollout_error_growth")
        plot_demo_trajectory_rollout(rollout_res, args.dt, demo_idx, "rollout_demo_traj")
        print_eval_metrics("rollout", metrics_ro)

    summary["elapsed_sec"] = time.time() - t0
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # 兼容旧文件名
    if "one_step" in eval_modes:
        with open(out_dir / "one_step_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
