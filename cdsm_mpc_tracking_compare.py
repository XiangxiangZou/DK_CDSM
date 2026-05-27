"""
cdsm_mpc_tracking_compare.py
============================
一键运行整套流程（尽量低耦合、便于维护）：
1) 训练残差 EDMD（名义模型一步残差）
2) 用训练出来的混合模型与名义模型做 MPC 轨迹跟踪对比

对比协议（公平性）
------------------
- 两套控制器使用 **同一套** NMPC 结构（代价函数、约束、求解器、超参数）。
- 唯一区别：预测模型
    - Nominal-MPC:  x_{k+1} = f_nom(x_k, u_k)
    - Hybrid-MPC:   x_{k+1} = f_nom(x_k, u_k) + r_hat(x_k, u_k)
- 真值被控对象：MuJoCo 绳驱模型（multi_joint_cable_dirven_space_robot.xml），
  控制输入为等效关节力矩 u=[tau_a,tau_b]，通过拮抗绳驱映射作用到 8 根绳执行器。

运行示例
--------
默认：训练 + 对比（推荐）
    python cdsm_mpc_tracking_compare.py

跳过训练：直接加载 residual EDMD 模型并对比
    python cdsm_mpc_tracking_compare.py --skip_train --model_dir <dir>

仅训练（不跑 MPC）
    python cdsm_mpc_tracking_compare.py --only_train

只跑名义 MPC（不使用混合预测器）
    python cdsm_mpc_tracking_compare.py --only_nominal --skip_train
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import minimize
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This script requires SciPy. Please install scipy (e.g. `pip install scipy`)."
    ) from e

from cdsm_rigid_nominal_model import CdsmRigidNominalModel, make_nominal_model
from utils_plot import get_save_dir, save_figure

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This script requires scikit-learn. Please install scikit-learn (e.g. `pip install scikit-learn`)."
    ) from e


# ============================================================
# Common: reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    np.random.seed(int(seed))


# ============================================================
# MuJoCo + cable helpers (self-contained)
# ============================================================
XML_DEFAULT = "multi_joint_cable_dirven_space_robot.xml"
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


def _require_mujoco() -> object:
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "This script needs the Python package `mujoco` to run data collection and MPC on the MuJoCo plant.\n"
            "Install it via `pip install mujoco` (or `pip install -r requirements.txt`)."
        ) from e
    return mujoco


def name_to_joint_id(model: object, name: str) -> int:
    mujoco = _require_mujoco()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in XML: {name}")
    return int(jid)


def name_to_actuator_id(model: object, name: str) -> int:
    mujoco = _require_mujoco()
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found in XML: {name}")
    return int(aid)


def name_to_tendon_id(model: object, name: str) -> int:
    mujoco = _require_mujoco()
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if tid < 0:
        raise ValueError(f"Tendon not found in XML: {name}")
    return int(tid)


def load_cable_model(xml_path: str, dt: float) -> Tuple[object, object, object, Dict[str, np.ndarray]]:
    mujoco = _require_mujoco()
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(dt)
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


def set_active_state(model: object, data: object, indices: Dict[str, np.ndarray], q: np.ndarray, dq: np.ndarray) -> None:
    mujoco = _require_mujoco()
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[indices["active_qpos"]] = q
    data.qvel[indices["active_dof"]] = dq
    for mimic_qpos, source_qpos in indices["mimic_pairs"]:
        data.qpos[mimic_qpos] = data.qpos[source_qpos]
    mujoco.mj_forward(model, data)


def get_active_state(data: object, indices: Dict[str, np.ndarray]) -> np.ndarray:
    q = data.qpos[indices["active_qpos"]]
    dq = data.qvel[indices["active_dof"]]
    return np.array([q[0], q[1], dq[0], dq[1]], dtype=np.float64)


def compute_tendon_jacobian_fd(
    model: object,
    scratch: object,
    q_ref: np.ndarray,
    tendon_ids_ordered: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    mujoco = _require_mujoco()
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

# -----------------------------
# Reference trajectory (reuse)
# -----------------------------
def cosine_ramp(t: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray]:
    tau = np.clip(t / max(T, 1e-9), 0.0, 1.0)
    s = 0.5 * (1.0 - np.cos(np.pi * tau))
    ds = 0.5 * np.pi / max(T, 1e-9) * np.sin(np.pi * tau)
    ds = np.where((t > 0) & (t < T), ds, 0.0)
    return s, ds


def build_joint_reference(
    dt: float,
    T_total: float,
    qa0: float,
    qa1: float,
    qb0: float,
    qb1: float,
    T_ramp: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    T_ramp = T_total if T_ramp is None else T_ramp
    t = np.arange(0.0, T_total + 0.5 * dt, dt)
    s, ds = cosine_ramp(t, T_ramp)
    qa = qa0 + (qa1 - qa0) * s
    qb = qb0 + (qb1 - qb0) * s
    dqa = (qa1 - qa0) * ds
    dqb = (qb1 - qb0) * ds
    q_ref = np.column_stack([qa, qb])
    dq_ref = np.column_stack([dqa, dqb])
    qdd_ref = np.zeros_like(q_ref)
    qdd_ref[1:, 0] = np.diff(dqa) / dt
    qdd_ref[1:, 1] = np.diff(dqb) / dt
    qdd_ref[0] = qdd_ref[1]
    return {"t": t, "q_ref": q_ref, "dq_ref": dq_ref, "qdd_ref": qdd_ref}


# ============================================================
# Residual EDMD (self-contained)
# ============================================================
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        return np.asarray(x_norm, dtype=np.float64) * self.std + self.mean


def build_feature_vector(x_raw: np.ndarray, u_raw: np.ndarray) -> np.ndarray:
    """
    z = [sin(qa), cos(qa), sin(qb), cos(qb), dqa, dqb, tau_a, tau_b]
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    u_raw = np.asarray(u_raw, dtype=np.float64)
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
    z_norm = np.asarray(z_norm, dtype=np.float64)
    cols = [np.ones((z_norm.shape[0], 1), dtype=np.float64), z_norm, z_norm * z_norm - 1.0]
    pairs = []
    for i in range(z_norm.shape[1]):
        for j in range(i + 1, z_norm.shape[1]):
            pairs.append((z_norm[:, i] * z_norm[:, j])[:, None])
    if pairs:
        cols.append(np.hstack(pairs))
    return np.hstack(cols)


def estimate_rbf_sigma(centers: np.ndarray) -> float:
    centers = np.asarray(centers, dtype=np.float64)
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
    z_norm = np.asarray(z_norm, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    diff = z_norm[:, None, :] - centers[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    rbf = np.exp(-0.5 * sqdist / (sigma * sigma))
    return np.hstack([np.ones((z_norm.shape[0], 1)), z_norm, rbf])


@dataclass
class ResidualEdmdConfig:
    dictionary: str  # "hermite" | "rbf"
    ridge: float
    rbf_centers: int
    rbf_sigma: Optional[float]
    rbf_seed: int


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
        self.weights = np.asarray(weights, dtype=np.float64)
        self.z_norm = z_norm
        self.r_norm = r_norm
        self.dictionary = dictionary
        self.centers = centers
        self.sigma = sigma
        self.cond_number = float(cond_number)
        self.feature_dim = int(feature_dim)

    def phi(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        u2 = np.atleast_2d(np.asarray(u, dtype=np.float64))
        z = build_feature_vector(x2, u2)
        zn = self.z_norm.transform(z)
        if self.dictionary == "hermite":
            return hermite_dictionary(zn)
        if self.dictionary == "rbf":
            if self.centers is None or self.sigma is None:
                raise RuntimeError("RBF model missing centers or sigma.")
            return rbf_dictionary(zn, self.centers, float(self.sigma))
        raise ValueError(f"Unknown dictionary: {self.dictionary}")

    def predict_residual(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        r_norm = self.phi(x, u) @ self.weights
        return self.r_norm.inverse(r_norm)[0]


def fit_residual_edmd(x: np.ndarray, u: np.ndarray, r: np.ndarray, cfg: ResidualEdmdConfig) -> ResidualEdmdModel:
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
        n_clusters = min(int(cfg.rbf_centers), zn.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=int(cfg.rbf_seed), batch_size=256, n_init=3
        )
        kmeans.fit(zn)
        centers = kmeans.cluster_centers_
        sigma = float(cfg.rbf_sigma) if cfg.rbf_sigma is not None else estimate_rbf_sigma(centers)
        phi = rbf_dictionary(zn, centers, sigma)
    else:
        raise ValueError(f"Unsupported dictionary: {cfg.dictionary}")

    gram = phi.T @ phi
    rhs = phi.T @ rn
    reg = float(cfg.ridge) * np.eye(gram.shape[0])
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
        feature_dim=int(phi.shape[1]),
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
        payload["sigma"] = np.array([float(model.sigma)])
    np.savez(path, **payload)


def load_residual_model(model_dir: Path) -> ResidualEdmdModel:
    npz_path = model_dir / "residual_edmd_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Residual EDMD model not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    dictionary = str(data["dictionary"][0])
    z_norm = Normalizer(data["z_mean"], data["z_std"])
    r_norm = Normalizer(data["r_mean"], data["r_std"])
    centers = data["centers"] if "centers" in data else None
    sigma = float(data["sigma"][0]) if "sigma" in data else None
    return ResidualEdmdModel(
        weights=data["weights"],
        z_norm=z_norm,
        r_norm=r_norm,
        dictionary=dictionary,
        centers=centers,
        sigma=sigma,
        cond_number=float(data["cond_number"][0]) if "cond_number" in data else 0.0,
        feature_dim=int(data["feature_dim"][0]) if "feature_dim" in data else 0,
    )


# -----------------------------
# NMPC (direct shooting)
# -----------------------------
@dataclass
class MpcConfig:
    horizon: int
    tau_max: float
    Qq: float
    Qdq: float
    R: float
    Rd: float
    maxiter: int


# ============================================================
# Data collection for residual training (PD)
# ============================================================
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
    q = params.A1 * np.sin(params.w1 * t + params.phi1) + params.A2 * np.sin(params.w2 * t + params.phi2)
    dq = (
        params.A1 * params.w1 * np.cos(params.w1 * t + params.phi1)
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
    return np.clip(tau, -tau_max, tau_max)


def collect_pd_trajectories(
    model: object,
    data: object,
    scratch: object,
    indices: Dict[str, np.ndarray],
    cfg: PDCollectConfig,
) -> Dict[str, np.ndarray]:
    mujoco = _require_mujoco()
    rng = np.random.RandomState(int(cfg.seed))
    n = int(cfg.traj_count)
    T = int(cfg.steps)

    states = np.zeros((n, T + 1, 4), dtype=np.float64)
    inputs = np.zeros((n, T, 2), dtype=np.float64)

    kp = np.array(cfg.kp, dtype=np.float64)
    kd = np.array(cfg.kd, dtype=np.float64)

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
            t = k * float(cfg.dt)
            qa_ref, dqa_ref = eval_sine_ref(ref_a, t)
            qb_ref, dqb_ref = eval_sine_ref(ref_b, t)
            q_ref = np.array([qa_ref, qb_ref], dtype=np.float64)
            dq_ref = np.array([dqa_ref, dqb_ref], dtype=np.float64)

            q = data.qpos[indices["active_qpos"]]
            dq = data.qvel[indices["active_dof"]]
            tau = pd_torque(q, dq, q_ref, dq_ref, kp, kd, float(cfg.tau_max))

            J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
            F_cable = cable_antagonistic_map(
                float(tau[0]), float(tau[1]), J, dof_j1, dof_j2, dof_j3, dof_j4
            )
            data.ctrl[indices["actuator_ids"]] = F_cable
            mujoco.mj_step(model, data)

            states[i, k + 1] = get_active_state(data, indices)
            inputs[i, k] = tau

    return {"states": states, "inputs": inputs}


def build_residual_dataset(
    dataset: Dict[str, np.ndarray],
    nominal: CdsmRigidNominalModel,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = dataset["states"]
    inputs = dataset["inputs"]
    n_traj, n_step, _ = inputs.shape

    x = states[:, :-1, :].reshape(-1, 4)
    u = inputs.reshape(-1, 2)
    xp_true = states[:, 1:, :].reshape(-1, 4)

    xp_nom = np.zeros_like(xp_true)
    for i in range(x.shape[0]):
        xp_nom[i] = nominal.step(x[i], u[i], dt=dt, apply_joint_limits=True)
    r = xp_true - xp_nom
    return x, u, r


def train_residual_edmd_end2end(
    *,
    xml: str,
    dt: float,
    nominal: CdsmRigidNominalModel,
    pd_cfg: PDCollectConfig,
    edmd_cfg: ResidualEdmdConfig,
    out_dir: Path,
) -> ResidualEdmdModel:
    model, data, scratch, indices = load_cable_model(xml, dt)
    raw = collect_pd_trajectories(model, data, scratch, indices, pd_cfg)
    np.savez(out_dir / "residual_train_dataset.npz", **raw)

    x, u, r = build_residual_dataset(raw, nominal, dt)
    edmd = fit_residual_edmd(x, u, r, edmd_cfg)
    save_residual_model(out_dir / "residual_edmd_model.npz", edmd, edmd_cfg)

    meta = {
        "xml": xml,
        "dt": dt,
        "pd_cfg": {
            "traj_count": pd_cfg.traj_count,
            "steps": pd_cfg.steps,
            "seed": pd_cfg.seed,
            "q_init_range": pd_cfg.q_init_range,
            "dq_init_range": pd_cfg.dq_init_range,
            "amp_range": list(pd_cfg.amp_range),
            "omega_range": list(pd_cfg.omega_range),
            "kp": list(pd_cfg.kp),
            "kd": list(pd_cfg.kd),
            "tau_max": pd_cfg.tau_max,
        },
        "edmd_cfg": {
            "dictionary": edmd_cfg.dictionary,
            "ridge": edmd_cfg.ridge,
            "rbf_centers": edmd_cfg.rbf_centers,
            "rbf_sigma": edmd_cfg.rbf_sigma,
            "rbf_seed": edmd_cfg.rbf_seed,
        },
        "train_stats": {
            "mean_residual_l2": float(np.linalg.norm(r, axis=1).mean()),
            "feature_dim": int(edmd.feature_dim),
            "cond_number": float(edmd.cond_number),
        },
    }
    with open(out_dir / "residual_train_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return edmd


def _clip_tau(u: np.ndarray, tau_max: float) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=np.float64), -tau_max, tau_max)


def predict_next_nominal(nominal: CdsmRigidNominalModel, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    return nominal.step(x, u, dt=dt, apply_joint_limits=True)


def predict_next_hybrid(
    nominal: CdsmRigidNominalModel,
    edmd: ResidualEdmdModel,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    x_nom = predict_next_nominal(nominal, x, u, dt)
    return x_nom + edmd.predict_residual(x, u)


def rollout_predict(
    x0: np.ndarray,
    u_seq: np.ndarray,
    *,
    dt: float,
    nominal: CdsmRigidNominalModel,
    edmd: Optional[ResidualEdmdModel],
) -> np.ndarray:
    u_seq = np.asarray(u_seq, dtype=np.float64)
    X = np.zeros((u_seq.shape[0] + 1, 4), dtype=np.float64)
    X[0] = np.asarray(x0, dtype=np.float64).reshape(4)
    x = X[0].copy()
    for k in range(u_seq.shape[0]):
        u = u_seq[k]
        if edmd is None:
            x = predict_next_nominal(nominal, x, u, dt)
        else:
            x = predict_next_hybrid(nominal, edmd, x, u, dt)
        X[k + 1] = x
    return X


def solve_mpc(
    x0: np.ndarray,
    ref_slice: Dict[str, np.ndarray],
    u_prev: np.ndarray,
    cfg: MpcConfig,
    *,
    dt: float,
    nominal: CdsmRigidNominalModel,
    edmd: Optional[ResidualEdmdModel],
    u_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Direct shooting NMPC:
      decision var: U = [u0..u_{N-1}] in R^{N x 2}
      cost: sum_k (q-qref)^T Q (q-qref) + (dq-dqref)^T Qd (dq-dqref) + u^T R u + (du)^T Rd (du)
      bounds: |tau| <= tau_max
    """
    N = int(cfg.horizon)
    q_ref = ref_slice["q_ref"][: N + 1]
    dq_ref = ref_slice["dq_ref"][: N + 1]

    u_prev = np.asarray(u_prev, dtype=np.float64).reshape(2)
    if u_init is None:
        u0 = np.tile(u_prev[None, :], (N, 1))
    else:
        u0 = np.asarray(u_init, dtype=np.float64).reshape(N, 2)
    u0 = _clip_tau(u0, cfg.tau_max)

    Q = np.diag([cfg.Qq, cfg.Qq, cfg.Qdq, cfg.Qdq]).astype(np.float64)

    def unpack(U_flat: np.ndarray) -> np.ndarray:
        return U_flat.reshape(N, 2)

    def obj(U_flat: np.ndarray) -> float:
        U = _clip_tau(unpack(U_flat), cfg.tau_max)
        X = rollout_predict(x0, U, dt=dt, nominal=nominal, edmd=edmd)
        cost = 0.0
        for k in range(N + 1):
            xk = X[k]
            ek = np.array([xk[0] - q_ref[k, 0], xk[1] - q_ref[k, 1], xk[2] - dq_ref[k, 0], xk[3] - dq_ref[k, 1]])
            cost += float(ek @ Q @ ek)
            if k < N:
                uk = U[k]
                cost += cfg.R * float(uk @ uk)
                du = uk - (u_prev if k == 0 else U[k - 1])
                cost += cfg.Rd * float(du @ du)
        return cost

    bounds = [(-cfg.tau_max, cfg.tau_max)] * (N * 2)
    t0 = time.perf_counter()
    res = minimize(
        obj,
        x0=u0.reshape(-1),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(cfg.maxiter), "ftol": 1e-6},
    )
    solve_ms = 1000.0 * (time.perf_counter() - t0)
    U_opt = _clip_tau(unpack(res.x), cfg.tau_max)
    info = {
        "success": float(1.0 if res.success else 0.0),
        "status": float(res.status),
        "iters": float(res.nit if hasattr(res, "nit") else 0.0),
        "solve_ms": float(solve_ms),
        "obj": float(res.fun),
    }
    return U_opt, info


# -----------------------------
# Closed-loop on MuJoCo plant
# -----------------------------
def run_mpc_on_mujoco(
    *,
    label: str,
    nominal: CdsmRigidNominalModel,
    edmd: Optional[ResidualEdmdModel],
    mpc_cfg: MpcConfig,
    xml: str,
    dt: float,
    ref: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, np.ndarray]:
    mujoco = _require_mujoco()

    model, data, scratch, indices = load_cable_model(xml, dt)
    n_step = len(ref["t"]) - 1

    # init to reference start
    q0_ref = ref["q_ref"][0]
    dq0_ref = ref["dq_ref"][0]
    set_active_state(model, data, indices, q0_ref, dq0_ref)
    mujoco.mj_forward(model, data)

    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])

    u_prev = np.zeros(2, dtype=np.float64)
    u_warm = None

    rec_t, rec_x, rec_u, rec_u_solve, rec_ok = [], [], [], [], []
    rec_q_ref, rec_dq_ref = [], []
    rec_solve_ms, rec_obj = [], []

    rng = np.random.RandomState(seed)
    _ = rng  # placeholder (keep seed deterministic if later add noise)

    for k in range(n_step):
        x_meas = get_active_state(data, indices)
        # horizon reference slice
        k_end = min(k + mpc_cfg.horizon + 1, ref["q_ref"].shape[0])
        q_ref_slice = ref["q_ref"][k:k_end]
        dq_ref_slice = ref["dq_ref"][k:k_end]
        # pad to N+1
        if q_ref_slice.shape[0] < mpc_cfg.horizon + 1:
            pad_n = mpc_cfg.horizon + 1 - q_ref_slice.shape[0]
            q_ref_slice = np.vstack([q_ref_slice, np.tile(q_ref_slice[-1], (pad_n, 1))])
            dq_ref_slice = np.vstack([dq_ref_slice, np.tile(dq_ref_slice[-1], (pad_n, 1))])

        ref_slice = {"q_ref": q_ref_slice, "dq_ref": dq_ref_slice}
        U_opt, info = solve_mpc(
            x_meas,
            ref_slice,
            u_prev,
            mpc_cfg,
            dt=dt,
            nominal=nominal,
            edmd=edmd,
            u_init=u_warm,
        )
        u_cmd = U_opt[0]
        u_cmd = _clip_tau(u_cmd, mpc_cfg.tau_max)

        # apply to MuJoCo via cable map
        J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
        F_cable = cable_antagonistic_map(
            float(u_cmd[0]), float(u_cmd[1]), J, dof_j1, dof_j2, dof_j3, dof_j4
        )
        data.ctrl[indices["actuator_ids"]] = F_cable
        mujoco.mj_step(model, data)

        # warm start: shift U
        u_warm = np.vstack([U_opt[1:], U_opt[-1:]])
        u_prev = u_cmd.copy()

        rec_t.append(float(data.time))
        rec_x.append(x_meas.copy())
        rec_u.append(u_cmd.copy())
        rec_u_solve.append(U_opt.copy())
        rec_ok.append(info["success"])
        rec_q_ref.append(ref["q_ref"][k].copy())
        rec_dq_ref.append(ref["dq_ref"][k].copy())
        rec_solve_ms.append(info["solve_ms"])
        rec_obj.append(info["obj"])

    log = {
        "label": np.array([label]),
        "t": np.array(rec_t),
        "x": np.array(rec_x),
        "u": np.array(rec_u),
        "q_ref": np.array(rec_q_ref),
        "dq_ref": np.array(rec_dq_ref),
        "solve_ms": np.array(rec_solve_ms),
        "obj": np.array(rec_obj),
        "success": np.array(rec_ok),
    }
    return log


def run_mpc_on_nominal_plant(
    *,
    label: str,
    nominal: CdsmRigidNominalModel,
    edmd: Optional[ResidualEdmdModel],
    mpc_cfg: MpcConfig,
    dt: float,
    ref: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    纯名义动力学作为被控对象（无 MuJoCo 依赖），用于在没有 mujoco 包时先完成对比。
    被控对象始终使用名义 step；控制器的预测模型由 edmd=None/edmd 决定。
    """
    n_step = len(ref["t"]) - 1
    x = np.array([ref["q_ref"][0, 0], ref["q_ref"][0, 1], ref["dq_ref"][0, 0], ref["dq_ref"][0, 1]], dtype=np.float64)
    u_prev = np.zeros(2, dtype=np.float64)
    u_warm = None

    rec_t, rec_x, rec_u, rec_ok = [], [], [], []
    rec_q_ref, rec_dq_ref = [], []
    rec_solve_ms, rec_obj = [], []

    for k in range(n_step):
        # horizon reference slice
        k_end = min(k + mpc_cfg.horizon + 1, ref["q_ref"].shape[0])
        q_ref_slice = ref["q_ref"][k:k_end]
        dq_ref_slice = ref["dq_ref"][k:k_end]
        if q_ref_slice.shape[0] < mpc_cfg.horizon + 1:
            pad_n = mpc_cfg.horizon + 1 - q_ref_slice.shape[0]
            q_ref_slice = np.vstack([q_ref_slice, np.tile(q_ref_slice[-1], (pad_n, 1))])
            dq_ref_slice = np.vstack([dq_ref_slice, np.tile(dq_ref_slice[-1], (pad_n, 1))])

        ref_slice = {"q_ref": q_ref_slice, "dq_ref": dq_ref_slice}
        U_opt, info = solve_mpc(
            x,
            ref_slice,
            u_prev,
            mpc_cfg,
            dt=dt,
            nominal=nominal,
            edmd=edmd,
            u_init=u_warm,
        )
        u_cmd = _clip_tau(U_opt[0], mpc_cfg.tau_max)

        # plant update uses nominal dynamics (no residual)
        x = predict_next_nominal(nominal, x, u_cmd, dt)

        u_warm = np.vstack([U_opt[1:], U_opt[-1:]])
        u_prev = u_cmd.copy()

        rec_t.append(float((k + 1) * dt))
        rec_x.append(x.copy())
        rec_u.append(u_cmd.copy())
        rec_ok.append(info["success"])
        rec_q_ref.append(ref["q_ref"][k].copy())
        rec_dq_ref.append(ref["dq_ref"][k].copy())
        rec_solve_ms.append(info["solve_ms"])
        rec_obj.append(info["obj"])

    return {
        "label": np.array([label]),
        "t": np.array(rec_t),
        "x": np.array(rec_x),
        "u": np.array(rec_u),
        "q_ref": np.array(rec_q_ref),
        "dq_ref": np.array(rec_dq_ref),
        "solve_ms": np.array(rec_solve_ms),
        "obj": np.array(rec_obj),
        "success": np.array(rec_ok),
    }


def tracking_metrics(log: Dict[str, np.ndarray]) -> Dict[str, float]:
    q = log["x"][:, :2]
    dq = log["x"][:, 2:]
    e = q - log["q_ref"]
    edq = dq - log["dq_ref"]
    return {
        "rmse_q": float(np.sqrt(np.mean(e * e))),
        "mae_q": float(np.mean(np.abs(e))),
        "max_abs_q": float(np.max(np.abs(e))),
        "rmse_dq": float(np.sqrt(np.mean(edq * edq))),
        "mean_solve_ms": float(np.mean(log["solve_ms"])),
        "success_ratio": float(np.mean(log["success"])),
    }


def plot_compare(log_nom: Dict[str, np.ndarray], log_hyb: Optional[Dict[str, np.ndarray]]) -> None:
    t = log_nom["t"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    labels = ["q_a", "q_b", "dq_a", "dq_b"]
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(t, log_nom["x"][:, i], lw=1.6, label="Nominal-MPC")
        if i < 2:
            ax.plot(t, log_nom["q_ref"][:, i], "k--", lw=1.4, label="ref" if i == 0 else None)
        else:
            ax.plot(t, log_nom["dq_ref"][:, i - 2], "k--", lw=1.4, label="ref" if i == 2 else None)
        if log_hyb is not None:
            ax.plot(log_hyb["t"], log_hyb["x"][:, i], lw=1.6, label="Hybrid-MPC")
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Joint-space tracking: Nominal-MPC vs Hybrid-MPC (MuJoCo plant)")
    fig.tight_layout()
    save_figure("mpc_tracking_compare_states")
    plt.close(fig)

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax2[0].plot(t, log_nom["u"][:, 0], lw=1.6, label="tau_a nominal")
    ax2[0].plot(t, log_nom["u"][:, 1], lw=1.6, label="tau_b nominal")
    if log_hyb is not None:
        ax2[0].plot(log_hyb["t"], log_hyb["u"][:, 0], lw=1.3, label="tau_a hybrid")
        ax2[0].plot(log_hyb["t"], log_hyb["u"][:, 1], lw=1.3, label="tau_b hybrid")
    ax2[0].set_ylabel("Torque (Nm)")
    ax2[0].legend(fontsize=8)
    ax2[0].grid(True, alpha=0.3)
    ax2[1].plot(t, log_nom["solve_ms"], lw=1.6, label="solve ms nominal")
    if log_hyb is not None:
        ax2[1].plot(log_hyb["t"], log_hyb["solve_ms"], lw=1.6, label="solve ms hybrid")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_ylabel("Solve time (ms)")
    ax2[1].legend(fontsize=8)
    ax2[1].grid(True, alpha=0.3)
    fig2.suptitle("MPC control + solver time")
    fig2.tight_layout()
    save_figure("mpc_tracking_compare_u_solve")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(9, 4))
    e_nom = log_nom["x"][:, :2] - log_nom["q_ref"]
    ax3.plot(t, np.linalg.norm(e_nom, axis=1), lw=1.6, label="|e_q| nominal")
    if log_hyb is not None:
        e_hyb = log_hyb["x"][:, :2] - log_hyb["q_ref"]
        ax3.plot(log_hyb["t"], np.linalg.norm(e_hyb, axis=1), lw=1.6, label="|e_q| hybrid")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Joint position error norm (rad)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    save_figure("mpc_tracking_compare_error_norm")
    plt.close(fig3)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Joint-space NMPC tracking on MuJoCo cable plant (nominal vs hybrid predictor).")
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--plant", choices=["mujoco", "nominal"], default="mujoco", help="Plant for closed-loop test.")
    p.add_argument("--skip_train", action="store_true", help="Skip EDMD training; load from --model_dir.")
    p.add_argument("--only_train", action="store_true", help="Only train residual EDMD; do not run MPC compare.")
    p.add_argument("--model_dir", type=str, default=None, help="Directory containing residual_edmd_model.npz")
    p.add_argument("--only_nominal", action="store_true", help="Run only nominal MPC (ignore hybrid).")

    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--T_track", type=float, default=6.0)
    p.add_argument("--T_ramp", type=float, default=5.0)
    p.add_argument("--qa0", type=float, default=-0.5)
    p.add_argument("--qa1", type=float, default=0.5)
    p.add_argument("--qb0", type=float, default=0.4)
    p.add_argument("--qb1", type=float, default=-0.4)

    # MPC hyperparameters
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--tau_max", type=float, default=45.0)
    p.add_argument("--Qq", type=float, default=40.0)
    p.add_argument("--Qdq", type=float, default=2.0)
    p.add_argument("--R", type=float, default=0.02)
    p.add_argument("--Rd", type=float, default=0.2)
    p.add_argument("--maxiter", type=int, default=80)

    # training hyperparameters (PD collect + EDMD)
    p.add_argument("--train_traj", type=int, default=120)
    p.add_argument("--train_steps", type=int, default=250)
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
    p.add_argument("--tau_max_train", type=float, default=45.0)
    p.add_argument("--dictionary", choices=["hermite", "rbf"], default="hermite")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--rbf_centers", type=int, default=200)
    p.add_argument("--rbf_sigma", type=float, default=None)
    p.add_argument("--rbf_seed", type=int, default=2007)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(get_save_dir())
    t0 = time.time()

    nominal = make_nominal_model(dt=args.dt)
    edmd: Optional[ResidualEdmdModel] = None

    # ---- Step 1: train/load residual EDMD ----
    if not args.only_nominal:
        if args.skip_train:
            if not args.model_dir:
                raise ValueError("--skip_train requires --model_dir")
            edmd = load_residual_model(Path(args.model_dir))
        else:
            if args.plant != "mujoco":
                raise ValueError("Training requires MuJoCo plant. Use --plant mujoco or pass --skip_train with --model_dir.")
            pd_cfg = PDCollectConfig(
                traj_count=args.train_traj,
                steps=args.train_steps,
                dt=args.dt,
                seed=args.seed,
                q_init_range=args.q_init_range,
                dq_init_range=args.dq_init_range,
                amp_range=(args.amp_min, args.amp_max),
                omega_range=(args.omega_min, args.omega_max),
                kp=(args.kp_a, args.kp_b),
                kd=(args.kd_a, args.kd_b),
                tau_max=args.tau_max_train,
            )
            edmd_cfg = ResidualEdmdConfig(
                dictionary=args.dictionary,
                ridge=args.ridge,
                rbf_centers=args.rbf_centers,
                rbf_sigma=args.rbf_sigma,
                rbf_seed=args.rbf_seed,
            )
            print("[train] Collecting PD data and fitting residual EDMD...")
            edmd = train_residual_edmd_end2end(
                xml=args.xml,
                dt=args.dt,
                nominal=nominal,
                pd_cfg=pd_cfg,
                edmd_cfg=edmd_cfg,
                out_dir=out_dir,
            )

    if args.only_train:
        print(f"[done] training outputs -> {out_dir}")
        return

    ref = build_joint_reference(
        dt=args.dt,
        T_total=args.T_track,
        qa0=args.qa0,
        qa1=args.qa1,
        qb0=args.qb0,
        qb1=args.qb1,
        T_ramp=args.T_ramp,
    )

    mpc_cfg = MpcConfig(
        horizon=args.horizon,
        tau_max=args.tau_max,
        Qq=args.Qq,
        Qdq=args.Qdq,
        R=args.R,
        Rd=args.Rd,
        maxiter=args.maxiter,
    )

    print("[mpc] Running Nominal-MPC on MuJoCo plant...")
    if args.plant == "mujoco":
        log_nom = run_mpc_on_mujoco(
            label="nominal_mpc",
            nominal=nominal,
            edmd=None,
            mpc_cfg=mpc_cfg,
            xml=args.xml,
            dt=args.dt,
            ref=ref,
            seed=args.seed,
        )
    else:
        log_nom = run_mpc_on_nominal_plant(
            label="nominal_mpc",
            nominal=nominal,
            edmd=None,
            mpc_cfg=mpc_cfg,
            dt=args.dt,
            ref=ref,
        )
    met_nom = tracking_metrics(log_nom)

    log_hyb = None
    met_hyb = None
    if edmd is not None:
        print("[mpc] Running Hybrid-MPC on MuJoCo plant...")
        if args.plant == "mujoco":
            log_hyb = run_mpc_on_mujoco(
                label="hybrid_mpc",
                nominal=nominal,
                edmd=edmd,
                mpc_cfg=mpc_cfg,
                xml=args.xml,
                dt=args.dt,
                ref=ref,
                seed=args.seed,
            )
        else:
            log_hyb = run_mpc_on_nominal_plant(
                label="hybrid_mpc",
                nominal=nominal,
                edmd=edmd,
                mpc_cfg=mpc_cfg,
                dt=args.dt,
                ref=ref,
            )
        met_hyb = tracking_metrics(log_hyb)

    np.savez(out_dir / "mpc_nominal_log.npz", **log_nom)
    if log_hyb is not None:
        np.savez(out_dir / "mpc_hybrid_log.npz", **log_hyb)

    plot_compare(log_nom, log_hyb)

    summary = {
        "xml": args.xml,
        "dt": args.dt,
        "mpc_config": vars(args),
        "metrics_nominal": met_nom,
        "metrics_hybrid": met_hyb,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "mpc_tracking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"  Nominal-MPC: RMSE(q)={met_nom['rmse_q']:.6g} rad, mean solve={met_nom['mean_solve_ms']:.2f} ms, success={met_nom['success_ratio']:.3f}")
    if met_hyb is not None:
        print(f"  Hybrid-MPC : RMSE(q)={met_hyb['rmse_q']:.6g} rad, mean solve={met_hyb['mean_solve_ms']:.2f} ms, success={met_hyb['success_ratio']:.3f}")
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()

