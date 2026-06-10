"""
cdsm_dkn_vs_edmd_prediction_compare.py
======================================

Prediction-only comparison on the MuJoCo cable-driven space manipulator:

    1. DKN: Shi & Meng style Deep Koopman Nonlinear model
       z = [x_n, phi(x_n)]
       u_hat = control_net(x_n, u_n)
       z_next = A z + B u_hat

    2. EDMD: fixed RBF dictionary with ridge regression
       z = [x_n, rbf(x_n)]
       z_next = A z + B u_n

The script only evaluates prediction. It does not run LQR/MPC and does not
attempt to invert the DKN control network. Data are collected from the MuJoCo
cable-driven plant with PD multi-sine trajectories. Torque clipping and cable
maximum-tension clipping are intentionally disabled in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires MuJoCo and PyTorch in the configured environment.") from exc

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires scikit-learn in the configured environment.") from exc

from utils_plot import get_save_dir, save_figure

XML_DEFAULT = str(
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "multi_joint_cable_driven_space_robot.xml"
)
STATE_LABELS = ["q_a", "q_b", "dq_a", "dq_b"]
STATE_DIM = 4
CONTROL_DIM = 2

# MuJoCo cable model names. This script keeps the collection logic local so it
# is independent of the residual-EDMD and previous comparison scripts.
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

# Preload is kept as the antagonistic cable-map baseline. Maximum tension
# clipping is disabled as requested.
F_PRELOAD = 20.0
F_MAX_CABLE: Optional[float] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        return np.asarray(x_norm, dtype=np.float64) * self.std + self.mean

    def to_json(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ============================================================================
# MuJoCo data collection
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
    tau_base = (m_p + m_m) * f_pre
    tau_eff = tau_des - tau_base
    eps = 1e-12
    candidates: List[Tuple[float, float, float, float]] = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > eps:
        u = max(tau_eff / m_p, 0.0)
        # Maximum cable tension clipping intentionally disabled:
        # u = min(u, f_max - f_pre)
        candidates.append((u, 0.0, abs(tau_eff - m_p * u), u))
    if abs(m_m) > eps:
        u = max(tau_eff / m_m, 0.0)
        # Maximum cable tension clipping intentionally disabled:
        # u = min(u, f_max - f_pre)
        candidates.append((0.0, u, abs(tau_eff - m_m * u), u))
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
    # Torque clipping intentionally disabled:
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
                float(tau[0]), float(tau[1]), J, dof_j1, dof_j2, dof_j3, dof_j4
            )
            data.ctrl[indices["actuator_ids"]] = F_cable
            mujoco.mj_step(model, data)
            states[i, k + 1] = get_active_state(data, indices)
            inputs[i, k] = tau
            cable_ctrl[i, k] = F_cable

    meta = {
        "kp": list(cfg.kp),
        "kd": list(cfg.kd),
        "tau_max": cfg.tau_max,
        "torque_limit_enabled": False,
        "f_preload": F_PRELOAD,
        "f_max_cable": F_MAX_CABLE,
        "cable_tension_limit_enabled": False,
        "control_mode": "pd_joint_torque_via_unlimited_local_cable_map",
    }
    return {
        "states": states,
        "inputs": inputs,
        "q_ref": q_ref_hist,
        "cable_ctrl": cable_ctrl,
    }, meta


def build_windows(states: np.ndarray, inputs: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n_traj, n_times, _ = states.shape
    n_step = inputs.shape[1]
    if n_step != n_times - 1:
        raise ValueError(f"inputs steps {n_step} != states-1 {n_times - 1}")
    if window > n_step:
        raise ValueError(f"window {window} larger than trajectory steps {n_step}")
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    for i in range(n_traj):
        for k in range(n_step - window + 1):
            xs.append(states[i, k : k + window + 1])
            us.append(inputs[i, k : k + window])
    return np.asarray(xs, dtype=np.float64), np.asarray(us, dtype=np.float64)


def sample_window_batch(
    Xw_norm: np.ndarray, Uw_norm: np.ndarray, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = Xw_norm.shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    x = torch.from_numpy(Xw_norm[idx].astype(np.float32)).to(device)
    u = torch.from_numpy(Uw_norm[idx].astype(np.float32)).to(device)
    return x, u


# ============================================================================
# DKN: Deep Koopman Nonlinear prediction model
# ============================================================================
@dataclass
class DKNConfig:
    lift_dim: int
    hidden: Tuple[int, ...]
    control_hidden: Tuple[int, ...]
    control_dim_hat: int
    activation: str
    bound_lift: bool
    window: int
    window_start: int
    epochs: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    grad_clip: float
    weight_decay: float
    w_state: float
    w_embed: float


class MLP(nn.Module):
    def __init__(self, widths: Tuple[int, ...], activation: str = "elu") -> None:
        super().__init__()
        act_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: List[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i != len(widths) - 2:
                layers.append(act_map[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DKNModel(nn.Module):
    """
    Shi & Meng style DKN for prediction:

        z = [x_n, phi(x_n)]
        u_hat = control_net([x_n, u_n])
        z_next = A z + B u_hat
        x_next_n = C z_next = z_next[:STATE_DIM]
    """

    def __init__(
        self,
        lift_dim: int,
        hidden: Tuple[int, ...],
        control_hidden: Tuple[int, ...],
        control_dim_hat: int,
        activation: str,
        bound_lift: bool,
    ) -> None:
        super().__init__()
        self.state_dim = STATE_DIM
        self.control_dim = CONTROL_DIM
        self.lift_dim = int(lift_dim)
        self.latent_dim = STATE_DIM + self.lift_dim
        self.control_dim_hat = int(control_dim_hat)
        self.bound_lift = bool(bound_lift)
        self.lift = MLP((STATE_DIM,) + tuple(hidden) + (self.lift_dim,), activation)
        self.control_net = MLP(
            (STATE_DIM + CONTROL_DIM,) + tuple(control_hidden) + (self.control_dim_hat,),
            activation,
        )
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim_hat, self.latent_dim, bias=False)
        self._init_linear_dynamics()

    def _init_linear_dynamics(self) -> None:
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.latent_dim))
            nn.init.xavier_uniform_(self.B.weight, gain=0.05)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        g = self.lift(x_norm)
        if self.bound_lift:
            g = torch.tanh(g)
        return torch.cat([x_norm, g], dim=-1)

    @staticmethod
    def state_from_latent(z: torch.Tensor) -> torch.Tensor:
        return z[..., :STATE_DIM]

    def control_encode(self, x_norm: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        return self.control_net(torch.cat([x_norm, u_norm], dim=-1))

    def koopman_step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        x_for_control = self.state_from_latent(z)
        u_hat = self.control_encode(x_for_control, u_norm)
        return self.A(z) + self.B(u_hat)


def curriculum_horizon(epoch: int, cfg: DKNConfig) -> int:
    w0 = max(1, min(cfg.window_start, cfg.window))
    ramp_epochs = max(1, int(0.6 * cfg.epochs))
    frac = min(1.0, (epoch - 1) / max(1, ramp_epochs - 1))
    return int(round(w0 + frac * (cfg.window - w0)))


def compute_dkn_losses(
    model: DKNModel,
    x_win: torch.Tensor,
    u_win: torch.Tensor,
    cfg: DKNConfig,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    S = int(horizon)
    xw = x_win[:, : S + 1]
    uw = u_win[:, :S]
    B = xw.shape[0]
    z_true = model.encode(xw.reshape(B * (S + 1), -1)).reshape(B, S + 1, -1)
    z = z_true[:, 0]
    state_loss = xw.new_zeros(())
    embed_loss = xw.new_zeros(())
    for k in range(S):
        z = model.koopman_step(z, uw[:, k])
        x_pred = model.state_from_latent(z)
        state_loss = state_loss + torch.mean((x_pred - xw[:, k + 1]) ** 2)
        embed_loss = embed_loss + torch.mean((z - z_true[:, k + 1]) ** 2)
    state_loss = state_loss / S
    embed_loss = embed_loss / S
    total = cfg.w_state * state_loss + cfg.w_embed * embed_loss
    return {"total": total, "state": state_loss, "embed": embed_loss}


@torch.no_grad()
def evaluate_dkn_window_loss(
    model: DKNModel,
    Xw_norm: np.ndarray,
    Uw_norm: np.ndarray,
    cfg: DKNConfig,
    device: torch.device,
    n_batches: int = 5,
) -> Dict[str, float]:
    model.eval()
    acc = {"total": 0.0, "state": 0.0, "embed": 0.0}
    for _ in range(n_batches):
        x, u = sample_window_batch(Xw_norm, Uw_norm, cfg.batch_size, device)
        losses = compute_dkn_losses(model, x, u, cfg, cfg.window)
        for key in acc:
            acc[key] += float(losses[key].item())
    for key in acc:
        acc[key] /= max(n_batches, 1)
    return acc


def train_dkn(
    train_win: Tuple[np.ndarray, np.ndarray],
    val_win: Tuple[np.ndarray, np.ndarray],
    cfg: DKNConfig,
    device: torch.device,
    out_dir: Path,
) -> Tuple[DKNModel, List[List[float]], Dict[str, float]]:
    Xw_tr, Uw_tr = train_win
    Xw_va, Uw_va = val_win
    model = DKNModel(
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        control_hidden=tuple(cfg.control_hidden),
        control_dim_hat=cfg.control_dim_hat,
        activation=cfg.activation,
        bound_lift=cfg.bound_lift,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_epoch = 0
    history: List[List[float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        horizon = curriculum_horizon(epoch, cfg)
        acc = {"total": 0.0, "state": 0.0, "embed": 0.0}
        for _ in range(cfg.steps_per_epoch):
            x, u = sample_window_batch(Xw_tr, Uw_tr, cfg.batch_size, device)
            losses = compute_dkn_losses(model, x, u, cfg, horizon)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for key in acc:
                acc[key] += float(losses[key].detach().item())
        for key in acc:
            acc[key] /= max(cfg.steps_per_epoch, 1)

        val = evaluate_dkn_window_loss(model, Xw_va, Uw_va, cfg, device)
        history.append(
            [
                float(epoch),
                acc["total"],
                acc["state"],
                acc["embed"],
                val["total"],
                val["state"],
                val["embed"],
            ]
        )
        print(
            f"[dkn] epoch {epoch:03d}/{cfg.epochs:03d} H={horizon:02d} "
            f"train={acc['total']:.3e} (state={acc['state']:.3e} embed={acc['embed']:.3e}) "
            f"val={val['total']:.3e}",
            flush=True,
        )
        if val["total"] < best_val:
            best_val = val["total"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val": best_val,
                    "epoch": epoch,
                },
                out_dir / "best_dkn.pt",
            )

    ckpt = torch.load(out_dir / "best_dkn.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, history, {"best_val": float(best_val), "best_epoch": float(best_epoch)}


class DKNPredictor:
    def __init__(
        self,
        model: DKNModel,
        x_normer: Normalizer,
        u_normer: Normalizer,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.x_normer = x_normer
        self.u_normer = u_normer
        self.device = device
        self.latent_dim = model.latent_dim
        self.name = "DKN"

    @torch.no_grad()
    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_n = self.x_normer.transform(np.atleast_2d(x_phys))
        z = self.model.encode(torch.from_numpy(x_n.astype(np.float32)).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    @torch.no_grad()
    def step(self, z: np.ndarray, u_n: np.ndarray) -> np.ndarray:
        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32).reshape(1, -1)).to(self.device)
        u_t = torch.from_numpy(np.asarray(u_n, dtype=np.float32).reshape(1, -1)).to(self.device)
        z_next = self.model.koopman_step(z_t, u_t)
        return z_next.cpu().numpy().reshape(-1).astype(np.float64)

    def recover(self, z: np.ndarray) -> np.ndarray:
        x_n = np.asarray(z, dtype=np.float64).reshape(-1)[:STATE_DIM]
        return self.x_normer.inverse(x_n)


# ============================================================================
# EDMD predictor
# ============================================================================
@dataclass
class EdmdConfig:
    n_centers: int
    rbf_sigma: Optional[float]
    ridge: float
    kmeans_seed: int


def estimate_rbf_sigma(centers: np.ndarray) -> float:
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


class EdmdPredictor:
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
        self.centers = np.asarray(centers, dtype=np.float64)
        self.sigma = float(sigma)
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.cond_number = float(cond_number)
        self.latent_dim = A.shape[0]
        self.name = "Koopman-EDMD"

    def _lift_norm(self, x_n: np.ndarray) -> np.ndarray:
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

    tmp = EdmdPredictor(
        x_normer,
        u_normer,
        centers,
        sigma,
        A=np.zeros((STATE_DIM + n_clusters, STATE_DIM + n_clusters)),
        B=np.zeros((STATE_DIM + n_clusters, CONTROL_DIM)),
    )
    Z = tmp._lift_norm(Xn)
    Zp = tmp._lift_norm(Xpn)
    Omega = np.hstack([Z, Un])
    M = Omega.shape[0]
    gram = (Omega.T @ Omega) / M
    rhs = (Omega.T @ Zp) / M
    reg = cfg.ridge * np.eye(gram.shape[0])
    G = np.linalg.solve(gram + reg, rhs)
    cond_number = float(np.linalg.cond(gram))

    D = Z.shape[1]
    A = G[:D, :].T.copy()
    B = G[D:, :].T.copy()
    return EdmdPredictor(x_normer, u_normer, centers, sigma, A, B, cond_number)


# ============================================================================
# Prediction evaluation and plotting
# ============================================================================
def predict_trajectory(pred: object, states: np.ndarray, inputs: np.ndarray, mode: str) -> np.ndarray:
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


def evaluate_predictor(pred: object, val_raw: Dict[str, np.ndarray], mode: str) -> Dict[str, object]:
    states = val_raw["states"]
    inputs = val_raw["inputs"]
    preds = np.zeros_like(states)
    for i in range(states.shape[0]):
        preds[i] = predict_trajectory(pred, states[i], inputs[i], mode)
    err = preds[:, 1:, :] - states[:, 1:, :]
    step_err = preds - states
    return {
        "preds": preds,
        "states_true": states,
        "rmse_by_state": np.sqrt(np.mean(err * err, axis=(0, 1))),
        "step_rmse": np.sqrt(np.mean(step_err * step_err, axis=(0, 2))),
        "total_rmse": float(np.sqrt(np.mean(err * err))),
    }


def plot_dynamic_response(
    res_dkn: Dict[str, object],
    res_ed: Dict[str, object],
    dt: float,
    traj_idx: int,
    mode: str,
    prefix: str,
) -> None:
    true = res_dkn["states_true"][traj_idx]
    dkn = res_dkn["preds"][traj_idx]
    ed = res_ed["preds"][traj_idx]
    t = np.arange(true.shape[0]) * dt
    units = ["rad", "rad", "rad/s", "rad/s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for i, label in enumerate(STATE_LABELS):
        ax = axes[i // 2, i % 2]
        ax.plot(t, true[:, i], "k-", lw=1.8, label="MuJoCo")
        ax.plot(t, dkn[:, i], "--", lw=1.4, color="C0", label="DKN")
        ax.plot(t, ed[:, i], "-.", lw=1.4, color="C1", label="Koopman-EDMD")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{label} ({units[i]})")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{mode} prediction: MuJoCo vs DKN vs Koopman-EDMD")
    fig.tight_layout()
    save_figure(f"{prefix}_dynamic_response")
    plt.close(fig)


def plot_error_curve(
    res_dkn: Dict[str, object],
    res_ed: Dict[str, object],
    dt: float,
    traj_idx: int,
    mode: str,
    prefix: str,
) -> None:
    true = res_dkn["states_true"][traj_idx]
    err_dkn = res_dkn["preds"][traj_idx] - true
    err_ed = res_ed["preds"][traj_idx] - true
    t = np.arange(true.shape[0]) * dt
    e_dkn = np.sqrt(np.mean(err_dkn * err_dkn, axis=1))
    e_ed = np.sqrt(np.mean(err_ed * err_ed, axis=1))
    rmse_dkn = float(np.sqrt(np.mean(err_dkn[1:] * err_dkn[1:])))
    rmse_ed = float(np.sqrt(np.mean(err_ed[1:] * err_ed[1:])))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, e_dkn, lw=1.6, color="C0", label=f"DKN (RMSE={rmse_dkn:.3g})")
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
    res_dkn: Dict[str, object],
    res_ed: Dict[str, object],
    dt: float,
    mode: str,
    prefix: str,
) -> None:
    sd = res_dkn["step_rmse"]
    se = res_ed["step_rmse"]
    t = np.arange(sd.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, sd, lw=1.8, color="C0", label="DKN")
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
    res_dkn: Dict[str, object],
    res_ed: Dict[str, object],
    mode: str,
    prefix: str,
) -> None:
    rd = res_dkn["rmse_by_state"]
    re = res_ed["rmse_by_state"]
    x = np.arange(STATE_DIM)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, rd, width, color="C0", label="DKN")
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
    dkn_pred: DKNPredictor,
    ed_pred: EdmdPredictor,
    val_raw: Dict[str, np.ndarray],
    dt: float,
    demo_idx: int,
) -> Dict[str, object]:
    print(f"  [{mode}] evaluating DKN & Koopman-EDMD...")
    res_dkn = evaluate_predictor(dkn_pred, val_raw, mode)
    res_ed = evaluate_predictor(ed_pred, val_raw, mode)
    prefix = mode
    plot_dynamic_response(res_dkn, res_ed, dt, demo_idx, mode, prefix)
    plot_error_curve(res_dkn, res_ed, dt, demo_idx, mode, prefix)
    plot_rmse_growth(res_dkn, res_ed, dt, mode, prefix)
    plot_rmse_by_state(res_dkn, res_ed, mode, prefix)
    print(
        f"  [{mode}] total RMSE: DKN={res_dkn['total_rmse']:.6g}  "
        f"Koopman-EDMD={res_ed['total_rmse']:.6g}"
    )
    return {
        "dkn": {
            "total_rmse": res_dkn["total_rmse"],
            "rmse_by_state": res_dkn["rmse_by_state"].tolist(),
        },
        "koopman_edmd": {
            "total_rmse": res_ed["total_rmse"],
            "rmse_by_state": res_ed["rmse_by_state"].tolist(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CDSM prediction comparison: DKN vs Koopman-EDMD.")
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--pred_mode", choices=["one_step", "rollout", "both"], default="rollout")

    # MuJoCo PD data collection. tau_max is accepted for bookkeeping but clipping is disabled.
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

    # DKN.
    p.add_argument("--lift_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    p.add_argument("--control_hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--control_dim_hat", type=int, default=CONTROL_DIM)
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--bound_lift", type=int, default=1)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--window_start", type=int, default=4)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--steps_per_epoch", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_embed", type=float, default=0.1)

    # EDMD.
    p.add_argument("--edmd_centers", type=int, default=200)
    p.add_argument("--edmd_sigma", type=float, default=None)
    p.add_argument("--edmd_ridge", type=float, default=1e-4)
    p.add_argument("--edmd_seed", type=int, default=2007)

    p.add_argument("--demo_traj", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== CDSM prediction compare: Shi-Meng DKN vs Koopman-EDMD ===")
    print(f"device={device}, pred_mode={args.pred_mode}, output={out_dir}")
    print("[policy] torque clipping disabled; cable max-tension clipping disabled")

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

    print("[2/5] Fitting shared normalizers...")
    x_all = train_raw["states"][:, :-1, :].reshape(-1, STATE_DIM)
    u_all = train_raw["inputs"].reshape(-1, CONTROL_DIM)
    x_normer = Normalizer.fit(x_all)
    u_normer = Normalizer.fit(u_all)

    print("[3/5] Training DKN...")
    Xw_tr, Uw_tr = build_windows(train_raw["states"], train_raw["inputs"], args.window)
    Xw_va, Uw_va = build_windows(val_raw["states"], val_raw["inputs"], args.window)
    Xw_tr_n = (Xw_tr - x_normer.mean) / x_normer.std
    Xw_va_n = (Xw_va - x_normer.mean) / x_normer.std
    Uw_tr_n = (Uw_tr - u_normer.mean) / u_normer.std
    Uw_va_n = (Uw_va - u_normer.mean) / u_normer.std

    dkn_cfg = DKNConfig(
        lift_dim=args.lift_dim,
        hidden=tuple(args.hidden),
        control_hidden=tuple(args.control_hidden),
        control_dim_hat=args.control_dim_hat,
        activation=args.activation,
        bound_lift=bool(args.bound_lift),
        window=args.window,
        window_start=args.window_start,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        w_state=args.w_state,
        w_embed=args.w_embed,
    )
    dkn_model, history, dkn_stats = train_dkn(
        (Xw_tr_n, Uw_tr_n), (Xw_va_n, Uw_va_n), dkn_cfg, device, out_dir
    )
    np.savetxt(
        out_dir / "dkn_training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_state,train_embed,val_total,val_state,val_embed",
        comments="",
    )
    dkn_pred = DKNPredictor(dkn_model, x_normer, u_normer, device)

    print("[4/5] Fitting Koopman-EDMD...")
    edmd_cfg = EdmdConfig(
        n_centers=args.edmd_centers,
        rbf_sigma=args.edmd_sigma,
        ridge=args.edmd_ridge,
        kmeans_seed=args.edmd_seed,
    )
    ed_pred = fit_full_edmd(train_raw["states"], train_raw["inputs"], x_normer, u_normer, edmd_cfg)
    print(
        f"      EDMD latent_dim={ed_pred.latent_dim}, sigma={ed_pred.sigma:.4g}, "
        f"cond(Gram)={ed_pred.cond_number:.3e}"
    )

    print("[5/5] Predicting and comparing...")
    modes = ["one_step", "rollout"] if args.pred_mode == "both" else [args.pred_mode]
    demo_idx = min(max(args.demo_traj, 0), max(args.val_traj - 1, 0))
    results: Dict[str, object] = {}
    for mode in modes:
        results[mode] = run_one_mode(mode, dkn_pred, ed_pred, val_raw, args.dt, demo_idx)

    summary = {
        "xml": args.xml,
        "dt": args.dt,
        "pred_mode": args.pred_mode,
        "dkn_config": asdict(dkn_cfg),
        "dkn_latent_dim": dkn_model.latent_dim,
        "dkn_control_dim_hat": dkn_model.control_dim_hat,
        "dkn_train_stats": dkn_stats,
        "edmd_config": asdict(edmd_cfg),
        "edmd_latent_dim": ed_pred.latent_dim,
        "edmd_sigma": ed_pred.sigma,
        "edmd_cond_gram": ed_pred.cond_number,
        "normalization": {"x": x_normer.to_json(), "u": u_normer.to_json()},
        "results": results,
        "collection": {
            "train": {**asdict(pd_train), "meta": train_meta},
            "val": {**asdict(pd_val), "meta": val_meta},
        },
        "limits": {
            "torque_limit_enabled": False,
            "cable_tension_limit_enabled": False,
            "f_max_cable": F_MAX_CABLE,
            "f_preload": F_PRELOAD,
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
