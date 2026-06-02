"""
cdsm_nmpc_tracking_deepkoopman.py
=================================
Cable-driven space manipulator (CDSM): joint-space trajectory tracking with
Nonlinear Model Predictive Control (NMPC), comparing two internal prediction
models on the **same MuJoCo "real" plant**:

    1. Nominal model      : x_{k+1} = f_nom(x_k, u_k)           (rigid 2-DOF)
    2. Hybrid model       : x_{k+1} = f_nom(x_k, u_k) + r_hat(x_k, u_k)
                            where r_hat is a controlled DeepKoopman residual.

Pipeline
--------
1. Train (or load) the controlled DeepKoopman residual model, reusing the data
   collection / training utilities of ``cdsm_hybrid_residual_deepkoopman.py``.
2. Re-implement the rigid nominal dynamics in PyTorch (differentiable, batched)
   so the whole prediction rollout is autograd-friendly. The torch nominal is
   numerically verified against the NumPy reference model.
3. Build a single-shooting NMPC: at every control instant it optimizes a future
   joint-torque sequence (squashed by ``tau_max * tanh``) with L-BFGS to minimize
   joint tracking error + control effort over a finite horizon.
4. Close the loop on MuJoCo: the optimal joint torque ``u0`` is mapped to 8
   antagonistic cable tensions and applied to the cable-driven plant.
5. Run the closed loop twice (nominal-NMPC and hybrid-NMPC) and compare.

Output figures (saved by ``utils_plot.save_figure``)
----------------------------------------------------
- joint angle trajectory tracking (q_a, q_b: reference vs nominal vs hybrid)
- tracking error of the two joints
- tracking-error RMSE (per joint + overall, bar chart) and running RMSE
- joint torque applied to MuJoCo (tau_a, tau_b)
- MuJoCo rope/cable tensions (8 cables, nominal vs hybrid)

Examples
--------
    python cdsm_nmpc_tracking_deepkoopman.py
    python cdsm_nmpc_tracking_deepkoopman.py --ckpt best_residual_deepkoopman.pt
    python cdsm_nmpc_tracking_deepkoopman.py --sim_steps 600 --horizon 20 --device cuda
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
    import mujoco
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires PyTorch and MuJoCo.") from exc

from cdsm_hybrid_residual_deepkoopman import (
    Normalizer,
    ResidualDeepKoopman,
    make_device,
    predict_hybrid_next,
    set_seed,
)
from cdsm_hybrid_residual_edmd import (
    ACTUATOR_NAMES,
    CABLE_NAMES,
    F_MAX_CABLE,
    F_PRELOAD,
    PDCollectConfig,
    build_residual_dataset,
    cable_antagonistic_map,
    collect_pd_trajectories,
    compute_nominal_next,
    compute_tendon_jacobian_fd,
    flatten_residual_data,
    get_active_state,
    load_cable_model,
    set_active_state,
)
from cdsm_rigid_nominal_model import (
    XML_MASS_M2_M4,
    XML_MASS_M3_M5,
    CdsmNominalParams,
    CdsmRigidNominalModel,
)
from utils_plot import get_save_dir, save_figure

XML_PATH = "multi_joint_cable_dirven_space_robot.xml"
STATE_LABELS = ["q_a", "q_b", "dq_a", "dq_b"]
JOINT_LABELS = ["q_a (1st stage)", "q_b (2nd stage)"]


def make_degraded_nominal(dt: float, mass_scale: float) -> CdsmRigidNominalModel:
    """
    Build a nominal model whose link masses are ``mass_scale`` times the true
    MuJoCo values. This represents *imperfect prior knowledge* of the plant:
    with ``mass_scale < 1`` the controller under-estimates inertia and, on its
    own, mis-actuates. The DeepKoopman residual is trained against this same
    degraded nominal, so the hybrid model recovers the true dynamics.
    """
    params = CdsmNominalParams(
        dt_default=dt,
        m2=XML_MASS_M2_M4 * mass_scale,
        m3=XML_MASS_M3_M5 * mass_scale,
        m4=XML_MASS_M2_M4 * mass_scale,
        m5=XML_MASS_M3_M5 * mass_scale,
    )
    return CdsmRigidNominalModel(params)


# ===================================================================
# Differentiable torch nominal model (mirror of CdsmRigidNominalModel)
# ===================================================================
class TorchNominalModel(nn.Module):
    """
    PyTorch re-implementation of the rigid 2-DOF nominal dynamics.

    Mirrors :class:`cdsm_rigid_nominal_model.CdsmRigidNominalModel`:
        M(q) ddq + C(q,dq) dq = tau
    integrated with semi-implicit Euler. The Coriolis force vector is computed
    from central finite differences of M(q) (same eps as the NumPy reference),
    so the values match the model the residual was trained against. The hard
    joint-limit clamping of the reference is intentionally **not** applied here
    to keep the rollout smooth for gradient-based NMPC.
    """

    def __init__(self, ref: CdsmRigidNominalModel, dt: float, coriolis_eps: float = 1e-5) -> None:
        super().__init__()
        p = ref.p
        self.dt = float(dt)
        self.eps = float(coriolis_eps)
        # geometry / inertia constants (registered as buffers for device moves)
        consts = {
            "L2": ref.L2, "L3": ref.L3, "L4": ref.L4, "L5": ref.L5,
            "r2": ref.r2, "r3": ref.r3, "r4": ref.r4, "r5": ref.r5,
            "I2": ref.I2, "I3": ref.I3, "I4": ref.I4, "I5": ref.I5,
            "m2": p.m2, "m3": p.m3, "m4": p.m4, "m5": p.m5,
        }
        for name, val in consts.items():
            self.register_buffer(name, torch.tensor(float(val), dtype=torch.float32))

    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Return symmetric 2x2 mass matrix, batched -> (B, 2, 2)."""
        qa = q[..., 0]
        qb = q[..., 1]
        t2 = qa
        t3 = 2.0 * qa
        t4 = 2.0 * qa + qb
        t5 = 2.0 * qa + 2.0 * qb
        s2, c2 = torch.sin(t2), torch.cos(t2)
        s3, c3 = torch.sin(t3), torch.cos(t3)
        s4, c4 = torch.sin(t4), torch.cos(t4)
        s5, c5 = torch.sin(t5), torch.cos(t5)
        zero = torch.zeros_like(qa)

        # per-link velocity-jacobian columns (c0 = d/dqa, c1 = d/dqb) and
        # angular jacobian (w_qa, w_qb)
        # link2
        l2_c0 = (-self.r2 * s2, self.r2 * c2)
        l2_c1 = (zero, zero)
        l2_w = (1.0, 0.0)
        # link3
        l3_c0 = (-self.L2 * s2 - 2.0 * self.r3 * s3, self.L2 * c2 + 2.0 * self.r3 * c3)
        l3_c1 = (zero, zero)
        l3_w = (2.0, 0.0)
        # link4
        l4_c0 = (
            -self.L2 * s2 - 2.0 * self.L3 * s3 - 2.0 * self.r4 * s4,
            self.L2 * c2 + 2.0 * self.L3 * c3 + 2.0 * self.r4 * c4,
        )
        l4_c1 = (-self.r4 * s4, self.r4 * c4)
        l4_w = (2.0, 1.0)
        # link5
        l5_c0 = (
            -self.L2 * s2 - 2.0 * self.L3 * s3 - 2.0 * self.L4 * s4 - 2.0 * self.r5 * s5,
            self.L2 * c2 + 2.0 * self.L3 * c3 + 2.0 * self.L4 * c4 + 2.0 * self.r5 * c5,
        )
        l5_c1 = (-self.L4 * s4 - 2.0 * self.r5 * s5, self.L4 * c4 + 2.0 * self.r5 * c5)
        l5_w = (2.0, 2.0)

        links = [
            (self.m2, self.I2, l2_c0, l2_c1, l2_w),
            (self.m3, self.I3, l3_c0, l3_c1, l3_w),
            (self.m4, self.I4, l4_c0, l4_c1, l4_w),
            (self.m5, self.I5, l5_c0, l5_c1, l5_w),
        ]

        m00 = torch.zeros_like(qa)
        m01 = torch.zeros_like(qa)
        m11 = torch.zeros_like(qa)
        for m, I, c0, c1, w in links:
            c0x, c0y = c0
            c1x, c1y = c1
            wqa, wqb = w
            m00 = m00 + m * (c0x * c0x + c0y * c0y) + I * (wqa * wqa)
            m01 = m01 + m * (c0x * c1x + c0y * c1y) + I * (wqa * wqb)
            m11 = m11 + m * (c1x * c1x + c1y * c1y) + I * (wqb * wqb)

        row0 = torch.stack([m00, m01], dim=-1)
        row1 = torch.stack([m01, m11], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def coriolis_force(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Coriolis/centrifugal generalized force h = C(q,dq) dq -> (B, 2)."""
        eps = self.eps
        # central differences for dM/dq_0 and dM/dq_1, batched in a single
        # mass_matrix evaluation (4 perturbations stacked on a new leading axis)
        e0 = torch.zeros_like(q)
        e1 = torch.zeros_like(q)
        e0[..., 0] = eps
        e1[..., 1] = eps
        q_pert = torch.stack([q + e0, q - e0, q + e1, q - e1], dim=0)  # (4, ..., 2)
        M_pert = self.mass_matrix(q_pert)  # (4, ..., 2, 2)
        dM = [
            (M_pert[0] - M_pert[1]) / (2.0 * eps),
            (M_pert[2] - M_pert[3]) / (2.0 * eps),
        ]

        h_components = []
        for k in range(2):
            hk = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)
            for i in range(2):
                for j in range(2):
                    c_kij = 0.5 * (
                        dM[i][..., k, j] + dM[j][..., k, i] - dM[k][..., i, j]
                    )
                    hk = hk + c_kij * dq[..., i] * dq[..., j]
            h_components.append(hk)
        return torch.stack(h_components, dim=-1)

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Advance one step. x: (..., 4), u: (..., 2) -> (..., 4)."""
        q = x[..., :2]
        dq = x[..., 2:]
        M = self.mass_matrix(q)
        h = self.coriolis_force(q, dq)
        rhs = (u - h).unsqueeze(-1)
        ddq = torch.linalg.solve(M, rhs).squeeze(-1)
        dq_next = dq + ddq * self.dt
        q_next = q + dq_next * self.dt
        return torch.cat([q_next, dq_next], dim=-1)


class HybridPredictor(nn.Module):
    """Differentiable one-step predictor: nominal (+ optional DK residual)."""

    def __init__(
        self,
        torch_nominal: TorchNominalModel,
        dk_model: ResidualDeepKoopman,
        x_normer: Normalizer,
        u_normer: Normalizer,
        r_normer: Normalizer,
    ) -> None:
        super().__init__()
        self.nominal = torch_nominal
        self.dk = dk_model
        self.register_buffer("x_mean", torch.tensor(x_normer.mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(x_normer.std, dtype=torch.float32))
        self.register_buffer("u_mean", torch.tensor(u_normer.mean, dtype=torch.float32))
        self.register_buffer("u_std", torch.tensor(u_normer.std, dtype=torch.float32))
        self.register_buffer("r_mean", torch.tensor(r_normer.mean, dtype=torch.float32))
        self.register_buffer("r_std", torch.tensor(r_normer.std, dtype=torch.float32))

    def residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_n = (x - self.x_mean) / self.x_std
        u_n = (u - self.u_mean) / self.u_std
        z = self.dk.encode(x_n)
        z_next = self.dk.koopman_step(z, u_n)
        r_n = self.dk.residual_decoder(z_next)
        return r_n * self.r_std + self.r_mean

    def step(self, x: torch.Tensor, u: torch.Tensor, use_residual: bool) -> torch.Tensor:
        x_nom = self.nominal.step(x, u)
        if not use_residual:
            return x_nom
        return x_nom + self.residual(x, u)


# ===================================================================
# NMPC (single shooting, torch L-BFGS, tanh-squashed torque)
# ===================================================================
@dataclass
class NMPCConfig:
    horizon: int          # number of decision steps (control intervals)
    dt: float             # plant integration step
    n_sub: int            # plant sub-steps held per control interval (mpc_dt = n_sub*dt)
    tau_max: float
    w_q: Tuple[float, float]
    w_dq: Tuple[float, float]
    w_u: float
    w_du: float
    w_terminal: float
    lbfgs_iters: int
    lr: float


class NMPCController:
    def __init__(
        self,
        predictor: HybridPredictor,
        cfg: NMPCConfig,
        use_residual: bool,
        device: torch.device,
    ) -> None:
        self.predictor = predictor
        self.cfg = cfg
        self.use_residual = use_residual
        self.device = device
        N = cfg.horizon
        # decision variables (pre-squash); persistent for warm starting
        self._v = torch.zeros((N, 2), dtype=torch.float32, device=device, requires_grad=True)
        # previously *applied* control (for correct rate-penalty reference)
        self._u_prev = torch.zeros(2, dtype=torch.float32, device=device)
        self.w_q = torch.tensor(cfg.w_q, dtype=torch.float32, device=device)
        self.w_dq = torch.tensor(cfg.w_dq, dtype=torch.float32, device=device)

    def reset(self) -> None:
        with torch.no_grad():
            self._v.zero_()
            self._u_prev.zero_()

    def _squash(self, v: torch.Tensor) -> torch.Tensor:
        return self.cfg.tau_max * torch.tanh(v)

    def _rollout_cost(
        self, x0: torch.Tensor, q_ref: torch.Tensor, dq_ref: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        x = x0
        # rate penalty is referenced to the control actually applied last cycle
        u_prev = self._u_prev
        cost = torch.zeros((), dtype=torch.float32, device=self.device)
        u_seq = self._squash(self._v)
        for k in range(cfg.horizon):
            uk = u_seq[k]
            # hold uk over n_sub plant sub-steps (coarse control interval)
            xb = x.unsqueeze(0)
            ub = uk.unsqueeze(0)
            for _ in range(cfg.n_sub):
                xb = self.predictor.step(xb, ub, self.use_residual)
            x = xb.squeeze(0)
            q_err = x[:2] - q_ref[k]
            dq_err = x[2:] - dq_ref[k]
            stage = torch.sum(self.w_q * q_err * q_err) + torch.sum(self.w_dq * dq_err * dq_err)
            if k == cfg.horizon - 1:
                stage = stage * cfg.w_terminal
            cost = cost + stage
            cost = cost + cfg.w_u * torch.sum(uk * uk)
            cost = cost + cfg.w_du * torch.sum((uk - u_prev) * (uk - u_prev))
            u_prev = uk
        return cost

    def solve(
        self, x0_np: np.ndarray, q_ref_np: np.ndarray, dq_ref_np: np.ndarray
    ) -> np.ndarray:
        """Return the optimal joint torque sequence (horizon, 2) for current step."""
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        q_ref = torch.tensor(q_ref_np, dtype=torch.float32, device=self.device)
        dq_ref = torch.tensor(dq_ref_np, dtype=torch.float32, device=self.device)

        # warm start: shift previous solution forward by one step
        with torch.no_grad():
            self._v[:-1] = self._v[1:].clone()
            self._v[-1] = self._v[-2].clone() if self.cfg.horizon > 1 else self._v[-1]

        opt = torch.optim.LBFGS(
            [self._v],
            lr=self.cfg.lr,
            max_iter=self.cfg.lbfgs_iters,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = self._rollout_cost(x0, q_ref, dq_ref)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            u_full = self._squash(self._v).detach()
            # remember the control we are about to apply for the next rate penalty
            self._u_prev = u_full[0].clone()
            u_seq = u_full.cpu().numpy().astype(np.float64)
        return u_seq


# ===================================================================
# Reference trajectory
# ===================================================================
@dataclass
class RefConfig:
    amp_a: float
    amp_b: float
    omega_a: float
    omega_b: float
    phase_b: float


def generate_reference(
    n_points: int, dt: float, cfg: RefConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth sinusoidal joint reference. Returns q_ref, dq_ref of shape (n,2)."""
    t = np.arange(n_points) * dt
    qa = cfg.amp_a * np.sin(cfg.omega_a * t)
    qb = cfg.amp_b * np.sin(cfg.omega_b * t + cfg.phase_b)
    dqa = cfg.amp_a * cfg.omega_a * np.cos(cfg.omega_a * t)
    dqb = cfg.amp_b * cfg.omega_b * np.cos(cfg.omega_b * t + cfg.phase_b)
    q_ref = np.stack([qa, qb], axis=1)
    dq_ref = np.stack([dqa, dqb], axis=1)
    return q_ref, dq_ref


# ===================================================================
# MuJoCo closed-loop runner
# ===================================================================
@dataclass
class ClosedLoopResult:
    q_actual: np.ndarray      # (T+1, 2)
    dq_actual: np.ndarray     # (T+1, 2)
    tau_applied: np.ndarray   # (T, 2) commanded joint torque
    cable_tension: np.ndarray  # (T, 8) realized MuJoCo cable tension
    solve_time: float


def _tension_sensor_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for n in CABLE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tension_" + n[len("cable"):])
        ids.append(int(model.sensor_adr[sid]) if sid >= 0 else -1)
    return np.array(ids, dtype=int)


def run_closed_loop(
    controller: NMPCController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    scratch: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    q_ref_full: np.ndarray,
    dq_ref_full: np.ndarray,
    sim_steps: int,
    horizon: int,
    x0: np.ndarray,
    label: str = "",
) -> ClosedLoopResult:
    controller.reset()
    actuator_ids = indices["actuator_ids"]
    tension_adr = _tension_sensor_ids(mj_model)
    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])

    set_active_state(mj_model, mj_data, indices, x0[:2].copy(), x0[2:].copy())

    q_hist = np.zeros((sim_steps + 1, 2), dtype=np.float64)
    dq_hist = np.zeros((sim_steps + 1, 2), dtype=np.float64)
    tau_hist = np.zeros((sim_steps, 2), dtype=np.float64)
    tension_hist = np.zeros((sim_steps, 8), dtype=np.float64)

    x_now = get_active_state(mj_data, indices)
    q_hist[0] = x_now[:2]
    dq_hist[0] = x_now[2:]

    n_sub = max(int(controller.cfg.n_sub), 1)
    last_idx = q_ref_full.shape[0] - 1
    u0 = np.zeros(2, dtype=np.float64)

    t_solve = 0.0
    for k in range(sim_steps):
        # re-solve at the coarse control rate; hold the control in between
        if k % n_sub == 0:
            x_now = get_active_state(mj_data, indices)
            # reference sampled on the coarse grid the predicted states land on:
            # decision j (1..horizon) -> plant time index k + j*n_sub
            idxs = [min(k + j * n_sub, last_idx) for j in range(1, horizon + 1)]
            q_ref_win = q_ref_full[idxs]
            dq_ref_win = dq_ref_full[idxs]
            t0 = time.time()
            u_seq = controller.solve(x_now, q_ref_win, dq_ref_win)
            t_solve += time.time() - t0
            u0 = u_seq[0]

        J = compute_tendon_jacobian_fd(mj_model, scratch, mj_data.qpos.copy(), indices["tendon_ids"])
        F_cable = cable_antagonistic_map(
            float(u0[0]), float(u0[1]), J, dof_j1, dof_j2, dof_j3, dof_j4
        )
        mj_data.ctrl[actuator_ids] = F_cable
        mujoco.mj_step(mj_model, mj_data)

        x_next = get_active_state(mj_data, indices)
        q_hist[k + 1] = x_next[:2]
        dq_hist[k + 1] = x_next[2:]
        tau_hist[k] = u0
        # realized cable tension from MuJoCo actuator force (gear=1 -> N)
        tension_hist[k] = mj_data.actuator_force[actuator_ids]

        if (k + 1) % max(sim_steps // 10, 1) == 0:
            print(f"      [{label}] step {k + 1}/{sim_steps}")

    return ClosedLoopResult(
        q_actual=q_hist,
        dq_actual=dq_hist,
        tau_applied=tau_hist,
        cable_tension=tension_hist,
        solve_time=t_solve,
    )


# ===================================================================
# Metrics
# ===================================================================
def tracking_metrics(q_actual: np.ndarray, q_ref: np.ndarray) -> Dict[str, object]:
    err = q_actual - q_ref
    rmse_per_joint = np.sqrt(np.mean(err * err, axis=0))
    overall_rmse = float(np.sqrt(np.mean(err * err)))
    # running RMSE up to each time
    cum_sq = np.cumsum(np.sum(err * err, axis=1))
    counts = np.arange(1, err.shape[0] + 1) * err.shape[1]
    running_rmse = np.sqrt(cum_sq / counts)
    return {
        "error": err,
        "rmse_per_joint": rmse_per_joint,
        "overall_rmse": overall_rmse,
        "running_rmse": running_rmse,
    }


# ===================================================================
# Training / loading the DeepKoopman residual model
# ===================================================================
def sample_rollout_batch(
    x_seq_norm: np.ndarray,
    u_seq_norm: np.ndarray,
    xp_seq_norm: np.ndarray,
    x_nom_next_seq_norm: np.ndarray,
    r_seq_norm: np.ndarray,
    batch_size: int,
    rollout_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample batched short trajectories for multi-step autoregressive training.

    Returns:
        x0:      (B, 4)
        u_seq:   (B, R, 2)
        xp_true: (B, R, 4)
        x_nom:   (B, R, 4)
        r_true:  (B, R, 4)
    """
    n_traj, T, _ = u_seq_norm.shape
    R = int(max(1, min(rollout_steps, T)))
    traj_idx = np.random.randint(0, n_traj, size=batch_size)
    start_idx = np.random.randint(0, T - R + 1, size=batch_size)

    x0 = np.empty((batch_size, 4), dtype=np.float32)
    u_seq = np.empty((batch_size, R, 2), dtype=np.float32)
    xp_true = np.empty((batch_size, R, 4), dtype=np.float32)
    x_nom = np.empty((batch_size, R, 4), dtype=np.float32)
    r_true = np.empty((batch_size, R, 4), dtype=np.float32)
    for b in range(batch_size):
        i = traj_idx[b]
        k = start_idx[b]
        x0[b] = x_seq_norm[i, k]
        u_seq[b] = u_seq_norm[i, k: k + R]
        xp_true[b] = xp_seq_norm[i, k: k + R]
        x_nom[b] = x_nom_next_seq_norm[i, k: k + R]
        r_true[b] = r_seq_norm[i, k: k + R]

    return (
        torch.from_numpy(x0).to(device),
        torch.from_numpy(u_seq).to(device),
        torch.from_numpy(xp_true).to(device),
        torch.from_numpy(x_nom).to(device),
        torch.from_numpy(r_true).to(device),
    )


def compute_multistep_losses(
    model: ResidualDeepKoopman,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    xp_seq: torch.Tensor,
    x_nom_next_seq: torch.Tensor,
    r_seq: torch.Tensor,
    w_residual: float,
    w_recon: float,
    w_linear: float,
    w_l2: float,
) -> Dict[str, torch.Tensor]:
    """
    Multi-step autoregressive loss over rollout steps.
    """
    B, R, _ = u_seq.shape
    xk = x0
    residual_loss = torch.zeros((), device=x0.device)
    recon_loss = torch.zeros((), device=x0.device)
    linear_loss = torch.zeros((), device=x0.device)

    for t in range(R):
        out = model(xk, u_seq[:, t, :])
        z_true_next = model.encode(xp_seq[:, t, :])
        r_pred = out["r_pred"]
        x_pred_next = x_nom_next_seq[:, t, :] + r_pred
        residual_loss = residual_loss + torch.mean((r_pred - r_seq[:, t, :]) ** 2)
        recon_loss = recon_loss + torch.mean((out["x_rec"] - xk) ** 2)
        linear_loss = linear_loss + torch.mean((out["z_next"] - z_true_next) ** 2)
        xk = x_pred_next

    inv_R = 1.0 / float(max(R, 1))
    residual_loss = residual_loss * inv_R
    recon_loss = recon_loss * inv_R
    linear_loss = linear_loss * inv_R

    l2 = torch.zeros((), device=x0.device)
    if w_l2 > 0:
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                l2 = l2 + torch.sum(p * p)
        l2 = w_l2 * l2

    total = w_residual * residual_loss + w_recon * recon_loss + w_linear * linear_loss + l2
    return {
        "total": total,
        "residual": residual_loss,
        "recon": recon_loss,
        "linear": linear_loss,
        "l2": l2.detach(),
    }


def train_dk_model(args: argparse.Namespace, device: torch.device, out_dir: Path) -> Dict[str, object]:
    """Collect data, train residual DeepKoopman, return model + normalizers."""
    print("[data] collecting MuJoCo cable-driven PD trajectories for training...")
    pd_cfg = PDCollectConfig(
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
        tau_max=float("inf"),
    )
    pd_cfg_val = PDCollectConfig(
        traj_count=max(args.train_traj // 5, 2),
        steps=args.steps,
        dt=args.dt,
        seed=args.seed + 1000,
        q_init_range=args.q_init_range,
        dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max),
        omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b),
        kd=(args.kd_a, args.kd_b),
        tau_max=float("inf"),
    )
    mj_model, mj_data, scratch, indices = load_cable_model(args.xml, args.dt)
    train_raw, _ = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_cfg)
    val_raw, _ = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_cfg_val)

    nominal = make_degraded_nominal(dt=args.dt, mass_scale=args.nominal_mass_scale)
    res_train = build_residual_dataset(train_raw, nominal, args.dt)
    res_val = build_residual_dataset(val_raw, nominal, args.dt)
    x_tr, u_tr, _, _ = flatten_residual_data(res_train)

    x_normer = Normalizer.fit(x_tr)
    u_normer = Normalizer.fit(u_tr)
    r_normer = Normalizer.fit(res_train["residuals"].reshape(-1, 4))
    train_arrays = {
        "x": x_normer.transform(res_train["states"]),
        "u": u_normer.transform(res_train["inputs"]),
        "xp": x_normer.transform(res_train["states"][:, 1:, :]),
        "x_nom_next": x_normer.transform(res_train["x_nom_next"]),
        "r": r_normer.transform(res_train["residuals"]),
    }
    val_arrays = {
        "x": x_normer.transform(res_val["states"]),
        "u": u_normer.transform(res_val["inputs"]),
        "xp": x_normer.transform(res_val["states"][:, 1:, :]),
        "x_nom_next": x_normer.transform(res_val["x_nom_next"]),
        "r": r_normer.transform(res_val["residuals"]),
    }

    n_transitions = train_arrays["u"].shape[0] * train_arrays["u"].shape[1]
    print(
        f"[train] residual DeepKoopman (multi-step): transitions={n_transitions}, "
        f"rollout_steps={args.rollout_steps}, epochs={args.epochs}"
    )
    dk_model = ResidualDeepKoopman(
        state_dim=4,
        control_dim=2,
        latent_dim=args.latent_dim,
        hidden=tuple(args.hidden),
        activation=args.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(dk_model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        dk_model.train()
        for _ in range(args.steps_per_epoch):
            batch = sample_rollout_batch(
                train_arrays["x"],
                train_arrays["u"],
                train_arrays["xp"],
                train_arrays["x_nom_next"],
                train_arrays["r"],
                args.batch_size,
                args.rollout_steps,
                device,
            )
            losses = compute_multistep_losses(
                dk_model, *batch,
                w_residual=args.w_residual, w_recon=args.w_recon,
                w_linear=args.w_linear, w_l2=args.w_l2,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(dk_model.parameters(), 5.0)
            optimizer.step()
        # robust validation: average loss over several batches to reduce noise
        dk_model.eval()
        vloss = 0.0
        n_vb = 20
        with torch.no_grad():
            for _ in range(n_vb):
                vb = sample_rollout_batch(
                    val_arrays["x"],
                    val_arrays["u"],
                    val_arrays["xp"],
                    val_arrays["x_nom_next"],
                    val_arrays["r"],
                    min(args.batch_size, val_arrays["x"].shape[0]),
                    args.rollout_steps,
                    device,
                )
                vloss += compute_multistep_losses(
                    dk_model, *vb,
                    w_residual=args.w_residual, w_recon=args.w_recon,
                    w_linear=args.w_linear, w_l2=args.w_l2,
                )["total"].item()
        vloss /= n_vb
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in dk_model.state_dict().items()}
        if epoch % max(args.epochs // 10, 1) == 0 or epoch == 1:
            print(f"  [ep {epoch:03d}] train_total={losses['total'].item():.4e}  val_total={vloss:.4e}")

    if best_state is not None:
        dk_model.load_state_dict(best_state)
    dk_model.eval()

    # one-step open-loop prediction sanity check: hybrid should beat nominal
    os_nom, os_hyb = _one_step_pred_rmse(
        dk_model, nominal, res_val, args.dt, x_normer, u_normer, r_normer, device
    )
    print("[check] one-step prediction RMSE on validation set (vs MuJoCo):")
    for i, lab in enumerate(STATE_LABELS):
        print(f"    {lab:5s}  nominal={os_nom[i]:.5g}  hybrid={os_hyb[i]:.5g}")
    nom_tot = float(np.sqrt(np.mean(os_nom ** 2)))
    hyb_tot = float(np.sqrt(np.mean(os_hyb ** 2)))
    print(f"    overall  nominal={nom_tot:.5g}  hybrid={hyb_tot:.5g}  "
          f"improvement={100.0 * (nom_tot - hyb_tot) / max(nom_tot, 1e-12):.1f}%")

    ckpt = {
        "model_state": dk_model.state_dict(),
        "config": {
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "activation": args.activation,
            "rollout_steps": args.rollout_steps,
        },
        "x_normer": x_normer.to_json(),
        "u_normer": u_normer.to_json(),
        "r_normer": r_normer.to_json(),
        "best_val": best_val,
        "nominal_mass_scale": args.nominal_mass_scale,
    }
    torch.save(ckpt, out_dir / "nmpc_residual_deepkoopman.pt")
    print(f"[train] saved checkpoint -> {out_dir / 'nmpc_residual_deepkoopman.pt'} (best_val={best_val:.4e})")
    return {
        "model": dk_model, "x_normer": x_normer, "u_normer": u_normer,
        "r_normer": r_normer, "mass_scale": args.nominal_mass_scale,
    }


def _one_step_pred_rmse(
    dk_model: ResidualDeepKoopman,
    nominal: CdsmRigidNominalModel,
    res_val: Dict[str, np.ndarray],
    dt: float,
    x_normer: Normalizer,
    u_normer: Normalizer,
    r_normer: Normalizer,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-state one-step prediction RMSE for nominal and hybrid models."""
    states = res_val["states"]
    inputs = res_val["inputs"]
    n_traj, n_step, _ = inputs.shape
    se_nom = np.zeros(4)
    se_hyb = np.zeros(4)
    count = 0
    for i in range(n_traj):
        for k in range(n_step):
            xk = states[i, k]
            uk = inputs[i, k]
            xtrue = states[i, k + 1]
            pn = compute_nominal_next(nominal, xk, uk, dt)
            ph = predict_hybrid_next(dk_model, nominal, xk, uk, dt, x_normer, u_normer, r_normer, device)
            se_nom += (pn - xtrue) ** 2
            se_hyb += (ph - xtrue) ** 2
            count += 1
    return np.sqrt(se_nom / count), np.sqrt(se_hyb / count)


def load_dk_model(ckpt_path: Path, device: torch.device) -> Dict[str, object]:
    print(f"[load] DeepKoopman residual checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dk_model = ResidualDeepKoopman(
        state_dim=4,
        control_dim=2,
        latent_dim=int(cfg["latent_dim"]),
        hidden=tuple(cfg["hidden"]),
        activation=cfg["activation"],
    ).to(device)
    dk_model.load_state_dict(ckpt["model_state"])
    dk_model.eval()

    def _normer(d: Dict[str, List[float]]) -> Normalizer:
        return Normalizer(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )

    return {
        "model": dk_model,
        "x_normer": _normer(ckpt["x_normer"]),
        "u_normer": _normer(ckpt["u_normer"]),
        "r_normer": _normer(ckpt["r_normer"]),
        "mass_scale": float(ckpt.get("nominal_mass_scale", 0.95)),
    }


def verify_torch_nominal(ref: CdsmRigidNominalModel, torch_nom: TorchNominalModel, dt: float) -> float:
    """Sanity check: torch nominal step must match the NumPy reference."""
    rng = np.random.RandomState(0)
    max_err = 0.0
    for _ in range(50):
        x = rng.uniform(-0.5, 0.5, size=4)
        u = rng.uniform(-30.0, 30.0, size=2)
        x_np = ref.step(x, u, dt=dt, apply_joint_limits=False)
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        ut = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        x_t = torch_nom.step(xt, ut).squeeze(0).numpy()
        max_err = max(max_err, float(np.max(np.abs(x_np - x_t))))
    return max_err


# ===================================================================
# Plotting
# ===================================================================
def plot_tracking(
    t: np.ndarray,
    q_ref: np.ndarray,
    res_nom: ClosedLoopResult,
    res_hyb: ClosedLoopResult,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i in range(2):
        ax = axes[i]
        ax.plot(t, q_ref[:, i], "k-", lw=2.0, label="Reference")
        ax.plot(t, res_nom.q_actual[:, i], "--", color="C0", lw=1.5, label="NMPC + Nominal")
        ax.plot(t, res_hyb.q_actual[:, i], "-.", color="C3", lw=1.5, label="NMPC + Hybrid (DeepKoopman)")
        ax.set_ylabel(f"{JOINT_LABELS[i]} [rad]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes[1].set_xlabel("Time [s]")
    fig.suptitle("Joint-space trajectory tracking on MuJoCo plant")
    fig.tight_layout()
    save_figure("tracking_joint_angles")
    plt.close(fig)


def plot_tracking_error(
    t: np.ndarray,
    m_nom: Dict[str, object],
    m_hyb: Dict[str, object],
) -> None:
    err_nom = m_nom["error"]
    err_hyb = m_hyb["error"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i in range(2):
        ax = axes[i]
        ax.plot(t, err_nom[:, i], color="C0", lw=1.4,
                label=f"Nominal (RMSE={m_nom['rmse_per_joint'][i]:.4g})")
        ax.plot(t, err_hyb[:, i], color="C3", lw=1.4,
                label=f"Hybrid (RMSE={m_hyb['rmse_per_joint'][i]:.4g})")
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
        ax.set_ylabel(f"{JOINT_LABELS[i]} error [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[1].set_xlabel("Time [s]")
    fig.suptitle("Joint tracking error: NMPC + Nominal vs NMPC + Hybrid")
    fig.tight_layout()
    save_figure("tracking_error")
    plt.close(fig)


def plot_rmse(
    t: np.ndarray,
    m_nom: Dict[str, object],
    m_hyb: Dict[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # bar chart: per-joint + overall
    ax = axes[0]
    labels = ["q_a", "q_b", "overall"]
    nom_vals = list(m_nom["rmse_per_joint"]) + [m_nom["overall_rmse"]]
    hyb_vals = list(m_hyb["rmse_per_joint"]) + [m_hyb["overall_rmse"]]
    xpos = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(xpos - width / 2, nom_vals, width, label="NMPC + Nominal", color="C0")
    b2 = ax.bar(xpos + width / 2, hyb_vals, width, label="NMPC + Hybrid", color="C3")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tracking RMSE [rad]")
    ax.set_title("Tracking-error RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3g}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)

    # running RMSE over time
    ax = axes[1]
    ax.plot(t, m_nom["running_rmse"], color="C0", lw=1.6, label="NMPC + Nominal")
    ax.plot(t, m_hyb["running_rmse"], color="C3", lw=1.6, label="NMPC + Hybrid")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Running RMSE [rad]")
    ax.set_title("Cumulative tracking RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_figure("tracking_rmse")
    plt.close(fig)


def plot_joint_torque(
    t_ctrl: np.ndarray,
    res_nom: ClosedLoopResult,
    res_hyb: ClosedLoopResult,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i in range(2):
        ax = axes[i]
        ax.plot(t_ctrl, res_nom.tau_applied[:, i], color="C0", lw=1.3, label="NMPC + Nominal")
        ax.plot(t_ctrl, res_hyb.tau_applied[:, i], color="C3", lw=1.3, label="NMPC + Hybrid")
        ax.set_ylabel(f"tau_{'ab'[i]} [N·m]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes[1].set_xlabel("Time [s]")
    fig.suptitle("Commanded joint torque applied to MuJoCo")
    fig.tight_layout()
    save_figure("joint_torque")
    plt.close(fig)


def plot_cable_tension(
    t_ctrl: np.ndarray,
    res_nom: ClosedLoopResult,
    res_hyb: ClosedLoopResult,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, res, title in (
        (axes[0], res_nom, "NMPC + Nominal"),
        (axes[1], res_hyb, "NMPC + Hybrid (DeepKoopman)"),
    ):
        for j, name in enumerate(CABLE_NAMES):
            ax.plot(t_ctrl, res.cable_tension[:, j], lw=1.0, color=cmap(j % 10), label=name)
        ax.set_ylabel("Cable tension [N]")
        ax.set_title(f"MuJoCo rope tensions — {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=7)
    axes[1].set_xlabel("Time [s]")
    fig.tight_layout()
    save_figure("cable_tension")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CDSM NMPC joint-space tracking: nominal vs hybrid DeepKoopman.")
    p.add_argument("--xml", default=XML_PATH)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--ckpt", default="", help="Path to a DeepKoopman checkpoint to load (skip training).")
    p.add_argument(
        "--nominal_mass_scale", type=float, default=0.9,
        help="Link-mass scale of the controller's nominal model vs the true MuJoCo plant "
             "(<1 => imperfect prior the residual must correct). Start near 1.0 to verify the "
             "controller itself can track, then lower it to expose the hybrid model's advantage.",
    )

    # data collection / training
    p.add_argument("--train_traj", type=int, default=80)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--q_init_range", type=float, default=1.3)
    p.add_argument("--dq_init_range", type=float, default=1.2)
    p.add_argument("--amp_min", type=float, default=-1.5)
    p.add_argument("--amp_max", type=float, default=1.5)
    p.add_argument("--omega_min", type=float, default=-1.2)
    p.add_argument("--omega_max", type=float, default=1.2)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps_per_epoch", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_residual", type=float, default=1.0)
    p.add_argument("--w_recon", type=float, default=0.05)
    p.add_argument("--w_linear", type=float, default=0.2)
    p.add_argument("--w_l2", type=float, default=1e-8)
    p.add_argument(
        "--rollout_steps",
        type=int,
        default=10,
        help="Multi-step rollout length used in both training and validation loss.",
    )

    # NMPC
    p.add_argument("--sim_steps", type=int, default=400)
    p.add_argument("--horizon", type=int, default=25,
                   help="Number of NMPC decision steps (control intervals).")
    p.add_argument("--mpc_dt", type=float, default=0.04,
                   help="MPC control interval (s); control is held over round(mpc_dt/dt) plant steps. "
                        "Lookahead = horizon * mpc_dt.")
    p.add_argument("--tau_max", type=float, default=50000)
    p.add_argument("--w_q", type=float, nargs=2, default=[8000.0, 3000.0])
    p.add_argument("--w_dq", type=float, nargs=2, default=[80.0, 40.0])
    p.add_argument("--w_u", type=float, default=1e-5)
    p.add_argument("--w_du", type=float, default=1e-4)
    p.add_argument("--w_terminal", type=float, default=10.0)
    p.add_argument("--lbfgs_iters", type=int, default=20)
    p.add_argument("--mpc_lr", type=float, default=1.0)

    # reference
    p.add_argument("--ref_amp_a", type=float, default=0.6)
    p.add_argument("--ref_amp_b", type=float, default=0.5)
    p.add_argument("--ref_omega_a", type=float, default=0.8)
    p.add_argument("--ref_omega_b", type=float, default=1.1)
    p.add_argument("--ref_phase_b", type=float, default=np.pi / 4.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== CDSM NMPC joint-space tracking: Nominal vs Hybrid DeepKoopman ===")
    print(f"device={device}, output={out_dir}")

    # ----- 1. residual DeepKoopman model -----
    if args.ckpt and Path(args.ckpt).exists():
        bundle = load_dk_model(Path(args.ckpt), device)
    else:
        bundle = train_dk_model(args, device, out_dir)
    dk_model = bundle["model"]
    x_normer = bundle["x_normer"]
    u_normer = bundle["u_normer"]
    r_normer = bundle["r_normer"]
    mass_scale = float(bundle.get("mass_scale", args.nominal_mass_scale))

    # ----- 2. torch nominal + hybrid predictor (same degraded nominal) -----
    ref_nominal = make_degraded_nominal(dt=args.dt, mass_scale=mass_scale)
    print(f"[model] controller nominal mass scale = {mass_scale} (true plant = 1.0)")
    torch_nominal = TorchNominalModel(ref_nominal, dt=args.dt).to(device)
    nom_err = verify_torch_nominal(ref_nominal, torch_nominal.cpu(), args.dt)
    torch_nominal = torch_nominal.to(device)
    print(f"[check] torch nominal vs NumPy nominal max abs step error = {nom_err:.2e}")

    predictor = HybridPredictor(torch_nominal, dk_model, x_normer, u_normer, r_normer).to(device)

    # coarse MPC control interval: hold each control over n_sub plant steps
    n_sub = max(int(round(args.mpc_dt / args.dt)), 1)
    lookahead_s = args.horizon * n_sub * args.dt
    print(f"[mpc] horizon={args.horizon} decisions x mpc_dt={n_sub * args.dt:.3g}s "
          f"(n_sub={n_sub}) -> lookahead={lookahead_s:.3g}s")

    # ----- 3. reference trajectory -----
    n_ref = args.sim_steps + args.horizon * n_sub + 2
    ref_cfg = RefConfig(
        amp_a=args.ref_amp_a, amp_b=args.ref_amp_b,
        omega_a=args.ref_omega_a, omega_b=args.ref_omega_b, phase_b=args.ref_phase_b,
    )
    q_ref_full, dq_ref_full = generate_reference(n_ref, args.dt, ref_cfg)
    x0 = np.array([q_ref_full[0, 0], q_ref_full[0, 1], dq_ref_full[0, 0], dq_ref_full[0, 1]], dtype=np.float64)

    # ----- 4. closed-loop runs -----
    mj_model, mj_data, scratch, indices = load_cable_model(args.xml, args.dt)
    nmpc_cfg = NMPCConfig(
        horizon=args.horizon, dt=args.dt, n_sub=n_sub, tau_max=args.tau_max,
        w_q=tuple(args.w_q), w_dq=tuple(args.w_dq),
        w_u=args.w_u, w_du=args.w_du, w_terminal=args.w_terminal,
        lbfgs_iters=args.lbfgs_iters, lr=args.mpc_lr,
    )

    print("[run] NMPC + Nominal model closed loop...")
    ctrl_nom = NMPCController(predictor, nmpc_cfg, use_residual=False, device=device)
    res_nom = run_closed_loop(
        ctrl_nom, mj_model, mj_data, scratch, indices,
        q_ref_full, dq_ref_full, args.sim_steps, args.horizon, x0, label="nominal",
    )

    print("[run] NMPC + Hybrid (DeepKoopman) model closed loop...")
    ctrl_hyb = NMPCController(predictor, nmpc_cfg, use_residual=True, device=device)
    res_hyb = run_closed_loop(
        ctrl_hyb, mj_model, mj_data, scratch, indices,
        q_ref_full, dq_ref_full, args.sim_steps, args.horizon, x0, label="hybrid",
    )

    # ----- 5. metrics + plots -----
    t_state = np.arange(args.sim_steps + 1) * args.dt
    t_ctrl = np.arange(args.sim_steps) * args.dt
    q_ref_plot = q_ref_full[: args.sim_steps + 1]

    m_nom = tracking_metrics(res_nom.q_actual, q_ref_plot)
    m_hyb = tracking_metrics(res_hyb.q_actual, q_ref_plot)

    print("-" * 60)
    print(f"  Nominal  RMSE: q_a={m_nom['rmse_per_joint'][0]:.5g}  "
          f"q_b={m_nom['rmse_per_joint'][1]:.5g}  overall={m_nom['overall_rmse']:.5g}")
    print(f"  Hybrid   RMSE: q_a={m_hyb['rmse_per_joint'][0]:.5g}  "
          f"q_b={m_hyb['rmse_per_joint'][1]:.5g}  overall={m_hyb['overall_rmse']:.5g}")
    impr = (m_nom["overall_rmse"] - m_hyb["overall_rmse"]) / m_nom["overall_rmse"] if m_nom["overall_rmse"] > 0 else 0.0
    print(f"  Overall tracking-RMSE improvement (hybrid vs nominal): {100.0 * impr:.1f}%")
    print(f"  Avg NMPC solve time/step: nominal={res_nom.solve_time / args.sim_steps * 1e3:.1f} ms, "
          f"hybrid={res_hyb.solve_time / args.sim_steps * 1e3:.1f} ms")

    plot_tracking(t_state, q_ref_plot, res_nom, res_hyb)
    plot_tracking_error(t_state, m_nom, m_hyb)
    plot_rmse(t_state, m_nom, m_hyb)
    plot_joint_torque(t_ctrl, res_nom, res_hyb)
    plot_cable_tension(t_ctrl, res_nom, res_hyb)

    # ----- 6. save data + summary -----
    np.savez(
        out_dir / "closed_loop_data.npz",
        t_state=t_state, t_ctrl=t_ctrl, q_ref=q_ref_plot,
        q_nom=res_nom.q_actual, q_hyb=res_hyb.q_actual,
        tau_nom=res_nom.tau_applied, tau_hyb=res_hyb.tau_applied,
        tension_nom=res_nom.cable_tension, tension_hyb=res_hyb.cable_tension,
    )
    summary = {
        "device": str(device),
        "nominal_mass_scale": mass_scale,
        "torch_nominal_max_step_error": nom_err,
        "nmpc": {
            "horizon": args.horizon, "mpc_dt": n_sub * args.dt, "n_sub": n_sub,
            "lookahead_s": lookahead_s, "tau_max": args.tau_max,
            "w_q": args.w_q, "w_dq": args.w_dq, "w_u": args.w_u,
            "w_du": args.w_du, "w_terminal": args.w_terminal,
            "lbfgs_iters": args.lbfgs_iters,
        },
        "reference": vars(ref_cfg) if hasattr(ref_cfg, "__dict__") else {},
        "rmse": {
            "nominal": {"per_joint": m_nom["rmse_per_joint"].tolist(), "overall": m_nom["overall_rmse"]},
            "hybrid": {"per_joint": m_hyb["rmse_per_joint"].tolist(), "overall": m_hyb["overall_rmse"]},
            "improvement_ratio": impr,
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "nmpc_tracking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
