"""
dkn_cdsm_mujoco.py
===================
Deep Koopman with Control (DKN/DKUC/DKAC) on cable-driven space manipulator (CDSM)
in MuJoCo. Follows Shi & Meng 2022 (https://github.com/HaojieSHI98/DeepKoopmanWithControl).

Key design choices:
- Data collection stores NORMALIZED states + raw torques → training/inference consistent
- DKUC mode (φ = u) used for LQR control so phi is directly interpretable as torque
- Analytical "identity" cable Jacobian: the FD Jacobian is extremely slow, so we compute
  the M(q) mapping on-the-fly from scratch state only when q changes significantly,
  cached otherwise.
- Training uses normalized state inputs → A, B operate in normalized space → no
  denormalization needed between LQR and the Koopman model.
- LQR output phi = u (torque) is clipped to safe range and passed to cable mapping.

Usage:
    python dkn_cdsm_mujoco.py
    python dkn_cdsm_mujoco.py --skip_training   (load saved DKN model)
    python dkn_cdsm_mujoco.py --control_mode uc  (use DKUC mode for simpler control)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

os.environ['MUJOCO_GL'] = 'glfw'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False
    print("[WARN] MuJoCo not installed.")

import torch
import torch.nn as nn

from multi_joint_cdsm_model import MultiJointSpaceRobot
from utils_plot import save_figure, get_save_dir


# ============================================================================
# 0. 全局常量
# ============================================================================
XML_PATH = str(
    Path(__file__).resolve().parents[3]
    / "assets"
    / "models"
    / "multi_joint_cable_driven_space_robot.xml"
)
CABLE_NAMES = ["cable11","cable12","cable13","cable14",
               "cable21","cable22","cable23","cable24"]
ACTUATOR_NAMES = ["winch_c"+n[len("cable"):] for n in CABLE_NAMES]
IDX_F1P = [0, 2]; IDX_F1M = [1, 3]; IDX_F2P = [4, 6]; IDX_F2M = [5, 7]
F_PRE = 20.0
F_MAX = 2000.0
DT_SIM = 0.02
STATE_DIM = 4
CTRL_DIM = 2


# ============================================================================
# 1. MuJoCo 工具
# ============================================================================
def load_cdsm_model(dt=DT_SIM, F_pre=F_PRE, F_max=F_MAX):
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()
    xml_str = re.sub(r'range="-1\.5708 1\.5708"', 'range="-1.7 1.7"', xml_str)
    xml_str = re.sub(r'ctrlrange="0\s+2000"', f'ctrlrange="0 {F_max:g}"', xml_str)
    xml_str = re.sub(r'timestep="[^"]*"', f'timestep="{dt:g}"', xml_str)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    tdn_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n) for n in CABLE_NAMES}
    act_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
    jnt_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
              for n in ("joint1","joint2","joint3","joint4")}
    site_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    for d, label in [(tdn_id,"tendon"),(act_id,"actuator"),(jnt_id,"joint")]:
        for k,v in d.items():
            if v < 0: raise RuntimeError(f"[CDSM] 未找到 {label} {k!r}")
    if site_ee < 0: raise RuntimeError("[CDSM] 未找到 site 'end_effector'")
    dof = {n: int(model.jnt_dofadr[jnt_id[n]]) for n in ("joint1","joint2","joint3","joint4")}
    indices = dict(tdn_id=tdn_id, act_id=act_id, jnt_id=jnt_id, site_ee=site_ee,
                   dof_j1=dof["joint1"], dof_j2=dof["joint2"],
                   dof_j3=dof["joint3"], dof_j4=dof["joint4"],
                   tdn_ids_ordered=np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int),
                   act_ids_ordered=np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int))
    return model, data, scratch, indices


def compute_tendon_jacobian_fd(model, scratch, q_ref, tdn_ids_ordered, eps=1e-6):
    nv = model.nv
    nt = len(tdn_ids_ordered)
    J = np.zeros((nt, nv), dtype=float)
    q0 = scratch.qpos.copy()
    for j in range(nv):
        scratch.qpos[:] = q0; scratch.qpos[j] += eps
        mujoco.mj_fwdPosition(model, scratch)
        Lp = np.array(scratch.ten_length, dtype=float)[tdn_ids_ordered].copy()
        scratch.qpos[:] = q0; scratch.qpos[j] -= eps
        mujoco.mj_fwdPosition(model, scratch)
        Lm = np.array(scratch.ten_length, dtype=float)[tdn_ids_ordered].copy()
        J[:, j] = (Lp - Lm) / (2.0 * eps)
    scratch.qpos[:] = q0; mujoco.mj_fwdPosition(model, scratch)
    return J


def solve_pair(m_p, m_m, tau_des, F_pre, F_max):
    u_max = F_max - F_pre
    tau_base = (m_p + m_m) * F_pre
    tau_eff = tau_des - tau_base
    EPS = 1e-12
    cand = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > EPS:
        u = max(tau_eff / m_p, 0.0); uc = min(u, u_max)
        cand.append((uc, 0.0, abs(tau_eff - m_p * uc), uc))
    if abs(m_m) > EPS:
        u = max(tau_eff / m_m, 0.0); uc = min(u, u_max)
        cand.append((0.0, uc, abs(tau_eff - m_m * uc), uc))
    u_p, u_m, _, _ = min(cand, key=lambda c: (c[2], c[3]))
    return F_pre + u_p, F_pre + u_m


def torque_to_cable(tau_a, tau_b, J, dof_j1, dof_j2, dof_j3, dof_j4, F_pre, F_max):
    a = J[:, dof_j1] + J[:, dof_j2]
    b = J[:, dof_j3] + J[:, dof_j4]
    m_p1 = a[IDX_F1P].sum(); m_m1 = a[IDX_F1M].sum()
    m_p2 = b[IDX_F2P].sum(); m_m2 = b[IDX_F2M].sum()
    F1p, F1m = solve_pair(m_p1, m_m1, tau_a, F_pre, F_max)
    F2p, F2m = solve_pair(m_p2, m_m2, tau_b, F_pre, F_max)
    F = np.zeros(8)
    F[IDX_F1P] = F1p; F[IDX_F1M] = F1m
    F[IDX_F2P] = F2p; F[IDX_F2M] = F2m
    return F


def get_state_from_mj(model, data, indices):
    jnt_id = indices["jnt_id"]
    qa = float(data.qpos[model.jnt_qposadr[jnt_id["joint1"]]])
    qb = float(data.qpos[model.jnt_qposadr[jnt_id["joint3"]]])
    dqa = float(data.qvel[indices["dof_j1"]])
    dqb = float(data.qvel[indices["dof_j3"]])
    return np.array([qa, qb, dqa, dqb], dtype=np.float64)


# ============================================================================
# 2. DKN 网络 (Shi & Meng 2022)
# ============================================================================
def gaussian_init_(n_units, std=1.0):
    sampler = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([std / n_units]))
    return sampler.sample((n_units, n_units))[..., 0]


class DKNNetwork(nn.Module):
    """
    z = [x; enc(x)]   (embedding preserves original state)
    z_{t+1} = A·z + B·φ(x,u)
    control_mode:
      "uc": φ(x,u) = u
      "ac": φ(x,u) = φ_net(x)·u
      "n" : φ(x,u) = φ_net(x,u)
    """
    def __init__(self, state_dim, encode_dim, control_dim,
                 hidden_widths=(128,128,128), control_mode="n",
                 spectral_radius_init=0.95):
        super().__init__()
        self.state_dim = state_dim
        self.encode_dim = encode_dim
        self.koopman_dim = state_dim + encode_dim
        self.control_dim = control_dim
        self.control_mode = control_mode

        enc_layers = OrderedDict()
        layers = [state_dim] + list(hidden_widths) + [encode_dim]
        for i in range(len(layers)-1):
            enc_layers[f"l{i}"] = nn.Linear(layers[i], layers[i+1])
            if i != len(layers)-2:
                enc_layers[f"a{i}"] = nn.ReLU()
        self.encode_net = nn.Sequential(enc_layers)

        if control_mode == "uc":
            pass  # no control net needed
        elif control_mode == "ac":
            cl = [state_dim] + list(hidden_widths) + [state_dim * control_dim]
            c_net = OrderedDict()
            for i in range(len(cl)-1):
                c_net[f"l{i}"] = nn.Linear(cl[i], cl[i+1])
                if i != len(cl)-2:
                    c_net[f"a{i}"] = nn.ReLU()
            self.control_net = nn.Sequential(c_net)
        else:  # DKN
            cl = [state_dim + control_dim] + list(hidden_widths) + [control_dim]
            c_net = OrderedDict()
            for i in range(len(cl)-1):
                c_net[f"l{i}"] = nn.Linear(cl[i], cl[i+1])
                if i != len(cl)-2:
                    c_net[f"a{i}"] = nn.ReLU()
            self.control_net = nn.Sequential(c_net)

        k = self.koopman_dim
        self.lA = nn.Linear(k, k, bias=False)
        self.lA.weight.data = gaussian_init_(k)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * spectral_radius_init

        if control_mode == "uc":
            self.lB = nn.Linear(control_dim, k, bias=False)
        elif control_mode == "ac":
            self.lB = nn.Linear(state_dim * control_dim, k, bias=False)
        else:
            self.lB = nn.Linear(control_dim, k, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def encode_phi(self, x, u):
        if self.control_mode == "uc":
            return u
        elif self.control_mode == "ac":
            Bf = self.control_net(x).view(-1, self.state_dim, self.control_dim)
            return torch.bmm(Bf, u.unsqueeze(-1)).squeeze(-1)
        else:
            return self.control_net(torch.cat([x, u], dim=-1))

    def forward(self, z, phi):
        return self.lA(z) + self.lB(phi)


# ============================================================================
# 3. 数据收集 (MuJoCo, 存储归一化状态 + 原始力矩)
# ============================================================================
def collect_training_data(num_trajs=5000, steps=15, dt=DT_SIM,
                          q_limit=np.pi/2-0.12, tau_limit=100.0,
                          seed=42, use_mujoco=True):
    """
    返回:
      data: [steps+1, num_trajs, ctrl_dim + state_dim]  (ctrl is raw, state is NORMALIZED)
      mean, std: used for normalization
    """
    rng = np.random.RandomState(seed)
    raw = np.empty((steps+1, num_trajs, STATE_DIM + CTRL_DIM), dtype=np.float32)

    if use_mujoco and _HAS_MUJOCO:
        model, data, scratch, indices = load_cdsm_model(dt=dt)

    for traj_i in range(num_trajs):
        qa = rng.uniform(-q_limit, q_limit)
        qb = rng.uniform(-q_limit, q_limit)
        dqa = rng.uniform(-0.5, 0.5)
        dqb = rng.uniform(-0.5, 0.5)
        tau_a = rng.uniform(-tau_limit, tau_limit)
        tau_b = rng.uniform(-tau_limit, tau_limit)

        for step_i in range(steps + 1):
            raw[step_i, traj_i, :] = [tau_a, tau_b, qa, qb, dqa, dqb]
            if step_i == steps:
                break

            tau_a += rng.uniform(-tau_limit*0.2, tau_limit*0.2)
            tau_b += rng.uniform(-tau_limit*0.2, tau_limit*0.2)
            tau_a = np.clip(tau_a, -tau_limit, tau_limit)
            tau_b = np.clip(tau_b, -tau_limit, tau_limit)

            if use_mujoco and _HAS_MUJOCO:
                qidx = model.jnt_qposadr
                data.qpos[qidx[indices["jnt_id"]["joint1"]]] = qa
                data.qpos[qidx[indices["jnt_id"]["joint2"]]] = qa
                data.qpos[qidx[indices["jnt_id"]["joint3"]]] = qb
                data.qpos[qidx[indices["jnt_id"]["joint4"]]] = qb
                data.qvel[indices["dof_j1"]] = dqa
                data.qvel[indices["dof_j2"]] = dqa
                data.qvel[indices["dof_j3"]] = dqb
                data.qvel[indices["dof_j4"]] = dqb
                mujoco.mj_forward(model, data)
                J = compute_tendon_jacobian_fd(model, scratch, np.array(data.qpos),
                                               indices["tdn_ids_ordered"])
                F_cable = torque_to_cable(tau_a, tau_b, J,
                                          indices["dof_j1"], indices["dof_j2"],
                                          indices["dof_j3"], indices["dof_j4"],
                                          F_PRE, F_MAX)
                data.ctrl[indices["act_ids_ordered"]] = F_cable
                mujoco.mj_step(model, data)
                qa = float(data.qpos[qidx[indices["jnt_id"]["joint1"]]])
                qb = float(data.qpos[qidx[indices["jnt_id"]["joint3"]]])
                dqa = float(data.qvel[indices["dof_j1"]])
                dqb = float(data.qvel[indices["dof_j3"]])
            else:
                robot_m = MultiJointSpaceRobot()
                q, dq = np.array([qa, qb]), np.array([dqa, dqb])
                tau = np.array([tau_a, tau_b])
                qn, dqn = robot_m.step_coupled(q, dq, tau, dt=dt)
                qa, qb = float(qn[0]), float(qn[1])
                dqa, dqb = float(dqn[0]), float(dqn[1])

            qa = np.clip(qa, -np.pi/2, np.pi/2)
            qb = np.clip(qb, -np.pi/2, np.pi/2)

    # Compute normalization from state portion
    flat = raw[:, :, CTRL_DIM:].reshape(-1, STATE_DIM)
    mean = flat.mean(axis=0, dtype=np.float64)
    std = flat.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)

    # Build output: [ctrl (raw), state (normalized)]
    data_out = raw.copy().astype(np.float64)
    data_out[:, :, CTRL_DIM:] = (raw[:, :, CTRL_DIM:] - mean.astype(np.float32)) / std.astype(np.float32)

    print(f"[数据] {num_trajs} 条轨迹 × {steps+1} 步")
    print(f"  state mean = {mean}")
    print(f"  state std  = {std}")
    return data_out, mean.astype(np.float64), std.astype(np.float64)


# ============================================================================
# 4. 训练
# ============================================================================
def compute_klinear_loss(data_batch, net, mse_loss, ctrl_dim, gamma, state_dim, all_loss=1):
    steps, B, _ = data_batch.shape
    device = next(net.parameters()).device
    x0 = data_batch[0, :, ctrl_dim:]
    z = net.encode(x0)
    loss = torch.zeros(1, dtype=torch.float64, device=device)
    aug_loss = torch.zeros(1, dtype=torch.float64, device=device)
    beta = 1.0
    beta_sum = 0.0

    for i in range(steps - 1):
        u_t = data_batch[i, :, :ctrl_dim]
        x_next_true = data_batch[i+1, :, ctrl_dim:]
        phi = net.encode_phi(z[:, :state_dim], u_t)
        z_next = net.forward(z, phi)
        beta_sum += beta
        if not all_loss:
            loss += beta * mse_loss(z_next[:, :state_dim], x_next_true)
        else:
            z_next_true = net.encode(x_next_true)
            loss += beta * mse_loss(z_next, z_next_true)
            z_proj = net.encode(z_next[:, :state_dim])
            aug_loss += mse_loss(z_proj, z_next)
        z = z_next
        beta *= gamma

    loss = loss / beta_sum
    aug_loss = aug_loss / beta_sum
    return loss + 0.5 * aug_loss


def eig_loss(net):
    device = next(net.parameters()).device
    A = net.lA.weight
    evals = torch.linalg.eigvals(A).abs()
    c = evals - torch.ones(1, dtype=torch.float64, device=device)
    mask = c > 0
    return c[mask].sum()


def train_dkn(train_data, val_data, state_dim=STATE_DIM, ctrl_dim=CTRL_DIM,
              encode_dim=20, hidden_width=128, layer_depth=3,
              control_mode="n", train_steps=50000, batch_size=256,
              gamma=0.8, learning_rate=1e-3, e_loss_weight=1.0,
              device=torch.device("cuda"), save_dir="."):
    hidden_widths = tuple([hidden_width] * layer_depth)
    net = DKNNetwork(state_dim, encode_dim, ctrl_dim, hidden_widths,
                     control_mode).to(device)
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.5)

    num_train = train_data.shape[1]
    num_val = val_data.shape[1]
    best_loss = 1e10
    history = []
    val_no_grad = 0 if control_mode == "uc" else 0

    print(f"[训练] dim={encode_dim}, mode={control_mode}, γ={gamma}, "
          f"e_loss={e_loss_weight}, 参数={sum(p.numel() for p in net.parameters()):,}")

    for step_i in range(train_steps):
        idx = np.random.choice(num_train, batch_size, replace=False)
        batch = torch.from_numpy(train_data[:, idx, :]).to(device).double()
        loss = compute_klinear_loss(batch, net, mse_loss, ctrl_dim, gamma, state_dim, all_loss=1)
        if e_loss_weight > 0:
            loss = loss + e_loss_weight * eig_loss(net)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if (step_i + 1) % 500 == 0:
            with torch.no_grad():
                idx_v = np.random.choice(num_val, min(batch_size, num_val), replace=False)
                batch_v = torch.from_numpy(val_data[:, idx_v, :]).to(device).double()
                vl = compute_klinear_loss(batch_v, net, mse_loss, ctrl_dim, gamma, state_dim, all_loss=0)
                el = float(eig_loss(net).item())
            vloss = float(vl.item())
            if vloss < best_loss:
                best_loss = vloss
                torch.save(net.state_dict(), os.path.join(save_dir, "dkn_best.pt"))
            history.append([step_i, float(loss.detach().item()), vloss, el])
            lr_now = optimizer.param_groups[0]["lr"]
            if step_i % 1000 == 0:
                print(f"  [{step_i:5d}/{train_steps}] train={float(loss.detach().item()):.4e}  "
                      f"val={vloss:.4e}  eig={el:.2e}  lr={lr_now:.2e}")

    net.load_state_dict(torch.load(os.path.join(save_dir, "dkn_best.pt"), weights_only=True))
    np.savetxt(os.path.join(save_dir, "dkn_training_history.csv"),
               np.array(history), delimiter=",", fmt="%.8e",
               header="step,train_loss,val_loss,eig_loss", comments="")
    print(f"[训练完成] best_val_loss = {best_loss:.6e}")
    return net


# ============================================================================
# 5. LQR
# ============================================================================
def dlqr(A, B, Q, R):
    try:
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A, B, Q, R)
        return np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    except ImportError:
        pass
    X = Q.copy()
    for _ in range(500):
        Xn = A.T @ X @ A - A.T @ X @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if np.max(np.abs(Xn - X)) < 1e-6:
            X = Xn; break
        X = Xn
    return np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)


def build_lqr_controller(net, state_dim=STATE_DIM):
    k = net.koopman_dim
    A_np = net.lA.weight.detach().cpu().numpy().astype(np.float64)
    B_np = net.lB.weight.detach().cpu().numpy().astype(np.float64)
    Q = np.eye(k, dtype=np.float64)
    # Heavily weight the actual state (angles/velocities), not encoding dimensions
    Q[:state_dim, :state_dim] = np.diag([100.0, 100.0, 10.0, 10.0])
    Q[state_dim:, state_dim:] *= 0.01
    R = np.eye(CTRL_DIM, dtype=np.float64) * 0.1
    K = dlqr(A_np, B_np[:, :CTRL_DIM], Q, R)
    return K, A_np, B_np


# ============================================================================
# 6. 图-8 轨迹
# ============================================================================
def generate_figure8(robot, num_trajs=1, dt=DT_SIM, period=8.0, num_cycles=3,
                     center_x=4.0, center_y=0.0, radius_x=1.2, radius_y=0.8):
    """
    Generate figure-8 trajectory in EE space, then compute joint references via IK.
    Returns N x 2 (joint commands) or N x (2 * num_trajs).
    """
    total_time = period * num_cycles
    N = int(total_time / dt) + 1
    t_vals = np.linspace(0, total_time, N)
    omega = 2.0 * np.pi / period
    x_des = center_x + radius_x * np.sin(omega * t_vals)
    y_des = center_y + radius_y * np.sin(2.0 * omega * t_vals)

    qa_des = np.zeros(N)
    qb_des = np.zeros(N)
    q_guess = np.array([0.1, 0.1])
    for i in range(N):
        q_sol, conv = robot.inverse_kinematics([x_des[i], y_des[i]], q_guess)
        if not conv:
            print(f"  [IK] t={t_vals[i]:.2f}s not converged, ({x_des[i]:.2f},{y_des[i]:.2f})")
        qa_des[i], qb_des[i] = q_sol[0], q_sol[1]
        q_guess = q_sol
    return qa_des, qb_des, x_des, y_des, t_vals


# ============================================================================
# 7. 跟踪 (MuJoCo)
# ============================================================================
def run_figure8_tracking(net, mean, std, device, dt=DT_SIM, tau_limit=200.0,
                         period=8.0, num_cycles=3, save_dir="."):
    if not _HAS_MUJOCO:
        print("[ERROR] 需要 MuJoCo")
        return {}

    robot_math = MultiJointSpaceRobot()
    qa_des, qb_des, x_des, y_des, t_ref = generate_figure8(robot_math, dt=dt,
                                                            period=period, num_cycles=num_cycles)
    N_ref = len(t_ref)
    total_steps = N_ref + int(2.0 / dt)

    model, data, scratch, indices = load_cdsm_model(dt=dt)
    K_lqr, A_np, B_np = build_lqr_controller(net)

    # Precompute desired embeddings
    z_des_arr = np.zeros((N_ref, net.koopman_dim), dtype=np.float64)
    for i in range(N_ref):
        x_n = np.array([qa_des[i], qb_des[i], 0.0, 0.0], dtype=np.float64)
        x_n = (x_n - mean) / std
        x_t = torch.from_numpy(x_n[None, :]).to(device).double()
        with torch.no_grad():
            z_des_arr[i] = net.encode(x_t).cpu().numpy().flatten()

    record_t, record_x, record_u, record_ee, record_err = [], [], [], [], []

    # Initialize to start
    jq = model.jnt_qposadr
    data.qpos[jq[indices["jnt_id"]["joint1"]]] = qa_des[0]
    data.qpos[jq[indices["jnt_id"]["joint2"]]] = qa_des[0]
    data.qpos[jq[indices["jnt_id"]["joint3"]]] = qb_des[0]
    data.qpos[jq[indices["jnt_id"]["joint4"]]] = qb_des[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    step_counter = 0
    # Cache for J: recompute only when state changes significantly
    _cached_q = None
    _cached_J = None

    def control_callback(m, d):
        nonlocal step_counter, _cached_q, _cached_J
        k = min(int(round(d.time / dt)), N_ref - 1)

        x_cur = get_state_from_mj(m, d, indices)  # physical
        x_norm = (x_cur - mean) / std

        z_des = z_des_arr[min(k, N_ref-1)]
        x_t = torch.from_numpy(x_norm[None, :]).to(device).double()
        with torch.no_grad():
            z_cur = net.encode(x_t).cpu().numpy().flatten()
        phi_des = -K_lqr @ (z_cur - z_des)  # = torque in DKUC mode

        tau_a = float(np.clip(phi_des[0], -tau_limit, tau_limit))
        tau_b = float(np.clip(phi_des[1], -tau_limit, tau_limit))

        # Cached Jacobian: recompute only when q changes significantly
        q_now = np.array(d.qpos, dtype=float)
        if _cached_q is None or np.max(np.abs(q_now - _cached_q)) > 0.01:
            _cached_J = compute_tendon_jacobian_fd(m, scratch, q_now, indices["tdn_ids_ordered"])
            _cached_q = q_now.copy()
        J = _cached_J

        F_cable = torque_to_cable(tau_a, tau_b, J,
                                  indices["dof_j1"], indices["dof_j2"],
                                  indices["dof_j3"], indices["dof_j4"],
                                  F_PRE, F_MAX)
        d.ctrl[indices["act_ids_ordered"]] = F_cable

        if step_counter % 1 == 0:
            record_t.append(d.time)
            record_x.append(x_cur.copy())
            record_u.append([tau_a, tau_b])
            p_ee = np.array(d.site_xpos[indices["site_ee"]], dtype=float)
            record_ee.append(p_ee[:2].copy())
            err = np.sqrt((p_ee[0] - x_des[k])**2 + (p_ee[1] - y_des[k])**2)
            record_err.append(err)
        step_counter += 1

    mujoco.set_mjcb_control(control_callback)
    print("\n[跟踪] MuJoCo 仿真中...")
    try:
        for _ in range(total_steps):
            mujoco.mj_step(model, data)
    finally:
        mujoco.set_mjcb_control(None)

    record_t = np.array(record_t)
    record_x = np.array(record_x)
    record_u = np.array(record_u)
    record_ee = np.array(record_ee)
    record_err = np.array(record_err)

    rms_e = np.sqrt(np.mean(record_err**2)) if len(record_err) > 0 else 0.0
    peak_e = np.max(record_err) if len(record_err) > 0 else 0.0
    print(f"[跟踪] {len(record_t)} 步,  RMS={rms_e*1000:.1f}mm,  Peak={peak_e*1000:.1f}mm")

    return dict(t=record_t, x=record_x, u=record_u, ee=record_ee, err=record_err,
                t_ref=t_ref, qa_des=qa_des, qb_des=qb_des, x_des=x_des, y_des=y_des)


# ============================================================================
# 8. 绘图
# ============================================================================
def plot_results(results, training_history=None, save_dir="."):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    t_ref = results["t_ref"]
    x_des = results["x_des"]
    y_des = results["y_des"]

    if training_history is not None and len(training_history) > 0:
        ax = axes[0, 0]
        ax.semilogy(training_history[:, 0], training_history[:, 1], "b-", lw=1.5, label="Train")
        ax.semilogy(training_history[:, 0], training_history[:, 2], "r-", lw=1.5, label="Val")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
        ax.set_title("DKN Training Loss"); ax.grid(True, alpha=0.4); ax.legend()

    ax = axes[0, 1]
    ax.plot(x_des, y_des, "k--", lw=2, label="Desired Figure-8")
    if len(results["ee"]) > 0:
        ax.plot(results["ee"][:, 0], results["ee"][:, 1], "r-", lw=1.5, alpha=0.85, label="Actual")
        ax.plot(results["ee"][0, 0], results["ee"][0, 1], "go", ms=8, label="Start")
        ax.plot(results["ee"][-1, 0], results["ee"][-1, 1], "bs", ms=8, label="End")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("EE Figure-8 Tracking"); ax.set_aspect("equal")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    ax = axes[0, 2]
    if len(results["t"]) > 0 and len(results["x"]) > 0:
        ts = results["t"]
        ax.plot(ts, np.rad2deg(results["x"][:, 0]), "b-", lw=1.5, label=r"$q_a$")
        ax.plot(ts, np.rad2deg(results["x"][:, 1]), "r-", lw=1.5, label=r"$q_b$")
        qa_i = np.interp(ts, t_ref, results["qa_des"])
        qb_i = np.interp(ts, t_ref, results["qb_des"])
        ax.plot(ts, np.rad2deg(qa_i), "b--", lw=1, alpha=0.6, label=r"$q_a^{des}$")
        ax.plot(ts, np.rad2deg(qb_i), "r--", lw=1, alpha=0.6, label=r"$q_b^{des}$")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.set_title("Joint Tracking"); ax.grid(True, alpha=0.4); ax.legend(fontsize=8)

    ax = axes[1, 0]
    if len(results["err"]) > 0:
        ax.plot(results["t"], results["err"] * 1000, "m-", lw=1.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (mm)")
    ax.set_title("Cartesian Error"); ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    if len(results["t"]) > 0 and len(results["x"]) > 0:
        ts = results["t"]
        qa_i = np.interp(ts, t_ref, results["qa_des"])
        qb_i = np.interp(ts, t_ref, results["qb_des"])
        ax.plot(ts, np.rad2deg(results["x"][:, 0] - qa_i), "b-", lw=1.5, label=r"$e_a$")
        ax.plot(ts, np.rad2deg(results["x"][:, 1] - qb_i), "r-", lw=1.5, label=r"$e_b$")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Joint Error (deg)")
    ax.set_title("Joint Error"); ax.grid(True, alpha=0.4); ax.legend()

    ax = axes[1, 2]
    if len(results["u"]) > 0:
        ax.plot(results["t"], results["u"][:, 0], "b-", lw=1.5, label=r"$\tau_a$")
        ax.plot(results["t"], results["u"][:, 1], "r-", lw=1.5, label=r"$\tau_b$")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
    ax.set_title("LQR Torques"); ax.grid(True, alpha=0.4); ax.legend()

    plt.suptitle("Deep Koopman (DKUC) Figure-8 Tracking", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig_name="dkn_figure8_tracking")
    plt.close()

    if len(results["ee"]) > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        ax2.plot(x_des, y_des, "k--", lw=2, alpha=0.7, label="Desired")
        ee = results["ee"]; n = len(ee)
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        for i in range(n-1):
            ax2.plot(ee[i:i+2, 0], ee[i:i+2, 1], color=colors[i], lw=1.0, alpha=0.7)
        ax2.plot(ee[0, 0], ee[0, 1], "go", ms=10, label="Start")
        ax2.plot(ee[-1, 0], ee[-1, 1], "rs", ms=10, label="End")
        ax2.scatter(x_des[0], y_des[0], c="orange", s=100, marker="*", zorder=5)
        ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
        ax2.set_title("EE Trajectory (color=time)")
        ax2.set_aspect("equal"); ax2.grid(True, alpha=0.4); ax2.legend(fontsize=9)
        plt.tight_layout()
        save_figure(fig_name="dkn_ee_detail")
        plt.close()
    print(f"[绘图] 完成: {save_dir}")


# ============================================================================
# 9. 主入口
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description="DKN/DKUC on CDSM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_train_trajs", type=int, default=3000)
    p.add_argument("--num_val_trajs", type=int, default=500)
    p.add_argument("--k_steps", type=int, default=15)
    p.add_argument("--dt", type=float, default=DT_SIM)
    p.add_argument("--tau_limit_data", type=float, default=100.0,
                   help="Torque limit for data collection (Nm)")
    p.add_argument("--tau_limit_ctrl", type=float, default=200.0,
                   help="Torque limit for control (Nm)")
    p.add_argument("--encode_dim", type=int, default=20)
    p.add_argument("--hidden_width", type=int, default=128)
    p.add_argument("--layer_depth", type=int, default=3)
    p.add_argument("--control_mode", type=str, default="uc", choices=["uc", "ac", "n"])
    p.add_argument("--train_steps", type=int, default=60000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--e_loss_weight", type=float, default=1.0)
    p.add_argument("--period", type=float, default=8.0)
    p.add_argument("--num_cycles", type=int, default=3)
    p.add_argument("--center_x", type=float, default=4.0)
    p.add_argument("--center_y", type=float, default=0.0)
    p.add_argument("--radius_x", type=float, default=1.2)
    p.add_argument("--radius_y", type=float, default=0.8)
    p.add_argument("--out_dir", type=str, default="dkn_cdsm_results")
    p.add_argument("--use_mujoco_data", action="store_true", default=True)
    p.add_argument("--skip_training", action="store_true", default=False)
    return p


def main():
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    fig_save_dir = get_save_dir()
    out_dir = Path(fig_save_dir) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Deep Koopman ({args.control_mode.upper()}) on CDSM  |  "
          f"Device: {device}")
    print(f"  encode_dim={args.encode_dim}  K={args.k_steps}  "
          f"γ={args.gamma}  train={args.train_steps}")
    print("=" * 70)

    # Step 1: 数据收集
    print("\n[Step 1] 收集数据...")
    data, mean, std = collect_training_data(
        num_trajs=args.num_train_trajs + args.num_val_trajs,
        steps=args.k_steps, dt=args.dt,
        tau_limit=args.tau_limit_data,
        seed=args.seed, use_mujoco=args.use_mujoco_data)
    np.savez(out_dir / "data_normalization.npz", mean=mean, std=std)
    train_data = data[:, :args.num_train_trajs, :]
    val_data = data[:, args.num_train_trajs:, :]
    print(f"  训练: {train_data.shape}  验证: {val_data.shape}")

    # Step 2: 训练
    print("\n[Step 2] 训练...")
    model_path = out_dir / "dkn_best.pt"
    if args.skip_training and model_path.exists():
        net = DKNNetwork(STATE_DIM, args.encode_dim, CTRL_DIM,
                         tuple([args.hidden_width]*args.layer_depth),
                         args.control_mode).to(device)
        net.double()
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("  [已加载已有模型]")
    else:
        net = train_dkn(train_data, val_data,
                        encode_dim=args.encode_dim,
                        hidden_width=args.hidden_width,
                        layer_depth=args.layer_depth,
                        control_mode=args.control_mode,
                        train_steps=args.train_steps,
                        batch_size=args.batch_size,
                        gamma=args.gamma,
                        learning_rate=args.lr,
                        e_loss_weight=args.e_loss_weight,
                        device=device, save_dir=str(out_dir))

    # Step 3: 跟踪
    print("\n[Step 3] 图-8 跟踪...")
    results = run_figure8_tracking(net, mean, std, device,
                                   dt=args.dt,
                                   tau_limit=args.tau_limit_ctrl,
                                   period=args.period,
                                   num_cycles=args.num_cycles,
                                   save_dir=str(out_dir))

    # Step 4: 绘图
    print("\n[Step 4] 绘图...")
    history = None
    hp = out_dir / "dkn_training_history.csv"
    if hp.exists():
        history = np.loadtxt(hp, delimiter=",", skiprows=1)
    plot_results(results, training_history=history, save_dir=str(out_dir))

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n全部完成!  模型: {model_path}  图: {fig_save_dir}")


if __name__ == "__main__":
    main()
