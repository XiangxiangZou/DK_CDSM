"""
cdsm_dkac_vs_edmd_tracking_control.py
=====================================

Shi & Meng style DKAC tracking-control reproduction on the MuJoCo
cable-driven space manipulator, compared with Koopman-EDMD control.

Main model forms
----------------
DKAC:
    z = [x_n, phi_x(x_n)]
    v = G(x_n) u_n
    z_next = A z + B v

Koopman-EDMD:
    z = [x_n, rbf(x_n)]
    z_next = A z + B u_n

Both controllers solve the same finite-horizon unconstrained linear-quadratic
tracking problem in Koopman space. For DKAC, the optimal lifted control v is
mapped back to the physical normalized input by the local least-squares inverse
of G(x). For EDMD, the optimized variable is directly u_n.

Data are collected from the MuJoCo cable-driven plant using broad-range PD
multi-sine trajectories. Torque clipping and cable maximum-tension clipping are
intentionally disabled in this file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from cdsm_dkn_vs_edmd_prediction_compare import (
    CABLE_NAMES,
    CONTROL_DIM,
    F_MAX_CABLE,
    F_PRELOAD,
    STATE_DIM,
    STATE_LABELS,
    XML_DEFAULT,
    EdmdConfig,
    EdmdPredictor,
    MLP,
    Normalizer,
    PDCollectConfig,
    build_windows,
    cable_antagonistic_map,
    collect_pd_trajectories,
    compute_tendon_jacobian_fd,
    fit_full_edmd,
    get_active_state,
    load_cable_model,
    make_device,
    sample_window_batch,
    set_active_state,
    set_seed,
)
from utils_plot import get_save_dir, save_figure


@dataclass
class DKACConfig:
    lift_dim: int
    hidden: Tuple[int, ...]
    control_hidden: Tuple[int, ...]
    control_dim_hat: int
    activation: str
    bound_lift: float
    identity_control_bias: bool
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


class DKACModel(nn.Module):
    """
    Deep Koopman Affine with Control.

    The auxiliary network returns a state-dependent matrix G(x), and the
    Koopman input is v = G(x) u. With control_dim_hat == CONTROL_DIM and
    identity_control_bias enabled, the initial map is close to v = u.
    """

    def __init__(
        self,
        lift_dim: int,
        hidden: Tuple[int, ...],
        control_hidden: Tuple[int, ...],
        control_dim_hat: int,
        activation: str,
        bound_lift: float,
        identity_control_bias: bool,
    ) -> None:
        super().__init__()
        self.lift_dim = int(lift_dim)
        self.latent_dim = STATE_DIM + self.lift_dim
        self.control_dim_hat = int(control_dim_hat)
        self.identity_control_bias = bool(identity_control_bias)
        self.encoder = MLP((STATE_DIM,) + tuple(hidden) + (self.lift_dim,), activation)
        self.control_net = MLP(
            (STATE_DIM,) + tuple(control_hidden) + (self.control_dim_hat * CONTROL_DIM,),
            activation,
        )
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim_hat, self.latent_dim, bias=False)
        self.bound_lift = float(bound_lift)
        self._init_linear()

    def _init_linear(self) -> None:
        with torch.no_grad():
            self.A.weight.zero_()
            self.A.weight += torch.eye(self.latent_dim)
            self.B.weight.zero_()
            rows = min(STATE_DIM, self.control_dim_hat)
            self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

    def lift(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_norm)
        if self.bound_lift > 0.0:
            h = self.bound_lift * torch.tanh(h / self.bound_lift)
        return torch.cat([x_norm, h], dim=-1)

    @staticmethod
    def state_from_latent(z: torch.Tensor) -> torch.Tensor:
        return z[..., :STATE_DIM]

    def control_matrix(self, x_norm: torch.Tensor) -> torch.Tensor:
        raw = self.control_net(x_norm)
        G = raw.reshape(-1, self.control_dim_hat, CONTROL_DIM)
        if self.identity_control_bias and self.control_dim_hat == CONTROL_DIM:
            eye = torch.eye(CONTROL_DIM, device=x_norm.device, dtype=x_norm.dtype)
            G = G + eye.reshape(1, CONTROL_DIM, CONTROL_DIM)
        return G

    def control_encode(self, x_norm: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        G = self.control_matrix(x_norm)
        return torch.bmm(G, u_norm.unsqueeze(-1)).squeeze(-1)

    def step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        x_for_control = self.state_from_latent(z)
        v = self.control_encode(x_for_control, u_norm)
        return self.A(z) + self.B(v)


def compute_dkac_losses(
    model: DKACModel,
    xseq: torch.Tensor,
    useq: torch.Tensor,
    w_state: float,
    w_embed: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon = useq.shape[1]
    z = model.lift(xseq[:, 0])
    state_loss = torch.zeros((), device=xseq.device)
    embed_loss = torch.zeros((), device=xseq.device)
    for k in range(horizon):
        z = model.step(z, useq[:, k])
        x_pred = model.state_from_latent(z)
        z_true = model.lift(xseq[:, k + 1])
        state_loss = state_loss + torch.mean((x_pred - xseq[:, k + 1]) ** 2)
        embed_loss = embed_loss + torch.mean((z - z_true) ** 2)
    state_loss = state_loss / horizon
    embed_loss = embed_loss / horizon
    total = float(w_state) * state_loss + float(w_embed) * embed_loss
    return total, state_loss.detach(), embed_loss.detach()


def train_dkac(
    train_norm: Dict[str, np.ndarray],
    val_norm: Dict[str, np.ndarray],
    cfg: DKACConfig,
    device: torch.device,
    out_dir: Path,
) -> Tuple[DKACModel, List[List[float]], Dict[str, float]]:
    Xw_train, Uw_train = build_windows(train_norm["states"], train_norm["inputs"], cfg.window)
    Xw_val, Uw_val = build_windows(val_norm["states"], val_norm["inputs"], cfg.window)

    model = DKACModel(
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        control_hidden=tuple(cfg.control_hidden),
        control_dim_hat=cfg.control_dim_hat,
        activation=cfg.activation,
        bound_lift=cfg.bound_lift,
        identity_control_bias=cfg.identity_control_bias,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_full = float("inf")
    best_path = out_dir / "best_dkac.pt"
    history: List[List[float]] = []
    val_x = torch.from_numpy(Xw_val.astype(np.float32)).to(device)
    val_u = torch.from_numpy(Uw_val.astype(np.float32)).to(device)

    for epoch in range(1, cfg.epochs + 1):
        h = min(cfg.window, max(1, cfg.window_start + (epoch - 1) * (cfg.window - cfg.window_start) // max(1, cfg.epochs - 1)))
        model.train()
        losses = []
        for _ in range(cfg.steps_per_epoch):
            xb, ub = sample_window_batch(Xw_train[:, : h + 1], Uw_train[:, :h], cfg.batch_size, device)
            opt.zero_grad(set_to_none=True)
            loss, state_loss, embed_loss = compute_dkac_losses(model, xb, ub, cfg.w_state, cfg.w_embed)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append([float(loss.detach().cpu()), float(state_loss.cpu()), float(embed_loss.cpu())])

        model.eval()
        with torch.no_grad():
            val_total, val_state, val_embed = compute_dkac_losses(
                model, val_x[:, : h + 1], val_u[:, :h], cfg.w_state, cfg.w_embed
            )
            val_full_total, val_full_state, val_full_embed = compute_dkac_losses(
                model, val_x, val_u, cfg.w_state, cfg.w_embed
            )
        train_mean = np.mean(np.asarray(losses, dtype=np.float64), axis=0)
        row = [
            float(epoch),
            float(train_mean[0]),
            float(train_mean[1]),
            float(train_mean[2]),
            float(val_total.cpu()),
            float(val_state.cpu()),
            float(val_embed.cpu()),
            float(h),
            float(val_full_total.cpu()),
            float(val_full_state.cpu()),
            float(val_full_embed.cpu()),
        ]
        history.append(row)
        if row[8] < best_val_full:
            best_val_full = row[8]
            torch.save(model.state_dict(), best_path)
        if epoch == 1 or epoch == cfg.epochs or epoch % max(1, cfg.epochs // 10) == 0:
            print(
                f"[dkac] epoch {epoch:03d}/{cfg.epochs:03d} H={h:02d} "
                f"train={row[1]:.3e} valH={row[4]:.3e} valFull={row[8]:.3e}",
                flush=True,
            )

    try:
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(state_dict)
    np.savetxt(
        out_dir / "dkac_training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header=(
            "epoch,train_total,train_state,train_embed,"
            "val_curriculum_total,val_curriculum_state,val_curriculum_embed,horizon,"
            "val_full_total,val_full_state,val_full_embed"
        ),
        comments="",
    )
    return model, history, {
        "best_val_full": best_val_full,
        "best_path": str(best_path),
        "best_selection": "fixed full-window validation loss",
    }


class DKACRuntime:
    def __init__(
        self,
        model: DKACModel,
        x_normer: Normalizer,
        u_normer: Normalizer,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.x_normer = x_normer
        self.u_normer = u_normer
        self.device = device
        self.A = model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B = model.B.weight.detach().cpu().numpy().astype(np.float64)
        self.C = np.zeros((STATE_DIM, model.latent_dim), dtype=np.float64)
        self.C[:, :STATE_DIM] = np.eye(STATE_DIM)
        self.latent_dim = model.latent_dim
        self.control_dim_hat = model.control_dim_hat

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_n = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            z = self.model.lift(torch.from_numpy(x_n).to(self.device)).cpu().numpy()[0]
        return z.astype(np.float64)

    def control_matrix(self, x_phys: np.ndarray) -> np.ndarray:
        x_n = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            G = self.model.control_matrix(torch.from_numpy(x_n).to(self.device)).cpu().numpy()[0]
        return G.astype(np.float64)

    def recover_u_norm(self, x_phys: np.ndarray, v: np.ndarray) -> np.ndarray:
        G = self.control_matrix(x_phys)
        return np.linalg.pinv(G, rcond=1e-6) @ np.asarray(v, dtype=np.float64).reshape(-1)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:STATE_DIM]
        x_phys = self.x_normer.inverse(x_norm.reshape(1, -1))[0]
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        v = self.control_matrix(x_phys) @ u_norm
        return self.A @ z + self.B @ v

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:STATE_DIM]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]


class EdmdRuntime:
    def __init__(self, predictor: EdmdPredictor) -> None:
        self.pred = predictor
        self.x_normer = predictor.x_normer
        self.u_normer = predictor.u_normer
        self.A = predictor.A
        self.B = predictor.B
        self.C = np.zeros((STATE_DIM, predictor.latent_dim), dtype=np.float64)
        self.C[:, :STATE_DIM] = np.eye(STATE_DIM)
        self.latent_dim = predictor.latent_dim
        self.control_dim_hat = CONTROL_DIM

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        return self.pred.lift(x_phys)

    def recover_u_norm(self, _x_phys: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=np.float64).reshape(CONTROL_DIM)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray) -> np.ndarray:
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        return self.A @ z + self.B @ u_norm

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:STATE_DIM]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]


@dataclass
class LqrTrackConfig:
    horizon: int
    Qq: float
    Qdq: float
    R: float
    Rd: float


class KoopmanLqrTracker:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cfg: LqrTrackConfig) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.cfg = cfg
        self.N = int(cfg.horizon)
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]

        D = self.A.shape[0]
        Phi = np.zeros((self.N * self.ny, D), dtype=np.float64)
        Gamma = np.zeros((self.N * self.ny, self.N * self.nu), dtype=np.float64)
        Apow = [np.eye(D)]
        for _ in range(self.N):
            Apow.append(self.A @ Apow[-1])
        for i in range(self.N):
            Phi[i * self.ny : (i + 1) * self.ny] = self.C @ Apow[i + 1]
            for j in range(i + 1):
                Gamma[
                    i * self.ny : (i + 1) * self.ny,
                    j * self.nu : (j + 1) * self.nu,
                ] = self.C @ Apow[i - j] @ self.B

        q_diag = np.tile(np.array([cfg.Qq, cfg.Qq, cfg.Qdq, cfg.Qdq], dtype=np.float64), self.N)
        Qbar = np.diag(q_diag)
        Rbar = np.eye(self.N * self.nu) * cfg.R
        Dmat = np.zeros((self.N * self.nu, self.N * self.nu), dtype=np.float64)
        for k in range(self.N):
            Dmat[k * self.nu : (k + 1) * self.nu, k * self.nu : (k + 1) * self.nu] = np.eye(self.nu)
            if k > 0:
                Dmat[k * self.nu : (k + 1) * self.nu, (k - 1) * self.nu : k * self.nu] = -np.eye(self.nu)
        Emat = np.zeros((self.N * self.nu, self.nu), dtype=np.float64)
        Emat[: self.nu] = np.eye(self.nu)
        Rdbar = np.eye(self.N * self.nu) * cfg.Rd
        H = Gamma.T @ Qbar @ Gamma + Rbar + Dmat.T @ Rdbar @ Dmat
        H = H + 1e-8 * np.eye(H.shape[0])

        self.Phi = Phi
        self.Gamma = Gamma
        self.Qbar = Qbar
        self.Rdbar = Rdbar
        self.Dmat = Dmat
        self.Emat = Emat
        self.Hinv = np.linalg.inv(H)

    def solve(self, z0: np.ndarray, ref_norm: np.ndarray, prev_v: np.ndarray) -> np.ndarray:
        r = np.asarray(ref_norm, dtype=np.float64).reshape(-1)
        v_prev = np.asarray(prev_v, dtype=np.float64).reshape(self.nu)
        y_free = self.Phi @ np.asarray(z0, dtype=np.float64).reshape(-1)
        b = self.Gamma.T @ self.Qbar @ (y_free - r)
        b = b - self.Dmat.T @ self.Rdbar @ self.Emat @ v_prev
        V = -self.Hinv @ b
        return V.reshape(self.N, self.nu)


def build_tracking_reference(
    dt: float,
    steps: int,
    amp_a: float,
    amp_b: float,
    omega_a: float,
    omega_b: float,
    phase_b: float,
) -> Dict[str, np.ndarray]:
    t = np.arange(steps + 1, dtype=np.float64) * dt
    q_a = amp_a * np.sin(omega_a * t)
    q_b = amp_b * np.sin(omega_b * t + phase_b)
    dq_a = amp_a * omega_a * np.cos(omega_a * t)
    dq_b = amp_b * omega_b * np.cos(omega_b * t + phase_b)
    return {
        "t": t,
        "q_ref": np.column_stack([q_a, q_b]),
        "dq_ref": np.column_stack([dq_a, dq_b]),
    }


def pad_ref_norm(ref: Dict[str, np.ndarray], x_normer: Normalizer, k: int, horizon: int) -> np.ndarray:
    out = np.zeros((horizon, STATE_DIM), dtype=np.float64)
    last = ref["q_ref"].shape[0] - 1
    for i in range(horizon):
        idx = min(k + 1 + i, last)
        x_ref = np.array(
            [ref["q_ref"][idx, 0], ref["q_ref"][idx, 1], ref["dq_ref"][idx, 0], ref["dq_ref"][idx, 1]],
            dtype=np.float64,
        )
        out[i] = x_normer.transform(x_ref.reshape(1, -1))[0]
    return out


def run_closed_loop(
    *,
    name: str,
    runtime: object,
    tracker: KoopmanLqrTracker,
    xml: str,
    dt: float,
    ref: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    import mujoco

    model, data, scratch, indices = load_cable_model(xml, dt)
    set_active_state(model, data, indices, ref["q_ref"][0], ref["dq_ref"][0])
    mujoco.mj_forward(model, data)

    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])
    steps = len(ref["t"]) - 1
    prev_v = np.zeros(tracker.nu, dtype=np.float64)

    rec: Dict[str, List[np.ndarray]] = {
        "t": [],
        "x": [],
        "u": [],
        "v": [],
        "q_ref": [],
        "dq_ref": [],
        "solve_ms": [],
        "cable_tensions": [],
    }
    report_every = max(1, steps // 5)
    print(f"[control:{name}] start steps={steps}, horizon={tracker.N}")
    for k in range(steps):
        x_meas = get_active_state(data, indices)
        z0 = runtime.lift(x_meas)
        ref_norm = pad_ref_norm(ref, runtime.x_normer, k, tracker.N)
        tic = time.perf_counter()
        V = tracker.solve(z0, ref_norm, prev_v)
        solve_ms = 1e3 * (time.perf_counter() - tic)
        v0 = V[0]
        u_norm = runtime.recover_u_norm(x_meas, v0)
        u_cmd = runtime.u_normer.inverse(u_norm.reshape(1, -1))[0]

        J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
        F_cable = cable_antagonistic_map(
            float(u_cmd[0]),
            float(u_cmd[1]),
            J,
            dof_j1,
            dof_j2,
            dof_j3,
            dof_j4,
            f_pre=F_PRELOAD,
            f_max=F_MAX_CABLE,
        )
        data.ctrl[indices["actuator_ids"]] = F_cable
        mujoco.mj_step(model, data)

        rec["t"].append(np.array(k * dt))
        rec["x"].append(x_meas.copy())
        rec["u"].append(u_cmd.copy())
        rec["v"].append(v0.copy())
        rec["q_ref"].append(ref["q_ref"][k].copy())
        rec["dq_ref"].append(ref["dq_ref"][k].copy())
        rec["solve_ms"].append(np.array(solve_ms))
        rec["cable_tensions"].append(F_cable.copy())
        prev_v = v0

        if k == 0 or (k + 1) % report_every == 0 or k + 1 == steps:
            err = float(np.linalg.norm(x_meas[:2] - ref["q_ref"][k]))
            print(f"[control:{name}] {k + 1:04d}/{steps} |e_q|={err:.4g} solve={solve_ms:.3f}ms")

    return {key: np.asarray(value) for key, value in rec.items()}


def tracking_metrics(log: Dict[str, np.ndarray]) -> Dict[str, object]:
    e_q = log["x"][:, :2] - log["q_ref"]
    e_dq = log["x"][:, 2:] - log["dq_ref"]
    return {
        "rmse_q": float(np.sqrt(np.mean(e_q * e_q))),
        "rmse_q_by_joint": np.sqrt(np.mean(e_q * e_q, axis=0)).tolist(),
        "max_abs_q": float(np.max(np.abs(e_q))),
        "rmse_dq": float(np.sqrt(np.mean(e_dq * e_dq))),
        "mean_solve_ms": float(np.mean(log["solve_ms"])),
        "peak_abs_tau": float(np.max(np.abs(log["u"]))),
        "peak_cable_tension": float(np.max(log["cable_tensions"])),
    }


def predict_validation_rollouts(runtime: object, val_raw: Dict[str, np.ndarray]) -> Dict[str, object]:
    """
    Open-loop model rollout on recorded validation inputs.

    This isolates model accuracy from controller tracking: every predictor starts
    from the true MuJoCo initial state and then rolls forward using the same
    recorded PD inputs from the validation dataset.
    """
    states = val_raw["states"]
    inputs = val_raw["inputs"]
    preds = np.zeros_like(states)
    for i in range(states.shape[0]):
        z = runtime.lift(states[i, 0])
        preds[i, 0] = states[i, 0]
        for k in range(inputs.shape[1]):
            z = runtime.step_latent(z, inputs[i, k])
            preds[i, k + 1] = runtime.recover_state(z)
    err = preds - states
    return {
        "preds": preds,
        "states_true": states,
        "rmse_by_state": np.sqrt(np.mean(err[:, 1:, :] * err[:, 1:, :], axis=(0, 1))),
        "step_rmse": np.sqrt(np.mean(err * err, axis=(0, 2))),
        "total_rmse": float(np.sqrt(np.mean(err[:, 1:, :] * err[:, 1:, :]))),
    }


def model_prediction_metrics(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "total_rmse": float(result["total_rmse"]),
        "rmse_by_state": np.asarray(result["rmse_by_state"], dtype=np.float64).tolist(),
        "final_step_rmse": float(np.asarray(result["step_rmse"], dtype=np.float64)[-1]),
    }


def plot_training_history(history: List[List[float]]) -> None:
    arr = np.asarray(history, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(arr[:, 0], arr[:, 1], label="train")
    ax.semilogy(arr[:, 0], arr[:, 4], label="val curriculum")
    if arr.shape[1] > 8:
        ax.semilogy(arr[:, 0], arr[:, 8], label="val full-window")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DKAC loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure("dkac_training_history")
    plt.close(fig)


def plot_model_prediction_error(
    res_dkac: Dict[str, object],
    res_edmd: Dict[str, object],
    dt: float,
) -> None:
    t = np.arange(np.asarray(res_dkac["step_rmse"]).shape[0]) * dt
    step_d = np.asarray(res_dkac["step_rmse"], dtype=np.float64)
    step_e = np.asarray(res_edmd["step_rmse"], dtype=np.float64)
    rmse_d = np.asarray(res_dkac["rmse_by_state"], dtype=np.float64)
    rmse_e = np.asarray(res_edmd["rmse_by_state"], dtype=np.float64)
    labels = STATE_LABELS + ["overall"]
    vals_d = np.r_[rmse_d, float(res_dkac["total_rmse"])]
    vals_e = np.r_[rmse_e, float(res_edmd["total_rmse"])]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2))
    axes[0].plot(t, step_d, lw=1.7, label=f"DKAC model (RMSE={float(res_dkac['total_rmse']):.3g})")
    axes[0].plot(t, step_e, lw=1.7, label=f"EDMD model (RMSE={float(res_edmd['total_rmse']):.3g})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("State RMSE")
    axes[0].set_title("Open-loop rollout error on validation MuJoCo trajectories")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, vals_d, width, label="DKAC model")
    axes[1].bar(x + width / 2, vals_e, width, label="EDMD model")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("RMSE in physical state coordinates")
    axes[1].set_title("Validation rollout RMSE by state")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Model prediction error: DKAC vs EDMD vs MuJoCo")
    fig.tight_layout()
    save_figure("model_prediction_error_compare")
    plt.close(fig)


def plot_joint_tracking(log_dkac: Dict[str, np.ndarray], log_edmd: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, label in enumerate(["q_a", "q_b"]):
        axes[j].plot(log_dkac["t"], log_dkac["q_ref"][:, j], "k--", lw=1.6, label="reference")
        axes[j].plot(log_dkac["t"], log_dkac["x"][:, j], lw=1.4, label="DKAC-LQR")
        axes[j].plot(log_edmd["t"], log_edmd["x"][:, j], lw=1.4, label="EDMD-LQR")
        axes[j].set_ylabel(f"{label} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Closed-loop joint tracking on MuJoCo")
    fig.tight_layout()
    save_figure("joint_tracking_compare")
    plt.close(fig)


def plot_tracking_error(log_dkac: Dict[str, np.ndarray], log_edmd: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, label in enumerate(["q_a", "q_b"]):
        e_d = log_dkac["x"][:, j] - log_dkac["q_ref"][:, j]
        e_e = log_edmd["x"][:, j] - log_edmd["q_ref"][:, j]
        axes[j].plot(log_dkac["t"], e_d, lw=1.4, label=f"DKAC RMSE={np.sqrt(np.mean(e_d * e_d)):.3g}")
        axes[j].plot(log_edmd["t"], e_e, lw=1.4, label=f"EDMD RMSE={np.sqrt(np.mean(e_e * e_e)):.3g}")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"e_{label} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint tracking error")
    fig.tight_layout()
    save_figure("tracking_error_compare")
    plt.close(fig)


def plot_tracking_rmse(log_dkac: Dict[str, np.ndarray], log_edmd: Dict[str, np.ndarray]) -> None:
    e_d = log_dkac["x"][:, :2] - log_dkac["q_ref"]
    e_e = log_edmd["x"][:, :2] - log_edmd["q_ref"]
    vals_d = np.r_[np.sqrt(np.mean(e_d * e_d, axis=0)), np.sqrt(np.mean(e_d * e_d))]
    vals_e = np.r_[np.sqrt(np.mean(e_e * e_e, axis=0)), np.sqrt(np.mean(e_e * e_e))]
    labels = ["q_a", "q_b", "overall"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, vals_d, width, label="DKAC-LQR")
    ax.bar(x + width / 2, vals_e, width, label="EDMD-LQR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (rad)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.suptitle("Tracking RMSE comparison")
    fig.tight_layout()
    save_figure("tracking_rmse_compare")
    plt.close(fig)


def plot_control_inputs(log_dkac: Dict[str, np.ndarray], log_edmd: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, label in enumerate(["tau_a", "tau_b"]):
        axes[j].plot(log_dkac["t"], log_dkac["u"][:, j], lw=1.3, label="DKAC-LQR")
        axes[j].plot(log_edmd["t"], log_edmd["u"][:, j], lw=1.3, label="EDMD-LQR")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"{label} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint torque commands (no torque clipping)")
    fig.tight_layout()
    save_figure("joint_torque_compare")
    plt.close(fig)


def plot_cable_tensions(log_dkac: Dict[str, np.ndarray], log_edmd: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, log, title in [
        (axes[0], log_dkac, "DKAC-LQR cable tensions"),
        (axes[1], log_edmd, "EDMD-LQR cable tensions"),
    ]:
        for i, name in enumerate(CABLE_NAMES):
            ax.plot(log["t"], log["cable_tensions"][:, i], lw=1.0, label=name)
        ax.axhline(F_PRELOAD, color="k", lw=0.8, alpha=0.5, label=f"preload={F_PRELOAD:.0f}N")
        ax.set_ylabel("Tension (N)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=5)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Cable tensions (no upper tension clipping)")
    fig.tight_layout()
    save_figure("cable_tensions_compare")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CDSM DKAC tracking control vs Koopman-EDMD.")
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=62)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")

    p.add_argument("--train_traj", type=int, default=160)
    p.add_argument("--val_traj", type=int, default=24)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--q_init_range", type=float, default=1.8)
    p.add_argument("--dq_init_range", type=float, default=1.4)
    p.add_argument("--amp_min", type=float, default=0.35)
    p.add_argument("--amp_max", type=float, default=1.6)
    p.add_argument("--omega_min", type=float, default=0.25)
    p.add_argument("--omega_max", type=float, default=1.8)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=90.0)
    p.add_argument("--kd_a", type=float, default=24.0)
    p.add_argument("--kd_b", type=float, default=20.0)
    p.add_argument("--tau_max", type=float, default=float("inf"))

    p.add_argument("--lift_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    p.add_argument("--control_hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--control_dim_hat", type=int, default=CONTROL_DIM)
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--bound_lift", type=float, default=10.0)
    p.add_argument("--no_identity_control_bias", action="store_true")
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--window_start", type=int, default=2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--steps_per_epoch", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_embed", type=float, default=0.1)

    p.add_argument("--edmd_centers", type=int, default=220)
    p.add_argument("--edmd_sigma", type=float, default=None)
    p.add_argument("--edmd_ridge", type=float, default=1e-4)
    p.add_argument("--edmd_seed", type=int, default=123)

    p.add_argument("--track_steps", type=int, default=650)
    p.add_argument("--track_horizon", type=int, default=25)
    p.add_argument("--track_amp_a", type=float, default=0.9)
    p.add_argument("--track_amp_b", type=float, default=0.75)
    p.add_argument("--track_omega_a", type=float, default=0.55)
    p.add_argument("--track_omega_b", type=float, default=0.75)
    p.add_argument("--track_phase_b", type=float, default=0.8)
    p.add_argument("--Qq", type=float, default=30.0)
    p.add_argument("--Qdq", type=float, default=2.0)
    p.add_argument("--R", type=float, default=0.15)
    p.add_argument("--Rd", type=float, default=0.05)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    t0 = time.time()
    print("=== CDSM DKAC tracking control vs Koopman-EDMD ===")
    print(f"device={device}, output={out_dir}")
    print("[policy] torque clipping disabled; cable max-tension clipping disabled")

    print("[1/7] Collecting broad-range PD MuJoCo data...")
    model, data, scratch, indices = load_cable_model(args.xml, args.dt)
    pd_train = PDCollectConfig(
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
    pd_val = PDCollectConfig(**{**asdict(pd_train), "traj_count": args.val_traj, "seed": args.seed + 1000})
    train_raw, train_meta = collect_pd_trajectories(model, data, scratch, indices, pd_train)
    val_raw, val_meta = collect_pd_trajectories(model, data, scratch, indices, pd_val)
    print(f"      train={train_raw['states'].shape}, val={val_raw['states'].shape}")

    print("[2/7] Normalizing data...")
    X_train = train_raw["states"].reshape(-1, STATE_DIM)
    U_train = train_raw["inputs"].reshape(-1, CONTROL_DIM)
    x_normer = Normalizer.fit(X_train)
    u_normer = Normalizer.fit(U_train)
    train_norm = {
        "states": x_normer.transform(train_raw["states"].reshape(-1, STATE_DIM)).reshape(train_raw["states"].shape),
        "inputs": u_normer.transform(train_raw["inputs"].reshape(-1, CONTROL_DIM)).reshape(train_raw["inputs"].shape),
    }
    val_norm = {
        "states": x_normer.transform(val_raw["states"].reshape(-1, STATE_DIM)).reshape(val_raw["states"].shape),
        "inputs": u_normer.transform(val_raw["inputs"].reshape(-1, CONTROL_DIM)).reshape(val_raw["inputs"].shape),
    }
    np.savez_compressed(out_dir / "train_pd_data.npz", **train_raw)
    np.savez_compressed(out_dir / "val_pd_data.npz", **val_raw)

    print("[3/7] Training Shi-Meng DKAC model...")
    dkac_cfg = DKACConfig(
        lift_dim=args.lift_dim,
        hidden=tuple(args.hidden),
        control_hidden=tuple(args.control_hidden),
        control_dim_hat=args.control_dim_hat,
        activation=args.activation,
        bound_lift=args.bound_lift,
        identity_control_bias=not args.no_identity_control_bias,
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
    dkac_model, history, train_info = train_dkac(train_norm, val_norm, dkac_cfg, device, out_dir)
    dkac_rt = DKACRuntime(dkac_model, x_normer, u_normer, device)
    plot_training_history(history)

    print("[4/7] Fitting Koopman-EDMD baseline...")
    edmd_cfg = EdmdConfig(
        n_centers=args.edmd_centers,
        rbf_sigma=args.edmd_sigma,
        ridge=args.edmd_ridge,
        kmeans_seed=args.edmd_seed,
    )
    edmd_pred = fit_full_edmd(train_raw["states"], train_raw["inputs"], x_normer, u_normer, edmd_cfg)
    edmd_rt = EdmdRuntime(edmd_pred)
    print(
        f"      EDMD latent_dim={edmd_pred.latent_dim}, sigma={edmd_pred.sigma:.4g}, "
        f"cond(Gram)={edmd_pred.cond_number:.3e}"
    )

    print("[5/7] Evaluating open-loop model prediction error...")
    pred_dkac = predict_validation_rollouts(dkac_rt, val_raw)
    pred_edmd = predict_validation_rollouts(edmd_rt, val_raw)
    np.savez_compressed(
        out_dir / "validation_model_rollouts.npz",
        dkac_pred=pred_dkac["preds"],
        edmd_pred=pred_edmd["preds"],
        true=pred_dkac["states_true"],
    )
    print(
        f"      model RMSE: DKAC={float(pred_dkac['total_rmse']):.6g}, "
        f"EDMD={float(pred_edmd['total_rmse']):.6g}"
    )

    print("[6/7] Running closed-loop Koopman-space LQR tracking on MuJoCo...")
    ref = build_tracking_reference(
        args.dt,
        args.track_steps,
        args.track_amp_a,
        args.track_amp_b,
        args.track_omega_a,
        args.track_omega_b,
        args.track_phase_b,
    )
    lqr_cfg = LqrTrackConfig(args.track_horizon, args.Qq, args.Qdq, args.R, args.Rd)
    dkac_tracker = KoopmanLqrTracker(dkac_rt.A, dkac_rt.B, dkac_rt.C, lqr_cfg)
    edmd_tracker = KoopmanLqrTracker(edmd_rt.A, edmd_rt.B, edmd_rt.C, lqr_cfg)
    log_dkac = run_closed_loop(name="DKAC-LQR", runtime=dkac_rt, tracker=dkac_tracker, xml=args.xml, dt=args.dt, ref=ref)
    log_edmd = run_closed_loop(name="EDMD-LQR", runtime=edmd_rt, tracker=edmd_tracker, xml=args.xml, dt=args.dt, ref=ref)
    np.savez_compressed(out_dir / "closed_loop_dkac.npz", **log_dkac)
    np.savez_compressed(out_dir / "closed_loop_edmd.npz", **log_edmd)

    print("[7/7] Plotting and saving summary...")
    plot_model_prediction_error(pred_dkac, pred_edmd, args.dt)
    plot_joint_tracking(log_dkac, log_edmd)
    plot_tracking_error(log_dkac, log_edmd)
    plot_tracking_rmse(log_dkac, log_edmd)
    plot_control_inputs(log_dkac, log_edmd)
    plot_cable_tensions(log_dkac, log_edmd)
    metrics = {"DKAC-LQR": tracking_metrics(log_dkac), "EDMD-LQR": tracking_metrics(log_edmd)}
    print(
        f"      RMSE_q: DKAC={metrics['DKAC-LQR']['rmse_q']:.6g}, "
        f"EDMD={metrics['EDMD-LQR']['rmse_q']:.6g}"
    )

    summary = {
        "script": Path(__file__).name,
        "model": "Shi-Meng DKAC z=[x,phi_x(x)], v=G(x)u, z+=Az+Bv",
        "baseline": "Koopman-EDMD RBF z=[x,rbf(x)], z+=Az+Bu",
        "args": vars(args),
        "dkac_config": asdict(dkac_cfg),
        "edmd_config": asdict(edmd_cfg),
        "lqr_config": asdict(lqr_cfg),
        "normalizers": {"x": x_normer.to_json(), "u": u_normer.to_json()},
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
        "train_info": train_info,
        "edmd_info": {
            "latent_dim": edmd_pred.latent_dim,
            "sigma": edmd_pred.sigma,
            "cond_number": edmd_pred.cond_number,
        },
        "model_prediction": {
            "DKAC": model_prediction_metrics(pred_dkac),
            "EDMD": model_prediction_metrics(pred_edmd),
            "protocol": "open-loop rollout on validation MuJoCo trajectories using recorded PD inputs",
        },
        "metrics": metrics,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
