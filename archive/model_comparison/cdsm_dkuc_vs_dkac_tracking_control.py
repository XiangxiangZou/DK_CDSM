"""
cdsm_dkuc_vs_dkac_tracking_control.py
=====================================

Compare Shi & Meng style DKUC and DKAC on the MuJoCo cable-driven space
manipulator:

    DKUC: z = [x_n, phi_x(x_n)],        z_next = A z + B u_n
    DKAC: z = [x_n, phi_x(x_n)], v=G(x)u_n, z_next = A z + B v

Both methods use the same broad-range PD data collection protocol and the same
finite-horizon unconstrained LQR tracker in Koopman space. The script also
plots open-loop model rollout error against the MuJoCo validation trajectories,
so model error can be separated from closed-loop tracking error.

Torque clipping and cable maximum-tension clipping are intentionally disabled.
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
    MLP,
    Normalizer,
    PDCollectConfig,
    build_windows,
    collect_pd_trajectories,
    load_cable_model,
    make_device,
    sample_window_batch,
    set_seed,
)
from cdsm_dkac_vs_edmd_tracking_control import (
    DKACConfig,
    DKACRuntime,
    KoopmanLqrTracker,
    LqrTrackConfig,
    build_tracking_reference,
    model_prediction_metrics,
    predict_validation_rollouts,
    run_closed_loop,
    tracking_metrics,
    train_dkac,
)
from utils_plot import get_save_dir, save_figure


@dataclass
class DKUCConfig:
    lift_dim: int
    hidden: Tuple[int, ...]
    activation: str
    bound_lift: float
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


class DKUCModel(nn.Module):
    """Deep Koopman with unchanged control: z_next = A z + B u."""

    def __init__(
        self,
        lift_dim: int,
        hidden: Tuple[int, ...],
        activation: str,
        bound_lift: float,
    ) -> None:
        super().__init__()
        self.lift_dim = int(lift_dim)
        self.latent_dim = STATE_DIM + self.lift_dim
        self.encoder = MLP((STATE_DIM,) + tuple(hidden) + (self.lift_dim,), activation)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(CONTROL_DIM, self.latent_dim, bias=False)
        self.bound_lift = float(bound_lift)
        self._init_linear()

    def _init_linear(self) -> None:
        with torch.no_grad():
            self.A.weight.zero_()
            self.A.weight += torch.eye(self.latent_dim)
            self.B.weight.zero_()
            rows = min(STATE_DIM, CONTROL_DIM)
            self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

    def lift(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_norm)
        if self.bound_lift > 0.0:
            h = self.bound_lift * torch.tanh(h / self.bound_lift)
        return torch.cat([x_norm, h], dim=-1)

    @staticmethod
    def state_from_latent(z: torch.Tensor) -> torch.Tensor:
        return z[..., :STATE_DIM]

    def step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(u_norm)


def compute_dkuc_losses(
    model: DKUCModel,
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


def train_dkuc(
    train_norm: Dict[str, np.ndarray],
    val_norm: Dict[str, np.ndarray],
    cfg: DKUCConfig,
    device: torch.device,
    out_dir: Path,
) -> Tuple[DKUCModel, List[List[float]], Dict[str, float]]:
    Xw_train, Uw_train = build_windows(train_norm["states"], train_norm["inputs"], cfg.window)
    Xw_val, Uw_val = build_windows(val_norm["states"], val_norm["inputs"], cfg.window)

    model = DKUCModel(
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        activation=cfg.activation,
        bound_lift=cfg.bound_lift,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_full = float("inf")
    best_path = out_dir / "best_dkuc.pt"
    history: List[List[float]] = []
    val_x = torch.from_numpy(Xw_val.astype(np.float32)).to(device)
    val_u = torch.from_numpy(Uw_val.astype(np.float32)).to(device)

    for epoch in range(1, cfg.epochs + 1):
        h = min(
            cfg.window,
            max(1, cfg.window_start + (epoch - 1) * (cfg.window - cfg.window_start) // max(1, cfg.epochs - 1)),
        )
        model.train()
        losses = []
        for _ in range(cfg.steps_per_epoch):
            xb, ub = sample_window_batch(Xw_train[:, : h + 1], Uw_train[:, :h], cfg.batch_size, device)
            opt.zero_grad(set_to_none=True)
            loss, state_loss, embed_loss = compute_dkuc_losses(model, xb, ub, cfg.w_state, cfg.w_embed)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append([float(loss.detach().cpu()), float(state_loss.cpu()), float(embed_loss.cpu())])

        model.eval()
        with torch.no_grad():
            val_total, val_state, val_embed = compute_dkuc_losses(
                model, val_x[:, : h + 1], val_u[:, :h], cfg.w_state, cfg.w_embed
            )
            val_full_total, val_full_state, val_full_embed = compute_dkuc_losses(
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
                f"[dkuc] epoch {epoch:03d}/{cfg.epochs:03d} H={h:02d} "
                f"train={row[1]:.3e} valH={row[4]:.3e} valFull={row[8]:.3e}",
                flush=True,
            )

    try:
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(state_dict)
    np.savetxt(
        out_dir / "dkuc_training_history.csv",
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


class DKUCRuntime:
    def __init__(
        self,
        model: DKUCModel,
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
        self.control_dim_hat = CONTROL_DIM

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_n = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            z = self.model.lift(torch.from_numpy(x_n).to(self.device)).cpu().numpy()[0]
        return z.astype(np.float64)

    def recover_u_norm(self, _x_phys: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=np.float64).reshape(CONTROL_DIM)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray) -> np.ndarray:
        u_norm = self.u_normer.transform(np.asarray(u_phys, dtype=np.float64).reshape(1, -1))[0]
        return self.A @ np.asarray(z, dtype=np.float64).reshape(-1) + self.B @ u_norm

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[:STATE_DIM]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]


def plot_training_histories(history_dkuc: List[List[float]], history_dkac: List[List[float]]) -> None:
    arr_u = np.asarray(history_dkuc, dtype=np.float64)
    arr_a = np.asarray(history_dkac, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, arr, title in [(axes[0], arr_u, "DKUC"), (axes[1], arr_a, "DKAC")]:
        ax.semilogy(arr[:, 0], arr[:, 1], label="train")
        val_label = "val curriculum" if arr.shape[1] > 8 else "val"
        ax.semilogy(arr[:, 0], arr[:, 4], label=val_label)
        if arr.shape[1] > 8:
            ax.semilogy(arr[:, 0], arr[:, 8], label="val full-window")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training histories")
    fig.tight_layout()
    save_figure("training_history_compare")
    plt.close(fig)


def plot_model_prediction_error_compare(
    res_dkuc: Dict[str, object],
    res_dkac: Dict[str, object],
    dt: float,
) -> None:
    t = np.arange(np.asarray(res_dkuc["step_rmse"]).shape[0]) * dt
    step_u = np.asarray(res_dkuc["step_rmse"], dtype=np.float64)
    step_a = np.asarray(res_dkac["step_rmse"], dtype=np.float64)
    rmse_u = np.asarray(res_dkuc["rmse_by_state"], dtype=np.float64)
    rmse_a = np.asarray(res_dkac["rmse_by_state"], dtype=np.float64)
    labels = STATE_LABELS + ["overall"]
    vals_u = np.r_[rmse_u, float(res_dkuc["total_rmse"])]
    vals_a = np.r_[rmse_a, float(res_dkac["total_rmse"])]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2))
    axes[0].plot(t, step_u, lw=1.7, label=f"DKUC model (RMSE={float(res_dkuc['total_rmse']):.3g})")
    axes[0].plot(t, step_a, lw=1.7, label=f"DKAC model (RMSE={float(res_dkac['total_rmse']):.3g})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("State RMSE")
    axes[0].set_title("Open-loop rollout error on validation MuJoCo trajectories")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, vals_u, width, label="DKUC model")
    axes[1].bar(x + width / 2, vals_a, width, label="DKAC model")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("RMSE in physical state coordinates")
    axes[1].set_title("Validation rollout RMSE by state")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Model prediction error: DKUC vs DKAC vs MuJoCo")
    fig.tight_layout()
    save_figure("model_prediction_error_compare")
    plt.close(fig)


def plot_joint_tracking_compare(log_dkuc: Dict[str, np.ndarray], log_dkac: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, label in enumerate(["q_a", "q_b"]):
        axes[j].plot(log_dkuc["t"], log_dkuc["q_ref"][:, j], "k--", lw=1.6, label="reference")
        axes[j].plot(log_dkuc["t"], log_dkuc["x"][:, j], lw=1.4, label="DKUC-LQR")
        axes[j].plot(log_dkac["t"], log_dkac["x"][:, j], lw=1.4, label="DKAC-LQR")
        axes[j].set_ylabel(f"{label} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Closed-loop joint tracking on MuJoCo")
    fig.tight_layout()
    save_figure("joint_tracking_compare")
    plt.close(fig)


def plot_tracking_error_compare(log_dkuc: Dict[str, np.ndarray], log_dkac: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, label in enumerate(["q_a", "q_b"]):
        e_u = log_dkuc["x"][:, j] - log_dkuc["q_ref"][:, j]
        e_a = log_dkac["x"][:, j] - log_dkac["q_ref"][:, j]
        axes[j].plot(log_dkuc["t"], e_u, lw=1.4, label=f"DKUC RMSE={np.sqrt(np.mean(e_u * e_u)):.3g}")
        axes[j].plot(log_dkac["t"], e_a, lw=1.4, label=f"DKAC RMSE={np.sqrt(np.mean(e_a * e_a)):.3g}")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"e_{label} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint tracking error")
    fig.tight_layout()
    save_figure("tracking_error_compare")
    plt.close(fig)


def plot_tracking_rmse_compare(log_dkuc: Dict[str, np.ndarray], log_dkac: Dict[str, np.ndarray]) -> None:
    e_u = log_dkuc["x"][:, :2] - log_dkuc["q_ref"]
    e_a = log_dkac["x"][:, :2] - log_dkac["q_ref"]
    vals_u = np.r_[np.sqrt(np.mean(e_u * e_u, axis=0)), np.sqrt(np.mean(e_u * e_u))]
    vals_a = np.r_[np.sqrt(np.mean(e_a * e_a, axis=0)), np.sqrt(np.mean(e_a * e_a))]
    labels = ["q_a", "q_b", "overall"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, vals_u, width, label="DKUC-LQR")
    ax.bar(x + width / 2, vals_a, width, label="DKAC-LQR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (rad)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.suptitle("Tracking RMSE comparison")
    fig.tight_layout()
    save_figure("tracking_rmse_compare")
    plt.close(fig)


def plot_control_inputs_compare(log_dkuc: Dict[str, np.ndarray], log_dkac: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, label in enumerate(["tau_a", "tau_b"]):
        axes[j].plot(log_dkuc["t"], log_dkuc["u"][:, j], lw=1.3, label="DKUC-LQR")
        axes[j].plot(log_dkac["t"], log_dkac["u"][:, j], lw=1.3, label="DKAC-LQR")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"{label} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint torque commands (no torque clipping)")
    fig.tight_layout()
    save_figure("joint_torque_compare")
    plt.close(fig)


def plot_cable_tensions_compare(log_dkuc: Dict[str, np.ndarray], log_dkac: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, log, title in [
        (axes[0], log_dkuc, "DKUC-LQR cable tensions"),
        (axes[1], log_dkac, "DKAC-LQR cable tensions"),
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
    p = argparse.ArgumentParser(description="CDSM DKUC vs DKAC tracking control comparison.")
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=72)
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
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--bound_lift", type=float, default=10.0)
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

    p.add_argument("--control_hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--control_dim_hat", type=int, default=CONTROL_DIM)
    p.add_argument("--no_identity_control_bias", action="store_true")

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
    print("=== CDSM DKUC vs DKAC tracking control ===")
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
    x_normer = Normalizer.fit(train_raw["states"].reshape(-1, STATE_DIM))
    u_normer = Normalizer.fit(train_raw["inputs"].reshape(-1, CONTROL_DIM))
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

    print("[3/7] Training DKUC model...")
    dkuc_cfg = DKUCConfig(
        lift_dim=args.lift_dim,
        hidden=tuple(args.hidden),
        activation=args.activation,
        bound_lift=args.bound_lift,
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
    dkuc_model, hist_dkuc, train_info_dkuc = train_dkuc(train_norm, val_norm, dkuc_cfg, device, out_dir)
    dkuc_rt = DKUCRuntime(dkuc_model, x_normer, u_normer, device)

    print("[4/7] Training DKAC model...")
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
    dkac_model, hist_dkac, train_info_dkac = train_dkac(train_norm, val_norm, dkac_cfg, device, out_dir)
    dkac_rt = DKACRuntime(dkac_model, x_normer, u_normer, device)
    plot_training_histories(hist_dkuc, hist_dkac)

    print("[5/7] Evaluating open-loop model prediction error...")
    pred_dkuc = predict_validation_rollouts(dkuc_rt, val_raw)
    pred_dkac = predict_validation_rollouts(dkac_rt, val_raw)
    np.savez_compressed(
        out_dir / "validation_model_rollouts.npz",
        dkuc_pred=pred_dkuc["preds"],
        dkac_pred=pred_dkac["preds"],
        true=pred_dkuc["states_true"],
    )
    print(
        f"      model RMSE: DKUC={float(pred_dkuc['total_rmse']):.6g}, "
        f"DKAC={float(pred_dkac['total_rmse']):.6g}"
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
    dkuc_tracker = KoopmanLqrTracker(dkuc_rt.A, dkuc_rt.B, dkuc_rt.C, lqr_cfg)
    dkac_tracker = KoopmanLqrTracker(dkac_rt.A, dkac_rt.B, dkac_rt.C, lqr_cfg)
    log_dkuc = run_closed_loop(name="DKUC-LQR", runtime=dkuc_rt, tracker=dkuc_tracker, xml=args.xml, dt=args.dt, ref=ref)
    log_dkac = run_closed_loop(name="DKAC-LQR", runtime=dkac_rt, tracker=dkac_tracker, xml=args.xml, dt=args.dt, ref=ref)
    np.savez_compressed(out_dir / "closed_loop_dkuc.npz", **log_dkuc)
    np.savez_compressed(out_dir / "closed_loop_dkac.npz", **log_dkac)

    print("[7/7] Plotting and saving summary...")
    plot_model_prediction_error_compare(pred_dkuc, pred_dkac, args.dt)
    plot_joint_tracking_compare(log_dkuc, log_dkac)
    plot_tracking_error_compare(log_dkuc, log_dkac)
    plot_tracking_rmse_compare(log_dkuc, log_dkac)
    plot_control_inputs_compare(log_dkuc, log_dkac)
    plot_cable_tensions_compare(log_dkuc, log_dkac)
    metrics = {"DKUC-LQR": tracking_metrics(log_dkuc), "DKAC-LQR": tracking_metrics(log_dkac)}
    print(
        f"      tracking RMSE_q: DKUC={metrics['DKUC-LQR']['rmse_q']:.6g}, "
        f"DKAC={metrics['DKAC-LQR']['rmse_q']:.6g}"
    )

    summary = {
        "script": Path(__file__).name,
        "models": {
            "DKUC": "z=[x,phi_x(x)], z_next=A z+B u",
            "DKAC": "z=[x,phi_x(x)], v=G(x)u, z_next=A z+B v",
        },
        "args": vars(args),
        "dkuc_config": asdict(dkuc_cfg),
        "dkac_config": asdict(dkac_cfg),
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
        "train_info": {"DKUC": train_info_dkuc, "DKAC": train_info_dkac},
        "model_prediction": {
            "DKUC": model_prediction_metrics(pred_dkuc),
            "DKAC": model_prediction_metrics(pred_dkac),
            "protocol": "open-loop rollout on validation MuJoCo trajectories using recorded PD inputs",
        },
        "tracking": metrics,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
