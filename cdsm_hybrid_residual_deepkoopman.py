"""
cdsm_hybrid_residual_deepkoopman.py
===================================
Cable-driven space manipulator (CDSM): nominal rigid model + DeepKoopman
residual dynamics prediction.

Pipeline
--------
1. Collect MuJoCo trajectories from ``multi_joint_cable_dirven_space_robot.xml``
   under PD joint-torque commands mapped to antagonistic cable tensions.
2. Use the rigid nominal model to compute
      r_k = x_{k+1}^{mj} - f_nom(x_k, u_k).
3. Train a controlled DeepKoopman residual model:
      z_k = encoder(x_k)
      z_{k+1} = A z_k + B u_k
      r_hat_k = decoder_residual(z_{k+1})
      x_hat_{k+1} = f_nom(x_k, u_k) + r_hat_k.
4. Compare nominal and hybrid open-loop rollouts against MuJoCo and plot
   dynamic response errors plus RMSE.

Examples
--------
    python cdsm_hybrid_residual_deepkoopman.py
    python cdsm_hybrid_residual_deepkoopman.py --train_traj 20 --val_traj 5 --steps 100 --epochs 30
    python cdsm_hybrid_residual_deepkoopman.py --device cuda --eval_mode both
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise SystemExit(
        "This script requires PyTorch. Install project dependencies."
    ) from exc

from cdsm_hybrid_residual_edmd import (
    PDCollectConfig,
    STATE_LABELS,
    build_residual_dataset,
    collect_pd_trajectories,
    compute_nominal_next,
    flatten_residual_data,
    load_cable_model,
    rollout_nominal,
    set_seed as set_numpy_seed,
)
from cdsm_rigid_nominal_model import CdsmRigidNominalModel, make_nominal_model
from utils_plot import get_save_dir, save_figure

XML_PATH = "multi_joint_cable_dirven_space_robot.xml"


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.std + self.mean

    def to_json(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


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


class ResidualDeepKoopman(nn.Module):
    """
    Controlled DeepKoopman residual model.

    The nonlinear encoder learns observables of x. The time-update itself is
    linear in latent state and normalized control: z+ = A z + B u.
    """

    def __init__(
        self,
        state_dim: int = 4,
        control_dim: int = 2,
        latent_dim: int = 32,
        hidden: Tuple[int, ...] = (128, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.encoder = MLP((state_dim,) + hidden + (latent_dim,), activation)
        self.state_decoder = MLP((latent_dim,) + hidden + (state_dim,), activation)
        self.residual_decoder = MLP((latent_dim,) + hidden + (state_dim,), activation)
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        self._init_linear_dynamics()

    def _init_linear_dynamics(self) -> None:
        with torch.no_grad():
            self.A.weight.zero_()
            eye = torch.eye(self.latent_dim)
            self.A.weight[: self.latent_dim, : self.latent_dim].copy_(0.98 * eye)
            nn.init.xavier_uniform_(self.B.weight, gain=0.1)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_norm)

    def koopman_step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(u_norm)

    def forward(self, x_norm: torch.Tensor, u_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x_norm)
        z_next = self.koopman_step(z, u_norm)
        return {
            "z": z,
            "z_next": z_next,
            "x_rec": self.state_decoder(z),
            "r_pred": self.residual_decoder(z_next),
        }


def set_seed(seed: int) -> None:
    set_numpy_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sample_transition_batch(
    x_norm: np.ndarray,
    u_norm: np.ndarray,
    xp_norm: np.ndarray,
    r_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x_norm.shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    return (
        torch.from_numpy(x_norm[idx]).to(device),
        torch.from_numpy(u_norm[idx]).to(device),
        torch.from_numpy(xp_norm[idx]).to(device),
        torch.from_numpy(r_norm[idx]).to(device),
    )


def compute_losses(
    model: ResidualDeepKoopman,
    x: torch.Tensor,
    u: torch.Tensor,
    xp: torch.Tensor,
    r: torch.Tensor,
    w_residual: float,
    w_recon: float,
    w_linear: float,
    w_l2: float,
) -> Dict[str, torch.Tensor]:
    out = model(x, u)
    z_true_next = model.encode(xp)
    residual_loss = torch.mean((out["r_pred"] - r) ** 2)
    recon_loss = torch.mean((out["x_rec"] - x) ** 2)
    linear_loss = torch.mean((out["z_next"] - z_true_next) ** 2)
    l2 = torch.zeros((), device=x.device)
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


@torch.no_grad()
def evaluate_loss(
    model: ResidualDeepKoopman,
    arrays: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
    n_batches: int = 20,
) -> Dict[str, float]:
    model.eval()
    acc = {"total": 0.0, "residual": 0.0, "recon": 0.0, "linear": 0.0}
    for _ in range(n_batches):
        batch = sample_transition_batch(
            arrays["x"], arrays["u"], arrays["xp"], arrays["r"], batch_size, device
        )
        losses = compute_losses(
            model,
            *batch,
            w_residual=args.w_residual,
            w_recon=args.w_recon,
            w_linear=args.w_linear,
            w_l2=args.w_l2,
        )
        for key in acc:
            acc[key] += float(losses[key].item())
    for key in acc:
        acc[key] /= n_batches
    return acc


@torch.no_grad()
def predict_residual_batch(
    model: ResidualDeepKoopman,
    x_raw: np.ndarray,
    u_raw: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    r_normer: Normalizer,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x_n = torch.from_numpy(x_normer.transform(np.atleast_2d(x_raw))).to(device)
    u_n = torch.from_numpy(u_normer.transform(np.atleast_2d(u_raw))).to(device)
    r_n = model(x_n, u_n)["r_pred"].cpu().numpy()
    return r_normer.inverse(r_n).astype(np.float64)


def predict_hybrid_next(
    model: ResidualDeepKoopman,
    nominal: CdsmRigidNominalModel,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    x_normer: Normalizer,
    u_normer: Normalizer,
    r_normer: Normalizer,
    device: torch.device,
) -> np.ndarray:
    x_nom = compute_nominal_next(nominal, x, u, dt)
    residual = predict_residual_batch(model, x, u, x_normer, u_normer, r_normer, device)[0]
    return x_nom + residual


def rollout_hybrid(
    model: ResidualDeepKoopman,
    nominal: CdsmRigidNominalModel,
    x0: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
    x_normer: Normalizer,
    u_normer: Normalizer,
    r_normer: Normalizer,
    device: torch.device,
) -> np.ndarray:
    traj = np.zeros((u_seq.shape[0] + 1, 4), dtype=np.float64)
    traj[0] = np.asarray(x0, dtype=np.float64).reshape(4)
    x = traj[0].copy()
    for k, u in enumerate(u_seq):
        x = predict_hybrid_next(model, nominal, x, u, dt, x_normer, u_normer, r_normer, device)
        traj[k + 1] = x
    return traj


def metrics_from_errors(err: np.ndarray) -> Dict[str, np.ndarray]:
    rmse_by_state = np.sqrt(np.mean(err * err, axis=tuple(range(err.ndim - 1))))
    mae_by_state = np.mean(np.abs(err), axis=tuple(range(err.ndim - 1)))
    out: Dict[str, np.ndarray] = {
        "rmse_by_state": rmse_by_state,
        "mae_by_state": mae_by_state,
        "total_rmse": np.array([float(np.sqrt(np.mean(err * err)))]),
        "total_mae": np.array([float(np.mean(np.abs(err)))]),
    }
    if err.ndim == 3:
        out["step_rmse"] = np.sqrt(np.mean(err * err, axis=(0, 2)))
        out["step_rmse_by_state"] = np.sqrt(np.mean(err * err, axis=0))
    return out


def evaluate_one_step(
    model: ResidualDeepKoopman,
    res_data: Dict[str, np.ndarray],
    nominal: CdsmRigidNominalModel,
    dt: float,
    x_normer: Normalizer,
    u_normer: Normalizer,
    r_normer: Normalizer,
    device: torch.device,
) -> Dict[str, object]:
    """
    One-step prediction along each validation trajectory (teacher forcing on MuJoCo x_k).

    At step k uses (states[i, k], inputs[i, k]) to predict k+1; initial time matches MuJoCo.
    Output layout matches :func:`evaluate_rollout` for shared plotting.
    """
    states = res_data["states"]
    inputs = res_data["inputs"]
    n_traj, n_times, _ = states.shape
    n_step = inputs.shape[1]
    if n_step != n_times - 1:
        raise ValueError(f"Expected inputs.shape[1] == states.shape[1] - 1, got {n_step} vs {n_times - 1}")

    pred_nom = np.zeros_like(states)
    pred_hyb = np.zeros_like(states)
    pred_nom[:, 0] = states[:, 0]
    pred_hyb[:, 0] = states[:, 0]
    for i in range(n_traj):
        for k in range(n_step):
            x_k = states[i, k]
            u_k = inputs[i, k]
            pred_nom[i, k + 1] = compute_nominal_next(nominal, x_k, u_k, dt)
            pred_hyb[i, k + 1] = predict_hybrid_next(
                model, nominal, x_k, u_k, dt, x_normer, u_normer, r_normer, device
            )
    return {
        "metrics": {
            "nominal": metrics_from_errors(pred_nom - states),
            "hybrid": metrics_from_errors(pred_hyb - states),
        },
        "pred_nominal": pred_nom,
        "pred_hybrid": pred_hyb,
        "states_true": states,
    }


def evaluate_rollout(
    model: ResidualDeepKoopman,
    res_data: Dict[str, np.ndarray],
    nominal: CdsmRigidNominalModel,
    dt: float,
    x_normer: Normalizer,
    u_normer: Normalizer,
    r_normer: Normalizer,
    device: torch.device,
) -> Dict[str, object]:
    states = res_data["states"]
    inputs = res_data["inputs"]
    n_traj = states.shape[0]
    pred_nom = np.zeros_like(states)
    pred_hyb = np.zeros_like(states)
    for i in range(n_traj):
        pred_nom[i] = rollout_nominal(nominal, states[i, 0], inputs[i], dt)
        pred_hyb[i] = rollout_hybrid(
            model, nominal, states[i, 0], inputs[i], dt, x_normer, u_normer, r_normer, device
        )
    return {
        "metrics": {
            "nominal": metrics_from_errors(pred_nom - states),
            "hybrid": metrics_from_errors(pred_hyb - states),
        },
        "pred_nominal": pred_nom,
        "pred_hybrid": pred_hyb,
        "states_true": states,
    }


def plot_training_history(history: List[List[float]]) -> None:
    arr = np.asarray(history, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    panels = [
        ("Total loss", 1, 5),
        ("Residual prediction", 2, 6),
        ("State reconstruction", 3, 7),
        ("Latent linearity", 4, 8),
    ]
    for ax, (title, tr_col, va_col) in zip(axes.ravel(), panels):
        ax.semilogy(arr[:, 0], arr[:, tr_col], label="train")
        ax.semilogy(arr[:, 0], arr[:, va_col], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Residual DeepKoopman training")
    fig.tight_layout()
    save_figure("training_history")
    plt.close(fig)


def plot_response_error_curve(
    rollout_result: Dict[str, object],
    dt: float,
    traj_idx: int,
    out_name: str,
    *,
    eval_label: str = "Open-loop rollout",
) -> None:
    true_traj = rollout_result["states_true"][traj_idx]
    pred_hyb = rollout_result["pred_hybrid"][traj_idx]
    err = pred_hyb - true_traj
    rmse_state = np.sqrt(np.mean(err * err, axis=0))
    total_rmse = float(np.sqrt(np.mean(err * err)))
    t = np.arange(true_traj.shape[0]) * dt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, label in enumerate(STATE_LABELS):
        axes[0].plot(t, err[:, i], lw=1.4, label=f"{label}, RMSE={rmse_state[i]:.3g}")
    axes[0].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("Hybrid - MuJoCo")
    axes[0].set_title(
        f"{eval_label} error, trajectory {traj_idx}; total RMSE={total_rmse:.6g}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(t, np.sqrt(np.mean(err * err, axis=1)), "C3", lw=1.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Instantaneous state RMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def plot_rollout_rmse_growth(
    metrics: Dict[str, Dict[str, np.ndarray]],
    dt: float,
    out_name: str,
    *,
    eval_label: str = "Open-loop rollout",
) -> None:
    t = np.arange(metrics["nominal"]["step_rmse"].shape[0]) * dt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, metrics["nominal"]["step_rmse"], lw=1.8, label="Nominal")
    ax.plot(t, metrics["hybrid"]["step_rmse"], lw=1.8, label="Nominal + DeepKoopman residual")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMSE over states")
    ax.set_title(f"{eval_label} RMSE vs MuJoCo")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def plot_rollout_state_response(
    rollout_result: Dict[str, object],
    dt: float,
    traj_idx: int,
    out_name: str,
    *,
    eval_label: str = "Open-loop rollout",
) -> None:
    states = rollout_result["states_true"][traj_idx]
    pred_nom = rollout_result["pred_nominal"][traj_idx]
    pred_hyb = rollout_result["pred_hybrid"][traj_idx]
    t = np.arange(states.shape[0]) * dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for i, label in enumerate(STATE_LABELS):
        ax = axes[i // 2, i % 2]
        ax.plot(t, states[:, i], "k-", lw=1.8, label="MuJoCo")
        ax.plot(t, pred_nom[:, i], "--", lw=1.2, label="Nominal")
        ax.plot(t, pred_hyb[:, i], "-.", lw=1.3, label="Hybrid DK")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"{eval_label}: MuJoCo vs nominal vs hybrid DeepKoopman, trajectory {traj_idx}"
    )
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def plot_eval_response_suite(
    eval_result: Dict[str, object],
    metrics: Dict[str, Dict[str, np.ndarray]],
    dt: float,
    demo_idx: int,
    file_prefix: str,
    *,
    eval_label: str,
) -> None:
    """RMSE growth, state response, and hybrid error curves (shared by one-step and rollout)."""
    plot_rollout_rmse_growth(
        metrics, dt, f"{file_prefix}_rmse_growth", eval_label=eval_label
    )
    plot_rollout_state_response(
        eval_result, dt, demo_idx, f"{file_prefix}_dynamic_response", eval_label=eval_label
    )
    plot_response_error_curve(
        eval_result,
        dt,
        demo_idx,
        f"{file_prefix}_hybrid_vs_mujoco_response_error_rmse",
        eval_label=eval_label,
    )


def metrics_to_json(metrics: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for name, m in metrics.items():
        entry: Dict[str, object] = {
            "total_rmse": float(m["total_rmse"][0]),
            "total_mae": float(m["total_mae"][0]),
            "rmse_by_state": m["rmse_by_state"].tolist(),
            "mae_by_state": m["mae_by_state"].tolist(),
        }
        if "step_rmse" in m:
            entry["step_rmse"] = m["step_rmse"].tolist()
        payload[name] = entry
    nom_rmse = float(metrics["nominal"]["total_rmse"][0])
    hyb_rmse = float(metrics["hybrid"]["total_rmse"][0])
    payload["improvement_ratio"] = (nom_rmse - hyb_rmse) / nom_rmse if nom_rmse > 0 else 0.0
    return payload


def print_eval_metrics(label: str, metrics: Dict[str, Dict[str, np.ndarray]]) -> None:
    nom_rmse = float(metrics["nominal"]["total_rmse"][0])
    hyb_rmse = float(metrics["hybrid"]["total_rmse"][0])
    improve = (nom_rmse - hyb_rmse) / nom_rmse if nom_rmse > 0 else 0.0
    print(f"  [{label}] RMSE nominal : {nom_rmse:.6g}")
    print(f"  [{label}] RMSE hybrid  : {hyb_rmse:.6g}")
    print(f"  [{label}] improvement  : {100.0 * improve:.1f}%")
    for i, lab in enumerate(STATE_LABELS):
        print(
            f"    {lab:5s}  nom={metrics['nominal']['rmse_by_state'][i]:.6g}  "
            f"hyb={metrics['hybrid']['rmse_by_state'][i]:.6g}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CDSM hybrid residual DeepKoopman model.")

    # ------------------------------------------------------------------
    # 数据与仿真
    # ------------------------------------------------------------------
    # MuJoCo 绳驱模型 XML 路径
    p.add_argument("--xml", default=XML_PATH)
    # 训练集 / 验证集轨迹条数
    p.add_argument("--train_traj", type=int, default=120)
    p.add_argument("--val_traj", type=int, default=24)
    # 每条轨迹仿真步数 (状态序列长度 = steps + 1)
    p.add_argument("--steps", type=int, default=500)
    # 积分步长 (s)；须与 XML 默认 timestep 及名义模型一致
    p.add_argument("--dt", type=float, default=0.01)
    # 随机种子 (NumPy / PyTorch / 采数)
    p.add_argument("--seed", type=int, default=20)
    # 训练设备: auto=有 CUDA 则用 GPU; cpu / cuda
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")

    # ------------------------------------------------------------------
    # PD 采数 (多正弦参考 + 拮抗绳驱映射生成 tau_a, tau_b)
    # ------------------------------------------------------------------
    # 初始关节角 / 角速度随机范围 (rad, rad/s)
    p.add_argument("--q_init_range", type=float, default=1.3)
    p.add_argument("--dq_init_range", type=float, default=1.2)
    # 正弦参考幅值与角频率随机范围
    p.add_argument("--amp_min", type=float, default=-1.5)
    p.add_argument("--amp_max", type=float, default=1.5)
    p.add_argument("--omega_min", type=float, default=-1.2)
    p.add_argument("--omega_max", type=float, default=1.2)
    # PD 增益：关节 a (第一级) / 关节 b (第二级)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)
    # 关节力矩限幅 (Nm)；已关闭，代码中 tau_max=float("inf")
    # p.add_argument("--tau_max", type=float, default=45.0)

    # ------------------------------------------------------------------
    # ResidualDeepKoopman 网络结构
    # z = encoder(x), z+ = A z + B u, r_hat = decoder_residual(z+)
    # ------------------------------------------------------------------
    # 潜变量维度；越大表达能力越强，需更多数据
    p.add_argument("--latent_dim", type=int, default=32)
    # 编码器/解码器 MLP 各隐层宽度，可多个整数如 --hidden 128 256 128
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    # 隐藏层激活函数
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")

    # ------------------------------------------------------------------
    # 训练超参数
    # ------------------------------------------------------------------
    # 训练轮数
    p.add_argument("--epochs", type=int, default=80)
    # 每轮随机小批量更新次数
    p.add_argument("--steps_per_epoch", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=256)
    # AdamW 学习率
    p.add_argument("--lr", type=float, default=1e-3)
    # 梯度范数裁剪上限；<=0 表示不裁剪
    p.add_argument("--grad_clip", type=float, default=5.0)
    # 损失权重: 残差预测 ||r_hat-r||^2
    p.add_argument("--w_residual", type=float, default=1.0)
    # 状态重构 ||x_rec-x||^2
    p.add_argument("--w_recon", type=float, default=0.05)
    # 潜空间线性一致性 ||z_next - encode(x')||^2
    p.add_argument("--w_linear", type=float, default=0.2)
    # 权重 L2 正则 (仅二维及以上参数)
    p.add_argument("--w_l2", type=float, default=1e-8)

    # ------------------------------------------------------------------
    # 评估与出图
    # ------------------------------------------------------------------
    # 演示轨迹在验证集中的索引 (用于画单条轨迹响应图)
    p.add_argument("--demo_traj", type=int, default=0)
    # one_step=每步用真值 x_k 预测; rollout=开环滚动; both=两种都评
    p.add_argument("--eval_mode", choices=["one_step", "rollout", "both"], default="rollout")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== CDSM hybrid residual DeepKoopman ===")
    print(f"device={device}, output={out_dir}")

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
        tau_max=float("inf"),
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
        tau_max=float("inf"),
    )

    print("[1/5] Collecting MuJoCo cable-driven trajectories...")
    mj_model, mj_data, scratch, indices = load_cable_model(args.xml, args.dt)
    train_raw, train_meta = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_cfg_train)
    val_raw, val_meta = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_cfg_val)
    np.savez(out_dir / "dataset_train.npz", **train_raw)
    np.savez(out_dir / "dataset_val.npz", **val_raw)
    print(f"      train states={train_raw['states'].shape}, val states={val_raw['states'].shape}")

    print("[2/5] Computing nominal-model residuals...")
    nominal = make_nominal_model(dt=args.dt)
    res_train = build_residual_dataset(train_raw, nominal, args.dt)
    res_val = build_residual_dataset(val_raw, nominal, args.dt)
    x_tr, u_tr, xp_tr, r_tr = flatten_residual_data(res_train)
    x_va, u_va, xp_va, _r_va = flatten_residual_data(res_val)

    x_normer = Normalizer.fit(x_tr)
    u_normer = Normalizer.fit(u_tr)
    r_normer = Normalizer.fit(r_tr)
    train_arrays = {
        "x": x_normer.transform(x_tr),
        "u": u_normer.transform(u_tr),
        "xp": x_normer.transform(xp_tr),
        "r": r_normer.transform(r_tr),
    }
    val_arrays = {
        "x": x_normer.transform(x_va),
        "u": u_normer.transform(u_va),
        "xp": x_normer.transform(xp_va),
        "r": r_normer.transform(_r_va),
    }
    print(f"      transitions={x_tr.shape[0]}, mean |residual|={np.linalg.norm(r_tr, axis=1).mean():.6g}")

    print("[3/5] Training controlled residual DeepKoopman...")
    dk_model = ResidualDeepKoopman(
        state_dim=4,
        control_dim=2,
        latent_dim=args.latent_dim,
        hidden=tuple(args.hidden),
        activation=args.activation,
    ).to(device)
    optimizer = optim.AdamW(dk_model.parameters(), lr=args.lr, weight_decay=0.0)
    best_val = float("inf")
    history: List[List[float]] = []

    for epoch in range(1, args.epochs + 1):
        dk_model.train()
        acc = {"total": 0.0, "residual": 0.0, "recon": 0.0, "linear": 0.0}
        for _ in range(args.steps_per_epoch):
            batch = sample_transition_batch(
                train_arrays["x"],
                train_arrays["u"],
                train_arrays["xp"],
                train_arrays["r"],
                args.batch_size,
                device,
            )
            losses = compute_losses(
                dk_model,
                *batch,
                w_residual=args.w_residual,
                w_recon=args.w_recon,
                w_linear=args.w_linear,
                w_l2=args.w_l2,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(dk_model.parameters(), args.grad_clip)
            optimizer.step()
            for key in acc:
                acc[key] += float(losses[key].detach().item())
        for key in acc:
            acc[key] /= args.steps_per_epoch
        val_loss = evaluate_loss(dk_model, val_arrays, args.batch_size, device, args)
        history.append(
            [
                epoch,
                acc["total"],
                acc["residual"],
                acc["recon"],
                acc["linear"],
                val_loss["total"],
                val_loss["residual"],
                val_loss["recon"],
                val_loss["linear"],
            ]
        )
        print(
            f"  [Ep {epoch:03d}] tr={acc['total']:.4e} "
            f"res={acc['residual']:.4e} lin={acc['linear']:.4e} | "
            f"val={val_loss['total']:.4e} val_res={val_loss['residual']:.4e}"
        )
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save(
                {
                    "model_state": dk_model.state_dict(),
                    "config": vars(args),
                    "x_normer": x_normer.to_json(),
                    "u_normer": u_normer.to_json(),
                    "r_normer": r_normer.to_json(),
                    "best_val": best_val,
                    "epoch": epoch,
                },
                out_dir / "best_residual_deepkoopman.pt",
            )

    np.savetxt(
        out_dir / "training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_residual,train_recon,train_linear,val_total,val_residual,val_recon,val_linear",
        comments="",
    )
    plot_training_history(history)

    print("[4/5] Loading best checkpoint and evaluating...")
    ckpt = torch.load(out_dir / "best_residual_deepkoopman.pt", map_location=device, weights_only=False)
    dk_model.load_state_dict(ckpt["model_state"])
    dk_model.eval()

    eval_modes = ["one_step", "rollout"] if args.eval_mode == "both" else [args.eval_mode]
    demo_idx = min(max(args.demo_traj, 0), max(args.val_traj - 1, 0))
    summary: Dict[str, object] = {
        "xml": args.xml,
        "dt": args.dt,
        "eval_mode": args.eval_mode,
        "model": {
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "activation": args.activation,
            "best_val": best_val,
        },
        "normalization": {
            "x": x_normer.to_json(),
            "u": u_normer.to_json(),
            "r": r_normer.to_json(),
        },
        "collection_meta": {
            "train": {**asdict(pd_cfg_train), "meta": train_meta},
            "val": {**asdict(pd_cfg_val), "meta": val_meta},
        },
    }

    if "one_step" in eval_modes:
        one_step_res = evaluate_one_step(
            dk_model, res_val, nominal, args.dt, x_normer, u_normer, r_normer, device
        )
        metrics = one_step_res["metrics"]
        summary["one_step"] = metrics_to_json(metrics)
        print_eval_metrics("one-step", metrics)
        plot_eval_response_suite(
            one_step_res,
            metrics,
            args.dt,
            demo_idx,
            "one_step",
            eval_label="One-step prediction",
        )

    if "rollout" in eval_modes:
        rollout_res = evaluate_rollout(
            dk_model, res_val, nominal, args.dt, x_normer, u_normer, r_normer, device
        )
        metrics = rollout_res["metrics"]
        summary["rollout"] = metrics_to_json(metrics)
        print_eval_metrics("rollout", metrics)
        plot_eval_response_suite(
            rollout_res,
            metrics,
            args.dt,
            demo_idx,
            "rollout",
            eval_label="Open-loop rollout",
        )

    print("[5/5] Saving summary...")
    summary["elapsed_sec"] = time.time() - t0
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
