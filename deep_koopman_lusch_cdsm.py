"""
Lusch 2018 style Deep Koopman training for CDSM model.

This script reproduces the core idea of:
Deep learning for universal linear embeddings of nonlinear dynamics (Lusch et al., 2018)
on the local cable-driven space manipulator model in this repository.

Key components:
1) Encoder / Decoder autoencoder on system state x.
2) Auxiliary omega networks to generate local spectral parameters from latent amplitudes.
3) Varying Koopman rollout in latent space using complex-conjugate blocks + real modes.
4) Multi-term loss: reconstruction + multi-step prediction + latent linearity consistency.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_joint_cdsm_model import MultiJointSpaceRobot
from utils_plot import save_figure, get_save_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, widths: Tuple[int, ...], activation: str = "elu") -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("MLP widths should contain at least input and output.")
        act_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i != len(widths) - 2:
                layers.append(act_map[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LuschKoopmanCDSM(nn.Module):
    """
    Lusch-like model:
      x -> encoder -> y
      y, omega(y) -> varying linear step -> y_next -> decoder -> x_next
    """

    def __init__(
        self,
        state_dim: int,
        encoder_hidden: Tuple[int, ...],
        decoder_hidden: Tuple[int, ...],
        omega_hidden: Tuple[int, ...],
        num_complex_pairs: int,
        num_real: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_complex_pairs = num_complex_pairs
        self.num_real = num_real
        self.latent_dim = 2 * num_complex_pairs + num_real
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        enc_widths = (state_dim,) + encoder_hidden + (self.latent_dim,)
        dec_widths = (self.latent_dim,) + decoder_hidden + (state_dim,)
        self.encoder = MLP(enc_widths, activation=activation)
        self.decoder = MLP(dec_widths, activation=activation)

        # One auxiliary net per eig-block as in Lusch 2018 code.
        self.omega_complex_nets = nn.ModuleList()
        for _ in range(num_complex_pairs):
            widths = (1,) + omega_hidden + (2,)  # [omega, mu]
            self.omega_complex_nets.append(MLP(widths, activation=activation))

        self.omega_real_nets = nn.ModuleList()
        for _ in range(num_real):
            widths = (1,) + omega_hidden + (1,)  # [mu]
            self.omega_real_nets.append(MLP(widths, activation=activation))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(y)

    def omega_from_latent(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        # y: [B, latent_dim]
        complex_omegas = []
        real_omegas = []

        for j in range(self.num_complex_pairs):
            idx = 2 * j
            pair = y[:, idx : idx + 2]
            radius = torch.sum(pair * pair, dim=1, keepdim=True)  # [B,1]
            complex_omegas.append(self.omega_complex_nets[j](radius))  # [B,2]

        for j in range(self.num_real):
            idx = 2 * self.num_complex_pairs + j
            one = y[:, idx : idx + 1]
            real_omegas.append(self.omega_real_nets[j](one))  # [B,1]

        return {"complex": complex_omegas, "real": real_omegas}

    def varying_step(self, y: torch.Tensor, dt: float) -> torch.Tensor:
        # Apply varying linear operator blockwise, matching Lusch's idea.
        omegas = self.omega_from_latent(y)
        out_parts = []

        # Complex conjugate 2x2 blocks.
        for j in range(self.num_complex_pairs):
            idx = 2 * j
            ypair = y[:, idx : idx + 2]
            omega = omegas["complex"][j][:, 0:1]
            mu = omegas["complex"][j][:, 1:2]
            scale = torch.exp(mu * dt)
            c = torch.cos(omega * dt)
            s = torch.sin(omega * dt)
            y0 = ypair[:, 0:1]
            y1 = ypair[:, 1:2]
            n0 = scale * (c * y0 - s * y1)
            n1 = scale * (s * y0 + c * y1)
            out_parts.append(torch.cat([n0, n1], dim=1))

        # Real blocks.
        for j in range(self.num_real):
            idx = 2 * self.num_complex_pairs + j
            yj = y[:, idx : idx + 1]
            muj = omegas["real"][j]
            out_parts.append(yj * torch.exp(muj * dt))

        return torch.cat(out_parts, dim=1)

    def rollout_from_y0(self, y0: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        ys = [y0]
        y = y0
        for _ in range(steps):
            y = self.varying_step(y, dt)
            ys.append(y)
        return torch.stack(ys, dim=0)  # [T+1,B,k]

    def forward(self, x0: torch.Tensor, steps: int, dt: float) -> Dict[str, torch.Tensor]:
        y0 = self.encode(x0)
        y_roll = self.rollout_from_y0(y0, steps=steps, dt=dt)
        x_roll = torch.stack([self.decode(y_roll[t]) for t in range(steps + 1)], dim=0)
        return {"y_roll": y_roll, "x_roll": x_roll}


@dataclass
class DatasetConfig:
    traj_count: int
    traj_len: int
    dt: float
    tau_max: float
    torque_smooth: float
    q_limit_margin: float


def generate_cdsm_trajectories(cfg: DatasetConfig) -> np.ndarray:
    """
    Returns tensor with shape [traj_count, traj_len, state_dim].
    state = [qa, qb, dqa, dqb]
    """
    robot = MultiJointSpaceRobot()
    all_data = np.zeros((cfg.traj_count, cfg.traj_len, 4), dtype=np.float32)
    q_lim = np.pi / 2.0 - cfg.q_limit_margin

    for n in range(cfg.traj_count):
        q = np.array(
            [
                np.random.uniform(-q_lim, q_lim),
                np.random.uniform(-q_lim, q_lim),
            ],
            dtype=float,
        )
        dq = np.random.uniform(low=-0.3, high=0.3, size=2).astype(float)
        tau = np.random.uniform(low=-cfg.tau_max, high=cfg.tau_max, size=2).astype(float)

        for t in range(cfg.traj_len):
            all_data[n, t] = np.array([q[0], q[1], dq[0], dq[1]], dtype=np.float32)

            # Slowly-varying random torque to excite nonlinear dynamics.
            tau_noise = np.random.uniform(-cfg.tau_max, cfg.tau_max, size=2)
            tau = (1.0 - cfg.torque_smooth) * tau + cfg.torque_smooth * tau_noise
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)

            q, dq = robot.step_coupled(q, dq, tau, dt=cfg.dt)

    return all_data


def normalize_data(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = data.reshape(-1, data.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    norm = (data - mean) / std
    return norm, mean, std


def sample_batch_windows(
    data: np.ndarray,
    batch_size: int,
    pred_steps: int,
    device: torch.device,
) -> torch.Tensor:
    # data shape [N, T, D] -> output [pred_steps+1, B, D]
    n_traj, t_len, d = data.shape
    need = pred_steps + 1
    if t_len < need:
        raise ValueError("Trajectory length must be >= pred_steps + 1.")
    starts_max = t_len - need

    out = np.zeros((need, batch_size, d), dtype=np.float32)
    traj_ids = np.random.randint(0, n_traj, size=batch_size)
    st_ids = np.random.randint(0, starts_max + 1, size=batch_size)
    for i in range(batch_size):
        out[:, i, :] = data[traj_ids[i], st_ids[i] : st_ids[i] + need, :]
    return torch.from_numpy(out).to(device)


def compute_loss(
    model: LuschKoopmanCDSM,
    batch_seq: torch.Tensor,
    dt: float,
    pred_steps: int,
    recon_lam: float,
    pred_lam: float,
    lin_lam: float,
    lin_steps: int,
) -> Dict[str, torch.Tensor]:
    # batch_seq [T+1, B, D]
    x0 = batch_seq[0]
    fwd = model(x0, steps=pred_steps, dt=dt)
    x_roll = fwd["x_roll"]       # [T+1,B,D]
    y_roll_pred = fwd["y_roll"]  # [T+1,B,K]

    # Reconstruction and multi-step prediction on original states.
    recon_loss = torch.mean((x_roll[0] - batch_seq[0]) ** 2)
    pred_loss = torch.mean((x_roll[1:] - batch_seq[1 : pred_steps + 1]) ** 2)

    # Latent linearity consistency:
    # compare encoded true future to latent rollout from y0.
    lin_horizon = min(lin_steps, pred_steps)
    true_y = torch.stack([model.encode(batch_seq[t]) for t in range(lin_horizon + 1)], dim=0)
    lin_loss = torch.mean((y_roll_pred[: lin_horizon + 1] - true_y) ** 2)

    total = recon_lam * recon_loss + pred_lam * pred_loss + lin_lam * lin_loss
    return {
        "total": total,
        "recon": recon_loss,
        "pred": pred_loss,
        "lin": lin_loss,
    }


@torch.no_grad()
def evaluate_model(
    model: LuschKoopmanCDSM,
    val_data: np.ndarray,
    device: torch.device,
    dt: float,
    pred_steps: int,
    recon_lam: float,
    pred_lam: float,
    lin_lam: float,
    lin_steps: int,
    eval_batches: int = 40,
    batch_size: int = 256,
) -> Dict[str, float]:
    model.eval()
    acc = {"total": 0.0, "recon": 0.0, "pred": 0.0, "lin": 0.0}
    for _ in range(eval_batches):
        batch = sample_batch_windows(val_data, batch_size, pred_steps, device)
        loss = compute_loss(model, batch, dt, pred_steps, recon_lam, pred_lam, lin_lam, lin_steps)
        for k in acc:
            acc[k] += float(loss[k].item())
    for k in acc:
        acc[k] /= eval_batches
    return acc


def plot_training_history(history: list, save_dir: str) -> None:
    """绘制训练 / 验证损失 vs Epoch 曲线"""
    epochs = [h[0] for h in history]
    train_total = [h[1] for h in history]
    val_total = [h[5] for h in history]
    train_recon = [h[2] for h in history]
    train_pred  = [h[3] for h in history]
    train_lin   = [h[4] for h in history]
    val_recon   = [h[6] for h in history]
    val_pred    = [h[7] for h in history]
    val_lin     = [h[8] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 总损失
    ax = axes[0, 0]
    ax.semilogy(epochs, train_total, "b-", lw=1.6, label="Train total")
    ax.semilogy(epochs, val_total, "r-", lw=1.6, label="Val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (log scale)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # 子图2: 重建损失
    ax = axes[0, 1]
    ax.semilogy(epochs, train_recon, "b-", lw=1.6, label="Train recon")
    ax.semilogy(epochs, val_recon, "r-", lw=1.6, label="Val recon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recon Loss")
    ax.set_title("Reconstruction Loss (log scale)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # 子图3: 预测损失
    ax = axes[1, 0]
    ax.semilogy(epochs, train_pred, "b-", lw=1.6, label="Train pred")
    ax.semilogy(epochs, val_pred, "r-", lw=1.6, label="Val pred")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pred Loss")
    ax.set_title("Multi-step Prediction Loss (log scale)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # 子图4: 线性一致性损失
    ax = axes[1, 1]
    ax.semilogy(epochs, train_lin, "b-", lw=1.6, label="Train lin")
    ax.semilogy(epochs, val_lin, "r-", lw=1.6, label="Val lin")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lin Loss")
    ax.set_title("Latent Linearity Loss (log scale)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.suptitle(f"Deep Koopman (Lusch 2018) Training History on CDSM", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig_name="training_history")
    plt.close()
    print(f"[绘图] 训练历史曲线已保存至: {save_dir}")


@torch.no_grad()
def plot_prediction_rollout(
    model: LuschKoopmanCDSM,
    val_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    dt: float,
    roll_steps: int,
    device: torch.device,
    save_dir: str,
    n_demo: int = 3,
) -> None:
    """
    在验证集随机挑 n_demo 条轨迹, 从 t=0 用 Koopman 模型自循环预测 roll_steps 步,
    绘制 real vs predicted 对比曲线 (qa, qb, dqa, dqb 四通道).
    """
    rng = np.random.RandomState(42)
    n_traj = val_raw.shape[0]
    indices = rng.choice(n_traj, size=min(n_demo, n_traj), replace=False)

    model.eval()
    fig, axes = plt.subplots(n_demo, 4, figsize=(18, 3.5 * n_demo))
    if n_demo == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        traj = val_raw[idx]  # [T, 4]
        x0_norm = torch.from_numpy((traj[0:1] - mean) / std).float().to(device)

        # 自循环预测
        norm_pred = [x0_norm.squeeze(0).cpu().numpy()]
        y = model.encode(x0_norm)
        for _ in range(roll_steps - 1):
            y = model.varying_step(y, dt)
            norm_pred.append(model.decode(y).squeeze(0).cpu().numpy())
        norm_pred = np.array(norm_pred)
        pred = norm_pred * std + mean  # 反归一化
        real = traj[:roll_steps]

        t = np.arange(roll_steps) * dt
        labels = ["qa (rad)", "qb (rad)", "dqa (rad/s)", "dqb (rad/s)"]
        for col, (lab, ax) in enumerate(zip(labels, axes[row])):
            ax.plot(t, real[:, col], "k-", lw=2.2, label="True")
            ax.plot(t, pred[:, col], "r--", lw=2.0, label="Predicted")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(lab)
            ax.set_title(f"Traj #{idx} — {lab}")
            ax.grid(True, alpha=0.4)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.suptitle("Koopman Model Multi-step Rollout on Validation Set", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig_name="prediction_rollout")
    plt.close()
    print(f"[绘图] 多步预测对比图已保存至: {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lusch 2018 Deep Koopman reproduction on CDSM.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])

    # Data.
    p.add_argument("--train_traj", type=int, default=1800)
    p.add_argument("--val_traj", type=int, default=300)
    p.add_argument("--traj_len", type=int, default=70)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--tau_max", type=float, default=8.0)
    p.add_argument("--torque_smooth", type=float, default=0.08)
    p.add_argument("--q_limit_margin", type=float, default=0.12)

    # Model.
    p.add_argument("--enc_hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--dec_hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--omega_hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--num_complex_pairs", type=int, default=4)
    p.add_argument("--num_real", type=int, default=4)
    p.add_argument("--activation", type=str, default="elu", choices=["relu", "elu", "tanh", "sigmoid"])

    # Training.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--pred_steps", type=int, default=10)
    p.add_argument("--lin_steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=5.0)

    # Loss weights.
    p.add_argument("--recon_lam", type=float, default=1.0)
    p.add_argument("--pred_lam", type=float, default=1.0)
    p.add_argument("--lin_lam", type=float, default=1.0)

    # Output.
    p.add_argument("--out_dir", type=str, default="koopman_lusch_cdsm_results")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 统一使用 Figs/<程序名>/<时间戳>/ 作为输出根目录
    fig_save_dir = get_save_dir()  # e.g. Figs/deep_koopman_lusch_cdsm/20260507_161803
    out_dir = Path(fig_save_dir) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Lusch-style Deep Koopman on CDSM ===")
    print(f"device={device}, seed={args.seed}")

    train_cfg = DatasetConfig(
        traj_count=args.train_traj,
        traj_len=args.traj_len,
        dt=args.dt,
        tau_max=args.tau_max,
        torque_smooth=args.torque_smooth,
        q_limit_margin=args.q_limit_margin,
    )
    val_cfg = DatasetConfig(
        traj_count=args.val_traj,
        traj_len=args.traj_len,
        dt=args.dt,
        tau_max=args.tau_max,
        torque_smooth=args.torque_smooth,
        q_limit_margin=args.q_limit_margin,
    )

    print("Generating train trajectories...")
    train_raw = generate_cdsm_trajectories(train_cfg)
    print("Generating val trajectories...")
    val_raw = generate_cdsm_trajectories(val_cfg)

    train_data, mean, std = normalize_data(train_raw)
    val_data = (val_raw - mean) / std
    np.savez(
        out_dir / "normalization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    model = LuschKoopmanCDSM(
        state_dim=4,
        encoder_hidden=tuple(args.enc_hidden),
        decoder_hidden=tuple(args.dec_hidden),
        omega_hidden=tuple(args.omega_hidden),
        num_complex_pairs=args.num_complex_pairs,
        num_real=args.num_real,
        activation=args.activation,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = {"total": 0.0, "recon": 0.0, "pred": 0.0, "lin": 0.0}

        for _ in range(args.steps_per_epoch):
            batch = sample_batch_windows(train_data, args.batch_size, args.pred_steps, device)
            losses = compute_loss(
                model,
                batch,
                dt=args.dt,
                pred_steps=args.pred_steps,
                recon_lam=args.recon_lam,
                pred_lam=args.pred_lam,
                lin_lam=args.lin_lam,
                lin_steps=args.lin_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k in run:
                run[k] += float(losses[k].item())

        for k in run:
            run[k] /= args.steps_per_epoch

        val = evaluate_model(
            model,
            val_data,
            device=device,
            dt=args.dt,
            pred_steps=args.pred_steps,
            recon_lam=args.recon_lam,
            pred_lam=args.pred_lam,
            lin_lam=args.lin_lam,
            lin_steps=args.lin_steps,
        )

        history.append(
            [
                epoch,
                run["total"],
                run["recon"],
                run["pred"],
                run["lin"],
                val["total"],
                val["recon"],
                val["pred"],
                val["lin"],
            ]
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train total={run['total']:.4e}, recon={run['recon']:.4e}, pred={run['pred']:.4e}, lin={run['lin']:.4e} | "
            f"val total={val['total']:.4e}, recon={val['recon']:.4e}, pred={val['pred']:.4e}, lin={val['lin']:.4e}"
        )

        if val["total"] < best_val:
            best_val = val["total"]
            ckpt = {
                "model_state": model.state_dict(),
                "config": vars(args),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "best_val": best_val,
            }
            torch.save(ckpt, out_dir / "best_model.pt")

    np.savetxt(
        out_dir / "history.csv",
        np.array(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_recon,train_pred,train_lin,val_total,val_recon,val_pred,val_lin",
        comments="",
    )
    print(f"Done. Best val total loss: {best_val:.6e}")
    print(f"Saved: {(out_dir / 'best_model.pt').as_posix()}")
    print(f"Saved: {(out_dir / 'history.csv').as_posix()}")

    # ====================================================================
    # 训练完成后: 绘制结果图并保存到 Figs 目录
    # ====================================================================
    print("\n[绘图] 开始生成训练评估图...")

    # 图1: 训练 & 验证损失历史
    plot_training_history(history, str(out_dir))

    # 图2: 加载最佳模型, 在验证集上做多步预测对比
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    plot_prediction_rollout(
        model=model,
        val_raw=val_raw,
        mean=mean,
        std=std,
        dt=args.dt,
        roll_steps=min(args.pred_steps + 20, args.traj_len),
        device=device,
        save_dir=str(out_dir),
        n_demo=3,
    )
    print("[绘图] 所有图片已保存至 Figs 文件夹.\n")


if __name__ == "__main__":
    main()
