"""
test_deep_koopman_cdsm.py
==========================
对 `deep_koopman_cdsm.py` 训练得到的 Deep Koopman 模型做 **全方位对比测试**:
用 MuJoCo 仿真重新生成全新轨迹作为 "真值", 再让 Koopman 模型仅凭 x₀ 在
潜空间做长期外推, 与 MuJoCo 真值逐时刻对齐, 给出多个角度的误差与可视化.

────────────────────────────────────────────────────────────────────────────
本脚本会做的事
────────────────────────────────────────────────────────────────────────────
(1) 自动找到 Figs/deep_koopman_cdsm/ 下最新一个时间戳目录, 加载:
        best_model.pt         (网络权重 + 配置)
        normalization.npz     (训练时的 mean / std)
(2) 用 MuJoCo 重新跑 N_TEST 条 **新种子** 的自治轨迹作为 ground truth.
(3) 用 Koopman 模型逐条做 long-horizon 自循环外推 (只看初始状态 x₀).
(4) 4 张诊断图 + 1 份数值统计文本, 全部存到
        Figs/test_deep_koopman_cdsm/<ts>/

   ├── time_series_comparison.{png,svg,pdf}
   │     6 子图: 任选 3 条轨迹 × (qa, qb, dqa, dqb) 4 通道时间序列对比
   ├── phase_portraits.{png,svg,pdf}
   │     2 子图: (qa, dqa) 与 (qb, dqb) 相图叠加 ground-truth + Koopman 预测
   ├── error_growth.{png,svg,pdf}
   │     2 子图: 全测试集 RMSE 随预测时刻 t 的增长曲线 (按通道) +
   │              末端 EE 误差增长曲线 (笛卡尔)
   ├── error_distribution.{png,svg,pdf}
   │     4 子图: 每个状态通道在某固定 horizon 下 (predicted vs true) 散点图
   └── summary_statistics.txt
         每条轨迹与全体的 RMSE / MAE / R² / Max-error 统计

────────────────────────────────────────────────────────────────────────────
运行
────────────────────────────────────────────────────────────────────────────
    # 默认: 自动找最新模型, 用 200 条 80 步轨迹做测试
    python test_deep_koopman_cdsm.py

    # 指定模型目录:
    python test_deep_koopman_cdsm.py --model_dir Figs/deep_koopman_cdsm/20260512_152030

    # 自定义测试规模:
    python test_deep_koopman_cdsm.py --n_test 500 --traj_len 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from utils_plot import save_figure, get_save_dir
# 复用训练脚本中的全部基础设施: 模型类 / 数据采集 / 配置 / 常量
from deep_koopman_cdsm import (
    DeepKoopmanCDSM, DataConfig, collect_mujoco_trajectories,
    STATE_DIM, DT_SIM,
)


# ============================================================================
# 1. 自动定位最新一次训练结果
# ============================================================================
def find_latest_model_dir(root: str = "Figs/deep_koopman_cdsm") -> Path:
    """在 Figs/deep_koopman_cdsm/ 下找最新一次训练 (时间戳最大的子目录).

    若 root 不存在或为空, 抛 FileNotFoundError, 提示先运行训练脚本.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"目录 {root_path} 不存在. 请先运行 `python deep_koopman_cdsm.py` 训练模型."
        )
    subdirs = [p for p in root_path.iterdir() if p.is_dir() and (p / "best_model.pt").exists()]
    if not subdirs:
        raise FileNotFoundError(
            f"在 {root_path} 下没有找到含 best_model.pt 的子目录, 请先训练模型."
        )
    # 按名字字典序 (时间戳格式 YYYYMMDD_HHMMSS, 字典序等于时间序)
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]


# ============================================================================
# 2. 加载模型 + 归一化
# ============================================================================
def load_trained_model(
    model_dir: Path, device: torch.device
) -> Tuple[DeepKoopmanCDSM, np.ndarray, np.ndarray, Dict]:
    """加载训练好的 Koopman 模型 (含网络权重, 训练时归一化参数, 配置)."""
    ckpt_path  = model_dir / "best_model.pt"
    norm_path  = model_dir / "normalization.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺失 checkpoint: {ckpt_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"缺失归一化参数: {norm_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"[加载] checkpoint <- {ckpt_path}")
    print(f"        训练时 epoch={ckpt.get('epoch', '?')}, "
          f"best_val={ckpt.get('best_val', float('nan')):.4e}")

    # 严格按 ckpt 里的 config 重建模型 (架构必须与训练时完全一致)
    model = DeepKoopmanCDSM(
        state_dim=STATE_DIM,
        encoder_hidden=tuple(cfg["enc_hidden"]),
        decoder_hidden=tuple(cfg["dec_hidden"]),
        omega_hidden=tuple(cfg["omega_hidden"]),
        num_complex_pairs=cfg["num_complex_pairs"],
        num_real=cfg["num_real"],
        activation=cfg["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = np.load(norm_path)
    mean = norm["mean"].astype(np.float32)
    std  = norm["std"].astype(np.float32)
    print(f"[加载] 归一化 <- {norm_path}: mean={mean.tolist()}, std={std.tolist()}")
    return model, mean, std, cfg


# ============================================================================
# 3. 用 Koopman 模型对 ground-truth 轨迹做整段外推预测
# ============================================================================
@torch.no_grad()
def koopman_rollout_batch(
    model: DeepKoopmanCDSM,
    x0_raw: np.ndarray,          # [N, 4]   原始 (未归一化) 的初始状态
    mean:   np.ndarray,
    std:    np.ndarray,
    dt:     float,
    steps:  int,
    device: torch.device,
) -> np.ndarray:
    """对 N 条轨迹的 x0_raw, 用 Koopman 模型逐步外推 (steps) 步, 返回 [steps, N, 4]
    的预测结果 (反归一化回物理量纲).

    注意:
      - 输入是 t=0 时刻的原始状态, 内部先归一化、编码到潜空间, 再演化.
      - 输出长度 = steps (从 t=1 算起, 与 ground truth 的 [1:steps+1] 对齐),
        不包括 t=0 自身.
    """
    n = x0_raw.shape[0]
    x0_n = torch.from_numpy(((x0_raw - mean) / std).astype(np.float32)).to(device)
    y = model.encode(x0_n)                              # [N, latent_dim]

    pred_n = np.zeros((steps, n, STATE_DIM), dtype=np.float32)
    for t in range(steps):
        y = model.koopman_step(y, dt)                   # 单步线性演化 K(λ(y)) · y
        x_n = model.decode(y).cpu().numpy()
        pred_n[t] = x_n
    return pred_n * std + mean                          # 反归一化


# ============================================================================
# 4. 误差统计
# ============================================================================
def compute_error_stats(
    true_traj: np.ndarray,                              # [N, T, 4]
    pred_traj: np.ndarray,                              # [N, T-1, 4]  (从 t=1 起)
) -> Dict[str, np.ndarray]:
    """计算多个角度的误差.

    返回 dict:
        "rmse_per_step"  : [T-1, 4]  每个时刻、每个通道的 RMSE (跨 N 轨迹)
        "mae_per_step"   : [T-1, 4]  同上, 用 MAE
        "rmse_per_traj"  : [N, 4]    每条轨迹的全程 RMSE
        "max_err_per_ch" : [4]       全测试集每个通道的最大误差
        "r2_per_ch"      : [4]       每个通道的 R² 系数
    """
    true_after = true_traj[:, 1:1 + pred_traj.shape[1]]      # 对齐, [N, steps, 4]
    err = pred_traj - true_after                              # [N, steps, 4]

    rmse_per_step  = np.sqrt(np.mean(err ** 2, axis=0))       # [steps, 4]
    mae_per_step   = np.mean(np.abs(err), axis=0)             # [steps, 4]
    rmse_per_traj  = np.sqrt(np.mean(err ** 2, axis=1))       # [N, 4]
    max_err_per_ch = np.max(np.abs(err.reshape(-1, 4)), axis=0)  # [4]

    # R² = 1 - SS_res / SS_tot, 在 (N*steps) 个 (true, pred) 对上算
    y_true = true_after.reshape(-1, 4)
    y_pred = pred_traj.reshape(-1, 4)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0) + 1e-12
    r2_per_ch = 1.0 - ss_res / ss_tot

    return {
        "rmse_per_step":  rmse_per_step,
        "mae_per_step":   mae_per_step,
        "rmse_per_traj":  rmse_per_traj,
        "max_err_per_ch": max_err_per_ch,
        "r2_per_ch":      r2_per_ch,
    }


# ============================================================================
# 5. 绘图
# ============================================================================
CHANNEL_LABELS = [
    r"$q_a$ (rad)",
    r"$q_b$ (rad)",
    r"$\dot q_a$ (rad/s)",
    r"$\dot q_b$ (rad/s)",
]
CHANNEL_NAMES = ["qa", "qb", "dqa", "dqb"]


def plot_time_series(
    true_traj: np.ndarray, pred_traj: np.ndarray, dt: float,
    indices_to_show: np.ndarray,
) -> None:
    """3 条轨迹 × 4 通道, 每个子图叠加 MuJoCo (实线) vs Koopman 预测 (虚线)."""
    n_show = len(indices_to_show)
    fig, axes = plt.subplots(n_show, 4, figsize=(18, 3.6 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    steps_pred = pred_traj.shape[1]
    t_true = np.arange(true_traj.shape[1]) * dt
    t_pred = np.arange(1, steps_pred + 1) * dt

    for row, idx in enumerate(indices_to_show):
        for col in range(4):
            ax = axes[row, col]
            ax.plot(t_true, true_traj[idx, :, col], "k-",  lw=2.4,
                    label="MuJoCo (truth)")
            ax.plot(t_pred, pred_traj[idx, :, col], "r--", lw=1.8,
                    label="Koopman (pred)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(CHANNEL_LABELS[col])
            ax.grid(True, alpha=0.4)
            if col == 0:
                ax.set_title(f"Test traj #{idx}", loc="left", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc="best")
    plt.suptitle(
        "Time-Series Comparison: Deep Koopman Prediction vs MuJoCo Ground Truth",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig_name="time_series_comparison")
    plt.close()


def plot_phase_portraits(
    true_traj: np.ndarray, pred_traj: np.ndarray, indices_to_show: np.ndarray,
) -> None:
    """两张相图: (qa, dqa) 与 (qb, dqb), 叠加多条轨迹的真值 + 预测."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(indices_to_show), 1)))

    for k, idx in enumerate(indices_to_show):
        color = color_cycle[k]
        # 第 1 组: (qa, dqa)
        axes[0].plot(true_traj[idx, :, 0], true_traj[idx, :, 2],
                     "-", color=color, lw=1.8, alpha=0.85,
                     label=f"#{idx} truth" if k < 3 else None)
        axes[0].plot(pred_traj[idx, :, 0], pred_traj[idx, :, 2],
                     "--", color=color, lw=1.4, alpha=0.85,
                     label=f"#{idx} pred" if k < 3 else None)
        axes[0].plot(true_traj[idx, 0, 0], true_traj[idx, 0, 2], "o",
                     color=color, ms=7, mfc="white", mew=2)
        # 第 2 组: (qb, dqb)
        axes[1].plot(true_traj[idx, :, 1], true_traj[idx, :, 3],
                     "-", color=color, lw=1.8, alpha=0.85)
        axes[1].plot(pred_traj[idx, :, 1], pred_traj[idx, :, 3],
                     "--", color=color, lw=1.4, alpha=0.85)
        axes[1].plot(true_traj[idx, 0, 1], true_traj[idx, 0, 3], "o",
                     color=color, ms=7, mfc="white", mew=2)

    for ax, (xlab, ylab, title) in zip(axes, [
        (r"$q_a$ (rad)", r"$\dot q_a$ (rad/s)", r"Phase portrait: $(q_a, \dot q_a)$"),
        (r"$q_b$ (rad)", r"$\dot q_b$ (rad/s)", r"Phase portrait: $(q_b, \dot q_b)$"),
    ]):
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(True, alpha=0.4)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
    axes[0].legend(fontsize=8, loc="best")
    plt.suptitle("Phase Portraits: Solid = MuJoCo, Dashed = Deep Koopman, "
                 "Circle = Initial Condition", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig_name="phase_portraits")
    plt.close()


def plot_error_growth(
    rmse_per_step: np.ndarray,                          # [steps, 4]
    ee_err_per_step: Optional[np.ndarray],              # [steps] or None
    dt: float,
) -> None:
    """RMSE 随时间增长曲线 (状态通道 + 末端 EE 笛卡尔误差)."""
    has_ee = ee_err_per_step is not None
    fig, axes = plt.subplots(1, 2 if has_ee else 1, figsize=(14 if has_ee else 8, 5.5))
    if not has_ee:
        axes = [axes]

    t = np.arange(1, rmse_per_step.shape[0] + 1) * dt

    # 第 1 子图: 各状态通道的 RMSE
    ax = axes[0]
    colors = ["C0", "C1", "C2", "C3"]
    for c in range(4):
        ax.plot(t, rmse_per_step[:, c], color=colors[c], lw=2.0,
                label=CHANNEL_LABELS[c])
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("RMSE  (per-channel, across test set)")
    ax.set_title("Prediction Error Growth (State Channels)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4, which="both")
    ax.legend(fontsize=9)

    # 第 2 子图: 末端笛卡尔误差
    if has_ee:
        ax2 = axes[1]
        ax2.plot(t, ee_err_per_step * 1000.0, "m-", lw=2.0,
                 label="EE Cartesian RMSE")
        ax2.set_xlabel("Prediction time (s)")
        ax2.set_ylabel("EE position RMSE (mm)")
        ax2.set_title("End-Effector Cartesian Error Growth")
        ax2.grid(True, alpha=0.4)
        ax2.legend(fontsize=9)

    plt.suptitle("How Prediction Error Grows over Horizon (Deep Koopman vs MuJoCo)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig_name="error_growth")
    plt.close()


def plot_error_distribution(
    true_traj: np.ndarray, pred_traj: np.ndarray, horizon: int,
) -> None:
    """每个通道在 t = horizon (固定步) 上 (truth vs pred) 的散点图."""
    horizon = min(horizon, pred_traj.shape[1] - 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.0))

    for c in range(4):
        ax = axes[c]
        truths = true_traj[:, horizon + 1, c]                   # +1 因为 pred 从 t=1 开始
        preds  = pred_traj[:, horizon, c]
        ax.scatter(truths, preds, s=12, alpha=0.55, edgecolors="none",
                   color=f"C{c}")
        lo = float(min(truths.min(), preds.min()))
        hi = float(max(truths.max(), preds.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2,
                alpha=0.7, label="y = x")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True {CHANNEL_LABELS[c]}")
        ax.set_ylabel(f"Predicted {CHANNEL_LABELS[c]}")
        ax.set_title(f"{CHANNEL_LABELS[c]}  @ horizon {horizon+1} steps")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, loc="best")
    plt.suptitle(f"Predicted vs True Scatter @ horizon = {horizon+1} steps "
                 f"(closer to y=x means better)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig_name="error_distribution")
    plt.close()


def write_summary(
    out_path: Path, stats: Dict[str, np.ndarray],
    test_n: int, traj_len: int, dt: float, model_dir: Path,
    ee_err_per_step: Optional[np.ndarray],
) -> None:
    """把测试统计数据写成可读 txt."""
    lines = []
    lines.append("=" * 78)
    lines.append("Deep Koopman vs MuJoCo  -  Test Report")
    lines.append("=" * 78)
    lines.append(f"训练目录                : {model_dir}")
    lines.append(f"测试轨迹条数 N          : {test_n}")
    lines.append(f"每条轨迹步数 T          : {traj_len}  (含 t=0; 预测 horizon = {traj_len-1})")
    lines.append(f"采样周期 dt             : {dt} s   (总时长 = {(traj_len-1)*dt:.2f} s)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("[每通道总体表现] (对全测试集 N * (T-1) 个 (true, pred) 样本汇总)")
    lines.append("-" * 78)
    lines.append(f"{'channel':>8s} | {'final_RMSE':>12s} | {'mean_RMSE':>12s} | "
                 f"{'mean_MAE':>10s} | {'max_|err|':>10s} | {'R²':>8s}")
    for c, lab in enumerate(CHANNEL_NAMES):
        lines.append(
            f"{lab:>8s} | "
            f"{stats['rmse_per_step'][-1, c]:>12.4e} | "
            f"{stats['rmse_per_step'][:, c].mean():>12.4e} | "
            f"{stats['mae_per_step'][:, c].mean():>10.4e} | "
            f"{stats['max_err_per_ch'][c]:>10.4e} | "
            f"{stats['r2_per_ch'][c]:>8.4f}"
        )
    lines.append("")

    if ee_err_per_step is not None:
        lines.append("-" * 78)
        lines.append("[末端 (End-Effector) 笛卡尔位置误差]")
        lines.append("-" * 78)
        lines.append(f"  EE final RMSE (t = {(len(ee_err_per_step))*dt:.2f}s) "
                     f": {ee_err_per_step[-1]*1000:.3f} mm")
        lines.append(f"  EE mean  RMSE (full horizon)        "
                     f": {ee_err_per_step.mean()*1000:.3f} mm")
        lines.append(f"  EE max   RMSE                       "
                     f": {ee_err_per_step.max()*1000:.3f} mm")
        lines.append("")

    lines.append("-" * 78)
    lines.append("[轨迹级 RMSE 分布] (每条轨迹的 full-horizon RMSE)")
    lines.append("-" * 78)
    rmse_traj = stats["rmse_per_traj"]
    lines.append(f"{'channel':>8s} | {'mean':>12s} | {'median':>12s} | "
                 f"{'90% perc':>12s} | {'max':>12s}")
    for c, lab in enumerate(CHANNEL_NAMES):
        vals = rmse_traj[:, c]
        lines.append(
            f"{lab:>8s} | "
            f"{vals.mean():>12.4e} | "
            f"{np.median(vals):>12.4e} | "
            f"{np.percentile(vals, 90):>12.4e} | "
            f"{vals.max():>12.4e}"
        )
    lines.append("")
    lines.append("=" * 78)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text + f"\n\n[保存] {out_path}")


# ============================================================================
# 6. 末端笛卡尔位置: 用 MuJoCo 模型 (mj_forward) 把 [qa, qb] 转成 EE 坐标
# ============================================================================
def compute_ee_positions(states_batch: np.ndarray) -> np.ndarray:
    """对 [N, T, 4] 的状态批量, 调用 MuJoCo 的 mj_forward 计算每帧末端坐标 [N, T, 2].

    这是为了让我们既能比较 (qa, qb, dqa, dqb) 也能比较 EE 的物理位置.
    """
    from deep_koopman_cdsm import load_mujoco_model
    model, data, idx = load_mujoco_model()
    n, t_len, _ = states_batch.shape
    ee_xy = np.zeros((n, t_len, 2), dtype=np.float32)
    import mujoco
    for k in range(n):
        for t in range(t_len):
            qa, qb = float(states_batch[k, t, 0]), float(states_batch[k, t, 1])
            data.qpos[idx["qadr"]["joint1"]] = qa
            data.qpos[idx["qadr"]["joint2"]] = qa
            data.qpos[idx["qadr"]["joint3"]] = qb
            data.qpos[idx["qadr"]["joint4"]] = qb
            mujoco.mj_forward(model, data)
            ee_xy[k, t] = data.site_xpos[idx["site_ee"]][:2]
    return ee_xy


# ============================================================================
# 7. 主程序
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test Deep Koopman vs MuJoCo on CDSM.")
    p.add_argument("--model_dir", type=str, default="",
                   help="训练输出目录 (留空则自动找最新一个)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--n_test",   type=int,   default=200, help="测试轨迹条数")
    p.add_argument("--traj_len", type=int,   default=80,  help="每条轨迹步数 (含 t=0)")
    p.add_argument("--seed",     type=int,   default=20260512, help="测试集 IC 随机种子")

    # IC 范围 (默认与训练一致, 也允许故意外推到训练分布外做泛化测试)
    p.add_argument("--qa_range",  type=float, default=1.0)
    p.add_argument("--qb_range",  type=float, default=1.0)
    p.add_argument("--dqa_range", type=float, default=0.8)
    p.add_argument("--dqb_range", type=float, default=0.8)

    p.add_argument("--n_show_ts",     type=int, default=3,
                   help="时间序列图展示几条轨迹")
    p.add_argument("--n_show_phase",  type=int, default=8,
                   help="相图叠加几条轨迹")
    p.add_argument("--scatter_horizon", type=int, default=39,
                   help="散点图对应的预测 horizon (步数, 从 0 起算)")
    p.add_argument("--compute_ee_err", action="store_true", default=True,
                   help="是否计算末端 EE 笛卡尔误差 (慢一些, 但更直观)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---------- 找到训练目录 + 加载模型 ----------
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = find_latest_model_dir()
    print("=" * 78)
    print(" Deep Koopman vs MuJoCo : Test Script")
    print("=" * 78)
    print(f"  使用训练目录: {model_dir}")
    print(f"  推理设备    : {device}")

    model, mean, std, train_cfg = load_trained_model(model_dir, device)
    dt = float(train_cfg.get("dt", DT_SIM))
    print(f"  采样周期 dt = {dt} s   ·   测试规模: "
          f"N={args.n_test}  T={args.traj_len}  (horizon={args.traj_len - 1} 步)")

    # ---------- 准备输出目录 ----------
    test_out_dir = Path(get_save_dir())
    test_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  结果输出    : {test_out_dir}\n")

    # ============================================================
    # Step 1. 用 MuJoCo 生成测试轨迹 (新种子, 训练时没见过)
    # ============================================================
    test_cfg = DataConfig(
        traj_count=args.n_test, traj_len=args.traj_len, dt=dt,
        qa_range=args.qa_range,  qb_range=args.qb_range,
        dqa_range=args.dqa_range, dqb_range=args.dqb_range,
        f_preload=train_cfg.get("f_preload", 20.0),
        seed=args.seed,
    )
    print("[Step 1] 生成 MuJoCo 测试轨迹 (ground truth)...")
    test_set = collect_mujoco_trajectories(test_cfg, verbose=True)
    true_traj = test_set["states"]                          # [N, T, 4]
    true_ee   = test_set["ee_xy"]                           # [N, T, 2]

    # ============================================================
    # Step 2. 用 Deep Koopman 模型外推
    # ============================================================
    print("\n[Step 2] 用 Deep Koopman 做长视距自循环外推...")
    t0 = time.time()
    x0 = true_traj[:, 0, :].astype(np.float32)              # [N, 4]
    pred_traj = koopman_rollout_batch(
        model, x0, mean, std, dt=dt,
        steps=args.traj_len - 1, device=device,
    )                                                       # [steps, N, 4]
    pred_traj = pred_traj.transpose(1, 0, 2)                # -> [N, steps, 4]
    print(f"    Koopman 推理耗时 = {time.time()-t0:.2f} s   "
          f"(每条 {(time.time()-t0)/args.n_test*1000:.2f} ms)")

    # ============================================================
    # Step 3. 误差统计 + (可选) 末端 EE 误差
    # ============================================================
    print("\n[Step 3] 计算误差统计...")
    stats = compute_error_stats(true_traj, pred_traj)
    print("  通道级最终 RMSE  : "
          + ", ".join([f"{CHANNEL_NAMES[c]}={stats['rmse_per_step'][-1, c]:.4e}"
                       for c in range(4)]))
    print("  通道级 R²         : "
          + ", ".join([f"{CHANNEL_NAMES[c]}={stats['r2_per_ch'][c]:.4f}"
                       for c in range(4)]))

    ee_err_per_step = None
    if args.compute_ee_err:
        print("\n  计算末端 EE 笛卡尔误差 (前向运动学 via MuJoCo)...")
        pred_ee = compute_ee_positions(pred_traj)            # [N, steps, 2]
        # 与 true_ee[:, 1:steps+1] 对齐
        true_ee_after = true_ee[:, 1:1 + pred_ee.shape[1]]
        ee_err = np.linalg.norm(pred_ee - true_ee_after, axis=-1)   # [N, steps]
        ee_err_per_step = np.sqrt(np.mean(ee_err ** 2, axis=0))     # [steps]
        print(f"  EE 末段 RMSE     : {ee_err_per_step[-1] * 1000:.3f} mm   "
              f"·  EE 全程均值 : {ee_err_per_step.mean() * 1000:.3f} mm")

    # ============================================================
    # Step 4. 绘图 + 摘要
    # ============================================================
    print("\n[Step 4] 绘图 + 输出摘要...")
    rng = np.random.RandomState(args.seed + 7)
    ts_idx = rng.choice(args.n_test, size=min(args.n_show_ts, args.n_test), replace=False)
    ph_idx = rng.choice(args.n_test, size=min(args.n_show_phase, args.n_test), replace=False)

    plot_time_series(true_traj, pred_traj, dt=dt, indices_to_show=ts_idx)
    plot_phase_portraits(true_traj, pred_traj, indices_to_show=ph_idx)
    plot_error_growth(stats["rmse_per_step"], ee_err_per_step, dt=dt)
    plot_error_distribution(true_traj, pred_traj, horizon=args.scatter_horizon)

    # 写文本统计
    write_summary(
        out_path=test_out_dir / "summary_statistics.txt",
        stats=stats, test_n=args.n_test, traj_len=args.traj_len,
        dt=dt, model_dir=model_dir, ee_err_per_step=ee_err_per_step,
    )

    print(f"\n=== 测试完成. 所有结果已保存至: {test_out_dir} ===\n")


if __name__ == "__main__":
    main()
