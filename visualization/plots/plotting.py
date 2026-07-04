"""结果绘图工具。

本模块只负责把 `run_03` 和 `run_04` 已经保存的数值结果画成图，不参与训练、
预测或控制计算。所有图片默认保存为 PNG，便于快速查看；原始数值仍以
`npz/json` 为准，方便后续重新绘图或做论文级排版。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_LABELS = ("qa", "qb", "dqa", "dqb")
CONTROL_LABELS = ("tau_a", "tau_b")


def _ensure_dir(out_dir: str | Path) -> Path:
    """确保绘图输出目录存在。"""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """统一保存并关闭 figure。"""
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_prediction_figures(
    *,
    out_dir: str | Path,
    rollouts: Dict[str, np.ndarray],
    metrics: Dict[str, object],
    models: Iterable[str],
    modes: Iterable[str],
    dt: float,
    demo_traj: int = 0,
) -> List[str]:
    """绘制 one-step/rollout 预测评估图。

    参数:
        out_dir: 图片保存目录。
        rollouts: `run_03` 中准备写入 `prediction_rollouts.npz` 的轨迹字典。
        metrics: `prediction_metrics.json` 对应的指标字典。
        models: 参与绘图的模型名。
        modes: 预测模式，通常为 `one_step/rollout`。
        dt: 采样周期，单位 s，用于横轴。
        demo_traj: 展示动态响应的轨迹编号。

    返回:
        已保存图片文件名列表。
    """
    out_path = _ensure_dir(out_dir)
    saved: List[str] = []
    true = np.asarray(rollouts["true"], dtype=np.float64)
    traj_idx = min(max(int(demo_traj), 0), true.shape[0] - 1)
    t = np.arange(true.shape[1], dtype=np.float64) * float(dt)

    for mode in modes:
        # 图 1：各模型 total RMSE 柱状图。
        model_names = [m for m in models if mode in metrics["models"].get(m, {})]
        total_rmse = [metrics["models"][m][mode]["total_rmse"] for m in model_names]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(model_names, total_rmse)
        ax.set_ylabel("Total RMSE")
        ax.set_title(f"{mode} prediction total RMSE")
        ax.grid(True, axis="y", alpha=0.3)
        fname = f"{mode}_total_rmse"
        _save(fig, out_path, fname)
        saved.append(f"{fname}.png")

        # 图 2：逐时刻 RMSE 增长曲线。
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        for model_name in model_names:
            step_rmse = np.asarray(metrics["models"][model_name][mode]["step_rmse"], dtype=np.float64)
            ax.plot(t[: step_rmse.shape[0]], step_rmse, lw=1.6, label=model_name.upper())
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Step RMSE")
        ax.set_title(f"{mode} prediction RMSE over time")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fname = f"{mode}_rmse_growth"
        _save(fig, out_path, fname)
        saved.append(f"{fname}.png")

        # 图 3：选定轨迹的 4 个状态动态响应。
        fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.2), sharex=True)
        for j, label in enumerate(STATE_LABELS):
            axes[j].plot(t, true[traj_idx, :, j], "k-", lw=1.8, label="true")
            for model_name in model_names:
                pred_key = f"{model_name}_{mode}_pred"
                if pred_key in rollouts:
                    pred = np.asarray(rollouts[pred_key], dtype=np.float64)
                    axes[j].plot(t, pred[traj_idx, :, j], "--", lw=1.2, label=model_name.upper())
            axes[j].set_ylabel(label)
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        axes[0].legend(ncol=min(len(model_names) + 1, 5), fontsize=8)
        fig.suptitle(f"{mode} dynamic response, trajectory {traj_idx}")
        fname = f"{mode}_dynamic_response_traj{traj_idx}"
        _save(fig, out_path, fname)
        saved.append(f"{fname}.png")

    return saved


def plot_tracking_figures(
    *,
    out_dir: str | Path,
    logs: Dict[str, Dict[str, np.ndarray]],
    metrics: Dict[str, object],
) -> List[str]:
    """绘制闭环跟踪对比图。

    参数:
        out_dir: 图片保存目录。
        logs: 每个模型的闭环日志。
        metrics: `tracking_metrics.json` 对应的指标字典。

    返回:
        已保存图片文件名列表。
    """
    out_path = _ensure_dir(out_dir)
    saved: List[str] = []
    model_names = list(logs.keys())
    if not model_names:
        return saved

    # 图 1：关节角跟踪。
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    first = logs[model_names[0]]
    for j, label in enumerate(("qa", "qb")):
        axes[j].plot(first["t"], first["q_ref"][:, j], "k--", lw=1.6, label="ref")
        for model_name in model_names:
            log = logs[model_name]
            axes[j].plot(log["t"], log["x_meas"][:, j], lw=1.3, label=model_name.upper())
        axes[j].set_ylabel(f"{label} (rad)")
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=min(len(model_names) + 1, 4), fontsize=8)
    fig.suptitle("Closed-loop joint tracking")
    _save(fig, out_path, "tracking_joint_positions")
    saved.append("tracking_joint_positions.png")

    # 图 2：关节角误差。
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    for j, label in enumerate(("qa", "qb")):
        for model_name in model_names:
            log = logs[model_name]
            err = log["x_meas"][:, j] - log["q_ref"][:, j]
            axes[j].plot(log["t"], err, lw=1.3, label=model_name.upper())
        axes[j].set_ylabel(f"e_{label} (rad)")
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=min(len(model_names), 4), fontsize=8)
    fig.suptitle("Closed-loop joint position error")
    _save(fig, out_path, "tracking_joint_errors")
    saved.append("tracking_joint_errors.png")

    # 图 3：关节力矩命令。
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    for j, label in enumerate(CONTROL_LABELS):
        for model_name in model_names:
            log = logs[model_name]
            axes[j].plot(log["t"], log["tau_cmd"][:, j], lw=1.3, label=model_name.upper())
        axes[j].set_ylabel(f"{label} (Nm)")
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=min(len(model_names), 4), fontsize=8)
    fig.suptitle("Joint torque commands")
    _save(fig, out_path, "tracking_tau_commands")
    saved.append("tracking_tau_commands.png")

    # 图 4：8 根绳张力。模型多时用轻线条显示，重点看峰值和趋势。
    fig, axes = plt.subplots(len(model_names), 1, figsize=(9.0, max(3.0, 2.6 * len(model_names))), sharex=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        log = logs[model_name]
        for i in range(log["cable_tensions"].shape[1]):
            ax.plot(log["t"], log["cable_tensions"][:, i], lw=0.9)
        ax.set_ylabel(f"{model_name.upper()}\nF (N)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Cable tension commands")
    _save(fig, out_path, "tracking_cable_tensions")
    saved.append("tracking_cable_tensions.png")

    # 图 5：各模型关节角 RMSE 柱状图。
    rmse = [metrics["models"][name]["rmse_q"] for name in model_names]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(model_names, rmse)
    ax.set_ylabel("Joint position RMSE (rad)")
    ax.set_title("Closed-loop tracking RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path, "tracking_rmse_summary")
    saved.append("tracking_rmse_summary.png")

    return saved


def plot_cartesian_tracking_figures(
    *,
    out_dir: str | Path,
    logs: Dict[str, Dict[str, np.ndarray]],
    ee_logs: Dict[str, np.ndarray],
    ref: Dict[str, np.ndarray],
    metrics: Dict[str, object],
) -> List[str]:
    """绘制末端笛卡尔轨迹经 IK 后的闭环跟踪图。

    参数:
        out_dir: 图片保存目录。
        logs: 每个模型的关节空间闭环日志。
        ee_logs: 每个模型由真实 `q_meas` 经 MuJoCo FK 得到的末端 xy。
        ref: 参考轨迹字典，至少包含 `t/q_ref/dq_ref/ee_ref`。
        metrics: 本次笛卡尔跟踪指标字典，用于 RMSE 柱状图。

    返回:
        已保存的 PNG 文件名列表。
    """
    out_path = _ensure_dir(out_dir)
    saved: List[str] = []
    model_names = list(logs.keys())
    if not model_names:
        return saved

    t_ref = np.asarray(ref["t"], dtype=np.float64)
    ee_ref = np.asarray(ref["ee_ref"], dtype=np.float64)
    q_ref = np.asarray(ref["q_ref"], dtype=np.float64)
    dq_ref = np.asarray(ref["dq_ref"], dtype=np.float64)

    # 图 1：期望末端轨迹和各模型真实末端轨迹。
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(ee_ref[:, 0], ee_ref[:, 1], "k--", lw=1.8, label="desired EE")
    for model_name in model_names:
        ee = ee_logs[model_name]
        ax.plot(ee[:, 0], ee[:, 1], lw=1.3, label=model_name.upper())
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("End-effector Cartesian path tracking")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_path, "cartesian_xy_path_tracking")
    saved.append("cartesian_xy_path_tracking.png")

    # 图 2：末端 x/y 位置随时间变化。
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    for j, label in enumerate(("x", "y")):
        axes[j].plot(t_ref, ee_ref[:, j], "k--", lw=1.6, label="desired")
        for model_name in model_names:
            log = logs[model_name]
            ee = ee_logs[model_name]
            axes[j].plot(log["t"], ee[:, j], lw=1.2, label=model_name.upper())
        axes[j].set_ylabel(f"{label} (m)")
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=min(len(model_names) + 1, 4), fontsize=8)
    fig.suptitle("End-effector x/y response")
    _save(fig, out_path, "cartesian_xy_response")
    saved.append("cartesian_xy_response.png")

    # 图 3：末端笛卡尔误差随时间变化。
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    for j, label in enumerate(("x", "y")):
        for model_name in model_names:
            log = logs[model_name]
            ee = ee_logs[model_name]
            n = min(ee.shape[0], ee_ref.shape[0])
            err = ee[:n, j] - ee_ref[:n, j]
            axes[j].plot(log["t"][:n], err, lw=1.2, label=model_name.upper())
        axes[j].set_ylabel(f"e_{label} (m)")
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=min(len(model_names), 4), fontsize=8)
    fig.suptitle("End-effector Cartesian tracking error")
    _save(fig, out_path, "cartesian_xy_errors")
    saved.append("cartesian_xy_errors.png")

    # 图 4：IK 生成的关节参考。
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 7.2), sharex=True)
    for j, label in enumerate(("qa", "qb")):
        axes[j].plot(t_ref, q_ref[:, j], lw=1.3)
        axes[j].set_ylabel(f"{label} (rad)")
        axes[j].grid(True, alpha=0.3)
    for j, label in enumerate(("dqa", "dqb")):
        axes[j + 2].plot(t_ref, dq_ref[:, j], lw=1.3)
        axes[j + 2].set_ylabel(f"{label} (rad/s)")
        axes[j + 2].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint reference generated by MuJoCo IK")
    _save(fig, out_path, "cartesian_ik_joint_reference")
    saved.append("cartesian_ik_joint_reference.png")

    # 图 5：各模型末端 RMSE 柱状图。
    rmse = [metrics["models"][name]["cartesian"]["rmse_ee"] for name in model_names]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(model_names, rmse)
    ax.set_ylabel("End-effector RMSE (m)")
    ax.set_title("Cartesian tracking RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path, "cartesian_rmse_summary")
    saved.append("cartesian_rmse_summary.png")

    return saved
