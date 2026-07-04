"""Plot closed-loop control results saved by control scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, figures_dir: Path, stem: str) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [figures_dir / f"{stem}.png", figures_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [str(path) for path in paths]


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def figures_dir_for_result(result_dir: str | Path) -> Path:
    """Return the shared figures directory for a control result run.

    Expected layout:
        control/outputs/<run_type>/<method>/<run_id>

    Figures are stored like prediction outputs:
        control/outputs/<run_type>/figures/<run_id>
    """
    root = Path(result_dir).resolve()
    if root.parent.parent.name in {"smoke_test", "full_run"}:
        return root.parent.parent / "figures" / root.name
    return root.parent / "figures" / root.name


def plot_closed_loop(log: dict[str, np.ndarray], figures_dir: str | Path, *, label: str = "model") -> list[str]:
    """Create joint, torque, cable, and Cartesian tracking figures."""
    figures = Path(figures_dir)
    t = np.asarray(log["t"], dtype=np.float64)
    x_meas = np.asarray(log["x_meas"], dtype=np.float64)
    x_ref = np.asarray(log["x_ref"], dtype=np.float64)
    out: list[str] = []

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    names = ("qa", "qb")
    for j, name in enumerate(names):
        axes[j].plot(t, x_ref[:, j], "k--", linewidth=1.5, label=f"{name} ref")
        axes[j].plot(t, x_meas[:, j], linewidth=1.2, label=f"{name} {label}")
        axes[j].set_ylabel("rad")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint position tracking")
    out += _save(fig, figures, "joint_position_tracking")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for j, name in enumerate(("dqa", "dqb")):
        idx = j + 2
        axes[j].plot(t, x_ref[:, idx], "k--", linewidth=1.5, label=f"{name} ref")
        axes[j].plot(t, x_meas[:, idx], linewidth=1.2, label=f"{name} {label}")
        axes[j].set_ylabel("rad/s")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint velocity tracking")
    out += _save(fig, figures, "joint_velocity_tracking")

    tau = np.asarray(log["tau_cmd"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, tau[:, 0], label="tau_a")
    ax.plot(t, tau[:, 1], label="tau_b")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Nm")
    ax.set_title("Joint torque command")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    out += _save(fig, figures, "joint_torque_command")

    tensions = np.asarray(log["cable_tensions"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i in range(tensions.shape[1]):
        ax.plot(t, tensions[:, i], linewidth=0.9, label=f"cable {i + 1}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("N")
    ax.set_title("Cable tensions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    out += _save(fig, figures, "cable_tensions")

    if "ee_meas" in log and "ee_ref" in log:
        ee_meas = np.asarray(log["ee_meas"], dtype=np.float64)
        ee_ref = np.asarray(log["ee_ref"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(ee_ref[:, 0], ee_ref[:, 1], "k--", linewidth=1.5, label="reference")
        ax.plot(ee_meas[:, 0], ee_meas[:, 1], linewidth=1.2, label=label)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("End-effector path")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        out += _save(fig, figures, "cartesian_path")

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for j, axis in enumerate(("x", "y")):
            axes[j].plot(t, ee_ref[:, j], "k--", linewidth=1.5, label=f"{axis} ref")
            axes[j].plot(t, ee_meas[:, j], linewidth=1.2, label=f"{axis} {label}")
            axes[j].set_ylabel("m")
            axes[j].grid(True, alpha=0.3)
            axes[j].legend(loc="best")
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("End-effector coordinate tracking")
        out += _save(fig, figures, "cartesian_coordinate_tracking")

    return out


def update_metrics_with_figures(result_dir: str | Path, figures: list[str]) -> None:
    metrics_path = Path(result_dir) / "metrics" / "tracking_metrics.json"
    if not metrics_path.exists():
        return
    payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["figures"] = figures
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_result_dir(result_dir: str | Path, *, model_name: str) -> list[str]:
    root = Path(result_dir).resolve()
    log_path = root / "arrays" / f"closed_loop_{model_name}.npz"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    figures = plot_closed_loop(_load_npz(log_path), figures_dir_for_result(root), label=model_name)
    update_metrics_with_figures(root, figures)
    return figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot closed-loop control results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--model", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    figures = plot_result_dir(args.result_dir, model_name=args.model)
    print("\n".join(figures))


if __name__ == "__main__":
    main()
