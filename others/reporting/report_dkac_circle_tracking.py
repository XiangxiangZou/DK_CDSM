"""Generate the requested DKAC circle-tracking figures from a closed-loop log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CABLE_LABELS = (
    "cable11",
    "cable12",
    "cable13",
    "cable14",
    "cable21",
    "cable22",
    "cable23",
    "cable24",
)
GREEN = "#18A558"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DKAC circle tracking, error, RMSE, torque, and cable figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Circle result directory containing closed_loop_dkac.npz.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory; defaults to result_dir/requested_outputs.",
    )
    return parser


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _load_log(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _find_closed_loop_log(result_dir: Path) -> Path:
    candidates = [
        result_dir / "closed_loop_dkac.npz",
        result_dir / "arrays" / "closed_loop_dkac.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"DKAC closed-loop log not found. Searched:\n  {searched}")


def main() -> None:
    args = build_parser().parse_args()
    result_dir = Path(args.result_dir).resolve()
    log_path = _find_closed_loop_log(result_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_dir / "requested_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    log = _load_log(log_path)
    t = np.asarray(log["t"], dtype=np.float64)
    q = np.asarray(log["x_meas"], dtype=np.float64)[:, :2]
    q_ref = np.asarray(log["q_ref"], dtype=np.float64)
    tau = np.asarray(log["tau_cmd"], dtype=np.float64)
    tensions = np.asarray(log["cable_tensions"], dtype=np.float64)
    ee = np.asarray(log["ee_meas"], dtype=np.float64)
    ee_ref = np.asarray(log["ee_ref"], dtype=np.float64)
    n = min(len(t), len(q), len(q_ref), len(ee), len(ee_ref))
    t, q, q_ref, tau, tensions, ee, ee_ref = (
        t[:n],
        q[:n],
        q_ref[:n],
        tau[:n],
        tensions[:n],
        ee[:n],
        ee_ref[:n],
    )
    q_error = q - q_ref
    ee_error = ee - ee_ref
    q_rmse = np.sqrt(np.mean(q_error * q_error, axis=0))
    ee_rmse_axis = np.sqrt(np.mean(ee_error * ee_error, axis=0))
    ee_rmse_norm = float(np.sqrt(np.mean(np.sum(ee_error * ee_error, axis=1))))

    files: list[str] = []

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot(ee_ref[:, 0], ee_ref[:, 1], "k--", lw=2.0, label="Desired")
    ax.plot(ee[:, 0], ee[:, 1], color=GREEN, lw=1.8, label="DKAC actual")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("End-effector circle tracking")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    files.append(_save(fig, out_dir, "01_end_effector_path_tracking"))

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.2), sharex=True)
    axes[0].plot(t, ee_error[:, 0], color=GREEN)
    axes[0].set_ylabel("e_x (m)")
    axes[1].plot(t, ee_error[:, 1], color=GREEN)
    axes[1].set_ylabel("e_y (m)")
    axes[2].plot(t, np.linalg.norm(ee_error, axis=1), color=GREEN)
    axes[2].set_ylabel("||e_xy|| (m)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("End-effector tracking error")
    files.append(_save(fig, out_dir, "02_end_effector_tracking_error"))

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    for index, label in enumerate(("qa", "qb")):
        axes[index].plot(t, q_ref[:, index], "k--", lw=1.7, label="Desired")
        axes[index].plot(t, q[:, index], color=GREEN, lw=1.4, label="DKAC actual")
        axes[index].set_ylabel(f"{label} (rad)")
        axes[index].grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint angle tracking")
    files.append(_save(fig, out_dir, "03_joint_angle_tracking"))

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    for index, label in enumerate(("qa", "qb")):
        axes[index].plot(t, q_error[:, index], color=GREEN)
        axes[index].set_ylabel(f"e_{label} (rad)")
        axes[index].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint angle tracking error")
    files.append(_save(fig, out_dir, "04_joint_angle_error"))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    values = [ee_rmse_axis[0], ee_rmse_axis[1], ee_rmse_norm]
    bars = ax.bar(("x", "y", "2D norm"), values, color=(GREEN, GREEN, "#0B6E3B"))
    ax.bar_label(bars, fmt="%.5f")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("End-effector error RMSE")
    ax.grid(axis="y", alpha=0.3)
    files.append(_save(fig, out_dir, "05_end_effector_rmse"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(("qa", "qb"), q_rmse, color=(GREEN, "#0B6E3B"))
    ax.bar_label(bars, fmt="%.6f")
    ax.set_ylabel("RMSE (rad)")
    ax.set_title("Joint angle RMSE")
    ax.grid(axis="y", alpha=0.3)
    files.append(_save(fig, out_dir, "06_joint_angle_rmse"))

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    for index, label in enumerate(("tau_a", "tau_b")):
        axes[index].plot(t, tau[:, index], color=GREEN)
        axes[index].set_ylabel(f"{label} (Nm)")
        axes[index].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Equivalent joint torque commands")
    files.append(_save(fig, out_dir, "07_equivalent_joint_torque"))

    fig, axes = plt.subplots(4, 2, figsize=(10.5, 9.0), sharex=True)
    for index, (ax, label) in enumerate(zip(axes.flat, CABLE_LABELS)):
        ax.plot(t, tensions[:, index], color=GREEN, lw=1.0)
        ax.set_ylabel("F (N)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Individual cable tensions")
    files.append(_save(fig, out_dir, "08_individual_cable_tensions"))

    summary = {
        "source_log": str(log_path),
        "samples": int(n),
        "joint_rmse_rad": {"qa": float(q_rmse[0]), "qb": float(q_rmse[1])},
        "end_effector_rmse_m": {
            "x": float(ee_rmse_axis[0]),
            "y": float(ee_rmse_axis[1]),
            "norm_2d": ee_rmse_norm,
        },
        "peak_abs_joint_torque_nm": {
            "tau_a": float(np.max(np.abs(tau[:, 0]))),
            "tau_b": float(np.max(np.abs(tau[:, 1]))),
        },
        "peak_cable_tension_n": {
            label: float(np.max(tensions[:, index]))
            for index, label in enumerate(CABLE_LABELS)
        },
        "figures": files,
    }
    (out_dir / "dkac_circle_tracking_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] DKAC circle report -> {out_dir}")


if __name__ == "__main__":
    main()
