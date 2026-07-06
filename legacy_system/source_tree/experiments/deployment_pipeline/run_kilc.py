"""Run Yu-Tan-style KILC on a CDSM Cartesian circle reference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._paths import PROJECT_ROOT

from cdsm.kinematics.ik import IKConfig, MujocoSiteIK
from cdsm.references.cartesian import (
    CartesianReferenceConfig,
    generate_cartesian_reference,
)
from cdsm.runtime.kilc_tracking import run_yu_tan_kilc_tracking
from koopman_control.control.yu_tan_kilc import YuTanKILCConfig
from koopman_control.data.artifacts import save_json
from koopman_control.evaluation.tracking import cartesian_tracking_metrics, tracking_metrics
from koopman_control.models.registry import load_continuous_dkuc_model


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deployment" / "kilc_cartesian_circle.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "results" / "deployment_pipeline" / "kilc" / "cartesian_circle"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Yu-Tan-style DKUC + KILC on a 20 s Cartesian circle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _override(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.artifact_dir is not None:
        config["artifact_dir"] = str(args.artifact_dir)
    if args.max_trials is not None:
        config["max_trials"] = int(args.max_trials)


def _make_output_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = str(config.get("tag", ""))
        suffix = f"_{tag}" if tag else ""
        out_dir = DEFAULT_OUTPUT_ROOT / f"{stamp}{suffix}"
    for child in ("metrics", "arrays", "figures", "logs"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    return out_dir


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _build_reference(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], IKConfig]:
    ref_cfg = config["reference"]
    cart_cfg = CartesianReferenceConfig(
        kind="circle",
        dt=float(config["dt"]),
        period=float(ref_cfg["duration"]),
        num_cycles=1.0,
        center_x=float(ref_cfg["center_x"]),
        center_y=float(ref_cfg["center_y"]),
        radius_x=float(ref_cfg["radius"]),
        radius_y=float(ref_cfg["radius"]),
        phase=float(ref_cfg.get("phase", 0.0)),
        start_hold=float(ref_cfg.get("start_hold", 0.0)),
        time_scaling=str(ref_cfg.get("time_scaling", "quintic")),
    )
    ik_cfg = IKConfig(**config["ik"])
    xml_path = PROJECT_ROOT / config["xml_path"]
    ik_solver = MujocoSiteIK(xml_path, float(config["dt"]), ik_cfg)
    cartesian = generate_cartesian_reference(cart_cfg)
    inverse = ik_solver.solve_trajectory(
        np.asarray(cartesian["xy_ref"], dtype=np.float64)
    )
    reference = {
        "t": np.asarray(cartesian["t"], dtype=np.float64),
        "q_ref": np.asarray(inverse["q_ref"], dtype=np.float64),
        "dq_ref": np.asarray(inverse["dq_ref"], dtype=np.float64),
        "ee_ref": np.asarray(cartesian["xy_ref"], dtype=np.float64),
        "dxy_ref": np.asarray(cartesian["dxy_ref"], dtype=np.float64),
        "ee_ik": np.asarray(inverse["ee_ik"], dtype=np.float64),
        "ik_error": np.asarray(inverse["ik_error"], dtype=np.float64),
        "ik_converged": np.asarray(inverse["ik_converged"], dtype=bool),
        "ik_iterations": np.asarray(inverse["ik_iterations"], dtype=np.int32),
    }
    return reference, ik_cfg


def _kilc_config(config: dict[str, Any]) -> YuTanKILCConfig:
    return YuTanKILCConfig(**config["kilc"])


def _trial_metrics(
    *,
    reference: dict[str, np.ndarray],
    log: dict[str, np.ndarray],
    ik_solver: MujocoSiteIK,
) -> dict[str, Any]:
    ee_meas = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
    count = len(log["t"])
    values = tracking_metrics(log)
    values["cartesian"] = cartesian_tracking_metrics(
        ee_meas=ee_meas,
        ee_ref=reference["ee_ref"][:count],
        ik_error=reference["ik_error"][:count],
    )
    values["lifted_error_rms"] = float(np.sqrt(np.mean(log["e_z"] ** 2)))
    values["tau_rms"] = float(np.sqrt(np.mean(log["control_cmd"] ** 2)))
    values["tau_peak_abs"] = float(np.max(np.abs(log["control_cmd"])))
    values["cable_tension_peak"] = float(np.max(log["cable_tensions"]))
    values["saturation_ratio"] = float(np.mean(log["saturated"].astype(np.float64)))
    return values


def _save_arrays(
    *,
    out_dir: Path,
    reference: dict[str, np.ndarray],
    result: dict[str, Any],
    final_ee_meas: np.ndarray,
) -> None:
    trials = result["trials"]
    first = trials[0]
    final = trials[-1]
    np.savez_compressed(
        out_dir / "arrays" / "kilc_result.npz",
        **{f"ref_{key}": value for key, value in reference.items()},
        first_x_meas=first["x_meas"],
        final_x_meas=final["x_meas"],
        final_ee_meas=final_ee_meas,
        final_z_meas=final["z_meas"],
        final_z_ref=final["z_ref"],
        final_e_z=final["e_z"],
        final_u_ilc=final["u_ilc"],
        final_u_adaptive=final["u_adaptive"],
        final_u_robust=final["u_robust"],
        final_u_total=final["u_total"],
        final_control_cmd=final["control_cmd"],
        final_cable_tensions=final["cable_tensions"],
        u_final_norm=result["u_final_norm"],
    )


def _plot(
    *,
    out_dir: Path,
    reference: dict[str, np.ndarray],
    result: dict[str, Any],
    metrics: dict[str, Any],
    final_ee_meas: np.ndarray,
) -> list[str]:
    trials = result["trials"]
    trial_metrics = metrics["trials"]
    first = trials[0]
    final = trials[-1]
    t_ref = reference["t"]
    q_ref = reference["q_ref"]
    dq_ref = reference["dq_ref"]
    t_first = first["t"]
    t_final = final["t"]
    figures: list[str] = []

    # ── Figure 1: Convergence ──────────────────────────────────
    idx = np.arange(len(trials))
    rmse_ee = np.array([m["cartesian"]["rmse_ee"] for m in trial_metrics])
    rmse_q = np.array([m["rmse_q"] for m in trial_metrics])
    lifted = np.array([m["lifted_error_rms"] for m in trial_metrics])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].semilogy(idx, rmse_ee, "o-")
    axes[0].set_title("Cartesian RMSE")
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("m")
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(idx, rmse_q, "o-")
    axes[1].set_title("Joint RMSE")
    axes[1].set_xlabel("Trial")
    axes[1].grid(True, alpha=0.3)
    axes[2].semilogy(idx, lifted, "o-")
    axes[2].set_title("Lifted error RMS")
    axes[2].set_xlabel("Trial")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "figures" / "kilc_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    # ── Figure 2: Joint position tracking (trial 0 vs final) ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for row, (joint, ylabel) in enumerate([("qa", "qa (rad)"), ("qb", "qb (rad)")]):
        ax = axes[row]
        col = 0 if joint == "qa" else 1
        ax.plot(t_ref, q_ref[:, col], "k--", linewidth=1.0, alpha=0.7, label="ref")
        ax.plot(t_first, first["x_meas"][:, col], color="C3", linewidth=0.8, label="trial 0")
        ax.plot(t_final, final["x_meas"][:, col], color="C0", linewidth=1.2, label=f"trial {len(trials)-1}")
        e_first = first["x_meas"][:, col] - first["x_ref"][:, col]
        e_final = final["x_meas"][:, col] - final["x_ref"][:, col]
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        # inset error
        ax_in = ax.inset_axes([0.62, 0.12, 0.35, 0.30])
        ax_in.plot(t_first, e_first * 1000, color="C3", linewidth=0.6, label="trial 0 err")
        ax_in.plot(t_final, e_final * 1000, color="C0", linewidth=1.0, label="final err")
        ax_in.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_in.set_ylabel("err (mrad)")
        ax_in.grid(True, alpha=0.3)
        ax_in.legend(fontsize=6)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Position Tracking: Trial 0 vs Final", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "joint_tracking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    # ── Figure 3: Joint velocity tracking ──────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for row, (joint, ylabel) in enumerate([("dqa", "dqa (rad/s)"), ("dqb", "dqb (rad/s)")]):
        ax = axes[row]
        col = 0 if joint == "dqa" else 1
        ax.plot(t_ref, dq_ref[:, col], "k--", linewidth=1.0, alpha=0.7, label="ref")
        ax.plot(t_first, first["x_meas"][:, 2 + col], color="C3", linewidth=0.8, label="trial 0")
        ax.plot(t_final, final["x_meas"][:, 2 + col], color="C0", linewidth=1.2, label=f"trial {len(trials)-1}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Velocity Tracking: Trial 0 vs Final", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "joint_velocity_tracking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    # ── Figure 4: Tracking error (trial 0 vs final) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    labels = [("qa error", "rad"), ("qb error", "rad"), ("dqa error", "rad/s"), ("dqb error", "rad/s")]
    for i, (title, unit) in enumerate(labels):
        ax = axes[i // 2, i % 2]
        col = i if i < 2 else i - 2
        offset = 0 if i < 2 else 2
        e0 = first["x_meas"][:, offset + col] - first["x_ref"][:, offset + col]
        ef = final["x_meas"][:, offset + col] - final["x_ref"][:, offset + col]
        ax.plot(t_first, e0 * 1000, color="C3", linewidth=0.6, label="trial 0")
        ax.plot(t_final, ef * 1000, color="C0", linewidth=1.0, label=f"trial {len(trials)-1}")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{title} ({unit})" if "rad/s" not in title else f"{title} (m{unit})")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Tracking Errors: Trial 0 vs Final", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "tracking_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    # ── Figure 5: Control torques ──────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_common = t_final
    for row, (joint_idx, jname) in enumerate([(0, "Joint A (tau_a)"), (1, "Joint B (tau_b)")]):
        ax = axes[row]
        # feedback, feedforward, and total
        u_fb = final.get("u_fb", np.zeros_like(final["control_cmd"]))
        u_ilc = final.get("u_ilc", np.zeros_like(final["control_cmd"]))
        u_total = final["control_cmd"]
        ax.plot(t_common, u_fb[:, joint_idx], color="C2", linewidth=0.6, alpha=0.8, label="u_fb (PD)")
        ax.plot(t_common, u_ilc[:, joint_idx], color="C4", linewidth=0.6, alpha=0.8, label="u_ilc (ILC ff)")
        ax.plot(t_common, u_total[:, joint_idx], "k-", linewidth=1.2, label="u_total")
        ax.set_ylabel("Torque (Nm)")
        ax.set_title(jname)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Control Torques — Final Trial", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "control_torques.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    # ── Figure 6: Cable tensions ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    cables = final["cable_tensions"]  # (N, 8)
    cable_names = ["c11", "c12", "c13", "c14", "c21", "c22", "c23", "c24"]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    # Upper: all 8 cables
    for i in range(8):
        axes[0].plot(t_common, cables[:, i], color=colors[i], linewidth=0.5, label=cable_names[i])
    axes[0].set_ylabel("Tension (N)")
    axes[0].set_title("All 8 Cable Tensions")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=4, loc="upper right")
    # Lower: antagonistic pair example (c11,c12 vs c13,c14)
    for i in [0, 1, 2, 3]:
        axes[1].plot(t_common, cables[:, i], color=colors[i], linewidth=0.7, label=cable_names[i])
    axes[1].set_ylabel("Tension (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Joint A Cable Group (c11–c14)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=4)
    fig.suptitle("Cable Tensions — Final Trial", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "figures" / "cable_tensions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    # ── Figure 7: Cartesian circle ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(reference["ee_ref"][:, 0], reference["ee_ref"][:, 1],
            "k--", linewidth=1.5, label="ref circle")
    ax.plot(final_ee_meas[:, 0], final_ee_meas[:, 1],
            "C0", linewidth=1.2, label=f"trial {len(trials)-1}")
    ax.plot(reference["ee_ref"][0, 0], reference["ee_ref"][0, 1],
            "k*", markersize=10, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Cartesian Circle Tracking (final trial)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "figures" / "cartesian_tracking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(str(path))

    return figures


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    _override(config, args)
    out_dir = _make_output_dir(config, args)
    artifact_dir = Path(config["artifact_dir"])
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    xml_path = PROJECT_ROOT / config["xml_path"]
    reference, ik_cfg = _build_reference(config)
    np.savez_compressed(out_dir / "arrays" / "cartesian_reference.npz", **reference)

    model = load_continuous_dkuc_model(artifact_dir, str(config.get("device", "cpu")))
    result = run_yu_tan_kilc_tracking(
        model=model,
        reference=reference,
        kilc_config=_kilc_config(config),
        dt=float(config["dt"]),
        xml_path=xml_path,
        max_trials=int(config["max_trials"]),
        tau_limit=float(config["tau_limit"]),
        f_preload=float(config["f_preload"]),
        f_max_cable=None if config.get("f_max_cable") is None else float(config["f_max_cable"]),
        show_progress=not args.quiet,
    )
    ik_solver = MujocoSiteIK(xml_path, float(config["dt"]), ik_cfg)
    trial_metrics = [
        _trial_metrics(reference=reference, log=log, ik_solver=ik_solver)
        for log in result["trials"]
    ]
    final_log = result["trials"][-1]
    final_ee_meas = ik_solver.forward_xy_batch(final_log["x_meas"][:, :2])
    metrics = {
        "artifact_dir": str(artifact_dir),
        "trajectory": "cartesian_circle",
        "duration": float(config["reference"]["duration"]),
        "dt": float(config["dt"]),
        "max_trials": int(config["max_trials"]),
        "ik_config": asdict(ik_cfg),
        "trials": trial_metrics,
        "final": trial_metrics[-1],
    }
    if not args.no_plots:
        metrics["figures"] = _plot(
            out_dir=out_dir,
            reference=reference,
            result=result,
            metrics=metrics,
            final_ee_meas=final_ee_meas,
        )
    _save_arrays(
        out_dir=out_dir,
        reference=reference,
        result=result,
        final_ee_meas=final_ee_meas,
    )
    save_json(out_dir / "metrics" / "kilc_metrics.json", metrics)
    save_json(
        out_dir / "manifest.json",
        {
            "entry_module": "experiments.deployment_pipeline.run_kilc",
            "argv": sys.argv,
            "config": config,
            "python_executable": sys.executable,
            "git_branch": _git_value("branch", "--show-current"),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(json.dumps(metrics["final"], indent=2))
    print(f"Result directory: {out_dir}")


if __name__ == "__main__":
    main()
