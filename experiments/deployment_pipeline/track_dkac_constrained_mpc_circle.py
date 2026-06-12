"""Run DKAC circle tracking with cable-tension-constrained MPC."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments._paths import PROJECT_ROOT

from cdsm.kinematics.ik import IKConfig, MujocoSiteIK
from cdsm.references.cartesian import (
    CartesianReferenceConfig,
    generate_cartesian_reference,
)
from cdsm.runtime.tracking import run_dkac_tension_constrained_mpc
from koopman_control.control.finite_horizon_lqr import LqrConfig
from koopman_control.data.artifacts import save_json
from koopman_control.evaluation.tracking import (
    cartesian_tracking_metrics,
    tracking_metrics,
)
from koopman_control.models.registry import load_control_model
from koopman_control.visualization.plotting import (
    plot_cartesian_tracking_figures,
    plot_tracking_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a Cartesian circle with DKAC constrained MPC. "
            "No arbitrary equivalent-torque limit is applied; the MPC "
            "bounds come from cable tensions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--xml",
        default=str(
            PROJECT_ROOT
            / "assets"
            / "models"
            / "multi_joint_cable_driven_space_robot.xml"
        ),
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--out_dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "results"
            / "deployment_pipeline"
            / "constrained_mpc"
        ),
    )
    parser.add_argument("--tag", default="")

    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--num_cycles", type=float, default=1.0)
    parser.add_argument("--center_x", type=float, default=5.0)
    parser.add_argument("--center_y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.45)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--start_hold", type=float, default=1.0)
    parser.add_argument(
        "--time_scaling",
        choices=("linear", "quintic"),
        default="quintic",
    )

    parser.add_argument("--ik_site", default="end_effector")
    parser.add_argument("--ik_max_iter", type=int, default=120)
    parser.add_argument("--ik_tol", type=float, default=1e-5)
    parser.add_argument("--ik_damping", type=float, default=1e-4)
    parser.add_argument("--ik_max_step", type=float, default=0.08)
    parser.add_argument("--ik_joint_margin", type=float, default=0.05)
    parser.add_argument("--ik_smooth_window_s", type=float, default=0.03)
    parser.add_argument("--ik_seed_a", type=float, default=0.1)
    parser.add_argument("--ik_seed_b", type=float, default=-0.1)

    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--Qq", type=float, default=60.0)
    parser.add_argument("--Qdq", type=float, default=3.0)
    parser.add_argument("--R", type=float, default=1e-3)
    parser.add_argument("--Rd", type=float, default=2e-2)
    parser.add_argument("--f_preload", type=float, default=20.0)
    parser.add_argument(
        "--f_max_cable",
        type=float,
        default=1000.0,
        help="Maximum tension of each cable in N.",
    )
    return parser


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = Path(base_dir) / f"{stamp}_dkac_constrained_mpc_circle{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("metrics", "arrays", "figures", "media", "logs"):
        (output / name).mkdir()
    return output


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _build_reference(
    args: argparse.Namespace,
    ik_solver: MujocoSiteIK,
) -> dict[str, np.ndarray]:
    cartesian_config = CartesianReferenceConfig(
        kind="circle",
        dt=args.dt,
        period=args.period,
        num_cycles=args.num_cycles,
        center_x=args.center_x,
        center_y=args.center_y,
        radius_x=args.radius,
        radius_y=args.radius,
        phase=args.phase,
        start_hold=args.start_hold,
        time_scaling=args.time_scaling,
    )
    cartesian = generate_cartesian_reference(cartesian_config)
    inverse = ik_solver.solve_trajectory(
        np.asarray(cartesian["xy_ref"], dtype=np.float64)
    )
    return {
        "t": np.asarray(cartesian["t"], dtype=np.float64),
        "q_ref": np.asarray(inverse["q_ref"], dtype=np.float64),
        "dq_ref": np.asarray(inverse["dq_ref"], dtype=np.float64),
        "ee_ref": np.asarray(cartesian["xy_ref"], dtype=np.float64),
        "dxy_ref": np.asarray(cartesian["dxy_ref"], dtype=np.float64),
        "ee_ik": np.asarray(inverse["ee_ik"], dtype=np.float64),
        "ik_error": np.asarray(inverse["ik_error"], dtype=np.float64),
        "ik_converged": np.asarray(inverse["ik_converged"], dtype=bool),
        "ik_iterations": np.asarray(
            inverse["ik_iterations"],
            dtype=np.int32,
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    output = _make_output_dir(args.out_dir, args.tag)
    ik_config = IKConfig(
        site_name=args.ik_site,
        max_iter=args.ik_max_iter,
        tol=args.ik_tol,
        damping=args.ik_damping,
        max_step=args.ik_max_step,
        joint_margin=args.ik_joint_margin,
        smooth_window_s=args.ik_smooth_window_s,
        q_seed_a=args.ik_seed_a,
        q_seed_b=args.ik_seed_b,
    )
    controller_config = LqrConfig(
        horizon=args.horizon,
        Qq=args.Qq,
        Qdq=args.Qdq,
        R=args.R,
        Rd=args.Rd,
    )
    manifest = {
        "entry_module": (
            "experiments.deployment_pipeline."
            "track_dkac_constrained_mpc_circle"
        ),
        "argv": sys.argv[1:],
        "python_executable": sys.executable,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "artifact_dir": str(Path(args.artifact_dir)),
        "xml": str(Path(args.xml)),
        "controller": "DKAC tension-constrained Koopman MPC",
        "controller_config": asdict(controller_config),
        "ik_config": asdict(ik_config),
        "constraints": {
            "equivalent_torque_limit": None,
            "f_preload_n": args.f_preload,
            "f_max_cable_n": args.f_max_cable,
            "linearization": (
                "current-state DKAC control map and tendon Jacobian are "
                "frozen over each horizon and recomputed every control step"
            ),
        },
    }
    save_json(output / "manifest.json", manifest)

    ik_solver = MujocoSiteIK(args.xml, args.dt, ik_config)
    reference = _build_reference(args, ik_solver)
    np.savez_compressed(output / "arrays" / "cartesian_reference.npz", **reference)

    model = load_control_model(args.artifact_dir, "dkac", args.device)
    log = run_dkac_tension_constrained_mpc(
        model=model,
        reference=reference,
        controller_config=controller_config,
        xml_path=args.xml,
        dt=args.dt,
        f_preload=args.f_preload,
        f_max_cable=args.f_max_cable,
    )
    count = len(log["t"])
    ee_meas = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
    log["ee_meas"] = ee_meas
    log["ee_ref"] = reference["ee_ref"][:count]
    log["ee_ik"] = reference["ee_ik"][:count]
    np.savez_compressed(output / "arrays" / "closed_loop_dkac.npz", **log)

    values = tracking_metrics(log)
    values["cartesian"] = cartesian_tracking_metrics(
        ee_meas=ee_meas,
        ee_ref=reference["ee_ref"][:count],
        ik_error=reference["ik_error"][:count],
    )
    tensions = np.asarray(log["cable_tensions"], dtype=np.float64)
    residual = np.asarray(log["allocation_residual"], dtype=np.float64)
    statuses = Counter(str(item) for item in log["mpc_status"].tolist())
    values["constraints"] = {
        "f_preload_n": float(args.f_preload),
        "f_max_cable_n": float(args.f_max_cable),
        "minimum_cable_tension_n": float(np.min(tensions)),
        "maximum_cable_tension_n": float(np.max(tensions)),
        "lower_violation_count": int(
            np.sum(tensions < args.f_preload - 1e-6)
        ),
        "upper_violation_count": int(
            np.sum(tensions > args.f_max_cable + 1e-6)
        ),
        "maximum_allocation_residual_nm": float(
            np.max(np.abs(residual))
        ),
        "minimum_torque_lower_margin_nm": float(
            np.min(log["tau_cmd"] - log["torque_lower"])
        ),
        "minimum_torque_upper_margin_nm": float(
            np.min(log["torque_upper"] - log["tau_cmd"])
        ),
    }
    values["mpc_solver"] = {
        "status_counts": dict(statuses),
        "mean_iterations": float(np.mean(log["mpc_iterations"])),
        "max_iterations": int(np.max(log["mpc_iterations"])),
    }
    values["finite"] = {
        key: bool(np.all(np.isfinite(value)))
        for key, value in (
            ("states", log["x_meas"]),
            ("torques", log["tau_cmd"]),
            ("cable_tensions", tensions),
        )
    }
    metrics = {
        "trajectory": "circle",
        "artifact_dir": str(Path(args.artifact_dir)),
        "models": {"dkac": values},
    }
    figures = plot_tracking_figures(
        out_dir=output / "figures",
        logs={"dkac": log},
        metrics=metrics,
    )
    figures += plot_cartesian_tracking_figures(
        out_dir=output / "figures",
        logs={"dkac": log},
        ee_logs={"dkac": ee_meas},
        ref=reference,
        metrics=metrics,
    )
    metrics["figures"] = figures
    save_json(output / "metrics" / "tracking_metrics.json", metrics)

    print(f"[done] output={output}")
    print(
        "[metrics] "
        f"rmse_q={values['rmse_q']:.8f} rad, "
        f"rmse_ee={values['cartesian']['rmse_ee']:.8f} m, "
        f"peak_tension={values['peak_cable_tension']:.6f} N"
    )
    print(
        "[constraints] "
        f"upper_violations={values['constraints']['upper_violation_count']}, "
        f"allocation_residual="
        f"{values['constraints']['maximum_allocation_residual_nm']:.3e} Nm"
    )


if __name__ == "__main__":
    main()
