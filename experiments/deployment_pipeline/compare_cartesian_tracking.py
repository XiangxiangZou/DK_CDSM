"""Compare Cartesian IK tracking with EDMD, DKUC, and DKAC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments._paths import PROJECT_ROOT

from cdsm.kinematics.ik import IKConfig, MujocoSiteIK
from cdsm.references.cartesian import (
    TIME_SCALINGS,
    TRAJECTORY_KINDS,
    CartesianReferenceConfig,
    generate_cartesian_reference,
)
from cdsm.runtime.tracking import run_joint_space_closed_loop_model
from koopman_control.control.finite_horizon_lqr import LqrConfig
from koopman_control.data.artifacts import save_json
from koopman_control.evaluation.tracking import (
    cartesian_tracking_metrics,
    logs_to_npz_payload,
    tracking_metrics,
)
from koopman_control.models.registry import (
    load_control_model,
    normalize_model_list,
)
from koopman_control.visualization.plotting import (
    plot_cartesian_tracking_figures,
    plot_tracking_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Cartesian references, solve IK, and compare Koopman controllers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
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
            / "cartesian_tracking"
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument(
        "--trajectories",
        nargs="+",
        choices=TRAJECTORY_KINDS,
        default=["figure8", "circle"],
    )
    parser.add_argument("--period", type=float, default=8.0)
    parser.add_argument("--num_cycles", type=float, default=1.0)
    parser.add_argument("--center_x", type=float, default=5.0)
    parser.add_argument("--center_y", type=float, default=0.0)
    parser.add_argument("--radius_x", type=float, default=0.6)
    parser.add_argument("--radius_y", type=float, default=0.35)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--start_hold", type=float, default=0.0)
    parser.add_argument(
        "--time_scaling",
        choices=TIME_SCALINGS,
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
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--Qq", type=float, default=40.0)
    parser.add_argument("--Qdq", type=float, default=2.0)
    parser.add_argument("--R", type=float, default=1e-3)
    parser.add_argument("--Rd", type=float, default=1e-2)
    parser.add_argument("--tau_limit", type=float, default=120.0)
    parser.add_argument("--f_preload", type=float, default=50.0)
    parser.add_argument(
        "--f_max_cable",
        type=float,
        default=2000.0,
        help="Per-cable tension limit; use a negative value to disable it.",
    )
    return parser


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = Path(base_dir) / f"{stamp}_cartesian_ik_tracking{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _build_reference(
    trajectory: str,
    args: argparse.Namespace,
    ik_solver: MujocoSiteIK,
) -> dict[str, np.ndarray]:
    config = CartesianReferenceConfig(
        kind=trajectory,
        dt=args.dt,
        period=args.period,
        num_cycles=args.num_cycles,
        center_x=args.center_x,
        center_y=args.center_y,
        radius_x=args.radius_x,
        radius_y=args.radius_y,
        phase=args.phase,
        start_hold=args.start_hold,
        time_scaling=args.time_scaling,
    )
    cartesian = generate_cartesian_reference(config)
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


def _ik_summary(reference: dict[str, np.ndarray]) -> dict[str, object]:
    error = reference["ik_error"]
    converged = reference["ik_converged"]
    q_ref = reference["q_ref"]
    return {
        "points": int(converged.shape[0]),
        "not_converged": int(np.sum(~converged)),
        "rmse_ik": float(np.sqrt(np.mean(error * error))),
        "max_abs_ik_error": float(np.max(np.abs(error))),
        "qa_min": float(np.min(q_ref[:, 0])),
        "qa_max": float(np.max(q_ref[:, 0])),
        "qb_min": float(np.min(q_ref[:, 1])),
        "qb_max": float(np.max(q_ref[:, 1])),
    }


def _run_trajectory(
    *,
    trajectory: str,
    root_dir: Path,
    args: argparse.Namespace,
    models: list[str],
    controller_config: LqrConfig,
    ik_config: IKConfig,
) -> dict[str, object]:
    output = root_dir / trajectory
    output.mkdir(parents=True, exist_ok=False)
    ik_solver = MujocoSiteIK(args.xml, args.dt, ik_config)
    reference = _build_reference(trajectory, args, ik_solver)
    np.savez_compressed(
        output / "cartesian_ik_reference.npz",
        **reference,
    )
    metrics: dict[str, object] = {
        "artifact_dir": str(Path(args.artifact_dir)),
        "trajectory": trajectory,
        "cartesian_reference": {
            "period": args.period,
            "num_cycles": args.num_cycles,
            "center_x": args.center_x,
            "center_y": args.center_y,
            "radius_x": args.radius_x,
            "radius_y": args.radius_y,
            "phase": args.phase,
            "start_hold": args.start_hold,
            "time_scaling": args.time_scaling,
            "dt": args.dt,
        },
        "ik_config": asdict(ik_config),
        "ik_summary": _ik_summary(reference),
        "lqr_config": asdict(controller_config),
        "models": {},
        "note": "DKN is prediction-only in this linear LQR comparison.",
    }
    logs: dict[str, dict[str, np.ndarray]] = {}
    ee_logs: dict[str, np.ndarray] = {}
    model_metrics = metrics["models"]
    assert isinstance(model_metrics, dict)

    for model_name in models:
        print(f"[tracking] {trajectory}/{model_name}")
        model = load_control_model(
            args.artifact_dir,
            model_name,
            args.device,
        )
        log = run_joint_space_closed_loop_model(
            model=model,
            reference=reference,
            controller_config=controller_config,
            tau_limit=args.tau_limit,
            xml_path=args.xml,
            dt=args.dt,
            f_preload=args.f_preload,
            f_max_cable=(
                None if args.f_max_cable < 0.0 else args.f_max_cable
            ),
        )
        count = len(log["t"])
        ee_meas = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
        log["ee_meas"] = ee_meas
        log["ee_ref"] = reference["ee_ref"][:count]
        log["ee_ik"] = reference["ee_ik"][:count]
        logs[model_name] = log
        ee_logs[model_name] = ee_meas

        values = tracking_metrics(log)
        values["cartesian"] = cartesian_tracking_metrics(
            ee_meas=ee_meas,
            ee_ref=reference["ee_ref"][:count],
            ik_error=reference["ik_error"][:count],
        )
        model_metrics[model_name] = values
        np.savez_compressed(
            output / f"closed_loop_{model_name}.npz",
            **log,
        )
        print(
            f"  rmse_q={values['rmse_q']:.6g}, "
            f"rmse_ee={values['cartesian']['rmse_ee']:.6g} m"
        )

    np.savez_compressed(
        output / "closed_loop_all_models.npz",
        **logs_to_npz_payload(logs),
    )
    if not args.no_plots:
        figures = plot_tracking_figures(
            out_dir=output,
            logs=logs,
            metrics=metrics,
        )
        figures += plot_cartesian_tracking_figures(
            out_dir=output,
            logs=logs,
            ee_logs=ee_logs,
            ref=reference,
            metrics=metrics,
        )
        metrics["figures"] = figures
    save_json(output / "cartesian_tracking_metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    models = normalize_model_list(args.models, control_only=True)
    controller_config = LqrConfig(
        args.horizon,
        args.Qq,
        args.Qdq,
        args.R,
        args.Rd,
    )
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
    output = _make_output_dir(args.out_dir, args.tag)
    summary: dict[str, object] = {
        "artifact_dir": str(Path(args.artifact_dir)),
        "models": models,
        "trajectories": {},
        "output_dir": str(output),
    }
    trajectories = summary["trajectories"]
    assert isinstance(trajectories, dict)
    for trajectory in args.trajectories:
        trajectories[trajectory] = _run_trajectory(
            trajectory=trajectory,
            root_dir=output,
            args=args,
            models=models,
            controller_config=controller_config,
            ik_config=ik_config,
        )
    save_json(output / "cartesian_tracking_summary.json", summary)
    print(f"[done] Cartesian IK tracking results -> {output}")


if __name__ == "__main__":
    main()
