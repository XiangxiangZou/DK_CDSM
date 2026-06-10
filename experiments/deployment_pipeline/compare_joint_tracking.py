"""Compare EDMD, DKUC, and DKAC joint-space tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments._paths import PROJECT_ROOT

from cdsm.runtime.tracking import run_joint_space_closed_loop_model
from koopman_control.control.finite_horizon_lqr import (
    LqrConfig,
    build_ramp_reference,
)
from koopman_control.data.artifacts import save_json
from koopman_control.evaluation.tracking import (
    logs_to_npz_payload,
    tracking_metrics,
)
from koopman_control.models.registry import (
    load_control_model,
    normalize_model_list,
)
from koopman_control.visualization.plotting import plot_tracking_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Koopman LQR joint tracking in MuJoCo.",
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
            / "tracking"
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--T_track", type=float, default=4.0)
    parser.add_argument("--T_ramp", type=float, default=2.0)
    parser.add_argument("--qa0", type=float, default=0.0)
    parser.add_argument("--qa1", type=float, default=0.6)
    parser.add_argument("--qb0", type=float, default=0.0)
    parser.add_argument("--qb1", type=float, default=-0.45)
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
    parser.add_argument("--no_plots", action="store_true")
    return parser


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = Path(base_dir) / f"{stamp}_tracking{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    models = normalize_model_list(args.models, control_only=True)
    output = _make_output_dir(args.out_dir, args.tag)
    controller_config = LqrConfig(
        args.horizon,
        args.Qq,
        args.Qdq,
        args.R,
        args.Rd,
    )
    ramp = build_ramp_reference(
        dt=args.dt,
        duration=args.T_track,
        start=[args.qa0, args.qb0],
        target=[args.qa1, args.qb1],
        ramp_duration=args.T_ramp,
    )
    reference = {
        "t": ramp["t"],
        "q_ref": ramp["values"],
        "dq_ref": ramp["rates"],
    }
    logs: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, object] = {
        "artifact_dir": str(artifact_dir),
        "models": {},
        "lqr_config": asdict(controller_config),
        "reference": {
            "T_track": args.T_track,
            "T_ramp": args.T_ramp,
            "qa0": args.qa0,
            "qa1": args.qa1,
            "qb0": args.qb0,
            "qb1": args.qb1,
        },
        "note": "DKN is prediction-only in this linear LQR comparison.",
    }
    model_metrics = metrics["models"]
    assert isinstance(model_metrics, dict)

    for model_name in models:
        print(f"[tracking] {model_name}")
        model = load_control_model(artifact_dir, model_name, args.device)
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
        logs[model_name] = log
        model_metrics[model_name] = tracking_metrics(log)
        np.savez_compressed(
            output / f"closed_loop_{model_name}.npz",
            **log,
        )
        print(f"  rmse_q={model_metrics[model_name]['rmse_q']:.6g}")

    np.savez_compressed(
        output / "closed_loop_all_models.npz",
        **logs_to_npz_payload(logs),
    )
    if not args.no_plots:
        metrics["figures"] = plot_tracking_figures(
            out_dir=output,
            logs=logs,
            metrics=metrics,
        )
    save_json(output / "tracking_metrics.json", metrics)
    print(f"[done] tracking results -> {output}")


if __name__ == "__main__":
    main()
