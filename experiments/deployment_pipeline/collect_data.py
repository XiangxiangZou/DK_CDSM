"""Collect a reusable CDSM MuJoCo trajectory dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from experiments._paths import PROJECT_ROOT

from cdsm.data_collection import (
    CollectionConfig,
    collect_pd_control,
    collect_random_excitation,
)
from cdsm.plants.mujoco import MujocoCablePlant
from koopman_control.data.artifacts import save_json
from koopman_control.data.datasets import save_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect CDSM MuJoCo data using random excitation or PD tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--random", action="store_true")
    mode.add_argument("--PDCtrl", "--pdctrl", dest="PDCtrl", action="store_true")
    parser.add_argument(
        "--xml",
        default=str(
            PROJECT_ROOT
            / "assets"
            / "models"
            / "multi_joint_cable_driven_space_robot.xml"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "data"
            / "raw"
            / "deployment_pipeline"
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--traj", type=int, default=120)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q_limit_ratio", type=float, default=0.90)
    parser.add_argument("--q_init_ratio", type=float, default=0.65)
    parser.add_argument("--dq_init_range", type=float, default=0.4)
    parser.add_argument("--tau_max", type=float, default=80.0)
    parser.add_argument("--boundary_kp", type=float, default=80.0)
    parser.add_argument("--boundary_kd", type=float, default=6.0)
    parser.add_argument("--random_tau", type=float, default=35.0)
    parser.add_argument("--random_hold_steps", type=int, default=8)
    parser.add_argument("--random_damping", type=float, default=0.8)
    parser.add_argument("--amp_min", type=float, default=0.15)
    parser.add_argument("--amp_max", type=float, default=0.55)
    parser.add_argument("--omega_min", type=float, default=0.7)
    parser.add_argument("--omega_max", type=float, default=2.3)
    parser.add_argument("--kp_a", type=float, default=80.0)
    parser.add_argument("--kp_b", type=float, default=70.0)
    parser.add_argument("--kd_a", type=float, default=8.0)
    parser.add_argument("--kd_b", type=float, default=7.0)
    parser.add_argument("--f_preload", type=float, default=50.0)
    parser.add_argument(
        "--f_max_cable",
        type=float,
        default=2000.0,
        help="Per-cable tension limit; use a negative value to disable it.",
    )
    return parser


def _make_output_dir(base_dir: str | Path, mode: str, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = Path(base_dir) / f"{stamp}_{mode}{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _make_config(args: argparse.Namespace) -> CollectionConfig:
    return CollectionConfig(
        traj_count=args.traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        q_limit_ratio=args.q_limit_ratio,
        q_init_ratio=args.q_init_ratio,
        dq_init_range=args.dq_init_range,
        random_tau=args.random_tau,
        random_hold_steps=args.random_hold_steps,
        random_damping=args.random_damping,
        boundary_kp=args.boundary_kp,
        boundary_kd=args.boundary_kd,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        kp_a=args.kp_a,
        kp_b=args.kp_b,
        kd_a=args.kd_a,
        kd_b=args.kd_b,
        tau_max=args.tau_max,
        f_preload=args.f_preload,
        f_max_cable=(
            None if args.f_max_cable < 0.0 else args.f_max_cable
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    mode = "random" if args.random else "PDCtrl"
    config = _make_config(args)
    output = _make_output_dir(args.out_dir, mode, args.tag)
    plant = MujocoCablePlant(args.xml, config.dt)
    if args.random:
        arrays, metadata = collect_random_excitation(plant, config)
    else:
        arrays, metadata = collect_pd_control(plant, config)

    dataset_path = output / "dataset.npz"
    save_dataset(dataset_path, arrays)
    save_json(
        output / "meta.json",
        {
            **metadata,
            "xml": str(Path(args.xml).resolve()),
            "output_dir": str(output.resolve()),
            "dataset_file": dataset_path.name,
            "collection_config": asdict(config),
        },
    )
    print(f"states={arrays['states'].shape}, inputs={arrays['inputs'].shape}")
    print(f"[done] dataset -> {dataset_path}")


if __name__ == "__main__":
    main()
