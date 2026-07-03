"""Collect uncontrolled CDSM trajectory data in MuJoCo."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from data_io import save_dataset, save_json
from mujoco_cdsm import CABLE_NAMES, CABLE_TENSION_LOWER_BOUND, INPUT_ORDER, STATE_ORDER, MujocoCDSM
from references import sample_initial_state, shrink_limits, soft_limit_guard


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = ROOT / "assets" / "multi_joint_cable_driven_space_robot.xml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect uncontrolled CDSM trajectory data. Use random for open-loop "
            "random torque excitation without PD tracking, or passive for free "
            "response with zero desired torque."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("random", "passive"), default="random")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--traj", type=int, default=40)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--limit_ratio",
        type=float,
        default=0.92,
        help="Fraction of XML physical joint limits used for safe collection.",
    )
    parser.add_argument("--q_init_ratio", type=float, default=0.65)
    parser.add_argument("--dq_init_range", type=float, default=0.4)
    parser.add_argument("--random_tau", type=float, default=35.0)
    parser.add_argument("--random_hold_steps", type=int, default=8)
    parser.add_argument("--random_damping", type=float, default=0.8)
    parser.add_argument("--guard_kp", type=float, default=80.0)
    parser.add_argument("--guard_kd", type=float, default=6.0)
    return parser


def _make_output_dir(base: Path, mode: str, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = base / f"{stamp}_uncontrolled_{mode}{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _collect(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    robot = MujocoCDSM(args.xml, args.dt)
    safe_limits = shrink_limits(robot.q_limits, args.limit_ratio)

    n_traj, steps = int(args.traj), int(args.steps)
    states = np.zeros((n_traj, steps + 1, 4), dtype=np.float64)
    inputs = np.zeros((n_traj, steps, 2), dtype=np.float64)
    q_ref_all = np.full((n_traj, steps, 2), np.nan, dtype=np.float64)
    dq_ref_all = np.full((n_traj, steps, 2), np.nan, dtype=np.float64)
    cable_ctrl = np.zeros((n_traj, steps, len(CABLE_NAMES)), dtype=np.float64)
    t = np.arange(steps, dtype=np.float64) * float(args.dt)

    for traj_idx in range(n_traj):
        q0, dq0 = sample_initial_state(
            rng,
            safe_limits,
            q_init_ratio=args.q_init_ratio,
            dq_init_range=args.dq_init_range,
        )
        robot.set_state(q0, dq0)
        states[traj_idx, 0] = robot.read_state()

        tau_hold = rng.uniform(-args.random_tau, args.random_tau, size=2)
        for step in range(steps):
            x = robot.read_state()
            q = x[:2]
            dq = x[2:]
            if args.mode == "random":
                if step % max(1, args.random_hold_steps) == 0:
                    tau_hold = rng.uniform(-args.random_tau, args.random_tau, size=2)
                tau = tau_hold - args.random_damping * dq
                tau += soft_limit_guard(q, dq, safe_limits, kp=args.guard_kp, kd=args.guard_kd)
            else:
                tau = np.zeros(2, dtype=np.float64)

            tensions = robot.torque_to_tensions(tau)
            robot.apply_cable_tensions(tensions)
            robot.step()

            inputs[traj_idx, step] = tau
            cable_ctrl[traj_idx, step] = tensions
            states[traj_idx, step + 1] = robot.read_state()

    arrays = {
        "t": t,
        "states": states,
        "inputs": inputs,
        "q_ref": q_ref_all,
        "dq_ref": dq_ref_all,
        "cable_ctrl": cable_ctrl,
    }
    metadata = {
        "mode": f"uncontrolled_{args.mode}",
        "xml": str(Path(args.xml).resolve()),
        "dt": float(args.dt),
        "seed": int(args.seed),
        "state_order": STATE_ORDER,
        "input_order": INPUT_ORDER,
        "cable_order": CABLE_NAMES,
        "xml_joint_limits_rad": robot.q_limits,
        "safe_joint_limits_rad": safe_limits,
        "safe_joint_limits_deg": np.degrees(safe_limits),
        "cable_tension_lower_bound_n": CABLE_TENSION_LOWER_BOUND,
        "cable_tension_upper_bound_n": None,
        "joint_torque_limit_n_m": None,
        "description": {
            "random": "Random torque excitation without PD tracking; soft limit guard is only for safety.",
            "passive": "Zero desired joint torque response from random initial states.",
        }[args.mode],
        "parameters": vars(args),
    }
    return arrays, metadata


def main() -> None:
    args = build_parser().parse_args()
    output = _make_output_dir(args.out_dir, args.mode, args.tag)
    arrays, metadata = _collect(args)
    dataset_path = output / "dataset.npz"
    summary = save_dataset(dataset_path, arrays)
    save_json(output / "metadata.json", {**metadata, "output_dir": output, "dataset_file": dataset_path.name})
    save_json(output / "summary.json", summary)

    print(f"mode=uncontrolled_{args.mode}")
    print(f"dataset={dataset_path}")
    print(f"states={arrays['states'].shape}, inputs={arrays['inputs'].shape}")
    print(f"safe_limits_deg={np.degrees(metadata['safe_joint_limits_rad'])}")
    print(f"peak_abs_tau={summary['peak_abs_tau']:.6g}, peak_cable_tension={summary['peak_cable_tension']:.6g}")


if __name__ == "__main__":
    main()
