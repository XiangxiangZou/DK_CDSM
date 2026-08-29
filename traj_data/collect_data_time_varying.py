"""Collect chronological CDSM streams with an unobserved sinusoidal torque."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np

try:  # Support both file-path and ``python -m traj_data...`` execution.
    from .data_io import save_dataset, save_json
    from .mujoco_cdsm import CABLE_NAMES, INPUT_ORDER, STATE_ORDER, MujocoCDSM
    from .references import random_joint_reference, sample_initial_state, shrink_limits, soft_limit_guard
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from data_io import save_dataset, save_json
    from mujoco_cdsm import CABLE_NAMES, INPUT_ORDER, STATE_ORDER, MujocoCDSM
    from references import random_joint_reference, sample_initial_state, shrink_limits, soft_limit_guard

ROOT = Path(__file__).resolve().parent
DEFAULT_XML = ROOT / "assets" / "multi_joint_cable_driven_space_robot.xml"
REFERENCE_RANGE_RATIO = 0.30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect strictly ordered CDSM trajectories under an absolute-time sinusoidal disturbance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run_type", choices=["smoke_test", "full_run"], default="full_run")
    parser.add_argument("--tag", default="")
    parser.add_argument("--traj", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--disturbance_amplitude", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--disturbance_frequency", type=float, nargs=2, default=(0.2, 0.3))
    parser.add_argument("--disturbance_phase", type=float, nargs=2, default=(0.0, 1.5707963267948966))
    parser.add_argument("--kp", type=float, nargs=2, default=(80.0, 70.0))
    parser.add_argument("--kd", type=float, nargs=2, default=(8.0, 7.0))
    parser.add_argument("--limit_ratio", type=float, default=0.92)
    return parser


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT.parent, check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if args.traj <= 0 or args.steps <= 0 or args.warmup_steps < 0 or args.dt <= 0:
        raise ValueError("trajectory, step, warm-up, and dt settings are invalid")
    rng = np.random.default_rng(args.seed)
    robot = MujocoCDSM(args.xml, args.dt)
    limits = shrink_limits(robot.q_limits, args.limit_ratio)
    states = np.empty((args.traj, args.steps + 1, 4), dtype=np.float64)
    inputs = np.empty((args.traj, args.steps, 2), dtype=np.float64)
    disturbance = np.empty_like(inputs)
    cable_ctrl = np.empty((args.traj, args.steps, len(CABLE_NAMES)), dtype=np.float64)
    t = np.arange(args.steps + 1, dtype=np.float64) * args.dt
    amplitude = np.asarray(args.disturbance_amplitude, dtype=np.float64)
    frequency = np.asarray(args.disturbance_frequency, dtype=np.float64)
    phase = np.asarray(args.disturbance_phase, dtype=np.float64)
    kp, kd = np.asarray(args.kp), np.asarray(args.kd)
    for trial in range(args.traj):
        q0, dq0 = sample_initial_state(rng, limits, q_init_ratio=0.65, dq_init_range=0.4)
        robot.set_state(q0, dq0)
        q_ref, dq_ref, _ = random_joint_reference(rng, limits, steps=args.steps + args.warmup_steps,
                                                  dt=args.dt, q_start=q0, waypoint_count=9,
                                                  range_ratio=REFERENCE_RANGE_RATIO)
        for warm in range(args.warmup_steps):
            x = robot.read_state()
            tau = kp * (q_ref[warm] - x[:2]) + kd * (dq_ref[warm] - x[2:])
            tau += soft_limit_guard(x[:2], x[2:], limits, kp=80.0, kd=6.0)
            robot.apply_cable_tensions(robot.torque_to_tensions(tau))
            robot.apply_joint_disturbance(np.zeros(2))
            robot.step()
        states[trial, 0] = robot.read_state()
        for step in range(args.steps):
            x = robot.read_state()
            index = args.warmup_steps + step
            tau = kp * (q_ref[index] - x[:2]) + kd * (dq_ref[index] - x[2:])
            tau += soft_limit_guard(x[:2], x[2:], limits, kp=80.0, kd=6.0)
            tensions = robot.torque_to_tensions(tau)
            applied = amplitude * np.sin(2.0 * np.pi * frequency * t[step] + phase)
            robot.apply_cable_tensions(tensions)
            robot.apply_joint_disturbance(applied)
            robot.step()
            inputs[trial, step], disturbance[trial, step] = tau, applied
            cable_ctrl[trial, step], states[trial, step + 1] = tensions, robot.read_state()
    arrays = {"t": t, "states": states, "inputs": inputs,
              "disturbance_torque": disturbance, "cable_ctrl": cable_ctrl}
    metadata = {"mode": "time_varying_sinusoidal_joint_disturbance",
                "disturbance_formula": "amplitude*sin(2*pi*frequency*t+phase)",
                "disturbance_is_model_input": False, "state_order": STATE_ORDER,
                "input_order": INPUT_ORDER, "cable_order": CABLE_NAMES,
                "xml": str(Path(args.xml).resolve()), "seed": args.seed, "dt": args.dt,
                "reference_range_ratio": REFERENCE_RANGE_RATIO,
                "safe_joint_limits_rad": limits,
                "parameters": vars(args), "git_branch": _git_value("branch", "--show-current"),
                "git_commit": _git_value("rev-parse", "HEAD")}
    return arrays, metadata


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    output = args.out_dir / args.run_type / f"{stamp}_time_varying_sine{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    arrays, metadata = collect(args)
    summary = save_dataset(output / "dataset.npz", arrays)
    limits = np.asarray(metadata["safe_joint_limits_rad"])
    positions = arrays["states"][..., :2]
    summary.update({"peak_abs_disturbance_torque": float(np.max(np.abs(arrays["disturbance_torque"]))),
                    "strict_time_increasing": bool(np.all(np.diff(arrays["t"]) > 0)),
                    "joint_limit_violation": bool(np.any(positions < limits[:, 0]) or
                                                  np.any(positions > limits[:, 1]))})
    save_json(output / "metadata.json", metadata)
    save_json(output / "summary.json", summary)
    print(f"[done] time-varying dataset -> {output / 'dataset.npz'}")


if __name__ == "__main__":
    main()
