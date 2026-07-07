"""PD sine tracking for two XL330 servos with MuJoCo visualization.

This is a hardware-facing script.  It drives two DYNAMIXEL servos in extended
position mode while mirroring the measured joint angles into the two-joint
MuJoCo model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.common.xl330_mujoco import (
    ADDR_GOAL_POSITION,
    ADDR_PRESENT_CURRENT,
    ADDR_PRESENT_POSITION,
    ADDR_PRESENT_PWM,
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    ADDR_TORQUE_ENABLE,
    CURRENT_UNIT_MA,
    DEFAULT_XML,
    DynamixelError,
    PWM_UNIT_PERCENT,
    RAD_PER_TICK,
    TORQUE_DISABLE,
    TORQUE_ENABLE,
    build_joint_indices,
    configure_viewer_camera,
    create_sync_goal_writer,
    create_sync_position_reader,
    effective_joint_limits,
    initialize_servos,
    int32_to_uint32,
    joint_angle_to_ticks,
    load_mujoco,
    open_dynamixel,
    read2,
    read4,
    read_active_joint_limits,
    sync_read_ticks,
    sync_write_goal_ticks,
    ticks_to_joint_angles,
    uint32_to_int32,
    write1,
    write4,
    write_joint_angles,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "hardware" / "outputs" / "two_servo_pd_sine"
DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 57600
DEFAULT_IDS = [10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track safe sinusoidal joint references on two XL330 servos and "
            "mirror measured angles into the MuJoCo two-joint model."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--ids", type=int, nargs=2, default=DEFAULT_IDS)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--signs", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument(
        "--zero-ticks",
        type=int,
        nargs=2,
        default=None,
        help="Optional servo tick values corresponding to q=[0,0]. Defaults to startup ticks.",
    )
    parser.add_argument(
        "--limit-ratio",
        type=float,
        default=0.80,
        help="Shrink XML joint limits around their centers before planning.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional symmetric absolute joint limit in radians, intersected with XML limits.",
    )
    parser.add_argument("--biases", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--amplitudes", type=float, nargs=2, default=[1.00, 1.00])
    parser.add_argument("--frequencies", type=float, nargs=2, default=[0.10, 0.10])
    parser.add_argument("--phases", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument(
        "--kp",
        type=float,
        default=40.0,
        help="Outer-loop proportional gain for commanded position increments.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=1.00,
        help="Outer-loop derivative gain for commanded position increments.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.10,
        help="Maximum commanded joint-position change per control cycle in radians.",
    )
    parser.add_argument("--profile-velocity", type=int, default=400)
    parser.add_argument("--profile-acceleration", type=int, default=100)
    parser.add_argument(
        "--feedback-stride",
        type=int,
        default=1,
        help="Read current/PWM every N control cycles and hold the last value between reads.",
    )
    parser.add_argument(
        "--comm-retries",
        type=int,
        default=3,
        help="Retry count for transient DYNAMIXEL packet errors.",
    )
    parser.add_argument(
        "--comm-retry-delay",
        type=float,
        default=0.02,
        help="Delay in seconds between DYNAMIXEL communication retries.",
    )
    parser.add_argument(
        "--torque-constant-nm-per-a",
        type=float,
        default=0.354,
        help=(
            "Optional calibrated torque constant used to convert Present Current "
            "to estimated torque. Default is the XL330-M288 5V stall torque/current ratio."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Validate model and trajectory without opening COM.")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the RUN confirmation before torque is enabled.",
    )
    parser.add_argument("--camera-lookat-x", type=float, default=3.2)
    parser.add_argument("--camera-lookat-y", type=float, default=0.0)
    parser.add_argument("--camera-lookat-z", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=6.4)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-90.0)
    return parser.parse_args()


def make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    (run_dir / "arrays").mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reference_at(t: float, biases: np.ndarray, amplitudes: np.ndarray,
                 frequencies: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * math.pi * frequencies
    arg = omega * float(t) + phases
    q_ref = biases + amplitudes * np.sin(arg)
    dq_ref = amplitudes * omega * np.cos(arg)
    return q_ref, dq_ref


def validate_reference(args: argparse.Namespace, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    biases = np.asarray(args.biases, dtype=np.float64)
    amplitudes = np.abs(np.asarray(args.amplitudes, dtype=np.float64))
    frequencies = np.asarray(args.frequencies, dtype=np.float64)
    phases = np.asarray(args.phases, dtype=np.float64)

    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.rate <= 0.0:
        raise ValueError("--rate must be positive.")
    if args.kp < 0.0 or args.kd < 0.0:
        raise ValueError("--kp and --kd must be non-negative.")
    if args.max_step <= 0.0:
        raise ValueError("--max-step must be positive.")
    if args.feedback_stride <= 0:
        raise ValueError("--feedback-stride must be positive.")
    if args.comm_retries < 0:
        raise ValueError("--comm-retries must be non-negative.")
    if args.comm_retry_delay < 0.0:
        raise ValueError("--comm-retry-delay must be non-negative.")
    if np.any(frequencies <= 0.0):
        raise ValueError("--frequencies must be positive.")

    ref_min = biases - amplitudes
    ref_max = biases + amplitudes
    if np.any(ref_min < lower) or np.any(ref_max > upper):
        raise ValueError(
            "Sine reference exceeds effective limits: "
            f"ref_min={ref_min.tolist()}, ref_max={ref_max.tolist()}, "
            f"lower={lower.tolist()}, upper={upper.tolist()}"
        )

    return {
        "biases": biases,
        "amplitudes": amplitudes,
        "frequencies": frequencies,
        "phases": phases,
        "ref_min": ref_min,
        "ref_max": ref_max,
    }


def retry_dxl(action: Any, *, retries: int, delay_s: float, label: str) -> Any:
    last_error: DynamixelError | None = None
    for attempt in range(retries + 1):
        try:
            return action()
        except DynamixelError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"Warning: {label} failed ({exc}); retry {attempt + 1}/{retries}")
            time.sleep(delay_s)
    if last_error is None:
        raise RuntimeError(f"{label} failed without a captured error.")
    raise last_error


def read_ticks(port: Any, packet: Any, ids: list[int], *, retries: int = 0,
               delay_s: float = 0.0) -> np.ndarray:
    return np.array(
        [
            uint32_to_int32(
                retry_dxl(
                    lambda dxl_id=dxl_id: read4(
                        port, packet, dxl_id, ADDR_PRESENT_POSITION, "present position"
                    ),
                    retries=retries,
                    delay_s=delay_s,
                    label=f"read present position from ID {dxl_id}",
                )
            )
            for dxl_id in ids
        ],
        dtype=np.int64,
    )


def uint16_to_int16(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def read_motor_feedback(port: Any, packet: Any, ids: list[int], *, retries: int = 0,
                        delay_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    current_ma = np.array(
        [
            uint16_to_int16(
                retry_dxl(
                    lambda dxl_id=dxl_id: read2(
                        port, packet, dxl_id, ADDR_PRESENT_CURRENT, "present current"
                    ),
                    retries=retries,
                    delay_s=delay_s,
                    label=f"read present current from ID {dxl_id}",
                )
            )
            * CURRENT_UNIT_MA
            for dxl_id in ids
        ],
        dtype=np.float64,
    )
    pwm_percent = np.array(
        [
            uint16_to_int16(
                retry_dxl(
                    lambda dxl_id=dxl_id: read2(
                        port, packet, dxl_id, ADDR_PRESENT_PWM, "present pwm"
                    ),
                    retries=retries,
                    delay_s=delay_s,
                    label=f"read present pwm from ID {dxl_id}",
                )
            )
            * PWM_UNIT_PERCENT
            for dxl_id in ids
        ],
        dtype=np.float64,
    )
    return current_ma, pwm_percent


def write_goal_ticks(port: Any, packet: Any, ids: list[int], goal_ticks: np.ndarray,
                     *, retries: int = 0, delay_s: float = 0.0) -> None:
    for dxl_id, tick in zip(ids, goal_ticks):
        retry_dxl(
            lambda dxl_id=dxl_id, tick=tick: write4(
                port,
                packet,
                dxl_id,
                ADDR_GOAL_POSITION,
                int32_to_uint32(int(tick)),
                "goal position",
            ),
            retries=retries,
            delay_s=delay_s,
            label=f"write goal position to ID {dxl_id}",
        )


def joint_angles_to_ticks(q: np.ndarray, zero_ticks: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.array(
        [
            joint_angle_to_ticks(float(angle), int(zero), float(sign))
            for angle, zero, sign in zip(q, zero_ticks, signs)
        ],
        dtype=np.int64,
    )


def configure_servo_motion(port: Any, packet: Any, ids: list[int],
                           profile_velocity: int, profile_acceleration: int,
                           *, retries: int = 0, delay_s: float = 0.0) -> None:
    for dxl_id in ids:
        retry_dxl(
            lambda dxl_id=dxl_id: write4(
                port, packet, dxl_id, ADDR_PROFILE_ACCELERATION,
                profile_acceleration, "profile acceleration",
            ),
            retries=retries,
            delay_s=delay_s,
            label=f"write profile acceleration to ID {dxl_id}",
        )
        retry_dxl(
            lambda dxl_id=dxl_id: write4(
                port, packet, dxl_id, ADDR_PROFILE_VELOCITY,
                profile_velocity, "profile velocity",
            ),
            retries=retries,
            delay_s=delay_s,
            label=f"write profile velocity to ID {dxl_id}",
        )


def set_torque(port: Any, packet: Any, ids: list[int], enabled: bool,
               *, retries: int = 0, delay_s: float = 0.0) -> None:
    value = TORQUE_ENABLE if enabled else TORQUE_DISABLE
    for dxl_id in ids:
        retry_dxl(
            lambda dxl_id=dxl_id: write1(
                port, packet, dxl_id, ADDR_TORQUE_ENABLE, value, "torque enable",
            ),
            retries=retries,
            delay_s=delay_s,
            label=f"write torque enable={value} to ID {dxl_id}",
        )


def set_torque_safely(port: Any, packet: Any, ids: list[int], enabled: bool,
                      *, retries: int, delay_s: float) -> None:
    try:
        set_torque(port, packet, ids, enabled, retries=retries, delay_s=delay_s)
    except Exception as exc:
        state = "enable" if enabled else "disable"
        print(f"Warning: failed to {state} torque during cleanup: {exc}")


def confirm_tracking(args: argparse.Namespace, lower: np.ndarray, upper: np.ndarray,
                     ref_info: dict[str, Any]) -> None:
    if args.no_confirm or args.plan_only:
        return
    print()
    print("This will enable servo torque and run PD sine tracking.")
    print(f"  ids={args.ids}, port={args.port}, duration={args.duration:.2f}s, rate={args.rate:.2f}Hz")
    print(f"  effective lower={lower.tolist()}, upper={upper.tolist()}")
    print(f"  sine min={ref_info['ref_min'].tolist()}, max={ref_info['ref_max'].tolist()}")
    typed = input("Type RUN to continue: ").strip()
    if typed != "RUN":
        raise SystemExit("Aborted before torque enable.")


def make_manifest(args: argparse.Namespace, run_dir: Path, xml_lower: np.ndarray, xml_upper: np.ndarray,
                  lower: np.ndarray, upper: np.ndarray, zero_ticks: np.ndarray,
                  ref_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry": "hardware/scripts/track_sine_position_mode.py",
        "run_dir": str(run_dir),
        "xml": str(args.xml),
        "port": args.port,
        "baud": args.baud,
        "ids": list(map(int, args.ids)),
        "zero_ticks": zero_ticks.astype(int).tolist(),
        "signs": list(map(float, args.signs)),
        "xml_lower": xml_lower.tolist(),
        "xml_upper": xml_upper.tolist(),
        "effective_lower": lower.tolist(),
        "effective_upper": upper.tolist(),
        "limit_ratio": float(args.limit_ratio),
        "limit": args.limit,
        "duration": float(args.duration),
        "rate": float(args.rate),
        "kp": float(args.kp),
        "kd": float(args.kd),
        "max_step": float(args.max_step),
        "profile_velocity": int(args.profile_velocity),
        "profile_acceleration": int(args.profile_acceleration),
        "feedback_stride": int(args.feedback_stride),
        "comm_retries": int(args.comm_retries),
        "comm_retry_delay": float(args.comm_retry_delay),
        "current_unit": "mA, read from Present Current(126)",
        "pwm_unit": "percent, Present PWM(124) raw value multiplied by 0.113",
        "torque_constant_nm_per_a": float(args.torque_constant_nm_per_a),
        "torque_note": (
            "XL330 reports Present Current as input-supply current, so estimated torque is a "
            "proxy unless torque_constant_nm_per_a was calibrated for this actuator and load path."
        ),
        "biases": ref_info["biases"].tolist(),
        "amplitudes": ref_info["amplitudes"].tolist(),
        "frequencies": ref_info["frequencies"].tolist(),
        "phases": ref_info["phases"].tolist(),
    }


def compute_metrics(log: dict[str, list[Any]], lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    q_meas = np.asarray(log["q_meas"], dtype=np.float64)
    q_ref = np.asarray(log["q_ref"], dtype=np.float64)
    q_mujoco = np.asarray(log["q_mujoco"], dtype=np.float64)
    current_ma = np.asarray(log["present_current_ma"], dtype=np.float64)
    torque_proxy = np.asarray(log["torque_proxy_nm"], dtype=np.float64)
    goal_clipped = np.asarray(log["goal_clipped"], dtype=bool)
    limit_margin = np.minimum(q_meas - lower[None, :], upper[None, :] - q_meas)
    err = q_ref - q_meas
    mujoco_err = q_ref - q_mujoco
    return {
        "samples": int(q_meas.shape[0]),
        "rmse_rad": np.sqrt(np.mean(err * err, axis=0)).tolist(),
        "mujoco_rmse_rad": np.sqrt(np.mean(mujoco_err * mujoco_err, axis=0)).tolist(),
        "mae_rad": np.mean(np.abs(err), axis=0).tolist(),
        "max_abs_error_rad": np.max(np.abs(err), axis=0).tolist(),
        "peak_abs_q_rad": np.max(np.abs(q_meas), axis=0).tolist(),
        "min_limit_margin_rad": np.min(limit_margin, axis=0).tolist(),
        "goal_clip_count": np.sum(goal_clipped, axis=0).astype(int).tolist(),
        "peak_abs_current_ma": np.max(np.abs(current_ma), axis=0).tolist(),
        "peak_abs_torque_proxy_nm": np.max(np.abs(torque_proxy), axis=0).tolist(),
    }


def append_log(log: dict[str, list[Any]], **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
        log.setdefault(key, []).append(value)


def plot_tracking_results(run_dir: Path, log: dict[str, list[Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = {key: np.asarray(values) for key, values in log.items()}
    t = arrays["t"]
    q_ref = arrays["q_ref"]
    q_meas = arrays["q_meas"]
    q_mujoco = arrays["q_mujoco"]
    error = arrays["error"]
    q_goal = arrays["q_goal"]
    current_ma = arrays["present_current_ma"]
    pwm_percent = arrays["present_pwm_percent"]
    torque_proxy = arrays["torque_proxy_nm"]
    joint_labels = ("joint a / ID10", "joint b / ID20")
    figure_paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, q_ref[:, index], "--", label="desired")
        ax.plot(t, q_mujoco[:, index], "-.", label="MuJoCo")
        ax.plot(t, q_meas[:, index], label="hardware")
        ax.set_ylabel(f"{joint_labels[index]}\nangle [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Desired vs MuJoCo vs hardware joint angles")
    fig.tight_layout()
    path = run_dir / "figures" / "joint_tracking_rad.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, error[:, index], color=f"C{index}")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{joint_labels[index]}\nerror [rad]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Tracking error: desired - hardware")
    fig.tight_layout()
    path = run_dir / "figures" / "tracking_error_rad.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, q_ref[:, index], "--", label="desired")
        ax.plot(t, q_goal[:, index], label="PD command")
        ax.set_ylabel(f"{joint_labels[index]}\ncommand [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("PD commanded joint positions")
    fig.tight_layout()
    path = run_dir / "figures" / "pd_goal_position_rad.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True)
    for index, label in enumerate(joint_labels):
        axes[0].plot(t, current_ma[:, index], label=label)
        axes[1].plot(t, pwm_percent[:, index], label=label)
        axes[2].plot(t, torque_proxy[:, index], label=label)
    axes[0].set_ylabel("current [mA]")
    axes[1].set_ylabel("PWM [%]")
    axes[2].set_ylabel("torque proxy [N m]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("Motor current, PWM, and estimated torque proxy")
    fig.tight_layout()
    path = run_dir / "figures" / "motor_current_pwm_torque_proxy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    return figure_paths


def save_log(run_dir: Path, log: dict[str, list[Any]], metrics: dict[str, Any]) -> list[Path]:
    arrays = {key: np.asarray(values) for key, values in log.items()}
    np.savez_compressed(run_dir / "arrays" / "tracking_log.npz", **arrays)
    save_json(run_dir / "metrics" / "tracking_metrics.json", metrics)
    return plot_tracking_results(run_dir, log)


def run_tracking(
    args: argparse.Namespace,
    mujoco: Any,
    model: Any,
    data: Any,
    qpos_index: dict[str, int],
    port: Any,
    packet: Any,
    zero_ticks: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ref_info: dict[str, Any],
    viewer: Any | None,
    sync_read_group: Any,
    sync_write_group: Any,
) -> dict[str, list[Any]]:
    ids = list(map(int, args.ids))
    signs = np.asarray(args.signs, dtype=np.float64)
    dt_target = 1.0 / float(args.rate)
    log: dict[str, list[Any]] = {}

    previous_q: np.ndarray | None = None
    previous_t: float | None = None
    previous_current_ma = np.zeros(2, dtype=np.float64)
    previous_pwm_percent = np.zeros(2, dtype=np.float64)
    cycle_index = 0
    t0 = time.perf_counter()
    next_tick = t0

    while True:
        now = time.perf_counter()
        t = now - t0
        if t >= args.duration:
            break
        if viewer is not None and not viewer.is_running():
            break

        ticks = sync_read_ticks(sync_read_group, packet, ids)
        if cycle_index % int(args.feedback_stride) == 0:
            previous_current_ma, previous_pwm_percent = read_motor_feedback(
                port,
                packet,
                ids,
                retries=int(args.comm_retries),
                delay_s=float(args.comm_retry_delay),
            )
        present_current_ma = previous_current_ma.copy()
        present_pwm_percent = previous_pwm_percent.copy()
        raw_q, q_meas, measured_clipped = ticks_to_joint_angles(ticks, zero_ticks, signs, lower, upper)
        if previous_q is None or previous_t is None:
            dq_meas = np.zeros(2, dtype=np.float64)
        else:
            dt_measured = max(t - previous_t, 1e-6)
            dq_meas = (q_meas - previous_q) / dt_measured
        previous_q = q_meas.copy()
        previous_t = t

        q_ref, dq_ref = reference_at(
            t,
            ref_info["biases"],
            ref_info["amplitudes"],
            ref_info["frequencies"],
            ref_info["phases"],
        )
        error = q_ref - q_meas
        derror = dq_ref - dq_meas
        q_step = np.clip(float(dt_target) * (float(args.kp) * error + float(args.kd) * derror),
                         -float(args.max_step), float(args.max_step))
        q_goal_raw = q_meas + q_step
        q_goal = np.clip(q_goal_raw, lower, upper)
        goal_clipped = np.abs(q_goal - q_goal_raw) > 1e-12
        goal_ticks = joint_angles_to_ticks(q_goal, zero_ticks, signs)

        sync_write_goal_ticks(sync_write_group, packet, ids, goal_ticks)
        write_joint_angles(mujoco, model, data, qpos_index, q_meas)
        q_mujoco = np.array(
            [data.qpos[qpos_index["joint1"]], data.qpos[qpos_index["joint3"]]],
            dtype=np.float64,
        )
        torque_proxy_nm = (present_current_ma / 1000.0) * float(args.torque_constant_nm_per_a)
        if viewer is not None:
            viewer.sync()

        append_log(
            log,
            t=t,
            ticks=ticks,
            raw_q=raw_q,
            q_meas=q_meas,
            q_mujoco=q_mujoco,
            dq_meas=dq_meas,
            q_ref=q_ref,
            dq_ref=dq_ref,
            q_goal=q_goal,
            goal_ticks=goal_ticks,
            error=error,
            present_current_ma=present_current_ma,
            present_pwm_percent=present_pwm_percent,
            torque_proxy_nm=torque_proxy_nm,
            measured_clipped=measured_clipped,
            goal_clipped=goal_clipped,
        )
        cycle_index += 1

        next_tick += dt_target
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

    return log


def main() -> int:
    args = parse_args()
    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    qpos_index = build_joint_indices(mujoco, model)
    xml_lower, xml_upper = read_active_joint_limits(mujoco, model)
    lower, upper = effective_joint_limits(
        xml_lower,
        xml_upper,
        limit_ratio=float(args.limit_ratio),
        symmetric_limit=args.limit,
    )
    ref_info = validate_reference(args, lower, upper)

    print(f"Loaded MuJoCo model: {args.xml}")
    print(f"Effective joint limits rad: lower={lower.tolist()}, upper={upper.tolist()}")
    print(f"Sine reference rad: min={ref_info['ref_min'].tolist()}, max={ref_info['ref_max'].tolist()}")

    if args.plan_only:
        return 0

    port = None
    packet = None
    log: dict[str, list[Any]] = {}
    zero_ticks = np.zeros(2, dtype=np.int64)
    run_dir: Path | None = None
    try:
        port, packet = open_dynamixel(args)
        initial_ticks = initialize_servos(port, packet, list(map(int, args.ids)))
        zero_ticks = (
            initial_ticks.astype(np.int64)
            if args.zero_ticks is None
            else np.asarray(args.zero_ticks, dtype=np.int64)
        )
        signs = np.asarray(args.signs, dtype=np.float64)
        current_ticks = read_ticks(
            port,
            packet,
            list(map(int, args.ids)),
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )
        _raw_q, q_initial, clipped = ticks_to_joint_angles(current_ticks, zero_ticks, signs, lower, upper)
        if np.any(clipped):
            raise RuntimeError(
                f"Initial servo position is outside effective limits: q={q_initial.tolist()}, "
                f"ticks={current_ticks.tolist()}"
            )
        write_joint_angles(mujoco, model, data, qpos_index, q_initial)
        confirm_tracking(args, lower, upper, ref_info)
        run_dir = make_run_dir(args.output_root)

        configure_servo_motion(
            port,
            packet,
            list(map(int, args.ids)),
            int(args.profile_velocity),
            int(args.profile_acceleration),
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )
        write_goal_ticks(
            port,
            packet,
            list(map(int, args.ids)),
            current_ticks,
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )
        set_torque(
            port,
            packet,
            list(map(int, args.ids)),
            True,
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )

        sync_read_group = create_sync_position_reader(port, packet, list(map(int, args.ids)))
        sync_write_group = create_sync_goal_writer(port, packet, list(map(int, args.ids)))

        viewer = None
        if not args.no_viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(model, data)
            camera_args = SimpleNamespace(
                camera_lookat_x=args.camera_lookat_x,
                camera_lookat_y=args.camera_lookat_y,
                camera_lookat_z=args.camera_lookat_z,
                camera_distance=args.camera_distance,
                camera_azimuth=args.camera_azimuth,
                camera_elevation=args.camera_elevation,
            )
            configure_viewer_camera(viewer, camera_args)

        try:
            log = run_tracking(
                args,
                mujoco,
                model,
                data,
                qpos_index,
                port,
                packet,
                zero_ticks,
                lower,
                upper,
                ref_info,
                viewer,
                sync_read_group,
                sync_write_group,
            )
        finally:
            if viewer is not None:
                viewer.close()
    finally:
        if port is not None and packet is not None:
            try:
                set_torque_safely(
                    port,
                    packet,
                    list(map(int, args.ids)),
                    False,
                    retries=int(args.comm_retries),
                    delay_s=float(args.comm_retry_delay),
                )
            finally:
                port.closePort()

    if run_dir is None:
        raise RuntimeError("Run directory was not created.")
    manifest = make_manifest(args, run_dir, xml_lower, xml_upper, lower, upper, zero_ticks, ref_info)
    save_json(run_dir / "manifest.json", manifest)
    if log:
        metrics = compute_metrics(log, lower, upper)
        figure_paths = save_log(run_dir, log, metrics)
        print(f"Saved log: {run_dir / 'arrays' / 'tracking_log.npz'}")
        print(f"Saved metrics: {run_dir / 'metrics' / 'tracking_metrics.json'}")
        print(f"Saved figures: {[str(path) for path in figure_paths]}")
        print(f"RMSE rad: {metrics['rmse_rad']}")
    else:
        print("No samples collected; manifest saved only.")
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
