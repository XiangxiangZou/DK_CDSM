"""Mirror two hand-rotated XL330 servos into the two-joint MuJoCo model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.common.xl330_mujoco import (
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    DEFAULT_XML,
    build_joint_indices,
    configure_viewer_camera,
    create_sync_position_reader,
    effective_joint_limits,
    enforce_servo_joint_limits,
    initialize_servos,
    load_mujoco,
    open_dynamixel,
    read_active_joint_limits,
    sync_read_ticks,
    ticks_to_joint_angles,
    write4,
    write_joint_angles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize two hand-rotated XL330 servos to a MuJoCo model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--ids", type=int, nargs=2, default=[10, 20])
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--zero-current", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-ticks", type=int, nargs=2, default=None)
    parser.add_argument("--signs", type=int, nargs=2, choices=(-1, 1), default=[1, 1])
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--limit-ratio", type=float, default=1.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--enforce-servo-limits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-velocity", type=int, default=80)
    parser.add_argument("--profile-acceleration", type=int, default=20)
    parser.add_argument("--print-every", type=float, default=0.5)
    parser.add_argument("--camera-lookat-x", type=float, default=3.2)
    parser.add_argument("--camera-lookat-y", type=float, default=0.0)
    parser.add_argument("--camera-lookat-z", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=6.4)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-90.0)
    return parser.parse_args()


def run_loop(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    qpos_index: dict[str, int],
    port: Any,
    packet: Any,
    ids: list[int],
    zero_ticks: np.ndarray,
    signs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    enforce_limits: bool,
    rate_hz: float,
    duration_s: float,
    print_every_s: float,
    viewer: Any | None,
) -> None:
    dt = 1.0 / max(float(rate_hz), 1.0)
    started = time.monotonic()
    next_print = started
    servo_limit_active = np.zeros(len(ids), dtype=bool)
    position_reader = create_sync_position_reader(port, packet, ids)

    while True:
        now = time.monotonic()
        if duration_s > 0.0 and now - started >= duration_s:
            break
        if viewer is not None and not viewer.is_running():
            break

        ticks = sync_read_ticks(position_reader, packet, ids)
        raw_q, q, clipped = ticks_to_joint_angles(ticks, zero_ticks, signs, lower, upper)

        if enforce_limits:
            enforce_servo_joint_limits(
                port=port,
                packet=packet,
                ids=ids,
                raw_q=raw_q,
                lower=lower,
                upper=upper,
                zero_ticks=zero_ticks,
                signs=signs,
                active=servo_limit_active,
            )

        write_joint_angles(mujoco, model, data, qpos_index, q)
        if viewer is not None:
            viewer.sync()

        if now >= next_print:
            print(
                "ticks=({:d}, {:d}) raw_q=({:+.4f}, {:+.4f}) q=({:+.4f}, {:+.4f}) rad{}{}".format(
                    int(ticks[0]),
                    int(ticks[1]),
                    float(raw_q[0]),
                    float(raw_q[1]),
                    float(q[0]),
                    float(q[1]),
                    " CLIPPED" if bool(np.any(clipped)) else "",
                    " SERVO_LIMIT" if bool(np.any(servo_limit_active)) else "",
                )
            )
            next_print = now + max(float(print_every_s), dt)

        elapsed = time.monotonic() - now
        time.sleep(max(0.0, dt - elapsed))


def main() -> int:
    args = parse_args()
    ids = [int(x) for x in args.ids]
    signs = np.asarray(args.signs, dtype=np.float64)

    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    qpos_index = build_joint_indices(mujoco, model)
    xml_lower, xml_upper = read_active_joint_limits(mujoco, model)
    lower, upper = effective_joint_limits(
        xml_lower,
        xml_upper,
        limit_ratio=args.limit_ratio,
        symmetric_limit=args.limit,
    )

    port = None
    try:
        port, packet = open_dynamixel(args)
        initial_ticks = initialize_servos(port, packet, ids)

        for dxl_id in ids:
            write4(port, packet, dxl_id, ADDR_PROFILE_ACCELERATION, args.profile_acceleration, "profile acceleration")
            write4(port, packet, dxl_id, ADDR_PROFILE_VELOCITY, args.profile_velocity, "profile velocity")

        if args.zero_ticks is not None:
            zero_ticks = np.asarray(args.zero_ticks, dtype=np.int64)
        elif args.zero_current:
            zero_ticks = initial_ticks
        else:
            zero_ticks = np.zeros(2, dtype=np.int64)

        print(f"zero_ticks={zero_ticks.tolist()}, signs={signs.tolist()}")
        print(f"effective limits: lower={lower.tolist()}, upper={upper.tolist()}")

        raw_q, q_initial, _ = ticks_to_joint_angles(initial_ticks, zero_ticks, signs, lower, upper)
        write_joint_angles(mujoco, model, data, qpos_index, q_initial)
        print(f"initial raw_q={raw_q.tolist()}, q={q_initial.tolist()}")

        if args.no_viewer:
            run_loop(
                mujoco=mujoco,
                model=model,
                data=data,
                qpos_index=qpos_index,
                port=port,
                packet=packet,
                ids=ids,
                zero_ticks=zero_ticks,
                signs=signs,
                lower=lower,
                upper=upper,
                enforce_limits=args.enforce_servo_limits,
                rate_hz=args.rate,
                duration_s=args.duration,
                print_every_s=args.print_every,
                viewer=None,
            )
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data) as viewer:
                configure_viewer_camera(viewer, args)
                run_loop(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    qpos_index=qpos_index,
                    port=port,
                    packet=packet,
                    ids=ids,
                    zero_ticks=zero_ticks,
                    signs=signs,
                    lower=lower,
                    upper=upper,
                    enforce_limits=args.enforce_servo_limits,
                    rate_hz=args.rate,
                    duration_s=args.duration,
                    print_every_s=args.print_every,
                    viewer=viewer,
                )
    finally:
        if port is not None:
            port.closePort()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
