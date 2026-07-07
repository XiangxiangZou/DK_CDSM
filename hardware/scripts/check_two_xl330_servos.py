"""Connectivity and basic sequential motion test for two DYNAMIXEL XL330 servos."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.common.xl330_mujoco import (
    ADDR_GOAL_POSITION,
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_LED,
    ADDR_MOVING,
    ADDR_OPERATING_MODE,
    ADDR_PRESENT_POSITION,
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    ADDR_TORQUE_ENABLE,
    LED_OFF,
    LED_ON,
    OPERATING_MODE_EXTENDED_POSITION,
    TORQUE_DISABLE,
    TORQUE_ENABLE,
    TICKS_PER_REVOLUTION,
    DynamixelError,
    int32_to_uint32,
    open_dynamixel,
    ping,
    read1,
    read4,
    read_servo,
    uint32_to_int32,
    write1,
    write4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test two XL330 servos on one DYNAMIXEL bus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--ids", type=int, nargs=2, default=[10, 20])
    parser.add_argument("--rev-ticks", type=int, default=TICKS_PER_REVOLUTION)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--position-threshold", type=int, default=20)
    parser.add_argument("--profile-velocity", type=int, default=80)
    parser.add_argument("--profile-acceleration", type=int, default=20)
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--keep-torque", action="store_true")
    parser.add_argument("--skip-led", action="store_true")
    return parser.parse_args()


def print_status(port: Any, packet: Any, dxl_id: int) -> int:
    ping_model = ping(port, packet, dxl_id)
    reading = read_servo(port, packet, dxl_id)
    operating_mode = read1(port, packet, dxl_id, ADDR_OPERATING_MODE, "operating mode")

    if reading.model_number != ping_model:
        raise DynamixelError(
            f"ID {dxl_id}: ping model {ping_model} != register model {reading.model_number}"
        )

    print(
        "ID {id}: model={model}, fw={fw}, mode={mode}, pos={pos}, "
        "vin={vin:.1f} V, temp={temp} C, hw_error=0x{err:02X}".format(
            id=dxl_id,
            model=reading.model_number,
            fw=reading.firmware_version,
            mode=operating_mode,
            pos=reading.ticks,
            vin=reading.input_voltage_v,
            temp=reading.temperature_c,
            err=reading.hardware_error_status,
        )
    )
    return int(reading.ticks)


def configure_servo(
    port: Any,
    packet: Any,
    dxl_id: int,
    *,
    profile_velocity: int,
    profile_acceleration: int,
    skip_led: bool,
) -> None:
    write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque enable")
    write1(
        port,
        packet,
        dxl_id,
        ADDR_OPERATING_MODE,
        OPERATING_MODE_EXTENDED_POSITION,
        "operating mode",
    )
    write4(port, packet, dxl_id, ADDR_PROFILE_ACCELERATION, profile_acceleration, "profile acceleration")
    write4(port, packet, dxl_id, ADDR_PROFILE_VELOCITY, profile_velocity, "profile velocity")
    if not skip_led:
        write1(port, packet, dxl_id, ADDR_LED, LED_ON, "LED")
    write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "torque enable")


def wait_for_position(
    port: Any,
    packet: Any,
    dxl_id: int,
    target: int,
    *,
    timeout: float,
    threshold: int,
) -> None:
    started = time.monotonic()
    while True:
        position = uint32_to_int32(read4(port, packet, dxl_id, ADDR_PRESENT_POSITION, "present position"))
        moving = read1(port, packet, dxl_id, ADDR_MOVING, "moving")
        error = target - position
        print(f"  ID {dxl_id}: pos={position}, target={target}, error={error}, moving={moving}")

        if abs(error) <= threshold and moving == 0:
            return
        if time.monotonic() - started > timeout:
            raise TimeoutError(
                f"ID {dxl_id}: timeout waiting for target {target}; "
                f"last position={position}, error={error}"
            )
        time.sleep(0.2)


def move_to_position(
    port: Any,
    packet: Any,
    dxl_id: int,
    target: int,
    *,
    timeout: float,
    threshold: int,
) -> None:
    print(f"Moving ID {dxl_id} to {target}")
    write4(
        port,
        packet,
        dxl_id,
        ADDR_GOAL_POSITION,
        int32_to_uint32(target),
        "goal position",
    )
    wait_for_position(
        port,
        packet,
        dxl_id,
        target,
        timeout=timeout,
        threshold=threshold,
    )


def run_motion_sequence(port: Any, packet: Any, args: argparse.Namespace, start_positions: dict[int, int]) -> None:
    for dxl_id in args.ids:
        start = int(start_positions[dxl_id])
        plus_one_rev = start + int(args.rev_ticks)

        move_to_position(
            port,
            packet,
            dxl_id,
            plus_one_rev,
            timeout=args.timeout,
            threshold=args.position_threshold,
        )
        time.sleep(float(args.pause))

        move_to_position(
            port,
            packet,
            dxl_id,
            start,
            timeout=args.timeout,
            threshold=args.position_threshold,
        )
        time.sleep(float(args.pause))


def confirm_motion(args: argparse.Namespace) -> None:
    if args.no_confirm:
        return

    print()
    print("Safety confirmation required.")
    print(f"  Port: {args.port}")
    print(f"  IDs: {args.ids}")
    print(f"  Motion: each servo moves +{args.rev_ticks} ticks, then returns.")
    print("  Make sure the servos are mechanically safe before continuing.")
    answer = input("Type RUN to start: ").strip()
    if answer != "RUN":
        raise SystemExit("Aborted by user.")


def disable_outputs(port: Any, packet: Any, ids: list[int], *, keep_torque: bool, skip_led: bool) -> None:
    for dxl_id in ids:
        try:
            if not keep_torque:
                write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque enable")
            if not skip_led:
                write1(port, packet, dxl_id, ADDR_LED, LED_OFF, "LED")
            hardware_error = read1(port, packet, dxl_id, ADDR_HARDWARE_ERROR_STATUS, "hardware error status")
            print(f"ID {dxl_id}: shutdown hw_error=0x{hardware_error:02X}")
        except Exception as exc:  # pragma: no cover - best effort hardware shutdown
            print(f"WARNING: failed to shut down ID {dxl_id}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    port = None
    try:
        port, packet = open_dynamixel(args)
        start_positions: dict[int, int] = {}

        print("Reading initial servo status")
        for dxl_id in args.ids:
            start_positions[int(dxl_id)] = print_status(port, packet, int(dxl_id))

        for dxl_id in args.ids:
            configure_servo(
                port,
                packet,
                int(dxl_id),
                profile_velocity=args.profile_velocity,
                profile_acceleration=args.profile_acceleration,
                skip_led=args.skip_led,
            )

        confirm_motion(args)
        run_motion_sequence(port, packet, args, start_positions)

        print("Final servo status")
        for dxl_id in args.ids:
            print_status(port, packet, int(dxl_id))
    finally:
        if port is not None:
            disable_outputs(
                port,
                packet,
                [int(x) for x in args.ids],
                keep_torque=args.keep_torque,
                skip_led=args.skip_led,
            )
            port.closePort()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
