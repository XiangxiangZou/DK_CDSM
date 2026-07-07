"""Current-mode joint reference tracking for two XL330 servos.

This script runs an external angle loop in DYNAMIXEL Operating Mode 0:

    PC PD controller -> Goal Current(102) -> motor motion
    Present Position(132) -> measured angle -> PC feedback loop

The default reference is a safe sinusoid for two active joints.  The previous
point-to-point behavior is still available with ``--reference point``.
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
    ADDR_GOAL_CURRENT,
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_OPERATING_MODE,
    ADDR_PRESENT_CURRENT,
    ADDR_PRESENT_POSITION,
    ADDR_PRESENT_PWM,
    ADDR_PRESENT_TEMPERATURE,
    ADDR_PRESENT_VELOCITY,
    ADDR_TORQUE_ENABLE,
    CURRENT_UNIT_MA,
    DEFAULT_XML,
    DynamixelError,
    OPERATING_MODE_CURRENT,
    PROTOCOL_VERSION,
    PRESENT_VELOCITY_UNIT_RPM,
    PWM_UNIT_PERCENT,
    RAD_PER_TICK,
    TORQUE_DISABLE,
    TORQUE_ENABLE,
    build_joint_indices,
    check_packet,
    configure_viewer_camera,
    create_sync_position_reader,
    create_sync_read_group,
    effective_joint_limits,
    load_dynamixel_sdk,
    load_mujoco,
    open_dynamixel,
    ping,
    read1,
    read2,
    read4,
    read_active_joint_limits,
    read_servo,
    sync_read_ticks,
    sync_read_unsigned,
    ticks_to_joint_angles,
    uint32_to_int32,
    write1,
    write_joint_angles,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "hardware" / "outputs" / "current_mode_tracking"
DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 57600
DEFAULT_IDS = [10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External PD joint-reference loop using XL330 Current Control Mode."
    )

    # =========================================================================
    # 硬件通信
    # =========================================================================
    parser.add_argument(
        "--port", default=DEFAULT_PORT,
        help="串口号（Windows: COMx, Linux: /dev/ttyUSBx）。",
    )
    parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD,
        help="波特率。XL330 默认 57600，可选 1M/2M/3M（需固件支持）。",
    )
    parser.add_argument(
        "--ids", type=int, nargs="+", default=DEFAULT_IDS,
        help="舵机 ID 列表。如 --ids 10 20。",
    )

    # =========================================================================
    # 参考轨迹
    # =========================================================================
    parser.add_argument(
        "--reference", choices=("sine", "point"), default="sine",
        help="轨迹类型。sine=正弦跟踪，point=阶跃到位。",
    )
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[1.0, 1.0],
        help="[point] 目标角度 [rad]。",
    )
    parser.add_argument(
        "--biases", type=float, nargs="+", default=[0.0, 0.0],
        help="[sine] 正弦偏置 [rad]。q_ref = bias + amp*sin(2pi*f*t + phase)。",
    )
    parser.add_argument(
        "--amplitudes", type=float, nargs="+", default=[1.0, 1.0],
        help="[sine] 正弦幅值 [rad]。必须在有效限位内。"
             " 峰值速度=2pi*f*A，峰值加速度=(2pi*f)^2*A。",
    )
    parser.add_argument(
        "--frequencies", type=float, nargs="+", default=[0.050, 0.050],
        help="[sine] 正弦频率 [Hz]。越高跟踪越难（加速度与 f^2 成正比）。",
    )
    parser.add_argument(
        "--phases", type=float, nargs="+", default=[0.0, 0.0],
        help="[sine] 初始相位 [rad]。0 = 从零点开始正向运动。",
    )

    # =========================================================================
    # 运行参数
    # =========================================================================
    parser.add_argument(
        "--xml", type=Path, default=DEFAULT_XML,
        help="MuJoCo XML 模型路径。",
    )
    parser.add_argument(
        "--duration", type=float, default=40.0,
        help="运行时长 [s]。",
    )
    parser.add_argument(
        "--rate", type=float, default=120.0,
        help="目标控制频率 [Hz]。实际帧率取决于通信开销，"
             "以输出 metrics 中的 mean_loop_hz 为准。设高了不会报错，只是跑不满。",
    )
    parser.add_argument(
        "--signs", type=float, nargs="+", default=[1.0, 1.0],
        help="舵机到关节角的方向符号。+1=同向，-1=反向。",
    )

    # =========================================================================
    # 安全限位
    # =========================================================================
    parser.add_argument(
        "--limit-ratio", type=float, default=0.90,
        help="关节限位缩放比例 (0, 1.0]。以 XML 限位中心为轴，"
             "半范围乘以 ratio。0.9=使用中间 90%%。",
    )
    parser.add_argument(
        "--limit", type=float, default=None,
        help="可选的对称限位 [rad]。如 --limit 1.0 → [-1.0, +1.0]。",
    )

    # =========================================================================
    # 电流 PD 核心调参区
    # =========================================================================
    parser.add_argument(
        "--kp-current", type=float, default=350.0,
        help="电流比例增益 [mA/rad]。"
             " 每 1 rad 位置误差 → 多少 mA 电流。\n"
             " goal_current = Kp*error + Kd*(dq_ref - dq_filt)。\n"
             " 调参：↑ 跟踪更紧 ↓ 误差更小，但过大→超调震荡。\n"
             " 当前默认 350，用于克服 20260707_104430 中暴露的静摩擦台阶。",
    )
    parser.add_argument(
        "--kd-current", type=float, default=0.5,
        help="电流微分增益 [mA/(rad/s)]。"
             " 每 1 rad/s 速度误差 → 多少 mA 电流（阻尼项）。\n"
             " 调参：↑ 抑制超调 ↓ 震荡，但过大→放大编码器噪声。\n"
             " 编码器量化噪声 ~0.046 rad/s，噪声电流 = Kd*0.046/rate。\n"
             " 典型范围：Kp/20 ~ Kp/40。",
    )
    parser.add_argument(
        "--max-current-ma", type=float, default=200.0,
        help="Goal Current 绝对值上限 [mA]。安全限幅，"
             "防止电流指令过大。XL330 堵转 ~1A，100mA 很安全。",
    )
    parser.add_argument(
        "--static-current-ma", type=float, default=0.0,
        help="静摩擦/死区补偿电流 [mA]。当 |error| 超过 static-deadband 时，"
             "沿误差方向额外叠加该电流；0 表示关闭补偿。",
    )
    parser.add_argument(
        "--static-deadband", type=float, default=0.00,
        help="启用静摩擦补偿的位置误差阈值 [rad]。",
    )

    # =========================================================================
    # 到位检测（point 模式）
    # =========================================================================
    parser.add_argument(
        "--position-tolerance", type=float, default=0.01,
        help="到位判据 [rad]：|error| < 此值视为到达目标。",
    )
    parser.add_argument(
        "--settle-time", type=float, default=0.5,
        help="稳定时间 [s]：误差保持在 tolerance 内超过此时间才算 settled。",
    )
    parser.add_argument(
        "--stop-when-settled",
        action=argparse.BooleanOptionalAction, default=None,
        help="settled 后是否自动停止。默认 point=开启，sine=关闭。",
    )

    # =========================================================================
    # 滤波与安全
    # =========================================================================
    parser.add_argument(
        "--feedback-stride", type=int, default=60,
        help="温度/硬件错误等慢变量每隔 N 帧读一次。大值提升帧率。",
    )
    parser.add_argument(
        "--effort-feedback-stride", type=int, default=5,
        help="Present Current/PWM 每隔 N 帧读一次。大值可显著提升电流闭环实际频率。",
    )
    parser.add_argument(
        "--velocity-filter-alpha", type=float, default=0.1,
        help="速度 EMA 平滑系数 (0, 1.0]。α→1 几乎不过滤，α→0.1 强平滑。"
             " dq_filt[k] = alpha*dq_raw[k] + (1-alpha)*dq_filt[k-1]。",
    )
    parser.add_argument(
        "--max-abs-velocity", type=float, default=10.0,
        help="速度安全上限 [rad/s]。超过触发紧急停止。",
    )
    parser.add_argument(
        "--max-temperature-c", type=float, default=80.0,
        help="温度安全上限 [°C]。超过触发紧急停止。",
    )

    # =========================================================================
    # 输出与 UI
    # =========================================================================
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录。每次运行创建时间戳子目录。",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="禁用 MuJoCo 查看器。",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="仅验证模型和参数，不打开串口。",
    )
    parser.add_argument(
        "--no-confirm", action="store_true", default=True,
        help="跳过 'Type RUN to continue' 确认。当前默认跳过。",
    )

    # =========================================================================
    # 通信鲁棒性
    # =========================================================================
    parser.add_argument(
        "--comm-retries", type=int, default=3,
        help="Dynamixel 通信失败重试次数。",
    )
    parser.add_argument(
        "--comm-retry-delay", type=float, default=0.02,
        help="重试等待时间 [s]。",
    )

    # =========================================================================
    # MuJoCo 自由相机
    # =========================================================================
    parser.add_argument(
        "--camera-lookat-x", type=float, default=3.2,
        help="相机注视点 X [m]。",
    )
    parser.add_argument(
        "--camera-lookat-y", type=float, default=0.0,
        help="相机注视点 Y [m]。",
    )
    parser.add_argument(
        "--camera-lookat-z", type=float, default=0.0,
        help="相机注视点 Z [m]。",
    )
    parser.add_argument(
        "--camera-distance", type=float, default=9.6,
        help="相机到注视点距离 [m]。",
    )
    parser.add_argument(
        "--camera-azimuth", type=float, default=90.0,
        help="相机方位角 [°]。90=侧视。",
    )
    parser.add_argument(
        "--camera-elevation", type=float, default=-90.0,
        help="相机俯仰角 [°]。-90=俯视。",
    )
    return parser.parse_args()


def uint16_to_int16(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def int16_to_uint16(value: int) -> int:
    return int(value) & 0xFFFF


def int32_to_signed(value: int) -> int:
    return uint32_to_int32(value)


def write2(port: Any, packet: Any, dxl_id: int, address: int, value: int, name: str) -> None:
    result, error = packet.write2ByteTxRx(port, dxl_id, address, int(value) & 0xFFFF)
    check_packet(packet, result, error, f"write {name}={value} to ID {dxl_id}")


def create_sync_current_writer(port: Any, packet: Any) -> Any:
    try:
        from dynamixel_sdk import GroupSyncWrite
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: dynamixel_sdk.") from exc
    return GroupSyncWrite(port, packet, ADDR_GOAL_CURRENT, 2)


def sync_write_goal_current(group: Any, packet: Any, ids: list[int],
                            current_ma: np.ndarray) -> None:
    try:
        from dynamixel_sdk import COMM_SUCCESS
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: dynamixel_sdk.") from exc

    group.clearParam()
    for dxl_id, current in zip(ids, current_ma):
        raw = int16_to_uint16(int(round(float(current) / CURRENT_UNIT_MA)))
        data = [(raw >> 0) & 0xFF, (raw >> 8) & 0xFF]
        if not group.addParam(dxl_id, data):
            raise DynamixelError(f"Failed to add goal-current data for ID {dxl_id}")
    result = group.txPacket()
    if result != COMM_SUCCESS:
        raise DynamixelError(f"Sync-write goal current failed: {packet.getTxRxResult(result)}")


def retry_dxl(action: Any, *, retries: int, delay_s: float, label: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"Warning: {label} failed ({exc}); retry {attempt + 1}/{retries}")
            time.sleep(delay_s)
    if last_error is None:
        raise RuntimeError(f"{label} failed without a captured error.")
    raise last_error


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


def vector_arg(values: list[float], name: str, ids: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != ids.shape:
        raise ValueError(f"--{name} must contain exactly one value per servo ID.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"--{name} must contain only finite values.")
    return vector


def reference_at(t: float, ref_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if ref_info["type"] == "point":
        return ref_info["targets"], np.zeros_like(ref_info["targets"])

    omega = 2.0 * math.pi * ref_info["frequencies"]
    arg = omega * float(t) + ref_info["phases"]
    q_ref = ref_info["biases"] + ref_info["amplitudes"] * np.sin(arg)
    dq_ref = ref_info["amplitudes"] * omega * np.cos(arg)
    return q_ref, dq_ref


def apply_static_current_compensation(goal_current: np.ndarray, error: np.ndarray,
                                      args: argparse.Namespace) -> np.ndarray:
    static_current = float(args.static_current_ma)
    deadband = float(args.static_deadband)
    if static_current <= 0.0:
        return goal_current
    active = np.abs(error) > deadband
    compensated = goal_current.copy()
    compensated[active] += np.sign(error[active]) * static_current
    return compensated


def validate_reference(args: argparse.Namespace, ids: np.ndarray,
                       lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    if args.reference == "point":
        targets = vector_arg(args.targets, "targets", ids)
        if np.any(targets < lower) or np.any(targets > upper):
            raise ValueError(
                f"Targets exceed effective limits: targets={targets.tolist()}, "
                f"lower={lower.tolist()}, upper={upper.tolist()}"
            )
        return {
            "type": "point",
            "targets": targets,
            "ref_min": targets.copy(),
            "ref_max": targets.copy(),
        }

    biases = vector_arg(args.biases, "biases", ids)
    amplitudes = np.abs(vector_arg(args.amplitudes, "amplitudes", ids))
    frequencies = vector_arg(args.frequencies, "frequencies", ids)
    phases = vector_arg(args.phases, "phases", ids)
    if np.any(frequencies <= 0.0):
        raise ValueError("--frequencies must be positive for sine tracking.")

    ref_min = biases - amplitudes
    ref_max = biases + amplitudes
    if np.any(ref_min < lower) or np.any(ref_max > upper):
        raise ValueError(
            "Sine reference exceeds effective limits: "
            f"ref_min={ref_min.tolist()}, ref_max={ref_max.tolist()}, "
            f"lower={lower.tolist()}, upper={upper.tolist()}"
        )

    return {
        "type": "sine",
        "biases": biases,
        "amplitudes": amplitudes,
        "frequencies": frequencies,
        "phases": phases,
        "ref_min": ref_min,
        "ref_max": ref_max,
    }


def should_stop_when_settled(args: argparse.Namespace) -> bool:
    if args.stop_when_settled is not None:
        return bool(args.stop_when_settled)
    return args.reference == "point"


def validate_args(args: argparse.Namespace, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(args.ids, dtype=np.int64)
    signs = np.asarray(args.signs, dtype=np.float64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("--ids must contain at least one servo ID.")
    if signs.shape != ids.shape:
        raise ValueError("--signs must contain exactly one sign per servo ID.")
    if ids.size != 2:
        raise ValueError("This v1 script expects exactly two IDs for the two-joint MuJoCo model.")
    if args.duration <= 0.0 or args.rate <= 0.0:
        raise ValueError("--duration and --rate must be positive.")
    if args.kp_current < 0.0 or args.kd_current < 0.0:
        raise ValueError("--kp-current and --kd-current must be non-negative.")
    if args.max_current_ma <= 0.0:
        raise ValueError("--max-current-ma must be positive.")
    if args.static_current_ma < 0.0 or args.static_deadband < 0.0:
        raise ValueError("--static-current-ma and --static-deadband must be non-negative.")
    if args.static_current_ma >= args.max_current_ma:
        raise ValueError("--static-current-ma must be smaller than --max-current-ma.")
    if args.position_tolerance <= 0.0 or args.settle_time < 0.0:
        raise ValueError("--position-tolerance must be positive and --settle-time non-negative.")
    if args.feedback_stride <= 0 or args.effort_feedback_stride <= 0:
        raise ValueError("--feedback-stride and --effort-feedback-stride must be positive.")
    if not 0.0 < args.velocity_filter_alpha <= 1.0:
        raise ValueError("--velocity-filter-alpha must be in (0, 1].")
    if args.max_abs_velocity <= 0.0 or args.max_temperature_c <= 0.0:
        raise ValueError("--max-abs-velocity and --max-temperature-c must be positive.")
    if args.comm_retries < 0 or args.comm_retry_delay < 0.0:
        raise ValueError("--comm-retries and --comm-retry-delay must be non-negative.")
    ref_info = validate_reference(args, ids, lower, upper)
    return {"ids": ids, "signs": signs, "ref_info": ref_info}


def read_ticks(port: Any, packet: Any, ids: list[int], *, retries: int, delay_s: float) -> np.ndarray:
    return np.array(
        [
            int32_to_signed(
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


def read_feedback(port: Any, packet: Any, ids: list[int], *, retries: int,
                  delay_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    present_current_ma = np.array(
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
    present_pwm_percent = np.array(
        [
            uint16_to_int16(
                retry_dxl(
                    lambda dxl_id=dxl_id: read2(port, packet, dxl_id, ADDR_PRESENT_PWM, "present pwm"),
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
    present_velocity_rad_s = np.array(
        [
            int32_to_signed(
                retry_dxl(
                    lambda dxl_id=dxl_id: read4(
                        port, packet, dxl_id, ADDR_PRESENT_VELOCITY, "present velocity"
                    ),
                    retries=retries,
                    delay_s=delay_s,
                    label=f"read present velocity from ID {dxl_id}",
                )
            )
            * PRESENT_VELOCITY_UNIT_RPM
            * (2.0 * math.pi / 60.0)
            for dxl_id in ids
        ],
        dtype=np.float64,
    )
    temperature_c = np.array(
        [
            retry_dxl(
                lambda dxl_id=dxl_id: read1(
                    port, packet, dxl_id, ADDR_PRESENT_TEMPERATURE, "temperature"
                ),
                retries=retries,
                delay_s=delay_s,
                label=f"read temperature from ID {dxl_id}",
            )
            for dxl_id in ids
        ],
        dtype=np.float64,
    )
    hardware_error = np.array(
        [
            retry_dxl(
                lambda dxl_id=dxl_id: read1(
                    port, packet, dxl_id, ADDR_HARDWARE_ERROR_STATUS, "hardware error status"
                ),
                retries=retries,
                delay_s=delay_s,
                label=f"read hardware error from ID {dxl_id}",
            )
            for dxl_id in ids
        ],
        dtype=np.int64,
    )
    return present_current_ma, present_pwm_percent, present_velocity_rad_s, temperature_c, hardware_error


def write_goal_current(port: Any, packet: Any, ids: list[int], current_ma: np.ndarray,
                       *, retries: int, delay_s: float) -> None:
    for dxl_id, current in zip(ids, current_ma):
        raw = int16_to_uint16(int(round(float(current) / CURRENT_UNIT_MA)))
        retry_dxl(
            lambda dxl_id=dxl_id, raw=raw: write2(
                port, packet, dxl_id, ADDR_GOAL_CURRENT, raw, "goal current"
            ),
            retries=retries,
            delay_s=delay_s,
            label=f"write goal current to ID {dxl_id}",
        )


def set_torque(port: Any, packet: Any, ids: list[int], enabled: bool,
               *, retries: int, delay_s: float) -> None:
    value = TORQUE_ENABLE if enabled else TORQUE_DISABLE
    for dxl_id in ids:
        retry_dxl(
            lambda dxl_id=dxl_id: write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, value, "torque enable"),
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


def zero_current_safely(port: Any, packet: Any, ids: list[int], *, retries: int, delay_s: float) -> None:
    try:
        write_goal_current(
            port,
            packet,
            ids,
            np.zeros(len(ids), dtype=np.float64),
            retries=retries,
            delay_s=delay_s,
        )
    except Exception as exc:
        print(f"Warning: failed to write zero current during cleanup: {exc}")


def initialize_current_mode(port: Any, packet: Any, ids: list[int], *, retries: int,
                            delay_s: float) -> np.ndarray:
    initial_ticks = []
    for dxl_id in ids:
        ping_model = retry_dxl(
            lambda dxl_id=dxl_id: ping(port, packet, dxl_id),
            retries=retries,
            delay_s=delay_s,
            label=f"ping ID {dxl_id}",
        )
        retry_dxl(
            lambda dxl_id=dxl_id: write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque enable"),
            retries=retries,
            delay_s=delay_s,
            label=f"disable torque on ID {dxl_id}",
        )
        retry_dxl(
            lambda dxl_id=dxl_id: write1(
                port, packet, dxl_id, ADDR_OPERATING_MODE,
                OPERATING_MODE_CURRENT, "operating mode",
            ),
            retries=retries,
            delay_s=delay_s,
            label=f"set current mode on ID {dxl_id}",
        )
        reading = retry_dxl(
            lambda dxl_id=dxl_id: read_servo(port, packet, dxl_id),
            retries=retries,
            delay_s=delay_s,
            label=f"read status from ID {dxl_id}",
        )
        if reading.model_number != ping_model:
            raise DynamixelError(
                f"ID {dxl_id}: ping model {ping_model} != register model {reading.model_number}"
            )
        if reading.hardware_error_status != 0:
            raise DynamixelError(
                f"ID {dxl_id}: hardware error status 0x{reading.hardware_error_status:02X}"
            )
        initial_ticks.append(reading.ticks)
        print(
            f"ID {dxl_id}: model={reading.model_number}, fw={reading.firmware_version}, "
            f"mode={OPERATING_MODE_CURRENT}, zero_tick={reading.ticks}, "
            f"vin={reading.input_voltage_v:.1f} V, temp={reading.temperature_c} C"
        )
    return np.asarray(initial_ticks, dtype=np.int64)


def confirm_run(args: argparse.Namespace, ref_info: dict[str, Any],
                lower: np.ndarray, upper: np.ndarray) -> None:
    if args.plan_only:
        return
    print()
    print("This will switch the servos to Current Control Mode and enable torque.")
    print(f"  ids={args.ids}, reference={ref_info['type']}, max_current={args.max_current_ma:.1f} mA")
    print(
        f"  static_current={args.static_current_ma:.1f} mA, "
        f"static_deadband={args.static_deadband:.4f} rad"
    )
    if ref_info["type"] == "point":
        print(f"  targets={ref_info['targets'].tolist()} rad")
    else:
        print(
            "  sine biases={}, amplitudes={}, frequencies={} Hz, phases={} rad".format(
                ref_info["biases"].tolist(),
                ref_info["amplitudes"].tolist(),
                ref_info["frequencies"].tolist(),
                ref_info["phases"].tolist(),
            )
        )
        print(f"  sine min={ref_info['ref_min'].tolist()}, max={ref_info['ref_max'].tolist()} rad")
    print(f"  effective lower={lower.tolist()}, upper={upper.tolist()}")
    print("  RUN confirmation is disabled; continuing without interactive input.")


def append_log(log: dict[str, list[Any]], **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
        log.setdefault(key, []).append(value)


def compute_metrics(log: dict[str, list[Any]], ref_info: dict[str, Any],
                    position_tolerance: float) -> dict[str, Any]:
    t = np.asarray(log["t"], dtype=np.float64)
    q = np.asarray(log["q_meas"], dtype=np.float64)
    q_ref = np.asarray(log["q_ref"], dtype=np.float64)
    error = q_ref - q
    cmd = np.asarray(log["goal_current_ma"], dtype=np.float64)
    present_current = np.asarray(log["present_current_ma"], dtype=np.float64)
    dt = np.diff(t)
    return {
        "reference": ref_info["type"],
        "samples": int(q.shape[0]),
        "duration_span_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "mean_loop_dt_s": float(np.mean(dt)) if dt.size else None,
        "mean_loop_hz": float(1.0 / np.mean(dt)) if dt.size and np.mean(dt) > 0.0 else None,
        "final_q_rad": q[-1].tolist(),
        "final_q_ref_rad": q_ref[-1].tolist(),
        "final_error_rad": error[-1].tolist(),
        "rmse_rad": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "mae_rad": np.mean(np.abs(error), axis=0).tolist(),
        "max_abs_error_rad": np.max(np.abs(error), axis=0).tolist(),
        "max_abs_reference_rad": np.max(np.abs(q_ref), axis=0).tolist(),
        "peak_abs_goal_current_ma": np.max(np.abs(cmd), axis=0).tolist(),
        "peak_abs_present_current_ma": np.max(np.abs(present_current), axis=0).tolist(),
        "zero_position_step_fraction": [
            float(np.mean(np.abs(np.diff(q[:, index])) < 1e-12))
            for index in range(q.shape[1])
        ],
        "final_within_tolerance": bool(np.all(np.abs(error[-1]) <= float(position_tolerance))),
    }


def plot_results(run_dir: Path, log: dict[str, list[Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = {key: np.asarray(values) for key, values in log.items()}
    t = arrays["t"]
    q_ref = arrays["q_ref"]
    q = arrays["q_meas"]
    error = arrays["error"]
    dq = arrays["dq_filtered"]
    goal_current = arrays["goal_current_ma"]
    present_current = arrays["present_current_ma"]
    labels = ("joint a / ID10", "joint b / ID20")
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, q_ref[:, index], "--", label="desired")
        ax.plot(t, q[:, index], label="measured")
        ax.set_ylabel(f"{labels[index]}\nangle [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Current-mode joint reference tracking")
    fig.tight_layout()
    path = run_dir / "figures" / "angle_tracking_rad.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, error[:, index], color=f"C{index}")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{labels[index]}\nerror [rad]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Tracking error: target - measured")
    fig.tight_layout()
    path = run_dir / "figures" / "tracking_error_rad.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, goal_current[:, index], label="Goal Current")
        ax.plot(t, present_current[:, index], label="Present Current")
        ax.set_ylabel(f"{labels[index]}\ncurrent [mA]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Commanded vs present current")
    fig.tight_layout()
    path = run_dir / "figures" / "current_command_vs_present.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(t, dq[:, index], color=f"C{index}")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{labels[index]}\nvelocity [rad/s]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Filtered velocity from encoder finite differences")
    fig.tight_layout()
    path = run_dir / "figures" / "velocity_rad_s.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def save_log(run_dir: Path, log: dict[str, list[Any]], metrics: dict[str, Any],
             ref_info: dict[str, Any]) -> list[Path]:
    arrays = {key: np.asarray(values) for key, values in log.items()}
    np.savez_compressed(run_dir / "arrays" / "current_mode_log.npz", **arrays)
    save_json(run_dir / "metrics" / "current_mode_metrics.json", metrics)
    save_json(run_dir / "metrics" / "reference_summary.json", {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in ref_info.items()
    })
    return plot_results(run_dir, log)


def make_manifest(args: argparse.Namespace, run_dir: Path, lower: np.ndarray, upper: np.ndarray,
                  zero_ticks: np.ndarray, ref_info: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "entry": "hardware/scripts/track_sine_current_mode.py",
        "run_dir": str(run_dir),
        "xml": str(args.xml),
        "port": args.port,
        "baud": int(args.baud),
        "ids": list(map(int, args.ids)),
        "operating_mode": OPERATING_MODE_CURRENT,
        "zero_ticks": zero_ticks.astype(int).tolist(),
        "signs": list(map(float, args.signs)),
        "effective_lower": lower.tolist(),
        "effective_upper": upper.tolist(),
        "reference": ref_info["type"],
        "reference_min_rad": ref_info["ref_min"].tolist(),
        "reference_max_rad": ref_info["ref_max"].tolist(),
        "duration": float(args.duration),
        "rate": float(args.rate),
        "kp_current_ma_per_rad": float(args.kp_current),
        "kd_current_ma_per_rad_s": float(args.kd_current),
        "max_current_ma": float(args.max_current_ma),
        "static_current_ma": float(args.static_current_ma),
        "static_deadband_rad": float(args.static_deadband),
        "feedback_stride": int(args.feedback_stride),
        "effort_feedback_stride": int(args.effort_feedback_stride),
        "run_confirmation": "disabled",
        "position_tolerance_rad": float(args.position_tolerance),
        "settle_time_s": float(args.settle_time),
        "current_note": "Goal Current and Present Current use 1 mA units; XL330 current is an input-side proxy.",
    }
    if ref_info["type"] == "point":
        manifest["targets_rad"] = ref_info["targets"].tolist()
    else:
        manifest["biases_rad"] = ref_info["biases"].tolist()
        manifest["amplitudes_rad"] = ref_info["amplitudes"].tolist()
        manifest["frequencies_hz"] = ref_info["frequencies"].tolist()
        manifest["phases_rad"] = ref_info["phases"].tolist()
    return manifest


def check_safety(q: np.ndarray, dq: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 temperature_c: np.ndarray, hardware_error: np.ndarray,
                 args: argparse.Namespace) -> None:
    if np.any(q < lower) or np.any(q > upper):
        raise RuntimeError(f"Joint limit exceeded: q={q.tolist()}, lower={lower.tolist()}, upper={upper.tolist()}")
    if np.any(np.abs(dq) > float(args.max_abs_velocity)):
        raise RuntimeError(f"Velocity limit exceeded: dq={dq.tolist()}")
    if np.any(temperature_c > float(args.max_temperature_c)):
        raise RuntimeError(f"Temperature limit exceeded: temp={temperature_c.tolist()} C")
    if np.any(hardware_error != 0):
        raise RuntimeError(f"Hardware error status nonzero: {hardware_error.tolist()}")


def run_loop(
    args: argparse.Namespace,
    mujoco: Any,
    model: Any,
    data: Any,
    qpos_index: dict[str, int],
    port: Any,
    packet: Any,
    ids: list[int],
    zero_ticks: np.ndarray,
    signs: np.ndarray,
    ref_info: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    viewer: Any | None,
    sync_read_pos: Any,
    sync_read_cur: Any,
    sync_read_pwm: Any,
    sync_write_cur: Any,
) -> dict[str, list[Any]]:
    dt_target = 1.0 / float(args.rate)
    log: dict[str, list[Any]] = {}
    previous_q: np.ndarray | None = None
    previous_t: float | None = None
    dq_filtered = np.zeros(len(ids), dtype=np.float64)
    present_current_ma = np.zeros(len(ids), dtype=np.float64)
    present_pwm_percent = np.zeros(len(ids), dtype=np.float64)
    temperature_c = np.zeros(len(ids), dtype=np.float64)
    hardware_error = np.zeros(len(ids), dtype=np.int64)
    settled_start: float | None = None
    stop_when_settled = should_stop_when_settled(args)
    cycle_index = 0
    t0 = time.perf_counter()
    next_tick = t0

    while True:
        now = time.perf_counter()
        t = now - t0
        if t >= float(args.duration):
            break
        if viewer is not None and not viewer.is_running():
            break

        # Hot loop: position is required every cycle; effort feedback is held
        # between lower-rate reads to keep the current loop responsive.
        ticks = sync_read_ticks(sync_read_pos, packet, ids)
        _raw_q, q, clipped = ticks_to_joint_angles(ticks, zero_ticks, signs, lower, upper)
        if np.any(clipped):
            raise RuntimeError(f"Measured joint angle was clipped by limits: q={q.tolist()}")

        if cycle_index % int(args.effort_feedback_stride) == 0:
            cur_raw = sync_read_unsigned(sync_read_cur, packet, ids, ADDR_PRESENT_CURRENT, 2)
            present_current_ma = np.array(
                [uint16_to_int16(int(v)) * CURRENT_UNIT_MA for v in cur_raw],
                dtype=np.float64,
            )
            pwm_raw = sync_read_unsigned(sync_read_pwm, packet, ids, ADDR_PRESENT_PWM, 2)
            present_pwm_percent = np.array(
                [uint16_to_int16(int(v)) * PWM_UNIT_PERCENT for v in pwm_raw],
                dtype=np.float64,
            )

        if previous_q is None or previous_t is None:
            dq_raw = np.zeros(len(ids), dtype=np.float64)
        else:
            dt_measured = max(t - previous_t, 1e-6)
            dq_raw = (q - previous_q) / dt_measured
        alpha = float(args.velocity_filter_alpha)
        dq_filtered = alpha * dq_raw + (1.0 - alpha) * dq_filtered
        previous_q = q.copy()
        previous_t = t

        # Slow-changing health reads: temperature + hardware error every N cycles
        if cycle_index % int(args.feedback_stride) == 0:
            temperature_c = np.array(
                [
                    read1(port, packet, dxl_id, ADDR_PRESENT_TEMPERATURE, "temperature")
                    for dxl_id in ids
                ],
                dtype=np.float64,
            )
            hardware_error = np.array(
                [
                    read1(port, packet, dxl_id, ADDR_HARDWARE_ERROR_STATUS, "hardware error status")
                    for dxl_id in ids
                ],
                dtype=np.int64,
            )
        # cycle_index not yet incremented — use the previous health values for the first stride

        q_ref, dq_ref = reference_at(t, ref_info)
        error = q_ref - q
        pd_current = (
            float(args.kp_current) * error
            + float(args.kd_current) * (dq_ref - dq_filtered)
        )
        static_active = np.abs(error) > float(args.static_deadband)
        goal_current = apply_static_current_compensation(pd_current, error, args)
        goal_current = np.clip(goal_current, -float(args.max_current_ma), float(args.max_current_ma))
        check_safety(q, dq_filtered, lower, upper, temperature_c, hardware_error, args)

        retry_dxl(
            lambda: sync_write_goal_current(sync_write_cur, packet, ids, goal_current),
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
            label="sync-write goal current",
        )
        write_joint_angles(mujoco, model, data, qpos_index, q)
        if viewer is not None:
            viewer.sync()

        append_log(
            log,
            t=t,
            ticks=ticks,
            q_meas=q,
            dq_raw=dq_raw,
            dq_filtered=dq_filtered,
            present_velocity_rad_s=dq_filtered,
            q_ref=q_ref,
            dq_ref=dq_ref,
            target_q=q_ref,
            error=error,
            pd_current_ma=pd_current,
            static_compensation_active=static_active,
            goal_current_ma=goal_current,
            present_current_ma=present_current_ma,
            present_pwm_percent=present_pwm_percent,
            temperature_c=temperature_c,
            hardware_error=hardware_error,
        )

        if stop_when_settled and np.all(np.abs(error) <= float(args.position_tolerance)):
            if settled_start is None:
                settled_start = t
            elif t - settled_start >= float(args.settle_time):
                print(f"Settled for {args.settle_time:.2f}s at t={t:.3f}s")
                break
        else:
            settled_start = None

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
    parsed = validate_args(args, lower, upper)
    ids = parsed["ids"].astype(int).tolist()
    signs = parsed["signs"]
    ref_info = parsed["ref_info"]

    print(f"Loaded MuJoCo model: {args.xml}")
    print(f"Effective joint limits rad: lower={lower.tolist()}, upper={upper.tolist()}")
    print(f"Reference type: {ref_info['type']}")
    if ref_info["type"] == "point":
        print(f"Targets rad: {ref_info['targets'].tolist()}")
    else:
        print(f"Sine reference rad: min={ref_info['ref_min'].tolist()}, max={ref_info['ref_max'].tolist()}")
        print(
            "Sine parameters: biases={}, amplitudes={}, frequencies={} Hz, phases={} rad".format(
                ref_info["biases"].tolist(),
                ref_info["amplitudes"].tolist(),
                ref_info["frequencies"].tolist(),
                ref_info["phases"].tolist(),
            )
        )
    if args.plan_only:
        return 0

    port = None
    packet = None
    run_dir: Path | None = None
    log: dict[str, list[Any]] = {}
    zero_ticks = np.zeros(len(ids), dtype=np.int64)
    try:
        port, packet = open_dynamixel(args)
        zero_ticks = initialize_current_mode(
            port,
            packet,
            ids,
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )
        write_goal_current(
            port,
            packet,
            ids,
            np.zeros(len(ids), dtype=np.float64),
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )
        initial_ticks = read_ticks(
            port,
            packet,
            ids,
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )
        _raw_q, q_initial, clipped = ticks_to_joint_angles(initial_ticks, zero_ticks, signs, lower, upper)
        if np.any(clipped):
            raise RuntimeError(f"Initial position outside effective limits: q={q_initial.tolist()}")
        write_joint_angles(mujoco, model, data, qpos_index, q_initial)
        confirm_run(args, ref_info, lower, upper)
        run_dir = make_run_dir(args.output_root)

        set_torque(
            port,
            packet,
            ids,
            True,
            retries=int(args.comm_retries),
            delay_s=float(args.comm_retry_delay),
        )

        # Pre-build sync groups for the hot loop. Position is read every cycle;
        # current/PWM and health feedback are lower-rate diagnostics.
        sync_read_pos = create_sync_position_reader(port, packet, ids)
        sync_read_cur = create_sync_read_group(port, packet, ids, ADDR_PRESENT_CURRENT, 2)
        sync_read_pwm = create_sync_read_group(port, packet, ids, ADDR_PRESENT_PWM, 2)
        sync_write_cur = create_sync_current_writer(port, packet)

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
            log = run_loop(
                args,
                mujoco,
                model,
                data,
                qpos_index,
                port,
                packet,
                ids,
                zero_ticks,
                signs,
                ref_info,
                lower,
                upper,
                viewer,
                sync_read_pos,
                sync_read_cur,
                sync_read_pwm,
                sync_write_cur,
            )
        finally:
            if viewer is not None:
                viewer.close()
    finally:
        if port is not None and packet is not None:
            zero_current_safely(
                port,
                packet,
                ids if "ids" in locals() else list(map(int, args.ids)),
                retries=int(args.comm_retries),
                delay_s=float(args.comm_retry_delay),
            )
            set_torque_safely(
                port,
                packet,
                ids if "ids" in locals() else list(map(int, args.ids)),
                False,
                retries=int(args.comm_retries),
                delay_s=float(args.comm_retry_delay),
            )
            port.closePort()

    if run_dir is None:
        raise RuntimeError("Run directory was not created.")
    manifest = make_manifest(args, run_dir, lower, upper, zero_ticks, ref_info)
    save_json(run_dir / "manifest.json", manifest)
    if log:
        metrics = compute_metrics(log, ref_info, float(args.position_tolerance))
        figure_paths = save_log(run_dir, log, metrics, ref_info)
        print(f"Saved log: {run_dir / 'arrays' / 'current_mode_log.npz'}")
        print(f"Saved metrics: {run_dir / 'metrics' / 'current_mode_metrics.json'}")
        print(f"Saved figures: {[str(path) for path in figure_paths]}")
        print(f"Final error rad: {metrics['final_error_rad']}")
    else:
        print("No samples collected; manifest saved only.")
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
