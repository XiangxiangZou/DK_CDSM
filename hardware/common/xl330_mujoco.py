"""Shared XL330/DYNAMIXEL and MuJoCo helpers for hardware scripts."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认 MuJoCo XML 模型文件路径
DEFAULT_XML = PROJECT_ROOT / "hardware" / "assets" / "multi_joint_space_robot.xml"

# ==============================================================================
# 全局常量：Dynamixel Protocol 2.0 控制表地址
# ==============================================================================

# Dynamixel Protocol 2.0（与 Protocol 1.0 不兼容）
PROTOCOL_VERSION = 2.0

# ---------- EEPROM 区域（掉电保存，慎写） ----------
ADDR_MODEL_NUMBER      = 0    # 型号编号（如 XL330-M288-T = 1190）
ADDR_FIRMWARE_VERSION  = 6    # 固件版本号（1 字节）
ADDR_OPERATING_MODE    = 11   # 工作模式（1 字节）：4=扩展位置控制模式

# ---------- RAM 区域（掉电丢失） ----------
ADDR_TORQUE_ENABLE         = 64   # 扭矩使能（1 字节）：0=脱力/自由, 1=使能/锁定
ADDR_HARDWARE_ERROR_STATUS = 70   # 硬件错误状态（1 字节）：0=正常
ADDR_PROFILE_ACCELERATION  = 108  # 位置控制模式的加速度档位（4 字节）
ADDR_PROFILE_VELOCITY      = 112  # 位置控制模式的速度档位（4 字节）
ADDR_GOAL_POSITION         = 116  # 目标位置（4 字节，单位：encoder tick）
ADDR_PRESENT_POSITION      = 132  # 当前实际位置（4 字节，单位：encoder tick）
ADDR_PRESENT_INPUT_VOLTAGE = 144  # 当前输入电压（2 字节，单位：0.1V）
ADDR_PRESENT_TEMPERATURE   = 146  # 当前温度（1 字节，单位：°C）

# ==============================================================================
# 全局常量：舵机控制参数
# ==============================================================================

# 扭矩开关状态
TORQUE_DISABLE = 0  # 扭矩禁用（舵机可被手拧动）
TORQUE_ENABLE  = 1  # 扭矩使能（舵机锁定/主动驱动）

# 工作模式：扩展位置控制模式
# 特点：支持多圈绝对位置（超出 ±360° 也不会回绕），不支持速度/电流控制
OPERATING_MODE_EXTENDED_POSITION = 4

# XL330 编码器分辨率：4096 ticks / 转
TICKS_PER_REVOLUTION = 4096

# 每个 tick 对应的弧度 = 2π / 4096 ≈ 0.001534 rad ≈ 0.088°
RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REVOLUTION

# ==============================================================================
# 全局常量：MuJoCo 关节名
# ==============================================================================

# MuJoCo 模型中所有关节的名字（joint1/joint2 是 ID10 驱动的主-从关节对，
# joint3/joint4 是 ID20 驱动的主-从关节对）
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")

# 其中真正由硬件舵机**主动驱动**的关节（主关节），从关节通过 MuJoCo 的
# equality/mimic 约束自动跟随
ACTIVE_JOINT_NAMES = ("joint1", "joint3")


# ==============================================================================
# 数据结构：舵机读数（ServoReading）
# ==============================================================================

@dataclass(frozen=True)  # frozen=True → 不可变对象，类似 namedtuple 但更灵活
class ServoReading:
    """一次完整的舵机状态快照（不可变）。"""
    dxl_id: int                # 舵机 ID（1-253）
    ticks: int                 # 当前编码器位置（有符号 int32 转换后）
    model_number: int          # 型号编号（应与 ping 结果一致）
    firmware_version: int      # 固件版本
    input_voltage_v: float     # 输入电压（V），实际值 = 寄存器值 × 0.1
    temperature_c: int         # 温度（°C）
    hardware_error_status: int # 硬件错误标志位（0=正常）


# ==============================================================================
ADDR_LED = 65
ADDR_GOAL_CURRENT = 102
ADDR_MOVING = 122
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
LED_OFF = 0
LED_ON = 1
OPERATING_MODE_CURRENT = 0
OPERATING_MODE_VELOCITY = 1
OPERATING_MODE_POSITION = 3
CURRENT_UNIT_MA = 1.0
PWM_UNIT_PERCENT = 0.113
PRESENT_VELOCITY_UNIT_RPM = 0.229

class DynamixelError(RuntimeError):
    """Dynamixel SDK 通信错误 或 舵机端硬件错误时抛出。"""


# ==============================================================================
# 延迟加载函数
# ==============================================================================

def load_dynamixel_sdk() -> tuple[type[Any], type[Any]]:
    """尝试导入 Dynamixel SDK。

    设计意图：延迟导入而非放在文件头部 import，这样：
      1. 仅在真正使用硬件时才需要安装 SDK，
      2. 即使在无 SDK 的机器上，仍可 import 本模块的其他部分。

    Returns:
        (PortHandler 类, PacketHandler 类)

    Raises:
        SystemExit: 若未安装 dynamixel-sdk 包，打印安装提示并退出。
    """
    try:
        from dynamixel_sdk import PacketHandler, PortHandler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: dynamixel_sdk. Install it in the project environment:\n"
            "& 'D:\\Apps\\Anaconda3\\envs\\env_dk_cdsm\\python.exe' -m pip install dynamixel-sdk"
        ) from exc
    return PortHandler, PacketHandler


def load_mujoco():
    """尝试导入 MuJoCo 库。

    同样采用延迟加载策略：
      1. 无需硬件时不必装 MuJoCo，
      2. 无 MuJoCo 环境下仍可 import 本模块。
    """
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: mujoco. Install it in the project environment before running sync."
        ) from exc
    return mujoco


# ==============================================================================
# 整型转换工具函数（二补数 ↔ 无符号整数）
# ==============================================================================

def uint32_to_int32(value: int) -> int:
    """将 32 位无符号整数的字节模式重新解释为 32 位有符号整数。

    Dynamixel SDK 的 read4ByteTxRx 返回无符号 Python int（范围为 [0, 2³²-1]），
    但 XL330 扩展位置模式下的编码器值是**有符号 32 位**（可正可负，表示多圈位置）。

    转换原理：
      1. 取低 32 位（& 0xFFFFFFFF）。
      2. 若最高位（bit31）= 1（即数值 ≥ 2³¹），说明是负数：
         减去 2³² 得到对应的有符号负数（二补数表示）。

    Examples:
        0x00000000 →  0
        0x00000001 →  1
        0x7FFFFFFF →  2147483647  (最大正数)
        0x80000000 → -2147483648  (最小负数)
        0xFFFFFFFF → -1
    """
    value &= 0xFFFFFFFF           # 确保只取低 32 位
    if value & 0x80000000:         # 检查 bit31（符号位）
        return value - 0x100000000 # 0x100000000 = 2³²
    return value


def int32_to_uint32(value: int) -> int:
    """将 32 位有符号整数转换为无符号整数（写入寄存器时使用）。

    Dynamixel 寄存器的 4 字节写操作需要正的 Python int。

    Examples:
        0  → 0x00000000
        1  → 0x00000001
        -1 → 0xFFFFFFFF
    """
    return int(value) & 0xFFFFFFFF


# ==============================================================================
# 通信结果检查
# ==============================================================================

def check_packet(packet: Any, result: int, error: int, action: str) -> None:
    """检查 Dynamixel 通信包的执行结果。

    每次读写操作后都应调用此函数：
      - result：通信层结果码（0 = 成功，非 0 = UART/超时/CRC 等通信故障）
      - error ：舵机端硬件错误标志（0 = 正常，非 0 = 过载/过热/过压等硬件故障）

    Args:
        packet: Dynamixel PacketHandler 实例
        result: 通信结果码
        error:  硬件错误码
        action: 操作描述（用于错误信息，如 "read present position from ID 10"）

    Raises:
        DynamixelError: 若通信失败或舵机报错
    """
    if result != 0:
        raise DynamixelError(f"{action} failed: {packet.getTxRxResult(result)}")
    if error != 0:
        raise DynamixelError(f"{action} device error: {packet.getRxPacketError(error)}")


# ==============================================================================
# 底层读写函数（封装 Dynamixel SDK 的协议操作）
# ==============================================================================

def read1(port: Any, packet: Any, dxl_id: int, address: int, name: str) -> int:
    """读 1 字节（Byte）寄存器。

    Returns:
        int: 寄存器值 [0, 255]

    用途举例：读取固件版本、温度、硬件错误状态、扭矩使能、工作模式。
    """
    value, result, error = packet.read1ByteTxRx(port, dxl_id, address)
    check_packet(packet, result, error, f"read {name} from ID {dxl_id}")
    return int(value)


def read2(port: Any, packet: Any, dxl_id: int, address: int, name: str) -> int:
    """读 2 字节（Word）寄存器。

    Returns:
        int: 寄存器值 [0, 65535]

    用途举例：读取型号编号、输入电压。
    """
    value, result, error = packet.read2ByteTxRx(port, dxl_id, address)
    check_packet(packet, result, error, f"read {name} from ID {dxl_id}")
    return int(value)


def read4(port: Any, packet: Any, dxl_id: int, address: int, name: str) -> int:
    """读 4 字节（DWord）寄存器。

    Returns:
        int: 无符号 32 位值。注意：后续需用 uint32_to_int32 转换为有符号。

    用途举例：读取当前位置、目标位置、Profile Velocity、Profile Acceleration。
    """
    value, result, error = packet.read4ByteTxRx(port, dxl_id, address)
    check_packet(packet, result, error, f"read {name} from ID {dxl_id}")
    return int(value)


def write1(port: Any, packet: Any, dxl_id: int, address: int, value: int, name: str) -> None:
    """写 1 字节（Byte）寄存器。

    用途举例：使能/禁用扭矩、设置工作模式。
    """
    result, error = packet.write1ByteTxRx(port, dxl_id, address, int(value))
    check_packet(packet, result, error, f"write {name}={value} to ID {dxl_id}")


def write4(port: Any, packet: Any, dxl_id: int, address: int, value: int, name: str) -> None:
    """写 4 字节（DWord）寄存器。

    注意：写入位置值时，需先调用 int32_to_uint32 将有符号 tick 转为无符号整型。
    用途举例：设置目标位置、Profile Velocity、Profile Acceleration。
    """
    result, error = packet.write4ByteTxRx(port, dxl_id, address, int(value))
    check_packet(packet, result, error, f"write {name}={value} to ID {dxl_id}")


# ==============================================================================
# 同步读写函数：一条指令操作多个舵机（减少 UART 往返）
# ==============================================================================

def _import_sync_classes() -> tuple[type[Any], type[Any], Any]:
    """Lazy-import GroupSyncRead / GroupSyncWrite / COMM_SUCCESS."""
    try:
        from dynamixel_sdk import (
            COMM_SUCCESS,
            GroupSyncRead,
            GroupSyncWrite,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: dynamixel_sdk."
        ) from exc
    return GroupSyncRead, GroupSyncWrite, COMM_SUCCESS


def create_sync_position_reader(port: Any, packet: Any, ids: list[int]) -> Any:
    """Create and return a pre-configured GroupSyncRead for Present Position (4 bytes).

    The returned group can be reused across cycles — just call
    :func:`sync_read_ticks` each frame.  Do NOT call ``addParam`` / ``clearParam``
    on the returned object yourself.
    """
    GroupSyncRead, _GroupSyncWrite, _comm = _import_sync_classes()
    group = GroupSyncRead(port, packet, ADDR_PRESENT_POSITION, 4)
    for dxl_id in ids:
        if not group.addParam(dxl_id):
            raise DynamixelError(
                f"Failed to add ID {dxl_id} to sync-read group (addr={ADDR_PRESENT_POSITION})"
            )
    return group


def sync_read_ticks(group: Any, packet: Any, ids: list[int]) -> np.ndarray:
    """Read Present Position ticks from a pre-configured GroupSyncRead group.

    Returns:
        np.ndarray (int64): signed tick values, one per servo in *ids* order.
    """
    _GroupSyncRead, _GroupSyncWrite, COMM_SUCCESS = _import_sync_classes()
    result = group.txRxPacket()
    if result != COMM_SUCCESS:
        raise DynamixelError(
            f"Sync-read present position failed: {packet.getTxRxResult(result)}"
        )
    ticks = []
    for dxl_id in ids:
        value = group.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
        ticks.append(uint32_to_int32(value))
    return np.array(ticks, dtype=np.int64)


def create_sync_goal_writer(port: Any, packet: Any, ids: list[int]) -> Any:
    """Create a GroupSyncWrite for Goal Position (4 bytes).

    The returned group must be re-armed via :func:`sync_write_goal_ticks` each
    cycle (it calls ``clearParam`` / ``addParam`` internally).
    """
    _GroupSyncRead, GroupSyncWrite, _comm = _import_sync_classes()
    group = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 4)
    return group


def sync_write_goal_ticks(
    group: Any, packet: Any, ids: list[int], goal_ticks: np.ndarray,
) -> None:
    """Write goal-position ticks to all servos in a single Sync Write transaction.

    This is a broadcast-style write (no per-servo status packets), so the
    transaction is very fast.  On failure a :exc:`DynamixelError` is raised.
    """
    _GroupSyncRead, _GroupSyncWrite, COMM_SUCCESS = _import_sync_classes()
    group.clearParam()
    for dxl_id, tick in zip(ids, goal_ticks):
        raw = int32_to_uint32(int(tick))
        data = [
            (raw >> 0) & 0xFF,
            (raw >> 8) & 0xFF,
            (raw >> 16) & 0xFF,
            (raw >> 24) & 0xFF,
        ]
        if not group.addParam(dxl_id, data):
            raise DynamixelError(
                f"Failed to add goal-position data for ID {dxl_id} to sync-write group"
            )
    result = group.txPacket()
    if result != COMM_SUCCESS:
        raise DynamixelError(
            f"Sync-write goal position failed: {packet.getTxRxResult(result)}"
        )


def create_sync_read_group(
    port: Any, packet: Any, ids: list[int], address: int, data_length: int,
) -> Any:
    """Create a GroupSyncRead for an arbitrary address and data length.

    The returned group can be reused across cycles.  Use
    :func:`sync_read_unsigned` to read the values each frame.
    """
    GroupSyncRead, _GroupSyncWrite, _comm = _import_sync_classes()
    group = GroupSyncRead(port, packet, address, data_length)
    for dxl_id in ids:
        if not group.addParam(dxl_id):
            raise DynamixelError(
                f"Failed to add ID {dxl_id} to sync-read group (addr={address}, len={data_length})"
            )
    return group


def sync_read_unsigned(
    group: Any, packet: Any, ids: list[int], address: int, data_length: int,
) -> np.ndarray:
    """Read unsigned integer values from a pre-configured GroupSyncRead group.

    Returns:
        np.ndarray (int64): raw unsigned register values, one per servo.
    """
    _GroupSyncRead, _GroupSyncWrite, COMM_SUCCESS = _import_sync_classes()
    result = group.txRxPacket()
    if result != COMM_SUCCESS:
        raise DynamixelError(
            f"Sync-read (addr={address}) failed: {packet.getTxRxResult(result)}"
        )
    values = [int(group.getData(dxl_id, address, data_length)) for dxl_id in ids]
    return np.array(values, dtype=np.int64)


# ==============================================================================
# Ping 函数：舵机在线检测
# ==============================================================================

def ping(port: Any, packet: Any, dxl_id: int) -> int:
    """向舵机发送 Ping 指令，检测舵机是否在线。

    Ping 是 Dynamixel 协议中最轻量的指令：舵机收到后立即回复自己的型号编号。

    Returns:
        int: 舵机型号编号（model number），可用于验证型号是否正确。

    Raises:
        DynamixelError: 若指定 ID 的舵机无应答（掉线/断电/ID 错误）。
    """
    model_number, result, error = packet.ping(port, dxl_id)
    check_packet(packet, result, error, f"ping ID {dxl_id}")
    return int(model_number)


# ==============================================================================
# 一次性批量读取舵机完整状态
# ==============================================================================

def read_servo(port: Any, packet: Any, dxl_id: int) -> ServoReading:
    """从指定 ID 的舵机读取完整的健康状态快照。

    读取内容（共 7 项）：
      1. 型号编号（2 字节，验证舵机型号）
      2. 固件版本（1 字节）
      3. 当前位置（4 字节，扩展位置模式 → 有符号 32 位 tick 值）
      4. 输入电压（2 字节，× 0.1 = V）
      5. 芯片温度（1 字节，°C）
      6. 硬件错误状态（1 字节）

    注意：每个读取都是独立的 UART 指令-应答往返，7 次读取 ≈ 7 个往返延迟。
    对实时性有更高要求时，可用 Dynamixel 的 Sync Read（同步读）指令一次读取全部。

    Returns:
        ServoReading: 包含全部状态值的不可变对象
    """
    model_number = read2(port, packet, dxl_id, ADDR_MODEL_NUMBER, "model number")
    firmware_version = read1(port, packet, dxl_id, ADDR_FIRMWARE_VERSION, "firmware version")
    # 当前位置：4 字节无符号 → 有符号 int32（支持负多圈位置）
    ticks = uint32_to_int32(
        read4(port, packet, dxl_id, ADDR_PRESENT_POSITION, "present position")
    )
    # 输入电压：寄存器值 × 0.1 = 实际电压（如 120 → 12.0V）
    input_voltage_v = 0.1 * read2(
        port, packet, dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, "present input voltage"
    )
    temperature_c = read1(port, packet, dxl_id, ADDR_PRESENT_TEMPERATURE, "temperature")
    hardware_error_status = read1(
        port, packet, dxl_id, ADDR_HARDWARE_ERROR_STATUS, "hardware error status"
    )
    return ServoReading(
        dxl_id=dxl_id,
        ticks=ticks,
        model_number=model_number,
        firmware_version=firmware_version,
        input_voltage_v=input_voltage_v,
        temperature_c=temperature_c,
        hardware_error_status=hardware_error_status,
    )
def name_to_joint_id(mujoco: Any, model: Any, name: str) -> int:
    """将 MuJoCo 关节名字符串转换为关节 ID（整数索引）。

    调用 mujoco.mj_name2id() 查找名为 name 的关节对象在模型中的索引。

    Args:
        mujoco: MuJoCo 模块
        model:  MuJoCo 模型对象
        name:   关节名字符串（如 "joint1"）

    Returns:
        int: 关节索引（≥ 0）

    Raises:
        ValueError: 若 XML 模型中不存在该关节名
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in XML: {name}")
    return int(jid)


def build_joint_indices(mujoco: Any, model: Any) -> dict[str, int]:
    """构建关节名 → qpos 数组索引的映射字典。

    在 MuJoCo 中，每个关节在状态向量 qpos（广义位置）中有自己的起始地址
    （jnt_qposadr），通过这个地址可以直接索引 qpos 数组。

    Returns:
        dict[str, int]:
            {"joint1": qpos_index1, "joint2": qpos_index2,
             "joint3": qpos_index3, "joint4": qpos_index4}
    """
    joint_ids = {name: name_to_joint_id(mujoco, model, name) for name in JOINT_NAMES}
    return {name: int(model.jnt_qposadr[jid]) for name, jid in joint_ids.items()}


# ==============================================================================
# 有效关节限位计算
# ==============================================================================

def read_active_joint_limits(mujoco: Any, model: Any) -> tuple[np.ndarray, np.ndarray]:
    """读取 XML 模型中两个**主动关节**的硬限位范围。

    前提条件：joint1 和 joint3 必须在 XML 中定义了 limited=true 和 range 属性。

    Returns:
        lower: np.ndarray shape=(2,), dtype=float64  — 两个关节的下限
        upper: np.ndarray shape=(2,), dtype=float64  — 两个关节的上限

    Raises:
        ValueError: 若某个主动关节在 XML 中未定义限位
    """
    lower = np.empty(2, dtype=np.float64)
    upper = np.empty(2, dtype=np.float64)
    for index, name in enumerate(ACTIVE_JOINT_NAMES):  # ("joint1", "joint3")
        joint_id = name_to_joint_id(mujoco, model, name)
        if int(model.jnt_limited[joint_id]) == 0:
            raise ValueError(f"Active joint {name} has no XML range limit.")
        # model.jnt_range[joint_id] 是 shape=(2,) 的数组 [lower, upper]
        lower[index], upper[index] = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    return lower, upper


def effective_joint_limits(
    xml_lower: np.ndarray,
    xml_upper: np.ndarray,
    *,
    limit_ratio: float,
    symmetric_limit: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算**有效**关节限位 = XML 限位 × 比例系数 ∩ 可选对称限位。

    计算步骤：
      1. limit_ratio 缩放：以 XML 范围中心为轴，将半范围乘以 ratio
         —— 缩小 ratio 可让关节在更大安全余量内运动
      2. symmetric_limit 裁剪：若提供 --limit 值，将范围与 [-limit, +limit] 取交集
         —— 确保关节角不会超过对称的绝对限位

    Args:
        xml_lower:       XML 原始下限（每关节）
        xml_upper:       XML 原始上限（每关节）
        limit_ratio:     缩放比例 (0, 1.0]
        symmetric_limit: 可选对称限位值（弧度），None 表示不使用

    Returns:
        (effective_lower, effective_upper): 最终限位数组

    Raises:
        ValueError: 若 ratio 参数无效，或最终下限 ≥ 上限
    """
    ratio = float(limit_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("--limit-ratio must be in (0, 1].")

    # 以 XML 范围中心为轴缩放
    center = 0.5 * (xml_lower + xml_upper)               # 中心点
    half_range = 0.5 * (xml_upper - xml_lower) * ratio    # 半范围 × 比例
    lower = center - half_range
    upper = center + half_range

    # 可选对称限位：取更严格者
    if symmetric_limit is not None:
        limit = float(symmetric_limit)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("--limit must be positive when provided.")
        lower = np.maximum(lower, -limit)  # lower 不能低于 -limit
        upper = np.minimum(upper, limit)   # upper 不能高于 +limit

    if np.any(lower >= upper):
        raise ValueError(f"Invalid effective limits: lower={lower}, upper={upper}")
    return lower, upper


# ==============================================================================
# 坐标转换：tick ↔ 关节角
# ==============================================================================

def ticks_to_joint_angles(
    ticks: np.ndarray,
    zero_ticks: np.ndarray,
    signs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将舵机编码器 tick 值转换为 MuJoCo 关节角（弧度）。

    转换公式：
        raw_q = sign × (tick_current - tick_zero) × RAD_PER_TICK
        q     = clip(raw_q, lower, upper)

    各变量的含义：
        ticks:       当前舵机编码器读数（有符号 int32）
        zero_ticks:  零点偏移（"零位"对应的编码器值）
        signs:       方向符号（+1 同向，-1 反向）
        lower/upper: 有效关节限位

    Returns:
        raw_q:     原始（未限幅）关节角
        clipped_q: 限幅后关节角
        clipped:   bool 数组，True 表示对应关节被限幅了

    Example:
        tick=2048, zero=0, sign=+1
        → raw_q = 2048 × (2π/4096) = π rad ≈ 180°
    """
    raw_q = signs * (ticks - zero_ticks) * RAD_PER_TICK
    clipped_q = np.clip(raw_q, lower, upper)              # NumPy 逐元素限幅
    clipped = np.abs(raw_q - clipped_q) > 1e-12           # 检测是否被裁剪（浮点容差）
    return raw_q, clipped_q, clipped


def joint_angle_to_ticks(q: float, zero_tick: int, sign: float) -> int:
    """反向转换：MuJoCo 关节角 → 舵机编码器 tick 值。

    用于软限位功能：当手拧舵机超出限位时，计算应推回的目标 tick 值。

    公式：
        target_tick = round( zero_tick + sign × q / RAD_PER_TICK )

    Args:
        q:         目标关节角（弧度）
        zero_tick: 零点编码器值
        sign:      方向符号

    Returns:
        int: 对应的舵机编码器 tick 值（四舍五入到整数）
    """
    return int(round(float(zero_tick) + float(sign) * float(q) / RAD_PER_TICK))


# ==============================================================================
# 软限位执行：主动制动越界舵机
# ==============================================================================

def enforce_servo_joint_limits(
    *,
    port: Any,
    packet: Any,
    ids: list[int],
    raw_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    zero_ticks: np.ndarray,
    signs: np.ndarray,
    active: np.ndarray,
) -> None:
    """对超越有效限位的舵机施加软件限位制动。

    核心逻辑（每个舵机独立处理）：

    情况 A — 关节角在限位内（lower ≤ raw_q ≤ upper）：
        ├─ 若该舵机之前处于制动状态 (active=True)
        │   → 关闭扭矩 (TORQUE_DISABLE)，让用户重新自由拖拽
        └─ 若本就自由 (active=False)
            → 不做任何操作

    情况 B — 关节角越界（raw_q < lower 或 raw_q > upper）：
        ├─ 计算限位边界对应的目标 tick 值
        ├─ 将 GOAL_POSITION 设为该边界值
        └─ 若之前扭矩已关闭 → 打开扭矩
           舵机会自动以 profile_velocity 速度向目标位置运动
           → 效果：舵机"弹回"到限位边界处

    设计意图：
        这是一种"软墙"机制。手拧舵机时碰不到物理限位，但仿真模型有角限位。
        当用户拧过仿真限位时，舵机自动使能并推回边界，给用户一个触觉反馈，
        同时确保仿真模型中的关节角不会超出有效范围。

    active 数组是**可变**的（传入引用），此函数会直接修改它。
    """
    for index, dxl_id in enumerate(ids):
        boundary_q: float | None = None  # 限位边界对应的关节角（若越界则非 None）

        # 判断是否越界及越界方向
        if raw_q[index] < lower[index]:
            boundary_q = float(lower[index])  # 低于下限 → 推回下限
        elif raw_q[index] > upper[index]:
            boundary_q = float(upper[index])  # 超出上限 → 推回上限

        # ---- 情况 A：未越界，恢复自由 ----
        if boundary_q is None:
            if bool(active[index]):
                # 之前在制动 → 关闭扭矩，用户重新自由拖拽
                write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque enable")
                active[index] = False
            continue  # 无需进一步操作

        # ---- 情况 B：越界，执行软限位推回 ----
        # 将关节角边界值转换回 tick
        boundary_tick = joint_angle_to_ticks(boundary_q, int(zero_ticks[index]), float(signs[index]))

        # 设置目标位置为限位边界（舵机会自动向此位置运动）
        write4(
            port, packet, dxl_id, ADDR_GOAL_POSITION,
            int32_to_uint32(boundary_tick), "goal position",
        )

        # 若之前扭矩关闭 → 打开扭矩（舵机开始向 boundary_tick 运动）
        if not bool(active[index]):
            write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "torque enable")
            active[index] = True


# ==============================================================================
# 将关节角写入 MuJoCo 模型
# ==============================================================================

def write_joint_angles(
    mujoco: Any,
    model: Any,
    data: Any,
    qpos_index: dict[str, int],
    q: np.ndarray,
) -> None:
    """将两个主动关节角同时写入 MuJoCo 数据结构的 qpos 向量。

    MuJoCo 的 equality/mimic 约束会自动将 joint2 跟随 joint1，
    joint4 跟随 joint3，所以只需写入主关节的值。

    写入后调用 mj_forward() 执行一次正向动力学计算，更新所有派生量
    （关节速度、加速度、外力等）。同时将速度和驱动力置零，
    因为这是纯运动学镜像——舵机位置直接映射，无动力学效应。

    Args:
        mujoco:     MuJoCo 模块
        model:      MuJoCo 模型对象
        data:       MuJoCo 数据对象（状态容器）
        qpos_index: {"joint1": idx, "joint2": idx, "joint3": idx, "joint4": idx}
        q:          [qa, qb] 两个主动关节角（弧度）
    """
    qa, qb = float(q[0]), float(q[1])

    # 主关节赋值（从关节通过 mimic 约束自动跟随）
    data.qpos[qpos_index["joint1"]] = qa
    data.qpos[qpos_index["joint2"]] = qa  # ID10 驱动的 mimic 从动关节
    data.qpos[qpos_index["joint3"]] = qb
    data.qpos[qpos_index["joint4"]] = qb  # ID20 驱动的 mimic 从动关节

    # 清除速度和驱动力（纯运动学镜像，不计动力学）
    data.qvel[:] = 0.0   # 广义速度 = 0
    data.ctrl[:] = 0.0   # 控制输入 = 0

    # 执行正向运动学/动力学计算，更新所有派生状态
    mujoco.mj_forward(model, data)


# ==============================================================================
# 查看器相机配置
# ==============================================================================

def configure_viewer_camera(viewer: Any, args: argparse.Namespace) -> None:
    """设置 MuJoCo 被动查看器的自由相机参数。

    viewer.cam 是一个 mjvCamera 结构体，控制渲染视角。

    视点 = lookat（注视点）+
           distance（距离）× direction(azimuth, elevation)（球坐标方向）

    默认 azimuth=90°, elevation=-90° → 相机从正上方垂直向下俯视机械臂平面。

    Args:
        viewer: MuJoCo 查看器对象
        args:   命令行参数（包含相机参数）
    """
    viewer.cam.lookat[:] = [
        float(args.camera_lookat_x),
        float(args.camera_lookat_y),
        float(args.camera_lookat_z),
    ]
    viewer.cam.distance  = float(args.camera_distance)
    viewer.cam.azimuth   = float(args.camera_azimuth)
    viewer.cam.elevation = float(args.camera_elevation)


# ==============================================================================
# Dynamixel 端口初始化
# ==============================================================================

def open_dynamixel(args: argparse.Namespace) -> tuple[Any, Any]:
    """打开串口并初始化 Dynamixel 通信层。

    步骤：
      1. 加载 SDK（延迟导入）
      2. 实例化 PortHandler（串口管理）和 PacketHandler（协议封包/解包）
      3. 打开指定串口
      4. 设置波特率

    Returns:
        (port, packet): PortHandler 和 PacketHandler 实例

    Raises:
        SystemExit: 若串口不存在或波特率设置失败
    """
    port_handler, packet_handler = load_dynamixel_sdk()
    port = port_handler(args.port)
    packet = packet_handler(PROTOCOL_VERSION)

    print(f"Opening {args.port} at {args.baud} baud")
    if not port.openPort():
        raise SystemExit(f"Failed to open port {args.port}")
    if not port.setBaudRate(args.baud):
        port.closePort()  # 打开成功但波特率设置失败 → 关闭串口后退出
        raise SystemExit(f"Failed to set baudrate {args.baud}")
    return port, packet


# ==============================================================================
# 舵机初始化
# ==============================================================================

def initialize_servos(port: Any, packet: Any, ids: list[int]) -> np.ndarray:
    """初始化所有舵机：检测、脱力、设模式、读取初始位置。

    对每个舵机依次执行：
      1. Ping          —— 确认在线，获取型号
      2. 关闭扭矩      —— 让用户可用手拧动
      3. 设为扩展位置控制模式 —— 多圈绝对位置模式
      4. 读取完整状态  —— 验证型号一致性 & 无硬件错误
      5. 打印状态信息

    Returns:
        np.ndarray (dtype=int64): 各舵机的初始编码器位置 [tick_ID10, tick_ID20]

    Raises:
        DynamixelError: 若 ping 型号与寄存器型号不一致，或舵机有硬件错误
    """
    initial_ticks = []
    for dxl_id in ids:
        # 1. 在线检测
        ping_model = ping(port, packet, dxl_id)

        # 2. 扭矩禁用 → 舵机进入"自由旋转"模式
        write1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque enable")

        # 3. 设为扩展位置控制模式
        #    此模式支持多圈（超出 ±360° 不回绕），编码器值随转圈线性增长
        write1(
            port, packet, dxl_id, ADDR_OPERATING_MODE,
            OPERATING_MODE_EXTENDED_POSITION, "operating mode",
        )

        # 4. 读取当前完整状态
        reading = read_servo(port, packet, dxl_id)

        # 4a. 安全检查：Ping 型号 vs 寄存器型号必须一致
        if reading.model_number != ping_model:
            raise DynamixelError(
                f"ID {dxl_id}: ping model {ping_model} != register model {reading.model_number}"
            )

        # 4b. 安全检查：无硬件错误标志
        if reading.hardware_error_status != 0:
            raise DynamixelError(
                f"ID {dxl_id}: hardware error status 0x{reading.hardware_error_status:02X}"
            )

        # 5. 打印舵机信息
        print(
            "ID {id}: model={model}, fw={fw}, ticks={ticks}, vin={vin:.1f} V, "
            "temp={temp} C, torque=disabled".format(
                id=dxl_id,
                model=reading.model_number,
                fw=reading.firmware_version,
                ticks=reading.ticks,
                vin=reading.input_voltage_v,
                temp=reading.temperature_c,
            )
        )
        initial_ticks.append(reading.ticks)

    return np.asarray(initial_ticks, dtype=np.int64)
