"""末端笛卡尔空间参考轨迹生成器。

本模块只描述“希望末端怎么走”，不做 IK、不做控制、不访问模型权重。三类控制
方法 EDMD/DKUC/DKAC 都应通过同一份 `xy_ref -> IK -> q_ref` 结果进行比较，避免
不同方法使用不同参考轨迹造成不公平。

当前提供三类默认轨迹：
1. `figure8`: 8 字/Lissajous 轨迹，适合检查模型在方向频繁切换时的预测和跟踪能力。
2. `circle`: 圆/椭圆轨迹，适合检查连续闭合曲线跟踪能力。
3. `square`: 正方形/矩形轨迹，适合检查直线段跟踪和拐角处的闭环响应。

默认时间规划使用五次多项式：
- 圆/8 字：对整条路径相位做五次时间缩放，保证起点和终点速度、加速度为 0。
- 正方形：每条边单独做五次插值，保证拐角处速度、加速度为 0。

后续如需改轨迹，优先改本文件中的 `generate_cartesian_reference` 或新增 kind，
不要在各控制脚本里分别手写轨迹公式。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


TRAJECTORY_KINDS = ("figure8", "circle", "square")
TIME_SCALINGS = ("quintic", "linear")


@dataclass(frozen=True)
class CartesianReferenceConfig:
    """笛卡尔参考轨迹参数。

    参数:
        kind: 轨迹类型。当前支持 `figure8`、`circle` 和 `square`。
        dt: 采样周期，单位 s；应与闭环控制周期一致。
        period: 单个轨迹周期，单位 s。越大轨迹越慢，闭环跟踪越容易。
        num_cycles: 轨迹周期数。总时长约为 `period*num_cycles + start_hold`。
        center_x: 轨迹中心的世界系 x 坐标，单位 m。默认 5.0 m，参考旧脚本经验，
            可让末端远离基座折叠和绳驱几何退化区域。
        center_y: 轨迹中心的世界系 y 坐标，单位 m。
        radius_x: x 方向半幅，单位 m。过大会使 IK 关节角接近限位。
        radius_y: y 方向半幅，单位 m。`circle` 中若与 `radius_x` 相等就是圆，
            否则就是椭圆；`square` 中若与 `radius_x` 相等就是正方形，否则就是矩形。
        phase: 轨迹相位，单位 rad。用于调整轨迹起点。
        start_hold: 起点保持时间，单位 s。真实机械臂部署时可用于先稳定在起点。
        time_scaling: 时间规划方式。`quintic` 使用五次多项式平滑插值；`linear` 保留
            原始匀速相位/线性分段，用于对照实验。
    """

    kind: str = "figure8"
    dt: float = 0.01
    period: float = 8.0
    num_cycles: float = 1.0
    center_x: float = 5.0
    center_y: float = 0.0
    radius_x: float = 0.6
    radius_y: float = 0.35
    phase: float = 0.0
    start_hold: float = 0.0
    time_scaling: str = "quintic"


def _time_vector(dt: float, period: float, num_cycles: float, start_hold: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成总时间 `t` 和扣除 hold 后的轨迹内部时间 `tau`。"""
    total_time = float(period) * float(num_cycles) + float(start_hold)
    n = int(round(total_time / float(dt))) + 1
    t = np.arange(n, dtype=np.float64) * float(dt)
    tau = np.maximum(t - float(start_hold), 0.0)
    return t, tau


def _quintic_smoothstep(s: np.ndarray) -> np.ndarray:
    """五次多项式时间缩放函数。

    参数:
        s: 归一化时间，通常在 `[0,1]` 内。函数会先裁剪到该范围。

    返回:
        `10s^3 - 15s^4 + 6s^5`。该函数满足：
        - `h(0)=0, h(1)=1`
        - `h'(0)=h'(1)=0`
        - `h''(0)=h''(1)=0`
        因此用于轨迹插值时，段起点和终点速度、加速度均为 0。
    """
    s_clip = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return 10.0 * s_clip**3 - 15.0 * s_clip**4 + 6.0 * s_clip**5


def _path_theta(
    *,
    tau: np.ndarray,
    period: float,
    num_cycles: float,
    phase: float,
    time_scaling: str,
) -> np.ndarray:
    """生成圆/8 字等闭合曲线的路径相位。

    参数:
        tau: 扣除起点保持后的轨迹内部时间，单位 s。
        period: 单周期时长，单位 s。
        num_cycles: 轨迹周期数。
        phase: 起始相位，单位 rad。
        time_scaling: `quintic` 或 `linear`。

    说明:
        对圆和 8 字这类连续闭合曲线，五次多项式作用在整条轨迹的相位上。这样不会在
        多周期中间强制停车，只在整段轨迹起点和终点实现零速度/零加速度。
    """
    total_motion_time = max(float(period) * float(num_cycles), 1e-12)
    if time_scaling == "quintic":
        progress = _quintic_smoothstep(tau / total_motion_time)
        return float(phase) + 2.0 * np.pi * float(num_cycles) * progress
    return float(phase) + 2.0 * np.pi * tau / float(period)


def _square_reference(
    *,
    theta: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    time_scaling: str,
) -> np.ndarray:
    """生成一条闭合正方形/矩形轨迹。

    参数:
        theta: 轨迹相位角，单位 rad。函数内部只使用归一化周期位置，因此支持多周期。
        center_x: 方形中心 x 坐标，单位 m。
        center_y: 方形中心 y 坐标，单位 m。
        radius_x: 方形 x 半边长，单位 m。
        radius_y: 方形 y 半边长，单位 m。等于 `radius_x` 时为正方形。
        time_scaling: `quintic` 时，每条边内部用五次多项式从一个顶点插值到下一顶点；
            `linear` 时，每条边保持原始线性插值。

    返回:
        末端 xy 参考，形状 `(N,2)`。路径从右下角出发，依次沿上、左、下、右四条边
        运动；整周期终点回到起点，保证完整闭合。
    """
    phase = (theta / (2.0 * np.pi)) % 1.0
    edge = np.floor(phase * 4.0).astype(np.int32)
    local = phase * 4.0 - edge
    edge = np.minimum(edge, 3)
    if time_scaling == "quintic":
        local = _quintic_smoothstep(local)

    x = np.zeros_like(phase, dtype=np.float64)
    y = np.zeros_like(phase, dtype=np.float64)

    # 第 1 边：右下 -> 右上。
    mask = edge == 0
    x[mask] = center_x + radius_x
    y[mask] = center_y - radius_y + 2.0 * radius_y * local[mask]

    # 第 2 边：右上 -> 左上。
    mask = edge == 1
    x[mask] = center_x + radius_x - 2.0 * radius_x * local[mask]
    y[mask] = center_y + radius_y

    # 第 3 边：左上 -> 左下。
    mask = edge == 2
    x[mask] = center_x - radius_x
    y[mask] = center_y + radius_y - 2.0 * radius_y * local[mask]

    # 第 4 边：左下 -> 右下。
    mask = edge == 3
    x[mask] = center_x - radius_x + 2.0 * radius_x * local[mask]
    y[mask] = center_y - radius_y

    return np.column_stack([x, y]).astype(np.float64)


def generate_cartesian_reference(cfg: CartesianReferenceConfig) -> Dict[str, np.ndarray | dict | str]:
    """生成末端笛卡尔参考轨迹。

    参数:
        cfg: 轨迹配置。所有轨迹参数集中在该 dataclass 中，便于三类控制方法复用。

    返回:
        字典字段：
        - `t`: 时间序列 `(N,)`，单位 s；
        - `xy_ref`: 末端位置参考 `(N,2)`，列为 `[x,y]`，单位 m；
        - `dxy_ref`: 末端速度参考 `(N,2)`，由数值梯度得到，单位 m/s；
        - `kind`: 轨迹类型；
        - `meta`: 参数快照，便于结果复现。
    """
    kind = cfg.kind.lower()
    if kind not in TRAJECTORY_KINDS:
        raise ValueError(f"Unsupported trajectory kind: {cfg.kind}; allowed={TRAJECTORY_KINDS}")
    time_scaling = cfg.time_scaling.lower()
    if time_scaling not in TIME_SCALINGS:
        raise ValueError(f"Unsupported time scaling: {cfg.time_scaling}; allowed={TIME_SCALINGS}")

    t, tau = _time_vector(cfg.dt, cfg.period, cfg.num_cycles, cfg.start_hold)

    if kind == "figure8":
        theta = _path_theta(
            tau=tau,
            period=cfg.period,
            num_cycles=cfg.num_cycles,
            phase=cfg.phase,
            time_scaling=time_scaling,
        )
        x = float(cfg.center_x) + float(cfg.radius_x) * np.sin(theta)
        y = float(cfg.center_y) + float(cfg.radius_y) * np.sin(2.0 * theta)
        xy_ref = np.column_stack([x, y]).astype(np.float64)
    elif kind == "circle":
        theta = _path_theta(
            tau=tau,
            period=cfg.period,
            num_cycles=cfg.num_cycles,
            phase=cfg.phase,
            time_scaling=time_scaling,
        )
        x = float(cfg.center_x) + float(cfg.radius_x) * np.cos(theta)
        y = float(cfg.center_y) + float(cfg.radius_y) * np.sin(theta)
        xy_ref = np.column_stack([x, y]).astype(np.float64)
    elif kind == "square":
        theta = float(cfg.phase) + 2.0 * np.pi * tau / float(cfg.period)
        xy_ref = _square_reference(
            theta=theta,
            center_x=float(cfg.center_x),
            center_y=float(cfg.center_y),
            radius_x=float(cfg.radius_x),
            radius_y=float(cfg.radius_y),
            time_scaling=time_scaling,
        )
    else:  # pragma: no cover - 前面的 allowed 检查已经覆盖
        raise ValueError(kind)

    dxy_ref = np.gradient(xy_ref, float(cfg.dt), axis=0)
    return {
        "t": t,
        "xy_ref": xy_ref,
        "dxy_ref": dxy_ref,
        "kind": kind,
        "meta": asdict(cfg),
    }


def build_parser() -> argparse.ArgumentParser:
    """构造单独生成笛卡尔参考轨迹的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成末端笛卡尔空间参考轨迹 npz，供 IK 和三类控制方法复用。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectory", choices=TRAJECTORY_KINDS, default="figure8", help="轨迹类型。")
    parser.add_argument("--dt", type=float, default=0.01, help="采样周期，单位 s，应与控制周期一致。")
    parser.add_argument("--period", type=float, default=8.0, help="单个轨迹周期，单位 s。")
    parser.add_argument("--num_cycles", type=float, default=1.0, help="轨迹周期数。")
    parser.add_argument("--center_x", type=float, default=5.0, help="轨迹中心 x 坐标，单位 m。")
    parser.add_argument("--center_y", type=float, default=0.0, help="轨迹中心 y 坐标，单位 m。")
    parser.add_argument("--radius_x", type=float, default=0.6, help="x 方向半幅，单位 m。")
    parser.add_argument("--radius_y", type=float, default=0.35, help="y 方向半幅，单位 m。")
    parser.add_argument("--phase", type=float, default=0.0, help="轨迹相位，单位 rad。")
    parser.add_argument("--start_hold", type=float, default=0.0, help="起点保持时间，单位 s。")
    parser.add_argument(
        "--time_scaling",
        choices=TIME_SCALINGS,
        default="quintic",
        help="时间规划方式；quintic 为五次多项式平滑插值，linear 为原始匀速相位/线性分段。",
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "results"
            / "deployment_pipeline"
            / "cartesian_reference"
        ),
        help="参考轨迹 npz 输出根目录。",
    )
    parser.add_argument("--tag", default="", help="输出文件名附加标签。")
    return parser


def main() -> None:
    """命令行入口：只生成笛卡尔参考轨迹，不执行 IK 或控制。"""
    args = build_parser().parse_args()
    cfg = CartesianReferenceConfig(
        kind=args.trajectory,
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
    ref = generate_cartesian_reference(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"{stamp}_{args.trajectory}_cartesian_reference{suffix}.npz"
    np.savez_compressed(out_path, t=ref["t"], xy_ref=ref["xy_ref"], dxy_ref=ref["dxy_ref"])
    print(f"[done] cartesian reference -> {out_path}")


if __name__ == "__main__":
    main()
