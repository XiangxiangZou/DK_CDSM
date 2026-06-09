"""MoJoCo 离线数据采集入口。

用途：
    为 DKUC、DKAC、EDMD、DKN 的模型预测/跟踪控制对比采集统一数据集。

两种采集模式：
    1. `--random`：在关节安全范围内施加随机关节力矩激励。
    2. `--PDCtrl`：在关节安全范围内用 PD 控制器跟踪随机多正弦参考。

输出：
    每次运行会创建一个时间戳目录，包含：
    - `dataset.npz`: states, inputs, q_ref, dq_ref, cable_ctrl。
    - `meta.json`: 采集模式、参数、关节安全范围、变量顺序。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .data_collection import CollectionConfig, collect_pd_control, collect_random_excitation
    from .mujoco_plant import MujocoCablePlant
except ImportError:  # pragma: no cover - direct script execution fallback
    from data_collection import CollectionConfig, collect_pd_control, collect_random_excitation
    from mujoco_plant import MujocoCablePlant


def _repo_root() -> Path:
    """返回项目根目录，用于构造默认 XML 路径。"""
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    参数设计原则：
    - `--random` 和 `--PDCtrl` 互斥，必须二选一。
    - 通用参数控制数据规模、采样周期和安全边界。
    - random 参数只影响随机激励模式。
    - PDCtrl 参数只影响 PD 参考轨迹跟踪模式。
    """
    parser = argparse.ArgumentParser(
        description="采集 CDSM MuJoCo 数据集，可选择随机激励或 PD 控制两种模式。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 采集模式：必须二选一，避免同一次运行混合两种数据协议。
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--random",
        action="store_true",
        help="使用关节空间安全范围内的分段随机力矩激励采集数据。",
    )
    mode.add_argument(
        "--PDCtrl",
        "--pdctrl",
        dest="PDCtrl",
        action="store_true",
        help="使用 PD 控制器跟踪关节空间安全范围内的随机参考轨迹采集数据。",
    )

    common = parser.add_argument_group("通用采集参数")
    common.add_argument(
        "--xml",
        default=str(_repo_root() / "multi_joint_cable_dirven_space_robot.xml"),
        help="MuJoCo 绳驱机械臂 XML 路径；后续接真实机械臂时该参数将由 real_arm_plant 替代。",
    )
    common.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "data" / "raw"),
        help="采集数据输出根目录；程序会在其下创建时间戳子目录。",
    )
    common.add_argument("--tag", default="", help="输出目录附加标签，便于区分实验批次。")
    common.add_argument("--traj", type=int, default=120, help="采集轨迹条数。")
    common.add_argument("--steps", type=int, default=300, help="每条轨迹的控制步数。")
    common.add_argument("--dt", type=float, default=0.01, help="控制周期/仿真步长，单位 s。")
    common.add_argument("--seed", type=int, default=42, help="随机种子，用于复现实验数据。")
    common.add_argument(
        "--q_limit_ratio",
        type=float,
        default=0.90,
        help="使用 XML 关节限位的比例；0.90 表示只用中心 90%% 角度范围。",
    )
    common.add_argument(
        "--q_init_ratio",
        type=float,
        default=0.65,
        help="初始关节角采样范围占安全关节范围的比例。",
    )
    common.add_argument(
        "--dq_init_range",
        type=float,
        default=0.4,
        help="初始角速度采样范围为 [-值, 值]，单位 rad/s。",
    )
    common.add_argument(
        "--tau_max",
        type=float,
        default=80.0,
        help="最终下发前对关节力矩进行裁剪的最大绝对值，单位 Nm。",
    )
    common.add_argument(
        "--boundary_kp",
        type=float,
        default=80.0,
        help="接近关节安全边界时的回拉比例增益，两种模式均使用。",
    )
    common.add_argument(
        "--boundary_kd",
        type=float,
        default=6.0,
        help="接近关节安全边界时的回拉阻尼增益，两种模式均使用。",
    )

    random_group = parser.add_argument_group("random 模式参数")
    random_group.add_argument(
        "--random_tau",
        type=float,
        default=35.0,
        help="随机激励力矩幅值上限，采样范围为 [-值, 值]，单位 Nm。",
    )
    random_group.add_argument(
        "--random_hold_steps",
        type=int,
        default=8,
        help="随机力矩保持步数；数值越大，激励越平滑但频率覆盖越低。",
    )
    random_group.add_argument(
        "--random_damping",
        type=float,
        default=0.8,
        help="随机模式附加速度阻尼系数，力矩中加入 -random_damping*dq。",
    )

    pd_group = parser.add_argument_group("PDCtrl 模式参数")
    pd_group.add_argument("--amp_min", type=float, default=0.15, help="随机参考轨迹最小幅值，单位 rad。")
    pd_group.add_argument("--amp_max", type=float, default=0.55, help="随机参考轨迹最大幅值，单位 rad。")
    pd_group.add_argument("--omega_min", type=float, default=0.7, help="随机参考轨迹最小角频率，单位 rad/s。")
    pd_group.add_argument("--omega_max", type=float, default=2.3, help="随机参考轨迹最大角频率，单位 rad/s。")
    pd_group.add_argument("--kp_a", type=float, default=80.0, help="qa 关节 PD 比例增益。")
    pd_group.add_argument("--kp_b", type=float, default=70.0, help="qb 关节 PD 比例增益。")
    pd_group.add_argument("--kd_a", type=float, default=8.0, help="qa 关节 PD 微分增益。")
    pd_group.add_argument("--kd_b", type=float, default=7.0, help="qb 关节 PD 微分增益。")
    return parser


def _make_config(args: argparse.Namespace) -> CollectionConfig:
    """把命令行参数转换为采集配置对象。"""
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
    )


def _output_dir(base_dir: str | Path, mode: str, tag: str) -> Path:
    """创建本次采集的时间戳输出目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = Path(base_dir) / f"{stamp}_{mode}{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def main() -> None:
    """执行一次完整采集并保存 `dataset.npz` 与 `meta.json`。"""
    args = build_parser().parse_args()
    mode = "random" if args.random else "PDCtrl"
    cfg = _make_config(args)

    out_dir = _output_dir(args.out_dir, mode, args.tag)
    plant = MujocoCablePlant(args.xml, cfg.dt)

    print("=== CDSM MuJoCo data collection ===")
    print(f"mode={mode}, output={out_dir}")
    print(f"xml={args.xml}")
    print(f"traj={cfg.traj_count}, steps={cfg.steps}, dt={cfg.dt}")

    if args.random:
        arrays, meta = collect_random_excitation(plant, cfg)
    else:
        arrays, meta = collect_pd_control(plant, cfg)

    # 压缩保存统一数据结构，后续四类模型训练和预测评估都读取该文件。
    dataset_path = out_dir / "dataset.npz"
    np.savez_compressed(dataset_path, **arrays)

    payload = {
        **meta,
        "xml": str(Path(args.xml).resolve()),
        "output_dir": str(out_dir.resolve()),
        "dataset_file": str(dataset_path.name),
        "collection_config": asdict(cfg),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"states={arrays['states'].shape}, inputs={arrays['inputs'].shape}")
    print(f"[done] dataset -> {dataset_path}")


if __name__ == "__main__":
    main()
