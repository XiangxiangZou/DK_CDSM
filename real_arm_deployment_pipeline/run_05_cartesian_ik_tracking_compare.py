"""第五部分：末端笛卡尔轨迹经 MuJoCo IK 后的关节空间跟踪对比。

流程：
1. 用 `cartesian_reference.py` 生成末端笛卡尔参考轨迹 `xy_ref`。
2. 用 `mujoco_ik.py` 根据 MuJoCo `end_effector` site 反解为 `q_ref,dq_ref`。
3. 把同一条关节参考分别交给 EDMD、DKUC、DKAC 的 Koopman LQR 控制器。
4. MuJoCo 作为真实机械臂逐周期反馈真实 `q,dq`，执行绳张力控制。
5. 保存关节空间和末端笛卡尔空间两类指标与结果图。

注意：
DKN 当前仍不进入本脚本，因为它还没有统一线性 Koopman LQR 控制接口。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from .cartesian_reference import (
        TIME_SCALINGS,
        CartesianReferenceConfig,
        TRAJECTORY_KINDS,
        generate_cartesian_reference,
    )
    from .model_registry import load_control_model, normalize_model_list
    from .mujoco_ik import IKConfig, MujocoSiteIK
    from .plotting import plot_cartesian_tracking_figures, plot_tracking_figures
    from .tracking_controller import LqrConfig
    from .tracking_eval import cartesian_tracking_metrics, logs_to_npz_payload, tracking_metrics
    from .tracking_runtime import run_joint_space_closed_loop_model
except ImportError:  # pragma: no cover - 支持直接运行本文件
    from cartesian_reference import TIME_SCALINGS, CartesianReferenceConfig, TRAJECTORY_KINDS, generate_cartesian_reference
    from model_registry import load_control_model, normalize_model_list
    from mujoco_ik import IKConfig, MujocoSiteIK
    from plotting import plot_cartesian_tracking_figures, plot_tracking_figures
    from tracking_controller import LqrConfig
    from tracking_eval import cartesian_tracking_metrics, logs_to_npz_payload, tracking_metrics
    from tracking_runtime import run_joint_space_closed_loop_model


def build_parser() -> argparse.ArgumentParser:
    """构造末端笛卡尔轨迹跟踪对比命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成末端笛卡尔轨迹，经 MuJoCo IK 转为关节参考，并比较 EDMD/DKUC/DKAC 闭环跟踪。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True, help="run_02 输出的 artifacts 根目录。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="要比较的控制模型，支持 all 或 edmd dkuc dkac 任意组合；DKN 暂不支持本线性 LQR。",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="神经模型推理设备。")
    parser.add_argument(
        "--xml",
        default=str(Path(__file__).resolve().parents[1] / "multi_joint_cable_dirven_space_robot.xml"),
        help="MuJoCo XML 路径；真实机械臂接入后由 real_arm_plant 配置替代。",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="控制周期/仿真步长，单位 s。")
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "results" / "cartesian_tracking"),
        help="末端笛卡尔轨迹跟踪结果输出根目录。",
    )
    parser.add_argument("--tag", default="", help="输出目录附加标签。")
    parser.add_argument("--no_plots", action="store_true", help="只保存 npz/json，不绘制 PNG 图。")

    traj_group = parser.add_argument_group("末端笛卡尔参考轨迹参数")
    traj_group.add_argument(
        "--trajectories",
        nargs="+",
        choices=TRAJECTORY_KINDS,
        default=["figure8", "circle"],
        help="要依次运行的末端轨迹类型。",
    )
    traj_group.add_argument("--period", type=float, default=8.0, help="单个末端轨迹周期，单位 s。")
    traj_group.add_argument("--num_cycles", type=float, default=1.0, help="每类轨迹运行的周期数。")
    traj_group.add_argument("--center_x", type=float, default=5.0, help="末端轨迹中心 x 坐标，单位 m。")
    traj_group.add_argument("--center_y", type=float, default=0.0, help="末端轨迹中心 y 坐标，单位 m。")
    traj_group.add_argument("--radius_x", type=float, default=0.6, help="末端轨迹 x 方向半幅，单位 m。")
    traj_group.add_argument("--radius_y", type=float, default=0.35, help="末端轨迹 y 方向半幅，单位 m。")
    traj_group.add_argument("--phase", type=float, default=0.0, help="末端轨迹相位，单位 rad。")
    traj_group.add_argument("--start_hold", type=float, default=0.0, help="起点保持时间，单位 s。")
    traj_group.add_argument(
        "--time_scaling",
        choices=TIME_SCALINGS,
        default="quintic",
        help="末端轨迹时间规划方式；quintic 为五次多项式平滑插值，linear 为原始匀速/线性轨迹。",
    )

    ik_group = parser.add_argument_group("MuJoCo 逆运动学参数")
    ik_group.add_argument("--ik_site", default="end_effector", help="用于 IK 的 MuJoCo site 名称。")
    ik_group.add_argument("--ik_max_iter", type=int, default=120, help="单个末端目标点 IK 最大迭代次数。")
    ik_group.add_argument("--ik_tol", type=float, default=1e-5, help="IK 末端位置收敛阈值，单位 m。")
    ik_group.add_argument("--ik_damping", type=float, default=1e-4, help="阻尼最小二乘 IK 阻尼系数。")
    ik_group.add_argument("--ik_max_step", type=float, default=0.08, help="IK 单次迭代最大关节增量，单位 rad。")
    ik_group.add_argument("--ik_joint_margin", type=float, default=0.05, help="主动关节限位内缩量，单位 rad。")
    ik_group.add_argument("--ik_smooth_window_s", type=float, default=0.03, help="IK 关节参考移动平均窗口，单位 s。")
    ik_group.add_argument("--ik_seed_a", type=float, default=0.1, help="第一帧 IK 的 qa 初值，单位 rad。")
    ik_group.add_argument("--ik_seed_b", type=float, default=-0.1, help="第一帧 IK 的 qb 初值，单位 rad。")

    lqr_group = parser.add_argument_group("Koopman LQR/MPC 参数")
    lqr_group.add_argument("--horizon", type=int, default=30, help="预测控制时域步数。")
    lqr_group.add_argument("--Qq", type=float, default=40.0, help="关节角误差权重。")
    lqr_group.add_argument("--Qdq", type=float, default=2.0, help="关节角速度误差权重。")
    lqr_group.add_argument("--R", type=float, default=1e-3, help="控制量幅值权重。")
    lqr_group.add_argument("--Rd", type=float, default=1e-2, help="控制增量权重。")
    lqr_group.add_argument("--tau_limit", type=float, default=120.0, help="执行前关节力矩限幅，单位 Nm。")
    return parser


def _make_root_output_dir(base_dir: str | Path, tag: str) -> Path:
    """创建本次末端轨迹跟踪实验的根输出目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = Path(base_dir) / f"{stamp}_cartesian_ik_tracking{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _save_json(path: Path, payload: dict) -> None:
    """保存 JSON，保留中文字段和注释可读性。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_reference(
    *,
    trajectory: str,
    args: argparse.Namespace,
    ik_solver: MujocoSiteIK,
) -> Dict[str, np.ndarray]:
    """生成某一类末端轨迹，并用 MuJoCo IK 转换为关节参考。"""
    cart_cfg = CartesianReferenceConfig(
        kind=trajectory,
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
    cart_ref = generate_cartesian_reference(cart_cfg)
    ik_ref = ik_solver.solve_trajectory(np.asarray(cart_ref["xy_ref"], dtype=np.float64))
    return {
        "t": np.asarray(cart_ref["t"], dtype=np.float64),
        "q_ref": np.asarray(ik_ref["q_ref"], dtype=np.float64),
        "dq_ref": np.asarray(ik_ref["dq_ref"], dtype=np.float64),
        "ee_ref": np.asarray(cart_ref["xy_ref"], dtype=np.float64),
        "dxy_ref": np.asarray(cart_ref["dxy_ref"], dtype=np.float64),
        "ee_ik": np.asarray(ik_ref["ee_ik"], dtype=np.float64),
        "ik_error": np.asarray(ik_ref["ik_error"], dtype=np.float64),
        "ik_converged": np.asarray(ik_ref["ik_converged"], dtype=bool),
        "ik_iterations": np.asarray(ik_ref["ik_iterations"], dtype=np.int32),
    }


def _save_reference_npz(out_dir: Path, ref: Dict[str, np.ndarray]) -> None:
    """保存笛卡尔参考和 IK 后关节参考。"""
    np.savez_compressed(
        out_dir / "cartesian_ik_reference.npz",
        t=ref["t"],
        ee_ref=ref["ee_ref"],
        dxy_ref=ref["dxy_ref"],
        q_ref=ref["q_ref"],
        dq_ref=ref["dq_ref"],
        ee_ik=ref["ee_ik"],
        ik_error=ref["ik_error"],
        ik_converged=ref["ik_converged"],
        ik_iterations=ref["ik_iterations"],
    )


def _ik_summary(ref: Dict[str, np.ndarray]) -> Dict[str, object]:
    """汇总 IK 质量，便于判断笛卡尔轨迹是否适合当前机械臂。"""
    ik_error = np.asarray(ref["ik_error"], dtype=np.float64)
    converged = np.asarray(ref["ik_converged"], dtype=bool)
    q_ref = np.asarray(ref["q_ref"], dtype=np.float64)
    return {
        "points": int(converged.shape[0]),
        "not_converged": int(np.sum(~converged)),
        "rmse_ik": float(np.sqrt(np.mean(ik_error * ik_error))),
        "max_abs_ik_error": float(np.max(np.abs(ik_error))),
        "qa_min": float(np.min(q_ref[:, 0])),
        "qa_max": float(np.max(q_ref[:, 0])),
        "qb_min": float(np.min(q_ref[:, 1])),
        "qb_max": float(np.max(q_ref[:, 1])),
    }


def _run_one_trajectory(
    *,
    trajectory: str,
    root_dir: Path,
    args: argparse.Namespace,
    models: List[str],
    lqr_cfg: LqrConfig,
    ik_cfg: IKConfig,
) -> Dict[str, object]:
    """运行单一末端轨迹的 IK 和三类模型闭环跟踪。"""
    traj_dir = root_dir / trajectory
    traj_dir.mkdir(parents=True, exist_ok=False)
    ik_solver = MujocoSiteIK(args.xml, args.dt, ik_cfg)
    ref = _build_reference(trajectory=trajectory, args=args, ik_solver=ik_solver)
    _save_reference_npz(traj_dir, ref)

    metrics: Dict[str, object] = {
        "artifact_dir": str(Path(args.artifact_dir)),
        "trajectory": trajectory,
        "cartesian_reference": {
            "period": args.period,
            "num_cycles": args.num_cycles,
            "center_x": args.center_x,
            "center_y": args.center_y,
            "radius_x": args.radius_x,
            "radius_y": args.radius_y,
            "phase": args.phase,
            "start_hold": args.start_hold,
            "time_scaling": args.time_scaling,
            "dt": args.dt,
        },
        "ik_config": asdict(ik_cfg),
        "ik_summary": _ik_summary(ref),
        "lqr_config": asdict(lqr_cfg),
        "models": {},
        "note": "DKN is excluded because it needs a separate nonlinear MPC/local-linearization interface.",
    }

    print(f"[trajectory] {trajectory}")
    print(
        "  IK: "
        f"not_converged={metrics['ik_summary']['not_converged']}/"
        f"{metrics['ik_summary']['points']}, "
        f"rmse_ik={metrics['ik_summary']['rmse_ik']:.6g} m"
    )

    logs: Dict[str, Dict[str, np.ndarray]] = {}
    ee_logs: Dict[str, np.ndarray] = {}
    artifact_dir = Path(args.artifact_dir)
    for model_name in models:
        print(f"  [tracking] {model_name}")
        model = load_control_model(artifact_dir, model_name, args.device)
        log = run_joint_space_closed_loop_model(
            model=model,
            xml=args.xml,
            dt=args.dt,
            ref=ref,
            lqr_cfg=lqr_cfg,
            tau_limit=args.tau_limit,
        )
        n = len(log["t"])
        ee_meas = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
        log["ee_meas"] = ee_meas
        log["ee_ref"] = ref["ee_ref"][:n]
        log["ee_ik"] = ref["ee_ik"][:n]
        logs[model_name] = log
        ee_logs[model_name] = ee_meas

        model_metrics = tracking_metrics(log)
        model_metrics["cartesian"] = cartesian_tracking_metrics(
            ee_meas=ee_meas,
            ee_ref=ref["ee_ref"][:n],
            ik_error=ref["ik_error"][:n],
        )
        metrics["models"][model_name] = model_metrics
        np.savez_compressed(traj_dir / f"closed_loop_{model_name}.npz", **log)
        print(
            f"    rmse_q={model_metrics['rmse_q']:.6g}, "
            f"rmse_ee={model_metrics['cartesian']['rmse_ee']:.6g} m"
        )

    np.savez_compressed(traj_dir / "closed_loop_all_models.npz", **logs_to_npz_payload(logs))
    if not args.no_plots:
        figures = plot_tracking_figures(out_dir=traj_dir, logs=logs, metrics=metrics)
        figures += plot_cartesian_tracking_figures(
            out_dir=traj_dir,
            logs=logs,
            ee_logs=ee_logs,
            ref=ref,
            metrics=metrics,
        )
        metrics["figures"] = figures
        print(f"  [plots] saved {len(figures)} figures")

    _save_json(traj_dir / "cartesian_tracking_metrics.json", metrics)
    return metrics


def main() -> None:
    """运行所有指定末端轨迹的 IK + 闭环跟踪对比。"""
    args = build_parser().parse_args()
    models = normalize_model_list(args.models, control_only=True)
    lqr_cfg = LqrConfig(args.horizon, args.Qq, args.Qdq, args.R, args.Rd)
    ik_cfg = IKConfig(
        site_name=args.ik_site,
        max_iter=args.ik_max_iter,
        tol=args.ik_tol,
        damping=args.ik_damping,
        max_step=args.ik_max_step,
        joint_margin=args.ik_joint_margin,
        smooth_window_s=args.ik_smooth_window_s,
        q_seed_a=args.ik_seed_a,
        q_seed_b=args.ik_seed_b,
    )
    root_dir = _make_root_output_dir(args.out_dir, args.tag)

    print("=== CDSM Cartesian IK tracking comparison ===")
    print(f"artifact_dir={Path(args.artifact_dir)}")
    print(f"models={models}")
    print(f"trajectories={args.trajectories}")
    print(f"output={root_dir}")

    summary: Dict[str, object] = {
        "artifact_dir": str(Path(args.artifact_dir)),
        "models": models,
        "trajectories": {},
        "output_dir": str(root_dir),
    }
    for trajectory in args.trajectories:
        summary["trajectories"][trajectory] = _run_one_trajectory(
            trajectory=trajectory,
            root_dir=root_dir,
            args=args,
            models=models,
            lqr_cfg=lqr_cfg,
            ik_cfg=ik_cfg,
        )
    _save_json(root_dir / "cartesian_tracking_summary.json", summary)
    print(f"[done] Cartesian IK tracking results -> {root_dir}")


if __name__ == "__main__":
    main()
