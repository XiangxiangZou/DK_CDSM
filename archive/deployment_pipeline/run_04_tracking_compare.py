"""第四部分：在线/闭环跟踪控制对比。

当前实现把 MuJoCo 当作真实机械臂：
    每个控制周期读取真实 `q,dq` -> 模型升维 `z` -> Koopman LQR 求控制 ->
    关节力矩映射为 8 根绳张力 -> MuJoCo 执行一步 -> 下一周期重新反馈。

参与统一线性 Koopman LQR 的模型：
    EDMD、DKUC、DKAC。

DKN 说明：
    DKN 当前不进入本脚本，因为它不是 `z_next=A z+B u` 的直接线性控制形式。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from .model_registry import load_control_model, normalize_model_list
    from .plotting import plot_tracking_figures
    from .tracking_controller import LqrConfig, build_ramp_reference
    from .tracking_eval import logs_to_npz_payload, tracking_metrics
    from .tracking_runtime import run_joint_space_closed_loop_model
except ImportError:  # pragma: no cover
    from model_registry import load_control_model, normalize_model_list
    from plotting import plot_tracking_figures
    from tracking_controller import LqrConfig, build_ramp_reference
    from tracking_eval import logs_to_npz_payload, tracking_metrics
    from tracking_runtime import run_joint_space_closed_loop_model


def build_parser() -> argparse.ArgumentParser:
    """构造闭环跟踪对比命令行参数。"""
    parser = argparse.ArgumentParser(
        description="在 MuJoCo/真实机械臂接口上比较 EDMD、DKUC、DKAC 的 Koopman LQR 跟踪控制。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True, help="run_02 输出的 artifacts 根目录。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="要比较的控制模型，支持 all 或 edmd dkuc dkac 任意组合；DKN 不支持本线性 LQR。",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="神经模型推理设备。")
    parser.add_argument(
        "--xml",
        default=str(
            Path(__file__).resolve().parents[2]
            / "assets"
            / "models"
            / "multi_joint_cable_driven_space_robot.xml"
        ),
        help="MuJoCo XML 路径；实物机械臂接入后由 real_arm_plant 配置替代。",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="控制周期/仿真步长，单位 s。")
    parser.add_argument(
        "--out_dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "results"
            / "deployment_pipeline"
            / "tracking"
        ),
        help="闭环跟踪结果输出根目录。",
    )
    parser.add_argument("--tag", default="", help="输出目录附加标签。")

    ref_group = parser.add_argument_group("参考轨迹参数")
    ref_group.add_argument("--T_track", type=float, default=4.0, help="闭环跟踪总时长，单位 s。")
    ref_group.add_argument("--T_ramp", type=float, default=2.0, help="参考轨迹从初值过渡到目标值的时长，单位 s。")
    ref_group.add_argument("--qa0", type=float, default=0.0, help="qa 初始参考角，单位 rad。")
    ref_group.add_argument("--qa1", type=float, default=0.6, help="qa 目标参考角，单位 rad。")
    ref_group.add_argument("--qb0", type=float, default=0.0, help="qb 初始参考角，单位 rad。")
    ref_group.add_argument("--qb1", type=float, default=-0.45, help="qb 目标参考角，单位 rad。")

    lqr_group = parser.add_argument_group("Koopman LQR/MPC 参数")
    lqr_group.add_argument("--horizon", type=int, default=30, help="预测控制时域步数。")
    lqr_group.add_argument("--Qq", type=float, default=40.0, help="关节角误差权重。")
    lqr_group.add_argument("--Qdq", type=float, default=2.0, help="关节角速度误差权重。")
    lqr_group.add_argument("--R", type=float, default=1e-3, help="控制量幅值权重。")
    lqr_group.add_argument("--Rd", type=float, default=1e-2, help="控制增量权重。")
    lqr_group.add_argument("--tau_limit", type=float, default=120.0, help="执行前关节力矩限幅，单位 Nm。")
    parser.add_argument("--no_plots", action="store_true", help="只保存 npz/json，不绘制 PNG 图。")
    return parser


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    """创建本次跟踪对比输出目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = Path(base_dir) / f"{stamp}_tracking{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _save_json(path: Path, payload: dict) -> None:
    """保存 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_closed_loop_model(**kwargs) -> Dict[str, np.ndarray]:
    """兼容旧调用名；实际实现已拆到 `tracking_runtime.py`。"""
    return run_joint_space_closed_loop_model(**kwargs)


def main() -> None:
    """运行 EDMD/DKUC/DKAC 闭环跟踪对比并保存日志。"""
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    models = normalize_model_list(args.models, control_only=True)
    out_dir = _make_output_dir(args.out_dir, args.tag)
    lqr_cfg = LqrConfig(args.horizon, args.Qq, args.Qdq, args.R, args.Rd)
    ref = build_ramp_reference(
        dt=args.dt,
        T_total=args.T_track,
        qa0=args.qa0,
        qa1=args.qa1,
        qb0=args.qb0,
        qb1=args.qb1,
        T_ramp=args.T_ramp,
    )

    print("=== CDSM tracking comparison ===")
    print(f"artifact_dir={artifact_dir}")
    print(f"models={models}")
    print(f"output={out_dir}")

    logs: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, object] = {
        "artifact_dir": str(artifact_dir),
        "models": {},
        "lqr_config": asdict(lqr_cfg),
        "reference": {
            "T_track": args.T_track,
            "T_ramp": args.T_ramp,
            "qa0": args.qa0,
            "qa1": args.qa1,
            "qb0": args.qb0,
            "qb1": args.qb1,
        },
        "note": "DKN is excluded because it needs a separate nonlinear MPC/local-linearization interface.",
    }

    for model_name in models:
        print(f"[tracking] {model_name}")
        model = load_control_model(artifact_dir, model_name, args.device)
        log = run_joint_space_closed_loop_model(
            model=model,
            xml=args.xml,
            dt=args.dt,
            ref=ref,
            lqr_cfg=lqr_cfg,
            tau_limit=args.tau_limit,
        )
        logs[model_name] = log
        metrics["models"][model_name] = tracking_metrics(log)
        np.savez_compressed(out_dir / f"closed_loop_{model_name}.npz", **log)
        print(f"  rmse_q={metrics['models'][model_name]['rmse_q']:.6g}")

    np.savez_compressed(out_dir / "closed_loop_all_models.npz", **logs_to_npz_payload(logs))
    if not args.no_plots:
        figures = plot_tracking_figures(out_dir=out_dir, logs=logs, metrics=metrics)
        metrics["figures"] = figures
        print(f"[plots] saved {len(figures)} tracking figures")
    _save_json(out_dir / "tracking_metrics.json", metrics)
    print(f"[done] tracking results -> {out_dir}")


if __name__ == "__main__":
    main()
