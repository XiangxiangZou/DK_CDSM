"""第三部分：离线模型预测验证。

用途：
    读取 `run_02_train_all_models.py` 生成的 artifacts，使用验证数据集对
    EDMD、DKUC、DKAC、DKN 做 one-step 和 rollout 预测评估。

输出：
    - `prediction_rollouts.npz`: 各模型预测轨迹和真实轨迹。
    - `prediction_metrics.json`: 所有模型和模式的统一误差指标。
    - `one_step_metrics.json`: one-step 指标子集。
    - `rollout_metrics.json`: rollout 指标子集。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .datasets import load_dataset
    from .model_registry import load_prediction_model, normalize_model_list
    from .plotting import plot_prediction_figures
    from .prediction_eval import evaluate_model
except ImportError:  # pragma: no cover
    from datasets import load_dataset
    from model_registry import load_prediction_model, normalize_model_list
    from plotting import plot_prediction_figures
    from prediction_eval import evaluate_model


def build_parser() -> argparse.ArgumentParser:
    """构造预测验证命令行参数。"""
    parser = argparse.ArgumentParser(
        description="离线验证 EDMD、DKUC、DKAC、DKN 的 one-step 和 rollout 预测误差。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True, help="run_02 输出的 artifacts 根目录。")
    parser.add_argument(
        "--dataset",
        default="",
        help="验证用 dataset.npz；为空时默认读取 artifact_dir/dataset_val.npz。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="要评估的模型，支持 all 或 edmd dkuc dkac dkn 任意组合。",
    )
    parser.add_argument(
        "--pred_mode",
        choices=["one_step", "rollout", "both"],
        default="both",
        help="预测评估模式。one_step 看局部一步误差；rollout 看开环长程误差。",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="神经模型推理设备。")
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "results" / "prediction"),
        help="预测评估结果输出根目录。",
    )
    parser.add_argument("--tag", default="", help="输出目录附加标签。")
    parser.add_argument("--demo_traj", type=int, default=0, help="动态响应图中展示的验证轨迹编号。")
    parser.add_argument("--dt", type=float, default=0.01, help="数据采样周期，单位 s，仅用于绘图横轴。")
    parser.add_argument("--no_plots", action="store_true", help="只保存 npz/json，不绘制 PNG 图。")
    return parser


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    """创建本次预测评估输出目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = Path(base_dir) / f"{stamp}_prediction{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _save_json(path: Path, payload: dict) -> None:
    """保存 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    """执行统一预测评估。"""
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    dataset_path = Path(args.dataset) if args.dataset else artifact_dir / "dataset_val.npz"
    models = normalize_model_list(args.models, control_only=False)
    modes = ["one_step", "rollout"] if args.pred_mode == "both" else [args.pred_mode]
    out_dir = _make_output_dir(args.out_dir, args.tag)

    print("=== CDSM prediction validation ===")
    print(f"artifact_dir={artifact_dir}")
    print(f"dataset={dataset_path}")
    print(f"models={models}, modes={modes}")
    print(f"output={out_dir}")

    dataset = load_dataset(dataset_path)
    rollouts = {"true": np.asarray(dataset["states"], dtype=np.float64)}
    metrics = {"artifact_dir": str(artifact_dir), "dataset": str(dataset_path), "models": {}}

    for model_name in models:
        print(f"[eval] {model_name}")
        model = load_prediction_model(artifact_dir, model_name, args.device)
        metrics["models"][model_name] = {}
        for mode in modes:
            result = evaluate_model(model, dataset, mode)
            rollouts[f"{model_name}_{mode}_pred"] = result["preds"]
            metrics["models"][model_name][mode] = result["metrics"]
            print(f"  {mode}: total_rmse={result['metrics']['total_rmse']:.6g}")

    np.savez_compressed(out_dir / "prediction_rollouts.npz", **rollouts)
    _save_json(out_dir / "prediction_metrics.json", metrics)

    if "one_step" in modes:
        _save_json(
            out_dir / "one_step_metrics.json",
            {name: item["one_step"] for name, item in metrics["models"].items() if "one_step" in item},
        )
    if "rollout" in modes:
        _save_json(
            out_dir / "rollout_metrics.json",
            {name: item["rollout"] for name, item in metrics["models"].items() if "rollout" in item},
        )
    if not args.no_plots:
        figures = plot_prediction_figures(
            out_dir=out_dir,
            rollouts=rollouts,
            metrics=metrics,
            models=models,
            modes=modes,
            dt=args.dt,
            demo_traj=args.demo_traj,
        )
        metrics["figures"] = figures
        _save_json(out_dir / "prediction_metrics.json", metrics)
        print(f"[plots] saved {len(figures)} prediction figures")
    print(f"[done] prediction results -> {out_dir}")


if __name__ == "__main__":
    main()
