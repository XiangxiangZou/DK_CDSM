"""Evaluate one-step and rollout prediction from saved artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments._paths import PROJECT_ROOT

from koopman_control.data.artifacts import save_json
from koopman_control.data.datasets import load_dataset
from koopman_control.evaluation.prediction import evaluate_model
from koopman_control.models.registry import (
    load_prediction_model,
    normalize_model_list,
)
from koopman_control.visualization.plotting import plot_prediction_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate EDMD, DKUC, DKAC, and DKN predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument(
        "--pred_mode",
        choices=["one_step", "rollout", "both"],
        default="both",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--out_dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "results"
            / "deployment_pipeline"
            / "prediction"
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--demo_traj", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--no_plots", action="store_true")
    return parser


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = Path(base_dir) / f"{stamp}_prediction{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    dataset_path = (
        Path(args.dataset)
        if args.dataset
        else artifact_dir / "dataset_val.npz"
    )
    models = normalize_model_list(args.models)
    modes = (
        ["one_step", "rollout"]
        if args.pred_mode == "both"
        else [args.pred_mode]
    )
    output = _make_output_dir(args.out_dir, args.tag)
    dataset = load_dataset(dataset_path)
    rollouts = {"true": np.asarray(dataset["states"], dtype=np.float64)}
    metrics: dict[str, object] = {
        "artifact_dir": str(artifact_dir),
        "dataset": str(dataset_path),
        "models": {},
    }
    model_metrics = metrics["models"]
    assert isinstance(model_metrics, dict)

    for model_name in models:
        print(f"[eval] {model_name}")
        model = load_prediction_model(
            artifact_dir,
            model_name,
            args.device,
        )
        model_metrics[model_name] = {}
        for mode in modes:
            result = evaluate_model(model, dataset, mode)
            rollouts[f"{model_name}_{mode}_pred"] = result["preds"]
            model_metrics[model_name][mode] = result["metrics"]
            print(
                f"  {mode}: "
                f"total_rmse={result['metrics']['total_rmse']:.6g}"
            )

    np.savez_compressed(output / "prediction_rollouts.npz", **rollouts)
    if "one_step" in modes:
        save_json(
            output / "one_step_metrics.json",
            {
                name: values["one_step"]
                for name, values in model_metrics.items()
            },
        )
    if "rollout" in modes:
        save_json(
            output / "rollout_metrics.json",
            {
                name: values["rollout"]
                for name, values in model_metrics.items()
            },
        )
    if not args.no_plots:
        metrics["figures"] = plot_prediction_figures(
            out_dir=output,
            rollouts=rollouts,
            metrics=metrics,
            models=models,
            modes=modes,
            dt=args.dt,
            demo_traj=args.demo_traj,
        )
    save_json(output / "prediction_metrics.json", metrics)
    print(f"[done] prediction results -> {output}")


if __name__ == "__main__":
    main()
