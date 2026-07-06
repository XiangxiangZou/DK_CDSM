"""Train EDMD, DKUC, DKAC, and DKN artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from experiments._paths import PROJECT_ROOT

from koopman_control.data.artifacts import save_json
from koopman_control.data.datasets import load_dataset, split_train_val
from koopman_control.training.edmd import EDMDTrainingConfig
from koopman_control.training.neural import NeuralTrainingConfig
from koopman_control.training.pipeline import (
    MODEL_ORDER,
    train_selected_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train EDMD, DKUC, DKAC, and DKN from a common dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--out_dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "deployment_pipeline"
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--lift_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    parser.add_argument("--control_hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--control_dim_hat", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    parser.add_argument("--bound_lift", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--window_start", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps_per_epoch", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--w_state", type=float, default=1.0)
    parser.add_argument("--w_embed", type=float, default=0.1)
    parser.add_argument("--edmd_centers", type=int, default=200)
    parser.add_argument("--edmd_sigma", type=float, default=None)
    parser.add_argument("--edmd_ridge", type=float, default=1e-4)
    parser.add_argument("--edmd_seed", type=int, default=2007)
    return parser


def _selected_models(raw_models: list[str]) -> list[str]:
    values = [item.lower() for item in raw_models]
    if "all" in values:
        return list(MODEL_ORDER)
    unknown = sorted(set(values) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unsupported models {unknown}; allowed={MODEL_ORDER}")
    return values


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output = Path(base_dir) / f"{stamp}_train_models{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def main() -> None:
    args = build_parser().parse_args()
    models = _selected_models(args.models)
    output = _make_output_dir(args.out_dir, args.tag)
    source = load_dataset(args.train_dataset)
    if args.val_dataset:
        train_data = source
        val_data = load_dataset(args.val_dataset)
        split_metadata = {
            "mode": "explicit_train_val",
            "train_dataset": args.train_dataset,
            "val_dataset": args.val_dataset,
        }
    else:
        train_data, val_data, split_metadata = split_train_val(
            source,
            args.val_ratio,
            args.seed,
        )

    neural_config = NeuralTrainingConfig(
        lift_dim=args.lift_dim,
        hidden=tuple(args.hidden),
        control_hidden=tuple(args.control_hidden),
        control_dim_hat=args.control_dim_hat,
        activation=args.activation,
        bound_lift=args.bound_lift,
        window=args.window,
        window_start=args.window_start,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        w_state=args.w_state,
        w_embed=args.w_embed,
    )
    edmd_config = EDMDTrainingConfig(
        n_centers=args.edmd_centers,
        rbf_sigma=args.edmd_sigma,
        ridge=args.edmd_ridge,
        kmeans_seed=args.edmd_seed,
    )
    save_json(
        output / "run_config.json",
        {
            "train_dataset": args.train_dataset,
            "val_dataset": args.val_dataset,
            "split": split_metadata,
            "models": models,
            "neural_config": asdict(neural_config),
            "edmd_config": asdict(edmd_config),
        },
    )
    summary = train_selected_models(
        train_data=train_data,
        val_data=val_data,
        models=models,
        neural_config=neural_config,
        edmd_config=edmd_config,
        device_name=args.device,
        seed=args.seed,
        output_dir=output,
    )
    print(f"[done] artifacts -> {output}")
    print(f"[done] trained models -> {list(summary['models'].keys())}")


if __name__ == "__main__":
    main()
