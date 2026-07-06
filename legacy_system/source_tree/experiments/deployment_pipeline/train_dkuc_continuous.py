"""Train Yu-Tan-compatible continuous-time DKUC artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments._paths import PROJECT_ROOT

from koopman_control.data.artifacts import save_json, save_normalizers
from koopman_control.data.datasets import load_dataset, save_dataset, split_train_val
from koopman_control.data.normalization import Normalizer
from koopman_control.training.continuous_dkuc import (
    ContinuousDKUCTrainingConfig,
    fit_continuous_dkuc,
)
from koopman_control.training.reproducibility import make_device, set_seed


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deployment" / "dkuc_continuous.json"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train continuous-time DKUC for Yu-Tan-style KILC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train_dataset", default=None)
    parser.add_argument("--val_dataset", default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    return parser


def _override(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in (
        "train_dataset",
        "val_dataset",
        "val_ratio",
        "dt",
        "seed",
        "device",
        "out_dir",
        "tag",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    for key in ("epochs", "steps_per_epoch"):
        value = getattr(args, key)
        if value is not None:
            config.setdefault("training", {})[key] = value


def _make_output_dir(config: dict[str, Any]) -> Path:
    base = Path(config["out_dir"])
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = str(config.get("tag", ""))
    suffix = f"_{tag}" if tag else ""
    output = base / f"{stamp}_train_dkuc_continuous{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    _override(config, args)
    set_seed(int(config["seed"]))
    device = make_device(str(config["device"]))
    output = _make_output_dir(config)

    source = load_dataset(PROJECT_ROOT / config["train_dataset"])
    if config.get("val_dataset"):
        train_data = source
        val_data = load_dataset(PROJECT_ROOT / config["val_dataset"])
        split_metadata = {
            "mode": "explicit_train_val",
            "train_dataset": config["train_dataset"],
            "val_dataset": config["val_dataset"],
        }
    else:
        train_data, val_data, split_metadata = split_train_val(
            source,
            float(config["val_ratio"]),
            int(config["seed"]),
        )

    x_normer = Normalizer.fit(
        train_data["states"].reshape(-1, train_data["states"].shape[-1])
    )
    u_normer = Normalizer.fit(
        train_data["inputs"].reshape(-1, train_data["inputs"].shape[-1])
    )
    save_normalizers(output / "normalizers.json", x_normer, u_normer)
    save_dataset(output / "dataset_train.npz", train_data)
    save_dataset(output / "dataset_val.npz", val_data)

    train_cfg = ContinuousDKUCTrainingConfig(**config["training"])
    result = fit_continuous_dkuc(
        train_states=train_data["states"],
        train_inputs=train_data["inputs"],
        val_states=val_data["states"],
        val_inputs=val_data["inputs"],
        x_normer=x_normer,
        u_normer=u_normer,
        dt=float(config["dt"]),
        config=train_cfg,
        output_dir=output / "dkuc_continuous",
        device=device,
    )
    summary = {
        "device": str(device),
        "seed": int(config["seed"]),
        "dt": float(config["dt"]),
        "split": split_metadata,
        "train_dataset": config["train_dataset"],
        "training_config": asdict(train_cfg),
        "models": {
            "dkuc": {
                "variant": "continuous",
                "artifact_dir": str(output / "dkuc_continuous"),
                "latent_dim": int(result["latent_dim"]),
                "best_val": float(result["best_val"]),
                "best_epoch": int(result["best_epoch"]),
                "control_capable": True,
            }
        },
    }
    save_json(output / "run_config.json", config)
    save_json(output / "training_summary.json", summary)
    print(f"[done] continuous DKUC artifacts -> {output}")


if __name__ == "__main__":
    main()
