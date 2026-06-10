"""Artifact-producing orchestration for the shared Koopman trainers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from koopman_control.data.artifacts import (
    save_json,
    save_normalizers,
)
from koopman_control.data.datasets import save_dataset
from koopman_control.data.normalization import Normalizer
from koopman_control.training.edmd import (
    EDMDTrainingConfig,
    fit_edmd,
)
from koopman_control.training.neural import (
    NeuralTrainingConfig,
    fit_neural_koopman,
)
from koopman_control.training.reproducibility import (
    make_device,
    set_seed,
)


MODEL_ORDER = ("edmd", "dkuc", "dkac", "dkn")
CONTROL_CAPABLE_MODELS = ("edmd", "dkuc", "dkac")


def _fit_normalizers(
    train_data: dict[str, np.ndarray],
) -> tuple[Normalizer, Normalizer]:
    states = np.asarray(train_data["states"], dtype=np.float64)
    inputs = np.asarray(train_data["inputs"], dtype=np.float64)
    return (
        Normalizer.fit(states.reshape(-1, states.shape[-1])),
        Normalizer.fit(inputs.reshape(-1, inputs.shape[-1])),
    )


def train_selected_models(
    *,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    models: Iterable[str],
    neural_config: NeuralTrainingConfig,
    edmd_config: EDMDTrainingConfig,
    device_name: str,
    seed: int,
    output_dir: str | Path,
) -> dict[str, object]:
    """Train selected models and write a portable artifact tree."""
    selected = [name.lower() for name in models]
    unknown = sorted(set(selected) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    set_seed(seed)
    device = make_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_dataset(output / "dataset_train.npz", train_data)
    save_dataset(output / "dataset_val.npz", val_data)

    x_normer, u_normer = _fit_normalizers(train_data)
    save_normalizers(output / "normalizers.json", x_normer, u_normer)

    summary: dict[str, object] = {
        "device": str(device),
        "seed": int(seed),
        "models_requested": selected,
        "control_capable_models": list(CONTROL_CAPABLE_MODELS),
        "state_order": ["qa", "qb", "dqa", "dqb"],
        "input_order": ["tau_a", "tau_b"],
        "train_shape": {
            "states": list(train_data["states"].shape),
            "inputs": list(train_data["inputs"].shape),
        },
        "val_shape": {
            "states": list(val_data["states"].shape),
            "inputs": list(val_data["inputs"].shape),
        },
        "neural_config": asdict(neural_config),
        "edmd_config": asdict(edmd_config),
        "models": {},
    }
    model_summaries = summary["models"]
    assert isinstance(model_summaries, dict)

    if "edmd" in selected:
        print("[train] EDMD")
        result = fit_edmd(
            states=train_data["states"],
            inputs=train_data["inputs"],
            x_normer=x_normer,
            u_normer=u_normer,
            config=edmd_config,
            output_dir=output / "edmd",
        )
        model_summaries["edmd"] = {
            "artifact_dir": str(output / "edmd"),
            "latent_dim": int(result["A"].shape[0]),
            "sigma": float(result["sigma"]),
            "cond_number": float(result["cond_number"]),
            "control_capable": True,
        }

    for kind in ("dkuc", "dkac", "dkn"):
        if kind not in selected:
            continue
        print(f"[train] {kind.upper()}")
        result = fit_neural_koopman(
            kind=kind,
            train_states=train_data["states"],
            train_inputs=train_data["inputs"],
            val_states=val_data["states"],
            val_inputs=val_data["inputs"],
            x_normer=x_normer,
            u_normer=u_normer,
            config=neural_config,
            output_dir=output / kind,
            device=device,
        )
        model = result["model"]
        model_summaries[kind] = {
            "artifact_dir": str(output / kind),
            "latent_dim": int(model.latent_dim),
            "best_val": float(result["best_val"]),
            "best_epoch": int(result["best_epoch"]),
            "control_capable": bool(result["control_capable"]),
        }
        if kind in {"dkac", "dkn"}:
            model_summaries[kind]["control_dim_hat"] = int(
                model.control_dim_hat
            )

    save_json(output / "training_summary.json", summary)
    return summary
