"""Compare EDMD, DKUC, DKAC, and DKN construction and prediction quality."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from koopman_control.data.datasets import load_dataset
from koopman_control.evaluation.prediction import evaluate_model
from koopman_control.models.registry import load_prediction_model

METHODS = ("edmd", "dkuc", "dkac", "dkn")
METHOD_LABELS = {
    "edmd": "EDMD",
    "dkuc": "DKUC",
    "dkac": "DKAC",
    "dkn": "DKN",
}
METHOD_COLORS = {
    "edmd": "#4C78A8",
    "dkuc": "#F58518",
    "dkac": "#54A24B",
    "dkn": "#E45756",
}
CONSTRUCTION = {
    "edmd": {
        "lifting": "Fixed Gaussian RBF dictionary",
        "control_encoding": "Physical normalized control u",
        "dynamics": "z_next = A z + B u",
        "training": "K-means centers + ridge regression",
        "control_capable": True,
    },
    "dkuc": {
        "lifting": "Learned state encoder phi(x)",
        "control_encoding": "Physical normalized control u",
        "dynamics": "z_next = A z + B u",
        "training": "Multi-step neural training",
        "control_capable": True,
    },
    "dkac": {
        "lifting": "Learned state encoder phi(x)",
        "control_encoding": "State-dependent affine map v = G(x)u",
        "dynamics": "z_next = A z + B v",
        "training": "Multi-step neural training",
        "control_capable": True,
    },
    "dkn": {
        "lifting": "Learned state encoder phi(x)",
        "control_encoding": "Nonlinear encoder u_hat = g(x,u)",
        "dynamics": "z_next = A z + B u_hat",
        "training": "Multi-step neural training",
        "control_capable": False,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare four Koopman model constructions on one artifact set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument(
        "--dataset",
        default="",
        help="Validation dataset; defaults to artifact_dir/dataset_val.npz.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "outputs" / "results" / "model_construction_comparison"),
    )
    parser.add_argument("--tag", default="")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _edmd_parameter_counts(model: Any) -> tuple[int, int, int]:
    linear = int(model.A.size + model.B.size)
    dictionary = int(model.centers.size + 1)
    return linear + dictionary, linear, dictionary


def _neural_parameter_counts(model: Any) -> tuple[int, int, int]:
    parameters = list(model.model.named_parameters())
    total = sum(int(value.numel()) for _, value in parameters)
    linear = sum(
        int(value.numel())
        for name, value in parameters
        if name.startswith("A.") or name.startswith("B.")
    )
    nonlinear = total - linear
    return total, linear, nonlinear


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = Path(base_dir) / f"{stamp}_four_model_construction{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_complexity_plot(rows: list[dict[str, Any]], out_dir: Path) -> str:
    labels = [row["method"].upper() for row in rows]
    colors = [METHOD_COLORS[row["method"]] for row in rows]
    parameters = [row["learned_numeric_parameters"] for row in rows]
    sizes = [row["artifact_size_kb"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, parameters, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Learned numeric parameters")
    axes[0].set_title("Model complexity")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, sizes, color=colors)
    axes[1].set_ylabel("Artifact size (KiB)")
    axes[1].set_title("Saved model footprint")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    filename = "model_complexity_comparison.png"
    fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def _save_rmse_plot(rows: list[dict[str, Any]], out_dir: Path) -> str:
    labels = [row["method"].upper() for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    one_step = [row["one_step_total_rmse"] for row in rows]
    rollout = [row["rollout_total_rmse"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, one_step, width, label="One-step")
    ax.bar(x + width / 2, rollout, width, label="Rollout")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Total RMSE (log scale)")
    ax.set_title("Prediction quality on the shared validation set")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    filename = "prediction_rmse_comparison.png"
    fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "latent_dim",
        "internal_control_dim",
        "learned_numeric_parameters",
        "linear_operator_parameters",
        "nonlinear_or_dictionary_parameters",
        "artifact_size_kb",
        "control_capable",
        "one_step_total_rmse",
        "rollout_total_rmse",
        "rollout_final_step_rmse",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _save_markdown(
    rows: list[dict[str, Any]],
    artifact_dir: Path,
    dataset_path: Path,
    path: Path,
) -> None:
    lines = [
        "# EDMD / DKUC / DKAC / DKN Model Construction Comparison",
        "",
        f"- Artifact set: `{artifact_dir}`",
        f"- Validation dataset: `{dataset_path}`",
        "",
        "| Method | Lifting | Control encoding | Latent dim | Parameters | Control-ready | One-step RMSE | Rollout RMSE |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {lifting} | {control} | {latent} | {params} | {ready} | {one:.6g} | {roll:.6g} |".format(
                label=METHOD_LABELS[row["method"]],
                lifting=row["lifting"],
                control=row["control_encoding"],
                latent=row["latent_dim"],
                params=row["learned_numeric_parameters"],
                ready="yes" if row["control_capable"] else "no",
                one=row["one_step_total_rmse"],
                roll=row["rollout_total_rmse"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- EDMD uses a fixed RBF dictionary and closed-form regression; it has no neural encoder.",
            "- DKUC learns the lifting map while keeping the physical control input unchanged.",
            "- DKAC adds a state-dependent affine control map, preserving a control-oriented internal input.",
            "- DKN uses the most flexible nonlinear state-control encoder, but the current project treats it as prediction-only.",
            "- One-step RMSE measures local fit; rollout RMSE is the stronger test of autonomous error accumulation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    dataset_path = (
        Path(args.dataset).resolve()
        if args.dataset
        else artifact_dir / "dataset_val.npz"
    )
    output_dir = _make_output_dir(args.out_dir, args.tag)
    dataset = load_dataset(dataset_path)
    training_summary = _load_json(artifact_dir / "training_summary.json")

    rows: list[dict[str, Any]] = []
    detailed_metrics: dict[str, Any] = {}
    for method in METHODS:
        print(f"[compare] loading {method}")
        model = load_prediction_model(artifact_dir, method, args.device)
        model_dir = artifact_dir / method
        config = _load_json(model_dir / "model_config.json")["config"]
        if method == "edmd":
            total_params, linear_params, nonlinear_params = _edmd_parameter_counts(model)
            internal_control_dim = int(model.B.shape[1])
        else:
            total_params, linear_params, nonlinear_params = _neural_parameter_counts(model)
            internal_control_dim = int(model.B.shape[1])

        one_step = evaluate_model(model, dataset, "one_step")["metrics"]
        rollout = evaluate_model(model, dataset, "rollout")["metrics"]
        construction = CONSTRUCTION[method]
        row = {
            "method": method,
            "model_form": construction["dynamics"],
            "lifting": construction["lifting"],
            "control_encoding": construction["control_encoding"],
            "training_method": construction["training"],
            "control_capable": construction["control_capable"],
            "state_dim": int(model.state_dim),
            "latent_dim": int(model.latent_dim),
            "internal_control_dim": internal_control_dim,
            "learned_numeric_parameters": total_params,
            "linear_operator_parameters": linear_params,
            "nonlinear_or_dictionary_parameters": nonlinear_params,
            "artifact_size_bytes": _directory_size(model_dir),
            "artifact_size_kb": round(_directory_size(model_dir) / 1024.0, 3),
            "one_step_total_rmse": one_step["total_rmse"],
            "rollout_total_rmse": rollout["total_rmse"],
            "rollout_final_step_rmse": rollout["final_step_rmse"],
            "model_config": config,
        }
        if method == "edmd":
            row["rbf_centers"] = int(model.centers.shape[0])
            row["rbf_sigma"] = float(model.sigma)
            row["edmd_condition_number"] = float(model.cond_number)
        rows.append(row)
        detailed_metrics[method] = {"one_step": one_step, "rollout": rollout}
        print(
            f"  latent={row['latent_dim']}, params={total_params}, "
            f"one_step={one_step['total_rmse']:.6g}, "
            f"rollout={rollout['total_rmse']:.6g}"
        )

    figures = [
        _save_complexity_plot(rows, output_dir),
        _save_rmse_plot(rows, output_dir),
    ]
    payload = {
        "artifact_dir": str(artifact_dir),
        "dataset": str(dataset_path),
        "dataset_shape": {
            "states": list(dataset["states"].shape),
            "inputs": list(dataset["inputs"].shape),
        },
        "training_context": {
            "device": training_summary.get("device"),
            "seed": training_summary.get("seed"),
            "hyper_params": training_summary.get("hyper_params"),
        },
        "models": {row["method"]: row for row in rows},
        "detailed_prediction_metrics": detailed_metrics,
        "figures": figures,
    }
    (output_dir / "model_construction_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_csv(rows, output_dir / "model_construction_comparison.csv")
    _save_markdown(
        rows,
        artifact_dir,
        dataset_path,
        output_dir / "model_construction_comparison.md",
    )
    print(f"[done] comparison results -> {output_dir}")


if __name__ == "__main__":
    main()
