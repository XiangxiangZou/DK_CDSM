"""Execute DKTV Plan 01 from time-varying collection through fixed-DKO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdsm.dktv_data import (  # noqa: E402
    assess_data_quality,
    collect_time_varying_pd,
    prove_time_variation,
    split_nominal_training_stream,
)
from common.model_artifacts import load_prediction_control_model  # noqa: E402
from koopman_control.dktv.config import load_foundation_config  # noqa: E402
from koopman_control.dktv.foundation import (  # noqa: E402
    coordinate_contract_check,
    evaluate_fixed_dko,
    train_and_freeze_initial_model,
)
from prediction.common import save_dataset, save_json  # noqa: E402
from prediction.dkuc_prediction import DKUCModel  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dktv" / "base.json"
PROVENANCE_FILES = (
    "AGENTS.md",
    "DKTV_PLAN_01_FOUNDATION.md",
    "DKTV_PLAN_01_REVIEW.md",
    "DKTV_PLAN_02_HAO_ACCUMULATIVE.md",
    "configs/dktv/base.json",
    "traj_data/mujoco_cdsm.py",
    "prediction/dkuc_prediction.py",
    "src/cdsm/dktv_data.py",
    "src/koopman_control/dktv/config.py",
    "src/koopman_control/dktv/foundation.py",
    "experiments/dktv/plan_01.py",
    "tests/test_dktv_foundation.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DKTV Plan 01: collect, validate, freeze one DKUC model, and evaluate fixed_dko.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-type", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured reproducibility seed")
    parser.add_argument("--run-id", default="", help="Use an exact run id instead of a timestamped id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Artifact root; tests may point this at a temporary directory",
    )
    return parser


def _portable(path: Path, output_root: Path | None = None) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        if output_root is not None:
            try:
                suffix = resolved.relative_to(output_root.resolve()).as_posix()
                return f"${{OUTPUT_ROOT}}/{suffix}"
            except ValueError:
                pass
    return f"${{EXTERNAL}}/{path.name}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, output_root: Path | None = None) -> dict[str, Any]:
    return {
        "path": _portable(path, output_root),
        "sha256": _hash_file(path),
        "bytes": int(path.stat().st_size),
    }


def _tree_records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(directory).as_posix(): {
            "sha256": _hash_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_provenance() -> dict[str, Any]:
    status = _git_value("status", "--porcelain", "--untracked-files=all").splitlines()
    return {
        "branch": _git_value("branch", "--show-current"),
        "commit": _git_value("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _source_provenance() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in PROVENANCE_FILES:
        path = PROJECT_ROOT / relative
        if path.is_file():
            records[relative] = {"sha256": _hash_file(path), "bytes": int(path.stat().st_size)}
    return records


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in arrays.items()})


def _write_log(path: Path, messages: list[str], message: str) -> None:
    messages.append(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(messages) + "\n", encoding="utf-8")
    print(message, flush=True)


def _run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.run_id)
        if clean != args.run_id or not clean:
            raise ValueError("run-id may contain only letters, digits, '-' and '_'")
        return clean
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tag = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.tag).strip("_")
    suffix = f"_{clean_tag}" if clean_tag else ""
    return f"{stamp}_plan01_{args.run_type}{suffix}"


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    config = load_foundation_config(config_path, args.run_type)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    selected_device = _device(args.device)
    run_id = _run_id(args)
    git = _git_provenance()
    sources = _source_provenance()

    result_dir = output_root / "results" / "dktv" / "plan_01" / run_id
    raw_dir = output_root / "data" / "raw" / run_id
    rejected_dir = output_root / "data" / "rejected" / run_id
    processed_dir = output_root / "data" / "processed" / run_id
    model_dir = output_root / "models" / "dktv" / run_id
    for directory in (
        result_dir / "metrics",
        result_dir / "arrays",
        result_dir / "figures",
        result_dir / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    log_path = result_dir / "logs" / "run.log"
    log_messages: list[str] = []
    save_json(result_dir / "config_snapshot.json", config)

    invocation = {
        "entry_module": "experiments.dktv.plan_01",
        "argv": [sys.executable, "-m", "experiments.dktv.plan_01", *sys.argv[1:]],
        "cwd": _portable(Path.cwd(), output_root),
    }
    environment = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "platform": platform.platform(),
    }
    save_json(result_dir / "logs" / "command.json", invocation)
    save_json(result_dir / "logs" / "environment.json", environment)
    _write_log(
        log_path,
        log_messages,
        f"start run_id={run_id} profile={args.run_type} seed={config['seed']} device={selected_device}",
    )
    _write_log(
        log_path,
        log_messages,
        f"provenance commit={git['commit']} dirty={git['dirty']} source_hashes={len(sources)}",
    )

    xml_path = PROJECT_ROOT / config["collection"]["xml"]
    arrays, metadata = collect_time_varying_pd(config, str(xml_path))
    raw_fields = sorted(arrays)
    reproducibility = {
        "seed": int(config["seed"]),
        "raw_field_count": len(raw_fields),
        "raw_fields": raw_fields,
        "required_field_count": 11,
        "source_dataset_filter": "nominal stage only for train and validation_nominal",
    }
    save_json(result_dir / "logs" / "reproducibility.json", reproducibility)
    quality = assess_data_quality(arrays, metadata, config)
    time_variation = prove_time_variation(config, str(xml_path))
    save_json(result_dir / "metrics" / "data_quality.json", quality)
    save_json(result_dir / "metrics" / "time_variation.json", time_variation)
    _write_log(
        log_path,
        log_messages,
        f"collected trajectories={quality['trajectory_count']} steps={quality['steps']} fields={len(raw_fields)}",
    )
    if not quality["accepted"]:
        rejected_dir.mkdir(parents=True, exist_ok=False)
        rejected_dataset = rejected_dir / "dataset.npz"
        _save_npz(rejected_dataset, arrays)
        save_json(rejected_dir / "metadata.json", metadata)
        save_json(rejected_dir / "rejection.json", quality)
        _write_log(log_path, log_messages, f"rejected reasons={quality['rejection_reasons']}")
        blockers = ["data_rejected"]
        if git["dirty"]:
            blockers.append("git_worktree_dirty")
        manifest = {
            "manifest_schema_version": 2,
            "plan": "DKTV_PLAN_01_FOUNDATION",
            "status": "rejected_data",
            "canonical": False,
            "canonical_blockers": blockers,
            "run_id": run_id,
            "git": git,
            "source_files": sources,
            "rejected_dataset": _file_record(rejected_dataset, output_root),
            "quality": quality,
            "time_variation": time_variation,
        }
        save_json(result_dir / "manifest.json", manifest)
        raise RuntimeError(f"collected data rejected: {quality['rejection_reasons']}")

    raw_dir.mkdir(parents=True, exist_ok=False)
    raw_dataset = raw_dir / "dataset.npz"
    _save_npz(raw_dataset, arrays)
    save_json(raw_dir / "metadata.json", metadata)
    save_json(raw_dir / "quality.json", quality)
    _write_log(log_path, log_messages, f"accepted raw_dataset={_portable(raw_dataset, output_root)}")

    train, validation_nominal, validation_stream, split = split_nominal_training_stream(arrays, config)
    processed_dir.mkdir(parents=True, exist_ok=False)
    train_path = processed_dir / "train_nominal.npz"
    validation_nominal_path = processed_dir / "validation_nominal.npz"
    validation_stream_path = processed_dir / "validation_stream.npz"
    split_path = processed_dir / "split.json"
    save_dataset(train_path, train)
    save_dataset(validation_nominal_path, validation_nominal)
    save_dataset(validation_stream_path, validation_stream)
    save_json(split_path, split)
    _write_log(log_path, log_messages, f"split train={split['train_indices']} validation={split['validation_indices']}")

    artifact = train_and_freeze_initial_model(
        train,
        validation_nominal,
        config,
        model_dir,
        _portable(train_path, output_root),
        _portable(validation_nominal_path, output_root),
        selected_device,
    )
    _write_log(log_path, log_messages, f"frozen model={_portable(model_dir, output_root)}")
    model = DKUCModel(model_dir, selected_device)
    coordinate_check = coordinate_contract_check(
        model,
        validation_stream["states"][0, 0],
        validation_stream["inputs"][0, 0],
    )
    save_json(result_dir / "metrics" / "coordinate_contract.json", coordinate_check)
    evaluation = evaluate_fixed_dko(model, validation_stream, config, result_dir)

    control_model = load_prediction_control_model(model_dir, "dkuc", selected_device)
    control_check = {
        "loadable": True,
        "model_name": control_model.name,
        "A_shape": list(control_model.A.shape),
        "B_shape": list(control_model.B.shape),
        "C_shape": list(control_model.C.shape),
        "lifted_dim": int(control_model.A.shape[0]),
        "input_dim": int(control_model.B.shape[1]),
        "C_output_coordinate": "normalized_state",
    }
    save_json(result_dir / "metrics" / "control_interface.json", control_check)
    _write_log(
        log_path,
        log_messages,
        f"control interface A={control_check['A_shape']} B={control_check['B_shape']} coordinate_contract={coordinate_check['passed']}",
    )

    acceptance = {
        "data_quality": bool(quality["accepted"]),
        "time_variation_proof": bool(time_variation["passed"]),
        "coordinate_contract": bool(coordinate_check["passed"]),
        "initial_artifact_loadable": bool(control_check["loadable"]),
        "fixed_dko_degradation_engineering_gate": bool(evaluation["degradation"]["passed"]),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    canonical = bool(acceptance["passed"] and not git["dirty"])
    canonical_blockers: list[str] = []
    if not acceptance["passed"]:
        canonical_blockers.append("acceptance_failed")
    if git["dirty"]:
        canonical_blockers.append("git_worktree_dirty")
    if canonical:
        status = "accepted_canonical"
    elif acceptance["passed"]:
        status = "accepted_noncanonical_dirty"
    else:
        status = "completed_with_failed_acceptance"

    manifest = {
        "manifest_schema_version": 2,
        "plan": "DKTV_PLAN_01_FOUNDATION",
        "status": status,
        "canonical": canonical,
        "canonical_blockers": canonical_blockers,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entry_module": "experiments.dktv.plan_01",
        "arguments": {
            "config": _portable(config_path, output_root),
            "run_type": args.run_type,
            "device_requested": args.device,
            "device_used": selected_device,
            "tag": args.tag,
            "seed_override": args.seed,
            "output_root": _portable(output_root, output_root),
        },
        "seed": int(config["seed"]),
        "python": {"version": platform.python_version(), "conda_environment": "env_dk_cdsm"},
        "platform": platform.platform(),
        "git": git,
        "source_files": sources,
        "config": _file_record(config_path, output_root),
        "datasets": {
            "raw": _file_record(raw_dataset, output_root),
            "raw_metadata": _file_record(raw_dir / "metadata.json", output_root),
            "raw_quality": _file_record(raw_dir / "quality.json", output_root),
            "train": _file_record(train_path, output_root),
            "validation_nominal": _file_record(validation_nominal_path, output_root),
            "validation_stream": _file_record(validation_stream_path, output_root),
            "split_file": _file_record(split_path, output_root),
            "split": split,
        },
        "model": {
            "artifact_dir": _portable(model_dir, output_root),
            "artifact_manifest": _file_record(model_dir / "artifact_manifest.json", output_root),
            "components": artifact["components"],
            "shared_by_methods": artifact["methods"],
            "coordinate_contract": artifact["coordinate_contract"],
        },
        "result_dir": _portable(result_dir, output_root),
        "result_files": {
            "metrics": _tree_records(result_dir / "metrics"),
            "arrays": _tree_records(result_dir / "arrays"),
            "figures": _tree_records(result_dir / "figures"),
        },
        "quality": quality,
        "time_variation": time_variation,
        "coordinate_contract_check": coordinate_check,
        "control_interface": control_check,
        "metrics": evaluation,
        "acceptance": acceptance,
        "interpretation_limit": (
            "The degradation ratio is an engineering gate from one configured stream, "
            "not a multi-seed causal or statistical paper conclusion."
        ),
    }
    save_json(result_dir / "manifest.json", manifest)
    _write_log(
        log_path,
        log_messages,
        f"completed status={status} canonical={canonical} blockers={canonical_blockers}",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = execute(args)
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "canonical": manifest["canonical"],
                "result_dir": manifest["result_dir"],
                "model_dir": manifest["model"]["artifact_dir"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
