"""Execute DKTV Plan 02 accumulative online Koopman updates and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction.dktv.config import stage_bounds  # noqa: E402
from prediction.dktv.online_model import (  # noqa: E402
    artifact_fingerprint,
    evaluate_methods,
    run_accumulative_replay,
    save_comparison_figures,
    update_summary,
)
from prediction.common import load_json, save_json  # noqa: E402
from prediction.dkuc_prediction import DKUCModel  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "prediction" / "dktv_accumulative_config.json"
PROVENANCE_FILES = (
    "AGENTS.md",
    "docs/dktv/plans/DKTV_PLAN_02_HAO_ACCUMULATIVE.md",
    "docs/dktv/formula_mapping/DKTV_PLAN_02_FORMULA_MAPPING.md",
    "prediction/dktv_base_config.json",
    "prediction/dktv_accumulative_config.json",
    "prediction/dkuc_prediction.py",
    "prediction/dktv/least_squares.py",
    "prediction/dktv/accumulative_update.py",
    "prediction/dktv/online_model.py",
    "prediction/dktv_accumulative_prediction.py",
    "tests/dktv/test_accumulative.py",
)
UPDATE_HISTORY_SCHEMA_VERSION = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Plan 02 accumulative DKTV against the shared Plan 01 fixed DKO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-type", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--plan01-run", required=True, help="Exact Plan 01 run id to reuse")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_plan02_config(path: str | Path, profile: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "unsupported Plan 02 config schema")
    _require(payload.get("method") == "dktv_accumulative", "method id is not frozen")
    _require(payload.get("baseline") == "fixed_dko", "baseline id is not frozen")
    _require(
        "fixed encoder" in payload.get("display_names", {}).get("dktv_accumulative", ""),
        "accumulative display name must disclose the fixed encoder boundary",
    )
    _require(float(payload.get("ridge_lambda", 0.0)) > 0.0, "ridge_lambda must be positive")
    _require(float(payload.get("oracle_tolerance", 0.0)) > 0.0, "oracle_tolerance must be positive")
    coordinates = payload.get("coordinates", {})
    _require(coordinates.get("state") == "normalized", "state coordinate must be normalized")
    _require(
        coordinates.get("input") == "normalized_applied_torque",
        "input coordinate must be normalized applied torque",
    )
    _require(coordinates.get("readout") == "normalized_state", "C0 readout must be normalized")
    _require(profile in payload.get("profiles", {}), f"unknown profile: {profile}")
    _require(
        payload.get("invalid_batch_policy") == "discard_invalid_batch",
        "invalid batch policy must be discard_invalid_batch",
    )
    config = deepcopy(payload)
    config["profile_name"] = profile
    config["profile"] = deepcopy(payload["profiles"][profile])
    _require(all(int(value) > 0 for value in config["profile"]["batch_sizes"]), "invalid batch size")
    return config


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _portable(path: Path, output_root: Path | None = None) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        if output_root is not None:
            try:
                return f"${{OUTPUT_ROOT}}/{resolved.relative_to(output_root.resolve()).as_posix()}"
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
        ["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
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
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.run_id)
        if clean != args.run_id or not clean:
            raise ValueError("run-id may contain only letters, digits, '-' and '_'")
        return clean
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tag = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.tag).strip("_")
    suffix = f"_{clean_tag}" if clean_tag else ""
    return f"{stamp}_plan02_{args.run_type}{suffix}"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _history_arrays(replays: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "update_history_schema_version": np.asarray(
            UPDATE_HISTORY_SCHEMA_VERSION, dtype=np.int64
        )
    }
    fields = (
        "attempt_index",
        "update_index",
        "model_version",
        "accepted",
        "previous_sample_count",
        "batch_sample_count",
        "cumulative_sample_count",
        "time_step",
        "online_sample_count",
        "accepted_online_sample_count",
        "update_time_s",
        "oracle_check_performed",
        "oracle_refit_time_s",
        "pre_update_batch_rmse",
        "post_update_batch_rmse",
        "recursive_rank",
        "recursive_minimum_singular_value",
        "recursive_condition_number",
        "recursive_regularized_condition_number",
        "recursive_spectral_radius_A",
        "recursive_finite",
        "oracle_A_max_abs_difference",
        "oracle_B_max_abs_difference",
        "oracle_rank",
        "oracle_minimum_singular_value",
        "oracle_condition_number",
        "oracle_regularized_condition_number",
        "updater_statistics_memory_bytes",
        "oracle_history_memory_bytes",
    )
    integer_fields = {
        "attempt_index",
        "update_index",
        "model_version",
        "previous_sample_count",
        "batch_sample_count",
        "cumulative_sample_count",
        "time_step",
        "online_sample_count",
        "accepted_online_sample_count",
        "oracle_rank",
        "updater_statistics_memory_bytes",
        "oracle_history_memory_bytes",
    }
    boolean_fields = {"accepted", "oracle_check_performed", "recursive_finite"}
    for method, replay in replays.items():
        flattened: list[dict[str, Any]] = []
        for record in replay.update_history:
            values = dict(record)
            diagnostics = record.get("diagnostics") or {}
            values.update(
                {
                    "recursive_rank": diagnostics.get("rank"),
                    "recursive_minimum_singular_value": diagnostics.get(
                        "minimum_singular_value"
                    ),
                    "recursive_condition_number": diagnostics.get("condition_number"),
                    "recursive_regularized_condition_number": diagnostics.get(
                        "regularized_condition_number"
                    ),
                    "recursive_spectral_radius_A": diagnostics.get("spectral_radius_A"),
                    "recursive_finite": diagnostics.get("finite", False),
                }
            )
            flattened.append(values)
        for field in fields:
            dtype = bool if field in boolean_fields else np.int64 if field in integer_fields else np.float64
            missing = False if field in boolean_fields else -1 if field in integer_fields else np.nan
            result[f"{method}_{field}"] = np.asarray(
                [record.get(field) if record.get(field) is not None else missing for record in flattened],
                dtype=dtype,
            )
    return result


def _history_schema(replays: dict[str, Any]) -> dict[str, Any]:
    return {
        "update_history_schema_version": UPDATE_HISTORY_SCHEMA_VERSION,
        "numeric_storage": "arrays/update_history.npz",
        "decision_storage": "logs/update_history.jsonl",
        "methods": list(replays),
        "missing_numeric_value": "NaN for float fields; -1 for integer fields",
        "decision_fields": [
            "method",
            "attempt_index",
            "update_index",
            "model_version",
            "time_step",
            "accepted",
            "reason",
            "invalid_batch_policy",
            "batch_disposition",
            "diagnostics",
            "oracle_check_performed",
        ],
        "pickle_required": False,
    }


def _write_history_jsonl(path: Path, replays: dict[str, Any]) -> None:
    lines = []
    for method, replay in replays.items():
        for record in replay.update_history:
            lines.append(json.dumps({"method": method, **record}, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _source_scenario_contract(
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "plan01_manifest_schema_version": int(source_manifest["manifest_schema_version"]),
        "artifact_schema_version": int(artifact_manifest["artifact_schema_version"]),
        "profile_name": source_config["profile_name"],
        "profile": {
            "trajectory_count": int(source_config["profile"]["trajectory_count"]),
            "steps": int(source_config["profile"]["steps"]),
        },
        "state": source_config["state"],
        "state_dim": int(source_config["state_dim"]),
        "input": source_config["input"],
        "input_dim": int(source_config["input_dim"]),
        "sample_dt_s": float(source_config["sample_dt_s"]),
        "lift": source_config["lift"],
        "lifted_dim": int(source_config["lifted_dim"]),
        "model_form": source_config["model_form"],
        "state_readout": source_config["state_readout"],
        "coordinates": source_config["coordinates"],
        "disturbance": source_config["disturbance"],
    }


def _comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    fixed_one = float(metrics["one_step"]["fixed_dko"]["total_rmse"])
    result: dict[str, Any] = {}
    for method in metrics["one_step"]:
        if method == "fixed_dko":
            continue
        method_one = float(metrics["one_step"][method]["total_rmse"])
        result[method] = {
            "one_step_fixed_to_method_ratio": fixed_one / max(method_one, np.finfo(float).eps),
            "one_step_improved": bool(method_one < fixed_one),
            "rollout_fixed_to_method_ratio": {
                horizon: float(metrics["rollout"]["fixed_dko"][horizon]["total_rmse"])
                / max(float(values["total_rmse"]), np.finfo(float).eps)
                for horizon, values in metrics["rollout"][method].items()
            },
            "stage_fixed_to_method_ratio": {
                stage: float(metrics["segmented"]["fixed_dko"][stage]["total_rmse"])
                / max(float(values["total_rmse"]), np.finfo(float).eps)
                for stage, values in metrics["segmented"][method].items()
            },
        }
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(PROJECT_ROOT, args.config)
    output_root = _resolve(PROJECT_ROOT, args.output_root)
    source_output_root = _resolve(PROJECT_ROOT, args.source_output_root)
    config = load_plan02_config(config_path, args.run_type)
    selected_device = _device(args.device)
    run_id = _make_run_id(args)
    git = _git_provenance()
    sources = _source_provenance()

    result_dir = output_root / "results" / "dktv" / "plan_02" / run_id
    for name in ("metrics", "arrays", "figures", "logs"):
        (result_dir / name).mkdir(parents=True, exist_ok=False)
    save_json(result_dir / "config_snapshot.json", config)
    save_json(
        result_dir / "logs" / "command.json",
        {
            "entry_module": "prediction.dktv_accumulative_prediction",
            "argv": [sys.executable, "-m", "prediction.dktv_accumulative_prediction", *sys.argv[1:]],
            "cwd": _portable(Path.cwd(), output_root),
        },
    )
    save_json(
        result_dir / "logs" / "environment.json",
        {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "platform": platform.platform(),
        },
    )

    log_messages: list[str] = []
    log_path = result_dir / "logs" / "run.log"

    def log(message: str) -> None:
        log_messages.append(message)
        log_path.write_text("\n".join(log_messages) + "\n", encoding="utf-8")
        print(message, flush=True)

    source_result = source_output_root / "results" / "dktv" / "plan_01" / args.plan01_run
    source_manifest_path = source_result / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_config = load_json(source_result / "config_snapshot.json")
    model_dir = source_output_root / "models" / "dktv" / args.plan01_run
    train_path = model_dir / "dataset_train.npz"
    stream_path = source_output_root / "data" / "processed" / args.plan01_run / "validation_stream.npz"
    artifact_manifest_path = model_dir / "artifact_manifest.json"
    artifact_manifest = load_json(artifact_manifest_path)
    _require(source_manifest["acceptance"]["passed"], "source Plan 01 acceptance failed")
    _require(artifact_manifest["artifact_schema_version"] >= 2, "Plan 01 artifact schema v2 required")
    _require(
        artifact_manifest["coordinate_contract"]["input_coordinate"]
        == "normalized_applied_torque",
        "source input coordinate mismatch",
    )
    log(
        f"source plan01_run={args.plan01_run} status={source_manifest['status']} "
        f"canonical={source_manifest['canonical']}"
    )

    train_data = _load_npz(train_path)
    stream_data = _load_npz(stream_path)
    maximum_steps = min(int(config["profile"]["maximum_steps"]), stream_data["inputs"].shape[1])
    stream_data = {
        **stream_data,
        "states": stream_data["states"][:, : maximum_steps + 1],
        "inputs": stream_data["inputs"][:, :maximum_steps],
        "applied_torque": stream_data["applied_torque"][:, :maximum_steps],
        "t": stream_data["t"][:, : maximum_steps + 1],
        "disturbance_torque": stream_data["disturbance_torque"][:, :maximum_steps],
    }
    model = DKUCModel(model_dir, selected_device)
    fingerprint = artifact_fingerprint(model_dir)
    replays: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for batch_size in config["profile"]["batch_sizes"]:
        method = f"dktv_accumulative_b{int(batch_size)}"
        log(f"replay method={method} batch_size={batch_size}")
        replay = run_accumulative_replay(
            model,
            train_data,
            stream_data,
            batch_size=int(batch_size),
            ridge_lambda=float(config["ridge_lambda"]),
            encoder_fingerprint=fingerprint,
            oracle_tolerance=float(config["oracle_tolerance"]),
            invalid_batch_policy=config["invalid_batch_policy"],
        )
        replays[method] = replay
        summaries[method] = update_summary(replay)
        replay.updater.save(result_dir / "arrays" / f"{method}_updater_final.npz")
        log(
            f"completed method={method} updates={summaries[method]['update_count']} "
            f"samples={summaries[method]['final_sample_count']} "
            f"oracle_A_diff={summaries[method]['maximum_oracle_A_difference']:.3g}"
        )

    stages = stage_bounds(source_config, maximum_steps)
    metrics, prediction_arrays = evaluate_methods(
        model,
        stream_data,
        replays,
        rollout_horizons=[
            value for value in config["evaluation"]["rollout_horizons"] if value <= maximum_steps
        ],
        window_stride=int(config["evaluation"]["window_stride"]),
        stage_definitions=stages,
        stage_rollout_horizon=int(config["evaluation"]["stage_rollout_horizon"]),
    )
    comparison = _comparison(metrics)
    save_json(result_dir / "metrics" / "one_step.json", metrics["one_step"])
    save_json(result_dir / "metrics" / "rollout.json", metrics["rollout"])
    save_json(result_dir / "metrics" / "segmented.json", metrics["segmented"])
    save_json(result_dir / "metrics" / "update_summary.json", summaries)
    save_json(result_dir / "metrics" / "comparison.json", comparison)

    for method, replay in replays.items():
        prediction_arrays[f"{method}_A_by_step"] = replay.A_by_step
        prediction_arrays[f"{method}_B_by_step"] = replay.B_by_step
        prediction_arrays[f"{method}_model_version_by_step"] = replay.model_version_by_step
    predictions_path = result_dir / "arrays" / "predictions.npz"
    history_path = result_dir / "arrays" / "update_history.npz"
    history_schema_path = result_dir / "arrays" / "update_history_schema.json"
    history_jsonl_path = result_dir / "logs" / "update_history.jsonl"
    _save_npz(predictions_path, prediction_arrays)
    _save_npz(history_path, _history_arrays(replays))
    save_json(history_schema_path, _history_schema(replays))
    _write_history_jsonl(history_jsonl_path, replays)
    display_names = {
        "fixed_dko": config["display_names"]["fixed_dko"],
        **{
            method: config["display_names"]["dktv_accumulative"].format(
                batch_size=replay.batch_size
            )
            for method, replay in replays.items()
        },
    }
    figures = save_comparison_figures(
        result_dir,
        metrics,
        prediction_arrays,
        summaries,
        display_names=display_names,
    )

    acceptance = {
        "source_plan01_engineering_acceptance": bool(source_manifest["acceptance"]["passed"]),
        "artifact_schema_v2": bool(artifact_manifest["artifact_schema_version"] >= 2),
        "fixed_encoder_and_normalizer": True,
        "recursive_matches_direct_refit": bool(
            all(replay.oracle_tolerance_passed for replay in replays.values())
        ),
        "sample_counts_monotonic": bool(
            all(replay.sample_count_monotonic for replay in replays.values())
        ),
        "all_updates_finite": bool(all(replay.all_updates_finite for replay in replays.values())),
        "no_rejected_updates": bool(
            all(replay.rejected_sample_count == 0 for replay in replays.values())
        ),
        "no_pending_snapshots": bool(all(replay.pending_sample_count == 0 for replay in replays.values())),
        "prediction_arrays_finite": bool(
            all(np.all(np.isfinite(values)) for values in prediction_arrays.values())
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    canonical = bool(
        acceptance["passed"] and not git["dirty"] and bool(source_manifest["canonical"])
    )
    blockers: list[str] = []
    if not acceptance["passed"]:
        blockers.append("acceptance_failed")
    if git["dirty"]:
        blockers.append("git_worktree_dirty")
    if not source_manifest["canonical"]:
        blockers.append("source_plan01_noncanonical")
    status = (
        "accepted_canonical"
        if canonical
        else "accepted_noncanonical" if acceptance["passed"] else "completed_with_failed_acceptance"
    )
    log(f"acceptance status={status} canonical={canonical} blockers={blockers}")

    manifest = {
        "manifest_schema_version": 2,
        "plan": "DKTV_PLAN_02_HAO_ACCUMULATIVE",
        "method": "dktv_accumulative",
        "baseline": "fixed_dko",
        "status": status,
        "canonical": canonical,
        "canonical_blockers": blockers,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entry_module": "prediction.dktv_accumulative_prediction",
        "arguments": {
            "config": _portable(config_path, output_root),
            "run_type": args.run_type,
            "plan01_run": args.plan01_run,
            "device_requested": args.device,
            "device_used": selected_device,
            "tag": args.tag,
        },
        "git": git,
        "source_files": sources,
        "config": _file_record(config_path, output_root),
        "source_plan01": {
            "run_id": args.plan01_run,
            "manifest": _file_record(source_manifest_path, source_output_root),
            "status": source_manifest["status"],
            "canonical": source_manifest["canonical"],
            "seed": source_manifest["seed"],
            "model_artifact": _file_record(artifact_manifest_path, source_output_root),
            "training_dataset": _file_record(train_path, source_output_root),
            "stream_dataset": _file_record(stream_path, source_output_root),
            "encoder_fingerprint": fingerprint,
            "scenario_contract": _source_scenario_contract(
                source_manifest, source_config, artifact_manifest
            ),
        },
        "display_names": display_names,
        "coordinates": config["coordinates"],
        "batch_sizes": config["profile"]["batch_sizes"],
        "stream": {
            "trajectory_count": int(stream_data["states"].shape[0]),
            "steps": maximum_steps,
            "online_snapshot_count": int(stream_data["inputs"].shape[0] * maximum_steps),
            "ordering": "time_major_then_trajectory",
            "causal_order": "predict_then_observe_then_update",
            "physical_semantics": "synchronous_multi_trajectory_simulation",
            "snapshots_per_simulation_step": int(stream_data["states"].shape[0]),
            "batch_size_to_simulation_steps": {
                str(batch_size): float(batch_size) / float(stream_data["states"].shape[0])
                for batch_size in config["profile"]["batch_sizes"]
            },
            "single_physical_robot_stream": False,
            "plan03_fairness_requirement": (
                "reuse identical time_major_then_trajectory ordering and batch semantics"
            ),
        },
        "update_history_contract": {
            "schema_version": UPDATE_HISTORY_SCHEMA_VERSION,
            "invalid_batch_policy": config["invalid_batch_policy"],
            "rejected_batch_disposition": "discarded_without_retry_or_oracle_append",
            "schema": _file_record(history_schema_path, output_root),
            "decision_log": _file_record(history_jsonl_path, output_root),
        },
        "metrics": metrics,
        "comparison": comparison,
        "update_summary": summaries,
        "figures": figures,
        "result_dir": _portable(result_dir, output_root),
        "result_files": {
            "metrics": _tree_records(result_dir / "metrics"),
            "arrays": _tree_records(result_dir / "arrays"),
            "figures": _tree_records(result_dir / "figures"),
            "logs": _tree_records(result_dir / "logs"),
        },
        "acceptance": acceptance,
        "control_experiment": {
            "executed": False,
            "reason": "Plan 02 prediction gate is evaluated first; MPC comparison remains optional.",
        },
        "interpretation_limit": (
            "Prediction improvements are engineering evidence for the configured seeds and stream; "
            "they are not yet a final paper-level statistical conclusion."
        ),
    }
    save_json(result_dir / "manifest.json", manifest)
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
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
