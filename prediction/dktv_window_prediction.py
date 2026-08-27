"""Execute Plan 03 sliding-window and selective Koopman comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_CONFIG = PROJECT_ROOT / "prediction" / "dktv_window_config.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traj_data.dktv_data import assess_data_quality, collect_time_varying_pd  # noqa: E402
from prediction.dktv_accumulative_prediction import (  # noqa: E402
    _device,
    _file_record,
    _git_provenance,
    _load_npz,
    _portable,
    _resolve,
    _source_scenario_contract,
    _tree_records,
)
from prediction.dktv.config import stage_bounds  # noqa: E402
from prediction.dktv.online_model import (  # noqa: E402
    artifact_fingerprint,
    evaluate_methods,
    normalized_pairs,
    run_accumulative_replay,
    run_window_replay,
    update_summary,
    window_update_summary,
)
from prediction.dktv.window_update import latent_rmse  # noqa: E402
from prediction.common import load_json, save_json  # noqa: E402
from prediction.dkuc_prediction import DKUCModel  # noqa: E402


PROVENANCE_FILES = (
    "AGENTS.md",
    "docs/dktv/plans/DKTV_PLAN_03_ZHANG_SLIDING_WINDOW.md",
    "docs/dktv/formula_mapping/DKTV_PLAN_03_FORMULA_MAPPING.md",
    "prediction/dktv_base_config.json",
    "prediction/dktv_accumulative_config.json",
    "prediction/dktv_window_config.json",
    "prediction/dktv/least_squares.py",
    "prediction/dktv/accumulative_update.py",
    "prediction/dktv/window_update.py",
    "prediction/dktv/selective_update.py",
    "prediction/dktv/online_model.py",
    "prediction/dktv_window_prediction.py",
    "tests/dktv/test_window_update.py",
    "tests/dktv/test_selective_update.py",
    "tests/dktv/test_window_replay.py",
)
HISTORY_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Plan 03 fixed, accumulative, window, and selective DKTV comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-type", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--plan01-run", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_plan03_config(path: str | Path, profile: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "unsupported Plan 03 config schema")
    _require(
        payload.get("methods")
        == ["fixed_dko", "dktv_accumulative", "otvdkl_window", "otvdkl_selective"],
        "four method identifiers are not frozen",
    )
    _require(float(payload.get("ridge_lambda", 0.0)) > 0.0, "ridge_lambda must be positive")
    _require(float(payload.get("oracle_tolerance", 0.0)) > 0.0, "oracle tolerance must be positive")
    _require(
        payload.get("reject_buffer_policy") in ("discard_on_reject", "retain_on_reject"),
        "invalid reject buffer policy",
    )
    _require(profile in payload.get("profiles", {}), f"unknown profile: {profile}")
    config = deepcopy(payload)
    config["profile_name"] = profile
    config["profile"] = deepcopy(payload["profiles"][profile])
    for window_size, batch_size in config["profile"]["window_ablation"]:
        _require(int(window_size) >= 18, "window must satisfy w >= r+m=18")
        _require(0 < int(batch_size) <= int(window_size), "invalid window/batch pair")
    return config


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": _hash_file(PROJECT_ROOT / name), "bytes": (PROJECT_ROOT / name).stat().st_size}
        for name in PROVENANCE_FILES
        if (PROJECT_ROOT / name).is_file()
    }


def _make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.run_id)
        if clean != args.run_id or not clean:
            raise ValueError("run-id may contain only letters, digits, '-' and '_'")
        return clean
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.tag).strip("_")
    return f"{stamp}_plan03_{args.run_type}" + (f"_{tag}" if tag else "")


def _trim_stream(data: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
    result = dict(data)
    for name in ("states", "t"):
        if name in result:
            result[name] = np.asarray(result[name])[:, : steps + 1].copy()
    for name in ("inputs", "applied_torque", "disturbance_torque"):
        if name in result:
            result[name] = np.asarray(result[name])[:, :steps].copy()
    return result


def _calibrate_epsilon(
    model: DKUCModel,
    stream: dict[str, np.ndarray],
    *,
    batch_size: int,
    quantile: float,
) -> tuple[float, list[float]]:
    z, target, u = normalized_pairs(
        model,
        np.asarray(stream["states"]),
        np.asarray(stream["inputs"]),
    )
    current = z.transpose(1, 0, 2).reshape(-1, z.shape[-1])
    next_values = target.transpose(1, 0, 2).reshape(-1, target.shape[-1])
    inputs = u.transpose(1, 0, 2).reshape(-1, u.shape[-1])
    values = [
        latent_rmse(model.A, model.B, current[start : start + batch_size],
                    inputs[start : start + batch_size], next_values[start : start + batch_size])
        for start in range(0, current.shape[0] - batch_size + 1, batch_size)
    ]
    _require(bool(values), "epsilon calibration produced no complete batch")
    return float(np.quantile(values, quantile)), values


def _independent_calibration_stream(
    source_config: dict[str, Any], calibration: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    calibration_config = deepcopy(source_config)
    calibration_config["seed"] = int(calibration["seed"])
    calibration_config["profile"] = {
        **calibration_config["profile"],
        "trajectory_count": int(calibration["trajectory_count"]),
        "steps": int(calibration["steps"]),
    }
    xml_path = PROJECT_ROOT / calibration_config["collection"]["xml"]
    collected, metadata = collect_time_varying_pd(calibration_config, str(xml_path))
    quality = assess_data_quality(collected, metadata, calibration_config)
    _require(
        quality["accepted"],
        f"independent calibration dataset failed quality checks: {quality['rejection_reasons']}",
    )
    nominal_stop = int(stage_bounds(calibration_config, int(calibration["steps"]))[0]["end_step"])
    stream = _trim_stream(collected, nominal_stop)
    return stream, {
        "seed": int(calibration["seed"]),
        "trajectory_count": int(calibration["trajectory_count"]),
        "configured_steps": int(calibration["steps"]),
        "nominal_steps_used": nominal_stop,
        "data_quality": quality,
        "evaluation_stream_overlap": False,
    }


def _scenario_stream(
    source_stream: dict[str, np.ndarray],
    source_config: dict[str, Any],
    *,
    rate_multiplier: float,
    noise_std: float,
    steps: int,
    trajectory_count: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if rate_multiplier == 1.0:
        stream = _trim_stream(source_stream, steps)
        origin = "plan01_validation_stream"
        quality: dict[str, Any] = {"accepted": True, "source": "accepted_plan01_dataset"}
    else:
        scenario_config = deepcopy(source_config)
        scenario_config["seed"] = int(seed + round(rate_multiplier * 1000))
        scenario_config["profile"] = {
            **scenario_config["profile"],
            "trajectory_count": int(trajectory_count),
            "steps": int(steps),
        }
        scenario_config["disturbance"]["angular_frequency_rad_s"] = [
            float(value) * float(rate_multiplier)
            for value in source_config["disturbance"]["angular_frequency_rad_s"]
        ]
        xml_path = PROJECT_ROOT / scenario_config["collection"]["xml"]
        collected, metadata = collect_time_varying_pd(scenario_config, str(xml_path))
        quality = assess_data_quality(collected, metadata, scenario_config)
        _require(quality["accepted"], f"rate ablation dataset failed quality checks: {quality['rejection_reasons']}")
        stream = _trim_stream(collected, steps)
        origin = "plan03_recollected_rate_ablation"
    clean_states = np.asarray(stream["states"], dtype=np.float64).copy()
    stream = dict(stream)
    stream["states_clean"] = clean_states
    if float(noise_std) > 0.0:
        rng = np.random.default_rng(seed + int(round(rate_multiplier * 10000)) + int(noise_std * 1e9))
        stream["states"] = clean_states + rng.normal(
            scale=float(noise_std), size=clean_states.shape
        )
    observed_states = np.asarray(stream["states"], dtype=np.float64)
    return stream, {
        "rate_multiplier": float(rate_multiplier),
        "measurement_noise_std_physical_state": float(noise_std),
        "origin": origin,
        "seed": int(seed),
        "data_quality": quality,
        "observed_states_finite": bool(np.all(np.isfinite(observed_states))),
        "observed_state_min": np.min(observed_states, axis=(0, 1)).tolist(),
        "observed_state_max": np.max(observed_states, axis=(0, 1)).tolist(),
        "clean_truth_preserved": True,
        "online_update_state": "states_observed",
        "primary_evaluation_truth": "states_clean",
    }


def _run_primary_replays(
    model: DKUCModel,
    train_data: dict[str, np.ndarray],
    stream: dict[str, np.ndarray],
    config: dict[str, Any],
    fingerprint: str,
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = config["profile"]
    common = {
        "ridge_lambda": float(config["ridge_lambda"]),
        "encoder_fingerprint": fingerprint,
        "oracle_tolerance": float(config["oracle_tolerance"]),
    }
    accumulative = run_accumulative_replay(
        model, train_data, stream, batch_size=int(profile["accumulative_batch_size"]), **common
    )
    window = run_window_replay(
        model, train_data, stream,
        window_size=int(profile["primary_window_size"]),
        batch_size=int(profile["primary_batch_size"]),
        low_dim_condition_limit=float(config["low_dim_condition_limit"]),
        window_condition_limit=float(config["window_condition_limit"]),
        **common,
    )
    selective = run_window_replay(
        model, train_data, stream,
        window_size=int(profile["primary_window_size"]),
        batch_size=int(profile["primary_batch_size"]),
        selective=True,
        epsilon=float(epsilon),
        reject_buffer_policy=str(config["reject_buffer_policy"]),
        low_dim_condition_limit=float(config["low_dim_condition_limit"]),
        window_condition_limit=float(config["window_condition_limit"]),
        **common,
    )
    replays = {
        "dktv_accumulative": accumulative,
        "otvdkl_window": window,
        "otvdkl_selective": selective,
    }
    summaries = {
        "dktv_accumulative": update_summary(accumulative),
        "otvdkl_window": window_update_summary(window),
        "otvdkl_selective": window_update_summary(selective),
    }
    return replays, summaries


def _add_base_ablations(
    replays: dict[str, Any],
    summaries: dict[str, Any],
    model: DKUCModel,
    train_data: dict[str, np.ndarray],
    stream: dict[str, np.ndarray],
    config: dict[str, Any],
    fingerprint: str,
    epsilon: float,
) -> None:
    profile = config["profile"]
    primary = (int(profile["primary_window_size"]), int(profile["primary_batch_size"]))
    common = {
        "ridge_lambda": float(config["ridge_lambda"]),
        "encoder_fingerprint": fingerprint,
        "oracle_tolerance": float(config["oracle_tolerance"]),
        "low_dim_condition_limit": float(config["low_dim_condition_limit"]),
        "window_condition_limit": float(config["window_condition_limit"]),
    }
    for pair in profile["window_ablation"]:
        window_size, batch_size = map(int, pair)
        if (window_size, batch_size) == primary:
            continue
        name = f"otvdkl_window_w{window_size}_b{batch_size}"
        replay = run_window_replay(
            model, train_data, stream, window_size=window_size, batch_size=batch_size, **common
        )
        replays[name], summaries[name] = replay, window_update_summary(replay)
    for factor in profile["epsilon_factors"]:
        factor = float(factor)
        if factor == 1.0:
            continue
        name = f"otvdkl_selective_epsx{factor:g}"
        replay = run_window_replay(
            model, train_data, stream, window_size=primary[0], batch_size=primary[1],
            selective=True, epsilon=epsilon * factor,
            reject_buffer_policy=str(config["reject_buffer_policy"]), **common,
        )
        replays[name], summaries[name] = replay, window_update_summary(replay)
    if profile["include_retain_policy_ablation"]:
        name = "otvdkl_selective_retain_on_reject"
        replay = run_window_replay(
            model, train_data, stream, window_size=primary[0], batch_size=primary[1],
            selective=True, epsilon=epsilon, reject_buffer_policy="retain_on_reject", **common,
        )
        replays[name], summaries[name] = replay, window_update_summary(replay)


def _adaptation_metrics(
    arrays: dict[str, np.ndarray], stages: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    consecutive = int(config["evaluation"]["adaptation_consecutive_steps"])
    final_fraction = float(config["evaluation"]["adaptation_final_fraction"])
    ratio = float(config["evaluation"]["adaptation_tolerance_ratio"])
    methods = [name.removesuffix("_one_step_rmse_by_step") for name in arrays
               if name.endswith("_one_step_rmse_by_step")]
    result: dict[str, Any] = {}
    for method in methods:
        values = np.asarray(arrays[f"{method}_one_step_rmse_by_step"])
        result[method] = {}
        for stage in stages[1:]:
            start, stop = int(stage["start_step"]), int(stage["end_step"])
            tail_start = max(start, stop - max(consecutive, int(round((stop - start) * final_fraction))))
            threshold = ratio * float(np.median(values[tail_start:stop]))
            delay = None
            for cursor in range(start, stop - consecutive + 1):
                if np.all(values[cursor : cursor + consecutive] <= threshold):
                    delay = cursor - start
                    break
            result[method][str(stage["name"])] = {
                "change_step": start,
                "delay_steps": delay,
                "settled": delay is not None,
                "threshold_rmse": threshold,
                "definition": f"first {consecutive} consecutive steps <= {ratio} * final-stage-tail median",
            }
    return result


def _comparisons(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fixed = metrics["one_step"]["fixed_dko"]["total_rmse"]
    accum = metrics["one_step"]["dktv_accumulative"]["total_rmse"]
    for method, values in metrics["one_step"].items():
        value = float(values["total_rmse"])
        result[method] = {
            "fixed_to_method_ratio": float(fixed / max(value, np.finfo(float).eps)),
            "accumulative_to_method_ratio": float(accum / max(value, np.finfo(float).eps)),
            "stage_improvement_vs_fixed": {
                stage: float(metrics["segmented"]["fixed_dko"][stage]["total_rmse"] - stage_value["total_rmse"])
                for stage, stage_value in metrics["segmented"][method].items()
            },
            "stage_improvement_vs_accumulative": {
                stage: float(metrics["segmented"]["dktv_accumulative"][stage]["total_rmse"] - stage_value["total_rmse"])
                for stage, stage_value in metrics["segmented"][method].items()
            },
        }
    return result


def _history_artifacts(result_dir: Path, replays: dict[str, Any]) -> None:
    numeric: dict[str, np.ndarray] = {
        "update_history_schema_version": np.asarray(HISTORY_SCHEMA_VERSION, dtype=np.int64)
    }
    diagnostics: dict[str, np.ndarray] = {}
    lines: list[str] = []
    for method, replay in replays.items():
        for record in replay.update_history:
            lines.append(json.dumps({"method": method, **record}, ensure_ascii=False))
        if not hasattr(replay, "window_size"):
            continue
        records = replay.update_history
        numeric[f"{method}_inserted_sample_ids"] = np.asarray(
            [record["inserted_sample_ids"] for record in records], dtype=np.int64
        )
        numeric[f"{method}_evicted_sample_ids"] = np.asarray(
            [record["evicted_sample_ids"] for record in records], dtype=np.int64
        )
        for field, dtype in (
            ("attempt_index", np.int64), ("model_version", np.int64),
            ("window_version", np.int64), ("time_step", np.int64),
            ("window_start_sample_id", np.int64), ("window_end_sample_id", np.int64),
            ("window_advanced", bool), ("accepted", bool),
            ("current_batch_rmse", np.float64), ("candidate_batch_rmse", np.float64),
            ("update_time_s", np.float64), ("total_update_time_s", np.float64),
            ("recursive_candidate_time_s", np.float64),
            ("direct_refit_oracle_time_s", np.float64), ("fallback_time_s", np.float64),
        ):
            missing = False if dtype is bool else -1 if dtype is np.int64 else np.nan
            numeric[f"{method}_{field}"] = np.asarray(
                [record[field] if record[field] is not None else missing for record in records], dtype=dtype
            )
        diagnostics[f"{method}_rank"] = np.asarray(
            [record["diagnostics"]["rank"] if record["diagnostics"] else -1 for record in records],
            dtype=np.int64,
        )
        for field in ("minimum_singular_value", "condition_number", "regularized_condition_number"):
            diagnostics[f"{method}_{field}"] = np.asarray(
                [record["diagnostics"][field] if record["diagnostics"] else np.nan for record in records]
            )
        diagnostics[f"{method}_recursive_A_max_abs_difference"] = np.asarray(
            [record["recursive_A_max_abs_difference"] if record["recursive_A_max_abs_difference"] is not None else np.nan for record in records]
        )
        diagnostics[f"{method}_window_memory_bytes"] = np.asarray(
            [record["window_memory_bytes"] for record in records], dtype=np.int64
        )
    np.savez_compressed(result_dir / "arrays" / "update_history.npz", **numeric)
    np.savez_compressed(result_dir / "arrays" / "window_diagnostics.npz", **diagnostics)
    (result_dir / "logs" / "update_history.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    save_json(
        result_dir / "arrays" / "update_history_schema.json",
        {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "numeric_storage": "arrays/update_history.npz",
            "diagnostics_storage": "arrays/window_diagnostics.npz",
            "decision_storage": "logs/update_history.jsonl",
            "methods": list(replays),
            "sample_order": "time_major_then_trajectory",
            "pickle_required": False,
        },
    )


def _figures(
    result_dir: Path, metrics: dict[str, Any], arrays: dict[str, np.ndarray], display: dict[str, str]
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    figure, axis = plt.subplots(figsize=(12, 6))
    for method in metrics["one_step"]:
        axis.plot(arrays[f"{method}_one_step_rmse_by_step"], label=display.get(method, method))
    axis.set(xlabel="step", ylabel="physical-state one-step RMSE", title="Plan 03 causal comparison")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    paths.append(result_dir / "figures" / "one_step_rmse_by_step.png")
    figure.savefig(paths[-1], dpi=180)
    plt.close(figure)

    primary = ["fixed_dko", "dktv_accumulative", "otvdkl_window", "otvdkl_selective"]
    horizons = list(metrics["rollout"]["fixed_dko"])
    positions = np.arange(len(horizons), dtype=float)
    width = 0.8 / len(primary)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for index, method in enumerate(primary):
        axis.bar(positions + index * width,
                 [metrics["rollout"][method][h]["total_rmse"] for h in horizons],
                 width=width, label=display.get(method, method))
    axis.set_xticks(positions + width * 1.5, horizons)
    axis.set(xlabel="rollout horizon", ylabel="physical-state RMSE", title="Shared evaluator rollout")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    paths.append(result_dir / "figures" / "rollout_rmse_by_horizon.png")
    figure.savefig(paths[-1], dpi=180)
    plt.close(figure)
    return [path.relative_to(result_dir).as_posix() for path in paths]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(PROJECT_ROOT, args.config)
    output_root = _resolve(PROJECT_ROOT, args.output_root)
    source_output_root = _resolve(PROJECT_ROOT, args.source_output_root)
    config = load_plan03_config(config_path, args.run_type)
    run_id = _make_run_id(args)
    selected_device = _device(args.device)
    git = _git_provenance()
    result_dir = output_root / "results" / "dktv" / "plan_03" / run_id
    for name in ("metrics", "arrays", "figures", "logs"):
        (result_dir / name).mkdir(parents=True, exist_ok=False)
    save_json(result_dir / "config_snapshot.json", config)
    save_json(result_dir / "logs" / "command.json", {
        "entry_module": "prediction.dktv_window_prediction",
        "argv": [sys.executable, "-m", "prediction.dktv_window_prediction", *sys.argv[1:]],
        "cwd": _portable(Path.cwd(), output_root),
    })
    save_json(result_dir / "logs" / "environment.json", {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "platform": platform.platform(),
    })
    log_path = result_dir / "logs" / "run.log"
    logs: list[str] = []
    def log(message: str) -> None:
        logs.append(message)
        log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")
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
    train_data, source_stream = _load_npz(train_path), _load_npz(stream_path)
    steps = min(int(config["profile"]["maximum_steps"]), source_stream["inputs"].shape[1])
    source_stream = _trim_stream(source_stream, steps)
    model = DKUCModel(model_dir, selected_device)
    fingerprint = artifact_fingerprint(model_dir)
    stages = stage_bounds(source_config, steps)
    calibration_stream, calibration_metadata = _independent_calibration_stream(
        source_config, config["epsilon_calibration"]
    )
    epsilon, calibration_values = _calibrate_epsilon(
        model, calibration_stream, batch_size=int(config["profile"]["primary_batch_size"]),
        quantile=float(config["epsilon_calibration"]["quantile"]),
    )
    calibration_path = result_dir / "arrays" / "epsilon_calibration_stream.npz"
    np.savez_compressed(
        calibration_path,
        states=np.asarray(calibration_stream["states"]),
        inputs=np.asarray(calibration_stream["inputs"]),
        t=np.asarray(calibration_stream["t"]),
    )
    save_json(result_dir / "metrics" / "epsilon_calibration.json", {
        **config["epsilon_calibration"], "epsilon": epsilon,
        "batch_rmse_values": calibration_values,
        "batch_count": len(calibration_values),
        "metadata": calibration_metadata,
        "dataset": _file_record(calibration_path, output_root),
    })
    log(f"source={args.plan01_run} seed={source_manifest['seed']} epsilon={epsilon:.9g}")

    scenario_results: dict[str, Any] = {}
    base_replays: dict[str, Any] | None = None
    base_summaries: dict[str, Any] | None = None
    base_metrics: dict[str, Any] | None = None
    base_arrays: dict[str, np.ndarray] | None = None
    scenario_arrays: dict[str, np.ndarray] = {}
    for rate in config["profile"]["rate_multipliers"]:
        for noise in config["profile"]["measurement_noise_std"]:
            name = f"rate{float(rate):g}_noise{float(noise):g}".replace(".", "p")
            stream, scenario = _scenario_stream(
                source_stream, source_config, rate_multiplier=float(rate), noise_std=float(noise),
                steps=steps, trajectory_count=int(source_stream["states"].shape[0]),
                seed=int(source_manifest["seed"]),
            )
            replays, summaries = _run_primary_replays(
                model, train_data, stream, config, fingerprint, epsilon
            )
            if float(rate) == 1.0 and float(noise) == 0.0:
                _add_base_ablations(
                    replays, summaries, model, train_data, stream, config, fingerprint, epsilon
                )
            metrics, arrays = evaluate_methods(
                model, stream, replays,
                rollout_horizons=[h for h in config["evaluation"]["rollout_horizons"] if h <= steps],
                window_stride=int(config["evaluation"]["window_stride"]),
                stage_definitions=stages,
                stage_rollout_horizon=int(config["evaluation"]["stage_rollout_horizon"]),
                truth_states=np.asarray(stream["states_clean"]),
            )
            observed_metrics, observed_arrays = evaluate_methods(
                model, stream, replays,
                rollout_horizons=[h for h in config["evaluation"]["rollout_horizons"] if h <= steps],
                window_stride=int(config["evaluation"]["window_stride"]),
                stage_definitions=stages,
                stage_rollout_horizon=int(config["evaluation"]["stage_rollout_horizon"]),
            )
            adaptation = _adaptation_metrics(arrays, stages, config)
            scenario_results[name] = {
                "scenario": scenario,
                "metrics": metrics,
                "metrics_clean_truth": metrics,
                "metrics_observed_truth": observed_metrics,
                "comparison": _comparisons(metrics),
                "comparison_observed_truth": _comparisons(observed_metrics),
                "adaptation_delay": adaptation,
                "update_summary": summaries,
            }
            for key, values in arrays.items():
                scenario_arrays[f"{name}_clean_{key}"] = values
            for key, values in observed_arrays.items():
                scenario_arrays[f"{name}_observed_{key}"] = values
            log(f"scenario={name} methods={len(metrics['one_step'])} window_rmse={metrics['one_step']['otvdkl_window']['total_rmse']:.9g}")
            if float(rate) == 1.0 and float(noise) == 0.0:
                base_replays, base_summaries = replays, summaries
                base_metrics, base_arrays = metrics, arrays
    if base_replays is None or base_metrics is None or base_arrays is None or base_summaries is None:
        raise RuntimeError("profile omitted required rate=1/noise=0 primary scenario")

    save_json(result_dir / "metrics" / "one_step.json", base_metrics["one_step"])
    save_json(result_dir / "metrics" / "rollout.json", base_metrics["rollout"])
    save_json(result_dir / "metrics" / "segmented.json", base_metrics["segmented"])
    save_json(result_dir / "metrics" / "update_summary.json", base_summaries)
    save_json(result_dir / "metrics" / "scenario_ablation.json", scenario_results)
    save_json(result_dir / "metrics" / "adaptation_delay.json",
              scenario_results["rate1_noise0"]["adaptation_delay"])
    np.savez_compressed(result_dir / "arrays" / "predictions.npz", **base_arrays)
    np.savez_compressed(result_dir / "arrays" / "scenario_predictions.npz", **scenario_arrays)
    _history_artifacts(result_dir, base_replays)
    display = {"fixed_dko": config["display_names"]["fixed_dko"]}
    for method, replay in base_replays.items():
        if method == "dktv_accumulative":
            display[method] = config["display_names"]["dktv_accumulative"].format(batch_size=replay.batch_size)
        elif method.startswith("otvdkl_selective"):
            display[method] = config["display_names"]["otvdkl_selective"].format(
                window_size=replay.window_size, batch_size=replay.batch_size
            )
        else:
            display[method] = config["display_names"]["otvdkl_window"].format(
                window_size=replay.window_size, batch_size=replay.batch_size
            )
    figures = _figures(result_dir, base_metrics, base_arrays, display)

    window_replays = [replay for replay in base_replays.values() if hasattr(replay, "window_size")]
    primary_scenarios_expected = (
        len(config["profile"]["rate_multipliers"]) * len(config["profile"]["measurement_noise_std"])
    )
    acceptance = {
        "source_plan01_engineering_acceptance": bool(source_manifest["acceptance"]["passed"]),
        "four_primary_methods_shared_stream_and_evaluator": all(
            all(name in item["metrics"]["one_step"] for name in config["methods"])
            for item in scenario_results.values()
        ),
        "candidate_matches_direct_refit_with_logged_fallback": all(
            replay.oracle_tolerance_passed for replay in window_replays
        ),
        "fixed_window_memory": all(replay.window_memory_constant for replay in window_replays),
        "window_boundaries_replayable": all(replay.window_boundaries_replayable for replay in window_replays),
        "all_window_candidates_finite": all(replay.all_updates_finite for replay in window_replays),
        "no_pending_snapshots": all(replay.pending_sample_count == 0 for replay in base_replays.values()),
        "both_reject_policies_executed": (
            "otvdkl_selective_retain_on_reject" in base_replays
            and base_replays["otvdkl_selective"].reject_buffer_policy == "discard_on_reject"
        ),
        "noise_and_rate_ablation_complete": len(scenario_results) == primary_scenarios_expected,
        "prediction_arrays_finite": all(np.all(np.isfinite(values)) for values in scenario_arrays.values()),
        "independent_epsilon_calibration": bool(
            calibration_metadata["evaluation_stream_overlap"] is False
            and int(calibration_metadata["seed"]) != int(source_manifest["seed"])
        ),
        "clean_and_observed_noise_truth_saved": all(
            item["scenario"]["clean_truth_preserved"]
            and "metrics_clean_truth" in item
            and "metrics_observed_truth" in item
            for item in scenario_results.values()
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    prediction_gate = bool(
        base_metrics["one_step"]["otvdkl_window"]["total_rmse"]
        < base_metrics["one_step"]["dktv_accumulative"]["total_rmse"]
        and base_metrics["segmented"]["otvdkl_window"]["time_varying"]["total_rmse"]
        < base_metrics["segmented"]["dktv_accumulative"]["time_varying"]["total_rmse"]
        and all(
            base_metrics["rollout"]["otvdkl_window"][horizon]["total_rmse"]
            < base_metrics["rollout"]["dktv_accumulative"][horizon]["total_rmse"]
            for horizon in base_metrics["rollout"]["otvdkl_window"]
        )
    )
    canonical = bool(acceptance["passed"] and not git["dirty"] and source_manifest["canonical"])
    blockers = ([] if acceptance["passed"] else ["acceptance_failed"])
    if git["dirty"]:
        blockers.append("git_worktree_dirty")
    if not source_manifest["canonical"]:
        blockers.append("source_plan01_noncanonical")
    status = "accepted_canonical" if canonical else "accepted_noncanonical" if acceptance["passed"] else "completed_with_failed_acceptance"
    log(f"status={status} acceptance={acceptance['passed']} prediction_gate={prediction_gate}")
    manifest = {
        "manifest_schema_version": 1,
        "plan": "DKTV_PLAN_03_ZHANG_SLIDING_WINDOW",
        "methods": config["methods"],
        "status": status,
        "canonical": canonical,
        "canonical_blockers": blockers,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entry_module": "prediction.dktv_window_prediction",
        "arguments": {
            "config": _portable(config_path, output_root), "run_type": args.run_type,
            "plan01_run": args.plan01_run, "device_requested": args.device,
            "device_used": selected_device, "tag": args.tag,
        },
        "git": git,
        "source_files": _source_provenance(),
        "config": _file_record(config_path, output_root),
        "source_plan01": {
            "run_id": args.plan01_run,
            "manifest": _file_record(source_manifest_path, source_output_root),
            "status": source_manifest["status"], "canonical": source_manifest["canonical"],
            "seed": source_manifest["seed"],
            "model_artifact": _file_record(artifact_manifest_path, source_output_root),
            "training_dataset": _file_record(train_path, source_output_root),
            "stream_dataset": _file_record(stream_path, source_output_root),
            "encoder_fingerprint": fingerprint,
            "scenario_contract": _source_scenario_contract(source_manifest, source_config, artifact_manifest),
        },
        "coordinates": config["coordinates"],
        "epsilon_calibration": {
            **config["epsilon_calibration"],
            "epsilon": epsilon,
            "metadata": calibration_metadata,
            "dataset": _file_record(calibration_path, output_root),
        },
        "stream": {
            "trajectory_count": int(source_stream["states"].shape[0]), "steps": steps,
            "ordering": "time_major_then_trajectory",
            "causal_order": "predict_then_observe_then_update",
            "batch_semantics": "identical_to_plan02",
        },
        "reject_buffer_policy": config["reject_buffer_policy"],
        "ablation_contract": {
            "window_batch_pairs": config["profile"]["window_ablation"],
            "epsilon_factors": config["profile"]["epsilon_factors"],
            "retain_policy_included": config["profile"]["include_retain_policy_ablation"],
            "rate_multipliers": config["profile"]["rate_multipliers"],
            "measurement_noise_std": config["profile"]["measurement_noise_std"],
        },
        "metrics": base_metrics,
        "scenario_ablation": scenario_results,
        "update_summary": base_summaries,
        "figures": figures,
        "result_dir": _portable(result_dir, output_root),
        "result_files": {
            "metrics": _tree_records(result_dir / "metrics"),
            "arrays": _tree_records(result_dir / "arrays"),
            "figures": _tree_records(result_dir / "figures"),
            "logs": _tree_records(result_dir / "logs"),
        },
        "acceptance": acceptance,
        "prediction_gate_for_optional_control": {
            "passed": prediction_gate,
            "definition": "primary window beats accumulative in overall one-step, time_varying stage, and every configured rollout horizon",
        },
        "control_experiment": {
            "executed": False,
            "reason": "Plan 03 Step 06 is optional; no MPC cost or controller was changed.",
        },
        "interpretation_limit": "Engineering prediction evidence only; canonical SDP-MPC and stability claims are out of scope.",
    }
    save_json(result_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = execute(args)
    print(json.dumps({
        "run_id": manifest["run_id"], "status": manifest["status"],
        "canonical": manifest["canonical"], "result_dir": manifest["result_dir"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
