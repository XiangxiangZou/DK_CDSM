"""Aggregate comparable Plan 02 runs without rerunning online identification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
AGGREGATE_MANIFEST_SCHEMA_VERSION = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate independently generated, comparable DKTV Plan 02 runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Plan 02 run ids")
    parser.add_argument("--profile", choices=("development", "final"), default="development")
    parser.add_argument("--tag", default="multiseed")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        if path.resolve().is_relative_to(PROJECT_ROOT.resolve())
        else f"${{OUTPUT_ROOT}}/{path.resolve().relative_to(root.resolve()).as_posix()}",
        "sha256": _hash_file(path),
        "bytes": int(path.stat().st_size),
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


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _paired_summary(fixed: np.ndarray, method: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(fixed - method, dtype=np.float64)
    ratio = np.asarray(fixed / np.maximum(method, np.finfo(np.float64).eps), dtype=np.float64)
    count = int(difference.size)
    mean_difference = float(np.mean(difference))
    std_difference = float(np.std(difference, ddof=1)) if count > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, df=count - 1) * std_difference / math.sqrt(count))
        if count > 1
        else 0.0
    )
    return {
        "difference_definition": "fixed_dko_rmse - method_rmse",
        "ratio_definition": "fixed_dko_rmse / method_rmse",
        "difference_values": difference.tolist(),
        "ratio_values": ratio.tolist(),
        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "difference_confidence_interval_95_student_t": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
        "mean_ratio": float(np.mean(ratio)),
        "std_ratio": float(np.std(ratio, ddof=1)) if count > 1 else 0.0,
        "win_count": int(np.count_nonzero(difference > 0.0)),
        "tie_count": int(np.count_nonzero(difference == 0.0)),
        "seed_count": count,
    }


def _all_numeric_metrics_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_numeric_metrics_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_metrics_finite(item) for item in value)
    return True


def _source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        name: record["sha256"] for name, record in sorted(manifest["source_files"].items())
    }


def _stage_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    fixed = manifest["metrics"]["segmented"]["fixed_dko"]
    return {
        stage: {
            "start_step": values["start_step"],
            "end_step": values["end_step"],
            "disturbance_scale": values["disturbance_scale"],
            "rollout_horizon": values["rollout_horizon"],
        }
        for stage, values in fixed.items()
    }


def _comparison_contract(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    reference = manifests[0]
    methods = list(reference["metrics"]["one_step"])
    reference_values = {
        "plan02_config_sha256": reference["config"]["sha256"],
        "source_file_hashes": _source_hashes(reference),
        "batch_sizes": reference["batch_sizes"],
        "coordinates": reference["coordinates"],
        "stream": {
            key: reference["stream"][key]
            for key in (
                "trajectory_count",
                "steps",
                "online_snapshot_count",
                "ordering",
                "causal_order",
                "physical_semantics",
                "snapshots_per_simulation_step",
                "batch_size_to_simulation_steps",
            )
        },
        "methods": methods,
        "rollout_horizons": list(reference["metrics"]["rollout"]["fixed_dko"]),
        "stages": _stage_contract(reference),
        "source_plan01_scenario": reference["source_plan01"]["scenario_contract"],
        "update_history_schema_version": reference["update_history_contract"]["schema_version"],
    }
    checks = {
        "shared_plan02_config_hash": all(
            item["config"]["sha256"] == reference_values["plan02_config_sha256"]
            for item in manifests
        ),
        "shared_source_file_hashes": all(
            _source_hashes(item) == reference_values["source_file_hashes"] for item in manifests
        ),
        "shared_batch_sizes": all(
            item["batch_sizes"] == reference_values["batch_sizes"] for item in manifests
        ),
        "shared_coordinates": all(
            item["coordinates"] == reference_values["coordinates"] for item in manifests
        ),
        "shared_stream_contract": all(
            all(item["stream"].get(key) == value for key, value in reference_values["stream"].items())
            for item in manifests
        ),
        "shared_methods": all(
            list(item["metrics"]["one_step"]) == methods for item in manifests
        ),
        "shared_rollout_horizons": all(
            list(item["metrics"]["rollout"]["fixed_dko"])
            == reference_values["rollout_horizons"]
            for item in manifests
        ),
        "shared_stage_definitions": all(
            _stage_contract(item) == reference_values["stages"] for item in manifests
        ),
        "shared_source_plan01_schema_and_scenario": all(
            item["source_plan01"]["scenario_contract"]
            == reference_values["source_plan01_scenario"]
            for item in manifests
        ),
        "shared_update_history_schema": all(
            item["update_history_contract"]["schema_version"]
            == reference_values["update_history_schema_version"]
            for item in manifests
        ),
        "all_metrics_finite": all(
            _all_numeric_metrics_finite(item["metrics"])
            and _all_numeric_metrics_finite(item["update_summary"])
            for item in manifests
        ),
    }
    return {"reference": reference_values, "checks": checks, "passed": all(checks.values())}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    manifests: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for run_id in args.runs:
        path = output_root / "results" / "dktv" / "plan_02" / run_id / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("plan") != "DKTV_PLAN_02_HAO_ACCUMULATIVE":
            raise ValueError(f"not a Plan 02 manifest: {run_id}")
        if not payload["acceptance"]["passed"]:
            raise ValueError(f"Plan 02 run did not pass engineering acceptance: {run_id}")
        manifests.append(payload)
        source_paths.append(path)
    minimum_seed_count = 10 if args.profile == "final" else 2
    if len(manifests) < minimum_seed_count:
        raise ValueError(
            f"{args.profile} profile requires at least {minimum_seed_count} distinct seeds"
        )
    seeds = [int(manifest["source_plan01"]["seed"]) for manifest in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("aggregate runs must use distinct Plan 01 seeds")
    comparison_contract = _comparison_contract(manifests)
    if not comparison_contract["passed"]:
        failed = [name for name, passed in comparison_contract["checks"].items() if not passed]
        raise ValueError(f"incompatible aggregate runs: {', '.join(failed)}")

    method_names = list(manifests[0]["metrics"]["one_step"])
    display_names = manifests[0]["display_names"]
    one_step_values = {
        method: np.asarray(
            [manifest["metrics"]["one_step"][method]["total_rmse"] for manifest in manifests],
            dtype=np.float64,
        )
        for method in method_names
    }
    one_step = {method: _summary(values) for method, values in one_step_values.items()}
    horizons = list(manifests[0]["metrics"]["rollout"]["fixed_dko"])
    rollout_values = {
        method: {
            horizon: np.asarray(
                [
                    manifest["metrics"]["rollout"][method][horizon]["total_rmse"]
                    for manifest in manifests
                ],
                dtype=np.float64,
            )
            for horizon in horizons
        }
        for method in method_names
    }
    rollout = {
        method: {horizon: _summary(values) for horizon, values in by_horizon.items()}
        for method, by_horizon in rollout_values.items()
    }
    stages = list(manifests[0]["metrics"]["segmented"]["fixed_dko"])
    segmented_values = {
        method: {
            stage: np.asarray(
                [
                    manifest["metrics"]["segmented"][method][stage]["total_rmse"]
                    for manifest in manifests
                ],
                dtype=np.float64,
            )
            for stage in stages
        }
        for method in method_names
    }
    segmented = {
        method: {stage: _summary(values) for stage, values in by_stage.items()}
        for method, by_stage in segmented_values.items()
    }
    online_methods = [method for method in method_names if method != "fixed_dko"]
    update_oracle = {
        method: {
            "maximum_A_difference": _summary(
                np.asarray(
                    [
                        manifest["update_summary"][method]["maximum_oracle_A_difference"]
                        for manifest in manifests
                    ]
                )
            ),
            "mean_recursive_update_time_s": _summary(
                np.asarray(
                    [
                        manifest["update_summary"][method]["mean_recursive_update_time_s"]
                        for manifest in manifests
                    ]
                )
            ),
            "mean_direct_refit_oracle_time_s": _summary(
                np.asarray(
                    [
                        manifest["update_summary"][method]["mean_direct_refit_oracle_time_s"]
                        for manifest in manifests
                    ]
                )
            ),
        }
        for method in online_methods
    }
    paired = {
        method: {
            "one_step": _paired_summary(one_step_values["fixed_dko"], one_step_values[method]),
            "rollout": {
                horizon: _paired_summary(
                    rollout_values["fixed_dko"][horizon], rollout_values[method][horizon]
                )
                for horizon in horizons
            },
            "segmented": {
                stage: _paired_summary(
                    segmented_values["fixed_dko"][stage], segmented_values[method][stage]
                )
                for stage in stages
            },
        }
        for method in online_methods
    }
    metrics = {
        "seed_count": len(seeds),
        "seeds": seeds,
        "one_step": one_step,
        "rollout": rollout,
        "segmented": segmented,
        "update_oracle": update_oracle,
        "paired_vs_fixed": paired,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{stamp}_plan02_aggregate_{args.profile}_{args.tag}"
    result_dir = output_root / "results" / "dktv" / "plan_02" / run_id
    for name in ("metrics", "arrays", "figures", "logs"):
        (result_dir / name).mkdir(parents=True, exist_ok=False)
    metrics_path = result_dir / "metrics" / "multiseed_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    arrays: dict[str, np.ndarray] = {"seeds": np.asarray(seeds, dtype=np.int64)}
    for method in method_names:
        arrays[f"{method}_one_step_rmse"] = one_step_values[method]
        for horizon in horizons:
            arrays[f"{method}_rollout_{horizon}_rmse"] = rollout_values[method][horizon]
        for stage in stages:
            arrays[f"{method}_stage_{stage}_rmse"] = segmented_values[method][stage]
    for method in online_methods:
        arrays[f"{method}_one_step_paired_difference"] = np.asarray(
            paired[method]["one_step"]["difference_values"]
        )
        arrays[f"{method}_one_step_paired_ratio"] = np.asarray(
            paired[method]["one_step"]["ratio_values"]
        )
        for horizon in horizons:
            arrays[f"{method}_rollout_{horizon}_paired_difference"] = np.asarray(
                paired[method]["rollout"][horizon]["difference_values"]
            )
            arrays[f"{method}_rollout_{horizon}_paired_ratio"] = np.asarray(
                paired[method]["rollout"][horizon]["ratio_values"]
            )
        for stage in stages:
            arrays[f"{method}_stage_{stage}_paired_difference"] = np.asarray(
                paired[method]["segmented"][stage]["difference_values"]
            )
            arrays[f"{method}_stage_{stage}_paired_ratio"] = np.asarray(
                paired[method]["segmented"][stage]["ratio_values"]
            )
    arrays_path = result_dir / "arrays" / "multiseed_metrics.npz"
    np.savez_compressed(arrays_path, **arrays)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(method_names))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        positions,
        [one_step[method]["mean"] for method in method_names],
        yerr=[one_step[method]["std"] for method in method_names],
        capsize=4,
    )
    axis.set_xticks(
        positions, [display_names.get(method, method) for method in method_names], rotation=16
    )
    axis.set_ylabel("one-step RMSE, mean ± sample std")
    axis.set_title(f"Plan 02 {args.profile} comparison (n={len(seeds)})")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure_path = result_dir / "figures" / "multiseed_one_step_rmse.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    paired_positions = np.arange(len(online_methods), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, method in enumerate(online_methods):
        values = np.asarray(paired[method]["one_step"]["difference_values"])
        offsets = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.zeros(1)
        axis.scatter(
            np.full(values.shape, paired_positions[index]) + offsets,
            values,
            alpha=0.7,
            s=28,
            label="per-seed paired difference" if index == 0 else None,
        )
        mean = float(paired[method]["one_step"]["mean_difference"])
        interval = paired[method]["one_step"]["difference_confidence_interval_95_student_t"]
        axis.errorbar(
            paired_positions[index],
            mean,
            yerr=np.asarray([[mean - interval[0]], [interval[1] - mean]]),
            fmt="D",
            color="black",
            capsize=6,
            markersize=6,
            label="mean and 95% Student-t CI" if index == 0 else None,
        )
    axis.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.2, label="no difference")
    axis.set_xticks(
        paired_positions,
        [f"b={manifests[0]['update_summary'][method]['batch_size']}" for method in online_methods],
    )
    axis.set_ylabel("paired one-step RMSE difference (fixed DKO - method)")
    axis.set_title(f"Plan 02 paired one-step improvements (n={len(seeds)})")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    paired_figure_path = result_dir / "figures" / "paired_one_step_difference_ci.png"
    figure.savefig(paired_figure_path, dpi=180)
    plt.close(figure)

    command = {
        "entry_module": "experiments.dktv.aggregate_plan_02",
        "argv": [sys.executable, "-m", "experiments.dktv.aggregate_plan_02", *sys.argv[1:]],
    }
    command_path = result_dir / "logs" / "command.json"
    command_path.write_text(json.dumps(command, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    git = _git_provenance()
    source_canonical = all(manifest["canonical"] for manifest in manifests)
    canonical = bool(not git["dirty"] and source_canonical)
    blockers = []
    if git["dirty"]:
        blockers.append("git_worktree_dirty")
    if not source_canonical:
        blockers.append("source_plan02_runs_noncanonical")
    interpretation_limit = (
        f"Final profile includes {len(seeds)} comparable seeds and satisfies the Plan 02 minimum; "
        "canonical paper claims still require canonical source runs."
        if args.profile == "final"
        else f"Development profile includes {len(seeds)} comparable seeds; final profile requires "
        "at least 10 distinct seeds."
    )
    manifest = {
        "manifest_schema_version": AGGREGATE_MANIFEST_SCHEMA_VERSION,
        "plan": "DKTV_PLAN_02_HAO_ACCUMULATIVE_MULTI_SEED",
        "run_id": run_id,
        "profile": args.profile,
        "minimum_seed_count": minimum_seed_count,
        "status": "accepted_canonical" if canonical else "accepted_noncanonical",
        "canonical": canonical,
        "canonical_blockers": blockers,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git,
        "aggregate_source": _record(Path(__file__), output_root),
        "display_names": display_names,
        "comparison_contract": comparison_contract,
        "source_runs": [
            {
                "run_id": source["run_id"],
                "seed": source["source_plan01"]["seed"],
                "manifest": _record(path, output_root),
            }
            for source, path in zip(manifests, source_paths)
        ],
        "seed_count": len(seeds),
        "seeds": seeds,
        "metrics": metrics,
        "artifacts": {
            "metrics": _record(metrics_path, output_root),
            "arrays": _record(arrays_path, output_root),
            "figure": _record(figure_path, output_root),
            "paired_figure": _record(paired_figure_path, output_root),
            "command": _record(command_path, output_root),
        },
        "acceptance": {
            "distinct_seeds": True,
            "minimum_seed_count_satisfied": len(seeds) >= minimum_seed_count,
            "all_source_runs_passed": True,
            "comparison_contract_passed": comparison_contract["passed"],
            "all_metrics_finite": comparison_contract["checks"]["all_metrics_finite"],
            "passed": True,
        },
        "interpretation_limit": interpretation_limit,
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = execute(build_parser().parse_args())
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "profile": manifest["profile"],
                "seed_count": manifest["seed_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
