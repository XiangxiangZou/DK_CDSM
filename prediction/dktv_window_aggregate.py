"""Aggregate comparable Plan 03 runs and paired window-vs-accumulative evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from prediction.dktv_accumulative_aggregate import (
    _all_numeric_metrics_finite,
    _git_provenance,
    _record,
    _summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PROVENANCE_FILES = (
    "AGENTS.md",
    "prediction/dktv_window_config.json",
    "prediction/dktv_accumulative_aggregate.py",
    "prediction/dktv_window_aggregate.py",
    "prediction/dktv_window_prediction.py",
    "prediction/dktv/online_model.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate independently generated, comparable Plan 03 runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", required=True)
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


def _source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {name: record["sha256"] for name, record in sorted(manifest["source_files"].items())}


def _aggregate_source_provenance() -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": _hash_file(PROJECT_ROOT / name), "bytes": (PROJECT_ROOT / name).stat().st_size}
        for name in PROVENANCE_FILES
    }


def _verify_source_result_files(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    run_dir = manifest_path.parent
    for category, records in manifest.get("result_files", {}).items():
        for relative, expected in records.items():
            path = run_dir / category / relative
            actual_hash = _hash_file(path) if path.is_file() else None
            actual_bytes = int(path.stat().st_size) if path.is_file() else None
            checks.append(
                {
                    "path": f"{category}/{relative}",
                    "exists": path.is_file(),
                    "sha256_matches": actual_hash == expected["sha256"],
                    "bytes_match": actual_bytes == int(expected["bytes"]),
                }
            )
    return {
        "checked_file_count": len(checks),
        "passed": bool(checks) and all(
            item["exists"] and item["sha256_matches"] and item["bytes_match"] for item in checks
        ),
        "failed_paths": [
            item["path"]
            for item in checks
            if not (item["exists"] and item["sha256_matches"] and item["bytes_match"])
        ],
    }


def _canonical_status(
    *, acceptance_passed: bool, sources_canonical: bool, aggregate_git_dirty: bool
) -> tuple[bool, list[str], str]:
    canonical = bool(acceptance_passed and sources_canonical and not aggregate_git_dirty)
    blockers: list[str] = []
    if not acceptance_passed:
        blockers.append("acceptance_failed")
    if not sources_canonical:
        blockers.append("source_runs_noncanonical")
    if aggregate_git_dirty:
        blockers.append("aggregate_worktree_dirty")
    status = (
        "accepted_canonical"
        if canonical
        else "accepted_noncanonical"
        if acceptance_passed
        else "completed_with_failed_acceptance"
    )
    return canonical, blockers, status


def _paired(baseline: np.ndarray, method: np.ndarray, baseline_name: str) -> dict[str, Any]:
    difference = np.asarray(baseline - method, dtype=np.float64)
    count = int(difference.size)
    mean = float(np.mean(difference))
    std = float(np.std(difference, ddof=1)) if count > 1 else 0.0
    half = (
        float(student_t.ppf(0.975, count - 1) * std / math.sqrt(count)) if count > 1 else 0.0
    )
    return {
        "difference_definition": f"{baseline_name}_rmse - method_rmse",
        "difference_values": difference.tolist(),
        "mean_difference": mean,
        "std_difference": std,
        "difference_confidence_interval_95_student_t": [mean - half, mean + half],
        "win_count": int(np.count_nonzero(difference > 0.0)),
        "tie_count": int(np.count_nonzero(difference == 0.0)),
        "loss_count": int(np.count_nonzero(difference < 0.0)),
        "seed_count": count,
    }


def _contract(
    manifests: list[dict[str, Any]], result_validations: list[dict[str, Any]]
) -> dict[str, Any]:
    reference = manifests[0]
    scenario_names = list(reference["scenario_ablation"])
    checks = {
        "shared_plan03_config": all(item["config"]["sha256"] == reference["config"]["sha256"] for item in manifests),
        "shared_source_files": all(_source_hashes(item) == _source_hashes(reference) for item in manifests),
        "shared_methods": all(item["methods"] == reference["methods"] for item in manifests),
        "shared_base_ablation_methods": all(
            list(item["metrics"]["one_step"]) == list(reference["metrics"]["one_step"])
            for item in manifests
        ),
        "shared_stream_contract": all(item["stream"] == reference["stream"] for item in manifests),
        "shared_scenario_contract": all(list(item["scenario_ablation"]) == scenario_names for item in manifests),
        "shared_ablations": all(item["ablation_contract"] == reference["ablation_contract"] for item in manifests),
        "shared_plan01_scenario": all(
            item["source_plan01"]["scenario_contract"] == reference["source_plan01"]["scenario_contract"]
            for item in manifests
        ),
        "all_metrics_finite": all(
            _all_numeric_metrics_finite(item["metrics"])
            and _all_numeric_metrics_finite(item["scenario_ablation"])
            and _all_numeric_metrics_finite(item["update_summary"])
            for item in manifests
        ),
        "all_source_result_files_verified": all(
            item["passed"] for item in result_validations
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    manifests: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    result_validations: list[dict[str, Any]] = []
    for run in args.runs:
        path = output_root / "results" / "dktv" / "plan_03" / run / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("plan") != "DKTV_PLAN_03_ZHANG_SLIDING_WINDOW":
            raise ValueError(f"not a Plan 03 run: {run}")
        if not payload["acceptance"]["passed"]:
            raise ValueError(f"run failed Plan 03 engineering acceptance: {run}")
        manifests.append(payload)
        source_paths.append(path)
        validation = _verify_source_result_files(payload, path)
        if not validation["passed"]:
            raise ValueError(
                f"source result-file hash verification failed for {run}: "
                f"{', '.join(validation['failed_paths'])}"
            )
        result_validations.append(validation)
    minimum = 10 if args.profile == "final" else 2
    if len(manifests) < minimum:
        raise ValueError(f"{args.profile} profile requires at least {minimum} distinct seeds")
    seeds = [int(item["source_plan01"]["seed"]) for item in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("aggregate runs must use distinct Plan 01 seeds")
    contract = _contract(manifests, result_validations)
    if not contract["passed"]:
        failed = [name for name, passed in contract["checks"].items() if not passed]
        raise ValueError(f"incompatible Plan 03 runs: {', '.join(failed)}")

    primary_methods = manifests[0]["methods"]
    methods = list(manifests[0]["metrics"]["one_step"])
    horizons = list(manifests[0]["metrics"]["rollout"]["fixed_dko"])
    stages = list(manifests[0]["metrics"]["segmented"]["fixed_dko"])
    one_values = {
        method: np.asarray([item["metrics"]["one_step"][method]["total_rmse"] for item in manifests])
        for method in methods
    }
    rollout_values = {
        method: {
            horizon: np.asarray([item["metrics"]["rollout"][method][horizon]["total_rmse"] for item in manifests])
            for horizon in horizons
        }
        for method in methods
    }
    stage_values = {
        method: {
            stage: np.asarray([item["metrics"]["segmented"][method][stage]["total_rmse"] for item in manifests])
            for stage in stages
        }
        for method in methods
    }
    summaries = {
        "one_step": {method: _summary(values) for method, values in one_values.items()},
        "rollout": {
            method: {horizon: _summary(values) for horizon, values in by_horizon.items()}
            for method, by_horizon in rollout_values.items()
        },
        "segmented": {
            method: {stage: _summary(values) for stage, values in by_stage.items()}
            for method, by_stage in stage_values.items()
        },
    }
    paired = {
        method: {
            "one_step": _paired(one_values["dktv_accumulative"], one_values[method], "dktv_accumulative"),
            "rollout": {
                horizon: _paired(
                    rollout_values["dktv_accumulative"][horizon], rollout_values[method][horizon],
                    "dktv_accumulative",
                )
                for horizon in horizons
            },
            "segmented": {
                stage: _paired(
                    stage_values["dktv_accumulative"][stage], stage_values[method][stage],
                    "dktv_accumulative",
                )
                for stage in stages
            },
        }
        for method in ("otvdkl_window", "otvdkl_selective")
    }
    scenarios: dict[str, Any] = {}
    scenario_arrays: dict[str, np.ndarray] = {}
    for scenario in manifests[0]["scenario_ablation"]:
        scenarios[scenario] = {}
        for truth_name, metric_field in (
            ("clean_truth", "metrics_clean_truth"),
            ("observed_truth", "metrics_observed_truth"),
        ):
            one_step_summary: dict[str, Any] = {}
            rollout_summary: dict[str, Any] = {}
            for method in primary_methods:
                one_values_scenario = np.asarray([
                    item["scenario_ablation"][scenario][metric_field]["one_step"][method]["total_rmse"]
                    for item in manifests
                ])
                one_step_summary[method] = _summary(one_values_scenario)
                scenario_arrays[
                    f"{scenario}_{truth_name}_{method}_one_step_rmse"
                ] = one_values_scenario
                rollout_summary[method] = {}
                for horizon in horizons:
                    rollout_scenario = np.asarray([
                        item["scenario_ablation"][scenario][metric_field]["rollout"][method][horizon]["total_rmse"]
                        for item in manifests
                    ])
                    rollout_summary[method][horizon] = _summary(rollout_scenario)
                    scenario_arrays[
                        f"{scenario}_{truth_name}_{method}_rollout_{horizon}_rmse"
                    ] = rollout_scenario
            scenarios[scenario][truth_name] = {
                "one_step": one_step_summary,
                "rollout": rollout_summary,
                "paired_window_vs_accumulative": {
                    "one_step": _paired(
                        scenario_arrays[f"{scenario}_{truth_name}_dktv_accumulative_one_step_rmse"],
                        scenario_arrays[f"{scenario}_{truth_name}_otvdkl_window_one_step_rmse"],
                        "dktv_accumulative",
                    ),
                    "rollout": {
                        horizon: _paired(
                            scenario_arrays[f"{scenario}_{truth_name}_dktv_accumulative_rollout_{horizon}_rmse"],
                            scenario_arrays[f"{scenario}_{truth_name}_otvdkl_window_rollout_{horizon}_rmse"],
                            "dktv_accumulative",
                        )
                        for horizon in horizons
                    },
                },
            }
    timing_fields = (
        "mean_update_time_s",
        "mean_recursive_candidate_time_s",
        "mean_direct_refit_oracle_time_s",
        "mean_fallback_time_s",
        "direct_refit_fallback_count",
    )
    timing_methods = ("otvdkl_window", "otvdkl_selective")
    timing_arrays: dict[str, np.ndarray] = {}
    update_timing: dict[str, Any] = {}
    for method in timing_methods:
        update_timing[method] = {
            "timing_contract": manifests[0]["update_summary"][method]["timing_contract"]
        }
        for field in timing_fields:
            values = np.asarray(
                [item["update_summary"][method][field] for item in manifests],
                dtype=np.float64,
            )
            timing_arrays[f"{method}_{field}"] = values
            update_timing[method][field] = _summary(values)
        ratio = (
            timing_arrays[f"{method}_mean_recursive_candidate_time_s"]
            / timing_arrays[f"{method}_mean_direct_refit_oracle_time_s"]
        )
        timing_arrays[f"{method}_recursive_to_direct_oracle_time_ratio"] = ratio
        update_timing[method]["recursive_to_direct_oracle_time_ratio"] = _summary(ratio)
    metrics = {
        "seed_count": len(seeds), "seeds": seeds, **summaries,
        "paired_vs_accumulative": paired,
        "scenario_clean_and_observed_truth": scenarios,
        "update_timing": update_timing,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{stamp}_plan03_aggregate_{args.profile}_{args.tag}"
    result_dir = output_root / "results" / "dktv" / "plan_03" / run_id
    for name in ("metrics", "arrays", "figures", "logs"):
        (result_dir / name).mkdir(parents=True, exist_ok=False)
    metrics_path = result_dir / "metrics" / "multiseed_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    arrays: dict[str, np.ndarray] = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        **scenario_arrays,
        **timing_arrays,
    }
    for method, values in one_values.items():
        arrays[f"{method}_one_step_rmse"] = values
    for method in ("otvdkl_window", "otvdkl_selective"):
        arrays[f"{method}_one_step_paired_vs_accumulative"] = np.asarray(
            paired[method]["one_step"]["difference_values"]
        )
    arrays_path = result_dir / "arrays" / "multiseed_metrics.npz"
    np.savez_compressed(arrays_path, **arrays)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(primary_methods))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions, [metrics["one_step"][method]["mean"] for method in primary_methods],
             yerr=[metrics["one_step"][method]["std"] for method in primary_methods], capsize=4)
    axis.set_xticks(positions, primary_methods, rotation=15)
    axis.set(ylabel="one-step RMSE, mean ± sample std", title=f"Plan 03 final comparison (n={len(seeds)})")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    overview = result_dir / "figures" / "multiseed_one_step_rmse.png"
    figure.savefig(overview, dpi=180)
    plt.close(figure)

    compared = ["otvdkl_window", "otvdkl_selective"]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for index, method in enumerate(compared):
        values = np.asarray(paired[method]["one_step"]["difference_values"])
        offsets = np.linspace(-0.08, 0.08, values.size)
        axis.scatter(np.full(values.shape, index) + offsets, values, alpha=0.7)
        mean = paired[method]["one_step"]["mean_difference"]
        low, high = paired[method]["one_step"]["difference_confidence_interval_95_student_t"]
        axis.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="black", capsize=6)
    axis.axhline(0.0, color="gray", linewidth=1)
    axis.set_xticks(range(len(compared)), compared)
    axis.set(ylabel="accumulative RMSE - method RMSE", title="Paired one-step differences and 95% Student-t CI")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    paired_figure = result_dir / "figures" / "paired_one_step_difference_ci.png"
    figure.savefig(paired_figure, dpi=180)
    plt.close(figure)

    aggregate_git = _git_provenance()
    acceptance = {
        "minimum_seed_count": minimum,
        "seed_count": len(seeds),
        "distinct_seeds": len(set(seeds)) == len(seeds),
        "comparison_contract_passed": contract["passed"],
        "all_metrics_finite": contract["checks"]["all_metrics_finite"],
        "all_source_result_files_verified": contract["checks"][
            "all_source_result_files_verified"
        ],
    }
    acceptance["passed"] = all(acceptance.values())
    canonical, blockers, status = _canonical_status(
        acceptance_passed=bool(acceptance["passed"]),
        sources_canonical=all(item["canonical"] for item in manifests),
        aggregate_git_dirty=bool(aggregate_git["dirty"]),
    )
    manifest = {
        "manifest_schema_version": 1,
        "plan": "DKTV_PLAN_03_ZHANG_SLIDING_WINDOW_AGGREGATE",
        "status": status,
        "canonical": canonical,
        "canonical_blockers": blockers,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entry_module": "prediction.dktv_window_aggregate",
        "profile": args.profile,
        "git": aggregate_git,
        "source_files": _aggregate_source_provenance(),
        "source_runs": [
            {"run_id": item["run_id"], "seed": item["source_plan01"]["seed"],
             "manifest": _record(path, output_root),
             "result_file_validation": validation}
            for item, path, validation in zip(manifests, source_paths, result_validations)
        ],
        "comparison_contract": contract,
        "metrics": metrics,
        "artifacts": {
            "metrics": _record(metrics_path, output_root),
            "arrays": _record(arrays_path, output_root),
            "overview_figure": _record(overview, output_root),
            "paired_figure": _record(paired_figure, output_root),
        },
        "acceptance": acceptance,
        "interpretation_limit": "Paired 10-seed engineering evidence; noncanonical sources remain noncanonical.",
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = execute(args)
    print(json.dumps({
        "run_id": manifest["run_id"], "status": manifest["status"],
        "seed_count": manifest["acceptance"]["seed_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
