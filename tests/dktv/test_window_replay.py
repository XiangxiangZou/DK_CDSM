"""Replay invariants independent of the learned Plan 01 artifact."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from prediction.dktv.least_squares import direct_refit
from prediction.dktv.window_update import SlidingWindowKoopmanUpdater
from prediction.dktv.online_model import artifact_fingerprint, run_window_replay
from prediction.dktv_window_aggregate import (
    _canonical_status,
    _verify_source_result_files,
)
from prediction.dktv_window_prediction import _scenario_stream

from .test_window_update import synthetic


ROOT = Path(__file__).resolve().parents[2]
PLAN01_SMOKE_RUN = "20260825_145551_plan01_smoke_baseline_reviewed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_long_replay_has_constant_memory_and_exact_boundaries() -> None:
    window_size, batch_size = 50, 5
    z, target, u = synthetic(samples=550)
    initial = direct_refit(z[:window_size], target[:window_size], u[:window_size], ridge_lambda=1e-3)
    updater = SlidingWindowKoopmanUpdater(
        z[:window_size], target[:window_size], u[:window_size],
        A0=initial.A, B0=initial.B, batch_size=batch_size, ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1", sample_ids=np.arange(window_size),
    )
    memory_values = [updater.memory_bytes]
    inserted: list[int] = []
    evicted: list[int] = []
    for start in range(window_size, z.shape[0], batch_size):
        record, _ = updater.update(
            z[start : start + batch_size], target[start : start + batch_size],
            u[start : start + batch_size], sample_ids=np.arange(start, start + batch_size),
            encoder_fingerprint="encoder-v1",
        )
        assert record.status == "accepted"
        inserted.extend(record.inserted_sample_ids)
        evicted.extend(record.evicted_sample_ids)
        memory_values.append(updater.memory_bytes)
    assert inserted == list(range(window_size, z.shape[0]))
    assert evicted == list(range(0, z.shape[0] - window_size))
    assert updater.sample_ids.tolist() == list(range(z.shape[0] - window_size, z.shape[0]))
    assert len(set(memory_values)) == 1
    assert updater.model_version == (z.shape[0] - window_size) // batch_size


def test_real_artifact_window_replay_is_causal_and_bounded() -> None:
    from prediction.dkuc_prediction import DKUCModel

    model_dir = ROOT / "outputs" / "models" / "dktv" / PLAN01_SMOKE_RUN
    stream_path = ROOT / "outputs" / "data" / "processed" / PLAN01_SMOKE_RUN / "validation_stream.npz"
    if not model_dir.is_dir() or not stream_path.is_file():
        import pytest
        pytest.skip("reviewed Plan 01 smoke artifact is unavailable")
    with np.load(model_dir / "dataset_train.npz", allow_pickle=False) as payload:
        train = {name: payload[name] for name in payload.files}
    with np.load(stream_path, allow_pickle=False) as payload:
        stream = {name: payload[name] for name in payload.files}
    stream["states"] = stream["states"][:, :21]
    stream["inputs"] = stream["inputs"][:, :20]
    model = DKUCModel(model_dir, "cpu")
    replay = run_window_replay(
        model, train, stream, window_size=50, batch_size=10,
        ridge_lambda=1e-3, encoder_fingerprint=artifact_fingerprint(model_dir),
        oracle_tolerance=1e-8,
    )
    assert replay.oracle_tolerance_passed
    assert replay.window_memory_constant
    assert replay.window_boundaries_replayable
    assert replay.pending_sample_count == 0
    assert replay.model_version_by_step[0] == 0
    assert replay.model_version_by_step[4] == 0
    assert replay.model_version_by_step[5] == 1


def test_plan03_smoke_cli_contract(tmp_path: Path) -> None:
    source = ROOT / "outputs" / "results" / "dktv" / "plan_01" / PLAN01_SMOKE_RUN / "manifest.json"
    if not source.is_file():
        import pytest
        pytest.skip("reviewed Plan 01 smoke artifact is unavailable")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    command = [
        sys.executable, "-m", "prediction.dktv_window_prediction", "--run-type", "smoke",
        "--plan01-run", PLAN01_SMOKE_RUN, "--device", "cpu",
        "--run-id", "pytest_plan03_e2e", "--output-root", str(tmp_path),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, check=False, capture_output=True,
        text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = tmp_path / "results" / "dktv" / "plan_03" / "pytest_plan03_e2e"
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["acceptance"]["passed"]
    assert manifest["methods"] == [
        "fixed_dko", "dktv_accumulative", "otvdkl_window", "otvdkl_selective"
    ]
    assert manifest["reject_buffer_policy"] == "discard_on_reject"
    assert manifest["epsilon_calibration"]["seed"] == 20260824
    assert manifest["epsilon_calibration"]["seed"] != manifest["source_plan01"]["seed"]
    assert manifest["epsilon_calibration"]["metadata"]["evaluation_stream_overlap"] is False
    calibration_dataset = result / "arrays" / "epsilon_calibration_stream.npz"
    assert calibration_dataset.is_file()
    assert _sha256(calibration_dataset) == manifest["epsilon_calibration"]["dataset"]["sha256"]
    base_scenario = manifest["scenario_ablation"]["rate1_noise0"]
    assert "metrics_clean_truth" in base_scenario
    assert "metrics_observed_truth" in base_scenario
    assert (result / "arrays" / "predictions.npz").is_file()
    assert (result / "arrays" / "update_history.npz").is_file()
    assert (result / "arrays" / "window_diagnostics.npz").is_file()
    assert (result / "metrics" / "adaptation_delay.json").is_file()
    with np.load(result / "arrays" / "update_history.npz", allow_pickle=False) as payload:
        assert int(payload["update_history_schema_version"]) == 1
        assert "otvdkl_window_inserted_sample_ids" in payload.files
        assert "otvdkl_window_recursive_candidate_time_s" in payload.files
        assert "otvdkl_window_direct_refit_oracle_time_s" in payload.files
        assert "otvdkl_window_fallback_time_s" in payload.files
    for category, records in manifest["result_files"].items():
        for relative, expected in records.items():
            path = result / category / relative
            assert path.stat().st_size == expected["bytes"]
            assert _sha256(path) == expected["sha256"]


def test_source_result_hash_verification_detects_mutation(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    metric_dir = result_dir / "metrics"
    metric_dir.mkdir(parents=True)
    metric = metric_dir / "metric.json"
    metric.write_text('{"value": 1}\n', encoding="utf-8")
    manifest_path = result_dir / "manifest.json"
    manifest = {
        "result_files": {
            "metrics": {
                "metric.json": {
                    "sha256": _sha256(metric),
                    "bytes": metric.stat().st_size,
                }
            }
        }
    }
    assert _verify_source_result_files(manifest, manifest_path)["passed"]
    metric.write_text('{"value": 2}\n', encoding="utf-8")
    validation = _verify_source_result_files(manifest, manifest_path)
    assert not validation["passed"]
    assert validation["failed_paths"] == ["metrics/metric.json"]


def test_aggregate_canonical_status_includes_aggregate_git_state() -> None:
    assert _canonical_status(
        acceptance_passed=True, sources_canonical=True, aggregate_git_dirty=False
    ) == (True, [], "accepted_canonical")
    canonical, blockers, status = _canonical_status(
        acceptance_passed=True, sources_canonical=True, aggregate_git_dirty=True
    )
    assert not canonical
    assert blockers == ["aggregate_worktree_dirty"]
    assert status == "accepted_noncanonical"


def test_noise_scenario_preserves_clean_truth_and_updates_observed_state() -> None:
    source = {
        "states": np.zeros((2, 6, 4), dtype=np.float64),
        "inputs": np.zeros((2, 5, 2), dtype=np.float64),
        "t": np.broadcast_to(np.arange(6, dtype=np.float64), (2, 6)).copy(),
    }
    scenario, metadata = _scenario_stream(
        source,
        {},
        rate_multiplier=1.0,
        noise_std=1e-3,
        steps=5,
        trajectory_count=2,
        seed=11,
    )
    assert np.array_equal(scenario["states_clean"], source["states"])
    assert not np.array_equal(scenario["states"], scenario["states_clean"])
    assert np.all(np.isfinite(scenario["states"]))
    assert metadata["online_update_state"] == "states_observed"
    assert metadata["primary_evaluation_truth"] == "states_clean"
