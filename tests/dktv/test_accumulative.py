"""Plan 02 accumulative least-squares, replay, and CLI checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from prediction.dktv.accumulative_update import AccumulativeKoopmanUpdater
from prediction.dktv.least_squares import (
    build_regressor,
    direct_refit,
    solve_statistics,
    sufficient_statistics,
)
from prediction.dktv.online_model import (
    artifact_fingerprint,
    run_accumulative_replay,
    update_summary,
)
from prediction.dktv_accumulative_aggregate import _comparison_contract, _paired_summary


ROOT = Path(__file__).resolve().parents[2]
PLAN01_SMOKE_RUN = "20260825_145551_plan01_smoke_baseline_reviewed"


def _synthetic(seed: int = 7, samples: int = 160):
    rng = np.random.default_rng(seed)
    latent_dim = 6
    input_dim = 2
    z = rng.normal(size=(samples, latent_dim))
    z[:, -1] = 1.0
    u = rng.normal(size=(samples, input_dim))
    A = 0.7 * np.eye(latent_dim)
    A[-1] = 0.0
    A[-1, -1] = 1.0
    B = rng.normal(scale=0.1, size=(latent_dim, input_dim))
    B[-1] = 0.0
    target = z @ A.T + u @ B.T + rng.normal(scale=1e-5, size=z.shape)
    target[:, -1] = 1.0
    return z, target, u


def test_sufficient_statistics_match_direct_refit() -> None:
    z, target, u = _synthetic()
    direct = direct_refit(z, target, u, ridge_lambda=1e-3, affine_constant=True)
    regressor = build_regressor(z, u)
    gram, cross = sufficient_statistics(regressor, target)
    statistics = solve_statistics(
        gram,
        cross,
        sample_count=z.shape[0],
        latent_dim=z.shape[1],
        input_dim=u.shape[1],
        ridge_lambda=1e-3,
        affine_constant=True,
    )
    assert np.allclose(statistics.A, direct.A, atol=1e-11, rtol=1e-11)
    assert np.allclose(statistics.B, direct.B, atol=1e-11, rtol=1e-11)
    assert statistics.A[-1, -1] == 1.0
    assert np.count_nonzero(statistics.B[-1]) == 0


@pytest.mark.parametrize("batch_size", [5, 10, 20])
def test_accumulative_updates_match_full_history_refit(batch_size: int) -> None:
    z, target, u = _synthetic(samples=180)
    initial = 60
    first = direct_refit(
        z[:initial], target[:initial], u[:initial], ridge_lambda=1e-3, affine_constant=True
    )
    updater = AccumulativeKoopmanUpdater.from_history(
        z[:initial],
        target[:initial],
        u[:initial],
        A0=first.A,
        B0=first.B,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
    )
    counts: list[int] = []
    for stop in range(initial + batch_size, z.shape[0] + 1, batch_size):
        start = stop - batch_size
        record, candidate = updater.update(
            z[start:stop],
            target[start:stop],
            u[start:stop],
            encoder_fingerprint="encoder-v1",
        )
        oracle = direct_refit(
            z[:stop], target[:stop], u[:stop], ridge_lambda=1e-3, affine_constant=True
        )
        assert record.accepted is True
        assert candidate is not None
        assert np.allclose(candidate.A, oracle.A, atol=1e-10, rtol=1e-10)
        assert np.allclose(candidate.B, oracle.B, atol=1e-10, rtol=1e-10)
        counts.append(record.cumulative_sample_count)
    assert all(right > left for left, right in zip(counts, counts[1:]))
    assert updater.sample_count == z.shape[0]
    assert updater.model_version == (z.shape[0] - initial) // batch_size


def test_rank_deficient_history_remains_finite_with_ridge() -> None:
    z = np.ones((40, 5), dtype=np.float64)
    u = np.ones((40, 2), dtype=np.float64)
    target = np.ones_like(z)
    result = direct_refit(z, target, u, ridge_lambda=1e-3, affine_constant=True)
    assert result.diagnostics.rank < z.shape[1] + u.shape[1]
    assert result.diagnostics.condition_number == float("inf")
    assert result.diagnostics.finite is True
    assert np.all(np.isfinite(result.A))
    assert np.all(np.isfinite(result.B))


def test_encoder_change_invalidates_accumulated_statistics() -> None:
    z, target, u = _synthetic(samples=80)
    initial = direct_refit(z[:40], target[:40], u[:40], ridge_lambda=1e-3)
    updater = AccumulativeKoopmanUpdater.from_history(
        z[:40],
        target[:40],
        u[:40],
        A0=initial.A,
        B0=initial.B,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
    )
    with pytest.raises(ValueError, match="fingerprint changed"):
        updater.update(
            z[40:45], target[40:45], u[40:45], encoder_fingerprint="encoder-v2"
        )


def test_numerical_failure_keeps_current_model_and_statistics() -> None:
    z, target, u = _synthetic(samples=80)
    initial = direct_refit(z[:40], target[:40], u[:40], ridge_lambda=1e-3)
    updater = AccumulativeKoopmanUpdater.from_history(
        z[:40],
        target[:40],
        u[:40],
        A0=initial.A,
        B0=initial.B,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
    )
    A_before = updater.A.copy()
    gram_before = updater.gram.copy()
    invalid = z[40:45].copy()
    invalid[0, 0] = np.nan
    record, candidate = updater.update(
        invalid, target[40:45], u[40:45], encoder_fingerprint="encoder-v1"
    )
    assert record.accepted is False
    assert record.reason.startswith("failed_numerical")
    assert candidate is None
    assert updater.sample_count == 40
    assert np.array_equal(updater.A, A_before)
    assert np.array_equal(updater.gram, gram_before)


def test_replay_discards_nonfinite_batch_and_continues_with_last_model() -> None:
    from prediction.dkuc_prediction import DKUCModel

    model_dir = ROOT / "outputs" / "models" / "dktv" / PLAN01_SMOKE_RUN
    stream_path = (
        ROOT
        / "outputs"
        / "data"
        / "processed"
        / PLAN01_SMOKE_RUN
        / "validation_stream.npz"
    )
    if not model_dir.is_dir() or not stream_path.is_file():
        pytest.skip("reviewed Plan 01 smoke artifact is not available")
    with np.load(model_dir / "dataset_train.npz", allow_pickle=False) as payload:
        train_data = {name: payload[name] for name in payload.files}
    with np.load(stream_path, allow_pickle=False) as payload:
        stream_data = {name: payload[name] for name in payload.files}
    stream_data["states"] = stream_data["states"][:, :13].copy()
    stream_data["inputs"] = stream_data["inputs"][:, :12].copy()
    stream_data["states"][0, 6, 0] = np.nan
    model = DKUCModel(model_dir, "cpu")
    replay = run_accumulative_replay(
        model,
        train_data,
        stream_data,
        batch_size=stream_data["states"].shape[0],
        ridge_lambda=1e-3,
        encoder_fingerprint=artifact_fingerprint(model_dir),
        oracle_tolerance=1e-8,
        invalid_batch_policy="discard_invalid_batch",
    )
    rejected = [record for record in replay.update_history if not record["accepted"]]
    accepted = [record for record in replay.update_history if record["accepted"]]
    assert rejected
    assert accepted
    assert [record["attempt_index"] for record in rejected] == [6, 7]
    assert [record["time_step"] for record in rejected] == [5, 6]
    accepted_after_rejection = [
        record
        for record in accepted
        if record["attempt_index"] > rejected[-1]["attempt_index"]
    ]
    assert accepted_after_rejection[0]["attempt_index"] == 8
    assert len(accepted_after_rejection) == 5
    assert all(record["oracle_check_performed"] is False for record in rejected)
    assert all(record["batch_disposition"] == "discarded_invalid_batch" for record in rejected)
    assert replay.rejected_sample_count == sum(record["batch_sample_count"] for record in rejected)
    assert replay.rejected_sample_count == 4
    assert replay.updater.sample_count == accepted[-1]["cumulative_sample_count"]
    assert replay.updater.sample_count == 212
    summary = update_summary(replay)
    assert summary["failed_count"] == len(rejected)
    assert summary["invalid_batch_policy"] == "discard_invalid_batch"


def test_updater_state_round_trip_and_replay(tmp_path: Path) -> None:
    z, target, u = _synthetic(samples=90)
    initial = direct_refit(z[:50], target[:50], u[:50], ridge_lambda=1e-3)
    updater = AccumulativeKoopmanUpdater.from_history(
        z[:50],
        target[:50],
        u[:50],
        A0=initial.A,
        B0=initial.B,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
    )
    updater.update(z[50:60], target[50:60], u[50:60], encoder_fingerprint="encoder-v1")
    state_path = tmp_path / "updater.npz"
    updater.save(state_path)
    restored = AccumulativeKoopmanUpdater.load(state_path)
    first, _ = updater.update(
        z[60:70], target[60:70], u[60:70], encoder_fingerprint="encoder-v1"
    )
    second, _ = restored.update(
        z[60:70], target[60:70], u[60:70], encoder_fingerprint="encoder-v1"
    )
    assert first.to_dict() | {"update_time_s": 0.0} == second.to_dict() | {"update_time_s": 0.0}
    assert np.array_equal(updater.gram, restored.gram)
    assert np.array_equal(updater.cross, restored.cross)
    assert np.array_equal(updater.A, restored.A)
    assert np.array_equal(updater.B, restored.B)


def _aggregate_contract_manifest() -> dict:
    stage = {
        "total_rmse": 0.1,
        "start_step": 0,
        "end_step": 20,
        "disturbance_scale": 0.0,
        "rollout_horizon": 10,
    }
    metric = {"total_rmse": 0.1}
    return {
        "config": {"sha256": "config"},
        "source_files": {"source.py": {"sha256": "source"}},
        "batch_sizes": [5],
        "coordinates": {"state": "normalized"},
        "stream": {
            "trajectory_count": 5,
            "steps": 20,
            "online_snapshot_count": 100,
            "ordering": "time_major_then_trajectory",
            "causal_order": "predict_then_observe_then_update",
            "physical_semantics": "synchronous_multi_trajectory_simulation",
            "snapshots_per_simulation_step": 5,
            "batch_size_to_simulation_steps": {"5": 1.0},
        },
        "source_plan01": {"scenario_contract": {"plan01_manifest_schema_version": 2}},
        "update_history_contract": {"schema_version": 2},
        "metrics": {
            "one_step": {"fixed_dko": metric, "dktv_accumulative_b5": metric},
            "rollout": {
                "fixed_dko": {"10": metric},
                "dktv_accumulative_b5": {"10": metric},
            },
            "segmented": {
                "fixed_dko": {"nominal": stage},
                "dktv_accumulative_b5": {"nominal": stage},
            },
        },
        "update_summary": {"dktv_accumulative_b5": {"maximum_condition_number": 3.0}},
    }


def test_aggregate_comparison_contract_rejects_mixed_config() -> None:
    first = _aggregate_contract_manifest()
    second = deepcopy(first)
    assert _comparison_contract([first, second])["passed"] is True
    second["config"]["sha256"] = "different"
    contract = _comparison_contract([first, second])
    assert contract["passed"] is False
    assert contract["checks"]["shared_plan02_config_hash"] is False


def test_paired_summary_records_ci_and_wins() -> None:
    paired = _paired_summary(np.asarray([2.0, 3.0, 4.0]), np.asarray([1.0, 2.0, 3.0]))
    assert paired["win_count"] == 3
    assert paired["mean_difference"] == 1.0
    assert len(paired["difference_confidence_interval_95_student_t"]) == 2


def test_plan_02_smoke_cli_end_to_end(tmp_path: Path) -> None:
    source_manifest = (
        ROOT / "outputs" / "results" / "dktv" / "plan_01" / PLAN01_SMOKE_RUN / "manifest.json"
    )
    if not source_manifest.is_file():
        pytest.skip("reviewed Plan 01 smoke artifact is not available")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "prediction.dktv_accumulative_prediction",
        "--run-type",
        "smoke",
        "--plan01-run",
        PLAN01_SMOKE_RUN,
        "--device",
        "cpu",
        "--run-id",
        "pytest_plan02_e2e",
        "--output-root",
        str(tmp_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result_dir = tmp_path / "results" / "dktv" / "plan_02" / "pytest_plan02_e2e"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["acceptance"]["passed"] is True
    assert manifest["acceptance"]["recursive_matches_direct_refit"] is True
    assert manifest["stream"]["causal_order"] == "predict_then_observe_then_update"
    assert (result_dir / "arrays" / "predictions.npz").is_file()
    assert (result_dir / "arrays" / "update_history.npz").is_file()
    assert (result_dir / "arrays" / "update_history_schema.json").is_file()
    assert (result_dir / "logs" / "update_history.jsonl").is_file()
    with np.load(result_dir / "arrays" / "update_history.npz", allow_pickle=False) as history:
        assert int(history["update_history_schema_version"]) == 2
        assert "dktv_accumulative_b10_accepted" in history.files
        assert "dktv_accumulative_b10_recursive_spectral_radius_A" in history.files
    assert (result_dir / "metrics" / "segmented.json").is_file()
    assert all(not path.startswith("/") for path in manifest["figures"])
