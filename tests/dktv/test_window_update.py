"""Direct and recursive fixed-window updater checks."""

from __future__ import annotations

import numpy as np
import pytest

from prediction.dktv.least_squares import (
    build_regressor,
    direct_refit,
    sufficient_statistics,
)
from prediction.dktv.window_update import SlidingWindowKoopmanUpdater


def synthetic(seed: int = 17, samples: int = 180):
    rng = np.random.default_rng(seed)
    latent_dim, input_dim = 7, 2
    z = rng.normal(size=(samples, latent_dim))
    z[:, -1] = 1.0
    u = rng.normal(size=(samples, input_dim))
    A = 0.72 * np.eye(latent_dim)
    A[-1] = 0.0
    A[-1, -1] = 1.0
    B = rng.normal(scale=0.08, size=(latent_dim, input_dim))
    B[-1] = 0.0
    target = z @ A.T + u @ B.T + rng.normal(scale=1e-5, size=z.shape)
    target[:, -1] = 1.0
    return z, target, u


@pytest.mark.parametrize("window_size,batch_size", [(50, 5), (100, 10), (100, 20)])
def test_recursive_window_matches_direct_refit(window_size: int, batch_size: int) -> None:
    z, target, u = synthetic(samples=window_size + 60)
    initial = direct_refit(z[:window_size], target[:window_size], u[:window_size], ridge_lambda=1e-3)
    updater = SlidingWindowKoopmanUpdater(
        z[:window_size],
        target[:window_size],
        u[:window_size],
        A0=initial.A,
        B0=initial.B,
        batch_size=batch_size,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(window_size),
        oracle_tolerance=1e-8,
    )
    memory = updater.memory_bytes
    for start in range(window_size, window_size + 60, batch_size):
        stop = start + batch_size
        record, candidate = updater.update(
            z[start:stop],
            target[start:stop],
            u[start:stop],
            sample_ids=np.arange(start, stop),
            encoder_fingerprint="encoder-v1",
        )
        assert record.status == "accepted"
        assert candidate is not None
        oracle = direct_refit(
            z[stop - window_size : stop],
            target[stop - window_size : stop],
            u[stop - window_size : stop],
            ridge_lambda=1e-3,
        )
        assert np.allclose(updater.A, oracle.A, atol=1e-8, rtol=1e-8)
        assert np.allclose(updater.B, oracle.B, atol=1e-8, rtol=1e-8)
        assert record.recursive_A_max_abs_difference is not None
        assert record.recursive_A_max_abs_difference <= 1e-8
        assert record.recursive_B_max_abs_difference <= 1e-8
        assert record.inserted_sample_ids == list(range(start, stop))
        assert record.evicted_sample_ids == list(range(start - window_size, stop - window_size))
        assert updater.sample_ids.tolist() == list(range(stop - window_size, stop))
        assert updater.z_current.shape[0] == window_size
        assert updater.memory_bytes == memory
        assert record.recursive_candidate_time_s is not None
        assert record.direct_refit_oracle_time_s is not None
        assert record.fallback_time_s is not None
        assert record.total_update_time_s == record.update_time_s
        assert record.total_update_time_s >= record.recursive_candidate_time_s
        assert record.total_update_time_s >= record.direct_refit_oracle_time_s


def test_window_rejects_wrong_batch_and_keeps_state() -> None:
    z, target, u = synthetic(samples=80)
    initial = direct_refit(z[:50], target[:50], u[:50], ridge_lambda=1e-3)
    updater = SlidingWindowKoopmanUpdater(
        z[:50], target[:50], u[:50], A0=initial.A, B0=initial.B,
        batch_size=5, ridge_lambda=1e-3, encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(50),
    )
    before = updater.sample_ids.copy()
    record, candidate = updater.update(
        z[50:54], target[50:54], u[50:54], sample_ids=np.arange(50, 54),
        encoder_fingerprint="encoder-v1",
    )
    assert record.status == "failed_numerical"
    assert candidate is None
    assert np.array_equal(updater.sample_ids, before)
    assert updater.model_version == 0


def test_rank_deficient_candidate_fails_without_mutation() -> None:
    z, target, u = synthetic(samples=60)
    initial = direct_refit(z[:50], target[:50], u[:50], ridge_lambda=1e-3)
    updater = SlidingWindowKoopmanUpdater(
        z[:50], target[:50], u[:50], A0=initial.A, B0=initial.B,
        batch_size=50, ridge_lambda=1e-3, encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(50),
    )
    constant_z = np.ones((50, z.shape[1]))
    constant_u = np.ones((50, u.shape[1]))
    constant_next = np.ones_like(constant_z)
    record, candidate = updater.update(
        constant_z, constant_next, constant_u, sample_ids=np.arange(50, 100),
        encoder_fingerprint="encoder-v1",
    )
    assert record.status == "failed_numerical"
    assert candidate is None
    assert updater.window_version == 0


def test_fallback_reanchors_statistics_from_raw_candidate_window() -> None:
    z, target, u = synthetic(samples=70)
    initial = direct_refit(z[:50], target[:50], u[:50], ridge_lambda=1e-3)
    updater = SlidingWindowKoopmanUpdater(
        z[:50], target[:50], u[:50], A0=initial.A, B0=initial.B,
        batch_size=5, ridge_lambda=1e-3, encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(50), oracle_tolerance=1e-20,
    )
    record, candidate = updater.update(
        z[50:55], target[50:55], u[50:55], sample_ids=np.arange(50, 55),
        encoder_fingerprint="encoder-v1",
    )
    assert record.status == "accepted"
    assert record.recursive_path == "direct_refit_fallback"
    assert record.fallback_time_s is not None and record.fallback_time_s > 0.0
    assert candidate is not None
    regressor = build_regressor(updater.z_current, updater.u_normalized)
    gram, cross = sufficient_statistics(regressor, updater.z_next)
    expected_inverse = np.linalg.inv(gram + 1e-3 * np.eye(gram.shape[0]))
    assert np.array_equal(updater.gram, gram)
    assert np.array_equal(updater.cross, cross)
    assert np.allclose(updater.inverse_regularized_gram, expected_inverse, atol=1e-12, rtol=1e-12)


def test_extreme_scale_overflow_fails_and_preserves_finite_state() -> None:
    z, target, u = synthetic(samples=60)
    initial = direct_refit(z[:50], target[:50], u[:50], ridge_lambda=1e-3)
    updater = SlidingWindowKoopmanUpdater(
        z[:50], target[:50], u[:50], A0=initial.A, B0=initial.B,
        batch_size=5, ridge_lambda=1e-3, encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(50),
    )
    before = (updater.A.copy(), updater.B.copy(), updater.gram.copy(), updater.sample_ids.copy())
    huge_z = np.full((5, z.shape[1]), 1e200)
    huge_z[:, -1] = 1.0
    huge_u = np.full((5, u.shape[1]), 1e200)
    huge_next = np.full_like(huge_z, 1e200)
    huge_next[:, -1] = 1.0
    record, candidate = updater.update(
        huge_z, huge_next, huge_u, sample_ids=np.arange(50, 55),
        encoder_fingerprint="encoder-v1",
    )
    assert record.status == "failed_numerical"
    assert candidate is None
    assert np.array_equal(updater.A, before[0])
    assert np.array_equal(updater.B, before[1])
    assert np.array_equal(updater.gram, before[2])
    assert np.array_equal(updater.sample_ids, before[3])
    assert np.all(np.isfinite(updater.A))
    assert np.all(np.isfinite(updater.B))
