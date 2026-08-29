"""Focused numerical checks for the restored DKTV and OTVDKL cores."""

from __future__ import annotations

import numpy as np

from prediction.common import direct_refit
from prediction.dktv_prediction import AccumulativeKoopmanUpdater
from prediction.otvdkl_prediction import (
    SelectiveWindowKoopmanUpdater,
    SlidingWindowKoopmanUpdater,
)


def _synthetic(seed: int = 17, samples: int = 120):
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


def test_dktv_accumulation_matches_full_history_refit() -> None:
    z, target, u = _synthetic()
    initial_count, batch_size = 60, 10
    initial = direct_refit(
        z[:initial_count], target[:initial_count], u[:initial_count], ridge_lambda=1e-3
    )
    updater = AccumulativeKoopmanUpdater.from_history(
        z[:initial_count],
        target[:initial_count],
        u[:initial_count],
        A0=initial.A,
        B0=initial.B,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
    )

    for start in range(initial_count, z.shape[0], batch_size):
        stop = start + batch_size
        record, candidate = updater.update(
            z[start:stop],
            target[start:stop],
            u[start:stop],
            encoder_fingerprint="encoder-v1",
        )
        assert record.accepted
        assert candidate is not None

    oracle = direct_refit(z, target, u, ridge_lambda=1e-3)
    assert np.allclose(updater.A, oracle.A, atol=1e-10, rtol=1e-10)
    assert np.allclose(updater.B, oracle.B, atol=1e-10, rtol=1e-10)
    assert updater.sample_count == z.shape[0]


def test_otvdkl_window_matches_direct_refit() -> None:
    z, target, u = _synthetic()
    window_size, batch_size = 50, 5
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
    )

    start, stop = window_size, window_size + batch_size
    record, candidate = updater.update(
        z[start:stop],
        target[start:stop],
        u[start:stop],
        sample_ids=np.arange(start, stop),
        encoder_fingerprint="encoder-v1",
    )
    oracle = direct_refit(z[stop - window_size : stop], target[stop - window_size : stop], u[stop - window_size : stop], ridge_lambda=1e-3)
    assert record.accepted
    assert candidate is not None
    assert np.allclose(updater.A, oracle.A, atol=1e-8, rtol=1e-8)
    assert np.allclose(updater.B, oracle.B, atol=1e-8, rtol=1e-8)


def test_otvdkl_selective_threshold_keeps_model_and_window() -> None:
    z, target, u = _synthetic()
    window_size, batch_size = 50, 5
    initial = direct_refit(z[:window_size], target[:window_size], u[:window_size], ridge_lambda=1e-3)
    base = SlidingWindowKoopmanUpdater(
        z[:window_size],
        target[:window_size],
        u[:window_size],
        A0=initial.A,
        B0=initial.B,
        batch_size=batch_size,
        ridge_lambda=1e-3,
        encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(window_size),
    )
    updater = SelectiveWindowKoopmanUpdater(base, epsilon=1e6)
    record, _ = updater.update(
        z[window_size : window_size + batch_size],
        target[window_size : window_size + batch_size],
        u[window_size : window_size + batch_size],
        sample_ids=np.arange(window_size, window_size + batch_size),
        encoder_fingerprint="encoder-v1",
    )
    assert record.status == "skipped_threshold"
    assert not record.accepted
    assert not record.window_advanced
    assert updater.model_version == 0
    assert updater.window_version == 0
