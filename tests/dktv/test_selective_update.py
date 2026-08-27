"""Selective-window state-machine and buffer-policy checks."""

from __future__ import annotations

import numpy as np
import pytest

from prediction.dktv.least_squares import direct_refit
from prediction.dktv.selective_update import SelectiveWindowKoopmanUpdater
from prediction.dktv.window_update import SlidingWindowKoopmanUpdater

from .test_window_update import synthetic


def make_selective(*, epsilon: float, policy: str, improvement_tolerance: float = 0.0):
    z, target, u = synthetic(samples=80)
    initial = direct_refit(z[:50], target[:50], u[:50], ridge_lambda=1e-3)
    base = SlidingWindowKoopmanUpdater(
        z[:50], target[:50], u[:50], A0=initial.A, B0=initial.B,
        batch_size=5, ridge_lambda=1e-3, encoder_fingerprint="encoder-v1",
        sample_ids=np.arange(50),
    )
    return (
        SelectiveWindowKoopmanUpdater(
            base,
            epsilon=epsilon,
            reject_buffer_policy=policy,
            improvement_tolerance=improvement_tolerance,
        ),
        z,
        target,
        u,
    )


def apply(updater, z, target, u, start=50):
    return updater.update(
        z[start : start + 5], target[start : start + 5], u[start : start + 5],
        sample_ids=np.arange(start, start + 5), encoder_fingerprint="encoder-v1",
    )


def test_selective_accepted_state() -> None:
    updater, z, target, u = make_selective(epsilon=0.0, policy="discard_on_reject")
    record, _ = apply(updater, z, target, u)
    assert record.status == "accepted"
    assert record.accepted and record.window_advanced
    assert updater.model_version == 1
    assert updater.window_version == 1


def test_selective_skipped_threshold_advances_window_only() -> None:
    updater, z, target, u = make_selective(epsilon=1e6, policy="discard_on_reject")
    record, _ = apply(updater, z, target, u)
    assert record.status == "skipped_threshold"
    assert not record.accepted and record.window_advanced
    assert updater.model_version == 0
    assert updater.window_version == 1
    assert updater.sample_ids[-5:].tolist() == list(range(50, 55))


@pytest.mark.parametrize(
    "policy,advanced,expected_tail",
    [
        ("discard_on_reject", False, list(range(45, 50))),
        ("retain_on_reject", True, list(range(50, 55))),
    ],
)
def test_reject_buffer_policy_has_explicit_window_semantics(
    policy: str, advanced: bool, expected_tail: list[int]
) -> None:
    updater, z, target, u = make_selective(
        epsilon=0.0, policy=policy, improvement_tolerance=1e6
    )
    record, _ = apply(updater, z, target, u)
    assert record.status == "rejected"
    assert record.window_advanced is advanced
    assert record.reject_buffer_policy == policy
    assert updater.model_version == 0
    assert updater.sample_ids[-5:].tolist() == expected_tail
    assert np.unique(updater.sample_ids).size == updater.window_size


def test_selective_failed_numerical_keeps_window_and_model() -> None:
    updater, z, target, u = make_selective(epsilon=0.0, policy="retain_on_reject")
    invalid = z[50:55].copy()
    invalid[0, 0] = np.nan
    record, candidate = updater.update(
        invalid, target[50:55], u[50:55], sample_ids=np.arange(50, 55),
        encoder_fingerprint="encoder-v1",
    )
    assert record.status == "failed_numerical"
    assert candidate is None
    assert not record.window_advanced
    assert updater.model_version == 0
    assert updater.window_version == 0
