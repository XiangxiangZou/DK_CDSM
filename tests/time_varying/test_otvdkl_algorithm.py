from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from prediction.common import direct_refit
from prediction.otvdkl_prediction import (
    SelectiveWindowKoopmanUpdater,
    SlidingWindowKoopmanUpdater,
    physical_state_rmse,
    predict_otvdkl_batch,
    run_otvdkl_replay,
)


def _samples(seed: int = 4, count: int = 96):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(count, 8))
    u = rng.normal(size=(count, 2))
    A = np.diag(np.linspace(0.65, 0.85, 8))
    B = rng.normal(scale=0.05, size=(8, 2))
    target = z @ A.T + u @ B.T + rng.normal(scale=1e-6, size=z.shape)
    return z, target, u


def _updater():
    z, target, u = _samples()
    fit = direct_refit(z[:48], target[:48], u[:48], ridge_lambda=1e-3, affine_constant=False)
    updater = SlidingWindowKoopmanUpdater(
        z[:48], target[:48], u[:48], A0=fit.A, B0=fit.B,
        state_dim=4, batch_size=8, ridge_lambda=1e-3,
        encoder_fingerprint="fixed-encoder", sample_ids=np.arange(48),
        affine_constant=False,
    )
    return z, target, u, updater


def test_ab_c_recursive_state_matches_direct_window_oracles() -> None:
    z, target, u, updater = _updater()
    record, candidate = updater.update(
        z[48:56], target[48:56], u[48:56], sample_ids=np.arange(48, 56),
        encoder_fingerprint="fixed-encoder",
    )
    assert record.accepted and candidate is not None
    assert record.candidate_A_max_abs_difference < 1e-8
    assert record.candidate_B_max_abs_difference < 1e-8
    assert record.candidate_C_max_abs_difference < 1e-8
    assert np.allclose(updater.C, candidate.direct_C, atol=1e-8, rtol=1e-8)
    expected_c_inverse = np.linalg.inv(
        candidate.c_gram + updater.ridge_lambda * np.eye(updater.latent_dim)
    )
    assert np.allclose(
        candidate.inverse_regularized_c_gram,
        expected_c_inverse,
        atol=1e-8,
        rtol=1e-8,
    )
    assert record.c_addition_system_condition_number is not None
    assert record.c_deletion_system_condition_number is not None
    assert record.recursive_path == "woodbury_add_delete"


def test_selective_threshold_does_not_construct_or_advance_candidate() -> None:
    z, target, u, base = _updater()
    before = base.sample_ids.copy()
    selective = SelectiveWindowKoopmanUpdater(base, epsilon=1e9)
    record, candidate = selective.update(
        z[48:56], target[48:56], u[48:56], sample_ids=np.arange(48, 56),
        encoder_fingerprint="fixed-encoder",
    )
    assert record.status == "skipped_threshold"
    assert candidate is None
    assert not record.window_advanced
    assert np.array_equal(base.sample_ids, before)


def test_checkpoint_resume_preserves_versions_statistics_and_pending(tmp_path) -> None:
    z, target, u, continuous = _updater()
    _, _, _, checkpointed = _updater()
    first = dict(
        z_current=z[48:56], z_next=target[48:56], u_normalized=u[48:56],
        sample_ids=np.arange(48, 56), encoder_fingerprint="fixed-encoder",
    )
    second = dict(
        z_current=z[56:64], z_next=target[56:64], u_normalized=u[56:64],
        sample_ids=np.arange(56, 64), encoder_fingerprint="fixed-encoder",
    )
    continuous.update(**first)
    continuous.update(**second)
    checkpointed.update(**first)
    path = tmp_path / "checkpoint.npz"
    checkpointed.save_checkpoint(
        path,
        pending={"sample_ids": np.array([56, 57]), "z_current": z[56:58]},
    )
    restored, pending = SlidingWindowKoopmanUpdater.load_checkpoint(path)
    assert np.array_equal(pending["sample_ids"], [56, 57])
    restored.update(**second)
    assert restored.model_version == continuous.model_version
    assert restored.window_version == continuous.window_version
    for name in (
        "sample_ids", "z_current", "z_next", "u_normalized", "A", "B", "C",
        "gram", "cross", "inverse_regularized_gram", "c_gram", "c_cross",
        "inverse_regularized_c_gram",
    ):
        assert np.allclose(getattr(restored, name), getattr(continuous, name))


def test_selective_checkpoint_restores_variant_configuration(tmp_path) -> None:
    _, _, _, base = _updater()
    updater = SelectiveWindowKoopmanUpdater(
        base,
        epsilon=0.125,
        improvement_tolerance=1e-7,
    )
    path = tmp_path / "otvdkl_star_checkpoint.npz"
    updater.save_checkpoint(path)
    restored, pending = SlidingWindowKoopmanUpdater.load_checkpoint(path)
    assert isinstance(restored, SelectiveWindowKoopmanUpdater)
    assert restored.epsilon == 0.125
    assert restored.improvement_tolerance == 1e-7
    assert pending == {}


class _IdentityNormalizer:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)

    def inverse(self, values):
        return np.asarray(values, dtype=np.float64)


class _ReadoutModel:
    state_dim = 2
    x_normer = _IdentityNormalizer()


def test_prediction_uses_learned_c_readout() -> None:
    model = _ReadoutModel()
    A = np.eye(3)
    B = np.zeros((3, 1))
    C = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    z = np.array([[1.0, 3.0, 5.0]])
    prediction = predict_otvdkl_batch(model, A, B, C, z, np.zeros((1, 1)))
    assert np.array_equal(prediction, [[3.0, 2.0]])


class _TorchLift:
    def __init__(self, torch_module):
        self._torch = torch_module

    def lift(self, values):
        return self._torch.cat((values, values[:, :1] ** 2), dim=1)


class _ReplayModel:
    def __init__(self):
        import torch

        self.state_dim = 4
        self.control_dim = 2
        self.x_normer = _IdentityNormalizer()
        self.u_normer = _IdentityNormalizer()
        self._torch = torch
        self.device = torch.device("cpu")
        self.model = _TorchLift(torch)
        self.config = SimpleNamespace(include_constant=False)
        self.A = np.zeros((5, 5), dtype=np.float64)
        self.B = np.zeros((5, 2), dtype=np.float64)

    def lift(self, state):
        values = np.asarray(state, dtype=np.float64).reshape(1, 4)
        return np.concatenate((values[0], values[0, :1] ** 2))


def _replay_dataset(seed: int, steps: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "states": rng.normal(size=(1, steps + 1, 4)),
        "inputs": rng.normal(size=(1, steps, 2)),
    }


def test_replay_predictions_are_causal_and_initial_model_matches_window() -> None:
    model = _ReplayModel()
    history = _replay_dataset(20, 24)
    stream = _replay_dataset(21, 12)
    changed_future = {key: value.copy() for key, value in stream.items()}
    changed_future["states"][:, 7:] += 100.0
    changed_future["inputs"][:, 7:] -= 50.0
    kwargs = dict(
        variant="otvdkl",
        window_size=10,
        batch_size=2,
        ridge_lambda=1e-3,
        encoder_fingerprint="fake-fixed-encoder",
        epsilon=0.0,
        improvement_tolerance=0.0,
        oracle_tolerance=1e-8,
        low_dim_condition_limit=1e12,
        window_condition_limit=1e12,
    )
    _, evidence, arrays = run_otvdkl_replay(model, history, stream, **kwargs)
    _, _, changed_arrays = run_otvdkl_replay(
        model,
        history,
        changed_future,
        **kwargs,
    )
    assert evidence["summary"]["initial_model_source"] == "direct_refit_of_initial_window"
    assert not np.allclose(arrays["otvdkl_A_by_step"][0], model.A)
    assert np.allclose(
        arrays["otvdkl_one_step"][:, :7],
        changed_arrays["otvdkl_one_step"][:, :7],
    )


def test_replay_records_explicit_history_window_and_latent_stability() -> None:
    model = _ReplayModel()
    history = _replay_dataset(22, 24)
    stream = _replay_dataset(23, 12)
    _, evidence, arrays = run_otvdkl_replay(
        model,
        history,
        stream,
        variant="otvdkl",
        window_size=10,
        batch_size=2,
        ridge_lambda=1e-3,
        encoder_fingerprint="fake-fixed-encoder",
        epsilon=0.0,
        improvement_tolerance=0.0,
        oracle_tolerance=1e-8,
        low_dim_condition_limit=1e12,
        window_condition_limit=1e12,
        history_window_start=3,
    )
    summary = evidence["summary"]
    radius = arrays["otvdkl_A_spectral_radius_by_step"]
    norm = arrays["otvdkl_A_spectral_norm_by_step"]
    assert summary["history_window_start"] == 3
    assert summary["history_window_end_exclusive"] == 13
    assert radius.shape == (12,)
    assert norm.shape == (12,)
    assert np.all(np.isfinite(radius))
    assert np.all(np.isfinite(norm))
    assert np.isclose(summary["maximum_A_spectral_radius"], np.max(radius))
    assert np.isclose(summary["maximum_A_spectral_norm"], np.max(norm))


def test_physical_error_ignores_unread_latent_residual() -> None:
    A = np.eye(6)
    B = np.zeros((6, 1))
    C = np.hstack([np.eye(2), np.zeros((2, 4))])
    z = np.zeros((3, 6))
    target = z.copy()
    target[:, 5] = 100.0
    assert physical_state_rmse(A, B, C, z, np.zeros((3, 1)), target[:, :2]) == 0.0
