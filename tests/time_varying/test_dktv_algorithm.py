"""Formula and neural-update tests for the full Hao-DKTV implementation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from prediction.common import Normalizer
from prediction.dkuc_prediction import DKUCConfig, make_network_class
from prediction.dktv_prediction import (
    HaoDKTVState,
    encoder_checksum,
    long_horizon_metrics,
    matched_rollout_start_indices,
    run_hao_dktv_replay,
    train_online_encoder,
)


def _samples(seed: int = 7, count: int = 90):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(count, 8))
    u = rng.normal(size=(count, 2))
    x = z[:, :4] + 0.03 * rng.normal(size=(count, 4))
    A = 0.8 * np.eye(8) + rng.normal(scale=0.01, size=(8, 8))
    B = rng.normal(scale=0.05, size=(8, 2))
    y = z @ A.T + u @ B.T
    return z, y, u, x


def test_matched_long_horizon_origins_and_metrics_are_reproducible() -> None:
    starts = matched_rollout_start_indices(steps=3000, max_horizon=1000, stride=50)
    assert starts[0] == 200
    assert starts[-1] == 2000
    assert starts.size == 37

    truth = np.zeros((2, 6, 4), dtype=np.float64)
    prediction = truth.copy()
    prediction[:, 1:] = 1.0
    latent_norm = np.broadcast_to(np.arange(6, dtype=np.float64), (2, 6)).copy()
    metrics, diagnostics = long_horizon_metrics(
        truth, prediction, dt=0.01, latent_norm=latent_norm
    )
    assert metrics["horizon_steps"] == 5
    assert metrics["horizon_seconds"] == 0.05
    assert metrics["window_count"] == 2
    assert metrics["total_rmse"] == 1.0
    assert metrics["maximum_latent_norm"] == 5.0
    assert diagnostics["rmse_by_lead_step"][0] == 0.0
    assert np.all(diagnostics["rmse_by_lead_step"][1:] == 1.0)


def test_hao_abc_recursion_matches_cumulative_ridge_refit(tmp_path) -> None:
    z, y, u, x = _samples()
    ridge, split = 1e-3, 45
    state = HaoDKTVState.from_history(z[:split], y[:split], u[:split], x[:split],
                                      ridge_lambda=ridge)
    for start in range(split, len(z), 9):
        assert state.update(z[start:start + 9], y[start:start + 9],
                            u[start:start + 9], x[start:start + 9])["accepted"]
    chi = np.concatenate((z, u), axis=1)
    expected_k = y.T @ chi @ np.linalg.solve(chi.T @ chi + ridge * np.eye(10), np.eye(10))
    expected_c = x.T @ z @ np.linalg.solve(z.T @ z + ridge * np.eye(8), np.eye(8))
    assert np.allclose(np.concatenate((state.A, state.B), axis=1), expected_k, atol=1e-9)
    assert np.allclose(state.C, expected_c, atol=1e-9)
    assert np.allclose(state.P_chi, np.linalg.solve(chi.T @ chi + ridge * np.eye(10), np.eye(10)), atol=1e-9)
    path = tmp_path / "state.npz"
    state.save(path)
    restored = HaoDKTVState.load(path)
    assert np.allclose(restored.C, state.C)
    assert restored.model_version == state.model_version


def test_hao_update_rejects_nonfinite_without_partial_mutation() -> None:
    z, y, u, x = _samples(count=30)
    state = HaoDKTVState.from_history(z[:20], y[:20], u[:20], x[:20], ridge_lambda=1e-3)
    before = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in vars(state).items()
    }
    y[20, 0] = np.nan
    result = state.update(z[20:], y[20:], u[20:], x[20:])
    assert not result["accepted"]
    for name, value in before.items():
        current = getattr(state, name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(current, value)
        else:
            assert current == value


def test_batch_partition_and_resume_update_are_equivalent(tmp_path) -> None:
    z, y, u, x = _samples(count=84)
    split = 24
    left = HaoDKTVState.from_history(z[:split], y[:split], u[:split], x[:split],
                                     ridge_lambda=1e-3)
    right = HaoDKTVState.from_history(z[:split], y[:split], u[:split], x[:split],
                                      ridge_lambda=1e-3)
    assert left.update(z[split:], y[split:], u[split:], x[split:])["accepted"]
    for start in range(split, len(z), 5):
        assert right.update(z[start:start + 5], y[start:start + 5],
                            u[start:start + 5], x[start:start + 5])["accepted"]
    for name in ("A", "B", "C", "gram_chi", "cross_chi", "gram_g", "cross_g",
                 "P_chi", "P_g"):
        assert np.allclose(getattr(left, name), getattr(right, name), atol=2e-9)

    checkpoint = tmp_path / "resume.npz"
    right.save(checkpoint)
    resumed = HaoDKTVState.load(checkpoint)
    extra = _samples(seed=19, count=11)
    assert right.update(*extra)["accepted"]
    assert resumed.update(*extra)["accepted"]
    for name in ("A", "B", "C", "P_chi", "P_g"):
        assert np.array_equal(getattr(right, name), getattr(resumed, name))


def test_finite_ill_conditioned_batch_remains_finite() -> None:
    z, y, u, x = _samples(count=30)
    state = HaoDKTVState.from_history(z[:20], y[:20], u[:20], x[:20], ridge_lambda=1e-2)
    direction = np.linspace(0.1, 0.8, z.shape[1])
    bad_z = np.outer(np.ones(10), direction)
    bad_u = np.ones((10, 2)) * 1e4
    bad_y = bad_z * 0.999999
    bad_x = bad_z[:, :4]
    result = state.update(bad_z, bad_y, bad_u, bad_x)
    assert result["accepted"]
    assert state.diagnostics()["finite"]
    assert np.isfinite(result["diagnostics"]["chi"]["regularized_condition_number"])


def test_full_online_loss_updates_reused_dkuc_encoder() -> None:
    import torch

    torch.manual_seed(3)
    config = DKUCConfig(lift_dim=4, hidden=(12,), include_constant=False)
    network = make_network_class()(config, 4, 2)
    normalizer = Normalizer(mean=np.zeros(4), std=np.ones(4))
    u_normalizer = Normalizer(mean=np.zeros(2), std=np.ones(2))
    model = SimpleNamespace(model=network, device=torch.device("cpu"), x_normer=normalizer,
                            u_normer=u_normalizer)
    rng = np.random.default_rng(5)
    x = rng.normal(scale=0.3, size=(32, 4))
    u = rng.normal(scale=0.2, size=(32, 2))
    x_next = 0.85 * x + np.pad(0.1 * u, ((0, 0), (0, 2)))
    with torch.no_grad():
        z = network.lift(torch.tensor(x, dtype=torch.float32)).numpy()
        zn = network.lift(torch.tensor(x_next, dtype=torch.float32)).numpy()
    state = HaoDKTVState.from_history(z, zn, u, x, ridge_lambda=1e-3)
    state.A[4:, 4:] *= 0.6
    state.C[:, 4:] += 0.15
    before = encoder_checksum(network)
    result = train_online_encoder(model, state, x, x_next, u, loss_weight=0.5, epochs=12,
                                  learning_rate=2e-3, weight_decay=0.0, grad_clip=5.0)
    assert result["accepted"]
    assert encoder_checksum(network) != before
    assert result["final"]["L"] <= result["initial"]["L"]
    assert np.isclose(result["final"]["L"], result["best_loss"], rtol=1e-6, atol=1e-8)
    assert state.encoder_version == 1


def test_full_online_training_rejects_zero_epochs() -> None:
    import pytest
    import torch

    config = DKUCConfig(lift_dim=2, hidden=(6,), include_constant=False)
    network = make_network_class()(config, 4, 2)
    model = SimpleNamespace(
        model=network,
        device=torch.device("cpu"),
        x_normer=Normalizer(mean=np.zeros(4), std=np.ones(4)),
        u_normer=Normalizer(mean=np.zeros(2), std=np.ones(2)),
    )
    x = np.zeros((4, 4))
    u = np.zeros((4, 2))
    with torch.no_grad():
        z = network.lift(torch.tensor(x, dtype=torch.float32)).numpy()
    state = HaoDKTVState.from_history(z, z, u, x, ridge_lambda=1e-3)
    with pytest.raises(ValueError, match="invalid online optimization"):
        train_online_encoder(model, state, x, x, u, loss_weight=0.5, epochs=0,
                             learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0)


def _toy_replay(seed: int = 31):
    import torch

    rng = np.random.default_rng(seed)
    config = DKUCConfig(lift_dim=3, hidden=(8,), include_constant=False)
    network = make_network_class()(config, 4, 2)
    x_normer = Normalizer(mean=np.zeros(4), std=np.ones(4))
    u_normer = Normalizer(mean=np.zeros(2), std=np.ones(2))
    model = SimpleNamespace(
        model=network, device=torch.device("cpu"), x_normer=x_normer,
        u_normer=u_normer, state_dim=4, control_dim=2, config=config, _torch=torch,
        A=network.A.weight.detach().numpy().astype(np.float64),
        B=network.B.weight.detach().numpy().astype(np.float64),
    )
    model.lift = lambda value: (
        network.lift(torch.tensor(np.asarray(value).reshape(1, 4), dtype=torch.float32))
        .detach().numpy().reshape(-1).astype(np.float64)
    )
    model.recover_state = lambda value: np.asarray(value)[:4]
    history_states = rng.normal(scale=0.2, size=(2, 11, 4))
    history_inputs = rng.normal(scale=0.1, size=(2, 10, 2))
    stream_states = rng.normal(scale=0.2, size=(2, 7, 4))
    stream_inputs = rng.normal(scale=0.1, size=(2, 6, 2))
    return model, {"states": history_states, "inputs": history_inputs}, {
        "states": stream_states, "inputs": stream_inputs
    }


def test_full_replay_keeps_fixed_baseline_frozen_and_trials_isolated() -> None:
    import torch

    torch.manual_seed(31)
    model, history, stream = _toy_replay()
    initial = {name: value.detach().clone() for name, value in model.model.state_dict().items()}
    final_state, evidence, arrays = run_hao_dktv_replay(
        model, history, stream, mode="full", batch_size=2, ridge_lambda=1e-3,
        loss_weight=0.5, online_epochs=3, online_lr=2e-2,
        online_weight_decay=0.0, grad_clip=5.0,
    )
    final_network = {name: value.detach().clone() for name, value in model.model.state_dict().items()}
    assert evidence["summary"]["full_coordinate_refit"]
    assert all(record.get("coordinate_consistent") for record in evidence["history"] if record["accepted"])

    model.model.load_state_dict(initial)
    expected = np.zeros_like(stream["states"])
    expected[:, 0] = stream["states"][:, 0]
    for trial in range(2):
        for step in range(stream["inputs"].shape[1]):
            z = model.lift(stream["states"][trial, step])
            un = model.u_normer.transform(stream["inputs"][trial, step:step + 1])[0]
            expected[trial, step + 1] = model.recover_state(model.A @ z + model.B @ un)
    assert np.allclose(arrays["fixed_dkuc_one_step"], expected, atol=1e-8)

    single_model, _, _ = _toy_replay()
    single_model.model.load_state_dict(initial)
    single_model.A = single_model.model.A.weight.detach().numpy().astype(np.float64)
    single_model.B = single_model.model.B.weight.detach().numpy().astype(np.float64)
    _, _, single = run_hao_dktv_replay(
        single_model, history, {key: value[1:2] for key, value in stream.items()},
        mode="full", batch_size=2, ridge_lambda=1e-3, loss_weight=0.5,
        online_epochs=3, online_lr=2e-2, online_weight_decay=0.0, grad_clip=5.0,
    )
    assert np.allclose(arrays["dktv_one_step"][1], single["dktv_one_step"][0], atol=1e-8)

    all_x = np.concatenate((history["states"][:, :-1].reshape(-1, 4),
                            stream["states"][1, :-1]), axis=0)
    all_y = np.concatenate((history["states"][:, 1:].reshape(-1, 4),
                            stream["states"][1, 1:]), axis=0)
    model.model.load_state_dict(final_network)
    with torch.no_grad():
        z = model.model.lift(torch.tensor(all_x, dtype=torch.float32)).numpy()
        zn = model.model.lift(torch.tensor(all_y, dtype=torch.float32)).numpy()
    direct = HaoDKTVState.from_history(z, zn, np.concatenate((
        history["inputs"].reshape(-1, 2), stream["inputs"][1]
    )), all_x, ridge_lambda=1e-3)
    for name in ("A", "B", "C", "gram_chi", "cross_chi", "gram_g", "cross_g"):
        assert np.allclose(getattr(final_state, name), getattr(direct, name), atol=1e-8)


def test_future_observation_mutation_does_not_change_past_predictions() -> None:
    import torch

    torch.manual_seed(47)
    first_model, history, stream = _toy_replay(seed=47)
    initial = {name: value.detach().clone() for name, value in first_model.model.state_dict().items()}
    second_model, _, _ = _toy_replay(seed=48)
    second_model.model.load_state_dict(initial)
    second_model.A = second_model.model.A.weight.detach().numpy().astype(np.float64)
    second_model.B = second_model.model.B.weight.detach().numpy().astype(np.float64)
    changed = {key: value[:1].copy() for key, value in stream.items()}
    changed["states"][:, 5:] += 100.0
    kwargs = dict(mode="frozen_encoder", batch_size=2, ridge_lambda=1e-3,
                  loss_weight=0.5, online_epochs=1, online_lr=1e-3,
                  online_weight_decay=0.0, grad_clip=5.0)
    _, _, original_arrays = run_hao_dktv_replay(
        first_model, history, {key: value[:1] for key, value in stream.items()}, **kwargs
    )
    _, _, changed_arrays = run_hao_dktv_replay(second_model, history, changed, **kwargs)
    assert np.array_equal(
        original_arrays["dktv_one_step"][:, :5], changed_arrays["dktv_one_step"][:, :5]
    )
