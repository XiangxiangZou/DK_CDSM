"""Causal replay and unified prediction evaluation for online Koopman models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from .accumulative_update import AccumulativeKoopmanUpdater
from .least_squares import direct_refit
from .selective_update import SelectiveWindowKoopmanUpdater
from .window_update import SlidingWindowKoopmanUpdater

if TYPE_CHECKING:
    from prediction.dkuc_prediction import DKUCModel


@dataclass
class AccumulativeReplay:
    batch_size: int
    one_step_prediction: np.ndarray
    A_by_step: np.ndarray
    B_by_step: np.ndarray
    model_version_by_step: np.ndarray
    update_history: list[dict[str, Any]]
    updater: AccumulativeKoopmanUpdater
    maximum_oracle_A_difference: float
    maximum_oracle_B_difference: float
    oracle_tolerance_passed: bool
    sample_count_monotonic: bool
    all_updates_finite: bool
    pending_sample_count: int
    rejected_sample_count: int
    invalid_batch_policy: str
    updater_statistics_memory_bytes: int
    oracle_history_memory_bytes_initial: int
    oracle_history_memory_bytes_final: int


@dataclass
class WindowReplay:
    method: str
    batch_size: int
    window_size: int
    epsilon: float | None
    reject_buffer_policy: str
    one_step_prediction: np.ndarray
    A_by_step: np.ndarray
    B_by_step: np.ndarray
    model_version_by_step: np.ndarray
    update_history: list[dict[str, Any]]
    updater: SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater
    maximum_oracle_A_difference: float
    maximum_oracle_B_difference: float
    oracle_tolerance_passed: bool
    all_updates_finite: bool
    pending_sample_count: int
    rejected_sample_count: int
    skipped_sample_count: int
    window_memory_bytes_initial: int
    window_memory_bytes_final: int
    window_memory_constant: bool
    window_boundaries_replayable: bool


def artifact_fingerprint(artifact_dir: str | Path) -> str:
    """Hash the fixed encoder, normalizers, and architecture metadata."""
    root = Path(artifact_dir)
    digest = hashlib.sha256()
    for name in ("encoder.pt", "normalizers.json", "model_config.json"):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def batch_lift(model: DKUCModel, states: np.ndarray) -> np.ndarray:
    """Lift arbitrary physical-state batches with the frozen encoder."""
    values = np.asarray(states, dtype=np.float64)
    flat = values.reshape(-1, model.state_dim)
    normalized = model.x_normer.transform(flat).astype(np.float32)
    with model._torch.no_grad():
        lifted = model.model.lift(model._torch.from_numpy(normalized).to(model.device))
    return lifted.cpu().numpy().astype(np.float64).reshape(*values.shape[:-1], -1)


def normalized_pairs(
    model: DKUCModel,
    states: np.ndarray,
    inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned ``z_k, z_k+1, u_norm,k`` arrays."""
    state_values = np.asarray(states, dtype=np.float64)
    input_values = np.asarray(inputs, dtype=np.float64)
    if state_values.ndim != 3 or input_values.ndim != 3:
        raise ValueError("states and inputs must have trajectory/time/feature axes")
    if state_values.shape[:2] != (input_values.shape[0], input_values.shape[1] + 1):
        raise ValueError("states and inputs are not transition-aligned")
    z = batch_lift(model, state_values)
    u = model.u_normer.transform(input_values.reshape(-1, model.control_dim)).reshape(
        input_values.shape
    )
    return z[:, :-1], z[:, 1:], u


def _recover_batch(model: DKUCModel, latent: np.ndarray) -> np.ndarray:
    normalized = np.asarray(latent, dtype=np.float64)[..., : model.state_dim]
    shape = normalized.shape
    return model.x_normer.inverse(normalized.reshape(-1, model.state_dim)).reshape(shape)


def _predict_batch(
    model: DKUCModel,
    A: np.ndarray,
    B: np.ndarray,
    z_current: np.ndarray,
    u_normalized: np.ndarray,
) -> np.ndarray:
    latent_next = np.asarray(z_current) @ np.asarray(A).T + np.asarray(u_normalized) @ np.asarray(B).T
    return _recover_batch(model, latent_next)


def run_accumulative_replay(
    model: DKUCModel,
    train_data: dict[str, np.ndarray],
    stream_data: dict[str, np.ndarray],
    *,
    batch_size: int,
    ridge_lambda: float,
    encoder_fingerprint: str,
    oracle_tolerance: float,
    invalid_batch_policy: str = "discard_invalid_batch",
) -> AccumulativeReplay:
    """Causally predict, then add observed snapshots in chronological batches."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if invalid_batch_policy != "discard_invalid_batch":
        raise ValueError("only discard_invalid_batch is supported")
    train_z, train_next, train_u = normalized_pairs(
        model,
        train_data["states"],
        train_data["inputs"],
    )
    stream_z, stream_next, stream_u = normalized_pairs(
        model, stream_data["states"], stream_data["inputs"]
    )
    latent_dim = stream_z.shape[-1]
    train_current_flat = train_z.reshape(-1, latent_dim)
    train_next_flat = train_next.reshape(-1, latent_dim)
    train_u_flat = train_u.reshape(-1, model.control_dim)
    updater = AccumulativeKoopmanUpdater.from_history(
        train_current_flat,
        train_next_flat,
        train_u_flat,
        A0=model.A,
        B0=model.B,
        ridge_lambda=ridge_lambda,
        encoder_fingerprint=encoder_fingerprint,
        affine_constant=bool(model.config.include_constant),
    )
    states = np.asarray(stream_data["states"], dtype=np.float64)
    _, steps = stream_u.shape[:2]
    prediction = np.zeros_like(states)
    prediction[:, 0] = states[:, 0]
    A_by_step = np.zeros((steps, latent_dim, latent_dim), dtype=np.float64)
    B_by_step = np.zeros((steps, latent_dim, model.control_dim), dtype=np.float64)
    versions = np.zeros(steps, dtype=np.int64)
    pending_z = np.empty((0, latent_dim), dtype=np.float64)
    pending_next = np.empty((0, latent_dim), dtype=np.float64)
    pending_u = np.empty((0, model.control_dim), dtype=np.float64)
    history_z, history_next, history_u = (
        train_current_flat.copy(), train_next_flat.copy(), train_u_flat.copy()
    )
    oracle_memory_initial = int(history_z.nbytes + history_next.nbytes + history_u.nbytes)
    update_history: list[dict[str, Any]] = []
    online_consumed = 0
    rejected_sample_count = 0
    for time_step in range(steps):
        A_by_step[time_step], B_by_step[time_step] = updater.A, updater.B
        versions[time_step] = updater.model_version
        prediction[:, time_step + 1] = _predict_batch(
            model, updater.A, updater.B, stream_z[:, time_step], stream_u[:, time_step]
        )
        pending_z = np.concatenate([pending_z, stream_z[:, time_step]], axis=0)
        pending_next = np.concatenate([pending_next, stream_next[:, time_step]], axis=0)
        pending_u = np.concatenate([pending_u, stream_u[:, time_step]], axis=0)
        while pending_z.shape[0] >= batch_size:
            batch_z, pending_z = pending_z[:batch_size], pending_z[batch_size:]
            batch_next, pending_next = pending_next[:batch_size], pending_next[batch_size:]
            batch_u, pending_u = pending_u[:batch_size], pending_u[batch_size:]
            online_consumed += batch_size
            result, candidate = updater.update(
                batch_z, batch_next, batch_u, encoder_fingerprint=encoder_fingerprint
            )
            oracle = None
            oracle_time = 0.0
            A_difference = B_difference = None
            if result.accepted:
                history_z = np.concatenate([history_z, batch_z], axis=0)
                history_next = np.concatenate([history_next, batch_next], axis=0)
                history_u = np.concatenate([history_u, batch_u], axis=0)
                oracle_started = perf_counter()
                oracle = direct_refit(
                    history_z,
                    history_next,
                    history_u,
                    ridge_lambda=ridge_lambda,
                    affine_constant=bool(model.config.include_constant),
                )
                oracle_time = perf_counter() - oracle_started
                if candidate is None:
                    raise RuntimeError("accepted update did not return a candidate")
                A_difference = float(np.max(np.abs(candidate.A - oracle.A)))
                B_difference = float(np.max(np.abs(candidate.B - oracle.B)))
            else:
                rejected_sample_count += int(result.batch_sample_count)
            record = result.to_dict()
            record.update(
                {
                    "attempt_index": len(update_history) + 1,
                    "time_step": int(time_step),
                    "online_sample_count": int(online_consumed),
                    "accepted_online_sample_count": int(
                        updater.sample_count - train_current_flat.shape[0]
                    ),
                    "invalid_batch_policy": invalid_batch_policy,
                    "batch_disposition": (
                        "accepted_into_statistics_and_oracle_history"
                        if result.accepted else "discarded_invalid_batch"
                    ),
                    "oracle_check_performed": oracle is not None,
                    "oracle_refit_time_s": float(oracle_time),
                    "oracle_A_max_abs_difference": A_difference,
                    "oracle_B_max_abs_difference": B_difference,
                    "oracle_rank": oracle.diagnostics.rank if oracle else None,
                    "oracle_minimum_singular_value": (
                        oracle.diagnostics.minimum_singular_value if oracle else None
                    ),
                    "oracle_condition_number": oracle.diagnostics.condition_number if oracle else None,
                    "oracle_regularized_condition_number": (
                        oracle.diagnostics.regularized_condition_number if oracle else None
                    ),
                    "updater_statistics_memory_bytes": updater.statistics_memory_bytes,
                    "oracle_history_memory_bytes": int(
                        history_z.nbytes + history_next.nbytes + history_u.nbytes
                    ),
                }
            )
            update_history.append(record)
    A_differences = [
        float(record["oracle_A_max_abs_difference"])
        for record in update_history if record["oracle_check_performed"]
    ]
    B_differences = [
        float(record["oracle_B_max_abs_difference"])
        for record in update_history if record["oracle_check_performed"]
    ]
    accepted_counts = [
        int(record["cumulative_sample_count"]) for record in update_history if record["accepted"]
    ]
    return AccumulativeReplay(
        batch_size=int(batch_size),
        one_step_prediction=prediction,
        A_by_step=A_by_step,
        B_by_step=B_by_step,
        model_version_by_step=versions,
        update_history=update_history,
        updater=updater,
        maximum_oracle_A_difference=max(A_differences, default=0.0),
        maximum_oracle_B_difference=max(B_differences, default=0.0),
        oracle_tolerance_passed=bool(
            max(A_differences, default=0.0) <= oracle_tolerance
            and max(B_differences, default=0.0) <= oracle_tolerance
        ),
        sample_count_monotonic=all(
            right > left for left, right in zip(accepted_counts, accepted_counts[1:])
        ),
        all_updates_finite=all(
            record["diagnostics"] is not None and record["diagnostics"]["finite"]
            for record in update_history if record["accepted"]
        ),
        pending_sample_count=int(pending_z.shape[0]),
        rejected_sample_count=int(rejected_sample_count),
        invalid_batch_policy=invalid_batch_policy,
        updater_statistics_memory_bytes=updater.statistics_memory_bytes,
        oracle_history_memory_bytes_initial=oracle_memory_initial,
        oracle_history_memory_bytes_final=int(
            history_z.nbytes + history_next.nbytes + history_u.nbytes
        ),
    )


def run_window_replay(
    model: DKUCModel,
    train_data: dict[str, np.ndarray],
    stream_data: dict[str, np.ndarray],
    *,
    window_size: int,
    batch_size: int,
    ridge_lambda: float,
    encoder_fingerprint: str,
    oracle_tolerance: float,
    selective: bool = False,
    epsilon: float = 0.0,
    reject_buffer_policy: str = "discard_on_reject",
    low_dim_condition_limit: float = 1e12,
    window_condition_limit: float = 1e12,
) -> WindowReplay:
    """Causally replay one fixed-window method on the shared Plan 01 stream."""
    if int(window_size) <= 0 or int(batch_size) <= 0:
        raise ValueError("window_size and batch_size must be positive")
    train_z, train_next, train_u = normalized_pairs(model, train_data["states"], train_data["inputs"])
    stream_z, stream_next, stream_u = normalized_pairs(
        model, stream_data["states"], stream_data["inputs"]
    )
    latent_dim = int(stream_z.shape[-1])
    train_current = train_z.reshape(-1, latent_dim)
    train_target = train_next.reshape(-1, latent_dim)
    train_input = train_u.reshape(-1, model.control_dim)
    if train_current.shape[0] < int(window_size):
        raise ValueError("training history is shorter than window_size")
    base = SlidingWindowKoopmanUpdater(
        train_current[-int(window_size) :],
        train_target[-int(window_size) :],
        train_input[-int(window_size) :],
        A0=model.A,
        B0=model.B,
        batch_size=int(batch_size),
        ridge_lambda=float(ridge_lambda),
        encoder_fingerprint=encoder_fingerprint,
        sample_ids=np.arange(-int(window_size), 0, dtype=np.int64),
        affine_constant=bool(model.config.include_constant),
        oracle_tolerance=float(oracle_tolerance),
        low_dim_condition_limit=float(low_dim_condition_limit),
        window_condition_limit=float(window_condition_limit),
    )
    updater: SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater
    if selective:
        updater = SelectiveWindowKoopmanUpdater(
            base,
            epsilon=float(epsilon),
            reject_buffer_policy=reject_buffer_policy,
        )
        method = "otvdkl_selective"
    else:
        updater = base
        method = "otvdkl_window"

    states = np.asarray(stream_data["states"], dtype=np.float64)
    trajectory_count, steps = stream_u.shape[:2]
    prediction = np.zeros_like(states)
    prediction[:, 0] = states[:, 0]
    A_by_step = np.zeros((steps, latent_dim, latent_dim), dtype=np.float64)
    B_by_step = np.zeros((steps, latent_dim, model.control_dim), dtype=np.float64)
    versions = np.zeros(steps, dtype=np.int64)
    pending_z = np.empty((0, latent_dim), dtype=np.float64)
    pending_next = np.empty((0, latent_dim), dtype=np.float64)
    pending_u = np.empty((0, model.control_dim), dtype=np.float64)
    pending_ids = np.empty(0, dtype=np.int64)
    update_history: list[dict[str, Any]] = []
    next_sample_id = 0
    rejected_sample_count = 0
    skipped_sample_count = 0
    memory_values = [int(updater.memory_bytes)]
    boundaries_ok = True

    for time_step in range(steps):
        A_by_step[time_step] = updater.A
        B_by_step[time_step] = updater.B
        versions[time_step] = updater.model_version
        prediction[:, time_step + 1] = _predict_batch(
            model, updater.A, updater.B, stream_z[:, time_step], stream_u[:, time_step]
        )
        ids = np.arange(next_sample_id, next_sample_id + trajectory_count, dtype=np.int64)
        next_sample_id += trajectory_count
        pending_z = np.concatenate([pending_z, stream_z[:, time_step]], axis=0)
        pending_next = np.concatenate([pending_next, stream_next[:, time_step]], axis=0)
        pending_u = np.concatenate([pending_u, stream_u[:, time_step]], axis=0)
        pending_ids = np.concatenate([pending_ids, ids])
        while pending_z.shape[0] >= int(batch_size):
            batch_z, pending_z = pending_z[:batch_size], pending_z[batch_size:]
            batch_next, pending_next = pending_next[:batch_size], pending_next[batch_size:]
            batch_u, pending_u = pending_u[:batch_size], pending_u[batch_size:]
            batch_ids, pending_ids = pending_ids[:batch_size], pending_ids[batch_size:]
            before_ids = updater.sample_ids.copy()
            record, candidate = updater.update(
                batch_z,
                batch_next,
                batch_u,
                sample_ids=batch_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            if record.status in ("rejected", "failed_numerical"):
                rejected_sample_count += int(record.batch_sample_count)
            if record.status == "skipped_threshold":
                skipped_sample_count += int(record.batch_sample_count)
            if record.window_advanced:
                expected = np.concatenate([before_ids[batch_size:], batch_ids])
                boundaries_ok = boundaries_ok and np.array_equal(updater.sample_ids, expected)
            else:
                boundaries_ok = boundaries_ok and np.array_equal(updater.sample_ids, before_ids)
            boundaries_ok = boundaries_ok and (
                updater.sample_ids.size == int(window_size)
                and np.unique(updater.sample_ids).size == int(window_size)
            )
            values = record.to_dict()
            values.update(
                {
                    "time_step": int(time_step),
                    "online_sample_count": int(batch_ids[-1] + 1),
                    "pending_sample_count_after_attempt": int(pending_ids.size),
                    "buffer_disposition": (
                        "advanced_window"
                        if record.window_advanced
                        else "discarded_batch_and_retained_window"
                    ),
                    "oracle_check_performed": bool(candidate is not None),
                    "window_memory_bytes": int(updater.memory_bytes),
                }
            )
            update_history.append(values)
            memory_values.append(int(updater.memory_bytes))

    A_differences = [
        float(record["candidate_A_max_abs_difference"])
        for record in update_history
        if record["candidate_A_max_abs_difference"] is not None
    ]
    B_differences = [
        float(record["candidate_B_max_abs_difference"])
        for record in update_history
        if record["candidate_B_max_abs_difference"] is not None
    ]
    finite = all(
        record["diagnostics"] is not None and record["diagnostics"]["finite"]
        for record in update_history
        if record["status"] in ("accepted", "rejected", "skipped_threshold")
    )
    return WindowReplay(
        method=method,
        batch_size=int(batch_size),
        window_size=int(window_size),
        epsilon=float(epsilon) if selective else None,
        reject_buffer_policy=reject_buffer_policy if selective else "not_applicable",
        one_step_prediction=prediction,
        A_by_step=A_by_step,
        B_by_step=B_by_step,
        model_version_by_step=versions,
        update_history=update_history,
        updater=updater,
        maximum_oracle_A_difference=max(A_differences, default=0.0),
        maximum_oracle_B_difference=max(B_differences, default=0.0),
        oracle_tolerance_passed=bool(
            max(A_differences, default=0.0) <= float(oracle_tolerance)
            and max(B_differences, default=0.0) <= float(oracle_tolerance)
        ),
        all_updates_finite=bool(finite),
        pending_sample_count=int(pending_ids.size),
        rejected_sample_count=int(rejected_sample_count),
        skipped_sample_count=int(skipped_sample_count),
        window_memory_bytes_initial=int(memory_values[0]),
        window_memory_bytes_final=int(memory_values[-1]),
        window_memory_constant=len(set(memory_values)) == 1,
        window_boundaries_replayable=bool(boundaries_ok),
    )
def prediction_metrics(true_states: np.ndarray, predicted_states: np.ndarray) -> dict[str, Any]:
    """Return the shared physical-state error contract."""
    true_values = np.asarray(true_states, dtype=np.float64)
    predicted = np.asarray(predicted_states, dtype=np.float64)
    if true_values.shape != predicted.shape or true_values.ndim != 3:
        raise ValueError("true and predicted states must be aligned 3-D arrays")
    error = predicted[:, 1:] - true_values[:, 1:]
    finite = np.isfinite(error)
    clean = np.where(finite, error, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        return {
            "total_rmse": float(np.sqrt(np.nanmean(clean * clean))),
            "rmse_by_state": np.sqrt(np.nanmean(clean * clean, axis=(0, 1))).tolist(),
            "state_labels": ["qa", "qb", "dqa", "dqb"],
            "max_abs_error": float(np.nanmax(np.abs(clean))),
            "nonfinite_prediction_count": int(np.count_nonzero(~np.isfinite(predicted))),
            "sample_count": int(error.shape[0] * error.shape[1]),
        }


def _rollout_windows(
    model: DKUCModel,
    states: np.ndarray,
    inputs: np.ndarray,
    A_by_step: np.ndarray,
    B_by_step: np.ndarray,
    *,
    horizon: int,
    starts: list[int],
    truth_states: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    normalized_inputs = model.u_normer.transform(inputs.reshape(-1, model.control_dim)).reshape(
        inputs.shape
    )
    truth_values = states if truth_states is None else np.asarray(truth_states, dtype=np.float64)
    if truth_values.shape != states.shape:
        raise ValueError("truth_states must match observed states")
    for trajectory in range(states.shape[0]):
        for start in starts:
            stop = start + horizon
            z = model.lift(states[trajectory, start])
            values = np.zeros((horizon + 1, model.state_dim), dtype=np.float64)
            values[0] = states[trajectory, start]
            A = A_by_step[start]
            B = B_by_step[start]
            for offset in range(horizon):
                z = A @ z + B @ normalized_inputs[trajectory, start + offset]
                values[offset + 1] = model.recover_state(z)
            truth.append(truth_values[trajectory, start : stop + 1])
            prediction.append(values)
    return np.stack(truth), np.stack(prediction)


def evaluate_methods(
    model: DKUCModel,
    stream_data: dict[str, np.ndarray],
    replays: dict[str, AccumulativeReplay],
    *,
    rollout_horizons: list[int],
    window_stride: int,
    stage_definitions: list[dict[str, Any]],
    stage_rollout_horizon: int,
    truth_states: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate fixed and online models with the same windows and metrics."""
    states = np.asarray(stream_data["states"], dtype=np.float64)
    truth_values = states if truth_states is None else np.asarray(truth_states, dtype=np.float64)
    if truth_values.shape != states.shape:
        raise ValueError("truth_states must match stream_data states")
    inputs = np.asarray(stream_data["inputs"], dtype=np.float64)
    steps = inputs.shape[1]
    fixed_A = np.broadcast_to(model.A, (steps, *model.A.shape)).copy()
    fixed_B = np.broadcast_to(model.B, (steps, *model.B.shape)).copy()
    fixed_z, _, fixed_u = normalized_pairs(model, states, inputs)
    fixed_prediction = np.zeros_like(states)
    fixed_prediction[:, 0] = states[:, 0]
    for time_step in range(steps):
        fixed_prediction[:, time_step + 1] = _predict_batch(
            model,
            model.A,
            model.B,
            fixed_z[:, time_step],
            fixed_u[:, time_step],
        )

    matrices = {"fixed_dko": (fixed_A, fixed_B)}
    predictions = {"fixed_dko": fixed_prediction}
    for method, replay in replays.items():
        matrices[method] = (replay.A_by_step, replay.B_by_step)
        predictions[method] = replay.one_step_prediction

    one_step = {
        method: prediction_metrics(truth_values, values)
        for method, values in predictions.items()
    }
    rollout: dict[str, Any] = {method: {} for method in matrices}
    for horizon in rollout_horizons:
        starts = list(range(0, steps - int(horizon) + 1, int(window_stride)))
        for method, (A_values, B_values) in matrices.items():
            window_truth, predicted = _rollout_windows(
                model,
                states,
                inputs,
                A_values,
                B_values,
                horizon=int(horizon),
                starts=starts,
                truth_states=truth_values,
            )
            rollout[method][str(horizon)] = prediction_metrics(window_truth, predicted)

    segmented: dict[str, Any] = {method: {} for method in matrices}
    for stage in stage_definitions:
        starts = list(
            range(
                int(stage["start_step"]),
                int(stage["end_step"]) - int(stage_rollout_horizon) + 1,
                int(window_stride),
            )
        )
        if not starts:
            raise ValueError(f"stage {stage['name']} is too short for rollout evaluation")
        for method, (A_values, B_values) in matrices.items():
            window_truth, predicted = _rollout_windows(
                model,
                states,
                inputs,
                A_values,
                B_values,
                horizon=int(stage_rollout_horizon),
                starts=starts,
                truth_states=truth_values,
            )
            metrics = prediction_metrics(window_truth, predicted)
            metrics.update(
                {
                    "start_step": int(stage["start_step"]),
                    "end_step": int(stage["end_step"]),
                    "disturbance_scale": float(stage["scale"]),
                    "rollout_horizon": int(stage_rollout_horizon),
                }
            )
            segmented[method][str(stage["name"])] = metrics

    per_step_rmse = {
        method: np.sqrt(
            np.mean((values[:, 1:] - truth_values[:, 1:]) ** 2, axis=(0, 2))
        )
        for method, values in predictions.items()
    }
    arrays: dict[str, np.ndarray] = {
        "states_true": truth_values,
        "states_observed": states,
        "inputs": inputs,
        "t": np.asarray(stream_data["t"]),
    }
    for method, values in predictions.items():
        arrays[f"{method}_one_step_prediction"] = values
        arrays[f"{method}_one_step_rmse_by_step"] = per_step_rmse[method]
    return {"one_step": one_step, "rollout": rollout, "segmented": segmented}, arrays


def update_summary(replay: AccumulativeReplay) -> dict[str, Any]:
    """Aggregate timing, memory, conditioning, and oracle diagnostics."""
    records = replay.update_history
    update_times = np.asarray([record["update_time_s"] for record in records], dtype=np.float64)
    accepted_records = [record for record in records if record["accepted"]]
    oracle_times = np.asarray(
        [record["oracle_refit_time_s"] for record in records if record["oracle_check_performed"]],
        dtype=np.float64,
    )
    conditions = np.asarray(
        [record["diagnostics"]["condition_number"] for record in accepted_records],
        dtype=np.float64,
    )
    regularized_conditions = np.asarray(
        [record["diagnostics"]["regularized_condition_number"] for record in accepted_records],
        dtype=np.float64,
    )
    mean = lambda values: float(np.mean(values)) if values.size else 0.0
    maximum = lambda values: float(np.max(values)) if values.size else 0.0
    return {
        "batch_size": replay.batch_size,
        "update_count": len(records),
        "accepted_count": int(sum(record["accepted"] for record in records)),
        "failed_count": int(sum(not record["accepted"] for record in records)),
        "initial_sample_count": int(records[0]["previous_sample_count"]) if records else 0,
        "final_sample_count": int(replay.updater.sample_count),
        "pending_sample_count": replay.pending_sample_count,
        "rejected_sample_count": replay.rejected_sample_count,
        "invalid_batch_policy": replay.invalid_batch_policy,
        "model_version": int(replay.updater.model_version),
        "sample_count_monotonic": replay.sample_count_monotonic,
        "all_updates_finite": replay.all_updates_finite,
        "maximum_oracle_A_difference": replay.maximum_oracle_A_difference,
        "maximum_oracle_B_difference": replay.maximum_oracle_B_difference,
        "oracle_tolerance_passed": replay.oracle_tolerance_passed,
        "mean_recursive_update_time_s": mean(update_times),
        "p95_recursive_update_time_s": (
            float(np.percentile(update_times, 95)) if update_times.size else 0.0
        ),
        "maximum_recursive_update_time_s": maximum(update_times),
        "mean_direct_refit_oracle_time_s": mean(oracle_times),
        "maximum_direct_refit_oracle_time_s": maximum(oracle_times),
        "maximum_condition_number": maximum(conditions),
        "maximum_regularized_condition_number": maximum(regularized_conditions),
        "updater_statistics_memory_bytes": replay.updater_statistics_memory_bytes,
        "oracle_history_memory_bytes_initial": replay.oracle_history_memory_bytes_initial,
        "oracle_history_memory_bytes_final": replay.oracle_history_memory_bytes_final,
    }


def window_update_summary(replay: WindowReplay) -> dict[str, Any]:
    """Aggregate state counts, timing, conditioning, oracle, and memory evidence."""
    records = replay.update_history
    update_times = np.asarray([record["update_time_s"] for record in records], dtype=np.float64)
    candidates = [record for record in records if record["diagnostics"] is not None]
    recursive_times = np.asarray(
        [record["recursive_candidate_time_s"] for record in candidates], dtype=np.float64
    )
    direct_times = np.asarray(
        [record["direct_refit_oracle_time_s"] for record in candidates], dtype=np.float64
    )
    fallback_times = np.asarray(
        [record["fallback_time_s"] for record in candidates], dtype=np.float64
    )
    conditions = np.asarray(
        [record["diagnostics"]["condition_number"] for record in candidates], dtype=np.float64
    )
    minimum_singular_values = np.asarray(
        [record["diagnostics"]["minimum_singular_value"] for record in candidates],
        dtype=np.float64,
    )
    ranks = np.asarray(
        [record["diagnostics"]["rank"] for record in candidates], dtype=np.int64
    )

    def statistic(values: np.ndarray, name: str) -> float:
        if not values.size:
            return 0.0
        if name == "mean":
            return float(np.mean(values))
        if name == "maximum":
            return float(np.max(values))
        if name == "minimum":
            return float(np.min(values))
        return float(np.percentile(values, float(name)))

    counts = {
        status: int(sum(record["status"] == status for record in records))
        for status in ("accepted", "rejected", "skipped_threshold", "failed_numerical")
    }
    return {
        "method": replay.method,
        "window_size": replay.window_size,
        "batch_size": replay.batch_size,
        "epsilon": replay.epsilon,
        "reject_buffer_policy": replay.reject_buffer_policy,
        "candidate_count": len(candidates),
        **{f"{name}_count": value for name, value in counts.items()},
        "model_version": int(replay.updater.model_version),
        "window_version": int(replay.updater.window_version),
        "pending_sample_count": replay.pending_sample_count,
        "rejected_sample_count": replay.rejected_sample_count,
        "skipped_sample_count": replay.skipped_sample_count,
        "all_updates_finite": replay.all_updates_finite,
        "window_boundaries_replayable": replay.window_boundaries_replayable,
        "maximum_oracle_A_difference": replay.maximum_oracle_A_difference,
        "maximum_oracle_B_difference": replay.maximum_oracle_B_difference,
        "maximum_raw_recursive_A_difference": max(
            (float(record["recursive_A_max_abs_difference"]) for record in candidates),
            default=0.0,
        ),
        "maximum_raw_recursive_B_difference": max(
            (float(record["recursive_B_max_abs_difference"]) for record in candidates),
            default=0.0,
        ),
        "oracle_tolerance_passed": replay.oracle_tolerance_passed,
        "mean_update_time_s": statistic(update_times, "mean"),
        "p95_update_time_s": statistic(update_times, "95"),
        "p99_update_time_s": statistic(update_times, "99"),
        "maximum_update_time_s": statistic(update_times, "maximum"),
        "timing_contract": {
            "recursive_candidate_excludes_direct_oracle": True,
            "direct_oracle_excluded_from_deployment_path": True,
            "total_includes_recursive_direct_oracle_decision_and_fallback": True,
        },
        "mean_recursive_candidate_time_s": statistic(recursive_times, "mean"),
        "p95_recursive_candidate_time_s": statistic(recursive_times, "95"),
        "p99_recursive_candidate_time_s": statistic(recursive_times, "99"),
        "maximum_recursive_candidate_time_s": statistic(recursive_times, "maximum"),
        "mean_direct_refit_oracle_time_s": statistic(direct_times, "mean"),
        "p95_direct_refit_oracle_time_s": statistic(direct_times, "95"),
        "p99_direct_refit_oracle_time_s": statistic(direct_times, "99"),
        "maximum_direct_refit_oracle_time_s": statistic(direct_times, "maximum"),
        "mean_fallback_time_s": statistic(fallback_times, "mean"),
        "p95_fallback_time_s": statistic(fallback_times, "95"),
        "p99_fallback_time_s": statistic(fallback_times, "99"),
        "maximum_fallback_time_s": statistic(fallback_times, "maximum"),
        "maximum_condition_number": statistic(conditions, "maximum"),
        "minimum_singular_value": statistic(minimum_singular_values, "minimum"),
        "minimum_rank": int(np.min(ranks)) if ranks.size else 0,
        "direct_refit_fallback_count": int(
            sum(record["recursive_path"] == "direct_refit_fallback" for record in candidates)
        ),
        "window_memory_bytes_initial": replay.window_memory_bytes_initial,
        "window_memory_bytes_final": replay.window_memory_bytes_final,
        "window_memory_constant": replay.window_memory_constant,
    }


def save_comparison_figures(
    result_dir: str | Path,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
    update_summaries: dict[str, dict[str, Any]],
    display_names: dict[str, str] | None = None,
) -> list[str]:
    """Save compact Plan 02 comparison figures and return relative paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(result_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = list(metrics["one_step"])
    labels = display_names or {}

    figure, axis = plt.subplots(figsize=(10, 5))
    for method in methods:
        axis.plot(
            arrays[f"{method}_one_step_rmse_by_step"], label=labels.get(method, method)
        )
    axis.set_xlabel("step")
    axis.set_ylabel("physical-state one-step RMSE")
    axis.set_title("Fixed DKO vs accumulative DKTV")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    error_path = figures / "one_step_rmse_by_step.png"
    figure.savefig(error_path, dpi=180)
    plt.close(figure)

    horizons = list(next(iter(metrics["rollout"].values())))
    positions = np.arange(len(horizons), dtype=np.float64)
    width = 0.8 / len(methods)
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(methods):
        values = [metrics["rollout"][method][horizon]["total_rmse"] for horizon in horizons]
        axis.bar(
            positions + index * width,
            values,
            width=width,
            label=labels.get(method, method),
        )
    axis.set_xticks(positions + width * (len(methods) - 1) / 2, horizons)
    axis.set_xlabel("rollout horizon")
    axis.set_ylabel("physical-state RMSE")
    axis.set_title("Causal model snapshot rollout comparison")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    rollout_path = figures / "rollout_rmse_by_horizon.png"
    figure.savefig(rollout_path, dpi=180)
    plt.close(figure)

    names = list(update_summaries)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    display_update_names = [labels.get(name, name) for name in names]
    axes[0].bar(
        display_update_names,
        [update_summaries[name]["mean_recursive_update_time_s"] for name in names],
    )
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Mean recursive update time")
    axes[1].bar(
        display_update_names,
        [update_summaries[name]["maximum_oracle_A_difference"] for name in names],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max abs difference")
    axes[1].set_title("Recursive vs direct-refit A")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    update_path = figures / "update_diagnostics.png"
    figure.savefig(update_path, dpi=180)
    plt.close(figure)
    return [path.relative_to(output).as_posix() for path in (error_path, rollout_path, update_path)]
