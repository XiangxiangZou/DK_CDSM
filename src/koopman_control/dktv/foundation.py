"""Freeze and evaluate the common fixed-DKO artifact used by all methods."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from koopman_control.dktv.config import stage_bounds
from prediction.common import (
    fit_state_input_normalizers,
    plot_prediction_errors,
    plot_prediction_states,
    predict_one_step,
    save_dataset,
    save_json,
    save_normalizers,
    set_seed,
)
from prediction.dkuc_prediction import DKUCConfig, DKUCModel, fit_dkuc


METHODS_SHARING_ARTIFACT = (
    "fixed_dko",
    "dktv_accumulative",
    "otvdkl_window",
    "otvdkl_selective",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_contract_check(
    model: DKUCModel,
    x_phys: np.ndarray,
    u_phys: np.ndarray,
) -> dict[str, Any]:
    """Verify the frozen physical/normalized/latent coordinate transformations."""
    state = np.asarray(x_phys, dtype=np.float64).reshape(model.state_dim)
    control = np.asarray(u_phys, dtype=np.float64).reshape(model.control_dim)
    x_normalized = model.x_normer.transform(state.reshape(1, -1))[0]
    u_normalized = model.u_normer.transform(control.reshape(1, -1))[0]
    z = model.lift(state)
    z_next_matrix = model.A @ z + model.B @ u_normalized
    z_next_api = model.step_latent(z, control)
    C0 = np.zeros((model.state_dim, model.A.shape[0]), dtype=np.float64)
    C0[:, : model.state_dim] = np.eye(model.state_dim, dtype=np.float64)
    x_normalized_next = C0 @ z_next_matrix
    x_physical_next = model.x_normer.inverse(x_normalized_next.reshape(1, -1))[0]
    x_physical_api = model.recover_state(z_next_api)
    checks = {
        "lift_state_prefix_matches_normalized_state": bool(
            np.allclose(z[: model.state_dim], x_normalized, rtol=1e-6, atol=1e-7)
        ),
        "latent_step_matches_normalized_input_formula": bool(
            np.allclose(z_next_api, z_next_matrix, rtol=1e-10, atol=1e-10)
        ),
        "physical_recovery_matches_C0_formula": bool(
            np.allclose(x_physical_api, x_physical_next, rtol=1e-10, atol=1e-10)
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "state_coordinate": "normalized",
        "input_coordinate": "normalized_applied_torque",
        "x_physical": state.tolist(),
        "x_normalized": x_normalized.tolist(),
        "u_physical": control.tolist(),
        "u_normalized": u_normalized.tolist(),
        "z_shape": list(z.shape),
        "z_next_shape": list(z_next_matrix.shape),
        "x_physical_next": x_physical_next.tolist(),
    }


def _batch_lift(model: DKUCModel, states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64).reshape(-1, model.state_dim)
    normalized = model.x_normer.transform(values).astype(np.float32)
    with model._torch.no_grad():
        lifted = model.model.lift(model._torch.from_numpy(normalized).to(model.device))
    return lifted.cpu().numpy().astype(np.float64)


def _ridge_refit(
    artifact_dir: Path,
    train_data: dict[str, np.ndarray],
    ridge_lambda: float,
    device: str,
) -> dict[str, float]:
    """Refit only A/B while keeping the trained encoder and normalizers fixed."""
    import torch

    model = DKUCModel(artifact_dir, device)
    states = np.asarray(train_data["states"], dtype=np.float64)
    inputs = np.asarray(train_data["inputs"], dtype=np.float64)
    z_current = _batch_lift(model, states[:, :-1])
    z_next = _batch_lift(model, states[:, 1:])
    u_normalized = model.u_normer.transform(inputs.reshape(-1, model.control_dim))
    regressor = np.concatenate([z_current, u_normalized], axis=1)
    gram = regressor.T @ regressor
    regularizer = float(ridge_lambda) * np.eye(gram.shape[0], dtype=np.float64)
    theta = np.linalg.solve(gram + regularizer, regressor.T @ z_next)
    latent_dim = model.A.shape[0]
    A0 = theta[:latent_dim].T
    B0 = theta[latent_dim:].T
    if model.config.include_constant:
        A0[-1] = 0.0
        A0[-1, -1] = 1.0
        B0[-1] = 0.0

    state_dict = model.model.state_dict()
    state_dict["A.weight"] = torch.from_numpy(A0.astype(np.float32))
    state_dict["B.weight"] = torch.from_numpy(B0.astype(np.float32))
    torch.save(state_dict, artifact_dir / "best_dkuc.pt")
    predicted = regressor @ np.concatenate([A0, B0], axis=1).T
    residual = predicted - z_next
    return {
        "ridge_lambda": float(ridge_lambda),
        "latent_fit_rmse": float(np.sqrt(np.mean(residual * residual))),
        "latent_fit_max_abs_error": float(np.max(np.abs(residual))),
    }


def train_and_freeze_initial_model(
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
    config: dict[str, Any],
    artifact_dir: str | Path,
    training_dataset: str,
    validation_dataset: str,
    device: str,
) -> dict[str, Any]:
    """Train one DKUC encoder and freeze the unique Plan 01 initial artifact."""
    import torch

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(int(config["seed"]))
    save_dataset(output / "dataset_train.npz", train_data)
    save_dataset(output / "dataset_val.npz", validation_data)
    x_normalizer, u_normalizer = fit_state_input_normalizers(train_data)
    save_normalizers(output / "normalizers.json", x_normalizer, u_normalizer)
    training_cfg = config["training"]
    profile = config["profile"]
    dkuc_config = DKUCConfig(
        lift_dim=int(config["encoder_output_dim"]),
        hidden=tuple(int(value) for value in training_cfg["hidden"]),
        activation=str(training_cfg["activation"]),
        bound_lift=float(training_cfg["bound_lift"]),
        window=int(training_cfg["window"]),
        window_start=int(training_cfg["window_start"]),
        epochs=int(profile["epochs"]),
        steps_per_epoch=int(profile["steps_per_epoch"]),
        batch_size=int(training_cfg["batch_size"]),
        lr=float(training_cfg["learning_rate"]),
        grad_clip=float(training_cfg["gradient_clip"]),
        weight_decay=float(training_cfg["weight_decay"]),
        w_state=float(training_cfg["state_loss_weight"]),
        w_embed=float(training_cfg["embedding_loss_weight"]),
        include_constant=True,
    )
    training = fit_dkuc(
        train_data["states"],
        train_data["inputs"],
        validation_data["states"],
        validation_data["inputs"],
        x_normalizer,
        u_normalizer,
        dkuc_config,
        output,
        torch.device(device),
    )
    ridge = _ridge_refit(output, train_data, float(config["ridge_lambda"]), device)
    model = DKUCModel(output, device)
    if model.A.shape != (int(config["lifted_dim"]), int(config["lifted_dim"])):
        raise RuntimeError(f"unexpected A0 shape: {model.A.shape}")
    if model.B.shape != (int(config["lifted_dim"]), int(config["input_dim"])):
        raise RuntimeError(f"unexpected B0 shape: {model.B.shape}")
    C0 = np.zeros((int(config["state_dim"]), int(config["lifted_dim"])), dtype=np.float64)
    C0[:, : int(config["state_dim"])] = np.eye(int(config["state_dim"]), dtype=np.float64)
    np.savez_compressed(
        output / "initial_model.npz",
        A0=model.A,
        B0=model.B,
        C0=C0,
        x_mean=model.x_normer.mean,
        x_std=model.x_normer.std,
        u_mean=model.u_normer.mean,
        u_std=model.u_normer.std,
    )
    encoder_state = {
        key.removeprefix("encoder."): value.detach().cpu()
        for key, value in model.model.state_dict().items()
        if key.startswith("encoder.")
    }
    torch.save(encoder_state, output / "encoder.pt")
    coordinate_contract = {
        "state_coordinate": "normalized",
        "input_coordinate": "normalized_applied_torque",
        "lift_definition": "z = [x_normalized, phi(x_normalized), 1]",
        "dynamics": "z_next = A0 @ z + B0 @ u_normalized",
        "readout": "x_normalized = C0 @ z",
        "state_normalization": "x_normalized = (x_physical - x_mean) / x_std",
        "input_normalization": "u_normalized = (u_physical - u_mean) / u_std",
        "physical_state_recovery": "x_physical = (C0 @ z) * x_std + x_mean",
        "normalizer_policy": "fixed_and_shared_by_all_methods_and_direct_refit_oracle",
    }
    component_names = (
        "dataset_train.npz",
        "dataset_val.npz",
        "normalizers.json",
        "best_dkuc.pt",
        "model_config.json",
        "dkuc_training_history.csv",
        "initial_model.npz",
        "encoder.pt",
    )
    components = {
        name: {"sha256": _hash_file(output / name), "bytes": (output / name).stat().st_size}
        for name in component_names
        if (output / name).is_file()
    }
    manifest = {
        "artifact_schema_version": 2,
        "artifact_type": "dktv_initial_fixed_dkuc",
        "methods": list(METHODS_SHARING_ARTIFACT),
        "encoder": "encoder.pt",
        "normalizer": "normalizers.json",
        "A0_B0_C0": "initial_model.npz",
        "loadable_dkuc_state": "best_dkuc.pt",
        "model_config": "model_config.json",
        "model_form": config["model_form"],
        "state_readout": config["state_readout"],
        "encoder_update": config["encoder_update"],
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "seed": int(config["seed"]),
        "state_order": config["state"],
        "input": config["input"],
        "lift": config["lift"],
        "lifted_dim": int(config["lifted_dim"]),
        "coordinate_contract": coordinate_contract,
        "components": components,
        "dkuc_config": asdict(dkuc_config),
        "training": training,
        "ridge_refit": ridge,
    }
    save_json(output / "artifact_manifest.json", manifest)
    return manifest


def _prediction_metrics(true_states: np.ndarray, pred_states: np.ndarray) -> dict[str, Any]:
    true_values = np.asarray(true_states, dtype=np.float64)
    pred_values = np.asarray(pred_states, dtype=np.float64)
    error = pred_values - true_values
    error = error[:, 1:]
    finite = np.isfinite(error)
    nonfinite_count = int(error.size - np.count_nonzero(finite))
    clean = np.where(finite, error, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        total_rmse = float(np.sqrt(np.nanmean(clean * clean)))
        state_rmse = np.sqrt(np.nanmean(clean * clean, axis=(0, 1))).tolist()
        max_abs = float(np.nanmax(np.abs(clean)))
    return {
        "total_rmse": total_rmse,
        "rmse_by_state": state_rmse,
        "state_labels": ["qa", "qb", "dqa", "dqb"],
        "max_abs_error": max_abs,
        "nonfinite_prediction_count": int(np.count_nonzero(~np.isfinite(pred_values))),
        "nonfinite_error_count": nonfinite_count,
        "sample_count": int(error.shape[0] * error.shape[1]),
    }


def _window_rollouts(
    model: DKUCModel,
    states: np.ndarray,
    inputs: np.ndarray,
    horizon: int,
    stride: int,
    starts: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = inputs.shape[1]
    start_values = starts if starts is not None else list(range(0, steps - horizon + 1, stride))
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    start_records: list[tuple[int, int]] = []
    for trajectory in range(states.shape[0]):
        for start in start_values:
            stop = start + horizon
            truth.append(states[trajectory, start : stop + 1])
            prediction.append(model.rollout(states[trajectory, start], inputs[trajectory, start:stop]))
            start_records.append((trajectory, start))
    if not truth:
        return (
            np.empty((0, horizon + 1, states.shape[-1]), dtype=np.float64),
            np.empty((0, horizon + 1, states.shape[-1]), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
        )
    return np.stack(truth), np.stack(prediction), np.asarray(start_records, dtype=np.int64)


def evaluate_fixed_dko(
    model: DKUCModel,
    validation_stream: dict[str, np.ndarray],
    config: dict[str, Any],
    result_dir: str | Path,
) -> dict[str, Any]:
    """Save unified one-step, horizon rollout, and stage metrics plus arrays."""
    output = Path(result_dir)
    metrics_dir = output / "metrics"
    arrays_dir = output / "arrays"
    figures_dir = output / "figures"
    for directory in (metrics_dir, arrays_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    states = np.asarray(validation_stream["states"], dtype=np.float64)
    inputs = np.asarray(validation_stream["inputs"], dtype=np.float64)
    one_step = np.stack([predict_one_step(model, states[i], inputs[i]) for i in range(states.shape[0])])
    full_rollout = np.stack([model.rollout(states[i, 0], inputs[i]) for i in range(states.shape[0])])
    one_step_metrics = _prediction_metrics(states, one_step)
    full_rollout_metrics = _prediction_metrics(states, full_rollout)

    stride = int(config["evaluation"]["window_stride"])
    horizon_metrics: dict[str, Any] = {}
    saved_arrays: dict[str, np.ndarray] = {
        "t": np.asarray(validation_stream["t"]),
        "disturbance_torque": np.asarray(validation_stream["disturbance_torque"]),
        "applied_torque": inputs,
        "states_true": states,
        "one_step_pred": one_step,
        "full_rollout_pred": full_rollout,
    }
    for horizon in config["evaluation"]["rollout_horizons"]:
        truth, prediction, starts = _window_rollouts(model, states, inputs, int(horizon), stride)
        horizon_metrics[str(horizon)] = _prediction_metrics(truth, prediction)
        saved_arrays[f"rollout_{horizon}_true"] = truth
        saved_arrays[f"rollout_{horizon}_pred"] = prediction
        saved_arrays[f"rollout_{horizon}_starts"] = starts

    stage_horizon = int(config["evaluation"]["stage_rollout_horizon"])
    stages: dict[str, Any] = {}
    for stage in stage_bounds(config, inputs.shape[1]):
        starts = list(
            range(
                int(stage["start_step"]),
                int(stage["end_step"]) - stage_horizon + 1,
                stride,
            )
        )
        truth, prediction, records = _window_rollouts(
            model,
            states,
            inputs,
            stage_horizon,
            stride,
            starts,
        )
        if truth.shape[0] == 0:
            raise RuntimeError(f"stage {stage['name']} is too short for evaluation horizon")
        stage_metrics = _prediction_metrics(truth, prediction)
        stage_metrics.update(
            {
                "start_step": int(stage["start_step"]),
                "end_step": int(stage["end_step"]),
                "disturbance_scale": float(stage["scale"]),
                "rollout_horizon": stage_horizon,
            }
        )
        stages[str(stage["name"])] = stage_metrics
        saved_arrays[f"stage_{stage['name']}_true"] = truth
        saved_arrays[f"stage_{stage['name']}_pred"] = prediction
        saved_arrays[f"stage_{stage['name']}_starts"] = records

    nominal_rmse = float(stages["nominal"]["total_rmse"])
    varying_rmse = float(stages["time_varying"]["total_rmse"])
    degradation_ratio = varying_rmse / max(nominal_rmse, np.finfo(np.float64).eps)
    degradation = {
        "nominal_stage_rmse": nominal_rmse,
        "time_varying_stage_rmse": varying_rmse,
        "time_varying_to_nominal_ratio": float(degradation_ratio),
        "required_min_ratio": float(config["evaluation"]["degradation_ratio_min"]),
        "passed": bool(degradation_ratio >= float(config["evaluation"]["degradation_ratio_min"])),
    }

    figure_paths = plot_prediction_states(states, one_step, figures_dir, "one_step")
    figure_paths += plot_prediction_errors(states, one_step, figures_dir, "one_step")
    figure_paths += plot_prediction_states(states, full_rollout, figures_dir, "rollout")
    figure_paths += plot_prediction_errors(states, full_rollout, figures_dir, "rollout")
    np.savez_compressed(arrays_dir / "fixed_dko_predictions.npz", **saved_arrays)
    save_json(metrics_dir / "one_step.json", one_step_metrics)
    save_json(metrics_dir / "full_rollout.json", full_rollout_metrics)
    save_json(metrics_dir / "rollout_horizons.json", horizon_metrics)
    save_json(metrics_dir / "rollout_by_stage.json", stages)
    save_json(metrics_dir / "degradation.json", degradation)
    figure_paths_relative = [
        Path(path).resolve().relative_to(output.resolve()).as_posix() for path in figure_paths
    ]
    return {
        "one_step": one_step_metrics,
        "full_rollout": full_rollout_metrics,
        "rollout_horizons": horizon_metrics,
        "rollout_by_stage": stages,
        "degradation": degradation,
        "arrays": "arrays/fixed_dko_predictions.npz",
        "figures": figure_paths_relative,
    }
