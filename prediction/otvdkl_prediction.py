"""Zhang et al. OTVDKL prediction and sliding-window update entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

try:  # Support both file-path and ``python -m prediction...`` execution.
    from .common import (
        INPUT_ORDER,
        STATE_ORDER,
        LeastSquaresResult,
        build_regressor,
        create_prediction_run_paths,
        direct_refit,
        dkuc_artifact_fingerprint,
        evaluate_predictions,
        lift_dkuc_transitions,
        load_dataset,
        plot_prediction_errors,
        plot_prediction_states,
        predict_dkuc_latent_batch,
        save_json,
        set_seed,
        snapshot_rollout_predictions,
        sufficient_statistics,
    )
    from .dkuc_prediction import DKUCModel
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from common import (
        INPUT_ORDER,
        STATE_ORDER,
        LeastSquaresResult,
        build_regressor,
        create_prediction_run_paths,
        direct_refit,
        dkuc_artifact_fingerprint,
        evaluate_predictions,
        lift_dkuc_transitions,
        load_dataset,
        plot_prediction_errors,
        plot_prediction_states,
        predict_dkuc_latent_batch,
        save_json,
        set_seed,
        snapshot_rollout_predictions,
        sufficient_statistics,
    )
    from dkuc_prediction import DKUCModel


def latent_rmse(
    A: np.ndarray,
    B: np.ndarray,
    z_current: np.ndarray,
    u_normalized: np.ndarray,
    z_next: np.ndarray,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        error = z_current @ A.T + u_normalized @ B.T - z_next
    return float(np.sqrt(np.mean(error * error)))


def physical_state_rmse(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    z_current: np.ndarray,
    u_normalized: np.ndarray,
    x_next_normalized: np.ndarray,
) -> float:
    """Equation (17)/(18) error in normalized physical-state coordinates."""
    with np.errstate(over="ignore", invalid="ignore"):
        z_prediction = z_current @ A.T + u_normalized @ B.T
        error = z_prediction @ C.T - np.asarray(x_next_normalized, dtype=np.float64)
        return float(np.sqrt(np.mean(error * error)))


def recover_otvdkl_batch(
    model: Any,
    C: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    """Decode latent values with the learned OTVDKL readout ``C_tau``."""
    latent_values = np.asarray(latent, dtype=np.float64)
    readout = np.asarray(C, dtype=np.float64)
    if readout.shape != (model.state_dim, latent_values.shape[-1]):
        raise ValueError("C shape does not match the latent and physical-state dimensions")
    normalized = latent_values @ readout.T
    shape = normalized.shape
    return model.x_normer.inverse(
        normalized.reshape(-1, model.state_dim)
    ).reshape(shape)


def predict_otvdkl_batch(
    model: Any,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    z_current: np.ndarray,
    u_normalized: np.ndarray,
) -> np.ndarray:
    """Apply ``C_tau(A_tau g(x_k) + B_tau u_k)`` in physical coordinates."""
    latent_next = (
        np.asarray(z_current, dtype=np.float64) @ np.asarray(A, dtype=np.float64).T
        + np.asarray(u_normalized, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T
    )
    return recover_otvdkl_batch(model, C, latent_next)


def otvdkl_snapshot_rollout_predictions(
    model: Any,
    dataset: dict[str, np.ndarray],
    A_by_step: np.ndarray,
    B_by_step: np.ndarray,
    C_by_step: np.ndarray,
    *,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the causal OTVDKL snapshot available at each rollout origin."""
    states = np.asarray(dataset["states"], dtype=np.float64)
    inputs = np.asarray(dataset["inputs"], dtype=np.float64)
    steps = int(inputs.shape[1])
    if int(horizon) <= 0 or int(horizon) > steps:
        raise ValueError("rollout horizon must be within the stream length")
    if int(stride) <= 0:
        raise ValueError("rollout stride must be positive")
    if (
        A_by_step.shape[0] != steps
        or B_by_step.shape[0] != steps
        or C_by_step.shape[0] != steps
    ):
        raise ValueError("A/B/C snapshots must align with stream steps")
    normalized_inputs = model.u_normer.transform(
        inputs.reshape(-1, model.control_dim)
    ).reshape(inputs.shape)
    starts = np.arange(0, steps - int(horizon) + 1, int(stride), dtype=np.int64)
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    for trajectory in range(states.shape[0]):
        for start_value in starts:
            start = int(start_value)
            A = np.asarray(A_by_step[start], dtype=np.float64)
            B = np.asarray(B_by_step[start], dtype=np.float64)
            C = np.asarray(C_by_step[start], dtype=np.float64)
            z = model.lift(states[trajectory, start])
            values = np.zeros((int(horizon) + 1, model.state_dim), dtype=np.float64)
            values[0] = states[trajectory, start]
            for offset in range(int(horizon)):
                z = A @ z + B @ normalized_inputs[trajectory, start + offset]
                values[offset + 1] = recover_otvdkl_batch(model, C, z)
            truth.append(states[trajectory, start : start + int(horizon) + 1])
            prediction.append(values)
    return np.stack(truth), np.stack(prediction)


@dataclass(frozen=True)
class WindowCandidate:
    """A non-mutating candidate for one exact remove/add operation."""

    z_current: np.ndarray
    z_next: np.ndarray
    u_normalized: np.ndarray
    sample_ids: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    inverse_regularized_gram: np.ndarray
    result: LeastSquaresResult
    direct_result: LeastSquaresResult
    C: np.ndarray
    direct_C: np.ndarray
    c_gram: np.ndarray
    c_cross: np.ndarray
    inverse_regularized_c_gram: np.ndarray
    recursive_A_max_abs_difference: float
    recursive_B_max_abs_difference: float
    candidate_A_max_abs_difference: float
    candidate_B_max_abs_difference: float
    recursive_C_max_abs_difference: float
    candidate_C_max_abs_difference: float
    addition_system_condition_number: float
    deletion_system_condition_number: float
    c_addition_system_condition_number: float
    c_deletion_system_condition_number: float
    recursive_path: str
    recursive_candidate_time_s: float
    direct_refit_oracle_time_s: float
    fallback_time_s: float
    total_candidate_time_s: float


@dataclass(frozen=True)
class WindowUpdateResult:
    status: str
    accepted: bool
    window_advanced: bool
    reason: str
    attempt_index: int
    model_version: int
    window_version: int
    batch_sample_count: int
    window_sample_count: int
    inserted_sample_ids: list[int]
    evicted_sample_ids: list[int]
    window_start_sample_id: int
    window_end_sample_id: int
    update_time_s: float
    recursive_candidate_time_s: float | None
    direct_refit_oracle_time_s: float | None
    fallback_time_s: float | None
    total_update_time_s: float
    current_batch_rmse: float | None
    candidate_batch_rmse: float | None
    diagnostics: dict[str, Any] | None
    direct_diagnostics: dict[str, Any] | None
    recursive_A_max_abs_difference: float | None
    recursive_B_max_abs_difference: float | None
    candidate_A_max_abs_difference: float | None
    candidate_B_max_abs_difference: float | None
    recursive_C_max_abs_difference: float | None
    candidate_C_max_abs_difference: float | None
    addition_system_condition_number: float | None
    deletion_system_condition_number: float | None
    c_addition_system_condition_number: float | None
    c_deletion_system_condition_number: float | None
    recursive_path: str | None
    new_batch_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OTVDKLModelSnapshot:
    """Immutable, version-consistent model consumed by a controller."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    C_struct: np.ndarray
    model_version: int
    window_version: int
    encoder_fingerprint: str
    state_dim: int
    input_dim: int
    latent_dim: int


class SlidingWindowKoopmanUpdater:
    """Maintain raw window samples and recursively replace its oldest batch."""

    def __init__(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        A0: np.ndarray,
        B0: np.ndarray,
        C0: np.ndarray | None = None,
        state_dim: int = 4,
        batch_size: int,
        ridge_lambda: float,
        encoder_fingerprint: str,
        sample_ids: np.ndarray | None = None,
        affine_constant: bool = True,
        oracle_tolerance: float = 1e-8,
        low_dim_condition_limit: float = 1e12,
        window_condition_limit: float = 1e12,
        allow_direct_fallback: bool = True,
    ) -> None:
        self.z_current = np.asarray(z_current, dtype=np.float64).copy()
        self.z_next = np.asarray(z_next, dtype=np.float64).copy()
        self.u_normalized = np.asarray(u_normalized, dtype=np.float64).copy()
        regressor = build_regressor(self.z_current, self.u_normalized)
        if self.z_next.shape != self.z_current.shape or not np.all(np.isfinite(self.z_next)):
            raise ValueError("z_next must be finite and match z_current")
        self.window_size = int(regressor.shape[0])
        self.batch_size = int(batch_size)
        self.latent_dim = int(self.z_current.shape[1])
        self.input_dim = int(self.u_normalized.shape[1])
        self.state_dim = int(state_dim)
        self.regressor_dim = self.latent_dim + self.input_dim
        if self.batch_size <= 0 or self.batch_size > self.window_size:
            raise ValueError("batch_size must be positive and no larger than the window")
        if self.window_size < self.regressor_dim:
            raise ValueError("window_size must be at least latent_dim + input_dim")
        if float(ridge_lambda) <= 0.0:
            raise ValueError("ridge_lambda must be positive")
        if not encoder_fingerprint:
            raise ValueError("encoder_fingerprint must be non-empty")
        self.A = np.asarray(A0, dtype=np.float64).copy()
        self.B = np.asarray(B0, dtype=np.float64).copy()
        self.C_struct = np.zeros((self.state_dim, self.latent_dim), dtype=np.float64)
        self.C_struct[:, : self.state_dim] = np.eye(self.state_dim)
        c_was_supplied = C0 is not None
        self.C = self.C_struct.copy() if C0 is None else np.asarray(C0, dtype=np.float64).copy()
        if self.A.shape != (self.latent_dim, self.latent_dim):
            raise ValueError("A0 shape does not match the window")
        if self.B.shape != (self.latent_dim, self.input_dim):
            raise ValueError("B0 shape does not match the window")
        if self.C.shape != (self.state_dim, self.latent_dim):
            raise ValueError("C0 shape does not match state_dim and latent_dim")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.B)):
            raise ValueError("initial model must be finite")
        if sample_ids is None:
            self.sample_ids = np.arange(-self.window_size, 0, dtype=np.int64)
        else:
            self.sample_ids = np.asarray(sample_ids, dtype=np.int64).copy()
        if self.sample_ids.shape != (self.window_size,) or np.unique(self.sample_ids).size != self.window_size:
            raise ValueError("sample_ids must be unique and aligned with the window")
        self.ridge_lambda = float(ridge_lambda)
        self.encoder_fingerprint = str(encoder_fingerprint)
        self.affine_constant = bool(affine_constant)
        self.oracle_tolerance = float(oracle_tolerance)
        self.low_dim_condition_limit = float(low_dim_condition_limit)
        self.window_condition_limit = float(window_condition_limit)
        self.allow_direct_fallback = bool(allow_direct_fallback)
        self.gram, self.cross = sufficient_statistics(regressor, self.z_next)
        self.inverse_regularized_gram = np.linalg.inv(
            self.gram + self.ridge_lambda * np.eye(self.regressor_dim)
        )
        self.c_gram = self.z_next.T @ self.z_next
        self.c_cross = self.z_next.T @ self.z_next[:, : self.state_dim]
        self.inverse_regularized_c_gram = np.linalg.inv(
            self.c_gram + self.ridge_lambda * np.eye(self.latent_dim)
        )
        if not c_was_supplied:
            self.C = (self.inverse_regularized_c_gram @ self.c_cross).T
        self.attempt_index = 0
        self.model_version = 0
        self.window_version = 0

    @property
    def memory_bytes(self) -> int:
        arrays = (
            self.z_current,
            self.z_next,
            self.u_normalized,
            self.sample_ids,
            self.gram,
            self.cross,
            self.inverse_regularized_gram,
            self.A,
            self.B,
            self.C,
            self.c_gram,
            self.c_cross,
            self.inverse_regularized_c_gram,
        )
        return int(sum(values.nbytes for values in arrays))

    def _recursive_result(
        self,
        inverse: np.ndarray,
        cross: np.ndarray,
        direct: LeastSquaresResult,
    ) -> LeastSquaresResult:
        with np.errstate(over="ignore", invalid="ignore"):
            theta = inverse @ cross
        A = theta[: self.latent_dim].T.copy()
        B = theta[self.latent_dim :].T.copy()
        if self.affine_constant:
            A[-1] = 0.0
            A[-1, -1] = 1.0
            B[-1] = 0.0
            theta = np.concatenate([A, B], axis=1).T
        return LeastSquaresResult(A=A, B=B, theta=theta, diagnostics=direct.diagnostics)

    def _replace_inverse(
        self,
        inverse: np.ndarray,
        new_rows: np.ndarray,
        outgoing_rows: np.ndarray,
        *,
        label: str,
    ) -> tuple[np.ndarray, float, float]:
        """Apply regularized add/delete Woodbury updates to one Gram inverse."""
        identity = np.eye(self.batch_size)
        with np.errstate(over="ignore", invalid="ignore"):
            addition_system = identity + new_rows @ inverse @ new_rows.T
        if not np.all(np.isfinite(addition_system)):
            raise FloatingPointError(f"{label} addition Woodbury system is non-finite")
        addition_condition = float(np.linalg.cond(addition_system))
        if (
            not np.isfinite(addition_condition)
            or addition_condition > self.low_dim_condition_limit
        ):
            raise np.linalg.LinAlgError(
                f"{label} addition Woodbury system is ill-conditioned"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            added_inverse = inverse - (
                inverse
                @ new_rows.T
                @ np.linalg.solve(addition_system, new_rows @ inverse)
            )
            deletion_system = identity - outgoing_rows @ added_inverse @ outgoing_rows.T
        if not np.all(np.isfinite(deletion_system)):
            raise FloatingPointError(f"{label} deletion Woodbury system is non-finite")
        deletion_condition = float(np.linalg.cond(deletion_system))
        if (
            not np.isfinite(deletion_condition)
            or deletion_condition > self.low_dim_condition_limit
        ):
            raise np.linalg.LinAlgError(
                f"{label} deletion Woodbury system is ill-conditioned"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            replaced_inverse = added_inverse + (
                added_inverse
                @ outgoing_rows.T
                @ np.linalg.solve(
                    deletion_system,
                    outgoing_rows @ added_inverse,
                )
            )
        replaced_inverse = 0.5 * (replaced_inverse + replaced_inverse.T)
        if not np.all(np.isfinite(replaced_inverse)):
            raise FloatingPointError(f"{label} updated inverse is non-finite")
        return replaced_inverse, addition_condition, deletion_condition

    def propose(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        sample_ids: np.ndarray,
        encoder_fingerprint: str,
    ) -> WindowCandidate:
        candidate_started = perf_counter()
        if str(encoder_fingerprint) != self.encoder_fingerprint:
            raise ValueError("encoder fingerprint changed; window statistics are invalid")
        new_z = np.asarray(z_current, dtype=np.float64)
        new_next = np.asarray(z_next, dtype=np.float64)
        new_u = np.asarray(u_normalized, dtype=np.float64)
        new_ids = np.asarray(sample_ids, dtype=np.int64)
        new_regressor = build_regressor(new_z, new_u)
        if new_regressor.shape[0] != self.batch_size:
            raise ValueError("each update must contain exactly batch_size samples")
        if new_next.shape != new_z.shape or not np.all(np.isfinite(new_next)):
            raise ValueError("z_next must be finite and match z_current")
        if new_ids.shape != (self.batch_size,) or np.unique(new_ids).size != self.batch_size:
            raise ValueError("new sample_ids must be unique and batch-aligned")
        if np.intersect1d(new_ids, self.sample_ids).size:
            raise ValueError("new sample_ids duplicate samples already in the window")

        outgoing_z = self.z_current[: self.batch_size]
        outgoing_next = self.z_next[: self.batch_size]
        outgoing_u = self.u_normalized[: self.batch_size]
        outgoing_regressor = build_regressor(outgoing_z, outgoing_u)
        candidate_inverse, add_condition, delete_condition = self._replace_inverse(
            self.inverse_regularized_gram,
            new_regressor,
            outgoing_regressor,
            label="A/B",
        )
        candidate_c_inverse, c_add_condition, c_delete_condition = self._replace_inverse(
            self.inverse_regularized_c_gram,
            new_next,
            outgoing_next,
            label="C",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            add_gram, add_cross = sufficient_statistics(new_regressor, new_next)
            out_gram, out_cross = sufficient_statistics(outgoing_regressor, outgoing_next)
        if not all(np.all(np.isfinite(values)) for values in (add_gram, add_cross, out_gram, out_cross)):
            raise FloatingPointError("batch sufficient statistics are non-finite")
        candidate_gram = self.gram + add_gram - out_gram
        candidate_gram = 0.5 * (candidate_gram + candidate_gram.T)
        candidate_cross = self.cross + add_cross - out_cross
        new_x_next = new_next[:, : self.state_dim]
        outgoing_x_next = outgoing_next[:, : self.state_dim]
        candidate_c_gram = (
            self.c_gram + new_next.T @ new_next - outgoing_next.T @ outgoing_next
        )
        candidate_c_gram = 0.5 * (candidate_c_gram + candidate_c_gram.T)
        candidate_c_cross = (
            self.c_cross
            + new_next.T @ new_x_next
            - outgoing_next.T @ outgoing_x_next
        )
        if not all(
            np.all(np.isfinite(values))
            for values in (
                candidate_gram,
                candidate_cross,
                candidate_c_gram,
                candidate_c_cross,
            )
        ):
            raise FloatingPointError("candidate sufficient statistics are non-finite")
        candidate_z = np.concatenate([self.z_current[self.batch_size :], new_z], axis=0)
        candidate_next = np.concatenate([self.z_next[self.batch_size :], new_next], axis=0)
        candidate_u = np.concatenate([self.u_normalized[self.batch_size :], new_u], axis=0)
        candidate_ids = np.concatenate([self.sample_ids[self.batch_size :], new_ids])
        recursive_candidate_time = perf_counter() - candidate_started
        direct_started = perf_counter()
        direct = direct_refit(
            candidate_z,
            candidate_next,
            candidate_u,
            ridge_lambda=self.ridge_lambda,
            affine_constant=self.affine_constant,
        )
        direct_C = np.linalg.solve(
            candidate_next.T @ candidate_next
            + self.ridge_lambda * np.eye(self.latent_dim),
            candidate_next.T @ candidate_next[:, : self.state_dim],
        ).T
        direct_refit_time = perf_counter() - direct_started
        if direct.diagnostics.rank < self.regressor_dim:
            raise np.linalg.LinAlgError("candidate window regressor is rank deficient")
        if np.linalg.matrix_rank(candidate_next) < self.latent_dim:
            raise np.linalg.LinAlgError("candidate C regressor is rank deficient")
        if (
            not np.isfinite(direct.diagnostics.condition_number)
            or direct.diagnostics.condition_number > self.window_condition_limit
        ):
            raise np.linalg.LinAlgError("candidate window regressor is ill-conditioned")
        recursive_resume = perf_counter()
        recursive = self._recursive_result(candidate_inverse, candidate_cross, direct)
        recursive_C = (candidate_c_inverse @ candidate_c_cross).T
        recursive_candidate_time += perf_counter() - recursive_resume
        with np.errstate(over="ignore", invalid="ignore"):
            A_difference = float(np.max(np.abs(recursive.A - direct.A)))
            B_difference = float(np.max(np.abs(recursive.B - direct.B)))
            C_difference = float(np.max(np.abs(recursive_C - direct_C)))
        recursive_finite = all(
            np.all(np.isfinite(values))
            for values in (
                recursive.A,
                recursive.B,
                recursive.theta,
                recursive_C,
                candidate_inverse,
                candidate_c_inverse,
            )
        )
        differences_finite = bool(
            np.isfinite(A_difference)
            and np.isfinite(B_difference)
            and np.isfinite(C_difference)
        )
        path = "woodbury_add_delete"
        fallback_time = 0.0
        if (
            not recursive_finite
            or not differences_finite
            or A_difference > self.oracle_tolerance
            or B_difference > self.oracle_tolerance
            or C_difference > self.oracle_tolerance
        ):
            if not self.allow_direct_fallback:
                raise FloatingPointError("recursive candidate did not match direct refit")
            fallback_started = perf_counter()
            candidate_regressor = build_regressor(candidate_z, candidate_u)
            candidate_gram, candidate_cross = sufficient_statistics(
                candidate_regressor, candidate_next
            )
            if not np.all(np.isfinite(candidate_gram)) or not np.all(
                np.isfinite(candidate_cross)
            ):
                raise FloatingPointError("fallback sufficient statistics are non-finite")
            candidate_inverse = np.linalg.inv(
                candidate_gram + self.ridge_lambda * np.eye(self.regressor_dim)
            )
            candidate_c_gram = candidate_next.T @ candidate_next
            candidate_c_cross = candidate_next.T @ candidate_next[:, : self.state_dim]
            candidate_c_inverse = np.linalg.inv(
                candidate_c_gram + self.ridge_lambda * np.eye(self.latent_dim)
            )
            if not all(
                np.all(np.isfinite(values))
                for values in (candidate_inverse, candidate_c_inverse)
            ):
                raise FloatingPointError("fallback inverse is non-finite")
            recursive = direct
            recursive_C = direct_C
            path = "direct_refit_fallback"
            fallback_time = perf_counter() - fallback_started
        candidate_A_difference = float(np.max(np.abs(recursive.A - direct.A)))
        candidate_B_difference = float(np.max(np.abs(recursive.B - direct.B)))
        candidate_C_difference = float(np.max(np.abs(recursive_C - direct_C)))
        return WindowCandidate(
            z_current=candidate_z,
            z_next=candidate_next,
            u_normalized=candidate_u,
            sample_ids=candidate_ids,
            gram=candidate_gram,
            cross=candidate_cross,
            inverse_regularized_gram=candidate_inverse,
            result=recursive,
            direct_result=direct,
            C=recursive_C,
            direct_C=direct_C,
            c_gram=candidate_c_gram,
            c_cross=candidate_c_cross,
            inverse_regularized_c_gram=candidate_c_inverse,
            recursive_A_max_abs_difference=A_difference,
            recursive_B_max_abs_difference=B_difference,
            candidate_A_max_abs_difference=candidate_A_difference,
            candidate_B_max_abs_difference=candidate_B_difference,
            recursive_C_max_abs_difference=C_difference,
            candidate_C_max_abs_difference=candidate_C_difference,
            addition_system_condition_number=add_condition,
            deletion_system_condition_number=delete_condition,
            c_addition_system_condition_number=c_add_condition,
            c_deletion_system_condition_number=c_delete_condition,
            recursive_path=path,
            recursive_candidate_time_s=float(recursive_candidate_time),
            direct_refit_oracle_time_s=float(direct_refit_time),
            fallback_time_s=float(fallback_time),
            total_candidate_time_s=float(perf_counter() - candidate_started),
        )

    def commit(self, candidate: WindowCandidate, *, accept_model: bool) -> None:
        self.z_current = candidate.z_current.copy()
        self.z_next = candidate.z_next.copy()
        self.u_normalized = candidate.u_normalized.copy()
        self.sample_ids = candidate.sample_ids.copy()
        self.gram = candidate.gram.copy()
        self.cross = candidate.cross.copy()
        self.inverse_regularized_gram = candidate.inverse_regularized_gram.copy()
        self.c_gram = candidate.c_gram.copy()
        self.c_cross = candidate.c_cross.copy()
        self.inverse_regularized_c_gram = candidate.inverse_regularized_c_gram.copy()
        self.window_version += 1
        if accept_model:
            self.A = candidate.result.A.copy()
            self.B = candidate.result.B.copy()
            self.C = candidate.C.copy()
            self.model_version += 1

    def snapshot(self) -> "OTVDKLModelSnapshot":
        return OTVDKLModelSnapshot(
            A=self.A.copy(), B=self.B.copy(), C=self.C.copy(),
            C_struct=self.C_struct.copy(), model_version=self.model_version,
            window_version=self.window_version, encoder_fingerprint=self.encoder_fingerprint,
            state_dim=self.state_dim, input_dim=self.input_dim, latent_dim=self.latent_dim,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        pending: dict[str, np.ndarray] | None = None,
        variant: str = "otvdkl",
        epsilon: float | None = None,
        improvement_tolerance: float = 0.0,
    ) -> None:
        payload = {
            "checkpoint_version": np.array(1, dtype=np.int64),
            "variant": np.array(variant),
            "epsilon": np.array(np.nan if epsilon is None else float(epsilon)),
            "improvement_tolerance": np.array(float(improvement_tolerance)),
            "new_batch_policy": np.array("discard_after_evaluation"),
            "z_current": self.z_current, "z_next": self.z_next,
            "u_normalized": self.u_normalized, "sample_ids": self.sample_ids,
            "A": self.A, "B": self.B, "C": self.C, "C_struct": self.C_struct,
            "gram": self.gram, "cross": self.cross,
            "inverse_regularized_gram": self.inverse_regularized_gram,
            "c_gram": self.c_gram, "c_cross": self.c_cross,
            "inverse_regularized_c_gram": self.inverse_regularized_c_gram,
            "model_version": np.array(self.model_version),
            "window_version": np.array(self.window_version),
            "attempt_index": np.array(self.attempt_index),
            "batch_size": np.array(self.batch_size), "ridge_lambda": np.array(self.ridge_lambda),
            "state_dim": np.array(self.state_dim),
            "affine_constant": np.array(self.affine_constant),
            "oracle_tolerance": np.array(self.oracle_tolerance),
            "low_dim_condition_limit": np.array(self.low_dim_condition_limit),
            "window_condition_limit": np.array(self.window_condition_limit),
            "allow_direct_fallback": np.array(self.allow_direct_fallback),
            "encoder_fingerprint": np.array(self.encoder_fingerprint),
        }
        for key, value in (pending or {}).items():
            payload[f"pending_{key}"] = np.asarray(value)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **payload)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
    ) -> tuple[
        "SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater",
        dict[str, np.ndarray],
    ]:
        with np.load(path, allow_pickle=False) as data:
            base = cls(
                data["z_current"], data["z_next"], data["u_normalized"],
                A0=data["A"], B0=data["B"], C0=data["C"],
                state_dim=int(data["state_dim"]), batch_size=int(data["batch_size"]),
                ridge_lambda=float(data["ridge_lambda"]),
                encoder_fingerprint=str(data["encoder_fingerprint"]),
                sample_ids=data["sample_ids"],
                affine_constant=bool(data["affine_constant"]),
                oracle_tolerance=float(data["oracle_tolerance"]),
                low_dim_condition_limit=float(data["low_dim_condition_limit"]),
                window_condition_limit=float(data["window_condition_limit"]),
                allow_direct_fallback=bool(data["allow_direct_fallback"]),
            )
            for name in ("gram", "cross", "inverse_regularized_gram", "c_gram", "c_cross", "inverse_regularized_c_gram"):
                setattr(base, name, data[name].copy())
            base.model_version = int(data["model_version"])
            base.window_version = int(data["window_version"])
            base.attempt_index = int(data["attempt_index"])
            pending = {key[8:]: data[key].copy() for key in data.files if key.startswith("pending_")}
            variant = str(data["variant"]) if "variant" in data.files else "otvdkl"
            epsilon = float(data["epsilon"]) if "epsilon" in data.files else np.nan
            improvement_tolerance = (
                float(data["improvement_tolerance"])
                if "improvement_tolerance" in data.files
                else 0.0
            )
        if variant == "otvdkl_star":
            if not np.isfinite(epsilon):
                raise ValueError("OTVDKL* checkpoint is missing a finite epsilon")
            updater: SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater = (
                SelectiveWindowKoopmanUpdater(
                    base,
                    epsilon=epsilon,
                    improvement_tolerance=improvement_tolerance,
                )
            )
        elif variant == "otvdkl":
            updater = base
        else:
            raise ValueError(f"unsupported checkpoint variant: {variant}")
        return updater, pending

    def _record(
        self,
        *,
        status: str,
        reason: str,
        started: float,
        new_ids: np.ndarray,
        evicted_ids: np.ndarray,
        candidate: WindowCandidate | None,
        current_rmse: float | None,
        candidate_rmse: float | None,
        window_advanced: bool,
        new_batch_policy: str = "discard_after_evaluation",
    ) -> WindowUpdateResult:
        elapsed = float(perf_counter() - started)
        return WindowUpdateResult(
            status=status,
            accepted=status == "accepted",
            window_advanced=window_advanced,
            reason=reason,
            attempt_index=self.attempt_index,
            model_version=self.model_version,
            window_version=self.window_version,
            batch_sample_count=int(new_ids.size),
            window_sample_count=self.window_size,
            inserted_sample_ids=new_ids.tolist(),
            evicted_sample_ids=evicted_ids.tolist(),
            window_start_sample_id=int(self.sample_ids[0]),
            window_end_sample_id=int(self.sample_ids[-1]),
            update_time_s=elapsed,
            recursive_candidate_time_s=(
                candidate.recursive_candidate_time_s if candidate else None
            ),
            direct_refit_oracle_time_s=(
                candidate.direct_refit_oracle_time_s if candidate else None
            ),
            fallback_time_s=candidate.fallback_time_s if candidate else None,
            total_update_time_s=elapsed,
            current_batch_rmse=current_rmse,
            candidate_batch_rmse=candidate_rmse,
            diagnostics=candidate.result.diagnostics.to_dict() if candidate else None,
            direct_diagnostics=candidate.direct_result.diagnostics.to_dict() if candidate else None,
            recursive_A_max_abs_difference=(
                candidate.recursive_A_max_abs_difference if candidate else None
            ),
            recursive_B_max_abs_difference=(
                candidate.recursive_B_max_abs_difference if candidate else None
            ),
            candidate_A_max_abs_difference=(
                candidate.candidate_A_max_abs_difference if candidate else None
            ),
            candidate_B_max_abs_difference=(
                candidate.candidate_B_max_abs_difference if candidate else None
            ),
            recursive_C_max_abs_difference=(
                candidate.recursive_C_max_abs_difference if candidate else None
            ),
            candidate_C_max_abs_difference=(
                candidate.candidate_C_max_abs_difference if candidate else None
            ),
            addition_system_condition_number=(
                candidate.addition_system_condition_number if candidate else None
            ),
            deletion_system_condition_number=(
                candidate.deletion_system_condition_number if candidate else None
            ),
            c_addition_system_condition_number=(
                candidate.c_addition_system_condition_number if candidate else None
            ),
            c_deletion_system_condition_number=(
                candidate.c_deletion_system_condition_number if candidate else None
            ),
            recursive_path=candidate.recursive_path if candidate else None,
            new_batch_policy=new_batch_policy,
        )

    def update(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        sample_ids: np.ndarray,
        encoder_fingerprint: str,
    ) -> tuple[WindowUpdateResult, WindowCandidate | None]:
        started = perf_counter()
        self.attempt_index += 1
        new_ids = np.asarray(sample_ids, dtype=np.int64)
        evicted = self.sample_ids[: self.batch_size].copy()
        try:
            current_rmse = physical_state_rmse(
                self.A, self.B, self.C, z_current, u_normalized,
                np.asarray(z_next)[:, : self.state_dim],
            )
            candidate = self.propose(
                z_current,
                z_next,
                u_normalized,
                sample_ids=new_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            candidate_rmse = physical_state_rmse(
                candidate.result.A, candidate.result.B, candidate.C,
                z_current, u_normalized, np.asarray(z_next)[:, : self.state_dim],
            )
            self.commit(candidate, accept_model=True)
            return self._record(
                status="accepted",
                reason="accepted",
                started=started,
                new_ids=new_ids,
                evicted_ids=evicted,
                candidate=candidate,
                current_rmse=current_rmse,
                candidate_rmse=candidate_rmse,
                window_advanced=True,
            ), candidate
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return self._record(
                status="failed_numerical",
                reason=f"failed_numerical:{type(error).__name__}:{error}",
                started=started,
                new_ids=new_ids,
                evicted_ids=evicted,
                candidate=None,
                current_rmse=None,
                candidate_rmse=None,
                window_advanced=False,
            ), None


class SelectiveWindowKoopmanUpdater:
    """OTVDKL* threshold triggering and negative-update rejection."""

    def __init__(
        self,
        updater: SlidingWindowKoopmanUpdater,
        *,
        epsilon: float,
        improvement_tolerance: float = 0.0,
    ) -> None:
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be non-negative")
        self.updater = updater
        self.epsilon = float(epsilon)
        self.improvement_tolerance = float(improvement_tolerance)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        pending: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.updater.save_checkpoint(
            path,
            pending=pending,
            variant="otvdkl_star",
            epsilon=self.epsilon,
            improvement_tolerance=self.improvement_tolerance,
        )

    def update(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        sample_ids: np.ndarray,
        encoder_fingerprint: str,
    ) -> tuple[WindowUpdateResult, WindowCandidate | None]:
        started = perf_counter()
        base = self.updater
        base.attempt_index += 1
        new_ids = np.asarray(sample_ids, dtype=np.int64)
        evicted = base.sample_ids[: base.batch_size].copy()
        try:
            current_rmse = physical_state_rmse(
                base.A, base.B, base.C, z_current, u_normalized,
                np.asarray(z_next)[:, : base.state_dim],
            )
            if current_rmse <= self.epsilon:
                return base._record(
                    status="skipped_threshold",
                    reason="current_batch_rmse_not_above_epsilon",
                    started=started,
                    new_ids=new_ids,
                    evicted_ids=evicted,
                    candidate=None,
                    current_rmse=current_rmse,
                    candidate_rmse=None,
                    window_advanced=False,
                ), None
            candidate = base.propose(
                z_current,
                z_next,
                u_normalized,
                sample_ids=new_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            candidate_rmse = physical_state_rmse(
                candidate.result.A, candidate.result.B, candidate.C,
                z_current, u_normalized, np.asarray(z_next)[:, : base.state_dim],
            )
            if candidate_rmse + self.improvement_tolerance >= current_rmse:
                return base._record(
                    status="rejected",
                    reason="candidate_not_better_on_new_batch",
                    started=started,
                    new_ids=new_ids,
                    evicted_ids=evicted,
                    candidate=candidate,
                    current_rmse=current_rmse,
                    candidate_rmse=candidate_rmse,
                    window_advanced=False,
                ), candidate
            base.commit(candidate, accept_model=True)
            return base._record(
                status="accepted",
                reason="accepted",
                started=started,
                new_ids=new_ids,
                evicted_ids=evicted,
                candidate=candidate,
                current_rmse=current_rmse,
                candidate_rmse=candidate_rmse,
                window_advanced=True,
            ), candidate
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return base._record(
                status="failed_numerical",
                reason=f"failed_numerical:{type(error).__name__}:{error}",
                started=started,
                new_ids=new_ids,
                evicted_ids=evicted,
                candidate=None,
                current_rmse=None,
                candidate_rmse=None,
                window_advanced=False,
            ), None

    def __getattr__(self, name: str):
        return getattr(self.updater, name)


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _effective_horizons(requested: list[int], steps: int) -> list[int]:
    values = sorted({int(value) for value in requested if 0 < int(value) <= steps})
    return values or [steps]


def _latent_stability(A: np.ndarray) -> tuple[float, float]:
    """Return spectral radius and induced 2-norm for one lifted matrix."""
    matrix = np.asarray(A, dtype=np.float64)
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    norm = float(np.linalg.norm(matrix, ord=2))
    return radius, norm


def run_otvdkl_replay(
    model: DKUCModel,
    history_data: dict[str, np.ndarray],
    stream_data: dict[str, np.ndarray],
    *,
    variant: str,
    window_size: int,
    batch_size: int,
    ridge_lambda: float,
    encoder_fingerprint: str,
    epsilon: float,
    improvement_tolerance: float,
    oracle_tolerance: float,
    low_dim_condition_limit: float,
    window_condition_limit: float,
    history_window_start: int | None = None,
    collect_stability_diagnostics: bool = True,
) -> tuple[
    SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Run one causal OTVDKL variant on an explicit chronological stream."""
    if variant not in {"otvdkl", "otvdkl_star"}:
        raise ValueError(f"unknown OTVDKL variant: {variant}")
    if int(window_size) <= 0 or int(batch_size) <= 0:
        raise ValueError("window_size and batch_size must be positive")
    history_z, history_next, history_u = lift_dkuc_transitions(model, history_data)
    stream_z, stream_next, stream_u = lift_dkuc_transitions(model, stream_data)
    latent_dim = int(stream_z.shape[-1])
    history_current = history_z.reshape(-1, latent_dim)
    history_target = history_next.reshape(-1, latent_dim)
    history_input = history_u.reshape(-1, model.control_dim)
    if history_current.shape[0] < int(window_size):
        raise ValueError(
            f"history has {history_current.shape[0]} samples, fewer than window_size={window_size}"
        )
    maximum_history_start = history_current.shape[0] - int(window_size)
    if history_window_start is None:
        selected_history_start = maximum_history_start
    else:
        selected_history_start = int(history_window_start)
        if selected_history_start < 0 or selected_history_start > maximum_history_start:
            raise ValueError(
                "history_window_start must select a complete window within history_dataset"
            )
    selected_history_end = selected_history_start + int(window_size)
    initial_z = history_current[selected_history_start:selected_history_end]
    initial_next = history_target[selected_history_start:selected_history_end]
    initial_u = history_input[selected_history_start:selected_history_end]
    initial_fit = direct_refit(
        initial_z,
        initial_next,
        initial_u,
        ridge_lambda=float(ridge_lambda),
        affine_constant=bool(model.config.include_constant),
    )
    initial_regressor_dim = latent_dim + model.control_dim
    if initial_fit.diagnostics.rank < initial_regressor_dim:
        raise np.linalg.LinAlgError("initial window regressor is rank deficient")
    if (
        not np.isfinite(initial_fit.diagnostics.condition_number)
        or initial_fit.diagnostics.condition_number > float(window_condition_limit)
    ):
        raise np.linalg.LinAlgError("initial window regressor is ill-conditioned")
    if np.linalg.matrix_rank(initial_next) < latent_dim:
        raise np.linalg.LinAlgError("initial C regressor is rank deficient")
    initial_C = np.linalg.solve(
        initial_next.T @ initial_next
        + float(ridge_lambda) * np.eye(latent_dim),
        initial_next.T @ initial_next[:, : model.state_dim],
    ).T
    base = SlidingWindowKoopmanUpdater(
        initial_z,
        initial_next,
        initial_u,
        A0=initial_fit.A,
        B0=initial_fit.B,
        C0=initial_C,
        state_dim=model.state_dim,
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
    if variant == "otvdkl_star":
        updater = SelectiveWindowKoopmanUpdater(
            base,
            epsilon=float(epsilon),
            improvement_tolerance=float(improvement_tolerance),
        )
    else:
        updater = base

    states = np.asarray(stream_data["states"], dtype=np.float64)
    trajectory_count, steps = stream_u.shape[:2]
    prediction = np.zeros_like(states)
    prediction[:, 0] = states[:, 0]
    A_by_step = np.empty((steps, latent_dim, latent_dim), dtype=np.float64)
    B_by_step = np.empty((steps, latent_dim, model.control_dim), dtype=np.float64)
    C_by_step = np.empty((steps, model.state_dim, latent_dim), dtype=np.float64)
    versions = np.empty(steps, dtype=np.int64)
    spectral_radius_by_step = np.empty(steps, dtype=np.float64)
    spectral_norm_by_step = np.empty(steps, dtype=np.float64)
    previous_stability_version: int | None = None
    current_spectral_radius = 0.0
    current_spectral_norm = 0.0
    pending_z = np.empty((0, latent_dim), dtype=np.float64)
    pending_next = np.empty((0, latent_dim), dtype=np.float64)
    pending_u = np.empty((0, model.control_dim), dtype=np.float64)
    pending_ids = np.empty(0, dtype=np.int64)
    next_sample_id = 0
    update_history: list[dict[str, Any]] = []
    memory_values = [int(updater.memory_bytes)]
    boundaries_replayable = True

    for time_step in range(steps):
        A_by_step[time_step] = updater.A
        B_by_step[time_step] = updater.B
        C_by_step[time_step] = updater.C
        versions[time_step] = updater.model_version
        if collect_stability_diagnostics:
            if previous_stability_version != updater.model_version:
                current_spectral_radius, current_spectral_norm = _latent_stability(
                    updater.A
                )
                previous_stability_version = updater.model_version
            spectral_radius_by_step[time_step] = current_spectral_radius
            spectral_norm_by_step[time_step] = current_spectral_norm
        prediction[:, time_step + 1] = predict_otvdkl_batch(
            model,
            updater.A,
            updater.B,
            updater.C,
            stream_z[:, time_step],
            stream_u[:, time_step],
        )
        ids = np.arange(next_sample_id, next_sample_id + trajectory_count, dtype=np.int64)
        next_sample_id += trajectory_count
        pending_z = np.concatenate((pending_z, stream_z[:, time_step]), axis=0)
        pending_next = np.concatenate((pending_next, stream_next[:, time_step]), axis=0)
        pending_u = np.concatenate((pending_u, stream_u[:, time_step]), axis=0)
        pending_ids = np.concatenate((pending_ids, ids))
        while pending_z.shape[0] >= int(batch_size):
            batch_z, pending_z = pending_z[:batch_size], pending_z[batch_size:]
            batch_next, pending_next = pending_next[:batch_size], pending_next[batch_size:]
            batch_u, pending_u = pending_u[:batch_size], pending_u[batch_size:]
            batch_ids, pending_ids = pending_ids[:batch_size], pending_ids[batch_size:]
            before_ids = updater.sample_ids.copy()
            record, _ = updater.update(
                batch_z,
                batch_next,
                batch_u,
                sample_ids=batch_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            if record.window_advanced:
                expected_ids = np.concatenate((before_ids[batch_size:], batch_ids))
                boundaries_replayable &= np.array_equal(updater.sample_ids, expected_ids)
            else:
                boundaries_replayable &= np.array_equal(updater.sample_ids, before_ids)
            boundaries_replayable &= (
                updater.sample_ids.size == int(window_size)
                and np.unique(updater.sample_ids).size == int(window_size)
            )
            values = record.to_dict()
            values["time_step"] = int(time_step)
            values["pending_sample_count"] = int(pending_ids.size)
            update_history.append(values)
            memory_values.append(int(updater.memory_bytes))

    status_counts = {
        status: sum(record["status"] == status for record in update_history)
        for status in ("accepted", "rejected", "skipped_threshold", "failed_numerical")
    }
    update_times = np.asarray(
        [record["update_time_s"] for record in update_history], dtype=np.float64
    )
    candidates = [record for record in update_history if record["diagnostics"] is not None]
    summary = {
        "method": variant,
        "update_rule": "zhang_sliding_window",
        "window_size": int(window_size),
        "batch_size": int(batch_size),
        "ridge_lambda": float(ridge_lambda),
        "epsilon": float(epsilon) if variant == "otvdkl_star" else None,
        "new_batch_policy": "discard_after_evaluation",
        "prediction_readout": "learned_C_tau",
        "initial_model_source": "direct_refit_of_initial_window",
        "history_window_start": int(selected_history_start),
        "history_window_end_exclusive": int(selected_history_end),
        "history_sample_count": int(history_current.shape[0]),
        "initial_regressor_rank": int(initial_fit.diagnostics.rank),
        "initial_regressor_condition_number": float(
            initial_fit.diagnostics.condition_number
        ),
        **{f"{name}_count": int(value) for name, value in status_counts.items()},
        "pending_sample_count": int(pending_ids.size),
        "model_version": int(updater.model_version),
        "window_version": int(updater.window_version),
        "window_memory_constant": len(set(memory_values)) == 1,
        "window_boundaries_replayable": bool(boundaries_replayable),
        "mean_update_time_s": float(np.mean(update_times)) if update_times.size else 0.0,
        "maximum_update_time_s": float(np.max(update_times)) if update_times.size else 0.0,
        "maximum_oracle_A_difference": max(
            (float(record["candidate_A_max_abs_difference"]) for record in candidates),
            default=0.0,
        ),
        "maximum_oracle_B_difference": max(
            (float(record["candidate_B_max_abs_difference"]) for record in candidates),
            default=0.0,
        ),
        "maximum_oracle_C_difference": max(
            (float(record["candidate_C_max_abs_difference"]) for record in candidates),
            default=0.0,
        ),
        "oracle_tolerance_passed": all(
            np.isfinite(float(record["candidate_A_max_abs_difference"]))
            and np.isfinite(float(record["candidate_B_max_abs_difference"]))
            and np.isfinite(float(record["candidate_C_max_abs_difference"]))
            and float(record["candidate_A_max_abs_difference"]) <= float(oracle_tolerance)
            and float(record["candidate_B_max_abs_difference"]) <= float(oracle_tolerance)
            and float(record["candidate_C_max_abs_difference"]) <= float(oracle_tolerance)
            for record in candidates
        ),
    }
    if collect_stability_diagnostics:
        summary.update(
            {
                "maximum_A_spectral_radius": float(np.max(spectral_radius_by_step)),
                "maximum_A_spectral_norm": float(np.max(spectral_norm_by_step)),
                "unstable_snapshot_fraction": float(
                    np.mean(spectral_radius_by_step > 1.0)
                ),
            }
        )
    arrays = {
        "states_true": states,
        "inputs": np.asarray(stream_data["inputs"], dtype=np.float64),
        f"{variant}_one_step": prediction,
        f"{variant}_A_by_step": A_by_step,
        f"{variant}_B_by_step": B_by_step,
        f"{variant}_C_by_step": C_by_step,
        f"{variant}_model_version_by_step": versions,
    }
    if collect_stability_diagnostics:
        arrays[f"{variant}_A_spectral_radius_by_step"] = spectral_radius_by_step
        arrays[f"{variant}_A_spectral_norm_by_step"] = spectral_norm_by_step
    pending = {
        "z_current": pending_z,
        "z_next": pending_next,
        "u_normalized": pending_u,
        "sample_ids": pending_ids,
        "next_sample_id": np.array(next_sample_id, dtype=np.int64),
    }
    return updater, {
        "summary": summary,
        "history": update_history,
        "pending": pending,
    }, arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Zhang et al. OTVDKL from a frozen DKUC artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True, help="Frozen DKUC artifact directory.")
    parser.add_argument(
        "--history_dataset",
        default="",
        help="Initial window dataset; defaults to <artifact_dir>/dataset_train.npz.",
    )
    parser.add_argument("--stream_dataset", required=True, help="Chronological online stream NPZ.")
    parser.add_argument("--variant", choices=["otvdkl", "otvdkl_star", "both"], default="both")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--history_window_start",
        type=int,
        default=-1,
        help="Flattened history-window start index; -1 selects the most recent window.",
    )
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--improvement_tolerance", type=float, default=0.0)
    parser.add_argument("--oracle_tolerance", type=float, default=1e-8)
    parser.add_argument("--low_dim_condition_limit", type=float, default=1e12)
    parser.add_argument("--window_condition_limit", type=float, default=1e12)
    parser.add_argument("--rollout_horizons", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--rollout_stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run_type", choices=["full_run", "smoke_test"], default="full_run")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--tag", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.ridge_lambda <= 0.0 or args.rollout_stride <= 0:
        raise ValueError("ridge_lambda and rollout_stride must be positive")
    if args.history_window_start < -1:
        raise ValueError("history_window_start must be -1 or non-negative")
    set_seed(args.seed)
    artifact_dir = Path(args.artifact_dir).resolve()
    history_path = (
        Path(args.history_dataset).resolve()
        if args.history_dataset
        else artifact_dir / "dataset_train.npz"
    )
    stream_path = Path(args.stream_dataset).resolve()
    fingerprint = dkuc_artifact_fingerprint(artifact_dir)
    model = DKUCModel(artifact_dir, args.device)
    history_data = load_dataset(history_path)
    stream_data = load_dataset(stream_path)
    paths = create_prediction_run_paths(
        "otvdkl", args.run_type, args.tag, args.out_dir or None
    )
    variants = ["otvdkl", "otvdkl_star"] if args.variant == "both" else [args.variant]
    arrays: dict[str, np.ndarray] = {
        "states_true": np.asarray(stream_data["states"], dtype=np.float64),
        "inputs": np.asarray(stream_data["inputs"], dtype=np.float64),
    }
    evidence: dict[str, dict[str, Any]] = {}
    updaters: dict[str, SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater] = {}
    for variant in variants:
        updater, variant_evidence, variant_arrays = run_otvdkl_replay(
            model,
            history_data,
            stream_data,
            variant=variant,
            window_size=args.window_size,
            batch_size=args.batch_size,
            ridge_lambda=args.ridge_lambda,
            encoder_fingerprint=fingerprint,
            epsilon=args.epsilon,
            improvement_tolerance=args.improvement_tolerance,
            oracle_tolerance=args.oracle_tolerance,
            low_dim_condition_limit=args.low_dim_condition_limit,
            window_condition_limit=args.window_condition_limit,
            history_window_start=(
                None if args.history_window_start == -1 else args.history_window_start
            ),
        )
        updaters[variant] = updater
        evidence[variant] = variant_evidence
        arrays.update(variant_arrays)

    stream_z, _, stream_u = lift_dkuc_transitions(model, stream_data)
    steps = int(stream_u.shape[1])
    fixed_prediction = np.zeros_like(arrays["states_true"])
    fixed_prediction[:, 0] = arrays["states_true"][:, 0]
    for time_step in range(steps):
        fixed_prediction[:, time_step + 1] = predict_dkuc_latent_batch(
            model, model.A, model.B, stream_z[:, time_step], stream_u[:, time_step]
        )
    arrays["fixed_dkuc_one_step"] = fixed_prediction
    fixed_spectral_radius, fixed_spectral_norm = _latent_stability(model.A)
    horizons = _effective_horizons(args.rollout_horizons, steps)
    metrics: dict[str, Any] = {
        "one_step": {
            "fixed_dkuc": evaluate_predictions(arrays["states_true"], fixed_prediction)
        },
        "rollout": {"fixed_dkuc": {}},
    }
    for variant in variants:
        metrics["one_step"][variant] = evaluate_predictions(
            arrays["states_true"], arrays[f"{variant}_one_step"]
        )
        metrics["rollout"][variant] = {}
    fixed_A = np.broadcast_to(model.A, (steps, *model.A.shape)).copy()
    fixed_B = np.broadcast_to(model.B, (steps, *model.B.shape)).copy()
    for horizon in horizons:
        truth, fixed_rollout = snapshot_rollout_predictions(
            model,
            stream_data,
            fixed_A,
            fixed_B,
            horizon=horizon,
            stride=args.rollout_stride,
        )
        metrics["rollout"]["fixed_dkuc"][str(horizon)] = evaluate_predictions(
            truth,
            fixed_rollout,
        )
        arrays[f"fixed_dkuc_rollout_h{horizon}_true"] = truth
        arrays[f"fixed_dkuc_rollout_h{horizon}_prediction"] = fixed_rollout
        for variant in variants:
            variant_truth, variant_rollout = otvdkl_snapshot_rollout_predictions(
                model,
                stream_data,
                arrays[f"{variant}_A_by_step"],
                arrays[f"{variant}_B_by_step"],
                arrays[f"{variant}_C_by_step"],
                horizon=horizon,
                stride=args.rollout_stride,
            )
            if not np.array_equal(variant_truth, truth):
                raise RuntimeError("fixed and OTVDKL rollout truth arrays are misaligned")
            metrics["rollout"][variant][str(horizon)] = evaluate_predictions(
                variant_truth,
                variant_rollout,
            )
            arrays[f"{variant}_rollout_h{horizon}_true"] = variant_truth
            arrays[f"{variant}_rollout_h{horizon}_prediction"] = variant_rollout

    non_finite_arrays = [
        name for name, values in arrays.items() if not np.all(np.isfinite(values))
    ]
    if non_finite_arrays:
        raise FloatingPointError(
            f"refusing to save non-finite prediction arrays: {non_finite_arrays}"
        )
    arrays_path = paths.artifact_dir / "prediction_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    save_json(paths.artifact_dir / "metrics.json", metrics)
    figure_paths: list[str] = []
    for variant in variants:
        save_json(
            paths.artifact_dir / f"{variant}_update_history.json",
            evidence[variant]["history"],
        )
        updater = updaters[variant]
        updater.save_checkpoint(
            paths.artifact_dir / f"final_{variant}_checkpoint.npz",
            pending=evidence[variant]["pending"],
        )
        figure_paths += plot_prediction_states(
            arrays["states_true"], arrays[f"{variant}_one_step"], paths.figures_dir, f"{variant}_one_step"
        )
        figure_paths += plot_prediction_errors(
            arrays["states_true"], arrays[f"{variant}_one_step"], paths.figures_dir, f"{variant}_one_step"
        )

    manifest = {
        "method": "otvdkl",
        "reference": "Zhang et al. sliding-window online time-varying deep Koopman learning",
        "variants": variants,
        "run_type": args.run_type,
        "run_id": paths.run_id,
        "entry_script": "prediction/otvdkl_prediction.py",
        "argv": [sys.executable, *sys.argv],
        "seed": args.seed,
        "device": str(model.device),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "source": {
            "dkuc_artifact_dir": str(artifact_dir),
            "dkuc_artifact_fingerprint": fingerprint,
            "history_dataset": str(history_path),
            "stream_dataset": str(stream_path),
        },
        "state_order": list(STATE_ORDER),
        "input_order": list(INPUT_ORDER),
        "config": {
            "window_size": args.window_size,
            "batch_size": args.batch_size,
            "history_window_start": args.history_window_start,
            "ridge_lambda": args.ridge_lambda,
            "epsilon": args.epsilon,
            "new_batch_policy": "discard_after_evaluation",
            "prediction_readout": "learned_C_tau",
            "improvement_tolerance": args.improvement_tolerance,
            "oracle_tolerance": args.oracle_tolerance,
            "rollout_horizons": horizons,
            "rollout_stride": args.rollout_stride,
            "encoder_frozen": True,
        },
        "fixed_dkuc_latent_stability": {
            "A_spectral_radius": fixed_spectral_radius,
            "A_spectral_norm": fixed_spectral_norm,
        },
        "update_summary": {
            variant: evidence[variant]["summary"] for variant in variants
        },
        "metrics": metrics,
        "artifacts": {
            "arrays": str(arrays_path),
            "figures": figure_paths,
            "model_checkpoints": {
                variant: str(paths.artifact_dir / f"final_{variant}_checkpoint.npz")
                for variant in variants
            },
        },
    }
    save_json(paths.artifact_dir / "manifest.json", manifest)
    print(f"[done] OTVDKL artifacts -> {paths.artifact_dir}")


if __name__ == "__main__":
    main()
