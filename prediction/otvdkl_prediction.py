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
    recursive_A_max_abs_difference: float
    recursive_B_max_abs_difference: float
    candidate_A_max_abs_difference: float
    candidate_B_max_abs_difference: float
    addition_system_condition_number: float
    deletion_system_condition_number: float
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
    addition_system_condition_number: float | None
    deletion_system_condition_number: float | None
    recursive_path: str | None
    reject_buffer_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        if self.A.shape != (self.latent_dim, self.latent_dim):
            raise ValueError("A0 shape does not match the window")
        if self.B.shape != (self.latent_dim, self.input_dim):
            raise ValueError("B0 shape does not match the window")
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
        with np.errstate(over="ignore", invalid="ignore"):
            add_system = (
                np.eye(self.batch_size)
                + new_regressor @ self.inverse_regularized_gram @ new_regressor.T
            )
        if not np.all(np.isfinite(add_system)):
            raise FloatingPointError("addition Woodbury system is non-finite")
        add_condition = float(np.linalg.cond(add_system))
        if not np.isfinite(add_condition) or add_condition > self.low_dim_condition_limit:
            raise np.linalg.LinAlgError("addition Woodbury system is ill-conditioned")
        with np.errstate(over="ignore", invalid="ignore"):
            add_inverse = self.inverse_regularized_gram - (
                self.inverse_regularized_gram
                @ new_regressor.T
                @ np.linalg.solve(add_system, new_regressor @ self.inverse_regularized_gram)
            )
            delete_system = (
                np.eye(self.batch_size)
                - outgoing_regressor @ add_inverse @ outgoing_regressor.T
            )
        if not np.all(np.isfinite(delete_system)):
            raise FloatingPointError("deletion Woodbury system is non-finite")
        delete_condition = float(np.linalg.cond(delete_system))
        if not np.isfinite(delete_condition) or delete_condition > self.low_dim_condition_limit:
            raise np.linalg.LinAlgError("deletion Woodbury system is ill-conditioned")
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_inverse = add_inverse + (
                add_inverse
                @ outgoing_regressor.T
                @ np.linalg.solve(delete_system, outgoing_regressor @ add_inverse)
            )
        candidate_inverse = 0.5 * (candidate_inverse + candidate_inverse.T)
        with np.errstate(over="ignore", invalid="ignore"):
            add_gram, add_cross = sufficient_statistics(new_regressor, new_next)
            out_gram, out_cross = sufficient_statistics(outgoing_regressor, outgoing_next)
        if not all(np.all(np.isfinite(values)) for values in (add_gram, add_cross, out_gram, out_cross)):
            raise FloatingPointError("batch sufficient statistics are non-finite")
        candidate_gram = self.gram + add_gram - out_gram
        candidate_gram = 0.5 * (candidate_gram + candidate_gram.T)
        candidate_cross = self.cross + add_cross - out_cross
        if not np.all(np.isfinite(candidate_gram)) or not np.all(np.isfinite(candidate_cross)):
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
        direct_refit_time = perf_counter() - direct_started
        if direct.diagnostics.rank < self.regressor_dim:
            raise np.linalg.LinAlgError("candidate window regressor is rank deficient")
        if (
            not np.isfinite(direct.diagnostics.condition_number)
            or direct.diagnostics.condition_number > self.window_condition_limit
        ):
            raise np.linalg.LinAlgError("candidate window regressor is ill-conditioned")
        recursive_resume = perf_counter()
        recursive = self._recursive_result(candidate_inverse, candidate_cross, direct)
        recursive_candidate_time += perf_counter() - recursive_resume
        with np.errstate(over="ignore", invalid="ignore"):
            A_difference = float(np.max(np.abs(recursive.A - direct.A)))
            B_difference = float(np.max(np.abs(recursive.B - direct.B)))
        recursive_finite = all(
            np.all(np.isfinite(values)) for values in (recursive.A, recursive.B, recursive.theta)
        )
        differences_finite = bool(np.isfinite(A_difference) and np.isfinite(B_difference))
        path = "woodbury_add_delete"
        fallback_time = 0.0
        if (
            not recursive_finite
            or not differences_finite
            or not np.all(np.isfinite(candidate_inverse))
            or A_difference > self.oracle_tolerance
            or B_difference > self.oracle_tolerance
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
            if not np.all(np.isfinite(candidate_inverse)):
                raise FloatingPointError("fallback inverse is non-finite")
            recursive = direct
            path = "direct_refit_fallback"
            fallback_time = perf_counter() - fallback_started
        candidate_A_difference = float(np.max(np.abs(recursive.A - direct.A)))
        candidate_B_difference = float(np.max(np.abs(recursive.B - direct.B)))
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
            recursive_A_max_abs_difference=A_difference,
            recursive_B_max_abs_difference=B_difference,
            candidate_A_max_abs_difference=candidate_A_difference,
            candidate_B_max_abs_difference=candidate_B_difference,
            addition_system_condition_number=add_condition,
            deletion_system_condition_number=delete_condition,
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
        self.window_version += 1
        if accept_model:
            self.A = candidate.result.A.copy()
            self.B = candidate.result.B.copy()
            self.model_version += 1

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
        policy: str = "not_applicable",
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
            addition_system_condition_number=(
                candidate.addition_system_condition_number if candidate else None
            ),
            deletion_system_condition_number=(
                candidate.deletion_system_condition_number if candidate else None
            ),
            recursive_path=candidate.recursive_path if candidate else None,
            reject_buffer_policy=policy,
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
            current_rmse = latent_rmse(self.A, self.B, z_current, u_normalized, z_next)
            candidate = self.propose(
                z_current,
                z_next,
                u_normalized,
                sample_ids=new_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            candidate_rmse = latent_rmse(
                candidate.result.A, candidate.result.B, z_current, u_normalized, z_next
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

    supported_policies = ("discard_on_reject", "retain_on_reject")

    def __init__(
        self,
        updater: SlidingWindowKoopmanUpdater,
        *,
        epsilon: float,
        reject_buffer_policy: str = "discard_on_reject",
        improvement_tolerance: float = 0.0,
    ) -> None:
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be non-negative")
        if reject_buffer_policy not in self.supported_policies:
            raise ValueError(f"unsupported reject buffer policy: {reject_buffer_policy}")
        self.updater = updater
        self.epsilon = float(epsilon)
        self.reject_buffer_policy = reject_buffer_policy
        self.improvement_tolerance = float(improvement_tolerance)

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
            current_rmse = latent_rmse(base.A, base.B, z_current, u_normalized, z_next)
            candidate = base.propose(
                z_current,
                z_next,
                u_normalized,
                sample_ids=new_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            candidate_rmse = latent_rmse(
                candidate.result.A, candidate.result.B, z_current, u_normalized, z_next
            )
            if current_rmse <= self.epsilon:
                base.commit(candidate, accept_model=False)
                return base._record(
                    status="skipped_threshold",
                    reason="current_batch_rmse_not_above_epsilon",
                    started=started,
                    new_ids=new_ids,
                    evicted_ids=evicted,
                    candidate=candidate,
                    current_rmse=current_rmse,
                    candidate_rmse=candidate_rmse,
                    window_advanced=True,
                    policy=self.reject_buffer_policy,
                ), candidate
            if candidate_rmse + self.improvement_tolerance >= current_rmse:
                retain = self.reject_buffer_policy == "retain_on_reject"
                if retain:
                    base.commit(candidate, accept_model=False)
                return base._record(
                    status="rejected",
                    reason="candidate_not_better_on_new_batch",
                    started=started,
                    new_ids=new_ids,
                    evicted_ids=evicted,
                    candidate=candidate,
                    current_rmse=current_rmse,
                    candidate_rmse=candidate_rmse,
                    window_advanced=retain,
                    policy=self.reject_buffer_policy,
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
                policy=self.reject_buffer_policy,
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
                policy=self.reject_buffer_policy,
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
    reject_buffer_policy: str,
    improvement_tolerance: float,
    oracle_tolerance: float,
    low_dim_condition_limit: float,
    window_condition_limit: float,
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
    base = SlidingWindowKoopmanUpdater(
        history_current[-window_size:],
        history_target[-window_size:],
        history_input[-window_size:],
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
    if variant == "otvdkl_star":
        updater = SelectiveWindowKoopmanUpdater(
            base,
            epsilon=float(epsilon),
            reject_buffer_policy=reject_buffer_policy,
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
    versions = np.empty(steps, dtype=np.int64)
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
        versions[time_step] = updater.model_version
        prediction[:, time_step + 1] = predict_dkuc_latent_batch(
            model, updater.A, updater.B, stream_z[:, time_step], stream_u[:, time_step]
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
        "reject_buffer_policy": (
            reject_buffer_policy if variant == "otvdkl_star" else "not_applicable"
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
        "oracle_tolerance_passed": all(
            float(record["candidate_A_max_abs_difference"]) <= float(oracle_tolerance)
            and float(record["candidate_B_max_abs_difference"]) <= float(oracle_tolerance)
            for record in candidates
        ),
    }
    arrays = {
        "states_true": states,
        "inputs": np.asarray(stream_data["inputs"], dtype=np.float64),
        f"{variant}_one_step": prediction,
        f"{variant}_A_by_step": A_by_step,
        f"{variant}_B_by_step": B_by_step,
        f"{variant}_model_version_by_step": versions,
    }
    return updater, {"summary": summary, "history": update_history}, arrays


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
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--reject_buffer_policy",
        choices=list(SelectiveWindowKoopmanUpdater.supported_policies),
        default="discard_on_reject",
    )
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
            reject_buffer_policy=args.reject_buffer_policy,
            improvement_tolerance=args.improvement_tolerance,
            oracle_tolerance=args.oracle_tolerance,
            low_dim_condition_limit=args.low_dim_condition_limit,
            window_condition_limit=args.window_condition_limit,
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
    horizons = _effective_horizons(args.rollout_horizons, steps)
    metrics: dict[str, Any] = {
        "one_step": {
            "fixed_dkuc": evaluate_predictions(arrays["states_true"], fixed_prediction)
        },
        "rollout": {"fixed_dkuc": {}},
    }
    model_snapshots: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "fixed_dkuc": (
            np.broadcast_to(model.A, (steps, *model.A.shape)).copy(),
            np.broadcast_to(model.B, (steps, *model.B.shape)).copy(),
        )
    }
    for variant in variants:
        metrics["one_step"][variant] = evaluate_predictions(
            arrays["states_true"], arrays[f"{variant}_one_step"]
        )
        metrics["rollout"][variant] = {}
        model_snapshots[variant] = (
            arrays[f"{variant}_A_by_step"], arrays[f"{variant}_B_by_step"]
        )
    for method, (A_values, B_values) in model_snapshots.items():
        for horizon in horizons:
            truth, prediction = snapshot_rollout_predictions(
                model,
                stream_data,
                A_values,
                B_values,
                horizon=horizon,
                stride=args.rollout_stride,
            )
            metrics["rollout"][method][str(horizon)] = evaluate_predictions(truth, prediction)
            arrays[f"{method}_rollout_h{horizon}_true"] = truth
            arrays[f"{method}_rollout_h{horizon}_prediction"] = prediction

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
        np.savez_compressed(
            paths.artifact_dir / f"final_{variant}_state.npz",
            A=updater.A,
            B=updater.B,
            sample_ids=updater.sample_ids,
            z_current=updater.z_current,
            z_next=updater.z_next,
            u_normalized=updater.u_normalized,
            model_version=np.array(updater.model_version, dtype=np.int64),
            window_version=np.array(updater.window_version, dtype=np.int64),
            encoder_fingerprint=np.array(fingerprint),
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
            "ridge_lambda": args.ridge_lambda,
            "epsilon": args.epsilon,
            "reject_buffer_policy": args.reject_buffer_policy,
            "improvement_tolerance": args.improvement_tolerance,
            "oracle_tolerance": args.oracle_tolerance,
            "rollout_horizons": horizons,
            "rollout_stride": args.rollout_stride,
            "encoder_frozen": True,
        },
        "update_summary": {
            variant: evidence[variant]["summary"] for variant in variants
        },
        "metrics": metrics,
        "artifacts": {
            "arrays": str(arrays_path),
            "figures": figure_paths,
            "final_states": {
                variant: str(paths.artifact_dir / f"final_{variant}_state.npz")
                for variant in variants
            },
        },
    }
    save_json(paths.artifact_dir / "manifest.json", manifest)
    print(f"[done] OTVDKL artifacts -> {paths.artifact_dir}")


if __name__ == "__main__":
    main()
