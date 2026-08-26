"""Fixed-length sliding-window Koopman least-squares updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .least_squares import LeastSquaresResult, build_regressor, direct_refit, sufficient_statistics


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
