"""Accumulative fixed-encoder Koopman matrix updater for DKTV Plan 02."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .least_squares import (
    LeastSquaresResult,
    build_regressor,
    solve_statistics,
    sufficient_statistics,
)


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    reason: str
    update_index: int
    model_version: int
    previous_sample_count: int
    batch_sample_count: int
    cumulative_sample_count: int
    update_time_s: float
    pre_update_batch_rmse: float | None
    post_update_batch_rmse: float | None
    diagnostics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _latent_rmse(A: np.ndarray, B: np.ndarray, z: np.ndarray, u: np.ndarray, y: np.ndarray) -> float:
    error = z @ A.T + u @ B.T - y
    return float(np.sqrt(np.mean(error * error)))


class AccumulativeKoopmanUpdater:
    """Maintain additive Gram/cross statistics and never remove old data."""

    state_schema_version = 1

    def __init__(
        self,
        *,
        gram: np.ndarray,
        cross: np.ndarray,
        sample_count: int,
        A: np.ndarray,
        B: np.ndarray,
        ridge_lambda: float,
        encoder_fingerprint: str,
        affine_constant: bool = True,
        update_index: int = 0,
        model_version: int = 0,
    ) -> None:
        self.gram = np.asarray(gram, dtype=np.float64).copy()
        self.cross = np.asarray(cross, dtype=np.float64).copy()
        self.A = np.asarray(A, dtype=np.float64).copy()
        self.B = np.asarray(B, dtype=np.float64).copy()
        self.sample_count = int(sample_count)
        self.ridge_lambda = float(ridge_lambda)
        self.encoder_fingerprint = str(encoder_fingerprint)
        self.affine_constant = bool(affine_constant)
        self.update_index = int(update_index)
        self.model_version = int(model_version)
        self.latent_dim = int(self.A.shape[0])
        self.input_dim = int(self.B.shape[1])
        expected_dim = self.latent_dim + self.input_dim
        if self.A.shape != (self.latent_dim, self.latent_dim):
            raise ValueError("A must be square")
        if self.B.shape[0] != self.latent_dim:
            raise ValueError("A/B latent dimensions do not match")
        if self.gram.shape != (expected_dim, expected_dim):
            raise ValueError("gram shape does not match A/B")
        if self.cross.shape != (expected_dim, self.latent_dim):
            raise ValueError("cross shape does not match A/B")
        if self.sample_count <= 0 or self.ridge_lambda <= 0.0:
            raise ValueError("sample_count and ridge_lambda must be positive")
        if not self.encoder_fingerprint:
            raise ValueError("encoder_fingerprint must be non-empty")
        if not all(
            np.all(np.isfinite(values))
            for values in (self.gram, self.cross, self.A, self.B)
        ):
            raise ValueError("initial updater state must be finite")

    @classmethod
    def from_history(
        cls,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        A0: np.ndarray,
        B0: np.ndarray,
        ridge_lambda: float,
        encoder_fingerprint: str,
        affine_constant: bool = True,
    ) -> "AccumulativeKoopmanUpdater":
        regressor = build_regressor(z_current, u_normalized)
        target = np.asarray(z_next, dtype=np.float64)
        if target.shape != np.asarray(z_current).shape or not np.all(np.isfinite(target)):
            raise ValueError("z_next must be finite and match z_current")
        gram, cross = sufficient_statistics(regressor, target)
        return cls(
            gram=gram,
            cross=cross,
            sample_count=regressor.shape[0],
            A=A0,
            B=B0,
            ridge_lambda=ridge_lambda,
            encoder_fingerprint=encoder_fingerprint,
            affine_constant=affine_constant,
        )

    @property
    def statistics_memory_bytes(self) -> int:
        return int(self.gram.nbytes + self.cross.nbytes)

    def update(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        encoder_fingerprint: str,
    ) -> tuple[UpdateResult, LeastSquaresResult | None]:
        """Add one batch and accept the finite ridge candidate."""
        started = perf_counter()
        previous_count = self.sample_count
        batch_count = int(np.asarray(z_current).shape[0]) if np.asarray(z_current).ndim else 0
        next_update_index = self.update_index + 1
        if str(encoder_fingerprint) != self.encoder_fingerprint:
            raise ValueError("encoder fingerprint changed; accumulated statistics are invalid")
        try:
            regressor = build_regressor(z_current, u_normalized)
            target = np.asarray(z_next, dtype=np.float64)
            if target.shape != np.asarray(z_current).shape or not np.all(np.isfinite(target)):
                raise ValueError("z_next must be finite and match z_current")
            batch_gram, batch_cross = sufficient_statistics(regressor, target)
            candidate_gram = self.gram + batch_gram
            candidate_cross = self.cross + batch_cross
            candidate_count = self.sample_count + regressor.shape[0]
            pre_rmse = _latent_rmse(
                self.A,
                self.B,
                np.asarray(z_current, dtype=np.float64),
                np.asarray(u_normalized, dtype=np.float64),
                target,
            )
            candidate = solve_statistics(
                candidate_gram,
                candidate_cross,
                sample_count=candidate_count,
                latent_dim=self.latent_dim,
                input_dim=self.input_dim,
                ridge_lambda=self.ridge_lambda,
                affine_constant=self.affine_constant,
            )
            if not candidate.diagnostics.finite:
                raise FloatingPointError("candidate A/B contains non-finite values")
            post_rmse = _latent_rmse(
                candidate.A,
                candidate.B,
                np.asarray(z_current, dtype=np.float64),
                np.asarray(u_normalized, dtype=np.float64),
                target,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            elapsed = perf_counter() - started
            return (
                UpdateResult(
                    accepted=False,
                    reason=f"failed_numerical:{type(error).__name__}:{error}",
                    update_index=next_update_index,
                    model_version=self.model_version,
                    previous_sample_count=previous_count,
                    batch_sample_count=batch_count,
                    cumulative_sample_count=self.sample_count,
                    update_time_s=float(elapsed),
                    pre_update_batch_rmse=None,
                    post_update_batch_rmse=None,
                    diagnostics=None,
                ),
                None,
            )

        self.gram = candidate_gram
        self.cross = candidate_cross
        self.sample_count = candidate_count
        self.A = candidate.A.copy()
        self.B = candidate.B.copy()
        self.update_index = next_update_index
        self.model_version += 1
        elapsed = perf_counter() - started
        return (
            UpdateResult(
                accepted=True,
                reason="accepted",
                update_index=self.update_index,
                model_version=self.model_version,
                previous_sample_count=previous_count,
                batch_sample_count=regressor.shape[0],
                cumulative_sample_count=self.sample_count,
                update_time_s=float(elapsed),
                pre_update_batch_rmse=pre_rmse,
                post_update_batch_rmse=post_rmse,
                diagnostics=candidate.diagnostics.to_dict(),
            ),
            candidate,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            state_schema_version=np.array(self.state_schema_version, dtype=np.int64),
            gram=self.gram,
            cross=self.cross,
            sample_count=np.array(self.sample_count, dtype=np.int64),
            A=self.A,
            B=self.B,
            ridge_lambda=np.array(self.ridge_lambda, dtype=np.float64),
            encoder_fingerprint=np.array(self.encoder_fingerprint),
            affine_constant=np.array(self.affine_constant, dtype=bool),
            update_index=np.array(self.update_index, dtype=np.int64),
            model_version=np.array(self.model_version, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AccumulativeKoopmanUpdater":
        with np.load(Path(path), allow_pickle=False) as payload:
            schema = int(payload["state_schema_version"])
            if schema != cls.state_schema_version:
                raise ValueError(f"unsupported updater state schema: {schema}")
            return cls(
                gram=payload["gram"],
                cross=payload["cross"],
                sample_count=int(payload["sample_count"]),
                A=payload["A"],
                B=payload["B"],
                ridge_lambda=float(payload["ridge_lambda"]),
                encoder_fingerprint=str(payload["encoder_fingerprint"]),
                affine_constant=bool(payload["affine_constant"]),
                update_index=int(payload["update_index"]),
                model_version=int(payload["model_version"]),
            )
