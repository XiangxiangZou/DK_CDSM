"""Ridge least-squares primitives for fixed-encoder Koopman updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatrixDiagnostics:
    sample_count: int
    regressor_dim: int
    rank: int
    minimum_singular_value: float
    condition_number: float
    regularized_condition_number: float
    spectral_radius_A: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeastSquaresResult:
    A: np.ndarray
    B: np.ndarray
    theta: np.ndarray
    diagnostics: MatrixDiagnostics


def build_regressor(z_current: np.ndarray, u_normalized: np.ndarray) -> np.ndarray:
    """Return row-stacked ``[z_k, u_norm,k]`` samples."""
    z = np.asarray(z_current, dtype=np.float64)
    u = np.asarray(u_normalized, dtype=np.float64)
    if z.ndim != 2 or u.ndim != 2 or z.shape[0] != u.shape[0]:
        raise ValueError("z_current and u_normalized must be aligned 2-D arrays")
    if z.shape[0] == 0:
        raise ValueError("at least one snapshot is required")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(u)):
        raise ValueError("regressor inputs must be finite")
    return np.concatenate([z, u], axis=1)


def sufficient_statistics(
    regressor: np.ndarray,
    z_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R.T@R`` and ``R.T@Y`` without retaining raw samples."""
    values = np.asarray(regressor, dtype=np.float64)
    target = np.asarray(z_next, dtype=np.float64)
    if values.ndim != 2 or target.ndim != 2 or values.shape[0] != target.shape[0]:
        raise ValueError("regressor and z_next must be aligned 2-D arrays")
    if values.shape[0] == 0:
        raise ValueError("at least one snapshot is required")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
        raise ValueError("least-squares samples must be finite")
    return values.T @ values, values.T @ target


def solve_statistics(
    gram: np.ndarray,
    cross: np.ndarray,
    *,
    sample_count: int,
    latent_dim: int,
    input_dim: int,
    ridge_lambda: float,
    affine_constant: bool,
    singular_values: np.ndarray | None = None,
) -> LeastSquaresResult:
    """Solve the common ridge system from accumulated statistics."""
    gram_values = np.asarray(gram, dtype=np.float64)
    cross_values = np.asarray(cross, dtype=np.float64)
    regressor_dim = int(latent_dim) + int(input_dim)
    if gram_values.shape != (regressor_dim, regressor_dim):
        raise ValueError("gram has an incompatible shape")
    if cross_values.shape != (regressor_dim, int(latent_dim)):
        raise ValueError("cross has an incompatible shape")
    if int(sample_count) <= 0 or float(ridge_lambda) <= 0.0:
        raise ValueError("sample_count and ridge_lambda must be positive")
    if not np.all(np.isfinite(gram_values)) or not np.all(np.isfinite(cross_values)):
        raise ValueError("sufficient statistics must be finite")

    regularized = gram_values + float(ridge_lambda) * np.eye(regressor_dim)
    theta = np.linalg.solve(regularized, cross_values)
    A = theta[:latent_dim].T.copy()
    B = theta[latent_dim:].T.copy()
    if affine_constant:
        A[-1] = 0.0
        A[-1, -1] = 1.0
        B[-1] = 0.0
        theta = np.concatenate([A, B], axis=1).T

    if singular_values is None:
        eigenvalues = np.linalg.eigvalsh(0.5 * (gram_values + gram_values.T))
        singular = np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1]
    else:
        singular = np.asarray(singular_values, dtype=np.float64)
    maximum = float(singular[0]) if singular.size else 0.0
    minimum = float(singular[-1]) if singular.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(regressor_dim, int(sample_count)) * maximum
    rank = int(np.count_nonzero(singular > tolerance))
    condition = (
        float(maximum / minimum)
        if rank == regressor_dim and minimum > 0.0
        else float("inf")
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    diagnostics = MatrixDiagnostics(
        sample_count=int(sample_count),
        regressor_dim=regressor_dim,
        rank=rank,
        minimum_singular_value=minimum,
        condition_number=condition,
        regularized_condition_number=float(np.linalg.cond(regularized)),
        spectral_radius_A=spectral_radius,
        finite=bool(np.all(np.isfinite(A)) and np.all(np.isfinite(B))),
    )
    return LeastSquaresResult(A=A, B=B, theta=theta, diagnostics=diagnostics)


def direct_refit(
    z_current: np.ndarray,
    z_next: np.ndarray,
    u_normalized: np.ndarray,
    *,
    ridge_lambda: float,
    affine_constant: bool = True,
) -> LeastSquaresResult:
    """Fit ``A/B`` directly from all supplied historical snapshots."""
    z = np.asarray(z_current, dtype=np.float64)
    target = np.asarray(z_next, dtype=np.float64)
    u = np.asarray(u_normalized, dtype=np.float64)
    regressor = build_regressor(z, u)
    if target.shape != z.shape or not np.all(np.isfinite(target)):
        raise ValueError("z_next must be finite and match z_current")
    gram, cross = sufficient_statistics(regressor, target)
    singular = np.linalg.svd(regressor, compute_uv=False)
    return solve_statistics(
        gram,
        cross,
        sample_count=regressor.shape[0],
        latent_dim=z.shape[1],
        input_dim=u.shape[1],
        ridge_lambda=ridge_lambda,
        affine_constant=affine_constant,
        singular_values=singular,
    )
