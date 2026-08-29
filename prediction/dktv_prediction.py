"""Hao et al. accumulative DKTV prediction and online-update entry point."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
        solve_statistics,
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
        solve_statistics,
        sufficient_statistics,
    )
    from dkuc_prediction import DKUCModel


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
                candidate_gram, candidate_cross, sample_count=candidate_count,
                latent_dim=self.latent_dim, input_dim=self.input_dim,
                ridge_lambda=self.ridge_lambda, affine_constant=self.affine_constant,
            )
            if not candidate.diagnostics.finite:
                raise FloatingPointError("candidate A/B contains non-finite values")
            post_rmse = _latent_rmse(candidate.A, candidate.B, np.asarray(z_current),
                                    np.asarray(u_normalized), target)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return (UpdateResult(False, f"failed_numerical:{type(error).__name__}:{error}",
                    next_update_index, self.model_version, previous_count, batch_count,
                    self.sample_count, perf_counter() - started, None, None, None), None)
        self.gram, self.cross, self.sample_count = candidate_gram, candidate_cross, candidate_count
        self.A, self.B = candidate.A.copy(), candidate.B.copy()
        self.update_index, self.model_version = next_update_index, self.model_version + 1
        return (UpdateResult(True, "accepted", self.update_index, self.model_version,
                previous_count, regressor.shape[0], self.sample_count, perf_counter() - started,
                pre_rmse, post_rmse, candidate.diagnostics.to_dict()), candidate)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, state_schema_version=self.state_schema_version, gram=self.gram,
                            cross=self.cross, sample_count=self.sample_count, A=self.A, B=self.B,
                            ridge_lambda=self.ridge_lambda, encoder_fingerprint=self.encoder_fingerprint,
                            affine_constant=self.affine_constant, update_index=self.update_index,
                            model_version=self.model_version)

    @classmethod
    def load(cls, path: str | Path) -> "AccumulativeKoopmanUpdater":
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["state_schema_version"]) != cls.state_schema_version:
                raise ValueError("unsupported updater state schema")
            return cls(gram=payload["gram"], cross=payload["cross"],
                       sample_count=int(payload["sample_count"]), A=payload["A"], B=payload["B"],
                       ridge_lambda=float(payload["ridge_lambda"]),
                       encoder_fingerprint=str(payload["encoder_fingerprint"]),
                       affine_constant=bool(payload["affine_constant"]),
                       update_index=int(payload["update_index"]), model_version=int(payload["model_version"]))


def _ridge_inverse(gram: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Return a symmetric regularized inverse without using a naked inverse."""
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram matrix must be square")
    if ridge_lambda < 0.0 or not np.all(np.isfinite(matrix)):
        raise ValueError("ridge_lambda and gram matrix must be finite and non-negative")
    regularized = matrix + float(ridge_lambda) * np.eye(matrix.shape[0])
    return np.linalg.solve(regularized, np.eye(matrix.shape[0]))


def _matrix_diagnostics(regressor: np.ndarray, matrix: np.ndarray,
                        regularized_system: np.ndarray | None = None) -> dict[str, Any]:
    singular = np.linalg.svd(regressor, compute_uv=False)
    minimum = float(singular[-1]) if singular.size else 0.0
    maximum = float(singular[0]) if singular.size else 0.0
    result = {
        "rank": int(np.linalg.matrix_rank(regressor)),
        "minimum_singular_value": minimum,
        "condition_number": float(maximum / minimum) if minimum > 0.0 else float("inf"),
        "spectral_norm": float(np.linalg.norm(matrix, ord=2)),
    }
    if regularized_system is not None:
        result["regularized_condition_number"] = float(np.linalg.cond(regularized_system))
        result["regularized_rank"] = int(np.linalg.matrix_rank(regularized_system))
    return result


class HaoDKTVState:
    """Atomic A/B/C ridge state for one fixed encoder coordinate system.

    ``update`` implements the accumulative Woodbury recursion when theta is
    fixed.  The full replay separately rebuilds this state from all consumed
    physical transitions after accepting a new theta.  That coordinate-consistent
    engineering strategy is linear in accumulated sample count per online batch.
    """

    state_schema_version = 2

    def __init__(
        self,
        *,
        gram_chi: np.ndarray,
        cross_chi: np.ndarray,
        gram_g: np.ndarray,
        cross_g: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        P_chi: np.ndarray,
        P_g: np.ndarray,
        sample_count: int,
        ridge_lambda: float,
        update_index: int = 0,
        model_version: int = 0,
        encoder_version: int = 0,
    ) -> None:
        for name, value in vars().copy().items():
            if name in {"self", "sample_count", "ridge_lambda", "update_index", "model_version", "encoder_version"}:
                continue
            setattr(self, name, np.asarray(value, dtype=np.float64).copy())
        self.sample_count = int(sample_count)
        self.ridge_lambda = float(ridge_lambda)
        self.update_index = int(update_index)
        self.model_version = int(model_version)
        self.encoder_version = int(encoder_version)
        self.latent_dim = int(self.A.shape[0])
        self.input_dim = int(self.B.shape[1])
        self.state_dim = int(self.C.shape[0])
        expected = self.latent_dim + self.input_dim
        shapes = {
            "A": (self.latent_dim, self.latent_dim), "B": (self.latent_dim, self.input_dim),
            "C": (self.state_dim, self.latent_dim), "gram_chi": (expected, expected),
            "cross_chi": (expected, self.latent_dim), "P_chi": (expected, expected),
            "gram_g": (self.latent_dim, self.latent_dim),
            "cross_g": (self.latent_dim, self.state_dim), "P_g": (self.latent_dim, self.latent_dim),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        if self.sample_count <= 0 or self.ridge_lambda < 0.0:
            raise ValueError("sample_count must be positive and ridge_lambda non-negative")

    @classmethod
    def from_history(cls, z: np.ndarray, z_next: np.ndarray, u: np.ndarray, x_norm: np.ndarray,
                     *, ridge_lambda: float) -> "HaoDKTVState":
        z, z_next, u, x_norm = (np.asarray(v, dtype=np.float64) for v in (z, z_next, u, x_norm))
        if z.ndim != 2 or z_next.shape != z.shape or u.ndim != 2 or x_norm.ndim != 2:
            raise ValueError("history arrays must be aligned two-dimensional samples")
        if not (z.shape[0] == u.shape[0] == x_norm.shape[0]) or not all(np.all(np.isfinite(v)) for v in (z, z_next, u, x_norm)):
            raise ValueError("history arrays must be aligned and finite")
        chi = np.concatenate((z, u), axis=1)
        gram_chi, cross_chi = chi.T @ chi, chi.T @ z_next
        gram_g, cross_g = z.T @ z, z.T @ x_norm
        P_chi = _ridge_inverse(gram_chi, ridge_lambda)
        P_g = _ridge_inverse(gram_g, ridge_lambda)
        K = cross_chi.T @ P_chi
        C = cross_g.T @ P_g
        return cls(gram_chi=gram_chi, cross_chi=cross_chi, gram_g=gram_g,
                   cross_g=cross_g, A=K[:, :z.shape[1]], B=K[:, z.shape[1]:], C=C,
                   P_chi=P_chi, P_g=P_g, sample_count=z.shape[0], ridge_lambda=ridge_lambda)

    @staticmethod
    def _woodbury(P: np.ndarray, rows: np.ndarray) -> np.ndarray:
        middle = np.eye(rows.shape[0]) + rows @ P @ rows.T
        solved = np.linalg.solve(middle, rows @ P)
        updated = P - P @ rows.T @ solved
        return 0.5 * (updated + updated.T)

    def update(self, z: np.ndarray, z_next: np.ndarray, u: np.ndarray,
               x_norm: np.ndarray) -> dict[str, Any]:
        """Atomically add one observed batch and update A/B/C and inverse states."""
        started = perf_counter()
        try:
            z, z_next, u, x_norm = (np.asarray(v, dtype=np.float64) for v in (z, z_next, u, x_norm))
            if z.ndim != 2 or z_next.shape != z.shape or u.shape != (z.shape[0], self.input_dim) or x_norm.shape != (z.shape[0], self.state_dim):
                raise ValueError("batch arrays have incompatible shapes")
            if not all(np.all(np.isfinite(v)) for v in (z, z_next, u, x_norm)):
                raise ValueError("batch arrays contain non-finite values")
            chi = np.concatenate((z, u), axis=1)
            P_chi = self._woodbury(self.P_chi, chi)
            P_g = self._woodbury(self.P_g, z)
            gram_chi = self.gram_chi + chi.T @ chi
            cross_chi = self.cross_chi + chi.T @ z_next
            gram_g = self.gram_g + z.T @ z
            cross_g = self.cross_g + z.T @ x_norm
            K = cross_chi.T @ P_chi
            C = cross_g.T @ P_g
            values = (P_chi, P_g, gram_chi, cross_chi, gram_g, cross_g, K, C)
            if not all(np.all(np.isfinite(v)) for v in values):
                raise FloatingPointError("candidate state contains non-finite values")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return {"accepted": False, "reason": f"{type(error).__name__}:{error}",
                    "update_index": self.update_index + 1, "model_version": self.model_version,
                    "update_time_s": perf_counter() - started}
        self.P_chi, self.P_g = P_chi, P_g
        self.gram_chi, self.cross_chi = gram_chi, cross_chi
        self.gram_g, self.cross_g = gram_g, cross_g
        self.A, self.B, self.C = K[:, :self.latent_dim], K[:, self.latent_dim:], C
        self.sample_count += z.shape[0]
        self.update_index += 1
        self.model_version += 1
        diagnostics = {
                       "chi": _matrix_diagnostics(
                           chi, K, gram_chi + self.ridge_lambda * np.eye(gram_chi.shape[0])
                       ),
                       "g": _matrix_diagnostics(
                           z, C, gram_g + self.ridge_lambda * np.eye(gram_g.shape[0])
                       ),
                       "A_spectral_norm": float(np.linalg.norm(self.A, 2)),
                       "A_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(self.A)))),
                       "B_spectral_norm": float(np.linalg.norm(self.B, 2)),
                       "C_spectral_norm": float(np.linalg.norm(self.C, 2))}
        return {"accepted": True, "reason": "accepted", "update_index": self.update_index,
                "model_version": self.model_version, "sample_count": self.sample_count,
                "update_time_s": perf_counter() - started, "diagnostics": diagnostics}

    def save(self, path: str | Path) -> None:
        np.savez_compressed(Path(path), state_schema_version=self.state_schema_version,
                            gram_chi=self.gram_chi, cross_chi=self.cross_chi,
                            gram_g=self.gram_g, cross_g=self.cross_g, A=self.A, B=self.B, C=self.C,
                            P_chi=self.P_chi, P_g=self.P_g, sample_count=self.sample_count,
                            ridge_lambda=self.ridge_lambda, update_index=self.update_index,
                            model_version=self.model_version, encoder_version=self.encoder_version)

    def diagnostics(self) -> dict[str, Any]:
        """Describe the actually committed cumulative ridge systems and matrices."""
        chi_system = self.gram_chi + self.ridge_lambda * np.eye(self.gram_chi.shape[0])
        g_system = self.gram_g + self.ridge_lambda * np.eye(self.gram_g.shape[0])
        return {
            "chi_regularized_condition_number": float(np.linalg.cond(chi_system)),
            "chi_regularized_rank": int(np.linalg.matrix_rank(chi_system)),
            "g_regularized_condition_number": float(np.linalg.cond(g_system)),
            "g_regularized_rank": int(np.linalg.matrix_rank(g_system)),
            "A_spectral_norm": float(np.linalg.norm(self.A, 2)),
            "A_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(self.A)))),
            "B_spectral_norm": float(np.linalg.norm(self.B, 2)),
            "C_spectral_norm": float(np.linalg.norm(self.C, 2)),
            "finite": all(np.all(np.isfinite(getattr(self, name))) for name in (
                "A", "B", "C", "P_chi", "P_g", "gram_chi", "cross_chi",
                "gram_g", "cross_g"
            )),
        }

    @classmethod
    def load(cls, path: str | Path) -> "HaoDKTVState":
        with np.load(path, allow_pickle=False) as data:
            if int(data["state_schema_version"]) != cls.state_schema_version:
                raise ValueError("unsupported Hao DKTV state schema")
            return cls(**{name: data[name] for name in (
                "gram_chi", "cross_chi", "gram_g", "cross_g", "A", "B", "C", "P_chi", "P_g",
                "sample_count", "ridge_lambda", "update_index", "model_version", "encoder_version")})


def encoder_checksum(network: Any) -> str:
    digest = hashlib.sha256()
    for value in network.encoder.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_online_encoder(model: DKUCModel, state: HaoDKTVState, x: np.ndarray, x_next: np.ndarray,
                         u: np.ndarray, *, loss_weight: float, epochs: int, learning_rate: float,
                         weight_decay: float, grad_clip: float) -> dict[str, Any]:
    """Optimize theta with frozen A/B/C; rollback all encoder parameters on failure."""
    import torch
    if not 0.0 <= loss_weight <= 1.0 or epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("invalid online optimization configuration")
    before_state = copy.deepcopy(model.model.encoder.state_dict())
    before_checksum = encoder_checksum(model.model)
    x_norm = model.x_normer.transform(np.asarray(x)).astype(np.float32)
    next_norm = model.x_normer.transform(np.asarray(x_next)).astype(np.float32)
    u_norm = model.u_normer.transform(np.asarray(u)).astype(np.float32)
    xt = torch.from_numpy(x_norm).to(model.device)
    yt = torch.from_numpy(next_norm).to(model.device)
    ut = torch.from_numpy(u_norm).to(model.device)
    A = torch.as_tensor(state.A, dtype=xt.dtype, device=model.device)
    B = torch.as_tensor(state.B, dtype=xt.dtype, device=model.device)
    C = torch.as_tensor(state.C, dtype=xt.dtype, device=model.device)
    optimizer = torch.optim.AdamW(model.model.encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[list[float]] = []

    def losses():
        z, z_next = model.model.lift(xt), model.model.lift(yt)
        l1 = torch.mean((z_next - z @ A.T - ut @ B.T) ** 2)
        l2 = torch.mean((xt - z @ C.T) ** 2)
        return l1, l2, loss_weight * l1 + (1.0 - loss_weight) * l2
    started = perf_counter()
    try:
        model.model.train()
        with torch.no_grad():
            initial = tuple(float(v.item()) for v in losses())
        best_loss, best_state = initial[2], copy.deepcopy(before_state)
        for epoch in range(epochs):
            l1, l2, total = losses()
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite online loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.model.encoder.parameters(), grad_clip) if grad_clip > 0 else torch.tensor(0.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite encoder gradient")
            optimizer.step()
            with torch.no_grad():
                post_l1, post_l2, post_total = losses()
            values = [float(epoch + 1), float(post_l1.item()), float(post_l2.item()),
                      float(post_total.item())]
            history.append(values)
            if values[-1] < best_loss:
                best_loss, best_state = values[-1], copy.deepcopy(model.model.encoder.state_dict())
        model.model.encoder.load_state_dict(best_state)
        with torch.no_grad():
            final = tuple(float(v.item()) for v in losses())
        if not np.isfinite(final).all():
            raise FloatingPointError("non-finite final online loss")
    except Exception as error:
        model.model.encoder.load_state_dict(before_state)
        model.model.eval()
        return {"accepted": False, "reason": f"{type(error).__name__}:{error}", "initial": None,
                "final": None, "history": history, "training_time_s": perf_counter() - started,
                "checksum_before": before_checksum, "checksum_after": before_checksum}
    model.model.eval()
    after_checksum = encoder_checksum(model.model)
    if after_checksum != before_checksum:
        state.encoder_version += 1
    return {"accepted": True, "reason": "accepted", "initial": dict(zip(("L1", "L2", "L"), initial)),
            "final": dict(zip(("L1", "L2", "L"), final)), "history": history,
            "best_loss": best_loss,
            "training_time_s": perf_counter() - started, "checksum_before": before_checksum,
            "checksum_after": after_checksum, "encoder_version": state.encoder_version}


def _lift_with_network(model: DKUCModel, network: Any, values: np.ndarray) -> np.ndarray:
    """Lift physical states with an explicitly selected network instance."""
    normalized = model.x_normer.transform(np.asarray(values).reshape(-1, model.state_dim)).astype(np.float32)
    with model._torch.no_grad():
        lifted = network.lift(model._torch.from_numpy(normalized).to(model.device))
    return lifted.detach().cpu().numpy().astype(np.float64)


def _consistent_state_from_physical_history(
    model: DKUCModel,
    x: np.ndarray,
    x_next: np.ndarray,
    u: np.ndarray,
    previous: HaoDKTVState,
) -> tuple[HaoDKTVState, dict[str, float]]:
    """Rebuild sufficient statistics so A/B/C share the accepted theta coordinates."""
    relift_started = perf_counter()
    z = _lift_with_network(model, model.model, x)
    z_next = _lift_with_network(model, model.model, x_next)
    normalized_u = model.u_normer.transform(np.asarray(u).reshape(-1, model.control_dim))
    normalized_x = model.x_normer.transform(np.asarray(x).reshape(-1, model.state_dim))
    relift_time = perf_counter() - relift_started
    refit_started = perf_counter()
    rebuilt = HaoDKTVState.from_history(
        z, z_next, normalized_u, normalized_x, ridge_lambda=previous.ridge_lambda
    )
    rebuilt.update_index = previous.update_index
    rebuilt.model_version = previous.model_version
    rebuilt.encoder_version = previous.encoder_version
    return rebuilt, {
        "coordinate_relift_time_s": relift_time,
        "coordinate_refit_time_s": perf_counter() - refit_started,
    }


LONG_ROLLOUT_BURN_IN_STEPS = 200


def _validated_rollout_starts(
    *,
    steps: int,
    horizon: int,
    stride: int,
    start_indices: np.ndarray | None,
) -> np.ndarray:
    if horizon <= 0 or horizon > steps or stride <= 0:
        raise ValueError("rollout horizon and stride are invalid")
    if start_indices is None:
        return np.arange(0, steps - horizon + 1, stride, dtype=np.int64)
    starts = np.asarray(start_indices, dtype=np.int64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("start_indices must be a non-empty one-dimensional array")
    if np.any(starts < 0) or np.any(starts + horizon > steps):
        raise ValueError("start_indices must leave room for the requested horizon")
    if np.unique(starts).size != starts.size or np.any(np.diff(starts) <= 0):
        raise ValueError("start_indices must be unique and strictly increasing")
    return starts


def matched_rollout_start_indices(steps: int, max_horizon: int, stride: int) -> np.ndarray:
    """Return common causal origins for every requested long-horizon metric."""
    last_start = int(steps) - int(max_horizon)
    if last_start < 0 or stride <= 0:
        raise ValueError("max_horizon and stride are incompatible with the stream")
    burn_in = min(LONG_ROLLOUT_BURN_IN_STEPS, last_start)
    starts = np.arange(burn_in, last_start + 1, int(stride), dtype=np.int64)
    if starts.size == 0:
        return np.asarray([last_start], dtype=np.int64)
    if starts[-1] != last_start:
        starts = np.append(starts, last_start)
    return starts


def dktv_snapshot_rollout_evidence(
    model: DKUCModel,
    states: np.ndarray,
    inputs: np.ndarray,
    lifted_by_step: np.ndarray,
    A_by_step: np.ndarray,
    B_by_step: np.ndarray,
    C_by_step: np.ndarray,
    *,
    horizon: int,
    stride: int,
    start_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out frozen online snapshots and retain latent-norm diagnostics."""
    states = np.asarray(states, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64)
    steps = int(inputs.shape[1])
    starts = _validated_rollout_starts(
        steps=steps,
        horizon=int(horizon),
        stride=int(stride),
        start_indices=start_indices,
    )
    truth_windows: list[np.ndarray] = []
    prediction_windows: list[np.ndarray] = []
    latent_norm_windows: list[np.ndarray] = []
    for trial in range(inputs.shape[0]):
        for start in starts:
            start = int(start)
            z = lifted_by_step[trial, start].copy()
            prediction = np.empty((horizon + 1, model.state_dim), dtype=np.float64)
            latent_norm = np.empty(horizon + 1, dtype=np.float64)
            prediction[0] = states[trial, start]
            latent_norm[0] = np.linalg.norm(z)
            A, B, C = A_by_step[trial, start], B_by_step[trial, start], C_by_step[trial, start]
            normalized_u = model.u_normer.transform(inputs[trial, start:start + horizon])
            for offset in range(horizon):
                z = A @ z + B @ normalized_u[offset]
                latent_norm[offset + 1] = np.linalg.norm(z)
                normalized_x = C @ z
                prediction[offset + 1] = model.x_normer.inverse(normalized_x[None])[0]
            truth_windows.append(states[trial, start:start + horizon + 1])
            prediction_windows.append(prediction)
            latent_norm_windows.append(latent_norm)
    return (
        np.asarray(truth_windows),
        np.asarray(prediction_windows),
        np.asarray(latent_norm_windows),
    )


def dktv_snapshot_rollouts(
    model: DKUCModel,
    states: np.ndarray,
    inputs: np.ndarray,
    lifted_by_step: np.ndarray,
    A_by_step: np.ndarray,
    B_by_step: np.ndarray,
    C_by_step: np.ndarray,
    *,
    horizon: int,
    stride: int,
    start_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out frozen online snapshots without consuming future observations."""
    truth, prediction, _ = dktv_snapshot_rollout_evidence(
        model,
        states,
        inputs,
        lifted_by_step,
        A_by_step,
        B_by_step,
        C_by_step,
        horizon=horizon,
        stride=stride,
        start_indices=start_indices,
    )
    return truth, prediction


def long_horizon_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    dt: float,
    latent_norm: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Summarize matched-origin open-loop behavior and retain redraw arrays."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 3 or prediction.shape != truth.shape or truth.shape[1] < 2:
        raise ValueError("long-horizon truth and prediction windows must align")
    error = prediction - truth
    predicted_error = error[:, 1:]
    window_rmse = np.sqrt(np.mean(predicted_error * predicted_error, axis=(1, 2)))
    rmse_by_lead = np.sqrt(np.mean(error * error, axis=(0, 2)))
    rmse_by_state_and_lead = np.sqrt(np.mean(error * error, axis=0))
    error_norm = np.linalg.norm(error, axis=2)
    metrics = evaluate_predictions(truth, prediction)
    metrics.update({
        "horizon_steps": int(truth.shape[1] - 1),
        "horizon_seconds": float((truth.shape[1] - 1) * dt),
        "window_count": int(truth.shape[0]),
        "finite_prediction": bool(np.all(np.isfinite(prediction))),
        "maximum_absolute_prediction": float(np.max(np.abs(prediction))),
        "maximum_state_error_norm": float(np.max(error_norm)),
        "window_rmse_mean": float(np.mean(window_rmse)),
        "window_rmse_std": float(np.std(window_rmse)),
        "window_rmse_p90": float(np.quantile(window_rmse, 0.90)),
        "window_rmse_maximum": float(np.max(window_rmse)),
    })
    diagnostics = {
        "window_rmse": window_rmse,
        "rmse_by_lead_step": rmse_by_lead,
        "rmse_by_state_and_lead_step": rmse_by_state_and_lead,
        "state_error_norm_by_window_and_lead": error_norm,
    }
    if latent_norm is not None:
        latent = np.asarray(latent_norm, dtype=np.float64)
        if latent.shape != truth.shape[:2]:
            raise ValueError("latent_norm must align with rollout windows and lead steps")
        metrics.update({
            "finite_latent_norm": bool(np.all(np.isfinite(latent))),
            "maximum_latent_norm": float(np.max(latent)),
            "latent_norm_p90": float(np.quantile(latent, 0.90)),
        })
        diagnostics["latent_norm_by_window_and_lead"] = latent
    return metrics, diagnostics


def _save_figure_pair(figure: Any, figures_dir: str | Path, stem: str) -> list[str]:
    target = Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = target / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(str(path))
    return paths


def plot_long_horizon_evidence(
    truth: np.ndarray,
    fixed_prediction: np.ndarray,
    dktv_prediction: np.ndarray,
    latent_norm: np.ndarray,
    start_indices: np.ndarray,
    *,
    dt: float,
    figures_dir: str | Path,
) -> list[str]:
    """Save long-horizon error growth, heatmap, representative windows, and latent norms."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [f"plot skipped: {exc}"]

    truth = np.asarray(truth, dtype=np.float64)
    fixed_prediction = np.asarray(fixed_prediction, dtype=np.float64)
    dktv_prediction = np.asarray(dktv_prediction, dtype=np.float64)
    latent_norm = np.asarray(latent_norm, dtype=np.float64)
    lead_time = np.arange(truth.shape[1], dtype=np.float64) * dt
    fixed_error = fixed_prediction - truth
    dktv_error = dktv_prediction - truth
    fixed_lead = np.sqrt(np.mean(fixed_error * fixed_error, axis=(0, 2)))
    dktv_lead = np.sqrt(np.mean(dktv_error * dktv_error, axis=(0, 2)))
    use_log_scale = np.max(fixed_lead) > 20.0 * max(float(np.max(dktv_lead)), 1e-12)
    paths: list[str] = []

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    axes = axes.reshape(-1)
    axes[0].plot(lead_time, fixed_lead, label="fixed DKUC", linewidth=1.7)
    axes[0].plot(lead_time, dktv_lead, label="full DKTV", linewidth=1.7)
    axes[0].set_ylabel("all-state RMSE")
    axes[0].legend(loc="best")
    for dim, label in enumerate(STATE_ORDER):
        ax = axes[dim + 1]
        fixed_state = np.sqrt(np.mean(fixed_error[:, :, dim] ** 2, axis=0))
        dktv_state = np.sqrt(np.mean(dktv_error[:, :, dim] ** 2, axis=0))
        ax.plot(lead_time, fixed_state, label="fixed DKUC", linewidth=1.4)
        ax.plot(lead_time, dktv_state, label="full DKTV", linewidth=1.4)
        ax.set_ylabel(f"{label} RMSE")
    axes[5].axis("off")
    for ax in axes[:5]:
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlabel("prediction lead time [s]")
        if use_log_scale:
            ax.set_yscale("log")
    scale_note = " (log scale)" if use_log_scale else ""
    fig.suptitle(f"Matched-origin long-horizon prediction error growth{scale_note}")
    fig.tight_layout()
    paths += _save_figure_pair(fig, figures_dir, "dktv_long_horizon_error_growth")
    plt.close(fig)

    error_norm = np.linalg.norm(dktv_error, axis=2)
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        error_norm,
        aspect="auto",
        origin="lower",
        extent=(lead_time[0], lead_time[-1], -0.5, error_norm.shape[0] - 0.5),
        cmap="magma",
    )
    ax.set_xlabel("prediction lead time [s]")
    ax.set_ylabel("matched rollout window")
    ax.set_title("Full DKTV state-error norm by origin and lead time")
    fig.colorbar(image, ax=ax, label="state-error norm")
    fig.tight_layout()
    paths += _save_figure_pair(fig, figures_dir, "dktv_long_horizon_error_heatmap")
    plt.close(fig)

    dktv_window_rmse = np.sqrt(np.mean(dktv_error[:, 1:] ** 2, axis=(1, 2)))
    order = np.argsort(dktv_window_rmse)
    representatives = {
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }
    repeated_starts = np.tile(np.asarray(start_indices, dtype=np.int64), truth.shape[0] // len(start_indices))
    for label, index in representatives.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for dim, state_label in enumerate(STATE_ORDER):
            ax = axes.reshape(-1)[dim]
            ax.plot(lead_time, truth[index, :, dim], label="true", linewidth=1.8)
            ax.plot(lead_time, fixed_prediction[index, :, dim], label="fixed DKUC", linestyle=":")
            ax.plot(lead_time, dktv_prediction[index, :, dim], label="full DKTV", linestyle="--")
            reference_scale = max(
                float(np.max(np.abs(truth[index, :, dim]))),
                float(np.max(np.abs(dktv_prediction[index, :, dim]))),
                1e-12,
            )
            if np.max(np.abs(fixed_prediction[index, :, dim])) > 20.0 * reference_scale:
                ax.set_yscale("symlog", linthresh=1e-2)
            ax.set_ylabel(state_label)
            ax.set_xlabel("prediction lead time [s]")
            ax.grid(True, linewidth=0.4, alpha=0.5)
        axes.reshape(-1)[0].legend(loc="best")
        start_step = int(repeated_starts[index])
        fig.suptitle(
            f"{label.capitalize()} long-horizon window: start={start_step * dt:.2f}s, "
            f"DKTV RMSE={dktv_window_rmse[index]:.4f}"
        )
        fig.tight_layout()
        paths += _save_figure_pair(fig, figures_dir, f"dktv_long_horizon_{label}_window_states")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lead_time, np.median(latent_norm, axis=0), label="median", linewidth=1.8)
    ax.plot(lead_time, np.quantile(latent_norm, 0.90, axis=0), label="P90", linewidth=1.5)
    ax.plot(lead_time, np.max(latent_norm, axis=0), label="maximum", linewidth=1.2)
    ax.set_xlabel("prediction lead time [s]")
    ax.set_ylabel("latent-state norm")
    ax.set_title("Full DKTV latent-state norm during long rollouts")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    paths += _save_figure_pair(fig, figures_dir, "dktv_long_horizon_latent_norm")
    plt.close(fig)
    return paths
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


def run_dktv_replay(
    model: DKUCModel,
    history_data: dict[str, np.ndarray],
    stream_data: dict[str, np.ndarray],
    *,
    batch_size: int,
    ridge_lambda: float,
    encoder_fingerprint: str,
) -> tuple[AccumulativeKoopmanUpdater, dict[str, Any], dict[str, np.ndarray]]:
    """Run causal Hao-style accumulation on an explicit online stream."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    history_z, history_next, history_u = lift_dkuc_transitions(model, history_data)
    stream_z, stream_next, stream_u = lift_dkuc_transitions(model, stream_data)
    latent_dim = int(stream_z.shape[-1])
    history_current = history_z.reshape(-1, latent_dim)
    history_target = history_next.reshape(-1, latent_dim)
    history_input = history_u.reshape(-1, model.control_dim)
    updater = AccumulativeKoopmanUpdater.from_history(
        history_current,
        history_target,
        history_input,
        A0=model.A,
        B0=model.B,
        ridge_lambda=float(ridge_lambda),
        encoder_fingerprint=encoder_fingerprint,
        affine_constant=bool(model.config.include_constant),
    )

    states = np.asarray(stream_data["states"], dtype=np.float64)
    steps = int(stream_u.shape[1])
    fixed_prediction = np.zeros_like(states)
    dktv_prediction = np.zeros_like(states)
    fixed_prediction[:, 0] = states[:, 0]
    dktv_prediction[:, 0] = states[:, 0]
    A_by_step = np.empty((steps, latent_dim, latent_dim), dtype=np.float64)
    B_by_step = np.empty((steps, latent_dim, model.control_dim), dtype=np.float64)
    versions = np.empty(steps, dtype=np.int64)
    pending_z = np.empty((0, latent_dim), dtype=np.float64)
    pending_next = np.empty((0, latent_dim), dtype=np.float64)
    pending_u = np.empty((0, model.control_dim), dtype=np.float64)
    update_history: list[dict[str, Any]] = []

    for time_step in range(steps):
        A_by_step[time_step] = updater.A
        B_by_step[time_step] = updater.B
        versions[time_step] = updater.model_version
        fixed_prediction[:, time_step + 1] = predict_dkuc_latent_batch(
            model, model.A, model.B, stream_z[:, time_step], stream_u[:, time_step]
        )
        dktv_prediction[:, time_step + 1] = predict_dkuc_latent_batch(
            model, updater.A, updater.B, stream_z[:, time_step], stream_u[:, time_step]
        )
        pending_z = np.concatenate((pending_z, stream_z[:, time_step]), axis=0)
        pending_next = np.concatenate((pending_next, stream_next[:, time_step]), axis=0)
        pending_u = np.concatenate((pending_u, stream_u[:, time_step]), axis=0)
        while pending_z.shape[0] >= int(batch_size):
            batch_z, pending_z = pending_z[:batch_size], pending_z[batch_size:]
            batch_next, pending_next = pending_next[:batch_size], pending_next[batch_size:]
            batch_u, pending_u = pending_u[:batch_size], pending_u[batch_size:]
            result, _ = updater.update(
                batch_z,
                batch_next,
                batch_u,
                encoder_fingerprint=encoder_fingerprint,
            )
            record = result.to_dict()
            record["time_step"] = int(time_step)
            record["pending_sample_count"] = int(pending_z.shape[0])
            update_history.append(record)

    accepted = [record for record in update_history if record["accepted"]]
    update_times = np.asarray(
        [record["update_time_s"] for record in update_history], dtype=np.float64
    )
    summary = {
        "method": "dktv",
        "update_rule": "hao_accumulative",
        "batch_size": int(batch_size),
        "ridge_lambda": float(ridge_lambda),
        "initial_sample_count": int(history_current.shape[0]),
        "final_sample_count": int(updater.sample_count),
        "update_count": len(update_history),
        "accepted_count": len(accepted),
        "rejected_count": len(update_history) - len(accepted),
        "pending_sample_count": int(pending_z.shape[0]),
        "model_version": int(updater.model_version),
        "all_updates_finite": all(
            record["diagnostics"] is not None and record["diagnostics"]["finite"]
            for record in accepted
        ),
        "mean_update_time_s": float(np.mean(update_times)) if update_times.size else 0.0,
        "maximum_update_time_s": float(np.max(update_times)) if update_times.size else 0.0,
        "statistics_memory_bytes": int(updater.statistics_memory_bytes),
    }
    arrays = {
        "states_true": states,
        "inputs": np.asarray(stream_data["inputs"], dtype=np.float64),
        "fixed_dkuc_one_step": fixed_prediction,
        "dktv_one_step": dktv_prediction,
        "A_by_step": A_by_step,
        "B_by_step": B_by_step,
        "model_version_by_step": versions,
    }
    return updater, {"summary": summary, "history": update_history}, arrays


def run_hao_dktv_replay(
    model: DKUCModel,
    history_data: dict[str, np.ndarray],
    stream_data: dict[str, np.ndarray],
    *,
    mode: str,
    batch_size: int,
    ridge_lambda: float,
    loss_weight: float,
    online_epochs: int,
    online_lr: float,
    online_weight_decay: float,
    grad_clip: float,
    resume_state: HaoDKTVState | None = None,
) -> tuple[HaoDKTVState, dict[str, Any], dict[str, np.ndarray]]:
    """Run each chronological trial independently with predict-observe-update causality."""
    if mode not in {"full", "frozen_encoder"} or batch_size <= 0:
        raise ValueError("mode and batch_size are invalid")
    if mode == "full" and online_epochs <= 0:
        raise ValueError("full mode requires online_epochs > 0")
    states = np.asarray(stream_data["states"], dtype=np.float64)
    inputs = np.asarray(stream_data["inputs"], dtype=np.float64)
    trials, steps = inputs.shape[:2]
    if resume_state is not None and trials != 1:
        raise ValueError("resume_state requires a single stream trial")
    initial_network = copy.deepcopy(model.model.state_dict())
    fixed_network = copy.deepcopy(model.model).to(model.device)
    fixed_network.load_state_dict(initial_network)
    fixed_network.eval()
    for parameter in fixed_network.parameters():
        parameter.requires_grad_(False)
    history_z, history_next, history_u = lift_dkuc_transitions(model, history_data)
    hz = history_z.reshape(-1, history_z.shape[-1])
    hn = history_next.reshape(-1, history_next.shape[-1])
    hu = history_u.reshape(-1, model.control_dim)
    hx = model.x_normer.transform(np.asarray(history_data["states"])[:, :-1].reshape(-1, model.state_dim))
    history_x = np.asarray(history_data["states"], dtype=np.float64)[:, :-1].reshape(-1, model.state_dim)
    history_y = np.asarray(history_data["states"], dtype=np.float64)[:, 1:].reshape(-1, model.state_dim)
    history_u_phys = np.asarray(history_data["inputs"], dtype=np.float64).reshape(-1, model.control_dim)
    template = HaoDKTVState.from_history(hz, hn, hu, hx, ridge_lambda=ridge_lambda)
    prediction = np.zeros_like(states)
    fixed_prediction = np.zeros_like(states)
    prediction[:, 0] = fixed_prediction[:, 0] = states[:, 0]
    latent_dim = template.latent_dim
    A_steps = np.empty((trials, steps, latent_dim, latent_dim))
    B_steps = np.empty((trials, steps, latent_dim, model.control_dim))
    C_steps = np.empty((trials, steps, model.state_dim, latent_dim))
    versions = np.empty((trials, steps), dtype=np.int64)
    encoder_versions = np.empty_like(versions)
    lifted_steps = np.empty((trials, steps, latent_dim), dtype=np.float64)
    update_history: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    final_state = template
    for trial in range(trials):
        model.model.load_state_dict(initial_network)
        model.model.eval()
        state = copy.deepcopy(resume_state if resume_state is not None else template)
        pending_x: list[np.ndarray] = []
        pending_y: list[np.ndarray] = []
        pending_u: list[np.ndarray] = []
        coordinate_x = [value.copy() for value in history_x]
        coordinate_y = [value.copy() for value in history_y]
        coordinate_u = [value.copy() for value in history_u_phys]
        for step in range(steps):
            A_steps[trial, step], B_steps[trial, step], C_steps[trial, step] = state.A, state.B, state.C
            versions[trial, step], encoder_versions[trial, step] = state.model_version, state.encoder_version
            z = model.lift(states[trial, step])
            lifted_steps[trial, step] = z
            fixed_z0 = _lift_with_network(model, fixed_network, states[trial, step])[0]
            un = model.u_normer.transform(inputs[trial, step].reshape(1, -1))[0]
            fixed_z = model.A @ fixed_z0 + model.B @ un
            fixed_prediction[trial, step + 1] = model.recover_state(fixed_z)
            next_z = state.A @ z + state.B @ un
            next_normalized = state.C @ next_z
            prediction[trial, step + 1] = model.x_normer.inverse(next_normalized.reshape(1, -1))[0]
            pending_x.append(states[trial, step].copy())
            pending_y.append(states[trial, step + 1].copy())
            pending_u.append(inputs[trial, step].copy())
            if len(pending_x) == batch_size:
                transaction_started = perf_counter()
                bx, by, bu = np.asarray(pending_x), np.asarray(pending_y), np.asarray(pending_u)
                bz = np.stack([model.lift(value) for value in bx])
                bn = np.stack([model.lift(value) for value in by])
                bun = model.u_normer.transform(bu)
                bxn = model.x_normer.transform(bx)
                state_before_update = copy.deepcopy(state)
                network_before_update = copy.deepcopy(model.model.state_dict())
                record = state.update(bz, bn, bun, bxn)
                record["pre_theta_update_diagnostics"] = record.get("diagnostics")
                record.update({"trial_id": trial, "time_index": step,
                               "pending_sample_count": 0, "mode": mode})
                update_history.append(record)
                if record["accepted"] and mode == "full":
                    training = train_online_encoder(
                        model, state, bx, by, bu, loss_weight=loss_weight,
                        epochs=online_epochs, learning_rate=online_lr,
                        weight_decay=online_weight_decay, grad_clip=grad_clip,
                    )
                    training.update({"trial_id": trial, "time_index": step,
                                     "update_index": state.update_index})
                    training_history.append(training)
                    if not training["accepted"]:
                        state = state_before_update
                        model.model.load_state_dict(network_before_update)
                        record["accepted"] = False
                        record["reason"] = f"rolled_back_encoder:{training['reason']}"
                        record["model_version"] = state.model_version
                    else:
                        candidate_x = [*coordinate_x, *(value.copy() for value in bx)]
                        candidate_y = [*coordinate_y, *(value.copy() for value in by)]
                        candidate_u = [*coordinate_u, *(value.copy() for value in bu)]
                        try:
                            state, refit_timing = _consistent_state_from_physical_history(
                                model, np.asarray(candidate_x), np.asarray(candidate_y),
                                np.asarray(candidate_u), state
                            )
                            if not state.diagnostics()["finite"]:
                                raise FloatingPointError("post-refit state is non-finite")
                        except Exception as error:
                            state = state_before_update
                            model.model.load_state_dict(network_before_update)
                            record["accepted"] = False
                            record["reason"] = f"rolled_back_coordinate_refit:{type(error).__name__}:{error}"
                            record["model_version"] = state.model_version
                            training["accepted"] = False
                            training["reason"] = record["reason"]
                        else:
                            coordinate_x, coordinate_y, coordinate_u = candidate_x, candidate_y, candidate_u
                            record.update(refit_timing)
                            record["coordinate_refit_sample_count"] = state.sample_count
                            record["coordinate_consistent"] = True
                            record["post_coordinate_refit_diagnostics"] = state.diagnostics()
                elif record["accepted"]:
                    coordinate_x.extend(value.copy() for value in bx)
                    coordinate_y.extend(value.copy() for value in by)
                    coordinate_u.extend(value.copy() for value in bu)
                    record["post_coordinate_refit_diagnostics"] = state.diagnostics()
                record["total_transaction_time_s"] = perf_counter() - transaction_started
                pending_x, pending_y, pending_u = [], [], []
        if pending_x:
            update_history.append({"accepted": False, "reason": "pending_incomplete_batch",
                                   "trial_id": trial, "time_index": steps - 1,
                                   "pending_sample_count": len(pending_x), "mode": mode,
                                   "model_version": state.model_version,
                                   "update_index": state.update_index + 1})
        final_state = state
    accepted = [v for v in update_history if v["accepted"]]
    training_times = [v["training_time_s"] for v in training_history]
    transaction_times = [v.get("total_transaction_time_s", 0.0) for v in accepted]
    relift_times = [v.get("coordinate_relift_time_s", 0.0) for v in accepted]
    refit_times = [v.get("coordinate_refit_time_s", 0.0) for v in accepted]
    evidence = {"summary": {"method": "dktv", "mode": mode, "trial_strategy": "independent_reset",
                "trial_count": trials, "batch_size": batch_size, "update_count": len(accepted),
                "rejected_count": len(update_history) - len(accepted),
                "encoder_update_count": sum(v["checksum_before"] != v["checksum_after"] for v in training_history),
                "mean_training_time_s": float(np.mean(training_times)) if training_times else 0.0,
                "mean_transaction_time_s": float(np.mean(transaction_times)) if transaction_times else 0.0,
                "maximum_transaction_time_s": float(np.max(transaction_times)) if transaction_times else 0.0,
                "mean_coordinate_relift_time_s": float(np.mean(relift_times)) if relift_times else 0.0,
                "mean_coordinate_refit_time_s": float(np.mean(refit_times)) if refit_times else 0.0,
                "historical_coordinate_approximation": False,
                "full_coordinate_refit": mode == "full"},
                "history": update_history, "training": training_history}
    arrays = {"states_true": states, "inputs": inputs, "fixed_dkuc_one_step": fixed_prediction,
              "dktv_one_step": prediction, "A_by_step": A_steps, "B_by_step": B_steps,
              "C_by_step": C_steps, "model_version_by_step": versions,
              "lifted_state_by_step": lifted_steps,
              "encoder_version_by_step": encoder_versions,
              "trial_id": np.repeat(np.arange(trials), steps),
              "time_index": np.tile(np.arange(steps), trials)}
    if "disturbance_torque" in stream_data:
        arrays["disturbance_torque"] = np.asarray(stream_data["disturbance_torque"])
    return final_state, evidence, arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full Hao-DKTV or its frozen-encoder accumulative ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True, help="Frozen DKUC artifact directory.")
    parser.add_argument(
        "--history_dataset",
        default="",
        help="Initial history dataset; defaults to <artifact_dir>/dataset_train.npz.",
    )
    parser.add_argument("--stream_dataset", required=True, help="Chronological online stream NPZ.")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    parser.add_argument("--mode", choices=["full", "frozen_encoder"], default="full")
    parser.add_argument("--loss_weight", type=float, default=0.5)
    parser.add_argument("--online_epochs", type=int, default=20)
    parser.add_argument("--online_lr", type=float, default=1e-4)
    parser.add_argument("--online_weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--resume_state", default="")
    parser.add_argument("--resume_model", default="")
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
    if bool(args.resume_state) != bool(args.resume_model):
        raise ValueError("resume_state and resume_model must be provided together")
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
    fixed_model = DKUCModel(artifact_dir, args.device)
    history_data = load_dataset(history_path)
    stream_data = load_dataset(stream_path)
    paths = create_prediction_run_paths(
        "dktv", args.run_type, args.tag, args.out_dir or None
    )
    if args.resume_model:
        if args.mode == "full":
            raise ValueError(
                "full resume is disabled until checkpoints preserve consumed physical transitions"
            )
        model.model.load_state_dict(model._torch.load(args.resume_model, map_location=model.device, weights_only=True))
    resume_state = HaoDKTVState.load(args.resume_state) if args.resume_state else None
    updater, update_evidence, arrays = run_hao_dktv_replay(
        model,
        history_data,
        stream_data,
        mode=args.mode,
        batch_size=args.batch_size,
        ridge_lambda=args.ridge_lambda,
        loss_weight=args.loss_weight,
        online_epochs=args.online_epochs,
        online_lr=args.online_lr,
        online_weight_decay=args.online_weight_decay,
        grad_clip=args.grad_clip,
        resume_state=resume_state,
    )

    steps = int(stream_data["inputs"].shape[1])
    horizons = _effective_horizons(args.rollout_horizons, steps)
    max_horizon = max(horizons)
    matched_starts = matched_rollout_start_indices(steps, max_horizon, args.rollout_stride)
    if "t" in stream_data and np.asarray(stream_data["t"]).size > 1:
        rollout_dt = float(np.median(np.diff(np.asarray(stream_data["t"], dtype=np.float64))))
    else:
        rollout_dt = 1.0
    metrics: dict[str, Any] = {
        "one_step": {
            "fixed_dkuc": evaluate_predictions(arrays["states_true"], arrays["fixed_dkuc_one_step"]),
            "dktv": evaluate_predictions(arrays["states_true"], arrays["dktv_one_step"]),
        },
        "fixed_horizon_rollout": {"fixed_dkuc": {}, "dktv": {}},
        "batch_rollout": {"dktv": {}},
        "matched_origin_protocol": {
            "maximum_horizon_steps": max_horizon,
            "maximum_horizon_seconds": max_horizon * rollout_dt,
            "burn_in_steps": int(matched_starts[0]),
            "stride_steps": args.rollout_stride,
            "start_count_per_trial": int(matched_starts.size),
            "trial_count": int(arrays["states_true"].shape[0]),
            "future_input_source": "recorded_control_sequence",
            "model_snapshot_frozen_within_window": True,
            "future_observations_consumed": False,
        },
    }
    fixed_A = np.broadcast_to(fixed_model.A, (steps, *fixed_model.A.shape)).copy()
    fixed_B = np.broadcast_to(fixed_model.B, (steps, *fixed_model.B.shape)).copy()
    max_truth, max_fixed_prediction = snapshot_rollout_predictions(
        fixed_model,
        stream_data,
        fixed_A,
        fixed_B,
        horizon=max_horizon,
        stride=args.rollout_stride,
        start_indices=matched_starts,
    )
    max_dktv_truth, max_dktv_prediction, max_dktv_latent_norm = dktv_snapshot_rollout_evidence(
        model,
        arrays["states_true"],
        arrays["inputs"],
        arrays["lifted_state_by_step"],
        arrays["A_by_step"],
        arrays["B_by_step"],
        arrays["C_by_step"],
        horizon=max_horizon,
        stride=args.rollout_stride,
        start_indices=matched_starts,
    )
    if not np.array_equal(max_truth, max_dktv_truth):
        raise RuntimeError("fixed and DKTV matched-origin truth windows are misaligned")
    for horizon in horizons:
        truth = max_truth[:, :horizon + 1]
        fixed_prediction = max_fixed_prediction[:, :horizon + 1]
        dktv_prediction = max_dktv_prediction[:, :horizon + 1]
        dktv_latent_norm = max_dktv_latent_norm[:, :horizon + 1]
        fixed_metrics, _ = long_horizon_metrics(
            truth, fixed_prediction, dt=rollout_dt
        )
        dktv_metrics, _ = long_horizon_metrics(
            truth, dktv_prediction, dt=rollout_dt, latent_norm=dktv_latent_norm
        )
        metrics["fixed_horizon_rollout"]["fixed_dkuc"][str(horizon)] = fixed_metrics
        metrics["fixed_horizon_rollout"]["dktv"][str(horizon)] = dktv_metrics
        arrays[f"fixed_dkuc_rollout_h{horizon}_true"] = truth
        arrays[f"fixed_dkuc_rollout_h{horizon}_prediction"] = fixed_prediction
        arrays[f"dktv_rollout_h{horizon}_true"] = truth
        arrays[f"dktv_rollout_h{horizon}_prediction"] = dktv_prediction

    _, fixed_long_diagnostics = long_horizon_metrics(
        max_truth, max_fixed_prediction, dt=rollout_dt
    )
    _, dktv_long_diagnostics = long_horizon_metrics(
        max_truth,
        max_dktv_prediction,
        dt=rollout_dt,
        latent_norm=max_dktv_latent_norm,
    )
    arrays["matched_rollout_start_indices"] = matched_starts
    arrays["matched_rollout_trial_id"] = np.repeat(
        np.arange(arrays["states_true"].shape[0], dtype=np.int64), matched_starts.size
    )
    arrays["matched_rollout_time_index"] = np.tile(
        matched_starts, arrays["states_true"].shape[0]
    )
    arrays["matched_rollout_lead_time_s"] = np.arange(max_horizon + 1) * rollout_dt
    arrays["fixed_dkuc_long_rmse_by_lead_step"] = fixed_long_diagnostics["rmse_by_lead_step"]
    arrays["fixed_dkuc_long_rmse_by_state_and_lead_step"] = fixed_long_diagnostics[
        "rmse_by_state_and_lead_step"
    ]
    arrays["fixed_dkuc_long_window_rmse"] = fixed_long_diagnostics["window_rmse"]
    arrays["dktv_long_rmse_by_lead_step"] = dktv_long_diagnostics["rmse_by_lead_step"]
    arrays["dktv_long_rmse_by_state_and_lead_step"] = dktv_long_diagnostics[
        "rmse_by_state_and_lead_step"
    ]
    arrays["dktv_long_window_rmse"] = dktv_long_diagnostics["window_rmse"]
    arrays["dktv_long_state_error_norm"] = dktv_long_diagnostics[
        "state_error_norm_by_window_and_lead"
    ]
    arrays["dktv_long_latent_norm"] = dktv_long_diagnostics[
        "latent_norm_by_window_and_lead"
    ]
    arrays["dktv_long_model_version_at_start"] = np.concatenate([
        arrays["model_version_by_step"][trial, matched_starts]
        for trial in range(arrays["states_true"].shape[0])
    ])
    arrays["dktv_long_encoder_version_at_start"] = np.concatenate([
        arrays["encoder_version_by_step"][trial, matched_starts]
        for trial in range(arrays["states_true"].shape[0])
    ])
    arrays["dktv_long_A_spectral_radius_at_start"] = np.asarray([
        np.max(np.abs(np.linalg.eigvals(arrays["A_by_step"][trial, start])))
        for trial in range(arrays["states_true"].shape[0])
        for start in matched_starts
    ])

    batch_horizon = min(args.batch_size, steps)
    batch_truth, batch_prediction = dktv_snapshot_rollouts(
        model, arrays["states_true"], arrays["inputs"], arrays["lifted_state_by_step"],
        arrays["A_by_step"], arrays["B_by_step"], arrays["C_by_step"],
        horizon=batch_horizon, stride=batch_horizon,
    )
    metrics["batch_rollout"]["dktv"][str(batch_horizon)] = evaluate_predictions(
        batch_truth, batch_prediction
    )
    arrays["dktv_batch_rollout_true"] = batch_truth
    arrays["dktv_batch_rollout_prediction"] = batch_prediction

    arrays_path = paths.artifact_dir / "prediction_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    updater.save(paths.artifact_dir / "final_dktv_state.npz")
    model._torch.save(model.model.state_dict(), paths.artifact_dir / "final_dktv_model.pt")
    save_json(paths.artifact_dir / "metrics.json", metrics)
    save_json(paths.artifact_dir / "update_history.json", update_evidence["history"])
    save_json(paths.artifact_dir / "training_history.json", update_evidence["training"])
    figure_paths = plot_prediction_states(
        arrays["states_true"], arrays["dktv_one_step"], paths.figures_dir, "dktv_one_step"
    )
    figure_paths += plot_prediction_errors(
        arrays["states_true"], arrays["dktv_one_step"], paths.figures_dir, "dktv_one_step"
    )
    figure_paths += plot_long_horizon_evidence(
        max_truth,
        max_fixed_prediction,
        max_dktv_prediction,
        max_dktv_latent_norm,
        matched_starts,
        dt=rollout_dt,
        figures_dir=paths.figures_dir,
    )
    manifest = {
        "method": "dktv",
        "reference": "Hao et al. accumulative online Koopman update",
        "result_status": "exploratory",
        "run_type": args.run_type,
        "run_id": paths.run_id,
        "entry_script": "prediction/dktv_prediction.py",
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
            "mode": args.mode,
            "batch_size": args.batch_size,
            "ridge_lambda": args.ridge_lambda,
            "loss_weight": args.loss_weight,
            "online_epochs": args.online_epochs,
            "online_lr": args.online_lr,
            "online_weight_decay": args.online_weight_decay,
            "grad_clip": args.grad_clip,
            "rollout_horizons": horizons,
            "rollout_stride": args.rollout_stride,
            "matched_rollout_burn_in_steps": int(matched_starts[0]),
            "matched_rollout_start_indices": matched_starts,
            "rollout_dt": rollout_dt,
            "encoder_frozen": args.mode == "frozen_encoder",
            "trial_strategy": "independent_reset",
            "resume_state": args.resume_state,
            "resume_model": args.resume_model,
        },
        "update_summary": update_evidence["summary"],
        "metrics": metrics,
        "artifacts": {
            "arrays": str(arrays_path),
            "final_state": str(paths.artifact_dir / "final_dktv_state.npz"),
            "final_model": str(paths.artifact_dir / "final_dktv_model.pt"),
            "figures": figure_paths,
        },
    }
    save_json(paths.artifact_dir / "manifest.json", manifest)
    print(f"[done] DKTV artifacts -> {paths.artifact_dir}")


if __name__ == "__main__":
    main()
