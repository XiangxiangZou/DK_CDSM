"""Hao et al. accumulative DKTV prediction and online-update entry point."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Hao et al. accumulative DKTV from a frozen DKUC artifact.",
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
        "dktv", args.run_type, args.tag, args.out_dir or None
    )
    updater, update_evidence, arrays = run_dktv_replay(
        model,
        history_data,
        stream_data,
        batch_size=args.batch_size,
        ridge_lambda=args.ridge_lambda,
        encoder_fingerprint=fingerprint,
    )

    steps = int(stream_data["inputs"].shape[1])
    horizons = _effective_horizons(args.rollout_horizons, steps)
    fixed_A = np.broadcast_to(model.A, arrays["A_by_step"].shape).copy()
    fixed_B = np.broadcast_to(model.B, arrays["B_by_step"].shape).copy()
    metrics: dict[str, Any] = {
        "one_step": {
            "fixed_dkuc": evaluate_predictions(arrays["states_true"], arrays["fixed_dkuc_one_step"]),
            "dktv": evaluate_predictions(arrays["states_true"], arrays["dktv_one_step"]),
        },
        "rollout": {"fixed_dkuc": {}, "dktv": {}},
    }
    for method, A_values, B_values in (
        ("fixed_dkuc", fixed_A, fixed_B),
        ("dktv", arrays["A_by_step"], arrays["B_by_step"]),
    ):
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
    updater.save(paths.artifact_dir / "final_dktv_state.npz")
    save_json(paths.artifact_dir / "metrics.json", metrics)
    save_json(paths.artifact_dir / "update_history.json", update_evidence["history"])
    figure_paths = plot_prediction_states(
        arrays["states_true"], arrays["dktv_one_step"], paths.figures_dir, "dktv_one_step"
    )
    figure_paths += plot_prediction_errors(
        arrays["states_true"], arrays["dktv_one_step"], paths.figures_dir, "dktv_one_step"
    )
    manifest = {
        "method": "dktv",
        "reference": "Hao et al. accumulative online Koopman update",
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
            "batch_size": args.batch_size,
            "ridge_lambda": args.ridge_lambda,
            "rollout_horizons": horizons,
            "rollout_stride": args.rollout_stride,
            "encoder_frozen": True,
        },
        "update_summary": update_evidence["summary"],
        "metrics": metrics,
        "artifacts": {
            "arrays": str(arrays_path),
            "final_state": str(paths.artifact_dir / "final_dktv_state.npz"),
            "figures": figure_paths,
        },
    }
    save_json(paths.artifact_dir / "manifest.json", manifest)
    print(f"[done] DKTV artifacts -> {paths.artifact_dir}")


if __name__ == "__main__":
    main()
