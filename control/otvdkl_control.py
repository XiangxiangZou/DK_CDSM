"""Zhang-style OTVDKL terminal design and lifted MPC.

The numerical components in this module are independently testable.  A run is
only called stability-conditioned when the solver status *and* the recomputed
matrix margins pass; successful optimization alone is not treated as proof.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from prediction.otvdkl_prediction import (
    OTVDKLModelSnapshot,
    SelectiveWindowKoopmanUpdater,
    SlidingWindowKoopmanUpdater,
)
from prediction.common import dkuc_artifact_fingerprint, lift_dkuc_transitions, load_dataset
from prediction.dkuc_prediction import DKUCModel
from control.io_utils import DEFAULT_XML, make_output_dir, manifest, save_json
from control.cable_interface import apply_joint_torque, joint_torque_bounds_from_cable_limits
from cdsm.plants.mujoco import MujocoCablePlant


@dataclass(frozen=True)
class TerminalSDPConfig:
    q_weight: float = 1.0
    r_weight: float = 1e-2
    positive_tolerance: float = 1e-7
    lmi_tolerance: float = 5e-6
    solver: str = "CLARABEL"


@dataclass(frozen=True)
class TerminalSDPResult:
    status: str
    usable: bool
    solver: str
    solve_time_s: float
    gamma: float
    P_bar: np.ndarray
    P: np.ndarray
    K: np.ndarray
    margins: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("P_bar", "P", "K"):
            values[key] = values[key].tolist()
        return values


def conservative_symmetric_input_bound(
    lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return the largest origin-centred box inside an asymmetric box."""
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("input bounds must be aligned and finite")
    if np.any(lo >= 0.0) or np.any(hi <= 0.0):
        raise ValueError("normalized input box must contain the origin strictly")
    bound = np.minimum(-lo, hi)
    if np.any(bound <= 0.0):
        raise ValueError("no non-empty symmetric input box exists")
    return bound


def _min_eig(matrix: np.ndarray) -> float:
    symmetric = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    return float(np.min(np.linalg.eigvalsh(symmetric)))


def terminal_sdp_margins(
    A: np.ndarray,
    B: np.ndarray,
    e0: np.ndarray,
    u_max: np.ndarray,
    gamma: float,
    P_bar: np.ndarray,
    P: np.ndarray,
    K: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> dict[str, float]:
    """Recompute all acceptance margins independently of CVXPY objects."""
    A, B = np.asarray(A), np.asarray(B)
    closed = A + B @ K
    lyapunov = P - closed.T @ P @ closed - Q - K.T @ R @ K
    input_lmi = np.block([[np.diag(u_max), K @ P_bar], [P_bar @ K.T, P_bar]])
    ellipsoid = float(gamma - np.asarray(e0) @ P @ np.asarray(e0))
    return {
        "min_eig_P_bar": _min_eig(P_bar),
        "min_eig_P": _min_eig(P),
        "lyapunov_min_eig": _min_eig(lyapunov),
        "input_lmi_min_eig": _min_eig(input_lmi),
        "current_ellipsoid_margin": ellipsoid,
    }


def solve_terminal_sdp(
    A: np.ndarray,
    B: np.ndarray,
    e0: np.ndarray,
    u_max: np.ndarray,
    config: TerminalSDPConfig = TerminalSDPConfig(),
) -> TerminalSDPResult:
    """Solve a convex terminal-set design in Zhang's ``P_bar/Y/gamma`` coordinates."""
    import cvxpy as cp

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    e0 = np.asarray(e0, dtype=np.float64).reshape(-1)
    u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)
    n, m = A.shape[0], B.shape[1]
    if A.shape != (n, n) or B.shape != (n, m) or e0.shape != (n,) or u_max.shape != (m,):
        raise ValueError("incompatible SDP dimensions")
    if np.any(u_max <= 0.0) or not all(np.all(np.isfinite(v)) for v in (A, B, e0, u_max)):
        raise ValueError("SDP data must be finite and u_max positive")

    X = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    gamma = cp.Variable(nonneg=True)
    AXBY = A @ X + B @ Y
    Q = np.eye(n) * float(config.q_weight)
    R = np.eye(m) * float(config.r_weight)
    Q_sqrt = np.eye(n) * np.sqrt(float(config.q_weight))
    R_sqrt = np.eye(m) * np.sqrt(float(config.r_weight))
    eye = np.eye(n)
    constraints = [X >> config.positive_tolerance * eye, gamma >= config.positive_tolerance]
    # Paper equation (23), with P_bar=X and Y=K P_bar.
    constraints += [cp.bmat([
        [X, AXBY.T, (Q_sqrt @ X).T, (R_sqrt @ Y).T],
        [AXBY, X, np.zeros((n, n)), np.zeros((n, m))],
        [Q_sqrt @ X, np.zeros((n, n)), gamma * np.eye(n), np.zeros((n, m))],
        [R_sqrt @ Y, np.zeros((m, n)), np.zeros((m, n)), gamma * np.eye(m)],
    ]) >> 0]
    constraints += [cp.bmat([[np.ones((1, 1)), e0[None, :]], [e0[:, None], X]]) >> 0]
    constraints += [cp.bmat([[np.diag(u_max), Y], [Y.T, X]]) >> 0]
    problem = cp.Problem(cp.Minimize(gamma), constraints)
    started = perf_counter()
    attempted: list[str] = []
    for solver in (config.solver, "SCS"):
        if solver in attempted:
            continue
        attempted.append(solver)
        try:
            kwargs = {"verbose": False, "warm_start": True}
            if solver == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 20000})
            problem.solve(solver=solver, **kwargs)
        except cp.error.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
    elapsed = perf_counter() - started
    used_solver = str(problem.solver_stats.solver_name) if problem.solver_stats else "none"
    empty = np.empty((0, 0), dtype=np.float64)
    if X.value is None or Y.value is None or gamma.value is None:
        return TerminalSDPResult(str(problem.status), False, used_solver, elapsed, np.nan, empty, empty, empty, {}, "solver_no_solution")
    P_bar = 0.5 * (np.asarray(X.value) + np.asarray(X.value).T)
    gamma_value = float(gamma.value)
    try:
        inverse = np.linalg.inv(P_bar)
    except np.linalg.LinAlgError:
        return TerminalSDPResult(str(problem.status), False, used_solver, elapsed, gamma_value, P_bar, empty, empty, {}, "singular_P_bar")
    K = np.asarray(Y.value) @ inverse
    P = gamma_value * inverse
    margins = terminal_sdp_margins(A, B, e0, u_max, gamma_value, P_bar, P, K, Q, R)
    required = ("min_eig_P_bar", "min_eig_P", "lyapunov_min_eig", "input_lmi_min_eig", "current_ellipsoid_margin")
    usable = all(np.isfinite(margins[key]) and margins[key] >= -config.lmi_tolerance for key in required)
    return TerminalSDPResult(str(problem.status), usable, used_solver, elapsed, gamma_value, P_bar, P, K, margins, "accepted" if usable else "residual_check_failed")


@dataclass(frozen=True)
class LiftedMPCConfig:
    horizon: int = 10
    state_weight: float = 1.0
    feature_weight: float = 1e-4
    input_weight: float = 1e-2
    solver_tolerance: float = 1e-7


class LiftedMPC:
    """Finite-horizon error-coordinate MPC for normalized physical torque."""

    def __init__(self, config: LiftedMPCConfig = LiftedMPCConfig()) -> None:
        self.config = config
        self.last_diagnostics: dict[str, Any] = {}

    def solve(
        self,
        snapshot: OTVDKLModelSnapshot,
        z0: np.ndarray,
        z_reference: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        terminal_P: np.ndarray,
        u_reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import osqp

        A, B = snapshot.A, snapshot.B
        n, m, horizon = snapshot.latent_dim, snapshot.input_dim, self.config.horizon
        refs = np.asarray(z_reference, dtype=np.float64)
        if refs.shape != (horizon + 1, n):
            raise ValueError("z_reference must have shape (horizon + 1, latent_dim)")
        u_ref = np.zeros((horizon, m)) if u_reference is None else np.asarray(u_reference, dtype=np.float64)
        lower, upper = np.asarray(lower), np.asarray(upper)
        if lower.shape != (m,) or upper.shape != (m,) or np.any(lower > upper):
            raise ValueError("invalid asymmetric MPC bounds")
        Q = np.eye(n) * self.config.feature_weight
        Q[: snapshot.state_dim, : snapshot.state_dim] = np.eye(snapshot.state_dim) * self.config.state_weight
        R = np.eye(m) * self.config.input_weight
        # Condense e_{k+1}=A e_k+B v_k.
        Phi = np.zeros((horizon * n, n))
        Gamma = np.zeros((horizon * n, horizon * m))
        powers = [np.eye(n)]
        for _ in range(horizon):
            powers.append(A @ powers[-1])
        for i in range(horizon):
            Phi[i*n:(i+1)*n] = powers[i+1]
            for j in range(i + 1):
                Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = powers[i-j] @ B
        q_blocks = [Q] * max(0, horizon - 1) + [np.asarray(terminal_P)]
        Qbar = sparse.block_diag(q_blocks, format="csc").toarray()
        Rbar = np.kron(np.eye(horizon), R)
        e0 = np.asarray(z0) - refs[0]
        # Reference dynamics mismatch is retained as an affine offset.
        drift = np.concatenate([A @ refs[i] - refs[i + 1] + B @ u_ref[i] for i in range(horizon)])
        cumulative = np.zeros(horizon * n)
        for i in range(horizon):
            cumulative[i*n:(i+1)*n] = drift[i*n:(i+1)*n]
            if i:
                cumulative[i*n:(i+1)*n] += A @ cumulative[(i-1)*n:i*n]
        base = Phi @ e0 + cumulative
        H = Gamma.T @ Qbar @ Gamma + Rbar
        g = Gamma.T @ Qbar @ base
        problem = osqp.OSQP()
        problem.setup(P=sparse.csc_matrix(2.0 * H), q=2.0 * g,
                      A=sparse.eye(horizon * m, format="csc"),
                      l=(np.broadcast_to(lower, (horizon, m)) - u_ref).reshape(-1),
                      u=(np.broadcast_to(upper, (horizon, m)) - u_ref).reshape(-1), verbose=False,
                      eps_abs=self.config.solver_tolerance, eps_rel=self.config.solver_tolerance,
                      polish=True)
        started = perf_counter()
        result = problem.solve()
        elapsed = perf_counter() - started
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP failed: {result.info.status}")
        v = np.asarray(result.x).reshape(horizon, m)
        u = v + u_ref
        predicted = (base + Gamma @ v.reshape(-1)).reshape(horizon, n) + refs[1:]
        self.last_diagnostics = {"status": str(result.info.status), "iterations": int(result.info.iter),
                                 "solve_time_s": elapsed, "objective": float(result.info.obj_val),
                                 "model_version": snapshot.model_version}
        return u, predicted


def safe_control_fallback(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Deterministic safe command: clipped zero in normalized coordinates."""
    return np.clip(np.zeros_like(np.asarray(lower, dtype=np.float64)), lower, upper)


def _initial_updater(
    model: DKUCModel,
    history_path: str | Path,
    *,
    window_size: int,
    batch_size: int,
    epsilon: float,
    ridge_lambda: float,
) -> SelectiveWindowKoopmanUpdater:
    history = load_dataset(history_path)
    z, z_next, u = lift_dkuc_transitions(model, history)
    latent = z.shape[-1]
    z = z.reshape(-1, latent)
    z_next = z_next.reshape(-1, latent)
    u = u.reshape(-1, model.control_dim)
    if z.shape[0] < window_size:
        raise ValueError("history does not contain enough causal transitions")
    base = SlidingWindowKoopmanUpdater(
        z[-window_size:], z_next[-window_size:], u[-window_size:],
        A0=model.A, B0=model.B, state_dim=model.state_dim,
        batch_size=batch_size, ridge_lambda=ridge_lambda,
        encoder_fingerprint=dkuc_artifact_fingerprint(model.artifact_dir),
        sample_ids=np.arange(-window_size, 0, dtype=np.int64),
        affine_constant=bool(model.config.include_constant),
    )
    return SelectiveWindowKoopmanUpdater(base, epsilon=epsilon)


def run_otvdkl_mujoco(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Run the shortest safety-gated CDSM integration of Algorithm 2."""
    artifact = Path(args.artifact_dir)
    history = Path(args.history_dataset) if args.history_dataset else artifact / "dataset_train.npz"
    model = DKUCModel(artifact, args.device)
    updater = _initial_updater(
        model, history, window_size=args.window_size, batch_size=args.batch_size,
        epsilon=args.epsilon, ridge_lambda=args.ridge_lambda,
    )
    transaction = OnlineOTVDKLControlTransaction(
        updater,
        mpc=LiftedMPC(LiftedMPCConfig(horizon=args.horizon)),
        sdp_config=TerminalSDPConfig(
            q_weight=args.sdp_q, r_weight=args.sdp_r,
            lmi_tolerance=args.lmi_tolerance,
        ),
        deadline_s=args.dt,
        maximum_online_sdp_dimension=args.max_online_sdp_dimension,
        fallback_control=model.u_normer.transform(np.zeros((1, model.control_dim)))[0],
    )
    plant = MujocoCablePlant(args.xml_path, args.dt)
    plant.set_state(np.zeros(2), np.zeros(2))
    steps = int(round(args.duration / args.dt))
    t_ref = np.arange(steps + args.horizon + 1) * args.dt
    q_ref_all = args.reference_amplitude * np.column_stack(
        [np.sin(2.0 * np.pi * args.reference_frequency * t_ref),
         np.sin(2.0 * np.pi * args.reference_frequency * t_ref + np.pi / 2.0)]
    )
    dq_ref_all = args.reference_amplitude * 2.0 * np.pi * args.reference_frequency * np.column_stack(
        [np.cos(2.0 * np.pi * args.reference_frequency * t_ref),
         np.cos(2.0 * np.pi * args.reference_frequency * t_ref + np.pi / 2.0)]
    )
    x_ref_all = np.hstack([q_ref_all, dq_ref_all])
    records: dict[str, list[Any]] = {key: [] for key in (
        "t", "x_meas", "x_ref", "tau_cmd", "u_normalized", "tensions",
        "allocation_residual", "torque_lower", "torque_upper", "model_version",
        "window_version", "update_status", "sdp_status", "mpc_status", "degraded",
        "degradation_reason", "total_time_s", "deadline_miss", "koopman_residual",
    )}
    previous: tuple[np.ndarray, np.ndarray] | None = None
    for k in range(steps):
        measured = plant.read_state()
        z_current = model.lift(measured)
        completed = None if previous is None else (previous[0], previous[1], z_current)
        refs = np.stack([model.lift(x_ref_all[min(k + j, len(x_ref_all) - 1)])
                         for j in range(args.horizon + 1)])
        torque_lower, torque_upper = joint_torque_bounds_from_cable_limits(
            plant, f_preload=args.f_preload, f_max_cable=args.f_max_cable,
        )
        lower_norm = model.u_normer.transform(torque_lower.reshape(1, -1))[0]
        upper_norm = model.u_normer.transform(torque_upper.reshape(1, -1))[0]
        result = transaction.step(z_current, refs, lower_norm, upper_norm,
                                  completed_transition=completed)
        u_normalized = np.clip(result.control, lower_norm, upper_norm)
        tau = model.u_normer.inverse(u_normalized.reshape(1, -1))[0]
        tau = np.clip(tau, torque_lower, torque_upper)
        u_normalized = model.u_normer.transform(tau.reshape(1, -1))[0]
        tensions, allocation = apply_joint_torque(
            plant, tau, f_preload=args.f_preload, f_max_cable=args.f_max_cable,
        )
        plant.step()
        z_after = model.lift(plant.read_state())
        residual = z_after - result.snapshot.A @ z_current - result.snapshot.B @ u_normalized
        records["t"].append(k * args.dt); records["x_meas"].append(measured)
        records["x_ref"].append(x_ref_all[k]); records["tau_cmd"].append(tau)
        records["u_normalized"].append(u_normalized); records["tensions"].append(tensions)
        records["allocation_residual"].append(allocation)
        records["torque_lower"].append(torque_lower); records["torque_upper"].append(torque_upper)
        records["model_version"].append(result.snapshot.model_version)
        records["window_version"].append(result.snapshot.window_version)
        records["update_status"].append(result.update["status"] if result.update else "pending")
        records["sdp_status"].append(result.sdp.status if result.sdp else "not_solved")
        records["mpc_status"].append(result.mpc["status"])
        records["degraded"].append(result.degraded); records["degradation_reason"].append(result.degradation_reason)
        records["total_time_s"].append(result.total_time_s)
        records["deadline_miss"].append(result.mpc["deadline_miss"])
        records["koopman_residual"].append(residual)
        previous = (z_current.copy(), u_normalized.copy())
        if not all(np.all(np.isfinite(value)) for value in (measured, tau, tensions, allocation, residual)):
            raise FloatingPointError(f"non-finite closed-loop value at step {k}")
    arrays = {key: np.asarray(value) for key, value in records.items()}
    output = make_output_dir(args.run_type, "otvdkl", args.tag)
    np.savez_compressed(output / "arrays" / "closed_loop_otvdkl.npz", **arrays)
    state_error = arrays["x_meas"] - arrays["x_ref"]
    metrics = {
        "joint_state_rmse": float(np.sqrt(np.mean(state_error ** 2))),
        "joint_position_rmse": float(np.sqrt(np.mean(state_error[:, :2] ** 2))),
        "degraded_steps": int(np.count_nonzero(arrays["degraded"])),
        "deadline_misses": int(np.count_nonzero(arrays["deadline_miss"])),
        "maximum_solve_time_s": float(np.max(arrays["total_time_s"])),
        "maximum_allocation_residual": float(np.max(np.abs(arrays["allocation_residual"]))),
        "minimum_tension": float(np.min(arrays["tensions"])),
        "maximum_tension": float(np.max(arrays["tensions"])),
        "maximum_koopman_residual_norm": float(np.max(np.linalg.norm(arrays["koopman_residual"], axis=1))),
        "finite_values": bool(all(np.all(np.isfinite(arrays[key])) for key in (
            "x_meas", "tau_cmd", "tensions", "allocation_residual", "koopman_residual"
        ))),
        "tension_violation_count": int(np.count_nonzero(
            (arrays["tensions"] < -1e-9) | (arrays["tensions"] > args.f_max_cable + 1e-9)
        )),
        "torque_bound_violation_count": int(np.count_nonzero(
            (arrays["tau_cmd"] < arrays["torque_lower"] - 1e-9)
            | (arrays["tau_cmd"] > arrays["torque_upper"] + 1e-9)
        )),
    }
    save_json(output / "metrics" / "tracking_metrics.json", metrics)
    update_values, update_counts = np.unique(arrays["update_status"], return_counts=True)
    update_metrics = {str(key): int(value) for key, value in zip(update_values, update_counts)}
    timing_metrics = {
        "mean_total_time_s": float(np.mean(arrays["total_time_s"])),
        "maximum_total_time_s": float(np.max(arrays["total_time_s"])),
        "deadline_misses": metrics["deadline_misses"],
    }
    stability_metrics = {
        "claim_level": "not_certified",
        "maximum_koopman_residual_norm": metrics["maximum_koopman_residual_norm"],
        "sdp_usable_steps": int(np.count_nonzero(arrays["sdp_status"] == "optimal")),
        "degraded_steps": metrics["degraded_steps"],
        "assumption_5_checked": False,
        "iss_lyapunov_checked": False,
    }
    save_json(output / "metrics" / "update_metrics.json", update_metrics)
    save_json(output / "metrics" / "timing_metrics.json", timing_metrics)
    save_json(output / "metrics" / "stability_metrics.json", stability_metrics)
    run_manifest = manifest("control/otvdkl_control.py", [sys.executable, *sys.argv])
    run_manifest.update({
        "method": "otvdkl_star_mpc", "claim_level": "paper_style_control_prototype",
        "artifact_dir": str(artifact), "history_dataset": str(history),
        "encoder_frozen": True, "readout_for_stability": "C_struct",
        "config": vars(args), "metrics": metrics,
    })
    save_json(output / "manifest.json", run_manifest)
    return output, metrics


@dataclass(frozen=True)
class ControlStepResult:
    control: np.ndarray
    predicted: np.ndarray
    snapshot: OTVDKLModelSnapshot
    update: dict[str, Any] | None
    sdp: TerminalSDPResult | None
    mpc: dict[str, Any]
    degraded: bool
    degradation_reason: str
    total_time_s: float


class OnlineOTVDKLControlTransaction:
    """Causal update -> snapshot -> SDP -> MPC transaction for Algorithm 2.

    The transition passed to :meth:`step` must end at the current measurement;
    it is committed before a snapshot is exposed to either optimizer.  No
    matrix from a rejected or failed candidate can leak into the controller.
    """

    def __init__(
        self,
        updater: SlidingWindowKoopmanUpdater | SelectiveWindowKoopmanUpdater,
        *,
        mpc: LiftedMPC,
        sdp_config: TerminalSDPConfig = TerminalSDPConfig(),
        deadline_s: float | None = None,
        maximum_online_sdp_dimension: int | None = None,
        fallback_control: np.ndarray | None = None,
    ) -> None:
        self.updater = updater
        self.mpc = mpc
        self.sdp_config = sdp_config
        self.deadline_s = deadline_s
        self.maximum_online_sdp_dimension = maximum_online_sdp_dimension
        self.fallback_control = None if fallback_control is None else np.asarray(
            fallback_control, dtype=np.float64
        ).reshape(self.updater.input_dim)
        self._z: list[np.ndarray] = []
        self._z_next: list[np.ndarray] = []
        self._u: list[np.ndarray] = []
        self._ids: list[int] = []
        self._next_sample_id = 0

    @property
    def pending_count(self) -> int:
        return len(self._ids)

    def _append_transition(self, transition: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        z, u, z_next = (np.asarray(value, dtype=np.float64).reshape(-1) for value in transition)
        if z.shape != (self.updater.latent_dim,) or z_next.shape != z.shape:
            raise ValueError("transition latent dimensions do not match updater")
        if u.shape != (self.updater.input_dim,):
            raise ValueError("transition input dimension does not match updater")
        if not all(np.all(np.isfinite(value)) for value in (z, u, z_next)):
            raise ValueError("transition must be finite")
        self._z.append(z.copy())
        self._u.append(u.copy())
        self._z_next.append(z_next.copy())
        self._ids.append(self._next_sample_id)
        self._next_sample_id += 1

    def _update_if_ready(self) -> dict[str, Any] | None:
        if self.pending_count < self.updater.batch_size:
            return None
        count = self.updater.batch_size
        record, _ = self.updater.update(
            np.stack(self._z[:count]), np.stack(self._z_next[:count]), np.stack(self._u[:count]),
            sample_ids=np.asarray(self._ids[:count], dtype=np.int64),
            encoder_fingerprint=self.updater.encoder_fingerprint,
        )
        del self._z[:count], self._z_next[:count], self._u[:count], self._ids[:count]
        return record.to_dict()

    def step(
        self,
        z_current: np.ndarray,
        z_reference: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        completed_transition: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        u_reference: np.ndarray | None = None,
    ) -> ControlStepResult:
        started = perf_counter()
        if completed_transition is not None:
            self._append_transition(completed_transition)
        update = self._update_if_ready()
        snapshot = self.updater.snapshot()
        z_current = np.asarray(z_current, dtype=np.float64).reshape(snapshot.latent_dim)
        refs = np.asarray(z_reference, dtype=np.float64)
        error = z_current - refs[0]
        try:
            if (
                self.maximum_online_sdp_dimension is not None
                and snapshot.latent_dim > self.maximum_online_sdp_dimension
            ):
                raise RuntimeError(
                    "online_sdp_dimension_limit:"
                    f"{snapshot.latent_dim}>{self.maximum_online_sdp_dimension}"
                )
            symmetric = conservative_symmetric_input_bound(lower, upper)
            sdp = solve_terminal_sdp(snapshot.A, snapshot.B, error, symmetric, self.sdp_config)
            if not sdp.usable:
                raise RuntimeError(f"SDP rejected: {sdp.reason}")
            controls, predicted = self.mpc.solve(
                snapshot, z_current, refs, lower, upper, sdp.P, u_reference
            )
            control = controls[0]
            degraded = False
            reason = ""
            mpc_diagnostics = dict(self.mpc.last_diagnostics)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error_value:
            sdp = locals().get("sdp")
            if self.fallback_control is None:
                control = safe_control_fallback(lower, upper)
            else:
                control = np.clip(self.fallback_control, lower, upper)
            predicted = np.empty((0, snapshot.latent_dim), dtype=np.float64)
            degraded = True
            reason = f"{type(error_value).__name__}:{error_value}"
            mpc_diagnostics = {"status": "safe_fallback", "model_version": snapshot.model_version}
        elapsed = perf_counter() - started
        mpc_diagnostics["deadline_miss"] = bool(
            self.deadline_s is not None and elapsed > self.deadline_s
        )
        return ControlStepResult(control, predicted, snapshot, update, sdp,
                                 mpc_diagnostics, degraded, reason, elapsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect the Zhang-style OTVDKL lifted-MPC controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=False, default="", help="Frozen DKUC artifact directory.")
    parser.add_argument("--history_dataset", default="", help="Causal initial-window dataset.")
    parser.add_argument("--xml_path", default=str(DEFAULT_XML))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run_type", choices=("smoke_test", "full_run"), default="smoke_test")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    parser.add_argument("--sdp_q", type=float, default=1e-3)
    parser.add_argument("--sdp_r", type=float, default=1e-3)
    parser.add_argument("--lmi_tolerance", type=float, default=5e-5)
    parser.add_argument(
        "--max_online_sdp_dimension", type=int, default=32,
        help="Safety guard: larger models enter explicit fallback instead of blocking.",
    )
    parser.add_argument("--f_preload", type=float, default=60.0)
    parser.add_argument("--f_max_cable", type=float, default=120.0)
    parser.add_argument("--reference_amplitude", type=float, default=0.01)
    parser.add_argument("--reference_frequency", type=float, default=0.1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true", help="Validate CLI configuration only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.duration <= 0 or args.dt <= 0 or args.horizon <= 0:
        raise ValueError("duration, dt, and horizon must be positive")
    if args.dry_run:
        print("[ok] OTVDKL-MPC configuration validated")
        return
    if not args.artifact_dir:
        raise ValueError("--artifact_dir is required for a physical run")
    output, metrics = run_otvdkl_mujoco(args)
    print(f"[done] OTVDKL-MPC artifacts -> {output}")
    print(f"[metrics] {metrics}")


if __name__ == "__main__":
    main()
