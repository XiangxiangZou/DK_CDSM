"""DKAC lifted Koopman MPC with cable-tension constraints."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from io_utils import DEFAULT_XML, make_output_dir, manifest, save_json
from cable_interface import (
    apply_joint_torque,
    joint_torque_bounds_from_cable_limits,
)
from model_artifacts import load_prediction_control_model, resolve_model_selection
from references import build_cartesian_circle_reference, ik_config_from_args

from cdsm.kinematics.ik import MujocoSiteIK
from cdsm.plants.mujoco import MujocoCablePlant
from koopman_control.evaluation.tracking import (
    cartesian_tracking_metrics,
    tracking_metrics,
)


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 40
    Qq: float = 60.0
    Qdq: float = 3.0
    R: float = 1e-3
    Rd: float = 2e-2


def _require_osqp():
    try:
        import osqp
    except ModuleNotFoundError:
        raise ModuleNotFoundError("osqp is required for MPC control") from None
    return osqp


class KoopmanMPC:
    """Constrained finite-horizon MPC over DKAC lifted dynamics."""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cfg: MPCConfig) -> None:
        self.cfg = cfg
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self._precompute()
        self.last_status = "not_solved"
        self.last_iterations = 0
        self._warm_start: np.ndarray | None = None

    def solve(
        self,
        z0: np.ndarray,
        ref_norm: np.ndarray,
        u_prev: np.ndarray,
        *,
        physical_from_internal: np.ndarray,
        physical_lower_norm: np.ndarray,
        physical_upper_norm: np.ndarray,
    ) -> np.ndarray:
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        ref = np.asarray(ref_norm, dtype=np.float64).reshape(-1)
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(self.control_dim)
        grad = self.gamma.T @ self.qbar @ (self.phi @ z0 - ref)
        grad = grad - self.dmat.T @ self.rdbar @ self.emat @ u_prev
        mapping = np.asarray(physical_from_internal, dtype=np.float64)
        physical_dim = mapping.shape[0]
        constraint_matrix = sparse.kron(
            sparse.eye(self.horizon, format="csc"),
            sparse.csc_matrix(mapping),
            format="csc",
        )
        problem = _require_osqp().OSQP()
        problem.setup(
            P=sparse.csc_matrix(self.hessian),
            q=np.asarray(grad, dtype=np.float64),
            A=constraint_matrix,
            l=np.tile(np.asarray(physical_lower_norm).reshape(physical_dim), self.horizon),
            u=np.tile(np.asarray(physical_upper_norm).reshape(physical_dim), self.horizon),
            verbose=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=4000,
            polish=True,
        )
        if self._warm_start is not None:
            problem.warm_start(x=self._warm_start)
        result = problem.solve()
        self.last_status = str(result.info.status)
        self.last_iterations = int(result.info.iter)
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP failed: {self.last_status}")
        solution = np.asarray(result.x, dtype=np.float64).reshape(
            self.horizon,
            self.control_dim,
        )
        self._warm_start = np.vstack([solution[1:], solution[-1:]]).reshape(-1)
        return solution

    def _precompute(self) -> None:
        horizon = int(self.cfg.horizon)
        A, B, C = self.A, self.B, self.C
        nz, nu, ny = A.shape[0], B.shape[1], C.shape[0]
        phi = np.zeros((horizon * ny, nz), dtype=np.float64)
        gamma = np.zeros((horizon * ny, horizon * nu), dtype=np.float64)
        powers = [np.eye(nz)]
        for _ in range(horizon):
            powers.append(A @ powers[-1])
        for i in range(horizon):
            phi[i * ny : (i + 1) * ny] = C @ powers[i + 1]
            for j in range(i + 1):
                gamma[i * ny : (i + 1) * ny, j * nu : (j + 1) * nu] = C @ powers[i - j] @ B
        qbar = np.diag(np.tile([self.cfg.Qq, self.cfg.Qq, self.cfg.Qdq, self.cfg.Qdq], horizon))
        rbar = np.eye(horizon * nu) * self.cfg.R
        rdbar = np.eye(horizon * nu) * self.cfg.Rd
        dmat = np.zeros((horizon * nu, horizon * nu), dtype=np.float64)
        for k in range(horizon):
            dmat[k * nu : (k + 1) * nu, k * nu : (k + 1) * nu] = np.eye(nu)
            if k > 0:
                dmat[k * nu : (k + 1) * nu, (k - 1) * nu : k * nu] = -np.eye(nu)
        emat = np.zeros((horizon * nu, nu), dtype=np.float64)
        emat[:nu, :] = np.eye(nu)
        self.phi = phi
        self.gamma = gamma
        self.qbar = qbar
        self.rdbar = rdbar
        self.dmat = dmat
        self.emat = emat
        self.hessian = gamma.T @ qbar @ gamma + rbar + dmat.T @ rdbar @ dmat
        self.hessian = self.hessian + 1e-9 * np.eye(self.hessian.shape[0])
        self.horizon = horizon
        self.control_dim = nu


def future_reference(model, states_ref: np.ndarray, k: int, horizon: int) -> np.ndarray:
    refs = np.asarray(states_ref, dtype=np.float64)
    out = np.zeros((horizon, refs.shape[1]), dtype=np.float64)
    for i in range(horizon):
        idx = min(k + 1 + i, refs.shape[0] - 1)
        out[i] = model.x_normer.transform(refs[idx : idx + 1])[0]
    return out


def _state_reference(reference: dict[str, np.ndarray]) -> np.ndarray:
    if "x_ref" in reference:
        return np.asarray(reference["x_ref"], dtype=np.float64)
    return np.hstack([reference["q_ref"], reference["dq_ref"]])


def run_dkac_mpc(
    *,
    model,
    reference: dict[str, np.ndarray],
    cfg: MPCConfig,
    xml_path: str | Path,
    dt: float,
    f_preload: float,
    f_max_cable: float,
) -> dict[str, np.ndarray]:
    if getattr(model, "name", "") != "dkac":
        raise ValueError("MPC control currently expects a DKAC model")
    plant = MujocoCablePlant(xml_path, dt)
    t = np.asarray(reference["t"], dtype=np.float64)
    x_ref = _state_reference(reference)
    plant.set_state(x_ref[0, :2], x_ref[0, 2:])
    controller = KoopmanMPC(model.A, model.B, model.C, cfg)
    previous_internal = np.zeros(model.B.shape[1], dtype=np.float64)
    records: dict[str, list] = {
        "t": [],
        "x_meas": [],
        "x_ref": [],
        "control_cmd": [],
        "internal_control": [],
        "normalized_control": [],
        "cable_tensions": [],
        "allocation_residual": [],
        "torque_lower": [],
        "torque_upper": [],
        "solve_ms": [],
        "mpc_status": [],
        "mpc_iterations": [],
    }
    for k in range(len(t) - 1):
        measured = plant.read_state()
        torque_lower, torque_upper = joint_torque_bounds_from_cable_limits(
            plant,
            f_preload=f_preload,
            f_max_cable=f_max_cable,
        )
        lower_norm = (torque_lower - model.u_normer.mean) / model.u_normer.std
        upper_norm = (torque_upper - model.u_normer.mean) / model.u_normer.std
        physical_from_internal = np.linalg.pinv(model.control_matrix(measured), rcond=1e-6)
        started = time.perf_counter()
        internal_sequence = controller.solve(
            model.lift(measured),
            future_reference(model, x_ref, k, cfg.horizon),
            previous_internal,
            physical_from_internal=physical_from_internal,
            physical_lower_norm=lower_norm,
            physical_upper_norm=upper_norm,
        )
        solve_ms = 1e3 * (time.perf_counter() - started)
        internal_command = internal_sequence[0]
        normalized_control = physical_from_internal @ internal_command
        tau_cmd = model.u_normer.inverse(normalized_control.reshape(1, -1))[0]
        tau_cmd = np.clip(tau_cmd, torque_lower, torque_upper)
        normalized_control = model.u_normer.transform(tau_cmd.reshape(1, -1))[0]
        executed_internal = model.control_matrix(measured) @ normalized_control
        tensions, residual = apply_joint_torque(
            plant,
            tau_cmd,
            f_preload=f_preload,
            f_max_cable=f_max_cable,
        )
        plant.step()
        records["t"].append(t[k])
        records["x_meas"].append(np.asarray(measured).copy())
        records["x_ref"].append(x_ref[k].copy())
        records["control_cmd"].append(tau_cmd.copy())
        records["internal_control"].append(executed_internal.copy())
        records["normalized_control"].append(normalized_control.copy())
        records["cable_tensions"].append(tensions.copy())
        records["allocation_residual"].append(residual.copy())
        records["torque_lower"].append(torque_lower.copy())
        records["torque_upper"].append(torque_upper.copy())
        records["solve_ms"].append(solve_ms)
        records["mpc_status"].append(controller.last_status)
        records["mpc_iterations"].append(controller.last_iterations)
        previous_internal = executed_internal
    log = {key: np.asarray(value) for key, value in records.items()}
    log["q_ref"] = log["x_ref"][:, :2]
    log["dq_ref"] = log["x_ref"][:, 2:]
    log["tau_cmd"] = log["control_cmd"]
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DKAC Koopman MPC with cable-tension constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", default="")
    parser.add_argument("--model_config", default=str(Path(__file__).resolve().parent / "model_selections.json"))
    parser.add_argument("--model_key", default="")
    parser.add_argument("--run_type", choices=("smoke_test", "full_run"), default="smoke_test")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--Qq", type=float, default=60.0)
    parser.add_argument("--Qdq", type=float, default=3.0)
    parser.add_argument("--R", type=float, default=1e-3)
    parser.add_argument("--Rd", type=float, default=2e-2)
    parser.add_argument("--f_preload", type=float, default=20.0)
    parser.add_argument("--f_max_cable", type=float, default=1000.0)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--num_cycles", type=float, default=1.0)
    parser.add_argument("--center_x", type=float, default=5.0)
    parser.add_argument("--center_y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.45)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--start_hold", type=float, default=1.0)
    parser.add_argument("--time_scaling", choices=("linear", "quintic"), default="quintic")
    parser.add_argument("--ik_site", default="end_effector")
    parser.add_argument("--ik_max_iter", type=int, default=120)
    parser.add_argument("--ik_tol", type=float, default=1e-5)
    parser.add_argument("--ik_damping", type=float, default=1e-4)
    parser.add_argument("--ik_max_step", type=float, default=0.08)
    parser.add_argument("--ik_joint_margin", type=float, default=0.05)
    parser.add_argument("--ik_smooth_window_s", type=float, default=0.03)
    parser.add_argument("--ik_seed_a", type=float, default=0.1)
    parser.add_argument("--ik_seed_b", type=float, default=-0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = MPCConfig(args.horizon, args.Qq, args.Qdq, args.R, args.Rd)
    output = make_output_dir(args.run_type, "mpc", args.tag)
    artifact_dir, model_name, model_selection = resolve_model_selection(
        controller="mpc",
        artifact_dir=args.artifact_dir,
        model_name="dkac",
        model_key=args.model_key,
        model_config=args.model_config,
    )
    ik_solver = MujocoSiteIK(args.xml, args.dt, ik_config_from_args(args))
    reference = build_cartesian_circle_reference(args, ik_solver)
    np.savez_compressed(output / "arrays" / "reference.npz", **reference)
    save_json(
        output / "manifest.json",
        {
            **manifest("control/mpc_control.py", sys.argv[1:]),
            "artifact_dir": artifact_dir,
            "model": model_name,
            "model_selection": model_selection,
            "config": asdict(cfg),
            "constraints": {
                "equivalent_torque_limit": None,
                "f_preload_n": args.f_preload,
                "f_max_cable_n": args.f_max_cable,
            },
        },
    )
    model = load_prediction_control_model(artifact_dir, model_name, args.device)
    log = run_dkac_mpc(
        model=model,
        reference=reference,
        cfg=cfg,
        xml_path=args.xml,
        dt=args.dt,
        f_preload=args.f_preload,
        f_max_cable=args.f_max_cable,
    )
    count = len(log["t"])
    log["ee_meas"] = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
    log["ee_ref"] = reference["ee_ref"][:count]
    np.savez_compressed(output / "arrays" / "closed_loop_dkac.npz", **log)
    values = tracking_metrics(log)
    values["cartesian"] = cartesian_tracking_metrics(
        ee_meas=log["ee_meas"],
        ee_ref=log["ee_ref"],
        ik_error=reference["ik_error"][:count],
    )
    tensions = np.asarray(log["cable_tensions"], dtype=np.float64)
    values["constraints"] = {
        "minimum_cable_tension_n": float(np.min(tensions)),
        "maximum_cable_tension_n": float(np.max(tensions)),
        "lower_violation_count": int(np.sum(tensions < args.f_preload - 1e-6)),
        "upper_violation_count": int(np.sum(tensions > args.f_max_cable + 1e-6)),
        "maximum_allocation_residual_nm": float(np.max(np.abs(log["allocation_residual"]))),
    }
    values["mpc_solver"] = {
        "status_counts": dict(Counter(str(item) for item in log["mpc_status"].tolist())),
        "mean_iterations": float(np.mean(log["mpc_iterations"])),
        "max_iterations": int(np.max(log["mpc_iterations"])),
    }
    save_json(output / "metrics" / "tracking_metrics.json", {"models": {"dkac": values}})
    print(f"[mpc] dkac: rmse_q={values['rmse_q']:.6g}")
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
