"""Finite-horizon lifted Koopman LQR control.

Run this file directly when using the LQR method.  The control core is the
``KoopmanLQR`` class; the CLI below only loads a model, builds a reference,
runs the closed loop, and saves arrays/metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from io_utils import DEFAULT_XML, make_output_dir, manifest, save_json
from cable_interface import apply_joint_torque
from model_artifacts import load_prediction_control_model, resolve_model_selection
from plotting import figures_dir_for_result, plot_closed_loop
from references import (
    build_cartesian_circle_reference,
    build_ramp_reference,
    ik_config_from_args,
)

from cdsm.kinematics.ik import MujocoSiteIK
from cdsm.plants.mujoco import MujocoCablePlant
from koopman_control.evaluation.tracking import (
    cartesian_tracking_metrics,
    tracking_metrics,
)


@dataclass(frozen=True)
class LQRConfig:
    horizon: int = 30
    Qq: float = 40.0
    Qdq: float = 2.0
    R: float = 1e-3
    Rd: float = 1e-2


class KoopmanLQR:
    """Unconstrained finite-horizon LQR on a lifted linear model."""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cfg: LQRConfig) -> None:
        self.cfg = cfg
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self._precompute()

    def solve(self, z0: np.ndarray, ref_norm: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        ref = np.asarray(ref_norm, dtype=np.float64).reshape(-1)
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(self.control_dim)
        grad = self.gamma.T @ self.qbar @ (self.phi @ z0 - ref)
        grad = grad - self.dmat.T @ self.rdbar @ self.emat @ u_prev
        return (-self.hinv @ grad).reshape(
            self.horizon,
            self.control_dim,
        )

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
        hessian = gamma.T @ qbar @ gamma + rbar + dmat.T @ rdbar @ dmat
        self.phi = phi
        self.gamma = gamma
        self.qbar = qbar
        self.rdbar = rdbar
        self.dmat = dmat
        self.emat = emat
        self.hinv = np.linalg.inv(hessian + 1e-9 * np.eye(hessian.shape[0]))
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


def run_lqr_tracking(
    *,
    model,
    reference: dict[str, np.ndarray],
    cfg: LQRConfig,
    xml_path: str | Path,
    dt: float,
    tau_limit: float,
    f_preload: float,
    f_max_cable: float | None,
) -> dict[str, np.ndarray]:
    plant = MujocoCablePlant(xml_path, dt)
    t = np.asarray(reference["t"], dtype=np.float64)
    x_ref = _state_reference(reference)
    plant.set_state(x_ref[0, :2], x_ref[0, 2:])
    controller = KoopmanLQR(model.A, model.B, model.C, cfg)
    previous_internal = np.zeros(model.B.shape[1], dtype=np.float64)
    records: dict[str, list] = {
        "t": [],
        "x_meas": [],
        "x_ref": [],
        "control_cmd": [],
        "internal_control": [],
        "cable_tensions": [],
        "allocation_residual": [],
        "solve_ms": [],
    }
    for k in range(len(t) - 1):
        measured = plant.read_state()
        z0 = model.lift(measured)
        ref_norm = future_reference(model, x_ref, k, cfg.horizon)
        started = time.perf_counter()
        internal_sequence = controller.solve(z0, ref_norm, previous_internal)
        solve_ms = 1e3 * (time.perf_counter() - started)
        internal_command = internal_sequence[0]
        tau_cmd = model.recover_control(measured, internal_command)
        tau_cmd = np.clip(tau_cmd, -tau_limit, tau_limit)
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
        records["control_cmd"].append(np.asarray(tau_cmd).copy())
        records["internal_control"].append(internal_command.copy())
        records["cable_tensions"].append(tensions.copy())
        records["allocation_residual"].append(residual.copy())
        records["solve_ms"].append(solve_ms)
        previous_internal = internal_command
    log = {key: np.asarray(value) for key, value in records.items()}
    log["q_ref"] = log["x_ref"][:, :2]
    log["dq_ref"] = log["x_ref"][:, 2:]
    log["tau_cmd"] = log["control_cmd"]
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run finite-horizon Koopman LQR control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", default="")
    parser.add_argument("--model", choices=("auto", "edmd", "dkuc", "dkac"), default="auto")
    parser.add_argument("--model_config", default=str(Path(__file__).resolve().parent / "model_selections.json"))
    parser.add_argument("--model_key", default="")
    parser.add_argument("--task", choices=("joint", "circle"), default="joint")
    parser.add_argument("--run_type", choices=("smoke_test", "full_run"), default="smoke_test")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--Qq", type=float, default=40.0)
    parser.add_argument("--Qdq", type=float, default=2.0)
    parser.add_argument("--R", type=float, default=1e-3)
    parser.add_argument("--Rd", type=float, default=1e-2)
    parser.add_argument("--tau_limit", type=float, default=120.0)
    parser.add_argument("--f_preload", type=float, default=50.0)
    parser.add_argument("--f_max_cable", type=float, default=2000.0)
    parser.add_argument("--T_track", type=float, default=4.0)
    parser.add_argument("--T_ramp", type=float, default=2.0)
    parser.add_argument("--qa0", type=float, default=0.0)
    parser.add_argument("--qa1", type=float, default=0.6)
    parser.add_argument("--qb0", type=float, default=0.0)
    parser.add_argument("--qb1", type=float, default=-0.45)
    parser.add_argument("--period", type=float, default=8.0)
    parser.add_argument("--num_cycles", type=float, default=1.0)
    parser.add_argument("--center_x", type=float, default=5.0)
    parser.add_argument("--center_y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.45)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--start_hold", type=float, default=0.0)
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


def _build_reference(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], MujocoSiteIK | None]:
    if args.task == "joint":
        return build_ramp_reference(
            dt=args.dt,
            duration=args.T_track,
            start=[args.qa0, args.qb0],
            target=[args.qa1, args.qb1],
            ramp_duration=args.T_ramp,
        ), None
    ik_solver = MujocoSiteIK(args.xml, args.dt, ik_config_from_args(args))
    return build_cartesian_circle_reference(args, ik_solver), ik_solver


def main() -> None:
    args = build_parser().parse_args()
    cfg = LQRConfig(args.horizon, args.Qq, args.Qdq, args.R, args.Rd)
    output = make_output_dir(args.run_type, "lqr", args.tag)
    artifact_dir, model_name, model_selection = resolve_model_selection(
        controller="lqr",
        artifact_dir=args.artifact_dir,
        model_name=args.model,
        model_key=args.model_key,
        model_config=args.model_config,
    )
    reference, ik_solver = _build_reference(args)
    np.savez_compressed(output / "arrays" / "reference.npz", **reference)
    save_json(
        output / "manifest.json",
        {
            **manifest("control/lqr_control.py", sys.argv[1:]),
            "artifact_dir": artifact_dir,
            "model": model_name,
            "model_selection": model_selection,
            "task": args.task,
            "config": asdict(cfg),
        },
    )

    model = load_prediction_control_model(artifact_dir, model_name, args.device)
    log = run_lqr_tracking(
        model=model,
        reference=reference,
        cfg=cfg,
        xml_path=args.xml,
        dt=args.dt,
        tau_limit=args.tau_limit,
        f_preload=args.f_preload,
        f_max_cable=None if args.f_max_cable < 0.0 else args.f_max_cable,
    )
    if ik_solver is not None:
        count = len(log["t"])
        log["ee_meas"] = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
        log["ee_ref"] = reference["ee_ref"][:count]
        log["ee_ik"] = reference["ee_ik"][:count]
    np.savez_compressed(output / "arrays" / f"closed_loop_{model.name}.npz", **log)
    values = tracking_metrics(log)
    if ik_solver is not None:
        values["cartesian"] = cartesian_tracking_metrics(
            ee_meas=log["ee_meas"],
            ee_ref=log["ee_ref"],
            ik_error=reference["ik_error"][: len(log["t"])],
        )
    metrics: dict[str, object] = {"models": {model.name: values}, "task": args.task}
    metrics["figures"] = plot_closed_loop(log, figures_dir_for_result(output), label=model.name)
    print(f"[lqr] {model.name}: rmse_q={values['rmse_q']:.6g}")
    save_json(output / "metrics" / "tracking_metrics.json", metrics)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
