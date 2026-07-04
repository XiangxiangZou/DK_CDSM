"""Continuous-DKUC lifted-state KILC control."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from io_utils import DEFAULT_XML, make_output_dir, manifest, save_json
from cable_interface import apply_joint_torque
from references import build_cartesian_circle_reference, ik_config_from_args

from cdsm.kinematics.ik import MujocoSiteIK
from cdsm.plants.mujoco import MujocoCablePlant
from koopman_control.evaluation.tracking import (
    cartesian_tracking_metrics,
    tracking_metrics,
)
from koopman_control.models.registry import load_continuous_dkuc_model


@dataclass(frozen=True)
class KILCConfig:
    learning_rate: float = 1.0
    adaptive_gain: float = 0.001
    robust_gain: float = 0.001
    robust_boundary: float = 0.1
    feedback_kp_a: float = 80.0
    feedback_kp_b: float = 70.0
    feedback_kd_a: float = 8.0
    feedback_kd_b: float = 7.0


@dataclass(frozen=True)
class KILCUpdate:
    e_z: np.ndarray
    u_ilc: np.ndarray
    u_adaptive: np.ndarray
    u_robust: np.ndarray
    u_total: np.ndarray


class LiftedKILC:
    """PD feedback online plus lifted-space ILC update between trials."""

    def __init__(self, A_c: np.ndarray, B_c: np.ndarray, *, dt: float, cfg: KILCConfig) -> None:
        self.A_c = np.asarray(A_c, dtype=np.float64)
        self.B_c = np.asarray(B_c, dtype=np.float64)
        self.dt = float(dt)
        self.cfg = cfg
        self.control_dim = int(self.B_c.shape[1])
        column_energy = np.maximum(np.sum(self.B_c * self.B_c, axis=0), 1e-8)
        self.L = self.B_c.T / column_energy.reshape(-1, 1)
        self.K = np.array(
            [
                [cfg.feedback_kp_a, 0.0, cfg.feedback_kd_a, 0.0],
                [0.0, cfg.feedback_kp_b, 0.0, cfg.feedback_kd_b],
            ],
            dtype=np.float64,
        )

    def feedback(self, e_phys: np.ndarray) -> np.ndarray:
        return self.K @ np.asarray(e_phys, dtype=np.float64).reshape(-1)

    def update(
        self,
        u_prev: np.ndarray,
        z_ref: np.ndarray,
        z_meas: np.ndarray,
    ) -> KILCUpdate:
        e_z = np.asarray(z_ref, dtype=np.float64) - np.asarray(z_meas, dtype=np.float64)
        projected = e_z @ self.L.T
        u_ilc = np.asarray(u_prev, dtype=np.float64) + self.cfg.learning_rate * projected
        u_adaptive = self.cfg.adaptive_gain * np.cumsum(projected, axis=0) * self.dt
        boundary = max(self.cfg.robust_boundary, 1e-9)
        u_robust = self.cfg.robust_gain * np.tanh(projected / boundary)
        return KILCUpdate(
            e_z=e_z,
            u_ilc=u_ilc,
            u_adaptive=u_adaptive,
            u_robust=u_robust,
            u_total=np.asarray(u_ilc + u_adaptive + u_robust, dtype=np.float64),
        )


def _state_reference(reference: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(reference["t"], dtype=np.float64)
    if "x_ref" in reference:
        x_ref = np.asarray(reference["x_ref"], dtype=np.float64)
    else:
        x_ref = np.hstack([reference["q_ref"], reference["dq_ref"]])
    return t, x_ref


def _control_norm_to_physical(model, values_norm: np.ndarray) -> np.ndarray:
    mean = np.asarray(model.u_normer.mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(model.u_normer.std, dtype=np.float64).reshape(1, -1)
    return np.asarray(values_norm, dtype=np.float64) * std + mean


def _run_trial(
    *,
    model,
    controller: LiftedKILC,
    plant: MujocoCablePlant,
    x_ref: np.ndarray,
    t: np.ndarray,
    controls_norm: dict[str, np.ndarray],
    tau_limit: float,
    f_preload: float,
    f_max_cable: float | None,
) -> dict[str, np.ndarray]:
    plant.set_state(x_ref[0, :2], x_ref[0, 2:])
    u_total_phys = _control_norm_to_physical(model, controls_norm["u_total"])
    records: dict[str, list] = {
        "t": [],
        "x_meas": [],
        "x_ref": [],
        "z_meas": [],
        "z_ref": [],
        "e_z": [],
        "u_fb": [],
        "u_total": [],
        "control_cmd": [],
        "cable_tensions": [],
        "allocation_residual": [],
        "saturated": [],
    }
    for k in range(len(t) - 1):
        measured = plant.read_state()
        u_fb = controller.feedback(x_ref[k] - measured)
        raw_cmd = u_fb + u_total_phys[k]
        tau_cmd = np.clip(raw_cmd, -tau_limit, tau_limit)
        tensions, residual = apply_joint_torque(
            plant,
            tau_cmd,
            f_preload=f_preload,
            f_max_cable=f_max_cable,
        )
        plant.step()
        measured_after = plant.read_state()
        z_meas = model.lift(measured_after)
        z_ref = model.lift(x_ref[k + 1])
        records["t"].append(t[k])
        records["x_meas"].append(measured.copy())
        records["x_ref"].append(x_ref[k].copy())
        records["z_meas"].append(z_meas)
        records["z_ref"].append(z_ref)
        records["e_z"].append(z_ref - z_meas)
        records["u_fb"].append(u_fb.copy())
        records["u_total"].append(u_total_phys[k].copy())
        records["control_cmd"].append(tau_cmd.copy())
        records["cable_tensions"].append(tensions.copy())
        records["allocation_residual"].append(residual.copy())
        records["saturated"].append(np.any(np.abs(raw_cmd) > tau_limit))
    return {key: np.asarray(value) for key, value in records.items()}


def run_kilc_tracking(
    *,
    model,
    reference: dict[str, np.ndarray],
    cfg: KILCConfig,
    xml_path: str | Path,
    dt: float,
    max_trials: int,
    tau_limit: float,
    f_preload: float,
    f_max_cable: float | None,
    show_progress: bool,
) -> dict[str, Any]:
    if str(getattr(model, "control_mode", "")) != "zdot=A_c z+B_c u_norm":
        raise ValueError("KILC requires a continuous-time DKUC model")
    plant = MujocoCablePlant(xml_path, dt)
    t, x_ref = _state_reference(reference)
    z_ref_all = np.vstack([model.lift(x) for x in x_ref[1:]])
    controller = LiftedKILC(model.A, model.B, dt=dt, cfg=cfg)
    zeros = np.zeros((len(t) - 1, model.control_dim), dtype=np.float64)
    controls_norm = {
        "u_ilc": zeros.copy(),
        "u_adaptive": zeros.copy(),
        "u_robust": zeros.copy(),
        "u_total": zeros.copy(),
    }
    trials: list[dict[str, np.ndarray]] = []
    for trial in range(int(max_trials)):
        log = _run_trial(
            model=model,
            controller=controller,
            plant=plant,
            x_ref=x_ref,
            t=t,
            controls_norm=controls_norm,
            tau_limit=tau_limit,
            f_preload=f_preload,
            f_max_cable=f_max_cable,
        )
        trials.append(log)
        if show_progress:
            error = np.asarray(log["x_ref"]) - np.asarray(log["x_meas"])
            print(f"[kilc] trial={trial} rmse_q={np.sqrt(np.mean(error[:, :2] ** 2)):.6g}")
        update = controller.update(
            controls_norm["u_total"],
            z_ref_all,
            np.asarray(log["z_meas"], dtype=np.float64),
        )
        controls_norm = {
            "u_ilc": update.u_ilc,
            "u_adaptive": update.u_adaptive,
            "u_robust": update.u_robust,
            "u_total": update.u_total,
        }
    return {"trials": trials, "u_final_norm": controls_norm["u_total"], "z_ref": z_ref_all}


def _trial_metrics(reference: dict[str, np.ndarray], log: dict[str, np.ndarray], ik_solver: MujocoSiteIK) -> dict[str, Any]:
    count = len(log["t"])
    ee_meas = ik_solver.forward_xy_batch(log["x_meas"][:, :2])
    values = tracking_metrics(log)
    values["cartesian"] = cartesian_tracking_metrics(
        ee_meas=ee_meas,
        ee_ref=reference["ee_ref"][:count],
        ik_error=reference["ik_error"][:count],
    )
    values["lifted_error_rms"] = float(np.sqrt(np.mean(log["e_z"] ** 2)))
    values["tau_peak_abs"] = float(np.max(np.abs(log["control_cmd"])))
    values["cable_tension_peak"] = float(np.max(log["cable_tensions"]))
    values["saturation_ratio"] = float(np.mean(log["saturated"].astype(np.float64)))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run continuous-DKUC KILC control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--run_type", choices=("smoke_test", "full_run"), default="smoke_test")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--tau_limit", type=float, default=120.0)
    parser.add_argument("--f_preload", type=float, default=50.0)
    parser.add_argument("--f_max_cable", type=float, default=2000.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--period", type=float, default=20.0)
    parser.add_argument("--num_cycles", type=float, default=1.0)
    parser.add_argument("--center_x", type=float, default=4.5)
    parser.add_argument("--center_y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--start_hold", type=float, default=0.0)
    parser.add_argument("--time_scaling", choices=("linear", "quintic"), default="quintic")
    parser.add_argument("--ik_site", default="end_effector")
    parser.add_argument("--ik_max_iter", type=int, default=200)
    parser.add_argument("--ik_tol", type=float, default=1e-5)
    parser.add_argument("--ik_damping", type=float, default=1e-4)
    parser.add_argument("--ik_max_step", type=float, default=0.08)
    parser.add_argument("--ik_joint_margin", type=float, default=0.15)
    parser.add_argument("--ik_smooth_window_s", type=float, default=0.03)
    parser.add_argument("--ik_seed_a", type=float, default=0.4)
    parser.add_argument("--ik_seed_b", type=float, default=-0.1)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--adaptive_gain", type=float, default=0.001)
    parser.add_argument("--robust_gain", type=float, default=0.001)
    parser.add_argument("--robust_boundary", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = KILCConfig(
        learning_rate=args.learning_rate,
        adaptive_gain=args.adaptive_gain,
        robust_gain=args.robust_gain,
        robust_boundary=args.robust_boundary,
    )
    output = make_output_dir(args.run_type, "kilc", args.tag)
    ik_solver = MujocoSiteIK(args.xml, args.dt, ik_config_from_args(args))
    reference = build_cartesian_circle_reference(args, ik_solver)
    np.savez_compressed(output / "arrays" / "reference.npz", **reference)
    save_json(
        output / "manifest.json",
        {
            **manifest("control/kilc_control.py", sys.argv[1:]),
            "artifact_dir": str(Path(args.artifact_dir)),
            "config": asdict(cfg),
        },
    )
    model = load_continuous_dkuc_model(args.artifact_dir, args.device)
    result = run_kilc_tracking(
        model=model,
        reference=reference,
        cfg=cfg,
        xml_path=args.xml,
        dt=args.dt,
        max_trials=args.max_trials,
        tau_limit=args.tau_limit,
        f_preload=args.f_preload,
        f_max_cable=None if args.f_max_cable < 0.0 else args.f_max_cable,
        show_progress=not args.quiet,
    )
    final = result["trials"][-1]
    np.savez_compressed(
        output / "arrays" / "kilc_result.npz",
        final_x_meas=final["x_meas"],
        final_e_z=final["e_z"],
        final_control_cmd=final["control_cmd"],
        final_cable_tensions=final["cable_tensions"],
        u_final_norm=result["u_final_norm"],
    )
    metrics = {
        "trials": [_trial_metrics(reference, log, ik_solver) for log in result["trials"]],
    }
    metrics["final"] = metrics["trials"][-1]
    save_json(output / "metrics" / "kilc_metrics.json", metrics)
    print(json.dumps(metrics["final"], indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
