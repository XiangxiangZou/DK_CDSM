"""CDSM runtime for Yu-Tan-style lifted-state KILC.

Paper-aligned architecture:
  - PD feedback (K * e_phys) is applied online during every trial.
  - ILC feedforward (u_ilc + u_adaptive + u_robust) is updated
    between trials using lifted-state error.
  - No empirical Q-filter — stability follows from Lyapunov analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cable_robotics.tension_allocator import F_PRELOAD
from cdsm.plants.base import CDSMPlant
from cdsm.plants.mujoco import MujocoCablePlant
from cdsm.runtime.tracking import apply_joint_torque_as_tensions
from koopman_control.control.yu_tan_kilc import (
    YuTanKILCConfig,
    YuTanKILCController,
)


@dataclass(frozen=True)
class KILCTrialSummary:
    trial: int
    rmse_q: float
    rmse_dq: float
    lifted_error_rms: float
    tau_rms: float
    tau_peak_abs: float
    cable_peak: float
    saturation_ratio: float


def _reference_state(reference: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(reference["t"], dtype=np.float64)
    if "x_ref" in reference:
        x_ref = np.asarray(reference["x_ref"], dtype=np.float64)
    else:
        x_ref = np.hstack(
            [
                np.asarray(reference["q_ref"], dtype=np.float64),
                np.asarray(reference["dq_ref"], dtype=np.float64),
            ]
        )
    if len(t) != len(x_ref):
        raise ValueError("reference t and state lengths do not match")
    return t, x_ref


def _lift_sequence(model, x_values: np.ndarray) -> np.ndarray:
    return np.vstack([model.lift(x) for x in np.asarray(x_values, dtype=np.float64)])


def _control_norm_to_physical(model, values_norm: np.ndarray) -> np.ndarray:
    """Convert absolute normalized control sequence to physical torque."""
    mean = np.asarray(model.u_normer.mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(model.u_normer.std, dtype=np.float64).reshape(1, -1)
    return np.asarray(values_norm, dtype=np.float64) * std + mean


def _run_trial(
    *,
    model,
    controller: YuTanKILCController,
    plant: CDSMPlant,
    x_ref: np.ndarray,
    t: np.ndarray,
    controls_norm: dict[str, np.ndarray],
    tau_limit: float,
    f_preload: float,
    f_max_cable: float | None,
) -> dict[str, np.ndarray]:
    """Execute one full trial with PD feedback + ILC feedforward.

    At each step:
        u_cmd = K @ (x_ref - x_meas)  +  u_ff(learned)
                \_____ PD feedback ___/    \__ ILC feedforward __/
    """
    plant.set_state(x_ref[0, :2], x_ref[0, 2:])
    u_total_phys = _control_norm_to_physical(model, controls_norm["u_total"])
    u_ilc_phys = _control_norm_to_physical(model, controls_norm["u_ilc"])
    u_adaptive_phys = _control_norm_to_physical(model, controls_norm["u_adaptive"])
    u_robust_phys = _control_norm_to_physical(model, controls_norm["u_robust"])
    records: dict[str, list] = {
        "t": [],
        "x_meas": [],
        "x_ref": [],
        "z_meas": [],
        "z_ref": [],
        "e_z": [],
        "u_fb": [],
        "u_ilc": [],
        "u_adaptive": [],
        "u_robust": [],
        "u_total": [],
        "control_cmd": [],
        "cable_tensions": [],
        "solve_ms": [],
        "saturated": [],
    }
    for k in range(len(t) - 1):
        measured = plant.read_state()
        e_phys = x_ref[k] - measured

        # PD feedback  (paperʼs K * e)
        u_fb = controller.feedback(e_phys)
        # ILC feedforward  (learned from previous trials)
        u_ff = u_total_phys[k]

        raw_cmd = u_fb + u_ff
        cmd = np.clip(raw_cmd, -tau_limit, tau_limit)

        tensions = apply_joint_torque_as_tensions(
            plant,
            cmd,
            f_preload=f_preload,
            f_max_cable=f_max_cable,
        )
        plant.step()

        # Record z AFTER step:  z(t_{k+1}) reflects the effect of u[k].
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
        records["u_ilc"].append(u_ilc_phys[k].copy())
        records["u_adaptive"].append(u_adaptive_phys[k].copy())
        records["u_robust"].append(u_robust_phys[k].copy())
        records["u_total"].append(u_ff.copy())
        records["control_cmd"].append(cmd.copy())
        records["cable_tensions"].append(tensions.copy())
        records["solve_ms"].append(0.0)
        records["saturated"].append(np.any(np.abs(raw_cmd) > tau_limit))
    return {key: np.asarray(value) for key, value in records.items()}


def _summarize_trial(trial: int, log: dict[str, np.ndarray]) -> KILCTrialSummary:
    error = np.asarray(log["x_ref"]) - np.asarray(log["x_meas"])
    tau = np.asarray(log["control_cmd"])
    cables = np.asarray(log["cable_tensions"])
    return KILCTrialSummary(
        trial=int(trial),
        rmse_q=float(np.sqrt(np.mean(error[:, :2] ** 2))),
        rmse_dq=float(np.sqrt(np.mean(error[:, 2:] ** 2))),
        lifted_error_rms=float(np.sqrt(np.mean(np.asarray(log["e_z"]) ** 2))),
        tau_rms=float(np.sqrt(np.mean(tau * tau))),
        tau_peak_abs=float(np.max(np.abs(tau))),
        cable_peak=float(np.max(cables)),
        saturation_ratio=float(np.mean(np.asarray(log["saturated"], dtype=np.float64))),
    )


def run_yu_tan_kilc_tracking(
    *,
    model,
    reference: dict[str, np.ndarray],
    kilc_config: YuTanKILCConfig,
    dt: float = 0.01,
    plant: CDSMPlant | None = None,
    xml_path: str | Path | None = None,
    max_trials: int = 20,
    tau_limit: float = 120.0,
    f_preload: float = F_PRELOAD,
    f_max_cable: float | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Run paper-aligned KILC (PD feedback + lifted-space ILC) on CDSM."""
    if str(getattr(model, "control_mode", "")) != "zdot=A_c z+B_c u_norm":
        raise ValueError("Yu-Tan-style KILC requires a continuous-time DKUC model")
    if plant is None:
        if xml_path is None:
            raise ValueError("plant or xml_path is required")
        plant = MujocoCablePlant(xml_path, dt)

    t, x_ref = _reference_state(reference)
    # z_ref_all[k] = lift(x_ref[k+1])  aligns with z_meas recorded after step k.
    z_ref_all = _lift_sequence(model, x_ref[1:])

    controller = YuTanKILCController(
        model.A,
        model.B,
        dt=dt,
        config=kilc_config,
    )

    zeros = np.zeros((len(t) - 1, model.control_dim), dtype=np.float64)
    controls_norm = {
        "u_ilc": zeros.copy(),
        "u_adaptive": zeros.copy(),
        "u_robust": zeros.copy(),
        "u_total": zeros.copy(),
    }
    trial_logs: list[dict[str, np.ndarray]] = []
    summaries: list[KILCTrialSummary] = []

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
        trial_logs.append(log)
        summary = _summarize_trial(trial, log)
        summaries.append(summary)
        if show_progress:
            print(
                f"[KILC] trial {trial:3d} "
                f"rmse_q={summary.rmse_q:.6g} "
                f"lifted={summary.lifted_error_rms:.6g} "
                f"tau_peak={summary.tau_peak_abs:.3f}"
            )

        update = controller.update(
            controls_norm["u_total"],
            z_ref_all,
            np.asarray(log["z_meas"], dtype=np.float64),
            np.asarray(log["t"], dtype=np.float64),
        )
        controls_norm = {
            "u_ilc": update.u_ilc,
            "u_adaptive": update.u_adaptive,
            "u_robust": update.u_robust,
            "u_total": update.u_total,
        }

    return {
        "trials": trial_logs,
        "trial_summaries": summaries,
        "u_final_norm": controls_norm["u_total"],
        "z_ref": z_ref_all,
        "config": kilc_config,
    }
