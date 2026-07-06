"""CDSM adapter for the generic Koopman closed-loop runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cable_robotics.tension_allocator import (
    F_PRELOAD,
    allocate_antagonistic_tensions,
    antagonistic_torque_bounds,
)
from cdsm.constants import make_tension_layout
from cdsm.plants.base import CDSMPlant
from cdsm.plants.mujoco import MujocoCablePlant
from koopman_control.control.closed_loop import (
    run_model_predictive_tracking,
)
from koopman_control.control.finite_horizon_lqr import LqrConfig
from koopman_control.control.finite_horizon_lqr import (
    KoopmanConstrainedMpcTracker,
    future_reference,
)


def apply_joint_torque_as_tensions(
    plant: CDSMPlant,
    tau_cmd: np.ndarray,
    *,
    f_preload: float = F_PRELOAD,
    f_max_cable: float | None = None,
) -> np.ndarray:
    """Allocate a two-joint command and apply the eight cable tensions."""
    jacobian = plant.compute_tendon_jacobian()
    layout = make_tension_layout(*plant.torque_dofs())
    tensions, _ = allocate_antagonistic_tensions(
        np.asarray(tau_cmd, dtype=np.float64),
        jacobian,
        layout,
        f_pre=f_preload,
        f_max=f_max_cable,
    )
    plant.apply_cable_tensions(tensions)
    return tensions


def run_joint_space_closed_loop_model(
    *,
    model,
    reference: dict[str, np.ndarray],
    controller_config: LqrConfig,
    tau_limit: float,
    plant: CDSMPlant | None = None,
    xml_path: str | Path | None = None,
    dt: float = 0.01,
    f_preload: float = F_PRELOAD,
    f_max_cable: float | None = None,
) -> dict[str, np.ndarray]:
    """Run CDSM tracking on an injected plant or a new MuJoCo plant."""
    if plant is None:
        if xml_path is None:
            raise ValueError("xml_path is required when plant is not provided")
        plant = MujocoCablePlant(xml_path, dt)
    if "x_ref" in reference:
        x_ref = np.asarray(reference["x_ref"], dtype=np.float64)
    else:
        x_ref = np.hstack(
            [
                np.asarray(reference["q_ref"], dtype=np.float64),
                np.asarray(reference["dq_ref"], dtype=np.float64),
            ]
        )

    def apply_control(target_plant, torque):
        return apply_joint_torque_as_tensions(
            target_plant,
            torque,
            f_preload=f_preload,
            f_max_cable=f_max_cable,
        )

    log = run_model_predictive_tracking(
        model=model,
        plant=plant,
        t=np.asarray(reference["t"], dtype=np.float64),
        states_ref=x_ref,
        initial_position_dim=2,
        controller_config=controller_config,
        control_limit=tau_limit,
        apply_control=apply_control,
    )
    # Compatibility aliases used by the existing plotting and metrics code.
    log["q_ref"] = log["x_ref"][:, :2]
    log["dq_ref"] = log["x_ref"][:, 2:]
    log["tau_cmd"] = log["control_cmd"]
    log["cable_tensions"] = log["actuator_cmd"]
    return log


def run_dkac_tension_constrained_mpc(
    *,
    model,
    reference: dict[str, np.ndarray],
    controller_config: LqrConfig,
    plant: CDSMPlant | None = None,
    xml_path: str | Path | None = None,
    dt: float = 0.01,
    f_preload: float = F_PRELOAD,
    f_max_cable: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Run DKAC MPC with constraints induced only by cable tensions."""
    if getattr(model, "name", "") != "dkac":
        raise ValueError("tension-constrained MPC currently requires DKAC")
    if plant is None:
        if xml_path is None:
            raise ValueError("xml_path is required when plant is not provided")
        plant = MujocoCablePlant(xml_path, dt)
    if not np.isfinite(f_max_cable) or f_max_cable < f_preload:
        raise ValueError("f_max_cable must be finite and >= f_preload")

    if "x_ref" in reference:
        x_ref = np.asarray(reference["x_ref"], dtype=np.float64)
    else:
        x_ref = np.hstack(
            [
                np.asarray(reference["q_ref"], dtype=np.float64),
                np.asarray(reference["dq_ref"], dtype=np.float64),
            ]
        )
    time_ref = np.asarray(reference["t"], dtype=np.float64)
    if len(time_ref) != len(x_ref):
        raise ValueError("reference t and state lengths do not match")

    plant.set_state(x_ref[0, :2], x_ref[0, 2:])
    tracker = KoopmanConstrainedMpcTracker(
        model.A,
        model.B,
        model.C,
        controller_config,
    )
    layout = make_tension_layout(*plant.torque_dofs())
    previous_internal = np.zeros(model.B.shape[1], dtype=np.float64)
    records: dict[str, list] = {
        "t": [],
        "x_meas": [],
        "x_ref": [],
        "control_cmd": [],
        "internal_control": [],
        "normalized_control": [],
        "actuator_cmd": [],
        "torque_lower": [],
        "torque_upper": [],
        "allocation_residual": [],
        "solve_ms": [],
        "mpc_iterations": [],
        "mpc_status": [],
    }

    import time

    for k in range(len(time_ref) - 1):
        measured = plant.read_state()
        z0 = model.lift(measured)
        normalized_ref = future_reference(
            model,
            x_ref,
            k,
            controller_config.horizon,
        )
        jacobian = plant.compute_tendon_jacobian()
        torque_lower, torque_upper = antagonistic_torque_bounds(
            jacobian,
            layout,
            f_pre=f_preload,
            f_max=f_max_cable,
        )
        lower_norm = (torque_lower - model.u_normer.mean) / model.u_normer.std
        upper_norm = (torque_upper - model.u_normer.mean) / model.u_normer.std
        # Receding-horizon linearization: freeze G(x) and the tendon geometry
        # over this QP, then recompute both from feedback at the next cycle.
        physical_from_internal = np.linalg.pinv(
            model.control_matrix(measured),
            rcond=1e-6,
        )

        started = time.perf_counter()
        internal_sequence = tracker.solve(
            z0,
            normalized_ref,
            previous_internal,
            physical_from_internal=physical_from_internal,
            physical_lower_norm=lower_norm,
            physical_upper_norm=upper_norm,
        )
        solve_ms = 1e3 * (time.perf_counter() - started)
        internal_command = internal_sequence[0]
        normalized_control = physical_from_internal @ internal_command
        physical_control = model.u_normer.inverse(
            normalized_control.reshape(1, -1)
        )[0]
        physical_control = np.clip(
            physical_control,
            torque_lower,
            torque_upper,
        )
        normalized_control = model.u_normer.transform(
            physical_control.reshape(1, -1)
        )[0]
        executed_internal = model.control_matrix(measured) @ normalized_control
        tensions, residual = allocate_antagonistic_tensions(
            physical_control,
            jacobian,
            layout,
            f_pre=f_preload,
            f_max=f_max_cable,
        )
        plant.apply_cable_tensions(tensions)
        plant.step()

        records["t"].append(time_ref[k])
        records["x_meas"].append(np.asarray(measured).copy())
        records["x_ref"].append(x_ref[k].copy())
        records["control_cmd"].append(physical_control.copy())
        records["internal_control"].append(executed_internal.copy())
        records["normalized_control"].append(normalized_control.copy())
        records["actuator_cmd"].append(tensions.copy())
        records["torque_lower"].append(torque_lower.copy())
        records["torque_upper"].append(torque_upper.copy())
        records["allocation_residual"].append(residual.copy())
        records["solve_ms"].append(solve_ms)
        records["mpc_iterations"].append(tracker.last_iterations)
        records["mpc_status"].append(tracker.last_status)
        previous_internal = executed_internal

    log = {
        key: np.asarray(values)
        for key, values in records.items()
    }
    log["q_ref"] = log["x_ref"][:, :2]
    log["dq_ref"] = log["x_ref"][:, 2:]
    log["tau_cmd"] = log["control_cmd"]
    log["cable_tensions"] = log["actuator_cmd"]
    return log
