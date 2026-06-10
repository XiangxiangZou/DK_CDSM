"""CDSM adapter for the generic Koopman closed-loop runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cable_robotics.tension_allocator import (
    F_PRELOAD,
    allocate_antagonistic_tensions,
)
from cdsm.constants import make_tension_layout
from cdsm.plants.base import CDSMPlant
from cdsm.plants.mujoco import MujocoCablePlant
from koopman_control.control.closed_loop import (
    run_model_predictive_tracking,
)
from koopman_control.control.finite_horizon_lqr import LqrConfig


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
