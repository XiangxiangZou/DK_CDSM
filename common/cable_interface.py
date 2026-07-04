"""Shared cable actuation interface for all control methods."""

from __future__ import annotations

import numpy as np

from cable_robotics.tension_allocator import (
    F_PRELOAD,
    allocate_antagonistic_tensions,
    antagonistic_torque_bounds,
)
from cdsm.constants import make_tension_layout
from cdsm.plants.base import CDSMPlant


def joint_torque_to_cable_tensions(
    plant: CDSMPlant,
    tau_cmd: np.ndarray,
    *,
    f_preload: float = F_PRELOAD,
    f_max_cable: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate joint torque to cable tensions and return residual torque."""
    jacobian = plant.compute_tendon_jacobian()
    layout = make_tension_layout(*plant.torque_dofs())
    tensions, residual = allocate_antagonistic_tensions(
        np.asarray(tau_cmd, dtype=np.float64),
        jacobian,
        layout,
        f_pre=f_preload,
        f_max=f_max_cable,
    )
    return tensions, residual


def apply_joint_torque(
    plant: CDSMPlant,
    tau_cmd: np.ndarray,
    *,
    f_preload: float = F_PRELOAD,
    f_max_cable: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate and apply cable tensions for a joint torque command."""
    tensions, residual = joint_torque_to_cable_tensions(
        plant,
        tau_cmd,
        f_preload=f_preload,
        f_max_cable=f_max_cable,
    )
    plant.apply_cable_tensions(tensions)
    return tensions, residual


def joint_torque_bounds_from_cable_limits(
    plant: CDSMPlant,
    *,
    f_preload: float = F_PRELOAD,
    f_max_cable: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return joint torque lower/upper bounds induced by cable limits."""
    jacobian = plant.compute_tendon_jacobian()
    layout = make_tension_layout(*plant.torque_dofs())
    return antagonistic_torque_bounds(
        jacobian,
        layout,
        f_pre=f_preload,
        f_max=f_max_cable,
    )
