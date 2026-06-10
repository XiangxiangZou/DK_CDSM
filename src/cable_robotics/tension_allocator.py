"""Configurable antagonistic cable-tension allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

F_PRELOAD = 20.0
F_MAX_CABLE: Optional[float] = None


@dataclass(frozen=True)
class AntagonisticLayout:
    """Map generalized joints to positive and negative cable groups."""

    cable_count: int
    positive_groups: tuple[tuple[int, ...], ...]
    negative_groups: tuple[tuple[int, ...], ...]
    dof_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        joint_count = len(self.dof_groups)
        if len(self.positive_groups) != joint_count:
            raise ValueError("positive_groups and dof_groups must align")
        if len(self.negative_groups) != joint_count:
            raise ValueError("negative_groups and dof_groups must align")
        indices = [
            index
            for groups in (self.positive_groups, self.negative_groups)
            for group in groups
            for index in group
        ]
        if not indices:
            raise ValueError("at least one cable group is required")
        if min(indices) < 0 or max(indices) >= self.cable_count:
            raise ValueError("cable index outside configured cable_count")


def solve_antagonistic_pair(
    m_p: float,
    m_m: float,
    tau_des: float,
    f_pre: float,
    f_max: Optional[float],
) -> Tuple[float, float, float]:
    """Allocate one desired joint torque to an antagonistic cable pair."""
    tau_base = (m_p + m_m) * f_pre
    tau_eff = tau_des - tau_base
    if f_max is None or not np.isfinite(f_max):
        increment_max = float("inf")
    else:
        increment_max = max(float(f_max) - float(f_pre), 0.0)

    candidates = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > 1e-12:
        increment = min(max(tau_eff / m_p, 0.0), increment_max)
        candidates.append(
            (
                increment,
                0.0,
                abs(tau_eff - m_p * increment),
                increment,
            )
        )
    if abs(m_m) > 1e-12:
        increment = min(max(tau_eff / m_m, 0.0), increment_max)
        candidates.append(
            (
                0.0,
                increment,
                abs(tau_eff - m_m * increment),
                increment,
            )
        )
    positive, negative, residual, _ = min(
        candidates,
        key=lambda item: (item[2], item[3]),
    )
    return f_pre + positive, f_pre + negative, residual


def _group_moment_arm(
    tendon_jacobian: np.ndarray,
    cable_group: Sequence[int],
    dof_group: Sequence[int],
) -> float:
    return float(
        np.asarray(tendon_jacobian, dtype=np.float64)[
            np.ix_(tuple(cable_group), tuple(dof_group))
        ].sum()
    )


def allocate_antagonistic_tensions(
    desired_torques: np.ndarray,
    tendon_jacobian: np.ndarray,
    layout: AntagonisticLayout,
    *,
    f_pre: float = F_PRELOAD,
    f_max: Optional[float] = F_MAX_CABLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate joint torques and return tensions plus residuals."""
    torques = np.asarray(desired_torques, dtype=np.float64).reshape(-1)
    if torques.shape[0] != len(layout.dof_groups):
        raise ValueError(
            "desired_torques length must match layout joint count"
        )
    tensions = np.full(
        layout.cable_count,
        float(f_pre),
        dtype=np.float64,
    )
    residuals = np.zeros(torques.shape[0], dtype=np.float64)
    for joint, torque in enumerate(torques):
        positive_group = layout.positive_groups[joint]
        negative_group = layout.negative_groups[joint]
        dof_group = layout.dof_groups[joint]
        m_positive = _group_moment_arm(
            tendon_jacobian,
            positive_group,
            dof_group,
        )
        m_negative = _group_moment_arm(
            tendon_jacobian,
            negative_group,
            dof_group,
        )
        positive, negative, residual = solve_antagonistic_pair(
            m_positive,
            m_negative,
            float(torque),
            float(f_pre),
            f_max,
        )
        tensions[list(positive_group)] = positive
        tensions[list(negative_group)] = negative
        residuals[joint] = residual
    return tensions, residuals


def cable_antagonistic_map(
    tau_a_des: float,
    tau_b_des: float,
    tendon_jacobian: np.ndarray,
    dof_j1: int,
    dof_j2: int,
    dof_j3: int,
    dof_j4: int,
    f_pre: float = F_PRELOAD,
    f_max: Optional[float] = F_MAX_CABLE,
) -> np.ndarray:
    """Compatibility wrapper for the current two-joint, eight-cable CDSM."""
    layout = AntagonisticLayout(
        cable_count=8,
        positive_groups=((0, 2), (4, 6)),
        negative_groups=((1, 3), (5, 7)),
        dof_groups=((dof_j1, dof_j2), (dof_j3, dof_j4)),
    )
    tensions, _ = allocate_antagonistic_tensions(
        np.array([tau_a_des, tau_b_des]),
        tendon_jacobian,
        layout,
        f_pre=f_pre,
        f_max=f_max,
    )
    return tensions
