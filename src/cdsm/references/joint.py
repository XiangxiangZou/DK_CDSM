"""Joint-space references for the two active CDSM coordinates."""

from __future__ import annotations

import numpy as np

from koopman_control.control.finite_horizon_lqr import (
    build_ramp_reference,
)


def build_joint_ramp_reference(
    *,
    dt: float,
    duration: float,
    q_start: np.ndarray,
    q_target: np.ndarray,
    ramp_duration: float,
) -> dict[str, np.ndarray]:
    """Build q/dq/x references for CDSM joint tracking."""
    result = build_ramp_reference(
        dt=dt,
        duration=duration,
        start=q_start,
        target=q_target,
        ramp_duration=ramp_duration,
    )
    q_ref = result["values"]
    dq_ref = result["rates"]
    return {
        "t": result["t"],
        "q_ref": q_ref,
        "dq_ref": dq_ref,
        "x_ref": np.hstack([q_ref, dq_ref]),
    }
