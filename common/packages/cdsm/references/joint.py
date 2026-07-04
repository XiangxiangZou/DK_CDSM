"""Joint-space references for the two active CDSM coordinates."""

from __future__ import annotations

import numpy as np


def build_joint_ramp_reference(
    *,
    dt: float,
    duration: float,
    q_start: np.ndarray,
    q_target: np.ndarray,
    ramp_duration: float,
) -> dict[str, np.ndarray]:
    """Build q/dq/x references for CDSM joint tracking."""
    q0 = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q_target, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape:
        raise ValueError("q_start and q_target must have the same shape")
    n = int(round(float(duration) / float(dt))) + 1
    t = np.arange(n, dtype=np.float64) * float(dt)
    s = np.clip(t / max(float(ramp_duration), float(dt)), 0.0, 1.0)
    s = 3.0 * s * s - 2.0 * s * s * s
    q_ref = q0[None, :] + (q1 - q0)[None, :] * s[:, None]
    dq_ref = np.gradient(q_ref, float(dt), axis=0)
    return {
        "t": t,
        "q_ref": q_ref,
        "dq_ref": dq_ref,
        "x_ref": np.hstack([q_ref, dq_ref]),
    }
