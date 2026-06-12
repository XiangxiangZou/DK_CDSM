import numpy as np

from koopman_control.control.finite_horizon_lqr import (
    KoopmanConstrainedMpcTracker,
    LqrConfig,
)


def test_constrained_mpc_respects_mapped_physical_bounds() -> None:
    tracker = KoopmanConstrainedMpcTracker(
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        cfg=LqrConfig(
            horizon=4,
            Qq=1.0,
            Qdq=1.0,
            R=1e-6,
            Rd=1e-6,
            output_weights=(10.0,),
        ),
    )
    solution = tracker.solve(
        z0=np.array([0.0]),
        ref_norm=np.full((4, 1), 10.0),
        u_prev_internal=np.array([0.0]),
        physical_from_internal=np.array([[2.0]]),
        physical_lower_norm=np.array([-0.5]),
        physical_upper_norm=np.array([0.5]),
    )
    mapped = 2.0 * solution[:, 0]
    assert tracker.last_status.lower().startswith("solved")
    assert np.all(mapped >= -0.5 - 1e-7)
    assert np.all(mapped <= 0.5 + 1e-7)
    assert mapped[0] > 0.49
