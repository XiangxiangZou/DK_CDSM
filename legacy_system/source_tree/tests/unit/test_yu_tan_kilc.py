import numpy as np

from koopman_control.control.yu_tan_kilc import (
    YuTanKILCConfig,
    YuTanKILCController,
)


def test_yu_tan_kilc_uses_lifted_error_and_records_control_terms() -> None:
    A_c = np.zeros((5, 5), dtype=np.float64)
    B_c = np.zeros((5, 2), dtype=np.float64)
    B_c[:2, :2] = np.eye(2)
    controller = YuTanKILCController(
        A_c,
        B_c,
        dt=0.01,
        config=YuTanKILCConfig(
            learning_rate=0.2,
            adaptive_gain=0.1,
            robust_gain=0.05,
            robust_boundary=0.5,
            filter_cutoff=20.0,
            control_limit=10.0,
        ),
    )
    u_prev = np.zeros((64, 2), dtype=np.float64)
    z_ref = np.zeros((64, 5), dtype=np.float64)
    z_meas = np.zeros((64, 5), dtype=np.float64)
    z_ref[:, 0] = 0.2
    z_ref[:, 1] = -0.1
    t = np.arange(64, dtype=np.float64) * 0.01

    update = controller.update(u_prev, z_ref, z_meas, t)

    assert update.e_z.shape == z_ref.shape
    assert update.u_ilc.shape == u_prev.shape
    assert update.u_adaptive.shape == u_prev.shape
    assert update.u_robust.shape == u_prev.shape
    assert update.u_total.shape == u_prev.shape
    assert np.isfinite(update.u_total).all()
    assert np.linalg.norm(update.u_total) > 0.0
    assert np.max(np.abs(update.u_total)) <= 10.0
