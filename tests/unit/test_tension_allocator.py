import numpy as np

from cable_robotics.safety import TensionLimits, apply_tension_limits
from cable_robotics.tension_allocator import (
    AntagonisticLayout,
    allocate_antagonistic_tensions,
)


def test_antagonistic_allocator_respects_preload_and_limit() -> None:
    jacobian = np.zeros((4, 2), dtype=np.float64)
    jacobian[0, 0] = 0.1
    jacobian[1, 0] = -0.1
    jacobian[2, 1] = 0.2
    jacobian[3, 1] = -0.2
    layout = AntagonisticLayout(
        cable_count=4,
        positive_groups=((0,), (2,)),
        negative_groups=((1,), (3,)),
        dof_groups=((0,), (1,)),
    )
    tensions, residuals = allocate_antagonistic_tensions(
        np.array([2.0, -3.0]),
        jacobian,
        layout,
        f_pre=5.0,
        f_max=40.0,
    )
    assert tensions.shape == (4,)
    assert residuals.shape == (2,)
    assert np.all(tensions >= 5.0)
    assert np.all(tensions <= 40.0)


def test_tension_rate_limit() -> None:
    limited = apply_tension_limits(
        np.array([20.0, 0.0]),
        TensionLimits(minimum=5.0, maximum=30.0, max_rate=10.0),
        previous=np.array([10.0, 10.0]),
        dt=0.1,
    )
    np.testing.assert_allclose(limited, np.array([11.0, 9.0]))
