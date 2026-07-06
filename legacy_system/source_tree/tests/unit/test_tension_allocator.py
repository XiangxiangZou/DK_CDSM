import numpy as np

from cable_robotics.safety import TensionLimits, apply_tension_limits
from cable_robotics.tension_allocator import (
    AntagonisticLayout,
    allocate_antagonistic_tensions,
    antagonistic_torque_bounds,
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


def test_antagonistic_torque_bounds_match_allocator_extrema() -> None:
    jacobian = np.array([[0.1], [-0.2]], dtype=np.float64)
    layout = AntagonisticLayout(
        cable_count=2,
        positive_groups=((0,),),
        negative_groups=((1,),),
        dof_groups=((0,),),
    )
    lower, upper = antagonistic_torque_bounds(
        jacobian,
        layout,
        f_pre=20.0,
        f_max=1000.0,
    )
    np.testing.assert_allclose(lower, np.array([-198.0]))
    np.testing.assert_allclose(upper, np.array([96.0]))

    for torque in (lower[0], upper[0]):
        tensions, residual = allocate_antagonistic_tensions(
            np.array([torque]),
            jacobian,
            layout,
            f_pre=20.0,
            f_max=1000.0,
        )
        assert np.all(tensions >= 20.0)
        assert np.all(tensions <= 1000.0)
        np.testing.assert_allclose(residual, np.zeros(1), atol=1e-10)
