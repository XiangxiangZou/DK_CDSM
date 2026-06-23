import numpy as np
import torch

from koopman_control.data.normalization import Normalizer
from koopman_control.models.networks import ContinuousDKUCNetwork
from koopman_control.training.continuous_dkuc import (
    build_continuous_training_arrays,
)


def test_continuous_dkuc_network_exposes_derivative_dynamics() -> None:
    model = ContinuousDKUCNetwork(6, (8,), "elu", 1.0)
    state = torch.zeros((3, 4))
    control = torch.zeros((3, 2))

    lifted = model.lift(state)
    zdot = model.derivative(lifted, control)

    assert lifted.shape == (3, 10)
    assert zdot.shape == lifted.shape
    assert model.state_from_latent(zdot).shape == state.shape


def test_build_continuous_training_arrays_uses_finite_differences() -> None:
    states = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.2, 0.0],
                [0.3, 0.0, 0.4, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    inputs = np.zeros((1, 2, 2), dtype=np.float64)
    x_normer = Normalizer.fit(states.reshape(-1, 4))
    u_normer = Normalizer.fit(inputs.reshape(-1, 2))

    arrays = build_continuous_training_arrays(
        states,
        inputs,
        dt=0.1,
        x_normer=x_normer,
        u_normer=u_normer,
    )

    assert arrays.x_norm.shape == (2, 4)
    assert arrays.u_norm.shape == (2, 2)
    assert arrays.xdot_norm.shape == (2, 4)
    np.testing.assert_allclose(arrays.xdot_phys[0], [1.0, 0.0, 2.0, 0.0])
    assert np.isfinite(arrays.xdot_norm).all()
