import torch

from koopman_control.models.networks import (
    DKACNetwork,
    DKNNetwork,
    DKUCNetwork,
)


def test_neural_koopman_network_shapes() -> None:
    state = torch.zeros((3, 4))
    control = torch.zeros((3, 2))

    dkuc = DKUCNetwork(6, (8,), "elu", 1.0)
    z_uc = dkuc.lift(state)
    assert dkuc.step(z_uc, control).shape == (3, 10)

    dkac = DKACNetwork(6, (8,), (8,), 2, "elu", 1.0, True)
    z_ac = dkac.lift(state)
    assert dkac.step(z_ac, control).shape == (3, 10)

    dkn = DKNNetwork(6, (8,), (8,), 2, "elu", True)
    z_n = dkn.encode(state)
    assert dkn.koopman_step(z_n, control).shape == (3, 10)
