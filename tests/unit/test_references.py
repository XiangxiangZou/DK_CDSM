import numpy as np

from cdsm.references.cartesian import (
    CartesianReferenceConfig,
    generate_cartesian_reference,
)
from cdsm.references.joint import build_joint_ramp_reference


def test_joint_reference_contains_state_reference() -> None:
    ref = build_joint_ramp_reference(
        dt=0.01,
        duration=0.1,
        q_start=np.array([0.0, 0.0]),
        q_target=np.array([0.2, -0.1]),
        ramp_duration=0.05,
    )
    assert ref["x_ref"].shape[1] == 4
    np.testing.assert_allclose(ref["q_ref"][-1], [0.2, -0.1])


def test_cartesian_reference_shapes() -> None:
    ref = generate_cartesian_reference(
        CartesianReferenceConfig(
            kind="circle",
            dt=0.05,
            period=1.0,
            num_cycles=1.0,
        )
    )
    assert ref["xy_ref"].shape[1] == 2
    assert ref["xy_ref"].shape == ref["dxy_ref"].shape
