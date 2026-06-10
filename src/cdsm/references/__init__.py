"""CDSM joint and Cartesian reference generators."""

from .cartesian import (
    CartesianReferenceConfig,
    generate_cartesian_reference,
)
from .joint import build_joint_ramp_reference

__all__ = [
    "CartesianReferenceConfig",
    "build_joint_ramp_reference",
    "generate_cartesian_reference",
]
