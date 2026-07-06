"""CDSM-specific names, dimensions, and asset paths."""

from __future__ import annotations

from pathlib import Path

from cable_robotics.tension_allocator import AntagonisticLayout

ACTIVE_JOINTS = ("joint1", "joint3")
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}

CABLE_NAMES = (
    "cable11",
    "cable12",
    "cable13",
    "cable14",
    "cable21",
    "cable22",
    "cable23",
    "cable24",
)
ACTUATOR_NAMES = tuple(
    "winch_c" + name[len("cable") :]
    for name in CABLE_NAMES
)

STATE_DIM = 4
CONTROL_DIM = 2

DEFAULT_XML_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "multi_joint_cable_driven_space_robot.xml"
)


def make_tension_layout(
    dof_j1: int,
    dof_j2: int,
    dof_j3: int,
    dof_j4: int,
) -> AntagonisticLayout:
    """Build the current two-joint, eight-cable allocation layout."""
    return AntagonisticLayout(
        cable_count=len(CABLE_NAMES),
        positive_groups=((0, 2), (4, 6)),
        negative_groups=((1, 3), (5, 7)),
        dof_groups=((dof_j1, dof_j2), (dof_j3, dof_j4)),
    )
