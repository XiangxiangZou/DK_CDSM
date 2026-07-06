"""CDSM-specific extension of the generic cable-driven plant contract."""

from __future__ import annotations

from typing import Protocol, Tuple

from cable_robotics.interfaces import CableDrivenPlant


class CDSMPlant(CableDrivenPlant, Protocol):
    def torque_dofs(self) -> Tuple[int, int, int, int]:
        """Return joint1 through joint4 MuJoCo-compatible DOF indices."""
        ...
