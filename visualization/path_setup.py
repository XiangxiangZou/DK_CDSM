"""Path bootstrap for standalone visualization scripts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_PACKAGES_ROOT = PROJECT_ROOT / "common" / "packages"
COMMON_ASSETS_ROOT = PROJECT_ROOT / "common" / "assets"
DEFAULT_XML = COMMON_ASSETS_ROOT / "multi_joint_cable_driven_space_robot.xml"


def ensure_paths() -> None:
    """Make compact common packages importable from direct script execution."""
    for root in (PROJECT_ROOT, COMMON_PACKAGES_ROOT):
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


ensure_paths()
