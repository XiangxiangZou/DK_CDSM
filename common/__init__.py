"""Shared utility modules for data, prediction, and control scripts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_ROOT = Path(__file__).resolve().parent / "packages"

for root in (PACKAGES_ROOT,):
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

from common.prediction_utils import *  # noqa: F401,F403,E402
