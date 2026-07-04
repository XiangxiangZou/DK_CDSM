"""Render a combined EDMD/DKUC/DKAC MuJoCo comparison animation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.path_setup import ensure_paths

ensure_paths()

from visualization.mujoco.combined_mujoco_animation import main


if __name__ == "__main__":
    main()
