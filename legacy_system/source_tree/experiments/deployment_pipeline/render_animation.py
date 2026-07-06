"""Render individual MuJoCo tracking animations from saved logs."""

from experiments._paths import SRC_ROOT  # noqa: F401
from cdsm.visualization.mujoco_animation import main


if __name__ == "__main__":
    main()
