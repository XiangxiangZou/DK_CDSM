"""Render individual MuJoCo tracking animations from saved logs."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/deployment_pipeline/run_06_render_mujoco_animation.py"
    )
