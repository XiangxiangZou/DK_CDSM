"""Render a combined EDMD/DKUC/DKAC MuJoCo comparison animation."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/deployment_pipeline/"
        "run_08_render_combined_mujoco_trajectory_gif.py"
    )
