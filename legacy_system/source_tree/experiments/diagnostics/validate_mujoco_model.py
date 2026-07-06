"""Run the standalone MuJoCo model structural checks."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script("archive/diagnostics/test_mujoco_CDSM.py")
