"""Validate the MuJoCo tendon Jacobian and torque mapping."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script("archive/diagnostics/mujoco_cdsm_jacobian.py")
