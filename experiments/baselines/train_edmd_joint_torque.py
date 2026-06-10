"""Run the joint-torque EDMD identification baseline."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/baselines/edmd_mujoco_cdsm_joint_torque.py"
    )
