"""Run the full Deep Koopman linear LQR/MPC experiment."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/control/cdsm_full_deepkoopman_lqr_mpc.py"
    )
