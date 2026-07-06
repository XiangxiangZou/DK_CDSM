"""Run the Deep Koopman latent-space MPC pipeline."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/control/cdsm_deepkoopman_latent_mpc_pipeline.py"
    )
