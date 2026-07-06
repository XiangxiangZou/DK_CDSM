"""Train and evaluate the nominal-model plus EDMD residual model."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/hybrid_models/cdsm_hybrid_residual_edmd.py"
    )
