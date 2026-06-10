"""Compare the earlier Deep Koopman implementation with EDMD."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/model_comparison/cdsm_koopman_vs_edmd_model_compare.py"
    )
