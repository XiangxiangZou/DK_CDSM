"""Compare DKN and EDMD prediction performance."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/model_comparison/cdsm_dkn_vs_edmd_prediction_compare.py"
    )
