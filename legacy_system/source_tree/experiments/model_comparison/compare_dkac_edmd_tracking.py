"""Compare DKAC and EDMD closed-loop tracking."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/model_comparison/cdsm_dkac_vs_edmd_tracking_control.py"
    )
