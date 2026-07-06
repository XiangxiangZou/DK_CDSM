"""Compare DKUC and DKAC prediction and tracking."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/model_comparison/cdsm_dkuc_vs_dkac_tracking_control.py"
    )
