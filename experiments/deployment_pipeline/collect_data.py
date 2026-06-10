"""Collect a unified CDSM dataset in MuJoCo."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script("archive/deployment_pipeline/run_01_collect_data.py")
