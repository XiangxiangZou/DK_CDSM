"""Train EDMD, DKUC, DKAC, and DKN on a unified dataset."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script("archive/deployment_pipeline/run_02_train_all_models.py")
