"""Evaluate a saved early Deep Koopman model on new trajectories."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script("archive/diagnostics/test_deep_koopman_cdsm.py")
