"""Audit desired torque to cable tension to realized torque mapping."""

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    run_project_script(
        "archive/diagnostics/cdsm_cable_tau_mapping_audit.py"
    )
