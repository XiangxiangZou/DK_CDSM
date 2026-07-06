"""Run the CDSM kinematic planning and IK diagnostic."""

import sys

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    if any(value in {"-h", "--help"} for value in sys.argv[1:]):
        print(
            "Run the legacy interactive CDSM kinematic planning experiment. "
            "This entry has no configurable argparse options."
        )
        raise SystemExit(0)
    run_project_script(
        "archive/diagnostics/mujoco_cdsm_kinematic_planning.py"
    )
