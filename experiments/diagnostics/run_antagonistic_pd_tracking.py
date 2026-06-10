"""Run the direct antagonistic-cable PD tracking experiment."""

import sys

from experiments._bootstrap import run_project_script

if __name__ == "__main__":
    if any(value in {"-h", "--help"} for value in sys.argv[1:]):
        print(
            "Run the legacy interactive antagonistic-cable PD tracking "
            "experiment. This entry has no configurable argparse options."
        )
        raise SystemExit(0)
    run_project_script(
        "archive/diagnostics/mujoco_cdsm_antagonistic_pd_tracking.py"
    )
