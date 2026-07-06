# Historical Implementations

This directory stores the original large single-file research programs. They
are retained for result reproduction and paper-traceability, not as the primary
location for reusable implementation.

- `model_comparison/`: pairwise Koopman comparison programs.
- `control/`: LQR, MPC, and robust-control programs.
- `hybrid_models/`: nominal model plus learned residual programs.
- `baselines/`: early Deep Koopman and EDMD baselines.
- `diagnostics/`: MuJoCo, cable mapping, IK, and interactive checks.
- `deployment_pipeline/`: unified collection, training, evaluation, and tracking workflow.
- `support/`: plotting and MuJoCo logging helpers required by historical programs.
- `legacy/`: superseded duplicates and dated backups.

Use the organized modules under `experiments/` to launch these programs. Small
compatibility imports remain in the repository root while the deployment
pipeline is migrated fully to `src/`.
