# Experiment Entry Points

This directory is the organized command-line layer for DK_CDSM research runs.
It does not contain reusable model or robot implementation code. Reusable code
belongs under `src/`; experiment entries only select parameters and launch a
workflow.

Run entries from the project root with the configured environment:

```powershell
& $env:PYTHON_EXE -m experiments.deployment_pipeline.collect_data --help
```

Use module execution (`-m`) rather than invoking an entry file by path. This
keeps the project root and `src/` import paths stable on Windows.

The deployment pipeline entries call reusable implementations in `src/`
directly. They do not execute scripts from `archive/`.

Some baseline, early control, model-comparison, and diagnostic entries remain
legacy adapters for historical reproducibility. New work must not add another
`archive/` dependency; migrate the reusable implementation into `src/` first.

## Deployment Pipeline

| Module | Purpose |
| --- | --- |
| `deployment_pipeline.collect_data` | Collect random or PD-controlled MuJoCo datasets |
| `deployment_pipeline.train_models` | Train EDMD, DKUC, DKAC, and DKN |
| `deployment_pipeline.validate_prediction` | Evaluate one-step and rollout prediction |
| `deployment_pipeline.compare_joint_tracking` | Compare joint-space closed-loop tracking |
| `deployment_pipeline.compare_cartesian_tracking` | Compare circle, square, and figure-eight tracking |
| `deployment_pipeline.render_animation` | Render one animation per control method |
| `deployment_pipeline.render_combined_animation` | Render a combined three-method animation |

## Model Comparison

| Module | Purpose |
| --- | --- |
| `model_comparison.compare_dkn_edmd_prediction` | DKN versus EDMD prediction |
| `model_comparison.compare_dkuc_dkac_tracking` | DKUC versus DKAC |
| `model_comparison.compare_dkac_edmd_tracking` | DKAC versus EDMD closed-loop tracking |
| `model_comparison.compare_deepkoopman_edmd` | Earlier Deep Koopman versus EDMD |
| `model_comparison.compare_model_construction` | Compare EDMD/DKUC/DKAC/DKN structure, complexity, and prediction quality |

## Control

| Module | Purpose |
| --- | --- |
| `control.run_full_deepkoopman_lqr_mpc` | Full-state Deep Koopman LQR/MPC |
| `control.run_latent_mpc_pipeline` | Latent-space Deep Koopman MPC |
| `control.compare_mpc_tracking` | Nominal and Koopman-assisted MPC comparison |
| `control.run_hybrid_smc_tracking` | Hybrid residual model with sliding-mode control |
| `control.report_dkac_circle_tracking` | Generate the requested DKAC circle tracking and actuator figures |

## Hybrid Models

| Module | Purpose |
| --- | --- |
| `hybrid_models.run_residual_edmd` | Nominal rigid model plus EDMD residual |
| `hybrid_models.run_residual_deepkoopman` | Nominal rigid model plus Deep Koopman residual |

## Baselines

| Module | Purpose |
| --- | --- |
| `baselines.train_lusch_deepkoopman` | Lusch-style autonomous Deep Koopman baseline |
| `baselines.train_edmd_joint_torque` | Joint-torque EDMD identification baseline |

## Diagnostics

| Module | Purpose |
| --- | --- |
| `diagnostics.audit_cable_torque_mapping` | Audit torque-to-tension mapping |
| `diagnostics.validate_tendon_jacobian` | Validate the tendon Jacobian |
| `diagnostics.run_antagonistic_pd_tracking` | Direct cable-driven PD tracking |
| `diagnostics.run_kinematic_planning` | Kinematic planning and IK |
| `diagnostics.validate_mujoco_model` | MuJoCo structure and actuation checks |
| `diagnostics.validate_deepkoopman_model` | Validate an early saved Deep Koopman model |

## Ownership Rules

- `experiments/` owns CLI entry points and experiment composition.
- `src/koopman_control/` owns reusable learning and control algorithms.
- `src/cable_robotics/` owns generic cable allocation and safety behavior.
- `src/cdsm/` owns this robot's MuJoCo, kinematics, references, and runtime.
- New reusable functions must not be implemented inside an experiment entry.
