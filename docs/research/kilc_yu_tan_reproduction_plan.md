# Yu & Tan-Style DKUC + KILC Reproduction Plan

## Repository Naming

Yu & Tan call their learned Koopman model a DKN, but in this repository that
structure belongs to the DKUC family: learned state lifting with direct
physical control input. The repository's existing DKN remains the
prediction-only nonlinear state-control encoder and is not used for KILC.

## Implemented Line

The KILC line is now:

1. Train a continuous-time DKUC model:
   `z = [x_norm, phi_theta(x_norm)]`, `zdot = A_c z + B_c u_norm`.
2. Generate a 20 s Cartesian circle reference.
3. Convert the circle to joint references through the existing MuJoCo IK.
4. Run repeated-trial lifted-state KILC with:
   `e_z = z_ref - z_meas`.
5. Record the decomposed control terms:
   `u_ilc`, `u_adaptive`, `u_robust`, and `u_total`.

## Entry Points

- Train continuous DKUC:
  `experiments.deployment_pipeline.train_dkuc_continuous`
- Run KILC on CDSM:
  `experiments.deployment_pipeline.run_kilc`

## Outputs

KILC results are written under:

`outputs/results/deployment_pipeline/kilc/cartesian_circle/<timestamp>/`

Each run stores `manifest.json`, `metrics/`, `arrays/`, `figures/`, and `logs/`.
