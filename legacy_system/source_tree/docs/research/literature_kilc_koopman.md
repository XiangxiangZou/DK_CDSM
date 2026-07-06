# Koopman ILC Literature Notes for CDSM

This note keeps the newly added Koopman/control references tied to concrete
repository work. It is not a full literature review.

## Primary Reproduction Target

- `Yu和Tan - 2026 - On the learning-based control of continuum robots with provable robustness, efficiency, and generali.pdf`
  - Role: main reference for the Koopman model plus iterative learning control
    direction.
  - Repository mapping: implemented as continuous-time DKUC plus lifted-state
    KILC. The paper's DKN name maps to this repository's DKUC family, not to the
    existing prediction-only `dkn` model.

## Closely Related Koopman Control References

- `Feizi 等 - 2025 - Deep Koopman Approach for Nonlinear Dynamics and Control of Tendon-Driven Continuum Robots.pdf`
  - Role: tendon-driven continuum robot Koopman control reference.
  - Repository mapping: useful when extending beyond direct-input DKUC/EDMD
    toward richer input/state coupling. Do not mix this with the first KILC
    baseline until the direct-input route is stable.

- `Bruder 等 - 2021 - Koopman-Based Control of a Soft Continuum Manipulator Under Variable Loading Conditions.pdf`
  - Role: variable-load Koopman modeling and observer-style robustness idea.
  - Repository mapping: later candidate for payload, cable-bias, or repeatable
    disturbance estimation in CDSM experiments.

- `Zhang 等 - 2026 - Learning Predictive Control with Deep Koopman Operators for Autonomous Vehicle Motion Planning.pdf`
  - Role: cross-domain Koopman predictive-control reference.
  - Repository mapping: secondary inspiration for receding-horizon learning
    control only; it should not drive the first Yu & Tan reproduction.

## Current Repository Line

The intended first line is deliberately narrow:

1. Train continuous-time DKUC with direct normalized physical control.
2. Run Cartesian-circle KILC through `experiments.deployment_pipeline.run_kilc`.
3. Save each meaningful run under
   `outputs/results/deployment_pipeline/kilc/cartesian_circle/<timestamp>/`.
4. Keep DKAC constrained MPC and repository DKN as separate research lines.
