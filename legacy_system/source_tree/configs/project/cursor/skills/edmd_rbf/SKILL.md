---
name: edmd-rbf-mujoco-diagnostics
description: Diagnose and improve EDMD/RBF Koopman prediction failures for MuJoCo robot models, especially when joint trajectories diverge, one joint predicts worse than another, or EDMD/RBF models trained from simulated robot data fail in one-step or rollout prediction.
---

# EDMD/RBF MuJoCo Diagnostics

Use this skill to analyze why an EDMD/RBF Koopman model fails to match MuJoCo robot dynamics and to propose concrete improvements. Treat plots, rollout errors, one-step errors, state definitions, dictionaries, and data collection settings as primary evidence.

## Diagnostic Priority

Start with this order. Do not tune RBF size or model complexity before checking the earlier items.

1. Separate one-step prediction from open-loop rollout.
   - If one-step prediction is poor, the issue is state definition, data, dictionary, input modeling, or numerical conditioning.
   - If one-step is good but rollout diverges, the issue is error accumulation, unstable learned modes, distribution shift, or lack of closed-loop correction.

2. Verify the state is Markov.
   - For rigid joints, `x = [q, qdot]` may be sufficient.
   - For tendon/cable-driven robots, include hidden actuator states when available: tendon length, tendon velocity, motor angle, motor velocity, tension, elastic deformation, backlash/slack state, or actuator internal states.
   - For space robots or floating-base systems, include base pose/orientation, base velocity/angular velocity, or conserved momentum-related variables if the base is not fixed.

3. Check angle and orientation representation.
   - Prefer `sin(q), cos(q)` over raw joint angles for periodic coordinates.
   - Do not regress quaternions as ordinary unconstrained Euclidean coordinates without normalization or a proper orientation error representation.

4. Check training/test coverage.
   - Compare train vs test ranges for each state and input dimension.
   - A joint that fails alone often has worse coverage, stronger coupling, or hidden state dependence.
   - If the predicted trajectory leaves the training distribution, treat subsequent rollout error as extrapolation.

5. Check RBF scaling and centers.
   - Standardize `x` and `u` before computing RBF distances.
   - Choose centers from training data, commonly with k-means.
   - Sweep center counts and kernel widths; use median center distance as an initial width heuristic.
   - If one joint fails, inspect whether RBF centers cover that joint's range and velocity range.

6. Check numerical conditioning.
   - Inspect `cond(G)` or singular values of the regression matrix.
   - Avoid plain `pinv` as the default for noisy or high-dimensional dictionaries.
   - Use ridge/Tikhonov regularization and sweep `lambda`.

## Recommended Model Forms

Prefer a controlled Koopman form for robot prediction:

```text
z_k = phi(x_k)
z_{k+1} = A z_k + B u_k
x_{k+1} = C z_{k+1}
```

or a direct one-step regressor:

```text
x_{k+1} = W [phi(x_k); u_k]
```

Use `phi(x,u)` with full state-input products only when data volume is sufficient and the interaction terms are justified. Full Kronecker/RBF expansion over both state and input often becomes ill-conditioned for complex robots.

For angular robots, start with:

```text
phi base = [sin(q), cos(q), qdot]
```

Then add RBF features over the normalized base state:

```text
phi(x) = [1, normalized_state, rbf(normalized_state)]
```

For tendon/cable-driven robots, expand the observed state before expanding the dictionary:

```text
x = [sin(q), cos(q), qdot, tendon_length, tendon_velocity, tension, motor_state]
```

If these variables are not available and prediction remains poor in one-step tests, state that the observed system is likely non-Markov under the current state definition.

## Interpreting Joint-Specific Failure

When one joint fails and another partly works:

- Do not conclude EDMD is globally invalid immediately.
- Check whether the failed joint has stronger cable coupling, larger inertia, stronger base reaction, larger friction, limits, saturation, or hidden tendon dynamics.
- Check whether the failed joint's trajectory is outside the training distribution.
- Check whether its velocity prediction is better than its position prediction; this can indicate angle representation or integration/rollout drift issues.
- Check whether the failed joint is closer to the base or carries stronger dynamic coupling.

For a plot where `q_a` fails but `q_b` trends correctly, prioritize:

```text
1. q_a training/test coverage
2. q_a hidden tendon or actuator state dependence
3. sin/cos angle representation
4. one-step q_a error
5. RBF centers/width around q_a and qdot_a
6. ridge regularization and stable rollout constraints
```

## Practical Improvement Path

Recommend this minimum sequence:

1. Replot one-step predictions for every state dimension.
2. Standardize state and input data.
3. Replace raw joint angles with `sin/cos`.
4. Train a controlled Koopman model `z_next = A z + B u` rather than mixing `x` and `u` inside every RBF term.
5. Add ridge regularization.
6. Sweep RBF centers and widths using validation one-step error.
7. Add tendon/actuator/base states if one-step error remains large.
8. Only after one-step is acceptable, evaluate 10-step, 50-step, and 200-step rollout.

Use this MATLAB-style ridge pattern when explaining implementation:

```matlab
% Z: lifted/current features, Y: next-state or next-lifted targets
lambda = 1e-6;
W = (Y * Z') / (Z * Z' + lambda * eye(size(Z,1)));
```

Mention that `lambda` should be swept, for example:

```text
1e-8, 1e-6, 1e-4, 1e-2
```

## Response Style

Give a direct diagnosis from the evidence first. Then give ordered next steps. Avoid claiming the repository method should work unchanged on complex MuJoCo robots. State clearly when the likely issue is not EDMD syntax but one of:

- incomplete state observation
- unsuitable dictionary
- poor data coverage
- input modeling mismatch
- numerical conditioning
- rollout error accumulation
- contact, friction, slack, saturation, or other non-smooth dynamics
