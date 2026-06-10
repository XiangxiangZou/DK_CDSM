# DK_CDSM Agent Workflow

This file defines the working rules for every agent operating in this
repository. Read it before inspecting, modifying, testing, or running the
project.

These rules apply to the repository root and all subdirectories. A nested
`AGENTS.md` may add stricter local rules, but it must not weaken this file.

## 1. Required Python Environment

The project Python environment is:

```text
PYTHON_ENV_PATH=D:\Apps\Anaconda3\envs\env_dk_cdsm
PYTHON_EXE=D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe
```

All Python scripts, modules, tests, and package operations must use
`PYTHON_EXE`. Do not use bare `python`, `pip`, or `pytest`.

Before the first Python command in a task, verify the interpreter:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -c "import sys; print(sys.executable)"
```

The output must resolve to the configured executable. If the environment is
missing or broken, stop and report the problem instead of silently switching
to another environment.

## 2. Start-of-Task Checklist

Every agent must:

1. Read this file.
2. Run `git status --short` and preserve existing user changes.
3. Inspect the relevant source, configuration, tests, and recent result
   metadata before deciding how to implement a change.
4. Prefer an existing project entry point or reusable API over adding a
   parallel implementation.
5. State the intended execution or edit scope before changing files.

Do not revert, overwrite, move, or delete unrelated files. Treat unexpected
working-tree changes as user-owned unless proven otherwise.

## 3. Repository Ownership

Use the following boundaries:

```text
src/            Reusable implementation code
experiments/    CLI entry points and experiment composition
configs/        Stable, reviewable defaults
tests/          Automated unit and integration tests
assets/         Required XML models and static project resources
archive/        Historical programs retained for reproducibility
outputs/        Generated local artifacts; never application source
```

More specifically:

- `src/koopman_control/` owns reusable learning, model, evaluation, and
  control algorithms.
- `src/cable_robotics/` owns generic cable allocation, safety, interfaces,
  and metrics.
- `src/cdsm/` owns CDSM-specific MuJoCo plants, kinematics, references,
  collection, and runtime behavior.
- `experiments/` may select parameters and compose workflows, but reusable
  algorithms must not be implemented there.
- `archive/` is compatibility and historical evidence. Do not build new
  reusable features there unless an existing archived workflow must be
  repaired.

## 4. Standard Execution Workflow

For code or experiment tasks, follow this order:

1. Inspect the relevant entry point and implementation path.
2. Verify the configured Python interpreter.
3. Run the cheapest useful check, such as `py_compile`, import, `--help`, or a
   focused unit test.
4. Make the smallest change that follows the current architecture.
5. Run focused tests for the changed behavior.
6. Run broader tests when shared contracts or cross-module behavior changed.
7. Inspect generated metrics and artifacts rather than relying only on a zero
   exit code.
8. Report the exact command, important parameters, metrics, and output paths.

Run experiment entries as modules from the repository root:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -m experiments.deployment_pipeline.collect_data --help
```

Prefer `-m experiments...` over invoking experiment files by path so imports
remain stable.

## 5. Experiment Reproducibility

Every meaningful experiment must preserve enough information to reproduce it:

- exact entry module;
- command-line arguments or copied configuration;
- random seed;
- source dataset path and filtering rules;
- model artifact path;
- Python environment;
- Git branch and, when available, commit hash;
- numerical metrics;
- raw arrays needed to redraw figures;
- generated figure and animation paths.

Use deterministic seeds where supported. Do not describe a run as successful
until its metrics and saved artifacts have been checked.

For model assessment:

- keep one-step prediction and rollout prediction as separate evidence;
- use rollout behavior for long-horizon model claims;
- use closed-loop joint and Cartesian tracking metrics for control claims;
- do not treat prediction-only DKN results as linear-LQR control evidence;
- inspect joint limits, torque saturation, cable tensions, and non-finite
  values before accepting collected data or control results.

## 6. Output and Artifact Rules

All generated research artifacts must stay under `outputs/`. They must not be
committed or pushed to Git.

Use these categories:

```text
outputs/data/raw/          Original collected data
outputs/data/processed/    Filtered or transformed datasets
outputs/data/rejected/     Invalid runs retained for diagnosis
outputs/models/            Trained models and normalizers
outputs/results/           Metrics, arrays, figures, and animations
outputs/archive/           Superseded smoke tests or legacy outputs
```

For a result run, keep numerical evidence separate from display products:

```text
<run>/
  manifest.json
  metrics/
  arrays/
  figures/
  media/
  logs/
```

Rules:

- Save raw `.npz` or equivalent arrays when figures may need to be redrawn.
- Save metrics in JSON rather than only printing them.
- Treat PNG, PDF, SVG, and GIF files as reproducible presentation products.
- Do not duplicate the same figure across multiple output trees.
- Do not place source code, required XML assets, or stable configuration in
  `outputs/`.
- Do not delete datasets, models, or results unless the user explicitly asks.
- Before accepting a dataset, verify finite values, state range, saturation,
  and cable-tension outliers.

## 7. Testing and Verification

Use the configured interpreter:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -m pytest
```

Minimum expectations:

- Python edit: run `py_compile` or an import check.
- CLI edit: run `--help` and a focused smoke test when practical.
- Reusable algorithm edit: run the related unit tests.
- Pipeline or artifact-contract edit: run an integration test or a small
  end-to-end workflow.
- Visualization edit: verify that generated files open and inspect at least
  one representative image or animation frame.

If a test cannot be run, state the reason and the remaining risk.

## 8. Dependency Policy

Dependencies may be changed only in the configured environment:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -m pip install <package>
```

After installing, upgrading, or removing a package:

1. Update `configs/project/requirements.txt`.
2. Update `configs/project/pyproject.toml` when project metadata or declared
   dependencies are affected.
3. Verify the package through `PYTHON_EXE`.
4. Report the dependency change.

Do not modify global Python, system Python, or user-site packages.

## 9. Git Rules

- `outputs/`, caches, local environments, IDE state, and logs must remain
  ignored by the root `.gitignore`.
- Before finishing, run `git diff --check`.
- Review `git status --short` and distinguish task changes from pre-existing
  changes.
- Never use `git reset --hard`, destructive checkout, or broad cleanup
  commands unless the user explicitly requests them.
- Do not amend, commit, push, or open a pull request unless requested.
- Do not commit large generated binaries merely to preserve a result; keep
  reproducibility metadata and generation code instead.

## 10. Completion Report

At the end of a task, report:

- files changed;
- commands or experiment entry modules run;
- tests and checks performed;
- key metrics, when applicable;
- exact output locations;
- failures, skipped checks, or residual risks.

Keep the report concise, but include enough concrete paths and values for the
next agent to continue without reconstructing the work.
