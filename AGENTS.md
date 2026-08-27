# DK_CDSM Agent Workflow

This file defines the working rules for every agent operating in this
repository. Read it before inspecting, modifying, testing, or running the
project.

These rules apply to the repository root and all subdirectories. A nested
`AGENTS.md` may add stricter local rules, but it must not weaken this file.

## 1. Required Python Environment

The project uses the Conda environment named `env_dk_cdsm`. Select the
configured executable for the current operating system:

```text
Windows:
PYTHON_ENV_PATH=D:\Apps\Anaconda3\envs\env_dk_cdsm
PYTHON_EXE=D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe

Linux:
PYTHON_ENV_PATH=/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm
PYTHON_EXE=/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

All Python scripts, modules, tests, and package operations must use the
platform-appropriate `PYTHON_EXE`. Do not use bare `python`, `pip`, or
`pytest`, and do not silently fall back to another Conda environment.

Before the first Python command in a task, verify the interpreter.

Windows PowerShell:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -c "import sys; print(sys.executable)"
```

Linux shell:

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -c "import platform, sys; print(sys.executable); print(platform.system())"
```

The Linux host may have ROS 2 in `PYTHONPATH` with a different Python minor
version. Prefix project Python commands with `env -u PYTHONPATH` so packages
under `/opt/ros` do not leak into this Conda environment. Add a task-specific
`PYTHONPATH` only after clearing the inherited value when it is genuinely
required.

The output must resolve to the configured executable and current operating
system. If that environment is missing or broken, stop and report the problem
instead of switching environments.

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

Use the repository's five-directory research workflow boundaries:

```text
traj_data/       Independently runnable data-collection programs
prediction/      Independently runnable prediction/model-identification methods
control/         Independently runnable control methods
visualization/   Independently runnable plotting and rendering programs
common/          Shared runtime utilities and required static resources
tests/           Automated unit and integration tests
docs/            Plans, formula mappings, execution reports, and reviews
legacy_system/   Historical programs retained for reproducibility
*/outputs/       Stage-local generated artifacts; never application source
```

More specifically:

- `prediction/` owns prediction models, online identification algorithms,
  their configuration, evaluation, and runnable entry points.
- Keep one primary runnable Python file per prediction method:

  ```text
  prediction/edmd_prediction.py
  prediction/dkuc_prediction.py
  prediction/dkac_prediction.py
  prediction/dkn_prediction.py
  prediction/dktv_prediction.py      Hao et al.: accumulative DKTV update
  prediction/otvdkl_prediction.py    Zhang et al.: sliding-window OTVDKL update
  ```

- DKTV and OTVDKL are two different prediction methods. Do not use `DKTV` as
  an umbrella name for both, and do not merge their experiment contracts or
  outputs.
- DKTV and OTVDKL should reuse the existing DKUC model architecture and stable
  loading/lifting/normalization APIs. Do not duplicate a second DKUC training
  implementation inside either method.
- `control/` owns control laws and closed-loop experiments such as LQR, MPC,
  and KILC. Prediction-model online updates do not belong in `control/`.
  Add `control/otvdkl_control.py` only when Zhang et al.'s actual
  stability-guaranteed controller is implemented and independently testable.
- `traj_data/` owns CDSM collection programs and data-generation helpers.
- `common/` owns code shared by more than one main workflow, including cable
  allocation, artifact adapters, metrics, references, and static XML assets.
- `visualization/` owns result plotting, report figures, and MuJoCo rendering.
- `legacy_system/` is compatibility and historical evidence. Do not build new
  reusable features there unless an existing archived workflow must be repaired.
- Keep method-specific helpers in the corresponding method file when
  practical. Move code to `common/` only when multiple independent methods
  genuinely share the same contract.
- Do not introduce a parallel `src/`, `experiments/`, or `configs/` hierarchy.
  The five main directories are the project architecture.
- Do not add a combined `time_varying_comparison.py` entry before DKTV and
  OTVDKL can each run and save results independently. Paper comparisons should
  consume the saved artifacts from the two independent runs.

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

Run experiment entries by file path from the repository root. Each main entry
must expose `--help` and run without changing into its subdirectory.

Windows PowerShell:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\prediction\dkuc_prediction.py --help
```

Linux shell:

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  prediction/dkuc_prediction.py --help
```

Do not add a wrapper under another hierarchy merely to launch an existing
entry. A method may additionally support `python -m prediction.<method>`, but
the file-path command is the required independent execution contract.

Keep source and configuration paths portable: use `pathlib.Path` in Python,
resolve project resources relative to the repository or module, and do not
persist machine-specific Windows or Linux absolute paths in stable configs or
manifests. Do not construct paths by concatenating `\` or `/`. Existing
PowerShell and batch entry points may remain, but new Python entry points must
also run from a Linux shell.

## 5. Experiment Reproducibility

Every meaningful experiment must preserve enough information to reproduce it:

- exact entry script;
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

Each independently runnable stage owns its generated artifacts locally. Do not
create a second root-level output workflow for new experiments.

Use these stage-local roots:

```text
traj_data/outputs/         Collected and validated datasets
prediction/outputs/        Trained models and prediction evidence
control/outputs/           Closed-loop arrays, metrics, and figures
visualization/outputs/     Rendered media and presentation figures
```

Use `smoke_test/` and `full_run/` below each stage root, and keep method names
below the run type where multiple methods share the same stage. New generated
artifacts must not be committed or pushed to Git. Existing tracked historical
artifacts are user-owned and must not be deleted or rewritten unless requested.

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
  any stage's `outputs/` directory.
- Do not delete datasets, models, or results unless the user explicitly asks.
- Before accepting a dataset, verify finite values, state range, saturation,
  and cable-tension outliers.

## 7. Testing and Verification

Use the configured interpreter.

Windows PowerShell:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -m pytest
```

Linux shell:

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python -m pytest
```

Minimum expectations:

- Python edit: run `py_compile` or an import check.
- CLI edit: run `--help` and a focused smoke test when practical.
- Reusable algorithm edit: run the related unit tests.
- Pipeline or artifact-contract edit: run an integration test or a small
  end-to-end workflow.
- Visualization edit: verify that generated files open and inspect at least
  one representative image or animation frame.

For non-interactive Linux runs, use `MPLBACKEND=Agg` for Matplotlib. If the
agent sandbox cannot write the user's Matplotlib configuration directory, set
`MPLCONFIGDIR` to a task-specific directory under `/tmp`. Use `MUJOCO_GL=egl`
for headless MuJoCo rendering when EGL is available, and report a skipped
render check rather than replacing the configured Python environment if the
host has no compatible graphics backend.

If a test cannot be run, state the reason and the remaining risk.

## 8. Dependency Policy

Dependencies may be changed only in the configured environment.

Windows PowerShell:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -m pip install <package>
```

Linux shell:

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m pip install <package>
```

Install the locked Linux dependencies with:

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

The additional index is required for the locked
`torch==2.5.1+cu121` wheel. After installation, run `-m pip check` with the
same interpreter and Linux `PYTHONPATH` isolation.

After installing, upgrading, or removing a package:

1. Update the root `requirements.txt`.
2. Update the root `pyproject.toml` when project metadata or declared
   dependencies are affected.
3. Verify the package through `PYTHON_EXE`.
4. Report the dependency change.

Do not modify global Python, system Python, or user-site packages.

## 9. Git Rules

- Do not add new generated artifacts from stage-local `outputs/` directories.
  Add appropriate root `.gitignore` rules when a new output path is introduced;
  preserve existing tracked historical artifacts unless the user requests a
  separate cleanup.
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
- commands or experiment entry scripts run;
- tests and checks performed;
- key metrics, when applicable;
- exact output locations;
- failures, skipped checks, or residual risks.

Keep the report concise, but include enough concrete paths and values for the
next agent to continue without reconstructing the work.
