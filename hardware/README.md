# Hardware Workflow

This folder contains the XL330 hardware-facing entry points used to validate
communication, mirror encoder angles into MuJoCo, and run simple two-servo
tracking tests.

## Layout

```text
hardware/
  assets/      MuJoCo XML assets used by the hardware scripts.
  common/      Shared XL330, DYNAMIXEL SDK, and MuJoCo helper code.
  docs/        Hardware notes and parameter documentation.
  scripts/     Runnable hardware entry points.
  outputs/     Generated experiment artifacts; ignored by Git.
```

## Runnable Scripts

Run scripts from the repository root with the configured project interpreter:

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' hardware\scripts\<script>.py --help
```

| Script | Main purpose |
|---|---|
| `hardware/scripts/check_two_xl330_servos.py` | Ping IDs 10 and 20, read health/status registers, then run a conservative one-revolution sequential motion in Extended Position Mode. |
| `hardware/scripts/mirror_two_servos_to_mujoco.py` | Disable torque, let the user hand-rotate the servos, and mirror encoder angles into the two-joint MuJoCo model. |
| `hardware/scripts/track_sine_position_mode.py` | Run sine tracking through XL330 Extended Position Mode by writing `Goal Position(116)`. |
| `hardware/scripts/track_sine_current_mode.py` | Run Mode 0 current-command external angle feedback by writing `Goal Current(102)` and reading `Present Position(132)`. |

## Shared Code

`hardware/common/xl330_mujoco.py` owns the reusable hardware layer:

- XL330 / DYNAMIXEL control-table constants.
- SDK lazy loading, packet checks, register read/write helpers.
- sync-read and sync-write helpers.
- encoder tick to joint-angle conversion.
- MuJoCo joint-index, limit, camera, and qpos bridge helpers.
- common servo initialization and health checks.

The scripts should keep experiment-specific control logic locally and import
shared communication, conversion, and MuJoCo functions from `hardware.common`.

## Generated Outputs

`hardware/outputs/` stores local experiment logs, metrics, arrays, and figures.
It is intentionally ignored by Git. Do not move source code or stable
configuration into `hardware/outputs/`.
