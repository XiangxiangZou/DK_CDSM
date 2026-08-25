"""Time-varying CDSM collection and quality contracts for DKTV Plan 01."""

from __future__ import annotations

from typing import Any

import numpy as np

from koopman_control.dktv.config import stage_bounds
from traj_data.mujoco_cdsm import CABLE_NAMES, MujocoCDSM
from traj_data.references import (
    random_joint_reference,
    sample_initial_state,
    shrink_limits,
    soft_limit_guard,
)


REQUIRED_DATA_FIELDS = (
    "t",
    "states",
    "commanded_torque",
    "applied_torque",
    "commanded_tensions",
    "effective_tensions",
    "allocation_residual",
    "disturbance_torque",
    "reference_state",
    "saturation_flags",
    "joint_limit_flags",
)


def sine_disturbance(config: dict[str, Any], time_s: float, scale: float) -> np.ndarray:
    """Evaluate ``scale * a * sin(omega*t + phase)`` for both joints."""
    disturbance = config["disturbance"]
    amplitude = np.asarray(disturbance["amplitude_n_m"], dtype=np.float64)
    omega = np.asarray(disturbance["angular_frequency_rad_s"], dtype=np.float64)
    phase = np.asarray(disturbance["phase_rad"], dtype=np.float64)
    return float(scale) * amplitude * np.sin(omega * float(time_s) + phase)


def _scale_by_step(config: dict[str, Any], steps: int) -> tuple[np.ndarray, np.ndarray]:
    scales = np.zeros(steps, dtype=np.float64)
    stage_ids = np.zeros(steps, dtype=np.int64)
    for stage_id, stage in enumerate(stage_bounds(config, steps)):
        scales[stage["start_step"] : stage["end_step"]] = float(stage["scale"])
        stage_ids[stage["start_step"] : stage["end_step"]] = stage_id
    return scales, stage_ids


def collect_time_varying_pd(
    config: dict[str, Any],
    xml_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect controlled trajectories with an unobserved sine joint torque."""
    profile = config["profile"]
    collection = config["collection"]
    seed = int(config["seed"])
    dt = float(config["sample_dt_s"])
    trajectory_count = int(profile["trajectory_count"])
    steps = int(profile["steps"])
    rng = np.random.default_rng(seed)
    robot = MujocoCDSM(xml_path, dt)
    safe_limits = shrink_limits(robot.q_limits, float(collection["safe_limit_ratio"]))
    scales, stage_ids = _scale_by_step(config, steps)

    states = np.zeros((trajectory_count, steps + 1, 4), dtype=np.float64)
    commanded_torque = np.zeros((trajectory_count, steps, 2), dtype=np.float64)
    applied_torque = np.zeros_like(commanded_torque)
    commanded_tensions = np.zeros((trajectory_count, steps, len(CABLE_NAMES)), dtype=np.float64)
    effective_tensions = np.zeros_like(commanded_tensions)
    allocation_residual = np.zeros_like(commanded_torque)
    disturbance_torque = np.zeros_like(commanded_torque)
    reference_state = np.zeros_like(states)
    saturation_flags = np.zeros((trajectory_count, steps, 10), dtype=bool)
    joint_limit_flags = np.zeros((trajectory_count, steps + 1, 2), dtype=bool)
    time_axis = np.broadcast_to(
        np.arange(steps + 1, dtype=np.float64) * dt,
        (trajectory_count, steps + 1),
    ).copy()

    kp = np.asarray(collection["kp"], dtype=np.float64)
    kd = np.asarray(collection["kd"], dtype=np.float64)
    torque_limit = float(collection["torque_limit_n_m"])
    tension_lower = float(collection["tension_lower_n"])
    tension_upper = float(collection["tension_upper_n"])

    for trajectory in range(trajectory_count):
        q0, dq0 = sample_initial_state(
            rng,
            safe_limits,
            q_init_ratio=float(collection["q_init_ratio"]),
            dq_init_range=float(collection["dq_init_range"]),
        )
        robot.set_state(q0, dq0)
        states[trajectory, 0] = robot.read_state()
        q_ref, dq_ref, _ = random_joint_reference(
            rng,
            safe_limits,
            steps=steps,
            dt=dt,
            q_start=q0,
            waypoint_count=int(collection["reference_waypoints"]),
            range_ratio=float(collection["reference_range_ratio"]),
        )
        reference_state[trajectory, :-1, :2] = q_ref
        reference_state[trajectory, :-1, 2:] = dq_ref
        reference_state[trajectory, -1] = reference_state[trajectory, -2]

        for step in range(steps):
            state = robot.read_state()
            q, dq = state[:2], state[2:]
            raw_torque = kp * (q_ref[step] - q) + kd * (dq_ref[step] - dq)
            raw_torque += soft_limit_guard(
                q,
                dq,
                safe_limits,
                kp=float(collection["guard_kp"]),
                kd=float(collection["guard_kd"]),
            )
            torque_saturated = np.abs(raw_torque) > torque_limit
            torque_command = np.clip(raw_torque, -torque_limit, torque_limit)

            tension_command = robot.torque_to_tensions(torque_command, f_min=tension_lower)
            tension_effective = np.clip(tension_command, tension_lower, tension_upper)
            tension_saturated = np.abs(tension_effective - tension_command) > 1e-10
            cable_torque = robot.equivalent_joint_torque(tension_effective)
            disturbance = sine_disturbance(config, step * dt, scales[step])

            robot.apply_cable_tensions(tension_effective)
            robot.apply_joint_disturbance(disturbance)
            robot.step()

            next_state = robot.read_state()
            states[trajectory, step + 1] = next_state
            commanded_torque[trajectory, step] = torque_command
            applied_torque[trajectory, step] = cable_torque
            commanded_tensions[trajectory, step] = tension_command
            effective_tensions[trajectory, step] = tension_effective
            allocation_residual[trajectory, step] = torque_command - cable_torque
            disturbance_torque[trajectory, step] = disturbance
            saturation_flags[trajectory, step, :2] = torque_saturated
            saturation_flags[trajectory, step, 2:] = tension_saturated
            joint_limit_flags[trajectory, step + 1] = (
                (next_state[:2] < robot.q_limits[:, 0])
                | (next_state[:2] > robot.q_limits[:, 1])
            )

    arrays = {
        "t": time_axis,
        "states": states,
        "commanded_torque": commanded_torque,
        "applied_torque": applied_torque,
        "commanded_tensions": commanded_tensions,
        "effective_tensions": effective_tensions,
        "allocation_residual": allocation_residual,
        "disturbance_torque": disturbance_torque,
        "reference_state": reference_state,
        "saturation_flags": saturation_flags,
        "joint_limit_flags": joint_limit_flags,
        "stage_id": np.broadcast_to(stage_ids, (trajectory_count, steps)).copy(),
        # Compatibility aliases consumed by the existing DKUC and plotting APIs.
        "inputs": applied_torque,
        "q_ref": reference_state[:, :-1, :2],
        "dq_ref": reference_state[:, :-1, 2:],
        "cable_ctrl": effective_tensions,
    }
    metadata = {
        "mode": "controlled_pd_time_varying",
        "seed": seed,
        "dt": dt,
        "state_order": ["qa", "qb", "dqa", "dqb"],
        "identification_input": "applied_torque",
        "input_order": ["tau_a", "tau_b"],
        "cable_order": list(CABLE_NAMES),
        "xml_joint_limits_rad": robot.q_limits.tolist(),
        "safe_joint_limits_rad": safe_limits.tolist(),
        "saturation_flag_order": [
            "torque_a",
            "torque_b",
            *[f"tension_{name}" for name in CABLE_NAMES],
        ],
        "stages": stage_bounds(config, steps),
    }
    return arrays, metadata


def assess_data_quality(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the Plan 01 finite/range/saturation/residual acceptance checks."""
    missing = [field for field in REQUIRED_DATA_FIELDS if field not in arrays]
    finite_by_field = {
        field: bool(np.all(np.isfinite(np.asarray(arrays[field]))))
        for field in REQUIRED_DATA_FIELDS
        if field in arrays
    }
    base: dict[str, Any] = {
        "accepted": False,
        "rejection_reasons": [],
        "missing_fields": missing,
        "shape_errors": [],
        "finite_by_field": finite_by_field,
        "trajectory_count": None,
        "steps": None,
        "state_min": None,
        "state_max": None,
        "input_min": None,
        "input_max": None,
        "peak_abs_commanded_torque_n_m": None,
        "peak_abs_applied_torque_n_m": None,
        "peak_effective_tension_n": None,
        "saturation_count": None,
        "joint_limit_count": None,
        "tension_outlier_count": None,
        "max_abs_allocation_residual_n_m": None,
        "xml_joint_limits_rad": metadata.get("xml_joint_limits_rad"),
    }
    if missing:
        base["rejection_reasons"] = [f"missing_fields:{','.join(missing)}"]
        return base

    states = np.asarray(arrays["states"], dtype=np.float64)
    applied = np.asarray(arrays["applied_torque"], dtype=np.float64)
    shape_errors: list[str] = []
    if states.ndim != 3 or states.shape[-1:] != (4,):
        shape_errors.append(f"states:{list(states.shape)} expected [N,S+1,4]")
    if applied.ndim != 3 or applied.shape[-1:] != (2,):
        shape_errors.append(f"applied_torque:{list(applied.shape)} expected [N,S,2]")
    if shape_errors:
        base["shape_errors"] = shape_errors
        base["rejection_reasons"] = ["shape_errors"]
        return base

    trajectory_count = int(states.shape[0])
    steps = int(applied.shape[1])
    expected_shapes = {
        "t": (trajectory_count, steps + 1),
        "states": (trajectory_count, steps + 1, 4),
        "commanded_torque": (trajectory_count, steps, 2),
        "applied_torque": (trajectory_count, steps, 2),
        "commanded_tensions": (trajectory_count, steps, len(CABLE_NAMES)),
        "effective_tensions": (trajectory_count, steps, len(CABLE_NAMES)),
        "allocation_residual": (trajectory_count, steps, 2),
        "disturbance_torque": (trajectory_count, steps, 2),
        "reference_state": (trajectory_count, steps + 1, 4),
        "saturation_flags": (trajectory_count, steps, 2 + len(CABLE_NAMES)),
        "joint_limit_flags": (trajectory_count, steps + 1, 2),
    }
    for field, expected in expected_shapes.items():
        actual = np.asarray(arrays[field]).shape
        if actual != expected:
            shape_errors.append(f"{field}:{list(actual)} expected {list(expected)}")
    if shape_errors:
        base.update(
            {
                "shape_errors": shape_errors,
                "rejection_reasons": ["shape_errors"],
                "trajectory_count": trajectory_count,
                "steps": steps,
            }
        )
        return base

    tensions = np.asarray(arrays["effective_tensions"], dtype=np.float64)
    residual = np.asarray(arrays["allocation_residual"], dtype=np.float64)
    saturation_count = int(np.count_nonzero(arrays["saturation_flags"]))
    joint_limit_count = int(np.count_nonzero(arrays["joint_limit_flags"]))
    residual_max = float(np.max(np.abs(residual))) if np.all(np.isfinite(residual)) else None
    tension_outlier = float(config["quality"]["tension_outlier_n"])
    tension_outlier_count = int(np.count_nonzero(tensions > tension_outlier))

    def finite_extrema(values: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
        if not np.all(np.isfinite(values)):
            return None, None
        return np.min(values, axis=(0, 1)).tolist(), np.max(values, axis=(0, 1)).tolist()

    state_min, state_max = finite_extrema(states)
    input_min, input_max = finite_extrema(applied)

    def finite_peak(values: np.ndarray, *, absolute: bool = False) -> float | None:
        if not np.all(np.isfinite(values)):
            return None
        data = np.abs(values) if absolute else values
        return float(np.max(data))

    reasons: list[str] = []
    if missing:
        reasons.append(f"missing_fields:{','.join(missing)}")
    if config["quality"]["reject_nonfinite"] and not all(finite_by_field.values()):
        reasons.append("nonfinite_required_field")
    if config["quality"]["reject_joint_limit"] and joint_limit_count:
        reasons.append(f"joint_limit_flags:{joint_limit_count}")
    if config["quality"]["reject_saturation"] and saturation_count:
        reasons.append(f"saturation_flags:{saturation_count}")
    if tension_outlier_count:
        reasons.append(f"tension_outliers:{tension_outlier_count}")
    tolerance = float(config["quality"]["allocation_residual_tolerance_n_m"])
    if residual_max is not None and residual_max > tolerance:
        reasons.append(f"allocation_residual:{residual_max:.9g}")

    base.update(
        {
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "trajectory_count": trajectory_count,
            "steps": steps,
            "state_min": state_min,
            "state_max": state_max,
            "input_min": input_min,
            "input_max": input_max,
            "peak_abs_commanded_torque_n_m": finite_peak(
                np.asarray(arrays["commanded_torque"], dtype=np.float64), absolute=True
            ),
            "peak_abs_applied_torque_n_m": finite_peak(applied, absolute=True),
            "peak_effective_tension_n": finite_peak(tensions),
            "saturation_count": saturation_count,
            "joint_limit_count": joint_limit_count,
            "tension_outlier_count": tension_outlier_count,
            "max_abs_allocation_residual_n_m": residual_max,
        }
    )
    return base


def split_nominal_training_stream(
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Deterministically split trajectories and isolate nominal training prefixes."""
    trajectory_count = int(arrays["states"].shape[0])
    steps = int(arrays["applied_torque"].shape[1])
    rng = np.random.default_rng(int(config["seed"]))
    order = rng.permutation(trajectory_count)
    train_count = min(
        max(1, int(np.floor(float(config["training"]["train_fraction"]) * trajectory_count))),
        trajectory_count - 1,
    )
    train_indices = np.sort(order[:train_count])
    validation_indices = np.sort(order[train_count:])
    nominal_stop = int(stage_bounds(config, steps)[0]["end_step"])

    def subset(indices: np.ndarray, stop: int | None) -> dict[str, np.ndarray]:
        state_stop = None if stop is None else stop + 1
        input_stop = stop
        result = {
            "states": np.asarray(arrays["states"])[indices, :state_stop].copy(),
            "inputs": np.asarray(arrays["applied_torque"])[indices, :input_stop].copy(),
            "applied_torque": np.asarray(arrays["applied_torque"])[indices, :input_stop].copy(),
            "t": np.asarray(arrays["t"])[indices, :state_stop].copy(),
            "disturbance_torque": np.asarray(arrays["disturbance_torque"])[indices, :input_stop].copy(),
        }
        return result

    train = subset(train_indices, nominal_stop)
    validation_nominal = subset(validation_indices, nominal_stop)
    validation_stream = subset(validation_indices, None)
    split = {
        "seed": int(config["seed"]),
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "nominal_stop_step": nominal_stop,
        "nominal_sample_count": int(train["inputs"].shape[0] * train["inputs"].shape[1]),
    }
    return train, validation_nominal, validation_stream, split


def prove_time_variation(config: dict[str, Any], xml_path: str) -> dict[str, Any]:
    """Compare the same state/input at different absolute times."""
    robot = MujocoCDSM(xml_path, float(config["sample_dt_s"]))
    state = np.array([0.1, -0.1, 0.05, -0.03], dtype=np.float64)
    torque = np.array([2.0, -1.5], dtype=np.float64)
    tensions: np.ndarray | None = None

    def next_state(time_s: float, disturbed: bool) -> tuple[np.ndarray, np.ndarray]:
        nonlocal tensions
        robot.set_state(state[:2], state[2:])
        if tensions is None:
            tensions = robot.torque_to_tensions(torque)
        disturbance = sine_disturbance(config, time_s, 1.0) if disturbed else np.zeros(2)
        robot.apply_cable_tensions(tensions)
        robot.apply_joint_disturbance(disturbance)
        robot.step()
        return robot.read_state(), disturbance

    time_a, time_b = 0.11, 0.37
    fixed_a, _ = next_state(time_a, False)
    fixed_b, _ = next_state(time_b, False)
    varying_a, disturbance_a = next_state(time_a, True)
    varying_b, disturbance_b = next_state(time_b, True)
    fixed_difference = float(np.linalg.norm(fixed_a - fixed_b))
    varying_difference = float(np.linalg.norm(varying_a - varying_b))
    return {
        "same_state": state.tolist(),
        "same_commanded_torque": torque.tolist(),
        "absolute_times_s": [time_a, time_b],
        "disturbance_torque_at_times": [disturbance_a.tolist(), disturbance_b.tolist()],
        "fixed_next_states": [fixed_a.tolist(), fixed_b.tolist()],
        "time_varying_next_states": [varying_a.tolist(), varying_b.tolist()],
        "fixed_next_state_difference_norm": fixed_difference,
        "time_varying_next_state_difference_norm": varying_difference,
        "passed": fixed_difference <= 1e-12 and varying_difference > 1e-8,
    }
