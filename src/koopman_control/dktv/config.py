"""Load and validate the common DKTV experiment contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


EXPECTED_METHODS = {
    "baseline": "fixed_dko",
    "accumulative": "dktv_accumulative",
    "window": "otvdkl_window",
    "selective": "otvdkl_selective",
}
EXPECTED_STATE = ["qa", "qb", "dqa", "dqb"]
EXPECTED_HORIZONS = [10, 20, 50, 100]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_foundation_config(config: dict[str, Any]) -> None:
    """Validate fields shared by Plan 01 and all later online methods."""
    _require(config.get("methods") == EXPECTED_METHODS, "DKTV method identifiers are not frozen")
    _require(config.get("state") == EXPECTED_STATE, "state order must be qa,qb,dqa,dqb")
    _require(config.get("state_dim") == len(EXPECTED_STATE), "state_dim does not match state")
    _require(config.get("input") == "applied_torque", "identification input must be applied_torque")
    _require(config.get("input_dim") == 2, "input_dim must be 2")
    _require(float(config.get("sample_dt_s", 0.0)) > 0.0, "sample_dt_s must be positive")
    _require(config.get("lift") == ["x", "phi(x)", "1"], "affine lift must be [x,phi(x),1]")
    expected_lifted = int(config["state_dim"]) + int(config["encoder_output_dim"]) + 1
    _require(config.get("lifted_dim") == expected_lifted, "lifted_dim must include x, phi(x), and 1")
    _require(config.get("encoder_update") is False, "online encoder updates are outside Plan 01")
    _require(config.get("model_form") == "affine", "model_form must be affine")
    _require(config.get("state_readout") == "exact", "state_readout must be exact")
    coordinates = config.get("coordinates", {})
    _require(coordinates.get("state") == "normalized", "state coordinate must be normalized")
    _require(
        coordinates.get("input") == "normalized_applied_torque",
        "input coordinate must be normalized_applied_torque",
    )
    _require(
        coordinates.get("lift_definition") == "z = [x_normalized, phi(x_normalized), 1]",
        "normalized affine lift definition is not frozen",
    )
    _require(
        coordinates.get("readout") == "x_normalized = C0 @ z",
        "C0 must read normalized state",
    )
    _require(
        coordinates.get("normalizer_policy")
        == "fixed_and_shared_by_all_methods_and_direct_refit_oracle",
        "all methods and the direct-refit oracle must share fixed normalizers",
    )
    _require(float(config.get("ridge_lambda", 0.0)) > 0.0, "ridge_lambda must be positive")
    _require(isinstance(config.get("seed"), int), "seed must be explicit and integral")

    disturbance = config.get("disturbance", {})
    _require(disturbance.get("type") == "sine_joint_torque", "only sine joint disturbance is supported")
    for field in ("amplitude_n_m", "angular_frequency_rad_s", "phase_rad"):
        _require(len(disturbance.get(field, [])) == 2, f"disturbance.{field} must have two values")
    stages = disturbance.get("stages", [])
    _require(len(stages) >= 2, "at least nominal and time-varying stages are required")
    cursor = 0.0
    names: set[str] = set()
    for stage in stages:
        start = float(stage["start_fraction"])
        end = float(stage["end_fraction"])
        _require(abs(start - cursor) < 1e-12, "disturbance stages must be contiguous")
        _require(start < end <= 1.0, "invalid disturbance stage bounds")
        _require(stage["name"] not in names, "disturbance stage names must be unique")
        names.add(str(stage["name"]))
        cursor = end
    _require(abs(cursor - 1.0) < 1e-12, "disturbance stages must cover the full stream")
    _require(stages[0]["name"] == "nominal" and float(stages[0]["scale"]) == 0.0, "first stage must be nominal")
    _require(stages[-1]["name"] == "time_varying" and float(stages[-1]["scale"]) > 0.0, "last stage must be time_varying")

    training = config.get("training", {})
    _require(1 <= int(training.get("window", 0)), "training.window must be positive")
    _require(0.0 < float(training.get("train_fraction", 0.0)) < 1.0, "train_fraction must be in (0,1)")
    evaluation = config.get("evaluation", {})
    _require(evaluation.get("rollout_horizons") == EXPECTED_HORIZONS, "rollout horizons must be 10/20/50/100")
    for name in ("smoke", "full"):
        profile = config.get("profiles", {}).get(name, {})
        _require(int(profile.get("trajectory_count", 0)) >= 2, f"{name} needs at least two trajectories")
        _require(int(profile.get("steps", 0)) >= max(EXPECTED_HORIZONS), f"{name} must support 100-step rollout")


def load_foundation_config(path: str | Path, profile: str) -> dict[str, Any]:
    """Return a validated config with the selected run profile materialized."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_foundation_config(payload)
    if profile not in payload["profiles"]:
        raise ValueError(f"unknown Plan 01 profile: {profile}")
    config = deepcopy(payload)
    config["profile_name"] = profile
    config["profile"] = deepcopy(payload["profiles"][profile])
    return config


def stage_bounds(config: dict[str, Any], steps: int) -> list[dict[str, Any]]:
    """Convert fractional disturbance stages to exact half-open step ranges."""
    bounds: list[dict[str, Any]] = []
    cursor = 0
    stages = config["disturbance"]["stages"]
    for index, stage in enumerate(stages):
        stop = steps if index == len(stages) - 1 else int(round(float(stage["end_fraction"]) * steps))
        stop = min(max(stop, cursor + 1), steps)
        bounds.append({**stage, "start_step": cursor, "end_step": stop})
        cursor = stop
    if cursor != steps:
        raise ValueError("stage rounding did not cover the stream")
    return bounds
