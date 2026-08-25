"""Focused Plan 01 contract and integration checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cdsm.dktv_data import (
    REQUIRED_DATA_FIELDS,
    assess_data_quality,
    prove_time_variation,
    sine_disturbance,
    split_nominal_training_stream,
)
from koopman_control.dktv.config import (
    EXPECTED_METHODS,
    load_foundation_config,
    stage_bounds,
    validate_foundation_config,
)
from koopman_control.dktv.foundation import coordinate_contract_check
from prediction.common import Normalizer
from prediction.dkuc_prediction import DKUCConfig, DKUCModel, make_network_class


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dktv" / "base.json"
XML_PATH = ROOT / "traj_data" / "assets" / "multi_joint_cable_driven_space_robot.xml"


def _config(profile: str = "smoke") -> dict:
    return load_foundation_config(CONFIG_PATH, profile)


def _valid_arrays(trajectory_count: int = 4, steps: int = 120) -> dict[str, np.ndarray]:
    states = np.zeros((trajectory_count, steps + 1, 4), dtype=np.float64)
    applied = np.zeros((trajectory_count, steps, 2), dtype=np.float64)
    values: dict[str, np.ndarray] = {
        "t": np.broadcast_to(np.arange(steps + 1) * 0.01, (trajectory_count, steps + 1)).copy(),
        "states": states,
        "commanded_torque": applied.copy(),
        "applied_torque": applied,
        "commanded_tensions": np.full((trajectory_count, steps, 8), 20.0),
        "effective_tensions": np.full((trajectory_count, steps, 8), 20.0),
        "allocation_residual": applied.copy(),
        "disturbance_torque": applied.copy(),
        "reference_state": states.copy(),
        "saturation_flags": np.zeros((trajectory_count, steps, 10), dtype=bool),
        "joint_limit_flags": np.zeros((trajectory_count, steps + 1, 2), dtype=bool),
        "inputs": applied,
    }
    assert set(REQUIRED_DATA_FIELDS).issubset(values)
    return values


def test_base_config_freezes_public_contract() -> None:
    config = _config()
    assert config["methods"] == EXPECTED_METHODS
    assert config["input"] == "applied_torque"
    assert config["lifted_dim"] == config["state_dim"] + config["encoder_output_dim"] + 1
    assert config["encoder_update"] is False
    assert config["coordinates"]["state"] == "normalized"
    assert config["coordinates"]["input"] == "normalized_applied_torque"
    assert [stage["name"] for stage in stage_bounds(config, 120)] == [
        "nominal",
        "transition",
        "time_varying",
    ]


def test_config_rejects_online_encoder_update() -> None:
    config = deepcopy(_config())
    config["encoder_update"] = True
    with pytest.raises(ValueError, match="encoder"):
        validate_foundation_config(config)


def test_sine_disturbance_depends_on_absolute_time() -> None:
    config = _config()
    first = sine_disturbance(config, 0.11, 1.0)
    second = sine_disturbance(config, 0.37, 1.0)
    assert not np.allclose(first, second)
    assert np.array_equal(sine_disturbance(config, 0.37, 0.0), np.zeros(2))


def test_quality_and_split_are_deterministic() -> None:
    config = _config()
    arrays = _valid_arrays()
    quality = assess_data_quality(
        arrays,
        {"xml_joint_limits_rad": [[-1.0, 1.0], [-1.0, 1.0]]},
        config,
    )
    assert quality["accepted"] is True
    first = split_nominal_training_stream(arrays, config)
    second = split_nominal_training_stream(arrays, config)
    assert first[3] == second[3]
    assert np.array_equal(first[0]["states"], second[0]["states"])
    assert np.all(first[0]["disturbance_torque"] == 0.0)


def test_quality_rejects_recorded_saturation() -> None:
    config = _config()
    arrays = _valid_arrays()
    arrays["saturation_flags"][0, 0, 0] = True
    quality = assess_data_quality(
        arrays,
        {"xml_joint_limits_rad": [[-1.0, 1.0], [-1.0, 1.0]]},
        config,
    )
    assert quality["accepted"] is False
    assert quality["rejection_reasons"] == ["saturation_flags:1"]


def test_quality_rejects_missing_field_without_key_error() -> None:
    arrays = _valid_arrays()
    arrays.pop("states")
    quality = assess_data_quality(arrays, {}, _config())
    assert quality["accepted"] is False
    assert quality["missing_fields"] == ["states"]
    assert quality["rejection_reasons"] == ["missing_fields:states"]


def test_quality_rejects_wrong_shape() -> None:
    arrays = _valid_arrays()
    arrays["commanded_torque"] = arrays["commanded_torque"][..., :1]
    quality = assess_data_quality(arrays, {}, _config())
    assert quality["accepted"] is False
    assert quality["rejection_reasons"] == ["shape_errors"]
    assert quality["shape_errors"] == ["commanded_torque:[4, 120, 1] expected [4, 120, 2]"]


def test_quality_rejects_nonfinite_value() -> None:
    arrays = _valid_arrays()
    arrays["states"][0, 0, 0] = np.nan
    quality = assess_data_quality(arrays, {}, _config())
    assert quality["accepted"] is False
    assert quality["rejection_reasons"] == ["nonfinite_required_field"]
    assert quality["state_min"] is None


def test_quality_rejects_allocation_residual_over_tolerance() -> None:
    config = _config()
    arrays = _valid_arrays()
    arrays["allocation_residual"][0, 0, 0] = (
        2.0 * config["quality"]["allocation_residual_tolerance_n_m"]
    )
    quality = assess_data_quality(arrays, {}, config)
    assert quality["accepted"] is False
    assert quality["rejection_reasons"] == ["allocation_residual:0.0002"]


def test_affine_dkuc_lift_has_constant_observable() -> None:
    import torch

    network_class = make_network_class()
    network = network_class(
        DKUCConfig(lift_dim=11, hidden=(8,), include_constant=True),
        state_dim=4,
        control_dim=2,
    )
    lifted = network.lift(torch.zeros((3, 4), dtype=torch.float32))
    assert tuple(lifted.shape) == (3, 16)
    assert torch.equal(lifted[:, -1], torch.ones(3))


def test_physical_normalized_latent_physical_coordinate_contract() -> None:
    class StubModel:
        state_dim = 4
        control_dim = 2
        x_normer = Normalizer(
            mean=np.array([1.0, -2.0, 0.5, -0.5]),
            std=np.array([2.0, 4.0, 0.25, 0.5]),
        )
        u_normer = Normalizer(mean=np.array([3.0, -1.0]), std=np.array([2.0, 5.0]))
        A = np.eye(5, dtype=np.float64)
        B = np.array(
            [[0.1, 0.0], [0.0, 0.2], [0.3, 0.0], [0.0, 0.4], [0.0, 0.0]],
            dtype=np.float64,
        )

        def lift(self, x_phys: np.ndarray) -> np.ndarray:
            normalized = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1))[0]
            return np.concatenate([normalized, np.ones(1)])

        def step_latent(self, z: np.ndarray, u_phys: np.ndarray) -> np.ndarray:
            normalized = self.u_normer.transform(np.asarray(u_phys).reshape(1, -1))[0]
            return self.A @ z + self.B @ normalized

        def recover_state(self, z: np.ndarray) -> np.ndarray:
            return self.x_normer.inverse(np.asarray(z[:4]).reshape(1, -1))[0]

    result = coordinate_contract_check(
        StubModel(),
        np.array([3.0, 2.0, 1.0, -1.0]),
        np.array([5.0, 4.0]),
    )
    assert result["passed"] is True
    assert result["x_normalized"] == [1.0, 1.0, 2.0, -1.0]
    assert result["u_normalized"] == [1.0, 1.0]


def test_same_state_input_different_time_proves_time_variation() -> None:
    result = prove_time_variation(_config(), str(XML_PATH))
    assert result["fixed_next_state_difference_norm"] <= 1e-12
    assert result["time_varying_next_state_difference_norm"] > 1e-8
    assert result["passed"] is True


def test_plan_01_smoke_cli_end_to_end(tmp_path: Path) -> None:
    run_id = "pytest_plan01_e2e"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "experiments.dktv.plan_01",
        "--run-type",
        "smoke",
        "--device",
        "cpu",
        "--run-id",
        run_id,
        "--output-root",
        str(tmp_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result_dir = tmp_path / "results" / "dktv" / "plan_01" / run_id
    model_dir = tmp_path / "models" / "dktv" / run_id
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["acceptance"]["passed"] is True
    assert manifest["status"].startswith("accepted_")
    assert manifest["coordinate_contract_check"]["passed"] is True
    assert (result_dir / "arrays" / "fixed_dko_predictions.npz").is_file()
    assert (result_dir / "metrics" / "rollout_horizons.json").is_file()
    assert (result_dir / "logs" / "command.json").is_file()
    assert all(not path.startswith("/") for path in manifest["metrics"]["figures"])
    reloaded = DKUCModel(model_dir, "cpu")
    assert reloaded.A.shape == (16, 16)
    assert reloaded.B.shape == (16, 2)
