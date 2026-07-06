"""Dependency-injected closed-loop tracking runtime."""

from __future__ import annotations

import time
from typing import Callable, Protocol

import numpy as np

from .finite_horizon_lqr import (
    KoopmanLqrTracker,
    LqrConfig,
    future_reference,
)


class FeedbackPlant(Protocol):
    """Minimum plant behavior required by the generic runtime."""

    def set_state(self, q: np.ndarray, dq: np.ndarray) -> None:
        ...

    def read_state(self) -> np.ndarray:
        ...

    def step(self) -> None:
        ...


ActuationCallback = Callable[[FeedbackPlant, np.ndarray], np.ndarray]


def run_model_predictive_tracking(
    *,
    model,
    plant: FeedbackPlant,
    t: np.ndarray,
    states_ref: np.ndarray,
    initial_position_dim: int,
    controller_config: LqrConfig,
    control_limit: float | np.ndarray,
    apply_control: ActuationCallback,
) -> dict[str, np.ndarray]:
    """Run feedback tracking without binding to a simulator or actuator type."""
    time_ref = np.asarray(t, dtype=np.float64)
    x_ref = np.asarray(states_ref, dtype=np.float64)
    if x_ref.shape[0] != time_ref.shape[0]:
        raise ValueError("t and states_ref lengths do not match")
    position_dim = int(initial_position_dim)
    plant.set_state(
        x_ref[0, :position_dim],
        x_ref[0, position_dim:],
    )
    tracker = KoopmanLqrTracker(
        model.A,
        model.B,
        model.C,
        controller_config,
    )
    previous_internal = np.zeros(model.B.shape[1], dtype=np.float64)
    records: dict[str, list] = {
        "t": [],
        "x_meas": [],
        "x_ref": [],
        "control_cmd": [],
        "internal_control": [],
        "actuator_cmd": [],
        "solve_ms": [],
    }
    for k in range(time_ref.shape[0] - 1):
        measured = plant.read_state()
        z0 = model.lift(measured)
        normalized_ref = future_reference(
            model,
            x_ref,
            k,
            controller_config.horizon,
        )
        started = time.perf_counter()
        internal_sequence = tracker.solve(
            z0,
            normalized_ref,
            previous_internal,
        )
        solve_ms = 1e3 * (time.perf_counter() - started)
        internal_command = internal_sequence[0]
        physical_control = model.recover_control(
            measured,
            internal_command,
        )
        physical_control = np.clip(
            physical_control,
            -np.asarray(control_limit),
            np.asarray(control_limit),
        )
        actuator_command = apply_control(plant, physical_control)
        plant.step()

        records["t"].append(time_ref[k])
        records["x_meas"].append(np.asarray(measured).copy())
        records["x_ref"].append(x_ref[k].copy())
        records["control_cmd"].append(np.asarray(physical_control).copy())
        records["internal_control"].append(internal_command.copy())
        records["actuator_cmd"].append(
            np.asarray(actuator_command).copy()
        )
        records["solve_ms"].append(solve_ms)
        previous_internal = internal_command
    return {
        key: np.asarray(values)
        for key, values in records.items()
    }
