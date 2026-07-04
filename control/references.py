"""Reference trajectory helpers shared by control methods."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from cdsm.kinematics.ik import IKConfig, MujocoSiteIK
from cdsm.references.cartesian import CartesianReferenceConfig, generate_cartesian_reference


def build_ramp_reference(
    *,
    dt: float,
    duration: float,
    start: Sequence[float],
    target: Sequence[float],
    ramp_duration: float,
) -> dict[str, np.ndarray]:
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if start_arr.shape != target_arr.shape:
        raise ValueError("start and target must have the same shape")
    n = int(round(float(duration) / float(dt))) + 1
    t = np.arange(n, dtype=np.float64) * float(dt)
    s = np.clip(t / max(float(ramp_duration), float(dt)), 0.0, 1.0)
    s = 3.0 * s * s - 2.0 * s * s * s
    values = start_arr[None, :] + (target_arr - start_arr)[None, :] * s[:, None]
    return {
        "t": t,
        "q_ref": values,
        "dq_ref": np.gradient(values, float(dt), axis=0),
    }


def ik_config_from_args(args: argparse.Namespace) -> IKConfig:
    return IKConfig(
        site_name=args.ik_site,
        max_iter=args.ik_max_iter,
        tol=args.ik_tol,
        damping=args.ik_damping,
        max_step=args.ik_max_step,
        joint_margin=args.ik_joint_margin,
        smooth_window_s=args.ik_smooth_window_s,
        q_seed_a=args.ik_seed_a,
        q_seed_b=args.ik_seed_b,
    )


def build_cartesian_circle_reference(
    args: argparse.Namespace,
    ik_solver: MujocoSiteIK,
) -> dict[str, np.ndarray]:
    cartesian = generate_cartesian_reference(
        CartesianReferenceConfig(
            kind="circle",
            dt=args.dt,
            period=args.period,
            num_cycles=args.num_cycles,
            center_x=args.center_x,
            center_y=args.center_y,
            radius_x=args.radius,
            radius_y=args.radius,
            phase=args.phase,
            start_hold=args.start_hold,
            time_scaling=args.time_scaling,
        )
    )
    inverse = ik_solver.solve_trajectory(
        np.asarray(cartesian["xy_ref"], dtype=np.float64)
    )
    return {
        "t": np.asarray(cartesian["t"], dtype=np.float64),
        "q_ref": np.asarray(inverse["q_ref"], dtype=np.float64),
        "dq_ref": np.asarray(inverse["dq_ref"], dtype=np.float64),
        "ee_ref": np.asarray(cartesian["xy_ref"], dtype=np.float64),
        "dxy_ref": np.asarray(cartesian["dxy_ref"], dtype=np.float64),
        "ee_ik": np.asarray(inverse["ee_ik"], dtype=np.float64),
        "ik_error": np.asarray(inverse["ik_error"], dtype=np.float64),
        "ik_converged": np.asarray(inverse["ik_converged"], dtype=bool),
        "ik_iterations": np.asarray(inverse["ik_iterations"], dtype=np.int32),
    }
