"""闭环跟踪日志指标计算与保存辅助函数。"""

from __future__ import annotations

from typing import Dict

import numpy as np


def tracking_metrics(
    log: Dict[str, np.ndarray],
    *,
    position_dim: int | None = None,
) -> Dict[str, object]:
    """计算闭环跟踪指标。

    参数:
        log: `run_04` 中单个模型的闭环日志。

    返回:
        包含关节角 RMSE、角速度 RMSE、峰值力矩、峰值张力和平均求解耗时的字典。
    """
    x = np.asarray(log["x_meas"], dtype=np.float64)
    if "x_ref" in log:
        x_ref = np.asarray(log["x_ref"], dtype=np.float64)
        split = position_dim or x_ref.shape[1] // 2
        q_ref = x_ref[:, :split]
        dq_ref = x_ref[:, split:]
    else:
        q_ref = np.asarray(log["q_ref"], dtype=np.float64)
        dq_ref = np.asarray(log["dq_ref"], dtype=np.float64)
        split = position_dim or q_ref.shape[1]
    tau = np.asarray(
        log.get("tau_cmd", log.get("control_cmd")),
        dtype=np.float64,
    )
    cable = np.asarray(
        log.get("cable_tensions", log.get("actuator_cmd")),
        dtype=np.float64,
    )
    solve_ms = np.asarray(log["solve_ms"], dtype=np.float64)
    e_q = x[:, :split] - q_ref
    e_dq = x[:, split:] - dq_ref
    return {
        "rmse_q": float(np.sqrt(np.mean(e_q * e_q))),
        "rmse_q_by_joint": np.sqrt(np.mean(e_q * e_q, axis=0)).tolist(),
        "max_abs_q_error": float(np.max(np.abs(e_q))),
        "rmse_dq": float(np.sqrt(np.mean(e_dq * e_dq))),
        "peak_abs_tau": float(np.max(np.abs(tau))),
        "peak_cable_tension": float(np.max(cable)),
        "mean_solve_ms": float(np.mean(solve_ms)),
        "max_solve_ms": float(np.max(solve_ms)),
    }


def logs_to_npz_payload(logs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """把多个模型日志展开为可保存到一个 npz 的 payload。"""
    payload: Dict[str, np.ndarray] = {}
    for name, log in logs.items():
        for key, value in log.items():
            payload[f"{name}_{key}"] = np.asarray(value)
    return payload


def cartesian_tracking_metrics(
    *,
    ee_meas: np.ndarray,
    ee_ref: np.ndarray,
    ik_error: np.ndarray | None = None,
) -> Dict[str, object]:
    """计算末端笛卡尔空间跟踪指标。

    参数:
        ee_meas: 闭环执行后的真实末端 xy，形状 `(N,2)`，单位 m。
        ee_ref: 期望末端 xy，形状 `(N,2)`，单位 m。通常来自 `cartesian_reference.py`。
        ik_error: 可选的 IK 几何误差 `ee_ik - ee_ref`，形状 `(N,2)`，单位 m。
            该误差反映“笛卡尔轨迹本身是否可由关节参考精确实现”，不等同于闭环误差。

    返回:
        包含末端 RMSE、各轴 RMSE、最大误差和可选 IK 误差统计的字典。
    """
    ee = np.asarray(ee_meas, dtype=np.float64)
    ref = np.asarray(ee_ref, dtype=np.float64)
    n = min(ee.shape[0], ref.shape[0])
    err = ee[:n] - ref[:n]
    out: Dict[str, object] = {
        "rmse_ee": float(np.sqrt(np.mean(err * err))),
        "rmse_ee_by_axis": np.sqrt(np.mean(err * err, axis=0)).tolist(),
        "max_abs_ee_error": float(np.max(np.abs(err))),
    }
    if ik_error is not None:
        ik = np.asarray(ik_error, dtype=np.float64)
        ik = ik[: min(n, ik.shape[0])]
        out["rmse_ik"] = float(np.sqrt(np.mean(ik * ik)))
        out["max_abs_ik_error"] = float(np.max(np.abs(ik)))
    return out
