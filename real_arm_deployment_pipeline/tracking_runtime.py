"""关节空间闭环跟踪运行时。

本模块只负责“给定关节参考轨迹后，如何在 MuJoCo/真实机械臂接口上执行一步步
反馈控制”。它不生成参考轨迹，也不关心参考轨迹来自 ramp、笛卡尔 IK，还是以后
真实机械臂示教数据。

拆出本模块的目的：
1. `run_04_tracking_compare.py` 可以继续做普通关节 ramp 跟踪。
2. 新增的笛卡尔轨迹程序可以先把末端轨迹经 IK 转为 `q_ref,dq_ref`，再复用同一
   闭环执行逻辑。
3. 后续替换真实机械臂时，上层程序仍只调用统一的 plant 接口和本模块函数。
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

try:
    from .cable_mapping import F_PRELOAD, cable_antagonistic_map
    from .mujoco_plant import MujocoCablePlant
    from .tracking_controller import KoopmanLqrTracker, LqrConfig, pad_ref_norm
except ImportError:  # pragma: no cover - 支持直接以脚本方式运行同目录程序
    from cable_mapping import F_PRELOAD, cable_antagonistic_map
    from mujoco_plant import MujocoCablePlant
    from tracking_controller import KoopmanLqrTracker, LqrConfig, pad_ref_norm


def tau_to_cable_tensions(plant: MujocoCablePlant, tau_cmd: np.ndarray) -> np.ndarray:
    """把二自由度等效关节力矩映射为当前构型下的 8 根绳张力。

    参数:
        plant: 当前被控对象实例。这里使用 MuJoCo plant，是因为需要当前构型下的
            tendon Jacobian；未来真实机械臂也应暴露等价的 Jacobian 或张力分配接口。
        tau_cmd: 控制器输出的等效关节力矩 `[tau_a, tau_b]`，单位 Nm。

    返回:
        8 根绳张力，顺序与 `cable_mapping.CABLE_NAMES` 一致，单位 N。
    """
    jac = plant.compute_tendon_jacobian()
    dof_j1, dof_j2, dof_j3, dof_j4 = plant.torque_dofs()
    return cable_antagonistic_map(
        float(tau_cmd[0]),
        float(tau_cmd[1]),
        jac,
        dof_j1,
        dof_j2,
        dof_j3,
        dof_j4,
        f_pre=F_PRELOAD,
        f_max=None,
    )


def run_joint_space_closed_loop_model(
    *,
    model,
    xml: str,
    dt: float,
    ref: Dict[str, np.ndarray],
    lqr_cfg: LqrConfig,
    tau_limit: float,
) -> Dict[str, np.ndarray]:
    """对单个 EDMD/DKUC/DKAC 控制模型运行关节空间闭环跟踪。

    参数:
        model: 已加载的控制模型适配器。必须暴露 `A/B/C/lift/recover_control`，
            即 EDMD、DKUC、DKAC 三类统一 Koopman LQR 可控模型。
        xml: MuJoCo XML 路径。真实机械臂接入后，此参数会被真实 plant 配置替代。
        dt: 控制周期/仿真步长，单位 s。
        ref: 关节空间参考轨迹字典，至少包含：
            - `t`: 时间序列，形状 `(N,)`；
            - `q_ref`: 关节角参考，形状 `(N,2)`，单位 rad；
            - `dq_ref`: 关节角速度参考，形状 `(N,2)`，单位 rad/s。
        lqr_cfg: Koopman LQR/MPC 的预测时域和权重配置。
        tau_limit: 关节力矩限幅，单位 Nm；限幅发生在映射为绳张力之前。

    返回:
        闭环日志字典，包含每个控制周期的真实状态、参考、力矩、绳张力和求解耗时。
    """
    plant = MujocoCablePlant(xml, dt)
    plant.set_state(ref["q_ref"][0], ref["dq_ref"][0])
    tracker = KoopmanLqrTracker(model.A, model.B, model.C, lqr_cfg)
    n_step = len(ref["t"]) - 1
    u_prev_internal = np.zeros(model.B.shape[1], dtype=np.float64)

    rec = {
        "t": [],
        "x_meas": [],
        "q_ref": [],
        "dq_ref": [],
        "tau_cmd": [],
        "internal_control": [],
        "cable_tensions": [],
        "solve_ms": [],
    }
    for k in range(n_step):
        x_meas = plant.read_state()
        z0 = model.lift(x_meas)
        ref_norm = pad_ref_norm(model, ref, k, lqr_cfg.horizon)
        tic = time.perf_counter()
        internal_seq = tracker.solve(z0, ref_norm, u_prev_internal)
        solve_ms = 1e3 * (time.perf_counter() - tic)
        internal_cmd = internal_seq[0]
        tau_cmd = model.recover_control(x_meas, internal_cmd)
        tau_cmd = np.clip(tau_cmd, -float(tau_limit), float(tau_limit))
        cable = tau_to_cable_tensions(plant, tau_cmd)
        plant.apply_cable_tensions(cable)
        plant.step()

        rec["t"].append(float(k * dt))
        rec["x_meas"].append(x_meas.copy())
        rec["q_ref"].append(ref["q_ref"][k].copy())
        rec["dq_ref"].append(ref["dq_ref"][k].copy())
        rec["tau_cmd"].append(tau_cmd.copy())
        rec["internal_control"].append(internal_cmd.copy())
        rec["cable_tensions"].append(cable.copy())
        rec["solve_ms"].append(float(solve_ms))
        u_prev_internal = internal_cmd

    return {key: np.asarray(value) for key, value in rec.items()}
