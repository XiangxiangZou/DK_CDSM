"""
mujoco_cdsm_figure8_tracking.py
===============================
MuJoCo 绳驱空间机械臂末端 "8字形" 轨迹跟踪 (纯 PD + 拮抗绳驱映射)

功能:
  1. 在笛卡尔空间中生成一个平滑的 "8字形" 轨迹 (Lissajous 曲线)
  2. 通过逆运动学 (IK) 将末端位置转换为关节角参考 (qa, qb)
  3. 关节空间 PD 控制器 + 拮抗映射器 → 8 根绳索张力 → data.ctrl
  4. 使用 MjSimLogger 录制高清 GIF 运动示意图
  5. 保存完整数据图:
     - 末端笛卡尔轨迹 (期望 vs 实际)
     - 关节角跟踪 (qa, qb 期望 vs 实际)
     - 关节角跟踪误差 (deg)
     - PD 命令力矩 (τa, τb)
     - 8 根绳索张力曲线

运行:
    python mujoco_cdsm_figure8_tracking.py
"""

from __future__ import annotations

import os
import sys
import re
import time
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "glfw")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from multi_joint_cdsm_model import MultiJointSpaceRobot
from utils_plot import save_figure, get_save_dir
from utils_mujoco_log import MjSimLogger, add_line_to_scene

# ============================================================================
# 0. 全局常量
# ============================================================================
XML_PATH = "multi_joint_cable_dirven_space_robot.xml"

CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

# 同侧对合并索引
IDX_F1P = [0, 2]    # cable11, cable13
IDX_F1M = [1, 3]    # cable12, cable14
IDX_F2P = [4, 6]    # cable21, cable23
IDX_F2M = [5, 7]    # cable22, cable24

# ============================================================================
# 1. 可调参数 (用户自由配置)
# ============================================================================
# --- 仿真与轨迹 ---
DT_CTRL   = 0.01          # 仿真步长 / 控制周期 (s)
PERIOD    = 8.0            # 单个 8 字周期 (s)
NUM_CYCLES = 3             # 循环圈数
CENTER_X  = 4.0            # 8 字形中心 X (m)
CENTER_Y  = 0.0            # 8 字形中心 Y (m)
RADIUS_X  = 1.2            # X 方向振幅 (m)
RADIUS_Y  = 0.8            # Y 方向振幅 (m)

# --- PD 控制增益 ---
KP_JOINT1 = 3500.0         # qa 比例增益 (Nm/rad)
KD_JOINT1 = 500.0          # qa 微分增益 (Nm·s/rad)
KP_JOINT2 = 1500.0         # qb 比例增益 (Nm/rad)
KD_JOINT2 = 200.0          # qb 微分增益 (Nm·s/rad)

# --- 绳驱参数 ---
F_PRE = 20.0               # 预紧力下限 (N)
F_MAX = 2000.0             # 单根绳张力上限 (N)

# --- 可视化 ---
TRAIL_SAMPLE_EVERY = 4     # 每隔多少步采样一次末端位置画轨迹拖尾
TRAIL_MAX_SEGMENTS  = 300  # 拖尾线段最大数量

# ============================================================================
# 2. 图-8 轨迹生成 + IK
# ============================================================================
def generate_figure8_trajectory(
    robot: MultiJointSpaceRobot,
    dt: float = DT_CTRL,
    period: float = PERIOD,
    num_cycles: int = NUM_CYCLES,
    center_x: float = CENTER_X,
    center_y: float = CENTER_Y,
    radius_x: float = RADIUS_X,
    radius_y: float = RADIUS_Y,
):
    """
    生成 "8字形" 末端笛卡尔轨迹, 并通过 IK 计算对应的关节角参考。
    返回: t_vals, qa_des, qb_des, x_des, y_des
    """
    total_time = period * num_cycles
    N = int(total_time / dt) + 1
    t_vals = np.linspace(0, total_time, N)
    omega = 2.0 * np.pi / period

    # Lissajous 8 字形: x = center_x + A sin(ωt), y = center_y + B sin(2ωt)
    x_des = center_x + radius_x * np.sin(omega * t_vals)
    y_des = center_y + radius_y * np.sin(2.0 * omega * t_vals)

    # 通过 IK 求解关节角
    qa_des = np.zeros(N)
    qb_des = np.zeros(N)
    q_guess = np.array([0.1, 0.1])
    ik_fail_count = 0
    for i in range(N):
        q_sol, conv = robot.inverse_kinematics([x_des[i], y_des[i]], q_guess)
        if not conv:
            ik_fail_count += 1
            if ik_fail_count <= 5:
                print(f"  [IK 警告] t={t_vals[i]:.2f}s, target=({x_des[i]:.3f},{y_des[i]:.3f}) 未收敛")
        qa_des[i], qb_des[i] = q_sol[0], q_sol[1]
        q_guess = q_sol

    if ik_fail_count > 0:
        print(f"  [IK] 总未收敛步数: {ik_fail_count}/{N}")

    # 过滤 IK 在 ±90° 附近的异常跳变 (低通平滑)
    from scipy.ndimage import uniform_filter1d
    window = max(3, int(0.05 / dt))  # ~50ms 窗口
    if window > 1:
        qa_des = uniform_filter1d(qa_des, size=window)
        qb_des = uniform_filter1d(qb_des, size=window)

    # 限幅到关节范围
    q_limit = np.pi / 2.0 - 0.05
    qa_des = np.clip(qa_des, -q_limit, q_limit)
    qb_des = np.clip(qb_des, -q_limit, q_limit)

    print(f"  轨迹生成完毕: {N} 个采样点, 总时长 {total_time:.1f}s, "
          f"qa∈[{np.min(qa_des):.2f},{np.max(qa_des):.2f}] rad, "
          f"qb∈[{np.min(qb_des):.2f},{np.max(qb_des):.2f}] rad")
    return t_vals, qa_des, qb_des, x_des, y_des


# ============================================================================
# 3. XML 加载 + 索引构建
# ============================================================================
def load_model(dt: float = DT_CTRL):
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()

    # 放宽关节限位
    xml_str = re.sub(
        r'range="-1\.5708 1\.5708"',
        'range="-1.7 1.7"',
        xml_str,
    )
    # 对齐 ctrlrange
    xml_str = re.sub(
        r'ctrlrange="0\s+2000"',
        f'ctrlrange="0 {F_MAX:g}"',
        xml_str,
    )
    # 对齐 timestep
    xml_str = re.sub(
        r'timestep="[^"]*"',
        f'timestep="{dt:g}"',
        xml_str,
    )
    # 确保 offwidth/offheight 足够大供 2K 渲染
    xml_str = re.sub(
        r'offwidth="\d+"',
        'offwidth="2560"',
        xml_str,
    )
    xml_str = re.sub(
        r'offheight="\d+"',
        'offheight="1440"',
        xml_str,
    )

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    return model, data, scratch, xml_str


def build_indices(model):
    tdn_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,   n) for n in CABLE_NAMES}
    act_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
    jnt_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    n) for n in JOINT_NAMES}
    site_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    for d, label in [(tdn_id, "tendon"), (act_id, "actuator"), (jnt_id, "joint")]:
        for k, v in d.items():
            if v < 0:
                raise RuntimeError(f"[模型自检] 未找到 {label} {k!r}")
    if site_ee < 0:
        raise RuntimeError("[模型自检] 未找到 site 'end_effector'")

    dof = {n: int(model.jnt_dofadr[jnt_id[n]]) for n in JOINT_NAMES}

    indices = dict(
        tdn_id=tdn_id,
        act_id=act_id,
        jnt_id=jnt_id,
        site_ee=site_ee,
        dof_j1=dof["joint1"],
        dof_j2=dof["joint2"],
        dof_j3=dof["joint3"],
        dof_j4=dof["joint4"],
        tdn_ids_ordered=np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int),
        act_ids_ordered=np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int),
    )
    return indices


# ============================================================================
# 4. 绳索 Jacobian (有限差分, 在 scratch 上)
# ============================================================================
def compute_tendon_jacobian_fd(
    model, scratch: mujoco.MjData, q_ref: np.ndarray,
    tdn_ids_ordered: np.ndarray, eps: float = 1e-6,
) -> np.ndarray:
    """返回 (8, nv) 的 tendon Jacobian, 行顺序 = CABLE_NAMES."""
    nv = model.nv
    nt = len(tdn_ids_ordered)
    J = np.zeros((nt, nv), dtype=float)
    q0 = scratch.qpos.copy()
    for j in range(nv):
        scratch.qpos[:] = q0
        scratch.qpos[j] += eps
        mujoco.mj_fwdPosition(model, scratch)
        Lp = np.array(scratch.ten_length, dtype=float)[tdn_ids_ordered].copy()
        scratch.qpos[:] = q0
        scratch.qpos[j] -= eps
        mujoco.mj_fwdPosition(model, scratch)
        Lm = np.array(scratch.ten_length, dtype=float)[tdn_ids_ordered].copy()
        J[:, j] = (Lp - Lm) / (2.0 * eps)
    scratch.qpos[:] = q0
    mujoco.mj_fwdPosition(model, scratch)
    return J


# ============================================================================
# 5. 拮抗映射器
# ============================================================================
def _solve_pair(m_p: float, m_m: float, tau_des: float,
                F_pre: float, F_max: float) -> tuple[float, float, float]:
    """单个模组的 1 等式 2 未知子问题."""
    u_max = F_max - F_pre
    tau_base = (m_p + m_m) * F_pre
    tau_eff = tau_des - tau_base
    EPS = 1e-12
    cand = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > EPS:
        u = max(tau_eff / m_p, 0.0)
        uc = min(u, u_max)
        cand.append((uc, 0.0, abs(tau_eff - m_p * uc), uc))
    if abs(m_m) > EPS:
        u = max(tau_eff / m_m, 0.0)
        uc = min(u, u_max)
        cand.append((0.0, uc, abs(tau_eff - m_m * uc), uc))
    u_p, u_m, res, _ = min(cand, key=lambda c: (c[2], c[3]))
    return F_pre + u_p, F_pre + u_m, res


def cable_antagonistic_map(
    tau_a_des: float,
    tau_b_des: float,
    J: np.ndarray,
    dof_j1: int, dof_j2: int, dof_j3: int, dof_j4: int,
    F_pre: float = F_PRE,
    F_max: float = F_MAX,
) -> tuple[np.ndarray, dict]:
    """期望关节力矩 → 8 根绳张力."""
    a = J[:, dof_j1] + J[:, dof_j2]
    b = J[:, dof_j3] + J[:, dof_j4]
    m_p1 = a[IDX_F1P].sum()
    m_m1 = a[IDX_F1M].sum()
    m_p2 = b[IDX_F2P].sum()
    m_m2 = b[IDX_F2M].sum()

    F1p, F1m, res_a = _solve_pair(m_p1, m_m1, tau_a_des, F_pre, F_max)
    F2p, F2m, res_b = _solve_pair(m_p2, m_m2, tau_b_des, F_pre, F_max)

    F = np.zeros(8, dtype=float)
    F[IDX_F1P] = F1p; F[IDX_F1M] = F1m
    F[IDX_F2P] = F2p; F[IDX_F2M] = F2m

    info = dict(m_p1=m_p1, m_m1=m_m1, m_p2=m_p2, m_m2=m_m2,
                F1p=F1p, F1m=F1m, F2p=F2p, F2m=F2m,
                res_a=res_a, res_b=res_b)
    return F, info


# ============================================================================
# 6. 主程序
# ============================================================================
def main():
    print("=" * 78)
    print("  MuJoCo 绳驱空间机械臂 —— 末端 8 字形轨迹跟踪 (纯 PD + 拮抗映射)")
    print("=" * 78)

    # ---- 6.1 解析模型 (IK用) ----
    robot_math = MultiJointSpaceRobot()

    # ---- 6.2 生成图-8 轨迹 ----
    print("\n[Step 1] 生成 8 字形轨迹...")
    t_vals, qa_des, qb_des, x_des, y_des = generate_figure8_trajectory(robot_math)
    N_REF = len(t_vals)
    T_TOTAL = t_vals[-1]

    # 计算期望末端速度 (用于 PD 前馈)
    dqa_des = np.gradient(qa_des, DT_CTRL)
    dqb_des = np.gradient(qb_des, DT_CTRL)

    # ---- 6.3 加载 MuJoCo 模型 ----
    print("\n[Step 2] 加载 MuJoCo 模型...")
    model, data, scratch, xml_text = load_model(dt=DT_CTRL)
    indices = build_indices(model)

    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  "
          f"ntendon={model.ntendon}  dt={DT_CTRL}")
    print(f"  模组1 (qa): Kp={KP_JOINT1}, Kd={KD_JOINT1}")
    print(f"  模组2 (qb): Kp={KP_JOINT2}, Kd={KD_JOINT2}")
    print(f"  绳驱参数: F_pre={F_PRE} N, F_max={F_MAX} N")
    print(f"  轨迹: {NUM_CYCLES} 圈 8 字形, 周期={PERIOD}s, 总时长={T_TOTAL:.1f}s")

    # ---- 6.4 初始化到起始构型 ----
    data.qpos[model.jnt_qposadr[indices["jnt_id"]["joint1"]]] = qa_des[0]
    data.qpos[model.jnt_qposadr[indices["jnt_id"]["joint2"]]] = qa_des[0]
    data.qpos[model.jnt_qposadr[indices["jnt_id"]["joint3"]]] = qb_des[0]
    data.qpos[model.jnt_qposadr[indices["jnt_id"]["joint4"]]] = qb_des[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    p0_ee = np.array(data.site_xpos[indices["site_ee"]], dtype=float)
    print(f"  起始末端位置: ({p0_ee[0]:+.3f}, {p0_ee[1]:+.3f})")

    # ---- 6.5 准备期望末端轨迹采样点 (用于场景叠加) ----
    TRAJ_SAMPLES = 200
    t_sample = np.linspace(0, T_TOTAL, TRAJ_SAMPLES)
    ee_des_xy = np.zeros((TRAJ_SAMPLES, 2))
    for i, ts in enumerate(t_sample):
        xi = np.interp(ts, t_vals, x_des)
        yi = np.interp(ts, t_vals, y_des)
        ee_des_xy[i] = [xi, yi]

    # ---- 6.6 MjSimLogger (GIF 录制) ----
    fig_save_dir = get_save_dir()

    # scene_decorator: 在每帧渲染前叠加期望轨迹 (白色虚线) 和实际末端轨迹 (红色实线)
    _WHITE = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    _RED   = np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float64)
    _DASH_PERIOD = 2
    ee_actual_xyz: list[np.ndarray] = []
    _step_counter = 0

    def scene_decorator(scene, d):
        nonlocal _step_counter
        _step_counter += 1

        # 期望轨迹: 白色虚线 8 字形
        for i in range(len(ee_des_xy) - 1):
            if (i // _DASH_PERIOD) % 2 != 0:
                continue
            p0 = np.array([ee_des_xy[i, 0], ee_des_xy[i, 1], 0.02])
            p1 = np.array([ee_des_xy[i+1, 0], ee_des_xy[i+1, 1], 0.02])
            add_line_to_scene(scene, p0, p1, rgba=_WHITE, width=3.0)

        # 实际末端轨迹: 红色实线拖尾
        if len(ee_actual_xyz) >= 2:
            pts = ee_actual_xyz[-TRAIL_MAX_SEGMENTS:]
            for j in range(len(pts) - 1):
                add_line_to_scene(scene, pts[j], pts[j+1], rgba=_RED, width=4.0)

    logger = MjSimLogger(
        model=model,
        xml_text=xml_text,
        enable_gif=True,
        gif_fps=30,
        gif_width=2560,
        gif_height=1440,
        camera_lookat=(3.0, 0.0, 0.0),
        camera_distance=12.0,
        camera_azimuth=90.0,
        camera_elevation=-90.0,
        dt=DT_CTRL,
        scene_decorator=scene_decorator,
        extra_gif_save_dir=fig_save_dir,
        extra_gif_basename="figure8_simulation_playback",
    )

    # ---- 6.7 数据记录缓冲 ----
    record_t   = []
    record_q   = []
    record_dq  = []
    record_ee  = []
    record_F   = []
    record_tau = []
    record_res = []

    # ---- 6.8 仿真主循环 ----
    act_ids_ordered = indices["act_ids_ordered"]
    tdn_ids_ordered = indices["tdn_ids_ordered"]
    dof_j1, dof_j2 = indices["dof_j1"], indices["dof_j2"]
    dof_j3, dof_j4 = indices["dof_j3"], indices["dof_j4"]
    site_ee = indices["site_ee"]
    jnt_id = indices["jnt_id"]

    # Jacobian 缓存
    _cached_q = None
    _cached_J = None

    N_SIM = N_REF + int(1.0 / DT_CTRL)  # 多跑 1 秒稳态
    step = 0
    peak_tau = {"a": 0.0, "b": 0.0}
    peak_F_val = 0.0
    saturate_count = 0

    print(f"\n[Step 3] 启动 MuJoCo 仿真 ({N_SIM} 步)...")
    print("  关闭 MuJoCo viewer 窗口或等待仿真结束.\n")

    try:
        while step < N_SIM:
            step_start = time.time()

            # --- (a) 参考轨迹插值 ---
            k = min(step, N_REF - 1)
            t_now = step * DT_CTRL
            qa_d = np.interp(t_now, t_vals, qa_des)
            qb_d = np.interp(t_now, t_vals, qb_des)
            dqa_d = np.interp(t_now, t_vals, dqa_des)
            dqb_d = np.interp(t_now, t_vals, dqb_des)

            # --- (b) 当前状态 + 纯 PD ---
            qa  = float(data.qpos[model.jnt_qposadr[jnt_id["joint1"]]])
            qb  = float(data.qpos[model.jnt_qposadr[jnt_id["joint3"]]])
            dqa = float(data.qvel[dof_j1])
            dqb = float(data.qvel[dof_j3])

            tau_a = KP_JOINT1 * (qa_d - qa) + KD_JOINT1 * (dqa_d - dqa)
            tau_b = KP_JOINT2 * (qb_d - qb) + KD_JOINT2 * (dqb_d - dqb)

            peak_tau["a"] = max(peak_tau["a"], abs(tau_a))
            peak_tau["b"] = max(peak_tau["b"], abs(tau_b))

            # --- (c) 绳索 Jacobian (缓存) ---
            q_now = np.array(data.qpos, dtype=float)
            if _cached_q is None or np.max(np.abs(q_now - _cached_q)) > 0.01:
                _cached_J = compute_tendon_jacobian_fd(model, scratch, q_now, tdn_ids_ordered)
                _cached_q = q_now.copy()
            J = _cached_J

            # --- (d) 拮抗映射 → 8 根绳张力 ---
            F_cable, info = cable_antagonistic_map(
                tau_a, tau_b, J, dof_j1, dof_j2, dof_j3, dof_j4,
            )
            data.ctrl[act_ids_ordered] = F_cable

            if info["res_a"] > 1e-3 or info["res_b"] > 1e-3:
                saturate_count += 1
            peak_F_val = max(peak_F_val, float(F_cable.max()))

            # --- (e) 记录 ---
            record_t.append(data.time)
            record_q.append([qa, qb])
            record_dq.append([dqa, dqb])
            p_ee = np.array(data.site_xpos[site_ee], dtype=float)
            record_ee.append(p_ee[:2].copy())
            record_F.append(F_cable.copy())
            record_tau.append([tau_a, tau_b])
            record_res.append([info["res_a"], info["res_b"]])

            if step % TRAIL_SAMPLE_EVERY == 0:
                ee_actual_xyz.append(np.array([p_ee[0], p_ee[1], 0.03]))

            # --- (f) 步进 ---
            mujoco.mj_step(model, data)
            logger.record(data)
            step += 1

            # 实时同步
            rest = DT_CTRL - (time.time() - step_start)
            if rest > 0:
                time.sleep(rest)

    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        logger.save_and_close()
        print(f"\n[仿真结束] 总步数 = {step}")

    # ---- 6.9 数据整理 ----
    record_t   = np.array(record_t)
    record_q   = np.array(record_q)
    record_dq  = np.array(record_dq)
    record_ee  = np.array(record_ee)
    record_F   = np.array(record_F)
    record_tau = np.array(record_tau)
    record_res = np.array(record_res)

    # 插值期望值到记录时间点
    qa_d_i  = np.interp(record_t, t_vals, qa_des)
    qb_d_i  = np.interp(record_t, t_vals, qb_des)
    dqa_d_i = np.interp(record_t, t_vals, dqa_des)
    dqb_d_i = np.interp(record_t, t_vals, dqb_des)
    x_d_i   = np.interp(record_t, t_vals, x_des)
    y_d_i   = np.interp(record_t, t_vals, y_des)

    cart_err = np.sqrt((record_ee[:, 0] - x_d_i) ** 2 + (record_ee[:, 1] - y_d_i) ** 2)

    # ---- 6.10 绘图 ----
    print("\n[Step 4] 绘制跟踪分析图...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 子图1: 末端笛卡尔轨迹
    ax = axes[0, 0]
    ax.plot(x_des, y_des, 'k--', lw=2, alpha=0.7, label="Desired Figure-8")
    ax.plot(record_ee[:, 0], record_ee[:, 1], 'r-', lw=1.5, alpha=0.85, label="Actual EE")
    ax.plot(record_ee[0, 0], record_ee[0, 1], 'go', ms=8, label="Start")
    ax.plot(record_ee[-1, 0], record_ee[-1, 1], 'bs', ms=8, label="End")
    ax.set_title("End-Effector Figure-8 Tracking")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # 子图2: 关节角跟踪
    ax = axes[0, 1]
    ax.plot(t_vals, np.rad2deg(qa_des), 'b--', lw=2, alpha=0.7, label=r"$q_a^{\mathrm{des}}$")
    ax.plot(record_t, np.rad2deg(record_q[:, 0]), 'b-', lw=1.8, alpha=0.85, label=r"$q_a$")
    ax.plot(t_vals, np.rad2deg(qb_des), 'r--', lw=2, alpha=0.7, label=r"$q_b^{\mathrm{des}}$")
    ax.plot(record_t, np.rad2deg(record_q[:, 1]), 'r-', lw=1.8, alpha=0.85, label=r"$q_b$")
    ax.set_title("Joint-Space Tracking")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=8)

    # 子图3: 关节角误差
    ax = axes[0, 2]
    ax.plot(record_t, np.rad2deg(qa_d_i - record_q[:, 0]), 'b-', lw=1.5, label=r"$e_{qa}$")
    ax.plot(record_t, np.rad2deg(qb_d_i - record_q[:, 1]), 'r-', lw=1.5, label=r"$e_{qb}$")
    ax.axhline(0, color='k', lw=0.6, alpha=0.3)
    ax.set_title("Joint Tracking Error")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (deg)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # 子图4: 笛卡尔误差
    ax = axes[1, 0]
    ax.plot(record_t, cart_err * 1000, 'm-', lw=1.8)
    ax.set_title("Cartesian Tracking Error")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|Δp| (mm)")
    ax.grid(True, alpha=0.4)

    # 子图5: PD 命令力矩
    ax = axes[1, 1]
    ax.plot(record_t, record_tau[:, 0], 'b-', lw=1.5, alpha=0.9, label=r"$\tau_a$")
    ax.plot(record_t, record_tau[:, 1], 'r-', lw=1.5, alpha=0.9, label=r"$\tau_b$")
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_title("PD Commanded Joint Torques")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"$\tau$ (Nm)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # 子图6: 绳索张力
    ax = axes[1, 2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    styles = ['-', '-', '-.', '-.', '--', '--', ':', ':']
    for i, name in enumerate(CABLE_NAMES):
        ax.plot(record_t, record_F[:, i], color=colors[i], ls=styles[i],
                lw=1.2, alpha=0.85, label=name)
    ax.axhline(F_PRE, color='k', lw=0.6, alpha=0.4, label=f"F_pre={F_PRE:.0f}N")
    ax.axhline(F_MAX, color='r', lw=0.6, alpha=0.4, label=f"F_max={F_MAX:.0f}N")
    ax.set_title("8 Cable Tension Commands")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("F (N)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6, ncol=2, loc='upper right')

    plt.suptitle("Cable-Driven Space Robot — Figure-8 Trajectory Tracking",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig_name="figure8_tracking_summary")
    plt.close()

    # 额外: 末端轨迹彩色时间图
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.plot(x_des, y_des, 'k--', lw=2, alpha=0.6, label="Desired")
    n_ee = len(record_ee)
    c_vals = np.linspace(0, 1, n_ee)
    ax2.scatter(record_ee[:, 0], record_ee[:, 1], c=c_vals, cmap='viridis',
                s=2, alpha=0.6, label="Actual (color=time)")
    ax2.plot(record_ee[0, 0], record_ee[0, 1], 'go', ms=10, label="Start")
    ax2.plot(record_ee[-1, 0], record_ee[-1, 1], 'rs', ms=10, label="End")
    ax2.scatter(x_des[0], y_des[0], c='orange', s=120, marker='*', zorder=5, label="Desired Start")
    ax2.set_title("EE Trajectory Detail (color = time progression)")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.4); ax2.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig_name="figure8_ee_trajectory_detail")
    plt.close()

    # ---- 6.11 统计摘要 ----
    rms_qa  = float(np.sqrt(np.mean((qa_d_i - record_q[:, 0]) ** 2)))
    rms_qb  = float(np.sqrt(np.mean((qb_d_i - record_q[:, 1]) ** 2)))
    rms_cart = float(np.sqrt(np.mean(cart_err ** 2)))
    max_qa  = float(np.max(np.abs(qa_d_i - record_q[:, 0])))
    max_qb  = float(np.max(np.abs(qb_d_i - record_q[:, 1])))
    max_cart = float(np.max(cart_err))

    print(f"\n{'='*60}")
    print(f"  跟踪质量统计")
    print(f"{'='*60}")
    print(f"  关节 RMS 误差 : e_qa={np.rad2deg(rms_qa):.4f}°   e_qb={np.rad2deg(rms_qb):.4f}°")
    print(f"  关节 峰值误差 : |e_qa|_max={np.rad2deg(max_qa):.4f}°   "
          f"|e_qb|_max={np.rad2deg(max_qb):.4f}°")
    print(f"  笛卡尔 RMS    : |Δp|_rms={rms_cart*1000:.3f} mm")
    print(f"  笛卡尔 峰值   : |Δp|_max={max_cart*1000:.3f} mm")
    print(f"  PD 峰值力矩   : |τa|={peak_tau['a']:.1f} Nm   |τb|={peak_tau['b']:.1f} Nm")
    print(f"  绳索峰值张力  : max F = {peak_F_val:.1f} N")
    print(f"  映射饱和步数  : {saturate_count}/{step}  ({100.0*saturate_count/max(step,1):.1f}%)")
    print(f"\n  全部结果已保存至: {fig_save_dir}")
    print(f"  MuJoCo 日志目录: {logger.save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()