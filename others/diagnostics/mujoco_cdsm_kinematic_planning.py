"""
mujoco_cdsm_kinematic_planning.py
=======================================
绳驱空间机械臂 —— 关节空间 **运动学层面** 的规划与控制
    (模型: multi_joint_cable_dirven_space_robot.xml, 本脚本不启用其动力学积分)

====================================================================
  动力学控制  vs  运动学控制  (本脚本的定位)
====================================================================

【动力学控制】 (dynamics-level control, 见 test_mujoco_cable_antagonistic_pd_tracking.py)
    q_des(t), dq_des(t)
           │
           ▼  PD 控制律
       τ_a, τ_b                              (关节期望力矩, 单位: Nm)
           │
           ▼  同侧对拮抗映射器 cable antagonistic map
     F_1, F_2, ..., F_8                      (8 根绳索张力, 单位: N)
           │
           ▼  data.ctrl   (MuJoCo <motor tendon> 执行器把 ctrl 解释为张力)
    绳张力 → 关节广义力矩 → MuJoCo 按  M(q)·q̈ + C(q,q̇)·q̇ = τ  积分 qpos / qvel
  控制量是 "力 / 力矩", 系统按 Newton-Euler 方程演化.

【运动学控制】 (kinematic-level control, 本脚本实现)
    q_des(t)                                 (只需要位置参考; 速度 / 加速度不需要)
           │
           ▼  每个步长的几何规划器:
           │    (1) 编码器读出当前关节角 q_cur = (qa, qb)
           │    (2) 误差 Δq = q_des - q_cur, 由 sign(Δq) 决定正转 / 反转
           │    (3) 速率限幅, 给出本步目标构型 q_next
           │    (4) 由 spreader 刚性几何反解出 8 根绳的目标长度 L_i(q_next)
           │    (5) 计算卷筒命令 ΔL_i = L_i(q_next) - L_i(q_cur)
           │
           ▼  执行层 (真机: 8 个位置伺服电机按 ΔL_i 同步卷放; 仿真等价: 直接设 qpos)
    刚性绳约束下机械臂只能去到 q_next, 两步之间靠 "几何一致性" 连续
  控制量是 "位置 / 长度", 不碰力 / 质量 / 惯性, 用几何反解关系就完成闭环.

====================================================================
  与真实硬件的对应关系
====================================================================
实物绳驱空间机械臂每个控制周期的行为:
    (a) 编码器读出 qa, qb 当前值;
    (b) 与期望轨迹 qa_des, qb_des 比较, 由 sign(q_des - q_cur) 决定正转 / 反转方向;
    (c) 用 spreader 与主连杆铰链的几何公式, 计算目标构型下每根绳的长度 L_i;
    (d) 求出每根绳应该收放的长度 ΔL_i, 下发给 8 个绳索卷筒电机 (位置伺服模式);
    (e) 卷筒电机卷放绳长, 在刚性绳约束下机械臂被强制进入目标构型.
本脚本的控制回调完成 (a)~(d), (e) 用 "写 qpos 再 mj_forward" 来等效:
    - 运动学仿真里不需要求解任何约束力, 只需要把派生量 (site 坐标, data.ten_length
      等) 更新到新构型即可;
    - 这样可以同时观察:
        · 目标构型下 8 根绳的目标长度 L_i^{cmd}(t);
        · 每步的卷筒动作 ΔL_i(t);
        · 累计卷放量 ∫|ΔL_i| dt   (反映每台电机的工作量);
        · 与动力学版本对比: 同一 q_des(t) 下, 卡尔曼 / 跟踪误差 / EE 误差如何差别.

====================================================================
  与动力学 PD 版本的预期差别
====================================================================
    - 跟踪误差:  动力学版本有 PD 的过渡误差 + 绳驱几何退化误差; 运动学版本只有
                  "速率限幅残差" (当 OMEGA_MAX 足够大时几乎为 0).
    - 绳张力:    动力学版本关心 F_i (N); 运动学版本关心 L_i (m), 不解力.
    - 几何退化:  |q|→90° 时动力学版拮抗映射列同号, 会漂移; 运动学版只要 q_des 在
                  硬限位以内都能精确到达 (因为不依赖拮抗映射, 直接写 qpos).
    - 与惯量 / 质量参数无关: 质量改变对运动学版没有任何影响.

运行:
    python mujoco_cdsm_kinematic_planning.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "glfw")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import re
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from multi_joint_cdsm_model import MultiJointSpaceRobot
from utils_plot import save_figure, get_save_dir
from utils_mujoco_log import MjSimLogger, add_line_to_scene

# 为了在运动学规划结束后回填 "该轨迹所需的绳张力" 作为诊断图, 我们复用动力学
# 版本里已验证过的 拮抗映射器 + tendon Jacobian FD (不调 main()), 只拿模块级函数.
from mujoco_cdsm_antagonistic_pd_tracking import (
    cable_antagonistic_map,
    compute_tendon_jacobian_fd,
    F_PRE as ANT_F_PRE,
    F_MAX as ANT_F_MAX,
    IDX_F1P, IDX_F1M, IDX_F2P, IDX_F2M,   # 仅用于 "同侧对一致性" 检验, 可选
)


# ============================================================================
# 0. 常量 / 配置
# ============================================================================
XML_PATH = str(
    Path(__file__).resolve().parents[1]
    / "assets"
    / "multi_joint_cable_driven_space_robot.xml"
)

# 绳索命名顺序 —— 与 XML <tendon> 块完全一致, 控制器所有 8 维数组都按此排列
CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]

# --------------------------------------------------------------------------
# 几何参数 (务必与 XML / multi_joint_cable_dirven_space_robot.py 保持严格一致)
#   L1   : Link1 基座连杆长度 (固定, 不动)
#   L2   : Link2 第一级 spreader 主轴长度
#   L3   : Link3 第一级主活动连杆长度
#   L4   : Link4 第二级 spreader 主轴长度
#   L5   : Link5 第二级主活动连杆长度
#   LS_HALF         : spreader top / bot 到所在 link 中线的横向半长 (= Ls/2 = 0.3)
#   SPREADER_CENTER : spreader 横杆中心沿 link 轴向到 link 原点的距离 (= L2/2 = 0.1)
# 这两个常量决定了 8 根绳的几何端点 —— 一旦 XML 修改, 这里必须同步.
# --------------------------------------------------------------------------
L1 = 2.0
L2 = 0.2
L3 = 2.0
L4 = 0.2
L5 = 2.0
LS_HALF = 0.3
SPREADER_CENTER = 0.1

# --------------------------------------------------------------------------
# 轨迹目标 (与动力学 PD 版本保持一致, 便于一一对比)
#   qa 从 -90° → +90°,  qb 从 +90° → -90°
# --------------------------------------------------------------------------
QA_START = -np.pi / 2.0
QA_END   =  np.pi / 2.0
QB_START =  np.pi / 2.0
QB_END   = -np.pi / 2.0

# 仿真步长 = 轨迹采样周期 = 运动学控制周期 (三者严格一致)
DT_CTRL = 0.01
T_RAMP  = 50.0        # s, 余弦加减速段
T_HOLD  = 0.0         # s, 末端保持段
T_TOTAL = T_RAMP + T_HOLD

# --------------------------------------------------------------------------
# 关节最大旋转角速度 —— 运动学层面的速率限幅
#   真机中这个限幅由 "卷筒电机最大线速度 / 模组传动比" 决定, 用于保证两步之间
#   差值 Δq 不会大到使绳索瞬间被拉断或失张. 当前 30 deg/s 对 50s / 180° 的
#   余弦轨迹 (峰值 ~5.7 deg/s) 远远足够, 基本不会触发限幅.
# --------------------------------------------------------------------------
OMEGA_MAX = np.deg2rad(30.0)

# --------------------------------------------------------------------------
# 可视化参数 (与动力学版本保持一致: 白色虚线 = 期望, 红色实线 = 实时末端)
# --------------------------------------------------------------------------
TRAJ_SAMPLES_DESIRED = 160
DASH_ON_OFF = 2
TRAIL_SAMPLE_EVERY = 4
TRAIL_MAX_SEGMENTS = max(
    200,
    int(2.0 * (T_TOTAL + 1.0) / (DT_CTRL * TRAIL_SAMPLE_EVERY)),
)


# ============================================================================
# 1. 解析几何: (qa, qb)  ->  8 根绳索的长度
# ============================================================================
def cable_lengths_analytical(qa: float, qb: float) -> np.ndarray:
    """
    根据 spreader / 主连杆刚性几何, 解析求出 8 根绳索在构型 (qa, qb) 下的长度.
    输出顺序严格对应 CABLE_NAMES.

    坐标推导 (世界系, 取 2D 平面 z=0):
        anchor_base    = (0, 0)                                    ← cable11/12 起点
        link2_origin   = (L1, 0)                                    (joint1 = qa)
        link3_origin   = link2_origin + R(qa)·(L2, 0)
                       = (L1 + L2·cos(qa),  L2·sin(qa))             ← anchor_l3
        target_l3      = link3_origin + R(2qa)·(L3, 0)              ← cable13/14 起点
        link4_origin   = target_l3                                   (joint3 = qb, joint2=qa 隐式满足)
        link5_origin   = target_l3 + R(2qa+qb)·(L4, 0)
        target_l5      = link5_origin + R(2qa+2qb)·(L5, 0)          ← cable23/24 起点

        spreader1 中心 = link2_origin + R(qa)·(SPREADER_CENTER, 0)
        spreader1_top  = spreader1_center + R(qa)·(0,  +LS_HALF)
                       = link2_origin + R(qa)·(SPREADER_CENTER,  +LS_HALF)
        spreader1_bot  = link2_origin + R(qa)·(SPREADER_CENTER,  -LS_HALF)

        spreader2_top  = target_l3 + R(2qa+qb)·(SPREADER_CENTER,  +LS_HALF)
        spreader2_bot  = target_l3 + R(2qa+qb)·(SPREADER_CENTER,  -LS_HALF)

    其中 R(θ)·(x, y) = (x·cosθ - y·sinθ,  x·sinθ + y·cosθ).

    返回:
        ndarray shape=(8,), 按 CABLE_NAMES 顺序: [L_c11, L_c12, L_c13, L_c14,
                                                  L_c21, L_c22, L_c23, L_c24]
    """
    ca,   sa   = np.cos(qa),           np.sin(qa)
    c2a,  s2a  = np.cos(2.0 * qa),     np.sin(2.0 * qa)
    cab,  sab  = np.cos(2.0 * qa + qb), np.sin(2.0 * qa + qb)      # α = 2qa + qb
    c2b,  s2b  = np.cos(2.0 * qa + 2.0 * qb), np.sin(2.0 * qa + 2.0 * qb)  # β = 2qa + 2qb

    # spreader1 端点 (= link2_origin + 绕 qa 旋转的 (SPREADER_CENTER, ±LS_HALF))
    link2_origin = np.array([L1, 0.0])
    s1_top = link2_origin + np.array([
        SPREADER_CENTER * ca - LS_HALF * sa,
        SPREADER_CENTER * sa + LS_HALF * ca,
    ])
    s1_bot = link2_origin + np.array([
        SPREADER_CENTER * ca + LS_HALF * sa,
        SPREADER_CENTER * sa - LS_HALF * ca,
    ])

    # 主连杆铰链锚点
    link3_origin = link2_origin + np.array([L2 * ca,  L2 * sa])     # ≡ anchor_l3
    target_l3    = link3_origin + np.array([L3 * c2a, L3 * s2a])
    anchor_l3    = link3_origin.copy()
    anchor_base  = np.zeros(2)

    # spreader2 端点 (= target_l3 + 绕 α=2qa+qb 旋转的 (SPREADER_CENTER, ±LS_HALF))
    s2_top = target_l3 + np.array([
        SPREADER_CENTER * cab - LS_HALF * sab,
        SPREADER_CENTER * sab + LS_HALF * cab,
    ])
    s2_bot = target_l3 + np.array([
        SPREADER_CENTER * cab + LS_HALF * sab,
        SPREADER_CENTER * sab - LS_HALF * cab,
    ])

    # 第二级远端锚点
    link5_origin = target_l3 + np.array([L4 * cab, L4 * sab])
    target_l5    = link5_origin + np.array([L5 * c2b, L5 * s2b])

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    return np.array([
        _dist(anchor_base, s1_top),   # cable11 : anchor_base   - spreader1_top
        _dist(anchor_base, s1_bot),   # cable12 : anchor_base   - spreader1_bot
        _dist(target_l3,   s1_top),   # cable13 : target_l3     - spreader1_top
        _dist(target_l3,   s1_bot),   # cable14 : target_l3     - spreader1_bot
        _dist(anchor_l3,   s2_top),   # cable21 : anchor_l3     - spreader2_top
        _dist(anchor_l3,   s2_bot),   # cable22 : anchor_l3     - spreader2_bot
        _dist(target_l5,   s2_top),   # cable23 : target_l5     - spreader2_top
        _dist(target_l5,   s2_bot),   # cable24 : target_l5     - spreader2_bot
    ], dtype=float)


# ============================================================================
# 2. 期望轨迹 (只需要位置, 不需要速度 / 加速度)
# ============================================================================
def _cosine_ramp(t: np.ndarray, T: float) -> np.ndarray:
    """余弦插值 s(t) ∈ [0, 1], 端点 0 速, 中段单调加速→减速."""
    tau = np.clip(t / T, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * tau))


def build_joint_reference(dt: float):
    """返回 (t_vals, qa_des, qb_des), 长度 = ceil(T_TOTAL/dt) + 1."""
    t_vals = np.arange(0.0, T_TOTAL + dt * 0.5, dt)
    s = _cosine_ramp(t_vals, T_RAMP)
    qa_des = QA_START + (QA_END - QA_START) * s
    qb_des = QB_START + (QB_END - QB_START) * s
    return t_vals, qa_des, qb_des


# ============================================================================
# 3. XML 加载 (仅用 MuJoCo 做 FK 显示 + tendon 长度派生量; 不启用动力学积分)
#    - 放宽 joint range 以允许 ±90° 端点
#    - 同步 timestep = DT_CTRL
#    - actuator 不使用 (ctrl 置 0), 所以 ctrlrange 无需改
# ============================================================================
def load_model():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()
    xml_str = re.sub(r'range="-1\.5708 1\.5708"', 'range="-1.7 1.7"', xml_str)
    xml_str = re.sub(r'timestep="[^"]*"', f'timestep="{DT_CTRL:g}"', xml_str)
    # --- 把 offscreen framebuffer 调到 2K (2560x1440) ------------------------
    # MjSimLogger 会用 mujoco.Renderer(..., height=1440, width=2560) 开离屏渲染;
    # 如果 XML 里 offwidth/offheight 小于请求分辨率, Renderer 初始化会失败.
    # 这里直接把 XML 里那行 <global offwidth=... offheight=.../> 替换成 2K.
    xml_str = re.sub(
        r'<global\s+offwidth="[^"]*"\s+offheight="[^"]*"\s*/>',
        '<global offwidth="1920" offheight="1080"/>',
        xml_str,
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    # 把改动后的 xml_str 也返回: MjSimLogger 用它存档 <basename>.xml 供日后回放
    return model, data, xml_str


def build_indices(model):
    tdn_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n) for n in CABLE_NAMES}
    jnt_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
              for n in ("joint1", "joint2", "joint3", "joint4")}
    site_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    for name, idx in list(tdn_id.items()) + list(jnt_id.items()):
        if idx < 0:
            raise RuntimeError(f"[模型自检失败] 未找到 {name!r}")
    if site_ee < 0:
        raise RuntimeError("[模型自检失败] 未找到 site 'end_effector'")
    return tdn_id, jnt_id, site_ee


# ============================================================================
# 4. 可视化 overlay: 期望 EE 白色虚线 + 实时 EE 红色实线
#    (与 test_mujoco_cable_antagonistic_pd_tracking.py 风格一致)
# ============================================================================
_WHITE = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
_RED   = np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float64)
_LINE_WIDTH_DES  = 3.0
_LINE_WIDTH_REAL = 4.0


def _init_line_segment(geom, frm, to, rgba, width):
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(frm, dtype=np.float64),
        np.asarray(to,  dtype=np.float64),
    )


def paint_viewer_overlay(viewer, ee_desired_xy, ee_actual_xyz):
    """把 "期望 EE 白色虚线 + 实时 EE 红色" 画到 viewer.user_scn (passive viewer 专用).

    user_scn 是一个独立的 "用户叠加层" 场景, 每帧重画前要先把 ngeom 归零.
    """
    user_scn = viewer.user_scn
    max_geom = int(user_scn.maxgeom)
    idx = 0
    for i in range(len(ee_desired_xy) - 1):
        if (i // DASH_ON_OFF) % 2 != 0:
            continue
        if idx >= max_geom:
            break
        p0 = np.array([ee_desired_xy[i,     0], ee_desired_xy[i,     1], 0.02])
        p1 = np.array([ee_desired_xy[i + 1, 0], ee_desired_xy[i + 1, 1], 0.02])
        _init_line_segment(user_scn.geoms[idx], p0, p1, _WHITE, _LINE_WIDTH_DES)
        idx += 1
    if len(ee_actual_xyz) >= 2:
        pts = ee_actual_xyz[-TRAIL_MAX_SEGMENTS - 1:]
        for j in range(len(pts) - 1):
            if idx >= max_geom:
                break
            _init_line_segment(user_scn.geoms[idx], pts[j], pts[j + 1], _RED, _LINE_WIDTH_REAL)
            idx += 1
    user_scn.ngeom = idx


def make_offscreen_scene_decorator(ee_desired_xy, ee_actual_xyz_list):
    """返回一个 closure, 供 MjSimLogger 每帧离屏渲染前调用, 在 renderer.scene 上
    追加 "期望 EE 白色虚线 + 实时 EE 红色" 两条线.

    - 与 paint_viewer_overlay 的关键区别:
        * renderer.scene 已被 update_scene 填入模型几何, 这里要 *追加* 而不是清零;
        * 因此使用 add_line_to_scene (它会自增 scene.ngeom), 不做 ngeom=idx 的覆盖.
    - ee_actual_xyz_list 是一个可变 list, closure 持有引用, 每帧都能看到最新的尾段.
    """
    def _decorator(scene, data):
        # (1) 期望轨迹 (白色虚线)
        for i in range(len(ee_desired_xy) - 1):
            if (i // DASH_ON_OFF) % 2 != 0:
                continue
            p0 = (ee_desired_xy[i,     0], ee_desired_xy[i,     1], 0.02)
            p1 = (ee_desired_xy[i + 1, 0], ee_desired_xy[i + 1, 1], 0.02)
            if not add_line_to_scene(scene, p0, p1, _WHITE, _LINE_WIDTH_DES):
                return
        # (2) 实时末端轨迹 (红色)
        if len(ee_actual_xyz_list) >= 2:
            pts = ee_actual_xyz_list[-TRAIL_MAX_SEGMENTS - 1:]
            for j in range(len(pts) - 1):
                if not add_line_to_scene(
                    scene, pts[j], pts[j + 1], _RED, _LINE_WIDTH_REAL
                ):
                    return
    return _decorator


# ============================================================================
# 5. 主循环: 纯运动学规划 + 逐步卷放绳长
# ============================================================================
def main():
    print("=" * 78)
    print("  MuJoCo 绳驱机械臂 —— 关节空间 纯运动学 规划与控制")
    print("=" * 78)

    robot_math = MultiJointSpaceRobot()
    model, data, xml_text = load_model()
    tdn_id, jnt_id, site_ee = build_indices(model)
    dt = float(model.opt.timestep)
    assert abs(dt - DT_CTRL) < 1e-9, (
        f"[致命] MuJoCo timestep ({dt}) 与 DT_CTRL ({DT_CTRL}) 不一致"
    )

    tdn_ids_ordered = np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int)
    qadr_j1 = int(model.jnt_qposadr[jnt_id["joint1"]])
    qadr_j2 = int(model.jnt_qposadr[jnt_id["joint2"]])
    qadr_j3 = int(model.jnt_qposadr[jnt_id["joint3"]])
    qadr_j4 = int(model.jnt_qposadr[jnt_id["joint4"]])

    print(f"  nq={model.nq}  ntendon={model.ntendon}  dt={dt}")
    print(f"  轨迹目标 : qa {np.rad2deg(QA_START):+.1f}° → {np.rad2deg(QA_END):+.1f}°, "
          f"qb {np.rad2deg(QB_START):+.1f}° → {np.rad2deg(QB_END):+.1f}°")
    print(f"  速率限幅 : ω_max = {np.rad2deg(OMEGA_MAX):.1f} deg/s  "
          f"(每步最多旋转 {np.rad2deg(OMEGA_MAX * dt):.3f}°)")

    # ------------------------------------------------------------------
    # 5.1  关节参考轨迹 (位置, 无需速度)
    # ------------------------------------------------------------------
    t_vals, qa_des, qb_des = build_joint_reference(dt)
    N_REF = len(t_vals)
    print(f"  参考轨迹 : T_ramp={T_RAMP}s + T_hold={T_HOLD}s, dt={dt}s, N_REF={N_REF}")

    # ------------------------------------------------------------------
    # 5.2  期望末端轨迹 (供 viewer overlay + 绘图)
    # ------------------------------------------------------------------
    ee_xy_dash = np.zeros((TRAJ_SAMPLES_DESIRED, 2))
    t_sample = np.linspace(0.0, T_TOTAL, TRAJ_SAMPLES_DESIRED)
    qa_sample = np.interp(t_sample, t_vals, qa_des)
    qb_sample = np.interp(t_sample, t_vals, qb_des)
    for i, (qa_i, qb_i) in enumerate(zip(qa_sample, qb_sample)):
        p5 = robot_math.forward_kinematics(qa_i, qb_i)[-1]
        ee_xy_dash[i] = [p5[0], p5[1]]

    X_des_full = np.zeros_like(t_vals)
    Y_des_full = np.zeros_like(t_vals)
    for i, (qa_i, qb_i) in enumerate(zip(qa_des, qb_des)):
        p5 = robot_math.forward_kinematics(qa_i, qb_i)[-1]
        X_des_full[i] = p5[0]
        Y_des_full[i] = p5[1]

    # ------------------------------------------------------------------
    # 5.3  初始化到起点构型
    # ------------------------------------------------------------------
    data.qpos[qadr_j1] = QA_START
    data.qpos[qadr_j2] = QA_START
    data.qpos[qadr_j3] = QB_START
    data.qpos[qadr_j4] = QB_START
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0                     # 本脚本不使用 actuator; 保持为 0
    mujoco.mj_forward(model, data)
    p0_ee = np.array(data.site_xpos[site_ee], dtype=float)
    print(f"  起始末端 : ({p0_ee[0]:+.3f}, {p0_ee[1]:+.3f}, {p0_ee[2]:+.3f})")

    # ------------------------------------------------------------------
    # 5.4  一致性自检: 解析几何公式 vs MuJoCo 内部 tendon 长度
    # ------------------------------------------------------------------
    L_ana0 = cable_lengths_analytical(QA_START, QB_START)
    L_muj0 = np.array(data.ten_length, dtype=float)[tdn_ids_ordered]
    consistency_err = float(np.max(np.abs(L_ana0 - L_muj0)))
    print(f"  几何一致性: max|L_analytical - L_mujoco| = {consistency_err:.2e} m  "
          f"(应 << 1e-6)")
    if consistency_err > 1e-5:
        print("  ⚠ 警告: 解析几何与 MuJoCo 不一致, 请检查 cable_lengths_analytical 的常数!")

    # ------------------------------------------------------------------
    # 5.5  记录缓冲
    # ------------------------------------------------------------------
    rec_t: list[float] = []
    rec_q: list[list[float]] = []            # 本步执行后的 (qa, qb)
    rec_qdes: list[list[float]] = []         # 本步的期望 (qa_des, qb_des)
    rec_p: list[np.ndarray] = []             # 末端 xyz
    rec_L_cur: list[np.ndarray] = []         # 本步 "卷筒动作前" 的当前绳长 (由 q_cur 反解)
    rec_L_cmd: list[np.ndarray] = []         # 本步 "卷筒动作后" 的目标绳长 (由 q_next 反解)
    rec_L_muj: list[np.ndarray] = []         # MuJoCo 内部 tendon 长度 (= L_cmd; 作为校验)
    rec_dL: list[np.ndarray] = []            # 本步每根绳的卷放量 ΔL_i = L_cmd - L_cur

    ee_actual_xyz: list[np.ndarray] = [p0_ee.copy()]
    total_travel = np.zeros(8, dtype=float)  # 累计 |ΔL| (m), 反映各卷筒工作量
    peak_abs_dL = 0.0                        # 所有步 / 所有绳中 |ΔL| 的峰值

    # ------------------------------------------------------------------
    # 5.5bis  MuJoCo 原生可回放日志 (utils_mujoco_log.MjSimLogger)
    #
    #   落盘文件 (都位于 outputs/simulation_logs/<program>/<program>_<timestamp>/):
    #       <basename>.xml        ← XML 副本  (可直接喂给 simulate.exe)
    #       <basename>.mjb        ← 二进制模型 (mj_saveModel 输出)
    #       <basename>.npz        ← 完整轨迹 (qpos/qvel/ctrl/ten_length/actuator_force)
    #       <basename>.gif        ← 2K 高清运动动画 (含期望轨迹 + 实时末端叠加)
    #       <basename>.meta.txt   ← 元信息 + 回放命令
    #
    #   另: GIF 同时复制一份到 utils_plot.get_save_dir() 的图像目录, 命名为
    #       <program>_<ts>_cable_kinematic_playback.gif, 与静态图并排存放,
    #       满足 "论文插图与动画放一起" 的习惯.
    #
    #   scene_decorator: 关键! 每帧离屏渲染前会把 "期望 EE 白色虚线 + 实时 EE
    #   红色" 叠加到 GIF 上, 让 GIF 与 passive viewer 里看到的一模一样.
    # ------------------------------------------------------------------
    figs_dir_for_gif_copy = get_save_dir()   # 立刻 materialize, 保证后面能写入
    # get_save_dir() 返回 "outputs/figures/<program>/<timestamp>";
    # 让 GIF 副本的命名严格遵守 utils_plot.save_figure 的 "<program>_<ts>_<fig_name>".
    _parts = figs_dir_for_gif_copy.replace("\\", "/").rstrip("/").split("/")
    _plot_program_name = _parts[-2] if len(_parts) >= 2 else "run"
    _plot_timestamp    = _parts[-1] if len(_parts) >= 1 else time.strftime('%Y%m%d_%H%M%S')
    extra_gif_basename = f"{_plot_program_name}_{_plot_timestamp}_cable_kinematic_playback"

    logger = MjSimLogger(
        model=model,
        xml_text=xml_text,
        enable_gif=True,
        gif_fps=30,
        gif_width=2560,        # 2K / QHD; 要求 XML offwidth/offheight >= 同值
        gif_height=1440,
        camera_lookat=(3.0, 0.0, 0.0),
        camera_distance=12.0,
        camera_azimuth=90.0,
        camera_elevation=-90.0,
        dt=dt,
        scene_decorator=make_offscreen_scene_decorator(ee_xy_dash, ee_actual_xyz),
        extra_gif_save_dir=figs_dir_for_gif_copy,
        extra_gif_basename=extra_gif_basename,
    )

    # 先把 "起点 t=0" 的状态也记一份, 让轨迹首帧有效
    logger.record(data)

    print("\n  启动 MuJoCo passive viewer. 关闭窗口或 Ctrl+C 退出仿真.\n")
    step = 0
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 12.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -90
            viewer.cam.lookat[:] = [3.0, 0.0, 0.0]

            N_HOLD_EXTRA = int(round(1.0 / dt))     # 结束后多跑 1s 看稳态
            N_SIM = N_REF + N_HOLD_EXTRA
            print(f"  仿真总步数 N_SIM={N_SIM}  (参考 {N_REF} 步 + 尾部稳态 {N_HOLD_EXTRA} 步)")

            last_sync = -1.0

            # ==================================================================
            # 每个 step 内都严格按实物的 (a)~(e) 五个子步骤执行一次
            # ==================================================================
            while viewer.is_running() and step < N_SIM:
                step_start = time.time()

                # ------ (a) 编码器采样: 读取当前关节角 ------
                qa_cur = float(data.qpos[qadr_j1])
                qb_cur = float(data.qpos[qadr_j3])
                L_cur = cable_lengths_analytical(qa_cur, qb_cur)

                # ------ (b) 取本步期望角度 (整数步号索引, 严格对齐) ------
                k_ref = min(step, N_REF - 1)
                qa_d = float(qa_des[k_ref])
                qb_d = float(qb_des[k_ref])

                # ------ (c) 差值 & 方向判断 ------
                #   err > 0  -> sign=+1  -> 关节应 "正转" (逆时针, z 轴右手正向)
                #   err < 0  -> sign=-1  -> 关节应 "反转" (顺时针)
                #   err = 0  -> sign= 0  -> 不动
                err_qa = qa_d - qa_cur
                err_qb = qb_d - qb_cur
                dir_qa = float(np.sign(err_qa))
                dir_qb = float(np.sign(err_qb))

                # ------ (d) 速率限幅给出本步的目标构型 q_next ------
                #   |步进| ≤ OMEGA_MAX · dt, 方向由 sign(err) 决定
                #   当 |err| < OMEGA_MAX·dt 时, 本步就能精确到位 (q_next = q_d)
                step_qa = dir_qa * min(abs(err_qa), OMEGA_MAX * dt)
                step_qb = dir_qb * min(abs(err_qb), OMEGA_MAX * dt)
                qa_next = qa_cur + step_qa
                qb_next = qb_cur + step_qb

                # ------ (e1) 由几何反解: 目标构型下 8 根绳的长度 ------
                L_cmd = cable_lengths_analytical(qa_next, qb_next)

                # ------ (e2) 本步每根绳的卷筒命令 ΔL_i ------
                #   ΔL > 0  => 目标绳长 > 当前绳长, 卷筒 "放绳"
                #   ΔL < 0  => 目标绳长 < 当前绳长, 卷筒 "收绳"
                #   真机下发给 8 个位置伺服电机的就是这 8 个 ΔL 值 (或累积目标长度).
                dL = L_cmd - L_cur
                total_travel += np.abs(dL)
                peak_abs_dL = max(peak_abs_dL, float(np.max(np.abs(dL))))

                # ------ (e3) 执行: 运动学等价于直接把 qpos 设到 q_next ------
                #   真机是 "卷筒位置伺服 + 刚性绳约束" 共同把机械臂推到 q_next;
                #   仿真里我们只关心构型更新 + 派生量 (site 坐标, ten_length), 所以
                #   不调 mj_step (会触发动力学积分, 需要力/质量), 而是:
                #       - 写 qpos = q_next   (joint1=joint2=qa_next; joint3=joint4=qb_next)
                #       - qvel  = 0           (运动学模式, 无动量)
                #       - mj_forward        → 刷新 site_xpos, ten_length, xfrc 等派生量
                data.qpos[qadr_j1] = qa_next
                data.qpos[qadr_j2] = qa_next
                data.qpos[qadr_j3] = qb_next
                data.qpos[qadr_j4] = qb_next
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                data.time += dt                   # 我们不调 mj_step, 所以手动推进时间戳
                step += 1

                # MjSimLogger: 记录一步状态 + 可能采一帧 GIF (每 stride 步一次)
                logger.record(data)

                # ------ 记录 ------
                p_now = np.array(data.site_xpos[site_ee], dtype=float)
                rec_t.append(data.time)
                rec_q.append([qa_next, qb_next])
                rec_qdes.append([qa_d, qb_d])
                rec_p.append(p_now.copy())
                rec_L_cur.append(L_cur.copy())
                rec_L_cmd.append(L_cmd.copy())
                rec_L_muj.append(np.array(data.ten_length, dtype=float)[tdn_ids_ordered].copy())
                rec_dL.append(dL.copy())

                if step % TRAIL_SAMPLE_EVERY == 0:
                    ee_actual_xyz.append(np.array([p_now[0], p_now[1], 0.03]))

                if data.time - last_sync > 1.0 / 30.0:
                    paint_viewer_overlay(viewer, ee_xy_dash, ee_actual_xyz)
                    viewer.sync()
                    last_sync = data.time

                # 实时播放 (每步尽量贴近 wall-clock dt)
                rest = dt - (time.time() - step_start)
                if rest > 0:
                    time.sleep(rest)
    finally:
        # 无论仿真是否因 Ctrl+C / 关窗口提前终止, 都要把已经采集到的轨迹 / GIF 落盘
        logger.save_and_close()

    # ------------------------------------------------------------------
    # 5.6  数据整理
    # ------------------------------------------------------------------
    rec_t     = np.array(rec_t)
    rec_q     = np.array(rec_q)
    rec_qdes  = np.array(rec_qdes)
    rec_p     = np.array(rec_p)
    rec_L_cur = np.array(rec_L_cur)
    rec_L_cmd = np.array(rec_L_cmd)
    rec_L_muj = np.array(rec_L_muj)
    rec_dL    = np.array(rec_dL)

    X_d_i = np.interp(rec_t, t_vals, X_des_full)
    Y_d_i = np.interp(rec_t, t_vals, Y_des_full)
    cart_err = np.sqrt((rec_p[:, 0] - X_d_i) ** 2 + (rec_p[:, 1] - Y_d_i) ** 2)

    # 运行时解析 vs MuJoCo 的一致性 (整段轨迹上 |L_cmd - L_muj| 的最大值)
    run_consistency = float(np.max(np.abs(rec_L_cmd - rec_L_muj)))

    # ------------------------------------------------------------------
    # 5.7  绘图 (6 子图, 与动力学版本同布局, 便于对比)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(17, 10))

    # (1) 末端笛卡尔跟踪
    ax1 = fig.add_subplot(231)
    ax1.plot(X_des_full, Y_des_full, 'k--', lw=2, label="Desired EE")
    ax1.plot(rec_p[:, 0], rec_p[:, 1], 'r-', lw=2, alpha=0.85, label="Actual EE")
    ax1.plot(rec_p[0, 0], rec_p[0, 1], 'go', ms=8, label="Start")
    ax1.plot(rec_p[-1, 0], rec_p[-1, 1], 'bs', ms=8, label="End")
    ax1.set_title("End-Effector Cartesian Tracking (Kinematic Planning)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_aspect('equal', adjustable='box'); ax1.grid(True); ax1.legend(fontsize=9)

    # (2) 关节角度跟踪
    ax2 = fig.add_subplot(232)
    ax2.plot(rec_t, np.rad2deg(rec_qdes[:, 0]), 'k--', lw=2, label=r"$q_a^{\mathrm{des}}$")
    ax2.plot(rec_t, np.rad2deg(rec_q[:, 0]),    'b-',  lw=2, alpha=0.85, label=r"$q_a$")
    ax2.plot(rec_t, np.rad2deg(rec_qdes[:, 1]), color='gray', ls='--', lw=2, label=r"$q_b^{\mathrm{des}}$")
    ax2.plot(rec_t, np.rad2deg(rec_q[:, 1]),    'r-',  lw=2, alpha=0.85, label=r"$q_b$")
    ax2.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax2.set_title("Joint-Space Kinematic Tracking")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Angle (deg)")
    ax2.grid(True); ax2.legend(fontsize=9)

    # (3) 关节跟踪残差 (只剩速率限幅残差; 应接近 0)
    ax3 = fig.add_subplot(233)
    ax3.plot(rec_t, np.rad2deg(rec_qdes[:, 0] - rec_q[:, 0]), 'b-', lw=2, label=r"$e_a$")
    ax3.plot(rec_t, np.rad2deg(rec_qdes[:, 1] - rec_q[:, 1]), 'r-', lw=2, label=r"$e_b$")
    ax3.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax3.set_title("Joint Tracking Residual (rate-limit only)")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Error (deg)")
    ax3.grid(True); ax3.legend()

    # (4) 笛卡尔跟踪误差
    ax4 = fig.add_subplot(234)
    ax4.plot(rec_t, cart_err * 1000.0, 'm-', lw=2)
    ax4.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax4.set_title("Cartesian Tracking Error")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("|Δp| (mm)"); ax4.grid(True)

    # (5) 8 根绳的目标长度 L_i^{cmd}(t)
    ax5 = fig.add_subplot(235)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    styles = ['-', '-', '-.', '-.', '--', '--', ':', ':']
    for i, name in enumerate(CABLE_NAMES):
        ax5.plot(rec_t, rec_L_cmd[:, i], color=colors[i], ls=styles[i],
                 lw=1.4, alpha=0.85, label=name)
    ax5.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax5.set_title(r"8 Target Cable Lengths $L_i^{\mathrm{cmd}}(t)$ (m)")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Length (m)")
    ax5.grid(True); ax5.legend(fontsize=7, ncol=2, loc='best')

    # (6) 每步卷放量 ΔL_i (mm) —— 直接对应真机里 8 个卷筒电机的位置步长命令
    ax6 = fig.add_subplot(236)
    for i, name in enumerate(CABLE_NAMES):
        ax6.plot(rec_t, rec_dL[:, i] * 1000.0, color=colors[i], ls=styles[i],
                 lw=1.2, alpha=0.85, label=name)
    ax6.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax6.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax6.set_title(r"Per-Step Cable Payout $\Delta L_i(t)$ (mm) — winch commands")
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel(r"$\Delta L$ (mm)")
    ax6.grid(True); ax6.legend(fontsize=7, ncol=2, loc='best')

    plt.tight_layout()
    save_figure(fig_name="cable_kinematic_planning")

    # ==================================================================
    # 5.7bis  诊断图: "逆动力学 + 拮抗映射" 反推的 8 根绳张力 F_i(t)
    # ==================================================================
    # 说明:
    #   运动学控制器本身不计算 / 不下发张力 —— 它只给位置命令.
    #   但在真机上, 如果要让机械臂实际跟上这段 q(t) 轨迹 (非理想刚性绳 +
    #   有惯量), 8 个卷筒电机必须提供一定的绳张力. 这张图用 **逆动力学**
    #   离线反推出这些 "必须提供的张力" 作为工程诊断:
    #       (1) 从 rec_q 用数值差分得 dq, ddq;
    #       (2) 用 robot_math.get_M/get_C 算 τ = M(q)·ddq + C(q,dq)·dq   (重力 0);
    #       (3) 用 compute_tendon_jacobian_fd 在每个 q 上算 8×nv 绳索 Jacobian;
    #       (4) 用与动力学 PD 版本一致的 cable_antagonistic_map 得 8 绳张力.
    #   用途:
    #       - 评估所选轨迹 / 速率限幅是否使张力超出伺服电机上限 F_MAX;
    #       - 与 test_mujoco_cable_antagonistic_pd_tracking.py 画的同一目标轨
    #         迹下 "真闭环" 张力对比, 看 PD 版本多花了多少张力对抗瞬态误差;
    #       - 绝对不会影响本脚本的运动学仿真 (纯后处理, 事后计算).
    # ==================================================================
    print("\n  ----- 后处理: 逆动力学反推绳张力 F_i(t) -----")
    t_post_0 = time.time()

    # 用 rec_t 作为时间轴, 数值微分得 dq, ddq
    dt_arr = np.gradient(rec_t)                       # ≈ dt 的向量
    dq_num   = np.gradient(rec_q,   axis=0) / dt_arr[:, None]
    ddq_num  = np.gradient(dq_num,  axis=0) / dt_arr[:, None]

    # 端点两步的 gradient 边界误差较大, 对 ddq 做轻度滑窗平滑 (窗口 7 点)
    def _smooth(x, k=7):
        if len(x) < k:
            return x
        kernel = np.ones(k) / k
        return np.vstack([
            np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])
        ]).T

    ddq_num = _smooth(ddq_num, k=7)

    scratch = mujoco.MjData(model)
    dof_j1 = int(model.jnt_dofadr[jnt_id["joint1"]])
    dof_j2 = int(model.jnt_dofadr[jnt_id["joint2"]])
    dof_j3 = int(model.jnt_dofadr[jnt_id["joint3"]])
    dof_j4 = int(model.jnt_dofadr[jnt_id["joint4"]])

    N_log = len(rec_t)
    rec_tau   = np.zeros((N_log, 2), dtype=float)     # τ_a, τ_b
    rec_Ften  = np.zeros((N_log, 8), dtype=float)     # 8 根绳张力
    rec_res   = np.zeros((N_log, 2), dtype=float)     # 映射残差 (res_a, res_b)

    q_full = np.zeros(model.nq, dtype=float)
    for i in range(N_log):
        qa_i, qb_i = rec_q[i]
        dqa_i, dqb_i = dq_num[i]
        ddqa_i, ddqb_i = ddq_num[i]

        # (1) 2-DOF 逆动力学 (gravity=0): τ = M·ddq + C·dq
        M = robot_math.get_M(np.array([qa_i, qb_i]))
        C = robot_math.get_C(np.array([qa_i, qb_i]), np.array([dqa_i, dqb_i]))
        tau = M @ np.array([ddqa_i, ddqb_i]) + C @ np.array([dqa_i, dqb_i])
        rec_tau[i] = tau

        # (2) tendon Jacobian (scratch MjData 上做 FD, 不污染 data)
        q_full[:] = 0.0
        q_full[qadr_j1] = qa_i
        q_full[qadr_j2] = qa_i
        q_full[qadr_j3] = qb_i
        q_full[qadr_j4] = qb_i
        J = compute_tendon_jacobian_fd(model, scratch, q_full, tdn_ids_ordered)

        # (3) 拮抗映射 -> 8 根绳张力
        F_cable, info = cable_antagonistic_map(
            float(tau[0]), float(tau[1]), J,
            dof_j1, dof_j2, dof_j3, dof_j4,
            ANT_F_PRE, ANT_F_MAX,
        )
        rec_Ften[i] = F_cable
        rec_res[i]  = [info["res_a"], info["res_b"]]

    print(f"  逆动力学反推耗时 ≈ {time.time() - t_post_0:.1f} s   (N={N_log})")

    # --------- 新图: 2 子图 (8 绳张力 + 2 关节逆动力学力矩) ---------
    fig2 = plt.figure(figsize=(15, 5.5))

    ax_t = fig2.add_subplot(121)
    for i, name in enumerate(CABLE_NAMES):
        ax_t.plot(rec_t, rec_Ften[:, i], color=colors[i], ls=styles[i],
                  lw=1.4, alpha=0.85, label=name)
    ax_t.axhline(ANT_F_PRE, color='k',  lw=0.6, alpha=0.5, label=f"F_pre={ANT_F_PRE:.0f} N")
    ax_t.axhline(ANT_F_MAX, color='r',  lw=0.6, alpha=0.5, label=f"F_max={ANT_F_MAX:.0f} N")
    ax_t.axvline(T_RAMP,    color='k', ls=':', alpha=0.5)
    ax_t.set_title("Required Cable Tensions $F_i(t)$  (inverse-dynamics, post-hoc)")
    ax_t.set_xlabel("Time (s)"); ax_t.set_ylabel("F (N)")
    ax_t.grid(True); ax_t.legend(fontsize=7, ncol=2, loc='best')

    ax_q = fig2.add_subplot(122)
    ax_q.plot(rec_t, rec_tau[:, 0], 'b-', lw=1.6, alpha=0.9, label=r"$\tau_a$ (inverse dyn.)")
    ax_q.plot(rec_t, rec_tau[:, 1], 'r-', lw=1.6, alpha=0.9, label=r"$\tau_b$ (inverse dyn.)")
    ax_q.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax_q.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax_q.set_title("Required Joint Torques (from achieved kinematic trajectory)")
    ax_q.set_xlabel("Time (s)"); ax_q.set_ylabel(r"$\tau$ (Nm)")
    ax_q.grid(True); ax_q.legend(fontsize=9)

    plt.tight_layout()
    save_figure(fig_name="cable_kinematic_tensions")

    # ------------------------------------------------------------------
    # 5.8  跟踪质量统计
    # ------------------------------------------------------------------
    rms_ea   = float(np.sqrt(np.mean((rec_qdes[:, 0] - rec_q[:, 0]) ** 2)))
    rms_eb   = float(np.sqrt(np.mean((rec_qdes[:, 1] - rec_q[:, 1]) ** 2)))
    rms_cart = float(np.sqrt(np.mean(cart_err ** 2)))
    max_ea   = float(np.max(np.abs(rec_qdes[:, 0] - rec_q[:, 0])))
    max_eb   = float(np.max(np.abs(rec_qdes[:, 1] - rec_q[:, 1])))
    max_cart = float(np.max(cart_err))

    print(f"\n  仿真完成, 总步数 = {step}, 记录长度 = {len(rec_t)}")
    print(f"  解析几何一致性 (全程): max|L_cmd - L_mujoco| = {run_consistency:.2e} m")
    print(f"  关节 RMS 误差 : e_a={np.rad2deg(rms_ea):.4f}°   e_b={np.rad2deg(rms_eb):.4f}°")
    print(f"  关节 峰值误差 : |e_a|_max={np.rad2deg(max_ea):.4f}°   "
          f"|e_b|_max={np.rad2deg(max_eb):.4f}°")
    print(f"  笛卡尔 RMS    : |Δp|_rms = {rms_cart*1000:.3f} mm   "
          f"|Δp|_max = {max_cart*1000:.3f} mm")
    print(f"  单步 |ΔL| 峰值: {peak_abs_dL*1000:.3f} mm  "
          f"(对应卷筒瞬时线速度 ≈ {peak_abs_dL/dt*1000:.1f} mm/s)")
    print("  各绳累计卷放量 (累积 |ΔL|, 即卷筒整程工作量):")
    for i, name in enumerate(CABLE_NAMES):
        print(f"    {name}: {total_travel[i]:.4f} m   "
              f"(净变化 ΔL_net = {rec_L_cmd[-1, i] - rec_L_cmd[0, i]:+.4f} m)")

    # 逆动力学反推的 "所需张力" 统计
    peak_tau_a = float(np.max(np.abs(rec_tau[:, 0])))
    peak_tau_b = float(np.max(np.abs(rec_tau[:, 1])))
    peak_F_all = float(np.max(rec_Ften))
    above_fmax = int(np.sum(rec_Ften > ANT_F_MAX))
    print(f"  逆动力学峰值 : |τa|_max={peak_tau_a:.2f} Nm   |τb|_max={peak_tau_b:.2f} Nm")
    print(f"  所需张力峰值 : max F_i = {peak_F_all:.1f} N   "
          f"(相对上限 F_max={ANT_F_MAX:.0f} N: 占比 {100.0 * peak_F_all / ANT_F_MAX:.1f}%)")
    if above_fmax > 0:
        print(f"  ⚠ 警告: 有 {above_fmax} 个 (绳, 时刻) 的 F_i > F_max, 当前速率 / 质量 下真机无法实现")
    print(f"  图片已保存至: {get_save_dir()}")


if __name__ == "__main__":
    main()
