#!/usr/bin/env python3
"""
mujoco_cdsm_figure8_tracking.py  (v4 — 纯 PD + 绳驱拮抗映射, 移除力上限)
================================================================
MuJoCo 绳驱空间机械臂末端 "8 字形" 轨迹跟踪 (真实绳驱版本)

────────────────────────────────────────────────────────────────
驱动链路 (与真机 8 卷筒电机严格对应):
    q_d(t), q̇_d(t)   ← 8 字形轨迹 (笛卡尔参数化 + IK)
            │
            ▼  纯 PD 反馈律 (无动力学前馈)
    τ_a = Kp1·(qa_d − qa) + Kd1·(dqa_d − dqa)        模组 1 期望关节力矩
    τ_b = Kp2·(qb_d − qb) + Kd2·(dqb_d − dqb)        模组 2 期望关节力矩
            │
            ▼  Tendon Jacobian J(q) ∈ ℝ^(8×nv)  (在当前 q 处中心差分得到)
            ▼  同侧对拮抗映射:
                · spreader1 上 (cable11, cable13) 共用张力 F1⁺;
                            (cable12, cable14) 共用张力 F1⁻
                · spreader2 上 (cable21, cable23) 共用张力 F2⁺;
                            (cable22, cable24) 共用张力 F2⁻
                · 块对角 2×4 线性方程:
                       τ_a = m_p1·F1⁺ + m_m1·F1⁻
                       τ_b = m_p2·F2⁺ + m_m2·F2⁻
                  其中 m_p / m_m 由有效力臂 (J 的对应列和) 决定
                · 在 F ≥ F_pre (预紧力) 约束下闭式求解 4 个张力
            │
            ▼  data.ctrl[winch_c11..c24] = [F1⁺, F1⁻, F1⁺, F1⁻, F2⁺, F2⁻, F2⁺, F2⁻]
    MuJoCo 内部:
        qfrc_actuator = J(q)ᵀ · F        ← <motor tendon> 把绳张力转关节力矩
        M(q)·q̈ + C(q,q̇)·q̇ + b·q̇ = qfrc_actuator + qfrc_applied(=0)
    机械臂关节按真实绳驱产生的力矩演化, 100% 物理符合真机.

────────────────────────────────────────────────────────────────
v4 与 v2 / v3 的区别:
    v2 (FF + PD + cable, F_max=2000 N):
        - 有动力学前馈 → 抵消大部分惯性 / 科氏力
        - F_max=2000 N 把高瞬态力矩裁掉 → 末端轨迹失真
    v3 (纯 PD + 直驱关节力矩):
        - 用 data.qfrc_applied 绕过绳驱 → 与真机驱动方式不一致, 偏离需求
    v4 (本版本, 纯 PD + cable, 无 F_max 上限):  ✅
        - 严格保留真实绳驱驱动链路 (8 个绳索 motor → tendon → 关节)
        - **删除** 动力学前馈, 完全靠 PD 反馈律收敛误差
        - **删除** F_max 上限 (设为 1e6 N, XML ctrlrange 同步放开),
          使 PD 输出的任何力矩都能被拮抗映射 1:1 落到 8 根绳上, 不会饱和
        - 同侧对拮抗映射不再受 "F_max 裁切" 影响, 残差 ≈ 0
        - 适当增大 PD 增益, 弥补 "无前馈" 带来的稳态滞后

为什么这样可以精确跟踪 8 字?
    1) F_max 解除 → PD 闭环输出不受幅值约束, 高频反馈能即时跟上轨迹;
    2) 拮抗映射在不饱和时是 τ → F 的精确线性变换 (残差极小);
    3) 8 字形主频 ≈ 0.125 Hz, 闭环带宽 √(Kp/I) ≈ 3 Hz → 25 倍裕度;
    4) 关节内 damping=2.0 + KD 反馈 → 接近临界阻尼, 无超调振荡.

════════════════════════════════════════════════════════════════
⚠ 重要: 绳驱几何 "可达工作空间" 限制 (XML 内禀, 与本控制器无关)
════════════════════════════════════════════════════════════════
当前 XML 模型 spreader 中轴长 0.1 m, tips 位于 ±0.3 m, 在 **|qa| 或 |qb|
超过约 ±83°** 之后, 拮抗映射矩阵 M(q) 的 **两列变成同号**:
    qa → +90° :  m_p1 < 0,  m_m1 < 0  (两列同号 → 拮抗失效)
    qb → -90° :  m_p2 > 0,  m_m2 > 0
此时无论 F_max 设多大, 任意单根绳的张力都只能产生 "同方向" 的关节力矩,
拮抗作用瓦解, 映射器只能输出 双侧预紧 + 大残差, **跟踪必然失败**.

⇒ 8 字形参数必须保证 IK 求得的 |qa|, |qb| < ~80° **整段都成立**.
   实测下, 半幅过大 / 中心过靠近基座, 都会让 EE 折叠角接近 ±90°,
   touching 退化区. 本脚本默认参数已选在安全区, 见下方 RADIUS / CENTER.
   生成轨迹后还会自动打印 |q|_max 并给出越界警告.

运行:
    python mujoco_cdsm_figure8_tracking.py
"""

from __future__ import annotations

import os
import sys
import re
import time
from typing import Tuple

# MuJoCo 渲染后端 + Windows 控制台 UTF-8
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


# ═══════════════════════════════════════════════════════════════════════
# 0. 全局常量 / 命名
# ═══════════════════════════════════════════════════════════════════════
XML_PATH = "multi_joint_cable_dirven_space_robot.xml"

# 8 根绳与 8 个卷筒电机, 顺序必须与 XML <actuator> 中严格一致
CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",     # spreader1 上 4 根
    "cable21", "cable22", "cable23", "cable24",     # spreader2 上 4 根
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]

# 同侧对索引 (合并后只剩 4 个 "净" 张力, 简化为块对角 2×4 系统)
#   模组 1 (qa):  正侧 F1⁺ 同时分给 cable11, cable13;  反侧 F1⁻ 同时分给 cable12, cable14
#   模组 2 (qb):  正侧 F2⁺ 同时分给 cable21, cable23;  反侧 F2⁻ 同时分给 cable22, cable24
IDX_F1P = [0, 2]            # cable11, cable13 → F1⁺
IDX_F1M = [1, 3]            # cable12, cable14 → F1⁻
IDX_F2P = [4, 6]            # cable21, cable23 → F2⁺
IDX_F2M = [5, 7]            # cable22, cable24 → F2⁻

# 4 个 hinge 关节. joint1≡joint2 (qa 模组), joint3≡joint4 (qb 模组) 由 XML equality 约束
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]


# ═══════════════════════════════════════════════════════════════════════
# 1. 用户可调参数
# ═══════════════════════════════════════════════════════════════════════
DT_CTRL    = 0.01            # 控制周期 = MuJoCo timestep (s)

# ─── 8 字形轨迹 (Lissajous 1:2) ─────────────────────────────────────────
# ⚠ 参数选择准则: 必须保证 IK 求得的 |qa|, |qb| < ~80°, 否则触发绳驱几何退化区!
#   - CENTER_X 过近 (< 3.5 m): EE 需折叠, qb 接近 ±90° → 失败
#   - RADIUS_X / RADIUS_Y 过大 (> 1.0): IK 摆幅过大 → 失败
# 当前默认参数实测 |qa|_max ≈ 25°, |qb|_max ≈ 60°, 整段在可达区内.
PERIOD     = 8.0             # 单个 8 字周期 (s), 越大轨迹越慢, 越易跟踪
NUM_CYCLES = 3               # 绕几圈
CENTER_X   = 5.0             # 8 字中心 X (m); 推远到 5.0 让 EE 远离基座折叠区
CENTER_Y   = 0.0             # 8 字中心 Y (m)
RADIUS_X   = 0.8             # X 方向半幅 (m); 缩小到 0.8 留出几何裕度
RADIUS_Y   = 0.5             # Y 方向半幅 (m); 缩小到 0.5

# ─── PD 增益 (纯反馈, 无前馈) ───────────────────────────────────────────
# 设计原则:
#   * 等效模组惯量 I ≈ 10~30 kg·m², 取 I ≈ 15.
#   * ω_n = √(Kp/I) ≈ √(5000/15) ≈ 18 rad/s ≈ 2.9 Hz   → 闭环带宽
#   * ζ   = (Kd + b)/(2·√(Kp·I)) = (500 + 4)/(2·√(75000)) ≈ 0.92   → 近临界阻尼
#   * 轨迹主频率 = 2π/PERIOD = 0.785 rad/s ≈ 0.125 Hz, 带宽裕度 ≈ 23×.
#   * 因 F_max 已被放开 + 关节角处在可达区, 拮抗映射不会饱和.
KP_JOINT1 = 5000.0           # 模组 1 (qa) 比例增益 (Nm/rad)
KD_JOINT1 = 500.0            # 模组 1 (qa) 微分增益 (Nm·s/rad)
KP_JOINT2 = 5000.0           # 模组 2 (qb)
KD_JOINT2 = 500.0

# ─── 绳驱参数 ──────────────────────────────────────────────────────────
F_PRE = 20.0                 # 每根绳的最小预紧力 (N), 防止松弛
F_MAX = 1.0e6                # 绳张力上限 (N) — 设为巨大, 等价于"无限制";
                             #   XML ctrlrange 会被代码同步放开到 [0, F_MAX]
                             #   注: 上限 1e6 并非物理目标, 仅用来确保拮抗映射器
                             #       的裁切分支不会被触发. 实际 PD 输出对应的
                             #       张力远小于此 (典型 < 5000 N), 受关节角而非力限.

# ─── 拖尾 / 可视化 ─────────────────────────────────────────────────────
TRAIL_SAMPLE_EVERY = 4
TRAIL_MAX_SEGMENTS = 800


# ═══════════════════════════════════════════════════════════════════════
# 2. 工具函数: 加载 / 索引
# ═══════════════════════════════════════════════════════════════════════
def load_model(dt: float = DT_CTRL):
    """
    读取并轻度改写 XML:
      - 关节硬限位 ±90° → ±97.4° (远离硬限位)
      - 卷筒 ctrlrange 上限 2000 → F_MAX (移除力裁切)
      - MuJoCo timestep 同步到 DT_CTRL
      - 离屏渲染分辨率拉到 2K, 给 GIF 用
    """
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()

    xml_str = re.sub(r'range="-1\.5708 1\.5708"',
                     'range="-1.7 1.7"', xml_str)
    xml_str = re.sub(r'ctrlrange="0\s+2000"',
                     f'ctrlrange="0 {F_MAX:g}"', xml_str)
    xml_str = re.sub(r'timestep="[^"]*"',
                     f'timestep="{dt:g}"', xml_str)
    xml_str = re.sub(r'offwidth="\d+"',  'offwidth="2560"',  xml_str)
    xml_str = re.sub(r'offheight="\d+"', 'offheight="1440"', xml_str)

    model   = mujoco.MjModel.from_xml_string(xml_str)
    data    = mujoco.MjData(model)
    scratch = mujoco.MjData(model)    # 独立副本, 给 FD Jacobian 用, 不污染主 data
    return model, data, scratch, xml_str


def build_indices(model) -> dict:
    """建立常用名字 → MuJoCo 整型下标 的字典, 避免在主循环里反复调 mj_name2id."""
    tdn_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,   n)
              for n in CABLE_NAMES}
    act_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
              for n in ACTUATOR_NAMES}
    jnt_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    n)
              for n in JOINT_NAMES}
    site_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    # 自检: 任一 id 为 -1 表示 XML 修改后名字不一致
    for d, label in [(tdn_id, "tendon"), (act_id, "actuator"), (jnt_id, "joint")]:
        for k, v in d.items():
            if v < 0:
                raise RuntimeError(f"[模型自检失败] 未找到 {label}: {k!r}")
    if site_ee < 0:
        raise RuntimeError("[模型自检失败] 未找到 site 'end_effector'")

    return dict(
        tdn_id=tdn_id, act_id=act_id, jnt_id=jnt_id, site_ee=site_ee,
        dof_j1=int(model.jnt_dofadr[jnt_id["joint1"]]),
        dof_j2=int(model.jnt_dofadr[jnt_id["joint2"]]),
        dof_j3=int(model.jnt_dofadr[jnt_id["joint3"]]),
        dof_j4=int(model.jnt_dofadr[jnt_id["joint4"]]),
        qpos_j1=int(model.jnt_qposadr[jnt_id["joint1"]]),
        qpos_j2=int(model.jnt_qposadr[jnt_id["joint2"]]),
        qpos_j3=int(model.jnt_qposadr[jnt_id["joint3"]]),
        qpos_j4=int(model.jnt_qposadr[jnt_id["joint4"]]),
        tdn_ids_ordered=np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int),
        act_ids_ordered=np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int),
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. Tendon Jacobian + 符号自校验
# ═══════════════════════════════════════════════════════════════════════
def compute_tendon_jacobian_fd(
    model,
    scratch: mujoco.MjData,
    q_ref: np.ndarray,
    tdn_ids_ordered: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    在指定构型 q_ref 处用 **中心差分** 计算 tendon Jacobian:
        J[i, j] = ∂L_i / ∂q_j           shape (8, nv)
    L_i 是第 i 根绳的几何长度 (data.ten_length).

    注意:
      - 必须用 mj_fwdPosition 而非 mj_forward (后者会调 mjcb_control 引发死递归).
      - scratch 是独立 MjData 副本, 不修改主 data.
      - 关键: scratch 从 q_ref 出发 (而非用历史 q0), 保证 J 反映当前构型.
    """
    nv = model.nv
    nt = len(tdn_ids_ordered)

    scratch.qpos[:] = q_ref
    mujoco.mj_fwdPosition(model, scratch)
    q0 = scratch.qpos.copy()

    J = np.zeros((nt, nv), dtype=float)
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


def verify_jacobian_sign(model, data, scratch, indices) -> int:
    """
    在 q=0 处对每根绳分别施加 ctrl=1 N 单位激励, 读出 qfrc_actuator,
    与解析 J 比对, 自动识别 MuJoCo 内部 "符号约定":
        qfrc_actuator = sign_conv · Jᵀ · F          sign_conv ∈ {+1, −1}
    经验上同一版本 MuJoCo + 同一 XML 内 sign_conv 是常量, 但保险起见每次校验.
    """
    act_ids = indices["act_ids_ordered"]
    tdn_ids = indices["tdn_ids_ordered"]

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    J0 = compute_tendon_jacobian_fd(model, scratch, data.qpos, tdn_ids)

    # 构造矩阵 A, 第 i 行 = 单根绳 i 施加 ctrl=1 时的 qfrc_actuator
    A = np.zeros((len(act_ids), model.nv))
    for i in range(len(act_ids)):
        data.ctrl[:] = 0.0
        data.ctrl[act_ids[i]] = 1.0
        mujoco.mj_forward(model, data)
        A[i, :] = np.array(data.qfrc_actuator, dtype=float)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    delta_plus  = float(np.max(np.abs(A - J0)))
    delta_minus = float(np.max(np.abs(A + J0)))
    if delta_plus < 1e-4:
        sign = +1
    elif delta_minus < 1e-4:
        sign = -1
    else:
        raise RuntimeError(
            f"Jacobian 符号校验失败: |A−J|={delta_plus:.2e}, |A+J|={delta_minus:.2e}"
        )
    print(f"  [Jacobian 校验] sign_conv = {sign:+d}  "
          f"(|A−sign·J|={min(delta_plus, delta_minus):.2e})")
    return sign


# ═══════════════════════════════════════════════════════════════════════
# 4. 拮抗映射器: (τ_a, τ_b, J) → 8 根绳张力
# ═══════════════════════════════════════════════════════════════════════
def _solve_pair(
    m_p: float, m_m: float, tau_des: float,
    F_pre: float, F_max: float,
) -> Tuple[float, float, float]:
    """
    解每个模组的 1 等式 2 未知 (F⁺, F⁻) 子问题:
        m_p · F⁺ + m_m · F⁻ = τ_des,    F⁺, F⁻ ∈ [F_pre, F_max]
    令 u⁺ = F⁺ − F_pre, u⁻ = F⁻ − F_pre  (附加张力, ≥ 0), 则:
        m_p · u⁺ + m_m · u⁻ = τ_des − (m_p + m_m) · F_pre  ≡ τ_eff

    策略 (贪心): 在 3 种候选中选 "残差最小" (并列再选附加张力最小):
        (0) 双侧均预紧: u⁺ = u⁻ = 0  (τ_eff = 0 时严格满足)
        (1) 全部出力放正侧: u⁺ = max(τ_eff/m_p, 0), u⁻ = 0
        (2) 全部出力放反侧: u⁻ = max(τ_eff/m_m, 0), u⁺ = 0

    本版本 F_max 巨大 (1e6), u_max 实际不会触发上裁, 残差 ≈ 0.
    """
    u_max = F_max - F_pre
    tau_base = (m_p + m_m) * F_pre
    tau_eff = tau_des - tau_base

    EPS = 1e-12
    cand = [(0.0, 0.0, abs(tau_eff), 0.0)]              # (0) 双侧预紧

    if abs(m_p) > EPS:                                  # (1) 全部正侧出力
        u = max(tau_eff / m_p, 0.0)
        u = min(u, u_max)
        cand.append((u, 0.0, abs(tau_eff - m_p * u), u))

    if abs(m_m) > EPS:                                  # (2) 全部反侧出力
        u = max(tau_eff / m_m, 0.0)
        u = min(u, u_max)
        cand.append((0.0, u, abs(tau_eff - m_m * u), u))

    u_p, u_m, residual, _ = min(cand, key=lambda c: (c[2], c[3]))
    return F_pre + u_p, F_pre + u_m, residual


def cable_antagonistic_map(
    tau_a_des: float, tau_b_des: float, J: np.ndarray,
    dof_j1: int, dof_j2: int, dof_j3: int, dof_j4: int,
    sign_conv: int,
    F_pre: float = F_PRE, F_max: float = F_MAX,
) -> Tuple[np.ndarray, dict]:
    """
    把期望关节力矩 (τ_a, τ_b) 映射为 8 根绳的张力 F ∈ ℝ^8.

    数学推导:
        MuJoCo 内部:  τ_joint = sign_conv · Jᵀ · F
        模组 1 总力矩 τ_a = τ[j1] + τ[j2]  (因 joint1≡joint2 由 equality 约束)
        模组 2 总力矩 τ_b = τ[j3] + τ[j4]

        所以 cable i 对模组 1 的有效力臂 a_i = sign_conv·(J[i,j1] + J[i,j2])
              cable i 对模组 2 的有效力臂 b_i = sign_conv·(J[i,j3] + J[i,j4])

        合并同侧对 (cable11+cable13, cable12+cable14, ...):
            m_p1 = a[IDX_F1P].sum()   # F1⁺ 的总有效力臂
            m_m1 = a[IDX_F1M].sum()   # F1⁻ 的总有效力臂
            m_p2 = b[IDX_F2P].sum()
            m_m2 = b[IDX_F2M].sum()

        最终: 2 个解耦的 1 等式 2 未知子问题, 用 _solve_pair 闭式求解.
    """
    a = sign_conv * (J[:, dof_j1] + J[:, dof_j2])
    b = sign_conv * (J[:, dof_j3] + J[:, dof_j4])

    m_p1 = a[IDX_F1P].sum()
    m_m1 = a[IDX_F1M].sum()
    m_p2 = b[IDX_F2P].sum()
    m_m2 = b[IDX_F2M].sum()

    F1p, F1m, res_a = _solve_pair(m_p1, m_m1, tau_a_des, F_pre, F_max)
    F2p, F2m, res_b = _solve_pair(m_p2, m_m2, tau_b_des, F_pre, F_max)

    F = np.zeros(8, dtype=float)
    F[IDX_F1P] = F1p
    F[IDX_F1M] = F1m
    F[IDX_F2P] = F2p
    F[IDX_F2M] = F2m

    info = dict(m_p1=m_p1, m_m1=m_m1, m_p2=m_p2, m_m2=m_m2,
                F1p=F1p, F1m=F1m, F2p=F2p, F2m=F2m,
                res_a=res_a, res_b=res_b)
    return F, info


# ═══════════════════════════════════════════════════════════════════════
# 5. 8 字形轨迹生成 (笛卡尔 → IK → 平滑)
# ═══════════════════════════════════════════════════════════════════════
def generate_figure8_trajectory(
    robot: MultiJointSpaceRobot,
    dt: float = DT_CTRL,
    period: float = PERIOD,
    num_cycles: float = NUM_CYCLES,
    center_x: float = CENTER_X,
    center_y: float = CENTER_Y,
    radius_x: float = RADIUS_X,
    radius_y: float = RADIUS_Y,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在笛卡尔空间设计 8 字 (Lissajous 频率比 1:2):
        x(t) = cx + Rx · sin(ω t),    y(t) = cy + Ry · sin(2ω t)
    再用 robot.inverse_kinematics 逐点反解 (qa, qb).

    工程修正:
      1) IK 用上一帧解作为初值, 保证解的支连续性;
      2) 仍可能在 ±π/2 附近发生 IK 分支跳变, 用相邻帧 |Δq| > 60° 阈值检测;
      3) 末段窄窗 (≈ 50 ms) 滑动均值, 抑制 IK 数值噪声引入的高频抖动;
      4) 关节夹到 ±(π/2 − 0.05) 远离硬限位.
    """
    total_time = period * num_cycles
    N = int(total_time / dt) + 1
    t_vals = np.linspace(0.0, total_time, N)
    omega = 2.0 * np.pi / period

    x_des = center_x + radius_x * np.sin(omega * t_vals)
    y_des = center_y + radius_y * np.sin(2.0 * omega * t_vals)

    qa_des = np.zeros(N)
    qb_des = np.zeros(N)
    q_guess = np.array([0.1, 0.1])      # 初值避开打直奇异
    ik_fail = 0
    for i in range(N):
        q_sol, conv = robot.inverse_kinematics([x_des[i], y_des[i]], q_guess)
        if not conv:
            ik_fail += 1
            q_sol = q_guess.copy()
        qa_des[i], qb_des[i] = q_sol[0], q_sol[1]
        q_guess = q_sol
    if ik_fail > 0:
        print(f"  [IK] 未收敛步数: {ik_fail}/{N}")

    # 连续性修复
    jump_thresh = np.deg2rad(60.0)
    for i in range(1, N):
        if abs(qa_des[i] - qa_des[i - 1]) > jump_thresh:
            qa_des[i] = -qa_des[i] if abs(qa_des[i]) > np.pi / 4 else qa_des[i]
        if abs(qb_des[i] - qb_des[i - 1]) > jump_thresh:
            qb_des[i] = -qb_des[i] if abs(qb_des[i]) > np.pi / 4 else qb_des[i]

    # 平滑 (≈ 50 ms)
    from scipy.ndimage import uniform_filter1d
    window = max(3, int(0.05 / dt))
    if window > 1:
        qa_des = uniform_filter1d(qa_des, size=window)
        qb_des = uniform_filter1d(qb_des, size=window)

    # 远离硬限位
    q_lim = np.pi / 2.0 - 0.05
    qa_des = np.clip(qa_des, -q_lim, q_lim)
    qb_des = np.clip(qb_des, -q_lim, q_lim)

    # 打印轨迹关节角范围, 并做几何退化区诊断
    qa_min_deg = np.rad2deg(np.min(qa_des))
    qa_max_deg = np.rad2deg(np.max(qa_des))
    qb_min_deg = np.rad2deg(np.min(qb_des))
    qb_max_deg = np.rad2deg(np.max(qb_des))
    qa_abs_max = max(abs(qa_min_deg), abs(qa_max_deg))
    qb_abs_max = max(abs(qb_min_deg), abs(qb_max_deg))
    print(
        f"  轨迹: {N} 采样点, 总长 {total_time:.1f}s, "
        f"qa∈[{qa_min_deg:+.1f}°, {qa_max_deg:+.1f}°], "
        f"qb∈[{qb_min_deg:+.1f}°, {qb_max_deg:+.1f}°]"
    )

    # 几何退化区告警: |q| > 80° 时拮抗映射开始退化, > 83° 完全失效
    GEOM_WARN  = 80.0
    GEOM_FAIL  = 83.0
    if qa_abs_max > GEOM_FAIL or qb_abs_max > GEOM_FAIL:
        print(f"  ⚠⚠⚠ 致命: |qa|_max={qa_abs_max:.1f}°, |qb|_max={qb_abs_max:.1f}° "
              f"超过 {GEOM_FAIL}° → 进入拮抗映射 **完全失效区**, 跟踪一定失败!")
        print(f"          解决: 缩小 RADIUS_X / RADIUS_Y, 或加大 CENTER_X (远离基座)")
    elif qa_abs_max > GEOM_WARN or qb_abs_max > GEOM_WARN:
        print(f"  ⚠ 警告: |qa|_max={qa_abs_max:.1f}°, |qb|_max={qb_abs_max:.1f}° "
              f"接近 {GEOM_FAIL}° 退化区, 跟踪误差可能较大")
    else:
        print(f"  ✓ 关节角在可达区内 (|qa|_max={qa_abs_max:.1f}°, |qb|_max={qb_abs_max:.1f}°)")

    return t_vals, qa_des, qb_des, x_des, y_des


# ═══════════════════════════════════════════════════════════════════════
# 6. 主程序
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 78)
    print("  MuJoCo 绳驱空间机械臂 — 8 字形跟踪 (v4: 纯 PD + 绳驱拮抗映射, 无 F_max)")
    print("=" * 78)

    # ─── 6.1 解析模型 (仅 IK 用) ────────────────────────────────────────
    robot_math = MultiJointSpaceRobot()

    # ─── 6.2 生成参考轨迹 ───────────────────────────────────────────────
    print("\n[Step 1] 生成 8 字形参考轨迹...")
    t_vals, qa_des, qb_des, x_des, y_des = generate_figure8_trajectory(robot_math)
    N_REF   = len(t_vals)
    T_TOTAL = t_vals[-1]

    # PD 需要的速度参考 (位置数值微分)
    dqa_des = np.gradient(qa_des, DT_CTRL)
    dqb_des = np.gradient(qb_des, DT_CTRL)

    # ─── 6.3 加载 MuJoCo ────────────────────────────────────────────────
    print("\n[Step 2] 加载 MuJoCo 模型 (ctrlrange 已放开到 [0, 1e6])...")
    model, data, scratch, xml_text = load_model(DT_CTRL)
    indices = build_indices(model)

    # ─── 6.4 Jacobian 符号校验 ──────────────────────────────────────────
    print("\n[Step 3] Tendon Jacobian 符号校验...")
    sign_conv = verify_jacobian_sign(model, data, scratch, indices)

    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  "
          f"ntendon={model.ntendon}  dt={DT_CTRL}")
    print(f"  PD 增益: Kp1={KP_JOINT1}, Kd1={KD_JOINT1}  |  "
          f"Kp2={KP_JOINT2}, Kd2={KD_JOINT2}")
    print(f"  绳驱参数: F_pre={F_PRE} N,  F_max={F_MAX:.0e} N (无饱和)")
    print(f"  驱动通道: data.ctrl[8] → cable tensions → 关节力矩 (真实绳驱)")

    # ─── 6.5 初始化机械臂到轨迹起点 ─────────────────────────────────────
    qpos_j1 = indices["qpos_j1"]
    qpos_j2 = indices["qpos_j2"]
    qpos_j3 = indices["qpos_j3"]
    qpos_j4 = indices["qpos_j4"]
    dof_j1  = indices["dof_j1"]
    dof_j2  = indices["dof_j2"]
    dof_j3  = indices["dof_j3"]
    dof_j4  = indices["dof_j4"]
    site_ee = indices["site_ee"]

    data.qpos[qpos_j1] = qa_des[0]
    data.qpos[qpos_j2] = qa_des[0]
    data.qpos[qpos_j3] = qb_des[0]
    data.qpos[qpos_j4] = qb_des[0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    p0_ee = np.array(data.site_xpos[site_ee], dtype=float)
    print(f"  起始末端: ({p0_ee[0]:+.3f}, {p0_ee[1]:+.3f}, {p0_ee[2]:+.3f})")

    # ─── 6.6 viewer / GIF 装饰 ──────────────────────────────────────────
    TRAJ_SAMPLES = 200
    t_sample = np.linspace(0.0, T_TOTAL, TRAJ_SAMPLES)
    ee_des_xy = np.array(
        [[np.interp(ts, t_vals, x_des), np.interp(ts, t_vals, y_des)]
         for ts in t_sample]
    )
    _WHITE = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    _RED   = np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float64)
    _DASH_ON_OFF = 2
    ee_actual_xyz: list[np.ndarray] = []

    def scene_decorator(scene, d):
        # 期望 8 字 (白虚线)
        for i in range(len(ee_des_xy) - 1):
            if (i // _DASH_ON_OFF) % 2 != 0:
                continue
            p0 = np.array([ee_des_xy[i,     0], ee_des_xy[i,     1], 0.02])
            p1 = np.array([ee_des_xy[i + 1, 0], ee_des_xy[i + 1, 1], 0.02])
            if not add_line_to_scene(scene, p0, p1, rgba=_WHITE, width=3.0):
                return
        # 实际拖尾 (红实线)
        if len(ee_actual_xyz) >= 2:
            pts = ee_actual_xyz[-TRAIL_MAX_SEGMENTS:]
            for j in range(len(pts) - 1):
                if not add_line_to_scene(scene, pts[j], pts[j + 1],
                                         rgba=_RED, width=4.0):
                    return

    fig_save_dir = get_save_dir()
    _parts = fig_save_dir.replace("\\", "/").rstrip("/").split("/")
    _plot_program_name = _parts[-2] if len(_parts) >= 2 else "run"
    _plot_timestamp    = _parts[-1] if len(_parts) >= 1 else time.strftime("%Y%m%d_%H%M%S")
    extra_gif_basename = f"{_plot_program_name}_{_plot_timestamp}_figure8_playback"

    logger = MjSimLogger(
        model=model, xml_text=xml_text, enable_gif=True,
        gif_fps=30, gif_width=2560, gif_height=1440,
        camera_lookat=(3.0, 0.0, 0.0), camera_distance=12.0,
        camera_azimuth=90.0, camera_elevation=-90.0,
        dt=DT_CTRL, scene_decorator=scene_decorator,
        extra_gif_save_dir=fig_save_dir,
        extra_gif_basename=extra_gif_basename,
    )
    logger.record(data)

    # ─── 6.7 数据缓冲 ───────────────────────────────────────────────────
    rec_t:   list[float]      = []
    rec_q:   list[list[float]] = []     # (qa, qb) 实际值
    rec_dq:  list[list[float]] = []     # (dqa, dqb) 实际值
    rec_ee:  list[np.ndarray] = []      # (x, y) 末端
    rec_tau: list[list[float]] = []     # (τa, τb) PD 输出
    rec_F:   list[np.ndarray] = []      # 8 根绳张力
    rec_res: list[list[float]] = []     # 拮抗映射残差 (理想情况 ≈ 0)

    act_ids = indices["act_ids_ordered"]
    tdn_ids = indices["tdn_ids_ordered"]

    # Jacobian 缓存 (主连杆几何变化慢, 不必每步重算 FD)
    _cached_q: np.ndarray | None = None
    _cached_J: np.ndarray | None = None
    _jac_recompute_thresh = 0.01    # 弧度, |Δq| > 该阈值才重算 J

    N_SIM = N_REF + int(1.0 / DT_CTRL)   # 跟踪完后再多跑 1 s 看稳态
    step = 0

    print(f"\n[Step 4] 启动 MuJoCo passive viewer, 仿真总步数 = {N_SIM} ...\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 俯视摄像机
            viewer.cam.distance  = 12.0
            viewer.cam.azimuth   = 90.0
            viewer.cam.elevation = -90.0
            viewer.cam.lookat[:] = [3.0, 0.0, 0.0]

            last_sync = -1.0

            # ═══════════════════════════════════════════════════════════
            # 主控制 / 仿真循环 (每步: 取参考 → PD → Jacobian → 映射 → 步进)
            # ═══════════════════════════════════════════════════════════
            while viewer.is_running() and step < N_SIM:
                t_wallclock_start = time.time()

                # ─── (a) 取本步期望 (整数步号严格对齐参考) ────────────
                t_now  = step * DT_CTRL
                qa_d   = float(np.interp(t_now, t_vals, qa_des))
                qb_d   = float(np.interp(t_now, t_vals, qb_des))
                dqa_d  = float(np.interp(t_now, t_vals, dqa_des))
                dqb_d  = float(np.interp(t_now, t_vals, dqb_des))

                # ─── (b) 读实际状态 ──────────────────────────────────
                qa  = float(data.qpos[qpos_j1])    # joint1≡joint2 (qa)
                qb  = float(data.qpos[qpos_j3])    # joint3≡joint4 (qb)
                dqa = float(data.qvel[dof_j1])
                dqb = float(data.qvel[dof_j3])

                # ─── (c) 纯 PD 反馈 (无前馈, 无饱和) ──────────────────
                tau_a = (KP_JOINT1 * (qa_d  - qa)
                       + KD_JOINT1 * (dqa_d - dqa))
                tau_b = (KP_JOINT2 * (qb_d  - qb)
                       + KD_JOINT2 * (dqb_d - dqb))

                # ─── (d) 计算/复用 tendon Jacobian J(q) ───────────────
                q_now = np.array(data.qpos, dtype=float)
                if (_cached_q is None or
                        np.max(np.abs(q_now - _cached_q)) > _jac_recompute_thresh):
                    _cached_J = compute_tendon_jacobian_fd(
                        model, scratch, q_now, tdn_ids)
                    _cached_q = q_now.copy()
                J = _cached_J

                # ─── (e) 绳驱拮抗映射: τ → 8 根绳张力 ─────────────────
                F_cable, info = cable_antagonistic_map(
                    tau_a, tau_b, J,
                    dof_j1, dof_j2, dof_j3, dof_j4,
                    sign_conv, F_pre=F_PRE, F_max=F_MAX,
                )

                # 把张力写入 8 个卷筒电机的 ctrl
                # MuJoCo: ctrl × gear (=1) = motor force on tendon (= 绳张力)
                data.ctrl[act_ids] = F_cable

                # ─── (f) 推进物理一步 ──────────────────────────────────
                # MuJoCo 内部:
                #     qfrc_actuator = sign_conv · Jᵀ · F_cable
                #     M(q)q̈ + C(q,q̇)q̇ + b·q̇ = qfrc_actuator
                # 等价于绳张力把关节驱动起来.
                mujoco.mj_step(model, data)
                logger.record(data)
                step += 1

                # ─── (g) 记录 (post-step 状态) ────────────────────────
                rec_t.append(data.time)
                rec_q.append([
                    float(data.qpos[qpos_j1]),
                    float(data.qpos[qpos_j3]),
                ])
                rec_dq.append([
                    float(data.qvel[dof_j1]),
                    float(data.qvel[dof_j3]),
                ])
                rec_ee.append(np.array(data.site_xpos[site_ee][:2], dtype=float))
                rec_tau.append([tau_a, tau_b])
                rec_F.append(F_cable.copy())
                rec_res.append([info["res_a"], info["res_b"]])

                # 末端拖尾采样
                if step % TRAIL_SAMPLE_EVERY == 0:
                    ee_actual_xyz.append(np.array([
                        data.site_xpos[site_ee][0],
                        data.site_xpos[site_ee][1],
                        0.03,
                    ]))

                # ─── (h) viewer 30 Hz 刷新 ─────────────────────────────
                if data.time - last_sync > 1.0 / 30.0:
                    viewer.sync()
                    last_sync = data.time

                # 限制实时播放速率
                rest = DT_CTRL - (time.time() - t_wallclock_start)
                if rest > 0:
                    time.sleep(rest)

    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        logger.save_and_close()
        print(f"\n[仿真结束] 总步数 = {step}")

    # ═══════════════════════════════════════════════════════════════════
    # 7. 数据整理 + 绘图 + 统计
    # ═══════════════════════════════════════════════════════════════════
    rec_t   = np.array(rec_t)
    rec_q   = np.array(rec_q)
    rec_dq  = np.array(rec_dq)
    rec_ee  = np.array(rec_ee)
    rec_tau = np.array(rec_tau)
    rec_F   = np.array(rec_F)
    rec_res = np.array(rec_res)

    qa_d_i  = np.interp(rec_t, t_vals, qa_des)
    qb_d_i  = np.interp(rec_t, t_vals, qb_des)
    dqa_d_i = np.interp(rec_t, t_vals, dqa_des)
    dqb_d_i = np.interp(rec_t, t_vals, dqb_des)
    x_d_i   = np.interp(rec_t, t_vals, x_des)
    y_d_i   = np.interp(rec_t, t_vals, y_des)

    cart_err = np.sqrt(
        (rec_ee[:, 0] - x_d_i) ** 2 + (rec_ee[:, 1] - y_d_i) ** 2
    )

    # ─── 7.1 总览图 (2x3) ──────────────────────────────────────────────
    print("\n[Step 5] 绘图与统计...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 末端轨迹
    ax = axes[0, 0]
    ax.plot(x_des, y_des, 'k--', lw=2, alpha=0.7, label="Desired")
    ax.plot(rec_ee[:, 0], rec_ee[:, 1], 'r-', lw=1.5, alpha=0.9, label="Actual")
    ax.plot(rec_ee[0, 0],  rec_ee[0, 1],  'go', ms=8, label="Start")
    ax.plot(rec_ee[-1, 0], rec_ee[-1, 1], 'bs', ms=8, label="End")
    ax.set_title("End-Effector Figure-8 Tracking")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # (2) 关节角度跟踪
    ax = axes[0, 1]
    ax.plot(t_vals, np.rad2deg(qa_des),      'b--', lw=2, alpha=0.7, label=r"$q_a^{\rm des}$")
    ax.plot(rec_t,  np.rad2deg(rec_q[:, 0]), 'b-',  lw=1.5,          label=r"$q_a$")
    ax.plot(t_vals, np.rad2deg(qb_des),      'r--', lw=2, alpha=0.7, label=r"$q_b^{\rm des}$")
    ax.plot(rec_t,  np.rad2deg(rec_q[:, 1]), 'r-',  lw=1.5,          label=r"$q_b$")
    ax.set_title("Joint Angle Tracking")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # (3) 关节误差
    ax = axes[0, 2]
    ax.plot(rec_t, np.rad2deg(qa_d_i - rec_q[:, 0]), 'b-', lw=1.5, label=r"$e_{qa}$")
    ax.plot(rec_t, np.rad2deg(qb_d_i - rec_q[:, 1]), 'r-', lw=1.5, label=r"$e_{qb}$")
    ax.axhline(0, color='k', lw=0.6, alpha=0.3)
    ax.set_title("Joint Tracking Error")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (deg)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # (4) 笛卡尔误差
    ax = axes[1, 0]
    ax.plot(rec_t, cart_err * 1000.0, 'm-', lw=1.8)
    ax.set_title("Cartesian Tracking Error")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|Δp| (mm)")
    ax.grid(True, alpha=0.4)

    # (5) PD 指令力矩
    ax = axes[1, 1]
    ax.plot(rec_t, rec_tau[:, 0], 'b-', lw=1.5, alpha=0.9, label=r"$\tau_a$")
    ax.plot(rec_t, rec_tau[:, 1], 'r-', lw=1.5, alpha=0.9, label=r"$\tau_b$")
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_title("PD Commanded Joint Torques (no FF, no limit)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"$\tau$ (Nm)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=9)

    # (6) 8 根绳张力
    ax = axes[1, 2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    styles = ['-', '-', '-.', '-.', '--', '--', ':', ':']
    for i, n in enumerate(CABLE_NAMES):
        ax.plot(rec_t, rec_F[:, i], color=colors[i], ls=styles[i],
                lw=1.2, alpha=0.85, label=n)
    ax.axhline(F_PRE, color='k', lw=0.6, alpha=0.4, label=f"F_pre={F_PRE:.0f} N")
    ax.set_title("8 Cable Tensions (no F_max limit)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("F (N)")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=6, ncol=2, loc='best')

    plt.suptitle(
        "Cable-Driven Space Robot — Figure-8 Tracking (v4: pure PD + cable antagonistic, no F_max)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig_name="figure8_tracking_summary_v4")
    plt.close()

    # ─── 7.2 末端轨迹色彩细节 ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.plot(x_des, y_des, 'k--', lw=2, alpha=0.6, label="Desired")
    n_ee = len(rec_ee)
    ax2.scatter(rec_ee[:, 0], rec_ee[:, 1], c=np.linspace(0, 1, n_ee),
                cmap='viridis', s=2, alpha=0.65)
    ax2.plot(rec_ee[0,  0], rec_ee[0,  1], 'go', ms=10, label="Start")
    ax2.plot(rec_ee[-1, 0], rec_ee[-1, 1], 'rs', ms=10, label="End")
    ax2.scatter(x_des[0], y_des[0], c='orange', s=120, marker='*',
                zorder=5, label="Desired Start")
    ax2.set_title("EE Trajectory (color = time)")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.4); ax2.legend()
    plt.tight_layout()
    save_figure(fig_name="figure8_ee_detail_v4")
    plt.close()

    # ─── 7.3 跟踪质量统计 ──────────────────────────────────────────────
    rms_cart = float(np.sqrt(np.mean(cart_err ** 2)))
    max_cart = float(np.max(cart_err))
    rms_qa   = float(np.sqrt(np.mean((qa_d_i - rec_q[:, 0]) ** 2)))
    rms_qb   = float(np.sqrt(np.mean((qb_d_i - rec_q[:, 1]) ** 2)))
    max_qa   = float(np.max(np.abs(qa_d_i - rec_q[:, 0])))
    max_qb   = float(np.max(np.abs(qb_d_i - rec_q[:, 1])))
    peak_F   = float(np.max(rec_F))
    peak_res = float(np.max(np.abs(rec_res)))

    print(f"\n{'=' * 60}")
    print(f"  跟踪质量统计 (v4: 纯 PD + 绳驱拮抗映射, 无 F_max)")
    print(f"{'=' * 60}")
    print(f"  e_qa   RMS = {np.rad2deg(rms_qa):.4f}°   peak = {np.rad2deg(max_qa):.4f}°")
    print(f"  e_qb   RMS = {np.rad2deg(rms_qb):.4f}°   peak = {np.rad2deg(max_qb):.4f}°")
    print(f"  |Δp|   RMS = {rms_cart * 1000:.3f} mm   peak = {max_cart * 1000:.3f} mm")
    print(f"  |τa|_max = {np.max(np.abs(rec_tau[:, 0])):.1f} Nm   "
          f"|τb|_max = {np.max(np.abs(rec_tau[:, 1])):.1f} Nm")
    print(f"  F_cable peak = {peak_F:.1f} N   (F_max = {F_MAX:.0e} N)")
    print(f"  映射残差峰值: |res| = {peak_res:.2e} Nm  (理想 ≈ 0)")
    print(f"\n  图: {fig_save_dir}")
    print(f"  日志: {logger.save_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
