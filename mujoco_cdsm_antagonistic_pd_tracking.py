"""
test_mujoco_cdsm_antagonistic_pd_tracking.py
=============================================
关节空间 纯 PD 轨迹跟踪 ———— 真实 8 绳拮抗驱动 (via data.ctrl)
  (模型: multi_joint_cable_dirven_space_robot.xml, gear=1, ctrlrange=[0, F_MAX] N)

本脚本是 `test_mujoco_direct_torque_pd_tracking.py` 的 "真实绳驱" 对应版本:
  direct (基线): PD -> τa, τb -> 直接写 data.qfrc_applied  (绕过绳索, 理想力矩源)
  本脚本        : PD -> τa, τb -> 拮抗映射器 -> 8 根绳张力 -> data.ctrl (经 MuJoCo 绳索层)

映射器核心 (见 test_mujoco_cable_jacobian.py 的校验):
  MuJoCo 内部传递:    qfrc_actuator = (+1) · J_tendon^T · F_ctrl
                      (经验测定, |A-J|_∞ < 1e-9)
  同侧对合并:         f11 = f13 = F1p,   f12 = f14 = F1m
                      f21 = f23 = F2p,   f22 = f24 = F2m
  力矩聚合到模组:     τ_a = qfrc[j1] + qfrc[j2],  τ_b = qfrc[j3] + qfrc[j4]
  由此得块对角 2×4 矩阵 M(q), 形如:
            [τ_a]   [ m_p1   m_m1     0      0  ] [F1p]
            [   ] = [                             ] [F1m]
            [τ_b]   [   0      0    m_p2   m_m2 ] [F2p]
                                                    [F2m]
  于是求解自动分解为两个 1×2 子问题, 每个子问题闭式可解:
     minimize F+ + F-
     s.t.  m_p·F+ + m_m·F- = τ_des
           F+, F- ∈ [F_pre, F_max]

  策略 (solve_pair): 在 "只 F+ 带力" / "只 F- 带力" / 双侧预紧 三种备选中,
     挑 残差最小; 残差并列时挑附加张力最小. 该贪心策略对
       - 反号 M 列 (正常差动区):          闭式给出 1 侧精确解, 另一侧保持 F_pre;
       - 同号 M 列 (qa → ±90° 几何退化):   若方向匹配仍能出力, 否则输出双侧 F_pre,
                                           即承担不可避免的 "几何寄生力矩 + 饱和".


════════════════════════════════════════════════════════════════════════════
  ⚠  重要提醒 —— 本 XML 绳驱几何的 "内禀可达区" 限制                           
════════════════════════════════════════════════════════════════════════════

当前 MuJoCo 模型 `multi_joint_cable_dirven_space_robot.xml` 的 spreader 布线
(spreader 中轴长 0.1 m, 上下 tips 位于 ±0.3 m), 在 **|qa| 或 |qb| 超过约
±83° 之后**, 拮抗映射矩阵 M(q) 的**两列变成同号**, 出现几何不可达区:

    - qa → +90° :  M1 两列均为负  (m_p1 < 0, m_m1 < 0)
                   → 任何正 PD 命令都被 preload 寄生力矩反向推离 +90°
    - qb → -90° :  M2 两列均为正  (m_p2 > 0, m_m2 > 0)
                   → PD 想拉 qb 到 -90° 时 cable 只能输出双侧预紧,
                     寄生力矩把 qb 顶回约 -84°

这是 **"绳索布线 → 可达工作空间"** 的**内禀几何限制**, 与映射器策略 / PD
增益 / 前馈补偿 **均无关**. 具体表现为:
    - hold 段目标若落在 ±90° 附近, 系统会 "反向漂移" 并在末端画出一段乱轨迹;
    - |q| < ~80° 区域 M 矩阵列反号, 拮抗映射可以闭式给出准确张力, 工作正常.

→ **使用者请据此合理设置 QA_START/QA_END/QB_START/QB_END 的目标幅度**:
  1. 若要做常规跟踪 / 验证, 建议把目标设在 **±80° 以内** (留 3° 稳定裕度),
     这样整条轨迹都在可达区, 本脚本的纯 PD 即可稳定跟踪.
  2. 若刻意要 **暴露这个几何退化**, 可以把目标设为 ±90°, 观察 hold 段
     如何被 preload 寄生力矩顶回, 图中会清楚看到 cable 版收敛不到端点
     而 direct_torque 基线版能收敛到端点 —— 两者对比就是"几何限制"的量化.
  3. 若要让 cable 版也能精确抵达 ±90°, **必须改 XML** 的 spreader 几何
     (加长 spreader 臂 / 调整入口点) 以扩大可达工作空间, 本脚本本身无法修复.

════════════════════════════════════════════════════════════════════════════

运行:
    python mujoco_cdsm_antagonistic_pd_tracking.py
"""
from __future__ import annotations

import os
import sys
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


XML_PATH = "multi_joint_cable_dirven_space_robot.xml"

CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]

# --------------------------------------------------------------------------
# 同侧对合并 (见 test_mujoco_cable_jacobian.py 的定义):
#   Module 1 (qa):  F1p := f11 = f13    F1m := f12 = f14
#   Module 2 (qb):  F2p := f21 = f23    F2m := f22 = f24
# 相关下标映射到 CABLE_NAMES 的行索引:
# --------------------------------------------------------------------------
IDX_F1P = [0, 2]    # cable11, cable13
IDX_F1M = [1, 3]    # cable12, cable14
IDX_F2P = [4, 6]    # cable21, cable23
IDX_F2M = [5, 7]    # cable22, cable24

# --------------------------------------------------------------------------
# PD 控制参数 (与 direct_torque 版本保持一致, 方便比对 RMS / 峰值误差)
# --------------------------------------------------------------------------
KP_JOINT1 = 3500.0
KD_JOINT1 = 500.0
KP_JOINT2 = 1500.0
KD_JOINT2 = 200.0

# --------------------------------------------------------------------------
# 绳驱参数
# --------------------------------------------------------------------------
F_PRE   = 20.0       # N, 预紧力下限 (每根绳在任何时刻都 >= F_PRE)
F_MAX   = 2000.0     # N, 每根绳上限 (XML ctrlrange=[0, 2000], gear=1)

# --------------------------------------------------------------------------
# 轨迹目标 (用户自由配置; 请先阅读 docstring 顶端的 "⚠ 内禀可达区警告")
#   - QA_START/END : joint1 = joint2 = qa 的起点 / 终点 (rad)
#   - QB_START/END : joint3 = joint4 = qb 的起点 / 终点 (rad)
# 推荐幅度: ±80° 以内 (本 XML 的绳索几何可达区); ±90° 会在 hold 段暴露几何退化.
# --------------------------------------------------------------------------
QA_START = -np.pi / 2.0
QA_END   =  np.pi / 2.0
QB_START =  np.pi / 2.0
QB_END   = -np.pi / 2.0

# 仿真 / 轨迹 / 控制周期三者严格一致
DT_CTRL = 0.01
T_RAMP  = 50.0
T_HOLD  = 0.0
T_TOTAL = T_RAMP + T_HOLD

TRAJ_SAMPLES_DESIRED = 160
DASH_ON_OFF = 2
TRAIL_SAMPLE_EVERY = 4
TRAIL_MAX_SEGMENTS = max(
    200,
    int(2.0 * (T_TOTAL + 1.0) / (DT_CTRL * TRAIL_SAMPLE_EVERY)),
)


# ============================================================================
# 1. 轨迹规划 (位置 + 速度, 供纯 PD 使用)
# ============================================================================
def _cosine_ramp(t: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    tau = np.clip(t / T, 0.0, 1.0)
    s   = 0.5 * (1.0 - np.cos(np.pi * tau))
    ds_in = 0.5 * np.pi / T * np.sin(np.pi * tau)
    mask = (t > 0) & (t < T)
    ds  = np.where(mask, ds_in, 0.0)
    return s, ds


def build_joint_reference(dt: float):
    t_vals = np.arange(0.0, T_TOTAL + dt * 0.5, dt)
    s, ds = _cosine_ramp(t_vals, T_RAMP)

    qa_des  = QA_START + (QA_END - QA_START) * s
    qb_des  = QB_START + (QB_END - QB_START) * s
    dqa_des = (QA_END - QA_START) * ds
    dqb_des = (QB_END - QB_START) * ds
    return t_vals, qa_des, qb_des, dqa_des, dqb_des


# ============================================================================
# 2. XML 加载 (保留 ctrlrange=[0, F_MAX], ctrllimited=true: 模拟真实伺服上限)
# ============================================================================
def load_model():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()
    # 放宽关节硬限位, 使目标 ±90° 不贴死区
    xml_str = re.sub(
        r'range="-1\.5708 1\.5708"',
        'range="-1.7 1.7"',
        xml_str,
    )
    # 同步 ctrlrange 上限到 F_MAX (XML 默认也是 2000)
    xml_str = re.sub(
        r'ctrlrange="0\s+2000"',
        f'ctrlrange="0 {F_MAX:g}"',
        xml_str,
    )
    # 对齐仿真步长 = 轨迹采样周期 = PD 控制周期
    xml_str = re.sub(
        r'timestep="[^"]*"',
        f'timestep="{DT_CTRL:g}"',
        xml_str,
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    return model, data


def build_indices(model):
    tdn_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,   n) for n in CABLE_NAMES}
    act_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
    jnt_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
              for n in ("joint1", "joint2", "joint3", "joint4")}
    site_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    for name, idx in list(tdn_id.items()) + list(act_id.items()) + list(jnt_id.items()):
        if idx < 0:
            raise RuntimeError(f"[模型自检失败] 未找到 {name!r}")
    if site_ee < 0:
        raise RuntimeError("[模型自检失败] 未找到 site 'end_effector'")
    return tdn_id, act_id, jnt_id, site_ee


# ============================================================================
# 3. 绳索 Jacobian —— 在 scratch MjData 上做有限差分, 不污染主仿真状态
# ============================================================================
def compute_tendon_jacobian_fd(
    model,
    scratch: mujoco.MjData,
    q_ref: np.ndarray,
    tendon_ids_ordered: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    返回 (8, nv) 的 tendon Jacobian, 行顺序 = CABLE_NAMES.
    只使用 scratch (副本) 的 qpos, 不修改主 data.
    """
    # 关键: 这里必须使用 mj_fwdPosition 而非 mj_forward.
    #   mj_forward 会调用 mj_fwdActuation, 后者会触发全局注册的 mjcb_control,
    #   而本函数本身是在 control_callback 里被调用的 -> 无限递归.
    #   mj_fwdPosition 只跑位置管线 (kinematics + comPos + tendon + ...), 不调 control,
    #   刚好够算 ten_length, 且不污染主 data / 不触发回调.
    nv = model.nv
    J = np.zeros((len(tendon_ids_ordered), nv), dtype=float)
    for j in range(nv):
        scratch.qpos[:] = q_ref
        scratch.qpos[j] = q_ref[j] + eps
        mujoco.mj_fwdPosition(model, scratch)
        L_plus = np.array(scratch.ten_length, dtype=float)[tendon_ids_ordered].copy()
        scratch.qpos[:] = q_ref
        scratch.qpos[j] = q_ref[j] - eps
        mujoco.mj_fwdPosition(model, scratch)
        L_minus = np.array(scratch.ten_length, dtype=float)[tendon_ids_ordered].copy()
        J[:, j] = (L_plus - L_minus) / (2.0 * eps)
    return J


# ============================================================================
# 4. 拮抗映射器: (τ_a_des, τ_b_des, J, q)  ->  8 根绳张力
# ============================================================================
def _solve_pair(m_p: float, m_m: float, tau_des: float,
                F_pre: float, F_max: float) -> tuple[float, float, float]:
    """
    求解单个模组的 1 等式 2 未知子问题:
        m_p · F+ + m_m · F- = tau_des,    F+, F- ∈ [F_pre, F_max]
    策略: 在 "+ 侧带力" / "- 侧带力" / 双侧预紧 三种候选里选残差最小 (次
         序并列则附加张力最小).

    返回: (F_plus, F_minus, residual)   residual = 实际达到的 τ 与 τ_des 的差.
    """
    u_max = F_max - F_pre                       # 附加张力上限
    tau_base = (m_p + m_m) * F_pre              # 双侧预紧自然产生的 "寄生力矩"
    tau_eff  = tau_des - tau_base               # 还需额外生成的力矩

    # 枚举候选 (u_p, u_m, residual, tension_l1)
    EPS = 1e-12
    cand: list[tuple[float, float, float, float]] = []

    # (0) 默认: 双侧预紧 (不出力)
    cand.append((0.0, 0.0, abs(tau_eff), 0.0))

    # (1) 所有附加张力放在 + 侧
    if abs(m_p) > EPS:
        u = tau_eff / m_p
        if u < 0.0:
            u = 0.0                              # 符号不匹配 -> 退到双侧预紧
        u_clip = min(u, u_max)
        residual = abs(tau_eff - m_p * u_clip)
        cand.append((u_clip, 0.0, residual, u_clip))

    # (2) 所有附加张力放在 - 侧
    if abs(m_m) > EPS:
        u = tau_eff / m_m
        if u < 0.0:
            u = 0.0
        u_clip = min(u, u_max)
        residual = abs(tau_eff - m_m * u_clip)
        cand.append((0.0, u_clip, residual, u_clip))

    u_p, u_m, res, _ = min(cand, key=lambda c: (c[2], c[3]))
    return F_pre + u_p, F_pre + u_m, res


def cable_antagonistic_map(
    tau_a_des: float,
    tau_b_des: float,
    J: np.ndarray,                # (8, nv)
    dof_j1: int, dof_j2: int, dof_j3: int, dof_j4: int,
    F_pre: float,
    F_max: float,
) -> tuple[np.ndarray, dict]:
    """
    把期望关节力矩映射到 8 根绳的张力.
    返回:
        F_cable  shape (8,)  按 CABLE_NAMES 顺序
        info     dict, 含 {m_p1, m_m1, m_p2, m_m2, F1p, F1m, F2p, F2m, res_a, res_b}
    """
    # 单根绳对 (τ_a, τ_b) 的有效力臂 (sign_conv = +1)
    a = J[:, dof_j1] + J[:, dof_j2]    # 8,
    b = J[:, dof_j3] + J[:, dof_j4]

    # 合并同侧对 -> 2×4 矩阵的列分量
    m_p1 = a[IDX_F1P[0]] + a[IDX_F1P[1]]
    m_m1 = a[IDX_F1M[0]] + a[IDX_F1M[1]]
    m_p2 = b[IDX_F2P[0]] + b[IDX_F2P[1]]
    m_m2 = b[IDX_F2M[0]] + b[IDX_F2M[1]]

    F1p, F1m, res_a = _solve_pair(m_p1, m_m1, tau_a_des, F_pre, F_max)
    F2p, F2m, res_b = _solve_pair(m_p2, m_m2, tau_b_des, F_pre, F_max)

    F = np.empty(8, dtype=float)
    F[IDX_F1P] = F1p
    F[IDX_F1M] = F1m
    F[IDX_F2P] = F2p
    F[IDX_F2M] = F2m

    info = dict(m_p1=m_p1, m_m1=m_m1, m_p2=m_p2, m_m2=m_m2,
                F1p=F1p, F1m=F1m, F2p=F2p, F2m=F2m,
                res_a=res_a, res_b=res_b)
    return F, info


# ============================================================================
# 5. 可视化: 白色虚线 (期望 EE)  +  红色实线 (实时 EE)
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


# ============================================================================
# 6. 主循环
# ============================================================================
def main():
    print("=" * 78)
    print("  MuJoCo 绳驱机械臂 —— 关节空间 纯 PD + 同侧对拮抗映射 (真实绳驱版本)")
    print("=" * 78)

    robot_math = MultiJointSpaceRobot()
    model, data = load_model()
    scratch = mujoco.MjData(model)             # 独立 scratch, 仅给 FD Jacobian 用
    tdn_id, act_id, jnt_id, site_ee = build_indices(model)
    dt = float(model.opt.timestep)
    assert abs(dt - DT_CTRL) < 1e-9, (
        f"[致命] MuJoCo timestep ({dt}) 与 DT_CTRL ({DT_CTRL}) 不一致"
    )

    print(f"  nq={model.nq}  nu={model.nu}  neq={model.neq}  ntendon={model.ntendon}  dt={dt}")
    print(f"  模组1 (qa) : Kp={KP_JOINT1} Nm/rad, Kd={KD_JOINT1} Nm·s/rad")
    print(f"  模组2 (qb) : Kp={KP_JOINT2} Nm/rad, Kd={KD_JOINT2} Nm·s/rad")
    gear_col = float(model.actuator_gear[act_id["winch_c11"], 0])
    ctrl_hi  = float(model.actuator_ctrlrange[act_id["winch_c11"], 1])
    is_lim   = bool(model.actuator_ctrllimited[act_id["winch_c11"]])
    print(f"  绳驱参数   : F_pre={F_PRE} N, F_max={F_MAX} N, gear={gear_col}, "
          f"ctrllimited={is_lim}, ctrlrange=[0, {ctrl_hi:.0f}] N")
    print(f"  轨迹目标   : qa {np.rad2deg(QA_START):+.1f}° → {np.rad2deg(QA_END):+.1f}°, "
          f"qb {np.rad2deg(QB_START):+.1f}° → {np.rad2deg(QB_END):+.1f}°")
    print(f"  ⚠ 注意: 目标 |q| 接近 ±90° 时会落入绳驱几何不可达区, 详见本文件顶端 docstring")

    # ---- 6.1 关节参考轨迹 ----
    t_vals, qa_des, qb_des, dqa_des, dqb_des = build_joint_reference(dt)
    N_REF = len(t_vals)
    print(f"  期望轨迹: T_ramp={T_RAMP}s + T_hold={T_HOLD}s, "
          f"dt={dt}s, 共 N_REF={N_REF} 个采样点")

    # ---- 6.2 期望末端轨迹 ----
    ee_xy = np.zeros((TRAJ_SAMPLES_DESIRED, 2))
    t_sample = np.linspace(0.0, T_TOTAL, TRAJ_SAMPLES_DESIRED)
    qa_sample = np.interp(t_sample, t_vals, qa_des)
    qb_sample = np.interp(t_sample, t_vals, qb_des)
    for i, (qa_i, qb_i) in enumerate(zip(qa_sample, qb_sample)):
        p5 = robot_math.forward_kinematics(qa_i, qb_i)[-1]
        ee_xy[i] = [p5[0], p5[1]]
    X_des_full = np.zeros_like(t_vals)
    Y_des_full = np.zeros_like(t_vals)
    for i, (qa_i, qb_i) in enumerate(zip(qa_des, qb_des)):
        p5 = robot_math.forward_kinematics(qa_i, qb_i)[-1]
        X_des_full[i] = p5[0]
        Y_des_full[i] = p5[1]

    # ---- 6.3 初始化到起始构型 ----
    data.qpos[model.jnt_qposadr[jnt_id["joint1"]]] = QA_START
    data.qpos[model.jnt_qposadr[jnt_id["joint2"]]] = QA_START
    data.qpos[model.jnt_qposadr[jnt_id["joint3"]]] = QB_START
    data.qpos[model.jnt_qposadr[jnt_id["joint4"]]] = QB_START
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    p0_ee = np.array(data.site_xpos[site_ee], dtype=float)
    print(f"  起始末端位置 (世界系): ({p0_ee[0]:+.3f}, {p0_ee[1]:+.3f}, {p0_ee[2]:+.3f})")

    # ---- 6.4 控制回调 (纯 PD + 拮抗映射 + data.ctrl) ----
    act_ids_ordered = np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int)
    tdn_ids_ordered = np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int)
    dof_j1 = int(model.jnt_dofadr[jnt_id["joint1"]])
    dof_j2 = int(model.jnt_dofadr[jnt_id["joint2"]])
    dof_j3 = int(model.jnt_dofadr[jnt_id["joint3"]])
    dof_j4 = int(model.jnt_dofadr[jnt_id["joint4"]])

    peak_tau = {"a": 0.0, "b": 0.0}
    peak_res = {"a": 0.0, "b": 0.0}                 # 映射器残差峰值
    peak_F   = {"F": 0.0}
    saturate_count = {"n": 0}                       # 映射残差 > 1e-3 的步数

    def control_callback(m, d):
        # (a) 参考点: 步号索引
        k = int(round(d.time / dt))
        if k < 0:
            k = 0
        elif k >= N_REF:
            k = N_REF - 1
        qa_d,  qb_d  = qa_des[k],  qb_des[k]
        dqa_d, dqb_d = dqa_des[k], dqb_des[k]

        # (b) 当前状态 + 纯 PD
        qa  = float(d.qpos[m.jnt_qposadr[jnt_id["joint1"]]])
        qb  = float(d.qpos[m.jnt_qposadr[jnt_id["joint3"]]])
        dqa = float(d.qvel[dof_j1])
        dqb = float(d.qvel[dof_j3])
        tau_a = KP_JOINT1 * (qa_d - qa) + KD_JOINT1 * (dqa_d - dqa)
        tau_b = KP_JOINT2 * (qb_d - qb) + KD_JOINT2 * (dqb_d - dqb)

        peak_tau["a"] = max(peak_tau["a"], abs(tau_a))
        peak_tau["b"] = max(peak_tau["b"], abs(tau_b))

        # (c) FD 绳索 Jacobian (在 scratch 上, 使用 mj_fwdPosition 避免回调递归)
        J = compute_tendon_jacobian_fd(m, scratch, np.array(d.qpos), tdn_ids_ordered)

        # (d) 拮抗映射 -> 8 根绳张力
        F_cable, info = cable_antagonistic_map(
            tau_a, tau_b, J,
            dof_j1, dof_j2, dof_j3, dof_j4,
            F_PRE, F_MAX,
        )
        d.ctrl[act_ids_ordered] = F_cable

        peak_res["a"] = max(peak_res["a"], info["res_a"])
        peak_res["b"] = max(peak_res["b"], info["res_b"])
        peak_F["F"]   = max(peak_F["F"], float(F_cable.max()))
        if info["res_a"] > 1e-3 or info["res_b"] > 1e-3:
            saturate_count["n"] += 1

    mujoco.set_mjcb_control(control_callback)

    # ---- 6.5 仿真主循环 ----
    record_t, record_q, record_dq, record_p, record_F, record_res = [], [], [], [], [], []
    ee_actual_xyz: list[np.ndarray] = [p0_ee.copy()]
    step = 0

    print("\n  启动 MuJoCo passive viewer. 关闭窗口或 Ctrl+C 退出仿真.\n")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 12.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -90
            viewer.cam.lookat[:] = [3.0, 0.0, 0.0]

            N_HOLD_EXTRA = int(round(1.0 / dt))
            N_SIM = N_REF + N_HOLD_EXTRA
            print(f"  仿真总步数 N_SIM={N_SIM}  (参考轨迹 {N_REF} 步 + 稳态 {N_HOLD_EXTRA} 步)")
            last_sync = -1.0

            while viewer.is_running() and step < N_SIM:
                step_start = time.time()
                mujoco.mj_step(model, data)
                step += 1

                record_t.append(data.time)
                record_q.append([
                    data.qpos[model.jnt_qposadr[jnt_id["joint1"]]],
                    data.qpos[model.jnt_qposadr[jnt_id["joint3"]]],
                ])
                record_dq.append([data.qvel[dof_j1], data.qvel[dof_j3]])
                p_now = np.array(data.site_xpos[site_ee], dtype=float)
                record_p.append(p_now.copy())
                record_F.append(np.array(data.ctrl[act_ids_ordered], dtype=float))
                record_res.append([peak_res["a"], peak_res["b"]])     # 仅峰值快照

                if step % TRAIL_SAMPLE_EVERY == 0:
                    ee_actual_xyz.append(np.array([p_now[0], p_now[1], 0.03]))

                if data.time - last_sync > 1.0 / 30.0:
                    paint_viewer_overlay(viewer, ee_xy, ee_actual_xyz)
                    viewer.sync()
                    last_sync = data.time

                rest = model.opt.timestep - (time.time() - step_start)
                if rest > 0:
                    time.sleep(rest)
    finally:
        mujoco.set_mjcb_control(None)

    # ---- 6.6 数据整理 + 绘图 ----
    record_t  = np.array(record_t)
    record_q  = np.array(record_q)
    record_dq = np.array(record_dq)
    record_p  = np.array(record_p)
    record_F  = np.array(record_F)

    qa_d_i  = np.interp(record_t, t_vals,  qa_des)
    qb_d_i  = np.interp(record_t, t_vals,  qb_des)
    dqa_d_i = np.interp(record_t, t_vals, dqa_des)
    dqb_d_i = np.interp(record_t, t_vals, dqb_des)
    X_d_i   = np.interp(record_t, t_vals, X_des_full)
    Y_d_i   = np.interp(record_t, t_vals, Y_des_full)

    record_tau_a = KP_JOINT1 * (qa_d_i - record_q[:, 0]) + KD_JOINT1 * (dqa_d_i - record_dq[:, 0])
    record_tau_b = KP_JOINT2 * (qb_d_i - record_q[:, 1]) + KD_JOINT2 * (dqb_d_i - record_dq[:, 1])

    fig = plt.figure(figsize=(17, 10))

    ax1 = fig.add_subplot(231)
    ax1.plot(X_des_full, Y_des_full, 'k--', lw=2, label="Desired EE")
    ax1.plot(record_p[:, 0], record_p[:, 1], 'r-', lw=2, alpha=0.85, label="Actual EE")
    ax1.plot(record_p[0, 0], record_p[0, 1], 'go', ms=8, label="Start")
    ax1.plot(record_p[-1, 0], record_p[-1, 1], 'bs', ms=8, label="End")
    ax1.set_title("End-Effector Cartesian Tracking (Cable Antagonistic)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_aspect('equal', adjustable='box'); ax1.grid(True); ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(232)
    ax2.plot(t_vals, np.rad2deg(qa_des), 'k--', lw=2, label=r"$q_a^{\mathrm{des}}$")
    ax2.plot(record_t, np.rad2deg(record_q[:, 0]), 'b-', lw=2, alpha=0.85, label=r"$q_a$")
    ax2.plot(t_vals, np.rad2deg(qb_des), 'gray', ls='--', lw=2, label=r"$q_b^{\mathrm{des}}$")
    ax2.plot(record_t, np.rad2deg(record_q[:, 1]), 'r-', lw=2, alpha=0.85, label=r"$q_b$")
    ax2.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax2.set_title("Joint-Space PD Tracking"); ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg)"); ax2.grid(True); ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(233)
    ax3.plot(record_t, np.rad2deg(qa_d_i - record_q[:, 0]), 'b-', lw=2, label=r"$e_a$")
    ax3.plot(record_t, np.rad2deg(qb_d_i - record_q[:, 1]), 'r-', lw=2, label=r"$e_b$")
    ax3.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax3.set_title("Joint Tracking Error"); ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Error (deg)"); ax3.grid(True); ax3.legend()

    ax4 = fig.add_subplot(234)
    cart_err = np.sqrt((record_p[:, 0] - X_d_i) ** 2 + (record_p[:, 1] - Y_d_i) ** 2)
    ax4.plot(record_t, cart_err, 'm-', lw=2)
    ax4.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax4.set_title("Cartesian Tracking Error"); ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("|Δp| (m)"); ax4.grid(True)

    ax5 = fig.add_subplot(235)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    styles = ['-', '-', '-.', '-.', '--', '--', ':', ':']
    for i, name in enumerate(CABLE_NAMES):
        ax5.plot(record_t, record_F[:, i], color=colors[i], ls=styles[i],
                 lw=1.4, alpha=0.85, label=name)
    ax5.axhline(F_PRE, color='k', lw=0.6, alpha=0.4, label=f"F_pre={F_PRE:.0f} N")
    ax5.axhline(F_MAX, color='r', lw=0.6, alpha=0.4, label=f"F_max={F_MAX:.0f} N")
    ax5.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax5.set_title("8 Cable Tension Commands (N)")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("F (N)")
    ax5.grid(True); ax5.legend(fontsize=7, ncol=2, loc='upper right')

    ax6 = fig.add_subplot(236)
    ax6.plot(record_t, record_tau_a, 'b-', lw=1.6, alpha=0.9, label=r"$\tau_a^{\mathrm{des}}$")
    ax6.plot(record_t, record_tau_b, 'r-', lw=1.6, alpha=0.9, label=r"$\tau_b^{\mathrm{des}}$")
    ax6.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax6.axvline(T_RAMP, color='k', ls=':', alpha=0.5)
    ax6.set_title("PD Commanded Joint Torques"); ax6.set_xlabel("Time (s)")
    ax6.set_ylabel(r"$\tau$ (Nm)"); ax6.grid(True); ax6.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig_name="cable_antagonistic_pd_tracking")

    # ---- 6.7 跟踪质量统计 ----
    rms_ea   = float(np.sqrt(np.mean((qa_d_i - record_q[:, 0]) ** 2)))
    rms_eb   = float(np.sqrt(np.mean((qb_d_i - record_q[:, 1]) ** 2)))
    rms_cart = float(np.sqrt(np.mean(cart_err ** 2)))
    max_ea   = float(np.max(np.abs(qa_d_i - record_q[:, 0])))
    max_eb   = float(np.max(np.abs(qb_d_i - record_q[:, 1])))
    max_cart = float(np.max(cart_err))
    print(f"\n  仿真完成, 总步数 = {step}, 记录长度 = {len(record_t)}")
    print(f"  PD 峰值     : |τa|={peak_tau['a']:.2f} Nm   |τb|={peak_tau['b']:.2f} Nm")
    print(f"  绳索峰值张力: max F_cable = {peak_F['F']:.1f} N   (上限 {F_MAX:.0f} N)")
    print(f"  映射残差峰值: |res_a|={peak_res['a']:.4f} Nm   |res_b|={peak_res['b']:.4f} Nm")
    print(f"  映射饱和步数: {saturate_count['n']} / {step}   "
          f"({100.0 * saturate_count['n'] / max(step,1):.1f}%)")
    print(f"  关节 RMS 误差 : e_a={np.rad2deg(rms_ea):.4f}°   e_b={np.rad2deg(rms_eb):.4f}°")
    print(f"  关节 峰值误差 : |e_a|_max={np.rad2deg(max_ea):.4f}°   "
          f"|e_b|_max={np.rad2deg(max_eb):.4f}°")
    print(f"  笛卡尔 RMS    : |Δp|_rms = {rms_cart*1000:.3f} mm   "
          f"|Δp|_max = {max_cart*1000:.3f} mm")
    print(f"  图片已保存至: {get_save_dir()}")


if __name__ == "__main__":
    main()
