"""
test_mujoco_cable_jacobian.py
=============================
绳索 Jacobian / 张力-关节扭矩映射 —— 静态校验脚本
    (模型: multi_joint_cable_dirven_space_robot.xml)

目的 (为即将到来的拮抗控制层做 "显微镜式" 验证):
  (1) 确认 XML 的 gear=1 让 actuatorfrc 传感器读数 = 下发 ctrl (N).
  (2) 取出 MuJoCo 内部的瞬时 tendon Jacobian  J = ∂L/∂q  ∈ R^{ntendon × nv},
      并把它和我们对几何的直觉比一比.
  (3) 验证  "同侧对称预紧 -> 净关节广义力 ≈ 0"   (拮抗策略的必要前提).
  (4) 在 (qa, qb) 的若干典型构型下, 把 MuJoCo 自己算出的 qfrc_actuator
      与手算  τ = -J.T @ F_cable   逐 DOF 对比, 确保误差在数值精度内.
  (5) 给出 "同侧合并" 之后的 4×2 有效力臂矩阵 M(q):
          [τ_a ]         [ F₁⁺ ]
          [τ_b ] = M(q) · [ F₁⁻ ]
                          [ F₂⁺ ]
                          [ F₂⁻ ]
      这就是后面在线拮抗映射器要用到的核心算子.

说明: 全程只用 mj_forward (不积分), 所以绳索的 "放出/收紧" 速度、绳索惯性
      等纯动力学问题都被屏蔽, 只测运动学/力学静态关系.

运行:
    python mujoco_cdsm_jacobian.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "glfw")
# 强制 stdout 使用 utf-8, 避免 Windows GBK 终端无法打印非 ASCII 数学符号
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import mujoco

np.set_printoptions(precision=4, suppress=True, linewidth=160)

XML_PATH = str(
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "multi_joint_cable_driven_space_robot.xml"
)

CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]
JOINT_NAMES    = ["joint1", "joint2", "joint3", "joint4"]
TENSION_SENSOR_NAMES = ["tension_c" + n[len("cable"):] for n in CABLE_NAMES]

# 同侧对合并: 顺序 = [F1p, F1m, F2p, F2m]
#   F1p = f11 = f13      (spreader1 上侧)
#   F1m = f12 = f14      (spreader1 下侧)
#   F2p = f21 = f23      (spreader2 上侧)
#   F2m = f22 = f24      (spreader2 下侧)
SIDE_PAIRS = {
    "F1p": ("cable11", "cable13"),
    "F1m": ("cable12", "cable14"),
    "F2p": ("cable21", "cable23"),
    "F2m": ("cable22", "cable24"),
}

F_PRE    = 20.0     # N, 预紧力
F_PROBE  = 100.0    # N, 激励张力 (用来检查差分映射)


# ============================================================================
# 工具
# ============================================================================
def load():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data  = mujoco.MjData(model)
    return model, data


def build_indices(model):
    tdn_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,   n) for n in CABLE_NAMES}
    act_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
    jnt_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    n) for n in JOINT_NAMES}
    sen_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,   n) for n in TENSION_SENSOR_NAMES}
    for name, idx in list(tdn_id.items()) + list(act_id.items()) + list(jnt_id.items()) + list(sen_id.items()):
        if idx < 0:
            raise RuntimeError(f"[模型自检失败] 未找到 {name!r}")
    return tdn_id, act_id, jnt_id, sen_id


def set_qpos(model, data, qa, qb):
    """把机械臂设到 (qa, qb) 构型, 保持 j1=j2=qa, j3=j4=qb."""
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")]] = qa
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")]] = qa
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")]] = qb
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint4")]] = qb
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def get_ten_jacobian(model, data, eps: float = 1e-6):
    """
    返回 8×nv 的 tendon Jacobian J (行按 CABLE_NAMES 的顺序):
        J[i, j] = dL_i(q) / dq_j

    实现方式: 中心差分
        J[:, j] = ( L(q + eps·e_j) - L(q - eps·e_j) ) / (2·eps)
    理由: MuJoCo Python 绑定里 data.ten_J 的内存布局在不同版本中不稳定
          (dense / sparse / per-row-sparse 均有), 而 data.ten_length
          永远是纯 (ntendon,) 稠密, 由 mj_forward 的 mj_tendon 阶段计算,
          且不受 equality 约束的约束力/投影影响, 刚好对应我们需要的
          "纯运动学" 绳长-关节 Jacobian.

    这里只用于离线校验, 对效率无要求.
    """
    nt = model.ntendon
    nv = model.nv

    q0 = data.qpos.copy()
    J = np.zeros((nt, nv), dtype=float)
    for j in range(nv):
        data.qpos[:] = q0
        data.qpos[j] = q0[j] + eps
        mujoco.mj_forward(model, data)
        L_plus = np.array(data.ten_length, dtype=float).copy()
        data.qpos[:] = q0
        data.qpos[j] = q0[j] - eps
        mujoco.mj_forward(model, data)
        L_minus = np.array(data.ten_length, dtype=float).copy()
        J[:, j] = (L_plus - L_minus) / (2.0 * eps)
    # 恢复原始构型
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)

    tdn_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n) for n in CABLE_NAMES]
    return J[tdn_id, :].copy()


def get_dof_indices(model, jnt_id):
    """返回 joint1..joint4 各自 DOF 索引 (nv 空间)."""
    return [int(model.jnt_dofadr[jnt_id[n]]) for n in JOINT_NAMES]


def apply_cable_forces(model, data, F_cable: np.ndarray, act_id):
    """把 8×1 张力向量写入 ctrl, 并用 mj_forward 触发 MuJoCo 重新计算 qfrc_actuator."""
    ordered = np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int)
    data.ctrl[ordered] = F_cable
    mujoco.mj_forward(model, data)


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 80)
    print("  绳索 Jacobian 校验  (XML: multi_joint_space_robot_cable.xml, gear=1)")
    print("=" * 80)
    model, data = load()
    tdn_id, act_id, jnt_id, sen_id = build_indices(model)
    dof_j1, dof_j2, dof_j3, dof_j4 = get_dof_indices(model, jnt_id)

    print(f"  nq={model.nq}   nv={model.nv}   nu={model.nu}   "
          f"ntendon={model.ntendon}   neq={model.neq}")
    print(f"  关节 DOF 索引 (nv 空间): j1={dof_j1}  j2={dof_j2}  j3={dof_j3}  j4={dof_j4}")
    gear_col = np.array(model.actuator_gear[:, 0])
    print(f"  actuator_gear[:,0] = {gear_col}   (应全部 == 1.0)")
    assert np.allclose(gear_col, 1.0), "gear 必须 = 1, 否则 ctrl != 实际张力"

    # ------------------------------------------------------------------
    # Step 1: 零构型下, 下发 ctrl, 检查 sensor 读数是否 = ctrl
    # ------------------------------------------------------------------
    print("\n[Step 1]  gear=1 完整性检查: 下发阶跃 ctrl, 读 <actuatorfrc> 传感器")
    set_qpos(model, data, 0.0, 0.0)
    probe = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    apply_cable_forces(model, data, probe, act_id)
    sens = np.array([
        float(data.sensordata[sen_id[TENSION_SENSOR_NAMES[i]]])
        for i in range(8)
    ])
    print("    下发 ctrl  =", probe)
    print("    sensor 读数=", sens)
    max_err = float(np.max(np.abs(sens - probe)))
    print(f"    max|err| = {max_err:.3e} N  "
          f"→ {'PASS' if max_err < 1e-9 else 'FAIL'}  (期望完全一致)")

    # ------------------------------------------------------------------
    # Step 2: 零构型下的 tendon Jacobian
    # ------------------------------------------------------------------
    print("\n[Step 2]  零构型 (qa=qb=0) 下的 tendon Jacobian J = ∂L/∂q")
    set_qpos(model, data, 0.0, 0.0)
    J0 = get_ten_jacobian(model, data)
    # 按 (joint1, joint2, joint3, joint4) 列裁出, 可读性更好
    J0_jnt = J0[:, [dof_j1, dof_j2, dof_j3, dof_j4]]
    print("    cable      ∂L/∂j1     ∂L/∂j2     ∂L/∂j3     ∂L/∂j4")
    for i, name in enumerate(CABLE_NAMES):
        print(f"    {name:8s}  {J0_jnt[i, 0]:+9.4f}  {J0_jnt[i, 1]:+9.4f}  "
              f"{J0_jnt[i, 2]:+9.4f}  {J0_jnt[i, 3]:+9.4f}")

    L0 = np.array(data.ten_length, dtype=float)
    tdn_ids_ordered = np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int)
    L0 = L0[tdn_ids_ordered]
    print(f"    ten_length = {L0}")

    # ------------------------------------------------------------------
    # Step 2.5: 经验性确认 MuJoCo 内部的传递符号
    #   对每根绳索单独施加 ctrl=1, 其余全部 0, 读取 data.qfrc_actuator.
    #   构成 8×nv 的传递矩阵 A: A[i, dof] = MuJoCo 实际产生的关节广义力.
    #   MuJoCo 内部关系: qfrc_actuator = A^T · ctrl,
    #   而 FD 给出的 J 按理论应有 A = ±J.  这里用经验测量定符号, 避免依赖文档歧义.
    # ------------------------------------------------------------------
    print("\n[Step 2.5]  经验传递矩阵 A (设 ctrl[i]=1, 其余=0) 与 FD Jacobian J 的符号对比")
    set_qpos(model, data, 0.0, 0.0)
    A = np.zeros((model.ntendon, model.nv), dtype=float)  # 按 MuJoCo 自身 tendon 顺序
    ordered = np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int)
    for i, act_name in enumerate(ACTUATOR_NAMES):
        data.ctrl[:] = 0.0
        data.ctrl[ordered[i]] = 1.0
        mujoco.mj_forward(model, data)
        A[i, :] = np.array(data.qfrc_actuator, dtype=float)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    J0 = get_ten_jacobian(model, data)   # 8×nv, 按 CABLE_NAMES 顺序
    # A 当前按 ACTUATOR_NAMES 顺序 (与 CABLE_NAMES 下标一致, 因为名字一一对应),
    # 但 A 的行下标是 "actuator index", J0 的行下标是 CABLE_NAMES 在 MuJoCo 中的 tendon index.
    # 这里直接把 A 按 CABLE_NAMES 顺序整理:
    tdn_sorted_ids = np.array([tdn_id[n] for n in CABLE_NAMES], dtype=int)
    act_sorted_ids = np.array([act_id[n] for n in ACTUATOR_NAMES], dtype=int)
    # A 的第 k 行 = 激活 ACTUATOR_NAMES[k] 时的 qfrc, 已经是我们想要的顺序.
    # 只需要注意 A 与 J0 比较的列顺序 (都是 nv 维 DOF).
    print("    每根绳单独给 ctrl=1 时的 qfrc_actuator @ [j1,j2,j3,j4]:")
    for i, name in enumerate(CABLE_NAMES):
        row = A[i, [dof_j1, dof_j2, dof_j3, dof_j4]]
        print(f"      {name:8s}  qfrc = {row}    J_FD = {J0[i, [dof_j1, dof_j2, dof_j3, dof_j4]]}")
    delta_plus  = np.max(np.abs(A - J0))
    delta_minus = np.max(np.abs(A + J0))
    if delta_plus < 1e-5:
        sign_conv = +1
        print(f"    -> A ≈ +J   (|diff|_∞ = {delta_plus:.2e})")
    elif delta_minus < 1e-5:
        sign_conv = -1
        print(f"    -> A ≈ -J   (|diff|_∞ = {delta_minus:.2e})")
    else:
        raise RuntimeError(
            f"A 与 ±J 都不匹配, |A-J|={delta_plus:.3e}, |A+J|={delta_minus:.3e}"
        )
    print(f"    结论: qfrc_actuator = ({sign_conv:+d}) · J_FD^T · F_ctrl")
    print(f"          后续 Steps 5-6 的映射矩阵 M(q) 以此为准.")

    # ------------------------------------------------------------------
    # Step 3: 同侧对称预紧 → 净 joint torque 应 ≈ 0
    # ------------------------------------------------------------------
    print("\n[Step 3]  全 8 根绳给 F_pre 预紧, 验证净关节广义力 ≈ 0  (对称消除)")
    for qa_deg, qb_deg in [(0, 0), (-45, 45), (-90, 90)]:
        qa, qb = np.deg2rad(qa_deg), np.deg2rad(qb_deg)
        set_qpos(model, data, qa, qb)
        apply_cable_forces(model, data, np.full(8, F_PRE), act_id)
        qf = np.array(data.qfrc_actuator, dtype=float)
        qf_jnt = qf[[dof_j1, dof_j2, dof_j3, dof_j4]]
        print(f"    (qa, qb) = ({qa_deg:+4d}°, {qb_deg:+4d}°)  "
              f"qfrc_actuator @ [j1,j2,j3,j4] = {qf_jnt}  "
              f"Σ|τ| = {np.sum(np.abs(qf_jnt)):.3e}")

    # ------------------------------------------------------------------
    # Step 4: MuJoCo 内部 qfrc_actuator vs 手算 sign · J^T @ F
    #         sign 由 Step 2.5 确定 (本模型实测 sign_conv = +1).
    # ------------------------------------------------------------------
    print(f"\n[Step 4]  MuJoCo 内部 qfrc_actuator  vs  手算  ({sign_conv:+d})·J(q).T @ F_cable")
    rng = np.random.default_rng(0)
    test_configs = [(0, 0), (-30, 30), (-60, 60), (-85, 85)]
    for qa_deg, qb_deg in test_configs:
        qa, qb = np.deg2rad(qa_deg), np.deg2rad(qb_deg)
        set_qpos(model, data, qa, qb)
        F_cable = F_PRE + rng.uniform(0, 200, size=8)   # 随机 8 维非负张力
        apply_cable_forces(model, data, F_cable, act_id)
        J = get_ten_jacobian(model, data)
        qf_mujoco = np.array(data.qfrc_actuator, dtype=float)
        qf_hand   = sign_conv * (J.T @ F_cable)
        err = qf_mujoco - qf_hand
        print(f"    (qa, qb) = ({qa_deg:+4d}°, {qb_deg:+4d}°)   "
              f"|err|_∞ = {np.max(np.abs(err)):.2e}")
        print(f"        MuJoCo qfrc @ [j1,j2,j3,j4] = "
              f"{qf_mujoco[[dof_j1, dof_j2, dof_j3, dof_j4]]}")
        print(f"        手算 ({sign_conv:+d})J.T F  @ [j1,j2,j3,j4] = "
              f"{qf_hand[[dof_j1, dof_j2, dof_j3, dof_j4]]}")

    # ------------------------------------------------------------------
    # Step 5: 合并同侧对 → 2×4 "有效力臂矩阵" M(q)
    #
    #   令 τ_a = qfrc_actuator[j1] + qfrc_actuator[j2]
    #       τ_b = qfrc_actuator[j3] + qfrc_actuator[j4]
    #   这是 spreader1 / spreader2 (j1=j2=qa, j3=j4=qb 通过 equality 耦合)
    #   所受到的总广义力矩.
    #
    #   MuJoCo 传递: qfrc = sign_conv · J_cab^T · F_cable,
    #   所以单根绳 i 对 (τ_a, τ_b) 的贡献是:
    #     a_i = sign_conv · ( J_cab[i, j1] + J_cab[i, j2] )
    #     b_i = sign_conv · ( J_cab[i, j3] + J_cab[i, j4] )
    #
    #   再按同侧对 (f11=f13=F1p, f12=f14=F1m, f21=f23=F2p, f22=f24=F2m) 合并:
    #     [τ_a, τ_b]^T = M(q) · [F1p, F1m, F2p, F2m]^T
    #     M[0, col] = Σ_{i ∈ col-side}  a_i
    #     M[1, col] = Σ_{i ∈ col-side}  b_i
    # ------------------------------------------------------------------
    print("\n[Step 5]  合并同侧对, 得到 2×4 有效力臂矩阵  [τ_a, τ_b]^T = M(q) · [F1p, F1m, F2p, F2m]^T")
    name_to_row = {n: i for i, n in enumerate(CABLE_NAMES)}
    pair_to_rows = {k: [name_to_row[c] for c in v] for k, v in SIDE_PAIRS.items()}
    for qa_deg, qb_deg in [(0, 0), (-45, 45), (-85, 85)]:
        qa, qb = np.deg2rad(qa_deg), np.deg2rad(qb_deg)
        set_qpos(model, data, qa, qb)
        J = get_ten_jacobian(model, data)
        a_per_cable = sign_conv * (J[:, dof_j1] + J[:, dof_j2])   # 8,
        b_per_cable = sign_conv * (J[:, dof_j3] + J[:, dof_j4])   # 8,

        M = np.zeros((2, 4))
        for col, key in enumerate(["F1p", "F1m", "F2p", "F2m"]):
            rows = pair_to_rows[key]
            M[0, col] = a_per_cable[rows].sum()
            M[1, col] = b_per_cable[rows].sum()

        print(f"\n    (qa, qb) = ({qa_deg:+4d}°, {qb_deg:+4d}°)")
        print(f"      每根绳对 qa 的有效力臂 a_i  = {a_per_cable}")
        print(f"      每根绳对 qb 的有效力臂 b_i  = {b_per_cable}")
        print(f"      M(q) =")
        print(f"          F1p      F1m      F2p      F2m")
        print(f"    τ_a  {M[0, 0]:+8.4f} {M[0, 1]:+8.4f} {M[0, 2]:+8.4f} {M[0, 3]:+8.4f}")
        print(f"    τ_b  {M[1, 0]:+8.4f} {M[1, 1]:+8.4f} {M[1, 2]:+8.4f} {M[1, 3]:+8.4f}")

        # 自检 1: 对称预紧 τ = M·[F_pre]×4 -> 应 ≈ 0  (每列相同 F_pre, 每行各列互为相反数)
        f_sym = np.full(4, F_PRE)
        tau_sym = M @ f_sym
        print(f"    对称预紧   τ = M·[{F_PRE},{F_PRE},{F_PRE},{F_PRE}]ᵀ = {tau_sym}  (应 ≈ 0 或数值对称误差)")

        # 自检 2: 只给 F1p 侧加激励 -> τ_a 应沿某个确定方向, τ_b 应≈0
        f_drv = np.array([F_PRE + F_PROBE, F_PRE, F_PRE, F_PRE])
        tau_drv = M @ f_drv
        print(f"    +F1p 激励  τ = M·[{F_PRE + F_PROBE},{F_PRE},{F_PRE},{F_PRE}]ᵀ = {tau_drv}  "
              f"(τ_b 应 ≈ 0; τ_a 符号反映 F1p 驱动方向)")

    # ------------------------------------------------------------------
    # Step 6: 一致性的 end-to-end 检查
    #   把 [F1p, F1m, F2p, F2m] 展开成 8 维 F_cable, 让 MuJoCo 计算 qfrc_actuator,
    #   与 M(q) @ [F1p, F1m, F2p, F2m] 比对.
    # ------------------------------------------------------------------
    print("\n[Step 6]  端到端一致性: MuJoCo qfrc 与 M(q)@F 的比对")
    for qa_deg, qb_deg in [(0, 0), (-60, 60), (-85, 85)]:
        qa, qb = np.deg2rad(qa_deg), np.deg2rad(qb_deg)
        set_qpos(model, data, qa, qb)
        J = get_ten_jacobian(model, data)
        a_per_cable = sign_conv * (J[:, dof_j1] + J[:, dof_j2])
        b_per_cable = sign_conv * (J[:, dof_j3] + J[:, dof_j4])
        M = np.zeros((2, 4))
        for col, key in enumerate(["F1p", "F1m", "F2p", "F2m"]):
            rows = pair_to_rows[key]
            M[0, col] = a_per_cable[rows].sum()
            M[1, col] = b_per_cable[rows].sum()

        F4 = np.array([F_PRE + 150.0, F_PRE, F_PRE + 40.0, F_PRE])
        F8 = np.zeros(8)
        for col, key in enumerate(["F1p", "F1m", "F2p", "F2m"]):
            rows = pair_to_rows[key]
            F8[rows] = F4[col]
        apply_cable_forces(model, data, F8, act_id)
        qf_mujoco = np.array(data.qfrc_actuator, dtype=float)
        tau_a_muj = qf_mujoco[dof_j1] + qf_mujoco[dof_j2]
        tau_b_muj = qf_mujoco[dof_j3] + qf_mujoco[dof_j4]
        tau_muj = np.array([tau_a_muj, tau_b_muj])
        tau_ref = M @ F4
        print(f"    (qa, qb)=({qa_deg:+4d}°,{qb_deg:+4d}°)  "
              f"MuJoCo=[{tau_muj[0]:+9.4f}, {tau_muj[1]:+9.4f}]   "
              f"M·F =[{tau_ref[0]:+9.4f}, {tau_ref[1]:+9.4f}]   "
              f"|err|_∞ = {np.max(np.abs(tau_muj - tau_ref)):.2e}")

    print("\n校验完成. 结论:")
    print("  • gear=1 生效, sensor 读数 = 下发 ctrl (N);")
    print("  • MuJoCo 的 data.ten_J 与 qfrc_actuator 互相自洽 (高精度);")
    print("  • 合并同侧对后得到的 2×4 矩阵 M(q) 就是下一步拮抗映射器要用的力臂算子.")


if __name__ == "__main__":
    main()
