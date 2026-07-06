"""
test_mujoco_cable.py
====================
针对 `multi_joint_cable_dirven_space_robot.xml` (8 电机拮抗绳驱版) 的独立测试脚本.

三件事:
  1) 加载 XML, 打印 MJCF 结构自检 (关节 / 绳索 / 执行器 / 约束数目 + 每根绳零位长度 L0);
  2) 启动 MuJoCo 交互式 3D viewer, 用 `--mode` 选择一种开环驱动:
       * sine      : 拮抗正弦摆动, 两级 spreader 频率不同, 便于目视验证绳驱符号关系;
       * preload   : 8 根绳仅施加恒定预紧, 观察机械臂能否保持静止 (零位不漂移)
                     — 用来验证 <equality> 耦合和等张力拮抗是否自洽;
       * step      : 在 t=1 s 对 spreader1 正向对施加阶跃张力, t=3 s 反转,
                     便于观察阶跃响应与回摆.
  3) 每 0.5 s 在终端打印 qa / qb 关节角 (deg), 以及 8 根绳索长度相对初值的变化 ΔL
     与当前电机张力指令, 用于核对:
       "qa > 0 时 cable11/cable13 变短, cable12/cable14 变长" 这一几何关系是否成立.

运行:
  python test_mujoco_cable.py                 # 默认 sine 模式
  python test_mujoco_cable.py --mode preload
  python test_mujoco_cable.py --mode step
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


XML_PATH = str(
    Path(__file__).resolve().parents[1]
    / "assets"
    / "multi_joint_cable_driven_space_robot.xml"
)

CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = [
    "winch_c11", "winch_c12", "winch_c13", "winch_c14",
    "winch_c21", "winch_c22", "winch_c23", "winch_c24",
]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
TENSION_SENSOR_NAMES = [
    "tension_c11", "tension_c12", "tension_c13", "tension_c14",
    "tension_c21", "tension_c22", "tension_c23", "tension_c24",
]
ENCODER_SENSOR_NAMES = [
    "encoder_joint1", "encoder_joint2", "encoder_joint3", "encoder_joint4",
]

F_PRELOAD = 20.0       # N, 预紧张力 (>=下限 + 一定裕量防松弛)
F_AMPLITUDE = 80.0     # N, 拮抗对额外张力幅值
W1, W2 = 1.2, 0.8      # rad/s, 两级 spreader 摆动角频率 (sine 模式)
F_STEP = 120.0         # N, step 模式阶跃张力
STEP_T_ON, STEP_T_FLIP = 1.0, 3.0
PRINT_INTERVAL = 0.5


def _check_model(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    print("\n" + "=" * 66)
    print(" MuJoCo 8-绳驱模型结构自检")
    print("=" * 66)
    print(f"  nq={model.nq}  nv={model.nv}  njnt={model.njnt} (期望 4)")
    print(f"  ntendon={model.ntendon} (期望 8)   nu={model.nu} (期望 8)")
    print(f"  neq={model.neq} (期望 2: joint1≡joint2, joint3≡joint4)")
    print(f"  nsensor={model.nsensor} (期望 12: 8 张力 + 4 编码器)")

    mujoco.mj_forward(model, data)
    print("\n  零位构型下 8 根绳索长度 L0 (m):")
    for n in CABLE_NAMES:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n)
        print(f"    {n:<8s}  L0 = {data.ten_length[tid]:.4f}")

    print("\n  执行器 -> 绳索 映射 & ctrlrange (N):")
    for n in ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        trn_id = int(model.actuator_trnid[aid, 0])
        trn_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)
        lo, hi = model.actuator_ctrlrange[aid]
        print(f"    {n:<11s} -> {trn_name:<8s}  ctrl∈[{lo:.1f}, {hi:.1f}]")
    print("=" * 66 + "\n")


def _build_indices(model: mujoco.MjModel) -> tuple[dict, dict, dict, dict, dict]:
    act_idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
    ten_idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n) for n in CABLE_NAMES}
    jnt_idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES}
    ten_sen_idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in TENSION_SENSOR_NAMES}
    enc_sen_idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in ENCODER_SENSOR_NAMES}
    for label, d in [
        ("actuator", act_idx),
        ("tendon", ten_idx),
        ("joint", jnt_idx),
        ("tension sensor", ten_sen_idx),
        ("encoder sensor", enc_sen_idx),
    ]:
        for k, v in d.items():
            if v < 0:
                raise RuntimeError(f"[模型自检失败] 未在 XML 中找到 {label} {k!r}")
    return act_idx, ten_idx, jnt_idx, ten_sen_idx, enc_sen_idx


def _sensor_value(model: mujoco.MjModel, data: mujoco.MjData, sid: int) -> float:
    """读取标量传感器 (actuatorfrc / jointpos 均为 dim=1) 的当前值."""
    adr = int(model.sensor_adr[sid])
    return float(data.sensordata[adr])


def _ctrl_sine(t: float, act_idx: dict) -> np.ndarray:
    """拮抗正弦: spreader1 用 W1, spreader2 用 W2."""
    u = np.full(8, F_PRELOAD, dtype=float)
    d1 = F_AMPLITUDE * np.sin(W1 * t)
    u[act_idx["winch_c11"]] += max(d1, 0.0)
    u[act_idx["winch_c13"]] += max(d1, 0.0)
    u[act_idx["winch_c12"]] += max(-d1, 0.0)
    u[act_idx["winch_c14"]] += max(-d1, 0.0)
    d2 = F_AMPLITUDE * np.sin(W2 * t)
    u[act_idx["winch_c21"]] += max(d2, 0.0)
    u[act_idx["winch_c23"]] += max(d2, 0.0)
    u[act_idx["winch_c22"]] += max(-d2, 0.0)
    u[act_idx["winch_c24"]] += max(-d2, 0.0)
    return u


def _ctrl_preload(_t: float, _act_idx: dict) -> np.ndarray:
    """所有 8 根绳等张力拮抗, 机械臂应保持静止 (<equality> 约束一致性测试)."""
    return np.full(8, F_PRELOAD, dtype=float)


def _ctrl_step(t: float, act_idx: dict) -> np.ndarray:
    """阶跃: t in [1,3) 正向对 +F_STEP, t>=3 反向对 +F_STEP, 其余时段仅预紧."""
    u = np.full(8, F_PRELOAD, dtype=float)
    if t >= STEP_T_FLIP:
        u[act_idx["winch_c12"]] += F_STEP
        u[act_idx["winch_c14"]] += F_STEP
    elif t >= STEP_T_ON:
        u[act_idx["winch_c11"]] += F_STEP
        u[act_idx["winch_c13"]] += F_STEP
    return u


MODE_TO_FN = {
    "sine": _ctrl_sine,
    "preload": _ctrl_preload,
    "step": _ctrl_step,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=tuple(MODE_TO_FN.keys()), default="sine",
                    help="开环驱动模式 (详见脚本顶部 docstring)")
    args = ap.parse_args()

    print(f"加载 MuJoCo 模型: {XML_PATH}  (mode={args.mode})")
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"[FAIL] 模型加载失败: {e}")
        return
    print("-> 加载成功")

    _check_model(model, data)
    act_idx, ten_idx, jnt_idx, ten_sen_idx, enc_sen_idx = _build_indices(model)
    L0 = data.ten_length.copy()
    ctrl_fn = MODE_TO_FN[args.mode]

    print("=" * 66)
    print(" MuJoCo 交互式 3D 查看器已启动")
    print("=" * 66)
    print("【操作】鼠标拖动旋转/平移, 滚轮缩放, 空格暂停, 退格重置, 关闭窗口退出")
    print("【输出】每 0.5 s 打印 4 个编码器读数 (deg) + 8 根绳的 (L, ΔL, 张力传感器 F)")
    print("【viewer】左侧 Sensor 面板会实时绘制 8 张力 + 4 编码器的曲线")
    print("=" * 66 + "\n")

    def control_callback(_m: mujoco.MjModel, d: mujoco.MjData) -> None:
        d.ctrl[:] = ctrl_fn(d.time, act_idx)

    mujoco.set_mjcb_control(control_callback)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 12.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -90
            viewer.cam.lookat[:] = [4.0, 0.0, 0.0]

            last_print = -1.0
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()

                if data.time - last_print >= PRINT_INTERVAL:
                    enc_deg = [
                        np.rad2deg(_sensor_value(model, data, enc_sen_idx[n]))
                        for n in ENCODER_SENSOR_NAMES
                    ]
                    header = (
                        f"t={data.time:6.2f}s | encoders[deg] "
                        f"j1={enc_deg[0]:+7.2f} j2={enc_deg[1]:+7.2f} "
                        f"j3={enc_deg[2]:+7.2f} j4={enc_deg[3]:+7.2f} | "
                    )
                    parts = []
                    for n in CABLE_NAMES:
                        tid = ten_idx[n]
                        sen_name = "tension_c" + n[len("cable"):]
                        F = _sensor_value(model, data, ten_sen_idx[sen_name])
                        L = float(data.ten_length[tid])
                        dL = L - float(L0[tid])
                        parts.append(f"{n}:L={L:.3f}(Δ{dL:+.3f})/F={F:5.1f}N")
                    print(header + " ".join(parts[:4]))
                    print(" " * len(header) + " ".join(parts[4:]))
                    last_print = data.time

                rest = model.opt.timestep - (time.time() - step_start)
                if rest > 0:
                    time.sleep(rest)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
