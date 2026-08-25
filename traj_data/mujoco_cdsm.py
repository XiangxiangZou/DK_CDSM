"""两自由度绳驱空间机械臂的轻量级 MuJoCo 适配器。

本模块封装了 MuJoCo 物理引擎的底层调用，为绳驱空间机械臂（CDSM）
提供状态读写、关节限位查询、缆索张力控制等高层接口。

机械臂结构：
    - 共 4 个关节（joint1~joint4），其中 joint1 和 joint3 为主动关节，
      joint2 和 joint4 为从动关节（分别跟随 joint1 和 joint3 运动）。
    - 每个主动关节由 4 根缆索驱动（共 8 根），缆索通过绞盘（winch actuator）
      施加张力来控制关节力矩。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 机械臂结构常量
# ---------------------------------------------------------------------------

# 主动关节名称（独立控制的两个关节）
ACTIVE_JOINTS = ("joint1", "joint3")

# 从动关节映射：key=从动关节, value=其跟随的主动关节
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}

# 8 根缆索的名称：cable1j 对应关节1的第j根缆索，cable2j 对应关节2（即关节3）的第j根缆索
CABLE_NAMES = (
    "cable11",
    "cable12",
    "cable13",
    "cable14",
    "cable21",
    "cable22",
    "cable23",
    "cable24",
)

# 绞盘执行器名称：将缆索名中的 "cable" 替换为 "winch_c"
# 例如 "cable11" -> "winch_c11"
ACTUATOR_NAMES = tuple("winch_c" + name[len("cable") :] for name in CABLE_NAMES)

# 状态向量的字段顺序：[关节a位置, 关节b位置, 关节a速度, 关节b速度]
STATE_ORDER = ("qa", "qb", "dqa", "dqb")

# 控制输入向量的字段顺序：[关节a力矩, 关节b力矩]
INPUT_ORDER = ("tau_a", "tau_b")

# 绳索张力下限。采集阶段不设置张力上限，只保证绳索不低于预紧力。
CABLE_TENSION_LOWER_BOUND = 20.0


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _require_mujoco():
    """延迟导入 MuJoCo 库，避免非必要环境下对 MuJoCo 的硬依赖。

    返回
    -------
    module
        mujoco 模块对象。

    抛出
    ------
    RuntimeError
        如果 mujoco 未安装。
    """
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("采集轨迹数据需要安装 mujoco 库") from exc
    return mujoco


def _name_to_id(mujoco, model, obj_type, name: str) -> int:
    """将 MuJoCo 对象名称转换为内部 ID。

    参数
    ----------
    mujoco : module
        MuJoCo 模块。
    model : MjModel
        MuJoCo 模型对象。
    obj_type : int
        MuJoCo 对象类型（如 mjOBJ_JOINT, mjOBJ_ACTUATOR 等）。
    name : str
        XML 中定义的对象名称。

    返回
    -------
    int
        对象的内部 ID。

    抛出
    ------
    ValueError
        如果名称在模型中不存在。
    """
    value = mujoco.mj_name2id(model, obj_type, name)
    if value < 0:
        raise ValueError(f"MuJoCo 对象未找到: {name}")
    return int(value)


def _solve_pair(
    m_positive: float,
    m_negative: float,
    tau_desired: float,
    f_min: float,
) -> Tuple[float, float]:
    """求解一对对抗式缆索的张力，使合力矩逼近目标力矩。

    每个关节由一对对抗的缆索组（正向组和反向组）驱动。张力只受
    下限约束，不设置上限；选取使合成力矩最接近目标值的一组张力。

    参数
    ----------
    m_positive : float
        正向缆索组的总力臂（合力矩 = 张力 × 力臂）。
    m_negative : float
        反向缆索组的总力臂。
    tau_desired : float
        目标关节力矩。
    f_min : float
        张力下限，所有缆索张力不得低于此值。

    返回
    -------
    Tuple[float, float]
        (f_positive, f_negative)：正向组和反向组的缆索张力。
    """
    # 张力下限产生的基准力矩（两方向力臂不等时不为零）
    tau_base = (m_positive + m_negative) * f_min
    # 需要由张力增量产生的有效力矩
    tau_effective = tau_desired - tau_base

    # 候选方案：(正向增量, 反向增量, 力矩误差, 总增量)
    # 方案1：不施加任何增量，仅维持张力下限
    candidates = [(0.0, 0.0, abs(tau_effective), 0.0)]
    # 方案2：仅正向组增加张力
    if abs(m_positive) > 1e-12:
        inc = max(tau_effective / m_positive, 0.0)
        candidates.append((inc, 0.0, abs(tau_effective - m_positive * inc), inc))
    # 方案3：仅反向组增加张力
    if abs(m_negative) > 1e-12:
        inc = max(tau_effective / m_negative, 0.0)
        candidates.append((0.0, inc, abs(tau_effective - m_negative * inc), inc))

    # 选择力矩误差最小的方案，误差相同时选总增量最小的
    positive, negative, _, _ = min(candidates, key=lambda item: (item[2], item[3]))
    return f_min + positive, f_min + negative


# ---------------------------------------------------------------------------
# MuJoCo 绳驱机械臂适配器
# ---------------------------------------------------------------------------


class MujocoCDSM:
    """轻量级 MuJoCo 封装，提供状态读写、关节限位和缆索张力控制接口。

    参数
    ----------
    xml_path : str | Path
        MuJoCo 场景 XML 文件路径。
    dt : float
        仿真时间步长（秒）。
    """

    def __init__(self, xml_path: str | Path, dt: float) -> None:
        # 延迟导入 MuJoCo
        self.mujoco = _require_mujoco()

        # 加载 XML 模型
        xml = Path(xml_path)
        if not xml.exists():
            raise FileNotFoundError(f"MuJoCo XML 文件未找到: {xml}")
        self.model = self.mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = float(dt)

        # 主仿真数据和临时数据（用于有限差分计算）
        self.data = self.mujoco.MjData(self.model)
        self.scratch = self.mujoco.MjData(self.model)
        self.dt = float(dt)

        # 构建内部索引映射
        self.indices = self._build_indices()

        # 读取主动关节的物理限位（来自 XML）
        self.q_limits = self._read_active_joint_limits()

    # -----------------------------------------------------------------------
    # 内部初始化方法
    # -----------------------------------------------------------------------

    def _build_indices(self) -> dict[str, np.ndarray | int]:
        """从 MuJoCo 模型中提取并缓存所有需要的索引。

        构建内容包括：
        - active_qpos: 主动关节在 qpos 向量中的位置索引
        - active_dof:  主动关节在 qvel 向量中的自由度索引
        - mimic_pairs: 从动关节与其源关节的 qpos 地址配对
        - actuator_ids: 8 个绞盘执行器的 ID
        - tendon_ids:   8 根缆索（肌腱）的 ID
        - dof_j1~j4:    各关节在速度向量中的自由度索引

        返回
        -------
        dict
            所有索引的字典。
        """
        # 获取四个关节的 MuJoCo 内部 ID
        joint_ids = {
            name: _name_to_id(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("joint1", "joint2", "joint3", "joint4")
        }
        # 主动关节 ID 数组
        active_joint_ids = np.array([joint_ids[name] for name in ACTIVE_JOINTS], dtype=int)

        # 构建从动→主动关节的位置地址配对
        mimic_pairs = []
        for mimic, source in MIMIC_JOINTS.items():
            mimic_pairs.append(
                (
                    self.model.jnt_qposadr[joint_ids[mimic]],
                    self.model.jnt_qposadr[joint_ids[source]],
                )
            )

        # 绞盘执行器 ID
        actuator_ids = np.array(
            [
                _name_to_id(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ACTUATOR_NAMES
            ],
            dtype=int,
        )

        # 缆索（肌腱）ID
        tendon_ids = np.array(
            [
                _name_to_id(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_TENDON, name)
                for name in CABLE_NAMES
            ],
            dtype=int,
        )

        return {
            "active_qpos": np.array([self.model.jnt_qposadr[jid] for jid in active_joint_ids], dtype=int),
            "active_dof": np.array([self.model.jnt_dofadr[jid] for jid in active_joint_ids], dtype=int),
            "mimic_pairs": np.array(mimic_pairs, dtype=int),
            "actuator_ids": actuator_ids,
            "tendon_ids": tendon_ids,
            "dof_j1": int(self.model.jnt_dofadr[joint_ids["joint1"]]),
            "dof_j2": int(self.model.jnt_dofadr[joint_ids["joint2"]]),
            "dof_j3": int(self.model.jnt_dofadr[joint_ids["joint3"]]),
            "dof_j4": int(self.model.jnt_dofadr[joint_ids["joint4"]]),
        }

    def _read_active_joint_limits(self) -> np.ndarray:
        """从 XML 模型中读取主动关节的物理限位。

        返回
        -------
        np.ndarray
            形状 (2, 2)，每行为 [下限, 上限]。

        抛出
        ------
        ValueError
            如果任一主动关节在 XML 中未定义有限限位。
        """
        limits = np.zeros((2, 2), dtype=np.float64)
        for row, name in enumerate(ACTIVE_JOINTS):
            jid = _name_to_id(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if int(self.model.jnt_limited[jid]) == 0:
                raise ValueError(f"{name} 在 XML 中未定义有限的物理关节限位")
            limits[row] = np.asarray(self.model.jnt_range[jid], dtype=np.float64)
        return limits

    # -----------------------------------------------------------------------
    # 状态读写
    # -----------------------------------------------------------------------

    def set_state(self, q: np.ndarray, dq: np.ndarray) -> None:
        """设置仿真状态（关节位置和速度）。

        将主动关节的位置和速度写入 MuJoCo 数据，同步从动关节，
        然后执行一次前向运动学计算。

        参数
        ----------
        q : np.ndarray
            主动关节位置，形状 (2,) [关节a, 关节b]。
        dq : np.ndarray
            主动关节速度，形状 (2,) [关节a速度, 关节b速度]。
        """
        q = np.asarray(q, dtype=np.float64).reshape(2)
        dq = np.asarray(dq, dtype=np.float64).reshape(2)
        # 清零所有状态
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.time = 0.0
        # 写入主动关节位置和速度
        self.data.qpos[self.indices["active_qpos"]] = q
        self.data.qvel[self.indices["active_dof"]] = dq
        # 同步从动关节位置
        self._sync_mimic_joints(self.data)
        # 执行前向运动学，更新依赖位置的计算量（如肌腱长度）
        self.mujoco.mj_forward(self.model, self.data)

    def _sync_mimic_joints(self, data) -> None:
        """将从动关节的位置同步为其对应主动关节的位置。

        joint2 复制 joint1 的位置，joint4 复制 joint3 的位置。
        """
        for mimic_qpos, source_qpos in self.indices["mimic_pairs"]:
            data.qpos[mimic_qpos] = data.qpos[source_qpos]

    def read_state(self) -> np.ndarray:
        """读取当前仿真状态。

        返回
        -------
        np.ndarray
            形状 (4,) 的状态向量 [qa, qb, dqa, dqb]：
            主动关节a位置、主动关节b位置、关节a速度、关节b速度。
        """
        q = self.data.qpos[self.indices["active_qpos"]]
        dq = self.data.qvel[self.indices["active_dof"]]
        return np.array([q[0], q[1], dq[0], dq[1]], dtype=np.float64)

    # -----------------------------------------------------------------------
    # 缆索力学计算
    # -----------------------------------------------------------------------

    def compute_tendon_jacobian(self, eps: float = 1e-6) -> np.ndarray:
        """通过中心差分计算肌腱长度对关节位置的雅可比矩阵。

        雅可比矩阵 J 的元素 J[i, j] = ∂L_i / ∂q_j，其中 L_i 为第 i 根
        缆索的长度，q_j 为第 j 个自由度的广义坐标。

        使用双侧有限差分（中心差分）计算，精度为 O(ε²)。

        参数
        ----------
        eps : float
            有限差分扰动步长，默认 1e-6。

        返回
        -------
        np.ndarray
            形状 (8, nv) 的雅可比矩阵，8 为缆索数，nv 为 MuJoCo 模型自由度总数。
        """
        tendon_ids = self.indices["tendon_ids"]
        # 保存当前 qpos 状态的副本
        q_ref = np.asarray(self.data.qpos, dtype=np.float64).copy()
        jac = np.zeros((len(tendon_ids), self.model.nv), dtype=np.float64)

        for dof in range(self.model.nv):
            # 正向扰动：q[dof] + ε
            self.scratch.qpos[:] = q_ref
            self.scratch.qpos[dof] = q_ref[dof] + eps
            self.mujoco.mj_fwdPosition(self.model, self.scratch)
            plus = np.asarray(self.scratch.ten_length, dtype=np.float64)[tendon_ids]

            # 反向扰动：q[dof] - ε
            self.scratch.qpos[:] = q_ref
            self.scratch.qpos[dof] = q_ref[dof] - eps
            self.mujoco.mj_fwdPosition(self.model, self.scratch)
            minus = np.asarray(self.scratch.ten_length, dtype=np.float64)[tendon_ids]

            # 中心差分：∂L/∂q ≈ (L⁺ - L⁻) / (2ε)
            jac[:, dof] = (plus - minus) / (2.0 * eps)
        return jac

    def torque_to_tensions(
        self,
        tau: np.ndarray,
        *,
        f_min: float = CABLE_TENSION_LOWER_BOUND,
    ) -> np.ndarray:
        """将期望关节力矩转换为各缆索的目标张力。

        核心思路：
        1. 计算当前位姿下缆索长度关于关节角度的雅可比矩阵。
        2. 通过虚功原理，力矩 τ = Jᵀ · f（缆索张力 f 通过雅可比转置映射为关节力矩）。
        3. 将 8 根缆索分为两组对抗式驱动对（关节a使用 cable1j，关节b使用 cable2j）。
        4. 只施加张力下限，不施加张力上限；求解使合成力矩最接近目标的张力分配。

        参数
        ----------
        tau : np.ndarray
            期望关节力矩，形状 (2,) [τ_a, τ_b]。
        f_min : float
            缆索张力下限，默认 20 N；所有缆索张力不低于此值。

        返回
        -------
        np.ndarray
            形状 (8,) 的缆索张力向量，按 CABLE_NAMES 顺序排列。
        """
        tau = np.asarray(tau, dtype=np.float64).reshape(2)

        # 计算当前位姿下的缆索长度雅可比
        jac = self.compute_tendon_jacobian()

        # 提取各关节自由度的列索引
        dof_j1 = int(self.indices["dof_j1"])
        dof_j2 = int(self.indices["dof_j2"])
        dof_j3 = int(self.indices["dof_j3"])
        dof_j4 = int(self.indices["dof_j4"])

        # 计算等效力臂：关节a = joint1 + joint2（从动），关节b = joint3 + joint4（从动）
        # 每行对应一个关节通道中各缆索对该关节的总力臂贡献
        arm_a = jac[:, dof_j1] + jac[:, dof_j2]  # 8 维向量，每根缆索对关节a的力臂
        arm_b = jac[:, dof_j3] + jac[:, dof_j4]  # 8 维向量，每根缆索对关节b的力臂

        # 为关节a求解对抗缆索组的张力
        # 正向组（索引 0,2 即 cable11, cable13），反向组（索引 1,3 即 cable12, cable14）
        f1p, f1m = _solve_pair(arm_a[0] + arm_a[2], arm_a[1] + arm_a[3], tau[0], f_min)
        # 为关节b求解对抗缆索组的张力
        # 正向组（索引 4,6 即 cable21, cable23），反向组（索引 5,7 即 cable22, cable24）
        f2p, f2m = _solve_pair(arm_b[4] + arm_b[6], arm_b[5] + arm_b[7], tau[1], f_min)

        # 组装 8 根缆索的张力
        tensions = np.empty(8, dtype=np.float64)
        tensions[[0, 2]] = f1p  # 关节a正向组：cable11, cable13
        tensions[[1, 3]] = f1m  # 关节a反向组：cable12, cable14
        tensions[[4, 6]] = f2p  # 关节b正向组：cable21, cable23
        tensions[[5, 7]] = f2m  # 关节b反向组：cable22, cable24
        return tensions

    def apply_cable_tensions(self, tensions: np.ndarray) -> None:
        """将缆索张力写入 MuJoCo 执行器控制信号。

        参数
        ----------
        tensions : np.ndarray
            形状 (8,) 的缆索张力，按 CABLE_NAMES 顺序排列。
        """
        values = np.asarray(tensions, dtype=np.float64).reshape(8)
        self.data.ctrl[self.indices["actuator_ids"]] = values

    def apply_joint_disturbance(self, torque: np.ndarray) -> None:
        """Apply an external torque to the two active joint DOFs.

        The disturbance is deliberately separate from the cable-produced
        ``applied_torque`` used by identification. This makes the absolute-time
        sine torque an unobserved change in the state transition law.
        """
        values = np.asarray(torque, dtype=np.float64).reshape(2)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.indices["active_dof"]] = values

    def equivalent_joint_torque(
        self,
        tensions: np.ndarray,
        tendon_jacobian: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map cable tensions to the two equivalent active-joint torques."""
        values = np.asarray(tensions, dtype=np.float64).reshape(8)
        jac = self.compute_tendon_jacobian() if tendon_jacobian is None else np.asarray(
            tendon_jacobian, dtype=np.float64
        )
        arm_a = jac[:, int(self.indices["dof_j1"])] + jac[:, int(self.indices["dof_j2"])]
        arm_b = jac[:, int(self.indices["dof_j3"])] + jac[:, int(self.indices["dof_j4"])]
        return np.array([arm_a @ values, arm_b @ values], dtype=np.float64)

    # -----------------------------------------------------------------------
    # 仿真步进
    # -----------------------------------------------------------------------

    def step(self) -> None:
        """执行一步 MuJoCo 物理仿真。

        调用 mj_step 前进一个时间步长（由 model.opt.timestep 决定），
        自动完成前向动力学、约束求解和状态更新。
        """
        self.mujoco.mj_step(self.model, self.data)
