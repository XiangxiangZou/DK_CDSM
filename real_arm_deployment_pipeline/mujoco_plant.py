"""MuJoCo 被控对象适配器。

模块职责：
1. 加载绳驱空间机械臂 MuJoCo XML。
2. 提供与真实机械臂一致的最小接口：读状态、写绳张力、执行一步。
3. 缓存主动关节、mimic 关节、actuator、tendon 和 DOF 索引。

后续接真实机械臂时，应保留上层调用语义，新增/替换 `RealArmPlant`：
- `read_state() -> [qa, qb, dqa, dqb]`
- `apply_cable_tensions(F8)`
- `step()`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from .cable_mapping import ACTUATOR_NAMES, CABLE_NAMES
except ImportError:  # pragma: no cover - direct script execution fallback
    from cable_mapping import ACTUATOR_NAMES, CABLE_NAMES

# 两个主动关节：qa 使用 joint1，qb 使用 joint3。
ACTIVE_JOINTS = ("joint1", "joint3")

# XML 中 joint2 跟随 joint1，joint4 跟随 joint3，用于保持 2-DOF 等效模型。
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}

# 统一状态和控制维度：x=[qa,qb,dqa,dqb]，u=[tau_a,tau_b]。
STATE_DIM = 4
CONTROL_DIM = 2


def _require_mujoco():
    """延迟导入 mujoco，便于在无 MuJoCo 环境中只阅读/导入其它模块。"""
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for data collection.") from exc
    return mujoco


def _name_to_joint_id(mujoco, model, name: str) -> int:
    """按名称查找 MuJoCo joint id，缺失时给出明确错误。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in XML: {name}")
    return int(jid)


def _name_to_actuator_id(mujoco, model, name: str) -> int:
    """按名称查找 MuJoCo actuator id，缺失时给出明确错误。"""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found in XML: {name}")
    return int(aid)


def _name_to_tendon_id(mujoco, model, name: str) -> int:
    """按名称查找 MuJoCo tendon id，缺失时给出明确错误。"""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if tid < 0:
        raise ValueError(f"Tendon not found in XML: {name}")
    return int(tid)


class MujocoCablePlant:
    """MuJoCo 绳驱机械臂封装。

    参数:
        xml_path: MuJoCo XML 文件路径。
        dt: 仿真步长/控制周期，单位 s。后续真实机械臂部署时也应使用同一含义。

    重要属性:
        q_limits: 两个主动关节的角度限制，形状 `(2,2)`，单位 rad。
        indices: MuJoCo 内部索引缓存，上层通常不直接使用。
    """

    def __init__(self, xml_path: str | Path, dt: float) -> None:
        self.mujoco = _require_mujoco()
        xml = Path(xml_path)
        if not xml.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml}")
        self.model = self.mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = float(dt)
        self.data = self.mujoco.MjData(self.model)
        self.scratch = self.mujoco.MjData(self.model)
        self.dt = float(dt)
        self.indices = self._build_indices()
        self.q_limits = self._read_active_joint_limits()

    def _build_indices(self) -> Dict[str, np.ndarray | int]:
        """缓存当前 XML 中采数和控制所需的所有 MuJoCo 索引。"""
        active_joint_ids = np.array(
            [_name_to_joint_id(self.mujoco, self.model, name) for name in ACTIVE_JOINTS],
            dtype=int,
        )
        active_qpos = np.array([self.model.jnt_qposadr[jid] for jid in active_joint_ids], dtype=int)
        active_dof = np.array([self.model.jnt_dofadr[jid] for jid in active_joint_ids], dtype=int)

        joint_ids = {
            name: _name_to_joint_id(self.mujoco, self.model, name)
            for name in ("joint1", "joint2", "joint3", "joint4")
        }
        dof_all = {name: int(self.model.jnt_dofadr[jid]) for name, jid in joint_ids.items()}

        mimic_pairs = []
        for mimic, source in MIMIC_JOINTS.items():
            mimic_jid = _name_to_joint_id(self.mujoco, self.model, mimic)
            source_jid = _name_to_joint_id(self.mujoco, self.model, source)
            mimic_pairs.append((self.model.jnt_qposadr[mimic_jid], self.model.jnt_qposadr[source_jid]))

        actuator_ids = np.array(
            [_name_to_actuator_id(self.mujoco, self.model, name) for name in ACTUATOR_NAMES],
            dtype=int,
        )
        tendon_ids = np.array(
            [_name_to_tendon_id(self.mujoco, self.model, name) for name in CABLE_NAMES],
            dtype=int,
        )

        return {
            "active_qpos": active_qpos,
            "active_dof": active_dof,
            "mimic_pairs": np.array(mimic_pairs, dtype=int),
            "actuator_ids": actuator_ids,
            "tendon_ids": tendon_ids,
            "dof_j1": dof_all["joint1"],
            "dof_j2": dof_all["joint2"],
            "dof_j3": dof_all["joint3"],
            "dof_j4": dof_all["joint4"],
        }

    def _read_active_joint_limits(self) -> np.ndarray:
        """读取主动关节角度范围，返回 `[[qa_min, qa_max], [qb_min, qb_max]]`。"""
        limits = np.zeros((len(ACTIVE_JOINTS), 2), dtype=np.float64)
        for i, name in enumerate(ACTIVE_JOINTS):
            jid = _name_to_joint_id(self.mujoco, self.model, name)
            if int(self.model.jnt_limited[jid]) == 0:
                limits[i] = (-np.inf, np.inf)
            else:
                limits[i] = np.asarray(self.model.jnt_range[jid], dtype=np.float64)
        return limits

    def set_state(self, q: np.ndarray, dq: np.ndarray) -> None:
        """设置主动关节状态并同步 mimic 关节。

        参数:
            q: 主动关节角度 `[qa, qb]`，单位 rad。
            dq: 主动关节角速度 `[dqa, dqb]`，单位 rad/s。
        """
        q = np.asarray(q, dtype=np.float64).reshape(2)
        dq = np.asarray(dq, dtype=np.float64).reshape(2)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[self.indices["active_qpos"]] = q
        self.data.qvel[self.indices["active_dof"]] = dq
        for mimic_qpos, source_qpos in self.indices["mimic_pairs"]:
            self.data.qpos[mimic_qpos] = self.data.qpos[source_qpos]
        self.mujoco.mj_forward(self.model, self.data)

    def read_state(self) -> np.ndarray:
        """读取统一状态向量 `[qa, qb, dqa, dqb]`。"""
        q = self.data.qpos[self.indices["active_qpos"]]
        dq = self.data.qvel[self.indices["active_dof"]]
        return np.array([q[0], q[1], dq[0], dq[1]], dtype=np.float64)

    def compute_tendon_jacobian(self, eps: float = 1e-6) -> np.ndarray:
        """计算 tendon length 对 MuJoCo 广义坐标的有限差分雅可比。

        参数:
            eps: 有限差分扰动量，单位 rad。太大降低精度，太小会放大数值噪声。

        返回:
            形状 `(8, nv)` 的雅可比矩阵，行顺序与 `CABLE_NAMES` 一致。
        """
        nv = self.model.nv
        tendon_ids = self.indices["tendon_ids"]
        jac = np.zeros((len(tendon_ids), nv), dtype=np.float64)
        q_ref = np.asarray(self.data.qpos, dtype=np.float64).copy()
        for j in range(nv):
            self.scratch.qpos[:] = q_ref
            self.scratch.qpos[j] = q_ref[j] + eps
            self.mujoco.mj_fwdPosition(self.model, self.scratch)
            length_plus = np.asarray(self.scratch.ten_length, dtype=np.float64)[tendon_ids]

            self.scratch.qpos[:] = q_ref
            self.scratch.qpos[j] = q_ref[j] - eps
            self.mujoco.mj_fwdPosition(self.model, self.scratch)
            length_minus = np.asarray(self.scratch.ten_length, dtype=np.float64)[tendon_ids]
            jac[:, j] = (length_plus - length_minus) / (2.0 * eps)
        return jac

    def apply_cable_tensions(self, tensions: np.ndarray) -> None:
        """下发 8 根绳张力到 MuJoCo actuator。

        参数:
            tensions: 形状 `(8,)`，单位 N，顺序与 `CABLE_NAMES` 一致。
        """
        tensions = np.asarray(tensions, dtype=np.float64).reshape(len(CABLE_NAMES))
        self.data.ctrl[self.indices["actuator_ids"]] = tensions

    def step(self) -> None:
        """按当前 `data.ctrl` 推进 MuJoCo 一个时间步。"""
        self.mujoco.mj_step(self.model, self.data)

    def torque_dofs(self) -> Tuple[int, int, int, int]:
        """返回 joint1..joint4 的 DOF 索引，供绳张力映射计算力矩臂。"""
        return (
            int(self.indices["dof_j1"]),
            int(self.indices["dof_j2"]),
            int(self.indices["dof_j3"]),
            int(self.indices["dof_j4"]),
        )
