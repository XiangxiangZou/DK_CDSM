"""基于 MuJoCo site Jacobian 的末端逆运动学。

本模块把 MuJoCo XML 当作真实机械臂的几何模型：
1. 读取 `end_effector` site 的世界坐标作为末端位置。
2. 调用 MuJoCo `mj_jacSite` 得到 site 对所有广义速度的雅可比。
3. 将 joint2 对 joint1、joint4 对 joint3 的 mimic 关系折算为二自由度雅可比：
   `d p / d qa = J_joint1 + J_joint2`，`d p / d qb = J_joint3 + J_joint4`。
4. 用阻尼最小二乘迭代求解 `[qa,qb]`。

该实现不依赖项目根目录下的解析 IK 类，因此后续 XML 细节变动时更容易与 MuJoCo
模型保持一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from .mujoco_plant import ACTIVE_JOINTS, MIMIC_JOINTS
except ImportError:  # pragma: no cover
    from mujoco_plant import ACTIVE_JOINTS, MIMIC_JOINTS


@dataclass(frozen=True)
class IKConfig:
    """MuJoCo 逆运动学参数。

    参数:
        site_name: 作为末端执行器的 MuJoCo site 名称，默认 `end_effector`。
        max_iter: 单个目标点 IK 的最大迭代次数。
        tol: 末端位置误差收敛阈值，单位 m。
        damping: 阻尼最小二乘的阻尼系数。越大越稳定，但收敛更慢、稳态误差更大。
        max_step: 单次迭代的最大关节增量，单位 rad，用于抑制接近奇异点时的跳变。
        joint_margin: 主动关节限位内缩量，单位 rad，避免参考轨迹贴着硬限位。
        smooth_window_s: 对 IK 后的关节参考做移动平均的时间窗，单位 s；设为 0 关闭。
        q_seed_a: 第一个点的 qa 初值，单位 rad。默认避开完全伸直奇异构型。
        q_seed_b: 第一个点的 qb 初值，单位 rad。
    """

    site_name: str = "end_effector"
    max_iter: int = 120
    tol: float = 1e-5
    damping: float = 1e-4
    max_step: float = 0.08
    joint_margin: float = 0.05
    smooth_window_s: float = 0.03
    q_seed_a: float = 0.1
    q_seed_b: float = -0.1


def _require_mujoco():
    """延迟导入 mujoco，便于无 MuJoCo 环境下阅读其它模块。"""
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for MuJoCo inverse kinematics.") from exc
    return mujoco


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """对二维时间序列做边界复制的移动平均。"""
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(values, dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return out


class MujocoSiteIK:
    """MuJoCo 末端 site 逆运动学求解器。

    参数:
        xml_path: MuJoCo XML 路径。
        dt: 采样周期，单位 s；只用于轨迹速度数值微分和配置记录。
        cfg: IK 参数配置。
    """

    def __init__(self, xml_path: str | Path, dt: float, cfg: IKConfig | None = None) -> None:
        self.mujoco = _require_mujoco()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")
        self.model = self.mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = float(dt)
        self.data = self.mujoco.MjData(self.model)
        self.dt = float(dt)
        self.cfg = cfg or IKConfig()
        self.indices = self._build_indices()
        self.q_limits = self._read_active_joint_limits()

    def _joint_id(self, name: str) -> int:
        """按名称查找 joint id。"""
        jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in XML: {name}")
        return int(jid)

    def _build_indices(self) -> Dict[str, int]:
        """缓存 IK 所需的 joint/site/qpos/dof 索引。"""
        active_ids = {name: self._joint_id(name) for name in ACTIVE_JOINTS}
        mimic_ids = {name: self._joint_id(name) for name in MIMIC_JOINTS}
        all_ids = {**active_ids, **mimic_ids}
        site_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_SITE,
            self.cfg.site_name,
        )
        if site_id < 0:
            raise ValueError(f"Site not found in XML: {self.cfg.site_name}")
        return {
            "joint1_qpos": int(self.model.jnt_qposadr[all_ids["joint1"]]),
            "joint2_qpos": int(self.model.jnt_qposadr[all_ids["joint2"]]),
            "joint3_qpos": int(self.model.jnt_qposadr[all_ids["joint3"]]),
            "joint4_qpos": int(self.model.jnt_qposadr[all_ids["joint4"]]),
            "joint1_dof": int(self.model.jnt_dofadr[all_ids["joint1"]]),
            "joint2_dof": int(self.model.jnt_dofadr[all_ids["joint2"]]),
            "joint3_dof": int(self.model.jnt_dofadr[all_ids["joint3"]]),
            "joint4_dof": int(self.model.jnt_dofadr[all_ids["joint4"]]),
            "joint1_id": int(all_ids["joint1"]),
            "joint3_id": int(all_ids["joint3"]),
            "site_id": int(site_id),
        }

    def _read_active_joint_limits(self) -> np.ndarray:
        """读取主动关节限位，并按 `joint_margin` 向内收缩。"""
        limits = np.zeros((2, 2), dtype=np.float64)
        for i, key in enumerate(("joint1_id", "joint3_id")):
            jid = self.indices[key]
            if int(self.model.jnt_limited[jid]) == 0:
                limits[i] = (-np.inf, np.inf)
            else:
                raw = np.asarray(self.model.jnt_range[jid], dtype=np.float64)
                limits[i, 0] = raw[0] + float(self.cfg.joint_margin)
                limits[i, 1] = raw[1] - float(self.cfg.joint_margin)
        return limits

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        """将 `[qa,qb]` 裁剪到主动关节安全范围内。"""
        return np.clip(np.asarray(q, dtype=np.float64).reshape(2), self.q_limits[:, 0], self.q_limits[:, 1])

    def _set_q(self, q: Sequence[float]) -> None:
        """设置主动关节和 mimic 关节角度，并刷新 MuJoCo 派生量。"""
        qa, qb = self._clip_q(np.asarray(q, dtype=np.float64))
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[self.indices["joint1_qpos"]] = qa
        self.data.qpos[self.indices["joint2_qpos"]] = qa
        self.data.qpos[self.indices["joint3_qpos"]] = qb
        self.data.qpos[self.indices["joint4_qpos"]] = qb
        self.mujoco.mj_forward(self.model, self.data)

    def forward_xy(self, q: Sequence[float]) -> np.ndarray:
        """用 MuJoCo 正运动学计算给定 `[qa,qb]` 的末端 xy 坐标。"""
        self._set_q(q)
        return np.asarray(self.data.site_xpos[self.indices["site_id"], :2], dtype=np.float64).copy()

    def forward_xy_batch(self, q_seq: np.ndarray) -> np.ndarray:
        """批量计算末端 xy 坐标。

        参数:
            q_seq: 关节角序列，形状 `(N,2)`，单位 rad。
        """
        q_arr = np.asarray(q_seq, dtype=np.float64)
        xy = np.zeros((q_arr.shape[0], 2), dtype=np.float64)
        for k, q in enumerate(q_arr):
            xy[k] = self.forward_xy(q)
        return xy

    def active_site_jacobian_xy(self, q: Sequence[float]) -> np.ndarray:
        """计算末端 xy 对 `[qa,qb]` 的 2x2 MuJoCo 等效雅可比。"""
        self._set_q(q)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.indices["site_id"])
        jxy = jacp[:2, :]
        return np.column_stack(
            [
                jxy[:, self.indices["joint1_dof"]] + jxy[:, self.indices["joint2_dof"]],
                jxy[:, self.indices["joint3_dof"]] + jxy[:, self.indices["joint4_dof"]],
            ]
        )

    def solve_xy(self, target_xy: Sequence[float], q_guess: Sequence[float] | None = None) -> Tuple[np.ndarray, dict]:
        """求解单个末端 xy 目标点对应的 `[qa,qb]`。

        参数:
            target_xy: 目标末端坐标 `[x,y]`，单位 m。
            q_guess: 本点迭代初值。轨迹 IK 中应传上一帧解，保证解支连续。

        返回:
            `(q, info)`，其中 `q` 是关节角，`info` 记录是否收敛、迭代次数和最终误差。
        """
        target = np.asarray(target_xy, dtype=np.float64).reshape(2)
        if q_guess is None:
            q = np.array([self.cfg.q_seed_a, self.cfg.q_seed_b], dtype=np.float64)
        else:
            q = np.asarray(q_guess, dtype=np.float64).reshape(2)
        q = self._clip_q(q)

        err_norm = np.inf
        converged = False
        for it in range(int(self.cfg.max_iter)):
            xy = self.forward_xy(q)
            err = target - xy
            err_norm = float(np.linalg.norm(err))
            if err_norm <= float(self.cfg.tol):
                converged = True
                break

            jac = self.active_site_jacobian_xy(q)
            lhs = jac @ jac.T + float(self.cfg.damping) * np.eye(2)
            dq = jac.T @ np.linalg.solve(lhs, err)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > float(self.cfg.max_step):
                dq = dq * (float(self.cfg.max_step) / step_norm)
            q = self._clip_q(q + dq)

        xy_final = self.forward_xy(q)
        final_err = target - xy_final
        final_norm = float(np.linalg.norm(final_err))
        info = {
            "converged": bool(converged or final_norm <= float(self.cfg.tol)),
            "iterations": int(it + 1),
            "error_norm": final_norm,
            "target_xy": target.tolist(),
            "achieved_xy": xy_final.tolist(),
        }
        return q.copy(), info

    def solve_trajectory(self, xy_ref: np.ndarray) -> Dict[str, np.ndarray | dict]:
        """对一整条末端轨迹逐点 IK，并生成关节参考。

        参数:
            xy_ref: 笛卡尔末端参考轨迹，形状 `(N,2)`，单位 m。

        返回:
            字典字段：
            - `q_ref`: IK 后的关节角参考 `(N,2)`；
            - `dq_ref`: 数值微分关节速度参考 `(N,2)`；
            - `ee_ik`: 平滑后关节参考经 MuJoCo FK 得到的末端 xy `(N,2)`；
            - `ik_error`: `ee_ik - xy_ref`；
            - `ik_converged`: 每个点是否达到 `tol`；
            - `ik_iterations`: 每个点迭代次数；
            - `meta`: IK 配置快照。
        """
        xy = np.asarray(xy_ref, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy_ref must have shape (N,2), got {xy.shape}")

        q_ref = np.zeros((xy.shape[0], 2), dtype=np.float64)
        converged = np.zeros(xy.shape[0], dtype=bool)
        iterations = np.zeros(xy.shape[0], dtype=np.int32)
        q_guess = np.array([self.cfg.q_seed_a, self.cfg.q_seed_b], dtype=np.float64)
        for k, target in enumerate(xy):
            q_sol, info = self.solve_xy(target, q_guess)
            q_ref[k] = q_sol
            converged[k] = bool(info["converged"])
            iterations[k] = int(info["iterations"])
            q_guess = q_sol

        window = int(round(float(self.cfg.smooth_window_s) / max(self.dt, 1e-12)))
        q_ref = _moving_average(q_ref, window)
        q_ref = np.vstack([self._clip_q(q) for q in q_ref])
        dq_ref = np.gradient(q_ref, self.dt, axis=0)
        ee_ik = self.forward_xy_batch(q_ref)
        ik_error = ee_ik - xy
        return {
            "q_ref": q_ref,
            "dq_ref": dq_ref,
            "ee_ik": ee_ik,
            "ik_error": ik_error,
            "ik_converged": converged,
            "ik_iterations": iterations,
            "meta": {
                "xml_path": str(self.xml_path),
                "dt": self.dt,
                "ik_config": asdict(self.cfg),
                "q_limits": self.q_limits.tolist(),
            },
        }
