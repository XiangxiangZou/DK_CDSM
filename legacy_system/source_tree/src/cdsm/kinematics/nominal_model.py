"""
cdsm_rigid_nominal_model.py
===========================
绳驱空间机械臂 (CDSM) 的 **2-DOF 刚体名义模型**, 严格参照 MuJoCo 文件
``multi_joint_space_robot.xml`` 的几何与关节耦合关系建立.

设计原则 (按用户要求)
--------------------
- 将机构视为 **平面刚体连杆链**, 不建模绳索张力、拮抗映射与摩擦;
- 控制输入为 **等效关节力矩** ``u = [tau_a, tau_b]``, 与 XML 中
  ``motor_qa`` / ``motor_qb`` 作用在 ``joint1`` / ``joint3`` 的抽象驱动一致;
- 通过 **角度等式约束** 体现绳驱十字架的同步转动:
      joint1 ≡ joint2  ->  第一级等效角 q_a
      joint3 ≡ joint4  ->  第二级等效角 q_b
- 不追求与 MuJoCo 数值完全一致, 仅保证 **运动学构型与主要惯性特征** 一致,
  供后续 ``名义模型 + Koopman 残差`` 混合建模使用.

与 XML 的对应关系
-----------------
+------------------+------------------------------------------+
| XML              | 名义模型                                 |
+------------------+------------------------------------------+
| Link1 固定 L1=2  | 基座, 不参与可动连杆动力学               |
| joint1 @ link2   | q_a, 绝对角 theta_2 = q_a                |
| joint2 @ link3   | 与 joint1 同步 -> theta_3 = 2*q_a        |
| joint3 @ link4   | q_b, 绝对角 theta_4 = 2*q_a + q_b        |
| joint4 @ link5   | 与 joint3 同步 -> theta_5 = 2*q_a + 2*q_b|
| L2=L4=0.2, L3=L5=2.0 | 连杆长度                             |
| m2=m4=1.161, m3=m5=2.866 | MuJoCo 可动段质量 (link1 固定不计) |
| 名义模型取 0.95 倍       | NOMINAL_MASS_SCALE = 0.95        |
| gravity=0        | 空间失重, 无重力项                       |
| joint range ±pi/2| 可选硬限位                               |
| timestep=0.01    | 默认积分步长                             |

状态与控制
----------
    x = [q_a, q_b, dq_a, dq_b]   (与 MuJoCo 2-DOF 抽象一致)
    u = [tau_a, tau_b]           (Nm, 作用在第一级 / 第二级等效关节)

动力学 (平面拉格朗日, 等效 2 关节)
--------------------------------
    M(q) * ddq + C(q, dq) * dq = tau

使用示例
--------
    from cdsm_rigid_nominal_model import CdsmRigidNominalModel, pack_state, unpack_state

    model = CdsmRigidNominalModel()
    x = pack_state(0.1, 0.05, 0.0, 0.0)
    x_next = model.step(x, u=[2.0, -1.0], dt=0.01)
    q_a, q_b, dq_a, dq_b = unpack_state(x_next)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# 与 multi_joint_space_robot.xml 一致的几何 / 惯性常数
# ---------------------------------------------------------------------------
XML_MODEL_NAME = "multi_joint_space_robot"
XML_TIMESTEP = 0.01
XML_JOINT_RANGE = (-np.pi / 2.0, np.pi / 2.0)  # range="-1.5708 1.5708"

# MuJoCo multi_joint_cable_dirven_space_robot.xml 可动连杆质量 [kg]
XML_MASS_M2_M4 = 1.161  # link2 / link4 (inertial)
XML_MASS_M3_M5 = 2.866  # link3 / link5 (geom mass)
NOMINAL_MASS_SCALE = 0.95  # 名义模型质量 = MuJoCo 质量 × 该系数


@dataclass(frozen=True)
class CdsmNominalParams:
    """名义模型参数 (几何对齐 XML; 质量为 MuJoCo 值的 NOMINAL_MASS_SCALE 倍)."""

    # 连杆长度 [m]
    L1: float = 2.0   # link1 固定基座
    L2: float = 0.2   # link2 spreader 主轴
    L3: float = 2.0   # link3 主连杆
    L4: float = 0.2   # link4 spreader 主轴
    L5: float = 2.0   # link5 主连杆

    # 十字架半长 [m]: Ls = 0.6, fromto ±0.3
    Ls1: float = 0.6
    Ls2: float = 0.6

    # 可动连杆质量 [kg] (link1 固定, 不计入 M(q)); 0.95 × MuJoCo
    m2: float = XML_MASS_M2_M4 * NOMINAL_MASS_SCALE
    m3: float = XML_MASS_M3_M5 * NOMINAL_MASS_SCALE
    m4: float = XML_MASS_M2_M4 * NOMINAL_MASS_SCALE
    m5: float = XML_MASS_M3_M5 * NOMINAL_MASS_SCALE

    # 质心沿杆长比例 (细杆近似: 中点)
    # link2/4 的 MuJoCo inertial 在 pos="0.1 0 0" -> 相对杆长 0.5
    com_ratio_short: float = 0.5
    com_ratio_long: float = 0.5

    # 转动惯量: 细杆 I = m*L^2/12 (运行时由质量与杆长计算)

    # 仿真选项
    dt_default: float = XML_TIMESTEP
    joint_limit: float = np.pi / 2.0


ArrayLike = Union[float, Sequence[float], np.ndarray]


def pack_state(q_a: float, q_b: float, dq_a: float, dq_b: float) -> np.ndarray:
    """组装 4 维状态向量."""
    return np.array([q_a, q_b, dq_a, dq_b], dtype=np.float64)


def unpack_state(x: np.ndarray) -> Tuple[float, float, float, float]:
    """解包状态向量."""
    x = np.asarray(x, dtype=np.float64).reshape(4)
    return float(x[0]), float(x[1]), float(x[2]), float(x[3])


class CdsmRigidNominalModel:
    """
    绳驱空间机械臂刚体名义模型 (2-DOF).

    运动学树 (世界系 x 向右, y 向上, 转动绕 z):
        p0 --L1--> p1 --L2,θ2=qa--> p2 --L3,θ3=2qa--> p3 --L4,θ4=2qa+qb--> p4 --L5,θ5=2qa+2qb--> p5
    """

    def __init__(self, params: Optional[CdsmNominalParams] = None) -> None:
        self.p = params if params is not None else CdsmNominalParams()

        self.L1 = self.p.L1
        self.L2 = self.p.L2
        self.L3 = self.p.L3
        self.L4 = self.p.L4
        self.L5 = self.p.L5
        self.Ls1 = self.p.Ls1
        self.Ls2 = self.p.Ls2

        # 质心到关节的距离 (用于雅可比)
        self.r2 = self.L2 * self.p.com_ratio_short
        self.r3 = self.L3 * self.p.com_ratio_long
        self.r4 = self.L4 * self.p.com_ratio_short
        self.r5 = self.L5 * self.p.com_ratio_long

        self.I2 = self.p.m2 * self.L2 ** 2 / 12.0
        self.I3 = self.p.m3 * self.L3 ** 2 / 12.0
        self.I4 = self.p.m4 * self.L4 ** 2 / 12.0
        self.I5 = self.p.m5 * self.L5 ** 2 / 12.0

    # ------------------------------------------------------------------
    # 运动学 (与 XML body 链 / equality 一致)
    # ------------------------------------------------------------------
    def link_absolute_angles(self, q_a: float, q_b: float) -> Dict[str, float]:
        """
        各连杆绝对转角 (rad), 对应 XML 中 joint 累积角.

        joint1 = q_a, joint2 = q_a  ->  link3 绝对角 = q_a + q_a = 2*q_a
        joint3 = q_b, joint4 = q_b  ->  link5 绝对角 = 2*q_a + q_b + q_b
        """
        return {
            "theta2": q_a,
            "theta3": 2.0 * q_a,
            "theta4": 2.0 * q_a + q_b,
            "theta5": 2.0 * q_a + 2.0 * q_b,
        }

    def forward_kinematics(
        self, q_a: float, q_b: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        正运动学: 返回关节点 p0..p5 (各为 2D 坐标).

        与 XML site 语义对应:
            p0 = anchor_base
            p2 附近 = anchor_l3 (link3 首端)
            p3 = target_l3 (link3 末端)
            p5 = end_effector / target_l5
        """
        th = self.link_absolute_angles(q_a, q_b)
        t2, t3, t4, t5 = th["theta2"], th["theta3"], th["theta4"], th["theta5"]

        p0 = np.array([0.0, 0.0])
        p1 = p0 + np.array([self.L1, 0.0])
        p2 = p1 + self.L2 * np.array([np.cos(t2), np.sin(t2)])
        p3 = p2 + self.L3 * np.array([np.cos(t3), np.sin(t3)])
        p4 = p3 + self.L4 * np.array([np.cos(t4), np.sin(t4)])
        p5 = p4 + self.L5 * np.array([np.cos(t5), np.sin(t5)])
        return p0, p1, p2, p3, p4, p5

    def end_effector_position(self, q_a: float, q_b: float) -> np.ndarray:
        """末端点 p5 坐标."""
        return self.forward_kinematics(q_a, q_b)[-1]

    def workspace_jacobian(self, q_a: float, q_b: float) -> np.ndarray:
        """
        末端 (x, y) 对 (q_a, q_b) 的 2x2 雅可比, 用于 IK / 速度映射.
        """
        t2, t3, t4, t5 = (
            self.link_absolute_angles(q_a, q_b)[k]
            for k in ("theta2", "theta3", "theta4", "theta5")
        )

        dx_dqa = (
            -self.L2 * np.sin(t2)
            - 2.0 * self.L3 * np.sin(t3)
            - 2.0 * self.L4 * np.sin(t4)
            - 2.0 * self.L5 * np.sin(t5)
        )
        dx_dqb = -self.L4 * np.sin(t4) - 2.0 * self.L5 * np.sin(t5)
        dy_dqa = (
            self.L2 * np.cos(t2)
            + 2.0 * self.L3 * np.cos(t3)
            + 2.0 * self.L4 * np.cos(t4)
            + 2.0 * self.L5 * np.cos(t5)
        )
        dy_dqb = self.L4 * np.cos(t4) + 2.0 * self.L5 * np.cos(t5)
        return np.array([[dx_dqa, dx_dqb], [dy_dqa, dy_dqb]], dtype=np.float64)

    def spreader_and_cable_anchors(
        self, q_a: float, q_b: float
    ) -> Dict[str, np.ndarray]:
        """
        返回 XML 中 spreader 与绳索锚点的几何位置 (仅用于可视化/调试, 不参与动力学).

        对应 tendon:
            cable11/12: anchor_base <-> spreader1_top/bot
            cable13/14: target_l3    <-> spreader1_top/bot
            cable21/22: anchor_l3    <-> spreader2_top/bot
            cable23/24: target_l5    <-> spreader2_top/bot
        """
        p0, p1, p2, p3, p4, p5 = self.forward_kinematics(q_a, q_b)
        th2 = q_a
        th4 = 2.0 * q_a + q_b

        # spreader1 中心在 link2 中点 (XML geom fromto 0.1 ±0.3)
        c1 = 0.5 * (p1 + p2)
        n1 = np.array([-np.sin(th2), np.cos(th2)])
        s1_top = c1 + 0.5 * self.Ls1 * n1
        s1_bot = c1 - 0.5 * self.Ls1 * n1

        c2 = 0.5 * (p3 + p4)
        n2 = np.array([-np.sin(th4), np.cos(th4)])
        s2_top = c2 + 0.5 * self.Ls2 * n2
        s2_bot = c2 - 0.5 * self.Ls2 * n2

        return {
            "anchor_base": p0,
            "anchor_l3": p2,
            "target_l3": p3,
            "anchor_l5": p4,
            "target_l5": p5,
            "spreader1_top": s1_top,
            "spreader1_bot": s1_bot,
            "spreader2_top": s2_top,
            "spreader2_bot": s2_bot,
        }

    # ------------------------------------------------------------------
    # 动力学 (平面 2-DOF 拉格朗日)
    # ------------------------------------------------------------------
    def mass_matrix(self, q: ArrayLike) -> np.ndarray:
        """等效 2x2 质量矩阵 M(q)."""
        q = np.asarray(q, dtype=np.float64).reshape(2)
        q_a, q_b = q[0], q[1]
        th = self.link_absolute_angles(q_a, q_b)
        t2, t3, t4, t5 = th["theta2"], th["theta3"], th["theta4"], th["theta5"]

        # 质心速度雅可比 d(p_com)/d[q_a, q_b]
        Jv2 = np.array([[-self.r2 * np.sin(t2), 0.0], [self.r2 * np.cos(t2), 0.0]])
        Jv3 = np.array(
            [
                [-self.L2 * np.sin(t2) - 2.0 * self.r3 * np.sin(t3), 0.0],
                [self.L2 * np.cos(t2) + 2.0 * self.r3 * np.cos(t3), 0.0],
            ]
        )
        Jv4 = np.array(
            [
                [
                    -self.L2 * np.sin(t2)
                    - 2.0 * self.L3 * np.sin(t3)
                    - 2.0 * self.r4 * np.sin(t4),
                    -self.r4 * np.sin(t4),
                ],
                [
                    self.L2 * np.cos(t2)
                    + 2.0 * self.L3 * np.cos(t3)
                    + 2.0 * self.r4 * np.cos(t4),
                    self.r4 * np.cos(t4),
                ],
            ]
        )
        Jv5 = np.array(
            [
                [
                    -self.L2 * np.sin(t2)
                    - 2.0 * self.L3 * np.sin(t3)
                    - 2.0 * self.L4 * np.sin(t4)
                    - 2.0 * self.r5 * np.sin(t5),
                    -self.L4 * np.sin(t4) - 2.0 * self.r5 * np.sin(t5),
                ],
                [
                    self.L2 * np.cos(t2)
                    + 2.0 * self.L3 * np.cos(t3)
                    + 2.0 * self.L4 * np.cos(t4)
                    + 2.0 * self.r5 * np.cos(t5),
                    self.L4 * np.cos(t4) + 2.0 * self.r5 * np.cos(t5),
                ],
            ]
        )

        Jw2 = np.array([[1.0, 0.0]])
        Jw3 = np.array([[2.0, 0.0]])
        Jw4 = np.array([[2.0, 1.0]])
        Jw5 = np.array([[2.0, 2.0]])

        M = (
            self.p.m2 * (Jv2.T @ Jv2)
            + self.I2 * (Jw2.T @ Jw2)
            + self.p.m3 * (Jv3.T @ Jv3)
            + self.I3 * (Jw3.T @ Jw3)
            + self.p.m4 * (Jv4.T @ Jv4)
            + self.I4 * (Jw4.T @ Jw4)
            + self.p.m5 * (Jv5.T @ Jv5)
            + self.I5 * (Jw5.T @ Jw5)
        )
        return M.astype(np.float64)

    def coriolis_matrix(self, q: ArrayLike, dq: ArrayLike, eps: float = 1e-5) -> np.ndarray:
        """科氏/离心矩阵 C(q, dq), 由 Christoffel 符号数值构造."""
        q = np.asarray(q, dtype=np.float64).reshape(2)
        dq = np.asarray(dq, dtype=np.float64).reshape(2)
        C = np.zeros((2, 2), dtype=np.float64)
        for k in range(2):
            for j in range(2):
                c_kj = 0.0
                for i in range(2):
                    q_p = q.copy()
                    q_p[i] += eps
                    q_m = q.copy()
                    q_m[i] -= eps
                    dMkj_dqi = (self.mass_matrix(q_p)[k, j] - self.mass_matrix(q_m)[k, j]) / (2.0 * eps)

                    q_p = q.copy()
                    q_p[j] += eps
                    q_m = q.copy()
                    q_m[j] -= eps
                    dMki_dqj = (self.mass_matrix(q_p)[k, i] - self.mass_matrix(q_m)[k, i]) / (2.0 * eps)

                    q_p = q.copy()
                    q_p[k] += eps
                    q_m = q.copy()
                    q_m[k] -= eps
                    dMij_dqk = (self.mass_matrix(q_p)[i, j] - self.mass_matrix(q_m)[i, j]) / (2.0 * eps)

                    c_kj += 0.5 * (dMkj_dqi + dMki_dqj - dMij_dqk) * dq[i]
                C[k, j] = c_kj
        return C

    def inverse_dynamics(
        self, q: ArrayLike, dq: ArrayLike, ddq: ArrayLike
    ) -> np.ndarray:
        """给定 (q, dq, ddq) 计算所需关节力矩 tau."""
        q = np.asarray(q, dtype=np.float64).reshape(2)
        dq = np.asarray(dq, dtype=np.float64).reshape(2)
        ddq = np.asarray(ddq, dtype=np.float64).reshape(2)
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        return M @ ddq + C @ dq

    def forward_dynamics(
        self, q: ArrayLike, dq: ArrayLike, tau: ArrayLike
    ) -> np.ndarray:
        """给定 (q, dq, tau) 计算广义加速度 ddq."""
        q = np.asarray(q, dtype=np.float64).reshape(2)
        dq = np.asarray(dq, dtype=np.float64).reshape(2)
        tau = np.asarray(tau, dtype=np.float64).reshape(2)
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        return np.linalg.solve(M, tau - C @ dq)

    # ------------------------------------------------------------------
    # 时间积分 (半隐式 Euler, 与简单名义模型一致)
    # ------------------------------------------------------------------
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None,
        *,
        apply_joint_limits: bool = True,
    ) -> np.ndarray:
        """
        推进一个时间步.

        Parameters
        ----------
        x : shape (4,)  [q_a, q_b, dq_a, dq_b]
        u : shape (2,)  [tau_a, tau_b]
        dt : 步长, 默认 XML timestep 0.01
        apply_joint_limits : 是否施加 ±pi/2 硬限位 (与 XML joint range 一致)
        """
        dt = self.p.dt_default if dt is None else float(dt)
        q_a, q_b, dq_a, dq_b = unpack_state(x)
        q = np.array([q_a, q_b], dtype=np.float64)
        dq = np.array([dq_a, dq_b], dtype=np.float64)
        tau = np.asarray(u, dtype=np.float64).reshape(2)

        ddq = self.forward_dynamics(q, dq, tau)
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt

        if apply_joint_limits:
            lim = self.p.joint_limit
            for i in range(2):
                if q_next[i] > lim:
                    q_next[i] = lim
                    dq_next[i] = 0.0
                elif q_next[i] < -lim:
                    q_next[i] = -lim
                    dq_next[i] = 0.0

        return pack_state(q_next[0], q_next[1], dq_next[0], dq_next[1])

    def rollout(
        self,
        x0: ArrayLike,
        u_seq: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        开环滚动仿真.

        Returns
        -------
        states : shape (T+1, 4)
        """
        u_seq = np.asarray(u_seq, dtype=np.float64)
        if u_seq.ndim != 2 or u_seq.shape[1] != 2:
            raise ValueError("u_seq must have shape (T, 2)")
        T = u_seq.shape[0]
        traj = np.zeros((T + 1, 4), dtype=np.float64)
        traj[0] = np.asarray(x0, dtype=np.float64).reshape(4)
        for k in range(T):
            traj[k + 1] = self.step(traj[k], u_seq[k], dt=dt)
        return traj

    @staticmethod
    def state_dim() -> int:
        return 4

    @staticmethod
    def control_dim() -> int:
        return 2

    def info(self) -> str:
        """打印与 XML 对齐的模型摘要."""
        p = self.p
        lines = [
            f"CDSM rigid nominal model  (ref: {XML_MODEL_NAME}.xml)",
            f"  DOF: q_a (joint1=joint2), q_b (joint3=joint4)",
            f"  Links: L1={p.L1} (fixed), L2=L4={p.L2}, L3=L5={p.L3}, Ls1=Ls2={p.Ls1}",
            f"  Masses (movable): m2=m4={p.m2}, m3=m5={p.m3}",
            f"  dt={p.dt_default}, joint_limit=±{p.joint_limit:.4f} rad",
            f"  Dynamics: M(q) ddq + C(q,dq) dq = tau",
            f"  Control: tau_a -> motor_qa, tau_b -> motor_qb",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 便捷工厂
# ---------------------------------------------------------------------------
def make_nominal_model(*, dt: float = XML_TIMESTEP) -> CdsmRigidNominalModel:
    """构造名义模型 (仅 M, C 与关节力矩 tau)."""
    params = CdsmNominalParams(dt_default=dt)
    return CdsmRigidNominalModel(params)


# ---------------------------------------------------------------------------
# 自检: 质量矩阵对称正定 + 简单仿真
# ---------------------------------------------------------------------------
def _self_test() -> None:
    model = make_nominal_model()
    print(model.info())

    q = np.array([0.2, -0.15])
    dq = np.array([0.1, 0.05])
    M = model.mass_matrix(q)
    assert M.shape == (2, 2)
    assert np.allclose(M, M.T, atol=1e-10)
    eig = np.linalg.eigvalsh(M)
    assert np.all(eig > 0), f"M(q) must be SPD, eigenvalues={eig}"

    tau = np.array([3.0, -2.0])
    ddq = model.forward_dynamics(q, dq, tau)
    tau_check = model.inverse_dynamics(q, dq, ddq)
    assert np.allclose(tau, tau_check, atol=1e-8)

    x0 = pack_state(0.1, 0.0, 0.0, 0.0)
    u_seq = np.column_stack(
        [3.0 * np.sin(0.1 * np.arange(100)), 2.0 * np.cos(0.12 * np.arange(100))]
    )
    traj = model.rollout(x0, u_seq)
    p5 = model.end_effector_position(traj[-1, 0], traj[-1, 1])
    print(f"  100-step rollout OK, final EE position = [{p5[0]:.4f}, {p5[1]:.4f}] m")


if __name__ == "__main__":
    _self_test()
