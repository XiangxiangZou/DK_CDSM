"""统一 Koopman-space LQR/MPC 跟踪控制器。

本模块实现的控制器用于 EDMD、DKUC、DKAC 三类可控模型。它假设模型能暴露
常值矩阵 `A/B/C`，其中：
- EDMD/DKUC: B 对应标准化物理控制 `u_n`。
- DKAC: B 对应内部控制 `v=G(x_n)u_n`。

控制器只求内部控制序列，具体如何还原为物理关节力矩由模型适配器的
`recover_control(x, internal_control)` 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LqrConfig:
    """有限时域 LQR/MPC 权重配置。

    参数:
        horizon: 预测时域步数。
        Qq: 关节角误差权重，作用于 `qa,qb`。
        Qdq: 关节角速度误差权重，作用于 `dqa,dqb`。
        R: 控制量幅值权重。
        Rd: 控制增量权重，抑制相邻周期控制跳变。
    """

    horizon: int = 30
    Qq: float = 40.0
    Qdq: float = 2.0
    R: float = 1e-3
    Rd: float = 1e-2
    output_weights: tuple[float, ...] | None = None


class KoopmanLqrTracker:
    """常值 Koopman 线性系统上的凝聚式有限时域 LQR。

    参数:
        A: 潜空间状态矩阵。
        B: 潜空间控制矩阵，控制维度可为物理标准化控制或 DKAC 内部控制。
        C: 潜状态到标准化状态的读出矩阵。
        cfg: LQR 权重和时域配置。
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cfg: LqrConfig) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.cfg = cfg
        self._precompute()

    def _precompute(self) -> None:
        """预计算凝聚预测矩阵和 Hessian 逆。"""
        n_h = int(self.cfg.horizon)
        A, B, C = self.A, self.B, self.C
        nz, nu, ny = A.shape[0], B.shape[1], C.shape[0]

        phi = np.zeros((n_h * ny, nz), dtype=np.float64)
        gamma = np.zeros((n_h * ny, n_h * nu), dtype=np.float64)
        powers = [np.eye(nz)]
        for _ in range(n_h):
            powers.append(A @ powers[-1])
        for i in range(n_h):
            phi[i * ny : (i + 1) * ny] = C @ powers[i + 1]
            for j in range(i + 1):
                gamma[i * ny : (i + 1) * ny, j * nu : (j + 1) * nu] = C @ powers[i - j] @ B

        if self.cfg.output_weights is None:
            if ny != 4:
                raise ValueError(
                    "output_weights is required when the model output "
                    f"dimension is {ny}"
                )
            step_weights = np.array(
                [
                    self.cfg.Qq,
                    self.cfg.Qq,
                    self.cfg.Qdq,
                    self.cfg.Qdq,
                ],
                dtype=np.float64,
            )
        else:
            step_weights = np.asarray(
                self.cfg.output_weights,
                dtype=np.float64,
            )
            if step_weights.shape != (ny,):
                raise ValueError(
                    f"output_weights must have shape ({ny},), "
                    f"got {step_weights.shape}"
                )
        q_diag = np.tile(step_weights, n_h)
        qbar = np.diag(q_diag)
        rbar = np.eye(n_h * nu) * self.cfg.R
        rdbar = np.eye(n_h * nu) * self.cfg.Rd

        dmat = np.zeros((n_h * nu, n_h * nu), dtype=np.float64)
        for k in range(n_h):
            dmat[k * nu : (k + 1) * nu, k * nu : (k + 1) * nu] = np.eye(nu)
            if k > 0:
                dmat[k * nu : (k + 1) * nu, (k - 1) * nu : k * nu] = -np.eye(nu)
        emat = np.zeros((n_h * nu, nu), dtype=np.float64)
        emat[:nu, :] = np.eye(nu)

        hessian = gamma.T @ qbar @ gamma + rbar + dmat.T @ rdbar @ dmat
        hessian = hessian + 1e-9 * np.eye(hessian.shape[0])

        self.phi = phi
        self.gamma = gamma
        self.qbar = qbar
        self.rdbar = rdbar
        self.dmat = dmat
        self.emat = emat
        self.hinv = np.linalg.inv(hessian)
        self.nu = nu
        self.n_h = n_h

    def solve(self, z0: np.ndarray, ref_norm: np.ndarray, u_prev_internal: np.ndarray) -> np.ndarray:
        """求解当前周期内部控制序列。

        参数:
            z0: 当前潜空间状态。
            ref_norm: 未来 horizon 步标准化参考状态，形状 `(horizon,4)`。
            u_prev_internal: 上一周期已经执行的内部控制量。

        返回:
            内部控制序列，形状 `(horizon, nu)`；实际只执行第 0 行。
        """
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        ref = np.asarray(ref_norm, dtype=np.float64).reshape(-1)
        u_prev = np.asarray(u_prev_internal, dtype=np.float64).reshape(self.nu)
        free_response = self.phi @ z0
        grad = self.gamma.T @ self.qbar @ (free_response - ref)
        grad = grad - self.dmat.T @ self.rdbar @ self.emat @ u_prev
        sol = -self.hinv @ grad
        return sol.reshape(self.n_h, self.nu)


def build_ramp_reference(
    *,
    dt: float,
    duration: float,
    start: Sequence[float],
    target: Sequence[float],
    ramp_duration: float,
) -> dict[str, np.ndarray]:
    """Build a smooth vector-valued ramp reference."""
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if start_arr.shape != target_arr.shape:
        raise ValueError("start and target must have the same shape")
    n = int(round(float(duration) / float(dt))) + 1
    t = np.arange(n, dtype=np.float64) * float(dt)
    s = np.clip(
        t / max(float(ramp_duration), float(dt)),
        0.0,
        1.0,
    )
    s = 3.0 * s * s - 2.0 * s * s * s
    values = start_arr[None, :] + (
        target_arr - start_arr
    )[None, :] * s[:, None]
    rates = np.gradient(values, float(dt), axis=0)
    return {"t": t, "values": values, "rates": rates}


def future_reference(
    model,
    states_ref: np.ndarray,
    k: int,
    horizon: int,
) -> np.ndarray:
    """Return normalized references for x[k+1] through x[k+horizon]."""
    refs = np.asarray(states_ref, dtype=np.float64)
    out = np.zeros((horizon, refs.shape[1]), dtype=np.float64)
    n = refs.shape[0]
    for i in range(horizon):
        idx = min(k + 1 + i, n - 1)
        out[i] = model.x_normer.transform(refs[idx : idx + 1])[0]
    return out
