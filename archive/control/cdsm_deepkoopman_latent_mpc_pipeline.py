"""
绳驱空间机械臂 (CDSM) DeepKoopman 混合模型 + 线性 MPC 端到端流水线
===================================================================

本脚本是一个"单文件"编排入口, 串起从数据采集到闭环对比的完整链路:

  [1] 用 PD 控制器在 MuJoCo 绳驱模型上采集关节空间轨迹数据 (训练 / 验证集);
  [2] 用刚体名义模型计算一步残差 r_k = x_{k+1}^{MuJoCo} - f_nom(x_k, u_k),
      并训练一个 ResidualDeepKoopman 残差动力学网络;
  [3] 构建"混合模型" f_hybrid(x,u) = f_nom(x,u) + r_hat(x,u), 并在该混合模型上
      做线性 MPC; 同时跑一套"仅名义模型"的线性 MPC 作为对照;
  [4] 在 MuJoCo 被控对象上闭环对比两套 MPC, 保存日志 / 指标 / 论文风格图。

--------------------------------------------------------------------
混合模型与需求的一致性 (本次修改重点)
--------------------------------------------------------------------
残差 r 在 ``build_residual_dataset`` 中是相对 ``compute_nominal_next``
(即 ``nominal.step(..., apply_joint_limits=True)``) 定义的。因此与之自洽的
"混合一步映射"必须是:

    f_hybrid(x,u) = f_nom(x,u; limits=True) + r_hat(x,u)

本脚本的混合 MPC 不再使用"潜空间线性读出层直接回归 MuJoCo 全状态"的代理模型,
而是直接对上式 f_hybrid 做有限差分线性化 (与名义 MPC 完全相同的处理流程),
仅把被线性化的一步映射从 f_nom 换成 f_nom + r_hat。这样:
  * 闭环 MPC 用的"混合模型"与开环动力学响应图用的"混合模型"是同一个对象;
  * 名义 MPC 与混合 MPC 唯一区别就是预测模型 (f_nom vs f_nom+r_hat), 对比公平。

--------------------------------------------------------------------
MPC 的优化形式 (重要)
--------------------------------------------------------------------
两套 MPC 都是"凝聚式 (condensed) 无约束线性二次型 (LQ) 滚动时域最优控制":

  1) 在当前工作点 (x0, u_prev) 把一步离散映射线性化为仿射系统
        x_{k+1} ≈ A x_k + B u_k + c;
  2) 沿预测时域 N 把状态/输出表示成控制序列 U 的仿射函数 (凝聚):
        Y = y0 + S·U   (S 为单位阶跃响应拼成的预测矩阵);
  3) 代价为二次型:
        J(U) = (Y-R)ᵀ Q̄ (Y-R) + Uᵀ R̄ U + (ΔU)ᵀ R̄_d (ΔU),
     其中 R 是参考, ΔU 为控制增量 (相邻控制差, 首步相对 u_prev);
  4) 无任何不等式约束 (本项目刻意不做力矩/张力上限), 故这是一个
     **无约束凸二次规划 (QP)**, 存在闭式解, 由一次线性求解给出:
        H U* = -b,  H = SᵀQ̄S + R̄ + DᵀR̄_dD,  b = SᵀQ̄(y0-R) - DᵀR̄_d E u_prev。
     等价于一个线性二次跟踪 / 带正则的最小二乘问题, 不需要迭代 QP 求解器。

控制约定:
  * 不做关节力矩限幅;
  * 绳张力映射 ``cable_antagonistic_map`` 以 f_max=inf 调用, 即不设张力上限,
    仅保留预紧下限 F_PRELOAD。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# MuJoCo 离屏/窗口渲染后端 (在导入 mujoco 前设置)
os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

# 用非交互 Agg 后端, 仅出图存盘, 不弹窗 (适合批处理 / 无显示环境)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires PyTorch in the configured project environment.") from exc

# ----------------------------------------------------------------------------
# 复用 DeepKoopman 残差模型相关组件:
#   Normalizer            : 输入/输出标准化器 (z-score)
#   ResidualDeepKoopman   : z=encoder(x), z+=A z+B u, r_hat=decoder(z+)
#   compute_losses/...    : 训练 / 评估损失
#   evaluate_rollout      : 开环多步预测评估 (名义 vs 混合 vs MuJoCo)
#   predict_hybrid_next   : 混合一步映射 f_nom + r_hat  ← 混合 MPC 线性化的核心
#   predict_residual_batch: 单纯的残差预测 r_hat(x,u) (反标准化后)
# ----------------------------------------------------------------------------
from cdsm_hybrid_residual_deepkoopman import (
    Normalizer,
    ResidualDeepKoopman,
    compute_losses,
    evaluate_loss,
    evaluate_rollout,
    make_device,
    predict_hybrid_next,
    predict_residual_batch,
    sample_transition_batch,
)

# ----------------------------------------------------------------------------
# 复用绳驱 MuJoCo 工具:
#   CABLE_NAMES / F_PRELOAD       : 8 根绳名称 / 预紧张力 (N)
#   PDCollectConfig               : PD 采数配置 dataclass
#   build_residual_dataset        : 计算名义模型残差数据集
#   cable_antagonistic_map        : 关节力矩 -> 8 根绳张力 拮抗映射
#   compute_tendon_jacobian_fd    : 有限差分 tendon 雅可比 (8 x nv)
#   flatten_residual_data         : 将 (轨迹, 步) 残差数据摊平为样本矩阵
#   get_active_state/set_active_state : 读写 MuJoCo 主动关节状态 [qa,qb,dqa,dqb]
#   load_cable_model              : 载入 XML 并返回 (model,data,scratch,indices)
# ----------------------------------------------------------------------------
from cdsm_hybrid_residual_edmd import (
    CABLE_NAMES,
    F_PRELOAD,
    PDCollectConfig,
    build_residual_dataset,
    cable_antagonistic_map,
    compute_tendon_jacobian_fd,
    flatten_residual_data,
    get_active_state,
    load_cable_model,
    set_active_state,
)
from cdsm_mpc_tracking_compare import build_joint_reference
from cdsm_rigid_nominal_model import CdsmRigidNominalModel, make_nominal_model
from utils_plot import get_save_dir, save_figure


# XML 模型文件名 (绳驱空间机械臂)
XML_DEFAULT = str(
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "multi_joint_cable_driven_space_robot.xml"
)
# 状态分量标签: q_a/q_b 为两个主动关节角, dq_a/dq_b 为对应角速度
STATE_LABELS = ["q_a", "q_b", "dq_a", "dq_b"]


def set_seed(seed: int) -> None:
    """统一设置 NumPy / Python random / PyTorch (含 CUDA) 随机种子, 保证可复现。"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 配置 dataclass
# ============================================================================
@dataclass
class DeepKoopmanTrainConfig:
    """ResidualDeepKoopman 网络结构与训练超参数。"""

    latent_dim: int          # 潜空间维度
    hidden: Tuple[int, ...]  # 编码器/解码器 MLP 隐层宽度
    activation: str          # 激活函数名 (relu/elu/tanh)
    epochs: int              # 训练轮数
    steps_per_epoch: int     # 每轮小批量更新次数
    batch_size: int          # 批大小
    lr: float                # AdamW 学习率
    grad_clip: float         # 梯度范数裁剪上限 (<=0 关闭)
    w_residual: float        # 残差预测损失权重 ||r_hat-r||^2
    w_recon: float           # 状态重构损失权重 ||x_rec-x||^2
    w_linear: float          # 潜空间线性一致性损失权重
    w_l2: float              # 权重 L2 正则系数


@dataclass
class LinearMpcConfig:
    """线性 MPC (名义 / 混合通用) 的超参数。"""

    horizon: int   # 预测时域步数 N
    Qq: float      # 关节角跟踪误差权重 (作用于 q_a, q_b)
    Qdq: float     # 关节角速度跟踪误差权重 (作用于 dq_a, dq_b)
    R: float       # 控制量幅值权重 (作用于 tau)
    Rd: float      # 控制增量 ΔU 权重 (平滑项)
    fd_eps_x: float  # 有限差分线性化时状态扰动步长
    fd_eps_u: float  # 有限差分线性化时控制扰动步长


@dataclass
class SineRefParams:
    """单关节多正弦参考的参数 (用于 PD 采数激励)。"""

    a1: float   # 第一正弦幅值
    a2: float   # 第二正弦幅值
    w1: float   # 第一正弦角频率
    w2: float   # 第二正弦角频率
    phi2: float  # 第二正弦相位


# ============================================================================
# DeepKoopman 运行期封装: 仅用于在闭环 MPC 中提供"混合一步映射"
# ============================================================================
class DeepKoopmanRuntime:
    """
    训练好的残差网络在推理期的轻量封装。

    职责非常单一: 给定原始 (未标准化) 的 (x,u), 返回
        * 残差     r_hat(x,u)              -> predict_residual
        * 混合下一步 f_nom(x,u)+r_hat(x,u) -> hybrid_next
    供混合 MPC 做有限差分线性化使用。内部负责标准化/反标准化与设备搬运。
    """

    def __init__(
        self,
        model: ResidualDeepKoopman,
        x_normer: Normalizer,
        u_normer: Normalizer,
        r_normer: Normalizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.x_normer = x_normer  # 状态标准化器 (训练时拟合)
        self.u_normer = u_normer  # 控制标准化器
        self.r_normer = r_normer  # 残差标准化器 (用于把网络输出反标准化回物理量)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict_residual(self, x_raw: np.ndarray, u_raw: np.ndarray) -> np.ndarray:
        """返回单点残差 r_hat(x,u), 形状 (4,), 已反标准化为物理量。"""
        r = predict_residual_batch(
            self.model, x_raw, u_raw, self.x_normer, self.u_normer, self.r_normer, self.device
        )
        return np.asarray(r, dtype=np.float64).reshape(-1)

    def hybrid_next(
        self, nominal: CdsmRigidNominalModel, x_raw: np.ndarray, u_raw: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        混合一步映射 x_{k+1} = f_nom(x_k,u_k) + r_hat(x_k,u_k)。

        直接复用 ``predict_hybrid_next``, 其内部名义部分用
        ``compute_nominal_next`` (apply_joint_limits=True), 与残差训练时的
        名义约定完全一致, 因此 f_nom+r_hat 与训练数据自洽。
        """
        return predict_hybrid_next(
            self.model,
            nominal,
            x_raw,
            u_raw,
            dt,
            self.x_normer,
            self.u_normer,
            self.r_normer,
            self.device,
        )


# ============================================================================
# [1] PD 采数: 多正弦参考激励 + 拮抗绳驱映射
# ============================================================================
def sample_sine_params(rng: np.random.RandomState, cfg: PDCollectConfig) -> SineRefParams:
    """从配置的幅值/角频率范围里随机生成一组多正弦参考参数。"""
    amp_lo, amp_hi = cfg.amp_range
    w_lo, w_hi = cfg.omega_range
    return SineRefParams(
        a1=float(rng.uniform(amp_lo, amp_hi)),
        a2=float(rng.uniform(0.5 * amp_lo, 0.8 * amp_hi)),
        w1=float(rng.uniform(w_lo, w_hi)),
        w2=float(rng.uniform(1.3 * w_lo, 1.8 * w_hi)),
        phi2=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def eval_sine_ref(params: SineRefParams, t: float) -> Tuple[float, float]:
    """在时刻 t 求多正弦参考的位置 q 及其解析速度 dq。"""
    q = params.a1 * np.sin(params.w1 * t) + params.a2 * np.sin(params.w2 * t + params.phi2)
    dq = params.a1 * params.w1 * np.cos(params.w1 * t) + params.a2 * params.w2 * np.cos(
        params.w2 * t + params.phi2
    )
    return float(q), float(dq)


def collect_pd_trajectories_unlimited(
    model: object,
    data: object,
    scratch: object,
    indices: Dict[str, np.ndarray],
    cfg: PDCollectConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    用 PD 控制器在 MuJoCo 上采集多条轨迹, 不设关节力矩 / 绳张力上限。

    每条轨迹: 随机初始 (q0,dq0) + 每个关节一组随机多正弦参考; 每步用
        tau = Kp*(q_ref-q) + Kd*(dq_ref-dq)
    计算期望关节力矩, 再经拮抗映射 (f_max=inf) 转成 8 根绳张力下发并积分。

    返回
    ----
    arrays : dict
        states     (traj, steps+1, 4)  关节状态序列 [qa,qb,dqa,dqb]
        inputs     (traj, steps,   2)  期望关节力矩 [tau_a,tau_b] (训练用控制量 u)
        q_ref      (traj, steps,   2)  PD 参考关节角
        cable_ctrl (traj, steps,   8)  实际下发的 8 根绳张力
    meta : dict  采集元信息 (限幅是否开启等)
    """
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for data collection.") from exc

    rng = np.random.RandomState(cfg.seed)
    states = np.zeros((cfg.traj_count, cfg.steps + 1, 4), dtype=np.float64)
    inputs = np.zeros((cfg.traj_count, cfg.steps, 2), dtype=np.float64)
    cable_ctrl = np.zeros((cfg.traj_count, cfg.steps, 8), dtype=np.float64)
    q_ref_hist = np.zeros((cfg.traj_count, cfg.steps, 2), dtype=np.float64)

    kp = np.asarray(cfg.kp, dtype=np.float64)
    kd = np.asarray(cfg.kd, dtype=np.float64)
    # 4 个关节自由度的 DOF 索引 (joint1..joint4); 主动关节为 joint1(qa)/joint3(qb),
    # 其余两个为同级耦合自由度, 拮抗映射时需要全部 4 个力臂列。
    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])

    for i in range(cfg.traj_count):
        # --- 随机初始状态 ---
        q0 = rng.uniform(-cfg.q_init_range, cfg.q_init_range, size=2)
        dq0 = rng.uniform(-cfg.dq_init_range, cfg.dq_init_range, size=2)
        set_active_state(model, data, indices, q0, dq0)
        states[i, 0] = get_active_state(data, indices)

        # --- 两个关节各一组随机多正弦参考 ---
        ref_a = sample_sine_params(rng, cfg)
        ref_b = sample_sine_params(rng, cfg)
        for k in range(cfg.steps):
            t = k * cfg.dt
            qa_ref, dqa_ref = eval_sine_ref(ref_a, t)
            qb_ref, dqb_ref = eval_sine_ref(ref_b, t)
            q_ref = np.array([qa_ref, qb_ref], dtype=np.float64)
            dq_ref = np.array([dqa_ref, dqb_ref], dtype=np.float64)
            q_ref_hist[i, k] = q_ref

            # --- PD 律计算期望关节力矩 ---
            x = get_active_state(data, indices)
            q = x[:2]
            dq = x[2:]
            tau = kp * (q_ref - q) + kd * (dq_ref - dq)

            # --- 关节力矩 -> 8 根绳张力 (无上限, 仅预紧下限) ---
            J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
            F_cable = cable_antagonistic_map(
                float(tau[0]),
                float(tau[1]),
                J,
                dof_j1,
                dof_j2,
                dof_j3,
                dof_j4,
                f_pre=F_PRELOAD,
                f_max=float("inf"),
            )
            data.ctrl[indices["actuator_ids"]] = F_cable
            mujoco.mj_step(model, data)

            # --- 记录该步数据 ---
            states[i, k + 1] = get_active_state(data, indices)
            inputs[i, k] = tau
            cable_ctrl[i, k] = F_cable

    meta: Dict[str, object] = {
        "kp": list(cfg.kp),
        "kd": list(cfg.kd),
        "torque_limit_enabled": False,
        "f_preload": F_PRELOAD,
        "cable_tension_limit_enabled": False,
        "control_mode": "pd_joint_torque_via_unlimited_cable_map",
    }
    return {"states": states, "inputs": inputs, "q_ref": q_ref_hist, "cable_ctrl": cable_ctrl}, meta


# ============================================================================
# [2] 训练 ResidualDeepKoopman 残差动力学网络
# ============================================================================
def train_deepkoopman(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    cfg: DeepKoopmanTrainConfig,
    device: torch.device,
    out_dir: Path,
) -> Tuple[ResidualDeepKoopman, List[List[float]], Dict[str, float]]:
    """
    在 (已标准化的) 残差数据上训练 ResidualDeepKoopman, 按验证集总损失保存最优权重。

    数据约定 (train_arrays / val_arrays):
        x  : 标准化状态  x_k
        u  : 标准化控制  u_k
        xp : 标准化下一状态 x_{k+1}  (用于潜空间线性一致性损失)
        r  : 标准化残差  r_k = x_{k+1} - f_nom(x_k,u_k)
    返回最优模型, 训练历史 (逐 epoch), 以及 best_val/best_epoch 统计。
    """
    model = ResidualDeepKoopman(
        state_dim=4,
        control_dim=2,
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        activation=cfg.activation,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    best_val = float("inf")
    best_epoch = 0
    history: List[List[float]] = []
    # evaluate_loss 用 argparse.Namespace 形式读取损失权重, 这里包一层
    loss_args = argparse.Namespace(
        w_residual=cfg.w_residual,
        w_recon=cfg.w_recon,
        w_linear=cfg.w_linear,
        w_l2=cfg.w_l2,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        acc = {"total": 0.0, "residual": 0.0, "recon": 0.0, "linear": 0.0}
        for _ in range(cfg.steps_per_epoch):
            # 随机采一个小批量转移样本
            batch = sample_transition_batch(
                train_arrays["x"],
                train_arrays["u"],
                train_arrays["xp"],
                train_arrays["r"],
                cfg.batch_size,
                device,
            )
            losses = compute_losses(
                model,
                *batch,
                w_residual=cfg.w_residual,
                w_recon=cfg.w_recon,
                w_linear=cfg.w_linear,
                w_l2=cfg.w_l2,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for key in acc:
                acc[key] += float(losses[key].detach().item())
        for key in acc:
            acc[key] /= max(cfg.steps_per_epoch, 1)

        # 用验证集若干批量估计泛化损失
        val_loss = evaluate_loss(model, val_arrays, cfg.batch_size, device, loss_args, n_batches=5)
        history.append(
            [
                float(epoch),
                acc["total"],
                acc["residual"],
                acc["recon"],
                acc["linear"],
                val_loss["total"],
                val_loss["residual"],
                val_loss["recon"],
                val_loss["linear"],
            ]
        )
        print(
            f"[dk] epoch {epoch:03d}/{cfg.epochs:03d} "
            f"train={acc['total']:.3e} val={val_loss['total']:.3e}",
            flush=True,
        )
        # 保存验证集总损失最优的检查点
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "train_config": asdict(cfg),
                    "best_val": best_val,
                    "epoch": epoch,
                },
                out_dir / "best_residual_deepkoopman.pt",
            )

    # 训练结束后回载最优权重
    ckpt = torch.load(out_dir / "best_residual_deepkoopman.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, history, {"best_val": float(best_val), "best_epoch": float(best_epoch)}


# ============================================================================
# [3] 线性 MPC 的通用积木: 有限差分线性化 / 凝聚 / 无约束 QP 闭式解
# ============================================================================
def finite_difference_nominal_linearization(
    nominal: CdsmRigidNominalModel,
    x0: np.ndarray,
    u0: np.ndarray,
    dt: float,
    eps_x: float,
    eps_u: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对名义一步映射 f_nom 在 (x0,u0) 处做中心差分线性化, 得到仿射模型
        x_{k+1} ≈ A x_k + B u_k + c。

    A = ∂f/∂x|_(x0,u0) (4x4), B = ∂f/∂u|_(x0,u0) (4x2),
    c = f0 - A x0 - B u0 (使线性化在工作点处精确成立)。
    注意: 此处名义模型用 apply_joint_limits=False, 避免限幅带来的不可微跳变。
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(4)
    u0 = np.asarray(u0, dtype=np.float64).reshape(2)
    f0 = nominal.step(x0, u0, dt=dt, apply_joint_limits=False)
    A = np.zeros((4, 4), dtype=np.float64)
    B = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps_x
        fp = nominal.step(x0 + dx, u0, dt=dt, apply_joint_limits=False)
        fm = nominal.step(x0 - dx, u0, dt=dt, apply_joint_limits=False)
        A[:, i] = (fp - fm) / (2.0 * eps_x)
    for j in range(2):
        du = np.zeros(2)
        du[j] = eps_u
        fp = nominal.step(x0, u0 + du, dt=dt, apply_joint_limits=False)
        fm = nominal.step(x0, u0 - du, dt=dt, apply_joint_limits=False)
        B[:, j] = (fp - fm) / (2.0 * eps_u)
    c = f0 - A @ x0 - B @ u0
    return A, B, c


def finite_difference_hybrid_linearization(
    runtime: DeepKoopmanRuntime,
    nominal: CdsmRigidNominalModel,
    x0: np.ndarray,
    u0: np.ndarray,
    dt: float,
    eps_x: float,
    eps_u: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对**混合一步映射** f_hybrid(x,u) = f_nom(x,u) + r_hat(x,u) 在 (x0,u0) 处做
    中心差分线性化, 得到仿射模型 x_{k+1} ≈ A x_k + B u_k + c。

    与 ``finite_difference_nominal_linearization`` 的唯一区别是被线性化的一步
    映射多了残差网络项 r_hat, 因此 A/B 同时包含了名义模型与残差的局部灵敏度。
    每次线性化共调用 f_hybrid (1 + 2*4 + 2*2) = 13 次, 每次含一次残差前向。
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(4)
    u0 = np.asarray(u0, dtype=np.float64).reshape(2)

    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return runtime.hybrid_next(nominal, x, u, dt)

    f0 = f(x0, u0)
    A = np.zeros((4, 4), dtype=np.float64)
    B = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps_x
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2.0 * eps_x)
    for j in range(2):
        du = np.zeros(2)
        du[j] = eps_u
        B[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2.0 * eps_u)
    c = f0 - A @ x0 - B @ u0
    return A, B, c


def simulate_affine(A: np.ndarray, B: np.ndarray, c: np.ndarray, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """前向仿真仿射系统 x_{k+1}=A x_k+B u_k+c, 返回 (N+1, dim) 的状态轨迹。"""
    X = np.zeros((U.shape[0] + 1, A.shape[0]), dtype=np.float64)
    X[0] = x0
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    for k, u in enumerate(U):
        x = A @ x + B @ u + c
        X[k + 1] = x
    return X


def condense_affine_system(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    x0: np.ndarray,
    N: int,
    output_fn,
    out_dim: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把仿射预测模型"凝聚"成关于控制序列 U 的仿射输出表达:
        Y(U) = y0 + S·U,
    其中 Y 为时域 1..N 的输出堆叠 (每步 output_fn(x) 维度 out_dim)。

    实现: 由于系统对 U 是仿射的,
        y0 = 控制全 0 时的输出轨迹 (零输入响应);
        S 的第 j 列 = 把 U 的第 j 个分量置 1 (其余 0) 后的输出 减去 y0
                    (即单位脉冲/阶跃响应), 因仿射性该差值精确且与工作点无关。
    """
    U0 = np.zeros((N, 2), dtype=np.float64)
    X0 = simulate_affine(A, B, c, x0, U0)
    y0 = np.vstack([output_fn(X0[k + 1]) for k in range(N)]).reshape(-1)
    S = np.zeros((N * out_dim, N * 2), dtype=np.float64)
    for j in range(N * 2):
        U = U0.copy()
        U.reshape(-1)[j] = 1.0
        X = simulate_affine(A, B, c, x0, U)
        y = np.vstack([output_fn(X[k + 1]) for k in range(N)]).reshape(-1)
        S[:, j] = y - y0
    return y0, S


def solve_unconstrained_linear_mpc(
    y0: np.ndarray,
    S: np.ndarray,
    ref_slice: Dict[str, np.ndarray],
    u_prev: np.ndarray,
    cfg: LinearMpcConfig,
) -> np.ndarray:
    """
    求解无约束线性二次型 (LQ) 滚动时域问题的闭式解。

    预测输出: Y = y0 + S U, 其中状态输出按 [q_a,q_b,dq_a,dq_b] 排列。
    参考:     R 由 q_ref/dq_ref 拼成 (角速度参考也纳入跟踪)。
    代价:
        J(U) = (Y-R)ᵀ Q̄ (Y-R) + Uᵀ R̄ U + (D U - E u_prev)ᵀ R̄_d (D U - E u_prev)
    其中
        Q̄   = blkdiag(Qq,Qq,Qdq,Qdq) 重复 N 次   (跟踪权重)
        R̄   = R · I                               (控制幅值权重)
        R̄_d = Rd · I                              (控制增量权重)
        D, E: 构造控制增量 ΔU_k = u_k - u_{k-1}, 首步 ΔU_0 = u_0 - u_prev。
    无不等式约束 => 凸 QP 有闭式最优:
        H U* = -b,
        H = SᵀQ̄S + R̄ + DᵀR̄_dD (+1e-9 I 数值正则),
        b = SᵀQ̄(y0-R) - DᵀR̄_d E u_prev。
    返回整段最优控制序列 U* (N,2); 滚动时域只取首步执行。
    """
    N = cfg.horizon
    # 参考: 取预测时域内的 q_ref/dq_ref (索引 1..N, 对应预测状态 x_1..x_N)
    q_ref = ref_slice["q_ref"][1 : N + 1]
    dq_ref = ref_slice["dq_ref"][1 : N + 1]
    x_ref = np.zeros((N, 4), dtype=np.float64)
    x_ref[:, :2] = q_ref
    x_ref[:, 2:] = dq_ref
    r = x_ref.reshape(-1)

    # 各项权重矩阵 (对角)
    q_diag = np.tile(np.array([cfg.Qq, cfg.Qq, cfg.Qdq, cfg.Qdq], dtype=np.float64), N)
    Qbar = np.diag(q_diag)
    Rbar = np.eye(2 * N, dtype=np.float64) * cfg.R
    Rdbar = np.eye(2 * N, dtype=np.float64) * cfg.Rd

    # 控制增量算子: ΔU = D U - E u_prev
    D = np.zeros((2 * N, 2 * N), dtype=np.float64)
    for k in range(N):
        D[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = np.eye(2)
        if k > 0:
            D[2 * k : 2 * k + 2, 2 * (k - 1) : 2 * (k - 1) + 2] = -np.eye(2)
    E = np.zeros((2 * N, 2), dtype=np.float64)
    E[:2, :] = np.eye(2)

    # 组装 Hessian 与梯度项, 闭式求解 H U = -b
    H = S.T @ Qbar @ S + Rbar + D.T @ Rdbar @ D + np.eye(2 * N) * 1e-9
    b = S.T @ Qbar @ (y0 - r) - D.T @ Rdbar @ E @ np.asarray(u_prev, dtype=np.float64).reshape(2)
    try:
        U = -np.linalg.solve(H, b)
    except np.linalg.LinAlgError:
        # H 病态时退化到最小二乘解
        U = -np.linalg.lstsq(H, b, rcond=1e-8)[0]
    return U.reshape(N, 2)


def pad_ref_slice(ref: Dict[str, np.ndarray], k: int, horizon: int) -> Dict[str, np.ndarray]:
    """
    取从第 k 步起、长度 horizon+1 的参考切片; 越界部分用末值填充 (zero-order hold),
    保证预测时域末尾总有参考可用。
    """
    k_end = min(k + horizon + 1, ref["q_ref"].shape[0])
    q_ref = ref["q_ref"][k:k_end]
    dq_ref = ref["dq_ref"][k:k_end]
    if q_ref.shape[0] < horizon + 1:
        pad = horizon + 1 - q_ref.shape[0]
        q_ref = np.vstack([q_ref, np.tile(q_ref[-1], (pad, 1))])
        dq_ref = np.vstack([dq_ref, np.tile(dq_ref[-1], (pad, 1))])
    return {"q_ref": q_ref, "dq_ref": dq_ref}


def solve_nominal_linear_mpc(
    nominal: CdsmRigidNominalModel,
    x0: np.ndarray,
    ref_slice: Dict[str, np.ndarray],
    u_prev: np.ndarray,
    cfg: LinearMpcConfig,
    dt: float,
) -> np.ndarray:
    """名义模型线性 MPC: 线性化 f_nom -> 凝聚 -> 无约束 QP 闭式解。"""
    A, B, c = finite_difference_nominal_linearization(
        nominal, x0, u_prev, dt, cfg.fd_eps_x, cfg.fd_eps_u
    )
    # 输出即状态本身 (output_fn = 恒等), 跟踪 [q,dq]
    y0, S = condense_affine_system(A, B, c, x0, cfg.horizon, lambda x: x, out_dim=4)
    return solve_unconstrained_linear_mpc(y0, S, ref_slice, u_prev, cfg)


def solve_hybrid_linear_mpc(
    runtime: DeepKoopmanRuntime,
    nominal: CdsmRigidNominalModel,
    x0: np.ndarray,
    ref_slice: Dict[str, np.ndarray],
    u_prev: np.ndarray,
    cfg: LinearMpcConfig,
    dt: float,
) -> np.ndarray:
    """
    混合模型线性 MPC: 线性化 f_hybrid = f_nom + r_hat -> 凝聚 -> 无约束 QP 闭式解。

    与名义 MPC 的处理流程逐字相同, 仅把被线性化的一步映射从 f_nom 换成
    f_nom + r_hat, 因此该控制器使用的预测模型与需求 2 的"混合模型"一致。
    """
    A, B, c = finite_difference_hybrid_linearization(
        runtime, nominal, x0, u_prev, dt, cfg.fd_eps_x, cfg.fd_eps_u
    )
    y0, S = condense_affine_system(A, B, c, x0, cfg.horizon, lambda x: x, out_dim=4)
    return solve_unconstrained_linear_mpc(y0, S, ref_slice, u_prev, cfg)


# ============================================================================
# 闭环: 在 MuJoCo 被控对象上跑一套 MPC
# ============================================================================
def run_mpc_on_mujoco(
    *,
    label: str,
    nominal: CdsmRigidNominalModel,
    runtime: Optional[DeepKoopmanRuntime],
    mpc_cfg: LinearMpcConfig,
    xml: str,
    dt: float,
    ref: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    在 MuJoCo 真值对象上闭环运行 MPC 并记录日志。

    runtime is None  -> 名义模型线性 MPC (对照组);
    runtime 非空     -> 混合模型 (f_nom+r_hat) 线性 MPC。
    每步: 量测状态 -> 求解 MPC 取首步关节力矩 -> 拮抗映射成绳张力 -> 推进一步。
    记录 t / x / u / q_ref / dq_ref / 求解耗时 / 8 根绳张力。
    """
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for closed-loop comparison.") from exc

    model, data, scratch, indices = load_cable_model(xml, dt)
    # 初始状态对齐到参考首点
    set_active_state(model, data, indices, ref["q_ref"][0], ref["dq_ref"][0])
    mujoco.mj_forward(model, data)

    n_step = len(ref["t"]) - 1
    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])
    u_prev = np.zeros(2, dtype=np.float64)  # 上一步控制 (用于 ΔU 平滑项)

    rec = {
        "t": [],
        "x": [],
        "u": [],
        "q_ref": [],
        "dq_ref": [],
        "solve_ms": [],
        "cable_tensions": [],
    }
    print(f"[mpc:{label}] start steps={n_step} horizon={mpc_cfg.horizon}", flush=True)
    t0 = time.perf_counter()
    report_every = max(1, n_step // 5)
    for k in range(n_step):
        x_meas = get_active_state(data, indices)
        ref_slice = pad_ref_slice(ref, k, mpc_cfg.horizon)

        # --- 求解 MPC (名义 / 混合) ---
        tic = time.perf_counter()
        if runtime is None:
            U = solve_nominal_linear_mpc(nominal, x_meas, ref_slice, u_prev, mpc_cfg, dt)
        else:
            U = solve_hybrid_linear_mpc(runtime, nominal, x_meas, ref_slice, u_prev, mpc_cfg, dt)
        solve_ms = 1e3 * (time.perf_counter() - tic)
        u_cmd = U[0].copy()  # 滚动时域: 只执行首步控制

        # --- 关节力矩 -> 8 根绳张力 (无上限) -> 推进 MuJoCo ---
        J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
        F_cable = cable_antagonistic_map(
            float(u_cmd[0]),
            float(u_cmd[1]),
            J,
            dof_j1,
            dof_j2,
            dof_j3,
            dof_j4,
            f_pre=F_PRELOAD,
            f_max=float("inf"),
        )
        data.ctrl[indices["actuator_ids"]] = F_cable
        mujoco.mj_step(model, data)

        # --- 记录 ---
        rec["t"].append(float(k * dt))
        rec["x"].append(x_meas.copy())
        rec["u"].append(u_cmd.copy())
        rec["q_ref"].append(ref["q_ref"][k].copy())
        rec["dq_ref"].append(ref["dq_ref"][k].copy())
        rec["solve_ms"].append(float(solve_ms))
        rec["cable_tensions"].append(F_cable.copy())
        u_prev = u_cmd

        if (k + 1) % report_every == 0 or k == 0 or k + 1 == n_step:
            err = float(np.linalg.norm(x_meas[:2] - ref["q_ref"][k]))
            print(f"[mpc:{label}] step {k + 1}/{n_step} solve={solve_ms:.2f}ms |e_q|={err:.4g}")

    print(f"[mpc:{label}] done elapsed={time.perf_counter() - t0:.2f}s", flush=True)
    out = {key: np.asarray(value) for key, value in rec.items()}
    out["label"] = np.array([label])
    return out


# ============================================================================
# 指标与持久化
# ============================================================================
def tracking_metrics(log: Dict[str, np.ndarray]) -> Dict[str, float]:
    """从闭环日志计算跟踪指标: 关节角/角速度 RMSE、峰值误差、求解耗时、峰值力矩/张力。"""
    e_q = log["x"][:, :2] - log["q_ref"]
    e_dq = log["x"][:, 2:] - log["dq_ref"]
    return {
        "rmse_q": float(np.sqrt(np.mean(e_q * e_q))),
        "rmse_q_by_joint": np.sqrt(np.mean(e_q * e_q, axis=0)).tolist(),
        "max_abs_q": float(np.max(np.abs(e_q))),
        "rmse_dq": float(np.sqrt(np.mean(e_dq * e_dq))),
        "mean_solve_ms": float(np.mean(log["solve_ms"])),
        "peak_abs_tau": float(np.max(np.abs(log["u"]))),
        "peak_cable_tension": float(np.max(log["cable_tensions"])),
    }


def metrics_to_json(metrics: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    """把 evaluate_rollout 的动力学指标 (含 numpy 数组) 转成可 JSON 序列化的字典。"""
    payload: Dict[str, object] = {}
    for key, m in metrics.items():
        payload[key] = {
            "total_rmse": float(m["total_rmse"][0]),
            "rmse_by_state": m["rmse_by_state"].tolist(),
            "total_mae": float(m["total_mae"][0]),
            "mae_by_state": m["mae_by_state"].tolist(),
        }
        if "step_rmse" in m:
            payload[key]["step_rmse"] = m["step_rmse"].tolist()
    return payload


def save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    """把日志字典存为 .npz。"""
    np.savez(path, **{key: value for key, value in payload.items()})


# ============================================================================
# [4] 出图
# ============================================================================
def plot_training_history(history: List[List[float]]) -> None:
    """训练曲线: 训练/验证总损失随 epoch 变化 (对数纵轴)。"""
    arr = np.asarray(history, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(arr[:, 0], arr[:, 1], label="train total")
    ax.semilogy(arr[:, 0], arr[:, 5], label="val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure("training_history")
    plt.close(fig)


def plot_dynamics_response(eval_result: Dict[str, object], dt: float, traj_idx: int) -> None:
    """
    动力学响应对比图: 对某条验证轨迹, 画 MuJoCo 真值 vs 名义模型 vs 混合模型
    (f_nom+r_hat) 的开环多步预测, 4 个状态分量各一子图。
    """
    true = eval_result["states_true"][traj_idx]
    nom = eval_result["pred_nominal"][traj_idx]
    hyb = eval_result["pred_hybrid"][traj_idx]
    t = np.arange(true.shape[0]) * dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for i, label in enumerate(STATE_LABELS):
        ax = axes[i // 2, i % 2]
        ax.plot(t, true[:, i], "k-", lw=1.8, label="MuJoCo")
        ax.plot(t, nom[:, i], "--", lw=1.3, label="Nominal")
        ax.plot(t, hyb[:, i], "-.", lw=1.4, label="Hybrid DK")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Dynamic response: MuJoCo vs nominal vs hybrid DeepKoopman")
    fig.tight_layout()
    save_figure("dynamic_response_compare")
    plt.close(fig)


def plot_tracking(log_nom: Dict[str, np.ndarray], log_hyb: Dict[str, np.ndarray]) -> None:
    """关节空间轨迹跟踪图: 参考 vs 名义 MPC vs 混合 MPC, 两关节各一子图。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, name in enumerate(["q_a", "q_b"]):
        axes[j].plot(log_nom["t"], log_nom["q_ref"][:, j], "k--", lw=1.5, label="ref")
        axes[j].plot(log_nom["t"], log_nom["x"][:, j], lw=1.4, label="Nominal linear MPC")
        axes[j].plot(log_hyb["t"], log_hyb["x"][:, j], lw=1.4, label="Hybrid linear MPC")
        axes[j].set_ylabel(f"{name} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint-space trajectory tracking")
    fig.tight_layout()
    save_figure("joint_tracking")
    plt.close(fig)


def plot_tracking_error(log_nom: Dict[str, np.ndarray], log_hyb: Dict[str, np.ndarray]) -> None:
    """关节空间跟踪误差图: e_q = q - q_ref 随时间变化, 两关节各一子图。"""
    e_nom = log_nom["x"][:, :2] - log_nom["q_ref"]
    e_hyb = log_hyb["x"][:, :2] - log_hyb["q_ref"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, name in enumerate(["q_a", "q_b"]):
        axes[j].plot(log_nom["t"], e_nom[:, j], lw=1.4, label="Nominal linear MPC")
        axes[j].plot(log_hyb["t"], e_hyb[:, j], lw=1.4, label="Hybrid linear MPC")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"e_{name} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint-space tracking error")
    fig.tight_layout()
    save_figure("tracking_error")
    plt.close(fig)


def plot_tracking_rmse(log_nom: Dict[str, np.ndarray], log_hyb: Dict[str, np.ndarray]) -> None:
    """跟踪 RMSE 柱状图: 名义 vs 混合, 按关节分组。"""
    e_nom = log_nom["x"][:, :2] - log_nom["q_ref"]
    e_hyb = log_hyb["x"][:, :2] - log_hyb["q_ref"]
    rmse_nom = np.sqrt(np.mean(e_nom * e_nom, axis=0))
    rmse_hyb = np.sqrt(np.mean(e_hyb * e_hyb, axis=0))
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    ax.bar(x - width / 2, rmse_nom, width, label="Nominal linear MPC")
    ax.bar(x + width / 2, rmse_hyb, width, label="Hybrid linear MPC")
    ax.set_xticks(x)
    ax.set_xticklabels(["q_a", "q_b"])
    ax.set_ylabel("RMSE (rad)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure("tracking_rmse")
    plt.close(fig)


def plot_torques(log_nom: Dict[str, np.ndarray], log_hyb: Dict[str, np.ndarray]) -> None:
    """关节力矩图: MPC 输出的期望关节力矩 tau_a/tau_b 随时间变化 (无限幅)。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, name in enumerate(["tau_a", "tau_b"]):
        axes[j].plot(log_nom["t"], log_nom["u"][:, j], lw=1.4, label="Nominal linear MPC")
        axes[j].plot(log_hyb["t"], log_hyb["u"][:, j], lw=1.4, label="Hybrid linear MPC")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"{name} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint torque commands, no torque clipping")
    fig.tight_layout()
    save_figure("joint_torques")
    plt.close(fig)


def plot_cable_tensions(log_nom: Dict[str, np.ndarray], log_hyb: Dict[str, np.ndarray]) -> None:
    """MuJoCo 上全部 8 根绳张力图: 名义 / 混合各一子图, 标注预紧线。"""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, log, title in [
        (axes[0], log_nom, "Nominal linear MPC cable tensions"),
        (axes[1], log_hyb, "Hybrid linear MPC cable tensions"),
    ]:
        for i, name in enumerate(CABLE_NAMES):
            ax.plot(log["t"], log["cable_tensions"][:, i], lw=1.2, color=colors[i], label=name)
        ax.axhline(F_PRELOAD, color="k", lw=0.8, alpha=0.5, label=f"F_pre={F_PRELOAD:.0f} N")
        ax.set_title(title)
        ax.set_ylabel("Tension (N)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("All MuJoCo cable tension commands, no upper tension clipping")
    fig.tight_layout()
    save_figure("cable_tensions")
    plt.close(fig)


# ============================================================================
# 命令行参数
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器 (默认值为快速冒烟级别, 出论文图需调大数据/训练规模)。"""
    p = argparse.ArgumentParser(description="CDSM DeepKoopman hybrid model + linear MPC pipeline.")
    # --- 全局 ---
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # --- 数据采集 (PD 多正弦) ---
    p.add_argument("--train_traj", type=int, default=160)
    p.add_argument("--val_traj", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--q_init_range", type=float, default=1.2)
    p.add_argument("--dq_init_range", type=float, default=1.0)
    p.add_argument("--amp_min", type=float, default=-1.2)
    p.add_argument("--amp_max", type=float, default=1.2)
    p.add_argument("--omega_min", type=float, default=0.3)
    p.add_argument("--omega_max", type=float, default=1.6)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)

    # --- DeepKoopman 网络与训练 ---
    p.add_argument("--latent_dim", type=int, default=48)
    p.add_argument("--hidden", type=int, nargs="+", default=[256,512,256])
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--w_residual", type=float, default=1.0)
    p.add_argument("--w_recon", type=float, default=0.05)
    p.add_argument("--w_linear", type=float, default=0.2)
    p.add_argument("--w_l2", type=float, default=1e-8)

    # --- 跟踪参考 与 MPC ---
    p.add_argument("--T_track", type=float, default=30.0)
    p.add_argument("--T_ramp", type=float, default=30.0)
    p.add_argument("--qa0", type=float, default=-0.8)
    p.add_argument("--qa1", type=float, default=0.8)
    p.add_argument("--qb0", type=float, default=0.6)
    p.add_argument("--qb1", type=float, default=-0.6)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--Qq", type=float, default=400.0)
    p.add_argument("--Qdq", type=float, default=1.0)
    p.add_argument("--R", type=float, default=1e-6)
    p.add_argument("--Rd", type=float, default=1e-4)
    p.add_argument("--fd_eps_x", type=float, default=1e-5)
    p.add_argument("--fd_eps_u", type=float, default=1e-4)
    p.add_argument("--demo_traj", type=int, default=0)
    return p


def main() -> None:
    """端到端编排: 采数 -> 残差数据 -> 训练 -> 评估出图 -> 闭环 MPC 对比 -> 汇总。"""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== CDSM DeepKoopman hybrid + linear MPC ===")
    print(f"device={device}, output={out_dir}")
    print("limits: joint torque clipping disabled; cable upper tension clipping disabled")

    # ---- PD 采数配置 (训练 / 验证) ----
    pd_train = PDCollectConfig(
        traj_count=args.train_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        q_init_range=args.q_init_range,
        dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max),
        omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b),
        kd=(args.kd_a, args.kd_b),
        tau_max=float("inf"),
    )
    pd_val = PDCollectConfig(
        traj_count=args.val_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed + 1000,
        q_init_range=args.q_init_range,
        dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max),
        omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b),
        kd=(args.kd_a, args.kd_b),
        tau_max=float("inf"),
    )

    # ---- [1/6] 采集 PD 轨迹数据 ----
    print("[1/6] Collecting unlimited PD MuJoCo trajectories...")
    mj_model, mj_data, scratch, indices = load_cable_model(args.xml, args.dt)
    train_raw, train_meta = collect_pd_trajectories_unlimited(mj_model, mj_data, scratch, indices, pd_train)
    val_raw, val_meta = collect_pd_trajectories_unlimited(mj_model, mj_data, scratch, indices, pd_val)
    np.savez(out_dir / "dataset_train.npz", **train_raw)
    np.savez(out_dir / "dataset_val.npz", **val_raw)
    print(f"      train={train_raw['states'].shape}, val={val_raw['states'].shape}")

    # ---- [2/6] 构建名义模型残差数据集 ----
    print("[2/6] Building nominal residual datasets...")
    nominal = make_nominal_model(dt=args.dt)
    res_train = build_residual_dataset(train_raw, nominal, args.dt)
    res_val = build_residual_dataset(val_raw, nominal, args.dt)
    x_tr, u_tr, xp_tr, r_tr = flatten_residual_data(res_train)
    x_va, u_va, xp_va, r_va = flatten_residual_data(res_val)

    # 标准化器 (仅用训练集拟合, 验证/推理共用)
    x_normer = Normalizer.fit(x_tr)
    u_normer = Normalizer.fit(u_tr)
    r_normer = Normalizer.fit(r_tr)
    train_arrays = {
        "x": x_normer.transform(x_tr),
        "u": u_normer.transform(u_tr),
        "xp": x_normer.transform(xp_tr),
        "r": r_normer.transform(r_tr),
    }
    val_arrays = {
        "x": x_normer.transform(x_va),
        "u": u_normer.transform(u_va),
        "xp": x_normer.transform(xp_va),
        "r": r_normer.transform(r_va),
    }

    # ---- [3/6] 训练残差 DeepKoopman ----
    print("[3/6] Training residual DeepKoopman...")
    train_cfg = DeepKoopmanTrainConfig(
        latent_dim=args.latent_dim,
        hidden=tuple(args.hidden),
        activation=args.activation,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        w_residual=args.w_residual,
        w_recon=args.w_recon,
        w_linear=args.w_linear,
        w_l2=args.w_l2,
    )
    dk_model, history, train_stats = train_deepkoopman(train_arrays, val_arrays, train_cfg, device, out_dir)
    np.savetxt(
        out_dir / "training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_residual,train_recon,train_linear,val_total,val_residual,val_recon,val_linear",
        comments="",
    )
    plot_training_history(history)

    # ---- [4/6] 构建混合模型运行期封装 + 开环动力学评估出图 ----
    print("[4/6] Building hybrid runtime and evaluating open-loop dynamics...")
    # runtime 封装训练好的残差网络, 供闭环混合 MPC 做 f_nom+r_hat 线性化
    runtime = DeepKoopmanRuntime(dk_model, x_normer, u_normer, r_normer, device)

    # 开环多步预测: 名义 vs 混合 (f_nom+r_hat) vs MuJoCo, 用于动力学响应对比图
    eval_result = evaluate_rollout(dk_model, res_val, nominal, args.dt, x_normer, u_normer, r_normer, device)
    dyn_metrics = eval_result["metrics"]
    demo_idx = min(max(args.demo_traj, 0), max(args.val_traj - 1, 0))
    plot_dynamics_response(eval_result, args.dt, demo_idx)

    # ---- [5/6] MuJoCo 闭环 MPC 对比 (名义 vs 混合) ----
    print("[5/6] Running MuJoCo closed-loop MPC comparison...")
    ref = build_joint_reference(
        dt=args.dt,
        T_total=args.T_track,
        qa0=args.qa0,
        qa1=args.qa1,
        qb0=args.qb0,
        qb1=args.qb1,
        T_ramp=args.T_ramp,
    )
    mpc_cfg = LinearMpcConfig(
        horizon=args.horizon,
        Qq=args.Qq,
        Qdq=args.Qdq,
        R=args.R,
        Rd=args.Rd,
        fd_eps_x=args.fd_eps_x,
        fd_eps_u=args.fd_eps_u,
    )
    log_nom = run_mpc_on_mujoco(
        label="nominal_linear_mpc",
        nominal=nominal,
        runtime=None,
        mpc_cfg=mpc_cfg,
        xml=args.xml,
        dt=args.dt,
        ref=ref,
        seed=args.seed,
    )
    log_hyb = run_mpc_on_mujoco(
        label="hybrid_linear_mpc",
        nominal=nominal,
        runtime=runtime,
        mpc_cfg=mpc_cfg,
        xml=args.xml,
        dt=args.dt,
        ref=ref,
        seed=args.seed,
    )
    save_npz(out_dir / "mpc_nominal_log.npz", log_nom)
    save_npz(out_dir / "mpc_hybrid_log.npz", log_hyb)

    # ---- [6/6] 出图 + 汇总 JSON ----
    print("[6/6] Plotting and saving summary...")
    plot_tracking(log_nom, log_hyb)
    plot_tracking_error(log_nom, log_hyb)
    plot_tracking_rmse(log_nom, log_hyb)
    plot_torques(log_nom, log_hyb)
    plot_cable_tensions(log_nom, log_hyb)

    summary = {
        "xml": args.xml,
        "dt": args.dt,
        "limits": {
            "joint_torque_limit_enabled": False,
            "cable_tension_upper_limit_enabled": False,
            "f_preload": F_PRELOAD,
        },
        "collection": {"train": {**asdict(pd_train), "meta": train_meta}, "val": {**asdict(pd_val), "meta": val_meta}},
        "deepkoopman": {
            "config": asdict(train_cfg),
            "stats": train_stats,
            "normalization": {"x": x_normer.to_json(), "u": u_normer.to_json(), "r": r_normer.to_json()},
            "dynamics_metrics": metrics_to_json(dyn_metrics),
        },
        "mpc": {
            "config": asdict(mpc_cfg),
            "form": "condensed unconstrained linear-quadratic (closed-form QP)",
            "hybrid_model": "f_nom + residual (finite-difference linearized)",
            "nominal": tracking_metrics(log_nom),
            "hybrid": tracking_metrics(log_hyb),
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
