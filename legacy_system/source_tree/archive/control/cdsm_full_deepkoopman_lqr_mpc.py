"""
绳驱空间机械臂 (CDSM) 完整 DeepKoopman 模型 + 线性 (LQR) MPC 关节跟踪
=====================================================================

本脚本是一个 *单文件* 端到端流水线, 与残差混合方案 (见
``cdsm_deepkoopman_latent_mpc_pipeline.py``) 的核心区别在于:

  >>> 这里用 DeepKoopman 直接学习 **完整** 的绳驱空间机械臂动力学,
      而不是学"名义模型 + 残差"。<<<

整体链路 (对应用户的 4 个任务):

  [1] PD 控制器在 MuJoCo 绳驱模型上采集关节空间轨迹数据 (多正弦激励);
  [2] 训练一个 *受控* DeepKoopman 完整动力学模型:
          z_k     = [x_k ; psi(x_k)]            (状态内嵌 + 非线性升维)
          z_{k+1} = A z_k + B u_k               (潜空间里 **全局线性**)
          x_k     = C z_k,   C = [I_4 | 0]       (线性可读出, 精确恢复状态)
      由于潜空间动力学天然线性, 该模型 *本身就是* 一个全局线性预测器;
  [3] 直接在该线性潜空间上做"线性 MPC"。因为预测模型 (A,B,C) 与工作点无关,
      整个有限时域二次型最优控制就退化成 **最简单的 LQR / 线性二次跟踪**:
      Hessian 与凝聚预测矩阵都可 **离线预计算一次**, 闭环每步仅一次
      矩阵-向量运算即得最优控制序列 (无需迭代 QP, 无需逐步线性化);
  [4] 出图: 完整动力学模型 vs MuJoCo 开环响应对比、关节角跟踪、跟踪误差 &
      RMSE、关节力矩、8 根绳索张力。

为什么 DeepKoopman + LQR 是自洽的
---------------------------------
经典 EDMD / MPC 需要在每个工作点对非线性模型做有限差分线性化; 而 Koopman
观测量把非线性动力学 *提升* 到一个高维空间, 在该空间里时间推进是严格线性的
(z_{k+1}=A z_k+B u_k)。因此跟踪问题就是一个标准的线性二次调节 (LQR) /
线性 MPC, 不存在"近似线性化"的工作点依赖, 这正是 Koopman 控制的核心优势。

控制约定 (与项目其余脚本一致)
------------------------------
  * 不做关节力矩限幅;
  * 关节力矩经 ``cable_antagonistic_map`` 拮抗映射成 8 根绳张力下发,
    仅保留预紧下限 ``F_PRELOAD``, 不设张力上限 (f_max=inf)。

运行
----
    # 完整 (GPU 推荐):
    python cdsm_full_deepkoopman_lqr_mpc.py
    # 快速冒烟:
    python cdsm_full_deepkoopman_lqr_mpc.py --train_traj 8 --val_traj 3 \
        --steps 120 --epochs 15 --device cpu
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

# MuJoCo 渲染后端 (导入 mujoco 前设置)
os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")  # 非交互后端, 仅存盘
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires PyTorch in the configured project environment.") from exc

# ----------------------------------------------------------------------------
# 复用绳驱 MuJoCo 工具 (与残差方案共用同一套被控对象/采数/拮抗映射):
#   CABLE_NAMES / F_PRELOAD       : 8 根绳名称 / 预紧张力 (N)
#   PDCollectConfig               : PD 采数配置 dataclass
#   collect_pd_trajectories       : 多正弦 PD 采集 MuJoCo 轨迹
#   cable_antagonistic_map        : 关节力矩 -> 8 根绳张力 拮抗映射
#   compute_tendon_jacobian_fd    : 有限差分 tendon 雅可比 (8 x nv)
#   get_active_state/set_active_state : 读写 MuJoCo 主动关节状态 [qa,qb,dqa,dqb]
#   load_cable_model              : 载入 XML 并返回 (model,data,scratch,indices)
# ----------------------------------------------------------------------------
from cdsm_hybrid_residual_edmd import (
    CABLE_NAMES,
    F_PRELOAD,
    PDCollectConfig,
    cable_antagonistic_map,
    collect_pd_trajectories,
    compute_tendon_jacobian_fd,
    get_active_state,
    load_cable_model,
    set_active_state,
)
from cdsm_mpc_tracking_compare import build_joint_reference
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
STATE_DIM = 4
CONTROL_DIM = 2


def set_seed(seed: int) -> None:
    """统一设置 NumPy / random / PyTorch (含 CUDA) 随机种子, 保证可复现。"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(name: str) -> torch.device:
    """解析设备名: auto 时优先用 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ============================================================================
# 标准化器 (z-score): 状态/控制均做标准化, 提升潜空间数值条件
# ============================================================================
@dataclass
class Normalizer:
    """逐维 z-score 标准化: x_norm = (x-mean)/std。"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        return np.asarray(x_norm, dtype=np.float64) * self.std + self.mean

    def to_json(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ============================================================================
# 配置 dataclass
# ============================================================================
@dataclass
class DeepKoopmanConfig:
    """完整 DeepKoopman 网络结构与训练超参数。"""

    lift_dim: int            # 非线性升维观测量个数 (潜空间额外维度)
    hidden: Tuple[int, ...]  # 升维 MLP 隐层宽度
    activation: str          # 激活函数 (relu/elu/tanh)
    window: int              # 多步预测窗口长度 S 的 **上限** (curriculum 终值)
    epochs: int              # 训练轮数
    steps_per_epoch: int     # 每轮小批量更新次数
    batch_size: int          # 批大小
    lr: float                # AdamW 学习率
    grad_clip: float         # 梯度范数裁剪上限 (<=0 关闭)
    w_pred: float            # 多步状态预测损失权重
    w_linear: float          # 潜空间线性一致性损失权重
    w_l2: float              # 权重 L2 正则系数
    # --- 以下为"长程回滚稳定性"相关项 (带默认值, 兼容旧调用) ---
    window_start: int = 4    # curriculum 多步窗口起始长度 (从短到长)
    w_stab: float = 10.0     # A 的谱范数稳定性惩罚权重 (压制 ||A||_2>rho_target)
    rho_target: float = 1.0  # 谱范数目标上限 (1.0 => 非扩张 => 任意步数有界)
    weight_decay: float = 1e-5  # AdamW 权重衰减 (收缩算子, 抑制非正规增长)
    bound_lift: bool = True  # 升维末层加 tanh, 使潜变量有界, 长程递推更稳


@dataclass
class LqrMpcConfig:
    """潜空间线性 MPC (本质即 LQR / 线性二次跟踪) 的超参数。"""

    horizon: int   # 预测时域步数 N
    Qq: float      # 关节角跟踪误差权重 (标准化状态空间, 作用于 q_a,q_b)
    Qdq: float     # 关节角速度跟踪误差权重 (作用于 dq_a,dq_b)
    R: float       # 控制量幅值权重 (标准化控制空间)
    Rd: float      # 控制增量 ΔU 权重 (平滑项)


# ============================================================================
# [2] 完整 DeepKoopman 网络
#   z = [x_n ; psi(x_n)],  z+ = A z + B u_n,  x_n = C z  (C=[I,0])
# ============================================================================
class MLP(nn.Module):
    """简单全连接 MLP (末层无激活)。"""

    def __init__(self, widths: Tuple[int, ...], activation: str = "elu") -> None:
        super().__init__()
        act_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: List[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i != len(widths) - 2:
                layers.append(act_map[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullDeepKoopman(nn.Module):
    """
    受控完整 DeepKoopman 动力学模型。

    与残差版的关键差别: 这里直接学 MuJoCo 全状态的一步映射, 不依赖任何名义
    模型。潜空间被构造成 **状态内嵌** 形式 z = [x_n, psi(x_n)], 因此:
      * 读出矩阵 C = [I_4 | 0] 固定且使状态恢复精确 (无需学习解码器);
      * 时间推进 z_{k+1}=A z_k + B u_n 在潜空间里严格线性 (Koopman 思想);
      * 跟踪控制因此可直接套用 LQR / 线性 MPC, 无需逐点线性化。

    所有量都在 *标准化* 坐标下: x_n=(x-mu_x)/sd_x, u_n=(u-mu_u)/sd_u。
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        control_dim: int = CONTROL_DIM,
        lift_dim: int = 32,
        hidden: Tuple[int, ...] = (128, 128),
        activation: str = "elu",
        bound_lift: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.lift_dim = lift_dim
        self.latent_dim = state_dim + lift_dim
        self.bound_lift = bound_lift
        # 非线性观测量 psi(x_n): 升维网络
        self.lift = MLP((state_dim,) + tuple(hidden) + (lift_dim,), activation)
        # 潜空间线性动力学 (无偏置, 纯 Koopman 形式)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(control_dim, self.latent_dim, bias=False)
        self._init_linear_dynamics()

    def _init_linear_dynamics(self) -> None:
        """A 初始化为近单位 (状态块尤其接近恒等, 契合 x_{k+1}≈x_k+dt*..); B 小。"""
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.latent_dim))
            nn.init.xavier_uniform_(self.B.weight, gain=0.05)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_n -> z = [x_n, psi(x_n)]; bound_lift 时 psi 经 tanh 有界。"""
        g = self.lift(x_norm)
        if self.bound_lift:
            g = torch.tanh(g)
        return torch.cat([x_norm, g], dim=-1)

    def koopman_step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        """潜空间线性推进 z_{k+1} = A z + B u_n。"""
        return self.A(z) + self.B(u_norm)

    @staticmethod
    def state_from_latent(z: torch.Tensor) -> torch.Tensor:
        """线性读出 x_n = C z, C=[I_4|0] (取潜变量前 4 维)。"""
        return z[..., :STATE_DIM]


# ============================================================================
# 训练数据: 从轨迹里切出长度 (window+1) 的连续片段
# ============================================================================
def build_windows(
    states: np.ndarray, inputs: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 (traj, T+1, 4) / (traj, T, 2) 轨迹切成多步预测窗口。

    返回
    ----
    Xw : (M, window+1, 4)  连续状态片段 (x_0..x_window)
    Uw : (M, window,   2)  对应控制片段 (u_0..u_{window-1})
    """
    n_traj, n_times, _ = states.shape
    n_step = inputs.shape[1]
    if n_step != n_times - 1:
        raise ValueError(f"inputs steps {n_step} != states-1 {n_times - 1}")
    if window > n_step:
        raise ValueError(f"window {window} larger than trajectory steps {n_step}")
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    for i in range(n_traj):
        for k in range(n_step - window + 1):
            xs.append(states[i, k : k + window + 1])
            us.append(inputs[i, k : k + window])
    Xw = np.asarray(xs, dtype=np.float64)
    Uw = np.asarray(us, dtype=np.float64)
    return Xw, Uw


def sample_window_batch(
    Xw_norm: np.ndarray, Uw_norm: np.ndarray, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """随机采一批窗口样本 (标准化后), 搬到目标设备。"""
    n = Xw_norm.shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    x = torch.from_numpy(Xw_norm[idx].astype(np.float32)).to(device)
    u = torch.from_numpy(Uw_norm[idx].astype(np.float32)).to(device)
    return x, u


def spectral_norm_power(W: torch.Tensor, n_iters: int = 12) -> torch.Tensor:
    """
    可微地估计方阵 W 的谱范数 sigma_max(W) (= sqrt(lambda_max(W^T W))), 用幂迭代。

    谱范数是长程稳定性的关键量: 若 ||W||_2 <= 1, 则 ||W^k||_2 <= 1 对任意 k 成立,
    因此潜空间线性 rollout 不会出现 (非正规) 瞬态放大与发散。
    """
    n = W.shape[1]
    v = torch.randn(n, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + 1e-12)
    u = W @ v
    for _ in range(n_iters):
        u = W @ v
        u = u / (u.norm() + 1e-12)
        v = W.t() @ u
        v = v / (v.norm() + 1e-12)
    return torch.dot(u, W @ v)


def compute_window_losses(
    model: FullDeepKoopman,
    x_win: torch.Tensor,
    u_win: torch.Tensor,
    cfg: DeepKoopmanConfig,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    """
    多步窗口损失 (标准化空间), 在前 ``horizon`` 步上滚动:
      L_pred   = Σ_k ||C(A^k...) - x_k_true||^2     多步状态预测 (rollout 关键)
      L_linear = Σ_k ||z_pred_k - encode(x_k_true)||^2  潜空间线性一致性
      L_stab   = w_stab * relu(||A||_2 - rho_target)^2  谱范数稳定性惩罚
      L_l2     = 权重正则
    其中 z_pred 从 x_0 出发, 用记录的控制 u 在潜空间线性递推 horizon 步。
    L_stab 同时压制 (a) 不稳定模态 |lambda|>1 与 (b) 非正规瞬态放大,
    这正是诊断中导致"一步准、多步崩"的两个根因。
    """
    S = int(horizon)
    xw = x_win[:, : S + 1]
    uw = u_win[:, :S]
    B = xw.shape[0]
    # 真值各步潜变量 (teacher) 与状态
    z_true = model.encode(xw.reshape(B * (S + 1), -1)).reshape(B, S + 1, -1)
    # 从 x_0 出发的潜空间线性 rollout
    z = z_true[:, 0]
    pred_loss = xw.new_zeros(())
    lin_loss = xw.new_zeros(())
    for k in range(S):
        z = model.koopman_step(z, uw[:, k])
        x_pred = model.state_from_latent(z)
        pred_loss = pred_loss + torch.mean((x_pred - xw[:, k + 1]) ** 2)
        lin_loss = lin_loss + torch.mean((z - z_true[:, k + 1]) ** 2)
    pred_loss = pred_loss / S
    lin_loss = lin_loss / S

    # 谱范数稳定性惩罚 (仅当 ||A||_2 超过目标时生效)
    stab = xw.new_zeros(())
    if cfg.w_stab > 0:
        sigma = spectral_norm_power(model.A.weight)
        over = torch.clamp(sigma - cfg.rho_target, min=0.0)
        stab = cfg.w_stab * over * over

    l2 = xw.new_zeros(())
    if cfg.w_l2 > 0:
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                l2 = l2 + torch.sum(p * p)
        l2 = cfg.w_l2 * l2
    total = cfg.w_pred * pred_loss + cfg.w_linear * lin_loss + stab + l2
    return {
        "total": total,
        "pred": pred_loss,
        "linear": lin_loss,
        "stab": stab.detach() if isinstance(stab, torch.Tensor) else stab,
        "l2": l2.detach(),
    }


def curriculum_horizon(epoch: int, cfg: DeepKoopmanConfig) -> int:
    """
    课程式多步窗口: 训练前 60% 轮里把滚动步数从 window_start 线性增大到 window,
    之后保持 window。短窗口先学好一步/几步动力学, 再逐步逼近长程, 提升 rollout 稳定性。
    """
    w0 = max(1, min(cfg.window_start, cfg.window))
    ramp_epochs = max(1, int(0.6 * cfg.epochs))
    frac = min(1.0, (epoch - 1) / max(1, ramp_epochs - 1))
    return int(round(w0 + frac * (cfg.window - w0)))


@torch.no_grad()
def evaluate_window_loss(
    model: FullDeepKoopman,
    Xw_norm: np.ndarray,
    Uw_norm: np.ndarray,
    cfg: DeepKoopmanConfig,
    device: torch.device,
    n_batches: int = 5,
) -> Dict[str, float]:
    """在验证窗口上估计平均损失 (用完整 window 步, 反映长程一致性)。"""
    model.eval()
    acc = {"total": 0.0, "pred": 0.0, "linear": 0.0}
    for _ in range(n_batches):
        x, u = sample_window_batch(Xw_norm, Uw_norm, cfg.batch_size, device)
        losses = compute_window_losses(model, x, u, cfg, cfg.window)
        for key in acc:
            acc[key] += float(losses[key].item())
    for key in acc:
        acc[key] /= max(n_batches, 1)
    return acc


def train_deepkoopman(
    train_win: Tuple[np.ndarray, np.ndarray],
    val_win: Tuple[np.ndarray, np.ndarray],
    cfg: DeepKoopmanConfig,
    device: torch.device,
    out_dir: Path,
) -> Tuple[FullDeepKoopman, List[List[float]], Dict[str, float]]:
    """训练完整 DeepKoopman, 按验证集总损失保存最优权重。"""
    Xw_tr, Uw_tr = train_win
    Xw_va, Uw_va = val_win
    model = FullDeepKoopman(
        state_dim=STATE_DIM,
        control_dim=CONTROL_DIM,
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        activation=cfg.activation,
        bound_lift=cfg.bound_lift,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_full = float("inf")
    best_epoch = 0
    history: List[List[float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        horizon = curriculum_horizon(epoch, cfg)
        acc = {"total": 0.0, "pred": 0.0, "linear": 0.0, "stab": 0.0}
        for _ in range(cfg.steps_per_epoch):
            x, u = sample_window_batch(Xw_tr, Uw_tr, cfg.batch_size, device)
            losses = compute_window_losses(model, x, u, cfg, horizon)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for key in acc:
                acc[key] += float(losses[key].detach().item())
        for key in acc:
            acc[key] /= max(cfg.steps_per_epoch, 1)

        val_full = evaluate_window_loss(model, Xw_va, Uw_va, cfg, device)
        with torch.no_grad():
            sigma_A = float(spectral_norm_power(model.A.weight).item())
        history.append(
            [
                float(epoch),
                acc["total"], acc["pred"], acc["linear"],
                val_full["total"], val_full["pred"], val_full["linear"],
            ]
        )
        print(
            f"[dk] epoch {epoch:03d}/{cfg.epochs:03d} H={horizon:02d} "
            f"train={acc['total']:.3e} (pred={acc['pred']:.3e} lin={acc['linear']:.3e} "
            f"stab={acc['stab']:.2e}) valFull={val_full['total']:.3e} ||A||2={sigma_A:.4f}",
            flush=True,
        )
        if val_full["total"] < best_val_full:
            best_val_full = val_full["total"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val_full": best_val_full,
                    "best_selection": "fixed full-window validation loss",
                    "epoch": epoch,
                },
                out_dir / "best_full_deepkoopman.pt",
            )

    ckpt = torch.load(out_dir / "best_full_deepkoopman.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, history, {
        "best_val_full": float(best_val_full),
        "best_epoch": float(best_epoch),
        "best_selection": "fixed full-window validation loss",
    }


# ============================================================================
# DeepKoopman 运行期封装: 提供 numpy 接口 (编码 + 取线性系统矩阵)
# ============================================================================
class KoopmanRuntime:
    """
    训练好的完整 DeepKoopman 在推理期的轻量封装。

    职责:
      * encode(x_phys) -> z (numpy): 物理状态 -> 标准化 -> 潜变量;
      * get_linear_system() -> (A_d, B_d, C): 取出潜空间线性系统矩阵 (numpy),
        其中 z_{k+1}=A_d z + B_d u_n, x_n = C z;
      * predict_state(z) -> x_phys: 潜变量 -> 标准化状态 -> 反标准化。
    标准化约定与训练完全一致, 控制是 *标准化* 控制 u_n。
    """

    def __init__(
        self,
        model: FullDeepKoopman,
        x_normer: Normalizer,
        u_normer: Normalizer,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.x_normer = x_normer
        self.u_normer = u_normer
        self.device = device
        # 取出线性系统矩阵 (列向量约定: z+ = A_d @ z + B_d @ u_n)
        self.A_d = model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B_d = model.B.weight.detach().cpu().numpy().astype(np.float64)
        D = model.latent_dim
        C = np.zeros((STATE_DIM, D), dtype=np.float64)
        C[:STATE_DIM, :STATE_DIM] = np.eye(STATE_DIM)
        self.C = C

    @torch.no_grad()
    def encode(self, x_phys: np.ndarray) -> np.ndarray:
        """物理状态 (4,) -> 潜变量 z (D,)。"""
        x_n = self.x_normer.transform(np.atleast_2d(x_phys))
        z = self.model.encode(torch.from_numpy(x_n.astype(np.float32)).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def predict_state_norm(self, z: np.ndarray) -> np.ndarray:
        """潜变量 -> 标准化状态 x_n。"""
        return self.C @ np.asarray(z, dtype=np.float64).reshape(-1)

    def predict_state(self, z: np.ndarray) -> np.ndarray:
        """潜变量 -> 物理状态 x。"""
        return self.x_normer.inverse(self.predict_state_norm(z))

    def rollout_state(self, x0_phys: np.ndarray, u_seq_phys: np.ndarray) -> np.ndarray:
        """
        用完整 DeepKoopman 做开环多步状态预测 (物理量)。

        z_0 = encode(x_0); z_{k+1}=A_d z_k + B_d u_n_k; x_{k+1}=denorm(C z_{k+1})。
        返回 (T+1, 4) 预测状态轨迹 (含初值)。
        """
        u_seq_phys = np.asarray(u_seq_phys, dtype=np.float64)
        T = u_seq_phys.shape[0]
        traj = np.zeros((T + 1, STATE_DIM), dtype=np.float64)
        traj[0] = np.asarray(x0_phys, dtype=np.float64).reshape(-1)
        z = self.encode(traj[0])
        u_n = self.u_normer.transform(u_seq_phys)
        for k in range(T):
            z = self.A_d @ z + self.B_d @ u_n[k]
            traj[k + 1] = self.predict_state(z)
        return traj


# ============================================================================
# [3] 潜空间线性 MPC == LQR / 线性二次跟踪 (凝聚式无约束闭式解)
# ============================================================================
class LatentLqrMpc:
    """
    在 DeepKoopman 潜空间线性系统 (A_d,B_d,C) 上的有限时域线性二次跟踪器。

    因为 (A_d,B_d,C) 是常量 (与状态/工作点无关), 凝聚预测矩阵与 Hessian 都可
    **离线预计算一次**。每个控制周期:
        1) z0 = encode(x_meas);
        2) 预测 (标准化) 状态序列 Y = Phi z0 + Gamma U;
        3) 代价 J = Σ (y_k-r_k)ᵀ Q (y_k-r_k) + u_kᵀ R u_k + Δuᵀ Rd Δu;
        4) 无约束 => 闭式最优 U* = Hinv (-(Gammaᵀ Q (Phi z0 - Rref)
                                          - Dᵀ Rd E u_prev)) , 取首步执行。
    这正是标准 LQR / 线性 MPC, 不含任何迭代或逐点线性化。

    所有内部量都在标准化坐标 (状态与控制), 参考也先标准化。
    """

    def __init__(self, runtime: KoopmanRuntime, cfg: LqrMpcConfig) -> None:
        self.rt = runtime
        self.cfg = cfg
        N = cfg.horizon
        A_d, B_d, C = runtime.A_d, runtime.B_d, runtime.C
        D = A_d.shape[0]
        nu = B_d.shape[1]
        ny = C.shape[0]

        # 凝聚: Y = Phi z0 + Gamma U, 其中输出 y_k = C z_k (k=1..N)
        Phi = np.zeros((N * ny, D), dtype=np.float64)
        Gamma = np.zeros((N * ny, N * nu), dtype=np.float64)
        Apow = [np.eye(D)]
        for _ in range(N):
            Apow.append(A_d @ Apow[-1])  # Apow[k] = A_d^k
        for i in range(N):  # 预测步 i -> 输出 y_{i+1}
            Phi[i * ny : (i + 1) * ny, :] = C @ Apow[i + 1]
            for j in range(i + 1):  # 控制 u_j (j=0..i)
                blk = C @ Apow[i - j] @ B_d
                Gamma[i * ny : (i + 1) * ny, j * nu : (j + 1) * nu] = blk

        # 权重 (标准化状态空间). 角/角速度分别加权
        q_diag = np.tile(
            np.array([cfg.Qq, cfg.Qq, cfg.Qdq, cfg.Qdq], dtype=np.float64), N
        )
        Qbar = np.diag(q_diag)
        Rbar = np.eye(N * nu) * cfg.R
        Rdbar = np.eye(N * nu) * cfg.Rd

        # 控制增量算子 ΔU = D U - E u_prev, 首步 ΔU_0 = u_0 - u_prev
        Dmat = np.zeros((N * nu, N * nu), dtype=np.float64)
        for k in range(N):
            Dmat[k * nu : (k + 1) * nu, k * nu : (k + 1) * nu] = np.eye(nu)
            if k > 0:
                Dmat[k * nu : (k + 1) * nu, (k - 1) * nu : k * nu] = -np.eye(nu)
        Emat = np.zeros((N * nu, nu), dtype=np.float64)
        Emat[:nu, :] = np.eye(nu)

        # Hessian (常量) 与其逆 (离线预计算一次)
        H = Gamma.T @ Qbar @ Gamma + Rbar + Dmat.T @ Rdbar @ Dmat + 1e-9 * np.eye(N * nu)
        self.Phi = Phi
        self.Gamma = Gamma
        self.Qbar = Qbar
        self.Rdbar = Rdbar
        self.Dmat = Dmat
        self.Emat = Emat
        self.Hinv = np.linalg.inv(H)
        self.N = N
        self.ny = ny
        self.nu = nu

    def solve(
        self, x_meas: np.ndarray, ref_norm: np.ndarray, u_prev_norm: np.ndarray
    ) -> np.ndarray:
        """
        求解一次 LQ-MPC, 返回整段最优 *标准化* 控制序列 (N, nu); 取首步执行。

        参数
        ----
        x_meas    : 当前物理量测状态 (4,)
        ref_norm  : (N, 4) 预测时域内的标准化参考 [q_n, dq_n]
        u_prev_norm : 上一步标准化控制 (nu,)
        """
        z0 = self.rt.encode(x_meas)
        y0 = self.Phi @ z0                 # 零控制预测 (标准化输出, 已含 z0 自由响应)
        r = np.asarray(ref_norm, dtype=np.float64).reshape(-1)
        u_prev = np.asarray(u_prev_norm, dtype=np.float64).reshape(self.nu)
        b = self.Gamma.T @ self.Qbar @ (y0 - r) - self.Dmat.T @ self.Rdbar @ self.Emat @ u_prev
        U = -self.Hinv @ b
        return U.reshape(self.N, self.nu)


def pad_ref_norm(
    ref: Dict[str, np.ndarray], x_normer: Normalizer, k: int, horizon: int
) -> np.ndarray:
    """
    取从第 k+1 步起、长度 horizon 的标准化参考 [q_n, dq_n] (对应预测状态 x_1..x_N)。
    越界部分用末值填充 (zero-order hold)。
    """
    q_ref = ref["q_ref"]
    dq_ref = ref["dq_ref"]
    M = q_ref.shape[0]
    out = np.zeros((horizon, STATE_DIM), dtype=np.float64)
    for i in range(horizon):
        idx = min(k + 1 + i, M - 1)
        x_ref = np.array(
            [q_ref[idx, 0], q_ref[idx, 1], dq_ref[idx, 0], dq_ref[idx, 1]],
            dtype=np.float64,
        )
        out[i] = x_normer.transform(x_ref.reshape(1, -1)).reshape(-1)
    return out


# ============================================================================
# 闭环: 在 MuJoCo 被控对象上跑潜空间 LQR-MPC
# ============================================================================
def run_mpc_on_mujoco(
    *,
    runtime: KoopmanRuntime,
    controller: LatentLqrMpc,
    x_normer: Normalizer,
    u_normer: Normalizer,
    xml: str,
    dt: float,
    ref: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    在 MuJoCo 真值对象上闭环运行潜空间 LQR-MPC 并记录日志。

    每步: 量测状态 -> encode -> 解 LQ-MPC 取首步标准化控制 -> 反标准化成关节
    力矩 -> 拮抗映射成 8 根绳张力 (无上限) -> 推进 MuJoCo 一步。
    记录 t / x / u(关节力矩) / q_ref / dq_ref / 求解耗时 / 8 根绳张力。
    """
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for closed-loop control.") from exc

    model, data, scratch, indices = load_cable_model(xml, dt)
    set_active_state(model, data, indices, ref["q_ref"][0], ref["dq_ref"][0])
    mujoco.mj_forward(model, data)

    n_step = len(ref["t"]) - 1
    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])
    u_prev_phys = np.zeros(CONTROL_DIM, dtype=np.float64)

    rec: Dict[str, List[np.ndarray]] = {
        "t": [], "x": [], "u": [], "q_ref": [], "dq_ref": [],
        "solve_ms": [], "cable_tensions": [],
    }
    print(f"[mpc] start steps={n_step} horizon={controller.cfg.horizon}", flush=True)
    t0 = time.perf_counter()
    report_every = max(1, n_step // 5)
    for k in range(n_step):
        x_meas = get_active_state(data, indices)
        ref_norm = pad_ref_norm(ref, x_normer, k, controller.cfg.horizon)
        u_prev_norm = u_normer.transform(u_prev_phys.reshape(1, -1)).reshape(-1)

        tic = time.perf_counter()
        U = controller.solve(x_meas, ref_norm, u_prev_norm)
        solve_ms = 1e3 * (time.perf_counter() - tic)
        u_cmd = u_normer.inverse(U[0].reshape(1, -1)).reshape(-1)  # 反标准化成关节力矩

        # 关节力矩 -> 8 根绳张力 (无上限) -> 推进 MuJoCo
        J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
        F_cable = cable_antagonistic_map(
            float(u_cmd[0]), float(u_cmd[1]), J,
            dof_j1, dof_j2, dof_j3, dof_j4,
            f_pre=F_PRELOAD, f_max=float("inf"),
        )
        data.ctrl[indices["actuator_ids"]] = F_cable
        mujoco.mj_step(model, data)

        rec["t"].append(float(k * dt))
        rec["x"].append(x_meas.copy())
        rec["u"].append(u_cmd.copy())
        rec["q_ref"].append(ref["q_ref"][k].copy())
        rec["dq_ref"].append(ref["dq_ref"][k].copy())
        rec["solve_ms"].append(float(solve_ms))
        rec["cable_tensions"].append(F_cable.copy())
        u_prev_phys = u_cmd

        if (k + 1) % report_every == 0 or k == 0 or k + 1 == n_step:
            err = float(np.linalg.norm(x_meas[:2] - ref["q_ref"][k]))
            print(f"[mpc] step {k + 1}/{n_step} solve={solve_ms:.2f}ms |e_q|={err:.4g}")

    print(f"[mpc] done elapsed={time.perf_counter() - t0:.2f}s", flush=True)
    return {key: np.asarray(value) for key, value in rec.items()}


# ============================================================================
# 开环动力学评估 (完整 DeepKoopman vs MuJoCo)
# ============================================================================
def evaluate_open_loop(
    runtime: KoopmanRuntime, val_raw: Dict[str, np.ndarray]
) -> Dict[str, object]:
    """
    在验证集上做开环多步预测: 完整 DeepKoopman vs MuJoCo 真值。

    每条轨迹用记录的 PD 控制 u 从真值初值递推, 误差累积反映模型多步保真度。
    """
    states = val_raw["states"]
    inputs = val_raw["inputs"]
    n_traj = states.shape[0]
    pred = np.zeros_like(states)
    for i in range(n_traj):
        pred[i] = runtime.rollout_state(states[i, 0], inputs[i])
    err = pred - states
    rmse_by_state = np.sqrt(np.mean(err * err, axis=(0, 1)))
    step_rmse = np.sqrt(np.mean(err * err, axis=(0, 2)))
    return {
        "pred": pred,
        "states_true": states,
        "rmse_by_state": rmse_by_state,
        "total_rmse": float(np.sqrt(np.mean(err * err))),
        "step_rmse": step_rmse,
    }


# ============================================================================
# 指标
# ============================================================================
def tracking_metrics(log: Dict[str, np.ndarray]) -> Dict[str, object]:
    """从闭环日志计算关节角/角速度 RMSE、峰值误差、求解耗时、峰值力矩/张力。"""
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


# ============================================================================
# [4] 出图
# ============================================================================
def plot_training_history(history: List[List[float]]) -> None:
    """训练曲线: 训练/验证总损失随 epoch (对数纵轴)。"""
    arr = np.asarray(history, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(arr[:, 0], arr[:, 1], label="train total")
    ax.semilogy(arr[:, 0], arr[:, 4], label="val full-window")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle("Full DeepKoopman training")
    fig.tight_layout()
    save_figure("training_history")
    plt.close(fig)


def plot_dynamics_response(eval_result: Dict[str, object], dt: float, traj_idx: int) -> None:
    """
    动力学响应对比图: 某条验证轨迹上, MuJoCo 真值 vs 完整 DeepKoopman 模型的
    开环多步预测, 4 个状态分量各一子图。
    """
    true = eval_result["states_true"][traj_idx]
    pred = eval_result["pred"][traj_idx]
    err = pred - true
    rmse_state = np.sqrt(np.mean(err * err, axis=0))
    t = np.arange(true.shape[0]) * dt
    units = ["rad", "rad", "rad/s", "rad/s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for i, label in enumerate(STATE_LABELS):
        ax = axes[i // 2, i % 2]
        ax.plot(t, true[:, i], "k-", lw=1.8, label="MuJoCo")
        ax.plot(t, pred[:, i], "--", lw=1.4, color="C1", label="Full DeepKoopman")
        ax.set_title(f"{label}  (RMSE={rmse_state[i]:.3g} {units[i]})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{label} ({units[i]})")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Open-loop dynamic response: full DeepKoopman vs MuJoCo")
    fig.tight_layout()
    save_figure("dynamic_response_compare")
    plt.close(fig)


def plot_rollout_rmse_growth(eval_result: Dict[str, object], dt: float) -> None:
    """开环 rollout RMSE 随时间增长曲线 (验证集平均)。"""
    step_rmse = eval_result["step_rmse"]
    t = np.arange(step_rmse.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, step_rmse, lw=1.8, color="C3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State RMSE")
    ax.set_title("Full DeepKoopman open-loop rollout RMSE vs MuJoCo")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure("rollout_rmse_growth")
    plt.close(fig)


def plot_joint_tracking(log: Dict[str, np.ndarray]) -> None:
    """关节角跟踪图: 参考 vs 闭环 (两关节各一子图)。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, name in enumerate(["q_a", "q_b"]):
        axes[j].plot(log["t"], log["q_ref"][:, j], "k--", lw=1.6, label="reference")
        axes[j].plot(log["t"], log["x"][:, j], lw=1.5, color="C0", label="DeepKoopman LQR-MPC")
        axes[j].set_ylabel(f"{name} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint-angle tracking (closed-loop on MuJoCo)")
    fig.tight_layout()
    save_figure("joint_tracking")
    plt.close(fig)


def plot_tracking_error(log: Dict[str, np.ndarray]) -> None:
    """关节角跟踪误差图 + 每关节 RMSE 标注。"""
    e_q = log["x"][:, :2] - log["q_ref"]
    rmse = np.sqrt(np.mean(e_q * e_q, axis=0))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, name in enumerate(["q_a", "q_b"]):
        axes[j].plot(log["t"], e_q[:, j], lw=1.4, color="C3",
                     label=f"e_{name}  (RMSE={rmse[j]:.4g} rad)")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"e_{name} (rad)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint-angle tracking error")
    fig.tight_layout()
    save_figure("tracking_error")
    plt.close(fig)


def plot_tracking_rmse(log: Dict[str, np.ndarray]) -> None:
    """跟踪 RMSE 柱状图 (按关节 + 总体)。"""
    e_q = log["x"][:, :2] - log["q_ref"]
    rmse_joint = np.sqrt(np.mean(e_q * e_q, axis=0))
    rmse_total = float(np.sqrt(np.mean(e_q * e_q)))
    labels = ["q_a", "q_b", "overall"]
    vals = [rmse_joint[0], rmse_joint[1], rmse_total]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=["C0", "C1", "C2"], width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3g}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (rad)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Joint-angle tracking RMSE")
    fig.tight_layout()
    save_figure("tracking_rmse")
    plt.close(fig)


def plot_torques(log: Dict[str, np.ndarray]) -> None:
    """关节力矩图: MPC 输出的期望关节力矩 tau_a/tau_b (无限幅)。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, name in enumerate(["tau_a", "tau_b"]):
        axes[j].plot(log["t"], log["u"][:, j], lw=1.4, color="C0")
        axes[j].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axes[j].set_ylabel(f"{name} (Nm)")
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint torque commands (no torque clipping)")
    fig.tight_layout()
    save_figure("joint_torques")
    plt.close(fig)


def plot_cable_tensions(log: Dict[str, np.ndarray]) -> None:
    """MuJoCo 上全部 8 根绳张力图, 标注预紧线。"""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(CABLE_NAMES):
        ax.plot(log["t"], log["cable_tensions"][:, i], lw=1.2, color=colors[i], label=name)
    ax.axhline(F_PRELOAD, color="k", lw=0.9, alpha=0.5, label=f"F_pre={F_PRELOAD:.0f} N")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tension (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=5)
    fig.suptitle("MuJoCo cable tensions (no upper tension clipping)")
    fig.tight_layout()
    save_figure("cable_tensions")
    plt.close(fig)


# ============================================================================
# 命令行参数
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数 (默认值面向 GPU 完整跑; 冒烟请调小数据/训练规模)。"""
    p = argparse.ArgumentParser(
        description="CDSM full DeepKoopman model + linear (LQR) MPC joint tracking."
    )
    # --- 全局 ---
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")

    # --- 数据采集 (PD 多正弦) ---
    p.add_argument("--train_traj", type=int, default=120)
    p.add_argument("--val_traj", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--q_init_range", type=float, default=1.0)
    p.add_argument("--dq_init_range", type=float, default=0.8)
    p.add_argument("--amp_min", type=float, default=-1.0)
    p.add_argument("--amp_max", type=float, default=1.0)
    p.add_argument("--omega_min", type=float, default=0.3)
    p.add_argument("--omega_max", type=float, default=1.5)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)
    p.add_argument("--tau_max", type=float, default=float("inf"))

    # --- 完整 DeepKoopman 网络与训练 ---
    p.add_argument("--lift_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    p.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    p.add_argument("--window", type=int, default=40, help="多步窗口上限 (curriculum 终值)")
    p.add_argument("--window_start", type=int, default=4, help="curriculum 多步窗口起始长度")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--steps_per_epoch", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--w_pred", type=float, default=1.0)
    p.add_argument("--w_linear", type=float, default=0.5)
    p.add_argument("--w_l2", type=float, default=1e-8)
    # 长程稳定性: 谱范数惩罚 + 权重衰减 + 有界升维
    p.add_argument("--w_stab", type=float, default=10.0, help="||A||_2>rho_target 的惩罚权重")
    p.add_argument("--rho_target", type=float, default=1.0, help="谱范数目标上限")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--bound_lift", type=int, default=1, help="1=升维末层加 tanh 有界, 0=不加")

    # --- 跟踪参考 与 LQR-MPC ---
    p.add_argument("--T_track", type=float, default=20.0)
    p.add_argument("--T_ramp", type=float, default=20.0)
    p.add_argument("--qa0", type=float, default=-0.8)
    p.add_argument("--qa1", type=float, default=0.8)
    p.add_argument("--qb0", type=float, default=0.6)
    p.add_argument("--qb1", type=float, default=-0.6)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--Qq", type=float, default=100.0)
    p.add_argument("--Qdq", type=float, default=1.0)
    p.add_argument("--R", type=float, default=1e-3)
    p.add_argument("--Rd", type=float, default=1e-2)
    p.add_argument("--demo_traj", type=int, default=0)
    return p


def main() -> None:
    """端到端编排: 采数 -> 训练完整 DeepKoopman -> 开环评估出图 -> 闭环 LQR-MPC -> 汇总。"""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== CDSM full DeepKoopman + linear (LQR) MPC ===")
    print(f"device={device}, output={out_dir}")
    print("limits: joint torque clipping disabled; cable upper tension clipping disabled")

    # ---- PD 采数配置 (训练 / 验证) ----
    pd_train = PDCollectConfig(
        traj_count=args.train_traj, steps=args.steps, dt=args.dt, seed=args.seed,
        q_init_range=args.q_init_range, dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max), omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b), kd=(args.kd_a, args.kd_b), tau_max=args.tau_max,
    )
    pd_val = PDCollectConfig(
        traj_count=args.val_traj, steps=args.steps, dt=args.dt, seed=args.seed + 1000,
        q_init_range=args.q_init_range, dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max), omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b), kd=(args.kd_a, args.kd_b), tau_max=args.tau_max,
    )

    # ---- [1/6] 采集 PD 轨迹数据 ----
    print("[1/6] Collecting PD MuJoCo trajectories...")
    mj_model, mj_data, scratch, indices = load_cable_model(args.xml, args.dt)
    train_raw, train_meta = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_train)
    val_raw, val_meta = collect_pd_trajectories(mj_model, mj_data, scratch, indices, pd_val)
    np.savez(out_dir / "dataset_train.npz", **train_raw)
    np.savez(out_dir / "dataset_val.npz", **val_raw)
    print(f"      train={train_raw['states'].shape}, val={val_raw['states'].shape}")

    # ---- [2/6] 标准化 + 构建多步窗口数据集 ----
    print("[2/6] Building normalized multi-step windows...")
    x_all = train_raw["states"][:, :-1, :].reshape(-1, STATE_DIM)
    u_all = train_raw["inputs"].reshape(-1, CONTROL_DIM)
    x_normer = Normalizer.fit(x_all)
    u_normer = Normalizer.fit(u_all)

    Xw_tr, Uw_tr = build_windows(train_raw["states"], train_raw["inputs"], args.window)
    Xw_va, Uw_va = build_windows(val_raw["states"], val_raw["inputs"], args.window)
    # 标准化窗口 (沿最后一维)
    Xw_tr_n = (Xw_tr - x_normer.mean) / x_normer.std
    Xw_va_n = (Xw_va - x_normer.mean) / x_normer.std
    Uw_tr_n = (Uw_tr - u_normer.mean) / u_normer.std
    Uw_va_n = (Uw_va - u_normer.mean) / u_normer.std
    print(f"      windows train={Xw_tr_n.shape}, val={Xw_va_n.shape}")

    # ---- [3/6] 训练完整 DeepKoopman ----
    print("[3/6] Training full DeepKoopman...")
    dk_cfg = DeepKoopmanConfig(
        lift_dim=args.lift_dim, hidden=tuple(args.hidden), activation=args.activation,
        window=args.window, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, lr=args.lr, grad_clip=args.grad_clip,
        w_pred=args.w_pred, w_linear=args.w_linear, w_l2=args.w_l2,
        window_start=args.window_start, w_stab=args.w_stab, rho_target=args.rho_target,
        weight_decay=args.weight_decay, bound_lift=bool(args.bound_lift),
    )
    dk_model, history, train_stats = train_deepkoopman(
        (Xw_tr_n, Uw_tr_n), (Xw_va_n, Uw_va_n), dk_cfg, device, out_dir
    )
    np.savetxt(
        out_dir / "training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_pred,train_linear,val_full_total,val_full_pred,val_full_linear",
        comments="",
    )
    plot_training_history(history)

    # ---- [4/6] 开环动力学评估 (完整 DeepKoopman vs MuJoCo) ----
    print("[4/6] Evaluating open-loop full-model dynamics...")
    runtime = KoopmanRuntime(dk_model, x_normer, u_normer, device)
    eval_result = evaluate_open_loop(runtime, val_raw)
    demo_idx = min(max(args.demo_traj, 0), max(args.val_traj - 1, 0))
    plot_dynamics_response(eval_result, args.dt, demo_idx)
    plot_rollout_rmse_growth(eval_result, args.dt)
    print(f"      open-loop total RMSE = {eval_result['total_rmse']:.6g}")

    # ---- [5/6] MuJoCo 闭环 LQR-MPC 跟踪 ----
    print("[5/6] Running MuJoCo closed-loop LQR-MPC tracking...")
    ref = build_joint_reference(
        dt=args.dt, T_total=args.T_track,
        qa0=args.qa0, qa1=args.qa1, qb0=args.qb0, qb1=args.qb1, T_ramp=args.T_ramp,
    )
    mpc_cfg = LqrMpcConfig(
        horizon=args.horizon, Qq=args.Qq, Qdq=args.Qdq, R=args.R, Rd=args.Rd
    )
    controller = LatentLqrMpc(runtime, mpc_cfg)
    log = run_mpc_on_mujoco(
        runtime=runtime, controller=controller, x_normer=x_normer, u_normer=u_normer,
        xml=args.xml, dt=args.dt, ref=ref,
    )
    np.savez(out_dir / "mpc_log.npz", **log)

    # ---- [6/6] 出图 + 汇总 JSON ----
    print("[6/6] Plotting and saving summary...")
    plot_joint_tracking(log)
    plot_tracking_error(log)
    plot_tracking_rmse(log)
    plot_torques(log)
    plot_cable_tensions(log)

    track_metrics = tracking_metrics(log)
    summary = {
        "xml": args.xml,
        "dt": args.dt,
        "model": "full DeepKoopman (state-inclusive lift, globally-linear latent dynamics)",
        "limits": {
            "joint_torque_limit_enabled": False,
            "cable_tension_upper_limit_enabled": False,
            "f_preload": F_PRELOAD,
        },
        "collection": {
            "train": {**asdict(pd_train), "meta": train_meta},
            "val": {**asdict(pd_val), "meta": val_meta},
        },
        "deepkoopman": {
            "config": asdict(dk_cfg),
            "latent_dim": dk_model.latent_dim,
            "stats": train_stats,
            "normalization": {"x": x_normer.to_json(), "u": u_normer.to_json()},
            "open_loop": {
                "total_rmse": eval_result["total_rmse"],
                "rmse_by_state": eval_result["rmse_by_state"].tolist(),
            },
        },
        "mpc": {
            "config": asdict(mpc_cfg),
            "form": "latent-space condensed unconstrained linear-quadratic (LQR), precomputed Hessian",
            "tracking": track_metrics,
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
