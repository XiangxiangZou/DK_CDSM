"""
deep_koopman_cdsm.py
====================
基于 Lusch 等 (2018) 《Deep learning for universal linear embeddings of nonlinear
dynamics》的 Deep Koopman 算子方法, 在 MuJoCo 仿真的 **绳驱空间机械臂 (CDSM)**
上做数据驱动的全局线性嵌入学习.

────────────────────────────────────────────────────────────────────────────
方法核心 (Lusch 2018)
────────────────────────────────────────────────────────────────────────────

目标: 寻找一组内在坐标 y = φ(x), 使非线性动力学在 y 空间是线性的:
                y_{k+1} = K(λ(y_k)) · y_k
其中 K 是块对角 Jordan 矩阵, 由辅助网络 Λ: y -> λ 局部确定特征值 λ = μ ± iω.

三个网络:
  (1) 编码器 (Encoder)  φ : x -> y          (MLP, 输入 R^n,  输出 R^p)
  (2) 解码器 (Decoder)  φ^{-1} : y -> x     (MLP, 输入 R^p,  输出 R^n)
  (3) 辅助网络 (Aux Λ): y -> (μ, ω)          (MLP, 每个特征块独立)

潜在空间结构: p = 2 * num_complex_pairs + num_real
  - 每对复共轭特征块占用 2 维 (yj, yj+1), 用 2x2 Jordan 块演化:
        B(μ, ω) = exp(μΔt) · [[cos(ωΔt), -sin(ωΔt)],
                              [sin(ωΔt),  cos(ωΔt)]]
    特征值由辅助网络以输入半径 r² = yj² + yj+1² 输出 (μ, ω) (圆对称).
  - 每个实模式占用 1 维, 仅做标量 exp(μΔt) 缩放; 特征值由辅助网络以
    输入 yj 输出 μ.

三项损失 (公式编号对应论文 (11)-(15)):
  L_recon = ||x_1 - φ^{-1}(φ(x_1))||²                 -- 自编码重构
  L_pred  = (1/Sp) Σ_m ||x_{m+1} - φ^{-1}(K^m φ(x_1))||²   -- 多步预测
  L_lin   = (1/(T-1)) Σ_m ||φ(x_{m+1}) - K^m φ(x_1)||²     -- 潜空间线性一致性
  L_inf   = ||x_1 - φ^{-1}(φ(x_1))||_inf + ||x_2 - φ^{-1}(K φ(x_1))||_inf
  L_total = α₁ (L_recon + L_pred) + L_lin + α₂ L_inf + α₃ ||W||₂²

────────────────────────────────────────────────────────────────────────────
本脚本 vs 原论文/原仓库的几个差异
────────────────────────────────────────────────────────────────────────────
1) 数据源: 原论文是 ODE 数值解 (MATLAB ode45); 本脚本用 **MuJoCo 仿真**
   `multi_joint_cable_dirven_space_robot.xml`, 这样能保留 8 绳拮抗 + 4 关节
   等约束 + 关节阻尼 + 几何耦合等真实物理特性, 数据天然带轻微数值噪声.
2) 状态空间: 取 x = [qa, qb, dqa, dqb] ∈ R⁴  (利用 joint1≡joint2, joint3≡joint4
   的等约束, 2-DOF 系统的最简自治状态).
3) 控制输入: 本脚本采集 **自治轨迹** (8 根绳索施加恒定预紧 F_PRELOAD),
   对应论文中 "无外加控制" 的设定. 关节阻尼 (XML damping=2.0) + 几何耦合
   会产生类似 "受阻尼非线性摆" 的连续谱衰减振荡, 是 Lusch 方法的典型用例.
4) 框架: 用 PyTorch 实现, 与原仓库 (TensorFlow 1.x) 结构等价.

────────────────────────────────────────────────────────────────────────────
输出 (所有结果统一写到 outputs/figures/<本脚本名>/<时间戳>/)
────────────────────────────────────────────────────────────────────────────
    outputs/figures/deep_koopman_cdsm/<ts>/
        ├── dataset.npz                  原始 MuJoCo 数据集 (train/val 轨迹)
        ├── normalization.npz            均值 / 标准差 (用于测试脚本反归一化)
        ├── best_model.pt                PyTorch checkpoint (state_dict + 配置)
        ├── training_history.csv         每个 epoch 的训练/验证损失数值
        ├── training_history.{png,svg,pdf}    损失曲线 (4 子图)
        ├── prediction_demo.{png,svg,pdf}     验证集上的多步预测对比图
        └── phase_diagram_train.{png,svg,pdf} 训练集相图 (qa-dqa, qb-dqb)

运行:
    python deep_koopman_cdsm.py                # 默认参数完整训练
    python deep_koopman_cdsm.py --epochs 60    # 自定义训练轮数
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 优先用 glfw 后端 (与项目其他脚本一致, 避免 EGL/OSMesa 不可用导致崩溃)
os.environ.setdefault("MUJOCO_GL", "glfw")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False
    print("[ERR] 未检测到 mujoco 包, 请先安装: pip install mujoco")

from utils_plot import save_figure, get_save_dir


# ============================================================================
# 0. 全局常量 (与 XML / 其他脚本严格保持一致)
# ============================================================================
XML_PATH = str(
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "multi_joint_cable_driven_space_robot.xml"
)

# 8 根绳索 + 8 个 winch 电机的名字顺序 (与 XML 中 <tendon> / <actuator> 一致)
CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

# 物理参数 (统一与项目中其他脚本对齐)
F_PRELOAD = 20.0      # N, 每根绳索的恒定预紧张力 (使系统自治, 但带几何耦合)
DT_SIM = 0.02         # 仿真步长 (= 数据采样周期); 论文 Δt 用法相同
STATE_DIM = 4         # 状态维数: [qa, qb, dqa, dqb]


# ============================================================================
# 1. 随机种子
# ============================================================================
def set_seed(seed: int) -> None:
    """同时种子 random / numpy / torch (含 CUDA), 保证可复现."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 2. MuJoCo 模型加载与索引建立
# ============================================================================
def load_mujoco_model(dt: float = DT_SIM) -> Tuple[mujoco.MjModel, mujoco.MjData, Dict]:
    """加载 MJCF 模型, 并把 timestep 与本脚本采样周期对齐.

    返回:
        model : MjModel
        data  : MjData (零初始化)
        indices : dict 包含各种 name -> id 索引
    """
    if not _HAS_MUJOCO:
        raise RuntimeError("MuJoCo 未安装")
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"未找到 XML 模型: {XML_PATH}")

    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_str = f.read()
    # 同步 timestep, 防 XML 里的 0.01 默认值与本脚本 DT_SIM 不一致
    xml_str = re.sub(r'timestep="[^"]*"', f'timestep="{dt:g}"', xml_str)
    # 略微放宽关节限位 (原 ±90° -> ±97°), 给采样留余量
    xml_str = re.sub(r'range="-1\.5708 1\.5708"', 'range="-1.7 1.7"', xml_str)

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    # 建立各类对象的 name -> id 索引, 后续 setpoint / 读 sensor 时用
    indices = {
        "tdn_id":   {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,   n) for n in CABLE_NAMES},
        "act_id":   {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES},
        "jnt_id":   {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    n) for n in JOINT_NAMES},
        "site_ee":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"),
    }
    # 一致性自检: 关键索引必须 >= 0
    for label, d in [("tendon", indices["tdn_id"]),
                     ("actuator", indices["act_id"]),
                     ("joint", indices["jnt_id"])]:
        for k, v in d.items():
            if v < 0:
                raise RuntimeError(f"[load_mujoco_model] 未找到 {label} {k!r}")
    if indices["site_ee"] < 0:
        raise RuntimeError("[load_mujoco_model] 未找到 site 'end_effector'")

    # 关节在 qpos / qvel 数组里的地址 (用于直接写 IC)
    indices["qadr"] = {n: int(model.jnt_qposadr[indices["jnt_id"][n]]) for n in JOINT_NAMES}
    indices["dadr"] = {n: int(model.jnt_dofadr[indices["jnt_id"][n]])  for n in JOINT_NAMES}

    return model, data, indices


# ============================================================================
# 3. MuJoCo 数据采集: 自治轨迹生成器
# ============================================================================
@dataclass
class DataConfig:
    """数据采集配置.

    含义:
        traj_count    : 轨迹条数
        traj_len      : 每条轨迹的步数 (含 t=0 那一帧)
        dt            : 仿真步长 (s)
        qa_range      : qa 初值的均匀分布范围 (rad), 对称
        qb_range      : qb 初值的均匀分布范围 (rad), 对称
        dqa_range     : dqa 初值范围 (rad/s)
        dqb_range     : dqb 初值范围 (rad/s)
        f_preload     : 8 根绳索的恒定预紧张力 (N), 让系统保持轻度耦合 + 自治
        seed          : 采集所用的 numpy 随机种子 (用于复现)
    """
    traj_count: int
    traj_len: int
    dt: float = DT_SIM
    qa_range: float = 1.0
    qb_range: float = 1.0
    dqa_range: float = 0.8
    dqb_range: float = 0.8
    f_preload: float = F_PRELOAD
    seed: int = 0


def collect_mujoco_trajectories(cfg: DataConfig, verbose: bool = True) -> Dict[str, np.ndarray]:
    """从 MuJoCo 中采集 cfg.traj_count 条 cfg.traj_len 步的自治轨迹.

    具体做法:
      (1) 每条轨迹独立采样 (qa₀, qb₀, dqa₀, dqb₀) 作为初始条件 (IC);
      (2) 写入 data.qpos / data.qvel, 然后调用 mj_forward 刷新派生量;
      (3) 设 data.ctrl[:] = f_preload (8 根绳都施加同样预紧);
      (4) 按 dt 步长积分 traj_len-1 步, 每步记录 [qa, qb, dqa, dqb] 与末端 xy.

    返回:
        {
          "states"      : np.ndarray [traj_count, traj_len, 4]   状态轨迹 (主数据)
          "ee_xy"       : np.ndarray [traj_count, traj_len, 2]   末端坐标 (用于可视化)
          "initial_ic"  : np.ndarray [traj_count, 4]             初始条件 (调试用)
          "dt"          : np.ndarray [1]                          采样周期
          "f_preload"   : np.ndarray [1]                          预紧张力
        }
    """
    if verbose:
        print(f"\n[采集] 准备从 MuJoCo 采集 {cfg.traj_count} 条 × {cfg.traj_len} 步轨迹...")
        print(f"        IC 范围: qa,qb ∈ ±{cfg.qa_range:.2f} rad, "
              f"dqa,dqb ∈ ±{cfg.dqa_range:.2f} rad/s")
        print(f"        预紧张力: {cfg.f_preload} N  ·  dt = {cfg.dt} s")

    model, data, idx = load_mujoco_model(dt=cfg.dt)
    rng = np.random.RandomState(cfg.seed)

    # 输出缓冲
    states = np.zeros((cfg.traj_count, cfg.traj_len, STATE_DIM), dtype=np.float32)
    ee_xy = np.zeros((cfg.traj_count, cfg.traj_len, 2), dtype=np.float32)
    ic_all = np.zeros((cfg.traj_count, STATE_DIM), dtype=np.float32)

    # 8 根绳索 ctrl 的恒定预紧 (一次写入, 整个采集都不动)
    ctrl_const = np.full(model.nu, cfg.f_preload, dtype=float)

    t_start = time.time()
    for n in range(cfg.traj_count):
        # ---------- IC: 在合理范围内均匀采样 ----------
        qa0  = float(rng.uniform(-cfg.qa_range,  cfg.qa_range))
        qb0  = float(rng.uniform(-cfg.qb_range,  cfg.qb_range))
        dqa0 = float(rng.uniform(-cfg.dqa_range, cfg.dqa_range))
        dqb0 = float(rng.uniform(-cfg.dqb_range, cfg.dqb_range))
        ic_all[n] = [qa0, qb0, dqa0, dqb0]

        # 写 qpos / qvel: joint1≡joint2 = qa, joint3≡joint4 = qb (等约束在 mj_step
        # 里会强制再校正, 这里手动 set 是为了让初始构型严格符合约束)
        data.qpos[idx["qadr"]["joint1"]] = qa0
        data.qpos[idx["qadr"]["joint2"]] = qa0
        data.qpos[idx["qadr"]["joint3"]] = qb0
        data.qpos[idx["qadr"]["joint4"]] = qb0
        data.qvel[idx["dadr"]["joint1"]] = dqa0
        data.qvel[idx["dadr"]["joint2"]] = dqa0
        data.qvel[idx["dadr"]["joint3"]] = dqb0
        data.qvel[idx["dadr"]["joint4"]] = dqb0
        data.act[:] = 0.0
        data.time = 0.0
        data.ctrl[:] = ctrl_const
        mujoco.mj_forward(model, data)

        # ---------- 记录 t=0 帧 ----------
        states[n, 0] = [
            float(data.qpos[idx["qadr"]["joint1"]]),
            float(data.qpos[idx["qadr"]["joint3"]]),
            float(data.qvel[idx["dadr"]["joint1"]]),
            float(data.qvel[idx["dadr"]["joint3"]]),
        ]
        ee_xy[n, 0] = data.site_xpos[idx["site_ee"]][:2]

        # ---------- 后续 traj_len-1 步 ----------
        for t in range(1, cfg.traj_len):
            data.ctrl[:] = ctrl_const                # 重申预紧, 防止其它代码改 ctrl
            mujoco.mj_step(model, data)
            states[n, t] = [
                float(data.qpos[idx["qadr"]["joint1"]]),
                float(data.qpos[idx["qadr"]["joint3"]]),
                float(data.qvel[idx["dadr"]["joint1"]]),
                float(data.qvel[idx["dadr"]["joint3"]]),
            ]
            ee_xy[n, t] = data.site_xpos[idx["site_ee"]][:2]

        if verbose and ((n + 1) % max(1, cfg.traj_count // 10) == 0):
            elapsed = time.time() - t_start
            rate = (n + 1) / max(1e-9, elapsed)
            eta = (cfg.traj_count - n - 1) / max(1e-9, rate)
            print(f"  [{n+1:>5d}/{cfg.traj_count}]  rate={rate:.1f} traj/s  ETA={eta:.1f}s")

    if verbose:
        print(f"[采集] 完成. 总耗时 {time.time()-t_start:.1f} s. "
              f"states.shape = {states.shape}")
    return {
        "states":     states,
        "ee_xy":      ee_xy,
        "initial_ic": ic_all,
        "dt":         np.array([cfg.dt], dtype=np.float32),
        "f_preload":  np.array([cfg.f_preload], dtype=np.float32),
    }


# ============================================================================
# 4. 数据归一化: 训练前对状态做标准化 (z-score)
# ============================================================================
def fit_normalization(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """对 data shape=[N, T, D] 计算 per-channel 的 mean / std (在 N*T 上聚合)."""
    flat = data.reshape(-1, data.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def apply_normalization(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean) / std


# ============================================================================
# 5. 核心网络: Lusch-style Deep Koopman
# ============================================================================
class MLP(nn.Module):
    """通用全连接网络 (隐藏层带激活, 输出层线性)."""
    def __init__(self, widths: Tuple[int, ...], activation: str = "relu") -> None:
        super().__init__()
        act_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: List[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i != len(widths) - 2:                    # 最后一层不带激活
                layers.append(act_map[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepKoopmanCDSM(nn.Module):
    """Lusch (2018) 风格 Deep Koopman 模型, 用于 CDSM 系统.

    Args:
      state_dim          : 输入状态维数 (此处为 4)
      encoder_hidden     : 编码器隐藏层宽度 tuple, 如 (128, 128)
      decoder_hidden     : 解码器隐藏层宽度 tuple
      omega_hidden       : 辅助 omega 网络隐藏层宽度 tuple
      num_complex_pairs  : 复共轭特征对数 (每对占 2 维潜空间)
      num_real           : 实特征值数 (每个占 1 维潜空间)
      activation         : 隐藏层激活函数

    潜空间维度: latent_dim = 2 * num_complex_pairs + num_real.
    """

    def __init__(
        self,
        state_dim: int,
        encoder_hidden: Tuple[int, ...] = (128, 128),
        decoder_hidden: Tuple[int, ...] = (128, 128),
        omega_hidden:   Tuple[int, ...] = (128, 128),
        num_complex_pairs: int = 2,
        num_real: int = 0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_complex_pairs = num_complex_pairs
        self.num_real = num_real
        self.latent_dim = 2 * num_complex_pairs + num_real
        if self.latent_dim <= 0:
            raise ValueError("latent_dim 必须 > 0")

        # 编码器: state_dim -> ... -> latent_dim
        self.encoder = MLP((state_dim,) + tuple(encoder_hidden) + (self.latent_dim,),
                           activation=activation)
        # 解码器: latent_dim -> ... -> state_dim
        self.decoder = MLP((self.latent_dim,) + tuple(decoder_hidden) + (state_dim,),
                           activation=activation)

        # 辅助网络: 一个特征块对应一个独立子网络
        #   - 复块: 输入 r² = yj²+yj+1², 输出 (omega, mu)
        #   - 实块: 输入 yj, 输出 mu
        self.omega_complex_nets = nn.ModuleList([
            MLP((1,) + tuple(omega_hidden) + (2,), activation=activation)
            for _ in range(num_complex_pairs)
        ])
        self.omega_real_nets = nn.ModuleList([
            MLP((1,) + tuple(omega_hidden) + (1,), activation=activation)
            for _ in range(num_real)
        ])

    # ----------------------------- 编码 / 解码 -----------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x ∈ [B, n] -> y ∈ [B, latent_dim]"""
        return self.encoder(x)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """y ∈ [B, latent_dim] -> x_rec ∈ [B, n]"""
        return self.decoder(y)

    # ---------------- 辅助网络: 由当前 y 输出局部特征值 (μ, ω) ----------------
    def compute_omegas(self, y: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """根据当前 y 输出每个特征块的 (μ, ω). 对复块按 ||y_pair||² 输入, 保证圆对称."""
        complex_omegas = []
        real_omegas = []
        for j in range(self.num_complex_pairs):
            idx = 2 * j
            pair = y[:, idx:idx + 2]                     # [B, 2]
            r_squared = torch.sum(pair * pair, dim=1, keepdim=True)   # [B, 1]
            complex_omegas.append(self.omega_complex_nets[j](r_squared))  # [B, 2] = (ω, μ)
        for j in range(self.num_real):
            idx = 2 * self.num_complex_pairs + j
            yj = y[:, idx:idx + 1]                       # [B, 1]
            real_omegas.append(self.omega_real_nets[j](yj))   # [B, 1] = (μ,)
        return {"complex": complex_omegas, "real": real_omegas}

    # -------------- 单步线性演化 K(λ(y)) · y, 与论文 (4) 完全一致 --------------
    def koopman_step(self, y: torch.Tensor, dt: float) -> torch.Tensor:
        """y_{k+1} = K(λ(y_k)) · y_k.

        - 复块: B(μ,ω) = exp(μΔt) · [[cosωΔt, -sinωΔt], [sinωΔt, cosωΔt]]
        - 实块: 标量 exp(μΔt) · y
        """
        omegas = self.compute_omegas(y)
        parts: List[torch.Tensor] = []

        # ------ 复共轭 2x2 Jordan 块 ------
        for j in range(self.num_complex_pairs):
            idx = 2 * j
            y_pair = y[:, idx:idx + 2]                   # [B, 2]
            omega = omegas["complex"][j][:, 0:1]         # [B, 1]
            mu    = omegas["complex"][j][:, 1:2]         # [B, 1]
            scale = torch.exp(mu * dt)
            cos = torch.cos(omega * dt)
            sin = torch.sin(omega * dt)
            y0 = y_pair[:, 0:1]
            y1 = y_pair[:, 1:2]
            # 论文 Eq.(4): 第一行 [cosωΔt, -sinωΔt]; 第二行 [sinωΔt, cosωΔt]
            n0 = scale * (cos * y0 - sin * y1)
            n1 = scale * (sin * y0 + cos * y1)
            parts.append(torch.cat([n0, n1], dim=1))

        # ------ 实特征值 1x1 块 ------
        for j in range(self.num_real):
            idx = 2 * self.num_complex_pairs + j
            yj = y[:, idx:idx + 1]
            mu = omegas["real"][j]                       # [B, 1]
            parts.append(yj * torch.exp(mu * dt))

        return torch.cat(parts, dim=1)                   # [B, latent_dim]

    # ---- 在潜空间做 m 步外推, 返回 m+1 个时刻的 y (含起点) ----
    def latent_rollout(self, y0: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        ys = [y0]
        y = y0
        for _ in range(steps):
            y = self.koopman_step(y, dt)
            ys.append(y)
        return torch.stack(ys, dim=0)                    # [steps+1, B, latent_dim]

    # ---- 全前向: x0 -> 解码出未来 m+1 个时刻的预测状态 ----
    def forward(self, x0: torch.Tensor, steps: int, dt: float) -> Dict[str, torch.Tensor]:
        y0 = self.encode(x0)
        y_roll = self.latent_rollout(y0, steps=steps, dt=dt)
        x_roll = torch.stack([self.decode(y_roll[t]) for t in range(steps + 1)], dim=0)
        return {"y_roll": y_roll, "x_roll": x_roll}


# ============================================================================
# 6. 数据批采样: 从 [N, T, D] 中随机切出 (steps+1) 步的滑窗
# ============================================================================
def sample_batch_windows(
    data: np.ndarray, batch_size: int, steps: int, device: torch.device
) -> torch.Tensor:
    """随机起始位置, 切 (steps+1) 步的窗口. 返回 [steps+1, B, D] 张量."""
    n_traj, t_len, d = data.shape
    need = steps + 1
    if t_len < need:
        raise ValueError(f"轨迹长度 {t_len} < steps+1 = {need}")
    starts_max = t_len - need

    out = np.zeros((need, batch_size, d), dtype=np.float32)
    traj_ids = np.random.randint(0, n_traj, size=batch_size)
    st_ids   = np.random.randint(0, starts_max + 1, size=batch_size)
    for i in range(batch_size):
        out[:, i, :] = data[traj_ids[i], st_ids[i]:st_ids[i] + need, :]
    return torch.from_numpy(out).to(device)


# ============================================================================
# 7. 损失函数 (严格对应论文 (12)-(15))
# ============================================================================
def compute_losses(
    model: DeepKoopmanCDSM,
    batch_seq: torch.Tensor,
    dt: float,
    pred_steps: int,
    lin_steps: int,
    alpha1: float, alpha_inf: float, l2_lam: float,
) -> Dict[str, torch.Tensor]:
    """计算 L_recon, L_pred, L_lin, L_inf, L_total.

    Args:
        batch_seq : [steps+1, B, D] 时间序列窗口
        pred_steps: 预测损失对未来多少步求和 (论文 Sp)
        lin_steps : 潜空间线性一致性对未来多少步求和 (论文 T-1)
        alpha1    : (L_recon + L_pred) 的权重
        alpha_inf : L_inf 的权重 (论文 α₂)
        l2_lam    : L2 正则权重 (论文 α₃)
    """
    x0 = batch_seq[0]                                 # [B, D]
    fwd = model(x0, steps=pred_steps, dt=dt)
    x_roll      = fwd["x_roll"]                       # [pred+1, B, D] 解码后预测
    y_roll_pred = fwd["y_roll"]                       # [pred+1, B, K] 潜空间外推

    # ---- (a) 自编码重构损失 ----  ||x_1 - φ⁻¹(φ(x_1))||²
    recon_loss = torch.mean((x_roll[0] - batch_seq[0]) ** 2)

    # ---- (b) 多步预测损失 ----  Σ_m ||x_{m+1} - φ⁻¹(K^m φ(x_1))||²
    pred_loss = torch.mean((x_roll[1:] - batch_seq[1:pred_steps + 1]) ** 2)

    # ---- (c) 潜空间线性一致性损失 ----  Σ_m ||φ(x_{m+1}) - K^m φ(x_1)||²
    lin_horizon = min(lin_steps, pred_steps)
    true_y = torch.stack([model.encode(batch_seq[t]) for t in range(lin_horizon + 1)], dim=0)
    lin_loss = torch.mean((y_roll_pred[:lin_horizon + 1] - true_y) ** 2)

    # ---- (d) L∞ 项: 防止预测最大误差被均值掩盖 ----
    inf_recon = torch.max(torch.abs(x_roll[0] - batch_seq[0]))
    inf_pred1 = torch.max(torch.abs(x_roll[1] - batch_seq[1]))
    inf_loss  = inf_recon + inf_pred1

    # ---- (e) L2 正则化项 ----
    l2 = torch.zeros(1, device=batch_seq.device)
    if l2_lam > 0:
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:                    # 只罚权重矩阵, 不罚偏置
                l2 = l2 + torch.sum(p * p)
        l2 = l2 * l2_lam

    total = alpha1 * (recon_loss + pred_loss) + lin_loss + alpha_inf * inf_loss + l2.squeeze()

    return {
        "total": total, "recon": recon_loss, "pred": pred_loss,
        "lin":   lin_loss, "inf":   inf_loss, "l2":   l2.squeeze().detach(),
    }


@torch.no_grad()
def evaluate(
    model: DeepKoopmanCDSM, val_data: np.ndarray, device: torch.device,
    dt: float, pred_steps: int, lin_steps: int,
    alpha1: float, alpha_inf: float, l2_lam: float,
    eval_batches: int = 20, batch_size: int = 256,
) -> Dict[str, float]:
    """在验证集上跑若干 batch, 取损失项的均值."""
    model.eval()
    acc = {"total": 0.0, "recon": 0.0, "pred": 0.0, "lin": 0.0, "inf": 0.0}
    for _ in range(eval_batches):
        batch = sample_batch_windows(val_data, batch_size, pred_steps, device)
        losses = compute_losses(model, batch, dt, pred_steps, lin_steps,
                                alpha1, alpha_inf, l2_lam)
        for k in acc:
            acc[k] += float(losses[k].detach().item())
    for k in acc:
        acc[k] /= eval_batches
    return acc


# ============================================================================
# 8. 绘图: 训练历史, 多步预测对比, 训练集相图
# ============================================================================
def plot_training_history(history: List[List[float]]) -> None:
    """画 4 个子图 (log y 轴): total / recon / pred / lin."""
    arr = np.array(history)
    epochs = arr[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("Total loss",        1, 6, "C0"),
        ("Reconstruction",    2, 7, "C1"),
        ("Multi-step pred",   3, 8, "C2"),
        ("Latent linearity",  4, 9, "C3"),
    ]
    for ax, (title, ci_tr, ci_va, color) in zip(axes.ravel(), panels):
        ax.semilogy(epochs, arr[:, ci_tr], "-", color=color, lw=1.8, label="Train")
        ax.semilogy(epochs, arr[:, ci_va], "--", color=color, lw=1.8, alpha=0.8, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title + " (log)")
        ax.set_title(title)
        ax.grid(True, alpha=0.4, which="both")
        ax.legend(fontsize=9)

    plt.suptitle("Deep Koopman (Lusch 2018) Training on CDSM (MuJoCo data)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig_name="training_history")
    plt.close()


@torch.no_grad()
def plot_prediction_demo(
    model: DeepKoopmanCDSM, val_raw: np.ndarray, mean: np.ndarray, std: np.ndarray,
    dt: float, roll_steps: int, device: torch.device, n_demo: int = 3,
) -> None:
    """从验证集随机抽 n_demo 条, 自循环预测 roll_steps 步, 与真值并排对比."""
    rng = np.random.RandomState(0)
    indices = rng.choice(val_raw.shape[0], size=min(n_demo, val_raw.shape[0]), replace=False)

    model.eval()
    fig, axes = plt.subplots(n_demo, 4, figsize=(18, 3.5 * n_demo))
    if n_demo == 1:
        axes = axes[np.newaxis, :]

    for row, traj_idx in enumerate(indices):
        traj_real = val_raw[traj_idx]                          # [T, 4] 原始 (未归一化)
        x0_n = torch.from_numpy(((traj_real[0:1] - mean) / std).astype(np.float32)).to(device)
        # 自循环预测 (在归一化空间)
        ys = [model.encode(x0_n)]
        for _ in range(roll_steps - 1):
            ys.append(model.koopman_step(ys[-1], dt))
        x_pred_n = torch.stack([model.decode(y) for y in ys], dim=0).squeeze(1).cpu().numpy()
        x_pred = x_pred_n * std + mean                          # 反归一化回物理量纲

        x_real = traj_real[:roll_steps]
        t = np.arange(roll_steps) * dt
        labels = [r"$q_a$ (rad)", r"$q_b$ (rad)", r"$\dot q_a$ (rad/s)", r"$\dot q_b$ (rad/s)"]
        for col, (lab, ax) in enumerate(zip(labels, axes[row])):
            ax.plot(t, x_real[:, col], "k-",  lw=2.4, label="MuJoCo (true)")
            ax.plot(t, x_pred[:, col], "r--", lw=1.8, label="Koopman (pred)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(lab)
            ax.set_title(f"Val traj #{traj_idx} — {lab}")
            ax.grid(True, alpha=0.4)
            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    plt.suptitle("Deep Koopman Multi-step Rollout vs MuJoCo Ground Truth (Validation)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig_name="prediction_demo")
    plt.close()


def plot_phase_diagram_train(states: np.ndarray) -> None:
    """画训练集的 (qa, dqa) 与 (qb, dqb) 相图, 给数据分布一个直观印象."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    n_show = min(80, states.shape[0])
    for n in range(n_show):
        axes[0].plot(states[n, :, 0], states[n, :, 2], lw=0.6, alpha=0.45)
        axes[1].plot(states[n, :, 1], states[n, :, 3], lw=0.6, alpha=0.45)
    axes[0].set_title(r"Training trajectories in $(q_a, \dot q_a)$")
    axes[0].set_xlabel(r"$q_a$ (rad)"); axes[0].set_ylabel(r"$\dot q_a$ (rad/s)")
    axes[1].set_title(r"Training trajectories in $(q_b, \dot q_b)$")
    axes[1].set_xlabel(r"$q_b$ (rad)"); axes[1].set_ylabel(r"$\dot q_b$ (rad/s)")
    for ax in axes:
        ax.grid(True, alpha=0.4)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
    plt.suptitle("Training Set Phase Portraits (random IC, autonomous decay)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig_name="phase_diagram_train")
    plt.close()


# ============================================================================
# 9. CLI 参数
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lusch 2018 Deep Koopman on CDSM (MuJoCo data).")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="cuda", choices=["auto", "cpu", "cuda"])

    # 数据采集
    p.add_argument("--train_traj", type=int, default=1500)
    p.add_argument("--val_traj",   type=int, default=300)
    p.add_argument("--traj_len",   type=int, default=80)
    p.add_argument("--dt",         type=float, default=DT_SIM)
    p.add_argument("--qa_range",   type=float, default=1.0)
    p.add_argument("--qb_range",   type=float, default=1.0)
    p.add_argument("--dqa_range",  type=float, default=0.8)
    p.add_argument("--dqb_range",  type=float, default=0.8)
    p.add_argument("--f_preload",  type=float, default=F_PRELOAD)

    # 模型架构
    p.add_argument("--enc_hidden",   type=int, nargs="+", default=[128, 128])
    p.add_argument("--dec_hidden",   type=int, nargs="+", default=[128, 128])
    p.add_argument("--omega_hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--num_complex_pairs", type=int, default=2)
    p.add_argument("--num_real",     type=int, default=0)
    p.add_argument("--activation",   type=str, default="relu",
                   choices=["relu", "elu", "tanh", "sigmoid"])

    # 训练
    p.add_argument("--epochs",          type=int,   default=40)
    p.add_argument("--steps_per_epoch", type=int,   default=120)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--pred_steps",      type=int,   default=10)
    p.add_argument("--lin_steps",       type=int,   default=10)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--grad_clip",       type=float, default=5.0)

    # 损失权重 (与论文表 4 风格一致)
    p.add_argument("--alpha1",     type=float, default=0.1,
                   help="(L_recon + L_pred) 的权重")
    p.add_argument("--alpha_inf",  type=float, default=1e-7,
                   help="L_inf 的权重")
    p.add_argument("--l2_lam",     type=float, default=1e-9,
                   help="L2 权重正则系数")
    return p


# ============================================================================
# 10. 主程序
# ============================================================================
def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n=== Deep Koopman on CDSM (Lusch 2018 重现) ===")
    print(f"  设备 : {device}   ·   随机种子 : {args.seed}")

    # ---------- 输出目录: outputs/figures/<本脚本名>/<时间戳>/ ----------
    out_dir = Path(get_save_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录 : {out_dir}")

    # ============================================================
    # Step 1. 用 MuJoCo 采集训练 / 验证轨迹
    # ============================================================
    train_cfg = DataConfig(
        traj_count=args.train_traj, traj_len=args.traj_len, dt=args.dt,
        qa_range=args.qa_range,  qb_range=args.qb_range,
        dqa_range=args.dqa_range, dqb_range=args.dqb_range,
        f_preload=args.f_preload, seed=args.seed,
    )
    val_cfg = DataConfig(
        traj_count=args.val_traj, traj_len=args.traj_len, dt=args.dt,
        qa_range=args.qa_range,  qb_range=args.qb_range,
        dqa_range=args.dqa_range, dqb_range=args.dqb_range,
        f_preload=args.f_preload, seed=args.seed + 1234,
    )

    train_set = collect_mujoco_trajectories(train_cfg, verbose=True)
    val_set   = collect_mujoco_trajectories(val_cfg,   verbose=True)

    train_raw = train_set["states"]                              # [Ntr, T, 4]
    val_raw   = val_set["states"]                                # [Nval, T, 4]

    # 训练集相图: 给数据分布一个直观印象
    plot_phase_diagram_train(train_raw)

    # 数据归一化 (z-score). 用 训练集 拟合, 然后应用到 val.
    mean, std = fit_normalization(train_raw)
    print(f"\n[归一化] mean = {mean.tolist()}\n            std  = {std.tolist()}")
    train_n = apply_normalization(train_raw, mean, std)
    val_n   = apply_normalization(val_raw,   mean, std)

    # 保存原始数据集 + 归一化参数, 测试脚本会直接读
    np.savez(
        out_dir / "dataset.npz",
        train_states=train_raw, val_states=val_raw,
        train_ee_xy=train_set["ee_xy"], val_ee_xy=val_set["ee_xy"],
        train_ic=train_set["initial_ic"], val_ic=val_set["initial_ic"],
        dt=np.array([args.dt], dtype=np.float32),
        f_preload=np.array([args.f_preload], dtype=np.float32),
    )
    np.savez(out_dir / "normalization.npz", mean=mean, std=std)
    print(f"[保存] 数据集与归一化参数 -> {out_dir.name}")

    # ============================================================
    # Step 2. 构建 Deep Koopman 模型 + 优化器
    # ============================================================
    model = DeepKoopmanCDSM(
        state_dim=STATE_DIM,
        encoder_hidden=tuple(args.enc_hidden),
        decoder_hidden=tuple(args.dec_hidden),
        omega_hidden=tuple(args.omega_hidden),
        num_complex_pairs=args.num_complex_pairs,
        num_real=args.num_real,
        activation=args.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型] 编码器隐藏={args.enc_hidden}  解码器隐藏={args.dec_hidden}")
    print(f"        omega 隐藏={args.omega_hidden}  "
          f"#复对={args.num_complex_pairs}  #实={args.num_real}  "
          f"latent_dim={model.latent_dim}")
    print(f"        可训练参数总数: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # ============================================================
    # Step 3. 训练循环
    # ============================================================
    best_val_total = math.inf
    history: List[List[float]] = []                  # epoch, tr_tot, tr_rec, tr_pre, tr_lin, tr_inf, va_tot, va_rec, va_pre, va_lin
    print(f"\n[训练] epochs={args.epochs}  steps/epoch={args.steps_per_epoch}  "
          f"batch={args.batch_size}  pred_steps={args.pred_steps}")
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_acc = {"total": 0.0, "recon": 0.0, "pred": 0.0, "lin": 0.0, "inf": 0.0}

        for _ in range(args.steps_per_epoch):
            batch = sample_batch_windows(train_n, args.batch_size, args.pred_steps, device)
            losses = compute_losses(
                model, batch, dt=args.dt,
                pred_steps=args.pred_steps, lin_steps=args.lin_steps,
                alpha1=args.alpha1, alpha_inf=args.alpha_inf, l2_lam=args.l2_lam,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for k in ep_acc:
                ep_acc[k] += float(losses[k].detach().item())

        for k in ep_acc:
            ep_acc[k] /= args.steps_per_epoch

        val = evaluate(
            model, val_n, device=device, dt=args.dt,
            pred_steps=args.pred_steps, lin_steps=args.lin_steps,
            alpha1=args.alpha1, alpha_inf=args.alpha_inf, l2_lam=args.l2_lam,
        )

        history.append([
            epoch,
            ep_acc["total"], ep_acc["recon"], ep_acc["pred"], ep_acc["lin"], ep_acc["inf"],
            val["total"],    val["recon"],    val["pred"],    val["lin"],
        ])

        print(f"  [Ep {epoch:03d}] "
              f"tr_tot={ep_acc['total']:.4e}  tr_rec={ep_acc['recon']:.4e}  "
              f"tr_pre={ep_acc['pred']:.4e}  tr_lin={ep_acc['lin']:.4e} | "
              f"va_tot={val['total']:.4e}  va_rec={val['recon']:.4e}  "
              f"va_pre={val['pred']:.4e}  va_lin={val['lin']:.4e}")

        # ---- 早停: 仅保留 val total 最优的 checkpoint ----
        if val["total"] < best_val_total:
            best_val_total = val["total"]
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "mean": mean, "std": std, "dt": args.dt,
                "state_dim": STATE_DIM,
                "best_val": best_val_total,
                "epoch": epoch,
            }, out_dir / "best_model.pt")

    train_time = time.time() - t_train_start
    print(f"\n[训练] 完成. 总耗时 {train_time:.1f} s. 最佳 val_total = {best_val_total:.4e}")

    # 保存损失曲线 csv
    np.savetxt(
        out_dir / "training_history.csv",
        np.array(history, dtype=np.float64),
        delimiter=",",
        header=("epoch,train_total,train_recon,train_pred,train_lin,train_inf,"
                "val_total,val_recon,val_pred,val_lin"),
        comments="",
    )
    print(f"[保存] checkpoint -> {out_dir / 'best_model.pt'}")
    print(f"[保存] 历史 csv   -> {out_dir / 'training_history.csv'}")

    # ============================================================
    # Step 4. 绘图: 训练历史 + 验证集多步预测 demo
    # ============================================================
    print("\n[绘图] 训练历史曲线 + 验证集多步预测对比...")
    plot_training_history(history)

    # 重新加载最佳模型再画 demo
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    plot_prediction_demo(
        model, val_raw, mean, std, dt=args.dt,
        roll_steps=min(args.traj_len, args.pred_steps + 30),
        device=device, n_demo=3,
    )

    print(f"\n=== 完成. 所有结果已保存至: {out_dir} ===\n")
    print("下一步: 运行测试脚本以全面对比 Koopman vs MuJoCo:")
    print("    python test_deep_koopman_cdsm.py\n")


if __name__ == "__main__":
    main()
