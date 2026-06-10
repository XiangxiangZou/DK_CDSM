"""
基于 MuJoCo 数据的关节力矩驱动 CDSM 的 EDMD 动力学预测程序。

本脚本的目标：
    1. 使用 multi_joint_space_robot.xml 作为 ground truth 仿真模型；
    2. 随机采样关节初始状态和关节力矩输入，生成训练/验证轨迹；
    3. 使用 EDMD 训练离散时间预测模型；
    4. 对比 Hermite 字典和 RBF 字典的多步预测能力；
    5. 保存模型参数、数据集、预测曲线和误差统计结果。

默认状态和输入定义：
    x = [qa, qb, dqa, dqb]
        qa, qb   : 两个等效主动关节角；
        dqa, dqb : 两个等效主动关节角速度。

    u = [tau_a, tau_b]
        tau_a, tau_b : 直接写入 MuJoCo motor actuator 的关节力矩输入。

这里使用的是实用的受控 EDMD 回归形式：
    x_{k+1} = phi(x_k, u_k) W

其中 phi(x,u) 是由状态和输入共同构造的字典函数。这个形式适合后续在
已知控制序列 u_k 的情况下做多步 rollout 预测。

运行示例：
    python edmd_mujoco_cdsm_joint_torque.py
    python edmd_mujoco_cdsm_joint_torque.py --dictionary both --train_traj 300
    python edmd_mujoco_cdsm_joint_torque.py --dictionary rbf --rbf_centers 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from utils_plot import get_save_dir, save_figure


# MuJoCo XML 文件。该文件应包含 motor_qa / motor_qb 两个直接力矩执行器。
XML_PATH = str(
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "multi_joint_space_robot.xml"
)

# 等效 2-DOF 模型中真正作为独立广义坐标读取的关节。
# XML 中 joint2 跟随 joint1，joint4 跟随 joint3，因此这里读取 joint1/joint3。
ACTIVE_JOINTS = ("joint1", "joint3")

# equality constraint 对应的从属关节。初始化状态时手动同步 qpos，
# 之后 MuJoCo 的 equality constraint 会在仿真推进中保持约束。
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}

# 直接写入 data.ctrl 的两个执行器名称。
ACTUATOR_NAMES = ("motor_qa", "motor_qb")


@dataclass
class DataConfig:
    """MuJoCo 数据采样配置。"""

    traj_count: int
    """轨迹条数。"""
    steps: int
    """每条轨迹的离散步数。实际状态点数量为 steps + 1。"""
    dt: float
    """MuJoCo 仿真步长。"""
    seed: int
    """随机种子，用于初始状态和输入采样。"""
    q_range: float
    """初始关节角采样范围：[-q_range, q_range]。"""
    dq_range: float
    """初始角速度采样范围：[-dq_range, dq_range]。"""
    tau_range: float
    """关节力矩采样范围：[-tau_range, tau_range]。"""
    hold_steps: int
    """随机力矩保持步数。增大该值可使输入更平滑。"""


@dataclass
class FitConfig:
    """EDMD 拟合配置。"""

    dictionary: str
    """字典类型：hermite 或 rbf。"""
    ridge: float
    """岭回归正则系数，用于缓解 Phi^T Phi 病态。"""
    include_trig: bool
    """是否把角度转换为 sin/cos 特征。"""
    rbf_centers: int
    """RBF 字典中心数量。仅 dictionary='rbf' 时使用。"""
    rbf_sigma: Optional[float]
    """RBF 核宽度。为 None 时自动按中心距离中位数估计。"""
    rbf_seed: int
    """RBF 中心随机抽样种子。"""


def set_seed(seed: int) -> None:
    """设置 NumPy 随机种子，保证实验可复现。"""
    np.random.seed(seed)


def name_to_joint_id(model: mujoco.MjModel, name: str) -> int:
    """按关节名称查询 MuJoCo joint id，并在缺失时给出明确错误。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in XML: {name}")
    return jid


def name_to_actuator_id(model: mujoco.MjModel, name: str) -> int:
    """按执行器名称查询 MuJoCo actuator id，并在缺失时给出明确错误。"""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found in XML: {name}")
    return aid


def load_model(xml_path: str, dt: float) -> Tuple[mujoco.MjModel, mujoco.MjData, Dict[str, np.ndarray]]:
    """
    加载 MuJoCo 模型，并预先缓存常用索引。

    MuJoCo 中读取/写入状态通常需要通过数组下标完成，例如：
        data.qpos[model.jnt_qposadr[joint_id]]
        data.qvel[model.jnt_dofadr[joint_id]]

    为避免采样循环中反复查表，这里一次性计算：
        active_qpos  : joint1/joint3 的 qpos 下标；
        active_dof   : joint1/joint3 的 qvel 下标；
        actuator_ids : motor_qa/motor_qb 的 ctrl 下标；
        mimic_pairs  : joint2<-joint1、joint4<-joint3 的 qpos 同步关系。
    """
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    active_joint_ids = np.array([name_to_joint_id(model, n) for n in ACTIVE_JOINTS], dtype=int)
    active_qpos = np.array([model.jnt_qposadr[j] for j in active_joint_ids], dtype=int)
    active_dof = np.array([model.jnt_dofadr[j] for j in active_joint_ids], dtype=int)
    actuator_ids = np.array([name_to_actuator_id(model, n) for n in ACTUATOR_NAMES], dtype=int)

    mimic_pairs = []
    for mimic, source in MIMIC_JOINTS.items():
        mimic_jid = name_to_joint_id(model, mimic)
        source_jid = name_to_joint_id(model, source)
        mimic_pairs.append((model.jnt_qposadr[mimic_jid], model.jnt_qposadr[source_jid]))

    indices = {
        "active_qpos": active_qpos,
        "active_dof": active_dof,
        "actuator_ids": actuator_ids,
        "mimic_pairs": np.array(mimic_pairs, dtype=int),
    }
    return model, data, indices


def set_active_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    q: np.ndarray,
    dq: np.ndarray,
) -> None:
    """
    设置一条新轨迹的初始状态。

    注意：
        - 只随机设置等效主动关节 joint1 / joint3；
        - joint2 / joint4 是从属关节，初始化时同步到 joint1 / joint3；
        - 调用 mj_forward 让 MuJoCo 根据 qpos/qvel 更新派生量。
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[indices["active_qpos"]] = q
    data.qvel[indices["active_dof"]] = dq
    for mimic_qpos, source_qpos in indices["mimic_pairs"]:
        data.qpos[mimic_qpos] = data.qpos[source_qpos]
    mujoco.mj_forward(model, data)


def get_active_state(data: mujoco.MjData, indices: Dict[str, np.ndarray]) -> np.ndarray:
    """从 MuJoCo data 中读取 EDMD 使用的 4 维状态 [qa, qb, dqa, dqb]。"""
    q = data.qpos[indices["active_qpos"]]
    dq = data.qvel[indices["active_dof"]]
    return np.array([q[0], q[1], dq[0], dq[1]], dtype=np.float64)


def generate_piecewise_inputs(
    rng: np.random.RandomState,
    steps: int,
    tau_range: float,
    hold_steps: int,
) -> np.ndarray:
    """
    生成分段常值随机力矩输入。

    直接每一步独立采样白噪声力矩会导致输入过于高频，不利于得到平滑、
    可解释的机器人轨迹。这里每 hold_steps 步采样一次新力矩，中间保持不变。
    """
    hold_steps = max(1, int(hold_steps))
    block_count = int(np.ceil(steps / hold_steps))
    blocks = rng.uniform(-tau_range, tau_range, size=(block_count, 2))
    u = np.repeat(blocks, hold_steps, axis=0)[:steps]
    return u.astype(np.float64)


def collect_trajectories(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    cfg: DataConfig,
) -> Dict[str, np.ndarray]:
    """
    使用 MuJoCo 批量采集轨迹数据。

    返回：
        states: shape = (traj_count, steps + 1, 4)
            每条轨迹包含 steps+1 个状态点。

        inputs: shape = (traj_count, steps, 2)
            inputs[i,k] 是从 states[i,k] 推进到 states[i,k+1] 时使用的力矩。
    """
    rng = np.random.RandomState(cfg.seed)
    states = np.zeros((cfg.traj_count, cfg.steps + 1, 4), dtype=np.float64)
    inputs = np.zeros((cfg.traj_count, cfg.steps, 2), dtype=np.float64)

    for i in range(cfg.traj_count):
        # 每条轨迹随机一个初始状态，保证训练数据覆盖多种相空间区域。
        q0 = rng.uniform(-cfg.q_range, cfg.q_range, size=2)
        dq0 = rng.uniform(-cfg.dq_range, cfg.dq_range, size=2)
        set_active_state(model, data, indices, q0, dq0)

        # 为当前轨迹生成完整的控制输入序列。
        u_seq = generate_piecewise_inputs(rng, cfg.steps, cfg.tau_range, cfg.hold_steps)
        states[i, 0] = get_active_state(data, indices)

        for k in range(cfg.steps):
            # 将关节力矩写入 motor actuator，然后推进 MuJoCo 一步。
            data.ctrl[indices["actuator_ids"]] = u_seq[k]
            mujoco.mj_step(model, data)
            states[i, k + 1] = get_active_state(data, indices)
            inputs[i, k] = u_seq[k]

    return {"states": states, "inputs": inputs}


def flatten_transitions(dataset: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将轨迹数据整理成监督学习形式。

    原始数据：
        states[i,k] -> 第 i 条轨迹第 k 个状态；
        inputs[i,k] -> 从 states[i,k] 到 states[i,k+1] 的输入。

    展平后：
        x  = 所有 x_k；
        u  = 所有 u_k；
        xp = 所有 x_{k+1}。
    """
    states = dataset["states"]
    inputs = dataset["inputs"]
    x = states[:, :-1, :].reshape(-1, states.shape[-1])
    u = inputs.reshape(-1, inputs.shape[-1])
    xp = states[:, 1:, :].reshape(-1, states.shape[-1])
    return x, u, xp


@dataclass
class Normalizer:
    """
    简单标准化器。

    EDMD 字典会包含多项式和 RBF 特征。如果不同状态/输入量纲差异过大，
    Phi^T Phi 容易病态，因此拟合前对 x、u、x_next 分别做标准化。
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        """根据样本均值和标准差构造标准化器。"""
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """物理量 -> 标准化量。"""
        return (x - self.mean) / self.std

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        """标准化量 -> 物理量。"""
        return x_norm * self.std + self.mean


def angle_features(x_norm: np.ndarray, include_trig: bool) -> np.ndarray:
    """
    构造角度相关特征。

    机器人关节角具有周期性，直接用 qa/qb 会让模型在角度边界附近看到
    人为不连续。默认加入 sin/cos 特征：
        [sin(qa), cos(qa), sin(qb), cos(qb), dqa, dqb]

    注意这里使用的是标准化后的角度变量。对当前小角度范围实验可用；
    后续若要覆盖接近 +/-pi 的范围，可以改成先对原始角度取 sin/cos，
    再整体标准化。
    """
    if not include_trig:
        return x_norm
    qa = x_norm[:, 0:1]
    qb = x_norm[:, 1:2]
    rest = x_norm[:, 2:]
    return np.hstack([np.sin(qa), np.cos(qa), np.sin(qb), np.cos(qb), rest])


def hermite_dictionary(x_norm: np.ndarray, u_norm: np.ndarray, include_trig: bool) -> np.ndarray:
    """
    构造 Hermite/二阶多项式风格 EDMD 字典。

    输入：
        x_norm: 标准化状态；
        u_norm: 标准化输入。

    先拼接 z=[angle_features(x), u]，然后构造：
        1                    常数项；
        z_i                  一阶项；
        z_i^2 - 1            二阶 Hermite 多项式项；
        z_i z_j              二阶交叉项。

    该字典用于建立：
        x_{k+1} = phi(x_k,u_k) W
    """
    z = np.hstack([angle_features(x_norm, include_trig), u_norm])
    cols = [np.ones((z.shape[0], 1), dtype=np.float64), z, z * z - 1.0]

    pairs = []
    for i in range(z.shape[1]):
        for j in range(i + 1, z.shape[1]):
            pairs.append((z[:, i] * z[:, j])[:, None])
    if pairs:
        cols.append(np.hstack(pairs))
    return np.hstack(cols)


def choose_rbf_centers(
    x_norm: np.ndarray,
    u_norm: np.ndarray,
    include_trig: bool,
    n_centers: int,
    seed: int,
) -> np.ndarray:
    """
    从训练样本中随机抽取 RBF 中心。

    RBF 字典直接在联合变量 z=[angle_features(x),u] 空间中放置中心。
    这里采用随机子采样，简单稳定；后续可以替换为 k-means 中心。
    """
    z = np.hstack([angle_features(x_norm, include_trig), u_norm])
    rng = np.random.RandomState(seed)
    n_centers = min(n_centers, z.shape[0])
    ids = rng.choice(z.shape[0], size=n_centers, replace=False)
    return z[ids].copy()


def estimate_rbf_sigma(centers: np.ndarray) -> float:
    """
    根据 RBF 中心之间的距离自动估计核宽度 sigma。

    使用随机中心对距离的中位数作为尺度，避免 sigma 过小导致特征近似
    one-hot，也避免 sigma 过大导致所有 RBF 特征几乎相同。
    """
    if centers.shape[0] < 2:
        return 1.0
    rng = np.random.RandomState(0)
    sample_count = min(1000, centers.shape[0] * (centers.shape[0] - 1) // 2)
    i = rng.randint(0, centers.shape[0], size=sample_count)
    j = rng.randint(0, centers.shape[0], size=sample_count)
    mask = i != j
    if not np.any(mask):
        return 1.0
    d = np.linalg.norm(centers[i[mask]] - centers[j[mask]], axis=1)
    sigma = float(np.median(d))
    return max(sigma, 1e-6)


def rbf_dictionary(
    x_norm: np.ndarray,
    u_norm: np.ndarray,
    include_trig: bool,
    centers: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    构造 RBF EDMD 字典。

    字典形式：
        phi = [1, z, exp(-||z-c_i||^2/(2 sigma^2))]

    保留 z 的线性项是为了让模型至少具备基本线性预测能力，RBF 部分负责
    拟合局部非线性。
    """
    z = np.hstack([angle_features(x_norm, include_trig), u_norm])
    diff = z[:, None, :] - centers[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    rbf = np.exp(-0.5 * sqdist / (sigma * sigma))
    return np.hstack([np.ones((z.shape[0], 1)), z, rbf])


class EDMDPredictor:
    """
    EDMD 预测器封装。

    保存内容包括：
        - 字典类型；
        - x/u/x_next 的标准化参数；
        - 回归权重 W；
        - RBF 中心和核宽度。

    预测时必须使用与训练阶段完全一致的标准化和字典构造方式。
    """

    def __init__(
        self,
        dictionary: str,
        x_norm: Normalizer,
        u_norm: Normalizer,
        y_norm: Normalizer,
        weights: np.ndarray,
        include_trig: bool,
        centers: Optional[np.ndarray] = None,
        sigma: Optional[float] = None,
    ) -> None:
        self.dictionary = dictionary
        self.x_norm = x_norm
        self.u_norm = u_norm
        self.y_norm = y_norm
        self.weights = weights
        self.include_trig = include_trig
        self.centers = centers
        self.sigma = sigma

    def phi(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """对物理量 x/u 构造 EDMD 字典特征 phi(x,u)。"""
        x2 = np.atleast_2d(x)
        u2 = np.atleast_2d(u)
        xn = self.x_norm.transform(x2)
        un = self.u_norm.transform(u2)
        if self.dictionary == "hermite":
            return hermite_dictionary(xn, un, self.include_trig)
        if self.dictionary == "rbf":
            if self.centers is None or self.sigma is None:
                raise RuntimeError("RBF predictor is missing centers or sigma.")
            return rbf_dictionary(xn, un, self.include_trig, self.centers, self.sigma)
        raise ValueError(f"Unknown dictionary: {self.dictionary}")

    def predict_one(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """一步预测：给定 x_k 和 u_k，返回 x_{k+1}。"""
        y_norm = self.phi(x, u) @ self.weights
        return self.y_norm.inverse(y_norm)[0]

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """
        多步滚动预测。

        从真实初始状态 x0 出发，每一步都使用上一时刻的预测状态继续预测。
        因此该误差会反映模型在长期闭环 rollout 中的累积误差。
        """
        out = np.zeros((u_seq.shape[0] + 1, x0.shape[0]), dtype=np.float64)
        out[0] = x0
        x = x0.copy()
        for k, u in enumerate(u_seq):
            x = self.predict_one(x, u)
            out[k + 1] = x
        return out


def fit_edmd(
    x: np.ndarray,
    u: np.ndarray,
    xp: np.ndarray,
    cfg: FitConfig,
) -> EDMDPredictor:
    """
    拟合 EDMD 预测器。

    数学形式：
        y = normalize(x_{k+1})
        Phi = phi(normalize(x_k), normalize(u_k))
        W = argmin ||Phi W - y||_2^2 + ridge ||W||_2^2

    对应闭式解：
        W = (Phi^T Phi + ridge I)^(-1) Phi^T Y
    """
    x_norm = Normalizer.fit(x)
    u_norm = Normalizer.fit(u)
    y_norm = Normalizer.fit(xp)

    xn = x_norm.transform(x)
    un = u_norm.transform(u)
    yn = y_norm.transform(xp)

    centers = None
    sigma = None
    if cfg.dictionary == "hermite":
        # Hermite 字典不需要额外参数，直接构造 Phi。
        phi = hermite_dictionary(xn, un, cfg.include_trig)
    elif cfg.dictionary == "rbf":
        # RBF 字典需要先确定中心和核宽度。
        centers = choose_rbf_centers(xn, un, cfg.include_trig, cfg.rbf_centers, cfg.rbf_seed)
        sigma = cfg.rbf_sigma if cfg.rbf_sigma is not None else estimate_rbf_sigma(centers)
        phi = rbf_dictionary(xn, un, cfg.include_trig, centers, sigma)
    else:
        raise ValueError(f"Unsupported dictionary: {cfg.dictionary}")

    gram = phi.T @ phi
    rhs = phi.T @ yn
    reg = cfg.ridge * np.eye(gram.shape[0])
    # 使用 solve 而不是显式求逆，数值上更稳。
    weights = np.linalg.solve(gram + reg, rhs)

    return EDMDPredictor(
        dictionary=cfg.dictionary,
        x_norm=x_norm,
        u_norm=u_norm,
        y_norm=y_norm,
        weights=weights,
        include_trig=cfg.include_trig,
        centers=centers,
        sigma=sigma,
    )


def evaluate_model(model: EDMDPredictor, dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    在验证集上评估多步预测误差。

    评价指标：
        rmse_by_state : 每个状态维度的 RMSE；
        mae_by_state  : 每个状态维度的 MAE；
        step_rmse     : 每个预测步长上的总体 RMSE；
        total_rmse    : 所有轨迹、所有步长、所有状态维度上的总体 RMSE。
    """
    states = dataset["states"]
    inputs = dataset["inputs"]
    pred = np.zeros_like(states)

    for i in range(states.shape[0]):
        pred[i] = model.rollout(states[i, 0], inputs[i])

    err = pred - states
    rmse_by_state = np.sqrt(np.mean(err * err, axis=(0, 1)))
    mae_by_state = np.mean(np.abs(err), axis=(0, 1))
    step_rmse = np.sqrt(np.mean(err * err, axis=(0, 2)))
    total_rmse = float(np.sqrt(np.mean(err * err)))

    return {
        "pred": pred,
        "err": err,
        "rmse_by_state": rmse_by_state,
        "mae_by_state": mae_by_state,
        "step_rmse": step_rmse,
        "total_rmse": np.array([total_rmse]),
    }


def plot_rollout_comparison(
    states: np.ndarray,
    inputs: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    out_name: str,
    max_traj: int = 3,
) -> None:
    """绘制若干条验证轨迹上的 MuJoCo 真实状态与 EDMD 预测状态对比。"""
    labels = [r"$q_a$", r"$q_b$", r"$\dot q_a$", r"$\dot q_b$"]
    n_show = min(max_traj, states.shape[0])
    fig, axes = plt.subplots(n_show, 4, figsize=(15, 3.2 * n_show), squeeze=False)
    t = np.arange(states.shape[1])

    for row in range(n_show):
        for col in range(4):
            ax = axes[row, col]
            ax.plot(t, states[row, :, col], "k-", lw=1.8, label="MuJoCo")
            for name, res in results.items():
                ax.plot(t, res["pred"][row, :, col], "--", lw=1.2, label=name)
            ax.set_title(f"traj {row + 1}: {labels[col]}")
            ax.grid(True, alpha=0.3)
            if row == n_show - 1:
                ax.set_xlabel("step")
            if col == 0:
                ax.set_ylabel("state")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=1 + len(results), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(out_name)
    plt.close(fig)


def plot_error_growth(results: Dict[str, Dict[str, np.ndarray]], out_name: str) -> None:
    """绘制多步 rollout 中 RMSE 随预测步数增长的曲线。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, res in results.items():
        ax.plot(res["step_rmse"], lw=1.8, label=name)
    ax.set_xlabel("prediction step")
    ax.set_ylabel("RMSE over states")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(out_name)
    plt.close(fig)


def save_npz_model(path: Path, model: EDMDPredictor) -> None:
    """
    保存模型到 npz 文件。

    保存的内容足够用于后续恢复预测器：
        - 字典类型；
        - 回归权重；
        - 标准化参数；
        - RBF 参数。
    """
    payload = {
        "dictionary": np.array([model.dictionary]),
        "weights": model.weights,
        "x_mean": model.x_norm.mean,
        "x_std": model.x_norm.std,
        "u_mean": model.u_norm.mean,
        "u_std": model.u_norm.std,
        "y_mean": model.y_norm.mean,
        "y_std": model.y_norm.std,
        "include_trig": np.array([int(model.include_trig)]),
    }
    if model.centers is not None:
        payload["centers"] = model.centers
    if model.sigma is not None:
        payload["sigma"] = np.array([model.sigma])
    np.savez(path, **payload)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数定义。"""
    p = argparse.ArgumentParser(
        description="Collect MuJoCo CDSM torque data and fit EDMD Hermite/RBF predictors."
    )
    p.add_argument("--xml", default=XML_PATH, help="Path to MuJoCo XML.")
    p.add_argument("--dictionary", choices=["hermite", "rbf", "both"], default="both")
    p.add_argument("--train_traj", type=int, default=200)
    p.add_argument("--val_traj", type=int, default=40)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--q_range", type=float, default=0.8)
    p.add_argument("--dq_range", type=float, default=0.8)
    p.add_argument("--tau_range", type=float, default=28.0)
    p.add_argument("--hold_steps", type=int, default=8)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--no_trig", action="store_true", help="Disable sin/cos angle features.")
    p.add_argument("--rbf_centers", type=int, default=250)
    p.add_argument("--rbf_sigma", type=float, default=None)
    return p


def main() -> None:
    """主流程：采集数据 -> 拟合 EDMD -> 验证预测 -> 保存结果。"""
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(get_save_dir())

    # 加载 MuJoCo 模型并解析 joint/actuator 索引。
    model, data, indices = load_model(args.xml, args.dt)

    # 训练集与验证集使用不同随机种子，避免验证轨迹与训练轨迹重复。
    train_cfg = DataConfig(
        traj_count=args.train_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        q_range=args.q_range,
        dq_range=args.dq_range,
        tau_range=args.tau_range,
        hold_steps=args.hold_steps,
    )
    val_cfg = DataConfig(
        traj_count=args.val_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed + 1000,
        q_range=args.q_range,
        dq_range=args.dq_range,
        tau_range=args.tau_range,
        hold_steps=args.hold_steps,
    )

    print("[data] collecting training trajectories...")
    t0 = time.time()
    train = collect_trajectories(model, data, indices, train_cfg)
    print(f"[data] train shape: states={train['states'].shape}, inputs={train['inputs'].shape}")

    print("[data] collecting validation trajectories...")
    val = collect_trajectories(model, data, indices, val_cfg)
    print(f"[data] val shape: states={val['states'].shape}, inputs={val['inputs'].shape}")

    # 将轨迹形式的数据变成一组监督学习样本 (x_k, u_k, x_{k+1})。
    x, u, xp = flatten_transitions(train)

    # 可以单独训练 Hermite/RBF，也可以一次训练两者用于对比。
    dicts = ["hermite", "rbf"] if args.dictionary == "both" else [args.dictionary]
    predictors: Dict[str, EDMDPredictor] = {}
    results: Dict[str, Dict[str, np.ndarray]] = {}
    summary: Dict[str, object] = {
        "xml": args.xml,
        "data_config": asdict(train_cfg),
        "validation_config": asdict(val_cfg),
        "fit": {},
    }

    for name in dicts:
        # 每种字典使用相同的数据集和正则化设置，便于公平比较。
        fit_cfg = FitConfig(
            dictionary=name,
            ridge=args.ridge,
            include_trig=not args.no_trig,
            rbf_centers=args.rbf_centers,
            rbf_sigma=args.rbf_sigma,
            rbf_seed=args.seed + 2000,
        )
        print(f"[fit] fitting {name} EDMD predictor...")
        fit_start = time.time()
        predictor = fit_edmd(x, u, xp, fit_cfg)
        fit_time = time.time() - fit_start
        predictors[name] = predictor

        print(f"[eval] evaluating {name} multi-step rollout...")
        res = evaluate_model(predictor, val)
        results[name] = res
        print(f"[eval] {name} total RMSE: {res['total_rmse'][0]:.6g}")

        save_npz_model(out_dir / f"edmd_{name}_model.npz", predictor)
        summary["fit"][name] = {
            "fit_config": asdict(fit_cfg),
            "fit_time_sec": fit_time,
            "feature_dim": int(predictor.weights.shape[0]),
            "output_dim": int(predictor.weights.shape[1]),
            "total_rmse": float(res["total_rmse"][0]),
            "rmse_by_state": res["rmse_by_state"].tolist(),
            "mae_by_state": res["mae_by_state"].tolist(),
            "rbf_sigma": predictor.sigma,
        }

    # 保存原始数据和预测结果，便于后续单独画图或做统计分析。
    np.savez(
        out_dir / "dataset_and_predictions.npz",
        train_states=train["states"],
        train_inputs=train["inputs"],
        val_states=val["states"],
        val_inputs=val["inputs"],
        **{f"{name}_pred": res["pred"] for name, res in results.items()},
        **{f"{name}_step_rmse": res["step_rmse"] for name, res in results.items()},
    )

    # 输出两类基本可视化：轨迹对比和误差增长。
    plot_rollout_comparison(val["states"], val["inputs"], results, "rollout_comparison")
    plot_error_growth(results, "error_growth")

    # 保存 JSON 摘要，记录参数、误差和运行时间。
    summary["elapsed_sec"] = time.time() - t0
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
