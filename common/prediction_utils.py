"""Koopman 预测脚本的共享工具模块。

本模块提供 Koopman 算子模型（EDMD/DKUC/DKAC/DKN）在训练和评估阶段
共用的基础工具，包括：

    - 数据归一化器（z-score 标准化）
    - JSON/NPZ 文件读写
    - 数据集验证、加载、保存与训练/验证集划分
    - 随机种子与计算设备管理
    - 滑动窗口构建（用于构造训练样本）
    - 模型预测与评估指标计算
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 数据格式常量
# ---------------------------------------------------------------------------

# 状态向量的字段顺序：[关节a位置, 关节b位置, 关节a速度, 关节b速度]
STATE_ORDER = ("qa", "qb", "dqa", "dqb")

# 控制输入向量的字段顺序：[关节a力矩, 关节b力矩]
INPUT_ORDER = ("tau_a", "tau_b")

# Koopman 模型类型的标识符列表
# edmd: Extended Dynamic Mode Decomposition（扩展动态模态分解）
# dkuc: Deep Koopman with Unconstrained Control（无约束控制的深度 Koopman）
# dkac: Deep Koopman with Affine Control（仿射控制的深度 Koopman）
# dkn:  Deep Koopman Network（通用深度 Koopman 网络）
MODEL_ORDER = ("edmd", "dkuc", "dkac", "dkn")


# ---------------------------------------------------------------------------
# 数据归一化器
# ---------------------------------------------------------------------------


@dataclass
class Normalizer:
    """z-score 标准化器：将数据变换为均值为 0、标准差为 1 的分布。

    采用 (x - mean) / std 的经典 z-score 归一化公式。标准差过小的
    维度会被 clamp 到 1.0，避免除以接近零的值导致数值不稳定。

    参数
    ----------
    mean : np.ndarray
        每个特征的均值，形状 (n_features,)。
    std : np.ndarray
        每个特征的标准差，形状 (n_features,)。
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        """从数据中估计均值和标准差，创建 Normalizer 实例。

        参数
        ----------
        values : np.ndarray
            形状为 (n_samples, n_features) 的原始数据。
        eps : float
            标准差下限阈值。当某维度标准差小于此值时，将其设为 1.0，
            避免归一化时除以极小值。

        返回
        -------
        Normalizer
            拟合好的归一化器。
        """
        arr = np.asarray(values, dtype=np.float64)
        mean = arr.mean(axis=0)  # 沿样本轴求均值
        std = arr.std(axis=0)    # 沿样本轴求标准差
        # 将接近常数的维度标准差 clamp 到 1.0，防止除以零
        std = np.where(std < eps, 1.0, std)
        return cls(mean.astype(np.float64), std.astype(np.float64))

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "Normalizer":
        """从 JSON 字典反序列化 Normalizer。

        参数
        ----------
        payload : dict
            包含 "mean" 和 "std" 键的字典。

        返回
        -------
        Normalizer
            反序列化后的归一化器。
        """
        return cls(
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["std"], dtype=np.float64),
        )

    def to_json(self) -> dict[str, list[float]]:
        """将 Normalizer 序列化为 JSON 兼容的字典。

        返回
        -------
        dict
            包含 "mean" 和 "std" 列表的字典。
        """
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def transform(self, values: np.ndarray) -> np.ndarray:
        """正向归一化：将原始数据变换到 z-score 空间。

        参数
        ----------
        values : np.ndarray
            原始数据。

        返回
        -------
        np.ndarray
            归一化后的数据，均值为 0，标准差为 1。
        """
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values_norm: np.ndarray) -> np.ndarray:
        """逆归一化：将 z-score 空间的数据还原到原始空间。

        参数
        ----------
        values_norm : np.ndarray
            归一化后的数据。

        返回
        -------
        np.ndarray
            还原到原始尺度的数据。
        """
        return np.asarray(values_norm, dtype=np.float64) * self.std + self.mean


# ---------------------------------------------------------------------------
# JSON 序列化工具
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """递归地将 Python 对象转换为 JSON 可序列化的形式。

    处理 NumPy 数组、Path 对象、NumPy 标量等 json.dumps 无法直接
    序列化的类型。

    参数
    ----------
    value : Any
        待转换的 Python 对象。

    返回
    -------
    Any
        JSON 兼容的等价对象。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """将字典保存为格式化的 JSON 文件。

    自动创建父目录，支持 NumPy 类型和 Path 对象的序列化。

    参数
    ----------
    path : str | Path
        输出文件路径。
    payload : dict
        待保存的字典。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """从 JSON 文件加载字典。

    参数
    ----------
    path : str | Path
        JSON 文件路径。

    返回
    -------
    dict
        解析后的字典。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 数据集选择表
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ROOT = PROJECT_ROOT / "prediction"
DEFAULT_DATASET_CONFIG = PREDICTION_ROOT / "dataset_selections.json"


def _project_path(value: str | Path) -> Path:
    """Resolve relative artifact paths from the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset_selection(
    *,
    method: str,
    train_dataset: str | Path = "",
    val_dataset: str | Path = "",
    dataset_key: str = "",
    dataset_config: str | Path = DEFAULT_DATASET_CONFIG,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve which dataset a prediction method should train on.

    Explicit ``--train_dataset`` remains supported and has highest priority.
    If it is omitted, ``dataset_key`` is resolved from ``dataset_config``.
    When ``dataset_key`` is omitted too, the method-specific default is used.
    """
    if str(train_dataset):
        return (
            str(_project_path(train_dataset)),
            str(_project_path(val_dataset)) if str(val_dataset) else "",
            {
                "source": "cli",
                "method": method,
                "dataset_key": "",
                "train_dataset": str(train_dataset),
                "val_dataset": str(val_dataset) if str(val_dataset) else "",
            },
        )

    config_path = _project_path(dataset_config)
    payload = load_json(config_path)
    method_name = method.lower()
    key = dataset_key or payload.get("method_defaults", {}).get(method_name, "")
    if not key:
        raise ValueError(
            f"No dataset_key provided and no default configured for method '{method_name}'"
        )
    datasets = payload.get("datasets", {})
    if key not in datasets:
        raise ValueError(f"Dataset key '{key}' is not defined in {config_path}")
    entry = dict(datasets[key])
    train_value = entry.get("train_dataset") or entry.get("dataset_path")
    if not train_value:
        raise ValueError(f"Dataset key '{key}' has no train_dataset")
    val_value = entry.get("val_dataset", "")
    return (
        str(_project_path(train_value)),
        str(_project_path(val_value)) if val_value else "",
        {
            "source": "dataset_config",
            "method": method_name,
            "dataset_config": str(config_path),
            "dataset_key": key,
            "entry": entry,
        },
    )


# ---------------------------------------------------------------------------
# 数据集读写与验证
# ---------------------------------------------------------------------------


def validate_dataset(arrays: dict[str, np.ndarray]) -> None:
    """验证数据集的形状和一致性。

    数据集必须包含两个键：
    - "states": 形状 (n_traj, n_steps+1, 4)，即每条轨迹包含 steps+1 个状态帧
      （因为需要包含初始状态和每步执行后的状态）
    - "inputs": 形状 (n_traj, n_steps, 2)，即每条轨迹包含 steps 个控制输入
      （每步执行前施加一个控制量）

    轨迹数、步数在两个数组间必须匹配：states 比 inputs 多一帧（初始状态）。

    参数
    ----------
    arrays : dict[str, np.ndarray]
        包含 "states" 与 "inputs" 的字典。

    抛出
    ------
    ValueError
        如果形状不正确或轨迹/步数维度不匹配。
    """
    states = np.asarray(arrays["states"])
    inputs = np.asarray(arrays["inputs"])
    if states.ndim != 3 or states.shape[-1] != 4:
        raise ValueError("states 形状必须为 (traj, steps+1, 4)，即 (轨迹数, 步数+1, 4维状态)")
    if inputs.ndim != 3 or inputs.shape[-1] != 2:
        raise ValueError("inputs 形状必须为 (traj, steps, 2)，即 (轨迹数, 步数, 2维输入)")
    if states.shape[0] != inputs.shape[0] or states.shape[1] != inputs.shape[1] + 1:
        raise ValueError("states 与 inputs 的轨迹数或步数维度不匹配")


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """从压缩的 NPZ 文件加载数据集。

    参数
    ----------
    path : str | Path
        NPZ 文件路径。

    返回
    -------
    dict[str, np.ndarray]
        加载并验证通过的数据集字典。
    """
    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    validate_dataset(arrays)
    return arrays


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """将数据集保存为压缩的 NPZ 文件。

    保存前会先验证数据格式，自动创建父目录。

    参数
    ----------
    path : str | Path
        输出文件路径（建议使用 .npz 扩展名）。
    arrays : dict[str, np.ndarray]
        包含 "states" 与 "inputs" 的字典。
    """
    validate_dataset(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: np.asarray(value) for key, value in arrays.items()})


# ---------------------------------------------------------------------------
# 训练/验证集划分
# ---------------------------------------------------------------------------


def split_train_val(
    arrays: dict[str, np.ndarray],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """将数据集按轨迹维度随机划分为训练集和验证集。

    划分在轨迹级别进行（而非帧级别），确保同一条轨迹的所有帧
    只出现在训练集或验证集中的一个，避免数据泄漏。

    参数
    ----------
    arrays : dict[str, np.ndarray]
        完整数据集，包含 "states" 和 "inputs"。
    val_ratio : float
        验证集所占比例，取值 (0, 1)。
    seed : int
        随机种子，保证划分可复现。

    返回
    -------
    tuple[dict, dict, dict]
        (train_data, val_data, split_meta)：
        - train_data: 训练集字典
        - val_data:   验证集字典
        - split_meta: 划分元信息（模式、比例、种子、索引等）
    """
    validate_dataset(arrays)
    n_traj = arrays["states"].shape[0]
    if n_traj < 2:
        raise ValueError("至少需要两条轨迹才能划分训练/验证集")
    # 使用 RandomState 保证可复现性（不受全局随机状态影响）
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_traj)
    # 验证集至少 1 条，至多 n_traj-1 条
    n_val = min(max(1, int(round(n_traj * val_ratio))), n_traj - 1)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    def take(indices: np.ndarray) -> dict[str, np.ndarray]:
        """按索引提取子集并返回独立副本。"""
        return {key: np.asarray(value)[indices].copy() for key, value in arrays.items()}

    return (
        take(train_idx),
        take(val_idx),
        {
            "mode": "split_single_dataset",      # 划分模式：从单个数据集切分
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# 归一化器的持久化
# ---------------------------------------------------------------------------


def save_normalizers(path: str | Path, x_normer: Normalizer, u_normer: Normalizer) -> None:
    """将状态归一化器和输入归一化器保存为单个 JSON 文件。

    参数
    ----------
    path : str | Path
        输出 JSON 文件路径。
    x_normer : Normalizer
        状态数据的归一化器。
    u_normer : Normalizer
        控制输入的归一化器。
    """
    save_json(path, {"x": x_normer.to_json(), "u": u_normer.to_json()})


def load_normalizers(path: str | Path) -> tuple[Normalizer, Normalizer]:
    """从 JSON 文件加载状态和输入归一化器。

    参数
    ----------
    path : str | Path
        归一化器 JSON 文件路径。

    返回
    -------
    tuple[Normalizer, Normalizer]
        (x_normer, u_normer)：状态归一化器和输入归一化器。
    """
    payload = load_json(path)
    return Normalizer.from_json(payload["x"]), Normalizer.from_json(payload["u"])


# ---------------------------------------------------------------------------
# 随机种子与计算设备
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """固定所有相关库的随机种子，确保实验可复现。

    依次设置 Python 内置 random、NumPy 和 PyTorch（如果可用）的种子。
    对于 CUDA，同时设置所有 GPU 的种子。

    参数
    ----------
    seed : int
        随机种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 设备的种子
    except ModuleNotFoundError:
        pass  # PyTorch 未安装时静默跳过


def make_device(name: str):
    """解析并返回 PyTorch 设备对象。

    参数
    ----------
    name : str
        设备名称。支持 "auto"（自动选择 GPU 或 CPU）、"cpu"、"cuda"、
        "cuda:0" 等 PyTorch 设备标识。

    返回
    -------
    torch.device
        PyTorch 设备对象。
    """
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# 输出目录管理
# ---------------------------------------------------------------------------


def make_output_dir(base: str | Path, method: str, tag: str = "") -> Path:
    """创建带时间戳的实验输出目录。

    目录命名格式：{时间戳}_{方法名}_prediction[{_标签}]
    例如：20260101_143000_dkac_prediction_testrun

    参数
    ----------
    base : str | Path
        输出根目录。
    method : str
        方法/模型名称（如 "edmd", "dkac"）。
    tag : str
        可选的附加标签，用于区分同一次运行的子实验。

    返回
    -------
    Path
        创建的输出目录路径。
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 精确到秒的时间戳
    suffix = f"_{tag}" if tag else ""
    output = Path(base) / f"{stamp}_{method}_prediction{suffix}"
    output.mkdir(parents=True, exist_ok=False)  # exist_ok=False 防止覆盖已有结果
    return output


@dataclass(frozen=True)
class PredictionRunPaths:
    """一次 prediction 运行的分类输出目录。"""

    run_id: str
    root_dir: Path
    artifact_dir: Path
    figures_dir: Path


def prediction_output_base(method: str, run_type: str = "full_run") -> Path:
    """返回 prediction 文件夹内部的分类输出根目录。"""
    del method
    if run_type not in {"full_run", "smoke_test"}:
        raise ValueError("run_type must be 'full_run' or 'smoke_test'")
    return PREDICTION_ROOT / "outputs" / run_type


def create_prediction_run_paths(
    method: str,
    run_type: str = "full_run",
    tag: str = "",
    out_dir: str | Path | None = None,
) -> PredictionRunPaths:
    """按运行类型创建模型/指标/数组目录和图片目录。

    非图片类结果放在：
        <run_type>/<method>/<timestamp>_<method>[_tag]/
    图片、PDF、GIF 等展示类结果放在：
        <run_type>/figures/<timestamp>_<method>[_tag]/
    """
    root = Path(out_dir) if out_dir else prediction_output_base(method, run_type)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    run_id = f"{stamp}_{method}{suffix}"
    artifact_dir = root / method / run_id
    figures_dir = root / "figures" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)
    return PredictionRunPaths(
        run_id=run_id,
        root_dir=root,
        artifact_dir=artifact_dir,
        figures_dir=figures_dir,
    )


# ---------------------------------------------------------------------------
# 训练数据加载
# ---------------------------------------------------------------------------


def load_train_val(
    train_dataset: str | Path,
    val_dataset: str | Path | None,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """加载训练和验证数据，支持两种模式。

    模式1 — 显式指定验证集：val_dataset 不为 None 时，直接使用指定的
    验证集文件，训练集和验证集各自独立加载。
    模式2 — 自动切分：val_dataset 为 None 时，从训练集文件中按比例
    随机切分出验证集。

    参数
    ----------
    train_dataset : str | Path
        训练数据集 NPZ 文件路径。
    val_dataset : str | Path | None
        验证数据集 NPZ 文件路径，为 None 时自动从训练集切分。
    val_ratio : float
        自动切分模式下验证集所占比例。
    seed : int
        自动切分模式下的随机种子。

    返回
    -------
    tuple[dict, dict, dict]
        (train_data, val_data, meta)：训练数据、验证数据和划分元信息。
    """
    source = load_dataset(train_dataset)
    if val_dataset:
        # 模式1：使用独立的验证集文件
        return (
            source,
            load_dataset(val_dataset),
            {
                "mode": "explicit_train_val",
                "train_dataset": str(train_dataset),
                "val_dataset": str(val_dataset),
            },
        )
    # 模式2：从训练集随机切分
    train_data, val_data, split_meta = split_train_val(source, val_ratio, seed)
    split_meta["train_dataset"] = str(train_dataset)
    return train_data, val_data, split_meta


def fit_state_input_normalizers(train_data: dict[str, np.ndarray]) -> tuple[Normalizer, Normalizer]:
    """在训练数据上拟合状态和输入的归一化器。

    将所有轨迹的数据展平为二维数组后分别拟合，即对整个训练集的
    所有帧（跨轨迹、跨时间步）统计全局均值和标准差。

    参数
    ----------
    train_data : dict[str, np.ndarray]
        训练数据字典，包含 "states" (n_traj, steps+1, 4) 和
        "inputs" (n_traj, steps, 2)。

    返回
    -------
    tuple[Normalizer, Normalizer]
        (x_normer, u_normer)：状态归一化器和输入归一化器。
    """
    # 将 (n_traj, n_steps, n_features) 展平为 (n_traj * n_steps, n_features)
    x_normer = Normalizer.fit(train_data["states"].reshape(-1, train_data["states"].shape[-1]))
    u_normer = Normalizer.fit(train_data["inputs"].reshape(-1, train_data["inputs"].shape[-1]))
    return x_normer, u_normer


# ---------------------------------------------------------------------------
# 滑动窗口构造
# ---------------------------------------------------------------------------


def build_windows(
    states: np.ndarray,
    inputs: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将轨迹数据切分为固定长度的滑动窗口样本。

    对每条轨迹，以滑动窗口方式提取连续 window 步的数据片段。每个窗口
    包含 window+1 个状态帧和 window 个控制输入帧。

    例如，对于一条 100 步的轨迹，window=10 时会产生 91 个窗口样本。
    窗口结构示意（以 window=3 为例）：

        x 窗口: [s0, s1, s2, s3]   ← 4 个状态帧（起始 + 3 步后）
        u 窗口: [u0, u1, u2]       ← 3 个控制输入

    参数
    ----------
    states : np.ndarray
        形状 (n_traj, n_steps+1, n_state_features) 的状态数组。
    inputs : np.ndarray
        形状 (n_traj, n_steps, n_input_features) 的控制输入数组。
    window : int
        窗口长度（步数），必须 ≥ 1 且不超过每条轨迹的步数。

    返回
    -------
    tuple[np.ndarray, np.ndarray]
        (x_windows, u_windows)：
        - x_windows: 形状 (total_windows, window+1, n_state_features)
        - u_windows: 形状 (total_windows, window, n_input_features)
    """
    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(inputs, dtype=np.float64)
    if u.shape[1] != x.shape[1] - 1:
        raise ValueError("inputs 步数必须比 states 少一步（inputs 不含初始状态帧）")
    if window < 1 or window > u.shape[1]:
        raise ValueError(f"window 必须在 [1, {u.shape[1]}] 范围内，当前值为 {window}")
    # 每条轨迹可产生的窗口数
    count = u.shape[1] - window + 1
    # 预分配输出数组
    x_windows = np.empty((x.shape[0] * count, window + 1, x.shape[2]), dtype=np.float64)
    u_windows = np.empty((x.shape[0] * count, window, u.shape[2]), dtype=np.float64)
    cursor = 0
    # 逐轨迹、逐起始位置滑动提取窗口
    for traj in range(x.shape[0]):
        for start in range(count):
            x_windows[cursor] = x[traj, start : start + window + 1]
            u_windows[cursor] = u[traj, start : start + window]
            cursor += 1
    return x_windows, u_windows


# ---------------------------------------------------------------------------
# 预测评估指标
# ---------------------------------------------------------------------------


def evaluate_predictions(true_states: np.ndarray, pred_states: np.ndarray) -> dict[str, Any]:
    """计算预测结果的多维度误差指标。

    计算以下指标：
    - total_rmse: 整体 RMSE（排除初始帧，因为初始状态是给定的）
    - rmse_by_state: 每个状态维度分别的 RMSE [qa, qb, dqa, dqb]
    - step_rmse: 每个预测步的 RMSE（含初始帧）
    - final_step_rmse: 最终预测步的 RMSE

    参数
    ----------
    true_states : np.ndarray
        真实状态，形状 (n_traj, n_steps+1, 4)。
    pred_states : np.ndarray
        预测状态，形状同上。

    返回
    -------
    dict
        各项评估指标的字典。
    """
    err = np.asarray(pred_states, dtype=np.float64) - np.asarray(true_states, dtype=np.float64)
    # 排除初始帧（第0步是给定的，无需预测），计算开环预测误差
    err_no_init = err[:, 1:, :]
    return {
        "total_rmse": float(np.sqrt(np.mean(err_no_init * err_no_init))),
        "rmse_by_state": np.sqrt(np.mean(err_no_init * err_no_init, axis=(0, 1))).tolist(),
        "step_rmse": np.sqrt(np.mean(err * err, axis=(0, 2))).tolist(),
        "final_step_rmse": float(np.sqrt(np.mean(err[:, -1, :] * err[:, -1, :]))),
        "state_labels": list(STATE_ORDER),
    }


# ---------------------------------------------------------------------------
# 模型预测与评估
# ---------------------------------------------------------------------------


def predict_one_step(model, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """单步递推预测：每步使用真实状态作为起点，仅向前预测一步。

    这是"一步预测"模式：在每步 k，以真实状态 x_k 为起点，用模型预测
    下一帧状态 x_{k+1} = f(x_k, u_k)。然后将预测的 x_{k+1} 作为下一步
    的起点，以此类推。注意这里虽然每一步都用真实状态作为 lift 的输入，
    但递推过程中使用的是前一帧的预测结果作为 recover_state 的输入——
    实际上这是一个开环 rollout。

    参数
    ----------
    model : Koopman 模型对象
        必须实现 lift(state), step_latent(z, u, x), recover_state(z) 三个方法。
    states : np.ndarray
        整条轨迹的真实状态，形状 (n_steps+1, n_state_features)。
    inputs : np.ndarray
        整条轨迹的控制输入，形状 (n_steps, n_input_features)。

    返回
    -------
    np.ndarray
        预测的状态轨迹，形状 (n_steps+1, n_state_features)。
    """
    pred = np.zeros_like(states, dtype=np.float64)
    pred[0] = states[0]  # 初始状态直接复制
    for k in range(inputs.shape[0]):
        # 将当前状态提升到 Koopman 隐空间
        z = model.lift(states[k])
        # 在隐空间中执行一步演化
        z_next = model.step_latent(z, inputs[k], states[k])
        # 将隐空间状态还原到原始状态空间
        pred[k + 1] = model.recover_state(z_next)
    return pred


def evaluate_model(model, dataset: dict[str, np.ndarray], mode: str) -> dict[str, Any]:
    """在数据集上评估 Koopman 模型的预测性能。

    支持两种评估模式：
    - "one_step": 单步递推预测（每步使用上一帧预测结果作为起点）
    - "rollout":  调用模型的 rollout 方法进行多步开环预测

    参数
    ----------
    model : Koopman 模型对象
        待评估的模型。
    dataset : dict[str, np.ndarray]
        评估数据集，包含 "states" 和 "inputs"。
    mode : str
        预测模式："one_step" 或 "rollout"。

    返回
    -------
    dict
        包含 "preds"（预测状态）、"states_true"（真实状态）和
        "metrics"（评估指标）的字典。
    """
    states = np.asarray(dataset["states"], dtype=np.float64)
    inputs = np.asarray(dataset["inputs"], dtype=np.float64)
    preds = np.zeros_like(states)
    for i in range(states.shape[0]):
        if mode == "one_step":
            preds[i] = predict_one_step(model, states[i], inputs[i])
        elif mode == "rollout":
            preds[i] = model.rollout(states[i, 0], inputs[i])
        else:
            raise ValueError(f"未知的预测模式: {mode}")
    return {"preds": preds, "states_true": states, "metrics": evaluate_predictions(states, preds)}


def plot_prediction_states(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    figures_dir: str | Path,
    stem: str,
    traj_index: int = 0,
) -> list[str]:
    """将一条验证轨迹的真实/预测状态曲线保存为 PNG 和 PDF。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 仅在绘图库缺失时触发
        return [f"plot skipped: {exc}"]

    true_arr = np.asarray(true_states, dtype=np.float64)
    pred_arr = np.asarray(pred_states, dtype=np.float64)
    idx = min(max(0, traj_index), true_arr.shape[0] - 1)
    steps = np.arange(true_arr.shape[1])
    target = Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes_flat = axes.reshape(-1)
    for dim, label in enumerate(STATE_ORDER):
        ax = axes_flat[dim]
        ax.plot(steps, true_arr[idx, :, dim], label="true", linewidth=1.8)
        ax.plot(steps, pred_arr[idx, :, dim], label="pred", linewidth=1.4, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    axes_flat[-2].set_xlabel("step")
    axes_flat[-1].set_xlabel("step")
    axes_flat[0].legend(loc="best")
    fig.suptitle(f"{stem} trajectory {idx}")
    fig.tight_layout()

    paths = []
    for suffix in ("png", "pdf"):
        path = target / f"{stem}_states.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_prediction_errors(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    figures_dir: str | Path,
    stem: str,
    traj_index: int = 0,
) -> list[str]:
    """Save prediction-error curves and step-wise RMSE as PNG/PDF."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [f"plot skipped: {exc}"]

    true_arr = np.asarray(true_states, dtype=np.float64)
    pred_arr = np.asarray(pred_states, dtype=np.float64)
    idx = min(max(0, traj_index), true_arr.shape[0] - 1)
    err = pred_arr - true_arr
    steps = np.arange(true_arr.shape[1])
    step_rmse = np.sqrt(np.mean(err * err, axis=(0, 2)))
    target = Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for dim, label in enumerate(STATE_ORDER):
        axes[0].plot(steps, err[idx, :, dim], linewidth=1.2, label=label)
    axes[0].set_ylabel("state error")
    axes[0].grid(True, linewidth=0.4, alpha=0.5)
    axes[0].legend(loc="best", ncol=2)

    q_err = np.linalg.norm(err[idx, :, :2], axis=1)
    dq_err = np.linalg.norm(err[idx, :, 2:], axis=1)
    axes[1].plot(steps, q_err, label="q error norm")
    axes[1].plot(steps, dq_err, label="dq error norm")
    axes[1].set_ylabel("norm")
    axes[1].grid(True, linewidth=0.4, alpha=0.5)
    axes[1].legend(loc="best")

    axes[2].plot(steps, step_rmse, color="tab:red", label="all-state RMSE")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(True, linewidth=0.4, alpha=0.5)
    axes[2].legend(loc="best")
    fig.suptitle(f"{stem} prediction error")
    fig.tight_layout()

    paths = []
    for suffix in ("png", "pdf"):
        path = target / f"{stem}_prediction_error.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def save_prediction_outputs(
    model,
    dataset: dict[str, np.ndarray],
    artifact_dir: str | Path,
    figures_dir: str | Path,
    mode: str,
) -> dict[str, Any]:
    """保存预测数组、指标和状态曲线图。"""
    result = evaluate_model(model, dataset, mode)
    artifact = Path(artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact / f"{mode}_prediction_rollouts.npz",
        states_true=result["states_true"],
        states_pred=result["preds"],
    )
    figure_paths = plot_prediction_states(result["states_true"], result["preds"], figures_dir, mode)
    figure_paths += plot_prediction_errors(result["states_true"], result["preds"], figures_dir, mode)
    metrics = dict(result["metrics"])
    metrics["figure_paths"] = figure_paths
    save_json(artifact / f"{mode}_metrics.json", metrics)
    return metrics
