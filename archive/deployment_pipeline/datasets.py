"""数据集读写与形状检查工具。

本模块服务于 `run_02/run_03/run_04`，保证不同阶段读取的 `dataset.npz`
始终遵循相同结构。训练、预测和跟踪控制都以这里的检查为准。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

try:
    from .mujoco_plant import CONTROL_DIM, STATE_DIM
except ImportError:  # pragma: no cover
    from mujoco_plant import CONTROL_DIM, STATE_DIM


def validate_dataset(arrays: Dict[str, np.ndarray], source: str | Path) -> None:
    """检查 `dataset.npz` 是否包含统一训练/评估所需的形状。

    参数:
        arrays: 从 npz 读出的数组字典。
        source: 数据来源路径，用于错误提示。
    """
    if "states" not in arrays or "inputs" not in arrays:
        raise ValueError(f"{source} must contain states and inputs")
    states = np.asarray(arrays["states"])
    inputs = np.asarray(arrays["inputs"])
    if states.ndim != 3 or states.shape[2] != STATE_DIM:
        raise ValueError(f"{source}: states must be (traj, steps+1, {STATE_DIM})")
    if inputs.ndim != 3 or inputs.shape[2] != CONTROL_DIM:
        raise ValueError(f"{source}: inputs must be (traj, steps, {CONTROL_DIM})")
    if states.shape[0] != inputs.shape[0] or states.shape[1] != inputs.shape[1] + 1:
        raise ValueError(f"{source}: states and inputs trajectory lengths do not match")


def load_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    """读取 `dataset.npz` 并返回普通字典。

    参数:
        path: npz 文件路径。

    返回:
        包含 `states/inputs/q_ref/dq_ref/cable_ctrl` 等键的字典。
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    validate_dataset(arrays, dataset_path)
    return arrays


def save_dataset(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """压缩保存统一数据集。"""
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
