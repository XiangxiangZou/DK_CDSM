"""Training and deployment normalization helpers.

本模块单独存在的原因：
1. 四类模型 DKUC、DKAC、EDMD、DKN 必须使用同一份状态/控制标准化参数。
2. 后续在线部署时，真实机械臂读到的 `q,dq` 也必须用训练时保存的参数归一化。
3. 标准化参数需要可保存为 JSON，便于跨脚本、跨实验批次复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Normalizer:
    """均值-标准差标准化器。

    参数:
        mean: 每个维度的均值。
        std: 每个维度的标准差。过小的标准差会被替换成 1，避免除零。

    约定:
        状态标准化器作用于 `x=[qa,qb,dqa,dqb]`。
        控制标准化器作用于 `u=[tau_a,tau_b]`。
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        """从样本矩阵拟合标准化参数。

        参数:
            values: 形状 `(N, dim)` 的样本矩阵。
            eps: 标准差下限，小于该值的维度视为常量维度。

        返回:
            可直接用于 `transform` 和 `inverse` 的标准化器。
        """
        arr = np.asarray(values, dtype=np.float64)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    @classmethod
    def from_json(cls, payload: Dict[str, List[float]]) -> "Normalizer":
        """从 JSON 字典恢复标准化器。"""
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """把物理量转换为标准化量。"""
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values_norm: np.ndarray) -> np.ndarray:
        """把标准化量还原为物理量。"""
        return np.asarray(values_norm, dtype=np.float64) * self.std + self.mean

    def to_json(self) -> Dict[str, List[float]]:
        """转换为可写入 `normalizers.json` 的普通字典。"""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
