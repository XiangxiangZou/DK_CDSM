"""模型加载注册表。

`run_03` 和 `run_04` 都需要从同一个 artifacts 根目录加载模型。这里集中处理：
- 模型名到适配器类的映射。
- `all` 模型选择逻辑。
- 跟踪控制时只允许加载 EDMD、DKUC、DKAC。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

try:
    from .model_dkac import DKACModel
    from .model_dkn import DKNModel
    from .model_dkuc import DKUCModel
    from .model_edmd import EDMDModel
except ImportError:  # pragma: no cover
    from model_dkac import DKACModel
    from model_dkn import DKNModel
    from model_dkuc import DKUCModel
    from model_edmd import EDMDModel


PREDICTION_MODELS = ("edmd", "dkuc", "dkac", "dkn")
CONTROL_MODELS = ("edmd", "dkuc", "dkac")


def normalize_model_list(raw_models: Iterable[str], *, control_only: bool = False) -> List[str]:
    """解析模型名列表。

    参数:
        raw_models: CLI 传入的模型名，支持 `all`。
        control_only: 为 True 时，只允许 EDMD/DKUC/DKAC。

    返回:
        规范化后的小写模型名列表。
    """
    allowed = CONTROL_MODELS if control_only else PREDICTION_MODELS
    values = [item.lower() for item in raw_models]
    if "all" in values:
        return list(allowed)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown or unsupported model names: {unknown}; allowed={allowed}")
    return values


def load_prediction_model(artifact_root: str | Path, model_name: str, device: str = "cpu"):
    """加载一个预测模型。

    参数:
        artifact_root: `run_02` 输出的 artifacts 根目录。
        model_name: `edmd/dkuc/dkac/dkn`。
        device: PyTorch 推理设备，EDMD 忽略该参数。
    """
    root = Path(artifact_root)
    name = model_name.lower()
    if name == "edmd":
        return EDMDModel(root / "edmd")
    if name == "dkuc":
        return DKUCModel(root / "dkuc", root, device)
    if name == "dkac":
        return DKACModel(root / "dkac", root, device)
    if name == "dkn":
        return DKNModel(root / "dkn", root, device)
    raise ValueError(f"Unsupported model: {model_name}")


def load_control_model(artifact_root: str | Path, model_name: str, device: str = "cpu"):
    """加载一个可用于统一 Koopman LQR/MPC 的模型。"""
    name = model_name.lower()
    if name not in CONTROL_MODELS:
        raise ValueError(f"{model_name} is not supported for linear Koopman tracking control.")
    return load_prediction_model(artifact_root, name, device)
