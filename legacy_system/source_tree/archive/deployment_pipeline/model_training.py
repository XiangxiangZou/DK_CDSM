"""四类 Koopman 模型的离线训练适配层。

本模块不重新定义 DKUC、DKAC、EDMD、DKN 的数学模型，而是复用项目根目录
已有对比脚本中的实现。这里负责把第一部分采集得到的统一数据集转换成四类
模型都能使用的训练输入，并把产物按部署目录结构保存。

训练产物约定：
- `artifacts/<run_id>/normalizers.json`: 四类模型共享的状态/控制标准化参数。
- `artifacts/<run_id>/<model>/model_config.json`: 该模型训练配置。
- `artifacts/<run_id>/<model>/best_*.pt` 或 `edmd_model.npz`: 模型权重/字典。
- `artifacts/<run_id>/<model>/runtime_matrices.npz`: 可用于预测/控制的矩阵。
- `artifacts/<run_id>/training_summary.json`: 本次四模型训练汇总。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# 直接执行新目录脚本时，sys.path[0] 是 real_arm_deployment_pipeline。
# 这里显式加入项目根目录，确保能导入现有 DKUC/DKAC/DKN/EDMD 实现。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .normalizers import Normalizer
except ImportError:  # pragma: no cover - direct script execution fallback
    from normalizers import Normalizer

from cdsm_dkn_vs_edmd_prediction_compare import (  # noqa: E402
    CONTROL_DIM,
    STATE_DIM,
    DKNConfig,
    DKNPredictor,
    EdmdConfig,
    build_windows,
    fit_full_edmd,
    make_device,
    set_seed,
    train_dkn,
)
from cdsm_dkac_vs_edmd_tracking_control import (  # noqa: E402
    DKACConfig,
    DKACRuntime,
    EdmdRuntime,
    predict_validation_rollouts,
    train_dkac,
)
from cdsm_dkuc_vs_dkac_tracking_control import DKUCConfig, DKUCRuntime, train_dkuc  # noqa: E402


MODEL_ORDER = ("edmd", "dkuc", "dkac", "dkn")
CONTROL_CAPABLE_MODELS = ("edmd", "dkuc", "dkac")


@dataclass(frozen=True)
class TrainHyperParams:
    """四类模型共用的训练超参数。

    参数:
        lift_dim: DKUC/DKAC/DKN 的神经升维维度。
        hidden: 状态升维网络隐藏层宽度。
        control_hidden: DKAC/DKN 控制编码网络隐藏层宽度。
        control_dim_hat: DKAC/DKN 内部控制编码维度。
        activation: 神经网络激活函数，可选 `relu/elu/tanh`。
        bound_lift: 是否限制升维特征幅值；对 DKUC/DKAC 是 tanh 幅值，对 DKN 是布尔开关。
        window: 多步训练窗口长度，必须不大于每条轨迹 `steps`。
        window_start: curriculum 初始窗口长度。
        epochs: 神经模型训练 epoch 数；EDMD 不使用该参数。
        steps_per_epoch: 每个 epoch 采样 mini-batch 次数。
        batch_size: mini-batch 窗口数量。
        lr: AdamW 学习率。
        grad_clip: 梯度裁剪阈值；小于等于 0 表示不裁剪。
        weight_decay: AdamW 权重衰减。
        w_state: 多步状态预测损失权重。
        w_embed: 潜空间线性一致性损失权重。
        edmd_centers: EDMD RBF 字典中心数量上限。
        edmd_sigma: EDMD RBF 宽度；None 表示由中心距离估计。
        edmd_ridge: EDMD 岭回归系数，作用在样本平均 Gram 矩阵上。
        edmd_seed: EDMD k-means 选中心随机种子。
    """

    lift_dim: int = 64
    hidden: Tuple[int, ...] = (128, 256, 128)
    control_hidden: Tuple[int, ...] = (128, 128)
    control_dim_hat: int = CONTROL_DIM
    activation: str = "elu"
    bound_lift: float = 1.0
    window: int = 40
    window_start: int = 4
    epochs: int = 120
    steps_per_epoch: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 5.0
    weight_decay: float = 1e-5
    w_state: float = 1.0
    w_embed: float = 0.1
    edmd_centers: int = 200
    edmd_sigma: float | None = None
    edmd_ridge: float = 1e-4
    edmd_seed: int = 2007


def _jsonable(value):
    """把 numpy/Path 等对象转换为 JSON 可序列化形式。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    """读取第一部分采集得到的 `dataset.npz`。

    参数:
        path: 数据集文件路径。必须至少包含 `states` 和 `inputs`。

    返回:
        普通字典，键包括 `states/inputs/q_ref/dq_ref/cable_ctrl`。
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    validate_dataset(arrays, dataset_path)
    return arrays


def validate_dataset(arrays: Dict[str, np.ndarray], source: str | Path) -> None:
    """检查数据集形状是否符合四类模型训练约定。"""
    if "states" not in arrays or "inputs" not in arrays:
        raise ValueError(f"{source} must contain states and inputs")
    states = np.asarray(arrays["states"])
    inputs = np.asarray(arrays["inputs"])
    if states.ndim != 3 or states.shape[2] != STATE_DIM:
        raise ValueError(f"{source}: states must have shape (traj, steps+1, {STATE_DIM})")
    if inputs.ndim != 3 or inputs.shape[2] != CONTROL_DIM:
        raise ValueError(f"{source}: inputs must have shape (traj, steps, {CONTROL_DIM})")
    if inputs.shape[0] != states.shape[0] or inputs.shape[1] != states.shape[1] - 1:
        raise ValueError(f"{source}: inputs shape must match states trajectory length")


def split_train_val(
    arrays: Dict[str, np.ndarray],
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    """把单个数据集按轨迹维度拆分为 train/val。

    参数:
        arrays: `load_dataset` 返回的数据集字典。
        val_ratio: 验证集轨迹比例，通常取 0.1-0.3。
        seed: 拆分随机种子。

    返回:
        `(train_arrays, val_arrays, split_meta)`。
    """
    n_traj = arrays["states"].shape[0]
    if n_traj < 2:
        raise ValueError("At least 2 trajectories are required when val_dataset is not provided.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_traj)
    n_val = int(round(n_traj * float(val_ratio)))
    n_val = min(max(1, n_val), n_traj - 1)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    def take(indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {key: np.asarray(value)[indices].copy() for key, value in arrays.items()}

    meta = {
        "mode": "split_single_dataset",
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    return take(train_idx), take(val_idx), meta


def fit_shared_normalizers(train_raw: Dict[str, np.ndarray]) -> Tuple[Normalizer, Normalizer]:
    """拟合四类模型共享的状态和控制标准化器。

    状态标准化使用训练集中所有时刻 `states`，控制标准化使用所有 `inputs`。
    这样 DKUC、DKAC、EDMD、DKN 在完全相同的物理量尺度上训练和比较。
    """
    x_all = train_raw["states"].reshape(-1, STATE_DIM)
    u_all = train_raw["inputs"].reshape(-1, CONTROL_DIM)
    return Normalizer.fit(x_all), Normalizer.fit(u_all)


def normalize_dataset(
    arrays: Dict[str, np.ndarray],
    x_normer: Normalizer,
    u_normer: Normalizer,
) -> Dict[str, np.ndarray]:
    """把原始数据集转换为神经模型训练用的标准化轨迹。"""
    states = arrays["states"]
    inputs = arrays["inputs"]
    return {
        "states": x_normer.transform(states.reshape(-1, STATE_DIM)).reshape(states.shape),
        "inputs": u_normer.transform(inputs.reshape(-1, CONTROL_DIM)).reshape(inputs.shape),
    }


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    """用 UTF-8 保存 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def save_dataset_copy(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """把训练/验证数据副本保存到本次 artifacts 目录中，便于实验复现。"""
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def save_runtime_matrices(path: str | Path, runtime: object, control_mode: str) -> None:
    """保存后续预测/控制会重复使用的 Koopman 矩阵。

    参数:
        runtime: 具有 `A/B/C` 属性的 runtime。
        control_mode: 说明 `B` 矩阵对应的控制变量，例如 `u_norm` 或 `v=G(x)u_norm`。
    """
    np.savez_compressed(
        path,
        A=np.asarray(runtime.A, dtype=np.float64),
        B=np.asarray(runtime.B, dtype=np.float64),
        C=np.asarray(runtime.C, dtype=np.float64),
        control_mode=np.array([control_mode]),
    )


def train_edmd_model(
    train_raw: Dict[str, np.ndarray],
    x_normer: Normalizer,
    u_normer: Normalizer,
    cfg: TrainHyperParams,
    out_dir: Path,
) -> Dict[str, object]:
    """训练 EDMD 模型并保存字典/矩阵产物。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    edmd_cfg = EdmdConfig(
        n_centers=cfg.edmd_centers,
        rbf_sigma=cfg.edmd_sigma,
        ridge=cfg.edmd_ridge,
        kmeans_seed=cfg.edmd_seed,
    )
    pred = fit_full_edmd(train_raw["states"], train_raw["inputs"], x_normer, u_normer, edmd_cfg)
    runtime = EdmdRuntime(pred)
    np.savez_compressed(
        out_dir / "edmd_model.npz",
        centers=pred.centers,
        sigma=np.array([pred.sigma], dtype=np.float64),
        A=pred.A,
        B=pred.B,
        cond_number=np.array([pred.cond_number], dtype=np.float64),
        x_mean=x_normer.mean,
        x_std=x_normer.std,
        u_mean=u_normer.mean,
        u_std=u_normer.std,
    )
    save_runtime_matrices(out_dir / "runtime_matrices.npz", runtime, "z_next=A z+B u_norm")
    save_json(out_dir / "model_config.json", {"model": "EDMD", "config": asdict(edmd_cfg)})
    return {
        "artifact_dir": str(out_dir),
        "latent_dim": int(pred.latent_dim),
        "sigma": float(pred.sigma),
        "cond_number": float(pred.cond_number),
        "control_capable": True,
    }


def train_dkuc_model(
    train_norm: Dict[str, np.ndarray],
    val_norm: Dict[str, np.ndarray],
    x_normer: Normalizer,
    u_normer: Normalizer,
    cfg: TrainHyperParams,
    device,
    out_dir: Path,
) -> Dict[str, object]:
    """训练 DKUC 模型并保存权重和 runtime 矩阵。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dkuc_cfg = DKUCConfig(
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        activation=cfg.activation,
        bound_lift=cfg.bound_lift,
        window=cfg.window,
        window_start=cfg.window_start,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        grad_clip=cfg.grad_clip,
        weight_decay=cfg.weight_decay,
        w_state=cfg.w_state,
        w_embed=cfg.w_embed,
    )
    model, _history, train_info = train_dkuc(train_norm, val_norm, dkuc_cfg, device, out_dir)
    runtime = DKUCRuntime(model, x_normer, u_normer, device)
    save_runtime_matrices(out_dir / "runtime_matrices.npz", runtime, "z_next=A z+B u_norm")
    save_json(out_dir / "model_config.json", {"model": "DKUC", "config": asdict(dkuc_cfg)})
    return {
        "artifact_dir": str(out_dir),
        "latent_dim": int(model.latent_dim),
        "train_info": train_info,
        "control_capable": True,
    }


def train_dkac_model(
    train_norm: Dict[str, np.ndarray],
    val_norm: Dict[str, np.ndarray],
    x_normer: Normalizer,
    u_normer: Normalizer,
    cfg: TrainHyperParams,
    device,
    out_dir: Path,
) -> Dict[str, object]:
    """训练 DKAC 模型并保存权重和 runtime 矩阵。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dkac_cfg = DKACConfig(
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        control_hidden=tuple(cfg.control_hidden),
        control_dim_hat=cfg.control_dim_hat,
        activation=cfg.activation,
        bound_lift=cfg.bound_lift,
        identity_control_bias=True,
        window=cfg.window,
        window_start=cfg.window_start,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        grad_clip=cfg.grad_clip,
        weight_decay=cfg.weight_decay,
        w_state=cfg.w_state,
        w_embed=cfg.w_embed,
    )
    model, _history, train_info = train_dkac(train_norm, val_norm, dkac_cfg, device, out_dir)
    runtime = DKACRuntime(model, x_normer, u_normer, device)
    save_runtime_matrices(out_dir / "runtime_matrices.npz", runtime, "z_next=A z+B v, v=G(x_norm)u_norm")
    save_json(out_dir / "model_config.json", {"model": "DKAC", "config": asdict(dkac_cfg)})
    return {
        "artifact_dir": str(out_dir),
        "latent_dim": int(model.latent_dim),
        "control_dim_hat": int(model.control_dim_hat),
        "train_info": train_info,
        "control_capable": True,
    }


def train_dkn_model(
    train_raw: Dict[str, np.ndarray],
    val_raw: Dict[str, np.ndarray],
    x_normer: Normalizer,
    u_normer: Normalizer,
    cfg: TrainHyperParams,
    device,
    out_dir: Path,
) -> Dict[str, object]:
    """训练 DKN 预测模型。

    DKN 的控制编码是状态相关的非线性网络，当前只保存预测模型产物；
    后续若要用于闭环跟踪，需要单独构造 nonlinear MPC 或局部线性化控制接口。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dkn_cfg = DKNConfig(
        lift_dim=cfg.lift_dim,
        hidden=tuple(cfg.hidden),
        control_hidden=tuple(cfg.control_hidden),
        control_dim_hat=cfg.control_dim_hat,
        activation=cfg.activation,
        bound_lift=bool(cfg.bound_lift),
        window=cfg.window,
        window_start=cfg.window_start,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        grad_clip=cfg.grad_clip,
        weight_decay=cfg.weight_decay,
        w_state=cfg.w_state,
        w_embed=cfg.w_embed,
    )
    Xw_tr, Uw_tr = build_windows(train_raw["states"], train_raw["inputs"], cfg.window)
    Xw_va, Uw_va = build_windows(val_raw["states"], val_raw["inputs"], cfg.window)
    Xw_tr_n = (Xw_tr - x_normer.mean) / x_normer.std
    Xw_va_n = (Xw_va - x_normer.mean) / x_normer.std
    Uw_tr_n = (Uw_tr - u_normer.mean) / u_normer.std
    Uw_va_n = (Uw_va - u_normer.mean) / u_normer.std

    model, history, train_info = train_dkn((Xw_tr_n, Uw_tr_n), (Xw_va_n, Uw_va_n), dkn_cfg, device, out_dir)
    np.savetxt(
        out_dir / "dkn_training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,train_state,train_embed,val_total,val_state,val_embed",
        comments="",
    )
    predictor = DKNPredictor(model, x_normer, u_normer, device)
    np.savez_compressed(
        out_dir / "runtime_matrices.npz",
        A=model.A.weight.detach().cpu().numpy().astype(np.float64),
        B=model.B.weight.detach().cpu().numpy().astype(np.float64),
        C=np.hstack([np.eye(STATE_DIM), np.zeros((STATE_DIM, model.lift_dim))]),
        control_mode=np.array(["prediction_only_state_dependent_control_encoder"]),
    )
    save_json(out_dir / "model_config.json", {"model": "DKN", "config": asdict(dkn_cfg)})
    return {
        "artifact_dir": str(out_dir),
        "latent_dim": int(predictor.latent_dim),
        "control_dim_hat": int(model.control_dim_hat),
        "train_info": train_info,
        "control_capable": False,
        "control_note": "DKN is prediction-only until a nonlinear MPC or local linearization interface is added.",
    }


def evaluate_trained_rollout(runtime: object, val_raw: Dict[str, np.ndarray]) -> Dict[str, object]:
    """在验证集上做统一开环 rollout，返回可写入 summary 的指标。"""
    result = predict_validation_rollouts(runtime, val_raw)
    return {
        "total_rmse": float(result["total_rmse"]),
        "rmse_by_state": np.asarray(result["rmse_by_state"], dtype=np.float64).tolist(),
        "final_step_rmse": float(np.asarray(result["step_rmse"], dtype=np.float64)[-1]),
    }


def train_selected_models(
    *,
    train_raw: Dict[str, np.ndarray],
    val_raw: Dict[str, np.ndarray],
    models: Iterable[str],
    cfg: TrainHyperParams,
    device_name: str,
    seed: int,
    out_dir: Path,
) -> Dict[str, object]:
    """训练指定模型集合。

    参数:
        train_raw: 训练数据，形状约定同 `dataset.npz`。
        val_raw: 验证数据，形状约定同 `dataset.npz`。
        models: 要训练的模型名集合，支持 `edmd/dkuc/dkac/dkn`。
        cfg: 训练超参数。
        device_name: PyTorch 设备名，`auto/cpu/cuda`。
        seed: 全局随机种子。
        out_dir: 本次训练输出目录。

    返回:
        可写入 `training_summary.json` 的训练汇总字典。
    """
    selected = [name.lower() for name in models]
    unknown = sorted(set(selected) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    set_seed(seed)
    device = make_device(device_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_copy(out_dir / "dataset_train.npz", train_raw)
    save_dataset_copy(out_dir / "dataset_val.npz", val_raw)

    x_normer, u_normer = fit_shared_normalizers(train_raw)
    save_json(out_dir / "normalizers.json", {"x": x_normer.to_json(), "u": u_normer.to_json()})

    train_norm = normalize_dataset(train_raw, x_normer, u_normer)
    val_norm = normalize_dataset(val_raw, x_normer, u_normer)

    summary: Dict[str, object] = {
        "device": str(device),
        "seed": int(seed),
        "models_requested": selected,
        "control_capable_models": list(CONTROL_CAPABLE_MODELS),
        "state_order": ["qa", "qb", "dqa", "dqb"],
        "input_order": ["tau_a", "tau_b"],
        "train_shape": {
            "states": list(train_raw["states"].shape),
            "inputs": list(train_raw["inputs"].shape),
        },
        "val_shape": {
            "states": list(val_raw["states"].shape),
            "inputs": list(val_raw["inputs"].shape),
        },
        "hyper_params": asdict(cfg),
        "models": {},
    }

    if "edmd" in selected:
        print("[train] EDMD")
        summary["models"]["edmd"] = train_edmd_model(train_raw, x_normer, u_normer, cfg, out_dir / "edmd")

    if "dkuc" in selected:
        print("[train] DKUC")
        summary["models"]["dkuc"] = train_dkuc_model(
            train_norm, val_norm, x_normer, u_normer, cfg, device, out_dir / "dkuc"
        )

    if "dkac" in selected:
        print("[train] DKAC")
        summary["models"]["dkac"] = train_dkac_model(
            train_norm, val_norm, x_normer, u_normer, cfg, device, out_dir / "dkac"
        )

    if "dkn" in selected:
        print("[train] DKN")
        summary["models"]["dkn"] = train_dkn_model(train_raw, val_raw, x_normer, u_normer, cfg, device, out_dir / "dkn")

    save_json(out_dir / "training_summary.json", summary)
    return summary
