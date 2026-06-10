"""第二部分：离线训练 EDMD、DKUC、DKAC、DKN 四类模型。

用途：
    读取第一部分 `run_01_collect_data.py` 采集得到的 `dataset.npz`，
    训练四类模型，并把产物保存到
    `outputs/models/deployment_pipeline/<timestamp>_<tag>/`。

典型调用：
    python real_arm_deployment_pipeline/run_02_train_all_models.py ^
        --train_dataset outputs/data/deployment_pipeline/raw/xxx/dataset.npz ^
        --val_ratio 0.2 --models all

输出：
    - 共享 `normalizers.json`
    - 每类模型的 `model_config.json`
    - 神经模型权重 `best_dkuc.pt/best_dkac.pt/best_dkn.pt`
    - EDMD 字典和矩阵 `edmd_model.npz`
    - 统一训练汇总 `training_summary.json`
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

try:
    from .datasets import load_dataset
    from .model_training import MODEL_ORDER, TrainHyperParams, save_json, split_train_val, train_selected_models
except ImportError:  # pragma: no cover - direct script execution fallback
    from datasets import load_dataset
    from model_training import MODEL_ORDER, TrainHyperParams, save_json, split_train_val, train_selected_models


def build_parser() -> argparse.ArgumentParser:
    """构造离线训练命令行参数。

    参数分组：
    - 数据参数：指定训练/验证数据集，或从单个数据集中自动拆分。
    - 模型选择参数：指定训练 all 或部分模型。
    - 神经模型参数：DKUC、DKAC、DKN 共用的网络和优化器参数。
    - EDMD 参数：RBF 字典和岭回归参数。
    """
    parser = argparse.ArgumentParser(
        description="离线训练 EDMD、DKUC、DKAC、DKN 四类模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("数据与输出参数")
    data_group.add_argument(
        "--train_dataset",
        required=True,
        help="训练数据 `dataset.npz` 路径；通常来自 run_01_collect_data.py 的输出目录。",
    )
    data_group.add_argument(
        "--val_dataset",
        default="",
        help="可选验证数据 `dataset.npz` 路径；若为空，则从 train_dataset 按 val_ratio 拆分。",
    )
    data_group.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="未指定 val_dataset 时，从 train_dataset 按轨迹维度划出的验证比例。",
    )
    data_group.add_argument("--seed", type=int, default=50, help="训练、拆分和神经网络初始化随机种子。")
    data_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch 训练设备；auto 表示优先使用 CUDA，否则使用 CPU。",
    )
    data_group.add_argument(
        "--out_dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "models"
            / "deployment_pipeline"
        ),
        help="模型产物输出根目录；程序会在其下创建时间戳子目录。",
    )
    data_group.add_argument("--tag", default="", help="输出目录附加标签，便于区分实验批次。")

    model_group = parser.add_argument_group("模型选择参数")
    model_group.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="要训练的模型，支持 all 或 edmd dkuc dkac dkn 的任意组合。",
    )

    nn_group = parser.add_argument_group("DKUC/DKAC/DKN 神经模型参数")
    nn_group.add_argument("--lift_dim", type=int, default=64, help="神经 Koopman 升维特征维度。")
    nn_group.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[128, 256, 128],
        help="状态升维网络隐藏层宽度列表。",
    )
    nn_group.add_argument(
        "--control_hidden",
        type=int,
        nargs="+",
        default=[128, 128],
        help="DKAC/DKN 控制编码网络隐藏层宽度列表。",
    )
    nn_group.add_argument(
        "--control_dim_hat",
        type=int,
        default=2,
        help="DKAC/DKN 内部控制编码维度；通常保持为 2 以便与 [tau_a,tau_b] 对齐。",
    )
    nn_group.add_argument(
        "--activation",
        choices=["relu", "elu", "tanh"],
        default="elu",
        help="神经网络隐藏层激活函数。",
    )
    nn_group.add_argument(
        "--bound_lift",
        type=float,
        default=1.0,
        help="升维特征幅值限制；DKUC/DKAC 使用该数值，DKN 中大于 0 表示启用 tanh 限幅。",
    )
    nn_group.add_argument("--window", type=int, default=40, help="多步训练窗口长度，必须小于等于每条轨迹步数。")
    nn_group.add_argument("--window_start", type=int, default=4, help="curriculum 初始窗口长度。")
    nn_group.add_argument("--epochs", type=int, default=120, help="神经模型训练 epoch 数；EDMD 不使用。")
    nn_group.add_argument("--steps_per_epoch", type=int, default=150, help="每个 epoch 的 mini-batch 更新次数。")
    nn_group.add_argument("--batch_size", type=int, default=256, help="每次更新采样的窗口数量。")
    nn_group.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率。")
    nn_group.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪阈值；小于等于 0 表示不裁剪。")
    nn_group.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW 权重衰减系数。")
    nn_group.add_argument("--w_state", type=float, default=1.0, help="多步状态预测损失权重。")
    nn_group.add_argument("--w_embed", type=float, default=0.1, help="潜空间线性一致性损失权重。")

    edmd_group = parser.add_argument_group("EDMD 参数")
    edmd_group.add_argument("--edmd_centers", type=int, default=200, help="EDMD RBF 字典中心数量上限。")
    edmd_group.add_argument(
        "--edmd_sigma",
        type=float,
        default=None,
        help="EDMD RBF 宽度；不提供时由 RBF 中心间距离自动估计。",
    )
    edmd_group.add_argument(
        "--edmd_ridge",
        type=float,
        default=1e-4,
        help="EDMD 岭回归系数，作用在样本平均 Gram 矩阵上。",
    )
    edmd_group.add_argument("--edmd_seed", type=int, default=2007, help="EDMD k-means 选中心随机种子。")
    return parser


def _selected_models(raw_models: List[str]) -> List[str]:
    """解析 `--models` 参数，返回规范化模型列表。"""
    values = [item.lower() for item in raw_models]
    if "all" in values:
        return list(MODEL_ORDER)
    unknown = sorted(set(values) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"未知模型名: {unknown}; 可选 {MODEL_ORDER} 或 all")
    return values


def _make_hyper_params(args: argparse.Namespace) -> TrainHyperParams:
    """把 CLI 参数整理成训练超参数对象。"""
    return TrainHyperParams(
        lift_dim=args.lift_dim,
        hidden=tuple(args.hidden),
        control_hidden=tuple(args.control_hidden),
        control_dim_hat=args.control_dim_hat,
        activation=args.activation,
        bound_lift=args.bound_lift,
        window=args.window,
        window_start=args.window_start,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        w_state=args.w_state,
        w_embed=args.w_embed,
        edmd_centers=args.edmd_centers,
        edmd_sigma=args.edmd_sigma,
        edmd_ridge=args.edmd_ridge,
        edmd_seed=args.edmd_seed,
    )


def _make_output_dir(base_dir: str | Path, tag: str) -> Path:
    """创建本次训练的 artifacts 子目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = Path(base_dir) / f"{stamp}_train_models{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def main() -> None:
    """读取数据、训练模型并写出训练汇总。"""
    args = build_parser().parse_args()
    models = _selected_models(args.models)
    cfg = _make_hyper_params(args)
    out_dir = _make_output_dir(args.out_dir, args.tag)

    print("=== CDSM offline model training ===")
    print(f"models={models}")
    print(f"train_dataset={args.train_dataset}")
    print(f"val_dataset={args.val_dataset or '<split from train_dataset>'}")
    print(f"output={out_dir}")

    train_source = load_dataset(args.train_dataset)
    if args.val_dataset:
        train_raw = train_source
        val_raw = load_dataset(args.val_dataset)
        split_meta = {"mode": "explicit_train_val", "train_dataset": args.train_dataset, "val_dataset": args.val_dataset}
    else:
        train_raw, val_raw, split_meta = split_train_val(train_source, args.val_ratio, args.seed)

    save_json(
        out_dir / "run_config.json",
        {
            "train_dataset": args.train_dataset,
            "val_dataset": args.val_dataset,
            "split": split_meta,
            "models": models,
            "hyper_params": asdict(cfg),
        },
    )

    summary = train_selected_models(
        train_raw=train_raw,
        val_raw=val_raw,
        models=models,
        cfg=cfg,
        device_name=args.device,
        seed=args.seed,
        out_dir=out_dir,
    )
    print(f"[done] artifacts -> {out_dir}")
    print(f"[done] trained models -> {list(summary['models'].keys())}")


if __name__ == "__main__":
    main()
