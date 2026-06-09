"""统一模型预测评估。

本模块对 EDMD、DKUC、DKAC、DKN 使用同一套 one-step 和 rollout 指标：
- one-step: 每一步都从真实 `x_k` 升维，只评估局部一步预测误差。
- rollout: 只从真实 `x_0` 升维一次，之后纯模型递推，更接近闭环控制前的稳定性检查。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

STATE_LABELS = ("qa", "qb", "dqa", "dqb")


def predict_one_step(model, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """对单条轨迹做 one-step 预测。

    参数:
        model: 实现 `lift/step_latent/recover_state` 的预测模型。
        states: 单条真实状态轨迹，形状 `(T+1,4)`。
        inputs: 单条真实控制序列，形状 `(T,2)`。

    返回:
        预测轨迹，形状 `(T+1,4)`，第 0 步等于真实初值。
    """
    pred = np.zeros_like(states, dtype=np.float64)
    pred[0] = states[0]
    for k in range(inputs.shape[0]):
        z = model.lift(states[k])
        z_next = model.step_latent(z, inputs[k], states[k])
        pred[k + 1] = model.recover_state(z_next)
    return pred


def predict_rollout(model, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """对单条轨迹做开环 rollout 预测。"""
    return model.rollout(states[0], inputs)


def evaluate_predictions(true_states: np.ndarray, pred_states: np.ndarray) -> Dict[str, object]:
    """计算统一预测误差指标。

    指标:
        total_rmse: 除初值外所有状态维度的总体 RMSE。
        rmse_by_state: 除初值外每个状态维度的 RMSE。
        step_rmse: 每个时间步跨状态维度和轨迹平均的 RMSE，含 t=0。
        final_step_rmse: 最后一步 RMSE，反映长程误差积累。
    """
    err = np.asarray(pred_states, dtype=np.float64) - np.asarray(true_states, dtype=np.float64)
    err_no_init = err[:, 1:, :]
    return {
        "total_rmse": float(np.sqrt(np.mean(err_no_init * err_no_init))),
        "rmse_by_state": np.sqrt(np.mean(err_no_init * err_no_init, axis=(0, 1))).tolist(),
        "step_rmse": np.sqrt(np.mean(err * err, axis=(0, 2))).tolist(),
        "final_step_rmse": float(np.sqrt(np.mean(err[:, -1, :] * err[:, -1, :]))),
        "state_labels": list(STATE_LABELS),
    }


def evaluate_model(model, dataset: Dict[str, np.ndarray], mode: str) -> Dict[str, object]:
    """在验证数据集上评估单个模型。

    参数:
        model: 预测模型。
        dataset: 包含 `states/inputs` 的验证数据。
        mode: `one_step` 或 `rollout`。
    """
    states = np.asarray(dataset["states"], dtype=np.float64)
    inputs = np.asarray(dataset["inputs"], dtype=np.float64)
    preds = np.zeros_like(states)
    for i in range(states.shape[0]):
        if mode == "one_step":
            preds[i] = predict_one_step(model, states[i], inputs[i])
        elif mode == "rollout":
            preds[i] = predict_rollout(model, states[i], inputs[i])
        else:
            raise ValueError(f"Unknown prediction mode: {mode}")
    metrics = evaluate_predictions(states, preds)
    return {"preds": preds, "states_true": states, "metrics": metrics}
