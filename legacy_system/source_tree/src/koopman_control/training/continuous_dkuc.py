"""Continuous-time DKUC training for Yu-Tan-style KILC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from koopman_control.data.artifacts import save_json, save_runtime_matrices
from koopman_control.data.normalization import Normalizer
from koopman_control.models.networks import ContinuousDKUCNetwork


@dataclass(frozen=True)
class ContinuousTrainingArrays:
    x_norm: np.ndarray
    u_norm: np.ndarray
    xdot_norm: np.ndarray
    xdot_phys: np.ndarray
    x_next_norm: np.ndarray


@dataclass(frozen=True)
class ContinuousDKUCTrainingConfig:
    lift_dim: int = 64
    hidden: tuple[int, ...] = (128, 256, 128)
    activation: str = "elu"
    bound_lift: float = 1.0
    epochs: int = 120
    steps_per_epoch: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 5.0
    weight_decay: float = 1e-5
    w_state_derivative: float = 1.0
    w_lift_derivative: float = 0.1
    w_readout: float = 0.1
    w_rollout: float = 0.2
    rollout_horizon: int = 10


def build_continuous_training_arrays(
    states: np.ndarray,
    inputs: np.ndarray,
    *,
    dt: float,
    x_normer: Normalizer,
    u_normer: Normalizer,
) -> ContinuousTrainingArrays:
    """Flatten trajectories and build finite-difference derivative targets."""
    states = np.asarray(states, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64)
    if states.ndim != 3 or inputs.ndim != 3:
        raise ValueError("states and inputs must be trajectory arrays")
    if states.shape[0] != inputs.shape[0] or states.shape[1] != inputs.shape[1] + 1:
        raise ValueError("states must have one more step than inputs")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    x = states[:, :-1, :].reshape(-1, states.shape[-1])
    x_next = states[:, 1:, :].reshape(-1, states.shape[-1])
    u = inputs.reshape(-1, inputs.shape[-1])
    xdot_phys = (x_next - x) / float(dt)
    x_norm = x_normer.transform(x)
    x_next_norm = x_normer.transform(x_next)
    u_norm = u_normer.transform(u)
    xdot_norm = xdot_phys / x_normer.std.reshape(1, -1)
    return ContinuousTrainingArrays(
        x_norm=x_norm.astype(np.float64),
        u_norm=u_norm.astype(np.float64),
        xdot_norm=xdot_norm.astype(np.float64),
        xdot_phys=xdot_phys.astype(np.float64),
        x_next_norm=x_next_norm.astype(np.float64),
    )


def _sample_indices(count: int, batch_size: int) -> np.ndarray:
    return np.random.randint(0, count, size=min(int(batch_size), count))


def fit_continuous_dkuc(
    *,
    train_states: np.ndarray,
    train_inputs: np.ndarray,
    val_states: np.ndarray,
    val_inputs: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    dt: float,
    config: ContinuousDKUCTrainingConfig,
    output_dir: str | Path,
    device,
) -> dict[str, object]:
    """Train a continuous-time DKUC model and write runtime artifacts."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = build_continuous_training_arrays(
        train_states,
        train_inputs,
        dt=dt,
        x_normer=x_normer,
        u_normer=u_normer,
    )
    val = build_continuous_training_arrays(
        val_states,
        val_inputs,
        dt=dt,
        x_normer=x_normer,
        u_normer=u_normer,
    )
    model = ContinuousDKUCNetwork(
        lift_dim=config.lift_dim,
        hidden=config.hidden,
        activation=config.activation,
        bound_lift=config.bound_lift,
        state_dim=train_states.shape[-1],
        control_dim=train_inputs.shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    best_value = float("inf")
    best_epoch = 0
    checkpoint_path = output / "best_dkuc_continuous.pt"
    history: list[list[float]] = []

    def tensors(arrays: ContinuousTrainingArrays, idx: np.ndarray):
        return (
            torch.from_numpy(arrays.x_norm[idx].astype(np.float32)).to(device),
            torch.from_numpy(arrays.u_norm[idx].astype(np.float32)).to(device),
            torch.from_numpy(arrays.xdot_norm[idx].astype(np.float32)).to(device),
            torch.from_numpy(arrays.x_next_norm[idx].astype(np.float32)).to(device),
        )

    def loss_for(arrays: ContinuousTrainingArrays, idx: np.ndarray):
        x, u, xdot, x_next = tensors(arrays, idx)
        z = model.lift(x)
        z_next_true = model.lift(x_next)
        zdot = model.derivative(z, u)
        state_derivative = torch.mean(
            (model.state_from_latent(zdot) - xdot) ** 2
        )
        lifted_derivative = torch.mean(
            (zdot - (z_next_true - z) / float(dt)) ** 2
        )
        readout = torch.mean((model.state_from_latent(z) - x) ** 2)
        z_rollout = model.euler_step(z, u, float(dt))
        rollout = torch.mean((model.state_from_latent(z_rollout) - x_next) ** 2)
        total = (
            config.w_state_derivative * state_derivative
            + config.w_lift_derivative * lifted_derivative
            + config.w_readout * readout
            + config.w_rollout * rollout
        )
        return total, state_derivative, lifted_derivative, readout, rollout

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        for _ in range(config.steps_per_epoch):
            idx = _sample_indices(train.x_norm.shape[0], config.batch_size)
            total, *_ = loss_for(train, idx)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
            optimizer.step()
            train_total += float(total.detach().item())
        train_total /= max(config.steps_per_epoch, 1)

        model.eval()
        with torch.no_grad():
            val_idx = _sample_indices(val.x_norm.shape[0], config.batch_size)
            losses = loss_for(val, val_idx)
        val_total = float(losses[0].item())
        history.append(
            [
                float(epoch),
                train_total,
                val_total,
                float(losses[1].item()),
                float(losses[2].item()),
                float(losses[3].item()),
                float(losses[4].item()),
            ]
        )
        if val_total < best_value:
            best_value = val_total
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    A = model.A.weight.detach().cpu().numpy().astype(np.float64)
    B = model.B.weight.detach().cpu().numpy().astype(np.float64)
    C = np.zeros((train_states.shape[-1], model.latent_dim), dtype=np.float64)
    C[:, : train_states.shape[-1]] = np.eye(train_states.shape[-1])
    save_runtime_matrices(
        output / "runtime_matrices.npz",
        A=A,
        B=B,
        C=C,
        control_mode="zdot=A_c z+B_c u_norm",
    )
    save_json(
        output / "model_config.json",
        {
            "model": "DKUC_CONTINUOUS",
            "config": asdict(config),
            "dt": float(dt),
        },
    )
    np.savetxt(
        output / "dkuc_continuous_training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header=(
            "epoch,train_total,val_total,val_state_derivative,"
            "val_lift_derivative,val_readout,val_rollout"
        ),
        comments="",
    )
    return {
        "model": model,
        "best_val": best_value,
        "best_epoch": best_epoch,
        "history": history,
        "artifact_dir": str(output),
        "latent_dim": int(model.latent_dim),
    }
