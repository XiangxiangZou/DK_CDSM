"""Shared multi-step trainer for DKUC, DKAC, and DKN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from koopman_control.data.artifacts import (
    save_json,
    save_runtime_matrices,
)
from koopman_control.data.normalization import Normalizer
from koopman_control.models.networks import (
    DKACNetwork,
    DKNNetwork,
    DKUCNetwork,
)
from koopman_control.training.windows import build_windows

ModelKind = Literal["dkuc", "dkac", "dkn"]


@dataclass(frozen=True)
class NeuralTrainingConfig:
    lift_dim: int = 64
    hidden: tuple[int, ...] = (128, 256, 128)
    control_hidden: tuple[int, ...] = (128, 128)
    control_dim_hat: int = 2
    activation: str = "elu"
    bound_lift: float = 1.0
    identity_control_bias: bool = True
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


def _make_model(
    kind: ModelKind,
    config: NeuralTrainingConfig,
    state_dim: int,
    control_dim: int,
):
    common = {
        "lift_dim": config.lift_dim,
        "hidden": config.hidden,
        "activation": config.activation,
        "state_dim": state_dim,
        "control_dim": control_dim,
    }
    if kind == "dkuc":
        return DKUCNetwork(
            **common,
            bound_lift=config.bound_lift,
        )
    if kind == "dkac":
        return DKACNetwork(
            **common,
            control_hidden=config.control_hidden,
            control_dim_hat=config.control_dim_hat,
            bound_lift=config.bound_lift,
            identity_control_bias=config.identity_control_bias,
        )
    if kind == "dkn":
        return DKNNetwork(
            **common,
            control_hidden=config.control_hidden,
            control_dim_hat=config.control_dim_hat,
            bound_lift=bool(config.bound_lift),
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def _curriculum_horizon(
    epoch: int,
    config: NeuralTrainingConfig,
) -> int:
    start = max(1, min(config.window_start, config.window))
    ramp_epochs = max(1, int(0.6 * config.epochs))
    fraction = min(
        1.0,
        (epoch - 1) / max(1, ramp_epochs - 1),
    )
    return int(round(start + fraction * (config.window - start)))


def _compute_loss(model, kind, x_sequence, u_sequence, horizon, config):
    import torch

    z_true = (
        model.encode(x_sequence[:, : horizon + 1].reshape(-1, x_sequence.shape[-1]))
        if kind == "dkn"
        else model.lift(x_sequence[:, : horizon + 1].reshape(-1, x_sequence.shape[-1]))
    )
    z_true = z_true.reshape(
        x_sequence.shape[0],
        horizon + 1,
        -1,
    )
    z = z_true[:, 0]
    state_loss = torch.zeros((), device=x_sequence.device)
    embed_loss = torch.zeros((), device=x_sequence.device)
    for step in range(horizon):
        if kind == "dkn":
            z = model.koopman_step(z, u_sequence[:, step])
        else:
            z = model.step(z, u_sequence[:, step])
        state_loss = state_loss + torch.mean(
            (model.state_from_latent(z) - x_sequence[:, step + 1]) ** 2
        )
        embed_loss = embed_loss + torch.mean(
            (z - z_true[:, step + 1]) ** 2
        )
    state_loss = state_loss / horizon
    embed_loss = embed_loss / horizon
    total = (
        config.w_state * state_loss
        + config.w_embed * embed_loss
    )
    return total, state_loss, embed_loss


def fit_neural_koopman(
    *,
    kind: ModelKind,
    train_states: np.ndarray,
    train_inputs: np.ndarray,
    val_states: np.ndarray,
    val_inputs: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    config: NeuralTrainingConfig,
    output_dir: str | Path,
    device,
) -> dict[str, object]:
    """Train one neural Koopman model and write portable artifacts."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train, u_train = build_windows(
        train_states,
        train_inputs,
        config.window,
    )
    x_val, u_val = build_windows(
        val_states,
        val_inputs,
        config.window,
    )
    x_train = x_normer.transform(x_train)
    x_val = x_normer.transform(x_val)
    u_train = u_normer.transform(u_train)
    u_val = u_normer.transform(u_val)
    model = _make_model(
        kind,
        config,
        train_states.shape[-1],
        train_inputs.shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    best_value = float("inf")
    best_epoch = 0
    checkpoint_path = output / f"best_{kind}.pt"
    history: list[list[float]] = []

    def sample(values_x, values_u):
        indices = np.random.randint(
            0,
            values_x.shape[0],
            size=min(config.batch_size, values_x.shape[0]),
        )
        x_batch = torch.from_numpy(
            values_x[indices].astype(np.float32)
        ).to(device)
        u_batch = torch.from_numpy(
            values_u[indices].astype(np.float32)
        ).to(device)
        return x_batch, u_batch

    for epoch in range(1, config.epochs + 1):
        model.train()
        horizon = _curriculum_horizon(epoch, config)
        train_total = 0.0
        for _ in range(config.steps_per_epoch):
            x_batch, u_batch = sample(x_train, u_train)
            total, _, _ = _compute_loss(
                model,
                kind,
                x_batch,
                u_batch,
                horizon,
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
            optimizer.step()
            train_total += float(total.detach().item())
        train_total /= max(config.steps_per_epoch, 1)

        model.eval()
        with torch.no_grad():
            x_batch, u_batch = sample(x_val, u_val)
            val_total, val_state, val_embed = _compute_loss(
                model,
                kind,
                x_batch,
                u_batch,
                config.window,
                config,
            )
        val_value = float(val_total.item())
        history.append(
            [
                float(epoch),
                train_total,
                val_value,
                float(val_state.item()),
                float(val_embed.item()),
            ]
        )
        if val_value < best_value:
            best_value = val_value
            best_epoch = epoch
            if kind == "dkn":
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(config),
                        "best_val": best_value,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
            else:
                torch.save(model.state_dict(), checkpoint_path)

    if kind == "dkn":
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    model.eval()

    A = model.A.weight.detach().cpu().numpy().astype(np.float64)
    B = model.B.weight.detach().cpu().numpy().astype(np.float64)
    state_dim = train_states.shape[-1]
    C = np.zeros((state_dim, model.latent_dim), dtype=np.float64)
    C[:, :state_dim] = np.eye(state_dim)
    control_mode = {
        "dkuc": "z_next=A z+B u_norm",
        "dkac": "z_next=A z+B v, v=G(x_norm)u_norm",
        "dkn": "prediction_only_state_dependent_control_encoder",
    }[kind]
    save_runtime_matrices(
        output / "runtime_matrices.npz",
        A=A,
        B=B,
        C=C,
        control_mode=control_mode,
    )
    save_json(
        output / "model_config.json",
        {"model": kind.upper(), "config": asdict(config)},
    )
    np.savetxt(
        output / f"{kind}_training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,val_total,val_state,val_embed",
        comments="",
    )
    return {
        "model": model,
        "best_val": best_value,
        "best_epoch": best_epoch,
        "history": history,
        "artifact_dir": str(output),
        "control_capable": kind != "dkn",
    }
