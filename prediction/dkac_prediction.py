"""DKAC prediction method with its own training and evaluation entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from common import (
    INPUT_ORDER,
    STATE_ORDER,
    Normalizer,
    build_windows,
    create_prediction_run_paths,
    fit_state_input_normalizers,
    load_json,
    load_train_val,
    make_device,
    save_dataset,
    save_json,
    save_normalizers,
    save_prediction_outputs,
    set_seed,
)

STATE_DIM = 4
CONTROL_DIM = 2


@dataclass(frozen=True)
class DKACConfig:
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


def activation_layer(name: str):
    import torch.nn as nn

    values = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
    if name not in values:
        raise ValueError(f"Unsupported activation: {name}")
    return values[name]


def mlp(widths: Sequence[int], activation: str):
    import torch.nn as nn

    layers: list[nn.Module] = []
    act = activation_layer(activation)
    for index in range(len(widths) - 1):
        layers.append(nn.Linear(widths[index], widths[index + 1]))
        if index != len(widths) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def make_network_class():
    import torch
    import torch.nn as nn

    class DKACNetwork(nn.Module):
        def __init__(self, config: DKACConfig, state_dim: int = STATE_DIM, control_dim: int = CONTROL_DIM):
            super().__init__()
            self.state_dim = int(state_dim)
            self.control_dim = int(control_dim)
            self.lift_dim = int(config.lift_dim)
            self.latent_dim = self.state_dim + self.lift_dim
            self.control_dim_hat = int(config.control_dim_hat)
            self.identity_control_bias = bool(config.identity_control_bias)
            self.encoder = mlp((self.state_dim, *tuple(config.hidden), self.lift_dim), config.activation)
            self.control_net = mlp(
                (self.state_dim, *tuple(config.control_hidden), self.control_dim_hat * self.control_dim),
                config.activation,
            )
            self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            self.B = nn.Linear(self.control_dim_hat, self.latent_dim, bias=False)
            self.bound_lift = float(config.bound_lift)
            with torch.no_grad():
                self.A.weight.copy_(torch.eye(self.latent_dim))
                self.B.weight.zero_()
                rows = min(self.state_dim, self.control_dim_hat)
                self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

        def lift(self, x_norm):
            features = self.encoder(x_norm)
            if self.bound_lift > 0.0:
                features = self.bound_lift * torch.tanh(features / self.bound_lift)
            return torch.cat([x_norm, features], dim=-1)

        def state_from_latent(self, z):
            return z[..., : self.state_dim]

        def control_matrix(self, x_norm):
            matrix = self.control_net(x_norm).reshape(-1, self.control_dim_hat, self.control_dim)
            if self.identity_control_bias and self.control_dim_hat == self.control_dim:
                eye = torch.eye(self.control_dim, device=x_norm.device, dtype=x_norm.dtype)
                matrix = matrix + eye.reshape(1, self.control_dim, self.control_dim)
            return matrix

        def control_encode(self, x_norm, u_norm):
            return torch.bmm(self.control_matrix(x_norm), u_norm.unsqueeze(-1)).squeeze(-1)

        def step(self, z, u_norm):
            return self.A(z) + self.B(self.control_encode(self.state_from_latent(z), u_norm))

    return DKACNetwork


def curriculum_horizon(epoch: int, config: DKACConfig) -> int:
    start = max(1, min(config.window_start, config.window))
    ramp_epochs = max(1, int(0.6 * config.epochs))
    fraction = min(1.0, (epoch - 1) / max(1, ramp_epochs - 1))
    return int(round(start + fraction * (config.window - start)))


def compute_loss(model, x_sequence, u_sequence, horizon: int, config: DKACConfig):
    import torch

    z_true = model.lift(x_sequence[:, : horizon + 1].reshape(-1, x_sequence.shape[-1]))
    z_true = z_true.reshape(x_sequence.shape[0], horizon + 1, -1)
    z = z_true[:, 0]
    state_loss = torch.zeros((), device=x_sequence.device)
    embed_loss = torch.zeros((), device=x_sequence.device)
    for step in range(horizon):
        z = model.step(z, u_sequence[:, step])
        state_loss = state_loss + torch.mean((model.state_from_latent(z) - x_sequence[:, step + 1]) ** 2)
        embed_loss = embed_loss + torch.mean((z - z_true[:, step + 1]) ** 2)
    return config.w_state * state_loss / horizon + config.w_embed * embed_loss / horizon


def fit_dkac(
    train_states: np.ndarray,
    train_inputs: np.ndarray,
    val_states: np.ndarray,
    val_inputs: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    config: DKACConfig,
    output_dir: str | Path,
    device,
) -> dict[str, object]:
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train, u_train = build_windows(train_states, train_inputs, config.window)
    x_val, u_val = build_windows(val_states, val_inputs, config.window)
    x_train = x_normer.transform(x_train)
    x_val = x_normer.transform(x_val)
    u_train = u_normer.transform(u_train)
    u_val = u_normer.transform(u_val)

    DKACNetwork = make_network_class()
    model = DKACNetwork(config, train_states.shape[-1], train_inputs.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_value = float("inf")
    best_epoch = 0
    history: list[list[float]] = []

    def sample(values_x, values_u):
        idx = np.random.randint(0, values_x.shape[0], size=min(config.batch_size, values_x.shape[0]))
        return (
            torch.from_numpy(values_x[idx].astype(np.float32)).to(device),
            torch.from_numpy(values_u[idx].astype(np.float32)).to(device),
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        horizon = curriculum_horizon(epoch, config)
        train_total = 0.0
        for _ in range(config.steps_per_epoch):
            xb, ub = sample(x_train, u_train)
            loss = compute_loss(model, xb, ub, horizon, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_total += float(loss.detach().item())
        train_total /= max(config.steps_per_epoch, 1)
        model.eval()
        with torch.no_grad():
            xb, ub = sample(x_val, u_val)
            val_loss = compute_loss(model, xb, ub, config.window, config)
        val_value = float(val_loss.item())
        history.append([float(epoch), train_total, val_value])
        if val_value < best_value:
            best_value = val_value
            best_epoch = epoch
            torch.save(model.state_dict(), output / "best_dkac.pt")

    save_json(output / "model_config.json", {"model": "DKAC", "config": asdict(config)})
    np.savetxt(
        output / "dkac_training_history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header="epoch,train_total,val_total",
        comments="",
    )
    return {"latent_dim": int(model.latent_dim), "best_val": best_value, "best_epoch": best_epoch}


class DKACModel:
    name = "dkac"

    def __init__(self, artifact_dir: str | Path, device: str = "cuda"):
        import torch

        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = DKACConfig(**load_json(self.artifact_dir / "model_config.json")["config"])
        normalizers = load_json(self.artifact_dir / "normalizers.json")
        self.x_normer = Normalizer.from_json(normalizers["x"])
        self.u_normer = Normalizer.from_json(normalizers["u"])
        self.state_dim = int(self.x_normer.mean.size)
        self.control_dim = int(self.u_normer.mean.size)
        self._torch = torch
        DKACNetwork = make_network_class()
        self.model = DKACNetwork(self.config, self.state_dim, self.control_dim).to(self.device)
        state_dict = torch.load(self.artifact_dir / "best_dkac.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.A = self.model.A.weight.detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.weight.detach().cpu().numpy().astype(np.float64)

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1)).astype(np.float32)
        with self._torch.no_grad():
            z = self.model.lift(self._torch.from_numpy(x_norm).to(self.device))
        return z.cpu().numpy().reshape(-1).astype(np.float64)

    def control_matrix(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1)).astype(np.float32)
        with self._torch.no_grad():
            matrix = self.model.control_matrix(self._torch.from_numpy(x_norm).to(self.device))
        return matrix.cpu().numpy()[0].astype(np.float64)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        if x_phys is None:
            x_phys = self.recover_state(z)
        u_norm = self.u_normer.transform(np.asarray(u_phys).reshape(1, -1))[0]
        internal_control = self.control_matrix(x_phys) @ u_norm
        return self.A @ np.asarray(z).reshape(-1) + self.B @ internal_control

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[: self.state_dim]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        controls = np.asarray(u_seq, dtype=np.float64)
        pred = np.zeros((controls.shape[0] + 1, self.state_dim), dtype=np.float64)
        pred[0] = np.asarray(x0, dtype=np.float64).reshape(self.state_dim)
        z = self.lift(pred[0])
        for k, control in enumerate(controls):
            z = self.step_latent(z, control, pred[k])
            pred[k + 1] = self.recover_state(z)
        return pred


def save_evaluation(
    model: DKACModel,
    dataset: dict[str, np.ndarray],
    output: Path,
    figures_dir: Path,
    mode: str,
) -> dict[str, object]:
    return save_prediction_outputs(model, dataset, output, figures_dir, mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the DKAC Koopman prediction method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--run_type", choices=["full_run", "smoke_test"], default="full_run")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--pred_mode", choices=["one_step", "rollout", "both"], default="both")
    parser.add_argument("--lift_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128])
    parser.add_argument("--control_hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--control_dim_hat", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "elu", "tanh"], default="elu")
    parser.add_argument("--bound_lift", type=float, default=1.0)
    parser.add_argument("--no_identity_control_bias", action="store_true")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--window_start", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps_per_epoch", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--w_state", type=float, default=1.0)
    parser.add_argument("--w_embed", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    paths = create_prediction_run_paths("dkac", args.run_type, args.tag, args.out_dir or None)
    output = paths.artifact_dir
    train_data, val_data, split_meta = load_train_val(args.train_dataset, args.val_dataset, args.val_ratio, args.seed)
    save_dataset(output / "dataset_train.npz", train_data)
    save_dataset(output / "dataset_val.npz", val_data)
    x_normer, u_normer = fit_state_input_normalizers(train_data)
    save_normalizers(output / "normalizers.json", x_normer, u_normer)
    config = DKACConfig(
        lift_dim=args.lift_dim,
        hidden=tuple(args.hidden),
        control_hidden=tuple(args.control_hidden),
        control_dim_hat=args.control_dim_hat,
        activation=args.activation,
        bound_lift=args.bound_lift,
        identity_control_bias=not args.no_identity_control_bias,
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
    )
    training = fit_dkac(
        train_data["states"],
        train_data["inputs"],
        val_data["states"],
        val_data["inputs"],
        x_normer,
        u_normer,
        config,
        output,
        device,
    )
    model = DKACModel(output, str(device))
    modes = ("one_step", "rollout") if args.pred_mode == "both" else (args.pred_mode,)
    metrics = {mode: save_evaluation(model, val_data, output, paths.figures_dir, mode) for mode in modes}
    save_json(
        output / "run_summary.json",
        {
            "method": "dkac",
            "run_type": args.run_type,
            "run_id": paths.run_id,
            "artifact_dir": str(paths.artifact_dir),
            "figures_dir": str(paths.figures_dir),
            "state_order": list(STATE_ORDER),
            "input_order": list(INPUT_ORDER),
            "train_dataset": args.train_dataset,
            "split": split_meta,
            "config": asdict(config),
            "training": training,
            "metrics": metrics,
        },
    )
    print(f"[done] DKAC artifacts -> {output}")


if __name__ == "__main__":
    main()
