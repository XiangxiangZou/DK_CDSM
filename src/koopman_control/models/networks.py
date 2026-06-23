"""Standalone neural network definitions for DKUC, DKAC, and DKN."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from koopman_control.constants import CONTROL_DIM, STATE_DIM


class MLP(nn.Module):
    """Small configurable multilayer perceptron."""

    def __init__(self, widths: Sequence[int], activation: str = "elu") -> None:
        super().__init__()
        act_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index != len(widths) - 2:
                layers.append(act_map[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class DKUCNetwork(nn.Module):
    """Deep Koopman model with unchanged control."""

    def __init__(
        self,
        lift_dim: int,
        hidden: Sequence[int],
        activation: str,
        bound_lift: float,
        state_dim: int = STATE_DIM,
        control_dim: int = CONTROL_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.lift_dim = int(lift_dim)
        self.latent_dim = self.state_dim + self.lift_dim
        self.encoder = MLP(
            (self.state_dim, *tuple(hidden), self.lift_dim),
            activation,
        )
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)
        self.bound_lift = float(bound_lift)
        self._init_linear()

    def _init_linear(self) -> None:
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.latent_dim))
            self.B.weight.zero_()
            rows = min(self.state_dim, self.control_dim)
            self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

    def lift(self, x_norm: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x_norm)
        if self.bound_lift > 0.0:
            features = self.bound_lift * torch.tanh(
                features / self.bound_lift
            )
        return torch.cat([x_norm, features], dim=-1)

    def state_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.state_dim]

    def step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(u_norm)


class ContinuousDKUCNetwork(DKUCNetwork):
    """Continuous-time DKUC network used for Yu-Tan-style KILC.

    The lifting map is the same state-embedded DKUC form as the discrete model:
    ``z=[x_norm, phi(x_norm)]``. The linear layers represent continuous-time
    dynamics ``zdot = A_c z + B_c u_norm``.
    """

    def _init_linear(self) -> None:
        with torch.no_grad():
            self.A.weight.zero_()
            self.B.weight.zero_()
            rows = min(self.state_dim, self.control_dim)
            self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

    def derivative(
        self,
        z: torch.Tensor,
        u_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.A(z) + self.B(u_norm)

    def euler_step(
        self,
        z: torch.Tensor,
        u_norm: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        return z + float(dt) * self.derivative(z, u_norm)


class DKACNetwork(nn.Module):
    """Deep Koopman model with a state-dependent affine control map."""

    def __init__(
        self,
        lift_dim: int,
        hidden: Sequence[int],
        control_hidden: Sequence[int],
        control_dim_hat: int,
        activation: str,
        bound_lift: float,
        identity_control_bias: bool,
        state_dim: int = STATE_DIM,
        control_dim: int = CONTROL_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.lift_dim = int(lift_dim)
        self.latent_dim = self.state_dim + self.lift_dim
        self.control_dim_hat = int(control_dim_hat)
        self.identity_control_bias = bool(identity_control_bias)
        self.encoder = MLP(
            (self.state_dim, *tuple(hidden), self.lift_dim),
            activation,
        )
        self.control_net = MLP(
            (
                self.state_dim,
                *tuple(control_hidden),
                self.control_dim_hat * self.control_dim,
            ),
            activation,
        )
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim_hat, self.latent_dim, bias=False)
        self.bound_lift = float(bound_lift)
        self._init_linear()

    def _init_linear(self) -> None:
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.latent_dim))
            self.B.weight.zero_()
            rows = min(self.state_dim, self.control_dim_hat)
            self.B.weight[:rows, :rows] = 0.01 * torch.eye(rows)

    def lift(self, x_norm: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x_norm)
        if self.bound_lift > 0.0:
            features = self.bound_lift * torch.tanh(
                features / self.bound_lift
            )
        return torch.cat([x_norm, features], dim=-1)

    def state_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.state_dim]

    def control_matrix(self, x_norm: torch.Tensor) -> torch.Tensor:
        raw = self.control_net(x_norm)
        matrix = raw.reshape(
            -1,
            self.control_dim_hat,
            self.control_dim,
        )
        if (
            self.identity_control_bias
            and self.control_dim_hat == self.control_dim
        ):
            eye = torch.eye(
                self.control_dim,
                device=x_norm.device,
                dtype=x_norm.dtype,
            )
            matrix = matrix + eye.reshape(
                1,
                self.control_dim,
                self.control_dim,
            )
        return matrix

    def control_encode(
        self,
        x_norm: torch.Tensor,
        u_norm: torch.Tensor,
    ) -> torch.Tensor:
        matrix = self.control_matrix(x_norm)
        return torch.bmm(matrix, u_norm.unsqueeze(-1)).squeeze(-1)

    def step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        x_for_control = self.state_from_latent(z)
        internal_control = self.control_encode(x_for_control, u_norm)
        return self.A(z) + self.B(internal_control)


class DKNNetwork(nn.Module):
    """Deep Koopman nonlinear-control prediction model."""

    def __init__(
        self,
        lift_dim: int,
        hidden: Sequence[int],
        control_hidden: Sequence[int],
        control_dim_hat: int,
        activation: str,
        bound_lift: bool,
        state_dim: int = STATE_DIM,
        control_dim: int = CONTROL_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.lift_dim = int(lift_dim)
        self.latent_dim = self.state_dim + self.lift_dim
        self.control_dim_hat = int(control_dim_hat)
        self.bound_lift = bool(bound_lift)
        self.lift = MLP(
            (self.state_dim, *tuple(hidden), self.lift_dim),
            activation,
        )
        self.control_net = MLP(
            (
                self.state_dim + self.control_dim,
                *tuple(control_hidden),
                self.control_dim_hat,
            ),
            activation,
        )
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim_hat, self.latent_dim, bias=False)
        self._init_linear()

    def _init_linear(self) -> None:
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.latent_dim))
            nn.init.xavier_uniform_(self.B.weight, gain=0.05)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        features = self.lift(x_norm)
        if self.bound_lift:
            features = torch.tanh(features)
        return torch.cat([x_norm, features], dim=-1)

    def state_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.state_dim]

    def control_encode(
        self,
        x_norm: torch.Tensor,
        u_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.control_net(torch.cat([x_norm, u_norm], dim=-1))

    def koopman_step(
        self,
        z: torch.Tensor,
        u_norm: torch.Tensor,
    ) -> torch.Tensor:
        x_for_control = self.state_from_latent(z)
        encoded_control = self.control_encode(x_for_control, u_norm)
        return self.A(z) + self.B(encoded_control)
