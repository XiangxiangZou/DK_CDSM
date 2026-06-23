"""Yu-Tan-style lifted-state Koopman iterative learning control.

Paper: Yu & Tan (2026), IJRR.  The controller combines:
  - PD feedback on physical state error  (paperʼs K * e term)
  - lifted-space ILC feedforward         (model-based B^T / column-energy)
  - adaptive integral term               (γ₁,  compensates model uncertainty)
  - robust tanh-saturation term          (γ₂,γ₃, compensates bounded disturbances)

No empirical Q-filter is used — stability follows from the Lyapunov
analysis in Sec. 3 of the paper under Assumptions 1–3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class YuTanKILCConfig:
    """Paper-aligned KILC configuration.

    Parameters
    ----------
    learning_rate:
        ILC feedforward step size.  Default 1.0 means the full model-based
        correction ``B_c^T / ||B_c columns||² · e_z`` is added each trial.
    adaptive_gain:
        Integral-of-projected-error gain (paperʼs γ₁).  Kept small so that
        the integral term slowly removes steady-state lifted error.
    robust_gain:
        tanh-saturation gain for bounded-disturbance rejection (paperʼs
        γ₂ / γ₃ neighbourhood).
    robust_boundary:
        Paperʼs ξ — width of the tanh linear region.
    feedback_kp_a / _b:
        PD proportional gain for joint a / b  [Nm / rad].
    feedback_kd_a / _b:
        PD derivative gain for joint a / b    [Nm / (rad/s)].
    """

    learning_rate: float = 1.0
    adaptive_gain: float = 0.001
    robust_gain: float = 0.001
    robust_boundary: float = 0.1

    feedback_kp_a: float = 40.0
    feedback_kp_b: float = 40.0
    feedback_kd_a: float = 4.0
    feedback_kd_b: float = 4.0


@dataclass(frozen=True)
class YuTanKILCUpdate:
    """Decomposed control sequence returned by :meth:`YuTanKILCController.update`."""

    e_z: np.ndarray  # (N, nz)  lifted tracking error
    u_ilc: np.ndarray  # (N, nu)  pure ILC feedforward
    u_adaptive: np.ndarray  # (N, nu)  integral correction
    u_robust: np.ndarray  # (N, nu)  tanh-saturation correction
    u_total: np.ndarray  # (N, nu)  u_ilc + u_adaptive + u_robust


class YuTanKILCController:
    """Paper-aligned lifted-state KILC (continuous-time DKUC)."""

    def __init__(
        self,
        A_c: np.ndarray,
        B_c: np.ndarray,
        *,
        dt: float,
        config: YuTanKILCConfig,
    ) -> None:
        self.A_c = np.asarray(A_c, dtype=np.float64)
        self.B_c = np.asarray(B_c, dtype=np.float64)
        self.dt = float(dt)
        self.config = config

        if self.A_c.ndim != 2 or self.A_c.shape[0] != self.A_c.shape[1]:
            raise ValueError("A_c must be square")
        if self.B_c.ndim != 2 or self.B_c.shape[0] != self.A_c.shape[0]:
            raise ValueError("B_c rows must match A_c")
        self.nz = int(self.A_c.shape[0])
        self.nu = int(self.B_c.shape[1])

        # Column-normalised transpose gain  (paperʼs B^T / ||B columns||²)
        column_energy = np.sum(self.B_c * self.B_c, axis=0)
        column_energy = np.maximum(column_energy, 1e-8)
        self.L = self.B_c.T / column_energy.reshape(-1, 1)

        # PD feedback gain  K: (nu, nx) = (2, 4)  for CDSM
        self.K = np.array(
            [
                [
                    float(config.feedback_kp_a),
                    0.0,
                    float(config.feedback_kd_a),
                    0.0,
                ],
                [
                    0.0,
                    float(config.feedback_kp_b),
                    0.0,
                    float(config.feedback_kd_b),
                ],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Feedback (online — called inside each trial)
    # ------------------------------------------------------------------

    def feedback(self, e_phys: np.ndarray) -> np.ndarray:
        """PD feedback on physical state error.

        u_fb = K @ e_phys    where  e_phys = x_ref - x_meas  (4,).
        """
        return self.K @ np.asarray(e_phys, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # ILC update (offline — called between trials)
    # ------------------------------------------------------------------

    def update(
        self,
        u_prev: np.ndarray,
        z_ref: np.ndarray,
        z_meas: np.ndarray,
        t: np.ndarray,
    ) -> YuTanKILCUpdate:
        """Compute the next-trial feedforward from lifted-state error.

        No Q-filter is applied — the Lyapunov-based design guarantees
        convergence under Assumptions 1–3 of the paper.
        """
        u_prev = np.asarray(u_prev, dtype=np.float64)
        z_ref = np.asarray(z_ref, dtype=np.float64)
        z_meas = np.asarray(z_meas, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        if z_ref.shape != z_meas.shape:
            raise ValueError("z_ref and z_meas must have identical shape")
        if z_ref.ndim != 2 or z_ref.shape[1] != self.nz:
            raise ValueError(f"z_ref must have shape (N, {self.nz})")
        if u_prev.shape != (z_ref.shape[0], self.nu):
            raise ValueError(
                f"u_prev must have shape ({z_ref.shape[0]}, {self.nu})"
            )
        if t.shape[0] != z_ref.shape[0]:
            raise ValueError("t length must match trajectory length")

        e_z = z_ref - z_meas
        projected = e_z @ self.L.T  # (N, nu)

        # P-type ILC  (Eq. 23 main term)
        u_ilc = u_prev + float(self.config.learning_rate) * projected

        # Adaptive integral  (γ₁ in paper)
        u_adaptive = (
            float(self.config.adaptive_gain)
            * np.cumsum(projected, axis=0)
            * self.dt
        )

        # Robust tanh saturation  (γ₂, γ₃ neighbourhood, ξ boundary)
        boundary = max(float(self.config.robust_boundary), 1e-9)
        u_robust = float(self.config.robust_gain) * np.tanh(
            projected / boundary
        )

        u_total = u_ilc + u_adaptive + u_robust
        # No Q-filter — convergence is guaranteed by Lyapunov analysis.
        return YuTanKILCUpdate(
            e_z=e_z,
            u_ilc=u_ilc,
            u_adaptive=u_adaptive,
            u_robust=u_robust,
            u_total=np.asarray(u_total, dtype=np.float64),
        )
