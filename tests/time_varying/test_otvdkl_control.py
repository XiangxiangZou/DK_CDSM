from __future__ import annotations

import numpy as np

from control.otvdkl_control import (
    LiftedMPC,
    LiftedMPCConfig,
    OnlineOTVDKLControlTransaction,
    TerminalSDPConfig,
    conservative_symmetric_input_bound,
    safe_control_fallback,
    solve_terminal_sdp,
    terminal_sdp_margins,
)
from prediction.otvdkl_prediction import OTVDKLModelSnapshot
from prediction.otvdkl_prediction import SlidingWindowKoopmanUpdater


def _snapshot(A, B):
    n, m = A.shape[0], B.shape[1]
    return OTVDKLModelSnapshot(A=A, B=B, C=np.eye(n), C_struct=np.eye(n),
        model_version=3, window_version=4, encoder_fingerprint="e", state_dim=n,
        input_dim=m, latent_dim=n)


def test_conservative_input_box_and_fallback() -> None:
    lower = np.array([-2.0, -0.5])
    upper = np.array([1.0, 3.0])
    assert np.array_equal(conservative_symmetric_input_bound(lower, upper), [1.0, 0.5])
    assert np.array_equal(safe_control_fallback(lower, upper), [0.0, 0.0])


def test_terminal_sdp_returns_recomputable_margins() -> None:
    A = np.array([[0.8, 0.1], [0.0, 0.75]])
    B = np.eye(2)
    config = TerminalSDPConfig(q_weight=1e-3, r_weight=1e-3, lmi_tolerance=2e-4)
    result = solve_terminal_sdp(A, B, np.zeros(2), np.ones(2) * 2.0, config)
    assert result.usable, (result.status, result.reason, result.margins)
    oracle = terminal_sdp_margins(A, B, np.zeros(2), np.ones(2) * 2.0,
                                  result.gamma, result.P_bar, result.P, result.K,
                                  np.eye(2) * config.q_weight,
                                  np.eye(2) * config.r_weight)
    assert oracle.keys() == result.margins.keys()
    assert all(np.isclose(oracle[key], result.margins[key]) for key in oracle)


def test_lifted_mpc_stabilizes_scalar_system_and_respects_bounds() -> None:
    snapshot = _snapshot(np.array([[1.05]]), np.array([[1.0]]))
    controller = LiftedMPC(LiftedMPCConfig(horizon=8, feature_weight=1.0))
    controls, predicted = controller.solve(snapshot, np.array([1.0]),
        np.zeros((9, 1)), np.array([-0.2]), np.array([0.2]), np.array([[2.0]]))
    assert controls.shape == (8, 1) and predicted.shape == (8, 1)
    assert np.all(controls >= -0.200001) and np.all(controls <= 0.200001)
    assert controls[0, 0] < 0.0
    assert controller.last_diagnostics["model_version"] == 3


def _online_transaction() -> OnlineOTVDKLControlTransaction:
    rng = np.random.default_rng(12)
    z = rng.normal(size=(12, 2))
    u = rng.normal(scale=0.1, size=(12, 1))
    target = z @ np.array([[0.75, 0.0], [0.0, 0.7]]).T + u @ np.array([[0.2, 0.1]])
    updater = SlidingWindowKoopmanUpdater(
        z, target, u, A0=np.eye(2) * 0.7, B0=np.array([[0.2], [0.1]]),
        state_dim=2, batch_size=2, ridge_lambda=1e-3,
        encoder_fingerprint="fixed", affine_constant=False,
    )
    return OnlineOTVDKLControlTransaction(
        updater, mpc=LiftedMPC(LiftedMPCConfig(horizon=3, feature_weight=1.0)),
        sdp_config=TerminalSDPConfig(q_weight=1e-3, r_weight=1e-3, lmi_tolerance=5e-4),
        deadline_s=10.0,
    )


def test_control_transaction_commits_only_complete_past_batch_before_snapshot() -> None:
    transaction = _online_transaction()
    refs = np.zeros((4, 2))
    first = transaction.step(np.zeros(2), refs, np.array([-1.0]), np.array([1.0]),
                             completed_transition=(np.array([0.1, 0.0]), np.array([0.0]), np.array([0.08, 0.0])))
    assert first.update is None and first.snapshot.model_version == 0
    second = transaction.step(np.zeros(2), refs, np.array([-1.0]), np.array([1.0]),
                              completed_transition=(np.array([0.08, 0.0]), np.array([0.0]), np.array([0.06, 0.0])))
    assert second.update is not None
    assert second.update["accepted"]
    assert second.snapshot.model_version == 1
    assert second.mpc["model_version"] == second.snapshot.model_version
    assert transaction.pending_count == 0


def test_control_transaction_has_deterministic_fallback_for_invalid_sdp_box() -> None:
    transaction = _online_transaction()
    result = transaction.step(np.zeros(2), np.zeros((4, 2)),
                              np.array([0.1]), np.array([1.0]))
    assert result.degraded
    assert result.mpc["status"] == "safe_fallback"
    assert np.array_equal(result.control, [0.1])
