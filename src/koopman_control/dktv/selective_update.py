"""Threshold-triggered and negative-update-rejecting window updates."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .window_update import SlidingWindowKoopmanUpdater, WindowCandidate, WindowUpdateResult, latent_rmse


class SelectiveWindowKoopmanUpdater:
    """Apply threshold skipping and candidate rejection to a window updater."""

    supported_policies = ("discard_on_reject", "retain_on_reject")

    def __init__(
        self,
        updater: SlidingWindowKoopmanUpdater,
        *,
        epsilon: float,
        reject_buffer_policy: str = "discard_on_reject",
        improvement_tolerance: float = 0.0,
    ) -> None:
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be non-negative")
        if reject_buffer_policy not in self.supported_policies:
            raise ValueError(f"unsupported reject buffer policy: {reject_buffer_policy}")
        self.updater = updater
        self.epsilon = float(epsilon)
        self.reject_buffer_policy = reject_buffer_policy
        self.improvement_tolerance = float(improvement_tolerance)

    def update(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        u_normalized: np.ndarray,
        *,
        sample_ids: np.ndarray,
        encoder_fingerprint: str,
    ) -> tuple[WindowUpdateResult, WindowCandidate | None]:
        started = perf_counter()
        base = self.updater
        base.attempt_index += 1
        new_ids = np.asarray(sample_ids, dtype=np.int64)
        evicted = base.sample_ids[: base.batch_size].copy()
        try:
            current_rmse = latent_rmse(base.A, base.B, z_current, u_normalized, z_next)
            candidate = base.propose(
                z_current,
                z_next,
                u_normalized,
                sample_ids=new_ids,
                encoder_fingerprint=encoder_fingerprint,
            )
            candidate_rmse = latent_rmse(
                candidate.result.A, candidate.result.B, z_current, u_normalized, z_next
            )
            if current_rmse <= self.epsilon:
                base.commit(candidate, accept_model=False)
                return base._record(
                    status="skipped_threshold",
                    reason="current_batch_rmse_not_above_epsilon",
                    started=started,
                    new_ids=new_ids,
                    evicted_ids=evicted,
                    candidate=candidate,
                    current_rmse=current_rmse,
                    candidate_rmse=candidate_rmse,
                    window_advanced=True,
                    policy=self.reject_buffer_policy,
                ), candidate
            if candidate_rmse + self.improvement_tolerance >= current_rmse:
                retain = self.reject_buffer_policy == "retain_on_reject"
                if retain:
                    base.commit(candidate, accept_model=False)
                return base._record(
                    status="rejected",
                    reason="candidate_not_better_on_new_batch",
                    started=started,
                    new_ids=new_ids,
                    evicted_ids=evicted,
                    candidate=candidate,
                    current_rmse=current_rmse,
                    candidate_rmse=candidate_rmse,
                    window_advanced=retain,
                    policy=self.reject_buffer_policy,
                ), candidate
            base.commit(candidate, accept_model=True)
            return base._record(
                status="accepted",
                reason="accepted",
                started=started,
                new_ids=new_ids,
                evicted_ids=evicted,
                candidate=candidate,
                current_rmse=current_rmse,
                candidate_rmse=candidate_rmse,
                window_advanced=True,
                policy=self.reject_buffer_policy,
            ), candidate
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return base._record(
                status="failed_numerical",
                reason=f"failed_numerical:{type(error).__name__}:{error}",
                started=started,
                new_ids=new_ids,
                evicted_ids=evicted,
                candidate=None,
                current_rmse=None,
                candidate_rmse=None,
                window_advanced=False,
                policy=self.reject_buffer_policy,
            ), None

    def __getattr__(self, name: str):
        return getattr(self.updater, name)
