"""RBF-dictionary EDMD fitting with portable artifact output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from koopman_control.data.artifacts import (
    save_json,
    save_runtime_matrices,
)
from koopman_control.data.normalization import Normalizer


@dataclass(frozen=True)
class EDMDTrainingConfig:
    n_centers: int = 200
    rbf_sigma: float | None = None
    ridge: float = 1e-4
    kmeans_seed: int = 2007


def estimate_rbf_sigma(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return 1.0
    rng = np.random.RandomState(0)
    count = min(2000, centers.shape[0])
    left = rng.randint(0, centers.shape[0], size=count)
    right = rng.randint(0, centers.shape[0], size=count)
    mask = left != right
    if not np.any(mask):
        return 1.0
    distances = np.linalg.norm(
        centers[left[mask]] - centers[right[mask]],
        axis=1,
    )
    return max(float(np.median(distances)), 1e-6)


def rbf_lift(
    x_norm: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(x_norm, dtype=np.float64))
    diff = values[:, None, :] - centers[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    features = np.exp(-0.5 * sqdist / (sigma * sigma))
    return np.hstack([values, features])


def fit_edmd(
    *,
    states: np.ndarray,
    inputs: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    config: EDMDTrainingConfig,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Fit EDMD and optionally write artifacts loadable by EDMDModel."""
    from sklearn.cluster import MiniBatchKMeans

    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(inputs, dtype=np.float64)
    state_dim = x.shape[-1]
    control_dim = u.shape[-1]
    current = x[:, :-1].reshape(-1, state_dim)
    following = x[:, 1:].reshape(-1, state_dim)
    controls = u.reshape(-1, control_dim)
    current_norm = x_normer.transform(current)
    following_norm = x_normer.transform(following)
    controls_norm = u_normer.transform(controls)

    n_clusters = min(config.n_centers, current_norm.shape[0])
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.kmeans_seed,
        batch_size=256,
        n_init=3,
    )
    kmeans.fit(current_norm)
    centers = kmeans.cluster_centers_.astype(np.float64)
    sigma = (
        float(config.rbf_sigma)
        if config.rbf_sigma is not None
        else estimate_rbf_sigma(centers)
    )
    lifted = rbf_lift(current_norm, centers, sigma)
    lifted_next = rbf_lift(following_norm, centers, sigma)
    omega = np.hstack([lifted, controls_norm])
    sample_count = omega.shape[0]
    gram = (omega.T @ omega) / sample_count
    rhs = (omega.T @ lifted_next) / sample_count
    solution = np.linalg.solve(
        gram + config.ridge * np.eye(gram.shape[0]),
        rhs,
    )
    latent_dim = lifted.shape[1]
    A = solution[:latent_dim].T.copy()
    B = solution[latent_dim:].T.copy()
    C = np.zeros((state_dim, latent_dim), dtype=np.float64)
    C[:, :state_dim] = np.eye(state_dim)
    result: dict[str, object] = {
        "centers": centers,
        "sigma": sigma,
        "A": A,
        "B": B,
        "C": C,
        "cond_number": float(np.linalg.cond(gram)),
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "edmd_model.npz",
            centers=centers,
            sigma=np.array([sigma]),
            A=A,
            B=B,
            cond_number=np.array([result["cond_number"]]),
            x_mean=x_normer.mean,
            x_std=x_normer.std,
            u_mean=u_normer.mean,
            u_std=u_normer.std,
        )
        save_runtime_matrices(
            target / "runtime_matrices.npz",
            A=A,
            B=B,
            C=C,
            control_mode="z_next=A z+B u_norm",
        )
        save_json(
            target / "model_config.json",
            {"model": "EDMD", "config": asdict(config)},
        )
    return result
