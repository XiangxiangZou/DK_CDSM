"""EDMD prediction method with its own training and evaluation entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common import (
    INPUT_ORDER,
    STATE_ORDER,
    Normalizer,
    create_prediction_run_paths,
    fit_state_input_normalizers,
    load_train_val,
    save_dataset,
    save_json,
    save_normalizers,
    save_prediction_outputs,
    set_seed,
)


@dataclass(frozen=True)
class EDMDConfig:
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
    distances = np.linalg.norm(centers[left[mask]] - centers[right[mask]], axis=1)
    return max(float(np.median(distances)), 1e-6)


def rbf_lift(x_norm: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    values = np.atleast_2d(np.asarray(x_norm, dtype=np.float64))
    diff = values[:, None, :] - centers[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return np.hstack([values, np.exp(-0.5 * sqdist / (sigma * sigma))])


class EDMDModel:
    name = "edmd"

    def __init__(self, artifact_dir: str | Path):
        with np.load(Path(artifact_dir) / "edmd_model.npz", allow_pickle=False) as data:
            self.centers = np.asarray(data["centers"], dtype=np.float64)
            self.sigma = float(np.asarray(data["sigma"]).reshape(-1)[0])
            self.A = np.asarray(data["A"], dtype=np.float64)
            self.B = np.asarray(data["B"], dtype=np.float64)
            self.x_normer = Normalizer(np.asarray(data["x_mean"]), np.asarray(data["x_std"]))
            self.u_normer = Normalizer(np.asarray(data["u_mean"]), np.asarray(data["u_std"]))
        self.state_dim = int(self.x_normer.mean.size)

    def lift(self, x_phys: np.ndarray) -> np.ndarray:
        x_norm = self.x_normer.transform(np.asarray(x_phys).reshape(1, -1))
        return rbf_lift(x_norm, self.centers, self.sigma).reshape(-1)

    def step_latent(self, z: np.ndarray, u_phys: np.ndarray, x_phys: np.ndarray | None = None) -> np.ndarray:
        del x_phys
        u_norm = self.u_normer.transform(np.asarray(u_phys).reshape(1, -1))[0]
        return self.A @ np.asarray(z).reshape(-1) + self.B @ u_norm

    def recover_state(self, z: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(z, dtype=np.float64).reshape(-1)[: self.state_dim]
        return self.x_normer.inverse(x_norm.reshape(1, -1))[0]

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        controls = np.asarray(u_seq, dtype=np.float64)
        pred = np.zeros((controls.shape[0] + 1, self.state_dim), dtype=np.float64)
        pred[0] = np.asarray(x0, dtype=np.float64).reshape(self.state_dim)
        z = self.lift(pred[0])
        for k, control in enumerate(controls):
            z = self.step_latent(z, control)
            pred[k + 1] = self.recover_state(z)
        return pred


def fit_edmd(
    states: np.ndarray,
    inputs: np.ndarray,
    x_normer: Normalizer,
    u_normer: Normalizer,
    config: EDMDConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    from sklearn.cluster import MiniBatchKMeans

    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(inputs, dtype=np.float64)
    current = x[:, :-1].reshape(-1, x.shape[-1])
    following = x[:, 1:].reshape(-1, x.shape[-1])
    controls = u.reshape(-1, u.shape[-1])
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
    sigma = float(config.rbf_sigma) if config.rbf_sigma is not None else estimate_rbf_sigma(centers)
    lifted = rbf_lift(current_norm, centers, sigma)
    lifted_next = rbf_lift(following_norm, centers, sigma)
    omega = np.hstack([lifted, controls_norm])
    gram = (omega.T @ omega) / omega.shape[0]
    rhs = (omega.T @ lifted_next) / omega.shape[0]
    solution = np.linalg.solve(gram + config.ridge * np.eye(gram.shape[0]), rhs)
    latent_dim = lifted.shape[1]
    A = solution[:latent_dim].T.copy()
    B = solution[latent_dim:].T.copy()
    cond_number = float(np.linalg.cond(gram))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "edmd_model.npz",
        centers=centers,
        sigma=np.array([sigma]),
        A=A,
        B=B,
        cond_number=np.array([cond_number]),
        x_mean=x_normer.mean,
        x_std=x_normer.std,
        u_mean=u_normer.mean,
        u_std=u_normer.std,
    )
    save_json(output / "model_config.json", {"model": "EDMD", "config": asdict(config)})
    return {"latent_dim": latent_dim, "sigma": sigma, "cond_number": cond_number}


def save_evaluation(
    model: EDMDModel,
    dataset: dict[str, np.ndarray],
    output: Path,
    figures_dir: Path,
    mode: str,
) -> dict[str, object]:
    return save_prediction_outputs(model, dataset, output, figures_dir, mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the EDMD Koopman prediction method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--run_type", choices=["full_run", "smoke_test"], default="full_run")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--pred_mode", choices=["one_step", "rollout", "both"], default="both")
    parser.add_argument("--edmd_centers", type=int, default=200)
    parser.add_argument("--edmd_sigma", type=float, default=None)
    parser.add_argument("--edmd_ridge", type=float, default=1e-4)
    parser.add_argument("--edmd_seed", type=int, default=2007)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    paths = create_prediction_run_paths("edmd", args.run_type, args.tag, args.out_dir or None)
    output = paths.artifact_dir
    train_data, val_data, split_meta = load_train_val(args.train_dataset, args.val_dataset, args.val_ratio, args.seed)
    save_dataset(output / "dataset_train.npz", train_data)
    save_dataset(output / "dataset_val.npz", val_data)
    x_normer, u_normer = fit_state_input_normalizers(train_data)
    save_normalizers(output / "normalizers.json", x_normer, u_normer)

    config = EDMDConfig(
        n_centers=args.edmd_centers,
        rbf_sigma=args.edmd_sigma,
        ridge=args.edmd_ridge,
        kmeans_seed=args.edmd_seed,
    )
    summary = fit_edmd(train_data["states"], train_data["inputs"], x_normer, u_normer, config, output)
    model = EDMDModel(output)
    modes = ("one_step", "rollout") if args.pred_mode == "both" else (args.pred_mode,)
    metrics = {mode: save_evaluation(model, val_data, output, paths.figures_dir, mode) for mode in modes}
    save_json(
        output / "run_summary.json",
        {
            "method": "edmd",
            "run_type": args.run_type,
            "run_id": paths.run_id,
            "artifact_dir": str(paths.artifact_dir),
            "figures_dir": str(paths.figures_dir),
            "state_order": list(STATE_ORDER),
            "input_order": list(INPUT_ORDER),
            "train_dataset": args.train_dataset,
            "split": split_meta,
            "config": asdict(config),
            "training": summary,
            "metrics": metrics,
        },
    )
    print(f"[done] EDMD artifacts -> {output}")


if __name__ == "__main__":
    main()
