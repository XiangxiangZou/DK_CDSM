"""Filter complete trajectories from a collected deployment dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject trajectories with unsafe joint angles or cable values."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_abs_q_deg", type=float, default=81.0)
    parser.add_argument("--max_cable_value", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.dataset.resolve()
    out_dir = args.out_dir.resolve()

    with np.load(source) as raw:
        arrays = {name: raw[name] for name in raw.files}

    states = arrays["states"]
    cable_ctrl = arrays["cable_ctrl"]
    n_traj = states.shape[0]
    if cable_ctrl.shape[0] != n_traj:
        raise ValueError("states and cable_ctrl must have the same trajectory count")

    q_deg = np.degrees(states[..., :2])
    reject_joint = np.any(np.abs(q_deg) > args.max_abs_q_deg, axis=(1, 2))
    reject_cable = np.any(cable_ctrl > args.max_cable_value, axis=(1, 2))
    reject = reject_joint | reject_cable
    keep = ~reject

    if not np.any(keep):
        raise RuntimeError("All trajectories were rejected")

    filtered = {
        name: value[keep] if value.ndim > 0 and value.shape[0] == n_traj else value
        for name, value in arrays.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_dataset = out_dir / "dataset.npz"
    np.savez_compressed(output_dataset, **filtered)

    steps_per_trajectory = int(arrays["inputs"].shape[1])
    summary = {
        "source_dataset": str(source),
        "output_dataset": str(output_dataset),
        "thresholds": {
            "max_abs_q_deg": args.max_abs_q_deg,
            "max_cable_value": args.max_cable_value,
        },
        "original_trajectories": int(n_traj),
        "kept_trajectories": int(np.count_nonzero(keep)),
        "rejected_trajectories": int(np.count_nonzero(reject)),
        "rejected_indices": np.flatnonzero(reject).tolist(),
        "rejected_joint_indices": np.flatnonzero(reject_joint).tolist(),
        "rejected_cable_indices": np.flatnonzero(reject_cable).tolist(),
        "steps_per_trajectory": steps_per_trajectory,
        "kept_transitions": int(np.count_nonzero(keep) * steps_per_trajectory),
    }
    (out_dir / "filtering_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"kept={summary['kept_trajectories']}/{n_traj}")
    print(f"rejected_indices={summary['rejected_indices']}")
    print(f"dataset={output_dataset}")


if __name__ == "__main__":
    main()
