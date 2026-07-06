import numpy as np
import pytest

from koopman_control.data.datasets import (
    split_train_val,
    validate_dataset,
)


def _dataset() -> dict[str, np.ndarray]:
    return {
        "states": np.zeros((5, 4, 3), dtype=np.float64),
        "inputs": np.zeros((5, 3, 2), dtype=np.float64),
    }


def test_validate_and_split_dataset_by_trajectory() -> None:
    arrays = _dataset()
    validate_dataset(arrays, state_dim=3, control_dim=2)
    train, val, metadata = split_train_val(arrays, 0.4, seed=7)
    assert train["states"].shape[0] == 3
    assert val["states"].shape[0] == 2
    assert metadata["train_indices"] != metadata["val_indices"]


def test_validate_dataset_rejects_wrong_time_relation() -> None:
    arrays = _dataset()
    arrays["inputs"] = np.zeros((5, 4, 2))
    with pytest.raises(ValueError, match="one more step"):
        validate_dataset(arrays)
