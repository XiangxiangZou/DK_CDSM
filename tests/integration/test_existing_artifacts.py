from pathlib import Path

import numpy as np
import pytest

from koopman_control.models.registry import load_prediction_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "deployment_pipeline"
    / "20260604_163124_train_models_fullrun_pd48x250"
)


@pytest.mark.parametrize("model_name", ["edmd", "dkuc", "dkac", "dkn"])
def test_existing_model_artifact_loads(model_name: str) -> None:
    if not ARTIFACT_ROOT.exists():
        pytest.skip("local deployment artifacts are not available")
    model = load_prediction_model(ARTIFACT_ROOT, model_name, device="cpu")
    state = np.zeros(4, dtype=np.float64)
    control = np.zeros(2, dtype=np.float64)
    prediction = model.rollout(state, control.reshape(1, 2))
    assert prediction.shape == (2, 4)
