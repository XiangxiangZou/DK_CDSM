import numpy as np

from koopman_control.evaluation.prediction import evaluate_predictions


def test_prediction_metrics_support_generic_state_dimension() -> None:
    true = np.zeros((2, 4, 3), dtype=np.float64)
    pred = true.copy()
    pred[:, 1:, 1] = 2.0
    metrics = evaluate_predictions(true, pred)
    assert metrics["state_labels"] == ["x0", "x1", "x2"]
    assert metrics["rmse_by_state"][1] == 2.0
