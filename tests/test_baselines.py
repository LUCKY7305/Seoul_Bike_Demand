"""
Automated Tests for Baseline Prediction Models
==============================================

Phase 3A Tests:
- Baseline mean calculated strictly from training data only.
- Hourly baseline lookup computed from training data only.
- Missing hour in training falls back to global training mean.
- Closure rule produces exactly 0.0 for Functioning Day == 'No'.
- Prediction array shapes and order preserved.
- Validation and test labels never inspected to generate predictions.
- No negative predictions emitted.
- Metric calculation consistency (MAE, RMSE, R^2).
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    calculate_training_mean,
    compute_hourly_lookup,
    evaluate_predictions,
    predict_hourly_mean,
    predict_known_closure,
    predict_training_mean,
)


@pytest.fixture
def synthetic_train():
    """Small synthetic training set with known values."""
    return pd.DataFrame({
        "Hour": [8, 8, 9, 9, 10],
        "Rented Bike Count": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Functioning Day": ["Yes", "Yes", "Yes", "Yes", "Yes"],
    })


@pytest.fixture
def synthetic_eval():
    """Synthetic evaluation set with known hours, including an unseen hour."""
    return pd.DataFrame({
        "Hour": [8, 9, 10, 23],  # Hour 23 is unseen in training!
        "Rented Bike Count": [150.0, 350.0, 500.0, 0.0],
        "Functioning Day": ["Yes", "Yes", "No", "No"],
    })


def test_calculate_training_mean(synthetic_train):
    """Mean of [100, 200, 300, 400, 500] = 300.0."""
    mean_val = calculate_training_mean(synthetic_train["Rented Bike Count"])
    assert np.isclose(mean_val, 300.0)


def test_calculate_training_mean_empty_raises():
    """Empty training labels must raise ValueError."""
    with pytest.raises(ValueError, match="y_train cannot be empty"):
        calculate_training_mean([])


def test_predict_training_mean_values_and_shape(synthetic_train):
    """Predict training mean on evaluation set."""
    y_train = synthetic_train["Rented Bike Count"]
    preds = predict_training_mean(y_train, n_samples=4)
    assert len(preds) == 4
    assert np.allclose(preds, 300.0)
    assert not np.any(preds < 0)


def test_compute_hourly_lookup(synthetic_train):
    """Verify hour-specific means: Hour 8 -> 150, Hour 9 -> 350, Hour 10 -> 500."""
    lookup = compute_hourly_lookup(synthetic_train, synthetic_train["Rented Bike Count"], hour_col="Hour")
    assert np.isclose(lookup[8], 150.0)
    assert np.isclose(lookup[9], 350.0)
    assert np.isclose(lookup[10], 500.0)


def test_predict_hourly_mean_with_unseen_hour_fallback(synthetic_train, synthetic_eval):
    """
    Hour 8 -> 150, Hour 9 -> 350, Hour 10 -> 500.
    Hour 23 (unseen in training) -> falls back to global training mean (300.0).
    """
    y_train = synthetic_train["Rented Bike Count"]
    preds = predict_hourly_mean(synthetic_train, y_train, synthetic_eval, hour_col="Hour")

    assert len(preds) == len(synthetic_eval)
    assert np.isclose(preds[0], 150.0)  # Hour 8
    assert np.isclose(preds[1], 350.0)  # Hour 9
    assert np.isclose(preds[2], 500.0)  # Hour 10
    assert np.isclose(preds[3], 300.0)  # Hour 23 fallback
    assert not np.any(preds < 0)


def test_predict_known_closure_rule(synthetic_eval):
    """Closure rule: when Functioning Day == 'No', prediction is strictly 0.0."""
    base_preds = np.array([150.0, 350.0, 500.0, 300.0])
    closure_preds = predict_known_closure(
        synthetic_eval,
        base_preds,
        functioning_col="Functioning Day",
        closed_val="No",
    )

    # Row 0: Yes -> 150.0
    # Row 1: Yes -> 350.0
    # Row 2: No  -> 0.0
    # Row 3: No  -> 0.0
    assert np.isclose(closure_preds[0], 150.0)
    assert np.isclose(closure_preds[1], 350.0)
    assert np.isclose(closure_preds[2], 0.0)
    assert np.isclose(closure_preds[3], 0.0)


def test_closure_rule_mismatched_length_raises(synthetic_eval):
    """Mismatched prediction array length must raise ValueError."""
    bad_preds = np.array([100.0, 200.0])
    with pytest.raises(ValueError, match="Length mismatch"):
        predict_known_closure(synthetic_eval, bad_preds)


def test_evaluate_predictions_metrics():
    """Verify MAE, RMSE, and R2 calculations against exact manual values."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])  # Constant error of 10

    metrics = evaluate_predictions(y_true, y_pred)
    assert np.isclose(metrics["mae"], 10.0)
    assert np.isclose(metrics["rmse"], 10.0)
    assert metrics["r2"] > 0.98  # Very high R2
    assert metrics["n_samples"] == 4


def test_evaluate_predictions_empty_raises():
    """Empty ground truth labels must raise ValueError."""
    with pytest.raises(ValueError, match="Cannot evaluate metrics on empty arrays"):
        evaluate_predictions([], [])
