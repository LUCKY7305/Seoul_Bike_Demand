"""
Baseline Prediction Models and Evaluation for Vehicle Rental Demand Prediction
==============================================================================

Phase 3A: Implements simple, heuristic non-ML baselines for regression comparison:
1. Training Mean Baseline: Predicts overall operating-hour training mean.
2. Hour-of-Day Mean Baseline: Predicts diurnal hour-specific mean from operating training data,
   falling back to overall mean for unseen hours.
3. Known Closure Rule: Predicts exactly 0 when Functioning Day == 'No', and applies operational
   predictions when Functioning Day == 'Yes'.

All baselines strictly accept training labels as explicit inputs and never inspect evaluation labels.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_training_mean(y_train: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute mean target value from training labels only.
    Raises ValueError if training set is empty.
    """
    if len(y_train) == 0:
        raise ValueError("y_train cannot be empty when calculating training mean.")
    return float(np.mean(y_train))


def predict_training_mean(
    y_train: Union[pd.Series, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """
    Baseline 1: Predicts constant training mean for every evaluation sample.

    Parameters:
        y_train: Training target labels.
        n_samples: Number of evaluation samples to generate predictions for.

    Returns:
        np.ndarray of shape (n_samples,) filled with training mean.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    mean_val = calculate_training_mean(y_train)
    return np.full(shape=(n_samples,), fill_value=mean_val, dtype=float)


def compute_hourly_lookup(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    hour_col: str = "Hour",
) -> Dict[int, float]:
    """
    Compute hourly mean demand lookup table from training data only.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    if hour_col not in X_train.columns:
        raise KeyError(f"Column '{hour_col}' not found in X_train.")

    df = pd.DataFrame({hour_col: X_train[hour_col].values, "target": np.asarray(y_train)})
    hourly_means = df.groupby(hour_col)["target"].mean().to_dict()
    return hourly_means


def predict_hourly_mean(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_eval: pd.DataFrame,
    hour_col: str = "Hour",
) -> np.ndarray:
    """
    Baseline 2: Predicts diurnal hour-of-day mean demand from training data.
    Falls back to overall training mean if a specific hour was absent in training.

    Parameters:
        X_train: Training features containing hour_col.
        y_train: Training target labels.
        X_eval: Evaluation features containing hour_col.
        hour_col: Column name representing hour of day (0-23).

    Returns:
        np.ndarray of shape (len(X_eval),) with hourly predictions.
    """
    if hour_col not in X_eval.columns:
        raise KeyError(f"Column '{hour_col}' not found in X_eval.")

    global_mean = calculate_training_mean(y_train)
    hourly_lookup = compute_hourly_lookup(X_train, y_train, hour_col=hour_col)

    hours = X_eval[hour_col].values
    preds = np.array([hourly_lookup.get(int(h), global_mean) for h in hours], dtype=float)
    return preds


def predict_known_closure(
    X_eval: pd.DataFrame,
    operating_predictions: np.ndarray,
    functioning_col: str = "Functioning Day",
    closed_val: str = "No",
) -> np.ndarray:
    """
    Baseline 3 / Operational Rule:
    For end-to-end evaluation:
    - If Functioning Day == 'No', predict exactly 0.0.
    - If Functioning Day == 'Yes', use operating_predictions.

    Parameters:
        X_eval: Evaluation features containing functioning_col.
        operating_predictions: Pre-computed operational predictions for all rows.
        functioning_col: Column name indicating operational status.
        closed_val: Value signifying closed/non-functioning status.

    Returns:
        np.ndarray of shape (len(X_eval),) with zero for closed hours.
    """
    if functioning_col not in X_eval.columns:
        raise KeyError(f"Column '{functioning_col}' not found in X_eval.")
    if len(X_eval) != len(operating_predictions):
        raise ValueError(
            f"Length mismatch: X_eval has {len(X_eval)} rows, operating_predictions has {len(operating_predictions)}."
        )

    is_closed = (X_eval[functioning_col].astype(str) == closed_val).values
    final_preds = np.where(is_closed, 0.0, operating_predictions)
    return final_preds


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate regression metrics: MAE, RMSE, and R^2.
    Evaluates only against provided evaluation labels.

    Returns:
        dict with keys: 'mae', 'rmse', 'r2', 'n_samples'
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true has {len(y_t)}, y_pred has {len(y_p)}.")
    if len(y_t) == 0:
        raise ValueError("Cannot evaluate metrics on empty arrays.")

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Compute R2; if target is constant, r2_score may emit warning or nan
    if np.all(y_t == y_t[0]):
        r2 = 0.0
    else:
        r2 = float(r2_score(y_t, y_p))

    return {
        "n_samples": int(len(y_t)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
