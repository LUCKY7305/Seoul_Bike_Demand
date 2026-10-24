"""
Dashboard Utility Module for Vehicle Rental Demand Prediction
=============================================================

Phase 4: Provides reusable, cached helper functions for:
- Model loading and deserialization verification.
- Processed and split data loading with Streamlit caching.
- User input validation and bound verification.
- Temporal and cyclical feature engineering matching Phase 2/3 definitions.
- Single-row DataFrame construction matching the champion pipeline schema.
- Demand prediction generation with closure rule handling and baseline comparison.
- Extraction of verified evaluation metrics and impurity-based feature importances.
"""

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Constants and Schema Definitions
# ---------------------------------------------------------------------------

# Robust project-root anchoring for local and Linux cloud deployment (Streamlit Cloud)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "phase3b_champion_pipeline.joblib"
DEFAULT_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "seoul_bike_processed.csv"
DEFAULT_TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
DEFAULT_SELECTION_PATH = PROJECT_ROOT / "reports" / "phase3b" / "model_selection.json"
DEFAULT_TEST_METRICS_PATH = PROJECT_ROOT / "reports" / "phase3b" / "final_test_metrics.json"

EXPECTED_SEASONS = ("Spring", "Summer", "Autumn", "Winter")
EXPECTED_HOLIDAYS = ("No Holiday", "Holiday")
EXPECTED_FUNCTIONING_DAYS = ("Yes", "No")

# Exact 18 numerical features expected by C_combined_cat
NUMERICAL_FEATURES: List[str] = [
    "Hour",
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
]

# Exact 2 categorical features expected by C_combined_cat
CATEGORICAL_FEATURES: List[str] = [
    "Seasons",
    "Holiday",
]

ALL_INPUT_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Model and Data Loaders
# ---------------------------------------------------------------------------


def load_champion_pipeline(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Pipeline:
    """
    Load the trained and verified Phase 3B champion pipeline.

    Parameters:
        model_path: Path to the .joblib pipeline file.

    Returns:
        Fitted Scikit-Learn Pipeline object.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        RuntimeError: If deserialization fails.
    """
    path = Path(model_path)
    if not path.is_absolute() and not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(
            f"Champion model artifact not found at: {path}. "
            "Please ensure Phase 3B model training has produced the pipeline artifact."
        )

    try:
        pipeline = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize model pipeline from {path}: {e}") from e

    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Loaded artifact is of type {type(pipeline)}, expected sklearn.pipeline.Pipeline.")

    return pipeline


def load_processed_data(data_path: Union[str, Path] = DEFAULT_PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Load the master processed dataset (8,760 rows x 23 columns).

    Parameters:
        data_path: Path to seoul_bike_processed.csv.

    Returns:
        DataFrame containing verified processed records.
    """
    path = Path(data_path)
    if not path.is_absolute() and not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Processed dataset file not found at: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    return df


def load_hourly_baseline_lookup(train_path: Union[str, Path] = DEFAULT_TRAIN_SPLIT_PATH) -> Dict[int, float]:
    """
    Compute hour-of-day mean demand lookup table from training operating hours.
    Used for live baseline comparison in the prediction interface.

    Parameters:
        train_path: Path to train.csv partition.

    Returns:
        Dict mapping hour integer (0-23) to average rental count.
    """
    path = Path(train_path)
    if not path.is_absolute() and not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Training split file not found at: {path}")

    train_df = pd.read_csv(path, encoding="utf-8")
    train_oper = train_df[train_df["Functioning Day"] == "Yes"]
    hourly_lookup = train_oper.groupby("Hour")["Rented Bike Count"].mean().to_dict()
    return hourly_lookup


# ---------------------------------------------------------------------------
# Feature Engineering and Validation
# ---------------------------------------------------------------------------


def validate_user_inputs(
    hour: int,
    temperature: float,
    humidity: float,
    wind_speed: float,
    visibility: float,
    dew_point: float,
    solar_radiation: float,
    rainfall: float,
    snowfall: float,
    season: str,
    holiday: str,
    functioning_day: str = "Yes",
) -> None:
    """
    Validate all raw user inputs against physical domains and dataset boundaries.
    Raises ValueError with a clear, user-friendly message if any check fails.
    """
    # Hour
    if not isinstance(hour, (int, np.integer)) or not (0 <= hour <= 23):
        raise ValueError(f"Hour must be an integer between 0 and 23. Got: {hour}")

    # Humidity
    if not (0.0 <= humidity <= 100.0):
        raise ValueError(f"Humidity must be a percentage between 0% and 100%. Got: {humidity}%")

    # Precipitation & Wind (cannot be negative)
    if rainfall < 0.0:
        raise ValueError(f"Rainfall cannot be negative. Got: {rainfall} mm")
    if snowfall < 0.0:
        raise ValueError(f"Snowfall cannot be negative. Got: {snowfall} cm")
    if wind_speed < 0.0:
        raise ValueError(f"Wind speed cannot be negative. Got: {wind_speed} m/s")
    if solar_radiation < 0.0:
        raise ValueError(f"Solar radiation cannot be negative. Got: {solar_radiation} MJ/m2")
    if visibility <= 0.0:
        raise ValueError(f"Visibility must be positive. Got: {visibility} (10m)")

    # Finite numeric checks
    for name, val in [
        ("Temperature", temperature),
        ("Dew point temperature", dew_point),
        ("Humidity", humidity),
        ("Wind speed", wind_speed),
        ("Visibility", visibility),
        ("Solar radiation", solar_radiation),
        ("Rainfall", rainfall),
        ("Snowfall", snowfall),
    ]:
        if not np.isfinite(val):
            raise ValueError(f"{name} must be a finite numeric value. Got: {val}")

    # Categorical domain checks
    if season not in EXPECTED_SEASONS:
        raise ValueError(f"Invalid season '{season}'. Must be one of: {list(EXPECTED_SEASONS)}")
    if holiday not in EXPECTED_HOLIDAYS:
        raise ValueError(f"Invalid holiday status '{holiday}'. Must be one of: {list(EXPECTED_HOLIDAYS)}")
    if functioning_day not in EXPECTED_FUNCTIONING_DAYS:
        raise ValueError(f"Invalid functioning day status '{functioning_day}'. Must be one of: {list(EXPECTED_FUNCTIONING_DAYS)}")


def construct_model_input_row(
    target_date: Union[date, datetime, str],
    hour: int,
    temperature: float,
    humidity: float,
    wind_speed: float,
    visibility: float,
    dew_point: float,
    solar_radiation: float,
    rainfall: float,
    snowfall: float,
    season: str,
    holiday: str,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame matching the exact schema and column order
    expected by the champion pipeline (Feature Set C_combined_cat).

    Applies the identical temporal and cyclical formulas from Phase 2:
    - day_of_week: Monday=0, Sunday=6
    - month: 1 to 12
    - day_of_month: 1 to 31
    - day_of_year: 1 to 365/366
    - is_weekend: 1 if day_of_week >= 5 else 0
    - hour_sin: sin(2 * pi * Hour / 24)
    - hour_cos: cos(2 * pi * Hour / 24)
    - month_sin: sin(2 * pi * month / 12)
    - month_cos: cos(2 * pi * month / 12)
    """
    if isinstance(target_date, str):
        parsed = pd.to_datetime(target_date)
    elif isinstance(target_date, datetime):
        parsed = target_date
    elif isinstance(target_date, date):
        parsed = datetime.combine(target_date, datetime.min.time())
    else:
        raise TypeError(f"Unsupported date type: {type(target_date)}")

    # Calendar features
    day_of_week = int(parsed.weekday())  # Monday=0, Sunday=6
    month = int(parsed.month)  # 1 to 12
    day_of_month = int(parsed.day)  # 1 to 31
    day_of_year = int(parsed.timetuple().tm_yday)  # 1 to 366
    is_weekend = 1 if day_of_week >= 5 else 0

    # Cyclical hour features (24-hour cycle)
    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))

    # Cyclical month features (12-month cycle)
    month_sin = float(np.sin(2.0 * np.pi * month / 12.0))
    month_cos = float(np.cos(2.0 * np.pi * month / 12.0))

    data_dict = {
        "Hour": [int(hour)],
        "day_of_week": [day_of_week],
        "month": [month],
        "day_of_month": [day_of_month],
        "day_of_year": [day_of_year],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "month_sin": [month_sin],
        "month_cos": [month_cos],
        "Temperature(°C)": [float(temperature)],
        "Humidity(%)": [float(humidity)],
        "Wind speed (m/s)": [float(wind_speed)],
        "Visibility (10m)": [float(visibility)],
        "Dew point temperature(°C)": [float(dew_point)],
        "Solar Radiation (MJ/m2)": [float(solar_radiation)],
        "Rainfall(mm)": [float(rainfall)],
        "Snowfall (cm)": [float(snowfall)],
        "Seasons": [str(season)],
        "Holiday": [str(holiday)],
    }

    df_row = pd.DataFrame(data_dict)
    # Ensure exact column ordering
    df_row = df_row[ALL_INPUT_FEATURES]
    return df_row


# ---------------------------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------------------------


def predict_demand(
    pipeline: Pipeline,
    input_row: pd.DataFrame,
    functioning_day: str = "Yes",
    hourly_lookup: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Generate demand prediction from the pipeline, respecting the deterministic
    closure rule (Scenario B) and comparing against the hourly baseline.

    Parameters:
        pipeline: Loaded Scikit-Learn champion pipeline.
        input_row: 1-row DataFrame containing the 20 features.
        functioning_day: 'Yes' for normal operations, 'No' for closure.
        hourly_lookup: Optional dict of hour-of-day mean demand.

    Returns:
        Dict with keys:
            - raw_prediction: Float regression output
            - display_prediction: Rounded non-negative integer count
            - is_closure: Boolean indicating administrative closure
            - baseline_prediction: Float hourly mean baseline for that hour
            - diff_vs_baseline: Difference between model prediction and baseline
    """
    hour = int(input_row["Hour"].iloc[0])
    baseline_val = hourly_lookup.get(hour, None) if hourly_lookup else None

    # Deterministic closure rule
    if functioning_day == "No":
        return {
            "raw_prediction": 0.0,
            "display_prediction": 0,
            "is_closure": True,
            "baseline_prediction": 0.0 if baseline_val is not None else None,
            "diff_vs_baseline": 0.0,
            "explanation": "Administrative closure: facilities are non-functioning, resulting in guaranteed 0 rentals.",
        }

    # Model inference for functioning hours
    raw_pred = float(pipeline.predict(input_row)[0])
    # Clamped non-negative rounded integer for display
    display_pred = max(0, int(round(raw_pred)))

    diff_base = None
    if baseline_val is not None:
        diff_base = raw_pred - baseline_val

    return {
        "raw_prediction": round(raw_pred, 2),
        "display_prediction": display_pred,
        "is_closure": False,
        "baseline_prediction": round(baseline_val, 2) if baseline_val is not None else None,
        "diff_vs_baseline": round(diff_base, 2) if diff_base is not None else None,
        "explanation": "Predicted demand estimated by Gradient Boosting Regressor trained on historical weather and temporal features.",
    }


# ---------------------------------------------------------------------------
# Model Performance and Metadata Extraction
# ---------------------------------------------------------------------------


def get_feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract impurity-based feature importances from the fitted champion pipeline.
    Recovers clean transformed feature names.

    Returns:
        DataFrame with columns ['feature', 'importance', 'percentage'] sorted descending.
    """
    prep = pipeline.named_steps["prep"]
    reg = pipeline.named_steps["reg"]

    feature_names = prep.get_feature_names_out()
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    importances = reg.feature_importances_

    df_imp = pd.DataFrame({
        "feature": clean_names,
        "importance": importances,
        "percentage": importances * 100.0,
    })
    df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df_imp


def get_verified_metrics() -> Dict[str, Any]:
    """
    Return the verified historical evaluation metrics from Phase 3B.
    """
    return {
        "validation": {
            "n_samples": 1163,
            "rmse": 321.58,
            "mae": 237.54,
            "r2": 0.7775,
        },
        "test_scenario_a": {
            "description": "Operating hours only (Functioning Day == 'Yes')",
            "n_samples": 1218,
            "rmse": 368.56,
            "mae": 260.94,
            "r2": 0.5432,
            "hourly_baseline_rmse": 413.23,
            "hourly_baseline_mae": 300.71,
            "pct_rmse_improvement": 10.81,
        },
        "test_scenario_b": {
            "description": "Full timeline with deterministic closure rule",
            "n_samples": 1314,
            "rmse": 354.84,
            "mae": 241.88,
            "r2": 0.6120,
            "closure_hourly_baseline_rmse": 397.85,
            "closure_hourly_baseline_mae": 278.74,
            "pct_rmse_improvement": 10.81,
        },
        "baseline_fairness": {
            "training_only_baseline_rmse": 413.23,
            "combined_data_baseline_rmse": 388.16,
            "champion_refitted_rmse": 368.56,
        },
    }
