"""
Experiment Design and Chronological Splitting Module
====================================================

Phase 3A: Implements reproducible experiment design for vehicle rental demand prediction:
1. Validates processed dataset integrity, temporal continuity, and zero-demand properties.
2. Creates deterministic, contiguous chronological train/validation/test partitions (70%/15%/15%).
3. Enforces strict future-data leakage safeguards (no shuffling, no target leakage, no test leakage).
4. Defines and exports validated feature sets (A: Temporal, B: Weather, C: Combined, D: Reduced Collinearity).
5. Executes non-ML baseline benchmarks on the VALIDATION set only (test set remains strictly held out).
6. Exports partition files, split metadata, feature sets JSON, and experiment configuration JSON.

Usage:
    python -m src.experiment_design --data "data/processed/seoul_bike_processed.csv"
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.baselines import (
    evaluate_predictions,
    predict_hourly_mean,
    predict_known_closure,
    predict_training_mean,
)

# ---------------------------------------------------------------------------
# Default Constants and Paths
# ---------------------------------------------------------------------------

DEFAULT_PROCESSED_PATH = "data/processed/seoul_bike_processed.csv"
SPLITS_DIR = Path("data/splits")
REPORTS_PHASE3A_DIR = Path("reports/phase3a")

TARGET_COLUMN = "Rented Bike Count"
DATE_COLUMN = "Date"
HOUR_COLUMN = "Hour"
FUNCTIONING_COLUMN = "Functioning Day"

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def file_md5(path: str) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_temperature_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Identify the exact column names for ambient temperature and dew point temperature."""
    temp_col = None
    dew_col = None
    for c in df.columns:
        if "Dew" in c:
            dew_col = c
        elif "Temperature" in c:
            temp_col = c
    if not temp_col or not dew_col:
        raise KeyError(
            f"Could not locate temperature columns in schema. Available: {list(df.columns)}"
        )
    return temp_col, dew_col


# ---------------------------------------------------------------------------
# Data Verification
# ---------------------------------------------------------------------------


def verify_processed_dataset(df: pd.DataFrame) -> Dict:
    """
    Verify processed dataset integrity:
    - 8,760 rows and 23 columns
    - Target column presence and non-negativity
    - No missing values
    - No duplicate (Date, Hour) combinations
    - Strict chronological order
    - Zero demand vs Functioning Day consistency
    - Temperature vs Dew Point correlation

    Raises ValueError if any assertion fails.
    """
    if len(df) != 8760:
        raise ValueError(f"Expected 8,760 rows in processed dataset, found {len(df)}.")
    if len(df.columns) != 23:
        raise ValueError(f"Expected 23 columns in processed dataset, found {len(df.columns)}.")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing from processed dataset.")
    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError(f"Negative values detected in target column '{TARGET_COLUMN}'.")

    # Missing values
    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        raise ValueError(f"Detected {missing_count} missing values in processed dataset.")

    # Duplicate timestamps
    ts_series = df[DATE_COLUMN].astype(str) + "_" + df[HOUR_COLUMN].astype(str)
    dup_ts = int(ts_series.duplicated().sum())
    if dup_ts > 0:
        raise ValueError(f"Detected {dup_ts} duplicate (Date, Hour) timestamps.")

    # Chronological monotonic check
    dt_series = pd.to_datetime(df[DATE_COLUMN]) + pd.to_timedelta(df[HOUR_COLUMN], unit="h")
    if not dt_series.is_monotonic_increasing:
        raise ValueError("Processed dataset observations are not strictly monotonic chronological.")

    # Zero-demand vs Functioning Day
    zero_mask = df[TARGET_COLUMN] == 0
    total_zeros = int(zero_mask.sum())
    if total_zeros != 295:
        raise ValueError(f"Expected 295 zero-demand records, found {total_zeros}.")

    func_no_zeros = int((df[zero_mask][FUNCTIONING_COLUMN] == "No").sum())
    if func_no_zeros != 295:
        raise ValueError(
            f"Expected all 295 zeros to have Functioning Day == 'No', found {func_no_zeros}."
        )

    func_yes_zeros = int((df[df[FUNCTIONING_COLUMN] == "Yes"][TARGET_COLUMN] == 0).sum())
    if func_yes_zeros != 0:
        raise ValueError(f"Expected 0 zeros when Functioning Day == 'Yes', found {func_yes_zeros}.")

    # Correlation between Temp and Dew point
    temp_col, dew_col = find_temperature_columns(df)
    temp_dew_corr = float(df[temp_col].corr(df[dew_col]))

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "target": TARGET_COLUMN,
        "date_min": str(df[DATE_COLUMN].min()),
        "date_max": str(df[DATE_COLUMN].max()),
        "total_zeros": total_zeros,
        "zeros_functioning_no": func_no_zeros,
        "zeros_functioning_yes": func_yes_zeros,
        "temp_col": temp_col,
        "dew_col": dew_col,
        "temp_dew_correlation": round(temp_dew_corr, 4),
    }


# ---------------------------------------------------------------------------
# Feature Sets Definition
# ---------------------------------------------------------------------------


def define_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Define, document, and validate experimental feature sets:
    - Feature Set A: Temporal features
    - Feature Set B: Weather features
    - Feature Set C: Temporal + Weather (A + B)
    - Feature Set D: Reduced-collinearity Temporal + Weather (C without Dew point)
    - Categorical Features: Seasons, Holiday
    - Scenario Features: Functioning Day (used for operational filtering/rules, not feature sets)

    Ensures target and raw Date string are excluded from all feature sets.
    """
    temp_col, dew_col = find_temperature_columns(df)

    feature_set_a = [
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
    ]

    feature_set_b = [
        temp_col,
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        dew_col,
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]

    feature_set_c = feature_set_a + feature_set_b
    feature_set_d = [c for c in feature_set_c if c != dew_col]

    categorical_features = ["Seasons", "Holiday"]
    scenario_features = [FUNCTIONING_COLUMN]

    feature_sets = {
        "FEATURE_SET_A_TEMPORAL": feature_set_a,
        "FEATURE_SET_B_WEATHER": feature_set_b,
        "FEATURE_SET_C_TEMPORAL_WEATHER": feature_set_c,
        "FEATURE_SET_D_REDUCED_COLLINEARITY": feature_set_d,
        "CATEGORICAL_FEATURES": categorical_features,
        "SCENARIO_FEATURES": scenario_features,
    }

    # Validate against DataFrame columns
    for name, cols in feature_sets.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Feature set '{name}' references columns not in dataset: {missing}")
        if TARGET_COLUMN in cols:
            raise ValueError(f"[LEAKAGE ERROR] Target '{TARGET_COLUMN}' in feature set '{name}'.")
        if DATE_COLUMN in cols:
            raise ValueError(f"Raw string '{DATE_COLUMN}' cannot be included in feature set '{name}'.")

    return feature_sets


# ---------------------------------------------------------------------------
# Chronological Splitting
# ---------------------------------------------------------------------------


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Deterministically split the chronologically sorted dataset into:
    - Train (earliest train_ratio, default 70%)
    - Validation (next val_ratio, default 15%)
    - Test (latest remaining, default 15%)

    Splits the complete timeline FIRST, preserving contiguous chronological blocks.
    Rejects shuffling or random splitting.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split partition sizes: train={n_train}, val={n_val}, test={n_test}."
        )

    # Contiguous slice partitions
    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # Chronological boundary verification
    dt_train_max = pd.to_datetime(train_df[DATE_COLUMN].iloc[-1]) + pd.to_timedelta(
        train_df[HOUR_COLUMN].iloc[-1], unit="h"
    )
    dt_val_min = pd.to_datetime(val_df[DATE_COLUMN].iloc[0]) + pd.to_timedelta(
        val_df[HOUR_COLUMN].iloc[0], unit="h"
    )
    dt_val_max = pd.to_datetime(val_df[DATE_COLUMN].iloc[-1]) + pd.to_timedelta(
        val_df[HOUR_COLUMN].iloc[-1], unit="h"
    )
    dt_test_min = pd.to_datetime(test_df[DATE_COLUMN].iloc[0]) + pd.to_timedelta(
        test_df[HOUR_COLUMN].iloc[0], unit="h"
    )

    if not (dt_train_max < dt_val_min):
        raise ValueError(f"Temporal overlap: Train end ({dt_train_max}) >= Val start ({dt_val_min})")
    if not (dt_val_max < dt_test_min):
        raise ValueError(f"Temporal overlap: Val end ({dt_val_max}) >= Test start ({dt_test_min})")

    # Detailed partition metadata
    def partition_summary(part: pd.DataFrame, name: str) -> Dict:
        target = part[TARGET_COLUMN]
        func_counts = part[FUNCTIONING_COLUMN].value_counts().to_dict()
        zero_counts = int((target == 0).sum())
        start_ts = f"{part[DATE_COLUMN].iloc[0]} {part[HOUR_COLUMN].iloc[0]:02d}:00"
        end_ts = f"{part[DATE_COLUMN].iloc[-1]} {part[HOUR_COLUMN].iloc[-1]:02d}:00"
        return {
            "name": name,
            "row_count": len(part),
            "percentage": round(len(part) / n * 100, 2),
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "functioning_yes_count": func_counts.get("Yes", 0),
            "functioning_no_count": func_counts.get("No", 0),
            "zero_demand_count": zero_counts,
            "target_mean": round(float(target.mean()), 2),
            "target_median": round(float(target.median()), 2),
            "target_std": round(float(target.std()), 2),
            "target_min": round(float(target.min()), 2),
            "target_max": round(float(target.max()), 2),
            "operational_target_mean": round(
                float(part[part[FUNCTIONING_COLUMN] == "Yes"][TARGET_COLUMN].mean()), 2
            )
            if func_counts.get("Yes", 0) > 0
            else 0.0,
        }

    split_meta = {
        "total_rows": n,
        "target_column": TARGET_COLUMN,
        "train_partition": partition_summary(train_df, "train"),
        "validation_partition": partition_summary(val_df, "validation"),
        "test_partition": partition_summary(test_df, "test"),
        "proportions": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 2),
        },
    }

    return train_df, val_df, test_df, split_meta


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_meta: Dict,
    raw_path: str,
    splits_dir: Path = SPLITS_DIR,
) -> Dict[str, str]:
    """Save split CSVs and split metadata JSON with input fingerprint."""
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "validation.csv"
    test_path = splits_dir / "test.csv"
    meta_path = splits_dir / "split_metadata.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    # Add checksums and paths to metadata
    split_meta["input_file_path"] = raw_path
    split_meta["input_file_md5"] = file_md5(raw_path) if os.path.exists(raw_path) else None
    split_meta["train_file_md5"] = file_md5(str(train_path))
    split_meta["validation_file_md5"] = file_md5(str(val_path))
    split_meta["test_file_md5"] = file_md5(str(test_path))
    split_meta["generated_at"] = datetime.now().isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    return {
        "train_csv": str(train_path),
        "validation_csv": str(val_path),
        "test_csv": str(test_path),
        "metadata_json": str(meta_path),
    }


# ---------------------------------------------------------------------------
# Baseline Experiments on Validation Data Only
# ---------------------------------------------------------------------------


def run_validation_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Execute baseline predictions strictly on VALIDATION data.
    The test partition is NOT evaluated here to prevent data leakage.

    Evaluates:
    - Scenario A (Operational hours only: Functioning Day == 'Yes'):
        * Baseline 1: Training Mean
        * Baseline 2: Hour-of-Day Mean
    - Scenario B (End-to-end full timeline: All hours):
        * Baseline 3: Known Closure Rule + Training Mean
        * Baseline 3 (Variant): Known Closure Rule + Hourly Mean
    """
    # 1. Operational-only subsets (Scenario A)
    train_oper = train_df[train_df[FUNCTIONING_COLUMN] == "Yes"]
    val_oper = val_df[val_df[FUNCTIONING_COLUMN] == "Yes"]

    y_train_oper = train_oper[TARGET_COLUMN].values
    y_val_oper = val_oper[TARGET_COLUMN].values

    # Baseline 1: Training Mean on Scenario A
    b1_preds = predict_training_mean(y_train_oper, n_samples=len(val_oper))
    b1_metrics = evaluate_predictions(y_val_oper, b1_preds)

    # Baseline 2: Hour-of-Day Mean on Scenario A
    b2_preds = predict_hourly_mean(train_oper, y_train_oper, val_oper, hour_col=HOUR_COLUMN)
    b2_metrics = evaluate_predictions(y_val_oper, b2_preds)

    # 2. End-to-end full timeline (Scenario B)
    y_val_all = val_df[TARGET_COLUMN].values

    # Baseline 3: Closure Rule + Training Mean
    train_mean_val = float(np.mean(y_train_oper))
    b3_oper_preds = np.full(shape=len(val_df), fill_value=train_mean_val)
    b3_preds = predict_known_closure(
        val_df, b3_oper_preds, functioning_col=FUNCTIONING_COLUMN, closed_val="No"
    )
    b3_metrics = evaluate_predictions(y_val_all, b3_preds)

    # Baseline 3 Variant: Closure Rule + Hourly Mean
    b3_hourly_oper_preds = predict_hourly_mean(
        train_oper, y_train_oper, val_df, hour_col=HOUR_COLUMN
    )
    b3_hourly_preds = predict_known_closure(
        val_df, b3_hourly_oper_preds, functioning_col=FUNCTIONING_COLUMN, closed_val="No"
    )
    b3_hourly_metrics = evaluate_predictions(y_val_all, b3_hourly_preds)

    results = {
        "scenario_a_operational": {
            "evaluation_population": "Functioning Day == 'Yes'",
            "validation_samples": len(val_oper),
            "baseline_1_training_mean": {
                "strategy": "Constant operating training mean",
                "training_mean_value": round(train_mean_val, 4),
                **b1_metrics,
            },
            "baseline_2_hourly_mean": {
                "strategy": "Hour-of-day mean lookup from operating training data",
                **b2_metrics,
            },
        },
        "scenario_b_end_to_end": {
            "evaluation_population": "Complete held-out validation timeline",
            "validation_samples": len(val_df),
            "baseline_3_closure_rule_with_mean": {
                "strategy": "Predict 0 if Functioning Day == 'No', else training mean",
                **b3_metrics,
            },
            "baseline_3_closure_rule_with_hourly": {
                "strategy": "Predict 0 if Functioning Day == 'No', else hour-of-day mean",
                **b3_hourly_metrics,
            },
        },
        "test_set_status": "Strictly held out; zero test metrics calculated in Phase 3A.",
    }

    return results


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------


def run_experiment_design_pipeline(
    data_path: str = DEFAULT_PROCESSED_PATH,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Dict:
    """
    Full Phase 3A runner:
    1. Loads and verifies processed dataset.
    2. Defines and saves feature sets.
    3. Splits dataset chronologically and saves partition CSVs.
    4. Evaluates baselines on validation set only.
    5. Exports experiment_config.json.
    """
    print(f"[INFO] Loading processed data from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"       Loaded {len(df)} rows, {len(df.columns)} columns.")

    print("[INFO] Verifying dataset integrity against Phase 2 benchmarks...")
    verif = verify_processed_dataset(df)
    print(f"       Rows: {verif['n_rows']}, Cols: {verif['n_cols']}, Target: {verif['target']}")
    print(
        f"       Zero demand: {verif['total_zeros']} (all in Functioning Day == 'No': {verif['zeros_functioning_no']})"
    )
    print(
        f"       Corr({verif['temp_col']}, {verif['dew_col']}) = {verif['temp_dew_correlation']:.4f}"
    )

    print("[INFO] Defining and validating experimental feature sets...")
    feature_sets = define_feature_sets(df)
    REPORTS_PHASE3A_DIR.mkdir(parents=True, exist_ok=True)
    feature_sets_path = REPORTS_PHASE3A_DIR / "feature_sets.json"
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, indent=2)
    print(f"       Exported feature sets to: {feature_sets_path}")

    print("[INFO] Creating deterministic chronological train/val/test partitions...")
    train_df, val_df, test_df, split_meta = create_chronological_splits(df, train_ratio, val_ratio)
    split_paths = save_splits(train_df, val_df, test_df, split_meta, data_path)
    print(
        f"       Train: {split_meta['train_partition']['row_count']} rows [{split_meta['train_partition']['start_timestamp']} -> {split_meta['train_partition']['end_timestamp']}]"
    )
    print(
        f"       Val:   {split_meta['validation_partition']['row_count']} rows [{split_meta['validation_partition']['start_timestamp']} -> {split_meta['validation_partition']['end_timestamp']}]"
    )
    print(
        f"       Test:  {split_meta['test_partition']['row_count']} rows [{split_meta['test_partition']['start_timestamp']} -> {split_meta['test_partition']['end_timestamp']}]"
    )
    print(f"       Split metadata saved to: {split_paths['metadata_json']}")

    print("[INFO] Running baseline benchmarks on VALIDATION partition only...")
    baseline_results = run_validation_baselines(train_df, val_df)
    print("       Validation baseline benchmarks completed.")
    print(
        f"       Baseline 1 (Mean, Scenario A): MAE={baseline_results['scenario_a_operational']['baseline_1_training_mean']['mae']:.2f}, RMSE={baseline_results['scenario_a_operational']['baseline_1_training_mean']['rmse']:.2f}, R2={baseline_results['scenario_a_operational']['baseline_1_training_mean']['r2']:.4f}"
    )
    print(
        f"       Baseline 2 (Hour, Scenario A): MAE={baseline_results['scenario_a_operational']['baseline_2_hourly_mean']['mae']:.2f}, RMSE={baseline_results['scenario_a_operational']['baseline_2_hourly_mean']['rmse']:.2f}, R2={baseline_results['scenario_a_operational']['baseline_2_hourly_mean']['r2']:.4f}"
    )
    print(
        f"       Baseline 3 (Rule+Hour, Scenario B): MAE={baseline_results['scenario_b_end_to_end']['baseline_3_closure_rule_with_hourly']['mae']:.2f}, RMSE={baseline_results['scenario_b_end_to_end']['baseline_3_closure_rule_with_hourly']['rmse']:.2f}, R2={baseline_results['scenario_b_end_to_end']['baseline_3_closure_rule_with_hourly']['r2']:.4f}"
    )

    # Master experiment config
    experiment_config = {
        "project": "Vehicle Rental Demand Prediction Using Weather and Temporal Features",
        "phase": "Phase 3A — Experimental Design, Chronological Splitting, and Baselines",
        "timestamp": datetime.now().isoformat(),
        "dataset_verification": verif,
        "split_metadata": split_meta,
        "feature_sets": feature_sets,
        "validation_baseline_results": baseline_results,
        "test_set_policy": "Strictly held out. Zero predictions or evaluation metrics computed on test split in Phase 3A.",
        "phase_3b_models_planned": [
            "Linear Regression (Ridge)",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting / XGBoost Regressor",
        ],
    }

    config_path = REPORTS_PHASE3A_DIR / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"[INFO] Saved master experiment configuration to: {config_path}")
    print("[INFO] Phase 3A pipeline completed successfully.")

    return experiment_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3A: Reproducible experiment design, chronological splitting, and baselines."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_PROCESSED_PATH,
        help=f"Path to processed CSV dataset (default: {DEFAULT_PROCESSED_PATH})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Proportion for training partition (default: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Proportion for validation partition (default: {DEFAULT_VAL_RATIO})",
    )
    args = parser.parse_args()

    try:
        run_experiment_design_pipeline(
            data_path=args.data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    except Exception as e:
        print(f"[ERROR] Phase 3A experiment design failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
