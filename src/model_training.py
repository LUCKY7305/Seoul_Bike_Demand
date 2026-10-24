"""
Machine Learning Pipelines, Training, Validation, and Evaluation Module
=======================================================================

Phase 3B: Implements end-to-end reproducible machine learning regression:
1. Loads verified chronological split partitions (Train, Validation, Test).
2. Generates 8 feature set configurations (Temporal, Weather, Combined, Reduced-Collinearity +/- Categoricals).
3. Constructs leakage-safe Scikit-Learn ColumnTransformer pipelines.
4. Trains and evaluates candidate regression models:
   - Linear Regression
   - Ridge Regression (regularized linear)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor (sklearn)
   - XGBoost Regressor
5. Evaluates validation metrics (RMSE, MAE, R^2) against Phase 3A baselines.
6. Locks the champion model configuration to `reports/phase3b/model_selection.json`.
7. Generates validation diagnostic figures in `reports/figures/phase3b/`.
8. Refits the locked champion pipeline on combined train + validation operating hours.
9. Performs final test evaluation on the untouched test partition (Scenario A & Scenario B).
10. Persists the champion pipeline to `models/phase3b_champion_pipeline.joblib`.

Usage:
    python -m src.model_training
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from src.baselines import (
    calculate_training_mean,
    compute_hourly_lookup,
    evaluate_predictions,
    predict_hourly_mean,
    predict_known_closure,
    predict_training_mean,
)

# ---------------------------------------------------------------------------
# Constants and Paths
# ---------------------------------------------------------------------------

SPLITS_DIR = Path("data/splits")
REPORTS_PHASE3A_DIR = Path("reports/phase3a")
REPORTS_PHASE3B_DIR = Path("reports/phase3b")
FIGURES_PHASE3B_DIR = Path("reports/figures/phase3b")
MODELS_DIR = Path("models")

TARGET_COLUMN = "Rented Bike Count"
DATE_COLUMN = "Date"
HOUR_COLUMN = "Hour"
FUNCTIONING_COLUMN = "Functioning Day"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data Loading and Filtering
# ---------------------------------------------------------------------------


def file_md5(path: str) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_partitions() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load the approved Phase 3A chronological split CSVs and metadata.
    Returns (train_df, val_df, test_df, metadata_dict).
    """
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"
    meta_path = SPLITS_DIR / "split_metadata.json"

    for p in [train_path, val_path, test_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required Phase 3A partition missing: {p}")

    train_df = pd.read_csv(train_path, encoding="utf-8")
    val_df = pd.read_csv(val_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Verify partition row counts
    assert len(train_df) == 6132, f"Expected 6,132 train rows, got {len(train_df)}"
    assert len(val_df) == 1314, f"Expected 1,314 val rows, got {len(val_df)}"
    assert len(test_df) == 1314, f"Expected 1,314 test rows, got {len(test_df)}"

    return train_df, val_df, test_df, metadata


def filter_operating_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter partition to operating hours (Scenario A: Functioning Day == 'Yes').
    Preserves row order and indices reset.
    """
    oper_df = df[df[FUNCTIONING_COLUMN] == "Yes"].copy().reset_index(drop=True)
    return oper_df


# ---------------------------------------------------------------------------
# Feature Configurations
# ---------------------------------------------------------------------------


def build_experiment_manifest(feature_sets_path: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Construct the 8 experimental feature configurations from Phase 3A definitions:
    1. A_temporal: Temporal only (10 features)
    2. A_temporal_cat: Temporal + Categorical (10 numeric + 2 categorical = 12)
    3. B_weather: Weather only (8 features)
    4. B_weather_cat: Weather + Categorical (8 numeric + 2 categorical = 10)
    5. C_combined: Combined Temporal + Weather (18 features)
    6. C_combined_cat: Combined + Categorical (18 numeric + 2 categorical = 20)
    7. D_reduced: Reduced Collinearity (17 features, C without Dew point)
    8. D_reduced_cat: Reduced Collinearity + Categorical (17 numeric + 2 categorical = 19)

    Saves and returns the experiment manifest dictionary.
    """
    if feature_sets_path is None:
        feature_sets_path = REPORTS_PHASE3A_DIR / "feature_sets.json"

    if not feature_sets_path.exists():
        raise FileNotFoundError(f"Missing feature sets definition: {feature_sets_path}")

    with open(feature_sets_path, "r", encoding="utf-8") as f:
        base_sets = json.load(f)

    set_a = base_sets["FEATURE_SET_A_TEMPORAL"]
    set_b = base_sets["FEATURE_SET_B_WEATHER"]
    set_c = base_sets["FEATURE_SET_C_TEMPORAL_WEATHER"]
    set_d = base_sets["FEATURE_SET_D_REDUCED_COLLINEARITY"]
    cat_cols = base_sets["CATEGORICAL_FEATURES"]

    configs = {
        "A_temporal": {
            "name": "Feature Set A (Temporal Only)",
            "num_features": set_a,
            "cat_features": [],
            "total_input_cols": len(set_a),
            "description": "Diurnal, day-of-week, and annual cyclical features only.",
        },
        "A_temporal_cat": {
            "name": "Feature Set A + Categorical",
            "num_features": set_a,
            "cat_features": cat_cols,
            "total_input_cols": len(set_a) + len(cat_cols),
            "description": "Temporal features plus Seasons and Holiday categoricals.",
        },
        "B_weather": {
            "name": "Feature Set B (Weather Only)",
            "num_features": set_b,
            "cat_features": [],
            "total_input_cols": len(set_b),
            "description": "Continuous meteorological measurements only.",
        },
        "B_weather_cat": {
            "name": "Feature Set B + Categorical",
            "num_features": set_b,
            "cat_features": cat_cols,
            "total_input_cols": len(set_b) + len(cat_cols),
            "description": "Weather measurements plus Seasons and Holiday categoricals.",
        },
        "C_combined": {
            "name": "Feature Set C (Temporal + Weather)",
            "num_features": set_c,
            "cat_features": [],
            "total_input_cols": len(set_c),
            "description": "All continuous temporal and weather features.",
        },
        "C_combined_cat": {
            "name": "Feature Set C + Categorical",
            "num_features": set_c,
            "cat_features": cat_cols,
            "total_input_cols": len(set_c) + len(cat_cols),
            "description": "All continuous features plus Seasons and Holiday.",
        },
        "D_reduced": {
            "name": "Feature Set D (Reduced Collinearity)",
            "num_features": set_d,
            "cat_features": [],
            "total_input_cols": len(set_d),
            "description": "Feature Set C omitting Dew point temperature (r=0.9128 with Temp).",
        },
        "D_reduced_cat": {
            "name": "Feature Set D + Categorical",
            "num_features": set_d,
            "cat_features": cat_cols,
            "total_input_cols": len(set_d) + len(cat_cols),
            "description": "Feature Set D plus Seasons and Holiday.",
        },
    }

    # Integrity verification
    for cid, cinfo in configs.items():
        all_cols = cinfo["num_features"] + cinfo["cat_features"]
        if TARGET_COLUMN in all_cols:
            raise ValueError(f"Leakage: target in config {cid}")
        if DATE_COLUMN in all_cols:
            raise ValueError(f"Leakage: raw Date in config {cid}")
        if FUNCTIONING_COLUMN in all_cols:
            raise ValueError(f"Leakage: Functioning Day in config {cid}")

    REPORTS_PHASE3B_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS_PHASE3B_DIR / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)

    return configs


# ---------------------------------------------------------------------------
# Pipeline Construction
# ---------------------------------------------------------------------------


def create_pipeline(
    model_family: str,
    num_cols: List[str],
    cat_cols: List[str],
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Construct a complete, leakage-safe Scikit-Learn Pipeline:
    - Preprocessing: ColumnTransformer with StandardScaler for linear models,
      passthrough for tree models, and OneHotEncoder for categoricals.
    - Regressor: initialized with supplied hyperparameters.
    """
    if hyperparameters is None:
        hyperparameters = {}

    is_linear = model_family in ["LinearRegression", "Ridge"]

    transformers = []
    if num_cols:
        num_transformer = StandardScaler() if is_linear else "passthrough"
        transformers.append(("num", num_transformer, num_cols))

    if cat_cols:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Instantiate regressor
    if model_family == "LinearRegression":
        regressor = LinearRegression(**hyperparameters)
    elif model_family == "Ridge":
        params = {"random_state": RANDOM_STATE, **hyperparameters}
        regressor = Ridge(**params)
    elif model_family == "DecisionTree":
        params = {"random_state": RANDOM_STATE, **hyperparameters}
        regressor = DecisionTreeRegressor(**params)
    elif model_family == "RandomForest":
        params = {"random_state": RANDOM_STATE, "n_jobs": -1, **hyperparameters}
        regressor = RandomForestRegressor(**params)
    elif model_family == "GradientBoosting":
        params = {"random_state": RANDOM_STATE, **hyperparameters}
        regressor = GradientBoostingRegressor(**params)
    elif model_family == "XGBoost":
        params = {"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0, **hyperparameters}
        regressor = xgb.XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("reg", regressor),
    ])

    return pipe


# ---------------------------------------------------------------------------
# Candidate Model Definitions
# ---------------------------------------------------------------------------


def get_candidate_models() -> List[Dict[str, Any]]:
    """
    Define candidate models and hyperparameter grids across the core model families:
    1. Linear Regression (baseline parametric model)
    2. Ridge Regression (regularized linear with alpha=0.1, 1.0, 10.0)
    3. Decision Tree (default, depth=10, depth=5)
    4. Random Forest (100 trees, depth=None, depth=20, depth=10)
    5. Gradient Boosting (sklearn, 100 trees, depth=3, depth=5, lr=0.05)
    6. XGBoost (100-150 trees, depth=5, lr=0.08)
    """
    candidates = [
        # Linear Models
        {"family": "LinearRegression", "tag": "default", "params": {}},
        {"family": "Ridge", "tag": "alpha_0.1", "params": {"alpha": 0.1}},
        {"family": "Ridge", "tag": "alpha_1.0", "params": {"alpha": 1.0}},
        {"family": "Ridge", "tag": "alpha_10.0", "params": {"alpha": 10.0}},

        # Decision Tree Regressors
        {"family": "DecisionTree", "tag": "default", "params": {"max_depth": None, "min_samples_leaf": 1}},
        {"family": "DecisionTree", "tag": "depth_10", "params": {"max_depth": 10, "min_samples_leaf": 5}},
        {"family": "DecisionTree", "tag": "depth_5", "params": {"max_depth": 5, "min_samples_leaf": 10}},

        # Random Forest Regressors
        {"family": "RandomForest", "tag": "default_100", "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}},
        {"family": "RandomForest", "tag": "depth_20", "params": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2}},
        {"family": "RandomForest", "tag": "depth_10", "params": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5}},

        # Gradient Boosting Regressors (sklearn)
        {"family": "GradientBoosting", "tag": "default_100", "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}},
        {"family": "GradientBoosting", "tag": "depth_5", "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}},
        {"family": "GradientBoosting", "tag": "slow_lr", "params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4}},

        # XGBoost Regressors
        {"family": "XGBoost", "tag": "default_100", "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}},
        {"family": "XGBoost", "tag": "tuned_150", "params": {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 6}},
    ]
    return candidates


# ---------------------------------------------------------------------------
# Training and Validation Runner
# ---------------------------------------------------------------------------


def train_and_evaluate_candidates(
    train_oper: pd.DataFrame,
    val_oper: pd.DataFrame,
    manifest: Dict[str, Dict],
    candidates: List[Dict[str, Any]],
    baseline_hourly_rmse: float = 600.9553,
    baseline_hourly_mae: float = 473.9518,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Train every candidate configuration on operating-hour training data,
    predict on operating-hour validation data, and record comparative metrics.

    Does NOT inspect test data!
    Returns (results_dataframe, trained_pipelines_dict).
    """
    y_train = train_oper[TARGET_COLUMN].values
    y_val = val_oper[TARGET_COLUMN].values

    records = []
    fitted_pipelines = {}

    total_runs = len(manifest) * len(candidates)
    run_idx = 0

    print(f"[INFO] Beginning candidate training: {len(manifest)} feature sets x {len(candidates)} models = {total_runs} candidate runs.")

    for cfg_id, cfg_info in manifest.items():
        num_cols = cfg_info["num_features"]
        cat_cols = cfg_info["cat_features"]
        feature_cols = num_cols + cat_cols

        X_train = train_oper[feature_cols]
        X_val = val_oper[feature_cols]

        for cand in candidates:
            run_idx += 1
            family = cand["family"]
            tag = cand["tag"]
            params = cand["params"]
            candidate_id = f"{family}__{tag}__{cfg_id}"

            t0 = time.time()
            try:
                pipe = create_pipeline(family, num_cols, cat_cols, params)
                pipe.fit(X_train, y_train)
                fit_time = time.time() - t0

                # Predict validation
                val_preds = pipe.predict(X_val)

                # Clamp negative predictions to 0 for realistic demand evaluation
                # but record unclipped metrics as standard regression evaluation
                metrics = evaluate_predictions(y_val, val_preds)
                mae = metrics["mae"]
                rmse = metrics["rmse"]
                r2 = metrics["r2"]

                # Improvement vs Baseline 2 (Hour-of-Day Mean)
                mae_diff = mae - baseline_hourly_mae  # negative is improvement
                rmse_diff = rmse - baseline_hourly_rmse
                pct_rmse_imp = ((baseline_hourly_rmse - rmse) / baseline_hourly_rmse) * 100.0
                beats_baseline = rmse < baseline_hourly_rmse

                record = {
                    "candidate_id": candidate_id,
                    "model_family": family,
                    "model_tag": tag,
                    "feature_config": cfg_id,
                    "feature_config_name": cfg_info["name"],
                    "has_categoricals": len(cat_cols) > 0,
                    "num_features_count": len(num_cols),
                    "cat_features_count": len(cat_cols),
                    "total_features_count": len(feature_cols),
                    "hyperparameters": json.dumps(params),
                    "fit_time_sec": round(fit_time, 3),
                    "val_samples": len(val_oper),
                    "val_mae": round(mae, 2),
                    "val_rmse": round(rmse, 2),
                    "val_r2": round(r2, 4),
                    "mae_diff_vs_hourly_base": round(mae_diff, 2),
                    "rmse_diff_vs_hourly_base": round(rmse_diff, 2),
                    "pct_rmse_improvement": round(pct_rmse_imp, 2),
                    "beats_hourly_baseline": beats_baseline,
                    "fit_status": "SUCCESS",
                    "error_message": None,
                }
                fitted_pipelines[candidate_id] = pipe

                if run_idx % 15 == 0 or run_idx == total_runs:
                    print(f"       [{run_idx:02d}/{total_runs}] {candidate_id:<45} | RMSE: {rmse:.2f} | R2: {r2:.4f} | Beats Base: {beats_baseline}")

            except Exception as e:
                fit_time = time.time() - t0
                record = {
                    "candidate_id": candidate_id,
                    "model_family": family,
                    "model_tag": tag,
                    "feature_config": cfg_id,
                    "feature_config_name": cfg_info["name"],
                    "has_categoricals": len(cat_cols) > 0,
                    "num_features_count": len(num_cols),
                    "cat_features_count": len(cat_cols),
                    "total_features_count": len(feature_cols),
                    "hyperparameters": json.dumps(params),
                    "fit_time_sec": round(fit_time, 3),
                    "val_samples": len(val_oper),
                    "val_mae": None,
                    "val_rmse": float("inf"),
                    "val_r2": None,
                    "mae_diff_vs_hourly_base": None,
                    "rmse_diff_vs_hourly_base": None,
                    "pct_rmse_improvement": None,
                    "beats_hourly_baseline": False,
                    "fit_status": "FAILED",
                    "error_message": str(e),
                }
                print(f"       [!] FAILED: {candidate_id} -> {e}")

            records.append(record)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="val_rmse", ascending=True).reset_index(drop=True)

    # Export candidate results table
    REPORTS_PHASE3B_DIR.mkdir(parents=True, exist_ok=True)
    results_csv_path = REPORTS_PHASE3B_DIR / "candidate_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved complete candidate results ({len(results_df)} runs) to: {results_csv_path}")

    return results_df, fitted_pipelines


# ---------------------------------------------------------------------------
# Model Selection and Lock
# ---------------------------------------------------------------------------


def lock_model_selection(
    results_df: pd.DataFrame,
    manifest: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Select the best model using the declared primary selection metric (Validation RMSE).
    Locks the selection record to JSON BEFORE any test evaluation.
    """
    if output_path is None:
        output_path = REPORTS_PHASE3B_DIR / "model_selection.json"

    successful = results_df[results_df["fit_status"] == "SUCCESS"].copy()
    if successful.empty:
        raise RuntimeError("No candidate models trained successfully!")

    # Sort strictly by validation RMSE ascending
    successful = successful.sort_values(by=["val_rmse", "val_mae"], ascending=[True, True]).reset_index(drop=True)
    champion = successful.iloc[0].to_dict()

    cfg_id = champion["feature_config"]
    cfg_info = manifest[cfg_id]

    lock_record = {
        "selection_status": "LOCKED",
        "lock_timestamp": datetime.now().isoformat(),
        "primary_selection_metric": "val_rmse (ascending)",
        "secondary_selection_metric": "val_mae (ascending)",
        "champion_candidate_id": champion["candidate_id"],
        "model_family": champion["model_family"],
        "model_tag": champion["model_tag"],
        "hyperparameters": json.loads(champion["hyperparameters"]),
        "feature_config": cfg_id,
        "feature_config_name": cfg_info["name"],
        "numerical_features": cfg_info["num_features"],
        "categorical_features": cfg_info["cat_features"],
        "validation_metrics": {
            "val_samples": champion["val_samples"],
            "val_rmse": champion["val_rmse"],
            "val_mae": champion["val_mae"],
            "val_r2": champion["val_r2"],
            "pct_rmse_improvement_vs_hourly_base": champion["pct_rmse_improvement"],
        },
        "ranking_context": {
            "total_candidates_evaluated": len(results_df),
            "successful_candidates": len(successful),
            "top_5_candidates": successful.head(5)[["candidate_id", "val_rmse", "val_mae", "val_r2"]].to_dict(orient="records"),
        },
        "test_evaluation_authorized": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lock_record, f, indent=2)

    print(f"[INFO] MODEL SELECTION LOCKED: {champion['candidate_id']}")
    print(f"       Validation RMSE: {champion['val_rmse']:.2f} | MAE: {champion['val_mae']:.2f} | R2: {champion['val_r2']:.4f}")
    print(f"       Selection lock saved to: {output_path}")

    return lock_record


# ---------------------------------------------------------------------------
# Validation Diagnostics
# ---------------------------------------------------------------------------


def generate_validation_diagnostics(
    champion_pipe: Pipeline,
    val_oper: pd.DataFrame,
    feature_cols: List[str],
    results_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> List[str]:
    """
    Generate 6 publication-grade diagnostic figures on the VALIDATION partition:
    1. Actual vs Predicted demand.
    2. Residuals vs Predicted demand.
    3. Residual distribution.
    4. Actual & Predicted over chronological validation window.
    5. Model family comparison bar chart.
    6. Feature importance / permutation analysis.

    Does NOT inspect test data!
    """
    if out_dir is None:
        out_dir = FIGURES_PHASE3B_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["figure.dpi"] = 150

    y_val = val_oper[TARGET_COLUMN].values
    X_val = val_oper[feature_cols]
    y_pred = champion_pipe.predict(X_val)
    residuals = y_val - y_pred

    generated_paths = []

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_val, y_pred, alpha=0.25, color="#2980b9", s=18, edgecolors="none", rasterized=True)
    max_val = max(float(y_val.max()), float(y_pred.max())) + 100
    ax.plot([0, max_val], [0, max_val], color="#e74c3c", linestyle="--", linewidth=2.0, label="Ideal 1:1 Reference")
    ax.set_title("Figure 1: Actual vs. Predicted Demand (Validation Operating Hours)", fontweight="bold")
    ax.set_xlabel("Actual Rented Bike Count")
    ax.set_ylabel("Predicted Rented Bike Count")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "01_actual_vs_predicted_val.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_paths.append(str(p1))

    # 2. Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.25, color="#8e44ad", s=18, edgecolors="none", rasterized=True)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Figure 2: Residuals vs. Predicted Demand (Validation Operating Hours)", fontweight="bold")
    ax.set_xlabel("Predicted Rented Bike Count")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    fig.tight_layout()
    p2 = out_dir / "02_residuals_vs_predicted_val.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_paths.append(str(p2))

    # 3. Residual Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#27ae60", bins=35, ax=ax, edgecolor="black", alpha=0.6)
    ax.axvline(float(np.mean(residuals)), color="#d9534f", linestyle="--", label=f"Mean Residual: {np.mean(residuals):.1f}")
    ax.axvline(float(np.median(residuals)), color="#2980b9", linestyle="-.", label=f"Median Residual: {np.median(residuals):.1f}")
    ax.set_title("Figure 3: Distribution of Prediction Residuals (Validation Set)", fontweight="bold")
    ax.set_xlabel("Residual (bikes / hour)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    p3 = out_dir / "03_residual_distribution_val.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_paths.append(str(p3))

    # 4. Chronological Sample Tracking
    fig, ax = plt.subplots(figsize=(11, 5))
    sample_len = min(168, len(val_oper))  # 1 full week (168 hours)
    ax.plot(range(sample_len), y_val[:sample_len], label="Actual Demand", color="black", linewidth=1.8)
    ax.plot(range(sample_len), y_pred[:sample_len], label="Champion Prediction", color="#e67e22", linewidth=2.0, linestyle="--")
    ax.set_title(f"Figure 4: Chronological Demand Tracking Sample ({sample_len} Consecutive Validation Hours)", fontweight="bold")
    ax.set_xlabel("Hour Index in Validation Sample")
    ax.set_ylabel("Rented Bike Count")
    ax.legend()
    fig.tight_layout()
    p4 = out_dir / "04_chronological_sample_val.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_paths.append(str(p4))

    # 5. Model Family Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5.5))
    best_by_family = results_df.groupby("model_family")["val_rmse"].min().sort_values()
    family_colors = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#e74c3c"]
    bars = ax.barh(best_by_family.index, best_by_family.values, color=family_colors[:len(best_by_family)], edgecolor="black")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.axvline(600.96, color="#c0392b", linestyle="--", linewidth=1.8, label="Baseline 2 (Hourly Mean: 600.96)")
    ax.axvline(760.02, color="#7f8c8d", linestyle=":", linewidth=1.8, label="Baseline 1 (Training Mean: 760.02)")
    ax.set_title("Figure 5: Best Validation RMSE Achieved by Model Family", fontweight="bold")
    ax.set_xlabel("Validation RMSE (lower is better)")
    ax.set_ylabel("Model Family")
    ax.set_xlim(0, max(best_by_family.max(), 760.02) * 1.12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    p5 = out_dir / "05_model_comparison_val_rmse.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_paths.append(str(p5))

    # 6. Feature Importance
    regressor = champion_pipe.named_steps["reg"]
    preprocessor = champion_pipe.named_steps["prep"]

    # Recover transformed feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    except Exception:
        clean_names = feature_cols

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        imp_df = pd.DataFrame({"feature": clean_names, "importance": importances})
        imp_df = imp_df.sort_values(by="importance", ascending=True).tail(15)  # top 15

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#3498db", edgecolor="black")
        ax.set_title("Figure 6: Top 15 Feature Importances (Selected Champion Model)", fontweight="bold")
        ax.set_xlabel("Gini / Gain Importance")
        fig.tight_layout()
        p6 = out_dir / "06_feature_importance_val.png"
        fig.savefig(p6)
        plt.close(fig)
        generated_paths.append(str(p6))
    elif hasattr(regressor, "coef_"):
        coefs = np.abs(regressor.coef_)
        imp_df = pd.DataFrame({"feature": clean_names, "importance": coefs})
        imp_df = imp_df.sort_values(by="importance", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#3498db", edgecolor="black")
        ax.set_title("Figure 6: Top 15 Standardized Absolute Coefficients (Linear Model)", fontweight="bold")
        ax.set_xlabel("Absolute Standardized Coefficient Magnitude")
        fig.tight_layout()
        p6 = out_dir / "06_feature_importance_val.png"
        fig.savefig(p6)
        plt.close(fig)
        generated_paths.append(str(p6))

    print(f"[INFO] Generated {len(generated_paths)} validation diagnostic figures in: {out_dir}")
    return generated_paths


# ---------------------------------------------------------------------------
# Final Test Evaluation (Executed ONLY After Selection is Locked!)
# ---------------------------------------------------------------------------


def refit_and_evaluate_final_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lock_record: Dict[str, Any],
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    STAGE NINE: Final Test Evaluation.
    1. Recreates champion pipeline from locked record.
    2. Fits pipeline on combined operating-hour data (Train + Validation, N=7,247).
    3. Persists champion model artifact to `models/phase3b_champion_pipeline.joblib`.
    4. Evaluates operating-hour test set (Scenario A, N=1,218).
    5. Evaluates matching Phase 3A baselines on operating-hour test set.
    6. Evaluates full timeline with known closure rule (Scenario B, N=1,314).
    7. Saves `reports/phase3b/final_test_metrics.json`.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare combined training + validation operating hours
    train_oper = filter_operating_hours(train_df)
    val_oper = filter_operating_hours(val_df)
    test_oper = filter_operating_hours(test_df)

    combined_oper = pd.concat([train_oper, val_oper], ignore_index=True)
    n_combined = len(combined_oper)
    assert n_combined == 6084 + 1163, f"Expected 7,247 combined operating rows, got {n_combined}"

    # Extract feature columns from lock record
    num_cols = lock_record["numerical_features"]
    cat_cols = lock_record["categorical_features"]
    feature_cols = num_cols + cat_cols

    X_train_comb = combined_oper[feature_cols]
    y_train_comb = combined_oper[TARGET_COLUMN].values

    # 2. Recreate and refit champion pipeline
    family = lock_record["model_family"]
    params = lock_record["hyperparameters"]

    print(f"[INFO] Refitting champion model ({family}) on combined Train+Val operating data ({n_combined} rows)...")
    champion_pipe = create_pipeline(family, num_cols, cat_cols, params)
    champion_pipe.fit(X_train_comb, y_train_comb)

    # 3. Save champion artifact
    artifact_path = models_dir / "phase3b_champion_pipeline.joblib"
    joblib.dump(champion_pipe, artifact_path)
    artifact_md5 = file_md5(str(artifact_path))
    print(f"[INFO] Saved champion pipeline artifact to: {artifact_path} (MD5: {artifact_md5})")

    # 4. Scenario A: Test Operating Hours Evaluation
    X_test_oper = test_oper[feature_cols]
    y_test_oper = test_oper[TARGET_COLUMN].values

    y_test_oper_pred = champion_pipe.predict(X_test_oper)
    model_test_a_metrics = evaluate_predictions(y_test_oper, y_test_oper_pred)

    # Baselines on test operating hours
    # Baseline 1: Operating Training Mean
    train_oper_mean = calculate_training_mean(train_oper[TARGET_COLUMN])
    b1_test_preds = predict_training_mean(train_oper[TARGET_COLUMN], n_samples=len(test_oper))
    b1_test_metrics = evaluate_predictions(y_test_oper, b1_test_preds)

    # Baseline 2: Hour-of-Day Mean
    b2_test_preds = predict_hourly_mean(train_oper, train_oper[TARGET_COLUMN], test_oper, hour_col=HOUR_COLUMN)
    b2_test_metrics = evaluate_predictions(y_test_oper, b2_test_preds)

    # 5. Scenario B: Full-Timeline Test Evaluation (Known Closure Rule)
    y_test_all = test_df[TARGET_COLUMN].values
    X_test_all = test_df[feature_cols]

    # Predict operational model on all test rows, then apply closure rule
    raw_test_preds = champion_pipe.predict(X_test_all)
    b3_model_full_preds = predict_known_closure(
        test_df, raw_test_preds, functioning_col=FUNCTIONING_COLUMN, closed_val="No"
    )
    model_test_b_metrics = evaluate_predictions(y_test_all, b3_model_full_preds)

    # Baseline 3 (Closure + Mean) on full test timeline
    b3_mean_base_preds = np.full(len(test_df), train_oper_mean)
    b3_base_preds = predict_known_closure(
        test_df, b3_mean_base_preds, functioning_col=FUNCTIONING_COLUMN, closed_val="No"
    )
    b3_test_metrics = evaluate_predictions(y_test_all, b3_base_preds)

    # Baseline 3 Variant (Closure + Hourly) on full test timeline
    b3_hourly_base_preds = predict_hourly_mean(train_oper, train_oper[TARGET_COLUMN], test_df, hour_col=HOUR_COLUMN)
    b3_hourly_full_preds = predict_known_closure(
        test_df, b3_hourly_base_preds, functioning_col=FUNCTIONING_COLUMN, closed_val="No"
    )
    b3_hourly_test_metrics = evaluate_predictions(y_test_all, b3_hourly_full_preds)

    test_report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "champion_pipeline_artifact": str(artifact_path),
        "champion_pipeline_md5": artifact_md5,
        "refit_training_samples": n_combined,
        "champion_model_family": family,
        "feature_config": lock_record["feature_config"],
        "scenario_a_operational_test": {
            "evaluation_population": "test.csv where Functioning Day == 'Yes'",
            "test_samples": len(test_oper),
            "champion_model": model_test_a_metrics,
            "baseline_1_training_mean": {
                "training_mean_value": round(train_oper_mean, 4),
                **b1_test_metrics,
            },
            "baseline_2_hourly_mean": b2_test_metrics,
            "mae_improvement_vs_hourly_base": round(b2_test_metrics["mae"] - model_test_a_metrics["mae"], 2),
            "rmse_improvement_vs_hourly_base": round(b2_test_metrics["rmse"] - model_test_a_metrics["rmse"], 2),
            "pct_rmse_improvement": round(
                ((b2_test_metrics["rmse"] - model_test_a_metrics["rmse"]) / b2_test_metrics["rmse"]) * 100.0, 2
            ),
        },
        "scenario_b_full_timeline_test": {
            "evaluation_population": "test.csv complete timeline (all 1,314 hours)",
            "test_samples": len(test_df),
            "operational_rule": "Predict 0 if Functioning Day == 'No', else champion model",
            "champion_with_closure_rule": model_test_b_metrics,
            "baseline_3_closure_plus_mean": b3_test_metrics,
            "baseline_3_closure_plus_hourly": b3_hourly_test_metrics,
            "pct_rmse_improvement_vs_closure_hourly": round(
                ((b3_hourly_test_metrics["rmse"] - model_test_b_metrics["rmse"]) / b3_hourly_test_metrics["rmse"]) * 100.0, 2
            ),
        },
    }

    test_json_path = REPORTS_PHASE3B_DIR / "final_test_metrics.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    print(f"[INFO] Final test evaluation completed successfully.")
    print(f"       Scenario A (Operating Test RMSE): {model_test_a_metrics['rmse']:.2f} (Hourly Base: {b2_test_metrics['rmse']:.2f})")
    print(f"       Scenario B (Full Timeline RMSE): {model_test_b_metrics['rmse']:.2f} (Closure+Hourly Base: {b3_hourly_test_metrics['rmse']:.2f})")
    print(f"       Saved final test report to: {test_json_path}")

    return test_report


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------


def run_phase3b_pipeline() -> Dict[str, Any]:
    """
    Execute complete Phase 3B pipeline following the strict 10-stage methodology:
    1. Load partitions and build experiment manifest.
    2. Train candidate models and evaluate validation metrics.
    3. Lock model selection record (before test evaluation).
    4. Generate validation diagnostic figures.
    5. Refit champion pipeline on combined Train+Val operating data.
    6. Evaluate final test metrics (Scenario A & B).
    """
    print("=" * 70)
    print("  PHASE 3B: MODEL TRAINING, VALIDATION, SELECTION & TEST EVALUATION")
    print("=" * 70)

    # 1. Load data
    train_df, val_df, test_df, split_meta = load_partitions()
    train_oper = filter_operating_hours(train_df)
    val_oper = filter_operating_hours(val_df)
    test_oper = filter_operating_hours(test_df)

    print(f"[INFO] Partitions loaded:")
    print(f"       Train: {len(train_df)} total | {len(train_oper)} operating")
    print(f"       Val:   {len(val_df)} total | {len(val_oper)} operating")
    print(f"       Test:  {len(test_df)} total | {len(test_oper)} operating (held out)")

    # 2. Build feature manifest
    manifest = build_experiment_manifest()
    print(f"[INFO] Feature manifest ready ({len(manifest)} configurations).")

    # 3. Train candidates
    candidates = get_candidate_models()
    results_df, fitted_pipelines = train_and_evaluate_candidates(
        train_oper, val_oper, manifest, candidates
    )

    # 4. Lock model selection
    lock_record = lock_model_selection(results_df, manifest)
    champion_id = lock_record["champion_candidate_id"]
    champion_pipe_val = fitted_pipelines[champion_id]

    # 5. Generate validation diagnostics
    cfg_info = manifest[lock_record["feature_config"]]
    feature_cols = cfg_info["num_features"] + cfg_info["cat_features"]
    diag_figures = generate_validation_diagnostics(
        champion_pipe_val, val_oper, feature_cols, results_df
    )

    # 6. Final test evaluation (strictly after lock)
    test_metrics = refit_and_evaluate_final_test(
        train_df, val_df, test_df, lock_record
    )

    summary = {
        "status": "PHASE_3B_COMPLETE",
        "lock_record": lock_record,
        "test_metrics": test_metrics,
        "diagnostic_figures": diag_figures,
        "candidate_results_file": str(REPORTS_PHASE3B_DIR / "candidate_results.csv"),
    }

    print("\n" + "=" * 70)
    print("  PHASE 3B PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 70)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3B: Machine learning regression model training, validation, selection, and testing."
    )
    args = parser.parse_args()

    try:
        run_phase3b_pipeline()
    except Exception as e:
        print(f"[ERROR] Phase 3B pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
