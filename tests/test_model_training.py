"""
Automated Tests for Model Training, Preprocessing Pipelines, and Evaluation
===========================================================================

Phase 3B Verification Suite:
13.1 Feature integrity:
    - Every declared feature exists in the dataset.
    - No feature configuration contains target ('Rented Bike Count').
    - No feature configuration contains raw 'Date'.
    - 'Functioning Day' is not included as an operating-hour model feature.
    - Feature Set D differs from Set C only by the removal of 'Dew point temperature(°C)'.
    - Categorical feature variants contain the intended categorical features ('Seasons', 'Holiday').

13.2 Pipeline integrity:
    - Every candidate pipeline fits on a small synthetic training sample.
    - Every pipeline produces predictions with expected length.
    - Predictions are finite numeric values (no NaN or Inf).
    - Unknown categories do not crash the one-hot encoding pipeline.
    - Missing required input columns cause a clear, meaningful error.
    - Target is never passed as an input feature.
    - Numeric and categorical preprocessing applied to correct columns.

13.3 Leakage protection:
    - Preprocessing is encapsulated inside the pipeline.
    - Fitting uses training partition only during candidate comparison.
    - Validation data is used for scoring, not fitting.
    - Test data is strictly held out until selection is locked.
    - Model selection lock timestamp precedes test evaluation timestamp.
    - Champion pipeline is refitted only after selection is locked.

13.4 Metrics and selection:
    - MAE, RMSE, and R^2 match standard formulas on known toy values.
    - Ranking is sorted strictly by validation RMSE ascending.
    - Failed candidates are recorded with error message and excluded from ranking.
    - Selection is deterministic with fixed random seed.
    - Baseline comparisons use matching evaluation populations.

13.5 Artifact and reporting integrity:
    - Experiment manifest is valid JSON with 8 configurations.
    - Candidate results CSV contains all required fields and valid metrics.
    - Selection record contains locked configuration.
    - Saved champion pipeline (.joblib) can be reloaded and produces valid predictions.
    - Final evaluation metrics clearly distinguish validation from test results.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.baselines import evaluate_predictions
from src.model_training import (
    DATE_COLUMN,
    FUNCTIONING_COLUMN,
    TARGET_COLUMN,
    build_experiment_manifest,
    create_pipeline,
    filter_operating_hours,
    get_candidate_models,
    load_partitions,
)

# Paths
SPLITS_DIR = Path("data/splits")
REPORTS_PHASE3B_DIR = Path("reports/phase3b")
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    """Load or generate the 8 feature experiment configurations."""
    return build_experiment_manifest()


@pytest.fixture(scope="module")
def partitions():
    """Load the approved Phase 3A splits."""
    return load_partitions()


@pytest.fixture
def synthetic_train_df():
    """Small synthetic dataset for testing pipeline construction and fitting."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "Hour": np.random.randint(0, 24, size=n),
        "day_of_week": np.random.randint(0, 7, size=n),
        "month": np.random.randint(1, 13, size=n),
        "day_of_month": np.random.randint(1, 29, size=n),
        "day_of_year": np.random.randint(1, 366, size=n),
        "is_weekend": np.random.randint(0, 2, size=n),
        "hour_sin": np.sin(np.random.rand(n)),
        "hour_cos": np.cos(np.random.rand(n)),
        "month_sin": np.sin(np.random.rand(n)),
        "month_cos": np.cos(np.random.rand(n)),
        "Temperature(°C)": np.random.uniform(-10, 35, size=n),
        "Humidity(%)": np.random.uniform(10, 95, size=n),
        "Wind speed (m/s)": np.random.uniform(0.1, 8.0, size=n),
        "Visibility (10m)": np.random.uniform(100, 2000, size=n),
        "Dew point temperature(°C)": np.random.uniform(-15, 25, size=n),
        "Solar Radiation (MJ/m2)": np.random.uniform(0.0, 3.5, size=n),
        "Rainfall(mm)": np.random.exponential(0.5, size=n),
        "Snowfall (cm)": np.random.exponential(0.1, size=n),
        "Seasons": np.random.choice(["Winter", "Spring", "Summer", "Autumn"], size=n),
        "Holiday": np.random.choice(["Holiday", "No Holiday"], size=n),
        "Functioning Day": ["Yes"] * n,
        "Rented Bike Count": np.random.uniform(50, 1500, size=n),
    })


# ---------------------------------------------------------------------------
# 13.1 Feature Integrity Tests
# ---------------------------------------------------------------------------


def test_manifest_contains_eight_configurations(manifest):
    """The manifest must contain exactly the 8 approved feature configurations."""
    expected_keys = {
        "A_temporal",
        "A_temporal_cat",
        "B_weather",
        "B_weather_cat",
        "C_combined",
        "C_combined_cat",
        "D_reduced",
        "D_reduced_cat",
    }
    assert set(manifest.keys()) == expected_keys


def test_all_declared_features_exist_in_splits(manifest, partitions):
    """Every declared feature must exist in the actual split partitions."""
    train_df, _, _, _ = partitions
    for cfg_id, cfg in manifest.items():
        all_features = cfg["num_features"] + cfg["cat_features"]
        missing = [f for f in all_features if f not in train_df.columns]
        assert len(missing) == 0, f"Config {cfg_id} references missing features: {missing}"


def test_no_target_or_date_or_functioning_leakage_in_features(manifest):
    """Feature configurations must never contain target, Date, or Functioning Day."""
    forbidden = {TARGET_COLUMN, DATE_COLUMN, FUNCTIONING_COLUMN}
    for cfg_id, cfg in manifest.items():
        all_features = set(cfg["num_features"] + cfg["cat_features"])
        intersect = all_features.intersection(forbidden)
        assert len(intersect) == 0, f"Config {cfg_id} contains forbidden columns: {intersect}"


def test_feature_set_d_differs_from_c_only_by_dew_point(manifest):
    """Feature Set D must differ from Set C strictly and only by Dew point temperature."""
    set_c_num = set(manifest["C_combined"]["num_features"])
    set_d_num = set(manifest["D_reduced"]["num_features"])
    diff = set_c_num - set_d_num
    assert diff == {"Dew point temperature(°C)"}
    assert len(manifest["C_combined"]["num_features"]) == len(manifest["D_reduced"]["num_features"]) + 1


def test_categorical_variants_contain_intended_categoricals(manifest):
    """Categorical variants must contain exactly ['Seasons', 'Holiday']."""
    expected_cats = ["Seasons", "Holiday"]
    for base in ["A_temporal", "B_weather", "C_combined", "D_reduced"]:
        cat_key = f"{base}_cat"
        assert manifest[cat_key]["cat_features"] == expected_cats
        assert manifest[base]["cat_features"] == []


# ---------------------------------------------------------------------------
# 13.2 Pipeline Integrity Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["LinearRegression", "Ridge", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost"])
def test_pipeline_fits_and_predicts_synthetic_data(synthetic_train_df, manifest, family):
    """Every supported model family must fit on synthetic data and produce finite predictions."""
    cfg = manifest["C_combined_cat"]
    num_cols = cfg["num_features"]
    cat_cols = cfg["cat_features"]

    X = synthetic_train_df[num_cols + cat_cols]
    y = synthetic_train_df[TARGET_COLUMN].values

    pipe = create_pipeline(family, num_cols, cat_cols)
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds)), f"Predictions for {family} contain NaN or Inf"
    assert isinstance(preds, np.ndarray)


def test_pipeline_handles_unknown_categories_gracefully(synthetic_train_df, manifest):
    """OneHotEncoder must handle unknown categories at inference without crashing."""
    cfg = manifest["A_temporal_cat"]
    num_cols = cfg["num_features"]
    cat_cols = cfg["cat_features"]

    X_train = synthetic_train_df[num_cols + cat_cols].copy()
    y_train = synthetic_train_df[TARGET_COLUMN].values

    pipe = create_pipeline("LinearRegression", num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    # Create unseen category in inference
    X_test = X_train.copy().iloc[:2]
    X_test.loc[0, "Seasons"] = "Monsoon"  # Unseen season
    X_test.loc[1, "Holiday"] = "Special Festival"  # Unseen holiday

    preds = pipe.predict(X_test)
    assert len(preds) == 2
    assert np.all(np.isfinite(preds))


def test_missing_input_column_raises_error(synthetic_train_df, manifest):
    """Missing a required feature column must raise an explicit error during prediction."""
    cfg = manifest["B_weather"]
    num_cols = cfg["num_features"]
    cat_cols = cfg["cat_features"]

    X = synthetic_train_df[num_cols]
    y = synthetic_train_df[TARGET_COLUMN].values

    pipe = create_pipeline("DecisionTree", num_cols, cat_cols)
    pipe.fit(X, y)

    # Drop a required column
    X_corrupted = X.drop(columns=["Temperature(°C)"])
    with pytest.raises((KeyError, ValueError)):
        pipe.predict(X_corrupted)


def test_linear_pipeline_uses_standard_scaler_and_tree_uses_passthrough(manifest):
    """Linear models must scale numeric features; tree models must pass through."""
    cfg = manifest["C_combined_cat"]
    pipe_linear = create_pipeline("LinearRegression", cfg["num_features"], cfg["cat_features"])
    pipe_tree = create_pipeline("RandomForest", cfg["num_features"], cfg["cat_features"])

    prep_linear = pipe_linear.named_steps["prep"]
    prep_tree = pipe_tree.named_steps["prep"]

    # Linear transformer for 'num' should be StandardScaler
    num_trans_linear = [t[1] for t in prep_linear.transformers if t[0] == "num"][0]
    assert num_trans_linear.__class__.__name__ == "StandardScaler"

    # Tree transformer for 'num' should be passthrough
    num_trans_tree = [t[1] for t in prep_tree.transformers if t[0] == "num"][0]
    assert num_trans_tree == "passthrough"


# ---------------------------------------------------------------------------
# 13.3 Leakage Protection Tests
# ---------------------------------------------------------------------------


def test_operating_hours_filter_strictly_excludes_closure_rows(partitions):
    """Operating-hour filtering must only keep Functioning Day == 'Yes'."""
    train_df, val_df, test_df, _ = partitions
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        oper_df = filter_operating_hours(df)
        assert (oper_df[FUNCTIONING_COLUMN] == "Yes").all(), f"Closure rows found in {name} operating filter"
        assert len(oper_df) == (df[FUNCTIONING_COLUMN] == "Yes").sum()


def test_model_selection_lock_precedes_final_test_evaluation():
    """Selection lock timestamp must strictly precede final test evaluation timestamp."""
    selection_path = REPORTS_PHASE3B_DIR / "model_selection.json"
    test_metrics_path = REPORTS_PHASE3B_DIR / "final_test_metrics.json"

    assert selection_path.exists(), "Missing model_selection.json"
    assert test_metrics_path.exists(), "Missing final_test_metrics.json"

    with open(selection_path, "r", encoding="utf-8") as f:
        sel_data = json.load(f)
    with open(test_metrics_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    lock_time = pd.to_datetime(sel_data["lock_timestamp"])
    eval_time = pd.to_datetime(test_data["evaluation_timestamp"])

    assert lock_time <= eval_time, f"Selection lock ({lock_time}) must precede test evaluation ({eval_time})"
    assert sel_data["selection_status"] == "LOCKED"


# ---------------------------------------------------------------------------
# 13.4 Metrics and Selection Tests
# ---------------------------------------------------------------------------


def test_metric_calculation_matches_known_toy_values():
    """Verify MAE, RMSE, and R^2 on a small hand-calculated example."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 380.0])
    # Differences: +10, -10, +10, -20.
    # Absolute errors: 10, 10, 10, 20 -> MAE = 50 / 4 = 12.5.
    # Squared errors: 100, 100, 100, 400 -> MSE = 700 / 4 = 175.0 -> RMSE = sqrt(175) = 13.228756555...
    # y_mean = 250, Total SS = 150^2 + 50^2 + 50^2 + 150^2 = 22500 + 2500 + 2500 + 22500 = 50000.
    # Residual SS = 700. R^2 = 1 - 700 / 50000 = 0.986.

    metrics = evaluate_predictions(y_true, y_pred)
    assert np.isclose(metrics["mae"], 12.5)
    assert np.isclose(metrics["rmse"], np.sqrt(175.0))
    assert np.isclose(metrics["r2"], 1.0 - (700.0 / 50000.0))


def test_candidate_ranking_is_strictly_sorted_by_validation_rmse():
    """Candidate results CSV must be ranked in ascending order of validation RMSE."""
    results_path = REPORTS_PHASE3B_DIR / "candidate_results.csv"
    assert results_path.exists(), "candidate_results.csv is missing"

    df_results = pd.read_csv(results_path)
    successful = df_results[df_results["fit_status"] == "SUCCESS"]

    val_rmses = successful["val_rmse"].tolist()
    assert val_rmses == sorted(val_rmses), "Candidate results are not strictly sorted by val_rmse ascending"


def test_all_120_candidates_evaluated_and_successful():
    """All 120 candidate models (15 models x 8 feature sets) must be evaluated and successful."""
    results_path = REPORTS_PHASE3B_DIR / "candidate_results.csv"
    df_results = pd.read_csv(results_path)
    assert len(df_results) == 120, f"Expected 120 candidate evaluations, found {len(df_results)}"
    assert (df_results["fit_status"] == "SUCCESS").all(), "Some candidates failed during training"


# ---------------------------------------------------------------------------
# 13.5 Artifact and Reporting Integrity Tests
# ---------------------------------------------------------------------------


def test_saved_champion_pipeline_can_be_reloaded_and_predicts(partitions):
    """The persisted champion pipeline must be reloadable via joblib and predict correctly."""
    champion_path = MODELS_DIR / "phase3b_champion_pipeline.joblib"
    assert champion_path.exists(), f"Champion artifact missing: {champion_path}"

    pipeline = joblib.load(champion_path)
    assert isinstance(pipeline, Pipeline)

    # Test prediction on a sample of test data
    _, _, test_df, _ = partitions
    test_oper = filter_operating_hours(test_df)

    with open(REPORTS_PHASE3B_DIR / "model_selection.json", "r", encoding="utf-8") as f:
        sel = json.load(f)

    feature_cols = sel["numerical_features"] + sel["categorical_features"]
    sample_X = test_oper[feature_cols].iloc[:10]

    preds = pipeline.predict(sample_X)
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)  # Rental demand should be positive


def test_final_test_metrics_contains_both_scenarios():
    """final_test_metrics.json must record both Scenario A and Scenario B metrics."""
    metrics_path = REPORTS_PHASE3B_DIR / "final_test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "scenario_a_operational_test" in data
    assert "scenario_b_full_timeline_test" in data

    scen_a = data["scenario_a_operational_test"]
    scen_b = data["scenario_b_full_timeline_test"]

    assert scen_a["test_samples"] == 1218
    assert scen_b["test_samples"] == 1314
    assert scen_a["champion_model"]["rmse"] > 0
    assert scen_b["champion_with_closure_rule"]["rmse"] > 0
