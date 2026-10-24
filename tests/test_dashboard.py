"""
Automated Tests for Dashboard Utilities, Feature Construction, and Inference
============================================================================

Phase 4 Test Suite:
- Feature engineering correctness (dates, cyclical formulas, column order).
- Input validation and boundary enforcement.
- Model loading, pipeline integrity, and inference reproducibility.
- Deterministic closure rule logic (Scenario B).
- Display rounding and non-negative clamping.
- Preservation of existing artifacts and test suite.
"""

from datetime import date, datetime
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.dashboard_utils import (
    ALL_INPUT_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    NUMERICAL_FEATURES,
    construct_model_input_row,
    get_feature_importance_df,
    get_verified_metrics,
    load_champion_pipeline,
    load_hourly_baseline_lookup,
    load_processed_data,
    predict_demand,
    validate_user_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def champion_pipeline():
    """Load the persisted champion pipeline."""
    return load_champion_pipeline(DEFAULT_MODEL_PATH)


@pytest.fixture
def valid_input_kwargs():
    """Valid set of user inputs for testing."""
    return {
        "target_date": date(2018, 6, 15),  # Friday (day_of_week=4, not weekend)
        "hour": 18,
        "temperature": 24.5,
        "humidity": 55.0,
        "wind_speed": 2.1,
        "visibility": 1800.0,
        "dew_point": 15.0,
        "solar_radiation": 1.25,
        "rainfall": 0.0,
        "snowfall": 0.0,
        "season": "Summer",
        "holiday": "No Holiday",
    }


# ---------------------------------------------------------------------------
# 16.A Feature Construction Tests
# ---------------------------------------------------------------------------


def test_feature_construction_exact_columns_and_shape(valid_input_kwargs):
    """Input row must have exactly shape (1, 20) with all required columns in order."""
    row = construct_model_input_row(**valid_input_kwargs)
    assert row.shape == (1, 20)
    assert list(row.columns) == ALL_INPUT_FEATURES
    assert not row.isnull().any().any()


def test_temporal_feature_calculations():
    """Verify calendar feature derivation for a known date."""
    # 2018-09-15 was a Saturday (day_of_week=5, day_of_month=15, month=9, day_of_year=258)
    row = construct_model_input_row(
        target_date="2018-09-15",
        hour=8,
        temperature=20.0,
        humidity=60.0,
        wind_speed=1.5,
        visibility=1900.0,
        dew_point=12.0,
        solar_radiation=0.8,
        rainfall=0.0,
        snowfall=0.0,
        season="Autumn",
        holiday="No Holiday",
    )
    assert row["day_of_week"].iloc[0] == 5
    assert row["is_weekend"].iloc[0] == 1
    assert row["month"].iloc[0] == 9
    assert row["day_of_month"].iloc[0] == 15
    assert row["day_of_year"].iloc[0] == 258
    assert row["Hour"].iloc[0] == 8


def test_cyclical_feature_formulas():
    """Verify sin and cos cyclical transformations for hour and month."""
    # Hour 6: 6/24 = 0.25 -> 2*pi*0.25 = pi/2 -> sin = 1.0, cos = 0.0
    # Month 3: 3/12 = 0.25 -> 2*pi*0.25 = pi/2 -> sin = 1.0, cos = 0.0
    row = construct_model_input_row(
        target_date=date(2018, 3, 20),
        hour=6,
        temperature=10.0,
        humidity=50.0,
        wind_speed=2.0,
        visibility=1500.0,
        dew_point=0.0,
        solar_radiation=0.5,
        rainfall=0.0,
        snowfall=0.0,
        season="Spring",
        holiday="No Holiday",
    )
    assert np.isclose(row["hour_sin"].iloc[0], 1.0)
    assert np.isclose(row["hour_cos"].iloc[0], 0.0, atol=1e-7)
    assert np.isclose(row["month_sin"].iloc[0], 1.0)
    assert np.isclose(row["month_cos"].iloc[0], 0.0, atol=1e-7)


def test_no_target_date_or_functioning_day_in_model_input(valid_input_kwargs):
    """Model input row must not contain Rented Bike Count, Date string, or Functioning Day."""
    row = construct_model_input_row(**valid_input_kwargs)
    forbidden = {"Rented Bike Count", "Date", "Functioning Day"}
    assert set(row.columns).isdisjoint(forbidden)


# ---------------------------------------------------------------------------
# 16.B Model Inference Tests
# ---------------------------------------------------------------------------


def test_champion_pipeline_loads_and_is_sklearn_pipeline(champion_pipeline):
    """The saved champion pipeline must load cleanly and be an sklearn Pipeline."""
    assert isinstance(champion_pipeline, Pipeline)
    assert "prep" in champion_pipeline.named_steps
    assert "reg" in champion_pipeline.named_steps


def test_prediction_output_structure_and_values(champion_pipeline, valid_input_kwargs):
    """Valid prediction must return finite raw prediction and non-negative integer."""
    row = construct_model_input_row(**valid_input_kwargs)
    result = predict_demand(champion_pipeline, row, functioning_day="Yes")

    assert "raw_prediction" in result
    assert "display_prediction" in result
    assert "is_closure" in result
    assert isinstance(result["display_prediction"], int)
    assert result["display_prediction"] >= 0
    assert np.isfinite(result["raw_prediction"])
    assert result["is_closure"] is False


def test_prediction_with_baseline_comparison(champion_pipeline, valid_input_kwargs):
    """Hourly baseline comparison is included when lookup table is provided."""
    row = construct_model_input_row(**valid_input_kwargs)
    lookup = {18: 1250.0}
    result = predict_demand(champion_pipeline, row, functioning_day="Yes", hourly_lookup=lookup)

    assert result["baseline_prediction"] == 1250.0
    assert result["diff_vs_baseline"] is not None
    assert np.isclose(result["diff_vs_baseline"], result["raw_prediction"] - 1250.0)


# ---------------------------------------------------------------------------
# 16.C Input Validation Tests
# ---------------------------------------------------------------------------


def test_validation_rejects_invalid_hour():
    """Hour outside 0-23 must raise ValueError."""
    with pytest.raises(ValueError, match="Hour must be an integer between 0 and 23"):
        validate_user_inputs(hour=25, temperature=20, humidity=50, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=0, snowfall=0,
                             season="Spring", holiday="No Holiday")


def test_validation_rejects_invalid_humidity():
    """Humidity outside [0, 100] must raise ValueError."""
    with pytest.raises(ValueError, match="Humidity must be a percentage between 0% and 100%"):
        validate_user_inputs(hour=12, temperature=20, humidity=120, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=0, snowfall=0,
                             season="Spring", holiday="No Holiday")


def test_validation_rejects_negative_precipitation():
    """Negative rainfall or snowfall must raise ValueError."""
    with pytest.raises(ValueError, match="Rainfall cannot be negative"):
        validate_user_inputs(hour=12, temperature=20, humidity=50, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=-1.5, snowfall=0,
                             season="Spring", holiday="No Holiday")

    with pytest.raises(ValueError, match="Snowfall cannot be negative"):
        validate_user_inputs(hour=12, temperature=20, humidity=50, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=0, snowfall=-0.5,
                             season="Spring", holiday="No Holiday")


def test_validation_rejects_invalid_categorical():
    """Unknown season or holiday must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid season"):
        validate_user_inputs(hour=12, temperature=20, humidity=50, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=0, snowfall=0,
                             season="Monsoon", holiday="No Holiday")

    with pytest.raises(ValueError, match="Invalid holiday status"):
        validate_user_inputs(hour=12, temperature=20, humidity=50, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=0, snowfall=0,
                             season="Spring", holiday="Festival")


def test_validation_rejects_non_finite_values():
    """NaN or Inf inputs must raise ValueError."""
    with pytest.raises(ValueError, match="must be a finite numeric value"):
        validate_user_inputs(hour=12, temperature=float("nan"), humidity=50, wind_speed=1, visibility=1000,
                             dew_point=10, solar_radiation=0.5, rainfall=0, snowfall=0,
                             season="Spring", holiday="No Holiday")


# ---------------------------------------------------------------------------
# 16.D Closure Rule Tests (Scenario B)
# ---------------------------------------------------------------------------


def test_closure_rule_returns_zero_demand(champion_pipeline, valid_input_kwargs):
    """When functioning_day is 'No', prediction must be exactly 0 without calling regressor."""
    row = construct_model_input_row(**valid_input_kwargs)
    result = predict_demand(champion_pipeline, row, functioning_day="No")

    assert result["raw_prediction"] == 0.0
    assert result["display_prediction"] == 0
    assert result["is_closure"] is True
    assert "closure" in result["explanation"].lower()


# ---------------------------------------------------------------------------
# 16.E Metadata and Data Loading Tests
# ---------------------------------------------------------------------------


def test_processed_data_loads_successfully():
    """Master processed dataset must load with 8,760 rows and 23 columns."""
    df = load_processed_data(DEFAULT_PROCESSED_DATA_PATH)
    assert len(df) == 8760
    assert df.shape[1] == 23
    assert "Rented Bike Count" in df.columns


def test_hourly_baseline_lookup_contains_24_hours():
    """Hourly lookup table must contain means for all 24 hours (0 to 23)."""
    lookup = load_hourly_baseline_lookup()
    assert len(lookup) == 24
    for h in range(24):
        assert h in lookup
        assert lookup[h] > 0.0


def test_feature_importance_sums_to_100_percent(champion_pipeline):
    """Feature importance extraction must return 24 features summing to ~100%."""
    df_imp = get_feature_importance_df(champion_pipeline)
    assert len(df_imp) == 24
    assert np.isclose(df_imp["percentage"].sum(), 100.0)
    assert df_imp["feature"].iloc[0] == "Temperature(°C)"


def test_verified_metrics_matches_phase3b_spec():
    """Verified metrics helper must match locked Phase 3B values."""
    metrics = get_verified_metrics()
    assert metrics["validation"]["rmse"] == 321.58
    assert metrics["validation"]["mae"] == 237.54
    assert metrics["validation"]["r2"] == 0.7775
    assert metrics["test_scenario_a"]["rmse"] == 368.56
    assert metrics["test_scenario_a"]["mae"] == 260.94
    assert metrics["test_scenario_a"]["r2"] == 0.5432
    assert metrics["test_scenario_b"]["rmse"] == 354.84
    assert metrics["test_scenario_b"]["r2"] == 0.6120


def test_app_module_imports_cleanly():
    """app.py must import without throwing syntax, import, or execution errors."""
    import app
    assert callable(app.main)
    assert callable(app.render_overview)
    assert callable(app.render_predict)
    assert callable(app.render_analysis)
    assert callable(app.render_performance)
    assert callable(app.render_about)


def test_temperature_scatter_trendline_with_statsmodels():
    """Verify that Plotly Express creates a temperature scatter plot with lowess trendline."""
    import plotly.express as px
    import statsmodels  # Must be installed and importable

    df_test = pd.DataFrame({
        "Temperature(°C)": [10.0, 15.0, 20.0, 25.0, 30.0],
        "Rented Bike Count": [200, 450, 700, 900, 850],
        "Seasons": ["Spring", "Spring", "Summer", "Summer", "Summer"],
    })
    fig = px.scatter(
        df_test,
        x="Temperature(°C)",
        y="Rented Bike Count",
        color="Seasons",
        trendline="lowess",
    )
    assert fig is not None
    assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# Phase 5: End-to-End Verification and Integrity Tests
# ---------------------------------------------------------------------------


def test_saved_champion_model_artifact_hash_and_size():
    """Verify the saved champion model artifact exists, is unchanged, and matches recorded MD5."""
    import hashlib
    assert DEFAULT_MODEL_PATH.exists(), f"Model artifact missing at {DEFAULT_MODEL_PATH}"
    file_bytes = DEFAULT_MODEL_PATH.read_bytes()
    md5_hash = hashlib.md5(file_bytes).hexdigest()
    assert md5_hash == "1000468ebd831efa8643731c99c83648"
    assert len(file_bytes) == 146244


def test_end_to_end_inference_all_categorical_combinations(champion_pipeline):
    """Verify inference succeeds across all season and holiday category combinations."""
    from src.dashboard_utils import EXPECTED_SEASONS, EXPECTED_HOLIDAYS
    for season in EXPECTED_SEASONS:
        for holiday in EXPECTED_HOLIDAYS:
            row = construct_model_input_row(
                target_date="2018-05-15",
                hour=14,
                temperature=22.5,
                humidity=50.0,
                wind_speed=2.0,
                visibility=1600.0,
                dew_point=11.5,
                solar_radiation=1.8,
                rainfall=0.0,
                snowfall=0.0,
                season=season,
                holiday=holiday,
            )
            result = predict_demand(champion_pipeline, row, functioning_day="Yes")
            assert np.isfinite(result["raw_prediction"])
            assert isinstance(result["display_prediction"], int)
            assert result["display_prediction"] >= 0
            assert result["is_closure"] is False


def test_closure_rule_never_invokes_pipeline(champion_pipeline, valid_input_kwargs):
    """Confirm the closure rule returns zero deterministically without evaluating regressor."""
    class BrokenPipeline:
        def predict(self, *args, **kwargs):
            raise AssertionError("Pipeline.predict should NOT be invoked for closed days!")

    row = construct_model_input_row(**valid_input_kwargs)
    result = predict_demand(BrokenPipeline(), row, functioning_day="No")
    assert result["raw_prediction"] == 0.0
    assert result["display_prediction"] == 0
    assert result["is_closure"] is True


def test_repeated_predictions_are_strictly_deterministic(champion_pipeline, valid_input_kwargs):
    """Repeated predictions with identical inputs must yield identical numerical results."""
    row = construct_model_input_row(**valid_input_kwargs)
    first = predict_demand(champion_pipeline, row, functioning_day="Yes")
    for _ in range(5):
        repeat = predict_demand(champion_pipeline, row, functioning_day="Yes")
        assert first["raw_prediction"] == repeat["raw_prediction"]
        assert first["display_prediction"] == repeat["display_prediction"]



