"""
Automated Test Suite for Preprocessing Pipeline
===============================================

Phase 2 Test Coverage:
- Date parsing and formatting
- Temporal feature calculations (day_of_week, month, day_of_month, day_of_year)
- Cyclical feature mathematical correctness (hour_sin/cos, month_sin/cos)
- Weekend flag calculation
- Target column and row count preservation
- Zero-demand record preservation
- Validation checks (invalid hours, negative rentals, illegal categoricals)
- Duplicate timestamp verification
- Processed CSV and metadata file reload verification
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    DEFAULT_METADATA_PATH,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_PATH,
    analyze_zero_demand,
    engineer_temporal_features,
    load_raw_data,
    validate_raw_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Create a minimal synthetic sample for unit tests without modifying real data."""
    return pd.DataFrame({
        "Date": ["01/12/2017", "02/12/2017", "15/06/2018"],  # Fri, Sat, Fri
        "Rented Bike Count": [254, 0, 1500],
        "Hour": [0, 12, 18],
        "Temperature(°C)": [-5.2, 2.0, 26.5],
        "Humidity(%)": [37, 45, 60],
        "Wind speed (m/s)": [2.2, 1.5, 1.0],
        "Visibility (10m)": [2000, 1800, 1950],
        "Dew point temperature(°C)": [-17.6, -8.0, 18.0],
        "Solar Radiation (MJ/m2)": [0.0, 1.5, 2.8],
        "Rainfall(mm)": [0.0, 0.0, 0.0],
        "Snowfall (cm)": [0.0, 0.0, 0.0],
        "Seasons": ["Winter", "Winter", "Summer"],
        "Holiday": ["No Holiday", "No Holiday", "No Holiday"],
        "Functioning Day": ["Yes", "No", "Yes"],
    })


@pytest.fixture
def processed_df() -> pd.DataFrame:
    """Load the saved processed dataset for validation tests."""
    assert os.path.exists(DEFAULT_PROCESSED_PATH), f"Processed CSV not found at {DEFAULT_PROCESSED_PATH}"
    return pd.read_csv(DEFAULT_PROCESSED_PATH, encoding="utf-8")


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Load raw dataset for comparison tests."""
    assert os.path.exists(DEFAULT_RAW_PATH), f"Raw CSV not found at {DEFAULT_RAW_PATH}"
    df, _ = load_raw_data(DEFAULT_RAW_PATH)
    return df


# ---------------------------------------------------------------------------
# Unit Tests on Synthetic Data
# ---------------------------------------------------------------------------


def test_date_parsing_and_formatting(synthetic_sample_df):
    """Verify dd/mm/yyyy string parses and formats to YYYY-MM-DD."""
    out = engineer_temporal_features(synthetic_sample_df)
    assert list(out["Date"]) == ["2017-12-01", "2017-12-02", "2018-06-15"]


def test_temporal_feature_calculations(synthetic_sample_df):
    """Verify day_of_week, month, day_of_month, day_of_year calculations."""
    out = engineer_temporal_features(synthetic_sample_df)

    # 2017-12-01: Friday (4), December (12), Day 1, DayOfYear 335
    row0 = out.iloc[0]
    assert row0["day_of_week"] == 4
    assert row0["month"] == 12
    assert row0["day_of_month"] == 1
    assert row0["day_of_year"] == 335

    # 2017-12-02: Saturday (5), December (12), Day 2, DayOfYear 336
    row1 = out.iloc[1]
    assert row1["day_of_week"] == 5
    assert row1["month"] == 12
    assert row1["day_of_month"] == 2
    assert row1["day_of_year"] == 336


def test_is_weekend_calculation(synthetic_sample_df):
    """Verify is_weekend: Friday=0, Saturday=1."""
    out = engineer_temporal_features(synthetic_sample_df)
    assert out.iloc[0]["is_weekend"] == 0  # Friday
    assert out.iloc[1]["is_weekend"] == 1  # Saturday
    assert out.iloc[2]["is_weekend"] == 0  # Friday


def test_hour_cyclical_features(synthetic_sample_df):
    """Verify sin/cos encoding for hours conforms to trigonometric unit circle."""
    out = engineer_temporal_features(synthetic_sample_df)

    # Hour 0: sin(0) = 0, cos(0) = 1
    assert np.isclose(out.iloc[0]["hour_sin"], 0.0, atol=1e-6)
    assert np.isclose(out.iloc[0]["hour_cos"], 1.0, atol=1e-6)

    # Hour 12: sin(pi) = 0, cos(pi) = -1
    assert np.isclose(out.iloc[1]["hour_sin"], 0.0, atol=1e-6)
    assert np.isclose(out.iloc[1]["hour_cos"], -1.0, atol=1e-6)

    # Hour 18: sin(3*pi/2) = -1, cos(3*pi/2) = 0
    assert np.isclose(out.iloc[2]["hour_sin"], -1.0, atol=1e-6)
    assert np.isclose(out.iloc[2]["hour_cos"], 0.0, atol=1e-6)

    # All sin/cos within [-1, 1]
    assert (out["hour_sin"] >= -1.0 - 1e-6).all() and (out["hour_sin"] <= 1.0 + 1e-6).all()
    assert (out["hour_cos"] >= -1.0 - 1e-6).all() and (out["hour_cos"] <= 1.0 + 1e-6).all()


def test_month_cyclical_features(synthetic_sample_df):
    """Verify sin/cos encoding for month conforms to trigonometric unit circle."""
    out = engineer_temporal_features(synthetic_sample_df)

    # Month 12: sin(2*pi) = 0, cos(2*pi) = 1
    assert np.isclose(out.iloc[0]["month_sin"], 0.0, atol=1e-6)
    assert np.isclose(out.iloc[0]["month_cos"], 1.0, atol=1e-6)

    # Month 6: sin(pi) = 0, cos(pi) = -1
    assert np.isclose(out.iloc[2]["month_sin"], 0.0, atol=1e-6)
    assert np.isclose(out.iloc[2]["month_cos"], -1.0, atol=1e-6)


def test_validation_rejects_invalid_hours(synthetic_sample_df):
    """Verify validate_raw_data raises ValueError on out-of-range Hour."""
    bad_df = synthetic_sample_df.copy()
    bad_df.loc[0, "Hour"] = 24
    with pytest.raises(ValueError, match="invalid Hour values"):
        validate_raw_data(bad_df)

    bad_df.loc[0, "Hour"] = -1
    with pytest.raises(ValueError, match="invalid Hour values"):
        validate_raw_data(bad_df)


def test_validation_rejects_negative_rentals(synthetic_sample_df):
    """Verify validate_raw_data raises ValueError on negative rental count."""
    bad_df = synthetic_sample_df.copy()
    bad_df.loc[0, "Rented Bike Count"] = -5
    with pytest.raises(ValueError, match="negative rental counts"):
        validate_raw_data(bad_df)


def test_validation_rejects_invalid_categorical(synthetic_sample_df):
    """Verify validate_raw_data catches illegal categorical domain values."""
    bad_df = synthetic_sample_df.copy()
    bad_df.loc[0, "Seasons"] = "Monsoon"
    with pytest.raises(ValueError, match="Unexpected values in 'Seasons'"):
        validate_raw_data(bad_df)

    bad_df2 = synthetic_sample_df.copy()
    bad_df2.loc[0, "Functioning Day"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected values in 'Functioning Day'"):
        validate_raw_data(bad_df2)


def test_validation_rejects_duplicate_timestamps(synthetic_sample_df):
    """Verify validate_raw_data catches duplicate (Date, Hour) combinations."""
    dup_df = pd.concat([synthetic_sample_df, synthetic_sample_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*timestamps"):
        validate_raw_data(dup_df)


# ---------------------------------------------------------------------------
# Integration Tests on Processed Dataset
# ---------------------------------------------------------------------------


def test_real_dataset_row_and_column_count(processed_df):
    """Verify processed dataset preserves all 8,760 records and has 23 columns."""
    assert len(processed_df) == 8760, f"Expected 8,760 rows, got {len(processed_df)}"
    assert len(processed_df.columns) == 23, f"Expected 23 columns, got {len(processed_df.columns)}"


def test_real_dataset_target_preservation(processed_df, raw_df):
    """Verify target column 'Rented Bike Count' exists and values match raw data exactly."""
    assert "Rented Bike Count" in processed_df.columns
    assert (processed_df["Rented Bike Count"] >= 0).all()
    # Sorted order should match since raw was already sorted
    assert (processed_df["Rented Bike Count"].values == raw_df["Rented Bike Count"].values).all()


def test_real_dataset_zero_demand_preservation(processed_df):
    """Verify exactly 295 zero-demand records, all occurring when Functioning Day == 'No'."""
    zero_mask = processed_df["Rented Bike Count"] == 0
    assert zero_mask.sum() == 295, f"Expected 295 zero-demand records, got {zero_mask.sum()}"

    # Verify all zeros occur when Functioning Day == 'No'
    zeros_func_no = (processed_df[zero_mask]["Functioning Day"] == "No").sum()
    assert zeros_func_no == 295, f"Expected all 295 zeros to have Functioning Day == 'No', got {zeros_func_no}"

    # Verify when Functioning Day == 'Yes', no zero demand exists
    func_yes_zeros = (processed_df[processed_df["Functioning Day"] == "Yes"]["Rented Bike Count"] == 0).sum()
    assert func_yes_zeros == 0, f"Expected 0 zeros when Functioning Day == 'Yes', got {func_yes_zeros}"


def test_real_dataset_missing_values(processed_df):
    """Verify 0 missing values across all columns."""
    missing_count = int(processed_df.isnull().sum().sum())
    assert missing_count == 0, f"Expected 0 missing values in processed dataset, found {missing_count}"


def test_real_dataset_duplicate_timestamps(processed_df):
    """Verify 0 duplicate (Date, Hour) combinations."""
    ts = processed_df["Date"].astype(str) + "_" + processed_df["Hour"].astype(str)
    assert not ts.duplicated().any(), f"Found {ts.duplicated().sum()} duplicate timestamps in processed dataset."


def test_metadata_file_validity():
    """Verify preprocessing_metadata.json is valid and contains required fields."""
    assert os.path.exists(DEFAULT_METADATA_PATH), f"Metadata file missing: {DEFAULT_METADATA_PATH}"
    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "raw_input_path",
        "processed_output_path",
        "input_row_count",
        "output_row_count",
        "features_added",
        "target_name",
        "date_format",
        "missing_values_count",
        "duplicate_timestamps",
        "zero_demand_records",
        "zero_demand_analysis",
    ]
    for k in required_keys:
        assert k in meta, f"Metadata JSON missing key: {k}"

    assert meta["input_row_count"] == 8760
    assert meta["output_row_count"] == 8760
    assert meta["missing_values_count"] == 0
    assert meta["duplicate_timestamps"] == 0
    assert meta["zero_demand_records"] == 295
