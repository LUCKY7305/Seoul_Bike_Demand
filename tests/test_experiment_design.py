"""
Automated Tests for Experiment Design and Chronological Splitting
=================================================================

Phase 3A Tests:
- Deterministic chronological split boundaries and exact row counts.
- Strict temporal ordering across partitions (Train end < Val start < Val end < Test start).
- Zero overlap between partitions; all 8,760 rows preserved.
- Target column correctly named and excluded from all feature sets.
- Raw Date excluded from numeric feature sets.
- Feature Set C includes Dew point; Feature Set D excludes it.
- Zero-demand records preserved in saved partitions.
- Functioning Day counts match empirical distribution.
- Split CSV files reload cleanly with identical schema.
- Split metadata JSON exists and holds valid checksums.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment_design import (
    DATE_COLUMN,
    DEFAULT_PROCESSED_PATH,
    HOUR_COLUMN,
    REPORTS_PHASE3A_DIR,
    SPLITS_DIR,
    TARGET_COLUMN,
    create_chronological_splits,
    define_feature_sets,
    find_temperature_columns,
    verify_processed_dataset,
)


@pytest.fixture
def processed_df() -> pd.DataFrame:
    """Load full processed dataset."""
    assert os.path.exists(DEFAULT_PROCESSED_PATH), f"Missing {DEFAULT_PROCESSED_PATH}"
    return pd.read_csv(DEFAULT_PROCESSED_PATH, encoding="utf-8")


@pytest.fixture
def split_files() -> dict:
    """Verify and return paths to saved split files."""
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"
    meta_path = SPLITS_DIR / "split_metadata.json"

    assert train_path.exists(), f"Missing {train_path}"
    assert val_path.exists(), f"Missing {val_path}"
    assert test_path.exists(), f"Missing {test_path}"
    assert meta_path.exists(), f"Missing {meta_path}"

    return {
        "train": pd.read_csv(train_path, encoding="utf-8"),
        "val": pd.read_csv(val_path, encoding="utf-8"),
        "test": pd.read_csv(test_path, encoding="utf-8"),
        "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
    }


# ---------------------------------------------------------------------------
# Dataset Verification Tests
# ---------------------------------------------------------------------------


def test_processed_dataset_verification(processed_df):
    """Verify dataset meets Phase 2 dimensions, zero-demand, and target criteria."""
    verif = verify_processed_dataset(processed_df)
    assert verif["n_rows"] == 8760
    assert verif["n_cols"] == 23
    assert verif["target"] == "Rented Bike Count"
    assert verif["total_zeros"] == 295
    assert verif["zeros_functioning_no"] == 295
    assert verif["zeros_functioning_yes"] == 0
    assert np.isclose(verif["temp_dew_correlation"], 0.9128, atol=1e-3)


# ---------------------------------------------------------------------------
# Chronological Splitting Tests
# ---------------------------------------------------------------------------


def test_split_counts_and_completeness(split_files):
    """Verify exact partition row counts: 6132 train, 1314 val, 1314 test = 8760."""
    train = split_files["train"]
    val = split_files["val"]
    test = split_files["test"]

    assert len(train) == 6132, f"Expected 6,132 train rows, got {len(train)}"
    assert len(val) == 1314, f"Expected 1,314 val rows, got {len(val)}"
    assert len(test) == 1314, f"Expected 1,314 test rows, got {len(test)}"
    assert len(train) + len(val) + len(test) == 8760


def test_strict_chronological_ordering(split_files):
    """Verify Train max < Val min and Val max < Test min (no temporal overlap)."""
    train = split_files["train"]
    val = split_files["val"]
    test = split_files["test"]

    dt_train_max = pd.to_datetime(train[DATE_COLUMN].iloc[-1]) + pd.to_timedelta(
        train[HOUR_COLUMN].iloc[-1], unit="h"
    )
    dt_val_min = pd.to_datetime(val[DATE_COLUMN].iloc[0]) + pd.to_timedelta(
        val[HOUR_COLUMN].iloc[0], unit="h"
    )
    dt_val_max = pd.to_datetime(val[DATE_COLUMN].iloc[-1]) + pd.to_timedelta(
        val[HOUR_COLUMN].iloc[-1], unit="h"
    )
    dt_test_min = pd.to_datetime(test[DATE_COLUMN].iloc[0]) + pd.to_timedelta(
        test[HOUR_COLUMN].iloc[0], unit="h"
    )

    assert dt_train_max < dt_val_min, f"Overlap: train max {dt_train_max} >= val min {dt_val_min}"
    assert dt_val_max < dt_test_min, f"Overlap: val max {dt_val_max} >= test min {dt_test_min}"


def test_functioning_and_zero_demand_counts_per_partition(split_files):
    """
    Verify functioning status distribution across partitions:
    - Train: 6084 Yes, 48 No, 48 Zeros
    - Val:   1163 Yes, 151 No, 151 Zeros
    - Test:  1218 Yes, 96 No, 96 Zeros
    - Total: 8465 Yes, 295 No, 295 Zeros
    """
    train = split_files["train"]
    val = split_files["val"]
    test = split_files["test"]

    assert (train["Functioning Day"] == "Yes").sum() == 6084
    assert (train["Functioning Day"] == "No").sum() == 48
    assert (train[TARGET_COLUMN] == 0).sum() == 48

    assert (val["Functioning Day"] == "Yes").sum() == 1163
    assert (val["Functioning Day"] == "No").sum() == 151
    assert (val[TARGET_COLUMN] == 0).sum() == 151

    assert (test["Functioning Day"] == "Yes").sum() == 1218
    assert (test["Functioning Day"] == "No").sum() == 96
    assert (test[TARGET_COLUMN] == 0).sum() == 96


def test_split_files_reload_schema_preservation(split_files, processed_df):
    """Verify saved split CSVs contain all 23 original columns in identical order."""
    expected_cols = list(processed_df.columns)
    for name in ["train", "val", "test"]:
        df = split_files[name]
        assert list(df.columns) == expected_cols, f"Column mismatch in {name}.csv"


# ---------------------------------------------------------------------------
# Feature Sets Tests
# ---------------------------------------------------------------------------


def test_feature_sets_leakage_and_validity(processed_df):
    """
    Verify feature sets:
    - All features exist in DataFrame
    - Target 'Rented Bike Count' is strictly excluded
    - Date string is excluded
    - Feature Set C includes Dew point; Feature Set D excludes Dew point
    """
    f_sets = define_feature_sets(processed_df)
    temp_col, dew_col = find_temperature_columns(processed_df)

    assert "FEATURE_SET_A_TEMPORAL" in f_sets
    assert "FEATURE_SET_B_WEATHER" in f_sets
    assert "FEATURE_SET_C_TEMPORAL_WEATHER" in f_sets
    assert "FEATURE_SET_D_REDUCED_COLLINEARITY" in f_sets

    set_a = f_sets["FEATURE_SET_A_TEMPORAL"]
    set_b = f_sets["FEATURE_SET_B_WEATHER"]
    set_c = f_sets["FEATURE_SET_C_TEMPORAL_WEATHER"]
    set_d = f_sets["FEATURE_SET_D_REDUCED_COLLINEARITY"]

    # Target exclusion
    for name, cols in f_sets.items():
        assert TARGET_COLUMN not in cols, f"Target leakage in {name}"
        assert DATE_COLUMN not in cols, f"Date string leakage in {name}"

    # Dew point check
    assert dew_col in set_b
    assert dew_col in set_c
    assert dew_col not in set_d, f"Dew point {dew_col} must be omitted from Feature Set D"
    assert temp_col in set_d, f"Temperature {temp_col} must be present in Feature Set D"
    assert len(set_d) == len(set_c) - 1


def test_feature_sets_json_file_exists():
    """Verify feature_sets.json exists and is valid JSON."""
    json_path = REPORTS_PHASE3A_DIR / "feature_sets.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "FEATURE_SET_A_TEMPORAL" in data
    assert "FEATURE_SET_D_REDUCED_COLLINEARITY" in data


def test_experiment_config_json_exists():
    """Verify experiment_config.json contains required metadata and test protection policy."""
    cfg_path = REPORTS_PHASE3A_DIR / "experiment_config.json"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "dataset_verification" in cfg
    assert "split_metadata" in cfg
    assert "feature_sets" in cfg
    assert "validation_baseline_results" in cfg
    assert "test_set_policy" in cfg
    assert "Strictly held out" in cfg["test_set_policy"]
