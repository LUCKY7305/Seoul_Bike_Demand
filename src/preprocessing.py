"""
Data Preprocessing Module for Vehicle Rental Demand Prediction
==============================================================

Phase 2: Loads raw dataset, validates schema and values, performs date parsing,
temporal and cyclical feature engineering, verifies zero-demand and integrity,
and saves the processed dataset along with metadata.

Usage:
    python -m src.preprocessing --input "data/raw/SeoulBikeData.csv" --output "data/processed/seoul_bike_processed.csv"
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Date",
    "Rented Bike Count",
    "Hour",
    "Seasons",
    "Holiday",
    "Functioning Day",
]

EXPECTED_SEASONS = {"Winter", "Spring", "Summer", "Autumn"}
EXPECTED_HOLIDAYS = {"No Holiday", "Holiday"}
EXPECTED_FUNCTIONING_DAYS = {"Yes", "No"}

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig"]

DEFAULT_RAW_PATH = "data/raw/SeoulBikeData.csv"
DEFAULT_PROCESSED_PATH = "data/processed/seoul_bike_processed.csv"
DEFAULT_METADATA_PATH = "data/processed/preprocessing_metadata.json"


# ---------------------------------------------------------------------------
# Loading and Encoding
# ---------------------------------------------------------------------------


def file_md5(path: str) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_data(filepath: str, encoding: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load raw CSV dataset trying several encodings if not specified.
    Returns (DataFrame, used_encoding).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")

    encodings = [encoding] if encoding else ENCODINGS_TO_TRY
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df, enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e

    raise RuntimeError(
        f"Failed to read {filepath} with encodings {encodings}. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Validate the raw DataFrame against expectations:
    - Required columns present
    - Non-empty DataFrame
    - Valid Hour range (0-23)
    - Non-negative target values
    - Expected categorical domain values
    - No duplicate (Date, Hour) timestamps

    Raises ValueError if any assertion fails.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    # Check Hour range
    if not pd.api.types.is_integer_dtype(df["Hour"]):
        raise ValueError("Hour column must be integer type.")
    if (df["Hour"] < 0).any() or (df["Hour"] > 23).any():
        invalid_hours = df[~df["Hour"].between(0, 23)]["Hour"].tolist()
        raise ValueError(f"Found invalid Hour values outside [0, 23]: {invalid_hours[:10]}")

    # Check target non-negativity
    target = df["Rented Bike Count"]
    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError("Target 'Rented Bike Count' must be numeric.")
    if (target < 0).any():
        neg_count = int((target < 0).sum())
        raise ValueError(f"Found {neg_count} negative rental counts in target column.")

    # Check categorical domains
    seasons = set(df["Seasons"].unique())
    if not seasons.issubset(EXPECTED_SEASONS):
        raise ValueError(f"Unexpected values in 'Seasons': {seasons - EXPECTED_SEASONS}")

    holidays = set(df["Holiday"].unique())
    if not holidays.issubset(EXPECTED_HOLIDAYS):
        raise ValueError(f"Unexpected values in 'Holiday': {holidays - EXPECTED_HOLIDAYS}")

    func_days = set(df["Functioning Day"].unique())
    if not func_days.issubset(EXPECTED_FUNCTIONING_DAYS):
        raise ValueError(
            f"Unexpected values in 'Functioning Day': {func_days - EXPECTED_FUNCTIONING_DAYS}"
        )

    # Check duplicate timestamps in raw data
    raw_ts = df["Date"].astype(str) + "_" + df["Hour"].astype(str)
    if raw_ts.duplicated().any():
        dup_count = int(raw_ts.duplicated().sum())
        raise ValueError(f"Found {dup_count} duplicate (Date, Hour) timestamps in raw data.")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Date and add engineered temporal and cyclical features:
    - day_of_week: Monday=0, Sunday=6
    - month: 1 to 12
    - day_of_month: 1 to 31
    - day_of_year: 1 to 365/366
    - is_weekend: 1 if day_of_week in [5, 6] else 0
    - hour_sin: sin(2 * pi * Hour / 24)
    - hour_cos: cos(2 * pi * Hour / 24)
    - month_sin: sin(2 * pi * month / 12)
    - month_cos: cos(2 * pi * month / 12)

    Standardizes Date column to 'YYYY-MM-DD' string format.
    Ensures observations are chronologically sorted.
    """
    out = df.copy()

    # Parse date explicitly as dd/mm/yyyy
    parsed_date = pd.to_datetime(out["Date"], format="%d/%m/%Y", errors="coerce")
    if parsed_date.isna().any():
        nat_count = int(parsed_date.isna().sum())
        raise ValueError(f"Failed to parse {nat_count} dates using format '%d/%m/%Y'.")

    # Add temporal columns
    out["_parsed_date"] = parsed_date
    out["day_of_week"] = parsed_date.dt.day_of_week.astype(int)
    out["month"] = parsed_date.dt.month.astype(int)
    out["day_of_month"] = parsed_date.dt.day.astype(int)
    out["day_of_year"] = parsed_date.dt.day_of_year.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # Cyclical hour encoding (24-hour cycle)
    out["hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24.0)

    # Cyclical month encoding (12-month cycle)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    # Sort chronologically by parsed date and Hour
    out = out.sort_values(by=["_parsed_date", "Hour"]).reset_index(drop=True)

    # Format Date as standard unambiguous YYYY-MM-DD string
    out["Date"] = out["_parsed_date"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["_parsed_date"])

    # Verify no nulls were introduced
    if out.isnull().any().any():
        null_cols = out.columns[out.isnull().any()].tolist()
        raise ValueError(f"Null values introduced during feature engineering in: {null_cols}")

    return out


# ---------------------------------------------------------------------------
# Zero Demand Analysis Helper
# ---------------------------------------------------------------------------


def analyze_zero_demand(df: pd.DataFrame) -> Dict:
    """
    Analyze zero-demand observations in relation to Functioning Day,
    seasons, and other features.
    """
    zero_mask = df["Rented Bike Count"] == 0
    total_zeros = int(zero_mask.sum())
    func_counts = df.groupby("Functioning Day")["Rented Bike Count"].apply(
        lambda s: int((s == 0).sum())
    ).to_dict()
    total_by_func = df["Functioning Day"].value_counts().to_dict()

    zeros_no = func_counts.get("No", 0)
    zeros_yes = func_counts.get("Yes", 0)
    tot_no = total_by_func.get("No", 0)
    tot_yes = total_by_func.get("Yes", 0)

    season_breakdown = df[zero_mask]["Seasons"].value_counts().to_dict()

    return {
        "total_zero_observations": total_zeros,
        "zero_demand_when_functioning_no": zeros_no,
        "zero_demand_when_functioning_yes": zeros_yes,
        "total_functioning_no_records": tot_no,
        "total_functioning_yes_records": tot_yes,
        "prop_zeros_in_functioning_no": (zeros_no / tot_no) if tot_no > 0 else 0.0,
        "prop_zeros_in_functioning_yes": (zeros_yes / tot_yes) if tot_yes > 0 else 0.0,
        "season_breakdown_of_zeros": season_breakdown,
    }


# ---------------------------------------------------------------------------
# Save and Reload Verification
# ---------------------------------------------------------------------------


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str,
    metadata_path: str,
    raw_input_path: str,
    features_added: List[str],
) -> Dict:
    """
    Save processed dataset to CSV and create metadata JSON.
    Reloads the CSV to verify schema and row count match.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")

    # Reload to verify
    reloaded = pd.read_csv(output_path, encoding="utf-8")
    if len(reloaded) != len(df):
        raise ValueError(
            f"Reload verification failed: row count mismatch ({len(reloaded)} vs {len(df)})."
        )
    if list(reloaded.columns) != list(df.columns):
        raise ValueError("Reload verification failed: column mismatch.")

    # Create metadata
    metadata = {
        "raw_input_path": raw_input_path,
        "raw_md5_hash": file_md5(raw_input_path) if os.path.exists(raw_input_path) else None,
        "processed_output_path": output_path,
        "processed_md5_hash": file_md5(output_path),
        "input_row_count": len(df),
        "output_row_count": len(reloaded),
        "input_col_count": len(df.columns) - len(features_added),
        "output_col_count": len(reloaded.columns),
        "features_added": features_added,
        "target_name": "Rented Bike Count",
        "date_format": "YYYY-MM-DD",
        "missing_values_count": int(reloaded.isnull().sum().sum()),
        "duplicate_timestamps": int(
            (reloaded["Date"].astype(str) + "_" + reloaded["Hour"].astype(str))
            .duplicated()
            .sum()
        ),
        "zero_demand_records": int((reloaded["Rented Bike Count"] == 0).sum()),
        "generation_timestamp": datetime.now().isoformat(),
        "columns_list": list(reloaded.columns),
    }

    meta_dir = Path(metadata_path).parent
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


def run_preprocessing_pipeline(
    raw_path: str = DEFAULT_RAW_PATH,
    output_path: str = DEFAULT_PROCESSED_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute full preprocessing pipeline.
    Returns (processed_df, metadata_dict).
    """
    print(f"[INFO] Loading raw data from: {raw_path}")
    raw_hash_before = file_md5(raw_path)
    df_raw, encoding_used = load_raw_data(raw_path)
    print(f"       Loaded {len(df_raw)} rows, {len(df_raw.columns)} cols (encoding={encoding_used})")

    print("[INFO] Validating raw data...")
    validate_raw_data(df_raw)
    print("       Raw data validation passed.")

    print("[INFO] Engineering temporal and cyclical features...")
    cols_before = set(df_raw.columns)
    df_processed = engineer_temporal_features(df_raw)
    features_added = [c for c in df_processed.columns if c not in cols_before]
    print(f"       Added {len(features_added)} features: {features_added}")
    print(f"       Processed shape: {df_processed.shape}")

    print("[INFO] Analyzing zero-demand observations...")
    zero_analysis = analyze_zero_demand(df_processed)
    print(f"       Total zero-demand records: {zero_analysis['total_zero_observations']}")
    print(f"       Functioning Day == 'No':  {zero_analysis['zero_demand_when_functioning_no']}")
    print(f"       Functioning Day == 'Yes': {zero_analysis['zero_demand_when_functioning_yes']}")

    print(f"[INFO] Saving processed dataset to: {output_path}")
    metadata = save_processed_dataset(
        df_processed, output_path, metadata_path, raw_path, features_added
    )
    metadata["zero_demand_analysis"] = zero_analysis

    # Re-verify original file hash to ensure no corruption
    raw_hash_after = file_md5(raw_path)
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("[CRITICAL] Raw dataset was modified during processing!")
    print("       Original raw dataset integrity verified (MD5 match).")

    # Update metadata with zero demand analysis
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Metadata saved to: {metadata_path}")
    print("[INFO] Preprocessing pipeline finished successfully.")
    return df_processed, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Seoul Bike Sharing dataset for demand prediction."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_RAW_PATH,
        help=f"Path to raw CSV dataset (default: {DEFAULT_RAW_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_PROCESSED_PATH,
        help=f"Path for processed CSV output (default: {DEFAULT_PROCESSED_PATH})",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"Path for metadata JSON (default: {DEFAULT_METADATA_PATH})",
    )
    args = parser.parse_args()

    try:
        run_preprocessing_pipeline(args.input, args.output, args.metadata)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
