"""
Data Inspection Script for Vehicle Rental Demand Prediction Project
====================================================================

Phase 1: Loads the dataset, performs thorough quality checks, and generates
a Markdown report and a JSON summary. Does NOT modify the original data.

Usage:
    python -m src.data_inspection --data "data/raw/SeoulBikeData.csv"
    python -m src.data_inspection --data "data/raw/SeoulBikeData.zip"
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_TARGET_NAMES = [
    "Rented Bike Count",
    "rented_bike_count",
    "rental_count",
    "count",
    "cnt",
    "bike_count",
    "demand",
]

KNOWN_DATE_NAMES = [
    "Date",
    "date",
    "datetime",
    "Datetime",
    "timestamp",
    "Timestamp",
    "dteday",
]

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig"]

REPORT_DIR = Path("reports")
REPORT_MD = REPORT_DIR / "phase1_dataset_report.md"
REPORT_JSON = REPORT_DIR / "phase1_dataset_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    """Print a visible section header to the console."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def file_md5(path: str) -> str:
    """Compute the MD5 hash of a file (for integrity tracking)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def try_read_csv(path: str) -> pd.DataFrame:
    """Attempt to read a CSV with several encodings, returning the first that works."""
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  [INFO] Successfully read CSV with encoding: {enc}")
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
    raise RuntimeError(
        f"Could not read {path} with any of {ENCODINGS_TO_TRY}. Last error: {last_err}"
    )


def resolve_csv_from_zip(zip_path: str) -> tuple[str, pd.DataFrame]:
    """Extract and read the first CSV found inside a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV file found inside {zip_path}")
        if len(csv_names) > 1:
            print(f"  [WARN] Multiple CSVs in archive: {csv_names}. Using first: {csv_names[0]}")
        chosen = csv_names[0]
        raw_bytes = zf.read(chosen)

    # Try encodings
    for enc in ENCODINGS_TO_TRY:
        try:
            text = raw_bytes.decode(enc)
            df = pd.read_csv(StringIO(text))
            print(f"  [INFO] Read {chosen} from ZIP with encoding: {enc}")
            return chosen, df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Could not decode {chosen} from ZIP with any known encoding.")


def detect_target_column(columns: list[str]) -> str | None:
    """Try to identify the target column from known names (case-insensitive)."""
    col_lower = {c.lower().strip(): c for c in columns}
    for known in KNOWN_TARGET_NAMES:
        if known.lower() in col_lower:
            return col_lower[known.lower()]
    return None


def detect_date_column(columns: list[str]) -> str | None:
    """Try to identify a date/timestamp column."""
    col_lower = {c.lower().strip(): c for c in columns}
    for known in KNOWN_DATE_NAMES:
        if known.lower() in col_lower:
            return col_lower[known.lower()]
    return None


def parse_dates(series: pd.Series) -> pd.Series | None:
    """Attempt to parse a series as dates, returning None on failure."""
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", None]:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Core Inspection Logic
# ---------------------------------------------------------------------------


def inspect_dataset(df: pd.DataFrame, filename: str, file_hash: str) -> dict:
    """
    Perform comprehensive inspection of the dataset.
    Returns a dictionary with all findings.
    """
    findings = {}
    n_rows, n_cols = df.shape

    # --- Basic info ---
    print_section("DATASET OVERVIEW")
    print(f"  Filename       : {filename}")
    print(f"  MD5 Hash       : {file_hash}")
    print(f"  Rows           : {n_rows:,}")
    print(f"  Columns        : {n_cols}")
    print(f"  Memory Usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    findings["filename"] = filename
    findings["md5_hash"] = file_hash
    findings["n_rows"] = n_rows
    findings["n_cols"] = n_cols
    findings["memory_kb"] = round(df.memory_usage(deep=True).sum() / 1024, 1)

    # --- Column names and types ---
    print_section("COLUMNS AND DATA TYPES")
    col_info = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        print(f"  {col:40s} {dtype_str}")
        col_info[col] = dtype_str
    findings["columns"] = col_info

    # --- First and last records ---
    print_section("FIRST 5 RECORDS")
    print(df.head().to_string(index=True))

    print_section("LAST 5 RECORDS")
    print(df.tail().to_string(index=True))

    findings["head"] = df.head().to_dict(orient="records")
    findings["tail"] = df.tail().to_dict(orient="records")

    # --- Missing values ---
    print_section("MISSING VALUES")
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    print(missing.to_string())
    print(f"\n  Total missing cells: {total_missing}")
    findings["missing_values"] = {col: int(v) for col, v in missing.items()}
    findings["total_missing"] = total_missing

    # --- Duplicate rows ---
    print_section("DUPLICATE ROWS")
    n_dup = int(df.duplicated().sum())
    print(f"  Exact duplicate rows: {n_dup}")
    findings["duplicate_rows"] = n_dup

    # --- Target column ---
    print_section("TARGET COLUMN DETECTION")
    target_col = detect_target_column(df.columns.tolist())
    if target_col:
        print(f"  Detected target column: '{target_col}'")
        t = df[target_col]
        target_stats = {
            "name": target_col,
            "dtype": str(t.dtype),
            "min": float(t.min()) if pd.api.types.is_numeric_dtype(t) else None,
            "max": float(t.max()) if pd.api.types.is_numeric_dtype(t) else None,
            "mean": round(float(t.mean()), 2) if pd.api.types.is_numeric_dtype(t) else None,
            "median": float(t.median()) if pd.api.types.is_numeric_dtype(t) else None,
            "std": round(float(t.std()), 2) if pd.api.types.is_numeric_dtype(t) else None,
            "zeros": int((t == 0).sum()) if pd.api.types.is_numeric_dtype(t) else None,
            "missing": int(t.isnull().sum()),
        }
        print(f"    dtype  : {target_stats['dtype']}")
        print(f"    min    : {target_stats['min']}")
        print(f"    max    : {target_stats['max']}")
        print(f"    mean   : {target_stats['mean']}")
        print(f"    median : {target_stats['median']}")
        print(f"    std    : {target_stats['std']}")
        print(f"    zeros  : {target_stats['zeros']}")
        print(f"    missing: {target_stats['missing']}")
        findings["target"] = target_stats
    else:
        print("  [WARNING] Could not automatically detect the target column.")
        print(f"  Available columns: {df.columns.tolist()}")
        findings["target"] = {"name": None, "warning": "Target column not auto-detected"}

    # --- Date analysis ---
    print_section("DATE / TEMPORAL ANALYSIS")
    date_col = detect_date_column(df.columns.tolist())
    date_info = {}
    if date_col:
        print(f"  Date column found: '{date_col}'")
        dates_parsed = parse_dates(df[date_col])
        if dates_parsed is not None:
            min_date = dates_parsed.min()
            max_date = dates_parsed.max()
            n_unique_dates = dates_parsed.dt.date.nunique()
            print(f"    Min date     : {min_date.strftime('%Y-%m-%d')}")
            print(f"    Max date     : {max_date.strftime('%Y-%m-%d')}")
            print(f"    Unique dates : {n_unique_dates}")
            date_info["column"] = date_col
            date_info["min_date"] = min_date.strftime("%Y-%m-%d")
            date_info["max_date"] = max_date.strftime("%Y-%m-%d")
            date_info["unique_dates"] = n_unique_dates

            # Duplicate timestamps (Date + Hour if available)
            hour_col = None
            if "Hour" in df.columns:
                hour_col = "Hour"
            elif "hour" in df.columns:
                hour_col = "hour"
            elif "hr" in df.columns:
                hour_col = "hr"

            if hour_col:
                ts_key = dates_parsed.astype(str) + "_" + df[hour_col].astype(str)
                dup_ts = int(ts_key.duplicated().sum())
                print(f"    Duplicate timestamps (Date+Hour): {dup_ts}")
                date_info["duplicate_timestamps"] = dup_ts
                date_info["hour_column"] = hour_col
            else:
                dup_ts_date_only = int(dates_parsed.duplicated().sum())
                print(f"    Duplicate date entries: {dup_ts_date_only}")
                date_info["duplicate_dates"] = dup_ts_date_only
        else:
            print(f"    [WARN] Could not parse '{date_col}' as dates.")
            date_info["column"] = date_col
            date_info["warning"] = "Could not parse dates"
    else:
        print("  No obvious date column detected.")
        date_info["warning"] = "No date column detected"
    findings["dates"] = date_info

    # --- Descriptive statistics ---
    print_section("DESCRIPTIVE STATISTICS (NUMERICAL)")
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        desc = num_df.describe()
        print(desc.to_string())
        # Convert to serialisable dict
        findings["descriptive_stats"] = {
            col: {str(idx): round(float(val), 4) for idx, val in desc[col].items()}
            for col in desc.columns
        }
    else:
        print("  No numerical columns found.")
        findings["descriptive_stats"] = {}

    # --- Negative / suspicious values ---
    print_section("NEGATIVE VALUES IN NUMERICAL COLUMNS")
    neg_report = {}
    for col in num_df.columns:
        neg_count = int((num_df[col] < 0).sum())
        if neg_count > 0:
            print(f"  {col}: {neg_count:,} negative values (min = {num_df[col].min()})")
            neg_report[col] = {"count": neg_count, "min_value": float(num_df[col].min())}
    if not neg_report:
        print("  No negative values found in numerical columns.")
    findings["negative_values"] = neg_report

    # Note about legitimate negatives
    if neg_report:
        print("\n  [NOTE] Negative temperatures are expected in Seoul's climate.")
        print("         These are NOT data errors. Do not remove automatically.")

    # --- Categorical columns ---
    print_section("CATEGORICAL COLUMN ANALYSIS")
    cat_df = df.select_dtypes(include="object")
    cat_report = {}
    for col in cat_df.columns:
        if col == date_col:
            continue  # Skip the date column
        uniques = cat_df[col].unique().tolist()
        n_unique = len(uniques)
        print(f"  {col}: {n_unique} unique values -> {uniques}")
        cat_report[col] = {"n_unique": n_unique, "values": uniques}
    findings["categorical_columns"] = cat_report

    # --- Summary ---
    print_section("INSPECTION COMPLETE")
    issues = []
    if total_missing > 0:
        issues.append(f"{total_missing} missing values detected")
    if n_dup > 0:
        issues.append(f"{n_dup} duplicate rows detected")
    if not target_col:
        issues.append("Target column not auto-detected")

    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    [!] {issue}")
    else:
        print("  [OK] No critical issues found. Dataset is ready for Phase 2.")

    findings["issues"] = issues
    findings["inspection_timestamp"] = datetime.now().isoformat()

    return findings


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_markdown_report(findings: dict) -> str:
    """Generate a comprehensive Markdown report from inspection findings."""
    lines = []
    a = lines.append

    a("# Phase 1 — Dataset Inspection Report")
    a("")
    a(f"**Generated**: {findings['inspection_timestamp']}")
    a("")

    # Overview
    a("## 1. Dataset Overview")
    a("")
    a(f"| Property | Value |")
    a(f"|----------|-------|")
    a(f"| Filename | `{findings['filename']}` |")
    a(f"| MD5 Hash | `{findings['md5_hash']}` |")
    a(f"| Rows | {findings['n_rows']:,} |")
    a(f"| Columns | {findings['n_cols']} |")
    a(f"| Memory | {findings['memory_kb']:.1f} KB |")
    a("")

    # Columns
    a("## 2. Columns and Data Types")
    a("")
    a("| # | Column Name | Data Type |")
    a("|---|-------------|-----------|")
    for i, (col, dtype) in enumerate(findings["columns"].items(), 1):
        a(f"| {i} | `{col}` | {dtype} |")
    a("")

    # Target
    a("## 3. Target Variable")
    a("")
    target = findings.get("target", {})
    if target.get("name"):
        a(f"**Detected target column**: `{target['name']}`")
        a("")
        a(f"| Statistic | Value |")
        a(f"|-----------|-------|")
        for key in ["dtype", "min", "max", "mean", "median", "std", "zeros", "missing"]:
            if key in target:
                a(f"| {key.capitalize()} | {target[key]} |")
        a("")
        if target.get("zeros") and target["zeros"] > 0:
            a(f"> **Note**: {target['zeros']} records have a rental count of zero. "
              f"These likely represent hours with no demand (e.g., late night, non-functioning days) "
              f"and should **not** be treated as data errors.")
            a("")
    else:
        a("⚠ **Target column could not be automatically identified.**")
        a("")

    # Date coverage
    a("## 4. Date and Temporal Coverage")
    a("")
    dates = findings.get("dates", {})
    if dates.get("min_date"):
        a(f"| Property | Value |")
        a(f"|----------|-------|")
        a(f"| Date Column | `{dates.get('column', 'N/A')}` |")
        a(f"| Min Date | {dates['min_date']} |")
        a(f"| Max Date | {dates['max_date']} |")
        a(f"| Unique Dates | {dates.get('unique_dates', 'N/A')} |")
        if "duplicate_timestamps" in dates:
            a(f"| Duplicate Timestamps (Date+Hour) | {dates['duplicate_timestamps']} |")
        if "hour_column" in dates:
            a(f"| Hour Column | `{dates['hour_column']}` |")
        a("")
    else:
        a("No date analysis available.")
        a("")

    # Missing values
    a("## 5. Missing Values")
    a("")
    mv = findings.get("missing_values", {})
    if findings["total_missing"] == 0:
        a("✅ **No missing values detected in any column.**")
    else:
        a(f"Total missing cells: **{findings['total_missing']}**")
        a("")
        a("| Column | Missing Count |")
        a("|--------|---------------|")
        for col, cnt in mv.items():
            if cnt > 0:
                a(f"| `{col}` | {cnt} |")
    a("")

    # Duplicates
    a("## 6. Duplicate Rows")
    a("")
    a(f"Exact duplicate rows: **{findings['duplicate_rows']}**")
    a("")

    # Descriptive stats
    a("## 7. Descriptive Statistics (Numerical Columns)")
    a("")
    stats = findings.get("descriptive_stats", {})
    if stats:
        stat_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        cols = list(stats.keys())
        # Build table header
        header = "| Statistic | " + " | ".join(f"`{c}`" for c in cols) + " |"
        sep = "|-----------|" + "|".join(["------"] * len(cols)) + "|"
        a(header)
        a(sep)
        for sk in stat_keys:
            row = f"| **{sk}** |"
            for c in cols:
                val = stats[c].get(sk, "")
                if isinstance(val, float):
                    row += f" {val:,.2f} |"
                else:
                    row += f" {val} |"
            a(row)
        a("")
    else:
        a("No numerical columns to summarize.")
        a("")

    # Negative values
    a("## 8. Negative Values")
    a("")
    neg = findings.get("negative_values", {})
    if neg:
        a("| Column | Negative Count | Min Value |")
        a("|--------|---------------|-----------|")
        for col, info in neg.items():
            a(f"| `{col}` | {info['count']:,} | {info['min_value']} |")
        a("")
        a("> **Interpretation**: Negative temperatures are physically expected in Seoul's winter climate "
          "(December–February). These values are legitimate and should **not** be removed.")
    else:
        a("✅ No negative values in numerical columns.")
    a("")

    # Categorical
    a("## 9. Categorical Columns")
    a("")
    cat = findings.get("categorical_columns", {})
    if cat:
        for col, info in cat.items():
            a(f"### `{col}` ({info['n_unique']} unique)")
            a("")
            a(f"Values: {', '.join(f'`{v}`' for v in info['values'])}")
            a("")
    else:
        a("No categorical columns found (excluding date).")
        a("")

    # First and last rows
    a("## 10. Sample Records")
    a("")
    a("### First 5 Records")
    a("")
    head = findings.get("head", [])
    if head:
        cols = list(head[0].keys())
        a("| " + " | ".join(cols) + " |")
        a("|" + "|".join(["---"] * len(cols)) + "|")
        for rec in head:
            a("| " + " | ".join(str(rec.get(c, "")) for c in cols) + " |")
    a("")
    a("### Last 5 Records")
    a("")
    tail = findings.get("tail", [])
    if tail:
        cols = list(tail[0].keys())
        a("| " + " | ".join(cols) + " |")
        a("|" + "|".join(["---"] * len(cols)) + "|")
        for rec in tail:
            a("| " + " | ".join(str(rec.get(c, "")) for c in cols) + " |")
    a("")

    # Issues summary
    a("## 11. Issues Summary")
    a("")
    issues = findings.get("issues", [])
    if issues:
        for issue in issues:
            a(f"- ⚠ {issue}")
    else:
        a("✅ **No critical issues detected.** The dataset is clean and ready for Phase 2 processing.")
    a("")

    # Recommendations
    a("## 12. Recommendations for Phase 2")
    a("")
    a("1. **Parse the `Date` column** into a proper datetime type and extract temporal features "
      "(day of week, month, year, is_weekend).")
    a("2. **Encode categorical variables** (`Seasons`, `Holiday`, `Functioning Day`) "
      "using one-hot or ordinal encoding.")
    a("3. **Feature engineering**: Consider interaction features (e.g., temperature × humidity), "
      "cyclic encoding for `Hour`, and lag features if time-series structure is preserved.")
    a("4. **Handle zero-demand records**: Investigate the 295 records where `Rented Bike Count = 0`. "
      "Many correspond to non-functioning hours — decide whether to keep, filter, or flag them.")
    a("5. **Outlier analysis**: Review extreme values in `Rainfall(mm)` (max=35.0) and "
      "`Snowfall (cm)` (max=8.8) for potential impact on model training.")
    a("6. **Correlation analysis**: Compute pairwise correlations to identify multicollinearity "
      "and strong predictors before model selection.")
    a("")
    a("---")
    a(f"*Report generated by `src/data_inspection.py` on {findings['inspection_timestamp']}*")

    return "\n".join(lines)


def generate_json_summary(findings: dict) -> str:
    """Produce a JSON-serialisable summary (excludes head/tail for brevity)."""
    summary = {
        "filename": findings["filename"],
        "md5_hash": findings["md5_hash"],
        "n_rows": findings["n_rows"],
        "n_cols": findings["n_cols"],
        "memory_kb": findings["memory_kb"],
        "columns": findings["columns"],
        "target": findings.get("target"),
        "dates": findings.get("dates"),
        "total_missing": findings["total_missing"],
        "missing_values": findings["missing_values"],
        "duplicate_rows": findings["duplicate_rows"],
        "negative_values": findings["negative_values"],
        "categorical_columns": findings.get("categorical_columns"),
        "descriptive_stats": findings.get("descriptive_stats"),
        "issues": findings["issues"],
        "inspection_timestamp": findings["inspection_timestamp"],
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a vehicle-rental-demand dataset and generate quality reports."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the dataset file (CSV or ZIP containing CSV).",
    )
    args = parser.parse_args()
    data_path = args.data

    # ---- Validate file existence ----
    if not os.path.isfile(data_path):
        print(f"[ERROR] File not found: {data_path}")
        sys.exit(1)

    print(f"[INFO] Data path: {data_path}")
    print(f"[INFO] File size: {os.path.getsize(data_path):,} bytes")

    # ---- Compute file hash (before any processing) ----
    original_hash = file_md5(data_path)
    print(f"[INFO] MD5 hash : {original_hash}")

    # ---- Load dataset ----
    ext = os.path.splitext(data_path)[1].lower()
    filename = os.path.basename(data_path)

    if ext == ".zip":
        print("[INFO] Detected ZIP archive. Extracting CSV...")
        inner_name, df = resolve_csv_from_zip(data_path)
        filename = f"{filename} -> {inner_name}"
    elif ext == ".csv":
        print("[INFO] Detected CSV file. Loading...")
        df = try_read_csv(data_path)
    else:
        print(f"[ERROR] Unsupported file format: {ext}")
        print("        Supported formats: .csv, .zip")
        sys.exit(1)

    # ---- Run inspection ----
    findings = inspect_dataset(df, filename, original_hash)

    # ---- Generate reports ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    md_report = generate_markdown_report(findings)
    REPORT_MD.write_text(md_report, encoding="utf-8")
    print(f"\n[INFO] Markdown report saved to: {REPORT_MD}")

    json_summary = generate_json_summary(findings)
    REPORT_JSON.write_text(json_summary, encoding="utf-8")
    print(f"[INFO] JSON summary saved to:    {REPORT_JSON}")

    # ---- Verify original file integrity ----
    post_hash = file_md5(data_path)
    if post_hash == original_hash:
        print(f"\n[INFO] [OK] Original dataset integrity verified (MD5 unchanged).")
    else:
        print(f"\n[WARNING] [!!] Original dataset may have been modified!")
        print(f"          Before: {original_hash}")
        print(f"          After:  {post_hash}")

    print("\n[INFO] Phase 1 inspection complete.")


if __name__ == "__main__":
    main()
