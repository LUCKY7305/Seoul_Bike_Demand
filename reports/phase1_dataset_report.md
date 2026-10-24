# Phase 1 — Dataset Inspection Report

**Generated**: 2026-10-22T20:08:29.657831

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Filename | `SeoulBikeData.csv` |
| MD5 Hash | `bf632307dcb419f67df589faaa0318e1` |
| Rows | 8,760 |
| Columns | 14 |
| Memory | 2607.8 KB |

## 2. Columns and Data Types

| # | Column Name | Data Type |
|---|-------------|-----------|
| 1 | `Date` | object |
| 2 | `Rented Bike Count` | int64 |
| 3 | `Hour` | int64 |
| 4 | `Temperature(°C)` | float64 |
| 5 | `Humidity(%)` | int64 |
| 6 | `Wind speed (m/s)` | float64 |
| 7 | `Visibility (10m)` | int64 |
| 8 | `Dew point temperature(°C)` | float64 |
| 9 | `Solar Radiation (MJ/m2)` | float64 |
| 10 | `Rainfall(mm)` | float64 |
| 11 | `Snowfall (cm)` | float64 |
| 12 | `Seasons` | object |
| 13 | `Holiday` | object |
| 14 | `Functioning Day` | object |

## 3. Target Variable

**Detected target column**: `Rented Bike Count`

| Statistic | Value |
|-----------|-------|
| Dtype | int64 |
| Min | 0.0 |
| Max | 3556.0 |
| Mean | 704.6 |
| Median | 504.5 |
| Std | 645.0 |
| Zeros | 295 |
| Missing | 0 |

> **Note**: 295 records have a rental count of zero. These likely represent hours with no demand (e.g., late night, non-functioning days) and should **not** be treated as data errors.

## 4. Date and Temporal Coverage

| Property | Value |
|----------|-------|
| Date Column | `Date` |
| Min Date | 2017-12-01 |
| Max Date | 2018-11-30 |
| Unique Dates | 365 |
| Duplicate Timestamps (Date+Hour) | 0 |
| Hour Column | `Hour` |

## 5. Missing Values

✅ **No missing values detected in any column.**

## 6. Duplicate Rows

Exact duplicate rows: **0**

## 7. Descriptive Statistics (Numerical Columns)

| Statistic | `Rented Bike Count` | `Hour` | `Temperature(°C)` | `Humidity(%)` | `Wind speed (m/s)` | `Visibility (10m)` | `Dew point temperature(°C)` | `Solar Radiation (MJ/m2)` | `Rainfall(mm)` | `Snowfall (cm)` |
|-----------|------|------|------|------|------|------|------|------|------|------|
| **count** | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 | 8,760.00 |
| **mean** | 704.60 | 11.50 | 12.88 | 58.23 | 1.72 | 1,436.83 | 4.07 | 0.57 | 0.15 | 0.08 |
| **std** | 645.00 | 6.92 | 11.94 | 20.36 | 1.04 | 608.30 | 13.06 | 0.87 | 1.13 | 0.44 |
| **min** | 0.00 | 0.00 | -17.80 | 0.00 | 0.00 | 27.00 | -30.60 | 0.00 | 0.00 | 0.00 |
| **25%** | 191.00 | 5.75 | 3.50 | 42.00 | 0.90 | 940.00 | -4.70 | 0.00 | 0.00 | 0.00 |
| **50%** | 504.50 | 11.50 | 13.70 | 57.00 | 1.50 | 1,698.00 | 5.10 | 0.01 | 0.00 | 0.00 |
| **75%** | 1,065.25 | 17.25 | 22.50 | 74.00 | 2.30 | 2,000.00 | 14.80 | 0.93 | 0.00 | 0.00 |
| **max** | 3,556.00 | 23.00 | 39.40 | 98.00 | 7.40 | 2,000.00 | 27.20 | 3.52 | 35.00 | 8.80 |

## 8. Negative Values

| Column | Negative Count | Min Value |
|--------|---------------|-----------|
| `Temperature(°C)` | 1,433 | -17.8 |
| `Dew point temperature(°C)` | 3,138 | -30.6 |

> **Interpretation**: Negative temperatures are physically expected in Seoul's winter climate (December–February). These values are legitimate and should **not** be removed.

## 9. Categorical Columns

### `Seasons` (4 unique)

Values: `Winter`, `Spring`, `Summer`, `Autumn`

### `Holiday` (2 unique)

Values: `No Holiday`, `Holiday`

### `Functioning Day` (2 unique)

Values: `Yes`, `No`

## 10. Sample Records

### First 5 Records

| Date | Rented Bike Count | Hour | Temperature(°C) | Humidity(%) | Wind speed (m/s) | Visibility (10m) | Dew point temperature(°C) | Solar Radiation (MJ/m2) | Rainfall(mm) | Snowfall (cm) | Seasons | Holiday | Functioning Day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01/12/2017 | 254 | 0 | -5.2 | 37 | 2.2 | 2000 | -17.6 | 0.0 | 0.0 | 0.0 | Winter | No Holiday | Yes |
| 01/12/2017 | 204 | 1 | -5.5 | 38 | 0.8 | 2000 | -17.6 | 0.0 | 0.0 | 0.0 | Winter | No Holiday | Yes |
| 01/12/2017 | 173 | 2 | -6.0 | 39 | 1.0 | 2000 | -17.7 | 0.0 | 0.0 | 0.0 | Winter | No Holiday | Yes |
| 01/12/2017 | 107 | 3 | -6.2 | 40 | 0.9 | 2000 | -17.6 | 0.0 | 0.0 | 0.0 | Winter | No Holiday | Yes |
| 01/12/2017 | 78 | 4 | -6.0 | 36 | 2.3 | 2000 | -18.6 | 0.0 | 0.0 | 0.0 | Winter | No Holiday | Yes |

### Last 5 Records

| Date | Rented Bike Count | Hour | Temperature(°C) | Humidity(%) | Wind speed (m/s) | Visibility (10m) | Dew point temperature(°C) | Solar Radiation (MJ/m2) | Rainfall(mm) | Snowfall (cm) | Seasons | Holiday | Functioning Day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30/11/2018 | 1003 | 19 | 4.2 | 34 | 2.6 | 1894 | -10.3 | 0.0 | 0.0 | 0.0 | Autumn | No Holiday | Yes |
| 30/11/2018 | 764 | 20 | 3.4 | 37 | 2.3 | 2000 | -9.9 | 0.0 | 0.0 | 0.0 | Autumn | No Holiday | Yes |
| 30/11/2018 | 694 | 21 | 2.6 | 39 | 0.3 | 1968 | -9.9 | 0.0 | 0.0 | 0.0 | Autumn | No Holiday | Yes |
| 30/11/2018 | 712 | 22 | 2.1 | 41 | 1.0 | 1859 | -9.8 | 0.0 | 0.0 | 0.0 | Autumn | No Holiday | Yes |
| 30/11/2018 | 584 | 23 | 1.9 | 43 | 1.3 | 1909 | -9.3 | 0.0 | 0.0 | 0.0 | Autumn | No Holiday | Yes |

## 11. Issues Summary

✅ **No critical issues detected.** The dataset is clean and ready for Phase 2 processing.

## 12. Recommendations for Phase 2

1. **Parse the `Date` column** into a proper datetime type and extract temporal features (day of week, month, year, is_weekend).
2. **Encode categorical variables** (`Seasons`, `Holiday`, `Functioning Day`) using one-hot or ordinal encoding.
3. **Feature engineering**: Consider interaction features (e.g., temperature × humidity), cyclic encoding for `Hour`, and lag features if time-series structure is preserved.
4. **Handle zero-demand records**: Investigate the 295 records where `Rented Bike Count = 0`. Many correspond to non-functioning hours — decide whether to keep, filter, or flag them.
5. **Outlier analysis**: Review extreme values in `Rainfall(mm)` (max=35.0) and `Snowfall (cm)` (max=8.8) for potential impact on model training.
6. **Correlation analysis**: Compute pairwise correlations to identify multicollinearity and strong predictors before model selection.

---
*Report generated by `src/data_inspection.py` on 2026-10-22T20:08:29.657831*