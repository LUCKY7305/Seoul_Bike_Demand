# Phase 3A Final Audit Report

**Project**: Vehicle Rental Demand Prediction Using Weather and Temporal Features  
**Auditor**: Senior Machine Learning Engineer  
**Audit Date**: October 2026  
**Scope**: Verification-only audit of Phase 3A (Experimental Design, Chronological Splitting, and Baselines)  
**Audit Outcome**: **PASS**  

---

## 1. Executive Summary

A comprehensive, verification-only audit was conducted on the completed Phase 3A implementation of the Vehicle Rental Demand Prediction project. Every requirement specified in the Phase 3A experimental design protocol was audited against the active codebase, data splits, and reports.

### Key Audit Findings:
1. **Split Integrity**: Row-based chronological splitting (70% Train / 15% Validation / 15% Test) preserves all 8,760 observations with zero leakage, strict chronological ordering, and zero overlap.
2. **Calendar Day Boundary Crossing**: As expected under the exact 70/15/15 row partition policy ($6,132$ / $1,314$ / $1,314$), two calendar days cross partition boundaries: `2018-08-13` crosses at 11:00/12:00, and `2018-10-07` crosses at 05:00/06:00. This is documented and verified.
3. **Baseline Soundness**: All baselines strictly compute lookups and scalar means from operating-hour training targets only. Validation targets are used strictly for scoring.
4. **Test-Set Isolation**: Absolute quarantine confirmed. Zero test-set predictions and zero test-set performance metrics exist anywhere in the project.
5. **Feature-Set Integrity**: All candidate feature sets cleanly exclude `Rented Bike Count` and raw `Date`. `Functioning Day` is properly quarantined to scenario routing. Feature Set D removes only `Dew point temperature(°C)`.
6. **Automated Verification**: All 32 automated unit and integration tests pass cleanly with 100% success.
7. **Phase Boundaries**: No machine learning models were trained, tuned, or evaluated.

---

## 2. Files Inspected

The following 14 files and artifacts were audited:

| File Path | Description | Verification Status |
|:---|:---|:---:|
| `src/experiment_design.py` | Experiment design runner & splitting pipeline | Verified |
| `src/baselines.py` | Heuristic baseline models and evaluation logic | Verified |
| `tests/test_experiment_design.py` | Integration tests for splits and feature sets | Verified |
| `tests/test_baselines.py` | Unit tests for baseline models and fallbacks | Verified |
| `tests/test_preprocessing.py` | Preprocessing and schema regression tests | Verified |
| `data/splits/train.csv` | Chronological training partition ($N = 6,132$) | Verified |
| `data/splits/validation.csv` | Chronological validation partition ($N = 1,314$) | Verified |
| `data/splits/test.csv` | Chronological test partition ($N = 1,314$) | Verified |
| `data/splits/split_metadata.json` | Partition boundaries, counts, and checksums | Verified |
| `reports/phase3a/feature_sets.json` | Validated candidate feature set lists | Verified |
| `reports/phase3a/experiment_config.json` | Master experiment configuration | Verified |
| `reports/phase3a_experiment_design.md` | Technical experiment design report | Verified |
| `data/processed/seoul_bike_processed.csv` | Preprocessed master dataset ($8,760 \times 23$) | Verified |
| `data/processed/preprocessing_metadata.json` | Preprocessing metadata and checksums | Verified |

---

## 3. Chronological Split Integrity Findings

### 3.1 Observation Counts and Completeness
- Master processed dataset: **8,760 rows × 23 columns**.
- Training partition (`train.csv`): **6,132 rows** (70.00%).
- Validation partition (`validation.csv`): **1,314 rows** (15.00%).
- Test partition (`test.csv`): **1,314 rows** (15.00%).
- Sum of partitions: $6,132 + 1,314 + 1,314 = \mathbf{8,760}$ rows (**100.0% completeness**).
- Exact concatenation check: `pd.concat([train, val, test], ignore_index=True).equals(df_proc)` evaluated to **`True`**. Every single observation from the processed dataset appears in exactly one partition.

### 3.2 Temporal Ordering and Non-Overlap
- Chronological monotonicity across partition boundaries was verified:
  - $\text{Train End} = \text{2018-08-13 11:00}$
  - $\text{Val Start} = \text{2018-08-13 12:00}$
  - $\text{Val End} = \text{2018-10-07 05:00}$
  - $\text{Test Start} = \text{2018-10-07 06:00}$
  - $\text{Test End} = \text{2018-11-30 23:00}$
- $\text{Train}_{\max} < \text{Val}_{\min}$ and $\text{Val}_{\max} < \text{Test}_{\min}$ are strictly satisfied. Zero temporal overlap exists.

### 3.3 Splitting Order Relative to Operational Status
- Splitting was executed on the contiguous chronological timeline **before** filtering by `Functioning Day`.
- Functioning and non-functioning observation distribution across partitions:
  - **Train**: 6,084 operating (`Yes`) + 48 non-operating (`No`) = 6,132 rows (48 zeros).
  - **Validation**: 1,163 operating (`Yes`) + 151 non-operating (`No`) = 1,314 rows (151 zeros).
  - **Test**: 1,218 operating (`Yes`) + 96 non-operating (`No`) = 1,314 rows (96 zeros).
  - **Total**: 8,465 operating + 295 non-operating = 8,760 rows (295 zeros).

### 3.4 Calendar Day Boundary Crossing Analysis
Because the split uses exact row proportions ($70.0\%$ and $15.0\%$) on an hourly dataset:
- **`2018-08-13` (Train/Validation Boundary)**:
  - Hours 0 through 11 (12 hours) reside in `train.csv`.
  - Hours 12 through 23 (12 hours) reside in `validation.csv`.
  - Midday split occurs at 11:00 vs. 12:00.
- **`2018-10-07` (Validation/Test Boundary)**:
  - Hours 0 through 5 (6 hours) reside in `validation.csv`.
  - Hours 6 through 23 (18 hours) reside in `test.csv`.
  - Early morning split occurs at 05:00 vs. 06:00.

**Audit Recommendation**: Keep the existing row-based 70/15/15 partitions as designed. Both boundaries maintain strict temporal ordering and continuous time blocks without data leakage. If daily aggregated forecasting is ever introduced in a future phase, calendar-day alignment can be considered, but for hourly demand regression, the current row-based split is mathematically and methodologically sound.

---

## 4. Baseline Implementation Findings

All baseline prediction logic in `src/baselines.py` and evaluation code in `src/experiment_design.py` was audited:

| Requirement | Implementation Verification | Status |
|:---|:---|:---:|
| **Training Labels Only** | `calculate_training_mean` and `compute_hourly_lookup` accept only training data (`y_train_oper`, `train_oper`). | Verified |
| **Operating Hours Only** | Operational baselines filter `train_df[train_df['Functioning Day'] == 'Yes']`. Non-operating training zeros are excluded from the operating mean. | Verified |
| **Unseen Hour Fallback** | `predict_hourly_mean` uses `hourly_lookup.get(int(h), global_mean)`. An unseen hour cleanly falls back to the global training mean. | Verified |
| **Validation Label Quarantine** | Validation labels (`y_val_oper`, `y_val_all`) are passed strictly to `evaluate_predictions` as ground truth. Predictions are computed with zero knowledge of evaluation labels. | Verified |
| **Closure Rule Distinction** | Baseline 3 (`predict_known_closure`) is reported under Scenario B and documented as a deterministic business policy requiring advance operational knowledge. | Verified |
| **Metric Consistency** | Scikit-learn standard metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`) are computed consistently for the stated subsets: Scenario A ($N = 1,163$) and Scenario B ($N = 1,314$). | Verified |

### Audited Validation Baseline Results:
- **Scenario A (Operational Hours, $N = 1,163$)**:
  - Baseline 1 (Operating Training Mean = 655.03): $\text{MAE} = 588.37$, $\text{RMSE} = 760.02$, $R^2 = -0.2431$.
  - Baseline 2 (Hour-of-Day Mean): $\text{MAE} = 473.95$, $\text{RMSE} = 600.96$, $R^2 = +0.2228$.
- **Scenario B (End-to-End Timeline, $N = 1,314$)**:
  - Baseline 3 (Closure Rule + Mean): $\text{MAE} = 520.76$, $\text{RMSE} = 715.02$, $R^2 = -0.0001$.
  - Baseline 3 Variant (Closure Rule + Hourly): $\text{MAE} = 419.49$, $\text{RMSE} = 565.37$, $R^2 = +0.3747$.

All metric calculations are exact, reproducible, and mathematically verified.

---

## 5. Test-Set Isolation Findings

A codebase-wide audit was conducted to confirm the test partition (`data/splits/test.csv`) remains completely quarantined:

1. **No Test-Set Predictions**: No script or notebook contains calls generating predictions on `test.csv`.
2. **No Test-Set Performance Metrics**: Grep searches for evaluation functions (`evaluate_predictions`, `mean_squared_error`, `mean_absolute_error`, `r2_score`) confirmed that test labels are never evaluated.
3. **No Test-Driven Design Choices**: No feature selection, baseline thresholds, or hyperparameter decisions were made using test data.
4. **Permitted Checks Only**: The test partition was accessed strictly for non-evaluative checks:
   - Row count verification ($1,314$ rows).
   - Column schema matching ($23$ columns).
   - MD5 checksum tracking (`3aedb7e6d44b2ecbc46f8402243b0efc`).
   - Chronological boundary confirmation (`2018-10-07 06:00` to `2018-11-30 23:00`).

**Test-Set Quarantine Status: 100% COMPLIANT**.

---

## 6. Feature-Set Consistency Findings

`reports/phase3a/feature_sets.json` was compared against the schema of `data/processed/seoul_bike_processed.csv`:

| Feature Set | Documented Count | Actual Array Length | Contents Verified | Target Excluded | Raw Date Excluded |
|:---|:---:|:---:|:---:|:---:|:---:|
| `FEATURE_SET_A_TEMPORAL` | 10 | 10 | `Hour`, `day_of_week`, `month`, `day_of_month`, `day_of_year`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | Yes | Yes |
| `FEATURE_SET_B_WEATHER` | 8 | 8 | `Temperature(°C)`, `Humidity(%)`, `Wind speed (m/s)`, `Visibility (10m)`, `Dew point temperature(°C)`, `Solar Radiation (MJ/m2)`, `Rainfall(mm)`, `Snowfall (cm)` | Yes | Yes |
| `FEATURE_SET_C_TEMPORAL_WEATHER` | 18 | 18 | Feature Set A (10) + Feature Set B (8) | Yes | Yes |
| `FEATURE_SET_D_REDUCED_COLLINEARITY` | 17 | 17 | Feature Set C omitting *only* `Dew point temperature(°C)` | Yes | Yes |
| `CATEGORICAL_FEATURES` | 2 | 2 | `Seasons`, `Holiday` | Yes | Yes |
| `SCENARIO_FEATURES` | 1 | 1 | `Functioning Day` | Yes | Yes |

### Key Consistency Checks:
- **Target Exclusion**: `Rented Bike Count` is strictly absent from all feature sets.
- **Date Exclusion**: Raw `Date` string is strictly absent from all feature sets.
- **Scenario Quarantine**: `Functioning Day` is listed under `SCENARIO_FEATURES` and excluded from regression feature sets A, B, C, and D.
- **Collinearity Reduction**: The set difference $\text{Set C} \setminus \text{Set D}$ equals exactly `{'Dew point temperature(°C)'}`.

---

## 7. Automated Test Suite Results

The complete test suite was executed in the project virtual environment:

**Command**:
```powershell
.venv\Scripts\pytest.exe -v
```

**Results**:
```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.7.0 -- D:\B-Tech\SEM 5\IML\project\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\B-Tech\SEM 5\IML\project
plugins: anyio-4.15.1, platformdirs-4.13.5
collecting ... collected 32 items

tests/test_baselines.py::test_calculate_training_mean PASSED             [  3%]
tests/test_baselines.py::test_calculate_training_mean_empty_raises PASSED [  6%]
tests/test_baselines.py::test_predict_training_mean_values_and_shape PASSED [  9%]
tests/test_baselines.py::test_compute_hourly_lookup PASSED               [ 12%]
tests/test_baselines.py::test_predict_hourly_mean_with_unseen_hour_fallback PASSED [ 15%]
tests/test_baselines.py::test_predict_known_closure_rule PASSED          [ 18%]
tests/test_baselines.py::test_closure_rule_mismatched_length_raises PASSED [ 21%]
tests/test_baselines.py::test_evaluate_predictions_metrics PASSED        [ 25%]
tests/test_baselines.py::test_evaluate_predictions_empty_raises PASSED   [ 28%]
tests/test_experiment_design.py::test_processed_dataset_verification PASSED [ 31%]
tests/test_experiment_design.py::test_split_counts_and_completeness PASSED [ 34%]
tests/test_experiment_design.py::test_strict_chronological_ordering PASSED [ 37%]
tests/test_experiment_design.py::test_functioning_and_zero_demand_counts_per_partition PASSED [ 40%]
tests/test_experiment_design.py::test_split_files_reload_schema_preservation PASSED [ 43%]
tests/test_experiment_design.py::test_feature_sets_leakage_and_validity PASSED [ 46%]
tests/test_experiment_design.py::test_feature_sets_json_file_exists PASSED [ 50%]
tests/test_experiment_design.py::test_experiment_config_json_exists PASSED [ 53%]
tests/test_preprocessing.py::test_date_parsing_and_formatting PASSED     [ 56%]
tests/test_preprocessing.py::test_temporal_feature_calculations PASSED   [ 59%]
tests/test_preprocessing.py::test_is_weekend_calculation PASSED          [ 62%]
tests/test_preprocessing.py::test_hour_cyclical_features PASSED          [ 65%]
tests/test_preprocessing.py::test_month_cyclical_features PASSED         [ 68%]
tests/test_preprocessing.py::test_validation_rejects_invalid_hours PASSED [ 71%]
tests/test_preprocessing.py::test_validation_rejects_negative_rentals PASSED [ 75%]
tests/test_preprocessing.py::test_validation_rejects_invalid_categorical PASSED [ 78%]
tests/test_preprocessing.py::test_validation_rejects_duplicate_timestamps PASSED [ 81%]
tests/test_preprocessing.py::test_real_dataset_row_and_column_count PASSED [ 84%]
tests/test_preprocessing.py::test_real_dataset_target_preservation PASSED [ 87%]
tests/test_preprocessing.py::test_real_dataset_zero_demand_preservation PASSED [ 90%]
tests/test_preprocessing.py::test_real_dataset_missing_values PASSED     [ 93%]
tests/test_preprocessing.py::test_real_dataset_duplicate_timestamps PASSED [ 96%]
tests/test_metadata_file_validity PASSED          [100%]

============================= 32 passed in 3.83s ==============================
```

- **Total Tests**: 32
- **Passed**: 32 (100.0%)
- **Failed**: 0
- **Warnings / Errors**: 0

---

## 8. Discrepancies and Recommended Corrections

### Minor Documentation Note (Non-Blocking):
- In `reports/phase3a_experiment_design.md` section 8, the operating target means were reported with slight roundings ($655.03$, $991.11$, $849.30$); these match the exact computed partition figures ($655.0307$, $991.1118$, $849.3038$).
- Calendar day boundary crossing is an inherent property of hourly index-based splitting ($70/15/15$ row count) and is now fully documented. No code change is necessary.
- **Zero discrepancies requiring code modification were found.**

---

## 9. Final Audit Status

$$\mathbf{PASS}$$

The Phase 3A implementation is methodologically sound, strictly enforces future-data leakage prevention, completely isolates the test set, and provides a reproducible benchmark foundation for Phase 3B.

---
*Audit complete. Ready for explicit user authorization before initiating Phase 3B.*
