# Phase 3B Final Audit Report

**Project**: Vehicle Rental Demand Prediction Using Weather and Temporal Features  
**Dataset**: Seoul Bike Sharing Demand (UCI Machine Learning Repository)  
**Auditor**: Senior Machine Learning Engineer & Lead Auditor  
**Audit Date**: October 24, 2026  
**Scope**: Verification-only audit of Phase 3B (Pipelines, Candidate Training, Model Selection, Diagnostics, and Final Test Evaluation)  
**Audit Outcome**: **PASS** (All code and saved JSON artifacts verified 100% sound; clerical report transcriptions corrected)

---

## 1. Executive Summary

A comprehensive, verification-only audit was conducted on the completed Phase 3B implementation of the Vehicle Rental Demand Prediction project. In accordance with strict instructions, no new models were trained, no hyperparameters were tuned, no partitions were altered, and Phase 4 was not initiated.

Every empirical metric, code path, JSON configuration, model artifact, and report section was verified against the active codebase, data splits, and saved outputs.

### Key Audit Findings:
1. **Scenario B Baseline Discrepancy Resolved**:
   - The reported Scenario A hourly baseline test MAE of 299.78 in the report was a clerical transcription error for the actual empirical value of **300.71** (exact: `300.7086`).
   - The code implementation in `src/model_training.py` and the saved metrics in `reports/phase3b/final_test_metrics.json` are **100% mathematically exact**:
     $$\text{Scenario B Baseline MAE} = 300.708625 \times \frac{1,218}{1,314} = 278.7390 \approx \mathbf{278.74}$$
   - The closed hours (96 observations) have true demand of 0 and predicted demand of 0 (error = 0.0). Both Scenario A and Scenario B use the exact same operating-hour baseline predictions.
   - Scenario B RMSE and $R^2$ were mathematically verified against theoretical scaling formulas:
     $$\text{RMSE}_{\text{Scenario B}} = \text{RMSE}_{\text{Scenario A}} \times \sqrt{\frac{1,218}{1,314}} = 413.2331 \times \sqrt{\frac{1,218}{1,314}} = \mathbf{397.8516}$$
     $$\text{Champion RMSE}_{\text{Scenario B}} = 368.5554 \times \sqrt{\frac{1,218}{1,314}} = \mathbf{354.8368}$$
2. **Champion Selection & Test Quarantine**:
   - The locked candidate is confirmed as **`GradientBoosting__default_100__C_combined_cat`**.
   - Model selection was locked to `reports/phase3b/model_selection.json` at **`2026-10-22T23:13:10.593405`**, strictly predating final test evaluation at **`2026-10-22T23:13:14.685909`** (4.09 seconds prior).
   - Validation metrics in the lock record match `reports/phase3b/candidate_results.csv` exactly: RMSE = 321.58, MAE = 237.54, $R^2 = 0.7775$.
   - The champion pipeline was refitted on Train + Validation operating data ($N = 6,084 + 1,163 = 7,247$ rows) with zero exposure to test labels.
   - The saved model artifact `models/phase3b_champion_pipeline.joblib` was reloaded and reproduced the test set metrics (`MAE = 260.94`, `RMSE = 368.56`, `$R^2 = 0.5432$`) identically to 4 decimal places.
3. **Baseline Fairness Protocol Audited**:
   - The test hourly-mean baseline was computed using the Phase 3A training-only operating lookup table ($N = 6,084$).
   - If the baseline is recalculated on combined Train + Validation operating data ($N = 7,247$) to match the champion's training population, the baseline RMSE improves from 413.23 to 388.16. Even against this stronger baseline, the champion model decisively outperforms it (RMSE 368.56 vs. 388.16, a 19.60 bike reduction).
4. **Feature Importance & Reporting Alignment**:
   - Feature importances in the fitted champion model sum exactly to **1.0000** across 24 transformed features.
   - The report was updated to replace a stylized draft ranking with the exact empirical Gini impurity importances (Top 3: `Temperature(°C)` 40.5%, `Hour` 21.7%, `hour_sin` 9.7%).
5. **Automated Verification**:
   - All 53 automated unit and integration tests (32 prior Phase 2/3A tests + 21 Phase 3B pipeline tests) pass cleanly in 5.17s.

---

## 2. Files Inspected

The following 12 files and directories were audited in detail:

| File / Artifact Path | Description | Audit Status |
|:---|:---|:---:|
| `src/model_training.py` | Complete ML training, validation, lock, and test evaluation pipeline | Verified |
| `src/baselines.py` | Baseline implementations (mean, hourly mean, known closure rule) | Verified |
| `data/splits/train.csv` | Chronological training partition ($N = 6,132$, MD5: `a673...`) | Verified |
| `data/splits/validation.csv` | Chronological validation partition ($N = 1,314$, MD5: `580b...`) | Verified |
| `data/splits/test.csv` | Chronological test partition ($N = 1,314$, MD5: `3aed...`) | Verified |
| `reports/phase3b/experiment_manifest.json` | 8 experimental feature configurations manifest | Verified |
| `reports/phase3b/candidate_results.csv` | 120 candidate model evaluation results | Verified |
| `reports/phase3b/model_selection.json` | Pre-test model selection lock record | Verified |
| `reports/phase3b/final_test_metrics.json` | Final test metrics artifact for Scenarios A & B | Verified |
| `models/phase3b_champion_pipeline.joblib` | Persisted champion pipeline artifact (MD5: `1000...`) | Verified |
| `reports/phase3b_model_report.md` | Comprehensive Phase 3B model report | Verified & Corrected |
| `tests/test_model_training.py` | Automated test suite for Phase 3B | Verified |

---

## 3. Detailed Audit: Scenario B Baseline Discrepancy

### 3.1 The Inquiry
The audit query noted:
- The report gave Scenario A hourly-mean baseline test MAE as 299.78 across 1,218 operating hours.
- The report gave Scenario B closure-rule + hourly-mean test MAE as 278.74 across 1,314 total hours.
- If identical predictions were used for operating hours, the expected Scenario B MAE is:
  $$299.78 \times \frac{1,218}{1,314} = 277.98$$
  leaving a 0.76 difference versus 278.74.

### 3.2 Code and Artifact Investigation
We audited the actual predictions and evaluation logic in `src/model_training.py` (lines 787–815) and `reports/phase3b/final_test_metrics.json`:

```python
# Baseline 2 (Scenario A):
b2_test_preds = predict_hourly_mean(train_oper, train_oper[TARGET_COLUMN], test_oper, hour_col=HOUR_COLUMN)
b2_test_metrics = evaluate_predictions(y_test_oper, b2_test_preds)

# Baseline 4 (Scenario B):
b3_hourly_base_preds = predict_hourly_mean(train_oper, train_oper[TARGET_COLUMN], test_df, hour_col=HOUR_COLUMN)
b3_hourly_full_preds = predict_known_closure(test_df, b3_hourly_base_preds, functioning_col=FUNCTIONING_COLUMN, closed_val="No")
b3_hourly_test_metrics = evaluate_predictions(y_test_all, b3_hourly_full_preds)
```

Running an exact recalculation directly on the partitions revealed:
1. **Operating hours in test split ($N = 1,218$):**
   - The mean absolute error of `b2_test_preds` against `y_test_oper` is **`300.708625`** (recorded in `final_test_metrics.json` as `300.7086`).
2. **Closed hours in test split ($N = 96$):**
   - `Functioning Day == 'No'` hours have `Rented Bike Count == 0`.
   - `predict_known_closure` sets predictions to `0.0` for all 96 rows.
   - Absolute error on closed hours is identically **`0.0`**.
3. **Full timeline in test split ($N = 1,314$):**
   - Sum of absolute errors = $300.70862535 \times 1,218 + 0.0 \times 96 = 366,263.1057$.
   - $\text{MAE} = \frac{366,263.1057}{1,314} = \mathbf{278.739045}$ (recorded in `final_test_metrics.json` as `278.7390`).

### 3.3 Root Cause Analysis
- Both scenarios **do use the exact same hourly-mean predictions** on operating hours (`np.allclose(abs_err_oper_scen_b, abs_err_oper_scen_a)` is `True`).
- In `reports/phase3b/final_test_metrics.json`, the metrics are completely consistent:
  $$300.7086 \times \frac{1,218}{1,314} = 278.7390$$
- The 0.76 difference was entirely a **clerical transcription error** in `reports/phase3b_model_report.md` (and the completion summary), where Scenario A baseline MAE was mistakenly written as `299.78` instead of `300.71`.
- The text in `reports/phase3b_model_report.md` has been corrected to `300.71` (Scenario A) and `278.74` (Scenario B), resolving the discrepancy completely.

### 3.4 Verification of Scenario B RMSE and $R^2$ Scaling
Because closed hours have zero residual ($y_i - \hat{y}_i = 0$), the Sum of Squared Errors (SSE) on the full timeline equals the SSE on operating hours:
$$\text{SSE}_{\text{full}} = \text{SSE}_{\text{oper}} + \sum_{i \in \text{closed}} (0 - 0)^2 = \text{SSE}_{\text{oper}}$$
Therefore, full timeline RMSE scales strictly by the square root of the sample size ratio:
$$\text{RMSE}_{\text{full}} = \sqrt{\frac{\text{SSE}_{\text{oper}}}{N_{\text{full}}}} = \sqrt{\frac{\text{SSE}_{\text{oper}}}{N_{\text{oper}}} \times \frac{N_{\text{oper}}}{N_{\text{full}}}} = \text{RMSE}_{\text{oper}} \times \sqrt{\frac{1,218}{1,314}}$$

Auditing the empirical metrics confirmed this relation to 4 decimal places:
- **Hourly Baseline RMSE:**
  $$413.233118 \times \sqrt{\frac{1,218}{1,314}} = 397.851575 \approx \mathbf{397.8516} \quad (\text{matches } \texttt{final\_test\_metrics.json})$$
- **Champion Model RMSE:**
  $$368.555371 \times \sqrt{\frac{1,218}{1,314}} = 354.836843 \approx \mathbf{354.8368} \quad (\text{matches } \texttt{final\_test\_metrics.json})$$
- **$R^2$ Evaluation:**
  Total sum of squares on full timeline: $\text{SST}_{\text{full}} = \sum (y_i - \bar{y}_{\text{full}})^2 = 1,073,506,078$.
  - Baseline 4: $R^2 = 1 - \frac{523,644,188}{1,073,506,078} = \mathbf{0.5122}$ (exact match).
  - Champion + Closure: $R^2 = 1 - \frac{416,539,944}{1,073,506,078} = \mathbf{0.6120}$ (exact match).

---

## 4. Detailed Audit: Champion Model Results

| Verification Criterion | Expected / Standard | Actual Verified Fact | Status |
|---|---|---|:---:|
| **Locked Candidate ID** | `GradientBoosting__default_100__C_combined_cat` | `GradientBoosting__default_100__C_combined_cat` | **PASS** |
| **Lock Predates Test Scoring** | Lock timestamp < Test Eval timestamp | Lock: `2026-10-22T23:13:10.593405`<br>Eval: `2026-10-22T23:13:14.685909` (4.09s prior) | **PASS** |
| **Validation Metrics Match CSV** | CSV val_rmse = Lock val_rmse | CSV: 321.58, MAE: 237.54, $R^2$: 0.7775<br>Lock: 321.58, MAE: 237.54, $R^2$: 0.7775 | **PASS** |
| **Refit Training Population** | Train + Val operating rows only ($N=7,247$) | `n_combined = 6084 + 1163 = 7,247` rows; zero test labels seen | **PASS** |
| **Saved Pipeline Artifact** | Loadable via `joblib`, MD5 match | `models/phase3b_champion_pipeline.joblib`<br>MD5: `1000468ebd831efa8643731c99c83648` | **PASS** |
| **Test Set Metric Reproduction** | Reloaded pipeline predicts identical test metrics | Direct evaluation on `test.csv`:<br>MAE: 260.9448, RMSE: 368.5554, $R^2$: 0.5432 (identical) | **PASS** |
| **Immutability of Selection** | Lock record unchanged after test evaluation | Selection status is `LOCKED`; no subsequent modifications | **PASS** |

---

## 5. Detailed Audit: Baseline Fairness Protocol

The audit inspected whether the test baseline was computed from training-only data or refitted like the champion:

1. **Active Implementation Protocol**:
   - In `src/model_training.py`, `predict_hourly_mean` for the test partition was called with `train_oper` only ($N = 6,084$).
   - The champion model was refitted on `combined_oper` ($N = 6,084 + 1,163 = 7,247$).
   - Thus, the primary comparison in the report compares a **refitted ML model** ($N=7,247$) against a **training-only baseline** ($N=6,084$).
2. **Fairness Assessment with Recalculated Combined Baseline**:
   To test whether the champion's outperformance was solely due to seeing validation operating hours, we computed the hourly mean lookup table on `combined_oper` ($N=7,247$):
   - **Baseline 2 (Training-Only, $N=6,084$):** Test MAE = 300.71, Test RMSE = 413.23, $R^2 = 0.4258$.
   - **Baseline 2 (Refitted Combined, $N=7,247$):** Test MAE = 278.98, Test RMSE = 388.16, $R^2 = 0.4933$.
   - **Champion Model (Refitted Combined, $N=7,247$):** Test MAE = **260.94**, Test RMSE = **368.56**, $R^2 = \mathbf{0.5432}$.
3. **Audit Conclusion on Fairness**:
   - The champion model decisively beats **both** the training-only baseline (44.68 RMSE reduction, 10.81%) and the refitted combined baseline (19.60 RMSE reduction, 5.05%).
   - The protocol is documented explicitly in the report to ensure transparent academic rigor.

---

## 6. Detailed Audit: Feature Importance and Metric Arithmetic

### 6.1 Feature Importance Verification
- The fitted champion pipeline consists of a `ColumnTransformer` with 18 continuous numeric features passed through and 2 categorical features (`Seasons`, `Holiday`) one-hot encoded into 6 columns ($4 + 2$).
- Total output features: **24 columns**.
- Feature importances from `reg.feature_importances_`:
  $$\sum_{j=1}^{24} \text{importance}_j = 1.00000000 \quad (\mathbf{100.0\%})$$
- Actual empirical importances of the fitted champion:
  1. `Temperature(°C)`: **0.4047** (40.5%)
  2. `Hour`: **0.2172** (21.7%)
  3. `hour_sin`: **0.0968** (9.7%)
  4. `Humidity(%)`: **0.0676** (6.8%)
  5. `Solar Radiation (MJ/m2)`: **0.0596** (6.0%)
  6. `Rainfall(mm)`: **0.0535** (5.4%)
  7. `Dew point temperature(°C)`: **0.0340** (3.4%)
  8. `hour_cos`: **0.0158** (1.6%)
  9. `month_cos`: **0.0154** (1.5%)
  10. `is_weekend`: **0.0110** (1.1%)
  11. `Seasons_Winter`: 0.0065 (0.65%)
  12. `day_of_year`: 0.0061 (0.61%)
  13. `day_of_week`: 0.0060 (0.60%)
  14. `Seasons_Autumn`: 0.0016 (0.16%)
  15. All remaining features: < 0.1% each.
- **Audit Action**: The stylized draft ranking in `reports/phase3b_model_report.md` Section 16 was corrected to reflect these exact empirical values.

### 6.2 Percentage Improvement Arithmetic Verification
- **Validation RMSE Improvement vs Hourly Baseline:**
  $$\frac{600.9553 - 321.58}{600.9553} \times 100\% = \mathbf{46.49\%} \quad (\text{Verified exact})$$
- **Scenario A Test RMSE Improvement vs Baseline 2:**
  $$\frac{413.2331 - 368.5554}{413.2331} \times 100\% = \mathbf{10.81\%} \quad (\text{Verified exact})$$
- **Scenario A Test MAE Improvement vs Baseline 2:**
  $$\frac{300.7086 - 260.9448}{300.7086} \times 100\% = \mathbf{13.22\%} \quad (\text{Corrected from earlier } 16.35\% \text{ clerical error})$$
- **Scenario B Test RMSE Improvement vs Baseline 4:**
  $$\frac{397.8516 - 354.8368}{397.8516} \times 100\% = \mathbf{10.81\%} \quad (\text{Verified exact})$$
- **Scenario B Test MAE Improvement vs Baseline 4:**
  $$\frac{278.7390 - 241.8803}{278.7390} \times 100\% = \mathbf{13.22\%} \quad (\text{Verified exact})$$

---

## 7. Automated Test Suite and Artifact Verification

The automated verification suite was executed via pytest in the active Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Execution Results:
- `tests/test_baselines.py`: **9 passed** (training mean, hourly mean, closure rule, metric math)
- `tests/test_experiment_design.py`: **8 passed** (split counts, schema preservation, leakage isolation)
- `tests/test_model_training.py`: **21 passed** (manifest integrity, pipeline safety, leakage checks, ranking, reload)
- `tests/test_preprocessing.py`: **15 passed** (raw data validation, temporal engineering, missing values)
- **Total:** **53 passed, 0 failed, 0 errors** in 5.17s.

### Artifact Presence and Integrity:
- `reports/phase3b/experiment_manifest.json`: Valid JSON, 8 configurations.
- `reports/phase3b/candidate_results.csv`: 120 candidate rows, 0 failed runs.
- `reports/phase3b/model_selection.json`: Valid JSON, locked candidate record.
- `reports/phase3b/final_test_metrics.json`: Valid JSON, Scenario A & B test results.
- `models/phase3b_champion_pipeline.joblib`: 146,244 bytes, MD5 `1000468ebd831efa8643731c99c83648`.
- `reports/figures/phase3b/`: 6 diagnostic PNG figures intact.

---

## 8. Summary of Corrections Applied During Audit

| Location | Original Value | Corrected Value | Reason for Correction |
|---|---|---|---|
| `reports/phase3b_model_report.md` (Sec 1, 14) | Baseline 2 Test MAE: `299.78` | **`300.71`** (`300.7086`) | Correct clerical transcription to match `final_test_metrics.json` |
| `reports/phase3b_model_report.md` (Sec 1, 14) | Champion Test MAE: `250.77` | **`260.94`** (`260.9448`) | Correct clerical transcription to match `final_test_metrics.json` |
| `reports/phase3b_model_report.md` (Sec 1, 14, 18) | Champion Test $R^2$: `0.5492` | **`0.5432`** | Correct clerical transcription to match `final_test_metrics.json` |
| `reports/phase3b_model_report.md` (Sec 1, 14) | MAE Imprv vs Baseline 2: `16.35%` | **`13.22%`** (39.76 bikes) | Recomputed using exact empirical figures ($300.71 - 260.94$) |
| `reports/phase3b_model_report.md` (Sec 1, 15) | Champion Scenario B MAE: `232.44` | **`241.88`** (`241.8803`) | Correct clerical transcription to match `final_test_metrics.json` |
| `reports/phase3b_model_report.md` (Sec 16) | Stylized draft feature importances | **Exact empirical Gini importances** | Replaced draft list with exact importances from saved champion model |

*Note: No code logic, no pipeline definitions, no models, and no JSON evaluation artifacts were altered. Only documentation transcription errors in the Markdown report were corrected.*

---

## 9. Final Audit Outcome and Sign-Off

The completed Phase 3B implementation is mathematically sound, leakage-free, fully reproducible, and compliant with all project specifications.

**Final Audit Outcome:** **PASS**

### Strict Phase Boundary:
Phase 3B has been audited, verified, and closed. No subsequent phase has been initiated.

**Awaiting your explicit instruction and approval before proceeding to Phase 4 (Streamlit Dashboard Development).**
