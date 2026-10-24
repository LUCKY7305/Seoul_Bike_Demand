# PHASE 3B REPORT: MACHINE LEARNING PIPELINES, MODEL TRAINING, VALIDATION, MODEL SELECTION, AND FINAL TEST EVALUATION

**Project:** Vehicle Rental Demand Prediction Using Weather and Temporal Features  
**Dataset:** Seoul Bike Sharing Demand (UCI Machine Learning Repository)  
**Phase:** Phase 3B  
**Author / Team:** Senior ML Engineer & IML Project Team  
**Evaluation Date:** October 22, 2026  
**Execution Environment:** Python 3.13.7 (MSC v.1944 64-bit AMD64), Windows OS  
**Core Dependencies:** `scikit-learn` 1.9.1, `pandas` 3.1.0, `numpy` 2.5.4, `xgboost` 3.4.1, `matplotlib` 3.11.2, `joblib` 1.6.0  

---

## 1. Executive Summary

Phase 3B of the Vehicle Rental Demand Prediction project establishes an end-to-end, leakage-safe machine learning regression pipeline, conducts candidate model training across 8 structured feature configurations and 6 model families (120 candidate models in total), performs model ranking and selection strictly on chronological validation data, locks the champion pipeline configuration, refits the locked model on training plus validation data ($N = 7,247$ operating hours), and evaluates final generalization performance on the untouched chronological test partition under both Scenario A (operating hours) and Scenario B (full timeline with known closure rule).

### Key Empirical Findings:
1. **Model Selection on Validation Partition ($N = 1,163$ operating hours):**
   - **Champion Model:** `GradientBoostingRegressor` (100 estimators, learning rate = 0.1, max depth = 3, random state = 42) trained on **Feature Set C + Categoricals** (`C_combined_cat`: 18 continuous temporal/weather features + 2 one-hot encoded categoricals `Seasons` and `Holiday`).
   - **Validation Performance:** **RMSE = 321.58**, **MAE = 237.54**, **$R^2 = 0.7775$**.
   - **Improvement over Phase 3A Baselines:** Achieves a **46.49% RMSE reduction** (44.51% absolute RMSE reduction from 600.96 to 321.58) and **49.88% MAE reduction** (from 473.95 to 237.54) over the Phase 3A Hour-of-Day Mean baseline.
2. **Model Selection Lock:**
   - The selection record was formally locked to `reports/phase3b/model_selection.json` on `2026-10-22T23:13:10.593405` prior to inspecting test partition labels (`2026-10-22T23:13:14.685909`).
3. **Generalization on Untouched Test Partition ($N = 1,218$ operating hours, Scenario A):**
   - **Champion Model (Refitted on Train+Val, $N=7,247$):** **RMSE = 368.56**, **MAE = 260.94**, **$R^2 = 0.5432$**.
   - **Hour-of-Day Baseline on Test:** RMSE = 413.23, MAE = 300.71, $R^2 = 0.4258$.
   - **Test Outperformance:** The champion model outperforms the hourly baseline by **10.81% in RMSE** (368.56 vs. 413.23) and **13.22% in MAE** (260.94 vs. 300.71, 39.76 bike reduction), demonstrating robust out-of-sample generalization across seasonal distribution shifts.
4. **Scenario B Operational Full-Timeline Evaluation ($N = 1,314$ hours):**
   - Combining the deterministic closure rule ($\hat{y} = 0$ when `Functioning Day == 'No'`) with the champion model yields **RMSE = 354.84**, **MAE = 241.88**, **$R^2 = 0.6120$**, outperforming the Phase 3A Closure + Hourly Mean baseline (RMSE = 397.85, MAE = 278.74, $R^2 = 0.5122$) by **10.81% in RMSE** and **13.22% in MAE**.
5. **Quality and Regression Testing:**
   - All 53 automated unit and integration tests (32 prior Phase 2/3A tests + 21 Phase 3B pipeline tests) pass in 8.60s without errors.

---

## 2. Project Objective

The core objective of Phase 3B is to construct, train, compare, and validate machine learning regression pipelines predicting hourly vehicle rental demand (`Rented Bike Count`) while addressing 9 fundamental research questions:

1. **Temporal Features Alone:** How well can diurnal, weekly, and annual cyclical features predict demand?
2. **Weather Features Alone:** How well can continuous meteorological measurements predict demand?
3. **Combined Features:** Does combining temporal and weather features yield superior predictive power?
4. **Collinearity Impact:** Does removing `Dew point temperature(°C)` (strongly correlated with `Temperature(°C)` at $r = 0.9128$) stabilize or improve regression models?
5. **Categorical Features:** Do calendar variables (`Seasons` and `Holiday`) enhance model accuracy?
6. **Best Model Family:** Which regression architecture (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost) achieves minimal error on chronological validation data?
7. **Baseline Outperformance:** Does the best model decisively beat the approved Phase 3A Hour-of-Day Mean baseline on identical operating-hour populations?
8. **Test Generalization:** How well does the locked champion model generalize to an untouched future chronological test partition?
9. **Empirical Limitations:** What real-world factors (seasonal drift, observed vs. forecast weather, single-year holdout) qualify the interpretation of these results?

---

## 3. Dataset and Split Verification

Phase 3B utilizes the approved, immutable chronological splits generated in Phase 3A. The partitions were verified against their documented schema, record counts, and cryptographic checksums before model training.

### Split Verification Summary:
| Partition | File Path | Total Rows | Operating Rows (`Yes`) | Non-Operating Rows (`No`) | Start Timestamp | End Timestamp | MD5 Checksum |
|---|---|---:|---:|---:|---|---|---|
| **Training** | `data/splits/train.csv` | 6,132 | 6,084 | 48 | 2017-12-01 00:00 | 2018-08-13 11:00 | `a673224132856d3f494aabcffd844ebc` |
| **Validation** | `data/splits/validation.csv` | 1,314 | 1,163 | 151 | 2018-08-13 12:00 | 2018-10-07 05:00 | `580be235f097f84ed39d0563f6d8f13f` |
| **Test** | `data/splits/test.csv` | 1,314 | 1,218 | 96 | 2018-10-07 06:00 | 2018-11-30 23:00 | `3aedb7e6d44b2ecbc46f8402243b0efc` |
| **Total** | — | **8,760** | **8,465** | **295** | 2017-12-01 00:00 | 2018-11-30 23:00 | — |

- **Integrity Checks Passed:**
  - Zero missing values across all columns.
  - Zero duplicate timestamps.
  - Partitions are contiguous, non-overlapping, and preserve chronological order.
  - Splitting was performed strictly prior to any filtering.

---

## 4. Evaluation Scenarios

To prevent population mismatch and misleading comparative claims, Phase 3B enforces two strictly delineated evaluation scenarios:

### 4.1 Scenario A: Operating-Hour Demand Prediction (Primary ML Benchmark)
- **Target Population:** Observations where facility operating status is open (`Functioning Day == 'Yes'`).
- **Population Counts:**
  - Training: 6,084 rows
  - Validation: 1,163 rows
  - Test: 1,218 rows
- **Rationale:** The regression task models actual demand dynamics when stations are operational. Closed hours represent administrative shutdowns (guaranteed zero rentals), which distort continuous regression metrics if mixed directly into model fitting.
- **Leakage Rule:** `Functioning Day` is strictly excluded from the regression feature matrix.

### 4.2 Scenario B: Full-Timeline Demand Prediction (Operational System)
- **Target Population:** All 8,760 chronological hours (including 295 facility closure hours).
- **Population Counts:**
  - Validation: 1,314 rows
  - Test: 1,314 rows
- **Operational Rule:**
  $$\hat{y} = \begin{cases} 0, & \text{if known operating status is 'No'} \\ f(X), & \text{if known operating status is 'Yes'} \end{cases}$$
- **Operational Assumption:** Operating status must be known or scheduled in advance at inference time. Reported separately from Scenario A.

---

## 5. Feature Configurations

From the Phase 3A base definitions (`reports/phase3a/feature_sets.json`), Phase 3B defined and saved an explicit 8-configuration experiment manifest in `reports/phase3b/experiment_manifest.json`:

```
Base Sets:
├── Set A (Temporal Only): 10 features [Hour, day_of_week, month, day_of_month, day_of_year, is_weekend, hour_sin, hour_cos, month_sin, month_cos]
├── Set B (Weather Only): 8 features [Temperature, Humidity, Wind speed, Visibility, Dew point temp, Solar Radiation, Rainfall, Snowfall]
├── Set C (Combined): 18 features [All Set A + All Set B]
└── Set D (Reduced Collinearity): 17 features [All Set C omitting Dew point temperature]

Categorical Expansion (+ Seasons, Holiday):
├── Set A + Categoricals (A_temporal_cat): 10 numeric + 2 categorical = 12 input features
├── Set B + Categoricals (B_weather_cat): 8 numeric + 2 categorical = 10 input features
├── Set C + Categoricals (C_combined_cat): 18 numeric + 2 categorical = 20 input features
└── Set D + Categoricals (D_reduced_cat): 17 numeric + 2 categorical = 19 input features
```

### Feature Leakage Audit:
- `Rented Bike Count` (target), `Date` (raw timestamp string), and `Functioning Day` (scenario filter) are strictly absent from all feature configurations.

---

## 6. Preprocessing and Leakage Prevention

All data preprocessing is embedded inside scikit-learn `Pipeline` and `ColumnTransformer` instances to guarantee zero information leakage across data splits:

1. **Continuous Numeric Features:**
   - **Linear & Regularized Models (LinearRegression, Ridge):** Standardized using `StandardScaler()`. Mean and variance parameters ($\mu, \sigma$) are fitted exclusively on the operating-hour training partition and applied blindly to validation/test partitions.
   - **Tree-Based Models (DecisionTree, RandomForest, GradientBoosting, XGBoost):** Processed via `"passthrough"`, preserving natural feature thresholds without scaling artifacts.
2. **Categorical Features (`Seasons`, `Holiday`):**
   - Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - Categories are discovered strictly from training observations. Unseen categories at inference time are mapped to all-zero indicator vectors without throwing runtime exceptions.
3. **Target Handling:**
   - The target `Rented Bike Count` is retained in original rental count units. No uninvertible logarithmic transformations or target scalers were applied, ensuring all reported MAE and RMSE metrics represent direct physical bike counts.

---

## 7. Candidate Models and Hyperparameter Configurations

Phase 3B systematically evaluated 15 distinct estimator configurations across all 8 feature sets, yielding 120 candidate models:

1. **Linear Regression:** Standard OLS (`fit_intercept=True`).
2. **Ridge Regression:** L2-regularized linear model with $\alpha \in \{0.1, 1.0, 10.0\}$.
3. **Decision Tree Regressor:**
   - `default`: `max_depth=None`, `min_samples_leaf=1`
   - `depth_10`: `max_depth=10`, `min_samples_leaf=5`
   - `depth_5`: `max_depth=5`, `min_samples_leaf=10`
4. **Random Forest Regressor:**
   - `default_100`: `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `n_jobs=-1`
   - `depth_20`: `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`, `n_jobs=-1`
   - `depth_10`: `n_estimators=100`, `max_depth=10`, `min_samples_leaf=5`, `n_jobs=-1`
5. **Gradient Boosting Regressor (`scikit-learn`):**
   - `default_100`: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`
   - `depth_5`: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`
   - `slow_lr`: `n_estimators=150`, `learning_rate=0.05`, `max_depth=4`
6. **XGBoost Regressor (`xgboost`):**
   - `default_100`: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`
   - `tuned_150`: `n_estimators=150`, `learning_rate=0.08`, `max_depth=6`

*All stochastic algorithms were seeded with `random_state = 42`.*

---

## 8. Experimental Methodology

The experiment execution in `src/model_training.py` enforced a strict staged protocol:
1. Load split partitions and verify integrity.
2. Build and persist `experiment_manifest.json`.
3. Filter training and validation sets to operating hours ($N_{\text{train}} = 6,084, N_{\text{val}} = 1,163$).
4. Fit all 120 candidate pipelines on training operating hours only.
5. Generate predictions on validation operating hours and calculate MAE, RMSE, and $R^2$.
6. Rank candidates by validation RMSE.
7. Lock the best candidate configuration into `model_selection.json` with timestamp and checksum verification.
8. Refit the locked champion pipeline on combined Train + Validation operating data ($N = 7,247$).
9. Evaluate the refitted champion on untouched Test operating data ($N = 1,218$, Scenario A) and full timeline ($N = 1,314$, Scenario B).
10. Persist model artifact to `models/phase3b_champion_pipeline.joblib`.

---

## 9. Phase 3A Baseline Results

The established Phase 3A baselines on the validation partition ($N = 1,163$ operating hours, $N = 1,314$ full timeline) serve as the benchmark:

| Baseline Description | Evaluation Population | Val MAE | Val RMSE | Val $R^2$ | Purpose in Phase 3B |
|---|---|---:|---:|---:|---|
| **Baseline 1: Training Mean** ($\bar{y}_{\text{train}} = 655.03$) | Operating Hours ($N=1,163$) | 588.37 | 760.02 | -0.2431 | Naive central tendency reference |
| **Baseline 2: Hour-of-Day Mean** | Operating Hours ($N=1,163$) | 473.95 | 600.96 | +0.2228 | **Primary benchmark for candidate models** |
| **Baseline 3: Closure Rule + Mean** | Full Timeline ($N=1,314$) | 520.76 | 715.02 | -0.0001 | Scenario B naive baseline |
| **Baseline 4: Closure Rule + Hourly Mean** | Full Timeline ($N=1,314$) | 419.49 | 565.37 | +0.3747 | **Scenario B operational benchmark** |

---

## 10. Validation Comparison Table for Successful Candidates

All 120 candidate runs executed successfully (0 errors). The complete results are recorded in `reports/phase3b/candidate_results.csv`. Below is the summary of the top 15 performing candidates and family champions:

### Top 15 Candidates Overall (Ranked by Validation RMSE):
| Rank | Candidate ID | Model Family | Feature Set | Val RMSE | Val MAE | Val $R^2$ | RMSE Imprv. vs Hourly Base | Beats Baseline? |
|---|---|---|---|---:|---:|---:|---:|:---:|
| **1** | `GradientBoosting__default_100__C_combined_cat` | GradientBoosting | `C_combined_cat` | **321.58** | **237.54** | **0.7775** | **+46.49%** | **YES (Champion)** |
| 2 | `GradientBoosting__default_100__C_combined` | GradientBoosting | `C_combined` | 321.64 | 237.42 | 0.7774 | +46.48% | YES |
| 3 | `RandomForest__depth_10__C_combined_cat` | RandomForest | `C_combined_cat` | 333.46 | 236.39 | 0.7607 | +44.51% | YES |
| 4 | `GradientBoosting__default_100__D_reduced` | GradientBoosting | `D_reduced` | 333.61 | 244.32 | 0.7605 | +44.49% | YES |
| 5 | `GradientBoosting__depth_5__C_combined` | GradientBoosting | `C_combined` | 333.75 | 238.46 | 0.7603 | +44.46% | YES |
| 6 | `RandomForest__depth_10__C_combined` | RandomForest | `C_combined` | 333.82 | 236.90 | 0.7602 | +44.45% | YES |
| 7 | `GradientBoosting__depth_5__C_combined_cat` | GradientBoosting | `C_combined_cat` | 335.81 | 243.47 | 0.7573 | +44.12% | YES |
| 8 | `XGBoost__default_100__C_combined_cat` | XGBoost | `C_combined_cat` | 338.35 | 246.47 | 0.7536 | +43.70% | YES |
| 9 | `RandomForest__depth_20__C_combined_cat` | RandomForest | `C_combined_cat` | 338.67 | 242.78 | 0.7532 | +43.64% | YES |
| 10 | `GradientBoosting__slow_lr__C_combined_cat` | GradientBoosting | `C_combined_cat` | 340.81 | 253.66 | 0.7500 | +43.29% | YES |
| 11 | `RandomForest__default_100__C_combined_cat` | RandomForest | `C_combined_cat` | 340.85 | 245.02 | 0.7499 | +43.28% | YES |
| 12 | `RandomForest__depth_20__C_combined` | RandomForest | `C_combined` | 340.91 | 244.20 | 0.7498 | +43.27% | YES |
| 13 | `GradientBoosting__depth_5__D_reduced` | GradientBoosting | `D_reduced` | 341.59 | 246.12 | 0.7488 | +43.16% | YES |
| 14 | `GradientBoosting__default_100__D_reduced_cat` | GradientBoosting | `D_reduced_cat` | 342.08 | 247.95 | 0.7481 | +43.08% | YES |
| 15 | `RandomForest__default_100__C_combined` | RandomForest | `C_combined` | 342.33 | 246.07 | 0.7478 | +43.04% | YES |

### Best Candidate Per Model Family:
| Model Family | Best Tag | Best Feature Config | Val RMSE | Val MAE | Val $R^2$ | Pct RMSE Imprv. vs Hourly Base |
|---|---|---|---:|---:|---:|---:|
| **Gradient Boosting** | `default_100` | `C_combined_cat` | **321.58** | **237.54** | **0.7775** | **+46.49%** |
| **Random Forest** | `depth_10` | `C_combined_cat` | 333.46 | 236.39 | 0.7607 | +44.51% |
| **XGBoost** | `default_100` | `C_combined_cat` | 338.35 | 246.47 | 0.7536 | +43.70% |
| **Decision Tree** | `depth_10` | `C_combined` | 366.74 | 258.00 | 0.7106 | +38.97% |
| **Linear Regression** | `default` | `D_reduced` | 567.84 | 435.80 | 0.3061 | +5.51% |
| **Ridge Regression** | `alpha_10.0` | `D_reduced` | 568.07 | 435.98 | 0.3055 | +5.47% |

---

## 11. Model Ranking and Selection Rationale

### Selection Rule:
In strict compliance with the Phase 3B specification, the model selection criterion was predefined as:
$$\text{Primary Metric: Minimum Validation RMSE}$$
$$\text{Secondary Metric: Minimum Validation MAE}$$
$$\text{Constraint: Evaluated exclusively on operating hours of the validation partition}$$

### Rationale:
1. `GradientBoosting__default_100__C_combined_cat` achieved the lowest validation RMSE (321.58) and an exceptional MAE of 237.54.
2. Its validation $R^2$ of 0.7775 explains nearly 78% of the rental variance during the validation window, representing a dramatic improvement over the hour-of-day mean baseline ($R^2 = 0.2228$).
3. The model exhibits strong structural regularisation (shallow depth $d=3$, 100 trees), protecting it from high-variance overfitting commonly observed in unconstrained trees (`DecisionTree__default` reached an RMSE of 542.44).

---

## 12. Model-Selection Lock and Selected Configuration

Before reading or evaluating the test partition, the selection decision was sealed in `reports/phase3b/model_selection.json`.

### Lock Metadata:
- **Lock Timestamp:** `2026-10-22T23:13:10.593405`
- **Candidate ID:** `GradientBoosting__default_100__C_combined_cat`
- **Model Architecture:** `sklearn.ensemble.GradientBoostingRegressor`
- **Hyperparameters:**
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 3
  - `random_state`: 42
- **Feature Set:** `C_combined_cat` (20 input features: 18 continuous numeric + 2 categoricals)
- **Numeric Pipeline:** Passthrough to preserve natural tree splits.
- **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

---

## 13. Validation Diagnostic Plots

Six high-resolution diagnostic visualisations were generated and saved in `reports/figures/phase3b/`:

1. **Actual vs. Predicted Demand (`01_actual_vs_predicted_val.png`):**
   - Displays validation actual vs. predicted bike rentals with the ideal 45-degree identity line.
   - Predictions track actual demand closely across the entire 0–3,500 rental count spectrum, with slight compression at peak values (> 3,000).
2. **Residuals vs. Predicted Demand (`02_residuals_vs_predicted_val.png`):**
   - Displays residuals ($y - \hat{y}$) across the predicted range.
   - Shows homoscedastic behavior centered tightly along the zero-residual line across low and mid demand levels, with modest heteroscedastic spread at high demand.
3. **Residual Distribution (`03_residual_distribution_val.png`):**
   - Histogram and KDE showing residuals centered near zero ($\mu \approx -2.1$) with near-Gaussian symmetry, confirming absence of systematic directional bias.
4. **Chronological Demand Sample (`04_chronological_sample_val.png`):**
   - Traces actual vs. predicted demand over a continuous two-week validation window (336 hours).
   - Highlights the model's ability to capture morning and evening commute peaks, weekend diurnal shifts, and midday weather-induced dips.
5. **Model Family Comparison (`05_model_comparison_val_rmse.png`):**
   - Bar chart illustrating validation RMSE across model families and Phase 3A baselines. Visually demonstrates the progression from naive mean (760.02) to hourly baseline (600.96), linear models (567.84), decision trees (366.74), random forests (333.46), and gradient boosting (321.58).
6. **Feature Importance (`06_feature_importance_val.png`):**
   - Displays the top 15 Gini/impurity-based feature importances from the fitted gradient boosting model.

---

## 14. Final Test Results for the Locked Champion and Matching Baselines

Following the selection lock, the champion pipeline was refitted on the combined training and validation operating records ($N_{\text{train+val}} = 6,084 + 1,163 = 7,247$ rows) and evaluated once on the untouched chronological test partition ($N = 1,218$ operating hours).

### Scenario A Evaluation Results:
| Model / Pipeline | Test MAE | Test RMSE | Test $R^2$ | RMSE Improvement vs Baseline 2 |
|---|---:|---:|---:|---:|
| **Baseline 1: Operating Training Mean** ($\bar{y} = 655.03$) | 447.49 | 578.89 | -0.1269 | — |
| **Baseline 2: Phase 3A Hour-of-Day Mean** | 300.71 | 413.23 | +0.4258 | Reference Baseline |
| **Refitted Champion: Gradient Boosting (`C_combined_cat`)** | **260.94** | **368.56** | **+0.5432** | **+10.81% (44.68 RMSE reduction)** |

### Generalization Performance Analysis:
- The locked champion pipeline achieves **RMSE = 368.56** and **MAE = 260.94** on the untouched test partition.
- It outperforms the hour-of-day mean baseline by **44.68 bikes in RMSE (10.81% improvement)** and **39.76 bikes in MAE (13.22% improvement)**.
- The out-of-sample $R^2$ of **0.5432** demonstrates that over 54% of the test partition's demand variance is captured by the model, even in the presence of seasonal transition into late autumn.

---

## 15. Scenario B Full-Timeline Results

Scenario B evaluates the operational policy over all 1,314 test hours (including 96 facility shutdown hours).

### Operational Evaluation Table:
| Policy / Model | Full Test MAE | Full Test RMSE | Full Test $R^2$ | RMSE Improvement vs Baseline 4 |
|---|---:|---:|---:|---:|
| **Baseline 3: Closure Rule + Mean** | 414.79 | 557.34 | +0.0427 | — |
| **Baseline 4: Closure Rule + Hourly Mean** | 278.74 | 397.85 | +0.5122 | Reference Baseline |
| **Scenario B Champion: Closure Rule + Refitted Model** | **241.88** | **354.84** | **+0.6120** | **+10.81% (43.01 RMSE reduction)** |

### Operational Takeaway:
When non-operating hours are deterministically predicted as zero, the overall full-timeline RMSE drops to **354.84** and $R^2$ increases to **0.6120**. This validates the dual-scenario design: administrative closures are perfectly resolved by business logic, allowing the ML model to specialize exclusively on continuous operating demand. Note that the relationship between Scenario A and Scenario B metrics reflects the exact scaling of zero-error closure predictions ($N_{\text{oper}}=1,218$ out of $N_{\text{total}}=1,314$):
$$\text{MAE}_{\text{Scenario B}} = \text{MAE}_{\text{Scenario A}} \times \frac{1218}{1314} = 260.9448 \times \frac{1218}{1314} = 241.8803$$
$$\text{Baseline 4 MAE} = \text{Baseline 2 MAE} \times \frac{1218}{1314} = 300.7086 \times \frac{1218}{1314} = 278.7390$$

---

## 16. Feature Importance and Model Interpretation

Feature importances were extracted from the fitted gradient boosting regressor (`C_combined_cat`):

### Top 10 Most Important Features (Gini / Impurity Reduction):
1. **`Temperature(°C)` (0.4047 / 40.5%):** Primary driver across the ensemble; pleasant ambient temperatures strongly encourage cycling, while freezing or extreme heat dampens demand.
2. **`Hour` (0.2172 / 21.7%):** Dominates diurnal decision splits, encoding the distinct commuting peaks (8:00 AM and 6:00 PM).
3. **`hour_sin` (0.0968 / 9.7%):** Cyclical continuous representation of the diurnal cycle, smoothly bridging evening into morning hours.
4. **`Humidity(%)` (0.0676 / 6.8%):** High relative humidity reduces outdoor cycling comfort.
5. **`Solar Radiation (MJ/m2)` (0.0596 / 6.0%):** Serves as an effective proxy for daylight and clear sky conditions.
6. **`Rainfall(mm)` (0.0535 / 5.4%):** Immediate deterrent; precipitous drops in demand occur even with light precipitation.
7. **`Dew point temperature(°C)` (0.0340 / 3.4%):** Encodes atmospheric moisture and perceived warmth (heat index).
8. **`hour_cos` (0.0158 / 1.6%):** Complementary cyclical component of the diurnal schedule.
9. **`month_cos` (0.0154 / 1.5%):** Cyclical annual progression encoding seasonal transitions.
10. **`is_weekend` (0.0110 / 1.1%):** Distinguishes weekend recreational demand profiles from weekday commuter peaks.

*The 24 transformed feature importances sum exactly to 1.0000 (100.0%). Categorical seasonal dummy variables (`Seasons`, `Holiday`) accounted for smaller direct impurity gains (`Seasons_Winter`: 0.0065, `Seasons_Autumn`: 0.0016, `Seasons_Spring`: 0.0007, `Holiday`: 0.0010) because cyclic temporal features (`month_cos`, `day_of_year`) and temperature already encode the broader seasonal state.*

---

## 17. Temperature / Dew Point Collinearity Analysis

In Phase 1 and 2, a Pearson correlation of $r = 0.9128$ was identified between `Temperature(°C)` and `Dew point temperature(°C)`. Phase 3B explicitly compared Feature Set C (with Dew Point) against Feature Set D (without Dew Point):

### Comparative Results across Models:
| Model Family | Feature Set C (with Dew Point) Val RMSE | Feature Set D (omitting Dew Point) Val RMSE | Collinearity Impact on Accuracy |
|---|---:|---:|---|
| **Linear Regression (Numeric Only)** | 568.43 | **567.84** | Omitting Dew Point **improves** linear RMSE by 0.59 and stabilizes coefficients |
| **Ridge Regression ($\alpha=10.0$)** | 568.52 | **568.07** | Omitting Dew Point slightly improves regularized linear RMSE |
| **Gradient Boosting (`default_100`)** | **321.64** | 333.61 | Retaining Dew Point **improves** tree ensemble RMSE by 11.97 |
| **Random Forest (`depth_10`)** | **333.82** | 337.89 | Retaining Dew Point improves tree ensemble RMSE by 4.07 |

### Scientific Conclusion:
- For **linear models**, high collinearity inflates coefficient variance and causes variance inflation; removing `Dew point temperature(°C)` stabilizes the model and slightly improves test error.
- For **tree-based ensemble models**, decision trees split recursively along orthogonal axes. Correlated features provide non-linear interaction subtleties (e.g. temperature combined with moisture yields heat index/perceived comfort) without causing numerical instability. Gradient boosting effectively leveraged both features.

---

## 18. Seasonal Drift and Generalization Analysis

The Seoul Bike Sharing dataset spans exactly one calendar year (Dec 1, 2017 to Nov 30, 2018). Under chronological splitting, each partition covers distinct calendar seasons:
- **Training Set (Dec 1, 2017 – Aug 13, 2018):** Winter, Spring, Early Summer. Mean operating demand = 655.03.
- **Validation Set (Aug 13, 2018 – Oct 7, 2018):** Peak Late Summer and Early Autumn. Mean operating demand = 991.11 (+51.3% increase over training).
- **Test Set (Oct 7, 2018 – Nov 30, 2018):** Mid to Late Autumn (cooling weather, shorter days). Mean operating demand = 849.30 (+29.7% increase over training).

### Impact on Metrics:
- On the validation set, peak cycling weather lifted the baseline RMSE to 600.96. The champion model adapted well, reducing RMSE to 321.58 ($R^2 = 0.7775$).
- On the test set, temperature dropped rapidly as winter approached. Baseline RMSE was 413.23. The champion model achieved RMSE = 368.56 ($R^2 = 0.5432$).
- The decrease in $R^2$ from validation (0.7775) to test (0.5432) reflects legitimate temporal concept drift: late autumn cycling behavior undergoes transition that cannot be fully anticipated by training on earlier months without multi-year historical depth.

---

## 19. Limitations

1. **Single-Year Dataset:** The dataset spans exactly 365 days. True inter-annual seasonality (comparing October 2018 with October 2017) cannot be learned without multi-year records.
2. **Observed vs. Forecast Weather:** The dataset provides historical observed weather. In actual production deployment, bike demand must be predicted using imperfect weather forecasts, which introduces forecast error variance into input features.
3. **Unobserved Demand Bottlenecks:** The target variable represents actual rentals completed, which is bounded by bike dock availability. On peak days, true demand may exceed available bike inventory (censored demand).
4. **Predictive vs. Causal Interpretation:** Feature importances reflect statistical predictive association, not causal levers. For instance, solar radiation is predictive of demand because it correlates with daylight and leisure hours, not because solar rays directly compel individuals to ride bicycles.

---

## 20. Automated Test Results

The full test suite was executed via pytest in the active Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Test Suite Execution Output:
- `tests/test_preprocessing.py`: **15 passed** (data parsing, validation, cyclical features)
- `tests/test_baselines.py`: **9 passed** (training mean, hourly lookup, closure rules, metric math)
- `tests/test_experiment_design.py`: **8 passed** (split integrity, boundary continuity, leakage checks)
- `tests/test_model_training.py`: **21 passed** (manifest integrity, pipeline safety, leakage prevention, ranking order, model reload)
- **Total:** **53 passed, 0 failed, 0 errors** in 8.60s.

---

## 21. Generated Artifacts and Reproducibility Instructions

### Generated Artifacts Summary:
1. `src/model_training.py`: End-to-end training and evaluation script.
2. `tests/test_model_training.py`: Automated verification suite.
3. `reports/phase3b/experiment_manifest.json`: Machine-readable specification of all 8 feature configurations.
4. `reports/phase3b/candidate_results.csv`: Complete results table for all 120 candidate models.
5. `reports/phase3b/model_selection.json`: Locked candidate configuration record.
6. `reports/phase3b/final_test_metrics.json`: Test metrics for Scenario A and Scenario B.
7. `models/phase3b_champion_pipeline.joblib`: Persisted Scikit-Learn champion pipeline (MD5: `1000468ebd831efa8643731c99c83648`).
8. `reports/figures/phase3b/`: Six diagnostic visualisations.

### Reproducibility Commands:
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Re-run complete training, validation, lock, and test evaluation pipeline
python -m src.model_training

# 3. Execute complete automated test suite
python -m pytest -v
```

---

## 22. Final Phase 3B Status

Phase 3B has been executed strictly within the approved project boundaries. Model training, leakage-safe pipeline construction, candidate ranking, validation diagnostics, pre-test selection lock, champion refitting, final test evaluation, and automated testing are **100% complete and fully verified**.

**Status:** **PHASE 3B COMPLETE — AWAITING EXPLICIT USER APPROVAL FOR PHASE 4.**
