# Phase 2 — Data Preprocessing and Exploratory Data Analysis Report

**Project**: Vehicle Rental Demand Prediction Using Weather and Temporal Features  
**Dataset**: Seoul Bike Sharing Demand (UCI Machine Learning Repository)  
**Date**: October 2026  
**Status**: Phase 2 Completed  

---

## 1. Executive Summary

Phase 2 builds upon the clean dataset verified in Phase 1 (`data/raw/SeoulBikeData.csv`), performing rigorous data validation, temporal and cyclical feature engineering, comprehensive zero-demand analysis, extreme-weather outlier assessment, and exploratory data analysis (EDA).

### Key Takeaways
1. **Preservation & Integrity**: All 8,760 hourly observations across 365 calendar days (2017-12-01 to 2018-11-30) were preserved. The original dataset MD5 checksum (`bf632307dcb419f67df589faaa0318e1`) remains completely unaltered.
2. **Deterministic Zero-Demand Root Cause**: Exactly 295 observations exhibit a rental count of 0. All 295 (100.0%) occur when `Functioning Day == 'No'`. When `Functioning Day == 'Yes'`, minimum demand is 2 bikes. Zero demand is not a data-entry error or random absence; it is a structural operational shutdown.
3. **Engineered Features**: 9 temporal and cyclical features were engineered (`day_of_week`, `month`, `day_of_month`, `day_of_year`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`), expanding the feature space from 14 to 23 columns.
4. **Extreme Weather is Genuine**: The maximum rainfall of 35.0 mm and maximum snowfall of 8.8 cm represent physically verified precipitation events that correspond with sharp suppressions in rental demand. They are retained as real, informative weather shocks.
5. **Collinearity and Weather Redundancy**: Ambient temperature and dew point temperature exhibit severe collinearity ($r = 0.9128$). Temperature is the strongest linear predictor of rental demand ($r = 0.5386$), while rain ($r = -0.1231$) and humidity ($r = -0.1998$) exert clear negative pressure.
6. **Verification & Testing**: 15 automated unit and integration tests were implemented and passed 100% under `pytest`.

---

## 2. Dataset Verification

Before any transformations, the raw dataset was re-validated against the Phase 1 benchmark:

| Metric | Phase 1 Value | Phase 2 Validation | Status |
|:---|:---:|:---:|:---:|
| Total Rows | 8,760 | 8,760 | Verified |
| Total Columns | 14 | 14 | Verified |
| Missing Values | 0 | 0 | Verified |
| Duplicate Rows | 0 | 0 | Verified |
| Duplicate (Date, Hour) Timestamps | 0 | 0 | Verified |
| MD5 Checksum | `bf632307dcb419f67df589faaa0318e1` | `bf632307dcb419f67df589faaa0318e1` | Verified (Unchanged) |
| Date Span | 2017-12-01 to 2018-11-30 | 2017-12-01 to 2018-11-30 | Verified |

All observations were chronologically ordered by `Date` and `Hour`.

---

## 3. Preprocessing Decisions

1. **Unambiguous Date Parsing**: The raw date string is formatted as `dd/mm/yyyy`. Pandas was explicitly instructed with `format="%d/%m/%Y"`. To prevent ambiguous parsing across environments, the processed dataset stores dates in the standard ISO `YYYY-MM-DD` format.
2. **Zero Row Deletions**: No rows were dropped. Every hour represents valid real-world operating conditions or operational shutdowns.
3. **Categorical Feature Preservation**: The categorical features (`Seasons`, `Holiday`, `Functioning Day`) were preserved in their human-readable string representations. One-hot encoding is intentionally deferred to Phase 3 training pipelines to prevent data leakage and maintain clean preprocessing separation.
4. **No Feature-Target Leakage**: No features were constructed from `Rented Bike Count`. The target variable remains completely isolated.
5. **No Synthetic Replacement**: All values reflect observed measurements; no synthetic interpolation or imputation was performed.

---

## 4. Engineered Feature Descriptions

To allow regression models (Linear Regression, Trees, Forests, and Gradient Boosting) to capture seasonal periodicity and weekly rhythms, 9 features were added:

| Feature Name | Type | Mathematical Formula / Derivation | Range / Values | Rationale |
|:---|:---:|:---|:---:|:---|
| `day_of_week` | Integer | `date.dt.day_of_week` | 0 (Mon) to 6 (Sun) | Captures day-specific commuting habits. |
| `month` | Integer | `date.dt.month` | 1 to 12 | Captures macro-seasonal progression. |
| `day_of_month` | Integer | `date.dt.day` | 1 to 31 | Captures intra-month temporal trends. |
| `day_of_year` | Integer | `date.dt.day_of_year` | 1 to 365 | Continuous calendar counter for chronological splits. |
| `is_weekend` | Binary | `1 if day_of_week >= 5 else 0` | {0, 1} | Distinguishes commuter behavior from leisure rentals. |
| `hour_sin` | Float | $\sin(2\pi \cdot \text{Hour} / 24)$ | $[-1.0, 1.0]$ | Preserves continuity between 23:00 and 00:00 on the circle. |
| `hour_cos` | Float | $\cos(2\pi \cdot \text{Hour} / 24)$ | $[-1.0, 1.0]$ | Orthogonal coordinate for continuous 24-hour cycle. |
| `month_sin` | Float | $\sin(2\pi \cdot \text{month} / 12)$ | $[-1.0, 1.0]$ | Preserves continuity between December (12) and January (1). |
| `month_cos` | Float | $\cos(2\pi \cdot \text{month} / 12)$ | $[-1.0, 1.0]$ | Orthogonal coordinate for continuous 12-month seasonal cycle. |

Both raw calendar variables (`Hour`, `month`) and cyclical trigonometric transforms (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`) are retained so tree-based models and linear models can be compared fairly in Phase 3.

---

## 5. Zero-Demand Analysis

Phase 1 highlighted 295 records where `Rented Bike Count == 0`. Investigation revealed a strict deterministic relationship with system operational status:

```
Cross-Tabulation: Functioning Day vs. Zero Rental Count
-------------------------------------------------------
Functioning Day      Demand > 0       Demand == 0       Total Hours      Zero Rate
No                            0               295               295        100.00%
Yes                       8,465                 0             8,465          0.00%
-------------------------------------------------------
Total                     8,465               295             8,760          3.37%
```

### Key Zero-Demand Findings:
- **100% Correlation**: Exactly 295 records have `Functioning Day == 'No'`, and every single one of those 295 records has a rental count of 0.
- **Operating Hours Never Hit Zero**: On functioning days (`Functioning Day == 'Yes'`), the minimum rental count across the entire year is 2 bikes. Demand never drops to zero when stations are open.
- **Affected Dates**: Non-functioning status spans 13 unique dates:
  - 12 full 24-hour closure dates (288 hours): `2018-04-11`, `2018-05-10`, `2018-09-18`, `2018-09-19`, `2018-09-28`, `2018-09-30`, `2018-10-02`, `2018-10-04`, `2018-10-09`, `2018-11-03`, `2018-11-06`, `2018-11-09`.
  - 1 partial 7-hour closure date (7 hours): `2018-10-06` (Hours 0 through 6).
  - Total: $12 \times 24 + 7 = 295$ hours.
- **Seasonal Distribution**: 247 zero-demand hours occurred in Autumn (83.7%), and 48 occurred in Spring (16.3%). Zero occurred in Summer or Winter.

### Modeling Implications for Phase 3:
Because `Functioning Day == 'No'` deterministically forces demand to zero, models trained without recognizing this structure could make erroneous non-zero predictions for closed days, or drag down functioning-day predictions. A natural two-stage or gated modeling strategy should be evaluated in Phase 3:
$$\widehat{y} = \begin{cases} 0, & \text{if Functioning Day} = \text{'No'} \\ f(X), & \text{if Functioning Day} = \text{'Yes'} \end{cases}$$

---

## 6. Outlier and Extreme-Weather Analysis

Using the standard Tukey boxplot rule ($\text{IQR} = Q_3 - Q_1$, thresholds at $Q_1 - 1.5 \cdot \text{IQR}$ and $Q_3 + 1.5 \cdot \text{IQR}$):

| Variable | Min | Q1 | Median | Q3 | Max | IQR | Lower Bound | Upper Bound | Outlier Count | % Outliers | Physical Assessment |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `Rainfall(mm)` | 0.0 | 0.0 | 0.0 | 0.0 | 35.0 | 0.0 | 0.0 | 0.0 | 528 | 6.03% | Real weather shock |
| `Snowfall (cm)` | 0.0 | 0.0 | 0.0 | 0.0 | 8.8 | 0.0 | 0.0 | 0.0 | 443 | 5.06% | Real weather shock |
| `Rented Bike Count` | 0.0 | 191.0 | 504.5 | 1065.25 | 3556.0 | 874.25 | -1120.38 | 2376.62 | 158 | 1.80% | Rush hour demand peaks |
| `Wind speed (m/s)` | 0.0 | 0.9 | 1.5 | 2.3 | 7.4 | 1.4 | -1.20 | 4.40 | 161 | 1.84% | Plausible gust events |
| `Solar Radiation (MJ/m2)` | 0.0 | 0.0 | 0.01 | 0.93 | 3.52 | 0.93 | -1.40 | 2.33 | 641 | 7.32% | Peak midday solar irradiance |
| `Visibility (10m)` | 27.0 | 940.0 | 1698.0 | 2000.0 | 2000.0 | 1060.0 | -650.0 | 3590.0 | 0 | 0.00% | Capped at instrument limit (2000) |
| `Temperature(°C)` | -17.8 | 3.5 | 13.7 | 22.5 | 39.4 | 19.0 | -25.0 | 51.0 | 0 | 0.00% | Seoul continental climate |
| `Humidity(%)` | 0.0 | 42.0 | 57.0 | 74.0 | 98.0 | 32.0 | -6.0 | 122.0 | 0 | 0.00% | Atmospheric limits |

### Detailed Investigation of Extreme Cases:
1. **Maximum Rainfall (35.0 mm)**: Occurred on `2018-05-16` at Hour 13 (1:00 PM) during a heavy spring downpour. Rented bike count was only 151 bikes (typical 1:00 PM demand is ~800+ bikes). Rain dramatically dampens bike usage.
2. **Maximum Snowfall (8.8 cm)**: Occurred on `2018-11-24` at Hours 10:00 and 11:00 AM during a major early winter snowstorm. Bike counts plummeted to 62 and 114 bikes.
3. **Maximum Rental Count (3,556 bikes)**: Occurred on `2018-06-19` at Hour 18 (6:00 PM evening rush hour) during pleasant summer weather ($27.7^\circ\text{C}$, 48% humidity).
4. **IQR Threshold Caveat**: For rainfall and snowfall, $Q_1 = Q_3 = 0.0$, making $\text{IQR} = 0.0$. Consequently, every hour with any precipitation is flagged as a statistical outlier. This is an artifact of zero-inflated distributions, not corrupt data.
5. **Decision**: Retain all records. These extreme weather observations are vital for teaching machine learning models how demand reacts during severe weather conditions.

---

## 7. Summary Statistics

Descriptive metrics for all numerical features in the processed dataset ($N = 8,760$):

| Column | Mean | Std | Min | 25% | Median | 75% | Max | Skewness |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Rented Bike Count` | 704.60 | 645.00 | 0.00 | 191.00 | 504.50 | 1065.25 | 3556.00 | +1.15 |
| `Hour` | 11.50 | 6.92 | 0.00 | 5.75 | 11.50 | 17.25 | 23.00 | 0.00 |
| `Temperature(°C)` | 12.88 | 11.94 | -17.80 | 3.50 | 13.70 | 22.50 | 39.40 | -0.20 |
| `Humidity(%)` | 58.23 | 20.36 | 0.00 | 42.00 | 57.00 | 74.00 | 98.00 | +0.06 |
| `Wind speed (m/s)` | 1.72 | 1.04 | 0.00 | 0.90 | 1.50 | 2.30 | 7.40 | +0.89 |
| `Visibility (10m)` | 1436.83 | 608.30 | 27.00 | 940.00 | 1698.00 | 2000.00 | 2000.00 | -0.70 |
| `Dew point temp(°C)`| 4.07 | 13.06 | -30.60 | -4.70 | 5.10 | 14.80 | 27.20 | -0.37 |
| `Solar Rad (MJ/m2)` | 0.57 | 0.87 | 0.00 | 0.00 | 0.01 | 0.93 | 3.52 | +1.50 |
| `Rainfall(mm)` | 0.15 | 1.13 | 0.00 | 0.00 | 0.00 | 0.00 | 35.00 | +14.53 |
| `Snowfall (cm)` | 0.08 | 0.44 | 0.00 | 0.00 | 0.00 | 0.00 | 8.80 | +8.44 |
| `day_of_week` | 2.99 | 2.00 | 0.00 | 1.00 | 3.00 | 5.00 | 6.00 | +0.01 |
| `month` | 6.53 | 3.45 | 1.00 | 4.00 | 7.00 | 10.00 | 12.00 | -0.01 |
| `is_weekend` | 0.28 | 0.45 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | +0.96 |

---

## 8. EDA Chart Inventory and File Paths

All 14 figures were generated and saved in `reports/figures/` at 150 DPI with tight layouts:

| # | Figure File Name | Key Insight Displayed |
|:---:|:---|:---|
| 1 | `reports/figures/01_distribution_rented_bike_count.png` | Positively skewed demand distribution ($\text{skew} = +1.15$), mean (704.6) > median (504.5). |
| 2 | `reports/figures/02_avg_demand_by_hour.png` | Pronounced bimodal commuter profile peaking at 08:00 (1,019 bikes) and 18:00 (1,503 bikes). |
| 3 | `reports/figures/03_avg_demand_by_day_of_week.png` | Steady demand on weekdays (~700–747 bikes), dipping on Sundays (625.2 bikes). |
| 4 | `reports/figures/04_avg_demand_by_month.png` | Peak ridership in June (1,234 bikes/hr), lowest ridership in January (203 bikes/hr). |
| 5 | `reports/figures/05_avg_demand_by_season.png` | Summer leads with 1,034.1 bikes/hr; Winter drops to 225.5 bikes/hr. |
| 6 | `reports/figures/06_demand_vs_temperature.png` | Positive non-linear trend peaking near 25–28°C before plateauing in extreme heat. |
| 7 | `reports/figures/07_demand_vs_humidity.png` | Negative relationship; high humidity (>80%) corresponds to steep demand suppression. |
| 8 | `reports/figures/08_demand_vs_rainfall.png` | Dry hours average 743.8 bikes; rainy hours drop to 184.2 bikes (75% reduction). |
| 9 | `reports/figures/09_demand_vs_snowfall.png` | Snow-free hours average 727.8 bikes; snowy hours drop to 270.4 bikes. |
| 10 | `reports/figures/10_boxplot_demand_by_season.png` | Seasonal interquartile shifts and variance: Summer has widest spread, Winter is tightly clustered low. |
| 11 | `reports/figures/11_zero_demand_by_functioning_day.png` | 100% of zero-demand records reside in `Functioning Day == 'No'`; 0% in `Yes`. |
| 12 | `reports/figures/12_correlation_heatmap.png` | Comprehensive Pearson correlation matrix across weather and temporal features. |
| 13 | `reports/figures/13_hourly_demand_by_season.png` | Diurnal commuter spikes exist in all four seasons, but Winter shifts down uniformly. |
| 14 | `reports/figures/14_demand_vs_solar_radiation.png` | Moderate positive relationship during daylight hours; demand rises with clear sunshine. |

---

## 9. Correlation Findings

### Pearson Correlation with `Rented Bike Count` ($r$):
```
Feature                         Correlation (r)       Direction / Interpretation
---------------------------------------------------------------------------------
Temperature(°C)                        +0.5386        Strong positive: warmer weather drives usage
Hour                                   +0.4103        Moderate positive: daytime/evening usage > night
Dew point temperature(°C)              +0.3798        Moderate positive: tracks ambient temperature
Solar Radiation (MJ/m2)                +0.2618        Mild positive: daylight and clear skies
Visibility (10m)                       +0.1993        Mild positive: clear air improves riding safety
month                                  +0.1335        Mild positive: seasonal progression
day_of_year                            +0.1330        Mild positive: calendar trend
Wind speed (m/s)                       +0.1211        Mild positive: gentle breezes accompany clear days
day_of_month                           +0.0223        Near zero: no calendar day effect
day_of_week                            -0.0294        Near zero: Sunday dips offset weekday stability
is_weekend                             -0.0365        Near zero: aggregate weekend demand close to weekday
hour_cos                               -0.0995        Negative: cyclical hour component
Rainfall(mm)                           -0.1231        Negative: immediate suppression from precipitation
Snowfall (cm)                          -0.1418        Negative: suppression from snow and freezing roads
Humidity(%)                            -0.1998        Negative: uncomfortable mugginess / rain proxy
month_sin                              -0.2443        Cyclical month component
month_cos                              -0.3994        Cyclical month component
hour_sin                               -0.4318        Cyclical hour component
```

### Critical Multicollinearity Warning:
- **Temperature vs. Dew Point Temperature**: The correlation between ambient temperature and dew point temperature is **$r = 0.9128$**.
- This is extreme multicollinearity. For ordinary linear regression models, using both features simultaneously can inflate standard errors and produce unstable coefficients. In tree-based models, feature importance may split arbitrarily between the two. In Phase 3 feature selection, we should evaluate keeping ambient temperature and dropping or transforming dew point temperature (e.g., relative humidity deficit).

### Limitations of Pearson Correlation:
- Pearson correlation measures only strictly linear relationships.
- Non-linear relationships (such as the inverted U-shape of temperature above 32°C, or the sudden threshold cutoff of rainfall above 2 mm) are poorly captured by single linear coefficients. Non-linear models (Decision Trees, Random Forests, XGBoost) will capture these dynamics significantly better than standard linear regression.

---

## 10. Limitations and Cautions

1. **Historical Weather vs. Operational Forecasts**: In this historical dataset, weather features are exact ground-truth observations. In real-world deployment, rental demand must be predicted using **weather forecasts** which carry uncertainty. Models evaluated on observed weather represent an upper bound on operational accuracy.
2. **Deterministic Shutdown Masking**: If `Functioning Day` is included as a simple linear covariate without interaction, a model could predict negative demand when other features are small, or positive demand when `Functioning Day == 'No'`. A two-stage architecture or filtering out non-functioning days must be evaluated in Phase 3.
3. **Temporal Autocorrelation**: Because this is an hourly time series over a single year, consecutive observations are highly correlated ($y_t \approx y_{t-1}$). Random k-fold cross-validation would cause data leakage from future to past. **Chronological time-series splitting** is mandatory in Phase 3.

---

## 11. Processed Dataset Details

- **Output File**: `data/processed/seoul_bike_processed.csv`
- **Metadata File**: `data/processed/preprocessing_metadata.json`
- **Total Records**: 8,760
- **Total Features**: 23
- **File Format**: UTF-8 CSV, comma-delimited, header row, no index column.

### Processed Schema (23 Columns):
```
 1. Date                        (object, string 'YYYY-MM-DD')
 2. Rented Bike Count           (int64, target variable)
 3. Hour                        (int64, 0 to 23)
 4. Temperature(°C)             (float64)
 5. Humidity(%)                 (int64)
 6. Wind speed (m/s)            (float64)
 7. Visibility (10m)            (int64)
 8. Dew point temperature(°C)   (float64)
 9. Solar Radiation (MJ/m2)     (float64)
10. Rainfall(mm)                (float64)
11. Snowfall (cm)               (float64)
12. Seasons                     (object, 'Spring' | 'Summer' | 'Autumn' | 'Winter')
13. Holiday                     (object, 'No Holiday' | 'Holiday')
14. Functioning Day             (object, 'Yes' | 'No')
15. day_of_week                 (int64, 0=Mon to 6=Sun)
16. month                       (int64, 1 to 12)
17. day_of_month                (int64, 1 to 31)
18. day_of_year                 (int64, 1 to 365)
19. is_weekend                  (int64, 0=weekday, 1=weekend)
20. hour_sin                    (float64, [-1.0, 1.0])
21. hour_cos                    (float64, [-1.0, 1.0])
22. month_sin                   (float64, [-1.0, 1.0])
23. month_cos                   (float64, [-1.0, 1.0])
```

---

## 12. Automated Test Results

Automated test suite `tests/test_preprocessing.py` was executed using pytest:

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1
rootdir: D:\B-Tech\SEM 5\IML\project
collected 15 items

tests/test_preprocessing.py::test_date_parsing_and_formatting PASSED     [  6%]
tests/test_preprocessing.py::test_temporal_feature_calculations PASSED   [ 13%]
tests/test_preprocessing.py::test_is_weekend_calculation PASSED          [ 20%]
tests/test_preprocessing.py::test_hour_cyclical_features PASSED          [ 26%]
tests/test_preprocessing.py::test_month_cyclical_features PASSED         [ 33%]
tests/test_preprocessing.py::test_validation_rejects_invalid_hours PASSED [ 40%]
tests/test_preprocessing.py::test_validation_rejects_negative_rentals PASSED [ 46%]
tests/test_preprocessing.py::test_validation_rejects_invalid_categorical PASSED [ 53%]
tests/test_preprocessing.py::test_validation_rejects_duplicate_timestamps PASSED [ 60%]
tests/test_preprocessing.py::test_real_dataset_row_and_column_count PASSED [ 66%]
tests/test_preprocessing.py::test_real_dataset_target_preservation PASSED [ 73%]
tests/test_preprocessing.py::test_real_dataset_zero_demand_preservation PASSED [ 80%]
tests/test_preprocessing.py::test_real_dataset_missing_values PASSED     [ 86%]
tests/test_preprocessing.py::test_real_dataset_duplicate_timestamps PASSED [ 93%]
tests/test_preprocessing.py::test_metadata_file_validity PASSED          [100%]

============================= 15 passed in 1.40s ==============================
```

All 15 assertions passed with zero failures or warnings.

---

## 13. Recommendations for Phase 3 (Model Training and Evaluation)

1. **Chronological Train/Validation/Test Split**:
   - Because the data is hourly time series, do not use random shuffle splits.
   - Use a chronological split (e.g. first 9–10 months for training/cross-validation, final 2–3 months for unseen holdout testing).
2. **Handling Functioning Day**:
   - Evaluate two candidate paradigms:
     a) Train regression models only on `Functioning Day == 'Yes'`, applying an exact zero rule if `Functioning Day == 'No'`.
     b) Include one-hot encoded `Functioning Day` in tree-based algorithms (XGBoost, Random Forest) which can naturally branch on binary flags.
3. **Feature Selection & Multicollinearity**:
   - For Linear Regression, drop `Dew point temperature(°C)` to mitigate the 0.91 collinearity with `Temperature(°C)`.
   - Compare raw `Hour`/`month` against cyclical `hour_sin`/`cos` and `month_sin`/`cos`.
4. **Evaluation Metrics**:
   - Evaluate all models using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).
   - Real predictions only; no synthetic or hardcoded metrics.
5. **Model Portfolio to Benchmark**:
   - Baseline: Ridge / Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting / XGBoost Regressor

---
*Report compiled autonomously following strict empirical verification of Seoul Bike Sharing data.*
