# Vehicle Rental Demand Prediction Using Weather and Temporal Features

## Project Objective

This project develops a machine learning system that predicts vehicle rental demand (specifically bike-sharing demand in Seoul, South Korea) using historical rental data, weather information, and temporal features.

The system compares the following regression models:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting / XGBoost Regression

All results are based on real data and real model predictions — no fabricated metrics or hardcoded outputs.

## Dataset

- **Source**: [UCI Seoul Bike Sharing Demand Dataset](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand)
- **Location**: `data/raw/SeoulBikeData.csv`
- **Records**: 8,760 rows × 14 columns
- **Date Coverage**: December 1, 2017 – November 30, 2018 (one full year, hourly)
- **Target Variable**: `Rented Bike Count`

## Project Directory Structure

```
vehicle-rental-demand-prediction/
├── data/
│   ├── raw/                  # Original, unmodified dataset
│   └── processed/            # Cleaned/transformed data (Phase 2+)
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py           # Package initializer
│   └── data_inspection.py    # Phase 1 dataset inspection script
├── reports/
│   ├── figures/              # Saved plots and charts
│   ├── phase1_dataset_report.md    # Markdown inspection report
│   └── phase1_dataset_summary.json # Machine-readable summary
├── models/                   # Trained model artifacts (Phase 3+)
├── app/                      # Streamlit dashboard (Phase 4+)
├── tests/                    # Test scripts
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Prerequisites

- Python 3.10 or later
- pip (Python package manager)
- Git (optional, for version control)

## Environment Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

> **Note (CMD):** If using Command Prompt instead of PowerShell, run:
> ```cmd
> .venv\Scripts\activate.bat
> ```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## How to Run Phase 1 (Data Inspection)

```powershell
python -m src.data_inspection --data "data/raw/SeoulBikeData.csv"
```

## How to Run Phase 2 (Preprocessing and EDA)

### 1. Execute Preprocessing Pipeline
```powershell
python -m src.preprocessing --input "data/raw/SeoulBikeData.csv" --output "data/processed/seoul_bike_processed.csv"
```

### 2. Run Exploratory Data Analysis & Generate Figures
```powershell
python -m src.eda --data "data/processed/seoul_bike_processed.csv" --output-dir "reports/figures"
```

## How to Run Phase 3B (Model Training, Selection, and Evaluation)

### 1. Run Complete Machine Learning Pipeline
```powershell
python -m src.model_training
```

## How to Run Phase 4 (Streamlit Web Dashboard)

### 1. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. Run All Automated Tests (76 Tests)
```powershell
pytest -v
```

## How to Run Phase 5 (Final Integration and Verification)

Run the full end-to-end test suite and verify model artifact checksum:
```powershell
python -m pytest -q
```

## Expected Output Files

| File | Description |
|------|-------------|
| `data/processed/seoul_bike_processed.csv` | Preprocessed dataset with 23 columns (8,760 rows) |
| `data/processed/preprocessing_metadata.json` | JSON metadata tracking preprocessing & zero-demand stats |
| `data/splits/train.csv` | Chronological training partition (6,132 rows, 70%) |
| `data/splits/validation.csv` | Chronological validation partition (1,314 rows, 15%) |
| `data/splits/test.csv` | Chronological test partition (1,314 rows, 15%, held out) |
| `data/splits/split_metadata.json` | JSON metadata with partition boundaries, counts, and checksums |
| `reports/phase1_dataset_report.md` | Detailed Markdown report with raw dataset statistics |
| `reports/phase1_dataset_summary.json` | JSON summary of Phase 1 inspection |
| `reports/phase2_eda_report.md` | Comprehensive Phase 2 EDA and data analysis report |
| `reports/phase3a_experiment_design.md` | Comprehensive Phase 3A experiment design report |
| `reports/phase3a/feature_sets.json` | Validated feature sets (A, B, C, D) |
| `reports/phase3a/experiment_config.json` | Master experiment design configuration and baseline metrics |
| `reports/phase3b_model_report.md` | Comprehensive Phase 3B model training, validation & test report |
| `reports/phase3b_final_audit.md` | Verification audit of Phase 3B models and metrics |
| `reports/phase3b/experiment_manifest.json` | Manifest of 8 experimental feature configurations |
| `reports/phase3b/candidate_results.csv` | Results table for all 120 candidate models |
| `reports/phase3b/model_selection.json` | Locked champion model configuration (pre-test lock) |
| `reports/phase3b/final_test_metrics.json` | Final test metrics for Scenario A and Scenario B |
| `models/phase3b_champion_pipeline.joblib` | Persisted Scikit-Learn champion pipeline |
| `reports/figures/phase3b/*.png` | 6 diagnostic plots (residuals, demand curves, importance) |
| `reports/figures/*.png` | 14 publication-grade EDA figures |
| `app.py` | Streamlit web application entry point |
| `src/dashboard_utils.py` | Dashboard helpers (feature derivation, model caching, validation) |
| `tests/test_dashboard.py` | Automated test suite for dashboard functionality (22 tests) |
| `reports/phase4_dashboard_report.md` | Comprehensive Phase 4 dashboard implementation report |
| `reports/phase5_final_verification_report.md` | Final project integration and verification report |
| `reports/deployment_readiness_report.md` | Streamlit Cloud deployment readiness audit report |

## Deployment on Streamlit Community Cloud

The dashboard is fully self-contained and pre-configured for instant deployment on [Streamlit Community Cloud](https://streamlit.io/cloud) using GitHub as the repository source.

### 1. Initialize Git and Commit the Repository
From the project directory on your local machine:
```powershell
# Initialize git if not already initialized
git init

# Stage all project files (essential model & processed data are tracked)
git add .

# Create initial deployment commit
git commit -m "Deploy: Vehicle rental demand prediction system"
```

### 2. Push to GitHub
Create a new GitHub repository (e.g., `vehicle-rental-demand-prediction`) under your GitHub account, then run:
```powershell
git remote add origin https://github.com/<your-username>/vehicle-rental-demand-prediction.git
git branch -M main
git push -u origin main
```

### 3. Deploy via Streamlit Community Cloud Dashboard
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<your-username>/vehicle-rental-demand-prediction`.
4. Select the branch: `main`.
5. Set **Main file path**: `app.py`.
6. (Optional) In **Advanced settings**, set the Python version to **3.11** or **3.12**.
7. Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, load the serialized model (`models/phase3b_champion_pipeline.joblib`), and serve the live application with full interactive prediction and demand analysis features.

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Project setup, dataset verification, data quality reporting | ✅ Complete |
| **Phase 2** | Data preprocessing, feature engineering, exploratory data analysis | ✅ Complete |
| **Phase 3A** | Experimental design, chronological splitting, feature sets, baselines | ✅ Complete |
| **Phase 3B** | Model training, tuning, comparison (Linear, Tree, Forest, XGBoost), final test evaluation | ✅ Complete |
| **Phase 4** | Dashboard development (Streamlit), interactive prediction, deployment documentation | ✅ Complete |
| **Phase 5** | Final integration, end-to-end testing, artifact integrity & project verification | ✅ Complete |

## Project Methodology & Champion Model Summary

- **Chronological Split**: 70% Train (Dec 1, 2017 – Aug 13, 2018), 15% Validation (Aug 13, 2018 – Oct 7, 2018), 15% Test (Oct 7, 2018 – Nov 30, 2018) preventing temporal data leakage.
- **Champion Model**: Gradient Boosting Regressor (100 estimators, learning rate = 0.1, max depth = 3) with Feature Set `C_combined_cat` (20 features: 18 numerical + 2 categorical).
- **Validation Metrics**: RMSE: 321.58, MAE: 237.54, R²: 0.7775 (Operating hours).
- **Final Test Metrics**:
  - *Scenario A (Operating hours)*: RMSE: 368.56, MAE: 260.94, R²: 0.5432.
  - *Scenario B (Full timeline with closure rule)*: RMSE: 354.84, MAE: 241.88, R²: 0.6120.
- **Operational Closure Rule**: When facilities are closed (`Functioning Day == 'No'`), demand is deterministically 0 without evaluating the regressor.
- **Limitations**: Evaluated on one chronological year in Seoul; extreme weather extrapolation and multi-year climate shifts may require periodic model retraining.

## License

This project is for academic purposes (B.Tech Semester 5 — Introduction to Machine Learning).


