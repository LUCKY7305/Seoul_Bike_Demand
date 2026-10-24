"""
Vehicle Rental Demand Prediction — Streamlit Web Application
============================================================

Phase 4: Production-grade interactive academic dashboard for:
1. Overview: Executive summary, key dataset statistics, operational motivation, and workflow.
2. Predict Demand: Interactive prediction form using the saved Phase 3B champion pipeline,
   with real-time feature derivation, input validation, baseline comparison, and closure rule.
3. Demand Analysis: Interactive Plotly exploratory analysis of historical demand patterns.
4. Model Performance: Verified Phase 3B evaluation results, baseline fairness comparison,
   impurity-based feature importance, and diagnostic figure gallery.
5. About the Project: Theoretical methodology, chronological splitting protocol, and limitations.

Run with:
    streamlit run app.py
"""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import (
    ALL_INPUT_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_TRAIN_SPLIT_PATH,
    EXPECTED_HOLIDAYS,
    EXPECTED_SEASONS,
    NUMERICAL_FEATURES,
    construct_model_input_row,
    get_feature_importance_df,
    get_verified_metrics,
    load_champion_pipeline,
    load_hourly_baseline_lookup,
    load_processed_data,
    predict_demand,
    validate_user_inputs,
)

# ---------------------------------------------------------------------------
# Page Configuration & Theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vehicle Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Academic Data-Science Aesthetic
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card-sub {
        font-size: 0.80rem;
        color: #64748b;
        margin-top: 4px;
    }
    /* Prediction Hero Card */
    .prediction-hero {
        background: linear-gradient(135deg, #0f766e 0%, #0e7490 100%);
        border-radius: 10px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .prediction-hero-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .prediction-hero-value {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .prediction-hero-unit {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.85;
        margin-top: 6px;
    }
    /* Info Box */
    .note-box {
        background-color: #f0fdfa;
        border-left: 4px solid #0d9488;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #134e4a;
    }
    .warn-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #78350f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading trained machine learning model...")
def get_pipeline():
    """Load and cache the Scikit-Learn champion pipeline."""
    return load_champion_pipeline(DEFAULT_MODEL_PATH)


@st.cache_resource(show_spinner="Computing baseline lookups...")
def get_baseline_lookup():
    """Compute and cache the hourly-mean baseline lookup table."""
    return load_hourly_baseline_lookup(DEFAULT_TRAIN_SPLIT_PATH)


@st.cache_data(show_spinner="Loading Seoul Bike Sharing dataset...")
def get_data():
    """Load and cache the master processed dataset."""
    return load_processed_data(DEFAULT_PROCESSED_DATA_PATH)


# ---------------------------------------------------------------------------
# Sidebar Navigation & Information
# ---------------------------------------------------------------------------


def render_sidebar():
    """Render sidebar navigation, model details, and project metadata."""
    with st.sidebar:
        st.markdown("## 🚲 Seoul Bike Demand")
        st.markdown(
            "**Vehicle Rental Demand Prediction**  \n"
            "*Machine Learning Regression System*"
        )
        st.markdown("---")

        page = st.radio(
            "**Navigation**",
            options=[
                "🏠 Project Overview",
                "🔮 Predict Demand",
                "📊 Demand Analysis",
                "📈 Model Performance",
                "ℹ️ About the Project",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### 🤖 Champion Model Info")
        st.markdown(
            "- **Algorithm**: Gradient Boosting Regressor\n"
            "- **Trees**: 100 estimators ($d=3, \\eta=0.1$)\n"
            "- **Feature Set**: `C_combined_cat` (20 inputs)\n"
            "- **Validation RMSE**: 321.58\n"
            "- **Test RMSE**: 368.56 ($R^2 = 0.5432$)"
        )

        st.markdown("### 📁 Dataset Specifications")
        st.markdown(
            "- **Records**: 8,760 hourly observations\n"
            "- **Timeframe**: Dec 1, 2017 – Nov 30, 2018\n"
            "- **Location**: Seoul, South Korea\n"
            "- **Source**: UCI Machine Learning Repository"
        )

        st.markdown("---")
        st.caption("IML College Project — Phase 4 Streamlit Dashboard")

        return page


# ---------------------------------------------------------------------------
# Page 1: Project Overview
# ---------------------------------------------------------------------------


def render_overview():
    """Render Page 1: Project Overview."""
    st.markdown("# Vehicle Rental Demand Prediction")
    st.markdown(
        "#### *Predict hourly bike-sharing demand using meteorological and temporal features.*"
    )

    st.markdown(
        """
        Public bike-sharing programs are a cornerstone of modern urban transport, reducing congestion 
        and emissions. However, fleet operators face significant challenges in **inventory rebalancing**: 
        stations in business districts frequently deplete during morning commutes and overflow in the 
        evenings. Extreme weather (freezing temperatures, heavy rain, excessive humidity) further disrupts 
        usage patterns.

        This system applies **machine learning regression pipelines** to historical data from the 
        Seoul Public Bike Sharing System (*Ddareungi*) to forecast hourly demand, enabling dynamic 
        fleet dispatching and resource allocation.
        """
    )

    st.markdown("---")
    st.subheader("Key Project Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Dataset Observations</div>
                <div class="metric-card-value">8,760</div>
                <div class="metric-card-sub">1 full calendar year (hourly)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Timeframe</div>
                <div class="metric-card-value">2017–2018</div>
                <div class="metric-card-sub">Dec 1, 2017 to Nov 30, 2018</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Validation RMSE</div>
                <div class="metric-card-value">321.58</div>
                <div class="metric-card-sub">46.5% lower than hourly baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Unseen Test RMSE</div>
                <div class="metric-card-value">368.56</div>
                <div class="metric-card-sub">Scenario A: Operating hours (R² = 0.5432)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Prediction Architecture Workflow")

    st.markdown(
        """
        ```text
        [User Inputs]
        Date & Time + Weather Conditions (Temperature, Humidity, Rain, Solar, etc.)
               │
               ▼
        [Feature Engineering Pipeline]
        Derives: day_of_week, month, day_of_year, is_weekend, hour_sin/cos, month_sin/cos
               │
               ▼
        [Leakage-Safe Preprocessing]
        Passthrough for 18 continuous metrics + One-Hot Encoding for Seasons & Holiday
               │
               ▼
        [Champion Estimator]
        Gradient Boosting Regressor (100 estimators, max depth 3, learning rate 0.1)
               │
               ▼
        [Operational Prediction]
        If Functioning Day == 'No' ──> Exactly 0 rentals (Administrative Closure Rule)
        If Functioning Day == 'Yes' ──> Continuous Regression Output (rounded integer bikes/hr)
        ```
        """
    )

    st.markdown("---")
    st.subheader("Dataset Summary & Integrity Audit")

    try:
        df = get_data()
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Dataset Schema Information**")
            summary_info = {
                "Metric": [
                    "Total Records",
                    "Total Columns",
                    "Target Variable",
                    "Missing Values",
                    "Duplicate Timestamps",
                    "Zero-Demand Observations",
                    "Zero-Demand During Open Hours",
                    "Zero-Demand During Closed Hours",
                ],
                "Verified Value": [
                    "8,760",
                    "23",
                    "Rented Bike Count (integer)",
                    "0 (100% complete)",
                    "0 (Unique hourly timestamps)",
                    "295 (3.37% of dataset)",
                    "0 (Demand is strictly positive when open)",
                    "295 (100% explained by facility closure)",
                ],
            }
            st.dataframe(pd.DataFrame(summary_info), use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("**Chronological Split Partitioning**")
            splits_info = {
                "Partition": ["Training (70%)", "Validation (15%)", "Test (15% Quarantine)"],
                "Total Rows": ["6,132", "1,314", "1,314"],
                "Operating Rows": ["6,084", "1,163", "1,218"],
                "Date Coverage": [
                    "2017-12-01 to 2018-08-13",
                    "2018-08-13 to 2018-10-07",
                    "2018-10-07 to 2018-11-30",
                ],
            }
            st.dataframe(pd.DataFrame(splits_info), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load dataset summary: {e}")


# ---------------------------------------------------------------------------
# Page 2: Predict Demand
# ---------------------------------------------------------------------------


def render_predict():
    """Render Page 2: Predict Demand."""
    st.markdown("# 🔮 Predict Hourly Bike Rental Demand")
    st.markdown(
        "Enter temporal parameters, meteorological observations, and calendar status "
        "to generate a demand estimate using the saved champion machine learning pipeline."
    )

    try:
        pipeline = get_pipeline()
        hourly_lookup = get_baseline_lookup()
    except Exception as e:
        st.error(f"Model initialization error: {e}")
        return

    st.markdown("---")

    with st.form("prediction_form"):
        col_temp, col_met = st.columns(2)

        with col_temp:
            st.subheader("1. Date & Time Parameters")
            target_date = st.date_input(
                "Prediction Date",
                value=date(2018, 9, 15),
                min_value=date(2017, 1, 1),
                max_value=date(2028, 12, 31),
                help="Date for prediction. Historical data spans Dec 2017 – Nov 2018.",
            )

            hour = st.slider(
                "Hour of Day (0 to 23)",
                min_value=0,
                max_value=23,
                value=18,
                step=1,
                help="Hour in 24-hour format. 8:00 AM and 6:00 PM (18:00) represent historical peak commute hours.",
            )

            # Auto-suggest Season based on Month
            month_idx = target_date.month
            default_season_idx = 0
            if month_idx in [3, 4, 5]:
                default_season_idx = 0  # Spring
            elif month_idx in [6, 7, 8]:
                default_season_idx = 1  # Summer
            elif month_idx in [9, 10, 11]:
                default_season_idx = 2  # Autumn
            else:
                default_season_idx = 3  # Winter

            season = st.selectbox(
                "Season",
                options=list(EXPECTED_SEASONS),
                index=default_season_idx,
                help="Meteorological season defined in Seoul.",
            )

            holiday = st.selectbox(
                "Holiday Status",
                options=list(EXPECTED_HOLIDAYS),
                index=0,
                help="Whether the day is an official public holiday in South Korea.",
            )

            functioning_day = st.selectbox(
                "Facility Operating Status (Scenario B Routing)",
                options=["Yes (Operational)", "No (Administrative Closure)"],
                index=0,
                help="When closed ('No'), the system applies the deterministic closure rule (0 demand).",
            )
            func_val = "Yes" if "Yes" in functioning_day else "No"

        with col_met:
            st.subheader("2. Meteorological Observations")

            col_met_1, col_met_2 = st.columns(2)
            with col_met_1:
                temperature = st.number_input(
                    "Temperature (°C)",
                    min_value=-25.0,
                    max_value=45.0,
                    value=22.0,
                    step=0.5,
                    help="Ambient dry bulb temperature in Celsius.",
                )
                humidity = st.number_input(
                    "Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=55.0,
                    step=1.0,
                    help="Relative atmospheric humidity percentage (0-100%).",
                )
                dew_point = st.number_input(
                    "Dew Point Temperature (°C)",
                    min_value=-35.0,
                    max_value=32.0,
                    value=12.0,
                    step=0.5,
                    help="Dew point temperature in Celsius (strongly correlated with temperature).",
                )
                wind_speed = st.number_input(
                    "Wind Speed (m/s)",
                    min_value=0.0,
                    max_value=15.0,
                    value=1.8,
                    step=0.1,
                    help="Average wind speed in meters per second.",
                )

            with col_met_2:
                visibility = st.number_input(
                    "Visibility (10m units)",
                    min_value=20.0,
                    max_value=2000.0,
                    value=1800.0,
                    step=50.0,
                    help="Atmospheric visibility in 10-meter units (e.g., 2000 = 20 km).",
                )
                solar_radiation = st.number_input(
                    "Solar Radiation (MJ/m²)",
                    min_value=0.0,
                    max_value=4.0,
                    value=0.85,
                    step=0.05,
                    help="Incoming solar radiation in megajoules per square meter.",
                )
                rainfall = st.number_input(
                    "Rainfall (mm)",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.5,
                    help="Precipitation accumulated during the hour in millimeters.",
                )
                snowfall = st.number_input(
                    "Snowfall (cm)",
                    min_value=0.0,
                    max_value=25.0,
                    value=0.0,
                    step=0.5,
                    help="Snowfall accumulation in centimeters.",
                )

        submit = st.form_submit_button("🚀 Predict Rental Demand", use_container_width=True)

    # Derived Feature Preview
    with st.expander("🔍 View Derived Temporal and Cyclical Features"):
        try:
            sample_row = construct_model_input_row(
                target_date, hour, temperature, humidity, wind_speed, visibility,
                dew_point, solar_radiation, rainfall, snowfall, season, holiday
            )
            st.dataframe(sample_row.T.rename(columns={0: "Feature Value"}), use_container_width=True)
        except Exception as err:
            st.caption(f"Feature calculation preview: {err}")

    if submit:
        # 1. Validation
        try:
            validate_user_inputs(
                hour=hour,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                visibility=visibility,
                dew_point=dew_point,
                solar_radiation=solar_radiation,
                rainfall=rainfall,
                snowfall=snowfall,
                season=season,
                holiday=holiday,
                functioning_day=func_val,
            )
        except ValueError as val_err:
            st.error(f"⚠️ Input Validation Error: {val_err}")
            return

        # 2. Construct Model Input
        try:
            input_df = construct_model_input_row(
                target_date=target_date,
                hour=hour,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                visibility=visibility,
                dew_point=dew_point,
                solar_radiation=solar_radiation,
                rainfall=rainfall,
                snowfall=snowfall,
                season=season,
                holiday=holiday,
            )
        except Exception as e:
            st.error(f"Error constructing model features: {e}")
            return

        # 3. Generate Prediction
        try:
            pred_dict = predict_demand(
                pipeline=pipeline,
                input_row=input_df,
                functioning_day=func_val,
                hourly_lookup=hourly_lookup,
            )
        except Exception as e:
            st.error(f"Inference pipeline execution failed: {e}")
            return

        # 4. Display Results Hero Banner
        disp_val = pred_dict["display_prediction"]
        raw_val = pred_dict["raw_prediction"]

        st.markdown(
            f"""
            <div class="prediction-hero">
                <div class="prediction-hero-title">Predicted Hourly Rental Demand</div>
                <div class="prediction-hero-value">{disp_val:,}</div>
                <div class="prediction-hero-unit">bikes / hour (rounded integer)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Context Cards
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Raw Continuous Regression Output",
                value=f"{raw_val:.2f} bikes",
                help="Direct output of the unrounded GradientBoostingRegressor model.",
            )

        with res_col2:
            base_val = pred_dict["baseline_prediction"]
            if base_val is not None:
                diff_val = pred_dict["diff_vs_baseline"]
                st.metric(
                    label=f"Phase 3A Hourly Mean Baseline (Hour {hour:02d}:00)",
                    value=f"{base_val:.1f} bikes",
                    delta=f"{diff_val:+.1f} vs baseline",
                    delta_color="normal",
                    help="Historical average demand for this hour during operational days.",
                )
            else:
                st.metric("Baseline Lookup", "N/A")

        with res_col3:
            day_name = target_date.strftime("%A")
            is_wknd = "Weekend" if target_date.weekday() >= 5 else "Weekday"
            st.metric(
                label="Calendar Context",
                value=f"{day_name}",
                delta=f"{is_wknd} • {season}",
                delta_color="off",
            )

        # Operational Closure or Normal Explanation
        if pred_dict["is_closure"]:
            st.markdown(
                """
                <div class="warn-box">
                    <strong>Administrative Closure Applied:</strong> Facility operating status was set to 'No'. 
                    Under Scenario B operational protocol, the deterministic closure rule predicts exactly 0 rentals.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="note-box">
                    <strong>Model Explanation:</strong> The prediction is generated by the locked Gradient Boosting Regressor 
                    using 20 input features. Non-linear relationships between ambient temperature, diurnal commuting schedule, 
                    and precipitation are accounted for.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Extrapolation Warning if outside dataset calendar year
        if target_date < date(2017, 12, 1) or target_date > date(2018, 11, 30):
            st.warning(
                f"ℹ️ **Temporal Extrapolation Note:** The selected date ({target_date}) lies outside the historical "
                "training window (Dec 1, 2017 to Nov 30, 2018). While cyclical features and seasonal mappings generalize, "
                "long-term fleet growth or urban shifts are not represented."
            )


# ---------------------------------------------------------------------------
# Page 3: Demand Analysis
# ---------------------------------------------------------------------------


def render_analysis():
    """Render Page 3: Demand Analysis."""
    st.markdown("# 📊 Interactive Demand Analysis")
    st.markdown(
        "Explore historical demand dynamics, weather interactions, and diurnal patterns "
        "across the 8,760 verified observations in the Seoul Bike Sharing dataset."
    )

    try:
        df = get_data()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # Interactive Filters
    st.subheader("Filter Observations")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        selected_seasons = st.multiselect(
            "Select Seasons",
            options=list(EXPECTED_SEASONS),
            default=list(EXPECTED_SEASONS),
        )

    with f_col2:
        selected_holidays = st.multiselect(
            "Select Holiday Status",
            options=list(EXPECTED_HOLIDAYS),
            default=list(EXPECTED_HOLIDAYS),
        )

    with f_col3:
        hour_range = st.slider("Select Hour Range", min_value=0, max_value=23, value=(0, 23))

    # Apply filters
    filtered_df = df[
        (df["Seasons"].isin(selected_seasons))
        & (df["Holiday"].isin(selected_holidays))
        & (df["Hour"].between(hour_range[0], hour_range[1]))
    ]

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** observations ({len(filtered_df)/len(df)*100:.1f}%)")

    if filtered_df.empty:
        st.warning("No records match the selected filter criteria. Please adjust your filters.")
        return

    st.markdown("---")

    # Chart 1: Hourly Demand Profile
    st.subheader("1. Diurnal Demand Pattern (Commuting vs. Leisure Rhythms)")
    hourly_avg = (
        filtered_df.groupby(["Hour", "is_weekend"])["Rented Bike Count"]
        .mean()
        .reset_index()
    )
    hourly_avg["Day Type"] = hourly_avg["is_weekend"].map({0: "Weekday (Mon–Fri)", 1: "Weekend (Sat–Sun)"})

    fig_hour = px.line(
        hourly_avg,
        x="Hour",
        y="Rented Bike Count",
        color="Day Type",
        markers=True,
        title="Average Hourly Demand: Weekday Commute Peaks vs. Weekend Afternoon Curve",
        labels={"Rented Bike Count": "Average Rented Bikes", "Hour": "Hour of Day (0–23)"},
        color_discrete_map={"Weekday (Mon–Fri)": "#0284c7", "Weekend (Sat–Sun)": "#f59e0b"},
    )
    fig_hour.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1), hovermode="x unified")
    st.plotly_chart(fig_hour, use_container_width=True)

    # Chart 2: Weekly & Seasonal Distribution
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("2. Weekly Demand Distribution")
        weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        weekly_avg = filtered_df.groupby("day_of_week")["Rented Bike Count"].mean().reset_index()
        weekly_avg["Weekday"] = weekly_avg["day_of_week"].map(weekday_map)
        weekly_avg = weekly_avg.sort_values("day_of_week")

        fig_week = px.bar(
            weekly_avg,
            x="Weekday",
            y="Rented Bike Count",
            title="Average Demand by Day of Week",
            color="Rented Bike Count",
            color_continuous_scale="Teal",
            labels={"Rented Bike Count": "Average Rentals"},
        )
        fig_week.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_week, use_container_width=True)

    with c_right:
        st.subheader("3. Seasonal Demand Comparison")
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        seasonal_avg = (
            filtered_df.groupby("Seasons")["Rented Bike Count"]
            .mean()
            .reindex(season_order)
            .dropna()
            .reset_index()
        )

        fig_season = px.bar(
            seasonal_avg,
            x="Seasons",
            y="Rented Bike Count",
            title="Average Demand by Season",
            color="Seasons",
            color_discrete_map={
                "Winter": "#93c5fd",
                "Spring": "#86efac",
                "Summer": "#f87171",
                "Autumn": "#fbbf24",
            },
            labels={"Rented Bike Count": "Average Rentals"},
        )
        fig_season.update_layout(showlegend=False)
        st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("---")

    # Chart 3: Meteorological Relationships
    st.subheader("4. Meteorological Factors vs. Rental Demand")

    tab_temp, tab_hum, tab_rain, tab_solar = st.tabs(
        ["🌡️ Temperature", "💧 Humidity", "🌧️ Rainfall", "☀️ Solar Radiation"]
    )

    with tab_temp:
        # Sample for fast, crisp scatter visualization
        sample_size = min(2000, len(filtered_df))
        sample_scatter = filtered_df.sample(sample_size, random_state=42)

        try:
            fig_temp = px.scatter(
                sample_scatter,
                x="Temperature(°C)",
                y="Rented Bike Count",
                color="Seasons",
                opacity=0.45,
                trendline="lowess",
                title=f"Demand vs. Ambient Temperature (Sample of {sample_size:,} records)",
                labels={"Temperature(°C)": "Temperature (°C)", "Rented Bike Count": "Rented Bikes"},
            )
        except Exception:
            # Fallback if statsmodels is unavailable or fitting fails
            fig_temp = px.scatter(
                sample_scatter,
                x="Temperature(°C)",
                y="Rented Bike Count",
                color="Seasons",
                opacity=0.45,
                title=f"Demand vs. Ambient Temperature (Sample of {sample_size:,} records)",
                labels={"Temperature(°C)": "Temperature (°C)", "Rented Bike Count": "Rented Bikes"},
            )
            st.caption("ℹ️ Note: Trendline omitted because statsmodels is not available.")

        st.plotly_chart(fig_temp, use_container_width=True)
        st.caption("Demand increases sharply from sub-zero temperatures up to ~25°C, tapering slightly during heatwaves.")

    with tab_hum:
        fig_hum = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df)), random_state=42),
            x="Humidity(%)",
            y="Rented Bike Count",
            color="Seasons",
            opacity=0.45,
            title="Demand vs. Relative Humidity (%)",
            labels={"Humidity(%)": "Relative Humidity (%)", "Rented Bike Count": "Rented Bikes"},
        )
        st.plotly_chart(fig_hum, use_container_width=True)
        st.caption("High relative humidity (> 80%) significantly depresses rental demand.")

    with tab_rain:
        # Filter to rainfall > 0 for informative scatter
        rain_subset = filtered_df[filtered_df["Rainfall(mm)"] > 0]
        if not rain_subset.empty:
            fig_rain = px.scatter(
                rain_subset,
                x="Rainfall(mm)",
                y="Rented Bike Count",
                color="Seasons",
                opacity=0.6,
                title=f"Demand During Rain Events ({len(rain_subset):,} wet hours)",
                labels={"Rainfall(mm)": "Rainfall (mm)", "Rented Bike Count": "Rented Bikes"},
            )
            st.plotly_chart(fig_rain, use_container_width=True)
        else:
            st.info("No rainfall events in the currently filtered data subset.")
        st.caption("Even light rainfall (< 2 mm/hr) causes immediate 70-90% demand drops.")

    with tab_solar:
        fig_solar = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df)), random_state=42),
            x="Solar Radiation (MJ/m2)",
            y="Rented Bike Count",
            color="Seasons",
            opacity=0.45,
            title="Demand vs. Solar Radiation (MJ/m²)",
            labels={"Solar Radiation (MJ/m2)": "Solar Radiation (MJ/m²)", "Rented Bike Count": "Rented Bikes"},
        )
        st.plotly_chart(fig_solar, use_container_width=True)
        st.caption("Solar radiation correlates positively with daylight hours and recreational cycling.")

    st.markdown("---")

    # Interactive Data Explorer Preview
    with st.expander("📁 Explore Filtered Historical Data (Top 50 Rows)"):
        st.dataframe(
            filtered_df[
                [
                    "Date",
                    "Hour",
                    "Rented Bike Count",
                    "Temperature(°C)",
                    "Humidity(%)",
                    "Wind speed (m/s)",
                    "Rainfall(mm)",
                    "Seasons",
                    "Holiday",
                    "Functioning Day",
                ]
            ].head(50),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page 4: Model Performance & Phase 3B Audit
# ---------------------------------------------------------------------------


def render_performance():
    """Render Page 4: Model Performance & Phase 3B Audit Results."""
    st.markdown("# 📈 Model Performance & Evaluation Audit")
    st.markdown(
        "Detailed performance results from Phase 3B: candidate comparisons, "
        "pre-test model selection lock, final test evaluation, and diagnostic analysis."
    )

    metrics = get_verified_metrics()

    st.markdown("---")
    st.subheader("1. Verified Model Performance")

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Validation Performance</div>
                <div class="metric-card-value">RMSE: {metrics['validation']['rmse']}</div>
                <div class="metric-card-sub">MAE: {metrics['validation']['mae']} | R²: {metrics['validation']['r2']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Scenario A: Test Operating Hours</div>
                <div class="metric-card-value">RMSE: {metrics['test_scenario_a']['rmse']}</div>
                <div class="metric-card-sub">MAE: {metrics['test_scenario_a']['mae']} | R²: {metrics['test_scenario_a']['r2']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Scenario B: Full Test Timeline</div>
                <div class="metric-card-value">RMSE: {metrics['test_scenario_b']['rmse']}</div>
                <div class="metric-card-sub">MAE: {metrics['test_scenario_b']['mae']} | R²: {metrics['test_scenario_b']['r2']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("2. Baseline Comparison & Training Population Fairness")

    st.markdown(
        """
        The chart below illustrates model outperformance against heuristic baselines on the **untouched test partition**.
        - **Training-Only Baseline**: The Phase 3A hourly-mean baseline derived from `train.csv` ($N=6,084$).
        - **Combined-Data Baseline**: The fairer baseline recalculated using combined Train + Validation operating data ($N=7,247$), matching the champion model's refitted training population.
        """
    )

    comp_df = pd.DataFrame({
        "Model / Method": [
            "Baseline: Hour Mean (Train-Only, N=6,084)",
            "Baseline: Hour Mean (Combined, N=7,247)",
            "Champion: Gradient Boosting (Refitted, N=7,247)",
        ],
        "Test RMSE": [413.23, 388.16, 368.56],
        "Type": ["Heuristic Baseline", "Fair Population Baseline", "Machine Learning Champion"],
    })

    fig_comp = px.bar(
        comp_df,
        x="Model / Method",
        y="Test RMSE",
        color="Type",
        text="Test RMSE",
        title="Comparison of Test RMSE (Lower is Better)",
        color_discrete_map={
            "Heuristic Baseline": "#94a3b8",
            "Fair Population Baseline": "#64748b",
            "Machine Learning Champion": "#0d9488",
        },
    )
    fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_comp.update_layout(yaxis_range=[0, 480], showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Impurity-Based Feature Importance")

    try:
        pipeline = get_pipeline()
        df_imp = get_feature_importance_df(pipeline)

        fig_imp = px.bar(
            df_imp.head(12).sort_values("percentage", ascending=True),
            x="percentage",
            y="feature",
            orientation="h",
            title="Top 12 Features by Impurity Reduction (Fitted Champion Pipeline)",
            labels={"percentage": "Importance Percentage (%)", "feature": "Transformed Feature"},
            color="percentage",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.caption(
            "Note: Impurity-based feature importance reflects predictive contribution within decision tree splits, "
            "not causal mechanisms. Temperature and diurnal hour dominate predictive gain."
        )

    except Exception as e:
        st.error(f"Could not extract feature importances: {e}")

    st.markdown("---")
    st.subheader("4. Phase 3B Validation Diagnostic Gallery")

    diag_tabs = st.tabs([
        "1. Actual vs Predicted",
        "2. Residuals vs Predicted",
        "3. Residual Distribution",
        "4. Time Series Sample",
        "5. Model Family Comparison",
    ])

    project_root = Path(__file__).resolve().parent
    fig_dir = project_root / "reports" / "figures" / "phase3b"

    with diag_tabs[0]:
        img_p1 = fig_dir / "01_actual_vs_predicted_val.png"
        if img_p1.exists():
            st.image(str(img_p1), caption="Actual vs. Predicted Rentals on Validation Operating Hours")
            st.caption("Points cluster along the 1:1 diagonal line, showing accurate tracking across demand levels.")
        else:
            st.info("Diagnostic image 01 not found.")

    with diag_tabs[1]:
        img_p2 = fig_dir / "02_residuals_vs_predicted_val.png"
        if img_p2.exists():
            st.image(str(img_p2), caption="Residuals (y - y_pred) vs. Predicted Demand")
            st.caption("Residuals are centered along 0 with slight heteroscedastic spread at peak rental demand.")
        else:
            st.info("Diagnostic image 02 not found.")

    with diag_tabs[2]:
        img_p3 = fig_dir / "03_residual_distribution_val.png"
        if img_p3.exists():
            st.image(str(img_p3), caption="Residual Error Distribution")
            st.caption("Distribution is symmetric and near-Gaussian centered near zero (μ ≈ -2.1).")
        else:
            st.info("Diagnostic image 03 not found.")

    with diag_tabs[3]:
        img_p4 = fig_dir / "04_chronological_sample_val.png"
        if img_p4.exists():
            st.image(str(img_p4), caption="Two-Week Chronological Demand Tracking (Validation Partition)")
            st.caption("Model captures both morning and evening rush-hour spikes across consecutive days.")
        else:
            st.info("Diagnostic image 04 not found.")

    with diag_tabs[4]:
        img_p5 = fig_dir / "05_model_comparison_val_rmse.png"
        if img_p5.exists():
            st.image(str(img_p5), caption="Validation RMSE Progression Across Model Families")
            st.caption("Visual progression: Naive Mean (760.02) -> Hourly Mean (600.96) -> Linear (567.84) -> Trees (366.74) -> Forest (333.46) -> Gradient Boosting (321.58).")
        else:
            st.info("Diagnostic image 05 not found.")


# ---------------------------------------------------------------------------
# Page 5: About the Project
# ---------------------------------------------------------------------------


def render_about():
    """Render Page 5: About the Project."""
    st.markdown("# ℹ️ About the Project")
    st.markdown("### Vehicle Rental Demand Prediction Using Weather and Temporal Features")

    st.markdown(
        """
        ### Academic Project Context
        This application represents the culmination of an end-to-end Machine Learning coursework project:
        - **Subject**: Introduction to Machine Learning (IML), Semester 5 B.Tech.
        - **Dataset**: Seoul Bike Sharing Demand dataset (UCI Machine Learning Repository).
        - **Core Question**: Can machine learning regression pipelines effectively predict hourly bike rental demand using temporal and weather signals, and does non-linear tree ensembling outperform traditional diurnal heuristic baselines?

        ---

        ### Chronological Splitting Protocol
        To eliminate data leakage and mirror real-world forecasting conditions, data partitioning was executed 
        **strictly chronologically** (no random shuffling):
        - **Training Partition (70%)**: 6,132 observations (Dec 1, 2017 to Aug 13, 2018)
        - **Validation Partition (15%)**: 1,314 observations (Aug 13, 2018 to Oct 7, 2018)
        - **Test Partition (15%)**: 1,314 observations (Oct 7, 2018 to Nov 30, 2018) — *Quarantined until model selection was sealed.*

        ---

        ### Evaluated Feature Configurations
        Eight feature configurations were systematically tested across 120 candidate model runs:
        - **Set A (Temporal Only)**: Diurnal, weekly, and annual cyclical features (10 features).
        - **Set B (Weather Only)**: Continuous meteorological measurements (8 features).
        - **Set C (Combined)**: Complete temporal and weather feature set (18 features).
        - **Set D (Reduced Collinearity)**: Set C omitting `Dew point temperature(°C)` ($r = 0.9128$ collinearity with Temperature).
        - **Variants + Categoricals**: Sets A, B, C, D expanded with one-hot encoded `Seasons` and `Holiday`.

        **Key Scientific Finding:** For Linear Regression, omitting Dew Point stabilized coefficients and improved RMSE. 
        For Gradient Boosting, retaining Dew Point improved RMSE by capturing non-linear moisture/perceived-temperature interactions.

        ---

        ### Real-World Limitations & Disclaimers
        1. **Single-Year Dataset**: Spanning exactly one calendar year (Dec 2017 to Nov 2018), inter-annual seasonality cannot be learned without multi-year depth.
        2. **Observed vs. Forecast Weather**: This model was trained on historical observed weather. Operational deployment requires integration with weather forecast APIs, which introduce forecast uncertainty.
        3. **Supply Constraints**: Target counts reflect completed rentals, which are bounded by station inventory. Peak unmet demand may exceed observed rentals (censored demand).
        4. **Predictive vs. Causal**: Feature importances indicate statistical predictive associations, not causal policy levers.
        """
    )


# ---------------------------------------------------------------------------
# Main Application Controller
# ---------------------------------------------------------------------------


def main():
    """Application main entry point."""
    selected_page = render_sidebar()

    if selected_page == "🏠 Project Overview":
        render_overview()
    elif selected_page == "🔮 Predict Demand":
        render_predict()
    elif selected_page == "📊 Demand Analysis":
        render_analysis()
    elif selected_page == "📈 Model Performance":
        render_performance()
    elif selected_page == "ℹ️ About the Project":
        render_about()


if __name__ == "__main__":
    main()
