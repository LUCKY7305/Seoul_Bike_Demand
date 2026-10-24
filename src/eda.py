"""
Exploratory Data Analysis (EDA) Module for Vehicle Rental Demand Prediction
==========================================================================

Phase 2: Computes summary statistics, outlier detection, zero-demand analysis,
correlation matrix, and generates 14 high-quality publication-grade figures.

Usage:
    python -m src.eda --data "data/processed/seoul_bike_processed.csv" --output-dir "reports/figures"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Plot Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR = Path("reports/figures")
PROCESSED_DATA_PATH = Path("data/processed/seoul_bike_processed.csv")

# Color palette definition
PALETTE = sns.color_palette("tab10")
SEASONS_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def set_plot_style() -> None:
    """Set consistent visual styling for charts."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def find_temp_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Find the exact column names for ambient temperature and dew point."""
    temp_col = None
    dew_col = None
    for c in df.columns:
        if "Dew" in c:
            dew_col = c
        elif "Temperature" in c:
            temp_col = c
    if not temp_col:
        raise KeyError("Ambient temperature column not found.")
    if not dew_col:
        raise KeyError("Dew point temperature column not found.")
    return temp_col, dew_col


# ---------------------------------------------------------------------------
# Figure Generators
# ---------------------------------------------------------------------------


def plot_01_distribution_rented_bike_count(df: pd.DataFrame, out_dir: Path) -> Path:
    """1. Distribution of Rented Bike Count (histogram + KDE)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    target = df["Rented Bike Count"]
    mean_val = target.mean()
    median_val = target.median()

    sns.histplot(target, kde=True, color="#2b5c8f", bins=40, ax=ax, edgecolor="black", alpha=0.6)
    ax.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="#5cb85c", linestyle="-.", linewidth=1.8, label=f"Median: {median_val:.1f}")

    ax.set_title("Figure 1: Distribution of Hourly Rented Bike Count (Target Variable)", fontweight="bold")
    ax.set_xlabel("Rented Bike Count (bikes / hour)")
    ax.set_ylabel("Frequency (hours)")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "01_distribution_rented_bike_count.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_02_avg_demand_by_hour(df: pd.DataFrame, out_dir: Path) -> Path:
    """2. Average rental demand by hour (diurnal commute peaks)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    hourly_stats = df.groupby("Hour")["Rented Bike Count"].agg(["mean", "std", "count"])
    hours = hourly_stats.index
    means = hourly_stats["mean"]
    se = hourly_stats["std"] / np.sqrt(hourly_stats["count"])

    ax.plot(hours, means, marker="o", color="#1f77b4", linewidth=2.2, label="Mean Demand")
    ax.fill_between(hours, means - 1.96 * se, means + 1.96 * se, color="#1f77b4", alpha=0.2, label="95% CI")

    # Annotate peak commute hours
    ax.annotate("Morning Peak (08:00)", xy=(8, means[8]), xytext=(3, means[8] + 250),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6))
    ax.annotate("Evening Peak (18:00)", xy=(18, means[18]), xytext=(12, means[18] + 200),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6))

    ax.set_title("Figure 2: Average Rental Demand by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 23)")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_xticks(range(0, 24))
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "02_avg_demand_by_hour.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_03_avg_demand_by_day_of_week(df: pd.DataFrame, out_dir: Path) -> Path:
    """3. Average rental demand by day of week (weekday vs weekend)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    dow_means = df.groupby("day_of_week")["Rented Bike Count"].mean()
    colors = ["#337ab7" if i < 5 else "#e67e22" for i in range(7)]

    bars = ax.bar(DAYS_ABBR, dow_means, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 15, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_title("Figure 3: Average Rental Demand by Day of Week", fontweight="bold")
    ax.set_xlabel("Day of Week (Blue=Weekday, Orange=Weekend)")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_ylim(0, dow_means.max() * 1.15)
    fig.tight_layout()

    out_path = out_dir / "03_avg_demand_by_day_of_week.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_04_avg_demand_by_month(df: pd.DataFrame, out_dir: Path) -> Path:
    """4. Average rental demand by month."""
    fig, ax = plt.subplots(figsize=(9, 5))
    month_stats = df.groupby("month")["Rented Bike Count"].mean()
    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax.plot(month_stats.index, month_stats.values, marker="s", color="#27ae60", linewidth=2.2, markersize=7)
    for m, val in zip(month_stats.index, month_stats.values):
        ax.annotate(f"{val:.0f}", xy=(m, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)

    ax.set_title("Figure 4: Average Rental Demand by Month", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months_names)
    ax.set_ylim(0, month_stats.max() * 1.15)
    fig.tight_layout()

    out_path = out_dir / "04_avg_demand_by_month.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_05_avg_demand_by_season(df: pd.DataFrame, out_dir: Path) -> Path:
    """5. Average rental demand by season."""
    fig, ax = plt.subplots(figsize=(7, 5))
    season_means = df.groupby("Seasons")["Rented Bike Count"].mean().reindex(SEASONS_ORDER)
    season_colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]

    bars = ax.bar(SEASONS_ORDER, season_means, color=season_colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 20, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Figure 5: Average Rental Demand by Season", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_ylim(0, season_means.max() * 1.15)
    fig.tight_layout()

    out_path = out_dir / "05_avg_demand_by_season.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_06_demand_vs_temperature(df: pd.DataFrame, out_dir: Path) -> Path:
    """6. Rental demand versus ambient temperature."""
    fig, ax = plt.subplots(figsize=(8, 5))
    temp_col, _ = find_temp_cols(df)

    ax.scatter(df[temp_col], df["Rented Bike Count"], alpha=0.15, color="#2980b9", s=15, rasterized=True)
    # Overlay binned trend line without mutating df
    temp_bins = pd.cut(df[temp_col], bins=25)
    bin_means = df.groupby(temp_bins, observed=True)[[temp_col, "Rented Bike Count"]].mean()
    ax.plot(bin_means[temp_col], bin_means["Rented Bike Count"], color="#c0392b", linewidth=2.8, label="Binned Mean Trend")

    ax.set_title("Figure 6: Rental Demand vs. Ambient Temperature (°C)", fontweight="bold")
    ax.set_xlabel("Ambient Temperature (°C)")
    ax.set_ylabel("Rented Bike Count")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "06_demand_vs_temperature.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_07_demand_vs_humidity(df: pd.DataFrame, out_dir: Path) -> Path:
    """7. Rental demand versus relative humidity."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Humidity(%)"], df["Rented Bike Count"], alpha=0.15, color="#16a085", s=15, rasterized=True)

    # Overlay binned trend line without mutating df
    hum_bins = pd.cut(df["Humidity(%)"], bins=20)
    bin_means = df.groupby(hum_bins, observed=True)[["Humidity(%)", "Rented Bike Count"]].mean()
    ax.plot(bin_means["Humidity(%)"], bin_means["Rented Bike Count"], color="#d35400", linewidth=2.8, label="Binned Mean Trend")

    ax.set_title("Figure 7: Rental Demand vs. Relative Humidity (%)", fontweight="bold")
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Rented Bike Count")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "07_demand_vs_humidity.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_08_demand_vs_rainfall(df: pd.DataFrame, out_dir: Path) -> Path:
    """8. Rental demand versus rainfall (mm)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # All data
    ax1.scatter(df["Rainfall(mm)"], df["Rented Bike Count"], alpha=0.25, color="#34495e", s=18)
    ax1.set_title("Demand vs. Rainfall (Full Range)", fontweight="bold")
    ax1.set_xlabel("Rainfall (mm)")
    ax1.set_ylabel("Rented Bike Count")

    rain_cat = np.where(df["Rainfall(mm)"] == 0, "No Rain (93.97%)", "Rainy (>0mm, 6.03%)")
    sns.boxplot(x=rain_cat, y=df["Rented Bike Count"], hue=rain_cat, ax=ax2, palette=["#3498db", "#e74c3c"], legend=False)
    ax2.set_title("Demand: Dry vs. Rainy Hours", fontweight="bold")
    ax2.set_xlabel("Rainfall Condition")
    ax2.set_ylabel("Rented Bike Count")

    fig.suptitle("Figure 8: Impact of Rainfall on Vehicle Rental Demand", fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / "08_demand_vs_rainfall.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_09_demand_vs_snowfall(df: pd.DataFrame, out_dir: Path) -> Path:
    """9. Rental demand versus snowfall (cm)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # All data
    ax1.scatter(df["Snowfall (cm)"], df["Rented Bike Count"], alpha=0.25, color="#7f8c8d", s=18)
    ax1.set_title("Demand vs. Snowfall (Full Range)", fontweight="bold")
    ax1.set_xlabel("Snowfall (cm)")
    ax1.set_ylabel("Rented Bike Count")

    snow_cat = np.where(df["Snowfall (cm)"] == 0, "No Snow (94.94%)", "Snow (>0cm, 5.06%)")
    sns.boxplot(x=snow_cat, y=df["Rented Bike Count"], hue=snow_cat, ax=ax2, palette=["#bdc3c7", "#2980b9"], legend=False)
    ax2.set_title("Demand: Snow-Free vs. Snowy Hours", fontweight="bold")
    ax2.set_xlabel("Snowfall Condition")
    ax2.set_ylabel("Rented Bike Count")

    fig.suptitle("Figure 9: Impact of Snowfall on Vehicle Rental Demand", fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / "09_demand_vs_snowfall.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_10_boxplot_demand_by_season(df: pd.DataFrame, out_dir: Path) -> Path:
    """10. Box plot of rental demand by season."""
    fig, ax = plt.subplots(figsize=(8, 5))
    season_colors = {"Spring": "#2ecc71", "Summer": "#e74c3c", "Autumn": "#f39c12", "Winter": "#3498db"}

    sns.boxplot(
        data=df,
        x="Seasons",
        y="Rented Bike Count",
        hue="Seasons",
        order=SEASONS_ORDER,
        palette=season_colors,
        legend=False,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
    )

    ax.set_title("Figure 10: Rental Demand Distribution Across Seasons (Diamond = Mean)", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Rented Bike Count")
    fig.tight_layout()

    out_path = out_dir / "10_boxplot_demand_by_season.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_11_zero_demand_by_functioning_day(df: pd.DataFrame, out_dir: Path) -> Path:
    """11. Zero-demand observations by Functioning Day."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    zero_df = df[df["Rented Bike Count"] == 0]
    non_zero_df = df[df["Rented Bike Count"] > 0]

    # Bar chart of zero demand counts
    zero_counts = zero_df["Functioning Day"].value_counts().reindex(["No", "Yes"], fill_value=0)
    bars1 = ax1.bar(["No (Closed)", "Yes (Operating)"], zero_counts, color=["#c0392b", "#27ae60"], edgecolor="black")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 8, f"{h}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Zero-Demand Records (Count = 295)", fontweight="bold")
    ax1.set_ylabel("Number of Zero-Demand Hours")
    ax1.set_ylim(0, 340)

    # Stacked breakdown of all 8,760 records
    crosstab = pd.crosstab(df["Functioning Day"], df["Rented Bike Count"] == 0, normalize="index") * 100
    crosstab.columns = ["Demand > 0", "Demand = 0"]
    crosstab.reindex(["Yes", "No"]).plot(
        kind="bar",
        stacked=True,
        color=["#2980b9", "#e74c3c"],
        ax=ax2,
        edgecolor="black",
        rot=0,
    )
    ax2.set_title("Proportion of Zero Rentals by Status", fontweight="bold")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlabel("Functioning Day Status")
    ax2.legend(loc="upper right")

    fig.suptitle("Figure 11: Relationship Between Zero Rentals and System Functioning Status", fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / "11_zero_demand_by_functioning_day.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_12_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """12. Correlation heatmap for numerical variables."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select numerical features (excluding cyclical columns for clean interpretation, or including them)
    num_cols = [
        "Rented Bike Count",
        "Hour",
        "month",
        "day_of_week",
        "is_weekend",
    ]
    # Weather columns
    temp_col, dew_col = find_temp_cols(df)
    weather_cols = [
        temp_col,
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        dew_col,
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]
    all_num = num_cols + weather_cols
    corr_matrix = df[all_num].corr()

    # Create readable column labels for plot
    clean_labels = [c.replace(" (m/s)", "").replace(" (10m)", "").replace(" (MJ/m2)", "") for c in all_num]

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
        linewidths=0.5,
    )

    ax.set_title("Figure 12: Pearson Correlation Heatmap for Numerical Features", fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    out_path = out_dir / "12_correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_13_hourly_demand_by_season(df: pd.DataFrame, out_dir: Path) -> Path:
    """13. Hourly demand patterns separated by season."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    season_colors = {"Spring": "#2ecc71", "Summer": "#e74c3c", "Autumn": "#f39c12", "Winter": "#3498db"}

    for season in SEASONS_ORDER:
        sub = df[df["Seasons"] == season]
        hourly_season = sub.groupby("Hour")["Rented Bike Count"].mean()
        ax.plot(
            hourly_season.index,
            hourly_season.values,
            marker="o",
            label=season,
            color=season_colors[season],
            linewidth=2.2,
            markersize=5,
        )

    ax.set_title("Figure 13: Diurnal Hourly Demand Trajectories Across Seasons", fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 23)")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_xticks(range(0, 24))
    ax.legend(title="Season", frameon=True)
    fig.tight_layout()

    out_path = out_dir / "13_hourly_demand_by_season.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_14_demand_vs_solar_radiation(df: pd.DataFrame, out_dir: Path) -> Path:
    """14. Rental demand versus solar radiation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Solar Radiation (MJ/m2)"], df["Rented Bike Count"], alpha=0.15, color="#f39c12", s=15, rasterized=True)

    # Binned trend line without mutating df
    solar_bins = pd.cut(df["Solar Radiation (MJ/m2)"], bins=20)
    bin_means = df.groupby(solar_bins, observed=True)[["Solar Radiation (MJ/m2)", "Rented Bike Count"]].mean()
    ax.plot(bin_means["Solar Radiation (MJ/m2)"], bin_means["Rented Bike Count"], color="#8e44ad", linewidth=2.8, label="Binned Mean Trend")

    ax.set_title("Figure 14: Rental Demand vs. Solar Radiation (MJ/m²)", fontweight="bold")
    ax.set_xlabel("Solar Radiation (MJ/m²)")
    ax.set_ylabel("Rented Bike Count")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "14_demand_vs_solar_radiation.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Numerical Statistics & Outlier Computation
# ---------------------------------------------------------------------------


def compute_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed descriptive statistics including skewness and IQR bounds."""
    temp_col, dew_col = find_temp_cols(df)
    target_and_weather = [
        "Rented Bike Count",
        "Hour",
        temp_col,
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        dew_col,
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]

    stats_list = []
    for col in target_and_weather:
        s = df[col]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_count = int(((s < lower) | (s > upper)).sum())

        stats_list.append({
            "Column": col,
            "Count": len(s),
            "Mean": round(float(s.mean()), 2),
            "Std": round(float(s.std()), 2),
            "Min": round(float(s.min()), 2),
            "Q1": round(q1, 2),
            "Median": round(float(s.median()), 2),
            "Q3": round(q3, 2),
            "Max": round(float(s.max()), 2),
            "IQR": round(iqr, 2),
            "Lower_Bound": round(lower, 2),
            "Upper_Bound": round(upper, 2),
            "Outlier_Count": outliers_count,
            "Outlier_Pct": round(outliers_count / len(s) * 100, 2),
            "Skewness": round(float(s.skew()), 2),
        })

    return pd.DataFrame(stats_list)


def compute_correlations(df: pd.DataFrame) -> pd.Series:
    """Compute Pearson correlations of all numerical features with the target."""
    num_cols = df.select_dtypes(include="number").columns
    corr_series = df[num_cols].corr()["Rented Bike Count"].sort_values(ascending=False)
    return corr_series


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eda(data_path: str = str(PROCESSED_DATA_PATH), out_dir: str = str(FIGURES_DIR)) -> Dict:
    """
    Run full EDA: compute statistics, outlier analysis, correlations,
    and generate all 14 figures.
    """
    set_plot_style()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading processed data from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"       Loaded {len(df)} rows, {len(df.columns)} cols")

    # Generate 14 figures
    print("[INFO] Generating 14 EDA figures...")
    plot_funcs = [
        ("01_distribution_rented_bike_count", plot_01_distribution_rented_bike_count),
        ("02_avg_demand_by_hour", plot_02_avg_demand_by_hour),
        ("03_avg_demand_by_day_of_week", plot_03_avg_demand_by_day_of_week),
        ("04_avg_demand_by_month", plot_04_avg_demand_by_month),
        ("05_avg_demand_by_season", plot_05_avg_demand_by_season),
        ("06_demand_vs_temperature", plot_06_demand_vs_temperature),
        ("07_demand_vs_humidity", plot_07_demand_vs_humidity),
        ("08_demand_vs_rainfall", plot_08_demand_vs_rainfall),
        ("09_demand_vs_snowfall", plot_09_demand_vs_snowfall),
        ("10_boxplot_demand_by_season", plot_10_boxplot_demand_by_season),
        ("11_zero_demand_by_functioning_day", plot_11_zero_demand_by_functioning_day),
        ("12_correlation_heatmap", plot_12_correlation_heatmap),
        ("13_hourly_demand_by_season", plot_13_hourly_demand_by_season),
        ("14_demand_vs_solar_radiation", plot_14_demand_vs_solar_radiation),
    ]

    generated_paths = []
    for name, func in plot_funcs:
        p = func(df, output_dir)
        generated_paths.append(str(p))
        print(f"       [OK] Generated {p.name}")

    # Summary statistics
    print("[INFO] Computing numerical summary and outlier metrics...")
    summary_df = compute_numerical_summary(df)

    # Correlation analysis
    print("[INFO] Computing Pearson correlation coefficients...")
    correlations = compute_correlations(df)

    results = {
        "generated_figures": generated_paths,
        "summary_statistics": summary_df.to_dict(orient="records"),
        "correlations_with_target": correlations.to_dict(),
    }

    print("[INFO] EDA successfully generated all figures and statistics.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for vehicle rental demand prediction.")
    parser.add_argument("--data", default=str(PROCESSED_DATA_PATH), help="Path to processed dataset CSV")
    parser.add_argument("--output-dir", default=str(FIGURES_DIR), help="Directory to save figures")
    args = parser.parse_args()

    try:
        run_eda(args.data, args.output_dir)
    except Exception as e:
        print(f"[ERROR] EDA failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
