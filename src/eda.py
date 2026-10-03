# FREIGHT RATE PREDICTION - EXPLORATORY DATA ANALYSIS (EDA)
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PATHS & SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "Data" / "train-test_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Create outputs/figures automatically
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


def save_plot(filename):
    """
    Save the current matplotlib figure inside outputs/figures.
    """
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {path}")
    plt.show()
    plt.close()


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print(f"Data path: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    print(missing)
else:
    print("No missing values.")


print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print("Duplicate rows:", df.duplicated().sum())

if "load_id" in df.columns:
    print("Duplicate load_id:", df["load_id"].duplicated().sum())


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print("\nNumerical statistics:")
print(df.describe())

print("\nCategorical statistics:")
print(df.describe(include=["object"]))


# ============================================================
# 5. TARGET ANALYSIS
# ============================================================

TARGET = "posted_rate"

print("\n" + "=" * 70)
print("TARGET ANALYSIS")
print("=" * 70)

print("\nTarget statistics:")
print(df[TARGET].describe())

print("\nTarget skewness:")
print(df[TARGET].skew())

print("\nTarget kurtosis:")
print(df[TARGET].kurtosis())


# Target distribution
plt.figure()
sns.histplot(
    df[TARGET],
    bins=60,
    kde=True
)

plt.title("Distribution of Posted Rate")
plt.xlabel("Posted Rate")
plt.ylabel("Frequency")

save_plot("01_target_distribution.png")


# Log-transformed target distribution
plt.figure()

sns.histplot(
    np.log1p(df[TARGET]),
    bins=60,
    kde=True
)

plt.title("Log-Transformed Distribution of Posted Rate")
plt.xlabel("log1p(Posted Rate)")
plt.ylabel("Frequency")

save_plot("02_log_target_distribution.png")


# Target boxplot
plt.figure()

sns.boxplot(
    x=df[TARGET]
)

plt.title("Posted Rate Boxplot")
plt.xlabel("Posted Rate")

save_plot("03_target_boxplot.png")


# ============================================================
# 6. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

numeric_features = [
    "distance",
    "weight",
    "market_index",
    "quote_signal"
]

for i, col in enumerate(numeric_features, start=4):

    if col not in df.columns:
        continue

    plt.figure()

    sns.histplot(
        df[col].dropna(),
        bins=50,
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    save_plot(f"{i:02d}_{col}_distribution.png")


# ============================================================
# 7. NUMERICAL FEATURES VS TARGET
# ============================================================

for i, col in enumerate(numeric_features, start=8):

    if col not in df.columns:
        continue

    plt.figure()

    sns.scatterplot(
        data=df,
        x=col,
        y=TARGET,
        alpha=0.25,
        s=20
    )

    plt.title(f"{col} vs Posted Rate")
    plt.xlabel(col)
    plt.ylabel("Posted Rate")

    save_plot(f"{i:02d}_{col}_vs_target.png")


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

print("\nCorrelation with posted_rate:")

target_corr = (
    correlation[TARGET]
    .sort_values(ascending=False)
)

print(target_corr)


# Full correlation heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Numerical Feature Correlation Matrix")

save_plot("12_correlation_heatmap.png")


# Correlation with target only
target_corr_plot = (
    correlation[TARGET]
    .drop(TARGET)
    .sort_values()
)

plt.figure(figsize=(10, 6))

target_corr_plot.plot(
    kind="barh"
)

plt.title("Correlation with Posted Rate")
plt.xlabel("Correlation")
plt.ylabel("Feature")

save_plot("13_target_correlation.png")


# ============================================================
# 9. EQUIPMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EQUIPMENT ANALYSIS")
print("=" * 70)

print("\nEquipment counts:")
print(df["equipment"].value_counts())

print("\nAverage posted rate by equipment:")
print(
    df.groupby("equipment")[TARGET]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
)


# Count plot
plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="equipment"
)

plt.title("Load Count by Equipment")
plt.xlabel("Equipment")
plt.ylabel("Number of Loads")

save_plot("14_equipment_counts.png")


# Equipment vs target
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="equipment",
    y=TARGET
)

plt.title("Posted Rate by Equipment")
plt.xlabel("Equipment")
plt.ylabel("Posted Rate")

save_plot("15_equipment_vs_target.png")


# ============================================================
# 10. PICKUP & DELIVERY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PICKUP / DELIVERY ANALYSIS")
print("=" * 70)

print("\nTop pickup locations:")
print(df["pickup"].value_counts().head(15))

print("\nTop delivery locations:")
print(df["delivery"].value_counts().head(15))


# Top pickup locations
top_pickup = df["pickup"].value_counts().head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    x=top_pickup.values,
    y=top_pickup.index
)

plt.title("Top 15 Pickup Locations")
plt.xlabel("Number of Loads")
plt.ylabel("Pickup Location")

save_plot("16_top_pickup_locations.png")


# Top delivery locations
top_delivery = df["delivery"].value_counts().head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    x=top_delivery.values,
    y=top_delivery.index
)

plt.title("Top 15 Delivery Locations")
plt.xlabel("Number of Loads")
plt.ylabel("Delivery Location")

save_plot("17_top_delivery_locations.png")


# ============================================================
# 11. DISTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DISTANCE ANALYSIS")
print("=" * 70)

print(df["distance"].describe())

print("\nDistance percentiles:")

print(
    df["distance"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)


# Distance histogram
plt.figure()

sns.histplot(
    df["distance"],
    bins=60,
    kde=True
)

plt.title("Distribution of Distance")
plt.xlabel("Distance")
plt.ylabel("Frequency")

save_plot("18_distance_distribution.png")


# Distance vs target
plt.figure()

sns.scatterplot(
    data=df,
    x="distance",
    y=TARGET,
    alpha=0.25,
    s=20
)

plt.title("Distance vs Posted Rate")
plt.xlabel("Distance")
plt.ylabel("Posted Rate")

save_plot("19_distance_vs_target.png")


# ============================================================
# 12. WEIGHT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WEIGHT ANALYSIS")
print("=" * 70)

print(df["weight"].describe())

print("\nMissing weight:")
print(df["weight"].isna().sum())


# Weight distribution
plt.figure()

sns.histplot(
    df["weight"].dropna(),
    bins=50,
    kde=True
)

plt.title("Distribution of Weight")
plt.xlabel("Weight")
plt.ylabel("Frequency")

save_plot("20_weight_distribution.png")


# Weight vs target
plt.figure()

sns.scatterplot(
    data=df,
    x="weight",
    y=TARGET,
    alpha=0.25,
    s=20
)

plt.title("Weight vs Posted Rate")
plt.xlabel("Weight")
plt.ylabel("Posted Rate")

save_plot("21_weight_vs_target.png")


# ============================================================
# 13. MARKET INDEX ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MARKET INDEX ANALYSIS")
print("=" * 70)

print(df["market_index"].describe())

print("\nMissing market_index:")
print(df["market_index"].isna().sum())


plt.figure()

sns.histplot(
    df["market_index"].dropna(),
    bins=50,
    kde=True
)

plt.title("Distribution of Market Index")
plt.xlabel("Market Index")
plt.ylabel("Frequency")

save_plot("22_market_index_distribution.png")


plt.figure()

sns.scatterplot(
    data=df,
    x="market_index",
    y=TARGET,
    alpha=0.25,
    s=20
)

plt.title("Market Index vs Posted Rate")
plt.xlabel("Market Index")
plt.ylabel("Posted Rate")

save_plot("23_market_index_vs_target.png")


# ============================================================
# 14. QUOTE SIGNAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("QUOTE SIGNAL ANALYSIS")
print("=" * 70)

print(df["quote_signal"].describe())


plt.figure()

sns.histplot(
    df["quote_signal"],
    bins=50,
    kde=True
)

plt.title("Distribution of Quote Signal")
plt.xlabel("Quote Signal")
plt.ylabel("Frequency")

save_plot("24_quote_signal_distribution.png")


plt.figure()

sns.scatterplot(
    data=df,
    x="quote_signal",
    y=TARGET,
    alpha=0.25,
    s=20
)

plt.title("Quote Signal vs Posted Rate")
plt.xlabel("Quote Signal")
plt.ylabel("Posted Rate")

save_plot("25_quote_signal_vs_target.png")


# ============================================================
# 15. DATE / TIME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DATE / TIME ANALYSIS")
print("=" * 70)

print("Minimum date:", df["date"].min())
print("Maximum date:", df["date"].max())


# Create temporary date features ONLY for EDA
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")
df["day_of_week"] = df["date"].dt.dayofweek
df["day_name"] = df["date"].dt.strftime("%A")


# Monthly load counts
monthly_counts = (
    df.groupby(df["date"].dt.to_period("M"))
    .size()
)

print("\nLoads by month:")
print(monthly_counts)


plt.figure(figsize=(11, 6))

monthly_counts.index = monthly_counts.index.astype(str)

sns.lineplot(
    x=monthly_counts.index,
    y=monthly_counts.values,
    marker="o"
)

plt.title("Monthly Load Volume")
plt.xlabel("Month")
plt.ylabel("Number of Loads")
plt.xticks(rotation=45)

save_plot("26_monthly_load_volume.png")


# Monthly average posted rate
monthly_rate = (
    df.groupby(df["date"].dt.to_period("M"))[TARGET]
    .mean()
)

monthly_rate.index = monthly_rate.index.astype(str)

plt.figure(figsize=(11, 6))

sns.lineplot(
    x=monthly_rate.index,
    y=monthly_rate.values,
    marker="o"
)

plt.title("Monthly Average Posted Rate")
plt.xlabel("Month")
plt.ylabel("Average Posted Rate")
plt.xticks(rotation=45)

save_plot("27_monthly_average_rate.png")


# Day of week
dow_counts = (
    df["day_name"]
    .value_counts()
    .reindex(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=dow_counts.index,
    y=dow_counts.values
)

plt.title("Load Volume by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Loads")
plt.xticks(rotation=30)

save_plot("28_day_of_week_volume.png")


# Posted rate by day of week
dow_rate = (
    df.groupby("day_name")[TARGET]
    .mean()
    .reindex(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=dow_rate.index,
    y=dow_rate.values
)

plt.title("Average Posted Rate by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Posted Rate")
plt.xticks(rotation=30)

save_plot("29_day_of_week_average_rate.png")


# ============================================================
# 16. OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)


def outlier_summary(column):
    """
    Calculate IQR-based outlier information.
    """
    data = df[column].dropna()

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[
        (data < lower) |
        (data > upper)
    ]

    print(f"\n{column}")
    print(f"Q1: {Q1:.4f}")
    print(f"Q3: {Q3:.4f}")
    print(f"IQR: {IQR:.4f}")
    print(f"Lower bound: {lower:.4f}")
    print(f"Upper bound: {upper:.4f}")
    print(f"Outliers: {len(outliers)}")
    print(f"Outlier percentage: {len(outliers) / len(data) * 100:.2f}%")


for col in [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    TARGET
]:

    outlier_summary(col)


# ============================================================
# 17. TARGET OUTLIERS
# ============================================================

Q1 = df[TARGET].quantile(0.25)
Q3 = df[TARGET].quantile(0.75)

IQR = Q3 - Q1

upper_target = Q3 + 1.5 * IQR

target_outliers = df[df[TARGET] > upper_target]

print("\n" + "=" * 70)
print("TARGET OUTLIERS")
print("=" * 70)

print("Target upper IQR bound:", upper_target)
print("Number of target outliers:", len(target_outliers))

print("\nTop 10 highest posted rates:")
print(
    df[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "weight",
            "equipment",
            "market_index",
            "quote_signal",
            TARGET
        ]
    ]
    .sort_values(TARGET, ascending=False)
    .head(10)
)


# ============================================================
# 18. ROUTE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ROUTE ANALYSIS")
print("=" * 70)

df["route"] = (
    df["pickup"].astype(str)
    + " -> "
    + df["delivery"].astype(str)
)

route_stats = (
    df.groupby("route")[TARGET]
    .agg(
        load_count="count",
        mean_rate="mean",
        median_rate="median"
    )
    .sort_values("load_count", ascending=False)
)

print("\nTop 20 routes by load count:")
print(route_stats.head(20))


top_routes = route_stats.head(15)

plt.figure(figsize=(11, 8))

sns.barplot(
    data=top_routes.reset_index(),
    x="load_count",
    y="route"
)

plt.title("Top 15 Routes by Load Volume")
plt.xlabel("Number of Loads")
plt.ylabel("Route")

save_plot("30_top_routes.png")


# ============================================================
# 19. ROUTE RATE ANALYSIS
# ============================================================

top_rate_routes = (
    df.groupby("route")
    .agg(
        load_count=(TARGET, "count"),
        mean_rate=(TARGET, "mean")
    )
)

# Only consider routes with at least 20 loads
top_rate_routes = (
    top_rate_routes[
        top_rate_routes["load_count"] >= 20
    ]
    .sort_values("mean_rate", ascending=False)
    .head(15)
)

print("\nTop routes by average posted rate")
print("(minimum 20 loads):")
print(top_rate_routes)


plt.figure(figsize=(11, 8))

sns.barplot(
    data=top_rate_routes.reset_index(),
    x="mean_rate",
    y="route"
)

plt.title("Routes with Highest Average Posted Rate")
plt.xlabel("Average Posted Rate")
plt.ylabel("Route")

save_plot("31_top_routes_by_average_rate.png")


# ============================================================
# 20. RATE PER MILE — EDA ONLY
# ============================================================
# IMPORTANT:
# This feature uses the target (posted_rate), so it MUST NOT
# be used as a model feature.

df["rate_per_mile"] = (
    df[TARGET] / df["distance"]
)

print("\n" + "=" * 70)
print("RATE PER MILE — EDA ONLY")
print("=" * 70)

print(df["rate_per_mile"].describe())


plt.figure()

sns.histplot(
    df["rate_per_mile"],
    bins=60,
    kde=True
)

plt.title("Distribution of Rate per Mile")
plt.xlabel("Posted Rate / Distance")
plt.ylabel("Frequency")

save_plot("32_rate_per_mile_distribution.png")


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(f"Dataset shape: {df.shape}")

print(f"\nCharts saved to:")
print(FIGURES_DIR)

print("\nTotal chart files:")

figure_files = sorted(FIGURES_DIR.glob("*.png"))

for file in figure_files:
    print(" -", file.name)

print(f"\nTotal charts saved: {len(figure_files)}")

print("\nEDA finished successfully.")

