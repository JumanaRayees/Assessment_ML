from pathlib import Path

import numpy as np
import pandas as pd



# Paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "Data" / "train-test_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "features_ready.csv"



# Feature Engineering Function


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering logic to any dataset.

    Important:
    - Does NOT use posted_rate to create predictors.
    - Missing values are intentionally preserved.
    - Imputation will happen later inside the training pipeline.
    """

    data = df.copy()

    
    # Date features
  

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(float)
    data["day_of_year"] = data["date"].dt.dayofyear

    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_month_start"] = data["date"].dt.is_month_start.astype(int)
    data["is_month_end"] = data["date"].dt.is_month_end.astype(int)

  
    # Route features

    data["route"] = (
        data["pickup"].astype(str)
        + " -> "
        + data["delivery"].astype(str)
    )

   
    # Geographic features
    

    data["lat_diff"] = (
        data["delivery_lat"] - data["pickup_lat"]
    )

    data["lon_diff"] = (
        data["delivery_lon"] - data["pickup_lon"]
    )

    data["abs_lat_diff"] = data["lat_diff"].abs()
    data["abs_lon_diff"] = data["lon_diff"].abs()

    # Haversine distance
    lat1 = np.radians(data["pickup_lat"])
    lon1 = np.radians(data["pickup_lon"])
    lat2 = np.radians(data["delivery_lat"])
    lon2 = np.radians(data["delivery_lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    earth_radius = 3958.8  # miles

    data["geo_distance"] = (
        2
        * earth_radius
        * np.arcsin(np.sqrt(a))
    )

    # Distance features


    data["log_distance"] = np.log1p(
        data["distance"].clip(lower=0)
    )

    data["distance_bucket"] = pd.cut(
        data["distance"],
        bins=[-np.inf, 250, 500, 1000, 1500, 2500, np.inf],
        labels=[
            "very_short",
            "short",
            "medium",
            "long",
            "very_long",
            "extreme",
        ],
    ).astype("object")

    
    # Weight features


    data["log_weight"] = np.log1p(
        data["weight"].clip(lower=0)
    )

    data["weight_bucket"] = pd.cut(
        data["weight"],
        bins=[-np.inf, 10000, 20000, 30000, 40000, np.inf],
        labels=[
            "very_light",
            "light",
            "medium",
            "heavy",
            "very_heavy",
        ],
    ).astype("object")

    # Weight relative to distance
    data["weight_per_distance"] = (
        data["weight"] / data["distance"].replace(0, np.nan)
    )

    
    # Market features
  

    data["market_index_deviation"] = (
        data["market_index"] - 1.0
    )

    # Quote signal features
   
    data["quote_signal_deviation"] = (
        data["quote_signal"] - 2.0
    )

    data["quote_signal_squared"] = (
        data["quote_signal"] ** 2
    )

    # Interaction features


    data["distance_x_weight"] = (
        data["distance"] * data["weight"]
    )

    data["distance_x_market"] = (
        data["distance"] * data["market_index"]
    )

    data["distance_x_quote"] = (
        data["distance"] * data["quote_signal"]
    )

 
    # Clean infinite values


    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


# Main


def main():
    print("Loading cleaned dataset...")

    df = pd.read_csv(TRAIN_PATH)

    print(f"Original shape: {df.shape}")

    df_features = create_features(df)

    print(f"Feature-engineered shape: {df_features.shape}")

    print("\nMissing values:")
    missing = df_features.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values.")

    # Leakage check
    print("\nLeakage check:")

    if "posted_rate" in df_features.columns:
        print(
            "posted_rate exists as TARGET and must NOT be used as a feature."
        )

    if "rate_per_mile" in df_features.columns:
        print(
            "WARNING: rate_per_mile exists and must NOT be used as a feature."
        )
    else:
        print("OK: rate_per_mile is not included.")

    # Save
    df_features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()