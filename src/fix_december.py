
from pathlib import Path
import pandas as pd
import joblib

from feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "final_extra_trees.joblib"
DECEMBER_PATH = PROJECT_ROOT / "Data" / "december-chart-inputs.csv"


# Coordinates for the required December scenario
LOCATION_COORDS = {
    "Lexington": (37.9887, -84.4777),
    "Fort Wayne": (41.0793, -85.1394),
}


print("Loading saved final model...")
model = joblib.load(MODEL_PATH)

print("Loading December data...")
df = pd.read_csv(DECEMBER_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# ADD REQUIRED LOCATION COORDINATES
# ============================================================

df["pickup_lat"] = df["pickup"].map(
    lambda x: LOCATION_COORDS[x][0]
)

df["pickup_lon"] = df["pickup"].map(
    lambda x: LOCATION_COORDS[x][1]
)

df["delivery_lat"] = df["delivery"].map(
    lambda x: LOCATION_COORDS[x][0]
)

df["delivery_lon"] = df["delivery"].map(
    lambda x: LOCATION_COORDS[x][1]
)


# ============================================================
# ADD MISSING MODEL INPUTS
# ============================================================

# December template does not provide these variables.
# Use neutral values so the model can generate the
# required scenario predictions.

df["market_index"] = 1.0
df["quote_signal"] = 0.0


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("Creating December features...")

X = create_features(
    df.copy()
)


# Remove columns that are not model inputs
if "load_id" in X.columns:
    X = X.drop(
        columns=["load_id"]
    )

if "posted_rate" in X.columns:
    X = X.drop(
        columns=["posted_rate"]
    )


# ============================================================
# PREDICTION
# ============================================================

print("Generating December predictions...")

predictions = model.predict(X)

predictions = predictions.clip(
    min=0.01
)

df["predicted_rate"] = predictions


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    DECEMBER_PATH,
    index=False
)


print("\nDecember predictions successfully generated!")
print(f"Rows: {len(df)}")
print(
    f"Missing predictions: "
    f"{df['predicted_rate'].isna().sum()}"
)
print(
    f"Minimum prediction: "
    f"{df['predicted_rate'].min():.2f}"
)
print(
    f"Maximum prediction: "
    f"{df['predicted_rate'].max():.2f}"
)
print(
    f"Mean prediction: "
    f"{df['predicted_rate'].mean():.2f}"
)

print(
    f"\nSaved to:\n{DECEMBER_PATH}"
)

