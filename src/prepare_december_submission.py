from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DECEMBER_PATH = PROJECT_ROOT / "Data" / "december-chart-inputs.csv"

REQUIRED_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

df = pd.read_csv(DECEMBER_PATH)

# Keep ONLY the original seven columns, in the required order
df = df[REQUIRED_COLUMNS]

# Save the submission-ready December file
df.to_csv(
    DECEMBER_PATH,
    index=False
)

print("December submission file prepared successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Missing predictions: {df['predicted_rate'].isna().sum()}")
print(f"Minimum prediction: {df['predicted_rate'].min():.2f}")
print(f"Maximum prediction: {df['predicted_rate'].max():.2f}")
print(f"Mean prediction: {df['predicted_rate'].mean():.2f}")