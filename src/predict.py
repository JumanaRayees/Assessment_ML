from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import create_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "Data" / "train-test_clean.csv"
VALIDATION_PATH = PROJECT_ROOT / "Data" / "validation.csv"
DECEMBER_PATH = PROJECT_ROOT / "Data" / "december-chart-inputs.csv"

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_OUTPUT = PROJECT_ROOT / "validation_predictions.csv"

TARGET = "posted_rate"


# ============================================================
# MODEL
# ============================================================

def build_model(X):

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop",
    )

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.8,
        min_samples_leaf=2,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    df = create_features(df)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL FREIGHT RATE MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_df = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"]
    )

    print(f"Training dataset: {train_df.shape}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nCreating training features...")

    train_df = prepare_features(train_df)

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[
            TARGET,
            "load_id"
        ]
    )

    y_train = train_df[TARGET]

    print(f"Training rows: {len(X_train):,}")
    print(f"Target mean: {y_train.mean():.2f}")

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    print("\nTraining final Extra Trees model...")

    model = build_model(X_train)

    model.fit(
        X_train,
        y_train
    )

    print("Final model trained.")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / "final_extra_trees.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to:\n  {model_path}")

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading 12,000 validation loads...")

    validation_df = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=["date"]
    )

    print(f"Validation dataset: {validation_df.shape}")

    validation_ids = validation_df["load_id"].copy()

    # --------------------------------------------------------
    # Validation features
    # --------------------------------------------------------

    X_validation = prepare_features(validation_df)

    X_validation = X_validation.drop(
        columns=["load_id"]
    )

    # --------------------------------------------------------
    # Predict validation
    # --------------------------------------------------------

    print("\nGenerating validation predictions...")

    predictions = model.predict(
        X_validation
    )

    predictions = np.maximum(
        predictions,
        0.01
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "load_id": validation_ids,
            "predicted_rate": predictions,
        }
    )

    prediction_df.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    print(
        f"\nPredictions saved to:\n  {VALIDATION_OUTPUT}"
    )

    # ========================================================
    # DECEMBER 2025 PREDICTIONS
    # ========================================================

    print("\nGenerating December predictions...")

    december_df = pd.read_csv(
        DECEMBER_PATH,
        parse_dates=["date"]
    )

    print(f"December dataset: {december_df.shape}")

    # Use EXACTLY the same feature preparation
    # as training and validation.
    X_december = prepare_features(
        december_df
    )

    # Remove load_id if present
    if "load_id" in X_december.columns:
        X_december = X_december.drop(
            columns=["load_id"]
        )

    # Remove target if present
    if TARGET in X_december.columns:
        X_december = X_december.drop(
            columns=[TARGET]
        )

    # Predict December rates
    december_predictions = model.predict(
        X_december
    )

    december_predictions = np.maximum(
        december_predictions,
        0.01
    )

    december_df["predicted_rate"] = (
        december_predictions
    )

    # Save December predictions
    december_df.to_csv(
        DECEMBER_PATH,
        index=False
    )

    print(
        f"December predictions saved to:\n  {DECEMBER_PATH}"
    )

    print("\nDecember prediction checks:")
    print(
        f"Rows: {len(december_df)}"
    )
    print(
        f"Missing predictions: "
        f"{december_df['predicted_rate'].isna().sum()}"
    )
    print(
        f"Minimum prediction: "
        f"{december_df['predicted_rate'].min():.2f}"
    )
    print(
        f"Maximum prediction: "
        f"{december_df['predicted_rate'].max():.2f}"
    )
    print(
        f"Mean prediction: "
        f"{december_df['predicted_rate'].mean():.2f}"
    )

    # ========================================================
    # VALIDATION PREDICTION CHECKS
    # ========================================================

    print("\nValidation prediction checks:")

    print(
        f"Rows: {len(prediction_df):,}"
    )

    print(
        f"Missing predictions: "
        f"{prediction_df['predicted_rate'].isna().sum()}"
    )

    print(
        f"Duplicate IDs: "
        f"{prediction_df['load_id'].duplicated().sum()}"
    )

    print(
        f"Minimum prediction: "
        f"{prediction_df['predicted_rate'].min():.2f}"
    )

    print(
        f"Maximum prediction: "
        f"{prediction_df['predicted_rate'].max():.2f}"
    )

    print(
        f"Mean prediction: "
        f"{prediction_df['predicted_rate'].mean():.2f}"
    )

    print("\nFinal prediction pipeline complete.")


if __name__ == "__main__":
    main()

