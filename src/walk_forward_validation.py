
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import create_features


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "train-test_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

TARGET = "posted_rate"


# ============================================================
# Model
# ============================================================

def create_model(X_train):

    categorical_features = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numeric_features = (
        X_train
        .select_dtypes(
            include=[
                "number",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Combined preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------------
    # Extra Trees
    # --------------------------------------------------------

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# Evaluation
# ============================================================

def evaluate_fold(
    train_df,
    validation_df,
    fold_name,
):

    print("\n" + "=" * 70)
    print(f"FOLD: {fold_name}")
    print("=" * 70)

    print(
        f"Training period: "
        f"{train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{validation_df['date'].min().date()} → "
        f"{validation_df['date'].max().date()}"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[
            TARGET,
            "load_id",
        ]
    )

    y_train = train_df[TARGET]

    X_valid = validation_df.drop(
        columns=[
            TARGET,
            "load_id",
        ]
    )

    y_valid = validation_df[TARGET]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    pipeline = create_model(
        X_train
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Extra Trees...")

    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_valid
    )

    # Freight rates must be positive.
    predictions = np.maximum(
        predictions,
        0.01,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_valid,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions,
        )
    )

    r2 = r2_score(
        y_valid,
        predictions,
    )

    print("\nResults:")

    print(
        f"MAE:  {mae:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    return {
        "fold": fold_name,
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "validation_start": validation_df["date"].min(),
        "validation_end": validation_df["date"].max(),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FREIGHT RATE WALK-FORWARD VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nCreating features...")

    df = create_features(df)

    df = df.dropna(
        subset=["date"]
    ).copy()

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Define folds
    # --------------------------------------------------------

    folds = [
        (
            "June → July",
            pd.Timestamp("2025-07-01"),
            pd.Timestamp("2025-08-01"),
        ),
        (
            "July → August",
            pd.Timestamp("2025-08-01"),
            pd.Timestamp("2025-09-01"),
        ),
        (
            "August → September",
            pd.Timestamp("2025-09-01"),
            pd.Timestamp("2025-10-01"),
        ),
        (
            "September → October",
            pd.Timestamp("2025-10-01"),
            pd.Timestamp("2025-11-01"),
        ),
    ]

    results = []

    # --------------------------------------------------------
    # Run folds
    # --------------------------------------------------------

    for fold_name, validation_start, validation_end in folds:

        train_df = df[
            df["date"] < validation_start
        ].copy()

        validation_df = df[
            (
                df["date"] >= validation_start
            )
            &
            (
                df["date"] < validation_end
            )
        ].copy()

        result = evaluate_fold(
            train_df,
            validation_df,
            fold_name,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 70)

    display_columns = [
        "fold",
        "train_rows",
        "validation_rows",
        "MAE",
        "RMSE",
        "R2",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    
    # Average metrics
    

    print("\nAverage performance:")

    print(
        f"Mean MAE:  "
        f"{results_df['MAE'].mean():,.2f}"
    )

    print(
        f"Mean RMSE: "
        f"{results_df['RMSE'].mean():,.2f}"
    )

    print(
        f"Mean R²:   "
        f"{results_df['R2'].mean():.4f}"
    )

    print("\nMetric stability:")

    print(
        f"MAE std:   "
        f"{results_df['MAE'].std():,.2f}"
    )

    print(
        f"RMSE std:  "
        f"{results_df['RMSE'].std():,.2f}"
    )

    
    # Save
   

    output_path = (
        OUTPUT_DIR
        / "walk_forward_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to:"
        f"\n  {output_path}"
    )

    print("\nWalk-forward validation complete.")


if __name__ == "__main__":
    main()

