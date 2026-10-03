from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import create_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "Data" / "train-test_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "posted_rate"


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    {
        "name": "baseline",
        "n_estimators": 300,
        "max_features": 0.8,
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    {
        "name": "more_trees",
        "n_estimators": 500,
        "max_features": 0.8,
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    {
        "name": "leaf_1",
        "n_estimators": 300,
        "max_features": 0.8,
        "min_samples_leaf": 1,
        "max_depth": None,
    },
    {
        "name": "leaf_4",
        "n_estimators": 300,
        "max_features": 0.8,
        "min_samples_leaf": 4,
        "max_depth": None,
    },
    {
        "name": "features_06",
        "n_estimators": 300,
        "max_features": 0.6,
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    {
        "name": "features_1",
        "n_estimators": 300,
        "max_features": 1.0,
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    {
        "name": "depth_30",
        "n_estimators": 300,
        "max_features": 0.8,
        "min_samples_leaf": 2,
        "max_depth": 30,
    },
]


# ============================================================
# PREPROCESSING
# ============================================================

def build_model(X_train, config):

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = ExtraTreesRegressor(
        n_estimators=config["n_estimators"],
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        max_depth=config["max_depth"],
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# EVALUATION
# ============================================================

def evaluate_fold(
    train_df,
    validation_df,
    config,
    fold_name,
):

    X_train = train_df.drop(
        columns=[TARGET, "load_id"]
    )

    y_train = train_df[TARGET]

    X_valid = validation_df.drop(
        columns=[TARGET, "load_id"]
    )

    y_valid = validation_df[TARGET]

    model = build_model(X_train, config)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    # Rates must be positive
    predictions = np.maximum(predictions, 0.01)

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions
        )
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    return {
        "configuration": config["name"],
        "fold": fold_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXTRA TREES HYPERPARAMETER TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nCreating features...")

    df = create_features(df)

    df = df.dropna(
        subset=["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

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

    all_results = []

    # --------------------------------------------------------
    # Test configurations
    # --------------------------------------------------------

    for config in CONFIGURATIONS:

        print("\n")
        print("=" * 70)
        print(
            f"CONFIGURATION: {config['name']}"
        )
        print("=" * 70)

        print(
            f"n_estimators     = {config['n_estimators']}"
        )

        print(
            f"max_features     = {config['max_features']}"
        )

        print(
            f"min_samples_leaf = {config['min_samples_leaf']}"
        )

        print(
            f"max_depth        = {config['max_depth']}"
        )

        for fold_name, validation_start, validation_end in folds:

            train_df = df[
                df["date"] < validation_start
            ].copy()

            validation_df = df[
                (df["date"] >= validation_start)
                &
                (df["date"] < validation_end)
            ].copy()

            print(
                f"\nRunning {fold_name}..."
            )

            result = evaluate_fold(
                train_df,
                validation_df,
                config,
                fold_name,
            )

            all_results.append(result)

            print(
                f"MAE:  {result['MAE']:.2f}"
            )

            print(
                f"RMSE: {result['RMSE']:.2f}"
            )

            print(
                f"R²:   {result['R2']:.4f}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        OUTPUT_DIR
        / "extra_trees_tuning_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate performance
    # --------------------------------------------------------

    summary = (
        results_df
        .groupby("configuration")
        .agg(
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            std_MAE=("MAE", "std"),
            std_RMSE=("RMSE", "std"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "mean_MAE"
    )

    summary_path = (
        OUTPUT_DIR
        / "extra_trees_tuning_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nResults saved to:")
    print(f"  {results_path}")

    print("\nSummary saved to:")
    print(f"  {summary_path}")

    print("\nTuning complete.")


if __name__ == "__main__":
    main()