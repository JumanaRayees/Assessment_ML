from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import create_features



# Paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "Data" / "train-test_clean.csv"

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)



# Configuration


TARGET = "posted_rate"

# October will be our holdout validation period.
VALIDATION_START = pd.Timestamp("2025-10-01")



# Metrics


def evaluate_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    """
    Train model and calculate validation metrics.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    # Rates must be positive
    predictions = np.maximum(predictions, 0.01)

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

    return model, predictions, mae, rmse, r2


# Main


def main():

    print("=" * 60)
    print("FREIGHT RATE MODEL TRAINING")
    print("=" * 60)

   
    # Load data
  

    print("\nLoading data...")

    df = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"],
    )

    print(f"Dataset shape: {df.shape}")

  
    # Feature engineering


    print("\nCreating features...")

    df = create_features(df)

    
    # Remove rows with invalid date
    

    df = df.dropna(
        subset=["date"]
    ).copy()

    
    # Time-based split
   

    train_df = df[
        df["date"] < VALIDATION_START
    ].copy()

    valid_df = df[
        df["date"] >= VALIDATION_START
    ].copy()

    print("\nTime-based split:")
    print(
        f"Training rows:   {len(train_df):,}"
    )
    print(
        f"Validation rows: {len(valid_df):,}"
    )

    print(
        f"\nTraining period: "
        f"{train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{valid_df['date'].min().date()} → "
        f"{valid_df['date'].max().date()}"
    )

 
    # Separate X / y
    

    X_train = train_df.drop(
        columns=[TARGET, "load_id"],
    )

    y_train = train_df[TARGET]

    X_valid = valid_df.drop(
        columns=[TARGET, "load_id"],
    )

    y_valid = valid_df[TARGET]

   
    # Identify columns
   

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    print(
        f"\nNumeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print("\nCategorical columns:")
    for col in categorical_features:
        print(f"  - {col}")

  
    # Preprocessing
   

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

    
    # Models
  

    models = {

        "dummy_mean": DummyRegressor(
            strategy="mean"
        ),

        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []


    # Train models
    

    for model_name, estimator in models.items():

        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        (
            fitted_model,
            predictions,
            mae,
            rmse,
            r2,
        ) = evaluate_model(
            pipeline,
            X_train,
            y_train,
            X_valid,
            y_valid,
        )

        print(
            f"MAE:  {mae:,.2f}"
        )

        print(
            f"RMSE: {rmse:,.2f}"
        )

        print(
            f"R²:   {r2:.4f}"
        )

        results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        # Save non-baseline model
        if model_name != "dummy_mean":

            model_path = (
                MODEL_DIR
                / f"{model_name}.joblib"
            )

            joblib.dump(
                fitted_model,
                model_path,
            )

            print(
                f"Saved model: {model_path}"
            )


    # Results
    

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_path = (
        PROJECT_ROOT
        / "outputs"
        / "validation_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()