from pathlib import Path
import numpy as np
import pandas as pd



# Configuration

DATA_PATH = Path("Data/train-test.csv")
OUTPUT_PATH = Path("Data/train-test_clean.csv")



# 1. Load data

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw training/development dataset."""
    df = pd.read_csv(path)

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# 2. Initial data-quality report

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a basic data-quality report."""

    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
    })

    report["duplicate_values"] = [
        df[col].duplicated().sum()
        for col in df.columns
    ]

    return report


def print_quality_report(df: pd.DataFrame):
    """Print data-quality information."""

    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    report = data_quality_report(df)
    print(report)

    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0])



# 3. Check duplicate rows


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Check duplicate rows and duplicate load IDs."""

    print("\n" + "=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")

    if "load_id" in df.columns:
        duplicate_load_ids = df["load_id"].duplicated().sum()
        print(f"Duplicate load_id values: {duplicate_load_ids}")

    return df



# 4. Remove exact duplicate rows


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""

    before = len(df)

    df = df.drop_duplicates().copy()

    removed = before - len(df)

    print(f"\nRemoved duplicate rows: {removed}")

    return df



# 5. Validate load_id


def validate_load_id(df: pd.DataFrame) -> pd.DataFrame:
    """Validate load_id values."""

    if "load_id" not in df.columns:
        print("\nWARNING: load_id column not found.")
        return df

    print("\n" + "=" * 70)
    print("LOAD ID VALIDATION")
    print("=" * 70)

    print(f"Missing load_id: {df['load_id'].isna().sum()}")
    print(f"Duplicate load_id: {df['load_id'].duplicated().sum()}")

    # Do not silently remove duplicate IDs here.
    # Duplicate IDs should be investigated first.

    return df


# 6. Clean numerical values


def clean_numerical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix clearly invalid numerical values.

    Important:
    We do NOT remove statistical outliers automatically.
    Freight rates can legitimately contain expensive loads.
    """

    print("\n" + "=" * 70)
    print("NUMERICAL VALUE VALIDATION")
    print("=" * 70)

  
    # Weight


    if "weight" in df.columns:

        negative_weight = (df["weight"] < 0).sum()

        print(f"Negative weight values: {negative_weight}")

        # Negative weight is physically impossible.
        # Convert invalid values to NaN so they can be
        # handled later by the preprocessing pipeline.
        df.loc[df["weight"] < 0, "weight"] = np.nan

        zero_weight = (df["weight"] == 0).sum()
        print(f"Zero weight values: {zero_weight}")

   
    # Distance
    

    if "distance" in df.columns:

        negative_distance = (df["distance"] < 0).sum()

        print(f"Negative distance values: {negative_distance}")

        # Negative distance is impossible.
        df.loc[df["distance"] < 0, "distance"] = np.nan

   
    # Market index
  

    if "market_index" in df.columns:

        negative_market_index = (df["market_index"] < 0).sum()

        print(f"Negative market_index values: {negative_market_index}")

        # Do not automatically replace negative market_index
        # unless domain knowledge confirms it is impossible.
        #
        # Therefore we only report it here.

   
    # Quote signal
  

    if "quote_signal" in df.columns:

        print(
            f"quote_signal min: "
            f"{df['quote_signal'].min()}"
        )

        print(
            f"quote_signal max: "
            f"{df['quote_signal'].max()}"
        )

    return df



# 7. Validate geographic coordinates


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate latitude and longitude ranges."""

    print("\n" + "=" * 70)
    print("GEOGRAPHIC VALIDATION")
    print("=" * 70)

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]

    for col in coordinate_columns:
        if col not in df.columns:
            continue

        print(f"\n{col}")
        print(f"  Missing: {df[col].isna().sum()}")
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")

    # Latitude must be between -90 and 90.
    if "pickup_lat" in df.columns:
        invalid = ~df["pickup_lat"].between(-90, 90)
        invalid &= df["pickup_lat"].notna()

        print(f"Invalid pickup latitude: {invalid.sum()}")

        df.loc[invalid, "pickup_lat"] = np.nan

    if "delivery_lat" in df.columns:
        invalid = ~df["delivery_lat"].between(-90, 90)
        invalid &= df["delivery_lat"].notna()

        print(f"Invalid delivery latitude: {invalid.sum()}")

        df.loc[invalid, "delivery_lat"] = np.nan

    # Longitude must be between -180 and 180.
    if "pickup_lon" in df.columns:
        invalid = ~df["pickup_lon"].between(-180, 180)
        invalid &= df["pickup_lon"].notna()

        print(f"Invalid pickup longitude: {invalid.sum()}")

        df.loc[invalid, "pickup_lon"] = np.nan

    if "delivery_lon" in df.columns:
        invalid = ~df["delivery_lon"].between(-180, 180)
        invalid &= df["delivery_lon"].notna()

        print(f"Invalid delivery longitude: {invalid.sum()}")

        df.loc[invalid, "delivery_lon"] = np.nan

    return df



# 8. Date validation


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime and detect invalid dates."""

    print("\n" + "=" * 70)
    print("DATE VALIDATION")
    print("=" * 70)

    date_columns = [
        col
        for col in df.columns
        if "date" in col.lower()
    ]

    for col in date_columns:

        print(f"\nProcessing: {col}")

        original_missing = df[col].isna().sum()

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        new_missing = df[col].isna().sum()

        invalid_created = new_missing - original_missing

        print(f"Original missing: {original_missing}")
        print(f"Invalid dates converted to NaT: {invalid_created}")
        print(f"Final missing: {new_missing}")

        if df[col].notna().any():
            print(f"Min date: {df[col].min()}")
            print(f"Max date: {df[col].max()}")

    return df



# 9. Validate categorical columns


def validate_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect categorical columns for inconsistent values."""

    print("\n" + "=" * 70)
    print("CATEGORICAL VALIDATION")
    print("=" * 70)

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        print("\n" + "-" * 50)
        print(f"Column: {col}")
        print(f"Unique values: {df[col].nunique(dropna=True)}")

        print(df[col].value_counts(dropna=False).head(20))

    return df



# 10. Clean string columns


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove accidental leading/trailing whitespace.

    We do NOT lowercase categorical values automatically because
    categorical normalization should be based on actual values.
    """

    string_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for col in string_columns:

        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return df



# 11. Validate target


def validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the regression target."""

    target = "posted_rate"

    print("\n" + "=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' was not found."
        )

    print(f"Target: {target}")
    print(f"Missing: {df[target].isna().sum()}")
    print(f"Minimum: {df[target].min()}")
    print(f"Maximum: {df[target].max()}")
    print(f"Mean: {df[target].mean():.2f}")
    print(f"Median: {df[target].median():.2f}")

    negative_target = (df[target] < 0).sum()

    print(f"Negative target values: {negative_target}")

    # Negative freight rates are invalid.
    if negative_target > 0:
        raise ValueError(
            "Negative posted_rate values were found. "
            "Investigate before training."
        )

    # Target missing values cannot be used for supervised training.
    missing_target = df[target].isna().sum()

    if missing_target > 0:
        print(
            f"WARNING: {missing_target} rows have missing "
            f"posted_rate and should not be used for training."
        )

    return df



# 12. Numerical summary


def numerical_summary(df: pd.DataFrame):
    """Print numerical statistics."""

    print("\n" + "=" * 70)
    print("NUMERICAL SUMMARY")
    print("=" * 70)

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    print(
        df[numerical_columns]
        .describe()
        .T
    )


# 13. Final validation


def final_validation(df: pd.DataFrame):
    """Run final data-quality checks."""

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(f"Final shape: {df.shape}")

    print(
        f"Duplicate rows: "
        f"{df.duplicated().sum()}"
    )

    if "load_id" in df.columns:
        print(
            f"Duplicate load_id: "
            f"{df['load_id'].duplicated().sum()}"
        )

    print("\nRemaining missing values:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing)

    print("\nData types:")
    print(df.dtypes)

    print("\nFinal target statistics:")

    if "posted_rate" in df.columns:
        print(df["posted_rate"].describe())



# 14. Main cleaning pipeline


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the complete cleaning process."""

    print("\n")
    print("#" * 70)
    print("# STARTING DATA CLEANING")
    print("#" * 70)

    # Basic string cleanup
    df = clean_string_columns(df)

    # Remove exact duplicate rows
    df = remove_duplicate_rows(df)

    # Validate load IDs
    df = validate_load_id(df)

    # Convert dates
    df = clean_dates(df)

    # Clean invalid numerical values
    df = clean_numerical_values(df)

    # Validate geographic coordinates
    df = validate_coordinates(df)

    # Inspect categorical values
    df = validate_categorical_columns(df)

    # Validate target
    df = validate_target(df)

    return df



# 15. Main execution


def main():

    # Load
    

    df = load_data(DATA_PATH)

   
    # Initial report
 

    print_quality_report(df)

    
    # Duplicate check
   

    check_duplicates(df)

    
    # Numerical summary BEFORE cleaning
   

    numerical_summary(df)

    
    # Cleaning
  

    df_clean = clean_data(df)

    
    # Final validation
   

    final_validation(df_clean)

    # Save cleaned dataset
   

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_clean.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(f"Saved cleaned data to:")
    print(OUTPUT_PATH)

    print(f"\nFinal shape: {df_clean.shape}")


if __name__ == "__main__":
    main()

