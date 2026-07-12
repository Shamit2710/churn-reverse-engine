"""
Data loading, cleaning, and feature engineering for the Telco Customer Churn dataset.

Run standalone with:
    python -m src.data_prep
"""

import pandas as pd
from src import config


def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean known issues in the Telco dataset:
    - TotalCharges is read as a string and has blank entries for customers
      with tenure == 0 (brand new customers who haven't been billed yet).
      We fix these explicitly instead of silently dropping rows.
    """
    df = df.copy()

    # TotalCharges has blank strings, not NaN -- pandas reads the column as object
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Rows where TotalCharges is missing are new customers (tenure == 0).
    # Their TotalCharges should logically be 0, not dropped.
    new_customer_mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customer_mask, "TotalCharges"] = 0

    # Any remaining NaNs are genuine data issues -- drop and note how many
    remaining_nulls = df["TotalCharges"].isna().sum()
    if remaining_nulls > 0:
        print(f"Dropping {remaining_nulls} rows with unexplained missing TotalCharges")
        df = df.dropna(subset=["TotalCharges"])

    # Standardize target column to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features needed downstream:
    - tenure_bucket: for segmentation (Layer 3)
    - Keep Contract, MonthlyCharges, TotalCharges as-is -- needed for
      revenue attribution (Layer 2)
    """
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=config.TENURE_BUCKETS,
        labels=config.TENURE_BUCKET_LABELS,
        include_lowest=True,
    )

    return df


def get_model_ready_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categoricals for modeling. Keeps customerID and raw
    business columns (Contract, MonthlyCharges, tenure) untouched in a
    separate frame so Layer 2 can use them after prediction.
    """
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != config.CUSTOMER_ID_COLUMN]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


def run_pipeline() -> pd.DataFrame:
    """Full data prep pipeline: load -> clean -> feature engineer -> save."""
    df = load_raw_data()
    print(f"Loaded {len(df)} rows")

    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows")

    df = engineer_features(df)

    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {config.PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run_pipeline()