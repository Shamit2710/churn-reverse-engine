"""
Layer 2: Revenue Attribution.

Converts churn probability into a dollar figure: Expected Revenue at Risk.
This is the core translation from "data science output" to "business number."

Run standalone with:
    python -m src.revenue_attribution
"""

import pandas as pd
import joblib

from src import config
from src.churn_model import load_processed_data, prepare_features, score_all_customers


def calculate_remaining_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate how much revenue a customer would generate if they DON'T churn.

    remaining_value = MonthlyCharges x expected_remaining_months

    expected_remaining_months comes from config.py and depends on contract
    type -- this is a stated assumption, not derived from the data.
    """
    df = df.copy()

    df["expected_remaining_months"] = df["Contract"].map(config.EXPECTED_REMAINING_MONTHS)
    df["remaining_value"] = df["MonthlyCharges"] * df["expected_remaining_months"]

    return df


def calculate_revenue_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    The key output of this layer:
    expected_revenue_at_risk = churn_probability x remaining_value

    This is the number that turns "73% likely to churn" into "$X at risk" --
    the thing a business can actually act on.
    """
    df = df.copy()
    df["expected_revenue_at_risk"] = df["churn_probability"] * df["remaining_value"]
    return df


def run_pipeline() -> pd.DataFrame:
    # Rebuild the scored dataframe (re-running Layer 1's scoring step)
    df = load_processed_data()
    X, y = prepare_features(df)
    model = joblib.load(config.MODEL_PATH)
    df_scored = score_all_customers(model, df, X)

    # Layer 2 additions
    df_scored = calculate_remaining_value(df_scored)
    df_scored = calculate_revenue_at_risk(df_scored)

    df_scored.to_csv(config.SCORED_DATA_PATH, index=False)
    print(f"Saved scored + revenue-attributed data to {config.SCORED_DATA_PATH}")

    return df_scored


if __name__ == "__main__":
    df_scored = run_pipeline()

    total_at_risk = df_scored["expected_revenue_at_risk"].sum()
    print(f"\nTotal expected revenue at risk across all customers: ${total_at_risk:,.2f}")

    print("\nTop 5 customers by revenue at risk:")
    top5 = df_scored.sort_values("expected_revenue_at_risk", ascending=False)
    print(top5[["customerID", "churn_probability", "remaining_value", "expected_revenue_at_risk"]].head())