"""
Layer 3: Segmentation & Prioritization.

Groups customers into business-meaningful segments (contract type x tenure
bucket) and ranks segments by TOTAL revenue at risk -- not average churn
probability. This is the layer that turns individual customer scores into
something a business can actually act on.

Run standalone with:
    python -m src.segmentation
"""

import pandas as pd

from src import config
from src.revenue_attribution import run_pipeline as run_revenue_pipeline


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group customers by Contract type x tenure_bucket.
    This gives 3 contract types x 4 tenure buckets = up to 12 segments,
    a manageable number a business could realistically act on.
    """
    df = df.copy()
    df["segment"] = df["Contract"] + " | " + df["tenure_bucket"].astype(str)
    return df


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each segment, compute the numbers that matter for prioritization:
    - customer_count
    - avg_churn_probability
    - total_revenue_at_risk  <-- the number we rank by, NOT probability
    """
    summary = df.groupby("segment").agg(
        customer_count=("customerID", "count"),
        avg_churn_probability=("churn_probability", "mean"),
        total_revenue_at_risk=("expected_revenue_at_risk", "sum"),
    ).reset_index()

    # Rank by TOTAL dollars at risk, not average probability.
    # This is the deliberate design choice: a segment with moderate risk
    # but many high-value customers can matter more than a small segment
    # with a scary-looking probability.
    summary = summary.sort_values("total_revenue_at_risk", ascending=False)
    summary["rank"] = range(1, len(summary) + 1)

    return summary


def run_pipeline():
    df_scored = run_revenue_pipeline()
    df_segmented = build_segments(df_scored)
    segment_summary = summarize_segments(df_segmented)

    return df_segmented, segment_summary


if __name__ == "__main__":
    df_segmented, segment_summary = run_pipeline()

    print("Segment priority ranking (by total revenue at risk):\n")
    print(segment_summary.to_string(index=False))