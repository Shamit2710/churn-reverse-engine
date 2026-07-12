"""
Layer 4: Intervention Cost-Justification (capacity-constrained).

For each segment, allocates customers to the best intervention that still
has capacity available. Segments are served in priority order (highest
total revenue at risk first), so limited resources like call-center hours
go to the customers where they matter most. When an intervention's
capacity runs out, remaining customers fall through to the next best
option -- this produces a realistic, varied allocation instead of
everyone getting the same recommendation.

Run standalone with:
    python -m src.intervention
"""

import pandas as pd

from src import config
from src.segmentation import run_pipeline as run_segmentation_pipeline


def add_avg_remaining_value(df_segmented: pd.DataFrame, segment_summary: pd.DataFrame) -> pd.DataFrame:
    """Add avg remaining_value per segment, needed to calculate dollar savings."""
    avg_value = df_segmented.groupby("segment")["remaining_value"].mean().reset_index()
    avg_value.columns = ["segment", "avg_remaining_value"]
    return segment_summary.merge(avg_value, on="segment")


def net_benefit_per_customer(avg_remaining_value: float, intervention_params: dict) -> float:
    """net benefit for ONE customer under a given intervention."""
    saved = intervention_params["churn_reduction_pp"] * avg_remaining_value
    cost = intervention_params["cost_per_customer"]
    return saved - cost


def allocate_interventions(segment_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy capacity-constrained allocation:
    1. Process segments in priority order (highest total_revenue_at_risk first).
    2. For each segment, try interventions from most effective to least
       (personal_retention_call -> discount_offer -> retention_email).
    3. Assign as many of the segment's customers as the intervention's
       remaining capacity allows. Leftover customers fall through to the
       next intervention in the list.
    4. If a segment's remaining customers get zero net benefit from every
       remaining option, they're marked "no action."
    """
    # Remaining capacity tracker -- None means unlimited
    remaining_capacity = {
        name: params["monthly_capacity"] for name, params in config.INTERVENTIONS.items()
    }

    # Try most effective (highest churn_reduction_pp) first
    intervention_order = sorted(
        config.INTERVENTIONS.items(),
        key=lambda item: item[1]["churn_reduction_pp"],
        reverse=True,
    )

    segment_summary = segment_summary.sort_values("total_revenue_at_risk", ascending=False)

    allocation_rows = []

    for _, row in segment_summary.iterrows():
        customers_left = row["customer_count"]
        segment_allocations = []  # (intervention_name, customers_assigned, net_benefit)

        for name, params in intervention_order:
            if customers_left <= 0:
                break

            per_customer_benefit = net_benefit_per_customer(row["avg_remaining_value"], params)
            if per_customer_benefit <= 0:
                continue  # not worth it regardless of capacity

            cap = remaining_capacity[name]
            if cap is None:
                assign_count = customers_left  # unlimited
            else:
                assign_count = min(customers_left, cap)

            if assign_count <= 0:
                continue

            net_benefit = per_customer_benefit * assign_count
            segment_allocations.append((name, assign_count, net_benefit))

            if cap is not None:
                remaining_capacity[name] -= assign_count
            customers_left -= assign_count

        # Anything left over after all interventions get no benefit -> no action
        if customers_left > 0:
            segment_allocations.append(("no action", customers_left, 0))

        for intervention_name, count, net_benefit in segment_allocations:
            allocation_rows.append({
                "segment": row["segment"],
                "recommended_action": intervention_name,
                "customers_assigned": count,
                "expected_net_benefit": net_benefit,
            })

    return pd.DataFrame(allocation_rows)


def run_pipeline():
    df_segmented, segment_summary = run_segmentation_pipeline()
    segment_summary = add_avg_remaining_value(df_segmented, segment_summary)
    allocation = allocate_interventions(segment_summary)

    return df_segmented, segment_summary, allocation


if __name__ == "__main__":
    df_segmented, segment_summary, allocation = run_pipeline()

    print("Capacity-constrained intervention allocation:\n")
    print(allocation.to_string(index=False))

    total_expected_saved = allocation["expected_net_benefit"].sum()
    print(f"\nTotal expected net revenue saved: ${total_expected_saved:,.2f}")

    print("\nCapacity utilization check:")
    for name in config.INTERVENTIONS:
        used = allocation[allocation["recommended_action"] == name]["customers_assigned"].sum()
        cap = config.INTERVENTIONS[name]["monthly_capacity"]
        cap_str = f"{cap}" if cap is not None else "unlimited"
        print(f"  {name}: {used} used / {cap_str} capacity")