"""
Churn-to-Revenue Attribution Engine -- Dashboard (minimal layout)

Run with:
    streamlit run app/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from src import config
from src.intervention import run_pipeline

st.set_page_config(page_title="Churn Revenue Engine", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# MINIMAL STYLING
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}
    div[data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        padding: 16px 20px;
    }
    div[data-testid="stMetricLabel"] {font-size: 13px; opacity: 0.7;}
    h1 {font-size: 28px; margin-bottom: 4px;}
    h3 {font-size: 18px; margin-top: 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df_segmented, segment_summary, allocation = run_pipeline()
    return df_segmented, segment_summary, allocation


df_segmented, segment_summary, allocation = load_data()

# ---------------------------------------------------------------------------
# HEADER + SUMMARY
# ---------------------------------------------------------------------------
st.title("Churn-to-Revenue Attribution Engine")
st.caption("Predicts churn, converts it into dollar-value risk, and recommends where to spend retention budget.")

total_at_risk = df_segmented["expected_revenue_at_risk"].sum()
total_saved = allocation["expected_net_benefit"].sum()
total_helped = allocation[allocation["recommended_action"] != "no action"]["customers_assigned"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Revenue at Risk", f"${total_at_risk:,.0f}")
col2.metric("Net Revenue Saved", f"${total_saved:,.0f}")
col3.metric("Customers Covered", f"{total_helped:,}")

st.write("")

# ---------------------------------------------------------------------------
# TABS -- keeps each view focused instead of one long scroll
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Segment Priority", "Intervention Plan", "Customer Lookup"])

with tab1:
    st.caption("Ranked by total revenue at risk, not average churn probability.")

    top_segments = segment_summary.sort_values("total_revenue_at_risk", ascending=False).head(8)
    fig = px.bar(
        top_segments.sort_values("total_revenue_at_risk"),
        x="total_revenue_at_risk",
        y="segment",
        orientation="h",
        labels={"total_revenue_at_risk": "", "segment": ""},
        height=350,
    )
    fig.update_traces(marker_color="#4C8BF5")
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        segment_summary[["segment", "customer_count", "avg_churn_probability", "total_revenue_at_risk"]]
        .rename(columns={
            "segment": "Segment", "customer_count": "Customers",
            "avg_churn_probability": "Avg Churn %", "total_revenue_at_risk": "Revenue at Risk",
        })
        .style.format({"Avg Churn %": "{:.0%}", "Revenue at Risk": "${:,.0f}"}),
        use_container_width=True, hide_index=True, height=280,
    )

with tab2:
    st.caption("Limited resources allocated to highest-priority segments first.")
    st.dataframe(
        allocation.rename(columns={
            "segment": "Segment", "recommended_action": "Action",
            "customers_assigned": "Customers", "expected_net_benefit": "Net Benefit",
        })
        .style.format({"Net Benefit": "${:,.0f}"}),
        use_container_width=True, hide_index=True, height=420,
    )

with tab3:
    selected_segment = st.selectbox("Segment", sorted(df_segmented["segment"].unique()))
    segment_customers = df_segmented[df_segmented["segment"] == selected_segment].sort_values(
        "expected_revenue_at_risk", ascending=False
    ).head(50)

    st.dataframe(
        segment_customers[["customerID", "churn_probability", "remaining_value", "expected_revenue_at_risk"]]
        .rename(columns={
            "customerID": "Customer", "churn_probability": "Churn %",
            "remaining_value": "Remaining Value", "expected_revenue_at_risk": "Revenue at Risk",
        })
        .style.format({"Churn %": "{:.0%}", "Remaining Value": "${:,.0f}", "Revenue at Risk": "${:,.0f}"}),
        use_container_width=True, hide_index=True, height=400,
    )