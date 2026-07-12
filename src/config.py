"""
All business assumptions for the Churn-to-Revenue Attribution Engine live here.
Keeping them in one file means every number in the final output can be traced
back to a single, explicit, editable assumption -- which is exactly what you'd
want to defend in an interview or a stakeholder review.
"""

# ---------------------------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/customers_clean.csv"
SCORED_DATA_PATH = "outputs/customers_scored.csv"
MODEL_PATH = "outputs/model.pkl"

TARGET_COLUMN = "Churn"          # Yes/No in the raw data
CUSTOMER_ID_COLUMN = "customerID"

# ---------------------------------------------------------------------------
# LAYER 2 -- REVENUE ATTRIBUTION
# ---------------------------------------------------------------------------
# Expected remaining months a customer stays, by contract type, if they DON'T churn.
# These are assumptions, not derived from data -- state this clearly in your README.
EXPECTED_REMAINING_MONTHS = {
    "Month-to-month": 6,
    "One year": 12,
    "Two year": 24,
}

# ---------------------------------------------------------------------------
# LAYER 3 -- SEGMENTATION
# ---------------------------------------------------------------------------
# Tenure buckets used to build segments (in months)
TENURE_BUCKETS = [0, 12, 24, 48, 72]
TENURE_BUCKET_LABELS = ["0-12mo", "12-24mo", "24-48mo", "48-72mo"]

# ---------------------------------------------------------------------------
# LAYER 4 -- INTERVENTIONS
# ---------------------------------------------------------------------------
# cost_per_customer: what it costs the business to run this intervention on one customer
# churn_reduction_pp: assumed reduction in churn probability, in percentage points
INTERVENTIONS = {
    "retention_email": {
        "cost_per_customer": 2,
        "churn_reduction_pp": 0.03,
        "monthly_capacity": None,  # cheap and automated -- effectively unlimited
    },
    "discount_offer": {
        "cost_per_customer": 15,
        "churn_reduction_pp": 0.10,
        "monthly_capacity": 3000,  # limited by discount/promo budget
    },
    "personal_retention_call": {
        "cost_per_customer": 25,
        "churn_reduction_pp": 0.15,
        "monthly_capacity": 1200,  # limited by call center headcount/hours
    },
}

# ---------------------------------------------------------------------------
# MODELING
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2