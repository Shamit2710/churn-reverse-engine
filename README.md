Churn-to-Revenue Attribution Engine

Most churn models stop at predicting who is likely to leave. This project goes further: it converts churn probability into dollar-value revenue at risk, prioritizes customer segments by financial impact rather than raw churn percentage, and recommends which retention interventions are actually worth their cost under realistic capacity constraints.

The problem

A subscription business loses customers every month. The common question is "who's going to churn?" The more useful question is "of the customers who might churn, which ones are worth spending money to save, and where should limited retention resources go?" That's a resource-allocation problem, not just a prediction problem — this project treats it as one.

Key result

On the Telco Customer Churn dataset (7,043 customers):


Churn model AUC-ROC: 0.84
Total expected revenue at risk: $996,165
41% of that risk ($408,956) sits in a single segment: month-to-month customers in their first year
Under realistic capacity constraints (limited call-center hours, limited discount budget), the achievable net revenue saved drops to $237,177 — a ~60% reduction from the uncapped estimate of $598,780. This gap is itself a finding: it quantifies the real cost of under-resourced retention operations, a case for investing in additional capacity.


How it works

1. Churn Prediction — Logistic Regression trained on customer usage, contract, and service data. Outputs a churn probability per customer. (AUC 0.84)

2. Revenue Attribution — Converts probability into a dollar figure:
expected_revenue_at_risk = churn_probability × remaining_customer_value
where remaining value is estimated from monthly charges and expected months remaining by contract type (assumptions documented in src/config.py).

3. Segmentation & Prioritization — Groups customers by contract type × tenure, then ranks segments by total revenue at risk, not average churn probability. A segment with a lower average churn rate but many high-value customers can carry as much total risk as a smaller, scarier-looking segment — this ranking makes that visible instead of hiding it.

4. Capacity-Constrained Intervention Allocation — Three retention interventions (email, discount, personal call) each have a cost and an assumed effectiveness, and two of the three have a monthly capacity limit. Customers are allocated to the best available intervention in priority order, so the highest-value segments get the most effective (and most limited) resources first, with lower-priority segments falling through to cheaper alternatives.

Assumptions (stated explicitly, not hidden)

All business assumptions live in src/config.py:


Expected remaining customer lifetime by contract type
Intervention costs, effectiveness, and monthly capacity


These are estimates, not derived from the data — a real deployment would calibrate them against actual retention campaign history.

Tech stack

Python, pandas, scikit-learn, Streamlit, Plotly

Project structure

src/
  data_prep.py           # cleaning + feature engineering
  churn_model.py          # Layer 1: churn prediction
  revenue_attribution.py  # Layer 2: probability -> dollars
  segmentation.py          # Layer 3: segment ranking
  intervention.py          # Layer 4: capacity-constrained allocation
  config.py                # all assumptions in one place
app/
  dashboard.py              # Streamlit dashboard

Running it

bashpython -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# place the Telco Customer Churn CSV at data/raw/telco_churn.csv
python -m src.data_prep
python -m src.churn_model
streamlit run app/dashboard.py

Dataset
 https://www.kaggle.com/datasets/blastchar/telco-customer-churn, 7,043 rows.