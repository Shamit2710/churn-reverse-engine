"""
Layer 1: Churn prediction.

Trains a Logistic Regression model on the cleaned Telco data and outputs
a churn probability for every customer.

Run standalone with:
    python -m src.churn_model
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

from src import config
from src.data_prep import get_model_ready_data


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned data saved by data_prep.py"""
    return pd.read_csv(config.PROCESSED_DATA_PATH)


def prepare_features(df: pd.DataFrame):
    """
    One-hot encode categoricals, then split into X (features) and y (target).
    customerID and tenure_bucket are dropped from X -- customerID isn't
    predictive, and tenure_bucket is a duplicate of tenure used only for
    segmentation later, not for the model.
    """
    df_encoded = get_model_ready_data(df)

    drop_cols = [config.CUSTOMER_ID_COLUMN, config.TARGET_COLUMN, "tenure_bucket"]
    drop_cols = [c for c in drop_cols if c in df_encoded.columns]

    X = df_encoded.drop(columns=drop_cols)
    y = df_encoded[config.TARGET_COLUMN]

    return X, y


def train_model(X_train, y_train) -> LogisticRegression:
    """Train a Logistic Regression model. max_iter raised because the
    one-hot encoded feature set can be slow to converge on defaults."""
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print AUC-ROC and a full classification report.
    AUC is used instead of plain accuracy because churn is imbalanced
    (~27% churn rate) -- accuracy alone would be misleading here."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return auc


def score_all_customers(model, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    Add a churn_probability column to the full customer dataframe.
    This is the output Layer 2 (revenue attribution) will consume.
    """
    df = df.copy()
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    return df


def run_pipeline():
    df = load_processed_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Score the FULL dataset (not just test set) -- every customer needs
    # a probability for the downstream revenue attribution step
    df_scored = score_all_customers(model, df, X)

    joblib.dump(model, config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    return df_scored, model


if __name__ == "__main__":
    df_scored, model = run_pipeline()
    print("\nSample of scored customers:")
    print(df_scored[[config.CUSTOMER_ID_COLUMN, "churn_probability"]].head())