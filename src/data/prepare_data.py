
import os
from pathlib import Path

import pandas as pd

from src.data.databricks_reader import load_customer_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data/raw/telco_customer_churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/telco_customer_churn.csv"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco Customer Churn dataset."""
    df = df.copy()

    # Remove customer identifiers
    df = df.drop(columns=["customerID"])

    # Convert blank TotalCharges values to missing values
    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$", pd.NA, regex=True
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Remove rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    return df


def prepare_data():
    source = os.getenv("DATA_SOURCE", "local").lower()

    if source == "databricks":
        df = load_customer_data()
        print("Source: Databricks")
    elif source == "local":
        df = pd.read_csv(RAW_PATH)
        print("Source: local CSV")
    else:
        raise ValueError(
            "DATA_SOURCE must be either 'local' or 'databricks'"
        )

    cleaned_df = preprocess_data(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_PATH, index=False)

    print("Original shape:", df.shape)
    print("Processed shape:", cleaned_df.shape)
    print("Saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    prepare_data()
