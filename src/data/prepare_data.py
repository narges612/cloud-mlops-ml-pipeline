
import os
from pathlib import Path

from src.data.databricks_reader import load_customer_data
from src.features.preprocessing import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data/raw/telco_customer_churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/telco_customer_churn.csv"


def prepare_data():
    source = os.getenv("DATA_SOURCE", "local").lower()

    if source == "databricks":
        df = load_customer_data()
        print("Source: Databricks")
    elif source == "local":
        import pandas as pd
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
