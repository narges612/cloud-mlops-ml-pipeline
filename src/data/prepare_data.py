
import pandas as pd

from src.features.preprocessing import preprocess_data

RAW_PATH = "data/raw/telco_customer_churn.csv"
PROCESSED_PATH = "data/processed/telco_customer_churn.csv"


def prepare_data():
    df = pd.read_csv(RAW_PATH)
    cleaned_df = preprocess_data(df)

    cleaned_df.to_csv(PROCESSED_PATH, index=False)

    print("Original shape:", df.shape)
    print("Processed shape:", cleaned_df.shape)
    print("Saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    prepare_data()