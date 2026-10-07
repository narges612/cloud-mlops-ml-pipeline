import pandas as pd


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