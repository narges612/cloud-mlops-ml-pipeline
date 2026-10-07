import pandas as pd

DATA_PATH = "data/raw/telco_customer_churn.csv"


def inspect_dataset():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    #print("\nChurn distribution:")
    #print(df["Churn"].value_counts())

    #print("\nMissing values:")
    #print(df.isnull().sum())

    #print("\nData types:")
    #print(df.dtypes)

    #print("\nBlank TotalCharges values:")
    #print((df["TotalCharges"].str.strip() == "").sum())


if __name__ == "__main__":
    inspect_dataset()