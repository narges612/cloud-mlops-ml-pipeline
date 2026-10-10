
from src.data.databricks_reader import load_customer_data


def test_load_customer_data():
    df = load_customer_data()

    assert len(df) == 7043
    assert "Churn" in df.columns

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print(df["Churn"].value_counts())

if __name__ == "__main__":
    test_load_customer_data()