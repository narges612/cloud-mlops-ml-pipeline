import pandas as pd

from src.data.prepare_data import preprocess_data

def test_preprocess_data():
    df = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [1, 0, 12],
        "TotalCharges": ["50.0", " ", "600.0"],
        "MonthlyCharges": [50.0, 60.0, 50.0],
        "Churn": ["No", "Yes", "No"],
    })

    result = preprocess_data(df)

    assert "customerID" not in result.columns
    assert len(result) == 2
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])
    assert result["TotalCharges"].isna().sum() == 0