from src.models.predict import predict_churn

def test_predict_churn():
    assert callable(predict_churn)