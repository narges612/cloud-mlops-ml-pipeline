import mlflow
import pandas as pd
import os

MODEL_URI = "models:/churn-prediction-model/1"

def predict_churn(model, customer):
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    return prediction, probability

def load_model():
    model_path = os.getenv("MODEL_PATH", MODEL_URI)
    return mlflow.sklearn.load_model(model_path)

sample = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.0,
    "TotalCharges": 660.0,
}])


if __name__ == "__main__":
    mlflow.set_tracking_uri("azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/425322da-2383-4f05-bbbe-03573b58e4d8/resourceGroups/rg-cloud-mlops/providers/Microsoft.MachineLearningServices/workspaces/mlops-workspace")
    model = load_model()
    prediction, probability = predict_churn(model, sample)

    print("Prediction:", "Churn" if prediction == 1 else "No Churn")
    print("Churn probability:", round(probability, 4))