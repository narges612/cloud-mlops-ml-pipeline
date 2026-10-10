
import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.models.predict import MODEL_URI, predict_churn, load_model

TRACKING_URI = (
    "azureml://northeurope.api.azureml.ms/mlflow/v1.0/"
    "subscriptions/425322da-2383-4f05-bbbe-03573b58e4d8/"
    "resourceGroups/rg-cloud-mlops/providers/"
    "Microsoft.MachineLearningServices/workspaces/mlops-workspace"
)

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Customer Churn Prediction API")

model = None


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.on_event("startup")
def load_prediction_model():
    global model
    model = load_model()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: Customer):
    customer_df = pd.DataFrame([customer.model_dump()])
    prediction, probability = predict_churn(model, customer_df)

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4),
    }
