# Cloud MLOps ML Pipeline

An end-to-end machine learning project for customer churn prediction, combining data ingestion, preprocessing, model training, experiment tracking, API deployment, and CI/CD automation.

## Architecture

- **Data source:** IBM Telco Customer Churn dataset, with data ingestion from Azure Databricks.
- **Data processing:** Python and pandas for cleaning and preprocessing.
- **Model training:** scikit-learn classification pipeline.
- **Experiment tracking and model registry:** MLflow integrated with Azure Machine Learning.
- **API:** FastAPI endpoints for health checks and churn predictions.
- **Containerization:** Docker.
- **Cloud deployment:** Azure Container Registry and Azure Container Apps.
- **CI/CD:** GitHub Actions for automated tests, Docker image builds, and deployment.

## Deployed API

Base URL: https://churn-prediction-api.gentleground-498115d9.northeurope.azurecontainerapps.io

- Health check: [`/health`](https://churn-prediction-api.gentleground-498115d9.northeurope.azurecontainerapps.io/health)
- Interactive API documentation: [`/docs`](https://churn-prediction-api.gentleground-498115d9.northeurope.azurecontainerapps.io/docs)
- Prediction endpoint: `POST /predict`

## Model Evaluation

The baseline logistic regression model achieved:

- Accuracy: **80.4%**
- ROC-AUC: **0.836**

These metrics are from the model's evaluation on the held-out test set.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the required environment variables in a local `.env` file. Never commit credentials or tokens.

Run the tests:

```bash
python -m pytest -v
```

## Run with Docker

Build the image:

```bash
docker build -t cloud-mlops-ml-pipeline .
```

Run the API using a local copy of the trained model:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/models/local \
  cloud-mlops-ml-pipeline
```

Then open http://localhost:8000/docs.

The local model directory must exist and contain the trained model artifacts before building the image.

## CI/CD

The GitHub Actions workflows run tests and automate deployment to Azure Container Apps after changes are pushed to the `main` branch. Deployment requires repository secrets for Azure authentication.

## Future Improvements

- Add data validation and pipeline orchestration.
- Extend experiment tracking and model monitoring.
- Improve automated deployment checks and integration testing.
- Explore distributed data processing with PySpark.
