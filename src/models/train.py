import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from src.data.prepare_data import prepare_data

DATA_PATH = "data/processed/telco_customer_churn.csv"


def train_model():
    prepare_data()
    df = pd.read_csv(DATA_PATH)
    mlflow.set_tracking_uri(
    "azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/425322da-2383-4f05-bbbe-03573b58e4d8/resourceGroups/rg-cloud-mlops/providers/Microsoft.MachineLearningServices/workspaces/mlops-workspace")
    mlflow.set_experiment("customer-churn-prediction")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    print("Features shape:", X.shape)
    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

   
    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, probabilities)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="churn-prediction-model",
            serialization_format="cloudpickle",
            )

        print("\nClassification report:")
        print(classification_report(y_test, predictions))
        print("Accuracy:", round(accuracy, 4))
        print("ROC-AUC:", round(roc_auc, 4))

if __name__ == "__main__":
    train_model()