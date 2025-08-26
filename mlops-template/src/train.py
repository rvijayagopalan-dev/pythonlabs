import os
from datetime import datetime

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from joblib import dump

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main():
    # Data
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris-experiment")

    with mlflow.start_run(run_name=f"train-{datetime.utcnow().isoformat()}"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "penalty": pipeline.named_steps["clf"].penalty,
            "C": pipeline.named_steps["clf"].C,
            "solver": pipeline.named_steps["clf"].solver,
        })

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

    # Promote latest un-staged version to Staging
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"]) or []
    if latest:
        version = latest[-1].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        print(f"Promoted {MODEL_NAME} v{version} to Staging")

    # Local fallback artifact
    os.makedirs("models/latest", exist_ok=True)
    dump(pipeline, "models/latest/model.pkl")
    print("Saved local fallback model to models/latest/model.pkl")


if __name__ == "__main__":
    main()
