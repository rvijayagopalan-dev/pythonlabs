import os
import pandas as pd
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")


def run(input_csv: str, output_csv: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.read_csv(input_csv)
    preds = model.predict(df)
    df["prediction"] = preds
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    run("data/input.csv", "data/output_scored.csv")
