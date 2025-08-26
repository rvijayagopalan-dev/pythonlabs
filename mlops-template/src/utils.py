import os
import time
import numpy as np
import pandas as pd
from joblib import load
from prometheus_client import Counter, Histogram
import mlflow

REQUEST_COUNTER = Counter("inference_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "Latency of inference requests")

MODEL_CACHE = {"model": None}


def load_model():
    if MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"]

    use_mlflow = os.getenv("USE_MLFLOW", "true").lower() == "true"
    if use_mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        model_name = os.getenv("MODEL_NAME", "iris-classifier")
        model_stage = os.getenv("MODEL_STAGE", "Staging")
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        MODEL_CACHE["model"] = model
        return model
    else:
        path = os.getenv("LOCAL_MODEL_PATH", "models/latest/model.pkl")
        model = load(path)
        MODEL_CACHE["model"] = model
        return model


def predict_df(model, df: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(df)
        proba = model.predict_proba(df)
    else:
        y_pred = model.predict(df)
        proba = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)
        except Exception:
            proba = None

    y_pred = y_pred.tolist() if isinstance(y_pred, (np.ndarray, list)) else list(y_pred)
    if proba is not None and hasattr(proba, "tolist"):
        proba = proba.tolist()
    return y_pred, proba


def measure_latency(func):
    def wrapper(*args, **kwargs):
        REQUEST_COUNTER.inc()
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            REQUEST_LATENCY.observe(time.perf_counter() - start)
    return wrapper
