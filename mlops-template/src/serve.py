import pandas as pd
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.schemas import PredictRequest, PredictResponse
from src.utils import load_model, predict_df, measure_latency

app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0",
    description="Real-time inference API with optional MLflow registry.",
)


@app.get("/healthz")
def health():
    try:
        _ = load_model()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
@measure_latency
def predict(payload: PredictRequest):
    model = load_model()
    df = pd.DataFrame([r.model_dump() for r in payload.records])
    y_pred, proba = predict_df(model, df)
    return PredictResponse(predictions=y_pred, probabilities=proba)
