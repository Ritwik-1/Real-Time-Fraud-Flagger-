from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

pipeline = joblib.load("model/fraud_xgb_pipeline.joblib")

from app.consumer import start_background_consumer
start_background_consumer()

@app.post("/predict")
def predict(transaction : dict):
    df = pd.DataFrame([transaction])
    proba = pipeline.predict_proba(df)[0][1]
    fraud = int(proba > 0.5)
    return {
        "fraud": fraud,
        "probability": proba
    }

@app.get("/")
def root():
    return {"message":"Fraud API running"}


# uvicorn app.main:app --reload

