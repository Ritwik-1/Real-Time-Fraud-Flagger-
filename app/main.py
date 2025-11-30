from fastapi import FastAPI
import joblib
import pandas as pd
from app.consumer import start_consumer

# START CONSUMER HERE
start_consumer() 

app = FastAPI()

pipeline = joblib.load("model/fraud_detection_calibrated_pipeline.joblib")

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    proba = pipeline.predict_proba(df)[0][1]
    fraud = int(proba > 0.5)
    return {"fraud": fraud, "probability": proba}

@app.get("/")
def root():
    return {"status": "Fraud Model API Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
