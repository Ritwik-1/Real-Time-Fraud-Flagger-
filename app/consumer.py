from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
import pandas as pd
import threading

# Load model pipeline
pipeline = joblib.load("model/fraud_pipeline.joblib")

# Kafka
consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    group_id="fraud-detector"
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)


def fraud_detector_loop():
    print("Kafka Fraud Detector Running...")
    for msg in consumer:
        txn = msg.value

        df = pd.DataFrame([txn])
        proba = pipeline.predict_proba(df)[0][1]
        fraud = int(proba > 0.5)

        alert = {
            "transaction_id": txn.get("Transaction_ID"),
            "fraud": fraud,
            "score": float(proba)
        }

        # send to fraud-alerts topic
        producer.send("fraud-alerts", alert)
        print("ALERT:", alert)


# Background thread starter
def start_background_consumer():
    thread = threading.Thread(target=fraud_detector_loop, daemon=True)
    thread.start()
