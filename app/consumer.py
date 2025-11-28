from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
import pandas as pd
import threading

def start_consumer():
    print("Starting Kafka Fraud Detector...")

    pipeline = joblib.load("model/fraud_xgb_pipeline.joblib")

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

    def loop():
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

            producer.send("fraud-alerts", alert)
            print("ALERT:", alert)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

if __name__ == "__main__":
    start_consumer()
