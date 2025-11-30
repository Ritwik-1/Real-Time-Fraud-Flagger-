from kafka import KafkaProducer
import json, time, pandas as pd

def run_producer():
    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    df = pd.read_csv("data/synthetic_fraud_dataset.csv")

    for _, row in df.iterrows():
        txn = row.to_dict()
        producer.send("transactions", txn)
        print("Sent:", row["Transaction_ID"])
        time.sleep(0.5)

if __name__ == "__main__":
    run_producer()
