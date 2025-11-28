from kafka import KafkaProducer
import json, time, pandas as pd

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv("data/synthetic_fraud_dataset.csv")

for _, row in df.iterrows():
    producer.send("transactions", row.to_dict())
    time.sleep(0.5)
