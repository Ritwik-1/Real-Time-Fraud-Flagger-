import streamlit as st
from kafka import KafkaConsumer
import json

st.title("Real-Time Fraud Dashboard")

consumer = KafkaConsumer(
    "fraud-alerts",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    group_id="fraud-dashboard"
)

alerts = []

for msg in consumer:
    alerts.append(msg.value)
    st.write(alerts)


# streamlit run dashboard/dashboard.py