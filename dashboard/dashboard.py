import streamlit as st
from kafka import KafkaConsumer
import json

def run_dashboard():
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

if __name__ == "__main__":
    run_dashboard()



# streamlit run dashboard/dashboard.py

# cd kafka
# docker-compose up -d

# docker exec -it $(docker ps -qf "ancestor=wurstmeister/kafka") bash

# kafka-topics.sh --create --topic transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
# kafka-topics.sh --create --topic fraud-alerts --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# uvicorn app.main:app --reload

# python app/producer.py

# streamlit run dashboard/dashboard.py


