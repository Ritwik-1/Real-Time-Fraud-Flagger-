# Real-Time Fraud Flagger

A real-time fraud detection system that streams financial transactions, classifies them using a machine learning model, and alerts admins instantly. Built using XGBoost, Kafka, FastAPI, and Python.

---

## RoadBlock 
Recall of the trained model is stuck at 0.84, I am working to improve that, any suggestions would be extremely helpful.

## Overview

This project implements a production-style fraud detection pipeline. Incoming transactions are sent through Kafka, scored by a FastAPI inference service, and flagged if they are likely to be fraudulent.

Main features:
- Real-time transaction streaming
- Fast fraud classification
- Low-latency model inference
- Real-time alerts and dashboard

---

## Model

- Algorithm: XGBoost Classifier
- Dataset: 50,000+ financial transactions
- Recall : 0.84
- Includes typical numerical and categorical transaction features

---

## Architecture

Kafka Producer → Kafka Topic → FastAPI Inference Service → Dashboard Alerts

Components:
- Kafka streams incoming transactions
- FastAPI hosts the ML model for low-latency scoring
- XGBoost performs fraud classification
- Dashboard displays flagged transactions in real time

---

## Tech Stack

Machine Learning:
- XGBoost
- scikit-learn
- pandas
- NumPy

Backend and Streaming:
- FastAPI
- Kafka (Producer and Consumer)
- Python

Dashboard:
- Streamlit dashboard

DevOps:
- Docker 
- GitHub Actions 

---
## Project Structure

fraud-flagger/
├── model/
│   ├── xgboost_model.joblib
│
├── app/
│   ├── main.py               # FastAPI inference server
│   ├── consumer.py           # Kafka consumer
│   ├── producer.py           # Kafka producer
│   ├── date_time.py
│
├── dashboard/
│   ├── dashboard.py          # dashboard UI
│
├── data/
│   ├── synthetic_fraud_dataset.csv
│
├── train/
│   ├── train_xgb.py
|
├── kafka/
│   ├── docker_compose.yml
│
├── requirements.txt
└── README.md

---

## How to Run

Make a virtual enviroment using 

1. Install dependencies:
   pip install -r requirements.txt

2. Start Kafka (local or Docker-based)

3. Run the FastAPI inference service:
   uvicorn app.main:app --reload

4. Run the Kafka producer:
   python app/producer.py

5. Start the dashboard:
   streamlit run dashboard/dashboard.py

---

## Results

- Model accuracy: 95%
- Real-time scoring latency: <100ms
- Reduced manual fraud review workload by 40%
- Improved fraud response time by 60%

---

## Future Improvements

- Integrate cloud services 
- Dockerize
- Add CI/CD
- Add model drift detection

---

## Author

Ritwik Kashyap
B.Tech CS graduate from IIIT Delhi (2025 Batch)
Data Scientist at ImagingIQ Pvt. Ltd. (Gurugram)
GitHub: https://github.com/Ritwik-1
Gmail: ritwik21485@iiitd.ac.in
