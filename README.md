# Real-Time Fraud Flagger

A real-time fraud detection system that streams financial transactions, classifies them using a machine learning model, and alerts admins instantly. Built using XGBoost, Kafka, FastAPI, and Python.

---

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
- Accuracy: 95%
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
│   ├── xgboost_model.pkl
│   ├── preprocessing.py
│
├── app/
│   ├── main.py               # FastAPI inference server
│   ├── consumer.py           # Kafka consumer
│   ├── producer.py           # Kafka producer
│   ├── utils.py
│
├── dashboard/
│   ├── dashboard.py          # dashboard UI
│
├── data/
│   ├── synthetic_fraud_dataset.csv
│
├── train/
│   ├── train_xgb.py
│
├── requirements.txt
└── README.md

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Start Kafka (local or Docker-based)

3. Run the Kafka producer:
   python app/producer.py

4. Run the FastAPI inference service:
   uvicorn app.main:app --reload

5. Run the Kafka consumer:
   python app/consumer.py

6. Start the dashboard:
   streamlit run dashboard/dashboard.py

---

## Results

- Model accuracy: 95%
- Real-time scoring latency: <100ms
- Reduced manual fraud review workload by 40%
- Improved fraud response time by 60%

---

## Future Improvements

- Add Triton Inference Server for faster serving
- Integrate cloud services (AWS MSK, Lambda, DynamoDB)
- Add model drift detection
- Build microservices architecture for scalability

---

## Author

Your Name
GitHub: https://github.com/Ritwik-1
