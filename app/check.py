import pandas as pd
import joblib
from date_time import DateTimeFeatures

# Load pipeline
model = joblib.load("model/fraud_detection_calibrated_pipeline.joblib")

sample = {
    "Transaction_ID": 12345,
    "User_ID": 987,
    "Timestamp": "2025-11-28 14:22:00",

    "Transaction_Amount": 5595,
    "Account_Balance": 450,
    "Daily_Transaction_Count": 3,
    "Avg_Transaction_Amount_7d": 20,
    "Failed_Transaction_Count_7d": 3,
    "Card_Age": 300,
    "Transaction_Distance": 12.4,

    "Transaction_Type": "Online",
    "Device_Type": "Mobile",
    "Authentication_Method": "OTP",
    "Card_Type": "Credit",
    "Is_Weekend": 0,

    "Location": "Mumbai",
    "Merchant_Category": "Electronics",

    "IP_Address_Flag": 0,
    "Previous_Fraudulent_Activity": 0
}

# Convert dict → DataFrame (model expects DataFrame)
sample_df = pd.DataFrame([sample])

pred_proba = model.predict_proba(sample_df)[0, 1]
pred_class = model.predict(sample_df)[0]

print("\n--- Prediction Result ---")
print("Fraud Probability:", pred_proba)
print("Fraud Label:", pred_class)
print("\nSUCCESS: Pipeline is working!")
