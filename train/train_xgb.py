# pip install xgboost scikit-learn category_encoders pandas joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import category_encoders as ce
import joblib
import numpy as np

df = pd.read_csv("data/synthetic_fraud_dataset.csv", parse_dates=["Timestamp"])

# 2. drop / basic cleaning
df = df.drop(columns=["Transaction_ID"])            
df = df.drop(columns=["User_ID"])

# print(df.columns)
# exit(0)

df = df.drop(columns=["Risk_Score"])

# 3. feature engineer timestamp
df["hour"] = df["Timestamp"].dt.hour
df["dow"]  = df["Timestamp"].dt.dayofweek
df["month"]= df["Timestamp"].dt.month
df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
df = df.drop(columns=["Timestamp"])

# 4. define target and features
y = df["Fraud_Label"]
X = df.drop(columns=["Fraud_Label"])

# 5. feature groups
numeric_feats = ["Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
                 "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
                 "Card_Age", "Transaction_Distance", "hour", "dow", "is_night", "month"]

# choose categorical features
cat_low_card = ["Transaction_Type", "Device_Type", "Authentication_Method", "Card_Type", "Is_Weekend"]  # small
cat_high_card = ["Location", "Merchant_Category"]  # treat with target/frequency encoding
binary_feats = ["IP_Address_Flag", "Previous_Fraudulent_Activity"]  # already 0/1

# 6. train/test split (stratify for class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 7. build preprocessing pipeline
# numeric imputer
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# low-cardinality one-hot
cat_low_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# high-cardinality: target encoder (fit on train only)
# we'll handle it outside ColumnTransformer because target encoder needs y during fit
# so build a simple pipeline wrapper
high_card_encoder = ce.TargetEncoder(cols=cat_high_card, smoothing=0.2)

# apply numeric + low-cat with ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, numeric_feats),
    ("lowcat", cat_low_pipeline, cat_low_card)
], remainder="passthrough")  # keep binary and other columns

# 8. full pipeline (we'll insert target encoder manually)
# Fit target encoder on train
X_train_te = high_card_encoder.fit_transform(X_train, y_train)
X_test_te = high_card_encoder.transform(X_test)

# Now apply preprocessor
X_train_pre = preprocessor.fit_transform(X_train_te)
X_test_pre = preprocessor.transform(X_test_te)

# 9. XGBoost - handle class imbalance
neg = (y_train==0).sum()
pos = (y_train==1).sum()
scale_pos_weight = neg / max(1, pos)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    # use_label_encoder=False,
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    n_jobs=6,
    random_state=42
)

print("PREPROCESSING DONE FULLY")
# print(X_train_pre.shape)
# print(y_train.shape)
# print(X_test_pre.shape)

# print("The first few rows:-")
# print(X_train_pre[:5])
# print(y_train[:5])
# print(X_test_pre[:5])

# print("Encoding is done right only, and preprocessing is okay")
# exit(0)

model.fit(
    X_train_pre, y_train,
    eval_set=[(X_train_pre, y_train), (X_test_pre, y_test)],
    # callbacks=[EarlyStopping(rounds=30, save_best=True)],
    verbose=50
)

# 10. evaluate
y_proba = model.predict_proba(X_test_pre)[:,1]
y_pred = (y_proba >= 0.5).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# 11. save pipeline + encoders + model
# joblib.dump({
#     "preprocessor": preprocessor,
#     "target_encoder": high_card_encoder,
#     "model": model,
#     "numeric_feats": numeric_feats,
#     "cat_low_card": cat_low_card,
#     "cat_high_card": cat_high_card
# }, "models/fraud_xgb_pipeline.joblib")

# PREPROCESSING DONE FULLY
# [0]     validation_0-auc:0.81881        validation_1-auc:0.80939
# [50]    validation_0-auc:0.84533        validation_1-auc:0.81023
# [99]    validation_0-auc:0.85411        validation_1-auc:0.80919
#               precision    recall  f1-score   support

#            0       0.85      1.00      0.92      6787
#            1       1.00      0.62      0.76      3213

#     accuracy                           0.88     10000
#    macro avg       0.92      0.81      0.84     10000
# weighted avg       0.90      0.88      0.87     10000

# ROC-AUC: 0.8091882005982493

