from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col="Timestamp"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = pd.to_datetime(X[self.col], errors="coerce")
        X["hour"] = X[self.col].dt.hour
        X["day_of_week"] = X[self.col].dt.dayofweek
        X["month"] = X[self.col].dt.month
        X["day"] = X[self.col].dt.day
        return X.drop(columns=[self.col])