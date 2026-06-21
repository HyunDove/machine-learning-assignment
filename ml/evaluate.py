import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/credit_risk_cleaned.csv")

FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
]


def evaluate():
    df = pd.read_csv(PROCESSED_PATH)
    regressor = joblib.load(os.path.join(MODEL_DIR, "loan_rate_model.pkl"))

    from sklearn.model_selection import train_test_split
    X = df[FEATURES]
    y = df["loan_int_rate"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    evaluate()
