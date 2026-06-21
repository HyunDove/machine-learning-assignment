import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/credit_risk_dataset.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/credit_risk_cleaned.csv")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["loan_int_rate"]).copy()
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    # 이상치 제거 (나이 > 100, 재직기간 > 60)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]

    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade",
                "cb_person_default_on_file"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def save_processed(df: pd.DataFrame):
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    df = load_raw()
    df = preprocess(df)
    save_processed(df)
    print(f"저장 완료: {PROCESSED_PATH} ({len(df)}행)")
