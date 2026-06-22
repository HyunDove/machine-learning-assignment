import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/credit_risk_cleaned.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
]
RATE_TARGET = "loan_int_rate"
STATUS_TARGET = "loan_status"

# compress=3: 무손실 압축, 파일 크기 약 5~6배 감소
_COMPRESS = 3


def train():
    df = pd.read_csv(PROCESSED_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 회귀 — 대출 금리 (RandomForestRegressor)
    X = df[FEATURES]
    y_rate = df[RATE_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y_rate, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)
    joblib.dump(regressor, os.path.join(MODEL_DIR, "loan_rate_model.pkl"), compress=_COMPRESS)

    # 분류 — 대출 부도 (XGBClassifier)
    features_no_status = [f for f in FEATURES if f != STATUS_TARGET]
    X_cls = df[features_no_status]
    y_cls = df[STATUS_TARGET]
    X_train_c, _, y_train_c, _ = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    classifier = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    classifier.fit(X_train_c, y_train_c)
    joblib.dump(classifier, os.path.join(MODEL_DIR, "loan_status_model.pkl"), compress=_COMPRESS)

    print("모델 저장 완료")
    return regressor, classifier, X_test, y_test


if __name__ == "__main__":
    train()
