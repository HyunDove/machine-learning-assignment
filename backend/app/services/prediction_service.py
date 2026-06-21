from ..models.schemas import PredictionRequest, PredictionResponse
from ..utils.model_loader import get_regressor, get_classifier
import pandas as pd

# LabelEncoder 알파벳 순 인코딩 (preprocessing.py와 동일)
_ENCODINGS = {
    "person_home_ownership": {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3},
    "loan_intent": {
        "DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5,
    },
    "loan_grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
    "cb_person_default_on_file": {"N": 0, "Y": 1},
}

_FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]
_FEATURES_NO_STATUS = [f for f in _FEATURES if f != "loan_status"]


class PredictionService:
    def predict(self, req: PredictionRequest) -> PredictionResponse:
        data = req.model_dump()
        for col, mapping in _ENCODINGS.items():
            data[col] = mapping[data[col]]

        regressor = get_regressor()
        classifier = get_classifier()

        df_reg = pd.DataFrame([data])[_FEATURES]
        df_cls = pd.DataFrame([data])[_FEATURES_NO_STATUS]

        rate = float(regressor.predict(df_reg)[0])
        prob = float(classifier.predict_proba(df_cls)[0][1])
        label = "부도" if prob >= 0.5 else "정상"

        return PredictionResponse(
            loan_int_rate=round(rate, 4),
            loan_status_prob=round(prob, 4),
            loan_status_label=label,
        )


prediction_service = PredictionService()
