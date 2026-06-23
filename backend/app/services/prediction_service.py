from ..models.schemas import PredictionRequest, PredictionResponse
from ..utils.model_loader import get_regressor
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
    "loan_intent", "loan_grade", "loan_amnt",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]


class PredictionService:
    def predict(self, req: PredictionRequest) -> PredictionResponse:
        data = req.model_dump()
        for col, mapping in _ENCODINGS.items():
            data[col] = mapping[data[col]]

        regressor = get_regressor()
        df = pd.DataFrame([data])[_FEATURES]
        rate = float(regressor.predict(df)[0])

        return PredictionResponse(loan_int_rate=round(rate, 4))


prediction_service = PredictionService()
