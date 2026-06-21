from ..models.schemas import PredictionRequest, PredictionResponse
from ..utils.model_loader import get_regressor, get_classifier
import pandas as pd


class PredictionService:
    def predict(self, req: PredictionRequest) -> PredictionResponse:
        df = pd.DataFrame([req.model_dump()])
        regressor = get_regressor()
        classifier = get_classifier()

        rate = float(regressor.predict(df)[0])
        prob = float(classifier.predict_proba(df)[0][1])
        label = "부도" if prob >= 0.5 else "정상"

        return PredictionResponse(
            loan_int_rate=round(rate, 4),
            loan_status_prob=round(prob, 4),
            loan_status_label=label,
        )


prediction_service = PredictionService()
