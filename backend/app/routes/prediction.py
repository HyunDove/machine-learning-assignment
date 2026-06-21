from flask import Blueprint, request, jsonify
from ..models.schemas import PredictionRequest
from ..services.prediction_service import prediction_service
from ..errors import ApiError
from pydantic import ValidationError

bp = Blueprint("prediction", __name__)


@bp.route("/", methods=["POST"])
def predict():
    try:
        body = PredictionRequest(**request.get_json())
    except ValidationError as e:
        raise ApiError(400, str(e))

    result = prediction_service.predict(body)
    return jsonify(result.model_dump()), 200
