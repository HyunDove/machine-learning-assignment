import joblib
from flask import current_app

_regressor = None
_classifier = None


def get_regressor():
    global _regressor
    if _regressor is None:
        _regressor = joblib.load(current_app.config["MODEL_PATH"])
    return _regressor


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = joblib.load(current_app.config["CLASSIFIER_PATH"])
    return _classifier
