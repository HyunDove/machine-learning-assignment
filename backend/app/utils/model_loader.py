import joblib
from flask import current_app

_regressor = None


def get_regressor():
    global _regressor
    if _regressor is None:
        _regressor = joblib.load(current_app.config["MODEL_PATH"])
    return _regressor
