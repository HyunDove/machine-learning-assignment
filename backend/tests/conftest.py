import os
import pytest
from app import create_app

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["MODEL_PATH"] = os.path.join(ROOT, "models", "loan_rate_model.pkl")
    app.config["CLASSIFIER_PATH"] = os.path.join(ROOT, "models", "loan_status_model.pkl")
    with app.test_client() as c:
        yield c
