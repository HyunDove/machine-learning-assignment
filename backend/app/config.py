import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "../models/loan_rate_model.pkl")
    CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "../models/loan_status_model.pkl")
