import os

from starlette.datastructures import CommaSeparatedStrings

ALLOWED_HOSTS = CommaSeparatedStrings(os.getenv("ALLOWED_HOSTS", ""))
API_V1_STR = "/api/v1"
PROJECT_NAME = "fraud_detection_api"

MODEL_PATH = './core/logic/ml_models/1_lgb_model.pkl'
