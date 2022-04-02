import io
import pickle
import time

import boto3
import numpy as np
from core.config import MODEL_PATH
from core.models.input import MessageInput
from core.models.output import MessageOutput
from fastapi import APIRouter

router = APIRouter()


model = pickle.load(open(MODEL_PATH, 'rb'))


@router.post("/fraud_detection_api", response_model=MessageOutput, tags=["fraud_detection_api"])
def fraud_detection_api(inputs: MessageInput):
    """
    Making predictions

    """

    # unpack the payload
    input_vector = np.zeros((1, 32))
    field_names = inputs.__fields__.keys()
    for index, field_name in enumerate(field_names):
        input_vector[0, index] = getattr(inputs, field_name)

    prediction = model.predict(input_vector, raw_score=True)
    prediction = 1 / (1 + np.exp(-1 * prediction))
    prediction = prediction[0]

    return {"fraud": 1 if prediction >= 0.5 else 0, "probability": prediction}
