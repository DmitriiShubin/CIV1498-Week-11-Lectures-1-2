from api.api_v1.api import router as api_router
from core.config import API_V1_STR, PROJECT_NAME
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI(title=PROJECT_NAME)

app.include_router(api_router, prefix=API_V1_STR)


handler = Mangum(app=app)


@app.get("/ping", summary="Check that the service is operational")
def pong():
    """
    Sanity check - this will let the user know that the service is operational.

    It is also used as part of the HEALTHCHECK. Docker uses curl to check that the API service is still running, by exercising this endpoint.

    """
    return {"ping": "pong!"}
