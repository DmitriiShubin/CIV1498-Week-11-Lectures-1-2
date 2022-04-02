from fastapi import APIRouter

from .endpoints.fraud_detection_api import router as triage_router

router = APIRouter()
router.include_router(triage_router, tags=["save_audio_data"])
