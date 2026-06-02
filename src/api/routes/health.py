from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter()
VERSION = "2.0.0"


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    models_ok = predictor is not None
    return HealthResponse(
        status="ok" if models_ok else "degraded",
        version=VERSION,
        models_loaded=models_ok,
        image_classes=len(predictor.image_clf.classes) if models_ok else 0,
        text_categories=len(predictor.text_clf.id2label) if models_ok else 0,
    )
