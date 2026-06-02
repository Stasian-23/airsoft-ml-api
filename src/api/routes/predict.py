"""Эндпоинт /predict — текст объявления и ссылки на фото (URL)."""
import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from ..auth import require_api_key
from ..schemas import ErrorResponse, FailedPhoto, ObjectPrediction, PredictRequest, PredictResponse
from ..settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_EXAMPLES = {
    "pistol": {
        "summary": "Пистолет (один товар)",
        "value": {
            "post_id": "post-001",
            "text": "Продаю пистолет WE Glock 17 GBB. 2 магазина, в отличном состоянии. 5000р, Москва.",
        },
    },
    "multiobject": {
        "summary": "Несколько товаров в одном объявлении",
        "value": {
            "post_id": "post-002",
            "text": (
                "Продаю комплект: АК-74 LCT (сток, 10 игр) + тактический жилет Wartech "
                "+ подсумок МОЛЛИ. Всё в хорошем состоянии. 25000р, Екатеринбург."
            ),
        },
    },
    "text_and_photos": {
        "summary": "Текст и ссылки на фото",
        "value": {
            "post_id": "post-003",
            "text": "Продам страйкбольный шлем FAST SF. Б/у, хорошее состояние. 8000р.",
            "photos": [
                {
                    "photo_id": "photo_1",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                }
            ],
        },
    },
    "sniper": {
        "summary": "Снайперская винтовка",
        "value": {
            "post_id": "post-004",
            "text": "VSR-10 от Snow Wolf, болтовая снайперская винтовка. Прицел в комплекте. 7000р, Москва.",
        },
    },
    "short_post": {
        "summary": "Короткое объявление без ключевых слов",
        "value": {
            "post_id": "post-005",
            "text": "продам кавер в мультике на 6б47 от стич профи\n2500\nЕкб\nПересыл",
        },
    },
}


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"description": "Validation error"},
        503: {"model": ErrorResponse},
    },
    tags=["Prediction"],
    summary="Классифицировать товары в объявлении",
    description=(
        "Принимает текст объявления и ссылки на фото (URL).\n\n"
        "Возвращает список найденных товаров — каждый с категорией, подкатегорией "
        "и уверенностью модели.\n\n"
        "Выберите готовый пример из выпадающего списка рядом с полем запроса."
    ),
)
async def predict(
    request: Request,
    body: Annotated[PredictRequest, Body(openapi_examples=_EXAMPLES)],
    _: str = Depends(require_api_key),
) -> PredictResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модели не загружены. Проверьте логи сервера.",
        )

    settings = get_settings()
    if body.photos and len(body.photos) > settings.max_images_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Слишком много фото. Максимум {settings.max_images_per_request}.",
        )

    photos_payload = (
        [{"photo_id": p.photo_id, "url": p.url} for p in body.photos]
        if body.photos else []
    )

    try:
        result = await predictor.predict(
            post_id=body.post_id,
            text=body.text,
            photos=photos_payload,
            image_timeout=settings.image_download_timeout,
            confidence_threshold=settings.image_confidence_threshold,
        )
    except Exception as exc:
        logger.exception("Ошибка предсказания для поста %s", body.post_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка классификации: {exc}",
        ) from exc

    return PredictResponse(
        post_id=result["post_id"],
        predictions=[ObjectPrediction(**p) for p in result["predictions"]],
        failed_photos=[FailedPhoto(**f) for f in result.get("failed_photos", [])],
        processing_time_ms=result["processing_time_ms"],
    )
