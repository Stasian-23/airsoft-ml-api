"""Эндпоинт /predict/upload — текст объявления и фото загружаемые с диска."""
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from ..auth import require_api_key
from ..schemas import ErrorResponse, FailedPhoto, ObjectPrediction, PredictResponse
from ..settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_CONTENT_TYPES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
})
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ


@router.post(
    "/predict/upload",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"description": "Validation error"},
        503: {"model": ErrorResponse},
    },
    tags=["Prediction"],
    summary="Классифицировать товары — загрузить фото с компьютера",
    description=(
        "Принимает текст объявления и фотографии загруженные с компьютера.\n\n"
        "Передайте поле **post_id**, опциональный **text**, и от 1 до 10 файлов "
        "в поле **photos** (multipart/form-data).\n\n"
        "Поддерживаемые форматы: JPEG, PNG, WebP. Максимальный размер файла: 10 МБ."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["post_id"],
                        "properties": {
                            "post_id": {
                                "type": "string",
                                "description": "Идентификатор объявления, например: post-001",
                            },
                            "text": {
                                "type": "string",
                                "description": "Текст объявления (необязательно, но рекомендуется)",
                            },
                            "photos": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Файлы изображений (до 10 штук, JPEG/PNG/WebP)",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def predict_upload(
    request: Request,
    post_id: str = Form(
        ...,
        min_length=1,
        max_length=256,
        description="Идентификатор объявления, например: post-001",
    ),
    text: str | None = Form(
        None,
        max_length=10_000,
        description="Текст объявления (необязательно, но рекомендуется)",
    ),
    photos: list[UploadFile] = File(
        default=[],
        description="Файлы изображений (до 10 штук, поле 'photos')",
    ),
    _: str = Depends(require_api_key),
) -> PredictResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модели не загружены. Проверьте логи сервера.",
        )

    has_text = bool(text and text.strip())
    has_photos = any(f.filename for f in photos)

    if not has_text and not has_photos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Укажите текст объявления или загрузите хотя бы одно фото.",
        )

    settings = get_settings()
    valid_uploads = [f for f in photos if f.filename]

    if len(valid_uploads) > settings.max_images_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Слишком много фото. Максимум {settings.max_images_per_request}.",
        )

    images: dict[str, bytes] = {}
    failed_photos: list[dict] = []

    for idx, upload in enumerate(valid_uploads):
        photo_id = f"photo_{idx + 1}"
        ct = upload.content_type or ""

        if ct and ct not in _ALLOWED_CONTENT_TYPES:
            failed_photos.append({
                "photo_id": photo_id,
                "reason": f"Неподдерживаемый тип файла: {ct!r}. Используйте JPEG, PNG или WebP.",
            })
            continue

        try:
            data = await upload.read()
            if not data:
                failed_photos.append({"photo_id": photo_id, "reason": "Файл пустой."})
                continue
            if len(data) > _MAX_FILE_SIZE:
                failed_photos.append({
                    "photo_id": photo_id,
                    "reason": f"Файл слишком большой ({len(data) // 1024} КБ). Максимум 10 МБ.",
                })
                continue
            images[photo_id] = data
        except Exception as exc:
            logger.warning("Ошибка чтения файла %s: %s", photo_id, exc)
            failed_photos.append({"photo_id": photo_id, "reason": f"Ошибка чтения: {exc}"})

    try:
        result = predictor.predict_with_bytes(
            post_id=post_id,
            text=text,
            images=images,
            confidence_threshold=settings.image_confidence_threshold,
        )
    except Exception as exc:
        logger.exception("Ошибка предсказания для поста %s", post_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка классификации: {exc}",
        ) from exc

    all_failed = failed_photos + result.get("failed_photos", [])
    return PredictResponse(
        post_id=result["post_id"],
        predictions=[ObjectPrediction(**p) for p in result["predictions"]],
        failed_photos=[FailedPhoto(**f) for f in all_failed],
        processing_time_ms=result["processing_time_ms"],
    )
