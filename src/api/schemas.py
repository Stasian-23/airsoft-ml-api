"""Pydantic-схемы запросов и ответов API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class PhotoInput(BaseModel):
    photo_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Идентификатор фото",
        examples=["photo_1"],
    )
    url: str = Field(
        ...,
        description="Прямая ссылка на изображение (http/https)",
        examples=["https://example.com/photo.jpg"],
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL должен начинаться с http:// или https://")
        if len(v) > 2048:
            raise ValueError("URL слишком длинный (максимум 2048 символов)")
        return v


class PredictRequest(BaseModel):
    post_id: str = Field(
        ..., min_length=1, max_length=256,
        description="Уникальный ID объявления",
        examples=["post-001"],
    )
    text: str | None = Field(
        None, max_length=10_000,
        description="Текст объявления (до 10 000 символов).",
        examples=["Продаю АК-74 LCT и тактический жилет. 20000р, Москва."],
    )
    photos: list[PhotoInput] | None = Field(
        None, max_length=10,
        description="Фото товара по URL (до 10 штук).",
    )

    @model_validator(mode="after")
    def require_text_or_photos(self) -> "PredictRequest":
        if not (self.text and self.text.strip()) and not self.photos:
            raise ValueError("Укажите текст объявления или хотя бы одну ссылку на фото")
        return self


class ObjectPrediction(BaseModel):
    object_id: str = Field(..., description="Порядковый номер товара в объявлении")
    category: str = Field(..., description="Общая категория")
    subcategory: str = Field(..., description="Подкатегория")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность от 0.0 до 1.0")
    photo_ids: list[str] = Field(..., description="Фото, на которых виден этот товар")
    source: str = Field(
        ...,
        description="Источник: 'text' — из текста, 'image' — с фото, 'both' — оба",
        examples=["text"],
    )


class FailedPhoto(BaseModel):
    photo_id: str
    reason: str


class PredictResponse(BaseModel):
    post_id: str
    predictions: list[ObjectPrediction]
    failed_photos: list[FailedPhoto] = Field(
        default_factory=list,
        description="Фото, которые не удалось загрузить или классифицировать",
    )
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    image_classes: int
    text_categories: int


class ErrorResponse(BaseModel):
    detail: str
