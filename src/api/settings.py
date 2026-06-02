"""Настройки приложения из .env файла."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    api_keys: str = Field(..., description="API ключи через запятую")
    models_dir: str = "models"
    image_confidence_threshold: float = 0.25
    image_download_timeout: int = 10
    max_images_per_request: int = 10
    rate_limit: str = "30/minute"

    # Вычисляется один раз при создании объекта
    _api_keys_set: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def _parse_api_keys(self) -> "Settings":
        object.__setattr__(
            self,
            "_api_keys_set",
            frozenset(k.strip() for k in self.api_keys.split(",") if k.strip()),
        )
        return self

    @property
    def api_keys_set(self) -> frozenset[str]:
        return self._api_keys_set


@lru_cache
def get_settings() -> Settings:
    return Settings()
