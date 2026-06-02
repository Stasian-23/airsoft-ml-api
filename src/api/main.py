"""
Точка входа FastAPI приложения.

При запуске (lifespan): загружает ML модели один раз, сохраняет в app.state.predictor.
Ограничение запросов через slowapi (учитывает X-Forwarded-For).
Структурированное JSON-логирование через structlog.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .routes.predict_upload import router as predict_upload_router
from .settings import get_settings

# Логирование
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


# Жизненный цикл приложения (заменяет устаревшие on_event startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from ..ml.predictor import AirsoftPredictor

    try:
        app.state.predictor = AirsoftPredictor(settings.models_dir)
        log.info("models_loaded", models_dir=settings.models_dir)
    except Exception as exc:
        log.error("models_load_failed", error=str(exc))
        app.state.predictor = None

    yield

    log.info("shutdown")


_DESCRIPTION = """
## Классификатор страйкбольных товаров

API определяет **категорию** и **подкатегорию** товаров из объявлений на маркетплейсе.
Поддерживает многообъектные объявления — одно объявление может содержать несколько товаров.

---

### Начало работы

1. Нажмите **Authorize** (вверху справа), вставьте API ключ, нажмите **Authorize → Close**
2. Откройте нужный эндпоинт → **Try it out** → заполните поля → **Execute**

---

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/v1/predict` | Текст + ссылки на фото (URL) |
| POST | `/api/v1/predict/upload` | Текст + фото загружаемые с компьютера |
| GET  | `/api/v1/health` | Статус сервера и моделей |

---

### Формат ответа

Для каждого найденного товара:

- **category** — `Страйкбольное оружие`, `Снаряжение и защита` или `Аксессуары и Запчасти`
- **subcategory** — тип товара: `АК / АКС`, `Пистолет`, `Шлем`, `Тактический жилет` и др.
- **confidence** — уверенность модели от `0.0` до `1.0`
- **photo_ids** — список фото на которых виден данный товар
- **source** — источник: `text`, `image` или `both`
"""

app = FastAPI(
    title="Airsoft ML API",
    description=_DESCRIPTION,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Убираем несериализуемый ctx из ошибок Pydantic v2
    errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )


app.include_router(health_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(predict_upload_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Airsoft ML API v2. Документация: /docs"}
