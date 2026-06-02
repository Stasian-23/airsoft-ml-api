"""
Основной ML-конвейер: входные данные API → предсказание → ответ.

Порядок работы (текст в приоритете):
  1. Классифицировать каждое фото независимо (EfficientNet-B0).
  2. Извлечь все товары из текста объявления по словарю ключевых слов.
  3. Объединить результаты: текст — основной список, фото — подтверждение или дополнение.
  4. Запасной вариант: если ничего не найдено — применить rubert-tiny2.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .config import SUBCATEGORY_LABELS, SUBCATEGORY_TO_CATEGORY
from .image_classifier import ImageClassifier
from .object_extractor import extract_objects_from_text
from .text_classifier import TextClassifier

logger = logging.getLogger(__name__)

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Минимальная уверенность rubert-tiny2 для возврата результата
_TEXT_CLF_THRESHOLD = 0.55


class AirsoftPredictor:
    def __init__(self, models_dir: str | Path = "models") -> None:
        models_dir = Path(models_dir)

        image_path = models_dir / "image_classifier.pt"
        text_dir = models_dir / "text_classifier"

        if not image_path.exists():
            raise RuntimeError(
                f"Модель изображений не найдена: {image_path}. "
                "Запустите training/train_image.py"
            )
        if not text_dir.exists():
            raise RuntimeError(
                f"Текстовая модель не найдена: {text_dir}. "
                "Запустите training/train_text.py"
            )

        self.image_clf = ImageClassifier(image_path)
        self.text_clf = TextClassifier(text_dir)
        logger.info(
            "Модели загружены: изображения(%d классов), текст(%d классов)",
            len(self.image_clf.classes),
            len(self.text_clf.id2label),
        )

    # Публичные методы

    async def predict(
        self,
        post_id: str,
        text: str | None,
        photos: list[dict],
        image_timeout: int = 10,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Принимает список {'photo_id': str, 'url': str}, скачивает фото и классифицирует."""
        t_start = time.perf_counter()
        raw_images, failed_photos = await self._download_all(photos, image_timeout)
        return self._run_pipeline(
            post_id, text, raw_images, failed_photos, confidence_threshold, t_start
        )

    def predict_with_bytes(
        self,
        post_id: str,
        text: str | None,
        images: dict[str, bytes],
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Тот же конвейер, но принимает байты напрямую (без загрузки по URL)."""
        return self._run_pipeline(
            post_id, text, images, [], confidence_threshold, time.perf_counter()
        )

    # Конвейер

    def _run_pipeline(
        self,
        post_id: str,
        text: str | None,
        raw_images: dict[str, bytes],
        failed_photos: list[dict],
        confidence_threshold: float,
        t_start: float,
    ) -> dict[str, Any]:
        # Шаг 1 — классификация фото
        photo_preds: list[dict] = []
        for photo_id, image_bytes in raw_images.items():
            try:
                subcat, conf = self.image_clf.predict(image_bytes)
                if conf >= confidence_threshold:
                    photo_preds.append({"photo_id": photo_id, "subcategory": subcat, "confidence": conf})
                else:
                    failed_photos.append({
                        "photo_id": photo_id,
                        "reason": f"Низкая уверенность ({conf:.2f} < {confidence_threshold})",
                    })
            except Exception as exc:
                logger.warning("Ошибка классификации фото %s: %s", photo_id, exc)
                failed_photos.append({"photo_id": photo_id, "reason": f"Ошибка классификации: {exc}"})

        # Шаг 2 — извлечение товаров из текста
        text_items = extract_objects_from_text(text or "")

        # Шаг 3 — объединение (текст в приоритете)
        merged = self._merge(text_items, photo_preds)

        # Шаг 4 — запасной вариант через rubert-tiny2
        if not merged and text and text.strip():
            category, conf = self.text_clf.predict(text)
            if conf >= _TEXT_CLF_THRESHOLD:
                merged.append({
                    "category": category,
                    "subcategory": "Не определено",
                    "confidence": round(conf, 4),
                    "photo_ids": [],
                    "source": "text",
                })

        predictions = [{"object_id": str(i + 1), **obj} for i, obj in enumerate(merged)]

        return {
            "post_id": post_id,
            "predictions": predictions,
            "failed_photos": failed_photos,
            "processing_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    @staticmethod
    def _merge(
        text_items: list[dict],
        photo_preds: list[dict],
    ) -> list[dict]:
        """
        Объединяет результаты текстового и визуального анализа.

        Алгоритм:
        - Товары из текста формируют основной список (source='text').
        - Фото, совпавшее с уже найденным товаром, повышает уверенность и меняет source на 'both'.
        - Фото с товаром, которого нет в тексте, добавляется как source='image'.
        - Итоговый список отсортирован по уверенности (убывание).
        """
        objects: dict[str, dict] = {}

        for item in text_items:
            sc = item["subcategory"]
            objects[sc] = {
                "category": item["category"],
                "subcategory": SUBCATEGORY_LABELS.get(sc, sc),
                "confidence": round(item["confidence"], 4),
                "photo_ids": [],
                "source": "text",
            }

        for photo in photo_preds:
            sc = photo["subcategory"]
            conf = photo["confidence"]

            if sc in objects:
                objects[sc]["photo_ids"].append(photo["photo_id"])
                objects[sc]["confidence"] = round(max(objects[sc]["confidence"], conf), 4)
                objects[sc]["source"] = "both"
            else:
                objects[sc] = {
                    "category": SUBCATEGORY_TO_CATEGORY.get(sc, "Аксессуары и Запчасти"),
                    "subcategory": SUBCATEGORY_LABELS.get(sc, sc),
                    "confidence": round(conf, 4),
                    "photo_ids": [photo["photo_id"]],
                    "source": "image",
                }

        result = list(objects.values())
        result.sort(key=lambda x: x["confidence"], reverse=True)
        return result

    # Загрузка изображений

    async def _download_all(
        self, photos: list[dict], timeout: int
    ) -> tuple[dict[str, bytes], list[dict]]:
        if not photos:
            return {}, []

        tasks = {p["photo_id"]: self._download(p["url"], timeout) for p in photos}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        successful: dict[str, bytes] = {}
        failed: list[dict] = []
        for photo_id, result in zip(tasks.keys(), results):
            if isinstance(result, bytes):
                successful[photo_id] = result
            else:
                reason = str(result) if result else "Неизвестная ошибка загрузки"
                failed.append({"photo_id": photo_id, "reason": reason})
                logger.warning("Не удалось загрузить фото %s: %s", photo_id, reason)

        return successful, failed

    @staticmethod
    async def _download(url: str, timeout: int) -> bytes:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=_DOWNLOAD_HEADERS,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if not any(t in content_type for t in ("image/", "application/octet-stream")):
                raise ValueError(f"Ответ сервера не является изображением: {content_type!r}")
            return resp.content
