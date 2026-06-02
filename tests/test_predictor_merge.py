"""
Тесты логики объединения результатов AirsoftPredictor._merge
(не требуют загрузки ML-моделей).

Запуск: pytest tests/test_predictor_merge.py -v
"""
import pytest

from src.ml.predictor import AirsoftPredictor


class TestMerge:

    def test_text_only(self):
        text_items = [
            {"subcategory": "ak", "category": "Страйкбольное оружие", "confidence": 0.9, "source": "text"},
        ]
        result = AirsoftPredictor._merge(text_items, [])
        assert len(result) == 1
        assert result[0]["source"] == "text"
        assert result[0]["subcategory"] == "АК / АКС"
        assert result[0]["photo_ids"] == []

    def test_photo_confirms_text(self):
        text_items = [
            {"subcategory": "pistol", "category": "Страйкбольное оружие", "confidence": 0.8, "source": "text"},
        ]
        photo_preds = [
            {"photo_id": "photo_1", "subcategory": "pistol", "confidence": 0.9},
        ]
        result = AirsoftPredictor._merge(text_items, photo_preds)
        assert len(result) == 1
        assert result[0]["source"] == "both"
        assert "photo_1" in result[0]["photo_ids"]
        assert result[0]["confidence"] == 0.9

    def test_photo_adds_new_item(self):
        text_items = [
            {"subcategory": "ak", "category": "Страйкбольное оружие", "confidence": 0.9, "source": "text"},
        ]
        photo_preds = [
            {"photo_id": "photo_1", "subcategory": "helmet", "confidence": 0.85},
        ]
        result = AirsoftPredictor._merge(text_items, photo_preds)
        subcats = {r["subcategory"] for r in result}
        assert "АК / АКС" in subcats
        assert "Шлем" in subcats
        helmet = next(r for r in result if r["subcategory"] == "Шлем")
        assert helmet["source"] == "image"

    def test_sorted_by_confidence_desc(self):
        text_items = [
            {"subcategory": "ak",   "category": "Страйкбольное оружие", "confidence": 0.7, "source": "text"},
            {"subcategory": "vest", "category": "Снаряжение и защита",  "confidence": 0.95, "source": "text"},
        ]
        result = AirsoftPredictor._merge(text_items, [])
        assert result[0]["confidence"] >= result[1]["confidence"]

    def test_empty_inputs(self):
        assert AirsoftPredictor._merge([], []) == []

    def test_photo_only(self):
        result = AirsoftPredictor._merge([], [
            {"photo_id": "photo_1", "subcategory": "pistol", "confidence": 0.88},
        ])
        assert len(result) == 1
        assert result[0]["source"] == "image"
        assert result[0]["subcategory"] == "Пистолет"

    def test_multiple_photos_same_subcategory(self):
        """Несколько фото одной подкатегории объединяются в один объект."""
        text_items = [
            {"subcategory": "rifle", "category": "Страйкбольное оружие", "confidence": 0.8, "source": "text"},
        ]
        photo_preds = [
            {"photo_id": "photo_1", "subcategory": "rifle", "confidence": 0.7},
            {"photo_id": "photo_2", "subcategory": "rifle", "confidence": 0.9},
        ]
        result = AirsoftPredictor._merge(text_items, photo_preds)
        assert len(result) == 1
        assert set(result[0]["photo_ids"]) == {"photo_1", "photo_2"}
        assert result[0]["confidence"] == 0.9
