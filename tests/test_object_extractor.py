"""
Тесты для src/ml/object_extractor.py

Запуск: pytest tests/test_object_extractor.py -v
"""
import pytest

from src.ml.object_extractor import extract_objects_from_text


class TestExtractObjects:

    def test_empty_text(self):
        assert extract_objects_from_text("") == []
        assert extract_objects_from_text("   ") == []
        assert extract_objects_from_text(None) == []  # type: ignore[arg-type]

    def test_single_weapon_ak(self):
        result = extract_objects_from_text("Продаю АК-74 LCT, Москва")
        subcats = {r["subcategory"] for r in result}
        assert "ak" in subcats

    def test_single_weapon_pistol(self):
        result = extract_objects_from_text("Продаю пистолет Glock 17")
        subcats = {r["subcategory"] for r in result}
        assert "pistol" in subcats

    def test_multi_object(self):
        text = "Продаю АК-74 LCT и тактический жилет Wartech. Отдам подсумок."
        result = extract_objects_from_text(text)
        subcats = {r["subcategory"] for r in result}
        assert "ak" in subcats
        assert "vest" in subcats
        assert "pouch" in subcats

    def test_preposition_filter(self):
        """«кобура под пистолет» — пистолет не продаётся."""
        result = extract_objects_from_text("Продаю кобуру под пистолет")
        subcats = {r["subcategory"] for r in result}
        assert "pistol" not in subcats
        assert "pouch" in subcats

    def test_weapon_suppresses_parts(self):
        """Если найдено оружие, «parts» убирается."""
        result = extract_objects_from_text(
            "Продаю АК-74, в комплекте гирбокс, мотор и приклад"
        )
        subcats = {r["subcategory"] for r in result}
        assert "ak" in subcats
        assert "parts" not in subcats

    def test_parts_without_weapon(self):
        """Если оружия нет, «parts» остаётся."""
        result = extract_objects_from_text("Продаю прицел коллиматор и приклад")
        subcats = {r["subcategory"] for r in result}
        assert "parts" in subcats

    def test_confidence_range(self):
        """Уверенность всегда в диапазоне [0, 1]."""
        result = extract_objects_from_text(
            "Продаю снайперскую винтовку VSR-10 Snow Wolf и шлем ops-core"
        )
        for item in result:
            assert 0.0 <= item["confidence"] <= 1.0

    def test_source_is_text(self):
        result = extract_objects_from_text("Продаю дробовик Remington 870")
        for item in result:
            assert item["source"] == "text"

    def test_category_field(self):
        result = extract_objects_from_text("Продаю АК-47")
        ak = next(r for r in result if r["subcategory"] == "ak")
        assert ak["category"] == "Страйкбольное оружие"

    def test_case_insensitive(self):
        result_lower = extract_objects_from_text("продаю ак-74")
        result_upper = extract_objects_from_text("ПРОДАЮ АК-74")
        assert {r["subcategory"] for r in result_lower} == {r["subcategory"] for r in result_upper}

    def test_helmet_tactical_headset(self):
        """Тактические наушники относятся к категории шлема."""
        result = extract_objects_from_text("Продаю тактические наушники Peltor Comtac")
        subcats = {r["subcategory"] for r in result}
        assert "helmet" in subcats

    def test_sniper_rifle(self):
        result = extract_objects_from_text("VSR-10 болтовая снайперская винтовка Snow Wolf")
        subcats = {r["subcategory"] for r in result}
        assert "rifle" in subcats

    def test_hk_series(self):
        result = extract_objects_from_text("Продам HK416 VFC")
        subcats = {r["subcategory"] for r in result}
        assert "HK" in subcats

    def test_no_false_positive_numbers(self):
        """Цифры в тексте не должны давать ложные срабатывания."""
        result = extract_objects_from_text("Цена 15000 рублей, торг уместен")
        assert result == []
