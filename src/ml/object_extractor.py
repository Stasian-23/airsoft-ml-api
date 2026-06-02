"""
Извлечение товаров из текста объявления по словарю ключевых слов.

Возвращает все подкатегории, чьи ключевые слова найдены в тексте.
Одно объявление может содержать несколько товаров.
"""
from __future__ import annotations

import re

from .config import KEYWORD_MAP, SUBCATEGORY_TO_CATEGORY, WEAPON_SUBCATS

# Предлоги, указывающие что ключевое слово — цель аксессуара, а не продаваемый товар.
# Например: «кобура под пистолет» — пистолет не продаётся; «trigger for Hicapa» — тоже.
_ACCESSORY_PREPS: frozenset[str] = frozenset({
    "под", "для", "к", "от", "на", "по", "через",
    "for",
})

# Паттерн для извлечения последнего слова перед совпадением
_LAST_WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]+$", re.UNICODE)


def _make_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Компилирует regex для списка ключевых слов с учётом границ слов.
    Длинные ключевые слова размещаются первыми (жадный поиск слева)."""
    sorted_kw = sorted(keywords, key=len, reverse=True)
    alternation = "|".join(re.escape(k.lower()) for k in sorted_kw)
    # Используем Unicode-осведомлённые lookaround для кириллицы и латиницы
    return re.compile(
        r"(?<![а-яёА-ЯЁa-zA-Z0-9])(" + alternation + r")(?![а-яёА-ЯЁa-zA-Z0-9])",
        re.IGNORECASE | re.UNICODE,
    )


# Предкомпилированные паттерны для каждой подкатегории
_COMPILED: dict[str, re.Pattern[str]] = {
    subcat: _make_pattern(kws) for subcat, kws in KEYWORD_MAP.items()
}


def _preceded_by_prep(text: str, match_start: int) -> bool:
    """Возвращает True если слово перед совпадением — предлог."""
    before = text[:match_start].rstrip()
    m = _LAST_WORD_RE.search(before)
    return bool(m and m.group().lower() in _ACCESSORY_PREPS)


def extract_objects_from_text(text: str) -> list[dict]:
    """
    Возвращает список найденных товаров:
        [{"subcategory": str, "category": str, "confidence": float, "source": "text"}, ...]

    Уверенность зависит от длины совпавшего ключевого слова (0.60 + 0.04 * len, max 1.0).
    Если в объявлении упомянуто оружие и «parts» одновременно — «parts» убирается,
    т.к. внутренние детали в тексте про оружие — описание комплектации, а не отдельный товар.
    """
    if not text or not text.strip():
        return []

    found: dict[str, float] = {}

    for subcat, pattern in _COMPILED.items():
        good_matches = [
            m.group() for m in pattern.finditer(text)
            if not _preceded_by_prep(text, m.start())
        ]
        if not good_matches:
            continue
        max_len = max(len(m) for m in good_matches)
        found[subcat] = min(0.60 + 0.04 * max_len, 1.0)

    # Убираем «запчасти» если в тексте нашли конкретное оружие
    if "parts" in found and found.keys() & WEAPON_SUBCATS:
        del found["parts"]

    return [
        {
            "subcategory": subcat,
            "category": SUBCATEGORY_TO_CATEGORY.get(subcat, "Аксессуары и Запчасти"),
            "confidence": round(conf, 4),
            "source": "text",
        }
        for subcat, conf in found.items()
    ]
