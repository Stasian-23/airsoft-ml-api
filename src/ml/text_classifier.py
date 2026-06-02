"""rubert-tiny2 классификатор текстов по категориям."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextClassifier:
    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.eval()

        with open(model_dir / "labels.json", encoding="utf-8") as f:
            data = json.load(f)
        self.id2label: dict[int, str] = {int(k): v for k, v in data["id2label"].items()}
        self.label2id: dict[str, int] = {v: int(k) for k, v in data["id2label"].items()}

    def predict(self, text: str) -> tuple[str, float]:
        """Возвращает (категория, уверенность) для лучшего совпадения."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            probs = torch.softmax(self._model(**inputs).logits, dim=1)[0]
        top_idx = int(probs.argmax())
        return self.id2label[top_idx], float(probs[top_idx])

    def predict_all(self, text: str) -> list[dict]:
        """Возвращает уверенность для всех категорий, от большей к меньшей."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            probs = torch.softmax(self._model(**inputs).logits, dim=1)[0]
        return sorted(
            [
                {"category": self.id2label[i], "confidence": float(probs[i])}
                for i in range(len(self.id2label))
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )
