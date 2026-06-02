"""EfficientNet-B0 классификатор изображений по подкатегориям."""
from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageClassifier:
    def __init__(self, model_path: str | Path) -> None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes: list[str] = checkpoint["classes"]

        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, len(self.classes)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model = model

    def _open_image(self, image_bytes: bytes) -> Image.Image:
        """Декодирует байты в PIL Image.
        Вызывает img.verify() для проверки целостности, затем открывает повторно
        (verify() исчерпывает файловый дескриптор внутри PIL)."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, Exception) as exc:
            raise ValueError(f"Не удалось декодировать изображение: {exc}") from exc

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        """Возвращает (ключ подкатегории, уверенность) для лучшего совпадения."""
        img = self._open_image(image_bytes)
        tensor = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
        top_idx = int(probs.argmax())
        return self.classes[top_idx], float(probs[top_idx])

    def predict_topk(self, image_bytes: bytes, k: int = 3) -> list[dict]:
        """Возвращает топ-k предсказаний в порядке убывания уверенности."""
        img = self._open_image(image_bytes)
        tensor = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
        top = probs.topk(min(k, len(self.classes)))
        return [
            {"subcategory": self.classes[int(idx)], "confidence": float(prob)}
            for idx, prob in zip(top.indices, top.values)
        ]
