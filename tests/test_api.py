"""
Интеграционные тесты FastAPI (без загрузки ML-моделей — predictor подменяется моком).

Запуск: pytest tests/test_api.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.settings import get_settings

TEST_API_KEY = "test-key-12345"


@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    """Подменяет настройки, чтобы не нужен был .env файл."""
    import src.api.settings as s
    s.get_settings.cache_clear()
    monkeypatch.setenv("API_KEYS", TEST_API_KEY)
    yield
    s.get_settings.cache_clear()


@pytest.fixture
def mock_predictor():
    """Мок предиктора с предзаданным ответом."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "post_id": "test-post",
        "predictions": [
            {
                "object_id": "1",
                "category": "Страйкбольное оружие",
                "subcategory": "АК / АКС",
                "confidence": 0.95,
                "photo_ids": [],
                "source": "text",
            }
        ],
        "failed_photos": [],
        "processing_time_ms": 10.0,
    }
    predictor.predict_with_bytes.return_value = predictor.predict.return_value
    return predictor


@pytest.fixture
def client(mock_predictor):
    app.state.predictor = mock_predictor
    with TestClient(app) as c:
        yield c
    app.state.predictor = None


class TestAuth:
    def test_missing_api_key_returns_401(self, client):
        resp = client.post("/api/v1/predict", json={
            "post_id": "1", "text": "Продаю АК-74",
        })
        assert resp.status_code == 401

    def test_wrong_api_key_returns_403(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"post_id": "1", "text": "Продаю АК-74"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_valid_api_key_passes(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"post_id": "1", "text": "Продаю АК-74"},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 200


class TestPredict:
    def test_text_only_request(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"post_id": "1", "text": "Продаю АК-74 LCT"},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["post_id"] == "1"
        assert isinstance(data["predictions"], list)
        assert "processing_time_ms" in data

    def test_no_text_no_photos_returns_422(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"post_id": "1"},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422

    def test_invalid_url_returns_422(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "post_id": "1",
                "photos": [{"photo_id": "p1", "url": "not-a-url"}],
            },
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422

    def test_response_schema(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"post_id": "post-1", "text": "Продаю АК-74"},
            headers={"X-API-Key": TEST_API_KEY},
        )
        data = resp.json()
        pred = data["predictions"][0]
        assert all(k in pred for k in ("object_id", "category", "subcategory", "confidence", "photo_ids", "source"))
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_models_not_loaded_returns_503(self, client):
        app.state.predictor = None
        resp = client.post(
            "/api/v1/predict",
            json={"post_id": "1", "text": "Продаю АК-74"},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 503


class TestHealth:
    def test_health_ok(self, client, mock_predictor):
        mock_predictor.image_clf.classes = ["ak", "pistol"]
        mock_predictor.text_clf.id2label = {0: "cat1", 1: "cat2", 2: "cat3"}
        app.state.predictor = mock_predictor
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_health_degraded_without_models(self, client):
        app.state.predictor = None
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["models_loaded"] is False
