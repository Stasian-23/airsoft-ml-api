"""
Сквозное тестирование живого API.

Использование:
    python scripts/test_api.py
    python scripts/test_api.py --base-url http://your-server:8000 --api-key YOUR_KEY
"""
import argparse
import sys

import httpx

BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"


def run(base_url: str, api_key: str) -> None:
    client = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=30)
    errors = 0

    def check(name: str, resp: httpx.Response, expected_status: int = 200) -> bool:
        ok = resp.status_code == expected_status
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name} → {resp.status_code}")
        if not ok:
            print(f"       Response: {resp.text[:300]}")
        return ok

    # Health
    errors += not check("GET /health", client.get("/api/v1/health"))

    # Predict — текст
    errors += not check(
        "POST /predict (текст, АК)",
        client.post("/api/v1/predict", json={
            "post_id": "test-001",
            "text": "Продаю АК-74 LCT. 15000р, Москва.",
        }),
    )

    # Predict — несколько товаров
    resp = client.post("/api/v1/predict", json={
        "post_id": "test-002",
        "text": "Продаю комплект: АК-74 LCT + тактический жилет Wartech + подсумок МОЛЛИ.",
    })
    ok = check("POST /predict (3 товара)", resp)
    errors += not ok
    if ok:
        preds = resp.json()["predictions"]
        print(f"       Найдено товаров: {len(preds)}")
        for p in preds:
            print(f"         {p['object_id']}. {p['category']} / {p['subcategory']} [{p['source']}]")

    # Predict — без ключа
    errors += not check(
        "POST /predict (без ключа → 401)",
        httpx.post(f"{base_url}/api/v1/predict", json={
            "post_id": "test-003", "text": "Продаю АК-74",
        }),
        expected_status=401,
    )

    # Predict — неверный ключ
    errors += not check(
        "POST /predict (неверный ключ → 403)",
        client.post("/api/v1/predict", json={
            "post_id": "test-004", "text": "Продаю АК-74",
        }, headers={"X-API-Key": "bad-key"}),
        expected_status=403,
    )

    # Predict — нет ни текста ни фото
    errors += not check(
        "POST /predict (нет текста и фото → 422)",
        client.post("/api/v1/predict", json={"post_id": "test-005"}),
        expected_status=422,
    )

    print(f"\n{'PASSED' if errors == 0 else 'FAILED'} ({errors} ошибок)")
    sys.exit(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key",  default=API_KEY)
    args = parser.parse_args()
    run(args.base_url, args.api_key)
