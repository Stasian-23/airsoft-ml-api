# Airsoft ML API v2

API машинного обучения для автоматической классификации товаров страйкбольного маркетплейса.
Определяет категорию и подкатегорию каждого товара по тексту объявления и фотографиям.
Поддерживает многообъектные объявления — одно объявление может содержать несколько товаров.

---

## Как работает классификация

Система использует гибридный подход: **текст в приоритете**, фото подтверждают или дополняют результат.

1. **Классификация фото** — каждое фото классифицируется независимо нейросетью EfficientNet-B0. Результат принимается только при уверенности выше порога (по умолчанию 25%).

2. **Извлечение объектов из текста** — текст сканируется по словарю ключевых слов (300+ терминов). Предлоги фильтруют «ложные» упоминания: «кобура под пистолет» — пистолет не продаётся.

3. **Объединение** — товары из текста формируют основной список. Если фото подтверждает товар из текста — `source: "both"`, уверенность повышается. Если фото нашло товар которого нет в тексте — `source: "image"`.

4. **Запасной вариант** — если ни текст, ни фото ничего не нашли, применяется rubert-tiny2. При уверенности ≥ 55% возвращается категория с `subcategory: "Не определено"`.

| source | Значение |
|--------|----------|
| `text` | Найдено в тексте объявления |
| `image` | Определено по фотографии |
| `both` | Текст и фото совпали |

---

## Модели

### Классификатор изображений — EfficientNet-B0
- Предобучен на ImageNet, дообучен на датасете страйкбольных фотографий
- 11 классов (подкатегории): ak, HK, M serias, mashinegun, pistol, rifle, shutgun, vest, helmet, pouch, backpack
- Инференс на CPU: ~150–300 мс

### Классификатор текста — rubert-tiny2
- `cointegrated/rubert-tiny2`, дообучен на 66 000 объявлений маркетплейса
- 3 класса (категории): Страйкбольное оружие, Снаряжение и защита, Аксессуары и Запчасти
- Точность на валидации: ~93%

---

## API

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/predict` | Текст + ссылки на фото (URL) |
| `POST` | `/api/v1/predict/upload` | Текст + загрузка фото с диска (multipart) |
| `GET`  | `/api/v1/health` | Статус сервера и моделей |

### Аутентификация

Все запросы к `/predict` требуют API ключ в заголовке:
```
X-API-Key: ваш_ключ
```

### Формат запроса (POST /api/v1/predict)

```json
{
  "post_id": "12345",
  "text": "Продаю АК-74 LCT и тактический жилет Wartech. 25000р.",
  "photos": [
    {"photo_id": "photo_1", "url": "https://example.com/photo1.jpg"}
  ]
}
```

### Формат ответа

```json
{
  "post_id": "12345",
  "predictions": [
    {
      "object_id": "1",
      "category": "Страйкбольное оружие",
      "subcategory": "АК / АКС",
      "confidence": 0.95,
      "photo_ids": ["photo_1"],
      "source": "both"
    },
    {
      "object_id": "2",
      "category": "Снаряжение и защита",
      "subcategory": "Тактический жилет",
      "confidence": 1.0,
      "photo_ids": [],
      "source": "text"
    }
  ],
  "failed_photos": [],
  "processing_time_ms": 245.3
}
```

---

## Локальный запуск

```bash
git clone https://github.com/your/airsoft-ml-api.git
cd airsoft-ml-api

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
python scripts/generate_api_key.py   # скопировать ключ в .env

uvicorn src.api.main:app --reload
# Документация: http://localhost:8000/docs
```

---

## Деплой через Docker

```bash
docker compose up -d
curl http://localhost:8000/api/v1/health
```

Образ использует CPU-версию PyTorch (~800 МБ вместо ~3 ГБ).
Рекомендуемый минимум сервера: 1 vCPU / 1 ГБ RAM + 2 ГБ swap.

---

## Обучение моделей

```bash
bash setup_venv.sh && source .venv/bin/activate

# Классификатор изображений (subcategory_images/dataset → models/image_classifier.pt)
python -m training.train_image --data_dir data/raw/dataset

# Классификатор текста (posts.parquet → models/text_classifier/)
python -m training.train_text --data_path data/raw/posts.parquet
```

---

## Тесты

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

---

## Структура проекта

```
airsoft-ml-api/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI (lifespan, rate limiting, logging)
│   │   ├── auth.py              # Аутентификация по API ключу
│   │   ├── schemas.py           # Pydantic-схемы
│   │   ├── settings.py          # Настройки из .env
│   │   └── routes/
│   │       ├── predict.py       # POST /predict (URL фото)
│   │       ├── predict_upload.py # POST /predict/upload (загрузка файлов)
│   │       └── health.py        # GET /health
│   └── ml/
│       ├── predictor.py         # Основной конвейер
│       ├── image_classifier.py  # EfficientNet-B0
│       ├── text_classifier.py   # rubert-tiny2
│       ├── object_extractor.py  # Извлечение объектов из текста
│       └── config.py            # Словарь ключевых слов и маппинг
├── tests/
│   ├── test_object_extractor.py # 15 unit-тестов
│   ├── test_predictor_merge.py  # 7 тестов логики объединения
│   └── test_api.py              # Интеграционные тесты API
├── training/
│   ├── train_image.py           # Обучение EfficientNet-B0
│   └── train_text.py            # Дообучение rubert-tiny2
├── scripts/
│   ├── generate_api_key.py
│   └── test_api.py              # Сквозное тестирование живого API
├── models/                      # Веса моделей (Git LFS)
├── Dockerfile                   # CPU-only PyTorch
├── docker-compose.yml
├── requirements.txt
├── requirements-train.txt
├── requirements-test.txt
└── .env.example
```

---

## Коды ошибок

| Код | Причина |
|-----|---------|
| `401` | API ключ не передан |
| `403` | Неверный API ключ |
| `400` | Слишком много фото (максимум 10) |
| `422` | Ошибка валидации (нет текста и фото, некорректный URL) |
| `503` | Модели не загружены |
