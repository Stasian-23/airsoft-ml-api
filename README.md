# Airsoft ML API

API машинного обучения для автоматической классификации товаров страйкбольного маркетплейса.
Определяет категорию и подкатегорию каждого товара по тексту объявления и фотографиям.
Поддерживает многообъектные объявления — одно объявление может содержать несколько товаров.

---

## Задеплоено

| | |
|---|---|
| **Сервер** | `80.78.245.35:8000` |
| **Swagger** | http://80.78.245.35:8000/docs |
| **Health** | http://80.78.245.35:8000/api/v1/health |
| **API ключ** | `Ch6xpfdnTTuYiiJtFognocrHHfXg4bYqsTDGmofMfog` |

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
- Предобучен на ImageNet, дообучен на датасете страйкбольных фотографий (921 изображение)
- 11 классов: `ak`, `HK`, `M serias`, `mashinegun`, `pistol`, `rifle`, `shutgun`, `vest`, `helmet`, `pouch`, `backpack`
- Обучение: 9 эпох, val_acc = **78.9%**
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

В Swagger (`/docs`) — кнопка **Authorize** (замок вверху справа).

Сгенерировать новый ключ:
```bash
python scripts/generate_api_key.py
```

### Формат запроса — POST /api/v1/predict/upload (multipart)

```
post_id   string        — ID объявления
text      string        — текст объявления
photo_1   file          — первое фото (опционально)
photo_2   file          — второе фото (опционально)
photo_3   file          — третье фото (опционально)
```

Пример curl:
```bash
curl -X POST http://80.78.245.35:8000/api/v1/predict/upload \
  -H "X-API-Key: Ch6xpfdnTTuYiiJtFognocrHHfXg4bYqsTDGmofMfog" \
  -F "post_id=1" \
  -F "text=АК-74М страйкбольный, металл. 8000р" \
  -F "photo_1=@/path/to/photo.jpg"
```

### Формат запроса — POST /api/v1/predict (JSON + URL фото)

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

## Деплой через Docker

```bash
git clone https://github.com/WorDem125/airsoft-ml-api.git
cd airsoft-ml-api

cp .env.example .env
# Вставить API ключ в .env

docker compose up -d --build
curl http://localhost:8000/api/v1/health
# Документация: http://localhost:8000/docs
```

Образ использует CPU-версию PyTorch (`torch==2.5.1+cpu`, ~800 МБ).
Минимум сервера: **1 vCPU / 2 ГБ RAM**.

---

## Локальный запуск

```bash
python3 -m venv .venv && source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate                              # Windows

pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env
# Вставить API ключ в .env

uvicorn src.api.main:app --reload
# Документация: http://localhost:8000/docs
```

---

## Обучение моделей

```bash
pip install -r requirements-train.txt

# Классификатор изображений (ImageFolder-структура → models/image_classifier.pt)
python -m training.train_image --data_dir data/raw/dataset --epochs 60 --batch_size 16

# Классификатор текста (posts.parquet → models/text_classifier/)
python -m training.train_text --data_path data/raw/posts.parquet
```

Структура датасета изображений:
```
data/raw/dataset/
├── ak/
├── pistol/
├── helmet/
└── ...
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
├── training/
│   ├── train_image.py           # Обучение EfficientNet-B0
│   └── train_text.py            # Дообучение rubert-tiny2
├── scripts/
│   ├── generate_api_key.py
│   └── test_api.py
├── models/                      # Веса моделей
│   ├── image_classifier.pt      # EfficientNet-B0 (16 МБ)
│   ├── image_classes.json
│   └── text_classifier/         # rubert-tiny2 (~111 МБ)
├── Dockerfile
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
