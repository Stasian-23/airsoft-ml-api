# Используем CPU-версию PyTorch (~800 МБ вместо ~3 ГБ GPU-версии)
FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем requirements для кэширования слоя зависимостей
COPY requirements.txt .

# Устанавливаем torch CPU-only отдельно (не из PyPI, а с официального индекса)
RUN pip install --no-cache-dir \
        torch==2.4.0+cpu \
        torchvision==0.19.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
