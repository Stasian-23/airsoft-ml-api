# Используем CPU-версию PyTorch (~800 МБ вместо ~3 ГБ GPU-версии)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch (~800 МБ вместо ~3 ГБ GPU-версии)
RUN pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
