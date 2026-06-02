#!/bin/bash
# Создаёт виртуальное окружение и устанавливает зависимости для обучения моделей.
# Для запуска API достаточно requirements.txt.

set -e

python3 -m venv .venv
source .venv/bin/activate

echo "Установка зависимостей для обучения..."
pip install --upgrade pip
pip install -r requirements-train.txt

echo ""
echo "Готово. Активация окружения: source .venv/bin/activate"
