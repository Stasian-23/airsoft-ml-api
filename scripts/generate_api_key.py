"""Генерирует безопасный API ключ и выводит готовую строку для .env."""
import secrets

key = secrets.token_urlsafe(32)
print(f"Сгенерированный ключ: {key}")
print(f"\nДобавьте в .env:\nAPI_KEYS={key}")
