"""Аутентификация запросов через API-ключ в заголовке X-API-Key."""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from .settings import get_settings

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_header_scheme)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует API ключ. Передайте его в заголовке 'X-API-Key'.",
        )
    if api_key not in get_settings().api_keys_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Неверный API ключ.",
        )
    return api_key
