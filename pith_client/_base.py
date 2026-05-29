"""Shared configuration and utilities for sync/async clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_BASE_URL = "http://localhost:8000"
HEALTH_TIMEOUT = 5.0
SESSION_TIMEOUT = 30.0
DEFAULT_TIMEOUT = 180.0


@dataclass(frozen=True)
class ClientConfig:
    """Immutable client configuration."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT
    health_timeout: float = HEALTH_TIMEOUT
    session_timeout: float = SESSION_TIMEOUT

    @property
    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"


def _extract_detail(resp_json: Any) -> str:
    """Pull error detail from a JSON response body."""
    if isinstance(resp_json, dict):
        return str(resp_json.get("detail", resp_json.get("error", "")))
    return str(resp_json)
