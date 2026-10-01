"""Bounded per-invocation request identity helpers."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

MAX_REQUEST_ID_CHARS = 200
REQUEST_ID_HASH_CHARS = 64
CONVERSATION_TURN_SERVER_PREFIX = "ct_srv_"
_VISIBLE_ASCII_RE = re.compile(r"^[\x20-\x7e]+$")
_LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_optional_request_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("request_id must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("request_id must not be empty")
    if not _VISIBLE_ASCII_RE.fullmatch(normalized):
        raise ValueError("request_id must contain visible ASCII characters only")
    if len(normalized) > MAX_REQUEST_ID_CHARS:
        raise ValueError(f"request_id must be at most {MAX_REQUEST_ID_CHARS} characters")
    return normalized


def new_conversation_turn_request_id() -> str:
    return CONVERSATION_TURN_SERVER_PREFIX + uuid.uuid4().hex


def hash_request_id(value: str | None) -> str | None:
    normalized = normalize_optional_request_id(value)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_request_id_hash(value: Any) -> bool:
    return isinstance(value, str) and _LOWER_SHA256_RE.fullmatch(value) is not None
