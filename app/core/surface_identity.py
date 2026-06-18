"""Consumer surface identity helpers."""

from __future__ import annotations

from app.core.models import CONTROL_CHARS_RE, SURFACE_ID_VALUES


SURFACE_TO_PLATFORM_HINT = {
    "codex_local_api": "codex",
    "claude_code": "claude_code",
    "claude_desktop_mcp": "claude_desktop",
    "cursor_mcp": "cursor",
    "vscode_copilot_mcp": "vscode",
    "windsurf_mcp": "windsurf",
    "cline_mcp": "cline",
    "local_api_cli": "local_api_cli",
}


def normalize_surface_id(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return "unknown"
    if CONTROL_CHARS_RE.search(cleaned):
        return "unknown"
    return cleaned if cleaned in SURFACE_ID_VALUES else "unknown"


def platform_hint_from_surface_id(surface_id: str | None) -> str:
    return SURFACE_TO_PLATFORM_HINT.get(normalize_surface_id(surface_id), "unknown")


def resolve_platform_hint(platform_hint: str | None, surface_id: str | None) -> str:
    explicit = (platform_hint or "").strip()
    if explicit and explicit != "unknown":
        return explicit
    return platform_hint_from_surface_id(surface_id)
