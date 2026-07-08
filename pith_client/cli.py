"""Allowlisted CLI for exec-capable hosts to reach the local Pith API directly."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from ._base import DEFAULT_BASE_URL

DEFAULT_TIMEOUT = 30.0
WORKSTREAM_ACTIVATION_GATE_EXIT_CODE = 3
MAX_STDIN_BYTES = 131072
ACTIVE_WORKSTREAM_RENDER_MAX_CHARS = 1200
TRANSPORT_LOG_PATH = Path.home() / ".pith" / "logs" / "pith_mcp_transport.jsonl"
SURFACE_ID_VALUES = frozenset(
    {
        "claude_code",
        "codex_local_api",
        "claude_desktop_mcp",
        "cursor_mcp",
        "cline_mcp",
        "local_api_cli",
        "vscode_copilot_mcp",
        "windsurf_mcp",
    }
)
SURFACE_ID_ALIASES = {
    "claude_chat": "claude_desktop_mcp",
    "claude_chat_mcp": "claude_desktop_mcp",
    "claude_cowork": "claude_desktop_mcp",
    "claude_cowork_mcp": "claude_desktop_mcp",
    "claude_desktop": "claude_desktop_mcp",
    "vscode": "vscode_copilot_mcp",
    "vscode_mcp": "vscode_copilot_mcp",
    "vs_code": "vscode_copilot_mcp",
    "vs_code_mcp": "vscode_copilot_mcp",
}
CONVERSATION_TURN_STARTUP_MAX_ATTEMPTS = int(os.environ.get("PITH_CLI_CONVERSATION_TURN_STARTUP_MAX_ATTEMPTS", "4"))
CONVERSATION_TURN_STARTUP_RETRY_CAP_S = float(os.environ.get("PITH_CLI_CONVERSATION_TURN_STARTUP_RETRY_CAP_S", "10"))
ALLOWED = {
    "health": ("GET", "/health"),
    "readyz": ("GET", "/readyz"),
    "pith_health": ("GET", "/pith_health"),
    "stats": ("GET", "/pith_stats"),
    "session_start": ("POST", "/session_start"),
    "conversation_turn": ("POST", "/conversation_turn"),
    "checkpoint": ("POST", "/checkpoint"),
    "curiosity_frontier": ("GET", "/pith_curiosity/experiment_frontier"),
    "session_end": ("POST", "/session_end"),
    "session_learn": ("POST", "/session_learn"),
    "write_request_status": ("POST", "/write_request_status"),
    "search": ("POST", "/pith_search"),
    "get_concept": ("GET", "/pith_get_concept"),
    "orient": ("GET", "/pith_orient"),
    "sessions_list": ("GET", "/sessions_list"),
    "related_concepts": ("GET", "/pith_related_concepts"),
    "questions": ("GET", "/pith_questions"),
    "learning_metrics": ("GET", "/learning_metrics"),
    "observability": ("GET", "/pith/observability"),
    "surface_activity": ("GET", "/diagnostics/surface_activity"),
    "metrics_dashboard": ("GET", "/metrics/dashboard"),
    "metrics_bg_tasks": ("GET", "/metrics/bg_tasks"),
    "metrics_summary": ("GET", "/metrics/summary"),
    "metrics_health_trend": ("GET", "/metrics/health_trend"),
    "cko_list": ("GET", "/pith/cko"),
    "workstreams": ("POST", "/pith_threads"),
}
AUTH_EXEMPT_OPERATIONS = frozenset({"health", "readyz"})
PSEUDO_OPERATIONS = frozenset({"lifecycle_diagnostic", "lifecycle_status", "list"})
OPERATION_DISCOVERY_SCHEMA_VERSION = "pith_cli_operation_discovery.v1"
SURFACE_RELIABILITY_CONTRACT_SCHEMA_VERSION = "surface_reliability_contract.v1"
OPERATION_EXAMPLES = {
    "conversation_turn": {
        "schema_version": OPERATION_DISCOVERY_SCHEMA_VERSION,
        "operation": "conversation_turn",
        "kind": "example",
        "command": "~/.pith/bin/pith api conversation_turn --stdin-json",
        "payload": {
            "surface_id": "codex_local_api",
            "origin_id": "codex_<short-workspace-or-thread-id>",
            "workspace_id": "<absolute workspace path>",
            "message": "<current user message>",
            "extracted_concepts_json": "[]",
        },
        "notes": [
            "Use message, not user_message.",
            "origin_id must match ^[A-Za-z0-9._:-]{1,128}$; do not use filesystem paths.",
            "extracted_concepts_json is a string containing JSON, not a raw array or object.",
        ],
    },
    "lifecycle_diagnostic": {
        "schema_version": OPERATION_DISCOVERY_SCHEMA_VERSION,
        "operation": "lifecycle_diagnostic",
        "kind": "example",
        "command": "~/.pith/bin/pith api lifecycle_diagnostic --stdin-json",
        "payload": {
            "surface_id": "codex_local_api",
            "session_id": "auto_<session-id>",
            "requested_surfaces": "codex_local_api,local_api_cli",
            "include_codex_local": True,
        },
        "notes": [
            "Requires at least one selector: session_id, origin_id, or workspace_id.",
            "surface_activity coverage is source coverage, not semantic success or lifecycle enforcement.",
            "surface_reliability is the product/operator contract for whether the selected surface can be trusted for this selector and window.",
            "Codex workspace-slug origin selectors normalize hyphen/underscore variants; session_id or workspace_id is preferred when available.",
        ],
    },
}
OPERATION_SCHEMAS = {
    "conversation_turn": {
        "schema_version": OPERATION_DISCOVERY_SCHEMA_VERSION,
        "operation": "conversation_turn",
        "kind": "operation_schema",
        "command": "~/.pith/bin/pith api conversation_turn --stdin-json",
        "method": "POST",
        "path": "/conversation_turn",
        "required": ["message"],
        "fields": {
            "surface_id": {"type": "string", "recommended": "codex_local_api"},
            "origin_id": {
                "type": "string",
                "pattern": r"^[A-Za-z0-9._:-]{1,128}$",
                "example": "codex_new_project",
            },
            "workspace_id": {"type": "string", "example": "/absolute/workspace/path"},
            "session_id": {"type": "string", "required_after_first_success": True},
            "message": {"type": "string", "required": True},
            "previous_message": {"type": "string", "required": False},
            "previous_response": {"type": "string", "required": False},
            "extracted_concepts_json": {
                "type": "string",
                "format": "JSON-encoded array string",
                "trivial_value": "[]",
            },
        },
        "invalid_common_shapes": [
            {"field": "user_message", "reason": "Use message for conversation_turn."},
            {"field": "origin_id", "reason": "Do not use filesystem paths or slashes."},
            {
                "field": "extracted_concepts_json",
                "reason": "Pass a JSON string, not a raw array or object.",
            },
        ],
    },
    "lifecycle_diagnostic": {
        "schema_version": OPERATION_DISCOVERY_SCHEMA_VERSION,
        "operation": "lifecycle_diagnostic",
        "kind": "operation_schema",
        "command": "~/.pith/bin/pith api lifecycle_diagnostic --stdin-json",
        "method": "LOCAL",
        "path": "",
        "required": ["one_of:session_id,origin_id,workspace_id"],
        "defaults": {
            "surface_id": "codex_local_api",
            "requested_surfaces": "codex_local_api,local_api_cli",
            "include_codex_local": True,
        },
        "fields": {
            "surface_id": {"type": "string", "default": "codex_local_api"},
            "session_id": {"type": "string", "required_one_of": True},
            "origin_id": {"type": "string", "required_one_of": True},
            "workspace_id": {"type": "string", "required_one_of": True},
            "requested_surfaces": {
                "type": "string|array",
                "default": "codex_local_api,local_api_cli",
            },
            "include_codex_local": {"type": "boolean", "default": True},
            "since": {"type": "string", "required": False},
            "until": {"type": "string", "required": False},
            "max_age_seconds": {"type": "integer", "required": False},
            "max_scan_files": {"type": "integer", "required": False},
            "min_confidence": {"type": "number", "required": False},
            "include_concept_samples": {"type": "boolean", "required": False},
            "max_samples_per_surface": {"type": "integer", "required": False},
        },
        "output_schema_version": "surface_lifecycle_diagnostic.v2",
        "outputs": {
            "surface_reliability": SURFACE_RELIABILITY_CONTRACT_SCHEMA_VERSION,
            "surface_reliability_matrix": (
                f"{SURFACE_RELIABILITY_CONTRACT_SCHEMA_VERSION}; present when requested_surface_coverage exists"
            ),
        },
    },
}
LIFECYCLE_STATUS_SCHEMA_VERSION = "surface_lifecycle_status.v1"
LIFECYCLE_DIAGNOSTIC_SCHEMA_VERSION = "surface_lifecycle_diagnostic.v2"
LIFECYCLE_DIAGNOSTIC_DEFAULT_SURFACE_ID = "codex_local_api"
LIFECYCLE_DIAGNOSTIC_DEFAULT_REQUESTED_SURFACES = "codex_local_api,local_api_cli"
LIFECYCLE_DIAGNOSTIC_CLAIM_BOUNDARY = "surface_activity_coverage_is_not_semantic_success_or_lifecycle_enforcement"
LIFECYCLE_STATUS_DEFAULT_MAX_SCAN_FILES = int(os.environ.get("PITH_LIFECYCLE_STATUS_MAX_SCAN_FILES", "500"))
LIFECYCLE_STATUS_MAX_SCAN_FILES_LIMIT = 5000
LIFECYCLE_STATUS_DEFAULT_MAX_AGE_SECONDS = int(
    os.environ.get("PITH_LIFECYCLE_STATUS_MAX_AGE_SECONDS", str(24 * 60 * 60))
)
LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT = 7 * 24 * 60 * 60
CLAUDE_CODE_LIFECYCLE_STATE_DIR = Path.home() / ".pith" / "cache" / "claude-code-lifecycle"
CODEX_LIFECYCLE_STATE_DIR = Path.home() / ".pith" / "cache" / "codex-lifecycle"
_ACTIVE_WORKSTREAM_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "recipe",
        "should",
        "step",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "whats",
        "where",
        "with",
        "you",
    ]
)
_ACTIVE_WORKSTREAM_TRIGGER_WORDS = frozenset(
    [
        "continue",
        "current",
        "find",
        "get",
        "next",
        "project",
        "recover",
        "resume",
        "status",
        "task",
        "work",
        "working",
    ]
)
_ACTIVE_WORKSTREAM_WORKFLOW_WORDS = frozenset(
    [
        "benchmark",
        "deploy",
        "design",
        "gauntlet",
        "implementation",
        "investigation",
        "pipeline",
        "retro",
        "spec",
        "verify",
        "workstream",
        "workstreams",
    ]
)
_ACTIVE_WORKSTREAM_EXACT_CONTINUATIONS = frozenset(
    {
        "continue",
        "resume",
        "status",
        "next",
        "next step",
        "next steps",
        "what next",
        "what is next",
        "whats next",
        "where were we",
        "pick up where we left off",
    }
)


def _resolve_api_key() -> str:
    env_key = os.environ.get("PITH_API_KEY") or os.environ.get("BRAIN_API_KEY", "")
    if env_key:
        return env_key

    env_file = Path.home() / ".pith" / ".env"
    if not env_file.exists():
        return ""

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("PITH_API_KEY=") and not line.startswith("#"):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    return ""


def _load_payload(args: argparse.Namespace) -> dict | None:
    if args.json_file and args.stdin_json:
        raise SystemExit("--json-file and --stdin-json are mutually exclusive")
    if args.json_file:
        return json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    if args.stdin_json:
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        if len(raw) > MAX_STDIN_BYTES:
            raise SystemExit("stdin JSON exceeds MAX_STDIN_BYTES")
        return json.loads(raw.decode("utf-8") or "{}")
    return None


def _normalize_surface_id(value: Any) -> str:
    cleaned = str(value or "").strip().lower()
    if cleaned in SURFACE_ID_ALIASES:
        return SURFACE_ID_ALIASES[cleaned]
    return cleaned if cleaned in SURFACE_ID_VALUES else ""


def _surface_id_alias_source(value: Any) -> str | None:
    cleaned = str(value or "").strip().lower()
    canonical = _normalize_surface_id(cleaned)
    if cleaned and canonical and cleaned != canonical:
        return cleaned
    return None


def _default_surface_id(operation: str) -> str:
    if operation not in {"conversation_turn", "session_start"}:
        return ""
    return (
        _normalize_surface_id(os.environ.get("PITH_SURFACE_ID"))
        or _normalize_surface_id(os.environ.get("PITH_CLI_SURFACE_ID"))
        or "local_api_cli"
    )


def _with_default_surface_payload(operation: str, payload: dict | None) -> dict | None:
    default_surface_id = _default_surface_id(operation)
    if not default_surface_id:
        return payload
    next_payload = dict(payload or {})
    if not _normalize_surface_id(next_payload.get("surface_id")):
        next_payload["surface_id"] = default_surface_id
    return next_payload


def _normalize_surface_activity_payload(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return payload
    requested_surfaces = payload.get("requested_surfaces")
    if not isinstance(requested_surfaces, (list, tuple)):
        return payload

    joined_requested_surfaces = ",".join(
        item
        for item in (
            _normalize_surface_id(surface) or str(surface).strip()
            for surface in requested_surfaces
        )
        if item
    )
    next_payload = dict(payload)
    next_payload["requested_surfaces"] = joined_requested_surfaces
    return next_payload


def _bounded_int(
    value: Any,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(parsed, max_value))


def _lifecycle_phase(
    status: str,
    *,
    verdict: str | None = None,
    reason: str | None = None,
    **evidence: Any,
) -> dict[str, Any]:
    phase = {"status": status, "verdict": verdict or status}
    if reason:
        phase["reason"] = reason
    for key, value in evidence.items():
        if value is not None:
            phase[key] = value
    return phase


def _lifecycle_status_base(
    *,
    payload: dict[str, Any],
    status: str,
    surface_id: str,
    limitations: list[str] | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": LIFECYCLE_STATUS_SCHEMA_VERSION,
        "status": status,
        "surface_id": surface_id or _normalize_surface_id(payload.get("surface_id")) or "unknown",
        "selector": {
            key: str(payload.get(key))
            for key in ("session_id", "origin_id", "workspace_id")
            if payload.get(key) is not None
        },
        "limitations": list(limitations or []),
    }
    alias_source = _surface_id_alias_source(payload.get("surface_id"))
    if alias_source:
        result["requested_surface_id"] = alias_source
        result["canonical_surface_id"] = result["surface_id"]
        result["limitations"].append(
            f"Requested surface_id {alias_source!r} was canonicalized to {result['surface_id']!r}."
        )
    if code:
        result["code"] = code
    return result


def _lifecycle_unsupported_status(payload: dict[str, Any], surface_id: str) -> dict[str, Any]:
    result = _lifecycle_status_base(
        payload=payload,
        status="unsupported",
        surface_id=surface_id,
        limitations=[
            "No read-only adapter-state reporter is implemented for this surface.",
            "Do not infer lifecycle execution from instructions or memory alone.",
        ],
    )
    unsupported = _lifecycle_phase(
        "unsupported",
        reason="adapter_status_reporter_unavailable",
    )
    result.update(
        {
            "context_phase": dict(unsupported),
            "model_visible_phase": dict(unsupported),
            "coherence_phase": dict(unsupported),
            "learning_phase": dict(unsupported),
            "overall_verdict": "unsupported",
        }
    )
    return result


def _lifecycle_selector(payload: dict[str, Any], surface_id: str) -> dict[str, str]:
    selector = {"surface_id": surface_id}
    for key in ("session_id", "origin_id", "workspace_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            selector[key] = str(value).strip()
    return selector


def _has_lifecycle_selector(selector: dict[str, str]) -> bool:
    return any(selector.get(key) for key in ("session_id", "origin_id", "workspace_id"))


class _LifecycleReporter:
    surface_id: str = ""

    def supports(self, surface_id: str) -> bool:
        return surface_id == self.surface_id

    def state_dir(self) -> Path | None:
        return None

    def build_status(
        self,
        payload: dict[str, Any],
        *,
        base_url: str,
        timeout: float,
        transport_mode: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def matches_selector(self, state: dict[str, Any], selector: dict[str, str]) -> bool:
        raise NotImplementedError

    def classify_context_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        return _lifecycle_phase("not_observed", verdict="not_observed", reason="context_not_observed")

    def classify_learning_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        return _lifecycle_phase("not_observed", verdict="not_observed", reason="learning_not_observed")

    def classify_checkpoint_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        return _lifecycle_phase("not_observed", verdict="not_observed", reason="checkpoint_not_observed")

    def classify_coherence_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        return _lifecycle_phase("not_observed", verdict="not_required", reason="coherence_not_probeable")

    def sanitize_state_output(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)


class _FunctionLifecycleReporter(_LifecycleReporter):
    def __init__(self, surface_id: str, state_dir_getter: Any, builder: Any) -> None:
        self.surface_id = surface_id
        self._state_dir_getter = state_dir_getter
        self._builder = builder

    def state_dir(self) -> Path | None:
        return self._state_dir_getter()

    def build_status(
        self,
        payload: dict[str, Any],
        *,
        base_url: str,
        timeout: float,
        transport_mode: str,
    ) -> dict[str, Any]:
        if self.surface_id == "codex_local_api":
            return self._builder(payload, base_url=base_url, timeout=timeout, transport_mode=transport_mode)
        return self._builder(payload)

    def matches_selector(self, state: dict[str, Any], selector: dict[str, str]) -> bool:
        if self.surface_id == "claude_code":
            return _claude_code_state_matches_selector(state, selector)
        if self.surface_id == "codex_local_api":
            return _codex_state_matches_selector(state, selector)
        return False


def _state_string_values(state: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for key in keys:
        value = state.get(key)
        if value is not None and str(value).strip():
            values.add(str(value).strip())
    return values


def _codex_origin_id_variants(origin_id: str | None) -> set[str]:
    value = str(origin_id or "").strip()
    if not value:
        return set()
    variants = {value}
    if value.startswith("codex_"):
        suffix = value[len("codex_") :]
        if suffix:
            variants.add("codex_" + suffix.replace("_", "-"))
            variants.add("codex_" + suffix.replace("-", "_"))
    return variants


def _claude_code_state_matches_selector(
    state: dict[str, Any],
    selector: dict[str, str],
) -> bool:
    surface_id = selector.get("surface_id")
    if surface_id and surface_id != "claude_code":
        return False

    session_id = selector.get("session_id")
    if session_id and session_id not in _state_string_values(
        state,
        (
            "pith_session_id",
            "pre_response_ct_session_id",
            "model_ct_session_id",
        ),
    ):
        return False

    origin_id = selector.get("origin_id")
    if origin_id and origin_id not in _state_string_values(
        state,
        (
            "pre_response_ct_origin_id",
            "model_ct_origin_id",
        ),
    ):
        return False

    workspace_id = selector.get("workspace_id")
    return not (
        workspace_id
        and workspace_id
        not in _state_string_values(
            state,
            ("pre_response_ct_workspace_id",),
        )
    )


def _codex_state_matches_selector(
    state: dict[str, Any],
    selector: dict[str, str],
) -> bool:
    surface_id = selector.get("surface_id")
    if surface_id and surface_id != "codex_local_api":
        return False

    session_id = selector.get("session_id")
    if session_id and session_id not in _state_string_values(
        state,
        (
            "pith_session_id",
            "codex_session_id",
            "model_ct_session_id",
        ),
    ):
        return False

    origin_id = selector.get("origin_id")
    if origin_id:
        state_origins = _state_string_values(
            state,
            (
                "origin_id",
                "model_ct_origin_id",
            ),
        )
        if not (_codex_origin_id_variants(origin_id) & state_origins):
            return False

    workspace_id = selector.get("workspace_id")
    return not (
        workspace_id
        and workspace_id
        not in _state_string_values(
            state,
            ("workspace_id",),
        )
    )


def _codex_state_relaxed_selector_keys(
    state: dict[str, Any],
    selector: dict[str, str],
) -> list[str]:
    matched: list[str] = []
    origin_id = selector.get("origin_id")
    if origin_id:
        state_origins = _state_string_values(
            state,
            (
                "origin_id",
                "model_ct_origin_id",
            ),
        )
        if _codex_origin_id_variants(origin_id) & state_origins:
            matched.append("origin_id")

    workspace_id = selector.get("workspace_id")
    if workspace_id and workspace_id in _state_string_values(state, ("workspace_id",)):
        matched.append("workspace_id")
    return matched


def _lifecycle_state_files(
    state_dir: Path,
    *,
    max_scan_files: int,
    max_age_seconds: int,
) -> list[Path]:
    if not state_dir.exists():
        return []

    now = time.time()
    candidates: list[tuple[float, Path]] = []
    for path in state_dir.glob("*.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if max_age_seconds >= 0 and now - mtime > max_age_seconds:
            continue
        candidates.append((mtime, path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates[:max_scan_files]]


def _read_lifecycle_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _codex_selector_fallback_evidence(
    files: list[Path],
    selector: dict[str, str],
    *,
    fallback_window: str,
) -> dict[str, Any]:
    if not selector.get("origin_id") and not selector.get("workspace_id"):
        return {
            "status": "none",
            "candidate_count": 0,
            "diagnostic_only": True,
            "reason": "no_relaxed_selector_keys",
            "fallback_window": fallback_window,
        }

    candidates: list[tuple[float, Path, dict[str, Any], list[str]]] = []
    now = time.time()
    for path in files:
        state = _read_lifecycle_state(path)
        if state is None or _codex_state_matches_selector(state, selector):
            continue
        matched_keys = _codex_state_relaxed_selector_keys(state, selector)
        if not matched_keys:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, path, state, matched_keys))

    if not candidates:
        return {
            "status": "none",
            "candidate_count": 0,
            "diagnostic_only": True,
            "reason": "no_relaxed_adapter_state_candidates",
            "fallback_window": fallback_window,
        }

    candidates.sort(key=lambda item: item[0], reverse=True)
    newest_mtime, newest_path, newest_state, matched_keys = candidates[0]
    strict_keys = [key for key in ("session_id", "origin_id", "workspace_id") if selector.get(key)]
    return {
        "status": "stale_or_non_session_candidate_only",
        "candidate_count": len(candidates),
        "newest_candidate_age_seconds": max(0, int(now - newest_mtime)),
        "newest_candidate_file": newest_path.name,
        "matched_relaxed_keys": matched_keys,
        "strict_selector_keys": strict_keys,
        "fallback_window": fallback_window,
        "diagnostic_only": True,
        "reason": "no_current_session_adapter_state" if selector.get("session_id") else "no_strict_adapter_state",
        "newest_candidate_origin_id": newest_state.get("origin_id"),
        "newest_candidate_workspace_id": newest_state.get("workspace_id"),
    }


def _parse_transport_ts(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


_TRANSPORT_LIFECYCLE_SAFE_KEYS = {
    "ts",
    "event",
    "operation",
    "transport_mode",
    "surface_id",
    "session_id",
    "resolved_session_id",
    "origin_id",
    "workspace_id",
    "request_id",
    "status",
    "api_status",
    "error",
    "previous_response_present",
    "previous_message_present",
    "extracted_concepts_present",
    "auto_learned",
    "learning_events",
    "accepted_learning_events",
    "checkpoint_task_id",
}


def _transport_events(
    *,
    max_scan_files: int,
    max_age_seconds: int,
    event_names: set[str],
) -> tuple[list[dict[str, Any]], int]:
    if not TRANSPORT_LOG_PATH.exists():
        return [], 0
    try:
        lines = TRANSPORT_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], 0

    now = time.time()
    corrupt_records = 0
    events: list[dict[str, Any]] = []
    for line in reversed(lines):
        if len(events) >= max_scan_files:
            break
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            corrupt_records += 1
            continue
        if not isinstance(entry, dict):
            corrupt_records += 1
            continue
        if entry.get("event") not in event_names:
            continue
        ts = _parse_transport_ts(entry.get("ts") or entry.get("timestamp"))
        if ts is not None and max_age_seconds >= 0 and now - ts > max_age_seconds:
            continue
        events.append(entry)
    return events, corrupt_records


def _transport_lifecycle_events(
    *,
    max_scan_files: int,
    max_age_seconds: int,
) -> tuple[list[dict[str, Any]], int]:
    return _transport_events(
        max_scan_files=max_scan_files,
        max_age_seconds=max_age_seconds,
        event_names={"lifecycle_api_call"},
    )


def _transport_route_events(
    *,
    max_scan_files: int,
    max_age_seconds: int,
) -> tuple[list[dict[str, Any]], int]:
    return _transport_events(
        max_scan_files=max_scan_files,
        max_age_seconds=max_age_seconds,
        event_names={"bridge_start", "tool_call", "lifecycle_api_call"},
    )


def _transport_route_diagnostic(
    *,
    events: list[dict[str, Any]],
    surface_id: str,
) -> dict[str, Any]:
    bridge_starts = []
    for event in events:
        if event.get("event") != "bridge_start":
            continue
        event_surface_id = event.get("surface_id")
        if event_surface_id == surface_id or (not event_surface_id and surface_id == "claude_desktop_mcp"):
            bridge_starts.append(event)
    if not bridge_starts:
        return {
            "schema_version": "transport_route_diagnostic.v1",
            "status": "no_transport_route_evidence",
            "surface_id": surface_id,
            "proof_effect": "diagnostic_only_not_lifecycle_proof",
            "tool_call_observed_since_start": False,
        }

    selected = bridge_starts[0]
    selected_ts = _parse_transport_ts(selected.get("ts") or selected.get("timestamp"))
    selected_pid = selected.get("pid")
    later_tool_calls = []
    if selected_ts is not None:
        for event in events:
            if event.get("event") != "tool_call" or event.get("pid") != selected_pid:
                continue
            event_ts = _parse_transport_ts(event.get("ts") or event.get("timestamp"))
            if event_ts is not None and event_ts >= selected_ts:
                later_tool_calls.append(event)

    if later_tool_calls:
        latest_tool = later_tool_calls[0]
        return {
            "schema_version": "transport_route_diagnostic.v1",
            "status": "tool_call_observed_without_matching_lifecycle_api_call",
            "surface_id": surface_id,
            "bridge_pid": selected_pid,
            "bridge_started_at": selected.get("ts") or selected.get("timestamp"),
            "last_tool_name": latest_tool.get("tool_name"),
            "last_tool_started_at": latest_tool.get("ts") or latest_tool.get("timestamp"),
            "tool_call_observed_since_start": True,
            "proof_effect": "diagnostic_only_not_lifecycle_proof",
            "operator_action": "inspect_lifecycle_api_call_or_tool_error",
        }

    return {
        "schema_version": "transport_route_diagnostic.v1",
        "status": "bridge_started_no_tool_call_observed",
        "surface_id": surface_id,
        "bridge_pid": selected_pid,
        "bridge_started_at": selected.get("ts") or selected.get("timestamp"),
        "exec_fallback_capability": selected.get("exec_fallback_capability"),
        "tool_call_observed_since_start": False,
        "proof_effect": "diagnostic_only_not_lifecycle_proof",
        "operator_action": "inspect_host_mcp_routing_or_restart_client",
    }


def _transport_event_matches_selector(state: dict[str, Any], selector: dict[str, str]) -> bool:
    surface_id = selector.get("surface_id")
    if surface_id and state.get("surface_id") != surface_id:
        return False

    session_id = selector.get("session_id")
    if session_id and session_id not in _state_string_values(
        state,
        ("session_id", "resolved_session_id", "cached_session_id"),
    ):
        return False

    origin_id = selector.get("origin_id")
    if origin_id and origin_id not in _state_string_values(state, ("origin_id",)):
        return False

    workspace_id = selector.get("workspace_id")
    return not (workspace_id and workspace_id not in _state_string_values(state, ("workspace_id",)))


class _TransportLifecycleReporter(_LifecycleReporter):
    def __init__(self, surface_id: str, *, enforcement_claim: str, learning_reason: str) -> None:
        self.surface_id = surface_id
        self.enforcement_claim = enforcement_claim
        self.learning_reason = learning_reason

    def state_dir(self) -> Path | None:
        return TRANSPORT_LOG_PATH.parent

    def matches_selector(self, state: dict[str, Any], selector: dict[str, str]) -> bool:
        return _transport_event_matches_selector(state, selector)

    def sanitize_state_output(self, state: dict[str, Any]) -> dict[str, Any]:
        return {key: state[key] for key in _TRANSPORT_LIFECYCLE_SAFE_KEYS if key in state}

    def classify_context_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("operation") != "conversation_turn":
            return _lifecycle_phase(
                "not_observed",
                verdict="not_observed",
                reason=f"{self.surface_id}_conversation_turn_not_observed",
            )
        if state.get("error") is True or str(state.get("status") or "").lower() in {"error", "failed"}:
            return _lifecycle_phase(
                "failed",
                verdict="not_enforced",
                reason=str(state.get("status") or state.get("api_status") or "conversation_turn_failed"),
                session_id=state.get("resolved_session_id") or state.get("session_id"),
                origin_id=state.get("origin_id"),
                workspace_id=state.get("workspace_id"),
                request_id=state.get("request_id"),
            )
        return _lifecycle_phase(
            "passed",
            verdict="observed",
            reason=self.enforcement_claim,
            session_id=state.get("resolved_session_id") or state.get("session_id"),
            origin_id=state.get("origin_id"),
            workspace_id=state.get("workspace_id"),
            request_id=state.get("request_id"),
        )

    def classify_learning_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        accepted_raw = state.get("accepted_learning_events")
        events_raw = state.get("learning_events")
        try:
            accepted = int(accepted_raw or 0)
        except (TypeError, ValueError):
            accepted = 0
        try:
            events = int(events_raw or 0)
        except (TypeError, ValueError):
            events = 0
        evidence = {
            "accepted_learning_events": accepted,
            "learning_events": events,
            "request_id": state.get("request_id"),
            "previous_response_present": state.get("previous_response_present"),
            "extracted_concepts_present": state.get("extracted_concepts_present"),
        }
        if state.get("operation") == "session_end" and state.get("error") is not True:
            return _lifecycle_phase("passed", verdict="observed", reason=self.learning_reason, **evidence)
        if state.get("operation") == "conversation_turn" and (
            state.get("auto_learned") or accepted > 0 or state.get("previous_response_present")
        ):
            return _lifecycle_phase("passed", verdict="observed", reason=self.learning_reason, **evidence)
        return _lifecycle_phase(
            "not_observed",
            verdict="not_observed",
            reason=f"{self.surface_id}_learning_not_observed",
            **evidence,
        )

    def classify_checkpoint_phase(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("operation") == "checkpoint" and state.get("error") is not True:
            return _lifecycle_phase(
                "passed",
                verdict="observed",
                request_id=state.get("request_id"),
                task_id=state.get("checkpoint_task_id"),
            )
        return _lifecycle_phase(
            "not_observed",
            verdict="not_observed",
            reason=f"{self.surface_id}_checkpoint_not_observed",
        )

    def build_status(
        self,
        payload: dict[str, Any],
        *,
        base_url: str,
        timeout: float,
        transport_mode: str,
    ) -> dict[str, Any]:
        del base_url, timeout, transport_mode
        selector = _lifecycle_selector(payload, self.surface_id)
        if not _has_lifecycle_selector(selector):
            result = _lifecycle_status_base(
                payload=payload,
                status="error",
                surface_id=self.surface_id,
                limitations=["Provide at least one of session_id, origin_id, or workspace_id."],
                code="selector_required",
            )
            result["overall_verdict"] = "error"
            return result

        max_scan_files = _bounded_int(
            payload.get("max_scan_files"),
            default=LIFECYCLE_STATUS_DEFAULT_MAX_SCAN_FILES,
            min_value=1,
            max_value=LIFECYCLE_STATUS_MAX_SCAN_FILES_LIMIT,
        )
        max_age_seconds = _bounded_int(
            payload.get("max_age_seconds"),
            default=LIFECYCLE_STATUS_DEFAULT_MAX_AGE_SECONDS,
            min_value=-1,
            max_value=LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT,
        )
        events, corrupt_records = _transport_lifecycle_events(
            max_scan_files=max_scan_files,
            max_age_seconds=max_age_seconds,
        )
        matches = [event for event in events if self.matches_selector(event, selector)]
        limitations = [
            f"{self.surface_id} reporter uses local transport lifecycle events and proves observation, not enforcement."
        ]
        if corrupt_records:
            limitations.append(f"Skipped {corrupt_records} unreadable transport lifecycle record(s).")
        if len(matches) > 1:
            limitations.append("Multiple matching transport lifecycle events found; newest selected.")
        result = _lifecycle_status_base(
            payload=payload,
            status="not_found" if not matches else "ok",
            surface_id=self.surface_id,
            limitations=limitations,
        )
        result.update(
            {
                "selector": selector,
                "state_dir": str(self.state_dir()),
                "state_source": str(TRANSPORT_LOG_PATH),
                "reporter_kind": "transport_lifecycle_log",
                "lifecycle_enforcement_claim": self.enforcement_claim,
                "scan": {
                    "scanned_files": len(events),
                    "matched_files": len(matches),
                    "corrupt_files": corrupt_records,
                    "max_scan_files": max_scan_files,
                    "max_age_seconds": max_age_seconds,
                },
            }
        )
        if not matches:
            route_events, route_corrupt_records = _transport_route_events(
                max_scan_files=max_scan_files,
                max_age_seconds=max_age_seconds,
            )
            route_diagnostic = _transport_route_diagnostic(events=route_events, surface_id=self.surface_id)
            if route_corrupt_records:
                limitations.append(f"Skipped {route_corrupt_records} unreadable transport route record(s).")
            limitations.append("Transport route diagnostics are diagnostic-only and do not prove lifecycle execution.")
            result["limitations"] = limitations
            not_observed = _lifecycle_phase(
                "not_observed",
                verdict="not_observed",
                reason="no_matching_transport_lifecycle_event",
            )
            result.update(
                {
                    "context_phase": dict(not_observed),
                    "model_visible_phase": _lifecycle_phase(
                        "not_observed",
                        verdict="not_required",
                        reason="model_visible_probe_not_available_for_transport_reporter",
                    ),
                    "coherence_phase": _lifecycle_phase(
                        "not_observed",
                        verdict="not_required",
                        reason="coherence_probe_not_available_for_transport_reporter",
                    ),
                    "learning_phase": dict(not_observed),
                    "checkpoint_phase": dict(not_observed),
                    "transport_route_diagnostic": route_diagnostic,
                    "overall_verdict": "not_observed",
                }
            )
            return result

        selected = self.sanitize_state_output(matches[0])
        context_phase = self.classify_context_phase(selected)
        learning_phase = self.classify_learning_phase(selected)
        checkpoint_phase = self.classify_checkpoint_phase(selected)
        if context_phase.get("status") == "passed" or learning_phase.get("status") == "passed":
            overall = "partial"
        elif context_phase.get("status") == "failed" or learning_phase.get("status") == "failed":
            overall = "failed"
        else:
            overall = "not_observed"
        result.update(
            {
                "selected_state_file": TRANSPORT_LOG_PATH.name,
                "selected_event": selected,
                "context_phase": context_phase,
                "model_visible_phase": _lifecycle_phase(
                    "not_observed",
                    verdict="not_required",
                    reason="model_visible_probe_not_available_for_transport_reporter",
                ),
                "coherence_phase": self.classify_coherence_phase(selected),
                "learning_phase": learning_phase,
                "checkpoint_phase": checkpoint_phase,
                "overall_verdict": overall,
            }
        )
        return result


def _claude_code_context_phase(state: dict[str, Any]) -> dict[str, Any]:
    hook_status = state.get("hook_pre_response_ct_status")
    pre_status = state.get("pre_response_ct_status")
    session_id = state.get("pre_response_ct_session_id") or state.get("pith_session_id")
    if hook_status == "recovered_by_backstop" and session_id:
        return _lifecycle_phase(
            "degraded",
            verdict="recovered",
            reason="hook_backstop_conversation_turn_recovered_after_pre_response_failure",
            session_id=str(session_id),
            origin_id=state.get("pre_response_ct_origin_id"),
            workspace_id=state.get("pre_response_ct_workspace_id"),
            request_id=state.get("hook_pre_response_ct_request_id") or state.get("pre_response_ct_request_id"),
        )
    if pre_status == "ok" and session_id:
        return _lifecycle_phase(
            "passed",
            verdict="enforced",
            session_id=str(session_id),
            origin_id=state.get("pre_response_ct_origin_id"),
            workspace_id=state.get("pre_response_ct_workspace_id"),
            request_id=state.get("pre_response_ct_request_id"),
        )
    if pre_status:
        return _lifecycle_phase(
            "failed",
            verdict="not_enforced",
            reason=str(pre_status),
            session_id=session_id,
            origin_id=state.get("pre_response_ct_origin_id"),
            workspace_id=state.get("pre_response_ct_workspace_id"),
        )
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="pre_response_conversation_turn_not_observed",
    )


def _claude_code_model_visible_phase(state: dict[str, Any]) -> dict[str, Any]:
    session_id = state.get("model_ct_session_id")
    if state.get("model_visible_ct_ok") and session_id:
        return _lifecycle_phase(
            "passed",
            verdict="observed",
            session_id=str(session_id),
            origin_id=state.get("model_ct_origin_id"),
            surface_id=state.get("model_ct_surface_id"),
            response_mode=state.get("model_ct_response_mode"),
        )
    if session_id:
        return _lifecycle_phase(
            "failed",
            verdict="not_enforced",
            reason=state.get("model_ct_coherence_reason") or "model_visible_conversation_turn_not_accepted",
            session_id=str(session_id),
            origin_id=state.get("model_ct_origin_id"),
            surface_id=state.get("model_ct_surface_id"),
            response_mode=state.get("model_ct_response_mode"),
        )
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="model_visible_conversation_turn_not_observed",
    )


def _claude_code_coherence_phase(state: dict[str, Any]) -> dict[str, Any]:
    status = state.get("model_ct_coherence_status")
    if status == "passed":
        return _lifecycle_phase(
            "passed",
            verdict="matched",
            reason=state.get("model_ct_coherence_reason"),
        )
    if status == "skipped_not_observed":
        return _lifecycle_phase(
            "not_observed",
            verdict="not_observed",
            reason=state.get("model_ct_coherence_reason") or "model_visible_conversation_turn_not_observed",
        )
    if status in {"failed", "unknown"}:
        return _lifecycle_phase(
            str(status),
            verdict="mismatch" if status == "failed" else "unknown",
            reason=state.get("model_ct_coherence_reason"),
        )
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="model_ct_coherence_not_observed",
    )


def _claude_code_learning_phase(state: dict[str, Any]) -> dict[str, Any]:
    learn_status = state.get("last_stop_learn_status")
    accepted = state.get("last_stop_learn_accepted_learning_events")
    events = state.get("last_stop_learn_learning_events")
    try:
        accepted_count = int(accepted or 0)
    except (TypeError, ValueError):
        accepted_count = 0
    try:
        event_count = int(events or 0)
    except (TypeError, ValueError):
        event_count = 0

    evidence = {
        "accepted_learning_events": accepted_count,
        "learning_events": event_count,
        "learning_capture_state": state.get("last_stop_learn_learning_capture_state"),
        "session_linkage_state": state.get("last_stop_learn_session_linkage_state"),
        "request_id": state.get("last_stop_learn_request_id"),
    }
    if learn_status == "committed" and accepted_count > 0:
        return _lifecycle_phase("passed", verdict="enforced", **evidence)
    if learn_status == "committed":
        return _lifecycle_phase(
            "failed",
            verdict="not_enforced",
            reason="committed_without_accepted_learning",
            **evidence,
        )
    if learn_status in {"processing", "unknown_pending"}:
        return _lifecycle_phase(
            "degraded",
            verdict="pending",
            reason=str(learn_status),
            **evidence,
        )
    if learn_status:
        return _lifecycle_phase(
            "failed",
            verdict="not_enforced",
            reason=str(learn_status),
            **evidence,
        )
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="stop_session_learn_not_observed",
    )


def _codex_context_phase(state: dict[str, Any]) -> dict[str, Any]:
    status = state.get("pre_response_ct_status")
    session_id = state.get("pith_session_id")
    evidence = {
        "session_id": str(session_id) if session_id else None,
        "origin_id": state.get("origin_id"),
        "workspace_id": state.get("workspace_id"),
        "request_id": state.get("pre_response_ct_request_id"),
        "additional_context_emitted": state.get("additional_context_emitted"),
    }
    if status == "ok" and session_id and state.get("additional_context_emitted"):
        return _lifecycle_phase("passed", verdict="enforced", **evidence)
    if status:
        return _lifecycle_phase("failed", verdict="not_enforced", reason=str(status), **evidence)
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="codex_user_prompt_submit_conversation_turn_not_observed",
    )


def _codex_lifecycle_proof_fields(state: dict[str, Any], context_phase: dict[str, Any]) -> dict[str, Any]:
    proof_status = str(state.get("lifecycle_proof_status") or "").strip()
    if not proof_status:
        reason = str(context_phase.get("reason") or state.get("pre_response_ct_status") or "")
        lowered = reason.lower()
        if "timed out" in lowered or "timeout" in lowered:
            proof_status = "transport_timeout"
        elif context_phase.get("status") == "passed":
            proof_status = "proof_ok"
        elif context_phase.get("status") == "not_observed":
            proof_status = "proof_unavailable"
        else:
            proof_status = "nonzero_json"

    delivery_status = str(state.get("context_delivery_status") or "").strip()
    if not delivery_status:
        delivery_status = "delivered" if context_phase.get("status") == "passed" else "unknown"

    if "can_claim_context_delivered" in state:
        can_claim_context = bool(state.get("can_claim_context_delivered"))
    else:
        can_claim_context = context_phase.get("status") == "passed" and delivery_status == "delivered"

    return {
        "lifecycle_proof_status": proof_status,
        "transport_status": state.get("transport_status")
        or ("ok" if context_phase.get("status") == "passed" else "unknown"),
        "context_delivery_status": delivery_status,
        "semantic_context_status": state.get("semantic_context_status") or "unknown",
        "workstream_status": state.get("workstream_status") or "not_applicable",
        "selector_status": state.get("selector_status") or "unknown",
        "can_claim_context_delivered": can_claim_context,
        "model_visible_message_kind": state.get("model_visible_message_kind") or proof_status,
        "operator_action": state.get("operator_action") or "inspect_lifecycle_status",
    }


def _codex_model_visible_phase(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("model_visible_ct_ok"):
        return _lifecycle_phase(
            "passed",
            verdict="observed",
            session_id=state.get("model_ct_session_id"),
            origin_id=state.get("model_ct_origin_id"),
            surface_id=state.get("model_ct_surface_id"),
        )
    return _lifecycle_phase(
        "not_observed",
        verdict="not_required",
        reason="model_visible_conversation_turn_marker_not_observed",
    )


def _codex_learning_phase(state: dict[str, Any]) -> dict[str, Any]:
    learn_status = state.get("learning_status")
    proof_prefix = ""
    proof_source = "current_stop"
    if (not learn_status or learn_status == "skipped") and state.get("last_stop_learning_status"):
        proof_prefix = "last_stop_"
        proof_source = "last_stop"
        learn_status = state.get("last_stop_learning_status")
    accepted = state.get(f"{proof_prefix}accepted_learning_events")
    events = state.get(f"{proof_prefix}learning_events")
    try:
        accepted_count = int(accepted or 0)
    except (TypeError, ValueError):
        accepted_count = 0
    try:
        event_count = int(events or 0)
    except (TypeError, ValueError):
        event_count = 0
    evidence = {
        "accepted_learning_events": accepted_count,
        "learning_events": event_count,
        "learning_capture_state": state.get(f"{proof_prefix}learning_capture_state"),
        "session_linkage_state": state.get(f"{proof_prefix}session_linkage_state"),
        "request_id": state.get(f"{proof_prefix}learning_request_id"),
        "stop_observed": state.get("last_stop_observed") if proof_prefix else state.get("stop_observed"),
        "proof_source": proof_source,
    }
    if (
        proof_source == "current_stop"
        and learn_status == "skipped"
        and accepted_count > 0
        and state.get("learning_request_id")
    ):
        learn_status = "committed"
        evidence["proof_source"] = "legacy_mixed_stop"
        evidence["legacy_stop_reset_detected"] = True
    if (
        proof_source == "last_stop"
        and learn_status in {None, "", "skipped"}
        and accepted_count > 0
        and evidence.get("request_id")
        and evidence.get("stop_observed")
    ):
        learn_status = "committed"
        evidence["proof_source"] = "last_stop"
        evidence["last_stop_status_gap_detected"] = True
    if learn_status == "committed" and accepted_count > 0:
        return _lifecycle_phase("passed", verdict="enforced", **evidence)
    if learn_status == "committed":
        return _lifecycle_phase(
            "failed",
            verdict="not_enforced",
            reason="committed_without_accepted_learning",
            **evidence,
        )
    if learn_status in {"processing", "unknown_pending"}:
        return _lifecycle_phase("degraded", verdict="pending", reason=str(learn_status), **evidence)
    if learn_status and learn_status != "skipped":
        return _lifecycle_phase("failed", verdict="not_enforced", reason=str(learn_status), **evidence)
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="codex_stop_session_learn_not_observed",
        **evidence,
    )


def _codex_coherence_phase(state: dict[str, Any]) -> dict[str, Any]:
    learning_request_id = state.get("learning_request_id") or state.get("last_stop_learning_request_id")
    if state.get("pith_session_id") and learning_request_id:
        return _lifecycle_phase(
            "passed",
            verdict="matched",
            reason="codex_context_and_learning_share_pith_session_id",
            session_id=state.get("pith_session_id"),
            request_id=learning_request_id,
        )
    return _lifecycle_phase(
        "not_observed",
        verdict="not_observed",
        reason="codex_same_session_learning_linkage_not_observed",
    )


def _codex_checkpoint_phase(state: dict[str, Any]) -> dict[str, Any]:
    status = state.get("checkpoint_status")
    if status == "ok":
        return _lifecycle_phase("passed", verdict="observed", request_id=state.get("checkpoint_request_id"))
    if status:
        return _lifecycle_phase("failed", verdict="not_observed", reason=str(status))
    return _lifecycle_phase("not_observed", verdict="not_observed", reason="codex_precompact_checkpoint_not_observed")


def _codex_overall_lifecycle_verdict(
    context_phase: dict[str, Any],
    learning_phase: dict[str, Any],
) -> str:
    statuses = [context_phase.get("status"), learning_phase.get("status")]
    if all(status == "passed" for status in statuses):
        return "enforced"
    if all(status == "not_observed" for status in statuses):
        return "not_observed"
    if any(status == "passed" for status in statuses):
        return "partial"
    return "failed"


def _overall_lifecycle_verdict(
    context_phase: dict[str, Any],
    model_visible_phase: dict[str, Any],
    coherence_phase: dict[str, Any],
    learning_phase: dict[str, Any],
) -> str:
    statuses = [
        context_phase.get("status"),
        model_visible_phase.get("status"),
        coherence_phase.get("status"),
        learning_phase.get("status"),
    ]
    if all(status == "passed" for status in statuses):
        return "enforced"
    if all(status == "not_observed" for status in statuses):
        return "not_observed"
    if any(status == "passed" for status in statuses):
        return "partial"
    return "failed"


_TURN_HISTORY_SAFE_KEYS = {
    "turn_seq",
    "started_at",
    "completed_at",
    "completion_reason",
    "pre_response_ct_status",
    "pre_response_ct_session_id",
    "pre_response_ct_request_id",
    "hook_pre_response_ct_status",
    "model_visible_status",
    "model_ct_session_id",
    "model_ct_surface_id",
    "model_ct_origin_id",
    "model_ct_response_mode",
    "model_ct_coherence_status",
    "model_ct_coherence_reason",
    "stop_observed",
    "learning_status",
    "learning_events",
    "accepted_learning_events",
}
_MISSED_MODEL_VISIBLE_STATUSES = {"not_observed", "skipped_not_observed"}


def _claude_code_turn_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    history = state.get("turn_history")
    if not isinstance(history, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        sanitized.append({key: item[key] for key in _TURN_HISTORY_SAFE_KEYS if key in item})
    return sanitized


def _claude_code_turn_summary(turn_history: list[dict[str, Any]]) -> dict[str, Any]:
    missed = [
        turn
        for turn in turn_history
        if turn.get("model_visible_status") in _MISSED_MODEL_VISIBLE_STATUSES
        or turn.get("model_ct_coherence_status") == "skipped_not_observed"
    ]
    observed = [turn for turn in turn_history if turn.get("model_visible_status") == "observed"]
    latest_seq = turn_history[-1].get("turn_seq") if turn_history else None
    return {
        "total_turns": len(turn_history),
        "latest_turn_seq": latest_seq,
        "model_visible_observed_turns": len(observed),
        "model_visible_missed_turns": len(missed),
        "first_missed_turn_seq": missed[0].get("turn_seq") if missed else None,
    }


def _claude_code_lifecycle_status(payload: dict[str, Any]) -> dict[str, Any]:
    surface_id = "claude_code"
    selector = _lifecycle_selector(payload, surface_id)
    if not _has_lifecycle_selector(selector):
        result = _lifecycle_status_base(
            payload=payload,
            status="error",
            surface_id=surface_id,
            limitations=["Provide at least one of session_id, origin_id, or workspace_id."],
            code="selector_required",
        )
        result["overall_verdict"] = "error"
        return result

    max_scan_files = _bounded_int(
        payload.get("max_scan_files"),
        default=LIFECYCLE_STATUS_DEFAULT_MAX_SCAN_FILES,
        min_value=1,
        max_value=LIFECYCLE_STATUS_MAX_SCAN_FILES_LIMIT,
    )
    max_age_seconds = _bounded_int(
        payload.get("max_age_seconds"),
        default=LIFECYCLE_STATUS_DEFAULT_MAX_AGE_SECONDS,
        min_value=-1,
        max_value=LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT,
    )

    files = _lifecycle_state_files(
        CLAUDE_CODE_LIFECYCLE_STATE_DIR,
        max_scan_files=max_scan_files,
        max_age_seconds=max_age_seconds,
    )
    corrupt_files = 0
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        state = _read_lifecycle_state(path)
        if state is None:
            corrupt_files += 1
            continue
        if _claude_code_state_matches_selector(state, selector):
            matches.append((path, state))

    limitations: list[str] = []
    if corrupt_files:
        limitations.append(f"Skipped {corrupt_files} unreadable lifecycle state file(s).")
    if len(matches) > 1:
        limitations.append("Multiple matching adapter states found; newest selected.")

    result = _lifecycle_status_base(
        payload=payload,
        status="not_found" if not matches else "ok",
        surface_id=surface_id,
        limitations=limitations,
    )
    result.update(
        {
            "selector": selector,
            "state_dir": str(CLAUDE_CODE_LIFECYCLE_STATE_DIR),
            "scan": {
                "scanned_files": len(files),
                "matched_files": len(matches),
                "corrupt_files": corrupt_files,
                "max_scan_files": max_scan_files,
                "max_age_seconds": max_age_seconds,
            },
        }
    )
    if not matches:
        not_observed = _lifecycle_phase(
            "not_observed",
            verdict="not_observed",
            reason="no_matching_adapter_state",
        )
        result.update(
            {
                "context_phase": dict(not_observed),
                "model_visible_phase": dict(not_observed),
                "coherence_phase": dict(not_observed),
                "learning_phase": dict(not_observed),
                "overall_verdict": "not_observed",
            }
        )
        return result

    selected_path, state = matches[0]
    context_phase = _claude_code_context_phase(state)
    model_visible_phase = _claude_code_model_visible_phase(state)
    coherence_phase = _claude_code_coherence_phase(state)
    learning_phase = _claude_code_learning_phase(state)
    turn_history = _claude_code_turn_history(state)
    turn_summary = _claude_code_turn_summary(turn_history)
    overall_verdict = _overall_lifecycle_verdict(
        context_phase,
        model_visible_phase,
        coherence_phase,
        learning_phase,
    )
    if overall_verdict == "enforced" and turn_summary["model_visible_missed_turns"] > 0:
        overall_verdict = "partial"
        limitations.append("One or more recorded turns missed model-visible conversation_turn.")
        result["limitations"] = list(limitations)
    result.update(
        {
            "selected_state_file": selected_path.name,
            "hook_turn_seq": state.get("hook_turn_seq"),
            "context_phase": context_phase,
            "model_visible_phase": model_visible_phase,
            "coherence_phase": coherence_phase,
            "learning_phase": learning_phase,
            "turn_history": turn_history,
            "turn_summary": turn_summary,
            "overall_verdict": overall_verdict,
        }
    )
    return result


def _db_checkpoint_status(
    payload: dict[str, Any],
    *,
    base_url: str,
    timeout: float,
    transport_mode: str,
) -> dict[str, Any]:
    checkpoint_payload: dict[str, Any] = {"action": "load"}
    selector: dict[str, str] = {}
    for key in ("task_id", "origin_id", "session_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            checkpoint_payload[key] = str(value).strip()
            selector[key] = str(value).strip()
    max_age_hours = payload.get("checkpoint_max_age_hours", payload.get("max_age_hours"))
    if max_age_hours is not None:
        checkpoint_payload["max_age_hours"] = max_age_hours
    if not selector:
        return {
            "status": "not_requested",
            "reason": "checkpoint_selector_absent",
            "selector": {},
        }

    try:
        headers = _build_headers("checkpoint", transport_mode)
    except SystemExit as exc:
        return {
            "status": "unavailable",
            "reason": "api_key_unavailable",
            "selector": selector,
            "message": str(exc),
        }

    try:
        response = requests.post(
            f"{base_url}{ALLOWED['checkpoint'][1]}",
            json=checkpoint_payload,
            headers=headers,
            timeout=timeout,
        )
        try:
            body = response.json()
        except ValueError:
            return {
                "status": "unavailable",
                "reason": "non_json_response",
                "selector": selector,
                "status_code": response.status_code,
                "message": response.text[:500],
            }
    except requests.RequestException as exc:
        return {
            "status": "unavailable",
            "reason": "request_failed",
            "selector": selector,
            "message": str(exc),
        }

    if response.status_code >= 400:
        return {
            "status": "unavailable",
            "reason": "http_error",
            "selector": selector,
            "status_code": response.status_code,
            "body": body,
        }
    if not isinstance(body, dict):
        return {
            "status": "unavailable",
            "reason": "invalid_response_shape",
            "selector": selector,
        }
    body = dict(body)
    body.setdefault("status", "ok")
    body["selector"] = selector
    return body


def _codex_transport_fallback_status(
    payload: dict[str, Any],
    *,
    base_url: str,
    timeout: float,
    transport_mode: str,
) -> dict[str, Any]:
    reporter = _TransportLifecycleReporter(
        "codex_local_api",
        enforcement_claim=(
            "first-class Codex local API lifecycle call observed; "
            "Codex state-file hook proof was not observed"
        ),
        learning_reason="first-class Codex local API learning evidence observed",
    )
    return reporter.build_status(
        payload,
        base_url=base_url,
        timeout=timeout,
        transport_mode=transport_mode,
    )


def _codex_lifecycle_status(
    payload: dict[str, Any],
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    transport_mode: str = "exec_http_fallback",
) -> dict[str, Any]:
    surface_id = "codex_local_api"
    selector = _lifecycle_selector(payload, surface_id)
    if not _has_lifecycle_selector(selector):
        result = _lifecycle_status_base(
            payload=payload,
            status="error",
            surface_id=surface_id,
            limitations=["Provide at least one of session_id, origin_id, or workspace_id."],
            code="selector_required",
        )
        result["overall_verdict"] = "error"
        return result

    max_scan_files = _bounded_int(
        payload.get("max_scan_files"),
        default=LIFECYCLE_STATUS_DEFAULT_MAX_SCAN_FILES,
        min_value=1,
        max_value=LIFECYCLE_STATUS_MAX_SCAN_FILES_LIMIT,
    )
    max_age_seconds = _bounded_int(
        payload.get("max_age_seconds"),
        default=LIFECYCLE_STATUS_DEFAULT_MAX_AGE_SECONDS,
        min_value=-1,
        max_value=LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT,
    )

    files = _lifecycle_state_files(
        CODEX_LIFECYCLE_STATE_DIR,
        max_scan_files=max_scan_files,
        max_age_seconds=max_age_seconds,
    )
    corrupt_files = 0
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        state = _read_lifecycle_state(path)
        if state is None:
            corrupt_files += 1
            continue
        if _codex_state_matches_selector(state, selector):
            matches.append((path, state))

    limitations: list[str] = [
        "Codex hook trust state is not automatically verified; hook configuration alone is not an enforced claim."
    ]
    if corrupt_files:
        limitations.append(f"Skipped {corrupt_files} unreadable lifecycle state file(s).")
    if len(matches) > 1:
        limitations.append("Multiple matching adapter states found; newest selected.")

    result = _lifecycle_status_base(
        payload=payload,
        status="not_found" if not matches else "ok",
        surface_id=surface_id,
        limitations=limitations,
    )
    result.update(
        {
            "selector": selector,
            "state_dir": str(CODEX_LIFECYCLE_STATE_DIR),
            "scan": {
                "scanned_files": len(files),
                "matched_files": len(matches),
                "corrupt_files": corrupt_files,
                "max_scan_files": max_scan_files,
                "max_age_seconds": max_age_seconds,
            },
            "db_checkpoint": _db_checkpoint_status(
                payload,
                base_url=base_url,
                timeout=timeout,
                transport_mode=transport_mode,
            ),
        }
    )
    if not matches:
        not_observed = _lifecycle_phase(
            "not_observed",
            verdict="not_observed",
            reason="no_matching_adapter_state",
        )
        proof_fields = _codex_lifecycle_proof_fields({}, dict(not_observed))
        fallback_evidence = _codex_selector_fallback_evidence(
            files,
            selector,
            fallback_window="current_proof_window",
        )
        if fallback_evidence.get("candidate_count") == 0:
            stale_max_age_seconds = _bounded_int(
                payload.get("fallback_max_age_seconds"),
                default=LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT,
                min_value=0,
                max_value=LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT,
            )
            stale_files = _lifecycle_state_files(
                CODEX_LIFECYCLE_STATE_DIR,
                max_scan_files=max_scan_files,
                max_age_seconds=stale_max_age_seconds,
            )
            fallback_evidence = _codex_selector_fallback_evidence(
                stale_files,
                selector,
                fallback_window="stale_diagnostic_window",
            )
        transport_fallback = _codex_transport_fallback_status(
            payload,
            base_url=base_url,
            timeout=timeout,
            transport_mode=transport_mode,
        )
        if transport_fallback.get("status") == "ok" and transport_fallback.get("overall_verdict") != "not_observed":
            transport_context_phase = dict(transport_fallback.get("context_phase") or {})
            transport_context_phase.setdefault(
                "reason",
                "codex_transport_lifecycle_observed_state_file_missing",
            )
            transport_context_phase["verdict"] = "observed"
            proof_fields = _codex_lifecycle_proof_fields(
                {
                    "lifecycle_proof_status": "transport_event_observed_state_file_missing",
                    "context_delivery_status": "unknown",
                    "can_claim_context_delivered": False,
                    "transport_status": "ok",
                    "semantic_context_status": "unknown",
                    "workstream_status": "not_applicable",
                    "selector_status": "transport_event_matched",
                    "operator_action": "inspect_codex_state_file_delivery",
                },
                transport_context_phase,
            )
            result["limitations"].append(
                "Matching Codex local API transport evidence was observed, but no matching Codex lifecycle state file was found."
            )
            result.update(
                {
                    "status": "ok",
                    "context_phase": transport_context_phase,
                    "model_visible_phase": transport_fallback.get("model_visible_phase") or dict(not_observed),
                    "coherence_phase": transport_fallback.get("coherence_phase") or dict(not_observed),
                    "learning_phase": transport_fallback.get("learning_phase") or dict(not_observed),
                    "checkpoint_phase": transport_fallback.get("checkpoint_phase") or dict(not_observed),
                    "lifecycle_proof": proof_fields,
                    "lifecycle_proof_status": proof_fields["lifecycle_proof_status"],
                    "context_delivery_status": proof_fields["context_delivery_status"],
                    "can_claim_context_delivered": proof_fields["can_claim_context_delivered"],
                    "selector_fallback_evidence": fallback_evidence,
                    "transport_lifecycle_fallback": transport_fallback,
                    "overall_verdict": "partial",
                }
            )
            return result
        result.update(
            {
                "context_phase": dict(not_observed),
                "model_visible_phase": dict(not_observed),
                "coherence_phase": dict(not_observed),
                "learning_phase": dict(not_observed),
                "checkpoint_phase": dict(not_observed),
                "lifecycle_proof": proof_fields,
                "lifecycle_proof_status": proof_fields["lifecycle_proof_status"],
                "context_delivery_status": proof_fields["context_delivery_status"],
                "can_claim_context_delivered": proof_fields["can_claim_context_delivered"],
                "selector_fallback_evidence": fallback_evidence,
                "overall_verdict": "not_observed",
            }
        )
        return result

    selected_path, state = matches[0]
    context_phase = _codex_context_phase(state)
    proof_fields = _codex_lifecycle_proof_fields(state, context_phase)
    model_visible_phase = _codex_model_visible_phase(state)
    coherence_phase = _codex_coherence_phase(state)
    learning_phase = _codex_learning_phase(state)
    checkpoint_phase = _codex_checkpoint_phase(state)
    result.update(
        {
            "selected_state_file": selected_path.name,
            "hook_version": state.get("hook_version"),
            "codex_session_id": state.get("codex_session_id"),
            "codex_turn_id": state.get("codex_turn_id"),
            "context_phase": context_phase,
            "model_visible_phase": model_visible_phase,
            "coherence_phase": coherence_phase,
            "learning_phase": learning_phase,
            "checkpoint_phase": checkpoint_phase,
            "lifecycle_proof": proof_fields,
            "lifecycle_proof_status": proof_fields["lifecycle_proof_status"],
            "context_delivery_status": proof_fields["context_delivery_status"],
            "can_claim_context_delivered": proof_fields["can_claim_context_delivered"],
            "overall_verdict": _codex_overall_lifecycle_verdict(context_phase, learning_phase),
        }
    )
    return result


def _lifecycle_reporters() -> dict[str, _LifecycleReporter]:
    return {
        "claude_code": _FunctionLifecycleReporter(
            "claude_code",
            lambda: CLAUDE_CODE_LIFECYCLE_STATE_DIR,
            _claude_code_lifecycle_status,
        ),
        "codex_local_api": _FunctionLifecycleReporter(
            "codex_local_api",
            lambda: CODEX_LIFECYCLE_STATE_DIR,
            _codex_lifecycle_status,
        ),
        "cursor_mcp": _TransportLifecycleReporter(
            "cursor_mcp",
            enforcement_claim="instruction-mediated; MCP tool call observed but not hook-enforced",
            learning_reason="instruction-mediated learning evidence observed",
        ),
        "claude_desktop_mcp": _TransportLifecycleReporter(
            "claude_desktop_mcp",
            enforcement_claim="instruction-mediated; Claude Desktop MCP tool call observed but not hook-enforced",
            learning_reason="Claude Desktop MCP learning evidence observed",
        ),
        "vscode_copilot_mcp": _TransportLifecycleReporter(
            "vscode_copilot_mcp",
            enforcement_claim="instruction-mediated; VS Code Copilot MCP tool call observed but not hook-enforced",
            learning_reason="VS Code Copilot MCP learning evidence observed",
        ),
        "windsurf_mcp": _TransportLifecycleReporter(
            "windsurf_mcp",
            enforcement_claim="instruction-mediated; Windsurf MCP tool call observed but not hook-enforced",
            learning_reason="Windsurf MCP learning evidence observed",
        ),
        "cline_mcp": _TransportLifecycleReporter(
            "cline_mcp",
            enforcement_claim="instruction-mediated; Cline MCP tool call observed but not hook-enforced",
            learning_reason="Cline MCP learning evidence observed",
        ),
        "local_api_cli": _TransportLifecycleReporter(
            "local_api_cli",
            enforcement_claim="manual/API-only; direct local API call observed",
            learning_reason="manual/API learning evidence observed",
        ),
    }


def _lifecycle_reporter_for(surface_id: str) -> _LifecycleReporter | None:
    return _lifecycle_reporters().get(surface_id)


def _build_lifecycle_status(
    payload: dict | None,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    transport_mode: str = "exec_http_fallback",
) -> dict[str, Any]:
    body = dict(payload or {})
    raw_surface_id = str(body.get("surface_id") or "").strip()
    surface_id = _normalize_surface_id(raw_surface_id)
    if not surface_id and raw_surface_id:
        return _lifecycle_unsupported_status(body, raw_surface_id.lower())
    surface_id = surface_id or "local_api_cli"
    reporter = _lifecycle_reporter_for(surface_id)
    if reporter is not None:
        return reporter.build_status(
            body,
            base_url=base_url,
            timeout=timeout,
            transport_mode=transport_mode,
        )
    return _lifecycle_unsupported_status(body, surface_id)


def _lifecycle_diagnostic_surface_activity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_passthrough = {
        "since",
        "until",
        "max_age_seconds",
        "max_scan_files",
        "min_confidence",
        "include_concept_samples",
        "max_samples_per_surface",
    }
    surface_payload = {key: payload[key] for key in allowed_passthrough if key in payload}
    surface_payload["requested_surfaces"] = payload.get(
        "requested_surfaces",
        LIFECYCLE_DIAGNOSTIC_DEFAULT_REQUESTED_SURFACES,
    )
    surface_payload["include_codex_local"] = payload.get("include_codex_local", True)
    normalized = _normalize_surface_activity_payload(surface_payload)
    return dict(normalized or surface_payload)


def _fetch_lifecycle_diagnostic_surface_activity(
    payload: dict[str, Any],
    *,
    base_url: str,
    timeout: float,
    transport_mode: str,
) -> dict[str, Any]:
    surface_payload = _lifecycle_diagnostic_surface_activity_payload(payload)
    try:
        headers = _build_headers("surface_activity", transport_mode)
    except SystemExit as exc:
        return {
            "status": "unavailable",
            "reason": "api_key_unavailable",
            "message": str(exc),
            "request_payload": surface_payload,
        }

    try:
        response = requests.get(
            f"{base_url}{ALLOWED['surface_activity'][1]}",
            params=surface_payload or None,
            headers=headers,
            timeout=timeout,
        )
        try:
            body = response.json()
        except ValueError:
            return {
                "status": "unavailable",
                "reason": "non_json_response",
                "status_code": response.status_code,
                "message": response.text[:500],
                "request_payload": surface_payload,
            }
    except requests.RequestException as exc:
        return {
            "status": "unavailable",
            "reason": "request_failed",
            "message": str(exc),
            "request_payload": surface_payload,
        }

    if response.status_code >= 400:
        return {
            "status": "unavailable",
            "reason": "http_error",
            "status_code": response.status_code,
            "body": body,
            "request_payload": surface_payload,
        }
    if not isinstance(body, dict):
        return {
            "status": "unavailable",
            "reason": "invalid_response_shape",
            "request_payload": surface_payload,
        }
    body = dict(body)
    body.setdefault("status", "ok")
    body["request_payload"] = surface_payload
    return body


def _lifecycle_diagnostic_surface_activity_summary(surface_activity: dict[str, Any]) -> dict[str, Any]:
    if surface_activity.get("status") == "unavailable":
        return dict(surface_activity)

    coverage = surface_activity.get("requested_surface_coverage")
    if not isinstance(coverage, dict):
        return {
            "status": "unavailable",
            "reason": "missing_requested_surface_coverage",
            "body_status": surface_activity.get("status"),
            "request_payload": surface_activity.get("request_payload"),
        }

    summary: dict[str, Any] = {
        "status": surface_activity.get("status", "ok"),
        "claim_boundary": surface_activity.get("claim_boundary"),
        "limitations": surface_activity.get("limitations", []),
        "request_payload": surface_activity.get("request_payload"),
        "requested_surface_coverage": {
            "schema_version": coverage.get("schema_version"),
            "overall_verdict": coverage.get("overall_verdict"),
            "requested_surfaces": coverage.get("requested_surfaces"),
            "surfaces": coverage.get("surfaces", []),
        },
    }
    if "period" in surface_activity:
        summary["period"] = surface_activity["period"]
    if "codex_local_threads" in surface_activity:
        codex_threads = surface_activity["codex_local_threads"]
        if isinstance(codex_threads, dict):
            summary["codex_local_threads"] = {
                "status": codex_threads.get("status"),
                "thread_count": codex_threads.get("thread_count"),
                "message_count": codex_threads.get("message_count"),
            }
    return summary


def _lifecycle_diagnostic_surface_row(
    surface_activity_summary: dict[str, Any],
    surface_id: str,
) -> dict[str, Any] | None:
    coverage = surface_activity_summary.get("requested_surface_coverage")
    if not isinstance(coverage, dict):
        return None
    surfaces = coverage.get("surfaces")
    if not isinstance(surfaces, list):
        return None
    for row in surfaces:
        if isinstance(row, dict) and row.get("surface_id") == surface_id:
            return row
    return None


def _lifecycle_diagnostic_activity_verdict(
    surface_activity_summary: dict[str, Any],
    surface_id: str,
) -> str:
    if surface_activity_summary.get("status") == "unavailable":
        return "unavailable"
    row = _lifecycle_diagnostic_surface_row(surface_activity_summary, surface_id)
    if isinstance(row, dict) and row.get("verdict"):
        return str(row["verdict"])
    coverage = surface_activity_summary.get("requested_surface_coverage")
    if isinstance(coverage, dict) and coverage.get("overall_verdict") == "unsupported":
        return "unsupported"
    return "absent"


def _fallback_surface_adapter_manifest(surface_id: str) -> dict[str, Any] | None:
    if surface_id not in SURFACE_ID_VALUES:
        return None
    return {
        "surface_id": surface_id,
        "context_enforcement_verdict": "unknown",
        "learning_capture_verdict": "unknown",
        "learning_probe_kind": "unknown",
        "learning_quality_mode": "unknown",
        "conformance_expectation": (
            "Static fallback only; canonical app.core.surface_lifecycle_contract manifest import was unavailable."
        ),
    }


def _lifecycle_diagnostic_get_surface_adapter(surface_id: str) -> Any:
    from app.core.surface_lifecycle_contract import get_surface_adapter

    return get_surface_adapter(surface_id)


def _lifecycle_diagnostic_adapter_manifest(surface_id: str) -> tuple[dict[str, Any] | None, str, str | None]:
    capability_error = None
    try:
        manifest = _lifecycle_diagnostic_get_surface_adapter(surface_id)
        if manifest is not None:
            return manifest.to_dict(), "surface_lifecycle_contract", None
    except Exception as exc:
        capability_error = f"{type(exc).__name__}: {exc}"

    fallback = _fallback_surface_adapter_manifest(surface_id)
    if fallback is not None:
        return fallback, "fallback_static_map", capability_error
    return None, "unavailable", capability_error


def _lifecycle_diagnostic_capabilities(
    *,
    surface_id: str,
    lifecycle_status: dict[str, Any],
    surface_activity_summary: dict[str, Any],
) -> dict[str, Any]:
    manifest, capability_source, capability_error = _lifecycle_diagnostic_adapter_manifest(surface_id)
    activity_verdict = _lifecycle_diagnostic_activity_verdict(surface_activity_summary, surface_id)
    user_visible_hosts: list[str] = []
    surface_canonicality = None
    manifest_supported = False
    lifecycle_enforcement_claim = lifecycle_status.get("lifecycle_enforcement_claim")
    if isinstance(manifest, dict):
        manifest_supported = True
        user_visible_hosts = list(manifest.get("user_visible_hosts") or [])
        surface_canonicality = manifest.get("canonicality")
        lifecycle_enforcement_claim = lifecycle_enforcement_claim or manifest.get("context_enforcement_verdict")
    lifecycle_reporter_supported = (
        _lifecycle_reporter_for(surface_id) is not None and lifecycle_status.get("status") != "unsupported"
    )
    lifecycle_reporter_observed = lifecycle_status.get("status") == "ok" and lifecycle_status.get(
        "overall_verdict"
    ) not in {"not_observed", "unsupported", "error"}
    return {
        "surface_id": surface_id,
        "known_surface": surface_id in SURFACE_ID_VALUES,
        "manifest_supported": manifest_supported,
        "activity_supported": surface_id in SURFACE_ID_VALUES and activity_verdict != "unsupported",
        "lifecycle_reporter_supported": lifecycle_reporter_supported,
        "lifecycle_reporter_observed": lifecycle_reporter_observed,
        "lifecycle_enforcement_claim": lifecycle_enforcement_claim or "unknown",
        "selector_supported": surface_id in SURFACE_ID_VALUES,
        "accepted_selector_keys": ["session_id", "origin_id", "workspace_id"],
        "capability_source": capability_source,
        "capability_error": capability_error,
        "adapter_manifest": manifest,
        "user_visible_hosts": user_visible_hosts,
        "surface_canonicality": surface_canonicality,
        "limitations": [
            "known_surface and manifest_supported are inventory claims, not runtime proof",
            "lifecycle_reporter_supported means a read-only adapter-state reporter exists; it does not mean lifecycle enforcement passed",
            "lifecycle_reporter_observed means matching read-only reporter evidence was found for this selector",
            "activity_supported means surface_activity can classify the requested surface; it does not prove semantic success",
            "user_visible_hosts are display and diagnostic labels, not accepted surface_id selectors",
        ],
    }


def _lifecycle_diagnostic_phase_is_degraded(phase: Any) -> bool:
    return isinstance(phase, dict) and (
        phase.get("status") in {"failed", "degraded"}
        or phase.get("verdict") in {"not_enforced", "pending", "mismatch", "unknown"}
    )


def _lifecycle_diagnostic_cross_plane_verdict(
    lifecycle_status: dict[str, Any],
    surface_activity_summary: dict[str, Any],
    surface_id: str,
) -> str:
    activity_verdict = _lifecycle_diagnostic_activity_verdict(surface_activity_summary, surface_id)
    lifecycle_verdict = str(lifecycle_status.get("overall_verdict") or lifecycle_status.get("status") or "unknown")
    lifecycle_status_value = str(lifecycle_status.get("status") or "unknown")
    proof_status = str(lifecycle_status.get("lifecycle_proof_status") or "")

    if lifecycle_status_value in {"error", "unsupported"} or activity_verdict in {"unavailable", "unsupported"}:
        return "diagnostic_degraded"
    if proof_status == "transport_timeout":
        return "lifecycle_proof_transport_timeout"
    if proof_status == "workstream_gate":
        return "lifecycle_workstream_gate"
    if proof_status == "semantic_sparse":
        return "lifecycle_semantic_sparse"

    activity_present = activity_verdict in {"covered", "sparse"}
    lifecycle_present = lifecycle_verdict in {"enforced", "partial", "failed"} or lifecycle_status_value == "ok"
    lifecycle_absent = lifecycle_verdict in {"not_observed"} or lifecycle_status_value == "not_found"

    if activity_present and lifecycle_verdict == "enforced":
        return "aligned_enforced"
    if activity_present and lifecycle_verdict in {"partial", "failed"}:
        return "activity_present_lifecycle_partial"
    if activity_present and lifecycle_absent:
        return "activity_present_lifecycle_not_observed"
    if not activity_present and lifecycle_present:
        return "activity_absent_lifecycle_present"
    if not activity_present and lifecycle_absent:
        return "both_absent"
    return "diagnostic_degraded"


def _lifecycle_diagnostic_surface_label(capabilities: dict[str, Any], surface_id: str) -> str:
    user_visible_hosts = capabilities.get("user_visible_hosts")
    if isinstance(user_visible_hosts, list) and user_visible_hosts:
        return str(user_visible_hosts[0])
    manifest = capabilities.get("adapter_manifest")
    if isinstance(manifest, dict):
        label = manifest.get("display_name") or manifest.get("label")
        if label:
            return str(label)
    return surface_id


def _lifecycle_diagnostic_reliability_operator_action(state: str) -> str:
    return {
        "reliable": "proceed",
        "proof_observed_partial": "use_with_caution",
        "proof_supported_unobserved": "use_fallback_artifacts",
        "configured_unobserved": "treat_as_not_observed",
        "unsupported_for_proof": "do_not_claim_reliable",
        "unknown_surface": "reject_selector",
        "unconfigured_unobserved": "do_not_claim_reliable",
    }.get(state, "do_not_claim_reliable")


def _lifecycle_diagnostic_reliability_label(state: str, surface_label: str) -> str:
    return {
        "reliable": f"{surface_label} lifecycle proof is reliable for this selector and window.",
        "proof_observed_partial": (
            f"{surface_label} lifecycle proof exists, but it is partial or not fully aligned."
        ),
        "proof_supported_unobserved": (
            f"{surface_label} supports lifecycle proof, but no matching proof was observed."
        ),
        "configured_unobserved": (
            f"{surface_label} is configured, but lifecycle proof was not observed."
        ),
        "unsupported_for_proof": (
            f"{surface_label} is configured, but this surface cannot currently prove lifecycle execution."
        ),
        "unknown_surface": f"{surface_label} is not a recognized Pith surface selector.",
        "unconfigured_unobserved": f"{surface_label} has no configured lifecycle reliability evidence.",
    }.get(state, f"{surface_label} lifecycle reliability could not be established.")


def _lifecycle_diagnostic_reliability_reasons(
    *,
    activity_verdict: str,
    lifecycle_verdict: str,
    lifecycle_status_value: str,
    proof_status: str,
    cross_plane_verdict: str,
    delivery_claim: Any,
    capabilities: dict[str, Any],
    route_status: str | None = None,
) -> list[str]:
    reasons = [
        f"activity_verdict:{activity_verdict}",
        f"lifecycle_verdict:{lifecycle_verdict}",
        f"lifecycle_status:{lifecycle_status_value}",
        f"cross_plane_verdict:{cross_plane_verdict}",
    ]
    if proof_status:
        reasons.append(f"lifecycle_proof_status:{proof_status}")
    if delivery_claim is not None:
        reasons.append(f"can_claim_context_delivered:{bool(delivery_claim)}")
    if not capabilities.get("known_surface"):
        reasons.append("surface_selector_unknown")
    if not capabilities.get("manifest_supported"):
        reasons.append("surface_manifest_missing")
    if not capabilities.get("lifecycle_reporter_supported"):
        reasons.append("lifecycle_reporter_unsupported")
    if capabilities.get("lifecycle_reporter_supported") and not capabilities.get("lifecycle_reporter_observed"):
        reasons.append("lifecycle_reporter_not_observed")
    if route_status:
        reasons.append(f"transport_route:{route_status}")
    return reasons


def _lifecycle_diagnostic_reliability_contract(
    *,
    surface_id: str,
    lifecycle_status: dict[str, Any],
    surface_activity_summary: dict[str, Any],
    capabilities: dict[str, Any],
    cross_plane_verdict: str,
) -> dict[str, Any]:
    activity_verdict = _lifecycle_diagnostic_activity_verdict(surface_activity_summary, surface_id)
    lifecycle_verdict = str(lifecycle_status.get("overall_verdict") or "unknown")
    lifecycle_status_value = str(lifecycle_status.get("status") or "unknown")
    proof_status = str(lifecycle_status.get("lifecycle_proof_status") or "")
    delivery_claim = lifecycle_status.get("can_claim_context_delivered")
    route_diagnostic = lifecycle_status.get("transport_route_diagnostic")
    route_status = (
        str(route_diagnostic.get("status"))
        if isinstance(route_diagnostic, dict) and route_diagnostic.get("status")
        else None
    )
    delivery_ok = True if delivery_claim is None else bool(delivery_claim)
    known_surface = bool(capabilities.get("known_surface"))
    manifest_supported = bool(capabilities.get("manifest_supported"))
    proof_supported = bool(capabilities.get("lifecycle_reporter_supported"))
    proof_observed = bool(capabilities.get("lifecycle_reporter_observed"))
    reliable = (
        known_surface
        and manifest_supported
        and proof_supported
        and proof_observed
        and activity_verdict == "covered"
        and lifecycle_verdict == "enforced"
        and cross_plane_verdict == "aligned_enforced"
        and delivery_ok
    )

    if not known_surface:
        state = "unknown_surface"
    elif not proof_supported and manifest_supported:
        state = "unsupported_for_proof"
    elif reliable:
        state = "reliable"
    elif proof_observed:
        state = "proof_observed_partial"
    elif proof_supported:
        state = "proof_supported_unobserved"
    elif manifest_supported:
        state = "configured_unobserved"
    else:
        state = "unconfigured_unobserved"

    surface_label = _lifecycle_diagnostic_surface_label(capabilities, surface_id)
    return {
        "schema_version": SURFACE_RELIABILITY_CONTRACT_SCHEMA_VERSION,
        "surface_id": surface_id,
        "surface_label": surface_label,
        "state": state,
        "reliable": state == "reliable",
        "operator_action": _lifecycle_diagnostic_reliability_operator_action(state),
        "user_label": _lifecycle_diagnostic_reliability_label(state, surface_label),
        "configured": known_surface and manifest_supported,
        "known_surface": known_surface,
        "manifest_supported": manifest_supported,
        "lifecycle_reporter_supported": proof_supported,
        "lifecycle_reporter_observed": proof_observed,
        "activity_verdict": activity_verdict,
        "lifecycle_verdict": lifecycle_verdict,
        "lifecycle_status": lifecycle_status_value,
        "lifecycle_proof_status": proof_status or None,
        "can_claim_context_delivered": delivery_claim,
        "cross_plane_verdict": cross_plane_verdict,
        "reasons": _lifecycle_diagnostic_reliability_reasons(
            activity_verdict=activity_verdict,
            lifecycle_verdict=lifecycle_verdict,
            lifecycle_status_value=lifecycle_status_value,
            proof_status=proof_status,
            cross_plane_verdict=cross_plane_verdict,
            delivery_claim=delivery_claim,
            capabilities=capabilities,
            route_status=route_status,
        ),
        "claim_boundary": (
            "selector_window_reliability_only; not semantic correctness, answer quality, or future-turn proof"
        ),
    }


def _lifecycle_diagnostic_matrix_surface_ids(
    surface_activity_summary: dict[str, Any],
) -> list[str]:
    coverage = surface_activity_summary.get("requested_surface_coverage")
    if not isinstance(coverage, dict):
        return []
    requested_surfaces = coverage.get("requested_surfaces")
    if not isinstance(requested_surfaces, list):
        return []
    surface_ids: list[str] = []
    for value in requested_surfaces:
        surface_id = str(value or "").strip()
        if surface_id and surface_id not in surface_ids:
            surface_ids.append(surface_id)
    return surface_ids


def _lifecycle_diagnostic_reliability_matrix(
    *,
    payload: dict[str, Any],
    primary_contract: dict[str, Any],
    surface_activity_summary: dict[str, Any],
    base_url: str,
    timeout: float,
    transport_mode: str,
) -> dict[str, Any] | None:
    surface_ids = _lifecycle_diagnostic_matrix_surface_ids(surface_activity_summary)
    if not surface_ids:
        return None

    rows: list[dict[str, Any]] = []
    for surface_id in surface_ids:
        if surface_id == primary_contract.get("surface_id"):
            rows.append(dict(primary_contract))
            continue

        surface_payload = {**payload, "surface_id": surface_id}
        if surface_id in SURFACE_ID_VALUES:
            lifecycle_status = _build_lifecycle_status(
                surface_payload,
                base_url=base_url,
                timeout=timeout,
                transport_mode=transport_mode,
            )
        else:
            lifecycle_status = _lifecycle_unsupported_status(surface_payload, surface_id)
        capabilities = _lifecycle_diagnostic_capabilities(
            surface_id=surface_id,
            lifecycle_status=lifecycle_status,
            surface_activity_summary=surface_activity_summary,
        )
        cross_plane_verdict = _lifecycle_diagnostic_cross_plane_verdict(
            lifecycle_status,
            surface_activity_summary,
            surface_id,
        )
        rows.append(
            _lifecycle_diagnostic_reliability_contract(
                surface_id=surface_id,
                lifecycle_status=lifecycle_status,
                surface_activity_summary=surface_activity_summary,
                capabilities=capabilities,
                cross_plane_verdict=cross_plane_verdict,
            )
        )

    return {
        "schema_version": SURFACE_RELIABILITY_CONTRACT_SCHEMA_VERSION,
        "surface_count": len(rows),
        "rows": rows,
        "limitations": [
            "Reliability is scoped to the selected surface, selector, and evidence window.",
            "Matrix rows reuse the requested surface_activity window.",
            "Reliable does not mean every future turn on the surface will be reliable.",
        ],
    }


def _lifecycle_diagnostic_selector_advice(
    payload: dict[str, Any],
    lifecycle_status: dict[str, Any],
    verdict: str,
) -> list[str]:
    advice: list[str] = []
    selector = lifecycle_status.get("selector")
    if not isinstance(selector, dict):
        selector = {}
    origin_id = str(selector.get("origin_id") or payload.get("origin_id") or "")
    if origin_id.startswith("codex_") and verdict == "activity_present_lifecycle_not_observed":
        advice.append(
            "Codex lifecycle_status normalizes hyphen/underscore variants for codex_ workspace-slug origins, but this selector still did not match hook state; retry with session_id or workspace_id."
        )
    if verdict in {"activity_present_lifecycle_not_observed", "both_absent"}:
        advice.append(
            "Run lifecycle_status with session_id first; use origin_id or workspace_id only when they are known to match adapter-state selectors."
        )
    if verdict == "activity_absent_lifecycle_present":
        advice.append(
            "Lifecycle hook state exists, but requested surface activity is absent; widen since/until or requested_surfaces before treating this as no activity."
        )
    return advice


def _lifecycle_diagnostic_selector_evidence(
    payload: dict[str, Any],
    lifecycle_status: dict[str, Any],
) -> dict[str, Any]:
    selector = lifecycle_status.get("selector")
    if not isinstance(selector, dict):
        selector = {}
    requested_keys = [key for key in ("session_id", "origin_id", "workspace_id") if payload.get(key)]
    matched_keys = [key for key in ("session_id", "origin_id", "workspace_id") if selector.get(key)]
    origin_id = str(selector.get("origin_id") or payload.get("origin_id") or "")
    looks_agents_origin = origin_id.startswith("codex_")
    status_ok = lifecycle_status.get("status") == "ok"
    return {
        "requested_selector_keys": requested_keys,
        "matched_selector_keys": matched_keys,
        "selector": selector,
        "origin_id_looks_agents_local_api": looks_agents_origin,
        "generated_hook_origin_equivalent": status_ok if looks_agents_origin else None,
        "notes": [
            "Codex lifecycle_status normalizes hyphen/underscore variants for codex_ workspace-slug origins; session_id or workspace_id remains preferred when available."
        ]
        if looks_agents_origin
        else [],
    }


def _lifecycle_diagnostic_commands(
    payload: dict[str, Any],
    surface_payload: dict[str, Any],
) -> list[str]:
    selector = {
        key: payload[key]
        for key in ("surface_id", "session_id", "origin_id", "workspace_id")
        if payload.get(key) is not None
    }
    return [
        "~/.pith/bin/pith api lifecycle_status --stdin-json",
        "~/.pith/bin/pith api surface_activity --stdin-json",
        json.dumps(
            {
                "lifecycle_status_payload": selector,
                "surface_activity_payload": surface_payload,
            },
            sort_keys=True,
        ),
    ]


def _lifecycle_diagnostic_legacy_codes(surface_id: str, *codes: str) -> list[str]:
    if surface_id != "codex_local_api":
        return []
    return list(codes)


def _lifecycle_diagnostic_finding(
    code: str,
    severity: str,
    message: str,
    *,
    legacy_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "legacy_codes": list(legacy_codes or []),
    }


def _lifecycle_diagnostic_findings(
    payload: dict[str, Any],
    lifecycle_status: dict[str, Any],
    surface_activity_summary: dict[str, Any],
    verdict: str,
    surface_id: str,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    if surface_activity_summary.get("status") == "unavailable":
        findings.append(
            _lifecycle_diagnostic_finding(
                "SURFACE_ACTIVITY_UNAVAILABLE",
                "warning",
                "surface_activity evidence was unavailable; lifecycle evidence is preserved but the cross-plane verdict is degraded.",
            )
        )

    if lifecycle_status.get("status") == "unsupported" or lifecycle_status.get("overall_verdict") == "unsupported":
        findings.append(
            _lifecycle_diagnostic_finding(
                "LIFECYCLE_REPORTER_UNSUPPORTED",
                "warning",
                f"{surface_id} does not have a read-only lifecycle adapter-state reporter.",
            )
        )

    proof_status = str(lifecycle_status.get("lifecycle_proof_status") or "")
    if verdict == "lifecycle_proof_transport_timeout":
        findings.append(
            _lifecycle_diagnostic_finding(
                "LIFECYCLE_PROOF_TRANSPORT_TIMEOUT",
                "warning",
                f"{surface_id} lifecycle proof timed out before model-visible context was confirmed.",
                legacy_codes=_lifecycle_diagnostic_legacy_codes(surface_id, "CODEX_CONTEXT_DEGRADED"),
            )
        )
    if proof_status == "health_down":
        findings.append(
            _lifecycle_diagnostic_finding(
                "LIFECYCLE_PROOF_HEALTH_DOWN",
                "warning",
                f"{surface_id} could not reach Pith lifecycle before model-visible context was confirmed.",
            )
        )
    if verdict == "lifecycle_workstream_gate":
        findings.append(
            _lifecycle_diagnostic_finding(
                "LIFECYCLE_WORKSTREAM_GATE",
                "warning",
                f"{surface_id} lifecycle returned, but workstream activation needs a decision.",
            )
        )
    if verdict == "lifecycle_semantic_sparse":
        findings.append(
            _lifecycle_diagnostic_finding(
                "LIFECYCLE_SEMANTIC_SPARSE",
                "info",
                f"{surface_id} lifecycle proof succeeded, but context is sparse, stale, or contested.",
            )
        )

    if verdict == "activity_present_lifecycle_not_observed":
        findings.append(
            _lifecycle_diagnostic_finding(
                "ACTIVITY_PRESENT_LIFECYCLE_NOT_OBSERVED",
                "warning",
                f"{surface_id} activity is present, but no matching lifecycle adapter-state was observed.",
                legacy_codes=_lifecycle_diagnostic_legacy_codes(
                    surface_id,
                    "CODEX_ACTIVITY_PRESENT_LIFECYCLE_NOT_OBSERVED",
                ),
            )
        )

    fallback_evidence = lifecycle_status.get("selector_fallback_evidence")
    if isinstance(fallback_evidence, dict) and int(fallback_evidence.get("candidate_count") or 0) > 0:
        findings.append(
            _lifecycle_diagnostic_finding(
                "STALE_LIFECYCLE_STATE_CANDIDATE",
                "info",
                (
                    f"{surface_id} has diagnostic-only same-origin/workspace lifecycle state, "
                    "but no matching current selector proof."
                ),
                legacy_codes=_lifecycle_diagnostic_legacy_codes(
                    surface_id,
                    "CODEX_STALE_LIFECYCLE_STATE_CANDIDATE",
                ),
            )
        )

    if verdict == "activity_present_lifecycle_partial":
        findings.append(
            _lifecycle_diagnostic_finding(
                "ACTIVITY_PRESENT_LIFECYCLE_PARTIAL",
                "warning",
                f"{surface_id} activity is present, but lifecycle enforcement is partial or failed.",
                legacy_codes=_lifecycle_diagnostic_legacy_codes(
                    surface_id,
                    "CODEX_ACTIVITY_PRESENT_LIFECYCLE_PARTIAL",
                ),
            )
        )

    if _lifecycle_diagnostic_phase_is_degraded(lifecycle_status.get("context_phase")):
        findings.append(
            _lifecycle_diagnostic_finding(
                "CONTEXT_DEGRADED",
                "warning",
                f"{surface_id} context phase is degraded or not enforced.",
                legacy_codes=_lifecycle_diagnostic_legacy_codes(
                    surface_id,
                    "CODEX_CONTEXT_DEGRADED",
                ),
            )
        )

    coverage = surface_activity_summary.get("requested_surface_coverage")
    if isinstance(coverage, dict):
        for row in coverage.get("surfaces", []):
            if isinstance(row, dict) and row.get("surface_id") == "local_api_cli" and row.get("verdict") == "absent":
                findings.append(
                    _lifecycle_diagnostic_finding(
                        "COMPARISON_SURFACE_ABSENT_EXPECTED",
                        "info",
                        "local_api_cli activity is absent in the default comparison set; this can be expected when the active turn uses codex_local_api.",
                        legacy_codes=["LOCAL_API_CLI_ABSENT_EXPECTED"],
                    )
                )
                break

    selector_advice = _lifecycle_diagnostic_selector_advice(payload, lifecycle_status, verdict)
    if any("codex_ workspace-slug origins" in item for item in selector_advice):
        findings.append(
            _lifecycle_diagnostic_finding(
                "SELECTOR_MISMATCH_SUSPECTED",
                "warning",
                "The selector looks like a Codex workspace-slug origin, but normalized variants did not match hook state.",
                legacy_codes=_lifecycle_diagnostic_legacy_codes(
                    surface_id,
                    "CODEX_SELECTOR_MISMATCH_SUSPECTED",
                ),
            )
        )
    return findings


def _build_lifecycle_diagnostic(
    payload: dict | None,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    transport_mode: str = "exec_http_fallback",
) -> dict[str, Any]:
    body = dict(payload or {})
    raw_surface_id = str(body.get("surface_id") or "").strip()
    normalized_surface_id = _normalize_surface_id(raw_surface_id)
    body["surface_id"] = (
        normalized_surface_id
        or (raw_surface_id.lower() if raw_surface_id else LIFECYCLE_DIAGNOSTIC_DEFAULT_SURFACE_ID)
    )
    lifecycle_status = _build_lifecycle_status(
        body,
        base_url=base_url,
        timeout=timeout,
        transport_mode=transport_mode,
    )
    if lifecycle_status.get("status") == "error":
        surface_activity_summary = {
            "status": "not_requested",
            "reason": "lifecycle_selector_error",
        }
        capabilities = _lifecycle_diagnostic_capabilities(
            surface_id=body["surface_id"],
            lifecycle_status=lifecycle_status,
            surface_activity_summary=surface_activity_summary,
        )
        reliability = _lifecycle_diagnostic_reliability_contract(
            surface_id=body["surface_id"],
            lifecycle_status=lifecycle_status,
            surface_activity_summary=surface_activity_summary,
            capabilities=capabilities,
            cross_plane_verdict="diagnostic_degraded",
        )
        return {
            "schema_version": LIFECYCLE_DIAGNOSTIC_SCHEMA_VERSION,
            "status": "error",
            "code": lifecycle_status.get("code"),
            "surface_id": body["surface_id"],
            "selector": lifecycle_status.get("selector", {}),
            "selector_evidence": _lifecycle_diagnostic_selector_evidence(body, lifecycle_status),
            "selector_fallback_evidence": lifecycle_status.get("selector_fallback_evidence"),
            "capabilities": capabilities,
            "surface_reliability": reliability,
            "claim_boundary": LIFECYCLE_DIAGNOSTIC_CLAIM_BOUNDARY,
            "lifecycle_status": lifecycle_status,
            "surface_activity": surface_activity_summary,
            "cross_plane_analysis": {
                "verdict": "diagnostic_degraded",
                "findings": [],
                "selector_advice": ["Provide at least one of session_id, origin_id, or workspace_id."],
                "next_diagnostic_commands": ["~/.pith/bin/pith api lifecycle_diagnostic --stdin-json"],
            },
        }

    surface_activity = _fetch_lifecycle_diagnostic_surface_activity(
        body,
        base_url=base_url,
        timeout=timeout,
        transport_mode=transport_mode,
    )
    surface_activity_summary = _lifecycle_diagnostic_surface_activity_summary(surface_activity)
    verdict = _lifecycle_diagnostic_cross_plane_verdict(
        lifecycle_status,
        surface_activity_summary,
        body["surface_id"],
    )
    surface_payload = _lifecycle_diagnostic_surface_activity_payload(body)
    selector_advice = _lifecycle_diagnostic_selector_advice(body, lifecycle_status, verdict)
    findings = _lifecycle_diagnostic_findings(
        body,
        lifecycle_status,
        surface_activity_summary,
        verdict,
        body["surface_id"],
    )

    status = "degraded" if verdict == "diagnostic_degraded" else "ok"
    selector_evidence = _lifecycle_diagnostic_selector_evidence(body, lifecycle_status)
    capabilities = _lifecycle_diagnostic_capabilities(
        surface_id=body["surface_id"],
        lifecycle_status=lifecycle_status,
        surface_activity_summary=surface_activity_summary,
    )
    reliability = _lifecycle_diagnostic_reliability_contract(
        surface_id=body["surface_id"],
        lifecycle_status=lifecycle_status,
        surface_activity_summary=surface_activity_summary,
        capabilities=capabilities,
        cross_plane_verdict=verdict,
    )
    result = {
        "schema_version": LIFECYCLE_DIAGNOSTIC_SCHEMA_VERSION,
        "status": status,
        "surface_id": body["surface_id"],
        "selector": lifecycle_status.get("selector", {}),
        "selector_evidence": selector_evidence,
        "selector_fallback_evidence": lifecycle_status.get("selector_fallback_evidence"),
        "capabilities": capabilities,
        "surface_reliability": reliability,
        "claim_boundary": LIFECYCLE_DIAGNOSTIC_CLAIM_BOUNDARY,
        "lifecycle_status": lifecycle_status,
        "surface_activity": surface_activity_summary,
        "cross_plane_analysis": {
            "verdict": verdict,
            "activity_verdict": _lifecycle_diagnostic_activity_verdict(
                surface_activity_summary,
                body["surface_id"],
            ),
            "lifecycle_verdict": lifecycle_status.get("overall_verdict"),
            "findings": findings,
            "selector_advice": selector_advice,
            "next_diagnostic_commands": _lifecycle_diagnostic_commands(body, surface_payload),
        },
    }
    reliability_matrix = _lifecycle_diagnostic_reliability_matrix(
        payload=body,
        primary_contract=reliability,
        surface_activity_summary=surface_activity_summary,
        base_url=base_url,
        timeout=timeout,
        transport_mode=transport_mode,
    )
    if reliability_matrix is not None:
        result["surface_reliability_matrix"] = reliability_matrix
    return result


def _build_headers(operation: str, transport_mode: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Pith-Transport": transport_mode,
    }
    if operation not in AUTH_EXEMPT_OPERATIONS:
        api_key = _resolve_api_key()
        if not api_key:
            raise SystemExit("PITH_API_KEY unavailable for authenticated fallback call")
        headers["X-API-Key"] = api_key
    return headers


def _operation_catalog() -> list[dict[str, Any]]:
    operations = [
        {
            "operation": operation,
            "method": method,
            "path": path,
            "auth_required": operation not in AUTH_EXEMPT_OPERATIONS,
            "example_available": operation in OPERATION_EXAMPLES,
            "schema_available": operation in OPERATION_SCHEMAS,
        }
        for operation, (method, path) in sorted(ALLOWED.items())
    ]
    operations.extend(
        {
            "operation": operation,
            "method": "LOCAL",
            "path": "",
            "auth_required": False,
            "example_available": operation in OPERATION_EXAMPLES,
            "schema_available": operation in OPERATION_SCHEMAS,
        }
        for operation in sorted(PSEUDO_OPERATIONS)
        if operation != "list"
    )
    return sorted(operations, key=lambda item: item["operation"])


def _validate_operation(parser: argparse.ArgumentParser, operation: str) -> None:
    if operation in ALLOWED or operation in PSEUDO_OPERATIONS:
        return
    choices = ", ".join(sorted([*ALLOWED, *PSEUDO_OPERATIONS]))
    parser.error(f"invalid choice: {operation!r} (choose from {choices})")


def _operation_discovery_payload(operation: str, kind: str) -> dict[str, Any]:
    registry = OPERATION_EXAMPLES if kind == "example" else OPERATION_SCHEMAS
    payload = registry.get(operation)
    if payload is not None:
        return payload
    available = ", ".join(sorted(registry)) or "none"
    raise SystemExit(f"No local {kind} registered for operation {operation!r}. Available operations: {available}")


def _response_detail(body: Any) -> str:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail, default=str)
        return json.dumps(body, default=str)
    return str(body)


def _is_retryable_conversation_turn_startup_503(
    operation: str,
    status_code: int,
    body: Any,
) -> bool:
    if operation != "conversation_turn" or status_code != 503:
        return False
    detail = _response_detail(body).lower()
    return "retrieval initialization" in detail or "retrieval recovery" in detail or "server startup" in detail


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return min(max(0.0, float(raw)), CONVERSATION_TURN_STARTUP_RETRY_CAP_S)
        except ValueError:
            pass
    return min(0.5 * (2**attempt), CONVERSATION_TURN_STARTUP_RETRY_CAP_S)


def _transport_iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _transport_event(event: str, **kwargs: Any) -> None:
    entry = {
        "ts": _transport_iso_now(),
        "event": event,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "api_url": kwargs.pop("api_url", None),
        **kwargs,
    }
    if entry["api_url"] is None:
        entry.pop("api_url")
    try:
        TRANSPORT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRANSPORT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


def _emit_lifecycle_api_call_event(
    operation: str,
    payload: dict[str, Any] | None,
    result: Any,
    *,
    transport_mode: str,
    api_status: Any = None,
    api_url: str | None = None,
) -> None:
    if operation not in {"conversation_turn", "checkpoint", "session_end", "session_learn"}:
        return
    args = payload or {}
    body = result if isinstance(result, dict) else {}
    surface_id = _normalize_surface_id(args.get("surface_id")) or _default_surface_id(operation) or "local_api_cli"
    if operation in {"checkpoint", "session_end", "session_learn"} and not _normalize_surface_id(
        args.get("surface_id")
    ):
        surface_id = "local_api_cli"
    auto_learned = body.get("auto_learned")
    auto_learned_events = auto_learned.get("events") if isinstance(auto_learned, dict) else None
    _transport_event(
        "lifecycle_api_call",
        operation=operation,
        transport_mode=transport_mode,
        surface_id=surface_id,
        session_id=args.get("session_id"),
        resolved_session_id=body.get("resolved_session_id") or body.get("session_id"),
        origin_id=args.get("origin_id") or body.get("origin_id"),
        workspace_id=args.get("workspace_id") or body.get("workspace_id"),
        request_id=args.get("request_id") or body.get("request_id"),
        status=body.get("status") or ("error" if body.get("error") is True else "ok"),
        api_status=api_status,
        error=bool(body.get("error") is True),
        previous_response_present=bool(args.get("previous_response")),
        previous_message_present=bool(args.get("previous_message")),
        extracted_concepts_present=bool(args.get("extracted_concepts_json")),
        auto_learned=bool(auto_learned),
        learning_events=body.get("learning_events") if body.get("learning_events") is not None else auto_learned_events,
        accepted_learning_events=body.get("accepted_learning_events"),
        checkpoint_task_id=args.get("task_id") or body.get("task_id"),
        api_url=api_url,
    )


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _emit_workstream_activation_api_event(
    operation: str,
    payload: dict[str, Any] | None,
    result: Any,
    *,
    transport_mode: str,
    elapsed_ms: float,
    api_status: Any = None,
    api_url: str | None = None,
) -> None:
    if operation != "workstreams":
        return
    args = payload or {}
    action = args.get("action")
    if action not in {
        "ensure_workstream_activation",
        "active_workstream",
        "workstream_context",
        "classify_workstreams",
    }:
        return
    body = result if isinstance(result, dict) else {}
    response_body = body.get("body") if isinstance(body.get("body"), dict) else body
    decision = response_body.get("activation_decision") if isinstance(response_body, dict) else None
    decision_body = decision if isinstance(decision, dict) else {}
    if isinstance(response_body, dict) and response_body.get("detail") and not response_body.get("status"):
        reason = response_body.get("detail")
    else:
        reason = response_body.get("reason") if isinstance(response_body, dict) else None
    _transport_event(
        "workstream_activation_api_call",
        timestamp=_transport_iso_now(),
        operation=operation,
        transport_mode=transport_mode,
        action=action,
        mode=args.get("mode"),
        read_only=response_body.get("read_only") if isinstance(response_body, dict) else None,
        status=response_body.get("status") if isinstance(response_body, dict) else body.get("code"),
        reason=reason,
        api_status=api_status,
        elapsed_ms=round(elapsed_ms, 2),
        origin_id_present=bool(args.get("origin_id")),
        session_id_present=bool(args.get("session_id")),
        current_task_id_present=bool(args.get("current_task_id")),
        thread_id_present=bool(args.get("thread_id")),
        active_binding_present=bool(isinstance(response_body, dict) and response_body.get("active_binding")),
        explicit_skip_present=bool(isinstance(response_body, dict) and response_body.get("explicit_skip")),
        decision_kind=decision_body.get("decision_kind"),
        required_action=decision_body.get("required_action"),
        recommended_next_action=decision_body.get("recommended_next_action"),
        parent_choice_state=decision_body.get("parent_choice_state"),
        active_binding_related=decision_body.get("active_binding_related"),
        skip_exception_kind=decision_body.get("skip_exception_kind"),
        skip_requires_reason=decision_body.get("skip_requires_reason"),
        recommended_count=(_list_len(response_body.get("recommended")) if isinstance(response_body, dict) else 0),
        advisory_candidate_count=(
            _list_len(response_body.get("advisory_candidates")) if isinstance(response_body, dict) else 0
        ),
        possible_match_count=(
            _list_len(response_body.get("possible_matches")) if isinstance(response_body, dict) else 0
        ),
        proof_or_maintenance_count=(
            _list_len(response_body.get("proof_or_maintenance")) if isinstance(response_body, dict) else 0
        ),
        needs_review_count=(_list_len(response_body.get("needs_review")) if isinstance(response_body, dict) else 0),
        error=bool(isinstance(body, dict) and body.get("error") is True),
        api_url=api_url,
    )


def _emit_workstream_activation_hint_event(
    operation: str,
    payload: dict[str, Any] | None,
    result: Any,
    *,
    transport_mode: str,
    elapsed_ms: float,
    api_status: Any = None,
    api_url: str | None = None,
) -> None:
    if operation != "conversation_turn" or not isinstance(result, dict):
        return
    hint = result.get("workstream_activation")
    if not isinstance(hint, dict):
        return
    args = payload or {}
    decision = hint.get("activation_decision") if isinstance(hint.get("activation_decision"), dict) else {}
    _transport_event(
        "workstream_activation_hint",
        timestamp=_transport_iso_now(),
        operation=operation,
        transport_mode=transport_mode,
        activation_state=hint.get("activation_state"),
        status=hint.get("status"),
        reason=hint.get("reason"),
        read_only=hint.get("read_only"),
        decision_needed=hint.get("decision_needed"),
        origin_id_present=bool(args.get("origin_id")),
        session_id_present=bool(args.get("session_id")),
        current_task_id_present=bool(args.get("current_task_id")),
        active_binding_present=bool(hint.get("active_binding")),
        explicit_skip_present=bool(hint.get("explicit_skip")),
        decision_kind=decision.get("decision_kind"),
        required_action=decision.get("required_action"),
        recommended_next_action=decision.get("recommended_next_action"),
        parent_choice_state=decision.get("parent_choice_state"),
        active_binding_related=decision.get("active_binding_related"),
        skip_exception_kind=decision.get("skip_exception_kind"),
        skip_requires_reason=decision.get("skip_requires_reason"),
        advisory_candidate_count=decision.get("advisory_candidate_count"),
        api_status=api_status,
        elapsed_ms=round(elapsed_ms, 2),
        api_url=api_url,
    )


def _emit_workstream_activation_gate_event(
    operation: str,
    payload: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    *,
    transport_mode: str,
    elapsed_ms: float,
    api_status: Any = None,
    api_url: str | None = None,
) -> None:
    if operation != "conversation_turn" or not isinstance(gate, dict):
        return
    args = payload or {}
    _transport_event(
        "workstream_activation_gate",
        timestamp=_transport_iso_now(),
        operation=operation,
        transport_mode=transport_mode,
        status=gate.get("status"),
        activation_state=gate.get("activation_state"),
        decision_kind=gate.get("decision_kind"),
        reason=gate.get("reason"),
        required_action=gate.get("required_action"),
        recommended_next_action=gate.get("recommended_next_action"),
        parent_choice_state=gate.get("parent_choice_state"),
        blocked=gate.get("status") == "blocked",
        read_only=gate.get("read_only"),
        active_binding_related=gate.get("active_binding_related"),
        skip_exception_kind=gate.get("skip_exception_kind"),
        skip_requires_reason=gate.get("skip_requires_reason"),
        origin_id_present=bool(args.get("origin_id")),
        session_id_present=bool(args.get("session_id")),
        current_task_id_present=bool(args.get("current_task_id")),
        candidate_detail_available=gate.get("candidate_detail_available"),
        recommended_count=gate.get("recommended_count"),
        advisory_candidate_count=gate.get("advisory_candidate_count"),
        possible_match_count=gate.get("possible_match_count"),
        proof_or_maintenance_count=gate.get("proof_or_maintenance_count"),
        needs_review_count=gate.get("needs_review_count"),
        api_status=api_status,
        elapsed_ms=round(elapsed_ms, 2),
        api_url=api_url,
    )


def _workstream_activation_gate_applies(operation: str, payload: dict[str, Any] | None) -> bool:
    return operation == "conversation_turn" and isinstance(payload, dict) and bool(payload.get("current_task_id"))


def _activation_gate_counts(source: dict[str, Any]) -> dict[str, int]:
    return {
        "recommended_count": _list_len(source.get("recommended")),
        "advisory_candidate_count": _list_len(source.get("advisory_candidates")),
        "possible_match_count": _list_len(source.get("possible_matches")),
        "proof_or_maintenance_count": _list_len(source.get("proof_or_maintenance")),
        "needs_review_count": _list_len(source.get("needs_review")),
    }


def _activation_decision_contract(decision: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "recommended_next_action",
        "decision_options",
        "parent_choice_state",
        "suggested_child_metadata",
        "suggested_create_metadata",
        "advisory_candidate_count",
    )
    return {key: decision[key] for key in allowed if key in decision}


def _activation_decision(source: dict[str, Any]) -> dict[str, Any]:
    decision = source.get("activation_decision")
    return decision if isinstance(decision, dict) else {}


def _active_binding_related_from_decision(decision: dict[str, Any]) -> bool | None:
    active_binding_related = decision.get("active_binding_related")
    if isinstance(active_binding_related, bool):
        return active_binding_related
    if isinstance(active_binding_related, str):
        normalized = active_binding_related.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False

    decision_kind = str(decision.get("decision_kind") or "").strip().lower()
    if decision_kind == "active_binding_related":
        return True
    if decision_kind == "active_binding_unrelated":
        return False
    return None


def _active_binding_related_from_render(render: Any) -> bool | None:
    if not isinstance(render, dict):
        return None
    decision = str(render.get("decision") or "").strip().lower()
    if decision == "render":
        return True
    if decision == "suppress":
        return False
    return None


def _activation_gate_from_state(
    source: dict[str, Any],
    *,
    fallback_used: bool,
    active_workstream_render: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = _activation_decision(source)
    if source.get("active_binding"):
        active_binding_related = _active_binding_related_from_decision(decision)
        if active_binding_related is True:
            return {
                "status": "passed",
                "activation_state": "active_binding",
                "decision_kind": "active_binding_related",
                "reason": "active_binding_related",
                "read_only": True,
                "required_action": "none",
                "active_binding_related": True,
                "fallback_candidate_lookup": fallback_used,
            }
        if active_binding_related is False:
            required_action = str(decision.get("required_action") or "choose_bind_or_create")
            decision_kind = str(decision.get("decision_kind") or "active_binding_unrelated")
            if decision_kind == "active_binding_unknown":
                decision_kind = "active_binding_unrelated"
        else:
            active_binding_related = _active_binding_related_from_render(active_workstream_render)
            if active_binding_related is True:
                return {
                    "status": "passed",
                    "activation_state": "active_binding",
                    "decision_kind": "active_binding_related",
                    "reason": "active_binding_related",
                    "read_only": True,
                    "required_action": "none",
                    "active_binding_related": True,
                    "fallback_candidate_lookup": fallback_used,
                }
            if active_binding_related is False:
                required_action = "choose_bind_or_create"
                decision_kind = "active_binding_unrelated"
            else:
                required_action = str(decision.get("required_action") or "confirm_active_binding_or_create")
                decision_kind = str(decision.get("decision_kind") or "active_binding_unknown")
        if active_binding_related is False or decision_kind == "active_binding_unknown":
            reason = "active_workstream_unrelated" if active_binding_related is False else "active_binding_unknown"
            gate = {
                "status": "blocked",
                "activation_state": decision_kind,
                "decision_kind": decision_kind,
                "reason": reason,
                "read_only": True,
                "required_action": required_action,
                "candidate_detail_available": bool(source.get("candidate_detail_available", True)),
                "fallback_candidate_lookup": fallback_used,
                "active_binding_related": active_binding_related,
            }
            gate.update(_activation_decision_contract(decision))
            return gate
        return {
            "status": "passed",
            "activation_state": "active_binding",
            "decision_kind": "active_binding_unknown",
            "reason": "active_binding_unknown",
            "read_only": True,
            "required_action": "none",
            "active_binding_related": None,
            "fallback_candidate_lookup": fallback_used,
        }
    if source.get("explicit_skip"):
        skip = source.get("explicit_skip")
        skip_exception_kind = skip.get("skip_exception_kind") if isinstance(skip, dict) else None
        return {
            "status": "passed",
            "activation_state": "explicit_skip",
            "decision_kind": decision.get("decision_kind") or "explicit_skip_exception",
            "reason": "explicit_skip_exception",
            "read_only": True,
            "required_action": "none",
            "skip_exception_kind": skip_exception_kind or decision.get("skip_exception_kind"),
            "fallback_candidate_lookup": fallback_used,
        }

    activation_state = str(decision.get("decision_kind") or source.get("activation_state") or "decision_needed")
    reason = activation_state if activation_state not in {"decision_needed"} else "activation_decision_required"
    if not decision and activation_state == "decision_needed":
        if _list_len(source.get("recommended")) > 0:
            activation_state = "bind_or_create_required"
            reason = "bind_or_create_required"
        elif _list_len(source.get("advisory_candidates")) > 0:
            activation_state = "operator_review_required"
            reason = "operator_review_required"
        else:
            activation_state = "create_required"
            reason = "create_required"
    if activation_state in {"disabled", "unavailable"}:
        reason = str(source.get("reason") or activation_state)
    gate = {
        "status": "blocked",
        "activation_state": activation_state,
        "decision_kind": decision.get("decision_kind") or activation_state,
        "reason": reason,
        "read_only": True,
        "required_action": decision.get("required_action")
        or (
            "choose_bind_or_create"
            if _list_len(source.get("recommended")) > 0
            else (
                "create_or_skip_or_confirm_candidate"
                if _list_len(source.get("advisory_candidates")) > 0
                else "create_and_bind_workstream"
            )
        ),
        "candidate_detail_available": bool(source.get("candidate_detail_available", True)),
        "fallback_candidate_lookup": fallback_used,
        "active_binding_related": decision.get("active_binding_related"),
        "skip_requires_reason": decision.get("skip_requires_reason"),
    }
    gate.update(_activation_gate_counts(source))
    gate.update(_activation_decision_contract(decision))
    return gate


def _candidate_payload_for_gate(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "action": "ensure_workstream_activation",
        "mode": "candidate",
        "current_task_id": payload.get("current_task_id"),
    }
    if payload.get("origin_id"):
        candidate["origin_id"] = payload.get("origin_id")
    if payload.get("session_id"):
        candidate["session_id"] = payload.get("session_id")
    if payload.get("message"):
        candidate["situation"] = payload.get("message")
    return {key: value for key, value in candidate.items() if value is not None}


def _resolve_workstream_activation_gate(
    operation: str,
    payload: dict[str, Any] | None,
    body: Any,
    *,
    base_url: str,
    headers: dict[str, str],
    timeout: float,
    transport_mode: str,
    started: float,
) -> dict[str, Any] | None:
    if not _workstream_activation_gate_applies(operation, payload):
        return None
    args = payload or {}
    if not args.get("origin_id") and not args.get("session_id"):
        return {
            "status": "blocked",
            "activation_state": "unavailable",
            "reason": "authority_required",
            "read_only": True,
            "required_action": "provide_origin_id_or_session_id",
            "candidate_detail_available": False,
            "fallback_candidate_lookup": False,
        }

    if isinstance(body, dict) and isinstance(body.get("workstream_activation"), dict):
        render = (
            body.get("active_workstream_render") if isinstance(body.get("active_workstream_render"), dict) else None
        )
        return _activation_gate_from_state(
            body["workstream_activation"],
            fallback_used=False,
            active_workstream_render=render,
        )

    candidate_payload = _candidate_payload_for_gate(args)
    candidate_started = time.perf_counter()
    api_status: Any = None
    try:
        response = requests.post(
            f"{base_url}{ALLOWED['workstreams'][1]}",
            json=candidate_payload,
            headers=headers,
            timeout=timeout,
        )
        api_status = response.status_code
        try:
            candidate_body: Any = response.json()
        except ValueError:
            candidate_body = {"error": True, "code": "NON_JSON_RESPONSE"}
        if response.status_code >= 400:
            candidate_body = {
                "error": True,
                "status_code": response.status_code,
                "body": candidate_body,
            }
    except requests.RequestException as exc:
        candidate_body = {
            "error": True,
            "code": "CONNECTION_FAILED",
            "message": str(exc),
        }
        api_status = "connection_failed"

    elapsed_ms = (time.perf_counter() - candidate_started) * 1000
    _emit_workstream_activation_api_event(
        "workstreams",
        candidate_payload,
        candidate_body,
        transport_mode=transport_mode,
        elapsed_ms=elapsed_ms,
        api_status=api_status,
        api_url=base_url,
    )
    if not isinstance(candidate_body, dict) or candidate_body.get("error"):
        return {
            "status": "blocked",
            "activation_state": "unavailable",
            "reason": "activation_candidate_lookup_failed",
            "read_only": True,
            "required_action": "retry_or_run_pith_api_workstreams_candidate",
            "candidate_detail_available": False,
            "fallback_candidate_lookup": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    render = (
        body.get("active_workstream_render")
        if isinstance(body, dict) and isinstance(body.get("active_workstream_render"), dict)
        else None
    )
    return _activation_gate_from_state(
        candidate_body,
        fallback_used=True,
        active_workstream_render=render,
    )


def _active_workstream_normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _active_workstream_tokens(value: Any) -> set[str]:
    text = _active_workstream_normalize_text(value)
    tokens = set(re.findall(r"[a-z0-9_:-]{3,}", text))
    return tokens - _ACTIVE_WORKSTREAM_STOPWORDS - _ACTIVE_WORKSTREAM_TRIGGER_WORDS


def _active_workstream_text_fields(active_workstream: dict[str, Any]) -> list[str]:
    workstream = active_workstream.get("workstream") or {}
    metadata = workstream.get("metadata") or {}
    fields = [
        workstream.get("title", ""),
        metadata.get("current_objective", ""),
        metadata.get("current_summary", ""),
        metadata.get("next_action", ""),
    ]
    blockers = metadata.get("blockers") or []
    if isinstance(blockers, list):
        fields.extend(str(blocker) for blocker in blockers)
    return [str(field) for field in fields if str(field or "").strip()]


def _active_workstream_has_topic_overlap(message: str, active_workstream: dict[str, Any]) -> bool:
    message_tokens = _active_workstream_tokens(message)
    if not message_tokens:
        return False
    workstream_tokens = _active_workstream_tokens(" ".join(_active_workstream_text_fields(active_workstream)))
    return bool(message_tokens & workstream_tokens)


def _active_workstream_has_trigger(message: str) -> bool:
    tokens = _active_workstream_tokens(message) | set(
        re.findall(r"[a-z0-9_:-]{3,}", _active_workstream_normalize_text(message))
    )
    return bool(tokens & (_ACTIVE_WORKSTREAM_TRIGGER_WORDS | _ACTIVE_WORKSTREAM_WORKFLOW_WORDS))


def _active_workstream_explicit_inspection(message: str) -> bool:
    text = _active_workstream_normalize_text(message)
    return "workstream" in text and any(word in text for word in ("active", "current", "inspect", "state", "status"))


def _active_workstream_exact_continuation(message: str) -> bool:
    text = _active_workstream_normalize_text(message).replace("what's", "whats")
    text = re.sub(r"[^a-z0-9_:-]+", " ", text).strip()
    return text in _ACTIVE_WORKSTREAM_EXACT_CONTINUATIONS


def _active_workstream_truncate(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _format_active_workstream_block(active_workstream: dict[str, Any]) -> str:
    workstream = active_workstream.get("workstream") or {}
    metadata = workstream.get("metadata") or {}
    blockers = metadata.get("blockers") or []
    blocker_text = ", ".join(str(blocker) for blocker in blockers if str(blocker).strip()) or "None"
    lines = [
        "Active Workstream Context (context only; does not override current instructions)",
        f"Title: {_active_workstream_truncate(workstream.get('title'), 160)}",
        f"Objective: {_active_workstream_truncate(metadata.get('current_objective'), 260)}",
        f"Summary: {_active_workstream_truncate(metadata.get('current_summary'), 260)}",
        f"Next: {_active_workstream_truncate(metadata.get('next_action'), 220)}",
        f"Blockers: {_active_workstream_truncate(blocker_text, 180)}",
        (
            f"Binding: {active_workstream.get('binding_source', 'unknown')} "
            f"thread={active_workstream.get('thread_id') or workstream.get('thread_id') or 'unknown'}"
        ),
    ]
    block = "\n".join(line for line in lines if not line.endswith(": "))
    return _active_workstream_truncate(block, ACTIVE_WORKSTREAM_RENDER_MAX_CHARS)


def _active_workstream_render_decision(result: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any] | None:
    active_workstream = result.get("active_workstream")
    if not isinstance(active_workstream, dict):
        return None

    reason = "no_topic_overlap"
    rendered_block = None
    status = active_workstream.get("status")
    binding_source = active_workstream.get("binding_source")
    workstream = active_workstream.get("workstream") or {}
    args = payload or {}

    if status != "ok":
        reason = "not_ok_status"
    elif not binding_source or binding_source == "none":
        reason = "no_explicit_binding"
    elif active_workstream.get("maintenance_filtered") or workstream.get("class") == "maintenance_cluster":
        reason = "maintenance_filtered"
    else:
        message = str(args.get("message") or "")
        has_overlap = _active_workstream_has_topic_overlap(message, active_workstream)
        broad_implicit_binding = binding_source in {"origin_id", "session_id"}
        task_identity_missing = not bool(str(args.get("current_task_id") or "").strip())
        if broad_implicit_binding and task_identity_missing:
            reason = "task_identity_required"
        elif _active_workstream_explicit_inspection(message):
            reason = "explicit_workstream_inspection"
            rendered_block = _format_active_workstream_block(active_workstream)
        elif _active_workstream_exact_continuation(message):
            reason = "exact_continuation"
            rendered_block = _format_active_workstream_block(active_workstream)
        elif args.get("compaction_detected") and has_overlap:
            reason = "compaction_topic_overlap"
            rendered_block = _format_active_workstream_block(active_workstream)
        elif _active_workstream_has_trigger(message) and has_overlap:
            reason = "topic_overlap"
            rendered_block = _format_active_workstream_block(active_workstream)

    return {
        "decision": "render" if rendered_block else "suppress",
        "reason": reason,
        "rendered_block": rendered_block,
        "rendered_chars": len(rendered_block or ""),
        "max_chars": ACTIVE_WORKSTREAM_RENDER_MAX_CHARS,
    }


def _apply_active_workstream_render_decision(result: Any, payload: dict[str, Any] | None) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    decision = _active_workstream_render_decision(result, payload)
    if decision is not None:
        result["active_workstream_render"] = decision


def _workstream_render_failure_reason(result: dict[str, Any]) -> str:
    code = str(result.get("code") or "")
    status_code = result.get("status_code")
    body = result.get("body")
    message = str(result.get("message") or "")
    haystack = f"{code} {status_code} {message} {body}".lower()

    if "invalid_session_id" in haystack:
        return "invalid_session_id"
    if code == "CONNECTION_FAILED":
        return "connection_failed"
    if code == "NON_JSON_RESPONSE":
        return "non_json_response"
    if status_code == 404:
        return "api_404"
    if status_code:
        return f"api_status_{status_code}"
    return "wrapper_non_json_or_error"


def _emit_active_workstream_render_event(
    operation: str,
    payload: dict[str, Any] | None,
    result: Any,
    *,
    transport_mode: str,
    elapsed_ms: float,
    api_status: Any = None,
    api_url: str | None = None,
) -> None:
    if operation != "conversation_turn":
        return

    args = payload or {}
    active_workstream = result.get("active_workstream") if isinstance(result, dict) else None
    render = result.get("active_workstream_render") if isinstance(result, dict) else None
    decision = render.get("decision") if isinstance(render, dict) else "none"
    reason = render.get("reason") if isinstance(render, dict) else "no_active_workstream_render"
    rendered_chars = render.get("rendered_chars") if isinstance(render, dict) else 0
    rendered_block = render.get("rendered_block") if isinstance(render, dict) else None
    error = bool(isinstance(result, dict) and result.get("error") is True)

    if error:
        decision = "none"
        reason = _workstream_render_failure_reason(result)
        api_status = result.get("status_code") or result.get("code") or api_status

    workstream = active_workstream.get("workstream") if isinstance(active_workstream, dict) else None
    _transport_event(
        "active_workstream_render",
        operation=operation,
        transport_mode=transport_mode,
        active_workstream_present=isinstance(active_workstream, dict),
        active_workstream_id=(
            active_workstream.get("thread_id") or (workstream or {}).get("thread_id")
            if isinstance(active_workstream, dict)
            else None
        ),
        decision=decision,
        reason=reason,
        content_blocks=1,
        rendered_chars=int(rendered_chars or 0),
        elapsed_ms=round(elapsed_ms, 2),
        api_status=api_status,
        error=error,
        message_preview=_active_workstream_truncate(args.get("message"), 180),
        rendered_preview=_active_workstream_truncate(rendered_block, 180) if rendered_block else None,
        api_url=api_url,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Direct Pith HTTP API client for exec-capable hosts")
    parser.add_argument("operation")
    parser.add_argument("--stdin-json", action="store_true")
    parser.add_argument("--json-file")
    parser.add_argument(
        "--transport-mode",
        choices=["first_class_api", "exec_http_fallback"],
        default="exec_http_fallback",
        help="Transport label for request headers and diagnostics.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PITH_API_URL") or os.environ.get("BRAIN_API_URL") or DEFAULT_BASE_URL,
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument(
        "--example",
        action="store_true",
        help="Print local example JSON for the operation and exit.",
    )
    discovery_group.add_argument(
        "--schema",
        action="store_true",
        help="Print local schema metadata for the operation and exit.",
    )
    args = parser.parse_args(argv)

    _validate_operation(parser, args.operation)
    if args.example or args.schema:
        kind = "example" if args.example else "schema"
        print(json.dumps(_operation_discovery_payload(args.operation, kind), sort_keys=True))
        return 0
    if args.operation == "list":
        print(json.dumps({"operations": _operation_catalog()}))
        return 0

    payload = _load_payload(args)
    if args.operation == "lifecycle_status":
        result = _build_lifecycle_status(
            payload,
            base_url=args.base_url,
            timeout=args.timeout,
            transport_mode=args.transport_mode,
        )
        print(json.dumps(result, sort_keys=True))
        return 1 if result.get("status") == "error" else 0
    if args.operation == "lifecycle_diagnostic":
        result = _build_lifecycle_diagnostic(
            payload,
            base_url=args.base_url,
            timeout=args.timeout,
            transport_mode=args.transport_mode,
        )
        print(json.dumps(result, sort_keys=True))
        return 1 if result.get("status") == "error" else 0

    method, endpoint = ALLOWED[args.operation]
    payload = _with_default_surface_payload(args.operation, payload)
    if args.operation == "workstreams" and isinstance(payload, dict) and "action" not in payload:
        legacy_action = payload.get("operation")
        if legacy_action:
            payload = {**payload, "action": legacy_action}
    if args.operation == "surface_activity":
        payload = _normalize_surface_activity_payload(payload)
    headers = _build_headers(args.operation, args.transport_mode)

    started = time.perf_counter()
    max_attempts = max(1, CONVERSATION_TURN_STARTUP_MAX_ATTEMPTS) if args.operation == "conversation_turn" else 1
    response = None
    body: Any = None
    for attempt in range(max_attempts):
        try:
            if method == "GET":
                response = requests.get(
                    f"{args.base_url}{endpoint}",
                    params=payload or None,
                    headers=headers,
                    timeout=args.timeout,
                )
            else:
                response = requests.post(
                    f"{args.base_url}{endpoint}",
                    json=payload or {},
                    headers=headers,
                    timeout=args.timeout,
                )
        except requests.RequestException as exc:
            body = {
                "error": True,
                "code": "CONNECTION_FAILED",
                "message": str(exc),
                "operation": args.operation,
                "transport_mode": args.transport_mode,
            }
            _emit_active_workstream_render_event(
                args.operation,
                payload,
                body,
                transport_mode=args.transport_mode,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                api_status="connection_failed",
                api_url=args.base_url,
            )
            _emit_workstream_activation_api_event(
                args.operation,
                payload,
                body,
                transport_mode=args.transport_mode,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                api_status="connection_failed",
                api_url=args.base_url,
            )
            _emit_lifecycle_api_call_event(
                args.operation,
                payload,
                body,
                transport_mode=args.transport_mode,
                api_status="connection_failed",
                api_url=args.base_url,
            )
            print(json.dumps(body))
            return 2

        try:
            body = response.json()
        except ValueError:
            body = {
                "error": True,
                "code": "NON_JSON_RESPONSE",
                "message": response.text[:500],
            }

        if attempt < max_attempts - 1 and _is_retryable_conversation_turn_startup_503(
            args.operation,
            response.status_code,
            body,
        ):
            delay_s = _retry_after_seconds(response, attempt)
            _transport_event(
                "conversation_turn_startup_retry",
                operation=args.operation,
                transport_mode=args.transport_mode,
                status_code=response.status_code,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                retry_after_s=delay_s,
                api_url=args.base_url,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            time.sleep(delay_s)
            continue
        break

    assert response is not None

    if response.status_code >= 400:
        error_body = {
            "error": True,
            "status_code": response.status_code,
            "body": body,
            "operation": args.operation,
            "transport_mode": args.transport_mode,
        }
        _emit_active_workstream_render_event(
            args.operation,
            payload,
            error_body,
            transport_mode=args.transport_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_status=response.status_code,
            api_url=args.base_url,
        )
        _emit_workstream_activation_api_event(
            args.operation,
            payload,
            error_body,
            transport_mode=args.transport_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_status=response.status_code,
            api_url=args.base_url,
        )
        _emit_lifecycle_api_call_event(
            args.operation,
            payload,
            error_body,
            transport_mode=args.transport_mode,
            api_status=response.status_code,
            api_url=args.base_url,
        )
        print(json.dumps(error_body))
        return 1

    if args.operation == "conversation_turn":
        _apply_active_workstream_render_decision(body, payload)
        activation_gate = _resolve_workstream_activation_gate(
            args.operation,
            payload,
            body,
            base_url=args.base_url,
            headers=headers,
            timeout=args.timeout,
            transport_mode=args.transport_mode,
            started=started,
        )
        if isinstance(activation_gate, dict) and activation_gate.get("status") == "blocked":
            body["workstream_activation_gate"] = activation_gate
        _emit_active_workstream_render_event(
            args.operation,
            payload,
            body,
            transport_mode=args.transport_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_status=response.status_code,
            api_url=args.base_url,
        )
        _emit_workstream_activation_hint_event(
            args.operation,
            payload,
            body,
            transport_mode=args.transport_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_status=response.status_code,
            api_url=args.base_url,
        )
        _emit_workstream_activation_gate_event(
            args.operation,
            payload,
            activation_gate,
            transport_mode=args.transport_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_status=response.status_code,
            api_url=args.base_url,
        )
        if isinstance(activation_gate, dict) and activation_gate.get("status") == "blocked":
            print(json.dumps(body))
            return WORKSTREAM_ACTIVATION_GATE_EXIT_CODE
    elif args.operation == "workstreams":
        _emit_workstream_activation_api_event(
            args.operation,
            payload,
            body,
            transport_mode=args.transport_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_status=response.status_code,
            api_url=args.base_url,
        )

    _emit_lifecycle_api_call_event(
        args.operation,
        payload,
        body,
        transport_mode=args.transport_mode,
        api_status=response.status_code,
        api_url=args.base_url,
    )
    print(json.dumps(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
