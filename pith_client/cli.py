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
CONVERSATION_TURN_STARTUP_MAX_ATTEMPTS = int(
    os.environ.get("PITH_CLI_CONVERSATION_TURN_STARTUP_MAX_ATTEMPTS", "4")
)
CONVERSATION_TURN_STARTUP_RETRY_CAP_S = float(
    os.environ.get("PITH_CLI_CONVERSATION_TURN_STARTUP_RETRY_CAP_S", "10")
)
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
PSEUDO_OPERATIONS = frozenset({"lifecycle_status", "list"})
LIFECYCLE_STATUS_SCHEMA_VERSION = "surface_lifecycle_status.v1"
LIFECYCLE_STATUS_DEFAULT_MAX_SCAN_FILES = int(
    os.environ.get("PITH_LIFECYCLE_STATUS_MAX_SCAN_FILES", "500")
)
LIFECYCLE_STATUS_MAX_SCAN_FILES_LIMIT = 5000
LIFECYCLE_STATUS_DEFAULT_MAX_AGE_SECONDS = int(
    os.environ.get("PITH_LIFECYCLE_STATUS_MAX_AGE_SECONDS", str(24 * 60 * 60))
)
LIFECYCLE_STATUS_MAX_AGE_SECONDS_LIMIT = 7 * 24 * 60 * 60
CLAUDE_CODE_LIFECYCLE_STATE_DIR = (
    Path.home() / ".pith" / "cache" / "claude-code-lifecycle"
)
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
    return cleaned if cleaned in SURFACE_ID_VALUES else ""


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
        for item in (str(surface).strip() for surface in requested_surfaces)
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


def _state_string_values(state: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for key in keys:
        value = state.get(key)
        if value is not None and str(value).strip():
            values.add(str(value).strip())
    return values


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
    return not (workspace_id and workspace_id not in _state_string_values(
        state,
        ("pre_response_ct_workspace_id",),
    ))


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


def _claude_code_context_phase(state: dict[str, Any]) -> dict[str, Any]:
    pre_status = state.get("pre_response_ct_status")
    session_id = state.get("pre_response_ct_session_id") or state.get("pith_session_id")
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
    if session_id:
        return _lifecycle_phase(
            "passed",
            verdict="observed",
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


def _overall_lifecycle_verdict(
    context_phase: dict[str, Any],
    learning_phase: dict[str, Any],
) -> str:
    if context_phase.get("status") == "passed" and learning_phase.get("status") == "passed":
        return "enforced"
    if context_phase.get("status") == "passed":
        return "partial"
    if context_phase.get("status") == "not_observed" and learning_phase.get("status") == "not_observed":
        return "not_observed"
    return "failed"


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
    result.update(
        {
            "selected_state_file": selected_path.name,
            "hook_turn_seq": state.get("hook_turn_seq"),
            "context_phase": context_phase,
            "model_visible_phase": model_visible_phase,
            "coherence_phase": coherence_phase,
            "learning_phase": learning_phase,
            "overall_verdict": _overall_lifecycle_verdict(context_phase, learning_phase),
        }
    )
    return result


def _build_lifecycle_status(payload: dict | None) -> dict[str, Any]:
    body = dict(payload or {})
    surface_id = _normalize_surface_id(body.get("surface_id")) or "local_api_cli"
    if surface_id != "claude_code":
        return _lifecycle_unsupported_status(body, surface_id)
    return _claude_code_lifecycle_status(body)


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
        }
        for operation, (method, path) in sorted(ALLOWED.items())
    ]
    operations.extend(
        {
            "operation": operation,
            "method": "LOCAL",
            "path": "",
            "auth_required": False,
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
    return (
        "retrieval initialization" in detail
        or "retrieval recovery" in detail
        or "server startup" in detail
    )


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
        recommended_count=(
            _list_len(response_body.get("recommended")) if isinstance(response_body, dict) else 0
        ),
        advisory_candidate_count=(
            _list_len(response_body.get("advisory_candidates")) if isinstance(response_body, dict) else 0
        ),
        possible_match_count=(
            _list_len(response_body.get("possible_matches")) if isinstance(response_body, dict) else 0
        ),
        proof_or_maintenance_count=(
            _list_len(response_body.get("proof_or_maintenance"))
            if isinstance(response_body, dict)
            else 0
        ),
        needs_review_count=(
            _list_len(response_body.get("needs_review")) if isinstance(response_body, dict) else 0
        ),
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
            reason = (
                "active_workstream_unrelated"
                if active_binding_related is False
                else "active_binding_unknown"
            )
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
            body.get("active_workstream_render")
            if isinstance(body.get("active_workstream_render"), dict)
            else None
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
        if _active_workstream_explicit_inspection(message):
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
            active_workstream.get("thread_id")
            or (workstream or {}).get("thread_id")
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
    parser = argparse.ArgumentParser(
        description="Direct Pith HTTP API client for exec-capable hosts"
    )
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
        default=os.environ.get("PITH_API_URL")
        or os.environ.get("BRAIN_API_URL")
        or DEFAULT_BASE_URL,
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    _validate_operation(parser, args.operation)
    if args.operation == "list":
        print(json.dumps({"operations": _operation_catalog()}))
        return 0

    payload = _load_payload(args)
    if args.operation == "lifecycle_status":
        result = _build_lifecycle_status(payload)
        print(json.dumps(result, sort_keys=True))
        return 1 if result.get("status") == "error" else 0

    method, endpoint = ALLOWED[args.operation]
    payload = _with_default_surface_payload(args.operation, payload)
    if args.operation == "surface_activity":
        payload = _normalize_surface_activity_payload(payload)
    headers = _build_headers(args.operation, args.transport_mode)

    started = time.perf_counter()
    max_attempts = (
        max(1, CONVERSATION_TURN_STARTUP_MAX_ATTEMPTS)
        if args.operation == "conversation_turn"
        else 1
    )
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

        if (
            attempt < max_attempts - 1
            and _is_retryable_conversation_turn_startup_503(
                args.operation,
                response.status_code,
                body,
            )
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

    print(json.dumps(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
