#!/usr/bin/env python3
"""
Pith MCP Server — Python implementation

1:1 port of server.js. Wraps the Pith REST API as an MCP server
for integration with Claude Desktop via stdio transport.

Eliminates Node.js dependency entirely. This is the critical path
for Pith Installer Path B.

Client-side state preserved from server.js:
  C1: Auto-session (conversation boundary detection)
  L3: Learning debt tracking (protocol enforcement)
  L4: Cognitive bootstrap (orientation injection)
  C4: Descriptive instructions (dynamic from Pith stats)
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _bootstrap_mcp_imports():
    """Import MCP SDK, surfacing any failure with structured diagnostic output.

    MCP-PYTHON-RES-001 v1.3. Catches a BROAD set of import-time failures:
      - ModuleNotFoundError: package missing entirely
      - ImportError: partial package, broken namespace
      - AttributeError: version mismatch (e.g. mcp 0.x where 1.x expected)
      - TypeError / RuntimeError: SDK init-side failures that escape import

    On ANY such failure: writes a self-diagnosis file AND prints a structured
    JSON error to stderr. Claude Desktop's MCP host captures stderr in
    ~/Library/Logs/Claude/mcp-server-pith.log AND surfaces the first stderr
    line in its own connection-error message.
    """
    try:
        import httpx as _httpx
        from mcp.server import Server as _Server
        from mcp.server.stdio import stdio_server as _stdio_server
        from mcp.types import TextContent as _TextContent
        from mcp.types import Tool as _Tool

        return _httpx, _Server, _stdio_server, _TextContent, _Tool
    except Exception as exc:
        exc_type = type(exc).__name__
        if isinstance(exc, ModuleNotFoundError):
            failure_class = "missing_package"
            missing_module = exc.name or "unknown"
        elif isinstance(exc, ImportError):
            failure_class = "partial_import"
            missing_module = getattr(exc, "name", None) or "unknown"
        elif isinstance(exc, AttributeError):
            failure_class = "version_mismatch"
            missing_module = "mcp"
        else:
            failure_class = "other_bootstrap_failure"
            missing_module = None
        diag = {
            "error": "pith_mcp_bootstrap_failed",
            "failure_class": failure_class,
            "exception_type": exc_type,
            "exception_message": str(exc)[:500],
            "missing_module": missing_module,
            "interpreter": sys.executable,
            "interpreter_version": sys.version.split()[0],
            "remediation": (
                "Configured interpreter is missing or has an incompatible mcp "
                "package. Re-run scripts/install.sh or update "
                "claude_desktop_config.json to point at an interpreter with "
                "`mcp>=1.0.0,<2.0.0` installed (e.g. ~/.pith/venv/bin/python3)."
            ),
            "doctor_command": "bash ~/.pith/pith-server/scripts/pith_mcp_doctor.sh --auto-repair",
        }
        diag_path = Path.home() / ".pith" / "diagnostics" / "mcp_bridge_failure.json"
        try:
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(json.dumps(diag, indent=2))
        except OSError:
            pass
        print(f"PITH_MCP_BOOTSTRAP_ERROR {json.dumps(diag)}", file=sys.stderr, flush=True)
        sys.stderr.flush()
        os._exit(70)  # EX_SOFTWARE; os._exit avoids finalizers with mcp half-imported


httpx, Server, stdio_server, TextContent, Tool = _bootstrap_mcp_imports()

# Runtime guard import is non-critical — wrap separately
try:
    from app.governance.runtime_install_guard import RuntimeInstallGuardError, ensure_safe_installed_runtime
except ImportError:

    class RuntimeInstallGuardError(RuntimeError):
        pass

    def ensure_safe_installed_runtime(*a, **kw):
        return None


# --- Configuration ---
PITH_API_URL = (
    os.getenv("PITH_API_URL")
    if os.getenv("PITH_API_URL") is not None
    else os.getenv("BRAIN_API_URL", "http://localhost:8000")
)


def _resolve_api_key() -> str:
    """OPS-163: Resolve API key with key-from-file fallback.

    Priority: PITH_API_KEY env > BRAIN_API_KEY env > ~/.pith/.env file.
    Eliminates the N-client key sync problem — clients can omit the key
    from their config and the wrapper reads it from the canonical source.
    """
    key = os.getenv("PITH_API_KEY") or os.getenv("BRAIN_API_KEY", "")
    if key:
        return key
    env_file = os.path.expanduser("~/.pith/.env")
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("PITH_API_KEY=") and not line.startswith("#"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        print(f"OPS-163: API key loaded from {env_file} (env var was empty)", file=sys.stderr)
                        return key
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"OPS-163: Failed to read {env_file}: {e}", file=sys.stderr)
    return ""


PITH_API_KEY = _resolve_api_key()
EXEC_FALLBACK_ENABLED = os.getenv("PITH_EXEC_FALLBACK_ENABLED", "1") == "1"
EXEC_FALLBACK_COMMAND = os.getenv(
    "PITH_EXEC_FALLBACK_COMMAND",
    f"{Path.home() / '.pith' / 'bin' / 'pith'} api-fallback",
)

# --- Deprecation warnings for legacy env vars ---
if os.getenv("BRAIN_API_URL") and not os.getenv("PITH_API_URL"):
    print("DEPRECATED: BRAIN_API_URL env var. Rename to PITH_API_URL.", file=sys.stderr)
if os.getenv("BRAIN_API_KEY") and not os.getenv("PITH_API_KEY"):
    print("DEPRECATED: BRAIN_API_KEY env var. Rename to PITH_API_KEY.", file=sys.stderr)

# --- Logging (stderr only — stdout is MCP transport) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("pith_mcp")

BRIDGE_NAME = "python_mcp"
BRIDGE_PROFILE = os.getenv("PITH_PROFILE") or os.getenv("BRAIN_PROFILE") or "default"
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


def _normalize_bridge_surface_id(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    return cleaned if cleaned in SURFACE_ID_VALUES else ""


BRIDGE_SURFACE_ID = _normalize_bridge_surface_id(os.getenv("PITH_SURFACE_ID"))
BRIDGE_PLATFORM_HINT = os.getenv("PITH_PLATFORM_HINT", "").strip()
MCP_CONVERSATION_TURN_COMPACT_MAX_CHARS = 12000
MCP_CONVERSATION_TURN_FULL_MAX_CHARS = 50000

# RUNG0 Component C (A8): per-origin authorship trust-tier. The Rung-0 loop's launchd
# unit sets PITH_PROVENANCE=agent_loop on ITS bridge process only; the human's bridge
# leaves it unset → 'human'. This is a transport-level marker, deliberately NOT a
# conversation_turn tool argument, so the in-turn model cannot read or forge it.
# Must match app.core.constants.PROVENANCE_VALUES (intentional duplication: the thin
# bridge must not import app internals — keep the two sets in sync).
_VALID_PROVENANCE = {"human", "agent_loop"}
_raw_provenance = os.getenv("PITH_PROVENANCE", "").strip()
if _raw_provenance and _raw_provenance not in _VALID_PROVENANCE:
    print(
        f"PITH_PROVENANCE='{_raw_provenance}' is not a recognized trust-tier "
        f"{sorted(_VALID_PROVENANCE)} — ignoring (concepts will be tagged 'human'). "
        f"This silently disables the Rung-0 authority cap; fix the env value.",
        file=sys.stderr,
        flush=True,
    )
    _raw_provenance = ""
BRIDGE_PROVENANCE = _raw_provenance or "human"
BRIDGE_OUTBOX_DIR = Path.home() / ".pith" / "state" / "bridge_outbox" / BRIDGE_NAME
_bridge_outbox_drain_lock: asyncio.Lock | None = None


def _make_request_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000):x}_{os.urandom(4).hex()}"


def _is_durable_transport_error(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict) or not result.get("error"):
        return False
    return result.get("code") in {
        "CONNECTION_REFUSED",
        "CONNECTION_RESET",
        "SERVER_RESTARTED",
        "RETRY_EXHAUSTED",
        "TIMEOUT",
        "WRITE_NOT_READY",
    }


def _ensure_bridge_outbox_dir() -> Path:
    BRIDGE_OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    return BRIDGE_OUTBOX_DIR


def _queue_outbox_path(request_id: str) -> Path:
    return _ensure_bridge_outbox_dir() / f"{request_id}.json"


def _load_outbox_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_outbox_record(record: dict[str, Any]) -> None:
    path = _queue_outbox_path(record["request_id"])
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _queue_durable_write(
    endpoint: str,
    method: str,
    body: dict[str, Any],
    request_id: str,
    error: dict[str, Any],
) -> dict[str, Any]:
    path = _queue_outbox_path(request_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    retry_count = 0
    if path.exists():
        existing = _load_outbox_record(path)
        retry_count = int(existing.get("retry_count", 0))
    record = {
        "request_id": request_id,
        "endpoint": endpoint,
        "method": method,
        "profile": BRIDGE_PROFILE,
        "body": body,
        "created_at": now,
        "retry_count": retry_count + 1,
        "last_error": error,
    }
    _write_outbox_record(record)
    return {
        "status": "queued",
        "persistence_state": "queued",
        "request_id": request_id,
        "endpoint": endpoint,
        "queued_at": now,
        "retry_count": record["retry_count"],
    }


async def _get_bridge_outbox_lock() -> asyncio.Lock:
    global _bridge_outbox_drain_lock
    if _bridge_outbox_drain_lock is None:
        _bridge_outbox_drain_lock = asyncio.Lock()
    return _bridge_outbox_drain_lock


async def _drain_bridge_outbox() -> None:
    lock = await _get_bridge_outbox_lock()
    if lock.locked():
        return

    async with lock:
        outbox_dir = _ensure_bridge_outbox_dir()
        for path in sorted(outbox_dir.glob("*.json")):
            try:
                record = _load_outbox_record(path)
                result = await call_pith_api(
                    record["endpoint"],
                    record.get("method", "POST"),
                    record.get("body"),
                    drain_outbox=False,
                )
                if result and not result.get("error"):
                    path.unlink(missing_ok=True)
                    continue

                if _is_durable_transport_error(result):
                    record["retry_count"] = int(record.get("retry_count", 0)) + 1
                    record["last_error"] = result
                    _write_outbox_record(record)
                    break

                logger.warning(
                    "Bridge outbox dropping unreplayable request %s for %s: %s",
                    record.get("request_id"),
                    record.get("endpoint"),
                    result,
                )
                path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(f"Bridge outbox replay failed for {path.name}: {exc}")
                break


def _schedule_bridge_outbox_drain() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_drain_bridge_outbox())


# --- BRIDGE-002: Persistent transport log ---
# Writes structured JSONL events to disk so diagnostics survive bridge death.
# State file tracks the latest event for quick status checks.
TRANSPORT_LOG_PATH = os.path.expanduser("~/.pith/logs/pith_mcp_transport.jsonl")
TRANSPORT_STATE_PATH = os.path.expanduser("~/.pith/logs/pith_mcp_transport_state.json")


def _transport_iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_bridge_health(*, updated_at: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "transport_state": "unknown",
        "http_state": "unknown",
        "auth_state": "unknown",
        "overlap_detected": False,
        "last_overlap": None,
        "last_error_code": None,
        "last_error_endpoint": None,
        "last_readyz": None,
        "last_shutdown_reason": None,
        "last_session_end_result": "none",
        "updated_at": updated_at,
    }


def _load_transport_state(*, default_started_at: str | None = None) -> dict[str, Any]:
    now_iso = default_started_at or _transport_iso_now()
    try:
        with open(TRANSPORT_STATE_PATH) as f:
            state = json.load(f)
            if not isinstance(state, dict):
                state = {}
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}

    merged = {
        "pid": state.get("pid", os.getpid()),
        "ppid": state.get("ppid", os.getppid()),
        "api_url": state.get("api_url", PITH_API_URL),
        "started_at": state.get("started_at", now_iso),
        "log_path": state.get("log_path", TRANSPORT_LOG_PATH),
        "state_path": state.get("state_path", TRANSPORT_STATE_PATH),
        "event_count": state.get("event_count", 0),
        **state,
    }
    existing_bridge_health = state.get("bridge_health")
    if not isinstance(existing_bridge_health, dict):
        existing_bridge_health = {}
    merged["bridge_health"] = {
        **_default_bridge_health(
            updated_at=existing_bridge_health.get("updated_at", now_iso)
        ),
        **existing_bridge_health,
    }
    merged["bridge_health"]["schema_version"] = 2
    return merged


def _persist_transport_state(state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(TRANSPORT_STATE_PATH), exist_ok=True)
    with open(TRANSPORT_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _update_bridge_health(**fields) -> None:
    """Persist the latest normalized bridge-health snapshot."""
    try:
        state = _load_transport_state()
        bridge_health = state["bridge_health"]
        bridge_health.update(fields)
        bridge_health["schema_version"] = 2
        bridge_health["updated_at"] = _transport_iso_now()
        _persist_transport_state(state)
    except OSError:
        pass


def _shutdown_profile(reason: str) -> dict[str, Any]:
    state = _load_transport_state()
    http_state = state.get("bridge_health", {}).get("http_state", "unknown")
    has_session = bool(_state.get("cached_session_id"))
    attempt_allowed = reason in {"idle_timeout", "max_age", "stdio_teardown"}
    if not has_session:
        return {"attempt_session_end": False, "default_result": "skipped_no_session"}
    if attempt_allowed or http_state == "reachable":
        return {"attempt_session_end": True, "default_result": "attempted_failed"}
    return {
        "attempt_session_end": False,
        "default_result": "skipped_backend_unhealthy",
    }


def _force_exit(code: int = 0) -> None:
    os._exit(code)


def _readyz_subset(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": payload.get("mode"),
        "write_state": payload.get("write_state"),
        "retrieval_state": payload.get("retrieval_state"),
    }


def _transport_event(event: str, **kwargs) -> None:
    """Append a structured event to the transport log and update state file."""
    entry = {
        "ts": _transport_iso_now(),
        "event": event,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "api_url": PITH_API_URL,
        **kwargs,
    }
    try:
        os.makedirs(os.path.dirname(TRANSPORT_LOG_PATH), exist_ok=True)
        with open(TRANSPORT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass
    # Update state file (latest snapshot)
    try:
        state = _load_transport_state(default_started_at=entry["ts"])
        state["last_event"] = entry
        state["event_count"] = state.get("event_count", 0) + 1
        if event == "tool_call" and kwargs.get("phase") == "start":
            state["last_tool_name"] = kwargs.get("tool_name")
            state["last_tool_started_at"] = entry["ts"]
        elif event == "tool_call" and kwargs.get("phase") in ("success", "error"):
            state["last_tool_completed_at"] = entry["ts"]
        if "cached_session_id" in kwargs:
            state["cached_session_id"] = kwargs["cached_session_id"]
        _persist_transport_state(state)
    except OSError:
        pass


def _workstream_render_failure_reason(result: dict[str, Any]) -> str:
    code = str(result.get("code") or "")
    status_code = result.get("status_code")
    details = str(result.get("details") or "")
    message = str(result.get("message") or "")
    haystack = f"{code} {status_code} {message} {details}".lower()

    if code == "REPO_HYGIENE_POLICY_BLOCK":
        return "repo_hygiene_policy_block"
    if "invalid_session_id" in haystack:
        return "invalid_session_id"
    if status_code == 503 or "http 503" in haystack or "not_ready" in haystack:
        return "api_503_not_ready"
    if status_code == 404 or code == "NOT_FOUND":
        return "api_404_error_payload"
    if "database deadlock" in haystack and "session_start" in haystack:
        return "db_deadlock_session_start"
    if "timeout" in haystack or "timed out" in haystack:
        return "timeout_startup_or_retrieval"
    return "wrapper_non_json_or_error"


def _emit_active_workstream_render_event(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    *,
    content_blocks: int,
    elapsed_ms: float,
    api_status: Any = None,
) -> None:
    if tool_name != "pith_conversation_turn":
        return

    active_workstream = result.get("active_workstream") if isinstance(result, dict) else None
    render = result.get("active_workstream_render") if isinstance(result, dict) else None
    decision = render.get("decision") if isinstance(render, dict) else "none"
    reason = render.get("reason") if isinstance(render, dict) else "no_active_workstream_render"
    rendered_chars = render.get("rendered_chars") if isinstance(render, dict) else 0

    if isinstance(result, dict) and result.get("error") is True:
        decision = "none"
        reason = _workstream_render_failure_reason(result)
        api_status = result.get("status_code") or result.get("code") or api_status

    try:
        origin_id_present = bool(_infer_origin_id(args))
    except ValueError:
        origin_id_present = False

    _transport_event(
        "active_workstream_render",
        origin_id_present=origin_id_present,
        active_workstream_present=isinstance(active_workstream, dict),
        decision=decision,
        reason=reason,
        content_blocks=content_blocks,
        rendered_chars=int(rendered_chars or 0),
        elapsed_ms=round(elapsed_ms, 2),
        api_status=api_status,
    )


def _validate_startup_auth() -> None:
    """Fail fast on missing or rejected MCP auth so clients don't boot half-broken."""
    if not PITH_API_KEY:
        raise RuntimeError("PITH_API_KEY is missing. Refusing to start MCP bridge with unauthenticated write tools.")

    try:
        health_resp = httpx.get(f"{PITH_API_URL}/health", timeout=5.0)
    except Exception as exc:
        logger.warning(f"Startup: Could not reach Pith server for validation: {exc}")
        return

    if not health_resp.is_success:
        logger.warning(f"Startup: Pith server unhealthy ({health_resp.status_code})")
        return

    _update_bridge_health(http_state="reachable")
    logger.info("Startup: Pith server reachable, health OK")
    try:
        auth_resp = httpx.get(
            f"{PITH_API_URL}/pith_health?detail=fast",
            headers={"X-API-Key": PITH_API_KEY},
            timeout=5.0,
        )
    except Exception as exc:
        # BRIDGE-004: Startup auth probe is best-effort. The first httpx.get
        # above (for /health) is already wrapped and returns on any failure,
        # but the /pith_health probe was unwrapped — a ReadTimeout or
        # connection error here would propagate out of asyncio.run(main())
        # and kill the bridge before the MCP handshake could complete.
        # Per-request failures will surface with a proper 401/403 once the
        # tool actually runs; don't let a flaky startup probe take down the
        # whole subprocess.
        logger.warning(f"Startup: API key validation skipped ({exc})")
        return
    if auth_resp.status_code in (401, 403):
        _update_bridge_health(auth_state="rejected")
        raise RuntimeError(
            "MCP API key rejected by server. Check that PITH_API_KEY in the client "
            "config matches PITH_API_KEY in ~/.pith/.env."
        )

    _update_bridge_health(auth_state="validated")
    logger.info("Startup: API key validated OK")


# --- C4: Static fallback instructions ---
EXEC_FALLBACK_INSTRUCTIONS = ""
if EXEC_FALLBACK_ENABLED:
    EXEC_FALLBACK_INSTRUCTIONS = (
        "\nDEGRADED-MODE EXEC FALLBACK:\n"
        "If pith MCP returns Transport closed or repeated timeouts, and non-Pith tools still work, stop using Pith MCP tools for this turn.\n"
        f"1. Verify backend readiness: {EXEC_FALLBACK_COMMAND} readyz\n"
        f"2. Continue the cognitive loop with stdin JSON, for example: {EXEC_FALLBACK_COMMAND} conversation_turn --stdin-json\n"
        "3. Use the same fallback surface for checkpoint, session_learn, and session_end until MCP succeeds again.\n"
        "4. Switch back only after a real Pith MCP call succeeds.\n"
        "5. Never send the same closeout or checkpoint through both MCP and exec fallback.\n"
    )


STATIC_FALLBACK_INSTRUCTIONS = f"""⚠️ MANDATORY FIRST ACTION — NON-NEGOTIABLE:
Before responding to the user's VERY FIRST message, call pith_conversation_turn(message=user's message).
This retrieves critical context about this user, their projects, and ongoing work.
Skipping this makes you appear amnesiac and wastes the user's time repeating context.
Do this BEFORE AskUserQuestion, BEFORE TodoWrite, BEFORE any other tool. Pith first. Always.

You have persistent memory that learns and remembers across conversations.

COGNITIVE LOOP (simplified — ONE tool does everything):
1. BEFORE responding: call pith_conversation_turn with:
   - message: the user's current message
   - previous_response: your LAST response (REQUIRED after first exchange)
   - previous_message: the user's previous message
   - extracted_concepts_json: 1-7 concepts from your previous response (JSON string)
2. RESPOND using activated context + any extraction_request hints in the response
3. When conversation ends: call pith_session_end with previous_response

That's it. ONE tool call per exchange. The server handles all learning automatically.
pith_session_learn is still available for explicit high-quality extraction when needed,
but the default path is conversation_turn.

CONCEPT FORMAT for extracted_concepts_json:
[{{"summary": "30-500 chars", "confidence": 0.6, "knowledge_area": "domain", "evidence": ["source >=10 chars"], "concept_type": "decision"}}]
ALWAYS set concept_type: observation, pattern, decision, principle, method, heuristic, cognitive_strategy.
If the exchange was casual/trivial, send '[]' (empty array) — do NOT invent filler.
SUMMARY PRECISION — summaries MUST preserve specific details, not abstract them:
Always include: proper nouns, specific numbers/amounts/dates/times, named entities
(restaurants, books, products, people, places, brands, titles, medications).
WRONG: "recommended a light beer for the lamb dish"
RIGHT: "recommended Pilsner or Lager for Seco de Cordero"
WRONG: "user's budget for renovation"  RIGHT: "user's renovation budget is $4,500"
If someone later asks "what was the name/number/time?" — the summary must have the answer.

SESSION LIFECYCLE:
- pith_session_start at conversation beginning (includes orientation + active checkpoint if any)
- pith_session_end when conversation concludes — ALWAYS include previous_response to capture final exchange

EXECUTION CHECKPOINTS (for cross-session resumption):
- pith_checkpoint save: Save what you're working on (task_id, done, active, next). Do this every 15 min or before risky work.
- pith_checkpoint load: Load most recent checkpoint or by task_id. Auto-loaded on session_start.
- Checkpoints are ephemeral (7-day TTL) and separate from knowledge concepts.
{EXEC_FALLBACK_INSTRUCTIONS}

EXTRACTION EXAMPLES — L1 vs L3+ (what to extract from your own responses):
BAD (L1 only): {{summary:'We fixed the validation bug by changing line 222', concept_type:'observation'}}
GOOD (L3): {{summary:'PRINCIPLE: When changing a validation limit, grep the entire codebase for all enforcement points — there is never just one gate', concept_type:'principle', evidence:['verified: second hardcoded check found at line 222']}}
BAD (L1): {{summary:'The budget warning field was missing from the response', concept_type:'observation'}}
GOOD (L3): {{summary:'HEURISTIC: Diagnostic signals created inside internal functions are silent failures unless traced through every calling layer to the end user', concept_type:'heuristic', evidence:['verified: budget_warnings lost between session_learn and conversation_turn']}}
The pattern: L1 captures WHAT happened. L3+ captures the REUSABLE LESSON a future session could apply to a different problem.
GOOD (factual/L1): {{summary:'VACUUM cannot run inside a transaction — pith storage uses isolation_level=None (autocommit) so VACUUM is safe to call from maintenance functions', concept_type:'observation', evidence:['verified: storage_backend.py line 259 isolation_level=None']}}
GOOD (factual/L1): {{summary:'phase5_7_incremental_vacuum always skips when auto_vacuum=0 — freelist pages are NOT reclaimed unless auto_vacuum=2', concept_type:'observation', evidence:['verified: maintenance.py line 1264 checks auto_vacuum_mode != 2']}}
Include factual/L1 concepts when your response contains specific verified facts about system behavior, thresholds, or configuration values.

Pith gets smarter with every conversation. Your job is to feed it quality knowledge."""


# --- Client-side state (C1, L3, L4) ---
SESSION_IDLE_TIMEOUT_S = 2 * 60 * 60  # 2 hours


def _lifecycle_seconds_env(name: str, default: int, *, min_value: int, max_value: int, allow_zero: bool = False) -> int:
    """Read a bounded lifecycle seconds override from the environment."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("LIFECYCLE: Ignoring invalid %s=%r; using %ss", name, raw, default)
        return default
    if allow_zero and value == 0:
        return 0
    if value < min_value:
        logger.warning("LIFECYCLE: Clamping %s=%ss up to %ss", name, value, min_value)
        return min_value
    if value > max_value:
        logger.warning("LIFECYCLE: Clamping %s=%ss down to %ss", name, value, max_value)
        return max_value
    return value


def _lifecycle_bool_env(name: str, default: bool) -> bool:
    """Read a lifecycle boolean override from the environment."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# --- Process lifecycle (separate from session lifecycle) ---
# Session timeout controls the Pith logical session; process timeout controls the OS process.
CONNECTED_IDLE_TIMEOUT_S = _lifecycle_seconds_env(
    "PITH_MCP_CONNECTED_IDLE_TIMEOUT_S",
    0,
    min_value=15 * 60,
    max_value=24 * 60 * 60,
    allow_zero=True,
)  # 0 disables connected idle self-exit; max-age remains bounded.
CONNECTED_MAX_AGE_S = _lifecycle_seconds_env(
    "PITH_MCP_CONNECTED_MAX_AGE_S",
    72 * 60 * 60,
    min_value=60 * 60,
    max_value=7 * 24 * 60 * 60,
)
CONNECTED_MAX_AGE_FORCE_EXIT = _lifecycle_bool_env("PITH_MCP_FORCE_MAX_AGE_EXIT", False)
HOST_SIGNAL_DEFERRAL_ENABLED = _lifecycle_bool_env("PITH_MCP_DEFER_HOST_SIGNALS", True)
REAP_STALE_IDLE_S = _lifecycle_seconds_env(
    "PITH_MCP_REAP_STALE_IDLE_S",
    60 * 60,
    min_value=15 * 60,
    max_value=24 * 60 * 60,
)
LEGACY_REAP_AGE_S = _lifecycle_seconds_env(
    "PITH_MCP_LEGACY_REAP_AGE_S",
    60 * 60,
    min_value=15 * 60,
    max_value=24 * 60 * 60,
)
WATCHDOG_INTERVAL_S = 30  # check frequency
HEARTBEAT_DIR = os.path.expanduser("~/.pith/run")
CONVERSATION_BOUNDARY_S = 2 * 60  # 2 minutes — new conversation detection
LEARNING_DEBT_THRESHOLD = 3

SUBSTANTIVE_TOOLS = frozenset(
    [
        "pith_conversation_turn",
        "pith_search",
        "pith_propose_concept",
        "pith_evolve_concept",
        "pith_get_concept",
        "pith_related_concepts",
        "pith_link_concepts",
    ]
)
LEARNING_TOOLS = frozenset(["pith_session_learn"])
SESSION_TOOLS = frozenset(["pith_session_start"])
META_TOOLS = frozenset(
    [
        "pith_stats",
        "pith_health",
        "pith_bridge_status",
        "pith_projection",
        "pith_orient",
        "pith_sessions_list",
        "pith_questions",
        "pith_session_end",
    ]
)

# Mutable state
_state = {
    "cached_session_id": None,
    "last_session_activity": None,
    "learning_debt": 0,
    "last_learn_timestamp": None,
    "total_calls_since_session_start": 0,
    "auto_session_created": False,
    "pending_bootstrap_orientation": None,
    "last_conv_turn_args": None,
    "is_first_ensure_session": True,
    "session_creation_lock": None,
    "codex_thread_id": None,
    "codex_rollout_path": None,
    "codex_rollout_mtime_ns": None,
    "codex_rollout_telemetry": None,
    "last_tool_call_time": None,
    "bridge_start_time": None,
    "original_ppid": None,
    "shutdown_initiated": False,
    "host_transport_open": False,
    "deferred_signal_counts": {},
    "deferred_signal_total": 0,
    "last_deferred_signal": None,
    "connected_max_age_reported": False,
}

BLOCKING_WORKSPACE_FINDING_CODES = {"DUPLICATE_BRANCH_OWNER", "REGISTRY_NOT_LIVE", "MISSING_PATH"}
BLOCKING_WORKSPACE_CLASSIFICATIONS = {"canonical_checkout", "unregistered_worktree", "archive_only_lane"}
DEFAULT_RUNTIME_ROOT_MARKERS = ("/_release_worktrees/",)


def _repo_hygiene_runtime_root_markers() -> tuple[str, ...]:
    try:
        from app.core.config import REPO_HYGIENE_RUNTIME_ROOT_MARKERS

        markers = REPO_HYGIENE_RUNTIME_ROOT_MARKERS
    except Exception:
        markers = os.environ.get(
            "PITH_REPO_HYGIENE_RUNTIME_ROOT_MARKERS",
            ",".join(DEFAULT_RUNTIME_ROOT_MARKERS),
        ).split(",")
    return tuple(str(marker).strip() for marker in markers if str(marker).strip())


def _workspace_runtime_exception_reason(workspace_context: dict[str, Any]) -> str | None:
    current_path = str(workspace_context.get("current_path") or "")
    for marker in _repo_hygiene_runtime_root_markers():
        if marker and marker in current_path:
            return "runtime_release_worktree"
    return None


def _session_audit_script_path() -> Path:
    override = os.environ.get("PITH_SESSION_AUDIT_SCRIPT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".pith" / "scripts" / "session_isolation_audit.py"


def _collect_workspace_context() -> dict[str, Any] | None:
    """Best-effort local workspace audit for operational policy enforcement."""
    audit_script = _session_audit_script_path()
    if not audit_script.exists():
        return None
    try:
        completed = subprocess.run(
            ["python3", str(audit_script), "--repo", os.getcwd(), "--mode", "warn", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if completed.returncode not in (0, 1):
            return None
        payload = json.loads(completed.stdout or "{}")
        return {
            "repo_root": payload.get("repo_root", ""),
            "current_path": payload.get("current_path", ""),
            "classification": payload.get("classification", "unknown"),
            "usage_policy": payload.get("usage_policy", ""),
            "current_branch": payload.get("current_branch", ""),
            "branch_owner": payload.get("branch_owner", ""),
            "active_worktree_count": payload.get("active_worktree_count", 0),
            "findings": payload.get("findings", []),
        }
    except Exception:
        return None


def _workspace_protocol_violation(workspace_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fail closed when the local workspace violates session-isolation protocol."""
    if not workspace_context:
        return None

    classification = workspace_context.get("classification", "unknown")
    findings = workspace_context.get("findings") or []
    blocking_codes = [
        item.get("code", "") for item in findings if item.get("code", "") in BLOCKING_WORKSPACE_FINDING_CODES
    ]
    violation = classification in BLOCKING_WORKSPACE_CLASSIFICATIONS or bool(blocking_codes)
    if not violation:
        return None
    exception_reason = _workspace_runtime_exception_reason(workspace_context)
    if exception_reason:
        return None

    detail_parts = [f"classification={classification}"]
    current_path = workspace_context.get("current_path")
    current_branch = workspace_context.get("current_branch")
    if current_path:
        detail_parts.append(f"path={current_path}")
    if current_branch:
        detail_parts.append(f"branch={current_branch}")
    if blocking_codes:
        detail_parts.append(f"findings={','.join(blocking_codes)}")

    return {
        "error": True,
        "code": "REPO_HYGIENE_POLICY_BLOCK",
        "message": (
            "Protocol violation: active work must run from a registered session worktree. " + " ".join(detail_parts)
        ),
        "workspace_context": workspace_context,
    }


def _get_codex_home() -> Path:
    """Return the Codex home directory for local rollout discovery."""
    raw = os.getenv("CODEX_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".codex"


def _resolve_codex_rollout_path(thread_id: str) -> Path | None:
    """Resolve the current thread's rollout path from the local Codex store."""
    cached_thread_id = _state.get("codex_thread_id")
    cached_path = _state.get("codex_rollout_path")
    if cached_thread_id == thread_id and cached_path:
        path = Path(cached_path)
        if path.exists():
            return path

    codex_home = _get_codex_home()
    for dirname in ("sessions", "archived_sessions"):
        root = codex_home / dirname
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"rollout-*{thread_id}.jsonl"))
        if matches:
            resolved = matches[-1]
            _state["codex_thread_id"] = thread_id
            _state["codex_rollout_path"] = str(resolved)
            _state["codex_rollout_mtime_ns"] = None
            _state["codex_rollout_telemetry"] = None
            return resolved
    return None


def _build_codex_context_telemetry(
    *,
    thread_id: str,
    rollout_path: Path,
    used_tokens: int,
    window_size_tokens: int,
    event_timestamp: str | None,
) -> dict[str, Any]:
    """Normalize native Codex rollout usage into structured context telemetry."""
    pressure_ratio = round(max(0.0, min(1.0, used_tokens / window_size_tokens)), 6)
    return {
        "schema_version": "1.0",
        "pressure_ratio": pressure_ratio,
        "measurement_source": "native_token_window",
        "measurement_confidence": "high",
        "measurement_scope": "current_window",
        "used_tokens": used_tokens,
        "window_size_tokens": window_size_tokens,
        "source_metadata": {
            "platform": "codex_desktop",
            "telemetry_origin": "codex_rollout_jsonl",
            "thread_id": thread_id,
            "rollout_path": str(rollout_path),
            "event_timestamp": event_timestamp,
        },
    }


def _load_codex_context_telemetry(rollout_path: Path, thread_id: str) -> dict[str, Any] | None:
    """Read the latest merge-eligible token telemetry from a Codex rollout file."""
    try:
        stat = rollout_path.stat()
    except OSError:
        return None

    cached_path = _state.get("codex_rollout_path")
    cached_mtime_ns = _state.get("codex_rollout_mtime_ns")
    cached_telemetry = _state.get("codex_rollout_telemetry")
    if cached_path == str(rollout_path) and cached_mtime_ns == stat.st_mtime_ns and cached_telemetry:
        return dict(cached_telemetry)

    latest_telemetry: dict[str, Any] | None = None
    try:
        with rollout_path.open(encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    envelope = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                payload = envelope.get("payload")
                if not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue

                info = payload.get("info")
                if not isinstance(info, dict):
                    continue

                last_usage = info.get("last_token_usage")
                used_tokens = last_usage.get("total_tokens") if isinstance(last_usage, dict) else None
                window_size_tokens = info.get("model_context_window")
                if (
                    not isinstance(used_tokens, int)
                    or not isinstance(window_size_tokens, int)
                    or window_size_tokens <= 0
                ):
                    continue

                latest_telemetry = _build_codex_context_telemetry(
                    thread_id=thread_id,
                    rollout_path=rollout_path,
                    used_tokens=used_tokens,
                    window_size_tokens=window_size_tokens,
                    event_timestamp=envelope.get("timestamp"),
                )
    except OSError:
        return None

    _state["codex_rollout_path"] = str(rollout_path)
    _state["codex_rollout_mtime_ns"] = stat.st_mtime_ns
    _state["codex_rollout_telemetry"] = latest_telemetry
    return dict(latest_telemetry) if latest_telemetry else None


def _infer_codex_context_telemetry(args: dict[str, Any]) -> dict[str, Any] | None:
    """Auto-populate structured telemetry from Codex rollout logs when safe."""
    if "context_telemetry" in args or "context_pressure" in args:
        return None

    thread_id = os.getenv("CODEX_THREAD_ID", "").strip()
    if not thread_id:
        return None

    rollout_path = _resolve_codex_rollout_path(thread_id)
    if rollout_path is None:
        return None

    return _load_codex_context_telemetry(rollout_path, thread_id)


_ORIGIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _normalize_origin_id(raw_origin_id: Any) -> str | None:
    if raw_origin_id is None:
        return None
    if not isinstance(raw_origin_id, str):
        raise ValueError("origin_id must be a string")
    origin_id = raw_origin_id.strip()
    if not origin_id:
        return None
    if not _ORIGIN_ID_PATTERN.fullmatch(origin_id):
        raise ValueError("origin_id must match ^[A-Za-z0-9._:-]{1,128}$")
    return origin_id


def _infer_origin_id(args: dict[str, Any]) -> str | None:
    explicit = _normalize_origin_id(args.get("origin_id"))
    if explicit:
        return explicit
    thread_id = os.getenv("CODEX_THREAD_ID", "").strip()
    if thread_id:
        return _normalize_origin_id(f"codex-thread:{thread_id}")
    return None


ACTIVE_WORKSTREAM_RENDER_MAX_CHARS = 1200
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


def _active_workstream_render_decision(result: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    active_workstream = result.get("active_workstream")
    if not isinstance(active_workstream, dict):
        return None

    reason = "no_topic_overlap"
    rendered_block = None
    status = active_workstream.get("status")
    binding_source = active_workstream.get("binding_source")
    workstream = active_workstream.get("workstream") or {}

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


def _apply_active_workstream_render_decision(result: Any, args: dict[str, Any]) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    try:
        decision = _active_workstream_render_decision(result, args)
        if decision is not None:
            result["active_workstream_render"] = decision
    except Exception as exc:
        logger.warning("WORKSTREAMS_PHASE4: render decision failed: %s", exc)


def _active_workstream_rendered_block_for_response(tool_name: str, result: Any) -> str | None:
    if tool_name != "pith_conversation_turn" or not isinstance(result, dict):
        return None
    decision = result.get("active_workstream_render")
    if isinstance(decision, dict) and decision.get("decision") == "render":
        block = decision.get("rendered_block")
        if isinstance(block, str) and block.strip():
            return block
    return None


# --- HTTP client ---
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create async HTTP client."""
    global _http_client
    if _http_client is None:
        headers = {"Content-Type": "application/json"}
        if PITH_API_KEY:
            headers["X-API-Key"] = PITH_API_KEY
        _http_client = httpx.AsyncClient(
            base_url=PITH_API_URL,
            headers=headers,
            timeout=30.0,
        )
    return _http_client


def _reset_client():
    """Reset the HTTP client to force a fresh TCP connection."""
    global _http_client
    if _http_client is not None:
        try:
            # Schedule close but don't await (fire-and-forget in sync context)
            asyncio.get_event_loop().create_task(_http_client.aclose())
        except Exception:
            pass
    _http_client = None
    logger.info("HTTP client reset — next call will create fresh connection")


def _api_timeout_for_endpoint(endpoint: str) -> float:
    """Per-endpoint bridge timeout. Reflection is the one synchronous endpoint
    that can exceed the 30s default at scale, so it gets longer, env-tunable
    budgets while every other endpoint keeps the 30s default."""
    if endpoint.startswith("/pith_reflect"):
        if "mode=full" in endpoint:
            return float(os.environ.get("PITH_MCP_FULL_REFLECT_TIMEOUT_S", "180"))
        # STOPGAP (REFLECT-INCR): incremental reflection is synchronous and can
        # approach the 30s default at ~26k concepts. The durable fix is the
        # forgetting SQL pushdown (A1); this env override is short-term headroom.
        return float(os.environ.get("PITH_MCP_INCREMENTAL_REFLECT_TIMEOUT_S", "60"))
    return 30.0


async def call_pith_api(
    endpoint: str,
    method: str = "GET",
    body: dict | None = None,
    *,
    drain_outbox: bool = True,
) -> dict[str, Any]:
    """Call the Pith REST API. Returns parsed JSON or error dict.

    Retries once on broken pipe / connection reset (server restart scenario).
    """
    max_retries = 2
    for attempt in range(max_retries):
        client = _get_client()
        try:
            request_timeout = _api_timeout_for_endpoint(endpoint)
            if method == "GET":
                if request_timeout == 30.0:
                    resp = await client.get(endpoint)
                else:
                    resp = await client.get(endpoint, timeout=request_timeout)
            else:
                if request_timeout == 30.0:
                    resp = await client.post(endpoint, json=body)
                else:
                    resp = await client.post(endpoint, json=body, timeout=request_timeout)

            if not resp.is_success:
                error_text = resp.text
                code = (
                    "AUTH_FAILED"
                    if resp.status_code in (401, 403)
                    else "INVALID_INPUT"
                    if resp.status_code == 400
                    else "NOT_FOUND"
                    if resp.status_code == 404
                    else "SERVER_ERROR"
                )
                # OPS-500-FIX: Retry on transient server errors (500/503) — likely DB lock contention
                if resp.status_code in (500, 503) and attempt < max_retries - 1:
                    logger.warning(
                        "OPS-500-FIX: HTTP %d on %s (attempt %d/%d), retrying...",
                        resp.status_code,
                        endpoint,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(1.0)
                    continue
                bridge_health_updates = {
                    "http_state": "reachable",
                    "last_error_code": code,
                    "last_error_endpoint": endpoint,
                }
                if resp.status_code in (500, 503):
                    bridge_health_updates["http_state"] = "server_error"
                if resp.status_code in (401, 403):
                    bridge_health_updates["auth_state"] = "rejected"
                _update_bridge_health(**bridge_health_updates)
                _transport_event(
                    "backend_error",
                    endpoint=endpoint,
                    method=method,
                    status_code=resp.status_code,
                    backend_error_code=code,
                    message=f"HTTP {resp.status_code}: {resp.reason_phrase}",
                    cached_session_id=_state.get("cached_session_id"),
                )
                return {
                    "error": True,
                    "code": code,
                    "status_code": resp.status_code,
                    "message": f"HTTP {resp.status_code}: {resp.reason_phrase}",
                    "details": error_text,
                    "tool": endpoint,
                }
            payload = resp.json()
            success_updates = {
                "transport_state": "open",
                "http_state": "reachable",
                "last_error_code": None,
                "last_error_endpoint": None,
            }
            if endpoint == "/readyz":
                success_updates["last_readyz"] = _readyz_subset(payload)
            _update_bridge_health(**success_updates)
            if drain_outbox:
                _schedule_bridge_outbox_drain()
            return payload
        except httpx.ConnectError:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Connection refused on {endpoint} (attempt {attempt + 1}), retrying with fresh client..."
                )
                _reset_client()
                await asyncio.sleep(1)
                continue
            _update_bridge_health(
                transport_state="degraded",
                http_state="refused",
                last_error_code="CONNECTION_REFUSED",
                last_error_endpoint=endpoint,
            )
            return {
                "error": True,
                "code": "CONNECTION_REFUSED",
                "message": "Pith server is not running",
                "hint": "Start it with: pith start (or python -m uvicorn app.server:app)",
                "tool": endpoint,
            }
        except (httpx.RemoteProtocolError, httpx.ReadError, ConnectionResetError, BrokenPipeError, OSError) as e:
            # Broken pipe / connection reset — server restarted, stale TCP connection
            if attempt < max_retries - 1:
                logger.warning(f"Connection error on {endpoint}: {e}. Resetting client and retrying...")
                _reset_client()
                await asyncio.sleep(0.5)
                continue
            _update_bridge_health(
                transport_state="degraded",
                http_state="reset",
                last_error_code="CONNECTION_RESET",
                last_error_endpoint=endpoint,
            )
            return {
                "error": True,
                "code": "CONNECTION_RESET",
                "message": f"Server connection lost: {e}",
                "hint": "Server may have restarted. This should auto-recover on next call.",
                "tool": endpoint,
            }
        except httpx.TimeoutException as e:
            _update_bridge_health(
                transport_state="degraded",
                http_state="timeout",
                last_error_code="TIMEOUT",
                last_error_endpoint=endpoint,
            )
            return {
                "error": True,
                "code": "TIMEOUT",
                "message": f"Pith API request timed out on {endpoint}: {e}",
                "hint": "Use exec fallback for this turn and retry direct MCP after the backend finishes or recovers.",
                "tool": endpoint,
            }
        except Exception as e:
            _update_bridge_health(
                transport_state="degraded",
                http_state="server_error",
                last_error_code="SERVER_ERROR",
                last_error_endpoint=endpoint,
            )
            return {
                "error": True,
                "code": "SERVER_ERROR",
                "message": str(e),
                "tool": endpoint,
            }
    # Unreachable, but defensive
    return {"error": True, "code": "RETRY_EXHAUSTED", "message": "All retries failed", "tool": endpoint}


async def _perform_durable_write(
    endpoint: str,
    body: dict[str, Any],
    *,
    request_id_prefix: str,
) -> dict[str, Any]:
    request_id = body.get("request_id") or _make_request_id(request_id_prefix)
    durable_body = dict(body)
    durable_body["request_id"] = request_id
    ready = await call_pith_api("/readyz", "GET", drain_outbox=False)
    if _is_durable_transport_error(ready):
        return _queue_durable_write(endpoint, "POST", durable_body, request_id, ready)
    if not ready.get("error") and ready.get("write_state") != "accepting":
        return _queue_durable_write(
            endpoint,
            "POST",
            durable_body,
            request_id,
            {
                "error": True,
                "code": "WRITE_NOT_READY",
                "message": "Server write path is not accepting requests yet",
                "details": {
                    "mode": ready.get("mode"),
                    "write_state": ready.get("write_state"),
                    "retrieval_state": ready.get("retrieval_state"),
                },
                "tool": endpoint,
            },
        )
    result = await call_pith_api(endpoint, "POST", durable_body)
    if _is_durable_transport_error(result):
        return _queue_durable_write(endpoint, "POST", durable_body, request_id, result)
    return result


# --- L3: Protocol enforcement ---
def _get_protocol_status(tool_name: str) -> dict:
    """Generate protocol status for injection into every response."""
    now = time.time()
    time_since_learn = round(now - _state["last_learn_timestamp"]) if _state["last_learn_timestamp"] else None
    status = {
        "session_active": bool(_state["cached_session_id"]),
        "session_auto_created": _state["auto_session_created"],
        "learning_debt": _state["learning_debt"],
        "calls_since_session_start": _state["total_calls_since_session_start"],
        "seconds_since_last_learn": time_since_learn,
    }
    if _state["learning_debt"] >= LEARNING_DEBT_THRESHOLD * 2:
        status["urgency"] = "critical"
        status["nudge"] = (
            f"KNOWLEDGE LOSS RISK: {_state['learning_debt']} substantive exchanges "
            "without pith_session_learn. Call pith_session_learn NOW with "
            "extracted_concepts_json."
        )
    elif _state["learning_debt"] >= LEARNING_DEBT_THRESHOLD:
        status["urgency"] = "warning"
        status["nudge"] = (
            f"Learning debt: {_state['learning_debt']} exchanges without "
            "pith_session_learn. Call pith_session_learn with "
            "extracted_concepts_json to capture knowledge."
        )
    else:
        status["urgency"] = "ok"
    return status


# --- L4: Cognitive bootstrap ---
def _format_bootstrap_orientation(session_start_result: dict) -> str:
    """TEMPORAL_AWARENESS v2.4: Concise bootstrap with temporal directive.

    Removed: STRATEGIC PRIORITIES, RECENT WORK (stale orientation data).
    Kept: Health snapshot, active goals (governance-scored), checkpoint.
    Added: Server time, temporal awareness protocol.
    """
    from app.core.datetime_utils import _utc_now

    parts = ["=== COGNITIVE BOOTSTRAP (auto-session created) ==="]
    parts.append(f"Server time: {_utc_now().isoformat()}")
    parts.append("You have persistent memory across sessions. Use it.")
    parts.append("")

    s = session_start_result.get("session")
    if s:
        parts.append(f"Session: {s['session_id']} | Started: {s['started_at']}")

    # Recovery info
    r = session_start_result.get("recovered_sessions")
    if r:
        parts.append(f"⚠️ {r['orphaned_sessions']} orphaned session(s) recovered: {r['warning']}")

    # Health snapshot (live-computed, not stale)
    intro = session_start_result.get("introspect_summary")
    if intro:
        h = intro.get("health", {})
        ident = intro.get("identity", {})
        parts.append(
            f"Pith: {h.get('concept_count', '?')} concepts | "
            f"avg confidence {h.get('avg_confidence', '?')} | "
            f"{ident.get('pith_age_days', '?')} days old"
        )
        strengths = intro.get("top_strengths", [])
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths[:3])}")

    # Active goals (Amendment 2: kept — these are governance-scored, not stale)
    orient = session_start_result.get("orientation")
    if orient and orient.get("where_going"):
        wg = orient["where_going"]
        goals = wg.get("active_goals", [])
        if goals:
            parts.append("")
            parts.append("ACTIVE GOALS:")
            for g in goals[:3]:
                parts.append(f"  • {g['summary']}")
        # REMOVED: STRATEGIC PRIORITIES (stale orientation data)
        # REMOVED: (was sourced from orientation.where_going.strategic_priorities)

    # Checkpoint (useful for session resumption)
    cp = session_start_result.get("checkpoint")
    if cp:
        parts.append("")
        parts.append(f"PENDING CHECKPOINT ({cp['task_id']}):")
        if cp.get("active"):
            parts.append(f"  Active: {cp['active']}")
        if cp.get("next"):
            parts.append(f"  Next: {', '.join(cp['next'])}")

    # REMOVED: RECENT WORK (stale orientation data — showed empty lines)

    parts.append("")
    parts.append("PROTOCOL: Concepts have age_minutes and freshness_label. Older concepts may be outdated.")
    parts.append("Call pith_conversation_turn BEFORE responding. Include extracted_concepts_json.")
    parts.append("=== END BOOTSTRAP ===")
    return "\n".join(parts)


# --- C1: Auto-session management ---
_session_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None


async def _get_session_lock() -> asyncio.Lock:
    """Lazy-init the session lock (must be created inside event loop)."""
    global _session_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    return _session_lock


async def ensure_session(tool_name: str) -> str | None:
    """C1: Auto-create session if none exists. Returns session ID."""
    now = time.time()

    # Conversation boundary detection
    if (
        _state["cached_session_id"]
        and _state["last_session_activity"]
        and (now - _state["last_session_activity"]) > CONVERSATION_BOUNDARY_S
    ):
        idle_s = round(now - _state["last_session_activity"])
        logger.info(f"C1: Conversation boundary detected ({idle_s}s idle). Ending stale session.")
        if _state["learning_debt"] > 0:
            logger.warning(f"L3: Boundary with debt {_state['learning_debt']}")
        try:
            await call_pith_api("/session_end", "POST")
        except Exception as e:
            logger.error(f"C1: session_end failed: {e}")
        _state["cached_session_id"] = None
        _state["last_session_activity"] = None
        _state["learning_debt"] = 0
        _state["total_calls_since_session_start"] = 0
        _state["last_conv_turn_args"] = None
        _state["pending_bootstrap_orientation"] = None

    # Short-circuit if session exists
    if _state["cached_session_id"]:
        return _state["cached_session_id"]

    lock = await _get_session_lock()
    async with lock:
        # Double-check after acquiring lock
        if _state["cached_session_id"]:
            return _state["cached_session_id"]

        # Check for existing active session
        sessions = await call_pith_api("/sessions_list?status=active&limit=1")
        if isinstance(sessions, list) and len(sessions) > 0:
            if _state["is_first_ensure_session"]:
                # C1.1: New process — end stale session
                logger.info(
                    f"C1.1: New process detected stale session {sessions[0].get('session_id', sessions[0].get('id'))}. Ending."
                )
                _state["is_first_ensure_session"] = False
                try:
                    await call_pith_api("/session_end", "POST")
                except Exception as e:
                    logger.error(f"C1.1: session_end failed: {e}")
                # Fall through to create new session
            else:
                _state["cached_session_id"] = sessions[0].get("session_id", sessions[0].get("id"))
                _state["last_session_activity"] = now
                return _state["cached_session_id"]

        _state["is_first_ensure_session"] = False

        # Create new session
        result = await call_pith_api("/session_start", "POST", {"context_hint": f"auto:{tool_name}"})

        if result and not result.get("error") and result.get("session"):
            _state["cached_session_id"] = result["session"]["session_id"]
            _state["last_session_activity"] = now
            _state["auto_session_created"] = True
            _state["learning_debt"] = 0
            _state["total_calls_since_session_start"] = 0
            _state["last_conv_turn_args"] = None
            logger.info(f"C1: Auto-session created for {tool_name}: {_state['cached_session_id']}")

            # L4: Capture bootstrap orientation
            _state["pending_bootstrap_orientation"] = _format_bootstrap_orientation(result)
            logger.info(f"L4: Bootstrap orientation captured ({len(_state['pending_bootstrap_orientation'])} chars)")

        return _state["cached_session_id"]


# --- C4: Dynamic instructions generation ---
async def generate_descriptive_instructions() -> str:
    """Generate instructions from live Pith data. Falls back to static."""
    if os.getenv("DESCRIPTIVE_INSTRUCTIONS_ENABLED") == "false":
        logger.info("Descriptive instructions disabled via env var")
        return STATIC_FALLBACK_INSTRUCTIONS

    try:
        stats, areas = await asyncio.gather(
            call_pith_api("/pith_stats?detail=fast"),
            call_pith_api("/knowledge_areas"),
        )
        if stats.get("error") or areas.get("error"):
            logger.warning("Pith API unavailable for instructions, using static fallback")
            return STATIC_FALLBACK_INSTRUCTIONS

        # Top 5 knowledge areas
        top_areas = ", ".join(f"{a['name']} ({a['concept_count']})" for a in (areas.get("areas") or [])[:5])

        # Maturity hint
        tc = stats.get("total_concepts", 0)
        if tc == 0:
            maturity_hint = (
                "Getting started: Just have a normal conversation. "
                "Pith captures decisions, discoveries, and patterns automatically."
            )
        elif tc < 10:
            maturity_hint = (
                f"Pith is young ({tc} concepts). Keep chatting — it gets dramatically better after 50+ concepts."
            )
        elif tc < 50:
            maturity_hint = (
                f"Pith is growing ({tc} concepts). Try asking 'what do you know about me?' to see what it's learned."
            )
        else:
            maturity_hint = ""

        lines = [
            "⚠️ MANDATORY FIRST ACTION — NON-NEGOTIABLE:",
            "Before responding to the user's VERY FIRST message, call pith_conversation_turn(message=user's message).",
            "This retrieves critical context about this user, their projects, and ongoing work.",
            "Skipping this makes you appear amnesiac and wastes the user's time repeating context.",
            "Do this BEFORE AskUserQuestion, BEFORE TodoWrite, BEFORE any other tool. Pith first. Always.",
            "",
            "You have persistent memory that learns and remembers across conversations.",
            (
                f"It contains {tc} concepts across {stats.get('associations', 0)} relationships."
                if tc > 0
                else "It's a fresh start — everything learned will come from YOUR conversations."
            ),
            maturity_hint,
            f"Key topics: {top_areas}." if top_areas else "",
            "",
            "COGNITIVE LOOP (simplified — ONE tool does everything):",
            "1. BEFORE responding: call pith_conversation_turn with:",
            "   - message: the user's current message",
            "   - previous_response: your LAST response (REQUIRED after first exchange)",
            "   - previous_message: the user's previous message",
            "   - extracted_concepts_json: 1-7 concepts from your previous response",
            "2. RESPOND using activated context + any extraction_request hints",
            "3. When conversation ends: call pith_session_end with previous_response",
            "",
            "ONE tool call per exchange. The server handles all learning automatically.",
            "pith_session_learn is still available for explicit extraction when needed.",
            "",
            'CONCEPT FORMAT: [{"summary": "30-500 chars", "confidence": 0.6, "knowledge_area": "domain", "evidence": ["source"], "concept_type": "decision"}]',
            "ALWAYS set concept_type: observation, pattern, decision, principle, method, heuristic, cognitive_strategy.",
            "If exchange was trivial, send '[]' — do NOT invent filler concepts.",
            "SUMMARY PRECISION: Always preserve proper nouns, specific numbers/amounts/dates/times, named entities.",
            'WRONG: "recommended a light beer" → RIGHT: "recommended Pilsner or Lager for Seco de Cordero"',
            "",
            "SESSION LIFECYCLE:",
            "- pith_session_start at conversation beginning (includes orientation)",
            "- pith_session_end when conversation concludes — ALWAYS include previous_response",
            "",
            "EXTRACTION EXAMPLES — L1 vs L3+:",
            "BAD (L1): {summary:'We fixed the bug by changing line 222', concept_type:'observation'}",
            "GOOD (L3): {summary:'PRINCIPLE: When changing a validation limit, grep the entire codebase for all enforcement points', concept_type:'principle', evidence:['verified: second check found at line 222']}",
            "GOOD (factual/L1): {summary:'MAX_ALWAYS_ACTIVATE is 6, CONTEXT_BUDGET_MAIN is 20 — leaving 14 contextual retrieval slots', concept_type:'observation', evidence:['verified: config.py lines 150-165']}",
            "",
            "Pith gets smarter with every conversation. Your job is to feed it quality knowledge.",
        ]
        instructions = "\n".join(line for line in lines if line is not None)
        logger.info(f"Descriptive instructions generated: {len(instructions)} chars")
        return instructions
    except Exception as e:
        logger.error(f"Failed to generate descriptive instructions: {e}")
        return STATIC_FALLBACK_INSTRUCTIONS


# --- Tool Definitions (schemas mirrored from server.js where applicable) ---
TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "pith_search",
        "description": "Search for concepts in Pith using semantic similarity",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "context": {"type": "string", "description": "Current context (optional)"},
                "goal": {"type": "string", "description": "Current goal/task (optional)"},
                "max_results": {"type": "number", "description": "Maximum results to return (default: 5)"},
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0-1.0, default: 0.0)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "pith_get_concept",
        "description": "Get detailed information about a specific concept, including version history",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_id": {"type": "string", "description": "The concept ID"},
                "version": {"type": "string", "description": "Specific version or 'latest' (default)"},
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "pith_related_concepts",
        "description": "Get concepts related to a specific concept through associations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_id": {"type": "string", "description": "The concept ID"},
                "max_depth": {"type": "number", "description": "Maximum depth for relationship traversal (default: 2)"},
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "pith_propose_concept",
        "description": "Propose a new concept to be learned by Pith",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_id": {"type": "string", "description": "Unique identifier (snake_case)"},
                "summary": {"type": "string", "description": "Clear, concise summary"},
                "evidence": {"type": "array", "items": {"type": "string"}, "description": "Evidence sources"},
                "signals": {"type": "array", "items": {"type": "string"}, "description": "Observable signals"},
                "knowledge_area": {"type": "string", "description": "Knowledge domain (default: 'general')"},
                "confidence": {"type": "number", "description": "Initial confidence (0.0-1.0, default: 0.5)"},
                "associations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of related concepts",
                },
                "concept_type": {
                    "type": "string",
                    "description": "Knowledge hierarchy type: observation, decision, principle, method, heuristic, cognitive_strategy, pattern",
                },
                "always_activate": {
                    "type": "boolean",
                    "description": "If true, injected into EVERY conversation_turn response",
                },
            },
            "required": ["concept_id", "summary", "evidence", "knowledge_area"],
        },
    },
    {
        "name": "pith_evolve_concept",
        "description": "Evolve an existing concept with new information or evidence",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_id": {"type": "string", "description": "The concept ID to evolve"},
                "new_summary": {"type": "string", "description": "Updated summary (optional)"},
                "new_evidence": {"type": "array", "items": {"type": "string"}, "description": "Additional evidence"},
                "new_signals": {"type": "array", "items": {"type": "string"}, "description": "New signals"},
                "confidence_change": {"type": "number", "description": "Change in confidence (-1.0 to 1.0)"},
                "new_concept_type": {"type": "string", "description": "Reclassify concept type"},
                "always_activate": {"type": "boolean", "description": "Set always-activate flag"},
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "pith_set_always_activate",
        "description": "Set or unset the always-activate flag on a concept.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_id": {"type": "string", "description": "The concept ID"},
                "value": {"type": "boolean", "description": "True to enable, false to disable (default: true)"},
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "pith_link_concepts",
        "description": "Create an association between two concepts",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_a": {"type": "string", "description": "First concept ID"},
                "concept_b": {"type": "string", "description": "Second concept ID"},
                "relation": {"type": "string", "description": "Type of relation"},
                "strength": {"type": "number", "description": "Relation strength (0.0-1.0, default: 0.5)"},
            },
            "required": ["concept_a", "concept_b", "relation"],
        },
    },
    {
        "name": "pith_stats",
        "description": "Get overall pith statistics and health metrics",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pith_health",
        "description": "Get detailed health analysis of the Pith system",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pith_bridge_status",
        "description": (
            "Get local MCP bridge process, heartbeat, runtime, and transport-log status without calling the backend."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pith_projection",
        "description": "Get predictive memory growth projection — velocity, per-KA growth, maturity distribution, capacity estimates",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pith_reflect",
        "description": "Run reflection/consolidation cycle to merge concepts, apply decay, and cleanup. Always returns reflection_summary (human-readable) plus key counts. Set verbose=true to include internal phase_timings and evidence_cv breakdowns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Reflection mode: 'incremental' or 'full' (default: 'incremental')",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "If true, include phase_timings and evidence_cv breakdowns. Default: false.",
                },
            },
        },
    },
    {
        "name": "pith_checkpoint",
        "description": "Save/load execution state for cross-session resumption.\n\nActions:\n- save: Upsert checkpoint (done[] is append-only via union merge)\n- load: Get most recent checkpoint, or by task_id\n- list: Show all active checkpoints\n- complete: Mark done (short 24h TTL)\n- touch: Extend TTL without changing content",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "load", "list", "complete", "touch"],
                    "description": "Checkpoint operation",
                },
                "task_id": {"type": "string", "description": "Human-readable work stream ID"},
                "status": {
                    "type": "string",
                    "enum": ["planning", "active", "blocked", "paused"],
                    "description": "Task status",
                },
                "description": {"type": "string", "description": "What we're working on"},
                "done": {"type": "array", "items": {"type": "string"}, "description": "Completed items"},
                "active": {"type": "string", "description": "Current item in progress"},
                "next": {"type": "array", "items": {"type": "string"}, "description": "Upcoming items"},
                "blockers": {"type": "array", "items": {"type": "string"}, "description": "Blockers"},
                "context": {"type": "object", "description": "Freeform key/value state"},
                "concept_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concept IDs created during this task",
                },
                "origin_id": {
                    "type": "string",
                    "description": "Stable client/thread/workstream identifier for checkpoint authority and optional replay",
                },
                "op_id": {
                    "type": "number",
                    "description": "Monotonic operation identifier within origin_id",
                },
                "payload_hash": {
                    "type": "string",
                    "description": "Optional payload fingerprint for replay diagnostics",
                },
                "request_id": {
                    "type": "string",
                    "description": "Optional idempotency key for durable checkpoint writes. Auto-generated when omitted on save/touch/complete.",
                },
                "ttl_days": {"type": "number", "description": "Override default 7-day TTL (max 30)"},
                "max_age_hours": {"type": "number", "description": "Max age for load (default: 24h)"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "pith_questions",
        "description": "Get pending questions Pith has about weak/uncertain concepts",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "number", "description": "Max questions (default: 10)"}},
        },
    },
    {
        "name": "pith_activate_context",
        "description": "Activate concepts based on current conversation context for faster retrieval",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Current conversation context"},
                "boost": {"type": "number", "description": "Activation boost level (0.0-1.0, default: 0.5)"},
            },
            "required": ["context"],
        },
    },
    {
        "name": "pith_set_goal",
        "description": "Set current goal for goal-directed concept retrieval",
        "inputSchema": {
            "type": "object",
            "properties": {"goal": {"type": "string", "description": "Goal type"}},
            "required": ["goal"],
        },
    },
    {
        "name": "pith_import_conversation",
        "description": "Import and learn from historical conversation text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_text": {"type": "string", "description": "The conversation text to import"},
                "source_id": {"type": "string", "description": "Source identifier"},
                "knowledge_area": {"type": "string", "description": "Knowledge area (default: 'imported')"},
            },
            "required": ["conversation_text", "source_id"],
        },
    },
    {
        "name": "pith_session_start",
        "description": "Start a new cognitive session. Bootstraps orientation and self-model introspection.\n\nPITH PROTOCOL ESSENTIALS:\n1. Pith First: Always call pith_conversation_turn BEFORE composing substantive responses.\n2. Learn After: Call pith_session_learn AFTER exchanges where decisions were made.\n3. Dual Learning: session_learn captures ~60-70% via heuristics. ALWAYS include extracted_concepts_json.\n4. Checkpoint every 30 min.\n5. Use specific knowledge_area values.\n6. Evolve existing concepts when status changes.",
        "inputSchema": {
            "type": "object",
            "properties": {"context_hint": {"type": "string", "description": "Session focus context"}},
        },
    },
    {
        "name": "pith_session_end",
        "description": "End the current cognitive session. ALWAYS include previous_response to prevent knowledge loss.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Optional session id to close explicitly"},
                "origin_id": {
                    "type": "string",
                    "description": "Stable client/thread/workstream identifier for closeout binding",
                },
                "previous_response": {"type": "string", "description": "Your last response to the user"},
                "previous_message": {"type": "string", "description": "The user's last message"},
                "extracted_concepts_json": {"type": "string", "description": "Concepts extracted from final response"},
                "request_id": {
                    "type": "string",
                    "description": "Optional idempotency key for durable session end writes. Auto-generated when omitted.",
                },
            },
        },
    },
    {
        "name": "pith_conversation_turn",
        "description": "MANDATORY FIRST CALL — call BEFORE composing ANY substantive response. Retrieves critical context AND auto-learns from your previous exchange. REQUIRED fields (after first exchange): message, previous_response, extracted_concepts_json (1-7 concepts from your previous response). The server auto-learns, retrieves relevant context, and may return extraction_request hints for knowledge gaps it detected. This single call replaces the old conversation_turn + session_learn workflow. Response includes is_resumption, orientation_summary, checkpoint_suggested, and extraction_request.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The user's current message"},
                "conversation_context": {
                    "type": "string",
                    "description": "Recent conversation context (max 2000 chars)",
                },
                "session_id": {"type": "string", "description": "Current session ID (optional)"},
                "max_concepts": {"type": "number", "description": "Max concepts to retrieve (default: 14)"},
                "include_predictions": {
                    "type": "boolean",
                    "description": "Include predictive activations (default: false)",
                },
                "compaction_detected": {
                    "type": "boolean",
                    "description": "Optional client-side compaction signal when the host knows compaction already occurred.",
                },
                "context_pressure": {
                    "type": "number",
                    "description": "Legacy raise-only context pressure hint in the 0.0-1.0 range.",
                },
                "context_telemetry": {
                    "type": "object",
                    "description": "Structured, model-agnostic context telemetry. Supports pressure_ratio, measurement_source, measurement_confidence, measurement_scope, used_tokens, window_size_tokens, and source_metadata.",
                },
                "previous_response": {"type": "string", "description": "Your previous response to the user"},
                "previous_message": {"type": "string", "description": "The user's previous message"},
                "extracted_concepts_json": {
                    "type": "string",
                    "description": "JSON string of 1-7 concept objects from your PREVIOUS response",
                },
                "origin_id": {
                    "type": "string",
                    "description": "Stable client/thread/workstream identifier for authoritative checkpoint binding.",
                },
                "current_task_id": {
                    "type": "string",
                    "description": "Explicit checkpoint task_id to prefer over origin binding when known.",
                },
                "context_authority_mode": {
                    "type": "string",
                    "enum": ["strict", "balanced", "permissive"],
                    "description": "How conservatively the server should present candidate working context.",
                },
                "surface_id": {
                    "type": "string",
                    "description": "Optional consumer surface identifier. Defaults to bridge PITH_SURFACE_ID when configured.",
                },
                "platform_hint": {
                    "type": "string",
                    "description": "Optional client platform hint. Usually derived by the server from surface_id.",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Stable workspace identifier for consumer lifecycle binding.",
                },
                "context_delivery_mode": {
                    "type": "string",
                    "description": "Consumer lifecycle context delivery mode.",
                },
                "surface_lifecycle_version": {
                    "type": "string",
                    "description": "Consumer lifecycle contract version.",
                },
                "response_mode": {
                    "type": "string",
                    "enum": ["compact", "full"],
                    "description": "Model-facing response shape. Defaults to compact; full is diagnostic and size-capped.",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "pith_session_learn",
        "description": "CRITICAL: Always include extracted_concepts_json with every call. Post-response learning. Without extracted_concepts, only ~60-70% of explicit insights are captured. Call after EVERY meaningful exchange.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_message": {"type": "string", "description": "The user's message"},
                "assistant_response": {"type": "string", "description": "The assistant's response"},
                "session_id": {"type": "string", "description": "Current session ID (optional)"},
                "knowledge_area": {"type": "string", "description": "Knowledge domain (default: 'conversation')"},
                "auto_associate": {"type": "boolean", "description": "Auto-link new concepts (default: true)"},
                "request_id": {
                    "type": "string",
                    "description": "Optional idempotency key for durable session learn writes. Auto-generated when omitted.",
                },
                "extracted_concepts": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Pre-extracted concepts (array variant)",
                },
                "extracted_concepts_json": {
                    "type": "string",
                    "description": "REQUIRED — JSON string of 1-7 concept objects",
                },
            },
            "required": ["user_message", "assistant_response"],
        },
    },
    {
        "name": "pith_orient",
        "description": "Generate present moment orientation — where Pith has been, where it is now, and where it's going.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time_window": {
                    "type": "string",
                    "description": "Time window: '1_day', '7_days' (default), '30_days', or 'all'",
                }
            },
        },
    },
    {
        "name": "pith_sessions_list",
        "description": "List past cognitive sessions with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter: 'active', 'ended', or 'recovered'"},
                "limit": {"type": "number", "description": "Max sessions (default: 20)"},
                "since": {"type": "string", "description": "Only sessions since ISO datetime"},
            },
        },
    },
    {
        "name": "pith_auto_associate_batch",
        "description": "Run batch auto-association across all concepts using TF-IDF cosine similarity. WARNING: Can take 30-60s on 300+ concepts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "description": "Preview without creating edges (default: false)"},
                "tier1_threshold": {"type": "number", "description": "Cosine similarity threshold (default: 0.12)"},
                "tier2_enabled": {"type": "boolean", "description": "Enable secondary tier (default: true)"},
            },
        },
    },
    {
        "name": "pith_validate_response",
        "description": "Validate draft response against active constraints from conversation_turn.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "response_text": {"type": "string", "description": "Your draft response to validate"},
                "constraint_set": {"type": "object", "description": "The constraint_set from conversation_turn"},
            },
            "required": ["response_text", "constraint_set"],
        },
    },
    {
        "name": "pith_benchmark",
        "description": "Run the internal CogGov governance diagnostic. Use this as local health evidence only; it is not public benchmark evidence. Modes: 'light' for dims 1-3, 'full' for all 6 + adversarial.",
        "inputSchema": {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["light", "full"], "description": "Diagnostic mode"}},
        },
    },
    {
        "name": "pith_cko_create",
        "description": "Create a Compound Knowledge Object (CKO) — Layer 4. Bundles related concepts into a coherent whole.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "CKO title"},
                "concept_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered constituent concept IDs",
                },
                "synthesis": {"type": "string", "description": "500-2000 char synthesis"},
                "knowledge_area": {"type": "string", "description": "Knowledge domain (default: 'general')"},
                "cko_type": {
                    "type": "string",
                    "enum": ["analysis", "plan", "assessment", "investigation"],
                    "description": "CKO type",
                },
            },
            "required": ["title", "concept_ids", "synthesis"],
        },
    },
    {
        "name": "pith_cko_get",
        "description": "Load a single CKO by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"cko_id": {"type": "string", "description": "CKO ID"}},
            "required": ["cko_id"],
        },
    },
    {
        "name": "pith_cko_search",
        "description": "Search CKOs for context assembly. Returns up to 3 CKOs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_area": {"type": "string", "description": "Optional knowledge_area filter"},
                "max_results": {"type": "number", "description": "Max CKOs (default: 3)"},
            },
        },
    },
    {
        "name": "pith_cko_update",
        "description": "Update a CKO's synthesis and/or constituent concept list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cko_id": {"type": "string", "description": "CKO ID"},
                "synthesis": {"type": "string", "description": "New synthesis text"},
                "concept_ids": {"type": "array", "items": {"type": "string"}, "description": "New concept IDs"},
            },
            "required": ["cko_id"],
        },
    },
    {
        "name": "pith_cko_lifecycle",
        "description": "Run CKO lifecycle management: refresh scores, archive stale CKOs, identify merge candidates.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pith_cko_list",
        "description": "List CKOs with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "degraded", "stale", "archived"],
                    "description": "Filter by status",
                },
                "knowledge_area": {"type": "string", "description": "Filter by knowledge area"},
                "limit": {"type": "number", "description": "Max results (default: 50)"},
            },
        },
    },
    # Wave 4: Belief Diff + Epistemic Migration
    {
        "name": "pith_belief_diff",
        "description": "Compare Pith's belief state at two points in time. Returns what was added, removed (superseded), changed (authority/maturity shift), and unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "t1": {"type": "string", "description": "ISO datetime for earlier state (e.g., '2026-03-01T00:00:00')"},
                "t2": {"type": "string", "description": "ISO datetime for later state (e.g., '2026-03-05T00:00:00')"},
                "knowledge_area": {"type": "string", "description": "Optional filter for specific knowledge domain"},
            },
            "required": ["t1", "t2"],
        },
    },
    {
        "name": "pith_migrate_epistemic",
        "description": "Migrate existing concepts to extended epistemic networks. Scans all current concepts and reclassifies based on provenance signals. Use dry_run=true (default) to preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "If true (default), report what WOULD change without changing it",
                    "default": True,
                },
            },
        },
    },
    # Wave 5: Narrative Threads + Cognitive Traces
    {
        "name": "pith_threads",
        "description": "Manage narrative threads and consumer-safe Workstreams. Use workstream_lifecycle for Workstream writes; this default consumer surface does not expose raw Workstream admin write actions. Use ensure_workstream_activation(candidate) as the read-only activation gate for substantive durable work; bind_existing, create_and_bind, and skip require explicit operator confirmation. Workstream context is continuity context, not instruction authority. Raw Workstream writes are available only through pith_threads_admin when admin raw writes are enabled.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "get",
                        "list",
                        "update",
                        "close",
                        "reactivate",
                        "link",
                        "unlink",
                        "similar",
                        "stats",
                        "classify_workstreams",
                        "workstream_context",
                        "active_workstream",
                        "ensure_workstream_activation",
                        "workstream_lifecycle",
                        "workstream_hygiene_dry_run",
                    ],
                    "description": "Action to perform (default: list)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["candidate", "bind_existing", "create_and_bind", "skip"],
                    "description": "ensure_workstream_activation mode; candidate is read-only, write modes require strict operator confirmation",
                },
                "verb": {
                    "type": "string",
                    "enum": ["start", "adopt", "progress", "complete", "reopen", "archive"],
                    "description": "Production Workstream lifecycle verb for workstream_lifecycle",
                },
                "thread_id": {
                    "type": "string",
                    "description": "Thread ID (required for get/update/close/reactivate/link/unlink/workstream_context and most Workstream write actions)",
                },
                "title": {"type": "string", "description": "Thread title (required for create, optional for update)"},
                "description": {"type": "string", "description": "Thread description (optional)"},
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": "Thread urgency tier (default: normal)",
                },
                "concept_id": {"type": "string", "description": "Concept ID (required for link/unlink)"},
                "role": {
                    "type": "string",
                    "enum": ["initiator", "member", "evidence", "blocker", "conclusion"],
                    "description": "Concept role in thread (default: member)",
                },
                "status": {"type": "string", "description": "Filter by status for list action"},
                "situation": {"type": "string", "description": "Situation description for similar action"},
                "intent": {"type": "string", "description": "Intent description for similar action (optional)"},
                "limit": {
                    "type": "number",
                    "description": "Max results for list, similar, and classify_workstreams actions (list default: 20, max: 100; similar default: 5)",
                },
                "max_refs": {
                    "type": "number",
                    "description": "Max typed references for workstream_context (default: 10, max: 20)",
                },
                "operator_mode": {
                    "type": "boolean",
                    "description": "Allow maintenance-cluster context blocks for operator review (default: false)",
                },
                "include_maintenance": {
                    "type": "boolean",
                    "description": "Include maintenance clusters in classify_workstreams (default: true)",
                },
                "include_concept_summaries": {
                    "type": "boolean",
                    "description": "Include concept summary fields in workstream_context refs (default: true)",
                },
                "agent_id": {"type": "string", "description": "Agent ID for classify_workstreams (default: default)"},
                "goal_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Goal IDs to associate (optional)",
                },
                "knowledge_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Knowledge areas to associate (optional)",
                },
                "current_objective": {
                    "type": "string",
                    "description": "Current Workstream objective for lifecycle metadata",
                },
                "current_summary": {
                    "type": "string",
                    "description": "Current Workstream summary for lifecycle metadata",
                },
                "next_action": {
                    "type": "string",
                    "description": "Next Workstream action for lifecycle metadata",
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current Workstream blockers",
                },
                "quality_state": {
                    "type": "string",
                    "enum": ["ok", "needs_review", "blocked"],
                    "description": "Workstream quality state",
                },
                "origin_id": {
                    "type": "string",
                    "description": "Stable origin ID for active_workstream/ensure_workstream_activation/workstream_lifecycle",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID fallback for active_workstream/ensure_workstream_activation/workstream_lifecycle",
                },
                "current_task_id": {
                    "type": "string",
                    "description": "Current task ID for exact Workstream authority",
                },
                "metadata": {
                    "type": "object",
                    "description": "Workstream metadata for ensure_workstream_activation create_and_bind",
                },
                "skip_reason": {
                    "type": "string",
                    "description": "Explicit reason for ensure_workstream_activation skip",
                },
                "operator_confirmed": {
                    "type": "boolean",
                    "description": "Strict operator confirmation for ensure_workstream_activation write modes; truthy strings do not count",
                },
                "include_proof_candidates": {
                    "type": "boolean",
                    "description": "Include capped proof/maintenance candidates in activation candidate response",
                },
                "op_id": {
                    "type": "number",
                    "description": "Optional monotonic operation ID for idempotent lifecycle writes",
                },
                "payload_hash": {
                    "type": "string",
                    "description": "Optional payload hash for idempotent lifecycle writes",
                },
                "request_id": {
                    "type": "string",
                    "description": "Optional lifecycle trace ID; Phase 1 does not provide durable request-id replay protection",
                },
                "created_by": {"type": "string", "description": "Workstream creator marker (default: user)"},
                "updated_by": {"type": "string", "description": "Workstream updater marker (default: user)"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "pith_threads_admin",
        "description": "Admin/debug Workstream raw write surface. Use only for explicit repair or operator-controlled curation when WORKSTREAMS_WRITE_ENABLED/admin_raw_writes_enabled is true. Default consumers should use pith_threads with workstream_lifecycle instead.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_workstream",
                        "promote_workstream",
                        "update_workstream",
                        "bind_workstream",
                        "clear_workstream_binding",
                        "workstream_hygiene_apply",
                        "workstream_hygiene_rollback",
                        "workstream_promote_discovery_candidate",
                        "workstream_demote_discovery_candidate",
                    ],
                    "description": "Raw admin Workstream action",
                },
                "thread_id": {"type": "string", "description": "Thread ID for raw admin Workstream actions"},
                "title": {"type": "string", "description": "Workstream title for create_workstream"},
                "description": {"type": "string", "description": "Thread description (optional)"},
                "urgency": {"type": "string", "enum": ["low", "normal", "high"], "description": "Thread urgency tier"},
                "goal_ids": {"type": "array", "items": {"type": "string"}, "description": "Goal IDs to associate"},
                "knowledge_areas": {"type": "array", "items": {"type": "string"}, "description": "Knowledge areas to associate"},
                "current_objective": {"type": "string", "description": "Current Workstream objective"},
                "current_summary": {"type": "string", "description": "Current Workstream summary"},
                "next_action": {"type": "string", "description": "Next Workstream action"},
                "blockers": {"type": "array", "items": {"type": "string"}, "description": "Current Workstream blockers"},
                "quality_state": {"type": "string", "enum": ["ok", "needs_review", "blocked"], "description": "Workstream quality state"},
                "origin_id": {"type": "string", "description": "Stable origin ID for raw binding repair"},
                "session_id": {"type": "string", "description": "Session ID fallback for raw binding repair"},
                "current_task_id": {"type": "string", "description": "Current task ID for exact binding repair"},
                "operator_confirmed": {"type": "boolean", "description": "Strict operator confirmation for admin/hygiene writes"},
                "evaluated_at": {"type": "string", "description": "Dry-run evaluated_at timestamp required for workstream_hygiene_apply"},
                "fingerprints": {"type": "object", "description": "Dry-run fingerprints required for workstream_hygiene_apply"},
                "proposed_states": {"type": "object", "description": "Dry-run proposed discovery states required for workstream_hygiene_apply"},
                "promotion_reason": {"type": "string", "description": "Required reason for workstream_promote_discovery_candidate"},
                "reason": {"type": "string", "description": "Required reason for workstream_demote_discovery_candidate"},
                "op_id": {"type": "number", "description": "Optional monotonic operation ID for idempotent raw binding writes"},
                "payload_hash": {"type": "string", "description": "Optional payload hash for idempotent raw binding writes"},
                "created_by": {"type": "string", "description": "Workstream creator marker"},
                "updated_by": {"type": "string", "description": "Workstream updater marker"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "pith_traces",
        "description": "Search and retrieve cognitive traces — structured learning event records. Actions: get (single trace), list (filter by session/trigger), search (TF-IDF query).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "list", "search"],
                    "description": "Action to perform (default: list)",
                },
                "trace_id": {"type": "string", "description": "Trace ID (required for get)"},
                "query": {"type": "string", "description": "Search query (required for search)"},
                "limit": {"type": "number", "description": "Max results (default: 20)"},
                "offset": {"type": "number", "description": "Result offset for pagination (default: 0)"},
                "session_id": {"type": "string", "description": "Filter by session ID (optional for list)"},
                "trigger_type": {"type": "string", "description": "Filter by trigger type (optional for list)"},
                "include_data": {"type": "boolean", "description": "Include full trace data (default: true)"},
            },
        },
    },
    # Metrics & observability tools
    {
        "name": "pith_learning_metrics",
        "description": "Learning performance dashboard — monitors extraction pipeline health. Shows type distribution, daily throughput, and budget utilization.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pith_metrics_dashboard",
        "description": "Critical 8 metrics dashboard — conversation turn latency, tier2 LLM costs, contradiction rates, cascade propagations, circuit breaker trips, retrieval latency, and budget overruns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO timestamp lower bound (default: last hour)"},
            },
        },
    },
    {
        "name": "pith_metrics_bg_tasks",
        "description": "Background task success/failure/cancelled rates by task name over the last 24 hours.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO timestamp lower bound (default: last 24 hours)"},
            },
        },
    },
    {
        "name": "pith_metrics_summary",
        "description": "Aggregated metrics summary with per-metric stats (count, mean, p95, max, min) and 7-day trends.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "number", "description": "Number of days to summarize (default: 7)"},
            },
        },
    },
    {
        "name": "pith_metrics_health_trend",
        "description": "Pith health score time series — daily health, maturity, connectivity, confidence, and freshness scores over N days.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "number", "description": "Number of days for trend (default: 7)"},
            },
        },
    },
    # Platform operations
    {
        "name": "pith_deploy_skills",
        "description": "Deploy skills from the active Pith skill store (~/.pith/skills when canonical hub mode is enabled; legacy ~/.claude/skills otherwise) to Claude/Cursor compatibility paths, Codex, and Cowork.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status_only": {
                    "type": "boolean",
                    "description": "If true, return status without re-deploying (default: false)",
                },
                "migrate": {
                    "type": "boolean",
                    "description": "If true, promote legacy skill stores into ~/.pith/skills before deploy",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true with migrate, report planned migration without writing",
                },
                "repair": {
                    "type": "boolean",
                    "description": "If true, repair generated surfaces from canonical content",
                },
                "verify_parity": {"type": "boolean", "description": "If true, include whole-skill parity verification"},
            },
        },
    },
]


# --- Extracted concepts parsing (P0.2: dual-format) ---
MAX_EXTRACTED_JSON_SIZE = 50_000


def _parse_extracted_concepts(args: dict) -> tuple[list | None, str]:
    """Parse extracted concepts from multiple transport formats.
    Returns (concepts_list_or_None, source_name).
    """
    # Path 1: Native array
    ec = args.get("extracted_concepts")
    if ec and isinstance(ec, list):
        return ec, "native_array"

    # Path 2: extracted_concepts as string
    if ec and isinstance(ec, str):
        if len(ec) > MAX_EXTRACTED_JSON_SIZE:
            logger.warning(f"extracted_concepts string too large: {len(ec)}")
        else:
            try:
                parsed = json.loads(ec)
                if isinstance(parsed, list):
                    return parsed, "string_parsed"
            except json.JSONDecodeError as e:
                logger.warning(f"extracted_concepts not valid JSON: {e}")

    # Path 3: Dedicated JSON string field
    ecj = args.get("extracted_concepts_json")
    if ecj:
        if not isinstance(ecj, str):
            logger.warning(f"extracted_concepts_json is not a string: {type(ecj)}")
        elif len(ecj) > MAX_EXTRACTED_JSON_SIZE:
            logger.warning(f"extracted_concepts_json too large: {len(ecj)}")
        else:
            try:
                parsed = json.loads(ecj)
                if isinstance(parsed, list):
                    return parsed, "json_fallback"
            except json.JSONDecodeError as e:
                logger.warning(f"extracted_concepts_json not valid JSON: {e}")

    return None, "none"


# --- Tool handlers ---
async def _handle_tool(name: str, args: dict) -> dict:
    """Route tool call to appropriate handler. Returns result dict."""

    # --- Simple REST wrappers (no client-side logic) ---
    if name == "pith_search":
        return await call_pith_api(
            "/pith_search",
            "POST",
            {
                "query": args["query"],
                "context": args.get("context"),
                "goal": args.get("goal"),
                "max_results": args.get("max_results", 5),
                "min_confidence": args.get("min_confidence", 0.0),
            },
        )

    if name == "pith_get_concept":
        params = f"concept_id={args['concept_id']}&version={args.get('version', 'latest')}"
        return await call_pith_api(f"/pith_get_concept?{params}")

    if name == "pith_related_concepts":
        params = f"concept_id={args['concept_id']}&max_depth={args.get('max_depth', 2)}"
        return await call_pith_api(f"/pith_related_concepts?{params}")

    if name == "pith_propose_concept":
        await ensure_session("propose_concept")
        return await call_pith_api(
            "/pith_propose_concept",
            "POST",
            {
                "concept_id": args["concept_id"],
                "summary": args["summary"],
                "evidence": args.get("evidence", []),
                "signals": args.get("signals", []),
                "knowledge_area": args.get("knowledge_area", "general"),
                "confidence": args.get("confidence", 0.5),
                "associations": args.get("associations", []),
                "concept_type": args.get("concept_type", "observation"),
                "always_activate": args.get("always_activate", False),
            },
        )

    if name == "pith_evolve_concept":
        await ensure_session("evolve_concept")
        payload = {
            "concept_id": args["concept_id"],
            "new_summary": args.get("new_summary"),
            "new_evidence": args.get("new_evidence", []),
            "new_signals": args.get("new_signals", []),
            "confidence_change": args.get("confidence_change", 0.0),
            "new_concept_type": args.get("new_concept_type"),
        }
        if "always_activate" in args:
            payload["always_activate"] = args["always_activate"]
        return await call_pith_api("/pith_evolve_concept", "POST", payload)

    if name == "pith_set_always_activate":
        return await call_pith_api(
            "/pith_set_always_activate",
            "POST",
            {
                "concept_id": args["concept_id"],
                "value": args.get("value", True),
            },
        )

    if name == "pith_link_concepts":
        await ensure_session("link_concepts")
        return await call_pith_api(
            "/pith_link_concepts",
            "POST",
            {
                "concept_a": args["concept_a"],
                "concept_b": args["concept_b"],
                "relation": args["relation"],
                "strength": args.get("strength", 0.5),
            },
        )

    if name == "pith_stats":
        return await call_pith_api("/pith_stats?detail=fast")

    if name == "pith_health":
        return await call_pith_api("/pith_health?detail=fast")

    if name == "pith_bridge_status":
        return _bridge_status()

    if name == "pith_projection":
        return await call_pith_api("/memory_projection")

    if name == "pith_reflect":
        mode = args.get("mode", "incremental")
        params = f"mode={mode}"
        if args.get("verbose") is not None:
            params += f"&verbose={str(args['verbose']).lower()}"
        return await call_pith_api(f"/pith_reflect?{params}", "POST")

    if name == "pith_checkpoint":
        try:
            origin_id = _infer_origin_id(args)
        except ValueError as exc:
            return {"error": str(exc)}
        payload = {
            "action": args.get("action", "save"),
            "task_id": args.get("task_id"),
            "status": args.get("status"),
            "description": args.get("description"),
            "done": args.get("done"),
            "active": args.get("active"),
            "next": args.get("next"),
            "blockers": args.get("blockers"),
            "context": args.get("context"),
            "concept_refs": args.get("concept_refs"),
            "session_id": _state["cached_session_id"],
            "origin_id": origin_id,
            "op_id": args.get("op_id"),
            "payload_hash": args.get("payload_hash"),
            "ttl_days": args.get("ttl_days"),
            "max_age_hours": args.get("max_age_hours"),
        }
        if payload["action"] in {"save", "touch", "complete"}:
            payload["request_id"] = args.get("request_id")
            return await _perform_durable_write("/checkpoint", payload, request_id_prefix="ckpt")
        return await call_pith_api("/checkpoint", "POST", payload)

    if name == "pith_questions":
        limit = args.get("limit", 10)
        return await call_pith_api(f"/pith_questions?limit={limit}")

    if name == "pith_activate_context":
        return await call_pith_api(
            f"/pith_activate_context?context={args['context']}&boost={args.get('boost', 0.5)}",
            "POST",
        )

    if name == "pith_set_goal":
        return await call_pith_api(f"/pith_set_goal?goal={args['goal']}", "POST")

    if name == "pith_import_conversation":
        from urllib.parse import urlencode

        params = urlencode(
            {
                "conversation_text": args["conversation_text"],
                "source_id": args["source_id"],
                "knowledge_area": args.get("knowledge_area", "imported"),
            }
        )
        return await call_pith_api(f"/pith_import_conversation?{params}", "POST")

    # --- Session lifecycle (complex client-side logic) ---
    if name == "pith_session_start":
        # Check idle timeout
        now = time.time()
        if _state["last_session_activity"] and (now - _state["last_session_activity"]) > SESSION_IDLE_TIMEOUT_S:
            if _state["learning_debt"] > 0:
                logger.warning(f"L3: Idle timeout with debt {_state['learning_debt']}")
            await call_pith_api("/session_end", "POST")
            _state["cached_session_id"] = None
            _state["last_session_activity"] = None
            _state["learning_debt"] = 0
            _state["total_calls_since_session_start"] = 0
            _state["last_conv_turn_args"] = None

        # Auto-end previous active session (D7)
        previous_ended = False
        current = await call_pith_api("/sessions_list?status=active&limit=1")
        if isinstance(current, list):
            if len(current) > 0:
                await call_pith_api("/session_end", "POST")
                previous_ended = True
        elif isinstance(current, dict) and current.get("error"):
            logger.warning("pith_session_start active-session probe failed: %s", current)
        elif current is not None:
            logger.warning(
                "pith_session_start active-session probe returned unexpected type: %s",
                type(current).__name__,
            )

        result = await call_pith_api(
            "/session_start",
            "POST",
            {
                "context_hint": args.get("context_hint", ""),
            },
        )

        if result and not result.get("error"):
            result["previous_session_ended"] = previous_ended
            _state["last_session_activity"] = time.time()
            if result.get("session"):
                _state["cached_session_id"] = result["session"]["session_id"]
            _state["learning_debt"] = 0
            _state["last_learn_timestamp"] = None
            _state["total_calls_since_session_start"] = 0
            _state["last_conv_turn_args"] = None
            _state["auto_session_created"] = False
            _state["pending_bootstrap_orientation"] = None
        return result

    if name == "pith_session_end":
        if _state["learning_debt"] > 0:
            logger.warning(f"L3: Session ending with debt {_state['learning_debt']}")
        try:
            origin_id = _infer_origin_id(args)
        except ValueError as exc:
            return {"error": str(exc)}
        end_payload = {
            "request_id": args.get("request_id"),
            "session_id": args.get("session_id") or _state["cached_session_id"],
            "origin_id": origin_id,
        }
        if args.get("previous_response"):
            end_payload["previous_response"] = args["previous_response"]
            if args.get("previous_message"):
                end_payload["previous_message"] = args["previous_message"]
            if args.get("extracted_concepts_json"):
                end_payload["extracted_concepts_json"] = args["extracted_concepts_json"]
        result = await _perform_durable_write("/session_end", end_payload, request_id_prefix="se")
        _state["last_session_activity"] = None
        _state["cached_session_id"] = None
        _state["learning_debt"] = 0
        _state["total_calls_since_session_start"] = 0
        _state["last_conv_turn_args"] = None
        # C4: Refresh instructions in background (non-blocking)
        asyncio.create_task(_refresh_instructions())
        return result

    if name == "pith_conversation_turn":
        workspace_context = _collect_workspace_context()
        violation = _workspace_protocol_violation(workspace_context)
        if violation is not None:
            return violation
        try:
            origin_id = _infer_origin_id(args)
        except ValueError as exc:
            return {"error": str(exc)}
        surface_id = _normalize_bridge_surface_id(args.get("surface_id")) or BRIDGE_SURFACE_ID
        requested_session_id = args.get("session_id") or None
        explicit_claude_code_origin = surface_id == "claude_code" and bool(origin_id)
        if not requested_session_id and not explicit_claude_code_origin:
            await ensure_session("conversation_turn")
        # Idle timeout check
        now = time.time()
        if (
            not requested_session_id
            and not explicit_claude_code_origin
            and _state["last_session_activity"]
            and (now - _state["last_session_activity"]) > SESSION_IDLE_TIMEOUT_S
        ):
            if _state["learning_debt"] > 0:
                logger.warning(f"L3: Idle timeout (conversation_turn) with debt {_state['learning_debt']}")
            await call_pith_api("/session_end", "POST")
            _state["cached_session_id"] = None
            _state["last_session_activity"] = None
            _state["learning_debt"] = 0
            _state["total_calls_since_session_start"] = 0
            _state["last_conv_turn_args"] = None

        session_id = requested_session_id
        if session_id is None and not explicit_claude_code_origin:
            session_id = _state.get("cached_session_id")
        if session_id is None and explicit_claude_code_origin:
            _transport_event(
                "cached_session_ignored_for_explicit_surface_origin",
                tool_name="pith_conversation_turn",
                cached_session_id=_state.get("cached_session_id"),
                requested_surface_id=surface_id,
                origin_id_present=True,
            )

        ct_payload = {
            "message": args["message"],
            "conversation_context": args.get("conversation_context", ""),
            "session_id": session_id,
            "max_concepts": args.get("max_concepts", 14),  # RAGAS RC-1: validated +4.7pp at 14 vs 10
            "include_predictions": args.get("include_predictions", False),
            "origin_id": origin_id,
            "current_task_id": args.get("current_task_id"),
            "context_authority_mode": args.get("context_authority_mode", "balanced"),
        }
        if surface_id:
            ct_payload["surface_id"] = surface_id
        platform_hint = (args.get("platform_hint") or BRIDGE_PLATFORM_HINT).strip()
        if platform_hint:
            ct_payload["platform_hint"] = platform_hint
        for lifecycle_key in ("workspace_id", "context_delivery_mode", "surface_lifecycle_version"):
            if args.get(lifecycle_key):
                ct_payload[lifecycle_key] = args[lifecycle_key]
        # RUNG0 Component C (A8): stamp this origin's trust-tier. Only sent when non-default
        # so human bridges keep the wire payload unchanged and rely on the server default.
        if BRIDGE_PROVENANCE != "human":
            ct_payload["provenance"] = BRIDGE_PROVENANCE
        inferred_context_telemetry = _infer_codex_context_telemetry(args)
        if "compaction_detected" in args:
            ct_payload["compaction_detected"] = args.get("compaction_detected")
        if "context_pressure" in args:
            ct_payload["context_pressure"] = args.get("context_pressure")
        if "context_telemetry" in args:
            ct_payload["context_telemetry"] = args.get("context_telemetry")
        elif inferred_context_telemetry is not None:
            ct_payload["context_telemetry"] = inferred_context_telemetry
        if workspace_context is not None:
            ct_payload["workspace_context"] = workspace_context
        # S-1: Auto-learn from previous exchange
        if args.get("previous_response"):
            ct_payload["previous_response"] = args["previous_response"]
            # INGEST-050: Auto-fill previous_message from cached state when agent omits it.
            # Closes verbatim ingestion gap — session_learn needs user_message to capture
            # raw text, but previous_message is optional and often omitted by agents.
            prev_msg = args.get("previous_message")
            if not prev_msg and _state.get("last_conv_turn_args"):
                prev_msg = _state["last_conv_turn_args"].get("message", "")
                if prev_msg:
                    logger.info(
                        "INGEST-050: Auto-filled previous_message from cache (%d chars)",
                        len(prev_msg),
                    )
            if prev_msg:
                ct_payload["previous_message"] = prev_msg
            if args.get("extracted_concepts_json"):
                ct_payload["extracted_concepts_json"] = args["extracted_concepts_json"]

        _state["last_conv_turn_args"] = args
        result = await call_pith_api("/conversation_turn", "POST", ct_payload)
        _apply_active_workstream_render_decision(result, args)
        if result and not result.get("error"):
            resolved_session_id = result.get("resolved_session_id")
            if requested_session_id and resolved_session_id == requested_session_id:
                _state["cached_session_id"] = resolved_session_id
            _state["last_session_activity"] = time.time()
        return result

    if name == "pith_session_learn":
        workspace_context = _collect_workspace_context()
        violation = _workspace_protocol_violation(workspace_context)
        if violation is not None:
            return violation
        await ensure_session("session_learn")
        # Idle timeout check
        now = time.time()
        if _state["last_session_activity"] and (now - _state["last_session_activity"]) > SESSION_IDLE_TIMEOUT_S:
            if _state["learning_debt"] > 0:
                logger.warning(f"L3: Idle timeout (session_learn) with debt {_state['learning_debt']}")
            await call_pith_api("/session_end", "POST")
            _state["cached_session_id"] = None
            _state["last_session_activity"] = None
            _state["learning_debt"] = 0
            _state["total_calls_since_session_start"] = 0
            _state["last_conv_turn_args"] = None

        learn_payload = {
            "user_message": args["user_message"],
            "assistant_response": args["assistant_response"],
            "request_id": args.get("request_id") or _make_request_id("sl"),
            "session_id": args.get("session_id"),
            "knowledge_area": args.get("knowledge_area", "conversation"),
            "auto_associate": args.get("auto_associate", True),
        }
        # RUNG0 Component C (A8): same per-origin tier for the direct-learn surface.
        # No-op for Stage 1 (session_learn is not in the loop allowlist) — present so a
        # future allowlist widening cannot silently bypass the cap.
        if BRIDGE_PROVENANCE != "human":
            learn_payload["provenance"] = BRIDGE_PROVENANCE

        # P0.2: Dual-format extracted_concepts
        concepts, source = _parse_extracted_concepts(args)
        if concepts and len(concepts) > 0:
            learn_payload["extracted_concepts"] = concepts
            logger.info(f"[session_learn] {len(concepts)} extracted concepts via {source}")
        else:
            logger.info(f"[session_learn] No extracted concepts. Args keys: {list(args.keys())}")

        result = await _perform_durable_write("/session_learn", learn_payload, request_id_prefix="sl")
        if result and not result.get("error"):
            _state["last_session_activity"] = time.time()
        return result

    if name == "pith_orient":
        params = f"?time_window={args['time_window']}" if args.get("time_window") else ""
        return await call_pith_api(f"/pith_orient{params}")

    if name == "pith_sessions_list":
        parts = []
        if args.get("status"):
            parts.append(f"status={args['status']}")
        if args.get("limit"):
            parts.append(f"limit={args['limit']}")
        if args.get("since"):
            parts.append(f"since={args['since']}")
        qs = "?" + "&".join(parts) if parts else ""
        return await call_pith_api(f"/sessions_list{qs}")

    if name == "pith_auto_associate_batch":
        return await call_pith_api(
            "/auto_associate_batch",
            "POST",
            {
                "dry_run": args.get("dry_run", False),
                "tier1_threshold": args.get("tier1_threshold", 0.12),
                "tier2_enabled": args.get("tier2_enabled", True),
            },
        )

    if name == "pith_validate_response":
        import logging as _logging
        import time as _time

        _pec_logger = _logging.getLogger("pec001.invocations")
        _t0 = _time.perf_counter()
        response_text = args.get("response_text", "")
        constraint_set = args.get("constraint_set", {})
        result = await call_pith_api(
            "/validate_response",
            "POST",
            {"response_text": response_text, "constraint_set": constraint_set},
        )
        _latency_ms = (_time.perf_counter() - _t0) * 1000
        try:
            _payload = result if isinstance(result, dict) else {}
            _pec_logger.info(
                "pec001 invocation",
                extra={
                    "response_len": len(response_text),
                    "constraint_count": len(constraint_set.get("constraints", [])),
                    "passed": _payload.get("passed"),
                    "skipped": _payload.get("skipped", False),
                    "violation_count": len(_payload.get("violations", [])),
                    "latency_ms": round(_latency_ms, 1),
                    "skip_reason": _payload.get("skip_reason"),
                },
            )
        except Exception:
            pass  # PEC-001 Fix 4: Never break the tool call on logging errors
        return result

    if name == "pith_benchmark":
        mode = args.get("mode", "full")
        return await call_pith_api(f"/benchmark?mode={mode}", "POST", {})

    # --- CKO tools ---
    if name == "pith_cko_create":
        return await call_pith_api(
            "/pith/cko",
            "POST",
            {
                "title": args["title"],
                "concept_ids": args["concept_ids"],
                "synthesis": args["synthesis"],
                "knowledge_area": args.get("knowledge_area", "general"),
                "cko_type": args.get("cko_type", "analysis"),
            },
        )

    if name == "pith_cko_get":
        return await call_pith_api(f"/pith/cko/{args['cko_id']}")

    if name == "pith_cko_search":
        parts = []
        if args.get("query_area"):
            parts.append(f"query_area={args['query_area']}")
        if args.get("max_results"):
            parts.append(f"max_results={args['max_results']}")
        qs = "?" + "&".join(parts) if parts else ""
        return await call_pith_api(f"/pith/cko/search{qs}", "POST")

    if name == "pith_cko_update":
        payload = {}
        if args.get("synthesis"):
            payload["synthesis"] = args["synthesis"]
        if args.get("concept_ids"):
            payload["concept_ids"] = args["concept_ids"]
        return await call_pith_api(f"/pith/cko/{args['cko_id']}", "PUT", payload)

    if name == "pith_cko_lifecycle":
        return await call_pith_api("/pith/cko/lifecycle", "POST")

    if name == "pith_cko_list":
        parts = []
        if args.get("status"):
            parts.append(f"status={args['status']}")
        if args.get("knowledge_area"):
            parts.append(f"knowledge_area={args['knowledge_area']}")
        if args.get("limit"):
            parts.append(f"limit={args['limit']}")
        qs = "?" + "&".join(parts) if parts else ""
        return await call_pith_api(f"/pith/cko{qs}")

    # Wave 4: Belief Diff + Epistemic Migration
    if name == "pith_belief_diff":
        payload = {"t1": args["t1"], "t2": args["t2"]}
        if args.get("knowledge_area"):
            payload["knowledge_area"] = args["knowledge_area"]
        return await call_pith_api("/belief_diff", "POST", payload)

    if name == "pith_migrate_epistemic":
        payload = {"dry_run": args.get("dry_run", True)}
        return await call_pith_api("/migrate_epistemic", "POST", payload)

    # Wave 5: Narrative Threads + Cognitive Traces
    if name == "pith_threads":
        return await call_pith_api("/pith_threads", "POST", args)

    if name == "pith_threads_admin":
        return await call_pith_api("/pith_threads", "POST", args)

    if name == "pith_traces":
        return await call_pith_api("/pith_traces", "POST", args)

    # Metrics & observability
    if name == "pith_learning_metrics":
        return await call_pith_api("/learning_metrics")

    if name == "pith_metrics_dashboard":
        qs = f"?since={args['since']}" if args.get("since") else ""
        return await call_pith_api(f"/metrics/dashboard{qs}")

    if name == "pith_metrics_bg_tasks":
        qs = f"?since={args['since']}" if args.get("since") else ""
        return await call_pith_api(f"/metrics/bg_tasks{qs}")

    if name == "pith_metrics_summary":
        qs = f"?days={args['days']}" if args.get("days") else ""
        return await call_pith_api(f"/metrics/summary{qs}")

    if name == "pith_metrics_health_trend":
        qs = f"?days={args['days']}" if args.get("days") else ""
        return await call_pith_api(f"/metrics/health_trend{qs}")

    # Platform operations (local, not API-proxied)
    if name == "pith_deploy_skills":
        from skill_deployer import deploy_skills

        return deploy_skills(
            status_only=args.get("status_only") is True,
            migrate=args.get("migrate") is True,
            dry_run=args.get("dry_run") is True,
            repair=args.get("repair") is True,
            verify_parity=args.get("verify_parity") is True,
        )

    if name == "pith_skill_health":
        from app.features.skill_index import get_skill_health

        return get_skill_health()

    if name == "pith_skill_graduation":
        from concept_to_skill import run_graduation_pipeline

        return run_graduation_pipeline()

    if name == "pith_skill_graduation_status":
        import json

        from app.storage.queries import get_metadata

        raw = get_metadata("skill_graduation_proposals")
        return json.loads(raw) if raw else {"proposals": [], "improvements": []}

    return {"error": True, "code": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}


# --- Background tasks ---
async def _refresh_instructions():
    """Refresh instructions cache after session end (non-blocking)."""
    try:
        new_instructions = await generate_descriptive_instructions()
        if new_instructions:
            logger.info("Instructions refreshed on session_end")
    except Exception as e:
        logger.error(f"Instructions refresh failed: {e}")


# --- MCP Server setup ---
mcp_server = Server("pith")


@mcp_server.list_resources()
async def list_resources() -> list:
    """Return empty resource list — required by Codex rmcp_client capability discovery."""
    return []


@mcp_server.list_resource_templates()
async def list_resource_templates() -> list:
    """Return empty resource template list — required by Codex rmcp_client capability discovery."""
    return []


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all tool definitions."""
    return [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in TOOL_DEFINITIONS
    ]


def _compact_concepts(concepts: Any) -> list[dict[str, Any]]:
    if not isinstance(concepts, list):
        return []
    compacted: list[dict[str, Any]] = []
    for concept in concepts[:3]:
        if not isinstance(concept, dict):
            continue
        compacted.append(
            {
                "concept_id": concept.get("concept_id") or concept.get("id"),
                "summary": str(concept.get("summary") or "")[:500],
            }
        )
    return compacted


def _conversation_turn_identity_envelope(
    result: dict[str, Any],
    *,
    request_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_args = request_args or {}
    return {
        "resolved_session_id": result.get("resolved_session_id"),
        "bind_status": result.get("bind_status"),
        "binding_source": result.get("binding_source"),
        "origin_id": result.get("origin_id") or request_args.get("origin_id"),
        "surface_id": result.get("surface_id") or request_args.get("surface_id") or BRIDGE_SURFACE_ID,
        "platform_hint": result.get("platform_hint") or request_args.get("platform_hint") or BRIDGE_PLATFORM_HINT,
        "workspace_id": result.get("workspace_id") or request_args.get("workspace_id"),
        "context_delivery_mode": result.get("context_delivery_mode") or request_args.get("context_delivery_mode"),
        "surface_lifecycle_version": result.get("surface_lifecycle_version")
        or request_args.get("surface_lifecycle_version"),
    }


def _compact_conversation_turn_result(
    result: dict[str, Any],
    *,
    request_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_args = request_args or {}
    identity = _conversation_turn_identity_envelope(result, request_args=request_args)
    return {
        "response_mode": "compact",
        "full_response_omitted": True,
        **identity,
        "is_first_call": result.get("is_first_call"),
        "is_resumption": result.get("is_resumption"),
        "activation_count": result.get("activation_count"),
        "activated_concepts": _compact_concepts(result.get("activated_concepts")),
        "auto_learned": result.get("auto_learned"),
        "checkpoint_suggested": result.get("checkpoint_suggested"),
        "active_workstream_render": result.get("active_workstream_render"),
        "workstream_activation_gate": result.get("workstream_activation_gate"),
        "_protocol": result.get("_protocol"),
    }


def _render_tool_json(name: str, arguments: dict, result: Any) -> str:
    if name != "pith_conversation_turn" or not isinstance(result, dict):
        return json.dumps(result, indent=2, default=str)
    response_mode = str((arguments or {}).get("response_mode") or "compact").lower()
    if response_mode == "full":
        full_payload = {
            "response_mode": "full",
            "identity": _conversation_turn_identity_envelope(result, request_args=arguments),
            "result": result,
            **result,
        }
        rendered = json.dumps(full_payload, indent=2, default=str)
        if len(rendered) <= MCP_CONVERSATION_TURN_FULL_MAX_CHARS:
            return rendered
        return json.dumps(
            {
                "response_mode": "full",
                "truncated": True,
                "truncated_at_chars": MCP_CONVERSATION_TURN_FULL_MAX_CHARS,
                **_conversation_turn_identity_envelope(result, request_args=arguments),
                "text": rendered[:MCP_CONVERSATION_TURN_FULL_MAX_CHARS],
            },
            indent=2,
            default=str,
        )
    compact = _compact_conversation_turn_result(result, request_args=arguments)
    rendered = json.dumps(compact, indent=2, default=str)
    if len(rendered) <= MCP_CONVERSATION_TURN_COMPACT_MAX_CHARS:
        return rendered
    compact["truncated"] = True
    compact["activated_concepts"] = []
    return json.dumps(compact, indent=2, default=str)


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool call with protocol enforcement and bootstrap injection."""
    started_at = time.perf_counter()
    _state["last_tool_call_time"] = time.time()  # Feed watchdog idle timer
    _transport_event("tool_call", tool_name=name, phase="start", cached_session_id=_state.get("cached_session_id"))
    try:
        result = await _handle_tool(name, arguments)

        # Handle standardized error envelope
        if isinstance(result, dict) and result.get("error") is True:
            _transport_event(
                "tool_call",
                tool_name=name,
                phase="error",
                code=result.get("code", "UNKNOWN"),
                error=result.get("message", ""),
                cached_session_id=_state.get("cached_session_id"),
            )
            parts = [f"Error [{result.get('code', 'UNKNOWN')}]: {result.get('message', '')}"]
            if result.get("details"):
                parts.append(result["details"])
            if result.get("hint"):
                parts.append(f"Hint: {result['hint']}")
            _emit_active_workstream_render_event(
                name,
                arguments,
                result,
                content_blocks=1,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                api_status=result.get("status_code") or result.get("code"),
            )
            return [TextContent(type="text", text="\n".join(parts))]

        # --- L3: Protocol enforcement ---
        is_conv_turn = name == "pith_conversation_turn"
        auto_learned = is_conv_turn and isinstance(result, dict) and result.get("auto_learned")
        client_attempted_learn = (
            is_conv_turn
            and _state["last_conv_turn_args"]
            and _state["last_conv_turn_args"].get("extracted_concepts_json")
            and _state["last_conv_turn_args"].get("extracted_concepts_json") != "[]"
        )

        if auto_learned:
            _state["learning_debt"] = 0
            _state["last_learn_timestamp"] = time.time()
            _state["total_calls_since_session_start"] += 1
        elif client_attempted_learn:
            _state["learning_debt"] = 0
            _state["total_calls_since_session_start"] += 1
        elif name in SUBSTANTIVE_TOOLS:
            _state["learning_debt"] += 1
            _state["total_calls_since_session_start"] += 1
        elif name in LEARNING_TOOLS:
            _state["learning_debt"] = 0
            _state["last_learn_timestamp"] = time.time()
            _state["total_calls_since_session_start"] += 1
        elif name not in META_TOOLS and name not in SESSION_TOOLS:
            _state["total_calls_since_session_start"] += 1

        # Inject protocol status
        try:
            protocol_status = _get_protocol_status(name)
            if isinstance(result, dict) and not isinstance(result, list):
                result["_protocol"] = protocol_status
                logger.debug(
                    f"L3: Protocol injected for {name}: "
                    f"debt={protocol_status['learning_debt']}, "
                    f"urgency={protocol_status.get('urgency', 'ok')}"
                )
        except Exception as e:
            logger.error(f"L3: Protocol injection failed: {e}")

        # Build content blocks
        content_blocks = []
        active_workstream_block = _active_workstream_rendered_block_for_response(name, result)
        if active_workstream_block:
            content_blocks.append(TextContent(type="text", text=active_workstream_block))
        content_blocks.append(TextContent(type="text", text=_render_tool_json(name, arguments, result)))

        # L4: Cognitive bootstrap injection (one-shot)
        if _state["pending_bootstrap_orientation"] and name != "pith_session_start":
            content_blocks.append(TextContent(type="text", text=_state["pending_bootstrap_orientation"]))
            logger.info(f"L4: Bootstrap orientation injected into {name} response")
            _state["pending_bootstrap_orientation"] = None

        _emit_active_workstream_render_event(
            name,
            arguments,
            result,
            content_blocks=len(content_blocks),
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            api_status="ok",
        )
        _transport_event(
            "tool_call", tool_name=name, phase="success", cached_session_id=_state.get("cached_session_id")
        )
        return content_blocks

    except Exception as e:
        logger.error(f"Tool call error for {name}: {e}", exc_info=True)
        _emit_active_workstream_render_event(
            name,
            arguments,
            {"error": True, "code": "WRAPPER_EXCEPTION", "message": str(e)},
            content_blocks=1,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            api_status=type(e).__name__,
        )
        _transport_event(
            "tool_call", tool_name=name, phase="error", error=str(e), cached_session_id=_state.get("cached_session_id")
        )
        return [TextContent(type="text", text=f"❌ Error: {e}")]


# --- Process lifecycle management ---
# Prevents zombie bridge accumulation. See BRIDGE_LIFECYCLE_IMPL_SPEC.md.


def _bridge_status() -> dict[str, Any]:
    """Return local bridge diagnostics without calling the backend."""
    runtime_path = Path(__file__).resolve().parent
    status: dict[str, Any] = {
        "current_pid": os.getpid(),
        "current_ppid": os.getppid(),
        "api_url": PITH_API_URL,
        "host_transport_open": bool(_state.get("host_transport_open")),
        "deferred_signal_counts": dict(_state.get("deferred_signal_counts") or {}),
        "deferred_signal_total": int(_state.get("deferred_signal_total") or 0),
        "last_deferred_signal": _state.get("last_deferred_signal"),
        "connected_max_age_exit_enabled": bool(CONNECTED_MAX_AGE_FORCE_EXIT),
        "runtime_path": str(runtime_path),
        "transport_log_path": TRANSPORT_LOG_PATH,
        "transport_state_path": TRANSPORT_STATE_PATH,
        "heartbeat_dir": HEARTBEAT_DIR,
        "last_transport_event": None,
        "heartbeats": [],
        "runtime_git_commit": None,
        "runtime_git_status_short": None,
    }

    try:
        completed = subprocess.run(
            ["git", "-C", str(runtime_path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode == 0:
            status["runtime_git_commit"] = completed.stdout.strip()
    except Exception as exc:
        status["runtime_git_error"] = str(exc)

    try:
        completed = subprocess.run(
            ["git", "-C", str(runtime_path), "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode == 0:
            status["runtime_git_status_short"] = completed.stdout.splitlines()
    except Exception as exc:
        status["runtime_git_status_error"] = str(exc)

    try:
        with open(TRANSPORT_STATE_PATH) as f:
            state = json.load(f)
        status["last_transport_event"] = state.get("last_event")
        status["transport_state"] = state
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        status["transport_state_error"] = str(exc)

    try:
        now = time.time()
        for path in sorted(Path(HEARTBEAT_DIR).glob("bridge.*.heartbeat")):
            try:
                with path.open() as f:
                    heartbeat = json.load(f)
                last_call = heartbeat.get("last_tool_call")
                if last_call is None:
                    last_call = heartbeat.get("start_time")
                if last_call is None:
                    last_call = now
                heartbeat["path"] = str(path)
                heartbeat["age_s"] = max(0.0, now - float(last_call))
                status["heartbeats"].append(heartbeat)
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
                status["heartbeats"].append({"path": str(path), "error": str(exc)})
    except OSError as exc:
        status["heartbeat_error"] = str(exc)

    return status


def _parse_etime(etime: str) -> float | None:
    """Parse ps elapsed time format into seconds.

    Formats: '10:30' (MM:SS), '1:10:30' (HH:MM:SS), '2-01:10:30' (D-HH:MM:SS)
    """
    if not etime:
        return None
    try:
        parts = etime.strip().replace("-", ":").split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 4:
            return parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3]
    except (ValueError, IndexError):
        pass
    return None


def _heartbeat_freshness_s(hb: dict[str, Any], fpath: str, now: float) -> float | None:
    heartbeat_at = hb.get("heartbeat_at")
    if heartbeat_at is not None:
        try:
            return now - float(heartbeat_at)
        except (TypeError, ValueError):
            pass
    try:
        return now - os.path.getmtime(fpath)
    except OSError:
        return None


def _heartbeat_parent_attached(hb: dict[str, Any]) -> bool:
    ppid = hb.get("ppid")
    original_ppid = hb.get("original_ppid")
    return bool(ppid and original_ppid and ppid == original_ppid and ppid != 1)


def _host_parent_attached(current_ppid: int | None = None, original_ppid: int | None = None) -> bool:
    """Return whether this bridge is still attached to the original host process."""
    if current_ppid is None:
        current_ppid = os.getppid()
    if original_ppid is None:
        original_ppid = _state.get("original_ppid")
    return bool(current_ppid and original_ppid and current_ppid == original_ppid and current_ppid != 1)


def _should_defer_host_signal(
    sig_name: str,
    *,
    state: dict[str, Any] | None = None,
    current_ppid: int | None = None,
) -> bool:
    """Return whether a host signal should be treated as advisory while attached."""
    if not HOST_SIGNAL_DEFERRAL_ENABLED:
        return False
    if sig_name not in {"SIGTERM", "SIGHUP"}:
        return False
    state = state or _state
    if state.get("shutdown_initiated"):
        return False
    if not state.get("host_transport_open"):
        return False
    return _host_parent_attached(current_ppid=current_ppid, original_ppid=state.get("original_ppid"))


def _record_deferred_host_signal(sig_name: str) -> None:
    """Persist a benign host signal without closing the MCP transport."""
    now = _transport_iso_now()
    counts = dict(_state.get("deferred_signal_counts") or {})
    counts[sig_name] = int(counts.get(sig_name) or 0) + 1
    _state["deferred_signal_counts"] = counts
    _state["deferred_signal_total"] = int(_state.get("deferred_signal_total") or 0) + 1
    _state["last_deferred_signal"] = {
        "signal": sig_name,
        "observed_at": now,
        "ppid": os.getppid(),
        "reason": "host_attached_transport_open",
    }
    logger.info("LIFECYCLE: Deferred %s while host transport remains attached", sig_name)
    _transport_event("host_signal_deferred", signal=sig_name, deferred_signal_total=_state["deferred_signal_total"])
    _update_bridge_health(
        host_transport_open=bool(_state.get("host_transport_open")),
        deferred_signal_counts=counts,
        deferred_signal_total=_state["deferred_signal_total"],
        last_deferred_signal=_state["last_deferred_signal"],
    )


def _reap_stale_bridges():
    """On startup, kill bridge processes that have been idle too long.

    Reads heartbeat files written by other bridge instances.
    Safe for multi-session: skips fresh, apparently attached sibling heartbeats.
    """
    os.makedirs(HEARTBEAT_DIR, exist_ok=True)
    my_pid = os.getpid()
    now = time.time()
    reaped = 0

    # Phase 1: Heartbeat-based reaping (new-code bridges)
    for fname in os.listdir(HEARTBEAT_DIR):
        if not fname.startswith("bridge.") or not fname.endswith(".heartbeat"):
            continue
        fpath = os.path.join(HEARTBEAT_DIR, fname)
        try:
            with open(fpath) as f:
                hb = json.load(f)
            pid = hb.get("pid")
            if not pid or pid == my_pid:
                continue

            # Is the process even alive?
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                os.unlink(fpath)
                continue
            except PermissionError:
                continue

            # Check idle time from heartbeat
            last_call = hb.get("last_tool_call", hb.get("start_time", 0))
            idle_s = now - last_call
            heartbeat_age_s = _heartbeat_freshness_s(hb, fpath, now)
            fresh_attached = (
                heartbeat_age_s is not None
                and heartbeat_age_s <= WATCHDOG_INTERVAL_S * 3
                and _heartbeat_parent_attached(hb)
            )
            if fresh_attached:
                continue
            if idle_s > REAP_STALE_IDLE_S:
                logger.info(
                    f"REAPER: Killing stale bridge PID {pid} "
                    f"(idle {idle_s:.0f}s > {REAP_STALE_IDLE_S}s; heartbeat_age={heartbeat_age_s})"
                )
                try:
                    os.kill(pid, signal.SIGTERM)
                    reaped += 1
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    os.unlink(fpath)
                except FileNotFoundError:
                    pass
        except (json.JSONDecodeError, KeyError, OSError):
            try:
                os.unlink(fpath)
            except (FileNotFoundError, OSError):
                pass

    # Phase 2: Transition reaper for old-code bridges (no heartbeat files)
    try:
        result = subprocess.run(["pgrep", "-f", "pith_mcp\\.py"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid == my_pid:
                continue
            hb_path = os.path.join(HEARTBEAT_DIR, f"bridge.{pid}.heartbeat")
            if os.path.exists(hb_path):
                continue
            try:
                ps_result = subprocess.run(
                    ["ps", "-o", "etime=", "-p", str(pid)], capture_output=True, text=True, timeout=3
                )
                etime = ps_result.stdout.strip()
                age_s = _parse_etime(etime)
                if age_s and age_s > LEGACY_REAP_AGE_S:
                    logger.info(f"REAPER: Killing legacy bridge PID {pid} (no heartbeat, age {age_s:.0f}s)")
                    os.kill(pid, signal.SIGTERM)
                    reaped += 1
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError, ValueError):
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if reaped:
        logger.info(f"REAPER: Total reaped on startup: {reaped}")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop):
    """Install OS signal handlers for graceful shutdown."""

    def _signal_handler(sig_name: str):
        logger.info(f"LIFECYCLE: Received {sig_name}")
        _transport_event("host_signal_received", signal=sig_name, host_transport_open=_state.get("host_transport_open"))
        if _should_defer_host_signal(sig_name):
            _record_deferred_host_signal(sig_name)
            return
        asyncio.ensure_future(_graceful_shutdown(f"signal_{sig_name}"))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig.name)
        except (ValueError, OSError) as e:
            logger.warning(f"LIFECYCLE: Could not install handler for {sig.name}: {e}")


async def _graceful_shutdown(reason: str, *, force_exit: bool = True):
    """Clean exit: end Pith session, remove heartbeat, log, exit.

    Uses os._exit(0) to avoid blocking on cleanup in broken-pipe scenarios.
    """
    if _state["shutdown_initiated"]:
        return
    _state["shutdown_initiated"] = True

    logger.info(f"LIFECYCLE: Graceful shutdown initiated. Reason: {reason}")

    session_end_result = _shutdown_profile(reason)["default_result"]
    _update_bridge_health(
        transport_state="closing",
        last_shutdown_reason=reason,
    )

    profile = _shutdown_profile(reason)
    if profile["attempt_session_end"]:
        try:
            result = await asyncio.wait_for(call_pith_api("/session_end", "POST"), timeout=3.0)
            if isinstance(result, dict) and result.get("error"):
                session_end_result = "attempted_failed"
                logger.warning(
                    "LIFECYCLE: session_end returned error during shutdown: %s",
                    result.get("message", result),
                )
            else:
                session_end_result = "attempted_ok"
                logger.info("LIFECYCLE: Pith session ended cleanly.")
        except Exception as e:
            session_end_result = "attempted_failed"
            logger.warning(f"LIFECYCLE: Session end failed during shutdown: {e}")
    else:
        session_end_result = profile["default_result"]

    heartbeat_path = os.path.join(HEARTBEAT_DIR, f"bridge.{os.getpid()}.heartbeat")
    try:
        os.unlink(heartbeat_path)
    except (FileNotFoundError, OSError):
        pass

    _transport_event("bridge_stop", reason=reason, session_end_result=session_end_result)
    _update_bridge_health(
        transport_state="closed",
        last_shutdown_reason=reason,
        last_session_end_result=session_end_result,
    )
    logger.info(f"LIFECYCLE: Exiting. PID={os.getpid()} Reason={reason}")
    if force_exit:
        _force_exit(0)


async def _watchdog():
    """Background task: self-terminate on orphaning, optional idle, or max-age.

    This is the PRIMARY defense against zombie bridge accumulation.
    """
    start_time = _state["bridge_start_time"]
    original_ppid = _state["original_ppid"]
    os.makedirs(HEARTBEAT_DIR, exist_ok=True)
    heartbeat_path = os.path.join(HEARTBEAT_DIR, f"bridge.{os.getpid()}.heartbeat")

    try:
        while not _state["shutdown_initiated"]:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            now = time.time()
            last_call = _state.get("last_tool_call_time") or start_time
            idle_s = now - last_call
            age_s = now - start_time

            # Write heartbeat for external visibility
            try:
                with open(heartbeat_path, "w") as f:
                    json.dump(
                        {
                            "pid": os.getpid(),
                            "ppid": os.getppid(),
                            "original_ppid": original_ppid,
                            "start_time": start_time,
                            "last_tool_call": last_call,
                            "heartbeat_at": now,
                            "idle_s": round(idle_s),
                            "age_s": round(age_s),
                            "host_transport_open": bool(_state.get("host_transport_open")),
                            "deferred_signal_counts": dict(_state.get("deferred_signal_counts") or {}),
                            "deferred_signal_total": int(_state.get("deferred_signal_total") or 0),
                            "last_deferred_signal": _state.get("last_deferred_signal"),
                            "connected_max_age_exit_enabled": bool(CONNECTED_MAX_AGE_FORCE_EXIT),
                        },
                        f,
                    )
            except OSError:
                pass

            # Check 1 (P0): Reparented to init (orphaned)
            current_ppid = os.getppid()
            if current_ppid == 1 or (original_ppid and current_ppid != original_ppid):
                logger.warning(f"LIFECYCLE: Parent changed ({original_ppid} -> {current_ppid}). Orphaned. Exiting.")
                await _graceful_shutdown("orphaned")
                return

            # Check 2 (P1): Optional connected idle timeout
            if CONNECTED_IDLE_TIMEOUT_S and idle_s > CONNECTED_IDLE_TIMEOUT_S:
                logger.info(
                    f"LIFECYCLE: Connected idle timeout ({idle_s:.0f}s > {CONNECTED_IDLE_TIMEOUT_S}s). Exiting."
                )
                await _graceful_shutdown("connected_idle_timeout")
                return

            # Check 3 (P1): Bounded connected max age
            if age_s > CONNECTED_MAX_AGE_S:
                if CONNECTED_MAX_AGE_FORCE_EXIT:
                    logger.info(
                        f"LIFECYCLE: Connected max age reached ({age_s:.0f}s > {CONNECTED_MAX_AGE_S}s). Exiting."
                    )
                    await _graceful_shutdown("connected_max_age")
                    return
                if not _state.get("connected_max_age_reported"):
                    _state["connected_max_age_reported"] = True
                    logger.info(
                        "LIFECYCLE: Connected max age reached "
                        f"({age_s:.0f}s > {CONNECTED_MAX_AGE_S}s); attached bridge remains resident."
                    )
                    _transport_event(
                        "connected_max_age_reached_attached",
                        age_s=round(age_s),
                        connected_max_age_s=CONNECTED_MAX_AGE_S,
                    )
                    _update_bridge_health(
                        connected_max_age_reached_at=_transport_iso_now(),
                        connected_max_age_exit_enabled=False,
                    )

            # Check 4 (P1): Stdin closed
            try:
                if sys.stdin.closed:
                    logger.warning("LIFECYCLE: stdin closed. Exiting.")
                    await _graceful_shutdown("stdin_closed")
                    return
            except Exception:
                pass

    except asyncio.CancelledError:
        pass
    finally:
        try:
            os.unlink(heartbeat_path)
        except FileNotFoundError:
            pass


# --- Entry point ---
async def main():
    """Start the MCP server on stdio transport with lifecycle management."""

    # Phase 0: Reap stale bridges from previous runs
    _reap_stale_bridges()

    # Phase 1: Initialize lifecycle state
    _state["bridge_start_time"] = time.time()
    _state["last_tool_call_time"] = time.time()  # Grace period from startup
    _state["original_ppid"] = os.getppid()
    _state["host_transport_open"] = False
    _state["deferred_signal_counts"] = {}
    _state["deferred_signal_total"] = 0
    _state["last_deferred_signal"] = None
    _state["connected_max_age_reported"] = False

    # Phase 2: C4 — Generate instructions before connecting
    instructions = await generate_descriptive_instructions()
    if instructions:
        logger.info(f"Instructions ready: {len(instructions)} chars")

    # BRIDGE-003: Check for prior bridge state (continuity across restarts)
    prior_session_id = None
    overlap_metadata = None
    try:
        with open(TRANSPORT_STATE_PATH) as f:
            prior_state = json.load(f)
        prior_pid = prior_state.get("pid")
        prior_session_id = prior_state.get("cached_session_id")
        # Check if prior bridge is actually dead
        if prior_pid and prior_pid != os.getpid():
            try:
                os.kill(prior_pid, 0)
                logger.info(f"BRIDGE-003: Prior bridge PID {prior_pid} still alive — not resuming")
                overlap_metadata = {
                    "prior_pid": prior_pid,
                    "prior_session_id": prior_session_id,
                    "observed_at": _transport_iso_now(),
                }
                prior_session_id = None
            except ProcessLookupError:
                logger.info(
                    f"BRIDGE-003: Prior bridge PID {prior_pid} is dead. "
                    f"Last session: {prior_session_id}, last tool: {prior_state.get('last_tool_name')}"
                )
            except PermissionError:
                # BRIDGE-003b: PID exists but we cannot signal it (EPERM).
                # Happens when launchd recycles the PID to a process owned by
                # another uid/session. Treat as non-resumable rather than crashing
                # the bridge on startup. Prior art: _reap_stale_bridges() at the
                # PermissionError branch in the same file handles the equivalent
                # case with `continue`.
                logger.info(
                    f"BRIDGE-003: Prior bridge PID {prior_pid} is not signalable from "
                    f"this context (EPERM); treating as non-resumable. "
                    f"Last session: {prior_session_id}"
                )
                prior_session_id = None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    logger.info(f"Pith MCP bridge starting (Python). PID={os.getpid()} PPID={os.getppid()} API={PITH_API_URL}")
    _update_bridge_health(transport_state="starting")
    if overlap_metadata is not None:
        _transport_event(
            "bridge_overlap_detected",
            prior_pid=overlap_metadata["prior_pid"],
            prior_session_id=overlap_metadata["prior_session_id"],
            observed_at=overlap_metadata["observed_at"],
        )
        _update_bridge_health(
            overlap_detected=True,
            last_overlap=overlap_metadata,
        )
    _transport_event("bridge_start", prior_session_id=prior_session_id)

    # Phase 3: Validate auth
    try:
        ensure_safe_installed_runtime(invocation_path=sys.argv[0])
    except RuntimeInstallGuardError as exc:
        logger.critical(f"Startup runtime-path validation failed: {exc}")
        raise

    try:
        ensure_safe_installed_runtime(invocation_path=sys.argv[0])
    except RuntimeInstallGuardError as exc:
        logger.critical(f"Startup runtime-path validation failed: {exc}")
        raise

    try:
        _validate_startup_auth()
    except RuntimeError as exc:
        logger.critical(f"Startup auth validation failed: {exc}")
        raise
    _update_bridge_health(
        transport_state="open",
        host_transport_open=False,
        deferred_signal_counts={},
        deferred_signal_total=0,
        last_deferred_signal=None,
        connected_max_age_exit_enabled=bool(CONNECTED_MAX_AGE_FORCE_EXIT),
    )

    # Phase 4: Install signal handlers
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    # Phase 5: Start watchdog background task
    asyncio.ensure_future(_watchdog())

    _schedule_bridge_outbox_drain()
    # Phase 6: Run MCP server on stdio
    shutdown_reason = "stdio_teardown"
    try:
        async with stdio_server() as (read_stream, write_stream):
            _state["host_transport_open"] = True
            _update_bridge_health(host_transport_open=True)
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
    except Exception:
        shutdown_reason = "stdio_run_error"
        raise
    finally:
        _state["host_transport_open"] = False
        _update_bridge_health(host_transport_open=False)
        if not _state["shutdown_initiated"]:
            await _graceful_shutdown(shutdown_reason, force_exit=False)


if __name__ == "__main__":
    asyncio.run(main())
