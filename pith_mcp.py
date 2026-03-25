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
import sys
import time
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# --- Configuration ---
PITH_API_URL = os.getenv("PITH_API_URL") if os.getenv("PITH_API_URL") is not None else os.getenv("BRAIN_API_URL", "http://localhost:8000")
PITH_API_KEY = os.getenv("PITH_API_KEY") or os.getenv("BRAIN_API_KEY", "")

# --- Deprecation warnings for legacy env vars ---
if os.getenv("BRAIN_API_URL") and not os.getenv("PITH_API_URL"):
    import sys; print("DEPRECATED: BRAIN_API_URL env var. Rename to PITH_API_URL.", file=sys.stderr)
if os.getenv("BRAIN_API_KEY") and not os.getenv("PITH_API_KEY"):
    import sys; print("DEPRECATED: BRAIN_API_KEY env var. Rename to PITH_API_KEY.", file=sys.stderr)

# --- Logging (stderr only — stdout is MCP transport) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("pith_mcp")

# --- C4: Static fallback instructions ---
STATIC_FALLBACK_INSTRUCTIONS = """⚠️ MANDATORY FIRST ACTION — NON-NEGOTIABLE:
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
[{"summary": "30-500 chars", "confidence": 0.6, "knowledge_area": "domain", "evidence": ["source >=10 chars"], "concept_type": "decision"}]
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

EXTRACTION EXAMPLES — L1 vs L3+ (what to extract from your own responses):
BAD (L1 only): {summary:'We fixed the validation bug by changing line 222', concept_type:'observation'}
GOOD (L3): {summary:'PRINCIPLE: When changing a validation limit, grep the entire codebase for all enforcement points — there is never just one gate', concept_type:'principle', evidence:['verified: second hardcoded check found at line 222']}
BAD (L1): {summary:'The budget warning field was missing from the response', concept_type:'observation'}
GOOD (L3): {summary:'HEURISTIC: Diagnostic signals created inside internal functions are silent failures unless traced through every calling layer to the end user', concept_type:'heuristic', evidence:['verified: budget_warnings lost between session_learn and conversation_turn']}
The pattern: L1 captures WHAT happened. L3+ captures the REUSABLE LESSON a future session could apply to a different problem.
GOOD (factual/L1): {summary:'VACUUM cannot run inside a transaction — pith storage uses isolation_level=None (autocommit) so VACUUM is safe to call from maintenance functions', concept_type:'observation', evidence:['verified: storage_backend.py line 259 isolation_level=None']}
GOOD (factual/L1): {summary:'phase5_7_incremental_vacuum always skips when auto_vacuum=0 — freelist pages are NOT reclaimed unless auto_vacuum=2', concept_type:'observation', evidence:['verified: maintenance.py line 1264 checks auto_vacuum_mode != 2']}
Include factual/L1 concepts when your response contains specific verified facts about system behavior, thresholds, or configuration values.

Pith gets smarter with every conversation. Your job is to feed it quality knowledge."""


# --- Client-side state (C1, L3, L4) ---
SESSION_IDLE_TIMEOUT_S = 2 * 60 * 60  # 2 hours
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
}


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


async def call_pith_api(endpoint: str, method: str = "GET", body: dict | None = None) -> dict[str, Any]:
    """Call the Pith REST API. Returns parsed JSON or error dict.

    Retries once on broken pipe / connection reset (server restart scenario).
    """
    max_retries = 2
    for attempt in range(max_retries):
        client = _get_client()
        try:
            if method == "GET":
                resp = await client.get(endpoint)
            else:
                resp = await client.post(endpoint, json=body)

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
                return {
                    "error": True,
                    "code": code,
                    "message": f"HTTP {resp.status_code}: {resp.reason_phrase}",
                    "details": error_text,
                    "tool": endpoint,
                }
            return resp.json()
        except httpx.ConnectError:
            if attempt < max_retries - 1:
                logger.warning(f"Connection refused on {endpoint} (attempt {attempt + 1}), retrying with fresh client...")
                _reset_client()
                await asyncio.sleep(1)
                continue
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
            return {
                "error": True,
                "code": "CONNECTION_RESET",
                "message": f"Server connection lost: {e}",
                "hint": "Server may have restarted. This should auto-recover on next call.",
                "tool": endpoint,
            }
        except Exception as e:
            return {
                "error": True,
                "code": "SERVER_ERROR",
                "message": str(e),
                "tool": endpoint,
            }
    # Unreachable, but defensive
    return {"error": True, "code": "RETRY_EXHAUSTED", "message": "All retries failed", "tool": endpoint}


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
    from app.datetime_utils import _utc_now
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
                logger.info(f"C1.1: New process detected stale session {sessions[0].get('session_id', sessions[0].get('id'))}. Ending.")
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
            call_pith_api("/pith_stats"),
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
                f"Pith is growing ({tc} concepts). "
                "Try asking 'what do you know about me?' to see what it's learned."
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


# --- Tool Definitions (all 36, identical schemas to server.js) ---
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
                "previous_response": {"type": "string", "description": "Your last response to the user"},
                "previous_message": {"type": "string", "description": "The user's last message"},
                "extracted_concepts_json": {"type": "string", "description": "Concepts extracted from final response"},
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
                "previous_response": {"type": "string", "description": "Your previous response to the user"},
                "previous_message": {"type": "string", "description": "The user's previous message"},
                "extracted_concepts_json": {
                    "type": "string",
                    "description": "JSON string of 1-7 concept objects from your PREVIOUS response",
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
        "description": "Run CogGov-Bench — behavioral governance benchmark. Measures 6 dimensions. Use 'light' for dims 1-3, 'full' for all 6 + adversarial.",
        "inputSchema": {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["light", "full"], "description": "Benchmark mode"}},
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
                "dry_run": {"type": "boolean", "description": "If true (default), report what WOULD change without changing it", "default": True},
            },
        },
    },
    # Wave 5: Narrative Threads + Cognitive Traces
    {
        "name": "pith_threads",
        "description": "Manage narrative threads — ongoing work streams, projects, and topics. Actions: create, get, list, update, close, reactivate, link, unlink, similar, stats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "get", "list", "update", "close", "reactivate", "link", "unlink", "similar", "stats"], "description": "Action to perform (default: list)"},
                "thread_id": {"type": "string", "description": "Thread ID (required for get/update/close/reactivate/link/unlink)"},
                "title": {"type": "string", "description": "Thread title (required for create, optional for update)"},
                "description": {"type": "string", "description": "Thread description (optional)"},
                "urgency": {"type": "string", "enum": ["low", "normal", "high"], "description": "Thread urgency tier (default: normal)"},
                "concept_id": {"type": "string", "description": "Concept ID (required for link/unlink)"},
                "role": {"type": "string", "enum": ["initiator", "member", "evidence", "blocker", "conclusion"], "description": "Concept role in thread (default: member)"},
                "status": {"type": "string", "description": "Filter by status for list action"},
                "situation": {"type": "string", "description": "Situation description for similar action"},
                "intent": {"type": "string", "description": "Intent description for similar action (optional)"},
                "limit": {"type": "number", "description": "Max results for similar action (default: 5)"},
                "goal_ids": {"type": "array", "items": {"type": "string"}, "description": "Goal IDs to associate (optional)"},
                "knowledge_areas": {"type": "array", "items": {"type": "string"}, "description": "Knowledge areas to associate (optional)"},
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
                "action": {"type": "string", "enum": ["get", "list", "search"], "description": "Action to perform (default: list)"},
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
        "description": "Deploy skills from ~/.claude/skills/ to all platforms (Claude Code, Cursor, Codex, Cowork). Re-deploy after adding skills or starting a new Cowork session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status_only": {"type": "boolean", "description": "If true, return status without re-deploying (default: false)"},
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
        return await call_pith_api("/pith_stats")

    if name == "pith_health":
        return await call_pith_api("/pith_health")

    if name == "pith_projection":
        return await call_pith_api("/memory_projection")

    if name == "pith_reflect":
        mode = args.get("mode", "incremental")
        params = f"mode={mode}"
        if args.get("verbose") is not None:
            params += f"&verbose={str(args['verbose']).lower()}"
        return await call_pith_api(f"/pith_reflect?{params}", "POST")

    if name == "pith_checkpoint":
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
            "ttl_days": args.get("ttl_days"),
            "max_age_hours": args.get("max_age_hours"),
        }
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
        if current and not current.get("error") and isinstance(current, list) and len(current) > 0:
            await call_pith_api("/session_end", "POST")
            previous_ended = True

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
        end_payload = {}
        if args.get("previous_response"):
            end_payload["previous_response"] = args["previous_response"]
            if args.get("previous_message"):
                end_payload["previous_message"] = args["previous_message"]
            if args.get("extracted_concepts_json"):
                end_payload["extracted_concepts_json"] = args["extracted_concepts_json"]
        result = await call_pith_api(
            "/session_end",
            "POST",
            end_payload if end_payload else None,
        )
        _state["last_session_activity"] = None
        _state["cached_session_id"] = None
        _state["learning_debt"] = 0
        _state["total_calls_since_session_start"] = 0
        _state["last_conv_turn_args"] = None
        # C4: Refresh instructions in background (non-blocking)
        asyncio.create_task(_refresh_instructions())
        return result

    if name == "pith_conversation_turn":
        await ensure_session("conversation_turn")
        # Idle timeout check
        now = time.time()
        if _state["last_session_activity"] and (now - _state["last_session_activity"]) > SESSION_IDLE_TIMEOUT_S:
            if _state["learning_debt"] > 0:
                logger.warning(f"L3: Idle timeout (conversation_turn) with debt {_state['learning_debt']}")
            await call_pith_api("/session_end", "POST")
            _state["cached_session_id"] = None
            _state["last_session_activity"] = None
            _state["learning_debt"] = 0
            _state["total_calls_since_session_start"] = 0
            _state["last_conv_turn_args"] = None

        ct_payload = {
            "message": args["message"],
            "conversation_context": args.get("conversation_context", ""),
            "session_id": args.get("session_id"),
            "max_concepts": args.get("max_concepts", 14),  # RAGAS RC-1: validated +4.7pp at 14 vs 10
            "include_predictions": args.get("include_predictions", False),
        }
        # S-1: Auto-learn from previous exchange
        if args.get("previous_response"):
            ct_payload["previous_response"] = args["previous_response"]
            if args.get("previous_message"):
                ct_payload["previous_message"] = args["previous_message"]
            if args.get("extracted_concepts_json"):
                ct_payload["extracted_concepts_json"] = args["extracted_concepts_json"]

        _state["last_conv_turn_args"] = args
        result = await call_pith_api("/conversation_turn", "POST", ct_payload)
        if result and not result.get("error"):
            _state["last_session_activity"] = time.time()
        return result

    if name == "pith_session_learn":
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
            "session_id": args.get("session_id"),
            "knowledge_area": args.get("knowledge_area", "conversation"),
            "auto_associate": args.get("auto_associate", True),
        }

        # P0.2: Dual-format extracted_concepts
        concepts, source = _parse_extracted_concepts(args)
        if concepts and len(concepts) > 0:
            learn_payload["extracted_concepts"] = concepts
            logger.info(f"[session_learn] {len(concepts)} extracted concepts via {source}")
        else:
            logger.info(f"[session_learn] No extracted concepts. Args keys: {list(args.keys())}")

        result = await call_pith_api("/session_learn", "POST", learn_payload)
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
        return await call_pith_api(
            "/validate_response",
            "POST",
            {
                "response_text": args["response_text"],
                "constraint_set": args["constraint_set"],
            },
        )

    if name == "pith_benchmark":
        mode = args.get("mode", "full")
        return await call_pith_api(f"/benchmark?mode={mode}", "POST", {})

    # --- CKO tools ---
    if name == "pith_cko_create":
        return await call_pith_api(
            "/cko/create",
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
        return await call_pith_api(f"/cko/{args['cko_id']}")

    if name == "pith_cko_search":
        parts = []
        if args.get("query_area"):
            parts.append(f"query_area={args['query_area']}")
        if args.get("max_results"):
            parts.append(f"max_results={args['max_results']}")
        qs = "?" + "&".join(parts) if parts else ""
        return await call_pith_api(f"/cko/search{qs}", "POST")

    if name == "pith_cko_update":
        payload = {}
        if args.get("synthesis"):
            payload["synthesis"] = args["synthesis"]
        if args.get("concept_ids"):
            payload["concept_ids"] = args["concept_ids"]
        return await call_pith_api(f"/cko/{args['cko_id']}", "PUT", payload)

    if name == "pith_cko_lifecycle":
        return await call_pith_api("/cko/lifecycle", "POST")

    if name == "pith_cko_list":
        parts = []
        if args.get("status"):
            parts.append(f"status={args['status']}")
        if args.get("knowledge_area"):
            parts.append(f"knowledge_area={args['knowledge_area']}")
        if args.get("limit"):
            parts.append(f"limit={args['limit']}")
        qs = "?" + "&".join(parts) if parts else ""
        return await call_pith_api(f"/cko{qs}")

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
        return deploy_skills(status_only=bool(args.get("status_only", False)))

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


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all 36 tool definitions."""
    return [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in TOOL_DEFINITIONS
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool call with protocol enforcement and bootstrap injection."""
    try:
        result = await _handle_tool(name, arguments)

        # Handle standardized error envelope
        if isinstance(result, dict) and result.get("error") is True:
            parts = [f"Error [{result.get('code', 'UNKNOWN')}]: {result.get('message', '')}"]
            if result.get("details"):
                parts.append(result["details"])
            if result.get("hint"):
                parts.append(f"Hint: {result['hint']}")
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
        content_blocks = [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        # L4: Cognitive bootstrap injection (one-shot)
        if _state["pending_bootstrap_orientation"] and name != "pith_session_start":
            content_blocks.append(TextContent(type="text", text=_state["pending_bootstrap_orientation"]))
            logger.info(f"L4: Bootstrap orientation injected into {name} response")
            _state["pending_bootstrap_orientation"] = None

        return content_blocks

    except Exception as e:
        logger.error(f"Tool call error for {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"❌ Error: {e}")]


# --- Entry point ---
async def main():
    """Start the MCP server on stdio transport."""
    # C4: Generate instructions before connecting
    instructions = await generate_descriptive_instructions()
    if instructions:
        # Store for potential future use — the MCP SDK may support
        # instructions via a different mechanism in Python
        logger.info(f"Instructions ready: {len(instructions)} chars")

    logger.info(f"Pith MCP server starting (Python). API: {PITH_API_URL}")

    # Startup validation: verify API key works against running server
    if PITH_API_KEY:
        try:
            import httpx as _httpx
            _resp = _httpx.get(f"{PITH_API_URL}/health", timeout=5.0)
            if _resp.is_success:
                logger.info("Startup: Pith server reachable, health OK")
                # Test auth on a write endpoint
                _auth_resp = _httpx.get(
                    f"{PITH_API_URL}/pith_health",
                    headers={"X-API-Key": PITH_API_KEY},
                    timeout=5.0,
                )
                if _auth_resp.status_code in (401, 403):
                    logger.error(
                        "STARTUP AUTH MISMATCH: MCP API key rejected by server. "
                        "Check that PITH_API_KEY in Claude Desktop config matches "
                        "PITH_API_KEY in ~/.pith/.env. Server may be using PITH_API_KEY."
                    )
                else:
                    logger.info("Startup: API key validated OK")
            else:
                logger.warning(f"Startup: Pith server unhealthy ({_resp.status_code})")
        except Exception as _e:
            logger.warning(f"Startup: Could not reach Pith server for validation: {_e}")

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
