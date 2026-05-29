"""Async Processing — Sleep-Time Compute.

Gap B from governance spec v1.3:
Background task runner for currency scan, authority recalibration,
contradiction detection, skill extraction, CKO assembly/lifecycle,
edge reclassification, governance event flush, and more.

Task priorities:
  P0: Currency scan, authority recalibration, contradiction detection,
      correction-to-skill pipeline
  P1: CKO assembly/lifecycle, evidence consolidation, edge reclassification,
      governance event flush
  P2: Association discovery, staleness alerts, criteria staleness detector,
      self-model update, index rebuild

Retry logic (v1.2 FIX: D11):
  P0: 3 retries with exponential backoff (1min, 5min, 15min)
  P1: 2 retries
  P2: 1 retry
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from app.core.constants import GOV_EVENT_STALENESS_ALERT
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TaskPriority(Enum):
    P0 = 0  # Critical: currency, authority, contradiction, skills
    P1 = 1  # Important: CKO, evidence, edges, event flush
    P2 = 2  # Background: association discovery, staleness, self-model


class TaskStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TaskConfig:
    """Configuration for an async task type."""

    task_type: str
    priority: TaskPriority
    interval_hours: float  # How often to run (0 = on-demand only)
    max_runtime_seconds: float = 120.0
    max_retries: int = 3  # Based on priority: P0=3, P1=2, P2=1
    batch_size: int = 100


# Task type registry — all known async tasks
TASK_CONFIGS: dict[str, TaskConfig] = {
    # CURRENCY-002: currency_scan REMOVED — legacy scorer using updated_at with simple
    # exp decay. Overwrote batch_compute_currency (spec-compliant multi-signal scorer).
    # Canonical currency recompute is now RETRIEVAL-015 via reflection._recompute_currency().
    "authority_recalibration": TaskConfig("authority_recalibration", TaskPriority.P0, 168.0, 60, 3, 100),
    "autolearn_maintenance": TaskConfig("autolearn_maintenance", TaskPriority.P0, 0.05, 30, 3, 25),
    "contradiction_detection": TaskConfig("contradiction_detection", TaskPriority.P0, 0, 120, 3, 50),
    "correction_to_skill": TaskConfig("correction_to_skill", TaskPriority.P0, 0, 60, 3, 50),
    "cko_assembly": TaskConfig("cko_assembly", TaskPriority.P1, 0, 120, 2, 100),
    "cko_lifecycle": TaskConfig("cko_lifecycle", TaskPriority.P1, 24.0, 60, 2, 100),
    "evidence_consolidation": TaskConfig("evidence_consolidation", TaskPriority.P1, 24.0, 120, 2, 100),
    "edge_reclassification": TaskConfig(
        "edge_reclassification", TaskPriority.P1, 24.0, 120, 2, 25
    ),  # DEBT-137: right-sized from 100
    "governance_event_archive": TaskConfig("governance_event_archive", TaskPriority.P1, 0, 30, 2, 500),
    "federation_event_prune": TaskConfig(
        "federation_event_prune", TaskPriority.P2, 24.0, 30, 1, 500
    ),  # FED-010: auto-prune consumed federation events
    "association_discovery": TaskConfig("association_discovery", TaskPriority.P2, 24.0, 180, 1, 100),
    "ka_reclassification": TaskConfig(
        "ka_reclassification", TaskPriority.P2, 24.0, 180, 1, 25
    ),  # DEBT-137: right-sized from 100 (matches KA_LLM_MAX_PER_RUN)
    "staleness_alerts": TaskConfig("staleness_alerts", TaskPriority.P2, 24.0, 60, 1, 200),
    "criteria_staleness_detector": TaskConfig("criteria_staleness_detector", TaskPriority.P2, 24.0, 60, 1, 500),
    "self_model_update": TaskConfig("self_model_update", TaskPriority.P2, 0, 60, 1, 100),
    "index_rebuild": TaskConfig("index_rebuild", TaskPriority.P2, 0, 300, 1, 500),
    # EXP-032: Experiment generation as async background task
    "experiment_generation": TaskConfig("experiment_generation", TaskPriority.P2, 24.0, 180, 1, 100),
}

PHASE1_HEAVY_TASK_TYPES = frozenset(
    {
        "edge_reclassification",
        "association_discovery",
        "ka_reclassification",
        "experiment_generation",
    }
)

# Retry backoff schedules by priority (in seconds)
RETRY_BACKOFFS = {
    TaskPriority.P0: [60, 300, 900],  # 1min, 5min, 15min
    TaskPriority.P1: [60, 300],  # 1min, 5min
    TaskPriority.P2: [120],  # 2min
}


# =============================================================================
# Task Execution Tracking (v1.2 FIX: D11)
# =============================================================================


ASYNC_TASK_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS async_task_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    items_processed INTEGER DEFAULT 0,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_runs_type ON async_task_runs(task_type);
CREATE INDEX IF NOT EXISTS idx_task_runs_status ON async_task_runs(status);
"""


def ensure_async_tables(conn) -> None:
    """Create async_task_runs table if it doesn't exist."""
    for stmt in ASYNC_TASK_RUNS_DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    # MONITOR-012: Idempotent migration for existing DBs missing duration_ms
    try:
        conn.execute("ALTER TABLE async_task_runs ADD COLUMN duration_ms REAL DEFAULT NULL")
    except Exception:
        pass  # Column already exists
    conn.commit()


def record_task_start(task_type: str, conn) -> int:
    """Record a task execution start. Returns the run ID."""
    now = _utc_now_iso()
    cur = conn.execute(
        "INSERT INTO async_task_runs (task_type, started_at, status) VALUES (?, ?, 'running')",
        (task_type, now),
    )
    conn.commit()
    return cur.lastrowid


def record_task_complete(run_id: int, status: str, items_processed: int, error_message: str | None, conn) -> None:
    """Record task completion with duration tracking (MONITOR-012)."""
    now = _utc_now_iso()
    # MONITOR-012: Calculate duration from started_at for FIX-8 effectiveness tracking
    row = conn.execute("SELECT started_at FROM async_task_runs WHERE id = ?", (run_id,)).fetchone()
    duration_ms = None
    if row and row[0]:
        from datetime import datetime

        try:
            start = datetime.fromisoformat(row[0])
            end = datetime.fromisoformat(now)
            duration_ms = (end - start).total_seconds() * 1000
        except (ValueError, TypeError):
            pass
    conn.execute(
        """UPDATE async_task_runs
           SET completed_at = ?, status = ?, items_processed = ?, error_message = ?, duration_ms = ?
           WHERE id = ?""",
        (now, status, items_processed, error_message, duration_ms, run_id),
    )
    conn.commit()


def get_last_successful_run(task_type: str, conn) -> str | None:
    """Get the timestamp of the last successful run for a task type."""
    row = conn.execute(
        """SELECT completed_at FROM async_task_runs
           WHERE task_type = ? AND status = 'success'
           ORDER BY completed_at DESC LIMIT 1""",
        (task_type,),
    ).fetchone()
    return row[0] if row else None


def _last_success_dt(task_type: str, conn) -> datetime | None:
    """Return the persisted last-success timestamp for a task, if parseable."""
    last_run = get_last_successful_run(task_type, conn)
    if not last_run:
        return None
    try:
        return _ensure_aware(datetime.fromisoformat(last_run))
    except (ValueError, TypeError):
        return None


def _task_is_due(task_type: str, config: TaskConfig, conn) -> bool:
    """Return whether an interval task is due using persisted success history."""
    if config.interval_hours <= 0:
        return False
    last_dt = _last_success_dt(task_type, conn)
    if last_dt is None:
        return True
    return last_dt < (_utc_now() - timedelta(hours=config.interval_hours))


def _task_freshness_sort_key(task_type: str, conn) -> tuple[int, str, str]:
    """Sort never-run/unparseable tasks first, then oldest successful task."""
    last_dt = _last_success_dt(task_type, conn)
    if last_dt is None:
        return (0, "", task_type)
    return (1, last_dt.isoformat(), task_type)


def _autolearn_queue_backlog(conn) -> dict:
    """Return autolearn queue backlog counts for health decisions."""
    try:
        from app.cognitive.autolearn_maintenance_queue import get_autolearn_maintenance_status

        status = get_autolearn_maintenance_status(conn)
    except Exception as exc:
        logger.debug("autolearn maintenance status unavailable: %s", exc)
        return {"status_unavailable": True}

    backlog = {"queued": 0, "running": 0, "failed": 0}
    for row in status.get("counts") or []:
        state = row.get("status")
        if state in backlog:
            backlog[state] += int(row.get("count") or 0)
    backlog["oldest_queued_at"] = status.get("oldest_queued_at")
    return backlog


def _autolearn_queue_has_backlog(backlog: dict | None) -> bool:
    if not backlog:
        return False
    if backlog.get("status_unavailable"):
        return True
    return any(int(backlog.get(state) or 0) > 0 for state in ("queued", "running", "failed"))


def get_degraded_tasks(conn) -> list[dict]:
    """Get tasks that have failed recently or haven't run within their interval."""
    degraded = []
    for task_type, config in TASK_CONFIGS.items():
        if config.interval_hours <= 0:
            continue  # On-demand only
        autolearn_backlog = _autolearn_queue_backlog(conn) if task_type == "autolearn_maintenance" else None
        last_run = get_last_successful_run(task_type, conn)
        if not last_run:
            if task_type == "autolearn_maintenance" and not _autolearn_queue_has_backlog(autolearn_backlog):
                continue
            payload = {"task_type": task_type, "reason": "never_run"}
            if autolearn_backlog:
                payload["queue_backlog"] = autolearn_backlog
            degraded.append(payload)
            continue
        try:
            last_dt = _ensure_aware(datetime.fromisoformat(last_run))
            threshold = _utc_now() - timedelta(hours=config.interval_hours * 2)
            if last_dt < threshold:
                if task_type == "autolearn_maintenance" and not _autolearn_queue_has_backlog(autolearn_backlog):
                    continue
                payload = {
                    "task_type": task_type,
                    "reason": "overdue",
                    "last_run": last_run,
                    "expected_interval_hours": config.interval_hours,
                }
                if autolearn_backlog:
                    payload["queue_backlog"] = autolearn_backlog
                degraded.append(payload)
        except (ValueError, TypeError):
            degraded.append({"task_type": task_type, "reason": "invalid_timestamp"})
    return degraded


# =============================================================================
# Task Implementations
# =============================================================================


# CURRENCY-002: run_currency_scan DELETED.
# Legacy scorer that used updated_at with simple exp decay (math.exp(-0.693 * days/hl)).
# Overwrote batch_compute_currency (spec-compliant multi-signal scorer in currency.py).
# PRIMARY root cause of currency saturation at 0.99+ (97.2% of concepts).
# Canonical recompute: batch_compute_currency() called by reflection._recompute_currency().
# See also: CURRENCY-003 (learning.py last_accessed corruption).


async def run_authority_recalibration(conn, batch_size: int = 100) -> int:
    """P0: Normalize authority distribution (v1.2).

    Ensures authority scores don't all cluster at 1.0 or 0.0.
    Applies percentile-based normalization within each concept_type.
    Returns number of concepts recalibrated.
    """
    updated = 0

    # Get distinct concept types
    types = conn.execute("SELECT DISTINCT concept_type FROM concepts WHERE concept_type IS NOT NULL").fetchall()

    for (ctype,) in types:
        rows = conn.execute(
            """SELECT id, authority_score FROM concepts
               WHERE concept_type = ? AND authority_score IS NOT NULL
               ORDER BY authority_score DESC""",
            (ctype,),
        ).fetchall()

        if len(rows) < 5:
            continue  # Not enough data for meaningful normalization

        # Calculate percentile ranks
        n = len(rows)
        for rank, (concept_id, current_auth) in enumerate(rows):
            # Map rank to 0.1-0.95 range (avoid extremes)
            target = 0.95 - (rank / (n - 1)) * 0.85 if n > 1 else 0.5

            # Blend: 70% current + 30% target (gentle normalization)
            new_auth = 0.7 * (current_auth or 0.5) + 0.3 * target
            new_auth = max(0.05, min(0.95, new_auth))

            # TB-4: Type-based authority cap — prevents evidence accumulation
            # from inflating low-hierarchy types above their natural ceiling.
            # Applied AFTER blending so percentile normalization still spreads
            # values within the capped range.
            from app.core.constants import TYPE_AUTHORITY_CAPS

            type_cap = TYPE_AUTHORITY_CAPS.get(ctype)
            if type_cap is not None:
                new_auth = min(new_auth, type_cap)

            if abs(new_auth - (current_auth or 0.5)) > 0.02:
                # AUTHORITY-001: Also persist effective_authority
                # KA-006: Sync both column AND blob to prevent desync
                effective = min(new_auth, type_cap) if type_cap is not None else new_auth
                conn.execute(
                    """UPDATE concepts
                       SET authority_score = ?, effective_authority = ?,
                           data = json_set(data,
                               '$.authority_score', ?,
                               '$.effective_authority', ?
                           )
                       WHERE id = ?""",
                    (round(new_auth, 4), round(effective, 4), round(new_auth, 4), round(effective, 4), concept_id),
                )
                updated += 1

        conn.commit()
        await asyncio.sleep(0)

    logger.info("Authority recalibration: %d concepts adjusted", updated)
    return updated


async def run_edge_reclassification(conn, batch_size: int = 100) -> int:
    """P1: Reclassify untyped 'related_to' edges (v1.2 FIX: D5).

    Uses contradiction detection logic (not naive heuristics):
    1. Load 'related_to' edges in batches
    2. For each edge, load both concepts
    3. Run Phase 1+2 contradiction check:
       - HARD CONTRADICTION → 'contradicts'
       - Same type + high similarity + same direction → 'supports'
       - One references/derives from other → 'derived_from'
       - Insufficient signal → leave as 'related_to'

    Returns number of edges reclassified.
    """
    edges = conn.execute(
        """SELECT source, target, strength FROM associations
           WHERE relation = 'related_to'
           ORDER BY RANDOM()
           LIMIT ?""",
        (batch_size * 3,),  # Over-fetch to allow filtering
    ).fetchall()

    if not edges:
        return 0

    from app.core.config import EDGE_LLM_BATCH_SIZE, EDGE_LLM_RECLASSIFICATION_ENABLED

    reclassified = 0
    llm_calls_remaining = EDGE_LLM_BATCH_SIZE  # Hard cap on LLM calls per run

    for source_id, target_id, strength in edges[:batch_size]:
        # Load both concepts
        src = conn.execute(
            "SELECT id, summary, concept_type FROM concepts WHERE id = ?",
            (source_id,),
        ).fetchone()
        tgt = conn.execute(
            "SELECT id, summary, concept_type FROM concepts WHERE id = ?",
            (target_id,),
        ).fetchone()

        if not src or not tgt:
            continue

        # Tier 1: keyword heuristic (fast, free)
        new_relation = _classify_edge(src, tgt)
        llm_confidence = None

        # Tier 2: LLM classification (when Tier 1 can't classify and budget remains)
        if new_relation is None and EDGE_LLM_RECLASSIFICATION_ENABLED and llm_calls_remaining > 0:
            new_relation, llm_confidence = await _classify_edge_llm(src, tgt)
            llm_calls_remaining -= 1

        if new_relation and new_relation != "related_to":
            classifier_tier = "llm" if llm_confidence is not None else "heuristic"
            try:
                conn.execute(
                    """UPDATE associations SET relation = ?
                       WHERE source = ? AND target = ? AND relation = 'related_to'""",
                    (new_relation, source_id, target_id),
                )
                reclassified += 1

                # Governance log for LLM reclassifications (for monitoring + rollback)
                if classifier_tier == "llm":
                    _log_edge_reclassification_event(
                        conn,
                        source_id,
                        target_id,
                        old_relation="related_to",
                        new_relation=new_relation,
                        confidence=llm_confidence,
                        classifier="llm",
                    )
            except Exception as e:
                # UNIQUE constraint: edge with (source, target, new_relation) already exists
                # Delete the duplicate 'related_to' edge instead
                if "UNIQUE constraint" in str(e):
                    conn.execute(
                        """DELETE FROM associations
                           WHERE source = ? AND target = ? AND relation = 'related_to'""",
                        (source_id, target_id),
                    )
                else:
                    raise
            else:
                # PERF-019: Invalidate cache only when UPDATE succeeded (else clause fires on try success)
                from app.storage import _invalidate_associations_cache

                _invalidate_associations_cache()

            # FIX-8 (EVOLUTION_CHAIN_BREAK): Commit after EACH item to release
            # DB write lock between LLM calls. Previously committed only at batch
            # end, holding RESERVED lock across all LLM API calls (37-99,052s).
            conn.commit()

        await asyncio.sleep(0)

    conn.commit()  # Final commit for any remaining changes
    logger.info("Edge reclassification: %d/%d edges updated", reclassified, len(edges[:batch_size]))
    return reclassified


# STABILITY-032: Shared LLM JSON response parser
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*\}')  # Flat JSON only — sufficient for LLM classification responses


def _strip_llm_json(raw_text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown wrappers and malformed output.

    Handles:
    - Clean JSON: {"key": "value"}
    - Markdown-wrapped: ```json\\n{...}\\n``` or ```{...}```
    - Inline language tag: ```json{...}```
    - Prose with embedded JSON: "The answer is {"key": "value"}"

    Returns parsed dict or None on failure.
    """
    text = raw_text.strip()

    # Strip markdown code block wrapper
    if text.startswith("```"):
        # Remove opening ``` line (with optional language tag)
        text = re.sub(r'^```\w*\n?', '', text)
        # Remove closing ```
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    # Attempt 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: Extract first JSON object via regex
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _classify_edge(src: tuple, tgt: tuple) -> str | None:
    """Classify an edge between two concepts based on their content.

    Returns new relation type or None if no reclassification warranted.
    """
    src_summary = (src[1] or "").lower()
    tgt_summary = (tgt[1] or "").lower()
    src_type = src[2] or "observation"
    tgt_type = tgt[2] or "observation"

    # Check for contradiction signals
    negation_pairs = [
        ("is a", "is not a"),
        ("should", "shouldn't"),
        ("must", "must not"),
        ("always", "never"),
        ("correct", "incorrect"),
        ("valid", "invalid"),
    ]

    for pos, neg in negation_pairs:
        if (pos in src_summary and neg in tgt_summary) or (neg in src_summary and pos in tgt_summary):
            return "contradicts"

    # Check for derivation signals
    derivation_markers = ["based on", "derived from", "extends", "builds on", "follows from"]
    for marker in derivation_markers:
        if marker in src_summary or marker in tgt_summary:
            return "derived_from"

    # Check for part_of signals
    part_of_markers = ["part of", "component of", "subset of", "within", "belongs to"]
    for marker in part_of_markers:
        if marker in src_summary or marker in tgt_summary:
            return "part_of"

    # Check for support signals (same type + similar direction)
    support_markers = ["confirms", "supports", "validates", "consistent with", "aligns with"]
    for marker in support_markers:
        if marker in src_summary or marker in tgt_summary:
            return "supports"

    # Same concept_type with high word overlap → likely supports
    if src_type == tgt_type:
        src_words = set(src_summary.split())
        tgt_words = set(tgt_summary.split())
        if src_words and tgt_words:
            overlap = len(src_words & tgt_words) / min(len(src_words), len(tgt_words))
            if overlap > 0.5:
                return "supports"

    return None  # Leave as related_to


# [OPS-027] Circuit breaker: trips on first AuthenticationError, resets on process restart
_EDGE_LLM_AUTH_FAILED: bool = False


async def _classify_edge_llm(src: tuple, tgt: tuple) -> tuple[str | None, float]:
    """Tier 2: LLM-based edge classification.

    Returns (relation_type, confidence) or (None, 0.0) on failure.
    Only called when Tier 1 heuristic returns None.
    """
    global _EDGE_LLM_AUTH_FAILED
    if _EDGE_LLM_AUTH_FAILED:
        return None, 0.0

    from app.core.config import (
        EDGE_LLM_ALLOWED_RELATIONS,
        EDGE_LLM_CONFIDENCE_THRESHOLD,
        EDGE_LLM_MODEL,
        EDGE_LLM_TIMEOUT_MS,
    )

    src_summary = src[1] or ""
    tgt_summary = tgt[1] or ""
    src_type = src[2] or "observation"
    tgt_type = tgt[2] or "observation"

    prompt = _build_edge_classification_prompt(src_summary, tgt_summary, src_type, tgt_type)

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(
            timeout=EDGE_LLM_TIMEOUT_MS / 1000.0,
        )
        response = await client.messages.create(
            model=EDGE_LLM_MODEL,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()

        # STABILITY-032: Robust JSON extraction from LLM response
        result = _strip_llm_json(raw_text)
        if result is None:
            logger.debug("Edge LLM: failed to parse JSON from response: %s", raw_text[:100])
            return None, 0.0

        relation = result.get("relation", "related_to")
        confidence = float(result.get("confidence", 0.0))

        # Validate
        if relation not in EDGE_LLM_ALLOWED_RELATIONS:
            logger.debug("Edge LLM: rejected relation '%s' (not in allowed set)", relation)
            return None, 0.0

        if confidence < EDGE_LLM_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Edge LLM: rejected %s (confidence %.2f < threshold %.2f)",
                relation,
                confidence,
                EDGE_LLM_CONFIDENCE_THRESHOLD,
            )
            return None, 0.0

        # MONITOR-119: record success metric for Anthropic-direct caller
        try:
            from app.ops.metrics import metrics as _edge_metrics

            _edge_metrics.record("anthropic_llm_call", 1.0, {"caller": "edge_reclassification"})
        except Exception:
            pass
        return relation, confidence

    except Exception as e:
        # MONITOR-119: record failure metric for Anthropic-direct caller
        try:
            from app.ops.metrics import metrics as _edge_fail_metrics

            _edge_fail_metrics.record(
                "anthropic_llm_failure",
                1.0,
                {"caller": "edge_reclassification", "error": str(e)[:80]},
            )
        except Exception:
            pass
        # [OPS-027] Check for permanent account-level failures first — trip circuit breaker, log once
        try:
            import anthropic as _anthropic

            if isinstance(e, _anthropic.AuthenticationError):
                _EDGE_LLM_AUTH_FAILED = True
                logger.error(
                    "OPS-027: Edge LLM disabled — API key rejected (401). "
                    "Rotate ANTHROPIC_API_KEY in .env. Error: %s",
                    e,
                )
                return None, 0.0
            if isinstance(e, _anthropic.PermissionDeniedError):
                _EDGE_LLM_AUTH_FAILED = True
                logger.error(
                    "OPS-027: Edge LLM disabled — API key lacks permissions (403). "
                    "Check key scopes at console.anthropic.com. Error: %s",
                    e,
                )
                return None, 0.0
            # [EXP-019] Credit depletion — permanent account-level block
            if isinstance(e, _anthropic.BadRequestError) and (
                "credit" in str(e).lower() or "billing" in str(e).lower()
            ):
                _EDGE_LLM_AUTH_FAILED = True
                logger.error(
                    "OPS-027: Edge LLM disabled — API credits depleted (400). "
                    "Top up at console.anthropic.com/settings/billing. Error: %s",
                    e,
                )
                return None, 0.0
        except ImportError:
            pass
        # Split: WARNING for infra errors, DEBUG for others
        _is_infra = False
        try:
            import anthropic

            _is_infra = isinstance(e, anthropic.APIConnectionError | anthropic.RateLimitError)
        except ImportError:
            pass
        if _is_infra:
            logger.warning("Edge LLM infrastructure error (non-fatal): %s", e)
        else:
            logger.debug("Edge LLM classification failed (non-fatal): %s", e)
        return None, 0.0


def _build_edge_classification_prompt(src_summary: str, tgt_summary: str, src_type: str, tgt_type: str) -> str:
    """Build the classification prompt for a concept pair."""
    return f"""Classify the semantic relationship between these two concepts.

Concept A [{src_type}]: {src_summary[:300]}
Concept B [{tgt_type}]: {tgt_summary[:300]}

Choose exactly ONE relationship type:
- "supports": A provides evidence for or validates B (or vice versa)
- "contradicts": A and B make incompatible claims
- "derived_from": One concept builds on, extends, or refines the other
- "part_of": One concept is a component or subset of the other
- "constrains": One concept limits or bounds the other
- "supersedes": One concept replaces or makes the other obsolete
- "related_to": Topically related but no clear semantic relationship above

Respond with ONLY a JSON object:
{{"relation": "<type>", "confidence": <0.0-1.0>}}

Be conservative. If unsure, return "related_to" with low confidence. Only classify as "contradicts" or "supersedes" if there is clear semantic opposition or replacement."""


def _log_edge_reclassification_event(
    conn,
    source_id: str,
    target_id: str,
    old_relation: str,
    new_relation: str,
    confidence: float,
    classifier: str,
) -> None:
    """Log edge reclassification for monitoring and rollback."""
    from app.storage import _utc_now_iso

    conn.execute(
        """INSERT INTO governance_events (event_type, concept_id, details, created_at)
           VALUES (?, ?, ?, ?)""",
        (
            "edge_reclassification",
            source_id,
            json.dumps(
                {
                    "source": source_id,
                    "target": target_id,
                    "old_relation": old_relation,
                    "new_relation": new_relation,
                    "confidence": round(confidence, 3),
                    "classifier": classifier,
                    "model": "claude-haiku-4-5-20251001",
                }
            ),
            _utc_now_iso(),
        ),
    )


MAX_EVIDENCE_PER_CONCEPT = 10  # Cap evidence arrays to prevent unbounded growth
MIN_EVIDENCE_LENGTH = 10  # Aligned with extraction.py minimum


def _evidence_content(ev) -> str:
    """Extract content string from evidence (handles str, dict, and object types)."""
    if isinstance(ev, str):
        return ev.strip()
    elif isinstance(ev, dict):
        return (ev.get("content") or "").strip()
    elif hasattr(ev, "content"):
        return (ev.content or "").strip()
    return ""


def _evidence_sort_key(ev) -> tuple:
    """Deterministic sort key: (reliability DESC, id ASC) for idempotent capping."""
    if isinstance(ev, dict):
        return (ev.get("reliability_weight", 0.5), ev.get("id", ""))
    return (0.5, "")


async def _classify_ka_llm(
    summary: str,
    evidence_snippets: list[str],
    canonical_kas: list[str],
) -> tuple[str | None, float, str]:
    """Tier 3: LLM-based KA classification for concepts embedding can't classify.

    Sends concept summary + evidence to Haiku, asks for KA classification.
    Returns (knowledge_area, confidence, reasoning) or (None, 0.0, "") on failure.
    """
    from app.core.config import (
        KA_LLM_CONFIDENCE_THRESHOLD,
        KA_LLM_MODEL,
        KA_LLM_RECLASSIFICATION_ENABLED,
        KA_LLM_TIMEOUT_MS,
    )

    if not KA_LLM_RECLASSIFICATION_ENABLED:
        return None, 0.0, ""

    prompt = _build_ka_classification_prompt(summary, evidence_snippets, canonical_kas)

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(
            timeout=KA_LLM_TIMEOUT_MS / 1000.0,
        )
        response = await client.messages.create(
            model=KA_LLM_MODEL,
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()

        # STABILITY-032: Robust JSON extraction from LLM response
        result = _strip_llm_json(raw_text)
        if result is None:
            logger.debug("KA LLM: failed to parse JSON from response: %s", raw_text[:100])
            return None, 0.0, ""

        ka = result.get("knowledge_area", "").strip().lower()
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")[:200]  # Truncate for storage

        # Validate against canonical set
        if ka not in canonical_kas:
            logger.debug("KA LLM: rejected '%s' (not in canonical set)", ka)
            return None, 0.0, ""

        # Reject meta-categories — defeats the purpose
        if ka in ("general", "unknown", "unclassified"):
            logger.debug("KA LLM: rejected meta-category '%s'", ka)
            return None, 0.0, ""

        if confidence < KA_LLM_CONFIDENCE_THRESHOLD:
            logger.debug(
                "KA LLM: rejected %s (confidence %.2f < threshold %.2f)",
                ka,
                confidence,
                KA_LLM_CONFIDENCE_THRESHOLD,
            )
            return None, 0.0, ""

        # MONITOR-119: record success metric for Anthropic-direct caller
        try:
            from app.ops.metrics import metrics as _ka_metrics

            _ka_metrics.record("anthropic_llm_call", 1.0, {"caller": "ka_classification"})
        except Exception:
            pass
        return ka, confidence, reasoning

    except Exception as e:
        # MONITOR-119: record failure metric for Anthropic-direct caller
        try:
            from app.ops.metrics import metrics as _ka_fail_metrics

            _ka_fail_metrics.record(
                "anthropic_llm_failure",
                1.0,
                {"caller": "ka_classification", "error": str(e)[:80]},
            )
        except Exception:
            pass
        _is_infra = False
        try:
            import anthropic

            _is_infra = isinstance(
                e, anthropic.APIConnectionError | anthropic.RateLimitError | anthropic.AuthenticationError
            )
        except ImportError:
            pass
        if _is_infra:
            logger.warning("KA LLM infrastructure error (non-fatal): %s", e)
        else:
            logger.debug("KA LLM classification failed (non-fatal): %s", e)
        return None, 0.0, ""


def _build_ka_classification_prompt(
    summary: str,
    evidence_snippets: list[str],
    canonical_kas: list[str],
) -> str:
    """Build the KA classification prompt for a concept."""
    evidence_text = ""
    if evidence_snippets:
        evidence_text = "\n\nEvidence:\n" + "\n".join(f"- {e[:150]}" for e in evidence_snippets[:3])

    ka_list = "\n".join(f"- {ka}" for ka in canonical_kas if ka not in ("general", "unknown", "unclassified"))

    return f"""Classify the knowledge area of this concept.

Concept summary: {summary[:500]}
{evidence_text}

Valid knowledge areas:
{ka_list}

Respond with a JSON object:
{{"knowledge_area": "<exact area from list above>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}

Rules:
- Choose the SINGLE most specific knowledge area that fits
- If the concept truly spans multiple areas, choose the dominant one
- confidence should reflect how clearly this concept belongs to the chosen area
- Do NOT choose "general", "unknown", or "unclassified"
- If you cannot confidently classify, set confidence below 0.5"""


async def run_evidence_consolidation(conn, batch_size: int = 100) -> int:
    """P1: Clean and cap evidence arrays across concepts.

    Evidence accumulates through evolve calls and session_learn auto-merges.
    This task:
    1. Removes evidence with empty/trivial content (<10 chars)
    2. Deduplicates by evidence `id` (preserves content-identical items from
       different sources — those represent corroboration, not duplication)
    3. Normalizes legacy string evidence into dict format
    4. Caps at MAX_EVIDENCE_PER_CONCEPT, keeping highest reliability (deterministic)

    Returns number of concepts consolidated.
    """
    # KA-005: Fetch IDs only, then read blob per-row to eliminate stale-data window.
    # Old pattern: fetchall() pre-loaded all blobs, creating a race where KA changes
    # made between fetchall and per-row UPDATE were overwritten by stale blobs.
    id_rows = conn.execute(
        """SELECT id FROM concepts
           WHERE status = 'active'
           ORDER BY RANDOM()
           LIMIT ?""",
        (batch_size,),
    ).fetchall()

    consolidated = 0
    total_removed = 0

    for (concept_id,) in id_rows:
        # Per-row blob read — always gets the freshest data
        row = conn.execute("SELECT data FROM concepts WHERE id = ?", (concept_id,)).fetchone()
        if not row:
            continue
        data_str = row[0]
        try:
            data = json.loads(data_str) if data_str else {}
        except (json.JSONDecodeError, TypeError):
            continue

        evidence = data.get("evidence", [])
        if not evidence or len(evidence) <= 1:
            continue

        original_count = len(evidence)
        cleaned = []

        # Step 1: Clean — remove trivial, normalize strings, deduplicate by id
        seen_ids = set()
        for ev in evidence:
            content = _evidence_content(ev)

            # Remove trivial/empty evidence
            if not content or len(content) < MIN_EVIDENCE_LENGTH:
                continue

            # Normalize legacy string evidence to dict format
            if isinstance(ev, str):
                ev = {
                    "content": content,
                    "source_type": "legacy",
                    "reliability_weight": 0.5,
                    "consolidated_at": _utc_now_iso(),
                }

            # Deduplicate by id (not content — identical content from different
            # sources is corroboration, not duplication)
            ev_id = ev.get("id") if isinstance(ev, dict) else None
            if ev_id:
                if ev_id in seen_ids:
                    continue
                seen_ids.add(ev_id)

            cleaned.append(ev)

        # Step 2: Cap at MAX_EVIDENCE_PER_CONCEPT (deterministic sort)
        if len(cleaned) > MAX_EVIDENCE_PER_CONCEPT:
            cleaned.sort(key=_evidence_sort_key, reverse=True)
            cleaned = cleaned[:MAX_EVIDENCE_PER_CONCEPT]

        if len(cleaned) != original_count:
            removed = original_count - len(cleaned)
            total_removed += removed
            data["evidence"] = cleaned
            try:
                # KA-006: Route through write gateway for column sync
                from app.storage import update_concept_data

                update_concept_data(conn, concept_id, data, require_current=False)
                consolidated += 1
            except Exception as e:
                logger.warning(f"Evidence consolidation: failed to update {concept_id}: {e}")
                continue

        await asyncio.sleep(0)  # Yield to event loop

    if consolidated > 0:
        conn.commit()

    logger.info(
        f"Evidence consolidation: {consolidated}/{len(id_rows)} concepts cleaned, "
        f"{total_removed} evidence items removed"
    )
    return consolidated


async def run_staleness_alerts(conn, batch_size: int = 200) -> int:
    """P2: Flag high-authority concepts with low currency.

    These are important concepts that may be outdated.
    Returns number of stale alerts generated.
    """
    rows = conn.execute(
        """SELECT id, summary, authority_score, currency_score, knowledge_area
           FROM concepts
           WHERE authority_score >= 0.6 AND currency_score < 0.3
           AND concept_type IN ('decision', 'principle', 'constraint', 'method')
           LIMIT ?""",
        (batch_size,),
    ).fetchall()

    alerts = 0
    for row in rows:
        concept_id, summary, auth, currency, area = row
        # Log as governance event
        try:
            conn.execute(
                """INSERT INTO governance_events
                   (event_type, concept_id, details, created_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (
                    GOV_EVENT_STALENESS_ALERT,
                    concept_id,
                    json.dumps(
                        {
                            "authority": auth,
                            "currency": currency,
                            "knowledge_area": area,
                            "summary_preview": (summary or "")[:100],
                        }
                    ),
                ),
            )
            alerts += 1
        except Exception:
            pass

    conn.commit()
    logger.info("Staleness alerts: %d high-authority stale concepts flagged", alerts)
    return alerts


async def run_criteria_staleness_detector(conn, batch_size: int = 500) -> int:
    """P2: COGGOV-014 stale-risk detector over the full eligible pool."""
    from app.governance.staleness_detector import run_criteria_staleness_detector as _run

    result = _run(conn, page_size=batch_size)
    return result.get("state_changes", 0) + result.get("cleared", 0)


async def run_governance_event_archive(conn) -> int:
    """P2: Archive governance events older than 90 days (v1.3 FIX: L12-1).

    Prevents governance_events table from growing unbounded.
    Returns number of events archived.
    """
    cutoff = (_utc_now() - timedelta(days=90)).isoformat()

    # Count events to archive
    count = conn.execute("SELECT COUNT(*) FROM governance_events WHERE created_at < ?", (cutoff,)).fetchone()[0]

    if count == 0:
        return 0

    # Delete old events (in production, would move to archive table first)
    conn.execute("DELETE FROM governance_events WHERE created_at < ?", (cutoff,))
    conn.commit()

    logger.info("Governance event archive: removed %d events older than 90 days", count)
    return count


async def run_federation_event_prune(conn, retention_days: int = 30) -> int:
    """FED-010: Auto-prune consumed federation_events older than retention period.

    Runs as a scheduled server task so pruning happens even when bridge isn't polling.
    """
    try:
        count = conn.execute(
            """SELECT COUNT(*) FROM federation_events
               WHERE consumed = 1
               AND consumed_at < datetime('now', ? || ' days')""",
            (f"-{retention_days}",),
        ).fetchone()[0]
    except Exception:
        return 0  # Table may not exist if federation not enabled

    if count == 0:
        return 0

    conn.execute(
        """DELETE FROM federation_events
           WHERE consumed = 1
           AND consumed_at < datetime('now', ? || ' days')""",
        (f"-{retention_days}",),
    )
    conn.commit()
    logger.info("Federation event prune: removed %d consumed events older than %d days", count, retention_days)
    return count


# =============================================================================
# Task Runner
# =============================================================================


async def run_ka_reclassification(conn, batch_size: int = 50, full_corpus: bool = False) -> int:
    """Reclassify concepts with knowledge_area in ('general', 'unclassified').

    Two-tier approach:
    1. Embedding triage: Load stored embedding from DB, compare against 24 canonical
       area embeddings. If score >= threshold and gap >= threshold, reclassify.
    2. LLM fallback: For concepts that embedding can't classify confidently,
       use LLM to determine the correct knowledge_area. (Phase 2 — TODO)

    TOOLING-023: full_corpus=True bypasses batch_size LIMIT for one-shot
    corpus-wide reclassification runs (e.g. post-migration cleanup).

    Returns number of concepts reclassified.
    """
    import numpy as np

    from app.cognitive.taxonomy import classify_ka_by_embedding
    # DEBT-107: _ensure_canonical_ka_embeddings removed — called internally by classify_ka_by_embedding

    # Fetch batch: prioritize "unclassified" (newer) over "general" (legacy)
    # TOOLING-023: full_corpus bypasses LIMIT for one-shot corpus-wide runs
    _base_sql = """SELECT id, summary, embedding, knowledge_area
           FROM concepts
           WHERE is_current = 1 AND status = 'active'
             AND knowledge_area IN ('general', 'unclassified')
             AND summary IS NOT NULL AND length(summary) > 20
           ORDER BY
             CASE WHEN knowledge_area = 'unclassified' THEN 0 ELSE 1 END,
             created_at DESC"""
    if full_corpus:
        rows = conn.execute(_base_sql).fetchall()
        logger.info("TOOLING-023: full_corpus — %d concepts queued for reclassification", len(rows))
    else:
        rows = conn.execute(_base_sql + " LIMIT ?", (batch_size,)).fetchall()

    if not rows:
        return 0

    reclassified = 0

    from app.core.config import KA_LLM_MAX_PER_RUN, KA_LLM_RECLASSIFICATION_ENABLED
    from app.cognitive.taxonomy import _CANONICAL_KA_DESCRIPTIONS

    # LLM budget cap per run
    llm_calls_remaining = KA_LLM_MAX_PER_RUN if KA_LLM_RECLASSIFICATION_ENABLED else 0
    canonical_ka_list = [
        k for k in _CANONICAL_KA_DESCRIPTIONS.keys() if k not in ("general", "unknown", "unclassified")
    ]

    for row in rows:
        # STABILITY-013: Cooperative yield — don't starve event loop
        await asyncio.sleep(0)
        concept_id = row[0]
        summary = row[1]
        embedding_bytes = row[2]
        old_ka = row[3]

        # Tier 1: Embedding classification (if embedding available)
        new_ka = None
        ka_confidence = 0.0
        ka_source = None
        ka_reasoning = ""

        if embedding_bytes:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            emb_ka, emb_score, emb_gap = classify_ka_by_embedding(summary, embedding=embedding)
            if emb_ka:
                new_ka = emb_ka
                ka_confidence = emb_score
                ka_source = "embedding_async"

        # Tier 2: LLM classification (when embedding can't classify)
        if not new_ka and llm_calls_remaining > 0:
            # Gather evidence snippets for LLM context
            evidence_snippets = []
            try:
                data_row = conn.execute(
                    "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if data_row and data_row[0]:
                    data = json.loads(data_row[0])
                    for ev in (data.get("evidence") or [])[:3]:
                        content = ev.get("content", "")
                        if content and len(content) > 10:
                            evidence_snippets.append(content[:150])
            except Exception:
                pass  # Non-fatal — LLM can classify from summary alone

            llm_ka, llm_confidence, llm_reasoning = await _classify_ka_llm(
                summary, evidence_snippets, canonical_ka_list
            )
            if llm_ka:
                new_ka = llm_ka
                ka_confidence = llm_confidence
                ka_source = "llm_async"
                ka_reasoning = llm_reasoning
            llm_calls_remaining -= 1
            # STABILITY-013: Yield to event loop after each LLM call
            await asyncio.sleep(0)

        if new_ka and new_ka != old_ka:
            try:
                # Update DB column
                # Note: json_set for $.knowledge_area removed (DEBT-135) — the full
                # blob is rewritten below with metadata updates, making it redundant.
                conn.execute(
                    """UPDATE concepts SET knowledge_area = ?
                       WHERE id = ? AND is_current = 1""",
                    (new_ka, concept_id),
                )
                # Update JSON blob metadata
                data_row = conn.execute(
                    "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if data_row and data_row[0]:
                    data = json.loads(data_row[0])
                    if "metadata" not in data:
                        data["metadata"] = {}
                    data["metadata"]["knowledge_area"] = new_ka
                    data["metadata"]["knowledge_area_source"] = ka_source
                    data["metadata"]["ka_confidence"] = ka_confidence
                    data["metadata"]["ka_reclassified_from"] = old_ka
                    data["metadata"]["ka_reclassified_at"] = _utc_now_iso()
                    # KA-006: Route through write gateway for column sync
                    from app.storage import update_concept_data

                    update_concept_data(conn, concept_id, data)

                # Log governance event for audit trail (mirrors edge_reclassification pattern)
                conn.execute(
                    """INSERT INTO governance_events (event_type, concept_id, details, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        "ka_reclassification",
                        concept_id,
                        json.dumps(
                            {
                                "old_ka": old_ka,
                                "new_ka": new_ka,
                                "confidence": round(ka_confidence, 3),
                                "source": ka_source,
                                "reasoning": ka_reasoning if ka_source == "llm_async" else None,
                            }
                        ),
                        _utc_now_iso(),
                    ),
                )

                reclassified += 1
                # FIX-8 (EVOLUTION_CHAIN_BREAK): Commit per item (was every 10).
                # Prevents DB lock during LLM calls.
                conn.commit()
                logger.info(
                    f"KA reclass: {concept_id} '{old_ka}' → '{new_ka}' (score={ka_confidence:.3f}, source={ka_source})"
                )
            except Exception as e:
                logger.warning(f"KA reclass failed for {concept_id}: {e}")

    conn.commit()
    # WAL checkpoint: flush to main DB file after batch writes.
    # Critical in FUSE environments where large WAL files + SHM coordination
    # issues can cause corruption (STABILITY-007).
    try:
        busy, log, checkpointed = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        if log > 0:
            logger.info("KA reclass WAL checkpoint: %d/%d frames flushed (%d busy)", checkpointed, log, busy)
    except Exception as e:
        logger.warning(f"KA reclass WAL checkpoint failed (non-fatal): {e}")
    logger.info("KA reclassification: %d/%d concepts reclassified", reclassified, len(rows))
    return reclassified


# ======================================================================
# KA-ARCH-001: Dynamic KA embedding rebuild
# ======================================================================

def _rebuild_ka_embeddings():
    """Rebuild the embedding cache for embedding-based KA classification.

    Replaces the static _CANONICAL_KA_DESCRIPTIONS approach with dynamic
    descriptions from the knowledge_areas table. Called after KA promotions.
    """
    try:
        import app.cognitive.taxonomy as tax
        from app.storage import get_db_connection

        db = get_db_connection()
        rows = db.execute(
            """SELECT name, description FROM knowledge_areas
               WHERE status IN ('seed', 'established', 'mature')
                 AND description IS NOT NULL AND description != ''"""
        ).fetchall()

        if not rows:
            return

        from app.storage.embedding import embedding_engine
        if not embedding_engine.is_available:
            return

        names = [r[0] for r in rows]
        descriptions = [r[1] for r in rows]
        embeddings = embedding_engine.embed_batch(descriptions)

        # Update taxonomy module-level cache
        tax._canonical_ka_keys = names
        tax._canonical_ka_embeddings = embeddings

        # Persist embeddings to DB for faster cold start
        for i, name in enumerate(names):
            db.execute(
                "UPDATE knowledge_areas SET embedding = ? WHERE name = ?",
                (embeddings[i].tobytes(), name)
            )
        db.commit()

        logger.info(f"KA embeddings rebuilt: {len(names)} areas")
    except Exception as e:
        logger.warning(f"KA embedding rebuild failed: {e}")


class AsyncTaskRunner:
    """Manages background task execution with retry logic and tracking.

    Runs as a FastAPI background task, triggered by:
    - session_end (contradiction detection, skill extraction, self-model)
    - Timer (currency scan, authority recalibration, CKO lifecycle)
    - Manual trigger via API

    Retry logic (v1.2 FIX: D11):
      P0: up to 3 retries with exponential backoff (1min, 5min, 15min)
      P1: up to 2 retries
      P2: retry once
    """

    def __init__(self):
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._last_run_times: dict[str, float] = {}

    async def run_task(self, task_type: str, **kwargs) -> dict:
        """Execute a single async task with tracking and retry.

        Returns dict with status, items_processed, error if any.
        """
        from app.storage import owned_connection

        config = TASK_CONFIGS.get(task_type)
        if not config:
            return {"status": "error", "error": f"Unknown task type: {task_type}"}

        # Check if already running
        if task_type in self._running_tasks and not self._running_tasks[task_type].done():
            return {"status": "already_running"}

        with owned_connection() as meta_conn:
            ensure_async_tables(meta_conn)
            run_id = record_task_start(task_type, meta_conn)

        retry_count = 0
        backoffs = RETRY_BACKOFFS.get(config.priority, [60])
        max_retries = min(config.max_retries, len(backoffs))

        while True:
            try:
                t0 = time.perf_counter()
                with owned_connection() as task_conn:
                    ensure_async_tables(task_conn)
                    items = await self._execute_task(task_type, task_conn, config, **kwargs)
                    record_task_complete(run_id, TaskStatus.SUCCESS.value, items, None, task_conn)
                elapsed = time.perf_counter() - t0

                self._last_run_times[task_type] = time.time()

                logger.info("Async task %s: %s items in %.1fs", task_type, items, elapsed)
                return {
                    "status": TaskStatus.SUCCESS.value,
                    "items_processed": items,
                    "elapsed_seconds": round(elapsed, 1),
                }

            except TimeoutError:
                with owned_connection() as meta_conn:
                    ensure_async_tables(meta_conn)
                    record_task_complete(run_id, TaskStatus.TIMEOUT.value, 0, "Exceeded max runtime", meta_conn)
                return {"status": TaskStatus.TIMEOUT.value, "error": "Exceeded max runtime"}

            except asyncio.CancelledError:
                with owned_connection() as meta_conn:
                    ensure_async_tables(meta_conn)
                    record_task_complete(run_id, TaskStatus.CANCELLED.value, 0, "Cancelled", meta_conn)
                raise

            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    with owned_connection() as meta_conn:
                        ensure_async_tables(meta_conn)
                        record_task_complete(run_id, TaskStatus.FAILED.value, 0, str(e), meta_conn)
                    logger.error(f"Async task {task_type} failed after {retry_count} retries: {e}")
                    return {"status": TaskStatus.FAILED.value, "error": str(e), "retries": retry_count}

                # Update retry count in DB
                with owned_connection() as meta_conn:
                    ensure_async_tables(meta_conn)
                    meta_conn.execute(
                        "UPDATE async_task_runs SET retry_count = ? WHERE id = ?",
                        (retry_count, run_id),
                    )
                    meta_conn.commit()

                backoff = backoffs[min(retry_count - 1, len(backoffs) - 1)]
                logger.warning(
                    f"Async task {task_type} failed (retry {retry_count}/{max_retries}), backoff {backoff}s: {e}"
                )
                await asyncio.sleep(backoff)

    async def _execute_task(self, task_type: str, conn, config: TaskConfig, **kwargs) -> int:
        """Dispatch to the appropriate task implementation.

        Returns number of items processed.
        """
        timeout = config.max_runtime_seconds

        # CURRENCY-002: currency_scan dispatch removed (legacy scorer)
        if task_type == "authority_recalibration":
            return await asyncio.wait_for(run_authority_recalibration(conn, config.batch_size), timeout=timeout)
        elif task_type == "autolearn_maintenance":
            from app.session.autolearn_maintenance import run_autolearn_maintenance_queue

            return await asyncio.wait_for(
                run_autolearn_maintenance_queue(conn, config.batch_size),
                timeout=timeout,
            )
        elif task_type == "edge_reclassification":
            return await asyncio.wait_for(run_edge_reclassification(conn, config.batch_size), timeout=timeout)
        elif task_type == "ka_reclassification":
            return await asyncio.wait_for(
                run_ka_reclassification(
                    conn, config.batch_size, full_corpus=kwargs.get("full_corpus", False)
                ),
                timeout=timeout,
            )
        elif task_type == "staleness_alerts":
            return await asyncio.wait_for(run_staleness_alerts(conn, config.batch_size), timeout=timeout)
        elif task_type == "criteria_staleness_detector":
            return await asyncio.wait_for(run_criteria_staleness_detector(conn, config.batch_size), timeout=timeout)
        elif task_type == "governance_event_archive":
            return await asyncio.wait_for(run_governance_event_archive(conn), timeout=timeout)
        elif task_type == "federation_event_prune":
            return await asyncio.wait_for(run_federation_event_prune(conn), timeout=timeout)
        elif task_type == "correction_to_skill":
            from app.features.skills import extract_skills_from_corrections

            skills = extract_skills_from_corrections(conn, kwargs.get("gov_ctx"))
            return len(skills)

        elif task_type == "cko_lifecycle":
            from app.features.cko import run_cko_lifecycle

            result = run_cko_lifecycle(conn)  # cko_lifecycle is sync
            if not isinstance(result, dict):
                logger.warning(f"cko_lifecycle returned non-dict: {type(result)}")
                return 0
            return result.get("refreshed", 0) + result.get("archived", 0)
        elif task_type == "self_model_update":
            from app.session.self_model import update_self_model

            update_self_model(conn)
            return 1
        elif task_type == "association_discovery":
            from app.cognitive.association import auto_associate_batch
            from app.core.models import AutoAssociateBatchRequest

            result = auto_associate_batch(AutoAssociateBatchRequest())
            return result.edges_created if hasattr(result, "edges_created") else 0
        elif task_type == "evidence_consolidation":
            return await asyncio.wait_for(run_evidence_consolidation(conn, config.batch_size), timeout=timeout)
        elif task_type == "experiment_generation":
            # EXP-032: Background experiment generation (synthesis + analogy)
            from app.features.experiments import (
                _load_experiment_corpus,
                generate_experiment,
            )

            try:
                concepts, associations, assoc_counts, salience_ranks, tfidf_cache = (
                    _load_experiment_corpus()
                )
            except Exception as prep_err:
                logger.warning(f"EXP-032: corpus prep failed: {prep_err}")
                return 0

            generated = 0
            for exp_type in ("cross_domain_synthesis", "analogy_detection"):
                try:
                    result = generate_experiment(
                        experiment_type=exp_type,
                        concepts=concepts,
                        associations=associations,
                        assoc_counts=assoc_counts,
                        salience_ranks=salience_ranks,
                        tfidf_cache=tfidf_cache,
                    )
                    if result and result.status != "insufficient_data":
                        generated += 1
                        logger.info(f"EXP-032: Generated {exp_type} experiment: {result.id}")
                except Exception as gen_err:
                    logger.warning(f"EXP-032: {exp_type} generation failed: {gen_err}")
            return generated
        else:
            logger.warning(f"No implementation for async task: {task_type}")
            return 0

    async def run_scheduled_tasks(self) -> dict[str, dict]:
        """Run all tasks that are due based on their interval.

        Called periodically (e.g., every hour) or on session_end.
        Returns dict of task_type -> result.
        """
        from app.storage import owned_connection

        # MAINT-014: Clean up orphaned 'running' records from previous crashed runs
        try:
            with owned_connection() as cleanup_conn:
                ensure_async_tables(cleanup_conn)
                for task_type_key, task_cfg in TASK_CONFIGS.items():
                    stale_cutoff = min(
                        int(task_cfg.max_runtime_seconds * 1.5), 600
                    )  # FIX-8: capped 10min (was 2x uncapped)
                    cleanup_conn.execute(
                        """UPDATE async_task_runs
                           SET status = 'timeout',
                               error_message = 'Orphaned — stale cleanup (MAINT-014)',
                               completed_at = datetime('now')
                           WHERE task_type = ? AND status = 'running'
                             AND started_at < datetime('now', ? || ' seconds')""",
                        (task_type_key, f"-{stale_cutoff}"),
                    )
                cleanup_conn.commit()
        except Exception as cleanup_err:
            logger.warning("MAINT-014 orphan cleanup failed: %s", cleanup_err)

        results = {}

        due_tasks: list[tuple[str, TaskConfig]] = []
        try:
            with owned_connection() as schedule_conn:
                ensure_async_tables(schedule_conn)
                for task_type, config in sorted(
                    TASK_CONFIGS.items(),
                    key=lambda x: (x[1].priority.value, x[1].max_runtime_seconds),
                ):
                    if _task_is_due(task_type, config, schedule_conn):
                        due_tasks.append((task_type, config))

                fast_tasks = [(t, c) for t, c in due_tasks if t not in PHASE1_HEAVY_TASK_TYPES]
                heavy_tasks = [(t, c) for t, c in due_tasks if t in PHASE1_HEAVY_TASK_TYPES]
                heavy_tasks.sort(key=lambda item: _task_freshness_sort_key(item[0], schedule_conn))
        except Exception as schedule_err:
            logger.warning("Scheduled task due-state lookup failed: %s", schedule_err)
            return {"scheduler": {"status": "failed", "error": str(schedule_err)}}

        deferred_heavy_tasks = heavy_tasks[1:]
        for task_type, _config in deferred_heavy_tasks:
            results[task_type] = {
                "status": "deferred_budget",
                "reason": "heavy_task_budget_isolation",
            }

        for task_type, config in fast_tasks + heavy_tasks[:1]:

            try:
                result = await self.run_task(task_type)
                results[task_type] = result
            except asyncio.CancelledError:
                # Outer Phase 1 timeout fired — record this task as cancelled
                # but preserve already-completed results. Once the outer timeout
                # fires, remaining tasks will also be cancelled immediately,
                # so break rather than continue.
                results[task_type] = {
                    "status": "cancelled",
                    "error": "Phase 1 timeout — outer budget exhausted",
                }
                logger.warning("STABILITY-036: Phase 1 budget exhausted during %s", task_type)
                break
            except Exception as e:
                results[task_type] = {
                    "status": "failed",
                    "error": str(e),
                }
                logger.error("Scheduled task %s failed unexpectedly: %s", task_type, e)

        return results

    async def run_session_end_tasks(self, session_id: str, conn, gov_ctx=None) -> dict[str, dict]:
        """Run tasks triggered by session_end.

        Tasks: contradiction detection, skill extraction, self-model update,
        KA evolution (KA-ARCH-001).
        """
        results = {}

        # Correction-to-skill pipeline
        results["correction_to_skill"] = await self.run_task("correction_to_skill", gov_ctx=gov_ctx)

        # Self-model update
        results["self_model_update"] = await self.run_task("self_model_update")

        # KA-ARCH-001: Dynamic KA lifecycle evolution
        try:
            from app.cognitive.taxonomy import _run_lease_guarded_ka_promotion, detect_ka_merges
            transitions = _run_lease_guarded_ka_promotion("session_end")
            merge_candidates = detect_ka_merges()

            if transitions:
                logger.info(f"KA evolution: {len(transitions)} transitions: {transitions}")
                _rebuild_ka_embeddings()

            if merge_candidates:
                for ka_a, ka_b, overlap in merge_candidates:
                    logger.info(f"KA merge candidate: '{ka_a}' + '{ka_b}' (overlap={overlap:.2f})")

            results["ka_evolution"] = {
                "status": "completed",
                "transitions": transitions,
                "merge_candidates_count": len(merge_candidates),
            }
        except Exception as e:
            logger.warning(f"KA evolution failed: {e}")
            results["ka_evolution"] = {"status": "failed", "error": str(e)}

        return results

    def get_status(self, conn) -> dict:
        """Get status of all async tasks for health endpoint."""
        ensure_async_tables(conn)
        status = {}
        for task_type in TASK_CONFIGS:
            last_run = get_last_successful_run(task_type, conn)
            status[task_type] = {
                "last_successful_run": last_run,
                "is_running": (task_type in self._running_tasks and not self._running_tasks[task_type].done()),
            }
        degraded = get_degraded_tasks(conn)
        return {"tasks": status, "degraded_tasks": degraded}


# Singleton runner
task_runner = AsyncTaskRunner()
