"""Cognitive Traces — Wave 4b.2

Structured learning event log with bidirectional concept linkage.
Traces capture the full reasoning arc: situation → intent → assessment →
justification → reflection. Reflection field is filled during reflection
cycles, not at creation time.

Also handles prediction tracking for confidence calibration (4b.1).

NEW file — no backward compat concerns.
"""

import json
import logging
import uuid
from datetime import timedelta

from app.core.datetime_utils import _utc_now, _utc_now_iso
from app.core.models import TraceRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Trace CRUD
# =============================================================================


def create_trace(
    session_id: str,
    trigger_type: str = "learning_event",
    situation: str = "",
    intent: str = "",
    assessment: str = "",
    justification: str = "",
    concept_refs: list[str] | None = None,
    agent_id: str = "default",
) -> TraceRecord:
    """Create and persist a cognitive trace.

    Args:
        session_id: Current session ID
        trigger_type: learning_event|correction|reflection|user_assertion
        situation: What was happening
        intent: What the agent was trying to do
        assessment: What was concluded
        justification: Why that conclusion
        concept_refs: List of concept IDs linked to this trace
        agent_id: Agent identifier (FC-MA-4b2)

    Returns:
        Created TraceRecord with persisted ID.
    """
    from app.storage import _db

    trace = TraceRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        created_at=_utc_now_iso(),
        trigger_type=trigger_type,
        situation=situation,
        intent=intent,
        assessment=assessment,
        justification=justification,
        concept_refs=concept_refs or [],
        agent_id=agent_id,
    )

    # Serialize structured fields into data JSON
    data = {
        "situation": trace.situation,
        "intent": trace.intent,
        "assessment": trace.assessment,
        "justification": trace.justification,
        "reflection": trace.reflection,  # Empty at creation
    }

    try:
        with _db() as conn:
            conn.execute(
                """INSERT INTO traces (id, session_id, created_at, trigger_type,
                   concept_refs, agent_id, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.id,
                    trace.session_id,
                    trace.created_at,
                    trace.trigger_type,
                    json.dumps(trace.concept_refs),
                    trace.agent_id,
                    json.dumps(data),
                ),
            )
        logger.debug(f"Trace created: {trace.id} ({trigger_type}, {len(trace.concept_refs)} refs)")
    except Exception as e:
        logger.warning(f"Failed to create trace: {e}")

    return trace


def load_trace(trace_id: str) -> TraceRecord | None:
    """Load a single trace by ID."""
    from app.storage import _db

    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT id, session_id, created_at, trigger_type, concept_refs, agent_id, data FROM traces WHERE id = ?",
                (trace_id,),
            ).fetchone()
            if not row:
                return None
            return row_to_trace(row)
    except Exception as e:
        logger.warning(f"Failed to load trace {trace_id}: {e}")
        return None


def list_traces_for_session(session_id: str, limit: int = 50) -> list[TraceRecord]:
    """List traces for a given session, ordered by creation time."""
    from app.storage import _db

    traces = []
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, session_id, created_at, trigger_type, concept_refs, agent_id, data "
                "FROM traces WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            for row in rows:
                t = row_to_trace(row)
                if t:
                    traces.append(t)
    except Exception as e:
        logger.warning(f"Failed to list traces for session {session_id}: {e}")
    return traces


def row_to_trace(row) -> TraceRecord | None:
    """Convert a DB row to a TraceRecord."""
    try:
        concept_refs = json.loads(row[4]) if row[4] else []
        data = json.loads(row[6]) if row[6] else {}
        return TraceRecord(
            id=row[0],
            session_id=row[1],
            created_at=row[2],
            trigger_type=row[3],
            concept_refs=concept_refs,
            agent_id=row[5] or "default",
            situation=data.get("situation", ""),
            intent=data.get("intent", ""),
            assessment=data.get("assessment", ""),
            justification=data.get("justification", ""),
            reflection=data.get("reflection", ""),
        )
    except Exception as e:
        logger.warning(f"Failed to parse trace row: {e}")
        return None


# =============================================================================
# Prediction Tracking (§4b.1 Tier 1)
# =============================================================================


def batch_log_predictions(
    predictions: list[dict],
    session_id: str,
) -> int:
    """Batch INSERT predictions from conversation_turn S2 [FIX C1].

    Args:
        predictions: List of dicts with {concept_id, confidence_at_retrieval}
        session_id: Current session ID

    Returns:
        Number of predictions logged.
    """
    from app.storage import _db

    if not predictions:
        return 0

    now = _utc_now_iso()
    rows = []
    for p in predictions:
        rows.append(
            (
                str(uuid.uuid4()),
                p["concept_id"],
                p["confidence_at_retrieval"],
                now,
                session_id,
                "pending",
                None,
                "evolution",
            )
        )

    try:
        with _db() as conn:
            conn.executemany(
                """INSERT INTO predictions
                   (id, concept_id, confidence_at_retrieval, retrieved_at,
                    session_id, outcome, outcome_at, outcome_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        logger.debug(f"Logged {len(rows)} predictions for session {session_id}")
        return len(rows)
    except Exception as e:
        logger.warning(f"Failed to batch log predictions: {e}")
        return 0


def resolve_predictions_for_concept(
    concept_id: str,
    outcome: str,
    outcome_source: str = "evolution",
) -> int:
    """Resolve pending predictions for a concept after evolution [FIX I1].

    Called after evolve_concept() completes. Sets outcome + outcome_at
    for all pending predictions of this concept.

    Args:
        concept_id: Concept that was evolved
        outcome: confirmed|revised|corrected
        outcome_source: evolution|correction|reflection

    Returns:
        Number of predictions resolved.
    """
    from app.storage import _db

    now = _utc_now_iso()
    try:
        with _db() as conn:
            cursor = conn.execute(
                """UPDATE predictions
                   SET outcome = ?, outcome_at = ?, outcome_source = ?
                   WHERE concept_id = ? AND outcome = 'pending'""",
                (outcome, now, outcome_source, concept_id),
            )
            count = cursor.rowcount
            if count > 0:
                logger.debug(f"Resolved {count} predictions for {concept_id} → {outcome}")
            return count
    except Exception as e:
        logger.warning(f"Failed to resolve predictions for {concept_id}: {e}")
        return 0


def expire_stale_predictions(timeout_days: int = 30) -> int:
    """Mark stale predictions (pending > timeout_days) [FIX T1].

    Called during reflection cycle. Stale predictions are excluded
    from calibration computation.

    Returns:
        Number of predictions marked stale.
    """
    from app.storage import _db

    cutoff = (_utc_now() - timedelta(days=timeout_days)).isoformat()
    try:
        with _db() as conn:
            cursor = conn.execute(
                """UPDATE predictions
                   SET outcome = 'stale', outcome_at = ?, outcome_source = 'stale_timeout'
                   WHERE outcome = 'pending' AND retrieved_at < ?""",
                (_utc_now_iso(), cutoff),
            )
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Expired {count} stale predictions (>{timeout_days} days)")
            return count
    except Exception as e:
        logger.warning(f"Failed to expire stale predictions: {e}")
        return 0


def get_calibration_data() -> dict:
    """SQL aggregation for calibration bins [FIX S1].

    Returns dict with:
        - total_predictions: int
        - predictions_with_outcomes: int
        - bins: list of CalibrationBin-shaped dicts
    """
    from app.storage import _db

    try:
        with _db() as conn:
            # Total counts
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            with_outcomes = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE outcome != 'pending' AND outcome != 'stale'"
            ).fetchone()[0]

            # Binned aggregation: 10 bins from 0.0-1.0
            bins = []
            for i in range(10):
                lower = i * 0.1
                upper = (i + 1) * 0.1
                row = conn.execute(
                    """SELECT
                        COUNT(*) as cnt,
                        SUM(CASE WHEN outcome IN ('revised', 'corrected') THEN 1 ELSE 0 END) as revisions,
                        AVG(confidence_at_retrieval) as avg_pred
                       FROM predictions
                       WHERE confidence_at_retrieval >= ? AND confidence_at_retrieval < ?
                       AND outcome != 'pending' AND outcome != 'stale'""",
                    (lower, upper if upper < 1.0 else 1.01),
                ).fetchone()

                cnt = row[0] or 0
                revisions = row[1] or 0
                avg_pred = row[2] or 0.0
                # avg_actual: 1.0 - revision_rate (confirmed = 1.0, revised/corrected = 0.0)
                avg_actual = 1.0 - (revisions / cnt) if cnt > 0 else 0.0

                bins.append(
                    {
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "prediction_count": cnt,
                        "revision_count": revisions,
                        "avg_predicted": round(avg_pred, 4),
                        "avg_actual": round(avg_actual, 4),
                        "gap": round(abs(avg_pred - avg_actual), 4),
                    }
                )

            return {
                "total_predictions": total,
                "predictions_with_outcomes": with_outcomes,
                "bins": bins,
            }
    except Exception as e:
        logger.warning(f"Failed to get calibration data: {e}")
        return {"total_predictions": 0, "predictions_with_outcomes": 0, "bins": []}
