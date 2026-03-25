"""RETRIEVAL-019: Progressive Evolution Backfill Pipeline.

Batch process that identifies concept pairs in the evolution zone
(cosine 0.50-0.82) and creates permanent supersession records via
execute_supersession(). Extends supersession coverage from cosine >= 0.82
down into the evolution zone.

Phases:
  0. Dry-run validation (count + sample, no staging)
  1. Candidate generation (embedding cosine within KA)
  2. Evaluation + precondition check
  3. Quality gate (auto-approve / manual / auto-reject)
  4. Commit via execute_supersession()
  5. Validation + auto-rollback

Design: RETRIEVAL_019_EVOLUTION_BACKFILL_DESIGN_v2.md
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.config import EVOLUTION_COSINE_MAX, EVOLUTION_COSINE_MIN
from app.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.storage import _db, _db_immediate, load_concept
from app.supersession import TYPE_RANK, execute_supersession

logger = logging.getLogger(__name__)


def _parse_iso(ts: str) -> datetime:
    """Parse ISO timestamp string to aware datetime."""
    if isinstance(ts, str):
        ts = ts.replace("Z", "+00:00")
    return _ensure_aware(datetime.fromisoformat(ts))


# =============================================================================
# Constants
# =============================================================================

# Quality gate thresholds
AUTO_APPROVE_COMPOSITE = 0.65  # RETRIEVAL-020: lowered from 0.70 (191 manual_review pairs in 0.65-0.70 band)
AUTO_REJECT_COMPOSITE = 0.50

# Default window for age difference between pairs
DEFAULT_WINDOW_DAYS = 14

# Freshness guard: newer concept must have content updated within N days
NEWER_FRESHNESS_DAYS = 30

# Auto-rollback threshold (10% degradation)
ROLLBACK_THRESHOLD = 0.10

# Validation query count
VALIDATION_QUERY_COUNT = 20


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class BackfillResult:
    """Result of a backfill run."""

    batch_id: str = ""
    knowledge_area: str = ""
    phase: str = "pending"
    candidates_generated: int = 0
    pairs_evaluated: int = 0
    auto_approved: int = 0
    manual_review: int = 0
    auto_rejected: int = 0
    committed: int = 0
    exec_rejected: int = 0
    rolled_back: bool = False
    duration_ms: float = 0.0
    errors: list = field(default_factory=list)


# =============================================================================
# Phase 1: Candidate Generation
# =============================================================================


def generate_candidates(knowledge_area: str, window_days: int = DEFAULT_WINDOW_DAYS):
    """Generate evolution-zone candidate pairs within a knowledge area.

    Uses the embedding engine's _index_matrix for cosine computation.
    Requires server process (embedding engine loaded in memory).
    """
    from app.embedding import embedding_engine as emb

    if emb is None or not hasattr(emb, "_id_to_pos") or not hasattr(emb, "_index_matrix"):
        raise RuntimeError("Embedding engine not loaded — run from server process")

    with _db() as conn:
        rows = conn.execute(
            "SELECT id, created_at FROM concepts WHERE knowledge_area = ? "
            "AND is_current = 1 AND status = 'active' ORDER BY created_at",
            (knowledge_area,),
        ).fetchall()

    concepts = [(r[0], r[1]) for r in rows]
    candidates = []

    for i, (cid_a, created_a) in enumerate(concepts):
        pos_a = emb._id_to_pos.get(cid_a)
        if pos_a is None:
            continue

        for j in range(i + 1, len(concepts)):
            cid_b, created_b = concepts[j]
            pos_b = emb._id_to_pos.get(cid_b)
            if pos_b is None:
                continue

            # Window check (B is newer since sorted by created_at)
            try:
                age_diff = (_parse_iso(created_b) - _parse_iso(created_a)).days
                if age_diff > window_days or age_diff < 0:
                    continue
            except (ValueError, TypeError):
                continue

            # Cosine similarity
            cosine = float(emb._index_matrix[pos_a] @ emb._index_matrix[pos_b])
            if not (EVOLUTION_COSINE_MIN <= cosine < EVOLUTION_COSINE_MAX):
                continue

            candidates.append((cid_a, cid_b, cosine, age_diff))

    return candidates


# =============================================================================
# Phase 2: Evaluation
# =============================================================================


def evaluate_pair(older_id: str, newer_id: str, cosine_score: float, age_diff_days: int):
    """Evaluate a candidate pair for evolution supersession.

    Scoring uses S5.6 weights scaled to 0.90 + 0.10 temporal addition.
    """
    older = load_concept(older_id, track_access=False)
    newer = load_concept(newer_id, track_access=False)
    if not older or not newer:
        return {"status": "skip", "reason": "concept_not_found"}

    # Precondition: auth_b > auth_a (STRICT, matching S5.6)
    auth_a = getattr(older, "authority_score", None) or 0.5
    auth_b = getattr(newer, "authority_score", None) or 0.5
    if auth_b <= auth_a:
        return {"status": "skip", "reason": "auth_lte"}

    # Precondition: type maturity (B >= A)
    type_a = getattr(older, "concept_type", "observation")
    type_b = getattr(newer, "concept_type", "observation")
    rank_a = TYPE_RANK.get(type_a, 1)
    rank_b = TYPE_RANK.get(type_b, 1)
    if rank_b < rank_a:
        return {"status": "skip", "reason": "type_rank"}

    # A5: Freshness guard — newer concept must have recent content
    newer_cut = getattr(newer, "content_updated_at", None) or getattr(newer, "created_at", "")
    try:
        content_age = (_utc_now() - _parse_iso(newer_cut)).days
        if content_age > NEWER_FRESHNESS_DAYS:
            return {"status": "skip", "reason": "newer_content_stale"}
    except (ValueError, TypeError):
        pass  # Can't parse — proceed

    # A4: Transitive guard — newer must not be superseded itself
    if getattr(newer, "superseded_by", None):
        return {"status": "skip", "reason": "newer_already_superseded"}

    # --- Scoring (RETRIEVAL-020: temporal_factor removed — always 0.0 at stage time) ---
    cosine_factor = (cosine_score - EVOLUTION_COSINE_MIN) / (EVOLUTION_COSINE_MAX - EVOLUTION_COSINE_MIN)
    authority_delta = min(1.0, (auth_b - auth_a) / 0.20)
    type_gap = rank_b - rank_a
    type_factor = min(1.0, type_gap / 3) if type_gap > 0 else 0.0

    composite = 0.40 * cosine_factor + 0.30 * authority_delta + 0.30 * type_factor

    return {
        "status": "evaluated",
        "composite": round(composite, 4),
        "cosine_factor": round(cosine_factor, 4),
        "authority_delta": round(authority_delta, 4),
        "type_factor": round(type_factor, 4),
        "type_progression": f"{type_a}({rank_a}) -> {type_b}({rank_b})",
        "auth_a": round(auth_a, 4),
        "auth_b": round(auth_b, 4),
    }


# =============================================================================
# Phase 3: Quality Gate + Staging
# =============================================================================


def stage_candidates(batch_id: str, knowledge_area: str, candidates: list, dry_run: bool = False):
    """Evaluate candidates and write to staging table with quality gate.

    Returns BackfillResult with counts.
    """
    result = BackfillResult(batch_id=batch_id, knowledge_area=knowledge_area)
    result.candidates_generated = len(candidates)
    staged_rows = []

    for older_id, newer_id, cosine, age_diff in candidates:
        eval_result = evaluate_pair(older_id, newer_id, cosine, age_diff)
        if eval_result["status"] == "skip":
            continue

        result.pairs_evaluated += 1
        composite = eval_result["composite"]

        # Quality gate (RETRIEVAL-020: secondary cosine gate removed — cosine already 40% of composite)
        if composite >= AUTO_APPROVE_COMPOSITE:
            status = "approved"
            result.auto_approved += 1
        elif composite >= AUTO_REJECT_COMPOSITE:
            status = "manual_review"
            result.manual_review += 1
        else:
            status = "rejected"
            result.auto_rejected += 1

        staged_rows.append(
            (
                batch_id,
                older_id,
                newer_id,
                cosine,
                composite,
                eval_result.get("authority_delta", 0),
                (eval_result.get("type_factor", 0) * 3),  # recover rank delta approx
                age_diff,
                json.dumps(eval_result),
                status,
                _utc_now_iso(),
            )
        )

    if not dry_run and staged_rows:
        with _db_immediate() as conn:
            conn.executemany(
                """INSERT INTO evolution_backfill_staging
                   (batch_id, older_concept_id, newer_concept_id, cosine_score,
                    composite_score, authority_delta, type_rank_delta,
                    content_age_days, rationale, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                staged_rows,
            )

    result.phase = "staged" if not dry_run else "dry_run"
    return result


# =============================================================================
# Phase 4: Commit
# =============================================================================


def commit_batch(batch_id: str) -> tuple[int, int]:
    """Commit approved pairs via execute_supersession().

    Returns (committed_count, exec_rejected_count).
    """
    committed = 0
    exec_rejected = 0

    with _db_immediate() as conn:
        approved = conn.execute(
            "SELECT id, older_concept_id, newer_concept_id FROM evolution_backfill_staging "
            "WHERE batch_id = ? AND status = 'approved'",
            (batch_id,),
        ).fetchall()

        for row_id, older_id, newer_id in approved:
            try:
                sup_result = execute_supersession(
                    old_concept_id=older_id,
                    new_concept_id=newer_id,
                    reason=f"evolution_backfill:{batch_id}",
                    conn=conn,
                )

                if sup_result.superseded_by_set:
                    conn.execute(
                        "UPDATE evolution_backfill_staging SET status = 'committed', "
                        "committed_at = ?, execute_result = ? WHERE id = ?",
                        (_utc_now_iso(), sup_result.reason, row_id),
                    )
                    committed += 1
                else:
                    conn.execute(
                        "UPDATE evolution_backfill_staging SET status = 'exec_rejected', "
                        "execute_result = ? WHERE id = ?",
                        (sup_result.reason, row_id),
                    )
                    exec_rejected += 1
            except Exception as e:
                logger.error(f"Backfill commit error for {older_id} -> {newer_id}: {e}")
                conn.execute(
                    "UPDATE evolution_backfill_staging SET status = 'exec_rejected', execute_result = ? WHERE id = ?",
                    (f"exception: {e}", row_id),
                )
                exec_rejected += 1

    # Emit metrics
    try:
        from app.metrics import metrics

        metrics.record("backfill_committed", committed, {"batch_id": batch_id})
        metrics.record("backfill_exec_rejected", exec_rejected, {"batch_id": batch_id})
        # MONITOR-040: Alert if exec_rejected rate exceeds 30%
        total = committed + exec_rejected
        if total > 0 and exec_rejected / total > 0.30:
            logger.warning(
                f"MONITOR-040: High backfill rejection rate — "
                f"exec_rejected={exec_rejected}/{total} ({exec_rejected/total:.0%}) for batch {batch_id}"
            )
            metrics.record("backfill_high_rejection_alert", 1, {"batch_id": batch_id})
    except Exception:
        pass

    return committed, exec_rejected


# =============================================================================
# Phase 5: Rollback
# =============================================================================


def rollback_batch(batch_id: str) -> int:
    """Rollback a committed batch. Reverses supersession effects.

    Returns count of rolled-back pairs.
    """
    count = 0
    with _db_immediate() as conn:
        committed = conn.execute(
            "SELECT id, older_concept_id, newer_concept_id FROM evolution_backfill_staging "
            "WHERE batch_id = ? AND status = 'committed'",
            (batch_id,),
        ).fetchall()

        for row_id, older_id, newer_id in committed:
            conn.execute(
                """
                UPDATE concepts
                SET superseded_by = NULL,
                    superseded_at = NULL,
                    is_current = 1,
                    currency_status = 'ACTIVE',
                    confidence = min(1.0, confidence + 0.3),
                    updated_at = ?
                WHERE id = ? AND superseded_by = ?
            """,
                (_utc_now_iso(), older_id, newer_id),
            )
            conn.execute(
                "UPDATE evolution_backfill_staging SET status = 'rolled_back' WHERE id = ?",
                (row_id,),
            )
            count += 1

    try:
        from app.metrics import metrics

        metrics.record("backfill_rollbacks", count, {"batch_id": batch_id})
    except Exception:
        pass

    logger.info(f"Backfill rollback: batch={batch_id} rolled_back={count}")
    return count


# =============================================================================
# Orchestrator
# =============================================================================


def run_backfill(
    knowledge_area: str,
    dry_run: bool = False,
    window_days: int = DEFAULT_WINDOW_DAYS,
    auto_commit: bool = True,
) -> BackfillResult:
    """Run the full backfill pipeline for a knowledge area.

    Args:
        knowledge_area: Target KA to process
        dry_run: If True, generate + evaluate but don't stage or commit
        window_days: Max age difference between pair members
        auto_commit: If True, commit approved pairs automatically

    Returns:
        BackfillResult with full pipeline stats
    """
    t0 = time.time()
    batch_id = f"bf-{knowledge_area[:8]}-{uuid.uuid4().hex[:8]}"

    logger.info(f"Backfill: starting batch={batch_id} ka={knowledge_area} dry_run={dry_run} window={window_days}d")

    result = BackfillResult(batch_id=batch_id, knowledge_area=knowledge_area)

    try:
        # Phase 1: Generate candidates
        result.phase = "generating"
        candidates = generate_candidates(knowledge_area, window_days)
        result.candidates_generated = len(candidates)
        logger.info(f"Backfill Phase 1: {len(candidates)} candidates in {knowledge_area}")

        if not candidates:
            result.phase = "complete"
            result.duration_ms = (time.time() - t0) * 1000
            return result

        # Phase 2+3: Evaluate + quality gate + stage
        result.phase = "evaluating"
        stage_result = stage_candidates(batch_id, knowledge_area, candidates, dry_run=dry_run)
        result.pairs_evaluated = stage_result.pairs_evaluated
        result.auto_approved = stage_result.auto_approved
        result.manual_review = stage_result.manual_review
        result.auto_rejected = stage_result.auto_rejected

        logger.info(
            f"Backfill Phase 2-3: evaluated={result.pairs_evaluated} "
            f"approved={result.auto_approved} manual={result.manual_review} "
            f"rejected={result.auto_rejected}"
        )

        if dry_run:
            result.phase = "dry_run_complete"
            result.duration_ms = (time.time() - t0) * 1000
            return result

        # Phase 4: Commit
        if auto_commit and result.auto_approved > 0:
            result.phase = "committing"
            committed, exec_rejected = commit_batch(batch_id)
            result.committed = committed
            result.exec_rejected = exec_rejected
            logger.info(f"Backfill Phase 4: committed={committed} exec_rejected={exec_rejected}")

        result.phase = "complete"

    except Exception as e:
        logger.error(f"Backfill error: {e}", exc_info=True)
        result.errors.append(str(e))
        result.phase = "error"

    result.duration_ms = (time.time() - t0) * 1000

    # Emit summary metrics
    try:
        from app.metrics import metrics

        metrics.record("backfill_candidates_generated", result.candidates_generated, {"knowledge_area": knowledge_area})
        metrics.record("backfill_pairs_evaluated", result.pairs_evaluated, {"knowledge_area": knowledge_area})
        metrics.record("backfill_auto_approved", result.auto_approved, {"knowledge_area": knowledge_area})
    except Exception:
        pass

    logger.info(f"Backfill complete: batch={batch_id} duration={result.duration_ms:.0f}ms")
    return result


def get_batch_status(batch_id: str) -> dict:
    """Get status summary for a batch."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM evolution_backfill_staging WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        ).fetchall()

    return {
        "batch_id": batch_id,
        "status_counts": {r[0]: r[1] for r in rows},
        "total": sum(r[1] for r in rows),
    }
