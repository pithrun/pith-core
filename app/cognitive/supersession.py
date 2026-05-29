"""Decision Supersession — §3.3 of Governance Correction Pipeline Spec.

Detects when a new decision replaces an old one and executes the
supersession: marks old as SUPERSEDED, generates anti-terms on new,
emits governance events.

Two trigger points:
  1. Write-time: When a new decision is ingested (called from learning.py)
  2. Correction-time: When a correction's correct_claim maps to a decision
     that contradicts the corrected_claim's source decision.

Amendments applied:
  - A3: Type-ranked tiebreaker for resolution
  - A5: Status precedence (SUPERSEDED > CONTESTED > STALE > ACTIVE)
  - A6: Idempotency guard + anti-term dedup
"""

import json
import logging
import re
import struct
import time
from dataclasses import dataclass
from datetime import timedelta

from app.core.constants import (
    GOV_EVENT_DECISION_SUPERSESSION,
    GOV_EVENT_SUPERSESSION_QUALITY_DEGRADATION,
    GOV_EVENT_SUPERSESSION_REVIEW,
)
from app.core.datetime_utils import _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# TF-IDF similarity threshold for supersession candidate detection
SUPERSESSION_SIMILARITY_THRESHOLD = 0.6  # Jaccard fallback threshold
SUPERSESSION_EMBEDDING_THRESHOLD = 0.75  # SUPER-006: lowered from 0.82 (CONTRA-007/008 fixed)

# Minimum concepts in knowledge_area before write-time supersession activates
MIN_CONCEPTS_FOR_SUPERSESSION = 5  # CONTRA-ACTIVATE-001: lowered from 50 (was blocking 75% of KAs)

# SUPER-011: Freshness cutoff for association transfer (skip edges older than this)
ASSOCIATION_TRANSFER_FRESHNESS_DAYS = 90

# Max anti-terms per concept (oldest evicted)
MAX_ANTI_TERMS_PER_CONCEPT = 10

# CURRENCY-009: Explicit supersession declaration parser.
# Matches "Supersedes <concept_id>" or "supersedes: <concept_id>" in summary text.
# Bypasses same-type/same-KA similarity gate — explicit ID declaration overrides.
# Handles cross-type supersession (e.g. 'decision' supersedes 'observation' in same KA),
# which is the failure mode that caused the 2026-03-20 live session incident.
_SUPERSEDES_TEXT_PATTERN = re.compile(
    r"\bsupersedes?\b[:\s]+([a-zA-Z0-9_\-]{8,80})",
    re.IGNORECASE,
)

# A3: Type rank for contradiction/supersession resolution
# DATA-022: Expanded from 8 → full CONCEPT_TYPES coverage.
# Any type not listed here previously defaulted to 0, making it always
# lose in supersession resolution and S5.6 evolution checks.
TYPE_RANK = {
    # L6: System models (highest abstract tier)
    "system_model": 8,
    # L5: Meta-reasoning
    "cognitive_strategy": 7,
    "heuristic": 6,
    # L4: Process knowledge
    "method": 5,
    "process": 5,  # legacy compat, equivalent to method
    # L3: Reusable rules
    "principle": 4,
    # L2.5: Constraints (outrank decisions — constraints limit the decision space)
    "constraint": 4,
    # L2: Decisions
    "decision": 3,
    "goal": 2,
    "hypothesis": 2,
    # L1.5: User preferences
    "preference": 1,
    # L1: Observations
    "observation": 1,
    "pattern": 1,
    # Legacy compat
    "client_extraction": 0,
}

# A5: Status precedence (higher = stronger, only escalate)
STATUS_PRECEDENCE = {
    "ACTIVE": 1,
    "STALE": 2,
    "CONTESTED": 3,
    "SUPERSEDED": 4,
    "CONTRADICTED": 5,
    "RESOLVED": 5,
}


@dataclass
class SupersessionResult:
    """Result of a supersession check."""

    superseded: bool = False
    old_concept_id: str = ""
    new_concept_id: str = ""
    reason: str = ""
    anti_terms_generated: int = 0
    associations_transferred: int = 0  # SUPER-011: -1 signals transfer error (fail-open)
    edge_created: bool = False  # SUPER-012: supersedes association edge
    superseded_by_set: bool = False  # SUPER-012: SQL column pointer
    evidence_carried_forward: int = 0  # SUPER-013: count of evidence items transferred
    cko_created: str | None = None  # CKO-002: auto-created CKO ID
    index_evicted: bool = False  # MAINT-034: Track whether index eviction succeeded
    time_ms: float = 0.0


def _decode_concept_data(raw_data) -> dict:
    if not raw_data:
        return {}
    if isinstance(raw_data, dict):
        return dict(raw_data)
    try:
        decoded = json.loads(raw_data)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _apply_branch_authority_event_for_supersession(
    *,
    old_concept_id: str,
    new_concept_id: str,
    reason: str,
    resolved_at: str,
    conn,
) -> None:
    """Best-effort branch authority metadata wiring for explicit supersession."""

    try:
        from app.cognitive.branch_provenance_metadata import build_supersession_branch_authority_metadata_patches

        old_row = conn.execute("SELECT data FROM concepts WHERE id = ?", (old_concept_id,)).fetchone()
        new_row = conn.execute("SELECT data FROM concepts WHERE id = ?", (new_concept_id,)).fetchone()
        if not old_row or not new_row:
            return

        old_data = _decode_concept_data(old_row[0])
        new_data = _decode_concept_data(new_row[0])
        old_metadata = old_data.get("metadata") if isinstance(old_data.get("metadata"), dict) else {}
        new_metadata = new_data.get("metadata") if isinstance(new_data.get("metadata"), dict) else {}

        patch_result = build_supersession_branch_authority_metadata_patches(
            selected_metadata=new_metadata,
            superseded_metadata=old_metadata,
            authority_source_event_id=f"supersession:{old_concept_id}:{new_concept_id}",
            authority_reason=reason or "explicit supersession",
            resolved_at=resolved_at,
            resolved_by="execute_supersession",
        )
        if patch_result.status != "ready":
            logger.debug(
                "Branch authority metadata not applied for supersession %s→%s: %s",
                old_concept_id,
                new_concept_id,
                patch_result.reason,
            )
            return

        old_data["metadata"] = {**old_metadata, **patch_result.superseded_metadata_patch}
        new_data["metadata"] = {**new_metadata, **patch_result.selected_metadata_patch}
        conn.execute("UPDATE concepts SET data = ? WHERE id = ?", (json.dumps(old_data), old_concept_id))
        conn.execute("UPDATE concepts SET data = ? WHERE id = ?", (json.dumps(new_data), new_concept_id))
    except Exception as exc:
        logger.warning(
            "Branch authority metadata application failed for supersession %s→%s: %s",
            old_concept_id,
            new_concept_id,
            exc,
        )


# =============================================================================
# Type-Ranked Resolution (Amendment A3)
# =============================================================================


def resolve_type_ranked(
    concept_a_type: str,
    concept_a_authority: float,
    concept_b_type: str,
    concept_b_authority: float,
) -> str:
    """Determine winner using type rank first, authority second.

    Returns: "a", "b", or "review" (flag for human review).
    """
    rank_a = TYPE_RANK.get(concept_a_type, 0)
    rank_b = TYPE_RANK.get(concept_b_type, 0)

    if rank_a != rank_b:
        return "a" if rank_a > rank_b else "b"

    # Same type — use authority as tiebreaker
    gap = abs(concept_a_authority - concept_b_authority)
    if gap <= 0.1:
        return "review"  # Too close to call

    return "a" if concept_a_authority > concept_b_authority else "b"


# =============================================================================
# Status Precedence (Amendment A5)
# =============================================================================


def escalate_status(current_status: str, new_status: str) -> str:
    """Only escalate currency_status, never downgrade.

    SUPERSEDED > CONTESTED > STALE > ACTIVE.
    """
    current_rank = STATUS_PRECEDENCE.get(current_status, 1)
    new_rank = STATUS_PRECEDENCE.get(new_status, 1)
    return new_status if new_rank >= current_rank else current_status


# =============================================================================
# Anti-Term Generation & Persistence
# =============================================================================


def generate_anti_terms(
    old_summary: str,
    old_concept_id: str,
) -> list[dict]:
    """Generate anti-terms from a superseded concept's summary.

    Extracts key noun phrases as terms to suppress in retrieval.
    Returns list of {"term": str, "source_concept_id": str}.
    """
    if not old_summary:
        return []

    # Extract significant words (>4 chars, not stopwords)
    stopwords = frozenset(
        [
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "will",
            "should",
            "could",
            "would",
            "about",
            "their",
            "there",
            "which",
            "when",
            "what",
            "where",
            "also",
            "into",
            "more",
            "than",
            "then",
            "some",
            "only",
            "very",
            "just",
            "being",
        ]
    )
    words = old_summary.lower().split()
    significant = [
        w.strip(".,;:!?()[]\"'") for w in words if len(w) > 4 and w.lower().strip(".,;:!?()[]\"'") not in stopwords
    ]

    # Deduplicate and cap
    seen = set()
    anti_terms = []
    for term in significant:
        if term and term not in seen:
            seen.add(term)
            anti_terms.append(
                {
                    "term": term,
                    "source_concept_id": old_concept_id,
                }
            )
        if len(anti_terms) >= MAX_ANTI_TERMS_PER_CONCEPT:
            break

    return anti_terms


def _persist_anti_terms(
    new_concept_id: str,
    anti_terms: list[dict],
    conn,
) -> int:
    """Persist anti-terms into the new concept's data blob.

    Amendment A6: Dedup by (term, source_concept_id) tuple.
    Returns count of new anti-terms added.
    """
    if not anti_terms:
        return 0

    row = conn.execute(
        "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
        (new_concept_id,),
    ).fetchone()
    if not row:
        return 0

    try:
        cdata = json.loads(row[0]) if row[0] else {}
    except (json.JSONDecodeError, TypeError):
        cdata = {}

    existing = cdata.get("anti_terms", [])
    existing_keys = {(at.get("term", ""), at.get("source_concept_id", "")) for at in existing}

    added = 0
    for at in anti_terms:
        key = (at["term"], at["source_concept_id"])
        if key not in existing_keys:
            existing.append(at)
            existing_keys.add(key)
            added += 1

    # Cap total anti-terms — evict oldest first
    if len(existing) > MAX_ANTI_TERMS_PER_CONCEPT:
        existing = existing[-MAX_ANTI_TERMS_PER_CONCEPT:]

    # DEBT-010: Atomic field-level update via json_set instead of full blob clobber.
    # Only touches $.anti_terms — won't overwrite other data fields modified concurrently.
    conn.execute(
        "UPDATE concepts SET data = json_set(COALESCE(data, '{}'), '$.anti_terms', json(?)) WHERE id = ? AND is_current = 1",
        (json.dumps(existing), new_concept_id),
    )
    return added


# =============================================================================
# CURRENCY-009: Explicit Text-Based Supersession Declaration
# =============================================================================


def _check_explicit_supersession_declaration(
    new_concept_id: str,
    conn,
) -> "SupersessionResult | None":
    """CURRENCY-009: Parse 'Supersedes <concept_id>' from summary and execute supersession.

    Bypasses same-type/same-KA gates — an explicit concept_id declaration in the
    summary text is an authoritative override that always fires, regardless of type.

    This handles cross-type supersession (e.g. a 'decision' concept superseding an
    'observation' about the same strategic topic), which is the failure mode that
    caused the 2026-03-20 live session incident:
        rec_20260313_3_adapter_architecture (decision) supersedes
        cognos_positioning_correction_feb18 (observation)
    The similarity-based pipeline missed it because concept_type differs.

    Called before similarity-based detection in check_supersession_on_write().
    """
    new_row = conn.execute(
        "SELECT summary FROM concepts WHERE id = ? AND is_current = 1",
        (new_concept_id,),
    ).fetchone()
    if not new_row or not new_row[0]:
        return None

    refs = _SUPERSEDES_TEXT_PATTERN.findall(new_row[0])
    if not refs:
        return None

    for old_id in refs:
        if old_id == new_concept_id:
            continue  # Skip self-reference

        old_row = conn.execute(
            "SELECT id, currency_status FROM concepts WHERE id = ? AND is_current = 1",
            (old_id,),
        ).fetchone()
        if not old_row:
            logger.debug(
                "CURRENCY-009: Declared supersession target %s not found (non-fatal)",
                old_id,
            )
            continue

        if old_row[1] in ("SUPERSEDED", "STALE", "DISCARDED"):
            logger.debug(
                "CURRENCY-009: %s already %s — skipping writeback",
                old_id,
                old_row[1],
            )
            continue

        result = execute_supersession(
            old_concept_id=old_id,
            new_concept_id=new_concept_id,
            reason=f"explicit_text_declaration ('{old_id}' named in summary)",
            conn=conn,
        )
        if result.superseded:
            logger.info(
                "CURRENCY-009: Explicit declaration supersession: %s → %s",
                new_concept_id,
                old_id,
            )
            return result

    return None


# =============================================================================
# Supersession Detection (Write-Time)
# =============================================================================


def detect_supersession_candidates(
    new_concept_id: str,
    conn,
) -> list[tuple[str, float]]:
    """Find existing concepts that the new concept might supersede.

    Called at write-time after a new decision-type concept is ingested.
    Returns list of (old_concept_id, similarity_score) candidates.

    Gate: Only runs when knowledge_area has >= MIN_CONCEPTS_FOR_SUPERSESSION
    concepts, to avoid false positives in sparse domains.
    """
    # Load the new concept
    new_row = conn.execute(
        """SELECT id, summary, concept_type, knowledge_area, confidence,
                  authority_score, currency_status
           FROM concepts
           WHERE id = ? AND is_current = 1""",
        (new_concept_id,),
    ).fetchone()
    if not new_row:
        return []

    new_summary = new_row[1] or ""
    new_type = new_row[2] or "observation"
    new_ka = new_row[3] or "general"

    # Gate: minimum concept density in knowledge_area
    ka_count = conn.execute(
        """SELECT COUNT(*) FROM concepts
           WHERE is_current = 1
             AND status != 'deleted'
             AND knowledge_area = ?""",
        (new_ka,),
    ).fetchone()[0]
    if ka_count < MIN_CONCEPTS_FOR_SUPERSESSION:
        return []

    # CONTRA-ACTIVATE-001: Type gate removed to enable cross-type supersession.
    # Safety preserved by: MIN_CONCEPTS gate (5), embedding similarity threshold
    # (0.75 cosine / 0.6 Jaccard), same knowledge_area, and TYPE_RANK resolution
    # which prevents lower-ranked types from superseding higher-ranked types.

    # Find same-KA, ALL types, ACTIVE/CONTESTED candidates (not already SUPERSEDED)
    # Amendment A6: exclude already-SUPERSEDED concepts
    candidates = conn.execute(
        """SELECT id, summary, concept_type, confidence, authority_score,
                  currency_status
           FROM concepts
           WHERE is_current = 1
             AND status != 'deleted'
             AND id != ?
             AND knowledge_area = ?
             AND (currency_status IS NULL
                  OR currency_status != 'SUPERSEDED')""",
        (new_concept_id, new_ka),
    ).fetchall()

    if not candidates:
        return []

    # SUPER-002: Try embedding-based similarity first, fallback to Jaccard
    try:
        from app.retrieval import retrieval_engine

        if retrieval_engine and hasattr(retrieval_engine, "embedding_engine"):
            embedding_results = _embedding_based_candidates(
                new_concept_id, new_summary, new_type, new_ka, candidates, conn
            )
            if embedding_results is not None:
                return embedding_results
    except Exception as e:
        logger.warning(f"Embedding supersession fallback to Jaccard: {e}")

    # FALLBACK: Original Jaccard word-overlap
    new_words = set(new_summary.lower().split())
    return _jaccard_based_candidates(new_words, candidates)


def _decode_embedding(raw) -> list | None:
    """Decode embedding from DB storage format.

    Supports both binary float32 BLOBs (production) and JSON arrays (test fixtures).
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        if len(raw) < 4 or len(raw) % 4 != 0:
            return None
        n = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(raw, list):
        return raw
    return None


def _embedding_based_candidates(
    new_concept_id: str,
    new_summary: str,
    new_type: str,
    new_ka: str,
    candidates: list,
    conn,
) -> list[tuple[str, float]] | None:
    """SUPER-002: Embedding-based supersession candidate detection.

    Returns list of (concept_id, similarity) or None if embeddings unavailable.
    """
    from app.retrieval import retrieval_engine

    # Get new concept's embedding from DB cache first
    new_row = conn.execute("SELECT embedding FROM concepts WHERE id = ?", (new_concept_id,)).fetchone()
    new_embedding = None
    if new_row and new_row[0]:
        new_embedding = _decode_embedding(new_row[0])

    # Fallback: compute embedding if not cached
    if new_embedding is None:
        try:
            new_embedding = retrieval_engine.embedding_engine.embed_text(new_summary)
        except Exception:
            return None  # Signal caller to use Jaccard fallback
    if new_embedding is None:
        return None

    results = []
    for cand in candidates:
        cand_id = cand[0]
        cand_summary = cand[1] or ""

        # Get cached embedding from DB
        cand_row = conn.execute("SELECT embedding FROM concepts WHERE id = ?", (cand_id,)).fetchone()
        if not cand_row or not cand_row[0]:
            continue

        cand_embedding = _decode_embedding(cand_row[0])
        if cand_embedding is None or len(cand_embedding) == 0:
            continue

        sim = _cosine_similarity_vectors(new_embedding, cand_embedding)
        if sim >= SUPERSESSION_EMBEDDING_THRESHOLD:
            results.append((cand_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _jaccard_based_candidates(new_words: set, candidates: list) -> list[tuple[str, float]]:
    """Fallback: Jaccard word-overlap supersession detection."""
    results = []
    for cand in candidates:
        cand_id = cand[0]
        cand_summary = cand[1] or ""
        cand_words = set(cand_summary.lower().split())
        if not cand_words or not new_words:
            continue
        intersection = new_words & cand_words
        union = new_words | cand_words
        similarity = len(intersection) / len(union) if union else 0.0
        if similarity >= SUPERSESSION_SIMILARITY_THRESHOLD:
            results.append((cand_id, similarity))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _cosine_similarity_vectors(a: list, b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================================
# Supersession Execution
# =============================================================================


def execute_supersession(
    old_concept_id: str,
    new_concept_id: str,
    reason: str,
    conn,
) -> SupersessionResult:
    """Execute supersession: mark old as SUPERSEDED, generate anti-terms, emit events.

    Must be called within an existing transaction (conn from db_immediate).
    DATA-060: Runtime assertion enforces this — raises if conn is not transactional.
    Amendment A5: Uses status escalation — only marks SUPERSEDED if current
    status precedence allows it.
    Amendment A6: Idempotency — checks is_current=1 AND not already SUPERSEDED.
    """
    # DATA-060: Enforce transaction boundary at runtime
    if hasattr(conn, "in_transaction") and not conn.in_transaction:
        raise RuntimeError(
            f"execute_supersession() requires a transactional connection "
            f"(use db_immediate()). Got non-transactional conn for {old_concept_id}→{new_concept_id}"
        )

    t0 = time.time()
    result = SupersessionResult(
        old_concept_id=old_concept_id,
        new_concept_id=new_concept_id,
    )

    # A6: Idempotency guard — verify old concept is still eligible
    # DEBT-031: Document column indices for positional access below
    # Columns: [0]=summary, [1]=concept_type, [2]=confidence, [3]=authority_score, [4]=currency_status
    old_row = conn.execute(
        """SELECT summary, concept_type, confidence, authority_score,
                  currency_status, data
           FROM concepts
           WHERE id = ? AND is_current = 1""",
        (old_concept_id,),
    ).fetchone()
    if not old_row:
        result.reason = "old_concept_not_found"
        result.time_ms = (time.time() - t0) * 1000
        return result

    old_summary = old_row[0] or ""
    old_status = old_row[4] or "ACTIVE"

    # A6: Already superseded — no-op
    if old_status == "SUPERSEDED":
        result.reason = "already_superseded"
        result.time_ms = (time.time() - t0) * 1000
        return result

    # Load new concept for type-ranked resolution
    new_row = conn.execute(
        """SELECT concept_type, confidence, authority_score
           FROM concepts
           WHERE id = ? AND is_current = 1""",
        (new_concept_id,),
    ).fetchone()
    if not new_row:
        result.reason = "new_concept_not_found"
        result.time_ms = (time.time() - t0) * 1000
        return result

    # A3: Type-ranked resolution — verify new concept actually wins
    winner = resolve_type_ranked(
        concept_a_type=new_row[0] or "observation",
        concept_a_authority=new_row[2] or 0.5,
        concept_b_type=old_row[1] or "observation",
        concept_b_authority=old_row[3] or 0.5,
    )

    if winner == "b":
        # Old concept is stronger — don't supersede
        result.reason = "old_concept_wins_type_ranked"
        result.time_ms = (time.time() - t0) * 1000
        return result

    if winner == "review":
        # Too close — flag for review, don't auto-supersede
        now = _utc_now_iso()
        conn.execute(
            """INSERT INTO governance_events
               (event_type, concept_id, details, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                GOV_EVENT_SUPERSESSION_REVIEW,
                old_concept_id,
                json.dumps(
                    {
                        "new_concept_id": new_concept_id,
                        "reason": reason,
                        "old_type": old_row[1],
                        "new_type": new_row[0],
                        "old_authority": round(old_row[3] or 0.5, 4),
                        "new_authority": round(new_row[2] or 0.5, 4),
                    }
                ),
                now,
            ),
        )
        result.reason = "flagged_for_review"
        result.time_ms = (time.time() - t0) * 1000
        return result

    # SUPER-014: Content quality gate — reject supersession if new concept
    # is significantly thinner than old concept (prevents knowledge degradation)
    old_evidence_count = 0
    new_evidence_count = 0
    try:
        old_data_row = conn.execute(
            "SELECT data FROM concepts WHERE id = ?",
            (old_concept_id,),
        ).fetchone()
        qg_new_row = conn.execute(
            "SELECT data, summary FROM concepts WHERE id = ?",
            (new_concept_id,),
        ).fetchone()
        if old_data_row and old_data_row[0]:
            old_data = json.loads(old_data_row[0]) if isinstance(old_data_row[0], str) else old_data_row[0]
            old_evidence_count = len(old_data.get("evidence", []))
        if qg_new_row and qg_new_row[0]:
            new_data = json.loads(qg_new_row[0]) if isinstance(qg_new_row[0], str) else qg_new_row[0]
            new_evidence_count = len(new_data.get("evidence", []))
    except Exception as qg_err:
        logger.warning("SUPER-014: quality gate data load failed: %s", qg_err)
        # Fail-open: proceed with supersession

    old_summary_len = len(old_summary)
    new_summary_len = len(qg_new_row[1]) if qg_new_row and qg_new_row[1] else 0

    # Quality score: weighted combination of summary length and evidence count
    # Higher = better quality
    old_quality = old_summary_len + (old_evidence_count * 100)
    new_quality = new_summary_len + (new_evidence_count * 100)

    # If old concept has >2x quality score and same type, flag for review
    if old_quality > 0 and new_quality > 0:
        quality_ratio = old_quality / new_quality
        if quality_ratio >= 2.0:
            now = _utc_now_iso()
            conn.execute(
                """INSERT INTO governance_events
                   (event_type, concept_id, details, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    GOV_EVENT_SUPERSESSION_QUALITY_DEGRADATION,
                    old_concept_id,
                    json.dumps(
                        {
                            "new_concept_id": new_concept_id,
                            "reason": reason,
                            "old_quality": old_quality,
                            "new_quality": new_quality,
                            "quality_ratio": round(quality_ratio, 2),
                            "old_summary_len": old_summary_len,
                            "new_summary_len": new_summary_len,
                            "old_evidence_count": old_evidence_count,
                            "new_evidence_count": new_evidence_count,
                        }
                    ),
                    now,
                ),
            )
            result.reason = "quality_degradation_flagged"
            result.time_ms = (time.time() - t0) * 1000
            return result

    # --- Winner is "a" (new concept) — execute supersession ---
    now = _utc_now_iso()
    final_status = "SUPERSEDED"

    # RETRIEVAL-014 Layer 1a: Consolidated UPDATE — set ALL canonical fields
    # atomically, including is_current = 0. This improves failure atomicity:
    # previously, if the confidence UPDATE failed after the status UPDATE,
    # the concept would be marked SUPERSEDED with unchanged confidence
    # (a partial state). The consolidated UPDATE ensures either all fields
    # update atomically or none do.
    old_confidence = old_row[2] if old_row[2] is not None else 0.5  # old_row[2] = confidence
    new_confidence = max(0.0, old_confidence - 0.3)

    # INGEST-016 Fix 3: Check if old concept is factual for valid_until.
    _old_is_factual = False
    if old_row[5]:  # old_row[5] = data blob
        try:
            _old_data = json.loads(old_row[5]) if isinstance(old_row[5], str) else old_row[5]
            _old_meta = _old_data.get("metadata", {}) if isinstance(_old_data, dict) else {}
            _old_is_factual = bool(_old_meta.get("is_factual"))
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    # DATA-070: Route lifecycle fields through the central transition helper
    # so SQL status/currentness and JSON mirrors cannot drift.
    from app.storage import apply_lifecycle_transition_conn

    apply_lifecycle_transition_conn(
        conn,
        old_concept_id,
        "supersede",
        superseded_by=new_concept_id,
        reason=reason,
        confidence=new_confidence,
        valid_until=now if _old_is_factual else None,
    )

    result.superseded_by_set = True  # SUPER-012: Track pointer was set
    _apply_branch_authority_event_for_supersession(
        old_concept_id=old_concept_id,
        new_concept_id=new_concept_id,
        reason=reason,
        resolved_at=now,
        conn=conn,
    )

    # INGEST-054: Delete typed edges for superseded concept (provenance cleanup)
    try:
        _typed_deleted = conn.execute(
            "DELETE FROM associations WHERE chain_id = ? AND direction = 'forward'",
            (old_concept_id,)
        ).rowcount
        if _typed_deleted > 0:
            logger.info("INGEST-054: Deleted %d typed edges for superseded %s",
                        _typed_deleted, old_concept_id)
    except Exception as _typed_err:
        logger.warning("INGEST-054: Typed edge cleanup failed for superseded %s: %s",
                       old_concept_id, _typed_err)

    # Generate and persist anti-terms on the NEW concept
    anti_terms = generate_anti_terms(old_summary, old_concept_id)
    added = _persist_anti_terms(new_concept_id, anti_terms, conn)
    result.anti_terms_generated = added

    # SUPER-012: Create supersedes association edge (from Path A behavior)
    from app.storage import _invalidate_associations_cache  # PERF-019

    try:
        conn.execute(
            """INSERT OR IGNORE INTO associations (source, target, relation, strength, created_at, mechanism)
               VALUES (?, ?, 'supersedes', 0.9, ?, 'supersession_execution')""",
            (new_concept_id, old_concept_id, now),
        )
        result.edge_created = True
    except Exception as edge_err:
        logger.warning("SUPER-012: supersedes edge creation failed %s→%s: %s", new_concept_id, old_concept_id, edge_err)
        result.edge_created = False
    if result.edge_created:
        _invalidate_associations_cache()  # PERF-019: outside try — avoids misleading edge_created=False on invalidation error

    # SUPER-012: Add supersession evidence to old concept (from Path A behavior)
    # RETRIEVAL-014 Layer 1a: is_current filter removed from evidence UPDATE.
    # Safe because:
    # (a) idempotency guard at function entry (line 448) prevents re-entry
    #     on concepts with is_current=0
    # (b) concept was just set to is_current=0 by consolidated UPDATE above
    # (c) evidence should be appended regardless of current status
    _evidence_entry = json.dumps(
        {
            "source_type": "system",
            "content": f"Superseded by {new_concept_id}: {reason}",
            "reliability_weight": 1.0,
        }
    )
    try:
        conn.execute(
            """UPDATE concepts SET data = json_set(
                 COALESCE(data, '{}'),
                 '$.evidence',
                 json_insert(
                   COALESCE(json_extract(data, '$.evidence'), '[]'),
                   '$[#]',
                   json(?)
                 )
               ) WHERE id = ?""",
            (_evidence_entry, old_concept_id),
        )
    except Exception as ev_err:
        logger.warning("SUPER-012: evidence addition failed for %s: %s", old_concept_id, ev_err)

    # MAINT-034: Robust index eviction with retry and ghost verification.
    # Silent failures create ghost entries that pollute retrieval until restart.
    from app.retrieval import retrieval_engine  # A2: Hoisted above retry loop

    _eviction_success = False
    for _eviction_attempt in range(2):  # Max 2 attempts (original + 1 retry)
        try:
            retrieval_engine.remove_concept(old_concept_id, persist=True)
            _eviction_success = True
            logger.debug(
                "MAINT-034: Evicted %s from retrieval index (attempt %d)",
                old_concept_id,
                _eviction_attempt + 1,
            )
            break
        except Exception as idx_err:
            if _eviction_attempt == 0:
                logger.warning(
                    "MAINT-034: Index eviction failed for %s (attempt 1, retrying): %s",
                    old_concept_id,
                    idx_err,
                )
            else:
                logger.error(
                    "MAINT-034: Index eviction FAILED for %s after 2 attempts — ghost entry remains: %s",
                    old_concept_id,
                    idx_err,
                )

    # MAINT-034: Ghost verification — confirm concept is no longer in index
    if _eviction_success:
        try:
            if hasattr(retrieval_engine.index, "contains_active_concept"):
                _still_in_index = retrieval_engine.index.contains_active_concept(old_concept_id)
            else:
                _idx = retrieval_engine.index.concept_id_to_idx.get(old_concept_id)
                _deleted = getattr(retrieval_engine.index, "deleted_indices", set())
                _still_in_index = _idx is not None and _idx not in _deleted
            if _still_in_index:
                logger.error(
                    "MAINT-034: GHOST DETECTED — %s still in index after successful remove_concept()",
                    old_concept_id,
                )
                _eviction_success = False
        except Exception:
            pass  # concept_id_to_idx may not exist on all index backends — skip verification

    result.index_evicted = _eviction_success

    # SUPER-013: Content carry-forward — transfer qualifying evidence from old→new
    EVIDENCE_CARRY_FORWARD_MIN_WEIGHT = 0.5
    EVIDENCE_CARRY_FORWARD_MAX_ITEMS = 5
    try:
        cf_data_row = conn.execute(
            "SELECT data FROM concepts WHERE id = ?",
            (old_concept_id,),
        ).fetchone()

        if cf_data_row and cf_data_row[0]:
            cf_data = json.loads(cf_data_row[0]) if isinstance(cf_data_row[0], str) else cf_data_row[0]
            cf_evidence = cf_data.get("evidence", [])

            # Filter: only high-reliability evidence, skip system-generated markers
            qualifying = [
                e
                for e in cf_evidence
                if isinstance(e, dict)
                and (e.get("reliability_weight", 0) or 0) >= EVIDENCE_CARRY_FORWARD_MIN_WEIGHT
                and e.get("source_type") != "system"
            ]

            if qualifying:
                qualifying.sort(key=lambda e: e.get("reliability_weight", 0), reverse=True)
                to_transfer = qualifying[:EVIDENCE_CARRY_FORWARD_MAX_ITEMS]

                for ev in to_transfer:
                    ev["carried_from"] = old_concept_id
                    ev["carry_timestamp"] = now

                for ev in to_transfer:
                    ev_json = json.dumps(ev)
                    conn.execute(
                        """UPDATE concepts SET data = json_set(
                             COALESCE(data, '{}'),
                             '$.evidence',
                             json_insert(
                               COALESCE(json_extract(data, '$.evidence'), '[]'),
                               '$[#]',
                               json(?)
                             )
                           ) WHERE id = ?""",
                        (ev_json, new_concept_id),
                    )

                result.evidence_carried_forward = len(to_transfer)
                logger.info(
                    "SUPER-013: Carried forward %d evidence items from %s to %s",
                    len(to_transfer),
                    old_concept_id,
                    new_concept_id,
                )
    except Exception as cf_err:
        logger.warning("SUPER-013: evidence carry-forward failed %s→%s: %s", old_concept_id, new_concept_id, cf_err)

    # SUPER-011: Transfer associations from old→new concept (fail-open)
    _transferred = 0
    _skipped_stale = 0
    _skipped_self = 0
    _skipped_dup = 0
    try:
        _freshness_cutoff = (_utc_now() - timedelta(days=ASSOCIATION_TRANSFER_FRESHNESS_DAYS)).isoformat()

        old_associations = conn.execute(
            """SELECT source, target, relation, strength, created_at, mechanism, direction, chain_id
               FROM associations
               WHERE (source = ? OR target = ?) AND relation != 'supersedes'""",
            (old_concept_id, old_concept_id),
        ).fetchall()

        for row in old_associations:
            src, tgt, rel, strength, created, mech, direction, chain = row

            # NULL created_at → transfer (don't skip edges just because they lack timestamps)
            if created and created < _freshness_cutoff:
                _skipped_stale += 1
                continue

            # Substitute old→new
            new_src = new_concept_id if src == old_concept_id else src
            new_tgt = new_concept_id if tgt == old_concept_id else tgt

            # Skip self-loops
            if new_src == new_tgt:
                _skipped_self += 1
                continue

            # Skip duplicates — PK is (source, target, relation)
            existing = conn.execute(
                "SELECT 1 FROM associations WHERE source = ? AND target = ? AND relation = ?",
                (new_src, new_tgt, rel),
            ).fetchone()
            if existing:
                _skipped_dup += 1
                continue

            conn.execute(
                """INSERT INTO associations (source, target, relation, strength, created_at, mechanism, direction, chain_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_src, new_tgt, rel, strength, now, f"supersession_transfer:{old_concept_id}", direction, chain),
            )
            _transferred += 1

        result.associations_transferred = _transferred
        logger.info(
            "SUPER-011: Transferred %d associations %s→%s (skipped: %d stale, %d self, %d dup)",
            _transferred,
            old_concept_id,
            new_concept_id,
            _skipped_stale,
            _skipped_self,
            _skipped_dup,
        )
        if _transferred > 0:
            _invalidate_associations_cache()  # PERF-019: once after batch, not per-edge
    except Exception as e:
        logger.warning("SUPER-011: Association transfer failed for %s→%s: %s", old_concept_id, new_concept_id, e)
        result.associations_transferred = -1

    # Emit governance event — same transaction (A1 pattern)
    conn.execute(
        """INSERT INTO governance_events
           (event_type, concept_id, details, created_at)
           VALUES (?, ?, ?, ?)""",
        (
            GOV_EVENT_DECISION_SUPERSESSION,
            old_concept_id,
            json.dumps(
                {
                    "new_concept_id": new_concept_id,
                    "reason": reason,
                    "old_type": old_row[1],
                    "new_type": new_row[0],
                    "old_authority": round(old_row[3] or 0.5, 4),
                    "new_authority": round(new_row[2] or 0.5, 4),
                    "anti_terms_generated": added,
                    "associations_transferred": _transferred,
                    "associations_skipped_stale": _skipped_stale,
                    "associations_skipped_self": _skipped_self,
                    "associations_skipped_dup": _skipped_dup,
                    "final_status": final_status,
                    "edge_created": result.edge_created,
                    "superseded_by_set": result.superseded_by_set,
                    "index_evicted": result.index_evicted,
                }
            ),
            now,
        ),
    )

    result.superseded = True
    result.reason = reason

    # CKO-002: Check if this supersession creates a chain worth bundling
    try:
        cko_id = maybe_create_supersession_cko(new_concept_id, old_concept_id, conn)
        if cko_id:
            result.cko_created = cko_id
    except Exception as e:
        logger.debug("CKO-002: auto-CKO creation skipped: %s", e)

    result.time_ms = (time.time() - t0) * 1000
    logger.info(
        "Supersession executed: %s superseded by %s (reason=%s, anti_terms=%d)",
        old_concept_id,
        new_concept_id,
        reason,
        added,
    )
    return result


def maybe_create_supersession_cko(
    new_concept_id: str,
    old_concept_id: str,
    conn,
) -> str | None:
    """CKO-002: Auto-create CKO when a supersession chain reaches depth >= 2.

    Returns CKO ID if created, None otherwise.
    """
    try:
        # Walk backward from old_concept to find chain predecessors
        chain = [new_concept_id, old_concept_id]
        current = old_concept_id
        max_depth = 5  # Safety cap

        while len(chain) < max_depth + 2:
            predecessor = conn.execute(
                """SELECT target FROM associations
                   WHERE source = ? AND relation = 'supersedes'
                   LIMIT 1""",
                (current,),
            ).fetchone()

            if not predecessor:
                break

            chain.append(predecessor[0])
            current = predecessor[0]

        # Need at least 3 concepts (depth >= 2) to create a CKO
        if len(chain) < 3:
            return None

        # Load summaries for synthesis
        summaries = []
        for cid in chain:
            row = conn.execute(
                "SELECT summary, knowledge_area FROM concepts WHERE id = ?",
                (cid,),
            ).fetchone()
            if row:
                summaries.append((cid, row[0] or "", row[1] or "general"))

        if not summaries:
            return None

        knowledge_area = summaries[0][2]  # Use newest concept's KA

        # Check if a CKO already covers this chain
        # DEBT-238: cognitive must not import features directly (Contract 3).
        _cko_mod = __import__("importlib").import_module("app.features.cko")
        ensure_cko_table = _cko_mod.ensure_cko_table
        ensure_cko_table(conn)

        existing = conn.execute(
            """SELECT id, concept_ids FROM compound_knowledge_objects
               WHERE knowledge_area = ? AND status = 'active'""",
            (knowledge_area,),
        ).fetchall()

        for cko_id, cko_concept_ids_json in existing:
            cko_concept_ids = json.loads(cko_concept_ids_json)
            overlap = set(chain) & set(cko_concept_ids)
            if len(overlap) >= len(chain) * 0.6:
                logger.info(
                    "CKO-002: chain already covered by CKO %s (%.0f%% overlap)", cko_id, len(overlap) / len(chain) * 100
                )
                return None

        # Create synthesis from chain summaries (newest first)
        synthesis_parts = []
        for i, (cid, summary, ka) in enumerate(summaries):
            clean = summary.replace("[SUPERSEDED] ", "").replace("[SUPERSEDED]", "")
            label = "Current" if i == 0 else f"v{len(summaries) - i}"
            synthesis_parts.append(f"{label}: {clean[:200]}")

        synthesis = f"Supersession chain ({len(chain)} concepts, {knowledge_area}): " + " → ".join(synthesis_parts)
        synthesis = synthesis[:2000]

        create_cko = _cko_mod.create_cko
        cko = create_cko(
            conn=conn,
            title=f"Supersession chain: {summaries[0][1][:60]}",
            concept_ids=chain,
            synthesis=synthesis,
            knowledge_area=knowledge_area,
            cko_type="analysis",
        )

        logger.info("CKO-002: Auto-created CKO %s from %d-concept chain", cko.id, len(chain))
        return cko.id

    except Exception as e:
        logger.warning("CKO-002: auto-CKO creation failed: %s", e)
        return None  # Fail-open


# =============================================================================
# Write-Time Trigger (called from learning.py after concept creation)
# =============================================================================


def check_supersession_on_write(
    new_concept_id: str,
    conn,
    *,
    raise_errors: bool = False,
) -> SupersessionResult | None:
    """Write-time supersession check.

    Called after a new decision-type concept is ingested.
    Step 0: Explicit text-declaration check (CURRENCY-009) — bypasses type/KA gates.
    Step 1: Similarity-based candidate detection (existing pipeline).
    Returns SupersessionResult or None if no supersession occurred.
    """
    try:
        # CURRENCY-009 Step 0: Explicit 'Supersedes <id>' declaration in summary text.
        # Fires BEFORE similarity detection — catches cross-type supersession that
        # same-type similarity gate misses (root cause of 2026-03-20 incident).
        explicit_result = _check_explicit_supersession_declaration(new_concept_id, conn)
        if explicit_result and explicit_result.superseded:
            return explicit_result

        candidates = detect_supersession_candidates(new_concept_id, conn)
        if not candidates:
            return None

        # Take the highest-similarity candidate
        best_old_id, best_sim = candidates[0]
        result = execute_supersession(
            old_concept_id=best_old_id,
            new_concept_id=new_concept_id,
            reason=f"write_time_supersession (similarity={best_sim:.3f})",
            conn=conn,
        )
        return result if result.superseded else None

    except Exception as e:
        logger.warning("Supersession check failed for %s: %s", new_concept_id, e)
        if raise_errors:
            raise
        return None
