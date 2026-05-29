"""Daily Dedup Background Scan — catch post-evolution duplicates.

Memory Integrity Spec v1.2, §5.9.1 (A4-H1):
Background job: scan all active concepts for duplicate pairs
that emerged after evolution. Reports duplicates for review
or auto-merge. Runs daily.

Feature-gated by DEDUP_AT_INGESTION_ENABLED.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field

from app.core.config import (
    EMBEDDING_EVOLVE_THRESHOLD,
    EMBEDDING_SKIP_THRESHOLD,
    FEATURE_FLAGS,
)
from app.core.constants import GOV_EVENT_DEDUP_SCAN_DUPLICATE
from app.core.datetime_utils import _utc_now_iso
from app.storage import _db

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEDUP_SCAN_COSINE_THRESHOLD = 0.85  # Same threshold as ingestion dedup
DEDUP_SCAN_TOP_K = 3  # Matches to check per concept
DEDUP_SCAN_BATCH_SIZE = 100  # Process concepts in batches
DEDUP_SCAN_MAX_CONCEPTS = 5000  # Safety cap to prevent runaway scans


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DuplicatePair:
    """A detected duplicate pair for review."""

    concept_a: str
    concept_b: str
    cosine_score: float
    dedup_zone: str = "SKIP"  # "SKIP" | "EVOLVE"
    method: str = "tfidf"  # "embedding" | "tfidf"
    detected_at: str = ""
    action_taken: str = "pending"  # "pending" | "merged" | "dismissed"

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = _utc_now_iso()


@dataclass
class DedupScanResult:
    """Result of a daily dedup scan."""

    status: str  # "completed" | "disabled" | "error"
    concepts_scanned: int = 0
    duplicates_found: int = 0
    duplicate_pairs: list[DuplicatePair] = field(default_factory=list)
    duration_ms: float = 0.0
    scan_timestamp: str = ""

    def __post_init__(self):
        if not self.scan_timestamp:
            self.scan_timestamp = _utc_now_iso()


# =============================================================================
# Scan Implementation
# =============================================================================


def load_active_concepts(conn=None) -> list[tuple[str, str]]:
    """Load all active (non-discarded, current) concepts for scanning.

    Returns list of (concept_id, summary) tuples.
    """
    if conn is None:
        conn_ctx = _db()
        conn = conn_ctx.__enter__()
        own_conn = True
    else:
        own_conn = False

    try:
        rows = conn.execute(
            """
            SELECT id, summary FROM concepts
            WHERE is_current = 1
              AND (maturity IS NULL OR maturity != 'DISCARDED')
            ORDER BY id
            LIMIT ?
        """,
            (DEDUP_SCAN_MAX_CONCEPTS,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception as e:
        logger.warning("Failed to load concepts for dedup scan: %s", e)
        return []
    finally:
        if own_conn:
            try:
                conn_ctx.__exit__(None, None, None)
            except Exception:
                pass


def _get_concept_ka(concept_id: str) -> str:
    """Get knowledge_area for a concept from DB. Returns '' if not found."""
    try:
        from app.storage import _db
        with _db() as conn:
            row = conn.execute(
                "SELECT knowledge_area FROM concepts WHERE id = ? AND is_current = 1",
                (concept_id,),
            ).fetchone()
            return row[0] if row and row[0] else ""
    except Exception as e:
        logger.warning("_get_concept_ka failed for concept_id=%s: %s", concept_id, e)
        return ""


def find_duplicates(
    concepts: list[tuple[str, str]],
) -> list[DuplicatePair]:
    """Scan concept pairs for duplicates using embedding cosine similarity.

    Uses the retrieval engine's search_for_dedup_embedding for consistency
    with the ingestion dedup path (session_learn, propose_concept).
    Falls back to TF-IDF if embeddings unavailable.

    Returns deduplicated list of DuplicatePair objects with zone classification.
    """
    try:
        from app.retrieval import retrieval_engine
    except ImportError:
        logger.warning("Retrieval engine not available for dedup scan")
        return []

    engine = retrieval_engine
    if engine is None:
        logger.warning("Retrieval engine is None, skipping dedup scan")
        return []

    # INGEST-006: Config-driven method selection (matches INGEST-005 pattern)
    _use_embedding = FEATURE_FLAGS.get("EMBEDDING_DEDUP_ENABLED", False)
    if _use_embedding:
        _skip_threshold = EMBEDDING_SKIP_THRESHOLD
        _evolve_threshold = EMBEDDING_EVOLVE_THRESHOLD
        _method = "embedding"
    else:
        _skip_threshold = DEDUP_SCAN_COSINE_THRESHOLD  # 0.85 fallback
        _evolve_threshold = 0.50
        _method = "tfidf"

    seen_pairs = set()  # (sorted pair) to avoid A→B and B→A duplicates
    duplicates = []

    for concept_id, summary in concepts:
        if not summary or len(summary.strip()) < 10:
            continue

        try:
            if _use_embedding:
                results = engine.search_for_dedup_embedding(summary, top_k=DEDUP_SCAN_TOP_K)
            else:
                results = engine.search_for_dedup_tfidf(summary, top_k=DEDUP_SCAN_TOP_K)

            for match in results:
                match_id = match.get("concept_id", "")
                cosine = match.get("cosine_score", 0.0)

                if match_id == concept_id:
                    continue  # Skip self-match

                # INGEST-007: Cross-KA merge guard for dedup scan
                from app.core.config import CROSS_KA_EVOLVE_THRESHOLD, ka_groups_match
                _concept_ka = _get_concept_ka(concept_id)
                _match_ka_val = match.get("knowledge_area", "")
                _ka_match_scan = ka_groups_match(_concept_ka, _match_ka_val)
                _effective_evolve_scan = _evolve_threshold if _ka_match_scan else CROSS_KA_EVOLVE_THRESHOLD

                # INGEST-006: Two-zone classification
                if cosine >= _skip_threshold:
                    _zone = "SKIP"
                elif cosine >= _effective_evolve_scan:
                    _zone = "EVOLVE"
                else:
                    continue  # Below evolve threshold — not a duplicate

                # Normalize pair to avoid duplicates
                pair_key = tuple(sorted([concept_id, match_id]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    duplicates.append(
                        DuplicatePair(
                            concept_a=pair_key[0],
                            concept_b=pair_key[1],
                            cosine_score=cosine,
                            dedup_zone=_zone,
                            method=_method,
                        )
                    )

                    # INGEST-006: Structured dedup log (DATA-055 format)
                    logger.info(
                        "DEDUP_DECISION: zone=%s cosine=%.4f "
                        "match=%s method=%s "
                        "skip_thresh=%s evolve_thresh=%.2f "
                        "caller=dedup_scan "
                        "summary_hash=%s",
                        _zone, cosine,
                        match_id, _method,
                        _skip_threshold, _evolve_threshold,
                        hashlib.sha256(summary.encode()).hexdigest()[:12],
                    )
        except Exception as e:
            logger.debug("Dedup check failed for %s: %s", concept_id, e)
            continue

    return duplicates


def log_dedup_findings(
    pairs: list[DuplicatePair],
    conn=None,
) -> int:
    """Log duplicate findings to governance_events for audit trail.

    Returns count of events logged.
    """
    if not pairs:
        return 0

    if conn is None:
        conn_ctx = _db()
        conn = conn_ctx.__enter__()
        own_conn = True
    else:
        own_conn = False

    logged = 0
    try:
        for pair in pairs:
            try:
                conn.execute(
                    """
                    INSERT INTO governance_events
                    (event_type, concept_id, detail, created_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        GOV_EVENT_DEDUP_SCAN_DUPLICATE,
                        pair.concept_a,
                        f"Duplicate of {pair.concept_b} (cosine={pair.cosine_score:.3f}, zone={pair.dedup_zone}, method={pair.method})",
                        pair.detected_at,
                    ),
                )
                logged += 1
            except Exception as e:
                logger.debug("Failed to log dedup event: %s", e)
        conn.commit()
    except Exception as e:
        logger.warning("Failed to commit dedup events: %s", e)
    finally:
        if own_conn:
            try:
                conn_ctx.__exit__(None, None, None)
            except Exception:
                pass

    return logged


# =============================================================================
# Main Job Entry Point
# =============================================================================


def run_daily_dedup_scan(conn=None) -> DedupScanResult:
    """Run the daily dedup background scan (A4-H1).

    Scans all active concepts for duplicates that may have emerged
    after evolution. Feature-gated by DEDUP_AT_INGESTION_ENABLED.

    Returns DedupScanResult with findings.
    """
    if not FEATURE_FLAGS.get("DEDUP_AT_INGESTION_ENABLED", False):
        return DedupScanResult(status="disabled")

    start = time.time()

    try:
        # 1. Load active concepts
        concepts = load_active_concepts(conn)
        if not concepts:
            return DedupScanResult(
                status="completed",
                concepts_scanned=0,
                duration_ms=(time.time() - start) * 1000,
            )

        # 2. Find duplicates
        duplicates = find_duplicates(concepts)

        # 3. Log findings
        if duplicates:
            log_dedup_findings(duplicates, conn)
            logger.info(
                "Daily dedup scan: %d concepts scanned, %d duplicates found",
                len(concepts),
                len(duplicates),
            )

        duration_ms = (time.time() - start) * 1000

        return DedupScanResult(
            status="completed",
            concepts_scanned=len(concepts),
            duplicates_found=len(duplicates),
            duplicate_pairs=duplicates,
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.error("Daily dedup scan failed: %s", e, exc_info=True)
        return DedupScanResult(
            status="error",
            duration_ms=(time.time() - start) * 1000,
        )
