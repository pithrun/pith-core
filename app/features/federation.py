"""Federation layer for multi-session coordination.

FED-013: Session Registry + heartbeat.
Future: FED-014 (Relevancy Broker), FED-015 (Write-time Conflict Detection).
"""

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# --- Configuration constants ---
HEARTBEAT_STALE_MINUTES = 5  # Mark session 'idle' after no heartbeat
HEARTBEAT_ENDED_MINUTES = 30  # Mark session 'ended' after no heartbeat
WORKING_CONTEXT_MAX_BYTES = 2048  # Cap serialized working_context size
CLEANUP_EVERY_N_TURNS = 10  # Run stale cleanup every Nth heartbeat


class SessionRegistry:
    """Tracks active sessions and their working contexts for federation.

    Extends the existing `sessions` table with heartbeat + working_context
    rather than creating a separate registry table (per gauntlet F1.1).
    """

    def __init__(self):
        self._heartbeat_count: int = 0  # Per-instance turn counter for cleanup trigger

    def update_heartbeat(
        self,
        session_id: str,
        working_context: dict | None = None,
    ) -> bool:
        """Update session heartbeat and working context.

        Called from conversation_turn after pipeline completes.
        Non-blocking — failure is logged but never raises.

        Args:
            session_id: Current session ID
            working_context: Dict with keys: activated_domains,
                top_knowledge_areas, message_keywords, recent_concept_ids

        Returns:
            True if heartbeat was updated successfully.
        """
        try:
            from app.core.config import get_feature_flag

            if not get_feature_flag("SESSION_REGISTRY_ENABLED", False):
                return False

            from app.storage import update_session

            now_iso = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            wc_json = None
            if working_context:
                wc_json = json.dumps(working_context, default=str)
                if len(wc_json) > WORKING_CONTEXT_MAX_BYTES:
                    # Truncate by removing least important fields
                    working_context.pop("recent_concept_ids", None)
                    wc_json = json.dumps(working_context, default=str)
                    if len(wc_json) > WORKING_CONTEXT_MAX_BYTES:
                        wc_json = wc_json[:WORKING_CONTEXT_MAX_BYTES]

            updated = update_session(
                session_id,
                last_heartbeat=now_iso,
                working_context_json=wc_json,
            )

            self._heartbeat_count += 1

            # Periodic stale cleanup (every Nth heartbeat)
            if self._heartbeat_count % CLEANUP_EVERY_N_TURNS == 0:
                self._cleanup_stale_sessions()

            return updated
        except Exception as e:
            logger.debug(f"FED-013: Heartbeat update failed (non-fatal): {e}")
            return False

    def _cleanup_stale_sessions(self) -> dict:
        """Mark stale sessions as idle/ended based on heartbeat age.

        Runs inline every CLEANUP_EVERY_N_TURNS heartbeats.
        Non-blocking — failure is logged but never raises.

        Returns:
            dict with counts: {"idle": N, "ended": N}
        """
        counts = {"idle": 0, "ended": 0}
        try:
            from app.storage import _db

            with _db() as conn:
                # Mark sessions idle if no heartbeat for HEARTBEAT_STALE_MINUTES
                cursor = conn.execute(
                    """UPDATE sessions SET status = 'idle'
                       WHERE status = 'active'
                       AND last_heartbeat IS NOT NULL
                       AND last_heartbeat < datetime('now', ?)""",
                    (f"-{HEARTBEAT_STALE_MINUTES} minutes",),
                )
                counts["idle"] = cursor.rowcount
                # Mark sessions ended if no heartbeat for HEARTBEAT_ENDED_MINUTES
                cursor = conn.execute(
                    """UPDATE sessions SET status = 'ended',
                       ended_at = datetime('now')
                       WHERE status IN ('active', 'idle')
                       AND last_heartbeat IS NOT NULL
                       AND last_heartbeat < datetime('now', ?)""",
                    (f"-{HEARTBEAT_ENDED_MINUTES} minutes",),
                )
                counts["ended"] = cursor.rowcount

            if counts["idle"] > 0 or counts["ended"] > 0:
                logger.info(f"FED-013: Stale cleanup — {counts['idle']} idle, {counts['ended']} ended")
                # Log governance event
                try:
                    from app.storage import _db as _db2

                    with _db2() as conn2:
                        conn2.execute(
                            """INSERT INTO governance_events
                               (event_type, details, created_at)
                               VALUES ('federation_stale_cleanup', ?, datetime('now'))""",
                            (json.dumps(counts),),
                        )
                except Exception:
                    pass  # Governance logging is best-effort
        except Exception as e:
            logger.debug(f"FED-013: Stale cleanup failed (non-fatal): {e}")

        return counts

    def cleanup_on_startup(self) -> int:
        """Mark all 'active' sessions as 'ended' on server startup.

        After a server restart, all previously active sessions are stale.
        Their heartbeats will never update. This prevents ghost sessions
        from polluting the registry.

        Returns:
            Number of sessions marked ended.
        """
        try:
            from app.storage import _db

            with _db() as conn:
                cursor = conn.execute(
                    """UPDATE sessions SET status = 'ended',
                       ended_at = datetime('now')
                       WHERE status IN ('active', 'idle')
                       AND last_heartbeat IS NOT NULL"""
                )
                count = cursor.rowcount

            if count > 0:
                logger.info(f"FED-013: Startup cleanup — {count} stale sessions ended")
            return count

        except Exception as e:
            logger.debug(f"FED-013: Startup cleanup failed (non-fatal): {e}")
            return 0

    def get_active_sessions(self, exclude_session_id: str | None = None) -> list[dict]:
        """Get all active/idle sessions with their working contexts.

        Used by FED-014 Relevancy Broker to find target sessions for
        knowledge propagation.

        Args:
            exclude_session_id: Session to exclude (typically the caller's own)

        Returns:
            List of dicts with session_id, status, last_heartbeat, working_context
        """
        try:
            from app.storage import _db

            with _db() as conn:
                query = """SELECT id, status, last_heartbeat, working_context_json
                           FROM sessions
                           WHERE status IN ('active', 'idle')
                           AND last_heartbeat IS NOT NULL"""
                params = []
                if exclude_session_id:
                    query += " AND id != ?"
                    params.append(exclude_session_id)
                query += " ORDER BY last_heartbeat DESC"
                rows = conn.execute(query, params).fetchall()

            results = []
            for row in rows:
                wc = None
                if row["working_context_json"]:
                    try:
                        wc = json.loads(row["working_context_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(
                    {
                        "session_id": row["id"],
                        "status": row["status"],
                        "last_heartbeat": row["last_heartbeat"],
                        "working_context": wc,
                    }
                )
            return results

        except Exception as e:
            logger.debug(f"FED-013: get_active_sessions failed (non-fatal): {e}")
            return []


# Module-level singleton (lazy init pattern matches other app/ modules)
_registry: SessionRegistry | None = None


def get_registry() -> SessionRegistry:
    """Get or create the module-level SessionRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = SessionRegistry()
    return _registry


# ===========================================================================
# FED-014: Relevancy Broker — semantic scoring for federation routing
# ===========================================================================

# Domain cache (A2: avoid repeated DB queries)
_domain_cache: dict | None = None
_domain_cache_at: float = 0.0
DOMAIN_CACHE_TTL = 300  # 5 minutes

PROPAGATION_THRESHOLD = 0.3  # Minimum for standard propagation
IMMEDIATE_THRESHOLD = 0.7  # High-priority injection


def _get_domain_mappings() -> dict[str, list[tuple[str, float]]]:
    """Get domain_area_mapping with caching. Returns {knowledge_area: [(domain_id, weight), ...]}."""
    global _domain_cache, _domain_cache_at
    import time

    now = time.time()
    if _domain_cache is not None and (now - _domain_cache_at) < DOMAIN_CACHE_TTL:
        return _domain_cache

    try:
        from app.storage import _db

        with _db() as conn:
            rows = conn.execute(
                "SELECT knowledge_area, domain_id, activation_weight FROM domain_area_mapping"
            ).fetchall()

        cache: dict[str, list[tuple[str, float]]] = {}
        for row in rows:
            ka = row["knowledge_area"]
            if ka not in cache:
                cache[ka] = []
            cache[ka].append((row["domain_id"], row["activation_weight"]))

        _domain_cache = cache
        _domain_cache_at = now
        return cache
    except Exception as e:
        logger.debug(f"FED-014: Domain cache refresh failed: {e}")
        return _domain_cache or {}


class RelevancyBroker:
    """Score concept relevancy against session contexts for federation routing.

    FED-014: Determines WHICH concepts should cross session boundaries.
    Uses cognitive domain overlap, keyword overlap, concept graph proximity,
    and authority priority scoring.
    """

    def score_relevancy(self, concept_data: dict, session_context: dict) -> float:
        """Score how relevant a concept is to a target session.

        Args:
            concept_data: {knowledge_area, summary, concept_id, authority_score, confidence}
            session_context: {activated_domains, top_knowledge_areas, message_keywords,
                             recent_concept_ids} — from SessionRegistry working_context

        Returns:
            Float 0.0-1.0 relevancy score
        """
        try:
            from app.core.config import get_feature_flag

            if not get_feature_flag("SESSION_REGISTRY_ENABLED", False):
                return 0.0

            # A5: Guard against empty/missing session context
            if not session_context:
                return 0.0

            ka = concept_data.get("knowledge_area", "")

            if ka in ("general", "", None):
                # "general" KA fallback — skip domain overlap
                score = (
                    self._keyword_overlap(
                        concept_data.get("summary", ""),
                        session_context.get("message_keywords", []),
                    )
                    * 0.6
                    + self._concept_graph_proximity(
                        concept_data.get("concept_id", ""),
                        session_context.get("recent_concept_ids", []),
                    )
                    * 0.3
                    + self._authority_priority(concept_data.get("authority_score", 0.0)) * 0.1
                )
            else:
                score = (
                    self._cognitive_domain_overlap(
                        ka,
                        session_context.get("activated_domains", []),
                    )
                    * 0.4
                    + self._keyword_overlap(
                        concept_data.get("summary", ""),
                        session_context.get("message_keywords", []),
                    )
                    * 0.3
                    + self._concept_graph_proximity(
                        concept_data.get("concept_id", ""),
                        session_context.get("recent_concept_ids", []),
                    )
                    * 0.2
                    + self._authority_priority(concept_data.get("authority_score", 0.0)) * 0.1
                )

            # Amendment 16: NaN/infinity guard
            import math

            if not math.isfinite(score):
                return 0.0

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"FED-014: score_relevancy failed (non-fatal): {e}")
            return 0.0

    def filter_for_session(
        self,
        concepts: list[dict],
        session_context: dict,
        threshold: float | None = None,
    ) -> list[dict]:
        """Filter and rank concepts by relevancy for a target session.

        Returns concepts with score >= threshold, sorted by score descending.
        Each returned dict gets a 'relevancy_score' and 'propagation_tier' key added.
        """
        if threshold is None:
            threshold = PROPAGATION_THRESHOLD

        scored = []
        for concept in concepts:
            score = self.score_relevancy(concept, session_context)
            if score >= threshold:
                concept_copy = dict(concept)
                concept_copy["relevancy_score"] = score
                if score >= IMMEDIATE_THRESHOLD:
                    concept_copy["propagation_tier"] = "IMMEDIATE"
                else:
                    concept_copy["propagation_tier"] = "STANDARD"
                scored.append(concept_copy)

        scored.sort(key=lambda c: c["relevancy_score"], reverse=True)
        return scored

    def _cognitive_domain_overlap(self, knowledge_area: str, active_domains: list) -> float:
        """Query domain_area_mapping, intersect with session's active domains.

        Returns max(activation_weight) across matched domains, or 0.0.
        """
        if not knowledge_area or not active_domains:
            return 0.0

        mappings = _get_domain_mappings()
        ka_domains = mappings.get(knowledge_area, [])

        if not ka_domains:
            return 0.0

        active_set = set(active_domains)
        max_weight = 0.0
        for domain_id, weight in ka_domains:
            if domain_id in active_set:
                max_weight = max(max_weight, weight)

        return max_weight

    def _keyword_overlap(self, summary: str, recent_keywords: list) -> float:
        """TF-IDF-like overlap between concept summary and session keywords.

        Returns len(intersection) / max(len(tokens), len(keywords), 1).
        """
        if not summary or not recent_keywords:
            return 0.0

        # Simple tokenization: lowercase, split, remove very short tokens
        tokens = {t.lower() for t in summary.split() if len(t) > 2}
        keywords = {k.lower() for k in recent_keywords if len(k) > 2}

        if not tokens or not keywords:
            return 0.0

        intersection = tokens & keywords
        denominator = max(len(tokens), len(keywords), 1)
        return len(intersection) / denominator

    def _concept_graph_proximity(self, concept_id: str, context_ids: list) -> float:
        """Check association table for edges between concept and session's recent concepts.

        Returns 1.0 if direct edge exists, 0.5 if 2-hop, 0.0 otherwise.
        """
        if not concept_id or not context_ids:
            return 0.0

        try:
            from app.storage import _db

            context_set = set(context_ids)

            with _db() as conn:
                # Check direct edges (1-hop)
                placeholders = ",".join("?" for _ in context_ids)
                rows = conn.execute(
                    f"""SELECT target FROM associations WHERE source = ?
                        AND target IN ({placeholders})
                        UNION
                        SELECT source FROM associations WHERE target = ?
                        AND source IN ({placeholders})""",
                    [concept_id, *context_ids, concept_id, *context_ids],
                ).fetchall()

                if rows:
                    return 1.0

                # Check 2-hop edges
                neighbors = conn.execute(
                    """SELECT target FROM associations WHERE source = ?
                       UNION
                       SELECT source FROM associations WHERE target = ?""",
                    (concept_id, concept_id),
                ).fetchall()

                neighbor_ids = {r[0] for r in neighbors}
                if neighbor_ids & context_set:
                    return 0.5

                # Check if any neighbor connects to any context concept
                if neighbor_ids:
                    n_placeholders = ",".join("?" for _ in neighbor_ids)
                    c_placeholders = ",".join("?" for _ in context_ids)
                    hop2 = conn.execute(
                        f"""SELECT 1 FROM associations
                            WHERE source IN ({n_placeholders})
                            AND target IN ({c_placeholders})
                            LIMIT 1""",
                        [*neighbor_ids, *context_ids],
                    ).fetchone()
                    if hop2:
                        return 0.5

            return 0.0
        except Exception as e:
            logger.debug(f"FED-014: graph_proximity failed: {e}")
            return 0.0

    def _authority_priority(self, authority_score: float | None) -> float:
        """Linear scale: higher authority = higher priority. Clamped 0.0-1.0."""
        if authority_score is None:
            return 0.0
        return max(0.0, min(float(authority_score), 1.0))


# Module-level broker singleton
_broker: RelevancyBroker | None = None


def get_broker() -> RelevancyBroker:
    """Get or create the module-level RelevancyBroker singleton."""
    global _broker
    if _broker is None:
        _broker = RelevancyBroker()
    return _broker


# ===========================================================================
# FED-015: Write-Time Conflict Detection
# ===========================================================================

from dataclasses import dataclass, field

CONFLICT_WINDOW_SECONDS = 3600  # 1 hour lookback
MAX_CONFLICTS_PER_WINDOW = 5  # Throttle: max per session per 5-min window
CONFLICT_CIRCUIT_BREAKER = 3  # Flag as high_conflict after this many


@dataclass
class ConflictReport:
    """Result from write-time cross-session conflict detection."""

    conflicts: list[dict] = field(default_factory=list)
    hard_count: int = 0
    soft_count: int = 0
    checked_count: int = 0
    elapsed_ms: float = 0.0
    throttled: bool = False
    circuit_breaker_tripped: bool = False


def _concept_to_scored(row: dict):
    """Convert DB row dict to ScoredConcept for contradiction primitives (A1)."""
    try:
        from app.cognitive.contradiction import ScoredConcept

        embedding = None
        raw_emb = row.get("embedding")
        if raw_emb is not None:
            import numpy as np

            embedding = np.frombuffer(raw_emb, dtype=np.float32)

        return ScoredConcept(
            concept_id=row.get("id", ""),
            summary=row.get("summary", ""),
            knowledge_area=row.get("knowledge_area", "general"),
            authority_score=float(row.get("authority_score", 0.0) or 0.0),
            currency_score=float(row.get("currency_score", 0.0) or 0.0),
            embedding=embedding,
            created_at=row.get("created_at"),
            concept_type=row.get("concept_type"),
        )
    except Exception as e:
        logger.debug(f"FED-015: _concept_to_scored failed: {e}")
        return None


def detect_write_conflict(
    new_concept_data: dict,
    source_session_id: str,
) -> ConflictReport:
    """Detect cross-session contradictions at write time.

    Calls existing _phase_1_check and _phase_2_check from contradiction.py
    with cross-session filtering. Does NOT block writes — conflicts are
    recorded for propagation but the concept is already written.

    Args:
        new_concept_data: {id, summary, knowledge_area, embedding, authority_score, ...}
        source_session_id: The writing session's ID

    Returns:
        ConflictReport with conflicts found
    """
    import time

    t0 = time.perf_counter()
    report = ConflictReport()

    try:
        from app.core.config import get_feature_flag

        if not get_feature_flag("SESSION_REGISTRY_ENABLED", False):
            return report

        if not source_session_id or not new_concept_data.get("summary"):
            return report

        from app.cognitive.contradiction import ContradictionType, _phase_2_check
        from app.storage import _db

        # Build ScoredConcept for the new concept
        new_scored = _concept_to_scored(new_concept_data)
        if new_scored is None:
            return report

        ka = new_concept_data.get("knowledge_area", "general")

        # 1. Find recent concepts from OTHER sessions in same knowledge_area
        with _db() as conn:
            rows = conn.execute(
                """SELECT id, summary, knowledge_area, authority_score,
                          currency_score, embedding, created_at, concept_type,
                          session_id
                   FROM concepts
                   WHERE knowledge_area = ?
                   AND session_id IS NOT NULL
                   AND session_id != ?
                   AND created_at > datetime('now', ?)
                   ORDER BY created_at DESC LIMIT 10""",
                (ka, source_session_id, f"-{CONFLICT_WINDOW_SECONDS} seconds"),
            ).fetchall()

        candidates = [dict(r) for r in rows]
        report.checked_count = len(candidates)

        if not candidates:
            report.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return report

        # 2. Run contradiction detection on each candidate
        import hashlib

        seen_hashes: set[str] = set()

        for cand_dict in candidates:
            try:
                cand_scored = _concept_to_scored(cand_dict)
                if cand_scored is None:
                    continue

                # Circuit breaker check
                if report.hard_count >= CONFLICT_CIRCUIT_BREAKER:
                    report.circuit_breaker_tripped = True
                    break

                # Throttle check
                if (report.hard_count + report.soft_count) >= MAX_CONFLICTS_PER_WINDOW:
                    report.throttled = True
                    break

                # Dedup by summary hash
                pair_hash = hashlib.sha256((new_scored.summary[:200] + cand_scored.summary[:200]).encode()).hexdigest()[
                    :16
                ]
                if pair_hash in seen_hashes:
                    continue
                seen_hashes.add(pair_hash)

                # CONTRA-018: Use Phase 2 semantic detection (replaces Phase 1 keyword negation)
                # Phase 2 uses embedding similarity + directional opposition —
                # same logic as retrieval-time detection, applied cross-session.
                result = _phase_2_check(new_scored, cand_scored)

                # A4: Cross-session fallback — direct embedding check if Phase 2
                # misses due to CROSS_TOPIC_OVERLAP_MIN gate (cross-session concepts
                # may have lower keyword overlap despite genuine opposition)
                if result is None:
                    if new_scored.embedding is not None and cand_scored.embedding is not None:
                        try:
                            from app.cognitive.contradiction import (
                                EMBEDDING_SAME_TOPIC_THRESHOLD,
                                ContradictionPair,
                                _cosine_similarity,
                                _has_directional_opposition,
                            )

                            sim = _cosine_similarity(new_scored.embedding, cand_scored.embedding)
                            if sim >= EMBEDDING_SAME_TOPIC_THRESHOLD:
                                is_opposition, signal = _has_directional_opposition(
                                    new_scored.summary, cand_scored.summary
                                )
                                if is_opposition:
                                    result = ContradictionPair(
                                        concept_a_id=new_scored.concept_id,
                                        concept_b_id=cand_scored.concept_id,
                                        contradiction_type=ContradictionType.HARD,
                                        detection_phase=2,
                                        similarity_score=sim,
                                        reason=f"Cross-session embedding opposition ({signal})",
                                    )
                        except ImportError:
                            pass

                # CONTRA-018: No-embedding fallback — when embeddings are unavailable,
                # use keyword overlap + directional opposition (Phase 2 logic without cosine).
                # Prevents silent regression for concepts ingested before embedding index exists.
                if result is None and (new_scored.embedding is None or cand_scored.embedding is None):
                    try:
                        from app.cognitive.contradiction import (
                            CROSS_TOPIC_OVERLAP_MIN,
                            ContradictionPair,
                            _compute_keyword_overlap_score,
                            _has_directional_opposition,
                        )

                        overlap = _compute_keyword_overlap_score(new_scored.summary, cand_scored.summary)
                        if overlap >= CROSS_TOPIC_OVERLAP_MIN:
                            is_opposition, signal = _has_directional_opposition(
                                new_scored.summary, cand_scored.summary
                            )
                            if is_opposition:
                                result = ContradictionPair(
                                    concept_a_id=new_scored.concept_id,
                                    concept_b_id=cand_scored.concept_id,
                                    contradiction_type=ContradictionType.HARD,
                                    detection_phase=2,
                                    reason=f"Cross-session keyword opposition, no embeddings ({signal})",
                                )
                    except ImportError:
                        pass

                if result is not None:
                    conflict_entry = {
                        "concept_id": cand_scored.concept_id,
                        "session_id": cand_dict.get("session_id", ""),
                        "type": result.contradiction_type.value
                        if hasattr(result.contradiction_type, "value")
                        else str(result.contradiction_type),
                        "reason": result.reason,
                        "phase": result.detection_phase,
                    }
                    report.conflicts.append(conflict_entry)

                    if result.contradiction_type == ContradictionType.HARD:
                        report.hard_count += 1
                    else:
                        report.soft_count += 1

            except Exception as e:
                logger.debug(f"FED-015: Candidate check failed (non-fatal): {e}")
                continue

        if report.hard_count > 0 or report.soft_count > 0:
            logger.info(
                f"FED-015: Write conflict detected — {report.hard_count} hard, "
                f"{report.soft_count} soft across {report.checked_count} candidates"
            )

    except Exception as e:
        logger.debug(f"FED-015: detect_write_conflict failed (non-fatal): {e}")

    report.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return report
