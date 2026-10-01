"""Auto-association pipeline for the Pith platform.

Provides batch and single-concept auto-association using TF-IDF cosine
similarity. Two-tier strategy:
  Tier 1 — Text similarity (cosine >= threshold) for all concepts
  Tier 2 — Domain-boosted (lower cosine + same knowledge_area) for orphans only

Created in Phase 1.3. All edges use "related_to" relation type.
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta, timezone

from app.core.metrics_facade import metrics
from app.core.models import (
    AutoAssociateBatchRequest,
    AutoAssociateBatchResponse,
    AutoAssociateMatch,
    AutoAssociateSingleRequest,
    AutoAssociateSingleResponse,
)
from app.retrieval import retrieval_engine
from app.storage import (
    add_association,
    count_orphan_concepts,
    get_associated_concept_ids,
    get_association_triples_for_pairs,
    get_knowledge_area_map_for_ids,
    get_metadata,
    load_concept,
    load_concepts_batch,
    load_unlinked_concept_window,
    set_metadata,
)

logger = logging.getLogger("pith.association")

_AUTO_ASSOCIATE_INVOCATION_SOURCES = frozenset({"api", "async_task", "maintenance", "direct"})
_UNLINKED_CURSOR_KEY = "association_discovery_unlinked_cursor_v1"
_UNLINKED_ASSOCIATION_LOCK = threading.Lock()


def _parse_unlinked_cursor(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        logger.warning("Ignoring malformed unlinked association cursor")
        return None
    if not isinstance(payload, dict):
        logger.warning("Ignoring non-object unlinked association cursor")
        return None
    created_at = payload.get("created_at")
    concept_id = payload.get("concept_id")
    if (
        not isinstance(created_at, str)
        or not created_at
        or len(created_at) > 64
        or not isinstance(concept_id, str)
        or not concept_id
        or len(concept_id) > 256
    ):
        logger.warning("Ignoring invalid unlinked association cursor fields")
        return None
    return created_at, concept_id


def _serialize_unlinked_cursor(created_at: str, concept_id: str) -> str:
    return json.dumps(
        {"concept_id": concept_id, "created_at": created_at},
        sort_keys=True,
        separators=(",", ":"),
    )


def _unlinked_age_buckets(rows: list[dict]) -> dict[str, int]:
    buckets = {"under_1d": 0, "1d_to_7d": 0, "7d_to_30d": 0, "over_30d": 0}
    now = datetime.now(UTC)
    for row in rows:
        try:
            created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age = now - created
        except (KeyError, TypeError, ValueError):
            continue
        if age < timedelta(days=1):
            buckets["under_1d"] += 1
        elif age < timedelta(days=7):
            buckets["1d_to_7d"] += 1
        elif age < timedelta(days=30):
            buckets["7d_to_30d"] += 1
        else:
            buckets["over_30d"] += 1
    return buckets


def _generate_unlinked_pairs(
    request: AutoAssociateBatchRequest,
    selected_rows: list[dict],
) -> tuple[list[tuple[str, str, float]], set[str], set[str], int]:
    """Generate bounded TF-IDF candidate pairs for selected unlinked concepts."""
    selected_ids = [row["concept_id"] for row in selected_rows]
    concept_map = load_concepts_batch(selected_ids)
    if selected_ids and not concept_map:
        raise RuntimeError("all selected unlinked association concepts were unavailable")
    unavailable_ids = {concept_id for concept_id in selected_ids if concept_id not in concept_map}
    lower_threshold = request.tier2_threshold if request.tier2_enabled else request.tier1_threshold
    pair_scores: dict[tuple[str, str], float] = {}
    pairs_evaluated = 0
    searches_succeeded = 0
    search_failures = 0

    for concept_id in selected_ids:
        concept = concept_map.get(concept_id)
        if concept is None:
            continue
        try:
            query_text = retrieval_engine._concept_to_document(concept)
            raw_results = retrieval_engine.index.search(
                query_text,
                top_k=request.max_candidates_per_concept + 1,
            )
        except Exception as exc:
            unavailable_ids.add(concept_id)
            search_failures += 1
            logger.warning("Unlinked association search failed for one concept: %s", type(exc).__name__)
            continue

        searches_succeeded += 1
        candidates_seen = 0
        for result_id, score_value in raw_results:
            if result_id == concept_id:
                continue
            if candidates_seen >= request.max_candidates_per_concept:
                break
            candidates_seen += 1
            pairs_evaluated += 1
            score = float(score_value)
            if score < lower_threshold:
                continue
            source, target = sorted((concept_id, result_id))
            pair_scores[(source, target)] = max(score, pair_scores.get((source, target), 0.0))

    if search_failures and searches_succeeded == 0:
        raise RuntimeError("all hydrated unlinked association searches failed")

    all_pairs = [
        (source, target, round(score, 4))
        for (source, target), score in pair_scores.items()
    ]
    all_pairs.sort(key=lambda pair: pair[2], reverse=True)
    return all_pairs, set(selected_ids), unavailable_ids, pairs_evaluated


def _auto_associate_metric_labels(
    invocation_source: str,
    batch_run_id: str,
    parent_run_id: str | int | None,
    **extra: str | int | float,
) -> dict[str, str | int | float]:
    source = invocation_source if invocation_source in _AUTO_ASSOCIATE_INVOCATION_SOURCES else "direct"
    labels: dict[str, str | int | float] = {
        "invocation_source": source,
        "batch_run_id": batch_run_id,
    }
    if parent_run_id is not None:
        labels["parent_run_id"] = str(parent_run_id)[:64]
    labels.update(extra)
    return labels


def _record_auto_associate_phase(
    phase: str,
    started_at: float,
    labels: dict[str, str | int | float],
) -> None:
    metrics.record(
        "auto_associate_batch_phase_latency_ms",
        round((time.perf_counter() - started_at) * 1000, 2),
        {**labels, "phase": phase},
    )


def auto_associate_batch(
    request: AutoAssociateBatchRequest,
    *,
    invocation_source: str = "direct",
    parent_run_id: str | int | None = None,
) -> AutoAssociateBatchResponse:
    """Run two-tier auto-association with candidate-bounded decision state."""
    lock = _UNLINKED_ASSOCIATION_LOCK if request.selection_mode == "unlinked" else nullcontext()
    with lock:
        return _auto_associate_batch_impl(
            request,
            invocation_source=invocation_source,
            parent_run_id=parent_run_id,
        )


def _auto_associate_batch_impl(
    request: AutoAssociateBatchRequest,
    *,
    invocation_source: str,
    parent_run_id: str | int | None,
) -> AutoAssociateBatchResponse:
    start_time = time.perf_counter()
    batch_run_id = uuid.uuid4().hex
    labels = _auto_associate_metric_labels(
        invocation_source,
        batch_run_id,
        parent_run_id,
        dry_run="true" if request.dry_run else "false",
        selection_mode=request.selection_mode,
    )

    phase_start = time.perf_counter()
    index_synced = retrieval_engine.sync_index()
    _record_auto_associate_phase("index_sync", phase_start, labels)

    phase_start = time.perf_counter()
    orphans_before = count_orphan_concepts()
    _record_auto_associate_phase("orphan_count_before", phase_start, labels)

    selected_rows: list[dict] = []
    selected_ids: set[str] = set()
    unavailable_ids: set[str] = set()
    concepts_available = 0
    concept_budget_exhausted = False
    cursor_start = None
    cursor_end = None
    age_buckets = {"under_1d": 0, "1d_to_7d": 0, "7d_to_30d": 0, "over_30d": 0}

    phase_start = time.perf_counter()
    if request.selection_mode == "unlinked":
        parsed_cursor = _parse_unlinked_cursor(get_metadata(_UNLINKED_CURSOR_KEY))
        if parsed_cursor:
            cursor_start = _serialize_unlinked_cursor(*parsed_cursor)
        window = load_unlinked_concept_window(
            request.max_concepts_evaluated,
            parsed_cursor[0] if parsed_cursor else None,
            parsed_cursor[1] if parsed_cursor else None,
        )
        selected_rows = list(window["rows"])
        concepts_available = int(window["available"])
        concept_budget_exhausted = concepts_available > len(selected_rows)
        age_buckets = _unlinked_age_buckets(selected_rows)
        candidate_search_started = time.perf_counter()
        all_pairs, selected_ids, unavailable_ids, pairs_evaluated = _generate_unlinked_pairs(
            request,
            selected_rows,
        )
        candidate_search_ms = round((time.perf_counter() - candidate_search_started) * 1000, 2)
        pairs_available = pairs_evaluated
        pair_budget_exhausted = False
        if selected_rows:
            last_row = selected_rows[-1]
            cursor_end = _serialize_unlinked_cursor(last_row["created_at"], last_row["concept_id"])
    else:
        lower_threshold = request.tier2_threshold if request.tier2_enabled else request.tier1_threshold
        candidate_search_started = time.perf_counter()
        all_pairs = retrieval_engine.pairwise_similarity(
            threshold=lower_threshold,
            max_pairs_evaluated=request.max_pairs_evaluated,
        )
        candidate_search_ms = round((time.perf_counter() - candidate_search_started) * 1000, 2)
        pairwise_stats = getattr(retrieval_engine, "_last_pairwise_similarity_stats", {}) or {}
        pairs_evaluated = int(pairwise_stats.get("pairs_evaluated", len(all_pairs)))
        pairs_available = int(pairwise_stats.get("pairs_available", pairs_evaluated))
        pair_budget_exhausted = bool(pairwise_stats.get("pair_budget_exhausted", False))
    _record_auto_associate_phase("pair_generation", phase_start, labels)

    candidate_ids = {concept_id for source, target, _ in all_pairs for concept_id in (source, target)}

    ka_map: dict[str, str | None] = {}
    if request.selection_mode == "unlinked" and candidate_ids:
        phase_start = time.perf_counter()
        ka_map = get_knowledge_area_map_for_ids(candidate_ids)
        all_pairs = [
            (source, target, score)
            for source, target, score in all_pairs
            if source in ka_map and target in ka_map
        ]
        candidate_ids = {
            concept_id
            for source, target, _ in all_pairs
            for concept_id in (source, target)
        }
        _record_auto_associate_phase("active_current_candidate_filter", phase_start, labels)

    phase_start = time.perf_counter()
    existing_edges = get_association_triples_for_pairs(
        (source, target, "related_to") for source, target, _ in all_pairs
    )
    _record_auto_associate_phase("duplicate_lookup", phase_start, labels)

    existing_participants: set[str] = set()
    if request.tier2_enabled:
        phase_start = time.perf_counter()
        existing_participants = get_associated_concept_ids(candidate_ids)
        _record_auto_associate_phase("participant_lookup", phase_start, labels)

        if not ka_map:
            phase_start = time.perf_counter()
            ka_map = get_knowledge_area_map_for_ids(candidate_ids)
            _record_auto_associate_phase("knowledge_area_lookup", phase_start, labels)

    edges_added_per_concept = defaultdict(int)
    tier1_edges_created = 0
    tier2_edges_created = 0
    edges_skipped_existing = 0
    edges_skipped_cap = 0
    edges_to_insert: list[tuple[str, str, float]] = []
    matched_selected_ids: set[str] = set()
    deferred_selected_ids: set[str] = set()
    evaluated_selected_ids = selected_ids - unavailable_ids

    phase_start = time.perf_counter()
    tier1_pairs = [(s, t, score) for s, t, score in all_pairs if score >= request.tier1_threshold]

    for source, target, score in tier1_pairs:
        triple = (source, target, "related_to")
        if triple in existing_edges:
            edges_skipped_existing += 1
            continue

        if (
            edges_added_per_concept[source] >= request.max_edges_per_concept
            or edges_added_per_concept[target] >= request.max_edges_per_concept
        ):
            edges_skipped_cap += 1
            deferred_selected_ids.update({source, target} & evaluated_selected_ids)
            continue

        strength = round(min(score, 0.80), 3)
        edges_to_insert.append((source, target, strength))
        existing_edges.add(triple)
        edges_added_per_concept[source] += 1
        edges_added_per_concept[target] += 1
        tier1_edges_created += 1
        matched_selected_ids.update({source, target} & evaluated_selected_ids)

    if request.tier2_enabled:
        new_edge_participants = {
            concept_id
            for source, target, _ in edges_to_insert
            for concept_id in (source, target)
        }
        still_orphan_ids = candidate_ids - existing_participants - new_edge_participants

        tier2_pairs = [
            (s, t, score)
            for s, t, score in all_pairs
            if score < request.tier1_threshold
            and score >= request.tier2_threshold
            and (s in still_orphan_ids or t in still_orphan_ids)
            and ka_map.get(s) == ka_map.get(t)
            and ka_map.get(s) is not None  # Don't match on None/None
        ]

        for source, target, score in tier2_pairs:
            triple = (source, target, "related_to")
            if triple in existing_edges:
                edges_skipped_existing += 1
                continue

            if (
                edges_added_per_concept[source] >= request.max_edges_per_concept
                or edges_added_per_concept[target] >= request.max_edges_per_concept
            ):
                edges_skipped_cap += 1
                deferred_selected_ids.update({source, target} & evaluated_selected_ids)
                continue

            strength = round(min(score * 0.8, 0.80), 3)
            edges_to_insert.append((source, target, strength))
            existing_edges.add(triple)
            edges_added_per_concept[source] += 1
            edges_added_per_concept[target] += 1
            tier2_edges_created += 1
            matched_selected_ids.update({source, target} & evaluated_selected_ids)
    _record_auto_associate_phase("edge_decisions", phase_start, labels)

    phase_start = time.perf_counter()
    if not request.dry_run:
        for source, target, strength in edges_to_insert:
            add_association(source, target, "related_to", strength)
        if request.selection_mode == "unlinked" and cursor_end is not None:
            set_metadata(_UNLINKED_CURSOR_KEY, cursor_end)
    _record_auto_associate_phase("persistence", phase_start, labels)

    phase_start = time.perf_counter()
    orphans_after = count_orphan_concepts() if not request.dry_run else orphans_before
    _record_auto_associate_phase("orphan_count_after", phase_start, labels)
    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 1)

    deferred_selected_ids -= matched_selected_ids
    concepts_evaluated = len(evaluated_selected_ids)
    concepts_matched = len(matched_selected_ids)
    concepts_deferred = len(deferred_selected_ids)
    concepts_no_match = concepts_evaluated - concepts_matched - concepts_deferred
    concepts_unavailable = len(unavailable_ids)

    metrics.record("auto_associate_batch_latency_ms", processing_time_ms, labels)
    metrics.record("auto_associate_batch_pairs_evaluated", pairs_evaluated, labels)
    metrics.record("auto_associate_batch_pairs_available", pairs_available, labels)
    metrics.record("auto_associate_batch_candidate_count", len(all_pairs), labels)
    metrics.record("auto_associate_batch_candidate_endpoint_count", len(candidate_ids), labels)
    metrics.record("auto_associate_batch_candidate_search_latency_ms", candidate_search_ms, labels)
    metrics.record("auto_associate_batch_concepts_available", concepts_available, labels)
    metrics.record("auto_associate_batch_concepts_evaluated", concepts_evaluated, labels)
    metrics.record("auto_associate_batch_concepts_matched", concepts_matched, labels)
    metrics.record("auto_associate_batch_concepts_no_match", concepts_no_match, labels)
    metrics.record("auto_associate_batch_concepts_deferred", concepts_deferred, labels)
    metrics.record("auto_associate_batch_concepts_unavailable", concepts_unavailable, labels)
    metrics.record("auto_associate_batch_concept_budget_exhausted", int(concept_budget_exhausted), labels)
    for bucket, count in age_buckets.items():
        metrics.record(
            "auto_associate_batch_selected_age_count",
            count,
            {**labels, "age_bucket": bucket},
        )
    if pair_budget_exhausted:
        metrics.record("auto_associate_batch_pair_budget_exhausted", 1, labels)

    logger.info(
        "auto_associate_batch[%s]: source=%s mode=%s dry_run=%s T1=%d, T2=%d, "
        "skipped_existing=%d, skipped_cap=%d, pairs=%d/%d, pair_budget_exhausted=%s, "
        "concepts=%d/%d matched=%d no_match=%d deferred=%d unavailable=%d, "
        "orphans %d->%d, %.1fms",
        batch_run_id,
        labels["invocation_source"],
        request.selection_mode,
        labels["dry_run"],
        tier1_edges_created,
        tier2_edges_created,
        edges_skipped_existing,
        edges_skipped_cap,
        pairs_evaluated,
        pairs_available,
        pair_budget_exhausted,
        concepts_evaluated,
        concepts_available,
        concepts_matched,
        concepts_no_match,
        concepts_deferred,
        concepts_unavailable,
        orphans_before,
        orphans_after,
        processing_time_ms,
    )

    return AutoAssociateBatchResponse(
        index_synced=index_synced,
        pairs_evaluated=pairs_evaluated,
        pairs_available=pairs_available,
        pair_budget_exhausted=pair_budget_exhausted,
        tier1_edges_created=tier1_edges_created,
        tier2_edges_created=tier2_edges_created,
        edges_skipped_existing=edges_skipped_existing,
        edges_skipped_cap=edges_skipped_cap,
        orphans_before=orphans_before,
        orphans_after=orphans_after,
        processing_time_ms=processing_time_ms,
        dry_run=request.dry_run,
        selection_mode=request.selection_mode,
        concepts_available=concepts_available,
        concepts_evaluated=concepts_evaluated,
        concepts_matched=concepts_matched,
        concepts_no_match=concepts_no_match,
        concepts_deferred=concepts_deferred,
        concepts_unavailable=concepts_unavailable,
        concept_budget_exhausted=concept_budget_exhausted,
        cursor_start=cursor_start,
        cursor_end=cursor_end,
    )


def auto_associate_single(concept_id: str, request: AutoAssociateSingleRequest, cached_triples: set | None = None) -> AutoAssociateSingleResponse:
    """Auto-associate a single concept with its nearest neighbors.

    Loads the concept's summary text, searches for similar concepts via
    TF-IDF, then creates "related_to" edges for matches above the threshold.
    Respects edge cap and existing edges.
    """
    start_time = time.time()

    # Load concept to get its summary text for the search query
    concept = load_concept(concept_id, track_access=False)
    if not concept:
        return AutoAssociateSingleResponse(
            concept_id=concept_id,
            edges_created=0,
            edges_skipped_existing=0,
            matches=[],
            processing_time_ms=round((time.time() - start_time) * 1000, 1),
        )

    # Use raw TF-IDF index search (not full retrieval pipeline) for speed.
    # This is consistent with batch which also uses raw cosine scores.
    # CRITICAL: Use full document text (same as what's indexed) not just summary.
    # summary-only queries produce systematically lower cosine scores vs the
    # pairwise matrix which compares full indexed vectors.
    query_text = retrieval_engine._concept_to_document(concept)
    raw_results = retrieval_engine.index.search(query_text, top_k=request.max_edges + 5)

    # RETRIEVAL-042: Supplement TF-IDF with embedding search for cross-domain association.
    # TF-IDF cannot link concepts sharing no terms (e.g., "shellfish allergy" ↔ "Dr. Amara Osei").
    # Embedding cosine captures semantic similarity that TF-IDF misses entirely.
    EMBEDDING_ASSOC_THRESHOLD = 0.35  # Cross-domain pairs score 0.35-0.45; TF-IDF threshold is 0.12
    try:
        from app.storage.embedding import embedding_engine
        if embedding_engine.is_available and embedding_engine.index_size > 0:
            summary_text = getattr(concept, "summary", "") or query_text
            emb_results = embedding_engine.search(summary_text, top_k=request.max_edges + 5)
            # Merge: build dict of concept_id → max(tfidf_score, emb_score)
            merged = {cid: score for cid, score in raw_results}
            for cid, emb_score in emb_results:
                if emb_score >= EMBEDDING_ASSOC_THRESHOLD:
                    merged[cid] = max(merged.get(cid, 0.0), emb_score)
            # Re-sort by score descending
            raw_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.debug(f"RETRIEVAL-042: Embedding association failed (fallback to TF-IDF): {e}")

    candidate_triples = []
    for result_id, score in raw_results:
        if result_id == concept_id:
            continue
        if score < request.threshold:
            continue
        source, target = sorted([concept_id, result_id])
        candidate_triples.append((source, target, "related_to"))

    existing_edges = get_association_triples_for_pairs(candidate_triples)
    if cached_triples is not None:
        existing_edges.update(cached_triples)
    matches = []
    edges_created = 0
    edges_skipped_existing = 0

    for result_id, score in raw_results:
        if result_id == concept_id:
            continue
        if score < request.threshold:
            continue
        if edges_created >= request.max_edges:
            break

        # Normalize direction for triple check
        source, target = sorted([concept_id, result_id])
        triple = (source, target, "related_to")
        already_exists = triple in existing_edges

        if already_exists:
            if cached_triples is not None:
                cached_triples.add(triple)
            edges_skipped_existing += 1
            matches.append(
                AutoAssociateMatch(
                    target_id=result_id,
                    cosine_score=round(score, 4),
                    edge_created=False,
                )
            )
        else:
            strength = round(min(score, 0.80), 3)
            add_association(concept_id, result_id, "related_to", strength)
            existing_edges.add(triple)
            if cached_triples is not None:
                cached_triples.add(triple)
            edges_created += 1
            matches.append(
                AutoAssociateMatch(
                    target_id=result_id,
                    cosine_score=round(score, 4),
                    edge_created=True,
                )
            )

    processing_time_ms = round((time.time() - start_time) * 1000, 1)

    logger.info(
        f"auto_associate_single: {concept_id} — {edges_created} created, "
        f"{edges_skipped_existing} existing, {processing_time_ms}ms"
    )

    return AutoAssociateSingleResponse(
        concept_id=concept_id,
        edges_created=edges_created,
        edges_skipped_existing=edges_skipped_existing,
        matches=matches,
        processing_time_ms=processing_time_ms,
    )


def prune_weak_intra_ka_associations(
    strength_threshold: float = 0.2,
    dry_run: bool = True,
) -> dict:
    """ARCH-O07: Prune weak intra-KA batch associations.

    Removes associations where:
    - mechanism IS NULL (auto_associate_batch)
    - strength < threshold
    - source and target are in the SAME knowledge_area

    Preserves all cross-KA associations (domain bridges) regardless of strength.
    """
    from app.storage import _db

    start_time = time.time()

    with _db() as conn:
        # Count what would be pruned
        count_row = conn.execute(
            """SELECT COUNT(*) FROM associations a
               JOIN concepts c1 ON a.source = c1.id
               JOIN concepts c2 ON a.target = c2.id
               WHERE a.mechanism IS NULL
               AND a.strength < ?
               AND c1.knowledge_area = c2.knowledge_area""",
            (strength_threshold,),
        ).fetchone()
        prune_count = count_row[0]

        if not dry_run and prune_count > 0:
            conn.execute(
                """DELETE FROM associations WHERE rowid IN (
                    SELECT a.rowid FROM associations a
                    JOIN concepts c1 ON a.source = c1.id
                    JOIN concepts c2 ON a.target = c2.id
                    WHERE a.mechanism IS NULL
                    AND a.strength < ?
                    AND c1.knowledge_area = c2.knowledge_area
                )""",
                (strength_threshold,),
            )

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    action = "pruned" if not dry_run else "would_prune"
    logger.info(f"ARCH-O07: {action} {prune_count} weak intra-KA associations ({elapsed_ms}ms)")

    return {
        "action": action,
        "pruned_count": prune_count,
        "strength_threshold": strength_threshold,
        "elapsed_ms": elapsed_ms,
    }


# =============================================================================
# RETRIEVAL-041: Decision Domain Bridge — Cross-Domain governs Edges
# =============================================================================

# KAs where DECISION concepts should create cross-domain governs edges
_DECISION_BRIDGE_SOURCE_KAS = frozenset({
    "product_strategy",
    "business_strategy",
    "strategic_recommendation",
    "strategy",
})

# KAs that are "downstream" of strategic decisions — task-domain queries that
# should be able to reach strategic DECISION concepts via S4 graph walk
_DECISION_BRIDGE_TARGET_KAS = frozenset({
    "competitive_analysis",
    "gtm_strategy",
    "marketing_discipline",
    "product_positioning",
    "implementation",
    "pith_engineering",
})

_DECISION_BRIDGE_AUTHORITY_FLOOR = 0.6   # Only wire high-authority decisions
_DECISION_BRIDGE_STRENGTH = 0.70         # governs edge weight (S4 uses as score multiplier)
_DECISION_BRIDGE_LIMIT = 10              # Max targets per decision concept
_DECISION_BRIDGE_RECENCY_DAYS = 14       # Only wire to recently-accessed targets


def auto_associate_decision_concept(concept_id: str, concept) -> int:
    """RETRIEVAL-041: Create governs edges from high-authority DECISION concepts to
    recently-active downstream KA concepts.

    Called at write-time after a DECISION concept is created in a strategic KA.
    Ensures that task-domain S4 graph walks can reach strategic decisions even when
    embedding similarity is low (different vocabulary, different domain).

    Without these edges, S4's 1-hop shadow expansion never crosses from benchmark/
    implementation domains into product_strategy/business_strategy domains — the
    root cause of the 2026-03-20 live session incident.

    Returns: count of governs edges created.
    """
    concept_type = getattr(concept, "concept_type", "") or ""
    if concept_type != "decision":
        return 0

    ka = getattr(concept, "knowledge_area", "") or ""
    if ka not in _DECISION_BRIDGE_SOURCE_KAS:
        return 0

    # Authority check — only bridge high-authority decisions
    authority = getattr(concept, "authority_score", None)
    if authority is None:
        # May be stored in metadata for newly-created concepts
        meta = getattr(concept, "metadata", {}) or {}
        authority = meta.get("authority_score", 0.0) or 0.0
    if (authority or 0.0) < _DECISION_BRIDGE_AUTHORITY_FLOOR:
        # New concepts start low; allow through if authority is unset (None/0)
        # — governance recompute will raise it. Skip only explicit low values.
        if authority is not None and authority > 0.0:
            logger.debug(
                "RETRIEVAL-041: Skipping %s — authority %.3f below floor %.3f",
                concept_id, authority, _DECISION_BRIDGE_AUTHORITY_FLOOR,
            )
            return 0

    try:
        from app.core.datetime_utils import _utc_now_iso
        from app.storage import _get_connection, _invalidate_associations_cache

        conn = _get_connection()
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=_DECISION_BRIDGE_RECENCY_DAYS)
        ).isoformat()

        ka_placeholders = ",".join("?" * len(_DECISION_BRIDGE_TARGET_KAS))
        downstream_rows = conn.execute(
            f"""SELECT id FROM concepts
                WHERE knowledge_area IN ({ka_placeholders})
                  AND currency_status NOT IN ('SUPERSEDED', 'STALE', 'DISCARDED')
                  AND is_current = 1
                  AND status = 'active'
                  AND last_organic_access > ?
                ORDER BY last_organic_access DESC
                LIMIT ?""",
            (*_DECISION_BRIDGE_TARGET_KAS, cutoff, _DECISION_BRIDGE_LIMIT),
        ).fetchall()

        if not downstream_rows:
            logger.debug("RETRIEVAL-041: No recent downstream concepts found for %s", concept_id)
            return 0

        now = _utc_now_iso()
        edges_created = 0

        for row in downstream_rows:
            target_id = row[0]
            # Idempotent: skip if governs edge already exists (PK: source, target, relation)
            existing = conn.execute(
                "SELECT 1 FROM associations WHERE source = ? AND target = ? AND relation = 'governs'",
                (concept_id, target_id),
            ).fetchone()
            if existing:
                continue

            conn.execute(
                """INSERT INTO associations
                   (source, target, relation, strength, created_at, mechanism)
                   VALUES (?, ?, 'governs', ?, ?, 'decision_domain_bridge')""",
                (concept_id, target_id, _DECISION_BRIDGE_STRENGTH, now),
            )
            edges_created += 1

        if edges_created > 0:
            conn.commit()
            _invalidate_associations_cache()
            logger.info(
                "RETRIEVAL-041: Created %d governs edges from DECISION %s to downstream KAs",
                edges_created,
                concept_id,
            )

        return edges_created

    except Exception as e:
        logger.warning(
            "RETRIEVAL-041: auto_associate_decision_concept failed for %s (non-fatal): %s",
            concept_id, e,
        )
        return 0
