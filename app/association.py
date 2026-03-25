"""Auto-association pipeline for the Pith platform.

Provides batch and single-concept auto-association using TF-IDF cosine
similarity. Two-tier strategy:
  Tier 1 — Text similarity (cosine >= threshold) for all concepts
  Tier 2 — Domain-boosted (lower cosine + same knowledge_area) for orphans only

Created in Phase 1.3. All edges use "related_to" relation type.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from app.metrics import metrics
from app.models import (
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
    get_all_association_triples,
    get_knowledge_area_map,
    list_concepts,
    load_concept,
)

logger = logging.getLogger("pith.association")


def auto_associate_batch(request: AutoAssociateBatchRequest) -> AutoAssociateBatchResponse:
    """Run two-tier auto-association pipeline on all active concepts.

    Steps:
      A. Sync TF-IDF index (ensure all concepts indexed)
      B. Compute pairwise cosine similarity at tier2 threshold (captures all candidates)
      C. Tier 1 pass — text similarity edges (cosine >= tier1_threshold)
      D. Tier 2 pass — domain-boosted edges for remaining orphans
      E. Bulk insert edges (or skip if dry_run)
      F. Return stats
    """
    start_time = time.time()

    # --- Step A: Index sync ---
    index_synced = retrieval_engine.sync_index()

    # --- Baseline metrics ---
    orphans_before = count_orphan_concepts()
    existing_edges = get_all_association_triples()

    # --- Step B: Compute pairwise similarity ---
    # Use the lower threshold to capture all candidate pairs for both tiers
    lower_threshold = request.tier2_threshold if request.tier2_enabled else request.tier1_threshold
    all_pairs = retrieval_engine.pairwise_similarity(threshold=lower_threshold)
    pairs_evaluated = len(all_pairs)

    # --- Build knowledge_area map for Tier 2 ---
    ka_map = {}
    if request.tier2_enabled:
        ka_map = get_knowledge_area_map()

    # --- Track edges per concept for cap enforcement ---
    edges_added_per_concept = defaultdict(int)

    # Stats accumulators
    tier1_edges_created = 0
    tier2_edges_created = 0
    edges_skipped_existing = 0
    edges_skipped_cap = 0
    edges_to_insert = []  # List of (source, target, strength) for deferred insert

    # --- Step C: Tier 1 pass ---
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
            continue

        strength = round(min(score, 0.80), 3)
        edges_to_insert.append((source, target, strength))
        existing_edges.add(triple)  # Prevent Tier 2 from re-adding
        edges_added_per_concept[source] += 1
        edges_added_per_concept[target] += 1
        tier1_edges_created += 1

    # --- Step D: Tier 2 pass (orphan rescue) ---
    if request.tier2_enabled:
        # Identify concepts still orphaned after Tier 1
        # A concept is "rescued" if it gained any edge in Tier 1
        concepts_with_new_edges = set()
        for source, target, _ in edges_to_insert:
            concepts_with_new_edges.add(source)
            concepts_with_new_edges.add(target)

        # Get concepts that are STILL orphans: no existing edges AND no new edges
        # We need to check against the original existing_edges set + new edges
        still_orphan_ids = set()
        all_edge_participants = set()
        for s, t, r in existing_edges:
            all_edge_participants.add(s)
            all_edge_participants.add(t)

        for cid in list_concepts():
            if cid not in all_edge_participants:
                still_orphan_ids.add(cid)

        # Tier 2 pairs: below tier1 but above tier2, AND same knowledge_area,
        # AND at least one side is still orphaned
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
                continue

            # Tier 2 strength is discounted (weaker text signal)
            strength = round(min(score * 0.8, 0.80), 3)
            edges_to_insert.append((source, target, strength))
            existing_edges.add(triple)
            edges_added_per_concept[source] += 1
            edges_added_per_concept[target] += 1
            tier2_edges_created += 1

    # --- Step E: Bulk insert ---
    if not request.dry_run:
        for source, target, strength in edges_to_insert:
            add_association(source, target, "related_to", strength)

    # --- Step F: Final metrics ---
    orphans_after = count_orphan_concepts() if not request.dry_run else orphans_before
    processing_time_ms = round((time.time() - start_time) * 1000, 1)

    # OBS-03: emit association batch latency to metrics DB
    metrics.record("auto_associate_batch_latency_ms", processing_time_ms)

    logger.info(
        f"auto_associate_batch: T1={tier1_edges_created}, T2={tier2_edges_created}, "
        f"skipped_existing={edges_skipped_existing}, skipped_cap={edges_skipped_cap}, "
        f"orphans {orphans_before}→{orphans_after}, {processing_time_ms}ms"
    )

    return AutoAssociateBatchResponse(
        index_synced=index_synced,
        pairs_evaluated=pairs_evaluated,
        tier1_edges_created=tier1_edges_created,
        tier2_edges_created=tier2_edges_created,
        edges_skipped_existing=edges_skipped_existing,
        edges_skipped_cap=edges_skipped_cap,
        orphans_before=orphans_before,
        orphans_after=orphans_after,
        processing_time_ms=processing_time_ms,
        dry_run=request.dry_run,
    )


def auto_associate_single(concept_id: str, request: AutoAssociateSingleRequest) -> AutoAssociateSingleResponse:
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
        from app.embedding import embedding_engine
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

    # Get existing edges for dedup
    existing_edges = get_all_association_triples()
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
    "pith_benchmarks",
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
        from app.storage import _get_connection, _invalidate_associations_cache
        from app.datetime_utils import _utc_now_iso

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
