"""
Integration Adapter for Pith Retrieval Engine
Provides backward-compatible interface for incremental TF-IDF index

CRITICAL: This is a DROP-IN REPLACEMENT for retrieval.py
Interface must match exactly for seamless integration.

P0.3: Hybrid architecture — embeddings for search, TF-IDF for dedup/auto-association.
"""

import functools
import os
import json
import logging
import math
from pathlib import Path

import numpy as np

from app.datetime_utils import _ensure_aware, _utc_now
from app.embedding import EMBEDDING_DIM, EMBEDDING_VERSION, embedding_engine
from app.incremental_tfidf import IncrementalTfidfIndex
from app.models import SearchQuery, SearchResult
from app.storage import (  # DEBT-022: _db hoisted from function-level
    INDEX_DIR,
    _db,
    list_concepts,
    list_concepts_full,
    load_concept,
)

# Import governance scoring config
try:
    from app.config import (
        KA_BOOST_WEIGHT,
        MIN_RETRIEVAL_SIMILARITY,  # RETRIEVAL-031
        RETRIEVAL_WEIGHT_AUTHORITY,
        RETRIEVAL_WEIGHT_CONFIDENCE,
        RETRIEVAL_WEIGHT_CONTEXT,
        RETRIEVAL_WEIGHT_CURRENCY,
        RETRIEVAL_WEIGHT_GOAL,
        RETRIEVAL_WEIGHT_SIMILARITY,  # DEBT-002: renamed from RETRIEVAL_WEIGHT_EMBEDDING
        RETRIEVAL_WEIGHT_STABILITY,
        get_feature_flag,
    )

    GOVERNANCE_SCORING = True
except ImportError:
    GOVERNANCE_SCORING = False
    KA_BOOST_WEIGHT = 0.2

# Import activation modules for enhanced retrieval
try:
    from app.goal_directed import goal_directed
    from app.predictive import predictive_activation

    ENHANCED_RETRIEVAL = True
except ImportError:
    ENHANCED_RETRIEVAL = False
    predictive_activation = None
    goal_directed = None

logger = logging.getLogger(__name__)


# INGEST-015: Fact-seeking query detection + boost constant
FACT_SEEKING_BOOST = 1.25

# INGEST-015 legacy markers (fallback when structural classifier disabled)
_FACT_QUERY_MARKERS = [
    "what is my ", "what's my ", "where do i ", "where am i ",
    "what do i do", "where do i work", "who is my ", "who's my ",
    "what is their ", "where does ", "remind me ", "do you know my ",
    "what's the name", "where do i live", "what city am i",
    "tell me about my ", "what company", "who do i work",
]


def _is_fact_seeking_query(query_text: str) -> bool:
    """Detect personal-fact-seeking queries for retrieval boost.

    INGEST-017: Uses structural query classifier when enabled,
    falls back to INGEST-015 markers when disabled.
    """
    try:
        if get_feature_flag("STRUCTURAL_QUERY_CLASSIFIER_ENABLED", True):
            from app.fact_classifier import is_fact_seeking_query
            return is_fact_seeking_query(query_text)
    except Exception:
        logger.debug("INGEST-017: structural query classifier unavailable, using marker fallback")

    q = query_text.lower()
    return any(m in q for m in _FACT_QUERY_MARKERS)


# ======================================================================
# KA-ARCH-001 Fix 7: Module-level KA inference (lru_cached)
# ======================================================================

@functools.lru_cache(maxsize=128)
def _infer_query_kas(query_text: str) -> tuple[str, ...]:
    """Infer relevant KAs for a query using keyword + embedding match.

    Returns a tuple of 1-3 KA names most relevant to the query text.
    Cached per unique query text to avoid recomputation across N concepts.

    Called from:
      - search_lightweight() Phase 1.5 (conversation_turn hot path)
      - _apply_ka_boost() in search() path
    """
    from app.taxonomy import infer_knowledge_area, classify_ka_by_embedding

    kas = set()

    # Tier 1: keyword inference (0ms — pure string matching)
    kw_ka = infer_knowledge_area(query_text)
    if kw_ka:
        kas.add(kw_ka)

    # Tier 2: embedding inference (if available, ~2ms)
    try:
        emb_ka, emb_score, emb_gap = classify_ka_by_embedding(query_text)
        if emb_ka and emb_score >= 0.45:  # Higher threshold for query inference
            kas.add(emb_ka)
    except Exception:
        pass

    return tuple(kas) if kas else ()


class RetrievalEngine:
    """
    Incremental TF-IDF based retrieval system.

    DROP-IN REPLACEMENT for original retrieval.py with 100-1000× performance improvement.

    Key differences from old system:
    - O(V_doc) add/update/remove vs O(N) full rebuild
    - Automatic IDF quality improvement every 50 changes
    - Thread-safe with explicit locking
    - Checkpointed every 100 operations
    """

    def __init__(self):
        """
        Initialize retrieval engine.

        Matches exact interface of original retrieval.py for drop-in replacement.
        """
        # Use INDEX_DIR from storage.py (same as original)
        self.index_path = str(INDEX_DIR / "incremental")
        self.index = IncrementalTfidfIndex()
        self._embeddings_initialized = False
        self._embeddings_available = False  # Set True only if init succeeds

        # Try to load existing index
        index_dir = Path(self.index_path)
        if index_dir.exists():
            try:
                self.index.load(self.index_path)
                logger.info(f"Loaded incremental index: {self.index.document_count} concepts")
            except Exception as e:
                logger.warning(f"Could not load index: {e}. Starting fresh.")
        else:
            logger.info("No existing index found, will build on first search/add")

        logger.info("RetrievalEngine initialized (incremental TF-IDF + embeddings)")

    def _init_embeddings(self):
        """Load or compute embeddings for all active concepts.

        Loads existing embeddings from SQLite (where embedding_version matches),
        batch-embeds any concepts missing embeddings, persists new embeddings,
        and builds the in-memory search index.

        EMBEDDING_RESILIENCE_SPEC v1.1: If sentence_transformers is unavailable,
        sets _embeddings_available=False and returns immediately. TF-IDF search
        remains fully functional as fallback.
        """
        if self._embeddings_initialized:
            return

        # Check if embeddings are even available
        if not embedding_engine.is_available:
            logger.info("Embeddings unavailable — skipping embedding init, TF-IDF only")
            self._embeddings_initialized = True
            self._embeddings_available = False
            return

        from app.storage import _db

        # Load concepts with and without embeddings
        # Memory Integrity Spec v1.2, §5.1.1: Exclude DISCARDED concepts from retrieval
        # RETRIEVAL-014 Layer 1b: Exclude SUPERSEDED concepts from index loading.
        # Defense-in-depth: Layer 1c evicts on supersession, but this prevents
        # re-entry on restart. _governance_score (line 705) also hard-filters,
        # but loading them wastes memory and compute.
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, embedding, embedding_version, data FROM concepts WHERE status = 'active' AND maturity != 'DISCARDED' AND is_current = 1"
            ).fetchall()

        if not rows:
            self._embeddings_initialized = True
            return

        existing_ids = []
        existing_embeddings = []
        needs_embedding = []  # (concept_id, searchable_text)

        for row in rows:
            cid = row["id"]
            if row["embedding"] and row["embedding_version"] == EMBEDDING_VERSION:
                # Load pre-computed embedding from BLOB
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                if emb.shape[0] == EMBEDDING_DIM:
                    existing_ids.append(cid)
                    existing_embeddings.append(emb)
                    continue

            # Need to compute embedding — extract searchable text
            import json

            try:
                data = json.loads(row["data"])
                concept = type(
                    "C",
                    (),
                    {
                        "summary": data.get("summary", ""),
                        "signals": data.get("signals", []),
                        "evidence": data.get("evidence", []),
                        "metadata": data.get("metadata", {}),
                        "hypotheses": [],
                    },
                )()
                text = self._concept_to_document(concept)
                needs_embedding.append((cid, text))
            except Exception as e:
                logger.warning(f"_init_embeddings: failed to parse {cid}: {e}")

        # Batch-embed missing concepts
        if needs_embedding:
            logger.info(f"Embedding {len(needs_embedding)} concepts (batch)...")
            texts = [text for _, text in needs_embedding]
            new_embeddings = embedding_engine.embed_batch(texts)

            # Persist to SQLite
            for i, (cid, _) in enumerate(needs_embedding):
                emb = new_embeddings[i]
                conn.execute(
                    "UPDATE concepts SET embedding = ?, embedding_version = ? WHERE id = ?",
                    (emb.tobytes(), EMBEDDING_VERSION, cid),
                )
                existing_ids.append(cid)
                existing_embeddings.append(emb)
            conn.commit()
            logger.info(f"Persisted {len(needs_embedding)} new embeddings to SQLite")

        # Build in-memory index
        if existing_embeddings:
            matrix = np.vstack(existing_embeddings)
            embedding_engine.build_index(existing_ids, matrix)

        self._embeddings_initialized = True
        self._embeddings_available = True
        logger.info(f"Embedding index ready: {len(existing_ids)} concepts")

    def add_concept(self, concept_id: str):
        """
        Add single concept to index (incremental).

        INTERFACE MATCH: Takes concept_id (string) just like original retrieval.py

        Performance: O(V_doc) vs O(N) full rebuild - 100-1000× faster

        Args:
            concept_id: ID of concept to add
        """
        # FIX-2: Direct SQL instead of load_concept() to avoid Pydantic crash
        # on concepts with incomplete data JSON blobs.
        from app.storage import _db

        with _db() as _conn:
            _row = _conn.execute(
                "SELECT data, summary FROM concepts WHERE id = ? AND is_current = 1",
                (concept_id,),
            ).fetchone()
        if not _row:
            logger.warning(f"Concept {concept_id} not found, skipping index add")
            return

        try:
            _data = json.loads(_row[0]) if _row[0] else {}
        except (json.JSONDecodeError, TypeError):
            _data = {}

        # Build searchable text from available data (no Pydantic required)
        _summary = _data.get("summary", "") or (_row[1] if _row[1] else "")
        _evidence_strs = []
        for _e in _data.get("evidence") or []:
            if isinstance(_e, str):
                _evidence_strs.append(_e)
            elif isinstance(_e, dict):
                _evidence_strs.append(_e.get("content", ""))
        _signals = _data.get("signals", [])
        _ka = _data.get("metadata", {}).get("knowledge_area", "") if isinstance(_data.get("metadata"), dict) else ""
        _ctype = _data.get("concept_type", "")

        # RETRIEVAL-057: Include prospective indexing implications
        _implications = _data.get("implications", [])
        _implications_text = " ".join(str(imp) for imp in _implications if isinstance(imp, str))

        # INGEST-034: Include event text in searchable content
        _event_texts = []
        for _evt in _data.get("events", []):
            _evt_parts = [_evt.get("action", "")]
            if _evt.get("cause"):
                _evt_parts.append(f"because {_evt['cause']}")
            if _evt.get("consequence"):
                _evt_parts.append(f"resulting in {_evt['consequence']}")
            if _evt.get("actors"):
                _evt_parts.append(f"involving {', '.join(_evt['actors'])}")
            _event_texts.append(" ".join(_evt_parts))

        # INGEST-037 Layer 4: Include fragment keywords in searchable text
        _frag_kw = _data.get("fragment_keywords", "") or ""
        if not _frag_kw:
            # Fallback: read column directly (fragment_keywords not in JSON blob)
            try:
                with _db() as _fk_conn:
                    _fk_row = _fk_conn.execute(
                        "SELECT fragment_keywords FROM concepts WHERE id = ?",
                        (concept_id,),
                    ).fetchone()
                    _frag_kw = _fk_row[0] if _fk_row and _fk_row[0] else ""
            except Exception:
                _frag_kw = ""  # Column may not exist yet (pre-migration)

        searchable_text = " ".join(
            filter(None, [
                _summary, _ka, _ctype,
                " ".join(_evidence_strs),
                " ".join(str(s) for s in _signals),
                _implications_text,  # RETRIEVAL-057
                " ".join(_event_texts),  # INGEST-034
                _frag_kw,  # INGEST-037 Layer 4
            ])
        )
        if not searchable_text.strip():
            logger.warning(f"Concept {concept_id} has no searchable content, skipping")
            return

        # Add incrementally (O(V_doc) operation)
        success = self.index.add_concept(concept_id, searchable_text)

        if success:
            logger.debug(f"Added concept {concept_id} to index (incremental)")

            # Auto-save every 10 operations
            if self.index.index_version % 10 == 0:
                self._auto_save()
        else:
            logger.debug(f"Concept {concept_id} already in index")

        # P0.3: Also compute and store embedding
        if self._embeddings_initialized:
            try:
                emb = embedding_engine.embed_text(searchable_text)
                # STABILITY-008: Validate embedding norm before persisting
                # Prevents all-zero or corrupt embeddings (KA-003 incident: 69% all-zero)
                import numpy as np

                emb_norm = float(np.linalg.norm(emb))
                if emb_norm < 0.5:
                    logger.warning(
                        f"STABILITY-008: Rejecting embedding for {concept_id} (norm={emb_norm:.4f} < 0.5 threshold)"
                    )
                else:
                    embedding_engine.update_embedding(concept_id, emb)
                    # COLD-START-FIX: Unlock embedding dedup after first successful add.
                    # _init_embeddings() on empty DB sets _embeddings_initialized=True
                    # but not _embeddings_available=True — this flips it once the index
                    # has at least one entry. See MEASURE-026 §14.
                    if not self._embeddings_available:
                        self._embeddings_available = True
                        logger.info(
                            "COLD-START-FIX: _embeddings_available set True "
                            f"after add_concept({concept_id})"
                        )
                    # Persist to SQLite
                    from app.storage import _db_immediate

                    with _db_immediate() as conn:
                        conn.execute(
                            "UPDATE concepts SET embedding = ?, embedding_version = ? WHERE id = ?",
                            (emb.tobytes(), EMBEDDING_VERSION, concept_id),
                        )
            except Exception as e:
                logger.warning(f"Embedding update failed for {concept_id}: {e}")

    def remove_concept(self, concept_id: str):
        """
        Remove concept from index.

        INTERFACE MATCH: Original retrieval.py rebuilds entire index.
        Incremental version: O(V_doc) lazy deletion vs O(N) rebuild - 100-1000× faster

        Args:
            concept_id: ID of concept to remove
        """
        success = self.index.remove_concept(concept_id)

        if success:
            logger.debug(f"Removed concept {concept_id} from index (incremental)")

            # Auto-save every 10 operations
            if self.index.index_version % 10 == 0:
                self._auto_save()

        # P0.3: Also remove from embedding index
        embedding_engine.remove_embedding(concept_id)

    def search(self, query: SearchQuery, agent_id: str = None, scope: str = "global") -> list[SearchResult]:
        """
        Search for relevant concepts.

        INTERFACE MATCH: Exact same signature as original retrieval.py
        Takes SearchQuery, returns List[SearchResult]

        AGENT-002: agent_id + scope params for scoped retrieval.
        scope='agent' filters to agent_id's own concepts + shared 'default'.
        scope='global' (default) returns all concepts for backward compat.

        EMBEDDING_RESILIENCE_SPEC v1.1 Fix B: Two-phase architecture.
        Phase 1 gets raw results from embeddings OR TF-IDF fallback.
        Phase 2 applies post-processing pipeline (same for both paths).

        Args:
            query: SearchQuery object with query string and filters
            agent_id: Optional agent_id for scoped retrieval
            scope: 'agent' (filtered) or 'global' (all concepts, default)

        Returns:
            List of SearchResult objects with concept metadata
        """
        import time as _time_mod

        _search_t0 = _time_mod.perf_counter()

        # Build TF-IDF index if empty (first-time migration case)
        if self.index.document_count == 0:
            logger.info("Index empty, building from existing concepts...")
            self.build_index()

        # P0.3: Initialize embedding index if not ready
        self._init_embeddings()
        _t_init = _time_mod.perf_counter()  # PERF-022

        # Pre-activate concepts if enhanced retrieval is available
        if ENHANCED_RETRIEVAL:
            predictive_activation.preload_for_query(query.query, query.context)
            _t_preload = _time_mod.perf_counter()  # PERF-022
            if query.goal:
                goal_directed.set_goal(query.goal, {"query": query.query, "source": "explicit"})
            else:
                inferred_goal = goal_directed.infer_goal(query.query)
                if inferred_goal:
                    goal_directed.set_goal(inferred_goal, {"query": query.query, "source": "inferred"})
            _t_goal = _time_mod.perf_counter()  # PERF-022
        else:
            _t_preload = _t_init  # PERF-022: no-op placeholders
            _t_goal = _t_init

        # ===== Phase 1: Get raw results (embedding OR TF-IDF) =====
        if self._embeddings_available and embedding_engine.index_size > 0:
            results, concept_cache = self._search_phase1_embeddings(query)
        else:
            results, concept_cache = self._search_phase1_tfidf(query)
        _t_phase1 = _time_mod.perf_counter()  # PERF-022

        # ===== Phase 1.5: KA-scoped supplement for cross-session coverage =====
        # RETRIEVAL-032: When Phase 1 returns results concentrated in one KA,
        # supplement with concepts from the SAME KA(s) that fell below the
        # cosine cutoff. This ensures cross-session facts get surfaced.
        if get_feature_flag("KA_CROSS_SESSION_SUPPLEMENT", False):
            results, concept_cache = self._supplement_ka_coverage(
                results, query, concept_cache
            )

        # ===== Phase 1.5b: Keyword supplement for low-quality embedding results =====
        # RAGAS-DIAG-001: When embedding top score < threshold, supplement with TF-IDF
        # keyword matches. Catches entity-specific queries that embeddings miss.
        from app.config import (
            KEYWORD_SUPPLEMENT_ENABLED,
            KEYWORD_SUPPLEMENT_THRESHOLD,
            KEYWORD_SUPPLEMENT_MAX,
        )
        if (
            KEYWORD_SUPPLEMENT_ENABLED
            and self._embeddings_available
            and results
        ):
            top_score = max(r.relevance_score for r in results) if results else 0
            if top_score < KEYWORD_SUPPLEMENT_THRESHOLD:
                logger.info(
                    f"RAGAS-DIAG-001: Low embedding quality (top={top_score:.3f} < "
                    f"{KEYWORD_SUPPLEMENT_THRESHOLD}). Running TF-IDF keyword supplement."
                )
                kw_results, kw_cache = self._search_phase1_tfidf(query)
                existing_ids = {r.concept_id for r in results}
                added = 0
                for kw_r in kw_results:
                    if kw_r.concept_id not in existing_ids and added < KEYWORD_SUPPLEMENT_MAX:
                        results.append(kw_r)
                        concept_cache.update(kw_cache)
                        existing_ids.add(kw_r.concept_id)
                        added += 1
                if added:
                    logger.info(f"RAGAS-DIAG-001: Added {added} TF-IDF supplements")

        _t_supplement = _time_mod.perf_counter()  # PERF-022

        # ===== Phase 2: Post-processing pipeline (same for both paths) =====
        results = self._apply_post_processing(results, query, _search_t0, _time_mod, concept_cache)
        _t_phase2 = _time_mod.perf_counter()  # PERF-022
        _t_supplement = locals().get("_t_supplement", _t_phase1)  # safe if flag off
        logger.info(
            "PERF-022 search() breakdown: init=%.1fms preload=%.1fms goal=%.1fms "
            "phase1=%.1fms supplement=%.1fms phase2=%.1fms total=%.1fms n=%d",
            (_t_init - _search_t0) * 1000,
            (_t_preload - _t_init) * 1000,
            (_t_goal - _t_preload) * 1000,
            (_t_phase1 - _t_goal) * 1000,
            (_t_supplement - _t_phase1) * 1000,
            (_t_phase2 - _t_supplement) * 1000,
            (_t_phase2 - _search_t0) * 1000,
            len(results),
        )

        # ===== AGENT-002: Scoped filtering (PERF-003: batch lookup) =====
        if agent_id and scope == "agent":
            aid_map = self._batch_concept_agent_ids([r.concept_id for r in results])
            results = [r for r in results if aid_map.get(r.concept_id, "default") in (agent_id, "default")]

        # ===== Final trim: enforce max_results after ALL post-processing =====
        # KA supplement (Phase 1.5) and post-processing can expand results
        # beyond max_results. Trim here as the single enforcement point.
        results = results[: query.max_results]

        return results

    def _batch_concept_agent_ids(self, concept_ids: list) -> dict:
        """Batch lookup agent_id for multiple concepts. Returns {concept_id: agent_id}.
        Reads the agent_id column directly — no JSON deserialization needed.
        PERF-003: Replaces N+1 _concept_agent_id() pattern."""
        if not concept_ids:
            return {}
        # DEBT-022: _db hoisted to module-level import
        with _db() as conn:
            placeholders = ",".join("?" * len(concept_ids))
            rows = conn.execute(
                f"SELECT id, agent_id FROM concepts WHERE id IN ({placeholders})",
                concept_ids,
            ).fetchall()
        result = {row[0]: (row[1] or "default") for row in rows}
        for cid in concept_ids:
            if cid not in result:
                result[cid] = "default"
        return result

    def _search_phase1_embeddings(self, query: SearchQuery) -> tuple[list[SearchResult], dict]:
        """Phase 1 (embedding path): Semantic search via embedding engine.

        Returns (results, concept_cache) where concept_cache is {concept_id: concept}
        for reuse in Phase 2 post-processing (PERF-016).
        """
        import time as _t_emb

        _emb_t0 = _t_emb.perf_counter()
        raw_results = embedding_engine.search(query.query, top_k=query.max_results * 2)
        _emb_ms = (_t_emb.perf_counter() - _emb_t0) * 1000

        # OBS-001: Embedding search latency + candidate count
        try:
            from app.metrics import metrics as _m_obs

            _m_obs.record("retrieval_embedding_search_ms", _emb_ms, {"candidates": len(raw_results)})
        except Exception:
            pass

        results = []
        concept_cache = {}  # PERF-016: Cache for Phase 2 reuse
        _gov_scored = 0
        for concept_id, emb_score in raw_results:
            if emb_score < 0.20:
                continue

            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue
            if concept.confidence < query.min_confidence:
                continue

            concept_cache[concept_id] = concept  # PERF-016: Cache loaded concept

            final_score = self._calculate_score(concept=concept, emb_score=emb_score, query=query)
            _gov_scored += 1

            results.append(
                SearchResult(
                    concept_id=concept.id,
                    version=concept.version,
                    summary=concept.summary,
                    confidence=concept.confidence,
                    relevance_score=final_score,
                    knowledge_area=concept.metadata.get("knowledge_area"),
                    ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                    maturity=getattr(concept, "maturity", None),
                )
            )

            if len(results) >= query.max_results:
                break

        # OBS-001: Governance scoring count
        try:
            from app.metrics import metrics as _m_gov

            _m_gov.record("retrieval_candidates_scored", _gov_scored, {"returned": len(results)})
        except Exception:
            pass

        return results, concept_cache

    def _search_phase1_tfidf(self, query: SearchQuery) -> tuple[list[SearchResult], dict]:
        """Phase 1 (TF-IDF fallback): Bag-of-words cosine search.

        EMBEDDING_RESILIENCE_SPEC v1.1: Used when embeddings are unavailable.
        TF-IDF provides ~0.65 precision vs ~0.85 for embeddings.

        Returns (results, concept_cache) for Phase 2 reuse (PERF-016).
        """
        raw_results = self.index.search(query.query, top_k=query.max_results)

        results = []
        concept_cache = {}  # PERF-016: Cache for Phase 2 reuse
        for concept_id, tfidf_score in raw_results:
            if tfidf_score < 0.05:
                continue

            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue
            if concept.confidence < query.min_confidence:
                continue

            concept_cache[concept_id] = concept  # PERF-016: Cache loaded concept

            # TF-IDF scoring: scores are typically 0.05-0.40 range
            score = tfidf_score * 0.5 + concept.confidence * 0.2 + concept.stability * 0.1
            if concept.last_accessed:
                score += 0.1
            if query.context:
                context_terms = set(query.context.lower().split())
                concept_terms = set(concept.summary.lower().split())
                overlap = len(context_terms & concept_terms)
                if overlap > 0:
                    score += 0.05
            if query.goal:
                if query.goal.lower() in concept.summary.lower():
                    score += 0.05

            # Federation Phase 0, Component 0.3: KA-boost uplift (ARCH-003: shared method)
            score = self._apply_ka_boost(score, concept, query)

            score = min(1.0, score)

            results.append(
                SearchResult(
                    concept_id=concept.id,
                    version=concept.version,
                    summary=concept.summary,
                    confidence=concept.confidence,
                    relevance_score=score,
                    knowledge_area=concept.metadata.get("knowledge_area"),
                    ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                    maturity=getattr(concept, "maturity", None),
                )
            )

        results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic
        # PERF-016: Filter cache to only returned results
        trimmed = results[: query.max_results]
        trimmed_ids = {r.concept_id for r in trimmed}
        concept_cache = {k: v for k, v in concept_cache.items() if k in trimmed_ids}
        return trimmed, concept_cache

    def _supplement_ka_coverage(
        self,
        results: list[SearchResult],
        query: SearchQuery,
        concept_cache: dict,
    ) -> tuple[list[SearchResult], dict]:
        """RETRIEVAL-032: KA-scoped supplementary search for cross-session coverage.

        When Phase 1 results cluster in specific KA(s), fetch additional concepts
        from those KA(s) that fell below the cosine cutoff. This catches facts
        from other sessions that used different vocabulary for the same topic.

        Product value: When a user asks about something that spans multiple
        sessions (e.g., "what's the status of my client relationships?"),
        this ensures ALL relevant facts surface, not just the top-N by cosine.
        """
        if not results:
            return results, concept_cache

        # Step 1: Identify dominant KA(s) from Phase 1 results
        from collections import Counter
        ka_counts = Counter(r.knowledge_area for r in results if r.knowledge_area)
        if not ka_counts:
            return results, concept_cache

        # Only supplement KAs that have 2+ concepts in results (signal of relevance)
        dominant_kas = [ka for ka, count in ka_counts.items()
                        if count >= 2 and ka not in ("general", "unclassified")]
        if not dominant_kas:
            return results, concept_cache

        # Step 2: Fetch concepts from dominant KA(s) not already in results
        from app.storage import _db
        existing_ids = {r.concept_id for r in results}
        supplement_results = []

        # Pre-compute query embedding once (outside loop)
        query_vec = None
        if self._embeddings_available:
            import numpy as np
            query_vec = embedding_engine.embed_text(query.query)

        with _db() as conn:
            for ka in dominant_kas[:3]:  # Cap at 3 KAs to bound cost
                placeholders = ",".join("?" * len(existing_ids)) if existing_ids else "''"
                rows = conn.execute(
                    f"""SELECT id, summary, confidence, knowledge_area
                       FROM concepts
                       WHERE knowledge_area = ?
                         AND is_current = 1
                         AND confidence >= ?
                         AND id NOT IN ({placeholders})
                       ORDER BY confidence DESC
                       LIMIT 20""",
                    [ka, query.min_confidence] + list(existing_ids),
                ).fetchall()

                for row in rows:
                    concept_id, summary, confidence, concept_ka = row
                    # Score via embedding similarity to query
                    if query_vec is not None and concept_id in embedding_engine._id_to_pos:
                        pos = embedding_engine._id_to_pos[concept_id]
                        emb_score = float(embedding_engine._index_matrix[pos] @ query_vec)
                    else:
                        emb_score = 0.10  # Fallback: low but non-zero

                    if emb_score < 0.12:  # Lower threshold than Phase 1 (0.20)
                        continue

                    concept = load_concept(concept_id, track_access=False)
                    if not concept:
                        continue

                    concept_cache[concept_id] = concept
                    supplement_results.append(SearchResult(
                        concept_id=concept_id,
                        version=getattr(concept, "version", "v1"),
                        summary=summary,
                        confidence=confidence,
                        relevance_score=emb_score,
                        knowledge_area=concept_ka or ka,
                        ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                        maturity=getattr(concept, "maturity", None),
                    ))

        if supplement_results:
            logger.info(
                "RETRIEVAL-032: KA supplement added %d concepts from KA(s) %s",
                len(supplement_results), dominant_kas,
            )
            # Merge and re-sort
            results = results + supplement_results
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic

        return results, concept_cache

    def _apply_post_processing(
        self,
        results: list[SearchResult],
        query: SearchQuery,
        _search_t0,
        _time_mod,
        concept_cache: dict | None = None,
    ) -> list[SearchResult]:
        """Phase 2: Post-processing pipeline applied to ALL search results.

        EMBEDDING_RESILIENCE_SPEC v1.1 Fix B: This runs identically regardless
        of whether results came from embeddings or TF-IDF fallback.

        PERF-016: concept_cache is {concept_id: concept} from Phase 1.
        Eliminates redundant load_concept() calls (was: 2N DB queries, now: 0).
        """
        if not results:
            return results

        if concept_cache is None:
            concept_cache = {}

        def _get_concept(concept_id: str):
            """PERF-016: Cache-first concept lookup."""
            if concept_id in concept_cache:
                return concept_cache[concept_id]
            # Fallback for concepts not in cache (shouldn't happen in normal flow)
            concept = load_concept(concept_id, track_access=False)
            if concept:
                concept_cache[concept_id] = concept
            return concept

        # Enhanced retrieval boosts
        if ENHANCED_RETRIEVAL and results:
            _pp_t0 = _time_mod.perf_counter()  # PERF-018: per-step timing
            scored = [(r.concept_id, r.relevance_score) for r in results]
            scored = predictive_activation.boost_retrieval_scores(scored, boost_weight=0.15)
            _pp_t1 = _time_mod.perf_counter()
            # PERF-018: pass concept_cache to avoid N DB reads (PERF-016 cache was bypassed)
            scored = goal_directed.boost_scores_by_goal(scored, concept_cache=concept_cache)
            _pp_t2 = _time_mod.perf_counter()
            logger.info(  # PERF-022: promoted from debug for visibility
                "PERF-018 post_processing: predictive=%.1fms goal=%.1fms n=%d",
                (_pp_t1 - _pp_t0) * 1000,
                (_pp_t2 - _pp_t1) * 1000,
                len(results),
            )
            score_dict = dict(scored)
            for result in results:
                if result.concept_id in score_dict:
                    result.relevance_score = score_dict[result.concept_id]
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic

        # Wave 4a §4a.2: SAL Multiplier
        try:
            from app.salience import apply_sal_multiplier

            for result in results:
                concept = _get_concept(result.concept_id)  # PERF-016: cache-first
                if concept:
                    sal = getattr(concept, "salience", 0.5) or 0.5
                    result.relevance_score = apply_sal_multiplier(sal, result.relevance_score)
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic
        except ImportError:
            pass

        # Wave 4b §4b.3: Preference salience floor
        try:
            from app.provenance import apply_preference_floor

            for result in results:
                concept = _get_concept(result.concept_id)  # PERF-016: cache-first
                if concept and concept.concept_type == "preference":
                    floor_sal = apply_preference_floor(concept, getattr(concept, "salience", 0.5))
                    if floor_sal > (getattr(concept, "salience", 0.5) or 0.5):
                        result.relevance_score = max(result.relevance_score, 0.3)
        except ImportError:
            pass

        # INGEST-015: Fact-seeking query boost — is_factual concepts score ×FACT_SEEKING_BOOST
        try:
            if get_feature_flag("FACT_SEEKING_BOOST_ENABLED", True) and _is_fact_seeking_query(query.query):
                boosted_count = 0
                for result in results:
                    concept = _get_concept(result.concept_id)
                    if concept and concept.metadata.get("is_factual", False):
                        result.relevance_score = min(result.relevance_score * FACT_SEEKING_BOOST, 1.0)
                        boosted_count += 1
                if boosted_count:
                    results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic
                    logger.debug("INGEST-015: fact boost applied, n=%d", boosted_count)
        except Exception:
            pass  # Non-fatal — retrieval degrades gracefully without the boost

        # WS2: Metric 7 — retrieval_search_latency_ms
        try:
            from app.metrics import metrics as _m7

            _m7.record(
                "retrieval_search_latency_ms",
                (_time_mod.perf_counter() - _search_t0) * 1000,
                {"result_count": len(results)},
            )
            # OBS-001: Embedding index size (proxy for cache/corpus coverage)
            _idx_size = getattr(embedding_engine, "index_size", 0) if embedding_engine else 0
            _m7.record("retrieval_index_size", _idx_size)
        except Exception:
            pass

        return results

    def _calculate_score(self, concept, emb_score: float, query: SearchQuery) -> float:
        """Calculate final relevance score with governance-enhanced weights.

        Governance formula (when available):
          emb_score * 0.35 + authority * 0.20 + currency * 0.15
          + confidence * 0.10 + stability * 0.05 + context * 0.08 + goal * 0.07

        Falls back to legacy formula if governance scoring unavailable.
        """
        if GOVERNANCE_SCORING:
            # Read cached governance scores from concept (zero extra DB queries)
            authority = getattr(concept, "authority_score", None)
            currency = getattr(concept, "currency_score", None)

            if authority is not None and currency is not None:
                # Context activation
                context_boost = 0.0
                if query.context:
                    context_terms = set(query.context.lower().split())
                    concept_terms = set(concept.summary.lower().split())
                    overlap = len(context_terms & concept_terms)
                    if overlap > 0:
                        context_boost = min(1.0, overlap / 3.0)

                # Goal relevance
                goal_boost = 0.0
                if query.goal:
                    if query.goal.lower() in concept.summary.lower():
                        goal_boost = 1.0
                    else:
                        # Partial goal match
                        goal_terms = set(query.goal.lower().split())
                        concept_terms = set(concept.summary.lower().split())
                        overlap = len(goal_terms & concept_terms)
                        if overlap > 0:
                            goal_boost = min(1.0, overlap / 2.0)

                score = (
                    emb_score * RETRIEVAL_WEIGHT_SIMILARITY
                    + authority * RETRIEVAL_WEIGHT_AUTHORITY
                    + currency * RETRIEVAL_WEIGHT_CURRENCY
                    + concept.confidence * RETRIEVAL_WEIGHT_CONFIDENCE
                    + concept.stability * RETRIEVAL_WEIGHT_STABILITY
                    + context_boost * RETRIEVAL_WEIGHT_CONTEXT
                    + goal_boost * RETRIEVAL_WEIGHT_GOAL
                )

                # Federation Phase 0, Component 0.3: KA-boost uplift (ARCH-003: shared method)
                score = self._apply_ka_boost(score, concept, query)

                return min(1.0, score)

        # Legacy formula (fallback when governance scores not computed yet)
        score = emb_score * 0.5  # Embedding similarity (primary signal)
        score += concept.confidence * 0.2
        score += concept.stability * 0.1
        if concept.last_accessed:
            score += 0.1
        if query.context:
            context_terms = set(query.context.lower().split())
            concept_terms = set(concept.summary.lower().split())
            overlap = len(context_terms & concept_terms)
            if overlap > 0:
                score += 0.05
        if query.goal:
            if query.goal.lower() in concept.summary.lower():
                score += 0.05
        # ARCH-003: KA-boost applied in legacy fallback (was missing — gap vs governance branch)
        score = self._apply_ka_boost(score, concept, query)
        return min(1.0, score)

    @staticmethod
    def _apply_ka_boost(score: float, concept, query) -> float:
        """ARCH-003: Shared KA-boost uplift for TF-IDF and embedding scoring paths.

        KA-ARCH-001: Now supports both explicit ka_boost lists AND auto-inference
        from query text when KA_AUTO_BOOST_ENABLED is set.
        """
        try:
            if not get_feature_flag("KA_RELATIVE_GOVERNANCE_ENABLED", False):
                return score

            ka_boost_list = getattr(query, "ka_boost", None)

            # Auto-infer if not explicitly provided and auto-boost enabled
            if not ka_boost_list and get_feature_flag("KA_AUTO_BOOST_ENABLED", False):
                query_text = getattr(query, 'query', '') or ''
                if query_text:
                    ka_boost_list = _infer_query_kas(query_text)

            if not ka_boost_list:
                return score

            concept_ka = concept.metadata.get("knowledge_area")
            if concept_ka and concept_ka in ka_boost_list:
                boost_w = getattr(query, "ka_boost_weight", KA_BOOST_WEIGHT)
                score += boost_w
        except Exception:
            pass
        return score

    def build_index(self):
        """
        Build index from all concepts in storage.

        INTERFACE MATCH: Same method as original retrieval.py

        Called automatically on first search if index is empty.
        Uses bulk load (single SQL query) instead of N+1 per-concept queries.
        Suppresses intermediate IDF recalculations during bulk add — only
        recalculates once at the end via force_idf_recalculation().
        """
        logger.info("Building incremental index from all concepts...")

        # Single query for all concepts — eliminates N+1 load_concept calls
        concepts = list_concepts_full()
        added = 0

        # Suppress intermediate IDF recalcs during bulk add — we do one
        # final recalculation at the end. Saves O(N) per threshold crossing.
        original_threshold = self.index.idf_update_threshold
        self.index.idf_update_threshold = len(concepts) + 100

        try:
            for concept in concepts:
                if concept.id in self.index.concept_id_to_idx:
                    continue  # Already indexed, skip (Fix 1)
                searchable_text = self._concept_to_document(concept)
                if self.index.add_concept(concept.id, searchable_text):
                    added += 1
                    if added % 50 == 0:
                        logger.info(f"Indexed {added} concepts...")
        finally:
            # Restore threshold for incremental adds during runtime
            self.index.idf_update_threshold = original_threshold

        # Single IDF recalculation over complete corpus
        self.index.force_idf_recalculation()

        # Save
        self._auto_save()

        logger.info(f"Index built: {added} concepts indexed")

    def _concept_to_document(self, concept) -> str:
        """
        Convert concept to searchable text.

        Handles both legacy string evidence and v2 Evidence objects (stored as dicts).
        """
        # Extract evidence text — handle str, dict (Evidence), or Evidence object
        evidence_texts = []
        for e in concept.evidence:
            if isinstance(e, str):
                evidence_texts.append(e)
            elif isinstance(e, dict):
                evidence_texts.append(e.get("content", ""))
            elif hasattr(e, "content"):
                evidence_texts.append(e.content)

        parts = [
            concept.summary,
            " ".join(concept.signals),
            " ".join(evidence_texts),
            concept.metadata.get("knowledge_area", ""),
        ]

        # Add hypothesis descriptions
        for hyp in concept.hypotheses:
            parts.append(hyp.description)

        # RETRIEVAL-057: Add prospective indexing implications
        _impl = concept.metadata.get("implications", []) if isinstance(concept.metadata, dict) else []
        for imp in _impl:
            if isinstance(imp, str):
                parts.append(imp)

        # INGEST-034: Include event text from metadata
        for _evt in concept.metadata.get("events", []) if isinstance(concept.metadata, dict) else []:
            _evt_parts = [_evt.get("action", "")]
            if _evt.get("cause"):
                _evt_parts.append(f"because {_evt['cause']}")
            if _evt.get("consequence"):
                _evt_parts.append(f"resulting in {_evt['consequence']}")
            if _evt.get("actors"):
                _evt_parts.append(f"involving {', '.join(_evt['actors'])}")
            parts.append(" ".join(_evt_parts))

        return " ".join(parts)

    def _auto_save(self):
        """Auto-save index periodically."""
        try:
            self.index.save(self.index_path)
            logger.debug(f"Auto-saved index to {self.index_path}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    def search_for_dedup_tfidf(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Raw TF-IDF cosine similarity for deduplication checks.

        Returns raw cosine scores (NOT blended with confidence/stability).
        Used by session_learn dedup logic where the three-zone thresholds
        (≥0.85 skip, 0.50-0.84 evolve, <0.50 create) require pure cosine.

        Returns list of dicts: {concept_id, cosine_score, knowledge_area}
        """
        if self.index.document_count == 0:
            return []

        raw_results = self.index.search(query_text, top_k=top_k)

        # FIX-2: Get DB connection for direct SQL (avoids load_concept Pydantic crash)
        from app.storage import _db

        with _db() as _dedup_conn:
            results = []
            for concept_id, cosine_score in raw_results:
                if cosine_score < 0.05:
                    continue
                # FIX-2: Direct SQL instead of load_concept() to avoid Pydantic crash
                _dedup_row = _dedup_conn.execute(
                    "SELECT data, knowledge_area FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if not _dedup_row:
                    continue
                try:
                    _dedup_data = json.loads(_dedup_row[0]) if _dedup_row[0] else {}
                except (json.JSONDecodeError, TypeError):
                    _dedup_data = {}
                _dedup_ka = ""
                if isinstance(_dedup_data.get("metadata"), dict):
                    _dedup_ka = _dedup_data["metadata"].get("knowledge_area", "")
                if not _dedup_ka and _dedup_row[1]:
                    _dedup_ka = _dedup_row[1]
                results.append(
                    {
                        "concept_id": concept_id,
                        "cosine_score": cosine_score,
                        "knowledge_area": _dedup_ka,
                        "evidence_count": len(_dedup_data.get("evidence", [])),
                    }
                )

        return results

    def search_for_dedup_tfidf_batch(self, summaries: list[str], top_k: int = 3) -> list[list[dict]]:
        """PERF-021: Batch TF-IDF dedup — N sequential index searches + 1 bulk DB query.

        Replaces N separate search_for_dedup_tfidf calls in session_learn with a single
        bulk WHERE IN query for the DB layer. The TF-IDF index.search() still runs N times
        (single-query interface, no native batch API) — only DB I/O is reduced from N
        round-trips to 1.

        Args:
            summaries: list of concept summary strings (one per insight)
            top_k: candidates per query (default 3, lower than single-call default of 5
                   because batch pre-filters by score ≥ 0.05 before bulk fetch)

        Returns:
            list of N result lists. Index i corresponds to summaries[i].
            Each inner list has same format as search_for_dedup_tfidf:
            [{concept_id, cosine_score, knowledge_area, evidence_count}, ...]
            Returns [[] * N] if index empty or no results above threshold.
        """
        if not summaries or self.index.document_count == 0:
            return [[] for _ in summaries]

        # Step 1: N sequential TF-IDF searches (index has no native batch API)
        batch_raw: list[list[tuple[str, float]]] = []
        for summary in summaries:
            raw = self.index.search(summary, top_k=top_k)
            batch_raw.append(raw)

        # Step 2: Collect all concept_ids that pass the 0.05 threshold
        needed_ids: set[str] = set()
        for results in batch_raw:
            for concept_id, score in results:
                if score >= 0.05:
                    needed_ids.add(concept_id)

        if not needed_ids:
            return [[] for _ in summaries]

        # Step 3: ONE bulk IN-clause query replaces N sequential SELECTs (the actual saving)
        from app.storage import _db

        concept_meta: dict[str, dict] = {}
        with _db() as conn:
            placeholders = ",".join("?" * len(needed_ids))
            rows = conn.execute(
                f"SELECT id, data, knowledge_area FROM concepts WHERE id IN ({placeholders}) AND is_current = 1",
                list(needed_ids),
            ).fetchall()
            for row in rows:
                try:
                    _data = json.loads(row[1]) if row[1] else {}
                except (json.JSONDecodeError, TypeError):
                    _data = {}
                _ka = ""
                if isinstance(_data.get("metadata"), dict):
                    _ka = _data["metadata"].get("knowledge_area", "")
                if not _ka and row[2]:
                    _ka = row[2]
                concept_meta[row[0]] = {
                    "knowledge_area": _ka,
                    "evidence_count": len(_data.get("evidence", [])),
                }

        # Step 4: Reconstruct per-query result lists using cached metadata
        output: list[list[dict]] = []
        for results in batch_raw:
            query_results = []
            for concept_id, score in results:
                if score < 0.05:
                    continue
                if concept_id not in concept_meta:
                    continue  # concept not found or not current
                meta = concept_meta[concept_id]
                query_results.append(
                    {
                        "concept_id": concept_id,
                        "cosine_score": score,
                        "knowledge_area": meta["knowledge_area"],
                        "evidence_count": meta["evidence_count"],
                    }
                )
            output.append(query_results)

        return output

    def search_for_dedup_embedding(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Embedding-based cosine similarity for deduplication checks.

        MATURITY-003 Part A: Replaces TF-IDF for dedup when embeddings are
        available. Uses all-MiniLM-L6-v2 sentence embeddings which handle
        paraphrases correctly (TF-IDF gives 0.10 where embeddings give 0.65+).

        Returns same format as search_for_dedup_tfidf for drop-in compatibility:
        list of dicts: {concept_id, cosine_score, knowledge_area, evidence_count}

        Falls back to search_for_dedup_tfidf if embeddings unavailable.
        """
        # Ensure embedding index is initialized
        self._init_embeddings()

        if not self._embeddings_available or embedding_engine.index_size == 0:
            logger.warning("search_for_dedup_embedding: embeddings unavailable, falling back to TF-IDF")
            return self.search_for_dedup_tfidf(query_text, top_k=top_k)

        # Embedding search
        raw_results = embedding_engine.search(query_text, top_k=top_k)

        results = []
        with _db() as _dedup_conn:  # DEBT-189: Use context manager (was: get_db_connection, leaked)
            for concept_id, emb_score in raw_results:
                if emb_score < 0.10:  # Floor: below this is noise
                    continue
                # Direct SQL (same as search_for_dedup_tfidf FIX-2)
                _dedup_row = _dedup_conn.execute(
                    "SELECT data, knowledge_area FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if not _dedup_row:
                    continue
                try:
                    _dedup_data = json.loads(_dedup_row[0]) if _dedup_row[0] else {}
                except (json.JSONDecodeError, TypeError):
                    _dedup_data = {}
                _dedup_ka = ""
                if isinstance(_dedup_data.get("metadata"), dict):
                    _dedup_ka = _dedup_data["metadata"].get("knowledge_area", "")
                if not _dedup_ka and _dedup_row[1]:
                    _dedup_ka = _dedup_row[1]
                results.append(
                    {
                        "concept_id": concept_id,
                        "cosine_score": emb_score,
                        "knowledge_area": _dedup_ka,
                        "evidence_count": len(_dedup_data.get("evidence", [])),
                    }
                )

        return results

    def search_for_dedup_embedding_batch(
        self, query_texts: list[str], top_k: int = 3
    ) -> list[list[dict]]:
        """Batch embedding dedup: encode all queries at once, batch DB lookups.

        PERF-036: Extends PERF-021 batch pattern to embedding path.
        Encodes N summaries in one embed_batch() call instead of N sequential
        embed_text() calls. Batch DB lookup with WHERE IN instead of N×top_k
        individual SELECTs.

        Returns list of N result lists (same format as search_for_dedup_embedding).
        Falls back to sequential search_for_dedup_tfidf on any failure.
        """
        self._init_embeddings()
        if not self._embeddings_available or embedding_engine.index_size == 0:
            logger.warning(
                "search_for_dedup_embedding_batch: embeddings unavailable, "
                "falling back to sequential TF-IDF"
            )
            return [self.search_for_dedup_tfidf(q, top_k=top_k) for q in query_texts]

        if not query_texts:
            return []

        # 1. Batch encode all queries (~120ms for 7, vs ~700ms sequential)
        query_vecs = embedding_engine.embed_batch(query_texts)  # (N, 384)

        # 2. Per-query similarity search + collect unique concept IDs
        all_concept_ids: set[str] = set()
        per_query_top: list[list[tuple[str, float]]] = []

        for i in range(len(query_texts)):
            scores = embedding_engine._index_matrix @ query_vecs[i]  # (M,)
            if len(scores) <= top_k:
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            hits = []
            for idx in top_indices:
                score = float(scores[idx])
                if score < 0.10:  # Floor: below this is noise
                    continue
                cid = embedding_engine._index_ids[idx]
                hits.append((cid, score))
                all_concept_ids.add(cid)
            per_query_top.append(hits)

        # 3. Batch DB lookup — one WHERE IN query instead of N×top_k SELECTs
        concept_data: dict[str, dict] = {}
        if all_concept_ids:
            placeholders = ",".join("?" * len(all_concept_ids))
            with _db() as conn:
                rows = conn.execute(
                    f"SELECT id, data, knowledge_area FROM concepts "
                    f"WHERE id IN ({placeholders}) AND is_current = 1",
                    list(all_concept_ids),
                ).fetchall()
                for row in rows:
                    try:
                        data = json.loads(row[1]) if row[1] else {}
                    except (json.JSONDecodeError, TypeError):
                        data = {}
                    ka = ""
                    if isinstance(data.get("metadata"), dict):
                        ka = data["metadata"].get("knowledge_area", "")
                    if not ka and row[2]:
                        ka = row[2]
                    concept_data[row[0]] = {
                        "knowledge_area": ka,
                        "evidence_count": len(data.get("evidence", [])),
                    }

        # 4. Assemble per-query results
        results: list[list[dict]] = []
        for hits in per_query_top:
            query_results = []
            for cid, score in hits:
                if cid in concept_data:
                    query_results.append({
                        "concept_id": cid,
                        "cosine_score": score,
                        "knowledge_area": concept_data[cid]["knowledge_area"],
                        "evidence_count": concept_data[cid]["evidence_count"],
                    })
            results.append(query_results)

        return results

    @staticmethod
    def _governance_score(concept, raw_score: float) -> float:
        """P1 GOV-SCORE-001: Compute governance-aware retrieval score.

        Combines raw similarity score with authority, currency, confidence,
        stability, freshness, and recency. Uses 'is not None' checks for
        numeric scores to avoid GA-9 falsy bug (0.0 treated as missing).

        Returns -1.0 if concept should be hard-filtered (STALE/SUPERSEDED).
        """
        from app.config import (
            RETRIEVAL_WEIGHT_AUTHORITY,
            RETRIEVAL_WEIGHT_CONFIDENCE,
            RETRIEVAL_WEIGHT_CURRENCY,
            RETRIEVAL_WEIGHT_SIMILARITY,  # DEBT-002
            RETRIEVAL_WEIGHT_STABILITY,
        )

        # Hard filter STALE/SUPERSEDED concepts
        _currency_status = getattr(concept, "currency_status", None) or (
            concept.metadata.get("currency_status") if hasattr(concept, "metadata") else None
        )
        if _currency_status in ("STALE", "SUPERSEDED"):
            return -1.0

        # GA-9 falsy-safe score reads
        _authority = getattr(concept, "authority_score", None)
        _authority = _authority if _authority is not None else 0.5
        _currency = getattr(concept, "currency_score", None)
        _currency = _currency if _currency is not None else 0.5

        score = (
            raw_score * RETRIEVAL_WEIGHT_SIMILARITY
            + _authority * RETRIEVAL_WEIGHT_AUTHORITY
            + _currency * RETRIEVAL_WEIGHT_CURRENCY
            + concept.confidence * RETRIEVAL_WEIGHT_CONFIDENCE
            + concept.stability * RETRIEVAL_WEIGHT_STABILITY
        )
        # FRESHNESS_UNIFIED_REDESIGN: Exponential decay freshness bonus
        from app.config import (
            RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS,
            RETRIEVAL_FRESHNESS_MAX_BONUS,
            RETRIEVAL_FRESHNESS_EVOLUTION_BONUS,
        )

        _freshness_ts = concept.last_organic_access or concept.last_accessed or concept.created_at
        if _freshness_ts:
            try:
                from datetime import datetime as _dt

                ts_str = _freshness_ts if isinstance(_freshness_ts, str) else str(_freshness_ts)
                age_days = (
                    _utc_now() - _ensure_aware(_dt.fromisoformat(ts_str.replace("Z", "+00:00")))
                ).total_seconds() / 86400.0
                _hl = max(0.1, RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS)  # guard negative/zero
                decay = math.exp(-math.log(2) / _hl * age_days)
                score += decay * RETRIEVAL_FRESHNESS_MAX_BONUS
            except Exception:
                logger.debug("freshness_bonus_parse_error concept_id=%s ts=%s", getattr(concept, "id", "?"), _freshness_ts)

        # Evolution bonus: evolved concepts get a small additional boost
        if concept.version and concept.version != "v1":
            score += RETRIEVAL_FRESHNESS_EVOLUTION_BONUS

        # RETRIEVAL-034 Layer 2: Soft ranking penalty for CONTRADICTED/CONTESTED
        # Deprioritize but don't exclude — stale knowledge is valuable context.
        # RESOLVED is treated as ACTIVE (no penalty). Applied after all additive
        # bonuses, before min(1.0) cap.
        from app.config import STALE_TRANSPARENCY_ENABLED, STALE_PENALTY_CONTRADICTED, STALE_PENALTY_CONTESTED
        if STALE_TRANSPARENCY_ENABLED and _currency_status:
            if _currency_status == "CONTRADICTED":
                # RETRIEVAL-056: Hard-filter CONTRADICTED concepts when winner has >2x authority.
                # Inspired by Kumiho Definition 7.3 (Two-Tier Epistemic Model).
                _winner_id = getattr(concept, "superseded_by", None)
                if _winner_id:
                    from app.storage import load_concept as _load_winner_concept
                    _winner = _load_winner_concept(_winner_id, track_access=False)
                    if _winner:
                        _winner_auth = getattr(_winner, "authority_score", None)
                        _winner_auth = _winner_auth if _winner_auth is not None else 0.5
                        if _winner_auth > _authority * 2.0:
                            return -1.0  # Clear winner — exclude loser from retrieval surface
                # No clear winner or no superseded_by — keep soft penalty
                score *= STALE_PENALTY_CONTRADICTED
            elif _currency_status == "CONTESTED":
                score *= STALE_PENALTY_CONTESTED

        return min(1.0, score)

    def search_lightweight(
        self, query_text: str, top_k: int = 10, min_confidence: float = 0.0, agent_id: str = None, scope: str = "global",
        include_deprecated: bool = False,
    ) -> list[SearchResult]:
        """Fast search without full preload scan.

        Uses semantic embeddings when available, falls back to TF-IDF when not.
        Skips predictive_activation.preload_for_query() and goal_directed.infer_goal().
        Used by conversation_turn where speed is critical.

        AGENT-002: agent_id + scope params for scoped retrieval.
        scope='agent' filters to agent_id's own concepts + shared 'default'.

        EMBEDDING_RESILIENCE_SPEC v1.1: Graceful TF-IDF fallback when
        sentence_transformers is unavailable. Enhanced retrieval boosts
        applied to both paths (Fix A, Attack 8).
        """
        # Ensure embedding index is ready (or set _embeddings_available=False)
        self._init_embeddings()

        # ===== Phase 1: Get raw results =====
        # OPT-1c: Soft timeout — if Phase 1 exceeds budget, return partial results.
        # Gauntlet A11: Check inside loop, not after. A15: Floor clamp 100ms.
        import time as _time_mod_slw
        _slw_start = _time_mod_slw.perf_counter()
        _slw_soft_timeout_s = max(0.1, float(os.environ.get("PITH_SLW_SOFT_TIMEOUT_MS", "500"))) / 1000.0
        _slw_timed_out = False

        if self._embeddings_available and embedding_engine.index_size > 0:
            # Embedding path
            raw_results = embedding_engine.search(query_text, top_k=top_k)
            results = []
            for _slw_i, (concept_id, emb_score) in enumerate(raw_results):
                # OPT-1c: Check wall clock every 10 iterations to limit overhead
                if _slw_i > 0 and _slw_i % 10 == 0:
                    if (_time_mod_slw.perf_counter() - _slw_start) > _slw_soft_timeout_s:
                        logger.warning(
                            f"OPT-1c: search_lightweight soft timeout at iteration {_slw_i}/{len(raw_results)} "
                            f"({(_time_mod_slw.perf_counter() - _slw_start)*1000:.0f}ms > "
                            f"{_slw_soft_timeout_s*1000:.0f}ms) — returning {len(results)} partial results"
                        )
                        _slw_timed_out = True
                        break
                if emb_score < MIN_RETRIEVAL_SIMILARITY:  # RETRIEVAL-031: raised from 0.15
                    continue
                concept = load_concept(concept_id, track_access=False)
                if not concept:
                    continue
                if concept.confidence < min_confidence:
                    continue

                score = self._governance_score(concept, emb_score)
                if score < 0:
                    if not include_deprecated:
                        continue  # Hard-filtered (STALE/SUPERSEDED)
                    score = 0.01  # RETRIEVAL-056: include_deprecated — floor score
                results.append(
                    SearchResult(
                        concept_id=concept.id,
                        version=concept.version,
                        summary=concept.summary,
                        confidence=concept.confidence,
                        relevance_score=score,
                        knowledge_area=concept.metadata.get("knowledge_area"),
                        ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                    maturity=getattr(concept, "maturity", None),
                    created_at=concept.created_at,  # RETRIEVAL-053
                    )
                )
        else:
            # TF-IDF fallback path
            if self.index.document_count == 0:
                return []
            raw_results = self.index.search(query_text, top_k=top_k)
            results = []
            for _slw_i, (concept_id, tfidf_score) in enumerate(raw_results):
                # OPT-1c: Same soft timeout as embedding path
                if _slw_i > 0 and _slw_i % 10 == 0:
                    if (_time_mod_slw.perf_counter() - _slw_start) > _slw_soft_timeout_s:
                        logger.warning(
                            f"OPT-1c: search_lightweight TF-IDF soft timeout at {_slw_i}/{len(raw_results)}"
                        )
                        _slw_timed_out = True
                        break
                if tfidf_score < MIN_RETRIEVAL_SIMILARITY * 0.5:  # RETRIEVAL-031: TF-IDF scale differs
                    continue
                concept = load_concept(concept_id, track_access=False)
                if not concept:
                    continue
                if concept.confidence < min_confidence:
                    continue

                score = self._governance_score(concept, tfidf_score)
                if score < 0:
                    if not include_deprecated:
                        continue  # Hard-filtered (STALE/SUPERSEDED)
                    score = 0.01  # RETRIEVAL-056: include_deprecated — floor score
                results.append(
                    SearchResult(
                        concept_id=concept.id,
                        version=concept.version,
                        summary=concept.summary,
                        confidence=concept.confidence,
                        relevance_score=score,
                        knowledge_area=concept.metadata.get("knowledge_area"),
                        ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                    maturity=getattr(concept, "maturity", None),
                    created_at=concept.created_at,  # RETRIEVAL-053
                    )
                )

        # ===== Phase 1.5: KA-aware boost (KA-ARCH-001 Fix 7) =====
        # OPT-1c: Skip enhancement phases if Phase 1 timed out — return core results fast.
        if _slw_timed_out and results:
            # Still apply KA exclusion (critical for correctness) but skip boosts
            from app.config import RETRIEVAL_KA_EXCLUDE as _exc_ka
            if _exc_ka:
                _bm = os.environ.get('PITH_BENCHMARK_MODE', '').lower() in ('true', '1')
                if not _bm:
                    results = [r for r in results if r.knowledge_area not in _exc_ka]
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))
            return results

        if get_feature_flag("KA_AUTO_BOOST_ENABLED", False) and results:
            inferred_kas = _infer_query_kas(query_text)
            if inferred_kas:
                for result in results:
                    concept_ka = result.knowledge_area
                    if concept_ka and concept_ka in inferred_kas:
                        result.relevance_score = min(1.0, result.relevance_score + KA_BOOST_WEIGHT)

        # ===== Phase 1.6: KA exclusion filter (RETRIEVAL-061) =====
        # Exclude benchmark/test KAs from interactive retrieval.
        # Bypassed in benchmark mode. Config: PITH_RETRIEVAL_KA_EXCLUDE env var.
        from app.config import RETRIEVAL_KA_EXCLUDE
        if RETRIEVAL_KA_EXCLUDE and results:
            _benchmark_mode = os.environ.get('PITH_BENCHMARK_MODE', '').lower() in ('true', '1')
            if not _benchmark_mode:
                _pre_exclude = len(results)
                results = [r for r in results if r.knowledge_area not in RETRIEVAL_KA_EXCLUDE]
                _excluded = _pre_exclude - len(results)
                if _excluded > 0:
                    logger.debug(f'RETRIEVAL-061: Excluded {_excluded} concepts from KAs {RETRIEVAL_KA_EXCLUDE}')

        # ===== Phase 2: Enhanced retrieval boosts (both paths) =====
        if ENHANCED_RETRIEVAL and results:
            scored = [(r.concept_id, r.relevance_score) for r in results]
            scored = predictive_activation.boost_retrieval_scores(scored, boost_weight=0.15)
            score_dict = dict(scored)
            for result in results:
                if result.concept_id in score_dict:
                    result.relevance_score = score_dict[result.concept_id]

        # RETRIEVAL-037b v4.2: Deterministic tiebreaker — when governance scores
        # tie (common: many MAB facts share identical authority/currency/confidence),
        # sort by concept_id to make budget cutoff deterministic across server restarts.
        results.sort(key=lambda r: (-r.relevance_score, r.concept_id))

        # ===== AGENT-002: Scoped filtering (PERF-003: batch lookup) =====
        if agent_id and scope == "agent":
            aid_map = self._batch_concept_agent_ids([r.concept_id for r in results])
            results = [r for r in results if aid_map.get(r.concept_id, "default") in (agent_id, "default")]
            results = results[:top_k]

        return results

    def sync_index(self) -> int:
        """Ensure all active concepts are in the TF-IDF index.

        Compares active concept IDs from storage against the index.
        Adds any missing concepts incrementally, then recalculates IDF once.

        Returns:
            Number of concepts added to the index.
        """
        # Get all active concept IDs from storage
        all_concepts = list_concepts_full()
        storage_ids = {c.id for c in all_concepts}

        # Get indexed concept IDs (exclude logically deleted rows)
        indexed_ids = set()
        for i, cid in enumerate(self.index.concept_ids):
            if i not in self.index.deleted_indices:
                indexed_ids.add(cid)

        # Find unindexed concepts
        missing_ids = storage_ids - indexed_ids
        if not missing_ids:
            logger.info("sync_index: all concepts already indexed")
            return 0

        logger.info(f"sync_index: {len(missing_ids)} concepts not in index, adding...")

        # Suppress intermediate IDF recalcs during bulk add
        original_threshold = self.index.idf_update_threshold
        self.index.idf_update_threshold = len(missing_ids) + 100

        added = 0
        try:
            concept_map = {c.id: c for c in all_concepts}
            for cid in missing_ids:
                concept = concept_map.get(cid)
                if concept:
                    searchable_text = self._concept_to_document(concept)
                    if self.index.add_concept(cid, searchable_text):
                        added += 1
        finally:
            self.index.idf_update_threshold = original_threshold

        # Single IDF recalculation over complete corpus
        if added > 0:
            self.index.force_idf_recalculation()
            self._auto_save()

        logger.info(f"sync_index: added {added} concepts to index")
        return added

    def pairwise_similarity(self, threshold: float = 0.12) -> list[tuple[str, str, float]]:
        """Compute all above-threshold concept pairs by cosine similarity.

        Uses the TF-IDF matrix for efficient pairwise computation.
        Excludes self-edges, deleted index entries, and normalizes direction
        (sorted pair IDs) to match edge direction normalization in storage.

        Args:
            threshold: Minimum cosine similarity to include (default 0.12).

        Returns:
            List of (concept_a, concept_b, cosine_score) tuples where
            concept_a < concept_b (normalized direction).
        """
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        if self.index.tfidf_matrix is None or self.index.document_count == 0:
            logger.warning("pairwise_similarity: empty index")
            return []

        matrix = self.index.tfidf_matrix
        n_docs = matrix.shape[0]

        # Build set of valid (non-deleted) row indices
        valid_indices = [i for i in range(n_docs) if i not in self.index.deleted_indices]
        if len(valid_indices) < 2:
            return []

        # Slice matrix to valid rows only for efficient computation
        valid_matrix = matrix[valid_indices]

        # Compute pairwise cosine similarity (returns dense ndarray)
        sim_matrix = sk_cosine(valid_matrix)

        # Extract above-threshold pairs (upper triangle only to avoid duplicates)
        pairs = []
        for i_idx in range(len(valid_indices)):
            for j_idx in range(i_idx + 1, len(valid_indices)):
                score = float(sim_matrix[i_idx, j_idx])
                if score >= threshold:
                    # Map back to concept IDs
                    cid_a = self.index.concept_ids[valid_indices[i_idx]]
                    cid_b = self.index.concept_ids[valid_indices[j_idx]]
                    # Normalize direction (sorted) to match edge storage normalization
                    source, target = sorted([cid_a, cid_b])
                    pairs.append((source, target, round(score, 4)))

        # Sort by score descending for priority processing
        pairs.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            f"pairwise_similarity: {len(pairs)} pairs above threshold {threshold} "
            f"from {len(valid_indices)} indexed concepts"
        )
        return pairs

    def verify_index_integrity(self) -> dict:
        """Compare index entries against active DB concepts. Detect ghosts and orphans.

        Ghosts: entries in the index that have no active DB concept (archived/deleted).
        Orphans: active DB concepts not in the index.

        SYSTEMIC_FIXES_SPEC v1.1 Fix 2: Uses list_concepts() which returns
        active-only IDs. Archived concepts SHOULD be absent from the index.

        Returns:
            dict with ghost_ids, orphan_ids, index_count, db_count, is_healthy.
        """
        # Active concept IDs from DB
        active_ids = set(list_concepts())

        # Indexed concept IDs (excluding logically deleted rows)
        indexed_ids = set()
        for i, cid in enumerate(self.index.concept_ids):
            if i not in self.index.deleted_indices:
                indexed_ids.add(cid)

        ghost_ids = indexed_ids - active_ids  # in index but not active in DB
        orphan_ids = active_ids - indexed_ids  # active in DB but not indexed

        result = {
            "ghost_ids": sorted(ghost_ids),
            "orphan_ids": sorted(orphan_ids),
            "index_count": len(indexed_ids),
            "db_count": len(active_ids),
            "ghosts": len(ghost_ids),
            "orphans": len(orphan_ids),
            "is_healthy": len(ghost_ids) == 0 and len(orphan_ids) == 0,
        }

        if ghost_ids:
            logger.warning(f"Index integrity: {len(ghost_ids)} ghost entries: {sorted(ghost_ids)[:5]}...")
        if orphan_ids:
            logger.warning(f"Index integrity: {len(orphan_ids)} orphan concepts: {sorted(orphan_ids)[:5]}...")
        if result["is_healthy"]:
            logger.info(f"Index integrity: healthy ({len(indexed_ids)} indexed, {len(active_ids)} active)")

        return result

    def repair_index_drift(self, dry_run: bool = False, integrity: dict = None) -> dict:
        """Auto-repair index drift by removing ghosts and adding orphans.

        SYSTEMIC_FIXES_SPEC v1.1 Fix 2: Safe repair — removes ghosts via
        existing remove_concept(), adds orphans via incremental add.

        Args:
            dry_run: If True, report what would happen without executing.
            integrity: Pre-computed integrity dict from verify_index_integrity().
                       If None, runs verification internally.

        Returns:
            dict with ghosts_removed, orphans_added, dry_run status.
        """
        if integrity is None:
            integrity = self.verify_index_integrity()

        if integrity["is_healthy"]:
            return {
                "status": "healthy",
                "ghosts_removed": 0,
                "orphans_added": 0,
                "dry_run": dry_run,
            }

        ghosts_removed = 0
        orphans_added = 0

        # Remove ghost entries
        for ghost_id in integrity["ghost_ids"]:
            if dry_run:
                logger.info(f"[DRY RUN] Would remove ghost: {ghost_id}")
            else:
                self.remove_concept(ghost_id)
                logger.info(f"Removed ghost from index: {ghost_id}")
            ghosts_removed += 1

        # Add orphan concepts
        if integrity["orphan_ids"]:
            concepts = list_concepts_full()
            concept_map = {c.id: c for c in concepts}

            for orphan_id in integrity["orphan_ids"]:
                concept = concept_map.get(orphan_id)
                if not concept:
                    continue
                if dry_run:
                    logger.info(f"[DRY RUN] Would add orphan: {orphan_id}")
                else:
                    searchable_text = self._concept_to_document(concept)
                    self.index.add_concept(orphan_id, searchable_text)
                    logger.info(f"Added orphan to index: {orphan_id}")
                orphans_added += 1

            if not dry_run and orphans_added > 0:
                self.index.force_idf_recalculation()
                self._auto_save()

        result = {
            "status": "repaired" if not dry_run else "dry_run",
            "ghosts_removed": ghosts_removed,
            "orphans_added": orphans_added,
            "dry_run": dry_run,
        }
        logger.info(f"Index repair: {result}")
        return result


# Global instance - EXACT MATCH of original retrieval.py
retrieval_engine = RetrievalEngine()
