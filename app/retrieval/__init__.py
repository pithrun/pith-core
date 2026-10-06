"""
Integration Adapter for Pith Retrieval Engine
Provides backward-compatible interface for incremental TF-IDF index

CRITICAL: This is a DROP-IN REPLACEMENT for retrieval.py
Interface must match exactly for seamless integration.

P0.3: Hybrid architecture — embeddings for search, TF-IDF for dedup/auto-association.
"""

import collections
import contextlib
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.datetime_utils import _ensure_aware, _utc_now
from app.core.deadline import TurnDeadline
from app.core.file_lock import lock_fd_exclusive, unlock_fd
from app.core.foreground_contract import (
    ForegroundContractConfig,
    ForegroundContractDecision,
    ForegroundDecision,
    build_foreground_contract_config,
    foreground_contract_mode_for_unit,
    get_foreground_contract,
)
from app.core.models import SearchQuery, SearchResult
from app.core.profile import resolve_data_dir
from app.retrieval import refresh_drain as _refresh_drain
from app.retrieval.foreground_embedding import foreground_embedding_process
from app.retrieval.incremental_tfidf import IncrementalTfidfIndex
from app.retrieval.query_intent import QueryIntentExpansion, expand_query_intent
from app.retrieval.searchable_text import (
    EMBEDDING_TEXT_CONTRACT_VERSION,
    build_searchable_text,
    build_searchable_text_from_concept,
    embedding_freshness_metadata,
)
from app.storage import (  # DEBT-022: hoisted from function-level
    INDEX_DIR,
    list_concepts,
    list_concepts_full,
    load_concept,
    load_concepts_batch,
    read_snapshot_db,  # T2-3: RLock-free read path for retrieval
)
from app.storage.embedding import EMBEDDING_DIM, EMBEDDING_VERSION, embedding_engine

try:
    from app.authority_chain import order_authority_candidates, resolve_authority_chain
except Exception:  # pragma: no cover - defensive fallback during partial installs
    order_authority_candidates = None
    resolve_authority_chain = None

# Import governance scoring config
try:
    from app.core.config import (
        AUTHORITY_ARTIFACT_BOOST_ENABLED,
        AUTHORITY_ARTIFACT_BOOST_WEIGHT,
        BENCHMARK,
        CROSS_SESSION_WINDOW_HOURS,  # SESSION-012: Concurrent session time window
        KA_BOOST_WEIGHT,
        MIN_RETRIEVAL_SIMILARITY,  # RETRIEVAL-031
        RETRIEVAL_WEIGHT_AUTHORITY,
        RETRIEVAL_WEIGHT_CONFIDENCE,
        RETRIEVAL_WEIGHT_CONTEXT,
        RETRIEVAL_WEIGHT_CURRENCY,
        RETRIEVAL_WEIGHT_GOAL,
        RETRIEVAL_WEIGHT_RECENCY,  # RETRIEVAL-100: Creation-time recency
        RETRIEVAL_WEIGHT_SESSION_PROXIMITY,  # SESSION-012: Cross-session boost
        RETRIEVAL_WEIGHT_SIMILARITY,  # DEBT-002: renamed from RETRIEVAL_WEIGHT_EMBEDDING
        RETRIEVAL_WEIGHT_STABILITY,
        RETRIEVAL_WEIGHT_UTILITY,  # RETRIEVAL-080: Feedback loop utility weight
        UTILITY_COLD_START,  # RETRIEVAL-080: Default utility for concepts without feedback
        get_feature_flag,
    )

    GOVERNANCE_SCORING = True
except ImportError:
    GOVERNANCE_SCORING = False
    KA_BOOST_WEIGHT = 0.2
    AUTHORITY_ARTIFACT_BOOST_ENABLED = False
    AUTHORITY_ARTIFACT_BOOST_WEIGHT = 0.18

# Import activation modules for enhanced retrieval
# DEBT-242: retrieval must not import features directly (Contract 5).
# Using importlib to break static dependency while preserving fallback behavior.
try:
    import importlib as _il

    _gd_mod = _il.import_module("app.features.goal_directed")
    goal_directed = _gd_mod.goal_directed
    from app.retrieval.predictive import predictive_activation

    ENHANCED_RETRIEVAL = True
except ImportError:
    ENHANCED_RETRIEVAL = False
    predictive_activation = None
    goal_directed = None

logger = logging.getLogger(__name__)

_EMBEDDING_FRESHNESS_MODES = {"observe", "repair", "enforce"}
_SEMANTIC_WARM_READINESS_SCHEMA = "retrieval.semantic_warm_readiness.v1"
_SEMANTIC_WARM_READY_STATES = frozenset({"not_started", "running", "ready", "failed", "skipped", "cancelled"})
_SEMANTIC_WARM_SOURCES = frozenset({"startup", "foreground", "direct", "unknown"})
_SEMANTIC_FULL_SEARCH_WAIT_BUDGET_MS = 50
_SEMANTIC_FULL_SEARCH_MAX_UNGAUNTLETED_WAIT_MS = 100
_PRELOAD_RECENT_P95_SKIP_MS = 150.0
_PRELOAD_MIN_DEADLINE_REMAINING_MS = 250.0
_PRELOAD_RECENT_SAMPLE_SIZE = 32


@dataclass(frozen=True)
class SemanticAdmissionDecision:
    use_semantic: bool
    semantic_state: str
    semantic_admission: str
    skip_reason: str | None
    wait_ms: float
    foreground_init_attempted: bool


def _default_semantic_warm_readiness() -> dict[str, Any]:
    return {
        "schema_version": _SEMANTIC_WARM_READINESS_SCHEMA,
        "state": "not_started",
        "source": "unknown",
        "last_attempt_at": None,
        "last_success_at": None,
        "last_failure_at": None,
        "last_result": None,
        "error_type": None,
        "query_path_warmed": False,
    }


def _bounded_warm_label(value: str | None, *, default: str | None = None) -> str | None:
    text = str(value or default or "").strip()
    return text[:80] if text else None


def _ensure_semantic_warm_readiness(engine: Any) -> None:
    if not hasattr(engine, "_semantic_warm_readiness_lock"):
        engine._semantic_warm_readiness_lock = threading.Lock()
    if not hasattr(engine, "_semantic_warm_readiness"):
        engine._semantic_warm_readiness = _default_semantic_warm_readiness()


def embedding_freshness_mode() -> str:
    value = os.environ.get("PITH_EMBEDDING_FRESHNESS_MODE", "observe").strip().lower()
    if value not in _EMBEDDING_FRESHNESS_MODES:
        logger.warning("Invalid PITH_EMBEDDING_FRESHNESS_MODE=%r; using observe", value)
        return "observe"
    return value


def _freshness_metadata_matches(row: Any, searchable_text: str) -> tuple[bool, str, dict[str, Any]]:
    metadata = embedding_freshness_metadata(searchable_text)
    stored_hash = row["embedding_text_hash"] if "embedding_text_hash" in row.keys() else None
    stored_contract = (
        row["embedding_text_contract_version"] if "embedding_text_contract_version" in row.keys() else None
    )
    if not stored_hash or not stored_contract:
        return False, "missing_metadata", metadata
    if int(stored_contract) != EMBEDDING_TEXT_CONTRACT_VERSION:
        return False, "contract_mismatch", metadata
    if stored_hash != metadata["embedding_text_hash"]:
        return False, "hash_mismatch", metadata
    return True, "fresh", metadata


def _embedding_update_tuple(emb: np.ndarray, searchable_text: str, concept_id: str) -> tuple[Any, ...]:
    metadata = embedding_freshness_metadata(searchable_text)
    return (
        emb.tobytes(),
        EMBEDDING_VERSION,
        metadata["embedding_text_hash"],
        metadata["embedding_text_contract_version"],
        metadata["embedding_refreshed_at"],
        concept_id,
    )


def _record_metric(
    name: str,
    value: float,
    labels: dict[str, str | int | float] | None = None,
    *,
    flush: bool = False,
) -> None:
    try:
        from app.core.metrics_facade import metrics

        metrics.record(name, value, labels or {})
        if flush:
            metrics.flush()
    except Exception:
        pass


def _record_search_stage_latency(
    stage: str,
    elapsed_ms: float,
    *,
    path: str,
    result_count: int | None = None,
    candidate_count: int | None = None,
    extra_labels: dict[str, str | int | float | bool] | None = None,
) -> None:
    labels: dict[str, str | int | float] = {"stage": stage, "path": path}
    if result_count is not None:
        labels["result_count"] = int(result_count)
    if candidate_count is not None:
        labels["candidate_count"] = int(candidate_count)
    if extra_labels:
        for key, value in extra_labels.items():
            if isinstance(value, bool):
                labels[key] = "true" if value else "false"
            else:
                labels[key] = value
    _record_metric("retrieval_search_stage_latency_ms", round(float(elapsed_ms), 2), labels)


_KA_SUPPLEMENT_MAX_KAS = 3
_KA_SUPPLEMENT_SQL_LIMIT_PER_KA = 20
_KA_SUPPLEMENT_MAX_ELIGIBLE_CANDIDATES = 24
_KA_SUPPLEMENT_MAX_ADDITIONS = 8
_KA_SUPPLEMENT_MIN_SIMILARITY = 0.12
_KA_SUPPLEMENT_BUDGET_MS = 120.0


_AUTHORITY_ARTIFACT_AUTHORITY_TERMS = frozenset(
    {
        "active",
        "approval",
        "approved",
        "authoritative",
        "current",
        "final",
        "frozen",
        "latest",
        "official",
    }
)
_AUTHORITY_ARTIFACT_DOMAIN_TERMS = frozenset(
    {
        "artifact",
        "branch",
        "commit",
        "doc",
        "docs",
        "freeze",
        "frozen",
        "ledger",
        "packet",
        "version",
    }
)
_AUTHORITY_ARTIFACT_SCOPE_TERMS = frozenset(
    {
        "copy",
        "launch",
        "pith",
        "public",
        "release",
    }
)
_AUTHORITY_ARTIFACT_EXPANSION_TERMS = (
    "official",
    "current",
    "docs",
    "reports",
    "commit",
    "freeze",
    "historical",
)
_AUTHORITY_ARTIFACT_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_AUTHORITY_ARTIFACT_COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
_AUTHORITY_ARTIFACT_FILE_RE = re.compile(
    r"(?:\bdocs/|\b[a-z0-9_./-]+\.md\b|\b[A-Z0-9_]+(?:PACKET|LEDGER|REPORT|ARTIFACT)[A-Z0-9_]*\b)",
    re.IGNORECASE,
)
_AUTHORITY_ARTIFACT_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE)
_AUTHORITY_ARTIFACT_SPECIFICITY_WEIGHT = 0.04


def _authority_artifact_tokens(text: str) -> set[str]:
    return set(_AUTHORITY_ARTIFACT_TOKEN_RE.findall((text or "").casefold()))


def _is_authority_artifact_query(query_text: str) -> bool:
    lowered = (query_text or "").casefold()
    tokens = _authority_artifact_tokens(lowered)
    has_authority = bool(tokens & _AUTHORITY_ARTIFACT_AUTHORITY_TERMS)
    has_authority = has_authority or "source of truth" in lowered or "go/no-go" in lowered
    has_domain = bool(tokens & _AUTHORITY_ARTIFACT_DOMAIN_TERMS)
    if has_domain and tokens & {"packet", "doc", "docs"}:
        scoped_domain_terms = _AUTHORITY_ARTIFACT_SCOPE_TERMS | (
            _AUTHORITY_ARTIFACT_DOMAIN_TERMS - {"packet", "doc", "docs"}
        )
        has_domain = bool(tokens & scoped_domain_terms)
    return has_authority and has_domain


def _expand_authority_artifact_query(query_text: str) -> str:
    if not _is_authority_artifact_query(query_text):
        return query_text
    tokens = _authority_artifact_tokens(query_text)
    expansion_terms = list(_AUTHORITY_ARTIFACT_EXPANSION_TERMS)
    if resolve_authority_chain is not None:
        try:
            expansion_terms.extend(resolve_authority_chain(query_text).expanded_terms)
        except Exception as exc:
            logger.debug("authority chain expansion skipped: %s", exc)
    missing_terms = [term for term in dict.fromkeys(expansion_terms) if term not in tokens]
    if not missing_terms:
        return query_text
    return f"{query_text} {' '.join(missing_terms)}"


def _authority_artifact_evidence_groups(summary: str) -> set[str]:
    text = summary or ""
    lowered = text.casefold()
    tokens = _authority_artifact_tokens(text)
    groups: set[str] = set()

    if _AUTHORITY_ARTIFACT_FILE_RE.search(text):
        groups.add("artifact_locator")
    if (
        _AUTHORITY_ARTIFACT_COMMIT_RE.search(text)
        or _AUTHORITY_ARTIFACT_VERSION_RE.search(text)
        or "source freeze" in lowered
        or "freeze" in tokens
        or "frozen" in tokens
        or "active" in tokens
    ):
        groups.add("commit_version_status")
    if (
        tokens & {"approved", "approval", "andrew", "current", "official", "authoritative"}
        or "source of truth" in lowered
        or "gated execution" in lowered
    ):
        groups.add("approval_current_authority")
    if tokens & {"historical", "older", "baseline", "superseded"} or "not current" in lowered:
        groups.add("historical_demotion")

    return groups


def _qualifies_for_authority_artifact_boost(summary: str) -> bool:
    groups = _authority_artifact_evidence_groups(summary)
    return "artifact_locator" in groups and len(groups) >= 2


def _authority_artifact_specificity_bonus(query_text: str, summary: str) -> float:
    query_tokens = _authority_artifact_tokens(query_text)
    if not (query_tokens & {"approval", "approved", "copy", "packet"}):
        return 0.0

    summary_tokens = _authority_artifact_tokens(summary)
    groups = _authority_artifact_evidence_groups(summary)
    asks_for_approval_packet = bool(query_tokens & {"approval", "approved"}) and bool(query_tokens & {"copy", "packet"})
    is_current_packet = (
        asks_for_approval_packet
        and "packet" in summary_tokens
        and {"approval_current_authority", "historical_demotion"}.issubset(groups)
    )
    if not is_current_packet:
        return 0.0
    return _AUTHORITY_ARTIFACT_SPECIFICITY_WEIGHT


def _apply_authority_artifact_boost(
    results: list[SearchResult],
    query_text: str,
    *,
    path: str,
) -> int:
    if not AUTHORITY_ARTIFACT_BOOST_ENABLED or not results:
        return 0
    if not _is_authority_artifact_query(query_text):
        return 0

    boosted = 0
    try:
        for result in results:
            if _qualifies_for_authority_artifact_boost(result.summary):
                result.relevance_score = min(
                    1.0,
                    result.relevance_score
                    + AUTHORITY_ARTIFACT_BOOST_WEIGHT
                    + _authority_artifact_specificity_bonus(query_text, result.summary),
                )
                boosted += 1
        if boosted:
            _sort_authority_artifact_results(results, query_text)
            _record_metric(
                "retrieval_authority_artifact_boost_applied_total",
                float(boosted),
                {"path": path},
            )
    except Exception as exc:
        logger.debug("authority artifact boost skipped: %s", exc)
        return 0
    return boosted


def _sort_authority_artifact_results(results: list[SearchResult], query_text: str) -> None:
    if order_authority_candidates is None:
        results.sort(key=lambda r: (-r.relevance_score, r.concept_id))
        return
    try:
        original_ids = [result.concept_id for result in results]
        ordered_ids = order_authority_candidates(original_ids, query_text)
        if ordered_ids == original_ids:
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))
            return
        order_index = {concept_id: idx for idx, concept_id in enumerate(ordered_ids)}
        results.sort(key=lambda r: (order_index.get(r.concept_id, len(results)), -r.relevance_score, r.concept_id))
    except Exception as exc:
        logger.debug("authority chain ordering skipped: %s", exc)
        results.sort(key=lambda r: (-r.relevance_score, r.concept_id))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int_clamped(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _embedding_feature_available() -> bool:
    return bool(getattr(embedding_engine, "is_available", False))


_DIAGNOSTIC_SOURCE_METADATA_FLAG = "PITH_RETRIEVAL_DIAGNOSTIC_SOURCE_METADATA"
_PRIVATE_DIAGNOSTIC_FLAGS = (
    "PITH_BENCHMARK_MODE",
    "PITH_PRIVATE_DIAGNOSTIC_MODE",
    "PITH_RETRIEVAL_DIAGNOSTIC_PRIVATE_MODE",
)
_ALLOWED_DIAGNOSTIC_METADATA_KEYS = {
    "beam_source_key",
    "beam_source_turn_id",
    "beam_source_turn_index",
    "beam_source_batch_idx",
    "beam_source_role",
    "beam_role",
    "benchmark_observation_date",
    "grouped_count_packet",
    "selection_facet",
    "preference_facet",
}
_FORBIDDEN_DIAGNOSTIC_METADATA_KEYS = {
    "answer",
    "answer_string",
    "expected_answer",
    "expected_source_ref",
    "expected_source_refs",
    "gold_id",
    "gold_ids",
    "benchmark_private",
    "qid",
    "question_id",
    "rubric",
    "source_ref",
    "source_chat_ids",
}


def _contains_forbidden_diagnostic_metadata(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_DIAGNOSTIC_METADATA_KEYS:
                return True
            if _contains_forbidden_diagnostic_metadata(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_diagnostic_metadata(item) for item in value)
    return False


def _diagnostic_source_metadata(concept: Any) -> dict[str, Any] | None:
    """Return in-process source metadata for private diagnostics only."""
    if not _env_flag(_DIAGNOSTIC_SOURCE_METADATA_FLAG):
        return None
    if not any(_env_flag(flag) for flag in _PRIVATE_DIAGNOSTIC_FLAGS):
        return None

    metadata = getattr(concept, "metadata", None)
    if not isinstance(metadata, dict) or not metadata:
        return None
    if _contains_forbidden_diagnostic_metadata(metadata):
        _record_metric("retrieval_diagnostic_source_metadata_blocked", 1.0, {"reason": "forbidden_key"})
        return None

    diagnostic = {
        key: metadata[key]
        for key in _ALLOWED_DIAGNOSTIC_METADATA_KEYS
        if key in metadata and metadata[key] not in (None, "")
    }
    if not diagnostic:
        return None
    _record_metric("retrieval_diagnostic_source_metadata_copied", 1.0, {"keys": str(len(diagnostic))})
    return diagnostic


_LEXICAL_EVIDENCE_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_LEXICAL_EVIDENCE_STOPWORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "who",
        "which",
        "how",
        "why",
        "is",
        "are",
        "was",
        "were",
        "did",
        "does",
        "do",
        "has",
        "have",
        "had",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "and",
        "or",
        "not",
        "be",
        "been",
        "being",
        "would",
        "could",
        "should",
        "will",
        "can",
        "may",
        "might",
        "shall",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "if",
        "still",
        "likely",
        "all",
        "must",
    }
)


@dataclass(frozen=True)
class LexicalEvidenceAdmissionResult:
    results: list[SearchResult]
    concept_cache: dict
    protected_ids: set[str]
    attempted: bool
    admitted_count: int
    reason: str
    support_trace: dict[str, Any] | None = None


def _lexical_evidence_tokens(text: str) -> set[str]:
    return {
        token
        for token in _LEXICAL_EVIDENCE_TOKEN_RE.findall((text or "").casefold())
        if len(token) > 2 and token not in _LEXICAL_EVIDENCE_STOPWORDS
    }


_MH262_CANARY_TRACE_LIMIT = 30


def _mh262_canary_retrieval_trace_enabled() -> bool:
    """Diagnostic-only trace gate shared with conversation_turn."""
    benchmark_enabled = bool(getattr(globals().get("BENCHMARK", None), "enabled", False))
    return _env_flag("PITH_MH262_CANARY_RETRIEVAL_TRACE", False) and (
        benchmark_enabled or _env_flag("PITH_BENCHMARK_READONLY", False)
    )


def _mh262_canary_trace_target_ids() -> list[str]:
    raw = os.environ.get("PITH_MH262_CANARY_TRACE_TARGET_IDS", "")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _mh262_trace_ids_from_scores(
    scored: list[tuple[str, float]],
    *,
    limit: int = _MH262_CANARY_TRACE_LIMIT,
) -> list[str]:
    return [str(concept_id) for concept_id, _score in scored[:limit]]


def _mh262_trace_score_sample(
    scored: list[tuple[str, float]],
    *,
    limit: int = _MH262_CANARY_TRACE_LIMIT,
) -> list[dict]:
    return [
        {"concept_id": str(concept_id), "score": round(float(score or 0.0), 6)} for concept_id, score in scored[:limit]
    ]


def _mh262_target_indices(ids: list[str], target_ids: list[str]) -> dict[str, int]:
    if not target_ids:
        return {}
    return {target_id: ids.index(target_id) for target_id in target_ids if target_id in ids}


def _mh262_predictive_activation_snapshot(activation_obj, *, limit: int = _MH262_CANARY_TRACE_LIMIT) -> dict:
    active_items = []
    active_concepts = getattr(activation_obj, "active_concepts", {}) or {}
    now = _utc_now()
    for concept_id, node in active_concepts.items():
        try:
            age_seconds = (now - _ensure_aware(node.timestamp)).total_seconds()
        except Exception:
            age_seconds = None
        active_items.append(
            {
                "concept_id": str(concept_id),
                "activation": round(float(getattr(node, "activation", 0.0) or 0.0), 6),
                "source": str(getattr(node, "source", "")),
                "age_seconds": round(float(age_seconds), 3) if age_seconds is not None else None,
            }
        )
    active_items.sort(key=lambda item: (-item["activation"], item["concept_id"]))
    top_items = active_items[:limit]
    target_ids = _mh262_canary_trace_target_ids()
    all_ids = [item["concept_id"] for item in active_items]
    return {
        "active_count": len(active_items),
        "top_active_concepts": top_items,
        "target_indices": _mh262_target_indices(all_ids, target_ids),
    }


def _mh262_trace_score_stage(
    trace: dict | None,
    stage: str,
    *,
    before_scores: list[tuple[str, float]] | None = None,
    after_scores: list[tuple[str, float]] | None = None,
) -> None:
    if trace is None:
        return
    target_ids = _mh262_canary_trace_target_ids()
    payload: dict = {}
    if before_scores is not None:
        before_ids = [str(concept_id) for concept_id, _score in before_scores]
        payload["before_count"] = len(before_scores)
        payload["before_ids"] = _mh262_trace_ids_from_scores(before_scores)
        payload["before_score_sample"] = _mh262_trace_score_sample(before_scores)
        payload["before_target_indices"] = _mh262_target_indices(before_ids, target_ids)
    if after_scores is not None:
        after_ids = [str(concept_id) for concept_id, _score in after_scores]
        payload["after_count"] = len(after_scores)
        payload["after_ids"] = _mh262_trace_ids_from_scores(after_scores)
        payload["after_score_sample"] = _mh262_trace_score_sample(after_scores)
        payload["after_target_indices"] = _mh262_target_indices(after_ids, target_ids)
    trace.setdefault("stages", {})[stage] = payload


# INGEST-015: Fact-seeking query detection + boost constant
FACT_SEEKING_BOOST = 1.25

# INGEST-015 legacy markers (fallback when structural classifier disabled)
_FACT_QUERY_MARKERS = [
    "what is my ",
    "what's my ",
    "where do i ",
    "where am i ",
    "what do i do",
    "where do i work",
    "who is my ",
    "who's my ",
    "what is their ",
    "where does ",
    "remind me ",
    "do you know my ",
    "what's the name",
    "where do i live",
    "what city am i",
    "tell me about my ",
    "what company",
    "who do i work",
]


def _is_fact_seeking_query(query_text: str) -> bool:
    """Detect personal-fact-seeking queries for retrieval boost.

    INGEST-017: Uses structural query classifier when enabled,
    falls back to INGEST-015 markers when disabled.
    """
    try:
        if get_feature_flag("STRUCTURAL_QUERY_CLASSIFIER_ENABLED", True):
            from app.cognitive.fact_classifier import is_fact_seeking_query

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
    from app.cognitive.taxonomy import classify_ka_by_embedding, infer_knowledge_area

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


class QuiesceDrainTimeout(RuntimeError):
    """Raised when in-flight index writers fail to drain within the quiesce window."""


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
        self._embedding_init_lock = threading.Lock()

        # Writer-quiesce state. The quiesce window pauses all in-process
        # index mutators (skip-not-block) so a rebuild+swap can run against a
        # frozen index_version. Drain waits only for writers already in-flight.
        self._quiesce = threading.Event()
        self._inflight_writers = 0
        self._writer_cv = threading.Condition()
        self._quiesce_skipped: collections.Counter = collections.Counter()
        self.last_canary_search_lightweight_trace: dict | None = None
        self.last_live_initial_admission_probe_trace: dict | None = None
        self.last_query_intent_trace: dict | None = None
        self.last_lexical_evidence_support_trace: dict | None = None
        self._semantic_warm_readiness_lock = threading.Lock()
        self._semantic_warm_readiness = _default_semantic_warm_readiness()
        self._preload_latency_ms: collections.deque[float] = collections.deque(maxlen=_PRELOAD_RECENT_SAMPLE_SIZE)

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

    def _ensure_semantic_warm_readiness_state(self) -> None:
        _ensure_semantic_warm_readiness(self)

    def record_semantic_warm_readiness(
        self,
        *,
        state: str,
        source: str,
        result: str | None = None,
        error: str | None = None,
        query_path_warmed: bool | None = None,
    ) -> None:
        bounded_state = state if state in _SEMANTIC_WARM_READY_STATES else "failed"
        bounded_source = source if source in _SEMANTIC_WARM_SOURCES else "unknown"
        now = _utc_now().isoformat()
        _ensure_semantic_warm_readiness(self)
        with self._semantic_warm_readiness_lock:
            next_state = dict(self._semantic_warm_readiness)
            next_state.update(
                {
                    "schema_version": _SEMANTIC_WARM_READINESS_SCHEMA,
                    "state": bounded_state,
                    "source": bounded_source,
                    "last_result": _bounded_warm_label(result, default=bounded_state),
                    "error_type": _bounded_warm_label(error),
                    "initialized": bool(getattr(self, "_embeddings_initialized", False)),
                    "available": bool(getattr(self, "_embeddings_available", False)),
                    "index_size": int(getattr(embedding_engine, "index_size", 0) or 0),
                }
            )
            if bounded_state == "running":
                next_state["last_attempt_at"] = now
            elif bounded_state == "ready":
                next_state["last_success_at"] = now
            elif bounded_state in {"failed", "cancelled"}:
                next_state["last_failure_at"] = now
            if query_path_warmed is not None:
                next_state["query_path_warmed"] = bool(query_path_warmed)
            self._semantic_warm_readiness = next_state

    def semantic_warm_readiness_snapshot(self) -> dict[str, Any]:
        _ensure_semantic_warm_readiness(self)
        with self._semantic_warm_readiness_lock:
            snapshot = dict(self._semantic_warm_readiness)
        snapshot["schema_version"] = _SEMANTIC_WARM_READINESS_SCHEMA
        snapshot["initialized"] = bool(getattr(self, "_embeddings_initialized", False))
        snapshot["available"] = bool(getattr(self, "_embeddings_available", False))
        snapshot["index_size"] = int(getattr(embedding_engine, "index_size", 0) or 0)
        return snapshot

    def semantic_full_search_state(self) -> str:
        if not _embedding_feature_available():
            return "unavailable"
        if bool(getattr(self, "_embeddings_available", False)) and int(getattr(embedding_engine, "index_size", 0) or 0) > 0:
            return "ready"
        snapshot = self.semantic_warm_readiness_snapshot()
        if snapshot.get("state") == "running":
            return "warming"
        if bool(getattr(self, "_embeddings_initialized", False)) and int(snapshot.get("index_size") or 0) <= 0:
            return "empty_index"
        return "tfidf_fallback"

    def _semantic_admission_labels(
        self,
        decision: SemanticAdmissionDecision,
        *,
        request_cluster_id: str,
    ) -> dict[str, str | int | float | bool]:
        labels: dict[str, str | int | float | bool] = {
            "request_cluster_id": request_cluster_id,
            "pid": os.getpid(),
            "semantic_state": decision.semantic_state,
            "semantic_admission": decision.semantic_admission,
            "foreground_init_attempted": decision.foreground_init_attempted,
        }
        if decision.skip_reason:
            labels["skip_reason"] = decision.skip_reason
        return labels

    def _semantic_ready_for_full_search(self) -> bool:
        return bool(getattr(self, "_embeddings_available", False)) and int(getattr(embedding_engine, "index_size", 0) or 0) > 0

    def _decide_semantic_admission(
        self,
        *,
        caller: str,
        deadline: TurnDeadline | None = None,
        allow_foreground_init: bool = False,
        bounded_wait_ms: int = _SEMANTIC_FULL_SEARCH_WAIT_BUDGET_MS,
    ) -> SemanticAdmissionDecision:
        wait_budget_ms = max(0, min(int(bounded_wait_ms), _SEMANTIC_FULL_SEARCH_MAX_UNGAUNTLETED_WAIT_MS))
        waited_ms = 0.0
        foreground_attempted = False

        def _finish(
            *,
            use_semantic: bool,
            semantic_state: str,
            semantic_admission: str,
            skip_reason: str | None,
        ) -> SemanticAdmissionDecision:
            decision = SemanticAdmissionDecision(
                use_semantic=use_semantic,
                semantic_state=semantic_state,
                semantic_admission=semantic_admission,
                skip_reason=skip_reason,
                wait_ms=round(waited_ms, 2),
                foreground_init_attempted=foreground_attempted,
            )
            labels = {
                "caller": caller,
                "semantic_state": decision.semantic_state,
                "semantic_admission": decision.semantic_admission,
                "foreground_init_attempted": "true" if decision.foreground_init_attempted else "false",
            }
            if decision.skip_reason:
                labels["skip_reason"] = decision.skip_reason
            _record_metric("retrieval_semantic_admission_total", 1.0, labels)
            if decision.wait_ms:
                _record_metric(
                    "retrieval_semantic_wait_ms",
                    decision.wait_ms,
                    {
                        "caller": caller,
                        "semantic_state": decision.semantic_state,
                        "semantic_admission": decision.semantic_admission,
                    },
                )
            return decision

        if self._semantic_ready_for_full_search():
            return _finish(
                use_semantic=True,
                semantic_state="ready",
                semantic_admission="ready",
                skip_reason=None,
            )

        if not _embedding_feature_available():
            return _finish(
                use_semantic=False,
                semantic_state="unavailable",
                semantic_admission="tfidf_fallback",
                skip_reason="semantic_unavailable",
            )

        if wait_budget_ms > 0 and self.semantic_warm_readiness_snapshot().get("state") == "running":
            wait_start = time.perf_counter()
            deadline_at = wait_start + (wait_budget_ms / 1000.0)
            while time.perf_counter() < deadline_at:
                if self._semantic_ready_for_full_search():
                    waited_ms = (time.perf_counter() - wait_start) * 1000.0
                    return _finish(
                        use_semantic=True,
                        semantic_state="ready",
                        semantic_admission="ready_after_wait",
                        skip_reason=None,
                    )
                time.sleep(min(0.01, max(0.0, deadline_at - time.perf_counter())))
            waited_ms = (time.perf_counter() - wait_start) * 1000.0

        if allow_foreground_init:
            if deadline and not deadline.can_start(
                "retrieval.full_search.embedding_init",
                min_remaining_ms=float(wait_budget_ms or _SEMANTIC_FULL_SEARCH_WAIT_BUDGET_MS),
            ):
                return _finish(
                    use_semantic=False,
                    semantic_state=self.semantic_full_search_state(),
                    semantic_admission="tfidf_fallback",
                    skip_reason="deadline_before_start",
                )
            foreground_attempted = True
            self.record_semantic_warm_readiness(
                state="running",
                source="foreground",
                result="foreground_init_started",
            )
            self._init_embeddings()
            if self._semantic_ready_for_full_search():
                self.record_semantic_warm_readiness(
                    state="ready",
                    source="foreground",
                    result="foreground_init_complete",
                )
                return _finish(
                    use_semantic=True,
                    semantic_state="ready",
                    semantic_admission="foreground_init",
                    skip_reason=None,
                )

        state = self.semantic_full_search_state()
        if state == "warming":
            reason = "semantic_warmup_running"
        elif state == "empty_index":
            reason = "semantic_empty_index"
        elif state == "unavailable":
            reason = "semantic_unavailable"
        elif not allow_foreground_init:
            reason = "foreground_init_disabled"
        else:
            reason = "semantic_not_initialized"
        return _finish(
            use_semantic=False,
            semantic_state=state,
            semantic_admission="tfidf_fallback",
            skip_reason=reason,
        )

    def _recent_preload_p95_ms(self) -> float | None:
        values = list(getattr(self, "_preload_latency_ms", []) or [])
        if not values:
            return None
        values.sort()
        idx = min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))
        return float(values[idx])

    def _record_preload_latency_ms(self, elapsed_ms: float) -> None:
        try:
            self._preload_latency_ms.append(float(elapsed_ms))
        except Exception:
            pass

    def _predictive_preload_skip_reason(
        self,
        decision: SemanticAdmissionDecision,
        *,
        deadline: TurnDeadline | None = None,
    ) -> str | None:
        if not decision.use_semantic:
            return decision.skip_reason or "semantic_not_ready"
        recent_p95 = self._recent_preload_p95_ms()
        p95_limit = _env_float("PITH_PRELOAD_RECENT_P95_SKIP_MS", _PRELOAD_RECENT_P95_SKIP_MS)
        if recent_p95 is not None and recent_p95 > p95_limit:
            return "preload_recent_p95_exceeded"
        if deadline and not deadline.can_start(
            "retrieval.predictive_preload",
            min_remaining_ms=_PRELOAD_MIN_DEADLINE_REMAINING_MS,
        ):
            return "deadline_before_start"
        return None

    @staticmethod
    def _trim_with_protected_ids(
        results: list[SearchResult],
        limit: int,
        protected_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        if not results:
            return results

        protected_ids = protected_ids or set()
        deduped: dict[str, SearchResult] = {}
        for result in results:
            existing = deduped.get(result.concept_id)
            if existing is None or result.relevance_score > existing.relevance_score:
                deduped[result.concept_id] = result

        ordered = sorted(deduped.values(), key=lambda r: (-r.relevance_score, r.concept_id))
        trimmed = ordered[:limit]
        trimmed_ids = {result.concept_id for result in trimmed}
        missing_protected = [
            result
            for result in ordered[limit:]
            if result.concept_id in protected_ids and result.concept_id not in trimmed_ids
        ]
        if not missing_protected:
            return trimmed

        for protected in missing_protected[:1]:
            replacement_index = None
            for idx in range(len(trimmed) - 1, -1, -1):
                if trimmed[idx].concept_id not in protected_ids:
                    replacement_index = idx
                    break
            if replacement_index is None:
                break
            trimmed[replacement_index] = protected
            trimmed_ids.add(protected.concept_id)

        return sorted(trimmed, key=lambda r: (-r.relevance_score, r.concept_id))

    @staticmethod
    def _lexical_candidate_current_state(concept_id: str) -> dict[str, Any] | None:
        try:
            with read_snapshot_db("retrieval_137_lexical_current_state") as conn:
                row = conn.execute(
                    """SELECT status, is_current, currency_status, superseded_by
                       FROM concepts WHERE id = ?""",
                    (concept_id,),
                ).fetchone()
        except Exception as exc:
            logger.debug("RETRIEVAL-137: current-state gate failed for %s: %s", concept_id, exc)
            return None
        if row is None:
            return None
        return {
            "status": row["status"],
            "is_current": int(row["is_current"] or 0),
            "currency_status": (row["currency_status"] or "ACTIVE").upper(),
            "superseded_by": row["superseded_by"],
        }

    def _admit_strong_lexical_evidence(
        self,
        results: list[SearchResult],
        query_text: str,
        *,
        path: str,
        concept_cache: dict | None = None,
        deadline: TurnDeadline | None = None,
    ) -> LexicalEvidenceAdmissionResult:
        from app.core.config import (
            LEXICAL_EVIDENCE_ADMISSION_ENABLED,
            LEXICAL_EVIDENCE_ADMISSION_MAX_PER_QUERY,
            LEXICAL_EVIDENCE_ADMISSION_MAX_RANK,
            LEXICAL_EVIDENCE_ADMISSION_MIN_OVERLAP_RATIO,
            LEXICAL_EVIDENCE_ADMISSION_MIN_QUERY_TOKENS,
            LEXICAL_EVIDENCE_ADMISSION_MIN_TFIDF,
            LEXICAL_EVIDENCE_ADMISSION_SCORE_CAP,
        )

        start = time.perf_counter()
        concept_cache = concept_cache if concept_cache is not None else {}
        protected_ids: set[str] = set()
        support_ids: set[str] = set()
        current_ids: set[str] = set()
        contested_ids: set[str] = set()

        def _support_trace(attempted: bool, admitted_count: int, reason: str) -> dict[str, Any]:
            return {
                "schema_version": "retrieval.lexical_evidence_support.v1",
                "path": path,
                "attempted": bool(attempted),
                "admitted": admitted_count > 0,
                "support_present": bool(support_ids),
                "reason": reason,
                "admitted_count": int(admitted_count),
                "admitted_ids": sorted(protected_ids),
                "support_ids": sorted(support_ids),
                "protected_ids": sorted(protected_ids),
                "current_ids": sorted(current_ids),
                "contested_ids": sorted(contested_ids),
                "trace_authority": "retrieval_137_lexical_admission",
                "runtime_eligible": True,
            }

        def _finish(attempted: bool, admitted_count: int, reason: str) -> LexicalEvidenceAdmissionResult:
            _record_metric(
                "retrieval_lexical_evidence_admission_attempt_total",
                1.0,
                {"path": path, "reason": reason, "attempted": str(attempted).lower()},
            )
            _record_metric(
                "retrieval_lexical_evidence_admission_latency_ms",
                round((time.perf_counter() - start) * 1000.0, 2),
                {"path": path, "reason": reason},
            )
            if admitted_count:
                _record_metric(
                    "retrieval_lexical_evidence_admission_admitted_total",
                    float(admitted_count),
                    {"path": path},
                )
            elif attempted:
                _record_metric(
                    "retrieval_lexical_evidence_admission_rejected_total",
                    1.0,
                    {"path": path, "reason": reason},
                )
            return LexicalEvidenceAdmissionResult(
                results,
                concept_cache,
                protected_ids,
                attempted,
                admitted_count,
                reason,
                _support_trace(attempted, admitted_count, reason),
            )

        if not LEXICAL_EVIDENCE_ADMISSION_ENABLED:
            return _finish(False, 0, "feature_disabled")
        if not results:
            return _finish(False, 0, "no_results")
        if self.index.document_count == 0:
            return _finish(False, 0, "empty_index")
        if deadline and not deadline.can_start("retrieval.lexical_evidence_admission", min_remaining_ms=50.0):
            deadline.skip("retrieval.lexical_evidence_admission", "deadline_before_start", priority="optional")
            return _finish(True, 0, "deadline_before_start")

        query_tokens = _lexical_evidence_tokens(query_text)
        if len(query_tokens) < LEXICAL_EVIDENCE_ADMISSION_MIN_QUERY_TOKENS:
            return _finish(True, 0, "query_too_short")

        existing_ids = {result.concept_id for result in results}
        raw_results = self.index.search(query_text, top_k=max(LEXICAL_EVIDENCE_ADMISSION_MAX_RANK, 1))
        admitted = 0
        reason = "rank_out_of_range"
        for rank, (concept_id, tfidf_score) in enumerate(raw_results, start=1):
            if rank > LEXICAL_EVIDENCE_ADMISSION_MAX_RANK:
                break
            already_present = concept_id in existing_ids
            if tfidf_score < LEXICAL_EVIDENCE_ADMISSION_MIN_TFIDF:
                reason = "score_below_floor"
                continue
            concept = concept_cache.get(concept_id)
            if concept is None:
                concept = load_concept(concept_id, track_access=False)
            if concept is None:
                reason = "concept_missing"
                continue
            try:
                concept_text = build_searchable_text_from_concept(concept)
            except Exception:
                concept_text = getattr(concept, "summary", "")
            concept_tokens = _lexical_evidence_tokens(concept_text)
            overlap_ratio = len(query_tokens & concept_tokens) / max(1, len(query_tokens))
            if overlap_ratio < LEXICAL_EVIDENCE_ADMISSION_MIN_OVERLAP_RATIO:
                reason = "overlap_below_floor"
                continue
            current_state = self._lexical_candidate_current_state(concept_id)
            if (
                current_state is None
                or current_state["status"] != "active"
                or current_state["is_current"] != 1
                or current_state["currency_status"] in {"STALE", "SUPERSEDED"}
                or current_state["superseded_by"] not in (None, "")
            ):
                reason = "not_current"
                continue
            score = self._governance_score(concept, tfidf_score)
            if score < 0:
                reason = "governance_filtered"
                continue
            score = min(LEXICAL_EVIDENCE_ADMISSION_SCORE_CAP, score)
            support_ids.add(concept_id)
            if current_state["is_current"] == 1:
                current_ids.add(concept_id)
            if current_state["currency_status"] == "CONTESTED":
                contested_ids.add(concept_id)
            if already_present:
                reason = "already_present"
                continue
            results.append(
                SearchResult(
                    concept_id=concept.id,
                    version=concept.version,
                    summary=concept.summary,
                    confidence=concept.confidence,
                    relevance_score=round(score, 4),
                    knowledge_area=concept.metadata.get("knowledge_area"),
                    ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                    maturity=getattr(concept, "maturity", None),
                    created_at=getattr(concept, "created_at", None),
                    metadata=_diagnostic_source_metadata(concept),
                )
            )
            concept_cache[concept_id] = concept
            protected_ids.add(concept_id)
            existing_ids.add(concept_id)
            admitted += 1
            reason = "admitted"
            if admitted >= LEXICAL_EVIDENCE_ADMISSION_MAX_PER_QUERY:
                break

        return _finish(True, admitted, reason)

    @contextlib.contextmanager
    def _writer_admission(self, op: str):
        """Admit or skip an index mutation depending on quiesce state.

        Yields True if the caller should proceed with the mutation, False if a
        quiesce window is active (caller must skip and return a typed no-op).
        Check+increment happen under the same condition to be TOCTOU-safe.
        """
        with self._writer_cv:
            if self._quiesce.is_set():
                self._quiesce_skipped[op] += 1
                admitted = False
            else:
                self._inflight_writers += 1
                admitted = True
        try:
            yield admitted
        finally:
            if admitted:
                with self._writer_cv:
                    self._inflight_writers -= 1
                    self._writer_cv.notify_all()

    @contextlib.contextmanager
    def quiesce_writers(self, *, drain_timeout_s: float = 10.0):
        """Open a quiesce window: set the gate, drain in-flight writers, yield.

        New writers skip immediately (skip-not-block). Drain waits only for
        writers already in-flight when the gate engaged. On drain timeout the
        gate is cleared and QuiesceDrainTimeout is raised. The gate is always
        cleared on exit (normal or exception).
        """
        self._quiesce.set()
        with self._writer_cv:
            drained = self._writer_cv.wait_for(lambda: self._inflight_writers == 0, timeout=drain_timeout_s)
        if not drained:
            self._quiesce.clear()
            with self._writer_cv:
                self._writer_cv.notify_all()
            raise QuiesceDrainTimeout(
                f"quiesce drain timed out after {drain_timeout_s}s ({self._inflight_writers} writers still in-flight)"
            )
        try:
            yield
        finally:
            self._quiesce.clear()
            with self._writer_cv:
                self._writer_cv.notify_all()

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
        if not self._embedding_init_lock.acquire(blocking=False):
            logger.info("Embedding init already in progress — using TF-IDF fallback until ready")
            self._embeddings_available = False
            return

        try:
            if self._embeddings_initialized:
                return

            # Check if embeddings are even available
            if not embedding_engine.is_available:
                logger.info("Embeddings unavailable — skipping embedding init, TF-IDF only")
                self._embeddings_initialized = True
                self._embeddings_available = False
                return

            # T2-3: read_snapshot_db — RLock-free read for embedding init
            # Load concepts with and without embeddings
            # Memory Integrity Spec v1.2, §5.1.1: Exclude DISCARDED concepts from retrieval
            # RETRIEVAL-014 Layer 1b: Exclude SUPERSEDED concepts from index loading.
            # Defense-in-depth: Layer 1c evicts on supersession, but this prevents
            # re-entry on restart. _governance_score (line 705) also hard-filters,
            # but loading them wastes memory and compute.
            freshness_mode = embedding_freshness_mode()
            with read_snapshot_db("init_embeddings") as conn:
                try:
                    rows = conn.execute(
                        """SELECT id, summary, data, fragment_keywords, embedding,
                                  embedding_version, embedding_text_hash,
                                  embedding_text_contract_version, embedding_refreshed_at
                           FROM concepts
                           WHERE status = 'active' AND maturity != 'DISCARDED' AND is_current = 1"""
                    ).fetchall()
                except Exception as exc:
                    logger.warning(
                        "_init_embeddings: freshness metadata columns unavailable; "
                        "falling back to observe-compatible legacy SELECT: %s",
                        exc,
                    )
                    rows = conn.execute(
                        """SELECT id, summary, data, embedding, embedding_version
                           FROM concepts
                           WHERE status = 'active' AND maturity != 'DISCARDED' AND is_current = 1"""
                    ).fetchall()

            if not rows:
                self._embeddings_initialized = True
                return

            existing_ids = []
            existing_embeddings = []
            needs_embedding = []  # (concept_id, searchable_text)
            freshness_counts: collections.Counter[str] = collections.Counter()

            for row in rows:
                cid = row["id"]
                if row["embedding"] and row["embedding_version"] == EMBEDDING_VERSION:
                    # Load pre-computed embedding from BLOB
                    emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                    if emb.shape[0] == EMBEDDING_DIM:
                        searchable_text = build_searchable_text(row)
                        is_fresh, state, _ = _freshness_metadata_matches(row, searchable_text)
                        freshness_counts[state] += 1
                        if is_fresh or freshness_mode in {"observe", "repair"}:
                            existing_ids.append(cid)
                            existing_embeddings.append(emb)
                            continue
                        # enforce mode refuses stale-compatible blobs; explicit
                        # repair CLI owns re-embedding so startup stays bounded.
                        continue
                    freshness_counts["wrong_dimension"] += 1
                elif row["embedding"] and row["embedding_version"] != EMBEDDING_VERSION:
                    freshness_counts["wrong_version"] += 1
                else:
                    freshness_counts["missing_blob"] += 1

                # Need to compute embedding — extract searchable text
                try:
                    needs_embedding.append((cid, build_searchable_text(row)))
                except Exception as e:
                    logger.warning(f"_init_embeddings: failed to parse {cid}: {e}")

            # Batch-embed missing concepts
            if needs_embedding:
                logger.info(f"Embedding {len(needs_embedding)} concepts (batch)...")
                texts = [text for _, text in needs_embedding]
                new_embeddings = embedding_engine.embed_batch(texts)

                # T2-3: Persist via _db_immediate (write path — needs RLock)
                from app.storage import _db_immediate

                with _db_immediate() as _w_conn:
                    for i, (cid, _) in enumerate(needs_embedding):
                        emb = new_embeddings[i]
                        _w_conn.execute(
                            """UPDATE concepts
                               SET embedding = ?, embedding_version = ?,
                                   embedding_text_hash = ?,
                                   embedding_text_contract_version = ?,
                                   embedding_refreshed_at = ?
                               WHERE id = ?""",
                            _embedding_update_tuple(emb, needs_embedding[i][1], cid),
                        )
                        existing_ids.append(cid)
                        existing_embeddings.append(emb)
                logger.info(f"Persisted {len(needs_embedding)} new embeddings to SQLite")

            # Build in-memory index
            if existing_embeddings:
                matrix = np.vstack(existing_embeddings)
                embedding_engine.build_index(existing_ids, matrix)

            self._embeddings_initialized = True
            self._embeddings_available = True
            if freshness_counts:
                logger.info(
                    "Embedding freshness init states mode=%s counts=%s",
                    freshness_mode,
                    dict(freshness_counts),
                )
            logger.info(f"Embedding index ready: {len(existing_ids)} concepts")
        finally:
            self._embedding_init_lock.release()

    def add_concept(self, concept_id: str):
        """
        Add single concept to index (incremental).

        INTERFACE MATCH: Takes concept_id (string) just like original retrieval.py

        Performance: O(V_doc) vs O(N) full rebuild - 100-1000× faster

        Args:
            concept_id: ID of concept to add
        """
        with self._writer_admission("add_concept") as admit:
            if not admit:
                return
            self._add_concept_inner(concept_id)

    def _load_concept_row_for_index(self, concept_id: str) -> dict | None:
        """Load the minimal current-concept fields for searchable-text assembly.

        Returns a mapping ``{data, summary, fragment_keywords}`` consumable by
        ``build_searchable_text`` (RETRIEVAL-125, A3), or ``None`` if the concept
        is not current/present. ``fragment_keywords`` is read defensively because
        the column may not exist pre-migration.
        """
        with read_snapshot_db("index_concept_load") as _conn:  # T2-3
            try:
                _row = _conn.execute(
                    "SELECT data, summary, fragment_keywords FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                _frag = _row[2] if _row is not None and len(_row) > 2 else ""
            except Exception:
                # fragment_keywords column absent (pre-migration) — fall back.
                _row = _conn.execute(
                    "SELECT data, summary FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                _frag = ""
        if not _row:
            return None
        return {"data": _row[0], "summary": _row[1], "fragment_keywords": _frag or ""}

    def _add_concept_inner(self, concept_id: str):
        # FIX-2: Direct SQL instead of load_concept() to avoid Pydantic crash
        # on concepts with incomplete data JSON blobs.
        # T2-3: read_snapshot_db — RLock-free read for concept load
        # RETRIEVAL-125 (A3): text assembly is the ONE shared build_searchable_text
        # helper, so the incremental add path, the in-place refresh path, and the
        # stale-index audit produce byte-identical searchable text.
        _row_map = self._load_concept_row_for_index(concept_id)
        if _row_map is None:
            logger.warning(f"Concept {concept_id} not found, skipping index add")
            return

        searchable_text = build_searchable_text(_row_map)
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
            # RETRIEVAL-125 Phase C: add_concept no-op'd because the concept is
            # already indexed. If its searchable text has since changed, the stored
            # term-counts are stale (and add can never refresh them). Enqueue for a
            # debounced steady-state refresh. Gated by PITH_TFIDF_REFRESH_DRAIN —
            # zero hot-path cost (no staleness probe) when disabled.
            try:
                if _refresh_drain.drain_enabled() and self.index.stored_terms_stale(concept_id, searchable_text):
                    _refresh_drain.refresh_queue.enqueue(concept_id)
            except Exception:
                pass

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
                        logger.info(f"COLD-START-FIX: _embeddings_available set True after add_concept({concept_id})")
                    # Persist to SQLite
                    from app.storage import _db_immediate

                    with _db_immediate() as conn:
                        conn.execute(
                            """UPDATE concepts
                               SET embedding = ?, embedding_version = ?,
                                   embedding_text_hash = ?,
                                   embedding_text_contract_version = ?,
                                   embedding_refreshed_at = ?
                               WHERE id = ?""",
                            _embedding_update_tuple(emb, searchable_text, concept_id),
                        )
            except Exception as e:
                logger.warning(f"Embedding update failed for {concept_id}: {e}")

    def remove_concept(
        self,
        concept_id: str,
        *,
        persist: bool = False,
        skip_periodic_checkpoint: bool = False,
        strict_persist: bool = False,
    ):
        """
        Remove concept from index.

        INTERFACE MATCH: Original retrieval.py rebuilds entire index.
        Incremental version: O(V_doc) lazy deletion vs O(N) rebuild - 100-1000× faster

        Args:
            concept_id: ID of concept to remove
            persist: Force an index checkpoint, including when already absent.
            skip_periodic_checkpoint: Suppress the legacy every-10th checkpoint.
            strict_persist: Propagate checkpoint failures to the retry owner.
        """
        if strict_persist and not persist:
            raise ValueError("strict_persist requires persist=True")
        with self._writer_admission("remove_concept") as admit:
            if not admit:
                return
            success = self.index.remove_concept(concept_id)

            if success:
                logger.debug(f"Removed concept {concept_id} from index (incremental)")

            # A requested persistence retry must checkpoint even when the in-memory
            # tombstone already exists from an earlier failed-save attempt.
            if persist:
                if strict_persist:
                    self._auto_save(strict=True)
                else:
                    self._auto_save()
            elif success and not skip_periodic_checkpoint and self.index.index_version % 10 == 0:
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
        _request_cluster_id = hashlib.sha256(
            f"{os.getpid()}:{time.time_ns()}:{id(query)}".encode()
        ).hexdigest()[:12]
        self.last_query_intent_trace = None
        self.last_lexical_evidence_support_trace = None

        intent_expansion: QueryIntentExpansion | None = None
        if get_feature_flag("QUERY_INTENT_EXPANSION_ENABLED", True):
            try:
                intent_expansion = expand_query_intent(
                    query.query,
                    input_scope="query_argument",
                    expansion_input_source="SearchQuery.query",
                )
                if intent_expansion.matched_aliases:
                    self.last_query_intent_trace = intent_expansion.to_trace()
                    merged_kas = list(getattr(query, "ka_boost", None) or [])
                    for ka in intent_expansion.inferred_kas:
                        if ka not in merged_kas:
                            merged_kas.append(ka)
                    query_update: dict[str, Any] = {"query": intent_expansion.expanded_query}
                    if merged_kas:
                        query_update["ka_boost"] = merged_kas
                    if hasattr(query, "model_copy"):
                        query = query.model_copy(update=query_update)
                    else:
                        query = query.copy(update=query_update)
                    _record_metric(
                        "query_intent.alias_match_total",
                        float(len(intent_expansion.matched_aliases)),
                        {"path": "search", "source": intent_expansion.source},
                    )
                    if intent_expansion.contamination_guard_blocked:
                        _record_metric(
                            "query_intent.contamination_guard_blocked_total",
                            1.0,
                            {"path": "search", "source": intent_expansion.source},
                        )
            except Exception as _qie:
                logger.debug("QUERY-INTENT: search expansion failed (non-fatal): %s", _qie)

        authority_expanded_query = _expand_authority_artifact_query(query.query)
        if authority_expanded_query != query.query:
            query_update = {"query": authority_expanded_query}
            if hasattr(query, "model_copy"):
                query = query.model_copy(update=query_update)
            else:
                query = query.copy(update=query_update)
            _record_metric("retrieval_authority_artifact_query_expanded_total", 1.0, {"path": "search"})

        # Build TF-IDF index if empty (first-time migration case)
        if self.index.document_count == 0:
            logger.info("Index empty, building from existing concepts...")
            self.build_index()

        _semantic_wait_ms = _env_int_clamped(
            "PITH_FULL_SEARCH_SEMANTIC_WAIT_MS",
            _SEMANTIC_FULL_SEARCH_WAIT_BUDGET_MS,
            0,
            _SEMANTIC_FULL_SEARCH_MAX_UNGAUNTLETED_WAIT_MS,
        )
        _semantic_decision = self._decide_semantic_admission(
            caller="search",
            allow_foreground_init=_env_flag("PITH_FULL_SEARCH_FOREGROUND_EMBEDDING_INIT", False),
            bounded_wait_ms=_semantic_wait_ms,
        )
        _embedding_phase1_used = _semantic_decision.use_semantic
        _search_path = "embedding" if _embedding_phase1_used else "tfidf"
        _semantic_stage_labels = self._semantic_admission_labels(
            _semantic_decision,
            request_cluster_id=_request_cluster_id,
        )
        _t_init = _time_mod.perf_counter()  # PERF-022

        # Pre-activate concepts if enhanced retrieval is available
        if ENHANCED_RETRIEVAL:
            _preload_skip_reason = self._predictive_preload_skip_reason(_semantic_decision)
            if _preload_skip_reason is None:
                _preload_start = _time_mod.perf_counter()
                predictive_activation.preload_for_query(query.query, query.context)
                self._record_preload_latency_ms((_time_mod.perf_counter() - _preload_start) * 1000)
                _preload_skipped = False
            else:
                _preload_skipped = True
                _record_metric(
                    "retrieval_preload_skipped_total",
                    1.0,
                    {
                        "path": _search_path,
                        "semantic_state": _semantic_decision.semantic_state,
                        "semantic_admission": _semantic_decision.semantic_admission,
                        "skip_reason": _preload_skip_reason,
                    },
                )
            _t_preload = _time_mod.perf_counter()  # PERF-022
            if query.goal:
                goal_directed.set_goal(query.goal, {"query": query.query, "source": "explicit"})
            else:
                inferred_goal = goal_directed.infer_goal(query.query)
                if inferred_goal:
                    goal_directed.set_goal(inferred_goal, {"query": query.query, "source": "inferred"})
            _t_goal = _time_mod.perf_counter()  # PERF-022
        else:
            _preload_skipped = True
            _preload_skip_reason = "enhanced_retrieval_disabled"
            _t_preload = _t_init  # PERF-022: no-op placeholders
            _t_goal = _t_init

        # ===== Phase 1: Get raw results (embedding OR TF-IDF) =====
        _record_search_stage_latency(
            "init_embeddings",
            (_t_init - _search_t0) * 1000,
            path=_search_path,
            extra_labels={**_semantic_stage_labels, "semantic_wait_ms": round(_semantic_decision.wait_ms, 2)},
        )
        _preload_stage_labels = {
            **_semantic_stage_labels,
            "preload_skipped": _preload_skipped,
        }
        if _preload_skip_reason:
            _preload_stage_labels["skip_reason"] = _preload_skip_reason
        _record_search_stage_latency(
            "preload_for_query",
            (_t_preload - _t_init) * 1000,
            path=_search_path,
            extra_labels=_preload_stage_labels,
        )
        _record_search_stage_latency(
            "goal_inference",
            (_t_goal - _t_preload) * 1000,
            path=_search_path,
            extra_labels=_semantic_stage_labels,
        )
        if _embedding_phase1_used:
            results, concept_cache = self._search_phase1_embeddings(query)
        else:
            results, concept_cache = self._search_phase1_tfidf(query)
        _t_phase1 = _time_mod.perf_counter()  # PERF-022
        _record_search_stage_latency(
            "phase1_total",
            (_t_phase1 - _t_goal) * 1000,
            path=_search_path,
            result_count=len(results),
            candidate_count=len(concept_cache),
            extra_labels=_semantic_stage_labels,
        )

        # ===== Phase 1.5: KA-scoped supplement for cross-session coverage =====
        # RETRIEVAL-032: When Phase 1 returns results concentrated in one KA,
        # supplement with concepts from the SAME KA(s) that fell below the
        # cosine cutoff. This ensures cross-session facts get surfaced.
        if get_feature_flag("KA_CROSS_SESSION_SUPPLEMENT", False):
            results, concept_cache = self._supplement_ka_coverage(
                results,
                query,
                concept_cache,
                path=_search_path,
            )

        # ===== Phase 1.5b: Keyword supplement for low-quality embedding results =====
        # RAGAS-DIAG-001: When embedding top score < threshold, supplement with TF-IDF
        # keyword matches. Catches entity-specific queries that embeddings miss.
        from app.core.config import (
            KEYWORD_SUPPLEMENT_ENABLED,
            KEYWORD_SUPPLEMENT_MAX,
            KEYWORD_SUPPLEMENT_THRESHOLD,
        )

        if KEYWORD_SUPPLEMENT_ENABLED and self._embeddings_available and results:
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

        lexical_admission = self._admit_strong_lexical_evidence(
            results,
            query.query,
            path="search" if _embedding_phase1_used else "search_tfidf",
            concept_cache=concept_cache,
        )
        results = lexical_admission.results
        concept_cache = lexical_admission.concept_cache
        self.last_lexical_evidence_support_trace = lexical_admission.support_trace

        _t_supplement = _time_mod.perf_counter()  # PERF-022
        _record_search_stage_latency(
            "supplement_admission",
            (_t_supplement - _t_phase1) * 1000,
            path=_search_path,
            result_count=len(results),
            candidate_count=len(concept_cache),
            extra_labels=_semantic_stage_labels,
        )

        # ===== Phase 2: Post-processing pipeline (same for both paths) =====
        results = self._apply_post_processing(
            results,
            query,
            _search_t0,
            _time_mod,
            concept_cache,
            search_path=_search_path,
        )
        _t_phase2 = _time_mod.perf_counter()  # PERF-022
        _t_supplement = locals().get("_t_supplement", _t_phase1)  # safe if flag off
        _record_search_stage_latency(
            "post_processing_total",
            (_t_phase2 - _t_supplement) * 1000,
            path=_search_path,
            result_count=len(results),
            extra_labels=_semantic_stage_labels,
        )
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
        _t_scope = _time_mod.perf_counter()
        if agent_id and scope == "agent":
            aid_map = self._batch_concept_agent_ids([r.concept_id for r in results])
            results = [r for r in results if aid_map.get(r.concept_id, "default") in (agent_id, "default")]
        _t_trim_start = _time_mod.perf_counter()
        _record_search_stage_latency(
            "agent_scope_filter",
            (_t_trim_start - _t_scope) * 1000,
            path=_search_path,
            result_count=len(results),
            extra_labels=_semantic_stage_labels,
        )

        # ===== Final trim: enforce max_results after ALL post-processing =====
        # KA supplement (Phase 1.5) and post-processing can expand results
        # beyond max_results. Trim here as the single enforcement point.
        results = self._trim_with_protected_ids(results, query.max_results, lexical_admission.protected_ids)
        _t_done = _time_mod.perf_counter()
        _record_search_stage_latency(
            "final_trim",
            (_t_done - _t_trim_start) * 1000,
            path=_search_path,
            result_count=len(results),
            extra_labels=_semantic_stage_labels,
        )
        _record_search_stage_latency(
            "total",
            (_t_done - _search_t0) * 1000,
            path=_search_path,
            result_count=len(results),
            extra_labels=_semantic_stage_labels,
        )

        return results

    def _batch_concept_agent_ids(self, concept_ids: list) -> dict:
        """Batch lookup agent_id for multiple concepts. Returns {concept_id: agent_id}.
        Reads the agent_id column directly — no JSON deserialization needed.
        PERF-003: Replaces N+1 _concept_agent_id() pattern."""
        if not concept_ids:
            return {}
        # T2-3: read_snapshot_db — RLock-free batch agent_id lookup
        with read_snapshot_db("batch_agent_ids") as conn:
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
        _record_search_stage_latency(
            "embedding_raw_search",
            _emb_ms,
            path="embedding",
            candidate_count=len(raw_results),
        )

        # OBS-001: Embedding search latency + candidate count
        try:
            from app.core.metrics_facade import metrics as _m_obs

            _m_obs.record("retrieval_embedding_search_ms", _emb_ms, {"candidates": len(raw_results)})
        except Exception:
            pass

        results = []
        concept_cache = {}  # PERF-016: Cache for Phase 2 reuse
        _gov_scored = 0
        _load_score_t0 = _t_emb.perf_counter()
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
                    created_at=getattr(concept, "created_at", None),
                    metadata=_diagnostic_source_metadata(concept),
                )
            )

            if len(results) >= query.max_results:
                break

        # OBS-001: Governance scoring count
        try:
            from app.core.metrics_facade import metrics as _m_gov

            _m_gov.record("retrieval_candidates_scored", _gov_scored, {"returned": len(results)})
        except Exception:
            pass
        _record_search_stage_latency(
            "embedding_current_load_score",
            (_t_emb.perf_counter() - _load_score_t0) * 1000,
            path="embedding",
            result_count=len(results),
            candidate_count=len(raw_results),
        )

        return results, concept_cache

    def _search_phase1_tfidf(self, query: SearchQuery) -> tuple[list[SearchResult], dict]:
        """Phase 1 (TF-IDF fallback): Bag-of-words cosine search.

        EMBEDDING_RESILIENCE_SPEC v1.1: Used when embeddings are unavailable.
        TF-IDF provides ~0.65 precision vs ~0.85 for embeddings.

        Returns (results, concept_cache) for Phase 2 reuse (PERF-016).
        """
        # Bind a local handle so the search runs against one consistent index
        # snapshot even if a rebuild+swap rebinds self.index concurrently.
        idx = self.index
        _tfidf_t0 = time.perf_counter()
        raw_results = idx.search(query.query, top_k=query.max_results)
        _tfidf_raw_ms = (time.perf_counter() - _tfidf_t0) * 1000
        _record_search_stage_latency(
            "tfidf_raw_search",
            _tfidf_raw_ms,
            path="tfidf",
            candidate_count=len(raw_results),
        )

        results = []
        concept_cache = {}  # PERF-016: Cache for Phase 2 reuse
        _load_score_t0 = time.perf_counter()
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
                    created_at=getattr(concept, "created_at", None),
                    metadata=_diagnostic_source_metadata(concept),
                )
            )

        _record_search_stage_latency(
            "tfidf_current_load_score",
            (time.perf_counter() - _load_score_t0) * 1000,
            path="tfidf",
            result_count=len(results),
            candidate_count=len(raw_results),
        )
        _sort_t0 = time.perf_counter()
        results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic
        # PERF-016: Filter cache to only returned results
        trimmed = results[: query.max_results]
        trimmed_ids = {r.concept_id for r in trimmed}
        concept_cache = {k: v for k, v in concept_cache.items() if k in trimmed_ids}
        _record_search_stage_latency(
            "tfidf_sort_trim",
            (time.perf_counter() - _sort_t0) * 1000,
            path="tfidf",
            result_count=len(trimmed),
            candidate_count=len(raw_results),
        )
        return trimmed, concept_cache

    def _supplement_ka_coverage(
        self,
        results: list[SearchResult],
        query: SearchQuery,
        concept_cache: dict,
        *,
        path: str = "embedding",
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
        dominant_kas = [ka for ka, count in ka_counts.items() if count >= 2 and ka not in ("general", "unclassified")]
        if not dominant_kas:
            return results, concept_cache

        # Step 2: Fetch concepts from dominant KA(s) not already in results
        # T2-3: read_snapshot_db — RLock-free KA supplement read
        existing_ids = {r.concept_id for r in results}
        scored_candidates: list[dict[str, Any]] = []
        scanned_count = 0
        budget_exhausted = False
        supplement_t0 = time.perf_counter()

        # Pre-compute query embedding once (outside loop)
        query_vec = None
        if self._embeddings_available:
            query_vec = embedding_engine.embed_text(query.query)

        with read_snapshot_db("supplement_ka") as conn:
            for ka in dominant_kas[:_KA_SUPPLEMENT_MAX_KAS]:
                if len(scored_candidates) >= _KA_SUPPLEMENT_MAX_ELIGIBLE_CANDIDATES:
                    budget_exhausted = True
                    break
                elapsed_ms = (time.perf_counter() - supplement_t0) * 1000
                if elapsed_ms >= _KA_SUPPLEMENT_BUDGET_MS:
                    budget_exhausted = True
                    break

                placeholders = ",".join("?" * len(existing_ids)) if existing_ids else "''"
                rows = conn.execute(
                    f"""SELECT id, summary, confidence, knowledge_area
                       FROM concepts
                       WHERE knowledge_area = ?
                         AND is_current = 1
                         AND confidence >= ?
                         AND id NOT IN ({placeholders})
                       ORDER BY confidence DESC
                       LIMIT ?""",
                    [ka, query.min_confidence] + list(existing_ids) + [_KA_SUPPLEMENT_SQL_LIMIT_PER_KA],
                ).fetchall()

                for row in rows:
                    scanned_count += 1
                    if len(scored_candidates) >= _KA_SUPPLEMENT_MAX_ELIGIBLE_CANDIDATES:
                        budget_exhausted = True
                        break
                    elapsed_ms = (time.perf_counter() - supplement_t0) * 1000
                    if elapsed_ms >= _KA_SUPPLEMENT_BUDGET_MS:
                        budget_exhausted = True
                        break

                    concept_id, summary, confidence, concept_ka = row
                    # Score via embedding similarity to query
                    if query_vec is not None and concept_id in embedding_engine._id_to_pos:
                        pos = embedding_engine._id_to_pos[concept_id]
                        emb_score = float(embedding_engine._index_matrix[pos] @ query_vec)
                    else:
                        emb_score = 0.10  # Fallback: low but non-zero

                    if emb_score < _KA_SUPPLEMENT_MIN_SIMILARITY:  # Lower threshold than Phase 1 (0.20)
                        continue

                    scored_candidates.append(
                        {
                            "concept_id": concept_id,
                            "summary": summary,
                            "confidence": confidence,
                            "knowledge_area": concept_ka or ka,
                            "relevance_score": emb_score,
                        }
                    )

        scored_candidates.sort(key=lambda item: (-item["relevance_score"], item["concept_id"]))
        admitted_candidates = scored_candidates[:_KA_SUPPLEMENT_MAX_ADDITIONS]
        admitted_ids = [item["concept_id"] for item in admitted_candidates]
        hydrated = load_concepts_batch(admitted_ids)
        supplement_results = []

        for item in admitted_candidates:
            concept_id = item["concept_id"]
            concept = hydrated.get(concept_id)
            if not concept:
                continue

            concept_cache[concept_id] = concept
            supplement_results.append(
                SearchResult(
                    concept_id=concept_id,
                    version=getattr(concept, "version", "v1"),
                    summary=item["summary"],
                    confidence=item["confidence"],
                    relevance_score=item["relevance_score"],
                    knowledge_area=item["knowledge_area"],
                    ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                    maturity=getattr(concept, "maturity", None),
                    created_at=getattr(concept, "created_at", None),
                    metadata=_diagnostic_source_metadata(concept),
                )
            )

        _record_search_stage_latency(
            "ka_cross_session_supplement",
            (time.perf_counter() - supplement_t0) * 1000,
            path=path,
            result_count=len(supplement_results),
            candidate_count=len(scored_candidates),
            extra_labels={
                "accepted_count": len(supplement_results),
                "scanned_count": scanned_count,
                "budget_exhausted": budget_exhausted,
            },
        )

        if supplement_results:
            logger.info(
                "RETRIEVAL-032: KA supplement added %d concepts from KA(s) %s",
                len(supplement_results),
                dominant_kas,
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
        *,
        search_path: str = "search",
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
        _post_t0 = _time_mod.perf_counter()

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
            _record_search_stage_latency(
                "post_predictive_boost",
                (_pp_t1 - _pp_t0) * 1000,
                path=search_path,
                result_count=len(results),
            )
            # PERF-018: pass concept_cache to avoid N DB reads (PERF-016 cache was bypassed)
            scored = goal_directed.boost_scores_by_goal(scored, concept_cache=concept_cache)
            _pp_t2 = _time_mod.perf_counter()
            _record_search_stage_latency(
                "post_goal_boost",
                (_pp_t2 - _pp_t1) * 1000,
                path=search_path,
                result_count=len(results),
            )
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
        _sal_t0 = _time_mod.perf_counter()
        try:
            from app.retrieval.salience import apply_sal_multiplier

            for result in results:
                concept = _get_concept(result.concept_id)  # PERF-016: cache-first
                if concept:
                    sal = getattr(concept, "salience", 0.5) or 0.5
                    result.relevance_score = apply_sal_multiplier(sal, result.relevance_score)
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))  # RETRIEVAL-037b v4.2: deterministic
        except ImportError:
            pass
        finally:
            _record_search_stage_latency(
                "post_salience",
                (_time_mod.perf_counter() - _sal_t0) * 1000,
                path=search_path,
                result_count=len(results),
            )

        # Wave 4b §4b.3: Preference salience floor
        _pref_t0 = _time_mod.perf_counter()
        try:
            from app.retrieval.provenance import apply_preference_floor

            for result in results:
                concept = _get_concept(result.concept_id)  # PERF-016: cache-first
                if concept and concept.concept_type == "preference":
                    floor_sal = apply_preference_floor(concept, getattr(concept, "salience", 0.5))
                    if floor_sal > (getattr(concept, "salience", 0.5) or 0.5):
                        result.relevance_score = max(result.relevance_score, 0.3)
        except ImportError:
            pass
        finally:
            _record_search_stage_latency(
                "post_preference_floor",
                (_time_mod.perf_counter() - _pref_t0) * 1000,
                path=search_path,
                result_count=len(results),
            )

        # INGEST-015: Fact-seeking query boost — is_factual concepts score ×FACT_SEEKING_BOOST
        _fact_t0 = _time_mod.perf_counter()
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
        finally:
            _record_search_stage_latency(
                "post_fact_boost",
                (_time_mod.perf_counter() - _fact_t0) * 1000,
                path=search_path,
                result_count=len(results),
            )

        _authority_t0 = _time_mod.perf_counter()
        _apply_authority_artifact_boost(results, query.query, path="search")
        _record_search_stage_latency(
            "post_authority_artifact_boost",
            (_time_mod.perf_counter() - _authority_t0) * 1000,
            path=search_path,
            result_count=len(results),
        )
        _record_search_stage_latency(
            "post_processing_internal_total",
            (_time_mod.perf_counter() - _post_t0) * 1000,
            path=search_path,
            result_count=len(results),
        )

        # WS2: Metric 7 — retrieval_search_latency_ms
        try:
            from app.core.metrics_facade import metrics as _m7

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

    @staticmethod
    def _compute_recency_score(concept) -> float:
        """DEBT-231: Shared recency score — eliminates duplication between
        _calculate_score and _governance_score.

        Returns exponential decay [0.0, 1.0] based on concept age,
        0.0 if recency weighting disabled, 0.5 as neutral fallback on error.
        """
        from app.core.config import RETRIEVAL_RECENCY_HALF_LIFE_DAYS, RETRIEVAL_WEIGHT_RECENCY

        if not (RETRIEVAL_WEIGHT_RECENCY > 0 and concept.created_at):
            return 0.0
        try:
            from datetime import datetime as _dt

            _ca_str = concept.created_at if isinstance(concept.created_at, str) else str(concept.created_at)
            _ca_dt = _ensure_aware(_dt.fromisoformat(_ca_str.replace("Z", "+00:00")))
            _age_days = max(0.0, (_utc_now() - _ca_dt).total_seconds() / 86400.0)
            _hl = max(0.1, RETRIEVAL_RECENCY_HALF_LIFE_DAYS)
            return math.exp(-math.log(2) / _hl * _age_days)
        except Exception:
            logger.debug("recency_score_parse_error concept_id=%s", getattr(concept, "id", "?"))
            return 0.5  # Safe fallback: neutral recency

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

                # RETRIEVAL-080: Read utility score (default to cold start if not yet populated)
                _utility = getattr(concept, "utility_score", None)
                if _utility is None:
                    _utility = UTILITY_COLD_START

                # RETRIEVAL-100: Creation-time recency signal (DEBT-231: extracted to _compute_recency_score)
                _recency_score = self._compute_recency_score(concept)

                score = (
                    emb_score * RETRIEVAL_WEIGHT_SIMILARITY
                    + authority * RETRIEVAL_WEIGHT_AUTHORITY
                    + currency * RETRIEVAL_WEIGHT_CURRENCY
                    + concept.confidence * RETRIEVAL_WEIGHT_CONFIDENCE
                    + concept.stability * RETRIEVAL_WEIGHT_STABILITY
                    + context_boost * RETRIEVAL_WEIGHT_CONTEXT
                    + goal_boost * RETRIEVAL_WEIGHT_GOAL
                    + _utility * RETRIEVAL_WEIGHT_UTILITY  # RETRIEVAL-080: Learned from feedback
                    + _recency_score * RETRIEVAL_WEIGHT_RECENCY  # RETRIEVAL-100: Creation-time recency
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
                query_text = getattr(query, "query", "") or ""
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
        with self._writer_admission("build_index") as admit:
            if not admit:
                return
            logger.info("Building incremental index from all concepts...")

            # Build into the live index, then persist.
            added = self._build_into(self.index)

            # Save
            self._auto_save()

            logger.info(f"Index built: {added} concepts indexed")

    def _build_into(self, target_index) -> int:
        """Populate ``target_index`` from all concepts in storage.

        Behavior-preserving extraction of the concept-iteration + add_concept +
        force_idf_recalculation loop from ``build_index``. Used by both
        ``build_index`` (target=self.index) and the fresh-rebuild repair
        routine (target=a throwaway IncrementalTfidfIndex).

        Returns the number of concepts indexed.
        """
        # Single query for all concepts — eliminates N+1 load_concept calls
        concepts = list_concepts_full()
        added = 0

        # Suppress intermediate IDF recalcs during bulk add — we do one
        # final recalculation at the end. Saves O(N) per threshold crossing.
        original_threshold = target_index.idf_update_threshold
        target_index.idf_update_threshold = len(concepts) + 100

        try:
            for concept in concepts:
                if concept.id in target_index.concept_id_to_idx:
                    continue  # Already indexed, skip (Fix 1)
                searchable_text = self._concept_to_document(concept)
                if target_index.add_concept(concept.id, searchable_text):
                    added += 1
                    if added % 50 == 0:
                        logger.info(f"Indexed {added} concepts...")
        finally:
            # Restore threshold for incremental adds during runtime
            target_index.idf_update_threshold = original_threshold

        # Single IDF recalculation over complete corpus
        target_index.force_idf_recalculation()

        return added

    def _concept_to_document(self, concept) -> str:
        """
        Convert concept to searchable text.

        Handles both legacy string evidence and v2 Evidence objects (stored as dicts).
        """
        return build_searchable_text_from_concept(concept)

    def _auto_save(self, *, strict: bool = False) -> bool:
        """Auto-save the index; optionally propagate failure to a retry owner."""
        try:
            self.index.save(self.index_path)
            logger.debug(f"Auto-saved index to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            if strict:
                raise
            return False

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
        # T2-3: read_snapshot_db — RLock-free dedup read
        with read_snapshot_db("dedup_tfidf") as _dedup_conn:
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
        # T2-3: read_snapshot_db — RLock-free batch dedup read
        concept_meta: dict[str, dict] = {}
        with read_snapshot_db("dedup_tfidf_batch") as conn:
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
        with read_snapshot_db("dedup_embedding") as _dedup_conn:  # T2-3: RLock-free (was: _db, DEBT-189)
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
                        "summary": _dedup_data.get(
                            "summary", ""
                        ),  # CONTRA-018: L1.8 needs summary for opposition check
                    }
                )

        return results

    def search_for_dedup_embedding_batch(self, query_texts: list[str], top_k: int = 3) -> list[list[dict]]:
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
                "search_for_dedup_embedding_batch: embeddings unavailable, falling back to sequential TF-IDF"
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
            with read_snapshot_db("dedup_embedding_batch") as conn:  # T2-3
                rows = conn.execute(
                    f"SELECT id, data, knowledge_area FROM concepts WHERE id IN ({placeholders}) AND is_current = 1",
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
                    query_results.append(
                        {
                            "concept_id": cid,
                            "cosine_score": score,
                            "knowledge_area": concept_data[cid]["knowledge_area"],
                            "evidence_count": concept_data[cid]["evidence_count"],
                        }
                    )
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
        from app.core.config import (
            RETRIEVAL_WEIGHT_AUTHORITY,
            RETRIEVAL_WEIGHT_CONFIDENCE,
            RETRIEVAL_WEIGHT_CURRENCY,
            RETRIEVAL_WEIGHT_RECENCY,  # RETRIEVAL-100
            RETRIEVAL_WEIGHT_SIMILARITY,  # DEBT-002
            RETRIEVAL_WEIGHT_STABILITY,
            STALE_RISK_AGING_PENALTY_ENABLED,
            STALE_RISK_PENALTY_AGING,
            STALE_RISK_PENALTY_REVIEW,
            STALE_RISK_REVIEW_PENALTY_ENABLED,
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

        # RETRIEVAL-100: Creation-time recency signal (DEBT-231: extracted to _compute_recency_score)
        _recency_score = RetrievalEngine._compute_recency_score(concept)

        score = (
            raw_score * RETRIEVAL_WEIGHT_SIMILARITY
            + _authority * RETRIEVAL_WEIGHT_AUTHORITY
            + _currency * RETRIEVAL_WEIGHT_CURRENCY
            + concept.confidence * RETRIEVAL_WEIGHT_CONFIDENCE
            + concept.stability * RETRIEVAL_WEIGHT_STABILITY
            + _recency_score * RETRIEVAL_WEIGHT_RECENCY  # RETRIEVAL-100
        )
        # FRESHNESS_UNIFIED_REDESIGN: Exponential decay freshness bonus
        from app.core.config import (
            RETRIEVAL_FRESHNESS_EVOLUTION_BONUS,
            RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS,
            RETRIEVAL_FRESHNESS_MAX_BONUS,
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
                logger.debug(
                    "freshness_bonus_parse_error concept_id=%s ts=%s", getattr(concept, "id", "?"), _freshness_ts
                )

        # Evolution bonus: evolved concepts get a small additional boost
        if concept.version and concept.version != "v1":
            score += RETRIEVAL_FRESHNESS_EVOLUTION_BONUS

        # RETRIEVAL-034 Layer 2: Soft ranking penalty for CONTRADICTED/CONTESTED
        # Deprioritize but don't exclude — stale knowledge is valuable context.
        # RESOLVED is treated as ACTIVE (no penalty). Applied after all additive
        # bonuses, before min(1.0) cap.
        from app.core.config import STALE_PENALTY_CONTESTED, STALE_PENALTY_CONTRADICTED, STALE_TRANSPARENCY_ENABLED

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

        _staleness_state = getattr(concept, "staleness_state", None)
        if _staleness_state == "AGING" and STALE_RISK_AGING_PENALTY_ENABLED:
            score *= STALE_RISK_PENALTY_AGING
        elif _staleness_state == "REVIEW" and STALE_RISK_REVIEW_PENALTY_ENABLED:
            score *= STALE_RISK_PENALTY_REVIEW

        return min(1.0, score)

    def search_lightweight(
        self,
        query_text: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
        agent_id: str = None,
        scope: str = "global",
        include_deprecated: bool = False,
        session_id: str | None = None,  # SESSION-012: Cross-session awareness
        deadline: TurnDeadline | None = None,
        query_intent_source_query: str | None = None,
        query_intent_expansion_enabled: bool = True,
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
        _slw_min_remaining_ms = _env_float("PITH_TURN_DEADLINE_MIN_RETRIEVAL_MS", 250.0)
        _slw_min_embedding_init_ms = _env_float("PITH_SLW_MIN_EMBEDDING_INIT_MS", 100.0)
        _slw_min_embedding_search_ms = _env_float("PITH_SLW_MIN_EMBEDDING_SEARCH_MS", 500.0)
        _slw_embedding_search_p95_limit_ms = _env_float(
            "PITH_FOREGROUND_EMBEDDING_SEARCH_P95_LIMIT_MS",
            500.0,
        )
        _slw_embedding_search_circuit_ttl_s = _env_float(
            "PITH_FOREGROUND_EMBEDDING_SEARCH_CIRCUIT_TTL_S",
            60.0,
        )
        _slw_embedding_search_cold_skip_enabled = _env_flag(
            "PITH_FOREGROUND_EMBEDDING_SEARCH_COLD_SKIP_ENABLED",
            True,
        )
        _slw_min_batch_load_ms = _env_float("PITH_SLW_MIN_BATCH_LOAD_MS", 250.0)
        _slw_embedding_search_admission_enabled = _env_flag(
            "PITH_SLW_EMBEDDING_SEARCH_ADMISSION_ENABLED",
            True,
        )
        _slw_foreground_embedding_enabled = _env_flag(
            "PITH_FOREGROUND_EMBEDDING_ENABLED",
            True,
        )
        _slw_foreground_embedding_bounded_enabled = _env_flag(
            "PITH_FOREGROUND_EMBEDDING_BOUNDED_ENABLED",
            False,
        )
        _slw_foreground_embedding_timeout_ms = max(
            50.0,
            min(
                2000.0,
                _env_float("PITH_FOREGROUND_EMBEDDING_SEARCH_TIMEOUT_MS", 600.0),
            ),
        )
        _slw_semantic_recovery_enabled = get_feature_flag("LIVE_SEMANTIC_RECOVERY_ENABLED", False)
        _slw_semantic_recovery_top_k = _env_int_clamped(
            "PITH_SLW_SEMANTIC_RECOVERY_TOP_K",
            10,
            1,
            50,
        )
        _slw_semantic_recovery_max_admitted = _env_int_clamped(
            "PITH_SLW_SEMANTIC_RECOVERY_MAX_ADMITTED",
            10,
            1,
            20,
        )
        _slw_semantic_recovery_min_score = _env_float(
            "PITH_SLW_SEMANTIC_RECOVERY_MIN_SCORE",
            MIN_RETRIEVAL_SIMILARITY,
        )
        _slw_min_semantic_recovery_ms = _env_float("PITH_SLW_MIN_SEMANTIC_RECOVERY_MS", 750.0)
        _slw_semantic_recovery_p95_limit_ms = _env_float(
            "PITH_FOREGROUND_SEMANTIC_RECOVERY_P95_LIMIT_MS",
            750.0,
        )
        _slw_semantic_recovery_circuit_ttl_s = _env_float(
            "PITH_FOREGROUND_SEMANTIC_RECOVERY_CIRCUIT_TTL_S",
            60.0,
        )
        _slw_foreground_embedding_init_enabled = _env_flag(
            "PITH_SLW_FOREGROUND_EMBEDDING_INIT_ENABLED",
            False,
        )
        self.last_canary_search_lightweight_trace = None
        self.last_live_initial_admission_probe_trace = None
        self.last_query_intent_trace = None
        self.last_lexical_evidence_support_trace = None
        intent_expansion: QueryIntentExpansion | None = None
        effective_query_text = query_text
        if query_intent_expansion_enabled and get_feature_flag("QUERY_INTENT_EXPANSION_ENABLED", True):
            try:
                if query_intent_source_query is not None:
                    intent_expansion = expand_query_intent(
                        query_intent_source_query,
                        assembled_query=query_text,
                        input_scope="raw_user_message",
                        expansion_input_source="query_intent_source_query",
                    )
                else:
                    intent_expansion = expand_query_intent(
                        query_text,
                        input_scope="query_argument",
                        expansion_input_source="query_text",
                    )
                if intent_expansion.matched_aliases or intent_expansion.contamination_guard_blocked:
                    effective_query_text = intent_expansion.expanded_query
                    self.last_query_intent_trace = intent_expansion.to_trace()
                    if intent_expansion.matched_aliases:
                        _record_metric(
                            "query_intent.alias_match_total",
                            float(len(intent_expansion.matched_aliases)),
                            {"path": "search_lightweight", "source": intent_expansion.source},
                        )
                    if intent_expansion.contamination_guard_blocked:
                        _record_metric(
                            "query_intent.contamination_guard_blocked_total",
                            1.0,
                            {"path": "search_lightweight", "source": intent_expansion.source},
                        )
            except Exception as _qie:
                logger.debug("QUERY-INTENT: search_lightweight expansion failed (non-fatal): %s", _qie)
                effective_query_text = query_text
        authority_expanded_query = _expand_authority_artifact_query(effective_query_text)
        if authority_expanded_query != effective_query_text:
            effective_query_text = authority_expanded_query
            _record_metric(
                "retrieval_authority_artifact_query_expanded_total",
                1.0,
                {"path": "search_lightweight"},
            )
        _slw_trace: dict | None = None
        if _mh262_canary_retrieval_trace_enabled():
            _slw_trace = {
                "schema_version": "mh262.search_lightweight_trace.v1",
                "limit": _MH262_CANARY_TRACE_LIMIT,
                "target_ids": _mh262_canary_trace_target_ids(),
                "stages": {},
                "predictive_activation": {},
            }
            if ENHANCED_RETRIEVAL and predictive_activation is not None:
                _slw_trace["predictive_activation"]["entry"] = _mh262_predictive_activation_snapshot(
                    predictive_activation
                )
            self.last_canary_search_lightweight_trace = _slw_trace
        _slw_timed_out = False
        _slw_timeout_metric_recorded = False
        _slw_timeout_min_results = max(0, int(_env_float("PITH_SLW_TIMEOUT_MIN_RESULTS", 1.0)))
        _slw_timeout_candidate_limit = max(
            1,
            int(_env_float("PITH_SLW_TIMEOUT_CANDIDATE_LIMIT", float(min(max(top_k, 1), 3)))),
        )
        _live_probe_trace_enabled = (
            _env_flag("PITH_LIVE_INITIAL_ADMISSION_PROBE_TRACE", False) or _slw_trace is not None
        )
        _live_probe_trace: dict[str, Any] | None = None
        if _live_probe_trace_enabled:
            _live_probe_trace = {
                "schema_version": "retrieval.live_initial_admission_probe.v1",
                "enabled": True,
                "behavior_changed": False,
                "semantic_warm_readiness": self.semantic_warm_readiness_snapshot(),
                "query_hash": hashlib.sha256(effective_query_text.encode("utf-8")).hexdigest()[:12],
                "top_k": int(top_k),
                "min_confidence": float(min_confidence),
                "search_lightweight_skipped_reason": None,
                "embedding": {
                    "available": bool(getattr(self, "_embeddings_available", False)),
                    "initialized": bool(getattr(self, "_embeddings_initialized", False)),
                    "index_size": int(getattr(embedding_engine, "index_size", 0) or 0),
                    "init_skipped_reason": None,
                    "search_started": False,
                    "search_skipped_reason": None,
                    "raw_count": 0,
                    "candidate_id_count": 0,
                    "loaded_count": 0,
                    "score_floor_rejected_count": 0,
                    "confidence_rejected_count": 0,
                    "governance_rejected_count": 0,
                    "result_count": 0,
                },
                "tfidf": {
                    "started": False,
                    "skipped_reason": None,
                    "fallback_reason": None,
                    "fallback_mode": "unknown",
                    "document_count": int(getattr(self.index, "document_count", 0) or 0),
                    "raw_count": 0,
                    "candidate_id_count": 0,
                    "loaded_count": 0,
                    "score_floor_rejected_count": 0,
                    "confidence_rejected_count": 0,
                    "governance_rejected_count": 0,
                    "result_count": 0,
                },
                "soft_timeout": {
                    "fired": False,
                    "min_results": int(_slw_timeout_min_results),
                    "candidate_limit": int(_slw_timeout_candidate_limit),
                    "partial_result_count": 0,
                },
                "lexical": None,
                "semantic_recovery": {
                    "enabled": bool(_slw_semantic_recovery_enabled),
                    "attempted": False,
                    "reason": "not_attempted",
                    "trigger_reason": None,
                    "top_k": int(_slw_semantic_recovery_top_k),
                    "max_admitted": int(_slw_semantic_recovery_max_admitted),
                    "min_score": float(_slw_semantic_recovery_min_score),
                    "search_started": False,
                    "raw_count": 0,
                    "candidate_id_count": 0,
                    "loaded_count": 0,
                    "score_floor_rejected_count": 0,
                    "confidence_rejected_count": 0,
                    "governance_rejected_count": 0,
                    "duplicate_rejected_count": 0,
                    "admitted_count": 0,
                    "admitted_ids": [],
                    "latency_ms": None,
                },
                "final_result_count": 0,
            }

        def _probe_set(section: str, key: str, value: Any) -> None:
            if _live_probe_trace is None:
                return
            target = _live_probe_trace.get(section)
            if isinstance(target, dict):
                target[key] = value

        def _probe_inc(section: str, key: str, amount: int = 1) -> None:
            if _live_probe_trace is None:
                return
            target = _live_probe_trace.get(section)
            if isinstance(target, dict):
                target[key] = int(target.get(key, 0) or 0) + amount

        def _finish_live_probe(results_for_trace: list[SearchResult]) -> list[SearchResult]:
            if _live_probe_trace is None:
                return results_for_trace
            _live_probe_trace["final_result_count"] = len(results_for_trace or [])
            _live_probe_trace["lexical"] = self.last_lexical_evidence_support_trace
            _live_probe_trace["semantic_warm_readiness"] = self.semantic_warm_readiness_snapshot()
            soft_timeout = _live_probe_trace.get("soft_timeout")
            if isinstance(soft_timeout, dict):
                soft_timeout["fired"] = bool(_slw_timed_out)
            self.last_live_initial_admission_probe_trace = _live_probe_trace
            path = "none"
            reason = str(_live_probe_trace.get("search_lightweight_skipped_reason") or "none")
            embedding_trace = _live_probe_trace.get("embedding")
            tfidf_trace = _live_probe_trace.get("tfidf")
            if isinstance(embedding_trace, dict) and embedding_trace.get("result_count"):
                path = "embedding"
            elif isinstance(tfidf_trace, dict) and tfidf_trace.get("result_count"):
                path = "tfidf"
            elif isinstance(tfidf_trace, dict) and tfidf_trace.get("started"):
                path = "tfidf"
                reason = str(tfidf_trace.get("skipped_reason") or tfidf_trace.get("fallback_reason") or reason)
            elif isinstance(embedding_trace, dict) and embedding_trace.get("search_started"):
                path = "embedding"
                reason = str(embedding_trace.get("search_skipped_reason") or reason)
            _record_metric(
                "retrieval_live_initial_admission_probe_emitted_total", 1.0, {"path": path, "reason": reason}
            )
            if len(results_for_trace or []) == 0:
                _record_metric(
                    "retrieval_live_initial_admission_probe_zero_result_total",
                    1.0,
                    {"path": path, "reason": reason},
                )
            if isinstance(embedding_trace, dict) and embedding_trace.get("search_skipped_reason"):
                _record_metric(
                    "retrieval_live_initial_admission_probe_embedding_skipped_total",
                    1.0,
                    {"path": "embedding", "reason": str(embedding_trace.get("search_skipped_reason"))},
                )
            if isinstance(tfidf_trace, dict) and tfidf_trace.get("fallback_reason"):
                _record_metric(
                    "retrieval_live_initial_admission_probe_tfidf_fallback_total",
                    1.0,
                    {"path": "tfidf", "reason": str(tfidf_trace.get("fallback_reason"))},
                )
            return results_for_trace

        if deadline and not deadline.can_start(
            "retrieval.search_lightweight",
            min_remaining_ms=_slw_min_remaining_ms,
        ):
            if _live_probe_trace is not None:
                _live_probe_trace["search_lightweight_skipped_reason"] = "deadline_before_start"
            deadline.skip(
                "retrieval.search_lightweight",
                "deadline_before_start",
                priority="optional",
                min_remaining_ms=_slw_min_remaining_ms,
            )
            return _finish_live_probe([])

        # Ensure embedding index is ready when already warm. In deadline-bound
        # turn paths, avoid minutes-long foreground hydration; TF-IDF remains
        # available while startup warms semantic embeddings in the background.
        if deadline and not self._embeddings_initialized and not _slw_foreground_embedding_init_enabled:
            self.record_semantic_warm_readiness(
                state="skipped",
                source="foreground",
                result="foreground_embedding_init_disabled",
            )
            _probe_set("embedding", "init_skipped_reason", "foreground_embedding_init_disabled")
            deadline.skip(
                "retrieval.embedding_init",
                "foreground_embedding_init_disabled",
                priority="optional",
                min_remaining_ms=_slw_min_embedding_init_ms,
            )
            _record_metric(
                "search_lightweight.embedding_init_ms",
                0.0,
                {"path": "tfidf", "reason": "foreground_disabled"},
            )
        else:
            if deadline and not deadline.can_start(
                "retrieval.embedding_init",
                min_remaining_ms=_slw_min_embedding_init_ms,
            ):
                deadline.skip(
                    "retrieval.embedding_init",
                    "deadline_before_start",
                    priority="optional",
                    min_remaining_ms=_slw_min_embedding_init_ms,
                )
                _probe_set("embedding", "init_skipped_reason", "deadline_before_start")
                self.record_semantic_warm_readiness(
                    state="skipped",
                    source="foreground",
                    result="deadline_before_start",
                )
                return _finish_live_probe([])
            _embedding_init_start = time.perf_counter()
            self.record_semantic_warm_readiness(
                state="running",
                source="foreground",
                result="foreground_init_started",
            )
            self._init_embeddings()
            _embedding_available = bool(getattr(self, "_embeddings_available", False))
            self.record_semantic_warm_readiness(
                state="ready" if _embedding_available else "skipped",
                source="foreground",
                result=("foreground_init_complete" if _embedding_available else "foreground_init_unavailable"),
            )
            _probe_set("embedding", "available", bool(getattr(self, "_embeddings_available", False)))
            _probe_set("embedding", "initialized", bool(getattr(self, "_embeddings_initialized", False)))
            _probe_set("embedding", "index_size", int(getattr(embedding_engine, "index_size", 0) or 0))
            _record_metric(
                "search_lightweight.embedding_init_ms",
                round((time.perf_counter() - _embedding_init_start) * 1000.0, 2),
                {"path": "embedding" if getattr(self, "_embeddings_available", False) else "tfidf"},
            )

        # ===== SESSION-012: Concurrent session detection =====
        # Post-scoring additive boost for concepts from sibling sessions.
        # Composable with _apply_ka_boost — both are post-scoring, independent.
        _concurrent_ids: set[str] = set()
        if session_id and get_feature_flag("CROSS_SESSION_BOOST_ENABLED", False):
            try:
                with read_snapshot_db("search_cross_session") as _cs_conn:  # T2-3
                    _cs_rows = _cs_conn.execute(
                        """SELECT id FROM sessions
                           WHERE status IN ('active', 'interrupted')
                             AND id != ?
                             AND started_at > datetime('now', ? || ' hours')""",
                        (session_id, f"-{CROSS_SESSION_WINDOW_HOURS}"),
                    ).fetchall()
                    _concurrent_ids = {r[0] for r in _cs_rows}
            except Exception as _cs_e:
                logger.debug(f"SESSION-012: concurrent session query failed (non-fatal): {_cs_e}")

        # ===== Phase 1: Get raw results =====
        # OPT-1c: Soft timeout — if Phase 1 exceeds budget, return partial results.
        # Gauntlet A11: Check inside loop, not after. A15: Floor clamp 100ms.
        import time as _time_mod_slw

        _slw_start = _time_mod_slw.perf_counter()
        _slw_soft_timeout_ms = _env_float("PITH_SLW_SOFT_TIMEOUT_MS", 500.0)
        if deadline:
            _slw_soft_timeout_ms = deadline.child_budget_ms(
                "retrieval.search_lightweight",
                requested_ms=_slw_soft_timeout_ms,
                min_remaining_ms=0.0,
            )
            if _slw_soft_timeout_ms <= 0:
                deadline.skip(
                    "retrieval.search_lightweight",
                    "deadline_child_budget_exhausted",
                    priority="optional",
                )
                if _live_probe_trace is not None:
                    _live_probe_trace["search_lightweight_skipped_reason"] = "deadline_child_budget_exhausted"
                return _finish_live_probe([])
            _slw_soft_timeout_s = max(0.001, _slw_soft_timeout_ms) / 1000.0
        else:
            _slw_soft_timeout_s = max(0.1, _slw_soft_timeout_ms) / 1000.0
        _slw_lexical_protected_ids: set[str] = set()
        _slw_semantic_recovery_protected_ids: set[str] = set()

        def _slw_should_check_timeout(candidate_index: int) -> bool:
            return _slw_timed_out or candidate_index == 0 or candidate_index % 10 == 0

        def _slw_should_stop_for_timeout(candidate_index: int, materialized_count: int) -> bool:
            if _slw_timeout_min_results <= 0:
                return True
            return materialized_count >= _slw_timeout_min_results or candidate_index >= _slw_timeout_candidate_limit

        def _record_slw_soft_timeout(
            path: str,
            candidate_index: int,
            candidate_total: int,
            elapsed_ms: float,
            partial_results: int,
            action: str,
        ) -> None:
            nonlocal _slw_timeout_metric_recorded
            if _live_probe_trace is not None:
                soft_timeout = _live_probe_trace.get("soft_timeout")
                if isinstance(soft_timeout, dict):
                    soft_timeout["fired"] = True
                    soft_timeout["partial_result_count"] = int(partial_results)
            if _slw_timeout_metric_recorded:
                return
            _slw_timeout_metric_recorded = True
            labels = {
                "path": path,
                "action": action,
                "candidate_index": candidate_index,
                "candidate_total": candidate_total,
                "partial_results": partial_results,
            }
            _record_metric(
                "search_lightweight.soft_timeout_total",
                1.0,
                labels,
                flush=True,
            )
            _record_metric(
                "search_lightweight.soft_timeout_elapsed_ms",
                round(elapsed_ms, 2),
                {"path": path, "action": action},
                flush=True,
            )

        def _record_slw_fallback_denied(reason: str, mode: str, denied_reason: str) -> None:
            _record_metric(
                "search_lightweight.fallback_denied_total",
                1.0,
                {"from": "embedding", "to": "tfidf", "reason": reason, "mode": mode, "denied": denied_reason},
            )

        def _semantic_recovery_trace() -> dict[str, Any] | None:
            if _live_probe_trace is None:
                return None
            trace = _live_probe_trace.get("semantic_recovery")
            return trace if isinstance(trace, dict) else None

        def _semantic_recovery_set(key: str, value: Any) -> None:
            trace = _semantic_recovery_trace()
            if trace is not None:
                trace[key] = value

        def _semantic_recovery_inc(key: str, amount: int = 1) -> None:
            trace = _semantic_recovery_trace()
            if trace is not None:
                trace[key] = int(trace.get(key, 0) or 0) + amount

        def _finish_semantic_recovery(reason: str, admitted_count: int = 0, *, attempted: bool = False) -> None:
            _semantic_recovery_set("reason", reason)
            _semantic_recovery_set("admitted_count", int(admitted_count))
            _record_metric(
                "search_lightweight.semantic_recovery_attempt_total",
                1.0,
                {
                    "reason": reason,
                    "attempted": str(bool(attempted)).lower(),
                },
            )
            if admitted_count:
                _record_metric(
                    "search_lightweight.semantic_recovery_admitted_total",
                    float(admitted_count),
                    {"reason": reason},
                )

        def _run_tfidf_path(fallback_reason: str | None = None, fallback_mode: str = "unknown") -> list[SearchResult]:
            nonlocal _slw_timed_out
            _probe_set("tfidf", "started", True)
            _probe_set("tfidf", "fallback_reason", fallback_reason)
            _probe_set("tfidf", "fallback_mode", fallback_mode)
            _probe_set("tfidf", "document_count", int(getattr(self.index, "document_count", 0) or 0))
            if self.index.document_count == 0:
                _probe_set("tfidf", "skipped_reason", "empty_index")
                if fallback_reason:
                    _record_slw_fallback_denied(fallback_reason, fallback_mode, "empty_index")
                return []
            if deadline and not deadline.can_start(
                "retrieval.tfidf_search",
                min_remaining_ms=_slw_min_remaining_ms,
            ):
                deadline.skip(
                    "retrieval.tfidf_search",
                    "deadline_before_start",
                    priority="optional",
                    min_remaining_ms=_slw_min_remaining_ms,
                )
                _probe_set("tfidf", "skipped_reason", "deadline_before_start")
                if fallback_reason:
                    _record_slw_fallback_denied(fallback_reason, fallback_mode, "deadline_before_start")
                return []
            _tfidf_search_start = time.perf_counter()
            raw_results_tfidf = self.index.search(effective_query_text, top_k=top_k)
            _probe_set("tfidf", "raw_count", len(raw_results_tfidf or []))
            _record_metric(
                "search_lightweight.tfidf_search_ms",
                round((time.perf_counter() - _tfidf_search_start) * 1000.0, 2),
                {"fallback_reason": fallback_reason or "direct", "fallback_mode": fallback_mode},
            )

            # PERF-076: Batch load all candidate concepts in one query
            _candidate_ids_tfidf = [cid for cid, score in raw_results_tfidf if score >= MIN_RETRIEVAL_SIMILARITY * 0.5]
            _probe_set("tfidf", "candidate_id_count", len(_candidate_ids_tfidf))
            if deadline and not deadline.can_start(
                "retrieval.load_concepts_batch",
                min_remaining_ms=_slw_min_batch_load_ms,
            ):
                deadline.skip(
                    "retrieval.load_concepts_batch",
                    "deadline_before_start",
                    priority="optional",
                    min_remaining_ms=_slw_min_batch_load_ms,
                )
                _probe_set("tfidf", "skipped_reason", "batch_deadline_before_start")
                if fallback_reason:
                    _record_slw_fallback_denied(fallback_reason, fallback_mode, "batch_deadline_before_start")
                return []
            from app.storage.concepts import load_concepts_batch

            _batch_load_start = time.perf_counter()
            _batch_cache_tfidf = load_concepts_batch(_candidate_ids_tfidf)
            _probe_set("tfidf", "loaded_count", len(_batch_cache_tfidf or {}))
            _record_metric(
                "search_lightweight.batch_load_ms",
                round((time.perf_counter() - _batch_load_start) * 1000.0, 2),
                {"path": "tfidf"},
            )

            tfidf_results = []
            for _slw_i, (concept_id, tfidf_score) in enumerate(raw_results_tfidf):
                # OPT-1c: Check at first iteration then every 10 (PERF-076 tightened)
                if _slw_should_check_timeout(_slw_i):
                    _slw_elapsed_ms = (_time_mod_slw.perf_counter() - _slw_start) * 1000.0
                    if _slw_elapsed_ms > _slw_soft_timeout_s * 1000.0:
                        _slw_timed_out = True
                        _slw_stop = _slw_should_stop_for_timeout(_slw_i, len(tfidf_results))
                        _record_slw_soft_timeout(
                            "tfidf",
                            _slw_i,
                            len(raw_results_tfidf),
                            _slw_elapsed_ms,
                            len(tfidf_results),
                            "stop" if _slw_stop else "materialize",
                        )
                        logger.warning(
                            f"OPT-1c: search_lightweight TF-IDF soft timeout at {_slw_i}/{len(raw_results_tfidf)} "
                            f"({_slw_elapsed_ms:.0f}ms > "
                            f"{_slw_soft_timeout_s * 1000:.0f}ms) — "
                            f"{'stopping' if _slw_stop else 'materializing emergency candidate'} with "
                            f"{len(tfidf_results)} partial results"
                        )
                        if _slw_stop:
                            break
                if tfidf_score < MIN_RETRIEVAL_SIMILARITY * 0.5:  # RETRIEVAL-031: TF-IDF scale differs
                    _probe_inc("tfidf", "score_floor_rejected_count")
                    continue
                concept = _batch_cache_tfidf.get(concept_id)  # PERF-076: dict lookup
                if not concept:
                    continue
                if concept.confidence < min_confidence:
                    _probe_inc("tfidf", "confidence_rejected_count")
                    continue

                score = self._governance_score(concept, tfidf_score)
                if score < 0:
                    if not include_deprecated:
                        _probe_inc("tfidf", "governance_rejected_count")
                        continue  # Hard-filtered (STALE/SUPERSEDED)
                    score = 0.01  # RETRIEVAL-056: include_deprecated — floor score
                # SESSION-012: Cross-session proximity boost (post-scoring, additive)
                if _concurrent_ids and getattr(concept, "session_id", None) in _concurrent_ids:
                    score = min(1.0, score + RETRIEVAL_WEIGHT_SESSION_PROXIMITY)
                tfidf_results.append(
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
                        metadata=_diagnostic_source_metadata(concept),
                    )
                )
            _probe_set("tfidf", "result_count", len(tfidf_results))
            return tfidf_results

        def _cancel_foreground_recovery_probe(config: ForegroundContractConfig, *, phase: str) -> None:
            try:
                get_foreground_contract(_record_metric).cancel_recovery_probe(config)
            except Exception as recovery_cancel_err:
                logger.debug(
                    "FOREGROUND-CONTRACT: %s recovery probe cancel failed: %s",
                    phase,
                    recovery_cancel_err,
                )

        def _run_embedding_encoder(
            query: str,
            result_limit: int,
            *,
            phase: str,
        ) -> tuple[list[tuple[str, float]] | None, str, float]:
            if deadline and deadline.enabled:
                if not _slw_foreground_embedding_enabled:
                    return None, "semantic_disabled", 0.0
                if _slw_foreground_embedding_bounded_enabled:
                    bounded_started = time.perf_counter()
                    remaining_ms = deadline.remaining_ms()
                    available_ms = max(
                        0.0,
                        (remaining_ms if remaining_ms is not None else 0.0)
                        - _slw_min_batch_load_ms,
                    )
                    timeout_ms = min(
                        _slw_foreground_embedding_timeout_ms,
                        available_ms,
                    )
                    if timeout_ms < 50.0:
                        return None, "deadline_tail_reserved", 0.0
                    worker_result = foreground_embedding_process.encode(
                        query,
                        timeout_ms=timeout_ms,
                    )
                    def _bounded_elapsed_ms() -> float:
                        return round(
                            max(
                                worker_result.elapsed_ms,
                                (time.perf_counter() - bounded_started) * 1000.0,
                            ),
                            2,
                        )
                    _record_metric(
                        "search_lightweight.foreground_embedding_process_ms",
                        worker_result.elapsed_ms,
                        {"phase": phase, "outcome": worker_result.outcome},
                    )
                    _record_metric(
                        "search_lightweight.foreground_embedding_process_total",
                        1.0,
                        {"phase": phase, "outcome": worker_result.outcome},
                    )
                    if worker_result.encode_ms is not None:
                        _record_metric(
                            "search_lightweight.foreground_embedding_encode_ms",
                            worker_result.encode_ms,
                            {"phase": phase, "outcome": worker_result.outcome},
                        )
                    if worker_result.outcome != "completed" or worker_result.vector is None:
                        return (
                            None,
                            worker_result.outcome,
                            _bounded_elapsed_ms(),
                        )
                    try:
                        raw = embedding_engine.search_vector(
                            worker_result.vector,
                            top_k=result_limit,
                        )
                    except (TypeError, ValueError):
                        return (
                            None,
                            "invalid_vector",
                            _bounded_elapsed_ms(),
                        )
                    return raw, "completed", _bounded_elapsed_ms()

            started = time.perf_counter()
            raw = embedding_engine.search(query, top_k=result_limit)
            return raw, "synchronous", round((time.perf_counter() - started) * 1000.0, 2)

        def _run_semantic_recovery(
            fallback_reason: str,
            current_results: list[SearchResult],
        ) -> list[SearchResult]:
            nonlocal _slw_semantic_recovery_protected_ids
            recovery_reasons = {
                "latency_over_limit",
                "recent_p95_over_limit",
                "recovery_probe_over_limit",
                "cold_start_no_samples",
            }
            _semantic_recovery_set("trigger_reason", fallback_reason)
            if not _slw_semantic_recovery_enabled:
                _finish_semantic_recovery("feature_disabled")
                return current_results
            if fallback_reason not in recovery_reasons:
                _finish_semantic_recovery("unsupported_trigger")
                return current_results
            if len(current_results or []) >= max(1, int(top_k)):
                _finish_semantic_recovery("sufficient_results")
                return current_results

            readiness = self.semantic_warm_readiness_snapshot()
            if (
                readiness.get("state") != "ready"
                or not readiness.get("available")
                or not readiness.get("initialized")
                or not readiness.get("query_path_warmed")
                or int(readiness.get("index_size") or 0) <= 0
            ):
                _finish_semantic_recovery("semantic_not_ready")
                return current_results
            if deadline and not deadline.can_start(
                "retrieval.semantic_recovery",
                min_remaining_ms=_slw_min_semantic_recovery_ms,
            ):
                deadline.skip(
                    "retrieval.semantic_recovery",
                    "deadline_before_start",
                    priority="optional",
                    min_remaining_ms=_slw_min_semantic_recovery_ms,
                )
                _finish_semantic_recovery("deadline_before_start")
                return current_results

            recovery_config_result = build_foreground_contract_config(
                recorder=_record_metric,
                caller="retrieval.semantic_recovery",
                unit="retrieval.semantic_recovery",
                criticality="quality_sensitive_optional",
                min_remaining_ms=_slw_min_semantic_recovery_ms,
                recent_p95_limit_ms=_slw_semantic_recovery_p95_limit_ms,
                mode=foreground_contract_mode_for_unit("retrieval.semantic_recovery"),
                circuit_ttl_s=_slw_semantic_recovery_circuit_ttl_s,
                skip_when_cold=False,
            )
            recovery_config = recovery_config_result.config
            if not recovery_config_result.valid:
                recovery_decision = ForegroundContractDecision(
                    unit=recovery_config_result.unit,
                    decision=ForegroundDecision.SKIP,
                    reason="invalid_config",
                    remaining_ms=deadline.remaining_ms() if deadline else None,
                    mode=recovery_config_result.mode,
                )
            else:
                recovery_decision = None
                try:
                    recovery_decision = get_foreground_contract(_record_metric).decide(
                        recovery_config,
                        deadline=deadline,
                        answer_path="unknown",
                    )
                except Exception as recovery_decision_err:
                    logger.debug(
                        "FOREGROUND-CONTRACT: semantic recovery decision failed: %s",
                        recovery_decision_err,
                    )
            if recovery_decision is not None and recovery_decision.decision is ForegroundDecision.SKIP:
                if deadline:
                    deadline.skip(
                        "retrieval.semantic_recovery",
                        recovery_decision.reason,
                        priority="optional",
                        min_remaining_ms=_slw_min_semantic_recovery_ms,
                    )
                _finish_semantic_recovery(recovery_decision.reason)
                return current_results

            _semantic_recovery_set("attempted", True)
            _semantic_recovery_set("search_started", True)
            try:
                raw_recovery_results, recovery_outcome, recovery_latency_ms = _run_embedding_encoder(
                    effective_query_text,
                    _slw_semantic_recovery_top_k,
                    phase="semantic_recovery",
                )
                _semantic_recovery_set("latency_ms", recovery_latency_ms)
                _semantic_recovery_set("worker_outcome", recovery_outcome)
                _semantic_recovery_set("raw_count", len(raw_recovery_results or []))
                _record_metric(
                    "search_lightweight.semantic_recovery_latency_ms",
                    recovery_latency_ms,
                    {"trigger_reason": fallback_reason, "outcome": recovery_outcome},
                )
            except Exception:
                _cancel_foreground_recovery_probe(
                    recovery_config,
                    phase="retrieval.semantic_recovery",
                )
                raise
            try:
                get_foreground_contract(_record_metric).record_latency_ms(
                    recovery_config,
                    recovery_latency_ms,
                    answer_path="unknown",
                )
            except Exception as recovery_latency_err:
                _cancel_foreground_recovery_probe(
                    recovery_config,
                    phase="retrieval.semantic_recovery",
                )
                logger.debug("FOREGROUND-CONTRACT: semantic recovery latency record failed: %s", recovery_latency_err)
            if raw_recovery_results is None:
                _finish_semantic_recovery(
                    f"embedding_{recovery_outcome}",
                    attempted=True,
                )
                return current_results

            candidate_ids = [cid for cid, score in raw_recovery_results if score >= _slw_semantic_recovery_min_score]
            _semantic_recovery_set("candidate_id_count", len(candidate_ids))
            if not candidate_ids:
                for _cid, _score in raw_recovery_results:
                    if _score < _slw_semantic_recovery_min_score:
                        _semantic_recovery_inc("score_floor_rejected_count")
                _finish_semantic_recovery("no_candidates_above_floor", attempted=True)
                return current_results

            from app.storage.concepts import load_concepts_batch

            concept_cache = load_concepts_batch(candidate_ids)
            _semantic_recovery_set("loaded_count", len(concept_cache or {}))
            existing_ids = {result.concept_id for result in current_results or []}
            admitted_ids: list[str] = []
            recovered_results = list(current_results or [])
            reason = "no_admission"
            for concept_id, semantic_score in raw_recovery_results:
                if semantic_score < _slw_semantic_recovery_min_score:
                    _semantic_recovery_inc("score_floor_rejected_count")
                    reason = "score_below_floor"
                    continue
                if concept_id in existing_ids:
                    _semantic_recovery_inc("duplicate_rejected_count")
                    reason = "already_present"
                    continue
                concept = concept_cache.get(concept_id)
                if concept is None:
                    reason = "concept_missing"
                    continue
                if concept.confidence < min_confidence:
                    _semantic_recovery_inc("confidence_rejected_count")
                    reason = "confidence_below_floor"
                    continue
                score = self._governance_score(concept, semantic_score)
                if score < 0:
                    if not include_deprecated:
                        _semantic_recovery_inc("governance_rejected_count")
                        reason = "governance_filtered"
                        continue
                    score = 0.01
                recovered_results.append(
                    SearchResult(
                        concept_id=concept.id,
                        version=concept.version,
                        summary=concept.summary,
                        confidence=concept.confidence,
                        relevance_score=score,
                        knowledge_area=concept.metadata.get("knowledge_area"),
                        ka_relative_authority=getattr(concept, "ka_relative_authority", None),
                        maturity=getattr(concept, "maturity", None),
                        created_at=concept.created_at,
                        metadata=_diagnostic_source_metadata(concept),
                    )
                )
                existing_ids.add(concept_id)
                _slw_semantic_recovery_protected_ids.add(concept_id)
                admitted_ids.append(concept_id)
                reason = "admitted"
                if len(admitted_ids) >= _slw_semantic_recovery_max_admitted:
                    break
            if admitted_ids:
                reason = "admitted"
            _semantic_recovery_set("admitted_ids", admitted_ids)
            _finish_semantic_recovery(reason, len(admitted_ids), attempted=True)
            return recovered_results

        if self._embeddings_available and embedding_engine.index_size > 0:
            if (
                _slw_embedding_search_admission_enabled
                and deadline
                and not deadline.can_start(
                    "retrieval.embedding_search",
                    min_remaining_ms=_slw_min_embedding_search_ms,
                )
            ):
                deadline.skip(
                    "retrieval.embedding_search",
                    "deadline_before_start",
                    priority="optional",
                    min_remaining_ms=_slw_min_embedding_search_ms,
                )
                _record_metric(
                    "search_lightweight.embedding_search_ms",
                    0.0,
                    {"path": "embedding", "reason": "deadline_before_start", "admission": "skipped"},
                )
                _probe_set("embedding", "search_skipped_reason", "deadline_before_start")
                _record_metric(
                    "search_lightweight.fallback_total",
                    1.0,
                    {"from": "embedding", "to": "tfidf", "reason": "deadline_before_start", "mode": "deadline"},
                )
                results = _run_tfidf_path("deadline_before_start", "deadline")
            else:
                # Embedding path
                _slw_embedding_search_ready_for_probe = (
                    bool(self._embeddings_available)
                    and bool(getattr(self, "_embeddings_initialized", False))
                    and int(getattr(embedding_engine, "index_size", 0) or 0) > 0
                )
                _slw_embedding_search_skip_when_cold = (
                    _slw_embedding_search_cold_skip_enabled and not _slw_embedding_search_ready_for_probe
                )
                _slw_foreground_config_result = build_foreground_contract_config(
                    recorder=_record_metric,
                    caller="retrieval.embedding_search",
                    unit="retrieval.embedding_search",
                    criticality="quality_sensitive_optional",
                    min_remaining_ms=_slw_min_embedding_search_ms,
                    recent_p95_limit_ms=_slw_embedding_search_p95_limit_ms,
                    mode=foreground_contract_mode_for_unit("retrieval.embedding_search"),
                    circuit_ttl_s=_slw_embedding_search_circuit_ttl_s,
                    skip_when_cold=_slw_embedding_search_skip_when_cold,
                )
                _slw_foreground_config = _slw_foreground_config_result.config
                if not _slw_foreground_config_result.valid:
                    _slw_fg_decision = ForegroundContractDecision(
                        unit=_slw_foreground_config_result.unit,
                        decision=ForegroundDecision.SKIP,
                        reason="invalid_config",
                        remaining_ms=deadline.remaining_ms() if deadline else None,
                        mode=_slw_foreground_config_result.mode,
                    )
                else:
                    _slw_fg_decision = None
                    try:
                        _slw_fg_decision = get_foreground_contract(_record_metric).decide(
                            _slw_foreground_config,
                            deadline=deadline,
                            answer_path="unknown",
                        )
                    except Exception as _slw_fg_err:
                        logger.debug(
                            "FOREGROUND-CONTRACT: retrieval shadow decision failed: %s",
                            _slw_fg_err,
                        )
                if _slw_fg_decision is not None and _slw_fg_decision.decision is ForegroundDecision.SKIP:
                    if deadline:
                        deadline.skip(
                            "retrieval.embedding_search",
                            _slw_fg_decision.reason,
                            priority="optional",
                            min_remaining_ms=_slw_min_embedding_search_ms,
                        )
                    _record_metric(
                        "search_lightweight.embedding_search_ms",
                        0.0,
                        {
                            "path": "embedding",
                            "reason": _slw_fg_decision.reason,
                            "admission": "skipped",
                        },
                    )
                    _probe_set("embedding", "search_skipped_reason", _slw_fg_decision.reason)
                    _record_metric(
                        "search_lightweight.fallback_total",
                        1.0,
                        {
                            "from": "embedding",
                            "to": "tfidf",
                            "reason": _slw_fg_decision.reason,
                            "mode": _slw_fg_decision.mode.value,
                        },
                    )
                    results = _run_tfidf_path(_slw_fg_decision.reason, _slw_fg_decision.mode.value)
                    results = _run_semantic_recovery(_slw_fg_decision.reason, results)
                else:
                    query_text = effective_query_text
                    _probe_set("embedding", "search_started", True)
                    try:
                        raw_results, embedding_outcome, _embedding_search_elapsed_ms = (
                            _run_embedding_encoder(
                                query_text,
                                top_k,
                                phase="embedding_search",
                            )
                        )
                        _probe_set("embedding", "raw_count", len(raw_results or []))
                        _record_metric(
                            "search_lightweight.embedding_search_ms",
                            _embedding_search_elapsed_ms,
                            {
                                "path": "embedding",
                                "admission": "started",
                                "outcome": embedding_outcome,
                            },
                        )
                    except Exception:
                        _cancel_foreground_recovery_probe(
                            _slw_foreground_config,
                            phase="retrieval.embedding_search",
                        )
                        raise
                    try:
                        get_foreground_contract(_record_metric).record_latency_ms(
                            _slw_foreground_config,
                            _embedding_search_elapsed_ms,
                            answer_path="unknown",
                        )
                    except Exception as _slw_fg_err:
                        _cancel_foreground_recovery_probe(
                            _slw_foreground_config,
                            phase="retrieval.embedding_search",
                        )
                        logger.debug("FOREGROUND-CONTRACT: retrieval latency record failed: %s", _slw_fg_err)

                    if raw_results is None:
                        fallback_reason = f"embedding_{embedding_outcome}"
                        _probe_set("embedding", "search_skipped_reason", fallback_reason)
                        if deadline:
                            deadline.skip(
                                "retrieval.embedding_search",
                                fallback_reason,
                                priority="optional",
                            )
                        _record_metric(
                            "search_lightweight.fallback_total",
                            1.0,
                            {
                                "from": "embedding",
                                "to": "tfidf",
                                "reason": fallback_reason,
                                "mode": "bounded_process",
                            },
                        )
                        results = _run_tfidf_path(fallback_reason, "bounded_process")
                    else:
                        # PERF-076: Batch load all candidate concepts in one query
                        _candidate_ids = [
                            cid for cid, score in raw_results if score >= MIN_RETRIEVAL_SIMILARITY
                        ]
                        _probe_set("embedding", "candidate_id_count", len(_candidate_ids))
                        if deadline and not deadline.can_start(
                            "retrieval.load_concepts_batch",
                            min_remaining_ms=_slw_min_batch_load_ms,
                        ):
                            deadline.skip(
                                "retrieval.load_concepts_batch",
                                "deadline_before_start",
                                priority="optional",
                                min_remaining_ms=_slw_min_batch_load_ms,
                            )
                            _probe_set("embedding", "search_skipped_reason", "batch_deadline_before_start")
                            return _finish_live_probe([])
                        from app.storage.concepts import load_concepts_batch

                        _batch_load_start = time.perf_counter()
                        _batch_cache = load_concepts_batch(_candidate_ids)
                        _probe_set("embedding", "loaded_count", len(_batch_cache or {}))
                        _record_metric(
                            "search_lightweight.batch_load_ms",
                            round((time.perf_counter() - _batch_load_start) * 1000.0, 2),
                            {"path": "embedding"},
                        )

                        results = []
                        for _slw_i, (concept_id, emb_score) in enumerate(raw_results):
                            # OPT-1c: Check at first iteration then every 10 (PERF-076 tightened)
                            if _slw_should_check_timeout(_slw_i):
                                _slw_elapsed_ms = (_time_mod_slw.perf_counter() - _slw_start) * 1000.0
                                if _slw_elapsed_ms > _slw_soft_timeout_s * 1000.0:
                                    _slw_timed_out = True
                                    _slw_stop = _slw_should_stop_for_timeout(_slw_i, len(results))
                                    _record_slw_soft_timeout(
                                        "embedding",
                                        _slw_i,
                                        len(raw_results),
                                        _slw_elapsed_ms,
                                        len(results),
                                        "stop" if _slw_stop else "materialize",
                                    )
                                    logger.warning(
                                        f"OPT-1c: search_lightweight soft timeout at iteration {_slw_i}/{len(raw_results)} "
                                        f"({_slw_elapsed_ms:.0f}ms > "
                                        f"{_slw_soft_timeout_s * 1000:.0f}ms) — "
                                        f"{'stopping' if _slw_stop else 'materializing emergency candidate'} with "
                                        f"{len(results)} partial results"
                                    )
                                    if _slw_stop:
                                        break
                            if emb_score < MIN_RETRIEVAL_SIMILARITY:
                                _probe_inc("embedding", "score_floor_rejected_count")
                                continue
                            concept = _batch_cache.get(concept_id)
                            if not concept:
                                continue
                            if concept.confidence < min_confidence:
                                _probe_inc("embedding", "confidence_rejected_count")
                                continue

                            score = self._governance_score(concept, emb_score)
                            if score < 0:
                                if not include_deprecated:
                                    _probe_inc("embedding", "governance_rejected_count")
                                    continue
                                score = 0.01
                            if _concurrent_ids and getattr(concept, "session_id", None) in _concurrent_ids:
                                score = min(1.0, score + RETRIEVAL_WEIGHT_SESSION_PROXIMITY)
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
                                    created_at=concept.created_at,
                                    metadata=_diagnostic_source_metadata(concept),
                                )
                            )
                        _probe_set("embedding", "result_count", len(results))
                        lexical_admission = self._admit_strong_lexical_evidence(
                            results,
                            effective_query_text,
                            path="search_lightweight",
                            deadline=deadline,
                        )
                        results = lexical_admission.results
                        _slw_lexical_protected_ids = lexical_admission.protected_ids
                        self.last_lexical_evidence_support_trace = lexical_admission.support_trace
        else:
            # TF-IDF fallback path
            results = _run_tfidf_path()

        if self.last_lexical_evidence_support_trace is None:
            lexical_admission = self._admit_strong_lexical_evidence(
                results,
                effective_query_text,
                path="search_lightweight_tfidf",
                deadline=deadline,
            )
            results = lexical_admission.results
            _slw_lexical_protected_ids = lexical_admission.protected_ids
            self.last_lexical_evidence_support_trace = lexical_admission.support_trace

        if _slw_trace is not None:
            _mh262_trace_score_stage(
                _slw_trace,
                "phase1_results",
                after_scores=[(r.concept_id, r.relevance_score) for r in results],
            )

        # ===== Phase 1.5: KA-aware boost (KA-ARCH-001 Fix 7) =====
        # OPT-1c: Skip enhancement phases if Phase 1 timed out — return core results fast.
        if _slw_timed_out and results:
            # Still apply KA exclusion (critical for correctness) but skip expensive boosts.
            from app.core.config import RETRIEVAL_KA_EXCLUDE as _exc_ka

            if _exc_ka:
                if not BENCHMARK.enabled:
                    results = [r for r in results if r.knowledge_area not in _exc_ka]
            _apply_authority_artifact_boost(
                results,
                effective_query_text,
                path="search_lightweight_timeout",
            )
            results.sort(key=lambda r: (-r.relevance_score, r.concept_id))
            results = self._trim_with_protected_ids(
                results,
                top_k,
                _slw_lexical_protected_ids | _slw_semantic_recovery_protected_ids,
            )
            return _finish_live_probe(results)

        if results:
            inferred_kas_set: set[str] = set()
            if get_feature_flag("KA_AUTO_BOOST_ENABLED", False):
                inferred_kas_set.update(_infer_query_kas(effective_query_text))
            if (
                get_feature_flag("QUERY_INTENT_EXPANSION_ENABLED", True)
                and intent_expansion
                and intent_expansion.inferred_kas
            ):
                inferred_kas_set.update(intent_expansion.inferred_kas)
            inferred_kas = tuple(inferred_kas_set)
            if inferred_kas:
                for result in results:
                    concept_ka = result.knowledge_area
                    if concept_ka and concept_ka in inferred_kas:
                        result.relevance_score = min(1.0, result.relevance_score + KA_BOOST_WEIGHT)

        # ===== Phase 1.6: KA exclusion filter (RETRIEVAL-061) =====
        # Exclude benchmark/test KAs from interactive retrieval.
        # Bypassed in benchmark mode. Config: PITH_RETRIEVAL_KA_EXCLUDE env var.
        from app.core.config import RETRIEVAL_KA_EXCLUDE

        if RETRIEVAL_KA_EXCLUDE and results:
            if not BENCHMARK.enabled:
                _pre_exclude = len(results)
                results = [r for r in results if r.knowledge_area not in RETRIEVAL_KA_EXCLUDE]
                _excluded = _pre_exclude - len(results)
                if _excluded > 0:
                    logger.debug(f"RETRIEVAL-061: Excluded {_excluded} concepts from KAs {RETRIEVAL_KA_EXCLUDE}")

        # ===== Phase 2: Enhanced retrieval boosts (both paths) =====
        if ENHANCED_RETRIEVAL and results:
            scored = [(r.concept_id, r.relevance_score) for r in results]
            if _slw_trace is not None:
                _slw_trace["predictive_activation"]["before_boost"] = _mh262_predictive_activation_snapshot(
                    predictive_activation
                )
            scored = predictive_activation.boost_retrieval_scores(scored, boost_weight=0.15)
            if _slw_trace is not None:
                _slw_trace["predictive_activation"]["after_boost"] = _mh262_predictive_activation_snapshot(
                    predictive_activation
                )
                _mh262_trace_score_stage(
                    _slw_trace,
                    "predictive_boost",
                    before_scores=[(r.concept_id, r.relevance_score) for r in results],
                    after_scores=scored,
                )
            score_dict = dict(scored)
            for result in results:
                if result.concept_id in score_dict:
                    result.relevance_score = score_dict[result.concept_id]

        _apply_authority_artifact_boost(
            results,
            effective_query_text,
            path="search_lightweight",
        )

        # RETRIEVAL-037b v4.2: Deterministic tiebreaker — when governance scores
        # tie (common: many MAB facts share identical authority/currency/confidence),
        # sort by concept_id to make budget cutoff deterministic across server restarts.
        results.sort(key=lambda r: (-r.relevance_score, r.concept_id))

        # ===== AGENT-002: Scoped filtering (PERF-003: batch lookup) =====
        if agent_id and scope == "agent":
            aid_map = self._batch_concept_agent_ids([r.concept_id for r in results])
            results = [r for r in results if aid_map.get(r.concept_id, "default") in (agent_id, "default")]
        results = self._trim_with_protected_ids(
            results,
            top_k,
            _slw_lexical_protected_ids | _slw_semantic_recovery_protected_ids,
        )

        if _slw_trace is not None:
            _mh262_trace_score_stage(
                _slw_trace,
                "final_results",
                after_scores=[(r.concept_id, r.relevance_score) for r in results],
            )

        return _finish_live_probe(results)

    def sync_index(self) -> int:
        """Ensure all active concepts are in the TF-IDF index.

        Compares active concept IDs from storage against the index.
        Adds any missing concepts incrementally, then recalculates IDF once.

        Returns:
            Number of concepts added to the index.
        """
        with self._writer_admission("sync_index") as admit:
            if not admit:
                return 0
            # Compare IDs first so the common no-op path does not hydrate every model.
            storage_ids = set(list_concepts())

            # Get indexed concept IDs (exclude logically deleted rows)
            idx = self.index
            indexed_ids = set()
            for i, cid in enumerate(idx.concept_ids):
                if i not in idx.deleted_indices:
                    indexed_ids.add(cid)

            # Find unindexed concepts
            missing_ids = storage_ids - indexed_ids
            if not missing_ids:
                logger.info("sync_index: all concepts already indexed")
                return 0

            logger.info(f"sync_index: {len(missing_ids)} concepts not in index, adding...")
            concept_map = load_concepts_batch(list(missing_ids))

            # Suppress intermediate IDF recalcs during bulk add
            original_threshold = self.index.idf_update_threshold
            self.index.idf_update_threshold = len(missing_ids) + 100

            added = 0
            try:
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

    def pairwise_similarity(
        self,
        threshold: float = 0.12,
        max_pairs_evaluated: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Compute all above-threshold concept pairs by cosine similarity.

        Uses the TF-IDF matrix for efficient pairwise computation.
        Excludes self-edges, deleted index entries, and normalizes direction
        (sorted pair IDs) to match edge direction normalization in storage.

        Args:
            threshold: Minimum cosine similarity to include (default 0.12).
            max_pairs_evaluated: Optional cap on pair comparisons for bounded
                maintenance runs.

        Returns:
            List of (concept_a, concept_b, cosine_score) tuples where
            concept_a < concept_b (normalized direction).
        """
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        # Bind a local handle once: a concurrent rebuild+swap rebinds
        # self.index atomically, so all multi-statement reads must use the
        # same snapshot to avoid mismatched row indices.
        idx = self.index
        if idx.tfidf_matrix is None or idx.document_count == 0:
            logger.debug("pairwise_similarity: empty index")
            self._last_pairwise_similarity_stats = {
                "pairs_evaluated": 0,
                "pairs_available": 0,
                "pair_budget_exhausted": False,
            }
            return []

        matrix = idx.tfidf_matrix
        n_docs = matrix.shape[0]

        # Build set of valid (non-deleted) row indices
        valid_indices = [i for i in range(n_docs) if i not in idx.deleted_indices]
        if len(valid_indices) < 2:
            self._last_pairwise_similarity_stats = {
                "pairs_evaluated": 0,
                "pairs_available": 0,
                "pair_budget_exhausted": False,
            }
            return []

        # Slice matrix to valid rows only for efficient computation
        valid_matrix = matrix[valid_indices]

        # Extract above-threshold pairs (upper triangle only to avoid duplicates).
        # Compute row windows instead of a full dense NxN matrix so maintenance
        # callers can put a real ceiling on comparisons and memory pressure.
        pairs = []
        pairs_evaluated = 0
        pairs_available = len(valid_indices) * (len(valid_indices) - 1) // 2
        pair_budget_exhausted = False
        for i_idx in range(len(valid_indices)):
            remaining = len(valid_indices) - i_idx - 1
            if remaining <= 0:
                continue
            comparisons_this_row = remaining
            if max_pairs_evaluated is not None:
                budget_remaining = max_pairs_evaluated - pairs_evaluated
                if budget_remaining <= 0:
                    pair_budget_exhausted = True
                    break
                comparisons_this_row = min(remaining, budget_remaining)
                pair_budget_exhausted = comparisons_this_row < remaining

            row_scores = sk_cosine(
                valid_matrix[i_idx],
                valid_matrix[i_idx + 1 : i_idx + 1 + comparisons_this_row],
            ).ravel()
            for offset, score_value in enumerate(row_scores, start=1):
                j_idx = i_idx + offset
                pairs_evaluated += 1
                score = float(score_value)
                if score >= threshold:
                    # Map back to concept IDs
                    cid_a = idx.concept_ids[valid_indices[i_idx]]
                    cid_b = idx.concept_ids[valid_indices[j_idx]]
                    # Normalize direction (sorted) to match edge storage normalization
                    source, target = sorted([cid_a, cid_b])
                    pairs.append((source, target, round(score, 4)))
            if pair_budget_exhausted:
                break

        # Sort by score descending for priority processing
        pairs.sort(key=lambda x: x[2], reverse=True)
        self._last_pairwise_similarity_stats = {
            "pairs_evaluated": pairs_evaluated,
            "pairs_available": pairs_available,
            "pair_budget_exhausted": pair_budget_exhausted,
        }

        logger.info(
            f"pairwise_similarity: {len(pairs)} pairs above threshold {threshold} "
            f"from {len(valid_indices)} indexed concepts "
            f"(evaluated={pairs_evaluated}/{pairs_available}, budget_exhausted={pair_budget_exhausted})"
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

        # Bind a local handle once (reader-safety against a concurrent swap).
        idx = self.index
        # Indexed concept IDs (excluding logically deleted rows)
        indexed_ids = set()
        for i, cid in enumerate(idx.concept_ids):
            if i not in idx.deleted_indices:
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
        with self._writer_admission("repair_index_drift") as admit:
            if not admit:
                # Resolve real integrity first so callers (server.py:4022 echoed
                # to client; server.py:1166 logged) get an accurate snapshot.
                post = integrity if integrity is not None else self.verify_index_integrity()
                return {
                    "status": "skipped_quiesce",
                    "ghosts_removed": 0,
                    "orphans_added": 0,
                    "failed_ghosts": [],
                    "failed_orphans": [],
                    "dry_run": dry_run,
                    "post_repair_integrity": post,
                }
            return self._repair_index_drift_inner(dry_run=dry_run, integrity=integrity)

    def _repair_index_drift_inner(self, dry_run: bool = False, integrity: dict = None) -> dict:
        if integrity is None:
            integrity = self.verify_index_integrity()

        if integrity["is_healthy"]:
            return {
                "status": "healthy",
                "ghosts_removed": 0,
                "orphans_added": 0,
                "failed_ghosts": [],
                "failed_orphans": [],
                "dry_run": dry_run,
                "post_repair_integrity": integrity,
            }

        ghosts_removed = 0
        orphans_added = 0
        failed_ghosts = []
        failed_orphans = []

        # Remove ghost entries
        for ghost_id in integrity["ghost_ids"]:
            if dry_run:
                logger.info(f"[DRY RUN] Would remove ghost: {ghost_id}")
            else:
                try:
                    self.remove_concept(ghost_id)
                    ghosts_removed += 1
                    logger.info(f"Removed ghost from index: {ghost_id}")
                except Exception as e:
                    failed_ghosts.append({"id": ghost_id, "error": str(e)})
                    logger.warning(f"Index repair failed to remove ghost {ghost_id}: {e}")

        # Add orphan concepts
        if integrity["orphan_ids"]:
            if dry_run:
                for orphan_id in integrity["orphan_ids"]:
                    logger.info(f"[DRY RUN] Would add orphan: {orphan_id}")
            else:
                concepts = list_concepts_full()
                concept_map = {c.id: c for c in concepts}

                for orphan_id in integrity["orphan_ids"]:
                    concept = concept_map.get(orphan_id)
                    if not concept:
                        failed_orphans.append({"id": orphan_id, "error": "concept_not_found"})
                        continue
                    try:
                        searchable_text = self._concept_to_document(concept)
                        added = self.index.add_concept(orphan_id, searchable_text)
                        if not added:
                            raise RuntimeError("index.add_concept returned False")
                        orphans_added += 1
                        logger.info(f"Added orphan to index: {orphan_id}")
                    except Exception as e:
                        failed_orphans.append({"id": orphan_id, "error": str(e)})
                        logger.warning(f"Index repair failed to add orphan {orphan_id}: {e}")

        if not dry_run and (ghosts_removed > 0 or orphans_added > 0):
            if orphans_added > 0:
                self.index.force_idf_recalculation()
            self._auto_save()

        post_repair_integrity = integrity if dry_run else self.verify_index_integrity()
        if not dry_run:
            failed_ghost_ids = {entry["id"] for entry in failed_ghosts}
            remaining_ghosts = set(post_repair_integrity.get("ghost_ids", []))
            for ghost_id in integrity["ghost_ids"]:
                if ghost_id in remaining_ghosts and ghost_id not in failed_ghost_ids:
                    failed_ghosts.append({"id": ghost_id, "error": "still_present_after_repair"})
                    failed_ghost_ids.add(ghost_id)

            failed_orphan_ids = {entry["id"] for entry in failed_orphans}
            remaining_orphans = set(post_repair_integrity.get("orphan_ids", []))
            for orphan_id in integrity["orphan_ids"]:
                if orphan_id in remaining_orphans and orphan_id not in failed_orphan_ids:
                    failed_orphans.append({"id": orphan_id, "error": "still_missing_after_repair"})
                    failed_orphan_ids.add(orphan_id)

        if dry_run:
            status = "dry_run"
        elif post_repair_integrity["is_healthy"]:
            status = "repaired"
        elif failed_ghosts or failed_orphans:
            status = "partial"
        else:
            status = "incomplete"

        result = {
            "status": status,
            "ghosts_removed": ghosts_removed,
            "orphans_added": orphans_added,
            "failed_ghosts": failed_ghosts,
            "failed_orphans": failed_orphans,
            "dry_run": dry_run,
            "post_repair_integrity": post_repair_integrity,
        }
        logger.info(f"Index repair: {result}")
        return result

    @staticmethod
    def _recount_df_delta(index) -> int:
        """Recompute document frequencies from per-doc term counts (non-deleted
        rows only) and return the max absolute deviation from the index's stored
        document_frequencies. 0 == perfectly consistent."""
        recount: dict[str, int] = {}
        for i, counts in enumerate(index.document_term_counts):
            if i in index.deleted_indices:
                continue
            for term in counts or {}:
                recount[term] = recount.get(term, 0) + 1
        max_delta = 0
        for term, term_id in index.vocabulary.items():
            stored = int(index.document_frequencies[term_id]) if term_id < len(index.document_frequencies) else 0
            delta = abs(stored - recount.get(term, 0))
            if delta > max_delta:
                max_delta = delta
        return max_delta

    def _count_gold_lexical_stale(self, index) -> int:
        """Count gold-pair concepts classified lexical_stale against ``index``.

        Reuses the read-only audit's build_audit() with index injected so the
        verification uses the exact same staleness classifier as the live audit.
        Returns -1 if the audit harness is unavailable (verification then relies
        on the structural invariants only)."""
        try:
            from scripts.retrieval_stale_index_audit import CLASS_LEXICAL_STALE, build_audit
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning(f"rebuild_and_swap_repair: stale audit unavailable: {exc}")
            return -1
        try:
            from app.storage.utils import DB_PATH

            db_path = Path(DB_PATH)
        except Exception:
            db_path = Path(resolve_data_dir()) / "pith.db"
        gold_path = Path(__file__).resolve().parents[2] / "tests" / "eval" / "gold_pairs.json"
        if not gold_path.exists() or not db_path.exists():
            return -1
        report = build_audit(
            db_path=db_path,
            gold_path=gold_path,
            high_authority_limit=0,
            index=index,
        )
        return sum(
            1
            for c in report.get("candidates", [])
            if c.get("classification") == CLASS_LEXICAL_STALE and "gold" in (c.get("sources") or [])
        )

    def _build_verified_fresh_index(self) -> tuple[IncrementalTfidfIndex, dict[str, Any]]:
        """Build and verify a fresh TF-IDF candidate without mutating live state."""
        fresh = IncrementalTfidfIndex()
        self._build_into(fresh)

        stale_after = self._count_gold_lexical_stale(fresh)
        if stale_after > 0:
            raise RuntimeError(
                f"verify failed: {stale_after} gold concepts still lexical_stale after rebuild"
            )

        df_delta = self._recount_df_delta(fresh)
        if df_delta != 0:
            raise RuntimeError(f"verify failed: DF recount delta {df_delta} != 0")

        count = fresh.document_count
        dtc = len(fresh.document_term_counts)
        cids = len(fresh.concept_ids)
        if not (count == dtc == cids):
            raise RuntimeError(
                f"verify failed: consistency invariant count={count} dtc={dtc} cids={cids}"
            )

        return fresh, {
            "stale_after": stale_after,
            "df_recount_delta": df_delta,
        }

    def rebuild_and_swap_repair(self, *, dry_run: bool = False) -> dict:
        """Rebuild the TF-IDF index into a fresh instance and atomically swap.

        Two-layer quiesce:
          Layer 2 (cross-process): hold the reflection.lock flock (NB) for the
            whole window; if a reflection writer holds it, defer cleanly.
          Layer 1 (in-process): quiesce_writers() freezes index_version while a
            fresh index is built and verified before the atomic rebind.

        MUST run in-process (via the runtime singleton). Restores the on-disk
        backup on any exception after the backup is taken. Returns a report.
        """
        report: dict[str, Any] = {
            "stale_before": None,
            "stale_after": None,
            "df_recount_delta": None,
            "v0": None,
            "duration_s": None,
            "swapped": False,
            "deferred": None,
            "dry_run": dry_run,
        }
        t0 = time.perf_counter()

        # --- Layer 2: cross-process reflection.lock flock (NB) ---
        lock_dir = Path(resolve_data_dir()) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "reflection.lock"
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                lock_fd_exclusive(fd)
            except BlockingIOError:
                report["deferred"] = "reflection_active"
                return report

            backup_path = None
            backup_made = False
            try:
                # --- Backup live index dir (skip in dry_run) ---
                v_for_backup = self.index.index_version
                backup_path = f"{self.index_path}.repair-backup-{v_for_backup}"
                if not dry_run and Path(self.index_path).exists():
                    if Path(backup_path).exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(self.index_path, backup_path)
                    backup_made = True

                # --- Layer 1: quiesce + build fresh + verify + swap ---
                with self.quiesce_writers():
                    v0 = self.index.index_version
                    report["v0"] = v0

                    stale_before = self._count_gold_lexical_stale(self.index)
                    report["stale_before"] = stale_before

                    fresh, gates = self._build_verified_fresh_index()
                    report.update(gates)

                    # No-writer assertion: a bypassing writer would have bumped this.
                    assert self.index.index_version == v0, (
                        f"no-writer assertion failed: index_version {self.index.index_version} != v0 {v0}"
                    )

                    if dry_run:
                        report["swapped"] = False
                    else:
                        # Atomic rebind, then persist the fresh index on disk.
                        self.index = fresh
                        fresh.save(self.index_path)
                        report["swapped"] = True

                # Success — drop the (now stale) backup for a real swap.
                if backup_made and not dry_run:
                    try:
                        shutil.rmtree(backup_path)
                    except Exception:
                        pass
                return report

            except BaseException:
                # Restore the on-disk backup (atomic-ish) if we got that far.
                if backup_made and backup_path and Path(backup_path).exists():
                    try:
                        if Path(self.index_path).exists():
                            shutil.rmtree(self.index_path)
                        shutil.move(backup_path, self.index_path)
                        logger.warning("rebuild_and_swap_repair: restored on-disk backup after failure")
                    except Exception as restore_exc:
                        logger.error(f"rebuild_and_swap_repair: backup restore FAILED: {restore_exc}")
                raise
        finally:
            try:
                unlock_fd(fd)
            finally:
                os.close(fd)
                report["duration_s"] = round(time.perf_counter() - t0, 3)

    def _refresh_concept_embedding(self, concept_id: str, searchable_text: str) -> bool:
        """Re-embed an existing concept's searchable text and persist it.

        Mirrors the embedding block in ``_add_concept_inner`` (norm guard,
        ``update_embedding`` overwrite, SQLite persist) but for the refresh path:
        the concept already has a (now stale) embedding, and
        ``embedding_engine.update_embedding`` overwrites in place. Best-effort —
        an embedding failure is logged and does NOT roll back the lexical refresh,
        exactly as in the add path. Returns True if the embedding was persisted.
        """
        if not self._embeddings_initialized:
            return False
        try:
            emb = embedding_engine.embed_text(searchable_text)
            emb_norm = float(np.linalg.norm(emb))
            if emb_norm < 0.5:
                # STABILITY-008: never persist an all-zero/corrupt embedding.
                logger.warning(f"refresh_concepts: rejecting embedding for {concept_id} (norm={emb_norm:.4f} < 0.5)")
                return False
            embedding_engine.update_embedding(concept_id, emb)
            if not self._embeddings_available:
                self._embeddings_available = True
            from app.storage import _db_immediate

            with _db_immediate() as conn:
                conn.execute(
                    """UPDATE concepts
                       SET embedding = ?, embedding_version = ?,
                           embedding_text_hash = ?,
                           embedding_text_contract_version = ?,
                           embedding_refreshed_at = ?
                       WHERE id = ?""",
                    _embedding_update_tuple(emb, searchable_text, concept_id),
                )
            return True
        except Exception as e:
            logger.warning(f"refresh_concepts: embedding refresh failed for {concept_id}: {e}")
            return False

    def _refresh_concepts_fresh(
        self,
        concept_ids,
        *,
        persist: bool,
        refresh_embeddings: bool,
        host_memory_pressure_state: str,
        host_pressure_reasons: list[str],
    ) -> dict:
        """Refresh through a verified fresh index while the old index serves readers."""
        report: dict[str, Any] = {
            "refreshed": [],
            "skipped": [],
            "stale_before": None,
            "stale_after": None,
            "df_recount_delta": None,
            "v0": None,
            "swapped": False,
            "embeddings_refreshed": 0,
            "deferred": None,
            "fresh_candidate": True,
            "host_memory_pressure_state": host_memory_pressure_state,
            "host_pressure_reasons": host_pressure_reasons,
            "post_swap_integrity": None,
            "post_repair_integrity": None,
            "requeue_ids": [],
            "convergence_error": None,
        }
        t0 = time.perf_counter()
        ids = list(dict.fromkeys(concept_ids))
        if not ids:
            report["duration_s"] = 0.0
            return report

        texts: dict[str, str] = {}
        for cid in ids:
            row = self._load_concept_row_for_index(cid)
            if row is None:
                report["skipped"].append({"id": cid, "reason": "not_current"})
                continue
            text = build_searchable_text(row)
            if not text.strip():
                report["skipped"].append({"id": cid, "reason": "empty_text"})
                continue
            texts[cid] = text

        if host_memory_pressure_state in {"high", "critical"}:
            report["deferred"] = f"host_memory_pressure_{host_memory_pressure_state}"
            report["duration_s"] = round(time.perf_counter() - t0, 3)
            return report

        lock_dir = Path(resolve_data_dir()) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "reflection.lock"
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                lock_fd_exclusive(fd)
            except BlockingIOError:
                report["deferred"] = "reflection_active"
                return report

            backup_path = None
            backup_made = False
            old_index = self.index
            try:
                v_for_backup = old_index.index_version
                backup_path = f"{self.index_path}.refresh-backup-{v_for_backup}"
                if persist and Path(self.index_path).exists():
                    if Path(backup_path).exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(self.index_path, backup_path)
                    backup_made = True

                with self.quiesce_writers():
                    v0 = old_index.index_version
                    report["v0"] = v0
                    report["stale_before"] = self._count_gold_lexical_stale(old_index)
                    fresh, gates = self._build_verified_fresh_index()
                    report.update(gates)

                    if self.index is not old_index or old_index.index_version != v0:
                        raise RuntimeError("no-writer assertion failed during fresh refresh")

                    for cid in texts:
                        if cid in fresh.concept_id_to_idx:
                            report["refreshed"].append(cid)
                        else:
                            report["skipped"].append({"id": cid, "reason": "not_in_index"})

                    self.index = fresh
                    if persist:
                        fresh.save(self.index_path)
                        report["swapped"] = True

                if backup_made:
                    try:
                        shutil.rmtree(backup_path)
                    except Exception:
                        logger.warning("refresh_concepts: could not remove successful refresh backup")

            except BaseException:
                self.index = old_index
                if backup_made and backup_path and Path(backup_path).exists():
                    try:
                        if Path(self.index_path).exists():
                            shutil.rmtree(self.index_path)
                        shutil.move(backup_path, self.index_path)
                        restored = IncrementalTfidfIndex()
                        if restored.load(self.index_path):
                            self.index = restored
                        logger.warning("refresh_concepts: restored fresh-refresh backup after failure")
                    except Exception as restore_exc:
                        logger.error(f"refresh_concepts: fresh-refresh backup restore FAILED: {restore_exc}")
                raise

            if persist:
                try:
                    post_swap = self.verify_index_integrity()
                    report["post_swap_integrity"] = post_swap
                    post_repair = post_swap
                    if not post_swap.get("is_healthy", False):
                        repair = self.repair_index_drift(integrity=post_swap)
                        post_repair = repair.get("post_repair_integrity") or self.verify_index_integrity()
                    report["post_repair_integrity"] = post_repair
                    report["requeue_ids"] = list(post_repair.get("orphan_ids", []))
                    if post_repair.get("ghost_ids"):
                        report["convergence_error"] = "remaining_ghosts"
                except Exception as convergence_exc:
                    report["convergence_error"] = type(convergence_exc).__name__
                    if report["post_swap_integrity"]:
                        report["requeue_ids"] = list(
                            report["post_swap_integrity"].get("orphan_ids", [])
                        )
                    logger.warning(
                        "refresh_concepts: post-swap convergence failed: %s",
                        convergence_exc,
                    )

            if persist and refresh_embeddings:
                for cid, text in texts.items():
                    if self._refresh_concept_embedding(cid, text):
                        report["embeddings_refreshed"] += 1
            return report
        finally:
            try:
                unlock_fd(fd)
            finally:
                os.close(fd)
                report["duration_s"] = round(time.perf_counter() - t0, 3)

    def refresh_concepts(
        self,
        concept_ids,
        *,
        persist: bool = True,
        refresh_embeddings: bool = True,
        host_memory_pressure_state: str = "unknown",
        host_pressure_reasons: list[str] | None = None,
    ) -> dict:
        """Refresh the stored representation of existing concepts in place.

        RETRIEVAL-125 design §4.2 — the steady-state per-concept refresh lane
        (Mode B), counterpart to the bulk clean-rebuild ``rebuild_and_swap_repair``
        (Mode A). For each id: assemble searchable text via the shared
        ``build_searchable_text`` helper, call ``index.refresh_concept`` (in-place
        DTC update), then recompute DF from scratch (A1 — DF is derived, never
        delta-maintained) and run one ``force_idf_recalculation`` over the batch.

        Mandatory verify gates run in-memory BEFORE persisting; on any failure the
        on-disk backup is restored and the in-memory index reloaded from it, so the
        index is left byte-identical to its pre-refresh state:
          * DF integrity: ``_recount_df_delta`` == 0 (the primary correctness gate).
          * Consistency invariant: count == dtc == cids.
          * Per-id parity: stored term counts == ``extract_terms(text)`` (overlap 1.0
            by construction — a mismatch means refresh_concept silently no-op'd).
          * No gold regression: gold lexical_stale count does not increase (sanity).
          * No-writer assertion: index_version advanced ONLY by our refreshes.

        Writers are quiesced (skip-not-block) for the whole sequence and a
        cross-process reflection.lock flock prevents a concurrent rebuild; if a
        reflection writer holds it, we defer cleanly.

        Args:
            concept_ids: existing concept ids to refresh.
            persist: write the refreshed index to disk on success. persist=False
                mutates only the in-memory index (no save, no backup, no rollback)
                and is intended for throwaway copy-backed harness/offline engines,
                NOT the live runtime singleton.
            refresh_embeddings: also re-embed + persist embeddings (live SQLite write;
                set False for copy/offline harness runs against a read-only DB).
            host_memory_pressure_state: normalized state sampled by the orchestration
                layer. High and critical states defer copy-backed rebuild work.
            host_pressure_reasons: bounded reason codes from the same snapshot.

        Returns a report dict (refreshed / skipped / gate metrics / swapped).
        """
        report: dict[str, Any] = {
            "refreshed": [],
            "skipped": [],
            "stale_before": None,
            "stale_after": None,
            "df_recount_delta": None,
            "v0": None,
            "swapped": False,
            "embeddings_refreshed": 0,
            "deferred": None,
        }
        t0 = time.perf_counter()
        ids = list(dict.fromkeys(concept_ids))  # de-dup, preserve order
        if not ids:
            report["duration_s"] = 0.0
            return report
        pressure_state = str(host_memory_pressure_state or "unknown").strip().lower()
        pressure_reasons = [str(reason)[:100] for reason in (host_pressure_reasons or [])[:20]]
        if _env_flag("PITH_TFIDF_FRESH_REFRESH_ENABLED", False):
            return self._refresh_concepts_fresh(
                ids,
                persist=persist,
                refresh_embeddings=refresh_embeddings,
                host_memory_pressure_state=pressure_state,
                host_pressure_reasons=pressure_reasons,
            )

        lock_dir = Path(resolve_data_dir()) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "reflection.lock"
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                lock_fd_exclusive(fd)
            except BlockingIOError:
                report["deferred"] = "reflection_active"
                return report

            backup_path = None
            backup_made = False
            try:
                v_for_backup = self.index.index_version
                backup_path = f"{self.index_path}.refresh-backup-{v_for_backup}"
                if persist and Path(self.index_path).exists():
                    if Path(backup_path).exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(self.index_path, backup_path)
                    backup_made = True

                with self.quiesce_writers():
                    v0 = self.index.index_version
                    report["v0"] = v0
                    report["stale_before"] = self._count_gold_lexical_stale(self.index)

                    # --- Refresh each id in place; remember texts for embeddings ---
                    texts: dict[str, str] = {}
                    for cid in ids:
                        row = self._load_concept_row_for_index(cid)
                        if row is None:
                            report["skipped"].append({"id": cid, "reason": "not_current"})
                            continue
                        text = build_searchable_text(row)
                        if not text.strip():
                            report["skipped"].append({"id": cid, "reason": "empty_text"})
                            continue
                        if self.index.refresh_concept(cid, text):
                            report["refreshed"].append(cid)
                            texts[cid] = text
                        else:
                            report["skipped"].append({"id": cid, "reason": "not_in_index"})

                    # --- A1: DF derived from scratch, then one IDF recalc over batch ---
                    self.index.recompute_document_frequencies()
                    self.index.force_idf_recalculation()

                    # --- Mandatory verify gates (in-memory, before persist) ---
                    report["stale_after"] = self._count_gold_lexical_stale(self.index)

                    df_delta = self._recount_df_delta(self.index)
                    report["df_recount_delta"] = df_delta
                    if df_delta != 0:
                        raise RuntimeError(f"verify failed: DF recount delta {df_delta} != 0")

                    count = self.index.document_count
                    dtc = len(self.index.document_term_counts)
                    cids = len(self.index.concept_ids)
                    if not (count == dtc == cids):
                        raise RuntimeError(f"verify failed: consistency invariant count={count} dtc={dtc} cids={cids}")

                    # Per-id parity: stored counts must equal a fresh extract of the
                    # text we fed in (overlap 1.0). A mismatch means a silent no-op.
                    for cid, text in texts.items():
                        ridx = self.index.concept_id_to_idx.get(cid)
                        if ridx is None or self.index.document_term_counts[ridx] != self.index.extract_terms(text):
                            raise RuntimeError(f"verify failed: per-id parity mismatch for {cid}")

                    # No gold regression (sanity; -1 == audit unavailable).
                    sb, sa = report["stale_before"], report["stale_after"]
                    if sb is not None and sa is not None and sb >= 0 and sa > sb:
                        raise RuntimeError(f"verify failed: gold lexical_stale regressed {sb} -> {sa}")

                    # No-writer assertion: only our N refreshes bumped the version.
                    expected_v = v0 + len(report["refreshed"])
                    if self.index.index_version != expected_v:
                        raise RuntimeError(
                            f"no-writer assertion failed: index_version "
                            f"{self.index.index_version} != expected {expected_v}"
                        )

                    if persist:
                        self.index.save(self.index_path)  # STABILITY-020 atomic swap
                        report["swapped"] = True
                    else:
                        report["swapped"] = False

                # --- Embeddings (best-effort, outside the quiesce window) ---
                if persist and refresh_embeddings:
                    for cid, text in texts.items():
                        if self._refresh_concept_embedding(cid, text):
                            report["embeddings_refreshed"] += 1

                if backup_made:
                    try:
                        shutil.rmtree(backup_path)
                    except Exception:
                        pass
                return report

            except BaseException:
                # Restore on-disk backup AND reload in-memory (it was mutated).
                if backup_made and backup_path and Path(backup_path).exists():
                    try:
                        if Path(self.index_path).exists():
                            shutil.rmtree(self.index_path)
                        shutil.move(backup_path, self.index_path)
                        restored = IncrementalTfidfIndex()
                        if restored.load(self.index_path):
                            self.index = restored
                        logger.warning("refresh_concepts: restored on-disk backup + reloaded after failure")
                    except Exception as restore_exc:
                        logger.error(f"refresh_concepts: backup restore FAILED: {restore_exc}")
                raise
        finally:
            try:
                unlock_fd(fd)
            finally:
                os.close(fd)
                report["duration_s"] = round(time.perf_counter() - t0, 3)

    def ensure_lifecycle_concepts_indexed(
        self,
        concept_ids,
        *,
        persist: bool = True,
        refresh_embeddings: bool = True,
    ) -> dict:
        """Ensure active/current concept IDs exist in the lexical index.

        Unlike ``refresh_concepts``, this lifecycle reconciliation path revives
        lazy tombstones and adds IDs that compaction removed from the mapping.
        One quiesce window and at most one strict persistence checkpoint cover
        the de-duplicated batch.
        """
        report: dict[str, Any] = {
            "ensured": [],
            "added": [],
            "revived": [],
            "skipped": [],
            "df_recount_delta": None,
            "swapped": False,
            "embeddings_refreshed": 0,
            "deferred": None,
        }
        started = time.perf_counter()
        ids = list(dict.fromkeys(str(value) for value in concept_ids if value))
        if not ids:
            report["duration_s"] = 0.0
            return report

        lock_dir = Path(resolve_data_dir()) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "reflection.lock"
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                lock_fd_exclusive(fd)
            except BlockingIOError:
                report["deferred"] = "reflection_active"
                return report

            backup_path = None
            backup_made = False
            try:
                backup_path = f"{self.index_path}.lifecycle-ensure-backup-{self.index.index_version}"
                if persist and Path(self.index_path).exists():
                    if Path(backup_path).exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(self.index_path, backup_path)
                    backup_made = True

                texts: dict[str, str] = {}
                with self.quiesce_writers():
                    v0 = self.index.index_version
                    for concept_id in ids:
                        with read_snapshot_db("lifecycle_index_ensure_load") as conn:
                            row = conn.execute(
                                """SELECT data, summary, fragment_keywords
                                   FROM concepts
                                   WHERE id=? AND status='active' AND is_current=1""",
                                (concept_id,),
                            ).fetchone()
                        if row is None:
                            report["skipped"].append({"id": concept_id, "reason": "not_active_current"})
                            continue
                        searchable_text = build_searchable_text(
                            {
                                "data": row[0],
                                "summary": row[1],
                                "fragment_keywords": row[2] or "",
                            }
                        )
                        if not searchable_text.strip():
                            report["skipped"].append({"id": concept_id, "reason": "empty_text"})
                            continue

                        mapped_idx = self.index.concept_id_to_idx.get(concept_id)
                        was_deleted = mapped_idx in getattr(self.index, "deleted_indices", set())
                        if mapped_idx is None:
                            changed = self.index.add_concept(concept_id, searchable_text)
                            if changed:
                                report["added"].append(concept_id)
                        else:
                            changed = self.index.refresh_concept(concept_id, searchable_text)
                            if changed and was_deleted:
                                report["revived"].append(concept_id)
                        if not changed:
                            raise RuntimeError(f"failed to ensure active index membership for {concept_id}")
                        report["ensured"].append(concept_id)
                        texts[concept_id] = searchable_text

                    if report["ensured"]:
                        self.index.recompute_document_frequencies()
                        self.index.force_idf_recalculation()

                    report["df_recount_delta"] = self._recount_df_delta(self.index)
                    if report["df_recount_delta"] != 0:
                        raise RuntimeError(
                            f"verify failed: DF recount delta {report['df_recount_delta']} != 0"
                        )
                    if not (
                        self.index.document_count
                        == len(self.index.document_term_counts)
                        == len(self.index.concept_ids)
                    ):
                        raise RuntimeError("verify failed: lifecycle ensure index consistency invariant")
                    for concept_id in report["ensured"]:
                        if not self.index.contains_active_concept(concept_id):
                            raise RuntimeError(f"{concept_id} remains absent from the live index")
                    expected_version = v0 + len(report["ensured"])
                    if self.index.index_version != expected_version:
                        raise RuntimeError(
                            f"no-writer assertion failed: index_version "
                            f"{self.index.index_version} != expected {expected_version}"
                        )
                    if persist:
                        self.index.save(self.index_path)
                        report["swapped"] = True

                if persist and refresh_embeddings:
                    for concept_id, searchable_text in texts.items():
                        if self._refresh_concept_embedding(concept_id, searchable_text):
                            report["embeddings_refreshed"] += 1
                if backup_made:
                    try:
                        shutil.rmtree(backup_path)
                    except Exception:
                        pass
                return report
            except BaseException:
                if backup_made and backup_path and Path(backup_path).exists():
                    if Path(self.index_path).exists():
                        shutil.rmtree(self.index_path)
                    shutil.move(backup_path, self.index_path)
                    restored = IncrementalTfidfIndex()
                    if restored.load(self.index_path):
                        self.index = restored
                    logger.warning("lifecycle ensure: restored index backup after failure")
                raise
        finally:
            try:
                unlock_fd(fd)
            finally:
                os.close(fd)
                report["duration_s"] = round(time.perf_counter() - started, 3)


# Global instance - EXACT MATCH of original retrieval.py
retrieval_engine = RetrievalEngine()


def _index_contains_active_concept(concept_id: str) -> bool:
    index = retrieval_engine.index
    if hasattr(index, "contains_active_concept"):
        return bool(index.contains_active_concept(concept_id))
    idx = index.concept_id_to_idx.get(concept_id)
    deleted = getattr(index, "deleted_indices", set())
    return idx is not None and idx not in deleted


def _persisted_index_contains_active_concept(concept_id: str) -> bool:
    document_map_path = Path(retrieval_engine.index_path) / "document_map.json"
    document_map = json.loads(document_map_path.read_text())
    idx = document_map.get("concept_id_to_idx", {}).get(concept_id)
    deleted = set(document_map.get("deleted_indices", []))
    return idx is not None and idx not in deleted


def evict_lifecycle_concepts(
    concept_ids,
    *,
    persist: bool,
    source: str,
) -> dict:
    """Evict exact IDs with bounded retry and one checkpoint per batch."""
    ids = list(dict.fromkeys(str(value) for value in concept_ids if value))
    report = {"succeeded": [], "failed": [], "attempts": 0}
    if not ids:
        return report
    for attempt in range(1, 3):
        report["attempts"] = attempt
        try:
            for concept_id in ids:
                retrieval_engine.remove_concept(
                    concept_id,
                    persist=False,
                    skip_periodic_checkpoint=True,
                )
            if persist:
                index = retrieval_engine.index
                if not (
                    index.document_count
                    == len(index.document_term_counts)
                    == len(index.concept_ids)
                ):
                    raise RuntimeError("refusing lifecycle eviction checkpoint for inconsistent index")
                retrieval_engine._auto_save(strict=True)
            failed = [concept_id for concept_id in ids if _index_contains_active_concept(concept_id)]
            if persist:
                failed.extend(
                    concept_id
                    for concept_id in ids
                    if concept_id not in failed and _persisted_index_contains_active_concept(concept_id)
                )
            if failed:
                raise RuntimeError(f"index eviction verification failed for {sorted(set(failed))}")
            report["succeeded"] = ids
            logger.info(
                "MAINT-097: lifecycle index batch eviction succeeded count=%d source=%s persist=%s attempt=%d",
                len(ids),
                source,
                persist,
                attempt,
            )
            return report
        except Exception as exc:
            log = logger.warning if attempt == 1 else logger.error
            log(
                "MAINT-097: lifecycle index batch eviction failed count=%d source=%s "
                "persist=%s attempt=%d error=%s",
                len(ids),
                source,
                persist,
                attempt,
                exc,
            )
    report["failed"] = ids
    return report


def ensure_lifecycle_concepts_indexed(
    concept_ids,
    *,
    persist: bool,
    source: str,
) -> dict:
    """Ensure exact active IDs with bounded retry and strict verification."""
    ids = list(dict.fromkeys(str(value) for value in concept_ids if value))
    result = {"succeeded": [], "failed": [], "attempts": 0, "deferred": None}
    if not ids:
        return result
    for attempt in range(1, 3):
        result["attempts"] = attempt
        try:
            report = retrieval_engine.ensure_lifecycle_concepts_indexed(
                ids,
                persist=persist,
                refresh_embeddings=persist,
            )
            if report.get("deferred"):
                result["deferred"] = report["deferred"]
                break
            skipped = {item["id"] for item in report.get("skipped", [])}
            failed = [
                concept_id
                for concept_id in ids
                if concept_id not in skipped and not _index_contains_active_concept(concept_id)
            ]
            if persist:
                failed.extend(
                    concept_id
                    for concept_id in ids
                    if concept_id not in skipped
                    and concept_id not in failed
                    and not _persisted_index_contains_active_concept(concept_id)
                )
            if failed:
                raise RuntimeError(f"index ensure verification failed for {sorted(set(failed))}")
            result["succeeded"] = [concept_id for concept_id in ids if concept_id not in skipped]
            result["failed"] = list(skipped)
            logger.info(
                "MAINT-097: lifecycle index batch ensure succeeded count=%d skipped=%d source=%s attempt=%d",
                len(result["succeeded"]),
                len(skipped),
                source,
                attempt,
            )
            return result
        except Exception as exc:
            log = logger.warning if attempt == 1 else logger.error
            log(
                "MAINT-097: lifecycle index batch ensure failed count=%d source=%s attempt=%d error=%s",
                len(ids),
                source,
                attempt,
                exc,
            )
    result["failed"] = ids
    return result


def evict_lifecycle_concept(
    concept_id: str,
    *,
    persist: bool,
    source: str,
) -> bool:
    """Evict one lifecycle-inactive concept with bounded retry and verification."""
    for attempt in range(1, 3):
        try:
            retrieval_engine.remove_concept(
                concept_id,
                persist=persist,
                skip_periodic_checkpoint=not persist,
                strict_persist=persist,
            )
            if _index_contains_active_concept(concept_id):
                raise RuntimeError(f"{concept_id} remains active in the live index")
            if persist and _persisted_index_contains_active_concept(concept_id):
                raise RuntimeError(f"{concept_id} remains active in the persisted index")
            logger.info(
                "MAINT-096: lifecycle index eviction succeeded "
                "concept_id=%s source=%s persist=%s attempt=%d",
                concept_id,
                source,
                persist,
                attempt,
            )
            return True
        except Exception as exc:
            log = logger.warning if attempt == 1 else logger.error
            log(
                "MAINT-096: lifecycle index eviction failed "
                "concept_id=%s source=%s persist=%s attempt=%d error=%s",
                concept_id,
                source,
                persist,
                attempt,
                exc,
            )
    return False
