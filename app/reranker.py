"""Neural Cross-Encoder Reranker for Production Retrieval.

Stage 1: Embedding search returns top-N candidates (default 40)
Stage 2: Cross-encoder reranks candidates by joint query-document scoring -> top-K (default 14)

Uses BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder.
Runs locally on CPU/MPS — no API calls.
Feature-gated via PITH_RERANKER env var.

Optimizations (PERF-040 / RETRIEVAL-058):
  - Eager model loading at startup via warmup() — eliminates 2339ms cold start
  - Confidence-gated reranking — skips when embedding scores have high spread
  - Sub-batch processing — processes in chunks to reduce padding waste

Replaces the LLM-based reranker (Haiku API, 300-500ms, per-query cost).
Spec: RETRIEVAL_RERANKER_BM25_SPEC (Fix A)
"""

import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger("pith.reranker")

# Model loaded lazily on first use, or eagerly via warmup()
_cross_encoder = None
_cross_encoder_load_failed = False
_cross_encoder_load_error: str | None = None
_cross_encoder_failure_count = 0
_MODEL_NAME = os.environ.get(
    "PITH_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)

# --- Confidence gate config ---
# Skip reranking when embedding score spread exceeds this threshold.
# High spread = embedding search already confident in ordering.
# Calibrate via PITH_RERANKER_GATE_SPREAD env var (default 0.15).
_GATE_SPREAD = float(os.environ.get("PITH_RERANKER_GATE_SPREAD", "0.15"))
_GATE_ENABLED = os.environ.get("PITH_RERANKER_GATE", "").lower() in ("true", "1")

# --- Sub-batch config ---
# Process candidates in sub-batches to reduce padding waste on MPS.
# Default sub-batch size of 4 balances parallelism vs padding overhead.
_SUB_BATCH_SIZE = int(os.environ.get("PITH_RERANKER_SUB_BATCH", "4"))


class RerankerUnavailable(RuntimeError):
    """Raised after cross-encoder initialization has failed in this process."""


def is_cross_encoder_available() -> bool:
    """Return whether the cross-encoder is eligible for use.

    This means "not known failed", not "already loaded"; lazy loading remains
    available until the first constructor or warmup-prediction failure.
    """
    return not _cross_encoder_load_failed


def get_cross_encoder_state() -> dict[str, Any]:
    """Expose process-local reranker availability for diagnostics/tests."""
    return {
        "loaded": _cross_encoder is not None,
        "available": is_cross_encoder_available(),
        "load_failed": _cross_encoder_load_failed,
        "load_error": _cross_encoder_load_error,
        "failure_count": _cross_encoder_failure_count,
        "model": _MODEL_NAME,
        "device": os.environ.get("PITH_RERANKER_DEVICE", "cpu"),
    }


def reset_cross_encoder_state_for_tests() -> None:
    """Reset process-local model state for focused tests."""
    global _cross_encoder, _cross_encoder_load_failed
    global _cross_encoder_load_error, _cross_encoder_failure_count
    _cross_encoder = None
    _cross_encoder_load_failed = False
    _cross_encoder_load_error = None
    _cross_encoder_failure_count = 0


def _mark_cross_encoder_unavailable(error: Exception) -> None:
    global _cross_encoder, _cross_encoder_load_failed
    global _cross_encoder_load_error, _cross_encoder_failure_count
    _cross_encoder = None
    _cross_encoder_load_failed = True
    _cross_encoder_load_error = str(error)
    _cross_encoder_failure_count += 1


def _get_cross_encoder():
    """Lazy-load the cross-encoder model. Thread-safe via GIL."""
    global _cross_encoder
    if _cross_encoder_load_failed:
        raise RerankerUnavailable(
            f"Cross-encoder unavailable after previous load failure: "
            f"{_cross_encoder_load_error}"
        )
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _device = os.environ.get("PITH_RERANKER_DEVICE", "cpu")
        logger.info(f"Loading cross-encoder model: {_MODEL_NAME} on {_device}")
        t0 = time.perf_counter()
        try:
            _cross_encoder = CrossEncoder(_MODEL_NAME, device=_device)
        except Exception as e:
            _mark_cross_encoder_unavailable(e)
            logger.warning(
                "Cross-encoder load failed; reranker unavailable until restart: %s",
                e,
            )
            raise RerankerUnavailable(str(e)) from e
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Cross-encoder loaded in {elapsed:.0f}ms")
    return _cross_encoder


def warmup() -> bool:
    """Eagerly load the cross-encoder model at server startup.

    PERF-040: Eliminates 2339ms cold start on first query.
    Call this from server initialization when PITH_RERANKER is enabled.
    """
    try:
        model = _get_cross_encoder()
        # Run a single dummy prediction to warm MPS/CUDA kernels
        model.predict([("warmup", "warmup")], show_progress_bar=False)
        logger.info("Reranker warmup complete — model loaded and kernels primed")
        return True
    except Exception as e:
        if not _cross_encoder_load_failed:
            _mark_cross_encoder_unavailable(e)
        logger.warning(f"Reranker warmup failed (non-fatal): {e}")
        return False


def _should_skip_reranking(candidates: list, stage2_k: int) -> bool:
    """Confidence gate: skip reranking when embedding scores have high spread.

    RETRIEVAL-058 Tier 2a: When the embedding search is already confident
    (top-1 score >> top-K score), reranking adds latency without changing
    the ordering. Skip it.

    Returns True if reranking should be skipped.
    """
    if not _GATE_ENABLED:
        return False

    if len(candidates) < 2:
        return True

    # Use relevance_score (embedding cosine similarity) from SearchResult
    scores = [getattr(c, "relevance_score", 0.0) for c in candidates]
    top_score = scores[0]  # candidates arrive sorted by embedding score
    kth_score = scores[min(stage2_k, len(scores) - 1)]
    spread = top_score - kth_score

    if spread >= _GATE_SPREAD:
        logger.info(
            f"Reranker gate: SKIP (spread={spread:.4f} >= threshold={_GATE_SPREAD}, "
            f"top={top_score:.4f}, k={kth_score:.4f})"
        )
        return True

    logger.debug(
        f"Reranker gate: PASS (spread={spread:.4f} < threshold={_GATE_SPREAD})"
    )
    return False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _record_reranker_metric(
    metric: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> None:
    try:
        from app.ops.metrics import metrics as _metrics

        _metrics.record(metric, value, labels or {})
    except Exception:
        pass


def _predict_sub_batched(model, pairs: list) -> np.ndarray:
    """Score query-document pairs in sub-batches to reduce padding waste.

    RETRIEVAL-058 Tier 2b: Instead of one big batch padded to the longest
    sequence, process in chunks of _SUB_BATCH_SIZE. Each sub-batch pads
    only to its own longest sequence, reducing wasted compute on MPS.
    """
    if _SUB_BATCH_SIZE <= 0 or len(pairs) <= _SUB_BATCH_SIZE:
        # Single batch — no sub-batching needed
        return model.predict(pairs, show_progress_bar=False)

    all_scores = []
    for i in range(0, len(pairs), _SUB_BATCH_SIZE):
        chunk = pairs[i : i + _SUB_BATCH_SIZE]
        chunk_scores = model.predict(chunk, show_progress_bar=False)
        all_scores.extend(chunk_scores if hasattr(chunk_scores, '__iter__') else [chunk_scores])

    return np.array(all_scores, dtype=np.float32)


def rerank_results(
    question: str,
    candidates: list,
    stage2_k: int = None,
    *,
    deadline: Any = None,
    phase: str = "retrieval.reranker",
    min_remaining_ms: float = 0.0,
    max_child_ms: float | None = None,
) -> list:
    """Rerank SearchResult candidates using neural cross-encoder.

    Args:
        question: The user's query (raw, not firmware-decorated)
        candidates: List of SearchResult objects from embedding search
        stage2_k: Number of results to return (default from env)

    Returns:
        Reranked list of SearchResult objects, len <= stage2_k
    """
    if stage2_k is None:
        stage2_k = int(os.environ.get("PITH_RERANKER_STAGE2_K", "14"))

    if len(candidates) <= stage2_k:
        logger.info(f"Reranker: only {len(candidates)} candidates, skipping rerank")
        _record_reranker_metric(
            "ct_phase_reranker_skipped_total",
            1.0,
            {"reason": "candidate_count"},
        )
        return candidates

    # RETRIEVAL-058: Confidence gate — skip when embedding spread is high
    if _should_skip_reranking(candidates, stage2_k):
        _record_reranker_metric(
            "ct_phase_reranker_skipped_total",
            1.0,
            {"reason": "confidence_gate"},
        )
        return candidates[:stage2_k]

    if (
        deadline is not None
        and getattr(deadline, "enabled", False)
        and _env_bool("PITH_RERANKER_DEADLINE_ENABLED", True)
    ):
        if not deadline.can_start(phase, min_remaining_ms=min_remaining_ms):
            deadline.skip(
                phase,
                "deadline_before_start",
                priority="optional",
                min_remaining_ms=min_remaining_ms,
            )
            _record_reranker_metric(
                "ct_phase_reranker_skipped_total",
                1.0,
                {"reason": "deadline_before_start"},
            )
            return candidates[:stage2_k]

        child_budget_ms = deadline.child_budget_ms(
            phase,
            max_child_ms
            if max_child_ms is not None
            else float(os.environ.get("PITH_RERANKER_MAX_CHILD_MS", "900")),
            min_remaining_ms=min_remaining_ms,
        )
        estimate_ms_per_candidate = _env_float_or_none(
            "PITH_RERANKER_ESTIMATE_MS_PER_CANDIDATE"
        )
        if estimate_ms_per_candidate is None:
            deadline.skip(
                phase,
                "missing_calibration",
                priority="optional",
                child_budget_ms=child_budget_ms,
            )
            _record_reranker_metric(
                "ct_phase_reranker_skipped_total",
                1.0,
                {"reason": "missing_calibration"},
            )
            return candidates[:stage2_k]

        if _env_bool("PITH_RERANKER_BUDGETED_SHRINK_ENABLED", False):
            max_candidates = int(child_budget_ms // estimate_ms_per_candidate)
            if max_candidates <= stage2_k:
                deadline.skip(
                    phase,
                    "calibrated_budget_too_small",
                    priority="optional",
                    child_budget_ms=child_budget_ms,
                    max_candidates=max_candidates,
                )
                _record_reranker_metric(
                    "ct_phase_reranker_skipped_total",
                    1.0,
                    {"reason": "calibrated_budget_too_small"},
                )
                return candidates[:stage2_k]
            candidates = candidates[:max_candidates]

    if not is_cross_encoder_available():
        logger.info("Reranker skipped: cross-encoder unavailable")
        _record_reranker_metric(
            "ct_phase_reranker_skipped_total",
            1.0,
            {"reason": "unavailable"},
        )
        return candidates[:stage2_k]

    t0 = time.perf_counter()

    try:
        model = _get_cross_encoder()

        # Build query-document pairs for cross-encoder scoring
        # Use full summary (cross-encoder handles its own truncation via max_length)
        pairs = [(question, c.summary or "") for c in candidates]

        # RETRIEVAL-058: Sub-batch scoring to reduce padding waste
        scores = _predict_sub_batched(model, pairs)

        # Sort by score descending, take top stage2_k
        ranked_indices = np.argsort(scores)[::-1][:stage2_k]
        reranked = [candidates[i] for i in ranked_indices]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_score = float(scores[ranked_indices[0]]) if len(ranked_indices) > 0 else 0.0
        gate_status = "gate=on" if _GATE_ENABLED else "gate=off"
        batch_mode = f"sub_batch={_SUB_BATCH_SIZE}" if _SUB_BATCH_SIZE > 0 and len(pairs) > _SUB_BATCH_SIZE else "single_batch"
        _record_reranker_metric("ct_phase_reranker_ms", elapsed_ms, {"state": "completed"})
        _record_reranker_metric(
            "ct_phase_reranker_candidate_count",
            float(len(candidates)),
        )
        logger.info(
            f"Reranker: {len(candidates)}->{len(reranked)} in {elapsed_ms:.0f}ms "
            f"(top_score={top_score:.3f}, {gate_status}, {batch_mode})"
        )
        return reranked

    except RerankerUnavailable as e:
        logger.info(f"Reranker unavailable ({e}), falling back to embedding order")
        _record_reranker_metric(
            "ct_phase_reranker_skipped_total",
            1.0,
            {"reason": "unavailable"},
        )
        return candidates[:stage2_k]

    except Exception as e:
        logger.warning(f"Reranker failed ({e}), falling back to embedding order")
        _record_reranker_metric(
            "ct_phase_reranker_skipped_total",
            1.0,
            {"reason": "error", "error": type(e).__name__},
        )
        return candidates[:stage2_k]
