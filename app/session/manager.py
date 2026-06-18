"""SessionManager orchestrator — thin class inheriting all mixin behavior.

Extracted from session/__init__.py lines 856-1040 per ARCH-009.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from app.core.constants import (
    FRESHNESS_EARLIER_TODAY_UPPER,
    FRESHNESS_HOURS_AGO_UPPER,
    FRESHNESS_JUST_NOW_MINS,
    FRESHNESS_MINUTES_AGO_UPPER,
    FRESHNESS_ONE_HOUR_UPPER,
    FRESHNESS_YESTERDAY_UPPER,
    GOV_EVENT_CCL_VIOLATIONS_DETECTED,
    GOV_EVENT_CIRCUIT_BREAKER_TRIPPED,
    GOV_EVENT_COMPACTION_REINJECTION,
    GOV_EVENT_CONTRADICTION_REVIEW,
    GOV_EVENT_CONVERSATION_TURN_COMPLETE,
    GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
    GOV_EVENT_RESUME_CONTEXT_INJECTION,
    MINUTES_PER_HOUR,
)
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.core.models import (
    ActivatedConcept,
    ActiveDirectionality,
    ActiveUncertainty,
    AreaStrength,
    Concept,
    ConceptEvolution,
    ConceptEvolutionRecord,
    ConversationTurnRequest,
    ConversationTurnResponse,
    CuriosityFrontierItem,
    CurrentStateAssessment,
    EvolvedConcept,
    GoalSummary,
    LearnedConcept,
    PendingQuestionSummary,
    PresentMomentOrientation,
    RecentConceptSummary,
    RecentEvolutionSummary,
    SearchResult,
    SessionEndRequest,
    SessionInfo,
    SessionLearnRequest,
    SessionLearnResponse,
    SessionStartResponse,
)
from app.session.self_model import self_model_manager
from app.storage import (
    _get_connection,
    cleanup_expired_snapshots,
    count_associations,
    count_sessions,
    get_related_concepts,
    list_concepts,
    load_associations,
    load_concept,
    load_recent_concepts,
    load_resume_snapshot,
    load_session_velocity,
    recover_interrupted_sessions,
    save_concept,
    # Resume Context v1.1
    save_resume_snapshot,
    save_session,
    update_session,
)
from app.storage.concepts import load_concepts_batch
from app.cognitive.taxonomy import (  # DEBT-030/DEBT-108
    classify_knowledge_area,
    infer_knowledge_area,
    normalize_knowledge_area,
)

logger = logging.getLogger(__name__)

_REQUEST_SESSION_ACTIVE: ContextVar[bool] = ContextVar(
    "pith_request_session_active",
    default=False,
)
_REQUEST_SESSION: ContextVar[SessionInfo | None] = ContextVar(
    "pith_request_session",
    default=None,
)

from app.session.orient import OrientMixin
from app.session.lifecycle import LifecycleMixin
from app.session.compaction import CompactionMixin
from app.session.resume import ResumeMixin
from app.session.turn import ConversationTurnMixin
from app.session.learn import SessionLearnMixin

logger = logging.getLogger(__name__)


class SessionManager(
    OrientMixin,
    LifecycleMixin,
    CompactionMixin,
    ResumeMixin,
    ConversationTurnMixin,
    SessionLearnMixin,
):
    """Manages session lifecycle and orientation generation."""

    # CTX-005: Compaction recovery quality factor weights (sum to 1.0)
    COMPACTION_QUALITY_HAS_SNAPSHOT: float = 0.4
    COMPACTION_QUALITY_HAS_TASK: float = 0.2
    COMPACTION_QUALITY_HAS_PINNED: float = 0.2
    COMPACTION_QUALITY_HAS_GIST: float = 0.2

    def __init__(self):
        self._global_current_session: SessionInfo | None = None
        self._in_memory_sessions: dict[str, SessionInfo] = {}
        self._recovery_done: bool = False
        # SESSION-011: Capture server startup time for recovery blast-radius scoping.
        # recover_interrupted_sessions() uses this to only interrupt sessions that
        # started before this process launched, protecting concurrent live sessions.
        from app.core.datetime_utils import _utc_now_iso as _startup_ts
        self._server_startup_iso: str = _startup_ts()
        self._conversation_turn_called: bool = False  # S0: first-call detection
        self._last_conversation_turn_at: float | None = None  # S0: timestamp for conversation boundary detection
        self._last_orientation_served_at: float | None = None  # S6: fallback orientation re-serve
        # B1: Active extraction request tracking (Attack 2 anti-nagging + Attack 5 suppression)
        self._last_extraction_request_types: set = set()
        self._suppressed_gap_types: set = set()
        # RETRO-001: One retrospective check per session
        self._retro_checked_this_session: bool = False
        # GOV-W2: Track activated concepts for correction detection on next turn
        self._last_activated_concept_ids: list[str] = []
        # GOV-W2: Cache activated concept dicts with embeddings for Layer 4 drift detection
        self._last_activated_concept_dicts: list[dict[str, Any]] = []
        # CTX Phase 2: Compaction detection state
        self._consecutive_empty_extractions: int = 0
        self._last_compaction_detected_at: float | None = None  # Cooldown tracking (CTX-2)
        self._compaction_false_positive_count: int = 0  # Session circuit breaker (CTX-2)
        self._compaction_predecessor_id: str | None = None  # SESSION-010: Cross-session predecessor
        self._compaction_detection_tier: str | None = None  # SESSION-010: HIGH or MEDIUM confidence
        self._episode_turn_counter: int = 0  # INFRA-002: monotonic per-session turn counter for episodes
        self._promoted_this_session: set[str] = set()  # ARCH-D05: rate-limit promotion checks
        self._cumulative_response_bytes: int = 0  # CTX-003: cumulative previous_response bytes for pressure scoring
        # CONCEPT_LIFECYCLE_SPEC L4: Track concept IDs created during session
        self._session_concept_ids: set[str] = set()
        # STABILITY-013: Strong references prevent GC of fire-and-forget tasks
        self._background_tasks: set = set()
        # PERF-FORT-2: Background auto-learn state (cross-turn)
        self._learn_executor = None  # Lazy-init ThreadPoolExecutor(max_workers=1)
        self._checkpoint_executor = None  # SESSION-004: Lazy-init for auto-checkpoint fire-and-forget
        self._last_autolearn_result: dict | None = None  # Previous turn's auto_learned dict
        self._last_autolearn_result_obj = None  # Previous turn's SessionLearnResponse object
        self._last_autolearn_budget_warnings: list = []  # Previous turn's budget warnings
        # PERF-013: Git state cache — populated once at session_start
        self.git_cache: GitCache | None = None
        self._cached_pinned_concepts: list[dict] | None = None  # CONTEXT-001: Turn-scoped cache
        self._cached_pinned_concepts_turn: int = -1  # Turn number when cache was set

    @property
    def current_session(self) -> SessionInfo | None:
        """Return request-local session when bound, else the global lifecycle session."""
        if _REQUEST_SESSION_ACTIVE.get():
            return _REQUEST_SESSION.get()
        return self._global_current_session

    @current_session.setter
    def current_session(self, value: SessionInfo | None) -> None:
        if _REQUEST_SESSION_ACTIVE.get():
            _REQUEST_SESSION.set(value)
        else:
            self._global_current_session = value

    def _active_session(self) -> SessionInfo | None:
        return self.current_session

    def _global_session(self) -> SessionInfo | None:
        return self._global_current_session

    def _in_memory_session_store(self) -> dict[str, SessionInfo]:
        store = getattr(self, "_in_memory_sessions", None)
        if store is None:
            store = {}
            self._in_memory_sessions = store
        return store

    def _remember_in_memory_session(self, session: SessionInfo | None) -> None:
        if session is not None and session.session_id:
            self._in_memory_session_store()[session.session_id] = session

    def _forget_in_memory_session(self, session_id: str | None) -> None:
        if session_id:
            self._in_memory_session_store().pop(session_id, None)

    def _in_memory_session(self, session_id: str | None) -> SessionInfo | None:
        if not session_id:
            return None
        session = self._in_memory_session_store().get(session_id)
        if session is not None and session.status == "active":
            return session
        return None

    def _push_request_session(self, session: SessionInfo | None):
        """Bind a request-local session without mutating global lifecycle state."""
        active_token = _REQUEST_SESSION_ACTIVE.set(True)
        session_token = _REQUEST_SESSION.set(session)
        return session_token, active_token

    def _pop_request_session(self, session_token, active_token) -> None:
        _REQUEST_SESSION.reset(session_token)
        _REQUEST_SESSION_ACTIVE.reset(active_token)

    def _on_bg_task_done(self, task) -> None:
        """Done callback for background tasks. Logs errors, removes reference, emits metrics."""
        self._background_tasks.discard(task)
        try:
            exc = task.exception()
            if exc:
                logger.error(f"Background task {task.get_name()} failed: {exc}")
                from app.ops.metrics import metrics as _bg_metrics

                _bg_metrics.record("bg_task_failure", 1.0, {"task": task.get_name()})
            else:
                from app.ops.metrics import metrics as _bg_metrics

                _bg_metrics.record("bg_task_success", 1.0, {"task": task.get_name()})
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            from app.ops.metrics import metrics as _bg_metrics

            _bg_metrics.record("bg_task_cancelled", 1.0, {"task": task.get_name()})
            pass  # Cancellation during shutdown is expected, not an error

    def _background_autolearn(
        self,
        learn_request,
        extracted: list | None,
        request_message: str,
        prev_msg: str,
        prev_response: str,
        bound_session: SessionInfo | None = None,
        raw_capture_ref: dict | None = None,
        active_binding_snapshot: dict | None = None,
    ):
        """PERF-FORT-2: Background auto-learn — runs in executor thread.

        Performs session_learn + episode recording + reflection tracking + pricing
        without blocking the conversation_turn response path.

        NOTE: session_learn uses the shared DB connection via storage backend.
        The storage backend's RLock serializes access, so background writes
        wait for main-path reads to complete and vice versa. This is safe
        but may add latency to whichever thread is waiting.

        STABILITY-037: Added shutdown check and DB retry with connection recycle.
        """
        import sqlite3
        from app.storage.backend import get_backend

        session_token = active_token = None
        if bound_session is not None:
            session_token, active_token = self._push_request_session(bound_session)

        # STABILITY-037: Don't start autolearn if server is shutting down
        try:
            backend = get_backend()
            if getattr(backend, 'is_shutting_down', False):
                logger.info("S-1-BG: Skipping autolearn — server shutting down")
                return

            _max_retries = 2
            _autolearn_succeeded = False
            for _attempt in range(_max_retries):
                try:
                    auto_learn_result = self.session_learn(learn_request)
                    logger.info(
                        f"S-1-BG: Auto-learned: {auto_learn_result.learning_events} events, "
                        f"sources={auto_learn_result.extraction_source_breakdown}"
                    )
                    if raw_capture_ref:
                        try:
                            from app.storage.turn_ingestion import mark_learning_status_default_db

                            mark_learning_status_default_db(
                                **raw_capture_ref,
                                status="attempted",
                                concepts_extracted=auto_learn_result.learning_events,
                            )
                        except Exception as _ledger_err:
                            logger.warning("turn_ingestion_ledger_update_failed: %s", _ledger_err)
                    _autolearn_succeeded = True
                    break  # Success — exit retry loop
                except sqlite3.ProgrammingError as e:
                    if "closed database" in str(e).lower() and _attempt < _max_retries - 1:
                        logger.warning(
                            f"S-1-BG: STABILITY-037 DB connection stale on attempt "
                            f"{_attempt + 1}/{_max_retries}, recycling and retrying: {e}"
                        )
                        try:
                            backend.get_connection()
                        except Exception:
                            pass
                        continue
                    logger.error(f"S-1-BG: Background auto-learn failed: {e}")
                    if raw_capture_ref:
                        try:
                            from app.storage.turn_ingestion import mark_learning_status_default_db

                            mark_learning_status_default_db(
                                **raw_capture_ref,
                                status="failed",
                                error=str(e),
                            )
                        except Exception as _ledger_err:
                            logger.warning("turn_ingestion_ledger_update_failed: %s", _ledger_err)
                    return

            if not _autolearn_succeeded:
                return

            # --- Post-autolearn tasks (outside retry loop) ---
            # Track rejected-after-request gaps
            if auto_learn_result and auto_learn_result.garbage_rejected > 0 and self._last_extraction_request_types:
                self._suppressed_gap_types.update(self._last_extraction_request_types)

            # --- INFRA-002: Episode recording (moved from main path) ---
            try:
                from app.core.config import FEATURE_FLAGS as _ep_ff
                if self.current_session and _ep_ff.get("EPISODES_ENABLED", False):
                    from app.features.episodes import record_episode
                    self._episode_turn_counter += 1
                    _ep_concept_ids = [c.concept_id for c in auto_learn_result.concepts_created]
                    _ep_changes = [
                        {"action": "created", "id": c.concept_id} for c in auto_learn_result.concepts_created
                    ] + [{"action": "evolved", "id": c.concept_id} for c in auto_learn_result.concepts_evolved]
                    record_episode(
                        session_id=self.current_session.session_id,
                        turn_number=self._episode_turn_counter,
                        intent_summary=(learn_request.knowledge_area or "")[:500],
                        classification=(learn_request.knowledge_area or "")[:200],
                        extracted_concept_ids=_ep_concept_ids,
                        concept_changes=_ep_changes,
                        raw_user_message=request_message[:5000] if request_message else None,
                        raw_assistant_response=(prev_response or "")[:5000] or None,
                    )
            except Exception as e:
                logger.warning(f"INFRA-002-BG: Episode recording failed (non-fatal): {e}")

            # --- RB-02: Reflection completion tracking (moved from main path) ---
            if auto_learn_result and auto_learn_result.learning_events > 0:
                try:
                    from app.storage import _db
                    from app.core.datetime_utils import _utc_now_iso as _bg_utc_now_iso
                    with _db() as conn:
                        conn.execute(
                            """UPDATE reflection_tracking
                               SET completed_at = ?,
                                   concepts_returned = ?,
                                   reflection_quality = 'auto_closed'
                               WHERE id = (
                                   SELECT id FROM reflection_tracking
                                   WHERE completed_at IS NULL
                                   ORDER BY created_at DESC LIMIT 1
                               )""",
                            (_bg_utc_now_iso(), auto_learn_result.learning_events),
                        )
                except Exception as e:
                    logger.debug(f"RB-02-BG: Reflection tracking failed (non-fatal): {e}")

            # --- PRICING-002: Meter concept-producing turns (moved from main path) ---
            if auto_learn_result and auto_learn_result.learning_events > 0:
                try:
                    from app.api.pricing import conversation_meter
                    conversation_meter.consume_turn()
                except Exception as e:
                    logger.debug(f"PRICING-002-BG: Metering failed (non-fatal): {e}")

            _workstream_link_result = None
            if auto_learn_result and auto_learn_result.concepts_created:
                try:
                    from app.features.threads import link_concepts_to_active_workstream

                    _workstream_link_result = link_concepts_to_active_workstream(
                        auto_learn_result.concepts_created,
                        binding_snapshot=active_binding_snapshot,
                    )
                    if _workstream_link_result.get("linked", 0) > 0:
                        logger.info(
                            "S-1-BG: Linked %s concepts to active Workstream %s",
                            _workstream_link_result.get("linked"),
                            _workstream_link_result.get("thread_id"),
                        )
                except Exception as e:
                    logger.warning("S-1-BG: Active Workstream concept linking failed (non-fatal): %s", e)

            # Store results for next turn's consumption (A1 amendment)
            # Build the auto_learned summary dict
            _auto_learned_dict = None
            _budget_warnings = auto_learn_result.budget_warnings or []
            if auto_learn_result.learning_events > 0:
                _auto_learned_dict = {
                    "events": auto_learn_result.learning_events,
                    "concepts_created": [c.concept_id for c in auto_learn_result.concepts_created],
                    "concepts_evolved": [c.concept_id for c in auto_learn_result.concepts_evolved],
                    "budget_warnings": _budget_warnings,
                }
                if _workstream_link_result:
                    _auto_learned_dict["workstream_links"] = _workstream_link_result
            # Atomic assignment — GIL protects reference swap
            self._last_autolearn_result = _auto_learned_dict
            self._last_autolearn_result_obj = auto_learn_result
            self._last_autolearn_budget_warnings = _budget_warnings

        except Exception as e:
            logger.error(f"S-1-BG: Background auto-learn failed: {e}", exc_info=True)
            self._last_autolearn_result = None
            self._last_autolearn_result_obj = None
            self._last_autolearn_budget_warnings = []
        finally:
            if session_token is not None and active_token is not None:
                self._pop_request_session(session_token, active_token)

    def _load_all_concepts(self) -> list[Concept]:
        """Load all active concepts from storage. Single scan point."""
        concept_ids = list_concepts()
        concepts_by_id = load_concepts_batch(concept_ids)
        return [concepts_by_id[cid] for cid in concept_ids if cid in concepts_by_id]
