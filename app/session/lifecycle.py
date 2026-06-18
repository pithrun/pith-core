"""LifecycleMixin — session start/end lifecycle management.

Extracted from session/__init__.py lines 1463-2201 per ARCH-009.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
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
from app.core.config import BENCHMARK_READONLY, BenchmarkIngestionMode
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
from app.session.helpers import REFLECTION_TRIGGER_THRESHOLD
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
    load_active_sessions_by_origin,
    load_session,
    load_session_velocity,
    recover_interrupted_sessions,
    save_concept,
    # Resume Context v1.1
    save_resume_snapshot,
    save_session,
    update_session,
)
from app.cognitive.taxonomy import (  # DEBT-030/DEBT-108
    classify_knowledge_area,
    infer_knowledge_area,
    normalize_knowledge_area,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Mixin providing lifecycle methods for SessionManager."""

    def start_session(
        self,
        context_hint: str = "",
        agent_id: str = "default",
        session_date: str | None = None,
        platform_hint: str = "unknown",
        surface_id: str = "unknown",
    ) -> SessionStartResponse:
        """Session bootstrap protocol: start session and return bootstrap context.

        Default path is intentionally lightweight: it avoids hydrating the full
        concept graph on every session_start. Set PITH_SESSION_START_FULL_BOOTSTRAP=1
        to restore the legacy full concept scan for targeted diagnostics.
        Persists session to SQLite for restart survival.
        """
        persist_session = not BENCHMARK_READONLY

        # One-time recovery: mark orphan active sessions from prior runs
        recovered_info = None
        if not self._recovery_done:
            recovered = 0
            if persist_session:
                # SESSION-011: Pass server startup time to scope blast radius —
                # only interrupt sessions that started before this process launched.
                recovered = recover_interrupted_sessions(
                    started_before=getattr(self, "_server_startup_iso", None)
                )
            else:
                logger.info("SESSION-001: Startup recovery skipped (PITH_BENCHMARK_READONLY)")
            self._recovery_done = True
            if recovered:
                logger.warning(
                    f"Startup recovery: {recovered} interrupted session(s) — "
                    f"knowledge from those sessions may be incomplete. "
                    f"Context compaction or session drop likely occurred."
                )
                recovered_info = {
                    "orphaned_sessions": recovered,
                    "warning": "Previous session(s) ended without pith_session_end. "
                    "Learning from those sessions may be incomplete.",
                }

        session_id = str(uuid.uuid4())[:8]
        now = _utc_now_iso()
        from app.core.surface_identity import normalize_surface_id, resolve_platform_hint

        resolved_surface_id = normalize_surface_id(surface_id)
        resolved_platform_hint = resolve_platform_hint(platform_hint, resolved_surface_id)

        self._conversation_turn_called = False  # S0: reset on new session
        self._last_orientation_served_at = None  # S6.1: reset so orientation re-serves
        # RAGAS-DIAG-001 Fix 3c: Store session_date for temporal anchoring in extraction
        self._session_date = session_date
        self._retro_checked_this_session = False  # RETRO-001: allow one check per session
        self._session_concept_ids = set()  # CONCEPT_LIFECYCLE_SPEC L4: reset per session

        session_info = SessionInfo(
            session_id=session_id,
            started_at=now,
            status="active",
            context_hint=context_hint,
            learning_event_count=0,
            agent_id=agent_id,
            platform_hint=resolved_platform_hint,
            surface_id=resolved_surface_id,
        )
        self.current_session = session_info
        self._remember_in_memory_session(session_info)

        # SESSION-012 v0.3: Store platform hint from session_start caller
        self._current_platform_hint = resolved_platform_hint
        self._current_surface_id = resolved_surface_id

        if persist_session:
            save_session(
                session_id=session_id,
                started_at=now,
                status="active",
                context_hint=context_hint,
                learning_event_count=0,
                agent_id=agent_id,
                model_id=getattr(self, "_current_model_id", "unknown"),
                platform_hint=resolved_platform_hint,
                surface_id=resolved_surface_id,
            )
        else:
            logger.info("SESSION-001: Session persistence skipped (PITH_BENCHMARK_READONLY)")

        # PERF-013: Populate git cache at session start
        try:
            from app.core.git_cache import GitCache

            self.git_cache = GitCache()
            self.git_cache.populate()
        except Exception as e:
            logger.warning(f"GitCache init failed (non-fatal): {e}")
            self.git_cache = None

        full_bootstrap = os.environ.get("PITH_SESSION_START_FULL_BOOTSTRAP") == "1"
        if full_bootstrap:
            # Legacy path: one disk scan, passed to both subsystems.
            concepts = self._load_all_concepts()
            introspect_data = self_model_manager.introspect(mode="summary", update=True, concepts=concepts)
            orientation = self.orient(concepts=concepts)
            bootstrap_mode = "full"
        else:
            concepts = []
            introspect_data = self_model_manager.introspect(
                mode="summary",
                update=False,
                concepts=None,
                generate_if_missing=False,
            )
            if isinstance(introspect_data, dict) and introspect_data.get("status") == "not_ready":
                orientation = PresentMomentOrientation(
                    generated_at=_utc_now_iso(),
                    generated_by="pith_lightweight_bootstrap",
                )
            else:
                orientation = self.orient(concepts=concepts)
            bootstrap_mode = "lightweight"

        # --- Checkpoint auto-load: surface recent execution state ---
        active_checkpoint = None
        try:
            from app.storage import load_checkpoint

            # If context_hint looks like a task_id, try loading that specific checkpoint
            cp = load_checkpoint(task_id=context_hint, max_age_hours=48) if context_hint else None
            if not cp:
                cp = load_checkpoint(max_age_hours=24)  # fallback to most recent
            if cp:
                active_checkpoint = {
                    "task_id": cp["task_id"],
                    "status": cp["status"],
                    "description": cp["description"],
                    "active": cp["active"],
                    "next": cp["next"],
                    "blockers": cp["blockers"],
                    "updated_at": cp["updated_at"],
                    "save_count": cp["save_count"],
                }
                logger.info(f"Checkpoint auto-loaded: {cp['task_id']} (status={cp['status']})")
        except Exception as e:
            logger.warning(f"Checkpoint auto-load failed (non-fatal): {e}")

        # --- GOV: Functional cognitive bootstrap ---
        # Loads high-authority constraints/decisions, surfaces stale alerts,
        # and retrieves governance actions since last session.
        bootstrap_data = None
        try:
            from app.session.bootstrap import build_bootstrap
            from app.storage import _db

            with _db() as conn:
                # Find last session end time for governance action tracking
                last_ended = None
                try:
                    row = conn.execute(
                        "SELECT ended_at FROM sessions WHERE status='ended' ORDER BY ended_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        last_ended = row[0]
                except Exception:
                    pass

                bootstrap_result = build_bootstrap(
                    conn=conn,
                    session_id=session_id,
                    is_resumption=False,
                    last_session_ended_at=last_ended,
                )
                bootstrap_data = bootstrap_result.to_dict()
                logger.info(
                    f"Bootstrap: {bootstrap_result.constraints_loaded} constraints, "
                    f"{bootstrap_result.decisions_loaded} decisions, "
                    f"{len(bootstrap_result.stale_alerts)} stale alerts"
                )
        except Exception as e:
            logger.warning(f"Bootstrap failed (non-fatal): {e}")

        persistence_mode = "persisted" if persist_session else "in-memory"
        logger.info(
            f"Session {session_id} started ({persistence_mode}, bootstrap={bootstrap_mode}): "
            f"{len(concepts)} concepts loaded, context='{context_hint[:50]}'"
        )

        current_session_id = getattr(self.current_session, "session_id", None)
        if current_session_id != session_id:
            logger.warning(
                "Session %s current_session changed during bootstrap (current=%s); "
                "returning persisted session_info and restoring manager state",
                session_id,
                current_session_id,
            )
            self.current_session = session_info

        response = SessionStartResponse(
            session=session_info,
            introspect_summary=introspect_data,
            orientation=orientation.model_dump(),
        )

        # Attach checkpoint and recovery info to response (outside Pydantic model)
        result = response.model_dump()
        if bootstrap_data:
            result["bootstrap"] = bootstrap_data
            if persist_session:
                # P3-4: Persist bootstrap marker to session record
                try:
                    import json as _json

                    from app.storage import update_session

                    session_data = _json.dumps(
                        {
                            "session_id": session_id,
                            "bootstrap": {
                                "constraints_loaded": bootstrap_result.constraints_loaded,
                                "decisions_loaded": bootstrap_result.decisions_loaded,
                                "stale_alerts": len(bootstrap_result.stale_alerts),
                            },
                        }
                    )
                    update_session(session_id, data=session_data)

                    # FED-013: Initialize heartbeat on session creation
                    try:
                        from app.features.federation import get_registry

                        get_registry().update_heartbeat(session_id)
                    except Exception:
                        pass  # Non-fatal — registry may not be enabled

                except Exception as e:
                    logger.warning(f"P3-4: Bootstrap persistence failed (non-fatal): {e}")
        if active_checkpoint:
            result["active_checkpoint"] = active_checkpoint
        if recovered_info:
            result["recovered_sessions"] = recovered_info
            # T1: Check orphaned sessions for retroactive reflection
            # REFLECT-021: T1 gated in auto_reflection.check_orphaned_sessions_for_reflection
            try:
                from app.cognitive.auto_reflection import (
                    check_orphaned_sessions_for_reflection,
                    mark_session_reflected,
                    record_reflection_event,
                )

                # Query orphaned sessions from DB
                from app.storage import _db

                with _db() as conn:
                    orphan_rows = conn.execute(
                        """SELECT id, started_at, ended_at, status,
                                  learning_event_count, data
                           FROM sessions
                           WHERE status IN ('interrupted', 'recovered')
                           ORDER BY ended_at DESC LIMIT 5"""
                    ).fetchall()
                orphan_sessions = [
                    {
                        "id": r[0],
                        "started_at": r[1],
                        "ended_at": r[2],
                        "status": r[3],
                        "learning_event_count": r[4],
                        "data": r[5],
                    }
                    for r in orphan_rows
                ]
                retro = check_orphaned_sessions_for_reflection(orphan_sessions)
                if retro:
                    result["retroactive_reflection"] = retro
                    record_reflection_event(
                        session_id=session_id,
                        trigger_type="T1_retroactive",
                        prompts_sent=len(retro.get("prompts", [])),
                        prompt_data=retro.get("prompts"),
                    )
                    # Mark the orphaned session so T1 doesn't fire again
                    mark_session_reflected(retro["orphaned_session_id"])
                    logger.info("T1 retroactive reflection attached to session_start response")
            except Exception as e:
                logger.warning(f"T1 retroactive reflection failed (non-fatal): {e}")

        # --- [dropout-recovery] C2b: Startup scan for dropout-missed sessions ---
        # Catches sessions that ended between server restarts (missed by C2 at end_session).
        # TTL-gated: only sessions ended within last 24h (A3). LIMIT 10 caps startup cost (A3).
        try:
            from app.core.config import get_feature_flag as _c2b_flag
            if _c2b_flag("PITH_SESSION_END_AUTOLEARN_ENABLED", True):
                from app.storage import _db as _c2b_db
                with _c2b_db() as _c2b_conn:
                    _dropout_rows = _c2b_conn.execute(
                        """SELECT id, last_previous_response
                           FROM sessions
                           WHERE status = 'ended'
                             AND learning_event_count = 0
                             AND last_previous_response IS NOT NULL
                             AND ended_at >= datetime('now', '-24 hours')
                           ORDER BY ended_at DESC
                           LIMIT 10"""
                    ).fetchall()
                for _d_id, _d_resp in _dropout_rows:
                    if not _d_resp or len(_d_resp) < 30:
                        continue
                    logger.info(
                        f"[dropout-recovery] C2b: replaying missed session {_d_id}, "
                        f"stored_len={len(_d_resp)}"
                    )
                    try:
                        _c2b_req = SessionLearnRequest(
                            user_message="",
                            assistant_response=_d_resp,
                            knowledge_area="conversation",
                            extracted_concepts=None,
                            session_id=_d_id,
                        )
                        _c2b_result = self.session_learn(_c2b_req)
                        logger.info(
                            f"[dropout-recovery] C2b: session {_d_id} captured "
                            f"{_c2b_result.learning_events} events"
                        )
                        # Clear stored response after confirmed dispatch
                        update_session(_d_id, last_previous_response=None)
                    except Exception as _c2b_item_err:
                        logger.warning(
                            f"[dropout-recovery] C2b: replay failed for {_d_id} "
                            f"(non-fatal): {_c2b_item_err}"
                        )
        except Exception as _c2b_outer_err:
            logger.warning(f"[dropout-recovery] C2b startup scan failed (non-fatal): {_c2b_outer_err}")

        result["previous_session_ended"] = recovered_info is None

        # --- RC-A: Cleanup expired resume snapshots (piggybacking on session start) ---
        if persist_session:
            try:
                cleaned = cleanup_expired_snapshots()
                if cleaned:
                    logger.info(f"RC-A: Cleaned {cleaned} expired resume snapshot(s)")
            except Exception as e:
                logger.warning(f"RC-A: Snapshot cleanup failed (non-fatal): {e}")

        return result

    def _resolve_session_end_binding(
        self,
        end_request: SessionEndRequest | None,
    ) -> tuple[SessionInfo | None, dict[str, Any]]:
        requested_session_id = (getattr(end_request, "session_id", None) or "").strip() or None
        requested_origin_id = getattr(end_request, "origin_id", None)

        if requested_session_id:
            row = load_session(requested_session_id)
            if row is None:
                session = self._in_memory_session(requested_session_id)
                if session is not None:
                    return session, {
                        "bind_status": "bound",
                        "binding_source": "explicit_in_memory_session_id",
                    }
                return None, {
                    "status": "session_not_found",
                    "bind_status": "unbound",
                    "binding_source": "explicit_session_id",
                    "session_id": requested_session_id,
                }
            session = self._session_info_from_row(row)
            if session is None:
                return None, {
                    "status": "session_not_found",
                    "bind_status": "unbound",
                    "binding_source": "explicit_session_id",
                    "session_id": requested_session_id,
                }
            if session.status != "active":
                return None, {
                    "status": "session_not_active",
                    "bind_status": "unbound",
                    "binding_source": "explicit_session_id",
                    "session_id": session.session_id,
                    "session_status": session.status,
                }
            return session, {"bind_status": "bound", "binding_source": "explicit_session_id"}

        if requested_origin_id:
            rows = load_active_sessions_by_origin(requested_origin_id)
            if not rows:
                return None, {
                    "status": "session_not_found",
                    "bind_status": "unbound",
                    "binding_source": "origin_id",
                    "origin_id": requested_origin_id,
                }
            if len(rows) > 1:
                return None, {
                    "status": "ambiguous_origin",
                    "bind_status": "unbound",
                    "binding_source": "origin_id",
                    "origin_id": requested_origin_id,
                    "candidate_count": len(rows),
                }
            session = self._session_info_from_row(rows[0])
            if session is None:
                return None, {
                    "status": "session_not_found",
                    "bind_status": "unbound",
                    "binding_source": "origin_id",
                    "origin_id": requested_origin_id,
                }
            return session, {"bind_status": "bound", "binding_source": "origin_id"}

        if not self.current_session:
            return None, {
                "status": "no_active_session",
                "bind_status": "unbound",
                "binding_source": "no_request_or_memory_session",
            }
        return self.current_session, {"bind_status": "bound", "binding_source": "in_memory_active"}

    def end_session(self, end_request: SessionEndRequest | None = None) -> dict:
        """Resolve closeout binding, then end the selected session."""
        session, binding = self._resolve_session_end_binding(end_request)
        if session is None:
            return binding

        token_pair = None
        if binding.get("binding_source") in {"explicit_session_id", "origin_id"}:
            token_pair = self._push_request_session(session)

        try:
            result = self._end_bound_session(end_request)
        finally:
            if token_pair:
                self._pop_request_session(*token_pair)

        if isinstance(result, dict):
            result.setdefault("bind_status", binding.get("bind_status"))
            result.setdefault("binding_source", binding.get("binding_source"))
            if getattr(session, "origin_id", None):
                result.setdefault("origin_id", session.origin_id)
        return result

    def _end_bound_session(self, end_request: SessionEndRequest | None = None) -> dict:
        """End current session. Optionally flush last exchange before closing.
        Flush access tracker, trigger reflection if learning_event_count >= threshold.
        Persists final state to SQLite."""
        if not self.current_session:
            return {"status": "no_active_session"}

        # --- C1: Last-exchange flush (Mechanism C) ---
        last_learn_result = None
        raw_capture_ref = None
        if end_request and end_request.previous_response and len(end_request.previous_response) >= 30:
            try:
                try:
                    from app.storage.turn_ingestion import (
                        capture_raw_turn_default_db,
                        raw_capture_enabled,
                        raw_capture_retention_days,
                    )

                    if raw_capture_enabled():
                        raw_session_id = self.current_session.session_id if self.current_session else "unknown"
                        raw_turn_id = f"{raw_session_id}:session_end:{int(time.time() * 1000)}"
                        capture_raw_turn_default_db(
                            session_id=raw_session_id,
                            turn_id=raw_turn_id,
                            source="session_end",
                            user_message=end_request.previous_message,
                            assistant_response=end_request.previous_response,
                            retention_days=raw_capture_retention_days(),
                        )
                        raw_capture_ref = {
                            "session_id": raw_session_id,
                            "turn_id": raw_turn_id,
                            "source": "session_end",
                        }
                except Exception as _capture_err:
                    try:
                        from app.ops.metrics import metrics as _capture_metrics

                        _capture_metrics.record("raw_turn_capture_failed", 1.0, {"source": "session_end"})
                    except Exception:
                        pass
                    logger.warning("session_end_raw_capture_failed: %s", _capture_err)

                # Parse Tier 2 concepts
                extracted = None
                if end_request.extracted_concepts_json:
                    try:
                        parsed = json.loads(end_request.extracted_concepts_json)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            extracted = parsed
                    except json.JSONDecodeError:
                        pass

                learn_req = SessionLearnRequest(
                    user_message=end_request.previous_message or "",
                    assistant_response=end_request.previous_response[:15000],
                    knowledge_area="conversation",
                    extracted_concepts=extracted,
                    session_id=self.current_session.session_id if self.current_session else None,
                )
                last_learn_result = self.session_learn(learn_req)
                if raw_capture_ref:
                    try:
                        from app.storage.turn_ingestion import mark_learning_status_default_db

                        mark_learning_status_default_db(
                            **raw_capture_ref,
                            status="attempted",
                            concepts_extracted=last_learn_result.learning_events if last_learn_result else 0,
                        )
                    except Exception as _ledger_err:
                        logger.warning("turn_ingestion_ledger_update_failed: %s", _ledger_err)
                logger.info(
                    f"Session end flush: {last_learn_result.learning_events} events, "
                    f"sources={last_learn_result.extraction_source_breakdown}"
                )
            except Exception as e:
                if raw_capture_ref:
                    try:
                        from app.storage.turn_ingestion import mark_learning_status_default_db

                        mark_learning_status_default_db(**raw_capture_ref, status="failed", error=str(e))
                    except Exception as _ledger_err:
                        logger.warning("turn_ingestion_ledger_update_failed: %s", _ledger_err)
                logger.warning(f"Session end flush failed (non-fatal): {e}")

        # --- [dropout-recovery] C2: Flush stored last_previous_response on auto-end ---
        # Fires when caller passed no previous_response (auto-end path delivers end_request=None),
        # session has a stored response (C1 wrote it on the last turn), and no learning has
        # occurred this session (natural guard — prevents double-dispatch per A2 amendment).
        _caller_has_response = (
            end_request
            and end_request.previous_response
            and len(end_request.previous_response) >= 30
        )
        from app.core.config import get_feature_flag as _c2_get_flag
        if (
            not _caller_has_response
            and self.current_session.learning_event_count == 0
            and _c2_get_flag("PITH_SESSION_END_AUTOLEARN_ENABLED", True)
        ):
            try:
                from app.storage import _db as _storage_db
                with _storage_db() as _conn:
                    _row = _conn.execute(
                        "SELECT last_previous_response FROM sessions WHERE id = ?",
                        (self.current_session.session_id,),
                    ).fetchone()
                _stored_response = _row[0] if _row else None
                if _stored_response and len(_stored_response) >= 30:
                    logger.info(
                        f"[dropout-recovery] C2: dispatching auto-learn for session "
                        f"{self.current_session.session_id}, stored_len={len(_stored_response)}"
                    )
                    _c2_learn_req = SessionLearnRequest(
                        user_message="",
                        assistant_response=_stored_response,
                        knowledge_area="conversation",
                        extracted_concepts=None,
                        session_id=self.current_session.session_id,
                    )
                    last_learn_result = self.session_learn(_c2_learn_req)
                    logger.info(
                        f"[dropout-recovery] C2: captured {last_learn_result.learning_events} events"
                    )
                    # A2 amendment: clear stored response after confirmed dispatch (prevent double-dispatch)
                    update_session(
                        self.current_session.session_id,
                        last_previous_response=None,
                    )
            except Exception as _c2_err:
                logger.warning(f"[dropout-recovery] C2 flush failed (non-fatal): {_c2_err}")

        self.current_session.ended_at = _utc_now_iso()
        self.current_session.status = "ended"

        # Persist to SQLite
        update_session(
            self.current_session.session_id,
            ended_at=self.current_session.ended_at,
            status="ended",
            learning_event_count=self.current_session.learning_event_count,
        )

        # Flush access tracker
        from app.storage import access_tracker

        flushed = access_tracker.flush()

        result = {
            "status": "ended",
            "session_id": self.current_session.session_id,
            "duration_seconds": self._session_duration(),
            "learning_events": self.current_session.learning_event_count,
            "access_records_flushed": flushed,
            "reflection_triggered": False,
            "last_exchange_flushed": last_learn_result is not None and last_learn_result.learning_events > 0,
        }

        # ARCH-002: Capture session state before clearing.
        _session_copy = self.current_session
        _concept_ids_copy = set(self._session_concept_ids) if self._session_concept_ids else set()
        _duration = self._session_duration()
        self._forget_in_memory_session(_session_copy.session_id if _session_copy else None)
        self.current_session = None

        # SUPPRESS-EMPTY-SESSIONS: Skip expensive end-of-session processing for 0-event sessions.
        # C2 dropout-recovery already ran above. Check current state (gauntlet B6: use _session_copy
        # which was captured AFTER C2 could have fired session_learn → record_learning_event).
        if _session_copy.learning_event_count == 0:
            logger.info(
                "SUPPRESS-EMPTY-SESSIONS: Skipping heavy session-end processing for %s "
                "(0 learning events after C2 recovery attempt)",
                _session_copy.session_id,
            )
            result["suppressed_empty_session"] = True
            logger.info(
                f"Session {_session_copy.session_id} ended: "
                f"0 learning events (empty session, heavy processing skipped)"
            )
            return result

        if BenchmarkIngestionMode.from_env().enabled:
            logger.info(
                "BENCHMARK-MODE: Skipping heavy session-end processing for %s "
                "(%s learning events)",
                _session_copy.session_id,
                _session_copy.learning_event_count,
            )
            result["suppressed_benchmark_mode"] = True
            result["heavy_tasks"] = "suppressed_benchmark_mode"
            logger.info(
                f"Session {_session_copy.session_id} ended: "
                f"{_session_copy.learning_event_count} learning events, "
                "heavy=suppressed_benchmark_mode"
            )
            return result

        # ARCH-002: Dispatch heavy tasks as background work when event loop available.
        # Heavy phase: reflection, T3, staleness, threads, currency, checkpoint, scheduled tasks.
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():

                async def _bg():
                    try:
                        self._run_end_session_heavy(_session_copy, _concept_ids_copy, _duration)
                    except Exception as e:
                        logger.warning(f"ARCH-002: Background end_session failed: {e}")

                task = loop.create_task(_bg(), name="end_session_heavy")
                self._background_tasks.add(task)
                task.add_done_callback(self._on_bg_task_done)
                result["heavy_tasks"] = "scheduled_background"
            else:
                raise RuntimeError("no loop")
        except RuntimeError:
            # Sync fallback (CLI, tests)
            self._run_end_session_heavy(_session_copy, _concept_ids_copy, _duration)

        logger.info(
            f"Session {_session_copy.session_id} ended: "
            f"{_session_copy.learning_event_count} learning events, "
            f"heavy={result.get('heavy_tasks', 'sync')}"
        )

        return result

    def _run_end_session_heavy(self, session_copy, session_concept_ids: set, session_duration: float = 0):
        """ARCH-002: Heavy end-session tasks — runs in background when event loop available.

        Includes: reflection, T3 prompts, staleness reconciliation, thread staleness,
        currency refresh, checkpoint auto-save, and scheduled async tasks.
        All operations are best-effort (non-fatal on failure).
        """
        result = {}  # Local result dict for logging only

        # C3: Trigger reflection if enough learning AND minimum duration met
        if session_copy.learning_event_count >= REFLECTION_TRIGGER_THRESHOLD and session_duration >= 300:
            try:
                from app.cognitive.reflection import reflection_engine

                reflection_engine.reflect(mode="incremental")
                result["reflection_triggered"] = True
                logger.info(f"Session end triggered reflection: {session_copy.learning_event_count} learning events")
            except Exception as e:
                logger.error(f"Session-end reflection failed: {e}")
                result["reflection_error"] = str(e)

        # --- T3: Full session-end reflection ---
        # Generate targeted reflection prompts for L1→L3 synthesis
        if session_copy:
            try:
                from app.cognitive.auto_reflection import (
                    _find_session_concepts,
                    generate_session_end_reflection,
                    record_reflection_event,
                )

                session_concept_ids = _find_session_concepts(session_copy.session_id)
                t3_reflection = generate_session_end_reflection(
                    session_concept_ids=session_concept_ids,
                    learning_event_count=session_copy.learning_event_count,
                    session_duration_seconds=session_duration,
                    unprocessed_bookmarks=session_copy.reflection_bookmarks,
                )
                if t3_reflection:
                    result["reflection_prompts"] = t3_reflection
                    result["reflection_required"] = True
                    record_reflection_event(
                        session_id=session_copy.session_id,
                        trigger_type="T3_session_end",
                        prompts_sent=len(t3_reflection.get("prompts", [])),
                        prompt_data=t3_reflection.get("prompts"),
                    )
                    logger.info(f"T3 session-end reflection: {len(t3_reflection.get('prompts', []))} prompts generated")
            except Exception as e:
                logger.warning(f"T3 session-end reflection failed (non-fatal): {e}")

        # --- Trigger 2: Staleness checkpoint reconciliation ---
        # Cross-reference checkpoint done[] items against existing concepts
        # to evolve any that are stale relative to checkpoint progress.
        if self.current_session and session_copy.learning_event_count > 0:
            try:
                from app.retrieval import retrieval_engine
                from app.cognitive.staleness import reconcile_checkpoint_concepts

                staleness_result = reconcile_checkpoint_concepts(
                    retrieval_engine=retrieval_engine,
                    supersede_fn=self._supersede_concept,
                )
                if staleness_result.concepts_staled > 0:
                    result["staleness_t2"] = {
                        "evolved": staleness_result.concepts_staled,
                        "details": staleness_result.details,
                        "time_ms": staleness_result.time_ms,
                    }
                    logger.info(
                        f"Staleness T2: Evolved {staleness_result.concepts_staled} stale concepts at session end"
                    )
            except Exception as e:
                logger.warning(f"Staleness T2 reconciliation failed (non-fatal): {e}")

        # --- Trigger 2b: Session-scoped concept reconciliation ---
        # CONCEPT_LIFECYCLE_SPEC L2: Detect in-session status transitions
        # (planned→committed, proposed→implemented) without checkpoint dependency.
        if self.current_session and session_copy.started_at:
            try:
                from app.retrieval import retrieval_engine
                from app.cognitive.staleness import reconcile_session_concepts

                t2b_result = reconcile_session_concepts(
                    session_start_iso=session_copy.started_at,
                    retrieval_engine=retrieval_engine,
                    supersede_fn=self._supersede_concept,
                )
                if t2b_result.concepts_staled > 0:
                    result["staleness_t2b"] = {
                        "superseded": t2b_result.concepts_staled,
                        "details": t2b_result.details,
                        "time_ms": t2b_result.time_ms,
                    }
                    logger.info(
                        f"Staleness T2b: Superseded {t2b_result.concepts_staled} "
                        f"stale concepts via session reconciliation"
                    )
            except Exception as e:
                logger.warning(f"Staleness T2b reconciliation failed (non-fatal): {e}")

        # --- Thread staleness detection (Wave 5) ---
        try:
            from app.features.threads import detect_stale_threads

            thread_actions = detect_stale_threads()
            if thread_actions:
                result["thread_staleness"] = thread_actions
                logger.info(f"Thread staleness: {len(thread_actions)} actions taken")
        except Exception as e:
            logger.debug(f"Thread staleness detection skipped: {e}")

        # --- RESOLVE-CONTRADICTIONS: Drain contradiction signal backlog at session end ---
        # consume_graph_contradiction_signals() exists in contradiction.py but was
        # only wired to reflection. Running here ensures steady-state drainage
        # regardless of whether reflection triggered. batch_size=50 keeps latency bounded.
        try:
            from app.cognitive.contradiction import consume_graph_contradiction_signals
            _contra_drain = consume_graph_contradiction_signals(batch_size=50)
            if _contra_drain.get("newly_resolved", 0) > 0:
                result["contradiction_drainage"] = {
                    "newly_resolved": _contra_drain["newly_resolved"],
                    "remaining": _contra_drain["total_events"] - _contra_drain["newly_resolved"],
                }
                logger.info(
                    "RESOLVE-CONTRADICTIONS: Drained %d contradiction signals at session end "
                    "(remaining: %d)",
                    _contra_drain["newly_resolved"],
                    _contra_drain["total_events"] - _contra_drain["newly_resolved"],
                )
        except Exception as e:
            logger.warning("RESOLVE-CONTRADICTIONS: Contradiction drainage failed (non-fatal): %s", e)

        # --- CONCEPT_LIFECYCLE_SPEC L4: Session-end currency refresh ---
        # Refresh currency_status for session-created concepts so the NEXT
        # session's orientation has fresh data.
        if session_concept_ids:
            try:
                from app.governance.currency import batch_compute_currency
                from app.storage import _db

                with _db() as conn:
                    updated = batch_compute_currency(conn, list(session_concept_ids))
                    if updated > 0:
                        logger.info(
                            f"LIFECYCLE L4: Session-end currency refresh — "
                            f"{updated}/{len(session_concept_ids)} concepts"
                        )
            except Exception as e:
                logger.warning(f"LIFECYCLE L4: Session-end currency refresh failed: {e}")

        # --- CKPT-001: Checkpoint lifecycle management on session end ---
        try:
            from app.storage import archive_stale_checkpoints, load_checkpoint, save_checkpoint

            # Phase 1: Archive stale checkpoints (>48h no update, not from this session)
            current_sid = session_copy.session_id if session_copy else None
            archived_count = archive_stale_checkpoints(exclude_session_id=current_sid)
            if archived_count > 0:
                result["checkpoints_archived"] = archived_count
                logger.info(f"CKPT-001: Archived {archived_count} stale checkpoint(s)")

            # Phase 2: Auto-save current session's checkpoint as paused
            # NOTE: Auto-COMPLETE is handled by staleness.py:587-611 (reconcile_checkpoint_concepts)
            # which runs separately in the heavy phase with stricter guards (save_count>=2, done non-empty).
            # We only handle active→paused here.
            if session_copy and session_copy.learning_event_count > 0:
                cp = load_checkpoint(max_age_hours=24)
                if cp and cp["status"] in ("active", "planning"):
                    # CKPT-002: Compress before saving as paused
                    from app.storage import compress_checkpoint
                    compressed = compress_checkpoint(cp)
                    save_checkpoint(
                        task_id=cp["task_id"],
                        description=cp["description"],
                        status="paused",
                        done=compressed.get("done", cp["done"]),
                        active=compressed.get("active", cp["active"]),
                        next_items=compressed.get("next", cp["next"]),
                        blockers=compressed.get("blockers", cp.get("blockers")),
                        context=compressed.get("context", cp.get("context")),
                        session_id=current_sid,
                    )
                    result["checkpoint_auto_saved"] = cp["task_id"]
                    logger.info(f"CKPT-001: Checkpoint {cp['task_id']} → paused")
        except Exception as e:
            logger.warning(f"CKPT-001: Checkpoint lifecycle failed (non-fatal): {e}")

        # KA-005: Run scheduled async tasks (including ka_reclassification) on session end.
        # Previously only called via /maintenance endpoint, meaning ka_reclassification
        # never ran automatically despite having a TASK_CONFIGS entry.
        # STABILITY-013: Fire-and-forget — do NOT block event loop.
        if session_copy and result.get("reflection_triggered"):
            try:
                import asyncio

                from app.ops.async_tasks import task_runner

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    # STABILITY-013: Fire-and-forget background task.
                    # Session-end tasks are best-effort; results not needed for response.
                    # CancelledError during shutdown is expected — finally ensures cleanup.
                    async def _run_session_end_tasks_bg():
                        try:
                            bg_results = await task_runner.run_scheduled_tasks()
                            if bg_results:
                                logger.info(f"Background session-end tasks completed: {list(bg_results.keys())}")
                        except asyncio.CancelledError:
                            logger.info("Background session-end tasks cancelled")
                            raise
                        except Exception as e:
                            logger.warning(f"Background session-end tasks failed: {e}")

                    task = loop.create_task(_run_session_end_tasks_bg(), name="session_end_scheduled_tasks")
                    # CRITICAL: Store strong reference to prevent GC (Gauntlet A1)
                    self._background_tasks.add(task)
                    task.add_done_callback(self._on_bg_task_done)
                    result["scheduled_tasks"] = {"status": "scheduled_background"}
                else:
                    sched_results = asyncio.run(task_runner.run_scheduled_tasks())
                    if sched_results:
                        result["scheduled_tasks"] = {
                            k: v.get("status", "unknown") for k, v in sched_results.items()
                        }
                        logger.info(f"Session-end scheduled tasks: {list(sched_results.keys())}")
            except Exception as e:
                logger.warning(f"Session-end scheduled tasks failed (non-fatal): {e}")

    def record_learning_event(self):
        """Increment learning event counter. Called by propose/evolve endpoints.
        Persists updated count to SQLite."""
        if self.current_session:
            self.current_session.learning_event_count += 1
            update_session(
                self.current_session.session_id,
                learning_event_count=self.current_session.learning_event_count,
            )

    def register_implicit_learning_event(
        self, event_type: str, concept_id: str, summary: str, source: str = "implicit"
    ):
        """C3: Register a learning event from propose/evolve/link operations.

        Appends session metadata (not a storage write). Only fires if a session
        is active. Records the event and increments the learning event counter.

        Args:
            event_type: concept_proposed | concept_evolved | concepts_linked
            concept_id: The concept ID involved
            summary: Brief description (truncated to 200 chars)
            source: Event source identifier
        """
        if not self.current_session:
            return

        event = {
            "type": event_type,
            "concept_id": concept_id,
            "summary": summary[:200],
            "source": source,
            "timestamp": _utc_now_iso(),
        }

        # Store on session metadata (in-memory list)
        if not hasattr(self.current_session, "_implicit_events"):
            self.current_session._implicit_events = []
        self.current_session._implicit_events.append(event)

        # Count toward reflection threshold
        self.record_learning_event()
        logger.debug(f"C3: Implicit learning event: {event_type} {concept_id}")
