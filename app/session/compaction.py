"""Compaction detection and working-context reconstruction.

Extracted from session/turn.py per ARCH-009b (Modularity Plan v2 Item 2a step 4).
"""

import logging
import time
from datetime import UTC, datetime

from app.core.datetime_utils import _ensure_aware, _utc_now
from app.core.models import (
    ActivatedConcept,
    ConversationTurnRequest,
)
from app.storage import (
    load_concept,
    load_resume_snapshot,
)

logger = logging.getLogger(__name__)


class CompactionMixin:
    """Compaction detection and working-context block building.

    Methods:
        _build_context_priority_hints
        _detect_compaction
        _build_working_context_block
        _handle_compaction_reinjection
    """

    def _build_context_priority_hints(
        self,
        activated: list[ActivatedConcept],
        aa_ids: set[str] | None = None,
    ) -> dict | None:
        """CTX Phase 1: Build priority hints for activated concepts.

        Classifies each activated concept as critical/high/normal/low
        based on concept_type, always-activate status, and governance scores.
        Returns a dict with critical_ids, ephemeral_ids, ttl_seconds, and
        total_critical_tokens estimate.

        Budget: ~2-5ms (loads concept_type per concept via load_concept).
        """
        from app.core.config import (
            CTX_TTL_ACTIVATED,
            CTX_TTL_CONSTRAINT,
            CTX_TTL_DECISION,
            CTX_TTL_FIRMWARE,
        )

        if not activated:
            return None

        aa_set = aa_ids or set()
        critical_ids = []
        ephemeral_ids = []
        ttl_seconds = {}
        total_critical_tokens = 0

        for ac in activated:
            cid = ac.concept_id

            # Always-activate / firmware → CRITICAL, no TTL
            if cid in aa_set:
                critical_ids.append(cid)
                ttl_seconds[cid] = CTX_TTL_FIRMWARE
                total_critical_tokens += 50  # ~50 tokens per firmware concept
                continue

            # Load concept type from knowledge for classification
            concept = None
            try:  # noqa: SIM105
                concept = load_concept(cid, track_access=False)
            except Exception:
                pass

            ctype = concept.concept_type if concept else "observation"

            if ctype in ("constraint", "firmware"):
                critical_ids.append(cid)
                ttl_seconds[cid] = CTX_TTL_CONSTRAINT
                total_critical_tokens += 50
            elif ctype == "decision":
                ttl_seconds[cid] = CTX_TTL_DECISION
                # Decisions with high authority are high-priority
                if concept and getattr(concept, "authority_score", 0) and concept.authority_score >= 0.70:
                    critical_ids.append(cid)
                    total_critical_tokens += 50
            elif ctype in ("principle", "method", "heuristic", "cognitive_strategy"):
                ttl_seconds[cid] = CTX_TTL_DECISION  # Same TTL as decisions
            elif ctype in ("observation", "pattern"):
                ttl_seconds[cid] = CTX_TTL_ACTIVATED
                ephemeral_ids.append(cid)
            else:
                ttl_seconds[cid] = CTX_TTL_ACTIVATED
                ephemeral_ids.append(cid)

        if not critical_ids and not ephemeral_ids:
            return None

        return {
            "critical_ids": critical_ids,
            "ephemeral_ids": ephemeral_ids,
            "ttl_seconds": ttl_seconds,
            "total_critical_tokens": total_critical_tokens,
        }

    def _detect_compaction(self, request: ConversationTurnRequest) -> bool:
        """Detect likely context compaction event. Budget: <1ms.

        Uses heuristic signals with a two-signal rule (same pattern as
        CCL correction detection). Returns True if 2+ signals fire.

        Guards:
        - Skip on is_first_call (CTX-3: Resume Context handles that)
        - Cooldown: max 1 detection per COMPACTION_COOLDOWN_SECONDS (CTX-2)
        - Session circuit breaker: disable after COMPACTION_FALSE_POSITIVE_LIMIT (CTX-2)
        """
        from app.core.config import (
            COMPACTION_AMNESIA_MIN_LENGTH,
            COMPACTION_CONTEXT_AMNESIA_MIN_TURNS,
            COMPACTION_COOLDOWN_SECONDS,
            COMPACTION_EMPTY_EXTRACTIONS_THRESHOLD,
            COMPACTION_FALSE_POSITIVE_LIMIT,
            COMPACTION_MIN_TURNS_FOR_DETECTION,
            COMPACTION_SIGNALS_REQUIRED,
            COMPACTION_TEMPORAL_GAP_SECONDS,
        )

        # Explicit client signal (Phase 4 future-proofing)
        if getattr(request, "compaction_detected", None) is True:
            self._last_compaction_detected_at = time.perf_counter()
            return True

        # --- SESSION-010: Path 1 — Cross-session DB detection (first call only) ---
        # After compaction, MCP restarts → fresh PithBrain → _conversation_turn_called=False.
        # In-memory state is gone, so we query the DB for a recently-interrupted predecessor.
        _is_first = not self._conversation_turn_called
        if _is_first:
            try:
                from app.core.config import (
                    COMPACTION_MIN_PREDECESSOR_EVENTS,
                    COMPACTION_PROXIMITY_SECONDS,
                )
                from app.storage import _db

                _current_model = getattr(self, "_current_model_id", "unknown")
                _current_session_id = (
                    self.current_session.session_id if self.current_session else None
                )
                if _current_session_id:
                    with _db() as conn:
                        # Tier 1: Interrupted predecessor with same model (HIGH confidence)
                        row = conn.execute(
                            """SELECT id, learning_event_count, pressure_score, model_id,
                                      ended_at, status
                               FROM sessions
                               WHERE status = 'interrupted'
                                 AND id != ?
                                 AND learning_event_count >= ?
                                 AND (julianday('now') - julianday(ended_at)) * 86400 < ?
                                 AND model_id = ?
                               ORDER BY ended_at DESC
                               LIMIT 1""",
                            (
                                _current_session_id,
                                COMPACTION_MIN_PREDECESSOR_EVENTS,
                                COMPACTION_PROXIMITY_SECONDS,
                                _current_model,
                            ),
                        ).fetchone()
                        _detection_tier = None
                        if row:
                            _detection_tier = "HIGH"
                        else:
                            # Tier 2: Ended predecessor with same model, no active sibling
                            row = conn.execute(
                                """SELECT id, learning_event_count, pressure_score, model_id,
                                          ended_at, status
                                   FROM sessions
                                   WHERE status = 'ended'
                                     AND id != ?
                                     AND learning_event_count >= ?
                                     AND (julianday('now') - julianday(ended_at)) * 86400 < ?
                                     AND model_id = ?
                                   ORDER BY ended_at DESC
                                   LIMIT 1""",
                                (
                                    _current_session_id,
                                    COMPACTION_MIN_PREDECESSOR_EVENTS,
                                    COMPACTION_PROXIMITY_SECONDS,
                                    _current_model,
                                ),
                            ).fetchone()
                            if row:
                                sibling = conn.execute(
                                    """SELECT id FROM sessions
                                       WHERE status = 'active'
                                         AND id != ?
                                         AND model_id = ?
                                       LIMIT 1""",
                                    (_current_session_id, _current_model),
                                ).fetchone()
                                if sibling:
                                    row = None  # Active sibling exists — suppress
                                else:
                                    _detection_tier = "MEDIUM"

                        if row and _detection_tier:
                            predecessor_id = row[0]
                            logger.info(
                                f"CTX S-0.5: SESSION-010 cross-session compaction detected — "
                                f"tier={_detection_tier}, predecessor={predecessor_id}, "
                                f"predecessor_events={row[1]}, model={_current_model}"
                            )
                            self._compaction_predecessor_id = predecessor_id
                            self._compaction_detection_tier = _detection_tier
                            self._last_compaction_detected_at = time.perf_counter()
                            return True
            except Exception as e:
                logger.warning(f"SESSION-010: Cross-session detection failed (non-fatal): {e}")

        # --- Path 2: Original in-memory heuristic detection (subsequent calls) ---

        # Guard: no session or no prior turns
        if not self.current_session or not self._last_conversation_turn_at:
            return False

        turn_count = self.current_session.learning_event_count

        # Guard: CTX-3 — skip on first call (in-memory signals are empty)
        if not self._conversation_turn_called:
            return False

        # Guard: too few turns for meaningful detection
        if turn_count < COMPACTION_MIN_TURNS_FOR_DETECTION:
            return False

        # Guard: CTX-2 — session circuit breaker (too many false positives)
        if self._compaction_false_positive_count >= COMPACTION_FALSE_POSITIVE_LIMIT:
            return False

        # Guard: CTX-2 — cooldown (max 1 detection per interval)
        now = time.perf_counter()
        if (
            self._last_compaction_detected_at is not None
            and (now - self._last_compaction_detected_at) < COMPACTION_COOLDOWN_SECONDS
        ):
            return False

        # --- Track consecutive empty extractions ---
        if request.extracted_concepts_json in (None, "", "[]"):
            self._consecutive_empty_extractions += 1
        else:
            self._consecutive_empty_extractions = 0

        # --- Signal 1: Temporal gap after active session ---
        gap = now - self._last_conversation_turn_at
        temporal_gap = gap > COMPACTION_TEMPORAL_GAP_SECONDS and turn_count >= COMPACTION_MIN_TURNS_FOR_DETECTION

        # --- Signal 2: Missing previous_response after established session ---
        prev_resp = request.previous_response or ""
        context_amnesia = (
            turn_count >= COMPACTION_CONTEXT_AMNESIA_MIN_TURNS and len(prev_resp) < COMPACTION_AMNESIA_MIN_LENGTH
        )

        # --- Signal 3: Consecutive empty extractions ---
        behavioral_regression = (
            turn_count >= COMPACTION_CONTEXT_AMNESIA_MIN_TURNS
            and self._consecutive_empty_extractions >= COMPACTION_EMPTY_EXTRACTIONS_THRESHOLD
        )

        # --- Two-signal rule ---
        signals = [temporal_gap, context_amnesia, behavioral_regression]
        signal_names = ["temporal_gap", "context_amnesia", "behavioral_regression"]
        fired = [n for n, s in zip(signal_names, signals, strict=False) if s]

        if len(fired) >= COMPACTION_SIGNALS_REQUIRED:
            logger.info(
                f"CTX S-0.5: Compaction detected — signals fired: {fired}, "
                f"gap={gap:.0f}s, prev_resp_len={len(prev_resp)}, "
                f"consecutive_empty={self._consecutive_empty_extractions}"
            )
            self._last_compaction_detected_at = now
            self._consecutive_empty_extractions = 0  # Reset after detection
            return True

        return False


    def _build_working_context_block(self, request) -> dict | None:
        """CONTEXT-001: Build structured working_context returned every turn.

        5 priority layers with 400-token budget:
        L1: Checkpoint state (highest priority — never trimmed)
        L2: Active task + domain
        L3: Session metadata
        L4: Tools used
        L5: Pinned concepts (lowest priority — trimmed first)

        Gated behind WORKING_CONTEXT_ENABLED feature flag.
        """
        import json as _wc_json

        from app.core.config import FEATURE_FLAGS as _wc_ff

        if not _wc_ff.get("WORKING_CONTEXT_ENABLED", False):
            return None

        from app.core.config import WORKING_CONTEXT_MAX_TOKENS

        wc: dict = {}

        # Hoisted checkpoint load — shared by L1, L2, L4
        _cp = None
        try:
            from app.storage import load_checkpoint

            session_id = None
            if self.current_session:
                session_id = self.current_session.session_id
            _cp = load_checkpoint(
                task_id=getattr(request, "current_task_id", None),
                origin_id=getattr(request, "origin_id", None),
                session_id=session_id,
            )
        except Exception:
            pass

        # L1: Checkpoint state (highest priority)
        if _cp:
            # CKPT-004: Consumer-friendly checkpoint presentation
            done_items = _cp.get("done") or []
            next_items = _cp.get("next") or []
            total_items = len(done_items) + len(next_items)
            completion_pct = round(len(done_items) / total_items * 100) if total_items > 0 else 0

            # Time since last update
            time_since = ""
            try:
                updated = _ensure_aware(datetime.fromisoformat(_cp.get("updated_at", "")))
                delta = _utc_now() - updated
                if delta.total_seconds() < 3600:
                    time_since = f"{int(delta.total_seconds() / 60)}m ago"
                elif delta.total_seconds() < 86400:
                    time_since = f"{int(delta.total_seconds() / 3600)}h ago"
                else:
                    time_since = f"{delta.days}d ago"
            except Exception:
                time_since = "unknown"

            # Resume hint — consumer-facing one-liner
            desc = (_cp.get("description") or "")[:60]
            active = (_cp.get("active") or "")[:40]
            is_authoritative = _cp.get("selection_authority") == "authoritative"
            if active and is_authoritative:
                resume_hint = f"You were working on: {active} ({desc})"
            elif desc and is_authoritative:
                resume_hint = f"Pick up where you left off: {desc}"
            else:
                resume_hint = None

            wc["checkpoint"] = {
                "task_id": _cp.get("task_id"),
                "origin_id": _cp.get("origin_id"),
                "selection_source": _cp.get("selection_source"),
                "selection_authority": _cp.get("selection_authority"),
                "description": (_cp.get("description") or "")[:80],
                "status": _cp.get("status"),
                "active": (_cp.get("active") or "")[:60],
                "done_count": len(done_items),
                "next_count": len(next_items),
                "completion_pct": completion_pct,
                "time_since_update": time_since,
                "resume_hint": resume_hint,
            }

        # L2: Active task + domain
        active_task = self._extract_active_task(
            request.message if hasattr(request, "message") else "",
            _cached_checkpoint=_cp,
        )
        if active_task:
            task_domain = None
            if self._last_activated_concept_ids:
                try:
                    from app.storage import load_concept
                    top_c = load_concept(self._last_activated_concept_ids[0], track_access=False)
                    if top_c:
                        task_domain = top_c.knowledge_area
                except Exception:
                    pass
            wc["task"] = {
                "active_task": active_task,
                "task_domain": task_domain,
            }

        # L3: Session metadata
        session = self.current_session
        if session:
            import time as _wc_time
            elapsed = 0.0
            if session.started_at:
                try:
                    started = datetime.fromisoformat(session.started_at)
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=UTC)
                    elapsed = (_wc_time.time() - started.timestamp()) / 60.0
                except Exception:
                    pass
            wc["session"] = {
                "session_id": session.session_id[:12],  # Truncate for token budget
                "turn_count": getattr(self, "_episode_turn_counter", 0),
                "learning_events": session.learning_event_count,
                "elapsed_minutes": round(elapsed, 1),
            }

        # L4: Tools used (from checkpoint context)
        if _cp and _cp.get("context", {}).get("tools"):
            wc["tools"] = _cp["context"]["tools"][:5]

        # L5: Pinned concepts (cached per turn)
        pinned = self._select_pinned_concepts()
        if pinned:
            wc["pinned_concepts"] = pinned

        # Token budget enforcement — priority trimming L5->L4->L3->L2 (keep L1)
        estimated_tokens = len(_wc_json.dumps(wc)) // 4
        if estimated_tokens > WORKING_CONTEXT_MAX_TOKENS:
            for trim_key in ["pinned_concepts", "tools", "session", "task"]:
                if trim_key in wc:
                    del wc[trim_key]
                if len(_wc_json.dumps(wc)) // 4 <= WORKING_CONTEXT_MAX_TOKENS:
                    break

        return wc if wc else None

    def _handle_compaction_reinjection(
        self, request: ConversationTurnRequest
    ) -> tuple[str | None, str | None, str | None, float]:
        """Re-inject critical context after compaction detection.

        Returns: (resume_context, orientation_summary, greeting_hint, recovery_quality)
        Loads the latest rolling snapshot (shared with Resume Context v1.1)
        and re-serves orientation.
        recovery_quality: 0.0–1.0 score reflecting how substantive the recovery was.
          0.4 base for having a snapshot, +0.2 each for active_task, pinned_concepts, gist.
        """
        resume_context = None
        orientation_summary = None
        greeting_hint = None
        recovery_quality = 0.0  # CTX-005: quality score for observability

        try:
            snapshot = load_resume_snapshot()
            if snapshot:
                # Build re-injection from snapshot (same format as Resume Context)
                active_task = snapshot.get("active_task", "")
                task_domain = snapshot.get("task_domain", "")
                gist = snapshot.get("last_exchange_gist", "")
                pinned = snapshot.get("pinned_concepts", [])
                pinned_summaries = [p.get("summary", "") for p in pinned if p.get("summary")]

                # CTX-005: compute recovery quality from snapshot contents
                recovery_quality = self.COMPACTION_QUALITY_HAS_SNAPSHOT
                if active_task:
                    recovery_quality += self.COMPACTION_QUALITY_HAS_TASK
                if pinned_summaries:
                    recovery_quality += self.COMPACTION_QUALITY_HAS_PINNED
                if gist:
                    recovery_quality += self.COMPACTION_QUALITY_HAS_GIST

                # CONTEXT-001: Prose fallback for resume_context (working_context carries structured data)
                parts = ["COMPACTION_RECOVERY: Your context was likely summarized."]
                parts.append("Check working_context field for full structured state.")
                if active_task:
                    parts.append(f"You were working on: {active_task}.")
                if task_domain:
                    parts.append(f"Domain: {task_domain}.")
                if pinned_summaries:
                    parts.append(f"Key context: [{', '.join(pinned_summaries[:5])}].")
                if gist:
                    parts.append(f"Last exchange touched: {gist}.")
                resume_context = " ".join(parts)

            # Re-serve orientation (TEMPORAL_AWARENESS v2.4)
            orientation_summary, greeting_hint = self._build_temporal_context(request, is_resumption=False)
            # Override greeting hint for compaction recovery
            greeting_hint = (
                "COMPACTION_RECOVERY. Your context was likely summarized. "
                "Critical operational context has been re-injected. "
                "Reference the resume_context for work state."
            )
        except Exception as e:
            logger.warning(f"CTX: Compaction re-injection failed (non-fatal): {e}")

        return resume_context, orientation_summary, greeting_hint, recovery_quality
