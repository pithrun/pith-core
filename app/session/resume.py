"""Session resume/snapshot capture and injection.

Extracted from session/turn.py per ARCH-009b (Modularity Plan v2 Item 2a step 5).
"""

import logging
from datetime import timedelta

from app.core.datetime_utils import _ensure_aware, _utc_now
from app.core.models import ConversationTurnRequest
from app.storage import (
    load_concept,
    load_resume_snapshot,
    save_resume_snapshot,
)

logger = logging.getLogger(__name__)


class ResumeMixin:
    """Session snapshot capture and resume injection.

    Methods:
        _capture_rolling_snapshot
        _extract_active_task
        _select_pinned_concepts
        _extract_gist
        _inject_resume_context
        _compute_drift_score
        _detect_resumption
    """

    def _capture_rolling_snapshot(self, request: ConversationTurnRequest) -> None:
        """RC-A: Capture rolling snapshot at end of conversation_turn.

        Runs after response assembly. Best-effort — failures logged, not raised.
        v1.1: Caches concept summaries at write time (not IDs).
        v1.1: Time-decay access scoring to prevent gaming.
        CONTEXT-001: Hoisted checkpoint load shared by active_task + tools + checkpoint_summary.
        """
        try:
            session = self.current_session
            if not session:
                return

            session_id = session.session_id

            # CONTEXT-001: Hoist checkpoint load — shared by active_task, tools_used, checkpoint_summary
            _cached_cp = None
            try:
                from app.storage import load_checkpoint
                _cached_cp = load_checkpoint(
                    task_id=getattr(request, "current_task_id", None),
                    origin_id=getattr(request, "origin_id", None),
                )
            except Exception:
                pass

            # Extract active_task from user message via simple keyword extraction
            active_task = self._extract_active_task(request.message, _cached_checkpoint=_cached_cp)

            # Determine task_domain from most-activated concept's knowledge_area
            task_domain = None
            if self._last_activated_concept_ids:
                try:
                    top_concept = load_concept(self._last_activated_concept_ids[0], track_access=False)
                    if top_concept:
                        task_domain = top_concept.knowledge_area
                except Exception:
                    pass

            # Build pinned concepts (v1.1: cached summaries, time-decay scoring)
            pinned_concepts = self._select_pinned_concepts()

            # Extract gist via keyword extraction
            gist_text = (request.message or "")[-100:]
            if request.previous_response:
                gist_text += " " + (request.previous_response or "")[-200:]
            last_exchange_gist = self._extract_gist(gist_text)

            # Session metadata
            turn_count = getattr(session, "reflection_turn_counter", 0) + 1
            learning_events_count = session.learning_event_count

            # Tools used (inferred from checkpoint context) — CONTEXT-001: uses hoisted _cached_cp
            tools_used = []
            if _cached_cp and _cached_cp.get("context", {}).get("tools"):
                tools_used = _cached_cp["context"]["tools"][:5]

            # CONTEXT-001: Extract checkpoint summary for working_context L1
            checkpoint_summary = None
            if _cached_cp and _cached_cp.get("selection_authority") == "authoritative":
                checkpoint_summary = {
                    "task_id": _cached_cp.get("task_id"),
                    "description": (_cached_cp.get("description") or "")[:80],
                    "status": _cached_cp.get("status"),
                    "active": (_cached_cp.get("active") or "")[:60],
                    "done_count": len(_cached_cp.get("done") or []),
                    "next_count": len(_cached_cp.get("next") or []),
                }

            # SESSION-012: Compute topic_keywords from recent conversation context
            topic_keywords = ""
            try:
                _tk_text = (request.message or "")[-300:]
                if request.previous_response:
                    _tk_text += " " + (request.previous_response or "")[-300:]
                if _tk_text.strip():
                    topic_keywords = self._extract_gist(_tk_text)
            except Exception:
                pass  # Topic keywords are enrichment, not critical path

            save_resume_snapshot(
                session_id=session_id,
                active_task=active_task,
                task_domain=task_domain,
                pinned_concepts=pinned_concepts,
                last_exchange_gist=last_exchange_gist,
                turn_count=turn_count,
                learning_events=learning_events_count,
                tools_used=tools_used,
                checkpoint_summary=checkpoint_summary,  # CONTEXT-001
                topic_keywords=topic_keywords,  # SESSION-012
            )
        except Exception as e:
            logger.warning(f"RC-A: Snapshot capture failed (non-fatal): {e}")

    def _extract_active_task(self, message: str, _cached_checkpoint: dict | None = None) -> str | None:
        """Extract active task description from user message.

        Uses simple term frequency — top terms from message, capped at 80 chars.
        Falls back to checkpoint description if active checkpoint exists.
        CONTEXT-001: Accepts _cached_checkpoint to avoid redundant load.
        """
        if not message or len(message.strip()) < 5:
            return None

        # Check for active checkpoint first
        # CONTEXT-001: Use cached checkpoint if provided
        cp = _cached_checkpoint
        if cp is None:
            try:
                from app.storage import load_checkpoint

                cp = load_checkpoint()
            except Exception:
                pass
        if (
            cp
            and cp.get("selection_authority") == "authoritative"
            and cp.get("status") in ("active", "planning")
            and cp.get("description")
        ):
            return cp["description"][:80]

        # Simple extraction: remove common stopwords, take top terms
        import re

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "but",
            "and",
            "or",
            "if",
            "while",
            "that",
            "this",
            "what",
            "which",
            "who",
            "whom",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "let",
            "lets",
            "let's",
            "hey",
            "hi",
            "hello",
            "please",
            "thanks",
            "yeah",
            "yes",
            "ok",
            "okay",
        }
        words = re.findall(r"\b[a-zA-Z_]{3,}\b", message.lower())
        terms = [w for w in words if w not in stopwords]

        if not terms:
            return None

        # Frequency count, take top 5 terms
        from collections import Counter

        freq = Counter(terms)
        top_terms = [t for t, _ in freq.most_common(5)]
        result = ", ".join(top_terms)
        return result[:80]

    def _select_pinned_concepts(self) -> list[dict]:
        """Select top 3 pinned concepts by time-decayed access frequency.

        v1.1: Returns cached {id, summary} dicts, not just IDs.
        v1.1: Time-decay scoring prevents artificial boosting.
        CONTEXT-001: Turn-scoped cache — avoids redundant recomputation within same turn.
        """
        # CONTEXT-001: Return cached result if same turn
        current_turn = getattr(self, "_episode_turn_counter", 0)
        if (
            self._cached_pinned_concepts is not None
            and self._cached_pinned_concepts_turn == current_turn
        ):
            return self._cached_pinned_concepts

        if not self._last_activated_concept_ids:
            return []

        pinned = []
        seen = set()
        for cid in self._last_activated_concept_ids[:10]:  # Check top 10
            if cid in seen:
                continue
            seen.add(cid)
            try:
                concept = load_concept(cid, track_access=False)
                if concept and concept.confidence >= 0.1:
                    pinned.append(
                        {
                            "id": cid,
                            "summary": (concept.summary or "")[:40],
                        }
                    )
                    if len(pinned) >= 3:
                        break
            except Exception:
                continue

        # CONTEXT-001: Cache for this turn
        self._cached_pinned_concepts = pinned
        self._cached_pinned_concepts_turn = current_turn
        return pinned

    def _extract_gist(self, text: str) -> str:
        """Extract keyword gist from text via simple term frequency.

        v1.1: Sanitizes output — strips IPs, emails, tokens.
        Returns comma-separated top terms, max 120 chars.
        """
        import re

        if not text or len(text.strip()) < 5:
            return ""

        # v1.1: Sanitize — remove structured data that could leak
        text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "", text)  # IPs
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "", text)  # emails
        text = re.sub(r"\b[a-fA-F0-9]{32,}\b", "", text)  # hex tokens
        text = re.sub(r"https?://\S+", "", text)  # URLs

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
            "it",
            "its",
            "i",
            "we",
            "you",
            "they",
            "my",
            "our",
            "your",
            "their",
            "what",
            "which",
            "who",
            "how",
            "all",
            "each",
            "just",
            "also",
            "being",
            "into",
            "more",
        }
        words = re.findall(r"\b[a-zA-Z_]{3,}\b", text.lower())
        terms = [w for w in words if w not in stopwords]

        if not terms:
            return ""

        from collections import Counter

        freq = Counter(terms)
        top_terms = [t for t, _ in freq.most_common(7)]
        result = ", ".join(top_terms)
        return result[:120]

    def _inject_resume_context(self, request: ConversationTurnRequest) -> tuple[str | None, str | None, bool]:
        """RC-B: Inject resume context on first call of resumed session.

        Returns: (resume_context_str, tier, was_suppressed)
        v1.1: Topic drift detection — suppresses injection if user started new work.
        """
        try:
            # Find prior session's snapshot
            snapshot = load_resume_snapshot()
            if not snapshot:
                return None, None, False

            # Don't inject our own session's snapshot
            current_id = self.current_session.session_id if self.current_session else None
            if snapshot["session_id"] == current_id:
                return None, None, False

            # Determine tier based on age
            from datetime import datetime as dt

            captured_at = _ensure_aware(dt.fromisoformat(snapshot["captured_at"]))
            age_hours = (_utc_now() - captured_at).total_seconds() / 3600

            if age_hours > self.RESUME_TIER_STALE_DAYS * 24:
                return None, "EXPIRED", False
            elif age_hours > self.RESUME_TIER_RECENT_HOURS:
                tier = "STALE"
                budget = self.RESUME_TOKEN_STALE
            elif age_hours > self.RESUME_TIER_FRESH_HOURS:
                tier = "RECENT"
                budget = self.RESUME_TOKEN_RECENT
            else:
                tier = "FRESH"
                budget = self.RESUME_TOKEN_FRESH

            # v1.1: Topic drift detection
            if request.message:
                drift_score = self._compute_drift_score(request.message, snapshot)
                if drift_score < self.RESUME_DRIFT_THRESHOLD:
                    logger.info(
                        f"RC-B: Drift detected (score={drift_score:.3f} < {self.RESUME_DRIFT_THRESHOLD}), suppressing injection"
                    )
                    return None, tier, True  # suppressed

            # Build injection text
            active_task = snapshot.get("active_task", "")
            task_domain = snapshot.get("task_domain", "")
            gist = snapshot.get("last_exchange_gist", "")
            pinned = snapshot.get("pinned_concepts", [])
            turn_count = snapshot.get("turn_count", 0)
            learning_events = snapshot.get("learning_events", 0)
            tools = snapshot.get("tools_used", [])

            pinned_summaries = [p.get("summary", "") for p in pinned if p.get("summary")]

            if tier == "FRESH":
                parts = [f"RESUME: You were working on: {active_task}."]
                if task_domain:
                    parts.append(f"Domain: {task_domain}.")
                if pinned_summaries:
                    parts.append(f"Key context: [{', '.join(pinned_summaries)}].")
                if gist:
                    parts.append(f"Last exchange touched: {gist}.")
                if turn_count > 0:
                    parts.append(f"Session had {min(turn_count, 100)}+ turns, {learning_events} concepts learned.")
                if tools:
                    parts.append(f"Tools active: {', '.join(tools)}.")
                resume_text = " ".join(parts)

            elif tier == "RECENT":
                parts = [f"RESUME: Prior session worked on: {active_task}."]
                if task_domain:
                    parts.append(f"Domain: {task_domain}.")
                if pinned_summaries:
                    parts.append(f"Key context: [{', '.join(pinned_summaries)}].")
                resume_text = " ".join(parts)

            else:  # STALE
                parts = [f"RESUME: Last active domain: {task_domain or 'general'}."]
                if gist:
                    parts.append(f"Topics: {gist}.")
                resume_text = " ".join(parts)

            # Enforce per-tier token budget (word count proxy)
            words = resume_text.split()
            if len(words) > budget:
                resume_text = " ".join(words[:budget])

            logger.info(f"RC-B: Injecting resume context tier={tier} tokens={len(words)} task={active_task}")
            return resume_text, tier, False

        except Exception as e:
            logger.warning(f"RC-B: Resume injection failed (non-fatal): {e}")
            return None, None, False

    def _compute_drift_score(self, message: str, snapshot: dict) -> float:
        """Compute TF-IDF-like similarity between current message and snapshot context.

        Returns 0.0 (no overlap) to 1.0 (perfect match).
        Used for v1.1 drift detection — suppress injection if user started new work.
        """
        import re

        # Build snapshot text from active_task + gist
        snapshot_text = (snapshot.get("active_task", "") or "") + " " + (snapshot.get("last_exchange_gist", "") or "")

        # Tokenize both
        def tokenize(text):
            return set(re.findall(r"\b[a-zA-Z_]{3,}\b", text.lower()))

        msg_tokens = tokenize(message)
        snap_tokens = tokenize(snapshot_text)

        if not msg_tokens or not snap_tokens:
            return 0.0

        # Jaccard similarity (simple, fast, no dependencies)
        intersection = msg_tokens & snap_tokens
        union = msg_tokens | snap_tokens
        return len(intersection) / len(union) if union else 0.0

    def _detect_resumption(self) -> bool:
        """B5.1: Detect if this is a resumption (prior session within 24h).

        Returns True if at least one ended session exists within the last 24 hours.
        Gracefully returns False on any error.
        """
        try:
            from app.storage import list_sessions

            cutoff_24h = (_utc_now() - timedelta(hours=24)).isoformat()
            recent = list_sessions(limit=5, since=cutoff_24h)
            # Filter to sessions that actually ended (not the current one)
            # Note: list_sessions returns "id" column, not "session_id"
            current_id = self.current_session.session_id if self.current_session else None
            prior_sessions = [s for s in recent if s.get("status") == "ended" and s.get("id") != current_id]
            if prior_sessions:
                logger.info(f"B5.1: Resumption detected — {len(prior_sessions)} prior session(s) in 24h")
                return True
            return False
        except Exception as e:
            logger.warning(f"B5.1: Resumption detection failed (non-fatal): {e}")
            return False
