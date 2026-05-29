"""Curiosity engine for self-generated questions."""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime

from app.core.datetime_utils import _utc_now, _utc_now_iso
from app.core.models import Question
from app.storage import _get_connection

logger = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.45
HIGH_CONFLICT = 2
LOW_STABILITY = 0.40
FRONTIER_LANE = "experiment_frontier"
_TEST_ARTIFACT_RE = re.compile(
    r"\[test-cleanup\]|\bd2 test run\b|\btest concept\b|\btest pipeline confirmation\b|\bec\d+\b.*\bremoved\b",
    re.IGNORECASE,
)
_STRICT_ARTIFACT_RE = re.compile(r"\b(fixture|dummy|synthetic)\b", re.IGNORECASE)
_MOCK_ARTIFACT_RE = re.compile(r"\bmock\b.*\b(test|fixture|data|sample)\b", re.IGNORECASE)
_RESOLVED_LEARNING_RE = re.compile(
    r"\b(already|now)\s+(resolved|fixed|upgraded|implemented)\b"
    r"|\b(has|have)\s+been\s+(resolved|fixed|replaced|implemented|superseded)\b"
    r"|\bsuperseded\s+by\b"
    r"|\bno\s+longer\s+actionable\b"
    r"|\bno\s+regression\s+here\b"
    r"|\bengine\s+upgraded\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]{4,}")
_THEME_STOPWORDS = {
    "about",
    "additional",
    "because",
    "concept",
    "concepts",
    "evidence",
    "experiment",
    "experiments",
    "learning",
    "question",
    "reasoning",
    "should",
    "summary",
    "these",
    "this",
    "would",
}


def _clip_text(text: str, limit: int = 140) -> str:
    """Keep generated dry-run questions compact and readable."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _looks_like_test_artifact(*parts: object) -> bool:
    """Detect obvious test fixtures without treating normal phrases as artifacts."""
    text = " ".join(str(p or "") for p in parts).lower()
    return bool(_TEST_ARTIFACT_RE.search(text) or _STRICT_ARTIFACT_RE.search(text) or _MOCK_ARTIFACT_RE.search(text))


def _looks_resolved_or_superseded(*parts: object) -> bool:
    """Detect explicit stale/resolved markers without rejecting future-tense plans."""
    text = " ".join(str(p or "") for p in parts)
    return bool(_RESOLVED_LEARNING_RE.search(text))


def _experiment_age_days(timestamp: object) -> int | None:
    """Return whole days since an experiment timestamp, or None when unknown."""
    if not timestamp:
        return None
    raw = str(timestamp).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0, (_utc_now() - parsed.astimezone(UTC)).days)


def _experiment_frontier_revalidation_days() -> int:
    """Minimum completed-experiment age before operational actions require revalidation."""
    try:
        return max(1, int(os.environ.get("PITH_EXPERIMENT_FRONTIER_REVALIDATION_DAYS", "7")))
    except (TypeError, ValueError):
        return 7


def _theme_key(experiment_type: str, text: str) -> str:
    """Build a coarse duplicate-pressure key; this is not semantic equivalence."""
    tokens = sorted({t for t in _TOKEN_RE.findall(str(text or "").lower()) if t not in _THEME_STOPWORDS})
    return f"{experiment_type}:{'-'.join(sorted(tokens[:8]))}"


class CuriosityEngine:
    """Generates questions for weak or uncertain concepts."""

    def detect_gaps(self) -> list[dict]:
        """Detect knowledge gaps in Pith.

        Optimized: single SQL query instead of N+1 load_concept calls.
        Fetches only the fields needed (confidence, stability, summary, data)
        and filters in SQL where possible.
        """
        gaps = []

        conn = _get_connection()
        # Single query replaces N+1 pattern (was: list_concepts + load_concept per ID)
        # Fetches only needed columns. ~1 query vs ~2200 queries.
        rows = conn.execute(
            """SELECT id, summary, confidence, stability, data
               FROM concepts WHERE status = 'active'
               AND maturity NOT IN ('QUARANTINED', 'DISCARDED')"""
        ).fetchall()

        for row in rows:
            concept_id, summary, confidence, stability = row[0], row[1], row[2], row[3]

            # Parse hypotheses count from JSON data for conflict detection
            conflict = 0
            try:
                data = json.loads(row[4]) if row[4] else {}
                conflict = len(data.get("hypotheses", []))
            except (json.JSONDecodeError, TypeError):
                pass

            reasons = []
            if confidence < LOW_CONFIDENCE:
                reasons.append("low_confidence")
            if stability < LOW_STABILITY:
                reasons.append("low_stability")
            if conflict >= HIGH_CONFLICT:
                reasons.append("conflicting_models")

            if reasons:
                gaps.append(
                    {
                        "concept_id": concept_id,
                        "summary": summary,
                        "confidence": confidence,
                        "stability": stability,
                        "conflict": conflict,
                        "reasons": reasons,
                    }
                )

        logger.info(f"Curiosity detect_gaps: {len(gaps)} gaps from {len(rows)} concepts (1 SQL query)")
        return gaps

    def generate_questions(self) -> list[Question]:
        """Generate questions for all detected gaps."""
        gaps = self.detect_gaps()
        questions = []

        for gap in gaps:
            question_text = self._generate_question(gap)
            priority = self._compute_priority(gap)

            question = Question(
                concept_id=gap["concept_id"],
                question=question_text,
                priority=priority,
                created_at=_utc_now_iso(),
                reasons=gap["reasons"],
            )

            questions.append(question)

        # Sort by priority
        questions.sort(key=lambda q: q.priority, reverse=True)

        return questions

    def _generate_question(self, gap: dict) -> str:
        """Generate appropriate question based on gap type."""
        summary = gap["summary"]
        reasons = gap["reasons"]

        if "conflicting_models" in reasons:
            return f"What evidence distinguishes competing explanations of: {summary}?"

        if "low_confidence" in reasons:
            return f"What additional evidence would increase confidence in: {summary}?"

        if "low_stability" in reasons:
            return f"In what contexts does this concept fail or change: {summary}?"

        return f"What is missing about: {summary}?"

    def _compute_priority(self, gap: dict) -> float:
        """Compute priority score for a gap."""
        return (1 - gap["confidence"]) * 0.5 + (1 - gap["stability"]) * 0.3 + min(gap["conflict"], 3) * 0.2

    def generate_experiment_frontier_questions(self, limit: int = 20, baseline_limit: int = 20) -> dict:
        """Generate read-only frontier questions from Experiment Engine outputs."""
        limit = max(1, int(limit))
        baseline_limit = max(0, int(baseline_limit))
        payload = {
            "dry_run": True,
            "lane": FRONTIER_LANE,
            "questions": [],
            "metrics": {
                "candidate_count": 0,
                "questions_returned": 0,
                "test_artifact_rejections": 0,
                "duplicate_pressure_groups": 0,
                "unresolved_reasoning": 0,
                "stale_revalidation_downgrades": 0,
                "resolved_or_stale_rejections": 0,
                "disposition_counts": {},
                "experiment_type_counts": {},
            },
            "baseline": self._experiment_frontier_baseline(baseline_limit),
        }

        try:
            from app.features.experiments import load_experiments

            reasoning = load_experiments(status=["reasoning"], limit=limit)
            completed = load_experiments(status=["completed"], limit=max(limit * 3, 50))
        except Exception as exc:
            payload["error"] = f"experiment_load_failed: {exc.__class__.__name__}"
            return payload

        candidates = []
        for experiment in reasoning:
            candidates.append(self._frontier_candidate(experiment, "resolve", 0.95))

        for experiment in completed:
            if not getattr(experiment, "concept_ids_produced", None):
                continue
            text = self._experiment_frontier_text(experiment)
            if _looks_resolved_or_superseded(text):
                payload["metrics"]["resolved_or_stale_rejections"] += 1
                continue
            confidence = getattr(getattr(experiment, "result", None), "confidence", 0.0) or 0.0
            disposition = "operationalize" if confidence >= 0.65 else "validate"
            priority = 0.82 if disposition == "operationalize" else 0.72
            updated_at = getattr(experiment, "updated_at", "") or getattr(experiment, "created_at", "")
            age_days = _experiment_age_days(updated_at)
            stale_days = _experiment_frontier_revalidation_days()
            revalidation_required = False
            if disposition == "operationalize" and age_days is not None and age_days >= stale_days:
                disposition = "validate"
                priority = 0.74
                revalidation_required = True
                payload["metrics"]["stale_revalidation_downgrades"] += 1
            candidates.append(
                self._frontier_candidate(
                    experiment,
                    disposition,
                    priority,
                    age_days=age_days,
                    revalidation_required=revalidation_required,
                )
            )

        payload["metrics"]["candidate_count"] = len(candidates)
        payload["metrics"]["unresolved_reasoning"] = len(reasoning)
        questions = self._rank_frontier_candidates(candidates, limit, payload["metrics"])
        payload["questions"] = questions
        payload["metrics"]["questions_returned"] = len(questions)
        payload["metrics"]["disposition_counts"] = dict(Counter(q["disposition"] for q in questions))
        payload["metrics"]["experiment_type_counts"] = dict(Counter(q["source_experiment_type"] for q in questions))
        return payload

    def _frontier_candidate(
        self,
        experiment,
        disposition: str,
        priority: float,
        age_days: int | None = None,
        revalidation_required: bool = False,
    ) -> dict:
        experiment_type = getattr(experiment, "experiment_type", "experiment")
        source_ids = list(getattr(experiment, "concept_ids_produced", []) or [])
        text = self._experiment_frontier_text(experiment)
        theme_text = self._experiment_frontier_text(experiment, include_ids=False)
        summary = _clip_text(text or experiment_type, 70)

        if disposition == "resolve":
            question = (
                f"Which evidence would make this unresolved {experiment_type} experiment worth "
                "promoting, rejecting, rewriting, or marking insufficient-data?"
            )
            reason = "unresolved_reasoning_experiment"
        elif disposition == "operationalize":
            question = f"What evidence justifies changing a decision, test, or backlog item for this {experiment_type} learning: {summary}?"
            reason = "high_confidence_experiment_learning"
        else:
            question = f"What concrete evidence would confirm or falsify this {experiment_type} learning: {summary}?"
            reason = "stale_completed_experiment_revalidation" if revalidation_required else "experiment_learning_needs_validation"

        return {
            "question": question,
            "disposition": disposition,
            "source_experiment_id": getattr(experiment, "id", ""),
            "source_experiment_type": experiment_type,
            "source_concept_ids": source_ids,
            "priority": priority,
            "reason": reason,
            "source_age_days": age_days,
            "theme_key": _theme_key(experiment_type, theme_text),
            "updated_at": getattr(experiment, "updated_at", "") or getattr(experiment, "created_at", ""),
            "_artifact": _looks_like_test_artifact(text, source_ids),
        }

    def _rank_frontier_candidates(self, candidates: list[dict], limit: int, metrics: dict) -> list[dict]:
        artifact_rejections = 0
        theme_groups: dict[str, list[dict]] = defaultdict(list)
        clean_candidates = []
        for candidate in candidates:
            if candidate.pop("_artifact", False):
                artifact_rejections += 1
                continue
            clean_candidates.append(candidate)
            theme_groups[candidate["theme_key"]].append(candidate)

        metrics["test_artifact_rejections"] = artifact_rejections
        duplicate_groups = [items for items in theme_groups.values() if len(items) > 1]
        metrics["duplicate_pressure_groups"] = len(duplicate_groups)

        ranked = []
        for items in duplicate_groups:
            representative = sorted(items, key=lambda c: (-c["priority"], c["source_experiment_id"]))[0]
            ranked.append(
                {
                    "question": (
                        "Do these experiment outputs express one stronger principle or redundant noise: "
                        f"{_clip_text(representative['question'], 120)}"
                    ),
                    "disposition": "consolidate",
                    "source_experiment_id": representative["source_experiment_id"],
                    "source_experiment_type": representative["source_experiment_type"],
                    "source_concept_ids": sorted({cid for item in items for cid in item["source_concept_ids"]}),
                    "source_experiment_ids": [item["source_experiment_id"] for item in items],
                    "priority": 0.88,
                    "reason": "duplicate_pressure_group",
                }
            )

        type_counts: Counter = Counter()
        type_cap = max(2, limit // 4)
        duplicate_theme_keys = {items[0]["theme_key"] for items in duplicate_groups}
        for candidate in sorted(clean_candidates, key=lambda c: (-c["priority"], c.get("updated_at", ""), c["source_experiment_id"])):
            if candidate.get("theme_key") in duplicate_theme_keys:
                continue
            exp_type = candidate["source_experiment_type"]
            if type_counts[exp_type] >= type_cap:
                continue
            type_counts[exp_type] += 1
            ranked.append({k: v for k, v in candidate.items() if k not in {"theme_key", "updated_at"}})

        ranked.sort(key=lambda q: (-q["priority"], q["source_experiment_id"]))
        return ranked[:limit]

    def _experiment_frontier_text(self, experiment, include_ids: bool = True) -> str:
        parts = []
        result = getattr(experiment, "result", None)
        if result:
            parts.extend([getattr(result, "synthesis", ""), getattr(result, "reasoning_trace", "")])
        for candidate in getattr(experiment, "candidates", []) or []:
            parts.append(getattr(candidate, "rationale", ""))
            if include_ids:
                parts.extend(getattr(candidate, "concept_ids", []) or [])
        if include_ids:
            parts.extend(getattr(experiment, "concept_ids_produced", []) or [])
        return " ".join(str(p) for p in parts if p)

    def _experiment_frontier_baseline(self, baseline_limit: int) -> dict:
        baseline = {
            "sample_size": 0,
            "generic_prompt_count": 0,
            "overlong_prompt_count": 0,
            "test_artifact_count": 0,
        }
        if baseline_limit <= 0:
            return baseline

        try:
            import app.features.question_queue as question_queue

            questions = question_queue.get_questions(limit=baseline_limit)
        except Exception as exc:
            baseline["error"] = f"baseline_queue_failed: {exc.__class__.__name__}"
            return baseline

        baseline["sample_size"] = len(questions)
        for question in questions:
            text = question.get("question", "") if isinstance(question, dict) else str(question)
            concept_id = question.get("concept_id", "") if isinstance(question, dict) else ""
            if text.startswith("What additional evidence would increase confidence in:"):
                baseline["generic_prompt_count"] += 1
            if len(text) > 180:
                baseline["overlong_prompt_count"] += 1
            if _looks_like_test_artifact(text, concept_id):
                baseline["test_artifact_count"] += 1
        return baseline


# Global instance
curiosity_engine = CuriosityEngine()
