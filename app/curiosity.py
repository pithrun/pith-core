"""Curiosity engine for self-generated questions."""

import json
import logging

from app.datetime_utils import _utc_now_iso
from app.models import Question
from app.storage import _get_connection

logger = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.45
HIGH_CONFLICT = 2
LOW_STABILITY = 0.40


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


# Global instance
curiosity_engine = CuriosityEngine()
