"""Skill Learning from Experience — dual extraction pipeline.

Gap C from the governance spec v1.3:
- Correction-driven extraction: 2+ corrections in same domain → skill candidate
- Trajectory-driven extraction: After successful complex tasks → reusable patterns
- Skill injection at S4.8 with constraint cross-check
- Skill effectiveness tracking with domain-active denominator fix (v1.2 S5)
- Skill currency with 60-day half-life
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

from app.core.constants import GOV_EVENT_SKILL_EXTRACTED
from app.core.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class LearnedSkill:
    """A skill extracted from correction patterns or successful trajectories.

    Spec C.1: Skills are reusable instructions injected at S4.8.
    """

    id: str
    name: str
    trigger_pattern: str  # Regex or keyword pattern
    trigger_context: list[str] = field(default_factory=list)  # Knowledge areas
    instruction: str = ""
    source_type: str = "correction"  # correction | trajectory | explicit
    source_corrections: list[str] = field(default_factory=list)
    extraction_confidence: float = 0.5
    times_applied: int = 0
    times_effective: int = 0
    effectiveness_rate: float = 0.0
    currency: float = 1.0  # v1.2: Skills need currency too
    status: str = "active"  # active | deprecated | merged
    created_at: str = ""
    updated_at: str = ""


# Skill currency half-life matches decisions (60 days)
SKILL_CURRENCY_HALF_LIFE_DAYS = 60.0

# Maximum skills injected per turn (spec C.3)
MAX_SKILL_INJECTIONS = 3

# Minimum extraction confidence to create a skill
MIN_EXTRACTION_CONFIDENCE = 0.60

# Minimum corrections in same domain before skill extraction
MIN_CORRECTIONS_FOR_SKILL = 2

# Effectiveness deprecation threshold (spec C.4)
EFFECTIVENESS_DEPRECATION_THRESHOLD = 0.30
MIN_APPLICATIONS_FOR_DEPRECATION = 10


# =============================================================================
# Skill Storage
# =============================================================================


def _skill_from_row(row: dict) -> LearnedSkill:
    """Convert a DB row to a LearnedSkill object."""
    return LearnedSkill(
        id=row["id"],
        name=row["name"],
        trigger_pattern=row["trigger_pattern"] or "",
        trigger_context=json.loads(row["trigger_context"]) if row["trigger_context"] else [],
        instruction=row["instruction"] or "",
        source_type=row["source_type"] or "correction",
        source_corrections=json.loads(row["source_corrections"]) if row["source_corrections"] else [],
        extraction_confidence=row["extraction_confidence"] or 0.5,
        times_applied=row["times_applied"] or 0,
        times_effective=row["times_effective"] or 0,
        effectiveness_rate=row["effectiveness_rate"] or 0.0,
        currency=row["currency"] or 1.0,
        status=row["status"] or "active",
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
    )


def save_skill(skill: LearnedSkill, conn) -> None:
    """Persist a learned skill to the DB."""
    now = _utc_now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO learned_skills
           (id, name, trigger_pattern, trigger_context, instruction,
            source_type, source_corrections, extraction_confidence,
            times_applied, times_effective, effectiveness_rate,
            currency, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            skill.id,
            skill.name,
            skill.trigger_pattern,
            json.dumps(skill.trigger_context),
            skill.instruction,
            skill.source_type,
            json.dumps(skill.source_corrections),
            skill.extraction_confidence,
            skill.times_applied,
            skill.times_effective,
            skill.effectiveness_rate,
            skill.currency,
            skill.status,
            skill.created_at or now,
            now,
        ),
    )
    conn.commit()


def load_skill(skill_id: str, conn) -> LearnedSkill | None:
    """Load a single skill by ID."""
    conn.row_factory = _dict_factory
    row = conn.execute("SELECT * FROM learned_skills WHERE id = ?", (skill_id,)).fetchone()
    if not row:
        return None
    return _skill_from_row(row)


def load_active_skills(conn) -> list[LearnedSkill]:
    """Load all active skills with positive currency."""
    conn.row_factory = _dict_factory
    rows = conn.execute(
        "SELECT * FROM learned_skills WHERE status = 'active' AND currency > 0.05 ORDER BY effectiveness_rate DESC"
    ).fetchall()
    return [_skill_from_row(r) for r in rows]


def _dict_factory(cursor, row):
    """SQLite row factory that returns dicts."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


# =============================================================================
# C.2: Correction-Driven Extraction
# =============================================================================


def extract_skills_from_corrections(conn, gov_ctx=None) -> list[LearnedSkill]:
    """Correction-driven extraction (spec C.2).

    Algorithm:
    1. Load corrections not yet extracted (skill_extracted = 0)
    2. Group by knowledge_area
    3. Where count >= MIN_CORRECTIONS_FOR_SKILL, synthesize a skill
    4. Cross-check against constraints (spec C.6 L18-2)
    5. Mark corrections as skill_extracted = 1

    Returns list of newly created skills.
    """
    conn.row_factory = _dict_factory
    new_skills = []

    # Step 1: Load unextracted corrections grouped by knowledge area
    rows = conn.execute(
        """SELECT COALESCE(json_extract(co.data, '$.metadata.knowledge_area'), 'general') AS knowledge_area,
                  COUNT(*) as cnt,
                  GROUP_CONCAT(c.id, '||') as correction_ids,
                  GROUP_CONCAT(COALESCE(c.corrected_claim, '') || ' → ' || COALESCE(c.correct_claim, ''), ' ||| ') as correction_texts
           FROM corrections c
           LEFT JOIN concepts co ON c.concept_id = co.id
           WHERE c.skill_extracted = 0
           GROUP BY knowledge_area
           HAVING cnt >= ?""",
        (MIN_CORRECTIONS_FOR_SKILL,),
    ).fetchall()

    if not rows:
        logger.debug("No correction clusters ready for skill extraction")
        return new_skills

    for row in rows:
        area = row["knowledge_area"] or "general"
        correction_ids = row["correction_ids"].split("||")
        correction_texts = row["correction_texts"]

        # Synthesize skill instruction from correction patterns
        skill_id = f"skill_{area}_{int(time.time())}"
        skill_name = f"Learned pattern in {area}"

        # Extract common instruction from correction texts
        instruction = _synthesize_instruction(correction_texts, area)
        if not instruction:
            continue

        # Build trigger pattern from the knowledge area
        trigger_pattern = _build_trigger_pattern(area, correction_texts)

        skill = LearnedSkill(
            id=skill_id,
            name=skill_name,
            trigger_pattern=trigger_pattern,
            trigger_context=[area],
            instruction=instruction,
            source_type="correction",
            source_corrections=correction_ids,
            extraction_confidence=min(0.5 + len(correction_ids) * 0.1, 0.95),
            created_at=_utc_now_iso(),
        )

        # C.6 L18-2: Cross-check against constraints at extraction time
        if not _passes_constraint_check(skill, conn, gov_ctx):
            logger.info(f"Skill {skill_id} contradicts active constraint — auto-deprecated")
            skill.status = "deprecated"

        save_skill(skill, conn)
        new_skills.append(skill)

        if gov_ctx:
            gov_ctx.log_event(
                GOV_EVENT_SKILL_EXTRACTED,
                concept_id=None,
                details={
                    "skill_id": skill_id,
                    "source_type": "correction",
                    "area": area,
                    "correction_count": len(correction_ids),
                    "status": skill.status,
                },
            )

        # Mark corrections as extracted
        placeholders = ",".join("?" * len(correction_ids))
        conn.execute(
            f"UPDATE corrections SET skill_extracted = 1 WHERE id IN ({placeholders})",
            correction_ids,
        )
        conn.commit()

    logger.info(f"Correction-driven extraction: {len(new_skills)} skills created")
    return new_skills


def _synthesize_instruction(correction_texts: str, area: str) -> str:
    """Extract a common instruction pattern from multiple correction texts.

    Simple heuristic: find the most common actionable phrases across corrections.
    Returns the synthesized instruction or empty string if no clear pattern.
    """
    if not correction_texts:
        return ""

    segments = [s.strip() for s in correction_texts.split("|||") if s.strip()]
    if len(segments) < MIN_CORRECTIONS_FOR_SKILL:
        return ""

    # Extract key phrases (words after action verbs)
    action_verbs = ["should", "must", "always", "never", "use", "avoid", "prefer", "instead"]
    instructions = []

    for seg in segments:
        lower = seg.lower()
        for verb in action_verbs:
            idx = lower.find(verb)
            if idx >= 0:
                # Take the clause starting from the verb
                clause = seg[idx:].strip()
                if len(clause) > 10:
                    instructions.append(clause)
                break

    if not instructions:
        # Fallback: use the longest correction as the instruction
        longest = max(segments, key=len)
        if len(longest) > 20:
            return f"In {area}: {longest}"
        return ""

    # Use the longest instruction as representative
    return max(instructions, key=len)


def _build_trigger_pattern(area: str, correction_texts: str) -> str:
    """Build a trigger pattern from the knowledge area and correction keywords.

    Returns a regex-compatible pattern string.
    """
    # Extract significant words from corrections (>4 chars, not stopwords)
    stopwords = {
        "about",
        "after",
        "again",
        "being",
        "could",
        "doing",
        "every",
        "first",
        "found",
        "going",
        "given",
        "their",
        "there",
        "these",
        "thing",
        "think",
        "those",
        "using",
        "which",
        "while",
        "would",
        "should",
        "because",
        "before",
        "between",
    }

    segments = correction_texts.split("|||")
    word_counts: dict[str, int] = {}
    for seg in segments:
        words = set(re.findall(r"\b[a-z]{5,}\b", seg.lower()))
        words -= stopwords
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

    # Words appearing in 2+ corrections are trigger keywords
    trigger_words = [w for w, c in sorted(word_counts.items(), key=lambda x: -x[1]) if c >= 2][:5]

    if trigger_words:
        return "|".join(trigger_words)
    return area.replace("_", "|")


# =============================================================================
# C.3: Skill Injection at S4.8
# =============================================================================


# =============================================================================
# C.4: Skill Effectiveness Tracking (v1.2 FIX: S5)
# =============================================================================


def track_skill_effectiveness(
    skill: LearnedSkill, session_had_corrections: bool, domain_was_active: bool, conn
) -> None:
    """Track skill effectiveness with domain-active denominator fix (v1.2 S5).

    effectiveness = sessions_with_no_corrections_in_domain
                  / sessions_where_skill_applied_AND_domain_active

    "Domain active" means: at least 1 concept in skill's trigger_context
    was retrieved in that session.

    Without this fix, a skill could reach 100% effectiveness simply
    because its domain was never discussed.

    Deprecate at effectiveness_rate < 0.3 after 10+ QUALIFIED applications.
    """
    if not domain_was_active:
        # Domain wasn't active — this session doesn't count for effectiveness
        return

    # Domain was active and skill was applied
    if not session_had_corrections:
        skill.times_effective += 1

    # Recalculate effectiveness rate
    if skill.times_applied > 0:
        skill.effectiveness_rate = skill.times_effective / skill.times_applied

    # Check for deprecation
    if (
        skill.times_applied >= MIN_APPLICATIONS_FOR_DEPRECATION
        and skill.effectiveness_rate < EFFECTIVENESS_DEPRECATION_THRESHOLD
    ):
        skill.status = "deprecated"
        logger.info(
            f"Skill {skill.id} deprecated: effectiveness {skill.effectiveness_rate:.2f} "
            f"< {EFFECTIVENESS_DEPRECATION_THRESHOLD} after {skill.times_applied} applications"
        )

    skill.updated_at = _utc_now_iso()
    save_skill(skill, conn)


# =============================================================================
# C.5: Skill Currency (v1.2)
# =============================================================================


# =============================================================================
# C.6: Constraint Cross-Checks (v1.3 FIX: L18-2)
# =============================================================================


def _passes_constraint_check(skill: LearnedSkill, conn, gov_ctx=None) -> bool:
    """Extraction-time constraint cross-check (C.6 step 1).

    When a new skill is extracted, cross-check its instruction against all
    active CONSTRAINT-level concepts. If instruction contradicts a constraint
    → return False (auto-deprecate).
    """
    try:
        conn.row_factory = _dict_factory
        constraints = conn.execute(
            """SELECT concept_id, summary FROM concepts
               WHERE concept_type = 'constraint'
               AND (authority_score IS NULL OR authority_score >= 0.5)"""
        ).fetchall()

        instruction_lower = skill.instruction.lower()
        for c in constraints:
            # Check if constraint summary contains opposing signals
            summary_lower = (c["summary"] or "").lower()
            # Simple anti-term check: if constraint says "never X" and skill says "X"
            for negation in ["never", "not", "don't", "avoid", "shouldn't"]:
                idx = summary_lower.find(negation)
                if idx >= 0:
                    # Extract the object of negation
                    after = summary_lower[idx + len(negation) :].strip()[:50]
                    key_words = [w for w in after.split() if len(w) > 4][:3]
                    for kw in key_words:
                        if kw in instruction_lower:
                            logger.info(f"Skill instruction contradicts constraint {c['concept_id']}: keyword '{kw}'")
                            return False
    except Exception as e:
        logger.warning(f"Constraint check failed (allowing skill): {e}")

    return True


def _passes_injection_constraint_check(skill: LearnedSkill, gov_ctx) -> bool:
    """Injection-time constraint cross-check (C.6 step 2, within 3ms budget).

    Quick cross-check at S4.8: scan constraint_set anti_terms against
    skill instruction. Catches constraints created AFTER skill was extracted.
    """
    if not gov_ctx or not hasattr(gov_ctx, "constraint_set"):
        return True

    constraint_set = gov_ctx.constraint_set
    if not constraint_set:
        return True

    instruction_lower = skill.instruction.lower()

    for constraint in constraint_set:
        if isinstance(constraint, dict):
            anti_terms = constraint.get("anti_terms", [])
            c_id = constraint.get("concept_id", "?")
        else:
            anti_terms = getattr(constraint, "anti_terms", [])
            c_id = getattr(constraint, "concept_id", "?")
        for anti_term in anti_terms:
            if anti_term and anti_term.lower() in instruction_lower:
                logger.info(f"Skill {skill.id} blocked by anti-term '{anti_term}' from constraint {c_id}")
                return False

    return True


# =============================================================================
# Trajectory-Driven Extraction (spec C.2)
# =============================================================================


# =============================================================================
# Migration helper — learned_skills table
# =============================================================================


LEARNED_SKILLS_DDL = """
CREATE TABLE IF NOT EXISTS learned_skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger_pattern TEXT,
    trigger_context TEXT,       -- JSON array of knowledge areas
    instruction TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'correction',
    source_corrections TEXT,    -- JSON array of correction IDs
    extraction_confidence REAL DEFAULT 0.5,
    times_applied INTEGER DEFAULT 0,
    times_effective INTEGER DEFAULT 0,
    effectiveness_rate REAL DEFAULT 0.0,
    currency REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_learned_skills_status
    ON learned_skills(status);
CREATE INDEX IF NOT EXISTS idx_learned_skills_area
    ON learned_skills(trigger_context);
"""


def ensure_learned_skills_table(conn) -> None:
    """Create the learned_skills table if it doesn't exist."""
    for stmt in LEARNED_SKILLS_DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
