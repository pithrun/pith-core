"""P0.2: Client extraction schemas and garbage detection.

Handles validation, quality filtering, and security checking for
client-extracted concepts in the session_learn pipeline.

v2: Type-aware garbage detection (Knowledge Hierarchy architecture)
- GROUNDED types (L1-L2): word grounding check (25% overlap with source)
- ABSTRACT types (L3-L6): coherence + provenance check (no word grounding)

v3: Hardened garbage detection
- Evidence quality minimum: evidence items must be >= 10 chars to count
- Stopword filtering: common English stopwords stripped before overlap calc
"""

import logging
import math
import os
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.cognitive.taxonomy import normalize_knowledge_area
from app.core.models import ABSTRACT_CONCEPT_TYPES, CONCEPT_TYPES

logger = logging.getLogger(__name__)

# Common English stopwords to filter before computing word overlap.
# Inflated overlap from stopwords makes the 25% threshold meaningless.
# Compact set — covers ~80% of stopword frequency in English text.
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
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
    "not",
    "no",
    "nor",
    "so",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "when",
    "where",
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
    "as",
    "into",
    "about",
    "up",
    "out",
    "just",
    "also",
    "very",
    "too",
    "only",
    "own",
    "same",
    "any",
    "there",
}

# Minimum character length for an evidence item to count as real provenance.
# Prevents trivial bypass like evidence: ["x"] or evidence: ["yes"].
MIN_EVIDENCE_LENGTH = 10


def _has_quality_evidence(evidence: list[str] | None) -> bool:
    """Check if concept has at least one evidence item meeting quality bar."""
    if not evidence:
        return False
    return any(len(e.strip()) >= MIN_EVIDENCE_LENGTH for e in evidence)


def _content_words(text: str) -> set[str]:
    """Extract content words (non-stopwords) from text."""
    return set(re.findall(r"\w+", text.lower())) - STOPWORDS


class ExtractedConcept(BaseModel):
    """Schema for client-extracted concepts in session_learn."""

    summary: str
    confidence: float | None = 0.50
    knowledge_area: str | None = "general"
    evidence: list[str] | None = Field(default_factory=list)
    signals: list[str] | None = Field(default_factory=list)
    concept_type: str | None = None  # None = client didn't specify (logged + treated as grounded)
    supersedes: list[str] | None = None  # Concept IDs this replaces (EXPLICIT_SUPERSESSION_SPEC v1.1)
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    # Benchmark source identity. These fields are intentionally explicit rather
    # than relying on Pydantic extras, so benchmark adapters can round-trip
    # provenance without polluting summary text.
    beam_source_key: str | None = None
    beam_source_turn_id: str | None = None
    # Some benchmark source corpora use composite turn indexes such as "2,17".
    # Treat this as provenance identity, not a numeric rank.
    beam_source_turn_index: str | int | None = None
    beam_source_batch_idx: int | None = None
    beam_source_role: str | None = None
    beam_role: str | None = None

    # RETRIEVAL-104: Edit-chain provenance for entity chain filtering
    edit_provenance: str | None = None  # JSON array of question_ids or session_ids

    @field_validator("supersedes")
    @classmethod
    def validate_supersedes(cls, v):
        if v is None:
            return v
        if len(v) > 5:
            raise ValueError(f"Too many supersessions ({len(v)}, max 5)")
        for cid in v:
            if not isinstance(cid, str) or len(cid) < 5:
                raise ValueError(f"Invalid concept ID in supersedes: {cid}")
        return v

    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, v):
        # Basic length floor — type-aware ceiling is in model_validator below
        if len(v) < 30:
            raise ValueError(f"Summary too short ({len(v)} chars, min 30)")
        # Hard ceiling for all types (defense-in-depth)
        if len(v) > 800:
            raise ValueError(f"Summary too long ({len(v)} chars, max 800)")
        return v

    @model_validator(mode="after")
    def validate_summary_for_type(self):
        """Type-aware summary length: abstract types get 800 chars, grounded get 500.
        L3+ concepts (principles, methods, heuristics) need more space to express
        generalizable patterns with conditions. See: conv_d190eb423ff7."""
        max_len = 800 if self.concept_type in ABSTRACT_CONCEPT_TYPES else 500
        if len(self.summary) > max_len:
            raise ValueError(
                f"Summary too long for {self.concept_type or 'untyped'} ({len(self.summary)} chars, max {max_len})"
            )
        return self

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        if v is None:
            return 0.50
        return max(0.35, min(0.85, v))

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_and_cap_evidence(cls, v):
        """Coerce string evidence to list (Sonnet sends strings), then cap."""
        if v is None:
            return []
        if isinstance(v, str):
            # Single string → wrap in list (common with Sonnet extraction)
            v = [v]
        if not isinstance(v, list):
            return []
        return [str(item)[:200] for item in v[:3]]

    @field_validator("signals", mode="before")
    @classmethod
    def coerce_signals(cls, v):
        """Coerce string signals to list (same pattern as evidence)."""
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        return [str(item)[:100] for item in v[:5]]

    @field_validator("concept_type")
    @classmethod
    def validate_concept_type(cls, v):
        if v is None:
            return None  # Preserve None — signals client didn't set type
        if v not in CONCEPT_TYPES:
            logger.warning(f"Unknown concept_type '{v}', defaulting to None (untyped)")
            return None
        return v


class GarbageDetector:
    """Detects incompetent (not adversarial) client extractions.

    Type-aware: Abstract concepts (principles, methods, strategies) use
    coherence + provenance checks instead of word grounding, because their
    value comes from transcending the source text.

    TODO(KTA-EC4): Abstract types are too permissive through garbage detector.
        Need quality floor for abstract concepts — currently they bypass word
        grounding entirely, allowing low-quality abstractions through.
    TODO(KTA): Log rejected concept types for observability. Currently rejections
        are logged individually but no aggregate type-level stats are tracked.
    """

    @staticmethod
    def detect_batch(concepts: list[ExtractedConcept], source_text: str) -> tuple[list[ExtractedConcept], list[dict]]:
        """Run garbage detection on a batch of extracted concepts.
        Returns (survivors, rejections) where rejections include reason.
        """
        source_words = _content_words(source_text)
        word_count = len(source_text.split())

        # Proportional count: separate budgets per type class.
        # Protocol tells clients to extract 1-5 concepts per exchange.
        # Floor of 5 grounded + 5 abstract = 10 minimum supports retrospective
        # exchanges that naturally produce many L3+ concepts.
        # Previous floor of 2 abstract killed valid L3+ concepts from retros.
        # See: conv_924d671369c6, conv_d190eb423ff7
        # INGEST-026: Floor matches client cap (PITH_MAX_INSIGHTS_PER_CALL, default 7).
        # Previous floor of 5 killed 2/7 facts per batch on short conversations.
        # FC_mh_6k RCA: 84/99 losses (85%) from this cap alone.
        try:
            _client_cap = int(os.environ.get("PITH_MAX_INSIGHTS_PER_CALL", "7"))
        except (ValueError, TypeError):
            _client_cap = 7
        _client_cap = max(1, _client_cap)  # Defensive floor
        max_grounded = max(_client_cap, math.ceil(word_count / 200))
        max_abstract = max(_client_cap, math.ceil(word_count / 200))

        survivors = []
        rejections = []
        grounded_count = 0
        abstract_count = 0
        untyped_count = 0

        for i, concept in enumerate(concepts):
            is_abstract = concept.concept_type in ABSTRACT_CONCEPT_TYPES
            is_untyped = concept.concept_type is None
            if is_untyped:
                untyped_count += 1

            # Check proportional limits per type class
            if is_abstract:
                if abstract_count >= max_abstract:
                    rejections.append(
                        {
                            "index": i,
                            "reason": f"abstract_count_exceeded (max {max_abstract})",
                            "summary_preview": concept.summary[:50],
                        }
                    )
                    continue
            else:
                if grounded_count >= max_grounded:
                    rejections.append(
                        {
                            "index": i,
                            "reason": f"grounded_count_exceeded (max {max_grounded})",
                            "summary_preview": concept.summary[:50],
                        }
                    )
                    continue

            rejection_reason = GarbageDetector._check_single(concept, source_words, is_abstract)
            if rejection_reason:
                rejections.append({"index": i, "reason": rejection_reason, "summary_preview": concept.summary[:50]})
            else:
                survivors.append(concept)
                if is_abstract:
                    abstract_count += 1
                else:
                    grounded_count += 1

        # Measurement signal: how many concepts arrive without explicit type?
        if untyped_count > 0:
            logger.info(
                f"garbage_detector: {untyped_count}/{len(concepts)} concepts untyped "
                f"(treated as grounded). Typed: {len(concepts) - untyped_count}"
            )

        return survivors, rejections

    @staticmethod
    def _check_single(concept: ExtractedConcept, source_words: set, is_abstract: bool) -> str | None:
        """Check a single concept. Returns rejection reason or None.

        Two paths:
        - GROUNDED (observations, decisions, etc.): word grounding check
        - ABSTRACT (principles, methods, strategies): coherence + provenance
        """
        summary = concept.summary

        # === COMMON CHECKS (all types) ===

        # Length bounds (defense-in-depth, validator should catch these)
        # Type-aware: abstract types get 800 chars, grounded get 500
        if len(summary) < 30:
            return f"summary_too_short ({len(summary)} chars)"
        max_len = 800 if is_abstract else 500
        if len(summary) > max_len:
            return f"summary_too_long ({len(summary)} chars, max {max_len})"

        # Minimal coherence: at least 3 unique content words (stopwords excluded)
        unique_words = _content_words(summary)
        if len(unique_words) < 3:
            return f"minimal_coherence_fail ({len(unique_words)} unique content words)"

        # Attack 4: Generic filler detection — catch compliance-without-quality
        GENERIC_WORDS = {
            "discussed",
            "various",
            "topics",
            "progress",
            "things",
            "worked",
            "looked",
            "talked",
            "reviewed",
            "considered",
            "general",
            "several",
            "different",
            "important",
            "relevant",
            "aspects",
            "elements",
            "areas",
            "items",
            "stuff",
            "mentioned",
            "addressed",
            "covered",
            "explored",
            "noted",
        }
        generic_ratio = len(unique_words & GENERIC_WORDS) / max(len(unique_words), 1)
        if generic_ratio > 0.40:
            return f"generic_filler ({generic_ratio:.0%} generic words)"

        # === TYPE-SPECIFIC CHECKS ===

        if is_abstract:
            return GarbageDetector._check_abstract(concept, unique_words)
        else:
            return GarbageDetector._check_grounded(concept, unique_words, source_words)

    @staticmethod
    def _check_grounded(concept: ExtractedConcept, unique_words: set, source_words: set) -> str | None:
        """Grounded type check: word overlap with source text.

        Two paths:
        - WITH quality evidence (>=10 chars per item): skip word grounding.
          Evidence is the provenance signal — the client deliberately crafted
          this concept and cited its origin. Trivial evidence like ["x"] or
          ["yes"] does NOT qualify. Count limits + coherence + length checks
          still catch garbage.
        - WITHOUT quality evidence: 25% content-word overlap required.
          Stopwords are already stripped from both sets, so this threshold
          measures real vocabulary overlap, not "the"/"and" inflation.
        """
        if source_words:
            if _has_quality_evidence(concept.evidence):
                return None  # Quality evidence = trusted provenance
            overlap = len(unique_words & source_words) / len(unique_words) if unique_words else 0
            if overlap < 0.25:
                return f"word_grounding_fail ({overlap:.0%} overlap, need 25%)"
        return None

    @staticmethod
    def _check_abstract(concept: ExtractedConcept, unique_words: set) -> str | None:
        """Abstract type check: coherence + provenance (NO word grounding).

        Abstract concepts (principles, methods, strategies) are EXPECTED to
        transcend the source text. Instead we check:
        1. Sufficient complexity (>= 5 unique words)
        2. Provenance: must have at least 1 quality evidence item
        3. Evidence coherence: evidence must relate to the summary (shared content words)

        v3 hardening: EC4 proved that jargon + unrelated evidence passed v2 checks.
        The evidence-coherence check ensures provenance is actually ABOUT the concept.
        """
        # KTA-EC4: Confidence floor for abstract types.
        # Catches test artifacts and near-zero confidence items (27 in production).
        # Defense-in-depth: Tier 3 clamps to >=0.30, session_learn to >=0.35.
        ABSTRACT_CONFIDENCE_FLOOR = 0.1
        if concept.confidence is not None and concept.confidence < ABSTRACT_CONFIDENCE_FLOOR:
            return f"abstract_confidence_floor ({concept.confidence:.3f} < {ABSTRACT_CONFIDENCE_FLOOR})"

        # Higher coherence bar for abstract types
        if len(unique_words) < 5:
            return f"abstract_coherence_fail ({len(unique_words)} unique words, need 5)"

        # Provenance requirement: abstract concepts must cite their origin
        # Evidence must meet quality bar (>= 10 chars) — not just ["x"]
        if not _has_quality_evidence(concept.evidence):
            return "abstract_provenance_fail (no quality evidence linking to origin)"

        # Evidence-coherence check: at least one quality evidence item must share
        # content words with the summary. This catches jargon paired with
        # unrelated evidence — the evidence must actually be ABOUT the concept.
        # Uses 4-char prefix matching to handle morphological variants
        # (e.g., "dogfood"/"dogfooding", "gap"/"gaps", "session"/"sessions").
        summary_prefixes = {w[:4] for w in unique_words if len(w) >= 4}
        evidence_coherent = False
        for ev in concept.evidence or []:
            if len(ev.strip()) < MIN_EVIDENCE_LENGTH:
                continue
            ev_words = _content_words(ev)
            ev_prefixes = {w[:4] for w in ev_words if len(w) >= 4}
            if len(ev_prefixes & summary_prefixes) >= 1:
                evidence_coherent = True
                break
        if not evidence_coherent:
            return "abstract_evidence_coherence_fail (no evidence shares content words with summary)"

        return None


# =============================================================================
# PERF-001: Tier 3 LLM Extraction
# =============================================================================

TIER3_EXTRACTION_PROMPT = """Analyze this conversation exchange and extract 1-{max_concepts} concepts that represent NEW knowledge, decisions, or insights NOT already captured in the provided existing concepts.

SESSION DATE: {session_date}
When the conversation uses relative time references ("last week", "2 months ago", "recently"), convert them to approximate absolute dates using the session date above. Example: if session date is 2023-03-15 and user says "about 2 months ago", store "around January 2023". If session date is "unknown", preserve relative references as-is.

Focus on:
- Implicit reasoning: architectural decisions embedded in code discussion
- Unstated constraints: requirements that shape the conversation but aren't explicitly listed
- Cross-domain connections: insights that bridge multiple knowledge areas
- Methodology patterns: recurring approaches or workflows being established
- Cognitive strategies: meta-reasoning patterns about HOW to think about problems
- User preferences: stated likes, dislikes, working style, or behavioral preferences (e.g., "I prefer X over Y", "I always want Z", "don't do W")

CONVERSATION:
<user_message>{user_message}</user_message>
<assistant_response>{assistant_response}</assistant_response>

EXISTING CONCEPTS (already extracted — do NOT duplicate these):
{existing_concepts}

KNOWN KNOWLEDGE AREAS (prefer these when assigning knowledge_area — only create a new area if none fit):
{ka_hints}

Return JSON array. Each concept:
{{"summary": "30-500 chars, clear standalone insight", "confidence": 0.3-0.8, "knowledge_area": "one of the known areas above, or a new descriptive domain if none fit", "concept_type": "principle|method|heuristic|cognitive_strategy|decision|pattern|observation|preference", "evidence": ["source text >=10 chars from conversation"], "is_factual": false, "temporal_category": null}}

CRITICAL — Summary Resolution Rules:
Summaries MUST preserve specific details, not abstract them:
- WRONG: "User upgraded their laptop's RAM" → RIGHT: "User upgraded laptop RAM to 16GB"
- WRONG: "User recommended a light beer" → RIGHT: "User recommended Pilsner or Lager for Seco de Cordero"
- WRONG: "User attended theater" → RIGHT: "User attended The Glass Menagerie"
- WRONG: "User's budget for the project" → RIGHT: "User's budget is $4,500 for the project"
Preserve in summaries: proper nouns, specific numbers/amounts/dates/times, named entities
(restaurants, books, products, medications, people, places, brands, titles).
Do NOT store verbatim transcripts — concepts should be standalone insights, not quotes.
The goal: if someone later asks "what was that name/number/time?" — the summary has the answer.

KEYWORD FIDELITY RULE (CRITICAL):
When the source text uses a SPECIFIC term, your summary MUST use that EXACT term.
Do NOT paraphrase specific nouns, entities, or domain terms into synonyms or hypernyms.
  BAD: "looking into expanding her family" when source says "adoption agencies" — USE "adoption agencies"
  BAD: "exploring care options" when source says "nursing homes" — USE "nursing homes"
  BAD: "studying life sciences" when source says "marine biology" — USE "marine biology"
  BAD: "attending a community event" when source says "LGBTQ support group" — USE "LGBTQ support group"
The test: if someone searches for the original keyword, will they find this concept? If not, you lost it.

DEMOGRAPHIC FACT EXTRACTION RULE (MANDATORY — EXTRACT-C2):
When the user's age, birthday, birth year, or decade of life is mentioned or inferable,
you MUST extract an explicit standalone concept stating the user's age as a number.
This applies whether the age is stated directly OR implied by context:
  DIRECT: "I'm 32" → extract "The user is 32 years old"
  DIRECT: "I just turned 25" → extract "The user is 25 years old (just turned)"
  IMPLIED: "still getting used to being in my 30s" → extract "The user is in their 30s"
  IMPLIED: assistant says "at 32, you're still young" → extract "The user is 32 years old"
  IMPLIED: "my 32nd birthday" → extract "The user is 32 years old"
Also ALWAYS extract as separate standalone facts:
  - Birth year if mentioned or computable
  - Birthday date/month if mentioned
  - Occupation/job title
  - Location/city of residence
These are HIGH-VALUE retrieval anchors. Missing them makes cross-session questions unanswerable.

POSSESSION & OWNERSHIP EXTRACTION RULE (MANDATORY — EXTRACT-C5):
When the user mentions acquiring, owning, buying, or possessing a specific item, extract a standalone
concept stating what they own — even if the item appears inside a broader anecdote or experience.
  "I bought their EP 'Midnight Sky'" → extract "User purchased EP 'Midnight Sky' by [artist]"
  "got my vinyl signed after the show" → extract "User owns a vinyl record by [artist]"
  "just picked up a Martin D-28" → extract "User purchased a Martin D-28 guitar"
  "downloaded the album on Spotify" → extract "User downloaded [album] by [artist] on Spotify"
Ownership signals: bought, purchased, picked up, got, ordered, downloaded, subscribed, own, have.
Do NOT collapse these into the surrounding experience. "Went to concert and got vinyl signed" contains
TWO facts: (1) attended concert, (2) owns a vinyl record. Extract BOTH as separate concepts.

Note on is_factual: Set true for personal facts including biographical info AND episodic details:
dates, times, amounts, quantities, events attended, purchases made, appointments, travel times.
When is_factual is true, set temporal_category to one of: "identity", "role", "activity", "relational".

Rules:
- Only extract concepts genuinely present in the conversation — do not infer beyond what's stated
- Extract BOTH abstract patterns AND specific factual observations from the same conversation
- For episodic facts (dates, amounts, events), always create observation-type concepts with is_factual=true
- Use "preference" ONLY for user-stated behavioral preferences — not assistant recommendations
- Each concept must be independently meaningful — not a fragment
- If no additional concepts exist beyond what's already captured, return []
- Maximum {max_concepts} concepts

CRITICAL — Evidence Resolution Rules:
In evidence items, ALWAYS preserve:
- Proper nouns (names of people, places, businesses, products, books, movies, songs)
- Brand names and model numbers (e.g., "Martin D-28", "MacBook Pro M3")
- Specific numbers, amounts, counts, scores, and ages
- Dates and time references (e.g., "March 15", "last Tuesday", "Q2 2025")
- Named locations (e.g., "5th Avenue Music Store", "Sugar Factory at Icon Park")
Do NOT abstract these into generic descriptions. "Purchased Martin D-28 at 5th Ave Music"
is correct evidence. "Bought guitar at store" loses critical retrieval detail.

JSON array:"""


def build_tier3_prompt(
    user_message: str,
    assistant_response: str,
    existing_concepts: list[dict],
    max_concepts: int = 3,
    session_date: str | None = None,  # RAGAS-DIAG-001 Fix 3b
    ka_hints: list[str] | None = None,  # KA-INJECT-001: extraction-time KA guidance
) -> str:
    """Build the Tier 3 LLM extraction prompt.

    Args:
        user_message: The user's message from the conversation turn
        assistant_response: The assistant's response
        existing_concepts: Concepts already extracted by Tier 1+2
        max_concepts: Maximum concepts to request from LLM
        session_date: ISO date string for temporal anchoring (None → "unknown")
        ka_hints: Known knowledge areas to guide LLM classification (None → generic list)
    """
    from app.core.config import TIER3_MAX_INPUT_CHARS

    existing_str = (
        "\n".join(f"- [{c.get('type', 'observation')}] {c.get('summary', '')[:200]}" for c in existing_concepts)
        or "(none)"
    )

    # KA-INJECT-001: Format KA hints for prompt injection
    if ka_hints:
        ka_hints_str = ", ".join(ka_hints)
    else:
        ka_hints_str = "knowledge, workflow, relationships, context, goals, observations"

    return TIER3_EXTRACTION_PROMPT.format(
        user_message=(user_message or "")[:TIER3_MAX_INPUT_CHARS],
        assistant_response=(assistant_response or "")[:TIER3_MAX_INPUT_CHARS],
        existing_concepts=existing_str,
        max_concepts=max_concepts,
        session_date=session_date or "unknown",  # RAGAS-DIAG-001 Fix 3b
        ka_hints=ka_hints_str,  # KA-INJECT-001
    )


def parse_tier3_response(raw_text: str, max_concepts: int = 3) -> list[dict]:
    """Parse Haiku's JSON response into validated concept dicts.

    Returns list of valid concept dicts, filtering out malformed entries.
    """
    import json
    import logging
    import re

    logger = logging.getLogger(__name__)

    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"PERF-001: Tier 3 response not valid JSON: {raw_text[:200]}")
        return []

    if not isinstance(parsed, list):
        logger.warning("PERF-001: Tier 3 response is not a JSON array")
        return []

    valid = []
    for item in parsed[:max_concepts]:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary", "")
        if not summary or len(summary) < 30:
            continue
        evidence = item.get("evidence", [])
        if not evidence or not isinstance(evidence, list):
            continue
        # Validate evidence items are strings >= 10 chars
        valid_evidence = [e for e in evidence if isinstance(e, str) and len(e) >= 10]
        if not valid_evidence:
            continue

        valid.append(
            {
                "summary": summary[:500],
                "confidence": min(max(item.get("confidence", 0.45), 0.30), 0.70),
                "type": item.get("concept_type", "observation"),
                "signals": [],
                "evidence": valid_evidence[:3],
                "knowledge_area": normalize_knowledge_area(item.get("knowledge_area", "conversation"), strict=True)[0],  # DATA-049
                "extraction_source": "llm_tier3",
                "was_untyped": False,
                "supersedes": None,
            }
        )

    return valid



# --- Prospective Indexing (RETRIEVAL-057) ---

def build_implications_prompt(
    summary: str,
    knowledge_area: str,
    concept_type: str,
    evidence: list[str],
    max_implications: int = 5,
) -> str:
    """Build prompt for generating hypothetical future retrieval scenarios.

    v0 prompt — REQUIRES ITERATION. Quality of implications determines retrieval
    improvement. This prompt is a starting point; expect 2-3 revision cycles based
    on empirical retrieval testing.
    """
    evidence_str = "; ".join(evidence[:3]) if evidence else "none"
    return f"""You are a memory indexing system. Given a knowledge concept, generate
{max_implications} short hypothetical future scenarios where this concept would be
relevant to retrieve. Each scenario should:

1. Use DIFFERENT vocabulary than the original summary
2. Describe a concrete situation where this knowledge would help
3. Be 1 sentence, max 100 characters
4. Cover diverse retrieval angles (debugging, planning, explaining, deciding)

Concept summary: {summary}
Knowledge area: {knowledge_area}
Type: {concept_type}
Evidence: {evidence_str}

Respond with a JSON array of strings. Example:
["scenario 1", "scenario 2", "scenario 3"]

Return ONLY the JSON array, no other text."""


def parse_implications_response(raw_text: str, max_count: int = 5) -> list[str]:
    """Parse LLM response into list of implication strings.

    Robust parser — handles markdown code blocks, trailing text, partial JSON.
    Returns empty list on parse failure (fail-open).
    """
    import json
    import re

    # Strip markdown code blocks if present
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(parsed, list):
        return []

    # Filter and truncate
    implications = []
    for item in parsed[:max_count]:
        if isinstance(item, str) and len(item.strip()) >= 10:
            implications.append(item.strip()[:150])  # Hard cap at 150 chars

    return implications



# --- Event Extraction (INGEST-034) ---

class EventTuple(BaseModel):
    """Structured causal event tuple extracted from conversation."""
    action: str  # What happened
    cause: str | None = None  # Why it happened
    consequence: str | None = None  # What resulted
    actors: list[str] = Field(default_factory=list)  # Entities involved
    confidence: float = 0.7  # LLM's confidence in extraction quality

    @field_validator("action")
    @classmethod
    def validate_action_length(cls, v):
        if len(v.strip()) < 10:
            raise ValueError(f"Action too short ({len(v)} chars, min 10)")
        if len(v) > 300:
            return v[:300]
        return v

    @field_validator("cause", "consequence")
    @classmethod
    def cap_length(cls, v):
        if v is not None and len(v) > 300:
            return v[:300]
        return v

    @field_validator("actors", mode="before")
    @classmethod
    def coerce_actors(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(a)[:100] for a in v[:10]]  # Cap 10 actors, 100 chars each

    def to_searchable_text(self) -> str:
        """Convert event tuple to text for embedding search."""
        parts = [self.action]
        if self.cause:
            parts.append(f"because {self.cause}")
        if self.consequence:
            parts.append(f"resulting in {self.consequence}")
        if self.actors:
            parts.append(f"involving {', '.join(self.actors)}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        """Serialize for JSON storage in concept data blob."""
        d = {"action": self.action}
        if self.cause:
            d["cause"] = self.cause
        if self.consequence:
            d["consequence"] = self.consequence
        if self.actors:
            d["actors"] = self.actors
        d["confidence"] = self.confidence
        return d


def build_event_extraction_prompt(conversation_text: str) -> str:
    """Build the LLM prompt for event extraction."""
    return f"""Extract structured events from this conversation. For each significant event,
identify the action taken, its cause (why), its consequence (what resulted), and the
actors/entities involved.

Return JSON array of event objects. Each event has:
- "action": what happened (REQUIRED, 10-300 chars)
- "cause": why it happened (optional, up to 300 chars)
- "consequence": what resulted (optional, up to 300 chars)
- "actors": list of entities involved (optional, up to 10 items)
- "confidence": your confidence in extraction quality (0.0-1.0)

Focus on:
- Decisions and their reasoning
- Technical choices and trade-offs
- Problems encountered and solutions applied
- Process changes and their motivations

Skip trivial actions (greetings, acknowledgments, simple Q&A).
Return empty array [] if no significant events found.
Maximum 5 events.

CONVERSATION:
{conversation_text}

Respond with ONLY a JSON array, no other text:"""


def parse_event_response(response_text: str) -> list[EventTuple]:
    """Parse LLM response into validated EventTuple objects.

    Handles malformed JSON gracefully — returns empty list on parse failure.
    """
    import json

    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("INGEST-034: Event extraction JSON parse failed")
        return []

    if not isinstance(raw, list):
        raw = [raw]  # Single event wrapped

    events = []
    for item in raw[:5]:  # Cap at EE_MAX_EVENTS_PER_CALL
        if not isinstance(item, dict):
            continue
        if not item.get("action"):
            continue
        try:
            event = EventTuple(**item)
            events.append(event)
        except Exception as e:
            logger.debug(f"INGEST-034: Skipping invalid event tuple: {e}")
            continue

    return events
