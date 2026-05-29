"""INGEST-044: Triple extraction from concept summaries.

Extracts (subject, predicate, object) triples from concept summary text.
Two-tier strategy:
  Tier 1 — Regex extraction using _PREDICATE_PATTERNS (zero cost, 76.0% coverage)
  Tier 2 — LLM extraction for unstructured summaries (Phase 2, optional)
"""

import logging
from typing import Optional

logger = logging.getLogger("pith.triple_extractor")

# Import predicate patterns from entity_chain (shared vocabulary)
from app.cognitive.entity_chain import _PREDICATE_PATTERNS

# Maximum allowed length for extracted subject/object strings.
# Prevents garbage triples from long text preceding pattern match.
_MAX_ENTITY_LEN = 100

# INGEST-053: Entity blocklist — regex artifact prefixes that produce
# high-degree hub nodes with no semantic value (e.g., "the name" → 180 edges).
_ENTITY_BLOCKLIST = frozenset({
    'the name', 'the type', 'the kind', 'the form', 'the title',
    'the number', 'the date', 'the cost', 'the price', 'the size',
    'the amount', 'the level', 'the rate', 'the result',
})

# INGEST-053: Maximum edges per entity. Prevents hub explosion from
# regex artifacts that survive blocklist (cap at storage layer).
_MAX_EDGES_PER_ENTITY = 20


class Triple:
    """A (subject, predicate, object) triple extracted from a concept summary."""
    __slots__ = ('subject', 'predicate', 'object', 'concept_id', 'extraction_method')

    def __init__(self, subject: str, predicate: str, obj: str,
                 concept_id: str, extraction_method: str = 'regex'):
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.concept_id = concept_id
        self.extraction_method = extraction_method

    def __repr__(self):
        return f"Triple({self.subject!r}, {self.predicate!r}, {self.object!r})"


_FILLER_PREFIXES = [
    'city of ', 'country of ', 'continent of ',
    'sport of ', 'language of ', 'religion of ',
    'field of ', 'region of ', 'province of ',
]


def extract_triples_regex(concept_id: str, summary: str) -> list[Triple]:
    """Tier 1: Deterministic regex extraction using _PREDICATE_PATTERNS.

    For summaries matching structured patterns like:
      "Valmiki was performed by The Beatles"
    Extracts: Triple(subject="Valmiki", predicate="performer", object="The Beatles")

    Returns empty list for unstructured summaries (no pattern match).
    """
    triples: list[Triple] = []
    summary_lower = summary.lower()

    for pattern, label in _PREDICATE_PATTERNS:
        idx = summary_lower.find(pattern)
        if idx >= 0:
            # Subject = text before the pattern
            subject = summary[:idx].strip().rstrip('.,;:!? ')
            # Object = text after the pattern
            obj = summary[idx + len(pattern):].strip().rstrip('.,;:!? ')

            if (subject and obj
                    and len(subject) > 1 and len(obj) > 1
                    and len(subject) <= _MAX_ENTITY_LEN
                    and len(obj) <= _MAX_ENTITY_LEN
                    and subject.lower() not in _ENTITY_BLOCKLIST
                    and obj.lower() not in _ENTITY_BLOCKLIST):
                # Strip filler prefixes from object ("city of X" → "X")
                obj_clean = obj
                for fp in _FILLER_PREFIXES:
                    if obj.lower().startswith(fp):
                        obj_clean = obj[len(fp):].strip()
                        break

                # Primary triple with cleaned object
                triples.append(Triple(
                    subject=subject,
                    predicate=label,
                    obj=obj_clean,
                    concept_id=concept_id,
                    extraction_method='regex',
                ))
                # Also store with full object for exact-match retrieval
                if obj_clean != obj:
                    triples.append(Triple(
                        subject=subject,
                        predicate=label,
                        obj=obj,
                        concept_id=concept_id,
                        extraction_method='regex_full',
                    ))
            break  # First matching pattern wins (ordered by specificity)

    return triples
