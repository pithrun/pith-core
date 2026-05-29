"""Unified entity detection for Pith.

Single module replacing:
- session.py:295 _NAMED_ENTITY_PATTERNS (7 patterns)
- fact_classifier.py:19 _HAS_NAMED_ENTITY (1 pattern)

Consumers:
- session.py P0-PRECISION evolution guard (via has_specific_entities)
- fact_classifier.py factual scoring (via has_specific_entities)

Ref: PREC-001 + KTA-EC4 spec, design doc v1.1
"""

import re as _re

# ── Unified Pattern Set ──────────────────────────
# Patterns 1-7: Proven (from session.py)
# Patterns 8-13: New PREC-001 (87% recall, 100% precision)

ENTITY_PATTERNS = [
    # 1. Dollar amounts ($4,500)
    _re.compile(r'\$[\d,]+'),
    # 2. Times (6:30 PM, 14:30)
    _re.compile(r'\d{1,2}:\d{2}\s*(?:[AaPp][Mm])?'),
    # 3. Numbers with units (16GB, 500mg, 85%)
    _re.compile(r'\d+\s*(?:GB|MB|TB|kg|lbs|oz|miles|km|mph|%)', _re.IGNORECASE),
    # 4. Multi-word proper nouns with optional lowercase connectors
    _re.compile(
        r'(?<![.!?]\s)[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,}'
        r'(?:\s+(?:[a-z]{1,3}\s+)?[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,})+'
    ),
    # 5. Paired proper nouns (Pilsner or Lager, Mac and Cheese)
    _re.compile(
        r'(?<=\s)[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,}'
        r'(?:\s+(?:or|and|&)\s+[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,})'
    ),
    # 6. ISO dates (2026-03-17)
    _re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    # 7. Named dates (March 25)
    _re.compile(
        r'(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2}',
        _re.IGNORECASE,
    ),
    # 8. Honorifics (Dr. Chen, Prof. Williams, Rev. Jackson)
    _re.compile(r'\b(?:Dr|Prof|Rev|Sr|Jr|Mrs|Mr|Ms)\.\s+[A-Z][a-z]+'),
    # 9. Medication dosages (500mg, 10mg, 0.5mg)
    _re.compile(r'\b\d+(?:\.\d+)?\s*mg\b', _re.IGNORECASE),
    # 10. Drug name suffixes
    _re.compile(
        r'\b\w+(?:statin|pril|cillin|mycin|formin|sartan|olol|dipine|'
        r'azole|setron|lukast|gliptin|tidine|prazole)\b',
        _re.IGNORECASE,
    ),
    # 11. Named entity context clues ("named Marcus", "called Jessica")
    _re.compile(r'\b(?:named|called|name\s+is)\s+[A-Z][a-z]+', _re.IGNORECASE),
    # 12. Address specifics (apartment 4B, suite 200, unit 3)
    _re.compile(r'\b(?:apartment|apt|suite|unit|room|floor)\s+\w+', _re.IGNORECASE),
    # 13. Place + institution suffix (Valley Medical, Goldman Sachs)
    _re.compile(
        r'[A-Z][a-z]+\s+(?:Medical|Hospital|University|College|Elementary|'
        r'Middle|High\s+School|Academy|Institute|Sachs|Stanley|Suisse|'
        r'Lynch|Fargo|Securities)',
        _re.IGNORECASE,
    ),
    # 14. Known entities (carried from fact_classifier.py hardcoded list)
    _re.compile(
        r'\b(?:Andrew|Pith|Claude|Anthropic|Google|GitHub|Slack|'
        r'Linear|Notion)\b'
    ),
]


def has_specific_entities(text: str) -> bool:
    """Check if text contains named entities or consumer-specific details.

    Used by: evolution specificity guard (session.py), fact classifier.
    Returns True if any pattern matches, indicating text contains
    specifics that should be preserved during evolution.
    """
    if not text:
        return False
    return any(p.search(text) for p in ENTITY_PATTERNS)


def detect_entity_signals(text: str) -> dict:
    """Rich entity analysis returning signal breakdown.

    Returns: {"has_entities": bool, "signals": list[str], "entity_count": int}
    """
    if not text:
        return {"has_entities": False, "signals": [], "entity_count": 0}
    signals = []
    for p in ENTITY_PATTERNS:
        if p.search(text):
            signals.append(p.pattern[:40])
    return {
        "has_entities": len(signals) > 0,
        "signals": signals,
        "entity_count": len(signals),
    }
