"""Core taxonomy utilities — JSON-seed-based KA normalization.

DEBT-234: Extracted from app.cognitive.taxonomy so that app.storage modules
can normalize knowledge areas without importing the cognitive layer (Contract 2).

This module is intentionally SIMPLE and DB-free:
  - normalize_knowledge_area: JSON-seed exact/alias/fuzzy, no provisional KA creation
  - get_canonical_areas: JSON-seed frozenset, no DB query

The full dynamic implementation (provisional KA creation, DB sync) remains in
app.cognitive.taxonomy for callers that can reach the cognitive layer.
"""
import json
import logging
import os
from difflib import get_close_matches
from pathlib import Path

logger = logging.getLogger(__name__)

# Raised from 0.6 to match app.cognitive.taxonomy threshold (KA-ARCH-001)
_FUZZY_CUTOFF = 0.75

# Module-level cache — loaded once per process
_seed_areas: frozenset | None = None
_alias_map: dict | None = None


def _load_seed() -> tuple[frozenset, dict]:
    """Load canonical KA seed list and alias map from taxonomy JSON.

    Profile-aware: respects TAXONOMY_PROFILE env var (default: 'developer').
    Caches result in module-level globals after first call.
    """
    global _seed_areas, _alias_map
    if _seed_areas is not None:
        assert _alias_map is not None
        return _seed_areas, _alias_map

    taxonomy_profile = os.environ.get("TAXONOMY_PROFILE", "developer")
    config_dir = Path(__file__).parent.parent / "config"
    config_path = config_dir / f"taxonomy_{taxonomy_profile}.json"

    if not config_path.exists():
        config_path = config_dir / "taxonomy.json"
        logger.warning("taxonomy_utils: profile '%s' not found, using taxonomy.json", taxonomy_profile)

    try:
        with open(config_path) as f:
            data = json.load(f)
        _seed_areas = frozenset(data.get("canonical_areas", []))
        _alias_map = data.get("alias_map", {})
    except Exception as e:
        logger.warning("taxonomy_utils: failed to load seed config: %s", e)
        _seed_areas = frozenset()
        _alias_map = {}

    return _seed_areas, _alias_map


def get_canonical_areas() -> frozenset:
    """Return seed canonical knowledge area names from taxonomy JSON.

    DEBT-234: Storage-safe version — no DB dependency, no provisional KA creation.
    For the full dynamic set (includes established/mature from DB), use
    app.cognitive.taxonomy.get_canonical_areas().
    """
    areas, _ = _load_seed()
    return areas


def normalize_knowledge_area(area: str, strict: bool = False) -> tuple[str, str]:
    """Normalize a knowledge area string against the seed taxonomy.

    Steps:
      1. Exact match against seed canonical areas
      2. Alias map lookup
      3. Fuzzy match at 0.75 threshold
      4. Fallback: return area as-is (permissive) or 'general' (strict)

    DEBT-234: Storage-safe subset of app.cognitive.taxonomy.normalize_knowledge_area.
    Does NOT create provisional KAs in the DB — use the cognitive version for that.

    Returns:
        (normalized_area, source) where source is one of:
        'canonical', 'alias', 'fuzzy', 'novel' (permissive), 'default' (strict/empty)
    """
    areas, aliases = _load_seed()

    area_lower = area.lower().strip() if area else ""
    if not area_lower:
        return ("general", "default")

    # 1. Exact match
    if area_lower in areas:
        return (area_lower, "canonical")

    # 2. Alias lookup
    if area_lower in aliases:
        return (aliases[area_lower], "alias")

    # 3. Fuzzy match
    matches = get_close_matches(area_lower, areas, n=1, cutoff=_FUZZY_CUTOFF)
    if matches:
        return (matches[0], "fuzzy")

    # 4. Fallback: map to 'general' regardless of strict mode (REFLECT-025 / DEBT-234).
    # Storage-layer normalization must not create novel KA fragments — use 'general'
    # as the safe fallback. Callers needing pass-through behavior use the full
    # app.cognitive.taxonomy.normalize_knowledge_area instead.
    return ("general", "default")
