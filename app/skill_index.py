"""ARCH-001: Model-Agnostic Skill Routing Index

Provides pith-native skill recommendations via conversation_turn.
Decouples skill triggering from Claude-specific description matching.

Design: Minimal stub — loads skill registry from DB metadata, returns
matching skill paths based on keyword overlap with user message.
Full TF-IDF/embedding matching deferred until non-Claude demand exists.

Zero non-Claude models use Pith today (Mar 2026). This is the hook point
for when they do — the recommended_skills field on ConversationTurnResponse
is the contract.
"""

import logging

logger = logging.getLogger(__name__)


# Skill registry: list of {name, description, path, keywords}
# Populated by skill deployer (SKILL-004) or manual registration.
# Stored in DB metadata table as JSON under key "skill_registry".
_SKILL_REGISTRY_KEY = "skill_registry"
_cached_registry: list | None = None


def _load_registry() -> list:
    """Load skill registry from DB metadata. Cached after first load."""
    global _cached_registry
    if _cached_registry is not None:
        return _cached_registry

    try:
        import json

        from app.storage import get_metadata

        raw = get_metadata(_SKILL_REGISTRY_KEY)
        if raw:
            _cached_registry = json.loads(raw)
        else:
            _cached_registry = []
    except Exception as e:
        logger.debug(f"ARCH-001: Could not load skill registry: {e}")
        _cached_registry = []

    return _cached_registry


def clear_cache():
    """Clear cached registry (call after skill registration changes)."""
    global _cached_registry
    _cached_registry = None


def recommend_skills(message: str, max_results: int = 3) -> list[str]:
    """Recommend skill file paths based on user message.

    Returns list of skill file paths the caller should read before responding.
    Empty list = no skill recommendations (most common case).

    Algorithm: Simple keyword overlap between message and skill keywords.
    Intentionally minimal — full semantic matching is premature until
    non-Claude consumers exist.
    """
    registry = _load_registry()
    if not registry:
        return []

    message_words = set(message.lower().split())
    if not message_words:
        return []

    scored = []
    for skill in registry:
        keywords = set(str(k).lower() for k in skill.get("keywords", []))
        if not keywords:
            continue
        overlap = len(message_words & keywords)
        if overlap > 0:
            scored.append((overlap, skill.get("path", "")))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in scored[:max_results] if path]


def register_skill(name: str, description: str, path: str, keywords: list[str]) -> bool:
    """Register a skill in Pith's skill index.

    Called by skill deployer (SKILL-004) during pith_deploy_skills.
    Persists to DB metadata for cross-session availability.
    """
    try:
        import json

        from app.storage import set_metadata

        registry = _load_registry()

        # Upsert: replace if name exists, append if new
        registry = [s for s in registry if s.get("name") != name]
        registry.append(
            {
                "name": name,
                "description": description,
                "path": path,
                "keywords": keywords,
            }
        )

        set_metadata(_SKILL_REGISTRY_KEY, json.dumps(registry))
        clear_cache()  # Invalidate cache after write
        logger.info(f"ARCH-001: Registered skill '{name}' with {len(keywords)} keywords")
        return True
    except Exception as e:
        logger.error(f"ARCH-001: Failed to register skill '{name}': {e}")
        return False
