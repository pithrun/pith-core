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
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# Skill registry: list of {name, description, path, keywords, registered_at, last_recommended, last_fired, recommend_count, fire_count}
# Populated by skill deployer (SKILL-004) or manual registration.
# Stored in DB metadata table as JSON under key "skill_registry".
_SKILL_REGISTRY_KEY = "skill_registry"
_cached_registry: list | None = None


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


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
                "registered_at": _now_iso(),
                "last_recommended": None,
                "last_fired": None,
                "recommend_count": 0,
                "fire_count": 0,
            }
        )

        set_metadata(_SKILL_REGISTRY_KEY, json.dumps(registry))
        clear_cache()  # Invalidate cache after write
        logger.info(f"ARCH-001: Registered skill '{name}' with {len(keywords)} keywords")
        return True
    except Exception as e:
        logger.error(f"ARCH-001: Failed to register skill '{name}': {e}")
        return False


def record_skill_event(skill_name: str, event_type: str):
    """Record a skill firing or recommendation event."""
    for skill in _load_registry():
        if skill.get("name") == skill_name:
            if event_type == "fire":
                skill["last_fired"] = _now_iso()
                skill["fire_count"] = skill.get("fire_count", 0) + 1
            elif event_type == "recommend":
                skill["last_recommended"] = _now_iso()
                skill["recommend_count"] = skill.get("recommend_count", 0) + 1
            break
    
    # Persist updated registry
    try:
        import json
        from app.storage import set_metadata
        registry = _load_registry()
        set_metadata(_SKILL_REGISTRY_KEY, json.dumps(registry))
        clear_cache()
    except Exception as e:
        logger.debug(f"ARCH-001: Failed to record skill event: {e}")


def get_skill_health() -> list:
    """Return health metrics for all registered skills."""
    now = datetime.now(timezone.utc)
    results = []
    for skill in _load_registry():
        fire_count = skill.get("fire_count", 0)
        rec_count = skill.get("recommend_count", 0)
        last_fired = skill.get("last_fired")
        registered = skill.get("registered_at")
        
        # Calculate staleness
        staleness_days = None
        if last_fired:
            try:
                last = datetime.fromisoformat(last_fired)
                staleness_days = (now - last).total_seconds() / 86400
            except (ValueError, TypeError):
                pass
        
        # Determine status
        if fire_count == 0:
            status = "never_fired"
        elif staleness_days is not None and staleness_days > 14:
            status = "stale"
        elif staleness_days is not None and staleness_days > 7:
            status = "dormant"
        else:
            status = "active"
        
        results.append({
            "name": skill.get("name"),
            "registered_at": registered,
            "last_fired": last_fired,
            "fire_count": fire_count,
            "recommend_count": rec_count,
            "fire_rate_pct": round(fire_count / rec_count * 100, 1) if rec_count > 0 else 0,
            "staleness_days": round(staleness_days, 1) if staleness_days is not None else None,
            "status": status,
        })
    return results
