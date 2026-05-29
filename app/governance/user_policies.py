"""User Policies — CRUD operations for user-configurable pith rules.

Phase 3 completion: Allows pith owners to define custom retention,
privacy, and behavior policies that the PolicyEngine enforces.

Per spec: ORIENTATION_V2_AND_PHASE3_COMPLETION_SPEC.md (Section B.1)

Policy Schema:
  retention: action={"max_age_days": int}, condition=optional{"concept_type": str}
  privacy:   action={"pattern": str, "redact_with": str}, condition=optional{"knowledge_area": str}
  behavior:  action={"directive": str}, condition=optional{"context": str}
"""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.datetime_utils import _utc_now_iso
from app.storage import _db

logger = logging.getLogger(__name__)

VALID_POLICY_TYPES = {"retention", "privacy", "behavior"}


@dataclass
class UserPolicy:
    """A user-defined policy rule."""

    id: str
    policy_type: str
    rule: str
    action: dict[str, Any]
    condition: dict[str, Any] | None = None
    enabled: bool = True
    priority: int = 50
    created_at: str = ""
    updated_at: str = ""


def _validate_json_field(obj: Any, field_name: str) -> None:
    """Validate that a field is JSON-serializable before DB insert."""
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Field '{field_name}' contains non-JSON-serializable data: {e}")


def _deserialize_policy_row(row) -> UserPolicy:
    """Convert a DB row tuple to a UserPolicy object."""
    return UserPolicy(
        id=row[0],
        policy_type=row[1],
        rule=row[2],
        condition=json.loads(row[3]) if row[3] else None,
        action=json.loads(row[4]),
        enabled=bool(row[5]),
        priority=row[6],
        created_at=row[7],
        updated_at=row[8],
    )


def _invalidate_policy_cache() -> None:
    """Invalidate the policy cache after modifications."""
    try:
        from app.governance.policy_cache import PolicyCache

        PolicyCache._cache.clear()
        logger.debug("Policy cache invalidated after modification")
    except Exception:
        pass  # Cache invalidation is best-effort


def create_policy(policy_type: str, rule: str, action: dict, condition: dict = None, priority: int = 50) -> UserPolicy:
    """Create a new user policy.

    Args:
        policy_type: One of 'retention', 'privacy', 'behavior'
        rule: Human-readable description of what this policy does
        action: What to do when triggered (JSON-serializable dict)
        condition: Optional filter for when to apply (JSON-serializable dict)
        priority: Higher = evaluated first (default 50)

    Returns:
        The created UserPolicy

    Raises:
        ValueError: If policy_type is invalid
    """
    if policy_type not in VALID_POLICY_TYPES:
        raise ValueError(f"Invalid policy_type '{policy_type}'. Must be one of: {VALID_POLICY_TYPES}")

    # Validate JSON-serializability before DB insert (review fix: early validation)
    _validate_json_field(action, "action")
    if condition:
        _validate_json_field(condition, "condition")

    now = _utc_now_iso()
    policy_id = f"pol_{uuid.uuid4().hex}"

    with _db() as conn:
        conn.execute(
            """
            INSERT INTO user_policies (id, policy_type, rule, condition, action, enabled, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
            (
                policy_id,
                policy_type,
                rule,
                json.dumps(condition) if condition else None,
                json.dumps(action),
                priority,
                now,
                now,
            ),
        )
        conn.commit()

    _invalidate_policy_cache()
    logger.info(f"Created user policy {policy_id}: {policy_type} — {rule[:60]}")
    return UserPolicy(
        id=policy_id,
        policy_type=policy_type,
        rule=rule,
        action=action,
        condition=condition,
        enabled=True,
        priority=priority,
        created_at=now,
        updated_at=now,
    )


def list_policies(policy_type: str = None, include_disabled: bool = False) -> list[UserPolicy]:
    """List user policies, optionally filtered by type.

    Args:
        policy_type: Filter to specific type (None = all)
        include_disabled: Include disabled policies (default False)

    Returns:
        List of UserPolicy objects, ordered by priority DESC
    """
    conditions = []
    params = []

    if not include_disabled:
        conditions.append("enabled = 1")
    if policy_type:
        if policy_type not in VALID_POLICY_TYPES:
            raise ValueError(f"Invalid policy_type '{policy_type}'")
        conditions.append("policy_type = ?")
        params.append(policy_type)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, policy_type, rule, condition, action, enabled, priority, created_at, updated_at
            FROM user_policies
            {where}
            ORDER BY priority DESC, created_at DESC
        """,
            tuple(params),
        ).fetchall()

    return [_deserialize_policy_row(row) for row in rows]


def get_policy(policy_id: str) -> UserPolicy | None:
    """Get a single policy by ID."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id, policy_type, rule, condition, action, enabled, priority, created_at, updated_at FROM user_policies WHERE id = ?",
            (policy_id,),
        ).fetchone()

    if not row:
        return None

    return _deserialize_policy_row(row)


def update_policy(
    policy_id: str,
    rule: str = None,
    action: dict = None,
    condition: dict = None,
    priority: int = None,
    enabled: bool = None,
) -> UserPolicy | None:
    """Update an existing policy. Only provided fields are changed.

    Returns:
        Updated UserPolicy, or None if not found
    """
    existing = get_policy(policy_id)
    if not existing:
        return None

    now = _utc_now_iso()
    updates = ["updated_at = ?"]
    params = [now]

    if rule is not None:
        updates.append("rule = ?")
        params.append(rule)
    if action is not None:
        _validate_json_field(action, "action")
        updates.append("action = ?")
        params.append(json.dumps(action))
    if condition is not None:
        _validate_json_field(condition, "condition")
        updates.append("condition = ?")
        params.append(json.dumps(condition))
    if priority is not None:
        updates.append("priority = ?")
        params.append(priority)
    if enabled is not None:
        updates.append("enabled = ?")
        params.append(1 if enabled else 0)

    params.append(policy_id)

    with _db() as conn:
        conn.execute(
            f"""
            UPDATE user_policies SET {", ".join(updates)} WHERE id = ?
        """,
            tuple(params),
        )
        conn.commit()

    _invalidate_policy_cache()
    logger.info(f"Updated user policy {policy_id}")
    return get_policy(policy_id)


def delete_policy(policy_id: str) -> bool:
    """Soft-delete a policy (sets enabled=0).

    Returns:
        True if policy existed and was disabled, False if not found
    """
    existing = get_policy(policy_id)
    if not existing:
        return False

    with _db() as conn:
        conn.execute("UPDATE user_policies SET enabled = 0, updated_at = ? WHERE id = ?", (_utc_now_iso(), policy_id))
        conn.commit()

    _invalidate_policy_cache()
    logger.info(f"Soft-deleted user policy {policy_id}")
    return True
