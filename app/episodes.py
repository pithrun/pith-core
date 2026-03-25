"""Episode recording and PII-safe retention.

Memory Integrity Spec v1.2, §5.2.5 [resolves C12]

Episodes capture metadata about each conversation turn (concept IDs created/evolved,
intent summary, classification) without storing raw text indefinitely.

Raw user_message and assistant_response are stored for 30 days then purged.
Metadata is retained permanently for audit trail.
"""

import logging
import uuid
from datetime import timedelta

from app.config import FEATURE_FLAGS
from app.datetime_utils import _utc_now, _utc_now_iso
from app.storage import get_db_connection

logger = logging.getLogger(__name__)


def record_episode(
    session_id: str,
    turn_number: int,
    intent_summary: str = "",
    classification: str = "",
    extracted_concept_ids: list[str] | None = None,
    concept_changes: list[dict] | None = None,
    raw_user_message: str | None = None,
    raw_assistant_response: str | None = None,
) -> str | None:
    """Record a conversation episode with metadata.

    Returns episode ID if recorded, None if feature is off.
    """
    if not FEATURE_FLAGS.get("EPISODES_ENABLED", False):
        return None

    import json

    episode_id = f"ep_{uuid.uuid4().hex[:12]}"
    now = _utc_now_iso()

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO episodes
            (id, session_id, turn_number, extracted_concept_ids, concept_changes,
             intent_summary, classification, world_timestamp, created_at,
             raw_user_message, raw_assistant_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                episode_id,
                session_id,
                turn_number,
                json.dumps(extracted_concept_ids or []),
                json.dumps(concept_changes or []),
                intent_summary[:500],  # Cap summary length
                classification,
                now,
                now,
                raw_user_message,
                raw_assistant_response,
            ),
        )
        conn.commit()
        logger.debug("Recorded episode %s for session %s turn %d", episode_id, session_id, turn_number)
        return episode_id
    except Exception as e:
        logger.warning("Episode recording failed (non-fatal): %s", e)
        return None


def update_episode_metadata(
    episode_id: str,
    intent_summary: str = "",
    classification: str = "",
    temporal_filter_outcome: str = "",
) -> bool:
    """Backfill episode metadata after server-side classification.

    Called after S2.5 question classification produces results (INFRA-005).
    Only updates non-empty fields to avoid overwriting client hints.

    Args:
        temporal_filter_outcome: JSON string recording filter decision (RETRIEVAL-029).
            Format: {"action":"filtered|fallback|skipped","before":N,"after":N}

    Returns True if update succeeded, False otherwise.
    """
    conn = get_db_connection()
    try:
        updates = []
        params = []

        if intent_summary:
            updates.append("intent_summary = ?")
            params.append(intent_summary[:500])

        if classification:
            updates.append("classification = ?")
            params.append(classification[:200])

        if temporal_filter_outcome:
            updates.append("temporal_filter_outcome = ?")
            params.append(temporal_filter_outcome[:500])

        if not updates:
            return True  # Nothing to update

        params.append(episode_id)
        conn.execute(
            f"UPDATE episodes SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()
        logger.debug("Updated episode %s metadata: classification=%s", episode_id, classification)
        return True
    except Exception as e:
        logger.warning("Episode metadata update failed (non-fatal): %s", e)
        return False


def purge_expired_raw_text(retention_days: int = 30) -> int:
    """Purge raw text from episodes older than retention period.

    Sets raw_user_message and raw_assistant_response to NULL,
    records raw_purged_at timestamp. Returns count of purged episodes.

    Run as periodic maintenance (daily recommended).
    """
    conn = get_db_connection()
    cutoff = (_utc_now() - timedelta(days=retention_days)).isoformat()
    now = _utc_now_iso()

    try:
        cursor = conn.execute(
            """
            UPDATE episodes
            SET raw_user_message = NULL,
                raw_assistant_response = NULL,
                raw_purged_at = ?
            WHERE created_at < ?
              AND raw_purged_at IS NULL
              AND (raw_user_message IS NOT NULL OR raw_assistant_response IS NOT NULL)
        """,
            (now, cutoff),
        )
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info("Purged raw text from %d episodes (retention=%dd)", count, retention_days)
        return count
    except Exception as e:
        logger.warning("Episode purge failed (non-fatal): %s", e)
        return 0


# =============================================================================
# Episodes Retention Job (CM-M4, §5.4.5)
# =============================================================================
# Tiered retention: 30-day hot → 180-day archive → 365-day purge.
# Hot: Full records with raw text (purge_expired_raw_text handles this).
# Archive: Metadata only, raw text already purged.
# Purge: Completely removed after 1 year.

EPISODE_RETENTION = {
    "hot_days": 30,  # Full records in episodes table
    "archive_days": 180,  # Metadata only (raw text purged)
    "purge_days": 365,  # Completely removed after 1 year
}


def archive_old_episodes(archive_days: int = None) -> int:
    """Move episodes older than archive_days to lightweight state.

    Archives episodes by ensuring raw text is purged and marking
    them as archived. This is idempotent — already-archived episodes
    are skipped.

    Returns count of newly archived episodes.
    """
    if archive_days is None:
        archive_days = EPISODE_RETENTION["archive_days"]

    conn = get_db_connection()
    cutoff = (_utc_now() - timedelta(days=archive_days)).isoformat()
    now = _utc_now_iso()

    try:
        # Ensure raw text is purged for old episodes (belt + suspenders)
        cursor = conn.execute(
            """
            UPDATE episodes
            SET raw_user_message = NULL,
                raw_assistant_response = NULL,
                raw_purged_at = COALESCE(raw_purged_at, ?)
            WHERE created_at < ?
              AND (raw_user_message IS NOT NULL OR raw_assistant_response IS NOT NULL)
        """,
            (now, cutoff),
        )
        archived_count = cursor.rowcount

        conn.commit()
        if archived_count > 0:
            logger.info("Archived %d episodes (older than %dd)", archived_count, archive_days)
        return archived_count
    except Exception as e:
        logger.warning("Episode archival failed (non-fatal): %s", e)
        return 0


def purge_old_episodes(purge_days: int = None) -> int:
    """Permanently delete episodes older than purge_days.

    This is the final tier — episodes are completely removed.
    Returns count of deleted episodes.
    """
    if purge_days is None:
        purge_days = EPISODE_RETENTION["purge_days"]

    conn = get_db_connection()
    cutoff = (_utc_now() - timedelta(days=purge_days)).isoformat()

    try:
        cursor = conn.execute(
            """
            DELETE FROM episodes WHERE created_at < ?
        """,
            (cutoff,),
        )
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info("Purged %d episodes (older than %dd)", count, purge_days)
        return count
    except Exception as e:
        logger.warning("Episode purge (full delete) failed (non-fatal): %s", e)
        return 0


def run_episode_retention_job() -> dict:
    """Run the full episode retention pipeline (CM-M4).

    Executes all three tiers in order:
    1. Raw text purge (30 days)
    2. Archive (180 days)
    3. Full purge (365 days)

    Returns summary of actions taken.
    Feature-gated by EPISODES_ENABLED.
    """
    if not FEATURE_FLAGS.get("EPISODES_ENABLED", False):
        return {"status": "disabled", "raw_purged": 0, "archived": 0, "deleted": 0}

    raw_purged = purge_expired_raw_text(EPISODE_RETENTION["hot_days"])
    archived = archive_old_episodes(EPISODE_RETENTION["archive_days"])
    deleted = purge_old_episodes(EPISODE_RETENTION["purge_days"])

    summary = {
        "status": "completed",
        "raw_purged": raw_purged,
        "archived": archived,
        "deleted": deleted,
        "retention_config": EPISODE_RETENTION,
    }
    logger.info("Episode retention job complete: %s", summary)
    return summary
