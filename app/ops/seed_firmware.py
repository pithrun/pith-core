"""Firmware seeder for Pith (P0-5 / P0-7).

Static, developer-controlled operational knowledge. ROM model:
- Only this script writes to the firmware table
- Runs on server startup (called from server.py startup_event)
- Idempotent: always overwrites all entries for current FIRMWARE_VERSION
- Tracks firmware_version in metadata table for upgrade detection

Firmware is physically isolated from concepts:
- Separate table (no TF-IDF index, no associations, no reflection/decay)
- Injected into conversation_turn via S4.7 as [FIRMWARE] prefixed entries
- Never mutated at runtime by agents or users

To add/modify firmware:
1. Add/edit entries in FIRMWARE_CATALOG below
2. Bump FIRMWARE_VERSION
3. Restart server (seed runs on startup)
"""

import logging

from app.storage import connection as _conn
from app.storage import get_metadata, save_firmware, set_metadata

logger = logging.getLogger(__name__)

# Bump this when firmware content changes.
# seed_firmware() uses this to detect whether re-seeding is needed.
FIRMWARE_VERSION = "1.2.0"

# --- Firmware Catalog ---
# Each entry: (id, summary, category)
# id: stable identifier (snake_case, descriptive)
# summary: the operational knowledge injected into every conversation_turn
# category: grouping for organization (e.g., "tool_routing", "protocol", "safety")
FIRMWARE_CATALOG = [
    # -- Protocol essentials --
    (
        "pith_conversation_turn_first",
        "ALWAYS call pith_conversation_turn BEFORE composing any substantive "
        "response. This retrieves essential context and auto-learns from the "
        "previous exchange. Without it, responses will be generic and amnesiac.",
        "protocol",
    ),
    (
        "extracted_concepts_required",
        "ALWAYS include extracted_concepts_json in pith_conversation_turn calls "
        "after the first exchange. Format: JSON array of 1-5 concept objects with "
        "summary, confidence, knowledge_area, evidence, and concept_type fields. "
        "Send '[]' for casual/trivial exchanges — never invent filler concepts.",
        "protocol",
    ),
]


def _delete_stale_firmware(active_ids: set[str]) -> int:
    """Delete firmware entries no longer present in the shipped catalog."""
    with _conn._db(operation="seed_firmware_cleanup") as conn:
        if active_ids:
            placeholders = ",".join("?" for _ in active_ids)
            cursor = conn.execute(
                f"DELETE FROM firmware WHERE id NOT IN ({placeholders})",
                tuple(sorted(active_ids)),
            )
        else:
            cursor = conn.execute("DELETE FROM firmware")
        return cursor.rowcount


def seed_firmware() -> dict:
    """Seed firmware table with current catalog.

    Returns dict with seeding results for logging.
    Idempotent: safe to call on every startup.
    """
    current_version = get_metadata("firmware_version")
    active_ids = {fw_id for fw_id, _, _ in FIRMWARE_CATALOG}

    if current_version == FIRMWARE_VERSION:
        stale_deleted = _delete_stale_firmware(active_ids)
        logger.info(
            f"seed_firmware: firmware v{FIRMWARE_VERSION} already seeded, "
            f"skipping ({len(FIRMWARE_CATALOG)} entries, deleted {stale_deleted} stale)"
        )
        return {
            "action": "skipped",
            "version": FIRMWARE_VERSION,
            "reason": "already_seeded",
            "entry_count": len(FIRMWARE_CATALOG),
            "stale_deleted": stale_deleted,
        }

    logger.info(
        f"seed_firmware: seeding firmware v{FIRMWARE_VERSION} "
        f"(previous: {current_version or 'none'}, "
        f"{len(FIRMWARE_CATALOG)} entries)"
    )

    seeded = 0
    for fw_id, summary, category in FIRMWARE_CATALOG:
        try:
            save_firmware(
                firmware_id=fw_id,
                summary=summary,
                category=category,
                firmware_version=FIRMWARE_VERSION,
            )
            seeded += 1
        except Exception as e:
            logger.error(f"seed_firmware: failed to seed '{fw_id}': {e}")

    stale_deleted = _delete_stale_firmware(active_ids)

    # Update version tracker
    set_metadata("firmware_version", FIRMWARE_VERSION)

    logger.info(
        f"seed_firmware: seeded {seeded}/{len(FIRMWARE_CATALOG)} entries, "
        f"deleted {stale_deleted} stale"
    )

    return {
        "action": "seeded",
        "version": FIRMWARE_VERSION,
        "previous_version": current_version,
        "entries_seeded": seeded,
        "entries_total": len(FIRMWARE_CATALOG),
        "stale_deleted": stale_deleted,
    }
