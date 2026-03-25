"""Non-Lossy Evolution — version-chain-based concept evolution.

Memory Integrity Spec v1.2, §5.2.4, §5.9.3, §5.9.4:
Instead of overwriting concept data, creates a NEW version and marks the
old version as superseded. This preserves full evolution history for:
  - Temporal snapshot queries (belief state at any point in time)
  - Forensic auditing (what did we believe and when?)
  - Rollback (if a bad evolution is detected, revert to prior version)

Schema additions (GOV-009):
  - is_current: 1 for latest version, 0 for superseded
  - superseded_at: timestamp when superseded
  - superseded_by: concept_id of the new version
  - version_chain_head: points to the original concept in the chain

Feature-gated by NONLOSSY_EVOLUTION_ENABLED.
"""

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta  # DEBT-192: hoisted from rate-limiter if-block

from app.datetime_utils import _utc_now, _utc_now_iso
from app.storage import _KA_SENTINELS

logger = logging.getLogger(__name__)


def _dedup_list(items: list) -> list:
    """Deduplicate a list that may contain unhashable types (dicts).

    Uses JSON serialization as a stable key for dict items.
    Preserves insertion order. Falls back to identity comparison for
    items that aren't JSON-serializable.
    """
    seen = set()
    result = []
    for item in items:
        try:
            # Hashable items (str, int, tuple) — use directly
            key = item
            if isinstance(item, dict):
                # Dicts aren't hashable — use sorted JSON as key
                key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(item)
        except TypeError:
            # Fallback for truly exotic unhashable types
            if item not in result:
                result.append(item)
    return result


def evolve_concept_nonlossy(
    concept_id: str,
    new_data: dict,
    conn: sqlite3.Connection,
) -> str | None:
    """Create a new version of a concept, marking old as superseded.

    Args:
        concept_id: ID of the concept to evolve
        new_data: Dict with updated fields (summary, confidence, etc.)
        conn: SQLite connection (caller manages transaction)

    Returns:
        The concept_id (same ID, new version) or None if concept not found.
    """
    from app.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("NONLOSSY_EVOLUTION_ENABLED", False):
        return None

    # Load the current version
    row = conn.execute(
        "SELECT id, version, data, created_at, updated_at FROM concepts WHERE id = ? AND is_current = 1",
        (concept_id,),
    ).fetchone()

    if not row:
        logger.warning("evolve_concept_nonlossy: concept %s not found or not current", concept_id)
        return None

    # MATURITY-003: Rate limiter — prevent rapid double-evolution race conditions
    # (audit found 10 pairs within 60s, some within 80ms)
    if row[4]:  # updated_at exists
        try:
            last_update = datetime.fromisoformat(row[4].replace("Z", "+00:00"))
            now = datetime.now(UTC)
            elapsed = (now - last_update).total_seconds()
            if elapsed < 1.0:
                logger.info(
                    "evolve_concept_nonlossy: rate-limited %s (%.2fs since last update)",
                    concept_id,
                    elapsed,
                )
                return None
        except (ValueError, TypeError):
            pass  # If timestamp parsing fails, proceed normally

    old_version = row[1]
    try:
        old_data = json.loads(row[2]) if row[2] else {}
    except (json.JSONDecodeError, TypeError):
        old_data = {}

    # FIX-3: Guard against trivial evolutions (mirrors should_evolve() logic)
    # Without this, ALL evolution requests create version churn unconditionally.
    from app.config import MIN_CONFIDENCE_CHANGE, MIN_EVIDENCE_CHANGE

    _conf_change = new_data.get("confidence_change", 0)
    _new_evidence = new_data.get("new_evidence", [])
    _new_summary = new_data.get("summary")
    _new_hypotheses = new_data.get("new_hypotheses", [])
    _new_concept_type = new_data.get("new_concept_type") or new_data.get("concept_type")

    _has_significant_confidence = abs(_conf_change) >= MIN_CONFIDENCE_CHANGE
    _has_new_evidence = len(_new_evidence) >= MIN_EVIDENCE_CHANGE
    _has_new_summary = bool(_new_summary) and _new_summary != old_data.get("summary", "")
    _has_new_hypotheses = bool(_new_hypotheses)
    _has_reclassification = bool(_new_concept_type) and _new_concept_type != old_data.get("concept_type", "observation")

    if not any(
        [_has_significant_confidence, _has_new_evidence, _has_new_summary, _has_new_hypotheses, _has_reclassification]
    ):
        logger.debug(
            "evolve_concept_nonlossy: skipping %s — insufficient changes "
            "(conf_change=%.3f, evidence=%d, summary_changed=%s)",
            concept_id,
            _conf_change,
            len(_new_evidence),
            _has_new_summary,
        )
        return None

    # Compute new version number
    try:
        ver_num = int(old_version.replace("v", "")) + 1
    except (ValueError, AttributeError):
        ver_num = 2
    new_version = f"v{ver_num}"

    # Determine version chain head
    chain_head_row = conn.execute(
        "SELECT version_chain_head FROM concepts WHERE id = ? AND is_current = 1",
        (concept_id,),
    ).fetchone()
    chain_head = chain_head_row[0] if chain_head_row and chain_head_row[0] else concept_id

    now = _utc_now_iso()

    # Archive the old version to concept_versions_archive (preserves history)
    # PK on concepts table is `id` alone, so we can't INSERT a second row.
    # Instead: archive old → UPDATE existing row in-place.
    try:
        conn.execute(
            """INSERT OR IGNORE INTO concept_versions_archive
               (id, version, summary, data, created_at, superseded_at, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (concept_id, old_version, old_data.get("summary", ""), json.dumps(old_data), row[3], now, now),
        )
    except sqlite3.OperationalError:
        # Archive table may not exist yet — log and continue
        logger.warning("concept_versions_archive table missing, skipping archive for %s", concept_id)

    # Merge old data with new data (lists are appended, not replaced)
    merged_data = {**old_data}
    for key, value in new_data.items():
        if value is not None:
            if key == "new_evidence":
                # Append new evidence to existing evidence list
                # Use JSON-based dedup to handle unhashable types (dicts)
                existing = merged_data.get("evidence", [])
                merged_data["evidence"] = _dedup_list(existing + value)
            elif key == "new_signals":
                # Append new signals to existing signals list
                # Use JSON-based dedup to handle unhashable types (dicts)
                existing = merged_data.get("signals", [])
                merged_data["signals"] = _dedup_list(existing + value)
            elif key == "new_hypotheses":
                # Append new hypotheses to existing hypotheses list
                existing = merged_data.get("hypotheses", [])
                merged_data["hypotheses"] = existing + value
            elif key == "confidence_change":
                pass  # Handled separately below
            else:
                merged_data[key] = value

    # Build the new version's summary (use new if provided, else keep old)
    old_summary = old_data.get("summary", "")
    new_summary = new_data.get("summary") or old_summary

    # Retrieval Defense W9: Evolution content guard (two layers)
    try:
        from app.config import FEATURE_FLAGS

        if FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False) and new_data.get("summary"):
            from app.policy import SUMMARY_OVERLAP_THRESHOLD, check_content_policy, compute_summary_overlap

            # Layer 1: Content policy regex on new summary
            if check_content_policy(new_summary):
                logger.warning("W9-L1: Content policy violation in evolution of %s — quarantining", concept_id)
                merged_data["maturity"] = "QUARANTINED"

            # Layer 2: Substantial rewrite detection via word overlap
            # W9 fix: Layer 2 must NOT overwrite Layer 1's QUARANTINED verdict.
            # QUARANTINED (content policy violation) takes priority over PROVISIONAL (rewrite detection).
            if old_summary and new_summary != old_summary:
                overlap = compute_summary_overlap(old_summary, new_summary)
                if overlap < SUMMARY_OVERLAP_THRESHOLD:
                    if merged_data.get("maturity") == "QUARANTINED":
                        logger.info(
                            "W9-L2: Substantial rewrite detected for %s (overlap=%.2f < %.2f) — "
                            "but Layer 1 already set QUARANTINED, preserving higher severity",
                            concept_id,
                            overlap,
                            SUMMARY_OVERLAP_THRESHOLD,
                        )
                    else:
                        logger.warning(
                            "W9-L2: Substantial rewrite detected for %s (overlap=%.2f < %.2f) — "
                            "downgrading to PROVISIONAL",
                            concept_id,
                            overlap,
                            SUMMARY_OVERLAP_THRESHOLD,
                        )
                        merged_data["maturity"] = "PROVISIONAL"
    except Exception as e:
        logger.warning("W9: Evolution content guard failed for %s: %s", concept_id, e)

    # Handle confidence_change (bridge from ConceptEvolution) vs absolute confidence
    if "confidence_change" in new_data:
        old_conf = old_data.get("confidence", 0.5)
        new_confidence = min(1.0, max(0.0, old_conf + new_data["confidence_change"]))
    else:
        new_confidence = new_data.get("confidence") or old_data.get("confidence", 0.5)
    # STABILITY-026: M3 compliance — cap confidence for PSIS-quarantined concepts
    from app.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
    _merged_evidence = merged_data.get("evidence", [])
    if PSIS_QUARANTINE_EVIDENCE_MARKER in _merged_evidence:
        new_confidence = min(new_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)
    new_concept_type = new_data.get("concept_type") or old_data.get("concept_type", "observation")

    # Stability growth: each evolution increases stability (concept is being refined)
    old_stability = old_data.get("stability", 0.5)
    new_stability = min(1.0, old_stability + 0.05)

    # DATA-045: Detect archival intent in summary — deactivate zombie concepts
    _is_archival = new_summary.strip().upper().startswith("ARCHIVED:")  # DATA-046: strip leading whitespace
    if _is_archival:
        logger.info("DATA-045: Summary starts with 'ARCHIVED:' for %s — marking is_current=0", concept_id)

    # Sync column values into the JSON blob so load_concept can reconstruct
    merged_data["id"] = concept_id
    merged_data["version"] = new_version
    merged_data["summary"] = new_summary
    merged_data["confidence"] = new_confidence
    merged_data["stability"] = new_stability
    merged_data["concept_type"] = new_concept_type
    merged_data["created_at"] = old_data.get("created_at", row[3])
    merged_data["updated_at"] = now
    merged_data["supersedes"] = old_version

    # Retrieval Defense W2 (evolve): Re-classify epistemic network with merged evidence
    try:
        from app.epistemic import classify_and_annotate_concept

        classified = classify_and_annotate_concept(merged_data)
        if classified:
            logger.info(
                "W2-evolve: Epistemic re-classification for %s: network=%s, verification=%s",
                concept_id,
                merged_data.get("epistemic_network"),
                merged_data.get("verification_status"),
            )
    except Exception as e:
        logger.warning("W2-evolve: Epistemic re-classification failed for %s: %s", concept_id, e)

    # KA-005: Resolve knowledge_area with sentinel awareness before writing
    blob_ka = merged_data.get("knowledge_area")
    meta_ka = (
        merged_data.get("metadata", {}).get("knowledge_area") if isinstance(merged_data.get("metadata"), dict) else None
    )
    if blob_ka in _KA_SENTINELS and meta_ka and meta_ka not in _KA_SENTINELS:
        merged_data["knowledge_area"] = meta_ka
    resolved_ka = merged_data.get("knowledge_area", "general")

    # KA-004: Enforce taxonomy normalization on nonlossy write path
    try:
        from app.taxonomy import normalize_knowledge_area

        resolved_ka, _ka_source = normalize_knowledge_area(resolved_ka, strict=False)
        merged_data["knowledge_area"] = resolved_ka  # Keep blob in sync
    except Exception as e:
        logger.error("KA-004: Taxonomy normalization failed in nonlossy: %s", e)

    # FIX-1 (EVOLUTION_CHAIN_BREAK): Check if concept is already superseded.
    # Original code unconditionally set is_current=1, superseded_at=NULL, superseded_by=NULL
    # which resurrected ALL superseded concepts on any evolution (CB #2).
    # Now: preserve supersession markers if already set. If concept is superseded,
    # still evolve it (add evidence) but keep is_current=0 and markers intact.
    existing_supersession = conn.execute(
        "SELECT superseded_by, superseded_at FROM concepts WHERE id = ? AND is_current = 1",
        (concept_id,),
    ).fetchone()

    # Determine is_current and supersession state
    # DEBT-188: filter sentinel values that are not real concept IDs
    _SUPERSESSION_SENTINELS = ("", "__orphaned_supersession__")
    if (
        existing_supersession
        and existing_supersession[0] is not None
        and existing_supersession[0] not in _SUPERSESSION_SENTINELS
    ):
        # Concept is superseded — evolve content but DON'T resurrect
        logger.info(
            "FIX-1: Preserving supersession markers for %s (superseded_by=%s)",
            concept_id,
            existing_supersession[0],
        )
        conn.execute(
            """UPDATE concepts
               SET version = ?, summary = ?, confidence = ?, concept_type = ?,
                   stability = ?, updated_at = ?, data = ?, version_chain_head = ?,
                   knowledge_area = ?
               WHERE id = ? AND is_current = 1""",
            (
                new_version,
                new_summary,
                new_confidence,
                new_concept_type,
                new_stability,
                now,
                json.dumps(merged_data),
                chain_head,
                resolved_ka,
                concept_id,
            ),
        )
    else:
        # Concept is NOT superseded — normal evolution
        # DATA-045: If summary indicates archival, set is_current=0 instead of 1
        _current_flag = 0 if _is_archival else 1
        conn.execute(
            """UPDATE concepts
               SET version = ?, summary = ?, confidence = ?, concept_type = ?,
                   stability = ?, updated_at = ?, data = ?, version_chain_head = ?,
                   knowledge_area = ?,
                   is_current = ?
               WHERE id = ? AND is_current = 1""",
            (
                new_version,
                new_summary,
                new_confidence,
                new_concept_type,
                new_stability,
                now,
                json.dumps(merged_data),
                chain_head,
                resolved_ka,
                _current_flag,
                concept_id,
            ),
        )

    logger.info(
        "Non-lossy evolution: %s %s → %s (chain_head=%s, old archived)",
        concept_id,
        old_version,
        new_version,
        chain_head,
    )

    # Phase 3 v1.1 WS2-2: Drift detection after evolution
    try:
        from app.config import FEATURE_FLAGS

        if FEATURE_FLAGS.get("DRIFT_DETECTION_ENABLED", False):
            from app.drift import measure_drift

            drift = measure_drift(concept_id)
            if drift.flagged:
                # Downgrade maturity to PROVISIONAL on drift detection
                conn.execute(
                    "UPDATE concepts SET maturity = 'PROVISIONAL', "
                    "data = json_set(data, '$.maturity', 'PROVISIONAL') "
                    "WHERE id = ? AND is_current = 1",
                    (concept_id,),
                )
                logger.warning(
                    "Drift flag triggered for %s: %s — maturity downgraded to PROVISIONAL",
                    concept_id,
                    drift.flag_reason,
                )
    except Exception as e:
        logger.warning("WS2-2: Drift detection failed for %s (non-fatal): %s", concept_id, e)

    return concept_id


def get_concept_history(
    concept_id: str,
    conn: sqlite3.Connection,
    include_archived: bool = False,
) -> list[dict]:
    """Get full version history for a concept following its version chain.

    Args:
        concept_id: The concept to get history for
        conn: SQLite connection
        include_archived: Whether to also check the archive table

    Returns:
        List of version dicts ordered by created_at ascending
    """
    versions = []

    # Get all versions from concepts table
    rows = conn.execute(
        """SELECT id, version, summary, confidence, concept_type, is_current,
                  superseded_at, created_at, updated_at
           FROM concepts
           WHERE id = ? OR version_chain_head = ?
           ORDER BY created_at ASC""",
        (concept_id, concept_id),
    ).fetchall()

    for row in rows:
        versions.append(
            {
                "id": row[0],
                "version": row[1],
                "summary": row[2],
                "confidence": row[3],
                "concept_type": row[4],
                "is_current": bool(row[5]),
                "superseded_at": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
        )

    # Optionally include archived versions
    if include_archived:
        try:
            archived_rows = conn.execute(
                """SELECT id, version, summary, NULL, NULL, 0, superseded_at, created_at, NULL
                   FROM concept_versions_archive
                   WHERE id = ?
                   ORDER BY created_at ASC""",
                (concept_id,),
            ).fetchall()
            for row in archived_rows:
                versions.append(
                    {
                        "id": row[0],
                        "version": row[1],
                        "summary": row[2],
                        "confidence": row[3],
                        "concept_type": row[4],
                        "is_current": False,
                        "superseded_at": row[6],
                        "created_at": row[7],
                        "updated_at": row[8],
                        "archived": True,
                    }
                )
        except sqlite3.OperationalError:
            pass  # Archive table doesn't exist yet

    versions.sort(key=lambda v: v.get("created_at", ""))
    return versions


def archive_old_versions(
    conn: sqlite3.Connection,
    retention_days: int = 90,
    keep_last_n: int = 5,
) -> int:
    """Move superseded versions older than retention to archive table.

    §5.4.2: Keep last 5 versions in hot table regardless of age.
    Archive everything older than retention_days. Never delete archive.

    Args:
        conn: SQLite connection
        retention_days: Days after which superseded versions get archived
        keep_last_n: Always keep this many versions per concept in hot table

    Returns:
        Number of versions archived
    """
    from app.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("NONLOSSY_EVOLUTION_ENABLED", False):
        return 0

    cutoff = (_utc_now() - timedelta(days=retention_days)).isoformat()

    # Find concepts with superseded versions older than cutoff
    # Exclude the last N versions per concept
    try:
        # Get candidates for archival
        rows = conn.execute(
            """SELECT id, version, data, created_at, superseded_at
               FROM concepts
               WHERE is_current = 0 AND superseded_at IS NOT NULL AND superseded_at < ?
               ORDER BY id, created_at DESC""",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0

    # Group by concept_id to enforce keep_last_n
    from collections import defaultdict

    by_concept = defaultdict(list)
    for row in rows:
        by_concept[row[0]].append(row)

    archived = 0
    for cid, versions in by_concept.items():
        # Count total versions (including current) for this concept
        total = conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE id = ? OR version_chain_head = ?",
            (cid, cid),
        ).fetchone()[0]

        if total <= keep_last_n:
            continue  # Don't archive if we'd go below minimum

        # Archive the oldest superseded versions beyond keep_last_n
        for v_row in versions[keep_last_n:]:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO concept_versions_archive
                       (id, version, data, created_at, superseded_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (v_row[0], v_row[1], v_row[2], v_row[3], v_row[4]),
                )
                conn.execute(
                    "DELETE FROM concepts WHERE id = ? AND version = ? AND is_current = 0",
                    (v_row[0], v_row[1]),
                )
                archived += 1
            except sqlite3.OperationalError:
                pass

    if archived:
        logger.info("Archived %d old concept versions (retention=%dd, keep=%d)", archived, retention_days, keep_last_n)
    return archived


def detect_version_chain_drift(
    history: list[dict],
    drift_threshold: float = 0.30,
) -> dict | None:
    """Detect numeric drift across a version history list.

    Args:
        history: List of version dicts (as returned by get_concept_history).
                 Must be ordered oldest-first.
        drift_threshold: Fractional change that constitutes drift (default 30%).

    Returns:
        dict with drift details if drift detected, None otherwise.
    """
    if len(history) < 2:
        return None
    first = history[0]
    latest = history[-1]
    first_summary = first.get("summary", "") or ""
    latest_summary = latest.get("summary", "") or ""
    if not first_summary or not latest_summary:
        return None
    nums_first = re.findall(r"\b\d+(?:\.\d+)?\b", first_summary)
    nums_latest = re.findall(r"\b\d+(?:\.\d+)?\b", latest_summary)
    if not nums_first or not nums_latest:
        return None
    v_first = float(nums_first[0])
    v_latest = float(nums_latest[0])
    if v_first == 0:
        return None
    delta_pct = abs(v_first - v_latest) / abs(v_first)
    if delta_pct >= drift_threshold:
        return {
            "concept_id": first.get("id", "unknown"),
            "drift_type": "numeric",
            "first_value": v_first,
            "latest_value": v_latest,
            "delta_pct": delta_pct,
            "versions": len(history),
            "first_summary": first_summary,
            "latest_summary": latest_summary,
        }
    return None

