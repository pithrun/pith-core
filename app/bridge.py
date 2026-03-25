"""Cross-pith federation bridge.

Polls federation_events from a source pith and injects qualifying
concepts into a target pith. Standalone process — does not import
or depend on the running pith server.

Usage:
    python -m app.bridge --source rose --target andrew
    python -m app.bridge --source rose --target andrew --dry-run
    python -m app.bridge --source rose --target andrew --backfill-only
    python -m app.bridge --source rose --target andrew --analyze
"""

import argparse
import contextlib

# import fcntl  # REMOVED by FED-027 (replaced with SQLite locking)
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.policy_engine import validate_concept
from app.profile import resolve_data_dir, resolve_db_path

logger = logging.getLogger(__name__)


def _fed_connect(db_path, row_factory=None):
    """FED-011: Federation-safe SQLite connection with busy_timeout."""
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.execute("PRAGMA busy_timeout = 5000")
    if row_factory:
        conn.row_factory = row_factory
    return conn


# --- Constants ---
BRIDGE_POLL_INTERVAL_SECONDS = 30
BRIDGE_BACKPRESSURE_LIMIT = 1000
MAX_BRIDGE_DEPTH = 3
BRIDGE_RATE_LIMIT_PER_HOUR = 200
TRUST_ESCALATION_FACTOR = 1.5
MAX_CONCEPT_BOUNCES = 3  # FED-026: Max times a concept_id can be bridged into this pith


@dataclass
class BridgeConfig:
    """Configuration for a bridge instance."""

    source_profile: str
    target_profile: str

    # Trust filters — calibrated against Rose pith data (avg conf 0.266)
    min_confidence: float = 0.25
    allowed_knowledge_areas: list = field(default_factory=list)
    blocked_knowledge_areas: list = field(default_factory=list)
    allowed_event_types: list = field(
        default_factory=lambda: ["concept_proposed", "concept_evolved", "backfill_concept"]
    )

    # Rate limiting
    max_events_per_hour: int = 200

    # Provenance
    tag_as_federated: bool = True

    # Backfill
    enable_initial_backfill: bool = True
    backfill_min_confidence: float = 0.3

    def __post_init__(self):
        """Validate bridge configuration (Addendum A6.1)."""
        if self.source_profile == self.target_profile:
            raise ValueError(f"Cannot bridge to self: source and target are both '{self.source_profile}'")
        if self.min_confidence < 0:
            raise ValueError(f"min_confidence must be non-negative, got {self.min_confidence}")
        if self.backfill_min_confidence < 0:
            raise ValueError(f"backfill_min_confidence must be non-negative, got {self.backfill_min_confidence}")
        if self.max_events_per_hour < 1:
            raise ValueError(f"max_events_per_hour must be positive, got {self.max_events_per_hour}")

    @property
    def bridge_id(self) -> str:
        return f"{self.source_profile}_to_{self.target_profile}"


class BridgeLock:
    """Cross-platform bridge lock using SQLite advisory locking (FED-027).

    Replaces fcntl.flock with a SQLite row lock that works on all platforms,
    has automatic stale detection (LOCK_STALE_SECONDS), and supports timeout.
    """

    LOCK_STALE_SECONDS = 600  # 10 min — if lock is older, assume stale

    def __init__(self, target_profile: str):
        self.target_profile = target_profile
        self.db_path = resolve_db_path(resolve_data_dir(target_profile))
        self._acquired = False

    def acquire(self, timeout: float = 0) -> bool:
        """Try to acquire the bridge lock. Returns True if acquired."""
        deadline = time.time() + timeout
        while True:
            conn = None  # G5 amendment: defensive init
            try:
                conn = _fed_connect(self.db_path)
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS bridge_locks (
                    lock_name TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL
                )"""
                )
                # Clean stale locks
                conn.execute(
                    "DELETE FROM bridge_locks WHERE lock_name = ? AND julianday('now') - julianday(acquired_at) > ?",
                    ("bridge", self.LOCK_STALE_SECONDS / 86400),
                )
                try:
                    conn.execute(
                        "INSERT INTO bridge_locks (lock_name, pid, acquired_at) VALUES (?, ?, ?)",
                        ("bridge", os.getpid(), datetime.now(UTC).isoformat()),
                    )
                    conn.commit()
                    self._acquired = True
                    return True
                except sqlite3.IntegrityError:
                    conn.rollback()
                    if time.time() >= deadline:
                        return False
                    time.sleep(1)
            except sqlite3.OperationalError:
                # H4 fix: SQLITE_BUSY can happen at any point — _fed_connect,
                # CREATE TABLE, DELETE, or INSERT. Retry instead of crashing.
                if time.time() >= deadline:
                    return False
                time.sleep(1)
            finally:
                if conn:
                    conn.close()

    def release(self):
        if not self._acquired:
            return
        try:
            conn = _fed_connect(self.db_path)
            try:
                conn.execute(
                    "DELETE FROM bridge_locks WHERE lock_name = ? AND pid = ?",
                    ("bridge", os.getpid()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass  # Best effort on release
        self._acquired = False
        # Clean up old fcntl lock file if it exists (migration from fcntl)
        old_lock = Path.home() / "pith-data" / self.target_profile / "locks" / "bridge.lock"
        if old_lock.exists():
            with contextlib.suppress(Exception):
                old_lock.unlink()


@dataclass
class BridgeMetrics:
    """In-memory metrics for bridge health reporting (A4.1)."""

    events_processed_total: int = 0
    events_failed_total: int = 0
    events_skipped_total: int = 0
    last_poll_at: str | None = None
    last_successful_bridge_at: str | None = None
    consecutive_empty_polls: int = 0
    current_lag: int = 0
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _check_concept_bounce_limit(target_db: Path, concept_id: str) -> bool:
    """FED-026: Return True if concept has NOT exceeded bounce limit."""
    if not concept_id:
        return True
    try:
        conn = _fed_connect(target_db)
        try:
            # G2 amendment: ensure index for bounce check performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gov_events_concept ON governance_events(concept_id)")
            count = conn.execute(
                "SELECT COUNT(*) FROM governance_events WHERE event_type = 'BRIDGE_CONCEPT_SEEN' AND concept_id = ?",
                (concept_id,),
            ).fetchone()[0]
            return count < MAX_CONCEPT_BOUNCES
        finally:
            conn.close()
    except Exception:
        # G4 amendment: fail open — don't block bridge on governance table issues
        logger.debug("Bounce limit check failed — allowing concept (defensive)")
        return True


def trust_filter(event: dict, config: BridgeConfig, target_db: Path | None = None) -> bool:
    """Apply trust filtering to a federation event.

    Returns True if the event should be bridged.
    """
    payload = json.loads(event["payload"]) if isinstance(event["payload"], str) else event["payload"]

    # Confidence check
    confidence = payload.get("confidence", 0)
    if confidence < config.min_confidence:
        return False

    # KA check
    ka = payload.get("knowledge_area", "general")
    if config.allowed_knowledge_areas and ka not in config.allowed_knowledge_areas:
        return False
    if ka in config.blocked_knowledge_areas:
        return False

    # Event type check
    if event["event_type"] not in config.allowed_event_types:
        return False

    # Loop prevention
    if event.get("origin_brain") == config.target_profile:
        return False
    if (event.get("bridge_depth") or 0) >= MAX_BRIDGE_DEPTH:
        return False

    # Trust escalation guard (payload-level)
    original_confidence = payload.get("original_confidence", confidence)
    if confidence > original_confidence * TRUST_ESCALATION_FACTOR:
        return False

    # H2 fix: Reject events with no concept_id — cannot track bounces or trust.
    # All allowed_event_types (concept_proposed, concept_evolved, backfill_concept)
    # are concept-level operations that must have concept_id.
    concept_id = event.get("concept_id") or payload.get("concept_id")
    if not concept_id:
        logger.warning("FED-026: Rejecting event with no concept_id — cannot track bounces")
        return False

    # FED-025: Cross-hop cumulative trust escalation check.
    # Verify incoming confidence against the target pith's stored value
    # for this concept. Prevents gradual confidence inflation across hops.
    if target_db:
        try:
            conn = _fed_connect(target_db)
            try:
                row = conn.execute(
                    "SELECT confidence FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if row is not None:
                    stored_confidence = row[0] if isinstance(row, tuple | list) else row["confidence"]
                    if confidence > stored_confidence * TRUST_ESCALATION_FACTOR:
                        logger.info(
                            "FED-025: Trust escalation blocked for %s: incoming %.3f > stored %.3f * %.1f",
                            concept_id,
                            confidence,
                            stored_confidence,
                            TRUST_ESCALATION_FACTOR,
                        )
                        return False
            finally:
                conn.close()
        except Exception as e:
            logger.debug("FED-025: Could not verify stored confidence for %s: %s", concept_id, e)

    # FED-026: Per-concept bounce tracking — stops ping-pong loops via evolution
    if target_db:
        if not _check_concept_bounce_limit(target_db, concept_id):
            logger.info("FED-026: Bounce limit reached for %s", concept_id)
            return False

    return True


def run_bridge(config: BridgeConfig, dry_run: bool = False) -> None:
    """Main bridge loop: poll source -> filter -> inject target."""
    # Self-bridge validated in BridgeConfig.__post_init__

    source_db = resolve_db_path(resolve_data_dir(config.source_profile))
    target_db = resolve_db_path(resolve_data_dir(config.target_profile))

    # Acquire lock to prevent concurrent bridge instances (A11.1)
    lock = BridgeLock(config.target_profile)
    if not lock.acquire():
        raise RuntimeError(f"Another bridge instance is already running for target '{config.target_profile}'")

    metrics = BridgeMetrics()
    start_time = time.time()

    logger.info(f"Bridge starting: {config.source_profile} -> {config.target_profile}")
    logger.info(f"Source DB: {source_db}")
    logger.info(f"Target DB: {target_db}")

    try:
        # Schema version handshake
        _verify_schema_compatibility(source_db, target_db)

        # Check if initial backfill needed
        if config.enable_initial_backfill:
            _maybe_run_backfill(config, source_db, target_db, dry_run)

        # Rate limiting state
        events_this_hour = 0
        hour_start = time.time()

        while True:
            metrics.uptime_seconds = time.time() - start_time
            metrics.last_poll_at = datetime.now(UTC).isoformat()

            # Reset hourly counter
            if time.time() - hour_start > 3600:
                events_this_hour = 0
                hour_start = time.time()

            # Rate limit check
            if events_this_hour >= config.max_events_per_hour:
                logger.debug("Rate limit reached, sleeping until next hour")
                time.sleep(60)
                continue

            # Backpressure check on target
            target_conn = _fed_connect(target_db, sqlite3.Row)
            try:
                unconsumed = target_conn.execute(
                    "SELECT COUNT(*) FROM federation_events WHERE consumed = 0 AND origin_brain IS NOT NULL"
                ).fetchone()[0]
                metrics.current_lag = unconsumed
                if unconsumed >= BRIDGE_BACKPRESSURE_LIMIT:
                    logger.warning(f"Backpressure: {unconsumed} unconsumed bridged events in target")
                    _log_bridge_event(
                        target_conn,
                        "BRIDGE_BACKPRESSURE",
                        None,
                        {"unconsumed": unconsumed, "limit": BRIDGE_BACKPRESSURE_LIMIT},
                    )
                    target_conn.commit()
                    time.sleep(300)
                    continue
            finally:
                target_conn.close()

            # Poll source for new events
            source_conn = _fed_connect(source_db, sqlite3.Row)
            try:
                rows = source_conn.execute(
                    """SELECT fe.* FROM federation_events fe
                       LEFT JOIN bridge_event_consumption bec
                         ON bec.bridge_id = ? AND bec.event_id = fe.id
                       WHERE bec.event_id IS NULL
                       ORDER BY fe.id ASC LIMIT 50""",
                    (config.bridge_id,),
                ).fetchall()
            finally:
                source_conn.close()

            if not rows:
                metrics.consecutive_empty_polls += 1
                time.sleep(BRIDGE_POLL_INTERVAL_SECONDS)
                continue

            metrics.consecutive_empty_polls = 0

            # Process events
            bridged = 0
            filtered = 0
            for row in rows:
                event = dict(row)

                if trust_filter(event, config, target_db=target_db):
                    if not dry_run:
                        try:
                            injected = _inject_event(config, target_db, event)
                            if injected:
                                metrics.events_processed_total += 1
                                metrics.last_successful_bridge_at = datetime.now(UTC).isoformat()
                            else:
                                metrics.events_skipped_total += 1
                        except Exception as e:
                            metrics.events_failed_total += 1
                            # FED-004: Ensure error-path connection is closed
                            err_conn = _fed_connect(target_db)
                            try:
                                _log_bridge_error(err_conn, event, e, config.bridge_id)
                            finally:
                                err_conn.close()
                    bridged += 1
                    events_this_hour += 1
                else:
                    filtered += 1
                    metrics.events_skipped_total += 1

                # Mark as consumed regardless of filter result
                if not dry_run:
                    _mark_consumed(source_db, config.bridge_id, event["id"])

            # FED-009: Graduate quarantine after each batch
            if not dry_run:
                grad, rej = _graduate_quarantine(target_db, config)
                if grad > 0 or rej > 0:
                    logger.info(f"Quarantine: {grad} graduated, {rej} rejected")

            if bridged > 0 or filtered > 0:
                logger.info(
                    f"Bridge batch: {bridged} bridged, {filtered} filtered, "
                    f"{events_this_hour}/{config.max_events_per_hour} this hour"
                )

            # Periodic pruning (every 100 polls)
            if metrics.events_processed_total > 0 and metrics.events_processed_total % 100 == 0:
                pruned = _prune_consumed_events(source_db)
                if pruned > 0:
                    logger.info(f"Pruned {pruned} consumed events older than 30 days")

            time.sleep(BRIDGE_POLL_INTERVAL_SECONDS)
    finally:
        lock.release()


def _inject_event(config: BridgeConfig, target_db: Path, event: dict) -> bool:
    """Inject a bridged event into the target pith.

    Returns True if the concept was written, False if rejected by validation/policy.
    """
    # FED-003: Validate concept data before writing to target
    payload = json.loads(event["payload"]) if isinstance(event["payload"], str) else event["payload"]
    # Merge concept_id from event envelope into payload for validation
    concept_data = {**payload, "id": event.get("concept_id")}

    # BUG-003: Single connection for entire inject lifecycle.
    # Previously, FED-023 and FED-024 rejection paths each opened their own
    # _fed_connect(target_db), risking SQLITE_BUSY under concurrent access.
    conn = _fed_connect(target_db)
    try:
        # H1 fix: Ensure governance_events exists so bounce tracking isn't blind.
        # Without this, _log_bridge_event silently skips BRIDGE_CONCEPT_SEEN and
        # _check_concept_bounce_limit fails open — allowing unlimited injections.
        conn.execute("""CREATE TABLE IF NOT EXISTS governance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            concept_id TEXT,
            details TEXT,
            created_at TEXT NOT NULL
        )""")
        # FED-009: Ensure quarantine staging table exists.
        conn.execute("""CREATE TABLE IF NOT EXISTS bridge_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            concept_id TEXT,
            source_session_id TEXT,
            source_model_id TEXT DEFAULT 'unknown',
            source_agent_id TEXT DEFAULT 'default',
            payload TEXT NOT NULL,
            origin_brain TEXT,
            bridge_depth INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            rejection_reason TEXT,
            bridge_id TEXT,
            quarantined_at TEXT NOT NULL,
            graduated_at TEXT,
            created_at TEXT NOT NULL
        )""")
        if not _validate_bridged_concept(concept_data):
            logger.warning("Skipping invalid bridged concept: %s", event.get("concept_id"))
            # FED-023: Log governance event for structural validation rejection
            with contextlib.suppress(Exception):
                _log_bridge_event(
                    conn,
                    "BRIDGE_CONCEPT_REJECTED",
                    event.get("concept_id"),
                    {
                        "source": config.source_profile,
                        "reason": "structural_validation_failed",
                        "bridge_depth": (event.get("bridge_depth") or 0) + 1,
                    },
                )
                conn.commit()
            return False

        # FED-024: Policy engine validation — blocks injection payloads, directive hijacking,
        # and always_activate abuse that structural validation alone cannot catch.
        policy_result = validate_concept(
            {
                "summary": payload.get("summary", ""),
                "concept_type": payload.get("concept_type"),
                "concept_id": event.get("concept_id", ""),
                "source": "bridge",
                "always_activate": payload.get("always_activate", False),
            }
        )
        if not policy_result["allowed"]:
            violation_details = "; ".join(v["detail"] for v in policy_result["violations"])
            logger.warning(
                "FED-024: Policy engine rejected bridged concept %s: %s", event.get("concept_id"), violation_details
            )
            # Log governance event for rejected concept
            with contextlib.suppress(Exception):
                _log_bridge_event(
                    conn,
                    "BRIDGE_POLICY_REJECTED",
                    event.get("concept_id"),
                    {
                        "source": config.source_profile,
                        "violations": policy_result["violations"],
                        "bridge_depth": (event.get("bridge_depth") or 0) + 1,
                    },
                )
                conn.commit()
            return False

        # H3 fix: Atomic bounce check within the injection transaction.
        # The trust_filter pre-check uses a separate connection, so two concurrent
        # bridge runs can both pass. This in-transaction check is authoritative.
        concept_id = event.get("concept_id")
        if concept_id:
            try:
                bounce_count = conn.execute(
                    "SELECT COUNT(*) FROM governance_events "
                    "WHERE event_type = 'BRIDGE_CONCEPT_SEEN' AND concept_id = ?",
                    (concept_id,),
                ).fetchone()[0]
                if bounce_count >= MAX_CONCEPT_BOUNCES:
                    logger.info("FED-026: Bounce limit reached for %s (atomic check)", concept_id)
                    conn.rollback()
                    return False
            except Exception:
                pass  # governance_events may not exist yet — created above by H1 fix

        # FED-009: Write to quarantine staging table instead of federation_events.
        now_iso = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT INTO bridge_quarantine
               (event_type, concept_id, source_session_id, source_model_id,
                source_agent_id, payload, origin_brain, bridge_depth,
                status, bridge_id, quarantined_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (
                event["event_type"],
                event.get("concept_id"),
                event.get("source_session_id"),
                event.get("source_model_id", "unknown"),
                event.get("source_agent_id", "default"),
                event["payload"],
                event.get("origin_brain") or config.source_profile,
                (event.get("bridge_depth") or 0) + 1,
                config.bridge_id,
                now_iso,
                now_iso,
            ),
        )
        _log_bridge_event(
            conn,
            "BRIDGE_CONCEPT_QUARANTINED",
            event.get("concept_id"),
            {"source": config.source_profile, "bridge_depth": (event.get("bridge_depth") or 0) + 1},
        )
        # FED-026: Record concept bounce for ping-pong detection
        _log_bridge_event(
            conn,
            "BRIDGE_CONCEPT_SEEN",
            event.get("concept_id"),
            {"source": config.source_profile, "bridge_depth": (event.get("bridge_depth") or 0) + 1},
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to inject event {event.get('id')}: {e}")
        with contextlib.suppress(Exception):
            _log_bridge_event(conn, "BRIDGE_ERROR", event.get("concept_id"), {"error": str(e)})
            conn.commit()
        return False
    finally:
        conn.close()


def _mark_consumed(source_db: Path, bridge_id: str, event_id: int) -> None:
    """Mark an event as consumed by this bridge."""
    conn = _fed_connect(source_db)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO bridge_event_consumption (bridge_id, event_id) VALUES (?, ?)", (bridge_id, event_id)
        )
        conn.commit()
    finally:
        conn.close()


def _graduate_quarantine(target_db: Path, config: BridgeConfig) -> tuple[int, int]:
    """FED-009: Move pending quarantine concepts to federation_events."""
    conn = _fed_connect(target_db, sqlite3.Row)
    graduated = 0
    rejected = 0
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS bridge_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL, concept_id TEXT,
            source_session_id TEXT, source_model_id TEXT DEFAULT 'unknown',
            source_agent_id TEXT DEFAULT 'default',
            payload TEXT NOT NULL, origin_brain TEXT,
            bridge_depth INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            rejection_reason TEXT, bridge_id TEXT,
            quarantined_at TEXT NOT NULL, graduated_at TEXT,
            created_at TEXT NOT NULL
        )""")
        rows = conn.execute("SELECT * FROM bridge_quarantine WHERE status = 'pending' ORDER BY id ASC").fetchall()
        for row in rows:
            row_dict = dict(row)
            payload = json.loads(row_dict["payload"]) if isinstance(row_dict["payload"], str) else row_dict["payload"]
            concept_data = {**payload, "id": row_dict["concept_id"]}
            if not _validate_bridged_concept(concept_data):
                conn.execute(
                    "UPDATE bridge_quarantine SET status='rejected', rejection_reason=? WHERE id=?",
                    ("re-validation_failed", row_dict["id"]),
                )
                rejected += 1
                continue
            conn.execute(
                """INSERT INTO federation_events
                   (event_type, concept_id, source_session_id, source_model_id,
                    source_agent_id, payload, origin_brain, bridge_depth, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row_dict["event_type"],
                    row_dict["concept_id"],
                    row_dict["source_session_id"],
                    row_dict["source_model_id"],
                    row_dict["source_agent_id"],
                    row_dict["payload"],
                    row_dict["origin_brain"],
                    row_dict["bridge_depth"],
                    row_dict["created_at"],
                ),
            )
            conn.execute(
                "UPDATE bridge_quarantine SET status='graduated', graduated_at=? WHERE id=?",
                (datetime.now(UTC).isoformat(), row_dict["id"]),
            )
            graduated += 1
        conn.commit()
    except Exception as e:
        logger.error("Quarantine graduation failed: %s", e)
        with contextlib.suppress(Exception):
            conn.rollback()
    finally:
        conn.close()
    return graduated, rejected


def _nuke_quarantine_batch(
    target_db: Path, *, bridge_id: str | None = None, origin_brain: str | None = None, reason: str = "manual_nuke"
) -> int:
    """FED-009: Reject all pending quarantine entries matching criteria."""
    conn = _fed_connect(target_db)
    try:
        if bridge_id:
            cursor = conn.execute(
                "UPDATE bridge_quarantine SET status='rejected', rejection_reason=? "
                "WHERE status='pending' AND bridge_id=?",
                (reason, bridge_id),
            )
        elif origin_brain:
            cursor = conn.execute(
                "UPDATE bridge_quarantine SET status='rejected', rejection_reason=? "
                "WHERE status='pending' AND origin_brain=?",
                (reason, origin_brain),
            )
        else:
            cursor = conn.execute(
                "UPDATE bridge_quarantine SET status='rejected', rejection_reason=? WHERE status='pending'", (reason,)
            )
        nuked = cursor.rowcount
        if nuked > 0:
            _log_bridge_event(
                conn,
                "QUARANTINE_BATCH_NUKED",
                None,
                {"count": nuked, "bridge_id": bridge_id, "origin_brain": origin_brain, "reason": reason},
            )
        conn.commit()
        return nuked
    finally:
        conn.close()


def _log_bridge_event(conn, event_type: str, concept_id: str | None, details: dict) -> None:
    """Log a governance event for bridge activity."""
    # FED-001: Check table exists before INSERT (matches _log_bridge_error pattern)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='governance_events'"
        ).fetchall()
    ]
    if "governance_events" not in tables:
        return
    conn.execute(
        """INSERT INTO governance_events (event_type, concept_id, details, created_at)
           VALUES (?, ?, ?, ?)""",
        (event_type, concept_id, json.dumps(details), datetime.now(UTC).isoformat()),
    )


def _validate_bridged_concept(concept_data: dict) -> bool:
    """Validate concept data before writing to target pith (A8.3 + FED-012)."""
    required_keys = {"id", "summary", "confidence", "knowledge_area"}
    if not required_keys.issubset(concept_data.keys()):
        logger.warning("Bridged concept missing required keys: %s", required_keys - concept_data.keys())
        return False
    if not isinstance(concept_data.get("confidence"), int | float):
        return False
    if not (0.0 <= concept_data["confidence"] <= 1.0):
        return False

    summary = concept_data.get("summary", "")

    # FED-012: Content quality gates — lightweight pattern guards, not LLM-based
    # (a) BUG-004: Type guard — summary must be a string (crash on int/None)
    if not isinstance(summary, str):
        return False
    # (b) Non-empty summary with minimum length
    if not summary or len(summary.strip()) < 10:
        logger.warning(
            "Bridged concept %s rejected: summary too short (%d chars)",
            concept_data.get("id"),
            len(summary),
        )
        return False
    # (c) Maximum length
    if len(summary) > 2000:
        return False
    # (d) Repeated character / gibberish detection (e.g., "aaaaaa" or "!@#$%^")
    if len(summary) > 20:
        unique_chars = len(set(summary.lower()))
        if unique_chars < 5:
            logger.warning(
                "Bridged concept %s rejected: gibberish (only %d unique chars)",
                concept_data.get("id"),
                unique_chars,
            )
            return False
    # (e) SQL injection pattern guard
    # BUG-002: Tightened from "UPDATE " (false positive on "Update the...")
    # BUG-004: Added UNION SELECT, ALTER TABLE; normalized whitespace
    _SQL_PATTERNS = (
        "DROP TABLE",
        "DELETE FROM",
        "INSERT INTO",
        "UPDATE SET",
        "UPDATE WHERE",
        "ALTER TABLE",
        "UNION SELECT",
        "--",
        "'; ",
        "1=1",
    )
    summary_normalized = " ".join(summary.upper().split())
    for pattern in _SQL_PATTERNS:
        if pattern in summary_normalized:
            logger.warning(
                "Bridged concept %s rejected: SQL pattern detected",
                concept_data.get("id"),
            )
            return False
    # (f) BUG-004: Newline spam detection
    if summary.count("\n") > len(summary) * 0.1 and len(summary) > 50:
        logger.warning(
            "Bridged concept %s rejected: newline spam",
            concept_data.get("id"),
        )
        return False

    return True


def _prune_consumed_events(db_path: Path, retention_days: int = 30) -> int:
    """Delete consumed federation_events older than retention period (A10.1).

    Returns number of pruned events.
    """
    conn = _fed_connect(db_path)
    try:
        cursor = conn.execute(
            """DELETE FROM federation_events
               WHERE consumed = 1
               AND consumed_at < datetime('now', ? || ' days')""",
            (f"-{retention_days}",),
        )
        pruned = cursor.rowcount
        conn.commit()
        return pruned
    finally:
        conn.close()


def check_federation_integrity(db_path: Path) -> list[str]:
    """Check federation data integrity (A10.2). Returns list of issues."""
    issues = []
    conn = _fed_connect(db_path)
    try:
        # 1. Events referencing non-existent concepts
        orphans = conn.execute(
            """SELECT COUNT(*) FROM federation_events fe
               LEFT JOIN concepts c ON c.id = fe.concept_id
               WHERE fe.concept_id IS NOT NULL AND c.id IS NULL"""
        ).fetchone()[0]
        if orphans > 0:
            issues.append(f"{orphans} federation events reference non-existent concepts")

        # 2. Consumed events with no consumed_at timestamp
        missing_ts = conn.execute(
            "SELECT COUNT(*) FROM federation_events WHERE consumed = 1 AND consumed_at IS NULL"
        ).fetchone()[0]
        if missing_ts > 0:
            issues.append(f"{missing_ts} consumed events missing consumed_at timestamp")

        # 3. Bridge depth exceeding MAX_BRIDGE_DEPTH
        deep = conn.execute(
            "SELECT COUNT(*) FROM federation_events WHERE bridge_depth > ?", (MAX_BRIDGE_DEPTH,)
        ).fetchone()[0]
        if deep > 0:
            issues.append(f"{deep} events exceed MAX_BRIDGE_DEPTH (possible loop)")
    finally:
        conn.close()
    return issues


def _log_bridge_error(conn, event: dict, error: Exception, bridge_id: str = "unknown") -> None:
    """Log bridge error as governance event (A3.4)."""
    try:
        # Check if governance_events table exists
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='governance_events'"
            ).fetchall()
        ]
        if "governance_events" not in tables:
            return
        conn.execute(
            """INSERT INTO governance_events
               (event_type, concept_id, details, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                "BRIDGE_ERROR",
                event.get("concept_id", "unknown"),
                json.dumps(
                    {
                        "bridge_id": bridge_id,
                        "event_id": event.get("id"),
                        "error": str(error)[:500],
                        "error_type": type(error).__name__,
                    }
                ),
                datetime.now(UTC).isoformat(),
            ),
        )
    except Exception:
        logger.error("Failed to log bridge error governance event: %s", error)


def _verify_schema_compatibility(source_db: Path, target_db: Path) -> None:
    """Verify both brains have compatible schemas (table + column level)."""
    # FED-029: Required columns for federation_events — a migration that adds a column
    # to one pith but not the other causes silent data loss or write failures.
    REQUIRED_FED_COLUMNS = {
        "event_type",
        "concept_id",
        "source_session_id",
        "source_model_id",
        "source_agent_id",
        "payload",
        "origin_brain",
        "bridge_depth",
        "created_at",
    }
    for db_path, label in [(source_db, "source"), (target_db, "target")]:
        conn = _fed_connect(db_path)
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "federation_events" not in tables:
                raise RuntimeError(f"{label} pith at {db_path} missing federation_events table. Run migrations first.")
            # FED-029: Column-level compatibility check
            columns = {r[1] for r in conn.execute("PRAGMA table_info(federation_events)").fetchall()}
            missing = REQUIRED_FED_COLUMNS - columns
            if missing:
                raise RuntimeError(
                    f"{label} pith at {db_path} missing federation_events columns: {missing}. Run migrations first."
                )
        finally:
            conn.close()


def _maybe_run_backfill(config: BridgeConfig, source_db: Path, target_db: Path, dry_run: bool) -> None:
    """One-time backfill of high-quality concepts from source."""
    # Check for BRIDGE_BACKFILL_COMPLETE governance event
    target_conn = _fed_connect(target_db)
    try:
        backfill_done = target_conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE event_type = ? AND details LIKE ?",
            ("BRIDGE_BACKFILL_COMPLETE", f"%{config.bridge_id}%"),
        ).fetchone()[0]
        if backfill_done > 0:
            logger.info("Backfill already done — skipping")
            return
    finally:
        target_conn.close()

    logger.info(
        f"Running initial backfill from {config.source_profile} (min_confidence={config.backfill_min_confidence})"
    )

    source_conn = _fed_connect(source_db, sqlite3.Row)
    try:
        concepts = source_conn.execute(
            """SELECT id, summary, confidence, knowledge_area,
                      concept_type, data, created_at
               FROM concepts
               WHERE confidence >= ? AND status = 'active'
               ORDER BY confidence DESC""",
            (config.backfill_min_confidence,),
        ).fetchall()
    finally:
        source_conn.close()
    logger.info(f"Backfill: {len(concepts)} concepts qualify")

    backfilled = 0
    skipped = 0  # FED-031: Track policy-rejected concepts separately
    for concept in concepts:
        ka = concept["knowledge_area"] or "general"
        if config.allowed_knowledge_areas and ka not in config.allowed_knowledge_areas:
            continue
        if ka in config.blocked_knowledge_areas:
            continue

        event = {
            "event_type": "backfill_concept",
            "concept_id": concept["id"],
            "source_model_id": "unknown",
            "source_agent_id": "default",
            "payload": json.dumps(
                {
                    "summary": concept["summary"],
                    "confidence": concept["confidence"],
                    "knowledge_area": ka,
                    "concept_type": concept["concept_type"],
                    "original_confidence": concept["confidence"],
                }
            ),
            "origin_brain": config.source_profile,
            "bridge_depth": 0,
        }

        if not dry_run:
            if not _inject_event(config, target_db, event):
                skipped += 1
                continue  # FED-031: Don't count policy-rejected concepts as backfilled
        backfilled += 1
        # Rate limit backfill
        if backfilled % 100 == 0 and backfilled > 0:
            logger.info(f"Backfill progress: {backfilled}/{len(concepts)}")
            time.sleep(1)

    logger.info(f"Backfill complete: {backfilled} concepts bridged, {skipped} rejected")

    # Mark backfill complete
    if not dry_run:
        target_conn = _fed_connect(target_db)
        try:
            _log_bridge_event(
                target_conn,
                "BRIDGE_BACKFILL_COMPLETE",
                None,
                {
                    "source": config.source_profile,
                    "count": backfilled,
                    "skipped": skipped,
                    "bridge_id": config.bridge_id,
                },
            )
            target_conn.commit()
        finally:
            target_conn.close()


def analyze_source(profile: str) -> dict:
    """Analyze a source pith's data distribution for bridge config.

    Usage: python -m app.bridge --source rose --analyze
    """
    db_path = resolve_db_path(resolve_data_dir(profile))
    conn = _fed_connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM concepts").fetchone()[0] or 0

        thresholds = {}
        for t in [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
            count = conn.execute("SELECT COUNT(*) FROM concepts WHERE confidence >= ?", (t,)).fetchone()[0]
            thresholds[t] = {"count": count, "pct": round(count / max(total, 1) * 100, 1)}

        ka_dist = conn.execute(
            "SELECT knowledge_area, COUNT(*) FROM concepts GROUP BY knowledge_area ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()

        return {
            "profile": profile,
            "total_concepts": total,
            "avg_confidence": round(avg_conf, 4),
            "confidence_thresholds": thresholds,
            "top_knowledge_areas": {r[0]: r[1] for r in ka_dist},
            "recommended_min_confidence": 0.25 if avg_conf < 0.35 else 0.4,
            "recommended_backfill_min": 0.3 if avg_conf < 0.35 else 0.5,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pith cross-pith federation bridge")
    parser.add_argument("--source", required=True, help="Source pith profile name")
    parser.add_argument("--target", help="Target pith profile name")
    parser.add_argument("--dry-run", action="store_true", help="Log but don't write")
    parser.add_argument("--backfill-only", action="store_true", help="Run backfill then exit")
    parser.add_argument("--analyze", action="store_true", help="Analyze source pith distribution")
    parser.add_argument("--config", help="Path to bridge config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.analyze:
        import pprint

        result = analyze_source(args.source)
        pprint.pprint(result)
    else:
        if not args.target:
            parser.error("--target is required for bridge mode")

        config = BridgeConfig(source_profile=args.source, target_profile=args.target)

        if args.backfill_only:
            source_db = resolve_db_path(resolve_data_dir(config.source_profile))
            target_db = resolve_db_path(resolve_data_dir(config.target_profile))
            _verify_schema_compatibility(source_db, target_db)
            _maybe_run_backfill(config, source_db, target_db, args.dry_run)
        else:
            run_bridge(config, dry_run=args.dry_run)
