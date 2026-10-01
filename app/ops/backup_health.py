"""Bounded cached proof for the maintenance recovery backup."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat as stat_module
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKUP_HEALTH_SCHEMA_VERSION = 1
BACKUP_HEALTH_MAX_BYTES = 16 * 1024
BACKUP_RESERVE_BYTES = 1024**3
BACKUP_OPERATION_ID = "maintenance_backup"
SIGNED_64_MAX = 2**63 - 1
BACKUP_HEALTH_SNAPSHOT_NAME = "backup_health_snapshot.json"
BACKUP_ARTIFACT_NAME = "pith_backup.db"
BACKUP_WARNING_AGE_HOURS = 12
BACKUP_CRITICAL_AGE_HOURS = 24
BACKUP_FUTURE_SKEW_SECONDS = 5 * 60

BACKUP_ATTEMPT_REASON_CODES = frozenset(
    {
        "verified",
        "backup_source_missing",
        "capacity_measurement_failed",
        "insufficient_headroom",
        "backup_failed",
        "backup_integrity_failed",
    }
)
BACKUP_RESPONSE_REASON_CODES = frozenset(
    {
        "ok",
        "no_backup",
        "unverified_stale_backup",
        "cached_proof_missing",
        "invalid_cached_proof",
        "no_verified_backup",
        "artifact_changed_since_verification",
        "stale_backup",
        "aging_backup",
        "capacity_unknown",
    }
) | (BACKUP_ATTEMPT_REASON_CODES - {"verified"})

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SNAPSHOT_KEYS = {"schema_version", "sampled_at", "last_attempt", "last_verified", "capacity"}
_ATTEMPT_KEYS = {"at", "state", "reason_code"}
_VERIFIED_KEYS = {
    "at",
    "backup_path",
    "backup_size_bytes",
    "backup_mtime_ns",
    "concept_count",
    "integrity",
}
_CAPACITY_KEYS = {"state", "sampled_at", "operations"}
_OPERATION_KEYS = {
    "id",
    "free_bytes",
    "required_bytes",
    "headroom_bytes",
    "admissible",
    "reason_code",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 string")
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    return value


def _require_int(value: object, name: str, *, allow_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value > SIGNED_64_MAX or value < (-SIGNED_64_MAX - 1 if allow_negative else 0):
        raise ValueError(f"{name} is outside the signed 64-bit range")
    return value


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a controlled identifier")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def unknown_backup_capacity(*, sampled_at: datetime | None = None) -> dict[str, Any]:
    """Return the only valid unknown-capacity representation."""
    measured_at = sampled_at or _utc_now()
    return {
        "state": "unknown",
        "sampled_at": _iso_utc(measured_at),
        "operations": [
            {
                "id": BACKUP_OPERATION_ID,
                "free_bytes": None,
                "required_bytes": None,
                "headroom_bytes": None,
                "admissible": None,
                "reason_code": "measurement_failed",
            }
        ],
    }


def measure_backup_capacity(source_path: Path, target_dir: Path) -> dict[str, Any]:
    """Measure admission headroom for one maintenance-backup copy."""
    sampled_at = _utc_now()
    try:
        source_bytes = _require_int(source_path.stat().st_size, "source size")
        family_bytes = source_bytes
        for suffix in ("-wal", "-shm"):
            companion = Path(f"{source_path}{suffix}")
            try:
                companion_bytes = companion.stat().st_size
            except FileNotFoundError:
                companion_bytes = 0
            family_bytes += _require_int(companion_bytes, f"source{suffix} size")
            if family_bytes > SIGNED_64_MAX:
                raise OverflowError("source family size overflow")
        required_bytes = family_bytes + BACKUP_RESERVE_BYTES
        if required_bytes > SIGNED_64_MAX:
            raise OverflowError("required size overflow")
        free_bytes = _require_int(shutil.disk_usage(target_dir).free, "free bytes")
        headroom_bytes = free_bytes - required_bytes
        _require_int(headroom_bytes, "headroom bytes", allow_negative=True)
    except (OSError, OverflowError, ValueError):
        return unknown_backup_capacity(sampled_at=sampled_at)

    admissible = headroom_bytes >= 0
    return {
        "state": "healthy" if admissible else "critical",
        "sampled_at": _iso_utc(sampled_at),
        "operations": [
            {
                "id": BACKUP_OPERATION_ID,
                "free_bytes": free_bytes,
                "required_bytes": required_bytes,
                "headroom_bytes": headroom_bytes,
                "admissible": admissible,
                "reason_code": "admissible" if admissible else "insufficient_headroom",
            }
        ],
    }


def validate_backup_health_snapshot(value: object) -> dict[str, Any]:
    """Validate and return a detached schema-v1 backup proof."""
    snapshot = _require_exact_keys(value, _SNAPSHOT_KEYS, "snapshot")
    version = snapshot["schema_version"]
    if isinstance(version, bool) or version != BACKUP_HEALTH_SCHEMA_VERSION:
        raise ValueError("unsupported backup-health schema version")

    sampled_at = _parse_timestamp(snapshot["sampled_at"], "sampled_at")

    attempt = _require_exact_keys(snapshot["last_attempt"], _ATTEMPT_KEYS, "last_attempt")
    attempt_at = _parse_timestamp(attempt["at"], "last_attempt.at")
    if attempt_at != sampled_at:
        raise ValueError("last_attempt.at must equal sampled_at")
    attempt_state = attempt["state"]
    if attempt_state not in {"healthy", "critical"}:
        raise ValueError("last_attempt.state is invalid")
    attempt_reason = _require_identifier(attempt["reason_code"], "last_attempt.reason_code")
    if attempt_reason not in BACKUP_ATTEMPT_REASON_CODES:
        raise ValueError("last_attempt.reason_code is invalid")
    if (attempt_state == "healthy") != (attempt_reason == "verified"):
        raise ValueError("last_attempt state and reason disagree")

    verified_value = snapshot["last_verified"]
    verified_at: datetime | None = None
    if verified_value is not None:
        verified = _require_exact_keys(verified_value, _VERIFIED_KEYS, "last_verified")
        verified_at = _parse_timestamp(verified["at"], "last_verified.at")
        if verified_at > sampled_at:
            raise ValueError("last_verified.at cannot be after sampled_at")
        backup_path = verified["backup_path"]
        if not isinstance(backup_path, str) or not backup_path or not Path(backup_path).is_absolute():
            raise ValueError("last_verified.backup_path must be absolute")
        _require_int(verified["backup_size_bytes"], "last_verified.backup_size_bytes")
        _require_int(verified["backup_mtime_ns"], "last_verified.backup_mtime_ns")
        _require_int(verified["concept_count"], "last_verified.concept_count")
        if verified["integrity"] != "ok":
            raise ValueError("last_verified.integrity must be ok")
    elif attempt_state == "healthy":
        raise ValueError("a healthy attempt requires last_verified")

    capacity = _require_exact_keys(snapshot["capacity"], _CAPACITY_KEYS, "capacity")
    capacity_state = capacity["state"]
    if capacity_state not in {"healthy", "critical", "unknown"}:
        raise ValueError("capacity.state is invalid")
    capacity_at = _parse_timestamp(capacity["sampled_at"], "capacity.sampled_at")
    if capacity_at > sampled_at:
        raise ValueError("capacity.sampled_at cannot be after sampled_at")
    operations = capacity["operations"]
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError("capacity.operations must contain exactly one operation")
    operation = _require_exact_keys(operations[0], _OPERATION_KEYS, "capacity operation")
    if operation["id"] != BACKUP_OPERATION_ID:
        raise ValueError("capacity operation id is invalid")
    reason = _require_identifier(operation["reason_code"], "capacity operation reason_code")

    numeric_keys = ("free_bytes", "required_bytes", "headroom_bytes")
    if capacity_state == "unknown":
        if any(operation[key] is not None for key in numeric_keys) or operation["admissible"] is not None:
            raise ValueError("unknown capacity must contain only null values")
        if reason != "measurement_failed":
            raise ValueError("unknown capacity reason is invalid")
    else:
        free_bytes = _require_int(operation["free_bytes"], "capacity.free_bytes")
        required_bytes = _require_int(operation["required_bytes"], "capacity.required_bytes")
        headroom_bytes = _require_int(operation["headroom_bytes"], "capacity.headroom_bytes", allow_negative=True)
        admissible = operation["admissible"]
        if not isinstance(admissible, bool):
            raise ValueError("capacity.admissible must be boolean")
        if headroom_bytes != free_bytes - required_bytes:
            raise ValueError("capacity arithmetic is inconsistent")
        if capacity_state == "healthy":
            if not admissible or headroom_bytes < 0 or reason != "admissible":
                raise ValueError("healthy capacity fields disagree")
        elif admissible or headroom_bytes >= 0 or reason != "insufficient_headroom":
            raise ValueError("critical capacity fields disagree")

    if attempt_state == "healthy":
        if verified_at != sampled_at:
            raise ValueError("successful proof time must equal sampled_at")
        if capacity_state != "healthy":
            raise ValueError("a healthy attempt requires healthy capacity")
    expected_capacity_state = {
        "verified": "healthy",
        "backup_source_missing": "unknown",
        "capacity_measurement_failed": "unknown",
        "insufficient_headroom": "critical",
        "backup_failed": "healthy",
        "backup_integrity_failed": "healthy",
    }[attempt_reason]
    if capacity_state != expected_capacity_state:
        raise ValueError("last_attempt reason and capacity state disagree")

    # Round-trip detaches caller-owned mutable dictionaries and normalizes JSON types.
    return json.loads(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))


def build_backup_health_snapshot(
    *,
    sampled_at: datetime,
    last_attempt: dict[str, Any],
    last_verified: dict[str, Any] | None,
    capacity: dict[str, Any],
) -> dict[str, Any]:
    """Build a strict snapshot from controlled producer fields."""
    snapshot = {
        "schema_version": BACKUP_HEALTH_SCHEMA_VERSION,
        "sampled_at": _iso_utc(sampled_at),
        "last_attempt": last_attempt,
        "last_verified": last_verified,
        "capacity": capacity,
    }
    return validate_backup_health_snapshot(snapshot)


def read_backup_health_snapshot(path: Path) -> dict[str, Any]:
    """Read one proof with an enforced byte cap and duplicate-key rejection."""
    file_stat = path.stat()
    if not stat_module.S_ISREG(file_stat.st_mode):
        raise ValueError("backup-health snapshot must be a regular file")
    if file_stat.st_size > BACKUP_HEALTH_MAX_BYTES:
        raise ValueError("backup-health snapshot exceeds byte cap")
    with path.open("rb") as handle:
        encoded = handle.read(BACKUP_HEALTH_MAX_BYTES + 1)
    if len(encoded) > BACKUP_HEALTH_MAX_BYTES:
        raise ValueError("backup-health snapshot exceeds byte cap")
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("backup-health snapshot is not valid JSON") from exc
    return validate_backup_health_snapshot(value)


def write_backup_health_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    """Atomically publish one validated proof using a unique same-dir temp."""
    normalized = validate_backup_health_snapshot(snapshot)
    encoded = (json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(encoded) > BACKUP_HEALTH_MAX_BYTES:
        raise ValueError("backup-health snapshot exceeds byte cap")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        committed = True
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            logger.warning("Backup-health directory fsync failed after commit: %s", exc)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not committed:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _empty_response(*, now: datetime, backup_path: Path, status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": BACKUP_HEALTH_SCHEMA_VERSION,
        "evidence_mode": "cached",
        "status": status,
        "reason": reason,
        "timestamp": _iso_utc(now),
        "sampled_at": None,
        "last_attempt": None,
        "last_verified": None,
        "capacity": None,
        "backup_age_hours": None,
        "backup_size_mb": None,
        "concept_count": None,
        "integrity": None,
        "backup_path": str(backup_path),
    }


def build_backup_health_response(data_dir: Path, *, now: datetime) -> dict[str, Any]:
    """Normalize cached proof into the compatibility health response."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UTC)
    backup_path = data_dir / BACKUP_ARTIFACT_NAME
    snapshot_path = data_dir / BACKUP_HEALTH_SNAPSHOT_NAME

    try:
        snapshot = read_backup_health_snapshot(snapshot_path)
    except FileNotFoundError:
        try:
            artifact_stat = backup_path.stat()
        except FileNotFoundError:
            return _empty_response(now=now, backup_path=backup_path, status="critical", reason="no_backup")
        except OSError:
            return _empty_response(
                now=now,
                backup_path=backup_path,
                status="critical",
                reason="invalid_cached_proof",
            )
        artifact_time = datetime.fromtimestamp(artifact_stat.st_mtime, UTC)
        artifact_age_hours = (now - artifact_time).total_seconds() / 3600
        if artifact_age_hours < -(BACKUP_FUTURE_SKEW_SECONDS / 3600):
            return _empty_response(
                now=now,
                backup_path=backup_path,
                status="critical",
                reason="invalid_cached_proof",
            )
        if artifact_age_hours > BACKUP_CRITICAL_AGE_HOURS:
            return _empty_response(
                now=now,
                backup_path=backup_path,
                status="critical",
                reason="unverified_stale_backup",
            )
        return _empty_response(
            now=now,
            backup_path=backup_path,
            status="warning",
            reason="cached_proof_missing",
        )
    except (OSError, ValueError):
        return _empty_response(
            now=now,
            backup_path=backup_path,
            status="critical",
            reason="invalid_cached_proof",
        )

    sampled_at = _parse_timestamp(snapshot["sampled_at"], "sampled_at")
    if sampled_at > now + timedelta(seconds=BACKUP_FUTURE_SKEW_SECONDS):
        return _empty_response(
            now=now,
            backup_path=backup_path,
            status="critical",
            reason="invalid_cached_proof",
        )

    last_verified = snapshot["last_verified"]
    backup_age_hours: float | None = None
    artifact_matches = False
    if last_verified is not None:
        if last_verified["backup_path"] != str(backup_path):
            return _empty_response(
                now=now,
                backup_path=backup_path,
                status="critical",
                reason="invalid_cached_proof",
            )
        verified_at = _parse_timestamp(last_verified["at"], "last_verified.at")
        backup_age_hours = max(0.0, (now - verified_at).total_seconds() / 3600)
        try:
            artifact_stat = backup_path.stat()
        except OSError:
            artifact_stat = None
        artifact_matches = artifact_stat is not None and (
            artifact_stat.st_size == last_verified["backup_size_bytes"]
            and artifact_stat.st_mtime_ns == last_verified["backup_mtime_ns"]
        )

    attempt = snapshot["last_attempt"]
    capacity_state = snapshot["capacity"]["state"]
    if attempt["state"] == "critical":
        status, reason = "critical", attempt["reason_code"]
    elif last_verified is None:
        status, reason = "critical", "no_verified_backup"
    elif not artifact_matches:
        status, reason = "critical", "artifact_changed_since_verification"
    elif backup_age_hours is not None and backup_age_hours > BACKUP_CRITICAL_AGE_HOURS:
        status, reason = "critical", "stale_backup"
    elif capacity_state == "critical":
        status, reason = "critical", "insufficient_headroom"
    elif capacity_state == "unknown":
        status, reason = "critical", "capacity_unknown"
    elif backup_age_hours is not None and backup_age_hours > BACKUP_WARNING_AGE_HOURS:
        status, reason = "warning", "aging_backup"
    else:
        status, reason = "healthy", "ok"

    response = {
        "schema_version": BACKUP_HEALTH_SCHEMA_VERSION,
        "evidence_mode": "cached",
        "status": status,
        "reason": reason,
        "timestamp": _iso_utc(now),
        "sampled_at": snapshot["sampled_at"],
        "last_attempt": attempt,
        "last_verified": last_verified,
        "capacity": snapshot["capacity"],
        "backup_age_hours": None,
        "backup_size_mb": None,
        "concept_count": None,
        "integrity": None,
        "backup_path": str(backup_path),
    }
    if last_verified is not None:
        response.update(
            {
                "backup_age_hours": round(backup_age_hours or 0.0, 1),
                "backup_size_mb": round(last_verified["backup_size_bytes"] / 1024 / 1024, 1),
                "concept_count": last_verified["concept_count"],
                "integrity": last_verified["integrity"],
            }
        )
    return response


__all__ = [
    "BACKUP_ARTIFACT_NAME",
    "BACKUP_ATTEMPT_REASON_CODES",
    "BACKUP_CRITICAL_AGE_HOURS",
    "BACKUP_HEALTH_MAX_BYTES",
    "BACKUP_HEALTH_SCHEMA_VERSION",
    "BACKUP_HEALTH_SNAPSHOT_NAME",
    "BACKUP_OPERATION_ID",
    "BACKUP_RESERVE_BYTES",
    "BACKUP_RESPONSE_REASON_CODES",
    "BACKUP_WARNING_AGE_HOURS",
    "SIGNED_64_MAX",
    "build_backup_health_response",
    "build_backup_health_snapshot",
    "measure_backup_capacity",
    "read_backup_health_snapshot",
    "unknown_backup_capacity",
    "validate_backup_health_snapshot",
    "write_backup_health_snapshot",
]
