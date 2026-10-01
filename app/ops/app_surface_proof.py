"""Signed exact-host proof contract for app-surface parity release gates."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEDGER_SCHEMA_VERSION = "pith_app_surface_proof_ledger.v2"
ROW_SCHEMA_VERSION = "pith_app_surface_proof.v2"
CONNECTION_PROOF_SCHEMA_VERSION = "pith_connection_proof.v1"
MAX_LEDGER_BYTES = 1024 * 1024
MAX_LEDGER_ROWS = 16
MAX_PROOF_AGE = dt.timedelta(hours=24)
MAX_FUTURE_SKEW = dt.timedelta(minutes=5)
LOCK_STALE_SECONDS = 10 * 60
ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_SURFACE_VARIANTS = (
    ("chatgpt", "chat"),
    ("chatgpt", "work"),
    ("claude_chat", "desktop"),
    ("claude_code", "code"),
)

ROW_FIELDS = {
    "schema_version",
    "trust_tier",
    "run_id",
    "target_os",
    "target_os_build",
    "surface_id",
    "host_variant",
    "host_version",
    "host_launch_at",
    "observed_at",
    "artifact_sha256",
    "package_version",
    "source_public_digest",
    "installed_public_digest",
    "tool_inventory",
    "evidence_source",
    "raw_response_sha256",
    "bind_status",
    "resolved_session_id",
    "session_active",
    "session_active_source",
    "auth_error",
    "key_id",
    "signature",
}


class AppSurfaceProofError(ValueError):
    """A classified proof-ledger failure safe for support output."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProofExpectations:
    run_id: str
    artifact_sha256: str
    package_version: str
    target_os_build: str
    host_versions: dict[str, str]
    key_id: str


def pair_key(surface_id: str, host_variant: str) -> str:
    return f"{surface_id}/{host_variant}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_signed_payload(row: dict[str, Any]) -> bytes:
    return _canonical_json({key: value for key, value in row.items() if key != "signature"}).encode("utf-8")


def _key_bytes(key: str | bytes) -> bytes:
    value = key.encode("utf-8") if isinstance(key, str) else key
    if len(value) < 32:
        raise AppSurfaceProofError("invalid_attestation_key", "Release proof key must contain at least 32 bytes.")
    return value


def sign_proof_row(row: dict[str, Any], key: str | bytes) -> dict[str, Any]:
    signed = dict(row)
    signed.pop("signature", None)
    signed["signature"] = hmac.new(_key_bytes(key), _canonical_signed_payload(signed), hashlib.sha256).hexdigest()
    return signed


def _parse_timestamp(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise AppSurfaceProofError("invalid_timestamp", f"{field} must be a bounded ISO-8601 string.")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AppSurfaceProofError("invalid_timestamp", f"{field} is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AppSurfaceProofError("invalid_timestamp", f"{field} must be timezone-aware.")
    return parsed.astimezone(dt.UTC)


def _require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise AppSurfaceProofError("invalid_identifier", f"{field} is invalid.")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or not value.isprintable():
        raise AppSurfaceProofError("invalid_text", f"{field} must contain 1-128 printable characters.")
    return value


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise AppSurfaceProofError("invalid_digest", f"{field} must be lowercase SHA-256 hex.")
    return value


def _validate_row_shape(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise AppSurfaceProofError("invalid_row", "Proof rows must be JSON objects.")
    missing = sorted(ROW_FIELDS - set(row))
    extra = sorted(set(row) - ROW_FIELDS)
    if missing or extra:
        raise AppSurfaceProofError("invalid_row_fields", "Proof row fields do not match the schema.")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise AppSurfaceProofError("invalid_row_schema", "Proof row schema version is unsupported.")
    _require_id(row.get("run_id"), "run_id")
    _require_id(row.get("surface_id"), "surface_id")
    _require_id(row.get("host_variant"), "host_variant")
    _require_id(row.get("key_id"), "key_id")
    _require_text(row.get("target_os_build"), "target_os_build")
    _require_text(row.get("host_version"), "host_version")
    _require_text(row.get("package_version"), "package_version")
    _require_id(row.get("resolved_session_id"), "resolved_session_id")
    _require_digest(row.get("artifact_sha256"), "artifact_sha256")
    _require_digest(row.get("source_public_digest"), "source_public_digest")
    _require_digest(row.get("installed_public_digest"), "installed_public_digest")
    _require_digest(row.get("raw_response_sha256"), "raw_response_sha256")
    signature = row.get("signature")
    if not isinstance(signature, str) or SIGNATURE_RE.fullmatch(signature) is None:
        raise AppSurfaceProofError("invalid_signature_shape", "Proof signature must be lowercase HMAC-SHA256 hex.")
    if row.get("target_os") != "windows":
        raise AppSurfaceProofError("target_os_mismatch", "Proof target OS must be windows.")
    inventory = row.get("tool_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {"tool_name", "exposed", "call_completed"}:
        raise AppSurfaceProofError("invalid_tool_inventory", "Tool inventory fields are invalid.")
    if (
        inventory.get("tool_name") != "pith_conversation_turn"
        or inventory.get("exposed") is not True
        or inventory.get("call_completed") is not True
    ):
        raise AppSurfaceProofError("tool_not_proven", "Pith conversation-turn exposure and completion are required.")
    return row


def validate_proof_ledger(
    payload: Any,
    *,
    expectations: ProofExpectations,
    key: str | bytes,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "run_id", "generated_at", "rows"}:
        raise AppSurfaceProofError("invalid_ledger_fields", "Proof ledger fields do not match the schema.")
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise AppSurfaceProofError("invalid_ledger_schema", "Proof ledger schema version is unsupported.")
    run_id = _require_id(payload.get("run_id"), "ledger.run_id")
    if run_id != expectations.run_id:
        raise AppSurfaceProofError("run_id_mismatch", "Proof ledger run ID does not match the expected run.")
    generated = _parse_timestamp(payload.get("generated_at"), "generated_at")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) > MAX_LEDGER_ROWS:
        raise AppSurfaceProofError("invalid_row_count", "Proof ledger row count is invalid.")
    expected_pairs = {pair_key(*pair) for pair in REQUIRED_SURFACE_VARIANTS}
    seen: set[str] = set()
    seen_raw_responses: set[str] = set()
    seen_session_ids: set[str] = set()
    validated: dict[str, dict[str, Any]] = {}
    instant = now or dt.datetime.now(dt.UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise AppSurfaceProofError("invalid_clock", "Validation clock must be timezone-aware.")
    instant = instant.astimezone(dt.UTC)
    if generated > instant + MAX_FUTURE_SKEW:
        raise AppSurfaceProofError("future_ledger", "Proof ledger generation time is too far in the future.")
    secret = _key_bytes(key)
    for raw_row in rows:
        row = _validate_row_shape(raw_row)
        pair = pair_key(str(row["surface_id"]), str(row["host_variant"]))
        if pair not in expected_pairs:
            raise AppSurfaceProofError("unknown_surface_variant", "Proof row surface/variant is not supported.")
        if pair in seen:
            raise AppSurfaceProofError(
                "duplicate_surface_variant", "Proof ledger contains a duplicate surface/variant."
            )
        seen.add(pair)
        raw_response_sha256 = str(row["raw_response_sha256"])
        if raw_response_sha256 in seen_raw_responses:
            raise AppSurfaceProofError(
                "duplicate_raw_response", "Each required surface must provide a distinct conversation-turn response."
            )
        seen_raw_responses.add(raw_response_sha256)
        resolved_session_id = str(row["resolved_session_id"])
        if resolved_session_id in seen_session_ids:
            raise AppSurfaceProofError(
                "duplicate_resolved_session", "Each required surface must resolve a distinct Pith session."
            )
        seen_session_ids.add(resolved_session_id)
        if row["trust_tier"] != "release_attested":
            raise AppSurfaceProofError("diagnostic_not_promotable", "Diagnostic proof cannot promote a release claim.")
        if row["key_id"] != expectations.key_id:
            raise AppSurfaceProofError("key_id_mismatch", "Proof key ID does not match the expected key.")
        expected_signature = hmac.new(secret, _canonical_signed_payload(row), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(str(row["signature"]), expected_signature):
            raise AppSurfaceProofError("invalid_signature", "Proof row signature verification failed.")
        if row["run_id"] != expectations.run_id:
            raise AppSurfaceProofError("run_id_mismatch", "Proof row run ID does not match the expected run.")
        if row["artifact_sha256"] != expectations.artifact_sha256:
            raise AppSurfaceProofError("artifact_mismatch", "Proof artifact digest does not match.")
        if row["package_version"] != expectations.package_version:
            raise AppSurfaceProofError("package_version_mismatch", "Proof package version does not match.")
        if row["target_os_build"] != expectations.target_os_build:
            raise AppSurfaceProofError("target_os_build_mismatch", "Proof target OS build does not match.")
        if row["host_version"] != expectations.host_versions.get(pair):
            raise AppSurfaceProofError("host_version_mismatch", "Proof host version does not match.")
        if row["source_public_digest"] != row["installed_public_digest"]:
            raise AppSurfaceProofError("source_digest_mismatch", "Source and installed public digests do not match.")
        launched = _parse_timestamp(row["host_launch_at"], "host_launch_at")
        observed = _parse_timestamp(row["observed_at"], "observed_at")
        if observed < launched:
            raise AppSurfaceProofError("observation_before_launch", "Proof observation predates host launch.")
        if observed > instant + MAX_FUTURE_SKEW:
            raise AppSurfaceProofError("future_observation", "Proof observation is too far in the future.")
        if instant - observed > MAX_PROOF_AGE:
            raise AppSurfaceProofError("stale_observation", "Proof observation is older than 24 hours.")
        if row["evidence_source"] != "current_conversation_turn_response":
            raise AppSurfaceProofError("invalid_evidence_source", "Proof must come from the current conversation turn.")
        if row["bind_status"] != "bound" or row["session_active"] is not True:
            raise AppSurfaceProofError("session_not_active", "Proof does not contain a bound active session.")
        if row["session_active_source"] != "conversation_turn._protocol.session_active":
            raise AppSurfaceProofError(
                "inferred_session_active", "Release proof requires direct protocol session state."
            )
        if row["auth_error"] is not None:
            raise AppSurfaceProofError("auth_failed", "Proof contains an authentication error.")
        validated[pair] = row
    missing = sorted(expected_pairs - seen)
    if missing:
        raise AppSurfaceProofError("missing_surface_variant", "Proof ledger is incomplete.")
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": "valid",
        "evidence_tier": "release_attested",
        "evidence_freshness": "fresh",
        "artifact_match": True,
        "package_version_match": True,
        "host_version_match": True,
        "attestation_status": "valid",
        "run_id": run_id,
        "connected_pairs": sorted(validated),
        "operator_attestation_limitation": (
            "HMAC authenticates the release operator record; it does not cryptographically identify the app UI mode."
        ),
    }


def load_proof_ledger(
    path: str | Path,
    *,
    expectations: ProofExpectations,
    key: str | bytes,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    ledger_path = Path(path).expanduser()
    if ledger_path.is_symlink():
        raise AppSurfaceProofError("ledger_symlink_rejected", "Proof ledger symlinks are not accepted.")
    try:
        size = ledger_path.stat().st_size
    except OSError as exc:
        raise AppSurfaceProofError("ledger_unreadable", "Proof ledger is not readable.") from exc
    if size > MAX_LEDGER_BYTES:
        raise AppSurfaceProofError("ledger_oversized", "Proof ledger exceeds the size limit.")
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AppSurfaceProofError("ledger_invalid_json", "Proof ledger is not valid JSON.") from exc
    return validate_proof_ledger(payload, expectations=expectations, key=key, now=now)


def project_connection_proof(raw_proof: Any, metadata: Any, *, key_id: str) -> dict[str, Any]:
    if not isinstance(raw_proof, dict) or raw_proof.get("schema_version") != CONNECTION_PROOF_SCHEMA_VERSION:
        raise AppSurfaceProofError("invalid_connection_proof", "Input must be a pith_connection_proof.v1 object.")
    if not isinstance(metadata, dict):
        raise AppSurfaceProofError("invalid_metadata", "Proof metadata must be an object.")
    required_metadata = {
        "run_id",
        "target_os_build",
        "surface_id",
        "host_variant",
        "host_version",
        "host_launch_at",
        "observed_at",
        "artifact_sha256",
        "package_version",
        "source_public_digest",
        "installed_public_digest",
    }
    if set(metadata) != required_metadata:
        raise AppSurfaceProofError("invalid_metadata_fields", "Proof metadata fields do not match the schema.")
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "trust_tier": "release_attested",
        **metadata,
        "target_os": "windows",
        "tool_inventory": {
            "tool_name": "pith_conversation_turn",
            "exposed": True,
            "call_completed": raw_proof.get("same_turn") is True,
        },
        "evidence_source": raw_proof.get("evidence_source"),
        "raw_response_sha256": hashlib.sha256(_canonical_json(raw_proof).encode("utf-8")).hexdigest(),
        "bind_status": raw_proof.get("bind_status"),
        "resolved_session_id": raw_proof.get("resolved_session_id"),
        "session_active": raw_proof.get("session_active"),
        "session_active_source": raw_proof.get("session_active_source"),
        "auth_error": raw_proof.get("auth_error"),
        "key_id": key_id,
        "signature": "0" * 64,
    }
    _validate_row_shape(row)
    pair = pair_key(str(row["surface_id"]), str(row["host_variant"]))
    if pair not in {pair_key(*item) for item in REQUIRED_SURFACE_VARIANTS}:
        raise AppSurfaceProofError("unknown_surface_variant", "Proof metadata surface/variant is unsupported.")
    if row["evidence_source"] != "current_conversation_turn_response":
        raise AppSurfaceProofError("invalid_evidence_source", "Proof must come from the current conversation turn.")
    if row["bind_status"] != "bound" or row["session_active"] is not True:
        raise AppSurfaceProofError("session_not_active", "Proof does not contain a bound active session.")
    if row["session_active_source"] != "conversation_turn._protocol.session_active":
        raise AppSurfaceProofError("inferred_session_active", "Release proof requires direct protocol session state.")
    if row["auth_error"] is not None:
        raise AppSurfaceProofError("auth_failed", "Proof contains an authentication error.")
    return row


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ProofLedgerLock:
    def __init__(self, ledger_path: Path):
        self.path = ledger_path.with_name(f"{ledger_path.name}.lock")
        self.token = secrets.token_hex(16)

    def __enter__(self) -> ProofLedgerLock:
        payload = {
            "pid": os.getpid(),
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "owner_token": self.token,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            self._reclaim_stale()
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise AppSurfaceProofError("ledger_locked", "Proof ledger is locked by another capture.") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
        return self

    def _reclaim_stale(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            created = _parse_timestamp(payload.get("created_at"), "lock.created_at")
            pid = int(payload.get("pid"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, AppSurfaceProofError):
            return
        age = dt.datetime.now(dt.UTC) - created
        if age > dt.timedelta(seconds=LOCK_STALE_SECONDS) and not _pid_alive(pid):
            try:
                self.path.unlink()
            except OSError:
                return

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("owner_token") == self.token:
                self.path.unlink()
        except (OSError, json.JSONDecodeError):
            return


def capture_proof_row(
    *,
    raw_proof: Any,
    metadata: Any,
    ledger_path: str | Path,
    key: str | bytes,
    key_id: str,
) -> dict[str, Any]:
    path = Path(ledger_path).expanduser()
    if path.is_symlink():
        raise AppSurfaceProofError("ledger_symlink_rejected", "Proof ledger symlinks are not accepted.")
    row = sign_proof_row(project_connection_proof(raw_proof, metadata, key_id=key_id), key)
    with ProofLedgerLock(path):
        if path.exists():
            if path.stat().st_size > MAX_LEDGER_BYTES:
                raise AppSurfaceProofError("ledger_oversized", "Proof ledger exceeds the size limit.")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise AppSurfaceProofError("ledger_invalid_json", "Proof ledger is not valid JSON.") from exc
            if not isinstance(payload, dict) or set(payload) != {"schema_version", "run_id", "generated_at", "rows"}:
                raise AppSurfaceProofError("invalid_ledger_fields", "Proof ledger fields do not match the schema.")
            if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise AppSurfaceProofError("invalid_ledger_schema", "Proof ledger schema version is unsupported.")
            if payload.get("run_id") != row["run_id"] or not isinstance(payload.get("rows"), list):
                raise AppSurfaceProofError("run_id_mismatch", "Existing proof ledger belongs to another run.")
            _parse_timestamp(payload.get("generated_at"), "generated_at")
            if len(payload["rows"]) > MAX_LEDGER_ROWS:
                raise AppSurfaceProofError("invalid_row_count", "Proof ledger row count exceeds the limit.")
            rows = []
            existing_pairs: set[str] = set()
            secret = _key_bytes(key)
            for item in payload["rows"]:
                existing = _validate_row_shape(item)
                existing_pair = pair_key(str(existing["surface_id"]), str(existing["host_variant"]))
                if existing_pair not in {pair_key(*pair) for pair in REQUIRED_SURFACE_VARIANTS}:
                    raise AppSurfaceProofError("unknown_surface_variant", "Existing proof row is unsupported.")
                if existing_pair in existing_pairs:
                    raise AppSurfaceProofError(
                        "duplicate_surface_variant", "Existing proof ledger contains duplicates."
                    )
                existing_pairs.add(existing_pair)
                if existing["run_id"] != row["run_id"] or existing["key_id"] != key_id:
                    raise AppSurfaceProofError("run_id_mismatch", "Existing proof row belongs to another run or key.")
                signature = hmac.new(secret, _canonical_signed_payload(existing), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(str(existing["signature"]), signature):
                    raise AppSurfaceProofError("invalid_signature", "Existing proof row signature is invalid.")
                rows.append(existing)
        else:
            rows = []
        row_pair = pair_key(str(row["surface_id"]), str(row["host_variant"]))
        rows = [
            item for item in rows if pair_key(str(item.get("surface_id")), str(item.get("host_variant"))) != row_pair
        ]
        rows.append(row)
        if len(rows) > MAX_LEDGER_ROWS:
            raise AppSurfaceProofError("invalid_row_count", "Proof ledger row count exceeds the limit.")
        rows.sort(key=lambda item: pair_key(str(item.get("surface_id")), str(item.get("host_variant"))))
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "run_id": row["run_id"],
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "rows": rows,
        }
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        if len(encoded) > MAX_LEDGER_BYTES:
            raise AppSurfaceProofError("ledger_oversized", "Proof ledger exceeds the size limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
    return {"status": "captured", "pair": row_pair, "ledger_path": str(path), "row_count": len(rows)}
