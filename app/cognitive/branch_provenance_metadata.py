"""Observe-only branch provenance metadata helpers.

This module is intentionally not wired into retrieval ranking or answer
construction. It provides a product-shaped metadata envelope for validating
branch provenance fixtures before any schema or runtime authority work.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

BRANCH_PROVENANCE_METADATA_KEY = "branch_provenance"
BRANCH_AUTHORITY_EVENT_METADATA_KEY = "branch_authority_event"
REQUIRED_BRANCH_PROVENANCE_FIELDS = (
    "assertion_key",
    "conflict_group_key",
    "source_event_id",
    "source_sequence",
    "source_hash",
    "source_type",
    "conflict_resolution_state",
    "branch_resolution_state",
)
REQUIRED_BRANCH_AUTHORITY_EVENT_FIELDS = (
    "authority_event_id",
    "authority_event_type",
    "authority_reason",
    "authority_policy_version",
    "conflict_group_key",
    "selected_assertion_key",
    "superseded_assertion_keys",
    "contested_assertion_keys",
    "authority_source_event_id",
    "authority_source_hash",
    "resolved_at",
    "resolved_by",
)

_VALID_STATES = {"active", "superseded", "contested", "unresolved", "deferred"}
_VALID_BRANCH_RESOLUTION_STATES = {
    "unresolved",
    "selected_authoritative",
    "superseded",
    "contested",
    "deferred",
}
_VALID_AUTHORITY_EVENT_TYPES = {
    "explicit_user_correction",
    "explicit_supersession",
    "owner_assertion",
    "manual_resolution",
    "trusted_policy_resolution",
}
_EXPLICIT_RESOLUTION_AUTHORITY_EVENT_TYPES = frozenset(
    {
        "explicit_user_correction",
        "explicit_supersession",
        "manual_resolution",
        "trusted_policy_resolution",
    }
)
_SINGLE_VALUED_BRANCH_PREDICATES = frozenset(
    {
        "religion_affiliation",
        "founded_by",
        "born_in_city",
        "created_by",
        "chairperson",
    }
)
_BENCHMARK_PRIVATE_RE = re.compile(r"\b(?:no\d+|gold_answer|expected_answer|expected_source_refs|score_label)\b", re.I)
_SOURCE_ORDER_AUTHORITY_RE = re.compile(
    r"\b(?:source[-_\s]?order|source_sequence|first source|latest source|highest rank|stable rank|ranked first|row id|score delta)\b",
    re.I,
)
_TAG_STRIP_RE = re.compile(r"\[[A-Z][A-Z0-9_-]*(?::[^\]]+)?\]\s*")
_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(.+?) is affiliated with the religion of (.+)$", re.I), "religion_affiliation"),
    (re.compile(r"^(.+?) was founded by (.+)$", re.I), "founded_by"),
    (re.compile(r"^(.+?) was born in the city of (.+)$", re.I), "born_in_city"),
    (re.compile(r"^(.+?) was created by (.+)$", re.I), "created_by"),
    (re.compile(r"^(.+?) is famous for (.+)$", re.I), "famous_for"),
    (re.compile(r"^(.+?) is employed by (.+)$", re.I), "employed_by"),
    (re.compile(r"^The chairperson of (.+?) is (.+)$", re.I), "chairperson"),
    (re.compile(r"^(.+?) works in the field of (.+)$", re.I), "works_field"),
    (re.compile(r"^(.+?) worked in the city of (.+)$", re.I), "worked_city"),
)


@dataclass(frozen=True)
class ParsedAssertion:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class BranchProvenanceMetadata:
    assertion_key: str
    conflict_group_key: str
    source_event_id: str
    source_sequence: int
    source_hash: str
    source_span: str | None
    source_type: str
    authority_reason: str | None
    branch_authority_event: dict[str, Any] | None
    conflict_resolution_state: str
    branch_resolution_state: str
    subject: str
    predicate: str
    object: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchProvenanceValidation:
    ready: bool
    missing_fields: tuple[str, ...]
    invalid_reasons: tuple[str, ...]
    assertion_key: str | None
    conflict_group_key: str | None
    conflict_resolution_state: str | None
    branch_resolution_state: str | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing_fields": list(self.missing_fields),
            "invalid_reasons": list(self.invalid_reasons),
            "assertion_key": self.assertion_key,
            "conflict_group_key": self.conflict_group_key,
            "conflict_resolution_state": self.conflict_resolution_state,
            "branch_resolution_state": self.branch_resolution_state,
        }


BranchAuthorityPatchStatus = Literal["ready", "abstained", "invalid_input"]


@dataclass(frozen=True)
class BranchAuthorityMetadataPatchResult:
    selected_metadata_patch: dict[str, Any]
    superseded_metadata_patch: dict[str, Any]
    status: BranchAuthorityPatchStatus
    reason: str


def normalize_branch_value(value: str) -> str:
    text = _TAG_STRIP_RE.sub("", str(value or "")).strip().lower().replace("'", "")
    text = text.replace("’", "")
    text = re.sub(r"^[\"'“”]+|[\"'“”.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return text.strip()


def parse_fixture_assertion(summary: str) -> ParsedAssertion:
    text = _TAG_STRIP_RE.sub("", str(summary or "")).strip().rstrip(".")
    for pattern, predicate in _FACT_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        subject = normalize_branch_value(match.group(1))
        obj = normalize_branch_value(match.group(2))
        if subject and obj and subject != obj:
            return ParsedAssertion(subject=subject, predicate=predicate, object=obj)
    raise ValueError(f"Unsupported branch provenance fixture fact: {summary!r}")


def _stable_key(prefix: str, *parts: str) -> str:
    canonical = json.dumps([normalize_branch_value(part) for part in parts], separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def build_source_hash(source_event_id: str, source_text: str) -> str:
    canonical = json.dumps(
        {
            "source_event_id": str(source_event_id or ""),
            "source_text": str(source_text or ""),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _predicate_allows_single_winner(predicate: str | None) -> bool:
    return str(predicate or "") in _SINGLE_VALUED_BRANCH_PREDICATES


def _authority_event_type_allows_explicit_resolution(event_type: str | None) -> bool:
    return str(event_type or "") in _EXPLICIT_RESOLUTION_AUTHORITY_EVENT_TYPES


def _predicate_allows_selected_authoritative(
    predicate: str | None,
    *,
    authority_event_type: str | None = None,
) -> bool:
    return _predicate_allows_single_winner(predicate) or _authority_event_type_allows_explicit_resolution(
        authority_event_type
    )


def _contains_benchmark_private_marker(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_BENCHMARK_PRIVATE_RE.search(value))
    if isinstance(value, dict):
        return any(
            _contains_benchmark_private_marker(item)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_benchmark_private_marker(item) for item in value)
    return False


def _validate_branch_authority_event(
    event: Any,
    *,
    assertion_key: str | None,
    conflict_group_key: str | None,
    predicate: str | None,
    branch_resolution_state: str | None,
) -> list[str]:
    if not isinstance(event, dict):
        return ["selected_authoritative requires branch_authority_event"]

    invalid: list[str] = []
    for field in REQUIRED_BRANCH_AUTHORITY_EVENT_FIELDS:
        value = event.get(field)
        if value is None or value == "":
            invalid.append(f"branch_authority_event missing {field}")

    event_type = event.get("authority_event_type")
    if event_type not in _VALID_AUTHORITY_EVENT_TYPES:
        invalid.append("branch_authority_event authority_event_type must be valid")
    if _contains_benchmark_private_marker(event):
        invalid.append("branch_authority_event must not contain benchmark-private markers")
    if event.get("conflict_group_key") != conflict_group_key:
        invalid.append("branch_authority_event conflict_group_key must match branch provenance")
    if not _predicate_allows_selected_authoritative(predicate, authority_event_type=event_type):
        invalid.append(
            "selected_authoritative requires a single-valued predicate policy "
            "or explicit authority resolution event"
        )

    selected_assertion_key = event.get("selected_assertion_key")
    superseded_assertion_keys = event.get("superseded_assertion_keys")
    contested_assertion_keys = event.get("contested_assertion_keys")
    if not isinstance(superseded_assertion_keys, list):
        invalid.append("branch_authority_event superseded_assertion_keys must be a list")
    if not isinstance(contested_assertion_keys, list):
        invalid.append("branch_authority_event contested_assertion_keys must be a list")

    if branch_resolution_state == "selected_authoritative" and selected_assertion_key != assertion_key:
        invalid.append("selected_authoritative assertion must match branch_authority_event selection")
    if branch_resolution_state == "superseded" and isinstance(superseded_assertion_keys, list):
        if assertion_key not in superseded_assertion_keys:
            invalid.append("superseded assertion must be listed in branch_authority_event")
    if branch_resolution_state == "contested" and isinstance(contested_assertion_keys, list):
        if assertion_key not in contested_assertion_keys:
            invalid.append("contested assertion must be listed in branch_authority_event")

    return invalid


def derive_branch_resolution_state(
    *,
    conflict_resolution_state: str,
    authority_reason: str | None = None,
    branch_resolution_state: str | None = None,
    branch_authority_event: dict[str, Any] | None = None,
    predicate: str | None = None,
) -> str:
    if branch_resolution_state:
        if branch_resolution_state not in _VALID_BRANCH_RESOLUTION_STATES:
            raise ValueError(f"invalid branch_resolution_state: {branch_resolution_state}")
        if branch_resolution_state == "selected_authoritative":
            if conflict_resolution_state != "active":
                raise ValueError("selected_authoritative requires active conflict_resolution_state")
            if not branch_authority_event:
                raise ValueError("selected_authoritative requires branch_authority_event")
            if not _predicate_allows_selected_authoritative(
                predicate,
                authority_event_type=branch_authority_event.get("authority_event_type"),
            ):
                raise ValueError(
                    "selected_authoritative requires a single-valued predicate policy "
                    "or explicit authority resolution event"
                )
        return branch_resolution_state
    _ = authority_reason
    if conflict_resolution_state in {"superseded", "contested", "deferred"}:
        return conflict_resolution_state
    return "unresolved"


def build_branch_provenance_metadata(
    summary: str,
    *,
    source_event_id: str,
    source_sequence: int,
    source_text: str | None = None,
    source_span: str | None = None,
    source_type: str = "synthetic_fixture",
    authority_reason: str | None = None,
    branch_authority_event: dict[str, Any] | None = None,
    conflict_resolution_state: str = "active",
    branch_resolution_state: str | None = None,
) -> BranchProvenanceMetadata:
    if conflict_resolution_state not in _VALID_STATES:
        raise ValueError(f"invalid conflict_resolution_state: {conflict_resolution_state}")
    if source_sequence < 0:
        raise ValueError("source_sequence must be non-negative")
    parsed = parse_fixture_assertion(summary)
    derived_branch_state = derive_branch_resolution_state(
        conflict_resolution_state=conflict_resolution_state,
        authority_reason=authority_reason,
        branch_resolution_state=branch_resolution_state,
        branch_authority_event=branch_authority_event,
        predicate=parsed.predicate,
    )
    assertion_key = _stable_key("assertion", parsed.subject, parsed.predicate, parsed.object)
    conflict_group_key = _stable_key("conflict_group", parsed.subject, parsed.predicate)
    if derived_branch_state in {"selected_authoritative", "superseded", "contested"} and branch_authority_event:
        invalid = _validate_branch_authority_event(
            branch_authority_event,
            assertion_key=assertion_key,
            conflict_group_key=conflict_group_key,
            predicate=parsed.predicate,
            branch_resolution_state=derived_branch_state,
        )
        if invalid:
            raise ValueError("; ".join(invalid))
    return BranchProvenanceMetadata(
        assertion_key=assertion_key,
        conflict_group_key=conflict_group_key,
        source_event_id=source_event_id,
        source_sequence=source_sequence,
        source_hash=build_source_hash(source_event_id, source_text or summary),
        source_span=source_span,
        source_type=source_type,
        authority_reason=authority_reason,
        branch_authority_event=dict(branch_authority_event) if branch_authority_event else None,
        conflict_resolution_state=conflict_resolution_state,
        branch_resolution_state=derived_branch_state,
        subject=parsed.subject,
        predicate=parsed.predicate,
        object=parsed.object,
    )


def build_branch_authority_event(
    *,
    selected_envelope: dict[str, Any],
    superseded_envelopes: list[dict[str, Any]] | None = None,
    contested_envelopes: list[dict[str, Any]] | None = None,
    authority_event_type: str,
    authority_reason: str,
    authority_source_event_id: str,
    resolved_at: str,
    resolved_by: str,
    authority_policy_version: str = "branch_authority_event_v1",
) -> dict[str, Any]:
    """Build a validated product-shaped branch authority event.

    This is a pure construction helper for future write paths. It does not
    mutate envelopes, storage, retrieval ranking, or answer construction.
    """

    selected = _validated_branch_authority_envelope(selected_envelope, role="selected")
    superseded = [
        _validated_branch_authority_envelope(envelope, role="superseded")
        for envelope in (superseded_envelopes or [])
    ]
    contested = [
        _validated_branch_authority_envelope(envelope, role="contested")
        for envelope in (contested_envelopes or [])
    ]
    siblings = [*superseded, *contested]

    conflict_group_key = selected["conflict_group_key"]
    predicate = selected.get("predicate")
    if not _predicate_allows_selected_authoritative(predicate, authority_event_type=authority_event_type):
        raise ValueError("branch authority event requires a single-valued predicate policy or explicit authority event")
    for envelope in siblings:
        if envelope.get("conflict_group_key") != conflict_group_key:
            raise ValueError("branch authority event envelopes must share conflict_group_key")

    required_inputs = {
        "authority_event_type": authority_event_type,
        "authority_reason": authority_reason,
        "authority_source_event_id": authority_source_event_id,
        "resolved_at": resolved_at,
        "resolved_by": resolved_by,
        "authority_policy_version": authority_policy_version,
    }
    for field, value in required_inputs.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"branch authority event missing {field}")
    if authority_event_type not in _VALID_AUTHORITY_EVENT_TYPES:
        raise ValueError("branch authority event authority_event_type must be valid")
    if _contains_benchmark_private_marker(required_inputs):
        raise ValueError("branch authority event must not contain benchmark-private markers")
    if _SOURCE_ORDER_AUTHORITY_RE.search(authority_reason):
        raise ValueError("source-order or rank signals are not valid branch authority")

    selected_key = selected["assertion_key"]
    superseded_keys = [envelope["assertion_key"] for envelope in superseded]
    contested_keys = [envelope["assertion_key"] for envelope in contested]
    source_payload = {
        "authority_event_type": authority_event_type.strip(),
        "authority_reason": authority_reason.strip(),
        "authority_source_event_id": authority_source_event_id.strip(),
        "conflict_group_key": conflict_group_key,
        "selected_assertion_key": selected_key,
        "superseded_assertion_keys": superseded_keys,
        "contested_assertion_keys": contested_keys,
        "resolved_at": resolved_at.strip(),
        "resolved_by": resolved_by.strip(),
        "authority_policy_version": authority_policy_version.strip(),
    }
    canonical_payload = json.dumps(source_payload, separators=(",", ":"), sort_keys=True)
    event = {
        "authority_event_id": _stable_key(
            "authority_event",
            conflict_group_key,
            selected_key,
            authority_event_type,
            authority_source_event_id,
            authority_reason,
        ),
        "authority_event_type": authority_event_type.strip(),
        "authority_reason": authority_reason.strip(),
        "authority_policy_version": authority_policy_version.strip(),
        "conflict_group_key": conflict_group_key,
        "selected_assertion_key": selected_key,
        "superseded_assertion_keys": superseded_keys,
        "contested_assertion_keys": contested_keys,
        "authority_source_event_id": authority_source_event_id.strip(),
        "authority_source_hash": build_source_hash(authority_source_event_id.strip(), canonical_payload),
        "resolved_at": resolved_at.strip(),
        "resolved_by": resolved_by.strip(),
    }
    invalid = _validate_branch_authority_event(
        event,
        assertion_key=selected_key,
        conflict_group_key=conflict_group_key,
        predicate=predicate,
        branch_resolution_state="selected_authoritative",
    )
    if invalid:
        raise ValueError("; ".join(invalid))
    return event


def build_supersession_branch_authority_metadata_patches(
    *,
    selected_metadata: Mapping[str, Any] | None,
    superseded_metadata: Mapping[str, Any] | None,
    authority_source_event_id: str,
    authority_reason: str,
    resolved_at: str,
    resolved_by: str,
) -> BranchAuthorityMetadataPatchResult:
    """Build metadata patches for an explicit product supersession event.

    Authority application is fail-closed and side-effect-free: callers decide
    whether and how to persist returned patches.
    """

    selected_envelope = extract_branch_provenance_metadata(selected_metadata)
    superseded_envelope = extract_branch_provenance_metadata(superseded_metadata)
    if not selected_envelope or not superseded_envelope:
        return BranchAuthorityMetadataPatchResult({}, {}, "abstained", "missing_branch_provenance")

    selected_validation = validate_branch_provenance_metadata(selected_envelope)
    superseded_validation = validate_branch_provenance_metadata(superseded_envelope)
    if not selected_validation.ready or not superseded_validation.ready:
        return BranchAuthorityMetadataPatchResult({}, {}, "invalid_input", "branch_provenance_not_ready")

    if selected_validation.conflict_group_key != superseded_validation.conflict_group_key:
        return BranchAuthorityMetadataPatchResult({}, {}, "invalid_input", "conflict_group_mismatch")

    try:
        authority_event = build_branch_authority_event(
            selected_envelope=selected_envelope,
            superseded_envelopes=[superseded_envelope],
            authority_event_type="explicit_supersession",
            authority_reason=authority_reason,
            authority_source_event_id=authority_source_event_id,
            resolved_at=resolved_at,
            resolved_by=resolved_by,
        )
        selected_updated, superseded_updated = apply_branch_authority_event(
            [selected_envelope, superseded_envelope],
            authority_event=authority_event,
        )
    except ValueError as exc:
        return BranchAuthorityMetadataPatchResult({}, {}, "invalid_input", str(exc))

    return BranchAuthorityMetadataPatchResult(
        selected_metadata_patch={BRANCH_PROVENANCE_METADATA_KEY: selected_updated},
        superseded_metadata_patch={BRANCH_PROVENANCE_METADATA_KEY: superseded_updated},
        status="ready",
        reason="authority_event_applied",
    )


def apply_branch_authority_event(
    envelopes: list[dict[str, Any]],
    *,
    authority_event: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach an authority event to copied branch provenance envelopes."""

    if not envelopes:
        raise ValueError("branch authority event application requires envelopes")
    if not isinstance(authority_event, dict):
        raise ValueError("authority_event must be a dict")
    conflict_group_key = authority_event.get("conflict_group_key")
    selected_key = authority_event.get("selected_assertion_key")
    superseded_keys = set(authority_event.get("superseded_assertion_keys") or [])
    contested_keys = set(authority_event.get("contested_assertion_keys") or [])
    classified_keys = {selected_key, *superseded_keys, *contested_keys}
    if not conflict_group_key or not selected_key:
        raise ValueError("authority_event must include conflict_group_key and selected_assertion_key")

    result: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in envelopes:
        envelope = _validated_branch_authority_envelope(item, role="apply")
        assertion_key = envelope["assertion_key"]
        if envelope.get("conflict_group_key") != conflict_group_key:
            raise ValueError("authority_event conflict_group_key must match every envelope")
        if assertion_key in seen_keys:
            raise ValueError("duplicate assertion_key in branch authority envelopes")
        seen_keys.add(assertion_key)
        if assertion_key == selected_key:
            conflict_state = "active"
            branch_state = "selected_authoritative"
        elif assertion_key in superseded_keys:
            conflict_state = "superseded"
            branch_state = "superseded"
        elif assertion_key in contested_keys:
            conflict_state = "contested"
            branch_state = "contested"
        else:
            raise ValueError("authority_event must classify every envelope")

        updated = dict(envelope)
        updated["conflict_resolution_state"] = conflict_state
        updated["branch_resolution_state"] = branch_state
        updated[BRANCH_AUTHORITY_EVENT_METADATA_KEY] = dict(authority_event)
        validation = validate_branch_provenance_metadata(updated)
        if not validation.ready:
            raise ValueError("; ".join(validation.invalid_reasons or validation.missing_fields))
        result.append(updated)

    extra_keys = classified_keys - seen_keys
    if extra_keys:
        raise ValueError("authority_event references assertion_key not present in envelopes")
    return result


def _validated_branch_authority_envelope(envelope: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValueError(f"{role} branch authority envelope must be a dict")
    if _contains_benchmark_private_marker(envelope):
        raise ValueError("branch authority envelope must not contain benchmark-private markers")
    validation = validate_branch_provenance_metadata(envelope)
    if not validation.ready:
        raise ValueError(f"{role} branch authority envelope is not ready")
    copied = dict(envelope)
    for field in ("assertion_key", "conflict_group_key", "predicate"):
        if not copied.get(field):
            raise ValueError(f"{role} branch authority envelope missing {field}")
    return copied


def build_synthetic_branch_provenance_fixture() -> list[dict[str, Any]]:
    rows = [
        ("bp-001", "Aster is affiliated with the religion of Solarianism", "source-event-1", 0),
        ("bp-002", "Aster is affiliated with the religion of Lunarianism", "source-event-2", 0),
        ("bp-003", "Solarianism was founded by Mira", "source-event-1", 1),
        ("bp-004", "Lunarianism was founded by Nox", "source-event-2", 1),
        ("bp-005", "Mira was born in the city of Oris", "source-event-1", 2),
        ("bp-006", "Nox was born in the city of Pel", "source-event-2", 2),
    ]
    fixture: list[dict[str, Any]] = []
    for concept_id, summary, source_event_id, source_sequence in rows:
        provenance = build_branch_provenance_metadata(
            summary,
            source_event_id=source_event_id,
            source_sequence=source_sequence,
            source_span=f"fixture:{source_sequence}",
            conflict_resolution_state="unresolved" if concept_id in {"bp-001", "bp-002"} else "active",
        )
        fixture.append(
            {
                "id": concept_id,
                "summary": summary,
                "metadata": {BRANCH_PROVENANCE_METADATA_KEY: provenance.to_metadata()},
            }
        )
    return fixture


def extract_branch_provenance_metadata(item: Any) -> dict[str, Any] | None:
    """Return a branch provenance envelope from a concept-like object or dict."""
    if item is None:
        return None
    if hasattr(item, "branch_provenance"):
        branch_provenance = getattr(item, "branch_provenance", None)
        if isinstance(branch_provenance, dict):
            return branch_provenance
    if hasattr(item, "metadata"):
        metadata = getattr(item, "metadata", None)
    elif isinstance(item, dict) and isinstance(item.get("metadata"), dict):
        metadata = item.get("metadata")
    elif isinstance(item, dict):
        metadata = item
    else:
        metadata = None
    if not isinstance(metadata, dict):
        return None
    envelope = metadata.get(BRANCH_PROVENANCE_METADATA_KEY, metadata)
    return envelope if isinstance(envelope, dict) else None


def validate_branch_provenance_metadata(item: Any) -> BranchProvenanceValidation:
    envelope = extract_branch_provenance_metadata(item) or {}
    missing = tuple(
        field
        for field in REQUIRED_BRANCH_PROVENANCE_FIELDS
        if field not in envelope or envelope.get(field) is None or envelope.get(field) == ""
    )
    invalid: list[str] = []
    source_sequence = envelope.get("source_sequence")
    if not isinstance(source_sequence, int) or source_sequence < 0:
        invalid.append("source_sequence must be a non-negative integer")
    state = envelope.get("conflict_resolution_state")
    if state not in _VALID_STATES:
        invalid.append("conflict_resolution_state must be a valid branch state")
    branch_state = envelope.get("branch_resolution_state")
    if branch_state not in _VALID_BRANCH_RESOLUTION_STATES:
        invalid.append("branch_resolution_state must be a valid branch resolution state")
    authority_event = envelope.get(BRANCH_AUTHORITY_EVENT_METADATA_KEY)
    if branch_state == "selected_authoritative":
        if state != "active":
            invalid.append("selected_authoritative requires active conflict_resolution_state")
        invalid.extend(
            _validate_branch_authority_event(
                authority_event,
                assertion_key=envelope.get("assertion_key"),
                conflict_group_key=envelope.get("conflict_group_key"),
                predicate=envelope.get("predicate"),
                branch_resolution_state=branch_state,
            )
        )
    elif branch_state in {"superseded", "contested"} and authority_event is not None:
        invalid.extend(
            _validate_branch_authority_event(
                authority_event,
                assertion_key=envelope.get("assertion_key"),
                conflict_group_key=envelope.get("conflict_group_key"),
                predicate=envelope.get("predicate"),
                branch_resolution_state=branch_state,
            )
        )
    ready = not missing and not invalid
    return BranchProvenanceValidation(
        ready=ready,
        missing_fields=missing,
        invalid_reasons=tuple(invalid),
        assertion_key=envelope.get("assertion_key"),
        conflict_group_key=envelope.get("conflict_group_key"),
        conflict_resolution_state=state,
        branch_resolution_state=branch_state,
    )


def summarize_branch_provenance_readiness(items: list[Any]) -> dict[str, Any]:
    envelopes = [extract_branch_provenance_metadata(item) for item in items]
    envelopes = [item for item in envelopes if item]
    validations = [validate_branch_provenance_metadata(item) for item in envelopes]
    unresolved = detect_unresolved_conflict_groups(envelopes)
    authorized_groups = {
        item.get("conflict_group_key")
        for item in envelopes
        if item.get("branch_resolution_state") == "selected_authoritative"
        and validate_branch_provenance_metadata(item).ready
    }
    return {
        "total": len(envelopes),
        "ready": sum(1 for item in validations if item.ready),
        "not_ready": sum(1 for item in validations if not item.ready),
        "unresolved_conflict_groups": len(unresolved),
        "uniquely_authorized_groups": len({group for group in authorized_groups if group}),
        "validations": [item.to_metadata() for item in validations],
    }


def detect_unresolved_conflict_groups(metadata_items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in metadata_items:
        envelope = item.get(BRANCH_PROVENANCE_METADATA_KEY, item)
        group_key = envelope.get("conflict_group_key")
        if group_key:
            groups.setdefault(group_key, []).append(envelope)

    unresolved: dict[str, list[dict[str, Any]]] = {}
    for group_key, items in groups.items():
        live_items = [
            item
            for item in items
            if item.get("branch_resolution_state") in {"selected_authoritative", "contested", "unresolved"}
        ]
        objects = {item.get("object") for item in live_items if item.get("object")}
        uniquely_authorized = [
            item
            for item in live_items
            if item.get("branch_resolution_state") == "selected_authoritative"
            and validate_branch_provenance_metadata(item).ready
        ]
        if len(objects) > 1 and len(uniquely_authorized) != 1:
            unresolved[group_key] = live_items
    return unresolved
