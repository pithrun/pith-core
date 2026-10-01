"""Task-owned managed-session admission and filesystem guards.

This module owns the durable binding/generation invariant.  It deliberately has
no route wiring: callers must first authenticate the request and select the
managed path, then keep the returned binding guard held until all accepted
post-response work has been registered.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from app.core.models import (
    ConversationTurnRequest,
    SessionBindingEnvelope,
    SessionEndRequest,
    SessionLearnRequest,
)
from app.storage.session_bindings import (
    WRITER_BARRIER_TRIGGER_NAMES,
    ManagedBindingFileGuard,
    ManagedEpisodeEffectFileGuard,
    ManagedEpisodePermit,
    ManagedFileGuardBusyError,
    ManagedQueuePermit,
    managed_queue_write_permit,
    managed_session_write_permit,
)
from app.storage.utils import validate_agent_id

_HEX_64 = frozenset("0123456789abcdef")
_SUPPORTED_SURFACES = {"codex_local_api"}
_ACTIVE_PHASE = "active"
_SUCCESSOR_PHASES = {"closing", "ended", "needs_attention"}


class ManagedBindingError(RuntimeError):
    """Typed managed-admission failure safe to translate at the API edge."""

    def __init__(self, code: str, *, status_code: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class BindingBusyError(ManagedBindingError):
    """Raised when another local process owns the binding guard."""

    def __init__(self) -> None:
        super().__init__("binding_busy", status_code=423)


class ManagedCloseDeferred(RuntimeError):
    """Raised when accepted episode work must settle before close effects run."""


@dataclass(frozen=True)
class ManagedEpisodeContext:
    """Immutable authority for one managed episode generation."""

    profile: str
    binding_hash: str
    generation: int
    session_id: str
    owner_surface_id: str
    owner_workspace_id: str
    protocol_version: int
    prior_session_id: str | None


@dataclass(frozen=True)
class ManagedShadowObservation:
    """Validated, read-only managed-envelope observation for shadow rollout."""

    profile: str
    surface_id: str
    protocol_version: int
    client_interturn_seconds: float | None


@dataclass(frozen=True)
class ManagedPostResponsePlan:
    """Immutable foreground registration plan for one managed turn."""

    profile: str
    binding_hash: str
    generation: int
    session_id: str
    raw_capture_json: str | None
    raw_learning_status_json: str | None
    last_previous_response: str | None
    autolearn_job_json: str | None
    autolearn_idempotency_key: str | None


_MANAGED_EPISODE_CONTEXT: ContextVar[ManagedEpisodeContext | None] = ContextVar(
    "pith_managed_episode_context",
    default=None,
)


@contextmanager
def managed_episode_context(context: ManagedEpisodeContext) -> Iterator[None]:
    """Expose immutable episode lineage only for the active managed request."""

    token = _MANAGED_EPISODE_CONTEXT.set(context)
    try:
        yield
    finally:
        _MANAGED_EPISODE_CONTEXT.reset(token)


def current_managed_episode_context() -> ManagedEpisodeContext | None:
    """Return request-local managed authority, never mutable manager state."""

    return _MANAGED_EPISODE_CONTEXT.get()


@dataclass(frozen=True)
class _BindingPolicy:
    mode: str
    idle_seconds: int
    writer_barrier_armed: bool
    triggers_complete: bool


class BindingGuard(ManagedBindingFileGuard):
    """Serializes admission and foreground registration for one binding."""

    def acquire(self):
        try:
            return super().acquire()
        except ManagedFileGuardBusyError as exc:
            raise BindingBusyError() from exc


class EpisodeEffectGuard(ManagedEpisodeEffectFileGuard):
    """Serializes slow closeout effects for one binding."""

    def acquire(self):
        try:
            return super().acquire()
        except ManagedFileGuardBusyError as exc:
            raise BindingBusyError() from exc


def _validate_hex64(value: str, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX_64 for char in value):
        raise ValueError(f"{field} must be 64 lowercase hex characters")


def _binding_hash(capability: str) -> str:
    _validate_hex64(capability, field="capability")
    return hashlib.sha256(bytes.fromhex(capability)).hexdigest()


def _record_shadow_observation(observation: ManagedShadowObservation) -> None:
    """Persist only low-cardinality aggregate shadow telemetry, best effort."""

    try:
        from app.ops.metrics import metrics

        labels = {
            "surface": observation.surface_id,
            "protocol": str(observation.protocol_version),
        }
        metrics.record("session_binding_shadow_request_total", 1.0, labels)
        if observation.client_interturn_seconds is not None:
            metrics.record(
                "session_binding_shadow_interturn_seconds",
                observation.client_interturn_seconds,
                labels,
            )
        metrics.flush()
    except Exception:
        # Shadow observability must never make the legacy request path unavailable.
        return


def observe_managed_shadow_turn(
    request: ConversationTurnRequest,
    *,
    profile: str,
) -> ManagedShadowObservation | None:
    """Validate a new managed envelope in shadow mode without lifecycle writes.

    Existing bindings are deliberately excluded: a paused or shadowed policy must
    continue serving their ownership-aware managed path.  Only a capability that
    has never enrolled may fall through to the unchanged legacy request route.
    """

    envelope = request.binding
    if envelope is None:
        raise ManagedBindingError("binding_required")
    if request.surface_id not in _SUPPORTED_SURFACES:
        raise ManagedBindingError("unsupported_binding_surface")
    if not (request.workspace_id or "").strip():
        raise ManagedBindingError("binding_workspace_required", status_code=422)
    if not profile.strip():
        raise ValueError("profile is required")

    binding_hash = _binding_hash(envelope.capability)
    with _read_db(operation="managed_turn_shadow_observation") as conn:
        policy = _load_policy(
            conn,
            profile=profile,
            protocol_version=envelope.protocol_version,
        )
        if policy.mode != "managed_shadow":
            return None
        if _load_binding(conn, profile=profile, binding_hash=binding_hash) is not None:
            return None

    observation = ManagedShadowObservation(
        profile=profile,
        surface_id=request.surface_id,
        protocol_version=envelope.protocol_version,
        client_interturn_seconds=envelope.client_interturn_seconds,
    )
    _record_shadow_observation(observation)
    return observation


def _db_immediate(*, operation: str):
    from app.storage import _db_immediate as storage_db_immediate

    return storage_db_immediate(operation=operation)


def _read_db(*, operation: str):
    from app.storage.connection import read_snapshot_db

    return read_snapshot_db(operation)


def _utc_iso(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("managed admission requires a timezone-aware timestamp")
    return now.astimezone(UTC).isoformat()


def _contains_capability_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() == "capability" or _contains_capability_key(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_capability_key(item) for item in value)
    return False


def _load_policy(conn: sqlite3.Connection, *, profile: str, protocol_version: int) -> _BindingPolicy:
    row = conn.execute(
        """SELECT mode, protocol_version, minimum_writer_protocol,
                  writer_barrier_armed, idle_seconds
           FROM session_binding_policy WHERE profile=?""",
        (profile,),
    ).fetchone()
    if row is None:
        raise ManagedBindingError("binding_disabled")
    mode = str(row["mode"] if hasattr(row, "keys") else row[0])
    configured_protocol = int(row["protocol_version"] if hasattr(row, "keys") else row[1])
    minimum_writer_protocol = int(row["minimum_writer_protocol"] if hasattr(row, "keys") else row[2])
    barrier_armed = int(row["writer_barrier_armed"] if hasattr(row, "keys") else row[3])
    idle_seconds = int(row["idle_seconds"] if hasattr(row, "keys") else row[4])
    if protocol_version != configured_protocol or protocol_version < minimum_writer_protocol:
        raise ManagedBindingError("unsupported_binding_protocol")
    present = {
        str(trigger_row[0])
        for trigger_row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_managed_%'"
        )
    }
    return _BindingPolicy(
        mode=mode,
        idle_seconds=idle_seconds,
        writer_barrier_armed=barrier_armed == 1,
        triggers_complete=set(WRITER_BARRIER_TRIGGER_NAMES).issubset(present),
    )


def _require_writer_barrier(policy: _BindingPolicy) -> None:
    if not policy.writer_barrier_armed:
        raise ManagedBindingError("writer_barrier_not_armed")
    if not policy.triggers_complete:
        raise ManagedBindingError("writer_barrier_incomplete")


def _require_new_enrollment_enabled(policy: _BindingPolicy) -> None:
    if policy.mode == "managed_enabled":
        return
    code = "binding_shadow_only" if policy.mode == "managed_shadow" else "binding_disabled"
    raise ManagedBindingError(code)


def _load_binding(conn: sqlite3.Connection, *, profile: str, binding_hash: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT binding_hash, profile, owner_surface_id, owner_workspace_id,
                  owner_native_conversation_hash, current_session_id, generation,
                  protocol_version, state, created_at, updated_at
           FROM session_bindings WHERE profile=? AND binding_hash=?""",
        (profile, binding_hash),
    ).fetchone()
    return dict(row) if row is not None else None


def _load_episode(conn: sqlite3.Connection, *, session_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT id, started_at, ended_at, status, binding_hash,
                  binding_generation, lifecycle_phase, last_activity_at
           FROM sessions WHERE id=?""",
        (session_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def _validate_owner(
    binding: dict[str, Any],
    *,
    surface_id: str,
    workspace_id: str,
    native_conversation_hash: str,
    protocol_version: int,
) -> None:
    expected = (
        surface_id,
        workspace_id,
        native_conversation_hash,
        protocol_version,
    )
    actual = (
        binding["owner_surface_id"],
        binding["owner_workspace_id"],
        binding["owner_native_conversation_hash"],
        int(binding["protocol_version"]),
    )
    if actual != expected:
        raise ManagedBindingError("binding_owner_mismatch")
    if binding["state"] != "active":
        raise ManagedBindingError("binding_tombstoned")


def _validate_envelope_authority(
    binding: dict[str, Any],
    *,
    native_conversation_hash: str,
    protocol_version: int,
) -> None:
    if binding["state"] != "active":
        raise ManagedBindingError("binding_tombstoned")
    if (
        binding["owner_native_conversation_hash"] != native_conversation_hash
        or _generation(binding["protocol_version"]) != protocol_version
    ):
        raise ManagedBindingError("binding_owner_mismatch")


def _validate_session_hint(
    conn: sqlite3.Connection,
    *,
    session_id: str | None,
    binding_hash: str,
) -> None:
    if not session_id:
        return
    hinted = _load_episode(conn, session_id=session_id)
    if hinted is None:
        raise ManagedBindingError("binding_mismatch")
    hinted_hash = hinted.get("binding_hash")
    if hinted_hash is None:
        return
    if hinted_hash != binding_hash:
        raise ManagedBindingError("binding_mismatch")


def _insert_episode(
    conn: sqlite3.Connection,
    *,
    request: ConversationTurnRequest,
    binding_hash: str,
    generation: int,
    session_id: str,
    timestamp: str,
) -> None:
    from app.core.surface_identity import normalize_surface_id, resolve_platform_hint

    surface_id = normalize_surface_id(request.surface_id)
    platform_hint = resolve_platform_hint(request.platform_hint, surface_id)
    permit = ManagedEpisodePermit(binding_hash=binding_hash, binding_generation=generation)
    with managed_session_write_permit(permit):
        conn.execute(
            """INSERT INTO sessions
               (id, started_at, status, learning_event_count, context_hint, data,
                agent_id, model_id, platform_hint, surface_id, origin_id,
                binding_hash, binding_generation, lifecycle_phase, last_activity_at)
               VALUES (?, ?, 'active', 0, 'managed', ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (
                session_id,
                timestamp,
                json.dumps({"session_id": session_id}, sort_keys=True),
                validate_agent_id(request.agent_id),
                request.model_id or "unknown",
                platform_hint,
                surface_id,
                request.origin_id,
                binding_hash,
                generation,
                timestamp,
            ),
        )


def _advance_activity(
    conn: sqlite3.Connection,
    *,
    binding_hash: str,
    generation: int,
    session_id: str,
    timestamp: str,
) -> None:
    permit = ManagedEpisodePermit(binding_hash=binding_hash, binding_generation=generation)
    with managed_session_write_permit(permit):
        cursor = conn.execute(
            """UPDATE sessions
               SET last_activity_at=CASE
                   WHEN last_activity_at IS NULL OR last_activity_at < ? THEN ?
                   ELSE last_activity_at
               END
               WHERE id=? AND binding_hash=? AND binding_generation=?
                 AND status='active' AND lifecycle_phase='active'""",
            (timestamp, timestamp, session_id, binding_hash, generation),
        )
    if cursor.rowcount != 1:
        raise ManagedBindingError("managed_episode_changed")


def _new_episode_id() -> str:
    return uuid.uuid4().hex


def _generation(value: object) -> int:
    if isinstance(value, bool):
        raise ManagedBindingError("binding_invariant_violation")
    try:
        generation = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManagedBindingError("binding_invariant_violation") from exc
    if generation < 0:
        raise ManagedBindingError("binding_invariant_violation")
    return generation


def resolve_managed_turn(
    request: ConversationTurnRequest,
    *,
    profile: str,
    now: datetime,
) -> tuple[ManagedEpisodeContext, BindingGuard]:
    """Resolve or create one exact managed episode and return its held guard."""

    envelope = request.binding
    if envelope is None:
        raise ManagedBindingError("binding_required")
    if request.surface_id not in _SUPPORTED_SURFACES:
        raise ManagedBindingError("unsupported_binding_surface")
    workspace_id = (request.workspace_id or "").strip()
    if not workspace_id:
        raise ManagedBindingError("binding_workspace_required", status_code=422)
    if not profile.strip():
        raise ValueError("profile is required")

    binding_hash = _binding_hash(envelope.capability)
    timestamp = _utc_iso(now)
    guard = BindingGuard(binding_hash, profile=profile).acquire()
    try:
        with _db_immediate(operation="managed_turn_admission") as conn:
            policy = _load_policy(
                conn,
                profile=profile,
                protocol_version=envelope.protocol_version,
            )
            binding = _load_binding(conn, profile=profile, binding_hash=binding_hash)
            requested_session_id = (request.session_id or "").strip() or None
            if binding is None:
                _require_new_enrollment_enabled(policy)
                _require_writer_barrier(policy)
                _validate_session_hint(
                    conn,
                    session_id=requested_session_id,
                    binding_hash=binding_hash,
                )
                generation = 0
                session_id = _new_episode_id()
                _insert_episode(
                    conn,
                    request=request,
                    binding_hash=binding_hash,
                    generation=generation,
                    session_id=session_id,
                    timestamp=timestamp,
                )
                conn.execute(
                    """INSERT INTO session_bindings
                       (binding_hash, profile, owner_surface_id, owner_workspace_id,
                        owner_native_conversation_hash, current_session_id, generation,
                        protocol_version, state, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                    (
                        binding_hash,
                        profile,
                        request.surface_id,
                        workspace_id,
                        envelope.native_conversation_hash,
                        session_id,
                        generation,
                        envelope.protocol_version,
                        timestamp,
                        timestamp,
                    ),
                )
                prior_session_id = None
            else:
                _require_writer_barrier(policy)
                _validate_owner(
                    binding,
                    surface_id=request.surface_id,
                    workspace_id=workspace_id,
                    native_conversation_hash=envelope.native_conversation_hash,
                    protocol_version=envelope.protocol_version,
                )
                _validate_session_hint(
                    conn,
                    session_id=requested_session_id,
                    binding_hash=binding_hash,
                )
                prior_session_id = None
                current_session_id = str(binding["current_session_id"] or "")
                if not current_session_id:
                    raise ManagedBindingError("binding_invariant_violation")
                current = _load_episode(conn, session_id=current_session_id)
                if current is None or current.get("binding_hash") != binding_hash:
                    raise ManagedBindingError("binding_invariant_violation")
                current_generation = _generation(binding["generation"])
                if _generation(current.get("binding_generation")) != current_generation:
                    raise ManagedBindingError("binding_invariant_violation")
                phase = current.get("lifecycle_phase")
                if phase == _ACTIVE_PHASE:
                    if current.get("status") != "active":
                        raise ManagedBindingError("binding_invariant_violation")
                    session_id = current_session_id
                    generation = current_generation
                    _advance_activity(
                        conn,
                        binding_hash=binding_hash,
                        generation=generation,
                        session_id=session_id,
                        timestamp=timestamp,
                    )
                    cursor = conn.execute(
                        """UPDATE session_bindings SET updated_at=?
                           WHERE profile=? AND binding_hash=? AND state='active'
                             AND current_session_id=? AND generation=?""",
                        (
                            timestamp,
                            profile,
                            binding_hash,
                            current_session_id,
                            current_generation,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ManagedBindingError("binding_generation_conflict")
                elif phase in _SUCCESSOR_PHASES:
                    if phase in {"ended", "needs_attention"} and current.get("status") != "ended":
                        raise ManagedBindingError("binding_invariant_violation")
                    if phase == "closing" and current.get("status") != "active":
                        raise ManagedBindingError("binding_invariant_violation")
                    prior_session_id = current_session_id
                    generation = current_generation + 1
                    session_id = _new_episode_id()
                    _insert_episode(
                        conn,
                        request=request,
                        binding_hash=binding_hash,
                        generation=generation,
                        session_id=session_id,
                        timestamp=timestamp,
                    )
                    cursor = conn.execute(
                        """UPDATE session_bindings
                           SET current_session_id=?, generation=?, updated_at=?
                           WHERE profile=? AND binding_hash=? AND state='active'
                             AND current_session_id=? AND generation=?""",
                        (
                            session_id,
                            generation,
                            timestamp,
                            profile,
                            binding_hash,
                            current_session_id,
                            current_generation,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ManagedBindingError("binding_generation_conflict")
                else:
                    raise ManagedBindingError("binding_invariant_violation")

        context = ManagedEpisodeContext(
            profile=profile,
            binding_hash=binding_hash,
            generation=generation,
            session_id=session_id,
            owner_surface_id=request.surface_id,
            owner_workspace_id=workspace_id,
            protocol_version=envelope.protocol_version,
            prior_session_id=prior_session_id,
        )
        return context, guard
    except Exception:
        guard.release()
        raise


def build_managed_post_response_plan(
    response: Any,
    *,
    context: ManagedEpisodeContext,
) -> ManagedPostResponsePlan:
    """Freeze the managed episode work attached by ``conversation_turn``.

    Capability-bearing request data is deliberately excluded.  The frozen plan
    contains only the already-authorized binding hash and exact episode
    generation needed by the durable queue and fenced runner.
    """

    raw_capture = getattr(response, "_pending_raw_capture", None)
    raw_learning_status = getattr(response, "_pending_raw_learning_status", None)
    last_previous_response = getattr(response, "_pending_last_previous_response", None)
    for payload in (raw_capture, raw_learning_status, last_previous_response):
        if payload is not None and payload.get("session_id") != context.session_id:
            raise ManagedBindingError("binding_context_mismatch")

    autolearn_args = getattr(response, "_pending_autolearn", None)
    autolearn_job_json = None
    autolearn_idempotency_key = None
    if autolearn_args is not None:
        if not isinstance(autolearn_args, tuple) or len(autolearn_args) != 8:
            raise ManagedBindingError("managed_post_response_registration_unavailable", status_code=503)
        (
            learn_request,
            extracted,
            request_message,
            prev_msg,
            prev_response,
            bound_session,
            raw_capture_ref,
            active_binding_snapshot,
        ) = autolearn_args
        learn_session_id = getattr(learn_request, "session_id", None)
        bound_session_id = getattr(bound_session, "session_id", None)
        if learn_session_id != context.session_id or bound_session_id != context.session_id:
            raise ManagedBindingError("binding_context_mismatch")
        if raw_capture_ref is not None and raw_capture_ref.get("session_id") != context.session_id:
            raise ManagedBindingError("binding_context_mismatch")
        external_request_id = getattr(learn_request, "request_id", None)
        if not external_request_id:
            raise ManagedBindingError("managed_post_response_registration_unavailable", status_code=503)

        learn_payload = (
            learn_request.model_dump(mode="json") if hasattr(learn_request, "model_dump") else dict(learn_request)
        )
        learn_payload["binding"] = None
        learn_payload["binding_generation"] = None
        job_payload = {
            "learn_request": learn_payload,
            "extracted": extracted,
            "request_message": request_message,
            "prev_msg": prev_msg,
            "prev_response": prev_response,
            "bound_session": (
                bound_session.model_dump(mode="json") if hasattr(bound_session, "model_dump") else dict(bound_session)
            ),
            "raw_capture_ref": raw_capture_ref,
            "active_binding_snapshot": active_binding_snapshot,
            "managed_context": {
                "profile": context.profile,
                "binding_hash": context.binding_hash,
                "binding_generation": context.generation,
                "session_id": context.session_id,
                "protocol_version": context.protocol_version,
            },
        }
        if _contains_capability_key(job_payload):
            raise ManagedBindingError("binding_context_mismatch")
        autolearn_job_json = json.dumps(job_payload, sort_keys=True, separators=(",", ":"))
        request_hash = hashlib.sha256(autolearn_job_json.encode("utf-8")).hexdigest()
        identity = "\0".join(
            (
                "v1",
                context.profile,
                context.binding_hash,
                "conversation_turn",
                str(external_request_id),
                request_hash,
            )
        )
        autolearn_idempotency_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return ManagedPostResponsePlan(
        profile=context.profile,
        binding_hash=context.binding_hash,
        generation=context.generation,
        session_id=context.session_id,
        raw_capture_json=(
            json.dumps(raw_capture, sort_keys=True, separators=(",", ":")) if raw_capture is not None else None
        ),
        raw_learning_status_json=(
            json.dumps(raw_learning_status, sort_keys=True, separators=(",", ":"))
            if raw_learning_status is not None
            else None
        ),
        last_previous_response=(
            str(last_previous_response["last_previous_response"]) if last_previous_response is not None else None
        ),
        autolearn_job_json=autolearn_job_json,
        autolearn_idempotency_key=autolearn_idempotency_key,
    )


def register_managed_post_response_plan(
    plan: ManagedPostResponsePlan,
    *,
    response: Any,
) -> bool:
    """Persist accepted managed turn capture work before guard release."""

    raw_capture = json.loads(plan.raw_capture_json) if plan.raw_capture_json else None
    raw_learning_status = json.loads(plan.raw_learning_status_json) if plan.raw_learning_status_json else None
    autolearn_payload = json.loads(plan.autolearn_job_json) if plan.autolearn_job_json else None
    try:
        if (
            raw_capture is not None
            or raw_learning_status is not None
            or plan.last_previous_response is not None
            or autolearn_payload is not None
        ):
            from app.storage.turn_ingestion import capture_raw_turn, mark_learning_status

            with _db_immediate(operation="managed_turn_post_response_registration") as conn:
                if raw_capture is not None:
                    capture_raw_turn(conn, **raw_capture)
                if raw_learning_status is not None:
                    mark_learning_status(conn, **raw_learning_status)
                if plan.last_previous_response is not None:
                    permit = ManagedEpisodePermit(
                        binding_hash=plan.binding_hash,
                        binding_generation=plan.generation,
                    )
                    with managed_session_write_permit(permit):
                        cursor = conn.execute(
                            """UPDATE sessions
                               SET last_previous_response=?
                               WHERE id=? AND binding_hash=? AND binding_generation=?
                                 AND status='active' AND lifecycle_phase='active'""",
                            (
                                plan.last_previous_response,
                                plan.session_id,
                                plan.binding_hash,
                                plan.generation,
                            ),
                        )
                    if cursor.rowcount != 1:
                        raise ManagedBindingError("managed_episode_changed")
                if autolearn_payload is not None:
                    if not plan.autolearn_idempotency_key:
                        raise ManagedBindingError("binding_invariant_violation")
                    from app.storage.lifecycle_jobs import enqueue_lifecycle_job_conn

                    job = enqueue_lifecycle_job_conn(
                        conn,
                        profile=plan.profile,
                        source="conversation_turn",
                        idempotency_key=plan.autolearn_idempotency_key,
                        stage="learn",
                        payload=autolearn_payload,
                        session_id=plan.session_id,
                        binding_hash=plan.binding_hash,
                        binding_generation=plan.generation,
                    )
                    if job.get("status") not in {"queued", "retry", "running", "committed"}:
                        raise ManagedBindingError(
                            "managed_post_response_registration_unavailable",
                            status_code=503,
                        )
    except ManagedBindingError:
        raise
    except Exception as exc:
        raise ManagedBindingError(
            "managed_post_response_registration_unavailable",
            status_code=503,
        ) from exc

    # The generic dispatcher may still execute global-only work, but managed
    # episode work registered above must never be registered a second time.
    object.__setattr__(response, "_pending_raw_capture", None)
    object.__setattr__(response, "_pending_raw_learning_status", None)
    object.__setattr__(response, "_pending_last_previous_response", None)
    object.__setattr__(response, "_pending_autolearn", None)
    return autolearn_payload is not None


def run_managed_conversation_turn_effect(
    job: dict[str, Any],
    *,
    run_effect,
) -> Any:
    """Run one claimed managed CT job under exact episode authority."""

    profile = str(job.get("profile") or "")
    job_id = str(job.get("job_id") or "")
    binding_hash = str(job.get("binding_hash") or "")
    session_id = str(job.get("session_id") or "")
    generation = _generation(job.get("binding_generation"))
    claim_token = str(job.get("claim_token") or "")
    _validate_hex64(binding_hash, field="binding_hash")
    _validate_hex64(claim_token, field="claim_token")
    if (
        not profile
        or not job_id
        or not session_id
        or job.get("source") != "conversation_turn"
        or job.get("stage") != "learn"
    ):
        raise ManagedBindingError("binding_invariant_violation")

    payload = job.get("payload") or {}
    managed_context = payload.get("managed_context") if isinstance(payload, dict) else None
    if not isinstance(managed_context, dict) or (
        managed_context.get("profile"),
        managed_context.get("binding_hash"),
        managed_context.get("binding_generation"),
        managed_context.get("session_id"),
    ) != (profile, binding_hash, generation, session_id):
        raise ManagedBindingError("binding_context_mismatch")

    effect_guard = EpisodeEffectGuard(binding_hash, profile=profile).acquire()
    try:
        binding_guard = BindingGuard(binding_hash, profile=profile).acquire()
        try:
            with _db_immediate(operation="managed_turn_effect_authorization") as conn:
                binding = _load_binding(conn, profile=profile, binding_hash=binding_hash)
                episode = _load_episode(conn, session_id=session_id)
                claim = conn.execute(
                    """SELECT 1 FROM lifecycle_jobs
                       WHERE profile=? AND job_id=? AND source='conversation_turn'
                         AND stage='learn' AND status='running' AND claim_token=?
                         AND session_id=? AND binding_hash=? AND binding_generation=?""",
                    (
                        profile,
                        job_id,
                        claim_token,
                        session_id,
                        binding_hash,
                        generation,
                    ),
                ).fetchone()
                if (
                    binding is None
                    or binding.get("state") != "active"
                    or binding.get("current_session_id") != session_id
                    or _generation(binding.get("generation")) != generation
                    or episode is None
                    or episode.get("binding_hash") != binding_hash
                    or _generation(episode.get("binding_generation")) != generation
                    or episode.get("status") != "active"
                    or episode.get("lifecycle_phase") not in {"active", "closing"}
                    or claim is None
                ):
                    raise ManagedBindingError("managed_episode_changed")
        finally:
            binding_guard.release()

        permit = ManagedEpisodePermit(binding_hash=binding_hash, binding_generation=generation)
        with managed_session_write_permit(permit):
            return run_effect()
    finally:
        effect_guard.release()


def _authorize_exact_episode_fields_conn(
    envelope: SessionBindingEnvelope,
    *,
    session_id: str | None,
    binding_generation: int | None,
    profile: str,
    conn: sqlite3.Connection,
) -> tuple[ManagedEpisodeContext, dict[str, Any]]:
    if not isinstance(session_id, str):
        raise ManagedBindingError("binding_episode_required")
    session_id = session_id.strip() or None
    if session_id is not None and len(session_id) > 200:
        raise ManagedBindingError("binding_episode_required")
    if session_id is None or binding_generation is None:
        raise ManagedBindingError("binding_episode_required")
    generation = _generation(binding_generation)
    binding_hash = _binding_hash(envelope.capability)
    policy = _load_policy(
        conn,
        profile=profile,
        protocol_version=envelope.protocol_version,
    )
    _require_writer_barrier(policy)
    binding = _load_binding(conn, profile=profile, binding_hash=binding_hash)
    if binding is None:
        raise ManagedBindingError("binding_mismatch")
    _validate_envelope_authority(
        binding,
        native_conversation_hash=envelope.native_conversation_hash,
        protocol_version=envelope.protocol_version,
    )
    episode = _load_episode(conn, session_id=session_id)
    if (
        episode is None
        or episode.get("binding_hash") != binding_hash
        or _generation(episode.get("binding_generation")) != generation
    ):
        raise ManagedBindingError("binding_mismatch")
    context = ManagedEpisodeContext(
        profile=profile,
        binding_hash=binding_hash,
        generation=generation,
        session_id=session_id,
        owner_surface_id=str(binding["owner_surface_id"]),
        owner_workspace_id=str(binding["owner_workspace_id"]),
        protocol_version=envelope.protocol_version,
        prior_session_id=None,
    )
    return context, episode


def _authorize_exact_episode_conn(
    request: SessionLearnRequest | SessionEndRequest,
    *,
    profile: str,
    conn: sqlite3.Connection,
) -> tuple[ManagedEpisodeContext, dict[str, Any]]:
    envelope = request.binding
    if envelope is None:
        raise ManagedBindingError("binding_required")
    return _authorize_exact_episode_fields_conn(
        envelope,
        session_id=request.session_id,
        binding_generation=request.binding_generation,
        profile=profile,
        conn=conn,
    )


def authorize_managed_write_request_status(
    envelope: SessionBindingEnvelope,
    *,
    session_id: str | None,
    binding_generation: int | None,
    profile: str,
) -> tuple[ManagedEpisodeContext, str, BindingGuard]:
    """Authorize a status lookup for one exact managed episode."""

    binding_hash = _binding_hash(envelope.capability)
    guard = BindingGuard(binding_hash, profile=profile).acquire()
    try:
        with _read_db(operation="managed_write_request_status_authorization") as conn:
            context, episode = _authorize_exact_episode_fields_conn(
                envelope,
                session_id=session_id,
                binding_generation=binding_generation,
                profile=profile,
                conn=conn,
            )
        return context, str(episode.get("lifecycle_phase") or "active"), guard
    except Exception:
        guard.release()
        raise


def _same_episode_authority(left: ManagedEpisodeContext, right: ManagedEpisodeContext) -> bool:
    return (
        left.profile,
        left.binding_hash,
        left.generation,
        left.session_id,
        left.owner_surface_id,
        left.owner_workspace_id,
        left.protocol_version,
    ) == (
        right.profile,
        right.binding_hash,
        right.generation,
        right.session_id,
        right.owner_surface_id,
        right.owner_workspace_id,
        right.protocol_version,
    )


def register_managed_learning(
    request: SessionLearnRequest,
    *,
    context: ManagedEpisodeContext,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Authorize one exact active episode for managed learning registration.

    The caller owns the surrounding binding guard and transaction.  This
    function does not execute learning; it creates an immutable, redacted
    registration result for the durable queue/replay layer.
    """

    authorized, episode = _authorize_exact_episode_conn(
        request,
        profile=context.profile,
        conn=conn,
    )
    if not _same_episode_authority(authorized, context):
        raise ManagedBindingError("binding_context_mismatch")
    if episode.get("status") != "active" or episode.get("lifecycle_phase") != "active":
        raise ManagedBindingError("managed_episode_not_active")
    return {
        "status": "registered",
        "profile": context.profile,
        "session_id": context.session_id,
        "binding_hash": context.binding_hash,
        "binding_generation": context.generation,
        "lifecycle_phase": "active",
    }


def resolve_managed_learning(
    request: SessionLearnRequest,
    *,
    profile: str,
) -> tuple[ManagedEpisodeContext, BindingGuard]:
    """Authorize one managed learn request while retaining its binding guard."""

    envelope = request.binding
    if envelope is None:
        raise ManagedBindingError("binding_required")
    binding_hash = _binding_hash(envelope.capability)
    guard = BindingGuard(binding_hash, profile=profile).acquire()
    try:
        with _db_immediate(operation="managed_learning_registration") as conn:
            context, _episode = _authorize_exact_episode_conn(
                request,
                profile=profile,
                conn=conn,
            )
            register_managed_learning(
                request,
                context=context,
                conn=conn,
            )
        return context, guard
    except Exception:
        guard.release()
        raise


def run_managed_session_learn_effect(
    job: dict[str, Any],
    *,
    run_effect,
) -> Any:
    """Run one claimed managed explicit-learn job under episode authority."""

    profile = str(job.get("profile") or "")
    job_id = str(job.get("job_id") or "")
    binding_hash = str(job.get("binding_hash") or "")
    session_id = str(job.get("session_id") or "")
    generation = _generation(job.get("binding_generation"))
    claim_token = str(job.get("claim_token") or "")
    _validate_hex64(binding_hash, field="binding_hash")
    _validate_hex64(claim_token, field="claim_token")
    if (
        not profile
        or not job_id
        or not session_id
        or job.get("source") != "session_learn"
        or job.get("stage") != "learn"
    ):
        raise ManagedBindingError("binding_invariant_violation")

    payload = job.get("payload") or {}
    managed_context = payload.get("managed_context") if isinstance(payload, dict) else None
    if not isinstance(managed_context, dict) or (
        managed_context.get("profile"),
        managed_context.get("binding_hash"),
        managed_context.get("binding_generation"),
        managed_context.get("session_id"),
    ) != (profile, binding_hash, generation, session_id):
        raise ManagedBindingError("binding_context_mismatch")

    effect_guard = EpisodeEffectGuard(binding_hash, profile=profile).acquire()
    try:
        binding_guard = BindingGuard(binding_hash, profile=profile).acquire()
        try:
            with _db_immediate(operation="managed_learning_effect_authorization") as conn:
                binding = _load_binding(conn, profile=profile, binding_hash=binding_hash)
                episode = _load_episode(conn, session_id=session_id)
                claim = conn.execute(
                    """SELECT 1 FROM lifecycle_jobs
                       WHERE profile=? AND job_id=? AND source='session_learn'
                         AND stage='learn' AND status='running' AND claim_token=?
                         AND session_id=? AND binding_hash=? AND binding_generation=?""",
                    (
                        profile,
                        job_id,
                        claim_token,
                        session_id,
                        binding_hash,
                        generation,
                    ),
                ).fetchone()
                if (
                    binding is None
                    or binding.get("state") != "active"
                    or episode is None
                    or episode.get("binding_hash") != binding_hash
                    or _generation(episode.get("binding_generation")) != generation
                    or episode.get("status") != "active"
                    or episode.get("lifecycle_phase") not in {"active", "closing"}
                    or claim is None
                ):
                    raise ManagedBindingError("managed_episode_changed")
        finally:
            binding_guard.release()

        permit = ManagedEpisodePermit(binding_hash=binding_hash, binding_generation=generation)
        with managed_session_write_permit(permit):
            return run_effect()
    finally:
        effect_guard.release()


def request_managed_close(
    request: SessionEndRequest,
    *,
    profile: str,
    reason: Literal["explicit", "idle"],
    now: datetime,
    begin_replay: Callable[[sqlite3.Connection, ManagedEpisodeContext], Any] | None = None,
) -> dict[str, Any]:
    """Atomically register replay, transition an episode, and enqueue close intent."""

    if reason not in {"explicit", "idle"}:
        raise ValueError("managed close reason must be explicit or idle")
    envelope = request.binding
    if envelope is None:
        raise ManagedBindingError("binding_required")
    binding_hash = _binding_hash(envelope.capability)
    timestamp = _utc_iso(now)
    guard = BindingGuard(binding_hash, profile=profile).acquire()
    try:
        with _db_immediate(operation="managed_close_request") as conn:
            context, episode = _authorize_exact_episode_conn(
                request,
                profile=profile,
                conn=conn,
            )
            replay_state = begin_replay(conn, context) if begin_replay is not None else None
            if replay_state is not None and replay_state.replay is not None:
                return {
                    "status": "replayed",
                    "session_id": context.session_id,
                    "binding_generation": context.generation,
                    "lifecycle_phase": episode.get("lifecycle_phase"),
                    "replay": replay_state.replay,
                    "_replay_state": replay_state,
                }
            phase = episode.get("lifecycle_phase")
            status = episode.get("status")
            if phase == "active":
                if status != "active":
                    raise ManagedBindingError("binding_invariant_violation")
                permit = ManagedEpisodePermit(
                    binding_hash=context.binding_hash,
                    binding_generation=context.generation,
                )
                with managed_session_write_permit(permit):
                    cursor = conn.execute(
                        """UPDATE sessions
                           SET lifecycle_phase='closing'
                           WHERE id=? AND binding_hash=? AND binding_generation=?
                             AND status='active' AND lifecycle_phase='active'""",
                        (
                            context.session_id,
                            context.binding_hash,
                            context.generation,
                        ),
                    )
                if cursor.rowcount != 1:
                    raise ManagedBindingError("managed_episode_changed")
                phase = "closing"
            elif phase == "closing":
                if status != "active":
                    raise ManagedBindingError("binding_invariant_violation")
            elif phase in {"ended", "needs_attention"}:
                if status != "ended":
                    raise ManagedBindingError("binding_invariant_violation")
                row = conn.execute(
                    """SELECT job_id, status FROM lifecycle_jobs
                       WHERE profile=? AND source='session_end' AND idempotency_key=?""",
                    (
                        profile,
                        f"close:{context.session_id}:{context.generation}:v1",
                    ),
                ).fetchone()
                result = {
                    "status": "already_terminal",
                    "session_id": context.session_id,
                    "binding_generation": context.generation,
                    "lifecycle_phase": phase,
                    "job_id": row[0] if row is not None else None,
                    "job_status": row[1] if row is not None else None,
                }
                if replay_state is not None:
                    result["_replay_state"] = replay_state
                return result
            else:
                raise ManagedBindingError("binding_invariant_violation")

            from app.storage.lifecycle_jobs import enqueue_lifecycle_job_conn

            end_request_payload = request.model_dump(mode="json")
            end_request_payload.pop("binding", None)
            job = enqueue_lifecycle_job_conn(
                conn,
                profile=profile,
                source="session_end",
                idempotency_key=f"close:{context.session_id}:{context.generation}:v1",
                stage="close",
                payload={
                    "reason": reason,
                    "session_id": context.session_id,
                    "binding_generation": context.generation,
                    "end_request": end_request_payload,
                    "managed_context": {
                        "profile": context.profile,
                        "binding_hash": context.binding_hash,
                        "binding_generation": context.generation,
                        "session_id": context.session_id,
                    },
                },
                session_id=context.session_id,
                binding_hash=context.binding_hash,
                binding_generation=context.generation,
                now=timestamp,
            )
            result = {
                "status": "closing",
                "session_id": context.session_id,
                "binding_generation": context.generation,
                "lifecycle_phase": phase,
                "job_id": job.get("job_id"),
                "job_status": job.get("status"),
            }
            if replay_state is not None:
                result["_replay_state"] = replay_state
            return result
    finally:
        guard.release()


def sweep_due_managed_episodes(
    *,
    profile: str,
    now: datetime | None = None,
    limit: int = 50,
    max_wall_seconds: float = 10.0,
) -> dict[str, Any]:
    """Register bounded idle-close intents for the current managed policy.

    The candidate query is advisory. Every enabled-mode candidate is rechecked
    under its binding guard and the closing transition plus durable close intent
    commit in one transaction. Shadow and disabled modes are strictly read-only.
    """

    if not profile.strip():
        raise ValueError("profile is required")
    candidate_limit = int(limit)
    if not 1 <= candidate_limit <= 50:
        raise ValueError("managed sweep limit must be between 1 and 50")
    wall_seconds = float(max_wall_seconds)
    if wall_seconds < 0:
        raise ValueError("managed sweep wall budget must be non-negative")
    current = now or datetime.now(UTC)
    timestamp = _utc_iso(current)
    started = time.monotonic()

    with _read_db(operation="managed_idle_sweep_candidates") as conn:
        try:
            policy = _load_policy(conn, profile=profile, protocol_version=1)
        except ManagedBindingError as exc:
            if exc.code != "binding_disabled":
                raise
            return {
                "profile": profile,
                "mode": "legacy_disabled",
                "inspected": 0,
                "due": 0,
                "closed": 0,
                "busy": 0,
                "skipped": 0,
                "shadow_only": 0,
                "policy_present": 0,
            }
        if policy.mode == "legacy_disabled":
            return {
                "profile": profile,
                "mode": "legacy_disabled",
                "inspected": 0,
                "due": 0,
                "closed": 0,
                "busy": 0,
                "skipped": 0,
                "shadow_only": 0,
                "policy_present": 1,
            }
        cutoff = _utc_iso(current - timedelta(seconds=policy.idle_seconds))
        candidates = [
            dict(row)
            for row in conn.execute(
                """SELECT s.id AS session_id, s.binding_hash,
                          s.binding_generation, s.last_activity_at
                   FROM session_bindings AS b
                   JOIN sessions AS s
                     ON s.id=b.current_session_id
                    AND s.binding_hash=b.binding_hash
                    AND s.binding_generation=b.generation
                   WHERE b.profile=? AND b.state='active'
                     AND s.status='active' AND s.lifecycle_phase='active'
                     AND s.last_activity_at IS NOT NULL
                     AND s.last_activity_at<=?
                   ORDER BY s.last_activity_at, s.id
                   LIMIT ?""",
                (profile, cutoff, candidate_limit),
            )
        ]

    result: dict[str, Any] = {
        "profile": profile,
        "mode": policy.mode,
        "inspected": len(candidates),
        "due": len(candidates),
        "closed": 0,
        "busy": 0,
        "skipped": 0,
    }
    if policy.mode == "managed_shadow":
        result["shadow_only"] = 1
        return result
    if policy.mode != "managed_enabled":
        raise ManagedBindingError("binding_disabled")
    _require_writer_barrier(policy)

    from app.storage.lifecycle_jobs import enqueue_lifecycle_job_conn

    for candidate in candidates:
        if time.monotonic() - started >= wall_seconds:
            result["wall_budget_exhausted"] = 1
            break
        binding_hash = str(candidate["binding_hash"])
        session_id = str(candidate["session_id"])
        generation = _generation(candidate["binding_generation"])
        try:
            guard = BindingGuard(binding_hash, profile=profile).acquire()
        except BindingBusyError:
            result["busy"] = int(result["busy"]) + 1
            continue
        try:
            with _db_immediate(operation="managed_idle_sweep_close") as conn:
                current_policy = _load_policy(conn, profile=profile, protocol_version=1)
                if current_policy.mode != "managed_enabled":
                    result["policy_changed"] = 1
                    break
                _require_writer_barrier(current_policy)
                current_cutoff = _utc_iso(current - timedelta(seconds=current_policy.idle_seconds))
                binding = _load_binding(conn, profile=profile, binding_hash=binding_hash)
                episode = _load_episode(conn, session_id=session_id)
                if (
                    binding is None
                    or binding.get("state") != "active"
                    or binding.get("current_session_id") != session_id
                    or _generation(binding.get("generation")) != generation
                    or episode is None
                    or episode.get("binding_hash") != binding_hash
                    or _generation(episode.get("binding_generation")) != generation
                    or episode.get("status") != "active"
                    or episode.get("lifecycle_phase") != "active"
                    or episode.get("last_activity_at") is None
                    or str(episode["last_activity_at"]) > current_cutoff
                ):
                    result["skipped"] = int(result["skipped"]) + 1
                    continue

                permit = ManagedEpisodePermit(
                    binding_hash=binding_hash,
                    binding_generation=generation,
                )
                with managed_session_write_permit(permit):
                    cursor = conn.execute(
                        """UPDATE sessions
                           SET lifecycle_phase='closing'
                           WHERE id=? AND binding_hash=? AND binding_generation=?
                             AND status='active' AND lifecycle_phase='active'
                             AND last_activity_at<=?""",
                        (session_id, binding_hash, generation, current_cutoff),
                    )
                if cursor.rowcount != 1:
                    result["skipped"] = int(result["skipped"]) + 1
                    continue
                enqueue_lifecycle_job_conn(
                    conn,
                    profile=profile,
                    source="session_end",
                    idempotency_key=f"close:{session_id}:{generation}:v1",
                    stage="close",
                    payload={
                        "reason": "idle",
                        "session_id": session_id,
                        "binding_generation": generation,
                        "end_request": {
                            "session_id": session_id,
                            "binding_generation": generation,
                        },
                        "managed_context": {
                            "profile": profile,
                            "binding_hash": binding_hash,
                            "binding_generation": generation,
                            "session_id": session_id,
                        },
                    },
                    session_id=session_id,
                    binding_hash=binding_hash,
                    binding_generation=generation,
                    now=timestamp,
                )
                result["closed"] = int(result["closed"]) + 1
        finally:
            guard.release()
    return result


def _managed_close_work_state(
    conn: sqlite3.Connection,
    *,
    profile: str,
    binding_hash: str,
    session_id: str,
    job_id: str,
) -> dict[str, int]:
    pending_jobs = conn.execute(
        """SELECT COUNT(*) FROM lifecycle_jobs
           WHERE profile=? AND binding_hash=? AND session_id=? AND job_id<>?
             AND source<>'session_end' AND status IN ('queued','retry','running')""",
        (profile, binding_hash, session_id, job_id),
    ).fetchone()[0]
    failed_jobs = conn.execute(
        """SELECT COUNT(*) FROM lifecycle_jobs
           WHERE profile=? AND binding_hash=? AND session_id=?
             AND source<>'session_end' AND status='failed'""",
        (profile, binding_hash, session_id),
    ).fetchone()[0]
    pending_replays = conn.execute(
        """SELECT COUNT(*) FROM write_request_replays
           WHERE profile=? AND binding_hash=? AND session_id=?
             AND endpoint<>'session_end' AND status='processing'""",
        (profile, binding_hash, session_id),
    ).fetchone()[0]
    failed_replays = conn.execute(
        """SELECT COUNT(*) FROM write_request_replays
           WHERE profile=? AND binding_hash=? AND session_id=?
             AND endpoint<>'session_end' AND status='failed'""",
        (profile, binding_hash, session_id),
    ).fetchone()[0]
    return {
        "pending": int(pending_jobs or 0) + int(pending_replays or 0),
        "failed": int(failed_jobs or 0) + int(failed_replays or 0),
    }


def _finalize_managed_episode_conn(
    conn: sqlite3.Connection,
    *,
    binding_hash: str,
    session_id: str,
    generation: int,
    lifecycle_phase: Literal["ended", "needs_attention"],
    now: str,
) -> None:
    permit = ManagedEpisodePermit(
        binding_hash=binding_hash,
        binding_generation=generation,
    )
    with managed_session_write_permit(permit):
        cursor = conn.execute(
            """UPDATE sessions
               SET status='ended', lifecycle_phase=?, ended_at=COALESCE(ended_at, ?)
               WHERE id=? AND binding_hash=? AND binding_generation=?
                 AND lifecycle_phase IN ('closing','ended','needs_attention')""",
            (lifecycle_phase, now, session_id, binding_hash, generation),
        )
    if cursor.rowcount != 1:
        raise ManagedBindingError("managed_episode_changed")


def _record_close_attention_conn(
    conn: sqlite3.Connection,
    *,
    job: dict[str, Any],
    payload_hash: str,
    error: str,
    now: str,
) -> None:
    conn.execute(
        """INSERT INTO session_close_step_receipts
           (session_id, step_name, step_version, binding_hash, binding_generation,
            state, claim_token, payload_hash, result_hash, created_at, updated_at, last_error)
           VALUES (?, 'managed_close_pipeline', 1, ?, ?, 'needs_attention', ?, ?, NULL, ?, ?, ?)
           ON CONFLICT(session_id, step_name, step_version) DO UPDATE SET
             state='needs_attention', claim_token=excluded.claim_token,
             updated_at=excluded.updated_at, last_error=excluded.last_error""",
        (
            job["session_id"],
            job["binding_hash"],
            int(job["binding_generation"]),
            job["claim_token"],
            payload_hash,
            now,
            now,
            error[:1000],
        ),
    )
    _finalize_managed_episode_conn(
        conn,
        binding_hash=str(job["binding_hash"]),
        session_id=str(job["session_id"]),
        generation=int(job["binding_generation"]),
        lifecycle_phase="needs_attention",
        now=now,
    )


def _terminal_close_response(
    *,
    job: dict[str, Any],
    lifecycle_phase: Literal["ended", "needs_attention"],
    error_class: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "status": lifecycle_phase,
        "session_id": str(job["session_id"]),
        "binding_generation": int(job["binding_generation"]),
        "lifecycle_phase": lifecycle_phase,
        "final_learning_state": "committed" if lifecycle_phase == "ended" else "needs_attention",
        "persistence_state": "committed" if lifecycle_phase == "ended" else "failed",
        "_protocol": {
            "binding_mode": "managed",
            "binding_protocol_version": 1,
            "binding_generation": int(job["binding_generation"]),
            "lifecycle_phase": lifecycle_phase,
        },
    }
    if error_class:
        response["error_class"] = error_class
    return response


def _settle_managed_close_replays_conn(
    conn: sqlite3.Connection,
    *,
    job: dict[str, Any],
    response: dict[str, Any],
    now: str,
) -> int:
    rows = conn.execute(
        """SELECT request_id, external_request_id, claim_token
           FROM write_request_replays
           WHERE endpoint='session_end' AND profile=? AND binding_hash=?
             AND session_id=? AND status='processing'""",
        (job["profile"], job["binding_hash"], job["session_id"]),
    ).fetchall()
    settled = 0
    replay_status = "committed" if response["lifecycle_phase"] == "ended" else "failed"
    for row in rows:
        claim_token = str(row["claim_token"] or "")
        _validate_hex64(claim_token, field="replay claim_token")
        payload = dict(response)
        payload["request_id"] = str(row["external_request_id"] or "")
        permit = ManagedQueuePermit(
            table_name="write_request_replays",
            binding_hash=str(job["binding_hash"]),
            claim_token=claim_token,
        )
        with managed_queue_write_permit(permit):
            cursor = conn.execute(
                """UPDATE write_request_replays
                   SET status=?, response_json=?, last_error=?, lease_owner=NULL,
                       lease_expires_at=NULL, next_retry_at=NULL, claim_token=NULL,
                       updated_at=?
                   WHERE endpoint='session_end' AND profile=? AND request_id=?
                     AND binding_hash=? AND session_id=? AND status='processing'
                     AND claim_token=?""",
                (
                    replay_status,
                    json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
                    None if replay_status == "committed" else str(response.get("error_class") or "needs_attention"),
                    now,
                    job["profile"],
                    row["request_id"],
                    job["binding_hash"],
                    job["session_id"],
                    claim_token,
                ),
            )
        settled += int(cursor.rowcount or 0)
    return settled


def terminalize_exhausted_stale_managed_session_end_jobs(
    *,
    profile: str,
    stale_before_iso: str,
    max_attempts: int,
    error: str,
    now: str | None = None,
    limit: int = 25,
) -> dict[str, int]:
    """Resolve exhausted managed close claims without replaying close effects.

    A durable committed/skipped receipt is sufficient to finish the episode as
    ended. Every other state is conservatively classified needs-attention. The
    effect and binding guards prevent racing a live worker; exact generation
    predicates prevent this cleanup from mutating a successor episode.
    """

    ts = now or _utc_iso(datetime.now(UTC))
    effective_attempts = max(0, int(max_attempts or 0))
    effective_limit = max(1, min(100, int(limit or 0)))
    with _read_db(operation="managed_close_stale_candidates") as conn:
        candidates = [
            dict(row)
            for row in conn.execute(
                """SELECT * FROM lifecycle_jobs
                   WHERE profile=? AND source='session_end' AND stage='close'
                     AND status='running' AND binding_hash IS NOT NULL
                     AND attempts >= ?
                     AND (
                        (lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                        OR (updated_at IS NOT NULL AND updated_at < ?)
                     )
                   ORDER BY updated_at ASC LIMIT ?""",
                (profile, effective_attempts, ts, stale_before_iso, effective_limit),
            ).fetchall()
        ]

    result = {"failed": 0, "committed": 0, "replays_settled": 0, "busy": 0}
    for candidate in candidates:
        binding_hash = str(candidate.get("binding_hash") or "")
        claim_token = str(candidate.get("claim_token") or "")
        _validate_hex64(binding_hash, field="binding_hash")
        _validate_hex64(claim_token, field="claim_token")
        try:
            effect_guard = EpisodeEffectGuard(binding_hash, profile=profile).acquire()
        except BindingBusyError:
            result["busy"] += 1
            continue
        try:
            try:
                binding_guard = BindingGuard(binding_hash, profile=profile).acquire()
            except BindingBusyError:
                result["busy"] += 1
                continue
            try:
                with _db_immediate(operation="managed_close_stale_terminalize") as conn:
                    row = conn.execute(
                        """SELECT * FROM lifecycle_jobs
                           WHERE profile=? AND job_id=? AND source='session_end' AND stage='close'
                             AND status='running' AND binding_hash=? AND session_id=?
                             AND binding_generation=? AND claim_token=? AND attempts >= ?
                             AND (
                                (lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                                OR (updated_at IS NOT NULL AND updated_at < ?)
                             )""",
                        (
                            profile,
                            candidate["job_id"],
                            binding_hash,
                            candidate["session_id"],
                            candidate["binding_generation"],
                            claim_token,
                            effective_attempts,
                            ts,
                            stale_before_iso,
                        ),
                    ).fetchone()
                    if row is None:
                        continue
                    job = dict(row)
                    try:
                        payload = json.loads(job.get("payload_json") or "{}")
                    except (TypeError, ValueError):
                        payload = {}
                    payload_hash = hashlib.sha256(
                        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
                    ).hexdigest()
                    receipt = conn.execute(
                        """SELECT state, payload_hash FROM session_close_step_receipts
                           WHERE session_id=? AND step_name='managed_close_pipeline' AND step_version=1""",
                        (job["session_id"],),
                    ).fetchone()
                    exact_episode = conn.execute(
                        """SELECT 1 FROM sessions
                           WHERE id=? AND binding_hash=? AND binding_generation=?
                             AND lifecycle_phase IN ('closing','ended','needs_attention')""",
                        (job["session_id"], binding_hash, int(job["binding_generation"])),
                    ).fetchone()
                    receipt_proves_commit = (
                        receipt is not None
                        and str(receipt["state"]) in {"committed", "skipped"}
                        and str(receipt["payload_hash"]) == payload_hash
                    )
                    if receipt_proves_commit and exact_episode is not None:
                        _finalize_managed_episode_conn(
                            conn,
                            binding_hash=binding_hash,
                            session_id=str(job["session_id"]),
                            generation=int(job["binding_generation"]),
                            lifecycle_phase="ended",
                            now=ts,
                        )
                        lifecycle_phase: Literal["ended", "needs_attention"] = "ended"
                        response = _terminal_close_response(job=job, lifecycle_phase=lifecycle_phase)
                        lifecycle_status = "committed"
                    else:
                        if exact_episode is not None:
                            _record_close_attention_conn(
                                conn,
                                job=job,
                                payload_hash=payload_hash,
                                error=error,
                                now=ts,
                            )
                        lifecycle_phase = "needs_attention"
                        response = _terminal_close_response(
                            job=job,
                            lifecycle_phase=lifecycle_phase,
                            error_class="ManagedCloseAttemptsExhausted",
                        )
                        lifecycle_status = "failed"

                    result["replays_settled"] += _settle_managed_close_replays_conn(
                        conn,
                        job=job,
                        response=response,
                        now=ts,
                    )
                    permit = ManagedQueuePermit(
                        table_name="lifecycle_jobs",
                        binding_hash=binding_hash,
                        claim_token=claim_token,
                    )
                    with managed_queue_write_permit(permit):
                        updated = conn.execute(
                            """UPDATE lifecycle_jobs
                               SET status=?, result_json=?, last_error=?, lease_owner=NULL,
                                   lease_expires_at=NULL, next_retry_at=NULL, claim_token=NULL,
                                   updated_at=?
                               WHERE profile=? AND job_id=? AND status='running'
                                 AND binding_hash=? AND session_id=? AND binding_generation=?
                                 AND claim_token=?""",
                            (
                                lifecycle_status,
                                json.dumps(response, sort_keys=True, separators=(",", ":"), default=str),
                                None if lifecycle_status == "committed" else error[:1000],
                                ts,
                                profile,
                                job["job_id"],
                                binding_hash,
                                job["session_id"],
                                job["binding_generation"],
                                claim_token,
                            ),
                        )
                    if updated.rowcount != 1:
                        raise ManagedBindingError("managed_episode_changed")
                    result[lifecycle_status] += 1
            finally:
                binding_guard.release()
        finally:
            effect_guard.release()
    return result


def run_managed_session_end_effect(
    job: dict[str, Any],
    *,
    run_effect: Callable[[], Any],
) -> dict[str, Any]:
    """Run one exact managed close under drain, effect, receipt, and CAS fences."""

    profile = str(job.get("profile") or "")
    job_id = str(job.get("job_id") or "")
    binding_hash = str(job.get("binding_hash") or "")
    session_id = str(job.get("session_id") or "")
    generation = _generation(job.get("binding_generation"))
    claim_token = str(job.get("claim_token") or "")
    _validate_hex64(binding_hash, field="binding_hash")
    _validate_hex64(claim_token, field="claim_token")
    if not profile or not job_id or not session_id or job.get("source") != "session_end" or job.get("stage") != "close":
        raise ManagedBindingError("binding_invariant_violation")
    payload = job.get("payload") or {}
    managed_context = payload.get("managed_context") if isinstance(payload, dict) else None
    if not isinstance(managed_context, dict) or (
        managed_context.get("profile"),
        managed_context.get("binding_hash"),
        managed_context.get("binding_generation"),
        managed_context.get("session_id"),
    ) != (profile, binding_hash, generation, session_id):
        raise ManagedBindingError("binding_context_mismatch")
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    effect_guard = EpisodeEffectGuard(binding_hash, profile=profile).acquire()
    try:
        binding_guard = BindingGuard(binding_hash, profile=profile).acquire()
        try:
            with _db_immediate(operation="managed_close_effect_authorization") as conn:
                binding = _load_binding(conn, profile=profile, binding_hash=binding_hash)
                episode = _load_episode(conn, session_id=session_id)
                claim = conn.execute(
                    """SELECT 1 FROM lifecycle_jobs
                       WHERE profile=? AND job_id=? AND source='session_end' AND stage='close'
                         AND status='running' AND claim_token=? AND session_id=?
                         AND binding_hash=? AND binding_generation=?""",
                    (profile, job_id, claim_token, session_id, binding_hash, generation),
                ).fetchone()
                if (
                    binding is None
                    or binding.get("state") != "active"
                    or episode is None
                    or episode.get("binding_hash") != binding_hash
                    or _generation(episode.get("binding_generation")) != generation
                    or episode.get("lifecycle_phase") not in {"closing", "ended", "needs_attention"}
                    or claim is None
                ):
                    raise ManagedBindingError("managed_episode_changed")

                work_state = _managed_close_work_state(
                    conn,
                    profile=profile,
                    binding_hash=binding_hash,
                    session_id=session_id,
                    job_id=job_id,
                )
                if work_state["pending"]:
                    raise ManagedCloseDeferred("managed_close_waiting_for_episode_work")
                now = _utc_iso(datetime.now(UTC))
                receipt = conn.execute(
                    """SELECT state, payload_hash FROM session_close_step_receipts
                       WHERE session_id=? AND step_name='managed_close_pipeline' AND step_version=1""",
                    (session_id,),
                ).fetchone()
                if work_state["failed"]:
                    _record_close_attention_conn(
                        conn,
                        job=job,
                        payload_hash=payload_hash,
                        error="accepted_episode_work_failed",
                        now=now,
                    )
                    return {
                        "status": "needs_attention",
                        "session_id": session_id,
                        "binding_generation": generation,
                        "lifecycle_phase": "needs_attention",
                        "error_class": "AcceptedEpisodeWorkFailed",
                        "final_learning_state": "failed",
                    }
                if receipt is not None:
                    state = str(receipt[0])
                    if str(receipt[1]) != payload_hash:
                        raise ManagedBindingError("binding_invariant_violation")
                    if state == "started":
                        _record_close_attention_conn(
                            conn,
                            job=job,
                            payload_hash=payload_hash,
                            error="uncertain_started_close_effect",
                            now=now,
                        )
                        return {
                            "status": "needs_attention",
                            "session_id": session_id,
                            "binding_generation": generation,
                            "lifecycle_phase": "needs_attention",
                            "error_class": "UncertainStartedCloseEffect",
                            "final_learning_state": "needs_attention",
                        }
                    terminal_phase = "ended" if state in {"committed", "skipped"} else "needs_attention"
                    _finalize_managed_episode_conn(
                        conn,
                        binding_hash=binding_hash,
                        session_id=session_id,
                        generation=generation,
                        lifecycle_phase=terminal_phase,
                        now=now,
                    )
                    return {
                        "status": terminal_phase,
                        "session_id": session_id,
                        "binding_generation": generation,
                        "lifecycle_phase": terminal_phase,
                        "final_learning_state": ("committed" if terminal_phase == "ended" else "needs_attention"),
                    }
                conn.execute(
                    """INSERT INTO session_close_step_receipts
                       (session_id, step_name, step_version, binding_hash, binding_generation,
                        state, claim_token, payload_hash, result_hash, created_at, updated_at, last_error)
                       VALUES (?, 'managed_close_pipeline', 1, ?, ?, 'started', ?, ?, NULL, ?, ?, NULL)""",
                    (session_id, binding_hash, generation, claim_token, payload_hash, now, now),
                )
        finally:
            binding_guard.release()

        permit = ManagedEpisodePermit(binding_hash=binding_hash, binding_generation=generation)
        try:
            with managed_session_write_permit(permit):
                effect_result = run_effect()
            if not isinstance(effect_result, dict) or effect_result.get("status") != "ended":
                raise RuntimeError("managed close effect did not end the exact episode")
        except Exception as exc:
            binding_guard = BindingGuard(binding_hash, profile=profile).acquire()
            try:
                with _db_immediate(operation="managed_close_effect_uncertain") as conn:
                    now = _utc_iso(datetime.now(UTC))
                    _record_close_attention_conn(
                        conn,
                        job=job,
                        payload_hash=payload_hash,
                        error=f"{type(exc).__name__}: {exc}",
                        now=now,
                    )
            finally:
                binding_guard.release()
            return {
                "status": "needs_attention",
                "session_id": session_id,
                "binding_generation": generation,
                "lifecycle_phase": "needs_attention",
                "error_class": type(exc).__name__,
                "final_learning_state": "needs_attention",
            }

        result_hash = hashlib.sha256(
            json.dumps(effect_result, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        binding_guard = BindingGuard(binding_hash, profile=profile).acquire()
        try:
            with _db_immediate(operation="managed_close_effect_commit") as conn:
                now = _utc_iso(datetime.now(UTC))
                receipt_update = conn.execute(
                    """UPDATE session_close_step_receipts
                       SET state='committed', result_hash=?, updated_at=?, last_error=NULL
                       WHERE session_id=? AND step_name='managed_close_pipeline' AND step_version=1
                         AND state='started' AND claim_token=? AND payload_hash=?""",
                    (result_hash, now, session_id, claim_token, payload_hash),
                )
                if receipt_update.rowcount != 1:
                    raise ManagedBindingError("managed_episode_changed")
                _finalize_managed_episode_conn(
                    conn,
                    binding_hash=binding_hash,
                    session_id=session_id,
                    generation=generation,
                    lifecycle_phase="ended",
                    now=now,
                )
        finally:
            binding_guard.release()
        result = dict(effect_result)
        result.update(
            {
                "status": "ended",
                "session_id": session_id,
                "binding_generation": generation,
                "lifecycle_phase": "ended",
                "final_learning_state": "committed",
            }
        )
        return result
    finally:
        effect_guard.release()
