"""Consumer-neutral Pith lifecycle surface contract and adapter manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFORMANCE_SCHEMA_VERSION = "surface_lifecycle_conformance.v3"
SURFACE_LIFECYCLE_VERSION = "3.0"

VERDICT_ENFORCED = "enforced"
VERDICT_INSTRUCTION_MEDIATED = "instruction-mediated"
VERDICT_MANUAL_API_ONLY = "manual/API-only"
VERDICT_UNSUPPORTED = "unsupported"

VALID_VERDICTS = {
    VERDICT_ENFORCED,
    VERDICT_INSTRUCTION_MEDIATED,
    VERDICT_MANUAL_API_ONLY,
    VERDICT_UNSUPPORTED,
}

VALID_CONTEXT_TRIGGER_TYPES = {
    "pre_response_hook",
    "model_instruction",
    "mcp_tool",
    "manual_api",
    "unsupported",
}

VALID_LEARNING_TRIGGER_TYPES = {
    "stop_hook",
    "session_end_fallback",
    "model_instruction",
    "manual_api",
    "none",
}

VALID_LEARNING_TRANSPORTS = {
    "local_api",
    "mcp_stdio",
    "hook_wrapper",
    "unavailable",
}

VALID_LEARNING_PROBE_KINDS = {
    "hook_transcript_stop",
    "api_session_learn",
    "instruction_observed",
    "not_probeable",
    "unsupported",
}

VALID_LEARNING_QUALITY_MODES = {
    "raw_evidence_capture",
    "verified_concepts_required",
    "trivial_skip_only",
    "not_supported",
}

VALID_COHERENCE_PROBE_KINDS = {
    "hook_model_tool_trace",
    "api_echo",
    "not_probeable",
    "unsupported",
}

VALID_COHERENCE_STATUSES = {
    "passed",
    "failed",
    "unknown",
    "skipped_not_observed",
    "not_probeable",
}

VERDICT_RANK = {
    VERDICT_UNSUPPORTED: 0,
    VERDICT_MANUAL_API_ONLY: 1,
    VERDICT_INSTRUCTION_MEDIATED: 2,
    VERDICT_ENFORCED: 3,
}

REQUIRED_TURN_FIELDS = (
    "message",
    "origin_id",
    "surface_id",
    "workspace_id",
    "context_delivery_mode",
)

FOLLOWUP_TURN_FIELDS = (
    "session_id",
    "previous_message",
    "previous_response",
    "extracted_concepts_json",
)


@dataclass(frozen=True)
class SurfaceAdapterManifest:
    """Adapter declaration for one consumer surface."""

    surface_id: str
    client_id: str
    label: str
    install_method: str
    context_trigger_type: str
    transport: str
    context_delivery_mode: str
    timeout_budget_ms: int
    degradation_behavior: str
    context_enforcement_verdict: str
    learning_trigger_type: str
    learning_transport: str
    learning_capture_verdict: str
    learning_probe_kind: str
    learning_failure_policy: str
    learning_quality_mode: str
    coherence_probe_kind: str = "not_probeable"
    coherence_verdict: str = VERDICT_UNSUPPORTED
    coherence_required: bool = False
    coherence_failure_policy: str = "No model-visible coherence proof is available."
    config_path_templates: tuple[str, ...] = ()
    config_markers: tuple[str, ...] = ("pith",)
    conformance_expectation: str = ""
    supports_fresh_consumer: bool = True
    supports_cold_start: bool = False

    def __post_init__(self) -> None:
        if self.context_enforcement_verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid context_enforcement_verdict: {self.context_enforcement_verdict}")
        if self.learning_capture_verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid learning_capture_verdict: {self.learning_capture_verdict}")
        if self.context_trigger_type not in VALID_CONTEXT_TRIGGER_TYPES:
            raise ValueError(f"invalid context_trigger_type: {self.context_trigger_type}")
        if self.learning_trigger_type not in VALID_LEARNING_TRIGGER_TYPES:
            raise ValueError(f"invalid learning_trigger_type: {self.learning_trigger_type}")
        if self.learning_transport not in VALID_LEARNING_TRANSPORTS:
            raise ValueError(f"invalid learning_transport: {self.learning_transport}")
        if self.learning_probe_kind not in VALID_LEARNING_PROBE_KINDS:
            raise ValueError(f"invalid learning_probe_kind: {self.learning_probe_kind}")
        if self.learning_quality_mode not in VALID_LEARNING_QUALITY_MODES:
            raise ValueError(f"invalid learning_quality_mode: {self.learning_quality_mode}")
        if self.coherence_probe_kind not in VALID_COHERENCE_PROBE_KINDS:
            raise ValueError(f"invalid coherence_probe_kind: {self.coherence_probe_kind}")
        if self.coherence_verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid coherence_verdict: {self.coherence_verdict}")
        if self.timeout_budget_ms <= 0:
            raise ValueError("timeout_budget_ms must be positive")

    @property
    def trigger_type(self) -> str:
        """Compatibility alias for v1 callers that inspect a trigger type."""
        return self.context_trigger_type

    @property
    def expected_verdict(self) -> str:
        """Compatibility alias; new code must use the split verdict fields."""
        return minimum_verdict(
            self.context_enforcement_verdict,
            self.learning_capture_verdict,
            self.coherence_verdict if self.coherence_required else self.context_enforcement_verdict,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = CONFORMANCE_SCHEMA_VERSION
        return data


def _learning_defaults_for_context_verdict(
    context_verdict: str,
) -> tuple[str, str, str, str, str, str]:
    if context_verdict == VERDICT_ENFORCED:
        return (
            "stop_hook",
            "hook_wrapper",
            VERDICT_ENFORCED,
            "hook_transcript_stop",
            "Capture assistant response via post-response hook or report degraded learning.",
            "raw_evidence_capture",
        )
    if context_verdict == VERDICT_MANUAL_API_ONLY:
        return (
            "manual_api",
            "local_api",
            VERDICT_MANUAL_API_ONLY,
            "api_session_learn",
            "Caller must explicitly invoke learning API; no consumer trigger is implied.",
            "verified_concepts_required",
        )
    if context_verdict == VERDICT_INSTRUCTION_MEDIATED:
        return (
            "model_instruction",
            "unavailable",
            VERDICT_INSTRUCTION_MEDIATED,
            "instruction_observed",
            "Model/operator must perform learning and must report degraded state if unavailable.",
            "verified_concepts_required",
        )
    return (
        "none",
        "unavailable",
        VERDICT_UNSUPPORTED,
        "unsupported",
        "Learning capture is unsupported.",
        "not_supported",
    )


def surface_adapter_from_dict(raw: dict[str, Any]) -> SurfaceAdapterManifest:
    """Load a v2 manifest, accepting v1 manifest JSON as input compatibility."""
    context_verdict = raw.get("context_enforcement_verdict") or raw.get("expected_verdict")
    context_trigger_type = raw.get("context_trigger_type") or raw.get("trigger_type")
    if context_verdict is None:
        context_verdict = VERDICT_UNSUPPORTED
    if context_trigger_type is None:
        context_trigger_type = "unsupported"
    (
        default_learning_trigger,
        default_learning_transport,
        default_learning_verdict,
        default_probe_kind,
        default_failure_policy,
        default_quality_mode,
    ) = _learning_defaults_for_context_verdict(str(context_verdict))

    return SurfaceAdapterManifest(
        surface_id=raw["surface_id"],
        client_id=raw["client_id"],
        label=raw["label"],
        install_method=raw["install_method"],
        context_trigger_type=str(context_trigger_type),
        transport=raw["transport"],
        context_delivery_mode=raw["context_delivery_mode"],
        timeout_budget_ms=int(raw["timeout_budget_ms"]),
        degradation_behavior=raw["degradation_behavior"],
        context_enforcement_verdict=str(context_verdict),
        learning_trigger_type=raw.get("learning_trigger_type", default_learning_trigger),
        learning_transport=raw.get("learning_transport", default_learning_transport),
        learning_capture_verdict=raw.get("learning_capture_verdict", default_learning_verdict),
        learning_probe_kind=raw.get("learning_probe_kind", default_probe_kind),
        learning_failure_policy=raw.get("learning_failure_policy", default_failure_policy),
        learning_quality_mode=raw.get("learning_quality_mode", default_quality_mode),
        coherence_probe_kind=raw.get("coherence_probe_kind", "not_probeable"),
        coherence_verdict=raw.get("coherence_verdict", VERDICT_UNSUPPORTED),
        coherence_required=bool(raw.get("coherence_required", False)),
        coherence_failure_policy=raw.get(
            "coherence_failure_policy",
            "No model-visible coherence proof is available.",
        ),
        config_path_templates=tuple(raw.get("config_path_templates", ())),
        config_markers=tuple(raw.get("config_markers", ("pith",))),
        conformance_expectation=raw.get("conformance_expectation", ""),
        supports_fresh_consumer=bool(raw.get("supports_fresh_consumer", True)),
        supports_cold_start=bool(raw.get("supports_cold_start", False)),
    )


SURFACE_LIFECYCLE_ADAPTERS: dict[str, SurfaceAdapterManifest] = {
    "claude_code": SurfaceAdapterManifest(
        surface_id="claude_code",
        client_id="claude_code",
        label="Claude Code",
        install_method="configure_clients Claude Code hook installation",
        context_trigger_type="pre_response_hook",
        transport="local_api",
        context_delivery_mode="hook_additional_context",
        timeout_budget_ms=4000,
        degradation_behavior="Emit hook additionalContext warning and do not claim Pith context.",
        context_enforcement_verdict=VERDICT_ENFORCED,
        learning_trigger_type="stop_hook",
        learning_transport="hook_wrapper",
        learning_capture_verdict=VERDICT_ENFORCED,
        learning_probe_kind="hook_transcript_stop",
        learning_failure_policy="Stop hook must call session_learn or record explicit degraded learning.",
        learning_quality_mode="raw_evidence_capture",
        coherence_probe_kind="hook_model_tool_trace",
        coherence_verdict=VERDICT_ENFORCED,
        coherence_required=True,
        coherence_failure_policy="PostToolUse model-visible conversation_turn must match hook session and surface.",
        config_path_templates=(".claude/settings.json", ".claude.json"),
        config_markers=("UserPromptSubmit", "Stop", "pith", "conversation_turn"),
        conformance_expectation=(
            "A UserPromptSubmit hook runs conversation_turn before response composition; "
            "a Stop hook captures the assistant response after the turn; "
            "the model-visible binding must also call pith_conversation_turn each substantive turn "
            "so hook capture and model/tool coherence can be verified."
        ),
        supports_fresh_consumer=True,
        supports_cold_start=True,
    ),
    "codex_local_api": SurfaceAdapterManifest(
        surface_id="codex_local_api",
        client_id="codex",
        label="Codex",
        install_method="AGENTS.md lifecycle instructions plus local API wrapper",
        context_trigger_type="model_instruction",
        transport="local_api",
        context_delivery_mode="local_api_first_call",
        timeout_budget_ms=4000,
        degradation_behavior="Instruction requires explicit degraded response if local API is unavailable.",
        context_enforcement_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_trigger_type="model_instruction",
        learning_transport="unavailable",
        learning_capture_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_probe_kind="instruction_observed",
        learning_failure_policy="Model/operator compliance is required; no post-response trigger is enforced.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(".codex/AGENTS.md", ".codex/config.toml"),
        config_markers=("conversation_turn", "pith api", "origin_id"),
        conformance_expectation="AGENTS.md instructs Codex to call the local API before substantive responses.",
    ),
    "claude_desktop_mcp": SurfaceAdapterManifest(
        surface_id="claude_desktop_mcp",
        client_id="claude_desktop",
        label="Claude Desktop MCP",
        install_method="MCP server configuration",
        context_trigger_type="mcp_tool",
        transport="mcp_stdio",
        context_delivery_mode="mcp_tool_call",
        timeout_budget_ms=5000,
        degradation_behavior="MCP tool failure must be surfaced as unavailable context.",
        context_enforcement_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_trigger_type="model_instruction",
        learning_transport="mcp_stdio",
        learning_capture_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_probe_kind="instruction_observed",
        learning_failure_policy="Model/tool compliance is required; no post-response trigger is enforced.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(
            "Library/Application Support/Claude/claude_desktop_config.json",
            ".config/Claude/claude_desktop_config.json",
        ),
        config_markers=("pith", "pith_conversation_turn", "mcpServers"),
        conformance_expectation="MCP config exposes pith_conversation_turn; model compliance is still required.",
    ),
    "cursor_mcp": SurfaceAdapterManifest(
        surface_id="cursor_mcp",
        client_id="cursor",
        label="Cursor MCP",
        install_method="MCP server configuration",
        context_trigger_type="mcp_tool",
        transport="mcp_stdio",
        context_delivery_mode="mcp_tool_call",
        timeout_budget_ms=5000,
        degradation_behavior="MCP tool failure must be surfaced as unavailable context.",
        context_enforcement_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_trigger_type="model_instruction",
        learning_transport="mcp_stdio",
        learning_capture_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_probe_kind="instruction_observed",
        learning_failure_policy="Model/tool compliance is required; no post-response trigger is enforced.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(".cursor/mcp.json",),
        config_markers=("pith", "pith_conversation_turn", "mcpServers"),
        conformance_expectation="MCP config exposes Pith tools; model compliance is still required.",
    ),
    "vscode_copilot_mcp": SurfaceAdapterManifest(
        surface_id="vscode_copilot_mcp",
        client_id="vscode",
        label="VS Code Copilot MCP",
        install_method="MCP server configuration plus Copilot instruction file",
        context_trigger_type="model_instruction",
        transport="mcp_stdio",
        context_delivery_mode="instruction_only",
        timeout_budget_ms=5000,
        degradation_behavior="Instruction file requires degraded response if tool/API path is unavailable.",
        context_enforcement_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_trigger_type="model_instruction",
        learning_transport="mcp_stdio",
        learning_capture_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_probe_kind="instruction_observed",
        learning_failure_policy="Model/tool compliance is required; no post-response trigger is enforced.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(
            ".copilot/instructions/pith-cognitive-loop.instructions.md",
            "Library/Application Support/Code/User/mcp.json",
            ".config/Code/User/mcp.json",
        ),
        config_markers=("pith_conversation_turn", "conversation_turn", "Agent mode"),
        conformance_expectation=(
            "Copilot instructions and MCP config expose lifecycle path; model compliance is required."
        ),
    ),
    "windsurf_mcp": SurfaceAdapterManifest(
        surface_id="windsurf_mcp",
        client_id="windsurf",
        label="Windsurf MCP",
        install_method="MCP server configuration",
        context_trigger_type="mcp_tool",
        transport="mcp_stdio",
        context_delivery_mode="mcp_tool_call",
        timeout_budget_ms=5000,
        degradation_behavior="MCP tool failure must be surfaced as unavailable context.",
        context_enforcement_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_trigger_type="model_instruction",
        learning_transport="mcp_stdio",
        learning_capture_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_probe_kind="instruction_observed",
        learning_failure_policy="Model/tool compliance is required; no post-response trigger is enforced.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(".codeium/windsurf/mcp_config.json",),
        config_markers=("pith", "pith_conversation_turn", "mcpServers"),
        conformance_expectation="MCP config exposes Pith tools; model compliance is still required.",
    ),
    "cline_mcp": SurfaceAdapterManifest(
        surface_id="cline_mcp",
        client_id="cline",
        label="Cline MCP",
        install_method="MCP server configuration",
        context_trigger_type="mcp_tool",
        transport="mcp_stdio",
        context_delivery_mode="mcp_tool_call",
        timeout_budget_ms=5000,
        degradation_behavior="MCP tool failure must be surfaced as unavailable context.",
        context_enforcement_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_trigger_type="model_instruction",
        learning_transport="mcp_stdio",
        learning_capture_verdict=VERDICT_INSTRUCTION_MEDIATED,
        learning_probe_kind="instruction_observed",
        learning_failure_policy="Model/tool compliance is required; no post-response trigger is enforced.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(
            "Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
            ".config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
        ),
        config_markers=("pith", "pith_conversation_turn", "mcpServers"),
        conformance_expectation="MCP config exposes Pith tools; model compliance is still required.",
    ),
    "local_api_cli": SurfaceAdapterManifest(
        surface_id="local_api_cli",
        client_id="local_api",
        label="Local API CLI",
        install_method="Direct local HTTP/API wrapper",
        context_trigger_type="manual_api",
        transport="local_api",
        context_delivery_mode="local_api_first_call",
        timeout_budget_ms=4000,
        degradation_behavior="CLI returns an explicit error; caller must not claim retrieved context.",
        context_enforcement_verdict=VERDICT_MANUAL_API_ONLY,
        learning_trigger_type="manual_api",
        learning_transport="local_api",
        learning_capture_verdict=VERDICT_MANUAL_API_ONLY,
        learning_probe_kind="api_session_learn",
        learning_failure_policy="Caller must explicitly invoke session_learn; no consumer trigger is implied.",
        learning_quality_mode="verified_concepts_required",
        config_path_templates=(),
        config_markers=(),
        conformance_expectation="Direct API call works, but no consumer turn trigger is implied.",
    ),
}


def get_surface_adapter(surface_id: str) -> SurfaceAdapterManifest | None:
    return SURFACE_LIFECYCLE_ADAPTERS.get((surface_id or "").strip().lower())


def surface_adapter_dicts() -> list[dict[str, Any]]:
    return [manifest.to_dict() for manifest in SURFACE_LIFECYCLE_ADAPTERS.values()]


def config_candidates(manifest: SurfaceAdapterManifest, home: Path) -> list[Path]:
    return [home / template for template in manifest.config_path_templates]


def inspect_config_presence(manifest: SurfaceAdapterManifest, home: Path) -> dict[str, Any]:
    """Inspect consumer config files without assuming a Pith developer path."""
    candidates = []
    marker_hit = False
    for path in config_candidates(manifest, home):
        exists = path.is_file()
        markers_found: list[str] = []
        if exists:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            markers_found = [marker for marker in manifest.config_markers if marker in text]
            marker_hit = marker_hit or bool(markers_found) or not manifest.config_markers
        candidates.append(
            {
                "path": str(path),
                "exists": exists,
                "markers_found": markers_found,
            }
        )
    configured = True if not manifest.config_path_templates else marker_hit
    return {
        "configured": configured,
        "home": str(home),
        "candidates": candidates,
    }


def build_turn_payload(
    manifest: SurfaceAdapterManifest,
    *,
    origin_id: str,
    workspace_id: str,
    message: str,
    session_id: str | None = None,
    previous_message: str | None = None,
    previous_response: str | None = None,
    extracted_concepts_json: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": message,
        "origin_id": origin_id,
        "workspace_id": workspace_id,
        "surface_id": manifest.surface_id,
        "context_delivery_mode": manifest.context_delivery_mode,
        "surface_lifecycle_version": SURFACE_LIFECYCLE_VERSION,
        "max_concepts": 1 if manifest.context_enforcement_verdict == VERDICT_ENFORCED else 4,
        "include_verbatim": manifest.context_enforcement_verdict != VERDICT_ENFORCED,
    }
    if session_id:
        payload["session_id"] = session_id
    if previous_message is not None:
        payload["previous_message"] = previous_message
    if previous_response is not None:
        payload["previous_response"] = previous_response
    if extracted_concepts_json is not None:
        payload["extracted_concepts_json"] = extracted_concepts_json
    return payload


def minimum_verdict(*verdicts: str) -> str:
    return min(verdicts, key=lambda verdict: VERDICT_RANK.get(verdict, 0))


def _verdict_claim(verdict: str) -> str:
    if verdict == VERDICT_ENFORCED:
        return "enforced"
    if verdict == VERDICT_INSTRUCTION_MEDIATED:
        return "not enforced; compliance depends on model/operator"
    if verdict == VERDICT_MANUAL_API_ONLY:
        return "manual/API-only; no consumer trigger is implied"
    return "unsupported"


def classify_context_payload_quality(first_turn_call: dict[str, Any]) -> dict[str, Any]:
    """Classify whether a successful first-turn call delivered usable context."""
    if not isinstance(first_turn_call, dict):
        return {"status": "unknown", "concept_count": 0, "has_orientation": False}

    if first_turn_call.get("status") not in {"ok", "skipped"}:
        return {
            "status": "degraded",
            "concept_count": 0,
            "has_orientation": False,
            "reason": first_turn_call.get("error") or "first-turn call failed",
        }
    if first_turn_call.get("status") == "skipped":
        return {
            "status": "unknown",
            "concept_count": 0,
            "has_orientation": False,
            "reason": first_turn_call.get("reason") or "first-turn call skipped",
        }

    response = first_turn_call.get("response")
    if not isinstance(response, dict):
        response = first_turn_call
    concepts = response.get("activated_concepts") or []
    concept_count = len(concepts) if isinstance(concepts, list) else 0
    has_orientation = bool(response.get("orientation_summary"))
    if concept_count > 0 or has_orientation:
        status = "delivered"
    elif response.get("resolved_session_id") or first_turn_call.get("resolved_session_id"):
        status = "registration_only"
    else:
        status = "unknown"

    governance_summary = response.get("governance_summary") if isinstance(response, dict) else None
    return {
        "status": status,
        "concept_count": concept_count,
        "has_orientation": has_orientation,
        "coverage_score": response.get("coverage_score") if isinstance(response, dict) else None,
        "phases_executed": (governance_summary or {}).get("phases_executed", []),
        "phases_skipped": (governance_summary or {}).get("phases_skipped", []),
    }


def evaluate_context_phase(
    manifest: SurfaceAdapterManifest,
    *,
    config_result: dict[str, Any],
    first_turn_call: dict[str, Any],
    lane: str,
) -> dict[str, Any]:
    limitations: list[str] = []
    configured = bool(config_result.get("configured"))
    call_status = first_turn_call.get("status")
    payload_quality = classify_context_payload_quality(first_turn_call)

    if lane == "fresh-consumer" and not manifest.supports_fresh_consumer:
        return {
            "status": "failed",
            "proof_status": "failed",
            "verdict": VERDICT_UNSUPPORTED,
            "claim": _verdict_claim(VERDICT_UNSUPPORTED),
            "limitations": ["adapter does not support fresh-consumer conformance"],
        }

    if not configured:
        return {
            "status": "failed",
            "proof_status": "failed",
            "verdict": VERDICT_UNSUPPORTED,
            "claim": _verdict_claim(VERDICT_UNSUPPORTED),
            "limitations": ["required consumer config artifact missing or lacks Pith lifecycle marker"],
        }

    if call_status == "ok":
        status = "passed"
        proof_status = "passed"
        if (
            manifest.context_enforcement_verdict == VERDICT_ENFORCED
            and payload_quality.get("status") != "delivered"
        ):
            status = "degraded"
            proof_status = "failed"
            limitations.append(
                "first-turn call registered the turn but did not deliver retrieved context"
            )
    elif call_status == "skipped":
        status = "skipped"
        proof_status = "skipped"
        limitations.append("first-turn API call was skipped by operator/test harness")
    else:
        return {
            "status": "failed",
            "proof_status": "failed",
            "verdict": VERDICT_UNSUPPORTED,
            "claim": _verdict_claim(VERDICT_UNSUPPORTED),
            "limitations": [first_turn_call.get("error") or "first-turn call failed"],
        }

    if manifest.context_enforcement_verdict == VERDICT_INSTRUCTION_MEDIATED:
        limitations.append("context is not enforced; compliance depends on model/operator")
    if lane == "dogfood":
        limitations.append("dogfood proof is local operational evidence, not consumer conformance")
    if lane != "cold-start" and manifest.supports_cold_start:
        limitations.append("cold-start behavior is a separate proof lane")

    return {
        "status": status,
        "proof_status": proof_status,
        "trigger_type": manifest.context_trigger_type,
        "verdict": manifest.context_enforcement_verdict,
        "claim": _verdict_claim(manifest.context_enforcement_verdict),
        "configured": configured,
        "call": first_turn_call,
        "context_payload_quality": payload_quality,
        "limitations": limitations,
    }


def evaluate_learning_phase(
    manifest: SurfaceAdapterManifest,
    *,
    learning_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    probe = learning_probe or {"status": "skipped", "reason": "learning probe not run"}
    probe_status = probe.get("status")
    limitations: list[str] = []

    if manifest.learning_capture_verdict == VERDICT_UNSUPPORTED:
        limitations.append("learning capture is unsupported for this surface")
        return {
            "status": "skipped",
            "proof_status": "skipped",
            "trigger_type": manifest.learning_trigger_type,
            "transport": manifest.learning_transport,
            "probe_kind": manifest.learning_probe_kind,
            "verdict": VERDICT_UNSUPPORTED,
            "claim": _verdict_claim(VERDICT_UNSUPPORTED),
            "probe": probe,
            "limitations": limitations,
        }

    if probe_status == "ok":
        status = "passed"
        proof_status = "passed"
        verdict = manifest.learning_capture_verdict
        if manifest.learning_capture_verdict == VERDICT_ENFORCED:
            try:
                accepted_events = int(probe.get("accepted_learning_events") or 0)
            except (TypeError, ValueError):
                accepted_events = 0
            try:
                learning_events = int(probe.get("learning_events") or 0)
            except (TypeError, ValueError):
                learning_events = 0
            capture_state = str(probe.get("learning_capture_state") or "")
            immediate_status = str(probe.get("immediate_status") or "")
            linkage_state = str(probe.get("session_linkage_state") or "")
            has_learning_evidence = (
                accepted_events > 0
                or learning_events > 0
                or capture_state == "accepted"
                or immediate_status == "committed"
            )
            if not has_learning_evidence:
                status = "failed"
                proof_status = "failed"
                verdict = VERDICT_UNSUPPORTED
                limitations.append("enforced learning proof lacks committed learning evidence")
            elif linkage_state != "linked":
                status = "failed"
                proof_status = "failed"
                verdict = VERDICT_UNSUPPORTED
                limitations.append("enforced learning proof lacks same-session linkage")
    elif probe_status == "skipped":
        status = "skipped"
        proof_status = "failed" if manifest.learning_capture_verdict == VERDICT_ENFORCED else "skipped"
        verdict = (
            VERDICT_UNSUPPORTED
            if manifest.learning_capture_verdict == VERDICT_ENFORCED
            else manifest.learning_capture_verdict
        )
        limitations.append(probe.get("reason") or "learning probe was skipped")
    elif probe_status == "degraded":
        status = "degraded"
        proof_status = "degraded"
        verdict = manifest.learning_capture_verdict
        limitations.append(probe.get("reason") or "learning probe is pending/degraded")
    else:
        status = "failed"
        proof_status = "failed"
        verdict = VERDICT_UNSUPPORTED
        limitations.append(probe.get("error") or probe.get("reason") or "learning probe failed")

    if verdict == VERDICT_INSTRUCTION_MEDIATED:
        limitations.append("learning is not enforced; compliance depends on model/operator")
    if verdict == VERDICT_MANUAL_API_ONLY:
        limitations.append("learning is manual/API-only; no consumer trigger is implied")

    return {
        "status": status,
        "proof_status": proof_status,
        "trigger_type": manifest.learning_trigger_type,
        "transport": manifest.learning_transport,
        "probe_kind": manifest.learning_probe_kind,
        "quality_mode": manifest.learning_quality_mode,
        "verdict": verdict,
        "claim": _verdict_claim(verdict),
        "probe": probe,
        "limitations": limitations,
    }


def evaluate_coherence_phase(
    manifest: SurfaceAdapterManifest,
    *,
    coherence_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    probe = coherence_probe or {
        "status": "not_probeable" if not manifest.coherence_required else "skipped_not_observed",
        "reason": "coherence probe not run",
    }
    probe_status = str(probe.get("coherence_status") or probe.get("status") or "unknown")
    limitations: list[str] = []

    if manifest.coherence_probe_kind in {"not_probeable", "unsupported"} and not manifest.coherence_required:
        limitations.append("model-visible coherence is not probeable for this surface")
        return {
            "status": "not_probeable",
            "proof_status": "skipped",
            "required": False,
            "probe_kind": manifest.coherence_probe_kind,
            "verdict": manifest.coherence_verdict,
            "claim": "coherence not probeable; no enforced claim is made",
            "probe": probe,
            "limitations": limitations,
        }

    if probe_status == "passed":
        status = "passed"
        proof_status = "passed"
        verdict = manifest.coherence_verdict
    elif probe_status == "skipped_not_observed":
        status = "skipped_not_observed"
        proof_status = "failed" if manifest.coherence_required else "skipped"
        verdict = VERDICT_UNSUPPORTED if manifest.coherence_required else manifest.coherence_verdict
        limitations.append(probe.get("reason") or "model-visible conversation_turn was not observed")
    elif probe_status == "not_probeable":
        status = "not_probeable"
        proof_status = "failed" if manifest.coherence_required else "skipped"
        verdict = VERDICT_UNSUPPORTED if manifest.coherence_required else manifest.coherence_verdict
        limitations.append(probe.get("reason") or "model-visible coherence is not probeable")
    elif probe_status in {"failed", "unknown"}:
        status = probe_status
        proof_status = "failed" if manifest.coherence_required or probe_status == "failed" else "skipped"
        verdict = VERDICT_UNSUPPORTED if manifest.coherence_required or probe_status == "failed" else manifest.coherence_verdict
        limitations.append(probe.get("reason") or probe.get("error") or "model-visible coherence failed")
    else:
        status = "unknown"
        proof_status = "failed" if manifest.coherence_required else "skipped"
        verdict = VERDICT_UNSUPPORTED if manifest.coherence_required else manifest.coherence_verdict
        limitations.append(f"invalid coherence status: {probe_status}")

    return {
        "status": status,
        "proof_status": proof_status,
        "required": manifest.coherence_required,
        "probe_kind": manifest.coherence_probe_kind,
        "verdict": verdict,
        "claim": _verdict_claim(verdict),
        "probe": probe,
        "limitations": limitations,
    }


def combine_overall_verdict(
    context_phase: dict[str, Any],
    learning_phase: dict[str, Any],
    coherence_phase: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context_verdict = context_phase.get("verdict", VERDICT_UNSUPPORTED)
    learning_verdict = learning_phase.get("verdict", VERDICT_UNSUPPORTED)
    coherence_phase = coherence_phase or {}
    coherence_required = bool(coherence_phase.get("required"))
    verdicts = [context_verdict, learning_verdict]
    if coherence_required:
        verdicts.append(coherence_phase.get("verdict", VERDICT_UNSUPPORTED))
    verdict = minimum_verdict(*verdicts)

    proof_inputs = [context_phase.get("proof_status"), learning_phase.get("proof_status")]
    if coherence_required:
        proof_inputs.append(coherence_phase.get("proof_status"))

    if all(status == "passed" for status in proof_inputs):
        proof_status = "passed"
    elif "failed" in set(proof_inputs):
        proof_status = "failed"
    elif "degraded" in set(proof_inputs):
        proof_status = "degraded"
    else:
        proof_status = "skipped"

    if context_verdict == VERDICT_ENFORCED and learning_verdict != VERDICT_ENFORCED:
        claim = f"split result: context enforced; learning {learning_phase.get('claim')}"
    elif coherence_required and coherence_phase.get("verdict") != VERDICT_ENFORCED:
        claim = f"split result: context and learning enforced; coherence {coherence_phase.get('claim')}"
    elif verdict == VERDICT_ENFORCED:
        claim = "fully enforced for context, learning, and coherence" if coherence_required else "fully enforced for context and learning"
    else:
        claim = _verdict_claim(verdict)

    return {
        "proof_status": proof_status,
        "verdict": verdict,
        "claim": claim,
        "context_verdict": context_verdict,
        "learning_verdict": learning_verdict,
        "coherence_verdict": coherence_phase.get("verdict") if coherence_phase else None,
    }


def evaluate_conformance(
    manifest: SurfaceAdapterManifest,
    *,
    config_result: dict[str, Any],
    first_turn_call: dict[str, Any],
    lane: str,
    learning_probe: dict[str, Any] | None = None,
    coherence_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context_phase = evaluate_context_phase(
        manifest,
        config_result=config_result,
        first_turn_call=first_turn_call,
        lane=lane,
    )
    learning_phase = evaluate_learning_phase(manifest, learning_probe=learning_probe)
    coherence_phase = evaluate_coherence_phase(manifest, coherence_probe=coherence_probe)
    overall = combine_overall_verdict(context_phase, learning_phase, coherence_phase)
    limitations = []
    limitations.extend(context_phase.get("limitations", []))
    limitations.extend(learning_phase.get("limitations", []))
    limitations.extend(coherence_phase.get("limitations", []))
    return {
        "context_phase": context_phase,
        "learning_phase": learning_phase,
        "coherence_phase": coherence_phase,
        "overall_verdict": overall,
        "limitations": limitations,
    }
