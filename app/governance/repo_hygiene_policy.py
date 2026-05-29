"""Operational repo hygiene policy evaluation for conversation_turn."""

from __future__ import annotations

from typing import Any

from app.core.config import FEATURE_FLAGS, REPO_HYGIENE_RUNTIME_ROOT_MARKERS
from app.core.models import WorkspaceContext
from app.governance.policy_engine import PolicyViolation, get_policy_engine

RULE_ID = "repo_hygiene_session_isolation"
CONSTRAINT_ID = "constraint_repo_hygiene_session_isolation"
BLOCKING_FINDING_CODES = {"DUPLICATE_BRANCH_OWNER", "REGISTRY_NOT_LIVE", "MISSING_PATH"}
VIOLATING_CLASSIFICATIONS = {"canonical_checkout", "unregistered_worktree"}
ANTI_TERMS = [
    "edit from canonical checkout",
    "work directly in canonical checkout",
    "continue in unregistered worktree",
    "skip session isolation audit",
]


class RepoHygienePolicyError(Exception):
    """Raised when session-isolation policy blocks the current workspace."""

    def __init__(self, detail: str, *, workspace_context: dict[str, Any] | None = None):
        super().__init__(detail)
        self.error_code = "REPO_HYGIENE_POLICY_BLOCK"
        self.detail = detail
        self.workspace_context = workspace_context or {}


def _coerce_context(workspace_context: WorkspaceContext | dict[str, Any] | None) -> WorkspaceContext | None:
    if workspace_context is None:
        return None
    if isinstance(workspace_context, WorkspaceContext):
        return workspace_context
    if isinstance(workspace_context, dict):
        return WorkspaceContext.model_validate(workspace_context)
    return None


def _runtime_exception_reason(ctx: WorkspaceContext) -> str | None:
    current_path = ctx.current_path or ""
    for marker in REPO_HYGIENE_RUNTIME_ROOT_MARKERS:
        if marker and marker in current_path:
            return "runtime_release_worktree"
    return None


def _build_detail(
    ctx: WorkspaceContext,
    *,
    violation: bool,
    exception_reason: str | None,
    blocking_codes: list[str],
) -> str:
    parts = [
        f"classification={ctx.classification or 'unknown'}",
        f"path={ctx.current_path or 'unknown'}",
    ]
    if ctx.current_branch:
        parts.append(f"branch={ctx.current_branch}")
    if blocking_codes:
        parts.append(f"findings={','.join(blocking_codes)}")
    if exception_reason:
        parts.append(f"exception={exception_reason}")
    prefix = "Repo hygiene violation" if violation else "Repo hygiene pass"
    return f"{prefix}: " + " ".join(parts)


def _build_constraint(detail: str) -> dict[str, Any]:
    return {
        "concept_id": CONSTRAINT_ID,
        "constraint": (
            "[ALWAYS] CONSTRAINT: Do not perform active coding from a canonical checkout or "
            "an unregistered worktree. Move work into a registered session worktree before "
            f"editing. {detail}"
        ),
        "authority": 0.99,
        "anti_terms": ANTI_TERMS,
        "presentation_mode": "CONSTRAINT",
    }


def evaluate_repo_hygiene_policy(
    workspace_context: WorkspaceContext | dict[str, Any] | None,
    *,
    gov_ctx=None,
) -> dict[str, Any] | None:
    """Evaluate workspace session-isolation state as an operational policy."""
    if not FEATURE_FLAGS.get("REPO_HYGIENE_POLICY_ENABLED", False):
        return None

    ctx = _coerce_context(workspace_context)
    if ctx is None or not (ctx.current_path or ctx.repo_root):
        return None

    blocking_codes = [
        finding.code
        for finding in (ctx.findings or [])
        if finding.code in BLOCKING_FINDING_CODES
    ]
    exception_reason = _runtime_exception_reason(ctx)
    violation = exception_reason is None and (
        ctx.classification in VIOLATING_CLASSIFICATIONS or bool(blocking_codes)
    )
    decision = "BLOCK" if violation else "PASS"
    detail = _build_detail(
        ctx,
        violation=violation,
        exception_reason=exception_reason,
        blocking_codes=blocking_codes,
    )

    if gov_ctx:
        gov_ctx.log_policy_decision(
            policy_name=RULE_ID,
            concept_id="",
            decision=decision,
            severity=decision,
        )
        gov_ctx.log_event(
            "REPO_HYGIENE_POLICY_EVALUATED",
            None,
            {
                "classification": ctx.classification,
                "current_path": ctx.current_path,
                "repo_root": ctx.repo_root,
                "current_branch": ctx.current_branch,
                "branch_owner": ctx.branch_owner,
                "active_worktree_count": ctx.active_worktree_count,
                "finding_codes": blocking_codes,
                "exception_reason": exception_reason,
                "violation": violation,
            },
        )

    if violation:
        get_policy_engine().log_violation(
            PolicyViolation(
                rule_id=RULE_ID,
                severity="BLOCK",
                concept_id="",
                detail=detail,
                caller_context=f"conversation_turn:{ctx.classification}",
            )
        )

    return {
        "rule_id": RULE_ID,
        "classification": ctx.classification,
        "current_path": ctx.current_path,
        "repo_root": ctx.repo_root,
        "current_branch": ctx.current_branch,
        "branch_owner": ctx.branch_owner,
        "active_worktree_count": ctx.active_worktree_count,
        "finding_codes": blocking_codes,
        "exception_reason": exception_reason,
        "violation": violation,
        "detail": detail,
        "constraint": _build_constraint(detail) if violation else None,
        "workspace_context": ctx.model_dump(),
    }
