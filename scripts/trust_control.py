#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.trust_governance import TrustGovernanceResolution, resolve_trust_governance

SCHEMA_VERSION = "trust_control_status.v0"
ERROR_SCHEMA_VERSION = "trust_control_error.v0"
REVIEW_OPERATIONS = {
    "review_next",
    "review_status",
    "review_list",
    "review_show",
    "review_focus",
    "review_decide",
    "review_run",
    "review_export",
    "review_consume",
    "review_measure",
    "review_calibration",
}

STATUS_MEANINGS = {
    "current": "Pith found a current authority for this question.",
    "mixed": "Pith found current authority plus older historical context.",
    "historical": "Pith found historical context, but it should not govern current decisions.",
    "ambiguous": "Pith found more than one possible domain and needs a narrower question.",
    "unknown": "Pith does not know a governing authority for this question yet.",
}


def _payload_error(message: str, field: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": True,
        "code": "INVALID_TRUST_CONTROL_PAYLOAD",
        "schema_version": ERROR_SCHEMA_VERSION,
        "message": message,
    }
    if field:
        result["field"] = field
    return result


def _question_from_payload(payload: Any) -> tuple[str | None, dict[str, Any] | None]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return None, _payload_error("trust payload must be a JSON object")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, _payload_error("question must be a non-empty string", "question")
    return question.strip(), None


def status_from_resolution(resolution: TrustGovernanceResolution) -> str:
    if resolution.intent == "ambiguous_domain":
        return "ambiguous"
    if resolution.intent == "no_known_authority":
        return "unknown"
    if resolution.current_authority is not None and resolution.historical_items:
        return "mixed"
    if resolution.current_authority is not None:
        return "current"
    if resolution.historical_items:
        return "historical"
    return "unknown"


def next_action_for_resolution(resolution: TrustGovernanceResolution) -> str:
    if resolution.intent == "ambiguous_domain":
        domains = ", ".join(resolution.matched_domain_ids)
        return f"Ask again with one specific domain or project area. Possible matches: {domains}."
    if resolution.intent == "no_known_authority":
        return "Ask about a specific policy, decision, claim, document, or project area."
    if resolution.current_authority is not None:
        return "Use the current authority above for decisions in this domain."
    if resolution.historical_items:
        return "Use this as historical context only; it does not govern current decisions."
    return "Clarify the domain or evidence Pith should use."


def build_status(question: str) -> dict[str, Any]:
    resolution = resolve_trust_governance(question)
    evidence = resolution.evidence if isinstance(resolution.evidence, dict) else {}
    status = status_from_resolution(resolution)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "meaning": STATUS_MEANINGS.get(status, STATUS_MEANINGS["unknown"]),
        "question": question,
        "domain_id": resolution.domain_id,
        "intent": resolution.intent,
        "current_authority": evidence.get("current_authority"),
        "historical_items": evidence.get("historical_items", []),
        "matched_domain_ids": list(resolution.matched_domain_ids),
        "reason_current_governs": evidence.get("reason_current_governs"),
        "next_action": next_action_for_resolution(resolution),
    }


def build_status_from_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        operation = payload.get("operation")
        if operation in REVIEW_OPERATIONS:
            from scripts import trust_review_control

            return trust_review_control.build_from_payload(payload)
        if operation in {"preview_correction", "apply_correction"}:
            return build_correction_from_payload(payload)
    question, error = _question_from_payload(payload)
    if error:
        return error
    assert question is not None
    return build_status(question)


def build_correction_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _payload_error("trust correction payload must be a JSON object")
    question, error = _question_from_payload(payload)
    if error:
        return error
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return _payload_error("message must be a non-empty string", "message")
    operation = payload.get("operation")
    from app.trust_governance_correction_control import apply_correction, build_correction_preview

    if operation == "preview_correction":
        return build_correction_preview(question, message)
    if operation == "apply_correction":
        return apply_correction(question, message, confirm=bool(payload.get("confirm", False)))
    return _payload_error("operation must be status, preview_correction, or apply_correction", "operation")


def _display_domain(domain_id: Any) -> str:
    if not domain_id:
        return "unknown"
    return str(domain_id).replace("_", " ")


def _source_text(item: dict[str, Any] | None) -> str:
    if not item:
        return "not available"
    parts: list[str] = []
    if item.get("source_path"):
        parts.append(str(item["source_path"]))
    if item.get("source_commit"):
        parts.append(f"@ {item['source_commit']}")
    if item.get("source_url"):
        parts.append(f"URL: {item['source_url']}")
    return " ".join(parts) if parts else "not available"


def _user_text(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace(" for the fixture domain", "")


def format_text(status: dict[str, Any]) -> str:
    lines = [
        "[Trust]",
        f"  Status:     {status.get('status', 'unknown')}",
        f"  Meaning:    {status.get('meaning') or STATUS_MEANINGS['unknown']}",
        f"  Domain:     {_display_domain(status.get('domain_id'))}",
        f"  Question:   {status.get('question', '')}",
    ]
    current = status.get("current_authority")
    if isinstance(current, dict):
        lines.extend(
            [
                f"  Current:    {current.get('title') or current.get('item_id')}",
                f"  Source:     {_source_text(current)}",
            ]
        )
        reason = _user_text(status.get("reason_current_governs") or current.get("reason"))
        if reason:
            lines.append(f"  Why:        {reason}")
    historical = status.get("historical_items")
    if isinstance(historical, list) and historical:
        lines.append("  Historical:")
        for item in historical:
            if not isinstance(item, dict):
                continue
            title = _user_text(item.get("title") or item.get("item_id"))
            state = item.get("state") or "historical"
            lines.append(f"    - {title} ({state}); Source: {_source_text(item)}")
    lines.extend(
        [
            "",
            str(status.get("next_action") or "Clarify the domain or evidence Pith should use."),
            "Correction path: use `pith trust correct --question QUESTION --message MESSAGE` to preview a local authority correction; add `--apply --confirm` only after review.",
        ]
    )
    return "\n".join(lines)


def format_correction_text(result: dict[str, Any]) -> str:
    lines = ["[Trust Correction]"]
    status = result.get("status") or result.get("proposed_action") or "preview"
    lines.append(f"  Status:     {status}")
    if result.get("domain_id"):
        lines.append(f"  Domain:     {_display_domain(result.get('domain_id'))}")
    envelope = result.get("envelope") if isinstance(result.get("envelope"), dict) else result
    if isinstance(envelope, dict):
        if envelope.get("intent"):
            lines.append(f"  Intent:     {envelope.get('intent')}")
        if envelope.get("ambiguity_state"):
            lines.append(f"  Resolution: {envelope.get('ambiguity_state')}")
        if envelope.get("proposed_action"):
            lines.append(f"  Action:     {envelope.get('proposed_action')}")
        if envelope.get("envelope_hash"):
            lines.append(f"  Evidence:   {envelope.get('envelope_hash')}")
    if result.get("override_path"):
        lines.append(f"  Override:   {result.get('override_path')}")
    proof = result.get("next_answer_proof")
    if isinstance(proof, dict) and proof.get("current_authority"):
        current = proof["current_authority"]
        if isinstance(current, dict):
            lines.append(f"  Current:    {current.get('title') or current.get('item_id')}")
    if result.get("mutates_authority"):
        lines.append("")
        lines.append("Correction applied. Future trust answers for this domain use the override above.")
    elif result.get("proposed_action") == "apply_authority_override":
        lines.append("")
        lines.append("Preview only. Re-run with --apply --confirm to update local trust authority.")
    else:
        reason = result.get("reason") or (envelope.get("ambiguity_state") if isinstance(envelope, dict) else None)
        lines.append("")
        lines.append(f"No trust authority was changed. Reason: {reason or 'not applicable'}.")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect what Pith currently trusts for a question.",
        epilog=(
            "Use 'pith trust explain QUESTION' for the same read-only inspection with user-facing guidance. "
            "Related commands: 'pith trust correct --question QUESTION --message MESSAGE' "
            "previews or applies explicit trust corrections; 'pith trust review status/list/focus/show' "
            "inspects implicit supersession proposals; 'pith trust review decide/run' records "
            "explicit review decisions; 'pith trust review consume' builds measured repair dry-run evidence; "
            "'pith trust review next' explains the safest next maintenance step."
        ),
    )
    parser.add_argument("question_parts", nargs="*", metavar="QUESTION")
    parser.add_argument("--question", dest="question")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def _correction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or apply a correction to Pith trust authority.",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--apply", action="store_true", dest="apply")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] == "review":
        from scripts import trust_review_control

        return trust_review_control.main(argv[1:])
    if argv and argv[0] == "correct":
        parser = _correction_parser()
        args = parser.parse_args(argv[1:])
        from app.trust_governance_correction_control import apply_correction, build_correction_preview

        result = (
            apply_correction(args.question, args.message, confirm=args.confirm)
            if args.apply
            else build_correction_preview(args.question, args.message)
        )
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(format_correction_text(result))
        return 1 if result.get("error") is True or result.get("status") == "rejected" else 0
    if argv and argv[0] == "explain":
        argv = argv[1:]

    parser = _parser()
    args = parser.parse_args(argv)
    question = (args.question or " ".join(args.question_parts)).strip()
    if not question:
        parser.error("question is required")
    status = build_status(question)
    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        print(format_text(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
