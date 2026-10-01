#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "trust_health_status.v0"
CLAIM_BOUNDARY = (
    "Internal product-continuity scorecard; not a public benchmark claim. "
    "This proves recurring fixture effectiveness, not arbitrary user-domain governance."
)
USER_MEANING = (
    "Pith can measure current trust checks and supersession edge risks. "
    "These checks do not prove every possible answer is correct."
)
DEFAULT_COMPACT_LOG_PATH = (
    Path.home() / ".pith" / "logs" / "monitoring" / "trust-governance-effectiveness-monitor.log"
)
DEFAULT_SUPERSESSION_COMPACT_LOG_PATH = (
    Path.home() / ".pith" / "logs" / "monitoring" / "supersession-edge-semantics-monitor.log"
)
DEFAULT_REPORTS_DIR = Path.home() / ".pith" / "reports" / "monitoring"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8000/health"
DEFAULT_LAUNCHD_LABEL = "dev.pith.monitor-trust-governance-effectiveness"
DEFAULT_SUPERSESSION_LAUNCHD_LABEL = "dev.pith.monitor-supersession-edge-semantics"
DEFAULT_HISTORY_LIMIT = 10
DEFAULT_WARN_AFTER_SECONDS = 36 * 60 * 60
DEFAULT_CRITICAL_AFTER_SECONDS = 72 * 60 * 60
DEFAULT_HEALTH_TIMEOUT_SECONDS = 2.0
DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_LOG_BYTES = 2_000_000
HISTORY_SCOPE_LABEL = (
    "compact monitor runs loaded by this command; not proof that every run was launchd-scheduled"
)
SUCCESS_LIKE_STATUSES = {"SUCCESS", "PASS"}
ALARM_SUMMARY_SCHEMA_VERSION = "trust_health_alarm_summary.v0"
ALARM_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def _alarm_explanation(
    *,
    kind: str,
    component: str,
    severity: str,
    message: str,
    action: str,
    raw_alarm: Any = None,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    explanation: dict[str, Any] = {
        "kind": kind,
        "component": component,
        "severity": severity,
        "message": message,
        "action": action,
    }
    if raw_alarm is not None:
        explanation["raw_alarm"] = str(raw_alarm)
    if evidence_path:
        explanation["evidence_path"] = evidence_path
    return explanation


def _component_label(component: str) -> str:
    return {
        "trust_governance": "Trust Governance Check",
        "supersession_edges": "Supersession Edge Health",
        "runtime": "Runtime",
        "scheduler": "Trust Health scheduler",
        "supersession_scheduler": "Supersession scheduler",
    }.get(component, component.replace("_", " ").title())


def _raw_alarm_kind(raw_alarm: Any) -> str:
    text = str(raw_alarm or "").lower()
    if "timeout_after_seconds" in text or "timed out" in text or "timeout" in text:
        return "monitor_timeout"
    if "missing_report" in text or "report_missing" in text or "no_report" in text:
        return "missing_report"
    if (
        "accuracy" in text
        or "threshold" in text
        or "false_governance" in text
        or "false_abstention" in text
        or "stale_superseded_suppression" in text
        or "evidence_completeness" in text
        or "no_known_authority_correctness" in text
    ):
        return "quality_alarm"
    return "unknown_alarm"


def _raw_alarm_explanation(component: str, raw_alarm: Any, evidence_path: str | None) -> dict[str, Any]:
    label = _component_label(component)
    kind = _raw_alarm_kind(raw_alarm)
    if kind == "monitor_timeout":
        return _alarm_explanation(
            kind=kind,
            component=component,
            severity="warning",
            message=f"{label} did not finish before its timeout.",
            action="Rerun the monitor once; if it repeats, inspect monitor runtime/log performance before trusting freshness.",
            raw_alarm=raw_alarm,
            evidence_path=evidence_path,
        )
    if kind == "missing_report":
        return _alarm_explanation(
            kind=kind,
            component=component,
            severity="warning",
            message=f"{label} expected an evidence report that is missing or unavailable.",
            action="Check the monitor report path and rerun the monitor so Pith has inspectable evidence.",
            raw_alarm=raw_alarm,
            evidence_path=evidence_path,
        )
    if kind == "quality_alarm":
        return _alarm_explanation(
            kind=kind,
            component=component,
            severity="critical",
            message=f"{label} completed but reported a trust-quality threshold failure.",
            action="Open the evidence report and repair the failing trust-health metric before relying on governed answers.",
            raw_alarm=raw_alarm,
            evidence_path=evidence_path,
        )
    return _alarm_explanation(
        kind=kind,
        component=component,
        severity="warning",
        message=f"{label} reported an unclassified alarm.",
        action="Open the evidence report or compact monitor log and classify the alarm before relying on governed answers.",
        raw_alarm=raw_alarm,
        evidence_path=evidence_path,
    )


def _classify_component_explanations(component: str, component_status: dict[str, Any]) -> list[dict[str, Any]]:
    latest_run = (
        component_status.get("latest_run")
        if isinstance(component_status.get("latest_run"), dict)
        else {}
    )
    evidence = (
        component_status.get("evidence")
        if isinstance(component_status.get("evidence"), dict)
        else {}
    )
    freshness = (
        component_status.get("freshness")
        if isinstance(component_status.get("freshness"), dict)
        else {}
    )
    status = str(component_status.get("status") or latest_run.get("status") or "UNKNOWN")
    evidence_path = (
        evidence.get("latest_report_path")
        or evidence.get("latest_markdown_path")
        or latest_run.get("report_path")
        or latest_run.get("markdown_path")
    )
    explanations = [
        _raw_alarm_explanation(component, raw_alarm, evidence_path)
        for raw_alarm in (latest_run.get("alarms") if isinstance(latest_run.get("alarms"), list) else [])
    ]
    if status == "ALARM" and evidence_path and evidence.get("full_report_available") is False:
        explanations.append(
            _alarm_explanation(
                kind="missing_report",
                component=component,
                severity="warning",
                message=f"{_component_label(component)} references an evidence report that is not available on disk.",
                action="Regenerate or recover the evidence report, then rerun pith trust-health.",
                evidence_path=str(evidence_path),
            )
        )
    if status == "NO_EVIDENCE":
        explanations.append(
            _alarm_explanation(
                kind="missing_evidence",
                component=component,
                severity="warning",
                message=f"{_component_label(component)} has no compact monitor evidence yet.",
                action="Run the Trust Health monitors once so Pith has current evidence to evaluate.",
                evidence_path=str(evidence_path) if evidence_path else None,
            )
        )
    if freshness.get("status") in {"stale_warning", "stale_critical"}:
        explanations.append(
            _alarm_explanation(
                kind="stale_evidence",
                component=component,
                severity="critical" if freshness.get("status") == "stale_critical" else "warning",
                message=f"{_component_label(component)} evidence is {freshness.get('status')}.",
                action="Refresh Trust Health evidence, then rerun pith trust-health.",
                evidence_path=str(evidence_path) if evidence_path else None,
            )
        )
    if status not in SUCCESS_LIKE_STATUSES and status not in {
        "ALARM",
        "NO_EVIDENCE",
        "STALE_WARNING",
        "STALE_CRITICAL",
        "OK_WITH_RISK_COUNTS",
    } and not explanations:
        explanations.append(
            _alarm_explanation(
                kind="unknown_alarm",
                component=component,
                severity="warning",
                message=f"{_component_label(component)} returned status {status}.",
                action="Inspect the compact monitor log and evidence report before relying on governed answers.",
                evidence_path=str(evidence_path) if evidence_path else None,
            )
        )
    return explanations


def _classify_runtime_explanations(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    if runtime.get("enabled") is False:
        return []
    if runtime.get("available") is False:
        return [
            _alarm_explanation(
                kind="runtime_issue",
                component="runtime",
                severity="warning",
                message="The local Pith runtime health endpoint is unavailable.",
                action="Restart the local Pith server, then rerun pith health --json and pith trust-health.",
                raw_alarm=runtime.get("error"),
            )
        ]
    if runtime.get("runtime_matches_canonical") is False:
        return [
            _alarm_explanation(
                kind="runtime_issue",
                component="runtime",
                severity="warning",
                message="The local Pith runtime does not match the canonical installed code.",
                action="Deploy or restart the installed runtime, then rerun pith trust-health.",
            )
        ]
    return []


def _classify_scheduler_explanations(scheduler: dict[str, Any], component: str) -> list[dict[str, Any]]:
    if scheduler.get("enabled") is False:
        return []
    if scheduler.get("status") not in {None, "available", "disabled"}:
        return [
            _alarm_explanation(
                kind="scheduler_issue",
                component=component,
                severity="warning",
                message=f"{_component_label(component)} is not reporting an available scheduler.",
                action="Repair or reload the Trust Health scheduler so measurements keep updating.",
                raw_alarm=scheduler.get("error") or scheduler.get("status"),
            )
        ]
    return []


def collect_alarm_explanations(*objects: dict[str, Any]) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for value in objects:
        if not isinstance(value, dict):
            continue
        candidates = value.get("alarm_explanations")
        if isinstance(candidates, list):
            explanations.extend(item for item in candidates if isinstance(item, dict))
    return explanations


def summarize_alarm_explanations(explanations: list[dict[str, Any]]) -> dict[str, Any]:
    severities = [str(item.get("severity") or "warning") for item in explanations]
    highest = "none"
    if severities:
        highest = max(severities, key=lambda value: ALARM_SEVERITY_RANK.get(value, 1))
    kinds = sorted({str(item.get("kind") or "unknown_alarm") for item in explanations})
    return {
        "schema_version": ALARM_SUMMARY_SCHEMA_VERSION,
        "count": len(explanations),
        "highest_severity": highest,
        "kinds": kinds,
        "items": explanations,
    }


def format_alarm_summary_lines(
    status: dict[str, Any],
    *,
    indent: str = "",
    max_items: int = 3,
) -> list[str]:
    summary = status.get("alarm_summary") if isinstance(status.get("alarm_summary"), dict) else {}
    items = summary.get("items") if isinstance(summary.get("items"), list) else []
    clean_items = [item for item in items if isinstance(item, dict)]
    if not clean_items:
        return []
    lines = [
        f"{indent}Alarm detail: {summary.get('count', len(clean_items))} issue(s), highest severity {summary.get('highest_severity', 'warning')}"
    ]
    for item in clean_items[:max_items]:
        label = _component_label(str(item.get("component") or "unknown"))
        kind = item.get("kind") or "unknown_alarm"
        message = item.get("message") or "Unclassified Trust Health issue."
        action = item.get("action") or "Inspect Trust Health evidence."
        lines.append(f"{indent}  - {label}: {kind} - {message}")
        lines.append(f"{indent}    Action: {action}")
    remaining = len(clean_items) - max_items
    if remaining > 0:
        lines.append(f"{indent}  - {remaining} more issue(s) in JSON output.")
    return lines


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def read_compact_entries(
    path: Path,
    max_bytes: int = DEFAULT_MAX_LOG_BYTES,
) -> tuple[list[dict[str, Any]], int, bool]:
    if not path.exists():
        return [], 0, False

    try:
        size = path.stat().st_size
        tail_limited = size > max_bytes
        with path.open("rb") as handle:
            if tail_limited:
                start = max(0, size - max_bytes)
                if start > 0:
                    handle.seek(start - 1)
                    previous = handle.read(1)
                    data = handle.read(max_bytes)
                    lines = data.splitlines()
                    if previous != b"\n" and lines:
                        lines = lines[1:]
                else:
                    lines = handle.read(max_bytes).splitlines()
            else:
                lines = handle.read().splitlines()
    except OSError:
        return [], 0, False

    entries: list[dict[str, Any]] = []
    corrupt_count = 0
    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            corrupt_count += 1
            continue
        if isinstance(value, dict):
            entries.append(value)
        else:
            corrupt_count += 1
    return entries, corrupt_count, tail_limited


def select_latest_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    latest_entry: dict[str, Any] | None = None
    latest_timestamp: datetime | None = None
    for entry in entries:
        timestamp = parse_timestamp(entry.get("timestamp"))
        if timestamp is None:
            continue
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_entry = entry

    return latest_entry if latest_entry is not None else entries[-1]


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def normalize_latest_run(entry: dict[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return {
            "timestamp": None,
            "status": "NO_EVIDENCE",
            "alarms": [],
            "metrics": {},
            "report_path": None,
            "row_count": 0,
        }

    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    latency = metrics.get("latency_ms") if isinstance(metrics.get("latency_ms"), dict) else {}
    normalized_metrics = {
        "governed_accuracy": _number_or_none(metrics.get("governed_accuracy")),
        "control_accuracy": _number_or_none(metrics.get("control_accuracy")),
        "absolute_accuracy_lift": _number_or_none(metrics.get("absolute_accuracy_lift")),
        "stale_superseded_suppression": _number_or_none(
            metrics.get("stale_superseded_suppression")
        ),
        "false_governance_rate": _number_or_none(metrics.get("false_governance_rate")),
        "false_abstention_rate": _number_or_none(metrics.get("false_abstention_rate")),
        "evidence_completeness": _number_or_none(metrics.get("evidence_completeness")),
        "no_known_authority_correctness": _number_or_none(
            metrics.get("no_known_authority_correctness")
        ),
        "latency_p95_ms": _number_or_none(latency.get("p95")),
    }
    scorecard = entry.get("scorecard") if isinstance(entry.get("scorecard"), dict) else {}
    alarms = entry.get("alarms") if isinstance(entry.get("alarms"), list) else []
    return {
        "timestamp": entry.get("timestamp"),
        "status": entry.get("status", "UNKNOWN"),
        "alarms": alarms,
        "metrics": normalized_metrics,
        "report_path": scorecard.get("report_path"),
        "row_count": scorecard.get("row_count", 0),
    }


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def normalize_supersession_edge_run(entry: dict[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return {
            "timestamp": None,
            "status": "NO_EVIDENCE",
            "alarms": [],
            "summary": {},
            "risk_counts": {},
            "report_path": None,
            "markdown_path": None,
            "claim_boundary": None,
        }

    audit = entry.get("audit") if isinstance(entry.get("audit"), dict) else {}
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    if not summary:
        summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    edge_kind_counts = summary.get("edge_kind_counts") if isinstance(summary.get("edge_kind_counts"), dict) else {}
    state_flag_counts = summary.get("state_flag_counts") if isinstance(summary.get("state_flag_counts"), dict) else {}
    dispositions = (
        summary.get("recommended_disposition_counts")
        if isinstance(summary.get("recommended_disposition_counts"), dict)
        else {}
    )
    alarms = entry.get("alarms") if isinstance(entry.get("alarms"), list) else []
    return {
        "timestamp": entry.get("timestamp"),
        "status": entry.get("status", audit.get("status", "UNKNOWN")),
        "alarms": alarms,
        "summary": summary,
        "risk_counts": {
            "total_edges": _int_or_zero(summary.get("total_edges")),
            "missing_identity_edges": _int_or_zero(summary.get("missing_identity_edges")),
            "unsafe_ambiguous_edges": _int_or_zero(summary.get("unsafe_ambiguous_edges")),
            "answer_governance_risk": _int_or_zero(state_flag_counts.get("answer_governance_risk")),
            "reflection_duplicate_cross_subject": _int_or_zero(
                edge_kind_counts.get("reflection_duplicate_cross_subject")
            ),
            "manual_review_required": _int_or_zero(dispositions.get("manual_review_required")),
        },
        "report_path": audit.get("report_path") or entry.get("report_path"),
        "markdown_path": audit.get("markdown_path") or entry.get("markdown_path"),
        "claim_boundary": entry.get("claim_boundary") or audit.get("claim_boundary"),
    }


def summarize_history(
    entries: list[dict[str, Any]],
    history_limit: int,
    tail_limited: bool = False,
    corrupt_line_count: int = 0,
) -> dict[str, Any]:
    sorted_entries = sorted(
        entries,
        key=lambda item: parse_timestamp(item.get("timestamp")) or datetime.min.replace(tzinfo=UTC),
    )
    timestamps = [parse_timestamp(entry.get("timestamp")) for entry in sorted_entries]
    valid_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    recent_entries = sorted_entries[-history_limit:]
    return {
        "scope": HISTORY_SCOPE_LABEL,
        "run_count": len(entries),
        "success_count": sum(
            1 for entry in entries if entry.get("status") in SUCCESS_LIKE_STATUSES
        ),
        "alarm_count": sum(1 for entry in entries if entry.get("alarms")),
        "corrupt_line_count": corrupt_line_count,
        "tail_limited": tail_limited,
        "window_start": valid_timestamps[0].isoformat() if valid_timestamps else None,
        "window_end": valid_timestamps[-1].isoformat() if valid_timestamps else None,
        "entries": [
            {
                "timestamp": entry.get("timestamp"),
                "status": entry.get("status", "UNKNOWN"),
                "alarm_count": len(entry.get("alarms") or []),
            }
            for entry in recent_entries
        ],
    }


def classify_freshness(
    latest_timestamp: str | None,
    now: datetime,
    warn_after_seconds: int,
    critical_after_seconds: int,
) -> dict[str, Any]:
    parsed = parse_timestamp(latest_timestamp)
    if parsed is None:
        return {
            "status": "unknown",
            "latest_age_seconds": None,
            "warn_after_seconds": warn_after_seconds,
            "critical_after_seconds": critical_after_seconds,
        }

    age_seconds = max(0, int((now.astimezone(UTC) - parsed).total_seconds()))
    if age_seconds >= critical_after_seconds:
        status = "stale_critical"
    elif age_seconds >= warn_after_seconds:
        status = "stale_warning"
    else:
        status = "fresh"
    return {
        "status": status,
        "latest_age_seconds": age_seconds,
        "warn_after_seconds": warn_after_seconds,
        "critical_after_seconds": critical_after_seconds,
    }


def resolve_report_path(entry: dict[str, Any] | None, reports_dir: Path) -> tuple[str | None, bool]:
    scorecard = entry.get("scorecard") if isinstance(entry, dict) else None
    report_path = scorecard.get("report_path") if isinstance(scorecard, dict) else None
    if isinstance(report_path, str) and report_path:
        path = Path(report_path).expanduser()
        return str(path), path.exists()

    try:
        candidates = sorted(
            reports_dir.glob("trust-governance-effectiveness-scorecard-*.json"),
            key=lambda path: path.name,
        )
    except OSError:
        candidates = []
    if not candidates:
        return None, False
    latest = candidates[-1]
    return str(latest), latest.exists()


def read_runtime_health(url: str, timeout_seconds: float, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "available": False, "status": "disabled"}
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "enabled": True,
            "available": False,
            "status": "unavailable",
            "error": str(exc),
            "url": url,
        }
    return {
        "enabled": True,
        "available": True,
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "runtime_matches_canonical": payload.get("runtime_matches_canonical"),
        "runtime_git_head": payload.get("runtime_git_head"),
        "canonical_head": payload.get("canonical_head"),
        "url": url,
    }


def parse_launchctl_output(text: str, label: str) -> dict[str, Any]:
    state_match = re.search(r"^\s*state = (?P<state>.+)$", text, re.MULTILINE)
    runs_match = re.search(r"^\s*runs = (?P<runs>\d+)$", text, re.MULTILINE)
    exit_match = re.search(r"^\s*last exit code = (?P<code>-?\d+)$", text, re.MULTILINE)
    hour_match = re.search(r'"Hour"\s*=>\s*(?P<hour>\d+)', text)
    minute_match = re.search(r'"Minute"\s*=>\s*(?P<minute>\d+)', text)
    schedule: dict[str, int] = {}
    if hour_match:
        schedule["Hour"] = int(hour_match.group("hour"))
    if minute_match:
        schedule["Minute"] = int(minute_match.group("minute"))
    return {
        "label": label,
        "available": True,
        "state": state_match.group("state").strip() if state_match else None,
        "runs": int(runs_match.group("runs")) if runs_match else None,
        "last_exit_code": int(exit_match.group("code")) if exit_match else None,
        "schedule": schedule,
    }


def read_scheduler_status(label: str, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "available": False, "label": label, "status": "disabled"}
    if platform.system() != "Darwin":
        return {
            "enabled": True,
            "available": False,
            "label": label,
            "status": "unsupported_platform",
        }
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            timeout=DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "enabled": True,
            "available": False,
            "label": label,
            "status": "unavailable",
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "enabled": True,
            "available": False,
            "label": label,
            "status": "unavailable",
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    parsed = parse_launchctl_output(result.stdout, label)
    parsed["enabled"] = True
    parsed["status"] = "available"
    return parsed


def _aggregate_status(latest_run: dict[str, Any], freshness: dict[str, Any]) -> str:
    if latest_run["status"] == "NO_EVIDENCE":
        return "NO_EVIDENCE"
    if latest_run.get("alarms") or latest_run.get("status") not in SUCCESS_LIKE_STATUSES:
        return "ALARM"
    if freshness.get("status") == "stale_critical":
        return "STALE_CRITICAL"
    if freshness.get("status") == "stale_warning":
        return "STALE_WARNING"
    return "SUCCESS"


def _has_supersession_risk_counts(risk_counts: dict[str, Any]) -> bool:
    risk_keys = {
        "missing_identity_edges",
        "unsafe_ambiguous_edges",
        "answer_governance_risk",
        "reflection_duplicate_cross_subject",
        "manual_review_required",
    }
    return any(_int_or_zero(risk_counts.get(key)) > 0 for key in risk_keys)


def build_supersession_edge_status(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    compact_log_path = Path(
        getattr(args, "supersession_compact_log_path", DEFAULT_SUPERSESSION_COMPACT_LOG_PATH)
    ).expanduser()
    entries, corrupt_line_count, tail_limited = read_compact_entries(
        compact_log_path,
        getattr(args, "max_log_bytes", DEFAULT_MAX_LOG_BYTES),
    )
    latest_entry = select_latest_entry(entries)
    latest_run = normalize_supersession_edge_run(latest_entry)
    freshness = classify_freshness(
        latest_run.get("timestamp"),
        now,
        getattr(args, "warn_after_seconds", DEFAULT_WARN_AFTER_SECONDS),
        getattr(args, "critical_after_seconds", DEFAULT_CRITICAL_AFTER_SECONDS),
    )
    status = _aggregate_status(latest_run, freshness)
    risk_counts = latest_run.get("risk_counts") if isinstance(latest_run.get("risk_counts"), dict) else {}
    if status == "SUCCESS" and _has_supersession_risk_counts(risk_counts):
        status = "OK_WITH_RISK_COUNTS"
    return {
        "status": status,
        "latest_run": latest_run,
        "freshness": freshness,
        "scheduler": read_scheduler_status(
            DEFAULT_SUPERSESSION_LAUNCHD_LABEL,
            enabled=not getattr(args, "no_supersession_scheduler", False),
        ),
        "evidence": {
            "compact_log_path": str(compact_log_path),
            "latest_report_path": latest_run.get("report_path"),
            "latest_markdown_path": latest_run.get("markdown_path"),
            "corrupt_line_count": corrupt_line_count,
            "tail_limited": tail_limited,
            "max_log_bytes": getattr(args, "max_log_bytes", DEFAULT_MAX_LOG_BYTES),
        },
        "history": summarize_history(
            entries,
            getattr(args, "history_limit", DEFAULT_HISTORY_LIMIT),
            tail_limited=tail_limited,
            corrupt_line_count=corrupt_line_count,
        ),
    }


def _aggregate_user_status(trust_status: str, supersession_status: str) -> str:
    statuses = {trust_status, supersession_status}
    if "ALARM" in statuses:
        return "ALARM"
    if "STALE_CRITICAL" in statuses:
        return "STALE_CRITICAL"
    if "STALE_WARNING" in statuses:
        return "STALE_WARNING"
    if statuses == {"NO_EVIDENCE"}:
        return "NO_EVIDENCE"
    if "OK_WITH_RISK_COUNTS" in statuses:
        return "OK_WITH_RISK_COUNTS"
    if "SUCCESS" in statuses:
        return "SUCCESS"
    return trust_status or supersession_status or "UNKNOWN"


def build_user_next_steps(
    trust_governance: dict[str, Any],
    supersession_edges: dict[str, Any],
    runtime: dict[str, Any],
    scheduler: dict[str, Any],
) -> dict[str, Any]:
    trust_status = str(trust_governance.get("status") or "UNKNOWN")
    trust_freshness = (
        trust_governance.get("freshness")
        if isinstance(trust_governance.get("freshness"), dict)
        else {}
    )
    supersession_status = str(supersession_edges.get("status") or "UNKNOWN")
    supersession_latest = (
        supersession_edges.get("latest_run")
        if isinstance(supersession_edges.get("latest_run"), dict)
        else {}
    )
    supersession_freshness = (
        supersession_edges.get("freshness")
        if isinstance(supersession_edges.get("freshness"), dict)
        else {}
    )
    supersession_risks = (
        supersession_latest.get("risk_counts")
        if isinstance(supersession_latest.get("risk_counts"), dict)
        else {}
    )

    actions: list[str] = []
    supersession_scheduler = (
        supersession_edges.get("scheduler")
        if isinstance(supersession_edges.get("scheduler"), dict)
        else {}
    )
    for explanation in collect_alarm_explanations(
        trust_governance,
        supersession_edges,
        runtime,
        scheduler,
        supersession_scheduler,
    ):
        action = str(explanation.get("action") or "").strip()
        if action and action not in actions:
            actions.append(action)
    if trust_status == "NO_EVIDENCE" or supersession_status == "NO_EVIDENCE":
        action = "Run the Trust Health monitors once so Pith has current evidence to evaluate."
        if action not in actions:
            actions.append(action)
    if trust_status == "ALARM" or supersession_status == "ALARM":
        action = "Open the evidence report and resolve the listed alarms before relying on governed answers."
        if action not in actions:
            actions.append(action)
    if trust_freshness.get("status") in {"stale_warning", "stale_critical"} or supersession_freshness.get(
        "status"
    ) in {"stale_warning", "stale_critical"}:
        action = "Refresh Trust Health evidence, then rerun pith trust-health."
        if action not in actions:
            actions.append(action)
    if runtime.get("enabled") is not False and (
        runtime.get("available") is False or runtime.get("runtime_matches_canonical") is False
    ):
        action = "Restart the local Pith server, then rerun pith health --json and pith trust-health."
        if action not in actions:
            actions.append(action)
    if scheduler.get("enabled") is not False and scheduler.get("status") not in {None, "available", "disabled"}:
        action = "Repair or reload the Trust Health scheduler so measurements keep updating."
        if action not in actions:
            actions.append(action)

    manual_review = _int_or_zero(supersession_risks.get("manual_review_required"))
    missing_identity = _int_or_zero(supersession_risks.get("missing_identity_edges"))
    cross_subject = _int_or_zero(supersession_risks.get("reflection_duplicate_cross_subject"))
    answer_risk = _int_or_zero(supersession_risks.get("answer_governance_risk"))
    if manual_review:
        actions.append(
            f"Review {manual_review} supersession edges that require a human decision before repair."
        )
    if missing_identity:
        actions.append(
            f"Repair missing identity evidence on {missing_identity} supersession edges so Pith can trust the chain."
        )
    if cross_subject:
        actions.append(
            f"Inspect {cross_subject} cross-subject duplicate edges before allowing them to govern answers."
        )
    if answer_risk and not any(
        count for count in (manual_review, missing_identity, cross_subject)
    ):
        actions.append(f"Inspect {answer_risk} answer-risk supersession edges in the latest evidence report.")

    if not actions:
        actions.append("No immediate action. Keep the scheduled monitors enabled and recheck after the next run.")

    return {
        "primary_user_action": actions[0],
        "user_next_steps": actions,
        "schema_version": "trust_health_user_next_steps.v0",
    }


def build_status(args: argparse.Namespace, now: datetime | None = None) -> dict[str, Any]:
    current_time = now or datetime.now(UTC)
    compact_log_path = Path(args.compact_log_path).expanduser()
    reports_dir = Path(args.reports_dir).expanduser()
    entries, corrupt_line_count, tail_limited = read_compact_entries(
        compact_log_path,
        args.max_log_bytes,
    )
    latest_entry = select_latest_entry(entries)
    latest_run = normalize_latest_run(latest_entry)
    report_path, full_report_available = resolve_report_path(latest_entry, reports_dir)
    latest_run["report_path"] = report_path
    freshness = classify_freshness(
        latest_run.get("timestamp"),
        current_time,
        args.warn_after_seconds,
        args.critical_after_seconds,
    )
    status = _aggregate_status(latest_run, freshness)
    runtime = read_runtime_health(
        args.health_url,
        DEFAULT_HEALTH_TIMEOUT_SECONDS,
        enabled=not args.no_runtime_health,
    )
    scheduler = read_scheduler_status(
        DEFAULT_LAUNCHD_LABEL,
        enabled=not args.no_scheduler,
    )
    supersession_edges = build_supersession_edge_status(args, current_time)
    trust_governance = {
        "status": status,
        "latest_run": latest_run,
        "history": summarize_history(
            entries,
            args.history_limit,
            tail_limited=tail_limited,
            corrupt_line_count=corrupt_line_count,
        ),
        "freshness": freshness,
        "scheduler": scheduler,
        "evidence": {
            "compact_log_path": str(compact_log_path),
            "latest_report_path": report_path,
            "full_report_available": full_report_available,
            "reports_dir": str(reports_dir),
            "corrupt_line_count": corrupt_line_count,
            "tail_limited": tail_limited,
            "max_log_bytes": args.max_log_bytes,
        },
    }
    trust_governance["alarm_explanations"] = _classify_component_explanations(
        "trust_governance",
        trust_governance,
    )
    supersession_edges["alarm_explanations"] = _classify_component_explanations(
        "supersession_edges",
        supersession_edges,
    )
    runtime["alarm_explanations"] = _classify_runtime_explanations(runtime)
    scheduler["alarm_explanations"] = _classify_scheduler_explanations(scheduler, "scheduler")
    supersession_scheduler = (
        supersession_edges.get("scheduler")
        if isinstance(supersession_edges.get("scheduler"), dict)
        else {}
    )
    supersession_scheduler["alarm_explanations"] = _classify_scheduler_explanations(
        supersession_scheduler,
        "supersession_scheduler",
    )
    all_alarm_explanations = collect_alarm_explanations(
        trust_governance,
        supersession_edges,
        runtime,
        scheduler,
        supersession_scheduler,
    )
    alarm_summary = summarize_alarm_explanations(all_alarm_explanations)
    user_actions = build_user_next_steps(
        trust_governance,
        supersession_edges,
        runtime,
        scheduler,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "overall_user_status": _aggregate_user_status(status, supersession_edges["status"]),
        "user_meaning": USER_MEANING,
        "primary_user_action": user_actions["primary_user_action"],
        "user_next_steps": user_actions["user_next_steps"],
        "user_action_schema_version": user_actions["schema_version"],
        "claim_boundary": CLAIM_BOUNDARY,
        "latest_run": latest_run,
        "history": trust_governance["history"],
        "freshness": freshness,
        "runtime": runtime,
        "scheduler": scheduler,
        "alarm_summary": alarm_summary,
        "trust_governance": trust_governance,
        "supersession_edges": supersession_edges,
        "evidence": {
            "compact_log_path": str(compact_log_path),
            "latest_report_path": report_path,
            "full_report_available": full_report_available,
            "reports_dir": str(reports_dir),
            "corrupt_line_count": corrupt_line_count,
            "tail_limited": tail_limited,
            "max_log_bytes": args.max_log_bytes,
        },
    }


def _fmt_number(value: Any) -> str:
    number = _number_or_none(value)
    return "?" if number is None else f"{float(number):.3f}"


def _fmt_int(value: Any) -> str:
    number = _number_or_none(value)
    return "?" if number is None else str(int(number))


def render_human(status: dict[str, Any]) -> str:
    latest = status.get("latest_run") or {}
    metrics = latest.get("metrics") or {}
    freshness = status.get("freshness") or {}
    evidence = status.get("evidence") or {}
    runtime = status.get("runtime") or {}
    scheduler = status.get("scheduler") or {}
    supersession = status.get("supersession_edges") if isinstance(status.get("supersession_edges"), dict) else {}
    supersession_latest = (
        supersession.get("latest_run") if isinstance(supersession.get("latest_run"), dict) else {}
    )
    supersession_risks = (
        supersession_latest.get("risk_counts")
        if isinstance(supersession_latest.get("risk_counts"), dict)
        else {}
    )
    supersession_freshness = (
        supersession.get("freshness") if isinstance(supersession.get("freshness"), dict) else {}
    )
    supersession_evidence = (
        supersession.get("evidence") if isinstance(supersession.get("evidence"), dict) else {}
    )
    supersession_scheduler = (
        supersession.get("scheduler") if isinstance(supersession.get("scheduler"), dict) else {}
    )
    lines = [
        f"Pith Trust Health: {status.get('overall_user_status', status.get('status', 'UNKNOWN'))}",
        "",
        f"Trust Governance Check: {status.get('status', 'UNKNOWN')}",
        f"  Latest run: {latest.get('timestamp') or '?'}",
        (
            "  Governed/control/lift: "
            f"{_fmt_number(metrics.get('governed_accuracy'))}  "
            f"/ {_fmt_number(metrics.get('control_accuracy'))}  "
            f"/ {_fmt_number(metrics.get('absolute_accuracy_lift'))}"
        ),
        f"  Alarms: {len(latest.get('alarms') or [])}",
        f"  Freshness: {freshness.get('status', 'unknown')}",
        f"  Evidence: {evidence.get('latest_report_path') or '?'}",
        "",
        f"Supersession Edge Health: {supersession.get('status', 'NO_EVIDENCE')}",
        f"  Latest run: {supersession_latest.get('timestamp') or '?'}",
        f"  Answer-risk edges: {_fmt_int(supersession_risks.get('answer_governance_risk'))}",
        f"  Missing identity evidence: {_fmt_int(supersession_risks.get('missing_identity_edges'))}",
        f"  Cross-subject duplicates: {_fmt_int(supersession_risks.get('reflection_duplicate_cross_subject'))}",
        f"  Manual review required: {_fmt_int(supersession_risks.get('manual_review_required'))}",
        f"  Alarms: {len(supersession_latest.get('alarms') or [])}",
        f"  Freshness: {supersession_freshness.get('status', 'unknown')}",
        f"  Evidence: {supersession_evidence.get('latest_report_path') or '?'}",
        "",
        (
            "Runtime: "
            f"{runtime.get('status', 'unknown')} "
            f"matches_canonical={runtime.get('runtime_matches_canonical', '?')}"
        ),
        (
            "Scheduler: "
            f"trust={scheduler.get('status', 'unknown')} "
            f"supersession={supersession_scheduler.get('status', 'unknown')}"
        ),
        f"Meaning: {status.get('user_meaning', USER_MEANING)}",
    ]
    lines.extend(format_alarm_summary_lines(status))
    next_steps = status.get("user_next_steps") if isinstance(status.get("user_next_steps"), list) else []
    if next_steps:
        lines.extend(["", "Next:"])
        lines.extend(f"  {index}. {step}" for index, step in enumerate(next_steps, start=1))
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _min_log_bytes(value: str) -> int:
    parsed = int(value)
    if parsed < 1024:
        raise argparse.ArgumentTypeError("must be >= 1024")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show local Pith Trust Health evidence.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--history-limit", type=_positive_int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--compact-log-path", default=str(DEFAULT_COMPACT_LOG_PATH))
    parser.add_argument("--supersession-compact-log-path", default=str(DEFAULT_SUPERSESSION_COMPACT_LOG_PATH))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    parser.add_argument(
        "--warn-after-seconds",
        type=_nonnegative_int,
        default=DEFAULT_WARN_AFTER_SECONDS,
    )
    parser.add_argument(
        "--critical-after-seconds",
        type=_nonnegative_int,
        default=DEFAULT_CRITICAL_AFTER_SECONDS,
    )
    parser.add_argument("--max-log-bytes", type=_min_log_bytes, default=DEFAULT_MAX_LOG_BYTES)
    parser.add_argument("--no-runtime-health", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--no-supersession-scheduler", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.critical_after_seconds < args.warn_after_seconds:
        parser.error("--critical-after-seconds must be >= --warn-after-seconds")
    status = build_status(args)
    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        print(render_human(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
