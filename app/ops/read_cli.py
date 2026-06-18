"""Human-friendly read-only CLI commands for Pith."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT = 10.0
SEARCH_TIMEOUT = 30.0
REDACTION_PATTERNS = (
    (re.compile(r"(PITH_API_KEY|BRAIN_API_KEY|API_KEY)=(\S+)"), r"\1=<redacted>"),
    (re.compile(r"(X-API-Key:\s*)(\S+)", re.IGNORECASE), r"\1<redacted>"),
)


def resolve_base_url(args_base_url: str | None = None) -> str:
    if args_base_url:
        return args_base_url.rstrip("/")
    if env_url := os.environ.get("PITH_API_URL"):
        return env_url.rstrip("/")
    port = os.environ.get("PITH_PORT", "8000")
    return f"http://127.0.0.1:{port}"


def _env_files() -> list[Path]:
    paths: list[Path] = []
    if pith_home := os.environ.get("PITH_HOME"):
        paths.append(Path(pith_home) / ".env")
    paths.append(Path.home() / ".pith" / ".env")
    return paths


def resolve_api_key() -> str:
    if key := os.environ.get("PITH_API_KEY", ""):
        return key
    for env_file in _env_files():
        if not env_file.exists():
            continue
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("#"):
                continue
            if line.startswith("PITH_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def fetch_json(
    base_url: str,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError("PITH_API_KEY unavailable for read command")
    headers = {"Accept": "application/json", "X-API-Key": api_key}
    data = None
    url = f"{base_url}{path}"
    if method == "GET" and payload:
        url = f"{url}?{urllib.parse.urlencode(payload)}"
    elif method == "POST":
        data = json.dumps(payload or {}).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {body[:300]}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{path} timed out after {timeout:g}s") from exc
    except socket.timeout as exc:
        raise RuntimeError(f"{path} timed out after {timeout:g}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise RuntimeError(f"{path} timed out after {timeout:g}s") from exc
        raise RuntimeError(f"{path} unreachable: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} returned non-JSON response") from exc


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _truncate(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def cmd_search(args: argparse.Namespace) -> int:
    query = " ".join(args.query).strip()
    payload = {"query": query, "max_results": args.limit}
    if args.since:
        payload["since"] = args.since
    if args.until:
        payload["until"] = args.until
    if args.time_field:
        payload["time_field"] = args.time_field
    data = fetch_json(resolve_base_url(args.base_url), "POST", "/pith_search", payload=payload, timeout=args.timeout)
    if args.json:
        _print_json(data)
        return 0
    results = data.get("results", []) if isinstance(data, dict) else []
    print(f"Search results for: {query}")
    if not results:
        print("  No results")
        return 0
    for idx, item in enumerate(results, start=1):
        print(
            f"{idx}. {item.get('concept_id', item.get('id', 'unknown'))} "
            f"score={_fmt_float(item.get('relevance_score'))} "
            f"conf={_fmt_float(item.get('confidence'))} "
            f"ka={item.get('knowledge_area', 'unknown')}"
        )
        print(f"   {_truncate(item.get('summary'), args.summary_chars)}")
    return 0


def cmd_concept_get(args: argparse.Namespace) -> int:
    payload = {"concept_id": args.concept_id, "version": args.version}
    data = fetch_json(resolve_base_url(args.base_url), "GET", "/pith_get_concept", payload=payload, timeout=args.timeout)
    if args.json:
        _print_json(data)
        return 0
    print(f"Concept: {data.get('id', args.concept_id)}")
    print(f"  type:       {data.get('concept_type', 'unknown')}")
    print(f"  ka:         {data.get('knowledge_area', 'unknown')}")
    print(f"  confidence: {_fmt_float(data.get('confidence'))}")
    print(f"  maturity:   {data.get('maturity', 'unknown')}")
    print(f"  status:     {data.get('status', 'unknown')}")
    print(f"  summary:    {_truncate(data.get('summary'), args.summary_chars)}")
    evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
    print(f"  evidence:   {len(evidence)} item(s)")
    return 0


def cmd_orient(args: argparse.Namespace) -> int:
    payload = {"time_window": args.window, "include_workstreams": args.workstreams}
    data = fetch_json(resolve_base_url(args.base_url), "GET", "/pith_orient", payload=payload, timeout=args.timeout)
    if args.json:
        _print_json(data)
        return 0
    print(f"Orientation: {args.window}")
    print(f"  generated_at: {data.get('generated_at', 'unknown')}")
    where_been = data.get("where_been", {}) if isinstance(data, dict) else {}
    concepts = where_been.get("concepts_created", []) if isinstance(where_been, dict) else []
    print(f"  recent concepts: {len(concepts)}")
    for item in concepts[: args.limit]:
        print(f"  - {item.get('concept_id', 'unknown')}: {_truncate(item.get('summary'), args.summary_chars)}")
    where_going = data.get("where_going", {}) if isinstance(data, dict) else {}
    if isinstance(where_going, dict):
        next_actions = where_going.get("next_actions") or where_going.get("recommendations") or []
        if next_actions:
            print("  next:")
            for action in next_actions[: args.limit]:
                print(f"  - {_truncate(action, args.summary_chars)}")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    payload = {"limit": args.limit}
    if args.status:
        payload["status"] = args.status
    if args.since:
        payload["since"] = args.since
    data = fetch_json(resolve_base_url(args.base_url), "GET", "/sessions_list", payload=payload, timeout=args.timeout)
    if args.json:
        _print_json(data)
        return 0
    sessions = data if isinstance(data, list) else data.get("sessions", [])
    print(f"Sessions: {len(sessions)}")
    for item in sessions:
        print(
            f"- {item.get('id', item.get('session_id', 'unknown'))} "
            f"{item.get('status', 'unknown')} "
            f"started={item.get('started_at', 'unknown')}"
        )
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    routes = {
        "dashboard": ("GET", "/metrics/dashboard", {"since": args.since} if args.since else {}),
        "summary": ("GET", "/metrics/summary", {"days": args.days}),
        "health-trend": ("GET", "/metrics/health_trend", {"days": args.days}),
        "bg-tasks": ("GET", "/metrics/bg_tasks", {"since": args.since} if args.since else {}),
    }
    method, path, payload = routes[args.kind]
    data = fetch_json(resolve_base_url(args.base_url), method, path, payload=payload, timeout=args.timeout)
    if args.json:
        _print_json(data)
        return 0
    print(f"Metrics: {args.kind}")
    if args.kind == "dashboard":
        print(f"  period: {data.get('period', 'unknown')}")
        for key in ("conversation_turn_latency_ms", "retrieval_search_latency_ms"):
            if isinstance(data.get(key), dict):
                block = data[key]
                print(f"  {key}: p95={block.get('p95', 'n/a')} count={block.get('count', 'n/a')}")
        print(f"  cascade_alert: {data.get('cascade_alert', False)}")
        print(f"  circuit_breaker_alert: {data.get('circuit_breaker_alert', False)}")
    elif args.kind == "summary":
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        for name, block in list(metrics.items())[: args.limit]:
            print(f"  {name}: count={block.get('count', 'n/a')} p95={block.get('p95', 'n/a')}")
    elif args.kind == "health-trend":
        trend = data.get("trend", data if isinstance(data, list) else [])
        for item in trend[: args.limit]:
            print(f"  {item}")
    else:
        items = list(data.items())[: args.limit] if isinstance(data, dict) else []
        for name, block in items:
            print(f"  {name}: {block}")
    return 0


def _log_paths() -> dict[str, Path]:
    pith_home = Path(os.environ.get("PITH_HOME", str(Path.home() / ".pith")))
    return {
        "pith": pith_home / "logs" / "pith.log",
        "err": pith_home / "logs" / "pith.err",
    }


def _redact(line: str) -> str:
    result = line.rstrip("\n")
    for pattern, replacement in REDACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return [f"<missing: {path}>"]
    return [_redact(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]]


def cmd_logs_snapshot(args: argparse.Namespace) -> int:
    paths = _log_paths()
    selected = paths.keys() if args.file == "both" else [args.file]
    data = {name: _tail(paths[name], args.lines) for name in selected}
    if args.json:
        _print_json(data)
        return 0
    for name, lines in data.items():
        print(f"== {paths[name]} ==")
        for line in lines:
            print(line)
    return 0


def add_common(parser: argparse.ArgumentParser, *, timeout: float = DEFAULT_TIMEOUT) -> None:
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=timeout)
    parser.add_argument("--json", action="store_true", help="Print raw JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pith", description="Pith read commands")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Search concepts")
    add_common(search, timeout=SEARCH_TIMEOUT)
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--summary-chars", type=int, default=140)
    search.add_argument("--from", dest="since", help="Filter results from YYYY-MM-DD or ISO timestamp")
    search.add_argument("--to", dest="until", help="Filter results through YYYY-MM-DD or ISO timestamp")
    search.add_argument(
        "--time-field",
        choices=["created_at", "valid_from", "original_date", "content_updated_at"],
        default=None,
        help="Temporal field to filter when --from/--to is supplied",
    )
    search.set_defaults(func=cmd_search)

    concept = sub.add_parser("concept", help="Read concepts")
    concept_sub = concept.add_subparsers(dest="concept_command", required=True)
    concept_get = concept_sub.add_parser("get", help="Get a concept by id")
    add_common(concept_get)
    concept_get.add_argument("concept_id")
    concept_get.add_argument("--version", default="latest")
    concept_get.add_argument("--summary-chars", type=int, default=180)
    concept_get.set_defaults(func=cmd_concept_get)

    orient = sub.add_parser("orient", help="Show present-moment orientation")
    add_common(orient)
    orient.add_argument("--window", default="7_days", choices=["1_day", "7_days", "30_days", "all"])
    orient.add_argument("--limit", type=int, default=8)
    orient.add_argument("--summary-chars", type=int, default=100)
    orient.add_argument("--workstreams", action="store_true")
    orient.set_defaults(func=cmd_orient)

    sessions = sub.add_parser("sessions", help="List sessions")
    add_common(sessions)
    sessions.add_argument("--status")
    sessions.add_argument("--limit", type=int, default=10)
    sessions.add_argument("--since")
    sessions.set_defaults(func=cmd_sessions)

    metrics = sub.add_parser("metrics", help="Show metrics snapshots")
    add_common(metrics)
    metrics.add_argument("kind", nargs="?", default="dashboard", choices=["dashboard", "summary", "health-trend", "bg-tasks"])
    metrics.add_argument("--days", type=int, default=7)
    metrics.add_argument("--since")
    metrics.add_argument("--limit", type=int, default=10)
    metrics.set_defaults(func=cmd_metrics)

    logs = sub.add_parser("logs", help="Read log snapshots")
    logs_sub = logs.add_subparsers(dest="logs_command", required=True)
    snapshot = logs_sub.add_parser("snapshot", help="Print a bounded log snapshot")
    snapshot.add_argument("--json", action="store_true")
    snapshot.add_argument("--file", choices=["pith", "err", "both"], default="both")
    snapshot.add_argument("--lines", type=int, default=80)
    snapshot.set_defaults(func=cmd_logs_snapshot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
