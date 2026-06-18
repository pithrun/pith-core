"""CLI helpers for `pith health`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 10.0


def resolve_base_url(args_base_url: str | None = None) -> str:
    if args_base_url:
        return args_base_url.rstrip("/")
    if env_url := os.environ.get("PITH_API_URL"):
        return env_url.rstrip("/")
    port = os.environ.get("PITH_PORT", "8000")
    return f"http://127.0.0.1:{port}"


def fetch_json(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{path} unreachable: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{path} timed out") from exc
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned non-object JSON")
    return data


def operational_ok(health: dict[str, Any], readyz: dict[str, Any]) -> bool:
    if health.get("service") != "pith" or readyz.get("service") != "pith":
        return False
    if health.get("status") == "unhealthy":
        return False
    return readyz.get("mode") == "ready"


def cognitive_warning(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if payload.get("startup_degraded") is True or payload.get("partial") is True:
        return True
    section_errors = payload.get("section_errors")
    return bool(section_errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pith health", description="Check Pith health")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--cognitive", action="store_true", help="Include fast cognitive health")
    parser.add_argument("--full", action="store_true", help="Include full cognitive health")
    parser.add_argument(
        "--fail-on-cognitive-warning",
        action="store_true",
        help="Exit 3 when cognitive health reports warnings",
    )
    parser.add_argument("--base-url", help="Override local Pith API base URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base_url = resolve_base_url(args.base_url)
    result: dict[str, Any] = {"base_url": base_url}
    exit_code = 0

    try:
        health = fetch_json(base_url, "/health", args.timeout)
        readyz = fetch_json(base_url, "/readyz", args.timeout)
        result["health"] = health
        result["readyz"] = readyz
        if not operational_ok(health, readyz):
            exit_code = 1
    except RuntimeError as exc:
        result["error"] = str(exc)
        exit_code = 1

    include_cognitive = args.cognitive or args.full
    if include_cognitive and "error" not in result:
        detail = "full" if args.full else "fast"
        try:
            cognitive = fetch_json(base_url, f"/pith_health?detail={detail}", args.timeout)
            result["cognitive"] = cognitive
            if args.fail_on_cognitive_warning and cognitive_warning(cognitive):
                exit_code = 3
        except RuntimeError as exc:
            result["cognitive_error"] = str(exc)
            if args.fail_on_cognitive_warning:
                exit_code = 3

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if "error" in result:
            print(f"Pith health: ERROR - {result['error']}")
        else:
            health = result.get("health", {})
            readyz = result.get("readyz", {})
            status = "OK" if exit_code == 0 else "DEGRADED"
            print(f"Pith health: {status}")
            print(f"  service: {health.get('service', 'unknown')}")
            print(f"  health:  {health.get('status', 'unknown')}")
            print(f"  ready:   {readyz.get('mode', 'unknown')}")
            if "cognitive" in result:
                cognitive = result["cognitive"]
                print(f"  cognitive: {cognitive.get('status', 'unknown')}")
                if "health_score" in cognitive:
                    print(f"  health_score: {cognitive['health_score']}")
            if "cognitive_error" in result:
                print(f"  cognitive: ERROR - {result['cognitive_error']}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
