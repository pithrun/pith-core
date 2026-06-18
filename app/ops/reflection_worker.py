"""Standalone worker process for full reflection runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.ops.reflection_runner import ReflectionRunRequest, run_reflection


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Pith reflection pass out of process")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    parser.add_argument("--source", default="worker")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_reflection(
        ReflectionRunRequest(
            mode=args.mode,
            source=args.source,
            require_ready=False,
            allow_degraded=True,
            timeout_seconds=args.timeout_seconds,
        )
    )
    payload = result.response_fields
    parent_run_id = os.environ.get("PITH_REFLECTION_WORKER_RUN_ID")
    if parent_run_id:
        payload["worker_run_id"] = payload.get("run_id")
        payload["run_id"] = parent_run_id
    if result.summary is not None:
        summary = result.summary.model_dump() if hasattr(result.summary, "model_dump") else result.summary
        payload["summary"] = summary
    _write_output(args.output, payload)
    return 0 if result.status not in {"failed", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
