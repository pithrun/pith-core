"""CLI for Pith Import — import conversation history from ChatGPT, Claude, etc.

Usage:
  python -m app.ops.import_cli run --source chatgpt --file ~/Downloads/chatgpt-export.zip --confirm-llm-processing
  python -m app.ops.import_cli run --source claude --file ~/Downloads/claude-export.zip --confirm-llm-processing
  python -m app.ops.import_cli run --resume --confirm-llm-processing
  python -m app.ops.import_cli cancel
  python -m app.ops.import_cli status
  python -m app.ops.import_cli report
  python -m app.ops.import_cli log
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────

SCAN_PATHS = [
    "~/Downloads/*chatgpt*.zip",
    "~/Downloads/*claude*.zip",
    "~/Downloads/conversations.json",
    "~/Desktop/*chatgpt*.zip",
    "~/Desktop/*claude*.zip",
]

SOURCE_DETECT_HINTS = {
    "chatgpt": ["chatgpt", "openai"],
    "claude": ["claude", "anthropic"],
}

EXPORT_GUIDE = textwrap.dedent("""\
    HOW TO EXPORT YOUR CONVERSATIONS:

      ChatGPT:
        1. Go to chatgpt.com → Settings → Data Controls
        2. Click "Export Data" → Confirm
        3. Check your email for a download link (usually arrives in <5 min)
        4. Download and unzip → find conversations.json
        5. Run: pith import run --source chatgpt --file ~/Downloads/conversations.json --confirm-llm-processing

      Claude:
        1. Go to claude.ai → Settings → Privacy
        2. Click "Export Data" → Confirm
        3. Check your email for a download link
        4. Download the ZIP file
        5. Run: pith import run --source claude --file ~/Downloads/claude-export.zip --confirm-llm-processing
""")

LLM_DISCLOSURE = (
    "⚠️  Concept extraction sends conversation excerpts to your configured LLM\n"
    "    provider for analysis. Your conversations are not stored by the LLM provider.\n"
    "    Use --local-only for privacy-preserving mode (coming soon).\n"
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_data_dir() -> Path:
    """Resolve the active profile's data directory."""
    # Mirror app/profile.py logic
    if d := os.environ.get("PITH_DATA_DIR"):
        return Path(d)
    if p := os.environ.get("PITH_PROFILE"):
        return Path.home() / "pith-data" / p
    return Path.home() / "pith-data" / "default"


def _checkpoint_dir() -> Path:
    return _resolve_data_dir() / "import_checkpoints"


def _status_file() -> Path:
    return _resolve_data_dir() / "import_status.json"


def _log_file() -> Path:
    return _resolve_data_dir() / "logs" / "import.log"


def _save_status(status: dict) -> None:
    """Persist import status to disk for --status queries."""
    sf = _status_file()
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(status, indent=2, default=str))


def _load_status() -> dict | None:
    sf = _status_file()
    if sf.exists():
        return json.loads(sf.read_text())
    return None


def _json_requested(args) -> bool:
    return getattr(args, "json", False) is True


def _print_json(data: dict | list) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _auto_scan() -> list[Path]:
    """Scan common download locations for export files."""
    found = []
    for pattern in SCAN_PATHS:
        expanded = glob.glob(os.path.expanduser(pattern))
        for f in expanded:
            found.append(Path(f))
    # Sort by modification time, newest first
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def _detect_source(filename: str) -> str | None:
    """Guess source from filename."""
    lower = filename.lower()
    for source, hints in SOURCE_DETECT_HINTS.items():
        if any(h in lower for h in hints):
            return source
    return None


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Render a simple ASCII progress bar."""
    if total == 0:
        return "[" + " " * width + "]  0%"
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:>6.1%}"


def _cli_progress_callback(progress) -> None:
    """Print progress updates to terminal."""
    p = progress if isinstance(progress, dict) else progress.as_dict()
    processed = p.get("processed", 0)
    total = p.get("total", 0)
    failed = p.get("failed", 0)
    elapsed = p.get("elapsed_seconds", 0)

    bar = _progress_bar(processed, total)
    rate = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / rate if rate > 0 else 0

    # Overwrite line in-place
    status_parts = [
        f"\r  {bar}  {processed}/{total}",
        f"  {rate:.1f}/s",
        f"  ETA {eta:.0f}s" if eta > 0 else "",
        f"  ({failed} failed)" if failed > 0 else "",
    ]
    sys.stdout.write("".join(status_parts))
    sys.stdout.flush()

    # Also update status file for --status queries
    _save_status({
        "state": "running",
        "processed": processed,
        "total": total,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "updated_at": datetime.now(UTC).isoformat(),
    })


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_run(args) -> int:
    """Run the import pipeline."""
    from app.ops.import_pipeline import run_import_pipeline

    disclosure_shown = False

    if args.local_only:
        print("✗ Local-only extraction is not available yet.")
        print("  Re-run without --local-only and with --confirm-llm-processing to use your configured LLM provider.")
        return 2

    if not args.confirm_llm_processing:
        if not sys.stdin.isatty():
            print("✗ Import run requires explicit LLM-processing consent.")
            print("  Add --confirm-llm-processing to acknowledge provider-backed extraction.")
            return 2
        print(f"\n{LLM_DISCLOSURE}")
        disclosure_shown = True
        answer = input("Continue with provider-backed extraction? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Import cancelled.")
            return 1
        args.confirm_llm_processing = True

    file_path = args.file
    source = args.source
    resume = args.resume

    # ── Resume mode: no file/source needed ──
    if resume:
        cp_dir = _checkpoint_dir()
        if not cp_dir.exists() or not any(cp_dir.iterdir()):
            print("✗ No checkpoint found. Nothing to resume.")
            return 1
        print("↻ Resuming interrupted import from checkpoint...")
        # Load checkpoint to get source/file info
        for cp_file in sorted(cp_dir.glob("*.json"), reverse=True):
            try:
                cp_data = json.loads(cp_file.read_text())
                source = source or cp_data.get("source", "chatgpt")
                file_path = file_path or cp_data.get("file_path")
                break
            except (json.JSONDecodeError, KeyError):
                continue
        if not file_path:
            print("✗ Checkpoint found but missing file path. Provide --file explicitly.")
            return 1

    # ── Auto-scan if no file provided ──
    if not file_path and not resume:
        print("No --file specified. Scanning common locations...\n")
        found = _auto_scan()
        if not found:
            print("  No export files found in ~/Downloads or ~/Desktop.")
            print("  Run 'pith import --help' for export instructions.\n")
            return 1

        print(f"  Found {len(found)} file(s):\n")
        for i, f in enumerate(found[:5], 1):
            detected = _detect_source(f.name)
            tag = f" [{detected}]" if detected else ""
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {i}. {f.name}{tag}  ({size_mb:.1f} MB, {mtime})")

        print()
        try:
            choice = input("  Use which file? [1]: ").strip()
            idx = int(choice) - 1 if choice else 0
            if idx < 0 or idx >= len(found[:5]):
                print("  Invalid choice.")
                return 1
            file_path = str(found[idx])
            if not source:
                source = _detect_source(found[idx].name)
        except (ValueError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 1

    # ── Validate source ──
    if not source:
        # Try to detect from filename
        if file_path:
            source = _detect_source(Path(file_path).name)
        if not source:
            print("✗ Could not detect source. Specify --source chatgpt or --source claude")
            return 1

    if not file_path:
        print("✗ No file specified. Use --file or let auto-scan find it.")
        return 1

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        print(f"✗ File not found: {path}")
        return 1

    if not disclosure_shown:
        print(f"\n{LLM_DISCLOSURE}")

    # ── Run pipeline ──
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Source:  {source}")
    print(f"  File:    {path.name} ({size_mb:.1f} MB)")
    print(f"  Resume:  {'yes' if resume else 'no'}")
    print()

    _save_status({
        "state": "starting",
        "source": source,
        "file": str(path),
        "started_at": datetime.now(UTC).isoformat(),
    })

    try:
        result = run_import_pipeline(
            file_path=str(path),
            source=source,
            skip_report=args.no_scan,
            resume=resume,
            checkpoint_dir=_checkpoint_dir(),
            progress_callback=_cli_progress_callback,
        )
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Import interrupted. Use 'pith import --resume' to continue.")
        _save_status({"state": "interrupted", "updated_at": datetime.now(UTC).isoformat()})
        return 130
    except Exception as e:
        print(f"\n\n  ✗ Import failed: {e}")
        _save_status({"state": "failed", "error": str(e), "updated_at": datetime.now(UTC).isoformat()})
        return 1

    # ── Print results ──
    print("\n")  # Clear progress bar line
    status = result.get("status", "unknown")
    progress = result.get("progress", {})
    processed = progress.get("processed", 0)
    total = progress.get("total", 0)
    failed = progress.get("failed", 0)
    concepts = progress.get("concepts_extracted", 0)

    if status == "completed":
        print(f"  ✓ Import complete: {processed}/{total} conversations processed")
        if failed > 0:
            print(f"    ⚠️  {failed} conversations failed (see 'pith import log')")
        if concepts > 0:
            print(f"    📊 {concepts} concepts extracted")
    elif status == "aborted":
        print(f"  ⚠️  Import aborted: {processed}/{total} conversations processed")
        print("    Use 'pith import --resume' to continue from checkpoint.")
    else:
        error = result.get("error", "Unknown error")
        print(f"  ✗ {error}")
        _save_status({"state": "failed", "error": error, "updated_at": datetime.now(UTC).isoformat()})
        return 1

    # ── Display report ──
    report = result.get("report")
    if report and report.get("type") != "error":
        # Persist report for 'pith import report'
        report_file = _resolve_data_dir() / "import_last_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, default=str))
        print()
        _print_report(report)

    # ── Save final status ──
    _save_status({
        "state": status,
        "source": source,
        "file": str(path),
        "processed": processed,
        "total": total,
        "failed": failed,
        "concepts_extracted": concepts,
        "completed_at": datetime.now(UTC).isoformat(),
        "report_type": report.get("type") if report else None,
    })

    return 0


def _print_report(report: dict) -> None:
    """Pretty-print an import report to the terminal."""
    rtype = report.get("type", "unknown")
    summary = report.get("summary", "")

    if rtype == "contradiction":
        print("  ─── Contradiction Report ───")
        print(f"  {summary}")
        contradictions = report.get("contradictions", [])
        for i, c in enumerate(contradictions[:10], 1):
            area = c.get("knowledge_area", "?")
            detail = c.get("description", c.get("summary", ""))
            print(f"    {i}. [{area}] {detail[:120]}")
    elif rtype == "belief_evolution":
        print("  ─── Belief Evolution Report ───")
        print(f"  {summary}")
        changes = report.get("changes", report.get("evolution", []))
        for i, ch in enumerate(changes[:10], 1):
            area = ch.get("knowledge_area", "?")
            detail = ch.get("description", ch.get("summary", ""))
            print(f"    {i}. [{area}] {detail[:120]}")
    else:
        print(f"  ─── Import Report ({rtype}) ───")
        if summary:
            print(f"  {summary}")


def cmd_cancel(args) -> int:
    """Cancel an in-progress import."""
    from app.ops.import_pipeline import cancel_import

    status = _load_status() or {}
    state = str(status.get("state", "")).lower()
    active_states = {"starting", "running", "cancel_requested"}
    if state not in active_states:
        message = "No active import to cancel."
        if _json_requested(args):
            _print_json({"ok": False, "state": state or None, "message": message})
        else:
            print(message)
        return 1

    cancel_import()
    updated = dict(status)
    updated["state"] = "cancel_requested"
    updated["updated_at"] = datetime.now(UTC).isoformat()
    _save_status(updated)
    if _json_requested(args):
        _print_json({"ok": True, "state": "cancel_requested", "message": "Cancel requested."})
    else:
        print("✓ Cancel requested. The import will stop after the current conversation.")
    return 0


def cmd_status(args) -> int:
    """Show current/last import status."""
    status = _load_status()
    if not status:
        if _json_requested(args):
            _print_json({"history_found": False, "status": None})
            return 0
        print("No import history found.")
        return 0

    if _json_requested(args):
        _print_json({"history_found": True, "status": status})
        return 0

    state = status.get("state", "unknown")
    source = status.get("source", "?")
    file_name = Path(status.get("file", "?")).name if status.get("file") else "?"

    state_icons = {
        "running": "⏳", "completed": "✓", "failed": "✗",
        "aborted": "⚠️", "interrupted": "⏸️", "cancel_requested": "🛑",
        "starting": "⏳",
    }
    icon = state_icons.get(state, "?")

    print(f"  {icon} State: {state}")
    print(f"    Source:    {source}")
    print(f"    File:      {file_name}")

    if "processed" in status:
        processed = status["processed"]
        total = status["total"]
        failed = status.get("failed", 0)
        concepts = status.get("concepts_extracted", 0)
        print(f"    Progress:  {processed}/{total} conversations")
        if failed:
            print(f"    Failed:    {failed}")
        if concepts:
            print(f"    Concepts:  {concepts}")

    if ts := status.get("completed_at") or status.get("updated_at"):
        print(f"    Updated:   {ts}")

    if status.get("report_type"):
        print(f"    Report:    {status['report_type']} (run 'pith import report' to view)")

    return 0


def cmd_report(args) -> int:
    """Show the last import report."""
    status = _load_status()
    if not status:
        if _json_requested(args):
            _print_json({"report_found": False, "reason": "no_import_history"})
            return 1
        print("No import history found. Run 'pith import' first.")
        return 1

    # The full result with report is saved separately
    report_file = _resolve_data_dir() / "import_last_report.json"
    if not report_file.exists():
        if _json_requested(args):
            _print_json({"report_found": False, "reason": "no_report_file"})
            return 1
        print("No report available from last import.")
        print("  Tip: Reports are generated when the import completes without --no-scan.")
        return 1

    report = json.loads(report_file.read_text())
    if _json_requested(args):
        _print_json({"report_found": True, "report": report})
        return 0
    _print_report(report)
    return 0


def cmd_log(args) -> int:
    """Show import error log."""
    log = _log_file()
    if not log.exists():
        if _json_requested(args):
            _print_json({"log_found": False, "lines": []})
            return 0
        print("No import log found.")
        return 0

    lines = log.read_text().splitlines()
    # Show last N lines
    n = args.lines if hasattr(args, "lines") else 50
    selected = lines[-n:]
    if _json_requested(args):
        _print_json({"log_found": True, "lines": selected})
        return 0
    for line in selected:
        print(line)
    return 0


# ── Argparse setup ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pith import",
        description="Import conversation history into Pith",
        epilog=EXPORT_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command")

    # ── run ──
    run_parser = sub.add_parser(
        "run", help="Run an import",
        epilog=EXPORT_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--source", "-s",
        choices=["chatgpt", "claude"],
        help="Source platform (auto-detected from filename if omitted)",
    )
    run_parser.add_argument(
        "--file", "-f",
        help="Path to export file (.zip or .json). Auto-scans ~/Downloads if omitted.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted import from checkpoint",
    )
    run_parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip contradiction/belief scan (faster, no report generated)",
    )
    run_parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use local-only extraction (not yet available)",
    )
    run_parser.add_argument(
        "--confirm-llm-processing",
        action="store_true",
        help="Acknowledge that extraction sends excerpts to your configured LLM provider",
    )

    # ── cancel ──
    cancel_parser = sub.add_parser("cancel", help="Cancel an in-progress import")
    cancel_parser.add_argument("--json", action="store_true")

    # ── status ──
    status_parser = sub.add_parser("status", help="Show current/last import status")
    status_parser.add_argument("--json", action="store_true")

    # ── report ──
    report_parser = sub.add_parser("report", help="View the last import report")
    report_parser.add_argument("--json", action="store_true")

    # ── log ──
    log_parser = sub.add_parser("log", help="Show import error log")
    log_parser.add_argument("--json", action="store_true")
    log_parser.add_argument(
        "--lines", "-n", type=int, default=50,
        help="Number of log lines to show (default: 50)",
    )

    # ── help ──
    sub.add_parser("help", help="Show import help")

    return parser


# ── Main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "help":
        parser.print_help()
        return 0

    dispatch = {
        "run": cmd_run,
        "cancel": cmd_cancel,
        "status": cmd_status,
        "report": cmd_report,
        "log": cmd_log,
    }

    handler = dispatch.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
