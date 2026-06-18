"""CLI for Pith Maintenance — autonomous cognitive lifecycle.

Usage:
  python -m app.ops.maintenance_cli run [--phases 1,2,3] [--dry-run]
  python -m app.ops.maintenance_cli autolearn-drain [--dry-run] [--max-rows N] [--max-seconds N] [--task-type TYPE]
  python -m app.ops.maintenance_cli reflection-requeue-pressure-deferrals [--dry-run]
  python -m app.ops.maintenance_cli status
  python -m app.ops.maintenance_cli install   # Install optional launchd scheduler
  python -m app.ops.maintenance_cli uninstall # Remove optional launchd scheduler
"""

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# INFRA-003: Load .env for feature flag overrides (PITH_FF_* env vars).
# python-dotenv is in requirements.txt. If missing, falls back to launchd env vars.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PLIST_NAME = "com.pith.maintenance"
PLIST_SOURCE = Path(__file__).parent.parent.parent / "scripts" / f"{PLIST_NAME}.plist"
LOCK_FILE = "/tmp/pith_maintenance.lock"
REFLECTION_PRESSURE_DEFERRAL_ERROR = "ReflectionDeferredError: pressure_protected_reflection_deferred"


def _launch_agents_dir() -> Path:
    override = os.environ.get("PITH_LAUNCH_AGENTS_DIR")
    if override:
        return Path(override)
    return Path.home() / "Library" / "LaunchAgents"


def _plist_dest() -> Path:
    return _launch_agents_dir() / f"{PLIST_NAME}.plist"


def _resolved_data_dir() -> str:
    from app.core.profile import resolve_data_dir

    return str(resolve_data_dir())


# Amendment 9: Dynamic plist template — resolved at install time
PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pith.maintenance</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>app.ops.maintenance_cli</string>
        <string>run</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_directory}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PITH_DATA_DIR</key>
        <string>{pith_data_dir}</string>
        <key>PITH_PROFILE</key>
        <string>{profile}</string>
        <key>PITH_MAINTENANCE_SOURCE</key>
        <string>external_launchd</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{log_dir}/maintenance_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/maintenance_stderr.log</string>
</dict>
</plist>
"""


def _acquire_maintenance_lock():
    lock_fd = open(LOCK_FILE, "w")  # noqa: SIM115 — lock must outlive context manager
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("✗ Maintenance already running (lock file held). Use --force to override.")
        lock_fd.close()
        sys.exit(1)
    return lock_fd


def _release_maintenance_lock(lock_fd) -> None:
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    lock_fd.close()


def cmd_run(args):
    """Run maintenance cycle."""
    # Amendment 1: Mutex lock — prevent parallel maintenance runs
    lock_fd = _acquire_maintenance_lock()
    lease = None
    try:
        try:
            from app.ops.maintenance import PHASE_TIMEOUT_SECONDS
            from app.ops.pressure_state import write_maintenance_lease

            source = os.environ.get("PITH_MAINTENANCE_SOURCE", "manual_cli")
            if source == "manual":
                source = "manual_cli"
            lease = write_maintenance_lease(
                source=source,
                phase="maintenance",
                expected_timeout_seconds=float(PHASE_TIMEOUT_SECONDS),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
        except Exception as lease_err:
            print(f"Warning: maintenance active lease write failed: {lease_err}")
        return _cmd_run_inner(args)
    finally:
        if lease is not None:
            try:
                from app.ops.pressure_state import clear_maintenance_lease

                clear_maintenance_lease(run_id=lease.run_id, pid=os.getpid())
            except Exception as lease_err:
                print(f"Warning: maintenance active lease clear failed: {lease_err}")
        _release_maintenance_lock(lock_fd)


def _rotate_logs():
    """Amendment 11: Rotate maintenance logs if they exceed 10 MB."""
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_ROTATIONS = 3
    data_dir = _resolved_data_dir()
    log_dir = os.path.join(data_dir, "logs")

    for log_name in ("maintenance_stdout.log", "maintenance_stderr.log"):
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            continue
        try:
            if os.path.getsize(log_path) <= MAX_LOG_SIZE:
                continue
            # Rotate: remove oldest, shift others, move current
            for i in range(MAX_ROTATIONS, 0, -1):
                older = f"{log_path}.{i}"
                if i == MAX_ROTATIONS and os.path.exists(older):
                    os.remove(older)
                elif i < MAX_ROTATIONS and os.path.exists(f"{log_path}.{i}"):
                    shutil.move(f"{log_path}.{i}", f"{log_path}.{i + 1}")
            shutil.move(log_path, f"{log_path}.1")
        except Exception as e:
            print(f"Warning: log rotation failed for {log_name}: {e}")


def _cmd_run_inner(args):
    """Inner run logic (called under mutex lock)."""
    # Amendment 11: Rotate logs before each run
    _rotate_logs()

    from app.ops.maintenance import run_maintenance_sync

    phases = None
    if args.phases:
        try:
            phases = [int(p.strip()) for p in args.phases.split(",")]
        except ValueError:
            print("Error: --phases must be comma-separated integers (e.g., 1,2,5)")
            sys.exit(1)
        invalid = [p for p in phases if p < 1 or p > 5]
        if invalid:
            print(f"Error: invalid phase numbers {invalid}. Valid: 1-5")
            sys.exit(1)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Running Pith maintenance...")
    print(f"  Profile: {os.environ.get('PITH_PROFILE', 'default')}")
    if phases:
        print(f"  Phases: {phases}")
    print()

    source = os.environ.get("PITH_MAINTENANCE_SOURCE", "manual")
    result = run_maintenance_sync(phases=phases, dry_run=args.dry_run, source=source)

    print(json.dumps(result, indent=2))

    if result.get("success"):
        print(f"\n✓ Maintenance complete in {result['duration_seconds']}s")
    else:
        print("\n✗ Maintenance had errors:")
        for err in result.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)


def _health_url() -> str:
    base = os.environ.get("PITH_API_URL", "http://localhost:8000").rstrip("/")
    if base.endswith("/health"):
        return base
    return f"{base}/health"


def _fetch_server_health(timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(_health_url(), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _print_scheduler_summary(health: dict | None) -> None:
    if health:
        effective_state = health.get("maintenance_state", "unknown")
        mode = health.get("maintenance_mode", "unknown")
        built_in_state = health.get("built_in_maintenance_state", "unknown")
        external = health.get("external_maintenance", {}) or {}
        external_installed = bool(external.get("scheduler_installed"))
        external_label = "installed" if external_installed else "not installed"

        print(f"\nEffective maintenance: {effective_state} ({mode} scheduler)")
        print(f"Built-in scheduler: {built_in_state}")
        print(
            "External launchd scheduler: "
            f"{external_label} (optional; daily 3:00 AM when enabled)"
        )
        return

    external_label = "installed" if _plist_dest().exists() else "not installed"
    print("\nServer maintenance state: unavailable")
    print(
        "External launchd scheduler: "
        f"{external_label} (optional; daily 3:00 AM when enabled)"
    )


def cmd_status(args):
    """Show maintenance task status."""
    from app.ops.async_tasks import task_runner
    from app.storage import _get_connection

    conn = _get_connection()
    status = task_runner.get_status(conn)

    print("Pith Maintenance Status")
    print("=" * 50)
    print()

    tasks = status.get("tasks", {})
    for task_type, info in sorted(tasks.items()):
        last_run = info.get("last_successful_run", "never")
        running = info.get("is_running", False)
        marker = "🔄" if running else "✓" if last_run != "never" else "⚠"
        print(f"  {marker} {task_type}: last run {last_run}")

    degraded = status.get("degraded_tasks", [])
    if degraded:
        print(f"\n⚠ Degraded tasks ({len(degraded)}):")
        for d in degraded:
            print(f"  - {d['task_type']}: {d['reason']}")
    else:
        print("\n✓ No degraded tasks")

    _print_scheduler_summary(_fetch_server_health())


def cmd_autolearn_drain(args):
    """Drain deferred autolearn maintenance rows under explicit budgets."""
    from app.core.config import (
        get_autolearn_maintenance_batch_size,
        get_autolearn_maintenance_catchup_max_rows,
        get_autolearn_maintenance_catchup_max_seconds,
    )
    from app.session.autolearn_maintenance import (
        get_autolearn_maintenance_drain_plan,
        run_autolearn_maintenance_catchup,
    )
    from app.storage import owned_connection

    max_rows = args.max_rows if args.max_rows is not None else get_autolearn_maintenance_catchup_max_rows()
    max_seconds = args.max_seconds if args.max_seconds is not None else get_autolearn_maintenance_catchup_max_seconds()
    batch_size = args.batch_size if args.batch_size is not None else get_autolearn_maintenance_batch_size()

    lock_fd = _acquire_maintenance_lock()
    try:
        with owned_connection() as conn:
            if args.dry_run:
                plan = get_autolearn_maintenance_drain_plan(conn, batch_size, task_type=args.task_type)
                result = {
                    "success": True,
                    "dry_run": True,
                    "max_rows": max_rows,
                    "max_seconds": max_seconds,
                    "task_type_filter": args.task_type,
                    "ready_before": plan["ready_total"],
                    "ready_after": plan["ready_total"],
                    "ready_delta": 0,
                    "processed_by_task": {task_type: 0 for task_type in plan["ready_by_task"]},
                    "oldest_ready_age_before_seconds": plan.get("oldest_ready_age_seconds"),
                    "oldest_ready_age_after_seconds": plan.get("oldest_ready_age_seconds"),
                    "oldest_ready_age_delta_seconds": 0,
                    "drain_rate_rows_per_second": 0.0,
                    "plan": plan,
                }
            else:
                result = asyncio.run(
                    run_autolearn_maintenance_catchup(
                        conn,
                        max_rows=max_rows,
                        max_seconds=max_seconds,
                        batch_size=batch_size,
                        task_type=args.task_type,
                    )
                )
                result["dry_run"] = False
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        _release_maintenance_lock(lock_fd)


def cmd_reflection_requeue_pressure_deferrals(args):
    """Requeue failed full-reflection jobs caused by pressure deferral."""
    from app.core.datetime_utils import _utc_now_iso
    from app.core.profile import get_active_profile
    from app.storage.lifecycle_jobs import (
        count_failed_lifecycle_jobs_by_error,
        requeue_failed_lifecycle_jobs_by_error,
    )

    profile = get_active_profile()
    error = args.error
    matching_count = count_failed_lifecycle_jobs_by_error(
        profile=profile,
        source="reflection_full",
        stage="reflect",
        error=error,
    )

    result = {
        "success": True,
        "dry_run": bool(args.dry_run),
        "profile": profile,
        "source": "reflection_full",
        "stage": "reflect",
        "matched_failed_rows": matching_count,
        "requeued_rows": 0,
        "error": error,
    }
    if not args.dry_run and matching_count:
        now = _utc_now_iso()
        result["requeued_rows"] = requeue_failed_lifecycle_jobs_by_error(
            profile=profile,
            source="reflection_full",
            stage="reflect",
            error=error,
            next_retry_at=now,
            now=now,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_install(args):
    """Install optional launchd scheduler for automatic maintenance.

    Amendment 9: Generates plist dynamically with current system paths
    instead of copying a static template. This ensures portability
    across different Python installs and data directory locations.
    """
    import plistlib

    # Resolve paths dynamically
    pith_dir = Path(__file__).parent.parent.parent.resolve()
    data_dir = _resolved_data_dir()
    log_dir = os.path.join(data_dir, "logs")
    python_path = sys.executable  # Use the CURRENT Python interpreter
    profile = os.environ.get("PITH_PROFILE", "default")

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate plist content
    plist_content = PLIST_TEMPLATE.format(
        python_path=python_path,
        working_directory=str(pith_dir),
        pith_data_dir=data_dir,
        profile=profile,
        log_dir=log_dir,
    )

    # Validate XML before installing
    try:
        plistlib.loads(plist_content.encode())
    except Exception as e:
        print(f"✗ Generated plist is invalid XML: {e}")
        sys.exit(1)

    # Ensure LaunchAgents directory exists
    plist_dest = _plist_dest()
    plist_dest.parent.mkdir(parents=True, exist_ok=True)

    # Unload old plist if present, write new one, load it
    subprocess.run(["launchctl", "unload", str(plist_dest)], capture_output=True)
    plist_dest.write_text(plist_content)
    print(f"✓ Generated plist at {plist_dest}")

    result = subprocess.run(["launchctl", "load", str(plist_dest)], capture_output=True)

    if result.returncode == 0:
        print("✓ Optional external launchd scheduler loaded")
        print("  Built-in server maintenance remains the default beta scheduler.")
        print("  Use launchd only when you want a separate macOS 3:00 AM maintenance job.")
        print(f"  Python: {python_path}")
        print(f"  Working dir: {pith_dir}")
        print(f"  Data dir: {data_dir}")
        print(f"  Profile: {profile}")
        print("  Schedule: daily at 3:00am")
        print(f"  Logs: {log_dir}/maintenance_stderr.log")
    else:
        print(f"✗ Failed to load: {result.stderr.decode()}")
        sys.exit(1)


def cmd_uninstall(args):
    """Remove launchd scheduler."""
    plist_dest = _plist_dest()
    if plist_dest.exists():
        subprocess.run(["launchctl", "unload", str(plist_dest)], capture_output=True)
        plist_dest.unlink()
        print(f"✓ Uninstalled scheduler from {plist_dest}")
    else:
        print("Scheduler was not installed.")


def main():
    parser = argparse.ArgumentParser(description="Pith Maintenance — autonomous cognitive lifecycle")
    sub = parser.add_subparsers(dest="command")

    # run
    run_parser = sub.add_parser("run", help="Run maintenance cycle")
    run_parser.add_argument("--phases", help="Comma-separated phase numbers (default: all)")
    run_parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    run_parser.set_defaults(func=cmd_run)

    # status
    status_parser = sub.add_parser("status", help="Show task status")
    status_parser.set_defaults(func=cmd_status)

    # autolearn-drain
    drain_parser = sub.add_parser("autolearn-drain", help="Drain deferred autolearn maintenance rows")
    drain_parser.add_argument("--dry-run", action="store_true", help="Show queue plan without claiming rows")
    drain_parser.add_argument("--max-rows", type=int, help="Maximum rows to process")
    drain_parser.add_argument("--max-seconds", type=int, help="Maximum wall-clock seconds to run")
    drain_parser.add_argument("--batch-size", type=int, help="Rows to claim per batch")
    drain_parser.add_argument(
        "--task-type",
        choices=["governance_recompute", "similarity_supersession", "subject_key_supersession"],
        help="Only drain one autolearn task type",
    )
    drain_parser.set_defaults(func=cmd_autolearn_drain)

    # reflection-requeue-pressure-deferrals
    reflection_requeue_parser = sub.add_parser(
        "reflection-requeue-pressure-deferrals",
        help="Requeue failed full-reflection jobs caused by pressure deferral",
    )
    reflection_requeue_parser.add_argument("--dry-run", action="store_true", help="Count matching rows only")
    reflection_requeue_parser.add_argument(
        "--error",
        default=REFLECTION_PRESSURE_DEFERRAL_ERROR,
        help="Exact failed lifecycle error to requeue",
    )
    reflection_requeue_parser.set_defaults(func=cmd_reflection_requeue_pressure_deferrals)

    # install
    install_parser = sub.add_parser("install", help="Install optional launchd scheduler (daily 3:00 AM)")
    install_parser.set_defaults(func=cmd_install)

    # uninstall
    uninstall_parser = sub.add_parser("uninstall", help="Remove optional launchd scheduler")
    uninstall_parser.set_defaults(func=cmd_uninstall)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
