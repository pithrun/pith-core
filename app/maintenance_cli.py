"""CLI for Pith Maintenance — autonomous cognitive lifecycle.

Usage:
  python -m app.maintenance_cli run [--phases 1,2,3] [--dry-run]
  python -m app.maintenance_cli status
  python -m app.maintenance_cli install   # Install launchd scheduler
  python -m app.maintenance_cli uninstall # Remove launchd scheduler
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# INFRA-003: Load .env for feature flag overrides (PITH_FF_* env vars).
# python-dotenv is in requirements.txt. If missing, falls back to launchd env vars.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PLIST_NAME = "com.pith.maintenance"
PLIST_SOURCE = Path(__file__).parent.parent / "scripts" / f"{PLIST_NAME}.plist"
PLIST_DEST = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
LOCK_FILE = "/tmp/pith_maintenance.lock"

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
        <string>app.maintenance_cli</string>
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
    </dict>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>15</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/maintenance_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/maintenance_stderr.log</string>
</dict>
</plist>
"""


def cmd_run(args):
    """Run maintenance cycle."""
    # Amendment 1: Mutex lock — prevent parallel maintenance runs
    lock_fd = open(LOCK_FILE, "w")  # noqa: SIM115 — lock must outlive context manager
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("✗ Maintenance already running (lock file held). Use --force to override.")
        sys.exit(1)

    try:
        return _cmd_run_inner(args)
    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()


def _rotate_logs():
    """Amendment 11: Rotate maintenance logs if they exceed 10 MB."""
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_ROTATIONS = 3
    data_dir = os.environ.get("PITH_DATA_DIR", "data")
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

    from app.maintenance import run_maintenance_sync

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

    result = run_maintenance_sync(phases=phases, dry_run=args.dry_run)

    print(json.dumps(result, indent=2))

    if result.get("success"):
        print(f"\n✓ Maintenance complete in {result['duration_seconds']}s")
    else:
        print("\n✗ Maintenance had errors:")
        for err in result.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)


def cmd_status(args):
    """Show maintenance task status."""
    from app.async_tasks import task_runner
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

    # Check if scheduler is installed
    if PLIST_DEST.exists():
        print(f"\n✓ Scheduler installed at {PLIST_DEST}")
    else:
        print("\n⚠ Scheduler not installed. Run: python -m app.maintenance_cli install")


def cmd_install(args):
    """Install launchd scheduler for automatic maintenance.

    Amendment 9: Generates plist dynamically with current system paths
    instead of copying a static template. This ensures portability
    across different Python installs and data directory locations.
    """
    import plistlib

    # Resolve paths dynamically
    pith_dir = Path(__file__).parent.parent.resolve()
    data_dir = os.environ.get("PITH_DATA_DIR", str(pith_dir / "data"))
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
    PLIST_DEST.parent.mkdir(parents=True, exist_ok=True)

    # Unload old plist if present, write new one, load it
    subprocess.run(["launchctl", "unload", str(PLIST_DEST)], capture_output=True)
    PLIST_DEST.write_text(plist_content)
    print(f"✓ Generated plist at {PLIST_DEST}")

    result = subprocess.run(["launchctl", "load", str(PLIST_DEST)], capture_output=True)

    if result.returncode == 0:
        print("✓ Scheduler loaded into launchd")
        print(f"  Python: {python_path}")
        print(f"  Working dir: {pith_dir}")
        print(f"  Data dir: {data_dir}")
        print(f"  Profile: {profile}")
        print("  Schedule: every 6 hours (3am, 9am, 3pm, 9pm)")
        print(f"  Logs: {log_dir}/maintenance_stderr.log")
    else:
        print(f"✗ Failed to load: {result.stderr.decode()}")
        sys.exit(1)


def cmd_uninstall(args):
    """Remove launchd scheduler."""
    if PLIST_DEST.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_DEST)], capture_output=True)
        PLIST_DEST.unlink()
        print(f"✓ Uninstalled scheduler from {PLIST_DEST}")
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

    # install
    install_parser = sub.add_parser("install", help="Install launchd scheduler")
    install_parser.set_defaults(func=cmd_install)

    # uninstall
    uninstall_parser = sub.add_parser("uninstall", help="Remove launchd scheduler")
    uninstall_parser.set_defaults(func=cmd_uninstall)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
