"""Profile-aware data directory resolution for Pith.

Single source of truth for determining where Pith data lives.
Supports multiple isolated Pith instances on one machine.

Resolution order:
  1. PITH_DATA_DIR env var (explicit override, highest priority)
  2. PITH_PROFILE env var → ~/pith-data/{profile}/
  3. Default: ~/pith-data/default/
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PITH_DATA_ROOT = Path.home() / "pith-data"


def resolve_data_dir(profile: str | None = None) -> Path:
    """Resolve the data directory using the priority chain.

    Args:
        profile: Explicit profile name (overrides env vars except PITH_DATA_DIR).

    Returns:
        Absolute Path to the data directory (created if needed).
    """
    # Priority 1: Explicit env var override
    explicit = os.environ.get("PITH_DATA_DIR")
    if explicit:
        data_dir = Path(explicit)
        logger.debug("Data dir from PITH_DATA_DIR: %s", data_dir)
        _ensure_dirs(data_dir)
        return data_dir

    # Priority 2: Profile name (arg or env)
    profile_name = profile or os.environ.get("PITH_PROFILE")
    if profile_name:
        data_dir = PITH_DATA_ROOT / profile_name
        logger.debug("Data dir from profile '%s': %s", profile_name, data_dir)
        _ensure_dirs(data_dir)
        return data_dir

    # Priority 3: Default profile
    data_dir = PITH_DATA_ROOT / "default"
    logger.debug("Data dir from default profile: %s", data_dir)
    _ensure_dirs(data_dir)
    return data_dir


def _ensure_dirs(base: Path) -> None:
    """Create standard subdirectory structure if missing."""
    for subdir in ("index", "backups", "logs"):
        (base / subdir).mkdir(parents=True, exist_ok=True)


# Default DB filename — used when no CANONICAL_DB.env exists
DEFAULT_DB_FILENAME = "pith.db"

# Env file name for per-profile DB override
CANONICAL_DB_ENV = "CANONICAL_DB.env"


def get_canonical_db_name(data_dir: Path) -> str:
    """Read the canonical DB filename for a profile directory.

    Resolution order:
      1. PITH_DB_NAME env var (explicit override, highest priority)
      2. CANONICAL_DB.env file in data_dir (per-profile declaration)
      3. Default: "pith.db"

    Returns:
        DB filename (not a full path — just the name like "pith.db").
    """
    # Priority 1: Env var override
    env_name = os.environ.get("PITH_DB_NAME")
    if env_name:
        logger.debug("DB name from PITH_DB_NAME: %s", env_name)
        return env_name

    # Priority 2: Per-profile declaration file
    env_file = data_dir / CANONICAL_DB_ENV
    if env_file.exists():
        try:
            content = env_file.read_text().strip()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "CANONICAL_DB":
                        db_name = value.strip().strip("'\"")
                        if db_name:
                            logger.debug("DB name from %s: %s", env_file, db_name)
                            return db_name
        except OSError:
            logger.warning("Could not read %s, using default", env_file)

    # Priority 3: Default
    logger.debug("DB name from default: %s", DEFAULT_DB_FILENAME)
    return DEFAULT_DB_FILENAME


def resolve_db_path(data_dir: Path | None = None) -> Path:
    """Resolve the full path to the canonical database file.

    Args:
        data_dir: Profile data directory. If None, calls resolve_data_dir().

    Returns:
        Absolute Path to the database file.
    """
    if data_dir is None:
        data_dir = resolve_data_dir()
    db_name = get_canonical_db_name(data_dir)
    db_path = data_dir / db_name
    logger.debug("Resolved DB path: %s", db_path)
    return db_path


def list_profiles() -> list[str]:
    """List all profile directories under ~/pith-data/."""
    if not PITH_DATA_ROOT.exists():
        return []
    return sorted(d.name for d in PITH_DATA_ROOT.iterdir() if d.is_dir() and not d.name.startswith("."))


def get_active_profile() -> str:
    """Return the name of the currently active profile.

    If using an explicit PITH_DATA_DIR that doesn't
    live under ~/pith-data/, returns the path string instead.
    """
    explicit = os.environ.get("PITH_DATA_DIR")
    if explicit:
        p = Path(explicit)
        try:
            return p.relative_to(PITH_DATA_ROOT).parts[0]
        except (ValueError, IndexError):
            return str(p)

    profile_name = os.environ.get("PITH_PROFILE")
    if profile_name:
        return profile_name

    return "default"
