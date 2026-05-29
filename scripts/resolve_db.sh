#!/bin/bash
# ============================================================
# Pith — Database Path Resolution (shared by all scripts)
# ============================================================
# Mirrors the priority chain in app/profile.py:
#   1. PITH_DATA_DIR env var (explicit override)
#   2. PITH_PROFILE env var → ~/pith-data/{profile}/
#   3. PITH_HOME/data/ (legacy flat layout)
#   4. PROJECT_DIR/data/ (dev layout)
#   5. ~/pith-data/default/ (fallback)
# At each location, checks pith.db first, then brain.db.
#
# Usage: source scripts/resolve_db.sh
#        resolve_pith_db   # sets DB_PATH, DATA_DIR
# ============================================================

resolve_pith_db() {
    local PITH_DATA_ROOT="$HOME/pith-data"
    local _PITH_HOME="${PITH_HOME:-$HOME/.pith}"

    # Helper: check a directory for pith.db or brain.db
    _check_dir() {
        local dir="$1"
        if [[ -f "$dir/pith.db" ]]; then
            DB_PATH="$dir/pith.db"
            DATA_DIR="$dir"
            return 0
        elif [[ -f "$dir/brain.db" ]]; then
            DB_PATH="$dir/brain.db"
            DATA_DIR="$dir"
            return 0
        fi
        return 1
    }

    # Priority 1: Explicit PITH_DATA_DIR
    if [[ -n "${PITH_DATA_DIR:-}" ]]; then
        _check_dir "$PITH_DATA_DIR" && return 0
    fi

    # Priority 2: PITH_PROFILE → ~/pith-data/{profile}/
    if [[ -n "${PITH_PROFILE:-}" ]]; then
        _check_dir "$PITH_DATA_ROOT/$PITH_PROFILE" && return 0
    fi

    # Priority 3: Legacy PITH_HOME/data/
    _check_dir "$_PITH_HOME/data" && return 0

    # Priority 4: PROJECT_DIR/data/ (dev layout, if PROJECT_DIR is set)
    if [[ -n "${PROJECT_DIR:-}" ]]; then
        _check_dir "$PROJECT_DIR/data" && return 0
    fi

    # Priority 5: Default profile
    _check_dir "$PITH_DATA_ROOT/default" && return 0

    # Nothing found — set DB_PATH to canonical location for error reporting
    if [[ -n "${PITH_PROFILE:-}" ]]; then
        DB_PATH="$PITH_DATA_ROOT/$PITH_PROFILE/pith.db"
        DATA_DIR="$PITH_DATA_ROOT/$PITH_PROFILE"
    else
        DB_PATH="$PITH_DATA_ROOT/default/pith.db"
        DATA_DIR="$PITH_DATA_ROOT/default"
    fi
    return 1
}
