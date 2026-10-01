#!/bin/bash
# ============================================================
# Pith — Beta Health Report
# ============================================================
# Usage: bash scripts/beta-report.sh
#        pith report     (if pith CLI is in PATH)
#
# Generates a snapshot of Pith health metrics for beta feedback.
# Copy-paste the output to share with the Pith team.
# ============================================================

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Load persisted configuration before resolving profile-dependent paths. Explicit
# caller values retain precedence over persisted values.
PITH_HOME="${PITH_HOME:-$HOME/.pith}"
PITH_EXPLICIT_PROFILE="${PITH_PROFILE:-}"
PITH_EXPLICIT_DATA_DIR="${PITH_DATA_DIR:-}"
PITH_EXPLICIT_API_KEY="${PITH_API_KEY:-}"
PITH_EXPLICIT_API_URL="${PITH_API_URL:-}"
PITH_EXPLICIT_PORT="${PITH_PORT:-}"
if [[ -f "$PITH_HOME/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PITH_HOME/.env"
    set +a
fi
[[ -n "$PITH_EXPLICIT_PROFILE" ]] && export PITH_PROFILE="$PITH_EXPLICIT_PROFILE"
[[ -n "$PITH_EXPLICIT_DATA_DIR" ]] && export PITH_DATA_DIR="$PITH_EXPLICIT_DATA_DIR"
[[ -n "$PITH_EXPLICIT_API_KEY" ]] && export PITH_API_KEY="$PITH_EXPLICIT_API_KEY"
[[ -n "$PITH_EXPLICIT_API_URL" ]] && export PITH_API_URL="$PITH_EXPLICIT_API_URL"
[[ -n "$PITH_EXPLICIT_PORT" ]] && export PITH_PORT="$PITH_EXPLICIT_PORT"

# Resolve paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Find database using shared resolution (mirrors app/profile.py)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/resolve_db.sh" 2>/dev/null || {
    echo -e "${RED}ERROR: resolve_db.sh not found at $SCRIPT_DIR/${NC}"; exit 1;
}
if ! resolve_pith_db; then
    echo -e "${RED}No database found.${NC}"
    echo "  Checked: ~/pith-data/\${PITH_PROFILE:-default}/{pith,brain}.db"
    echo "  Checked: $PITH_HOME/data/{pith,brain}.db"
    echo "  Checked: $PROJECT_DIR/data/{pith,brain}.db"
    echo "  Make sure Pith has been started at least once."
    exit 1
fi

# Load API key. The sourced home environment follows explicit caller input;
# legacy installed/project files remain fallbacks.
API_KEY="${PITH_API_KEY:-}"
if [[ -z "$API_KEY" ]] && [[ -f "$PITH_HOME/config/api.key" ]]; then
    API_KEY=$(cat "$PITH_HOME/config/api.key" 2>/dev/null)
elif [[ -z "$API_KEY" ]] && [[ -f "$PROJECT_DIR/.env" ]]; then
    API_KEY=$(grep '^PITH_API_KEY=' "$PROJECT_DIR/.env" 2>/dev/null | tail -1 | cut -d'=' -f2-)
fi
CURL_HEADERS=()
if [[ -n "$API_KEY" ]]; then
    CURL_HEADERS=(-H "X-API-Key:$API_KEY")
fi

API="${PITH_API_URL:-http://localhost:${PITH_PORT:-8000}}"
while [[ "$API" == */ ]]; do
    API="${API%/}"
done
API_TIMEOUT="${PITH_REPORT_API_TIMEOUT:-5}"
API_DISPLAY=$(python3 - "$API" 2>/dev/null <<'PY'
import sys
from urllib.parse import urlsplit, urlunsplit

try:
    parts = urlsplit(sys.argv[1])
    if not parts.scheme or not parts.hostname:
        raise ValueError("API URL requires a scheme and hostname")
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    print(urlunsplit((parts.scheme, host, "", "", "")))
except ValueError:
    print("<invalid>")
PY
)
[[ -n "$API_DISPLAY" ]] || API_DISPLAY="<invalid>"

echo ""
echo -e "${BOLD}================================${NC}"
echo -e "${BOLD}  Pith Beta Report${NC}"
echo -e "${BOLD}================================${NC}"
echo ""
echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# --- Health check (via API if server is running) ---
# API reachability and source coherence are separate states. API-backed sections
# are permitted only when /health identifies the same database as this report.
API_REACHABLE=false
API_COHERENT=false
API_IDENTITY_STATUS="unavailable"
API_DB_PATH=""
API_PROFILE=""
if HEALTH=$(curl -sf --max-time "$API_TIMEOUT" "${CURL_HEADERS[@]}" -- "$API/health" 2>/dev/null) \
    && [[ -n "$HEALTH" ]]; then
    API_REACHABLE=true
    API_DB_PATH=$(printf '%s' "$HEALTH" | python3 -c '
import json, sys
value = json.load(sys.stdin).get("db_path")
print(value if isinstance(value, str) else "")
' 2>/dev/null)
    API_PROFILE=$(printf '%s' "$HEALTH" | python3 -c '
import json, sys
value = json.load(sys.stdin).get("profile")
print(value if isinstance(value, str) else "")
' 2>/dev/null)
fi

if [ "$API_REACHABLE" = false ]; then
    echo -e "${YELLOW}Pith API not responding — using direct DB access${NC}"
    echo ""
elif [[ -z "$API_DB_PATH" ]]; then
    API_IDENTITY_STATUS="unverified"
    echo -e "${YELLOW}Pith API online, but database identity is unavailable — using direct DB access${NC}"
    echo ""
else
    LOCAL_DB_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$DB_PATH")
    API_DB_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$API_DB_PATH")
    if [[ "$LOCAL_DB_REAL" == "$API_DB_REAL" ]]; then
        API_COHERENT=true
        API_IDENTITY_STATUS="matched"
        echo -e "${GREEN}Pith API: online (database identity matched)${NC}"
        echo ""
    else
        API_IDENTITY_STATUS="mismatch"
        echo -e "${YELLOW}Pith API/database mismatch — using direct DB access${NC}"
        echo "  API database:       $API_DB_PATH"
        echo "  Report database:    $DB_PATH"
        echo ""
    fi
fi

# --- Stats (try API, fall back to direct DB) ---
echo -e "${BOLD}--- Pith Stats ---${NC}"
if [ "$API_COHERENT" = true ]; then
    STATS=$(curl -sf --max-time "$API_TIMEOUT" "${CURL_HEADERS[@]}" -- \
        "$API/pith_stats?detail=fast&freshness=authoritative&force_refresh=true&capture_budget_ms=2500" \
        2>/dev/null)
    STATS_RENDERED=""
    if [[ -n "$STATS" ]]; then
        STATS_RENDERED=$(printf '%s' "$STATS" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("  Total concepts:     {}".format(d.get("total_concepts", 0)))
print("  Associations:       {}".format(d.get("associations", 0)))
print("  Knowledge areas:    {}".format(d.get("knowledge_areas", 0)))
print("  Avg confidence:     {:.2f}".format(d.get("avg_confidence", 0)))
print("  Avg stability:      {:.2f}".format(d.get("avg_stability", 0)))
status = d.get("freshness_status")
state = d.get("freshness_state") or "unknown"
reason = d.get("freshness_reason") or "unspecified"
if not status:
    if d.get("stale"):
        status = "legacy_stale"
        reason = "freshness_status_missing"
    elif d.get("partial"):
        status = "legacy_partial"
        reason = "freshness_status_missing"
    else:
        status = "unknown"
        reason = "freshness_status_missing"
details = [status, state, reason]
if d.get("cache_age_ms") is not None:
    details.append("cache_age_ms={}".format(d.get("cache_age_ms")))
if d.get("max_stale_ms") is not None:
    details.append("max_stale_ms={}".format(d.get("max_stale_ms")))
print("  Stats freshness:    {}".format("; ".join(details)))
' 2>/dev/null)
    fi
    if [ -z "$STATS_RENDERED" ]; then
        echo -e "${YELLOW}  Stats API timed out or failed — using direct DB access${NC}"
        echo "  Stats source:       direct database (authoritative API failed)"
        python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA busy_timeout=10000')
total = conn.execute(\"SELECT COUNT(*) FROM concepts WHERE status='active'\").fetchone()[0]
assoc = conn.execute('SELECT COUNT(*) FROM associations').fetchone()[0]
areas = conn.execute(\"SELECT COUNT(DISTINCT knowledge_area) FROM concepts WHERE status='active'\").fetchone()[0]
avg_conf = conn.execute(\"SELECT AVG(confidence) FROM concepts WHERE status='active'\").fetchone()[0] or 0
avg_stab = conn.execute(\"SELECT AVG(stability) FROM concepts WHERE status='active'\").fetchone()[0] or 0
print(f'  Total concepts:     {total}')
print(f'  Associations:       {assoc}')
print(f'  Knowledge areas:    {areas}')
print(f'  Avg confidence:     {avg_conf:.2f}')
print(f'  Avg stability:      {avg_stab:.2f}')
conn.close()
"
    else
        echo "  Stats source:       API authoritative request"
        echo "$STATS_RENDERED"
    fi
else
    case "$API_IDENTITY_STATUS" in
        mismatch) STATS_SOURCE="direct database (API/database mismatch)" ;;
        unverified) STATS_SOURCE="direct database (API identity unverified)" ;;
        *) STATS_SOURCE="direct database (API unavailable)" ;;
    esac
    echo "  Stats source:       $STATS_SOURCE"
    python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA busy_timeout=10000')
total = conn.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
assoc = conn.execute('SELECT COUNT(*) FROM associations').fetchone()[0]
areas = conn.execute('SELECT COUNT(DISTINCT knowledge_area) FROM concepts').fetchone()[0]
avg_conf = conn.execute('SELECT AVG(confidence) FROM concepts').fetchone()[0] or 0
print(f'  Total concepts:     {total}')
print(f'  Associations:       {assoc}')
print(f'  Knowledge areas:    {areas}')
print(f'  Avg confidence:     {avg_conf:.2f}')
conn.close()
"
fi
echo ""

# --- Cognitive Velocity (try API, fall back to DB) ---
echo -e "${BOLD}--- Cognitive Velocity (7 days) ---${NC}"
if [ "$API_COHERENT" = true ]; then
    ORIENT=$(curl -sf --max-time "$API_TIMEOUT" "${CURL_HEADERS[@]}" -- \
        "$API/pith_orient?time_window=7_days" 2>/dev/null)
    ORIENT_RENDERED=""
    if [[ -n "$ORIENT" ]]; then
        ORIENT_RENDERED=$(printf '%s' "$ORIENT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
where = d.get('where_am_i', {})
vel = where.get('cognitive_velocity', {})
print(f'  Sessions (7d):      {vel.get(\"sessions_in_window\", 0)}')
print(f'  Concepts created:   {vel.get(\"concepts_created_in_window\", 0)}')
print(f'  Concepts evolved:   {vel.get(\"concepts_evolved_in_window\", 0)}')
print(f'  Learning events:    {vel.get(\"learning_events_in_window\", 0)}')
print(f'  Growth rate:        {vel.get(\"knowledge_growth_rate\", 0)}/day')
print(f'  Trend:              {vel.get(\"trend\", \"unknown\")}')
" 2>/dev/null)
    fi
    if [[ -n "$ORIENT_RENDERED" ]]; then
        echo "$ORIENT_RENDERED"
    else
        echo "  (orientation API unavailable)"
    fi
else
    echo "  (requires a coherent running server — start with: pith start)"
fi
echo ""

# --- Session Summary (direct DB — works offline) ---
echo -e "${BOLD}--- Session Summary ---${NC}"
python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA busy_timeout=10000')
conn.row_factory = sqlite3.Row
total = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
ended = conn.execute(\"SELECT COUNT(*) FROM sessions WHERE status='ended'\").fetchone()[0]
recovered = conn.execute(\"SELECT COUNT(*) FROM sessions WHERE status='recovered'\").fetchone()[0]
active = conn.execute(\"SELECT COUNT(*) FROM sessions WHERE status='active'\").fetchone()[0]
avg_learn = conn.execute('SELECT AVG(learning_event_count) FROM sessions WHERE learning_event_count > 0').fetchone()[0]
max_learn = conn.execute('SELECT MAX(learning_event_count) FROM sessions').fetchone()[0]
print(f'  Total sessions:     {total}')
print(f'  Ended normally:     {ended}')
print(f'  Recovered (crash):  {recovered}')
print(f'  Still active:       {active}')
print(f'  Avg learning/sess:  {avg_learn:.1f}' if avg_learn else '  Avg learning/sess:  0')
print(f'  Max learning/sess:  {max_learn or 0}')
conn.close()
" 2>/dev/null
echo ""

# --- Concept Quality (direct DB) ---
echo -e "${BOLD}--- Concept Quality ---${NC}"
python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA busy_timeout=10000')
total = conn.execute(\"SELECT COUNT(*) FROM concepts WHERE status='active'\").fetchone()[0]
if total == 0:
    print('  No concepts yet — keep chatting!')
else:
    high_conf = conn.execute('SELECT COUNT(*) FROM concepts WHERE confidence >= 0.7').fetchone()[0]
    low_conf = conn.execute('SELECT COUNT(*) FROM concepts WHERE confidence < 0.3').fetchone()[0]
    orphans = conn.execute('''
        SELECT COUNT(*) FROM concepts c
        WHERE NOT EXISTS (SELECT 1 FROM associations WHERE source=c.id OR target=c.id)
    ''').fetchone()[0]
    areas = conn.execute('''
        SELECT knowledge_area, COUNT(*) as cnt
        FROM concepts WHERE knowledge_area IS NOT NULL
        GROUP BY knowledge_area ORDER BY cnt DESC LIMIT 5
    ''').fetchall()
    print(f'  High confidence (>=0.7): {high_conf}/{total} ({high_conf*100//total}%)')
    print(f'  Low confidence (<0.3):   {low_conf}/{total} ({low_conf*100//total}%)')
    print(f'  Orphan concepts:         {orphans}/{total} ({orphans*100//total}%)')
    print(f'  Top areas:')
    for area, cnt in areas:
        print(f'    {area}: {cnt}')
conn.close()
" 2>/dev/null
echo ""

# --- Environment ---
echo -e "${BOLD}--- Environment ---${NC}"
echo "  Platform:           $(uname -s) $(uname -m)"
echo "  Python:             $(python3 --version 2>/dev/null)"
echo "  Node.js:            $(node --version 2>/dev/null)"
echo "  Pith home:          $PITH_HOME"
echo "  Profile:            ${PITH_PROFILE:-not set}"
echo "  Data directory:     $DATA_DIR"
echo "  API origin:         $API_DISPLAY"
echo "  API profile:        ${API_PROFILE:-unknown}"
echo "  API DB identity:    $API_IDENTITY_STATUS"
echo "  Database:           $DB_PATH"
DB_SIZE=$(du -h "$DB_PATH" 2>/dev/null | cut -f1)
echo "  DB size:            ${DB_SIZE:-unknown}"
# Show capabilities if available
CAP_FILE="$PITH_HOME/.install_capabilities"
if [[ -f "$CAP_FILE" ]]; then
    echo "  Capabilities:"
    while IFS= read -r line; do
        echo "    $line"
    done < "$CAP_FILE"
fi
echo ""

echo -e "${BOLD}================================${NC}"
echo -e "${BOLD}  End of Report${NC}"
echo -e "${BOLD}================================${NC}"
echo ""
echo "Copy everything above and share with the Pith team."
echo ""
