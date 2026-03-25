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

# Resolve paths
PITH_HOME="${PITH_HOME:-$HOME/.pith}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Find database using shared resolution (mirrors app/profile.py)
source "$SCRIPT_DIR/resolve_db.sh" 2>/dev/null || {
    echo -e "${RED}ERROR: resolve_db.sh not found at $SCRIPT_DIR/${NC}"; exit 1;
}
if ! resolve_pith_db; then
    echo -e "${RED}No database found.${NC}"
    echo "  Checked: ~/pith-data/\${PITH_PROFILE:-default}/{pith,pith}.db"
    echo "  Checked: $PITH_HOME/data/{pith,pith}.db"
    echo "  Checked: $PROJECT_DIR/data/{pith,pith}.db"
    echo "  Make sure Pith has been started at least once."
    exit 1
fi

# Resolve Python and API key based on data location
if [[ "$DATA_DIR" == "$PITH_HOME/data" ]]; then
    PYTHON_EXE="$PITH_HOME/.venv/bin/python3"
    API_KEY_FILE="$PITH_HOME/config/api.key"
else
    PYTHON_EXE="python3"
    # Look for .env in project dir or next to DB
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        API_KEY_FILE="$PROJECT_DIR/.env"
    else
        API_KEY_FILE=""
    fi
fi

# Load API key
API_KEY=""
if [[ -f "$API_KEY_FILE" ]] && [[ "$API_KEY_FILE" == *.key ]]; then
    API_KEY=$(cat "$API_KEY_FILE" 2>/dev/null)
elif [[ -f "$API_KEY_FILE" ]]; then
    API_KEY=$(grep "PITH_API_KEY=" "$API_KEY_FILE" 2>/dev/null | cut -d'=' -f2)
fi
HEADERS=""
if [[ -n "$API_KEY" ]]; then
    HEADERS="-H X-API-Key:$API_KEY"
fi

API="http://localhost:8000"

echo ""
echo -e "${BOLD}================================${NC}"
echo -e "${BOLD}  Pith Beta Report${NC}"
echo -e "${BOLD}================================${NC}"
echo ""
echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# --- Health check (via API if server is running) ---
HEALTH=$(curl -sf $HEADERS "$API/health" 2>/dev/null)
if [ -z "$HEALTH" ]; then
    echo -e "${YELLOW}Pith API not responding — using direct DB access${NC}"
    echo ""
    API_ONLINE=false
else
    echo -e "${GREEN}Pith API: online${NC}"
    echo ""
    API_ONLINE=true
fi

# --- Stats (try API, fall back to direct DB) ---
echo -e "${BOLD}--- Pith Stats ---${NC}"
if [ "$API_ONLINE" = true ]; then
    STATS=$(curl -sf $HEADERS "$API/pith_stats")
    echo "$STATS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Total concepts:     {d.get(\"total_concepts\", 0)}')
print(f'  Associations:       {d.get(\"associations\", 0)}')
print(f'  Knowledge areas:    {d.get(\"knowledge_areas\", 0)}')
print(f'  Avg confidence:     {d.get(\"avg_confidence\", 0):.2f}')
print(f'  Avg stability:      {d.get(\"avg_stability\", 0):.2f}')
"
else
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
if [ "$API_ONLINE" = true ]; then
    ORIENT=$(curl -sf $HEADERS "$API/pith_orient?time_window=7_days")
    echo "$ORIENT" | python3 -c "
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
"
else
    echo "  (requires running server — start with: pith start)"
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
