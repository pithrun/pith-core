# Pith™ — Governed Persistent Memory for AI Agents

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)
[![v1.0.0](https://img.shields.io/badge/version-1.0.0-orange.svg)](CHANGELOG.md)

Your AI agent loses context every session. Decisions forgotten, patterns re-learned, contradictions ignored. Pith fixes this — not just by storing memories, but by **governing** them.

## Why Pith

Every AI memory system can store and retrieve. Pith does what they can't:

- **Session continuity** — Context carries across sessions. Session compacted? Pith re-injects what matters. No cold starts.
- **Contradiction detection** — When new knowledge conflicts with existing beliefs, Pith flags it and resolves it with auditable evidence trails.
- **Trust scoring** — Every piece of knowledge has a computed authority score based on provenance, not LLM self-report.
- **Temporal decay** — Old knowledge loses weight automatically. Your agent stops relying on stale information.
- **Belief lifecycle** — Knowledge moves through states (active → contested → resolved → superseded → stale) instead of being statically stored forever.

Pith works with **any MCP client**: Claude Desktop, Claude Code, Cursor, Windsurf, Cline, and VS Code.

Your data stays on your machine. Nothing is sent to external servers.

## Quick Start (5 minutes)

### Prerequisites
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **An MCP client** — [Claude Desktop](https://claude.ai/download), [Cursor](https://cursor.sh), [Windsurf](https://codeium.com/windsurf), or any MCP-compatible editor

### Install

Extract the Pith folder, then run the installer from inside it.

**Mac/Linux:**
```bash
cd /path/to/pith
bash scripts/install.sh
```

**Windows (PowerShell as Administrator):**
```powershell
cd C:\path\to\pith
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The installer takes 2-5 minutes and handles everything:
1. Checks system requirements (Python 3.10+, disk space)
2. Creates ~/.pith/ directory structure
3. Sets up Python virtual environment with dependencies
4. Generates a secure API key
5. Configures MCP clients (auto-detects installed clients)
6. Sets up auto-start (launchd on Mac, systemd on Linux, Task Scheduler on Windows)
7. Schedules automatic backups
8. Runs a health check to verify everything works

### After Setup

1. **Restart your MCP client** completely (Cmd+Q / Ctrl+Q, then reopen)
2. **Set your preferences** — open `USER_PREFERENCES.md` in a text editor, copy the code block contents into your client's settings
3. Open a new conversation and verify: type "Run pith status and show me the results"
4. Just chat normally — Pith learns automatically

> **Important:** Use the `.md` file with a text editor, not the `.docx` version. Copying from Word can introduce invisible formatting.

### CLI Commands

After installation, use the `pith` command:

```bash
pith status        # Check if Pith is running
pith start         # Start the server
pith stop          # Stop the server
pith restart       # Restart the server
pith logs          # View server logs (follow mode)
pith backup        # Create a WAL-safe database backup
pith restore       # Restore from most recent backup
pith update        # Update Pith to latest version
pith version       # Show version and system info
pith uninstall     # Remove Pith from this machine
```

## How It Works

Pith runs as a local Python service alongside your MCP client. During conversations, your agent:

1. **Orients** — retrieves relevant knowledge ranked by trust and relevance before responding
2. **Learns** — extracts decisions, patterns, and discoveries after each exchange
3. **Detects contradictions** — flags when new knowledge conflicts with existing beliefs
4. **Governs** — scores authority, decays stale knowledge, tracks belief state transitions

Between sessions, Pith runs a **REM cycle** — Reflection, Evaluation, Maintenance — that recalibrates confidence scores, resolves contradictions, and compounds knowledge overnight.

## Architecture

```
MCP Client  <->  MCP Wrapper (pith_mcp.py)  <->  Pith API (FastAPI)  <->  SQLite
```

Managed by launchd (macOS), systemd (Linux), or Task Scheduler (Windows).

## Backing Up Your Data

Your knowledge database contains all learned knowledge. Protect it with regular backups.

### One-Time Backup

```bash
pith backup
# Or directly:
bash scripts/backup/safe_backup.sh
```

Backups use SQLite backup API (WAL-safe, even while the server is running). Saved to `~/.pith/backups/` with timestamps.

> **Important:** Never copy `pith.db` directly — this can corrupt the file if WAL mode is active. Always use `pith backup`.

### Automated Backups (set up by installer)

The installer configures automatic backups:
- **Every 3 hours** (6am-11pm): Creates a local backup
- **Daily at 2:30am**: Syncs the latest backup to cloud storage

The sync script auto-detects Google Drive and iCloud on macOS.

### Restoring from Backup

```bash
pith restore
```

This safely stops the server, copies the latest backup, verifies database integrity, and restarts.

## Upgrading from Previous Versions

If you had a previous Pith installation (Docker-based or earlier beta):
1. Back up your data: `cp ~/.pith/data/pith.db ~/pith_backup.db`
2. Run `pith uninstall` (or delete ~/.pith/)
3. Run the new installer: `bash scripts/install.sh`
4. Copy your backup back: `cp ~/pith_backup.db ~/.pith/data/pith.db`

## Troubleshooting

**Pith tools not showing in your MCP client?**
- Restart your client completely (Cmd+Q, not just close window)
- Check config: `cat ~/Library/Application\ Support/Claude/claude_desktop_config.json`
- Verify Pith server entry exists with correct path

**API not responding?**
- Check status: `pith status`
- View logs: `pith logs`
- Restart: `pith restart`

**Port 8000 in use?**
- Find what is using it: `lsof -ti:8000`
- Kill it: `lsof -ti:8000 | xargs kill`
- Then: `pith start`

**Embeddings not available?**
- Check: `pith version` (shows embedding status)
- Intel Macs require Python 3.10-3.12 for embedding support
- Pith works without embeddings using TF-IDF search (slightly reduced quality)

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full results including CogGov-Bench scores, retrieval latency, and comparative analysis.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
