# Pith — Persistent Cognitive Infrastructure

A persistent memory system for Claude. Your AI learns from every conversation — decisions, patterns, discoveries — and retrieves them when relevant.

## Quick Start (5 minutes)

### Prerequisites
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Claude Desktop** — [claude.ai/download](https://claude.ai/download)

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
3. Sets up Python virtual environment with dependencies4. Generates a secure API key
5. Configures MCP clients (Claude Desktop + others if installed)
6. Sets up auto-start (launchd on Mac, systemd on Linux, Task Scheduler on Windows)
7. Schedules automatic backups
8. Runs a health check to verify everything works

### After Setup

1. **Restart Claude Desktop** completely (Cmd+Q / Ctrl+Q, then reopen)
2. **Set your preferences** — open `USER_PREFERENCES.md` in a text editor, copy the code block contents into Claude Desktop Settings > Profile & Preferences > Save
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

Pith runs as a local Python service alongside Claude Desktop. An MCP (Model Context Protocol) wrapper connects Claude to the Pith API. During conversations, Claude:

1. **Retrieves** relevant knowledge before responding
2. **Learns** decisions, patterns, and discoveries after each exchange
3. **Consolidates** knowledge through periodic reflection

Your data stays on your machine. Nothing is sent to external servers.

## Architecture

```
Claude Desktop  <->  MCP Wrapper (pith_mcp.py)  <->  Pith API (FastAPI)  <->  SQLite
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

**Pith tools not showing in Claude Desktop?**
- Restart Claude Desktop completely (Cmd+Q, not just close window)
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

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.