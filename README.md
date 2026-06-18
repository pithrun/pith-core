# Pith — AI Agent Memory with Cognitive Governance

Pith is governed memory for AI agents that works across models: persistent project context, contradiction detection, durable learning, and checkpoints across sessions.

## Quick Start (5 minutes)

### Prerequisites
- **macOS developer preview:** Pith's initial public preview is macOS-first.
- **Python 3.10+:** on macOS arm64, the installer can provision a Pith-managed Python 3.12 runtime under `~/.pith/runtime/python` when no compatible Python is present. It does not replace or modify system Python.
- **Supported AI app:** verified launch workflows are Claude Desktop, Cursor, VS Code, and Codex.

### Install

```bash
curl -fsSL https://pith.run/install | bash
```

Older macOS machines may only have Apple's Python 3.9. In a normal interactive install, Pith will offer to install its own Python runtime under `~/.pith`. For noninteractive automation, opt in explicitly:

```bash
curl -fsSL https://pith.run/install | PITH_AUTO_PYTHON=1 bash
```

Windows support is not part of the initial developer preview.

The installer takes 2-5 minutes and handles everything:
1. Checks system requirements (OS, disk space, Python runtime)
2. Creates ~/.pith/ directory structure
3. Sets up a Python virtual environment with dependencies
4. Generates a secure API key
5. Asks which AI app surfaces to configure, then writes only the selected MCP/client files
6. Sets up auto-start with launchd on macOS
7. Schedules automatic backups
8. Runs a health check to verify everything works

### After Setup

**Required setup**

1. Open a new terminal window so your shell picks up the `pith` command.
2. Claude Desktop users: if you pasted Pith's instructions during install, this is already done. If you skipped it or need to redo it, run `pith protocol`, then paste the copied prompt into Claude Desktop `Settings > General > Instructions for Claude`.
3. Restart each configured AI client completely before testing it. For Claude Desktop, use Cmd+Q / Ctrl+Q, then reopen.

**Verification checks**

1. **Core install:** open a new terminal and run `pith status`. It should report `Health: OK (Pith)`.
2. **Claude Desktop:** open a fresh conversation and ask a normal project-context question, without reminding Claude to use Pith. Claude Desktop is best-effort configured: MCP tools are available, but first-turn lifecycle depends on model/tool invocation unless a deterministic adapter is installed. If you need proof, check for a fresh `pith_conversation_turn` entry in `~/Library/Logs/Claude/mcp-server-pith.log`.
3. **Codex:** confirm `~/.codex/AGENTS.md` exists and references `pith api conversation_turn`. If Codex was installed after Pith, rerun the installer or client configuration.
4. **VS Code:** restart VS Code, run **MCP: List Servers**, and confirm `pith` appears from `~/Library/Application Support/Code/User/mcp.json`. Then open Chat Diagnostics and confirm `~/.copilot/instructions/pith-cognitive-loop.instructions.md` is loaded for Agent Chat. If Agent Chat says Pith is not connected but can find the active server/config, treat that as a VS Code tool-selection/instruction-loading issue, not a failed Pith service.

### macOS Developer Preview Client Matrix

| Surface | Launch status | Notes |
|---|---|---|
| Terminal CLI | Verified | `pith status`, `pith health`, `pith start`, `pith logs`, `pith doctor`, `pith clients`, `pith support bundle`, `pith import`, backup, and restore are covered by the smoke path. |
| launchd service | Verified | macOS auto-start is installed by the native installer. |
| Claude Code | T0 deterministic when lifecycle hook is installed and verified | The `UserPromptSubmit` hook calls `conversation_turn` before Claude answers and injects returned context or an explicit degraded marker. Restart Claude Code after hook installation. |
| Claude Desktop | Verified MCP config + manual instructions step; best-effort configured | MCP config is automatic; if the instructions step was skipped during install, run `pith protocol` and paste it into Instructions for Claude. MCP availability means tools are available, not that first-turn lifecycle is deterministic. |
| Codex app | Verified config + HTTP/API-first lifecycle instructions | Codex should use `~/.pith/bin/pith api ...` for lifecycle calls; MCP remains available when healthy. `api-fallback` remains a legacy/recovery alias. |
| VS Code | Configurable MCP + Copilot instruction file; beta behavior not yet equivalent to Claude/Codex | User config lives at `~/Library/Application Support/Code/User/mcp.json`; project config lives at `~/.pith/pith-server/.vscode/mcp.json`; Copilot instruction lives at `~/.copilot/instructions/pith-cognitive-loop.instructions.md`. Agent Chat must be in Agent mode with Pith tools enabled, and the model may not call Pith automatically on every turn. |
| Cursor | MCP config template + Global/User Rule step | The installer writes `~/.cursor/mcp.json`, saves a Cursor rule snippet at `~/.pith/CURSOR_GLOBAL_RULE.txt`, and copies it to the clipboard on macOS. Cursor can see the Pith server, but current user testing shows Cursor does not automatically call Pith every turn from MCP config alone. Paste the snippet into Cursor Settings > Rules as a Global/User Rule, or use project `AGENTS.md`, until the installer has a verified automatic Cursor instruction flow. |

### CLI Commands

After installation, use the `pith` command:

```bash
pith status        # Check if Pith is running
pith health        # Check operational health/readiness
pith start         # Start the server
pith stop          # Stop the server
pith restart       # Restart the server
pith logs          # View server logs (follow mode)
pith logs snapshot # View bounded redacted logs
pith doctor        # Run read-only install diagnostics
pith clients       # Show detected/configured client surfaces
pith support bundle # Create a redacted support bundle
pith import        # Import conversation exports safely
pith api           # First-class local HTTP/API lifecycle calls
pith api-fallback  # Legacy/recovery alias for HTTP/API lifecycle calls
pith backup        # Create a WAL-safe database backup
pith restore       # Restore from most recent backup
pith update        # Update Pith to latest version
pith version       # Show version and system info
pith runtime status # Show Python runtime provenance
pith runtime repair # Reinstall a Pith-managed Python runtime
pith uninstall     # Remove Pith from this machine
```

### macOS Smoke Report

For support diagnostics, run this from the installed Pith folder:

```bash
scripts/macos-surface-smoke.sh
```

The report is read-only. It records app inventory, CLI inventory, config presence, Pith health, HTTP/API status, Claude MCP log tail, and an internal leak scan.

## How It Works

Pith runs as a local Python service with three connection modes for supported AI app surfaces:

- **Local HTTP/API lifecycle:** CLI-capable agents such as Codex can call `~/.pith/bin/pith api conversation_turn --stdin-json` before responding, then `checkpoint` or `session_end` at the right lifecycle boundary.
- **MCP tool bridge:** MCP-capable clients such as Claude Desktop and VS Code can connect through `pith_mcp.py` and call `pith_` tools directly when those tools are enabled by the client.
- **Terminal CLI:** You manage the local service with commands such as `pith status`, `pith health`, `pith start`, `pith logs`, `pith doctor`, `pith clients`, `pith support bundle`, `pith import`, and `pith backup`.

During conversations, your agent:

1. **Retrieves** relevant knowledge before responding
2. **Learns** decisions, patterns, and discoveries after each exchange
3. **Consolidates** knowledge through periodic reflection

Pith stores and serves your data locally. When an AI client uses retrieved context in a conversation, that client may send conversation content to its configured model provider under that client's settings and terms.

## Architecture

```
Codex / CLI Agent  <->  pith api command          <->  Pith API (FastAPI)  <->  SQLite
Claude / MCP App   <->  MCP Wrapper (pith_mcp.py) <->  Pith API (FastAPI)  <->  SQLite
VS Code Agent      <->  MCP tools when enabled     <->  Pith API (FastAPI)  <->  SQLite
```

The macOS developer preview is managed by launchd. Other operating system install paths are source/developer flows and are not part of the initial public preview.

## Benchmarks

Pith has score-bearing launch evidence for MemoryAgentBench / FactConsolidation and documented reportable evidence for LoCoMo-Plus official Cognitive all401. On LoCoMo-Plus official Cognitive all401, Pith records 100.00% accuracy over 401/401 rows. On MemoryAgentBench / FactConsolidation multi-hop, Pith records 95.0% Exact Match on the 6K lane and 68.0% Exact Match / 68.2 F1 on the 262K lane. As verified on June 12, 2026, those MAB scores are ahead of the published MAB / FactConsolidation multi-hop comparator scores.

For methodology, score terms, caveats, and evidence files, see the Pith benchmarks page: https://pith.run/benchmarks.

## Backing Up Your Pith

Your knowledge database contains all learned knowledge. Protect it with regular backups.

### One-Time Backup

```bash
pith backup
# Or directly:
bash scripts/backup/safe_backup.sh
```

Backups use SQLite backup API (WAL-safe, even while the server is running). Saved to `~/.pith/backups/` with timestamps.

> **Important:** Never copy `pith.db` directly — this can corrupt the file if WAL mode is active. Always use `pith backup`.

### Automated Backups

The installer configures automatic backups:
- **Automatic maintenance backups:** Creates a WAL-safe local backup while the server is running
- **Daily launchd backup:** Runs a local backup on macOS

Manual backups are also available with `pith backup`.

### Restoring from Backup

```bash
pith restore
```

This safely stops the server, copies the latest backup, verifies database integrity, and restarts.

## Upgrading from Previous Versions

If you had a previous Pith installation (Docker-based or earlier beta):
1. Back up your data with `pith backup` before uninstalling. If the old CLI is unavailable, stop Pith completely before copying database files.
2. Run `pith uninstall` (or delete ~/.pith/)
3. Run the new installer: `curl -fsSL https://pith.run/install | bash`
4. Restore with `pith restore` after installation.

## Troubleshooting

**Pith tools not showing in your MCP client?**
- Restart your client completely (Cmd+Q, not just close window)
- Check your MCP config file for the Pith server entry with the correct path
- Claude Desktop: `cat ~/Library/Application\ Support/Claude/claude_desktop_config.json`

**VS Code sees the config but says Pith is not connected?**
- Confirm you are using Agent mode, not plain Ask/Edit chat.
- Open **Configure Tools** in the chat input and enable the Pith MCP tools for the request.
- Run **MCP: List Servers** and start or restart the `pith` server if needed.
- Open Chat Diagnostics and confirm the Pith instruction file is loaded. VS Code MCP availability means the tools are available to the agent; it does not guarantee the model will call them on every turn.

**Cursor sees Pith but does not call it automatically?**
- Open Cursor Settings > Rules and add a Global/User Rule that tells Cursor to call Pith before substantive responses.
- The installer saves a starter rule at `~/.pith/CURSOR_GLOBAL_RULE.txt`.
- If you prefer project-local instructions, add the Pith cognitive loop to the project `AGENTS.md`.
- Treat `~/.cursor/mcp.json` as tool availability, not automatic lifecycle enforcement.

**API not responding?**
- Check status: `pith status`
- View logs: `pith logs`
- Restart: `pith restart`

**Port 8000 in use?**
- Check whether it is already Pith: `pith status`
- Inspect the listener: `lsof -nP -iTCP:8000 -sTCP:LISTEN`
- If it is a stale Pith process, run `pith restart`. If it is another app, stop that app or choose a different Pith port before starting Pith.

**Embeddings not available?**
- Check: `pith version` (shows embedding status)
- Intel Macs require Python 3.10-3.12 for embedding support
- Pith works without embeddings using TF-IDF search (slightly reduced quality)

**Prefer to manage Python yourself?**
- Run the public installer with `PITH_NO_AUTO_PYTHON=1` to disable automatic runtime provisioning.
- Run `curl -fsSL https://pith.run/install | PITH_AUTO_PYTHON=1 bash` for noninteractive macOS arm64 installs.
- If you are installing from an extracted release artifact instead of the public route, use `PITH_AUTO_PYTHON=1 bash scripts/install.sh`.

**Claude Desktop logs show a transport close?**
- `tools/list` or `resources/list` in `~/Library/Logs/Claude/mcp-server-pith.log` proves MCP startup.
- A 600-second idle transport close can be normal bridge lifecycle noise.
- If a fresh Claude prompt does not produce `pith_conversation_turn`, rerun `pith protocol`, paste into Instructions for Claude, save, and restart Claude.

## Runtime Notice

On macOS arm64, Pith may install a user-local CPython 3.12 runtime from the `astral-sh/python-build-standalone` project. The runtime URL and SHA-256 digest are pinned in `scripts/install.sh`; `pith version` and `pith runtime status` show the installed runtime provenance.

## Support

Pith is a free open-source developer preview.

- The local developer preview has no active Pith usage caps.
- Pricing or budget fields surfaced by diagnostics are reserved for future or explicitly enabled environments; they are not active in the public preview by default.
- Bugs and setup questions: use GitHub Issues.
- Security vulnerabilities: email security@pith.run.
- Founder-led onboarding or team pilots: email info@pith.run.

Support is best-effort during the developer preview.

## License

Apache 2.0 — see [LICENSE](LICENSE) for terms.
