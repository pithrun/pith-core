# Pith

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)
[![v1.0.1](https://img.shields.io/badge/version-1.0.1-orange.svg)](CHANGELOG.md)

Pith is a local cognitive layer for AI development environments. It gives supported agents persistent project memory: decisions, patterns, discoveries, and context that can be retrieved when they become relevant again.

## Why Pith

AI agents lose context across sessions, compaction, app restarts, and tool failures. Pith adds a governed local memory service so an agent can:

- Retrieve relevant past context before responding
- Learn decisions and durable patterns after an exchange
- Save task checkpoints for later resumption
- Track belief lifecycle, staleness, contradictions, and provenance
- Keep project memory local by default

Pith is not a hosted memory database. It runs on your machine, stores data in SQLite, and connects through local HTTP/API calls and MCP where the client supports it.

## Quick Start

```bash
curl -fsSL https://pith.run/install | bash
```

Then open a new terminal and run:

```bash
pith status
```

For client setup details, see [QUICKSTART.md](QUICKSTART.md).

## Supported Surfaces

Primary launch support is macOS arm64 with the local installer. The installer can provision a Pith-managed Python 3.12 runtime under `~/.pith/runtime/python` when no compatible Python is present.

| Surface | Status | Notes |
|---|---|---|
| Terminal CLI | Verified | `pith status`, `pith start`, `pith logs`, backup, restore, and runtime diagnostics are covered by the macOS smoke path. |
| launchd service | Verified | macOS auto-start is installed by the native installer. |
| Claude Desktop | Verified config plus manual instructions step | MCP config is automatic. If instructions were skipped, run `pith protocol`, paste the result into Claude Desktop instructions, save, and restart Claude. |
| Codex | Verified config plus HTTP/API lifecycle instructions | Codex should use `~/.pith/bin/pith api ...` for lifecycle calls. MCP remains available when healthy. |
| VS Code | Configurable MCP plus Copilot instruction file | Agent Chat behavior depends on VS Code tool selection and instruction loading. |
| Windsurf | Config template plus observed automatic invocation | Clean macOS smoke coverage is still pending. |
| Cursor | Config template plus Global/User Rule step | MCP config exposes tools; automatic lifecycle use depends on rules or project instructions. |
| Linux / Windows | Source or experimental path | Public launch smoke coverage is macOS-first. |

## CLI Commands

```bash
pith status         # Check if Pith is running
pith start          # Start the server
pith stop           # Stop the server
pith restart        # Restart the server
pith logs           # View server logs
pith api            # Local HTTP/API lifecycle calls
pith api-fallback   # Legacy/recovery alias
pith backup         # Create a WAL-safe database backup
pith restore        # Restore from most recent backup
pith update         # Update Pith to latest version
pith version        # Show version and runtime provenance
pith runtime status # Show managed Python runtime status
pith uninstall      # Remove Pith from this machine
```

## How It Works

Pith runs as a local Python service with three connection modes:

- Local HTTP/API lifecycle calls for CLI-capable agents such as Codex
- MCP tools for clients such as Claude Desktop and VS Code
- Terminal CLI commands for service management, backup, restore, and diagnostics

During conversations, an agent can retrieve relevant context, learn durable information, detect contradictions, and checkpoint work state. Pith stores and serves the local memory database. When an AI client uses retrieved context in a conversation, that client may send conversation content to its configured model provider under that client's settings and terms.

## Architecture

```text
Codex / CLI Agent  <->  pith api command          <->  Pith API (FastAPI)  <->  SQLite
Claude / MCP App   <->  MCP Wrapper (pith_mcp.py) <->  Pith API (FastAPI)  <->  SQLite
VS Code Agent      <->  MCP tools when enabled    <->  Pith API (FastAPI)  <->  SQLite
```

## Port Selection

Pith defaults to port `8000`, but the installer now supports flexible local ports. If `8000` is occupied by another app, the installer scans for an available port up to `8020` and persists the selected value in `~/.pith/.env`.

You can choose a specific port:

```bash
PITH_PORT=8123 curl -fsSL https://pith.run/install | bash
```

Or widen the automatic scan range:

```bash
PITH_PORT_SCAN_MAX=8050 curl -fsSL https://pith.run/install | bash
```

## Backups

Use `pith backup` for WAL-safe backups. Do not copy `pith.db` directly while the server is running.

```bash
pith backup
pith restore
```

Backups are stored under `~/.pith/backups/` by default.

## Benchmarks

The current public-release build reports `62.9/100` on the internal CogGov-Bench full suite. Treat this as release evidence, not a broad marketing claim, until the public benchmark ledger and reproduction instructions are finalized. See [BENCHMARKS.md](BENCHMARKS.md).

## Security

Pith binds locally by default, uses API-key authentication for local endpoints, and stores memory data on the user's machine. Report security issues using [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
