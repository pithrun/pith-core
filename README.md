<!-- mcp-name: io.github.pithrun/pith -->
# Pith — AI Agent Memory with Cognitive Governance

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://python.org)
![Platform: macOS developer preview](https://img.shields.io/badge/Platform-macOS%20developer%20preview-lightgrey.svg)

> Governed memory for AI agents that works across models: persistent project context, contradiction detection, durable learning, and checkpoints across sessions.

## Why Pith?

AI agents are stateless. Every conversation starts from zero. Pith fixes that.

Pith runs as a local memory service for supported AI development workflows. It captures what your agent learns during conversations — decisions you've made, patterns discovered, principles established — and retrieves that knowledge when it's relevant again. Over time, your AI develops genuine continuity: it remembers your architecture decisions, your preferences, your project context.

**The AI stack has three layers. Pith focuses on the cognitive governance layer.**

| Layer | Who's building it | Without it |
|---|---|---|
| Security Governance — controls what agents *can access* | Microsoft, Rubrik, Cisco | Unauthorized access |
| Storage & Recall — stores what agents *have seen* | Mem0, Zep, Letta, Anthropic | Amnesia |
| **Cognitive Governance — controls what agents *should believe*** | **Pith** | **Stale beliefs, contradictions, drift** |

## How Pith compares to memory tools (source-checked June 10, 2026)

| Focus | Pith | Mem0 | Zep / Graphiti | Letta |
|---|---|---|---|---|
| Public positioning | Governed local project memory for AI agents: decisions, corrections, contradictions, and trusted context carried across sessions | Universal, self-improving memory layer for LLM applications | Temporal knowledge graph and context graph platform for agents | Platform and tools for stateful agents that remember and learn |
| Change over time | Tracks belief lifecycle, contradictions, decay, and authority so agents can revise stale project knowledge instead of merely retrieving it | Persistent memory, user/context adaptation, and memory operations | Temporal facts, provenance, and evolving graph context | Persistent agent state, memory blocks, and memory-first agent workflows |
| Local / self-hosted path | Local SQLite-backed developer preview; your memory database stays on your machine while supported clients connect through local API/MCP workflows | Open-source package and self-hosted REST API docs | Graphiti open source; Zep platform available separately | Open-source project with local and hosted deployment paths |
| Launch comparison stance | Best fit when the problem is governed local project continuity: preserving decisions and corrections, resolving conflicting knowledge, and keeping stale memory from steering future work | Adjacent memory layer | Adjacent temporal graph memory | Adjacent stateful-agent platform |

Sources: [Mem0 docs](https://docs.mem0.ai/introduction), [Mem0 OSS REST API](https://docs.mem0.ai/open-source/features/rest-api), [Graphiti docs](https://help.getzep.com/graphiti/getting-started/overview), [Letta](https://www.letta.com/), and [Letta GitHub](https://github.com/letta-ai/letta).


## Benchmarks

Pith has score-bearing launch evidence for MemoryAgentBench / FactConsolidation and documented reportable evidence for LoCoMo-Plus official Cognitive all401. On LoCoMo-Plus official Cognitive all401, Pith records `100.00%` accuracy over `401/401` rows. On MemoryAgentBench / FactConsolidation multi-hop, Pith records `95.0%` Exact Match on the 6K lane and `68.0%` Exact Match / `68.2` F1 on the 262K lane. As verified on June 12, 2026, those MAB scores are ahead of the published MAB / FactConsolidation multi-hop comparator scores.

For methodology, score terms, caveats, and evidence files, see the [Pith benchmarks page](https://pith.run/benchmarks).

## Demo

![Pith terminal demo — install, stats, status, cogOS closing card](demo/demo.gif)

> Install → knowledge base stats → server status. 12 seconds. [View the cast file](demo/demo.cast) to replay locally with `asciinema play demo/demo.cast`.

## Quick Start

### Prerequisites

- **macOS developer preview** — Pith's initial public preview is macOS-first.
- **Python 3.10+** — on macOS arm64, the installer can provision a Pith-managed Python 3.12 runtime inside `~/.pith` if no compatible Python is present. It does not replace or modify system Python.
- **A supported AI app** — verified launch workflows are Claude Desktop, Cursor, VS Code, and Codex.

### macOS Developer Preview Client Matrix

| Surface | Launch status | Notes |
|---|---|---|
| Terminal CLI | Verified | `pith status`, `pith health`, `pith start`, `pith logs`, `pith doctor`, `pith clients`, `pith support bundle`, `pith import`, and backups run locally. |
| launchd service | Verified | macOS auto-start is installed by the native installer. |
| Claude Desktop | Verified config + manual instructions step | MCP config is written automatically; if you did not complete the instructions step during install, run `pith protocol` and paste into Claude Desktop `Settings > General > Instructions for Claude`. Once configured, users should not have to remind Claude to use Pith. |
| Codex app | Verified config + HTTP/API-first lifecycle instructions | Uses `~/.pith/bin/pith api ...` for lifecycle calls; MCP remains available for richer tools when healthy. `api-fallback` remains a legacy/recovery alias. |
| VS Code | Configurable MCP + Copilot instruction file; beta behavior not yet equivalent to Claude/Codex | User config lives at `~/Library/Application Support/Code/User/mcp.json`; project config lives at `~/.pith/pith-server/.vscode/mcp.json`; Copilot instruction lives at `~/.copilot/instructions/pith-cognitive-loop.instructions.md`. Agent Chat must be in Agent mode with Pith tools enabled, and the model may not call Pith automatically on every turn. |
| Cursor | MCP config template + Global/User Rule step | The installer writes `~/.cursor/mcp.json`, saves a Cursor rule snippet at `~/.pith/CURSOR_GLOBAL_RULE.txt`, and copies it to the clipboard on macOS. Cursor can see the Pith server, but current user testing shows Cursor does not automatically call Pith every turn from MCP config alone. Paste the snippet into Cursor Settings > Rules as a Global/User Rule, or use project `AGENTS.md`, until the installer has a verified automatic Cursor instruction flow. |

### Install (5 minutes)

Pith is launching as a macOS-first developer preview.

```bash
curl -fsSL https://pith.run/install | bash
```

Older macOS machines may only have Apple's Python 3.9. In a normal interactive install, Pith will offer to install its own Python runtime under `~/.pith`. For noninteractive automation, opt in explicitly:

```bash
curl -fsSL https://pith.run/install | PITH_AUTO_PYTHON=1 bash
```

Windows support is not part of the initial developer preview.

The installer handles everything: Python venv, dependencies, API key generation, AI app surface selection, MCP client configuration, auto-start setup, and health verification.

### After Install

**Required setup**

1. Open a new terminal window so your shell picks up the `pith` command.
2. Claude Desktop users: if you pasted Pith's instructions during install, this is already done. If you skipped it or need to redo it, run `pith protocol`, then paste the copied prompt into Claude Desktop `Settings > General > Instructions for Claude`. The same prompt is saved at `~/.pith/SYSTEM_PROMPT.md`.
3. Restart each configured AI client completely before testing it. For Claude Desktop, use Cmd+Q / Ctrl+Q, then reopen.

**Verification checks**

1. **Core install:** run `pith status` and confirm it reports `Health: OK (Pith)`.
2. **Claude Desktop:** open a fresh conversation and ask a normal project-context question, without reminding Claude to use Pith. A working setup should call Pith automatically or use retrieved Pith context. If needed, check `~/Library/Logs/Claude/mcp-server-pith.log` for a new `pith_conversation_turn` call.
3. **Codex:** confirm `~/.codex/AGENTS.md` exists and references `pith api conversation_turn`. If Codex was installed after Pith, rerun the installer or client configuration.
4. **VS Code:** restart VS Code, run **MCP: List Servers**, and confirm `pith` appears from `~/Library/Application Support/Code/User/mcp.json`. Open Chat Diagnostics and confirm `~/.copilot/instructions/pith-cognitive-loop.instructions.md` is loaded for Agent Chat. If Agent Chat says Pith is not connected but can find the active server/config, treat that as a VS Code tool-selection/instruction-loading issue, not a failed Pith service.

### Verify

```bash
pith status    # Should show "running" + health OK
pith stats     # Shows concept count, knowledge areas, DB size
scripts/macos-surface-smoke.sh  # macOS surface inventory and smoke report
```

## How It Works

Pith runs as a local Python service with three connection modes:

- **Local HTTP/API lifecycle:** CLI-capable agents such as Codex can call `~/.pith/bin/pith api conversation_turn --stdin-json` before responding, then `checkpoint` or `session_end` at the right lifecycle boundary.
- **MCP tool bridge:** MCP-capable clients such as Claude Desktop and VS Code can connect through `pith_mcp.py` and call `pith_` tools directly when those tools are enabled by the client.
- **Terminal CLI:** You manage the local service with commands such as `pith status`, `pith health`, `pith start`, `pith logs`, `pith doctor`, `pith clients`, `pith support bundle`, `pith import`, and `pith backup`.

During conversations, your agent:

1. **Retrieves** relevant knowledge before responding
2. **Learns** decisions, patterns, and discoveries after each exchange
3. **Consolidates** knowledge through periodic reflection and maintenance

```
Codex / CLI Agent  ←→  pith api command  ←→  Pith API (FastAPI)  ←→  SQLite
Claude / MCP App   ←→  MCP Bridge        ←→  Pith API (FastAPI)  ←→  SQLite
VS Code Agent      ←→  MCP tools enabled ←→  Pith API (FastAPI)  ←→  SQLite
                       (pith_mcp.py)         (localhost:8000)        (pith.db)
```

Pith stores and serves your data locally at `~/pith-data/`. When an AI client uses retrieved context in a conversation, that client may send conversation content to its configured model provider under that client's settings and terms.

## CLI Reference

| Command | Description |
|---------|-------------|
| `pith start` | Start the Pith server |
| `pith stop` | Stop the server |
| `pith restart` | Restart the server |
| `pith status` | Check if running + health |
| `pith health` | Check operational health/readiness |
| `pith stats` | Quick knowledge summary |
| `pith logs` | Tail server logs (follow mode) |
| `pith logs snapshot` | Print a bounded redacted log snapshot |
| `pith doctor` | Run read-only install and service diagnostics |
| `pith clients` | Show detected and configured client surfaces |
| `pith support bundle` | Create a redacted support bundle |
| `pith import` | Import conversation exports safely |
| `pith api` | First-class local HTTP/API lifecycle calls for CLI-capable agents |
| `pith api-fallback` | Legacy/recovery alias for the local HTTP/API lifecycle path |
| `pith backup` | Create a WAL-safe database backup |
| `pith restore` | Restore from most recent backup |
| `pith update` | Update Pith to latest version |
| `pith version` | Show version and system info |
| `pith report` | Full diagnostic report |
| `pith profiles` | List available profiles |
| `pith maintenance run` | Run maintenance cycle |
| `pith uninstall` | Remove Pith completely |

All commands support `--profile NAME` for multi-profile setups.

## Backing Up Your Data

Pith protects your knowledge base with both automatic and manual backups.

**Automatic backups** run every 6 hours as part of the maintenance cycle. The server creates a WAL-safe copy at `~/pith-data/{profile}/pith_backup.db` using Python's `sqlite3.Connection.backup()` — no server downtime, no WAL corruption risk. You can check backup health anytime:

```bash
curl -s http://localhost:8000/health/backup | python3 -m json.tool
```

The `/health/backup` endpoint returns status (`healthy`, `warning`, `critical`), backup age, integrity check, and concept count — useful for monitoring dashboards.

**Optional webhook alerts:** Set the `PITH_BACKUP_ALERT_WEBHOOK_URL` environment variable to receive alerts when backups fail. Alerts use an escalating cooldown (2h → 6h → 24h) to avoid notification fatigue. Works with Slack, Discord, or any webhook endpoint.

**Manual backups** give you timestamped snapshots:

```bash
pith backup                    # WAL-safe, works while server is running
pith restore                   # Restore from most recent backup
pith restore ~/my_backup.db    # Restore from specific file
```

Manual backups are saved to `~/pith-data/{profile}/backups/` with timestamps. The macOS installer also sets up automated daily backups at 2:00 AM via launchd.

> **Important:** Never copy `pith.db` directly while the server is running — use `pith backup` or let the automatic backup handle it.

## Multi-Profile Support

Pith supports multiple isolated instances for different contexts:

```bash
pith --profile work start      # Separate knowledge base for work
pith --profile personal start  # Separate knowledge base for personal
pith profiles                  # List all profiles with DB sizes
```

Each profile gets its own database, indexes, backups, and logs under `~/pith-data/{name}/`.

## Troubleshooting

**Pith tools not showing in your MCP client?**
- Restart your client completely (Cmd+Q, not just close window)
- Check your MCP config file for the Pith server entry with the correct path
- Claude Desktop: `cat ~/Library/Application\ Support/Claude/claude_desktop_config.json`
- VS Code: `cat ~/Library/Application\ Support/Code/User/mcp.json`; then run **MCP: List Servers** and start `pith`
- Cursor: check `~/.cursor/mcp.json`

**Cursor sees Pith but does not call it automatically?**
- Open Cursor Settings > Rules and add a Global/User Rule that tells Cursor to call Pith before substantive responses.
- The installer saves a starter rule at `~/.pith/CURSOR_GLOBAL_RULE.txt` and copies it to the clipboard on macOS.
- If you prefer project-local instructions, add the Pith cognitive loop to the project `AGENTS.md`.
- Treat `~/.cursor/mcp.json` as tool availability, not automatic lifecycle enforcement.

**VS Code sees the config but says Pith is not connected?**
- Confirm you are using Agent mode, not plain Ask/Edit chat.
- Open **Configure Tools** in the chat input and enable the Pith MCP tools for the request.
- Run **MCP: List Servers** and start or restart the `pith` server if needed.
- Open Chat Diagnostics and confirm the Pith instruction file is loaded. VS Code MCP availability means the tools are available to the agent; it does not guarantee the model will call them on every turn.

**Claude Desktop lists Pith but does not call it every turn?**
- Run `pith protocol` and paste the prompt into Claude Desktop `Settings > General > Instructions for Claude`.
- Restart Claude Desktop and start a fresh conversation.
- `tools/list` or `resources/list` in `~/Library/Logs/Claude/mcp-server-pith.log` proves MCP startup. A 600-second idle transport close can be normal bridge lifecycle noise. Missing `pith_conversation_turn` after a fresh prompt usually means the custom instructions are not active.

**API not responding?**
- Check status: `pith status`
- Check port: `lsof -ti:8000`
- Restart: `pith restart`

**Port 8000 in use?**
- Check whether it is already Pith: `pith status`
- Inspect the listener: `lsof -nP -iTCP:8000 -sTCP:LISTEN`
- If it is a stale Pith process, run `pith restart`. If it is another app, stop that app or choose a different Pith port before starting Pith.

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

Copyright (c) 2026 Andrew Estey-Ang.
