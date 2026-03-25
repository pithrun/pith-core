# Pith™ Quick Start

Get persistent memory for your AI agent in under 5 minutes.

> **Primary support:** Claude Desktop. Cursor, Windsurf, and Zed also work — see step 1 for client-specific setup.

## 1. Install

```bash
# Mac/Linux
cd /path/to/pith
bash scripts/install.sh
```

The installer handles Python venv, dependencies, MCP config, auto-start, and backups.

## 2. Connect Your MCP Client

**Claude Desktop:** Quit completely (Cmd+Q / Ctrl+Q), then reopen. Pith tools appear automatically.

**Cursor / Windsurf / Zed:** The installer writes an MCP config entry. Restart your editor — Pith appears as a connected tool server. See `scripts/configure_clients.py` for manual configuration.

## 3. Verify

In any connected MCP client, ask:

```
Run pith status and show me the results
```

You should see the server running with 0 concepts (fresh install).

## 4. Set Your Preferences

Open `USER_PREFERENCES.md` in a text editor. Copy the code block into your client's system prompt or profile:
- **Claude Desktop:** Settings → Profile & Preferences → Save
- **Cursor / Windsurf / Zed:** Add to your global system prompt in editor AI settings

## 5. Start Using Pith

Just chat normally. Pith learns automatically from every conversation. There's nothing special you need to do — Claude will call Pith tools behind the scenes.

## Core API (3 functions)

If you're building on Pith or want to understand what's happening under the hood:

| Function | What it does |
|----------|-------------|
| `pith_conversation_turn` | Retrieves relevant context AND learns from the previous exchange in a single call |
| `pith_checkpoint` | Saves/loads execution state for cross-session task resumption |
| `pith_session_end` | Closes a session and triggers consolidation |

These three functions cover 90%+ of typical usage. The full API exposes 40+ tools for search, reflection, knowledge graph operations, and more — see the [full documentation](README.md).

## Common Commands

```bash
pith status     # Is it running?
pith logs       # What's happening?
pith backup     # Protect your data
pith restart    # Fix most issues
```

## Need Help?

- Full docs: [README.md](README.md)
- Troubleshooting: see README.md § Troubleshooting
- File an issue: [github.com/pithrun/pith-core/issues](https://github.com/pithrun/pith-core/issues)