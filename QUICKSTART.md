# Pith Quick Start

Get persistent memory for Claude in under 5 minutes.

## 1. Install

```bash
# Mac/Linux
cd /path/to/pith
bash scripts/install.sh
```

The installer handles Python venv, dependencies, MCP config, auto-start, and backups.

## 2. Restart Claude Desktop

Quit Claude Desktop completely (Cmd+Q / Ctrl+Q), then reopen it. Pith tools appear automatically.

## 3. Verify

Open a new conversation and type:

```
Run pith status and show me the results
```

You should see the server running with 0 concepts (fresh install).

## 4. Set Your Preferences

Open `USER_PREFERENCES.md` in a text editor. Copy the code block contents into:
**Claude Desktop** → Settings → Profile & Preferences → Save

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
- File an issue: [github.com/pith-ai/pith-core/issues](https://github.com/pith-ai/pith-core/issues)