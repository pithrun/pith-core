# Pith Quick Start

Install Pith, verify the local service, then connect your AI client.

## 1. Install

### macOS

```bash
curl -fsSL https://pith.run/install | bash
```

The installer sets up `~/.pith`, creates a virtual environment, configures the local service, generates an API key, writes selected client configuration, and runs a health check.

If you need a non-default port:

```bash
PITH_PORT=8123 curl -fsSL https://pith.run/install | bash
```

If port `8000` is occupied, the installer can automatically choose a free port in the configured scan range and persist it.

### Windows PowerShell

```powershell
Invoke-WebRequest https://github.com/pithrun/pith-core/releases/latest/download/install.ps1 -UseBasicParsing -OutFile install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The Windows installer verifies the server ZIP checksum, provisions a managed Python runtime when needed, keeps its virtual environment on a short external path, configures Task Scheduler, and runs a health check.

## 2. Verify

Open a new terminal window so your shell picks up the `pith` command:

```bash
pith status
```

A healthy install should report that Pith is running.

## 3. Connect Your Client

### Claude Desktop

Restart Claude Desktop completely. If you skipped the instructions step during install, run:

```bash
pith protocol
```

Paste the copied instructions into Claude Desktop settings, save, quit Claude completely, and reopen it.

### Codex

Confirm the platform-specific Codex `AGENTS.md` contains the Pith cognitive loop and references `pith api conversation_turn --stdin-json`. If Codex was installed after Pith, rerun the installer or the client configuration step.

### VS Code, Cursor, and Windsurf

The installer writes MCP/client configuration where possible. Restart the app after installation. Some clients require an additional global rule or instruction file before the model reliably calls Pith every turn.

## 4. Use It

Use your AI client normally. A configured agent should retrieve Pith context before substantive responses and learn durable decisions after meaningful exchanges.

For manual lifecycle calls, use:

```bash
~/.pith/bin/pith api conversation_turn --stdin-json
~/.pith/bin/pith api checkpoint --stdin-json
~/.pith/bin/pith api session_end --stdin-json
```

## Common Commands

```bash
pith status
pith logs
pith restart
pith backup
pith version
```

## Help

- Full docs: [README.md](README.md)
- Security reporting: [SECURITY.md](SECURITY.md)
- Issues: [github.com/pithrun/pith-core/issues](https://github.com/pithrun/pith-core/issues)
