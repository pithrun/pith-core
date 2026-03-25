# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Pith™, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email: **security@pith.run**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide an initial assessment within 7 days.

## Security Architecture

Pith runs entirely on your local machine. Your knowledge data never leaves your device.

- **Local-only by default** — No external API calls for storage or retrieval
- **SQLite with WAL mode** — Database integrity maintained even during crashes
- **API key authentication** — Local server requires API key for all requests
- **No telemetry** — Pith does not collect usage data or phone home
