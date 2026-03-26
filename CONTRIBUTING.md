# Contributing to Pith™

Pith is an Apache 2.0 open-source project. Contributions are welcome — bug reports, documentation improvements, new MCP client support, and architecture discussion.

## Contributor License Agreement (CLA)

All contributors must sign our [Individual Contributor License Agreement](https://gist.github.com/esteyangandrew/ee6d6cbc74646c74f97f4cec4d970429) before we can merge your pull request. The CLA is a lightweight agreement that confirms you have the right to contribute your code and that your contributions are licensed under Apache 2.0.

When you open your first PR, the CLA assistant bot will comment with instructions. You only need to sign once — it covers all future contributions to any Pithrun repository.

## Quick orientation

Pith is an MCP server that adds governed persistent memory to AI agents. The entry point is `pith_mcp.py` (the MCP bridge). Core logic lives in `app/`. The benchmark suite is `app/coggov_bench.py`.

```
pith-core/
├── pith_mcp.py          # MCP server entry point
├── app/
│   ├── session.py       # Core conversation_turn / session logic
│   ├── retrieval.py     # TF-IDF + embedding + graph-walk retrieval
│   ├── contradiction.py # Embedding-based contradiction detection
│   ├── coggov_bench.py  # CogGov-Bench evaluation suite
│   └── config.py        # All tunable constants
├── scripts/
│   ├── install.sh       # Installer (Mac/Linux)
│   ├── install.ps1      # Installer (Windows)
│   └── configure_clients.py  # MCP client auto-configuration
├── migrations/          # SQLite schema migrations
└── data/                # Seed data, templates
```

## Development setup

**Requirements:** Python 3.12 exactly. The test suite enforces this.

```bash
git clone https://github.com/pithrun/pith-core
cd pith-core

# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key (required for embedding-based features)
# Canonical location: ~/.pith/.env — do NOT use pith-core/.env
mkdir -p ~/.pith
echo 'PITH_API_KEY=your-openai-key' >> ~/.pith/.env
```

> **Note on API key location:** `PITH_API_KEY` lives in `~/.pith/.env`, not in the project directory. The server loads it from there at startup. Keep it out of the repo.

## Running tests

```bash
# Full suite (3,528+ tests, ~60s)
PYTHONPATH=. .venv/bin/pytest tests/ -x

# Fast smoke tests only
PYTHONPATH=. .venv/bin/pytest tests/ -m smoke -x

# Single file
PYTHONPATH=. .venv/bin/pytest tests/test_session.py -x -v
```

Tests require Python 3.12. Running with a different version will fail at conftest import.

## Running the benchmark

```bash
# CogGov-Bench (requires an active pith.db with real usage data)
PYTHONPATH=. python3 -c "
import sqlite3
from app.coggov_bench import run_coggov_bench
conn = sqlite3.connect('path/to/pith.db')
conn.row_factory = sqlite3.Row
result = run_coggov_bench(conn)
print(f'Score: {result.composite_score}/100')
"
```

## What's in scope

- Bug fixes with a failing test that demonstrates the issue
- New MCP client support in `scripts/configure_clients.py`
- Documentation improvements (README, QUICKSTART, BENCHMARKS)
- Performance improvements with benchmark evidence
- Retrieval quality improvements (measured against existing test suite)
- Architecture discussion in Issues

## What's out of scope

- Changes to tuning parameters in `config.py` (retrieval weights, contradiction thresholds, authority caps, decay constants). These are calibrated against the production database and are not open for external tuning. Open an Issue to discuss if you believe a parameter is wrong.
- Removal of governance features (belief lifecycle, contradiction detection, currency decay). These are core to the product.

## Submitting a pull request

1. Fork the repo and create a branch: `git checkout -b fix/my-fix`
2. Make your change with a test (new tests go in `tests/`)
3. Run the full suite: `PYTHONPATH=. .venv/bin/pytest tests/ -x`
4. Sign the CLA when prompted by the bot (first-time contributors only)
5. Open a PR against `main` with a clear description of what and why
6. Reference any related Issue numbers

PR titles should follow: `fix: ...`, `feat: ...`, `docs: ...`, `perf: ...`

## Reporting bugs

Open a GitHub Issue with:
- Pith version (`pith status` output)
- OS and Python version
- Minimal reproduction steps
- Expected vs. actual behavior

For security vulnerabilities, **do not open a public Issue**. Email `security@pith.run` instead. See [SECURITY.md](SECURITY.md).

## Code style

- Python: PEP 8, no line length limit enforced but prefer under 100 chars
- No new external dependencies without discussion in an Issue first
- SQLite migrations go in `migrations/` as numbered files
- All new MCP tools must be listed in `pith_mcp.py`'s tool registry

## License

By contributing, you agree that your contributions are licensed under the Apache 2.0 License. All contributors must have a signed [CLA](https://gist.github.com/esteyangandrew/ee6d6cbc74646c74f97f4cec4d970429) on file.
