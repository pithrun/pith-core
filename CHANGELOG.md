# Changelog

All notable changes to Pith™ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-03-25

### Added
- **Governed persistent memory** for AI agents via MCP
- **5-state belief lifecycle**: active, contested, resolved, superseded, stale
- **Contradiction detection**: multi-phase pipeline with embedding-based similarity
- **Epistemic authority scoring**: provenance-based trust, not LLM self-report
- **Temporal currency decay**: knowledge loses weight as it ages
- **Knowledge area segmentation**: 35 dynamic domains with cross-domain retrieval
- **CogGov-Bench**: governance benchmark suite (69.0/100 composite, 16/21 scenarios)
- **Session compaction recovery**: automatic context re-injection when sessions compact
- **REM cycle**: overnight reflection, evaluation, and maintenance
- **Cross-platform support**: macOS, Linux, Windows
- **6 MCP clients**: Claude Desktop, Claude Code, Cursor, Windsurf, Cline, VS Code
- **CLI tooling**: `pith` command for status, start, stop, backup, restore, update
- **Automated backups**: WAL-safe SQLite backups every 3 hours + daily cloud sync
- **40+ MCP tools**: full API surface for search, reflection, knowledge graph ops
