# Changelog

All notable changes to Pith are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2026-06-26

### Added
- Added bounded install-success telemetry after the macOS installer verifies a durable local Pith service.
- Added opt-out handling for `PITH_TELEMETRY_DISABLED=1`, `DO_NOT_TRACK=1`, and local-only install behavior.

### Changed
- Refreshed package and installer version metadata for v1.0.5.

## [1.0.4] - 2026-06-24

### Changed
- Removed private-beta wording from installer prompts and local-build output.
- Removed an internal ticket marker from the public installer source comment.

## [1.0.1] - 2026-05-29

### Added
- Public install path support for `https://pith.run/install`.
- Flexible installer port selection with `PITH_PORT`, `PITH_DEFAULT_PORT`, and `PITH_PORT_SCAN_MAX`.
- Local HTTP/API lifecycle guidance for Codex via `~/.pith/bin/pith api`.
- macOS surface smoke script for local launch verification.
- Pith-managed Python runtime support on macOS arm64 when no compatible Python is present.

### Changed
- Public repository payload now matches the public release package layout.
- Installer persists the selected API port in `~/.pith/.env` and propagates it into client/service configuration.
- Client support language now distinguishes verified surfaces from configurable or experimental surfaces.
- Benchmark copy now reflects current release evidence and avoids stale comparative claims.

### Fixed
- Installer no longer requires users to permanently free port `8000` when another local app owns it.
- Public docs no longer reference retired setup files or stale preference-copy instructions.

## [1.0.0] - 2026-03-25

### Added
- Governed persistent memory for AI agents via MCP.
- Local FastAPI service with SQLite storage.
- Belief lifecycle, contradiction detection, temporal currency, and provenance-aware authority scoring.
- CLI tooling for status, start, stop, restart, logs, backup, restore, update, version, and uninstall.
- WAL-safe backup and restore support.
- Initial public prerelease package.
