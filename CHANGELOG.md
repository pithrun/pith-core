# Changelog

All notable changes to Pith are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.7] - 2026-10-01

### Added
- Adds a Windows x64 developer preview with a checksum-verified PowerShell installer and dedicated server ZIP.
- Adds managed Windows Python provisioning, a short external virtual-environment path, Task Scheduler auto-start and backups, and platform-aware client configuration.
- Adds hosted Windows release proof for deep user paths, semantic embeddings, installed Claude hook lifecycle, repeated start, restart, uninstall, cleanup, and post-uninstall refusal.

### Changed
- Expands public install and quick-start guidance from macOS-only to scoped macOS and Windows developer previews.
- Requires release validation to check the Windows installer scripts, ZIP checksum, Claude extension package checksum, and archive cache hygiene.

### Fixed
- Makes repeated Windows starts idempotent and keeps uninstall terminal even while Windows releases owned files and scheduled tasks.
- Waits for the owned child process to be reaped after forced launcher shutdown, eliminating an intermittent end-of-input failure.

## [1.0.6] - 2026-07-07

### Changed
- Publishes the current public developer preview as macOS-only.
- Removes unsupported Windows installer and zip assets from the public release payload.
- Updates public install copy and packaging metadata so the release does not imply Windows support.

### Fixed
- Rebuilds the server tarball without macOS AppleDouble metadata files.
- Adds release-build safeguards so generated Python caches and macOS metadata do not ship in the tarball.

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
