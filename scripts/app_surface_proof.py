#!/usr/bin/env python3
"""Capture and verify signed Windows app-surface release proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ops.app_surface_proof import (  # noqa: E402
    REQUIRED_SURFACE_VARIANTS,
    AppSurfaceProofError,
    ProofExpectations,
    capture_proof_row,
    load_proof_ledger,
    pair_key,
    validate_proof_ledger,
)

MAX_METADATA_BYTES = 64 * 1024
MAX_RAW_PROOF_BYTES = 64 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _release_key() -> tuple[str, str]:
    key = os.environ.get("PITH_RELEASE_PROOF_KEY", "")
    key_id = os.environ.get("PITH_RELEASE_PROOF_KEY_ID", "")
    if not key or not key_id:
        raise AppSurfaceProofError("attestation_missing", "Release proof key and key ID are required.")
    return key, key_id


def _read_json_file(path_value: str) -> object:
    path = Path(path_value).expanduser()
    if path.is_symlink():
        raise AppSurfaceProofError("metadata_symlink_rejected", "Metadata symlinks are not accepted.")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AppSurfaceProofError("metadata_unreadable", "Metadata file is not readable.") from exc
    if size > MAX_METADATA_BYTES:
        raise AppSurfaceProofError("metadata_oversized", "Metadata file exceeds the size limit.")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AppSurfaceProofError("metadata_invalid_json", "Metadata file is not valid JSON.") from exc


def _read_stdin_json() -> object:
    raw = sys.stdin.read(MAX_RAW_PROOF_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_RAW_PROOF_BYTES:
        raise AppSurfaceProofError("proof_oversized", "Standard input exceeds the proof size limit.")
    try:
        return json.loads(raw, parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AppSurfaceProofError("proof_invalid_json", "Standard input is not one valid JSON object.") from exc


def _host_versions(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise AppSurfaceProofError("invalid_host_version", "Host versions use SURFACE/VARIANT=VERSION.")
        pair, version = value.split("=", 1)
        if not pair or not version or pair in output:
            raise AppSurfaceProofError("invalid_host_version", "Host version entries must be unique and non-empty.")
        output[pair] = version
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture or verify signed Windows app-surface proof.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Read one raw connection proof from stdin and update a ledger.")
    capture.add_argument("--metadata", required=True)
    capture.add_argument("--ledger", required=True)
    verify = subparsers.add_parser("verify", help="Verify a complete proof ledger.")
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--expected-proof-run-id", required=True)
    verify.add_argument("--expected-artifact-sha256", required=True)
    verify.add_argument("--expected-package-version", required=True)
    verify.add_argument("--expected-target-os-build", required=True)
    verify.add_argument("--expected-host-version", action="append", default=[], metavar="SURFACE/VARIANT=VERSION")
    release = subparsers.add_parser(
        "verify-release",
        help="Verify a signed ledger against the exact release ZIP and version tag.",
    )
    release.add_argument("--ledger", required=True)
    release.add_argument("--artifact", required=True)
    release.add_argument("--tag", required=True)
    return parser


def _verify_release(args: argparse.Namespace, key: str, key_id: str) -> dict[str, object]:
    tag_match = re.fullmatch(r"v([0-9]+\.[0-9]+\.[0-9]+(?:-rc[0-9]+)?)", args.tag)
    if not tag_match:
        raise AppSurfaceProofError("invalid_release_tag", "Release tag is not a supported Pith version.")
    artifact = Path(args.artifact)
    if artifact.is_symlink() or not artifact.is_file():
        raise AppSurfaceProofError("artifact_unreadable", "Release artifact is not a regular file.")
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AppSurfaceProofError("artifact_unreadable", "Release artifact is not readable.") from exc

    payload = _read_json_file(args.ledger)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) or not payload["rows"]:
        raise AppSurfaceProofError("invalid_ledger_fields", "Proof ledger does not contain proof rows.")
    rows = payload["rows"]
    run_id = payload.get("run_id")
    target_builds = {row.get("target_os_build") for row in rows if isinstance(row, dict)}
    host_versions = {
        pair_key(str(row.get("surface_id")), str(row.get("host_variant"))): row.get("host_version")
        for row in rows
        if isinstance(row, dict)
    }
    expected_pairs = {pair_key(*pair) for pair in REQUIRED_SURFACE_VARIANTS}
    if len(target_builds) != 1 or set(host_versions) != expected_pairs:
        raise AppSurfaceProofError("release_matrix_incomplete", "Release proof matrix is incomplete or inconsistent.")
    if not isinstance(run_id, str):
        raise AppSurfaceProofError("invalid_identifier", "Proof ledger run ID is invalid.")
    expectations = ProofExpectations(
        run_id=run_id,
        artifact_sha256=digest.hexdigest(),
        package_version=tag_match.group(1),
        target_os_build=str(next(iter(target_builds))),
        host_versions={pair: str(version) for pair, version in host_versions.items()},
        key_id=key_id,
    )
    result = validate_proof_ledger(payload, expectations=expectations, key=key)
    return {**result, "release_artifact_sha256": digest.hexdigest(), "release_tag": args.tag}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        key, key_id = _release_key()
        if args.command == "capture":
            result = capture_proof_row(
                raw_proof=_read_stdin_json(),
                metadata=_read_json_file(args.metadata),
                ledger_path=args.ledger,
                key=key,
                key_id=key_id,
            )
        elif args.command == "verify":
            expectations = ProofExpectations(
                run_id=args.expected_proof_run_id,
                artifact_sha256=args.expected_artifact_sha256,
                package_version=args.expected_package_version,
                target_os_build=args.expected_target_os_build,
                host_versions=_host_versions(args.expected_host_version),
                key_id=key_id,
            )
            result = load_proof_ledger(args.ledger, expectations=expectations, key=key)
        else:
            result = _verify_release(args, key, key_id)
    except AppSurfaceProofError as exc:
        print(json.dumps({"status": "error", "code": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
