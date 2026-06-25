from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "public_safety_scan.py"
SPEC = importlib.util.spec_from_file_location("public_safety_scan", SCRIPT_PATH)
assert SPEC is not None
public_safety_scan = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["public_safety_scan"] = public_safety_scan
SPEC.loader.exec_module(public_safety_scan)


def test_metadata_scan_rejects_local_paths() -> None:
    local_path = "/".join(["", "Users", "pith", "Desktop", "artifact"])
    findings = public_safety_scan.scan_text(
        "pull request / commit metadata",
        f"Verification used {local_path}.",
        set(),
    )

    assert [finding.label for finding in findings] == ["absolute local user path"]


def test_metadata_scan_rejects_private_launch_terms() -> None:
    launch_phrase = " ".join(["internal", "launch"])
    staging_phrase = " ".join(["private", "staging"])
    findings = public_safety_scan.scan_text(
        "pull request / commit metadata",
        f"This prepares {launch_phrase} notes for {staging_phrase}.",
        set(),
    )

    assert {finding.label for finding in findings} == {
        "private release phrase",
        "nonpublic staging phrase",
    }


def test_public_facing_pr_missing_audience_review_evidence_fails() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="Update installer",
        body="- [x] Public-safe diff reviewed.",
        event_path="/tmp/event.json",
        error="",
    )

    findings = public_safety_scan.audience_review_findings(["scripts/install.sh"], metadata)

    assert [finding.value for finding in findings] == [
        "Public surfaces inventoried",
        "Audience classification",
        "PR title/body audience review",
        "Residual deferrals",
    ]


def test_public_facing_pr_with_all_audience_review_evidence_passes() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="Update installer",
        body="\n".join(
            [
                "- [x] Public surfaces inventoried: scripts/install.sh",
                "- [x] Audience classification: installer user",
                "- [x] PR title/body audience review: developer-facing and leak-safe",
                "- [x] Residual deferrals: none",
            ]
        ),
        event_path="/tmp/event.json",
        error="",
    )

    assert public_safety_scan.audience_review_findings(["scripts/install.sh"], metadata) == []


def test_public_facing_pr_accepts_explicit_not_applicable_deferrals() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="Update release validation",
        body="\n".join(
            [
                "- [x] Public surfaces inventoried: release-validation workflow",
                "- [x] Audience classification: contributor",
                "- [x] PR title/body audience review: complete",
                "- [ ] Residual deferrals: N/A",
            ]
        ),
        event_path="/tmp/event.json",
        error="",
    )

    assert public_safety_scan.audience_review_findings(
        [".github/workflows/release-validation.yml"],
        metadata,
    ) == []


def test_public_facing_pr_unchecked_template_lines_do_not_count_as_evidence() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="Update README",
        body="\n".join(
            [
                "- [ ] Public surfaces inventoried",
                "- [ ] Audience classification",
                "- [ ] PR title/body audience review",
                "- [ ] Residual deferrals",
            ]
        ),
        event_path="/tmp/event.json",
        error="",
    )

    findings = public_safety_scan.audience_review_findings(["README.md"], metadata)

    assert len(findings) == 4


def test_non_public_code_change_does_not_require_audience_review_evidence() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="Refactor API internals",
        body="",
        event_path="",
        error="GITHUB_EVENT_PATH is unavailable",
    )

    assert public_safety_scan.audience_review_findings(["app/api/server.py"], metadata) == []


def test_workflow_dispatch_without_pr_event_does_not_require_audience_review_evidence() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="",
        body="",
        event_path="",
        error="GITHUB_EVENT_PATH is unavailable",
        event_name="workflow_dispatch",
    )

    assert public_safety_scan.audience_review_findings(["README.md"], metadata) == []


def test_public_facing_empty_pr_body_fails() -> None:
    metadata = public_safety_scan.PullRequestMetadata(
        title="Update README",
        body="",
        event_path="/tmp/event.json",
        error="",
    )

    findings = public_safety_scan.audience_review_findings(["README.md"], metadata)

    assert len(findings) == 1
    assert findings[0].value == "empty PR body"


def test_public_facing_classification_true_paths() -> None:
    assert public_safety_scan.is_public_facing_change(
        [
            "README.md",
            "scripts/install.ps1",
            "pith-server-latest.tar.gz",
            ".github/pull_request_template.md",
            ".github/scripts/public_safety_scan.py",
        ]
    )


def test_public_facing_classification_adversarial_false_positives() -> None:
    # ADVERSARIAL-FP: near-boundary path names that should remain developer-internal.
    paths = [
        "docs/README.md",
        "README.md.bak",
        "scripts/install-helper.sh",
        "app/scripts/install.sh",
        "app/public_safety_scan.py",
        ".github/dependabot.yml",
        "../README.md",
        "/README.md",
        None,
        42,
    ]

    assert not public_safety_scan.is_public_facing_change(paths)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASS {name}")
