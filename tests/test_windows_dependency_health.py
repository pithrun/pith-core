"""Owned pure fixtures for the installer checker; no application or brain access."""

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from importlib import metadata
from pathlib import Path
from unittest import mock

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

CHECKER = Path(__file__).resolve().parents[1] / "scripts/windows_dependency_health.py"
spec = importlib.util.spec_from_file_location("windows_dependency_health", CHECKER)
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


class FakeDistribution:
    def __init__(self, prefix, name="core", version="1.0", requires=None, rows=None):
        self.prefix = prefix
        self.metadata = {"Name": name}
        self.version = version
        self.requires = requires
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(rows if rows is not None else [("core.py", "", "")])
        self.record = stream.getvalue()

    def read_text(self, filename):
        assert filename == "RECORD"
        return self.record

    def locate_file(self, path):
        return self.prefix / path

    @property
    def files(self):
        raise AssertionError("Must audit raw RECORD, never filtered Distribution.files")


def lookup_for(distributions):
    def lookup(name):
        if name not in distributions:
            raise metadata.PackageNotFoundError(name)
        return distributions[name]

    return lookup


def test_health_error_redacts_absolute_and_secret_exception_details():
    error = health.HealthError("record_invalid", "bad name", "/secret/api.key")
    assert error.diagnostic() == {"code": "record_invalid"}
    assert str(error) == "record_invalid"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-be"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_requirements_encodings_quotes_and_comments(tmp_path, encoding, newline):
    # ADVERSARIAL-FP: deterministic BOMs/newlines and quoted # remain valid.
    path = tmp_path / "roots.txt"
    text = newline.join(["# comment", "core>=1 # inline", 'other; platform_version != "a#b"'])
    raw = text.encode(encoding)
    if encoding == "utf-16-be":
        raw = b"\xfe\xff" + raw
    path.write_bytes(raw)
    assert [root.name for root in health.parse_requirements(path)] == ["core", "other"]


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "requirements_empty"),
        ("# only comment", "requirements_empty"),
        ('core; python_version < "0"', "requirements_empty"),
        ("-r other.txt", "requirements_invalid"),
        ("--index-url https://evil", "requirements_invalid"),
        ("core @ https://evil/pkg.whl", "requirements_invalid"),
        ("core>=1\\", "requirements_invalid"),
        ("core\x00", "requirements_invalid"),
        ("'; DROP TABLE concepts; --", "requirements_invalid"),
        ("a" * (health.MAX_FIELD_CHARS + 1), "requirements_invalid"),
        ("core\n" * (health.MAX_ROOTS + 1), "requirements_invalid"),
    ],
)
def test_requirements_reject_invalid_boundary_injection_and_empty(tmp_path, text, code):
    path = tmp_path / "roots.txt"
    path.write_text(text)
    with pytest.raises(health.HealthError, match=code):
        health.parse_requirements(path)


def test_requirements_missing_invalid_encoding_and_size(tmp_path):
    path = tmp_path / "roots.txt"
    with pytest.raises(health.HealthError, match="requirements_unreadable"):
        health.parse_requirements(path)
    for raw in (b"\x80core", b"a" * (health.MAX_REQUIREMENTS_BYTES + 1)):
        path.write_bytes(raw)
        with pytest.raises(health.HealthError, match="requirements_invalid"):
            health.parse_requirements(path)


@pytest.mark.parametrize("value", [None, 0, True, [], {}, b"core"])
def test_requirements_fuzz_wrong_types(value):
    with pytest.raises(health.HealthError, match="requirements_unreadable"):
        health.parse_requirements(value)


def test_missing_packaging_has_stable_result_even_module_import_safe(tmp_path):
    original = __import__

    def without_packaging(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ImportError("SECRET")
        return original(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=without_packaging):
        result = health.run_check(tmp_path / "roots")
    assert result["failures"] == [{"code": "packaging_unavailable"}]


def test_resolver_cycles_late_extras_and_platform_markers(tmp_path):
    # ADVERSARIAL-FP: late extras and legitimate Windows-only edges must pass.
    distributions = {
        "core": FakeDistribution(tmp_path, requires=["later", "shared"]),
        "shared": FakeDistribution(
            tmp_path, "shared", requires=['extra-dep; extra == "feature"', 'win; sys_platform == "win32"']
        ),
        "later": FakeDistribution(tmp_path, "later", requires=["core", "shared[feature]"]),
        "extra-dep": FakeDistribution(tmp_path, "extra-dep"),
        "win": FakeDistribution(tmp_path, "win"),
        "packaging": FakeDistribution(tmp_path, "packaging"),
    }
    environment = {**default_environment(), "sys_platform": "win32"}
    selected = health.resolve_closure([Requirement("core")], lookup_for(distributions), environment)
    assert set(selected) == set(distributions)
    assert "torch" not in selected  # Optional-only distributions are not core roots.


def test_resolver_checks_constraint_on_every_already_visited_edge(tmp_path):
    distributions = {
        "core": FakeDistribution(tmp_path, requires=["shared<1", "shared>=1"]),
        "shared": FakeDistribution(tmp_path, "shared"),
        "packaging": FakeDistribution(tmp_path, "packaging"),
    }
    with pytest.raises(health.HealthError, match="version_mismatch"):
        health.resolve_closure([Requirement("core")], lookup_for(distributions))


@pytest.mark.parametrize("change", ["wrong-name", "wrong-version", "bad-requires", "url"])
def test_resolver_invalid_metadata(tmp_path, change):
    core = FakeDistribution(tmp_path)
    if change == "wrong-name":
        core.metadata = {"Name": "alien"}
    if change == "wrong-version":
        core.version = "not-a-version"
    if change == "bad-requires":
        core.requires = ["invalid???"]
    if change == "url":
        core.requires = ["other @ https://evil"]
    with pytest.raises(health.HealthError, match="metadata_invalid"):
        health.resolve_closure(
            [Requirement("core")], lookup_for({"core": core, "packaging": FakeDistribution(tmp_path, "packaging")})
        )


def test_resolver_missing_limits_and_no_applicable_core(tmp_path, monkeypatch):
    package = FakeDistribution(tmp_path, "packaging")
    lookup = lookup_for({"packaging": package, "core": FakeDistribution(tmp_path)})
    with pytest.raises(health.HealthError, match="distribution_missing"):
        health.resolve_closure([Requirement("missing")], lookup)
    for roots in ([], [Requirement('core; python_version < "0"')]):
        with pytest.raises(health.HealthError, match="requirements_empty"):
            health.resolve_closure(roots, lookup)
    monkeypatch.setattr(health, "MAX_DISTRIBUTIONS", 1)
    with pytest.raises(health.HealthError, match="closure_limit"):
        health.resolve_closure([Requirement("core")], lookup)
    monkeypatch.setattr(health, "MAX_DISTRIBUTIONS", 400)
    monkeypatch.setattr(health, "MAX_EDGES", 1)
    with pytest.raises(health.HealthError, match="closure_limit"):
        health.resolve_closure([Requirement("core")], lookup)


@pytest.mark.parametrize(
    "path",
    [
        "core.py",
        "native.PYD",
        "native.DLL",
        "native.dylib",
        "native.SO",
        "native.so.1.2",
        "comma,name.py",
        "space name.py",
        "../Scripts/tool.py",
    ],
)
def test_record_valid_runtime_and_safe_parent(tmp_path, path):
    # ADVERSARIAL-FP: runtime suffixes, comma/space names and contained ../Scripts.
    site = tmp_path / "site"
    site.mkdir()
    target = site / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fixture")
    result = health.audit_records({"core": FakeDistribution(site, rows=[(path, "sha256=YWJj", "7")])}, tmp_path)
    assert result["failure_count"] == 0
    assert result["runtime_file_count"] == 1


def test_record_bytecode_ignored_docs_advisory_and_full_counts(tmp_path):
    # ADVERSARIAL-FP: generated pyc/pyo and missing docs are not runtime failures.
    (tmp_path / "core.py").write_text("fixture")
    rows = [("core.py", "", ""), ("gone.pyc", "", ""), ("gone.pyo", "", "")]
    rows += [(f"doc-{number}.txt", "", "") for number in range(25)]
    result = health.audit_records({"core": FakeDistribution(tmp_path, rows=rows)}, tmp_path)
    assert result["failure_count"] == 0
    assert result["advisory_count"] == 25
    assert len(result["advisories"]) == health.MAX_EXAMPLES


def test_record_optional_only_absence_does_not_expand_selection(tmp_path):
    # ADVERSARIAL-FP: unselected optional embedding packages are not audited.
    (tmp_path / "core.py").write_text("fixture")
    result = health.audit_records({"core": FakeDistribution(tmp_path)}, tmp_path)
    assert result["failure_count"] == 0
    assert not (tmp_path / "torch.py").exists()


def test_resolver_caps_even_inactive_metadata_edges(tmp_path, monkeypatch):
    core = FakeDistribution(tmp_path, requires=['other; python_version < "0"'] * 5)
    monkeypatch.setattr(health, "MAX_EDGES", 4)
    with pytest.raises(health.HealthError, match="closure_limit"):
        health.resolve_closure(
            [Requirement("core")], lookup_for({"core": core, "packaging": FakeDistribution(tmp_path, "packaging")})
        )


@pytest.mark.parametrize(
    "path",
    [
        "/outside.py",
        "C:/outside.py",
        "C:outside.py",
        "\\\\host\\share\\a.py",
        "\\root.py",
        "x\\y.py",
        "x:stream.py",
        "../../outside.py",
        "../outside.txt",
        "null\x00.py",
        "line\n.py",
        "control\x7f.py",
        "control\x85.py",
    ],
)
def test_record_escape_and_control_vectors_blocked(tmp_path, path):
    site = tmp_path / "site"
    site.mkdir()
    if "\x00" in path:
        # Python 3.10's CSV writer rejects NUL before the checker can inspect it.
        # Feed the deliberately invalid raw RECORD instead; never skip this attack.
        dist = FakeDistribution(site, rows=[])
        dist.record = path + ",,\r\n"
    else:
        dist = FakeDistribution(site, rows=[(path, "", "")])
    result = health.audit_records({"core": dist}, site)
    assert result["failure_count"] >= 1
    assert any(item["code"] in ("record_invalid", "unsafe_record_path") for item in result["failures"])


def test_record_symlink_outside_prefix_rejected_before_is_file(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret")
    try:
        (site / "core.py").symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable in this test environment")
    with mock.patch.object(Path, "is_file", side_effect=AssertionError("Must not inspect escaped target")):
        result = health.audit_records({"core": FakeDistribution(site)}, site)
    assert result["failures"][0]["code"] == "unsafe_record_path"


def test_record_compound_escape_overflow_never_resolves_target(tmp_path):
    dist = FakeDistribution(tmp_path, rows=[("C:/secret/" + "x" * health.MAX_FIELD_CHARS, "", "")])
    with mock.patch.object(dist, "locate_file", side_effect=AssertionError("No target access")):
        result = health.audit_records({"core": dist}, tmp_path)
    assert result["failures"][0] == {"code": "record_invalid", "distribution": "core"}


def test_record_permission_race_fails_with_stable_code(tmp_path):
    dist = FakeDistribution(tmp_path)
    with mock.patch.object(dist, "read_text", side_effect=PermissionError("SECRET PATH")):
        result = health.audit_records({"core": dist}, tmp_path)
    assert result["failures"][0] == {"code": "record_invalid", "distribution": "core"}


def test_checker_is_packaged_and_native_contract_uses_exact_artifact():
    root = CHECKER.parents[1]
    # Check the distributed files directly in the public release repository.
    expected = CHECKER.read_bytes()
    with tarfile.open(root / "pith-server-latest.tar.gz") as archive:
        assert archive.extractfile("scripts/windows_dependency_health.py").read() == expected
        assert archive.extractfile("scripts/install.ps1").read() == (root / "install.ps1").read_bytes()
    with zipfile.ZipFile(root / "pith-server-latest.zip") as archive:
        assert archive.read("scripts/windows_dependency_health.py") == expected
        assert archive.read("scripts/install.ps1") == (root / "install.ps1").read_bytes()
    native = (root / "tests/windows_installer_health_contract.ps1").read_text()
    assert 'if ($null -ne $script:ProbePlan)' in native
    assert "false_only_probe_plan_is_not_ready" in native


@pytest.mark.parametrize(
    "record,code",
    [
        (None, "record_missing"),
        ("", "record_missing"),
        ("\n", "record_missing"),
        ("core.py,\n", "record_invalid"),
        ('"unterminated', "record_invalid"),
        ("core.py,sha256=!,1\n", "record_invalid"),
        ("core.py,, -1\n", "record_invalid"),
        ("core.py,,1.0\n", "record_invalid"),
        ("core.py,,١\n", "record_invalid"),
        ("x" * (health.MAX_FIELD_CHARS + 1) + ",,\n", "record_invalid"),
    ],
)
def test_record_format_fuzzer(tmp_path, record, code):
    dist = FakeDistribution(tmp_path)
    dist.record = record
    result = health.audit_records({"core": dist}, tmp_path)
    assert result["failures"][0]["code"] == code


def test_record_missing_runtime_full_failures_and_limit_guards(tmp_path, monkeypatch):
    dist = FakeDistribution(tmp_path, rows=[(f"missing-{number}.py", "", "") for number in range(25)])
    result = health.audit_records({"core": dist}, tmp_path)
    assert result["failure_count"] == 25
    assert len(result["failures"]) == 20
    for constant in ("MAX_RECORD_BYTES", "MAX_RECORD_ROWS", "MAX_TOTAL_ROWS"):
        with monkeypatch.context() as patch:
            patch.setattr(health, constant, 1)
            result = health.audit_records({"core": dist}, tmp_path)
            assert any(item["code"] == "record_limit" for item in result["failures"])
    result = health.audit_records({"core": FakeDistribution(tmp_path, rows=[("doc.txt", "", "")])}, tmp_path)
    assert any(item["code"] == "runtime_files_empty" for item in result["failures"])


def good_import_receipt():
    return {
        "schema_version": 1,
        "scope": "critical_library_imports",
        "status": "pass",
        "imports_checked": 13,
        "failure_count": 0,
        "failures": [],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "array",
        "null",
        "missing",
        "boolean",
        "float",
        "scope",
        "status-array",
        "count",
        "failures",
        "contaminated",
        "oversized",
    ],
)
def test_import_receipt_type_confusion_and_contamination_rejected(mutation):
    receipt = good_import_receipt()
    if mutation == "array":
        receipt = [receipt]
    if mutation == "null":
        receipt = None
    if mutation == "missing":
        receipt.pop("status")
    if mutation == "boolean":
        receipt["schema_version"] = True
    if mutation == "float":
        receipt["imports_checked"] = 13.0
    if mutation == "scope":
        receipt["scope"] = "core_runtime_dependencies"
    if mutation == "status-array":
        receipt["status"] = ["pass"]
    if mutation == "count":
        receipt["imports_checked"] = 12
    if mutation == "failures":
        receipt["failures"] = [{"exception": "secret"}]
    stdout = json.dumps(receipt).encode()
    if mutation == "contaminated":
        stdout += b"trailing"
    if mutation == "oversized":
        stdout += b" " * health.MAX_OUTPUT_BYTES
    with mock.patch.object(health.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout, b"")):
        with pytest.raises(health.HealthError, match="import_result_invalid"):
            health.check_imports()


def test_import_process_real_exit_deadline_and_argv():
    with mock.patch.object(
        health.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, json.dumps(good_import_receipt()).encode(), b""),
    ) as launch:
        assert health.check_imports()["imports_checked"] == 13
        assert launch.call_args.args[0] == [sys.executable, "-I", "-B", str(CHECKER), "--imports-only"]
        assert launch.call_args.kwargs["timeout"] == 60
        assert "shell" not in launch.call_args.kwargs
    with mock.patch.object(health.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 60)):
        with pytest.raises(health.HealthError, match="import_timeout"):
            health.check_imports()
    with mock.patch.object(health.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, b"{}", b"SECRET")):
        with pytest.raises(health.HealthError, match="import_failed"):
            health.check_imports()


def test_imports_only_uses_exact_modules_and_redacts_exception(capsys):
    def noisy_import(module):
        print("SECRET")
        if module == "mcp":
            raise RuntimeError("SECRET KEY")

    with mock.patch.object(health.importlib, "import_module", side_effect=noisy_import) as imports:
        assert health.main(["--imports-only"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert [call.args[0] for call in imports.call_args_list] == list(health.CRITICAL_IMPORTS)
    assert result["failures"] == [{"module": "mcp", "exception": "RuntimeError"}]
    assert result["imports_checked"] == 13


def test_run_check_pass_retains_advisory_and_no_writes(tmp_path, monkeypatch):
    (tmp_path / "core.py").write_text("fixture")
    requirements = tmp_path / "roots.txt"
    requirements.write_text("core")
    selected = {"core": FakeDistribution(tmp_path, rows=[("core.py", "", ""), ("doc.txt", "", "")])}
    monkeypatch.setattr(health, "resolve_closure", lambda roots: selected)
    monkeypatch.setattr(health.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(health, "check_imports", good_import_receipt)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = health.run_check(requirements)
    assert result["status"] == "pass"
    assert result["distribution_count"] == result["runtime_file_count"] == 1
    assert result["advisory_count"] == 1
    assert result["failure_count"] == 0 and result["failures"] == []
    assert before == sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))


def test_run_check_missing_runtime_stops_import_and_unexpected_is_redacted(tmp_path):
    with (
        mock.patch.object(health, "parse_requirements", return_value=[Requirement("core")]),
        mock.patch.object(health, "resolve_closure", return_value={"core": FakeDistribution(tmp_path)}),
        mock.patch.object(health, "check_imports") as imports,
    ):
        assert health.run_check(tmp_path / "roots")["status"] == "fail"
        imports.assert_not_called()
    with mock.patch.object(health, "parse_requirements", side_effect=RuntimeError("SECRET")):
        assert health.run_check(tmp_path / "roots")["failures"] == [{"code": "internal_error"}]


@pytest.mark.parametrize(
    "args,exit_code",
    [([], 2), (["--requirements", "missing", "--imports-only"], 2), (["--requirements", "missing"], 1)],
)
def test_real_subprocess_usage_and_failure_are_nonzero(args, exit_code):
    completed = subprocess.run([sys.executable, "-I", "-B", str(CHECKER), *args], capture_output=True, timeout=10)
    assert completed.returncode == exit_code
    if exit_code == 1:
        receipt = json.loads(completed.stdout)
        assert receipt["status"] == "fail"
        assert receipt["failures"] == [{"code": "requirements_unreadable"}]


def test_main_success_emits_one_json_object(tmp_path, capsys):
    with mock.patch.object(health, "run_check", return_value={"status": "pass"}):
        assert health.main(["--requirements", str(tmp_path / "roots")]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "pass"}
