"""Software identity across real source CLI entrypoints; no OCR quality claim."""
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from localocr import version

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", [
    "local-ai-ocr-worker.py", "local-ai-ocr-entry.py", "localocr-models-entry.py",
])
def test_script_versions_from_unrelated_cwd_in_isolated_python(tmp_path, entry):
    # -I excludes cwd and PYTHONPATH: the scripts must resolve their own source
    # package, and version-only invocation must not require a model directory.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts" / entry), "--version"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "LocalOCR 0.1.0-rc.2\n"
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", ["localocr", "localocr.bundle_cli"])
def test_package_cli_versions_without_required_operational_arguments(module):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "LocalOCR 0.1.0-rc.2\n"
    assert result.stderr == ""


@pytest.mark.parametrize("command", ["create-baseline", "activate"])
def test_model_subcommand_version_remains_an_explicit_model_identity(monkeypatch, capsys, command):
    from localocr import bundle_cli
    received = []
    def operation(*args):
        received.append(args)
        return SimpleNamespace(version=args[-1], root=Path("legacy-test-package"))
    if command == "create-baseline":
        monkeypatch.setattr(bundle_cli, "create_baseline", operation)
        arguments = [command, "--models", "existing-models", "--output", "new-package", "--version", "baseline-1"]
    else:
        monkeypatch.setattr(bundle_cli, "activate_bundle", operation)
        arguments = [command, "--store", "existing-store", "--version", "baseline-1"]
    monkeypatch.setattr(sys, "argv", ["bundle_cli", *arguments])
    bundle_cli.main()
    assert received and received[0][-1] == "baseline-1"
    assert '"version": "baseline-1"' in capsys.readouterr().out


def test_model_manager_version_does_not_enter_job_or_model_operations(monkeypatch, capsys):
    from localocr import runtime_entry
    monkeypatch.setattr(sys, "argv", ["localocr-models.exe", "--version"])
    def forbidden():
        raise AssertionError("Version-only queries must not start model-job state")
    monkeypatch.setattr(runtime_entry, "establish_job", forbidden)
    runtime_entry.models_main()
    assert capsys.readouterr().out == "LocalOCR 0.1.0-rc.2\n"


def test_runtime_version_does_not_validate_paths_or_start_worker(monkeypatch, capsys):
    from localocr import runtime_entry
    def forbidden(*args, **kwargs):
        raise AssertionError("Version-only queries must not access models or infer")
    monkeypatch.setattr(runtime_entry, "validate", forbidden)
    monkeypatch.setattr(runtime_entry, "load_worker", forbidden)
    assert runtime_entry.main(["--version"]) == 0
    assert capsys.readouterr().out == "LocalOCR 0.1.0-rc.2\n"


def test_artifact_basenames_follow_approved_independent_identities():
    assert version.SOFTWARE_VERSION == "0.1.0-rc.2"
    assert version.BASELINE_MODEL_ID == "text-ppocrv6-1.0.0"
    expected = {
        "runtime_dir": "localocr-runtime-0.1.0-rc.2-windows-x64",
        "runtime_zip": "localocr-runtime-0.1.0-rc.2-windows-x64.zip",
        "source_zip": "localocr-source-0.1.0-rc.2.zip",
        "model_zip": "localocr-model-text-ppocrv6-1.0.0.zip",
        "inventory": "localocr-runtime-0.1.0-rc.2-windows-x64-inventory.json",
        "delivery": "localocr-delivery-0.1.0-rc.2.json",
        "build_dir": "runtime-build-0.1.0-rc.2",
    }
    assert version.artifact_names() == expected
    names = version.artifact_names()
    names["runtime_zip"] = "caller-local-edit"
    assert version.artifact_names() == expected


def test_software_version_does_not_change_schema_or_legacy_model_acceptance():
    from localocr import bundles
    assert bundles.SCHEMA == "localocr.bundle.v1"
    assert bundles._version("baseline-1") == "baseline-1"
    assert bundles._version(version.BASELINE_MODEL_ID) == version.BASELINE_MODEL_ID
    assert bundles.COMPATIBILITY == {
        "architecture": "ppocr-det-cls-rec-v1",
        "dictionary": "onnx-embedded-v1",
        "preprocessing": "rapidocr-3.10.0-default-v1",
        "postprocessing": "localocr-symbol-reread-v1",
    }
