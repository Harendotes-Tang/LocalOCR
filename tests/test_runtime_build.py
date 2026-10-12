from pathlib import Path
import pytest
from tools.LocalAiOcr.build_runtime import audit_payload, merge_tree, digest

def test_shared_runtime_conflicting_bytes_rejected(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir(); right.mkdir()
    (left / 'shared.dll').write_bytes(b'original')
    (right / 'shared.dll').write_bytes(b'different')
    with pytest.raises(RuntimeError, match='Conflicting'):
        merge_tree(left, right)
    assert (right / 'shared.dll').read_bytes() == b'different'

def test_equal_shared_files_and_new_entry_merge(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(); target.mkdir()
    (source / 'shared.dll').write_bytes(b'same')
    (target / 'shared.dll').write_bytes(b'same')
    (source / 'entry.exe').write_bytes(b'new entry')
    merge_tree(source, target)
    assert digest(target / 'entry.exe') == digest(source / 'entry.exe')

@pytest.mark.parametrize('relative', ['weights.onnx', 'fonts/a.otf', '_internal/torch/core.dll', 'models/a.json', 'candidate-v4/meta.json', 'datasets/private.png', '_internal/fastapi/__init__.py'])
def test_payload_forbids_training_network_framework_assets(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'forbidden')
    with pytest.raises(RuntimeError, match='Non-runtime'):
        audit_payload(tmp_path)

def test_required_curated_model_license_is_allowed(tmp_path):
    path = tmp_path / '_internal/docs/licenses/ocr/local-ai/rapidocr__MODEL_LICENSES.md'
    path.parent.mkdir(parents=True)
    path.write_text('Model licensing notice, no weights')
    assert audit_payload(tmp_path) == [path]


def test_remote_target_rejected_before_any_target_access(monkeypatch, tmp_path):
    import localocr.bundles as bundles
    from tools.LocalAiOcr.build_runtime import validate_build_paths
    def deny(path):
        raise ValueError('remote rejected')
    monkeypatch.setattr(bundles, 'require_local_path', deny)
    monkeypatch.setattr(Path, 'exists', lambda path: pytest.fail('target accessed before local check'))
    with pytest.raises(ValueError, match='remote rejected'):
        validate_build_paths(tmp_path / 'output', tmp_path / 'work')

def test_reparse_rejected_before_any_target_access(monkeypatch, tmp_path):
    import localocr.bundles as bundles
    from tools.LocalAiOcr.build_runtime import validate_build_paths
    monkeypatch.setattr(bundles, 'require_local_path', lambda path: None)
    def deny(path, allow_missing=False):
        raise ValueError('reparse rejected')
    monkeypatch.setattr(bundles, '_directory', deny)
    monkeypatch.setattr(Path, 'exists', lambda path: pytest.fail('target accessed before directory check'))
    with pytest.raises(ValueError, match='reparse rejected'):
        validate_build_paths(tmp_path / 'output', tmp_path / 'work')
def test_build_environment_removes_ambient_native_search_path(monkeypatch):
    from tools.LocalAiOcr.build_runtime import minimal_environment
    monkeypatch.setenv('SystemRoot', 'C:\\Windows')
    monkeypatch.setenv('PATH', 'C:\\untrusted-native-cache')
    monkeypatch.setenv('PYTHONPATH', 'C:\\unrelated-service')
    env = minimal_environment()
    assert 'untrusted-native-cache' not in env['PATH']
    assert 'PYTHONPATH' not in env


def test_source_snapshot_covers_repository_import_closure_and_helper():
    from tools.LocalAiOcr import build_runtime
    snapshot = build_runtime.source_snapshot()
    records = {item['path']: item for item in snapshot['files']}
    helper = build_runtime.REPO / 'tools/LocalAiOcr/build_worker.py'
    assert helper.is_file()
    assert records[helper.relative_to(build_runtime.REPO).as_posix()] == {
        'path': 'tools/LocalAiOcr/build_worker.py',
        'bytes': helper.stat().st_size, 'sha256': digest(helper)}
    closure = build_runtime.repository_import_closure([build_runtime.REPO / 'tools/LocalAiOcr/build_runtime.py'])
    assert helper in closure
    assert {path.relative_to(build_runtime.REPO).as_posix() for path in closure} <= set(records)


def test_missing_internal_helper_is_not_treated_as_external_dependency(tmp_path):
    from tools.LocalAiOcr.build_runtime import repository_import_closure
    directory = tmp_path / 'tools/LocalAiOcr'
    directory.mkdir(parents=True)
    entry = directory / 'build_runtime.py'
    entry.write_text('def build():\n    from tools.LocalAiOcr.build_worker import remove_unused_video_runtime\n')
    with pytest.raises(RuntimeError, match='tools.LocalAiOcr.build_worker'):
        repository_import_closure([entry], tmp_path)


def test_snapshot_source_copy_imports_helper_without_workspace_fallback(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    from tools.LocalAiOcr import build_runtime
    copied = tmp_path / 'extracted-source'
    for item in build_runtime.source_snapshot()['files']:
        target = copied / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build_runtime.REPO / item['path'], target)
    code = (
        'import sys,json; from pathlib import Path; '
        'root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root)); '
        'from tools.LocalAiOcr import build_runtime,build_worker; '
        'assert Path(build_runtime.__file__).resolve().is_relative_to(root); '
        'assert Path(build_worker.__file__).resolve().is_relative_to(root); '
        'empty=root.parent/"empty-payload";empty.mkdir(); '
        'assert build_worker.remove_unused_video_runtime(empty)==[]; '
        'assert root/"tools/LocalAiOcr/build_worker.py" in '
        'build_runtime.repository_import_closure([root/"tools/LocalAiOcr/build_runtime.py"],root); '
        'print(json.dumps({"helper_imported_from_extracted_source":True}))'
    )
    result = subprocess.run([sys.executable, '-I', '-c', code, str(copied)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['helper_imported_from_extracted_source'] is True


def copy_source_snapshot(root):
    import json
    import shutil
    from tools.LocalAiOcr import build_runtime
    snapshot = build_runtime.source_snapshot()
    for record in snapshot['files']:
        target = root / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build_runtime.REPO / record['path'], target)
    manifest = root / 'provenance/source-snapshot.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'schema_version': 'localocr.source-snapshot.v1', 'source_snapshot': snapshot}), encoding='utf-8')
    return snapshot, manifest


def test_extracted_archive_snapshot_is_verified_without_outer_git(tmp_path):
    import json
    import subprocess
    import sys
    from tools.LocalAiOcr import build_runtime
    root = tmp_path / 'source'
    snapshot, manifest = copy_source_snapshot(root)
    code = (
        'import sys,json; from pathlib import Path; '
        'root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root)); '
        'from tools.LocalAiOcr import build_runtime,build_worker; '
        'build_runtime.subprocess.check_output=lambda *a,**k: (_ for _ in ()).throw(AssertionError("must not call outer git")); '
        'snapshot=build_runtime.source_snapshot(); '
        'assert snapshot["source_kind"]=="archive-snapshot"; '
        'assert Path(build_worker.__file__).resolve().is_relative_to(root); '
        'print(json.dumps(snapshot))'
    )
    result = subprocess.run([sys.executable, '-I', '-c', code, str(root)], cwd=build_runtime.REPO, capture_output=True, text=True, check=True)
    copied = json.loads(result.stdout)
    assert copied['base_commit'] == snapshot['base_commit']
    assert copied['working_tree_dirty'] == snapshot['working_tree_dirty']
    assert copied['files'] == snapshot['files']
    help_code = 'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv=["build_runtime", "--help"];runpy.run_module("tools.LocalAiOcr.build_runtime",run_name="__main__")'
    help_result = subprocess.run([sys.executable, '-I', '-c', help_code, str(root)], cwd=build_runtime.REPO, capture_output=True, text=True, check=True)
    assert '--output' in help_result.stdout


@pytest.mark.parametrize('mutation', ['tamper', 'missing', 'escape', 'duplicate', 'undeclared_helper'])
def test_archive_snapshot_rejects_invalid_source_inventory(monkeypatch, tmp_path, mutation):
    import json
    from tools.LocalAiOcr import build_runtime
    root = tmp_path / 'source'
    snapshot, manifest = copy_source_snapshot(root)
    helper = root / 'tools/LocalAiOcr/build_worker.py'
    if mutation == 'tamper':
        helper.write_text('tampered')
    elif mutation == 'missing':
        helper.unlink()
    elif mutation == 'escape':
        snapshot['files'].append({'path': '../escape.py', 'bytes': 0, 'sha256': '0' * 64})
    elif mutation == 'duplicate':
        snapshot['files'].append(snapshot['files'][0])
    else:
        snapshot['files'] = [record for record in snapshot['files'] if record['path'] != 'tools/LocalAiOcr/build_worker.py']
    manifest.write_text(json.dumps({'schema_version': 'localocr.source-snapshot.v1', 'source_snapshot': snapshot}), encoding='utf-8')
    monkeypatch.setattr(build_runtime, 'REPO', root)
    # Default arguments bind the real checkout, so use this copied root for
    # the dependency scanner while preserving its actual implementation.
    original_closure = build_runtime.repository_import_closure
    monkeypatch.setattr(build_runtime, 'repository_import_closure', lambda files: original_closure(files, root))
    monkeypatch.setattr(build_runtime.subprocess, 'check_output', lambda *a, **kw: pytest.fail('outer Git must not be queried'))
    with pytest.raises(RuntimeError):
        build_runtime.source_snapshot()


def test_archive_missing_provenance_does_not_use_outer_git(monkeypatch, tmp_path):
    from tools.LocalAiOcr import build_runtime
    monkeypatch.setattr(build_runtime, 'REPO', tmp_path)
    monkeypatch.setattr(build_runtime.subprocess, 'check_output', lambda *a, **kw: pytest.fail('outer Git must not be queried'))
    with pytest.raises(RuntimeError, match='Missing archive source file'):
        build_runtime.source_snapshot()


def test_archive_reparse_is_rejected_before_descendant_lookup(monkeypatch, tmp_path):
    import stat
    from types import SimpleNamespace
    from tools.LocalAiOcr import build_runtime
    directory = tmp_path / 'linked'
    directory.mkdir()
    monkeypatch.setattr(build_runtime, 'REPO', tmp_path)
    original_lstat = Path.lstat
    def checked_lstat(path):
        if path == directory:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        if path == directory / 'file.py':
            pytest.fail('reparse target was accessed')
        return original_lstat(path)
    monkeypatch.setattr(Path, 'lstat', checked_lstat)
    with pytest.raises(RuntimeError, match='reparse'):
        build_runtime.checked_archive_file('linked/file.py')


def test_git_checkout_root_mismatch_rejected_before_commit_lookup(monkeypatch, tmp_path):
    from tools.LocalAiOcr import build_runtime
    (tmp_path / '.git').mkdir()
    monkeypatch.setattr(build_runtime, 'REPO', tmp_path)
    calls = []
    def outside_root(command, **kwargs):
        calls.append(command)
        return str(tmp_path.parent)
    monkeypatch.setattr(build_runtime.subprocess, 'check_output', outside_root)
    with pytest.raises(RuntimeError, match='root does not match'):
        build_runtime.source_snapshot()
    assert calls == [['git', 'rev-parse', '--show-toplevel']]


@pytest.mark.parametrize('relative', ['../outside.py', 'C:/outside.py', 'dir\\file.py', 'dir//file.py', 'dir./file.py', 'dir/file.py ', 'CON.py', 'dir/file?.py'])
def test_archive_source_path_aliases_rejected_before_filesystem(monkeypatch, relative):
    from tools.LocalAiOcr.build_runtime import checked_archive_file
    monkeypatch.setattr(Path, 'lstat', lambda path: pytest.fail('invalid archive path touched filesystem'))
    with pytest.raises(RuntimeError, match='Invalid archive source path'):
        checked_archive_file(relative)


@pytest.mark.parametrize('stdout', [b'MentorRecorder local AI OCR 2\n', b'LocalOCR 0.1.0\n'])
def test_frozen_version_must_match_central_identity(monkeypatch, tmp_path, stdout):
    import subprocess
    from tools.LocalAiOcr import build_runtime
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout if command[-1] == '--version' else b'help', b'')
    monkeypatch.setattr(build_runtime.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='software version mismatch'):
        build_runtime.verify_frozen_versions(tmp_path, ['local-ai-ocr'])


def test_frozen_version_checks_both_exes_with_system_only_environment(monkeypatch, tmp_path):
    import subprocess
    from localocr.version import VERSION_TEXT
    from tools.LocalAiOcr import build_runtime
    calls = []
    monkeypatch.setenv('PYTHONPATH', 'ambient-workspace')
    def fake_run(command, **kwargs):
        calls.append(command)
        assert 'PYTHONPATH' not in kwargs['env']
        assert kwargs['check'] and kwargs['timeout'] == 30
        return subprocess.CompletedProcess(command, 0, (VERSION_TEXT + '\r\n').encode() if command[-1] == '--version' else b'help', b'')
    monkeypatch.setattr(build_runtime.subprocess, 'run', fake_run)
    checks = build_runtime.verify_frozen_versions(tmp_path, ['local-ai-ocr', 'localocr-models'])
    assert [check['entrypoint'] for check in checks] == ['local-ai-ocr.exe', 'localocr-models.exe']
    assert all(check['stdout'] == VERSION_TEXT + '\n' for check in checks)
    assert [command[-1] for command in calls] == ['--help', '--version', '--help', '--version']


def test_powershell_defaults_project_central_names_from_unrelated_cwd(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    import sys
    from localocr.version import artifact_names
    from tools.LocalAiOcr import build_runtime
    if sys.platform != 'win32':
        pytest.skip('PowerShell entry is Windows-specific')
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if not shell:
        shell = str(Path(build_runtime.minimal_environment()['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
    result_file = tmp_path / 'build-call.json'
    fake_python = tmp_path / 'python-mock.ps1'
    names_json = json.dumps(artifact_names()).replace("'", "''")
    fake_python.write_text(
        'if ($args[2] -match "artifact_names") { Write-Output \'' + names_json + "'; exit 0 }\n" +
        "$payload = @{ cwd = (Get-Location).Path; args = @($args) }\n" +
        "$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath '" + str(result_file).replace("'", "''") + "' -Encoding utf8\nexit 0\n",
        encoding='utf-8')
    subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(build_runtime.REPO / 'scripts/build-localocr-runtime.ps1'), '-BuildPython', str(fake_python), '-Notices', 'relative-notices'], cwd=tmp_path, capture_output=True, text=True, check=True)
    payload = json.loads(result_file.read_text(encoding='utf-8-sig'))
    assert Path(payload['cwd']).resolve() == build_runtime.REPO
    args = payload['args']
    assert Path(args[args.index('--output') + 1]) == build_runtime.REPO / 'artifacts' / artifact_names()['runtime_dir']
    assert Path(args[args.index('--build-dir') + 1]) == build_runtime.REPO / 'artifacts' / artifact_names()['build_dir']
    assert Path(args[args.index('--notices') + 1]) == build_runtime.REPO / 'relative-notices'
def test_source_snapshot_includes_version_contract_and_current_zip_producer():
    from tools.LocalAiOcr import build_runtime
    records = {item['path']: item for item in build_runtime.source_snapshot()['files']}
    for relative in ('localocr/version.py', 'tools/LocalAiOcr/package_runtime.py', 'docs/contracts/versioning.md'):
        source = build_runtime.REPO / relative
        assert records[relative]['bytes'] == source.stat().st_size
        assert records[relative]['sha256'] == digest(source)
    assert len(build_runtime.CONTRACT_FILES) == 4
    assert all('docs/contracts/' + name in records for name in build_runtime.CONTRACT_FILES)
