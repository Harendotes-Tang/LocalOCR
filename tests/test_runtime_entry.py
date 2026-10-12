"""Broker envelope regressions; no mocked response is OCR acceptance."""
import argparse
from pathlib import Path
import subprocess
import sys
import pytest
from localocr import runtime_entry as entry


def args(tmp_path, **updates):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    image = tmp_path / "input.png"
    image.write_bytes(b"not decoded by the broker")
    values = dict(models=str(models), manifest=None, model_version=None, input=str(image),
                  output=str(tmp_path / "result.json"), format="json", self_test=False,
                  network_audit_self_test=False, timeout=90., _runtime_core=False, owned_png=False)
    values.update(updates)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("timeout", [0, -1, 91, float('nan'), float('inf')])
def test_timeout_envelope(tmp_path, timeout):
    with pytest.raises(ValueError, match="Timeout"):
        entry.validate(args(tmp_path, timeout=timeout))


def test_json_ownership_and_partial(tmp_path):
    value = args(tmp_path)
    Path(value.output).write_text("unrelated")
    with pytest.raises(ValueError, match="already exists"):
        entry.validate(value)
    Path(value.output).unlink()
    Path(value.output + ".partial").write_text("unrelated partial")
    with pytest.raises(ValueError, match="already exists"):
        entry.validate(value)


def test_tsv_new_and_existing(tmp_path):
    value = args(tmp_path, format="tsv")
    assert entry.validate(value)[1]
    Path(value.output).write_text("old tsv")
    assert not entry.validate(value)[1]


def test_manifest_belongs_to_models(tmp_path):
    value = args(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    value.manifest = str(manifest)
    with pytest.raises(ValueError, match="must belong"):
        entry.validate(value)


def test_large_raw_input(tmp_path):
    value = args(tmp_path)
    with open(value.input, "wb") as handle:
        handle.truncate(entry.MAX_INPUT + 1)
    with pytest.raises(ValueError, match="20 MiB"):
        entry.validate(value)


def test_relative_paths_rejected(tmp_path):
    value = args(tmp_path, models="relative")
    with pytest.raises(ValueError, match="Absolute"):
        entry.validate(value)


def test_source_dispatch_is_fixed():
    command = entry.child_command(["--models", "x"])
    assert command[0] == sys.executable
    assert Path(command[1]).name == "local-ai-ocr-entry.py"
    assert command[2] == "--_runtime-core"


def test_frozen_dispatch_is_same_exe(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert entry.child_command(["--models", "x"]) == [sys.executable, "--_runtime-core", "--models", "x"]


def test_remote_rejected_before_target_access(monkeypatch):
    from localocr import bundles
    touched=[]
    monkeypatch.setattr(bundles, "require_local_path", lambda path: (_ for _ in ()).throw(ValueError("remote")))
    monkeypatch.setattr(bundles, "_directory", lambda path: touched.append(path))
    with pytest.raises(ValueError, match="remote"):
        entry.local_path(str(Path.cwd() / "remote"), new=True)
    assert touched == []


def test_core_bounds_before_payload_write(monkeypatch):
    worker = argparse.Namespace(serialize_result=lambda *a: b"x" * (entry.MAX_OUTPUT + 1),
                                serialize_json_result=lambda *a: b"{}")
    def fake_main():
        worker.serialize_result(None)
    worker.main = fake_main
    monkeypatch.setattr(entry, "validate", lambda args: (None, False))
    monkeypatch.setattr(entry, "load_worker", lambda: worker)
    old_argv = sys.argv[:]
    try:
        with pytest.raises(ValueError, match="8 MiB"):
            entry.core(argparse.Namespace(network_audit_self_test=False, owned_png=False, input=None, output=None,
                models="x", manifest=None, model_version=None, format="tsv", self_test=True), [])
    finally:
        sys.argv = old_argv

@pytest.mark.skipif(sys.platform != 'win32', reason='Windows kill-on-close job acceptance')
def test_real_windows_job_kills_core_on_parent_cancel(tmp_path):
    import time
    import psutil
    root = Path(__file__).resolve().parents[1]
    models = root / 'artifacts/model-store/versions/baseline-1'
    if not models.is_dir():
        pytest.skip('Local baseline artifact unavailable')
    command=[sys.executable, str(root/'scripts/local-ai-ocr-entry.py'), '--models', str(models),
             '--manifest', str(models/'manifest.json'), '--self-test']
    process=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    children=[]
    try:
        limit=time.monotonic()+10
        while time.monotonic()<limit and process.poll() is None:
            children=psutil.Process(process.pid).children(recursive=True)
            if children:
                break
            time.sleep(.02)
        assert children, 'Broker never spawned real core'
        process.kill()
        process.wait(timeout=5)
        limit=time.monotonic()+5
        while time.monotonic()<limit and any(child.is_running() for child in children):
            time.sleep(.02)
        assert not any(child.is_running() for child in children), 'Cancelled broker orphaned inference core'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.communicate(timeout=5)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows deadline acceptance')
def test_real_source_broker_deadline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    models = root/'artifacts/model-store/versions/baseline-1'
    if not models.is_dir():
        pytest.skip('Local baseline artifact unavailable')
    result=subprocess.run([sys.executable, str(root/'scripts/local-ai-ocr-entry.py'),
        '--models',str(models),'--manifest',str(models/'manifest.json'),'--self-test','--timeout','.01'],
        capture_output=True,text=True,timeout=10)
    assert result.returncode == 124
    assert 'deadline exceeded' in result.stderr


def test_private_core_direct_flag_rejected(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/local-ai-ocr-entry.py'),
        '--_runtime-core','--models',str(tmp_path),'--self-test'], capture_output=True,text=True,timeout=5)
    assert result.returncode == 1
    assert 'supervising broker' in result.stderr


def test_new_output_lstat_detects_dangling_entry(monkeypatch,tmp_path):
    from localocr import bundles
    monkeypatch.setattr(bundles,'_directory',lambda path: None)
    monkeypatch.setattr(Path,'exists',lambda path: False)
    monkeypatch.setattr(Path,'lstat',lambda path: object())
    with pytest.raises(ValueError,match='already exists'):
        entry.local_path(str(tmp_path/'dangling.partial'),new=True)


def test_foreign_output_race_and_partial_preserved(monkeypatch,tmp_path):
    import io
    value=args(tmp_path)
    foreign=Path(value.output)
    partial=Path(value.output+'.partial')
    real_validate=entry.validate
    def racing_validate(parsed):
        result=real_validate(parsed)
        foreign.write_bytes(b'foreign result')
        partial.write_bytes(b'foreign partial')
        return result
    class Child:
        def __init__(self, command, **kwargs):
            self.stdin=io.BytesIO()
            self.path=Path(command[command.index('--output')+1])
        def wait(self,timeout=None):
            self.path.write_bytes(b'{}')
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(entry,'validate',racing_validate)
    monkeypatch.setattr(entry,'establish_job',lambda: None)
    monkeypatch.setattr(entry.subprocess,'Popen',Child)
    result=entry.main(['--models',value.models,'--input',value.input,'--output',value.output,'--format','json'])
    assert result == 1
    assert foreign.read_bytes() == b'foreign result'
    assert partial.read_bytes() == b'foreign partial'
    assert not Path(value.output+'.runtime-claim').exists()


def test_existing_claim_is_not_removed(monkeypatch,tmp_path):
    value=args(tmp_path)
    claim=Path(value.output+'.runtime-claim')
    claim.mkdir()
    (claim/'foreign').write_text('keep')
    monkeypatch.setattr(entry,'establish_job',lambda: None)
    result=entry.main(['--models',value.models,'--input',value.input,'--output',value.output,'--format','json'])
    assert result == 1
    assert (claim/'foreign').read_text() == 'keep'


def test_tsv_dangling_output_is_rejected(monkeypatch,tmp_path):
    value=args(tmp_path,format='tsv')
    output=Path(value.output)
    original_lstat=Path.lstat
    original_exists=Path.exists
    def output_lstat(path,*args,**kwargs):
        if path == output:
            import os
            import stat
            return os.stat_result((stat.S_IFLNK|0o777,0,0,1,0,0,0,0,0,0))
        return original_lstat(path,*args,**kwargs)
    def output_exists(path):
        return False if path == output else original_exists(path)
    monkeypatch.setattr(Path,'lstat',output_lstat)
    monkeypatch.setattr(Path,'exists',output_exists)
    with pytest.raises(ValueError,match='reparse'):
        entry.validate(value)


def test_dynamic_worker_load_never_writes_cache(monkeypatch,tmp_path):
    worker=tmp_path/'fixed-worker.py'
    worker.write_text('VALUE = 37\n',encoding='utf-8')
    monkeypatch.setattr(entry,'worker_path',lambda: worker)
    monkeypatch.setattr(sys,'dont_write_bytecode',False)
    module=entry.load_worker()
    assert module.VALUE == 37
    assert not (tmp_path/'__pycache__').exists()
    assert {path.name for path in tmp_path.iterdir()} == {'fixed-worker.py'}
    assert sys.dont_write_bytecode is False
