"""Negative resource controls and real owned-process receipts for the new round."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

from tools.PretrainedOcr.common import digest, read, write
from tools.PretrainedOcr.round_limits import BoundGPUBudget, RoundLimits, supervise


@pytest.fixture
def limits(tmp_path):
    old = tmp_path / 'old.json'
    write(old, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 7200.0,
                'used_seconds': 4.0, 'runs': [{'elapsed_seconds': 4.0}], 'active': None})
    return RoundLimits(tmp_path / 'new', old, digest(old))


def test_history_binding_is_immutable_and_closed(limits):
    budget = BoundGPUBudget(limits)
    state = budget.initialize()
    assert state['round_binding']['previous_used_seconds'] == 4
    previous = read(limits.previous_budget)
    previous['active'] = {'pid': 1}
    write(limits.previous_budget, previous)
    with pytest.raises(ValueError, match='Historical GPU ledger changed'):
        budget.initialize()
    fresh = RoundLimits(limits.root, limits.previous_budget, digest(limits.previous_budget))
    with pytest.raises(RuntimeError, match='not closed'):
        fresh.audit()


def test_round_identity_cannot_be_replaced(limits):
    budget = BoundGPUBudget(limits)
    budget.initialize()
    value = read(budget.path)
    value['round_binding']['previous_used_seconds'] = 0
    write(budget.path, value)
    with pytest.raises(ValueError, match='binding changed'):
        budget.initialize()


def test_disk_and_actual_artifact_reservations(limits, monkeypatch):
    limits.root.mkdir()
    (limits.root / 'file').write_bytes(b'x' * 20)
    small = RoundLimits(limits.root, limits.previous_budget, limits.previous_sha256, maximum_bytes=24)
    assert small.audit()['artifact_bytes'] == 20
    with pytest.raises(RuntimeError, match='quota'):
        small.audit(5)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(RuntimeError, match='8 GiB'):
        limits.audit()


@pytest.mark.parametrize('directory', [False, True])
def test_audit_counts_survivors_when_enumerated_entry_disappears(limits, monkeypatch, directory):
    limits.root.mkdir()
    (limits.root / 'kept').write_bytes(b'x' * 20)
    transient = limits.root / 'browser-scratch'
    transient.mkdir() if directory else transient.write_bytes(b'temporary')
    original = Path.lstat

    def removed_during_scan(path, *args, **kwargs):
        if path == transient:
            transient.rmdir() if directory else transient.unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'lstat', removed_during_scan)
    result = limits.audit()
    assert result['artifact_bytes'] == 20 and result['file_count'] == 1
    small = RoundLimits(limits.root, limits.previous_budget, limits.previous_sha256, maximum_bytes=24)
    with pytest.raises(RuntimeError, match='quota'):
        small.audit(5)


@pytest.mark.parametrize('error', [PermissionError, OSError])
def test_audit_does_not_ignore_other_inventory_errors(limits, monkeypatch, error):
    limits.root.mkdir()
    unreadable = limits.root / 'unreadable'
    unreadable.write_bytes(b'payload')
    original = Path.lstat

    def denied(path, *args, **kwargs):
        if path == unreadable:
            raise error('inventory unavailable')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'lstat', denied)
    with pytest.raises(error, match='inventory unavailable'):
        limits.audit()


def test_open_lease_and_exception_are_not_reset(limits):
    budget = BoundGPUBudget(limits)
    budget.initialize()
    with pytest.raises(LookupError):
        with budget.run('failed-native', 30):
            time.sleep(0.02)
            with pytest.raises(RuntimeError, match='Unfinished'):
                budget._state()
            raise LookupError('failure')
    after = read(budget.path)
    assert after['active'] is None and after['runs'][0]['status'] == 'failed'
    assert after['used_seconds'] > 0
    assert limits.history() == 4


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True])
def test_invalid_usage_refused(limits, value):
    budget = BoundGPUBudget(limits)
    budget.initialize()
    state = read(budget.path)
    state['used_seconds'] = value
    budget.path.write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(ValueError):
        budget.initialize()


def test_paths_cannot_escape_or_reuse_parent_ledger(limits):
    with pytest.raises(ValueError, match='inside'):
        limits.confined(limits.root.parent / 'outside.log')
    with pytest.raises(ValueError, match='Native receipt'):
        supervise(limits, 'bad-child', 30, [sys.executable, '-c', 'pass'], limits.root / 'native.log',
                  child_budget=limits.root / 'gpu-budget.json')


def test_actual_process_output_is_closed_before_hash(limits):
    result = supervise(limits, 'cpu-process-contract-probe', 30,
                       [sys.executable, '-c', 'print("native-output")'], limits.root / 'probe.log')
    assert result['status'] == 'completed' and result['native_exit_code'] == 0
    assert result['log_sha256'] == digest(result['log'])
    assert Path(result['log']).read_text().strip() == 'native-output'
    assert read(result['outer_budget'])['active'] is None
    assert result['closed_run']['elapsed_seconds'] > 0
    assert result['closed_run']['cleanup_confirmed'] is True
    assert result['allocator_zero_observed'] is False


def test_hard_deadline_terminates_owned_native_process(limits):
    with pytest.raises(RuntimeError, match='hard-deadline'):
        supervise(limits, 'deadline-cpu-control', 1.5,
                  [sys.executable, '-c', 'import time;time.sleep(30)'], limits.root / 'deadline.log',
                  cleanup_seconds=1)
    receipt = read(limits.root / 'deadline.log.receipt.json')
    assert receipt['stop_reason'] == 'hard-deadline'
    assert receipt['native_exit_code'] is not None
    assert read(receipt['outer_budget'])['active'] is None


def test_stdout_failure_still_stops_native_before_closing(limits, monkeypatch):
    import builtins
    import tools.PretrainedOcr.round_limits as module
    captured = []
    original_stop = module._stop_owned
    def record_stop(process):
        captured.append(process)
        original_stop(process)
    def broken(*args, **kwargs):
        raise BrokenPipeError('closed output pipe')
    monkeypatch.setattr(module, '_stop_owned', record_stop)
    monkeypatch.setattr(builtins, 'print', broken)
    with pytest.raises((BrokenPipeError, RuntimeError)):
        supervise(limits, 'stdout-cpu-control', 30,
                  [sys.executable, '-c', 'import time;time.sleep(30)'], limits.root / 'stdout.log')
    assert len(captured) == 1 and captured[0].poll() is not None
    assert read(limits.root / 'gpu-budget.json')['active'] is None


def test_failed_native_cleanup_keeps_exclusive_active_lease(limits, monkeypatch):
    import subprocess
    import tools.PretrainedOcr.round_limits as module
    captured = []
    original_stop = module._stop_owned
    previous = supervise(limits, 'earlier-closed-cpu-control', 30,
                         [sys.executable, '-c', 'pass'], limits.root / 'previous.log')
    assert previous['closed_run']['label'] == 'earlier-closed-cpu-control'
    def failed_stop(process):
        captured.append(process)
        raise subprocess.TimeoutExpired('taskkill', 1)
    monkeypatch.setattr(module, '_stop_owned', failed_stop)
    try:
        with pytest.raises(RuntimeError):
            supervise(limits, 'cleanup-cpu-control', 1.5,
                      [sys.executable, '-c', 'import time;time.sleep(30)'], limits.root / 'cleanup.log',
                      cleanup_seconds=1)
        budget = BoundGPUBudget(limits)
        state = read(budget.path)
        assert captured[0].poll() is None
        assert state['active']['native_pid'] == captured[0].pid
        assert state['active']['audit_required'] is True
        assert budget.lock.exists()
        assert len(state['runs']) == 1
        receipt = read(limits.root / 'cleanup.log.receipt.json')
        assert receipt['closed_run'] is None
        with pytest.raises(FileExistsError):
            with budget.run('second', 30):
                pytest.fail('An unclosed native process must block another lease')
    finally:
        for process in captured:
            original_stop(process)


def test_exited_root_without_confirmed_owned_tree_keeps_active_lease(limits):
    budget = BoundGPUBudget(limits)
    process = SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0,
                              _round_cleanup_confirmed=False)
    with pytest.raises(RuntimeError, match='unconfirmed child'):
        with budget.run('root-exited-child-unconfirmed', 30):
            budget.attach_native(process)
            raise RuntimeError('unconfirmed child')
    state = read(budget.path)
    assert state['active']['audit_required'] is True
    assert state['active']['cleanup_confirmed'] is False
    assert state['runs'] == [] and budget.lock.exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows confined-job setup contract')
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_supervisor_setup_failure_binds_root_before_cleanup_and_retains_uncertain_lease(limits, monkeypatch, cleanup_fails):
    from tools.PretrainedOcr import _owned_job as owned, round_limits as module
    events = []
    process = SimpleNamespace(pid=12345, returncode=None, _handle=33)
    process.poll = lambda: process.returncode
    class Job:
        def attach_and_resume(self, actual):
            assert actual is process
            state = read(limits.root/'gpu-budget.json')
            assert state['active']['native_pid'] == process.pid
            events.append('attach')
            raise PermissionError('fixture resume denied')
        def stop(self, actual, deadline):
            assert actual is process and deadline <= time.monotonic()+15
            events.append('stop')
            process.returncode = 15
            if cleanup_fails:
                raise PermissionError('fixture job query denied after root exit')
        def close(self):
            events.append('close')
    def spawn(*args, **kwargs):
        assert kwargs['creationflags'] == owned.CREATE_SUSPENDED
        return process
    monkeypatch.setattr(owned, 'NativeJob', Job)
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    with pytest.raises(RuntimeError, match='fixture'):
        supervise(limits, 'partial-setup', 30, ['fixture-only'], limits.root/'partial.log')
    state = read(limits.root/'gpu-budget.json')
    receipt = read(limits.root/'partial.log.receipt.json')
    assert events == ['attach', 'stop'] and receipt['native_exit_confirmed'] is True
    if cleanup_fails:
        assert state['active']['audit_required'] is True and state['runs'] == []
        assert receipt['closed_run'] is None and BoundGPUBudget(limits).lock.exists()
    else:
        assert state['active'] is None and state['runs'][-1]['status'] == 'failed'
        assert receipt['closed_run']['cleanup_confirmed'] is True


@pytest.mark.skipif(os.name != 'nt', reason='Windows native job/WMI independence')
@pytest.mark.parametrize('failure', ['call cancelled', 'quota violation'])
def test_native_cleanup_does_not_depend_on_taskkill_wmi(limits, monkeypatch, failure):
    from tools.PretrainedOcr import round_limits as module
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout=b'', stderr=failure.encode())
    monkeypatch.setattr(module.subprocess, 'run', unavailable)
    with pytest.raises(RuntimeError, match='hard-deadline'):
        supervise(limits, 'wmi-independent', 1.5,
                  [sys.executable, '-c', 'import time;time.sleep(30)'], limits.root/'wmi.log', cleanup_seconds=1)
    receipt = read(limits.root/'wmi.log.receipt.json')
    assert calls == [] and receipt['native_exit_confirmed'] is True
    assert receipt['closed_run']['cleanup_confirmed'] is True


@pytest.mark.skipif(os.name != 'nt', reason='Windows descendant inheritance after root exit')
def test_naturally_exited_root_cleans_live_child_and_grandchild(limits):
    import psutil
    marker = limits.root/'descendants.json'
    child_code = (
        "import subprocess,sys,time,json,os,psutil; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "rows=[{'pid':os.getpid(),'create_time':psutil.Process().create_time()},"
        "{'pid':p.pid,'create_time':psutil.Process(p.pid).create_time()}]; "
        f"marker=Path({str(marker)!r}); temporary=marker.with_suffix('.tmp'); "
        "temporary.write_text(json.dumps(rows)); temporary.replace(marker); time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"marker=Path({str(marker)!r}); deadline=time.monotonic()+10\n"
        "while not marker.exists():\n"
        " if time.monotonic()>deadline: raise RuntimeError('child marker timeout')\n"
        " time.sleep(.01)\n"
    )
    receipt = supervise(limits, 'root-exits-with-descendants', 30,
                        [sys.executable, '-c', parent_code], limits.root/'descendants.log')
    assert receipt['native_exit_code'] == 0 and receipt['closed_run']['cleanup_confirmed'] is True
    for identity in read(marker):
        try:
            process = psutil.Process(identity['pid'])
            assert process.create_time() != identity['create_time'] or not process.is_running()
        except psutil.NoSuchProcess:
            pass


@pytest.mark.parametrize('field', ['maximum_bytes', 'minimum_free'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True, 1.5])
def test_resource_configuration_cannot_disable_guards(limits, field, invalid):
    with pytest.raises(ValueError, match='authorization'):
        RoundLimits(limits.root, limits.previous_budget, limits.previous_sha256, **{field: invalid})


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True, -1, 7201])
def test_new_budget_cannot_change_authorization(limits, invalid):
    with pytest.raises(ValueError):
        BoundGPUBudget(limits, invalid)


def test_zero_second_ledger_binds_history_and_refuses_every_lease(limits, capsys):
    """A CPU-only round keeps a closed, bound ledger whose ceiling is 0 s: no lease can ever open."""
    from tools.PretrainedOcr import round_limits as rl
    budget = BoundGPUBudget(limits, 0.0)
    state = budget.initialize()
    assert state['maximum_seconds'] == 0.0 and state['used_seconds'] == 0.0 and state['active'] is None
    assert state['round_binding']['previous_used_seconds'] == 4
    with pytest.raises(RuntimeError, match='exhausted'):
        with budget.run('never', 10):
            pass
    after = read(budget.path)
    assert after['used_seconds'] == 0.0 and after['runs'] == [] and after['active'] is None
    assert not budget.lock.exists()
    # The CLI exposes the same ceiling explicitly; the default stays the authorized 7200 s.
    rl.main(['initialize', '--root', str(limits.root), '--previous-budget', str(limits.previous_budget),
             '--previous-sha256', limits.previous_sha256, '--maximum-seconds', '0'])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['maximum_seconds'] == 0.0
    with pytest.raises(ValueError):
        BoundGPUBudget(limits, 7200.0).initialize()  # identity of the existing zero-second ledger cannot change


def test_zero_second_ledger_is_a_valid_closed_predecessor(tmp_path):
    old = tmp_path / 'zero.json'
    write(old, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 0.0,
                'used_seconds': 0.0, 'runs': [], 'active': None})
    assert RoundLimits(tmp_path / 'next', old, digest(old)).history() == 0.0


@pytest.mark.parametrize('field', ['seconds', 'cleanup_seconds'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True, -1])
def test_supervisor_deadline_configuration_is_finite(limits, field, invalid):
    config = {'seconds': 30, 'cleanup_seconds': 1, field: invalid}
    with pytest.raises(ValueError):
        supervise(limits, 'bad-deadline', command=[sys.executable, '-c', 'pass'],
                  log=limits.root / 'invalid.log', **config)
    assert not (limits.root / 'gpu-budget.json').exists()


def test_inventory_respects_native_cleanup_deadline(limits):
    from tools.PretrainedOcr.common import Deadline
    limits.root.mkdir()
    (limits.root / 'actual').write_bytes(b'x')
    with pytest.raises(TimeoutError, match='cleanup deadline'):
        limits.audit(deadline=Deadline(0), minimum_remaining=1)


def test_authorized_two_gib_quota_is_explicit_and_bounded(limits, capsys):
    from tools.PretrainedOcr import round_limits as rl
    two = 2 * 1024**3
    wide = RoundLimits(limits.root, limits.previous_budget, limits.previous_sha256, maximum_bytes=two)
    assert wide.binding()['maximum_artifact_bytes'] == two
    with pytest.raises(ValueError, match='authorization'):
        RoundLimits(limits.root, limits.previous_budget, limits.previous_sha256, maximum_bytes=two + 1)
    limits.root.mkdir()
    (limits.root / 'actual').write_bytes(b'x' * 16)
    small = RoundLimits(limits.root, limits.previous_budget, limits.previous_sha256, maximum_bytes=16)
    small.audit(0)
    with pytest.raises(RuntimeError, match='exceed 16 bytes'):
        small.audit(1)
    rl.main(['audit', '--root', str(limits.root), '--previous-budget', str(limits.previous_budget),
             '--previous-sha256', limits.previous_sha256, '--maximum-bytes', str(two)])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['binding']['maximum_artifact_bytes'] == two and printed['artifact_bytes'] == 16
    assert rl.describe_bytes(1024**3) == '1 GiB' and rl.describe_bytes(two) == '2 GiB'


def test_lease_is_capped_to_the_remaining_budget_and_refused_once_spent(tmp_path, monkeypatch):
    """A lease can never run past the authorization: longer requests are capped, a spent ledger refuses."""
    old = tmp_path / 'closed.json'
    write(old, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 0.0, 'used_seconds': 0.0, 'runs': [], 'active': None})
    limits = RoundLimits(tmp_path / 'round', old, digest(old))
    budget = BoundGPUBudget(limits, 7200.0)
    state = budget.initialize()
    state['used_seconds'] = 7000.0
    state['runs'] = [{'label': 'earlier', 'elapsed_seconds': 7000.0}]
    write(budget.path, state)
    with budget.run('capped', 2400) as deadline:
        assert deadline.limit == pytest.approx(200.0)
    spent = read(budget.path)
    spent['used_seconds'] = 7200.0
    spent['runs'] = [{'label': 'earlier', 'elapsed_seconds': 7200.0}]
    write(budget.path, spent)
    with pytest.raises(RuntimeError, match='exhausted'):
        with budget.run('refused', 10):
            pass
    assert read(budget.path)['used_seconds'] == 7200.0 and not budget.lock.exists()


def test_tampered_predecessor_ledger_blocks_the_new_round(tmp_path):
    old = tmp_path / 'closed.json'
    write(old, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 0.0, 'used_seconds': 0.0, 'runs': [], 'active': None})
    limits = RoundLimits(tmp_path / 'round', old, digest(old))
    write(old, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 0.0, 'used_seconds': 1.0,
                'runs': [{'elapsed_seconds': 1.0}], 'active': None})
    with pytest.raises(ValueError, match='changed'):
        BoundGPUBudget(limits, 7200.0).initialize()


def test_zero_second_round_bound_to_a_closed_training_ledger_refuses_every_lease(tmp_path):
    """A CPU-only round may follow a closed GPU training ledger: the binding records its usage, no lease opens."""
    old = tmp_path / 'trained.json'
    write(old, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 7200.0, 'used_seconds': 2376.438000000001,
                'runs': [{'label': 'train', 'elapsed_seconds': 2376.438000000001}], 'active': None})
    limits = RoundLimits(tmp_path / 'round', old, digest(old))
    budget = BoundGPUBudget(limits, 0.0)
    state = budget.initialize()
    assert state['maximum_seconds'] == 0.0 and state['used_seconds'] == 0.0 and state['runs'] == []
    assert state['round_binding']['previous_used_seconds'] == 2376.438000000001
    with pytest.raises(RuntimeError, match='exhausted'):
        with budget.run('never', 10):
            pass
    assert read(budget.path)['used_seconds'] == 0.0 and not budget.lock.exists()
    tampered = read(old)
    tampered['used_seconds'] = 0.0
    tampered['runs'] = []
    write(old, tampered)
    with pytest.raises(ValueError, match='changed'):
        budget.initialize()
