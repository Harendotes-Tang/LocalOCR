"""Deterministic kernel-contract tests; never open or terminate a real process."""
from types import SimpleNamespace
import subprocess
import time

import pytest

from tools.PretrainedOcr import _owned_job as owned


class Api:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.active_count = 1

    def call(self, name, *args):
        self.calls.append((name, *args))
        if name == self.failure:
            raise PermissionError('fixture '+name)

    def create_job(self):
        self.call('create_job')
        return 11

    def close(self, handle):
        self.call('close', handle)

    def process_id(self, handle):
        self.call('process_id', handle)
        return 123

    def assign(self, job, process):
        self.call('assign', job, process)

    def threads(self, pid):
        self.call('threads', pid)
        return [456]

    def open_thread(self, tid):
        self.call('open_thread', tid)
        return 22

    def thread_owner(self, handle):
        self.call('thread_owner', handle)
        return 123

    def resume(self, handle):
        self.call('resume', handle)
        return 1

    def active(self, handle):
        self.call('active', handle)
        return self.active_count

    def terminate(self, handle):
        self.call('terminate', handle)
        self.active_count = 0


class Process:
    pid = 123
    _handle = 33
    returncode = None

    def __init__(self):
        self.calls = []

    def poll(self):
        return self.returncode

    def kill(self):
        self.calls.append(('kill', self._handle))
        self.returncode = 15

    def wait(self, timeout):
        self.calls.append(('wait', timeout))
        self.returncode = 15 if self.returncode is None else self.returncode
        return self.returncode


def test_confine_pinned_root_before_resuming_exact_initial_thread():
    api, process = Api(), Process()
    job = owned.NativeJob(api)
    job.attach_and_resume(process)
    assert api.calls == [('create_job',), ('process_id', 33), ('assign', 11, 33), ('threads', 123),
                         ('open_thread', 456), ('process_id', 33), ('thread_owner', 22),
                         ('resume', 22), ('close', 22)]
    assert job.assigned and job.process_handle == process._handle
    job.stop(process, time.monotonic()+15)
    assert ('terminate', 11) in api.calls and api.calls.count(('close', 11)) == 1
    assert not any(call[0] == 'kill' for call in process.calls)


@pytest.mark.parametrize('failure', ['assign', 'threads', 'open_thread', 'thread_owner', 'resume'])
def test_partial_setup_failure_never_resumes_unconfined_root_and_closes_handles(failure):
    api, process = Api(failure), Process()
    job = owned.NativeJob(api)
    with pytest.raises(PermissionError):
        job.attach_and_resume(process)
    if failure != 'resume':
        assert not any(call[0] == 'resume' for call in api.calls)
    if failure == 'assign':
        api.active_count = 0  # Root is still suspended outside the empty fresh job.
    job.stop(process, time.monotonic()+15)
    assert api.calls.count(('close', 11)) == 1
    assert api.calls.count(('close', 22)) == (1 if failure in ('thread_owner', 'resume') else 0)
    assert (('kill', 33) in process.calls) is (failure == 'assign')


@pytest.mark.parametrize('boundary', ['root-handle-mismatch', 'thread-reused', 'multiple-threads', 'suspend-count-zero', 'suspend-count-two'])
def test_replaced_or_ambiguous_identity_fails_before_unrelated_resume(boundary):
    api, process = Api(), Process()
    if boundary == 'root-handle-mismatch':
        api.process_id = lambda handle: 999
    elif boundary == 'thread-reused':
        api.thread_owner = lambda handle: 999
    elif boundary == 'multiple-threads':
        api.threads = lambda pid: [456, 789]
    else:
        def ambiguous_resume(handle):
            api.calls.append(('resume', handle))
            return 0 if boundary == 'suspend-count-zero' else 2
        api.resume = ambiguous_resume
    job = owned.NativeJob(api)
    with pytest.raises(RuntimeError):
        job.attach_and_resume(process)
    assert api.calls.count(('resume', 22)) == int(boundary.startswith('suspend-count'))
    if not job.assigned:
        api.active_count = 0
    if boundary == 'root-handle-mismatch':
        with pytest.raises(RuntimeError, match='handle identity differs'):
            job.stop(process, time.monotonic()+15)
        assert not process.calls and ('terminate', 11) not in api.calls
    else:
        job.stop(process, time.monotonic()+15)


def test_exited_root_with_live_reparented_children_still_terminates_whole_confined_job():
    api, process = Api(), Process()
    job = owned.NativeJob(api)
    job.attach_and_resume(process)
    process.returncode = 0
    api.active_count = 3  # Children/grandchild remain after root exit.
    job.stop(process, time.monotonic()+15)
    assert ('terminate', 11) in api.calls
    assert process.returncode == 0 and api.active_count == 0


@pytest.mark.parametrize('failure', ['active', 'terminate'])
def test_query_or_termination_denied_never_confirms_cleanup(failure):
    api, process = Api(), Process()
    job = owned.NativeJob(api)
    job.attach_and_resume(process)
    process.returncode = 0
    api.failure = failure
    with pytest.raises(PermissionError):
        job.stop(process, time.monotonic()+15)
    assert not process.calls and api.calls.count(('close', 11)) == 1
    with pytest.raises(RuntimeError, match='no longer available'):
        job.stop(process, time.monotonic()+15)


def test_live_job_uses_one_shared_fifteen_second_deadline(monkeypatch):
    api, process = Api(), Process()
    job = owned.NativeJob(api)
    job.attach_and_resume(process)
    api.terminate = lambda handle: None  # Native call returns but child is not confirmed gone.
    ticks = iter([10., 20., 26.])
    monkeypatch.setattr(owned.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(owned.time, 'sleep', lambda seconds: None)
    with pytest.raises(subprocess.TimeoutExpired):
        job.stop(process, 25.)
    assert not process.calls and api.calls.count(('close', 11)) == 1


def test_successful_wait_receives_only_remaining_original_budget(monkeypatch):
    api, process = Api(), Process()
    job = owned.NativeJob(api)
    job.attach_and_resume(process)
    ticks = iter([10., 24., 24.5])
    monkeypatch.setattr(owned.time, 'monotonic', lambda: next(ticks))
    job.stop(process, 25.)
    assert process.calls == [('wait', 1.)]
