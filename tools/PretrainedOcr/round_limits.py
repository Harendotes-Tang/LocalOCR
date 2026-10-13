"""Bind an optimization round to closed history and supervise its resource lease.

The outer ledger is the sole authorization counter. Native adapter ledgers are
child-process receipts; their elapsed time is already included in the outer
lease. This module never removes historical files or resets an unfinished lease.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

from .common import Deadline, GPUBudget, digest, read, utc, write

MAXIMUM_SECONDS = 7200.0
MAXIMUM_ARTIFACT_BYTES = 1024**3
AUTHORIZED_ARTIFACT_CEILING = 2 * 1024**3
MINIMUM_FREE_BYTES = 8 * 1024**3


def describe_bytes(value):
    return f'{value // 1024**3} GiB' if value % 1024**3 == 0 else f'{value} bytes'


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid {name}')
    return float(value)


def _no_reparse(path):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.exists():
            info = parent.lstat()
            if parent.is_symlink() or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
                raise ValueError('Research resource path contains a reparse point')
    return path.resolve()


class RoundLimits:
    """Validate actual bytes/free space and the immutable predecessor ledger."""
    def __init__(self, root, previous_budget, previous_sha256, *, maximum_bytes=MAXIMUM_ARTIFACT_BYTES,
                 minimum_free=MINIMUM_FREE_BYTES):
        self.root = _no_reparse(root)
        self.previous_budget = _no_reparse(previous_budget)
        if self.previous_budget.is_relative_to(self.root):
            raise ValueError('Historical ledger must be outside the new round')
        if not isinstance(previous_sha256, str) or len(previous_sha256) != 64:
            raise ValueError('Historical ledger SHA256 is required')
        self.previous_sha256 = previous_sha256
        if (type(maximum_bytes) is not int or type(minimum_free) is not int
                or not 0 < maximum_bytes <= AUTHORIZED_ARTIFACT_CEILING or minimum_free < MINIMUM_FREE_BYTES):
            raise ValueError('Resource limits exceed user authorization')
        self.maximum_bytes = maximum_bytes
        self.minimum_free = minimum_free

    def history(self):
        if digest(self.previous_budget) != self.previous_sha256:
            raise ValueError('Historical GPU ledger changed')
        state = read(self.previous_budget)
        maximum = _finite(state.get('maximum_seconds'), 'historical maximum')
        used = _finite(state.get('used_seconds'), 'historical usage')
        # A closed zero-second ledger (CPU-only round) is a valid predecessor: it binds history without usage.
        if state.get('schema_version') != 'localocr.gpu-budget.v1' or not 0 <= maximum <= MAXIMUM_SECONDS or used > maximum:
            raise ValueError('Invalid historical GPU ledger')
        if state.get('active') is not None or self.previous_budget.with_name(self.previous_budget.name + '.lock').exists():
            raise RuntimeError('Historical GPU lease is not closed')
        runs = state.get('runs')
        if not isinstance(runs, list) or not math.isclose(sum(_finite(r.get('elapsed_seconds'), 'historical run') for r in runs), used,
                                                        rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Historical elapsed intervals do not match usage')
        return used

    def binding(self):
        return {'schema_version': 'localocr.gpu-round-binding.v1',
                'round_root': str(self.root), 'previous_budget': str(self.previous_budget),
                'previous_sha256': self.previous_sha256, 'previous_used_seconds': self.history(),
                'maximum_artifact_bytes': self.maximum_bytes, 'minimum_free_bytes': self.minimum_free}

    def confined(self, path):
        path = _no_reparse(path)
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('New output must remain inside its round root')
        return path

    def audit(self, reserve_bytes=0, *, deadline=None, minimum_remaining=0):
        self.history()
        if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int) or reserve_bytes < 0:
            raise ValueError('Expected output reservation must be a nonnegative integer')
        size = count = 0
        _no_reparse(self.root)
        if self.root.exists():
            for directory, directories, files in os.walk(self.root, followlinks=False):
                if deadline is not None and deadline.remaining() <= minimum_remaining:
                    raise TimeoutError('Resource inventory reached native cleanup deadline')
                for name in (*directories, *files):
                    if deadline is not None and deadline.remaining() <= minimum_remaining:
                        raise TimeoutError('Resource inventory reached native cleanup deadline')
                    path = Path(directory) / name
                    info = path.lstat()
                    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                        raise ValueError('Research resource path contains a reparse point')
                    if stat.S_ISREG(info.st_mode):
                        size += info.st_size
                        count += 1
        _no_reparse(self.root)
        existing = self.root
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if size + reserve_bytes > self.maximum_bytes:
            raise RuntimeError('New round artifact quota would exceed ' + describe_bytes(self.maximum_bytes))
        if free - reserve_bytes < self.minimum_free:
            raise RuntimeError('Research disk reserve would fall below ' + describe_bytes(self.minimum_free))
        return {'created_utc': utc(), 'artifact_bytes': size, 'file_count': count,
                'reserved_bytes': reserve_bytes, 'free_bytes': free, 'binding': self.binding()}


class BoundGPUBudget(GPUBudget):
    """Persist the historical binding alongside common's exclusive total lease."""
    def __init__(self, limits, maximum_seconds=MAXIMUM_SECONDS):
        # A 0 s ceiling is the explicit CPU-only authorization: the ledger exists and binds
        # history, and every lease is refused as exhausted before any native process starts.
        if not 0 <= _finite(maximum_seconds, 'new maximum') <= MAXIMUM_SECONDS:
            raise ValueError('GPU budget exceeds user authorization')
        self.limits = limits
        self.native_process = None
        self.active_record = None
        super().__init__(limits.confined(limits.root / 'gpu-budget.json'), maximum_seconds)

    @contextlib.contextmanager
    def run(self, label, limit_seconds=2700):
        """Close only after the tracked native process has actually exited.

        A failed termination keeps the original active interval and exclusive
        lock. Its entire elapsed time must be charged by an explicit subsequent
        audit; restarting a supervisor cannot acquire another GPU lease.
        """
        if not label or _finite(limit_seconds, 'lease duration') <= 0:
            raise ValueError('GPU lease needs a label and positive duration')
        self.limits.audit()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, str(os.getpid()).encode('ascii'))
        finally:
            os.close(fd)
        active = None
        closed = False
        try:
            state = self._state()
            available = self.maximum - state['used_seconds']
            if available <= 0:
                raise RuntimeError('Approved GPU budget exhausted')
            deadline = Deadline(min(float(limit_seconds), available))
            active = {'label': label, 'pid': os.getpid(), 'started_utc': utc(), 'limit_seconds': deadline.limit}
            self.active_record = active
            state['active'] = active
            write(self.path, state)
            status = 'completed'
            try:
                yield deadline
                if deadline.expired():
                    status = 'deadline-reached'
            except BaseException:
                status = 'failed'
                raise
            finally:
                if self.native_process is not None and self.native_process.poll() is None:
                    state['active'] = {**active, 'native_pid': self.native_process.pid,
                                       'cleanup_confirmed': False, 'audit_required': True}
                    write(self.path, state)
                else:
                    elapsed = time.monotonic() - deadline.started
                    state['used_seconds'] += elapsed
                    closed_record = dict(active)
                    if self.native_process is not None:
                        closed_record.update(cleanup_confirmed=True, native_exit_code=self.native_process.returncode)
                    state['runs'].append({**closed_record, 'ended_utc': utc(), 'elapsed_seconds': elapsed, 'status': status})
                    state['active'] = None
                    write(self.path, state)
                    closed = True
        finally:
            if (closed or active is None) and self.lock.exists():
                self.lock.unlink()

    def attach_native(self, process, child_budget=None):
        """Persist the actual native identity before monitoring or UI output."""
        self.native_process = process
        if self.active_record is None:
            raise RuntimeError('A native process needs an existing outer lease')
        self.active_record.update(native_pid=process.pid, cleanup_confirmed=False)
        if child_budget is not None:
            self.active_record['child_budget'] = str(self.limits.confined(child_budget))
        state = read(self.path)
        if state.get('active', {}).get('pid') != os.getpid() or state.get('round_binding') != self.limits.binding():
            raise RuntimeError('Outer lease owner changed before native binding')
        state['active'] = self.active_record
        write(self.path, state)

    def _state(self):
        self.limits.audit()
        binding = self.limits.binding()
        exists = self.path.exists()
        state = super()._state()
        if exists and state.get('round_binding') != binding:
            raise ValueError('Optimization round binding changed')
        used = _finite(state.get('used_seconds'), 'current GPU usage')
        if used > self.maximum:
            raise RuntimeError('Approved GPU budget exhausted')
        if not math.isclose(sum(_finite(r.get('elapsed_seconds'), 'current run') for r in state['runs']), used,
                            rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Current elapsed intervals do not match usage')
        state['round_binding'] = binding
        return state

    def initialize(self):
        if self.path.exists():
            return self._state()
        self.limits.audit(8192)
        if self.lock.exists():
            raise RuntimeError('Exclusive GPU lease exists')
        # Exclusive creation prevents a second initializer from replacing a lease.
        state = self._state()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return state


def _stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=15)
    else:
        process.kill()
    process.wait(timeout=15)


def supervise(limits, label, seconds, command, log, *, reserve_bytes=0, child_budget=None, cleanup_seconds=15):
    """Run one owned native process, stop at its deadline, and bind closed output.

    The cleanup interval is withheld from the native process allowance. An
    abnormal child lease remains untouched for a subsequent explicit PID audit.
    No claim about allocator contents is inferred from process termination.
    """
    seconds = _finite(seconds, 'supervised duration')
    cleanup_seconds = _finite(cleanup_seconds, 'cleanup duration')
    if not command or not label or seconds <= cleanup_seconds or cleanup_seconds < 1:
        raise ValueError('Supervised run needs a command and a cleanup reserve')
    log = limits.confined(log)
    child_budget = limits.confined(child_budget) if child_budget else None
    budget = BoundGPUBudget(limits)
    if child_budget == budget.path:
        raise ValueError('Native receipt cannot lock the outer authorization ledger')
    limits.audit(reserve_bytes)
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError('Preserve the prior native run log')
    receipt = log.with_suffix(log.suffix + '.receipt.json')
    if receipt.exists():
        raise FileExistsError('Preserve the prior run receipt')
    started = utc()
    process = None
    reason = 'process-exit'
    failure = None
    before = limits.audit()
    try:
        with budget.run(label, limit_seconds=seconds) as deadline:
            if deadline.limit <= cleanup_seconds:
                raise RuntimeError('Remaining GPU time cannot cover cleanup')
            with log.open('xb', buffering=0) as stream:
                env = os.environ.copy()
                env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                budget.native_process = process
                try:
                    budget.attach_native(process, child_budget)
                    print(json.dumps({'event': 'native-start', 'pid': process.pid, 'label': label,
                                      'lease_limit_seconds': deadline.limit}), flush=True)
                    next_audit = time.monotonic()
                    while process.poll() is None:
                        if deadline.remaining() <= cleanup_seconds:
                            reason = 'hard-deadline'
                            break
                        if time.monotonic() >= next_audit:
                            limits.audit(deadline=deadline, minimum_remaining=cleanup_seconds)
                            next_audit = time.monotonic() + 10
                        time.sleep(0.25)
                finally:
                    _stop_owned(process)
                    stream.flush()
                    os.fsync(stream.fileno())
            if reason != 'process-exit' or process.returncode != 0:
                raise RuntimeError(f'Native run ended: {reason}, exit={process.returncode}')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    after = limits.audit()
    state = read(budget.path)
    closed_run = None
    if state.get('active') is None and state['runs'] and budget.active_record is not None:
        last = state['runs'][-1]
        if (last.get('label'), last.get('pid'), last.get('started_utc')) == (
                label, os.getpid(), budget.active_record['started_utc']):
            closed_run = last
    value = {'schema_version': 'localocr.supervised-gpu-run.v1', 'started_utc': started, 'ended_utc': utc(),
             'label': label, 'command': list(command), 'native_pid': process.pid if process else None,
             'native_exit_code': process.returncode if process else None, 'stop_reason': reason,
             'failure': failure, 'status': 'failed' if failure else 'completed',
             'outer_budget': str(budget.path), 'outer_budget_sha256': digest(budget.path),
             'native_exit_confirmed': process is None or process.poll() is not None,
             'active_lease': state.get('active'),
             'closed_run': closed_run,
             'log': str(log), 'log_sha256': digest(log) if log.exists() else None,
             'child_budget': str(child_budget) if child_budget else None,
             'child_budget_sha256': digest(child_budget) if child_budget and child_budget.exists() else None,
             'before': before, 'after': after, 'allocator_zero_observed': False}
    write(receipt, value)
    print(json.dumps({'event': 'native-end', 'status': value['status'], 'receipt': str(receipt),
                      'used_seconds': state['used_seconds'], 'artifact_bytes': after['artifact_bytes']}), flush=True)
    if failure:
        raise RuntimeError(failure)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize', 'audit', 'run'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--previous-budget', required=True)
    parser.add_argument('--previous-sha256', required=True)
    parser.add_argument('--label')
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--log')
    parser.add_argument('--child-budget')
    parser.add_argument('--reserve-bytes', type=int, default=0)
    parser.add_argument('--maximum-bytes', type=int, default=MAXIMUM_ARTIFACT_BYTES,
                        help='Explicitly authorized new-artifact quota for this round')
    parser.add_argument('--maximum-seconds', type=float, default=MAXIMUM_SECONDS,
                        help='Authorized GPU ceiling for a new ledger; 0 declares a CPU-only round')
    arguments = list(sys.argv[1:] if argv is None else argv)
    marker = arguments.index('--') if '--' in arguments else len(arguments)
    command = arguments[marker + 1:] if marker < len(arguments) else []
    args = parser.parse_args(arguments[:marker])
    limits = RoundLimits(args.root, args.previous_budget, args.previous_sha256, maximum_bytes=args.maximum_bytes)
    if args.action == 'initialize':
        print(json.dumps(BoundGPUBudget(limits, args.maximum_seconds).initialize()))
    elif args.action == 'audit':
        print(json.dumps(limits.audit(args.reserve_bytes)))
    else:
        # A supervisor killed without Python cleanup must also terminate its
        # native descendants. The persistent lease still requires PID audit.
        from localocr.runtime_job import establish_job
        establish_job()
        supervise(limits, args.label, args.seconds, command, args.log,
                  reserve_bytes=args.reserve_bytes, child_budget=args.child_budget)


if __name__ == '__main__':
    main()
