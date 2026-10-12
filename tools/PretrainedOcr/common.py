"""Frozen research inputs, exact token identity and persistent exclusive GPU budget.

The budget measures leased wall time conservatively, including CPU work during a
GPU lease. Incomplete leases fail closed until explicitly audited; restarting a
process never grants a fresh budget. No API or production model store is used.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

MINIMUM_FREE_BYTES = 8 * 1024**3


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers or scanners may briefly deny replacement even when
        # the target is writable. Bound retries to 0.63 seconds; other failures
        # and persistent permissions remain fatal and keep the old receipt.
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                if getattr(error, 'winerror', None) not in (5, 32) or attempt == 6:
                    raise
                time.sleep(0.01 * (2 ** attempt))
    finally:
        if temporary.exists():
            temporary.unlink()


def confined(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Research input must use a confined relative path')
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError('Research input escapes dataset root')
    return result


def dataset_binding(root):
    root = Path(root).resolve()
    meta = read(root / 'dataset.json')
    frozen = read(root / 'freeze.json')
    if meta.get('schema_version') != 'localocr.pretrained.dataset.v1':
        raise ValueError('Unknown pretrained dataset schema')
    if frozen.get('schema_version') != 'localocr.pretrained.freeze.v1':
        raise ValueError('Unknown pretrained dataset freeze')
    if digest(root / 'dataset.json') != frozen.get('dataset_sha256'):
        raise ValueError('Dataset differs from freeze')
    if meta.get('synthetic_only') is not True:
        raise ValueError('Pretrained training requires synthetic-only data')
    return frozen


def read_split(root, split, allow_heldout=False):
    if split not in ('train', 'dev', 'heldout'):
        raise ValueError('Unknown research split')
    if split == 'heldout' and not allow_heldout:
        raise ValueError('Heldout cannot be consumed by training or vocabulary selection')
    root = Path(root).resolve()
    frozen = dataset_binding(root)
    meta = read(root / 'dataset.json')
    item = meta['splits'][split]
    path = confined(root, item['file'])
    actual = digest(path)
    if actual != item['sha256'] or actual != frozen['split_sha256'][split]:
        raise ValueError('Split differs from freeze')
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != item['count']:
        raise ValueError('Split row count differs')
    for row in rows:
        if row.get('split') != split or not isinstance(row.get('text'), str):
            raise ValueError('Invalid exact row or split')
        if hashlib.sha256(row['text'].encode('utf-8')).hexdigest() != row['text_sha256']:
            raise ValueError('Exact Unicode label digest differs')
        image = confined(root, row['image'])
        if digest(image) != row['sha256']:
            raise ValueError('Image differs from freeze')
        if row.get('provenance') != {'source': 'original', 'license': 'CC0-1.0'}:
            raise ValueError('Only original synthetic labels can train research models')
    return rows


def train_characters(root):
    return sorted({c for row in read_split(root, 'train') for c in row['text']}, key=ord)


def token_mapping(old_tokens, new_tokens):
    """Return old-index -> new-index without conflating blank, space or specials."""
    if len(set(old_tokens)) != len(old_tokens) or len(set(new_tokens)) != len(new_tokens):
        raise ValueError('Duplicate token identities')
    new = {token: index for index, token in enumerate(new_tokens)}
    if any(token not in new for token in old_tokens):
        raise ValueError('Pretrained token removed')
    return {index: new[token] for index, token in enumerate(old_tokens)}


def extended_characters(original, training):
    if any(not isinstance(c, str) or len(c) != 1 for c in original + training):
        raise ValueError('Dictionary entries must be one Unicode codepoint')
    if len(set(original)) != len(original):
        raise ValueError('Duplicate original characters')
    disallowed = set('\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029')
    if disallowed.intersection(original + training):
        raise ValueError('Dictionary newline codec cannot preserve this codepoint')
    existing = set(original)
    return list(original) + sorted(set(training) - existing, key=ord)


def disk_guard(path, minimum=MINIMUM_FREE_BYTES):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise RuntimeError('Research disk reserve below 8 GiB')
    return free


class Deadline:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.limit = seconds

    def remaining(self):
        return max(0.0, self.limit - (time.monotonic() - self.started))

    def expired(self):
        return self.remaining() <= 0


class GPUBudget:
    """Exclusive file lease and persisted cumulative elapsed training budget.

    A crashed active lease is not automatically cleared. The caller must audit
    its owner and account the interval before continuing; no restart-reset path
    is exposed. This protects a second worker from concurrent GPU execution.
    """
    def __init__(self, path, max_seconds=7200):
        self.path = Path(path).resolve()
        self.lock = self.path.with_name(self.path.name + '.lock')
        if max_seconds <= 0 or max_seconds > 7200:
            raise ValueError('GPU budget cannot exceed approved 7200 seconds')
        self.maximum = float(max_seconds)

    def _state(self):
        if not self.path.exists():
            return {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': self.maximum,
                    'used_seconds': 0.0, 'runs': [], 'active': None}
        state = read(self.path)
        if state.get('schema_version') != 'localocr.gpu-budget.v1' or state.get('maximum_seconds') != self.maximum:
            raise ValueError('GPU budget identity changed')
        if not isinstance(state.get('used_seconds'), (int, float)) or state['used_seconds'] < 0:
            raise ValueError('Invalid persisted GPU usage')
        if state.get('active') is not None:
            raise RuntimeError('Unfinished GPU lease requires explicit audit')
        return state

    @contextlib.contextmanager
    def run(self, label, limit_seconds=2700):
        if not label or limit_seconds <= 0:
            raise ValueError('GPU lease needs a label and positive duration')
        disk_guard(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, str(os.getpid()).encode('ascii'))
        os.close(fd)
        active = None
        try:
            state = self._state()
            available = self.maximum - state['used_seconds']
            if available <= 0:
                raise RuntimeError('Approved GPU budget exhausted')
            deadline = Deadline(min(float(limit_seconds), available))
            active = {'label': label, 'pid': os.getpid(), 'started_utc': utc(), 'limit_seconds': deadline.limit}
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
                elapsed = time.monotonic() - deadline.started
                state['used_seconds'] += elapsed
                state['runs'].append({**active, 'ended_utc': utc(), 'elapsed_seconds': elapsed, 'status': status})
                state['active'] = None
                write(self.path, state)
        finally:
            if self.lock.exists():
                self.lock.unlink()
