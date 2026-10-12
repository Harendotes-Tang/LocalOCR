"""Meaningful exact-token, frozen-input and persistent-budget failure cases."""
import hashlib
import pytest
from tools.PretrainedOcr import common as c


def test_token_identity_space_moves_blank_preserved():
    assert c.token_mapping(['<blank>', 'A', ' '], ['<blank>', 'A', '字', ' ']) == {0: 0, 1: 1, 2: 3}
    with pytest.raises(ValueError, match='removed'):
        c.token_mapping(['<blank>', 'A'], ['<blank>', '字'])
    with pytest.raises(ValueError, match='Duplicate'):
        c.token_mapping(['A', 'A'], ['A'])


def test_expansion_preserves_exact_codepoints():
    assert c.extended_characters(['A', 'é'], ['a', ' ', 'e', '\u0301']) == ['A', 'é', ' ', 'a', 'e', '\u0301']
    with pytest.raises(ValueError, match='newline'):
        c.extended_characters(['A'], ['\u2028'])
    with pytest.raises(ValueError, match='one Unicode'):
        c.extended_characters(['A'], ['👩‍💻'])


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'image.png').write_bytes(b'test fixture bytes')
    row = {'split': 'train', 'text': ' A e\u0301  ', 'image': 'image.png',
           'text_sha256': hashlib.sha256(' A e\u0301  '.encode()).hexdigest(),
           'sha256': c.digest(root / 'image.png'),
           'provenance': {'source': 'original', 'license': 'CC0-1.0'}}
    c.write(root / 'train.json', [row])
    sha = c.digest(root / 'train.json')
    c.write(root / 'dataset.json', {'schema_version': 'localocr.pretrained.dataset.v1',
                                 'synthetic_only': True, 'splits': {'train': {'file': 'train.json', 'count': 1, 'sha256': sha}}})
    c.write(root / 'freeze.json', {'schema_version': 'localocr.pretrained.freeze.v1',
                                 'dataset_sha256': c.digest(root / 'dataset.json'), 'split_sha256': {'train': sha}})
    return root


def test_train_reader_does_not_open_heldout(dataset):
    assert c.read_split(dataset, 'train')[0]['text'] == ' A e\u0301  '
    assert c.train_characters(dataset) == [' ', 'A', 'e', '\u0301']
    with pytest.raises(ValueError, match='Heldout'):
        c.read_split(dataset, 'heldout')
    # No heldout file exists; the training-only read still succeeded.


def test_byte_and_label_mutation_fail_closed(dataset):
    (dataset / 'image.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Image differs'):
        c.read_split(dataset, 'train')
    c.write(dataset / 'dataset.json', {'changed': True})
    with pytest.raises(ValueError):
        c.read_split(dataset, 'train')


def test_confined_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        c.confined(tmp_path, '../other')
    with pytest.raises(ValueError):
        c.confined(tmp_path, tmp_path / 'other')


def test_budget_persists_failures_and_rejects_parallel(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'disk_guard', lambda path: 20 * 1024**3)
    clock = iter([1., 2., 4., 5., 5., 6.])
    monkeypatch.setattr(c.time, 'monotonic', lambda: next(clock))
    path = tmp_path / 'gpu.json'
    with pytest.raises(RuntimeError, match='probe failed'):
        with c.GPUBudget(path, 10).run('probe', 5):
            with pytest.raises(FileExistsError):
                with c.GPUBudget(path, 10).run('parallel', 5):
                    pass
            raise RuntimeError('probe failed')
    state = c.read(path)
    assert state['used_seconds'] == 1 and state['runs'][0]['status'] == 'failed'
    with c.GPUBudget(path, 10).run('second', 5) as deadline:
        assert deadline.remaining() == 4
    assert c.read(path)['used_seconds'] == 3
    assert not path.with_name('gpu.json.lock').exists()


def test_crashed_budget_is_not_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'disk_guard', lambda path: 20 * 1024**3)
    path = tmp_path / 'gpu.json'
    c.write(path, {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 7200.,
                   'used_seconds': 20., 'runs': [], 'active': {'pid': 1}})
    with pytest.raises(RuntimeError, match='explicit audit'):
        with c.GPUBudget(path).run('restart'):
            pass
    assert c.read(path)['used_seconds'] == 20


def test_budget_approval_cannot_increase(tmp_path):
    with pytest.raises(ValueError, match='approved'):
        c.GPUBudget(tmp_path / 'gpu.json', 7201)
    c.write(tmp_path / 'gpu.json', {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 7200.,
                                 'used_seconds': 7200., 'runs': [], 'active': None})
    with pytest.raises(RuntimeError, match='exhausted'):
        with c.GPUBudget(tmp_path / 'gpu.json').run('extra'):
            pass


@pytest.mark.parametrize('separator', ['\n', '\r', '\v', '\f', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029'])
def test_all_splitlines_separators_rejected(separator):
    assert ('A' + separator + 'B').splitlines() == ['A', 'B']
    with pytest.raises(ValueError, match='newline codec'):
        c.extended_characters(['A'], [separator])
    with pytest.raises(ValueError, match='newline codec'):
        c.extended_characters([separator], ['A'])


def test_atomic_write_nonsharing_error_is_not_retried(tmp_path, monkeypatch):
    target = tmp_path / 'receipt.json'
    target.write_text('{"old": true}', encoding='utf-8')
    attempts = []
    def refuse(*args):
        attempts.append(args)
        raise OSError('disk write failure')
    monkeypatch.setattr(c.os, 'replace', refuse)
    with pytest.raises(OSError, match='disk write failure'):
        c.write(target, {'new': True})
    assert len(attempts) == 1
    assert c.read(target) == {'old': True}
    assert not list(tmp_path.glob('*.tmp-*'))


def test_atomic_write_persistent_windows_denial_is_bounded(tmp_path, monkeypatch):
    target = tmp_path / 'receipt.json'
    c.write(target, {'old': True})
    sleeps = []
    attempts = []
    def refuse(*args):
        attempts.append(args)
        error = OSError('persistent Windows access denial')
        error.winerror = 5
        raise error
    monkeypatch.setattr(c.os, 'replace', refuse)
    monkeypatch.setattr(c.time, 'sleep', sleeps.append)
    with pytest.raises(OSError, match='persistent Windows'):
        c.write(target, {'new': True})
    assert len(attempts) == 7
    assert sum(sleeps) == pytest.approx(0.63)
    assert c.read(target) == {'old': True}
    assert not list(tmp_path.glob('*.tmp-*'))


@pytest.mark.skipif(c.os.name != 'nt', reason='real Windows sharing semantics')
def test_atomic_write_survives_real_windows_reader(tmp_path):
    import ctypes
    import threading
    target = tmp_path / 'receipt.json'
    c.write(target, {'old': True})
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    try:
        with pytest.raises(OSError) as denied:
            c.os.replace(target, tmp_path / 'would-rename.json')
        assert denied.value.winerror in (5, 32)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    timer = threading.Timer(0.08, kernel.CloseHandle, args=(handle,))
    timer.start()
    try:
        c.write(target, {'new': True})
    finally:
        timer.join()
    assert c.read(target) == {'new': True}
    assert not list(tmp_path.glob('*.tmp-*'))
