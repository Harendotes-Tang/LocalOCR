"""Exact identity and empty-target regressions independent of GPU availability."""
import numpy as np
import pytest
from tools.PretrainedOcr.common import token_mapping
from tools.PretrainedOcr.paddle_adapter import tokens, remap_array, encode_labels, decode


def test_all_head_axes_keep_space_and_special_weights():
    old, new = ['A', '\u00e9'], ['A', '\u00e9', '\u0301', '\u200d', '\ufe0f']
    for head in ('ctc', 'nrtr'):
        before, after = tokens(old, head), tokens(new, head)
        mapping = token_mapping(before, after)
        assert before.index(' ') != after.index(' ')
        for axis in (0, 1):
            a = np.arange(len(before) * 3).reshape((len(before), 3) if axis == 0 else (3, len(before)))
            b = np.full((len(after), 3) if axis == 0 else (3, len(after)), -1)
            result = remap_array(a, b, mapping, axis)
            for source, target in mapping.items():
                assert np.array_equal(np.take(result, target, axis=axis), np.take(a, source, axis=axis))
            added = set(range(len(after))) - set(mapping.values())
            assert all((np.take(result, target, axis=axis) == -1).all() for target in added)


def test_blank_bos_eos_and_exact_unicode_labels():
    chars = ['A', '\u0301', '\u200d', '\ufe0f']
    texts = ['', ' A  ', 'A\u0301\u200d\ufe0f']
    ctc, nrtr, lengths = encode_labels(texts, chars, 4)
    assert lengths.tolist() == [0, 4, 4]
    assert not ctc[0].any()
    assert nrtr[0].tolist() == [2, 3, 0, 0, 0, 0]
    assert ctc[1].tolist() == [5, 1, 5, 5]
    assert nrtr[1].tolist() == [2, 8, 4, 8, 8, 3]
    with pytest.raises(ValueError):
        encode_labels(['B'], chars, 4)
    with pytest.raises(ValueError):
        encode_labels([' A  '], chars, 3)


def test_ctc_preserves_repeated_space_when_blank_separates():
    chars = ['A', '\u0301']
    ids = [3, 1, 1, 0, 2, 3, 0, 3]
    probabilities = np.eye(4, dtype='float32')[ids][None]
    assert decode(probabilities, chars) == [' A\u0301  ']
    with pytest.raises(ValueError):
        decode(np.zeros((1, 3, 2)), chars)


def test_remap_rejects_incomplete_or_duplicate_identity():
    with pytest.raises(ValueError):
        remap_array(np.zeros((3, 2)), np.zeros((4, 2)), {0: 0}, 0)
    with pytest.raises(ValueError):
        remap_array(np.zeros((3, 2)), np.zeros((4, 2)), {0: 0, 1: 0, 2: 2}, 0)


def test_empty_ctc_exact_blank_path_gradient_and_length_normalization():
    paddle = pytest.importorskip('paddle')
    from tools.PretrainedOcr.paddle_adapter import exact_ctc_loss
    paddle.set_device('cpu')
    for time_steps in (4, 8):
        logits = paddle.zeros([1, time_steps, 3], dtype='float32')
        logits.stop_gradient = False
        result = exact_ctc_loss(logits, paddle.zeros([1, 1], dtype='int64'),
                                paddle.zeros([1], dtype='int64'), [''], None)
        assert float(result) == pytest.approx(np.log(3), rel=1e-6)
        result.backward()
        gradient = logits.grad.numpy()
        assert np.isfinite(gradient).all()
        assert np.allclose(gradient[:, :, 0], -2 / (3 * time_steps))
        assert np.allclose(gradient[:, :, 1:], 1 / (3 * time_steps))
        assert np.abs(gradient).sum() == pytest.approx(4 / 3, rel=1e-6)


def test_empty_ctc_extreme_logits_are_exact_unclipped():
    paddle = pytest.importorskip('paddle')
    from tools.PretrainedOcr.paddle_adapter import exact_ctc_loss
    paddle.set_device('cpu')
    logits = paddle.to_tensor([[[1000., -1000., 500., 2000.]]], stop_gradient=False)
    result = exact_ctc_loss(logits, paddle.zeros([1, 1], dtype='int64'),
                            paddle.zeros([1], dtype='int64'), [''], None)
    assert float(result) == pytest.approx(1000)
    result.backward()
    assert np.allclose(logits.grad.numpy(), [[[-1, 0, 0, 1]]])


def test_exception_unwinds_real_model_optimizer_and_chained_tensor_before_release():
    import gc
    import weakref
    paddle = pytest.importorskip('paddle')
    from tools.PretrainedOcr.paddle_adapter import _owned_execution, _own, _move_owned_to_cpu
    paddle.set_device('cpu')
    events, references = [], []
    def fail_with_tensors(model):
        prediction = model(paddle.ones([4, 2]))
        references.append(weakref.ref(prediction))
        prediction.sum().backward()
        raise OSError('injected checkpoint/disk failure after real backward')
    def body(owned):
        model = _own(owned, 'models', paddle.nn.Linear(2, 2))
        optimizer = _own(owned, 'optimizers', paddle.optimizer.Adam(parameters=model.parameters()))
        references.extend([weakref.ref(model), weakref.ref(optimizer)])
        # Populate real optimizer accumulators before the nested failing frame.
        model(paddle.ones([4, 2])).sum().backward()
        optimizer.step()
        optimizer.clear_grad()
        try:
            fail_with_tensors(model)
        except OSError as error:
            raise RuntimeError('injected training wrapper') from error
    def failure(error):
        assert isinstance(error.__cause__, OSError)
        events.append('failure-flushed-inside-lease')
    def dispose(owned):
        assert events == ['failure-flushed-inside-lease']
        _move_owned_to_cpu(owned)
        gc.collect()
        assert all(reference() is None for reference in references)
        events.append('released-before-lease-exit')
    with pytest.raises(RuntimeError, match='injected training wrapper'):
        _owned_execution(body, dispose, failure)
    assert events[-1] == 'released-before-lease-exit'


def test_source_inventory_rejects_unmanifested_missing_or_changed_file(tmp_path):
    from tools.PretrainedOcr.common import digest
    from tools.PretrainedOcr.paddle_adapter import _verify_source_inventory
    source = tmp_path / 'official'
    source.mkdir()
    valid = source / 'module.py'
    valid.write_text('official = True', encoding='utf-8')
    expected = {'module.py': digest(valid)}
    _verify_source_inventory(source, expected)
    injected = source / 'unmanifested.py'
    injected.write_text('injected = True', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory differs'):
        _verify_source_inventory(source, expected)
    injected.unlink()
    valid.write_text('official = False', encoding='utf-8')
    with pytest.raises(ValueError, match='bytes differ'):
        _verify_source_inventory(source, expected)
    valid.unlink()
    with pytest.raises(ValueError, match='inventory differs'):
        _verify_source_inventory(source, expected)


def test_cached_ppocr_module_rejects_foreign_source(monkeypatch, tmp_path):
    import importlib.util
    import sys
    import types
    from tools.PretrainedOcr.paddle_adapter import _verify_cached_modules
    foreign = tmp_path / 'foreign.py'
    foreign.write_text('foreign = True', encoding='utf-8')
    module = types.ModuleType('ppocr.foreign_negative_control')
    module.__file__ = str(foreign)
    module.__spec__ = importlib.util.spec_from_file_location(module.__name__, foreign)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match='outside fixed official source'):
        _verify_cached_modules()


def test_step_zero_cannot_claim_finetuned_checkpoint(tmp_path, monkeypatch):
    from tools.PretrainedOcr.common import write
    from tools.PretrainedOcr.paddle_adapter import _training_checkpoint
    import tools.PretrainedOcr.paddle_adapter as adapter
    monkeypatch.setattr(adapter, 'ROUND', tmp_path)
    owned = tmp_path / 'paddle'
    owned.mkdir()
    prepared, run = owned / 'prepared', owned / 'run'
    prepared.mkdir()
    run.mkdir()
    checkpoint = run / 'best.pdparams'
    checkpoint.write_bytes(b'unchanged-initialization')
    write(prepared / 'initialization.json', {'dataset': {}, 'checkpoint': {}})
    write(run / 'training.json', {'schema_version': 'localocr.paddle.training.v1', 'selected_weights_changed': False})
    with pytest.raises(ValueError, match='selected changed'):
        _training_checkpoint(checkpoint, prepared)


def test_cleanup_failure_with_closed_stdout_still_hard_exits_before_lease_exit(monkeypatch, tmp_path):
    import contextlib
    from types import SimpleNamespace
    import tools.PretrainedOcr.paddle_adapter as adapter
    events = []
    class HardExit(BaseException):
        pass
    class Budget:
        def __init__(self, path):
            pass
        @contextlib.contextmanager
        def run(self, label, seconds):
            events.append('lease-enter')
            try:
                yield SimpleNamespace(remaining=lambda: 50)
            finally:
                events.append('lease-exit')
    def synchronize():
        raise OSError('injected cleanup synchronization failure')
    def closed_print(*args, **kwargs):
        events.append('stdout-broken')
        raise BrokenPipeError('injected closed stdout')
    def hard_exit(code):
        assert code == 70
        events.append('hard-exit')
        raise HardExit()
    monkeypatch.setattr(adapter, 'GPUBudget', Budget)
    monkeypatch.setattr(adapter, '_paddle', lambda device: SimpleNamespace(device=SimpleNamespace(
        synchronize=synchronize, cuda=SimpleNamespace(memory_allocated=lambda: 0))))
    monkeypatch.setattr(adapter, 'print', closed_print, raising=False)
    monkeypatch.setattr(adapter.os, '_exit', hard_exit)
    with pytest.raises(HardExit):
        adapter._gpu_execution('negative-control', 60, tmp_path / 'probe.json', lambda deadline, owned: {})
    assert events == ['lease-enter', 'stdout-broken', 'hard-exit', 'lease-exit']


def test_fixed_loader_bypasses_unverified_bytecode_cache(monkeypatch, tmp_path):
    import importlib.machinery
    import marshal
    import types
    from tools.PretrainedOcr.common import digest
    from tools.PretrainedOcr.paddle_adapter import _FixedSourceLoader
    source = tmp_path / 'official.py'
    source.write_text('OFFICIAL_SOURCE = True\n', encoding='utf-8')
    malicious_code = compile('UNVERIFIED_BYTECODE_MARKER = True', str(source), 'exec')
    reads = []
    def unverified_cache(loader, path):
        reads.append(path)
        return marshal.dumps(malicious_code)
    monkeypatch.setattr(importlib.machinery.SourceFileLoader, 'get_data', unverified_cache)
    loader = _FixedSourceLoader('ppocr.cache_negative_control', str(source), digest(source))
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    assert module.OFFICIAL_SOURCE is True
    assert not hasattr(module, 'UNVERIFIED_BYTECODE_MARKER')
    assert reads == []


def test_direct_official_loss_imports_also_bypass_bytecode_cache(monkeypatch):
    import importlib.machinery
    paddle = pytest.importorskip('paddle')
    from tools.PretrainedOcr.paddle_adapter import source_audit, _official_loss
    paddle.set_device('cpu')
    source_audit()
    original = importlib.machinery.SourceFileLoader.get_data
    reads = []
    def cache_negative_control(loader, path):
        if str(path).endswith('.pyc') and ('rec_ctc_loss' in str(path) or 'rec_nrtr_loss' in str(path)):
            reads.append(str(path))
            raise AssertionError('Unverified loss bytecode cache was read')
        return original(loader, path)
    monkeypatch.setattr(importlib.machinery.SourceFileLoader, 'get_data', cache_negative_control)
    losses = _official_loss()
    assert [type(loss).__name__ for loss in losses] == ['CTCLoss', 'NRTRLoss']
    assert reads == []


def test_weighted_empty_ctc_keeps_time_normalization_and_finite_gradient():
    paddle = pytest.importorskip('paddle')
    from tools.PretrainedOcr.paddle_adapter import exact_ctc_loss
    paddle.set_device('cpu')
    for time_steps in (4, 8):
        logits = paddle.zeros([1, time_steps, 3], dtype='float32')
        logits.stop_gradient = False
        result = exact_ctc_loss(logits, paddle.zeros([1, 1], dtype='int64'),
                                paddle.zeros([1], dtype='int64'), [''], None, 4)
        assert float(result) == pytest.approx(4 * np.log(3), rel=1e-6)
        result.backward()
        gradient = logits.grad.numpy()
        assert np.isfinite(gradient).all()
        assert np.allclose(gradient[:, :, 0], -8 / (3 * time_steps))
        assert np.abs(gradient).sum() == pytest.approx(16 / 3, rel=1e-6)
    for weight in (0, -1, float('nan'), float('inf'), 65):
        with pytest.raises(ValueError, match='weight'):
            exact_ctc_loss(logits, None, None, [''], None, weight)


def test_frozen_real_tiny_backbone_and_bn_do_not_change_after_head_updates():
    paddle = pytest.importorskip('paddle')
    import tools.PretrainedOcr.paddle_adapter as adapter
    paddle.set_device('cpu')
    paddle.seed(20261011)
    adapter.source_audit()
    model, chars, info = adapter._load_prepared(adapter.ROUND / 'paddle/prepared-v2-seeded', device='cpu')
    policy = adapter.configure_finetune(model, True)
    original = {name: tensor.numpy().copy() for name, tensor in model.backbone.state_dict().items()}
    head = model.head.ctc_head.fc2.weight.numpy().copy()
    optimizer = paddle.optimizer.Adam(learning_rate=2e-5,
                                      parameters=[p for p in model.parameters() if not p.stop_gradient])
    losses = adapter._official_loss()
    images = np.random.default_rng(20261011).uniform(-1, 1, [2, 3, 48, 512]).astype('float32')
    for texts in ([' A  ', ' A  '], ['', '']):
        model.eval()
        adapter.set_training_mode(model, True)
        assert model.training and not model.backbone.training
        outcome = adapter.train_step(model, optimizer, losses, images, texts, chars,
                                     info['maximum_label_length'], empty_ctc_weight=4)
        assert outcome['gradient_parameters'] == policy['trainable_parameters'] > 0
    assert all(np.array_equal(original[name], tensor.numpy()) for name, tensor in model.backbone.state_dict().items())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert not np.array_equal(head, model.head.ctc_head.fc2.weight.numpy())


def test_warm_start_requires_external_parent_anchor_before_reading_data(tmp_path):
    from tools.PretrainedOcr.paddle_adapter import train
    with pytest.raises(ValueError, match='externally frozen'):
        train(tmp_path / 'no-prepared', tmp_path / 'no-data', tmp_path / 'new-output', warm_start='untrusted')


def test_training_receipt_self_resigning_cannot_replace_external_anchor(tmp_path, monkeypatch):
    import hashlib
    import tools.PretrainedOcr.paddle_adapter as adapter
    from tools.PretrainedOcr.common import write
    monkeypatch.setattr(adapter, 'ROUND', tmp_path)
    run, prepared = tmp_path / 'paddle/run', tmp_path / 'paddle/prepared'
    run.mkdir(parents=True)
    prepared.mkdir()
    checkpoint = run / 'best.pdparams'
    checkpoint.write_bytes(b'local-operator-selected-checkpoint')
    write(prepared / 'initialization.json', {})
    write(run / 'training.json', {'selected_weights_changed': True})
    anchor = hashlib.sha256((run / 'training.json').read_bytes()).hexdigest()
    checkpoint.write_bytes(b'replaced-checkpoint')
    write(run / 'training.json', {'selected_weights_changed': True, 'best_checkpoint':
                                 {'file': 'best.pdparams', 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}})
    with pytest.raises(ValueError, match='Externally anchored'):
        adapter._training_checkpoint(checkpoint, prepared, anchor)


def test_parent_training_cycle_rejected_before_loading_any_weights(tmp_path, monkeypatch):
    import tools.PretrainedOcr.paddle_adapter as adapter
    monkeypatch.setattr(adapter, 'ROUND', tmp_path)
    run, prepared = tmp_path / 'paddle/run', tmp_path / 'paddle/prepared'
    run.mkdir(parents=True)
    prepared.mkdir()
    checkpoint = run / 'best.pdparams'
    checkpoint.write_bytes(b'not-deserialized')
    (run / 'training.json').write_text('{}')
    (prepared / 'initialization.json').write_text('{}')
    with pytest.raises(ValueError, match='cyclic'):
        adapter._training_checkpoint(checkpoint, prepared, _seen={run / 'training.json'})


@pytest.mark.parametrize('target_location', ['inside', 'outside'])
def test_windows_junction_evidence_rejected_even_with_matching_bytes(tmp_path, target_location):
    import os
    import subprocess
    import tools.PretrainedOcr.paddle_adapter as adapter
    from tools.PretrainedOcr.common import digest
    if os.name != 'nt':
        pytest.skip('Windows junction control')
    owned = tmp_path / 'owned'
    owned.mkdir()
    target = (owned if target_location == 'inside' else tmp_path) / 'real'
    target.mkdir()
    artifact = target / 'weights.bin'
    artifact.write_bytes(b'non-model-control')
    link = owned / 'junction'
    command = "New-Item -ItemType Junction -Path '" + str(link).replace("'", "''") + "' -Target '" + str(target).replace("'", "''") + "' | Out-Null"
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True, capture_output=True)
    try:
        with pytest.raises(ValueError, match='reparse'):
            adapter._closure_file({'file': 'junction/weights.bin', 'sha256': digest(artifact)}, owned, owned)
    finally:
        assert link.absolute().is_relative_to(tmp_path.absolute())
        assert link.lstat().st_file_attributes & 0x400
        os.rmdir(link)  # Removes this verified test junction, retaining its target.
        assert artifact.read_bytes() == b'non-model-control'


@pytest.mark.parametrize('filename', ['vocabulary.json', 'dictionary.txt'])
def test_training_vocabulary_and_dictionary_reparse_policy_precedes_deserialization(tmp_path, monkeypatch, filename):
    from pathlib import Path
    from types import SimpleNamespace
    import tools.PretrainedOcr.paddle_adapter as adapter
    from tools.PretrainedOcr.common import digest, write
    monkeypatch.setattr(adapter, 'ROUND', tmp_path)
    run, prepared = tmp_path / 'paddle/run', tmp_path / 'paddle/prepared'
    run.mkdir(parents=True)
    prepared.mkdir()
    checkpoint, initialized = run / 'best.pdparams', prepared / 'initialized.pdparams'
    checkpoint.write_bytes(b'never-deserialized-selected-weights')
    initialized.write_bytes(b'never-deserialized-initialized-weights')
    vocabulary, dictionary = prepared / 'vocabulary.json', prepared / 'dictionary.txt'
    vocabulary.write_bytes(b'{"tokens": ["<blank>", "A"]}')
    dictionary.write_bytes(b'A\n')
    init_ref = {'file': 'initialized.pdparams', 'sha256': digest(initialized)}
    binding = {'synthetic_control': True}
    write(prepared / 'initialization.json', {'source_commit': adapter.COMMIT,
          'source_checkpoint_sha256': adapter.CHECKPOINT_SHA, 'checkpoint': init_ref,
          'dataset': binding, 'dataset_root': 'not-read', 'vocabulary_sha256': digest(vocabulary),
          'dictionary_sha256': digest(dictionary)})
    write(run / 'training.json', {'schema_version': 'localocr.paddle.training.v1',
          'selected_weights_changed': True, 'actual_weights_changed': True, 'steps': 1,
          'best_checkpoint': {'file': 'best.pdparams', 'sha256': digest(checkpoint)},
          'preparation': {'file': str(prepared / 'initialization.json'), 'sha256': digest(prepared / 'initialization.json')},
          'initialized_checkpoint': init_ref, 'dataset': binding})
    monkeypatch.setattr(adapter, 'dataset_binding', lambda _: binding)
    original = Path.lstat
    poisoned = prepared / filename
    def reparse_metadata(path, *args, **kwargs):
        properties = original(path, *args, **kwargs)
        if path == poisoned:
            return SimpleNamespace(st_mode=properties.st_mode, st_file_attributes=0x400)
        return properties
    monkeypatch.setattr(Path, 'lstat', reparse_metadata)
    with pytest.raises(ValueError, match='reparse'):
        adapter._training_checkpoint(checkpoint, prepared, digest(run / 'training.json'))


def test_windows_junction_ancestor_of_owned_root_rejected(tmp_path):
    import os
    import subprocess
    import tools.PretrainedOcr.paddle_adapter as adapter
    from tools.PretrainedOcr.common import digest
    if os.name != 'nt':
        pytest.skip('Windows junction control')
    real = tmp_path / 'real'
    root = real / 'owned'
    root.mkdir(parents=True)
    artifact = root / 'weights.bin'
    artifact.write_bytes(b'non-model-ancestor-control')
    link = tmp_path / 'junction'
    command = "New-Item -ItemType Junction -Path '" + str(link).replace("'", "''") + "' -Target '" + str(real).replace("'", "''") + "' | Out-Null"
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True, capture_output=True)
    try:
        owned = link / 'owned'
        with pytest.raises(ValueError, match='root contains a reparse'):
            adapter._closure_file({'file': 'weights.bin', 'sha256': digest(artifact)}, owned, owned)
    finally:
        assert link.absolute().is_relative_to(tmp_path.absolute())
        assert link.lstat().st_file_attributes & 0x400
        os.rmdir(link)  # Removes the verified test junction, retaining its target.
        assert artifact.read_bytes() == b'non-model-ancestor-control'
