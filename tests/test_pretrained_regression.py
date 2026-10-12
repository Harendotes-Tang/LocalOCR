"""Legacy regression scope and actual isolated CPU tensor controls."""
import argparse
import base64
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
import tempfile

from PIL import Image
import pytest

from tools.PretrainedOcr import regression as r
from tools.PretrainedOcr import evaluate as e
from tools.PretrainedOcr import common as c

# A fixed real ONNX tensor fixture, not an OCR accuracy claim.
GRAPH = 'CAg6vQIKOQoFaW1hZ2USBG1lYW4iClJlZHVjZU1lYW4qDQoEYXhlc0ABQAKgAQcqDwoIa2VlcGRpbXMYAKABAgohCgRtZWFuCgRheGlzEghleHBhbmRlZCIJVW5zcXVlZXplChwKCGV4cGFuZGVkCgR6ZXJvEgV6ZXJvcyIDTXVsCiMKBXplcm9zCgZzY29yZXMSDXByb2JhYmlsaXRpZXMiA0FkZBINcmVhbC1jcHUtdGVzdCoNCAEQBzoBAkIEYXhpcyoOEAEiBAAAAABCBHplcm8qIggBCAEIBBABIhAAAAAAAACAPwAAAAAAAAAAQgZzY29yZXNaIQoFaW1hZ2USGAoWCAESEgoDEgFOCgIIAwoCCDAKAxIBV2IlCg1wcm9iYWJpbGl0aWVzEhQKEggBEg4KAxIBTgoDEgFUCgIIBEIECgAQDQ=='


@pytest.fixture
def artifacts():
    with tempfile.TemporaryDirectory(prefix='old-regression-contract-', dir=e.ARTIFACTS) as path:
        yield Path(path)


def legacy(root, label=' A e\u0301  ', kind='plain'):
    root.mkdir(parents=True)
    image = Image.new('RGB', (32, 16), 'white')
    image.save(root / 'image.png')
    sha = hashlib.sha256(label.encode('utf-8')).hexdigest()
    row = {'id': 'regression-one', 'kind': kind, 'label': label, 'label_sha256': sha,
           'group': 'negative:blank-one' if kind in ('blank', 'graphic') else 'text:' + sha,
           'recipe_id': 'blank-one', 'path': 'image.png', 'sha256': c.digest(root / 'image.png'),
           'width': 32, 'height': 16, 'usage': 'regression',
           'provenance': {'source': 'original', 'license': 'CC0-1.0'},
           'region': [8, 2, 10, 10]}
    c.write(root / 'regression.json', [row])
    checksum = c.digest(root / 'regression.json')
    # The other splits deliberately do not exist: the runner must not read them.
    c.write(root / 'dataset.json', {'schema_version': 'localocr.dataset.v1', 'license': 'CC0-1.0',
             'synthetic_only': True, 'splits': {'regression': {'file': 'regression.json', 'count': 1, 'sha256': checksum},
                                              'test': {'file': 'MISSING-TEST.json'}, 'acceptance': {'file': 'MISSING-ACCEPTANCE.json'}}})
    c.write(root / 'freeze.json', {'schema_version': 'localocr.freeze.v1',
             'dataset_sha256': c.digest(root / 'dataset.json'), 'split_sha256': {'regression': checksum}})
    return root


def candidate(root, identity, family, stage='finetuned'):
    root.mkdir(parents=True)
    (root / 'model.onnx').write_bytes(base64.b64decode(GRAPH))
    (root / 'checkpoint.bin').write_bytes(b'Tensor contract fixture; not pretrained OCR')
    (root / 'dictionary.txt').write_text('A\ne\n', encoding='utf-8')
    c.write(root / 'vocabulary.json', {'schema_version': 'localocr.pretrained.vocabulary.v1',
             'tokens': ['<blank>', 'A', 'e', ' '], 'blank_id': 0, 'space_id': 3})
    file = lambda name: {'file': name, 'sha256': c.digest(root / name)}
    c.write(root / 'candidate.json', {'schema_version': e.CANDIDATE_SCHEMA, 'id': identity,
             'family': family, 'stage': stage, 'model': file('model.onnx'),
             'vocabulary': dict(file('vocabulary.json'), blank_id=0, space_id=3),
             'preprocessing': {'height': 48, 'color': 'RGB',
                 'width': {'mode': 'dynamic', 'min': 16, 'max': 1024, 'multiple': 4, 'rounding': 'ceil-multiple'},
                 'padding': 'right-zero-normalized', 'normalize': {'mean': [.5] * 3, 'std': [.5] * 3}, 'resize': 'PIL-bilinear'},
             'decoder': {'type': 'ctc', 'output': 'probabilities', 'blank_id': 0, 'repeat': 'ctc'},
             'source': {'checkpoint': file('checkpoint.bin'), 'dictionary': file('dictionary.txt')},
             'promote': False, 'enabled': False})
    return root / 'candidate.json'


def choices(root):
    paths = [candidate(root / identity, identity, family, stage)
             for family in ('one', 'two')
             for identity, stage in ((family + '-before', 'official-pretrained'), (family + '-after', 'finetuned'))]
    bindings = [e.candidate_binding(path) for path in paths]
    baseline = {'id': 'production-baseline', 'version': 'tensor-fixture'}
    dataset = {'dataset_sha256': 'tensor-fixture-only'}
    report = {'schema_version': e.REPORT_SCHEMA, 'split': 'dev', 'independent_heldout': False,
              'candidate_bindings': bindings, 'baseline_binding': baseline, 'dataset_binding': dataset}
    c.write(root / 'dev.json', report)
    c.write(root / 'budget.json', {'schema_version': 'localocr.gpu-budget.v1', 'maximum_seconds': 7200,
                                  'used_seconds': 100, 'active': None})
    record = lambda name: {'file': str(root / name), 'sha256': c.digest(root / name)}
    receipt = {'schema_version': e.SELECTION_SCHEMA, 'usage': 'research-only-selected-for-evaluation',
               'frozen_utc': '2026-10-11T16:00:00Z', 'promote': False, 'all_bound_candidates_evaluated_together': True,
               'candidate_bindings': bindings, 'baseline_binding': baseline, 'dataset_binding': dataset,
               'selected_id': 'one-after', 'dev_report': record('dev.json'), 'budget_receipt': record('budget.json')}
    c.write(root / 'selection.json', receipt)
    return paths[1::2], bindings[1::2], baseline, root / 'selection.json'


def rebind(root):
    meta = c.read(root / 'dataset.json')
    meta['splits']['regression']['sha256'] = c.digest(root / 'regression.json')
    c.write(root / 'dataset.json', meta)
    frozen = c.read(root / 'freeze.json')
    frozen['dataset_sha256'] = c.digest(root / 'dataset.json')
    frozen['split_sha256']['regression'] = c.digest(root / 'regression.json')
    c.write(root / 'freeze.json', frozen)


def test_only_old_regression_read_preserves_raw_unicode_full_canvas(artifacts):
    root = legacy(artifacts / 'data')
    rows, binding = r.regression_rows(root)
    assert rows[0]['text'] == ' A e\u0301  '
    assert rows[0]['group_id'].startswith('text:')
    assert e.image_for_row(root, rows[0]).size == (32, 16)
    assert binding['scope'] == 'old-synthetic-development-regression'
    assert binding['independent_heldout'] is False and binding['other_splits_read'] is False
    assert binding['row_count'] == 1 and len(binding['records']) == 1


@pytest.mark.parametrize('field,value,match', [
    ('label', 'A é', 'label'), ('usage', 'test', 'identity'),
    ('provenance', {'source': 'private', 'license': 'CC0-1.0'}, 'provenance'),
    ('group', 'not-exact-label', 'group'), ('width', 31, 'dimensions'),
    ('path', '../escape.png', 'confined'), ('kind', 'icon', 'identity'),
])
def test_self_rebound_rows_still_validate_contract(artifacts, field, value, match):
    root = legacy(artifacts / 'data')
    rows = c.read(root / 'regression.json')
    rows[0][field] = value
    c.write(root / 'regression.json', rows)
    rebind(root)
    with pytest.raises(ValueError, match=match):
        r.regression_rows(root)


def test_image_and_split_bytes_are_frozen(artifacts):
    root = legacy(artifacts / 'data')
    Image.new('RGB', (32, 16), 'black').save(root / 'image.png')
    with pytest.raises(ValueError, match='bytes'):
        r.regression_rows(root)
    root = legacy(artifacts / 'other')
    with (root / 'regression.json').open('a', encoding='utf-8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='split differs'):
        r.regression_rows(root)


def test_negative_labels_empty_and_duplicate_id_rejected(artifacts):
    root = legacy(artifacts / 'negative', '', 'blank')
    rows, _ = r.regression_rows(root)
    assert rows[0]['text'] == ''
    data = c.read(root / 'regression.json')
    data.append(data[0])
    c.write(root / 'regression.json', data)
    meta = c.read(root / 'dataset.json')
    meta['splits']['regression']['count'] = 2
    c.write(root / 'dataset.json', meta)
    rebind(root)
    with pytest.raises(ValueError, match='identity'):
        r.regression_rows(root)


def test_frozen_two_family_choices_and_references(artifacts):
    paths, bound, baseline, selection = choices(artifacts)
    assert r.frozen_choices(selection, bound, baseline)['sha256'] == c.digest(selection)
    with pytest.raises(ValueError, match='two frozen'):
        r.frozen_choices(selection, bound[:1], baseline)
    changed = dict(bound[0], model_sha256='changed')
    with pytest.raises(ValueError, match='two frozen'):
        r.frozen_choices(selection, [changed, bound[1]], baseline)
    with pytest.raises(ValueError, match='matching frozen'):
        r.frozen_choices(selection, bound, {'id': 'production-baseline', 'version': 'different'})
    with (artifacts / 'dev.json').open('a', encoding='utf-8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='reference changed'):
        r.frozen_choices(selection, bound, baseline)


@pytest.mark.parametrize('active,used', [({'pid': 1}, 100), (None, 7201), (None, True), (None, float('inf'))])
def test_completed_original_budget_required(artifacts, active, used):
    _, bound, baseline, selection = choices(artifacts)
    budget = c.read(artifacts / 'budget.json')
    budget.update(active=active, used_seconds=used)
    # Infinity is deliberately hand-written to exercise validation of nonstandard JSON.
    (artifacts / 'budget.json').write_text(json.dumps(budget), encoding='utf-8')
    receipt = c.read(selection)
    receipt['budget_receipt']['sha256'] = c.digest(artifacts / 'budget.json')
    c.write(selection, receipt)
    with pytest.raises(ValueError, match='budget must be completed'):
        r.frozen_choices(selection, bound, baseline)


def test_source_guard_runs_before_data_reader(artifacts, monkeypatch):
    args = argparse.Namespace(output=artifacts / 'never.json', source_sha256='bad', dataset='must-not-open')
    monkeypatch.setattr(r, 'regression_rows', lambda _: pytest.fail('data read before source guard'))
    with pytest.raises(ValueError, match='source inventory changed'):
        r.worker(args)


def test_actual_candidate_cpu_worker_full_canvas_and_resources(artifacts, monkeypatch):
    dataset = legacy(artifacts / 'data')
    paths, _, baseline, selection = choices(artifacts)
    args = argparse.Namespace(dataset=dataset, output=artifacts / 'worker.json', source_sha256=r.source_binding()['sha256'],
           candidate=paths[0], baseline_models=None, selection_baseline_models='tensor-baseline-fixture',
           selected_candidate=paths, selection=selection)
    # The worker installs a process-wide audit hook which cannot be removed.
    # Keep that policy inside its real child; only baseline fixture metadata is
    # substituted. Native ONNX inference and all resource measurements run.
    code = """
import argparse,json,sys
from tools.PretrainedOcr import regression as r
from tools.PretrainedOcr import evaluate as e
baseline=json.loads(sys.argv[1])
e.baseline_binding=lambda _:baseline
r.worker(argparse.Namespace(**json.loads(sys.argv[2])))
"""
    completed = subprocess.run([sys.executable, '-c', code, json.dumps(baseline),
        json.dumps(vars(args), default=str)], cwd=e.REPOSITORY, capture_output=True,
        text=True, encoding='utf-8', timeout=30)
    assert completed.returncode == 0, completed.stderr
    # The pytest parent still owns its service/process role after child exit.
    with socket.socket():
        pass
    assert subprocess.run([sys.executable, '-c', 'pass'], timeout=10).returncode == 0
    result = c.read(args.output)
    assert result['predictions'] == [{'id': 'regression-one', 'text': 'A', 'score': 1.0}]
    resources = result['resources']
    assert resources['providers'] == ['CPUExecutionProvider']
    assert resources['intra_op_threads'] == 2 and resources['inter_op_threads'] == 1
    assert resources['warm']['count'] == 1 and resources['cold_load_and_first_crop_seconds'] > 0
    assert resources['peak_rss_bytes'] > 0
    if e.sys.platform == 'win32':
        assert resources['peak_private_bytes'] > 0 and resources['peak_commit_bytes'] > 0
    assert result['data_binding']['scope'] == r.SCOPE
    assert result['inference_binding'] == e.candidate_binding(paths[0])


def test_immutable_output_rejects_existing_before_models(artifacts):
    output = artifacts / 'existing'
    output.mkdir()
    args = argparse.Namespace(output=output)
    with pytest.raises(ValueError, match='output already exists'):
        r.evaluate(args)


def test_stdout_only_anonymous_integrity(artifacts, capsys):
    root = legacy(artifacts / 'data')
    r.main(['verify-data', '--dataset', str(root)])
    captured = capsys.readouterr().out
    result = json.loads(captured)
    assert result['row_count'] == 1 and result['scope'] == r.SCOPE
    assert 'image.png' not in captured and ' A e' not in captured
