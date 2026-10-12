"""Raw Unicode specialist selection and invalid development evidence controls."""
import copy
import json
import math
from pathlib import Path
import pytest
from tools.PretrainedOcr import quality_selection as q


def case():
    texts = [('face', '[^_^]'), ('face', '(o  o)'), ('plain', ' A  '),
             ('punctuation', '[]：~'), ('blank', ''), ('graphic', ''), ('emoji', '👩\u200d💻')]
    rows = [dict(id=str(i), image=f'{i}.png', kind=k, text=t) for i, (k, t) in enumerate(texts)]
    predictions = [dict(id=r['id'], text=r['text']) for r in rows]
    return rows, predictions


def test_raw_unicode_and_family_metrics_do_not_normalize():
    rows, predictions = case()
    predictions[2]['text'] = 'A'
    predictions[3]['text'] = '[]:~'
    predictions[6]['text'] = '👩💻'
    s = q.summarize(rows, predictions)
    assert s['by_kind']['plain']['exact'] == 0
    assert s['by_kind']['plain']['character_errors'] == 3
    assert s['by_kind']['punctuation']['character_errors'] == 1
    assert s['by_kind']['emoji']['character_errors'] == 1
    assert s['by_character_family']['ASCIIspace']['character_errors'] == 3
    assert s['by_character_family']['combining_joiner']['character_errors'] == 1
    assert s['normalization'] == 'none; raw Unicode and ASCII spaces'
    q.validate_summary(json.loads(json.dumps(s, allow_nan=False)))


def test_overall_exact_gain_cannot_win_when_face_degrades():
    rows, predictions = case()
    best = copy.deepcopy(predictions)
    best[2]['text'] = 'bad'; best[3]['text'] = 'bad'; best[6]['text'] = 'bad'
    candidate = copy.deepcopy(predictions); candidate[0]['text'] = '[o_o]'
    a, b = q.summarize(rows, best), q.summarize(rows, candidate)
    assert b['overall']['exact'] > a['overall']['exact']
    assert not q.face_better(b, a)


def test_any_face_metric_regression_cannot_win():
    rows, predictions = case()
    best = copy.deepcopy(predictions); best[0]['text'] = '[x_x]'
    candidate = copy.deepcopy(predictions); candidate[1]['text'] = '[[[[[[[['
    assert not q.face_better(q.summarize(rows, candidate), q.summarize(rows, best))


def test_face_improvement_selects_even_when_total_exact_drops():
    rows, predictions = case()
    best = copy.deepcopy(predictions); best[0]['text'] = '[x_x]'
    candidate = copy.deepcopy(predictions)
    candidate[2]['text'] = 'bad'; candidate[3]['text'] = 'bad'; candidate[6]['text'] = 'bad'
    a, b = q.summarize(rows, best), q.summarize(rows, candidate)
    assert b['overall']['exact'] < a['overall']['exact']
    assert q.face_better(b, a)
    assert not q.face_better(a, a)


def test_negative_regression_blocks_checkpoint_but_is_not_production_gate():
    rows, predictions = case()
    best = copy.deepcopy(predictions); best[0]['text'] = '[x_x]'
    candidate = copy.deepcopy(predictions); candidate[5]['text'] = 'O'
    assert not q.face_better(q.summarize(rows, candidate), q.summarize(rows, best))


def test_reference_protection_is_measured_without_silent_repair():
    rows, predictions = case(); candidate = copy.deepcopy(predictions)
    candidate[2]['text'] = 'A'; candidate[5]['text'] = 'O'
    s = q.summarize(rows, candidate, reference=predictions)
    assert s['by_kind']['plain']['reference_correct_to_wrong'] == 1
    assert s['by_kind']['graphic']['new_negative_nonempty'] == 1
    assert s['reference_available'] is True
    assert q.summarize(rows, candidate)['reference_available'] is False


@pytest.mark.parametrize('change', ['empty', 'missing_face', 'missing_plain', 'missing_punctuation',
                                   'missing_blank', 'missing_graphic', 'duplicate_id', 'wrong_id',
                                   'wrong_image', 'wrong_count', 'surrogate', 'nonempty_negative'])
def test_invalid_rows_predictions_fail_closed(change):
    rows, predictions = case()
    if change == 'empty': rows, predictions = [], []
    elif change.startswith('missing_'):
        kind = change[8:]; pairs = [(r, p) for r, p in zip(rows, predictions) if r['kind'] != kind]
        rows, predictions = [x[0] for x in pairs], [x[1] for x in pairs]
    elif change == 'duplicate_id': rows[1]['id'] = rows[0]['id']
    elif change == 'wrong_id': predictions[0]['id'] = 'foreign'
    elif change == 'wrong_image': predictions[0]['image'] = 'foreign.png'
    elif change == 'wrong_count': predictions.pop()
    elif change == 'surrogate': predictions[0]['text'] = '\ud800'
    elif change == 'nonempty_negative': rows[5]['text'] = 'O'
    with pytest.raises(ValueError): q.summarize(rows, predictions)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, None, '0'])
def test_nonfinite_and_invalid_metric_rejected(value):
    rows, predictions = case(); s = q.summarize(rows, predictions)
    s['by_kind']['face']['cer'] = value
    with pytest.raises(ValueError): q.validate_summary(s)
    with pytest.raises(ValueError): q.face_better(s, q.summarize(rows, predictions))


def test_different_data_denominators_and_forged_aggregates_rejected():
    rows, predictions = case(); a = q.summarize(rows, predictions)
    b = copy.deepcopy(a); b['overall']['exact'] -= 1
    with pytest.raises(ValueError): q.face_better(b, a)
    b = copy.deepcopy(a); del b['by_kind']['graphic']
    with pytest.raises(ValueError): q.face_better(b, a)
    rows[0]['text'] += ' '; predictions[0]['text'] += ' '
    with pytest.raises(ValueError): q.face_better(q.summarize(rows, predictions), a)


def test_original_class_support_uses_train_only_and_all_original_identities():
    rows, _ = case(); v = ['<blank>', '[', ']', '(', ')', '{', '}', 'A', ' ', '~', '：']
    c = q.original_class_coverage(rows, v)
    assert c['original_codepoints'] == 10
    assert c['supported_original_codepoints'] == 8
    assert c['families']['PS']['original_codepoints'] == 8
    assert c['basic_symbols']['{']['train_occurrences'] == 0
    assert c['basic_symbols']['[']['train_occurrences'] == 2
    assert c['source'] == 'provided training rows only; never development or heldout'
    with pytest.raises(ValueError): q.original_class_coverage(rows, v + ['['])


def test_view_agreement_is_raw_and_requires_distinct_pixel_identities():
    rows, predictions = case(); first = predictions[:2]; second = copy.deepcopy(first)
    second[1]['text'] = '(o o)'
    s = q.two_view_statistics(rows[:2], first, second, ['a'*64, 'b'*64], ['c'*64, 'd'*64])
    assert s == {'pairs': 2, 'raw_agreement': 1, 'both_exact': 1, 'agreement_and_exact': 1}
    with pytest.raises(ValueError): q.two_view_statistics(rows[:2], first, second, ['a'*64]*2, ['a'*64]*2)


@pytest.mark.parametrize('module_name', ['paddle_adapter', 'openocr_adapter'])
def test_adapter_checkpoint_policies_keep_legacy_and_reject_face_regression(module_name):
    from importlib import import_module
    adapter = import_module('tools.PretrainedOcr.' + module_name)
    rows, predictions = case(); best = copy.deepcopy(predictions)
    best[2]['text'] = 'bad'; best[3]['text'] = 'bad'; best[6]['text'] = 'bad'
    candidate = copy.deepcopy(predictions); candidate[0]['text'] = '[x_x]'
    a, b = q.summarize(rows, best), q.summarize(rows, candidate)
    metrics = lambda x: {'quality': x, 'exact_correct': x['overall']['exact'],
                         'edit_distance': x['overall']['character_errors'],
                         'exact_accuracy': x['overall']['exact_accuracy']}
    assert adapter.development_better(metrics(b), metrics(a), 'legacy')
    assert not adapter.development_better(metrics(b), metrics(a), 'face-pareto')
    assert adapter.development_better(metrics(a), None, 'face-pareto')
    with pytest.raises(ValueError): adapter.development_better(metrics(a), metrics(a), 'unknown')
    with pytest.raises(ValueError): adapter.development_better({}, metrics(a), 'face-pareto')


def test_tiny_cli_explicit_root_budget_frequency_and_legacy_defaults(tmp_path, monkeypatch):
    from tools.PretrainedOcr import paddle_adapter as adapter
    monkeypatch.setattr(adapter, 'REPO', tmp_path)
    monkeypatch.setattr(adapter, 'ROUND', adapter.ROUND)
    monkeypatch.setattr(adapter, 'BUDGET', adapter.BUDGET)
    artifact_root = tmp_path / 'artifacts'; artifact_root.mkdir()
    root = artifact_root / 'new-round' / 'models'
    calls = []; monkeypatch.setattr(adapter, 'train', lambda **kwargs: calls.append(kwargs))
    inputs = adapter.INPUT; source = adapter.SOURCE
    adapter.main(['--research-root', str(root), 'train', '--prepared', 'prep', '--dataset', 'data',
                  '--output', 'run', '--budget', 'child-budget.json', '--selection-policy', 'face-pareto',
                  '--eval-every-seconds', '30', '--seconds', '600'])
    assert adapter.ROUND == root and adapter.BUDGET == root / 'gpu-budget.json'
    assert adapter.INPUT == inputs and adapter.SOURCE == source
    assert calls[-1]['budget'] == 'child-budget.json'
    assert calls[-1]['selection_policy'] == 'face-pareto' and calls[-1]['eval_every_seconds'] == 30
    adapter.main(['train', '--prepared', 'prep', '--dataset', 'data', '--output', 'run'])
    assert calls[-1]['selection_policy'] == 'legacy' and calls[-1]['budget'] is None
    assert calls[-1]['eval_every_seconds'] == 180
    for invalid in (artifact_root, tmp_path / 'foreign', artifact_root / '..' / 'foreign'):
        with pytest.raises(ValueError): adapter.set_research_root(invalid)


def test_openocr_cli_explicit_face_policy_and_legacy_default(monkeypatch):
    from tools.PretrainedOcr import openocr_adapter as adapter
    calls = []; monkeypatch.setattr(adapter, 'train', lambda args, probe=False: calls.append((args, probe)))
    adapter.main(['train', '--prepared', 'prep', '--budget', 'child-budget.json',
                  '--selection-policy', 'face-pareto', '--eval-every', '250', '--seconds', '600'])
    assert calls[-1][0].selection_policy == 'face-pareto' and calls[-1][0].eval_every == 250
    assert calls[-1][0].budget == 'child-budget.json' and not calls[-1][1]
    adapter.main(['probe', '--prepared', 'prep', '--budget', 'child-budget.json'])
    assert calls[-1][0].selection_policy == 'legacy' and calls[-1][1]


def test_tiny_empty_invalid_frequency_fails_before_model_or_gpu_loading(tmp_path):
    from tools.PretrainedOcr import paddle_adapter as adapter
    for invalid in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='frequency'):
            adapter.train('not-read', 'not-read', tmp_path / 'new', eval_every_seconds=invalid)
    with pytest.raises(ValueError, match='selection policy'):
        adapter.train('not-read', 'not-read', tmp_path / 'new', selection_policy='unknown')


def test_actual_torch_cpu_checkpoint_save_reload_is_exact(tmp_path):
    torch = pytest.importorskip('torch')
    from tools.PretrainedOcr import openocr_adapter as adapter
    initialized = torch.cuda.is_initialized()
    model = torch.nn.Linear(3, 4).cpu()
    path = tmp_path / 'saved.pth'
    adapter.save_checkpoint(path, model, 'a'*64, 7)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    assert restored['steps'] == 7 and restored['vocabulary_sha256'] == 'a'*64
    assert all(torch.equal(value, restored['state_dict'][key]) for key, value in model.state_dict().items())
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize('tamper', ['none', 'missing_quality', 'source_sha', 'initial_checkpoint',
                                    'initial_inventory', 'history_quality', 'unknown_policy', 'policy_conflict'])
def test_interrupted_openocr_specialist_is_frozen_and_cannot_bypass_quality(tmp_path, tamper):
    torch = pytest.importorskip('torch')
    from tools.PretrainedOcr import openocr_adapter as adapter
    from tools.PretrainedOcr import common
    rows, best_predictions = case(); initial_predictions = copy.deepcopy(best_predictions)
    initial_predictions[0]['text'] = '[x_x]'
    before = q.summarize(rows, initial_predictions); after = q.summarize(rows, best_predictions)
    torch.save({'state_dict': {'w': torch.zeros(1)}, 'steps': 0}, tmp_path / 'initial.pth')
    torch.save({'state_dict': {'w': torch.ones(1)}, 'steps': 7}, tmp_path / 'best.pth')
    source = tmp_path / 'quality-selection-source.py'; source.write_bytes(Path(q.__file__).read_bytes())
    initial = {'metrics': {'quality': before}, 'checkpoint_sha256': common.digest(tmp_path / 'initial.pth')}
    selection = {'step': 7, 'checkpoint_sha256': common.digest(tmp_path / 'best.pth'),
                 'selection_policy': 'face-pareto', 'quality': after, 'dev_metrics': {'quality': after},
                 'quality_selection_sha256': common.digest(source)}
    failure = {'steps': 7, 'history': [{'step': 7, 'dev_metrics': {'quality': copy.deepcopy(after)}}]}
    if tamper == 'missing_quality': selection['quality'] = None
    elif tamper == 'source_sha': selection['quality_selection_sha256'] = '0'*64
    elif tamper == 'initial_checkpoint': initial['checkpoint_sha256'] = '0'*64
    elif tamper == 'history_quality': failure['history'][0]['dev_metrics']['quality'] = before
    elif tamper == 'unknown_policy': selection['selection_policy'] = 'unknown'
    common.write(tmp_path / 'quality-initial-dev.json', initial)
    common.write(tmp_path / 'best-selection.json', selection)
    common.write(tmp_path / 'last-failure.json', failure)
    names = ['initial.pth', 'best.pth', 'best-selection.json', 'last-failure.json',
             'quality-selection-source.py', 'quality-initial-dev.json']
    frozen = {'completed': False, 'exit_code': 1,
              'files': [{'file': n, 'sha256': common.digest(tmp_path / n)} for n in names]}
    if tamper == 'initial_inventory': frozen['files'] = [x for x in frozen['files'] if x['file'] != 'quality-initial-dev.json']
    elif tamper == 'policy_conflict': frozen['selection_policy'] = 'legacy'
    common.write(tmp_path / 'run-freeze.json', frozen)
    if tamper == 'none':
        result, files = adapter.finetune_evidence(tmp_path, 'best.pth')
        assert result['run_completed'] is False and result['steps'] == 7
        assert 'quality-selection-source.py' in files and 'quality-initial-dev.json' in files
    else:
        with pytest.raises(ValueError): adapter.finetune_evidence(tmp_path, 'best.pth')
