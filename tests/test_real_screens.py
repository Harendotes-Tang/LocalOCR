"""Known-input real screenshot evaluation: manifest validation and private-flag scoring."""
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.RegionKaomoji import real_screens as rs


def manifest(tmp_path, **changes):
    (tmp_path / 'screens').mkdir(exist_ok=True)
    Image.new('RGB', (40, 20), 'white').save(tmp_path / 'screens' / 'shot-001.png')
    value = {'schema': rs.MANIFEST_SCHEMA, 'usage': 'independent-evaluation-only', 'environment': 'unit',
             'screens': [{'file': 'screens/shot-001.png', 'lines': [
                 {'kind': 'body', 'text': 'river 晴川'}, {'kind': 'face', 'text': '(≧▽≦)'},
                 {'kind': 'body', 'text': '今天 (´･ω･`) 好', 'faces': ['(´･ω･`)']}, {'kind': 'face', 'text': 'orz', 'ambiguous': True}]}]}
    value.update(changes)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def test_manifest_requires_independent_usage_local_pngs_and_exact_lines(tmp_path):
    loaded = rs.load_manifest(manifest(tmp_path))
    lines = loaded['value']['screens'][0]['lines']
    assert lines[1]['grammar_accepted'] is True and lines[3]['grammar_accepted'] is False
    for changes in ({'usage': 'training'}, {'schema': 'x'}, {'screens': []},
                    {'screens': [{'file': '../shot.png', 'lines': [{'kind': 'face', 'text': 'x'}]}]},
                    {'screens': [{'file': 'screens/shot-001.png', 'lines': [{'kind': 'emoji', 'text': 'x'}]}]},
                    {'screens': [{'file': 'screens/shot-001.png', 'lines': [{'kind': 'face', 'text': ' '}]}]},
                    {'screens': [{'file': 'screens/shot-001.png', 'lines': [{'kind': 'body', 'text': 'a', 'faces': ['b']}]}]},
                    {'screens': [{'file': 'screens/shot-001.png', 'lines': [{'kind': 'face', 'text': 'x', 'ambiguous': 'yes'}]}]}):
        with pytest.raises(ValueError):
            rs.load_manifest(manifest(tmp_path, **changes))


def test_scoring_counts_exact_regions_containment_and_lost_body_lines(tmp_path):
    loaded = rs.load_manifest(manifest(tmp_path))
    screen = loaded['value']['screens'][0]
    prediction = {'baseline': [{'text': 'river 晴川'}, {'text': '(=▽=)'}, {'text': '今天 (´･w･`) 好'}],
                  'baseline_text': 'river 晴川\n(=▽=)\n今天 (´･w･`) 好',
                  'models': {'tiny': {'regions': [{'text': 'river 晴川'}, {'text': '(≧▽≦)'}, {'text': '今天 (´･ω･`) 好'}],
                                      'text': 'river 晴川\n(≧▽≦)\n今天 (´･ω･`) 好'},
                             'bad': {'regions': [{'text': '(≧▽≦)'}], 'text': '(≧▽≦)'}}}
    scored = rs.score_screen(screen, prediction, ['tiny', 'bad'])
    summary = rs.aggregate([scored], ['tiny', 'bad'])
    assert summary['faces'] == 1 and summary['faces_ambiguous_excluded'] == 1 and summary['body_lines'] == 2 and summary['embedded_faces'] == 1
    # The misread body line is not contained in the baseline text, so it cannot be "lost" later.
    assert summary['baseline'] == {'face_exact_region': 0, 'face_contained': 0, 'body_contained': 1, 'embedded_face_contained': 0}
    assert summary['models']['tiny'] == {'face_exact_region': 1, 'face_contained': 1, 'face_wrong_to_exact': 1, 'face_exact_to_wrong': 0,
                                         'body_contained': 2, 'body_lost': 0, 'embedded_face_contained': 1, 'gates': {'no_body_lost': True}}
    assert summary['models']['bad']['body_lost'] == 1 and summary['models']['bad']['gates']['no_body_lost'] is False
    assert all('text' not in record for record in scored)  # private strings never enter the aggregate records


def test_output_must_belong_to_a_registered_round(tmp_path, monkeypatch):
    from tools.PretrainedOcr import round_profile as rp
    import argparse
    monkeypatch.setattr(rp, 'ROUNDS', ())
    monkeypatch.setattr(rs.e, 'EVALUATION_ROOT', tmp_path)
    args = argparse.Namespace(manifest=str(manifest(tmp_path)), output=str(tmp_path / 'evaluation-real'), config=None, classifier=None,
                              baseline_models=str(tmp_path), candidate=[])
    with pytest.raises(ValueError, match='registered research round'):
        rs.evaluate_manifest(args)


def unit_round(tmp_path, monkeypatch):
    from tools.PretrainedOcr import round_profile as rp
    profile = rp.RoundProfile(name='unit-real', change='unit', root=tmp_path / 'unit-real', maximum_bytes=2 * 1024 ** 3,
                              owned_bytes=900 * 1024 ** 2, previous_budget=tmp_path / 'old.json', previous_sha256='0' * 64,
                              previous_used_seconds=0.0, maximum_seconds=0.0, specialist_families=1)
    profile.root.mkdir()
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(rs.e, 'EVALUATION_ROOT', tmp_path)
    return profile


def test_probe_registration_and_independent_first_use_claim(tmp_path, monkeypatch):
    profile = unit_round(tmp_path, monkeypatch)
    loaded = rs.load_manifest(manifest(tmp_path))
    config = rs.e.configuration()
    output = profile.root / 'evaluation' / 'known-input-001'
    sidecar = profile.root / rs.PROBE_SIDECAR
    sidecar.write_text(json.dumps({'schema': 'localocr.development-demotion.v1', 'kind': rs.PROBE_KIND, 'original_rewritten': False,
                                   'new_round': 'unit', 'old_datasets': [{'file': 'manifest.json', 'sha256': loaded['sha256']}]}), encoding='utf-8')
    assert rs.probe_registration(loaded['sha256'])['round'] == 'unit-real'
    with pytest.raises(ValueError, match='development probe'):
        rs.claim_first_use(loaded, output, config)
    sidecar.unlink()
    assert rs.probe_registration(loaded['sha256']) is None
    receipt = rs.claim_first_use(loaded, output, config, dual=[])
    path = rs.first_use_path(profile.root, loaded['sha256'])
    assert Path(receipt['file']) == path and path.exists()
    value = json.loads(path.read_text(encoding='utf-8'))
    assert value['schema_version'] == rs.FIRST_USE_SCHEMA and value['manifest']['sha256'] == loaded['sha256']
    assert value['config']['object_sha256'] == rs.e.hash_object(config) and value['source_inventory']['files'] > 0
    with pytest.raises(ValueError, match='already consumed'):
        rs.claim_first_use(loaded, output, config)
    assert rs.existing_first_use(loaded['sha256']) == path


def test_evaluate_manifest_claims_before_models_load_and_probe_needs_registration(tmp_path, monkeypatch):
    import argparse
    profile = unit_round(tmp_path, monkeypatch)
    path = manifest(tmp_path)
    args = argparse.Namespace(manifest=str(path), output=str(profile.root / 'evaluation' / 'ki'), config=None, classifier=None,
                              baseline_models=str(tmp_path / 'no-models'), candidate=[], probe=True, dual=[])
    with pytest.raises(ValueError, match='Probe mode requires'):
        rs.evaluate_manifest(args)
    args.probe = False
    with pytest.raises(Exception, match='(?i)model|baseline|No such|not exist|missing'):
        rs.evaluate_manifest(args)
    sha = rs.load_manifest(path)['sha256']
    assert rs.first_use_path(profile.root, sha).exists()
    with pytest.raises(ValueError, match='already consumed'):
        rs.evaluate_manifest(args)


def test_width_diagnostic_counts_only_faces_that_differ_in_outer_bracket_width(tmp_path):
    loaded = rs.load_manifest(manifest(tmp_path))
    screen = loaded['value']['screens'][0]
    prediction = {'baseline': [{'text': 'river 晴川'}, {'text': '(=▽=)'}], 'baseline_text': 'river 晴川\n(=▽=)',
                  'routing': {'shape_detection': {'threshold': 80, 'base_pairs': 1, 'core_pairs': 3, 'merged_bracket_candidates': 2}},
                  'models': {'tiny': {'regions': [{'text': 'river 晴川'}, {'text': '（≧▽≦）'}], 'text': 'river 晴川\n（≧▽≦）',
                                      'decisions': [{'accepted': True, 'bracket_width_source': 'raw-view'}, {'accepted': False, 'bracket_width_source': None}]}}}
    scored = rs.score_screen(screen, prediction, ['tiny'])
    face = next(r for r in scored if r['kind'] == 'face' and not r['ambiguous'])
    assert face['models']['tiny']['exact_region'] is False and face['models']['tiny']['exact_except_outer_bracket_width'] is True
    summary = rs.aggregate([scored], ['tiny'])
    assert summary['models']['tiny']['face_exact_region'] == 0
    diag = rs.diagnostics([scored], [prediction], ['tiny'])
    assert diag['shape_base_pairs'] == 1 and diag['shape_core_pairs'] == 3 and diag['merged_bracket_candidates'] == 2 and diag['shape_threshold'] == 80
    assert diag['models']['tiny'] == {'face_exact_except_outer_bracket_width': 1, 'bracket_width_sources': {'raw-view': 1}}


# ---- bracket-pairs round: probe reports bind their per-screen predictions and proofs recompute them

BINDING = {'models': [{'id': 'unit', 'candidate_sha256': 'a' * 64}], 'config_sha256': 'c' * 64,
           'classifier': {'sha256': 'k' * 64}, 'baseline': {'manifest_sha256': 'b' * 64}}


def _prediction(lose_body=False):
    face_text = '(≧▽≦)'
    regions = [{'text': face_text}, {'text': '今天 (´･ω･`) 好'}] if lose_body else \
        [{'text': 'river 晴川'}, {'text': face_text}, {'text': '今天 (´･ω･`) 好'}]
    return {'baseline': [{'text': 'river 晴川'}, {'text': '(=▽=)'}, {'text': '今天 (´･w･`) 好'}],
            'baseline_text': 'river 晴川\n(=▽=)\n今天 (´･w･`) 好', 'first_pass': {'split_parents': 1},
            'routing': {'shape_detection': {'threshold': 80, 'base_pairs': 1, 'core_pairs': 2, 'merged_bracket_candidates': 1}},
            'models': {'unit': {'regions': regions, 'text': '\n'.join(r['text'] for r in regions),
                                'decisions': [{'accepted': True, 'bracket_width_source': 'pixels'}]}}}


def two_screen_manifest(tmp_path):
    (tmp_path / 'screens').mkdir(exist_ok=True)
    lines = [{'kind': 'body', 'text': 'river 晴川'}, {'kind': 'face', 'text': '(≧▽≦)'},
             {'kind': 'body', 'text': '今天 (´･ω･`) 好', 'faces': ['(´･ω･`)']}]
    for index, colour in ((1, 'white'), (2, 'gray')):
        Image.new('RGB', (40, 20), colour).save(tmp_path / 'screens' / f'shot-00{index}.png')
    value = {'schema': rs.MANIFEST_SCHEMA, 'usage': 'independent-evaluation-only', 'environment': 'unit',
             'screens': [{'file': f'screens/shot-00{index}.png', 'lines': lines} for index in (1, 2)]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def register_probe(profile, loaded):
    sidecar = profile.root / rs.PROBE_SIDECAR
    sidecar.write_text(json.dumps({'schema': 'localocr.development-demotion.v1', 'kind': rs.PROBE_KIND, 'original_rewritten': False,
                                   'new_round': 'unit', 'old_datasets': [{'file': 'manifest.json', 'sha256': loaded['sha256']}]}), encoding='utf-8')


def probe_fixture(tmp_path, monkeypatch, with_sha=True, lose_body=False):
    """A complete two-screen probe report written the way ``evaluate_manifest`` writes one (no model inference)."""
    from tools.PretrainedOcr import common
    profile = unit_round(tmp_path, monkeypatch)
    loaded = rs.load_manifest(two_screen_manifest(tmp_path))
    register_probe(profile, loaded)
    output = profile.root / 'evaluation' / 'known-input-probe-unit'
    output.mkdir(parents=True)
    scored, predictions = [], []
    with (output / 'predictions-private.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for screen in loaded['value']['screens']:
            prediction = _prediction(lose_body)
            lines = rs.score_screen(screen, prediction, ['unit'])
            scored.append(lines)
            predictions.append(prediction)
            stream.write(json.dumps({'file': screen['file'], 'sha256': common.digest(loaded['root'] / screen['file']), 'seconds': .1,
                                     'prediction': prediction, 'lines': lines}, ensure_ascii=False) + '\n')
    report = {'schema_version': rs.SCHEMA, 'manifest': {'file': loaded['file'], 'sha256': loaded['sha256'], 'screens': 2},
              **BINDING, 'summary': rs.summarize(scored, predictions, ['unit']), 'independent': False,
              'probe': rs.probe_registration(loaded['sha256']), 'promote': False, 'enabled': False}
    if with_sha:
        report['predictions_private_sha256'] = common.digest(output / 'predictions-private.jsonl')
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    return output, loaded


def _rewrite(path, change):
    value = json.loads(path.read_text(encoding='utf-8'))
    change(value)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def _resign(output):
    from tools.PretrainedOcr import common
    _rewrite(output / 'evaluation.json', lambda value: value.update(
        predictions_private_sha256=common.digest(output / 'predictions-private.jsonl')))


def _rows(output):
    return (output / 'predictions-private.jsonl').read_text(encoding='utf-8').split('\n')[:-1]


def _write_rows(output, rows):
    (output / 'predictions-private.jsonl').write_text(''.join(row + '\n' for row in rows), encoding='utf-8')


def _selection(tmp_path):
    value = {'selected_id': 'unit', 'pipeline_binding': {'models': [{'candidate_sha256': 'a' * 64}], 'config_sha256': 'c' * 64,
                                                          'config': {'classifier': {'sha256': 'k' * 64}}, 'baseline': {'manifest_sha256': 'b' * 64},
                                                          'dataset': {'dataset_sha256': 's' * 64}}}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_recompute_probe_reproduces_the_reported_summary(tmp_path, monkeypatch):
    output, _ = probe_fixture(tmp_path, monkeypatch)
    report = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
    assert rs.recompute_probe(output / 'evaluation.json') == report['summary']
    assert report['summary']['body_lines'] == 4 and report['summary']['split'] == {'parents': 2}


def test_recompute_probe_refuses_a_report_without_the_predictions_sha(tmp_path, monkeypatch):
    output, _ = probe_fixture(tmp_path, monkeypatch, with_sha=False)
    with pytest.raises(ValueError, match='predictions'):
        rs.recompute_probe(output / 'evaluation.json')


def test_recompute_probe_refuses_a_summary_claiming_no_lost_body_lines(tmp_path, monkeypatch):
    output, _ = probe_fixture(tmp_path, monkeypatch, lose_body=True)
    report = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
    assert report['summary']['models']['unit']['body_lost'] == 2
    _rewrite(output / 'evaluation.json', lambda value: value['summary']['models']['unit'].update(
        body_lost=0, gates={'no_body_lost': True}))
    with pytest.raises(ValueError, match='summary'):
        rs.recompute_probe(output / 'evaluation.json')


def test_recompute_probe_refuses_changed_predictions(tmp_path, monkeypatch):
    output, _ = probe_fixture(tmp_path, monkeypatch)
    rows = _rows(output)
    row = json.loads(rows[0])
    row['prediction']['models']['unit']['regions'][1]['text'] = '(=▽=)'
    _write_rows(output, [json.dumps(row, ensure_ascii=False), rows[1]])
    with pytest.raises(ValueError, match='predictions'):
        rs.recompute_probe(output / 'evaluation.json')
    _resign(output)  # a consistent re-signature still fails: the recorded lines no longer score from the prediction
    with pytest.raises(ValueError, match='score'):
        rs.recompute_probe(output / 'evaluation.json')


def test_recompute_probe_refuses_a_changed_screenshot(tmp_path, monkeypatch):
    output, loaded = probe_fixture(tmp_path, monkeypatch)
    Image.new('RGB', (40, 20), 'black').save(loaded['root'] / 'screens' / 'shot-002.png')
    with pytest.raises(ValueError, match='screenshot'):
        rs.recompute_probe(output / 'evaluation.json')


def test_recompute_probe_refuses_a_changed_manifest(tmp_path, monkeypatch):
    output, loaded = probe_fixture(tmp_path, monkeypatch)
    _rewrite(Path(loaded['file']), lambda value: value.update(environment='changed'))
    with pytest.raises(ValueError, match='manifest'):
        rs.recompute_probe(output / 'evaluation.json')


def test_recompute_probe_refuses_a_changed_development_registration(tmp_path, monkeypatch):
    output, _ = probe_fixture(tmp_path, monkeypatch)
    _rewrite(output / 'evaluation.json', lambda value: value['probe'].update(sidecar_sha256='0' * 64))
    with pytest.raises(ValueError, match='registration'):
        rs.recompute_probe(output / 'evaluation.json')


@pytest.mark.parametrize('change', ['reorder', 'truncate', 'extra'])
def test_recompute_probe_refuses_reordered_missing_or_extra_screens(tmp_path, monkeypatch, change):
    output, _ = probe_fixture(tmp_path, monkeypatch)
    rows = _rows(output)
    _write_rows(output, {'reorder': rows[::-1], 'truncate': rows[:1], 'extra': rows + rows[:1]}[change])
    _resign(output)
    with pytest.raises(ValueError, match='order'):
        rs.recompute_probe(output / 'evaluation.json')


def test_rows_keep_unicode_line_separators_inside_strings(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps({'text': 'a b c\u0085d'}, ensure_ascii=False) + '\n', encoding='utf-8')
    assert rs.read_rows(path) == [{'text': 'a b c\u0085d'}]


def test_regression_proof_recomputes_a_complete_probe_report(tmp_path, monkeypatch):
    from tools.PretrainedOcr import quality_evaluate as q
    output, _ = probe_fixture(tmp_path, monkeypatch)
    proof = q.regression_proof(output / 'evaluation.json', _selection(tmp_path))
    report = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
    assert proof['kind'] == 'known-input-probe' and proof['passed'] is True
    assert proof['body_lines'] == 4 and proof['body_lost'] == 0
    assert proof['predictions_private_sha256'] == report['predictions_private_sha256']


def test_regression_proof_fails_on_recomputed_lost_body_lines(tmp_path, monkeypatch):
    from tools.PretrainedOcr import quality_evaluate as q
    output, _ = probe_fixture(tmp_path, monkeypatch, lose_body=True)
    selection = _selection(tmp_path)
    proof = q.regression_proof(output / 'evaluation.json', selection)
    assert proof['passed'] is False and proof['body_lost'] == 2
    _rewrite(output / 'evaluation.json', lambda value: value['summary']['models']['unit'].update(
        body_lost=0, gates={'no_body_lost': True}))
    with pytest.raises(ValueError, match='summary'):
        q.regression_proof(output / 'evaluation.json', selection)


def test_evaluate_manifest_binds_the_predictions_sha_and_recomputes(tmp_path, monkeypatch):
    import argparse
    profile = unit_round(tmp_path, monkeypatch)
    path = two_screen_manifest(tmp_path)
    register_probe(profile, rs.load_manifest(path))

    class Stub:
        def __init__(self, source):
            self.binding = {'id': 'unit', 'candidate_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64}
    monkeypatch.setattr(rs.e, 'FullBaseline', Stub)
    monkeypatch.setattr(rs.e, 'Recognizer', Stub)
    monkeypatch.setattr(rs.e, 'predict_scene', lambda image, baseline, models, config, classifier=None: _prediction())
    args = argparse.Namespace(manifest=str(path), output=str(profile.root / 'evaluation' / 'known-input-probe-unit'), config=None,
                              classifier=None, baseline_models=str(tmp_path), candidate=[str(tmp_path / 'candidate.json')], probe=True, dual=[])
    report = rs.evaluate_manifest(args)
    from tools.PretrainedOcr import common
    assert report['predictions_private_sha256'] == common.digest(Path(args.output) / 'predictions-private.jsonl')
    assert rs.recompute_probe(Path(args.output) / 'evaluation.json') == report['summary']
