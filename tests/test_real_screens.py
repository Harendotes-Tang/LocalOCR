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
