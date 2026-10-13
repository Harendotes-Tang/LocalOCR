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
