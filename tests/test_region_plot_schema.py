"""Region documentation schema tests; all numeric results are transparent fixtures."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.Documentation import plot_region_evaluation as p


def fixture(heldout=True):
    binding = {'sha256': 'a' * 64, 'split_sha256': {'dev': 'b' * 64, 'heldout': 'c' * 64},
               'counts': deepcopy(p.COUNTS)}
    reports = {}
    for split, scale in (('dev', 1), ('heldout', 2)):
        faces = p.COUNTS[split]['faces']
        loc = {'faces': faces, 'found_any': faces - 5 * scale,
               'complete_ink_coverage': faces - 9 * scale, 'safe_complete_coverage': faces - 20 * scale,
               'missed': 5 * scale, 'partial_only': 4 * scale,
               'eligible_complete_ink_coverage': faces - 22 * scale,
               'routed_complete_ink_coverage': faces - 24 * scale, 'safe_routed_complete_coverage': faces - 26 * scale,
               'candidate_count': 300 * scale,
               'intersecting_body': 170 * scale, 'intersecting_graphic': 25 * scale,
               'routed_candidate_count': 130 * scale, 'routed_candidates_intersecting_body': 30 * scale,
               'routed_candidates_intersecting_graphic': 12 * scale,
               'body_regions': 180 * scale, 'body_regions_with_candidate': 160 * scale, 'body_regions_routed': 30 * scale,
               'graphic_regions': 40 * scale, 'graphic_regions_with_candidate': 25 * scale, 'graphic_regions_routed': 12 * scale,
               'negative_scenes': 40 * scale, 'negative_scenes_with_candidate': 20 * scale, 'negative_scenes_routed': 10 * scale}
        models = []
        for index, family in enumerate(p.FAMILIES):
            models.append({'id': family + ('-current' if index == 0 else '-fixture-final'), 'family': family,
                'stage': 'production' if index == 0 else 'final-fusion', 'model_sha256': str(index + 1) * 64,
                'vocabulary_sha256': 'd' * 64, 'preprocessing_sha256': 'e' * 64,
                'face_regions': {'count': faces, 'exact': 3 * scale + index * 4,
                                 'characters': 1600 * scale, 'character_errors': 400 * scale - index * 30},
                'body': {'regions': 180 * scale, 'baseline_exact': 100 * scale, 'final_exact': (100 - index) * scale,
                         'baseline_correct_to_wrong': index * scale, 'erroneous_rewrites': index * 3 * scale,
                         'body_intersection_transcript_changes': index * 2 * scale, 'new_body_intersection_errors': index * scale},
                'negative': {'blank': {'scenes': 20 * scale, 'baseline_nonempty': 0, 'final_nonempty': index * scale, 'new_nonempty': index * scale},
                             'graphic-only': {'scenes': 20 * scale, 'baseline_nonempty': 2 * scale, 'final_nonempty': (index + 2) * scale, 'new_nonempty': index * scale},
                             'graphic_regions': {'regions': 40 * scale, 'baseline_nonempty': 2 * scale, 'final_nonempty': (index + 2) * scale, 'new_nonempty': index * scale}}})
        reports[split] = {'report_sha256': ('f' if split == 'dev' else '9') * 64,
            'dataset_sha256': binding['sha256'], 'split_sha256': binding['split_sha256'][split],
            'pipeline_sha256': '8' * 64, 'config_sha256': '7' * 64,
            'selection_sha256': None if split == 'dev' else '6' * 64,
            'first_use_sha256': None if split == 'dev' else '5' * 64,
            'evaluation_backend': p.BACKEND, 'evaluation_scope': p.SCOPE,
            'runtime': {'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1},
            'scenes': p.COUNTS[split]['scenes'], 'localization': loc, 'models': models}
    return {'schema_version': p.SCHEMA, 'fixture': True, 'dataset': binding,
            'plot_font': {'sha256': p.FONT_SHA256, 'license': 'OFL-1.1'},
            'dev': reports['dev'], 'heldout': reports['heldout'] if heldout else None}


def test_complete_and_development_only_are_distinct():
    p.validate(fixture())
    p.validate(fixture(False))


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(private_text='do not publish'),
    lambda x: x['dev']['models'][1].update(prediction='do not publish'),
    lambda x: x['dev']['localization'].update(roi=[0, 0, 10, 10]),
    lambda x: x['dev']['runtime'].update(path='D:/private/file'),
    lambda x: x['dev']['models'][1]['body'].update(reference='do not publish'),
    lambda x: x['dev']['models'][1].update(id='D:/private/file'),
])
def test_fields_cannot_carry_raw_material(mutation):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError):
        p.validate(data)


@pytest.mark.parametrize('mutation', [
    lambda x: x['dev'].update(evaluation_backend='CUDA-native'),
    lambda x: x['dev'].update(evaluation_scope='single-crop'),
    lambda x: x['dev']['models'][1].update(stage='raw-specialist'),
    lambda x: x['dev']['runtime'].update(providers=['CUDAExecutionProvider']),
    lambda x: x['dev']['runtime'].update(intra_op_threads=8),
    lambda x: x['dev']['runtime'].update(inter_op_threads=True),
    lambda x: x['heldout'].update(report_sha256='f' * 64),
    lambda x: x['dev'].update(selection_sha256='0' * 64),
    lambda x: x['dev'].update(dataset_sha256='0' * 64),
    lambda x: x['dev'].update(split_sha256='c' * 64),
    lambda x: x['heldout'].update(first_use_sha256=None),
    lambda x: x['heldout'].update(selection_sha256=None),
    lambda x: x['heldout']['models'][1].update(model_sha256='0' * 64),
    lambda x: x['heldout'].update(config_sha256='0' * 64),
    lambda x: x['dataset']['counts']['dev'].update(faces=119),
    lambda x: x['dev']['models'].pop(),
])
def test_scope_runtime_freeze_and_stage_bindings(mutation):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError):
        p.validate(data)


@pytest.mark.parametrize('mutation', [
    lambda x: x['dev']['localization'].update(safe_complete_coverage=120),
    lambda x: x['dev']['localization'].update(missed=0),
    lambda x: x['dev']['localization'].update(intersecting_body=301),
    lambda x: x['dev']['localization'].update(body_regions_routed=181),
    lambda x: x['dev']['localization'].update(routed_candidates_intersecting_body=131),
    lambda x: x['dev']['localization'].update(routed_complete_ink_coverage=119),
    lambda x: x['dev']['localization'].update(body_regions_routed=170),
    lambda x: x['dev']['models'][1]['negative']['blank'].update(new_nonempty=0),
    lambda x: x['dev']['models'][1]['face_regions'].update(count=119),
    lambda x: x['dev']['models'][1]['face_regions'].update(character_errors=1),
    lambda x: x['dev']['models'][1]['face_regions'].update(characters=1601),
    lambda x: x['dev']['models'][1]['body'].update(baseline_correct_to_wrong=101),
    lambda x: x['dev']['models'][1]['body'].update(new_body_intersection_errors=181),
    lambda x: x['dev']['models'][1]['negative']['blank'].update(final_nonempty=21),
    lambda x: x['dev']['models'][1]['negative']['blank'].update(scenes=21),
    lambda x: x['dev']['models'][1]['negative']['graphic_regions'].update(regions=39),
    lambda x: x['dev']['localization'].update(missed=True),
])
def test_denominators_arithmetic_and_integer_types(mutation):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError):
        p.validate(data)


def test_overlap_counts_need_not_be_exclusive():
    data = fixture(False)
    data['dev']['localization'].update(intersecting_body=290, intersecting_graphic=290)
    p.validate(data)


def test_zero_candidates_have_no_invented_rate_or_coverage():
    data = fixture(False)
    loc = data['dev']['localization']
    for key in ('found_any', 'partial_only', 'complete_ink_coverage', 'safe_complete_coverage',
                'eligible_complete_ink_coverage', 'routed_complete_ink_coverage', 'safe_routed_complete_coverage',
                'candidate_count', 'routed_candidate_count', 'intersecting_body', 'intersecting_graphic',
                'routed_candidates_intersecting_body', 'routed_candidates_intersecting_graphic',
                'body_regions_with_candidate', 'body_regions_routed', 'graphic_regions_with_candidate', 'graphic_regions_routed',
                'negative_scenes_with_candidate', 'negative_scenes_routed'):
        loc[key] = 0
    loc['missed'] = loc['faces']
    p.validate(data)
    loc['intersecting_body'] = 1
    with pytest.raises(ValueError):
        p.validate(data)


def test_empty_reference_and_null_reports_are_rejected():
    data = fixture(False)
    data['dev']['models'][0]['face_regions']['characters'] = 0
    with pytest.raises(ValueError):
        p.validate(data)
    data = fixture(False)
    data['dev'] = None
    with pytest.raises(ValueError):
        p.validate(data)


def test_unseen_zero_baseline_correct_denominator_is_explicit():
    data = fixture(False)
    for model in data['dev']['models']:
        model['body'].update(baseline_exact=0, baseline_correct_to_wrong=0, final_exact=0)
    p.validate(data)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_json_parser_rejects_ambiguous_input(raw):
    with pytest.raises(ValueError):
        p.strict_json(raw)


def test_fixture_destination_precedes_font_read(tmp_path, monkeypatch):
    source = tmp_path / 'anonymous-fixture.json'
    source.write_text(json.dumps(fixture(False)), encoding='utf-8')
    seen = []
    original = Path.read_bytes
    def read(path):
        seen.append(path)
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(ValueError, match='Fixture output'):
        p.main(['--data', str(source), '--font', str(tmp_path / 'private-font'),
                '--output', str(Path(p.__file__).resolve().parents[2] / 'docs' / 'forbidden-fixture')])
    assert seen == [source]
