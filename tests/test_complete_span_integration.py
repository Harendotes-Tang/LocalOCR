"""Optional-span integration guards; fixtures do not measure model quality."""
import copy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools.RegionKaomoji import candidates as c, fusion as f, line_spans


def pixels():
    image = Image.new('RGB', (160, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.arc((20, 20, 34, 50), 90, 270, fill='black', width=2)
    draw.arc((88, 20, 102, 50), 270, 90, fill='black', width=2)
    draw.ellipse((43, 31, 50, 38), outline='black', width=2)
    draw.ellipse((72, 31, 79, 38), outline='black', width=2)
    draw.line((54, 43, 68, 43), fill='black', width=2)
    return np.asarray(image)


def config():
    return {**copy.deepcopy(c.DEFAULT_CONFIG), 'line_spans': line_spans.configuration(),
            'local_ownership': {'strategy': 'pixel-partition-v1'}}


def test_absent_options_are_not_injected_or_run():
    normalized = c.configuration()
    assert 'line_spans' not in normalized and 'local_ownership' not in normalized
    actual = c.locate_candidates(pixels(), [])
    assert 'line_span_detection' not in actual
    assert all('local_ownership' not in item['evidence'] for item in actual['candidates'])


@pytest.mark.parametrize('key,value', [('line_spans', None), ('line_spans', {}),
                                      ('local_ownership', None), ('local_ownership', True),
                                      ('local_ownership', {'strategy': 'unchecked'})])
def test_unknown_or_partial_optional_contract_fails(key, value):
    with pytest.raises(ValueError):
        c.configuration({**copy.deepcopy(c.DEFAULT_CONFIG), key: value})


def test_old_routable_candidates_have_priority_under_shared_count_cap(monkeypatch):
    image = pixels()
    base = {**copy.deepcopy(c.DEFAULT_CONFIG), 'maximum_candidates': 1}
    old = c.locate_candidates(image, [], base)
    def forbidden(*args, **kwargs):
        raise AssertionError('Exhausted shared cap must stop proposal and materialization work')
    monkeypatch.setattr(line_spans, 'propose_spans', forbidden)
    monkeypatch.setattr(c, '_span_record', forbidden)
    new = c.locate_candidates(image, [], {**base, 'line_spans': line_spans.configuration()})
    old_eligible = [item for item in old['candidates'] if item['route_eligible']]
    assert old_eligible
    assert new['candidates'] == old_eligible
    assert new['discovered_count'] <= 1
    assert new['crop_pixels'] <= base['maximum_crop_pixels']
    assert new['line_span_detection']['rejections']['shared-budget-exhausted'] == 1


def test_optional_route_is_bounded_and_does_not_mutate_original_pixels():
    image = pixels()
    before = image.copy()
    setting = config()
    output = c.locate_candidates(image, [], setting)
    assert np.array_equal(image, before)
    assert len(output['candidates']) <= setting['maximum_candidates']
    assert len({item['candidate_id'] for item in output['candidates']}) == len(output['candidates'])
    assert len({tuple(item['bbox']) for item in output['candidates']}) == len(output['candidates'])
    assert output['crop_pixels'] == sum(c.area(item['bbox']) for item in output['candidates'])
    assert output['crop_pixels'] <= setting['maximum_crop_pixels']
    for item in output['candidates']:
        if 'line_span' in item['evidence']:
            limits = item['evidence']['line_span']['crop_limits']
            box = item['bbox']
            assert limits[0] <= box[0] < box[2] <= limits[2]
            assert limits[1] <= box[1] < box[3] <= limits[3]
            assert item['evidence']['line_span']['body_safe'] is False


def test_fusion_recomputes_optional_evidence_and_rejects_forged_proof():
    image = pixels()
    setting = config()
    actual = c.locate_candidates(image, [], setting)['candidates']
    assert actual
    forged = copy.deepcopy(actual)
    forged[0]['evidence']['local_ownership'] = {'parent_index': 0, 'proof': {'safe': True}}
    with pytest.raises(ValueError, match='differs from automatic'):
        f.fuse_candidates(image, [], forged, {}, candidate_config=setting)


def test_dual_gate_does_not_accept_relative_gain_without_four_strict_reports():
    from tools.PretrainedOcr import quality_evaluate as q, round_profile as rp
    with pytest.raises(ValueError, match='four strict'):
        q._complete_span_proof([], 'not-read.json', rp.COMPLETE_SPANS.root)


@pytest.mark.parametrize('passed,matching', [(False, True), (True, False)])
def test_dual_gate_rejects_failed_or_wrong_selection_proof(tmp_path, monkeypatch, passed, matching):
    from tools.PretrainedOcr import common, quality_evaluate as q, round_profile as rp
    from tools.RegionKaomoji import span_quality
    expected = {'file': 'selected-dev.json', 'sha256': 'a'*64}
    selection = tmp_path/'selection.json'
    common.write(selection, {'selected_id': 'tiny', 'dev_report': expected})
    actual = expected if matching else {'file': 'other-dev.json', 'sha256': 'b'*64}
    monkeypatch.setattr(span_quality, 'verify_round_development', lambda *args, **kwargs: {
        'passed': passed, 'reports': {'final_new': {'report': actual}}})
    with pytest.raises(ValueError, match='thresholds failed|actual scene selection'):
        q._complete_span_proof(['a', 'b', 'c', 'd'], selection, rp.COMPLETE_SPANS.root)
