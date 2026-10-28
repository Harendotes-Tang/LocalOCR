"""Position proposals exercise real component boundaries, never OCR accuracy."""
import copy

import numpy as np
import pytest

from tools.RegionKaomoji import candidates as c, position_spans as positions


def scene():
    core = np.zeros((60, 120), dtype=np.uint8)
    core[16:36, 4:17] = core[16:36, 77:91] = 1
    for x, right in ((25, False), (65, True)):
        stroke = np.zeros((20, 5), dtype=np.uint8)
        stroke[:, 0] = stroke[0, :] = stroke[-1, :] = 1
        core[16:36, x:x+5] |= np.fliplr(stroke) if right else stroke
    core[24:27, 35:38] = core[24:27, 53:56] = 1
    core[32:34, 43:49] = 1
    base = core.copy()
    base[10:12, 35:37] = base[27:29, 20:23] = 1
    return base, core, [{'bbox': [4, 9, 91, 36]}]


def rows():
    return [{'parent_bbox': [4, 9, 91, 36], 'parent_index': 0, 'trace_sha256': 'a'*64,
             'tokens': [{'text': '[', 'x_start': 25, 'x_end': 30},
                        {'text': '..', 'x_start': 35, 'x_end': 56},
                        {'text': ']', 'x_start': 65, 'x_end': 70}]}]


def run(mode, trace_rows=(), limits=None):
    base, core, lines = scene()
    before = base.copy(), core.copy()
    value = positions.propose_spans(base, core, lines, [], [], trace_rows=trace_rows,
        limits=positions.DEFAULT_LIMITS if limits is None else limits,
        options={**positions.DEFAULT_OPTIONS, 'mode': mode})
    assert np.array_equal(base, before[0]) and np.array_equal(core, before[1])
    return value


def test_ctc_hint_recovers_complete_weak_top_and_exterior_ink():
    result = run('ctc', rows())
    full = next(p for p in result['proposals'] if p['ink_bbox'] == [20, 10, 70, 36])
    proof = full['position_span']
    assert proof['horizontal_hint'] == [25, 70]
    assert proof['body_safe'] is False and proof['original_pixels_required']
    assert proof['base_only_pixels'] == 10
    assert proof['trace_sha256'] == 'a'*64


def test_union_shares_cap_and_deduplicates_pixel_envelopes():
    limited = {**positions.DEFAULT_LIMITS, 'maximum_candidates': 1}
    result = run('union', rows(), limited)
    assert len(result['proposals']) == 1
    assert result['crop_pixels'] == np.prod(np.diff(np.array(result['proposals'][0]['ink_bbox']).reshape(2, 2), axis=0))
    assert result['proposals'][0]['position_span']['proposal_kind'] == 'components'


def test_missing_trace_keeps_component_fallback_and_ctc_declares_missing():
    assert run('components')['proposals']
    assert run('union')['proposals']
    assert not run('ctc')['proposals']
    assert run('ctc')['rejections']['trace-unavailable-or-unsupported'] == 1


def test_token_with_multiple_codepoints_uses_its_full_extent():
    row = {'tokens': [{'text': '[(..)]', 'x_start': 20, 'x_end': 70}]}
    hints = list(positions._hints([row]))
    assert len(hints) == 2 and all(h[1:] == (20, 70) for h in hints)


def test_invalid_original_extent_is_rejected():
    value = rows()
    value[0]['tokens'][0]['x_start'] = float('nan')
    with pytest.raises(ValueError, match='extent'):
        run('ctc', value)


def test_ctc_cannot_cross_a_complete_core_stroke():
    base, core, lines = scene()
    core[0:20, 40] = 1
    base |= core
    result = positions.propose_spans(base, core, lines, [], [], trace_rows=rows(),
        limits=positions.DEFAULT_LIMITS, options={**positions.DEFAULT_OPTIONS, 'mode': 'ctc'})
    assert not result['proposals']
    assert result['rejections']['component-crosses-physical-line']


def test_absent_option_does_not_invoke_position_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Position helper must be opt-in')
    monkeypatch.setattr(positions, 'propose_spans', forbidden)
    base, _, _ = scene()
    pixels = np.repeat((255-base*255)[:, :, None], 3, axis=2)
    result = c.locate_candidates(pixels, [])
    assert 'position_span_detection' not in result


def test_routable_legacy_candidates_keep_priority_when_cap_is_full(monkeypatch):
    base, _, _ = scene()
    pixels = np.repeat((255-base*255)[:, :, None], 3, axis=2)
    settings = {**copy.deepcopy(c.DEFAULT_CONFIG), 'maximum_candidates': 1}
    old = c.locate_candidates(pixels, [], settings)
    assert old['candidates'][0]['route_eligible']
    def forbidden(*args, **kwargs):
        raise AssertionError('No new allocation under exhausted cap')
    monkeypatch.setattr(positions, 'propose_spans', forbidden)
    actual = c.locate_candidates(pixels, [], {**settings,
        'position_spans': {**positions.DEFAULT_OPTIONS, 'mode': 'ctc'}})
    assert actual['candidates'] == old['candidates']
    assert actual['position_span_detection']['rejections']['shared-budget-exhausted']


@pytest.mark.parametrize('extra', [{'position_spans': None},
    {'position_spans': {}}, {'position_spans': {'strategy': 'ctc-components-v1', 'mode': 'bad'}},
    {'position_spans': positions.DEFAULT_OPTIONS, 'local_windows': {}},
    {'position_spans': positions.DEFAULT_OPTIONS, 'line_spans': {}}])
def test_invalid_or_conflicting_proposer_contracts_fail_closed(extra):
    with pytest.raises(ValueError):
        c.configuration({**copy.deepcopy(c.DEFAULT_CONFIG), **extra})
