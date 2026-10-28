"""Compact CTC mechanism fixtures; no model session or accuracy assertion."""
import copy
import numpy as np
import pytest
from tools.RegionKaomoji import ctc_trace as trace


def probabilities(ids, count):
    value = np.zeros((len(ids), count), dtype=np.float32)
    value[np.arange(len(ids)), ids] = .95
    return value


def geometry(**changes):
    result = dict(crop_bbox=[10, 20, 110, 40], input_shape=[20, 100],
                  content_width=240, batch_width=480, resize_height=48,
                  padding=[0, 0, 0, 0], selected_view='raw', rotation=0,
                  inverse_perspective=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                  input_index=0, sorted_index=1, batch_index=0, batch_ordinal=0,
                  source_indices=[0], before_text='a')
    result.update(changes)
    return result


def test_blank_repeats_spaces_and_multicodepoint_tokens_are_exact():
    row = trace.compact_path(probabilities([0, 1, 1, 0, 1, 2, 3, 3, 0], 4),
                             ['blank', '^', ' ', 'e\u0301'], geometry(batch_width=240), trace.TraceLimits())
    assert row['text'] == '^^ e\u0301'
    assert row['runs'] == [[0, 0, 1], [1, 1, 3], [0, 3, 4], [1, 4, 5],
                           [2, 5, 6], [3, 6, 8], [0, 8, 9]]
    assert [item['text'] for item in row['tokens']] == ['^', '^', ' ', 'e\u0301']
    assert [(item['char_start'], item['char_end']) for item in row['tokens']] == [(0, 1), (1, 2), (2, 3), (3, 5)]
    assert row['tokens'][0]['x_start'] == pytest.approx(10 + 100 / 9)


def test_batch_padding_is_not_divided_over_the_crop():
    row = trace.compact_path(probabilities([0, 1, 0, 0], 2), ['blank', '('], geometry(), trace.TraceLimits())
    token = row['tokens'][0]
    assert (token['x_start'], token['x_end']) == (60., 110.)
    with pytest.raises(trace.UnsupportedTrace, match='padding'):
        trace.compact_path(probabilities([0, 0, 0, 1], 2), ['blank', '('], geometry(), trace.TraceLimits())


@pytest.mark.parametrize('change', [{'rotation': 180}, {'inverse_perspective': None},
                                   {'source_indices': [0, 1]}, {'selected_view': 'guessed'}])
def test_unknown_rotation_merge_and_mapping_do_not_make_positions(change):
    with pytest.raises(trace.UnsupportedTrace):
        trace.compact_path(probabilities([1, 0], 2), ['blank', '('], geometry(**change), trace.TraceLimits())


def test_limits_reject_before_argmax_and_before_record_growth():
    class Oversize(np.ndarray):
        def argmax(self, *args, **kwargs):
            raise AssertionError('oversize input must be rejected before argmax')
    value = probabilities([1] * 8, 2).view(Oversize)
    with pytest.raises(trace.UnsupportedTrace, match='step'):
        trace.compact_path(value, ['blank', '('], geometry(), trace.TraceLimits(max_steps_per_row=4))
    with pytest.raises(trace.UnsupportedTrace, match='byte'):
        trace.compact_path(probabilities([1, 0, 1], 2), ['blank', '('], geometry(),
                           trace.TraceLimits(max_record_bytes=50))


def test_sidecar_requires_live_authentication_and_revalidates_entries():
    row = trace.compact_path(probabilities([0, 1, 0], 2), ['blank', '('],
                             geometry(content_width=240, batch_width=240), trace.TraceLimits())
    entry = {'bbox': [10, 20, 110, 40], 'text': '('}
    sidecar = trace.make_sidecar('a' * 64, [200, 100], {'model': 'fixture'}, [row], [entry], {})
    assert len(trace.validate_sidecar('a' * 64, [200, 100], [entry], sidecar)) == 1
    assert not trace.validate_sidecar('a' * 64, [200, 100], [entry], dict(sidecar))
    mutated = copy.deepcopy(sidecar)
    mutated['rows'][0]['tokens'][0]['x_start'] = 99
    assert not trace.validate_sidecar('a' * 64, [200, 100], [entry], mutated)
    assert not trace.validate_sidecar('b' * 64, [200, 100], [entry], sidecar)
    assert not trace.validate_sidecar('a' * 64, [200, 100], [dict(entry, text='different')], sidecar)
    assert not trace.make_sidecar('a' * 64, [200, 100], {'model': 'fixture'},
                                  [row], [entry, entry], {})['rows']


def test_impossible_sidecar_header_raises_and_overflow_flag_stays_inside_cap():
    with pytest.raises(trace.UnsupportedTrace, match='sidecar-header-byte-cap'):
        trace.make_sidecar('a' * 64, [200, 100], {'model': 'fixture'}, [], [], {},
                           trace.TraceLimits(max_total_bytes=1))
    row = trace.compact_path(probabilities([0, 1, 0], 2), ['blank', '('],
                             geometry(batch_width=240), trace.TraceLimits())
    entry = {'bbox': [10, 20, 110, 40], 'text': '('}
    value = trace.make_sidecar('a' * 64, [200, 100], {'model': 'fixture'}, [row], [entry], {},
                               trace.TraceLimits(max_total_bytes=700))
    assert not value['rows'] and value['stats']['sidecar_row_byte_cap'] is True
    assert len(trace.encoded(value)) <= 700
