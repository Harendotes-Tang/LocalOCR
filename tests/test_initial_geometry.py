"""First-pass transform fixtures; no model sessions or quality claims."""
import copy

import numpy as np
import pytest

from tools.RegionKaomoji import ctc_trace as ct, position_spans
from tools.RegionKaomoji.initial_trace import geometry_for_crop, horizontal_extent


def geometry():
    result = geometry_for_crop([[20, 30], [120, 30], [120, 50], [20, 50]], [20, 100],
        {'preprocess': {'ratio_h': 2., 'ratio_w': 2.}, 'padding_1': {'top': 10, 'left': 5}}, [400, 200])
    result.update(detector_index=3, source_indices=[3], input_index=3, sorted_index=0,
                  batch_index=0, batch_ordinal=0, resize_height=48, content_width=240,
                  batch_width=480, before_text='', classification={'enabled': False, 'rotated': False})
    return result


def test_native_resize_and_detector_padding_are_inverted_before_original_hint():
    value = geometry()
    assert value['crop_bbox'] == [30, 40, 230, 80]
    assert horizontal_extent(value, 25, 50) == pytest.approx((80, 130))
    prediction = np.zeros((8, 2), np.float32)
    prediction[:, 0] = .9
    prediction[1] = [0, .95]
    path = ct.compact_path(prediction, ['blank', '('], value)
    assert path['tokens'][0]['x_start'] == pytest.approx(80)
    assert path['tokens'][0]['x_end'] == pytest.approx(130)


def test_mild_perspective_uses_full_strip_not_bounding_rectangle_interpolation():
    quad = np.asarray([[20, 30], [120, 32], [118, 52], [22, 50]], np.float32)
    width = int(max(np.linalg.norm(quad[0]-quad[1]), np.linalg.norm(quad[2]-quad[3])))
    height = int(max(np.linalg.norm(quad[0]-quad[3]), np.linalg.norm(quad[1]-quad[2])))
    value = geometry_for_crop(quad, [height, width],
        {'preprocess': {'ratio_h': 1., 'ratio_w': 1.}, 'padding_1': {'top': 0, 'left': 0}}, [200, 100])
    low, high = horizontal_extent(value, 0, 25)
    assert low == pytest.approx(20.)
    assert high > 45.  # Bottom of this perspective strip is farther right.


@pytest.mark.parametrize('field,value', [('inverse_perspective', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ('crop_bbox', [30, 40, 231, 80]), ('rotation', 180), ('classification', {'rotated': True}),
    ('detector_index', None), ('original_size', [False, 200])])
def test_forged_geometry_is_rejected(field, value):
    record = geometry()
    record[field] = value
    with pytest.raises((ct.UnsupportedTrace, ValueError)):
        ct._geometry(record)


@pytest.mark.parametrize('quad,shape', [
    ([[20, 30], [30, 30], [30, 80], [20, 80]], [50, 10]),
    ([[20, 30], [120, 80], [120, 100], [20, 50]], [20, 111]),
    ([[20, 30], [120, 30], [120, 30], [20, 30]], [0, 100]),
    ([[120, 30], [20, 30], [20, 50], [120, 50]], [20, 100]),
    ([[-1, 30], [100, 30], [100, 50], [-1, 50]], [20, 101]),
])
def test_rotation_singular_reversal_and_clipping_are_unsupported(quad, shape):
    with pytest.raises(ct.UnsupportedTrace):
        geometry_for_crop(quad, shape,
            {'preprocess': {'ratio_h': 1., 'ratio_w': 1.}, 'padding_1': {'top': 0, 'left': 0}}, [400, 200])


def test_initial_sidecar_cannot_be_rebound_by_same_text_and_rectangle():
    value = geometry()
    prediction = np.zeros((8, 2), np.float32)
    prediction[:, 0] = .9
    prediction[1] = [0, .95]
    path = ct.compact_path(prediction, ['blank', '('], value)
    entries = [{'bbox': [30, 40, 230, 80], 'text': '(', 'score': .95}]
    assert ct.make_sidecar('a'*64, [400, 200], {}, [path], entries, {})['rows'] == []
    path['geometry'].update(final_parent_index=0, selected_parent={'index': 0, **entries[0]})
    sidecar = ct.make_sidecar('a'*64, [400, 200], {}, [path], entries, {})
    assert len(ct.validate_sidecar('a'*64, [400, 200], entries, sidecar)) == 1
    changed = copy.deepcopy(entries)
    changed[0]['score'] = .8
    assert ct.validate_sidecar('a'*64, [400, 200], changed, sidecar) == []


def test_optional_config_preserves_prior_defaults_and_only_enables_ctc_modes():
    old = {'strategy': 'ctc-components-v1', 'mode': 'ctc'}
    assert position_spans.configuration(old) == old
    assert position_spans.configuration({**old, 'initial_trace': 'horizontal-v1'})['initial_trace'] == 'horizontal-v1'
    for change in ({'initial_trace': None}, {'initial_trace': True},
                   {'initial_trace': 'horizontal-v1', 'mode': 'components'}):
        with pytest.raises(ValueError):
            position_spans.configuration({**old, **change})
