"""Independent pixel controls; these tests do not measure recognition quality."""
import numpy as np
import pytest

from tools.RegionKaomoji import bracket_profiles as bp


def square(height, width, stem_width=1, cap_rows=1):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :stem_width] = 1
    mask[:cap_rows, :] = mask[-cap_rows:, :] = 1
    return mask


def stroke(centers):
    """One-pixel-wide connected path with an explicitly measured center per row."""
    mask = np.zeros((len(centers), max(centers) + 1), dtype=np.uint8)
    mask[np.arange(len(centers)), centers] = 1
    return mask


@pytest.mark.parametrize('height,width,stem_width,cap_rows', [
    (8, 3, 1, 1), (12, 3, 2, 1), (16, 5, 1, 1),
    (23, 6, 3, 2), (32, 8, 3, 2),
])
def test_square_caps_have_same_row_witnesses_and_mirror_symmetry(height, width, stem_width, cap_rows):
    mask = square(height, width, stem_width, cap_rows)
    for right, local in ((False, mask), (True, np.fliplr(mask))):
        result = bp.classify(local)
        assert result is not None and result['kind'] == 'right-angle'
        assert result['direction'] == ('right' if right else 'left')
        measurements = result['measurements']
        stem = measurements['stem_column']
        assert measurements['stem_coverage'] == 1.0
        for side in ('top', 'bottom'):
            row = measurements[f'{side}_cap_witness_row']
            extension = measurements[f'{side}_cap_extension']
            assert extension >= measurements['required_cap_extension']
            xs = slice(stem-extension, stem+1) if right else slice(stem, stem+extension+1)
            assert local[row, xs].all()


@pytest.mark.parametrize('centers', [
    [2, 1, 1, 0, 0, 0, 0, 1, 1, 2],
    [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3],
    [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3],
])
def test_curves_do_not_turn_cross_row_projection_into_caps(centers):
    mask = stroke(centers)
    for right, local in ((False, mask), (True, np.fliplr(mask))):
        result = bp.classify(local)
        assert result is not None and result['kind'] in ('curve', 'small-curve')
        assert result['direction'] == ('right' if right else 'left')
        assert result['measurements']['top_endpoint_displacement'] >= .75
        assert result['measurements']['bottom_endpoint_displacement'] >= .75
        assert len(result['measurements']['row_center_profile']) == len(centers)


def test_brace_needs_a_localized_central_point_and_stable_shoulders():
    mask = stroke([4, 3, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2, 3, 4])
    for right, local in ((False, mask), (True, np.fliplr(mask))):
        result = bp.classify(local)
        assert result is not None and result['kind'] == 'brace'
        assert result['direction'] == ('right' if right else 'left')
        assert result['measurements']['central_point_displacement'] >= 1
        assert result['measurements']['upper_shoulder_range'] <= .5
        assert result['measurements']['lower_shoulder_range'] <= .5


def test_missing_caps_and_closed_frames_are_not_classified():
    missing = square(24, 6, 2, 2)
    missing[-2:, 2:] = 0
    closed = square(24, 6, 1, 1)
    closed[:, -1] = 1
    for local in (missing, closed):
        assert bp.classify(local) is None
        assert bp.classify(np.fliplr(local)) is None


@pytest.mark.parametrize('local', [
    np.zeros((16, 4), dtype=np.uint8),
    np.ones((16, 4), dtype=np.uint8),
    stroke([3, 2, 1, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 1, 2, 3]),
    # A pointed V has neither a stable round center nor two stable brace shoulders.
    stroke([4, 3, 3, 2, 2, 1, 1, 0, 1, 1, 2, 2, 3, 3, 4]),
])
def test_ambiguous_or_non_bracket_masks_do_not_manufacture_shapes(local):
    assert bp.classify(local) is None
    assert bp.classify(np.fliplr(local)) is None


def test_missing_rows_and_multiple_rails_fail_closed():
    local = square(16, 5)
    local[8] = 0
    assert bp.classify(local) is None
    local = stroke([2, 1, 0, 0, 0, 0, 0, 0, 1, 2])
    local[4, -1] = 1
    assert bp.classify(local) is None


def test_rounding_spur_cannot_supply_an_independent_brace_point():
    # Two-pixel quantization can leave a tiny spur in an ordinary round arc.
    # End bending dominates that spur; it is not enough to declare a brace.
    local = np.zeros((22, 5), dtype=np.uint8)
    starts = [0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 0, 0]
    widths = [2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    for row, (start, width) in enumerate(zip(starts, widths)):
        local[row, start:start+width] = 1
    for mask in (local, np.fliplr(local)):
        result = bp.classify(mask)
        assert result is None or result['kind'] in ('curve', 'small-curve')


def test_independent_pinned_font_shapes_do_not_cross_families():
    from pathlib import Path
    from tools.PretrainedOcr.data import MixedRenderer
    from tools.RegionKaomoji import candidates

    fonts = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'
    if not fonts.exists():
        pytest.skip('Pinned local OFL fonts absent; font control not measured')
    renderer = MixedRenderer(fonts)
    detected = 0
    for pair, family in (('()', 'curve'), ('[]', 'square'), ('{}', 'brace'),
                         ('\uff08\uff09', 'curve'), ('\uff3b\uff3d', 'square')):
        for index, glyph in enumerate(pair):
            for size in range(12, 33):
                image = np.asarray(renderer.render(glyph, size=size)[0].convert('RGB'))
                mask = candidates.foreground(image, 80, 51)
                component = max(candidates._components(mask, 2), key=lambda item: item['pixels'])
                left, top, right, bottom = component['bbox']
                result = bp.classify(mask[top:bottom, left:right])
                if result is None:
                    continue
                actual = {'curve': 'curve', 'small-curve': 'curve', 'right-angle': 'square', 'brace': 'brace'}[result['kind']]
                assert actual == family, (hex(ord(glyph)), size, result)
                assert result['direction'] == ('left' if index == 0 else 'right')
                detected += 1
    # Keep the independent control useful if a later change rejects everything.
    # This is a mechanics check, not a dataset quality/recognition threshold.
    assert detected > 105
