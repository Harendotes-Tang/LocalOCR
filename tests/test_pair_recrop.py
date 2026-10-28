"""Original-pixel recrop controls; these do not measure OCR accuracy."""
import copy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools.RegionKaomoji import candidates as c


def fixture(arm=True, body=True):
    image = Image.new('RGB', (260, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.arc((40, 20, 54, 50), 90, 270, fill='black', width=2)
    draw.arc((108, 20, 122, 50), 270, 90, fill='black', width=2)
    draw.ellipse((63, 31, 70, 38), outline='black', width=2)
    draw.ellipse((87, 31, 94, 38), outline='black', width=2)
    draw.line((75, 43, 83, 43), fill='black', width=2)
    if arm:
        draw.line((138, 28, 148, 44), fill='black', width=2)
    if body:
        draw.rectangle((154, 22, 180, 48), fill='black')
    return np.asarray(image)


def options():
    return dict(c.DEFAULT_CONFIG, pair_recrop={'maximum_gap_height': .58, 'maximum_exterior_cells': 2})


def pair_candidates(result):
    return [item for item in result['candidates'] if 'image-bracket-pair' in item['sources']]


def test_recrop_preserves_far_arm_and_stops_before_dense_body():
    image = fixture()
    old = pair_candidates(c.locate_candidates(image, []))
    new = pair_candidates(c.locate_candidates(image, [], options()))
    assert old and old[0]['bbox'][2] < 138
    assert len(new) == 1 and 149 <= new[0]['bbox'][2] < 154
    assert not new[0]['evidence']['edge_ink']
    assert new[0]['evidence']['pair_recrop']['changed']


def test_recrop_keeps_body_context_and_pixel_proof_current():
    image = fixture()
    baseline = [{'bbox': [36, 16, 184, 54], 'text': '(ω_ω)ノ正文', 'score': .1}]
    result = c.locate_candidates(image, baseline, options())
    faces = pair_candidates(result)
    assert len(faces) == 1
    candidate = faces[0]
    assert candidate['evidence']['neighbor_body_overlap']
    assert candidate['baseline_indices'] == []
    assert candidate['evidence']['mixed_region_separable']
    fresh = c.pixel_evidence(image, candidate['bbox'], options())
    assert fresh['paired_bracket_shapes'] == candidate['evidence']['paired_bracket_shapes']
    assert fresh['pixel_count'] == candidate['evidence']['pixel_count']


def test_recrop_does_not_invent_arms_or_slice_dense_components():
    image = fixture(arm=False)
    result = pair_candidates(c.locate_candidates(image, [], options()))
    assert len(result) == 1 and result[0]['bbox'][2] < 154
    # Its original-pixel envelope contains either every pixel of a component or none.
    config = c.configuration(options())
    mask, _ = c._masks(image, config)
    box = result[0]['evidence']['ink_bbox']
    for component in c._components(mask, 2):
        b = component['bbox']
        overlap = c.intersection(box, b)
        assert not overlap or overlap == c.area(b)


def test_recrop_is_opt_in_and_bounded():
    assert 'pair_recrop' not in c.configuration()
    for value in (None, True, {}, {'maximum_gap_height': .9, 'maximum_exterior_cells': 2},
                  {'maximum_gap_height': .58, 'maximum_exterior_cells': True},
                  {'maximum_gap_height': .58, 'maximum_exterior_cells': 20}):
        with pytest.raises(ValueError):
            c.configuration(dict(c.DEFAULT_CONFIG, pair_recrop=value))


def test_recrop_does_not_create_parallel_variants_for_one_anchor():
    first = c.locate_candidates(fixture(body=False), [], options())
    assert len(pair_candidates(first)) == 1
    assert len({tuple(item['bbox']) for item in first['candidates']}) == len(first['candidates'])
    assert first == c.locate_candidates(fixture(body=False), [], copy.deepcopy(options()))


def test_recrop_does_not_cut_a_component_from_another_physical_line():
    image = Image.fromarray(fixture())
    ImageDraw.Draw(image).rectangle((147, 48, 152, 67), fill='black')
    result = pair_candidates(c.locate_candidates(np.asarray(image), [], options()))
    assert len(result) == 1
    assert result[0]['evidence']['ink_bbox'][2] < 147
    assert not result[0]['evidence']['pair_recrop']['changed']


def test_boundary_check_covers_diagonal_connectivity_and_hollow_frames():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[5, 5] = mask[6, 6] = 1
    assert c._crosses_component_boundary(mask, [3, 3, 6, 6])
    assert not c._crosses_component_boundary(mask, [3, 3, 8, 8])
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 1
    assert not c._crosses_component_boundary(mask, [3, 3, 8, 8])


def test_adjacent_faces_keep_two_anchors_without_absorbing_the_next_face():
    image = Image.new('RGB', (270, 80), 'white')
    draw = ImageDraw.Draw(image)
    for offset in (0, 84):
        draw.arc((40+offset, 20, 54+offset, 50), 90, 270, fill='black', width=2)
        draw.arc((98+offset, 20, 112+offset, 50), 270, 90, fill='black', width=2)
        draw.ellipse((58+offset, 31, 65+offset, 38), outline='black', width=2)
        draw.ellipse((85+offset, 31, 92+offset, 38), outline='black', width=2)
        draw.line((71+offset, 43, 77+offset, 43), fill='black', width=2)
    config = options()
    config['pair_selection'] = {'family_match': True, 'innermost_partner': True}
    faces = pair_candidates(c.locate_candidates(np.asarray(image), [], config))
    assert len(faces) == 2
    assert faces[0]['evidence']['ink_bbox'][2] < faces[1]['evidence']['ink_bbox'][0]
    assert all(item['route_eligible'] for item in faces)


def test_new_profile_cannot_disable_family_matching():
    for selection in (None, {'family_match': False, 'innermost_partner': True}):
        config = dict(c.DEFAULT_CONFIG, bracket_profile='row-contiguous-v1')
        if selection is not None:
            config['pair_selection'] = selection
        with pytest.raises(ValueError, match='family matching'):
            c.configuration(config)
        with pytest.raises(ValueError, match='family matching'):
            c.pixel_evidence(fixture(), [35, 15, 185, 55], config)


def test_recrop_keeps_low_mark_guard_on_rendered_punctuation():
    from pathlib import Path
    from tools.PretrainedOcr.data import MixedRenderer
    fonts = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'
    if not fonts.exists():
        pytest.skip('Pinned local OFL font absent')
    renderer = MixedRenderer(fonts)
    image = np.asarray(renderer.render('ヽ(^_^)ノ、', size=16)[0].convert('RGB'))
    config = dict(c.DEFAULT_CONFIG, shape_contrast_threshold=80,
                  low_mark_guard={'maximum_height_ratio': .45, 'minimum_top_ratio': .5})
    old = pair_candidates(c.locate_candidates(image, [], config))
    new = pair_candidates(c.locate_candidates(image, [], {**config, 'pair_recrop': options()['pair_recrop']}))
    assert old and new
    assert max(p['bbox'][2] for p in new) <= max(p['bbox'][2] for p in old)
