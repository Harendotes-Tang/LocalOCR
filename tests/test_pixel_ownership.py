"""Original-pixel ownership controls; declared readings are not OCR accuracy."""
import copy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools.RegionKaomoji import candidates as c
from tools.RegionKaomoji import pixel_ownership as p


def image_fixture(*, arm=True, body=False, nested=False):
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
    if nested:
        draw.arc((20, 15, 34, 55), 90, 270, fill='black', width=2)
        draw.arc((126, 15, 140, 55), 270, 90, fill='black', width=2)
    return np.asarray(image)


def candidate(image, identity, box, ink_box, text, *, novel=False, config=None):
    evidence = c.pixel_evidence(image, box, config)
    evidence.update(ink_bbox=ink_box, neighbor_body_overlap=False,
                    clipped_canvas=False, mixed_region_separable=False)
    return {'candidate_id': identity, 'bbox': box,
            'sources': ['image-line-span'] if novel else ['image-bracket-pair'],
            'baseline_indices': [], 'splice_baseline_index': None,
            'evidence': evidence, 'specialist_text': text}


def pair(image=None, *, novel_complete=False):
    image = image_fixture() if image is None else image
    full = candidate(image, 'original' if not novel_complete else 'new-full',
                     [37, 17, 153, 54], [40, 20, 150, 51], '(ω_ω)ノ', novel=novel_complete)
    short = candidate(image, 'new-short' if not novel_complete else 'original',
                      [37, 17, 126, 54], [40, 20, 123, 51], '(ω_ω)', novel=not novel_complete)
    return image, full, short


def test_complete_arm_owns_shared_core_once_independent_of_order():
    image, full, short = pair()
    for order in ([full, short], [short, full]):
        proof = p.resolve_conflicts(image, order)
        assert set(proof['suppressed']) == {'new-short'}
        record = proof['suppressed']['new-short']
        assert record['winner'] == 'original'
        assert record['reason'] == 'pixel-owned-complete-expression'
        assert record['shared_core_pixels'] > 0 and record['same_bracket_anchors']
        assert record['original_image_sha256'] == c.image_digest(image)


def test_proved_complete_novel_can_keep_arm_over_original_short():
    image, full, short = pair(novel_complete=True)
    proof = p.resolve_conflicts(image, [short, full])
    assert set(proof['suppressed']) == {'original'}
    assert proof['suppressed']['original']['winner'] == 'new-full'
    assert proof['suppressed']['original']['suppressed_candidate_id'] == 'original'


def test_faint_complete_exterior_arm_uses_base_pixels_without_core_erasure():
    image = Image.fromarray(image_fixture(arm=False))
    ImageDraw.Draw(image).line((138, 28, 148, 44), fill=(180, 180, 180), width=2)
    image = np.asarray(image)
    config = {**c.DEFAULT_CONFIG, 'shape_contrast_threshold': 80}
    full = candidate(image, 'new-full', [37, 17, 153, 54], [40, 20, 150, 51], '(ω_ω)ノ', novel=True, config=config)
    short = candidate(image, 'original', [37, 17, 126, 54], [40, 20, 123, 51], '(ω_ω)', config=config)
    assert not full['evidence']['pair_geometry']['exterior_ink']['right']
    proof = p.resolve_conflicts(image, [short, full], config)
    assert proof['suppressed']['original']['winner'] == 'new-full'
    assert proof['candidate_evidence']['new-full']['base_exterior_ink']['right']


def test_same_pixels_without_safe_arm_reading_preserve_original():
    image, full, short = pair(novel_complete=True)
    full['specialist_text'] = '(ω_ω)'
    proof = p.resolve_conflicts(image, [full, short])
    assert set(proof['suppressed']) == {'new-full'}
    assert proof['suppressed']['new-full']['reason'] == 'pixel-owned-ambiguous-preserve-original'


def test_identical_boxes_and_strings_without_actual_ink_do_not_claim_pixels():
    image = np.full((80, 260, 3), 255, dtype=np.uint8)
    first = candidate(image, 'first', [30, 15, 155, 55], [35, 20, 150, 50], '(ω_ω)')
    second = {**copy.deepcopy(first), 'candidate_id': 'second', 'sources': ['image-line-span']}
    proof = p.resolve_conflicts(image, [first, second])
    assert proof['suppressed'] == {} and proof['conflicts'] == []


def test_equal_strings_and_overlapping_padding_do_not_merge_disjoint_components():
    image = Image.new('RGB', (260, 80), 'white')
    draw = ImageDraw.Draw(image)
    for x in (30, 125):
        draw.arc((x, 20, x+14, 50), 90, 270, fill='black', width=2)
        draw.arc((x+65, 20, x+79, 50), 270, 90, fill='black', width=2)
        draw.ellipse((x+24, 30, x+30, 37), outline='black', width=2)
        draw.ellipse((x+46, 30, x+52, 37), outline='black', width=2)
        draw.line((x+34, 43, x+42, 43), fill='black', width=2)
    image = np.asarray(image)
    first = candidate(image, 'first', [27, 17, 125, 54], [30, 20, 110, 51], '(ω_ω)')
    second = candidate(image, 'second', [112, 17, 208, 54], [125, 20, 205, 51], '(ω_ω)', novel=True)
    assert c.intersection(first['bbox'], second['bbox']) > 0
    proof = p.resolve_conflicts(image, [first, second])
    assert proof['suppressed'] == {} and proof['conflicts'] == []


def test_nested_brackets_preserve_original_instead_of_preferring_bigger_box():
    image = image_fixture(arm=False, nested=True)
    original = candidate(image, 'original', [37, 17, 126, 54], [40, 20, 123, 51], '(ω_ω)')
    outer = candidate(image, 'outer', [17, 12, 144, 59], [20, 15, 141, 56], '(ω_ω)', novel=True)
    proof = p.resolve_conflicts(image, [outer, original])
    assert set(proof['suppressed']) == {'outer'}
    assert proof['suppressed']['outer']['winner'] == 'original'
    assert proof['suppressed']['outer']['reason'] == 'pixel-owned-ambiguous-preserve-original'


def test_dense_body_in_larger_candidate_never_becomes_complete_arm_proof():
    image = image_fixture(body=True)
    _, _, original = pair(image, novel_complete=True)
    large = candidate(image, 'body-expanded', [37, 17, 184, 54], [40, 20, 181, 51], '(ω_ω)ノ', novel=True)
    proof = p.resolve_conflicts(image, [large, original])
    assert set(proof['suppressed']) == {'body-expanded'}
    assert proof['suppressed']['body-expanded']['winner'] == 'original'


def test_cut_core_component_cannot_supply_complete_dominance():
    image, _, original = pair(novel_complete=True)
    cut = candidate(image, 'cut-arm', [37, 17, 145, 54], [40, 20, 144, 51], '(ω_ω)ノ', novel=True)
    proof = p.resolve_conflicts(image, [cut, original])
    assert set(proof['suppressed']) == {'cut-arm'}
    assert not proof['candidate_evidence']['cut-arm']['complete_components']


def test_geometry_forgery_is_not_trusted():
    image, full, short = pair()
    full['evidence']['pair_geometry']['right_bbox'] = [154, 20, 181, 49]
    with pytest.raises(ValueError, match='original pixels'):
        p.resolve_conflicts(image, [full, short])


@pytest.mark.parametrize('value', [None, {}, True, {'strategy': 'rectangles'},
                                  {'strategy': 'component-exclusion-v1', 'threshold': 0}])
def test_invalid_options_are_rejected(value):
    if value is None:
        assert p.configuration() == {'strategy': 'component-exclusion-v1'}
    else:
        with pytest.raises(ValueError):
            p.configuration(value)


def test_input_complexity_and_parentless_contract_are_bounded():
    image, full, short = pair()
    with pytest.raises(ValueError, match='bounded'):
        p.resolve_conflicts(image, [full]*49)
    with pytest.raises(ValueError, match='identity'):
        p.resolve_conflicts(image, [full, copy.deepcopy(full)])
    full['baseline_indices'] = [0]
    with pytest.raises(ValueError, match='parentless'):
        p.resolve_conflicts(image, [full, short])


def test_original_pixels_and_callers_are_unchanged_and_proof_is_deterministic():
    image, full, short = pair()
    before = image.copy(), copy.deepcopy([full, short])
    actual = p.resolve_conflicts(image, [full, short])
    assert actual == p.resolve_conflicts(image, [full, short])
    assert np.array_equal(image, before[0]) and [full, short] == before[1]
