"""Pixel partition mechanics; these fixtures make no OCR-quality claim."""
import copy

import numpy as np
import pytest

from tools.RegionKaomoji import local_ownership as ownership


def fixture():
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[12:26, 10:18] = 1  # Left body.
    mask[12:26, 35:39] = 1
    mask[17:22, 44:48] = 1
    mask[12:26, 53:57] = 1  # Complete face-like stroke group.
    mask[12:26, 74:82] = 1  # Right body.
    entries = [{'index': 0, 'bbox': [7, 9, 85, 29],
                'text': '左(oo)右', 'score': .5}]
    return mask, [30, 9, 62, 29], entries


def prove(mask, box, entries, owned=(), neighbors=(0,), **config):
    return ownership.prove_local_partition(mask, box, entries, owned, neighbors, **config)


def test_partitions_all_remaining_ink_into_ordered_contexts():
    mask, box, entries = fixture()
    result = prove(mask, box, entries)
    assert result['parent_index'] == 0
    assert [crop['role'] for crop in result['context_crops']] == ['left', 'right']
    assert sum(crop['visible_pixels'] for crop in result['context_crops']) == 224
    assert ownership.audit_local_partition(result, mask, box, entries, (), (0,))


def test_roi_may_extend_past_parent_to_capture_whole_arm_component():
    mask, box, entries = fixture()
    mask[:, 74:] = 0
    entries[0]['bbox'][2] = 50  # Parent stops at a blank band before a complete final stroke.
    result = prove(mask, box, entries)
    assert result is not None
    assert result['proof']['support_bbox'] == [7, 9, 62, 29]
    assert [crop['role'] for crop in result['context_crops']] == ['left']


@pytest.mark.parametrize('width,height', [(1, 1), (2, 4), (4, 7)])
def test_detached_punctuation_letters_and_digits_are_kept_in_context(width, height):
    mask, box, entries = fixture()
    mask[22:22 + height, 66:66 + width] = 1
    result = prove(mask, box, entries)
    assert result is not None
    right = result['context_crops'][1]
    assert right['bbox'][0] <= 66
    assert right['visible_pixels'] == 112 + width * height


def test_component_cut_by_roi_is_rejected_including_diagonal_bridge():
    mask, box, entries = fixture()
    mask[15, 29] = mask[16, 30] = 1
    assert prove(mask, box, entries) is None


def test_component_cut_by_parent_is_rejected_even_inside_roi():
    mask, box, entries = fixture()
    entries[0]['bbox'][2] = 55
    assert prove(mask, box, entries) is None


def test_ink_on_other_line_above_roi_is_not_discarded():
    mask, box, entries = fixture()
    entries[0]['bbox'][1] = 1
    mask[3:6, 40:43] = 1
    assert prove(mask, box, entries) is None


def test_other_line_to_side_does_not_become_same_line_context():
    mask, box, entries = fixture()
    entries[0]['bbox'][1] = 1
    box[1] = 1
    mask[3:6, 22:25] = 1
    assert prove(mask, box, entries) is None


def test_owned_single_parent_is_allowed_without_dropping_context_ink():
    mask, box, entries = fixture()
    result = prove(mask, box, entries, owned=(0,), neighbors=())
    assert result is not None and len(result['context_crops']) == 2


def test_same_parent_reference_in_owned_and_neighbor_lists_is_deduplicated():
    mask, box, entries = fixture()
    assert prove(mask, box, entries, owned=(0,), neighbors=(0,)) is not None


def test_owned_plus_distinct_neighbor_parent_is_ambiguous():
    mask, box, entries = fixture()
    entries.append({'index': 1, 'bbox': [32, 10, 60, 28], 'text': '(oo)', 'score': .4})
    assert prove(mask, box, entries, owned=(1,), neighbors=(0,)) is None


def test_omitted_overlapping_parent_cannot_manufacture_uniqueness():
    mask, box, entries = fixture()
    entries.append({'index': 1, 'bbox': [32, 10, 60, 28], 'text': '(oo)', 'score': .4})
    assert prove(mask, box, entries) is None


def test_ambiguous_context_envelope_is_rejected():
    mask, box, entries = fixture()
    entries.append({'index': 1, 'bbox': [7, 9, 20, 29], 'text': '左', 'score': .8})
    assert prove(mask, box, entries) is None


def test_too_small_blank_band_rejects_instead_of_absorbing_token():
    mask, box, entries = fixture()
    mask[16:20, 63:66] = 1
    mask[16:20, 61] = 1
    assert prove(mask, box, entries) is None


def test_empty_remainder_cannot_create_a_character_mapping():
    mask, box, entries = fixture()
    mask[:, :30] = mask[:, 62:] = 0
    assert prove(mask, box, entries) is None


@pytest.mark.parametrize('kind', ['mask', 'box', 'parent_text', 'parent_box', 'context', 'digest', 'config'])
def test_audit_rejects_changed_inputs_or_receipt(kind):
    mask, box, entries = fixture()
    result = prove(mask, box, entries)
    altered = copy.deepcopy(result)
    config = {}
    if kind == 'mask':
        mask[0, 0] = 1  # Full original mask identity, even outside the support envelope.
    elif kind == 'box':
        box[0] -= 1
    elif kind == 'parent_text':
        entries[0]['text'] += '。'
    elif kind == 'parent_box':
        entries[0]['bbox'][0] -= 1
    elif kind == 'context':
        altered['context_crops'][0]['visible_pixels'] -= 1
    elif kind == 'digest':
        altered['proof']['sha256'] = '0' * 64
    elif kind == 'config':
        config['padding'] = 2
    assert not ownership.audit_local_partition(altered, mask, box, entries, (), (0,), **config)


@pytest.mark.parametrize('config', [dict(padding=True), dict(padding=-1),
                                  dict(minimum_blank_band=0),
                                  dict(minimum_blank_band=1),
                                  dict(maximum_parent_height_ratio=float('nan'))])
def test_malformed_config_is_rejected(config):
    mask, box, entries = fixture()
    with pytest.raises(ValueError):
        prove(mask, box, entries, **config)


def test_unknown_roi_ink_does_not_acquire_a_text_replacement_proof():
    mask, box, entries = fixture()
    entries[0]['bbox'][2] = 50
    # ROI extends beyond the detector parent and includes another isolated mark.
    box[2] = 72
    mask[:, 74:] = 0
    mask[11, 67] = 1
    result = prove(mask, box, entries)
    assert result is not None
    # An extra component inside ROI is deliberately NOT called safe face ink by
    # this module. Existing recognition/independent-body guards remain required.
    assert result['proof']['roi_visible_pixels'] == int(mask[box[1]:box[3], box[0]:box[2]].sum())
    assert 'text_span' not in result['proof'] and 'safe_face' not in result['proof']


def test_parent_digest_matches_existing_raw_parent_contract():
    from tools.RegionKaomoji.candidates import baseline_parent_digest

    mask, box, entries = fixture()
    assert prove(mask, box, entries)['proof']['parent_sha256'] == baseline_parent_digest(entries[0])


def test_receipt_is_stable_and_does_not_mutate_inputs():
    mask, box, entries = fixture()
    original_mask, original_box, original_entries = mask.copy(), box[:], copy.deepcopy(entries)
    first = prove(mask, box, entries)
    assert first == prove(mask.astype(bool), box, entries)
    assert np.array_equal(mask, original_mask)
    assert box == original_box and entries == original_entries
