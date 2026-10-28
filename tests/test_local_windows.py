"""Original-pixel local-window controls, without OCR or accuracy claims."""
import copy

import numpy as np
import pytest

from tools.RegionKaomoji import local_windows as windows


def bracket(mask, x, y=16, right=False):
    local = np.zeros((20, 5), dtype=np.uint8)
    local[:, 0] = local[0, :] = local[-1, :] = 1
    mask[y:y+20, x:x+5] |= np.fliplr(local) if right else local


def fixture(*, weak=False, bridge=False, arm=False):
    core = np.zeros((60, 120), dtype=np.uint8)
    core[16:36, 4:17] = core[16:36, 77:91] = 1
    bracket(core, 25)
    bracket(core, 65, right=True)
    core[24:27, 35:38] = core[24:27, 53:56] = 1
    core[32:34, 43:49] = 1
    if arm:
        core[27:30, 19:22] = 1
        core[27:30, 73:75] = 1
    base = core.copy()
    if weak:
        base[10:12, 35:37] = 1
        base[27:29, 20:23] = 1
    if bridge:
        base[26, 15:79] = 1
    return base, core, [{'bbox': [4, 9 if weak else 16, 91, 36]}]


def run(base, core, lines, *, seeds=(), entries=(), limits=None, options=None):
    return windows.propose_windows(base, core, lines, entries, seeds,
        limits=windows.DEFAULT_LIMITS if limits is None else limits,
        options=windows.DEFAULT_OPTIONS if options is None else options)


def test_actual_brackets_anchor_complete_local_window_and_no_text_is_needed():
    base, core, lines = fixture()
    result = run(base, core, lines)
    face = next(p for p in result['proposals'] if p['ink_bbox'] == [25, 16, 70, 36])
    assert face['sources'] == ['image-local-window']
    evidence = face['local_window']
    assert evidence['bracket_strokes'] == [[25, 16, 30, 36], [65, 16, 70, 36]]
    assert len(evidence['anchor_core_component_ids']) == 2
    assert evidence['body_safe'] is False
    assert evidence['original_pixels_required'] is True


def test_complete_core_outside_arms_are_first_hypothesis_and_neighbors_bound_padding():
    base, core, lines = fixture(arm=True)
    result = run(base, core, lines)
    first = result['proposals'][0]
    assert first['ink_bbox'] == [19, 16, 75, 36]
    assert first['local_window']['crop_limits'][0] >= 18
    assert first['local_window']['crop_limits'][2] <= 76
    assert all(p['ink_bbox'][0] > 17 and p['ink_bbox'][2] < 77 for p in result['proposals'])


def test_weak_point_and_exterior_arm_are_recovered_whole_from_original_base():
    base, core, lines = fixture(weak=True)
    face = run(base, core, lines)['proposals'][0]
    assert face['ink_bbox'] == [20, 10, 70, 36]
    assert face['local_window']['base_only_pixels'] == 10
    assert len(face['local_window']['base_only_component_ids']) == 2


def test_weak_arm_tied_with_body_is_declined_instead_of_ignoring_its_component():
    base, core, lines = fixture()
    base[27:29, 19:23] = 1
    result = run(base, core, lines)
    assert not result['proposals']
    assert result['rejections']['ambiguous-base-only-ownership']


def test_halo_bridge_keeps_core_whole_and_records_crossings_without_mutating_pixels():
    base, core, lines = fixture(bridge=True)
    before = base.copy(), core.copy(), copy.deepcopy(lines)
    result = run(base, core, lines)
    assert result['proposals']
    evidence = result['proposals'][0]['local_window']
    assert evidence['base_bridge']['present']
    assert evidence['base_bridge']['crossing_pixels'] > 0
    assert np.array_equal(base, before[0]) and np.array_equal(core, before[1])
    assert lines == before[2]


def test_core_crossing_the_local_window_boundary_is_declined():
    base, core, lines = fixture()
    # A branch of the actual left bracket passes above the supplied physical line.
    core[12:17, 25] = 1
    base |= core
    result = run(base, core, lines)
    assert not result['proposals']
    assert result['rejections']['anchor-crosses-physical-line']


def test_unrelated_core_crossing_physical_line_does_not_cancel_local_pair():
    base, core, lines = fixture()
    core[8:36, 4:17] = 1
    base |= core
    result = run(base, core, lines)
    assert [25, 16, 70, 36] in [p['ink_bbox'] for p in result['proposals']]


def test_neighbor_faces_remain_independent_in_one_long_sparse_line():
    core = np.zeros((60, 180), dtype=np.uint8)
    for x in (10, 64):
        bracket(core, x)
        bracket(core, x+45, right=True)
        core[25:28, x+15:x+18] = 1
        core[25:28, x+30:x+33] = 1
    result = run(core.copy(), core, [{'bbox': [10, 16, 114, 36]}])
    boxes = [p['ink_bbox'] for p in result['proposals']]
    assert [10, 16, 60, 36] in boxes and [64, 16, 114, 36] in boxes
    assert all(b[2] <= 64 or b[0] >= 60 for b in boxes)


def test_nested_actual_pairs_can_propose_inner_and_complete_outer_windows():
    core = np.zeros((60, 120), dtype=np.uint8)
    for x, right in ((10, False), (25, False), (65, True), (80, True)):
        bracket(core, x, right=right)
    core[25:28, 35:38] = core[25:28, 53:56] = 1
    result = run(core.copy(), core, [{'bbox': [10, 16, 85, 36]}])
    boxes = [p['ink_bbox'] for p in result['proposals']]
    assert [10, 16, 85, 36] in boxes and [25, 16, 70, 36] in boxes


def test_seed_pair_metadata_cannot_manufacture_actual_brackets():
    core = np.zeros((60, 120), dtype=np.uint8)
    core[16:36, 25:27] = core[16:36, 65:67] = 1
    core[25:28, 35:38] = core[25:28, 53:56] = 1
    seed = {'ink_bbox': [25, 16, 67, 36],
            'pair': {'strokes': [[25, 16, 27, 36], [65, 16, 67, 36]]}}
    assert run(core.copy(), core, [{'bbox': [25, 16, 67, 36]}], seeds=[seed])['proposals'] == []


def test_local_neighbors_do_not_expand_to_arbitrary_run_endpoints():
    core = np.zeros((60, 220), dtype=np.uint8)
    bracket(core, 90)
    bracket(core, 135, right=True)
    core[25:28, 105:108] = core[25:28, 120:123] = 1
    for x in (20, 30, 40, 50, 60, 70, 80, 150, 160, 170, 180, 190):
        core[27:29, x:x+2] = 1
    result = run(core.copy(), core, [{'bbox': [20, 16, 192, 36]}])
    assert result['proposals']
    assert all(p['ink_bbox'][0] >= 70 and p['ink_bbox'][2] <= 162 for p in result['proposals'])
    assert max(result['seed_hypothesis_counts']) <= 4


@pytest.mark.parametrize('limit', [{'maximum_candidates': 1}, {'maximum_crop_pixels': 900},
                                  {'maximum_crop_pixels': 901}])
def test_exhausted_shared_budget_stops_extra_envelope_work(monkeypatch, limit):
    base, core, lines = fixture()
    base = np.concatenate([base] * 20)
    core = np.concatenate([core] * 20)
    lines = [{'bbox': [4, 16+i*60, 91, 36+i*60]} for i in range(20)]
    calls = []
    original = windows._envelope

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(windows, '_envelope', measured)
    result = run(base, core, lines, limits={**windows.DEFAULT_LIMITS, **limit})
    assert len(result['proposals']) == 1
    assert len(calls) == 1
    assert result['rejections']['total-window-budget'] == 1


def test_initial_budget_too_small_skips_components_and_bracket_scans(monkeypatch):
    base, core, lines = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError('No component/pixel discovery is possible within the budget')

    monkeypatch.setattr(windows, '_components', forbidden)
    result = run(base, core, lines, limits={**windows.DEFAULT_LIMITS, 'maximum_crop_pixels': 6})
    assert result['proposals'] == []
    assert result['rejections']['total-window-budget'] == 1


def test_budget_stops_bracket_profile_discovery_on_later_lines(monkeypatch):
    base, core, _ = fixture()
    base = np.concatenate([base]*20)
    core = np.concatenate([core]*20)
    lines = [{'bbox': [4, 16+i*60, 91, 36+i*60]} for i in range(20)]
    calls = []
    original = windows._anchors

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(windows, '_anchors', measured)
    result = run(base, core, lines, limits={**windows.DEFAULT_LIMITS, 'maximum_candidates': 1})
    assert len(calls) == 1 and len(result['proposals']) == 1


def test_dense_body_inside_a_pair_is_not_declared_safe_or_proposed():
    base, core, lines = fixture()
    core[16:36, 40:55] = 1
    base |= core
    result = run(base, core, lines)
    assert result['proposals'] == []
    assert result['rejections']['dense-core-inside-brackets']


def test_explicit_real_seed_changes_order_only_and_does_not_consult_readings():
    core = np.zeros((60, 180), dtype=np.uint8)
    for x in (10, 64):
        bracket(core, x)
        bracket(core, x+45, right=True)
        core[25:28, x+15:x+18] = core[25:28, x+30:x+33] = 1
    lines = [{'bbox': [10, 16, 114, 36]}]
    seed = {'ink_bbox': [64, 16, 114, 36],
            'pair': {'strokes': [[64, 16, 69, 36], [109, 16, 114, 36]]}}
    result = run(core.copy(), core, lines, seeds=[seed])
    assert result['proposals'][0]['ink_bbox'] == [64, 16, 114, 36]
    assert result['proposals'][0]['local_window']['seed_kind'] == 'proposal'
    assert set(map(tuple, (p['ink_bbox'] for p in result['proposals']))) == {
        (10, 16, 60, 36), (64, 16, 114, 36)}


def test_non_dict_pair_metadata_is_ignored_and_never_manufactures_evidence():
    base, core, lines = fixture()
    assert run(base, core, lines, seeds=[{'ink_bbox': [25, 16, 70, 36], 'pair': True}]) == run(base, core, lines)


def test_global_added_candidate_ceiling_remains_48():
    base, core, _ = fixture()
    base = np.concatenate([base]*50)
    core = np.concatenate([core]*50)
    lines = [{'bbox': [4, 16+i*60, 91, 36+i*60]} for i in range(50)]
    result = run(base, core, lines)
    assert len(result['proposals']) == 48
    assert result['crop_pixels'] == 48*900
    assert result['rejections']['total-window-budget'] == 1


def test_large_window_can_be_skipped_then_later_small_pair_can_fit():
    base, core, lines = fixture()
    base = np.concatenate((base, np.zeros_like(base)))
    core = np.concatenate((core, np.zeros_like(core)))
    # A compact valid pair on another row (minimum span remains >=1.2*height).
    small = np.zeros((8, 3), dtype=np.uint8)
    small[:, 0] = small[0, :] = small[-1, :] = 1
    core[76:84, 10:13] = small
    core[76:84, 20:23] = np.fliplr(small)
    base |= core
    lines += [{'bbox': [10, 76, 23, 84]}]
    result = run(base, core, lines, limits={**windows.DEFAULT_LIMITS, 'maximum_crop_pixels': 110})
    assert [p['ink_bbox'] for p in result['proposals']] == [[10, 76, 23, 84]]


def test_geometry_does_not_read_baseline_text_or_seed_recognition():
    class Unreadable:
        def __str__(self):
            raise AssertionError('Recognition text must not be consulted')

    base, core, lines = fixture()
    first = run(base, core, lines, entries=[{'index': 4, 'bbox': [4, 16, 91, 36], 'text': Unreadable()}])
    second = run(base, core, lines, entries=[{'index': 4, 'bbox': [4, 16, 91, 36], 'text': 'changed'}])
    assert first == second
    assert first['proposals'][0]['local_window']['geometric_parent_indices'] == [4]


@pytest.mark.parametrize('updates', [None, True, {}, {'strategy': 'other'},
    {'maximum_neighbor_cells': 3}, {'maximum_windows_per_seed': 5},
    {'maximum_cells_per_window': 33}, {'maximum_gap_height': float('nan')},
    {'maximum_neighbor_cells': True}])
def test_unsafe_or_malformed_options_are_rejected(updates):
    base, core, lines = fixture()
    options = updates if not isinstance(updates, dict) or not updates else {**windows.DEFAULT_OPTIONS, **updates}
    with pytest.raises(ValueError):
        windows.propose_windows(base, core, lines, [], [], limits=windows.DEFAULT_LIMITS, options=options)


def test_options_are_detached_resource_caps_cannot_increase_and_blank_is_empty():
    options = windows.configuration()
    options['maximum_neighbor_cells'] = 0
    assert windows.configuration() == windows.DEFAULT_OPTIONS
    base, core, lines = fixture()
    for key in ('maximum_candidates', 'maximum_crop_pixels', 'maximum_pixels', 'maximum_components'):
        with pytest.raises(ValueError):
            run(base, core, lines, limits={**windows.DEFAULT_LIMITS, key: windows.DEFAULT_LIMITS[key]+1})
    blank = np.zeros_like(base)
    assert run(blank, blank, [])['proposals'] == []


def test_invalid_mask_canvas_and_geometry_are_rejected():
    base, core, lines = fixture()
    invalid = core.copy()
    invalid[0, 0] = 1
    for b, c, ln in ((base, invalid, lines), (base[:, :4], core, lines),
                     (base*2, core, lines), (base, core, [{'bbox': [-1, 16, 91, 36]}])):
        with pytest.raises(ValueError):
            run(b, c, ln)
