"""Geometry controls for opt-in complete line spans; no OCR quality claims."""
import copy

import numpy as np
import pytest

from tools.RegionKaomoji import line_spans as spans


def fixture(bridge=False, mark=False):
    core = np.zeros((60, 110), dtype=np.uint8)
    core[16:36, 4:17] = 1  # Dense body cells bound a sparse symbol run.
    core[16:36, 77:91] = 1
    core[16:36, 25:27] = core[16:36, 65:67] = 1
    core[24:27, 35:38] = core[24:27, 53:56] = 1
    core[32:34, 43:49] = 1
    base = core.copy()
    if bridge:
        base[26, 15:79] = 1
    if mark:
        base[10:12, 35:37] = 1
        base[27:29, 20:23] = 1  # Base-only exterior arm, nearer face than body.
    lines = [{'bbox': [4, 9 if mark else 16, 91, 36]}]
    return base, core, lines


def run(base, core, lines, seeds=(), entries=(), **kwargs):
    return spans.propose_spans(base, core, lines, entries, seeds,
                               limits=kwargs.get('limits', spans.DEFAULT_LIMITS),
                               options=kwargs.get('options', spans.DEFAULT_OPTIONS))


def test_physical_line_enumerates_sparse_complete_span_without_brackets_or_text():
    base, core, lines = fixture()
    result = run(base, core, lines)
    face = next(p for p in result['proposals'] if p['ink_bbox'] == [25, 16, 67, 36])
    assert face['sources'] == ['image-line-span']
    assert face['line_span']['body_safe'] is False
    assert face['line_span']['seed_kind'] == 'physical-line'
    assert not face['line_span']['base_bridge']['present']


def test_single_core_blank_column_separates_face_from_adjacent_dense_body():
    base, core, lines = fixture()
    core[16:36, 17:24] = core[16:36, 68:77] = 1
    base = core.copy()
    result = run(base, core, lines)
    face = next(p for p in result['proposals'] if p['ink_bbox'] == [25, 16, 67, 36])
    assert face['line_span']['crop_limits'][0] == 25
    assert face['line_span']['crop_limits'][2] == 67


def test_reclaims_base_only_point_and_exterior_arm_as_whole_components():
    base, core, lines = fixture(mark=True)
    result = run(base, core, lines)
    face = next(p for p in result['proposals'] if p['ink_bbox'][0] == 20)
    assert face['ink_bbox'] == [20, 10, 67, 36]
    assert face['line_span']['base_only_pixels'] == 10
    assert len(face['line_span']['base_only_component_ids']) == 2


def test_halo_bridge_is_recorded_without_cutting_core_or_mutating_inputs():
    base, core, lines = fixture(bridge=True)
    before = base.copy(), core.copy(), copy.deepcopy(lines)
    result = run(base, core, lines)
    face = next(p for p in result['proposals'] if p['ink_bbox'][0] > 17 and p['ink_bbox'][2] < 77)
    evidence = face['line_span']
    assert evidence['base_bridge']['present']
    assert evidence['base_bridge']['crossing_pixels'] > 0
    assert evidence['original_pixels_required']
    assert np.array_equal(base, before[0]) and np.array_equal(core, before[1])
    assert lines == before[2]


def test_two_separate_spans_keep_distinct_boundaries_and_do_not_union():
    base, core, lines = fixture()
    seed = {'ink_bbox': [25, 16, 56, 36], 'sources': ['image-components']}
    result = run(base, core, lines, [seed])
    boxes = [p['ink_bbox'] for p in result['proposals']]
    assert [25, 16, 56, 36] in boxes and [25, 16, 67, 36] in boxes
    assert len(boxes) == len(set(map(tuple, boxes)))
    assert all(b[0] >= 17 and b[2] <= 77 for b in boxes)


def test_core_component_crossing_line_boundary_is_rejected():
    base, core, _ = fixture()
    lines = [{'bbox': [4, 20, 91, 36]}]
    result = run(base, core, lines)
    assert not result['proposals']
    assert result['rejections']['line-cuts-core-component']


def test_diagonally_connected_core_is_never_cut_by_seed():
    base, core, lines = fixture()
    core[28, 39] = core[29, 40] = 1
    base |= core
    seed = {'ink_bbox': [25, 16, 40, 36], 'sources': ['first-pass-symbol-geometry']}
    for item in run(base, core, lines, [seed])['proposals']:
        box = item['ink_bbox']
        assert not (box[0] <= 39 < box[2] <= 40)


def test_ambiguous_base_only_ownership_rejects_clipping_hypothesis():
    base, core, lines = fixture()
    base[27:29, 19:23] = 1  # Exactly tied between body end17 and face start25.
    result = run(base, core, lines)
    assert not result['proposals']
    assert result['rejections']['ambiguous-base-only-ownership']


def test_unrelated_second_physical_line_is_not_absorbed():
    base, core, lines = fixture()
    core[43:54, 35:42] = 1
    base |= core
    lines.append({'bbox': [35, 43, 42, 54]})
    first = [p for p in run(base, core, lines)['proposals'] if p['line_span']['physical_line_index'] == 0]
    assert first and all(p['ink_bbox'][3] <= 36 for p in first)


def test_budgets_and_four_hypotheses_per_seed_are_fixed_ceiling():
    base, core, lines = fixture()
    seed = {'ink_bbox': [35, 24, 56, 34], 'sources': ['image-components']}
    limits = dict(spans.DEFAULT_LIMITS, maximum_candidates=1)
    result = run(base, core, lines, [seed], limits=limits)
    assert len(result['proposals']) == 1
    assert result['rejections']['total-span-budget']
    assert result['crop_pixels'] <= limits['maximum_crop_pixels']
    tiny = dict(spans.DEFAULT_LIMITS, maximum_crop_pixels=10)
    assert not run(base, core, lines, limits=tiny)['proposals']
    many = run(base, core, lines, [seed] * 10)
    assert max(many['seed_hypothesis_counts'], default=0) <= 4


def test_one_seed_crossing_multiple_lines_has_only_four_total_hypotheses():
    base, core, lines = fixture()
    base = np.concatenate((base, base), axis=0)
    core = np.concatenate((core, core), axis=0)
    lines += [{'bbox': [4, 76, 91, 96]}]
    seed = {'ink_bbox': [35, 0, 56, 120]}
    result = run(base, core, lines, [seed])
    selected = [p for p in result['proposals'] if p['line_span']['seed_kind'] == 'proposal']
    assert len(selected) <= 4
    record = next(r for r in result['seed_hypotheses'] if r['kind'] == 'proposal' and r['index'] == 0)
    assert record['attempted'] == 4


def test_global_added_candidate_cap_cannot_exceed_48():
    base, core, lines = fixture()
    base = np.concatenate([base] * 50, axis=0)
    core = np.concatenate([core] * 50, axis=0)
    lines = [{'bbox': [4, 16 + i * 60, 91, 36 + i * 60]} for i in range(50)]
    result = run(base, core, lines)
    assert len(result['proposals']) == 48
    assert result['rejections']['total-span-budget'] == 1


@pytest.mark.parametrize('limit', [{'maximum_candidates': 1}, {'maximum_crop_pixels': 840},
                                  {'maximum_crop_pixels': 841}])
def test_exhausted_budget_stops_envelope_work_across_all_later_seeds_and_lines(monkeypatch, limit):
    base, core, _ = fixture()
    base = np.concatenate([base] * 50, axis=0)
    core = np.concatenate([core] * 50, axis=0)
    lines = [{'bbox': [4, 16 + i * 60, 91, 36 + i * 60]} for i in range(50)]
    seeds = [{'ink_bbox': line['bbox']} for line in lines]
    calls = []
    original = spans._envelope

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(spans, '_envelope', measured)
    result = run(base, core, lines, seeds, limits={**spans.DEFAULT_LIMITS, **limit})
    assert len(result['proposals']) == 1
    assert len(calls) == 1
    assert result['rejections']['total-span-budget'] == 1


def test_core_area_over_remaining_budget_skips_expensive_envelope(monkeypatch):
    base, core, lines = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError('Core envelope cannot fit the remaining crop budget')

    monkeypatch.setattr(spans, '_envelope', forbidden)
    result = run(base, core, lines, limits={**spans.DEFAULT_LIMITS, 'maximum_crop_pixels': 10})
    assert result['proposals'] == []
    assert result['rejections']['total-span-budget'] == 1


def test_oversized_hypothesis_does_not_discard_a_later_smaller_valid_span():
    base, core, lines = fixture()
    base = np.concatenate((base, np.zeros_like(base)), axis=0)
    core = np.concatenate((core, np.zeros_like(core)), axis=0)
    core[76:86, 4:6] = 1
    base |= core
    lines += [{'bbox': [4, 76, 6, 86]}]
    result = run(base, core, lines, limits={**spans.DEFAULT_LIMITS, 'maximum_crop_pixels': 100})
    assert [p['ink_bbox'] for p in result['proposals']] == [[4, 76, 6, 86]]
    assert result['crop_pixels'] == 20


def test_geometry_never_reads_baseline_text_or_non_geometric_seed_fields():
    class Unreadable:
        def __str__(self):
            raise AssertionError('Text was consulted')

    base, core, lines = fixture()
    first = run(base, core, lines, entries=[{'index': 8, 'bbox': [4, 16, 91, 36], 'text': Unreadable()}])
    second = run(base, core, lines, entries=[{'index': 8, 'bbox': [4, 16, 91, 36], 'text': 'different'}])
    assert first == second
    assert all(p['line_span']['geometric_parent_indices'] == [8] for p in first['proposals'])


@pytest.mark.parametrize('option', [None, True, {}, {'strategy': 'other'},
    {**spans.DEFAULT_OPTIONS, 'maximum_spans_per_seed': 5},
    {**spans.DEFAULT_OPTIONS, 'minimum_blank_columns': True},
    {**spans.DEFAULT_OPTIONS, 'maximum_gap_height': float('nan')},
    {**spans.DEFAULT_OPTIONS, 'maximum_cells_per_span': 33}])
def test_malformed_explicit_options_are_refused(option):
    base, core, lines = fixture()
    with pytest.raises(ValueError):
        run(base, core, lines, options=option)


def test_default_configuration_is_detached_and_limits_cannot_increase():
    first = spans.configuration()
    first['maximum_spans_per_seed'] = 1
    assert spans.configuration() == spans.DEFAULT_OPTIONS
    base, core, lines = fixture()
    for key in ('maximum_candidates', 'maximum_crop_pixels', 'maximum_components', 'maximum_pixels'):
        with pytest.raises(ValueError):
            run(base, core, lines, limits={**spans.DEFAULT_LIMITS, key: spans.DEFAULT_LIMITS[key] + 1})


def test_invalid_input_masks_or_boxes_are_refused_and_blank_is_empty():
    base, core, lines = fixture()
    invalid = core.copy()
    invalid[0, 0] = 1
    for b, c, ln in ((base, invalid, lines), (base[:, :10], core, lines),
                     (base * 2, core, lines), (base, core, [{'bbox': [-1, 0, 9, 9]}])):
        with pytest.raises(ValueError):
            run(b, c, ln)
    blank = np.zeros_like(base)
    assert run(blank, blank, [])['proposals'] == []
