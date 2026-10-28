"""Pinned native pipeline with deterministic detector/session stubs, no models."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools.RegionKaomoji import ctc_trace as ct, evaluate as evaluation
from tools.RegionKaomoji.initial_trace import InitialTraceAdapter, _METHODS
from test_baseline_trace import native_recognizer, load_worker


def native_engine(*, patterns=None, classify=True, rotated=False, preprocess=False, padding=False):
    from rapidocr import RapidOCR
    from rapidocr.ch_ppocr_det import TextDetOutput
    from rapidocr.ch_ppocr_cls import TextClsOutput
    engine = object.__new__(RapidOCR)
    rec, calls = native_recognizer()
    state = {'offset': 0, 'det': 0, 'cls': 0}
    boxes = np.asarray([[[10, 10], [110, 10], [110, 30], [10, 30]],
                        [[20, 45], [180, 45], [180, 65], [20, 65]],
                        [[15, 75], [95, 75], [95, 95], [15, 95]]], np.float32)
    engine.cfg = SimpleNamespace(Global=SimpleNamespace(use_preprocess_img=preprocess,
        use_vertical_padding=padding, text_score=.5, font_path=None), Rec=SimpleNamespace(lang_type='ch', font_path=None))
    engine.use_det, engine.use_cls, engine.use_rec = True, classify, True
    engine.return_word_box, engine.return_single_char_box = False, False
    engine.text_score, engine.min_side_len, engine.max_side_len = .5, 30, 180
    engine.min_height, engine.width_height_ratio = 110 if padding else 10, -1
    engine.load_img = lambda image: image
    def detect(image):
        state['det'] += 1
        state['offset'] = 0
        located = boxes.copy()
        if padding:
            located[:, :, 1] += max(0, (image.shape[0]-110)//2)
        return TextDetOutput(boxes=located, scores=[.99]*3, elapse=0.)
    class Classifier:
        cls_thresh = .9
        def __call__(self, images):
            state['cls'] += 1
            result = copy.deepcopy(images)
            if rotated:
                result[0] = np.rot90(result[0], 2).copy()
            return TextClsOutput(img_list=result,
                cls_res=[('180' if rotated and i == 0 else '0', .95) for i in range(len(images))])
    engine.text_det, engine.text_cls, engine.text_rec = detect, Classifier(), rec
    if patterns is not None:
        def session(batch):
            calls.append(tuple(batch.shape))
            values = np.zeros((len(batch), 80, len(rec.postprocess_op.character)), np.float32)
            for i in range(len(batch)):
                ids, score = patterns[state['offset'] + i]
                path = ids + [0] * (80-len(ids))
                values[i, np.arange(80), path] = score
            state['offset'] += len(batch)
            return values
        rec.session = session
    return engine, rec, calls, state


def canonical(result, size=(220, 110)):
    worker = load_worker()
    import json
    raw = json.loads(worker.serialize_json_result(result, *size, 'fixture'))
    return [{'bbox': [r['rect']['x'], r['rect']['y'], r['rect']['x']+r['rect']['width'],
                       r['rect']['y']+r['rect']['height']], 'text': r['text'], 'score': r['score']}
            for r in raw['regions']]


def observe(engine, pixels, *, events=(), plan=None, limits=None):
    adapter = InitialTraceAdapter(engine, image_sha256='a'*64, image_size=[pixels.shape[1], pixels.shape[0]],
                                  identity={'model': 'fixture'}, limits=limits)
    with adapter:
        with adapter.observe_initial(pixels):
            result = engine(pixels)
        entries = canonical(result, (pixels.shape[1], pixels.shape[0]))
        for event, payload in events:
            adapter.observe_production_reread(event, payload)
        adapter.finish_production(entries)
        adapter.bind_initial_entries(entries, plan or {'parents': {}, 'children': {}})
        sidecar = adapter.sidecar(entries)
    return result, entries, sidecar, adapter


def test_initial_observation_preserves_native_boxes_scores_counts_sort_and_restores():
    engine, rec, calls, state = native_engine()
    pixels = np.arange(220*110*3, dtype=np.uint8).reshape(110, 220, 3)
    expected = engine(pixels)
    original = {name: getattr(engine, name) for name in _METHODS}
    decoder, resize = rec.postprocess_op, rec.resize_norm_img
    actual, entries, sidecar, adapter = observe(engine, pixels)
    assert actual.txts == expected.txts and actual.scores == expected.scores
    assert np.array_equal(actual.boxes, expected.boxes)
    assert len(calls) == 4 and state['det'] == state['cls'] == 2
    rows = ct.validate_sidecar('a'*64, [220, 110], entries, sidecar)
    assert len(rows) == 3
    assert [row['trace']['geometry']['sorted_index'] for row in rows] == [1, 2, 0]
    assert all(row['trace']['geometry']['geometry_kind'] == 'first-pass-horizontal-v1' for row in rows)
    assert adapter.receipt()['initial_rows_bound'] == 3
    assert all(getattr(engine, name) == method for name, method in original.items())
    assert rec.postprocess_op is decoder and rec.resize_norm_img == resize
    assert not any(name in vars(engine) for name in _METHODS)


def test_native_empty_and_score_filter_keep_actual_detector_identity():
    # Native ratio sort is detector indices 2,0,1. Empty row2 and low-score row0
    # leave source detector1 as final slot0, not a bbox/text search result.
    engine, _, _, _ = native_engine(patterns=[([0], .95), ([1, 2, 3], .3), ([1, 2, 3], .95)], classify=False)
    _, entries, sidecar, _ = observe(engine, np.zeros((110, 220, 3), np.uint8))
    assert len(entries) == len(sidecar['rows']) == 1
    geometry = sidecar['rows'][0]['trace']['geometry']
    assert geometry['detector_index'] == 1
    assert geometry['initial_output_index'] == geometry['production_parent_index'] == geometry['final_parent_index'] == 0


def test_classification_rotation_is_rejected_even_if_uniform_pixels_match():
    engine, _, _, _ = native_engine(rotated=True)
    _, _, sidecar, adapter = observe(engine, np.zeros((110, 220, 3), np.uint8))
    assert len(sidecar['rows']) == 2
    assert adapter.receipt()['unsupported']['initial-classified-rotation-or-pixels'] == 1


def test_applied_equal_text_reread_cannot_resurrect_initial_identity():
    engine, _, _, _ = native_engine()
    events = [('selection', {'applied': True, 'source_indices': [0], 'synthetic': True})]
    _, _, sidecar, _ = observe(engine, np.zeros((110, 220, 3), np.uint8), events=events)
    assert [row['parent_index'] for row in sidecar['rows']] == [1, 2]


def test_split_plan_uses_explicit_unsplit_map_and_never_painted_children():
    engine, _, _, _ = native_engine()
    plan = {'parents': {'0': {}}, 'children': {'0': 0}, 'map': {'1': 1, '2': 2}}
    _, _, sidecar, _ = observe(engine, np.zeros((110, 220, 3), np.uint8), plan=plan)
    assert [row['parent_index'] for row in sidecar['rows']] == [1, 2]


def test_child_reread_indices_do_not_invalidate_production_initial_rows():
    engine, _, _, _ = native_engine()
    pixels = np.zeros((110, 220, 3), np.uint8)
    adapter = InitialTraceAdapter(engine, image_sha256='a'*64, image_size=[220, 110], identity={})
    with adapter:
        with adapter.observe_initial(pixels):
            result = engine(pixels)
        entries = canonical(result)
        adapter.finish_production(entries)
        adapter.observe_reread('selection', {'applied': True, 'source_indices': [0], 'synthetic': True})
        adapter.bind_initial_entries(entries, {'parents': {}, 'children': {}})
        assert len(adapter.sidecar(entries)['rows']) == 3


def test_budget_overflow_and_unknown_operations_do_not_change_native_execution():
    engine, _, calls, _ = native_engine()
    result, _, sidecar, adapter = observe(engine, np.zeros((110, 220, 3), np.uint8), limits=ct.TraceLimits(max_rows=1))
    assert len(result.txts) == 3 and len(calls) == 2 and sidecar['rows'] == []
    assert adapter.receipt()['unsupported']['initial-row-cap-or-count'] == 1


def test_engine_and_recognizer_hooks_restore_after_native_failure():
    engine, rec, _, _ = native_engine()
    original = {name: getattr(engine, name) for name in _METHODS}
    decoder, resize = rec.postprocess_op, rec.resize_norm_img
    def broken(_):
        raise RuntimeError('native sentinel')
    rec.session = broken
    with pytest.raises(RuntimeError, match='native sentinel'):
        observe(engine, np.zeros((110, 220, 3), np.uint8))
    assert all(getattr(engine, name) == method for name, method in original.items())
    assert rec.postprocess_op is decoder and rec.resize_norm_img == resize


def test_observer_failure_falls_back_without_affecting_native_result(monkeypatch):
    engine, _, calls, _ = native_engine()
    def broken(*_):
        raise RuntimeError('observer sentinel')
    monkeypatch.setattr(InitialTraceAdapter, '_cropped', broken)
    result, _, sidecar, adapter = observe(engine, np.zeros((110, 220, 3), np.uint8))
    assert len(result.txts) == 3 and len(calls) == 2 and sidecar['rows'] == []
    assert adapter.receipt()['unsupported']['initial-observation-failure'] >= 1


def test_fullbaseline_optional_path_wires_initial_bindings_without_worker_changes():
    engine, rec, calls, _ = native_engine()
    baseline = object.__new__(evaluation.FullBaseline)
    worker = load_worker()
    baseline.engine = engine
    baseline.worker = SimpleNamespace(serialize_json_result=worker.serialize_json_result,
                                     reread_symbols=lambda engine, pixels, result, observer=None: result)
    baseline.binding, baseline.runtime_receipt = {'version': 'fixture'}, {'fixture': True}
    image = Image.new('RGB', (220, 110), 'white')
    with baseline.trace_capture(image, initial=True) as adapter:
        entries, _ = baseline.predict(image)
        adapter.bind_initial_entries(entries, {'parents': {}, 'children': {}})
        assert len(adapter.sidecar(entries)['rows']) == 3
    assert len(calls) == 2 and not hasattr(baseline, '_trace_observer')


@pytest.mark.parametrize('preprocess,padding', [(True, False), (False, True)])
def test_actual_native_preprocess_and_padding_are_bound(preprocess, padding):
    engine, _, _, _ = native_engine(preprocess=preprocess, padding=padding)
    pixels = np.zeros((110, 220, 3), np.uint8)
    expected = engine(pixels)
    actual, entries, sidecar, _ = observe(engine, pixels)
    assert actual.txts == expected.txts and actual.scores == expected.scores
    assert np.array_equal(actual.boxes, expected.boxes)
    rows = ct.validate_sidecar('a'*64, [220, 110], entries, sidecar)
    assert rows
    operation = rows[0]['trace']['geometry']['operations']
    if preprocess:
        assert operation['preprocess']['ratio_w'] != 1
    else:
        assert operation['padding_1']['top'] > 0


def test_merged_production_reread_invalidates_sources_and_reindexes_later_parent():
    engine, _, _, _ = native_engine()
    pixels = np.zeros((110, 220, 3), np.uint8)
    adapter = InitialTraceAdapter(engine, image_sha256='a'*64, image_size=[220, 110], identity={})
    with adapter:
        with adapter.observe_initial(pixels):
            result = engine(pixels)
        original = canonical(result)
        adapter.observe_production_reread('selection', {'applied': True, 'source_indices': [0, 1], 'synthetic': True})
        entries = [{'bbox': [10, 10, 180, 65], 'text': 'merged', 'score': .9}, original[2]]
        adapter.finish_production(entries)
        adapter.bind_initial_entries(entries, {'parents': {}, 'children': {}})
        rows = ct.validate_sidecar('a'*64, [220, 110], entries, adapter.sidecar(entries))
        assert len(rows) == 1 and rows[0]['parent_index'] == 1
        assert rows[0]['trace']['geometry']['detector_index'] == 2


def test_unsplit_parent_index_remaps_after_painted_child_expansion():
    engine, _, _, _ = native_engine()
    pixels = np.zeros((110, 220, 3), np.uint8)
    adapter = InitialTraceAdapter(engine, image_sha256='a'*64, image_size=[220, 110], identity={})
    with adapter:
        with adapter.observe_initial(pixels):
            result = engine(pixels)
        original = canonical(result)
        adapter.finish_production(original)
        entries = [dict(original[0], text='painted1'), dict(original[0], text='painted2'), *original[1:]]
        adapter.bind_initial_entries(entries, {'parents': {'0': {}}, 'children': {'0': 0, '1': 0}, 'map': {'2': 1, '3': 2}})
        rows = ct.validate_sidecar('a'*64, [220, 110], entries, adapter.sidecar(entries))
        assert [row['parent_index'] for row in rows] == [2, 3]
        assert [row['trace']['geometry']['production_parent_index'] for row in rows] == [1, 2]


def test_initial_scope_is_single_use_and_cannot_retain_paths_from_another_call():
    engine, _, calls, _ = native_engine()
    pixels = np.zeros((110, 220, 3), np.uint8)
    adapter = InitialTraceAdapter(engine, image_sha256='a'*64, image_size=[220, 110], identity={})
    with adapter:
        with adapter.observe_initial(pixels):
            result = engine(pixels)
        entries = canonical(result)
        adapter.finish_production(entries)
        adapter.bind_initial_entries(entries, {'parents': {}, 'children': {}})
        assert len(adapter.sidecar(entries)['rows']) == 3
        with pytest.raises(ValueError, match='one active'):
            with adapter.observe_initial(pixels):
                engine(pixels)
    assert len(calls) == 2


def test_late_observer_failure_drops_already_bound_initial_paths():
    engine, _, _, _ = native_engine()
    _, entries, _, adapter = observe(engine, np.zeros((110, 220, 3), np.uint8))
    def broken():
        raise ValueError('late binding failure')
    adapter._safe(broken)
    assert adapter.sidecar(entries)['rows'] == []
    assert adapter.receipt()['initial_rows_bound'] == 0


def test_small_byte_budget_preserves_all_native_results_with_empty_trace():
    engine, _, calls, _ = native_engine()
    result, _, sidecar, adapter = observe(engine, np.zeros((110, 220, 3), np.uint8),
                                         limits=ct.TraceLimits(max_total_bytes=2048))
    assert len(result.txts) == 3 and len(calls) == 2
    assert sidecar is None or sidecar['rows'] == []
    assert adapter.receipt()['compact_bytes'] == 0


def test_unknown_engine_method_override_is_rejected_before_observation():
    engine, _, calls, _ = native_engine()
    engine.preprocess_img = lambda image: (image, {})
    with pytest.raises(ValueError, match='unmodified RapidOCR'):
        InitialTraceAdapter(engine, image_sha256='a'*64, image_size=[220, 110], identity={})
    assert not calls
