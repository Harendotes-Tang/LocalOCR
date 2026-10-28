"""Observe pinned RapidOCR code with a tiny deterministic in-memory session stub."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.RegionKaomoji.baseline_trace import BaselineTraceAdapter
from tools.RegionKaomoji import ctc_trace as trace


def native_recognizer():
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
    recognizer = object.__new__(TextRecognizer)
    recognizer.rec_image_shape = [3, 48, 320]
    recognizer.rec_batch_num = 2
    recognizer.cfg = SimpleNamespace(lang_type='ch', font_path=None)
    recognizer.postprocess_op = CTCLabelDecode(character=['(', '^', ')'])
    calls = []
    def session(batch):
        calls.append(tuple(batch.shape))
        ids = [0, 1, 2, 0, 2, 3, 0] + [0] * 73
        values = np.zeros((len(batch), len(ids), len(recognizer.postprocess_op.character)), np.float32)
        values[:, np.arange(len(ids)), ids] = .95
        return values
    recognizer.session = session
    return recognizer, calls


def test_observer_preserves_native_outputs_forward_count_and_sorted_padding():
    from rapidocr.ch_ppocr_rec import TextRecInput
    rec, calls = native_recognizer()
    wide, narrow = np.zeros((20, 300, 3), np.uint8), np.zeros((20, 100, 3), np.uint8)
    args = TextRecInput(img=[wide, narrow], return_word_box=False)
    expected = rec(args)
    assert len(calls) == 1
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[400, 100], identity={'model': 'fixture'})
    with adapter:
        with adapter.known_crops([{'bbox': [0, 0, 300, 20], 'source_indices': [0], 'before_text': '('},
                                  {'bbox': [0, 30, 100, 50], 'source_indices': [1], 'before_text': '('}]):
            actual = rec(args)
    assert actual.txts == expected.txts and actual.scores == expected.scores
    assert len(calls) == 2
    entries = [{'bbox': [0, 0, 300, 20], 'text': actual.txts[0]},
               {'bbox': [0, 30, 100, 50], 'text': actual.txts[1]}]
    sidecar = adapter.sidecar(entries)
    rows = trace.validate_sidecar('a' * 64, [400, 100], entries, sidecar)
    assert len(rows) == 2
    assert [row['trace']['geometry']['input_index'] for row in rows] == [0, 1]
    assert [row['trace']['geometry']['sorted_index'] for row in rows] == [1, 0]
    assert [row['trace']['geometry']['batch_width'] for row in rows] == [720, 720]
    assert [row['trace']['geometry']['content_width'] for row in rows] == [720, 240]


def load_worker():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'local-ai-ocr-worker.py'
    spec = importlib.util.spec_from_file_location('localocr_trace_worker_fixture', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def test_symbol_reread_observer_binds_real_selected_raw_view_without_extra_forward():
    worker = load_worker()
    rec, calls = native_recognizer()
    engine = SimpleNamespace(_load_rec_model=lambda: rec)
    pixels = np.zeros((80, 130, 3), np.uint8)
    pixels[25:39, 25:45] = 220
    initial = SimpleNamespace(boxes=[[[20, 20], [50, 20], [50, 40], [20, 40]]], txts=['('], scores=[.9])
    expected = worker.reread_symbols(engine, pixels, initial)
    expected_calls = len(calls)
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[130, 80], identity={'model': 'fixture'})
    with adapter:
        actual = worker.reread_symbols(engine, pixels, initial, observer=adapter.observe_reread)
    assert actual.txts == expected.txts and actual.scores == expected.scores
    assert actual.boxes == expected.boxes
    assert len(calls) == 2 * expected_calls
    l, t = actual.boxes[0][0]
    r, b = actual.boxes[0][2]
    sidecar = adapter.sidecar([{'bbox': [int(value) for value in (l, t, r, b)], 'text': actual.txts[0]}])
    assert len(sidecar['rows']) == 1
    assert sidecar['rows'][0]['selected_view'] == 'raw'
    assert sidecar['rows'][0]['trace']['geometry']['source_indices'] == [0]


def test_unknown_first_pass_never_guesses_reverse_perspective():
    from rapidocr.ch_ppocr_rec import TextRecInput
    rec, calls = native_recognizer()
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[130, 80], identity={'model': 'fixture'})
    with adapter:
        result = rec(TextRecInput(img=[np.zeros((20, 100, 3), np.uint8)], return_word_box=False))
    assert len(calls) == 1
    assert adapter.sidecar([{'bbox': [0, 0, 100, 20], 'text': result.txts[0]}])['rows'] == []
    assert adapter.receipt()['unsupported'].get('unbound-input-transform') == 1


def test_total_row_cap_falls_back_but_does_not_skip_native_forward():
    from rapidocr.ch_ppocr_rec import TextRecInput
    rec, calls = native_recognizer()
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[300, 80],
                                   identity={'model': 'fixture'}, limits=trace.TraceLimits(max_rows=1))
    images = [np.zeros((20, 100, 3), np.uint8), np.zeros((20, 200, 3), np.uint8)]
    with adapter:
        with adapter.known_crops([{'bbox': [0, 0, 100, 20], 'source_indices': [0]},
                                  {'bbox': [0, 30, 200, 50], 'source_indices': [1]}]):
            output = rec(TextRecInput(img=images, return_word_box=False))
    assert len(calls) == 1 and len(output.txts) == 2
    assert adapter.receipt()['unsupported']['row-cap'] >= 1


def test_native_text_rewrite_is_not_attributed_to_its_old_ctc_path():
    from rapidocr.ch_ppocr_rec import TextRecInput
    rec, calls = native_recognizer()
    original = rec.postprocess_op
    class RewritingDecoder:
        character = original.character
        def __call__(self, *args, **kwargs):
            lines, words = original(*args, **kwargs)
            return [(text + 'X', score) for text, score in lines], words
    rec.postprocess_op = RewritingDecoder()
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[130, 80], identity={'model': 'fixture'})
    with adapter:
        with adapter.known_crops([{'bbox': [0, 0, 100, 20], 'source_indices': [0]}]):
            output = rec(TextRecInput(img=[np.zeros((20, 100, 3), np.uint8)], return_word_box=False))
    assert output.txts == ('(^^)X',) and len(calls) == 1
    assert adapter.receipt()['unsupported']['native-decoding-mismatch'] == 1
    assert not adapter.sidecar([{'bbox': [0, 0, 100, 20], 'text': output.txts[0]}])['rows']


def test_byte_cap_and_synthetic_selection_preserve_native_output():
    from rapidocr.ch_ppocr_rec import TextRecInput
    rec, calls = native_recognizer()
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[130, 80],
                                   identity={'model': 'fixture'}, limits=trace.TraceLimits(max_total_bytes=2048))
    with adapter:
        with adapter.known_crops([{'bbox': [0, 0, 100, 20], 'source_indices': [0]}]):
            output = rec(TextRecInput(img=[np.zeros((20, 100, 3), np.uint8)], return_word_box=False))
        adapter.observe_reread('selection', {'applied': True, 'synthetic': True, 'source_indices': [0]})
    value = adapter.sidecar([{'bbox': [0, 0, 100, 20], 'text': output.txts[0]}])
    assert output.txts == ('(^^)',) and len(calls) == 1
    assert not value['rows']
    assert len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) <= 2048
    assert adapter.receipt()['unsupported']['synthetic-or-merged-reread'] == 1
    assert adapter.receipt()['compact_bytes'] == 0


def test_worker_callback_failure_cannot_change_ocr_and_exact_vote_index_is_exposed():
    worker = load_worker()
    selected = []
    value = worker.choose_reread('(', [('invalid', .99), ('(^^)', .8), ('( ^^ )', .9)], observer=selected.append)
    assert value == ('(^^)', .8) and selected == [1]
    rec, calls = native_recognizer()
    engine = SimpleNamespace(_load_rec_model=lambda: rec)
    pixels = np.zeros((80, 130, 3), np.uint8)
    pixels[25:39, 25:45] = 220
    initial = SimpleNamespace(boxes=[[[20, 20], [50, 20], [50, 40], [20, 40]]], txts=['('], scores=[.9])
    expected = worker.reread_symbols(engine, pixels, initial)
    def broken(*_):
        raise ValueError('research observer failure')
    actual = worker.reread_symbols(engine, pixels, initial, observer=broken)
    assert actual.txts == expected.txts and actual.scores == expected.scores
    assert actual.boxes == expected.boxes
    assert len(calls) == 4


def test_impossible_header_emits_no_sidecar_and_keeps_native_forward():
    from rapidocr.ch_ppocr_rec import TextRecInput
    rec, calls = native_recognizer()
    adapter = BaselineTraceAdapter(rec, image_sha256='a' * 64, image_size=[130, 80],
                                   identity={'model': 'fixture'}, limits=trace.TraceLimits(max_total_bytes=1))
    with adapter:
        with adapter.known_crops([{'bbox': [0, 0, 100, 20], 'source_indices': [0]}]):
            output = rec(TextRecInput(img=[np.zeros((20, 100, 3), np.uint8)], return_word_box=False))
    assert output.txts == ('(^^)',) and len(calls) == 1
    assert adapter.sidecar([{'bbox': [0, 0, 100, 20], 'text': output.txts[0]}]) is None
    assert adapter.receipt()['compact_bytes'] == 0
    assert adapter.receipt()['unsupported']['sidecar-header-byte-cap'] == 1
