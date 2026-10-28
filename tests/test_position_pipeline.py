"""Actual research plumbing with declared OCR fixtures; no quality claims."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.RegionKaomoji import candidates as c, ctc_trace, evaluate as e, fusion as f, owned_fusion
from tests.test_baseline_trace import native_recognizer
from tests.test_pixel_ownership_integration import request
from tests.test_position_spans import scene


def test_predict_scene_commits_cross_owned_duplicate_after_actual_fusion():
    image, baseline, _, views, candidate_config = request(
        baseline=[{'bbox': [130, 17, 153, 54], 'text': '(x_x)ノ', 'score': .7}])
    readings = {record['image_sha256']: (record['text'], record['score'])
                for records in views.values() for record in records}
    reader = SimpleNamespace(binding={'id': 'declared-fixture-only'},
        predict=lambda pixels: readings[c.image_digest(pixels)])
    base = SimpleNamespace(predict=lambda pixels: (baseline, {}))
    config = e.configuration()
    config.update(candidate=candidate_config, views=['raw', 'edge-pad-6'])
    config['split']['enabled'] = False
    legacy = e.predict_scene(Image.fromarray(image), base, [reader], config)['models'][reader.binding['id']]
    assert [region['text'] for region in legacy['regions']] == ['(ω_ω)ノ', '(ω_ω)']
    config['fusion']['owned_ownership'] = owned_fusion.configuration()
    actual = e.predict_scene(Image.fromarray(image), base, [reader], config)['models'][reader.binding['id']]
    assert actual['text'] == '(ω_ω)ノ'
    assert actual['accepted_count'] == 2 and actual['final_accepted_count'] == 1
    assert len(actual['regions']) == 1 and actual['owned_ownership']['suppressed']


@pytest.mark.parametrize('body,baseline', [(True, []), (False, []),
    (True, [{'bbox': [130, 17, 153, 54], 'text': '(x_x)ノ', 'score': .7}])])
def test_strategy_off_matches_preserved_pre_round_pipeline_snapshot(body, baseline):
    """Local snapshot replay covers semantic fields; timings are measured anew."""
    snapshot = Path(__file__).resolve().parents[1]/'artifacts/ownership-position-20261022/source-before-round'
    if not (snapshot/'tools/RegionKaomoji/evaluate.py').is_file():
        pytest.skip('Local pre-round source snapshot is intentionally not distributed')
    def load(name):
        spec = importlib.util.spec_from_file_location('position_snapshot_'+name,
            snapshot/'tools/RegionKaomoji'/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    old_c, old_f, old_e = load('candidates'), load('fusion'), load('evaluate')
    old_f.c, old_e.c, old_e.f = old_c, old_c, old_f
    image, entries, _, views, candidate_config = request(body=body, baseline=baseline)
    readings = {record['image_sha256']: (record['text'], record['score'])
                for records in views.values() for record in records}
    reader = SimpleNamespace(binding={'id': 'declared-fixture-only'},
        predict=lambda pixels: readings[c.image_digest(pixels)])
    base = SimpleNamespace(predict=lambda pixels: (entries, {}))
    config = e.configuration()
    config.update(candidate=candidate_config, views=['raw', 'edge-pad-6'])
    config['split']['enabled'] = False
    before = old_e.predict_scene(Image.fromarray(image), base, [reader], config)
    after = e.predict_scene(Image.fromarray(image), base, [reader], config)
    from tools.RegionKaomoji.position_quality import prediction_identity
    assert prediction_identity([{'prediction': before}]) == prediction_identity([{'prediction': after}])


def test_fullbaseline_trace_context_restores_native_hooks_even_on_failure():
    rec, calls = native_recognizer()
    baseline = object.__new__(e.FullBaseline)
    baseline.engine = SimpleNamespace(_load_rec_model=lambda: rec)
    baseline.binding, baseline.runtime_receipt = {'id': 'fixture'}, {'fixture': True}
    decoder, resize = rec.postprocess_op, rec.resize_norm_img
    with pytest.raises(RuntimeError, match='sentinel'):
        with baseline.trace_capture(Image.new('RGB', (120, 60), 'white')):
            assert baseline._trace_observer is not None
            raise RuntimeError('sentinel')
    assert not hasattr(baseline, '_trace_observer')
    assert rec.postprocess_op is decoder and rec.resize_norm_img == resize
    assert calls == []


def trace_fixture():
    base, core, _ = scene()
    pixels = np.repeat((255-base*255)[:, :, None], 3, axis=2)
    pixels[(base != 0) & (core == 0)] = 215
    box = [4, 9, 91, 36]
    geometry = {'crop_bbox': box, 'input_shape': [27, 87], 'content_width': 155,
        'batch_width': 320, 'resize_height': 48, 'padding': [0, 0, 0, 0],
        'selected_view': 'raw', 'rotation': 0,
        'inverse_perspective': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        'input_index': 0, 'sorted_index': 0, 'batch_index': 0, 'batch_ordinal': 0,
        'source_indices': [0], 'before_text': '[^]'}
    prediction = np.zeros((100, 4), np.float32)
    prediction[:, 0] = .9
    for step, token in ((12, 1), (22, 2), (34, 3)):
        prediction[step] = 0
        prediction[step, token] = .95
    path = ctc_trace.compact_path(prediction, ['blank', '[', '^', ']'], geometry)
    entries = [{'bbox': box, 'text': path['text'], 'score': .9}]
    sidecar = ctc_trace.make_sidecar(c.image_digest(pixels), [120, 60], {'fixture': True}, [path], entries, {})
    config = {**c.DEFAULT_CONFIG, 'shape_contrast_threshold': 80,
        'position_spans': {'strategy': 'ctc-components-v1', 'mode': 'ctc'}}
    return pixels, entries, sidecar, config


def test_live_position_sidecar_reaches_actual_proposer_and_fusion_reaudit():
    pixels, entries, sidecar, config = trace_fixture()
    result = c.locate_candidates(pixels, entries, config, position_trace=sidecar)
    assert result['position_span_detection']['proposals'] > 0
    final = f.fuse_candidates(pixels, entries, result['candidates'], {},
        candidate_config=config, position_trace=sidecar)
    assert final['accepted_count'] == 0  # No fixture readings supplied to fusion.
    assert any(item['sources'] == ['image-position-ctc'] for item in result['candidates'])
    # A JSON-style copied receipt cannot manufacture the same new candidates.
    plain = dict(sidecar)
    rejected = c.locate_candidates(pixels, entries, config, position_trace=plain)
    assert rejected['position_span_detection']['rejections']['trace-unavailable-or-unsupported']
    with pytest.raises(ValueError, match='differs from automatic|unknown automatic'):
        f.fuse_candidates(pixels, entries, result['candidates'], {},
            candidate_config=config, position_trace=plain)
