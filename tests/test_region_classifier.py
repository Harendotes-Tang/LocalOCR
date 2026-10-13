"""Crop classifier evidence: preprocessing, receipt binding, routing and fusion integration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tools.PretrainedOcr import common
from tools.RegionKaomoji import candidates as c, classifier as cl, fusion as f


def test_preprocess_is_polarity_invariant_and_aspect_preserving():
    light = np.full((20, 60, 3), 240, dtype=np.uint8)
    light[5:15, 10:50] = 20
    dark = 255 - light
    a, b = cl.preprocess(light), cl.preprocess(dark)
    assert a.shape == b.shape == (1, cl.INPUT_HEIGHT, cl.INPUT_WIDTH) and np.allclose(a, b, atol=1 / 255)
    assert a[0, :, 96:].max() == 0 and a[0, :, :96].max() > .5  # 60x20 crop scales to 96 columns
    tall = cl.preprocess(np.full((64, 16, 3), 255, dtype=np.uint8))
    assert tall.shape == (1, cl.INPUT_HEIGHT, cl.INPUT_WIDTH) and tall.max() == 0
    wide = cl.preprocess(np.full((10, 400), 0, dtype=np.uint8))
    assert wide.shape == (1, cl.INPUT_HEIGHT, cl.INPUT_WIDTH)
    with pytest.raises(ValueError):
        cl.preprocess(np.zeros((0, 10, 3), dtype=np.uint8))


def receipt(tmp_path, **changes):
    (tmp_path / 'model.onnx').write_bytes(b'not-a-real-graph')
    value = {'schema_version': cl.SCHEMA, 'id': 'kaomoji-crop-classifier-exp-20261013-000', 'classes': list(cl.CLASSES),
             'preprocessing': cl.PREPROCESSING, 'model': {'file': 'model.onnx', 'sha256': cl.digest(tmp_path / 'model.onnx')},
             'threshold': .7, 'promote': False, 'enabled': False,
             'training': {'threshold_selected_on': 'dev', 'heldout_consumed': False, 'dataset_sha256': 'a' * 64}}
    for key, item in changes.items():
        if isinstance(item, dict) and isinstance(value.get(key), dict):
            value[key] = {**value[key], **item}
        else:
            value[key] = item
    common.write(tmp_path / 'classifier.json', value)
    return tmp_path / 'classifier.json'


def test_receipt_binding_fails_closed_on_tampering(tmp_path):
    path = receipt(tmp_path)
    binding = cl.classifier_binding(path)
    assert binding['threshold'] == .7 and binding['model_sha256'] == cl.digest(tmp_path / 'model.onnx')
    assert binding['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for changes in ({'schema_version': 'x'}, {'threshold': 1.0}, {'threshold': 0}, {'promote': True}, {'classes': ['face']},
                    {'training': {'threshold_selected_on': 'heldout'}}, {'training': {'heldout_consumed': True}},
                    {'model': {'sha256': '0' * 64}}, {'model': {'file': '../model.onnx'}}):
        with pytest.raises(ValueError):
            cl.classifier_binding(receipt(tmp_path, **changes))


class StubClassifier:
    """Declared evidence stub; never a trained model."""
    binding = {'id': 'stub', 'sha256': 'b' * 64}
    threshold = .7

    def __init__(self, face):
        self.face = face
        self.calls = 0

    def evidence(self, image):
        self.calls += 1
        probs = {'face': self.face, 'bracketed-text': (1 - self.face) / 2, 'graphic': (1 - self.face) / 2}
        return {'id': 'stub', 'threshold': self.threshold, 'probabilities': probs, 'face_probability': self.face,
                'face_evidence': self.face >= self.threshold}


def plain_box_image():
    image = Image.new('RGB', (160, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 23, 50), fill='black')
    draw.rectangle((100, 20, 103, 50), fill='black')
    draw.rectangle((45, 30, 50, 36), fill='black')
    draw.rectangle((72, 30, 77, 36), fill='black')
    draw.line((54, 43, 68, 43), fill='black', width=2)
    return np.asarray(image)


def test_classifier_evidence_can_route_without_bracket_shapes_and_is_recorded():
    image = plain_box_image()
    baseline = [{'bbox': [18, 18, 106, 53], 'text': '(w_w)', 'score': .2}]
    without = c.locate_candidates(image, baseline)
    assert without['candidates'] and not any(item['route_eligible'] for item in without['candidates'])
    assert all(not item['evidence']['extra_shape_support'] for item in without['candidates'])
    low = StubClassifier(.2)
    rejected = c.locate_candidates(image, baseline, classifier=low)
    assert low.calls == len(rejected['candidates']) and not any(item['route_eligible'] for item in rejected['candidates'])
    assert all(item['evidence']['classifier']['face_probability'] == .2 for item in rejected['candidates'])
    high = StubClassifier(.95)
    accepted = c.locate_candidates(image, baseline, classifier=high)
    eligible = [item for item in accepted['candidates'] if item['route_eligible']]
    assert eligible and all(item['evidence']['independent_face_evidence'] for item in eligible)
    assert eligible[0]['route_reason'] == 'independent-pixels-and-bounded-face-evidence'


def test_fusion_audits_candidates_with_the_same_classifier():
    image = plain_box_image()
    baseline = [{'bbox': [18, 18, 106, 53], 'text': '(w_w)', 'score': .2}]
    high = StubClassifier(.95)
    candidates = c.locate_candidates(image, baseline, classifier=high)['candidates']
    views = {}
    for candidate in candidates:
        views[candidate['candidate_id']] = [{'model_id': 'specialist', 'view_id': t, 'transform': t,
                                             'image_sha256': c.image_digest(f.view_image(image, candidate['bbox'], t)),
                                             'text': '(ω_ω)', 'score': .9} for t in ('raw', 'edge-pad-6')]
    output = f.fuse_candidates(image, baseline, candidates, views, classifier=high)
    assert output['accepted_count'] == 1 and output['regions'][0]['text'] == '(ω_ω)'
    with pytest.raises(ValueError, match='differs'):
        f.fuse_candidates(image, baseline, candidates, views)
    # A pixel-only insertion (no baseline region) still needs bracket-pair shapes.
    pixel_only = c.locate_candidates(image, [], classifier=high)['candidates']
    views = {item['candidate_id']: [{'model_id': 'specialist', 'view_id': t, 'transform': t,
                                     'image_sha256': c.image_digest(f.view_image(image, item['bbox'], t)), 'text': '(ω_ω)', 'score': .9}
                                    for t in ('raw', 'edge-pad-6')] for item in pixel_only}
    output = f.fuse_candidates(image, [], pixel_only, views, classifier=high)
    assert output['accepted_count'] == 0


def test_threshold_selection_bounds_false_positives():
    from tools.RegionKaomoji import train_classifier as tc
    probs = np.array([[.9, .05, .05], [.8, .1, .1], [.3, .6, .1], [.6, .3, .1], [.1, .1, .8]])
    labels = np.array([0, 0, 1, 1, 2])
    threshold, table, met = tc.select_threshold(probs, labels, maximum_fpr=0.0)
    assert met and threshold > .6 and all(row['false_positive_rate'] == 0 for row in table if row['threshold'] >= threshold)
    hopeless = np.array([[.995, .005, 0.], [.995, .005, 0.]])
    assert tc.select_threshold(hopeless, np.array([0, 1]), maximum_fpr=0.0)[2] is False
    assert tc.LABELS['face'] == 0 and tc.LABELS['graphic'] == 2 and tc.LABELS['emoji'] == 1


def test_pipeline_configuration_binds_classifier_receipt(tmp_path, monkeypatch):
    from tools.RegionKaomoji import evaluate as e
    monkeypatch.setattr(e, 'EVALUATION_ROOT', tmp_path)
    assert e.configuration()['classifier'] is None
    path = receipt(tmp_path)
    config = e.configuration(classifier=str(path))
    assert config['classifier']['threshold'] == .7 and config['classifier']['file'] == str(path.resolve())
    common.write(tmp_path / 'config.json', config)
    reloaded = e.configuration(tmp_path / 'config.json')
    assert reloaded['classifier'] == config['classifier']
    (tmp_path / 'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError):
        e.configuration(tmp_path / 'config.json')


def test_real_onnx_classifier_round_trip(tmp_path):
    onnx = pytest.importorskip('onnx')
    from onnx import helper, TensorProto, numpy_helper
    weight = numpy_helper.from_array(np.zeros((3, cl.INPUT_HEIGHT * cl.INPUT_WIDTH), dtype=np.float32), 'w')
    bias = numpy_helper.from_array(np.array([2., 0., -1.], dtype=np.float32), 'b')
    shape = numpy_helper.from_array(np.array([-1, cl.INPUT_HEIGHT * cl.INPUT_WIDTH], dtype=np.int64), 'shape')
    graph = helper.make_graph([helper.make_node('Reshape', ['input', 'shape'], ['flat']), helper.make_node('Gemm', ['flat', 'w', 'b'], ['logits'], transB=1)],
                              'stub', [helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', 1, cl.INPUT_HEIGHT, cl.INPUT_WIDTH])],
                              [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 3])], initializer=[weight, bias, shape])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, tmp_path / 'model.onnx')
    path = receipt(tmp_path)
    value = json.loads(path.read_text(encoding='utf-8'))
    value['model']['sha256'] = cl.digest(tmp_path / 'model.onnx')
    common.write(path, value)
    loaded = cl.CropClassifier(path)
    evidence = loaded.evidence(np.full((20, 40, 3), 255, dtype=np.uint8))
    assert evidence['face_evidence'] and evidence['face_probability'] > .7 and set(evidence['probabilities']) == set(cl.CLASSES)
