"""Crop classifier evidence for routing: ONNX CPU inference and bound metadata only.

The classifier never writes text. Its face probability is one more piece of
independent pixel evidence beside paired bracket shapes; the threshold is
selected on development data and frozen with the routing configuration.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

SCHEMA = 'localocr.crop-classifier.v1'
CLASSES = ('face', 'bracketed-text', 'graphic')
INPUT_HEIGHT, INPUT_WIDTH = 32, 128
PREPROCESSING = {'height': INPUT_HEIGHT, 'width': INPUT_WIDTH, 'color': 'gray-ink-magnitude',
                 'resize': 'cv2-area-or-linear', 'aspect': 'preserved-then-right-zero-padded',
                 'normalize': 'abs(gray/255 - border-median)'}


def preprocess(image):
    """Grayscale ink magnitude on a fixed canvas; dark-on-light and light-on-dark crops look alike."""
    array = np.asarray(image)
    if array.size == 0 or array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError('Classifier crop must be a nonempty uint8 image')
    if array.ndim == 3:
        array = cv2.cvtColor(np.ascontiguousarray(array[:, :, :3]), cv2.COLOR_RGB2GRAY)
    gray = array.astype(np.float32) / 255.0
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    ink = np.abs(gray - float(np.median(border)))
    height, width = ink.shape
    scale = INPUT_HEIGHT / height
    target_width = max(1, min(INPUT_WIDTH, int(round(width * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(ink, (target_width, INPUT_HEIGHT), interpolation=interpolation)
    canvas = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    canvas[:, :target_width] = resized
    return canvas[None]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classifier_binding(path):
    """Validate the classifier receipt and return the bound identity used by pipeline configurations."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    meta = json.loads(raw.decode('utf-8'))
    if (meta.get('schema_version') != SCHEMA or not isinstance(meta.get('id'), str) or not meta['id']
            or tuple(meta.get('classes', ())) != CLASSES or meta.get('preprocessing') != PREPROCESSING
            or meta.get('promote', False) or meta.get('enabled', False)):
        raise ValueError('Unknown or non-research crop classifier receipt')
    threshold = meta.get('threshold')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Classifier threshold must be a finite probability inside (0,1)')
    model = meta.get('model', {})
    model_path = path.parent / str(model.get('file', ''))
    if not model_path.is_file() or not model_path.resolve().is_relative_to(path.parent) or digest(model_path) != model.get('sha256'):
        raise ValueError('Classifier model bytes differ from the receipt')
    training = meta.get('training', {})
    if not isinstance(training, dict) or training.get('threshold_selected_on') != 'dev' or training.get('heldout_consumed') is not False:
        raise ValueError('Classifier threshold must come from development data only')
    return {'id': meta['id'], 'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'model_sha256': model['sha256'], 'model_bytes': model_path.stat().st_size,
            'threshold': float(threshold), 'classes': list(CLASSES), 'dataset_sha256': training.get('dataset_sha256')}


class CropClassifier:
    def __init__(self, path):
        import onnxruntime as ort
        self.binding = classifier_binding(path)
        meta = json.loads(Path(path).read_bytes().decode('utf-8'))
        model_bytes = (Path(path).parent / meta['model']['file']).read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != self.binding['model_sha256']:
            raise ValueError('Classifier model snapshot changed')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_bytes, options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        dynamic = lambda value: value is None or isinstance(value, str)
        if (self.session.get_providers() != ['CPUExecutionProvider'] or len(inputs) != 1 or len(outputs) != 1
                or inputs[0].shape[1:] != [1, INPUT_HEIGHT, INPUT_WIDTH] or not dynamic(inputs[0].shape[0])
                or len(outputs[0].shape) != 2 or outputs[0].shape[1] != len(CLASSES)):
            raise ValueError('Crop classifier ONNX must take [N,1,32,128] and emit [N,3] logits')
        self.input_name = inputs[0].name
        self.threshold = self.binding['threshold']
        probe = self.session.run(None, {self.input_name: np.zeros((2, 1, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)})[0]
        if probe.shape != (2, len(CLASSES)) or not np.isfinite(probe).all():
            raise ValueError('Crop classifier probe failed')

    def probabilities(self, images):
        batch = np.stack([preprocess(image) for image in images]).astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def evidence(self, image):
        """Routing evidence for one crop; probabilities are recorded, the decision is a threshold on face."""
        probs = self.probabilities([image])[0]
        face = float(probs[0])
        if not math.isfinite(face):
            raise ValueError('Nonfinite classifier probability')
        return {'id': self.binding['id'], 'threshold': self.threshold,
                'probabilities': {name: float(value) for name, value in zip(CLASSES, probs)},
                'face_probability': face, 'face_evidence': face >= self.threshold}
