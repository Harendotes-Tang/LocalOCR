"""Observe pinned RapidOCR first-pass geometry without performing OCR or warps.

Only actual horizontal crops whose reading survives native filtering, production
rereads and the unsplit parent map become positional hints. Instance wrappers
borrow images during calls and retain bounded digests/compact paths, never image
copies or probability tensors. All uncertain lineage falls back to no initial row.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
import time

import cv2
import numpy as np

from tools.RegionKaomoji import ctc_trace as ct
from tools.RegionKaomoji.baseline_trace import BaselineTraceAdapter

GEOMETRY = 'first-pass-horizontal-v1'
_METHODS = ('preprocess_img', 'detect_and_crop', 'crop_text_regions',
            'cls_and_rotate', 'recognize_txt', 'build_final_output')


def _rect(points, size):
    return [max(0, math.floor(float(points[:, 0].min()))),
            max(0, math.floor(float(points[:, 1].min()))),
            min(size[0], math.ceil(float(points[:, 0].max()))),
            min(size[1], math.ceil(float(points[:, 1].max())))]


def geometry_for_crop(quad, shape, operations, image_size):
    """Reconstruct the pinned crop matrix from observed native quad and output.

    ``operations`` is the actual mutated preprocessing record passed to native
    detection. No image resampling occurs here. Mild horizontal perspective is
    supported; portrait/rotated, clipped or nearly singular quads are rejected.
    """
    if set(operations) != {'preprocess', 'padding_1'}:
        raise ct.UnsupportedTrace('initial-unknown-preprocess-operations')
    ratios = operations['preprocess']
    padding = operations['padding_1']
    if set(ratios) != {'ratio_h', 'ratio_w'} or set(padding) != {'top', 'left'}:
        raise ct.UnsupportedTrace('initial-unknown-preprocess-fields')
    rw, rh = ratios['ratio_w'], ratios['ratio_h']
    left, top = padding['left'], padding['top']
    if (any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (rw, rh))
            or any(type(x) is not int or x < 0 for x in (left, top))):
        raise ct.UnsupportedTrace('initial-invalid-preprocess')
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ct.UnsupportedTrace('initial-invalid-quad')
    width = int(max(np.linalg.norm(points[0]-points[1]), np.linalg.norm(points[2]-points[3])))
    height = int(max(np.linalg.norm(points[0]-points[3]), np.linalg.norm(points[1]-points[2])))
    if (width < 2 or height < 2 or width < height or list(shape) != [height, width]
            or width * height > 1_000_000):
        raise ct.UnsupportedTrace('initial-rotated-or-invalid-crop')
    # Fixed orientation criteria, independent of labels or candidate score.
    for a, b in ((points[0], points[1]), (points[3], points[2])):
        if b[0] <= a[0] or abs(float(b[1]-a[1])) > float(b[0]-a[0]) * .15:
            raise ct.UnsupportedTrace('initial-nonhorizontal-quad')
    if (not cv2.isContourConvex(points)
            or any(points[b, 1] <= points[a, 1] for a, b in ((0, 3), (1, 2)))):
        raise ct.UnsupportedTrace('initial-reversed-quad')
    standard = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    matrix = cv2.getPerspectiveTransform(points, standard)
    normal = (points - points.mean(axis=0)) / max(width, height)
    normalized = cv2.getPerspectiveTransform(normal.astype(np.float32),
                                             (standard / max(width, height)).astype(np.float32))
    if not np.isfinite(matrix).all() or np.linalg.cond(normalized) > 1e6:
        raise ct.UnsupportedTrace('initial-singular-perspective')
    try:
        inverse = np.asarray([[rw, 0., -left*rw], [0., rh, -top*rh], [0., 0., 1.]]) @ np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ct.UnsupportedTrace('initial-singular-perspective') from error
    inverse /= inverse[2, 2]
    original = points.copy()
    original[:, 0] = (original[:, 0] - left) * rw
    original[:, 1] = (original[:, 1] - top) * rh
    if (not np.isfinite(inverse).all() or np.any(original < 0)
            or np.any(original[:, 0] > image_size[0]) or np.any(original[:, 1] > image_size[1])):
        raise ct.UnsupportedTrace('initial-clipped-original-quad')
    return {'geometry_kind': GEOMETRY, 'rotation': 0, 'selected_view': 'initial',
            'inverse_perspective': inverse.tolist(), 'detector_quad': points.tolist(),
            'operations': {'preprocess': dict(ratios), 'padding_1': dict(padding)},
            'original_size': list(image_size), 'input_shape': [height, width],
            'padding': [0, 0, 0, 0], 'crop_bbox': _rect(original, image_size)}


def validate_geometry(value):
    """Recheck every transform term before compact paths or sidecars are used."""
    if value.get('rotation') != 0 or value.get('selected_view') != 'initial':
        raise ct.UnsupportedTrace('initial-unsupported-view-or-rotation')
    size = value.get('original_size')
    if (not isinstance(size, list) or len(size) != 2
            or any(type(x) is not int or x <= 0 for x in size)):
        raise ct.UnsupportedTrace('initial-invalid-original-size')
    expected = geometry_for_crop(value.get('detector_quad'), value.get('input_shape'),
                                 value.get('operations', {}), size)
    for key in ('inverse_perspective', 'crop_bbox', 'padding'):
        if value.get(key) != expected[key]:
            raise ct.UnsupportedTrace('initial-transform-binding-mismatch')
    if type(value.get('detector_index')) is not int or value['detector_index'] < 0:
        raise ct.UnsupportedTrace('initial-missing-detector-index')
    if value.get('classification', {}).get('rotated') is not False:
        raise ct.UnsupportedTrace('initial-unbound-classification')


def horizontal_extent(geometry, start, end):
    """Inverse-map the full time-bin strip; this is not a character box."""
    height = geometry['input_shape'][0]
    points = np.asarray([[start, 0, 1], [end, 0, 1], [end, height, 1], [start, height, 1]], float)
    mapped = points @ np.asarray(geometry['inverse_perspective'], float).T
    if (not np.isfinite(mapped).all() or np.any(mapped[:, 2] <= 1e-9)):
        raise ct.UnsupportedTrace('initial-unbounded-time-strip')
    xs = mapped[:, 0] / mapped[:, 2]
    left, _, right, _ = geometry['crop_bbox']
    low, high = max(float(left), float(xs.min())), min(float(right), float(xs.max()))
    if not left <= low < high <= right:
        raise ct.UnsupportedTrace('initial-time-strip-outside-parent')
    return low, high


class InitialTraceAdapter(BaselineTraceAdapter):
    """Optional first-pass observer sharing the existing total trace budgets."""

    def __init__(self, engine, **kwargs):
        from rapidocr import RapidOCR
        if any(getattr(getattr(engine, name, None), '__func__', None) is not getattr(RapidOCR, name)
               for name in _METHODS):
            raise ValueError('Initial observer requires the pinned unmodified RapidOCR methods')
        super().__init__(engine._load_rec_model(), **kwargs)
        self.engine = engine
        from rapidocr.utils import process_img
        for role, value in (('engine', type(engine)), ('crop_transform', process_img.get_rotate_crop_image),
                            ('base_adapter', BaselineTraceAdapter)):
            path = Path(inspect.getfile(value)).resolve()
            self.identity['source_files'][role] = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if len(ct.encoded(self.identity)) > 65536:
            raise ValueError('Initial CTC identity exceeds its bounded header')
        self._initial_paths, self._production_paths = {}, {}
        self._initial_count = 0
        self._replaced, self._removed = set(), set()
        self._initial_rows_bound = 0
        self._initial_active = False
        self._initial_started = False
        self._descriptors, self._classified = [], []

    def _safe(self, callback, *args, **kwargs):
        at = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        except Exception:
            self._unsupported['initial-observation-failure'] += 1
            self._initial_paths.clear()
            self._production_paths.clear()
            self._paths = [path for path in self._paths if path['geometry'].get('geometry_kind') != GEOMETRY]
            self._initial_rows_bound = 0
            self._descriptors, self._classified = [], []
            self._initial_failed = True
            return None
        finally:
            self._seconds += time.perf_counter() - at

    @contextmanager
    def observe_initial(self, pixels):
        """Wrap only the original engine call; rereads have a separate scope."""
        if self._initial_active or self._initial_started or not self._active:
            raise ValueError('Initial observation requires one active nonnested adapter')
        self._initial_started = True
        saved = {}
        self._initial_active, self._initial_failed = True, False
        self._original_digest = self._pixel_digest(pixels)
        self._operation_record = None
        self._initial_paths.clear()
        self._production_paths.clear()
        self._replaced.clear()
        self._removed.clear()
        self._initial_count = self._initial_rows_bound = 0
        self._bound = False

        def wrap(name, before=None, after=None):
            original = getattr(self.engine, name)
            saved[name] = (name in vars(self.engine), original)
            def observed(*args, **kwargs):
                if before:
                    self._safe(before, *args, **kwargs)
                # Native exceptions propagate unchanged and are never swallowed.
                result = original(*args, **kwargs)
                if after:
                    self._safe(after, result, *args, **kwargs)
                return result
            setattr(self.engine, name, observed)

        try:
            wrap('preprocess_img', after=self._preprocessed)
            wrap('detect_and_crop', before=self._detecting)
            wrap('crop_text_regions', after=self._cropped)
            wrap('cls_and_rotate', after=self._classified_images)
            wrap('recognize_txt', before=self._recognizing, after=self._recognized)
            wrap('build_final_output', before=self._before_final, after=self._after_final)
            yield
        finally:
            for name, (had, original) in reversed(list(saved.items())):
                if had:
                    setattr(self.engine, name, original)
                else:
                    delattr(self.engine, name)
            self._sample_map.clear()
            self._reread_paths.clear()
            self._pending.clear()
            self._descriptors, self._classified = [], []
            self._final_expected = None
            self._detector_paths = {}
            self._operation_record = None
            self._initial_active = False
            if self._initial_failed:
                self._initial_paths.clear()

    def _preprocessed(self, result, image):
        if list(image.shape[:2]) != self.image_size[::-1] or self._pixel_digest(image) != self._original_digest:
            raise ct.UnsupportedTrace('initial-original-image-mismatch')
        self._preprocess_shape = list(result[0].shape[:2])
        ratios = result[1].get('preprocess', {})
        if (ratios.get('ratio_h') != image.shape[0] / result[0].shape[0]
                or ratios.get('ratio_w') != image.shape[1] / result[0].shape[1]):
            raise ct.UnsupportedTrace('initial-preprocess-ratio-mismatch')
        self._preprocess_record = ct.encoded(result[1])

    def _detecting(self, image, operations):
        if list(image.shape[:2]) != self._preprocess_shape or ct.encoded(operations) != self._preprocess_record:
            raise ct.UnsupportedTrace('initial-preprocess-lineage-mismatch')
        self._operation_record = operations

    def _cropped(self, crops, image, boxes):
        if self._initial_failed:
            return
        if len(crops) != len(boxes) or len(crops) > self.limits.max_rows:
            self._unsupported['initial-row-cap-or-count'] += 1
            return
        descriptors = []
        for index, (crop, quad) in enumerate(zip(crops, boxes)):
            try:
                geometry = geometry_for_crop(quad, crop.shape[:2], self._operation_record, self.image_size)
                geometry.update(detector_index=index, source_indices=[index], before_text='',
                                input_pixel_sha256=self._pixel_digest(crop),
                                classification={'enabled': False, 'rotated': False})
                descriptors.append(geometry)
            except ct.UnsupportedTrace as error:
                self._unsupported[str(error)] += 1
                descriptors.append(None)
        self._descriptors = descriptors
        self._classified = descriptors if not self.engine.use_cls else []

    def _classified_images(self, result, images):
        output, classification = result
        if len(images) != len(self._descriptors) or len(output) != len(images) or len(classification.cls_res) != len(images):
            raise ct.UnsupportedTrace('initial-classification-count-mismatch')
        threshold = float(self.engine._load_cls_model().cls_thresh)
        self._classified = []
        for descriptor, image, item in zip(self._descriptors, output, classification.cls_res):
            if descriptor is None:
                self._classified.append(None)
                continue
            label, score = str(item[0]), float(item[1])
            rotated = '180' in label and score > threshold
            if (not math.isfinite(score) or rotated or list(image.shape[:2]) != descriptor['input_shape']
                    or self._pixel_digest(image) != descriptor['input_pixel_sha256']):
                self._unsupported['initial-classified-rotation-or-pixels'] += 1
                self._classified.append(None)
                continue
            self._classified.append(dict(descriptor, classification={
                'enabled': True, 'label': label, 'score': score, 'threshold': threshold, 'rotated': False}))

    def _recognizing(self, images):
        self._sample_map.clear()
        self._reread_paths.clear()
        if self._initial_failed or len(images) != len(self._classified):
            return
        order = np.argsort(np.asarray([image.shape[1] / image.shape[0] for image in images]))
        positions = {int(index): at for at, index in enumerate(order)}
        for index, (image, descriptor) in enumerate(zip(images, self._classified)):
            if descriptor is None:
                continue
            if (id(image) in self._sample_map or list(image.shape[:2]) != descriptor['input_shape']
                    or self._pixel_digest(image) != descriptor['input_pixel_sha256']):
                raise ct.UnsupportedTrace('initial-recognition-image-mismatch')
            self._sample_map[id(image)] = dict(descriptor, input_index=index, sorted_index=positions[index])
        self._scope_sort_index = self._batch_ordinal = 0

    def _recognized(self, result, images):
        self._sample_map.clear()
        if self._initial_failed or len(images) != len(result.txts):
            return
        self._detector_paths = {}
        for index, path in self._reread_paths.items():
            if path['text'] == result.txts[index]:
                self._detector_paths[index] = path
            else:
                self._unsupported['initial-final-recognition-mismatch'] += 1
        self._reread_paths.clear()

    def _before_final(self, image, detection, classification, recognition, crops, operations):
        self._final_expected = None
        if self._initial_failed or detection.boxes is None or recognition.txts is None:
            return
        if len(detection.boxes) > self.limits.max_rows or len(detection.boxes) != len(recognition.txts):
            return
        from rapidocr.utils.process_img import map_boxes_to_original
        boxes = map_boxes_to_original(detection.boxes.copy(), operations, image.shape[0], image.shape[1])
        self._final_expected = [(i, box.tolist(), text, float(score))
                                for i, (box, text, score) in enumerate(zip(boxes, recognition.txts, recognition.scores))
                                if text.strip() and float(score) >= self.engine.text_score]

    def _after_final(self, result, *_):
        expected = self._final_expected
        if expected is None or getattr(result, 'boxes', None) is None:
            return
        if not len(result.boxes) == len(result.txts) == len(result.scores) == len(expected):
            raise ct.UnsupportedTrace('initial-filter-count-mismatch')
        self._initial_count = len(expected)
        for output_index, ((source_index, box, text, score), actual_box, actual_text, actual_score) in enumerate(
                zip(expected, result.boxes, result.txts, result.scores)):
            if box != np.asarray(actual_box).tolist() or text != actual_text or score != float(actual_score):
                raise ct.UnsupportedTrace('initial-filter-lineage-mismatch')
            path = getattr(self, '_detector_paths', {}).get(source_index)
            if path is None:
                continue
            if path['geometry']['crop_bbox'] != _rect(np.asarray(box), self.image_size):
                self._unsupported['initial-filter-box-mismatch'] += 1
                continue
            if self._grow_path(path, dict(initial_output_index=output_index, before_text=text,
                                         before_text_sha256=ct.text_digest(text), initial_score=score)):
                self._initial_paths[output_index] = path

    def _grow_path(self, path, updates):
        """Charge final provenance to the same compact byte/record allowance."""
        previous = path['geometry']
        old_size = len(ct.encoded(path))
        path['geometry'] = dict(previous, **updates)
        size = len(ct.encoded(path))
        delta = max(0, size-old_size)
        reserve = len(ct.encoded(self.identity)) + 1024
        if size > self.limits.max_record_bytes or self._bytes + delta + reserve > self.limits.max_total_bytes:
            path['geometry'] = previous
            self._unsupported['initial-final-lineage-byte-cap'] += 1
            return False
        self._bytes += delta
        return True

    def observe_production_reread(self, event, payload):
        """Invalidate original identities even when replacement text is equal."""
        if event == 'selection' and payload.get('applied'):
            indices = payload.get('source_indices', [])
            if (not isinstance(indices, list) or not indices or len(indices) > self.limits.max_rows
                    or any(type(i) is not int or not 0 <= i < self._initial_count for i in indices)):
                self._initial_paths.clear()
                self._unsupported['initial-reread-lineage-mismatch'] += 1
            else:
                self._replaced.update(indices)
                self._removed.update(indices[1:])
        self.observe_reread(event, payload)

    def finish_production(self, entries):
        """Bind the surviving initial indices to actual production output slots."""
        self._production_paths.clear()
        original_indices = [i for i in range(self._initial_count) if i not in self._removed]
        if len(original_indices) != len(entries):
            self._unsupported['initial-production-count-mismatch'] += 1
            return
        for index, source in enumerate(original_indices):
            path = self._initial_paths.get(source)
            if path is None or source in self._replaced:
                continue
            entry, geometry = entries[index], path['geometry']
            if (entry['text'] != path['text'] or entry['bbox'] != geometry['crop_bbox']
                    or entry['score'] != geometry['initial_score']):
                self._unsupported['initial-production-reading-mismatch'] += 1
                continue
            if self._grow_path(path, {'production_parent_index': index}):
                self._production_paths[index] = path

    def bind_initial_entries(self, entries, plan):
        """Only explicit unsplit mappings retain first-pass paths."""
        if self._bound:
            return
        self._bound = True
        mapping = plan.get('map') if plan.get('parents') else {str(i): i for i in range(len(entries))}
        if (not isinstance(mapping, dict) or len(mapping) > 4096
                or any(type(value) is not int for value in mapping.values())
                or len(set(mapping.values())) != len(mapping)):
            self._unsupported['initial-missing-unsplit-map'] += 1
            return
        for key, production_index in mapping.items():
            index = int(key)
            if not 0 <= index < len(entries) or key in plan.get('children', {}):
                continue
            path = self._production_paths.get(production_index)
            if path is None:
                continue
            parent = entries[index]
            if (parent['bbox'] != path['geometry']['crop_bbox'] or parent['text'] != path['text']
                    or parent['score'] != path['geometry']['initial_score']):
                self._unsupported['initial-unsplit-reading-mismatch'] += 1
                continue
            if not self._grow_path(path, dict(final_parent_index=index, source_indices=[production_index],
                selected_parent={'index': index, 'bbox': list(parent['bbox']),
                                 'text': parent['text'], 'score': parent['score']})):
                continue
            self._paths.append(path)
            self._initial_rows_bound += 1

    def receipt(self):
        return dict(super().receipt(), initial_rows_bound=self._initial_rows_bound)
