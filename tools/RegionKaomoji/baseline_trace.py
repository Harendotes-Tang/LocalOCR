"""Opt-in observation of pinned RapidOCR's already executed CTC forwards.

The instance-local wrappers call native preprocessing/postprocessing exactly
once. Initial detector/perspective/classifier mappings and synthetic child
inputs are unsupported; actual axis-aligned worker rereads carry explicit
request and selected-view lineage. No ONNX session or inference is created.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import time

import numpy as np

from tools.RegionKaomoji import ctc_trace as ct


class _ObservedDecoder:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, prediction, *args, **kwargs):
        result = self.original(prediction, *args, **kwargs)
        started = time.perf_counter()
        try:
            self.owner._capture(prediction, result)
        except Exception:
            self.owner._unsupported['observer-capture-failure'] += 1
        finally:
            self.owner._pending.clear()
            self.owner._batch_ordinal += 1
            self.owner._seconds += time.perf_counter() - started
        return result


class BaselineTraceAdapter:
    """A single-image, non-reentrant context around an existing recognizer.

    ``observe_reread`` is the optional worker callback. ``known_crops`` is for
    callers that already possess exact, unmodified, axis-aligned source crops;
    it must never describe painted-out split inputs or guessed initial boxes.
    ``sidecar(entries)`` binds only uniquely surviving final baseline readings.
    Collection statistics are deterministic; measured observer time is separate
    in ``receipt()`` so repeat quality payloads remain comparable.
    """

    def __init__(self, recognizer, *, image_sha256, image_size, identity, limits=None):
        self.recognizer = recognizer
        self.limits = limits or ct.TraceLimits()
        if (not isinstance(image_sha256, str) or len(image_sha256) != 64
                or any(c not in '0123456789abcdef' for c in image_sha256)
                or len(image_size) != 2 or any(type(x) is not int or x <= 0 for x in image_size)):
            raise ValueError('CTC observation requires exact original image binding')
        if importlib.metadata.version('rapidocr') != '3.10.0':
            raise ValueError('CTC observer supports the pinned RapidOCR 3.10.0 interface only')
        self.image_sha256, self.image_size = image_sha256, list(image_size)
        self._characters = tuple(recognizer.postprocess_op.character)
        files = {}
        for role, value in (('recognizer', type(recognizer)),
                            ('decoder', type(recognizer.postprocess_op)), ('adapter', type(self)),
                            ('trace', ct.compact_path)):
            path = Path(inspect.getfile(value)).resolve()
            files[role] = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.identity = {'baseline': identity, 'runtime': 'rapidocr-3.10.0',
                         'dictionary_sha256': ct.digest(self._characters),
                         'dictionary_size': len(self._characters), 'source_files': files,
                         'rec_image_shape': list(recognizer.rec_image_shape),
                         'rec_batch_num': int(recognizer.rec_batch_num)}
        if len(ct.encoded(self.identity)) > 65536:
            raise ValueError('CTC identity exceeds its bounded header')
        self._paths, self._pending, self._sample_map = [], [], {}
        self._reread_paths = {}
        self._known_order = None
        self._scope_sort_index = self._batch_ordinal = 0
        self._unsupported = Counter()
        self._rows = self._steps = self._batches = self._bytes = 0
        self._seconds = 0.
        self._active = False

    def __enter__(self):
        if self._active or isinstance(self.recognizer.postprocess_op, _ObservedDecoder):
            raise ValueError('CTC observers cannot nest or share a recognizer')
        self._active = True
        self._original_decoder = self.recognizer.postprocess_op
        self._original_resize = self.recognizer.resize_norm_img
        self._had_resize = 'resize_norm_img' in vars(self.recognizer)
        self.recognizer.postprocess_op = _ObservedDecoder(self._original_decoder, self)

        def observed_resize(image, max_wh_ratio):
            output = self._original_resize(image, max_wh_ratio)
            started = time.perf_counter()
            try:
                self._record_resize(image, output, max_wh_ratio)
            except Exception:
                self._unsupported['observer-resize-failure'] += 1
                self._pending.append(None)
            finally:
                self._seconds += time.perf_counter() - started
            return output

        self.recognizer.resize_norm_img = observed_resize
        return self

    def __exit__(self, *_):
        self.recognizer.postprocess_op = self._original_decoder
        if self._had_resize:
            self.recognizer.resize_norm_img = self._original_resize
        else:
            del self.recognizer.resize_norm_img
        self._active = False
        self._pending.clear()
        self._sample_map.clear()
        self._reread_paths.clear()
        self._known_order = None

    @contextmanager
    def known_crops(self, descriptors):
        """Bind only existing unmodified original axis crops in their input order.

        Used for explicit source-crop mechanisms. The production split helper
        paints sibling spans, so its child forward is deliberately not bound.
        """
        if self._known_order is not None or self._sample_map:
            raise ValueError('Known crop observation cannot overlap rereads')
        if len(descriptors) > self.limits.max_rows:
            self._unsupported['row-cap'] += len(descriptors)
            yield
            return
        bound = []
        for index, descriptor in enumerate(descriptors):
            bbox = descriptor['bbox']
            bound.append(dict(descriptor, crop_bbox=list(bbox), input_index=index,
                              selected_view='axis-crop', padding=[0, 0, 0, 0],
                              input_shape=[bbox[3] - bbox[1], bbox[2] - bbox[0]],
                              before_text=descriptor.get('before_text', '')))
        ratios = [item['input_shape'][1] / item['input_shape'][0] for item in bound]
        self._known_order = [bound[int(i)] for i in np.argsort(np.asarray(ratios))]
        self._scope_sort_index = self._batch_ordinal = 0
        try:
            yield
        finally:
            self._known_order = None
            self._pending.clear()

    def observe_reread(self, event, payload):
        """Receive real worker requests and its actual selected vote provenance.

        Observer failures must not affect OCR output. Samples are borrowed only
        while recording their object IDs; tensors/images are never retained.
        """
        started = time.perf_counter()
        try:
            if event == 'requests':
                self._sample_map.clear()
                self._reread_paths.clear()
                samples, descriptors = payload['samples'], payload['descriptors']
                if len(samples) != len(descriptors) or len(samples) > self.limits.max_rows:
                    self._unsupported['row-cap'] += max(1, len(samples))
                    return
                indices = np.argsort(np.asarray([sample.shape[1] / sample.shape[0] for sample in samples]))
                sorted_positions = {int(index): position for position, index in enumerate(indices)}
                for index, (sample, descriptor) in enumerate(zip(samples, descriptors)):
                    if id(sample) in self._sample_map:
                        self._sample_map.clear()
                        self._unsupported['ambiguous-sample-identity'] += 1
                        return
                    if (len(descriptor.get('before_text', '')) > self.limits.max_record_bytes
                            or sample.shape[0] * sample.shape[1] > 1_000_000):
                        self._unsupported['input-descriptor-cap'] += 1
                        continue
                    self._sample_map[id(sample)] = dict(descriptor, input_index=index,
                                                        sorted_index=sorted_positions[index],
                                                        input_shape=list(sample.shape[:2]),
                                                        input_pixel_sha256=self._pixel_digest(sample))
                self._scope_sort_index = self._batch_ordinal = 0
            elif event == 'selection':
                if not payload.get('applied'):
                    return
                if payload.get('synthetic') or len(payload.get('source_indices', [])) != 1:
                    self._unsupported['synthetic-or-merged-reread'] += 1
                    return
                path = self._reread_paths.get(payload.get('selected_sample_index'))
                if path is None or path['text'] != payload.get('selected_text'):
                    self._unsupported['unbound-selected-reread'] += 1
                    return
                if (path['geometry']['source_indices'] != payload['source_indices']
                        or path['geometry']['before_text'] != payload.get('before_text')
                        or path['geometry']['crop_bbox'] != payload.get('bbox')):
                    self._unsupported['reread-lineage-mismatch'] += 1
                    return
                self._paths.append(path)
            elif event == 'complete':
                self._sample_map.clear()
                self._reread_paths.clear()
        except Exception:
            self._unsupported['observer-event-failure'] += 1
        finally:
            self._seconds += time.perf_counter() - started

    def _record_resize(self, image, output, max_wh_ratio):
        if len(self._pending) >= self.limits.max_rows:
            self._unsupported['row-cap'] += 1
            return
        descriptor = self._sample_map.get(id(image))
        if descriptor is not None and descriptor['sorted_index'] != self._scope_sort_index:
            self._unsupported['native-sorted-order-mismatch'] += 1
            self._scope_sort_index += 1
            self._pending.append(None)
            return
        if descriptor is None and self._known_order is not None:
            if self._scope_sort_index < len(self._known_order):
                descriptor = dict(self._known_order[self._scope_sort_index],
                                  sorted_index=self._scope_sort_index)
                if image.shape[0] * image.shape[1] > 1_000_000:
                    self._unsupported['input-descriptor-cap'] += 1
                    self._pending.append(None)
                    return
                descriptor['input_pixel_sha256'] = self._pixel_digest(image)
        self._scope_sort_index += 1
        if descriptor is None:
            self._unsupported['unbound-input-transform'] += 1
            self._pending.append(None)
            return
        if list(image.shape[:2]) != descriptor['input_shape']:
            self._unsupported['input-shape-mismatch'] += 1
            self._pending.append(None)
            return
        height = int(self.recognizer.rec_image_shape[1])
        width = int(height * max_wh_ratio)
        if output.shape != (3, height, width):
            self._unsupported['native-normalized-shape-mismatch'] += 1
            self._pending.append(None)
            return
        geometry = dict(descriptor, content_width=min(width, math.ceil(height * image.shape[1] / image.shape[0])),
                        batch_width=width, resize_height=height,
                        batch_index=len(self._pending), batch_ordinal=self._batch_ordinal)
        if descriptor.get('geometry_kind') != 'first-pass-horizontal-v1':
            geometry.update(rotation=0, inverse_perspective=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        geometry['before_text_sha256'] = ct.text_digest(geometry.get('before_text', ''))
        geometry.pop('bbox', None)
        self._pending.append(geometry)

    @staticmethod
    def _pixel_digest(image):
        """Hash borrowed pixels a row at a time, without an image-size copy."""
        digest = hashlib.sha256()
        for row in image:
            digest.update(memoryview(np.ascontiguousarray(row)))
        return digest.hexdigest()

    def _capture(self, prediction, native):
        self._batches += 1
        if (not isinstance(prediction, np.ndarray) or prediction.ndim != 3
                or prediction.shape[0] != len(self._pending) or len(native[0]) != len(self._pending)):
            self._unsupported['native-batch-binding-mismatch'] += 1
            return
        for index, geometry in enumerate(self._pending):
            if geometry is None:
                continue
            if self._rows >= self.limits.max_rows:
                self._unsupported['row-cap'] += 1
                continue
            steps = prediction.shape[1]
            if steps > self.limits.max_steps_per_row or self._steps + steps > self.limits.max_total_steps:
                self._unsupported['step-cap'] += 1
                continue
            remaining = self.limits.max_total_bytes - self._bytes - len(ct.encoded(self.identity)) - 1024
            if remaining <= 0:
                self._unsupported['total-byte-cap'] += 1
                continue
            self._rows += 1
            self._steps += steps
            try:
                bounded = replace(self.limits, max_record_bytes=min(self.limits.max_record_bytes, remaining))
                path = ct.compact_path(prediction[index], self._characters, geometry, bounded)
                if path['text'] != native[0][index][0]:
                    raise ct.UnsupportedTrace('native-decoding-mismatch')
                size = len(ct.encoded(path))
                self._bytes += size
                if self._known_order is None:
                    self._reread_paths[geometry['input_index']] = path
                else:
                    self._paths.append(path)
            except ct.UnsupportedTrace as error:
                self._unsupported[str(error)] += 1

    def sidecar(self, entries):
        started = time.perf_counter()
        try:
            # Identity and original image bounds were fixed before native calls.
            value = ct.make_sidecar(self.image_sha256, self.image_size, self.identity,
                                    self._paths, entries, self._stats(), self.limits)
            return value
        except ct.UnsupportedTrace as error:
            self._unsupported[str(error)] += 1
            return None
        finally:
            self._seconds += time.perf_counter() - started

    def _stats(self):
        return {'observed_batches': self._batches, 'rows_attempted': self._rows,
                'steps_observed': self._steps, 'compact_bytes': self._bytes,
                'unsupported': dict(sorted(self._unsupported.items()))}

    def receipt(self):
        return dict(self._stats(), observer_seconds=self._seconds, limits=asdict(self.limits),
                    extra_forwards=0, full_probability_tensor_persisted=False)
