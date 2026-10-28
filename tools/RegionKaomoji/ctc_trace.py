"""Bounded CTC paths and live, image-bound horizontal research clues.

These intervals describe recognizer time bins, never measured glyph boxes.
Only an in-process observer may mint a sidecar; deserialized predictions are
evidence records, not authority to create new inference proposals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import secrets

import numpy as np

SCHEMA = 'localocr.baseline-ctc-sidecar.v1'
PATH_SCHEMA = 'localocr.compact-ctc-path.v1'
_AUTH_KEY = secrets.token_bytes(32)
_IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


@dataclass(frozen=True)
class TraceLimits:
    """Pre-score per-image caps; none authorizes an extra model invocation."""

    max_rows: int = 256
    max_steps_per_row: int = 2048
    max_total_steps: int = 32768
    max_record_bytes: int = 32768
    max_total_bytes: int = 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError('CTC caps must be positive integers')
        defaults = (256, 2048, 32768, 32768, 1024 * 1024)
        if any(value > bound for value, bound in zip(asdict(self).values(), defaults)):
            raise ValueError('CTC caps cannot exceed the declared research limits')


class UnsupportedTrace(ValueError):
    """A trace cannot be collected or mapped within its declared contract."""


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _geometry(value):
    """Validate the supported axis crop, including every resize/padding term."""
    if not isinstance(value, dict):
        raise UnsupportedTrace('missing-geometry')
    value = json.loads(encoded(value))
    initial = value.get('geometry_kind') == 'first-pass-horizontal-v1'
    if initial:
        from tools.RegionKaomoji.initial_trace import validate_geometry
        validate_geometry(value)
    elif value.get('rotation') != 0 or value.get('inverse_perspective') != _IDENTITY:
        raise UnsupportedTrace('unsupported-rotation-or-perspective')
    if value.get('selected_view') not in (('initial',) if initial else ('raw', 'inverse', 'padded', 'axis-crop')):
        raise UnsupportedTrace('unsupported-selected-view')
    sources = value.get('source_indices')
    if not isinstance(sources, list) or len(sources) != 1 or type(sources[0]) is not int or sources[0] < 0:
        raise UnsupportedTrace('unsupported-merged-parent')
    bbox = value.get('crop_bbox')
    shape, padding = value.get('input_shape'), value.get('padding')
    if (not isinstance(bbox, list) or len(bbox) != 4 or any(type(x) is not int for x in bbox)
            or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
            or not isinstance(shape, list) or len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)
            or not isinstance(padding, list) or len(padding) != 4
            or any(type(x) is not int or x < 0 for x in padding)):
        raise UnsupportedTrace('invalid-axis-crop')
    if not initial and shape != [bbox[3] - bbox[1] + padding[1] + padding[3],
                 bbox[2] - bbox[0] + padding[0] + padding[2]]:
        raise UnsupportedTrace('crop-shape-padding-mismatch')
    if padding != ([6, 4, 6, 4] if value['selected_view'] == 'padded' else [0, 0, 0, 0]):
        raise UnsupportedTrace('unsupported-view-padding')
    for key in ('content_width', 'batch_width', 'resize_height'):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise UnsupportedTrace('invalid-resize-binding')
    if (value['resize_height'] != 48 or value['content_width'] > value['batch_width']
            or value['content_width'] != min(value['batch_width'], math.ceil(48 * shape[1] / shape[0]))):
        raise UnsupportedTrace('resize-content-width-mismatch')
    for key in ('input_index', 'sorted_index', 'batch_index', 'batch_ordinal'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise UnsupportedTrace('missing-batch-order')
    if not isinstance(value.get('before_text', ''), str):
        raise UnsupportedTrace('invalid-parent-before-text')
    if ('before_text_sha256' in value
            and value['before_text_sha256'] != text_digest(value.get('before_text', ''))):
        raise UnsupportedTrace('parent-before-text-digest-mismatch')
    if 'source_entries' in value:
        parents = value['source_entries']
        if (not isinstance(parents, list) or len(parents) != 1
                or parents[0].get('index') != sources[0]
                or parents[0].get('text') != value.get('before_text', '')):
            raise UnsupportedTrace('source-parent-lineage-mismatch')
        box = parents[0].get('box')
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(point, list) or len(point) != 2
                       or any(type(number) not in (int, float) or not math.isfinite(number) for number in point)
                       for point in box)):
            raise UnsupportedTrace('source-parent-geometry-mismatch')
    return value


def _tokens(runs, symbols, geometry, steps):
    """Convert complete nonblank runs into conservative horizontal time bins."""
    bbox = geometry['crop_bbox']
    padding_left = geometry['padding'][0]
    view_width = geometry['input_shape'][1]
    initial = geometry.get('geometry_kind') == 'first-pass-horizontal-v1'
    crop_width = view_width if initial else bbox[2] - bbox[0]
    scale = geometry['batch_width'] * view_width / (steps * geometry['content_width'])
    tokens, text, offset = [], [], 0
    for token_id, start, end in runs:
        if token_id == 0:
            continue
        token = symbols[str(token_id)]
        raw_start, raw_end = start * scale - padding_left, end * scale - padding_left
        if raw_end <= 0 or raw_start >= crop_width:
            raise UnsupportedTrace('emitted-token-in-padding')
        low, high = max(0., raw_start), min(float(crop_width), raw_end)
        if initial:
            from tools.RegionKaomoji.initial_trace import horizontal_extent
            x_start, x_end = horizontal_extent(geometry, low, high)
        else:
            x_start, x_end = bbox[0] + low, bbox[0] + high
        tokens.append({'id': token_id, 'text': token, 'start': start, 'end': end,
                       'char_start': offset, 'char_end': offset + len(token),
                       'x_start': float(x_start), 'x_end': float(x_end),
                       'clipped_to_content': raw_start < 0 or raw_end > crop_width})
        text.append(token)
        offset += len(token)
    return ''.join(text), tokens


def compact_path(prediction, characters, geometry, limits=None):
    """Observe one existing T×C output without copying or retaining that tensor.

    Shape/step limits precede argmax allocation. Run growth reserves a strict
    byte bound before appending. Native text must subsequently be checked by
    the adapter, since this helper alone does not establish selected lineage.
    """
    limits = limits or TraceLimits()
    geometry = _geometry(geometry)
    if not isinstance(prediction, np.ndarray) or prediction.ndim != 2:
        raise UnsupportedTrace('invalid-native-output-shape')
    steps, count = prediction.shape
    if not 0 < steps <= limits.max_steps_per_row:
        raise UnsupportedTrace('step-cap')
    if not isinstance(characters, (list, tuple)) or len(characters) != count or count < 2:
        raise UnsupportedTrace('dictionary-output-mismatch')
    if characters[0] != 'blank' or any(not isinstance(token, str) for token in characters):
        raise UnsupportedTrace('unsupported-dictionary')
    # This O(T) allocation is bounded before it is created. No T×C copy exists.
    indices = prediction.argmax(axis=1)
    runs, symbols, size_bound = [], {}, len(encoded(geometry)) + 512
    for start in range(steps):
        token_id = int(indices[start])
        if not math.isfinite(float(prediction[start, token_id])):
            raise UnsupportedTrace('nonfinite-native-path')
        if runs and runs[-1][0] == token_id:
            runs[-1][2] = start + 1
            continue
        token = characters[token_id]
        # JSON escapes can expand a character six-fold. The fixed allowance
        # covers the run, derived token fields, symbol ID and float strings.
        size_bound += 384 + len(token.encode('utf-8')) * 6
        if size_bound > limits.max_record_bytes:
            raise UnsupportedTrace('record-byte-cap')
        symbols[str(token_id)] = token
        runs.append([token_id, start, start + 1])
    text, tokens = _tokens(runs, symbols, geometry, steps)
    value = {'schema_version': PATH_SCHEMA, 'steps': steps, 'dictionary_size': count,
             'geometry': geometry, 'runs': runs, 'symbols': symbols, 'text': text, 'tokens': tokens}
    if len(encoded(value)) > limits.max_record_bytes:
        raise UnsupportedTrace('record-byte-cap')
    return value


def _validated_path(value):
    if not isinstance(value, dict) or value.get('schema_version') != PATH_SCHEMA:
        raise UnsupportedTrace('invalid-path-schema')
    geometry = _geometry(value.get('geometry'))
    steps = value.get('steps')
    if type(steps) is not int or not 0 < steps <= TraceLimits().max_steps_per_row:
        raise UnsupportedTrace('invalid-path-steps')
    runs, symbols = value.get('runs'), value.get('symbols')
    if not isinstance(runs, list) or not isinstance(symbols, dict) or not runs:
        raise UnsupportedTrace('invalid-compact-runs')
    cursor, previous = 0, None
    for run in runs:
        if (not isinstance(run, list) or len(run) != 3 or any(type(x) is not int for x in run)
                or not 0 <= run[0] < value.get('dictionary_size', 0) or run[0] == previous
                or run[1] != cursor or not cursor < run[2] <= steps
                or not isinstance(symbols.get(str(run[0])), str)):
            raise UnsupportedTrace('invalid-run-order')
        previous, cursor = run[0], run[2]
    if cursor != steps or symbols.get('0', 'blank') != 'blank':
        raise UnsupportedTrace('incomplete-path')
    text, tokens = _tokens(runs, symbols, geometry, steps)
    if value.get('text') != text or value.get('tokens') != tokens:
        raise UnsupportedTrace('derived-token-mismatch')
    return geometry


class _LiveSidecar(dict):
    """JSON payload plus nonserialized process-local collection authority."""


def seal_sidecar(value):
    """Seal the bounded observer payload without serializing random authority."""
    if not isinstance(value, _LiveSidecar):
        raise TypeError('Only observer-created sidecars can be sealed')
    value._auth = hmac.new(_AUTH_KEY, encoded(value), hashlib.sha256).digest()
    return value


def make_sidecar(image_sha256, image_size, identity, paths, entries, stats, limits=None):
    """Bind uniquely materialized actual readings; ambiguous matches are absent."""
    limits = limits or TraceLimits()
    bounded_stats = dict(stats, sidecar_row_byte_cap=False)
    value = _LiveSidecar(schema_version=SCHEMA, image_sha256=image_sha256,
                         image_size=list(image_size), identity=identity, rows=[],
                         limits=asdict(limits), stats=bounded_stats,
                         semantics='horizontal CTC hints; original pixels and existing guards remain authoritative')
    size = len(encoded(value))
    if size > limits.max_total_bytes:
        # No trace payload is emitted when even its mandatory bindings cannot
        # fit. A rewritten "small" header could still violate a tiny cap.
        raise UnsupportedTrace('sidecar-header-byte-cap')
    rows = []
    for parent_index, parent in enumerate(entries):
        if len(rows) >= limits.max_rows:
            break
        if sum(item.get('bbox') == parent.get('bbox') and item.get('text') == parent.get('text')
               for item in entries) != 1:
            continue
        matches = [path for path in paths if path['geometry']['crop_bbox'] == parent.get('bbox')
                   and path['text'] == parent.get('text')
                   and _selected_initial_parent(path['geometry'], parent_index, parent)]
        if len(matches) != 1:
            continue
        path = matches[0]
        _validated_path(path)
        row = {'parent_index': parent_index, 'parent_bbox': list(parent['bbox']),
               'text': parent['text'], 'text_sha256': text_digest(parent['text']),
               'selected_view': path['geometry']['selected_view'],
               'trace_sha256': digest(path), 'tokens': path['tokens'], 'trace': path}
        row_bytes = len(encoded(row)) + 1
        if size + row_bytes > limits.max_total_bytes:
            # This flag was reserved in the header; true is one byte shorter
            # than false, so recording fallback cannot push it over the cap.
            value['stats']['sidecar_row_byte_cap'] = True
            break
        rows.append(row)
        size += row_bytes
    value['rows'] = rows
    return seal_sidecar(value)


def validate_sidecar(image_sha256, image_size, entries, sidecar):
    """Return verified live rows or an empty list on any lineage/mapping mismatch.

    A JSON reloaded report deliberately has no live collection authority. This
    check must run on the original observer sidecar during proposal and re-audit.
    """
    try:
        if (not isinstance(sidecar, _LiveSidecar)
                or not hmac.compare_digest(getattr(sidecar, '_auth', b''),
                                           hmac.new(_AUTH_KEY, encoded(sidecar), hashlib.sha256).digest())
                or sidecar.get('schema_version') != SCHEMA
                or sidecar.get('image_sha256') != image_sha256
                or sidecar.get('image_size') != list(image_size)):
            return []
        rows, seen = sidecar['rows'], set()
        if not isinstance(rows, list) or len(rows) > TraceLimits().max_rows:
            return []
        for row in rows:
            index = row['parent_index']
            if type(index) is not int or index in seen or not 0 <= index < len(entries):
                return []
            seen.add(index)
            parent = entries[index]
            geometry = _validated_path(row['trace'])
            if (row['parent_bbox'] != parent['bbox'] or row['text'] != parent['text']
                    or row['text_sha256'] != text_digest(parent['text'])
                    or row['trace_sha256'] != digest(row['trace'])
                    or geometry['crop_bbox'] != parent['bbox']
                    or row['trace']['text'] != parent['text']
                    or row['tokens'] != row['trace']['tokens']
                    or row['selected_view'] != geometry['selected_view']
                    or not _selected_initial_parent(geometry, index, parent)):
                return []
            if not (0 <= parent['bbox'][0] < parent['bbox'][2] <= image_size[0]
                    and 0 <= parent['bbox'][1] < parent['bbox'][3] <= image_size[1]):
                return []
        return rows
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return []


def _selected_initial_parent(geometry, index, parent):
    """Initial paths require explicit final lineage, never a text/box lookup."""
    if geometry.get('geometry_kind') != 'first-pass-horizontal-v1':
        return True
    return (geometry.get('final_parent_index') == index
            and geometry.get('selected_parent') == {
                'index': index, 'bbox': parent.get('bbox'), 'text': parent.get('text'),
                'score': parent.get('score')})
