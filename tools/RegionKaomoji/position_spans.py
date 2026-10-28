"""Bounded position hints grounded in unchanged original connected components.

CTC token extents are horizontal proposals only. The caller validates their
image/parent/runtime binding; this module never treats a token as pixel, body or
recognition evidence. Component envelopes retain the existing conservative
weak-ink, neighboring-core and crop-padding checks.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from tools.PretrainedOcr import kaomoji_grammar as grammar
from tools.RegionKaomoji import line_spans, local_windows

SCHEMA = 'localocr.position-span.v1'
DEFAULT_OPTIONS = {'strategy': 'ctc-components-v1', 'mode': 'union'}
DEFAULT_LIMITS = dict(line_spans.DEFAULT_LIMITS)
MAXIMUM_HINTS = 48
MAXIMUM_HYPOTHESES = 4
MAXIMUM_CELLS = 32
GAP_HEIGHT = .6


def configuration(options=None):
    """Validate the complete fixed-mode strategy; numeric caps are not tunable."""
    value = copy.deepcopy(DEFAULT_OPTIONS if options is None else options)
    if (not isinstance(value, dict) or set(value) not in (set(DEFAULT_OPTIONS), set(DEFAULT_OPTIONS) | {'initial_trace'})
            or value['strategy'] != DEFAULT_OPTIONS['strategy']
            or value['mode'] not in ('components', 'ctc', 'union')):
        raise ValueError('Unknown or incomplete position-span options')
    if 'initial_trace' in value and (value['initial_trace'] != 'horizontal-v1' or value['mode'] not in ('ctc', 'union')):
        raise ValueError('Initial trace requires the fixed horizontal CTC strategy')
    return value


def _hints(rows):
    pairs = {}
    for opening, closing in grammar.bracket_pairs():
        pairs.setdefault(opening, set()).add(closing)
    closes = set().union(*pairs.values())
    count = 0
    for row in rows:
        stack = []
        for token in row['tokens']:
            # A vocabulary token may contain several Unicode codepoints. Its
            # actual timestep extent is shared; no invented character slicing.
            for char in token['text']:
                if char in pairs:
                    stack.append((char, token))
                elif char in closes and stack:
                    left, opening = stack.pop()
                    if char not in pairs[left]:
                        stack.clear()
                        continue
                    yield row, opening['x_start'], token['x_end']
                    count += 1
                    if count >= MAXIMUM_HINTS:
                        return


def _validate(base, core, physical_lines, entries, rows, limits):
    for mask in (base, core):
        if (not isinstance(mask, np.ndarray) or mask.ndim != 2
                or mask.size > limits['maximum_pixels']
                or mask.dtype not in (np.dtype('bool'), np.dtype('uint8'))
                or not np.all((mask == 0) | (mask == 1))):
            raise ValueError('Position masks must be bounded binary arrays')
    if base.shape != core.shape or not all(base.shape) or np.any(core & ~base):
        raise ValueError('Position core must be a subset of the same base canvas')
    if (not isinstance(physical_lines, (list, tuple))
            or len(physical_lines) > limits['maximum_components']
            or not isinstance(entries, (list, tuple))
            or len(entries) > limits['maximum_baseline_regions']
            or not isinstance(rows, (list, tuple))
            or len(rows) > limits['maximum_baseline_regions']):
        raise ValueError('Position geometry exceeds the fixed complexity cap')
    height, width = base.shape
    for row in rows:
        line_spans._bbox(row['parent_bbox'], width, height)
        if not isinstance(row.get('tokens'), list) or len(row['tokens']) > 65536:
            raise ValueError('Position token rows must be bounded')
        for token in row['tokens']:
            if (not isinstance(token.get('text'), str)
                    or any(isinstance(token.get(key), bool)
                           or not isinstance(token.get(key), (int, float))
                           or not math.isfinite(token[key]) for key in ('x_start', 'x_end'))
                    or not 0 <= token['x_start'] <= token['x_end'] <= width):
                raise ValueError('Invalid original-coordinate token extent')
    return [line_spans._bbox(line['bbox'], width, height) for line in physical_lines]


def propose_spans(base_mask, core_mask, physical_lines, original_entries, seeds,
                  *, limits, options, trace_rows=()):
    """Return complete pixel envelopes under the caller's remaining shared cap.

    ``trace_rows`` MUST have passed ctc_trace.validate_sidecar against the same
    original image and first-pass entries. It supplies no vertical boundary or
    safety vote. Empty/unsupported rows leave the component path available.
    """
    if options is None:
        raise ValueError('Omit unused position spans instead of null')
    options, limits = configuration(options), line_spans._limits(limits)
    lines = _validate(base_mask, core_mask, physical_lines, original_entries, trace_rows, limits)
    result = {'proposals': [], 'rejections': {}, 'crop_pixels': 0}
    seen = set()

    def reject(reason):
        result['rejections'][reason] = result['rejections'].get(reason, 0) + 1

    def add(proposal):
        box = proposal['ink_bbox']
        pixels = (box[2]-box[0]) * (box[3]-box[1])
        if tuple(box) in seen:
            return
        if (len(result['proposals']) >= limits['maximum_candidates']
                or result['crop_pixels'] + pixels > limits['maximum_crop_pixels']):
            reject('shared-position-budget')
            return
        seen.add(tuple(box))
        result['proposals'].append(proposal)
        result['crop_pixels'] += pixels

    if options['mode'] in ('components', 'union'):
        components = local_windows.propose_windows(base_mask, core_mask, physical_lines,
            original_entries, seeds, limits=limits, options=local_windows.configuration())
        result['rejections'].update(components['rejections'])
        for proposal in components['proposals']:
            add({'ink_bbox': proposal['ink_bbox'], 'sources': ['image-position-components'],
                 'position_span': {**proposal['local_window'], 'schema_version': SCHEMA,
                                   'proposal_kind': 'components'}})
    if options['mode'] == 'components':
        return result
    if not trace_rows:
        reject('trace-unavailable-or-unsupported')
        return result
    base, core = np.ascontiguousarray(base_mask, dtype=np.uint8), np.ascontiguousarray(core_mask, dtype=np.uint8)
    base_labels, base_items = line_spans._components(base, limits['maximum_components'])
    core_labels, core_items = line_spans._components(core, limits['maximum_components'])
    basins = line_spans._core_basins(core_items, core_labels, base_labels)
    faint = line_spans._faint_owners(base_items, core_items, basins)
    for row, x_start, x_end in _hints(trace_rows):
        if len(result['proposals']) >= limits['maximum_candidates']:
            reject('shared-position-budget')
            break
        parent = row['parent_bbox']
        if not parent[0] <= x_start < x_end <= parent[2]:
            reject('time-hint-outside-parent')
            continue
        matching = [(i, box) for i, box in enumerate(lines)
                    if box[3]-box[1] >= limits['minimum_height']
                    and min(box[3], parent[3])-max(box[1], parent[1]) > 0
                    and box[0] < x_end and x_start < box[2]]
        if len(matching) != 1:
            reject('ambiguous-physical-line')
            continue
        line_index, line_box = matching[0]
        cells = line_spans._cells([item for item in core_items
                                  if line_spans._overlap(line_box, item['bbox'])], 1)
        selected = [i for i, cell in enumerate(cells)
                    if cell['bbox'][0] < x_end and x_start < cell['bbox'][2]]
        if len(selected) < 2:
            reject('insufficient-hint-components')
            continue
        first, last = min(selected), max(selected)
        scale = max(cell['bbox'][3]-cell['bbox'][1] for cell in cells[first:last+1])
        low, high = first, last
        for step in (-1, 1):
            at = first-1 if step < 0 else last+1
            edge = cells[first]['bbox'][0] if step < 0 else cells[last]['bbox'][2]
            for _ in range(2):
                if not 0 <= at < len(cells):
                    break
                cell, box = cells[at], cells[at]['bbox']
                gap = edge-box[2] if step < 0 else box[0]-edge
                if (gap > scale*GAP_HEIGHT or local_windows._dense(cell, scale, core)
                        or box[3]-box[1] > scale*1.25):
                    break
                if step < 0:
                    low, edge = at, box[0]
                else:
                    high, edge = at, box[2]
                at += step
        hypotheses = list(dict.fromkeys(((low, high), (first, last), (low, last), (first, high))))
        for start, end in hypotheses[:MAXIMUM_HYPOTHESES]:
            if end-start+1 > MAXIMUM_CELLS or any(local_windows._dense(cell, scale, core)
                                                for cell in cells[start:end+1]):
                reject('dense-or-overlong-component-range')
                continue
            if any(not line_spans._contains(line_box, item['bbox'])
                   for cell in cells[start:end+1] for item in cell['components']):
                reject('component-crosses-physical-line')
                continue
            envelope, reason = line_spans._envelope(base, base_labels, core_items, basins,
                base_items, faint, cells, tuple(range(start, end+1)), line_box, scale, GAP_HEIGHT)
            if reason:
                reject(reason)
                continue
            box, evidence = envelope
            if (not limits['minimum_height'] <= box[3]-box[1] <= limits['maximum_height']
                    or box[2]-box[0] > limits['maximum_width']):
                reject('position-size-limit')
                continue
            add({'ink_bbox': box, 'sources': ['image-position-ctc'],
                 'position_span': {'schema_version': SCHEMA, 'proposal_kind': 'ctc',
                     'physical_line_index': line_index, 'parent_index': row['parent_index'],
                     'trace_sha256': row['trace_sha256'], 'horizontal_hint': [x_start, x_end],
                     'cell_range': [start, end+1], 'body_safe': False,
                     'original_pixels_required': True, **evidence}})
    return result
