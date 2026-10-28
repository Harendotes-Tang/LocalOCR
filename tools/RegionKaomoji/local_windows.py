"""Bounded bracket/neighbor windows on the unchanged original ink masks.

Actual core components supply bracket anchors. At most two local neighbor cells
per side extend a pair; no hypothesis expands to a physical line's run ends.
The shared line-span envelope retains whole core strokes, faint components and
auditable halo crossings. These are geometric proposals, never body-ownership or
face proofs: the caller must retain RGB pixels and all existing admission guards.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from tools.RegionKaomoji import bracket_profiles
from tools.RegionKaomoji import line_spans

SCHEMA = 'localocr.local-window.v1'
DEFAULT_OPTIONS = {
    'strategy': 'bracket-components-v1', 'maximum_neighbor_cells': 2,
    'maximum_gap_height': .6, 'maximum_windows_per_seed': 4,
    'maximum_cells_per_window': 32,
}
DEFAULT_LIMITS = dict(line_spans.DEFAULT_LIMITS)
_components = line_spans._components
_envelope = line_spans._envelope
_FAMILY = {'curve': 'curve', 'small-curve': 'curve',
           'right-angle': 'square', 'brace': 'brace'}


def configuration(options=None):
    """Return detached options within the predeclared finite local ceilings."""
    value = copy.deepcopy(DEFAULT_OPTIONS if options is None else options)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_OPTIONS):
        raise ValueError('Unknown or incomplete local-window options')
    if value['strategy'] != 'bracket-components-v1':
        raise ValueError('Unsupported local-window strategy')
    for key, low, high in (('maximum_neighbor_cells', 0, 2),
                           ('maximum_windows_per_seed', 1, 4),
                           ('maximum_cells_per_window', 2, 32)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError('Local-window integer bound outside supported range')
    gap = value['maximum_gap_height']
    if (isinstance(gap, bool) or not isinstance(gap, (int, float))
            or not math.isfinite(gap) or not .3 <= gap <= .6):
        raise ValueError('Local-window gap must be finite and between .3 and .6')
    return value


def _dense(cell, scale, core):
    left, top, right, bottom = cell['bbox']
    w, h = right-left, bottom-top
    return (w >= scale*.65 and h >= scale*.65
            and int(core[top:bottom, left:right].sum()) / (w*h) >= .4)


def _anchors(core_labels, items, line_box, limits, reject):
    """Classify each complete original component, never caller-supplied shapes."""
    anchors = []
    for item in items:
        box = item['bbox']
        h, w = box[3]-box[1], box[2]-box[0]
        if not limits['minimum_height'] <= h <= limits['maximum_height'] or not 2 <= w <= h*.55:
            continue
        if not line_spans._contains(line_box, box):
            reject('anchor-crosses-physical-line')
            continue
        local = core_labels[box[1]:box[3], box[0]:box[2]] == item['id']
        observed = bracket_profiles.classify(local)
        if observed is not None:
            anchors.append({**item, **observed})
    return anchors


def _pairs(anchors, cells, options, limits):
    """Match nested pairs locally; a close-then-open sequence starts a new face.

    A stack avoids quadratic all-pairs scans and cross-pairing two neighboring
    expressions. Mismatched families do not supply an alternative pair. Distant
    open brackets expire at the fixed component-window bound.
    """
    positions = {item['id']: index for index, cell in enumerate(cells)
                 for item in cell['components']}
    stack = []
    for anchor in sorted(anchors, key=lambda item: (item['bbox'][0], item['bbox'][2], item['id'])):
        position = positions[anchor['id']]
        stack = [left for left in stack
                 if position-positions[left['id']] < options['maximum_cells_per_window']]
        if anchor['direction'] == 'left':
            stack.append(anchor)
            continue
        if not stack:
            continue
        left = stack.pop()
        if _FAMILY[left['kind']] != _FAMILY[anchor['kind']]:
            stack.clear()
            continue
        a, b = left['bbox'], anchor['bbox']
        scale = max(a[3]-a[1], b[3]-b[1])
        if (a[2] >= b[0] or not 1.2*scale <= b[2]-a[0] <= min(14*scale, limits['maximum_width'])
                or min(a[3]-a[1], b[3]-b[1])/scale < .65
                or abs((a[1]+a[3])-(b[1]+b[3])) > .6*scale):
            continue
        yield left, anchor, positions[left['id']], position, scale


def _neighbors(cells, first, last, scale, anchor_ids, core, line_box, options):
    low, high = first, last
    for step in (-1, 1):
        position = first-1 if step < 0 else last+1
        edge = cells[first]['bbox'][0] if step < 0 else cells[last]['bbox'][2]
        for _ in range(options['maximum_neighbor_cells']):
            if not 0 <= position < len(cells):
                break
            cell = cells[position]
            box = cell['bbox']
            gap = edge-box[2] if step < 0 else box[0]-edge
            # A different bracket starts another expression or a nested level;
            # only its own actual matched pair may supply that wider window.
            if (not line_spans._contains(line_box, box)
                    or gap > scale*options['maximum_gap_height'] or box[3]-box[1] > scale*1.25
                    or _dense(cell, scale, core)
                    or any(item['id'] in anchor_ids for item in cell['components'])):
                break
            if step < 0:
                low, edge = position, box[0]
            else:
                high, edge = position, box[2]
            position += step
    return low, high


def _seed_pairs(seeds):
    # Existing real pair metadata only prioritizes a recomputed anchor; it never
    # creates shape evidence or changes the pixels included by a window.
    result = {}
    for index, seed in enumerate(seeds):
        pair = seed.get('pair')
        if not isinstance(pair, dict):
            continue
        strokes = pair.get('strokes')
        if (not isinstance(strokes, (list, tuple)) or len(strokes) != 2
                or any(not isinstance(box, (list, tuple)) or len(box) != 4
                       or any(type(v) is not int for v in box) for box in strokes)):
            continue
        result.setdefault(tuple(map(tuple, strokes)), index)
    return result


def _seed_identity(left, right, seed_pairs, ordinal):
    index = seed_pairs.get((tuple(left['bbox']), tuple(right['bbox'])))
    return ('proposal', index) if index is not None else ('bracket-components', ordinal)


def propose_windows(base_mask, core_mask, physical_lines, original_entries, seeds, *, limits, options):
    """Return bounded original-coordinate windows and geometric audit records.

    Input geometry follows ``line_spans.propose_spans``. ``seeds[].pair.strokes``
    may prioritize corresponding actual anchors; OCR text is never consulted.
    Every seed attempts at most four local envelopes. The caller supplies only
    its *remaining* shared budget, clamps final padding to ``crop_limits`` and
    charges the padded crop in the existing shared cap. Omitting this optional
    path preserves the legacy behavior; passing null is an explicit error.
    """
    if options is None:
        raise ValueError('Omit the optional local-window path instead of passing null')
    options = configuration(options)
    limits = line_spans._limits(limits)
    for mask in (base_mask, core_mask):
        if (not isinstance(mask, np.ndarray) or mask.ndim != 2
                or mask.size > limits['maximum_pixels']
                or mask.dtype not in (np.dtype('bool'), np.dtype('uint8'))
                or not np.all((mask == 0) | (mask == 1))):
            raise ValueError('Local-window masks must be bounded binary arrays')
    if (base_mask.shape != core_mask.shape or not all(base_mask.shape)
            or np.any(core_mask & ~base_mask)):
        raise ValueError('Core ink must be a subset of the same nonempty base canvas')
    height, width = base_mask.shape
    if (not isinstance(physical_lines, (list, tuple)) or len(physical_lines) > limits['maximum_components']
            or not isinstance(original_entries, (list, tuple)) or len(original_entries) > limits['maximum_baseline_regions']
            or not isinstance(seeds, (list, tuple)) or len(seeds) > limits['maximum_components']):
        raise ValueError('Local-window geometry exceeds input complexity ceiling')
    lines = [line_spans._bbox(item['bbox'], width, height) for item in physical_lines]
    entries = []
    for item in original_entries:
        index = item['index']
        if type(index) is not int or index < 0:
            raise ValueError('Original baseline index must be a nonnegative integer')
        entries.append((index, line_spans._bbox(item['bbox'], width, height)))
    if len({index for index, _ in entries}) != len(entries):
        raise ValueError('Duplicate original baseline index')
    for item in seeds:
        line_spans._bbox(item.get('ink_bbox', item.get('bbox')), width, height)
    seed_pairs = _seed_pairs(seeds)
    result = {'proposals': [], 'rejections': {}, 'crop_pixels': 0,
              'seed_hypothesis_counts': [], 'seed_hypotheses': []}
    attempts, seen = {}, set()

    def reject(reason):
        result['rejections'][reason] = result['rejections'].get(reason, 0)+1

    def finish():
        result['seed_hypotheses'] = list(attempts.values())
        result['seed_hypothesis_counts'] = [item['attempted'] for item in attempts.values()]
        return result

    def exhausted():
        return (len(result['proposals']) >= limits['maximum_candidates']
                or limits['maximum_crop_pixels']-result['crop_pixels'] < limits['minimum_height'])

    if exhausted():
        reject('total-window-budget')
        return finish()
    if not lines:
        return finish()
    base = np.ascontiguousarray(base_mask, dtype=np.uint8)
    core = np.ascontiguousarray(core_mask, dtype=np.uint8)
    base_labels, base_items = _components(base, limits['maximum_components'])
    core_labels, core_items = _components(core, limits['maximum_components'])
    basins = line_spans._core_basins(core_items, core_labels, base_labels)
    faint = line_spans._faint_owners(base_items, core_items, basins)
    ordinal = 0
    for line_index, line_box in enumerate(lines):
        if exhausted():
            reject('total-window-budget')
            return finish()
        intersecting = [item for item in core_items if line_spans._overlap(line_box, item['bbox'])]
        anchors = _anchors(core_labels, intersecting, line_box, limits, reject)
        if len(anchors) < 2:
            continue
        # Crossing, unrelated core still bounds padding; omitting that cell
        # would make a first/last window annex the physical line's outer band.
        cells = line_spans._cells(intersecting, 1)
        anchor_ids = {item['id'] for item in anchors}
        pairs = []
        for pair in _pairs(anchors, cells, options, limits):
            kind, index = _seed_identity(pair[0], pair[1], seed_pairs, ordinal)
            pairs.append((pair, kind, index))
            ordinal += 1
        # Larger nested complete pairs precede their inner windows. Explicit
        # seed priority is stable and solely geometric.
        pairs.sort(key=lambda item: (item[2] if item[1] == 'proposal' else len(seeds),
                   item[0][0]['bbox'][0], -item[0][1]['bbox'][2]))
        for (left, right, first, last, scale), kind, seed_index in pairs:
            if exhausted():
                reject('total-window-budget')
                return finish()
            record = attempts.setdefault((kind, seed_index),
                {'kind': kind, 'index': seed_index, 'attempted': 0})
            if any(not line_spans._contains(line_box, item['bbox'])
                   for cell in cells[first:last+1] for item in cell['components']):
                reject('window-crosses-physical-line')
                continue
            if any(_dense(cell, scale, core) for cell in cells[first:last+1]):
                reject('dense-core-inside-brackets')
                continue
            low, high = _neighbors(cells, first, last, scale, anchor_ids, core, line_box, options)
            hypotheses = list(dict.fromkeys(((low, high), (first, last), (low, last), (first, high))))
            for start, end in hypotheses:
                if exhausted():
                    reject('total-window-budget')
                    return finish()
                if record['attempted'] >= options['maximum_windows_per_seed']:
                    break
                record['attempted'] += 1
                if end-start+1 > options['maximum_cells_per_window']:
                    reject('window-cell-limit')
                    continue
                selected = tuple(range(start, end+1))
                core_box = line_spans._union([cells[index]['bbox'] for index in selected])
                minimum_area = (core_box[2]-core_box[0])*max(core_box[3]-core_box[1], limits['minimum_height'])
                if minimum_area > limits['maximum_crop_pixels']-result['crop_pixels']:
                    reject('total-window-budget')
                    continue
                envelope, reason = _envelope(base, base_labels, core_items, basins, base_items, faint,
                    cells, selected, line_box, scale, options['maximum_gap_height'])
                if reason:
                    reject(reason)
                    continue
                box, evidence = envelope
                w, h = box[2]-box[0], box[3]-box[1]
                if not limits['minimum_height'] <= h <= limits['maximum_height'] or w > limits['maximum_width']:
                    reject('window-size-limit')
                    continue
                if tuple(box) in seen:
                    continue
                if result['crop_pixels']+w*h > limits['maximum_crop_pixels']:
                    reject('total-window-budget')
                    continue
                selected_ids = set(evidence['core_component_ids'])
                complete_base = sorted({index for index, owners in basins.items() if owners <= selected_ids}
                                       | set(evidence['base_only_component_ids']))
                seen.add(tuple(box))
                result['crop_pixels'] += w*h
                result['proposals'].append({'ink_bbox': box, 'sources': ['image-local-window'],
                    'local_window': {'schema_version': SCHEMA, 'strategy': options['strategy'],
                        'physical_line_index': line_index, 'seed_kind': kind, 'seed_index': seed_index,
                        'bracket_strokes': [left['bbox'], right['bbox']],
                        'anchor_core_component_ids': [left['id'], right['id']],
                        'bracket_shapes': [{key: anchor[key] for key in ('kind', 'direction', 'measurements')}
                                           for anchor in (left, right)],
                        'cell_range': [start, end+1], 'base_component_ids': complete_base,
                        'geometric_parent_indices': [index for index, b in entries if line_spans._overlap(box, b)],
                        'original_pixels_required': True, 'body_safe': False, **evidence}})
    return finish()
