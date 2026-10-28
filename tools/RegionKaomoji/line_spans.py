"""Bounded original-pixel line spans, independent of OCR readings.

Coordinates are half-open original-canvas rectangles. Core components establish
uncuttable glyph boundaries; faint base-only components are recovered whole.
Splitting a base component that joins several core glyphs records its low-contrast
bridge. The caller must retain original RGB pixels and independently prove body
ownership: none of this module's geometric proposals is body-safe by itself.
"""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np

SCHEMA = 'localocr.line-span.v1'
DEFAULT_OPTIONS = {
    'strategy': 'core-bands-v1', 'minimum_blank_columns': 1,
    'maximum_gap_height': 1.25, 'maximum_cells_per_span': 32,
    'maximum_spans_per_seed': 4,
}
DEFAULT_LIMITS = {
    'maximum_pixels': 50_000_000, 'maximum_components': 8192,
    'maximum_candidates': 48, 'maximum_crop_pixels': 1_000_000,
    'minimum_height': 7, 'maximum_height': 72, 'maximum_width': 1024,
    'maximum_baseline_regions': 4096,
}


def configuration(options=None):
    """Return a detached complete configuration, rejecting unknown/unsafe options."""
    value = copy.deepcopy(DEFAULT_OPTIONS if options is None else options)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_OPTIONS):
        raise ValueError('Unknown or incomplete line-span options')
    if value['strategy'] != 'core-bands-v1':
        raise ValueError('Unsupported line-span strategy')
    for key, low, high in (('minimum_blank_columns', 1, 4),
                           ('maximum_cells_per_span', 2, 32),
                           ('maximum_spans_per_seed', 1, 4)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError('Line-span integer bound outside supported range')
    gap = value['maximum_gap_height']
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap) or not .3 <= gap <= 1.25:
        raise ValueError('Line-span gap must be finite and between .3 and 1.25')
    return value


def _limits(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_LIMITS):
        raise ValueError('Unknown or incomplete line-span limits')
    if any(type(item) is not int or item < 1 for item in value.values()):
        raise ValueError('Line-span limits must be positive integers')
    for key in DEFAULT_LIMITS:
        if key != 'minimum_height' and value[key] > DEFAULT_LIMITS[key]:
            raise ValueError('Line-span resource ceiling cannot be increased')
    if value['minimum_height'] > value['maximum_height']:
        raise ValueError('Invalid line-span height range')
    return dict(value)


def _bbox(value, width, height):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(v) is not int for v in value)
            or not 0 <= value[0] < value[2] <= width
            or not 0 <= value[1] < value[3] <= height):
        raise ValueError('Line-span box outside original canvas')
    return list(value)


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _components(mask, maximum):
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count - 1 > maximum:
        raise ValueError('Original mask exceeds line-span component ceiling')
    items = [{'id': index, 'bbox': [int(x), int(y), int(x + w), int(y + h)], 'pixels': int(pixels)}
             for index, (x, y, w, h, pixels) in enumerate(statistics[1:count], 1)]
    return labels, items


def _cells(items, minimum_blank):
    """Join only overlapping x projections, never the legacy two-pixel tolerance."""
    result = []
    for item in sorted(items, key=lambda c: (c['bbox'][0], c['bbox'][2], c['id'])):
        if result and item['bbox'][0] - result[-1]['bbox'][2] < minimum_blank:
            result[-1]['bbox'] = _union([result[-1]['bbox'], item['bbox']])
            result[-1]['components'].append(item)
        else:
            result.append({'bbox': list(item['bbox']), 'components': [item]})
    return result


def _runs(cells, scale, gap, base):
    """Dense full-square cells are geometric barriers, never a text classifier."""
    runs, current = [], []
    for index, cell in enumerate(cells):
        left, top, right, bottom = cell['bbox']
        density = float(base[top:bottom, left:right].sum()) / ((right - left) * (bottom - top))
        dense = (right - left >= scale * .65 and bottom - top >= scale * .65 and density >= .4)
        separated = bool(current and left - cells[current[-1]]['bbox'][2] > scale * gap)
        if dense or separated:
            if current:
                runs.append(current)
            current = []
        if not dense:
            current.append(index)
    if current:
        runs.append(current)
    return runs


def _boundary(mask, box, labels):
    """Record inside pixels adjacent across a rectangle to the same base ink."""
    left, top, right, bottom = box
    height, width = mask.shape
    found, ids, sides = set(), set(), set()
    for name, x, other in (('left', left, left - 1), ('right', right - 1, right)):
        if not 0 <= other < width:
            continue
        for y in np.flatnonzero(mask[top:bottom, x]) + top:
            if mask[max(0, y - 1):min(height, y + 2), other].any():
                found.add((int(x), int(y)))
                ids.add(int(labels[y, x])); sides.add(name)
    for name, y, other in (('top', top, top - 1), ('bottom', bottom - 1, bottom)):
        if not 0 <= other < height:
            continue
        for x in np.flatnonzero(mask[y, left:right]) + left:
            if mask[other, max(0, x - 1):min(width, x + 2)].any():
                found.add((int(x), int(y)))
                ids.add(int(labels[y, x])); sides.add(name)
    return {'present': bool(found), 'crossing_pixels': len(found),
            'component_ids': sorted(ids), 'sides': sorted(sides)}


def _core_basins(core_items, core_labels, base_labels):
    owners = {}
    for item in core_items:
        left, top, right, bottom = item['bbox']
        local = core_labels[top:bottom, left:right]
        position = int(np.flatnonzero(local == item['id'])[0])
        y, x = divmod(position, right - left)
        owners.setdefault(int(base_labels[top + y, left + x]), set()).add(item['id'])
    return owners


def _faint_owners(base_items, core_items, basins):
    """Assign whole faint components to all equally nearest complete core glyphs."""
    if not core_items:
        return []
    boxes = np.asarray([item['bbox'] for item in core_items], dtype=np.int64)
    result = []
    for item in base_items:
        if item['id'] in basins:
            continue
        b = item['bbox']
        dx = np.maximum(np.maximum(boxes[:, 0] - b[2], b[0] - boxes[:, 2]), 0)
        dy = np.maximum(np.maximum(boxes[:, 1] - b[3], b[1] - boxes[:, 3]), 0)
        distance = dx * dx + dy * dy
        nearest = int(distance.min())
        ids = {core_items[int(i)]['id'] for i in np.flatnonzero(distance == nearest)}
        result.append({**item, 'owners': ids, 'distance_squared': nearest})
    return result


def _envelope(base, base_labels, core_items, basins, base_items, faint, cells, selected, line_box, scale, gap):
    selected_items = [item for index in selected for item in cells[index]['components']]
    selected_ids = {item['id'] for item in selected_items}
    core_box = _union([item['bbox'] for item in selected_items])
    left_neighbour = max((cell['bbox'][2] for i, cell in enumerate(cells)
                          if i < selected[0]), default=0)
    right_neighbour = min((cell['bbox'][0] for i, cell in enumerate(cells)
                           if i > selected[-1]), default=base.shape[1])
    # Mid-band basins retain halo pixels without annexing neighbouring core.
    low = (left_neighbour + core_box[0]) // 2 if selected[0] else line_box[0]
    high = (core_box[2] + right_neighbour + 1) // 2 if selected[-1] < len(cells) - 1 else line_box[2]
    local = base[line_box[1]:line_box[3], low:high]
    yy, xx = np.nonzero(local)
    boxes = [[int(xx.min()) + low, int(yy.min()) + line_box[1],
              int(xx.max()) + low + 1, int(yy.max()) + line_box[1] + 1]]
    for item in base_items:
        owned = basins.get(item['id'], set())
        if owned and owned <= selected_ids:
            boxes.append(item['bbox'])
    reclaimed = []
    for item in faint:
        selected_owners = item['owners'] & selected_ids
        near = item['distance_squared'] <= (scale * gap) ** 2
        if selected_owners and near:
            if not item['owners'] <= selected_ids:
                return None, 'ambiguous-base-only-ownership'
            boxes.append(item['bbox']); reclaimed.append(item)
    box = _union(boxes)
    for item in core_items:
        if _overlap(box, item['bbox']):
            if item['id'] not in selected_ids or not _contains(box, item['bbox']):
                return None, 'unselected-or-cut-core-component'
    for item in faint:
        if _overlap(box, item['bbox']) and (not _contains(box, item['bbox']) or item not in reclaimed):
            return None, 'unaccounted-base-only-component'
    bridge = _boundary(base, box, base_labels)
    # A cut through a component with no outside core is an unaccounted faint
    # branch, not the halo bridge between independently complete glyphs.
    if any(not (basins.get(index, set()) - selected_ids) for index in bridge['component_ids']):
        return None, 'unaccounted-base-component-cut'
    height, width = base.shape
    crop_limits = [0, 0, width, height]
    for item in core_items:
        if item['id'] in selected_ids:
            continue
        b = item['bbox']
        if b[1] < box[3] and box[1] < b[3]:
            if b[2] <= box[0]:
                crop_limits[0] = max(crop_limits[0], b[2] + 1)
            elif b[0] >= box[2]:
                crop_limits[2] = min(crop_limits[2], b[0] - 1)
        if b[0] < box[2] and box[0] < b[2]:
            if b[3] <= box[1]:
                crop_limits[1] = max(crop_limits[1], b[3] + 1)
            elif b[1] >= box[3]:
                crop_limits[3] = min(crop_limits[3], b[1] - 1)
    if not _contains(crop_limits, box):
        return None, 'no-core-separation-band'
    return (box, {'core_bbox': core_box, 'core_component_ids': sorted(selected_ids),
                  'base_only_component_ids': [item['id'] for item in reclaimed],
                  'base_only_pixels': sum(item['pixels'] for item in reclaimed),
                  'base_bridge': bridge, 'crop_limits': crop_limits}), None


def propose_spans(base_mask, core_mask, physical_lines, original_entries, seeds, *, limits, options):
    """Propose independent bounded envelopes using geometry only.

    ``physical_lines`` holds original-coordinate ``bbox`` dictionaries. Entries
    need only ``index`` and ``bbox``; seeds need ``ink_bbox`` (or ``bbox``).
    Text, recognition, classification and truth fields are never consulted.
    Explicit seed hypotheses precede entry and implicit line seeds. Each seed
    attempts at most four spans; exact equal envelopes deduplicate without union.
    ``crop_limits`` must also clamp downstream padding. Existing candidates and
    their resource consumption must be handled by the caller's shared cap.
    """
    if options is None:
        raise ValueError('Omit the optional line-span path instead of passing null')
    options = configuration(options)
    limits = _limits(limits)
    for mask in (base_mask, core_mask):
        if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size > limits['maximum_pixels']
                or mask.dtype not in (np.dtype('bool'), np.dtype('uint8'))
                or not np.all((mask == 0) | (mask == 1))):
            raise ValueError('Line-span masks must be bounded binary arrays')
    if base_mask.shape != core_mask.shape or not all(base_mask.shape) or np.any(core_mask & ~base_mask):
        raise ValueError('Core ink must be a subset of the same nonempty base canvas')
    base = np.ascontiguousarray(base_mask, dtype=np.uint8)
    core = np.ascontiguousarray(core_mask, dtype=np.uint8)
    height, width = base.shape
    if (not isinstance(physical_lines, (list, tuple)) or len(physical_lines) > limits['maximum_components']
            or not isinstance(original_entries, (list, tuple)) or len(original_entries) > limits['maximum_baseline_regions']
            or not isinstance(seeds, (list, tuple)) or len(seeds) > limits['maximum_components']):
        raise ValueError('Line-span geometry exceeds input complexity ceiling')
    line_boxes = [_bbox(item['bbox'], width, height) for item in physical_lines]
    entry_boxes = []
    for item in original_entries:
        index = item['index']
        if type(index) is not int or index < 0:
            raise ValueError('Original baseline index must be a nonnegative integer')
        entry_boxes.append((index, _bbox(item['bbox'], width, height)))
    if len({index for index, _ in entry_boxes}) != len(entry_boxes):
        raise ValueError('Duplicate original baseline index')
    seed_boxes = [_bbox(item.get('ink_bbox', item.get('bbox')), width, height) for item in seeds]
    base_labels, base_items = _components(base, limits['maximum_components'])
    core_labels, core_items = _components(core, limits['maximum_components'])
    basins = _core_basins(core_items, core_labels, base_labels)
    faint = _faint_owners(base_items, core_items, basins)
    result = {'proposals': [], 'rejections': {}, 'crop_pixels': 0,
              'seed_hypothesis_counts': [], 'seed_hypotheses': []}
    def reject(reason):
        result['rejections'][reason] = result['rejections'].get(reason, 0) + 1
    seen, seed_attempts = set(), {}

    def finish():
        result['seed_hypotheses'] = list(seed_attempts.values())
        result['seed_hypothesis_counts'] = [item['attempted'] for item in seed_attempts.values()]
        return result

    def budget_exhausted():
        # Even a one-column accepted crop must meet the minimum height. Once
        # that area cannot fit, no later seed or line can produce an output.
        return (len(result['proposals']) >= limits['maximum_candidates']
                or limits['maximum_crop_pixels'] - result['crop_pixels'] < limits['minimum_height'])

    for line_index, line_box in enumerate(line_boxes):
        if budget_exhausted():
            reject('total-span-budget'); return finish()
        items = [item for item in core_items if _overlap(line_box, item['bbox'])]
        if not items:
            reject('line-without-core'); continue
        if any(not _contains(line_box, item['bbox']) for item in items):
            reject('line-cuts-core-component'); continue
        cells = _cells(items, options['minimum_blank_columns'])
        heights = sorted(cell['bbox'][3] - cell['bbox'][1] for cell in cells)
        typical = heights[int(.8 * (len(heights) - 1))]
        scale = max(h for h in heights if h <= typical * 1.3)
        runs = _runs(cells, scale, options['maximum_gap_height'], base)
        line_seeds = [('proposal', index, box) for index, box in enumerate(seed_boxes) if _overlap(line_box, box)]
        line_seeds += [('baseline-geometry', index, box) for index, box in entry_boxes if _overlap(line_box, box)]
        line_seeds.append(('physical-line', line_index, line_box))
        for kind, seed_index, seed_box in line_seeds:
            if budget_exhausted():
                reject('total-span-budget'); return finish()
            key = (kind, seed_index)
            attempted = seed_attempts.setdefault(key, {'kind': kind, 'index': seed_index, 'attempted': 0})
            remaining = options['maximum_spans_per_seed'] - attempted['attempted']
            if remaining <= 0:
                continue
            hypotheses = []
            for run in runs:
                inside = [i for i in run if _overlap(cells[i]['bbox'], seed_box)]
                if not inside:
                    continue
                low, high = min(inside), max(inside)
                for first, last in ((run[0], run[-1]), (low, high), (run[0], high), (low, run[-1])):
                    hypothesis = tuple(range(first, last + 1))
                    if hypothesis not in hypotheses:
                        hypotheses.append(hypothesis)
                    if len(hypotheses) >= remaining:
                        break
                if len(hypotheses) >= remaining:
                    break
            for selected in hypotheses:
                if budget_exhausted():
                    reject('total-span-budget'); return finish()
                attempted['attempted'] += 1
                if len(selected) > options['maximum_cells_per_span']:
                    reject('span-cell-limit'); continue
                core_box = _union([cells[index]['bbox'] for index in selected])
                minimum_area = (core_box[2] - core_box[0]) * max(
                    core_box[3] - core_box[1], limits['minimum_height'])
                if minimum_area > limits['maximum_crop_pixels'] - result['crop_pixels']:
                    # Recovery can only enlarge the complete core envelope.
                    # Skip this hypothesis cheaply, but a later smaller span
                    # may still fit the remaining pixel budget.
                    reject('total-span-budget'); continue
                envelope, reason = _envelope(base, base_labels, core_items, basins, base_items, faint,
                                             cells, selected, line_box, scale, options['maximum_gap_height'])
                if reason:
                    reject(reason); continue
                box, details = envelope
                w, h = box[2] - box[0], box[3] - box[1]
                if not limits['minimum_height'] <= h <= limits['maximum_height'] or w > limits['maximum_width']:
                    reject('span-size-limit'); continue
                if tuple(box) in seen:
                    continue
                # This module consumes ink-envelope area. The caller must also
                # account for padding in its existing shared final-crop budget.
                if (len(result['proposals']) >= limits['maximum_candidates']
                        or result['crop_pixels'] + w * h > limits['maximum_crop_pixels']):
                    reject('total-span-budget'); continue
                seen.add(tuple(box)); result['crop_pixels'] += w * h
                result['proposals'].append({'ink_bbox': box, 'sources': ['image-line-span'],
                    'line_span': {'schema_version': SCHEMA, 'strategy': options['strategy'],
                                  'physical_line_index': line_index, 'seed_kind': kind, 'seed_index': seed_index,
                                  'cell_range': [selected[0], selected[-1] + 1],
                                  'geometric_parent_indices': [index for index, b in entry_boxes if _overlap(box, b)],
                                  'original_pixels_required': True, 'body_safe': False, **details}})
    return finish()
