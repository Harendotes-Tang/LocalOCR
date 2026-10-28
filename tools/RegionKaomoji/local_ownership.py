"""Conservative original-mask partitions for one current baseline parent.

This module proves a pixel partition, never a text replacement. Callers still
need production context OCR, unique raw-byte mapping, independent face evidence
and view agreement. In particular, an isolated token inside the ROI cannot be
classified as face/body from this geometry proof alone.
"""
from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np


SCHEMA = 'localocr.local-ownership.v1'
MAXIMUM_PIXELS = 50_000_000
MAXIMUM_SUPPORT_PIXELS = 1_000_000
MAXIMUM_ENTRIES = 4096


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _mask_digest(mask):
    height, width = mask.shape
    return hashlib.sha256(width.to_bytes(8, 'big') + height.to_bytes(8, 'big')
                          + np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def _box(value, shape):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, (int, np.integer))
                   for item in value)):
        raise ValueError('Partition boxes require four half-open integer coordinates')
    left, top, right, bottom = map(int, value)
    if not 0 <= left < right <= shape[1] or not 0 <= top < bottom <= shape[0]:
        raise ValueError('Partition box outside the original mask')
    return [left, top, right, bottom]


def _indices(values, count):
    if not isinstance(values, (list, tuple)) or len(values) > count:
        raise ValueError('Partition indices must be a bounded sequence')
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer))
           or not 0 <= item < count for item in values):
        raise ValueError('Partition index is not a current baseline position')
    return sorted(set(map(int, values)))


def _crosses(mask, box):
    """Any 8-connected ink edge crossing a half-open box boundary."""
    left, top, right, bottom = box
    height, width = mask.shape
    for inner_x, outer_x in ((left, left - 1), (right - 1, right)):
        if not 0 <= outer_x < width:
            continue
        for shift in (-1, 0, 1):
            low, high = max(top, -shift), min(bottom, height - shift)
            if np.any(mask[low:high, inner_x] & mask[low + shift:high + shift, outer_x]):
                return True
    for inner_y, outer_y in ((top, top - 1), (bottom - 1, bottom)):
        if not 0 <= outer_y < height:
            continue
        for shift in (-1, 0, 1):
            low, high = max(left, -shift), min(right, width - shift)
            if np.any(mask[inner_y, low:high] & mask[outer_y, low + shift:high + shift]):
                return True
    return False


def _overlap(first, second):
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def _visible_overlap(mask, first, second):
    if not _overlap(first, second):
        return False
    return bool(mask[max(first[1], second[1]):min(first[3], second[3]),
                     max(first[0], second[0]):min(first[2], second[2])].any())


def _inputs(mask, bbox, entries, owned_indices, neighbor_indices,
            padding, minimum_blank_band, maximum_parent_height_ratio):
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size
            or mask.size > MAXIMUM_PIXELS or mask.dtype not in (np.dtype(bool), np.dtype('uint8'))
            or np.any(mask > 1)):
        raise ValueError('Partition requires a bounded original binary mask')
    if type(padding) is not int or not 0 <= padding <= 16:
        raise ValueError('Context padding outside the supported range')
    if type(minimum_blank_band) is not int or not 2 <= minimum_blank_band <= 32:
        raise ValueError('Context blank band outside the supported range')
    if (isinstance(maximum_parent_height_ratio, bool)
            or not isinstance(maximum_parent_height_ratio, (int, float))
            or not math.isfinite(maximum_parent_height_ratio)
            or not 1 <= maximum_parent_height_ratio <= 2):
        raise ValueError('Parent height ratio outside the supported range')
    box = _box(bbox, mask.shape)
    if not isinstance(entries, list) or len(entries) > MAXIMUM_ENTRIES:
        raise ValueError('Partition requires bounded current baseline entries')
    current = []
    for position, entry in enumerate(entries):
        if (not isinstance(entry, dict) or not {'index', 'bbox', 'text', 'score'} <= entry.keys()
                or isinstance(entry['index'], bool)
                or not isinstance(entry['index'], (int, np.integer)) or entry['index'] != position
                or not isinstance(entry['text'], str) or len(entry['text']) > 4096
                or isinstance(entry['score'], (bool, str))
                or not isinstance(entry['score'], (int, float, np.number))
                or not math.isfinite(float(entry['score']))):
            raise ValueError('Partition baseline entry differs from current indexed input')
        current.append({'index': position, 'bbox': _box(entry['bbox'], mask.shape),
                        'text': entry['text'], 'score': float(entry['score'])})
    owned = _indices(owned_indices, len(current))
    neighbors = _indices(neighbor_indices, len(current))
    config = {'padding': padding, 'minimum_blank_band': minimum_blank_band,
              'maximum_parent_height_ratio': float(maximum_parent_height_ratio)}
    return box, current, owned, neighbors, config


def prove_local_partition(mask, bbox, entries, owned_indices, neighbor_indices,
                          *, padding=3, minimum_blank_band=2,
                          maximum_parent_height_ratio=1.8):
    """Partition union(parent, ROI) into complete ROI and left/right ink.

    ``entries`` are current validated baseline regions, whose list positions and
    ``index`` agree. Index sequences refer to those current positions; the union
    must identify exactly one parent. The ROI need not be contained in that
    parent's detector envelope. All union ink outside the ROI is represented in
    context crops, including one-pixel punctuation and ink outside the parent's
    old envelope. No pixels are erased and no component is area-filtered.

    Returns ``None`` for unsafe/ambiguous geometry. Malformed or unbounded input
    raises ``ValueError``. A result has ``parent_index``, ``context_crops`` and
    ``proof``. The proof binds the full original binary mask, all current entries,
    index declarations, ROI and bounds. It does not bind RGB pixels; the caller
    must additionally bind/recompute against the original RGB image.
    """
    box, current, owned, neighbors, config = _inputs(
        mask, bbox, entries, owned_indices, neighbor_indices,
        padding, minimum_blank_band, maximum_parent_height_ratio)
    references = sorted(set(owned) | set(neighbors))
    if len(references) != 1:
        return None
    index = references[0]
    parent = current[index]
    parent_box = parent['bbox']
    left, top, right, bottom = box
    if (not _visible_overlap(mask, box, parent_box)
            or parent_box[3] - parent_box[1] > (bottom - top) * maximum_parent_height_ratio):
        return None
    support = [min(left, parent_box[0]), min(top, parent_box[1]),
               max(right, parent_box[2]), max(bottom, parent_box[3])]
    if (support[2] - support[0]) * (support[3] - support[1]) > MAXIMUM_SUPPORT_PIXELS:
        return None
    # The ROI, parent and total support must each respect original components.
    # A clipped detector parent cannot silently become a complete text receipt.
    if any(_crosses(mask, item) for item in (box, parent_box, support)):
        return None
    for other in current:
        if other['index'] == index:
            continue
        if _overlap(box, other['bbox']) or _visible_overlap(mask, support, other['bbox']):
            return None
    sx, sy, sr, sb = support
    local = mask[sy:sb, sx:sr]
    yy, xx = np.where(local)
    gx, gy = xx + sx, yy + sy
    within_x = (gx >= left) & (gx < right)
    inside = within_x & (gy >= top) & (gy < bottom)
    if not inside.any() or np.any(within_x & ~inside):
        return None
    inside_left, inside_right = int(gx[inside].min()), int(gx[inside].max())
    inside_top, inside_bottom = int(gy[inside].min()), int(gy[inside].max()) + 1
    contexts = []
    for role, selected in (('left', gx < left), ('right', gx >= right)):
        if not selected.any():
            continue
        cx, cy = gx[selected], gy[selected]
        # Side ink on another physical row is not silently folded into context.
        if np.any((cy < top) | (cy >= bottom)):
            return None
        gap = (inside_left - int(cx.max()) - 1 if role == 'left'
               else int(cx.min()) - inside_right - 1)
        if gap < minimum_blank_band:
            return None
        selection_mask = np.zeros(local.shape, dtype=np.uint8)
        selection_mask[cy - sy, cx - sx] = 1
        count, _, stats, _ = cv2.connectedComponentsWithStats(selection_mask, 8)
        for component in stats[1:count]:
            component_top = sy + int(component[cv2.CC_STAT_TOP])
            component_bottom = component_top + int(component[cv2.CC_STAT_HEIGHT])
            if component_bottom <= inside_top or component_top >= inside_bottom:
                return None
        crop = [max(sx, int(cx.min()) - padding), max(sy, int(cy.min()) - padding),
                min(sr, int(cx.max()) + 1 + padding), min(sb, int(cy.max()) + 1 + padding)]
        if role == 'left':
            crop[2] = min(crop[2], left - 1)
        else:
            crop[0] = max(crop[0], right + 1)
        if (crop[0] > int(cx.min()) or crop[1] > int(cy.min())
                or crop[2] <= int(cx.max()) or crop[3] <= int(cy.max())
                or _crosses(mask, crop)):
            return None
        contexts.append({'role': role, 'bbox': crop, 'blank_band_pixels': gap,
                         'visible_pixels': int(selected.sum())})
    if not contexts:
        return None
    # Each source ink pixel belongs to exactly one of the three declarations.
    if int(inside.sum()) + sum(item['visible_pixels'] for item in contexts) != len(gx):
        return None
    parent_identity = {'schema_version': 'localocr.baseline-parent.v1',
                       'index': index, 'bbox': parent_box, 'text': parent['text']}
    proof = {'schema_version': SCHEMA, 'mask_sha256': _mask_digest(mask),
             'entries_sha256': _digest(current), 'bbox': box, 'support_bbox': support,
             'parent_index': index, 'parent_sha256': _digest(parent_identity),
             'owned_indices': owned, 'neighbor_indices': neighbors, 'config': config,
             'roi_visible_pixels': int(inside.sum()), 'support_visible_pixels': len(gx),
             'context_crops': contexts}
    proof['sha256'] = _digest(proof)
    return {'parent_index': index, 'context_crops': contexts, 'proof': proof}


def audit_local_partition(proposal, mask, bbox, entries, owned_indices, neighbor_indices,
                          *, padding=3, minimum_blank_band=2,
                          maximum_parent_height_ratio=1.8):
    """Regenerate against current inputs; altered/extra/missing fields fail closed."""
    if not isinstance(proposal, dict):
        return False
    try:
        expected = prove_local_partition(
            mask, bbox, entries, owned_indices, neighbor_indices, padding=padding,
            minimum_blank_band=minimum_blank_band,
            maximum_parent_height_ratio=maximum_parent_height_ratio)
        return expected is not None and proposal == expected
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
