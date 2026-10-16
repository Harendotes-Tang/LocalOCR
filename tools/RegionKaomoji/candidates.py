"""Bounded original-pixel candidates; no scene labels or character coordinates.

Both branches consume the same original RGB canvas. OCR geometry is one source;
foreground connected components provide an independent source when OCR omitted a
face. Pixel shapes are routing evidence, not an independent text prediction.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import unicodedata

import cv2
import numpy as np
from PIL import Image

from tools.PretrainedOcr import kaomoji_grammar as kg

SCHEMA = 'localocr.region-candidates.v1'
DEFAULT_CONFIG = {
    'maximum_pixels': 50_000_000, 'maximum_baseline_regions': 4096,
    'maximum_candidates': 48, 'maximum_crop_pixels': 1_000_000,
    'maximum_components': 8192, 'maximum_bracket_scan_width': 4096,
    'contrast_threshold': 8, 'minimum_component_area': 2,
    'minimum_height': 7, 'maximum_height': 72, 'maximum_width': 1024,
    'line_overlap': 0.45, 'horizontal_gap_height': 0.58,
    'padding': 3, 'baseline_coverage': 0.80, 'baseline_ink_coverage': 0.98, 'merge_iou': 0.45,
    'minimum_context_blank_band': 2,
    'background_window': 51, 'exterior_bridge_ratio': 0.45, 'shape_contrast_threshold': 8,
}
# Keys introduced after a configuration was frozen take their default when absent, so frozen files stay valid.
# ``shape_contrast_threshold`` defaults to ``contrast_threshold`` (None marks "same as the base mask"): the
# core-ink mask it selects feeds bracket shape detection and pair interior counting only, never envelopes.
OPTIONAL_CONFIG = {'background_window': 51, 'exterior_bridge_ratio': 0.45, 'shape_contrast_threshold': None}
# Structured keys of the browser-font round. They are validated when present and never inserted when absent,
# so every earlier frozen configuration normalizes (and hashes) exactly as before.
# ``small_text_shape``: units lower than ``below_height`` detect bracket shapes on their local contrast map
# upsampled ``upsample`` times (shape evidence only). ``low_mark_guard``: an exterior cell no taller than
# ``maximum_height_ratio`` of the bracket height whose top lies below ``minimum_top_ratio`` of it (a low
# mark such as a line-end comma or full stop) is never bridged into a face candidate.
STRUCTURED_OPTIONAL = ('small_text_shape', 'low_mark_guard')
# Bridging from a bracket pair absorbs exterior atoms (arms, tails) but must stop at body glyphs. On the
# development scenes exterior atom cells are never "full-square" while most attached CJK body glyph cells
# are; cells group components whose x-ranges lie within CELL_TOLERANCE pixels (radicals of one glyph).
CELL_TOLERANCE = 2
FULL_SQUARE_RATIO = .65
BODY_DENSITY = .4
# Every grammar set is a projection of the canonical atom table; no literal copy lives here.
ATOMS = kg.default_atoms()
FACE_LETTERS = kg.face_letters(ATOMS)
BRACKETS = kg.bracket_pairs(ATOMS)
DEFAULT_SPLIT = {
    'enabled': True, 'blank_band_ratio': 0.5, 'minimum_band_pixels': 4, 'minimum_child_width_ratio': 0.35,
    'minimum_child_ink': 6, 'maximum_children': 6, 'child_padding': 2, 'maximum_parent_width': 1600,
}


def split_configuration(config=None):
    """Validate the pre-recognition split settings; a missing section means disabled."""
    value = copy.deepcopy(DEFAULT_SPLIT if config is None else config)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_SPLIT):
        raise ValueError('Unknown or incomplete split configuration')
    if type(value['enabled']) is not bool:
        raise ValueError('Split enabled flag must be boolean')
    for key in ('minimum_band_pixels', 'minimum_child_ink', 'maximum_children', 'child_padding', 'maximum_parent_width'):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('Split integer limits must be nonnegative integers')
    for key in ('blank_band_ratio', 'minimum_child_width_ratio'):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or not 0 < item <= 4:
            raise ValueError('Split ratios must be finite positive numbers')
    if not 2 <= value['maximum_children'] <= 16 or value['minimum_band_pixels'] < 2 or value['minimum_child_ink'] < 1:
        raise ValueError('Split limits outside the supported range')
    return value


def split_children(mask, bbox, config=None):
    """Split one first-pass box at measured blank column bands; no text or truth is consulted.

    Returns child boxes in reading order, or an empty list when the box has no
    qualifying band, a child would be too narrow or faint, or limits are hit.
    """
    config = split_configuration(config)
    left, top, right, bottom = bbox
    if not 0 <= left < right <= mask.shape[1] or not 0 <= top < bottom <= mask.shape[0]:
        raise ValueError('Split box outside the original canvas')
    height, width = bottom - top, right - left
    if not config['enabled'] or width > config['maximum_parent_width']:
        return []
    local = mask[top:bottom, left:right]
    columns = local.any(axis=0)
    if not columns.any():
        return []
    minimum_band = max(config['minimum_band_pixels'], int(round(height * config['blank_band_ratio'])))
    segments = []
    index = 0
    while index < width:
        if not columns[index]:
            index += 1
            continue
        start = index
        while index < width and columns[index]:
            index += 1
        segments.append([start, index])
    merged = [segments[0]]
    for start, end in segments[1:]:
        if start - merged[-1][1] >= minimum_band:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    if not 2 <= len(merged) <= config['maximum_children']:
        return []
    minimum_width = max(1, int(round(height * config['minimum_child_width_ratio'])))
    children = []
    for position, (start, end) in enumerate(merged):
        ink = int(local[:, start:end].sum())
        if end - start < minimum_width or ink < config['minimum_child_ink']:
            return []
        pad = config['child_padding']
        # Children stay tight around their own ink: a padded child never reaches a neighbour's ink,
        # so an automatic face candidate beside it is not reported as overlapping body geometry.
        lower = 0 if position == 0 else merged[position - 1][1]
        upper = width if position == len(merged) - 1 else merged[position + 1][0]
        child_left = max(lower, start - pad)
        child_right = min(upper, end + pad)
        children.append([left + child_left, top, left + child_right, bottom])
    return children


def strip_exterior_atoms(text, atoms=None):
    """Remove surrounding whitespace and declared arm/tail atom characters from both ends."""
    atoms = ATOMS if atoms is None else atoms
    leading = {char for atom in atoms.arms_left for char in atom}
    trailing = {char for role in ('arms_right', 'tails') for atom in getattr(atoms, role) for char in atom}
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or text[start] in leading):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in trailing):
        end -= 1
    return text[start:end]


def ordinary_anchor_evidence(text, atoms=None):
    """Letters outside the outermost declared bracket span and any digits, from the baseline's own reading.

    Letters inside the bracket span may be baseline misreads of face glyphs and
    are not evidence on their own; digits and exterior non-face letters are.
    """
    atoms = ATOMS if atoms is None else atoms
    bracket_chars = atoms.bracket_chars
    exterior_atoms = {char for role in ('arms_left', 'arms_right', 'tails') for atom in getattr(atoms, role) for char in atom}
    positions = [index for index, char in enumerate(text) if char in bracket_chars]
    low, high = (positions[0], positions[-1]) if positions else (len(text), -1)
    outside = [char for index, char in enumerate(text)
               if unicodedata.category(char).startswith('L') and char not in exterior_atoms and not low < index < high]
    digits = [char for char in text if unicodedata.category(char).startswith('N')]
    return outside, digits


def interior_glyphs(text, atoms=None):
    """Non-space glyph count strictly inside the outermost declared bracket span (0 without a span)."""
    atoms = ATOMS if atoms is None else atoms
    positions = [index for index, char in enumerate(text) if char in atoms.bracket_chars]
    if len(positions) < 2:
        return 0
    return sum(1 for char in text[positions[0] + 1:positions[-1]] if not char.isspace())


def single_bracket_pair(text, atoms=None):
    """True when the text holds exactly one declared bracket pair at its ends."""
    atoms = ATOMS if atoms is None else atoms
    positions = [(index, char) for index, char in enumerate(text) if char in atoms.bracket_chars]
    return (len(positions) == 2 and positions[0][0] == 0 and positions[1][0] == len(text) - 1
            and (positions[0][1], positions[1][1]) in atoms.brackets)


def configuration(config=None):
    value = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
    if isinstance(value, dict):
        value = {**{key: default for key, default in OPTIONAL_CONFIG.items() if key not in value}, **value}
        if value.get('shape_contrast_threshold') is None:
            value['shape_contrast_threshold'] = value.get('contrast_threshold')
    if not isinstance(value, dict) or set(value) - set(STRUCTURED_OPTIONAL) != set(DEFAULT_CONFIG):
        raise ValueError('Unknown or incomplete candidate configuration')
    _structured_options(value)
    for key, default in DEFAULT_CONFIG.items():
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError('Candidate configuration requires finite numeric values')
        if isinstance(default, int) and not isinstance(item, int):
            raise ValueError('Integer candidate limit required')
        if item <= 0:
            raise ValueError('Candidate configuration must be positive')
    for key in ('line_overlap', 'baseline_coverage', 'baseline_ink_coverage', 'merge_iou'):
        if not 0 < value[key] <= 1:
            raise ValueError('Candidate overlap outside (0,1]')
    if value['maximum_pixels'] > DEFAULT_CONFIG['maximum_pixels'] or value['maximum_candidates'] > 48 or value['maximum_crop_pixels'] > 1_000_000 or value['maximum_baseline_regions'] > 4096 or value['maximum_components'] > 8192 or value['maximum_bracket_scan_width'] > 4096:
        raise ValueError('Research resource ceiling cannot be increased')
    if value['minimum_context_blank_band'] < 2:
        raise ValueError('Mixed-region blank band must contain at least two pixels')
    if value['minimum_height'] > value['maximum_height'] or value['contrast_threshold'] > 255:
        raise ValueError('Invalid candidate size/contrast limits')
    shape = value['shape_contrast_threshold']
    if not isinstance(shape, int) or not 1 <= shape <= 255:
        raise ValueError('Shape contrast threshold must be an integer between 1 and 255')
    window = value['background_window']
    if not isinstance(window, int) or window % 2 == 0 or not 3 <= window <= 201:
        raise ValueError('Background window must be an odd integer between 3 and 201 pixels')
    ratio = value['exterior_bridge_ratio']
    if not isinstance(ratio, float) or not .3 <= ratio <= .8:
        raise ValueError('Exterior bridge ratio must be a float between 0.3 and 0.8 of the bracket height')
    return value


def _structured_options(value):
    if any(key in value and value[key] is None for key in STRUCTURED_OPTIONAL):
        raise ValueError('Omit an unused structured candidate key instead of setting it to null')
    small = value.get('small_text_shape')
    if small is not None and (not isinstance(small, dict) or set(small) != {'below_height', 'upsample'}
                              or any(isinstance(small[k], bool) or not isinstance(small[k], int) for k in small)
                              or not 8 <= small['below_height'] <= 64 or not 2 <= small['upsample'] <= 4):
        raise ValueError('small_text_shape needs integer below_height 8..64 and upsample 2..4')
    guard = value.get('low_mark_guard')
    if guard is not None and (not isinstance(guard, dict) or set(guard) != {'maximum_height_ratio', 'minimum_top_ratio'}
                              or any(not isinstance(guard[k], float) or not 0 < guard[k] < 1 for k in guard)):
        raise ValueError('low_mark_guard needs float maximum_height_ratio and minimum_top_ratio in (0,1)')


def rgb_image(image, maximum_pixels=50_000_000):
    array = np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError('Original image must be RGB uint8')
    if min(array.shape[:2]) <= 0 or array.shape[0] * array.shape[1] > maximum_pixels:
        raise ValueError('Image exceeds pixel ceiling or is empty')
    return np.ascontiguousarray(array)


def image_digest(image):
    array = rgb_image(image)
    return hashlib.sha256(array.shape[1].to_bytes(8, 'big') + array.shape[0].to_bytes(8, 'big') + array.tobytes()).hexdigest()


def baseline_regions(baseline, size, maximum=4096):
    if not isinstance(baseline, list) or len(baseline) > maximum:
        raise ValueError('Baseline must be a bounded region list')
    width, height = size
    result = []
    for index, item in enumerate(baseline):
        if not isinstance(item, dict) or not {'bbox', 'text', 'score'} <= set(item):
            raise ValueError('Missing baseline geometry/text/score')
        box = item['bbox']
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in box):
            raise ValueError('Baseline half-open box must contain four integers')
        left, top, right, bottom = map(int, box)
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError('Baseline box outside original canvas')
        if not isinstance(item['text'], str) or len(item['text']) > 4096 or isinstance(item['score'], (bool,str)) or not isinstance(item['score'], (int,float,np.number)) or not math.isfinite(float(item['score'])):
            raise ValueError('Baseline text/score invalid')
        result.append({'bbox': [left, top, right, bottom], 'text': item['text'], 'score': float(item['score']), 'index': index})
    return result


def baseline_parent_digest(entry):
    """Bind a validated parent's index, box and raw text without normalization.

    The parent must be an entry returned by baseline_regions. Scores remain
    uncalibrated observations and do not identify the text/geometry receipt.
    """
    receipt = {'schema_version': 'localocr.baseline-parent.v1',
               'index': entry['index'], 'bbox': entry['bbox'], 'text': entry['text']}
    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def foreground(image, threshold=8, window=None):
    """Ink above a fixed contrast against the local background.

    The background is the per-channel median over ``window`` pixels, so light
    text on dark cards and dark text on light pages both yield glyph ink while
    flat card bodies stay background. ``window`` must exceed glyph strokes and
    is validated by ``configuration``.
    """
    return np.ascontiguousarray((contrast_map(image, window) >= threshold).astype(np.uint8))


def contrast_map(image, window=None):
    """Per-pixel maximum channel distance from the local median background (the quantity ``foreground`` thresholds)."""
    array = rgb_image(image)
    window = DEFAULT_CONFIG['background_window'] if window is None else window
    if not isinstance(window, int) or window % 2 == 0 or not 3 <= window <= 201:
        raise ValueError('Background window must be an odd integer between 3 and 201 pixels')
    background = cv2.medianBlur(np.ascontiguousarray(array), window)
    return np.max(np.abs(array.astype(np.int16) - background.astype(np.int16)), axis=2)


def _small_shapes(contrast, box, config):
    """Bracket pairs of one small unit found on its upsampled contrast, mapped back to image pixels.

    Returns (pairs in image coordinates, shape components in unit-local original coordinates). Only shape
    evidence changes: candidate boxes, masks and every other stage keep the original resolution.
    """
    k = config['small_text_shape']['upsample']
    left, top, right, bottom = box
    local = contrast[top:bottom, left:right].astype(np.float32)
    up = cv2.resize(local, ((right - left) * k, (bottom - top) * k), interpolation=cv2.INTER_LINEAR)
    core = np.ascontiguousarray((up >= config['shape_contrast_threshold']).astype(np.uint8))
    def back(b, x0=0, y0=0):
        return [x0 + b[0] // k, y0 + b[1] // k, x0 + -(-b[2] // k), y0 + -(-b[3] // k)]
    pairs = [{**pair, 'left_bbox': back(pair['left_bbox'], left, top), 'right_bbox': back(pair['right_bbox'], left, top), 'upsample': k}
             for pair in _bracket_shapes(core, (0, 0))]
    components = [{**item, 'bbox': back(item['bbox'])} for item in _components(core, config['minimum_component_area'] * k * k)]
    return pairs, components


def _small_unit(config, height):
    small = config.get('small_text_shape')
    return small is not None and height < small['below_height']


def _bracket_extent(components, strokes):
    """(top, height) of the bracket pair measured on the same ink mask as ``components``.

    Strokes may come from the stricter core mask, whose glyphs are about a pixel smaller than base-mask
    components; each stroke is therefore replaced by the smallest base component that intersects it (the
    bracket glyph itself), keeping the stroke when none does.
    """
    boxes = []
    for stroke in strokes:
        hits = [item['bbox'] for item in components if intersection(item['bbox'], stroke)]
        boxes.append(min(hits, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])) if hits else stroke)
    return min(bb[1] for bb in boxes), max(bb[3] - bb[1] for bb in boxes)


def _low_mark(box, extent, guard):
    """A low, small exterior mark relative to the bracket pair (a line-end comma or full stop).

    ``extent`` is the bracket (top, height) on the mark's own mask. Height and width both stay within
    ``maximum_height_ratio`` of the bracket height and the top lies below ``minimum_top_ratio`` of it;
    wide low arcs such as a tail arch are therefore never low marks.
    """
    top, height = extent
    limit = guard['maximum_height_ratio'] * height
    return (box[3] - box[1]) <= limit and (box[2] - box[0]) <= limit and box[1] >= top + guard['minimum_top_ratio'] * height


def _strip_low_marks(components, strokes, guard):
    """Drop outermost low marks beyond the bracket strokes; returns (kept components, stripped sides).

    A mark must be an isolated column: ink stacked above it (the dot of an exclamation mark) belongs to
    the same glyph and is kept.
    """
    a, b = strokes
    extent = _bracket_extent(components, strokes)
    kept, sides = list(components), set()
    def isolated(item):
        x0, x1 = item['bbox'][0], item['bbox'][2]
        return not any(other is not item and other['bbox'][0] < x1 and other['bbox'][2] > x0 for other in kept)
    for side in ('right', 'left'):
        while True:
            beyond = [item for item in kept if (item['bbox'][0] >= b[2] if side == 'right' else item['bbox'][2] <= a[0])]
            if not beyond:
                break
            outer = max(beyond, key=lambda item: item['bbox'][2]) if side == 'right' else min(beyond, key=lambda item: item['bbox'][0])
            if not (_low_mark(outer['bbox'], extent, guard) and isolated(outer)):
                break
            kept.remove(outer)
            sides.add(side)
    return kept, sides


def _trim_low_marks(box, pair, guard):
    """Cut a candidate envelope before outermost low marks lying beyond the bracket pair on either side."""
    box = list(box)
    inside = [item for cell in pair['cells'] for item in cell['components'] if intersection(item['bbox'], box)]
    kept, sides = _strip_low_marks(inside, pair['strokes'], guard)
    if kept and 'right' in sides:
        box[2] = min(box[2], max(item['bbox'][2] for item in kept))
    if kept and 'left' in sides:
        box[0] = max(box[0], min(item['bbox'][0] for item in kept))
    return box


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def is_face_text(text, allow_arms=True):
    """Validate one complete expression against the canonical atom grammar.

    Raw Unicode is never modified. A separate letter token or a second
    expression cannot enlarge a replacement span. Mixed baseline splices
    prohibit exterior arms because context OCR cannot distinguish an omitted
    body letter from an arm outside the paired strokes. Pixel, body and view
    confirmation still belong to fusion.
    """
    return kg.is_face_text(text, ATOMS, allow_arms)


def _components(mask, minimum_area):
    count, _, statistics, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [{'bbox': [int(x), int(y), int(x+w), int(y+h)], 'pixels': int(pixels)}
            for x, y, w, h, pixels in statistics[1:count] if pixels >= minimum_area]


def _physical_lines(components, config):
    lines = []
    for component in sorted(components, key=lambda c: (c['bbox'][1], c['bbox'][0])):
        box = component['bbox']
        selected = None
        for line in lines:
            overlap = min(line['bbox'][3], box[3]) - max(line['bbox'][1], box[1])
            minimum = min(line['bbox'][3]-line['bbox'][1], box[3]-box[1])
            if overlap >= minimum * config['line_overlap']:
                selected = line
                break
        if selected is None:
            lines.append({'bbox': list(box), 'components': [component]})
        else:
            selected['components'].append(component)
            selected['bbox'] = [min(selected['bbox'][0], box[0]), min(selected['bbox'][1], box[1]), max(selected['bbox'][2], box[2]), max(selected['bbox'][3], box[3])]
    return lines


def _line_chunks(line, config):
    """Split one physical line into chunks at horizontal gaps wider than the line-height ratio."""
    chunks = []
    # Tall emblems or icons on a text line must not widen the gap that joins chunks: the line
    # height is the tallest component that is not an outlier against the 80th-percentile height,
    # so bracket strokes of an expression still set the scale while a 2-3x taller icon does not.
    heights = sorted(item['bbox'][3] - item['bbox'][1] for item in line['components'])
    typical = heights[int(.8 * (len(heights) - 1))]
    line_height = max(h for h in heights if h <= typical * 1.3)
    for component in sorted(line['components'], key=lambda c: c['bbox'][0]):
        box = component['bbox']
        if chunks and box[0] - chunks[-1]['bbox'][2] <= line_height * config['horizontal_gap_height']:
            chunk = chunks[-1]
            chunk['components'].append(component)
            chunk['bbox'] = [min(chunk['bbox'][0], box[0]), min(chunk['bbox'][1], box[1]), max(chunk['bbox'][2], box[2]), max(chunk['bbox'][3], box[3])]
        else:
            chunks.append({'bbox': list(box), 'components': [component]})
    return chunks


def _line_groups(components, config):
    groups = []
    for line in _physical_lines(components, config):
        groups.extend(_line_chunks(line, config))
    return groups


def _structural_split(components, config):
    """Separate structural components (card borders, separators, whole columns) from glyph-sized ones.

    A component taller than the maximum glyph height or wider than the maximum crop width can never
    belong to a candidate; left in place it would merge every text line it overlaps into one line.
    """
    glyphs, structural = [], []
    for component in components:
        left, top, right, bottom = component['bbox']
        (structural if bottom - top > config['maximum_height'] or right - left > config['maximum_width'] else glyphs).append(component)
    return glyphs, structural


def _right_angle_profile(local):
    """Measure one straight stem with separate inward top/bottom end caps.

    End caps occupy a small number of rows, so average curvature is not a
    criterion here. A second vertical rail, missing cap or broad middle stroke
    fails this branch; closed table frames cannot supply two isolated brackets.
    """
    h, w = local.shape
    if w < 3:
        return None
    band = max(1, int(math.ceil(h * .20)))
    middle = local[band:h-band]
    if not len(middle) or not np.all(middle.any(axis=1)):
        return None
    widths = [int(np.flatnonzero(row)[-1]-np.flatnonzero(row)[0]+1) for row in middle]
    if max(widths) > max(1, int(math.ceil(w * .35))):
        return None
    support = middle.sum(axis=0)
    tied = np.flatnonzero(support == support.max())
    # A thick stem occupies several equally supported columns. np.argmax
    # always chooses its left edge, which rejects the mirrored right bracket.
    # Check the two outer tied columns under the same coverage/cap limits.
    for stem in dict.fromkeys((int(tied[0]), int(tied[-1]))):
        coverage = float(np.count_nonzero(local[:, stem]) / h)
        if coverage < .85:
            continue
        edge_band = max(0, int(math.floor(w * .25)))
        if stem <= edge_band:
            direction = 'left'
            extensions = [int(np.flatnonzero(local[:band].any(axis=0))[-1]-stem),
                          int(np.flatnonzero(local[h-band:].any(axis=0))[-1]-stem)]
        elif stem >= w-1-edge_band:
            direction = 'right'
            extensions = [int(stem-np.flatnonzero(local[:band].any(axis=0))[0]),
                          int(stem-np.flatnonzero(local[h-band:].any(axis=0))[0])]
        else:
            continue
        minimum_cap = max(2, int(math.ceil(w * .50)))
        if min(extensions) < minimum_cap:
            continue
        return {'kind': 'right-angle', 'direction': direction,
                'measurements': {'stem_column': stem, 'stem_coverage': coverage,
                                 'top_cap_extension': extensions[0],
                                 'bottom_cap_extension': extensions[1],
                                 'maximum_middle_ink_width': max(widths)}}
    return None


def _small_curve_profile(profile, h):
    """Require matching endpoint extrema for a quantized 6..16 pixel curve.

    Both ends must bend in the same direction away from the central stroke.
    This bounded scale branch leaves the original average-curvature threshold
    intact for every larger component and records its actual displacements.
    """
    if not 6 <= h <= 16:
        return None
    n = len(profile)
    middle = profile[n//3:max(n//3+1, 2*n//3)]
    center = float(np.median(middle))
    if float(np.ptp(middle)) > .75:
        return None
    band = max(1, n//4)
    top = profile[:band]-center
    bottom = profile[-band:]-center
    for direction, sign in (('left', 1), ('right', -1)):
        upper = float(np.max(top*sign))
        lower = float(np.max(bottom*sign))
        if upper >= .75 and lower >= .75 and np.min(top*sign) >= -.25 and np.min(bottom*sign) >= -.25:
            return {'kind': 'small-curve', 'direction': direction,
                    'measurements': {'top_endpoint_displacement': upper,
                                     'bottom_endpoint_displacement': lower,
                                     'middle_profile_range': float(np.ptp(middle))}}
    return None


def _bracket_shapes(mask, origin):
    """Find and audit separate curved, small-curve and right-angle pixel pairs."""
    components = _components(mask, 2)
    if not components:
        return []
    ink_top = min(c['bbox'][1] for c in components)
    ink_bottom = max(c['bbox'][3] for c in components)
    ink_height = ink_bottom - ink_top
    shapes = []
    for component in components:
        left, top, right, bottom = component['bbox']
        h, w = bottom-top, right-left
        if h < max(6, ink_height*.60) or w > h*.55 or component['pixels']/(h*w) > .78:
            continue
        local = mask[top:bottom, left:right]
        rows = [(int(np.where(row)[0].min()), int(np.where(row)[0].max())) for row in local if row.any()]
        if len(rows) < h*.65:
            continue
        profile = np.array([(a+b)/2 for a,b in rows])
        edge_count = max(1, len(profile)//5)
        edge = float(np.mean(np.concatenate((profile[:edge_count], profile[-edge_count:]))))
        middle = float(np.mean(profile[len(profile)//3:max(len(profile)//3+1, 2*len(profile)//3)]))
        curvature = middle-edge
        observed = _right_angle_profile(local)
        if observed is None and abs(curvature) >= (.65 if h < 12 else 1.0):
            observed = {'kind': 'curve', 'direction': 'left' if curvature < 0 else 'right',
                        'measurements': {'average_curvature': curvature,
                                         'required_absolute_curvature': .65 if h < 12 else 1.0}}
        if observed is None:
            observed = _small_curve_profile(profile, h)
        if observed is None:
            continue
        shapes.append({'bbox': [left+origin[0], top+origin[1], right+origin[0], bottom+origin[1]],
                       'height': h, **observed})
    pairs = []
    for left in shapes:
        for right in shapes:
            a, b = left['bbox'], right['bbox']
            if left['direction'] != 'left' or right['direction'] != 'right' or a[2] >= b[0]:
                continue
            span = b[2]-a[0]
            if not 1.2*max(left['height'], right['height']) <= span <= 14*max(left['height'], right['height']):
                continue
            if min(left['height'], right['height']) / max(left['height'], right['height']) < .65:
                continue
            if abs((a[1]+a[3])-(b[1]+b[3])) > max(left['height'], right['height'])*.6:
                continue
            pairs.append({'left_bbox': a, 'right_bbox': b,
                          'shape_kinds': sorted({left['kind'], right['kind']}),
                          'left_shape': {'kind': left['kind'], 'measurements': left['measurements']},
                          'right_shape': {'kind': right['kind'], 'measurements': right['measurements']}})
    return pairs


def _glyph_cells(components, tolerance=CELL_TOLERANCE):
    """Group components of one physical line into glyph cells: x-ranges overlapping or within ``tolerance``.

    Returns (cells, owner) where cells is a list of {'bbox', 'components'} sorted by x and owner maps a
    component index to its cell index. Radicals of one CJK glyph share a cell; an arm stroke is its own cell.
    """
    order = sorted(range(len(components)), key=lambda i: components[i]['bbox'][0])
    cells, owner = [], {}
    reach = None
    for index in order:
        box = components[index]['bbox']
        if cells and box[0] - reach <= tolerance:
            cell = cells[-1]
            cell['bbox'] = [min(cell['bbox'][0], box[0]), min(cell['bbox'][1], box[1]), max(cell['bbox'][2], box[2]), max(cell['bbox'][3], box[3])]
            cell['components'].append(components[index])
            reach = max(reach, box[2])
        else:
            cells.append({'bbox': list(box), 'components': [components[index]]})
            reach = box[2]
        owner[index] = len(cells) - 1
    return cells, owner


def _full_square(cell, height):
    box = cell['bbox']
    return (box[3] - box[1]) >= height * FULL_SQUARE_RATIO and (box[2] - box[0]) >= height * FULL_SQUARE_RATIO


def _body_like(cell, height):
    """A full-square cell whose ink density reaches BODY_DENSITY: a body glyph on the development scenes.

    Full-square exterior atoms (⊂ ∩ つ ☆ outlines) are thin, so density separates them from CJK body
    glyphs: at 0.4 no outline arm and 1% of exterior cells are flagged while 71% of attached body cells are.
    """
    box = cell['bbox']
    area_ = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    density = sum(item['pixels'] for item in cell['components']) / area_
    return _full_square(cell, height) and density >= BODY_DENSITY


def _outside_body_components(pairs, components, height, origin=(0, 0)):
    """Drop core-mask bracket pairs whose stroke lies inside a dense full-square base-mask component.

    Removing antialiasing halos also separates the strokes of dense CJK glyphs, and a thin curved
    stroke inside one would otherwise pass for a bracket. The test is per base component (the glyph
    the halo held together), not per glyph cell: a bracket glued to its cheek or eye forms a sparse
    component and survives, while a dense body glyph (full-square at ``height``, density >= BODY_DENSITY)
    rejects every stroke found inside it.
    """
    dense = []
    for component in components:
        if _body_like({'bbox': component['bbox'], 'components': [component]}, height):
            box = component['bbox']
            dense.append([box[0] + origin[0] - 1, box[1] + origin[1] - 1, box[2] + origin[0] + 1, box[3] + origin[1] + 1])
    if not dense:
        return pairs
    def inside(stroke):
        return any(stroke[0] >= box[0] and stroke[2] <= box[2] and stroke[1] >= box[1] and stroke[3] <= box[3] for box in dense)
    return [pair for pair in pairs if not inside(pair['left_bbox']) and not inside(pair['right_bbox'])]


def _line_scale(line):
    """Line height for scaling: the tallest component that is not an outlier against the 80th percentile."""
    heights = sorted(item['bbox'][3] - item['bbox'][1] for item in line['components'])
    typical = heights[int(.8 * (len(heights) - 1))]
    return max(h for h in heights if h <= typical * 1.3)


def _bracket_groups(mask, components, config, guard_body_cells=False, contrast=None):
    """Use actual paired strokes to bridge internal spaces, without label hints.

    Pairs are detected per chunk and, when the line height is in range, per line, so a taller icon or
    body glyph elsewhere on the line cannot hide bracket strokes behind the whole-line ink-height filter.
    A group is the run of glyph cells around the pair: every cell inside the pair span, then cells
    outward on each side while the gap stays within the bridge ratio of the line height, the cell is not
    taller than the line and is not a full-square glyph (a body character on the development scenes).
    """
    groups=[]
    ratio=config['exterior_bridge_ratio']
    for line in _physical_lines(components,config):
        line_components=line['components']
        cells,_=_glyph_cells(line_components)
        height=_line_scale(line)
        seen=set()
        for unit in (*_line_chunks(line,config),line):
            left,top,right,bottom=unit['bbox'];unit_height=bottom-top
            if not config['minimum_height']<=unit_height<=config['maximum_height'] or right-left>config['maximum_bracket_scan_width']:
                continue
            if contrast is not None and _small_unit(config,unit_height):
                found=_small_shapes(contrast,unit['bbox'],config)[0]
            else:
                found=_bracket_shapes(mask[top:bottom,left:right],(left,top))
            if guard_body_cells or (contrast is not None and _small_unit(config,unit_height)):
                found=_outside_body_components(found,line_components,height)
            for pair in found:
                key=(tuple(pair['left_bbox']),tuple(pair['right_bbox']))
                if key in seen:
                    continue
                seen.add(key)
                a,b=pair['left_bbox'],pair['right_bbox']
                inside=[i for i,cell in enumerate(cells) if cell['bbox'][0]<b[2] and cell['bbox'][2]>a[0]]
                if not inside:
                    continue
                low,high=min(inside),max(inside)
                guard=config.get('low_mark_guard')
                bracket=_bracket_extent(line_components,(a,b)) if guard is not None else None
                for step in (-1,1):
                    index=(low if step<0 else high)+step
                    extent=cells[low]['bbox'][0] if step<0 else cells[high]['bbox'][2]
                    while 0<=index<len(cells):
                        cell=cells[index];box=cell['bbox']
                        gap=extent-box[2] if step<0 else box[0]-extent
                        if gap>height*ratio or box[3]-box[1]>height*1.25 or _body_like(cell,height):
                            break
                        if guard is not None and _low_mark(box,bracket,guard):
                            break
                        extent=box[0] if step<0 else box[2]
                        if step<0: low=index
                        else: high=index
                        index+=step
                selected=[item for cell in cells[low:high+1] for item in cell['components']]
                if guard is not None:
                    selected=_strip_low_marks(selected,(a,b),guard)[0]
                box=[min(item['bbox'][0] for item in selected),min(item['bbox'][1] for item in selected),max(item['bbox'][2] for item in selected),max(item['bbox'][3] for item in selected)]
                if box[2]-box[0]<=config['maximum_width']:
                    groups.append({'bbox':box,'components':selected,'pair_span':[a[0],b[2]],'scale':height,
                                   'line_cells':cells,'strokes':[list(a),list(b)]})
    return groups


def _masks(array, config):
    """Base ink mask and the core-ink mask for shape detection (the same object when thresholds agree)."""
    mask = foreground(array, config['contrast_threshold'], config['background_window'])
    if config['shape_contrast_threshold'] == config['contrast_threshold']:
        return mask, mask
    return mask, foreground(array, config['shape_contrast_threshold'], config['background_window'])


def _crop_evidence(array, bbox, config, mask, core, contrast=None):
    """Pixel evidence for one crop plus the (base, core) bracket-pair counts used for diagnostics.

    Envelope facts (pixel count, components, edge ink, support) come from the base mask; bracket
    strokes and the pair interior count come from the core mask, so antialiasing halos that glue a
    bracket to its neighbour cannot hide the stroke while never changing what the crop covers.
    """
    left, top, right, bottom = bbox
    local = mask[top:bottom, left:right]
    components = _components(local, config['minimum_component_area'])
    occupied = np.argwhere(local)
    edge_ink = bool(local[0].any() or local[-1].any() or local[:,0].any() or local[:,-1].any())
    ink_height = (max(item['bbox'][3] for item in components) - min(item['bbox'][1] for item in components)) if components else 0
    if contrast is not None and components and _small_unit(config, ink_height):
        pairs, shape_components = _small_shapes(contrast, bbox, config)
        if pairs:
            pairs = _outside_body_components(pairs, components, ink_height, (left, top))
        base_pairs = _bracket_shapes(local, (left, top))
    elif core is mask:
        shape_local, shape_components = local, components
        pairs = _bracket_shapes(local, (left, top))
        base_pairs = pairs
    else:
        shape_local = core[top:bottom, left:right]
        shape_components = _components(shape_local, config['minimum_component_area'])
        pairs = _bracket_shapes(shape_local, (left, top))
        if pairs and components:
            ink_height = max(item['bbox'][3] for item in components) - min(item['bbox'][1] for item in components)
            pairs = _outside_body_components(pairs, components, ink_height, (left, top))
        base_pairs = _bracket_shapes(local, (left, top))
    shape_counts = {kind: sum(kind in pair['shape_kinds'] for pair in pairs)
                    for kind in ('curve', 'small-curve', 'right-angle')}
    evidence = {'pixel_count': int(local.sum()), 'component_count': len(components),
                'edge_ink': edge_ink, 'paired_bracket_shapes': pairs, 'bracket_shape_counts': shape_counts,
                'pixel_support': bool(len(occupied) >= 6 and len(components) >= 3),
                'extra_shape_support': bool(pairs),
                'pair_geometry': pair_geometry(pairs, shape_components, (left, top)),
                'image_sha256': image_digest(array)}
    return evidence, {'base_pairs': len(base_pairs), 'core_pairs': len(pairs)}


def pixel_evidence(image, bbox, config=None):
    config = configuration(config)
    array = rgb_image(image, config['maximum_pixels'])
    left, top, right, bottom = bbox
    if not 0 <= left < right <= array.shape[1] or not 0 <= top < bottom <= array.shape[0]:
        raise ValueError('Candidate box outside image')
    mask, core = _masks(array, config)
    contrast = contrast_map(array, config['background_window']) if config.get('small_text_shape') is not None else None
    return _crop_evidence(array, bbox, config, mask, core, contrast)[0]


def pair_geometry(pairs, components, origin):
    """Interior component count and exterior ink presence around the widest bracket pair.

    Components are crop-local; the result is pixel evidence only. A component
    counts as interior when its horizontal centre lies between the two bracket
    strokes, and as exterior ink when it lies entirely beyond a stroke.
    """
    if not pairs:
        return None
    pair = max(pairs, key=lambda item: item['right_bbox'][2] - item['left_bbox'][0])
    # Count pairs whose spans do not overlap: two complete expressions side by side.
    spans = sorted((item['left_bbox'][0], item['right_bbox'][2]) for item in pairs)
    disjoint, reach = 0, -1
    for start, end in spans:
        if start >= reach:
            disjoint += 1
            reach = end
    left_edge, right_edge = pair['left_bbox'][0], pair['right_bbox'][2]
    inner_left, inner_right = pair['left_bbox'][2], pair['right_bbox'][0]
    interior, exterior_left, exterior_right = 0, False, False
    for component in components:
        x0, _, x1, _ = component['bbox']
        x0 += origin[0]
        x1 += origin[0]
        centre = (x0 + x1) / 2
        if inner_left <= centre <= inner_right and x0 >= inner_left - 1 and x1 <= inner_right + 1:
            interior += 1
        elif x1 <= left_edge + 1 and centre < left_edge:
            exterior_left = True
        elif x0 >= right_edge - 1 and centre > right_edge:
            exterior_right = True
    return {'left_bbox': list(pair['left_bbox']), 'right_bbox': list(pair['right_bbox']), 'disjoint_pairs': disjoint,
            'interior_components': interior, 'exterior_ink': {'left': exterior_left, 'right': exterior_right}}


def exterior_atoms(text, atoms=None):
    """Whether a confirmed face carries exterior atoms on each side (arms left; arms or tails right)."""
    parsed = kg.parse_face(text, ATOMS if atoms is None else atoms)
    if parsed is None:
        return None
    return {'left': bool(parsed['arms_left']), 'right': bool(parsed['arms_right'] or parsed['tails'])}


def _mixed_contexts(mask, bbox, owned, neighbors, entries, config):
    """Partition one broad OCR envelope only at measured blank pixel bands."""
    if owned or len(neighbors)!=1:
        return None,[]
    index=neighbors[0]['index'];entry=entries[index]
    x,y,right,bottom=entry['bbox'];left,top,crop_right,crop_bottom=bbox
    if bottom-y>(crop_bottom-top)*1.8 or not x<=left<crop_right<=right:
        return None,[]
    source=mask[y:bottom,x:right]
    yy,xx=np.where(source)
    if not len(xx):
        return None,[]
    gx=xx+x;gy=yy+y
    within_x=(gx>=left)&(gx<crop_right)
    if np.any(within_x & ((gy<top)|(gy>=crop_bottom))):
        return None,[]
    inside=within_x & (gy>=top)&(gy<crop_bottom)
    if not np.any(inside):
        return None,[]
    inside_left=int(gx[inside].min());inside_right=int(gx[inside].max())
    contexts=[]
    for role,selected in (('left',gx<left),('right',gx>=crop_right)):
        if not np.any(selected):
            continue
        sx,sy=gx[selected],gy[selected]
        gap=inside_left-int(sx.max())-1 if role=='left' else int(sx.min())-inside_right-1
        if gap<config['minimum_context_blank_band']:
            return None,[]
        pad=config['padding']
        box=[max(x,int(sx.min())-pad),max(y,int(sy.min())-pad),min(right,int(sx.max())+1+pad),min(bottom,int(sy.max())+1+pad)]
        if role=='left':box[2]=min(box[2],left-1)
        else:box[0]=max(box[0],crop_right+1)
        if box[0]>int(sx.min()) or box[2]<=int(sx.max()) or area(box)<=0:
            return None,[]
        contexts.append({'role':role,'bbox':box,'blank_band_pixels':gap,'visible_pixels':int(selected.sum())})
    if not contexts:
        return None,[]
    return index,contexts


def _route_eligibility(original, evidence):
    """Gate specialist work after discovery, without expected scene text."""
    if evidence['edge_ink'] or evidence['clipped_canvas']:
        return False,'incomplete-or-edge-clipped-crop'
    if evidence['neighbor_body_overlap'] and not evidence['mixed_region_separable']:
        return False,'neighbor-body-overlap'
    if not evidence['pixel_support'] or not evidence.get('independent_face_evidence', evidence['extra_shape_support']):
        return False,'unconfirmed-independent-pixel-face-shape'
    geometry=evidence.get('pair_geometry')
    if geometry is not None:
        # Measured bracket strokes exist: pixels decide. The baseline's own
        # reading of the ROI (misread arms, digits, letters) is not evidence.
        if geometry['disjoint_pairs']>1:
            return False,'multiple-bracket-pairs'
        if geometry['interior_components']<2:
            return False,'insufficient-interior-ink'
    else:
        # Classifier-only evidence keeps the conservative text guards: digits,
        # letters outside the bracket span and single-letter brackets.
        letters=[char for char in original if unicodedata.category(char)[0] in 'LN']
        outside,digits=ordinary_anchor_evidence(original)
        if outside and not is_face_text(original):
            return False,'observed-ordinary-body-anchors'
        if digits and not is_face_text(original):
            return False,'observed-ordinary-numbers'
        symbols=sum(unicodedata.category(char)[0] in 'PS' for char in original)
        if letters and not is_face_text(original) and not (2<=len(letters)<=12 and symbols>=1):
            return False,'ordinary-body-or-single-letter-brackets'
    if len(original)>64:
        return False,'observed-body-length-limit'
    return True,'independent-pixels-and-bounded-face-evidence'


def _body_beside_pair(container, pair_box, cells, scale):
    """Does ``container`` hold a full-square glyph cell outside the pair box on the pair's line?"""
    for cell in cells:
        box = cell['bbox']
        if box[2] <= container[0] or box[0] >= container[2] or box[3] <= container[1] or box[1] >= container[3]:
            continue
        if box[2] <= pair_box[0] or box[0] >= pair_box[2]:
            if _body_like(cell, scale):
                return True
    return False


def _merge_proposals(proposed, config):
    """Merge overlapping proposals; a bracket-pair envelope decides against containers that hold body glyphs.

    A chunk or first-pass box that contains a pair is kept (union) when its extra ink is only arm-like cells,
    because the pair bridge may stop short of a far arm; it yields to the pair box when it also holds a
    full-square glyph cell, which on the development scenes is body text.
    """
    merged = []
    for proposal in proposed:
        box = proposal['ink_bbox']
        pair = proposal.get('pair')
        same = None
        for existing in merged:
            old = existing['ink_bbox']; overlap = intersection(box, old)
            # Containment merges only a pixel chunk with its pair. A first-pass box that is much larger than the
            # pair stays a separate candidate (as before): the pair may stop short of a far arm the box covers.
            contained = ((pair is not None and 'image-components' in existing['sources'] and overlap >= .9 * area(box))
                         or (existing.get('pair') is not None and 'image-components' in proposal['sources'] and overlap >= .9 * area(old)))
            if overlap / max(1, area(box) + area(old) - overlap) >= config['merge_iou'] or contained:
                same = existing; break
        if same is None:
            merged.append(dict(proposal))
            continue
        same['sources'] = sorted(set(same['sources'] + proposal['sources']))
        old = same['ink_bbox']
        union = [min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]), max(old[3], box[3])]
        pixel_container = 'image-components' in same['sources'] and same.get('pair') is None
        if pair is not None and same.get('pair') is None:
            # A pixel chunk meets its pair: keep the chunk unless it holds body beside the pair. A first-pass
            # box never widens a candidate: pixel envelopes stay authoritative over OCR padding.
            same['pair'] = pair
            same['ink_bbox'] = union if pixel_container and not _body_beside_pair(old, box, pair['cells'], pair['scale']) else list(box)
            if config.get('low_mark_guard') is not None:
                same['ink_bbox'] = _trim_low_marks(same['ink_bbox'], pair, config['low_mark_guard'])
        elif pair is None and same.get('pair') is not None and 'image-components' in proposal['sources']:
            anchor = same['pair']
            same['ink_bbox'] = same['ink_bbox'] if _body_beside_pair(box, anchor['box'], anchor['cells'], anchor['scale']) else union
            if config.get('low_mark_guard') is not None:
                same['ink_bbox'] = _trim_low_marks(same['ink_bbox'], anchor, config['low_mark_guard'])
        elif pair is None and 'first-pass-symbol-geometry' in proposal['sources']:
            pass  # OCR geometry only adds its source; the pixel envelope is unchanged.
        else:
            same['ink_bbox'] = union
    return merged


def locate_candidates(image, baseline, config=None, classifier=None):
    """Discover candidates from pixels and first-pass geometry; an optional classifier adds face evidence.

    The classifier only contributes a recorded probability compared with its
    frozen threshold; bracket-pair shapes remain an independent alternative.
    Structural components (borders, separators) are excluded before line
    construction; bracket-pair envelopes decide against containers holding body.
    """
    config = configuration(config)
    array = rgb_image(image, config['maximum_pixels'])
    height, width = array.shape[:2]
    entries = baseline_regions(baseline, (width, height), config['maximum_baseline_regions'])
    mask, core = _masks(array, config)
    contrast = contrast_map(array, config['background_window']) if config.get('small_text_shape') is not None else None
    components = _components(mask, config['minimum_component_area'])
    if len(components)>config['maximum_components']:
        raise ValueError('Original image exceeds component complexity ceiling')
    components, structural = _structural_split(components, config)
    proposed = []
    rejections = {}
    shape_detection = {'threshold': config['shape_contrast_threshold'], 'base_pairs': 0, 'core_pairs': 0, 'merged_bracket_candidates': 0}
    def reject(reason):
        rejections[reason] = rejections.get(reason, 0) + 1
    for group in _line_groups(components, config):
        box = group['bbox']
        h, w = box[3]-box[1], box[2]-box[0]
        if not config['minimum_height'] <= h <= config['maximum_height'] or w > config['maximum_width'] or len(group['components']) < 3:
            reject('pixel-size-or-component-limit')
            continue
        proposed.append({'ink_bbox': box, 'sources': ['image-components']})
    for group in _bracket_groups(core,components,config,guard_body_cells=core is not mask,contrast=contrast):
        proposed.append({'ink_bbox':group['bbox'],'sources':['image-bracket-pair'],
                         'pair':{'box':list(group['bbox']),'cells':group['line_cells'],'scale':group['scale'],'strokes':group['strokes']}})
    for entry in entries:
        text = entry['text']
        symbols = sum(unicodedata.category(c)[0] in 'PS' for c in text)
        if not is_face_text(text) and not (3 <= len(text) <= 64 and symbols >= 2):
            continue
        box = entry['bbox']; h = box[3]-box[1]
        if not config['minimum_height'] <= h <= config['maximum_height'] or box[2]-box[0] > config['maximum_width']:
            reject('ocr-size-limit'); continue
        proposed.append({'ink_bbox': list(box), 'sources': ['first-pass-symbol-geometry']})
    merged = _merge_proposals(proposed, config)
    candidates = []; crop_pixels = 0
    for proposal in sorted(merged, key=lambda p: (p['ink_bbox'][1], p['ink_bbox'][0])):
        left, top, right, bottom = proposal['ink_bbox']; padding = config['padding']
        box = [max(0,left-padding),max(0,top-padding),min(width,right+padding),min(height,bottom+padding)]
        owned = []; neighbors = []; body_overlap = False; baseline_pixels = []
        for entry in entries:
            overlap = intersection(box, entry['bbox'])
            if not overlap:
                continue
            coverage = overlap/area(entry['bbox'])
            ox,oy,oright,obottom=entry['bbox']
            entry_pixels=int(mask[oy:obottom,ox:oright].sum())
            captured=int(mask[max(box[1],oy):min(box[3],obottom),max(box[0],ox):min(box[2],oright)].sum())
            ink_coverage=captured/entry_pixels if entry_pixels else 0.
            baseline_pixels.append({'index':entry['index'],'visible_pixels':entry_pixels,'captured_pixels':captured,'ink_coverage':ink_coverage,'box_coverage':coverage})
            if coverage >= config['baseline_coverage'] or (entry_pixels>=6 and ink_coverage>=config['baseline_ink_coverage']):
                owned.append(entry)
            else:
                neighbors.append({'index':entry['index'],'bbox':entry['bbox'],'overlap_pixels':overlap,'coverage':coverage})
                body_overlap = True
        # Guard other physical OCR fields before any ROI expansion can consume them.
        for entry in entries:
            if entry in owned:
                continue
            other = entry['bbox']
            if intersection(proposal['ink_bbox'], other):
                continue
            if min(bottom,other[3])-max(top,other[1]) >= min(bottom-top,other[3]-other[1])*.5:
                if other[2] <= left: box[0] = max(box[0],other[2]+1)
                elif other[0] >= right: box[2] = min(box[2],other[0]-1)
            if min(right,other[2])-max(left,other[0]) > 0:
                if other[3] <= top: box[1] = max(box[1],other[3]+1)
                elif other[1] >= bottom: box[3] = min(box[3],other[1]-1)
        if box[0] >= box[2] or box[1] >= box[3] or area(box) > config['maximum_crop_pixels']:
            reject('invalid-or-oversized-crop'); continue
        evidence, pair_counts = _crop_evidence(array, box, config, mask, core, contrast)
        if classifier is not None:
            evidence['classifier'] = classifier.evidence(np.ascontiguousarray(array[box[1]:box[3], box[0]:box[2]]))
        evidence['independent_face_evidence'] = bool(evidence['extra_shape_support'] or evidence.get('classifier', {}).get('face_evidence', False))
        evidence.update({'neighbor_body_overlap':body_overlap,'neighbor_regions':neighbors,
                         'baseline_region_count':len(owned),'baseline_pixel_coverage':baseline_pixels,'clipped_canvas':left==0 or top==0 or right==width or bottom==height,
                         'ink_bbox':proposal['ink_bbox']})
        splice_index,context_crops=_mixed_contexts(mask,box,owned,neighbors,entries,config)
        evidence['mixed_region_separable']=splice_index is not None
        evidence['mixed_baseline_index']=splice_index
        if not evidence['pixel_support']:
            reject('insufficient-pixel-support'); continue
        if len(candidates) >= config['maximum_candidates'] or crop_pixels + area(box) > config['maximum_crop_pixels']:
            reject('total-crop-budget'); continue
        crop_pixels += area(box)
        shape_detection['base_pairs'] += pair_counts['base_pairs']
        shape_detection['core_pairs'] += pair_counts['core_pairs']
        shape_detection['merged_bracket_candidates'] += int(pair_counts['core_pairs'] > 0 and pair_counts['base_pairs'] == 0)
        ordered = sorted(owned, key=lambda e:(e['bbox'][1],e['bbox'][0]))
        original=''.join(e['text'] for e in ordered)
        eligible,route_reason=_route_eligibility(original,evidence)
        candidates.append({'candidate_id': f'candidate-{len(candidates):03d}', 'bbox':box,
                           'sources':proposal['sources'], 'baseline_indices':[e['index'] for e in ordered],
                           'baseline_text':original, 'baseline_scores':[e['score'] for e in ordered],
                           'route_eligible':eligible,'route_reason':route_reason,'splice_baseline_index':splice_index,
                           'baseline_parent_sha256':baseline_parent_digest(entries[splice_index]) if splice_index is not None else None,
                           'context_crops':context_crops,'evidence':evidence})
    return {'schema_version':SCHEMA,'image_size':[width,height],'image_sha256':image_digest(array),
            'config':config,'candidates':candidates,'rejections':rejections,'crop_pixels':crop_pixels,
            'structural_components':len(structural),'shape_detection':shape_detection,
            'discovered_count':len(candidates),'route_eligible_count':sum(item['route_eligible'] for item in candidates)}
