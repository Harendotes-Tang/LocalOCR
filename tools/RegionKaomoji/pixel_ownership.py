"""Opt-in exclusion of accepted parentless readings of the same original ink.

The caller must first regenerate candidates and apply every existing fusion
guard. This pass can only suppress an accepted insertion; it cannot create a
candidate, vote, body exception or replacement. Component identities and bracket
ownership are recomputed from the original RGB canvas, never from OCR strings.
"""
from __future__ import annotations

import copy
import hashlib
from typing import TypeAlias, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image as PilImage

from tools.RegionKaomoji import candidates as c

SCHEMA = 'localocr.pixel-ownership.v1'
DEFAULT_OPTIONS = {'strategy': 'component-exclusion-v1'}
LEGACY_SOURCES = frozenset(('image-bracket-pair', 'image-components', 'first-pass-symbol-geometry'))

JsonValue: TypeAlias = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']
JsonObject: TypeAlias = dict[str, JsonValue]
PixelOwnershipOptions: TypeAlias = dict[str, str]


class AcceptedInsertion(TypedDict):
    """Audited candidate fields consumed after the caller's acceptance guards."""

    candidate_id: str
    bbox: list[int]
    sources: list[str]
    baseline_indices: list[int]
    splice_baseline_index: None
    evidence: JsonObject
    specialist_text: str


class OwnershipResult(TypedDict):
    """Serializable suppression decisions and their recomputed pixel receipts."""

    schema_version: str
    strategy: str
    suppressed: dict[str, JsonObject]
    conflicts: list[JsonObject]
    candidate_evidence: dict[str, JsonObject]


def configuration(options: PixelOwnershipOptions | None = None) -> PixelOwnershipOptions:
    """Validate the complete optional strategy; absence is handled by fusion."""
    value = copy.deepcopy(DEFAULT_OPTIONS if options is None else options)
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_OPTIONS)
            or value['strategy'] != DEFAULT_OPTIONS['strategy']):
        raise ValueError('Unknown or incomplete pixel-ownership options')
    return value


def _box(value, width, height):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(part) is not int for part in value)
            or not 0 <= value[0] < value[2] <= width
            or not 0 <= value[1] < value[3] <= height):
        raise ValueError('Pixel-ownership box outside original canvas')
    return list(value)


def _contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _shared(mask, first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return int(mask[top:bottom, left:right].sum()) if left < right and top < bottom else 0


def _components(mask, box, maximum):
    """Bound labels to one crop; identities include absolute pixels and extent.

    A cut component is still reported locally but never establishes complete
    dominance. Returning None at the component cap permits only conservative
    original-result fallback, without allocating a full-canvas label image.
    """
    left, top, right, bottom = box
    local = np.ascontiguousarray(mask[top:bottom, left:right], dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    if count - 1 > maximum:
        return None
    result = []
    for index, (x, y, width, height, pixels) in enumerate(stats[1:count], 1):
        bbox = [int(x)+left, int(y)+top, int(x+width)+left, int(y+height)+top]
        bits = np.ascontiguousarray(labels[y:y+height, x:x+width] == index, dtype=np.uint8)
        digest = hashlib.sha256(bits.tobytes()).hexdigest()
        result.append({'bbox': bbox, 'pixels': int(pixels), 'mask_sha256': digest,
                       'key': (*bbox, int(pixels), digest)})
    return result


def _descriptor(array, mask, core, contrast, item, config, image_sha256):
    height, width = mask.shape
    box = _box(item['bbox'], width, height)
    evidence = item['evidence']
    ink = _box(evidence['ink_bbox'], width, height)
    if not _contains(box, ink):
        raise ValueError('Pixel-ownership ink box outside its crop')
    observed, _ = c._crop_evidence(array, box, config, mask, core, contrast)
    # Recompute actual geometry even for standalone mechanism callers. In
    # fusion the entire candidate was already checked against c.locate_candidates.
    for key in ('image_sha256', 'pixel_count', 'paired_bracket_shapes', 'pair_geometry'):
        if evidence.get(key) != observed[key]:
            raise ValueError('Pixel-ownership evidence differs from original pixels')
    base_parts = _components(mask, ink, config['maximum_components'])
    core_parts = _components(core, ink, config['maximum_components'])
    complete = (base_parts is not None and core_parts is not None
                and not c._crosses_component_boundary(mask, ink)
                and not c._crosses_component_boundary(core, ink)
                and int(mask[box[1]:box[3], box[0]:box[2]].sum())
                == int(mask[ink[1]:ink[3], ink[0]:ink[2]].sum()))
    geometry = observed['pair_geometry']
    pairs = {(tuple(pair['left_bbox']), tuple(pair['right_bbox']))
             for pair in observed['paired_bracket_shapes']}
    # Shape detection uses the core mask, while an antialiased exterior arm
    # may exist only in the lower-contrast base mask. Keep that whole component
    # as evidence rather than inventing or erasing it to match the reading.
    exterior = ({side: any(part['bbox'][2] <= geometry['left_bbox'][0] if side == 'left'
                          else part['bbox'][0] >= geometry['right_bbox'][2]
                          for part in base_parts or []) for side in ('left', 'right')}
                if geometry else None)
    sides = c.exterior_atoms(item['specialist_text'])
    safe = bool(complete and geometry and len(pairs) == 1
                and geometry['disjoint_pairs'] == 1 and geometry['interior_components'] >= 2
                and sides == exterior and not observed['edge_ink']
                and not evidence.get('clipped_canvas', True)
                and not evidence.get('neighbor_body_overlap', True))
    return {'identity': item['candidate_id'], 'ink_bbox': ink, 'geometry': geometry,
            'legacy': bool(LEGACY_SOURCES.intersection(item.get('sources', ()))),
            'base_parts': base_parts or [], 'core_parts': core_parts or [],
            'base_keys': {part['key'] for part in base_parts or []},
            'core_keys': {part['key'] for part in core_parts or []},
            'complete_components': bool(complete), 'safe_complete_reading': safe,
            'base_exterior_ink': exterior,
            'original_image_sha256': image_sha256}


def _same_anchors(core, first, second):
    a, b = first['geometry'], second['geometry']
    return bool(a and b and _shared(core, a['left_bbox'], b['left_bbox'])
                and _shared(core, a['right_bbox'], b['right_bbox']))


def _dominates(full, short, same_anchors, config):
    """A strict component superset may win only through verified exterior ink.

    Dense/tall cells and long jumps are existing geometric body barriers. A
    larger rectangle, extra interior components, another bracket pair, a cut
    stroke or unaccounted OCR arm never grants complete ownership.
    """
    if (not same_anchors or not full['safe_complete_reading'] or not short['safe_complete_reading']
            or not short['core_keys'] <= full['core_keys']
            or not short['base_keys'] < full['base_keys']):
        return False
    geometry = full['geometry']
    left, right = geometry['left_bbox'], geometry['right_bbox']
    scale = max(left[3]-left[1], right[3]-right[1])
    extras = [part for part in full['base_parts'] if part['key'] not in short['base_keys']]
    if any(not (part['bbox'][2] <= left[0] or part['bbox'][0] >= right[2]) for part in extras):
        return False
    cells, _ = c._glyph_cells(extras)
    limit = config.get('pair_recrop', {}).get('maximum_exterior_cells', 2)
    if len(cells) > limit or any(c._body_like(cell, scale)
                               or cell['bbox'][3]-cell['bbox'][1] > scale*1.25 for cell in cells):
        return False
    gap = config['horizontal_gap_height']*scale
    for side in ('left', 'right'):
        selected = [cell for cell in cells if (cell['bbox'][2] <= left[0] if side == 'left'
                                               else cell['bbox'][0] >= right[2])]
        selected.sort(key=lambda cell: cell['bbox'][0], reverse=side == 'left')
        reach = left[0] if side == 'left' else right[2]
        for cell in selected:
            box = cell['bbox']
            if (reach-box[2] if side == 'left' else box[0]-reach) > gap:
                return False
            reach = box[0] if side == 'left' else box[2]
    return True


def resolve_conflicts(image: NDArray[np.uint8] | PilImage, accepted: list[AcceptedInsertion],
                      candidate_config: JsonObject | None = None,
                      config: PixelOwnershipOptions | None = None) -> OwnershipResult:
    """Return suppression proofs for already accepted, audited parentless crops.

    ``accepted`` holds automatic candidate dictionaries with an additional raw
    ``specialist_text``. No input is changed. Safe complete arms can supersede a
    short crop; unresolved actual-ink conflicts preserve legacy-source results
    before novel proposals. Two ambiguous legacy results are retained together,
    because this pass cannot reinterpret the original result. Disjoint owned
    ink remains independent even when padded rectangles or strings coincide.
    """
    options = configuration(config)
    settings = c.configuration(candidate_config)
    array = c.rgb_image(image, settings['maximum_pixels'])
    if not isinstance(accepted, list) or len(accepted) > settings['maximum_candidates']:
        raise ValueError('Pixel-ownership candidates must be bounded')
    identities = set()
    pixels = 0
    height, width = array.shape[:2]
    for item in accepted:
        if (not isinstance(item, dict) or not isinstance(item.get('candidate_id'), str)
                or not item['candidate_id'] or item['candidate_id'] in identities):
            raise ValueError('Invalid or duplicate pixel-ownership identity')
        identities.add(item['candidate_id'])
        if item.get('baseline_indices') != [] or item.get('splice_baseline_index') is not None:
            raise ValueError('Pixel-ownership applies only to parentless insertions')
        if not isinstance(item.get('specialist_text'), str):
            raise ValueError('Pixel-ownership requires the accepted raw reading')
        pixels += c.area(_box(item['bbox'], width, height))
    if pixels > settings['maximum_crop_pixels']:
        raise ValueError('Pixel-ownership crop pixels must be bounded')
    result: OwnershipResult = {'schema_version': SCHEMA, 'strategy': options['strategy'],
                               'suppressed': {}, 'conflicts': [], 'candidate_evidence': {}}
    if len(accepted) < 2:
        return result
    mask, core = c._masks(array, settings)
    contrast = c.contrast_map(array, settings['background_window']) if settings.get('small_text_shape') else None
    image_sha256 = c.image_digest(array)
    descriptors = [_descriptor(array, mask, core, contrast, item, settings, image_sha256) for item in accepted]
    for item in descriptors:
        result['candidate_evidence'][item['identity']] = {
            'ink_bbox': item['ink_bbox'], 'complete_components': item['complete_components'],
            'safe_complete_reading': item['safe_complete_reading'],
            'base_exterior_ink': item['base_exterior_ink'],
            'base_component_count': len(item['base_parts']), 'core_component_count': len(item['core_parts']),
            'legacy_source': item['legacy'], 'original_image_sha256': image_sha256}
    kept = []
    # Stable partition by audited source, never candidate-name spelling or text.
    for item in sorted(descriptors, key=lambda record: not record['legacy']):
        conflicts = []
        for previous in kept:
            shared = _shared(core, item['ink_bbox'], previous['ink_bbox'])
            if shared:
                same = _same_anchors(core, item, previous)
                shared_components = item['core_keys'] & previous['core_keys']
                if not same and not shared_components:
                    continue
                record = {'candidate_id': item['identity'], 'other_candidate_id': previous['identity'],
                          'shared_core_pixels': shared, 'same_bracket_anchors': same,
                          'shared_core_components': [{'bbox': list(key[:4]), 'pixels': key[4],
                                                      'mask_sha256': key[5]} for key in sorted(shared_components)],
                          'candidate_ink_bbox': item['ink_bbox'], 'other_ink_bbox': previous['ink_bbox'],
                          'candidate_complete_components': item['complete_components'],
                          'other_complete_components': previous['complete_components'],
                          'candidate_reading_supported': item['safe_complete_reading'],
                          'other_reading_supported': previous['safe_complete_reading'],
                          'original_image_sha256': image_sha256}
                result['conflicts'].append(record)
                conflicts.append((previous, same, record))
        if not conflicts:
            kept.append(item)
            continue
        if all(_dominates(item, previous, same, settings) for previous, same, _ in conflicts):
            for previous, _, record in conflicts:
                result['suppressed'][previous['identity']] = {
                    **record, 'suppressed_candidate_id': previous['identity'],
                    'winner': item['identity'], 'reason': 'pixel-owned-complete-expression'}
                kept.remove(previous)
            kept.append(item)
            continue
        dominant = next(((previous, record) for previous, same, record in conflicts
                         if _dominates(previous, item, same, settings)), None)
        if dominant is not None:
            previous, record = dominant
            result['suppressed'][item['identity']] = {
                **record, 'suppressed_candidate_id': item['identity'],
                'winner': previous['identity'], 'reason': 'pixel-owned-complete-expression'}
        elif item['legacy'] and all(previous['legacy'] for previous, _, _ in conflicts):
            kept.append(item)
        else:
            previous, _, record = conflicts[0]
            result['suppressed'][item['identity']] = {
                **record, 'suppressed_candidate_id': item['identity'],
                'winner': previous['identity'], 'reason': 'pixel-owned-ambiguous-preserve-original'}
    # A later proved complete crop can supersede an earlier winner. Keep each
    # suppression receipt tied to the final retained owner, with its lineage.
    for proof in result['suppressed'].values():
        chain = [proof['winner']]
        while chain[-1] in result['suppressed']:
            chain.append(result['suppressed'][chain[-1]]['winner'])
            if len(chain) > len(descriptors):
                raise RuntimeError('Cyclic pixel-ownership winner chain')
        if len(chain) > 1:
            proof['ownership_chain'] = chain
            proof['winner'] = chain[-1]
    return result
