"""Optional final-owner exclusion and atomic production-parent materialization.

Fusion must first regenerate candidates and apply all its existing guards. This
module never accepts a reading: it can remove a duplicate only after actual
final materialization and a strict original-component/exterior-arm proof. A
body parent is a text transaction, never a rectangular ownership claim.
"""
from __future__ import annotations

import copy

from tools.RegionKaomoji import candidates as c, pixel_ownership as p

SCHEMA = 'localocr.owned-fusion.v1'
DEFAULT_OPTIONS = {'strategy': 'atomic-face-ownership-v1'}


def configuration(options=None):
    """Validate the complete optional research contract without injecting it."""
    value = copy.deepcopy(DEFAULT_OPTIONS if options is None else options)
    if not isinstance(value, dict) or value != DEFAULT_OPTIONS:
        raise ValueError('Unknown or incomplete final-owner options')
    return value


def _child_parents(region, owner):
    indices = set(region.get('baseline_indices', []))
    if region.get('baseline_index') is not None:
        indices.add(region['baseline_index'])
    return {owner[index] for index in indices if index in owner}


def _restore(parent_index, plan, reason):
    entry = plan['parents'][str(parent_index)]['entry']
    return {'bbox': list(entry['bbox']), 'text': entry['text'], 'score': entry['score'],
            'source': 'baseline', 'baseline_index': parent_index, 'split_restored': True,
            'split_unverifiable': True, 'atomic_restored': True, 'atomic_reason': reason}


def materialize_atomic(regions, plan, config, materialize_fn):
    """Materialize each split parent once or restore its exact original bytes.

    ``materialize_fn`` remains the existing guarded splitter. Multiple accepted
    child edits are deliberately not combined: all children of their parent
    fall back together. A postcheck also retracts any lifted version whose
    parent was restored by that splitter. Inputs are never mutated.
    """
    owner = {int(key): value for key, value in plan['children'].items()}
    if not plan['parents']:
        return materialize_fn(copy.deepcopy(regions), plan, config)
    edits, invalid, conflicts = {}, set(), []
    for region in regions:
        parents = _child_parents(region, owner)
        if region.get('source', 'baseline') != 'baseline':
            for parent in parents:
                edits.setdefault(parent, []).append(region)
            if len(parents) > 1:
                invalid.update(parents)
    for parent, items in sorted(edits.items()):
        if len(items) > 1 or parent in invalid:
            invalid.add(parent)
            conflicts.append({'parent_index': parent, 'reason': 'multiple-parent-face-edits',
                              'candidate_ids': [item.get('candidate_id') for item in items]})
    safe = [copy.deepcopy(region) for region in regions
            if not _child_parents(region, owner).intersection(invalid)]
    output, stats = materialize_fn(safe, plan, config)
    output, stats = copy.deepcopy(output), copy.deepcopy(stats)
    def final_parents(region):
        if region.get('split_parent_index') is not None:
            return {region['split_parent_index']}
        index = region.get('baseline_index')
        return {index} if str(index) in plan['parents'] and region.get('split_restored') else set()
    # A late restoration invalidates every tentative version of that parent,
    # including an already lifted full-parent splice from an earlier child.
    by_parent = {}
    for region in output:
        for parent in final_parents(region):
            by_parent.setdefault(parent, []).append(region)
    for parent, items in by_parent.items():
        restored = [item for item in items if item.get('split_restored')]
        lifted = [item for item in items if item.get('lifted_from_child') is not None]
        if (restored and len(items) > 1) or len(lifted) > 1:
            invalid.add(parent)
            conflicts.append({'parent_index': parent, 'reason': 'parent-materialization-withdrawn',
                              'candidate_ids': [item.get('candidate_id') for item in items
                                                if item.get('candidate_id') is not None]})
    output = [region for region in output if not final_parents(region).intersection(invalid)]
    output.extend(_restore(parent, plan, 'atomic-parent-conflict') for parent in sorted(invalid))
    output.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    stats.update(restored=len({region['baseline_index'] for region in output if region.get('split_restored')}),
                 restored_unverifiable=len({region['baseline_index'] for region in output
                                            if region.get('split_unverifiable')}),
                 parent_level_splices=sum(region.get('lifted_from_child') is not None for region in output),
                 atomic_restored=len(invalid), atomic_conflicts=conflicts)
    return output, stats


def _body_proof(region, entries):
    """Return the verified production-parent edit, or None without guessing."""
    proof = region.get('splice')
    index = region.get('baseline_index')
    if not isinstance(proof, dict) or type(index) is not int or not 0 <= index < len(entries):
        return None
    parent = entries[index]
    span = proof.get('original_text_span')
    face = region.get('regional_text')
    if (not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)
            or not 0 <= span[0] <= span[1] <= len(parent['text']) or not isinstance(face, str)
            or not c.is_face_text(face)):
        return None
    start, end = span
    if (proof.get('baseline_index') != index
            or proof.get('baseline_parent_sha256') != c.baseline_parent_digest(parent)
            or proof.get('ordinary_prefix_suffix_preserved') is not True
            or proof.get('preserved_prefix') != parent['text'][:start]
            or proof.get('preserved_suffix') != parent['text'][end:]
            or proof.get('original_face_text') != parent['text'][start:end]
            or proof.get('replacement_text') != face or region['bbox'] != parent['bbox']
            or region['text'] != parent['text'][:start]+face+parent['text'][end:]):
        return None
    return {'parent_index': index, 'parent': parent, 'start': start, 'end': end,
            'prefix': parent['text'][:start], 'suffix': parent['text'][end:]}


def _claim(region, candidate, decision, entries):
    """Bound final face bytes and geometry without using an entire body box."""
    if region.get('source') == 'baseline-with-regional-splice':
        body = _body_proof(region, entries)
        if body is None:
            return None, 'unverifiable-production-parent-bytes'
        ink = region.get('regional_ink_bbox')
        if ink != candidate['evidence'].get('ink_bbox'):
            return None, 'regional-face-ink-not-candidate-audited-ink'
        text, operation = region['regional_text'], 'regional-splice'
    elif region.get('source') == 'specialist-exact-views':
        # Split materialization intentionally retains child indices on plain
        # replacement regions. Even an in-range index can name an unrelated
        # production face, so no production/body ownership can be inferred
        # from it. A lifted regional splice follows the complete parent proof
        # branch above instead; it is the only supported split transaction.
        if region.get('split_parent_index') is not None or region.get('lifted_from_child') is not None:
            return None, 'unmapped-split-child-production-ownership'
        body, text = None, region.get('text')
        if region.get('ink_bbox') != candidate['evidence'].get('ink_bbox') or region['bbox'] != candidate['bbox']:
            return None, 'final-face-geometry-differs'
        indices = region.get('baseline_indices', [])
        if (not isinstance(indices, list) or any(type(index) is not int or not 0 <= index < len(entries)
                                               or not c.is_face_text(entries[index]['text']) for index in indices)):
            return None, 'owned-parent-has-unproved-body'
        operation = 'baseline-replacement' if indices else 'parentless-insertion'
    else:
        return None, 'unsupported-final-operation'
    if text != decision.get('specialist_text') or not c.is_face_text(text):
        return None, 'final-face-reading-differs'
    return {'candidate': candidate, 'region': region, 'body': body, 'text': text,
            'operation': operation}, None


def _transfer(winner, loser, regions):
    """Remove a duplicate without reintroducing baseline bytes or losing body.

    A parentless winner can take over a face-only baseline replacement. A
    losing splice with preserved body cannot be enlarged safely: its old
    prefix/suffix may already read the winner's extra arm pixels. Text spans
    and a parent rectangle do not prove that those pixels are free. Keep that
    entire group until an independently audited pixel partition can do so.
    """
    win, lose = winner['region'], loser['region']
    if loser['body'] is not None:
        return False
    owned = lose.get('baseline_indices', [])
    if owned:
        if winner['operation'] != 'parentless-insertion':
            return False
        value = copy.deepcopy(win)
        value['baseline_indices'] = list(owned)
        value['ownership_transferred_from'] = lose['candidate_id']
        regions[regions.index(win)] = value
        winner.update(region=value, operation='baseline-replacement')
    regions.remove(lose)
    return True


def commit_ownership(image, baseline, candidates, fused, final_regions, candidate_config=None, options=None):
    """Commit strict final-owner suppression; retain provisional fusion fields.

    ``baseline`` is the production first pass, whereas candidate indices and
    materialized split-child indices may still refer to children. Only unsplit
    region indices or a complete regional parent proof address ``baseline``.
    The caller supplies the very candidates previously re-audited by fusion;
    component/anchor claims are nevertheless recomputed from original pixels.
    ``final_regions`` is the atomic, strategy-off ownership result. Unsupported
    body/ink mappings remain exactly as supplied, with an explicit receipt.
    """
    options, settings = configuration(options), c.configuration(candidate_config)
    array = c.rgb_image(image, settings['maximum_pixels'])
    height, width = array.shape[:2]
    entries = c.baseline_regions(baseline, (width, height), settings['maximum_baseline_regions'])
    if not isinstance(candidates, list) or len(candidates) > settings['maximum_candidates']:
        raise ValueError('Final-owner candidates must be bounded')
    by_id, pixels = {}, 0
    for candidate in candidates:
        identity = candidate.get('candidate_id') if isinstance(candidate, dict) else None
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError('Invalid or duplicate final-owner candidate identity')
        by_id[identity] = candidate
        pixels += c.area(p._box(candidate['bbox'], width, height))
    if pixels > settings['maximum_crop_pixels']:
        raise ValueError('Final-owner crop pixels must be bounded')
    if (not isinstance(final_regions, list)
            or len(final_regions) > settings['maximum_baseline_regions']+settings['maximum_candidates']):
        raise ValueError('Final-owner output must be bounded')
    decisions = fused.get('decisions')
    if not isinstance(decisions, list) or len(decisions) > settings['maximum_candidates']:
        raise ValueError('Final-owner decisions must be bounded')
    accepted = {item['candidate_id']: item for item in decisions if item.get('accepted') is True}
    regions = copy.deepcopy(final_regions)
    result = {'schema_version': SCHEMA, 'strategy': options['strategy'], 'suppressed': {},
              'conflicts': [], 'candidate_evidence': {}, 'fallbacks': []}
    mask, core = c._masks(array, settings)
    contrast = c.contrast_map(array, settings['background_window']) if settings.get('small_text_shape') else None
    digest = c.image_digest(array)
    claims, present = [], set()
    for region in regions:
        identity = region.get('candidate_id')
        if identity not in accepted or identity not in by_id:
            continue
        if identity in present:
            raise ValueError('Duplicate final-owner materialized candidate identity')
        present.add(identity)
        claim, reason = _claim(region, by_id[identity], accepted[identity], entries)
        if claim is None:
            # Its actual candidate ink still participates in group ambiguity:
            # silently ignoring an unproved body owner could permit a partial
            # commit elsewhere in the very same conflict group.
            claim = {'candidate': by_id[identity], 'region': region, 'body': None,
                     'text': accepted[identity].get('specialist_text'), 'operation': 'unsupported'}
            result['fallbacks'].append({'candidate_id': identity, 'reason': reason})
        claim['fallback_reason'] = reason
        item = dict(by_id[identity], specialist_text=claim['text'])
        descriptor = p._descriptor(array, mask, core, contrast, item, settings, digest)
        claim['descriptor'] = descriptor
        result['candidate_evidence'][identity] = {
            'operation': claim['operation'], 'ink_bbox': descriptor['ink_bbox'],
            'complete_components': descriptor['complete_components'],
            'safe_complete_reading': descriptor['safe_complete_reading'],
            'base_exterior_ink': descriptor['base_exterior_ink'],
            'production_parent_index': claim['body']['parent_index'] if claim['body'] else None,
            'production_parent_indices': ([] if reason is not None else
                                          [claim['body']['parent_index']] if claim['body'] else
                                          list(region.get('baseline_indices', []))),
            'materialized_split_parent_index': region.get('split_parent_index'),
            'candidate_baseline_indices': list(by_id[identity].get('baseline_indices', [])),
            'candidate_splice_baseline_index': by_id[identity].get('splice_baseline_index'),
            'verified_parent_span': [claim['body']['start'], claim['body']['end']] if claim['body'] else None,
            'verified_parent_sha256': (c.baseline_parent_digest(claim['body']['parent'])
                                       if claim['body'] else None),
            'original_image_sha256': digest}
        claims.append(claim)
    suppressed, links = set(), {index: set() for index in range(len(claims))}
    for index, first in enumerate(claims):
        for other in range(index+1, len(claims)):
            second = claims[other]
            a, b = first['descriptor'], second['descriptor']
            shared = p._shared(core, a['ink_bbox'], b['ink_bbox'])
            same = p._same_anchors(core, a, b)
            if shared and (same or a['core_keys'] & b['core_keys']):
                links[index].add(other)
                links[other].add(index)
                result['conflicts'].append({'candidate_id': first['candidate']['candidate_id'],
                                            'other_candidate_id': second['candidate']['candidate_id'],
                                            'shared_core_pixels': shared, 'same_bracket_anchors': same})
    visited = set()
    for start in links:
        if start in visited:
            continue
        indices, pending = set(), [start]
        while pending:
            index = pending.pop()
            if index not in indices:
                indices.add(index)
                pending.extend(links[index]-indices)
        visited.update(indices)
        group = [claims[index] for index in sorted(indices)]
        if len(group) < 2:
            continue
        identities = [item['candidate']['candidate_id'] for item in group]
        if any(item['fallback_reason'] for item in group):
            result['fallbacks'].append({'candidate_ids': identities, 'reason': 'ambiguous-group-body-or-face'})
            continue
        # One strictly complete owner must dominate the entire connected group.
        # Equal ink with different readings, nesting, or several independent
        # owned parents is ambiguous; no partial suppression is committed.
        possible = [item for item in group if all(
            other is item or p._dominates(item['descriptor'], other['descriptor'],
                                         p._same_anchors(core, item['descriptor'], other['descriptor']), settings)
            for other in group)]
        if len(possible) != 1:
            result['fallbacks'].append({'candidate_ids': identities, 'reason': 'ambiguous-component-ownership'})
            continue
        winner_id = possible[0]['candidate']['candidate_id']
        trial_regions, trial_group = copy.deepcopy(regions), copy.deepcopy(group)
        for item in trial_group:
            item['region'] = next(region for region in trial_regions
                                  if region.get('candidate_id') == item['candidate']['candidate_id'])
        winner = next(item for item in trial_group if item['candidate']['candidate_id'] == winner_id)
        losers = [item for item in trial_group if item is not winner]
        if not all(_transfer(winner, loser, trial_regions) for loser in losers):
            result['fallbacks'].append({'candidate_ids': identities, 'winner': winner_id,
                                        'reason': ('body-splice-transfer-pixel-context-unproved'
                                                   if any(item['body'] is not None for item in losers)
                                                   else 'ambiguous-multiple-owned-parents')})
            continue
        regions = trial_regions
        for loser in losers:
            lose_id = loser['candidate']['candidate_id']
            full, short = winner['descriptor'], loser['descriptor']
            suppressed.add(lose_id)
            result['suppressed'][lose_id] = {'suppressed_candidate_id': lose_id, 'winner': winner_id,
                                            'reason': 'final-pixel-owned-complete-expression',
                                            'shared_core_pixels': p._shared(core, full['ink_bbox'], short['ink_bbox']),
                                            'same_bracket_anchors': p._same_anchors(core, full, short),
                                            'winner_ink_bbox': full['ink_bbox'], 'loser_ink_bbox': short['ink_bbox'],
                                            'winner_final_operation': winner['operation'],
                                            'loser_operation': loser['operation'],
                                            'preserved_parent_index': (winner['body']['parent_index']
                                                                       if winner['body'] else None),
                                            'original_image_sha256': digest}
    # The materialized winner is checked again after all body transfers. This
    # assertion cannot turn provisional acceptance into actual output.
    final_positions = {region['candidate_id']: index for index, region in enumerate(regions)
                       if region.get('candidate_id') in accepted}
    for proof in result['suppressed'].values():
        if proof['winner'] not in final_positions:
            raise RuntimeError('Final-owner suppression lost its materialized winner')
        proof['winner_final_region'] = final_positions[proof['winner']]
    terminal = []
    for decision in decisions:
        identity = decision['candidate_id']
        live = decision.get('accepted') is True and identity in final_positions
        terminal.append({'candidate_id': identity, 'provisional_accepted': decision.get('accepted') is True,
                         'final_accepted': live,
                         'final_reason': ('final-pixel-owned-complete-expression' if identity in suppressed
                                          else 'materialized-final-owner' if live else 'not-materialized'),
                         'final_region_index': final_positions.get(identity),
                         **({'owned_ownership': result['suppressed'][identity]} if identity in suppressed else {})})
    output = copy.deepcopy(fused)
    output.update(final_regions=regions, final_decisions=terminal,
                  final_accepted_count=sum(item['final_accepted'] for item in terminal),
                  owned_ownership=result, owned_ownership_pending=False)
    return output
