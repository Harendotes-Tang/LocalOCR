"""Original-pixel/parent commit fixtures; these tests do not measure OCR quality."""
import copy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools.RegionKaomoji import candidates as c, fusion as f, owned_fusion as o
from tools.RegionKaomoji.evaluate import materialize_splits


def fixture(*, faint=False, nested=False, body=False):
    image = Image.new('RGB', (260, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.arc((40, 20, 54, 50), 90, 270, fill='black', width=2)
    draw.arc((108, 20, 122, 50), 270, 90, fill='black', width=2)
    draw.ellipse((63, 31, 70, 38), outline='black', width=2)
    draw.ellipse((87, 31, 94, 38), outline='black', width=2)
    draw.line((75, 43, 83, 43), fill='black', width=2)
    draw.line((138, 28, 148, 44), fill=(180, 180, 180) if faint else 'black', width=2)
    if nested:
        draw.arc((20, 15, 34, 55), 90, 270, fill='black', width=2)
        draw.arc((126, 15, 140, 55), 270, 90, fill='black', width=2)
    if body:
        draw.rectangle((185, 22, 205, 48), fill='black')
    image = np.asarray(image)
    config = {**c.DEFAULT_CONFIG, **({'shape_contrast_threshold': 80} if faint else {})}
    def item(identity, box, ink, text, owned):
        evidence = c.pixel_evidence(image, box, config)
        evidence.update(ink_bbox=ink, neighbor_body_overlap=False, clipped_canvas=False)
        candidate = {'candidate_id': identity, 'bbox': box, 'baseline_indices': [0] if owned else [],
                     'splice_baseline_index': None, 'sources': ['image-bracket-pair'] if owned else ['image-line-span'],
                     'evidence': evidence}
        region = {'candidate_id': identity, 'bbox': box, 'ink_bbox': ink, 'text': text, 'score': .9,
                  'source': 'specialist-exact-views', 'baseline_indices': [0] if owned else []}
        return candidate, region
    full, full_region = item('full', [37, 17, 153, 54], [40, 20, 150, 51], '(ω_ω)ノ', True)
    short, short_region = item('short', [37, 17, 126, 54], [40, 20, 123, 51], '(ω_ω)', False)
    baseline = [{'bbox': full['bbox'], 'text': '(x_x)ノ', 'score': .5}]
    return image, baseline, [full, short], [full_region, short_region], config


def fused(regions, *, absent=()):
    return {'regions': copy.deepcopy(regions), 'accepted_count': len(regions),
            'decisions': [{'candidate_id': r['candidate_id'], 'accepted': True, 'reason': 'fixture-guard-accepted',
                           'specialist_text': r.get('regional_text', r['text'])} for r in regions] +
                         [{'candidate_id': value, 'accepted': True, 'reason': 'fixture-restored',
                           'specialist_text': '(ω_ω)ノ'} for value in absent]}


def commit(args, regions=None, provisional=None):
    image, baseline, candidates, original, config = args
    regions = original if regions is None else regions
    return o.commit_ownership(image, baseline, candidates, fused(original) if provisional is None else provisional,
                              regions, config, o.configuration())


def test_owned_complete_and_parentless_short_use_actual_final_owner():
    args = fixture()
    before = fused(args[3])
    output = commit(args, provisional=before)
    assert [r['text'] for r in output['final_regions']] == ['(ω_ω)ノ']
    assert output['accepted_count'] == 2 and output['decisions'] == before['decisions']
    assert output['final_accepted_count'] == 1
    proof = output['owned_ownership']['suppressed']['short']
    assert proof['winner'] == 'full' and proof['shared_core_pixels'] > 0
    assert proof['winner_final_region'] == 0
    assert before == fused(args[3])


def test_restored_winner_cannot_suppress_present_short():
    args = fixture()
    output = commit(args, [args[3][1]], fused(args[3]))
    assert output['final_regions'] == [args[3][1]]
    assert output['owned_ownership']['suppressed'] == {}
    assert not next(d for d in output['final_decisions'] if d['candidate_id'] == 'full')['final_accepted']


def test_faint_exterior_arm_survives_core_threshold():
    output = commit(fixture(faint=True))
    assert [r['text'] for r in output['final_regions']] == ['(ω_ω)ノ']
    assert output['owned_ownership']['candidate_evidence']['full']['base_exterior_ink']['right']


def test_ambiguous_nested_or_body_geometry_preserves_strategy_off_output():
    for kind in ('nested', 'body'):
        args = fixture(**{kind: True})
        image, _, candidates, regions, config = args
        # Explicitly include the outer pair or dense body in the alleged face.
        box, ink = (([17, 12, 153, 59], [20, 15, 150, 56]) if kind == 'nested'
                    else ([37, 17, 210, 54], [40, 20, 206, 51]))
        evidence = c.pixel_evidence(image, box, config)
        evidence.update(ink_bbox=ink, neighbor_body_overlap=False, clipped_canvas=False)
        candidates[0].update(bbox=box, evidence=evidence)
        regions[0].update(bbox=box, ink_bbox=ink)
        output = commit(args)
        assert output['final_regions'] == args[3]
        assert output['owned_ownership']['suppressed'] == {}


def test_independent_equal_faces_with_overlapping_padding_are_both_retained():
    image = Image.new('RGB', (260, 80), 'white')
    draw = ImageDraw.Draw(image)
    candidates, regions = [], []
    for identity, x in (('first', 30), ('second', 125)):
        draw.arc((x, 20, x+14, 50), 90, 270, fill='black', width=2)
        draw.arc((x+65, 20, x+79, 50), 270, 90, fill='black', width=2)
        draw.ellipse((x+24, 30, x+30, 37), outline='black', width=2)
        draw.ellipse((x+46, 30, x+52, 37), outline='black', width=2)
        draw.line((x+34, 43, x+42, 43), fill='black', width=2)
    image = np.asarray(image)
    for identity, box, ink in [('first', [27, 17, 125, 54], [30, 20, 110, 51]),
                               ('second', [112, 17, 208, 54], [125, 20, 205, 51])]:
        evidence = c.pixel_evidence(image, box)
        evidence.update(ink_bbox=ink, neighbor_body_overlap=False, clipped_canvas=False)
        candidates.append({'candidate_id': identity, 'bbox': box, 'baseline_indices': [],
                           'splice_baseline_index': None, 'sources': [], 'evidence': evidence})
        regions.append({'candidate_id': identity, 'bbox': box, 'ink_bbox': ink, 'text': '(ω_ω)',
                        'score': .9, 'source': 'specialist-exact-views', 'baseline_indices': []})
    output = o.commit_ownership(image, [], candidates, fused(regions), regions)
    assert c.intersection(regions[0]['bbox'], regions[1]['bbox']) > 0
    assert output['final_regions'] == regions and output['owned_ownership']['conflicts'] == []


def splice_region(region, parent, old, prefix='body ', suffix=' tail'):
    parent.update(text=prefix+old+suffix, bbox=[0, 10, 250, 65])
    proof = {'baseline_index': 0, 'original_text_span': [len(prefix), len(prefix)+len(old)],
             'original_face_text': old, 'replacement_text': region['text'], 'preserved_prefix': prefix,
             'preserved_suffix': suffix, 'ordinary_prefix_suffix_preserved': True,
             'baseline_parent_sha256': c.baseline_parent_digest(dict(parent, index=0))}
    return {**region, 'bbox': parent['bbox'], 'text': prefix+region['text']+suffix,
            'baseline_index': 0, 'source': 'baseline-with-regional-splice',
            'regional_text': region['text'], 'regional_bbox': region['bbox'],
            'regional_ink_bbox': region['ink_bbox'], 'splice': proof}


def test_parent_body_is_not_ownership_and_winner_splice_keeps_raw_bytes():
    args = fixture(body=True)
    args[3][0] = splice_region(args[3][0], args[1][0], '(x_x)ノ', '正文 A  ', '  B尾')
    output = commit(args)
    assert [r['text'] for r in output['final_regions']] == ['正文 A  (ω_ω)ノ  B尾']
    evidence = output['owned_ownership']['candidate_evidence']['full']
    assert evidence['ink_bbox'] == [40, 20, 150, 51]
    assert evidence['ink_bbox'] != args[1][0]['bbox']


def test_body_splice_loser_without_pixel_context_partition_preserves_entire_group():
    args = fixture()
    full, short = args[3]
    full['baseline_indices'] = []
    short['baseline_indices'] = [0]
    args[3][1] = splice_region(short, args[1][0], '(x_x)', '原文  ', '  尾')
    output = commit(args)
    assert output['final_regions'] == args[3]
    assert output['owned_ownership']['suppressed'] == {}
    assert any(item['reason'] == 'body-splice-transfer-pixel-context-unproved'
               for item in output['owned_ownership']['fallbacks'])


def test_preserved_suffix_arm_is_never_duplicated_by_parentless_full_transfer():
    args = fixture()
    args[3][0]['baseline_indices'] = []
    args[3][1] = splice_region(args[3][1], args[1][0], '(x_x)', 'body ', 'ノ tail')
    output = commit(args)
    assert output['final_regions'] == args[3]
    assert output['owned_ownership']['suppressed'] == {}
    assert not any('ノノ' in region['text'] for region in output['final_regions'])


def original_context_splice(args, region_index, *, parent_text, right_text):
    """Run the original pixel partition and context-splice guard on fixture OCR."""
    image, baseline, candidates, regions, config = args
    region, candidate = regions[region_index], candidates[region_index]
    baseline[0].update(bbox=[0, 10, 250, 65], text=parent_text)
    entries = c.baseline_regions(baseline, (image.shape[1], image.shape[0]))
    mask, _ = c._masks(image, config)
    index, declarations = c._mixed_contexts(mask, candidate['bbox'], [], [{'index': 0}], entries, config)
    assert index == 0 and [item['role'] for item in declarations] == ['right']
    candidate.update(splice_baseline_index=0, baseline_parent_sha256=c.baseline_parent_digest(entries[0]),
                     context_crops=declarations)
    records = [{'role': item['role'], 'bbox': item['bbox'],
                'image_sha256': c.image_digest(f.view_image(image, item['bbox'], 'raw')),
                'text': right_text, 'score': .9, 'model_id': 'production-baseline',
                'baseline_parent_sha256': candidate['baseline_parent_sha256']} for item in declarations]
    text, reason, proof = f._context_splice(image, candidate, entries, records, region['text'])
    assert reason is None
    regions[region_index] = {**region, 'bbox': baseline[0]['bbox'], 'text': text,
                            'baseline_index': 0, 'source': 'baseline-with-regional-splice',
                            'regional_text': region['text'], 'regional_bbox': region['bbox'],
                            'regional_ink_bbox': region['ink_bbox'], 'splice': proof}
    return declarations


def test_actual_short_context_guard_accepts_arm_in_suffix_but_final_transfer_refuses():
    args = fixture(body=True)
    args[3][0]['baseline_indices'] = []
    contexts = original_context_splice(args, 1, parent_text='(x_x)ノ tail', right_text='ノ tail')
    # The measured right context genuinely includes the arm pixels that the
    # complete parentless winner would add. The existing raw context proof is
    # valid, but does not authorize transferring ownership of those characters.
    arm = [138, 28, 150, 46]
    assert c.intersection(contexts[0]['bbox'], arm) > 0
    assert args[3][1]['text'] == '(ω_ω)ノ tail'
    output = commit(args)
    assert output['final_regions'] == args[3] and output['owned_ownership']['suppressed'] == {}
    assert not any('ノノ' in region['text'] for region in output['final_regions'])


def test_actual_complete_context_proof_keeps_body_winner_and_suppresses_short_insertion():
    args = fixture(body=True)
    contexts = original_context_splice(args, 0, parent_text='(x_x) tail', right_text='tail')
    assert c.intersection(contexts[0]['bbox'], args[2][0]['evidence']['ink_bbox']) == 0
    before = copy.deepcopy(args[3][0])
    output = commit(args)
    assert output['final_regions'] == [before]
    assert output['final_regions'][0]['text'] == '(ω_ω)ノ tail'
    assert output['owned_ownership']['suppressed']['short']['winner'] == 'full'


def test_unverifiable_body_proof_preserves_entire_ambiguous_group():
    args = fixture()
    args[3][0]['baseline_indices'] = []
    args[3][1] = splice_region(args[3][1], args[1][0], '(x_x)', 'body ', ' tail')
    args[3][1]['splice']['preserved_suffix'] = 'forged'
    output = commit(args)
    assert output['final_regions'] == args[3] and output['owned_ownership']['suppressed'] == {}


def test_ambiguous_third_body_owner_prevents_partial_group_commit():
    args = fixture()
    candidate = copy.deepcopy(args[2][1])
    candidate['candidate_id'] = 'body-short'
    args[2].append(candidate)
    region = copy.deepcopy(args[3][1])
    region['candidate_id'] = 'body-short'
    region = splice_region(region, args[1][0], '(x_x)', 'body ', ' tail')
    region['splice']['preserved_suffix'] = 'unproved'
    args[3].append(region)
    # Let the complete face itself be parentless so both possible transfer
    # paths would otherwise be permitted independently.
    args[3][0]['baseline_indices'] = []
    output = commit(args)
    assert output['final_regions'] == args[3]
    assert output['owned_ownership']['suppressed'] == {}
    assert any(item['reason'] == 'ambiguous-group-body-or-face'
               for item in output['owned_ownership']['fallbacks'])


def test_verified_third_body_owner_rolls_back_other_tentative_suppression():
    args = fixture()
    args[3][0]['baseline_indices'] = []
    candidate = copy.deepcopy(args[2][1])
    candidate['candidate_id'] = 'body-short'
    args[2].append(candidate)
    region = copy.deepcopy(args[3][1])
    region['candidate_id'] = 'body-short'
    args[3].append(splice_region(region, args[1][0], '(x_x)', 'body ', 'ノ tail'))
    # The ordinary short insertion can be tentatively removed first, but the
    # body loser cannot transfer. The entire group, including that insertion,
    # must be restored byte for byte rather than partially committed.
    output = commit(args)
    assert output['final_regions'] == args[3]
    assert output['owned_ownership']['suppressed'] == {}
    assert output['final_accepted_count'] == 3


def test_plain_owned_loser_does_not_reintroduce_its_original_baseline_reading():
    args = fixture()
    args[3][0]['baseline_indices'] = []
    args[3][1]['baseline_indices'] = [0]
    args[1][0]['text'] = '(x_x)'
    output = commit(args)
    assert len(output['final_regions']) == 1
    assert output['final_regions'][0]['text'] == '(ω_ω)ノ'
    assert output['final_regions'][0]['baseline_indices'] == [0]
    assert output['final_regions'][0]['candidate_id'] == 'full'


def test_real_localizer_and_guarded_fusion_reproduce_cross_owned_duplicate():
    from tests.test_pixel_ownership_integration import request, run
    # A first-pass box covers only the exterior arm. This is an intentionally
    # imperfect detector box, with declared fixture OCR; localization and all
    # actual fusion pixel/view guards run rather than manufacturing acceptance.
    args = request(baseline=[{'bbox': [130, 17, 153, 54], 'text': '(x_x)ノ', 'score': .7}])
    provisional = run(args)
    assert [region['text'] for region in provisional['regions']] == ['(ω_ω)ノ', '(ω_ω)']
    assert provisional['regions'][0]['baseline_indices'] == [0]
    assert provisional['regions'][1]['baseline_indices'] == []
    image, baseline, candidates, _, config = args
    output = o.commit_ownership(image, baseline, candidates, provisional, provisional['regions'], config)
    assert [region['text'] for region in output['final_regions']] == ['(ω_ω)ノ']
    assert output['accepted_count'] == 2 and output['final_accepted_count'] == 1
    pending = f.fuse_candidates(image, baseline, candidates, args[3],
                                {**f.DEFAULT_CONFIG, 'owned_ownership': o.configuration()}, config)
    assert pending['owned_ownership_pending'] is True
    assert 'final_regions' not in pending
    committed = o.commit_ownership(image, baseline, candidates, pending, pending['regions'], config)
    assert committed['owned_ownership_pending'] is False


def test_actual_parent_restoration_withdraws_its_owner_before_exclusion():
    args = fixture()
    plan = {'parents': {'0': {'entry': {'bbox': [0, 10, 250, 65], 'text': '(x_x)ノ body', 'score': .5},
                               'children': [0, 1]}},
            'children': {'0': 0, '1': 0}, 'map': {}}
    body = {'bbox': [160, 17, 200, 54], 'text': 'cannot locate this', 'score': .5,
            'source': 'baseline', 'baseline_index': 1}
    final, stats = o.materialize_atomic(args[3]+[body], plan, {}, materialize_splits)
    assert stats['restored'] == 1 and not any(region.get('candidate_id') == 'full' for region in final)
    output = o.commit_ownership(args[0], [plan['parents']['0']['entry']], args[2],
                                fused(args[3]), final, args[4])
    assert output['final_regions'] == final and output['owned_ownership']['suppressed'] == {}
    assert any(region.get('candidate_id') == 'short' for region in final)


def test_split_child_index_collision_cannot_claim_unrelated_production_face():
    args = fixture(body=True)
    image, _, candidates, regions, config = args
    production = [
        {'bbox': [0, 0, 10, 10], 'text': 'unrelated', 'score': .5},
        {'bbox': [0, 10, 208, 65], 'text': '(x_x)ノ body', 'score': .5},
        {'bbox': [212, 17, 250, 54], 'text': '(o_o)', 'score': .5}]
    plan = {'parents': {'1': {'entry': production[1], 'children': [2, 3]}},
            'children': {'2': 1, '3': 1}, 'map': {'0': 0, '1': 2}}
    # Split-child index 2 is deliberately also a valid, different production
    # index whose text happens to parse as a face. The materializer retains that
    # child index and annotates its actual parent instead of remapping it.
    candidates[0]['baseline_indices'] = [2]
    regions[0]['baseline_indices'] = [2]
    body = {'bbox': [182, 19, 208, 52], 'text': 'body', 'score': .5,
            'source': 'baseline', 'baseline_index': 3}
    final, stats = o.materialize_atomic(regions+[body], plan, {}, materialize_splits)
    child = next(region for region in final if region.get('candidate_id') == 'full')
    assert child['split_parent_index'] == 1 and child['baseline_indices'] == [2]
    assert stats['restored'] == 0
    output = o.commit_ownership(image, production, candidates, fused(regions), final, config)
    assert output['final_regions'] == final and output['owned_ownership']['suppressed'] == {}
    assert any(item.get('candidate_id') == 'full'
               and item['reason'] == 'unmapped-split-child-production-ownership'
               for item in output['owned_ownership']['fallbacks'])
    receipt = output['owned_ownership']['candidate_evidence']['full']
    assert receipt['operation'] == 'unsupported'
    assert receipt['production_parent_indices'] == []
    assert receipt['materialized_split_parent_index'] == 1


def test_lifted_marker_without_complete_parent_splice_proof_is_unsupported():
    args = fixture()
    args[3][0]['lifted_from_child'] = 0
    output = commit(args)
    assert output['final_regions'] == args[3] and output['owned_ownership']['suppressed'] == {}
    assert any(item.get('candidate_id') == 'full'
               and item['reason'] == 'unmapped-split-child-production-ownership'
               for item in output['owned_ownership']['fallbacks'])


def test_lifted_splice_with_complete_production_proof_still_owns_final_face():
    args = fixture(body=True)
    args[3][0] = splice_region(args[3][0], args[1][0], '(x_x)ノ', 'body ', ' tail')
    args[3][0].update(split_parent_index=0, lifted_from_child=7)
    output = commit(args)
    assert output['final_regions'] == [args[3][0]]
    assert output['owned_ownership']['suppressed']['short']['winner'] == 'full'
    assert output['owned_ownership']['candidate_evidence']['full']['production_parent_indices'] == [0]


def test_forged_original_pixel_claim_and_duplicate_identity_fail_closed():
    args = fixture()
    args[2][0]['evidence']['pair_geometry']['right_bbox'] = [180, 20, 200, 50]
    with pytest.raises(ValueError, match='original pixels'):
        commit(args)
    args = fixture()
    args[2].append(copy.deepcopy(args[2][0]))
    with pytest.raises(ValueError, match='identity'):
        commit(args)


def atomic_fixture():
    parent = {'bbox': [0, 0, 250, 20], 'text': 'ab(w_w) cd(x_x) ef', 'score': .5}
    plan = {'parents': {'0': {'entry': parent, 'children': [0, 1]}},
            'children': {'0': 0, '1': 0}, 'map': {}}
    regions = []
    for index, text, prefix, suffix in [(0, '(o_o)', 'ab', ' cd'), (1, '(O_O)', '', ' ef')]:
        regions.append({'bbox': [index*135, 0, 130+index*120, 20], 'text': prefix+text+suffix,
                        'score': .9, 'source': 'baseline-with-regional-splice', 'baseline_index': index,
                        'regional_bbox': [index*135, 0, 130+index*120, 20], 'regional_text': text,
                        'candidate_id': f'child-{index}', 'model_id': 'fixture-only',
                        'splice': {'preserved_prefix': prefix, 'preserved_suffix': suffix,
                                   'replacement_text': text, 'method': 'whole-reading-agreement'}})
    return regions, plan


def test_two_child_whole_splices_restore_parent_once_atomically():
    regions, plan = atomic_fixture()
    legacy, _ = materialize_splits(regions, plan, {})
    assert len(legacy) == 2 and legacy[0]['text'] != legacy[1]['text']
    output, stats = o.materialize_atomic(regions, plan, {}, materialize_splits)
    assert len(output) == 1 and output[0]['text'] == plan['parents']['0']['entry']['text']
    assert output[0]['split_restored'] and output[0]['atomic_restored']
    assert stats['restored'] == 1 and stats['parent_level_splices'] == 0
    assert stats['atomic_conflicts'][0]['candidate_ids'] == ['child-0', 'child-1']


def test_single_valid_child_splice_still_lifts_with_exact_parent_body():
    regions, plan = atomic_fixture()
    untouched = {'bbox': regions[1]['bbox'], 'text': '(x_x) ef', 'score': .5,
                 'source': 'baseline', 'baseline_index': 1}
    output, stats = o.materialize_atomic([regions[0], untouched], plan, {}, materialize_splits)
    assert len(output) == 1 and output[0]['text'] == 'ab(o_o) cd(x_x) ef'
    assert output[0]['candidate_id'] == 'child-0'
    assert output[0]['splice']['preserved_prefix'] == 'ab'
    assert output[0]['splice']['preserved_suffix'] == ' cd(x_x) ef'
    assert stats['restored'] == 0 and stats['parent_level_splices'] == 1


def test_atomic_preserves_unrelated_parentless_region_and_empty_plan_compatibility():
    regions, plan = atomic_fixture()
    extra = {'bbox': [0, 30, 30, 50], 'text': '(o_o)', 'score': .9,
             'source': 'specialist-exact-views', 'candidate_id': 'independent', 'baseline_indices': []}
    output, _ = o.materialize_atomic(regions+[extra], plan, {}, materialize_splits)
    assert extra in output and len(output) == 2
    empty = {'parents': {}, 'children': {}, 'map': {}}
    assert o.materialize_atomic([extra], empty, {}, materialize_splits) == materialize_splits([extra], empty, {})


@pytest.mark.parametrize('value', [None, {}, True, {'strategy': 'rectangle'},
                                  {'strategy': 'atomic-face-ownership-v1', 'disable_guard': True}])
def test_fusion_owned_option_requires_exact_declared_contract(value):
    with pytest.raises(ValueError):
        f.configuration({**f.DEFAULT_CONFIG, 'owned_ownership': value})


def test_legacy_configuration_is_unchanged_and_new_option_is_explicit():
    assert f.configuration() == f.DEFAULT_CONFIG
    assert f.configuration({**f.DEFAULT_CONFIG, 'owned_ownership': o.configuration()})['owned_ownership'] == o.configuration()
