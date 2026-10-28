"""Real localization through fusion, with declared fixture OCR only."""
import copy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools.RegionKaomoji import candidates as c, fusion as f, line_spans


def request(*, body=True, baseline=None):
    image = Image.new('RGB', (260, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.arc((40, 20, 54, 50), 90, 270, fill='black', width=2)
    draw.arc((108, 20, 122, 50), 270, 90, fill='black', width=2)
    draw.ellipse((63, 31, 70, 38), outline='black', width=2)
    draw.ellipse((87, 31, 94, 38), outline='black', width=2)
    draw.line((75, 43, 83, 43), fill='black', width=2)
    draw.line((138, 28, 148, 44), fill='black', width=2)
    if body:
        draw.rectangle((154, 22, 180, 48), fill='black')
    image = np.asarray(image)
    config = {**copy.deepcopy(c.DEFAULT_CONFIG),
              'pair_recrop': {'maximum_gap_height': .58, 'maximum_exterior_cells': 2},
              'line_spans': line_spans.configuration()}
    baseline = [] if baseline is None else baseline
    candidates = c.locate_candidates(image, baseline, config)['candidates']
    views = {}
    for item in candidates:
        text = '(ω_ω)ノ' if item['evidence']['ink_bbox'][2] >= 150 else '(ω_ω)'
        views[item['candidate_id']] = [
            {'model_id': 'declared-fixture-only', 'view_id': transform, 'transform': transform,
             'image_sha256': c.image_digest(f.view_image(image, item['bbox'], transform)),
             'text': text, 'score': .9} for transform in ('raw', 'edge-pad-6')]
    return image, baseline, candidates, views, config


def run(args, *, owned=True):
    image, baseline, candidates, views, candidate_config = args
    config = {**copy.deepcopy(f.DEFAULT_CONFIG),
              **({'pixel_ownership': {'strategy': 'component-exclusion-v1'}} if owned else {})}
    return f.fuse_candidates(image, baseline, candidates, views, config,
                             candidate_config=candidate_config)


def test_real_candidate_producer_reproduces_old_duplicate_and_opt_in_keeps_arm_once():
    args = request()
    legacy = run(args, owned=False)
    assert [r['text'] for r in legacy['regions']] == ['(ω_ω)ノ', '(ω_ω)']
    for ordering in (args[2], list(reversed(args[2]))):
        output = run((*args[:2], ordering, *args[3:]))
        assert [r['text'] for r in output['regions']] == ['(ω_ω)ノ']
        assert output['accepted_count'] == 1
        suppressed = [d for d in output['decisions'] if d['reason'] == 'pixel-owned-complete-expression']
        assert len(suppressed) == 1 and suppressed[0]['pixel_ownership']['shared_core_pixels'] > 0


def test_ambiguous_complete_reading_retains_original_result_without_new_duplicate():
    args = request()
    # The original producer's full candidate is accepted under the existing
    # parentless guards even if declared OCR omits its arm. New exclusion must
    # not use that omission as permission for a larger unproved takeover.
    for records in args[3].values():
        for record in records:
            record['text'] = '(ω_ω)'
    output = run(args)
    assert len(output['regions']) == 1
    assert output['regions'][0]['candidate_id'].startswith('candidate-')
    assert any(d['reason'] == 'pixel-owned-ambiguous-preserve-original' for d in output['decisions'])


def test_existing_view_guards_still_reject_before_ownership():
    args = request()
    for records in args[3].values():
        records[1]['text'] = '(ω _ω)'
    output = run(args)
    assert output['regions'] == [] and output['accepted_count'] == 0
    assert all(d['reason'] != 'pixel-owned-complete-expression' for d in output['decisions'])


def test_real_adjacent_faces_keep_equal_readings_with_overlapping_padding():
    image = Image.new('RGB', (260, 80), 'white')
    draw = ImageDraw.Draw(image)
    for x in (30, 125):
        draw.arc((x, 20, x+14, 50), 90, 270, fill='black', width=2)
        draw.arc((x+65, 20, x+79, 50), 270, 90, fill='black', width=2)
        draw.ellipse((x+24, 30, x+30, 37), outline='black', width=2)
        draw.ellipse((x+46, 30, x+52, 37), outline='black', width=2)
        draw.line((x+34, 43, x+42, 43), fill='black', width=2)
    image = np.asarray(image)
    config = {**c.DEFAULT_CONFIG, 'padding': 10,
              'pair_selection': {'family_match': True, 'innermost_partner': True},
              'pair_recrop': {'maximum_gap_height': .58, 'maximum_exterior_cells': 2}}
    candidates = c.locate_candidates(image, [], config)['candidates']
    views = {item['candidate_id']: [
        {'model_id': 'declared-fixture-only', 'view_id': transform, 'transform': transform,
         'image_sha256': c.image_digest(f.view_image(image, item['bbox'], transform)),
         'text': '(ω_ω)', 'score': .9} for transform in ('raw', 'edge-pad-6')]
        for item in candidates}
    output = run((image, [], candidates, views, config))
    assert len(output['regions']) == 2 and all(r['text'] == '(ω_ω)' for r in output['regions'])
    assert c.intersection(*[r['bbox'] for r in output['regions']]) > 0
    assert all('pixel_ownership' not in d for d in output['decisions'])


def test_missing_short_view_or_forged_pixels_never_steals_valid_complete_insertion():
    for forged in (False, True):
        args = request()
        short = next(item for item in args[2] if item['sources'] == ['image-line-span'] and item['route_eligible'])
        records = args[3][short['candidate_id']]
        if forged:
            records[0]['image_sha256'] = '0'*64
        else:
            records.pop()
        output = run(args)
        assert len(output['regions']) == 1 and output['regions'][0]['text'] == '(ω_ω)ノ'
        decision = next(d for d in output['decisions'] if d['candidate_id'] == short['candidate_id'])
        assert decision['reason'] == ('view-pixels-differ' if forged else 'insufficient-independent-views')
        assert 'pixel_ownership' not in decision


def test_forged_candidate_or_ownership_proof_still_fails_original_regeneration():
    args = request()
    forged = copy.deepcopy(args[2])
    forged[0]['evidence']['pixel_ownership'] = {'safe': True, 'winner': 'forged'}
    with pytest.raises(ValueError, match='differs from automatic'):
        run((*args[:2], forged, *args[3:]))


def test_baseline_replacement_and_body_guards_are_unchanged():
    baseline = [{'bbox': [36, 16, 184, 54], 'text': '正文123', 'score': .7}]
    args = request(baseline=baseline)
    legacy, opted = run(args, owned=False), run(args)
    assert legacy['regions'] == opted['regions']
    assert opted['regions'][0]['text'] == '正文123'
    assert not any('pixel_ownership' in decision for decision in opted['decisions'])


def test_absent_option_is_not_injected_or_called(monkeypatch):
    from tools.RegionKaomoji import pixel_ownership
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing option must not execute new pass')
    monkeypatch.setattr(pixel_ownership, 'resolve_conflicts', forbidden)
    first = run(request(), owned=False)
    second = run(request(), owned=False)
    assert first == second
    assert 'pixel_ownership' not in first['config']
    assert all('pixel_ownership' not in d for d in first['decisions'])


@pytest.mark.parametrize('value', [None, {}, True, {'strategy': 'rectangles'}])
def test_fusion_optional_contract_rejects_incomplete_or_unknown_values(value):
    with pytest.raises(ValueError):
        f.configuration({**f.DEFAULT_CONFIG, 'pixel_ownership': value})
