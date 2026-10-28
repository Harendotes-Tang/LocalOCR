"""Real automatic proof/fusion integration; declared OCR outputs are fixtures.

No inference or scene truth enters localization. These tests exercise mechanics
from original pixels through actual candidate regeneration, rather than claiming
that a model recognized the drawn shapes or that OCR quality improved.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools.RegionKaomoji import candidates as c
from tools.RegionKaomoji import fusion as f
from tools.RegionKaomoji import local_ownership


def fixture_request():
    canvas = Image.new('RGB', (400, 80), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.arc((55, 20, 69, 50), 90, 270, fill='black', width=2)
    draw.arc((301, 20, 315, 50), 270, 90, fill='black', width=2)
    for x in range(80, 295, 20):
        draw.ellipse((x, 31, x + 7, 38), outline='black', width=2)
    # Complete dense body glyphs, with original blank bands on both sides.
    draw.rectangle((22, 23, 43, 46), fill='black')
    draw.rectangle((327, 23, 348, 46), fill='black')
    image = np.asarray(canvas)
    left, right = '左A\u03011', '右B2'
    prefix, suffix = left + '\u00a0\u3000', '\u2009' + right
    baseline = [{'bbox': [19, 17, 352, 54], 'text': prefix + '(Д_Д)' + suffix, 'score': .4}]
    config = dict(c.DEFAULT_CONFIG, local_ownership={'strategy': 'pixel-partition-v1'})
    located = c.locate_candidates(image, baseline, config)
    candidate = next(item for item in located['candidates']
                     if item['route_eligible'] and 'local_ownership' in item['evidence'])
    identity = candidate['candidate_id']
    views = {identity: [
        {'model_id': 'mechanics-fixture-specialist', 'view_id': transform,
         'transform': transform, 'image_sha256': c.image_digest(f.view_image(image, candidate['bbox'], transform)),
         'text': '(ω_ω)', 'score': .9}
        for transform in ('raw', 'edge-pad-6')]}
    contexts = {identity: [
        {'role': declaration['role'], 'bbox': declaration['bbox'][:],
         'image_sha256': c.image_digest(f.view_image(image, declaration['bbox'], 'raw')),
         'text': left if declaration['role'] == 'left' else right,
         'score': .4, 'model_id': 'production-baseline',
         'baseline_parent_sha256': candidate['baseline_parent_sha256']}
        for declaration in candidate['context_crops']]}
    return {'image': image, 'baseline': baseline, 'config': config,
            'candidates': located['candidates'], 'candidate': candidate,
            'views': views, 'contexts': contexts, 'prefix': prefix, 'suffix': suffix}


def fuse(request):
    return f.fuse_candidates(request['image'], request['baseline'], request['candidates'],
                             request['views'], candidate_config=request['config'],
                             contexts=request['contexts'])


def source_digests():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (root / 'localocr', root / 'tools')
            for path in directory.rglob('*.py')}


def test_real_local_proof_enables_byte_exact_body_preserving_splice():
    before = source_digests()
    request = fixture_request()
    candidate = request['candidate']
    entries = c.baseline_regions(request['baseline'], (400, 80))
    normalized = c.configuration(request['config'])
    mask, _ = c._masks(request['image'], normalized)
    assert candidate['baseline_indices'] == [0]
    # A geometrically owned parent previously bypassed context construction.
    assert c._mixed_contexts(mask, candidate['bbox'], [entries[0]], [], entries, normalized) == (None, [])
    proof = candidate['evidence']['local_ownership']
    assert local_ownership.audit_local_partition(proof, mask, candidate['bbox'], entries, [0], [])
    assert proof['parent_index'] == candidate['splice_baseline_index'] == 0
    assert proof['proof']['parent_sha256'] == candidate['baseline_parent_sha256']
    assert [item['role'] for item in proof['context_crops']] == ['left', 'right']
    assert proof['proof']['support_visible_pixels'] == (
        proof['proof']['roi_visible_pixels']
        + sum(item['visible_pixels'] for item in proof['context_crops']))

    output = fuse(request)  # Regenerates the actual candidate/proof from current pixels.
    assert output['accepted_count'] == 1
    assert len(output['regions']) == 1
    region = output['regions'][0]
    assert region['source'] == 'baseline-with-regional-splice'
    assert region['text'] == request['prefix'] + '(ω_ω)' + request['suffix']
    assert region['bbox'] == request['baseline'][0]['bbox']
    assert region['score'] == request['baseline'][0]['score']
    assert region['splice']['preserved_prefix'].encode('utf-8') == request['prefix'].encode('utf-8')
    assert region['splice']['preserved_suffix'].encode('utf-8') == request['suffix'].encode('utf-8')
    assert region['splice']['original_face_text'] == '(Д_Д)'
    assert source_digests() == before


def test_optional_path_is_required_to_obtain_context_splice_proof():
    request = fixture_request()
    del request['config']['local_ownership']
    request['candidates'] = c.locate_candidates(request['image'], request['baseline'], request['config'])['candidates']
    assert all('local_ownership' not in item['evidence'] and item['splice_baseline_index'] is None
               for item in request['candidates'])


@pytest.mark.parametrize('alteration,reason', [
    ('context-parent', 'baseline-context-parent-provenance-differs'),
    ('context-pixels', 'baseline-context-pixels-differ'),
    ('context-text', 'context-not-unique-raw-prefix-suffix'),
    ('insufficient-views', 'insufficient-independent-views'),
    ('disagreeing-views', 'exact-unicode-view-disagreement'),
])
def test_context_or_view_failure_keeps_every_original_body_byte(alteration, reason):
    request = fixture_request()
    identity = request['candidate']['candidate_id']
    if alteration == 'context-parent':
        request['contexts'][identity][0]['baseline_parent_sha256'] = '0' * 64
    elif alteration == 'context-pixels':
        request['contexts'][identity][0]['image_sha256'] = '0' * 64
    elif alteration == 'context-text':
        request['contexts'][identity][0]['text'] = '左A1'  # Combining mark is a real raw-byte difference.
    elif alteration == 'insufficient-views':
        request['views'][identity].pop()
    elif alteration == 'disagreeing-views':
        request['views'][identity][1]['text'] = '（ω_ω）'
    output = fuse(request)
    assert output['accepted_count'] == 0
    assert len(output['regions']) == 1
    assert output['regions'][0]['source'] == 'baseline'
    assert output['regions'][0]['text'].encode('utf-8') == request['baseline'][0]['text'].encode('utf-8')
    assert next(item for item in output['decisions'] if item['candidate_id'] == identity)['reason'] == reason


def test_candidate_proof_is_regenerated_and_forged_digest_is_rejected():
    request = fixture_request()
    request['candidates'] = copy.deepcopy(request['candidates'])
    identity = request['candidate']['candidate_id']
    candidate = next(item for item in request['candidates'] if item['candidate_id'] == identity)
    candidate['evidence']['local_ownership']['proof']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='differs from automatic original-image localization'):
        fuse(request)
