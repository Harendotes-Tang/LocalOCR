"""Pre-recognition split, exterior-anchor routing and split materialization; no model claim."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tools.RegionKaomoji import candidates as c, evaluate as e, fusion as f


def blobs(positions, width=200, height=30, blob=12):
    mask = np.zeros((height, width), dtype=np.uint8)
    for start in positions:
        mask[8:22, start:start + blob] = 1
    return mask


def test_split_children_at_blank_bands_only():
    config = c.split_configuration()
    bbox = [0, 0, 200, 30]
    assert c.split_children(blobs([10]), bbox, config) == []
    children = c.split_children(blobs([10, 60, 120]), bbox, config)
    assert len(children) == 3 and all(box[1] == 0 and box[3] == 30 for box in children)
    assert children[0][0] == 8 and children[0][2] == 24 and children[1][0] == 58
    assert c.split_children(blobs([10, 28]), bbox, config) == []  # 6-pixel gap is below the band
    assert c.split_children(blobs([10, 60]), bbox, dict(config, enabled=False)) == []
    faint = blobs([10, 60]); faint[:, 60:72] = 0; faint[10, 60:63] = 1
    assert c.split_children(faint, bbox, config) == []  # too little ink in a child
    assert c.split_children(blobs([10, 60]), bbox, dict(config, maximum_parent_width=100)) == []
    with pytest.raises(ValueError):
        c.split_children(blobs([10, 60]), [0, 0, 300, 30], config)
    for bad in ({'enabled': 1}, {'blank_band_ratio': 0}, {'maximum_children': 1}, {'minimum_band_pixels': 1}, {'nope': 1}):
        with pytest.raises(ValueError):
            c.split_configuration({**config, **bad})


@pytest.mark.parametrize('text,outside,digits', [
    ('(w_w)', [], []), ('(ABC)', [], []), ('o (ω_ω)', ['o'], []), ('(ω_ω) o', ['o'], []), ('[123]', [], ['1', '2', '3']),
    ('(0_0)', [], ['0', '0']), ('つ(Д_Д)つ', [], []), ('w_w', ['w', 'w'], []), ('(ω3ω)', [], ['3']), ('订单 (ω_ω)', ['订', '单'], []),
])
def test_exterior_letters_and_digits_are_the_only_text_anchors(text, outside, digits):
    assert c.ordinary_anchor_evidence(text) == (outside, digits)


def angular(height=24):
    top, left = 10, 10
    right, bottom = left + height * 2, top + height - 1
    image = Image.new('RGB', (right + 20, bottom + 20), 'white')
    draw = ImageDraw.Draw(image)
    cap = max(3, height // 4)
    draw.line([(left + cap - 1, top), (left, top), (left, bottom), (left + cap - 1, bottom)], fill='black', width=1)
    draw.line([(right - cap + 1, top), (right, top), (right, bottom), (right - cap + 1, bottom)], fill='black', width=1)
    eye_y = top + max(2, height // 3)
    for x in (left + height * 2 // 3, left + height * 4 // 3):
        draw.rectangle((x, eye_y, x + 1, eye_y + 1), fill='black')
    draw.line((left + height - 1, top + height - 3, left + height + 1, top + height - 3), fill='black', width=1)
    return np.asarray(image)


@pytest.mark.parametrize('original', ['[ABC]', '[w_w]', '[123]', '[333]', 'o [ω3ω]', '[ω3ω] o', '[o]', '[T]', 'iC[<·<]', 'ノっ()!'])
def test_measured_bracket_strokes_route_regardless_of_baseline_reading(original):
    # With measured bracket strokes and interior ink, the baseline's own reading of the ROI is not evidence.
    baseline = [{'bbox': [10, 10, 59, 34], 'text': original, 'score': .1}]
    output = c.locate_candidates(angular(), baseline)
    eligible = [item for item in output['candidates'] if item['route_eligible']]
    assert eligible and all(item['route_reason'] == 'independent-pixels-and-bounded-face-evidence' for item in eligible)
    assert all(item['evidence']['pair_geometry']['interior_components'] >= 2 for item in eligible)


def lone_token_fixture():
    image = Image.new('RGB', (80, 44), 'white')
    draw = ImageDraw.Draw(image)
    draw.line([(16, 10), (12, 10), (12, 33), (16, 33)], fill='black', width=1)
    draw.line([(52, 10), (56, 10), (56, 33), (52, 33)], fill='black', width=1)
    draw.ellipse((28, 17, 40, 27), outline='black', width=2)
    return np.asarray(image)


def test_single_interior_component_is_body_not_face():
    output = c.locate_candidates(lone_token_fixture(), [{'bbox': [10, 8, 58, 36], 'text': '[o]', 'score': .3}])
    assert output['candidates'] and not any(item['route_eligible'] for item in output['candidates'])
    assert any(item['route_reason'] == 'insufficient-interior-ink' for item in output['candidates'])


class StubClassifier:
    binding = {'id': 'stub', 'sha256': 'b' * 64}
    threshold = .7

    def evidence(self, image):
        return {'id': 'stub', 'threshold': .7, 'probabilities': {'face': .95, 'bracketed-text': .03, 'graphic': .02},
                'face_probability': .95, 'face_evidence': True}


def no_stroke_fixture():
    image = Image.new('RGB', (160, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 23, 50), fill='black')
    draw.rectangle((100, 20, 103, 50), fill='black')
    draw.rectangle((45, 30, 50, 36), fill='black')
    draw.rectangle((72, 30, 77, 36), fill='black')
    draw.line((54, 43, 68, 43), fill='black', width=2)
    return np.asarray(image)


@pytest.mark.parametrize('original,reason', [
    ('[123]', 'observed-ordinary-numbers'), ('o [ω3ω]', 'observed-ordinary-body-anchors'),
    ('[ω3ω] o', 'observed-ordinary-body-anchors'), ('[o]', 'ordinary-body-or-single-letter-brackets'),
])
def test_text_guards_remain_for_classifier_only_evidence(original, reason):
    baseline = [{'bbox': [18, 18, 106, 53], 'text': original, 'score': .2}]
    output = c.locate_candidates(no_stroke_fixture(), baseline, classifier=StubClassifier())
    owning = [item for item in output['candidates'] if item['baseline_indices']]
    assert owning and not any(item['route_eligible'] for item in owning)
    assert any(item['route_reason'] == reason for item in owning)
    assert all(item['evidence']['pair_geometry'] is None for item in output['candidates'])


def face_pixels():
    image = Image.new('RGB', (160, 80), 'white')
    draw = ImageDraw.Draw(image)
    draw.arc((20, 20, 34, 50), 90, 270, fill='black', width=2)
    draw.arc((88, 20, 102, 50), 270, 90, fill='black', width=2)
    draw.ellipse((43, 31, 50, 38), outline='black', width=2)
    draw.ellipse((72, 31, 79, 38), outline='black', width=2)
    draw.line((54, 43, 68, 43), fill='black', width=2)
    return np.asarray(image)


def fuse(original, text='(ω_ω)', image=None, bbox=(20, 20, 103, 51)):
    image = face_pixels() if image is None else image
    baseline = [{'bbox': list(bbox), 'text': original, 'score': .0001}]
    candidates = c.locate_candidates(image, baseline)['candidates']
    views = {}
    for candidate in candidates:
        views[candidate['candidate_id']] = [{'model_id': 'specialist', 'view_id': t, 'transform': t,
                                             'image_sha256': c.image_digest(f.view_image(image, candidate['bbox'], t)),
                                             'text': text, 'score': .9} for t in ('raw', 'edge-pad-6')]
    return f.fuse_candidates(image, baseline, candidates, views)


def face_with_exterior_blob():
    image = Image.fromarray(face_pixels())
    ImageDraw.Draw(image).ellipse((4, 30, 12, 38), fill='black')
    return np.asarray(image)


def test_fusion_accounts_for_pixels_not_for_the_baseline_reading():
    # Any baseline reading of a pixel-confirmed face may be replaced; the pixels are the evidence.
    for misread in ('(w_w)', '(A_A)', '(ABC)', '(TT)', '(A1A)', '(0_0)', 'o (A_A)', 'iC(<·<)', '()'):
        output = fuse(misread)
        assert output['accepted_count'] == 1 and output['regions'][0]['text'] == '(ω_ω)', misread
        assert output['regions'][0]['baseline_indices'] == [0]
    # Exterior ink must be accounted for by exterior atoms, and arms cannot be invented.
    blob = face_with_exterior_blob()
    output = fuse('o (A_A)', image=blob, bbox=(2, 20, 103, 51))
    assert output['accepted_count'] == 0 and any(item['reason'] == 'exterior-ink-and-arms-disagree' for item in output['decisions'])
    output = fuse('o (A_A)', text='つ(ω_ω)', image=blob, bbox=(2, 20, 103, 51))
    assert output['accepted_count'] == 1 and output['regions'][0]['text'] == 'つ(ω_ω)'
    output = fuse('(A_A)', text='つ(ω_ω)')
    assert output['accepted_count'] == 0 and any(item['reason'] == 'exterior-ink-and-arms-disagree' for item in output['decisions'])
    output = fuse('[o]', image=lone_token_fixture(), bbox=(10, 8, 58, 36))
    assert output['accepted_count'] == 0 and any(item['reason'] == 'insufficient-interior-ink' for item in output['decisions'])
    assert fuse('(ABC)', text='(ABC)')['accepted_count'] == 0  # non-face output never replaces


class SplitStub:
    """Declared stub: a production reading of one merged box and a child recognizer."""
    binding = {'id': 'production-baseline'}

    def __init__(self, texts):
        self.texts = texts

    def predict(self, image):
        return [{'bbox': [8, 6, 112, 52], 'text': 'LEFT(w_w)', 'score': .2}], {'unit_fixture': True}

    def first_pass(self, image, split_config):
        entries, native = self.predict(image)
        first_pass, plan = e.split_entries(image, entries, split_config, lambda crops: list(self.texts[:len(crops)]))
        return entries, native, first_pass, plan


def merged_scene():
    image = Image.new('RGB', (200, 60), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 20, 40, 40), fill='black')  # body ink block
    draw.arc((60, 15, 74, 45), 90, 270, fill='black', width=2)
    draw.arc((96, 15, 110, 45), 270, 90, fill='black', width=2)
    draw.ellipse((78, 26, 83, 31), outline='black', width=2)
    draw.ellipse((88, 26, 93, 31), outline='black', width=2)
    draw.line((82, 38, 90, 38), fill='black', width=2)
    return image


def test_split_entries_recognize_children_and_keep_parent_plan():
    image = merged_scene()
    entries = [{'bbox': [8, 6, 112, 52], 'text': 'LEFT(w_w)', 'score': .2}]
    config = dict(c.split_configuration(), blank_band_ratio=.3)
    first_pass, plan = e.split_entries(image, entries, config, lambda crops: [('LEFT', .9), ('(w_w)', .4)][:len(crops)])
    assert len(first_pass) == 2 and [item['text'] for item in first_pass] == ['LEFT', '(w_w)']
    assert plan['parents']['0']['children'] == [0, 1] and plan['children'] == {'0': 0, '1': 0} and plan['map'] == {}
    assert first_pass[0]['bbox'][2] <= first_pass[1]['bbox'][0] and first_pass[1]['bbox'][1] == 6
    disabled, plan2 = e.split_entries(image, entries, dict(config, enabled=False), lambda crops: pytest.fail('no recognition'))
    assert [item['text'] for item in disabled] == ['LEFT(w_w)'] and plan2['parents'] == {}
    with pytest.raises(ValueError, match='count'):
        e.split_entries(image, entries, config, lambda crops: [])


def test_materialize_keeps_split_only_when_a_child_was_replaced():
    plan = {'parents': {'1': {'entry': {'bbox': [0, 0, 100, 20], 'text': 'A(w_w)', 'score': .5}, 'children': [1, 2]}},
            'children': {'1': 1, '2': 1}, 'map': {'0': 0, '3': 2}}
    untouched = [{'bbox': [0, 30, 20, 40], 'text': 'x', 'score': .1, 'source': 'baseline', 'baseline_index': 0},
                 {'bbox': [0, 0, 30, 20], 'text': 'A', 'score': .5, 'source': 'baseline', 'baseline_index': 1},
                 {'bbox': [40, 0, 100, 20], 'text': '(w_w)', 'score': .5, 'source': 'baseline', 'baseline_index': 2},
                 {'bbox': [0, 50, 20, 60], 'text': 'y', 'score': .1, 'source': 'baseline', 'baseline_index': 3}]
    regions, stats = e.materialize_splits(untouched, plan, {})
    assert stats == {'materialized': 0, 'restored': 1, 'restored_unverifiable': 0, 'parent_level_splices': 0}
    assert [r['text'] for r in regions] == ['A(w_w)', 'x', 'y'] and regions[0]['split_restored'] and regions[0]['baseline_index'] == 1
    assert [r['baseline_index'] for r in regions] == [1, 0, 2]
    replaced = list(untouched)
    replaced[2] = {'bbox': [40, 0, 100, 20], 'text': '(ω_ω)', 'score': .9, 'source': 'specialist-exact-views', 'baseline_indices': [2]}
    regions, stats = e.materialize_splits(replaced, plan, {})
    assert stats == {'materialized': 1, 'restored': 0, 'restored_unverifiable': 0, 'parent_level_splices': 0}
    assert [r['text'] for r in regions] == ['A', '(ω_ω)', 'x', 'y']
    assert regions[0]['source'] == 'baseline-split-child' and regions[0]['split_parent_index'] == 1 and regions[0]['child_reading'] == 'A'
    assert regions[1]['baseline_indices'] == [2] and regions[1]['split_parent_index'] == 1
    assert regions[2]['baseline_index'] == 0 and regions[3]['baseline_index'] == 2
    # A body child that lost a space carries the parent's own bytes; an unlocatable child restores the parent.
    spaced = {'parents': {'1': {'entry': {'bbox': [0, 0, 100, 20], 'text': 'water 房间 麦田(w_w)', 'score': .5}, 'children': [1, 2]}},
              'children': {'1': 1, '2': 1}, 'map': {'0': 0, '3': 2}}
    lost_space = list(replaced)
    lost_space[1] = dict(replaced[1], text='water房间麦田')
    regions, stats = e.materialize_splits(lost_space, spaced, {})
    assert stats['materialized'] == 1 and regions[0]['text'] == 'water 房间 麦田' and regions[0]['child_reading'] == 'water房间麦田'
    garbage = list(replaced)
    garbage[1] = dict(replaced[1], text='different reading')
    regions, stats = e.materialize_splits(garbage, spaced, {})
    assert stats == {'materialized': 0, 'restored': 1, 'restored_unverifiable': 1, 'parent_level_splices': 0}
    assert [r['text'] for r in regions] == ['water 房间 麦田(w_w)', 'x', 'y'] and regions[0]['split_unverifiable']


def test_predict_scene_restores_parent_when_no_child_becomes_a_face(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'EVALUATION_ROOT', tmp_path)
    config = e.configuration()
    config['split'] = dict(config['split'], blank_band_ratio=.3)
    class NoCall:
        binding = {'id': 'unit'}
        def predict(self, image):
            return '(ω_ω)', .9
    result = e.predict_scene(merged_scene(), SplitStub([('LEFT', .9), ('(w_w)', .4)]), [NoCall()], config)
    assert result['baseline'][0]['text'] == 'LEFT(w_w)' and result['first_pass']['split_parents'] == 1
    assert [item['text'] for item in result['first_pass']['entries']] == ['LEFT', '(w_w)']
    model = result['models']['unit']
    assert (model['split']['materialized'], model['split']['restored']) in ((1, 0), (0, 1))
    if model['split']['materialized']:
        assert [r['text'] for r in model['regions']] == ['LEFT', '(ω_ω)']
    else:
        assert [r['text'] for r in model['regions']] == ['LEFT(w_w)'] and model['regions'][0]['split_restored']


def test_regression_split_requires_registered_demotion(tmp_path, monkeypatch):
    from tools.PretrainedOcr import round_profile as rp
    monkeypatch.setattr(rp, 'ROUNDS', ())
    monkeypatch.setattr(e, 'scene_binding', lambda root: {'split_sha256': {'heldout': 'x'}})
    monkeypatch.setattr(e.common, 'read', lambda path: {'splits': {'heldout': {'file': 'heldout.json'}}})
    (tmp_path / 'heldout.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen heldout split changed'):
        e.demotion_receipt(tmp_path)
    assert e.data_split(e.REGRESSION_SPLIT) == 'heldout' and e.data_split('dev') == 'dev'


def test_adjacent_body_children_coalesce_into_the_parent_bytes():
    plan = {'parents': {'0': {'entry': {'bbox': [0, 0, 160, 20], 'text': 'A, B(w_w)', 'score': .5}, 'children': [0, 1, 2]}},
            'children': {'0': 0, '1': 0, '2': 0}, 'map': {}}
    regions = [{'bbox': [0, 0, 30, 20], 'text': 'A,', 'score': .5, 'source': 'baseline', 'baseline_index': 0},
               {'bbox': [40, 0, 60, 20], 'text': 'B', 'score': .5, 'source': 'baseline', 'baseline_index': 1},
               {'bbox': [70, 0, 160, 20], 'text': '(ω_ω)', 'score': .9, 'source': 'specialist-exact-views', 'baseline_indices': [2]}]
    output, stats = e.materialize_splits(regions, plan, {})
    assert stats['materialized'] == 1
    body = [r for r in output if r['source'] == 'baseline-split-child']
    assert len(body) == 1 and body[0]['text'] == 'A, B' and body[0]['bbox'] == [0, 0, 60, 20]
    assert body[0]['child_reading'] == 'A, B' and body[0]['merged_children'] == [0, 1]
    assert e.split_children_keep_parent_reading(body, [plan['parents']['0']['entry']])
    # A face between two body children keeps them separate.
    regions[1], regions[2] = regions[2], dict(regions[1], bbox=[120, 0, 160, 20], baseline_index=2)
    regions[1] = dict(regions[1], bbox=[40, 0, 110, 20], baseline_indices=[1])
    plan['parents']['0']['entry']['text'] = 'A,(w_w)B'
    output, stats = e.materialize_splits(regions, plan, {})
    assert [r['text'] for r in output if r['source'] == 'baseline-split-child'] == ['A,', 'B']


def splice_region(index, text, face, prefix, suffix, bbox):
    return {'bbox': bbox, 'text': text, 'score': .5, 'source': 'baseline-with-regional-splice', 'baseline_index': index,
            'regional_bbox': [bbox[0] + 10, bbox[1], bbox[2] - 10, bbox[3]], 'regional_ink_bbox': [bbox[0] + 10, bbox[1], bbox[2] - 10, bbox[3]],
            'regional_text': face, 'candidate_id': 'candidate-000', 'model_id': 'm',
            'splice': {'baseline_index': index, 'original_text_span': [len(prefix), len(text) - len(suffix) - len(face) + len(prefix)],
                       'original_face_text': 'x', 'replacement_text': face, 'preserved_prefix': prefix, 'preserved_suffix': suffix,
                       'context_crops': [], 'context_records': [], 'baseline_parent_sha256': 'y', 'ordinary_prefix_suffix_preserved': True,
                       'method': 'whole-reading-agreement'}}


def test_whole_reading_splice_on_a_split_child_lifts_to_the_production_parent():
    plan = {'parents': {'0': {'entry': {'bbox': [0, 0, 200, 20], 'text': 'ab(w_w) cd', 'score': .5}, 'children': [0, 1]}},
            'children': {'0': 0, '1': 0}, 'map': {}}
    regions = [splice_region(0, 'ab(ω_ω)', '(ω_ω)', 'ab', '', [0, 0, 120, 20]),
               {'bbox': [130, 0, 200, 20], 'text': 'cd', 'score': .5, 'source': 'baseline', 'baseline_index': 1}]
    output, stats = e.materialize_splits(regions, plan, {})
    assert stats == {'materialized': 0, 'restored': 0, 'restored_unverifiable': 0, 'parent_level_splices': 1}
    assert len(output) == 1
    region = output[0]
    assert region['text'] == 'ab(ω_ω) cd' and region['bbox'] == [0, 0, 200, 20] and region['baseline_index'] == 0
    assert region['splice']['original_text_span'] == [2, 7] and region['splice']['original_face_text'] == '(w_w)'
    assert region['splice']['preserved_prefix'] == 'ab' and region['splice']['preserved_suffix'] == ' cd'
    assert region['regional_text'] == '(ω_ω)' and region['lifted_from_child'] == 0
    # Parent bytes between the located prefix and the next child that are not one bracketed fragment restore the parent.
    plan['parents']['0']['entry']['text'] = 'ab(w_w)xy cd'
    output, stats = e.materialize_splits(regions, plan, {})
    assert stats == {'materialized': 0, 'restored': 1, 'restored_unverifiable': 1, 'parent_level_splices': 0}
    assert output[0]['text'] == 'ab(w_w)xy cd' and output[0]['split_restored']
