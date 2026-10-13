"""Split-child body accounting, exterior atom stripping and ink-box splice overlap."""
import numpy as np

from tools.RegionKaomoji import candidates as c, evaluate as e


def test_strip_exterior_atoms_keeps_the_bracketed_core_only():
    assert c.strip_exterior_atoms('ノノ(∀>)!') == '(∀>)'
    assert c.strip_exterior_atoms(' つ(Д_Д)つ!! ') == '(Д_Д)'
    assert c.strip_exterior_atoms('(x□·)ノ！') == '(x□·)'
    assert c.strip_exterior_atoms('o (A_A)') == 'o (A_A)'
    assert c.strip_exterior_atoms('') == ''


def test_split_children_keep_parent_reading_requires_contiguous_parent_bytes():
    production = [{'bbox': [0, 0, 300, 40], 'text': 'ノつ（X・)！below/家园窗口ノ[°益ω]!', 'score': .3}]
    child = {'bbox': [100, 0, 200, 40], 'text': 'below/家园窗口', 'score': .5, 'source': 'baseline-split-child', 'split_parent_index': 0}
    assert e.split_children_keep_parent_reading([child], production)
    assert not e.split_children_keep_parent_reading([dict(child, text='below/家园 窗口')], production)
    assert not e.split_children_keep_parent_reading([dict(child, source='baseline')], production)
    assert not e.split_children_keep_parent_reading([dict(child, split_parent_index=3)], production)
    assert not e.split_children_keep_parent_reading([child, {'bbox': [0, 0, 10, 10], 'text': 'x', 'score': .1, 'source': 'specialist-exact-views'}], production)
    assert not e.split_children_keep_parent_reading([], production)
    untouched = {'bbox': [0, 0, 300, 40], 'text': production[0]['text'], 'score': .3, 'source': 'baseline', 'baseline_index': 0}
    assert e.split_children_keep_parent_reading([untouched], production)
    assert not e.split_children_keep_parent_reading([dict(untouched, text='changed')], production)
    face = {'bbox': [0, 0, 30, 40], 'ink_bbox': [3, 5, 27, 35], 'text': '(ω_ω)', 'score': .9, 'source': 'specialist-exact-views'}
    assert e.overlap_box(face) == [3, 5, 27, 35] and e.overlap_box(untouched) == [0, 0, 300, 40]


def test_split_children_are_recognized_on_masked_parent_crops():
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (200, 60), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 20, 40, 40), fill='black')
    draw.rectangle((120, 20, 150, 40), fill='black')
    seen = []
    def recognize(crops):
        seen.extend(crops)
        return [('A', .9), ('B', .8)][:len(crops)]
    entries = [{'bbox': [8, 6, 160, 52], 'text': 'AB', 'score': .2}]
    first_pass, plan = e.split_entries(image, entries, dict(c.split_configuration(), blank_band_ratio=.3), recognize)
    assert [item['text'] for item in first_pass] == ['A', 'B'] and len(seen) == 2
    # Every child crop keeps the parent's full extent; the other child's columns are background.
    for crop in seen:
        assert crop.shape == (46, 152, 3)
    assert seen[0][:, 120 - 8:150 - 8].min() == 255 and seen[0][:, 12 - 8:40 - 8].min() == 0
    assert seen[1][:, 12 - 8:40 - 8].min() == 255 and seen[1][:, 120 - 8:150 - 8].min() == 0


def test_locate_parent_bytes_ignores_spaces_but_outputs_parent_bytes_only():
    parent = 'ノつ[X-·]！ water/房间 麦田'
    assert e.locate_parent_bytes(parent, 'water/房间麦田') == 'water/房间 麦田'
    assert e.locate_parent_bytes(parent, 'water/房间 麦田') == 'water/房间 麦田'
    assert e.locate_parent_bytes(parent, ' water /房间') == 'water/房间'
    assert e.locate_parent_bytes(parent, 'music/前面画面') is None
    assert e.locate_parent_bytes('ab ab', 'ab') is None  # ambiguous occurrence
    assert e.locate_parent_bytes(parent, '') is None and e.locate_parent_bytes('', 'x') is None and e.locate_parent_bytes(parent, '   ') is None


def test_body_preserving_splice_uses_ink_box_when_present():
    body_mask = np.zeros((40, 300), dtype=bool)
    body_mask[10:30, 150:290] = True
    before = [{'bbox': [0, 0, 300, 40], 'text': 'LEFT  (Д_Д)  RIGHT', 'score': .4}]
    splice = {'original_text_span': [6, 11], 'baseline_index': 0, 'ordinary_prefix_suffix_preserved': True,
              'preserved_prefix': 'LEFT  ', 'preserved_suffix': '  RIGHT', 'replacement_text': '(ω_ω)'}
    after = [{'bbox': [0, 0, 300, 40], 'text': 'LEFT  (ω_ω)  RIGHT', 'score': .4, 'source': 'baseline-with-regional-splice',
              'baseline_index': 0, 'regional_bbox': [60, 5, 155, 35], 'regional_ink_bbox': [63, 8, 148, 32],
              'regional_text': '(ω_ω)', 'splice': splice}]
    assert e.body_preserving_splices(before, after, body_mask) == (True, 1)
    padded_only = [dict(after[0])]
    del padded_only[0]['regional_ink_bbox']
    assert e.body_preserving_splices(before, padded_only, body_mask) == (False, 0)
