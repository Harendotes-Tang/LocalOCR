"""Browser real-font card scenes: width rewriting, comment planning, page markup, exact masks, offline rendering."""
import dataclasses
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.OcrDatasets.render import Unsupported
from tools.PretrainedOcr import kaomoji_grammar as kg, grammar_corpus as gc, round_profile as rp
from tools.RegionKaomoji import browser_scenes as bs, card_scenes as cs, scenes

REPO = Path(__file__).resolve().parents[1]
FONTS = REPO / 'artifacts/ocr-data-fonts-v1/fonts.json'
ATOMS = kg.default_atoms()


def _edge_available():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe').exists()


needs_edge = pytest.mark.skipif(not _edge_available(), reason='Playwright or the local Edge is absent')


def test_width_families_are_the_declared_paren_and_square_families():
    families = bs.width_families(ATOMS)
    assert {pair for pair in families} == {('(', ')'), ('（', '）'), ('(', '）'), ('（', ')'), ('[', ']'), ('［', '］'), ('[', '］'), ('［', ']')}
    assert families[('(', ')')]['fullwidth'] == [('（', '）')] and sorted(families[('(', ')')]['mixed']) == [('(', '）'), ('（', ')')]
    assert ('「', '」') not in families and ('｢', '｣') not in families


@pytest.mark.parametrize('target', ['ascii', 'fullwidth', 'mixed'])
def test_rewrite_width_keeps_the_face_and_reports_its_class(target):
    families, rng = bs.width_families(ATOMS), random.Random(3)
    text = bs.rewrite_width('(^_^)', ATOMS, families, target, rng)
    assert kg.parse_face(text, ATOMS) is not None and text[1:-1] == '^_^'
    assert bs.width_class(text, ATOMS, families) == target


def test_non_family_brackets_are_never_rewritten():
    families = bs.width_families(ATOMS)
    assert bs.rewrite_width('【^_^】', ATOMS, families, 'fullwidth', random.Random(1)) == '【^_^】'
    assert bs.width_class('【^_^】', ATOMS, families) == 'none'


def test_paste_faces_parse_and_are_new():
    assert len(set(bs.PASTE_FACES)) == len(bs.PASTE_FACES)
    assert all(kg.parse_face(text, ATOMS) is not None for text in bs.PASTE_FACES)
    assert not set(bs.PASTE_FACES) & set(bs.COMMON_FACES)


def test_common_pool_variants_own_their_core_partition():
    pool = bs.common_pool(ATOMS, lambda text: True)
    families = bs.width_families(ATOMS)
    for split, faces in pool.items():
        assert faces
        for face in faces:
            assert face['core_partition'] == split == gc.partition(face['core_key'])
            assert bs.width_class(face['text'], ATOMS, families) == face['bracket_width_class']
    assert {f['bracket_width_class'] for f in pool['heldout']} >= {'ascii', 'fullwidth', 'mixed'}


def test_face_picker_samples_widths_before_the_partition():
    allowed = gc.allowed_atoms(ATOMS, {atom for role in kg.LIST_ROLES if role != 'gaps' for atom in getattr(ATOMS, role)} | set(ATOMS.bracket_chars))
    picker = bs.FacePicker('heldout', random.Random(5), ATOMS, allowed, bs.common_pool(ATOMS, lambda text: True), set())
    faces = [picker() for _ in range(300)]
    assert all(face['core_partition'] == 'heldout' == gc.partition(face['core_key']) for face in faces)
    classes = {name: sum(face['bracket_width_class'] == name for face in faces) for name in ('ascii', 'fullwidth', 'mixed', 'none')}
    assert classes['ascii'] > 60 and classes['fullwidth'] > 60 and classes['mixed'] > 5
    assert {face['source'] for face in faces} >= {'atom-table-heldout-partition', 'common-structure'}


def test_comment_plans_never_wrap_and_bodies_carry_no_face():
    assert all(sep not in ('\n', 'long', 'split') for plan in bs.PLANS.values() for _, sep in plan)
    rng, seen = random.Random(9), set()
    faces = lambda: {'text': '(^_^)', 'core_key': None}
    for negative in (False, True):
        for _ in range(200):
            kind, segments = bs.plan_comment(rng, faces, ATOMS, seen, negative)
            assert not negative or kind in bs.NEGATIVE_PLANS
            for segment_kind, payload, separator in segments:
                assert separator in ('', ' ')
                if segment_kind == 'body':
                    assert not cs.contains_face(payload, ATOMS)


def _rows(seed=2, count=3):
    rng, seen = random.Random(seed), set()
    picker = bs.FacePicker('dev', rng, ATOMS, gc.allowed_atoms(ATOMS, {a for r in kg.LIST_ROLES if r != 'gaps' for a in getattr(ATOMS, r)} | set(ATOMS.bracket_chars)),
                           bs.common_pool(ATOMS, lambda text: True), seen)
    return [bs.plan_row(rng, picker, ATOMS, seen, False) for _ in range(count)]


def test_page_regions_spans_and_single_text_node_comment_markup():
    rows = _rows()
    text, regions = bs.page_regions(rows)
    for region in regions:
        if region['kind'] != 'graphic':
            a, b = region['span']
            assert text[a:b] == region['text']
    style = {'font_size': 18, 'name_weight': 700}
    page = bs.page_html(style, rows)
    assert page == bs.page_html(style, rows)
    assert 'http' not in page and '<span data-r' not in page.split('class="comment"')[1].split('</div>')[0]
    assert page.count('data-r=') + sum(len(row['segments']) for row in rows) == len(regions)


def test_cross_split_face_core_or_comment_reuse_is_rejected():
    face = {'kind': 'kaomoji', 'text': '(^_^)', 'core_digest': 'c' * 64}
    body = {'kind': 'body', 'text': '今天的野队意外地顺利', 'column': 'comment'}
    def row(split, text, regions):
        return {'split': split, 'source_split': split, 'recipe_family': bs.family(split), 'scene_kind': 'card-list',
                'provenance': {'source': 'original', 'license': 'CC0-1.0'}, 'text': text, 'text_sha256': bs.th(text),
                'pixel_sha256': bs.th(split + text), 'regions': regions}
    for shared in (face, body):
        owners, pixelsets = {}, set()
        bs._check_row_identity(row('dev', 'a', [face if shared is face else dict(face, text='(T_T)', core_digest='d' * 64), shared if shared is body else body]), 'dev', owners, pixelsets)
        other = [shared, dict(face, text='(o_o)', core_digest='e' * 64)] if shared is body else [shared]
        with pytest.raises(ValueError, match='Cross-split leakage'):
            bs._check_row_identity(row('heldout', 'b', other), 'heldout', owners, pixelsets)


def test_utf16_offsets_count_astral_characters_twice():
    assert bs._utf16('a𐐘b') == 4 and bs._utf16('（・ω・）') == 5


def _comment_regions():
    return [{'kind': 'body', 'column': 'comment', 'row': 0, 'segment': 0, 'text': 'ab'},
            {'kind': 'kaomoji', 'column': 'comment', 'row': 0, 'segment': 1, 'text': '(^_^)'},
            {'kind': 'body', 'column': 'stamp', 'row': 0, 'text': 'x'}]


def test_shared_ink_of_a_face_and_its_neighbour_goes_to_the_face():
    masks = [np.zeros((4, 10), bool) for _ in range(3)]
    masks[0][1, 0:5] = True
    masks[1][1, 4:9] = True
    masks[2][3, 9] = True
    exclusive, contact = bs.resolve_overlaps(masks, _comment_regions())
    assert exclusive[1][1, 4] and not exclusive[0][1, 4] and contact == {1: 1}
    masks[2][1, 2] = True
    with pytest.raises(Unsupported, match='region-overlap'):
        bs.resolve_overlaps(masks, _comment_regions())


def test_clipped_overhang_ink_goes_to_a_face_within_reach_or_fails():
    masks = [np.zeros((6, 20), bool) for _ in range(3)]
    masks[0][2, 2:6] = True
    masks[1][2, 9:14] = True
    masks[2][5, 19] = True
    ink = np.any(masks, axis=0)
    ink[2, 7] = True                                   # overhang between the body and the face
    completed, contact = bs.assign_clipped(ink, masks, _comment_regions())
    assert completed[1][2, 7] and contact == {1: 1}
    ink[0, 18] = True                                  # ink near no comment segment
    with pytest.raises(Unsupported, match='mask-union'):
        bs.assign_clipped(ink, masks, _comment_regions())


def test_clipped_reach_is_measured_on_the_captured_masks_and_self_overhang_is_not_contact():
    regions = [{'kind': 'kaomoji', 'column': 'comment', 'row': 0, 'segment': 0, 'text': '(o_O)'},
               {'kind': 'body', 'column': 'stamp', 'row': 0, 'text': 'x'}]
    masks = [np.zeros((6, 30), bool) for _ in range(2)]
    masks[0][2, 2:8] = True
    masks[1][5, 29] = True
    ink = np.any(masks, axis=0)
    ink[3, 9] = True                                   # the face's own overhang, no neighbour in reach
    completed, contact = bs.assign_clipped(ink, masks, regions)
    assert completed[0][3, 9] and contact == {}
    ink[2, 9:16] = True                                # a run reaching past the radius must not chain outward
    with pytest.raises(Unsupported, match='mask-union'):
        bs.assign_clipped(ink, masks, regions)


def test_failed_launch_restores_temp_and_removes_the_profile(tmp_path, monkeypatch):
    pytest.importorskip('playwright')
    import os
    before = (os.environ.get('TEMP'), os.environ.get('TMP'))
    monkeypatch.setitem(bs.DRIVER, 'channel', 'no-such-channel')
    temp = tmp_path / 'browser-temp'
    with pytest.raises(Exception):
        with bs.BrowserSession(temp):
            pass
    assert (os.environ.get('TEMP'), os.environ.get('TMP')) == before and not temp.exists()


@needs_edge
def test_session_blocks_and_fails_on_non_local_requests(tmp_path):
    page = tmp_path / 'page.html'
    page.write_text('<html><body><img src="https://example.invalid/x.png">x</body></html>', encoding='utf-8')
    temp = tmp_path / 'browser-temp'
    with bs.BrowserSession(temp) as session:
        context = session.context(1.0, (200, 100))
        tab = context.new_page()
        tab.goto(page.as_uri(), wait_until='load')
        context.close()
        assert session.blocked
        with pytest.raises(RuntimeError, match='Non-local'):
            session.check()
    assert not temp.exists()


@needs_edge
def test_session_reports_platform_fonts_for_regions(tmp_path):
    rows = _rows(count=2)
    style = {'device_scale_factor': 1.25, 'viewport': [1488, 785], 'rows': 2, 'font_size': 18, 'name_weight': 700}
    path = tmp_path / 'scene.html'
    path.write_text(bs.page_html(style, rows), encoding='utf-8')
    with bs.BrowserSession(tmp_path / 'browser-temp') as session:
        page = bs.ScenePage(session, style, path)
        try:
            assert page.capture().shape == (981, 1860, 3)
            fonts = page.platform_fonts(1)
        finally:
            page.close()
        session.check()
    assert any(font['family'] == 'Microsoft YaHei' and font['glyphs'] > 0 for font in fonts)


@needs_edge
def test_an_overflowing_comment_is_replanned_not_wrapped(tmp_path):
    template = _rows(count=1)[0]
    long_row = dict(template, segments=[('body', '这是一条很长很长的评论' * 30, '')])
    plans = iter([long_row, template])
    style = {'device_scale_factor': 1.0, 'viewport': [1860, 981], 'rows': 1, 'font_size': 18, 'name_weight': 700}
    with bs.BrowserSession(tmp_path / 'browser-temp') as session:
        rows = bs.fitting_rows(session, style, lambda index: next(plans), tmp_path / 'browser-temp' / 'attempt.html')
        session.check()
    assert rows == [template]


@pytest.fixture(scope='module')
def tiny_dataset(tmp_path_factory):
    if not (_edge_available() and FONTS.exists()):
        pytest.skip('Playwright, Edge or the pinned OFL fonts are absent')
    base = tmp_path_factory.mktemp('browser-round')
    if not base.resolve().is_relative_to((REPO / 'artifacts').resolve()):
        pytest.skip('Scene generation must run with --basetemp under artifacts/')
    profile = dataclasses.replace(rp.BROWSER_RENDERING, name='unit-browser', root=base / 'unit-round', previous_budget=base / 'old.json')
    saved = rp.ROUNDS
    rp.ROUNDS = (profile,)
    try:
        root = profile.root / 'scenes' / 'browser-unit'
        binding = bs.generate(root, FONTS, (5, 5), seed=11, rows_override=2)
    finally:
        rp.ROUNDS = saved
    return root, binding


def test_tiny_dataset_validates_through_the_shared_dispatch(tiny_dataset):
    root, binding = tiny_dataset
    assert scenes.validate(root) == binding
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    assert meta['environment']['browser']['blocked_requests'] == 0 and meta['environment']['fonts_distributed'] is False
    assert all(meta['environment']['platform_fonts'][family] for family in meta['environment']['platform_fonts'])
    assert not (root / 'browser-temp').exists()
    rows = json.loads((root / 'dev.json').read_text(encoding='utf-8'))
    assert any(row['scene_kind'] == 'card-list-negative' for row in rows)
    assert all('\n' not in region['text'] for row in rows for region in row['regions'] if region['kind'] != 'graphic')


def test_tampered_delivery_or_mask_fails_validation(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    copy = tmp_path / 'copy'
    shutil.copytree(root, copy)
    rows = json.loads((copy / 'dev.json').read_text(encoding='utf-8'))
    mask = copy / rows[0]['regions'][1]['mask']
    with Image.open(mask) as image:
        array = np.asarray(image).copy()
    array[0, 0] = 255 - array[0, 0]
    Image.fromarray(array).save(mask)
    with pytest.raises(ValueError):
        bs.validate(copy)


def test_face_core_key_is_rederived_from_the_text(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    copy = tmp_path / 'core'
    shutil.copytree(root, copy)
    (copy / 'freeze.json').unlink()
    (copy / 'heldout-seal.json').unlink()
    rows = json.loads((copy / 'dev.json').read_text(encoding='utf-8'))
    face = next(region for row in rows for region in row['regions'] if region['kind'] == 'kaomoji')
    face['core_key']['bracket'] = ['{', '}']               # a key that no longer matches the face text
    face['core_digest'] = gc.core_digest(face['core_key'])
    (copy / 'dev.json').write_text(json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    meta = json.loads((copy / 'dataset.json').read_text(encoding='utf-8'))
    meta['splits']['dev']['sha256'] = scenes.sha(copy / 'dev.json')
    (copy / 'dataset.json').write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='core partition or grammar identity'):
        bs.validate(copy)
