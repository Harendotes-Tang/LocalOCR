"""Record-card scenes: real card layout, inline/attached/wrapped faces, exact multi-colour masks and dispatch."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.RegionKaomoji import card_scenes as cards, scenes as old
from tools.PretrainedOcr import grammar_corpus as gc, kaomoji_grammar as kg, quality_data as qd, round_profile as rp

FONTS = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'


def test_styles_and_plans_are_actual_and_split_disjoint():
    assert set(cards.STYLES) == {'dev', 'heldout'}
    assert cards.STYLES['dev']['recipe'] != cards.STYLES['heldout']['recipe']
    assert not set(map(tuple, cards.STYLES['dev']['cards'])) & set(map(tuple, cards.STYLES['heldout']['cards']))
    for split in cards.STYLES:
        assert cards.family(split) == 'card-scene-grammar:' + cards.th(cards.STYLES[split]['recipe'])
    assert set(cards.COMMENT_PLANS) == set(cards.COMMENT_WEIGHTS)
    assert set(cards.NEGATIVE_KINDS) <= set(cards.COMMENT_PLANS) and all('face' not in {k for k, _ in cards.COMMENT_PLANS[n]} for n in cards.NEGATIVE_KINDS)
    for plan in cards.COMMENT_PLANS.values():
        assert all(kind in ('body', 'face') and separator in ('', ' ', '\n', '?') for kind, separator in plan)
        assert not plan or plan[0][1] == ''
    assert any('\n' in {sep for _, sep in plan} for plan in cards.COMMENT_PLANS.values())


def test_body_text_never_carries_a_grammatical_face():
    atoms = kg.default_atoms()
    assert cards.contains_face('看看(^_^)吧', atoms)
    assert cards.contains_face('(;_;)', atoms)
    assert not cards.contains_face('（目移）', atoms)
    assert not cards.contains_face('plain 030 .jpg', atoms)
    assert not cards.contains_face('', atoms)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    if not FONTS.exists():
        pytest.skip('Pinned local OFL fonts absent')
    profile = rp.RoundProfile(name='unit-cards', change='unit', root=tmp_path / 'unit-cards', maximum_bytes=2 * 1024 ** 3,
                              owned_bytes=900 * 1024 ** 2, previous_budget=tmp_path / 'old.json', previous_sha256='0' * 64,
                              previous_used_seconds=1.0, specialist_families=1)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    root = profile.root / 'scenes' / 'fixture'
    cards.generate(root, FONTS, (5, 5), seed=71)
    return root


def test_generation_requires_registered_round_and_fresh_output(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'ROUNDS', ())
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    with pytest.raises(ValueError, match='registered research round'):
        cards.generate(tmp_path / 'anywhere' / 'scenes', FONTS, (2, 2))
    with pytest.raises(ValueError, match='Positive scene counts'):
        cards.generate(tmp_path / 'anywhere' / 'scenes', FONTS, (0, 2))


def test_card_layout_masks_partitions_and_dispatch(dataset):
    binding = old.validate(dataset)
    assert binding == cards.validate(dataset) == cards.read(dataset / 'freeze.json')
    meta = cards.read(dataset / 'dataset.json')
    assert meta['generator_id'] == cards.GENERATOR_ID and meta['atoms_binding'] == kg.binding()
    assert meta['synthetic_only'] is True and meta['private_inputs_consumed'] is False
    for split in ('dev', 'heldout'):
        rows = cards.read(dataset / (split + '.json'))
        assert len(rows) == 5 and all(row['scene_kind'] in cards.SCENE_KINDS and row['recipe_family'] == cards.family(split) for row in rows)
        negatives = [row for row in rows if row['scene_kind'] == 'card-list-negative']
        assert len(negatives) == 1 and not any(region['kind'] == 'kaomoji' for region in negatives[0]['regions'])
        assert all(any(region['kind'] == 'kaomoji' for region in row['regions']) for row in rows if row['scene_kind'] == 'card-list')
        assert binding['counts'][split]['negative_scenes'] == 1
        faces = [region for row in rows for region in row['regions'] if region['kind'] == 'kaomoji']
        assert faces and all(gc.partition(region['core_key']) == split for region in faces)
        assert all(kg.parse_face(region['text']) is not None for region in faces)
        graphics = [region for row in rows for region in row['regions'] if region['kind'] == 'graphic']
        assert {region['geometry'] for region in graphics} == {'gold-emblem', 'square-icon'}
        assert sum(region['geometry'] == 'gold-emblem' for region in graphics) == 5 * cards.STYLES[split]['rows']
        closes = [region for row in rows for region in row['regions'] if region['text'] == '×']
        assert len(closes) == 5 * cards.STYLES[split]['rows']
        levels = [region for row in rows for region in row['regions'] if region['text'].startswith('Lv.')]
        assert len(levels) == 5 * cards.STYLES[split]['rows']
        for row in rows:
            assert row['text'].count('\n') >= cards.STYLES[split]['rows'] - 1
            with Image.open(dataset / row['image']) as im, Image.open(dataset / row['background']) as bg:
                arr, base = np.asarray(im.convert('RGB')), np.asarray(bg.convert('RGB'))
            assert arr.shape == base.shape and len({tuple(c) for c in base.reshape(-1, 3)}) >= 3
            for region in row['regions']:
                if region['kind'] != 'graphic':
                    a, b = region['span']
                    assert row['text'][a:b] == region['text']


def test_attached_and_spaced_faces_share_the_comment_reading_order(dataset):
    rows = cards.read(dataset / 'dev.json') + cards.read(dataset / 'heldout.json')
    attached = spaced = 0
    for row in rows:
        regions = [r for r in row['regions'] if r['kind'] != 'graphic']
        for previous, current in zip(regions, regions[1:]):
            if current['kind'] != 'kaomoji' or previous['kind'] != 'body' or previous['text'] == '×':
                continue
            separator = row['text'][previous['span'][1]:current['span'][0]]
            if separator == '':
                attached += 1
                assert current['bbox'][0] >= previous['bbox'][2]
            elif separator == ' ':
                spaced += 1
                assert current['bbox'][0] > previous['bbox'][2] + 3
            else:
                assert separator in ('\n', cards.COLUMN_GAP)
    assert attached + spaced > 0


def test_validation_detects_background_and_mask_tampering(dataset):
    row = cards.read(dataset / 'dev.json')[0]
    region = next(r for r in row['regions'] if r['kind'] == 'kaomoji')
    with Image.open(dataset / region['mask']) as mask:
        altered = mask.copy()
    altered.putpixel((0, 0), 255)
    altered.save(dataset / region['mask'])
    with pytest.raises(ValueError):
        cards.validate(dataset)
