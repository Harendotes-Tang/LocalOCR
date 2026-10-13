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


# --- v2: width wrapping, folded columns, cut faces and fragments (improve-region-candidate-localization) ---

@pytest.fixture
def dataset_v2(tmp_path, monkeypatch):
    if not FONTS.exists():
        pytest.skip('Pinned local OFL fonts absent')
    profile = rp.RoundProfile(name='unit-cards-v2', change='unit', root=tmp_path / 'unit-cards-v2', maximum_bytes=2 * 1024 ** 3,
                              owned_bytes=900 * 1024 ** 2, previous_budget=tmp_path / 'old.json', previous_sha256='0' * 64,
                              previous_used_seconds=0.0, maximum_seconds=0.0, specialist_families=1)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    root = profile.root / 'scenes' / 'fixture-v2'
    cards.generate(root, FONTS, (8, 8), seed=72, version=2)
    return root


def test_v2_plans_cut_index_and_fragments_are_declared():
    atoms = kg.default_atoms()
    assert set(cards.COMMENT_PLANS) < set(cards.COMMENT_PLANS_V2) and set(cards.COMMENT_PLANS_V2) == set(cards.COMMENT_WEIGHTS_V2)
    assert set(cards.NEGATIVE_KINDS_V2) <= set(cards.COMMENT_PLANS_V2)
    assert all('face' not in {k for k, _ in cards.COMMENT_PLANS_V2[n]} for n in cards.NEGATIVE_KINDS_V2)
    assert cards.STYLES_V2['dev']['recipe'] != cards.STYLES['dev']['recipe'] and cards.family('dev', 2) != cards.family('dev')
    assert not any(cards.contains_face(fragment, atoms) for fragment in cards.FRAGMENTS)
    import random
    rng = random.Random(3)
    for text in ('(´･ω･`)', 'ヽ(´▽`)/', '(╯°□°)╯︵ ┻━┻', '「ω_ω」'):
        for _ in range(20):
            cut = cards.split_face_index(text, atoms, rng)
            assert kg.parse_face(text[:cut], atoms) is None and kg.parse_face(text[cut:], atoms) is None
    with pytest.raises(cards.Unsupported):
        cards.split_face_index('no brackets', atoms, rng)


def test_v2_layouts_are_actual_validated_and_dispatched(dataset_v2):
    binding = old.validate(dataset_v2)
    assert binding == cards.validate(dataset_v2) == cards.read(dataset_v2 / 'freeze.json')
    meta = cards.read(dataset_v2 / 'dataset.json')
    assert meta['generator_id'] == cards.GENERATOR_ID_V2 and 'wrap-face-split' in meta['comment_plans']
    rows = cards.read(dataset_v2 / 'dev.json') + cards.read(dataset_v2 / 'heldout.json')
    assert all(row['recipe_family'] == cards.family(row['split'], 2) for row in rows)
    unsupported = [r for row in rows for r in row['regions'] if r['kind'] == 'unsupported']
    cut_faces = {(row['id'], r['source_face']) for row in rows for r in row['regions'] if r['kind'] == 'unsupported'}
    assert unsupported and len(unsupported) == 2 * len(cut_faces)
    for region in unsupported:
        assert region['part'] in ('left', 'right') and kg.parse_face(region['text']) is None and kg.parse_face(region['source_face']) is not None
    for row in rows:
        parts = [r for r in row['regions'] if r['kind'] == 'unsupported']
        for left in (p for p in parts if p['part'] == 'left'):
            right = next(p for p in parts if p['part'] == 'right' and p['source_face'] == left['source_face'])
            assert left['text'] + right['text'] == left['source_face']
            assert right['bbox'][1] > left['bbox'][3] - 4 and right['bbox'][0] < left['bbox'][0]  # second line, further left
            assert row['text'][left['span'][1]:right['span'][0]] == '\n'
    folded = [r for row in rows for r in row['regions'] if r.get('column')]
    assert folded and {r['column'] for r in folded} == {'duty', 'stamp'}
    for row in rows:
        lefts = [r for r in row['regions'] if r.get('column') and r['part'] == 'left']
        for left in lefts:
            right = next(r for r in row['regions'] if r.get('column') == left['column'] and r['part'] == 'right' and r['span'][0] > left['span'][1])
            assert right['bbox'][1] > left['bbox'][1] and row['text'][left['span'][1]:right['span'][0]] == '\n'
    wrapped = [r for row in rows for r in row['regions'] if r.get('wrapped') == 'width']
    assert wrapped and all(r['kind'] == 'body' for r in wrapped)
    fragments = [r for row in rows for r in row['regions'] if r['kind'] == 'body' and r['text'] in cards.FRAGMENTS]
    assert fragments
    assert any(row['layout']['two_line_rows'] for row in rows) and binding['counts']['dev']['unsupported_faces'] + binding['counts']['heldout']['unsupported_faces'] == len(cut_faces)
    for row in rows:
        with Image.open(dataset_v2 / row['image']) as im:
            assert im.height >= 16 + cards.STYLES_V2[row['split']]['rows'] * (cards.STYLES_V2[row['split']]['height'] + cards.STYLES_V2[row['split']]['gap'])


def test_v1_dataset_rejects_unsupported_regions(dataset):
    rows = cards.read(dataset / 'dev.json')
    row = next(r for r in rows if sum(g['kind'] == 'kaomoji' for g in r['regions']) >= 2)  # keep the scene positive
    region = next(r for r in row['regions'] if r['kind'] == 'kaomoji')
    region['kind'] = 'unsupported'
    import json
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    (dataset / 'dev.json').write_text(payload, encoding='utf-8')
    # Keep the split hash consistent so the refusal comes from the region kind, not from tampering detection.
    meta = cards.read(dataset / 'dataset.json')
    meta['splits']['dev']['sha256'] = cards.sha(dataset / 'dev.json')
    (dataset / 'dataset.json').write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    (dataset / 'freeze.json').unlink()
    (dataset / 'heldout-seal.json').unlink()
    with pytest.raises(ValueError, match='Unsupported regions are v2'):
        cards.validate(dataset)
