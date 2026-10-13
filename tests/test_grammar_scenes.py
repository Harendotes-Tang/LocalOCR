"""Grammar-round scenes: partitioned faces, declared ink effects, exact masks and dispatch."""
from pathlib import Path

import numpy as np
import pytest

from tools.RegionKaomoji import grammar_scenes as scenes, scenes as old
from tools.PretrainedOcr import grammar_corpus as gc, kaomoji_grammar as kg, quality_data as qd, round_profile as rp

FONTS = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'


def test_layout_asts_are_actual_and_disjoint_and_extend_previous_round():
    assert set(scenes.RECIPES['dev']) == set(scenes.RECIPES['heldout']) == set(scenes.TYPES)
    assert not set(scenes.RECIPES['dev'].values()) & set(scenes.RECIPES['heldout'].values())
    for split, recipes in scenes.RECIPES.items():
        for kind, expression in recipes.items():
            assert scenes.family(split, kind) == 'grammar-scene-grammar:' + scenes.th(expression)
    assert set(scenes.NEW_TYPES) <= set(scenes.TYPES) and 'wide-gap-body' in scenes.TYPES and 'adjacent-faces' in scenes.TYPES


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    if not FONTS.exists():
        pytest.skip('Pinned local OFL fonts absent')
    profile = rp.RoundProfile(name='unit-grammar', change='unit', root=tmp_path / 'unit-grammar', maximum_bytes=2 * 1024 ** 3,
                              owned_bytes=900 * 1024 ** 2, previous_budget=tmp_path / 'old.json', previous_sha256='0' * 64,
                              previous_used_seconds=1.0, specialist_families=1)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    root = profile.root / 'scenes' / 'fixture'
    scenes.generate(root, FONTS, (len(scenes.TYPES), len(scenes.TYPES)), seed=61)
    return root


def test_generation_requires_registered_round(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'ROUNDS', ())
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    with pytest.raises(ValueError, match='registered research round'):
        scenes.generate(tmp_path / 'anywhere' / 'scenes', FONTS, (2, 2))


def test_real_masks_partitions_effects_and_dispatch(dataset):
    binding = old.validate(dataset)
    assert binding == scenes.validate(dataset)
    meta = scenes.read(dataset / 'dataset.json')
    assert meta['generator_id'] == scenes.GENERATOR_ID and meta['atoms_binding'] == kg.binding()
    for split in ('dev', 'heldout'):
        rows = scenes.read(dataset / (split + '.json'))
        kinds = {row['scene_kind'] for row in rows}
        assert set(scenes.TYPES) == kinds
        faces = [region for row in rows for region in row['regions'] if region['kind'] == 'kaomoji']
        assert faces and all(gc.partition(region['core_key']) == split for region in faces)
        assert all(kg.parse_face(region['text']) is not None for region in faces)
        effects = [region['rendering']['effects'] for row in rows for region in row['regions'] if region['kind'] != 'graphic']
        assert any(e.get('outline') for e in effects) and any(e.get('shadow') for e in effects)
        wide = [region for row in rows if row['scene_kind'] == 'wide-gap-body' for region in row['regions'] if region['kind'] == 'body']
        assert wide and all('    ' in region['text'] for region in wide)
        adjacent = [row for row in rows if row['scene_kind'] == 'adjacent-faces']
        assert adjacent and sum(region['kind'] == 'kaomoji' for region in adjacent[0]['regions']) == 2
        corner = [region for row in rows if row['scene_kind'] == 'corner-face' for region in row['regions'] if region['kind'] == 'kaomoji']
        assert corner and all(region['text'].lstrip('つノ╯╰ヽヾ☞m∩ヘ┌╭＼\\⊂ლ٩ง')[0] in '「『｢' for region in corner)
        for row in rows:
            masks = [np.asarray(scenes.Image.open(dataset / region['mask'])) > 0 for region in row['regions']]
            image = np.asarray(scenes.Image.open(dataset / row['image']).convert('RGB'))
            if row['scene_kind'] != 'blank':
                assert np.array_equal(np.any(masks, axis=0), np.any(image != np.array(scenes.BG), axis=2))
    inputs = scenes.read(dataset / 'inference-inputs.json')
    assert set(inputs['jobs']['dev'][0]) == set(scenes.JOB_KEYS)


def test_core_partition_and_span_tampering_rejected(dataset):
    path = dataset / 'dev.json'
    rows = scenes.read(path)
    face = next(region for row in rows for region in row['regions'] if region['kind'] == 'kaomoji')
    face['core_partition'] = 'heldout'
    old.write(path, rows)
    meta = scenes.read(dataset / 'dataset.json')
    meta['splits']['dev']['sha256'] = scenes.sha(path)
    old.write(dataset / 'dataset.json', meta)
    with pytest.raises(ValueError, match='core partition'):
        scenes.validate(dataset)
