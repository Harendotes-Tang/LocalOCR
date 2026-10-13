"""Grammar-round crop generation: round-scoped quota, effects, isolation and tamper rejection."""
import dataclasses
from pathlib import Path

import pytest

from tools.PretrainedOcr import grammar_corpus as gc, grammar_data as data, kaomoji_grammar as kg, quality_data as qd
from tools.PretrainedOcr import round_profile as rp

FONTS = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'


def unit_round(tmp_path, monkeypatch):
    profile = rp.RoundProfile(name='unit-grammar', change='unit', root=tmp_path / 'unit-grammar', maximum_bytes=2 * 1024 ** 3,
                              owned_bytes=900 * 1024 ** 2, previous_budget=tmp_path / 'old.json', previous_sha256='0' * 64,
                              previous_used_seconds=1.0, specialist_families=1)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    return profile


@pytest.fixture
def generated(tmp_path, monkeypatch):
    if not FONTS.exists():
        pytest.skip('Pinned local OFL fonts absent')
    profile = unit_round(tmp_path, monkeypatch)
    root = profile.root / 'data' / 'fixture'
    data.generate(root, FONTS, (20, 20, 20), (2, 2, 2), seed=77)
    return root


def update(root, split, fn):
    path = root / (split + '.json')
    rows = data.read(path)
    fn(rows[0])
    qd.write(path, rows)
    meta = data.read(root / 'dataset.json')
    meta['splits'][split]['sha256'] = data.sha(path)
    qd.write(root / 'dataset.json', meta)


def test_generation_requires_registered_round(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'ROUNDS', ())
    monkeypatch.setattr(qd, 'ARTIFACTS', tmp_path)
    with pytest.raises(ValueError, match='registered research round'):
        data.generate(tmp_path / 'anywhere' / 'data', FONTS, (2, 2, 2), (1, 1, 1))


def test_effect_draws_follow_declared_split_sets():
    import random
    rng = random.Random(3)
    draws = [data.draw_effects(split, rng) for split in ('train', 'dev', 'heldout') for _ in range(60)]
    assert {d['effect_set'] for d in draws[:120]} == {'A'} and {d['effect_set'] for d in draws[120:]} == {'B'}
    assert any('outline' in d['effects'] for d in draws) and any('shadow' in d['effects'] for d in draws)
    assert any('texture' in d['effects'] for d in draws) and any(d['jpeg'] for d in draws) and any(d['blur'] for d in draws)
    assert all(d['effects'].get('outline', 1) == 1 for d in draws[:120]) and all(d['effects'].get('outline', 2) == 2 for d in draws[120:])
    assert all(d['effects']['background'] and d['effects']['ink'] for d in draws)


def test_actual_generation_counts_isolation_effects_and_adapter_read(generated):
    from tools.PretrainedOcr.common import read_split, train_characters
    binding = data.validate(generated)
    assert binding['counts']['train'] == {'count': 40, 'positive_groups': 18, 'groups': 20}
    meta = data.read(generated / 'dataset.json')
    assert meta['generator_id'] == data.GENERATOR_ID and meta['atoms_binding'] == kg.binding()
    assert meta['isolation']['core_combination'].startswith('disjoint') and meta['split_effect_sets'] == data.SPLIT_EFFECT_SETS
    assert meta['excluded_atoms'] and all(e['reason'] for e in meta['excluded_atoms'])
    assert train_characters(generated) == meta['train_vocabulary']
    rows = {split: data.read(generated / (split + '.json')) for split in ('train', 'dev', 'heldout')}
    cores = {s: {r['core_digest'] for r in rows[s] if r['kind'] == 'face'} for s in rows}
    assert not cores['train'] & cores['heldout'] and not cores['dev'] & cores['heldout']
    assert {r['transform']['effect_set'] for r in rows['train']} == {'A'} and {r['transform']['effect_set'] for r in rows['heldout']} == {'B'}
    assert any(r['transform']['effects'].get('outline') or r['transform']['effects'].get('shadow') for r in rows['train'] + rows['heldout'])
    assert len(read_split(generated, 'train')) == 40
    with pytest.raises(ValueError, match='Heldout'):
        read_split(generated, 'heldout')
    assert data.read(generated / 'heldout-seal.json')['results_seen'] is False
    assert 'brackets' in data.read(generated / 'train-length-statistics.json')


def test_core_partition_and_structure_tampering_rejected(generated):
    update(generated, 'dev', lambda row: row.update(core_partition='train'))
    with pytest.raises(ValueError, match='core partition|structure'):
        data.validate(generated)


def test_effect_set_tampering_rejected(generated):
    def swap(row):
        row['transform'] = dict(row['transform'], effect_set='B')
    update(generated, 'train', swap)
    with pytest.raises(ValueError, match='Effect set'):
        data.validate(generated)


def test_label_and_image_tampering_rejected(generated):
    update(generated, 'train', lambda row: row.update(text=row['text'] + ' '))
    with pytest.raises(ValueError, match='Unicode|structure'):
        data.validate(generated)


def test_profile_quota_labels_and_root_scope(tmp_path, monkeypatch):
    profile = unit_round(tmp_path, monkeypatch)
    profile = dataclasses.replace(profile, maximum_bytes=2 * 1024 ** 3, owned_bytes=1024 ** 2)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(qd.shutil, 'disk_usage', lambda _: type('D', (), {'free': 9 * 1024 ** 3})())
    target = profile.root / 'data' / 'x'
    target.mkdir(parents=True)
    (target / 'payload').write_bytes(b'x' * (1024 ** 2 + 1))
    assert qd.quota_root(target) == profile.root.resolve()
    with pytest.raises(ValueError, match='1MiB'):
        qd.storage_gate(target)
    monkeypatch.setattr(qd, 'OWNED_BYTE_LIMIT', 1)
    legacy = tmp_path / 'legacy-root'
    legacy.mkdir()
    (legacy / 'payload').write_bytes(b'xx')
    with pytest.raises(ValueError, match='350MiB'):
        qd.storage_gate(legacy)
