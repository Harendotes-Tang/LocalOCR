"""Synthetic guard plumbing only; no independent dataset or OCR evidence.

Metadata/freezes use real lightweight binding checks. Historical source bindings
are fake fixtures and exposure.load_bound is stubbed; no labels, model, browser,
old inventory or heldout is opened by these tests.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.RegionKaomoji import position_data as data


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def source(root, split):
    def ref(name):
        return {'file': str(root / name), 'sha256': sha(str(root / name))}
    return {'dataset': str(root), 'split': split, 'manifest': ref(split + '.json'),
            'dataset_metadata': ref('dataset.json'), 'freeze': ref('freeze.json'), 'rows': 1}


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    root, pixel, previous = (tmp_path / name for name in ('ownership-position-20261022', 'pixel', 'complete'))
    monkeypatch.setattr(data.rp, 'OWNED_POSITION', SimpleNamespace(root=root))
    monkeypatch.setattr(data.rp, 'PIXEL_OWNED_SPANS', SimpleNamespace(root=pixel))
    monkeypatch.setattr(data.rp, 'COMPLETE_SPANS', SimpleNamespace(root=previous))
    sources = {'fixture-source.py': sha('source')}
    monkeypatch.setattr(data.quality, 'source_inventory', lambda: deepcopy(sources))
    legacy = source(tmp_path / 'historical-authorized-dev', 'dev')
    old = {'schema_version': data.exposure.SCHEMA, 'sources': [legacy]}
    old_path = pixel / 'exposure-exclusions.json'
    save(old_path, old)
    monkeypatch.setattr(data, 'PREVIOUS_EXPOSURE_SHA256', data.common.digest(old_path))
    added = [source(pixel / 'scenes/browser-cards-v4', 'dev'),
             source(pixel / 'data/browser-crops-v4', 'train'), source(pixel / 'data/browser-crops-v4', 'dev'),
             source(previous / 'scenes/browser-cards-v3', 'dev')]
    excluded_path = root / 'exposure-exclusions.json'
    current = {'schema_version': data.exposure.SCHEMA, 'sources': [legacy, *added]}
    save(excluded_path, current)
    crop, scene = root / 'data/browser-crops-v5', root / 'scenes/browser-cards-v5'
    binding = {'file': str(excluded_path), 'sha256': data.common.digest(excluded_path),
               'target_text_scope': data.exposure.TARGET_TEXT_SCOPE, 'core_policy': data.exposure.CORE_POLICY}
    for role, path, split_names, schema in [
        ('fresh_crop', crop, ('train', 'dev', 'heldout'), 'localocr.pretrained.dataset.v1'),
        ('fresh_scene', scene, ('dev', 'heldout'), 'localocr.region-scenes.v1')]:
        meta = {'schema_version': schema, 'generator_id': data.GENERATORS[role],
                'synthetic_only': True, 'browser_rendered': True, 'seed': data.SEED,
                'exposure_exclusions': binding,
                'splits': {split: {'file': split + '.json', 'sha256': sha(role + split)} for split in split_names}}
        save(path / 'dataset.json', meta)
    freeze_path = root / 'validation/source-inventory-before-dev.json'
    replayed = []
    monkeypatch.setattr(data.exposure, 'load_bound',
                        lambda value: replayed.append(deepcopy(value)) or SimpleNamespace(binding=value))
    def resign():
        # Deliberately rebind fixture metadata/freezes, so negative tests reach
        # the lineage guard rather than merely failing an unrelated stale SHA.
        frozen = {'schema_version': 'localocr.position-before-dev-freeze.v1',
                  'source_inventory': deepcopy(sources), 'references': [], 'datasets': {}}
        for role, path in [('fresh_crop', crop), ('fresh_scene', scene)]:
            meta = data.common.read(path / 'dataset.json')
            freeze = {'schema_version': 'localocr.pretrained.freeze.v1' if role == 'fresh_crop'
                      else 'localocr.region-scenes.freeze.v1', 'dataset_sha256': data.common.digest(path / 'dataset.json'),
                      'split_sha256': {s: v['sha256'] for s, v in meta['splits'].items()}}
            save(path / 'freeze.json', freeze)
            frozen['datasets'][role] = {'freeze': data.quality.reference(path / 'freeze.json'), 'binding': freeze}
            frozen['references'].extend(data.quality.reference(path / name) for name in ('dataset.json', 'freeze.json'))
        frozen['references'].extend(data.quality.reference(path) for path in (old_path, excluded_path))
        save(freeze_path, frozen)
    resign()
    def rebind_exposure():
        current_binding = {**binding, 'sha256': data.common.digest(excluded_path)}
        for path in (crop, scene):
            meta = data.common.read(path / 'dataset.json')
            meta['exposure_exclusions'] = current_binding
            save(path / 'dataset.json', meta)
        resign()
    return SimpleNamespace(root=root, crop=crop, scene=scene, old=old_path, excluded=excluded_path,
                           frozen=freeze_path, resign=resign, rebind=rebind_exposure, replayed=replayed)


def test_fresh_guard_binds_actual_metadata_freezes_and_authorized_exposure(fresh):
    result = data.verify_fresh_data(fresh.crop, fresh.scene)
    assert result['schema_version'] == data.SCHEMA
    assert result['before_dev'] == data.quality.reference(fresh.frozen)
    assert len(fresh.replayed) == 1
    assert result['promote'] is result['enabled'] is result['app_acceptance'] is False
    assert not list(fresh.root.rglob('heldout.json'))
    assert not list(fresh.root.rglob('*.png'))


@pytest.mark.parametrize('which', ['crop', 'scene'])
def test_old_or_sibling_paths_fail_before_any_exposure_replay(fresh, which):
    arguments = [fresh.crop, fresh.scene]
    index = 0 if which == 'crop' else 1
    arguments[index] = arguments[index].with_name('browser-crops-v4' if index == 0 else 'browser-cards-v4')
    with pytest.raises(ValueError, match='exact new v5'):
        data.verify_fresh_data(*arguments)
    assert not fresh.replayed


@pytest.mark.parametrize('field,value', [('seed', 20261020), ('seed', 20261022.),
    ('generator_id', 'copied-old-generator'), ('browser_rendered', False), ('synthetic_only', False)])
def test_wrong_generation_metadata_rejected_even_when_frozen(fresh, field, value):
    path = fresh.crop / 'dataset.json'
    meta = data.common.read(path)
    meta[field] = value
    save(path, meta)
    fresh.resign()
    with pytest.raises(ValueError, match='generator|rendering|seed'):
        data.verify_fresh_data(fresh.crop, fresh.scene)
    assert not fresh.replayed


@pytest.mark.parametrize('mutation', ['different-exposure', 'wrong-path', 'missing-reference', 'source-drift', 'dataset-drift', 'wrong-schema', 'old-inventory-drift'])
def test_before_dev_binding_and_common_exposure_cannot_be_substituted(fresh, mutation):
    frozen = data.common.read(fresh.frozen)
    if mutation in ('different-exposure', 'wrong-path'):
        meta = data.common.read(fresh.scene / 'dataset.json')
        meta['exposure_exclusions']['sha256' if mutation == 'different-exposure' else 'file'] = sha('other')
        save(fresh.scene / 'dataset.json', meta)
        fresh.resign()
    elif mutation == 'old-inventory-drift':
        save(fresh.old, {'changed': True})
        fresh.resign()
    else:
        if mutation == 'missing-reference':
            frozen['references'].pop()
        elif mutation == 'source-drift':
            frozen['source_inventory']['fixture-source.py'] = sha('changed')
        elif mutation == 'dataset-drift':
            frozen['datasets']['fresh_crop']['binding']['dataset_sha256'] = sha('changed')
        else:
            frozen['schema_version'] = 'localocr.pixel-owned-before-dev-freeze.v1'
        save(fresh.frozen, frozen)
    with pytest.raises(ValueError):
        data.verify_fresh_data(fresh.crop, fresh.scene)
    assert not fresh.replayed


@pytest.mark.parametrize('mutation', ['lost-old', 'changed-old', 'old-type-drift', 'missing-v4-dev', 'unauthorized-heldout', 'duplicate', 'missing-binding'])
def test_exposure_preserves_every_old_binding_and_only_approved_additions(fresh, mutation):
    inventory = data.common.read(fresh.excluded)
    if mutation == 'lost-old':
        inventory['sources'].pop(0)
    elif mutation == 'changed-old':
        inventory['sources'][0]['rows'] = 2
    elif mutation == 'old-type-drift':
        inventory['sources'][0]['rows'] = 1.0
    elif mutation == 'missing-v4-dev':
        inventory['sources'].pop(1)
    elif mutation == 'unauthorized-heldout':
        inventory['sources'].append(source(fresh.root / 'sealed', 'heldout'))
    elif mutation == 'duplicate':
        inventory['sources'].append(inventory['sources'][0])
    else:
        inventory['sources'][1].pop('manifest')
    save(fresh.excluded, inventory)
    fresh.rebind()
    with pytest.raises(ValueError):
        data.verify_fresh_data(fresh.crop, fresh.scene)
    assert not fresh.replayed


@pytest.mark.parametrize('which', ['crop', 'scene'])
def test_copied_seen_split_digest_fails_despite_new_root_seed_and_valid_freeze(fresh, which):
    inventory = data.common.read(fresh.excluded)
    path = getattr(fresh, which) / 'dataset.json'
    meta = data.common.read(path)
    meta['splits']['dev']['sha256'] = inventory['sources'][1]['manifest']['sha256']
    save(path, meta)
    fresh.resign()
    with pytest.raises(ValueError, match='copies a previously exposed'):
        data.verify_fresh_data(fresh.crop, fresh.scene)
    assert not fresh.replayed


def test_exposure_replay_or_source_mutation_cannot_return_a_stale_receipt(fresh, monkeypatch):
    def changed(binding):
        save(fresh.excluded, {'changed': True})
        return SimpleNamespace(binding=binding)
    monkeypatch.setattr(data.exposure, 'load_bound', changed)
    with pytest.raises(ValueError, match='changed during verification'):
        data.verify_fresh_data(fresh.crop, fresh.scene)


def test_missing_metadata_is_an_explicit_guard_rejection(fresh):
    # This only removes one generated temporary fixture file, never workspace data.
    (fresh.crop / 'dataset.json').unlink()
    with pytest.raises(ValueError, match='Unreadable or malformed'):
        data.verify_fresh_data(fresh.crop, fresh.scene)
    assert not fresh.replayed


def test_original_consumed_heldout_binding_is_preserved_without_new_authorization(fresh, monkeypatch):
    old = data.common.read(fresh.old)
    consumed = source(fresh.root.parent / 'historical-consumed', 'heldout')
    consumed['first_use'] = {'file': str(fresh.root.parent / 'historical-first-use.json'), 'sha256': sha('receipt')}
    old['sources'].append(consumed)
    save(fresh.old, old)
    monkeypatch.setattr(data, 'PREVIOUS_EXPOSURE_SHA256', data.common.digest(fresh.old))
    current = data.common.read(fresh.excluded)
    current['sources'].append(deepcopy(consumed))
    save(fresh.excluded, current)
    fresh.rebind()
    assert data.verify_fresh_data(fresh.crop, fresh.scene)['schema_version'] == data.SCHEMA
    assert fresh.replayed
