"""Round-dispatch and immutable-proof plumbing; fixtures are not CPU quality."""
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.PretrainedOcr import quality_evaluate as q, round_profile as rp
from tools.RegionKaomoji import joint_quality


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return q.reference(path)


def fixture(tmp_path, monkeypatch, *, passed=True, matching=True):
    profile = dataclasses.replace(rp.PIXEL_OWNED_SPANS, root=tmp_path / 'round')
    monkeypatch.setattr(rp, 'PIXEL_OWNED_SPANS', profile)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(q, 'source_inventory', lambda: {'fixture.py': 'a' * 64})
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    manifest = save(profile.root / 'routing/configuration-manifest.json',
                    {'frozen_utc': now.isoformat()})
    frozen = save(profile.root / 'validation/source-inventory-before-dev.json', {
        'schema_version': 'localocr.pixel-owned-before-dev-freeze.v1',
        'source_inventory': q.source_inventory(), 'references': [manifest],
        'frozen_utc': (now + timedelta(seconds=1)).isoformat()})
    scored = {name: {pop: save(profile.root / name / pop / 'evaluation.json',
        {'completed_utc': (now + timedelta(seconds=2)).isoformat()})
        for pop in ('anchor', 'fresh')} for name in joint_quality.TIERS}
    request = save(profile.root / 'evaluation/joint-development-inputs.json', {
        'schema_version': 'localocr.pixel-owned-development-inputs.v1',
        'selected_id': 'tiny', 'configuration_manifest': manifest,
        'source_freeze': frozen, 'scored_reports': scored, 'repeated_reports': {},
        'passed': True})  # Caller verdict deliberately has no authority.
    selected = scored['joint']['fresh']
    selection = save(profile.root / 'evaluation/scene-selection.json', {
        'selected_id': 'tiny', 'dev_report': selected})
    proof_ref = selected if matching else scored['local-exclusive']['fresh']
    seen = []
    def verify(scored_paths, repeated_paths, selected_id, **kwargs):
        seen.append((scored_paths, repeated_paths, selected_id, kwargs))
        return {'passed': passed, 'selected_tier': 'joint' if passed else 'start',
                'reports': {'joint': {'fresh': {'report': proof_ref}}}}
    monkeypatch.setattr(joint_quality, 'verify_round_development', verify)
    return profile, request, selection, frozen, manifest, seen


def test_pixel_dual_requires_real_joint_inputs_before_reading_selection():
    with pytest.raises(ValueError, match='requires joint development'):
        q._pixel_joint_proof(None, 'never-read.json', rp.PIXEL_OWNED_SPANS.root)


def test_other_round_cannot_substitute_joint_proof():
    assert q._pixel_joint_proof(None, 'never-read.json', rp.COMPLETE_SPANS.root) is None
    with pytest.raises(ValueError, match='only to their registered'):
        q._pixel_joint_proof('never-read.json', 'never-read.json', rp.COMPLETE_SPANS.root)


def test_dispatch_passes_bound_manifest_and_exact_selected_report(tmp_path, monkeypatch):
    profile, request, selection, _, manifest, seen = fixture(tmp_path, monkeypatch)
    value = q._pixel_joint_proof(request['file'], selection['file'], profile.root)
    assert value['inputs'] == request and value['proof']['passed']
    assert seen[0][3]['configuration_manifest'] == manifest
    assert seen[0][1] == {} and seen[0][2] == 'tiny'


@pytest.mark.parametrize('passed,matching', [(False, True), (True, False)])
def test_failed_or_wrong_selected_proof_cannot_freeze(tmp_path, monkeypatch, passed, matching):
    profile, request, selection, _, _, _ = fixture(tmp_path, monkeypatch, passed=passed, matching=matching)
    with pytest.raises(ValueError, match='latency failed|actual scene selection'):
        q._pixel_joint_proof(request['file'], selection['file'], profile.root)


@pytest.mark.parametrize('mutation', ['source', 'manifest', 'future-freeze'])
def test_before_development_freeze_cannot_be_replaced(tmp_path, monkeypatch, mutation):
    profile, request, selection, frozen, _, seen = fixture(tmp_path, monkeypatch)
    value = json.loads(Path(frozen['file']).read_text(encoding='utf-8'))
    if mutation == 'source':
        value['source_inventory'] = {}
    elif mutation == 'manifest':
        value['references'] = []
    else:
        value['frozen_utc'] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    new_ref = save(profile.root / 'validation/source-inventory-before-dev.json', value)
    req = json.loads(Path(request['file']).read_text(encoding='utf-8'))
    req['source_freeze'] = new_ref
    save(profile.root / 'evaluation/joint-development-inputs.json', req)
    with pytest.raises(ValueError, match='before-dev freeze|must follow'):
        q._pixel_joint_proof(request['file'], selection['file'], profile.root)
    assert not seen


def test_freeze_detects_request_mutation_during_verification(tmp_path, monkeypatch):
    profile, request, selection, _, _, _ = fixture(tmp_path, monkeypatch)
    original = joint_quality.verify_round_development
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        save(profile.root / 'evaluation/joint-development-inputs.json', {'tampered': True})
        return result
    monkeypatch.setattr(joint_quality, 'verify_round_development', changed)
    with pytest.raises(ValueError, match='inputs changed'):
        q._pixel_joint_proof(request['file'], selection['file'], profile.root)


@pytest.mark.parametrize('which', ['selection', 'unselected-report'])
def test_freeze_detects_original_evidence_race(tmp_path, monkeypatch, which):
    profile, request, selection, _, _, _ = fixture(tmp_path, monkeypatch)
    original = joint_quality.verify_round_development
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = (Path(selection['file']) if which == 'selection' else
                profile.root / 'complete-exclusive/anchor/evaluation.json')
        save(path, {'replaced': True})
        return result
    monkeypatch.setattr(joint_quality, 'verify_round_development', changed)
    with pytest.raises(ValueError, match='original report changed'):
        q._pixel_joint_proof(request['file'], selection['file'], profile.root)


@pytest.mark.parametrize('operation', ['freeze', 'recompute', 'predecessor'])
def test_new_round_cannot_borrow_consumed_old_crop(operation):
    old_crop = rp.BROWSER_FONT.root / 'data/browser-crops-v1'
    old_selection = rp.BROWSER_FONT.root / 'evaluation/crop-selection.json'
    new_scene = rp.PIXEL_OWNED_SPANS.root / 'scenes/browser-cards-v4'
    new_selection = rp.PIXEL_OWNED_SPANS.root / 'evaluation/scene-selection.json'
    with pytest.raises(ValueError, match='must share the new round'):
        if operation == 'freeze':
            q.freeze_dual(old_selection, new_selection, old_crop, new_scene,
                          rp.PIXEL_OWNED_SPANS.root / 'evaluation/dual-freeze.json')
        elif operation == 'recompute':
            q.recompute_development(old_selection, new_selection, old_crop, new_scene)
        else:
            q.predecessor_crop_receipt(old_selection, old_crop, new_scene)


def test_new_round_cannot_write_dual_into_old_round():
    root = rp.PIXEL_OWNED_SPANS.root
    with pytest.raises(ValueError, match='must share the new round'):
        q.freeze_dual(root/'crop-selection.json', root/'scene-selection.json',
                      root/'data/crops', root/'scenes/cards', rp.COMPLETE_SPANS.root/'dual.json')


def test_verify_checks_slots_even_when_dual_file_is_in_an_old_round(tmp_path, monkeypatch):
    root = rp.PIXEL_OWNED_SPANS.root
    value = {'schema_version': q.DUAL_SCHEMA, 'production_enabled': False, 'promote': False,
             'enabled': False, 'source_inventory': q.source_inventory(),
             'crop_selection': {'file': str(root/'crop-selection.json')},
             'scene_selection': {'file': str(root/'scene-selection.json')},
             'crop_dataset': {'path': str(root/'data/crops')},
             'scene_dataset': {'path': str(root/'scenes/cards')}}
    path = tmp_path/'dual.json'
    save(path, value)
    monkeypatch.setattr(q, 'bound', lambda record: Path(record['file']))
    with pytest.raises(ValueError, match='must share the new round'):
        q.verify_dual(path, 'scene', root/'scene-selection.json', root/'scenes/cards')


def test_pixel_crop_contract_requires_three_candidates_plus_bound_baseline(monkeypatch):
    models = [{'id': name} for name in ('official', 'old', 'current')]
    frozen = {'schema_version': 'localocr.pixel-owned-before-dev-freeze.v1',
              'source_inventory': q.source_inventory(),
              'models': {'candidates': models, 'baseline': {'sha256': 'baseline'}}}
    monkeypatch.setattr(q.common, 'read', lambda path: frozen)
    good = {'candidate_bindings': models, 'baseline_binding': frozen['models']['baseline']}
    q._pixel_crop_models(good)
    with pytest.raises(ValueError, match='four-model set'):
        q._pixel_crop_models({**good, 'candidate_bindings': models[-1:]})
    with pytest.raises(ValueError, match='four-model set'):
        q._pixel_crop_models({**good, 'baseline_binding': {'sha256': 'other'}})


@pytest.mark.parametrize('missing', ['crop_dataset', 'scene_dataset'])
def test_new_round_missing_dataset_slot_cannot_bypass_ownership(tmp_path, monkeypatch, missing):
    root = rp.PIXEL_OWNED_SPANS.root
    value = {'schema_version': q.DUAL_SCHEMA, 'production_enabled': False, 'promote': False,
             'enabled': False, 'source_inventory': q.source_inventory(),
             'crop_selection': {'file': str(root/'crop-selection.json')},
             'scene_selection': {'file': str(root/'scene-selection.json')},
             'crop_dataset': {'path': str(root/'data/crops')},
             'scene_dataset': {'path': str(root/'scenes/cards')}}
    del value[missing]
    path = tmp_path/'dual.json'
    save(path, value)
    monkeypatch.setattr(q, 'bound', lambda record: Path(record['file']))
    with pytest.raises(ValueError, match='must share the new round'):
        q.verify_dual(path, 'scene', root/'scene-selection.json', root/'scenes/cards')
