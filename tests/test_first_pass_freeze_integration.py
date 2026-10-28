"""Position proof dispatch fixtures are not real worker quality or first use."""
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.PretrainedOcr import quality_evaluate as q, round_profile as rp
from tools.RegionKaomoji import first_pass_quality


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return q.reference(path)


def setup(tmp_path, monkeypatch, passed=True, matching=True):
    profile = dataclasses.replace(rp.FIRST_PASS_POSITION, root=tmp_path/'round')
    monkeypatch.setattr(rp, 'FIRST_PASS_POSITION', profile)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    monkeypatch.setattr(q, 'source_inventory', lambda: {'fixture.py': 'a'*64})
    now = datetime.now(timezone.utc)-timedelta(minutes=5)
    manifest = save(profile.root/'routing/configuration-manifest.json', {'frozen_utc': now.isoformat()})
    frozen = save(profile.root/'validation/source-inventory-before-dev.json', {
        'schema_version': 'localocr.first-pass-before-dev-freeze.v1',
        'source_inventory': q.source_inventory(), 'references': [manifest],
        'frozen_utc': (now+timedelta(seconds=1)).isoformat()})
    scored = {tier: {population: save(profile.root/tier/population/'evaluation.json', {
        'completed_utc': (now+timedelta(seconds=4)).isoformat(),
        'resources': {'started_utc': (now+timedelta(seconds=2)).isoformat()}})
        for population in ('anchor', 'fresh')} for tier in first_pass_quality.TIERS}
    request = save(profile.root/'evaluation/position-inputs.json', {
        'schema_version': 'localocr.first-pass-development-inputs.v1', 'selected_id': 'fixture',
        'configuration_manifest': manifest, 'source_freeze': frozen,
        'scored_reports': scored, 'repeated_reports': {}, 'passed': True})
    selection = save(profile.root/'evaluation/selection.json', {
        'selected_id': 'fixture', 'dev_report': scored['union']['fresh']})
    calls = []
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {'passed': passed, 'selected_tier': 'union' if passed else 'start',
                'reports': {'union': {'fresh': {'report': scored['union' if matching else 'local']['fresh']}}}}
    monkeypatch.setattr(first_pass_quality, 'verify_round_development', verify)
    return profile, request, selection, frozen, manifest, scored, calls


def test_first_pass_proof_required_before_selection_read():
    with pytest.raises(ValueError, match='requires four-tier'):
        q._first_pass_development_proof(None, 'never-read.json', rp.FIRST_PASS_POSITION.root)
    with pytest.raises(ValueError, match='only to the first-pass round'):
        q._first_pass_development_proof('never-read.json', 'never-read.json', rp.PIXEL_OWNED_SPANS.root)


def test_dispatch_uses_original_refs_and_not_the_request_verdict(tmp_path, monkeypatch):
    profile, request, selection, _, manifest, _, calls = setup(tmp_path, monkeypatch)
    value = q._first_pass_development_proof(request['file'], selection['file'], profile.root)
    assert value['inputs'] == request and value['proof']['passed']
    assert calls[0][1]['configuration_manifest'] == manifest


@pytest.mark.parametrize('passed,matching', [(False, True), (True, False)])
def test_wrong_or_failed_recomputed_proof_refuses(tmp_path, monkeypatch, passed, matching):
    profile, request, selection, *_ = setup(tmp_path, monkeypatch, passed, matching)
    with pytest.raises(ValueError, match='failed|actual scene selection'):
        q._first_pass_development_proof(request['file'], selection['file'], profile.root)


def test_worker_must_start_after_source_freeze_not_merely_finish_later(tmp_path, monkeypatch):
    profile, request, selection, frozen, _, scored, calls = setup(tmp_path, monkeypatch)
    path = Path(scored['initial']['anchor']['file'])
    value = json.loads(path.read_text())
    stamp = datetime.fromisoformat(json.loads(Path(frozen['file']).read_text())['frozen_utc'])
    value['resources']['started_utc'] = (stamp-timedelta(seconds=1)).isoformat()
    scored['initial']['anchor'] = save(path, value)
    req = json.loads(Path(request['file']).read_text())
    req['scored_reports'] = scored
    save(Path(request['file']), req)
    with pytest.raises(ValueError, match='must follow source freeze'):
        q._first_pass_development_proof(request['file'], selection['file'], profile.root)
    assert not calls


def test_bound_request_cannot_race_verification(tmp_path, monkeypatch):
    profile, request, selection, *_ = setup(tmp_path, monkeypatch)
    original = first_pass_quality.verify_round_development
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        save(Path(request['file']), {'tampered': True})
        return result
    monkeypatch.setattr(first_pass_quality, 'verify_round_development', mutate)
    with pytest.raises(ValueError, match='changed during verification'):
        q._first_pass_development_proof(request['file'], selection['file'], profile.root)


@pytest.mark.parametrize('operation', ['freeze', 'recompute', 'predecessor'])
def test_first_pass_slots_cannot_borrow_previous_crop_data(operation):
    old, new = rp.PIXEL_OWNED_SPANS.root, rp.FIRST_PASS_POSITION.root
    with pytest.raises(ValueError, match='must share the new round'):
        if operation == 'freeze':
            q.freeze_dual(old/'selection', new/'selection', old/'data', new/'scenes', new/'dual')
        elif operation == 'recompute':
            q.recompute_development(old/'selection', new/'selection', old/'data', new/'scenes')
        else:
            q.predecessor_crop_receipt(old/'selection', old/'data', new/'scenes')


@pytest.mark.parametrize('missing', ['crop_dataset', 'scene_dataset'])
def test_first_pass_verify_missing_slot_fails_closed(tmp_path, monkeypatch, missing):
    root = rp.FIRST_PASS_POSITION.root
    value = {'schema_version': q.DUAL_SCHEMA, 'production_enabled': False, 'promote': False,
        'enabled': False, 'source_inventory': q.source_inventory(),
        'crop_selection': {'file': str(root/'crop-selection')},
        'scene_selection': {'file': str(root/'scene-selection')},
        'crop_dataset': {'path': str(root/'data')}, 'scene_dataset': {'path': str(root/'scenes')}}
    del value[missing]
    path = tmp_path/'dual.json'
    save(path, value)
    monkeypatch.setattr(q, 'bound', lambda reference: Path(reference['file']))
    with pytest.raises(ValueError, match='must share the new round'):
        q.verify_dual(path, 'scene', root/'scene-selection', root/'scenes')


def test_first_pass_crop_contract_requires_current_four_models(monkeypatch):
    models = [{'id': name} for name in ('official', 'old', 'current')]
    frozen = {'schema_version': 'localocr.first-pass-before-dev-freeze.v1',
        'source_inventory': q.source_inventory(), 'models': {'candidates': models, 'baseline': {'id': 'baseline'}}}
    monkeypatch.setattr(q.common, 'read', lambda path: frozen)
    q._first_pass_crop_models({'candidate_bindings': models, 'baseline_binding': {'id': 'baseline'}})
    with pytest.raises(ValueError, match='four-model set'):
        q._first_pass_crop_models({'candidate_bindings': models[-1:], 'baseline_binding': {'id': 'baseline'}})
