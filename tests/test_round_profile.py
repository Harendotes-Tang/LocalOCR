"""Round registry: root-scoped quotas, immutable predecessors and family counts."""
import dataclasses
import math
from pathlib import Path

import pytest

from tools.PretrainedOcr import round_profile as rp


def unit_profile(tmp_path, **changes):
    base = rp.RoundProfile(name='unit-round', change='unit', root=tmp_path / 'unit-round', maximum_bytes=64,
                           owned_bytes=32, previous_budget=tmp_path / 'old.json', previous_sha256='0' * 64,
                           previous_used_seconds=1.0, specialist_families=1)
    return dataclasses.replace(base, **changes)


def test_registered_rounds_have_distinct_roots_and_external_predecessors():
    roots = [Path(profile.root).resolve() for profile in rp.ROUNDS]
    assert len(set(roots)) == len(roots)
    for profile in rp.ROUNDS:
        assert not Path(profile.previous_budget).resolve().is_relative_to(Path(profile.root).resolve())
        assert 0 < profile.owned_bytes <= profile.maximum_bytes <= 2 * rp.GIB
    assert rp.GRAMMAR.previous_budget == rp.QUALITY.root / 'gpu-budget.json'
    assert rp.GRAMMAR.maximum_bytes == 2 * rp.GIB and rp.QUALITY.maximum_bytes == rp.GIB
    assert rp.GRAMMAR.specialist_families == 1 and rp.QUALITY.specialist_families == 2
    assert rp.CANDIDATES.previous_budget == rp.GRAMMAR.root / 'gpu-budget.json'
    assert rp.CANDIDATES.maximum_seconds == 0.0 and rp.CANDIDATES.maximum_bytes == rp.GIB
    assert rp.CANDIDATES.specialist_families == 1 and rp.CANDIDATES.change == 'improve-region-candidate-localization'
    assert rp.REAL_RENDERING.previous_budget == rp.CANDIDATES.root / 'gpu-budget.json'
    assert rp.REAL_RENDERING.maximum_seconds == 0.0 and rp.REAL_RENDERING.previous_used_seconds == 0.0
    assert rp.REAL_RENDERING.maximum_bytes == rp.GIB and rp.REAL_RENDERING.specialist_families == 1
    assert rp.REAL_RENDERING.change == 'improve-real-rendering-bracket-evidence'
    assert rp.CURRENT is rp.REAL_RENDERING and rp.by_name(rp.REAL_RENDERING.name) is rp.REAL_RENDERING
    assert rp.by_name(rp.CANDIDATES.name) is rp.CANDIDATES
    assert rp.by_name(rp.GRAMMAR.name) is rp.GRAMMAR


def test_profile_lookup_is_root_scoped_not_prefix_scoped(tmp_path, monkeypatch):
    profile = unit_profile(tmp_path)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    assert rp.profile_for(profile.root / 'models' / 'x.json') is profile
    assert rp.profile_for(profile.root) is profile
    assert rp.profile_for(tmp_path / 'unit-round-sibling' / 'x.json') is None
    assert rp.profile_for(tmp_path) is None
    assert rp.round_root(profile.root / 'a', default='fallback') == profile.root
    assert rp.round_root(tmp_path / 'elsewhere', default='fallback') == 'fallback'
    assert rp.specialist_families(profile.root / 'data') == 1
    assert rp.specialist_families(tmp_path / 'elsewhere') == rp.DEFAULT_SPECIALIST_FAMILIES
    with pytest.raises(KeyError):
        rp.by_name('missing')


@pytest.mark.parametrize('changes', [
    {'maximum_bytes': 0}, {'owned_bytes': 65}, {'owned_bytes': 1.0}, {'previous_sha256': 'ABC'},
    {'specialist_families': 0}, {'specialist_families': 3},
    {'maximum_seconds': -1.0}, {'maximum_seconds': 7201.0}, {'maximum_seconds': float('nan')}, {'maximum_seconds': True},
])
def test_profile_invariants_fail_closed(tmp_path, changes):
    with pytest.raises(ValueError):
        unit_profile(tmp_path, **changes)


def test_zero_second_profile_is_allowed_for_cpu_only_rounds(tmp_path):
    assert unit_profile(tmp_path, maximum_seconds=0.0).maximum_seconds == 0.0


def test_candidates_predecessor_ledger_matches_actual_closed_history():
    ledger = Path(rp.CANDIDATES.previous_budget)
    if not ledger.exists():
        pytest.skip('Historical grammar-round ledger is absent on this host')
    from tools.PretrainedOcr.common import digest, read
    assert digest(ledger) == rp.CANDIDATES.previous_sha256
    state = read(ledger)
    assert state['active'] is None
    assert math.isclose(state['used_seconds'], rp.CANDIDATES.previous_used_seconds, abs_tol=1e-6)
    assert not ledger.with_name(ledger.name + '.lock').exists()


def test_real_rendering_predecessor_ledger_is_the_closed_zero_second_candidates_ledger():
    ledger = Path(rp.REAL_RENDERING.previous_budget)
    if not ledger.exists():
        pytest.skip('Historical candidates-round ledger is absent on this host')
    from tools.PretrainedOcr.common import digest, read
    assert digest(ledger) == rp.REAL_RENDERING.previous_sha256
    state = read(ledger)
    assert state['active'] is None and state['runs'] == []
    assert state['maximum_seconds'] == 0.0 and state['used_seconds'] == rp.REAL_RENDERING.previous_used_seconds == 0.0
    assert not ledger.with_name(ledger.name + '.lock').exists()


def test_predecessor_inside_round_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        unit_profile(tmp_path, previous_budget=tmp_path / 'unit-round' / 'old.json')


def test_grammar_predecessor_ledger_matches_actual_closed_history():
    ledger = Path(rp.GRAMMAR.previous_budget)
    if not ledger.exists():
        pytest.skip('Historical quality-round ledger is absent on this host')
    from tools.PretrainedOcr.common import digest, read
    assert digest(ledger) == rp.GRAMMAR.previous_sha256
    state = read(ledger)
    assert state['active'] is None
    assert math.isclose(state['used_seconds'], rp.GRAMMAR.previous_used_seconds, abs_tol=1e-6)
    assert not ledger.with_name(ledger.name + '.lock').exists()


def test_scene_freeze_family_count_follows_round_profile(tmp_path, monkeypatch):
    from tools.RegionKaomoji import evaluate as scene
    profile = unit_profile(tmp_path)
    inputs = str(profile.root / 'scenes' / 'inference-inputs.json')
    one = {'models': [{'family': 'tiny', 'stage': 'finetuned'}], 'inputs': {'file': inputs}}
    two = {'models': [{'family': 'tiny', 'stage': 'finetuned'}, {'family': 'rep', 'stage': 'finetuned'}],
           'inputs': {'file': inputs}}
    with pytest.raises(ValueError, match='families'):
        scene.final_families(one)
    scene.final_families(two)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    scene.final_families(one)
    with pytest.raises(ValueError, match='families'):
        scene.final_families(two)
    with pytest.raises(ValueError, match='families'):
        scene.final_families({'models': [{'family': 'tiny', 'stage': 'official-pretrained'}], 'inputs': {'file': inputs}})


def test_crop_family_coverage_requires_before_and_after_per_required_family():
    from tools.PretrainedOcr.evaluate import family_coverage
    tiny = [{'family': 'tiny', 'stage': 'official-pretrained'}, {'family': 'tiny', 'stage': 'finetuned'}]
    rep = [{'family': 'rep', 'stage': 'expanded-head-initialization'}, {'family': 'rep', 'stage': 'finetuned'}]
    assert family_coverage(tiny, 1) and family_coverage(tiny + rep, 2)
    assert not family_coverage(tiny, 2) and not family_coverage(tiny + rep, 1)
    assert not family_coverage(tiny[1:], 1) and not family_coverage(tiny[:1], 1)


def test_scene_output_namespace_follows_round_profile(tmp_path, monkeypatch):
    from tools.RegionKaomoji import evaluate as scene
    profile = unit_profile(tmp_path)
    monkeypatch.setattr(rp, 'ROUNDS', (profile,))
    good = profile.root / 'evaluation-dev' / 'evaluation.json'
    assert scene.output_path(good) == Path(good).resolve()
    with pytest.raises(ValueError):
        scene.output_path(profile.root / 'models' / 'evaluation.json')
