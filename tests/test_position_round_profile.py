"""The opt-in position round preserves historical defaults and resource bounds."""
import math
from pathlib import Path

import pytest

from tools.PretrainedOcr import common, round_profile as rp


def test_position_profile_is_registered_explicitly_without_changing_old_default():
    profile = rp.OWNED_POSITION
    assert rp.by_name('ownership-position-20261022') is profile
    assert profile in rp.ROUNDS
    assert rp.CURRENT is rp.PIXEL_OWNED_SPANS
    assert profile.change == 'improve-owned-position-localization'
    assert profile.root == rp.ARTIFACTS / 'ownership-position-20261022'
    assert profile.maximum_seconds == profile.previous_used_seconds == 0.0
    assert profile.maximum_bytes == 2 * rp.GIB
    assert profile.owned_bytes == 1200 * 1024 ** 2
    assert common.MINIMUM_FREE_BYTES == 8 * rp.GIB
    assert profile.specialist_families == 1
    assert profile.previous_budget == rp.PIXEL_OWNED_SPANS.root / 'gpu-budget.json'
    assert profile.previous_sha256 == '7992c1d4b53399867e9ed8b4b597e244cb8234f8cf09d7185818daf1dc58ef18'


def test_position_profile_does_not_match_neighboring_old_or_prefix_paths():
    root = rp.OWNED_POSITION.root
    assert rp.profile_for(root / 'evaluation-dev/evaluation.json') is rp.OWNED_POSITION
    assert rp.profile_for(root.with_name(root.name + '-other') / 'evaluation.json') is None
    assert rp.profile_for(rp.PIXEL_OWNED_SPANS.root / 'validation/x.json') is rp.PIXEL_OWNED_SPANS
    assert rp.profile_for(rp.COMPLETE_SPANS.root / 'validation/x.json') is rp.COMPLETE_SPANS


def test_new_crop_generation_reaches_storage_gate_without_changing_default(tmp_path, monkeypatch):
    from tools.RegionKaomoji import browser_crops
    root = tmp_path / 'fresh-crops'
    monkeypatch.setattr(browser_crops.qd, 'output_root', lambda value: root)
    monkeypatch.setattr(browser_crops.rp, 'profile_for', lambda value: rp.OWNED_POSITION)
    def storage_gate(value, reserve):
        assert value == root and reserve == 256 * 1024 ** 2
        raise RuntimeError('Reached registered round storage guard')
    monkeypatch.setattr(browser_crops.qd, 'storage_gate', storage_gate)
    with pytest.raises(RuntimeError, match='Reached registered'):
        browser_crops.generate(root, 'not-read.json', counts=(100, 1000, 2000))
    assert rp.CURRENT is rp.PIXEL_OWNED_SPANS
    assert not root.exists()


def test_unrelated_historical_crop_generation_still_refuses(tmp_path, monkeypatch):
    from tools.RegionKaomoji import browser_crops
    root = tmp_path / 'old-round'
    monkeypatch.setattr(browser_crops.qd, 'output_root', lambda value: root)
    monkeypatch.setattr(browser_crops.rp, 'profile_for', lambda value: rp.COMPLETE_SPANS)
    with pytest.raises(ValueError, match='current round'):
        browser_crops.generate(root, 'not-read.json', counts=(100, 1000, 2000))
    assert not root.exists()


def test_position_predecessor_is_the_actual_closed_zero_second_pixel_ledger():
    profile = rp.OWNED_POSITION
    path = Path(profile.previous_budget)
    if not path.exists():
        pytest.skip('Preceding research ledger is absent on this host')
    assert common.digest(path) == profile.previous_sha256
    value = common.read(path)
    assert value['active'] is None and value['runs'] == []
    assert value['maximum_seconds'] == 0.0
    assert math.isclose(value['used_seconds'], profile.previous_used_seconds, abs_tol=1e-6)
    assert not path.with_name(path.name + '.lock').exists()


def test_position_root_quota_and_floor_cannot_be_borrowed_from_predecessor(tmp_path, monkeypatch):
    from tools.PretrainedOcr import quality_evaluate as quality
    root = rp.OWNED_POSITION.root
    calls = []
    monkeypatch.setattr(common, 'disk_guard', lambda path, floor: calls.append((path, floor)))
    quality.round_write_guard(root / 'evaluation-dev/evaluation.json', 17)
    assert calls and calls[0][1] == 8 * rp.GIB + 17


def test_position_round_refuses_low_disk_before_traversing_resources(monkeypatch):
    from tools.PretrainedOcr import quality_evaluate as quality, round_limits
    def low_disk(path, floor):
        assert floor == 8 * rp.GIB
        raise ValueError('Low disk fixture')
    def forbidden(path):
        raise AssertionError('No traversal or workload may start below the floor')
    monkeypatch.setattr(common, 'disk_guard', low_disk)
    monkeypatch.setattr(round_limits, '_no_reparse', forbidden)
    with pytest.raises(ValueError, match='Low disk'):
        quality.round_write_guard(rp.OWNED_POSITION.root)
