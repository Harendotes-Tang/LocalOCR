"""Local original-pixel proposals through existing candidate and fusion guards."""
import copy

import numpy as np
import pytest

from tools.RegionKaomoji import candidates as c, fusion, local_windows


def pixels():
    core = np.zeros((60, 120), dtype=np.uint8)
    core[16:36, 4:17] = core[16:36, 77:91] = 1
    for x, right in ((25, False), (65, True)):
        stroke = np.zeros((20, 5), dtype=np.uint8)
        stroke[:, 0] = stroke[0, :] = stroke[-1, :] = 1
        core[16:36, x:x+5] |= np.fliplr(stroke) if right else stroke
    core[24:27, 35:38] = core[24:27, 53:56] = 1
    core[32:34, 43:49] = 1
    image = np.repeat((255-core*255)[:, :, None], 3, axis=2)
    image[10:12, 35:37] = 215
    image[27:29, 20:23] = 215
    return image


def settings():
    return {**copy.deepcopy(c.DEFAULT_CONFIG), 'shape_contrast_threshold': 80,
            'local_windows': local_windows.configuration()}


def materialize(image, entries=()):
    config = c.configuration(settings())
    mask, core = c._masks(image, config)
    proposals = local_windows.propose_windows(mask, core, [{'bbox': [4, 9, 91, 36]}], entries, [],
        limits={key: config[key] for key in local_windows.DEFAULT_LIMITS},
        options=config['local_windows'])['proposals']
    assert proposals
    built = c._span_record(image, proposals[0], entries, config, mask, core, None, None,
                           'local-window-000', evidence_key='local_window')
    assert built is not None
    return config, built[0]


def test_real_local_window_materializes_complete_weak_ink_and_clamped_rgb_crop():
    image = pixels()
    before = image.copy()
    _, item = materialize(image)
    evidence = item['evidence']['local_window']
    assert evidence['body_safe'] is False and evidence['base_only_pixels'] > 0
    assert item['sources'] == ['image-local-window']
    box, limits = item['bbox'], evidence['crop_limits']
    assert limits[0] <= box[0] <= 20 < 37 <= box[2] <= limits[2]
    assert limits[1] <= box[1] <= 10 < 29 <= box[3] <= limits[3]
    assert np.all(image[10:12, 35:37] == 215)
    assert np.all(image[27:29, 20:23] == 215)
    raw_view = fusion.view_image(image, box, 'raw')
    assert np.array_equal(raw_view, before[box[1]:box[3], box[0]:box[2]])
    assert np.array_equal(image, before)


def test_materialization_requires_existing_context_proof_for_mixed_body():
    entry = {'index': 0, 'bbox': [4, 9, 91, 36], 'text': 'body [..] tail', 'score': .99}
    _, item = materialize(pixels(), [entry])
    assert item['evidence']['neighbor_body_overlap']
    assert item['baseline_indices'] == [] and item['splice_baseline_index'] == 0
    assert item['baseline_parent_sha256'] == c.baseline_parent_digest(entry)
    assert {part['role'] for part in item['context_crops']} == {'left', 'right'}
    assert item['evidence']['local_window']['body_safe'] is False
    # Geometry may declare context crops; unchanged fusion still requires their
    # actual baseline readings before replacing the mixed parent's body bytes.


def test_absent_option_never_calls_new_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy path must not run local windows')
    monkeypatch.setattr(local_windows, 'propose_windows', forbidden)
    result = c.locate_candidates(pixels(), [])
    assert 'local_windows' not in result['config']
    assert 'local_window_detection' not in result


def test_shared_full_cap_stops_proposer_and_materializer(monkeypatch):
    image = pixels()
    base = {**copy.deepcopy(c.DEFAULT_CONFIG), 'maximum_candidates': 1}
    old = c.locate_candidates(image, [], base)
    eligible = [item for item in old['candidates'] if item['route_eligible']]
    assert eligible
    def forbidden(*args, **kwargs):
        raise AssertionError('Exhausted cap must stop new work')
    monkeypatch.setattr(local_windows, 'propose_windows', forbidden)
    monkeypatch.setattr(c, '_span_record', forbidden)
    result = c.locate_candidates(image, [], {**base, 'local_windows': local_windows.configuration()})
    assert result['candidates'] == eligible
    assert result['local_window_detection']['rejections']['shared-budget-exhausted'] == 1


def test_optional_candidates_share_caps_and_fusion_rejects_forged_geometry():
    image = pixels()
    config = settings()
    result = c.locate_candidates(image, [], config)
    assert result['local_window_detection']['proposals'] > 0
    assert result['crop_pixels'] == sum(c.area(item['bbox']) for item in result['candidates'])
    assert result['crop_pixels'] <= config['maximum_crop_pixels']
    assert len(result['candidates']) <= config['maximum_candidates']
    forged = copy.deepcopy(result['candidates'])
    assert forged
    forged[0]['evidence']['local_window'] = {'body_safe': True}
    with pytest.raises(ValueError, match='differs from automatic'):
        fusion.fuse_candidates(image, [], forged, {}, candidate_config=config)


@pytest.mark.parametrize('extra', [{'local_windows': None}, {'local_windows': {}},
    {'local_windows': local_windows.configuration(), 'line_spans': {}}])
def test_invalid_or_conflicting_window_options_fail_closed(extra):
    with pytest.raises(ValueError):
        c.configuration({**copy.deepcopy(c.DEFAULT_CONFIG), **extra})
