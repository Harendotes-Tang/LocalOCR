"""Synthetic gate fixtures exercise validation only, never OCR accuracy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.RegionKaomoji import joint_quality as joint
from tools.RegionKaomoji import span_quality as span


MODEL = 'fixture-specialist'
SHA = '1' * 64
SOURCE = 'tools/RegionKaomoji/joint_quality.py'


def evidence(anchor, exact, safe, tier='start', *, seconds=2., stamp=0):
    count = 244 if anchor else 256
    config = {'views': ['raw', 'edge-pad-4'], 'limits': {'maximum_candidates': 48},
              'candidate': {'fixture_tier': tier}}
    dataset = {'dataset_sha256': span.ANCHOR_DATASET_SHA256 if anchor else '2' * 64,
               'split_sha256': {'dev': span.ANCHOR_DEV_SHA256 if anchor else '3' * 64,
                                'heldout': '4' * 64},
               'counts': {'dev': {'scenes': 60, 'faces': count}}}
    value = {'schema_version': 'localocr.region-research.v1', 'split': 'dev',
             'completed_utc': (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=stamp)).isoformat(),
             'actual_cpu_inference': True, 'independent_heldout': False,
             'ground_truth_inference_input': False, 'promote': False, 'enabled': False,
             'pipeline_binding': {'dataset': dataset, 'inputs': {'file': 'anchor' if anchor else 'fresh', 'sha256': SHA},
                 'models': [{'id': MODEL, 'candidate_sha256': SHA, 'model_sha256': SHA}],
                 'baseline': {'manifest_sha256': SHA}, 'classifier': {'sha256': SHA},
                 'implementation': {'tools/RegionKaomoji/span_quality.py': SHA, SOURCE: SHA},
                 'config': config, 'config_sha256': span._hash(config), 'promote': False, 'enabled': False},
             'summary': {'scenes': 60, 'candidate_stage': {'faces': count, 'safe_complete_coverage': safe},
                 'models': {MODEL: {'isolated_kaomoji': {'count': count, 'exact': exact},
                     'body': {'regions': 8, **dict.fromkeys(span.BODY_COUNTERS, 0)},
                     'negative': {'negative_scenes': 12, 'graphic_regions': 2,
                                  **dict.fromkeys(span.NEGATIVE_COUNTERS, 0)}}}},
             'resources': {'warm': {'count': 60, 'p50_seconds': seconds, 'p95_seconds': seconds},
                 'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1,
                 'actual_baseline_detection': True, 'actual_specialist_models': 1,
                 'no_python_children_or_socket_io': True, 'capture': 'successful-owned-subprocess',
                 'worker_exit_code': 0, 'worker_pid': stamp + 1000, 'warmup_scenes': 3,
                 'total_warm_seconds': seconds * 60, 'peak_rss_bytes': 1000, 'process_wall_seconds': 5.,
                 'measurement_machine': {'schema_version': 'localocr.cpu-measurement-machine.v1',
                                         'fingerprint_sha256': SHA},
                 'baseline_runtime': {'schema_version': 'fixture-runtime'}},
             'target_correctness': [{'scene_id': 'fixture-scene', 'truth_id': str(i), 'correct': i < exact}
                                    for i in range(count)]}
    return value


@pytest.fixture
def scored():
    return {tier: {'anchor': evidence(True, 20 if tier == 'start' else 32, 66 if tier == 'start' else 100, tier),
                   'fresh': evidence(False, 16 if tier == 'start' else 20, 64 if tier == 'start' else 100, tier)}
            for tier in joint.TIERS}


def quality(scored):
    return joint.summarize_quality(scored, MODEL)


def repetitions(scored, tier, *, start_times=None, candidate_times=None):
    result = {}
    for population in joint.POPULATIONS:
        rows = []
        for index in range(6):
            role = 'start' if index % 2 == 0 else tier
            view = deepcopy(scored[role][population])
            stamp = 10 + index
            view['completed_utc'] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=stamp)).isoformat()
            times = ((start_times if role == 'start' else candidate_times) or [2.] * 180)[(index // 2) * 60:(index // 2 + 1) * 60]
            from tools.PretrainedOcr.evaluate import percentiles
            view['resources']['warm'] = percentiles(times)
            view['resources']['total_warm_seconds'] = sum(times)
            rows.append({'view': view, 'seconds': times, 'output_sha256': SHA})
        result[population] = rows
    return result


def test_quality_is_independent_of_single_report_latency(scored):
    for tier in joint.TIERS[1:]:
        for value in scored[tier].values():
            value['resources']['warm']['p95_seconds'] = 100.
    verdict = quality(scored)
    assert verdict['quality_eligible_tiers'] == list(joint.TIERS[1:])
    assert verdict['latency_status'] == 'NOT_RUN'
    assert verdict['selected_tier'] == 'start' and not verdict['passed']
    assert 'warm_p95_within_115_percent' not in verdict['tiers']['joint']['anchor']['checks']


def test_earlier_failure_does_not_block_joint_quality(scored):
    scored['complete-exclusive']['anchor']['summary']['candidate_stage']['safe_complete_coverage'] = 99
    assert quality(scored)['quality_eligible_tiers'] == ['local-exclusive', 'joint']


@pytest.mark.parametrize('population', joint.POPULATIONS)
def test_total_gain_cannot_hide_previous_correct_loss(scored, population):
    rows = scored['joint'][population]['target_correctness']
    rows[0]['correct'], rows[40]['correct'] = False, True
    assert not quality(scored)['tiers']['joint']['passed']


@pytest.mark.parametrize('counter', span.BODY_COUNTERS + span.NEGATIVE_COUNTERS)
def test_six_protection_counters_are_hard_requirements(scored, counter):
    model = scored['joint']['fresh']['summary']['models'][MODEL]
    model['body' if counter in span.BODY_COUNTERS else 'negative'][counter] = 1
    assert not quality(scored)['tiers']['joint']['passed']


@pytest.mark.parametrize('mutation', ['missing-tier', 'extra-tier', 'missing-population', 'heldout', 'source', 'machine', 'different-config', 'anchor', 'fresh-alias', 'missing-joint-source', 'malformed'])
def test_quality_rejects_malformed_cross_identity_evidence(scored, mutation):
    if mutation == 'missing-tier':
        scored.pop('joint')
    elif mutation == 'extra-tier':
        scored['post-hoc'] = scored['joint']
    elif mutation == 'missing-population':
        scored['joint'].pop('fresh')
    elif mutation == 'heldout':
        scored['joint']['fresh']['split'] = 'heldout'
    elif mutation == 'source':
        scored['joint']['fresh']['pipeline_binding']['implementation'][SOURCE] = '2' * 64
    elif mutation == 'machine':
        scored['joint']['fresh']['resources']['measurement_machine']['fingerprint_sha256'] = '2' * 64
    elif mutation == 'different-config':
        binding = scored['joint']['fresh']['pipeline_binding']
        binding['config']['candidate'] = {'extra': True}
        binding['config_sha256'] = span._hash(binding['config'])
    elif mutation == 'anchor':
        for value in (v['anchor'] for v in scored.values()):
            value['pipeline_binding']['dataset']['dataset_sha256'] = '5' * 64
    elif mutation == 'fresh-alias':
        for value in (v['fresh'] for v in scored.values()):
            value['pipeline_binding']['dataset'] = deepcopy(scored['start']['anchor']['pipeline_binding']['dataset'])
    elif mutation == 'missing-joint-source':
        for pair in scored.values():
            for value in pair.values():
                value['pipeline_binding']['implementation'].pop(SOURCE)
    else:
        scored['joint']['anchor']['summary']['models'][MODEL]['isolated_kaomoji']['exact'] = True
    with pytest.raises(ValueError):
        quality(scored)


def test_pool_all_samples_instead_of_fastest_run_or_average_p95(scored):
    times = [2.] * 171 + [6.] * 9
    repeats = repetitions(scored, 'joint', candidate_times=times)
    verdict = joint.summarize_repeated_latency(scored, 'joint', repeats, MODEL)
    # NumPy's existing linear percentile interpolates across the 95% boundary.
    assert verdict['anchor']['candidate']['count'] == 180
    assert verdict['anchor']['candidate']['p95_seconds'] == pytest.approx(2.2)
    assert verdict['passed']
    assert verdict['pooled_candidate']['count'] == 360


def test_actual_pooled_ratio_fails_without_drift_deduction(scored):
    verdict = joint.summarize_repeated_latency(scored, 'joint', repetitions(scored, 'joint', candidate_times=[2.301] * 180), MODEL)
    assert verdict['anchor']['ratio'] > 1.15 and not verdict['passed']


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'order', 'bad-order-stamp', 'quality-drift', 'output-drift', 'times-missing', 'times-nan', 'times-count', 'times-warm', 'times-total', 'different-config'])
def test_repeat_gate_rejects_missing_unstable_or_unverified_samples(scored, mutation):
    repeats = repetitions(scored, 'joint')
    rows = repeats['anchor']
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'extra':
        rows.append(rows[0])
    elif mutation == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == 'bad-order-stamp':
        rows[1]['view']['completed_utc'] = rows[0]['view']['completed_utc']
    elif mutation == 'quality-drift':
        rows[1]['view']['target_correctness'][0]['correct'] = False
    elif mutation == 'output-drift':
        rows[1]['output_sha256'] = '2' * 64
    elif mutation == 'times-missing':
        rows[1].pop('seconds')
    elif mutation == 'times-nan':
        rows[1]['seconds'][0] = float('nan')
    elif mutation == 'times-count':
        rows[1]['seconds'].pop()
    elif mutation == 'times-warm':
        rows[1]['view']['resources']['warm']['p95_seconds'] += .1
    elif mutation == 'times-total':
        rows[1]['view']['resources']['total_warm_seconds'] += .1
    else:
        binding = rows[1]['view']['pipeline_binding']
        binding['config']['candidate'] = {'wrong': True}
        binding['config_sha256'] = span._hash(binding['config'])
    signatures = {population: {'start': SHA, 'joint': SHA} for population in joint.POPULATIONS}
    with pytest.raises(ValueError):
        joint.summarize_repeated_latency(scored, 'joint', repeats, MODEL, scored_output_sha256=signatures)


def test_quality_failure_cannot_obtain_latency_claim(scored):
    scored['joint']['anchor']['summary']['candidate_stage']['safe_complete_coverage'] = 99
    with pytest.raises(ValueError, match='quality'):
        joint.summarize_repeated_latency(scored, 'joint', repetitions(scored, 'joint'), MODEL)


@pytest.mark.parametrize('population', joint.POPULATIONS)
def test_repeat_phase_waits_for_all_predeclared_scoring(scored, population):
    repeats = repetitions(scored, 'complete-exclusive')
    scored['joint'][population]['completed_utc'] = '2026-01-01T00:00:20+00:00'
    with pytest.raises(ValueError, match='chronological'):
        joint.summarize_repeated_latency(scored, 'complete-exclusive', repeats, MODEL)


@pytest.mark.parametrize('population,counter,value', [('anchor', 'exact', 29), ('fresh', 'exact', 16), ('anchor', 'safe', 99)])
def test_anchor_floors_and_each_population_strict_gain(scored, population, counter, value):
    report = scored['joint'][population]
    if counter == 'exact':
        report['summary']['models'][MODEL]['isolated_kaomoji']['exact'] = value
        for i, target in enumerate(report['target_correctness']):
            target['correct'] = i < value
    else:
        report['summary']['candidate_stage']['safe_complete_coverage'] = value
    assert not quality(scored)['tiers']['joint']['passed']


def test_ranking_uses_fresh_then_anchor_then_combined_p95_then_declared_order(scored):
    q = quality(scored)
    timing = {tier: {'passed': True, 'pooled_candidate': {'p95_seconds': 2.}}
              for tier in joint.TIERS[1:]}
    assert joint.select_tier(q, timing)['selected_tier'] == 'complete-exclusive'
    timing['local-exclusive']['pooled_candidate']['p95_seconds'] = 1.9
    assert joint.select_tier(q, timing)['selected_tier'] == 'local-exclusive'
    q['tiers']['joint']['anchor']['final_exact'] += 1
    assert joint.select_tier(q, timing)['selected_tier'] == 'joint'
    q['tiers']['complete-exclusive']['fresh_development']['final_exact'] += 1
    assert joint.select_tier(q, timing)['selected_tier'] == 'complete-exclusive'
    for item in timing.values():
        item['passed'] = False
    result = joint.select_tier(q, timing)
    assert result['selected_tier'] == 'start' and not result['passed']


def test_prediction_fingerprint_ignores_known_timings_only():
    records = [{'id': 'x', 'seconds': 1., 'prediction': {'timings': {'baseline_seconds': 1.},
        'models': {MODEL: {'regions': [{'text': 'x', 'bbox': [0, 0, 2, 2]}],
                           'views': {'c': [{'text': 'x', 'score': .9}]},
                           'decisions': [{'accepted': True}], 'specialist_seconds': 1.,
                           'fusion_seconds': 1., 'view_latency': {'count': 2, 'p95_seconds': 1.}}}}}]
    mutated = deepcopy(records)
    mutated[0]['seconds'] = 8.
    mutated[0]['prediction']['timings']['baseline_seconds'] = 8.
    mutated[0]['prediction']['models'][MODEL]['fusion_seconds'] = 8.
    assert joint.prediction_identity(records) == joint.prediction_identity(mutated)
    mutated[0]['prediction']['models'][MODEL]['views']['c'][0]['score'] = .8
    assert joint.prediction_identity(records) != joint.prediction_identity(mutated)


@pytest.fixture
def replay_fixture(tmp_path, monkeypatch, scored):
    """Stub existing replay/masks only for wrapper orchestration tests."""
    from tools.PretrainedOcr import quality_evaluate as existing
    from tools.RegionKaomoji import evaluate as scene

    paths, views = {}, {}
    serial = 0
    def write(value):
        nonlocal serial
        serial += 1
        directory = tmp_path / ('evaluation-' + str(serial))
        directory.mkdir()
        value = deepcopy(value)
        value['pipeline_binding']['inputs']['file'] = str(tmp_path / ('anchor' if value['pipeline_binding']['dataset']['counts']['dev']['faces'] == 244 else 'fresh') / 'inference-inputs.json')
        value['outputs'] = {}
        for name, filename in [('predictions', 'predictions.jsonl'), ('resources', 'resources.json'), ('alignment', 'alignment-private.json')]:
            output = directory / filename
            output.write_text('{}', encoding='utf-8')
            value['outputs'][name] = existing.reference(output)
        path = directory / 'evaluation.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        views[str(path)] = value
        return path
    for tier, populations in scored.items():
        paths[tier] = {population: write(value) for population, value in populations.items()}
    manifest = {'schema_version': 'localocr.joint-configurations.v1', 'frozen_utc': '2025-12-31T00:00:00+00:00',
        'selected_id': MODEL, 'configurations': {tier: pair['anchor']['pipeline_binding']['config'] for tier, pair in scored.items()},
        'populations': {population: {key: views[str(paths['start'][population])]['pipeline_binding'][key] for key in ('dataset', 'inputs')}
                        for population in joint.POPULATIONS}, 'source_inventory': {'fixture-source': SHA},
        'models': scored['start']['anchor']['pipeline_binding']['models'],
        'baseline': scored['start']['anchor']['pipeline_binding']['baseline']}
    manifest_path = tmp_path / 'configuration-manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(existing.rp, 'profile_for', lambda path: SimpleNamespace(name='pixel-owned-spans-20261020', root=tmp_path))
    monkeypatch.setattr(existing, 'source_inventory', lambda: {'fixture-source': SHA})
    monkeypatch.setattr(scene, 'output_path', lambda path: Path(path).resolve())
    monkeypatch.setattr(existing, '_scene_regression_summary', lambda path, report: deepcopy(report['summary']))
    monkeypatch.setattr(scene, 'image_jobs', lambda root, split, inputs: root)
    def truth(root, split, jobs):
        count = 244 if root.name == 'anchor' else 256
        return [{'id': 'fixture-scene', 'regions': [{'kind': 'kaomoji', 'id': str(i), 'text': 'X'} for i in range(count)]},
                *[{'id': 'empty-' + str(i), 'regions': []} for i in range(59)]]
    monkeypatch.setattr(scene, 'load_truth', truth)
    monkeypatch.setattr(scene, 'masks_for_row', lambda root, row: [])
    def predictions(path, jobs):
        report = next(value for value in views.values() if value['outputs']['predictions']['file'] == path)
        return [{'id': 'fixture-scene', 'seconds': 2., 'prediction': {'models': {MODEL: {'regions': report['target_correctness']}}}},
                *[{'id': 'empty-' + str(i), 'seconds': 2., 'prediction': {'models': {MODEL: {'regions': []}}}}
                  for i in range(59)]]
    monkeypatch.setattr(scene, 'read_predictions', predictions)
    monkeypatch.setattr(scene, 'aligned_region', lambda regions, index, masks, config: ('X' if regions[index]['correct'] else '?', 0))
    return tmp_path, paths, views, manifest_path, existing, write


def test_quality_wrapper_recomputes_all_eight_reports_and_binds_manifest(replay_fixture):
    root, paths, views, manifest, existing, write = replay_fixture
    result = joint.verify_round_quality(paths, MODEL, round_root=root, configuration_manifest=existing.reference(manifest))
    assert result['quality_eligible_tiers'] == list(joint.TIERS[1:])
    assert result['reports']['joint']['fresh']['report'] == existing.reference(paths['joint']['fresh'])
    assert result['source_inventory'] == {'fixture-source': SHA}


@pytest.mark.parametrize('mutation', ['old-round', 'duplicate', 'heldout', 'manifest-tamper', 'manifest-late', 'fresh-mismatch', 'source-drift', 'output-tamper'])
def test_wrapper_rejects_unsafe_paths_manifest_or_output_changes(replay_fixture, monkeypatch, mutation):
    root, paths, views, manifest, existing, write = replay_fixture
    reference = existing.reference(manifest)
    if mutation == 'old-round':
        paths['joint']['fresh'] = root.parent / 'old-round/evaluation.json'
    elif mutation == 'duplicate':
        paths['joint']['fresh'] = paths['start']['fresh']
    elif mutation == 'heldout':
        value = views[str(paths['joint']['fresh'])]
        value['split'] = 'heldout'
        paths['joint']['fresh'].write_text(json.dumps(value), encoding='utf-8')
    elif mutation in ('manifest-tamper', 'manifest-late', 'fresh-mismatch'):
        value = json.loads(manifest.read_text(encoding='utf-8'))
        if mutation == 'manifest-late':
            value['frozen_utc'] = '2026-01-02T00:00:00+00:00'
        elif mutation == 'fresh-mismatch':
            value['populations']['fresh']['dataset']['dataset_sha256'] = '5' * 64
        else:
            value['selected_id'] = 'tampered'
        manifest.write_text(json.dumps(value), encoding='utf-8')
        if mutation != 'manifest-tamper':
            reference = existing.reference(manifest)
    elif mutation == 'source-drift':
        inventory = iter([{'fixture-source': SHA}, {'fixture-source': '2' * 64}])
        monkeypatch.setattr(existing, 'source_inventory', lambda: next(inventory))
    else:
        def mutate(path, report):
            Path(report['outputs']['predictions']['file']).write_text('tamper', encoding='utf-8')
            return report['summary']
        monkeypatch.setattr(existing, '_scene_regression_summary', mutate)
    with pytest.raises(ValueError):
        joint.verify_round_quality(paths, MODEL, round_root=root, configuration_manifest=reference)


def test_failed_quality_wrapper_requires_no_repeats_and_retains_start(replay_fixture):
    root, paths, views, manifest, existing, write = replay_fixture
    for tier in joint.TIERS[1:]:
        value = views[str(paths[tier]['anchor'])]
        value['summary']['candidate_stage']['safe_complete_coverage'] = 99
        paths[tier]['anchor'].write_text(json.dumps(value), encoding='utf-8')
    result = joint.verify_round_development(paths, {}, MODEL, round_root=root, configuration_manifest=existing.reference(manifest))
    assert not result['passed'] and result['selected_tier'] == 'start'
    assert result['latency_status'] == 'NOT_RUN'
    with pytest.raises(ValueError, match='quality-eligible'):
        joint.verify_round_development(paths, {'joint': {}}, MODEL, round_root=root, configuration_manifest=existing.reference(manifest))


def fixture_repeats(paths, views, write):
    return {tier: {population: [write(evidence(population == 'anchor',
            20 if population == 'anchor' and i % 2 == 0 else
            32 if population == 'anchor' else 16 if i % 2 == 0 else 20,
            66 if population == 'anchor' and i % 2 == 0 else
            64 if i % 2 == 0 else 100,
            'start' if i % 2 == 0 else tier, stamp=10 + i)) for i in range(6)]
            for population in joint.POPULATIONS} for tier in joint.TIERS[1:]}


def test_wrapper_replays_all_distinct_raw_repeats_and_returns_deterministic_bindable_proof(replay_fixture):
    root, paths, views, manifest, existing, write = replay_fixture
    repeated = fixture_repeats(paths, views, write)
    kwargs = dict(round_root=root, configuration_manifest=existing.reference(manifest))
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    first = joint.verify_round_development(paths, repeated, MODEL, **kwargs)
    assert first['passed'] and first['selected_tier'] == 'complete-exclusive'
    assert len(first['repeated_reports']['joint']['fresh']) == 6
    assert first['latency']['joint']['fresh']['candidate']['count'] == 180
    assert first == joint.verify_round_development(paths, repeated, MODEL, **kwargs)
    assert before == {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('mutation', ['report-reuse', 'output-reuse', 'missing-tier', 'missing-repeat', 'misordered', 'cross-round', 'semantic-drift', 'resource-tamper', 'handwritten-aggregate'])
def test_wrapper_rejects_reused_missing_or_unstable_original_repeat_evidence(replay_fixture, monkeypatch, mutation):
    root, paths, views, manifest, existing, write = replay_fixture
    repeated = fixture_repeats(paths, views, write)
    target = repeated['joint']['fresh'][1]
    report = views[str(target)]
    if mutation == 'report-reuse':
        repeated['joint']['fresh'][1] = paths['joint']['fresh']
    elif mutation == 'output-reuse':
        report['outputs']['resources'] = deepcopy(views[str(paths['joint']['fresh'])]['outputs']['resources'])
        target.write_text(json.dumps(report), encoding='utf-8')
    elif mutation == 'missing-tier':
        repeated.pop('joint')
    elif mutation == 'missing-repeat':
        repeated['joint']['fresh'].pop()
    elif mutation == 'misordered':
        rows = repeated['joint']['fresh']
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == 'cross-round':
        repeated['joint']['fresh'][1] = root.parent / 'old-round/evaluation.json'
    elif mutation == 'semantic-drift':
        from tools.RegionKaomoji import evaluate as scene
        original = scene.read_predictions
        def drift(path, jobs):
            result = original(path, jobs)
            if path == report['outputs']['predictions']['file']:
                result[0]['prediction']['models'][MODEL]['decisions'] = [{'reason': 'changed'}]
            return result
        monkeypatch.setattr(scene, 'read_predictions', drift)
    elif mutation == 'resource-tamper':
        Path(report['outputs']['resources']['file']).write_text('tampered', encoding='utf-8')
    else:
        repeated['joint']['fresh'] = {'pooled_p95': 1., 'passed': True}
    with pytest.raises(ValueError):
        joint.verify_round_development(paths, repeated, MODEL, round_root=root,
                                       configuration_manifest=existing.reference(manifest))


def test_unapproved_population_rejected_before_truth_or_replay(replay_fixture, monkeypatch):
    root, paths, views, manifest, existing, write = replay_fixture
    value = views[str(paths['joint']['fresh'])]
    value['pipeline_binding']['dataset']['dataset_sha256'] = '5' * 64
    paths['joint']['fresh'].write_text(json.dumps(value), encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('Unapproved truth/replay must not be reached')
    monkeypatch.setattr(existing, '_scene_regression_summary', forbidden)
    # Only exercise the affected report; earlier valid reports may replay.
    from tools.RegionKaomoji import evaluate as scene
    declared = json.loads(manifest.read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='predeclared'):
        joint._verified(paths['joint']['fresh'], MODEL, declared, 'joint', 'fresh', root, existing, scene, set())
