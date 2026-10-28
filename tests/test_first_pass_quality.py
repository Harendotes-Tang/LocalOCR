"""Synthetic gate fixtures exercise validation only, never OCR accuracy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.RegionKaomoji import first_pass_quality as position
from tools.RegionKaomoji import span_quality as span


MODEL = 'fixture-specialist'
SHA = '1' * 64
SOURCE = 'tools/RegionKaomoji/first_pass_quality.py'


def set_interval(value, stamp=None):
    resources = value['resources']
    start = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=stamp * 10000)
             if stamp is not None else datetime.fromisoformat(resources['started_utc']))
    duration = resources['total_warm_seconds'] + 5.
    completed = start + timedelta(seconds=duration)
    resources.update(started_utc=start.isoformat(), completed_utc=completed.isoformat(),
                     process_wall_seconds=duration)
    value['completed_utc'] = (completed + timedelta(seconds=1)).isoformat()
    return value


def evidence(anchor, exact, safe, tier='start', *, seconds=2., stamp=0):
    count = 244 if anchor else 256
    config = {'views': ['raw', 'edge-pad-4'], 'limits': {'maximum_candidates': 48},
              'candidate': {}, 'fusion': {}}
    config = position.configuration_tiers(config)[tier]
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
    return set_interval(value, stamp)


@pytest.fixture
def scored():
    return {tier: {'anchor': evidence(True, 20 if tier == 'start' else 32, 66 if tier == 'start' else 100, tier, stamp=2 * index),
                   'fresh': evidence(False, 16 if tier == 'start' else 20, 64 if tier == 'start' else 100, tier, stamp=2 * index + 1)}
            for index, tier in enumerate(position.TIERS)}


def quality(scored):
    return position.summarize_quality(scored, MODEL)


def repetitions(scored, tier, *, start_times=None, candidate_times=None):
    result = {}
    for population_index, population in enumerate(position.POPULATIONS):
        rows = []
        for index in range(6):
            role = 'start' if index % 2 == 0 else tier
            view = deepcopy(scored[role][population])
            stamp = 10 + population_index * 6 + index
            times = ((start_times if role == 'start' else candidate_times) or [2.] * 180)[(index // 2) * 60:(index // 2 + 1) * 60]
            from tools.PretrainedOcr.evaluate import percentiles
            view['resources']['warm'] = percentiles(times)
            view['resources']['total_warm_seconds'] = sum(times)
            set_interval(view, stamp)
            rows.append({'view': view, 'seconds': times, 'output_sha256': SHA})
        result[population] = rows
    return result


def test_quality_is_independent_of_single_report_latency(scored):
    for tier in position.TIERS[1:]:
        for value in scored[tier].values():
            value['resources']['warm']['p95_seconds'] = 100.
    verdict = quality(scored)
    assert verdict['quality_eligible_tiers'] == list(position.TIERS[1:])
    assert verdict['latency_status'] == 'NOT_RUN'
    assert verdict['selected_tier'] == 'start' and not verdict['passed']
    assert 'warm_p95_within_115_percent' not in verdict['tiers']['union']['anchor']['checks']


def test_earlier_failure_does_not_block_first_pass_quality(scored):
    scored['initial']['anchor']['summary']['candidate_stage']['safe_complete_coverage'] = 99
    assert quality(scored)['quality_eligible_tiers'] == ['local', 'union']


@pytest.mark.parametrize('population', position.POPULATIONS)
def test_total_gain_cannot_hide_previous_correct_loss(scored, population):
    rows = scored['union'][population]['target_correctness']
    rows[0]['correct'], rows[40]['correct'] = False, True
    assert not quality(scored)['tiers']['union']['passed']


@pytest.mark.parametrize('counter', span.BODY_COUNTERS + span.NEGATIVE_COUNTERS)
def test_six_protection_counters_are_hard_requirements(scored, counter):
    model = scored['union']['fresh']['summary']['models'][MODEL]
    model['body' if counter in span.BODY_COUNTERS else 'negative'][counter] = 1
    assert not quality(scored)['tiers']['union']['passed']


@pytest.mark.parametrize('mutation', ['missing-tier', 'extra-tier', 'missing-population', 'heldout', 'source', 'machine', 'different-config', 'anchor', 'fresh-alias', 'missing-position-source', 'malformed'])
def test_quality_rejects_malformed_cross_identity_evidence(scored, mutation):
    if mutation == 'missing-tier':
        scored.pop('union')
    elif mutation == 'extra-tier':
        scored['post-hoc'] = scored['union']
    elif mutation == 'missing-population':
        scored['union'].pop('fresh')
    elif mutation == 'heldout':
        scored['union']['fresh']['split'] = 'heldout'
    elif mutation == 'source':
        scored['union']['fresh']['pipeline_binding']['implementation'][SOURCE] = '2' * 64
    elif mutation == 'machine':
        scored['union']['fresh']['resources']['measurement_machine']['fingerprint_sha256'] = '2' * 64
    elif mutation == 'different-config':
        binding = scored['union']['fresh']['pipeline_binding']
        binding['config']['candidate'] = {'extra': True}
        binding['config_sha256'] = span._hash(binding['config'])
    elif mutation == 'anchor':
        for value in (v['anchor'] for v in scored.values()):
            value['pipeline_binding']['dataset']['dataset_sha256'] = '5' * 64
    elif mutation == 'fresh-alias':
        for value in (v['fresh'] for v in scored.values()):
            value['pipeline_binding']['dataset'] = deepcopy(scored['start']['anchor']['pipeline_binding']['dataset'])
    elif mutation == 'missing-position-source':
        for pair in scored.values():
            for value in pair.values():
                value['pipeline_binding']['implementation'].pop(SOURCE)
    else:
        scored['union']['anchor']['summary']['models'][MODEL]['isolated_kaomoji']['exact'] = True
    with pytest.raises(ValueError):
        quality(scored)


def test_pool_all_samples_instead_of_fastest_run_or_average_p95(scored):
    times = [2.] * 171 + [6.] * 9
    repeats = repetitions(scored, 'union', candidate_times=times)
    verdict = position.summarize_repeated_latency(scored, 'union', repeats, MODEL)
    # NumPy's existing linear percentile interpolates across the 95% boundary.
    assert verdict['anchor']['candidate']['count'] == 180
    assert verdict['anchor']['candidate']['p95_seconds'] == pytest.approx(2.2)
    assert verdict['passed']
    assert verdict['pooled_candidate']['count'] == 360


def test_actual_pooled_ratio_fails_without_drift_deduction(scored):
    verdict = position.summarize_repeated_latency(scored, 'union', repetitions(scored, 'union', candidate_times=[2.301] * 180), MODEL)
    assert verdict['anchor']['ratio'] > 1.15 and not verdict['passed']


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'order', 'bad-order-stamp', 'quality-drift', 'output-drift', 'times-missing', 'times-nan', 'times-count', 'times-warm', 'times-total', 'different-config'])
def test_repeat_gate_rejects_missing_unstable_or_unverified_samples(scored, mutation):
    repeats = repetitions(scored, 'union')
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
    signatures = {population: {'start': SHA, 'union': SHA} for population in position.POPULATIONS}
    with pytest.raises(ValueError):
        position.summarize_repeated_latency(scored, 'union', repeats, MODEL, scored_output_sha256=signatures)


def test_quality_failure_cannot_obtain_latency_claim(scored):
    scored['union']['anchor']['summary']['candidate_stage']['safe_complete_coverage'] = 99
    with pytest.raises(ValueError, match='quality'):
        position.summarize_repeated_latency(scored, 'union', repetitions(scored, 'union'), MODEL)


@pytest.mark.parametrize('population', position.POPULATIONS)
def test_repeat_phase_waits_for_all_predeclared_scoring(scored, population):
    repeats = repetitions(scored, 'initial')
    set_interval(scored['union'][population], 30)
    with pytest.raises(ValueError, match='chronological'):
        position.summarize_repeated_latency(scored, 'initial', repeats, MODEL)


@pytest.mark.parametrize('population,counter,value', [('anchor', 'exact', 29), ('fresh', 'exact', 16), ('anchor', 'safe', 99)])
def test_anchor_floors_and_each_population_strict_gain(scored, population, counter, value):
    report = scored['union'][population]
    if counter == 'exact':
        report['summary']['models'][MODEL]['isolated_kaomoji']['exact'] = value
        for i, target in enumerate(report['target_correctness']):
            target['correct'] = i < value
    else:
        report['summary']['candidate_stage']['safe_complete_coverage'] = value
    assert not quality(scored)['tiers']['union']['passed']


def test_ranking_uses_fresh_then_anchor_then_combined_p95_then_declared_order(scored):
    q = quality(scored)
    timing = {tier: {'passed': True, 'pooled_candidate': {'p95_seconds': 2.}}
              for tier in position.TIERS[1:]}
    assert position.select_tier(q, timing)['selected_tier'] == 'initial'
    timing['local']['pooled_candidate']['p95_seconds'] = 1.9
    assert position.select_tier(q, timing)['selected_tier'] == 'local'
    q['tiers']['union']['anchor']['final_exact'] += 1
    assert position.select_tier(q, timing)['selected_tier'] == 'union'
    q['tiers']['initial']['fresh_development']['final_exact'] += 1
    assert position.select_tier(q, timing)['selected_tier'] == 'initial'
    for item in timing.values():
        item['passed'] = False
    result = position.select_tier(q, timing)
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
    assert position.prediction_identity(records) == position.prediction_identity(mutated)
    mutated[0]['prediction']['models'][MODEL]['views']['c'][0]['score'] = .8
    assert position.prediction_identity(records) != position.prediction_identity(mutated)


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
    previous = tmp_path.parent / (tmp_path.name + '-predecessor')
    retained = previous / 'routing/final.json'
    retained.parent.mkdir(parents=True)
    retained.write_text(json.dumps(scored['start']['anchor']['pipeline_binding']['config']), encoding='utf-8')
    old_manifest = retained.with_name('configuration-manifest.json')
    old_manifest.write_text(json.dumps({'schema_version': 'localocr.position-configurations.v1',
        'selected_id': MODEL, 'configurations': {'start': scored['start']['anchor']['pipeline_binding']['config']},
        'models': scored['start']['anchor']['pipeline_binding']['models'],
        'baseline': scored['start']['anchor']['pipeline_binding']['baseline']}), encoding='utf-8')
    monkeypatch.setattr(existing.rp, 'OWNED_POSITION', SimpleNamespace(root=previous))
    monkeypatch.setattr(scene, 'configuration', lambda path, classifier=None: span._read(path))
    manifest = {'schema_version': 'localocr.first-pass-configurations.v1', 'frozen_utc': '2025-12-31T00:00:00+00:00',
        'start_configuration': existing.reference(retained),
        'start_configuration_manifest': existing.reference(old_manifest),
        'selected_id': MODEL, 'configurations': {tier: pair['anchor']['pipeline_binding']['config'] for tier, pair in scored.items()},
        'populations': {population: {key: views[str(paths['start'][population])]['pipeline_binding'][key] for key in ('dataset', 'inputs')}
                        for population in position.POPULATIONS}, 'source_inventory': {'fixture-source': SHA},
        'models': scored['start']['anchor']['pipeline_binding']['models'],
        'baseline': scored['start']['anchor']['pipeline_binding']['baseline']}
    manifest_path = tmp_path / 'configuration-manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(existing.rp, 'profile_for', lambda path: SimpleNamespace(name='first-pass-position-20261027', root=tmp_path))
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
    result = position.verify_round_quality(paths, MODEL, round_root=root, configuration_manifest=existing.reference(manifest))
    assert result['quality_eligible_tiers'] == list(position.TIERS[1:])
    assert result['reports']['union']['fresh']['report'] == existing.reference(paths['union']['fresh'])
    assert result['source_inventory'] == {'fixture-source': SHA}
    assert set(result['reports']['union']['fresh']['worker_interval']) == {
        'started_utc', 'completed_utc', 'process_wall_seconds'}


@pytest.mark.parametrize('mutation', ['old-round', 'duplicate', 'heldout', 'manifest-tamper', 'manifest-late', 'fresh-mismatch', 'source-drift', 'output-tamper'])
def test_wrapper_rejects_unsafe_paths_manifest_or_output_changes(replay_fixture, monkeypatch, mutation):
    root, paths, views, manifest, existing, write = replay_fixture
    reference = existing.reference(manifest)
    if mutation == 'old-round':
        paths['union']['fresh'] = root.parent / 'old-round/evaluation.json'
    elif mutation == 'duplicate':
        paths['union']['fresh'] = paths['start']['fresh']
    elif mutation == 'heldout':
        value = views[str(paths['union']['fresh'])]
        value['split'] = 'heldout'
        paths['union']['fresh'].write_text(json.dumps(value), encoding='utf-8')
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
        position.verify_round_quality(paths, MODEL, round_root=root, configuration_manifest=reference)


def test_failed_quality_wrapper_requires_no_repeats_and_retains_start(replay_fixture):
    root, paths, views, manifest, existing, write = replay_fixture
    for tier in position.TIERS[1:]:
        value = views[str(paths[tier]['anchor'])]
        value['summary']['candidate_stage']['safe_complete_coverage'] = 99
        paths[tier]['anchor'].write_text(json.dumps(value), encoding='utf-8')
    result = position.verify_round_development(paths, {}, MODEL, round_root=root, configuration_manifest=existing.reference(manifest))
    assert not result['passed'] and result['selected_tier'] == 'start'
    assert result['latency_status'] == 'NOT_RUN'
    with pytest.raises(ValueError, match='quality-eligible'):
        position.verify_round_development(paths, {'union': {}}, MODEL, round_root=root, configuration_manifest=existing.reference(manifest))


def fixture_repeats(paths, views, write):
    return {tier: {population: [write(evidence(population == 'anchor',
            20 if population == 'anchor' and i % 2 == 0 else
            32 if population == 'anchor' else 16 if i % 2 == 0 else 20,
            66 if population == 'anchor' and i % 2 == 0 else
            64 if i % 2 == 0 else 100,
            'start' if i % 2 == 0 else tier, stamp=10 + tier_index * 12 + population_index * 6 + i)) for i in range(6)]
            for population_index, population in enumerate(position.POPULATIONS)}
            for tier_index, tier in enumerate(position.TIERS[1:])}


def test_wrapper_replays_all_distinct_raw_repeats_and_returns_deterministic_bindable_proof(replay_fixture):
    root, paths, views, manifest, existing, write = replay_fixture
    repeated = fixture_repeats(paths, views, write)
    kwargs = dict(round_root=root, configuration_manifest=existing.reference(manifest))
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    first = position.verify_round_development(paths, repeated, MODEL, **kwargs)
    assert first['passed'] and first['selected_tier'] == 'initial'
    assert len(first['repeated_reports']['union']['fresh']) == 6
    assert first['latency']['union']['fresh']['candidate']['count'] == 180
    assert first == position.verify_round_development(paths, repeated, MODEL, **kwargs)
    assert before == {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('mutation', ['report-reuse', 'output-reuse', 'missing-tier', 'missing-repeat', 'misordered', 'cross-round', 'semantic-drift', 'resource-tamper', 'handwritten-aggregate'])
def test_wrapper_rejects_reused_missing_or_unstable_original_repeat_evidence(replay_fixture, monkeypatch, mutation):
    root, paths, views, manifest, existing, write = replay_fixture
    repeated = fixture_repeats(paths, views, write)
    target = repeated['union']['fresh'][1]
    report = views[str(target)]
    if mutation == 'report-reuse':
        repeated['union']['fresh'][1] = paths['union']['fresh']
    elif mutation == 'output-reuse':
        report['outputs']['resources'] = deepcopy(views[str(paths['union']['fresh'])]['outputs']['resources'])
        target.write_text(json.dumps(report), encoding='utf-8')
    elif mutation == 'missing-tier':
        repeated.pop('union')
    elif mutation == 'missing-repeat':
        repeated['union']['fresh'].pop()
    elif mutation == 'misordered':
        rows = repeated['union']['fresh']
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == 'cross-round':
        repeated['union']['fresh'][1] = root.parent / 'old-round/evaluation.json'
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
        repeated['union']['fresh'] = {'pooled_p95': 1., 'passed': True}
    with pytest.raises(ValueError):
        position.verify_round_development(paths, repeated, MODEL, round_root=root,
                                       configuration_manifest=existing.reference(manifest))


def test_unapproved_population_rejected_before_truth_or_replay(replay_fixture, monkeypatch):
    root, paths, views, manifest, existing, write = replay_fixture
    value = views[str(paths['union']['fresh'])]
    value['pipeline_binding']['dataset']['dataset_sha256'] = '5' * 64
    paths['union']['fresh'].write_text(json.dumps(value), encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('Unapproved truth/replay must not be reached')
    monkeypatch.setattr(existing, '_scene_regression_summary', forbidden)
    # Only exercise the affected report; earlier valid reports may replay.
    from tools.RegionKaomoji import evaluate as scene
    declared = json.loads(manifest.read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='predeclared'):
        position._verified(paths['union']['fresh'], MODEL, declared, 'union', 'fresh', root, existing, scene, set())


def test_round_sources_and_tiers_cannot_inherit_predecessor_globals(scored):
    from tools.RegionKaomoji import position_quality as predecessor
    assert position.ROUND == 'first-pass-position-20261027'
    assert position.SOURCE == 'tools/RegionKaomoji/first_pass_quality.py'
    assert position.TIERS == ('start', 'initial', 'local', 'union')
    assert position.QUALITY_SCHEMA != predecessor.QUALITY_SCHEMA
    assert position.DEVELOPMENT_SCHEMA != predecessor.DEVELOPMENT_SCHEMA
    assert quality(scored)['schema_version'] == 'localocr.first-pass-development-quality.v1'
    assert predecessor.TIERS == ('start', 'ownership', 'components', 'ctc', 'union')
    for pair in scored.values():
        for value in pair.values():
            implementation = value['pipeline_binding']['implementation']
            implementation[predecessor.SOURCE] = implementation.pop(position.SOURCE)
    with pytest.raises(ValueError, match='First-pass quality source'):
        quality(scored)


@pytest.mark.parametrize('mutation', ['mode', 'initial', 'old-guard', 'budget', 'extra-option', 'start-mode', 'numeric-type', 'missing-initial-trace', 'missing-local-proof'])
def test_four_tiers_allow_only_the_predeclared_strategy_differences(scored, mutation):
    for population in position.POPULATIONS:
        config = scored['union'][population]['pipeline_binding']['config']
        if mutation == 'mode':
            config['candidate']['position_spans']['mode'] = 'ctc'
        elif mutation == 'initial':
            config['fusion']['owned_ownership']['strategy'] = 'bbox-only'
        elif mutation == 'old-guard':
            config['fusion']['protect_body_letters_numbers'] = False
        elif mutation == 'budget':
            config['candidate']['maximum_candidates'] = 49
        elif mutation == 'extra-option':
            config['candidate']['position_spans']['threshold'] = .1
        elif mutation == 'start-mode':
            config = scored['start'][population]['pipeline_binding']['config']
            config['candidate']['position_spans'] = {'strategy': 'ctc-components-v1', 'mode': 'union'}
        elif mutation == 'missing-initial-trace':
            config['candidate']['position_spans'].pop('initial_trace')
        elif mutation == 'missing-local-proof':
            config['candidate'].pop('local_ownership')
        else:
            config['limits']['maximum_candidates'] = 48.0
        for tier in position.TIERS:
            binding = scored[tier][population]['pipeline_binding']
            binding['config_sha256'] = span._hash(binding['config'])
    with pytest.raises(ValueError, match='configuration|retained start'):
        quality(scored)


def test_configuration_factory_keeps_start_independent_and_exact_options():
    start = {'candidate': {'padding': 3}, 'fusion': {'protect_body': True}, 'views': ['raw', 'edge-pad-4']}
    unchanged = deepcopy(start)
    result = position.configuration_tiers(start)
    assert start == unchanged and result['start'] == unchanged
    assert tuple(result) == position.TIERS
    assert result['initial']['candidate']['position_spans'] == {
        'strategy': 'ctc-components-v1', 'mode': 'ctc', 'initial_trace': 'horizontal-v1'}
    assert 'local_ownership' not in result['initial']['candidate']
    assert result['local']['candidate']['position_spans'] == {
        'strategy': 'ctc-components-v1', 'mode': 'components'}
    assert result['local']['candidate']['local_ownership'] == {'strategy': 'pixel-partition-v1'}
    result['union']['candidate']['padding'] = 9
    assert result['start'] == unchanged and result['local']['candidate']['padding'] == 3


@pytest.mark.parametrize('mutation', ['missing-ref', 'wrong-ref', 'different-start', 'old-schema'])
def test_manifest_rejects_unbound_predecessor_before_replay(replay_fixture, monkeypatch, mutation):
    root, paths, views, manifest, existing, write = replay_fixture
    value = json.loads(manifest.read_text(encoding='utf-8'))
    if mutation == 'missing-ref':
        value.pop('start_configuration')
    elif mutation == 'wrong-ref':
        value['start_configuration']['sha256'] = '5' * 64
    elif mutation == 'different-start':
        # A different retained file must fail even if all four differences agree.
        prior = Path(value['start_configuration']['file'])
        prior.write_text('{"not": "the retained start"}', encoding='utf-8')
        value['start_configuration'] = existing.reference(prior)
    else:
        value['schema_version'] = 'localocr.joint-configurations.v1'
    manifest.write_text(json.dumps(value), encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('No original mask/truth replay before manifest admission')
    monkeypatch.setattr(existing, '_scene_regression_summary', forbidden)
    with pytest.raises(ValueError):
        position.verify_round_quality(paths, MODEL, round_root=root,
                                      configuration_manifest=existing.reference(manifest))


def test_predecessor_config_is_rechecked_after_raw_replay(replay_fixture, monkeypatch):
    root, paths, views, manifest, existing, write = replay_fixture
    value = json.loads(manifest.read_text(encoding='utf-8'))
    original = existing._scene_regression_summary
    def changed(path, report):
        result = original(path, report)
        Path(value['start_configuration']['file']).write_text('changed', encoding='utf-8')
        return result
    monkeypatch.setattr(existing, '_scene_regression_summary', changed)
    with pytest.raises(ValueError, match='predecessor configuration changed'):
        position.verify_round_quality(paths, MODEL, round_root=root,
                                      configuration_manifest=existing.reference(manifest))


def test_predecessor_start_uses_frozen_normalized_defaults_not_raw_file_equality(replay_fixture, monkeypatch):
    from tools.RegionKaomoji import evaluate as scene
    root, paths, views, manifest, existing, write = replay_fixture
    value = json.loads(manifest.read_text(encoding='utf-8'))
    raw = Path(value['start_configuration']['file'])
    raw.write_text('{"raw": "omits defaults and classifier"}', encoding='utf-8')
    value['start_configuration'] = existing.reference(raw)
    manifest.write_text(json.dumps(value), encoding='utf-8')
    calls = []
    def normalize(path, classifier=None):
        assert path == raw
        calls.append(classifier)
        return deepcopy(value['configurations']['start'])
    monkeypatch.setattr(scene, 'configuration', normalize)
    proof = position.verify_round_quality(paths, MODEL, round_root=root,
                                          configuration_manifest=existing.reference(manifest))
    assert calls == [None]
    assert proof['start_configuration'] == value['start_configuration']
    assert proof['start_configuration_manifest'] == value['start_configuration_manifest']


@pytest.mark.parametrize('mutation', ['missing-ref', 'wrong-ref', 'old-model', 'wrong-schema'])
def test_predecessor_full_config_binding_is_required_before_replay(replay_fixture, monkeypatch, mutation):
    root, paths, views, manifest, existing, write = replay_fixture
    value = json.loads(manifest.read_text(encoding='utf-8'))
    if mutation == 'missing-ref':
        value.pop('start_configuration_manifest')
    elif mutation == 'wrong-ref':
        value['start_configuration_manifest']['sha256'] = '5' * 64
    else:
        old_path = Path(value['start_configuration_manifest']['file'])
        old = json.loads(old_path.read_text(encoding='utf-8'))
        if mutation == 'old-model':
            old['models'][0]['id'] = 'different-specialist'
        else:
            old['schema_version'] = position.CONFIGURATION_SCHEMA
        old_path.write_text(json.dumps(old), encoding='utf-8')
        value['start_configuration_manifest'] = existing.reference(old_path)
    manifest.write_text(json.dumps(value), encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('No replay before predecessor admission')
    monkeypatch.setattr(existing, '_scene_regression_summary', forbidden)
    with pytest.raises(ValueError):
        position.verify_round_quality(paths, MODEL, round_root=root,
                                      configuration_manifest=existing.reference(manifest))


def test_both_population_samples_are_pooled_for_the_full_latency_gate(scored):
    repeats = repetitions(scored, 'union', start_times=[2.] * 180, candidate_times=[2.2] * 180)
    verdict = position.summarize_repeated_latency(scored, 'union', repeats, MODEL)
    assert verdict['pooled_start']['count'] == verdict['pooled_candidate']['count'] == 360
    assert verdict['pooled_ratio'] == pytest.approx(1.1)
    assert verdict['pooled_passed'] and verdict['passed']


def test_combined_pool_cannot_be_replaced_by_separate_population_p95(scored):
    from tools.PretrainedOcr.evaluate import percentiles
    # Both per-population p95 ratios are 1.0, but their rare slow samples
    # cross the combined distribution's 95% boundary in different places.
    repeats = repetitions(scored, 'union')
    samples = {
        'anchor': {'start': [1.] * 170 + [100.] * 10, 'union': [1.] * 170 + [100.] * 10},
        'fresh': {'start': [1.] * 180, 'union': [1.] * 172 + [100.] * 8},
    }
    for population, rows in repeats.items():
        for index, row in enumerate(rows):
            role = 'start' if index % 2 == 0 else 'union'
            seconds = samples[population][role][(index // 2) * 60:(index // 2 + 1) * 60]
            row['seconds'] = seconds
            row['view']['resources']['warm'] = percentiles(seconds)
            row['view']['resources']['total_warm_seconds'] = sum(seconds)
            set_interval(row['view'])
    verdict = position.summarize_repeated_latency(scored, 'union', repeats, MODEL)
    assert verdict['anchor']['passed']
    assert verdict['fresh']['passed']
    assert verdict['pooled_ratio'] > 1.15 and not verdict['pooled_passed']
    assert not verdict['passed']


@pytest.mark.parametrize('mutation', ['missing-start', 'missing-completed', 'naive', 'reverse', 'wall-drift', 'report-before-worker'])
def test_quality_requires_legal_owner_resource_intervals(scored, mutation):
    view = scored['union']['fresh']
    resources = view['resources']
    if mutation == 'missing-start':
        resources.pop('started_utc')
    elif mutation == 'missing-completed':
        resources.pop('completed_utc')
    elif mutation == 'naive':
        resources['started_utc'] = '2026-01-01T00:00:00'
    elif mutation == 'reverse':
        resources['completed_utc'] = resources['started_utc']
    elif mutation == 'wall-drift':
        resources['process_wall_seconds'] += 1.
    else:
        view['completed_utc'] = resources['started_utc']
    with pytest.raises(ValueError):
        quality(scored)


def test_score_start_must_follow_the_predeclared_configuration_freeze(replay_fixture, monkeypatch):
    root, paths, views, manifest, existing, write = replay_fixture
    path = paths['start']['anchor']
    view = views[str(path)]
    start = datetime(2025, 12, 30, 23, 59, 0, tzinfo=timezone.utc)
    # Ends after declaration, but its actual owner interval began before it.
    view['resources']['started_utc'] = start.isoformat()
    view['resources']['completed_utc'] = (start + timedelta(seconds=125)).isoformat()
    view['completed_utc'] = (start + timedelta(seconds=126)).isoformat()
    path.write_text(json.dumps(view), encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('Before-freeze owner interval must fail before truth replay')
    monkeypatch.setattr(existing, '_scene_regression_summary', forbidden)
    with pytest.raises(ValueError, match='start|interval'):
        position.verify_round_quality(paths, MODEL, round_root=root,
                                      configuration_manifest=existing.reference(manifest))


def test_score_intervals_cannot_overlap_even_if_completed_order_is_valid(scored):
    previous = scored['start']['anchor']
    current = scored['start']['fresh']
    started = datetime.fromisoformat(previous['resources']['started_utc']) + timedelta(seconds=1)
    current['resources']['started_utc'] = started.isoformat()
    current['resources']['completed_utc'] = (started + timedelta(seconds=125)).isoformat()
    current['completed_utc'] = (started + timedelta(seconds=126)).isoformat()
    with pytest.raises(ValueError, match='overlap|serial'):
        quality(scored)


def test_repeat_actual_start_must_follow_previous_completion(scored):
    repeats = repetitions(scored, 'union')
    first, second = repeats['anchor'][:2]
    # Preserve report completion ordering while running the candidate early.
    early = datetime.fromisoformat(first['view']['resources']['started_utc']) + timedelta(seconds=1)
    second['view']['resources']['started_utc'] = early.isoformat()
    second['view']['resources']['completed_utc'] = (early + timedelta(seconds=125)).isoformat()
    second['view']['completed_utc'] = (early + timedelta(seconds=126)).isoformat()
    with pytest.raises(ValueError, match='chronological|overlap|serial'):
        position.summarize_repeated_latency(scored, 'union', repeats, MODEL)


def test_repeat_populations_must_share_one_serial_worker_timeline(scored):
    repeats = repetitions(scored, 'union')
    for anchor, fresh in zip(repeats['anchor'], repeats['fresh']):
        for key in ('started_utc', 'completed_utc', 'process_wall_seconds'):
            fresh['view']['resources'][key] = anchor['view']['resources'][key]
        fresh['view']['completed_utc'] = anchor['view']['completed_utc']
    with pytest.raises(ValueError, match='overlap|serial'):
        position.summarize_repeated_latency(scored, 'union', repeats, MODEL)


def test_repeats_wait_for_scored_report_scoring_after_owner_exit(scored):
    repeats = repetitions(scored, 'union')
    first_repeat_start = datetime.fromisoformat(repeats['anchor'][0]['view']['resources']['started_utc'])
    scored['union']['fresh']['completed_utc'] = (first_repeat_start + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match='chronological'):
        position.summarize_repeated_latency(scored, 'union', repeats, MODEL)


def test_next_repeat_waits_for_previous_report_scoring_after_owner_exit(scored):
    repeats = repetitions(scored, 'union')
    first, second = repeats['anchor'][:2]
    second_start = datetime.fromisoformat(second['view']['resources']['started_utc'])
    first['view']['completed_utc'] = (second_start + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match='chronological'):
        position.summarize_repeated_latency(scored, 'union', repeats, MODEL)


def test_repeat_tiers_cannot_overlap_separate_eligible_runs(replay_fixture):
    root, paths, views, manifest, existing, write = replay_fixture
    repeated = fixture_repeats(paths, views, write)
    for population in position.POPULATIONS:
        for reference_path, changed_path in zip(repeated['initial'][population], repeated['local'][population]):
            reference, changed = views[str(reference_path)], views[str(changed_path)]
            for key in ('started_utc', 'completed_utc', 'process_wall_seconds'):
                changed['resources'][key] = reference['resources'][key]
            changed['completed_utc'] = reference['completed_utc']
            changed_path.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='overlap|serial'):
        position.verify_round_development(paths, repeated, MODEL, round_root=root,
                                          configuration_manifest=existing.reference(manifest))


def test_cross_population_post_worker_scoring_cannot_overlap_the_next_owner(scored):
    repeats = repetitions(scored, 'union')
    earlier = repeats['anchor'][-1]['view']
    later = repeats['fresh'][0]['view']
    assert datetime.fromisoformat(earlier['resources']['completed_utc']) < datetime.fromisoformat(later['resources']['started_utc'])
    earlier['completed_utc'] = (datetime.fromisoformat(later['resources']['started_utc']) + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match='overlap|serial'):
        position.summarize_repeated_latency(scored, 'union', repeats, MODEL)
