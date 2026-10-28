"""Gate mechanics only: fixture verdicts never represent OCR quality evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.RegionKaomoji import span_quality as quality


MODEL = 'fixture-specialist'
SHA = '1' * 64


def bind_config(value):
    binding = value['pipeline_binding']
    payload = json.dumps(binding['config'], sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(',', ':')).encode('utf-8')
    binding['config_sha256'] = hashlib.sha256(payload).hexdigest()


def evidence(*, anchor, exact, safe, p95=2.):
    count = 244 if anchor else 20
    dataset = {'dataset_sha256': quality.ANCHOR_DATASET_SHA256 if anchor else '2' * 64,
               'split_sha256': {'dev': quality.ANCHOR_DEV_SHA256 if anchor else '3' * 64,
                                'heldout': '4' * 64},
               'counts': {'dev': {'scenes': 2, 'faces': count}}}
    value = {'schema_version': 'localocr.region-research.v1', 'split': 'dev',
            'actual_cpu_inference': True, 'independent_heldout': False,
            'ground_truth_inference_input': False,
            'promote': False, 'enabled': False,
            'pipeline_binding': {'dataset': dataset, 'inputs': {'sha256': SHA},
                 'models': [{'id': MODEL, 'candidate_sha256': SHA, 'model_sha256': SHA}],
                 'baseline': {'manifest_sha256': SHA}, 'classifier': {'sha256': SHA},
                 'implementation': {'tools/RegionKaomoji/span_quality.py': SHA},
                 'config': {'views': ['raw', 'edge-pad-4'], 'limits': {'maximum_candidates': 48}},
                 'promote': False, 'enabled': False},
            'summary': {'scenes': 2, 'candidate_stage': {'faces': count, 'safe_complete_coverage': safe},
                 'models': {MODEL: {'isolated_kaomoji': {'count': count, 'exact': exact},
                     'body': {'regions': 8, **dict.fromkeys(quality.BODY_COUNTERS, 0)},
                     'negative': {'negative_scenes': 2, 'graphic_regions': 2,
                                  **dict.fromkeys(quality.NEGATIVE_COUNTERS, 0)}}}},
            'resources': {'warm': {'count': 2, 'p50_seconds': 1., 'p95_seconds': p95},
                 'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2,
                 'inter_op_threads': 1, 'actual_baseline_detection': True,
                 'actual_specialist_models': 1, 'no_python_children_or_socket_io': True,
                 'capture': 'successful-owned-subprocess', 'worker_exit_code': 0,
                 'worker_pid': 123, 'warmup_scenes': 3, 'total_warm_seconds': 4.,
                 'peak_rss_bytes': 1000, 'process_wall_seconds': 5.,
                 'measurement_machine': {'schema_version': 'localocr.cpu-measurement-machine.v1',
                                         'fingerprint_sha256': SHA},
                 'baseline_runtime': {'schema_version': 'fixture-runtime'}},
            'target_correctness': [{'scene_id': 'fixture-scene', 'truth_id': str(i),
                                    'correct': i < exact} for i in range(count)]}
    bind_config(value)
    return value


@pytest.fixture
def four():
    return [evidence(anchor=True, exact=20, safe=66),
            evidence(anchor=True, exact=32, safe=100, p95=2.3),
            evidence(anchor=False, exact=2, safe=10),
            evidence(anchor=False, exact=5, safe=15, p95=2.2)]


def run(four):
    return quality.summarize_gate(*four, MODEL)


def set_value(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def test_pure_fixture_gate_checks_approved_thresholds_without_enabling_anything(four):
    before = deepcopy(four)
    result = run(four)
    assert result['passed']
    assert result['anchor']['lost_targets'] == 0
    assert result['anchor']['warm_p95_ratio'] == 1.15
    assert result['anchor']['final_safe_complete'] == 100
    assert result['fresh_development']['exact_gain'] == 3
    assert result['evidence_kind'].startswith('pure-gate-summary')
    assert all(result[key] is False for key in ('promote', 'enabled', 'production_enabled', 'app_acceptance'))
    assert before == four


def test_higher_total_does_not_hide_a_previously_correct_loss(four):
    records = four[1]['target_correctness']
    records[0]['correct'], records[32]['correct'] = False, True
    result = run(four)
    assert result['anchor']['exact_gain'] == 12
    assert result['anchor']['lost_targets'] == 1
    assert result['anchor']['checks']['all_start_correct_retained'] is False
    assert result['passed'] is False


@pytest.mark.parametrize('slot,path,value', [
    (1, ('summary', 'candidate_stage', 'safe_complete_coverage'), 99),
    (1, ('resources', 'warm', 'p95_seconds'), 2.300001),
    (3, ('resources', 'warm', 'p95_seconds'), 2.300001),
    *[(1, ('summary', 'models', MODEL, 'body', name), 1) for name in quality.BODY_COUNTERS],
    *[(3, ('summary', 'models', MODEL, 'negative', name), 1) for name in quality.NEGATIVE_COUNTERS],
])
def test_well_formed_failed_measurements_return_explicit_failure(four, slot, path, value):
    set_value(four[slot], path, value)
    assert run(four)['passed'] is False


@pytest.mark.parametrize('slot,exact', [(1, 29), (1, 20), (3, 2)])
def test_anchor_floor_and_strict_improvement_are_not_ties(four, slot, exact):
    four[slot]['summary']['models'][MODEL]['isolated_kaomoji']['exact'] = exact
    for i, record in enumerate(four[slot]['target_correctness']):
        record['correct'] = i < exact
    assert run(four)['passed'] is False


@pytest.mark.parametrize('slot,path,value', [
    (0, ('split',), 'heldout'),
    (2, ('split',), 'heldout-regression'),
    (0, ('independent_heldout',), True),
    (1, ('promote',), True),
    (1, ('pipeline_binding', 'enabled'), True),
    (0, ('actual_cpu_inference',), False),
    (0, ('ground_truth_inference_input',), True),
    (0, ('schema_version',), 'localocr.oracle-diagnostic.v1'),
    (0, ('resources', 'warm', 'p95_seconds'), 0),
    (0, ('resources', 'warm', 'p95_seconds'), float('nan')),
    (0, ('resources', 'warm', 'p95_seconds'), float('inf')),
    (0, ('resources', 'warm', 'p50_seconds'), -1),
    (0, ('resources', 'warm', 'p50_seconds'), 3),
    (0, ('resources', 'warm', 'count'), 0),
    (0, ('resources', 'warm', 'count'), True),
    (0, ('resources', 'intra_op_threads'), True),
    (0, ('resources', 'providers'), ['CUDAExecutionProvider']),
    (0, ('resources', 'capture'), 'simulated'),
    (0, ('resources', 'measurement_machine'), {}),
    (0, ('resources', 'peak_rss_bytes'), float('nan')),
    (0, ('summary', 'models', MODEL, 'body', 'regions'), 0),
    (0, ('summary', 'models', MODEL, 'negative', 'graphic_regions'), 0),
    (0, ('summary', 'models', MODEL, 'negative', 'negative_scenes'), 0),
    (0, ('summary', 'models', MODEL, 'body', 'erroneous_rewrites'), None),
    (0, ('summary', 'models', MODEL, 'negative', 'new_nonempty_scenes'), False),
    (0, ('summary', 'candidate_stage', 'safe_complete_coverage'), 245),
    (0, ('summary', 'models', MODEL, 'isolated_kaomoji', 'exact'), 245),
    (0, ('pipeline_binding', 'implementation'), {}),
])
def test_missing_invalid_or_fabricated_measurements_raise(four, slot, path, value):
    set_value(four[slot], path, value)
    with pytest.raises(ValueError):
        run(four)


@pytest.mark.parametrize('path,value', [
    (('pipeline_binding', 'models', 0, 'model_sha256'), '5' * 64),
    (('pipeline_binding', 'baseline', 'manifest_sha256'), '5' * 64),
    (('pipeline_binding', 'implementation', 'tools/RegionKaomoji/span_quality.py'), '5' * 64),
    (('pipeline_binding', 'classifier', 'sha256'), '5' * 64),
    (('resources', 'measurement_machine', 'fingerprint_sha256'), '5' * 64),
    (('resources', 'baseline_runtime', 'schema_version'), 'different-runtime'),
    (('pipeline_binding', 'config', 'views'), ['raw']),
])
def test_unequal_models_machine_sources_or_measurement_settings_raise(four, path, value):
    set_value(four[1], path, value)
    bind_config(four[1])
    with pytest.raises(ValueError, match='bindings differ'):
        run(four)


def test_start_and_final_may_differ_but_each_strategy_is_identical_across_datasets(four):
    for slot in (1, 3):
        four[slot]['pipeline_binding']['config']['candidate'] = {'line_spans': {'enabled': True}}
        bind_config(four[slot])
    assert four[0]['pipeline_binding']['config_sha256'] != four[1]['pipeline_binding']['config_sha256']
    assert run(four)['passed']


@pytest.mark.parametrize('slot', [0, 1, 2, 3])
def test_each_dataset_cannot_use_a_different_complete_strategy_to_pass_its_own_gate(four, slot):
    # Views, limits, model and source are unchanged, and each quality pair still
    # passes; only a complete strategy check prevents this mismatched comparison.
    four[slot]['pipeline_binding']['config']['candidate'] = {'line_spans': {'enabled': True}}
    bind_config(four[slot])
    with pytest.raises(ValueError, match='Cross-dataset starting/final full configuration'):
        run(four)


def test_configuration_cannot_hide_different_strategy_behind_a_copied_digest(four):
    four[1]['pipeline_binding']['config']['candidate'] = {'local_ownership': {'enabled': True}}
    with pytest.raises(ValueError, match='Full configuration digest differs'):
        run(four)


def test_pairs_require_same_data_targets_and_protection_population(four):
    four[1]['target_correctness'][-1]['truth_id'] = 'other'
    with pytest.raises(ValueError, match='populations differ'):
        run(four)
    four[1] = evidence(anchor=True, exact=32, safe=100)
    four[1]['summary']['models'][MODEL]['body']['regions'] = 9
    with pytest.raises(ValueError, match='populations differ'):
        run(four)


def test_correctness_cannot_be_incomplete_duplicated_or_disagree_with_exact(four):
    four[1]['target_correctness'][0]['correct'] = False
    with pytest.raises(ValueError, match='strict exact'):
        run(four)
    four[1]['target_correctness'][0]['correct'] = True
    four[1]['target_correctness'][-1] = four[1]['target_correctness'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        run(four)
    four[1]['target_correctness'].pop()
    with pytest.raises(ValueError, match='Every measured face'):
        run(four)


def test_anchor_is_pinned_and_new_development_is_distinct(four):
    for value in four[:2]:
        value['pipeline_binding']['dataset']['dataset_sha256'] = '6' * 64
    with pytest.raises(ValueError, match='244-target anchor'):
        run(four)
    for value in four:
        value['pipeline_binding']['dataset']['dataset_sha256'] = quality.ANCHOR_DATASET_SHA256
    with pytest.raises(ValueError, match='Fresh development'):
        run(four)


def test_dev_split_cannot_alias_heldout(four):
    dataset = four[0]['pipeline_binding']['dataset']
    dataset['split_sha256']['heldout'] = dataset['split_sha256']['dev']
    with pytest.raises(ValueError, match='alias heldout'):
        run(four)


def test_finite_but_overflowing_latency_ratio_cannot_escape_json_evidence(four):
    for value in (four[0], four[2]):
        value['resources']['warm'].update(p50_seconds=1e-308, p95_seconds=1e-308)
    four[1]['resources']['warm']['p95_seconds'] = 1e308
    with pytest.raises(ValueError, match='latency ratio'):
        run(four)


@pytest.mark.parametrize('count,exact,passed', [(10, 1, True), (1000, 100, True), (1000, 99, False), (522, 29, False)])
def test_heldout_floor_is_fixed_ten_percent_and_never_app_acceptance(count, exact, passed):
    summary = {'models': {MODEL: {'isolated_kaomoji': {'count': count, 'exact': exact}}}}
    result = quality.summarize_heldout_floor(summary, MODEL)
    assert result['passed'] is passed and result['app_acceptance'] is False
    assert result['minimum_exact_ratio'] == .1


@pytest.mark.parametrize('count,exact', [(0, 0), (1, 2), (True, 1), (1, float('nan'))])
def test_heldout_floor_rejects_empty_or_invalid_populations(count, exact):
    with pytest.raises(ValueError):
        quality.summarize_heldout_floor({'models': {MODEL: {'isolated_kaomoji': {'count': count, 'exact': exact}}}}, MODEL)


@pytest.fixture
def replay_fixture(tmp_path, monkeypatch, four):
    """Mock the replay boundary to test orchestration only; never launches inference."""
    from tools.PretrainedOcr import quality_evaluate as existing
    from tools.RegionKaomoji import evaluate as scene

    paths, views, calls = [], {}, []
    for i, value in enumerate(four):
        directory = tmp_path / ('evaluation-fixture-' + str(i))
        directory.mkdir()
        value['outputs'] = {}
        for name, filename in [('predictions', 'predictions.jsonl'), ('resources', 'resources.json'),
                               ('alignment', 'alignment-private.json')]:
            output = directory / filename
            output.write_text('{}', encoding='utf-8')
            value['outputs'][name] = existing.reference(output)
        value['pipeline_binding']['inputs']['file'] = str(tmp_path / ('anchor' if i < 2 else 'fresh') / 'inference-inputs.json')
        path = directory / 'evaluation.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        paths.append(path)
        views[str(path)] = value
    monkeypatch.setattr(existing.rp, 'profile_for', lambda path: SimpleNamespace(name='complete-spans-20261019', root=tmp_path))
    monkeypatch.setattr(existing, 'source_inventory', lambda: {'fixture-source': SHA})
    monkeypatch.setattr(scene, 'output_path', lambda path: Path(path).resolve())

    def replay(path, report):
        calls.append(path)
        return deepcopy(report['summary'])
    monkeypatch.setattr(existing, '_scene_regression_summary', replay)
    monkeypatch.setattr(scene, 'image_jobs', lambda root, split, inputs: root)

    def truth(root, split, jobs):
        count = 244 if root.name == 'anchor' else 20
        return [{'id': 'fixture-scene', 'regions': [{'kind': 'kaomoji', 'id': str(i), 'text': 'X'} for i in range(count)]}]
    monkeypatch.setattr(scene, 'load_truth', truth)
    monkeypatch.setattr(scene, 'masks_for_row', lambda root, row: [])

    def predictions(path, jobs):
        report = next(value for value in views.values() if value['outputs']['predictions']['file'] == path)
        return [{'prediction': {'models': {MODEL: {'regions': report['target_correctness']}}}}]
    monkeypatch.setattr(scene, 'read_predictions', predictions)
    monkeypatch.setattr(scene, 'aligned_region', lambda regions, index, masks, config: ('X' if regions[index]['correct'] else '?', 0))
    return tmp_path, paths, calls, views, existing


def test_wrapper_recomputes_correctness_and_binds_existing_evidence_without_writing(replay_fixture):
    root, paths, calls, views, existing = replay_fixture
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    result = quality.verify_round_development(*paths, MODEL, round_root=root)
    assert result['passed'] is True and len(calls) == 4
    assert result['evidence_kind'] == 'recomputed bound original-pixel CPU development'
    assert result['reports']['final_new']['report'] == existing.reference(paths[3])
    assert before == {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_wrapper_rejects_heldout_before_replay_or_truth(replay_fixture):
    root, paths, calls, views, existing = replay_fixture
    report = views[str(paths[0])]
    report['split'] = 'heldout'
    paths[0].write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='Only development'):
        quality.verify_round_development(*paths, MODEL, round_root=root)
    assert calls == []


def test_wrapper_rechecks_output_bytes_after_replay(replay_fixture, monkeypatch):
    root, paths, calls, views, existing = replay_fixture
    original = existing._scene_regression_summary
    def mutate(path, report):
        value = original(path, report)
        Path(report['outputs']['predictions']['file']).write_text('tampered', encoding='utf-8')
        return value
    monkeypatch.setattr(existing, '_scene_regression_summary', mutate)
    with pytest.raises(ValueError, match='output changed'):
        quality.verify_round_development(*paths, MODEL, round_root=root)


def test_wrapper_rejects_a_source_change_during_replay(replay_fixture, monkeypatch):
    root, paths, calls, views, existing = replay_fixture
    inventories = iter([{'source': SHA}, {'source': '2' * 64}])
    monkeypatch.setattr(existing, 'source_inventory', lambda: next(inventories))
    with pytest.raises(ValueError, match='Sources changed'):
        quality.verify_round_development(*paths, MODEL, round_root=root)


def test_wrapper_rejects_old_round_path_and_duplicate_reports(replay_fixture):
    root, paths, calls, views, existing = replay_fixture
    with pytest.raises(ValueError, match='Four distinct'):
        quality.verify_round_development(paths[0], paths[0], *paths[2:], MODEL, round_root=root)
    with pytest.raises(ValueError, match='Four distinct'):
        quality.verify_round_development(root.parent / 'old-round/evaluation.json', *paths[1:], MODEL, round_root=root)
    assert calls == []
