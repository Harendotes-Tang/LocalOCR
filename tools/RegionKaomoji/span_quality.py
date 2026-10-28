"""Fail-closed quality gates for the complete-span CPU research round.

``summarize_gate`` is a pure mechanism, not proof of inference. Its four inputs
are scene development reports augmented with ``target_correctness``: a list of
``{scene_id, truth_id, correct}`` records for *every* face, using strict regional
Unicode equality. Only ``verify_round_development`` constructs those records
from re-verified worker predictions and original truth masks. Neither function
writes files, runs inference, reads heldout data, or authorizes first use itself.
The caller must bind the returned proof into its existing paired freeze.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re


DEVELOPMENT_SCHEMA = 'localocr.complete-span-development-gates.v1'
HELDOUT_SCHEMA = 'localocr.complete-span-heldout-floor.v1'
ANCHOR_DATASET_SHA256 = 'e118e408529880f39f8e1cf0414045258343b1c64ebf8e82a9a7ffb57bc24613'
ANCHOR_DEV_SHA256 = 'ab41ee814b67b3614197724ef01004e2124c376313fbdcbfcd81931626dc00aa'
BODY_COUNTERS = ('baseline_correct_to_wrong', 'erroneous_rewrites',
                 'new_body_intersection_errors', 'body_literal_deleted')
NEGATIVE_COUNTERS = ('new_nonempty_scenes', 'new_graphic_nonempty')
_SOURCE = 'tools/RegionKaomoji/span_quality.py'
_ROUND = 'complete-spans-20261019'


def _object(value, name):
    if not isinstance(value, Mapping):
        raise ValueError('Missing/object expected: ' + name)
    return value


def _count(value, key, *, positive=False):
    result = _object(value, key).get(key)
    if type(result) is not int or result < (1 if positive else 0):
        raise ValueError('Missing/noninteger quality counter: ' + key)
    return result


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing/nonpositive finite measurement: ' + name)
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Missing/SHA256 expected: ' + name)
    return value


def _hash(value):
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ValueError('Evidence must be finite JSON data') from error
    return hashlib.sha256(payload).hexdigest()


def _face(summary, identity):
    models = _object(summary.get('models'), 'models')
    model = _object(models.get(identity), 'selected model')
    face = _object(model.get('isolated_kaomoji'), 'strict isolated_kaomoji')
    count, exact = _count(face, 'count', positive=True), _count(face, 'exact')
    if exact > count:
        raise ValueError('Face exact exceeds measured population')
    return model, count, exact


def _targets(records, count, exact):
    if not isinstance(records, list) or len(records) != count:
        raise ValueError('Every measured face needs target correctness')
    targets, correct = set(), set()
    for record in records:
        _object(record, 'target correctness')
        scene, truth = record.get('scene_id'), record.get('truth_id')
        if (not isinstance(scene, str) or not scene or not isinstance(truth, str) or not truth
                or type(record.get('correct')) is not bool):
            raise ValueError('Malformed target correctness')
        key = (scene, truth)
        if key in targets:
            raise ValueError('Duplicate measured target identity')
        targets.add(key)
        if record['correct']:
            correct.add(key)
    if len(correct) != exact:
        raise ValueError('Target correctness differs from recomputed strict exact')
    return targets, correct


def _evidence(value, identity):
    value = _object(value, 'development evidence')
    if (value.get('schema_version') != 'localocr.region-research.v1'
            or value.get('split') != 'dev' or value.get('actual_cpu_inference') is not True
            or value.get('independent_heldout') is not False
            or value.get('ground_truth_inference_input') is not False
            or value.get('promote') is not False or value.get('enabled') is not False):
        raise ValueError('Requires disabled actual CPU development evidence, never heldout')
    binding = _object(value.get('pipeline_binding'), 'pipeline_binding')
    if binding.get('promote') is not False or binding.get('enabled') is not False:
        raise ValueError('Pipeline must remain disabled')
    dataset = _object(binding.get('dataset'), 'dataset')
    _sha(dataset.get('dataset_sha256'), 'dataset_sha256')
    split = _object(dataset.get('split_sha256'), 'split_sha256')
    _sha(split.get('dev'), 'dev SHA256')
    if split.get('dev') == split.get('heldout'):
        raise ValueError('Development cannot alias heldout')
    summary = _object(value.get('summary'), 'summary')
    scenes = _count(summary, 'scenes', positive=True)
    population = _object(_object(dataset.get('counts'), 'dataset counts').get('dev'), 'dev counts')
    if _count(population, 'scenes', positive=True) != scenes:
        raise ValueError('Development scene population differs')
    model, count, exact = _face(summary, identity)
    if _count(population, 'faces', positive=True) != count:
        raise ValueError('Development face population differs')
    stage = _object(summary.get('candidate_stage'), 'candidate_stage')
    safe = _count(stage, 'safe_complete_coverage')
    if _count(stage, 'faces', positive=True) != count or safe > count:
        raise ValueError('Safe complete coverage population differs')
    body, negative = _object(model.get('body'), 'body'), _object(model.get('negative'), 'negative')
    populations = {'body_regions': _count(body, 'regions', positive=True),
                   'negative_scenes': _count(negative, 'negative_scenes', positive=True),
                   'graphic_regions': _count(negative, 'graphic_regions', positive=True)}
    errors = {key: _count(body, key) for key in BODY_COUNTERS}
    errors.update({key: _count(negative, key) for key in NEGATIVE_COUNTERS})
    targets, correct = _targets(value.get('target_correctness'), count, exact)
    models = binding.get('models')
    if (not isinstance(models, list) or not models
            or any(not isinstance(m, Mapping) or not isinstance(m.get('id'), str) or not m['id'] for m in models)
            or len({m.get('id') for m in models}) != len(models)
            or identity not in {m.get('id') for m in models}):
        raise ValueError('Selected model must have unique frozen model bindings')
    for item in models:
        _sha(item.get('candidate_sha256'), 'candidate_sha256')
        _sha(item.get('model_sha256'), 'model_sha256')
    baseline = _object(binding.get('baseline'), 'baseline')
    _sha(baseline.get('manifest_sha256'), 'baseline manifest_sha256')
    sources = _object(binding.get('implementation'), 'implementation')
    if not sources or _SOURCE not in sources:
        raise ValueError('Quality gate source must be included in frozen implementation')
    for name, digest in sources.items():
        _sha(digest, 'source ' + str(name))
    resources = _object(value.get('resources'), 'resources')
    warm = _object(resources.get('warm'), 'warm resources')
    if (_count(warm, 'count', positive=True) != scenes
            or resources.get('providers') != ['CPUExecutionProvider']
            or type(resources.get('intra_op_threads')) is not int or resources['intra_op_threads'] != 2
            or type(resources.get('inter_op_threads')) is not int or resources['inter_op_threads'] != 1
            or resources.get('actual_baseline_detection') is not True
            or type(resources.get('actual_specialist_models')) is not int
            or resources['actual_specialist_models'] != len(models)
            or resources.get('no_python_children_or_socket_io') is not True
            or resources.get('capture') != 'successful-owned-subprocess'
            or type(resources.get('worker_exit_code')) is not int or resources['worker_exit_code'] != 0):
        raise ValueError('Requires measured isolated CPU resources with identical fixed threads')
    _count(resources, 'worker_pid', positive=True)
    _count(resources, 'warmup_scenes', positive=True)
    p50 = _positive(warm.get('p50_seconds'), 'warm p50')
    p95 = _positive(warm.get('p95_seconds'), 'warm p95')
    if p95 < p50:
        raise ValueError('Warm p95 cannot be below p50')
    for key in ('total_warm_seconds', 'peak_rss_bytes', 'process_wall_seconds'):
        _positive(resources.get(key), key)
    machine = _object(resources.get('measurement_machine'), 'measurement_machine')
    if (set(machine) != {'schema_version', 'fingerprint_sha256'}
            or machine['schema_version'] != 'localocr.cpu-measurement-machine.v1'):
        raise ValueError('Same-machine measurement receipt required')
    _sha(machine['fingerprint_sha256'], 'measurement machine fingerprint')
    runtime = _object(resources.get('baseline_runtime'), 'actual baseline runtime')
    if not runtime:
        raise ValueError('Actual baseline runtime is required')
    config = _object(binding.get('config'), 'config')
    if not isinstance(config.get('views'), list) or not config['views'] or not config.get('limits'):
        raise ValueError('View and resource limits must be bound')
    if _sha(binding.get('config_sha256'), 'config_sha256') != _hash(config):
        raise ValueError('Full configuration digest differs from configuration bytes')
    _hash(binding)
    common = {'models': models, 'baseline': baseline, 'classifier': binding.get('classifier'),
              'implementation': sources, 'views': config['views'], 'limits': config['limits'],
              'machine': machine, 'baseline_runtime': runtime,
              'providers': resources['providers'], 'intra_op_threads': resources['intra_op_threads'],
              'inter_op_threads': resources['inter_op_threads']}
    return {'binding': binding, 'dataset': dataset, 'count': count, 'exact': exact, 'safe': safe,
            'targets': targets, 'correct': correct, 'populations': populations, 'errors': errors,
            'p95': p95, 'common': common}


def _pair(start, final, *, anchor):
    if (start['dataset'] != final['dataset'] or start['targets'] != final['targets']
            or start['populations'] != final['populations']
            or start['binding'].get('inputs') != final['binding'].get('inputs')):
        raise ValueError('Paired target/data/protection populations differ')
    lost, gained = start['correct'] - final['correct'], final['correct'] - start['correct']
    ratio = final['p95'] / start['p95']
    if not math.isfinite(ratio):
        raise ValueError('Nonfinite warm latency ratio')
    latency = ratio <= 1.15
    checks = {'exact_improved': final['exact'] > start['exact'],
              'all_start_correct_retained': not lost,
              'zero_new_body_and_negative_errors': not any(final['errors'].values()),
              'warm_p95_within_115_percent': latency}
    if anchor:
        if (start['dataset']['dataset_sha256'] != ANCHOR_DATASET_SHA256
                or start['dataset']['split_sha256']['dev'] != ANCHOR_DEV_SHA256
                or start['count'] != 244):
            raise ValueError('Approved 244-target anchor identity differs')
        checks.update(safe_complete_at_least_100=final['safe'] >= 100,
                      strict_exact_at_least_30=final['exact'] >= 30)
    return {'count': final['count'], 'start_exact': start['exact'], 'final_exact': final['exact'],
            'exact_gain': final['exact'] - start['exact'], 'gained_targets': len(gained),
            'lost_targets': len(lost), 'start_safe_complete': start['safe'],
            'final_safe_complete': final['safe'], 'protection_population': final['populations'],
            'final_new_errors': final['errors'], 'warm_p95_start_seconds': start['p95'],
            'warm_p95_final_seconds': final['p95'], 'warm_p95_ratio': ratio,
            'target_set_sha256': _hash(sorted(final['targets'])),
            'start_correct_targets_sha256': _hash(sorted(start['correct'])),
            'final_correct_targets_sha256': _hash(sorted(final['correct'])),
            'checks': checks, 'passed': all(checks.values())}


def summarize_gate(start_anchor, final_anchor, start_new, final_new, selected_id):
    """Return strict development verdicts from four already-verified evidence views.

    Malformed/mismatched evidence raises ``ValueError``; well-formed measurements
    missing any quality threshold return ``passed=False`` with explicit checks.
    Pure fixture calls only test the mechanism. The verification wrapper below
    supplies real regional correctness; callers must not handwrite that array.
    """
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError('Selected specialist identity required')
    values = [_evidence(value, selected_id) for value in
              (start_anchor, final_anchor, start_new, final_new)]
    if any(value['common'] != values[0]['common'] for value in values[1:]):
        raise ValueError('Machine/model/baseline/source/thread/view bindings differ')
    if (values[0]['binding']['config_sha256'] != values[2]['binding']['config_sha256']
            or values[1]['binding']['config_sha256'] != values[3]['binding']['config_sha256']):
        raise ValueError('Cross-dataset starting/final full configuration bindings differ')
    if values[0]['dataset']['dataset_sha256'] == values[2]['dataset']['dataset_sha256']:
        raise ValueError('Fresh development cannot reuse the anchor dataset')
    anchor, fresh = _pair(*values[:2], anchor=True), _pair(*values[2:], anchor=False)
    return {'schema_version': DEVELOPMENT_SCHEMA, 'selected_id': selected_id,
            'evidence_kind': 'pure-gate-summary; verified worker proof required before freeze',
            'counting': 'strict regional exact raw Unicode; no width or tail normalization',
            'anchor': anchor, 'fresh_development': fresh,
            'common_binding_sha256': _hash(values[0]['common']),
            'passed': anchor['passed'] and fresh['passed'],
            'promote': False, 'enabled': False, 'production_enabled': False, 'app_acceptance': False}


def summarize_heldout_floor(summary, selected_id):
    """Additional 10% strict face floor, not independent evidence or APP approval.

    The caller must separately verify the once-only report, paired freeze and
    existing body/negative research gates. Invalid or empty counters raise;
    a measured score below the fixed floor is a normal ``passed=False`` result.
    """
    _, count, exact = _face(_object(summary, 'heldout summary'), selected_id)
    passed = exact * 10 >= count
    return {'schema_version': HELDOUT_SCHEMA, 'selected_id': selected_id,
            'count': count, 'exact': exact, 'exact_ratio': exact / count,
            'minimum_exact_ratio': .10, 'passed': passed,
            'evidence_kind': 'additional floor only; verify once-only heldout and other gates separately',
            'promote': False, 'enabled': False, 'production_enabled': False, 'app_acceptance': False}


def _read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON evidence: ' + value)
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream, parse_constant=invalid)


def _verified_view(path, selected_id, quality, scene):
    report = _object(_read(path), 'development report')
    # The split check deliberately precedes replay, jobs, image masks or truth.
    if report.get('split') != 'dev' or report.get('independent_heldout') is not False:
        raise ValueError('Only development reports may enter strict development gates')
    report_ref = quality.reference(path)
    summary = quality._scene_regression_summary(path, report)
    binding = report['pipeline_binding']
    root = Path(binding['inputs']['file']).parent
    jobs = scene.image_jobs(root, 'dev', binding['inputs']['file'])
    rows = scene.load_truth(root, 'dev', jobs)
    predictions = scene.read_predictions(report['outputs']['predictions']['file'], jobs)
    correctness = []
    for row, record in zip(rows, predictions):
        masks = scene.masks_for_row(root, row)
        regions = record['prediction']['models'][selected_id]['regions']
        for index, truth in enumerate(row['regions']):
            if truth['kind'] == 'kaomoji':
                text, _ = scene.aligned_region(regions, index, masks, binding['config'])
                correctness.append({'scene_id': row['id'], 'truth_id': truth['id'],
                                    'correct': text == truth['text']})
    if len(rows) != len(predictions) or quality.reference(path) != report_ref:
        raise ValueError('Development report changed during replay')
    # Recheck every output after the second read, not just the report summary.
    for output in report['outputs'].values():
        if quality.common.digest(output['file']) != output['sha256']:
            raise ValueError('Bound development output changed during replay')
    return ({**report, 'summary': summary, 'target_correctness': correctness},
            {'report': report_ref, 'outputs': report['outputs'],
             'pipeline_binding_sha256': _hash(binding), 'dataset': binding['dataset']})


def verify_round_development(start_anchor, final_anchor, start_new, final_new,
                             selected_id, *, round_root):
    """Recompute all four current-source dev reports and return a bindable proof.

    Arguments are paths to existing scene ``evaluation.json`` files in this
    round and the selected model ID. Reports must come from the same freshly
    captured CPU machine; old round timings cannot enter. Existing evaluator
    verification checks model/config/source identities and owned CPU outputs;
    original masks then supply a strict correctness set for each face. No
    output path is accepted, so this helper cannot overwrite existing evidence.

    Malformed evidence raises ``ValueError`` (or a filesystem error); quality
    failure returns ``passed=False``. This result grants no heldout access.
    """
    # Lazy imports avoid the evaluator's existing quality_evaluate import cycle.
    from tools.PretrainedOcr import quality_evaluate as quality
    from tools.RegionKaomoji import evaluate as scene

    root = scene.safe_path(round_root)
    profile = quality.rp.profile_for(root)
    if profile is None or profile.name != _ROUND or root != Path(profile.root).resolve():
        raise ValueError('Strict development proof requires the registered complete-span round')
    paths = [scene.output_path(path) for path in (start_anchor, final_anchor, start_new, final_new)]
    if len(set(paths)) != 4 or any(not path.is_relative_to(root) for path in paths):
        raise ValueError('Four distinct current-round development reports required')
    before = quality.source_inventory()
    pairs = [_verified_view(path, selected_id, quality, scene) for path in paths]
    verdict = summarize_gate(*(pair[0] for pair in pairs), selected_id)
    if quality.source_inventory() != before:
        raise ValueError('Sources changed during strict development verification')
    return {**verdict, 'evidence_kind': 'recomputed bound original-pixel CPU development',
            'reports': dict(zip(('start_anchor', 'final_anchor', 'start_new', 'final_new'),
                                (pair[1] for pair in pairs))),
            'source_inventory': before}
