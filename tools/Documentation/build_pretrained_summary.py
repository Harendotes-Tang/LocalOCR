"""Build hash-bound, anonymous OCR chart summaries without model execution.

Only evaluation and training receipts are read. Dataset split files, image
samples, model weights and private manifests are never opened. Input paths,
labels, predictions and arbitrary source fields are excluded from both outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts'
SCHEMA = 'localocr.pretrained.readme-evaluation.v1'
FONT_SHA256 = '2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b'
KINDS = ('plain', 'face', 'punctuation', 'emoji', 'blank', 'graphic')
FAMILIES = ('ppocrv6-tiny', 'repsvtr-mobile')
FAMILY_ALIASES = {'official-ppocrv6-tiny': FAMILIES[0], 'openocr-repsvtr-mobile': FAMILIES[1],
                  FAMILIES[0]: FAMILIES[0], FAMILIES[1]: FAMILIES[1]}
SERIES = (('production-baseline', 'production'), (FAMILIES[0], 'official-pretrained'),
          (FAMILIES[0], 'finetuned'), (FAMILIES[1], 'official-pretrained'), (FAMILIES[1], 'finetuned'))
SHA = re.compile(r'[a-f0-9]{64}\Z')
IDENTITY = re.compile(r'[a-z0-9][a-z0-9._-]{0,95}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(isinstance(value, str) and SHA.fullmatch(value), 'Expected a lowercase SHA256 binding')
    return value


def identity(value):
    require(isinstance(value, str) and IDENTITY.fullmatch(value), 'Expected a non-path research identity')
    return value


def count(value, maximum=None):
    require(type(value) is int and value >= 0 and (maximum is None or value <= maximum),
            'Invalid integer count in source receipt')
    return value


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'Duplicate JSON field in source receipt')
        result[key] = value
    return result


def constant(value):
    raise ValueError('Nonfinite JSON number in source receipt')


class Snapshot:
    """Bind the exact bytes consumed and recheck them before any output write."""
    def __init__(self, path, role, fixture=False):
        path = Path(path)
        require(not str(path).startswith(('\\\\', '//')), 'Receipt paths must remain local')
        self.path = path.resolve()
        require(fixture or self.path.is_relative_to(ARTIFACTS.resolve()),
                'Actual receipts must remain in LocalOCR artifacts')
        allowed = {'dev-evaluation': {'evaluation.json'}, 'heldout-evaluation': {'evaluation.json'},
                   'run-preparation': {'prepare.json', 'initialization.json'},
                   'run-curve': {'training.json', 'training-progress.json', 'history.json', 'last-failure.json'},
                   'run-freeze': {'run-freeze.json', 'completed-run-freeze.json'}, 'run-selection': {'best-selection.json'}}
        require(self.path.name in allowed[role], 'Only explicit evaluation/preparation/training receipt files are accepted')
        require(self.path.stat().st_size <= 128 * 1024 * 1024, 'Receipt exceeds 128 MiB input limit')
        self.raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self.value = json.loads(self.raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=constant)
        self.role = role
        self.fixture = fixture

    def stable(self):
        require(hashlib.sha256(self.path.read_bytes()).hexdigest() == self.sha256,
                'Source receipt changed during anonymous summary construction')


def binding(value):
    require(isinstance(value, dict) and value.get('schema_version') == 'localocr.pretrained.freeze.v1'
            and value.get('license') == 'CC0-1.0', 'Expected the original frozen synthetic dataset binding')
    sha(value.get('dataset_sha256'))
    splits, sizes = value.get('split_sha256'), value.get('counts')
    require(isinstance(splits, dict) and set(splits) == {'train', 'dev', 'heldout'}, 'Incomplete frozen split hash binding')
    for digest in splits.values():
        sha(digest)
    require(isinstance(sizes, dict) and set(sizes) == {'train', 'dev', 'heldout'}, 'Incomplete frozen split counts')
    for split, expected in (('train', 31000), ('dev', 2100), ('heldout', 4200)):
        require(isinstance(sizes[split], dict) and count(sizes[split].get('count')) == expected,
                'This round requires 31000 train / 2100 dev / 4200 heldout images')
    return value


def metric(value, expected_count=None, negative=False):
    require(isinstance(value, dict), 'Missing category metric')
    result = {key: count(value.get(key)) for key in ('count', 'exact', 'characters', 'character_errors', 'false_nonempty_outputs')}
    require(result['count'] > 0 and (expected_count is None or result['count'] == expected_count),
            'Category sample denominator differs')
    require(result['exact'] <= result['count'] and result['character_errors'] >= result['count'] - result['exact'],
            'Exact count disagrees with raw error count')
    require(result['false_nonempty_outputs'] <= result['count'], 'Negative output count exceeds sample count')
    if negative:
        require(result['characters'] == 0 and result['false_nonempty_outputs'] == result['count'] - result['exact'],
                'Negative output count disagrees with exact count')
    else:
        require(result['characters'] > 0 and result['false_nonempty_outputs'] == 0,
                'Positive category has invalid reference/negative counts')
    cer = value.get('cer')
    if result['characters']:
        require(isinstance(cer, (int, float)) and not isinstance(cer, bool) and math.isfinite(cer)
                and math.isclose(cer, result['character_errors'] / result['characters'], rel_tol=0, abs_tol=1e-12),
                'Reported CER differs from raw reference/error counts')
    else:
        require(cer is None, 'Negative category cannot report a CER denominator')
    return result


def cpu_report(snapshot, split, complete=True):
    report = snapshot.value
    require(not (isinstance(report, dict) and report.get('_fixture') is True) or snapshot.fixture,
            'A labelled tool fixture cannot be published as actual evaluation')
    require(isinstance(report, dict) and report.get('schema_version') == 'localocr.pretrained.evaluation.v1'
            and report.get('split') == split and report.get('promote') is False and report.get('enabled') is False
            and report.get('independent_heldout') is (split == 'heldout')
            and report.get('normalization') == 'none; exact Unicode, case, whitespace and graphemes',
            'Expected actual raw-Unicode, research-only paired CPU evaluation')
    frozen = binding(report.get('dataset_binding'))
    baseline = report.get('baseline_binding')
    require(isinstance(baseline, dict) and baseline.get('id') == 'production-baseline', 'Missing production baseline identity')
    baseline_sha = sha(baseline.get('models', {}).get('rec', {}).get('sha256'))
    candidates = report.get('candidate_bindings')
    require(isinstance(candidates, list), 'Missing candidate identity bindings')
    aliases, roles, identities = {}, {SERIES[0]}, {'production-baseline'}
    for item in candidates:
        require(isinstance(item, dict), 'Invalid candidate binding')
        model_id = identity(item.get('id'))
        family = FAMILY_ALIASES.get(item.get('family'))
        role = family, item.get('stage')
        require(role in SERIES[1:] and role not in roles and model_id not in identities,
                'Repeated or unsupported candidate family/stage/identity')
        roles.add(role); identities.add(model_id)
        aliases[model_id] = family
        sha(item.get('candidate_sha256')); sha(item.get('model_sha256')); sha(item.get('vocabulary_sha256'))
    require(not complete or roles == set(SERIES), 'Paired CPU evaluation must contain the baseline and both official before/finetuned after pairs')
    models = report.get('models')
    require(isinstance(models, dict) and set(models) == identities, 'CPU model results differ from bound model identities')
    if split == 'dev':
        require(report.get('first_use') is None, 'Development report cannot consume heldout')
    else:
        first = report.get('first_use')
        require(isinstance(first, dict), 'Independent heldout report needs a first-use receipt binding')
        sha(first.get('sha256')); sha(first.get('selection_sha256'))
    expected = 2100 if split == 'dev' else 4200
    anonymous, denominators = [], None
    for model_id in identities:
        value = models[model_id]
        require(isinstance(value, dict), 'Missing bound CPU model result')
        resources = value.get('resources')
        require(isinstance(resources, dict) and resources.get('providers') == ['CPUExecutionProvider']
                and resources.get('intra_op_threads') == 2 and resources.get('inter_op_threads') == 1,
                'Final paired metrics must come from the frozen ONNX CPU evaluator')
        summary = value.get('summary')
        require(isinstance(summary, dict) and summary.get('total_images') == expected
                and summary.get('ambiguous_excluded') == 0, 'Incomplete or ambiguous synthetic CPU evaluation')
        categories = summary.get('by_kind')
        require(isinstance(categories, dict) and set(categories) == set(KINDS), 'Incomplete CPU category coverage')
        selected = {kind: metric(categories[kind], negative=kind in ('blank', 'graphic')) for kind in KINDS}
        require(sum(item['count'] for item in selected.values()) == expected, 'CPU category counts do not total the frozen split')
        totals = {key: sum(item[key] for item in selected.values()) for key in ('count', 'exact', 'characters', 'character_errors', 'false_nonempty_outputs')}
        overall = summary.get('overall')
        require(isinstance(overall, dict) and all(overall.get(key) == total for key, total in totals.items()),
                'Overall CPU aggregates disagree with category totals')
        these = {kind: (item['count'], item['characters']) for kind, item in selected.items()}
        require(denominators is None or denominators == these, 'Paired CPU models have different sample/reference denominators')
        denominators = these
        if model_id == 'production-baseline':
            item = {'id': model_id, 'family': 'production-baseline', 'stage': 'production', 'model_sha256': baseline_sha, 'by_kind': selected}
        else:
            meta = next(candidate for candidate in candidates if candidate['id'] == model_id)
            require(value.get('stage') == meta['stage'], 'Result stage differs from bound candidate stage')
            moves = value.get('transitions_from_production', {}).get('by_kind')
            require(isinstance(moves, dict) and all(kind in moves for kind in ('plain', 'punctuation')),
                    'Missing ordinary-text protection transitions')
            protected = {}
            for kind in ('plain', 'punctuation'):
                move = moves[kind]
                require(isinstance(move, dict) and move.get('count') == selected[kind]['count'], 'Protection transition denominator differs')
                correct = count(move.get('baseline_correct_to_wrong'), selected[kind]['count'])
                rewrites = count(move.get('erroneous_rewrites'), selected[kind]['count'])
                require(correct <= rewrites and rewrites <= selected[kind]['count'] - selected[kind]['exact'],
                        'Protection transitions disagree with candidate exact count')
                require(correct <= models['production-baseline']['summary']['by_kind'][kind]['exact'],
                        'Correct-to-wrong exceeds baseline correct count')
                protected[kind] = {'count': selected[kind]['count'], 'baseline_correct_to_wrong': correct, 'erroneous_rewrites': rewrites}
            item = {'id': model_id, 'family': aliases[model_id], 'stage': meta['stage'],
                    'model_sha256': meta['model_sha256'], 'by_kind': selected,
                    'transitions_from_production': {'by_kind': protected}}
        anonymous.append(item)
    for family in FAMILIES:
        members = [candidate for candidate in candidates if FAMILY_ALIASES.get(candidate['family']) == family]
        if len(members) == 2:
            require(members[0].get('preprocessing') == members[1].get('preprocessing')
                    and isinstance(members[0].get('preprocessing'), dict),
                    'Official before and finetuned after preprocessing differs within a family')
    anonymous.sort(key=lambda model: SERIES.index((model['family'], model['stage'])))
    return frozen, anonymous


def native_point(value, step, family, expected_denominators):
    require(isinstance(value, dict) and isinstance(value.get('by_kind'), dict)
            and set(value['by_kind']) == set(KINDS), 'A native development point must cover every category')
    fields = ('samples', 'exact_correct', 'reference_codepoints', 'edit_distance') if family == 'repsvtr-mobile' else ('rows', 'exact_correct', 'reference_codepoints', 'edit_distance')
    totals = [0, 0, 0, 0]
    for kind in KINDS:
        item = value['by_kind'][kind]
        require(isinstance(item, dict), 'Missing native development category metric')
        values = [count(item.get(field)) for field in fields]
        require((values[0], values[2]) == expected_denominators[kind], 'Native development sample/reference denominator differs from CPU split')
        require(values[1] <= values[0] and values[3] >= values[0] - values[1], 'Native exact/errors disagree')
        negative = item.get('negative_false_outputs' if family == 'repsvtr-mobile' else 'negative_nonempty')
        require(count(negative, values[0]) == (values[0] - values[1] if kind in ('blank', 'graphic') else 0),
                'Native negative output count disagrees with exact count')
        totals = [old + new for old, new in zip(totals, values)]
    require(totals[0] == 2100 and totals[2] > 0, 'Native development point is not the full frozen 2100 images')
    accuracy = value.get('exact_accuracy')
    require(isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and math.isfinite(accuracy)
            and math.isclose(accuracy, totals[1] / totals[0], rel_tol=0, abs_tol=1e-12), 'Native exact ratio differs from integer count')
    if family == 'ppocrv6-tiny':
        require(value.get('ignore_space') is False, 'Paddle native development must preserve spaces')
        for key, total in zip(('rows', 'exact_correct', 'reference_codepoints', 'edit_distance'), totals):
            require(value.get(key) == total, 'Paddle native overall metric differs from category totals')
    return {'step': count(step), 'count': totals[0], 'exact': totals[1], 'characters': totals[2], 'character_errors': totals[3]}


def curve(family, run_id, preparation, history, frozen, denominators, run_freeze=None, selection=None):
    require(family in FAMILIES, 'Unsupported training family')
    identity(run_id)
    prepared = preparation.value
    require(isinstance(prepared, dict), 'Missing training preparation receipt')
    if family == 'repsvtr-mobile':
        require(prepared.get('family') == 'openocr-repsvtr-mobile', 'OpenOCR preparation family differs')
        prepared_binding = prepared.get('dataset_binding')
    else:
        require(prepared.get('schema_version') == 'localocr.paddle.initialization.v1', 'Paddle preparation schema differs')
        prepared_binding = prepared.get('dataset')
    require(binding(prepared_binding) == frozen, 'Training preparation dataset binding differs from paired CPU development')
    value = history.value
    failed = isinstance(value, dict) and value.get('schema_version') == 'localocr.openocr.failure.v1'
    require(not failed or (family == 'repsvtr-mobile' and run_freeze is not None and selection is not None),
            'Failed OpenOCR curve requires its immutable run-freeze and best-selection receipts')
    if isinstance(value, dict):
        require(isinstance(value.get('history'), list), 'Missing actual optimizer history')
        if family == 'ppocrv6-tiny':
            require(value.get('schema_version') == 'localocr.paddle.training.v1' and value.get('dataset') == frozen
                    and value.get('dev_rows') == 2100 and value.get('train_rows') == 31000
                    and value.get('actual_weights_changed') is True
                    and type(value.get('selected_weights_changed')) is bool, 'Paddle completed training binding/update proof differs')
            preparation_binding = value.get('preparation')
            require(isinstance(preparation_binding, dict) and preparation_binding.get('sha256') == preparation.sha256,
                    'Paddle training does not bind this preparation receipt')
        elif not failed:
            require(value.get('nonzero_gradient') is True and value.get('save_reload_strict') is True
                    and isinstance(value.get('changed_tensors'), list) and bool(value['changed_tensors']),
                    'OpenOCR completed training has no effective update/reload proof')
            sha(value.get('initial_checkpoint_sha256')); sha(value.get('checkpoint_sha256'))
            require(value.get('preparation_sha256') == preparation.sha256, 'OpenOCR training preparation SHA differs')
            require(selection is not None, 'Completed OpenOCR run requires its actual best-selection receipt')
        records = value['history']
        require(count(value.get('steps')) > 0, 'Completed run has no optimizer updates')
    else:
        require(isinstance(value, list), 'Training history must be a completed receipt or a native progress list')
        records = value
    points, previous = [], -1
    for record in records:
        require(isinstance(record, dict), 'Malformed optimizer history point')
        step = count(record.get('step'))
        require(step > previous, 'Optimizer step sequence repeats or decreases')
        previous = step
        if 'dev_metrics' in record or 'dev' in record:
            require(('dev_metrics' in record) == (family == 'repsvtr-mobile'), 'Native metric shape belongs to another family')
            points.append(native_point(record['dev_metrics' if family == 'repsvtr-mobile' else 'dev'], step, family, denominators))
    require(bool(points), 'No completed full native development evaluation in the supplied history')
    require(previous > 0, 'An initialization-only snapshot is not an effective training run')
    if isinstance(value, dict):
        require(previous <= value['steps'], 'Recorded optimizer history exceeds completed step count')
    closure = {'family': family, 'run_id': run_id, 'preparation_sha256': preparation.sha256, 'curve_sha256': history.sha256,
               'completed': False if failed else (True if isinstance(value, dict) else None),
               'status': 'failed' if failed else ('completed' if isinstance(value, dict) else 'progress-snapshot'),
               'actual_steps': value['steps'] if isinstance(value, dict) else previous,
               'checkpoint_selection': {'outcome': 'unrecorded', 'step': None, 'sha256': None},
               'selection_scope': 'run-local checkpoint; final family selection is separate'}
    if isinstance(value, dict) and family == 'ppocrv6-tiny':
        start = value.get('start_checkpoint');last = value.get('last_checkpoint')
        require(isinstance(start, dict) and isinstance(last, dict), 'Paddle completed run lacks actual start/last checkpoint bindings')
        sha(start.get('sha256'));sha(last.get('sha256'))
        require(start['sha256'] != last['sha256'], 'Paddle effective updates cannot retain identical last weights')
        closure['start_checkpoint_sha256'] = start['sha256'];closure['last_checkpoint_sha256'] = last['sha256']
        if value['selected_weights_changed']:
            chosen = value.get('best_checkpoint');exposure = value.get('selected_training_exposure')
            require(isinstance(chosen, dict) and isinstance(exposure, dict), 'New Paddle best lacks selected checkpoint/exposure')
            step = count(exposure.get('step'))
            selected_point = native_point(value.get('best_dev'), step, family, denominators)
            require(step > 0 and selected_point in points, 'Paddle selected best lacks its completed development point')
            require(sha(chosen.get('sha256')) != start['sha256'], 'Paddle new best equals the starting parent weights')
            require(value.get('selection_outcome', 'new-effective-best') == 'new-effective-best', 'Paddle new-best outcome disagrees')
            closure['checkpoint_selection'] = {'outcome': 'run-best', 'step': step, 'sha256': chosen['sha256']}
        else:
            require(value.get('warm_start') is True and value.get('best_checkpoint') is None
                    and value.get('selected_training_exposure') is None
                    and value.get('selection_outcome') == 'parent-retained-no-development-improvement',
                    'Paddle no-newbest requires a completed warm run explicitly retaining the parent')
            before = native_point(value.get('initial_dev'), 0, family, denominators)
            best = native_point(value.get('best_dev'), 0, family, denominators)
            require(before == best and all((point['exact'], -point['character_errors']) <=
                    (before['exact'], -before['character_errors']) for point in points),
                    'Parent-retained outcome disagrees with completed development improvement')
            closure['checkpoint_selection'] = {'outcome': 'parent-retained', 'step': None, 'sha256': start['sha256']}
    elif isinstance(value, dict) and not failed:
        chosen = selection.value
        require(isinstance(chosen, dict) and chosen.get('dev_split_sha256') == frozen['split_sha256']['dev'],
                'OpenOCR run-best development split differs')
        step = count(chosen.get('step'))
        selected_point = native_point(chosen.get('dev_metrics'), step, family, denominators)
        require(step > 0 and selected_point in points, 'OpenOCR run-best lacks a completed development point')
        ratio = value.get('best_dev_exact')
        require(type(ratio) in (int, float) and math.isclose(ratio, selected_point['exact'] / 2100, rel_tol=0, abs_tol=1e-12),
                'OpenOCR run-best accuracy differs from selected receipt')
        closure.update(start_checkpoint_sha256=value['initial_checkpoint_sha256'], last_checkpoint_sha256=value['checkpoint_sha256'],
                       selection_sha256=selection.sha256, checkpoint_selection={'outcome': 'run-best', 'step': step,
                                                                              'sha256': sha(chosen.get('checkpoint_sha256'))})
    if not failed and run_freeze is not None:
        frozen_run = run_freeze.value
        require(isinstance(frozen_run, dict) and frozen_run.get('schema_version') == 'localocr.openocr.run-freeze.v1'
                and frozen_run.get('run_id') == run_id and frozen_run.get('completed') is True
                and type(frozen_run.get('exit_code')) is int and frozen_run['exit_code'] == 0
                and frozen_run.get('heldout_read') is False and frozen_run.get('actual_steps') == value['steps']
                and frozen_run.get('selected_step') == closure['checkpoint_selection']['step']
                and frozen_run.get('gpu_allocated_before_lease_release') == 0,
                'Completed OpenOCR freeze identity/completion/selection proof differs')
        files = frozen_run.get('files')
        require(isinstance(files, list), 'Completed run needs its frozen source manifest')
        for consumed in (preparation, history, selection):
            listed = [item for item in files if isinstance(item, dict) and item.get('file') == consumed.path.name]
            require(len(listed) == 1 and listed[0].get('sha256') == consumed.sha256, 'Completed run frozen source SHA differs')
        for name, expected in (('initial.pth', closure['start_checkpoint_sha256']),
                               ('last.pth', closure['last_checkpoint_sha256']),
                               ('best.pth', closure['checkpoint_selection']['sha256'])):
            listed = [item for item in files if isinstance(item, dict) and item.get('file') == name]
            require(len(listed) == 1 and listed[0].get('sha256') == expected, 'Completed freeze checkpoint SHA binding differs')
        closure['run_freeze_sha256'] = run_freeze.sha256
    if failed:
        frozen_run = run_freeze.value
        require(isinstance(frozen_run, dict) and frozen_run.get('schema_version') == 'localocr.openocr.run-freeze.v1'
                and frozen_run.get('run_id') == run_id and frozen_run.get('completed') is False
                and frozen_run.get('process_death_verified') is True and frozen_run.get('heldout_read') is False
                and frozen_run.get('actual_steps') == value['steps'], 'Failed run freeze identity/completion/step proof differs')
        files = frozen_run.get('files')
        require(isinstance(files, list), 'Failed run needs its frozen source file manifest')
        for consumed in (preparation, history, selection):
            listed = [item for item in files if isinstance(item, dict) and item.get('file') == consumed.path.name]
            require(len(listed) == 1 and listed[0].get('sha256') == consumed.sha256, 'Failed run frozen source SHA differs')
        chosen = selection.value
        require(isinstance(chosen, dict) and count(chosen.get('step')) == frozen_run.get('selected_step')
                and chosen.get('dev_split_sha256') == frozen['split_sha256']['dev'], 'Failed run selected step/split differs')
        sha(chosen.get('checkpoint_sha256'))
        selected = native_point(chosen.get('dev_metrics'), chosen['step'], family, denominators)
        require(selected in points, 'Failed run selected checkpoint has no matching completed development point')
        failure = value.get('original_failure')
        require(isinstance(failure, dict) and failure == frozen_run.get('original_failure'), 'Failed run error binding differs')
        error_type = failure.get('type')
        require(error_type in ('PermissionError', 'TimeoutError', 'RuntimeError', 'ValueError', 'OSError'), 'Unrecognized frozen failure type')
        match = re.search(r'\[WinError (\d+)\]', failure.get('message', ''))
        closure.update({'run_freeze_sha256': run_freeze.sha256, 'selection_sha256': selection.sha256,
                        'selected_step': chosen['step'], 'selected_checkpoint_sha256': chosen['checkpoint_sha256'],
                        'failure_type': error_type, 'failure_code': 'WinError' + match.group(1) if match else None,
                        'checkpoint_selection': {'outcome': 'selected-before-failure', 'step': chosen['step'],
                                                 'sha256': chosen['checkpoint_sha256']}})
    closure_sha = hashlib.sha256(json.dumps(closure, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
    return {'family': family, 'run_id': run_id, 'source_sha256': closure_sha,
            'dataset_sha256': frozen['dataset_sha256'], 'dev_split_sha256': frozen['split_sha256']['dev'],
            'dev_count': 2100, 'evaluation_backend': 'CUDA-native', 'points': points,
            'run_state': {'status': closure['status'], 'actual_steps': closure['actual_steps'],
                          'checkpoint_selection': closure['checkpoint_selection']}}, closure


def build(dev_report, runs, heldout_report=None, fixture=False):
    """Extract whitelist aggregates and a path-free provenance receipt."""
    snapshots = [Snapshot(dev_report, 'dev-evaluation', fixture)]
    frozen, dev_models = cpu_report(snapshots[0], 'dev')
    denominators = {kind: (item['count'], item['characters']) for kind, item in dev_models[0]['by_kind'].items()}
    training, closures, ids, curve_sources = [], [], set(), set()
    for family, run_id, preparation_path, history_path in runs:
        require(run_id not in ids, 'Repeated training run identity')
        ids.add(run_id)
        preparation = Snapshot(preparation_path, 'run-preparation', fixture)
        history = Snapshot(history_path, 'run-curve', fixture)
        require(history.sha256 not in curve_sources, 'The same training receipt cannot impersonate another iteration')
        curve_sources.add(history.sha256)
        extra = []
        if isinstance(history.value, dict) and family == 'repsvtr-mobile':
            if history.value.get('schema_version') == 'localocr.openocr.failure.v1':
                extra = [Snapshot(history.path.parent / 'run-freeze.json', 'run-freeze', fixture),
                         Snapshot(history.path.parent / 'best-selection.json', 'run-selection', fixture)]
            else:
                freeze_path = history.path.parent / 'completed-run-freeze.json'
                extra = [Snapshot(freeze_path, 'run-freeze', fixture) if freeze_path.exists() else None,
                         Snapshot(history.path.parent / 'best-selection.json', 'run-selection', fixture)]
        item, closure = curve(family, run_id, preparation, history, frozen, denominators, *extra)
        snapshots.extend(snapshot for snapshot in (preparation, history, *extra) if snapshot is not None)
        training.append(item);closures.append(closure)
    require({item['family'] for item in training} == set(FAMILIES), 'Provide actual development curve receipts for both trained families')
    heldout = None
    if heldout_report is not None:
        snapshot = Snapshot(heldout_report, 'heldout-evaluation', fixture)
        held_binding, models = cpu_report(snapshot, 'heldout')
        require(held_binding == frozen, 'Heldout and development dataset bindings differ')
        require(snapshot.value['candidate_bindings'] == snapshots[0].value['candidate_bindings']
                and snapshot.value['baseline_binding'] == snapshots[0].value['baseline_binding'],
                'Heldout model bindings differ from frozen paired development')
        snapshots.append(snapshot)
        heldout = {'report_sha256': snapshot.sha256, 'dataset_sha256': frozen['dataset_sha256'],
                   'split_sha256': frozen['split_sha256']['heldout'], 'count': 4200,
                   'evaluation_backend': 'ONNX Runtime CPU', 'models': models}
    summary = {'schema_version': SCHEMA, 'fixture': fixture,
               'dataset': {'sha256': frozen['dataset_sha256'], 'split_sha256': {key: frozen['split_sha256'][key] for key in ('dev', 'heldout')},
                           'counts': {'dev': 2100, 'heldout': 4200}},
               'plot_font': {'sha256': FONT_SHA256, 'license': 'OFL-1.1'}, 'training': training, 'heldout': heldout}
    receipt = {'schema_version': 'localocr.pretrained.summary-build.v1', 'fixture': fixture,
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'dev_report_sha256': snapshots[0].sha256, 'heldout_report_sha256': heldout['report_sha256'] if heldout else None,
               'sources': [{'role': snapshot.role, 'sha256': snapshot.sha256} for snapshot in snapshots], 'run_closures': closures,
               'scope': 'single synthetic canvas recognizers; excludes automatic candidate localization/fusion and APP acceptance',
               'raw_images_labels_predictions_paths_exported': False}
    for snapshot in snapshots:
        snapshot.stable()
    return summary, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-report', type=Path, required=True, help='Complete five-model paired CPU development evaluation.json')
    parser.add_argument('--heldout-report', type=Path, help='Optional final completed five-model heldout evaluation.json; never a dataset split')
    parser.add_argument('--run', nargs=4, action='append', default=[], metavar=('FAMILY', 'RUN_ID', 'PREPARATION_JSON', 'CURVE_JSON'),
                        help='Repeat real runs; curve is training/progress/history or frozen last-failure.json (requires sibling run-freeze and best-selection)')
    parser.add_argument('--output', type=Path, required=True, help='Fresh anonymous summary JSON; receipt is written beside it')
    parser.add_argument('--fixture', action='store_true', help='Transparent non-evaluation tool fixture; output remains in artifacts')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(ROOT.resolve()), 'Summary output must remain in LocalOCR')
    require(output.suffix == '.json', 'Anonymous summary output must use .json')
    require(not str(args.output).startswith(('\\\\', '//')), 'Output must remain local')
    require(not args.fixture or output.is_relative_to(ARTIFACTS.resolve()), 'Fixture summaries must remain in local artifacts')
    receipt_path = output.with_name(output.stem + '.build-receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'Summary and receipt output must be fresh')
    summary, receipt = build(args.dev_report, args.run, args.heldout_report, args.fixture)
    encoded = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    receipt['summary_sha256'] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded)
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'fixture': args.fixture, 'training_runs': len(summary['training']),
                      'heldout_models': len(summary['heldout']['models']) if summary['heldout'] else 0,
                      'summary_sha256': receipt['summary_sha256']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
