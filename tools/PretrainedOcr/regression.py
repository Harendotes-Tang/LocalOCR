"""Previously viewed original synthetic regression; fixed choices, actual CPU.

Only the legacy regression split is read. Its full rendered canvases retain raw
Unicode and spaces. This development evidence cannot establish generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from . import evaluate as e
from .common import confined, digest, disk_guard, read, utc

SCOPE = 'old-synthetic-development-regression'
SOURCE_FILES = (
    'tools/PretrainedOcr/regression.py', 'tools/PretrainedOcr/evaluate.py',
    'tools/PretrainedOcr/common.py', 'tools/OcrDatasets/evaluate.py',
    'tools/OcrDatasets/validate.py', 'tools/OcrDatasets/private.py',
    'scripts/local-ai-ocr-worker.py', 'localocr/__init__.py',
    'localocr/bundles.py', 'localocr/version.py',
)
KINDS = ('plain', 'face', 'punctuation', 'emoji', 'blank', 'graphic')


def source_binding():
    files = {name: digest(e.REPOSITORY / name) for name in SOURCE_FILES}
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'files': files, 'sha256': hashlib.sha256(encoded).hexdigest()}


def regression_rows(root):
    """Bind the old regression bytes and decoded pixels, without reading other splits."""
    root = e.local_path(root)
    metadata = read(root / 'dataset.json')
    frozen = read(root / 'freeze.json')
    if (metadata.get('schema_version') != 'localocr.dataset.v1'
            or metadata.get('synthetic_only') is not True
            or metadata.get('license') != 'CC0-1.0'
            or frozen.get('schema_version') != 'localocr.freeze.v1'
            or digest(root / 'dataset.json') != frozen.get('dataset_sha256')):
        raise ValueError('Legacy original synthetic dataset differs from freeze')
    item = metadata.get('splits', {}).get('regression', {})
    if (item.get('file') != 'regression.json' or type(item.get('count')) is not int
            or item['count'] <= 0):
        raise ValueError('Only the legacy regression split is accepted')
    path = confined(root, item['file'])
    checksum = digest(path)
    if checksum != item.get('sha256') or checksum != frozen.get('split_sha256', {}).get('regression'):
        raise ValueError('Legacy regression split differs from freeze')
    original = read(path)
    if not isinstance(original, list) or len(original) != item['count']:
        raise ValueError('Legacy regression row count differs')
    rows, records, identities = [], [], set()
    for record in original:
        text = record.get('label')
        kind = record.get('kind')
        identity = record.get('id')
        if (not isinstance(identity, str) or not identity or identity in identities
                or kind not in KINDS or record.get('usage') != 'regression'
                or record.get('provenance') != {'source': 'original', 'license': 'CC0-1.0'}
                or not isinstance(text, str)
                or hashlib.sha256(text.encode('utf-8')).hexdigest() != record.get('label_sha256')):
            raise ValueError('Legacy exact label, original provenance or identity differs')
        identities.add(identity)
        negative = kind in ('blank', 'graphic')
        group = ('negative:' + record.get('recipe_id', '')) if negative else 'text:' + record['label_sha256']
        if record.get('group') != group or (negative and (text or not record.get('recipe_id'))) or (not negative and not text):
            raise ValueError('Legacy label/group contract differs')
        image_path = confined(root, record['path'])
        if digest(image_path) != record.get('sha256'):
            raise ValueError('Legacy rendered image bytes differ')
        with Image.open(image_path) as image:
            image.load()
            rgb = image.convert('RGB')
        if (type(record.get('width')) is not int or type(record.get('height')) is not int
                or rgb.size != (record['width'], record['height'])):
            raise ValueError('Legacy rendered image dimensions differ')
        pixel = hashlib.sha256(rgb.width.to_bytes(8, 'big') + rgb.height.to_bytes(8, 'big') + rgb.tobytes()).hexdigest()
        rows.append({'id': identity, 'text': text, 'text_sha256': record['label_sha256'],
                     'image': record['path'], 'sha256': record['sha256'], 'pixel_sha256': pixel,
                     'width': rgb.width, 'height': rgb.height, 'kind': kind,
                     'subtype': 'legacy-' + kind, 'family': 'legacy-' + kind, 'group_id': group})
        records.append({'id': identity, 'image_sha256': record['sha256'], 'pixel_sha256': pixel,
                        'text_sha256': record['label_sha256'], 'width': rgb.width, 'height': rgb.height})
    return rows, {'dataset_file': str(root / 'dataset.json'), 'dataset_sha256': digest(root / 'dataset.json'),
                  'freeze_sha256': digest(root / 'freeze.json'), 'regression_sha256': checksum,
                  'row_count': len(rows), 'records': records,
                  'scope': SCOPE, 'independent_heldout': False, 'other_splits_read': False,
                  'canvas': 'full rendered canvas; annotated region does not select a crop'}


def frozen_choices(selection_path, candidates, baseline):
    path = e.local_path(selection_path)
    selection = read(path)
    if (selection.get('schema_version') != e.SELECTION_SCHEMA
            or selection.get('usage') != 'research-only-selected-for-evaluation'
            or selection.get('promote') is not False
            or selection.get('all_bound_candidates_evaluated_together') is not True
            or not selection.get('frozen_utc')
            or selection.get('baseline_binding') != baseline):
        raise ValueError('Old regression requires matching frozen research selection')
    bound = selection.get('candidate_bindings')
    if not isinstance(bound, list):
        raise ValueError('Frozen candidate inventory missing')
    final = [candidate for candidate in bound if candidate.get('stage') == 'finetuned']
    if (len(final) != 2 or len({candidate.get('family') for candidate in final}) != 2
            or len(candidates) != 2 or len({candidate['id'] for candidate in candidates}) != 2
            or any(candidate not in final for candidate in candidates)
            or selection.get('selected_id') not in {candidate['id'] for candidate in final}):
        raise ValueError('Old regression accepts the two frozen finetuned families')
    for role in ('dev_report', 'budget_receipt'):
        reference = selection.get(role, {})
        if not isinstance(reference, dict) or not isinstance(reference.get('file'), str):
            raise ValueError('Frozen selection reference missing')
        if digest(e.local_path(reference['file'])) != reference.get('sha256'):
            raise ValueError('Frozen selection reference changed')
    development = read(e.local_path(selection['dev_report']['file']))
    if (development.get('schema_version') != e.REPORT_SCHEMA or development.get('split') != 'dev'
            or development.get('independent_heldout') is not False
            or development.get('candidate_bindings') != bound
            or development.get('baseline_binding') != baseline
            or development.get('dataset_binding') != selection.get('dataset_binding')):
        raise ValueError('Frozen development evidence differs')
    budget = read(e.local_path(selection['budget_receipt']['file']))
    used = budget.get('used_seconds')
    if (budget.get('schema_version') != 'localocr.gpu-budget.v1'
            or budget.get('maximum_seconds') != 7200 or budget.get('active') is not None
            or type(used) not in (int, float) or not math.isfinite(used) or not 0 <= used <= 7200):
        raise ValueError('Frozen selection budget must be completed')
    return {'file': str(path), 'sha256': digest(path), 'frozen_utc': selection['frozen_utc']}


def worker(args):
    output = e.local_output(args.output)
    if output.exists():
        raise ValueError('Regression worker output already exists')
    disk_guard(output)
    sources = source_binding()
    if sources['sha256'] != args.source_sha256:
        raise ValueError('Regression source inventory changed')
    rows, binding = regression_rows(args.dataset)
    if args.candidate:
        candidate = e.candidate_binding(args.candidate)
        baseline = e.baseline_binding(args.selection_baseline_models)
        all_paths = args.selected_candidate
        choices = [e.candidate_binding(path) for path in all_paths]
        frozen_choices(args.selection, choices, baseline)
        if candidate not in choices:
            raise ValueError('Regression child candidate is not frozen')
    # Baseline construction also installs this audited no-network policy.
    policy_spec = importlib.util.spec_from_file_location('localocr_regression_policy', e.REPOSITORY / 'scripts/local-ai-ocr-worker.py')
    policy = importlib.util.module_from_spec(policy_spec)
    policy_spec.loader.exec_module(policy)
    sys.addaudithook(policy.deny_network)
    measurement = e.MemoryMeasurement()
    started = time.perf_counter()
    try:
        model = e.Recognizer(args.candidate) if args.candidate else e.Baseline(args.baseline_models)
        loaded = time.perf_counter() - started
        first = e.image_for_row(args.dataset, rows[0])
        first_at = time.perf_counter()
        model.predict(first)
        first_latency = time.perf_counter() - first_at
        for _ in range(2):
            model.predict(first)
        predictions, timings = [], []
        for row in rows:
            image = e.image_for_row(args.dataset, row)
            at = time.perf_counter()
            text, score = model.predict(image)
            timings.append(time.perf_counter() - at)
            predictions.append({'id': row['id'], 'text': text, 'score': score})
        resources = {**measurement.finish(), 'session_load_seconds': loaded,
                     'cold_first_crop_seconds': first_latency,
                     'cold_load_and_first_crop_seconds': loaded + first_latency,
                     'warm': e.percentiles(timings), 'total_warm_seconds': sum(timings),
                     'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1,
                     'warmup_crops': 3, 'preprocessing_and_decode_in_latency': True,
                     'onnxruntime_version': __import__('onnxruntime').__version__,
                     'load_scope': 'model construction and dynamic probes; baseline loads three fixed sessions',
                     'score_semantics': 'raw uncalibrated recognizer score; not a replacement probability'}
        if regression_rows(args.dataset)[1] != binding or source_binding() != sources:
            raise ValueError('Regression dataset or sources changed during inference')
        current = e.candidate_binding(args.candidate) if args.candidate else e.baseline_binding(args.baseline_models)
        if model.binding != current:
            raise ValueError('Regression model changed during inference')
        e.write_new(output, {'predictions': predictions, 'tokens': model.tokens, 'resources': resources,
                             'inference_binding': model.binding, 'data_binding': binding, 'source_binding': sources})
    except BaseException:
        if measurement.thread.is_alive():
            measurement.finish()
        raise


def run_recognizer(args, output, sources, candidate=None):
    command = [sys.executable, '-m', 'tools.PretrainedOcr.regression', '_worker',
               '--dataset', str(args.dataset), '--output', str(output), '--source-sha256', sources['sha256']]
    if candidate:
        command += ['--candidate', str(candidate), '--selection', str(args.selection),
                    '--selection-baseline-models', str(args.baseline_models)]
        for path in args.candidate:
            command += ['--selected-candidate', str(path)]
    else:
        command += ['--baseline-models', str(args.baseline_models)]
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '2'
    environment['PYTHONUTF8'] = '1'
    at = time.perf_counter()
    process = subprocess.run(command, cwd=e.REPOSITORY, env=environment, capture_output=True,
                             text=True, encoding='utf-8', timeout=5400)
    if process.returncode:
        e.write_new(Path(output).with_suffix('.failure.json'), {'returncode': process.returncode, 'stderr': process.stderr[-16000:]})
        raise RuntimeError('Actual old regression CPU worker failed; see artifact diagnostic')
    result = read(output)
    result['resources']['process_wall_seconds'] = time.perf_counter() - at
    return result


def evaluate(args):
    output = e.local_output(args.output)
    if output.exists():
        raise ValueError('Old regression output already exists')
    disk_guard(output)
    rows, data = regression_rows(args.dataset)
    sources = source_binding()
    baseline = e.baseline_binding(args.baseline_models)
    candidates = [e.candidate_binding(path) for path in args.candidate]
    selection = frozen_choices(args.selection, candidates, baseline)
    output.mkdir(parents=True, exist_ok=False)
    models, prediction_files, baseline_predictions = {}, {}, None
    pairs = [('production-baseline', None, baseline)] + [(binding['id'], path, binding) for path, binding in zip(args.candidate, candidates)]
    for index, (identity, path, expected) in enumerate(pairs):
        file = output / ('baseline-predictions.json' if path is None else 'candidate-' + str(index) + '-predictions.json')
        result = run_recognizer(args, file, sources, path)
        e.validate_predictions(rows, result['predictions'])
        if (result.get('inference_binding') != expected or result.get('data_binding') != data
                or result.get('source_binding') != sources
                or any(type(p['score']) not in (int, float) for p in result['predictions'])):
            raise ValueError('Old regression child did not consume the bound inputs')
        models[identity] = {'summary': e.summary(rows, result['predictions'], result['tokens']),
                            'resources': result['resources'], 'stage': expected.get('stage', 'production-baseline')}
        if path is None:
            baseline_predictions = result['predictions']
        else:
            models[identity]['transitions_from_production'] = e.transitions(rows, baseline_predictions, result['predictions'])
        prediction_files[identity] = {'file': str(file), 'sha256': digest(file)}
    if (regression_rows(args.dataset)[1] != data or source_binding() != sources
            or e.baseline_binding(args.baseline_models) != baseline
            or [e.candidate_binding(path) for path in args.candidate] != candidates
            or frozen_choices(args.selection, candidates, baseline) != selection):
        raise ValueError('Old regression fixed inputs changed during comparison')
    e.add_comparisons_and_gates(models)
    report = {'schema_version': e.REPORT_SCHEMA, 'evaluated_utc': utc(), 'split': SCOPE,
              'scope': 'previously viewed original synthetic regression; fixed family choices; no independent generalization claim',
              'normalization': 'none; exact Unicode, case, whitespace and graphemes',
              'canvas': data['canvas'], 'data_binding': data, 'source_binding': sources,
              'selection': selection, 'baseline_binding': baseline, 'candidate_bindings': candidates,
              'prediction_files': prediction_files, 'models': models,
              'independent_heldout': False, 'private_inputs': False, 'training': False, 'promote': False, 'enabled': False}
    e.write_new(output / 'evaluation.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify-data')
    verify.add_argument('--dataset', required=True)
    check = commands.add_parser('evaluate')
    for field in ('dataset', 'selection', 'baseline-models', 'output'):
        check.add_argument('--' + field, required=True)
    check.add_argument('--candidate', action='append', required=True)
    child = commands.add_parser('_worker')
    for field in ('dataset', 'output', 'source-sha256'):
        child.add_argument('--' + field, required=True)
    choice = child.add_mutually_exclusive_group(required=True)
    choice.add_argument('--candidate')
    choice.add_argument('--baseline-models')
    child.add_argument('--selection')
    child.add_argument('--selection-baseline-models')
    child.add_argument('--selected-candidate', action='append', default=[])
    args = parser.parse_args(argv)
    if args.command == '_worker':
        worker(args)
    elif args.command == 'verify-data':
        _, binding = regression_rows(args.dataset)
        print(json.dumps({key: binding[key] for key in ('dataset_sha256', 'regression_sha256', 'row_count', 'scope', 'independent_heldout', 'other_splits_read')}))
    else:
        report = evaluate(args)
        print(json.dumps({'split': report['split'], 'independent_heldout': False,
                          'models': {identity: {'overall': model['summary']['overall'], 'resources': model['resources']}
                                     for identity, model in report['models'].items()}, 'promote': False}, ensure_ascii=True))


if __name__ == '__main__':
    main()
