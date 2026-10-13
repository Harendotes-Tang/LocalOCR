"""Exact, paired research CPU evaluation and single-use heldout receipts.

Every recognizer runs in its own process. Predictions and input paths remain in
artifacts; aggregate reports never promote or modify a production model store.
The full rendered canvas is consumed, including its leading/trailing spaces.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import unicodedata

import numpy as np
from PIL import Image
import regex

from .common import confined, dataset_binding, digest, disk_guard, read, read_split, utc, write
from . import round_profile as rp
from tools.PretrainedOcr.quality_evaluate import exact_metrics as metrics, raw_crop_gate, round_write_guard, is_round, verify_dual

REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACTS = REPOSITORY / 'artifacts'
CANDIDATE_SCHEMA = 'localocr.pretrained.candidate.v1'
REPORT_SCHEMA = 'localocr.pretrained.evaluation.v1'
SELECTION_SCHEMA = 'localocr.pretrained.selection.v1'
STAGES = ('official-pretrained', 'expanded-head-initialization', 'finetuned')


def evaluator_source_inventory():
    paths = ('tools/PretrainedOcr/evaluate.py', 'tools/PretrainedOcr/common.py',
             'tools/OcrDatasets/evaluate.py', 'scripts/local-ai-ocr-worker.py',
             'localocr/bundles.py', 'tools/PretrainedOcr/quality_evaluate.py')
    return [{'file': name, 'size': (REPOSITORY / name).stat().st_size,
             'sha256': digest(REPOSITORY / name)} for name in paths]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_cpu_result(result, rows, binding, frozen, split, source_inventory=None):
    expected_inventory = evaluator_source_inventory() if source_inventory is None else source_inventory
    if (result.get('actual_cpu') is not True or result.get('inference_binding') != binding
            or result.get('dataset_binding') != frozen or result.get('split') != split
            or result.get('source_inventory') != expected_inventory):
        raise ValueError('Actual CPU worker input/source evidence differs')
    validate_predictions(rows, result.get('predictions', []))
    tokens = result.get('tokens')
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError('Worker vocabulary evidence differs')
    resources = result.get('resources', {})
    if (resources.get('providers') != ['CPUExecutionProvider']
            or type(resources.get('intra_op_threads')) is not int or resources['intra_op_threads'] != 2
            or type(resources.get('inter_op_threads')) is not int or resources['inter_op_threads'] != 1
            or type(resources.get('warm', {}).get('count')) is not int or resources['warm']['count'] != len(rows)):
        raise ValueError('Actual CPU resource evidence differs')
    for name in ('peak_rss_bytes', 'session_load_seconds', 'cold_first_crop_seconds',
                 'cold_load_and_first_crop_seconds', 'total_warm_seconds', 'process_wall_seconds'):
        if not finite_number(resources.get(name)) or resources[name] <= 0:
            raise ValueError('Actual CPU resource evidence is missing or invalid')
    for name in ('p50_seconds', 'p95_seconds'):
        if not finite_number(resources['warm'].get(name)) or resources['warm'][name] < 0:
            raise ValueError('Actual CPU latency evidence differs')


def validate_dev_evidence(report, report_path, dataset, candidates, baseline, source_inventory=None):
    """Re-derive the development summaries from the bound worker outputs.

    ``source_inventory`` defaults to the current evaluator closure. A predecessor round's frozen
    crop report, whose identity is bound by that round's consumed first-use receipt, is checked
    against the inventory it recorded instead: its predictions are still re-summarized here.
    """
    frozen = dataset_binding(dataset)
    expected_inventory = evaluator_source_inventory() if source_inventory is None else source_inventory
    if (report.get('schema_version') != REPORT_SCHEMA or report.get('split') != 'dev'
            or report.get('independent_heldout') is not False or report.get('promote') is not False
            or report.get('enabled') is not False or report.get('candidate_bindings') != candidates
            or report.get('baseline_binding') != baseline or report.get('dataset_binding') != frozen
            or report.get('source_inventory') != expected_inventory):
        raise ValueError('Selection requires actual development evaluation evidence')
    rows = read_split(dataset, 'dev')
    evidence = report.get('worker_results')
    expected = [('production-baseline', baseline)] + [(b['id'], b) for b in candidates]
    if not rows or not isinstance(evidence, dict) or set(evidence) != {k for k, _ in expected}:
        raise ValueError('Development worker prediction evidence is missing')
    models, predictions = {}, None
    for identity, binding in expected:
        path = referenced_file(Path(report_path).parent, evidence[identity], confined_path=True)
        result = read(path)
        expected_capture = {'file': path.name, 'sha256': digest(path), 'worker_pid': result.get('worker_pid'),
                            'source_inventory': expected_inventory, 'actual_cpu': True,
                            'capture': 'successful-owned-subprocess-stdout'}
        if evidence[identity] != expected_capture or type(result.get('worker_pid')) is not int or result['worker_pid'] <= 0:
            raise ValueError('Captured worker/report completion evidence differs')
        validate_cpu_result(result, rows, binding, frozen, 'dev', expected_inventory)
        value = {'summary': summary(rows, result['predictions'], result['tokens']), 'resources': result['resources']}
        if predictions is None:
            predictions = result['predictions']
        else:
            value.update(transitions_from_production=transitions(rows, predictions, result['predictions']), stage=binding['stage'])
        models[identity] = value
    add_comparisons_and_gates(models)
    if report.get('models') != models:
        raise ValueError('Development summaries/resources differ from actual worker predictions')
    return models


def local_path(path):
    if str(path).startswith(('\\\\', '//')):
        raise ValueError('Research paths must remain local')
    return Path(path).resolve()


def local_output(path):
    result = local_path(path)
    if not result.is_relative_to(ARTIFACTS.resolve()) or result == ARTIFACTS.resolve():
        raise ValueError('Research output must remain in LocalOCR artifacts')
    return result


def write_new(path, value):
    path = local_output(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    round_write_guard(path, len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def referenced_file(root, record, confined_path=False):
    if not isinstance(record, dict) or not isinstance(record.get('file'), str):
        raise ValueError('Missing bound input file')
    path = confined(root, record['file']) if confined_path else local_path(root / record['file'])
    if not path.is_file() or digest(path) != record.get('sha256'):
        raise ValueError('Bound input checksum differs')
    return path


def candidate_binding(path):
    path = local_path(path)
    metadata_bytes = path.read_bytes()
    meta = json.loads(metadata_bytes.decode('utf-8'))
    if (meta.get('schema_version') != CANDIDATE_SCHEMA or not meta.get('id')
            or not isinstance(meta.get('id'), str) or meta.get('id') == 'production-baseline' or not meta.get('family')
            or meta.get('stage') not in STAGES):
        raise ValueError('Unknown research candidate identity or stage')
    if meta.get('promote', False) or meta.get('enabled', False):
        raise ValueError('Research candidate cannot enable production')
    model = referenced_file(path.parent, meta.get('model'), True)
    vocabulary = referenced_file(path.parent, meta.get('vocabulary'), True)
    vocabulary_bytes = vocabulary.read_bytes()
    if hashlib.sha256(vocabulary_bytes).hexdigest() != meta['vocabulary']['sha256']:
        raise ValueError('Vocabulary snapshot checksum differs')
    vocab = json.loads(vocabulary_bytes.decode('utf-8'))
    tokens = vocab.get('tokens')
    if (vocab.get('schema_version') != 'localocr.pretrained.vocabulary.v1'
            or not isinstance(tokens, list) or len(tokens) < 2
            or any(not isinstance(t, str) for t in tokens)
            or len(tokens) != len(set(tokens)) or tokens[0] != '<blank>'
            or any(len(t) != 1 or 0xD800 <= ord(t) <= 0xDFFF for t in tokens[1:])):
        raise ValueError('Vocabulary must preserve unique exact Unicode scalar tokens')
    blank, space = vocab.get('blank_id'), vocab.get('space_id')
    if type(blank) is not int or blank != 0 or type(meta['vocabulary'].get('blank_id')) is not int or meta['vocabulary'].get('blank_id') != blank:
        raise ValueError('Blank identity differs')
    if (not isinstance(space, int) or isinstance(space, bool) or space <= 0
            or space >= len(tokens) or tokens[space] != ' '
            or meta['vocabulary'].get('space_id') != space):
        raise ValueError('Space identity differs')
    validate_preprocessing(meta.get('preprocessing'))
    decoder = meta.get('decoder')
    if (not isinstance(decoder, dict) or decoder.get('type') != 'ctc'
            or decoder.get('repeat') != 'ctc' or type(decoder.get('blank_id')) is not int or decoder.get('blank_id') != 0
            or decoder.get('output') not in ('probabilities', 'logits')):
        raise ValueError('Unsupported explicit CTC decoder')
    source = meta.get('source')
    if not isinstance(source, dict):
        raise ValueError('Candidate needs bound source inputs')
    for role in ('checkpoint', 'dictionary'):
        referenced_file(path.parent, source.get(role))
    references = {}
    def inspect(value, name):
        if isinstance(value, dict):
            if 'file' in value:
                actual = referenced_file(path.parent, value)
                references[name] = {'file': str(actual), 'sha256': digest(actual)}
            else:
                for key, item in value.items():
                    inspect(item, name + '.' + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                inspect(item, name + '.' + str(i))
    inspect(source, 'source')
    return {'id': meta['id'], 'family': meta['family'], 'stage': meta['stage'],
            'candidate_file': str(path), 'candidate_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
            'model_sha256': digest(model), 'model_bytes': model.stat().st_size,
            'vocabulary_sha256': hashlib.sha256(vocabulary_bytes).hexdigest(), 'token_count': len(tokens),
            'preprocessing': meta['preprocessing'], 'decoder': decoder,
            'source': source, 'source_files': references}


def validate_preprocessing(meta):
    if (not isinstance(meta, dict) or meta.get('height') != 48
            or meta.get('color') not in ('RGB', 'BGR')
            or meta.get('padding') != 'right-zero-normalized'
            or meta.get('resize') not in ('PIL-bilinear', 'cv2-linear')):
        raise ValueError('Unsupported explicit image preprocessing')
    width = meta.get('width', {})
    low, high, multiple = width.get('min'), width.get('max'), width.get('multiple')
    if (width.get('mode') != 'dynamic' or width.get('rounding') != 'ceil-multiple'
            or not isinstance(low, int) or isinstance(low, bool) or low < 1
            or not isinstance(multiple, int) or isinstance(multiple, bool) or multiple < 1
            or low % multiple
            or (high is not None and (not isinstance(high, int) or isinstance(high, bool)
                                      or high < low or high % multiple))):
        raise ValueError('Unsupported dynamic width policy')
    normalization = meta.get('normalize', {})
    mean, std = normalization.get('mean'), normalization.get('std')
    if (not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in mean + std)
            or any(x <= 0 for x in std)):
        raise ValueError('Invalid explicit normalization')


def prepare_image(image, preprocessing):
    """Return float32 CHW; clip only the declared maximum, never crop the canvas."""
    validate_preprocessing(preprocessing)
    rgb = image.convert('RGB')
    if rgb.width < 1 or rgb.height < 1:
        raise ValueError('Empty research image')
    width = preprocessing['width']
    natural = max(1, math.ceil(48 * rgb.width / rgb.height))
    content = min(natural, width['max']) if width['max'] is not None else natural
    target = max(width['min'], math.ceil(content / width['multiple']) * width['multiple'])
    if width['max'] is not None:
        target = min(target, width['max'])
    if preprocessing['resize'] == 'PIL-bilinear':
        pixels = np.asarray(rgb.resize((content, 48), Image.Resampling.BILINEAR))
    else:
        import cv2
        cv2.setNumThreads(2)
        pixels = cv2.resize(np.asarray(rgb), (content, 48), interpolation=cv2.INTER_LINEAR)
    if preprocessing['color'] == 'BGR':
        pixels = pixels[:, :, ::-1]
    normalized = pixels.astype(np.float32).transpose(2, 0, 1) / 255.0
    normalization = preprocessing['normalize']
    normalized = ((normalized - np.asarray(normalization['mean'], dtype=np.float32)[:, None, None])
                  / np.asarray(normalization['std'], dtype=np.float32)[:, None, None])
    result = np.zeros((3, 48, target), dtype=np.float32)
    result[:, :, :content] = normalized
    return result


def decode_ctc(output, tokens, decoder):
    values = np.asarray(output)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(tokens) or not np.isfinite(values).all():
        raise ValueError('Invalid finite CTC T/C output')
    if decoder['output'] == 'logits':
        exp = np.exp(values.astype(np.float64) - values.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    else:
        probabilities = values
        if (values.min() < -1e-6 or values.max() > 1 + 1e-6
                or not np.allclose(values.sum(axis=1), 1, atol=1e-4)):
            raise ValueError('Declared probabilities are not probabilities')
    ids = probabilities.argmax(axis=1)
    selected = [(int(value), float(probabilities[i, value])) for i, value in enumerate(ids)
                if value != decoder['blank_id'] and (i == 0 or value != ids[i - 1])]
    return ''.join(tokens[value] for value, _ in selected), (statistics.mean(score for _, score in selected) if selected else 0.0)


class Recognizer:
    def __init__(self, path):
        import onnxruntime as ort
        self.binding = candidate_binding(path)
        metadata_bytes = Path(path).read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != self.binding['candidate_sha256']:
            raise ValueError('Candidate snapshot changed')
        self.metadata = json.loads(metadata_bytes.decode('utf-8'))
        vocabulary_bytes = confined(Path(path).parent, self.metadata['vocabulary']['file']).read_bytes()
        if hashlib.sha256(vocabulary_bytes).hexdigest() != self.binding['vocabulary_sha256']:
            raise ValueError('Vocabulary snapshot changed')
        self.tokens = json.loads(vocabulary_bytes.decode('utf-8'))['tokens']
        model_bytes = confined(Path(path).parent, self.metadata['model']['file']).read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != self.binding['model_sha256']:
            raise ValueError('Model snapshot changed')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_bytes,
                                            options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        dynamic = lambda value: value is None or isinstance(value, str)
        if (self.session.get_providers() != ['CPUExecutionProvider'] or len(inputs) != 1 or len(outputs) != 1
                or inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)'):
            raise ValueError('Research ONNX must be one float CPU input/output')
        shape, output = inputs[0].shape, outputs[0].shape
        if (len(shape) != 4 or shape[1:3] != [3, 48] or not dynamic(shape[0]) or not dynamic(shape[3])
                or len(output) != 3 or not dynamic(output[0]) or not dynamic(output[1])
                or output[2] != len(self.tokens)):
            raise ValueError('Research ONNX requires dynamic N/W and CTC N/T/C axes')
        self.input_name = inputs[0].name
        times = []
        for batch, width in ((1, 64), (2, 320)):
            probe = self.session.run(None, {self.input_name: np.zeros((batch, 3, 48, width), dtype=np.float32)})[0]
            if probe.ndim != 3 or probe.shape[0] != batch or probe.shape[2] != len(self.tokens) or not np.isfinite(probe).all():
                raise ValueError('Actual dynamic batch/CTC probe failed')
            decode_ctc(probe[0], self.tokens, self.metadata['decoder'])
            times.append(probe.shape[1])
        if times[0] == times[1]:
            raise ValueError('Actual CTC time did not vary with width')

    def predict(self, image):
        tensor = prepare_image(image, self.metadata['preprocessing'])
        result = self.session.run(None, {self.input_name: tensor[None]})[0]
        if result.ndim != 3 or result.shape[0] != 1:
            raise ValueError('Research output lost batch identity')
        return decode_ctc(result[0], self.tokens, self.metadata['decoder'])


def baseline_binding(root):
    from localocr.bundles import validate_bundle
    root = local_path(root)
    bundle = validate_bundle(root)
    return {'id': 'production-baseline', 'version': bundle.manifest['version'],
            'manifest_file': str(root / 'manifest.json'),
            'manifest_sha256': digest(root / 'manifest.json'),
            'models': {role: {'sha256': digest(root / entry['file']), 'bytes': (root / entry['file']).stat().st_size}
                       for role, entry in bundle.manifest['models'].items()}}


class Baseline:
    def __init__(self, root):
        self.binding = baseline_binding(root)
        path = REPOSITORY / 'scripts' / 'local-ai-ocr-worker.py'
        spec = importlib.util.spec_from_file_location('localocr_research_baseline', path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        self.engine = worker.create_engine(local_path(root), local_path(root) / 'manifest.json')
        self.recognizer = self.engine._load_rec_model()
        self.tokens = self.recognizer.postprocess_op.character[1:]

    def predict(self, image):
        from rapidocr.ch_ppocr_rec import TextRecInput
        result = self.recognizer(TextRecInput(img=np.asarray(image.convert('RGB'))[:, :, ::-1].copy(), return_word_box=False))
        if len(result.txts) != 1 or len(result.scores) != 1 or not math.isfinite(float(result.scores[0])):
            raise ValueError('Production raw recognizer output differs')
        return str(result.txts[0]), float(result.scores[0])


def image_for_row(root, row):
    path = confined(root, row['image'])
    if digest(path) != row['sha256']:
        raise ValueError('Frozen crop bytes changed')
    with Image.open(path) as source:
        source.load()
        image = source.convert('RGB')
    pixel = hashlib.sha256(image.width.to_bytes(8, 'big') + image.height.to_bytes(8, 'big') + image.tobytes()).hexdigest()
    if image.size != (row['width'], row['height']) or pixel != row['pixel_sha256']:
        raise ValueError('Frozen crop dimensions/pixels changed')
    return image


def percentiles(times):
    if not times:
        return {'count': 0, 'p50_seconds': None, 'p95_seconds': None}
    return {'count': len(times), 'p50_seconds': statistics.median(times),
            'p95_seconds': float(np.percentile(times, 95))}


class MemoryMeasurement:
    """Windows native peak RSS plus sampled private bytes; sampling is explicit."""
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.before = self.process.memory_info()
        self.private = getattr(self.before, 'private', None)
        self.rss = self.before.rss
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stopped.wait(0.01):
            info = self.process.memory_info()
            self.rss = max(self.rss, info.rss)
            if hasattr(info, 'private'):
                self.private = max(self.private, info.private)

    def finish(self):
        self.stopped.set()
        self.thread.join()
        info = self.process.memory_info()
        return {'platform': sys.platform, 'peak_rss_bytes': max(self.rss, getattr(info, 'peak_wset', info.rss)),
                'rss_measurement': 'Windows native peak_wset' if hasattr(info, 'peak_wset') else 'sampled RSS',
                'peak_private_bytes': max(self.private, info.private) if hasattr(info, 'private') else None,
                'private_measurement': 'sampled every 0.01 seconds; current private at exit' if hasattr(info, 'private') else 'unavailable',
                'peak_commit_bytes': getattr(info, 'peak_pagefile', None),
                'initial_rss_bytes': self.before.rss, 'initial_private_bytes': getattr(self.before, 'private', None)}


def worker(args):
    output = local_output(args.output)
    if output.exists():
        raise ValueError('Worker output already exists')
    disk_guard(output)
    private_images = None
    if args.private_manifest:
        rows, private_images, _ = private_rows(args.private_manifest)
        root = None
    else:
        root = local_path(args.dataset)
    if not args.private_manifest and args.split == 'heldout':
        validate_worker_heldout(root, args.heldout_usage_sha256, args.candidate, args.baseline_models, output=output, consume=True)
    if not args.private_manifest:
        rows = read_split(root, args.split, allow_heldout=args.split == 'heldout')
    if not rows:
        raise ValueError('Empty evaluation split')
    source_inventory = evaluator_source_inventory()
    measurement = MemoryMeasurement()
    started = time.perf_counter()
    try:
        if args.candidate:
            spec = importlib.util.spec_from_file_location('localocr_research_policy', REPOSITORY / 'scripts' / 'local-ai-ocr-worker.py')
            policy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(policy)
            sys.addaudithook(policy.deny_network)
        model = Baseline(args.baseline_models) if args.baseline_models else Recognizer(args.candidate)
        loaded = time.perf_counter() - started
        first = private_images[0] if private_images is not None else image_for_row(root, rows[0])
        first_started = time.perf_counter()
        model.predict(first)
        first_latency = time.perf_counter() - first_started
        # Warm-up executes real inference and is excluded from steady-state timing.
        for _ in range(2):
            model.predict(first)
        predictions, timings = [], []
        for index, row in enumerate(rows):
            image = private_images[index] if private_images is not None else image_for_row(root, row)
            at = time.perf_counter()
            text, score = model.predict(image)
            elapsed = time.perf_counter() - at
            timings.append(elapsed)
            predictions.append({'id': row['id'], 'text': text, 'score': score})
        resources = {**measurement.finish(), 'session_load_seconds': loaded,
                     'cold_first_crop_seconds': first_latency, 'cold_load_and_first_crop_seconds': loaded + first_latency,
                     'warm': percentiles(timings), 'total_warm_seconds': sum(timings),
                     'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1,
                     'preprocessing_and_decode_in_latency': True, 'warmup_crops': 3,
                     'load_scope': 'model construction plus interface/dynamic validation; baseline loads its three fixed sessions',
                     'score_semantics': 'mean retained CTC maxima; uncalibrated; production retains RapidOCR supplied score'}
        if source_inventory != evaluator_source_inventory():
            raise ValueError('Worker source changed during inference')
        resources['process_wall_seconds'] = time.perf_counter() - started
        resources['onnxruntime_version'] = __import__('onnxruntime').__version__
        write_new(output, {'predictions': predictions, 'resources': resources, 'tokens': model.tokens, 'inference_binding': model.binding,
                           'actual_cpu': True, 'dataset_binding': dataset_binding(root) if root else None,
                           'split': args.split, 'source_inventory': source_inventory, 'worker_pid': os.getpid()})
        print('LOCAL_OCR_CPU_RECEIPT ' + json.dumps({'file': output.name, 'sha256': digest(output),
              'worker_pid': os.getpid(), 'source_inventory': source_inventory, 'actual_cpu': True}, allow_nan=False), flush=True)
    except BaseException:
        if measurement.thread.is_alive():
            measurement.finish()
        raise


def run_recognizer(dataset, split, output, candidate=None, baseline=None, first_use=None, private_manifest=None):
    command = [sys.executable, '-m', 'tools.PretrainedOcr.evaluate', '_worker',
               '--split', split, '--output', str(output)]
    command += ['--private-manifest', str(private_manifest)] if private_manifest else ['--dataset', str(dataset)]
    command += ['--candidate', str(candidate)] if candidate else ['--baseline-models', str(baseline)]
    if first_use is not None:
        command += ['--heldout-usage-sha256', first_use['sha256']]
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '2'
    environment['PYTHONUTF8'] = '1'
    at = time.perf_counter()
    process = subprocess.run(command, cwd=REPOSITORY, env=environment, capture_output=True, text=True,
                             encoding='utf-8', timeout=5400)
    if process.returncode:
        # Preserve diagnostic data locally, without printing sample predictions.
        write_new(Path(output).with_suffix('.failure.json'), {'returncode': process.returncode, 'stderr': process.stderr[-16000:]})
        raise RuntimeError('Actual CPU recognizer failed; see artifact diagnostic')
    receipts = [line[len('LOCAL_OCR_CPU_RECEIPT '):] for line in process.stdout.splitlines() if line.startswith('LOCAL_OCR_CPU_RECEIPT ')]
    if len(receipts) != 1:
        raise ValueError('Actual CPU child did not capture one completion receipt')
    captured = json.loads(receipts[0])
    result = read(output)
    expected = {'file': Path(output).name, 'sha256': digest(output), 'worker_pid': result.get('worker_pid'),
                'source_inventory': evaluator_source_inventory(), 'actual_cpu': True}
    if captured != expected or result.get('source_inventory') != expected['source_inventory'] or result.get('actual_cpu') is not True:
        raise ValueError('Captured CPU worker completion/input evidence differs')
    result['process_wall_seconds'] = time.perf_counter() - at
    result['worker_file'] = {**captured, 'capture': 'successful-owned-subprocess-stdout'}
    return result


def summary(rows, predictions, tokens):
    paired = [{**row, 'label': row['text'], 'prediction': prediction['text']} for row, prediction in zip(rows, predictions) if not row.get('ambiguous', False)]
    vocabulary = set(tokens) - {'<blank>', 'blank'}
    def item(part):
        labels = ''.join(row['label'] for row in part)
        unsupported = sum(character not in vocabulary for character in labels)
        groups = defaultdict(list)
        for row in part:
            groups[row['group_id']].append(row)
        return {**metrics(part, 'prediction'), 'groups': len(groups),
                'groups_all_variants_exact': sum(all(r['label'] == r['prediction'] for r in group) for group in groups.values()),
                'vocabulary_codepoints': len(vocabulary), 'label_codepoints': len(labels),
                'unsupported_codepoint_occurrences': unsupported,
                'fully_covered_rows': sum(all(c in vocabulary for c in r['label']) for r in part),
                'false_nonempty_outputs': sum(bool(r['prediction']) for r in part if r['kind'] in ('blank', 'graphic'))}
    result = {'overall': item(paired), 'total_images': len(rows), 'ambiguous_excluded': len(rows) - len(paired)}
    for field in ('kind', 'subtype', 'family', 'group_id'):
        partition = defaultdict(list)
        for row in paired:
            partition[row[field]].append(row)
        result['by_' + field] = {key: item(partition[key]) for key in sorted(partition)}
    repeats = lambda text: any(a == b for a, b in zip(regex.findall(r'\X', text), regex.findall(r'\X', text)[1:]))
    result['repeated_graphemes'] = item([r for r in paired if repeats(r['label'])])
    result['combining_or_joined'] = item([r for r in paired if any(unicodedata.combining(c) or c in ('\u200d', '\ufe0f') for c in r['label'])])
    result['emoji'] = item([r for r in paired if r['kind'] == 'emoji'])
    return result


def transitions(rows, baseline, candidate):
    def item(indices):
        return {'count': len(indices),
                'baseline_correct_to_wrong': sum(baseline[i]['text'] == rows[i]['text'] and candidate[i]['text'] != rows[i]['text'] for i in indices),
                'erroneous_rewrites': sum(candidate[i]['text'] != baseline[i]['text'] and candidate[i]['text'] != rows[i]['text'] for i in indices),
                'baseline_wrong_to_correct': sum(baseline[i]['text'] != rows[i]['text'] and candidate[i]['text'] == rows[i]['text'] for i in indices),
                'negative_false_outputs': sum(bool(candidate[i]['text']) for i in indices if rows[i]['kind'] in ('blank', 'graphic')),
                'new_negative_false_outputs': sum(not baseline[i]['text'] and bool(candidate[i]['text']) for i in indices if rows[i]['kind'] in ('blank', 'graphic'))}
    exact = [i for i, r in enumerate(rows) if not r.get('ambiguous', False)]
    return {'overall': item(exact), 'ambiguous_excluded': len(rows) - len(exact), 'by_kind': {kind: item([i for i in exact if rows[i]['kind'] == kind]) for kind in sorted({rows[i]['kind'] for i in exact})}}


def validate_predictions(rows, predictions):
    if (len(rows) != len(predictions) or any(row['id'] != p.get('id') or not isinstance(p.get('text'), str)
            or not isinstance(p.get('score'), (int, float)) or not math.isfinite(p['score']) for row, p in zip(rows, predictions))):
        raise ValueError('Actual prediction identities/count/scores differ')


def add_comparisons_and_gates(models):
    baseline = models['production-baseline']['summary']['by_kind']
    for identity, model in models.items():
        if identity == 'production-baseline':
            continue
        categories = model['summary']['by_kind']
        moves = model['transitions_from_production']['by_kind']
        plain_kinds = [k for k in ('plain', 'punctuation') if k in categories]
        face = categories.get('face', {})
        original = baseline.get('face', {})
        gains = {'exact_gain': face.get('exact', 0) - original.get('exact', 0),
                 'cer_reduction': original['cer'] - face['cer'] if original.get('cer') is not None and face.get('cer') is not None else None,
                 'symbol_cer_reduction': original['symbol_cer'] - face['symbol_cer'] if original.get('symbol_cer') is not None and face.get('symbol_cer') is not None else None}
        model['face_change_from_production'] = gains
        model['quality_gates'] = {
            'ordinary_text_protection': categories.get('plain', {}).get('count', 0) > 0 and all(moves[k]['baseline_correct_to_wrong'] == 0 and moves[k]['erroneous_rewrites'] == 0 for k in plain_kinds),
            'negative_protection': all(categories.get(k, {}).get('count', 0) > 0 and moves[k]['negative_false_outputs'] == 0 and moves[k]['new_negative_false_outputs'] == 0 for k in ('blank', 'graphic')),
            'face_improvement': bool(face.get('count')) and gains['exact_gain'] > 0 and gains['cer_reduction'] is not None and gains['cer_reduction'] > 0 and gains['symbol_cer_reduction'] is not None and gains['symbol_cer_reduction'] > 0,
            'emoji_threshold': None, 'production_delivery_qualified': False,
            'delivery_limit': 'research only; synthetic crops do not verify real ROI/APP/production resources/license'}
        if all(k in categories for k in ('face','plain','punctuation','blank','graphic')):
            model['research_quality_gates']=raw_crop_gate(models['production-baseline'],model)


def private_rows(manifest):
    from tools.OcrDatasets.private import read_private
    original, images, binding = read_private(local_path(manifest))
    rows = [{**row, 'text': row['label'], 'subtype': 'private-manual-crop', 'family': 'private-' + row['kind'],
             'group_id': row['id']} for row in original]
    return rows, images, binding


def private_evaluate(args):
    output = local_output(args.output)
    if output.exists():
        raise ValueError('Evaluation output already exists')
    disk_guard(output)
    bindings = [candidate_binding(path) for path in args.candidate]
    if len({b['id'] for b in bindings}) != len(bindings):
        raise ValueError('Candidate ids must be unique')
    baseline = baseline_binding(args.baseline_models)
    rows, _, binding = private_rows(args.private_manifest)
    output.mkdir(parents=True, exist_ok=False)
    base = run_recognizer(None, 'dev', output / 'baseline-predictions.json', baseline=args.baseline_models, private_manifest=args.private_manifest)
    validate_predictions(rows, base['predictions'])
    if base.get('inference_binding') != baseline:
        raise ValueError('Baseline child loaded different bytes')
    models = {'production-baseline': {'summary': summary(rows, base['predictions'], base['tokens']), 'resources': base['resources']}}
    for index, (path, candidate) in enumerate(zip(args.candidate, bindings)):
        result = run_recognizer(None, 'dev', output / f'candidate-{index}-predictions.json', candidate=local_path(path), private_manifest=args.private_manifest)
        validate_predictions(rows, result['predictions'])
        if result.get('inference_binding') != candidate:
            raise ValueError('Candidate child loaded different bytes')
        models[candidate['id']] = {'summary': summary(rows, result['predictions'], result['tokens']),
                                  'transitions_from_production': transitions(rows, base['predictions'], result['predictions']),
                                  'resources': result['resources'], 'stage': candidate['stage']}
    if private_rows(args.private_manifest)[2] != binding or bindings != [candidate_binding(path) for path in args.candidate] or baseline_binding(args.baseline_models) != baseline:
        raise ValueError('Private/model inputs changed during development regression')
    add_comparisons_and_gates(models)
    report = {'schema_version': REPORT_SCHEMA, 'evaluated_utc': utc(), 'split': 'private-development-regression',
              'private_binding': binding, 'candidate_bindings': bindings, 'baseline_binding': baseline,
              'models': models, 'promote': False, 'enabled': False, 'independent_heldout': False,
              'scope': 'previously seen manually supplied private crops; ambiguous labels excluded; not fresh independent acceptance',
              'normalization': 'none; exact Unicode, case, whitespace and graphemes'}
    write_new(output / 'evaluation.json', report)
    return report


def selection_for_heldout(dataset, selection_path, candidates, baseline, output_dir=None, dual_freeze=None):
    dual=verify_dual(dual_freeze,'crop',selection_path,dataset) if is_round(dataset) and dual_freeze else None
    if is_round(dataset) and dual is None:raise ValueError('New round heldout requires dual freeze before access')
    receipt = read(selection_path)
    frozen = dataset_binding(dataset)
    if (receipt.get('schema_version') != SELECTION_SCHEMA or receipt.get('dataset_binding') != frozen
            or receipt.get('candidate_bindings') != candidates or receipt.get('baseline_binding') != baseline
            or receipt.get('promote') is not False or receipt.get('usage') != 'research-only-selected-for-evaluation'):
        raise ValueError('Heldout requires unchanged frozen selection inputs')
    at = datetime.fromisoformat(receipt.get('frozen_utc', ''))
    if at.tzinfo is None or at > datetime.now(timezone.utc):
        raise ValueError('Selection needs an elapsed aware frozen timestamp')
    for name in ('dev_report', 'budget_receipt'):
        referenced_file(Path(selection_path).parent, receipt[name])
    dev = read(receipt['dev_report']['file'])
    if (dev.get('split') != 'dev' or dev.get('candidate_bindings') != candidates
            or dev.get('baseline_binding') != baseline or dev.get('dataset_binding') != frozen):
        raise ValueError('Selection dev evidence differs')
    validate_dev_evidence(dev, receipt['dev_report']['file'], dataset, candidates, baseline)
    seal_path = Path(dataset) / 'heldout-seal.json'
    seal = read(seal_path)
    if (seal.get('schema_version') != 'localocr.heldout-seal.v1' or seal.get('results_seen') is not False
            or seal.get('labels_used_for_vocabulary') is not False or seal.get('dataset_sha256') != frozen['dataset_sha256']
            or seal.get('heldout_sha256') != frozen['split_sha256']['heldout']):
        raise ValueError('Heldout is not sealed unseen data')
    usage = local_output(Path(dataset) / 'heldout-first-use.json')
    output_dir = local_output(output_dir or Path(dataset) / 'heldout-evaluation')
    slots = [{'kind': 'baseline', 'binding': baseline, 'output': str(output_dir / 'baseline-predictions.json')}]
    slots.extend({'kind': 'candidate', 'binding': binding, 'output': str(output_dir / f'candidate-{index}-predictions.json')}
                 for index, binding in enumerate(candidates))
    for index, slot in enumerate(slots):
        slot['key'] = str(index)
    claim = {'schema_version': 'localocr.pretrained.heldout-first-use.v1', 'claimed_utc': utc(),
             'selection_file': str(local_path(selection_path)), 'selection_sha256': digest(selection_path), 'seal_sha256': digest(seal_path),
             'dataset_binding': frozen, 'status': 'consumed-before-reading; failures remain consumed', 'pid': os.getpid(),
             'worker_slots': slots, 'dual_freeze':dual}
    write_new(usage, claim)
    return {'file': str(usage), 'sha256': digest(usage), 'selection_sha256': digest(selection_path)}


def owned_worker_parent(owner_pid):
    if os.getppid() == owner_pid:
        return True
    # Windows venv python.exe is a redirector: the actual Python child belongs
    # to exactly one known launcher, whose parent is the claiming evaluator.
    if sys.platform != 'win32':
        return False
    import psutil
    try:
        parent = psutil.Process(os.getppid())
        command = parent.cmdline()
        return (parent.ppid() == owner_pid and local_path(parent.exe()) == local_path(sys.executable)
                and command[1:4] == ['-m', 'tools.PretrainedOcr.evaluate', '_worker'])
    except (psutil.Error, OSError):
        return False


def validate_worker_heldout(dataset, expected_usage_sha256, candidate, baseline, output=None, consume=False):
    usage_path = Path(dataset) / 'heldout-first-use.json'
    if not expected_usage_sha256 or digest(usage_path) != expected_usage_sha256:
        raise ValueError('Heldout worker requires the claimed first-use digest')
    claim = read(usage_path)
    frozen = dataset_binding(dataset)
    if claim.get('schema_version') != 'localocr.pretrained.heldout-first-use.v1' or claim.get('dataset_binding') != frozen:
        raise ValueError('Heldout worker claim differs')
    selection_path = local_path(claim['selection_file'])
    if is_round(dataset):
        proof=claim.get('dual_freeze')
        if not proof or digest(proof['file'])!=proof['sha256'] or verify_dual(proof['file'],'crop',selection_path,dataset)!=proof:raise ValueError('Worker dual freeze differs')
    if digest(selection_path) != claim.get('selection_sha256'):
        raise ValueError('Heldout worker selection changed')
    receipt = read(selection_path)
    if receipt.get('schema_version') != SELECTION_SCHEMA or receipt.get('dataset_binding') != frozen or receipt.get('promote') is not False:
        raise ValueError('Heldout worker selection identity differs')
    if candidate and candidate_binding(candidate) not in receipt['candidate_bindings']:
        raise ValueError('Heldout worker candidate not selected')
    if baseline and baseline_binding(baseline) != receipt['baseline_binding']:
        raise ValueError('Heldout worker baseline differs')
    frozen_at = datetime.fromisoformat(receipt.get('frozen_utc', ''))
    if frozen_at.tzinfo is None or frozen_at > datetime.now(timezone.utc):
        raise ValueError('Heldout worker selection timestamp differs')
    seal_path = Path(dataset) / 'heldout-seal.json'
    seal = read(seal_path)
    if (digest(seal_path) != claim.get('seal_sha256') or seal.get('schema_version') != 'localocr.heldout-seal.v1'
            or seal.get('results_seen') is not False or seal.get('labels_used_for_vocabulary') is not False
            or seal.get('dataset_sha256') != frozen['dataset_sha256']
            or seal.get('heldout_sha256') != frozen['split_sha256']['heldout']):
        raise ValueError('Heldout worker seal changed or was previously seen')
    for name in ('dev_report', 'budget_receipt'):
        referenced_file(selection_path.parent, receipt[name])
    dev = read(receipt['dev_report']['file'])
    if (dev.get('schema_version') != REPORT_SCHEMA or dev.get('split') != 'dev'
            or dev.get('candidate_bindings') != receipt['candidate_bindings']
            or dev.get('baseline_binding') != receipt['baseline_binding'] or dev.get('dataset_binding') != frozen):
        raise ValueError('Heldout worker development selection evidence differs')
    validate_dev_evidence(dev, receipt['dev_report']['file'], dataset, receipt['candidate_bindings'], receipt['baseline_binding'])
    if any(candidate_binding(item['candidate_file']) != item for item in receipt['candidate_bindings']):
        raise ValueError('Heldout worker another selected candidate changed')
    if receipt['baseline_binding'].get('manifest_file') and baseline_binding(Path(receipt['baseline_binding']['manifest_file']).parent) != receipt['baseline_binding']:
        raise ValueError('Heldout worker production baseline changed')
    if consume:
        if not owned_worker_parent(claim['pid']):
            raise ValueError('Heldout worker must belong to the claiming evaluation process')
        binding = candidate_binding(candidate) if candidate else baseline_binding(baseline)
        kind = 'candidate' if candidate else 'baseline'
        slots = [slot for slot in claim.get('worker_slots', []) if slot.get('kind') == kind and slot.get('binding') == binding]
        if len(slots) != 1 or local_output(output) != local_output(slots[0]['output']):
            raise ValueError('Heldout worker has no matching fixed output slot')
        slot = slots[0]
        if slot['key'] not in {str(i) for i in range(len(claim['worker_slots']))}:
            raise ValueError('Invalid heldout slot key')
        consumed = Path(dataset) / 'heldout-first-use-workers' / (slot['key'] + '.json')
        write_new(consumed, {'schema_version': 'localocr.pretrained.heldout-worker-use.v1',
                            'usage_sha256': expected_usage_sha256, 'claimed_utc': utc(),
                            'slot': slot, 'pid': os.getpid(), 'status': 'consumed-before-reading; failures remain consumed'})


def evaluate(args):
    captured_source = evaluator_source_inventory()
    dataset, output = local_path(args.dataset), local_output(args.output)
    if output.exists():
        raise ValueError('Evaluation output already exists')
    disk_guard(output)
    bindings = [candidate_binding(path) for path in args.candidate]
    if len({b['id'] for b in bindings}) != len(bindings):
        raise ValueError('Candidate ids must be unique')
    baseline = baseline_binding(args.baseline_models)
    frozen = dataset_binding(dataset)
    first_use = None
    if args.split == 'heldout':
        if not args.selection:
            raise ValueError('Heldout requires selection receipt before any heldout access')
        first_use = selection_for_heldout(dataset, local_path(args.selection), bindings, baseline, output_dir=output, dual_freeze=getattr(args,'dual_freeze',None))
    rows = read_split(dataset, args.split, allow_heldout=args.split == 'heldout')
    if not rows:
        raise ValueError('Evaluation needs rows')
    output.mkdir(parents=True, exist_ok=False)
    base = run_recognizer(dataset, args.split, output / 'baseline-predictions.json', baseline=args.baseline_models, first_use=first_use)
    validate_predictions(rows, base['predictions'])
    if base.get('inference_binding') != baseline:
        raise ValueError('Baseline child loaded different bytes')
    validate_cpu_result(base, rows, baseline, frozen, args.split)
    workers = {'production-baseline': base['worker_file']}
    models = {'production-baseline': {'summary': summary(rows, base['predictions'], base['tokens']), 'resources': base['resources']}}
    for index, (path, binding) in enumerate(zip(args.candidate, bindings)):
        disk_guard(output)
        result = run_recognizer(dataset, args.split, output / f'candidate-{index}-predictions.json', candidate=local_path(path), first_use=first_use)
        validate_predictions(rows, result['predictions'])
        if result.get('inference_binding') != binding:
            raise ValueError('Candidate child loaded different bytes')
        validate_cpu_result(result, rows, binding, frozen, args.split)
        workers[binding['id']] = result['worker_file']
        models[binding['id']] = {'summary': summary(rows, result['predictions'], result['tokens']),
                                'transitions_from_production': transitions(rows, base['predictions'], result['predictions']),
                                'resources': result['resources'], 'stage': binding['stage']}
    if frozen != dataset_binding(dataset) or bindings != [candidate_binding(path) for path in args.candidate] or baseline != baseline_binding(args.baseline_models):
        raise ValueError('Frozen research inputs changed during evaluation')
    if captured_source != evaluator_source_inventory():
        raise ValueError('Evaluator source changed during evaluation')
    # Recheck all consumed images, labels and split bytes before recording success.
    for row in read_split(dataset, args.split, allow_heldout=args.split == 'heldout'):
        image_for_row(dataset, row)
    add_comparisons_and_gates(models)
    report = {'schema_version': REPORT_SCHEMA, 'evaluated_utc': utc(), 'split': args.split,
              'dataset_binding': frozen, 'candidate_bindings': bindings, 'baseline_binding': baseline,
              'models': models, 'worker_results': workers, 'source_inventory': captured_source, 'first_use': first_use, 'promote': False, 'enabled': False,
              'independent_heldout': args.split == 'heldout', 'normalization': 'none; exact Unicode, case, whitespace and graphemes',
              'scope': 'same full synthetic canvases; raw recognizer CPU; excludes automatic ROI and APP lifecycle acceptance'}
    write_new(output / 'evaluation.json', report)
    return report


def family_coverage(current, required):
    """Every one of the ``required`` families needs a finetuned and a pre-finetune entry."""
    families = {entry['family'] for entry in current}
    return len(families) == required and all(
        any(e['family'] == family and e['stage'] == 'finetuned' for e in current)
        and any(e['family'] == family and e['stage'] in STAGES[:2] for e in current) for family in families)


def freeze_selection(args):
    report_path, budget_path = local_path(args.dev_report), local_path(args.budget_receipt)
    report, budget = read(report_path), read(budget_path)
    if (report.get('schema_version') != REPORT_SCHEMA or report.get('split') != 'dev'
            or report.get('independent_heldout') is not False or report.get('promote') is not False):
        raise ValueError('Selection requires actual development evaluation')
    if (budget.get('schema_version') != 'localocr.gpu-budget.v1' or budget.get('active') is not None
            or budget.get('maximum_seconds') != 7200 or not finite_number(budget.get('used_seconds'))
            or not 0 <= budget['used_seconds'] <= 7200):
        raise ValueError('Selection needs completed approved cumulative budget')
    if not isinstance(args.reason, str) or not args.reason.strip():
        raise ValueError('Selection needs an explicit development reason')
    current = [candidate_binding(path) for path in args.candidate]
    if current != report['candidate_bindings'] or baseline_binding(args.baseline_models) != report['baseline_binding'] or dataset_binding(args.dataset) != report['dataset_binding']:
        raise ValueError('Development results no longer bind current inputs')
    if not family_coverage(current, rp.specialist_families(args.dataset)):
        raise ValueError('Final selection must bind every required family before and after')
    if args.selected_id not in {entry['id'] for entry in current if entry['stage'] == 'finetuned'}:
        raise ValueError('Primary research selection must name an evaluated finetuned candidate')
    validate_dev_evidence(report, report_path, args.dataset, current, report['baseline_binding'])
    output = local_output(args.output)
    disk_guard(output)
    if output.exists():
        raise ValueError('Selection receipt already exists')
    value = {'schema_version': SELECTION_SCHEMA, 'frozen_utc': utc(), 'dataset_binding': report['dataset_binding'],
             'candidate_bindings': current, 'baseline_binding': report['baseline_binding'],
             'dev_report': {'file': str(report_path), 'sha256': digest(report_path)},
             'budget_receipt': {'file': str(budget_path), 'sha256': digest(budget_path)},
             'selected_id': args.selected_id, 'selection_reason': args.reason,
             'usage': 'research-only-selected-for-evaluation', 'promote': False,
             'all_bound_candidates_evaluated_together': True,
             'quality_gate_snapshot': {entry['id']: report['models'][entry['id']].get('quality_gates') for entry in current}}
    write_new(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', '_worker'):
        command = commands.add_parser(name)
        command.add_argument('--dataset', required=name == 'evaluate')
        command.add_argument('--split', choices=('dev', 'heldout'), default='dev')
        command.add_argument('--output', required=True)
        if name == 'evaluate':
            command.add_argument('--candidate', action='append', default=[])
            command.add_argument('--baseline-models', required=True)
            command.add_argument('--selection')
            command.add_argument('--dual-freeze')
        else:
            choice = command.add_mutually_exclusive_group(required=True)
            choice.add_argument('--candidate')
            choice.add_argument('--baseline-models')
            command.add_argument('--heldout-usage-sha256')
            command.add_argument('--private-manifest')
    private = commands.add_parser('private-evaluate')
    for name in ('private-manifest', 'baseline-models', 'output'):
        private.add_argument('--' + name, required=True)
    private.add_argument('--candidate', action='append', required=True)
    freeze = commands.add_parser('freeze-selection')
    for name in ('dataset', 'dev-report', 'budget-receipt', 'baseline-models', 'selected-id', 'reason', 'output'):
        freeze.add_argument('--' + name, required=True)
    freeze.add_argument('--candidate', action='append', required=True)
    args = parser.parse_args(argv)
    if args.command == '_worker':
        worker(args)
    elif args.command == 'freeze-selection':
        receipt = freeze_selection(args)
        print(json.dumps({'schema_version': receipt['schema_version'], 'selected_id': receipt['selected_id'], 'promote': False}))
    else:
        result = private_evaluate(args) if args.command == 'private-evaluate' else evaluate(args)
        # Path and text identities in private bindings stay out of console output.
        print(json.dumps({'schema_version': result['schema_version'], 'split': result['split'],
                          'models': {key: {'overall': value['summary']['overall'], 'resources': value['resources']}
                                     for key, value in result['models'].items()}, 'promote': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
