"""Offline CPU whole-scene research; truth is consumed only after inference."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
import numpy as np
from PIL import Image
from tools.PretrainedOcr import common
from tools.PretrainedOcr.evaluate import (Recognizer, baseline_binding, candidate_binding,
                                         MemoryMeasurement, percentiles)
from tools.OcrDatasets.evaluate import metrics
from tools.RegionKaomoji import candidates as c, fusion as f

REPO = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO / 'artifacts' / 'region-kaomoji-20261011'
SCHEMA = 'localocr.region-research.v1'
CONFIG_SCHEMA = 'localocr.region-pipeline-config.v1'
SELECTION_SCHEMA = 'localocr.region-selection.v1'
USAGE_SCHEMA = 'localocr.region-heldout-first-use.v1'
JOB_KEYS = {'id', 'image', 'sha256', 'pixel_sha256', 'width', 'height'}
LIMITS = {'maximum_source_bytes': 20 * 1024**2, 'maximum_pixels': 50_000_000,
          'maximum_candidates': 48, 'maximum_crop_pixels': 1_000_000,
          'maximum_scene_output_bytes': 8 * 1024**2, 'scene_timeout_seconds': 90,
          'batch_timeout_seconds': 5400}
VIEWS = ['raw', 'edge-pad-6']


def safe_path(path):
    path = Path(path)
    if str(path).startswith(('\\\\', '//')) or '..' in path.parts:
        raise ValueError('Only local paths without traversal are supported')
    absolute = Path(os.path.abspath(path))
    for parent in reversed((absolute, *absolute.parents)):
        if os.path.lexists(parent):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Research reparse paths are unsupported')
    return absolute


def confined(root, name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Expected a confined portable relative file')
    path = Path(name)
    if path.is_absolute() or any(part in ('.', '..') for part in path.parts):
        raise ValueError('Research file escapes its root')
    root = safe_path(root)
    target = safe_path(root / path)
    if target == root or not target.is_relative_to(root):
        raise ValueError('Research file escapes its root')
    return target


def output_path(path):
    path, root = safe_path(path), safe_path(EVALUATION_ROOT)
    if path == root or not path.is_relative_to(root) or not path.relative_to(root).parts[0].startswith('evaluation'):
        raise ValueError('Output must belong to this research evaluation namespace')
    return path


def write_new(path, value):
    path = output_path(path)
    common.disk_guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_path(path.parent)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def scene_binding(root):
    root = safe_path(root)
    meta, frozen = common.read(root / 'dataset.json'), common.read(root / 'freeze.json')
    if (meta.get('schema_version') != 'localocr.region-scenes.v1' or meta.get('synthetic_only') is not True
            or frozen.get('schema_version') != 'localocr.region-scenes.freeze.v1'
            or common.digest(root / 'dataset.json') != frozen.get('dataset_sha256')
            or set(meta.get('splits', {})) != {'dev', 'heldout'}
            or frozen.get('split_sha256') != {key: value['sha256'] for key, value in meta['splits'].items()}):
        raise ValueError('Scene metadata differs from its frozen contract')
    return frozen


def input_binding(root, path=None):
    root = safe_path(root); path = safe_path(path or root / 'inference-inputs.json')
    raw = path.read_bytes(); value = json.loads(raw.decode('utf-8'))
    frozen = scene_binding(root)
    if (value.get('schema_version') != 'localocr.region-scenes.inputs.v1'
            or value.get('dataset_sha256') != frozen['dataset_sha256']
            or value.get('split_sha256') != frozen['split_sha256']
            or set(value.get('jobs', {})) != {'dev', 'heldout'}):
        raise ValueError('GT-free scene input index differs from freeze')
    return {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'dataset_sha256': frozen['dataset_sha256'], 'split_sha256': frozen['split_sha256']}


def image_jobs(root, split, inputs=None, allow_heldout=False):
    if split not in ('dev', 'heldout') or split == 'heldout' and not allow_heldout:
        raise ValueError('Heldout images require a claimed frozen first use')
    binding = input_binding(root, inputs)
    value = common.read(binding['file'])
    if common.digest(binding['file']) != binding['sha256']:
        raise ValueError('Scene input index changed while reading')
    jobs = value['jobs'][split]
    meta = common.read(safe_path(root) / 'dataset.json')
    if not isinstance(jobs, list) or len(jobs) != meta['splits'][split]['count'] or not jobs:
        raise ValueError('GT-free job count differs')
    identifiers = set()
    for job in jobs:
        if (not isinstance(job, dict) or set(job) != JOB_KEYS or not isinstance(job['id'], str)
                or not job['id'] or job['id'] in identifiers
                or any(type(job[key]) is not int or job[key] < 1 for key in ('width', 'height'))
                or job['width'] * job['height'] > LIMITS['maximum_pixels']):
            raise ValueError('Invalid or truth-bearing inference job')
        identifiers.add(job['id'])
        path = confined(root, job['image'])
        if not path.is_file() or not 0 < path.stat().st_size <= LIMITS['maximum_source_bytes'] or common.digest(path) != job['sha256']:
            raise ValueError('Frozen source image bytes differ')
    return jobs


def image_for_job(root, job):
    path = confined(root, job['image'])
    if not 0 < path.stat().st_size <= LIMITS['maximum_source_bytes'] or common.digest(path) != job['sha256']:
        raise ValueError('Frozen source image changed')
    with Image.open(path) as source:
        if (source.format not in ('PNG', 'JPEG') or getattr(source, 'n_frames', 1) != 1
                or source.size != (job['width'], job['height']) or source.width * source.height > LIMITS['maximum_pixels']):
            raise ValueError('Unsafe original scene image')
        source.load(); image = source.convert('RGB')
    if c.image_digest(image) != job['pixel_sha256']:
        raise ValueError('Frozen original scene pixels changed')
    return image


def configuration(path=None):
    value = common.read(safe_path(path)) if path else {
        'schema_version': CONFIG_SCHEMA, 'candidate': c.configuration(), 'fusion': f.configuration(),
        'views': VIEWS, 'limits': LIMITS, 'join': {'same_line': '  ', 'different_line': '\n', 'vertical_overlap': .5}}
    if not isinstance(value, dict) or set(value) != {'schema_version', 'candidate', 'fusion', 'views', 'limits', 'join'}:
        raise ValueError('Unknown pipeline configuration')
    if value['schema_version'] != CONFIG_SCHEMA or value['views'] != VIEWS or value['limits'] != LIMITS:
        raise ValueError('Fixed inference budgets or independent views cannot change')
    if value['join'] != {'same_line': '  ', 'different_line': '\n', 'vertical_overlap': .5}:
        raise ValueError('Unsupported declared output reading order')
    value = json.loads(json.dumps(value, allow_nan=False))
    return {**value, 'candidate': c.configuration(value['candidate']), 'fusion': f.configuration(value['fusion'])}


def implementation_binding():
    names = ['scripts/local-ai-ocr-worker.py', 'localocr/__init__.py', 'localocr/bundles.py', 'localocr/version.py',
             'tools/RegionKaomoji/__init__.py', 'tools/RegionKaomoji/evaluate.py',
             'tools/RegionKaomoji/candidates.py', 'tools/RegionKaomoji/fusion.py',
             'tools/PretrainedOcr/__init__.py', 'tools/PretrainedOcr/common.py', 'tools/PretrainedOcr/evaluate.py',
             'tools/OcrDatasets/__init__.py', 'tools/OcrDatasets/evaluate.py',
             'tools/OcrDatasets/validate.py', 'tools/OcrDatasets/private.py']
    return {name: common.digest(safe_path(REPO / name)) for name in names}


def pipeline_binding(root, models, baseline, config, inputs=None):
    bindings = [candidate_binding(safe_path(path)) for path in models]
    if len({item['id'] for item in bindings}) != len(bindings) or len(bindings) > 2:
        raise ValueError('At most two unique research specialists are permitted')
    for item in bindings:
        for reference in item['source_files'].values():
            safe_path(reference['file'])
    baseline = safe_path(baseline)
    return {'schema_version': 'localocr.region-pipeline-binding.v1', 'dataset': scene_binding(root),
            'inputs': input_binding(root, inputs), 'baseline': baseline_binding(baseline),
            'models': bindings, 'config': config, 'config_sha256': hash_object(config),
            'implementation': implementation_binding(), 'promote': False, 'enabled': False}


def load_worker():
    path = REPO / 'scripts/local-ai-ocr-worker.py'
    spec = importlib.util.spec_from_file_location('localocr_region_cpu_worker', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def baseline_runtime_receipt(recognizer):
    import cv2
    tokens = list(recognizer.postprocess_op.character)
    ignored = list(recognizer.postprocess_op.get_ignored_tokens())
    if ignored != [0] or not tokens or any(not isinstance(token,str) for token in tokens):
        raise ValueError('Actual baseline decoder dictionary differs')
    spaces = [index for index,token in enumerate(tokens) if token == ' ']
    if not spaces: raise ValueError('Actual baseline dictionary has no space token')
    sources = {}
    for name, obj in [('recognizer',recognizer), ('decoder',recognizer.postprocess_op)]:
        path = safe_path(inspect.getsourcefile(type(obj)))
        sources[name] = {'file':str(path),'sha256':common.digest(path)}
    version = importlib.metadata.version('rapidocr')
    if version != '3.10.0': raise ValueError('Unfrozen baseline preprocessing runtime')
    vocabulary = {'schema_version':'localocr.baseline-decoder-tokens.v1','tokens':tokens,
                  'blank_id':0,'space_ids':spaces}
    preprocessing = {'schema_version':'localocr.baseline-rec-preprocessing.v1',
         'rec_image_shape':list(map(int,recognizer.rec_image_shape)), 'rec_batch_num':int(recognizer.rec_batch_num),
         'color':'BGR','resize':'cv2.INTER_LINEAR',
         'width':'int(height * max(configured_width / height, maximum input aspect ratio in the recognition batch))',
         'content_width':'min(batch_width, ceil(height * input_width / input_height))',
         'normalize':{'scale':255,'mean':[.5]*3,'std':[.5]*3},
         'padding':'right-zero-normalized','layout':'NCHW-float32', 'return_word_box':False,
         'rapidocr_version':version,'opencv_version':cv2.__version__,
         'implementation':{name:record['sha256'] for name,record in sources.items()}}
    algorithm='sha256(canonical JSON: sorted keys, UTF-8, ensure_ascii=false, compact separators)'
    return {'schema_version':'localocr.baseline-recognizer.v1','rapidocr_version':version,'source_files':sources,
            'vocabulary':{'sha256':hash_object(vocabulary),'token_count':len(tokens),'blank_id':0,
                          'blank_token':tokens[0],'space_ids':spaces,'hash_algorithm':algorithm},
            'preprocessing':{'sha256':hash_object(preprocessing),'configuration':preprocessing,'hash_algorithm':algorithm}}


def verify_baseline_runtime(receipt):
    if (receipt.get('schema_version') != 'localocr.baseline-recognizer.v1'
            or receipt.get('rapidocr_version') != '3.10.0'
            or receipt.get('vocabulary',{}).get('blank_id') != 0
            or type(receipt.get('vocabulary',{}).get('token_count')) is not int
            or receipt['vocabulary']['token_count'] < 2
            or len(receipt['vocabulary'].get('sha256','')) != 64
            or receipt.get('preprocessing',{}).get('sha256') != hash_object(receipt['preprocessing']['configuration'])
            or set(receipt.get('source_files',{})) != {'recognizer','decoder'}):
        raise ValueError('Actual baseline runtime receipt differs')
    for record in receipt['source_files'].values():
        if common.digest(safe_path(record['file'])) != record['sha256']:
            raise ValueError('Actual baseline preprocessing implementation changed')


class FullBaseline:
    def __init__(self, root):
        self.binding = baseline_binding(root)
        self.worker = load_worker()
        self.engine = self.worker.create_engine(safe_path(root), safe_path(root) / 'manifest.json')
        self.runtime_receipt = baseline_runtime_receipt(self.engine._load_rec_model())

    def predict(self, image):
        pixels = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
        initial = self.engine(pixels)
        self.worker.serialize_json_result(initial, image.width, image.height, self.binding['version'])
        result = self.worker.reread_symbols(self.engine, pixels, initial)
        raw = json.loads(self.worker.serialize_json_result(result, image.width, image.height, self.binding['version']))
        canonical = []
        for region in raw['regions']:
            rect = region['rect']; canonical.append({'bbox': [rect['x'], rect['y'], rect['x'] + rect['width'],
                                                           rect['y'] + rect['height']],
                                                   'text': region['text'], 'score': region['score']})
        return canonical, raw

    def predict_context(self, image):
        from rapidocr.ch_ppocr_rec import TextRecInput
        pixels = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
        result = self.engine._load_rec_model()(TextRecInput(img=pixels, return_word_box=False))
        if (len(result.txts) != 1 or len(result.scores) != 1 or not isinstance(result.txts[0], str)
                or not math.isfinite(float(result.scores[0]))):
            raise ValueError('Actual baseline context recognizer output differs')
        return result.txts[0], float(result.scores[0])


def joined_regions(regions, config):
    lines = []
    for region in sorted(regions, key=lambda x: (x['bbox'][1], x['bbox'][0])):
        left, top, right, bottom = region['bbox']
        for line in lines:
            overlap = min(bottom, line['bottom']) - max(top, line['top'])
            if overlap >= min(bottom - top, line['bottom'] - line['top']) * config['join']['vertical_overlap']:
                line['items'].append(region); line['top'] = min(top, line['top']); line['bottom'] = max(bottom, line['bottom'])
                break
        else:
            lines.append({'top': top, 'bottom': bottom, 'items': [region]})
    return config['join']['different_line'].join(config['join']['same_line'].join(item['text'] for item in sorted(line['items'], key=lambda x: x['bbox'][0]))
                                                 for line in sorted(lines, key=lambda x: x['top']))


def predict_scene(image, baseline, models, config):
    """Inference boundary: receives pixels and model objects, never scene truth."""
    timings = {}; started = time.perf_counter()
    entries, native = baseline.predict(image); timings['baseline_seconds'] = time.perf_counter() - started
    started = time.perf_counter(); routing = c.locate_candidates(image, entries, config['candidate'])
    timings['candidate_seconds'] = time.perf_counter() - started
    views = {}; contexts = {}; selected = []; skipped = []; pixels = 0; context_times = []
    for candidate in routing['candidates']:
        if not candidate.get('route_eligible', False):
            skipped.append({'candidate_id': candidate['candidate_id'], 'reason': candidate.get('route_reason', 'ineligible')})
            continue
        samples = [f.view_image(image, candidate['bbox'], transform) for transform in config['views']]
        declarations = candidate.get('context_crops', [])
        if (not isinstance(declarations, list) or len(declarations) > 2
                or len({item['role'] for item in declarations}) != len(declarations)):
            raise ValueError('Automatic context collection exceeds the two-side budget')
        context_arrays = [(item, f.view_image(image, item['bbox'], 'raw')) for item in declarations]
        cost = (sum(array.shape[0] * array.shape[1] for array in samples) * max(1, len(models))
                + sum(array.shape[0] * array.shape[1] for _, array in context_arrays))
        if len(selected) >= config['limits']['maximum_candidates'] or pixels + cost > config['limits']['maximum_crop_pixels']:
            skipped.append({'candidate_id': candidate['candidate_id'], 'reason': 'independent-view-pixel-budget'})
            continue
        pixels += cost; selected.append(candidate); views[candidate['candidate_id']] = samples
        if models and context_arrays:
            records = []
            for declaration, array in context_arrays:
                at = time.perf_counter(); text, score = baseline.predict_context(Image.fromarray(array))
                context_times.append(time.perf_counter() - at)
                records.append({'role': declaration['role'], 'bbox': declaration['bbox'],
                                'image_sha256': c.image_digest(array), 'text': text,
                                'baseline_parent_sha256': candidate['baseline_parent_sha256'],
                                'score': float(score), 'model_id': 'production-baseline'})
            contexts[candidate['candidate_id']] = records
    timings['context_seconds'] = sum(context_times)
    timings['context_latency'] = percentiles(context_times)
    results = {}
    for model in models:
        identity = model.binding['id']; records = {}; infer_times = []
        for candidate in selected:
            records[candidate['candidate_id']] = []
            for transform, array in zip(config['views'], views[candidate['candidate_id']]):
                at = time.perf_counter(); text, score = model.predict(Image.fromarray(array)); infer_times.append(time.perf_counter() - at)
                records[candidate['candidate_id']].append({'model_id': identity, 'view_id': identity + ':' + transform,
                    'transform': transform, 'image_sha256': c.image_digest(array), 'text': text, 'score': float(score)})
        at = time.perf_counter()
        fused = f.fuse_candidates(image, entries, routing['candidates'], records, config['fusion'], config['candidate'], contexts=contexts)
        results[identity] = {'regions': fused['regions'], 'decisions': fused['decisions'], 'views': records,
                             'text': joined_regions(fused['regions'], config),
                             'specialist_seconds': sum(infer_times), 'fusion_seconds': time.perf_counter() - at,
                             'view_latency': percentiles(infer_times), 'accepted_count': fused['accepted_count']}
    result = {'baseline': entries, 'baseline_native': native, 'baseline_text': joined_regions(entries, config),
              'routing': routing, 'models': results, 'contexts': contexts,
              'view_budget': {'pixels': pixels, 'selected': len(selected), 'context_crops': len(context_times),
              'skipped': skipped}, 'timings': timings}
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')) > config['limits']['maximum_scene_output_bytes']:
        raise ValueError('Scene research output exceeds byte limit')
    return result


def _timestamp(value):
    date = datetime.fromisoformat(value)
    if date.tzinfo is None or date > datetime.now(timezone.utc):
        raise ValueError('Frozen selection needs an actual UTC timestamp')
    return date


def usage_path(binding):
    return output_path(EVALUATION_ROOT / ('evaluation-heldout-first-use-' + binding['dataset']['split_sha256']['heldout'] + '.json'))


def validate_budget(budget):
    if (budget.get('schema_version') != 'localocr.gpu-budget.v1' or budget.get('active') is not None
            or budget.get('maximum_seconds') != 7200 or type(budget.get('used_seconds')) not in (int, float)
            or not math.isfinite(budget['used_seconds']) or not 0 <= budget['used_seconds'] <= 7200):
        raise ValueError('Selection needs the completed shared 7200-second budget receipt')


def final_families(binding):
    models = binding['models']
    if (len(models) != 2 or len({item['family'] for item in models}) != 2
            or any(item['stage'] != 'finetuned' for item in models)):
        raise ValueError('Freeze requires the same two finetuned specialist families')


def verify_dev_evidence(path, binding):
    path = output_path(path); dev = common.read(path)
    if (dev.get('schema_version') != SCHEMA or dev.get('split') != 'dev'
            or dev.get('pipeline_binding') != binding or dev.get('actual_cpu_inference') is not True
            or dev.get('independent_heldout') is not False or dev.get('promote') is not False
            or dev.get('enabled') is not False
            or dev.get('summary', {}).get('scenes') != binding['dataset']['counts']['dev']['scenes']
            or set(dev.get('summary', {}).get('models', {})) != {item['id'] for item in binding['models']}):
        raise ValueError('Selection needs the unchanged actual whole-scene development report')
    _timestamp(dev['completed_utc'])
    outputs = dev.get('outputs', {})
    required = {'predictions': 'predictions.jsonl', 'resources': 'resources.json',
                'alignment': 'alignment-private.json'}
    if set(outputs) != set(required):
        raise ValueError('Development predictions, resources and alignment must be bound')
    for key, name in required.items():
        record = outputs[key]; target = output_path(record['file'])
        if target != path.parent / name or common.digest(target) != record['sha256']:
            raise ValueError('Frozen development output bytes changed')
    resources = common.read(outputs['resources']['file'])
    actual = resources.get('resources', {})
    if (resources.get('pipeline_binding') != binding or actual != dev.get('resources')
            or actual.get('providers') != ['CPUExecutionProvider'] or actual.get('intra_op_threads') != 2
            or actual.get('inter_op_threads') != 1 or actual.get('actual_baseline_detection') is not True
            or actual.get('actual_specialist_models') != len(binding['models'])
            or actual.get('no_python_children_or_socket_io') is not True):
        raise ValueError('Development report lacks bound actual isolated CPU resources')
    verify_baseline_runtime(actual['baseline_runtime'])
    return dev


def verify_selection(path, binding):
    final_families(binding)
    path = safe_path(path); raw = path.read_bytes(); value = json.loads(raw.decode('utf-8'))
    if (value.get('schema_version') != SELECTION_SCHEMA or value.get('pipeline_binding') != binding
            or value.get('promote') is not False or value.get('enabled') is not False
            or value.get('selected_id') not in {item['id'] for item in binding['models']}):
        raise ValueError('Frozen selection differs from current whole-scene pipeline')
    frozen_time = _timestamp(value['frozen_utc'])
    if not isinstance(value.get('selection_reason'), str) or not value['selection_reason'].strip():
        raise ValueError('Frozen selection needs an explicit development reason')
    for key in ('dev_report', 'budget_receipt'):
        record = value[key]
        if common.digest(safe_path(record['file'])) != record['sha256']:
            raise ValueError('Frozen selection evidence changed')
    dev = verify_dev_evidence(value['dev_report']['file'], binding)
    if value.get('baseline_runtime') != dev['resources']['baseline_runtime']:
        raise ValueError('Frozen baseline runtime differs from actual development inference')
    if value.get('quality_gate_snapshot') != {key:item['quality_gates'] for key,item in dev['summary']['models'].items()}:
        raise ValueError('Frozen quality gates differ from actual development inference')
    if _timestamp(dev['completed_utc']) > frozen_time:
        raise ValueError('Development evaluation must complete before frozen selection')
    validate_budget(common.read(value['budget_receipt']['file']))
    return {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'value': value}


def claim_heldout(root, selection, binding, output):
    evidence = verify_selection(selection, binding)
    seal_path = safe_path(root) / 'heldout-seal.json'; seal = common.read(seal_path)
    if (seal.get('schema_version') != 'localocr.region-scenes.seal.v1'
            or seal.get('inference_results_seen') is not False or seal.get('requires_frozen_pipeline_selection') is not True
            or seal.get('dataset_sha256') != binding['dataset']['dataset_sha256']
            or seal.get('heldout_sha256') != binding['dataset']['split_sha256']['heldout']):
        raise ValueError('Heldout seal changed or already has results')
    value = {'schema_version': USAGE_SCHEMA, 'claimed_utc': common.utc(), 'pid': os.getpid(),
             'selection': {key: evidence[key] for key in ('file', 'sha256')},
             'pipeline_binding': binding, 'seal_sha256': common.digest(seal_path),
             'output': str(output), 'worker_output': str(output / '.worker/predictions.jsonl'),
             'worker_jobs': str(output / '.worker/jobs.json'),
             'status': 'claimed-before-reading-heldout-images-or-truth; failure remains consumed'}
    path = usage_path(binding); write_new(path, value)
    return {'file': str(path), 'sha256': common.digest(path)}


def owned_parent(owner):
    if os.getppid() == owner: return True
    if sys.platform != 'win32': return False
    import psutil
    try:
        parent = psutil.Process(os.getppid()); command = parent.cmdline()
        return (parent.ppid() == owner and safe_path(parent.exe()) == safe_path(sys.executable)
                and command[1:4] == ['-m', 'tools.RegionKaomoji.evaluate', '_worker'])
    except (psutil.Error, OSError):
        return False


def consume_heldout(root, usage, binding, jobs, output):
    path = usage_path(binding)
    if not usage or common.digest(path) != usage:
        raise ValueError('Heldout worker requires its first-use digest')
    value = common.read(path)
    if (value.get('schema_version') != USAGE_SCHEMA or value.get('pipeline_binding') != binding
            or not owned_parent(value['pid']) or safe_path(value['worker_output']) != safe_path(output)
            or safe_path(value['worker_jobs']) != safe_path(jobs)):
        raise ValueError('Heldout worker owner, slot or pipeline differs')
    verify_selection(value['selection']['file'], binding)
    if common.digest(value['selection']['file']) != value['selection']['sha256']:
        raise ValueError('Heldout worker selection bytes changed')
    seal_path = safe_path(root) / 'heldout-seal.json'
    seal = common.read(seal_path)
    if common.digest(seal_path) != value['seal_sha256'] or seal.get('inference_results_seen') is not False:
        raise ValueError('Heldout worker seal changed')
    write_new(path.with_name(path.stem + '-worker.json'), {'schema_version': 'localocr.region-heldout-worker-use.v1',
              'usage_sha256': usage, 'pid': os.getpid(), 'claimed_utc': common.utc(),
              'status': 'consumed-before-reading-images; failed workers cannot replay'})


def worker(args):
    root, output, jobs_path = safe_path(args.dataset), output_path(args.output), safe_path(args.jobs)
    if output.exists(): raise ValueError('CPU worker output already exists')
    config = configuration(args.config)
    binding = pipeline_binding(root, args.candidate, args.baseline_models, config, args.inputs)
    if common.digest(safe_path(args.binding)) != args.binding_sha256 or common.read(args.binding) != binding:
        raise ValueError('CPU worker differs from captured pipeline binding')
    if args.split == 'heldout': consume_heldout(root, args.usage_sha256, binding, jobs_path, output)
    if common.digest(jobs_path) != args.jobs_sha256:
        raise ValueError('Truth-free job manifest checksum differs')
    envelope = common.read(jobs_path)
    if (set(envelope) != {'schema_version', 'split', 'pipeline_sha256', 'jobs'}
            or envelope['schema_version'] != 'localocr.region-inference-jobs.v1' or envelope['split'] != args.split
            or envelope['pipeline_sha256'] != hash_object(binding)):
        raise ValueError('Invalid truth-free worker job envelope')
    jobs = image_jobs(root, args.split, args.inputs, allow_heldout=args.split == 'heldout')
    if envelope['jobs'] != jobs:
        raise ValueError('Worker jobs differ from GT-free frozen input index')
    # This policy belongs exclusively to this already-spawned CPU worker.
    sys.addaudithook(load_worker().deny_network)
    common.disk_guard(output); output.parent.mkdir(parents=True, exist_ok=True)
    memory = MemoryMeasurement(); at = time.perf_counter()
    def expire():
        os._exit(124)
    timer = threading.Timer(config['limits']['scene_timeout_seconds'], expire); timer.daemon = True; timer.start()
    timings = []
    try:
        baseline = FullBaseline(safe_path(args.baseline_models)); models = [Recognizer(safe_path(path)) for path in args.candidate]
        verify_baseline_runtime(baseline.runtime_receipt)
        if args.split == 'heldout':
            first_use = common.read(usage_path(binding))
            frozen = common.read(first_use['selection']['file'])
            if baseline.runtime_receipt != frozen['baseline_runtime']:
                raise ValueError('Heldout actual baseline runtime differs from frozen development')
        load_seconds = time.perf_counter() - at
        first = image_for_job(root, jobs[0]); at = time.perf_counter(); predict_scene(first, baseline, models, config)
        cold = time.perf_counter() - at; timer.cancel(); timer.join()
        for _ in range(2):
            timer = threading.Timer(config['limits']['scene_timeout_seconds'], expire); timer.daemon=True; timer.start()
            predict_scene(first, baseline, models, config); timer.cancel(); timer.join()
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            for job in jobs:
                timer = threading.Timer(config['limits']['scene_timeout_seconds'], expire); timer.daemon=True; timer.start()
                image = image_for_job(root, job); at = time.perf_counter(); prediction = predict_scene(image, baseline, models, config)
                elapsed = time.perf_counter() - at; timings.append(elapsed)
                common.disk_guard(output)
                stream.write(json.dumps({'id':job['id'],'image_sha256':job['sha256'],'pixel_sha256':job['pixel_sha256'],
                                         'prediction':prediction,'seconds':elapsed}, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush(); timer.cancel(); timer.join()
            os.fsync(stream.fileno())
        if (pipeline_binding(root, args.candidate, args.baseline_models, config, args.inputs) != binding
                or baseline_runtime_receipt(baseline.engine._load_rec_model()) != baseline.runtime_receipt):
            raise ValueError('Bound pipeline or actual baseline runtime changed during inference')
        resources = {**memory.finish(), 'load_seconds':load_seconds,'first_scene_seconds':cold,
                     'warm':percentiles(timings),'total_warm_seconds':sum(timings),'warmup_scenes':3,
                     'providers':['CPUExecutionProvider'],'intra_op_threads':2,'inter_op_threads':1,
                     'scene_timeout_seconds':90,'actual_baseline_detection':True,
                     'actual_specialist_models':len(models),'no_python_children_or_socket_io':True,
                     'onnxruntime_version':__import__('onnxruntime').__version__,
                     'baseline_runtime':baseline.runtime_receipt}
        write_new(output.with_suffix('.resources.json'), {'pipeline_binding':binding,'resources':resources})
    finally:
        timer.cancel(); timer.join()
        if memory.thread.is_alive(): memory.finish()


def stop_process(process):
    if process.poll() is not None: return
    import psutil
    try:
        descendants=psutil.Process(process.pid).children(recursive=True)
    except psutil.Error: descendants=[]
    for child in descendants:
        try: child.kill()
        except psutil.Error: pass
    process.kill(); process.wait(timeout=10)


def cleanup_temporary(root):
    # Remove only this runner's fixed temporary names, never recursively delete
    # an unknown entry or follow a reparse point inserted by another actor.
    root=safe_path(root)
    for name in ('jobs.json','config.json','binding.json','predictions.jsonl','predictions.resources.json','stdout.txt','stderr.txt'):
        path=safe_path(root/name)
        if path.exists():
            if not stat.S_ISREG(path.lstat().st_mode): raise ValueError('Unexpected temporary file type')
            path.unlink()
    try: root.rmdir()
    except OSError: pass


def diagnostic_tail(path):
    if not path.exists(): return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-16000))
        return stream.read(16000).decode('utf-8',errors='replace')


def copy_new(source,destination):
    destination=output_path(destination)
    common.disk_guard(destination)
    with safe_path(source).open('rb') as src, destination.open('xb') as target:
        shutil.copyfileobj(src,target,1024*1024)
        target.flush(); os.fsync(target.fileno())


def read_predictions(path, jobs):
    predictions = []; maximum = LIMITS['maximum_scene_output_bytes'] + 16384
    with safe_path(path).open('rb') as stream:
        while True:
            raw = stream.readline(maximum + 1)
            if not raw: break
            if len(raw) > maximum or not raw.endswith(b'\n') or len(predictions) >= len(jobs):
                raise ValueError('Worker scene payload is unbounded or incomplete')
            predictions.append(json.loads(raw.decode('utf-8')))
    if [item['id'] for item in predictions] != [job['id'] for job in jobs]:
        raise ValueError('CPU predictions lost the frozen image ordering')
    return predictions


def run_worker(args, output, binding, jobs, usage=None):
    temporary = output / '.worker'
    paths = {key: temporary / name for key, name in [('jobs','jobs.json'),('config','config.json'),
             ('binding','binding.json'),('predictions','predictions.jsonl')]}
    process = None; started=time.perf_counter(); temporary_created=False
    try:
        temporary.mkdir(); temporary_created=True
        write_new(paths['jobs'], {'schema_version':'localocr.region-inference-jobs.v1','split':args.split,
                                 'pipeline_sha256':hash_object(binding),'jobs':jobs})
        write_new(paths['config'], binding['config']); write_new(paths['binding'], binding)
        command = [sys.executable,'-m','tools.RegionKaomoji.evaluate','_worker','--dataset',str(safe_path(args.dataset)),
                   '--inputs',binding['inputs']['file'],'--split',args.split,'--baseline-models',str(safe_path(args.baseline_models)),
                   '--output',str(paths['predictions']),'--jobs',str(paths['jobs']),'--jobs-sha256',common.digest(paths['jobs']),
                   '--config',str(paths['config']),'--binding',str(paths['binding']),'--binding-sha256',common.digest(paths['binding'])]
        for path in args.candidate: command += ['--candidate',str(safe_path(path))]
        if usage: command += ['--usage-sha256',usage['sha256']]
        environment = dict(os.environ); environment.pop('PYTHONPATH',None); environment['PYTHONUTF8']='1'
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): environment[name]='2'
        with (temporary / 'stdout.txt').open('x',encoding='utf-8') as stdout, (temporary / 'stderr.txt').open('x',encoding='utf-8') as stderr:
            process=subprocess.Popen(command,cwd=REPO,env=environment,stdout=stdout,stderr=stderr)
            try: code=process.wait(timeout=LIMITS['batch_timeout_seconds'])
            except subprocess.TimeoutExpired:
                stop_process(process); raise TimeoutError('CPU scene batch exceeded timeout')
        if code:
            raise RuntimeError('Actual CPU whole-scene worker failed with exit ' + str(code))
        predictions=read_predictions(paths['predictions'],jobs)
        resources=common.read(paths['predictions'].with_suffix('.resources.json'))
        if resources['pipeline_binding']!=binding: raise ValueError('Actual CPU binding differs')
        resources['resources']['process_wall_seconds']=time.perf_counter()-started
        copy_new(paths['predictions'],output/'predictions.jsonl')
        write_new(output/'resources.json',resources)
        return predictions,resources['resources']
    except BaseException as error:
        if process is not None: stop_process(process)
        write_new(output/'failure.json',{'schema_version':SCHEMA,'status':'failed; no acceptance claim',
                  'error_type':type(error).__name__,'diagnostic':diagnostic_tail(temporary/'stderr.txt') if temporary_created else '',
                  'elapsed_seconds':time.perf_counter()-started,'heldout_remains_consumed':args.split=='heldout'})
        if temporary_created and paths['predictions'].exists(): copy_new(paths['predictions'],output/'partial-predictions.jsonl')
        raise
    finally:
        if process is not None and process.poll() is None: stop_process(process)
        if temporary_created: cleanup_temporary(temporary)


def load_truth(root, split, jobs, allow_heldout=False):
    if split=='heldout' and not allow_heldout: raise ValueError('Heldout truth requires the first-use claim')
    frozen=scene_binding(root); meta=common.read(safe_path(root)/'dataset.json')
    path=confined(root,meta['splits'][split]['file'])
    if common.digest(path)!=frozen['split_sha256'][split]: raise ValueError('Frozen truth split changed')
    rows=common.read(path)
    if [{key:row[key] for key in JOB_KEYS} for row in rows]!=jobs: raise ValueError('Truth and actual image jobs differ')
    for row in rows:
        if (row['split']!=split or row['source_split']!=split or row['provenance']!={'source':'original','license':'CC0-1.0'}
                or hashlib.sha256(row['text'].encode('utf-8')).hexdigest()!=row['text_sha256']):
            raise ValueError('Scene truth identity/provenance differs')
        for region in row['regions']:
            if region['kind'] not in ('body','kaomoji','graphic'): raise ValueError('Unknown scene truth region')
            if region['kind']!='graphic':
                a,b=region['span']
                if not 0<=a<b<=len(row['text']) or row['text'][a:b]!=region['text']: raise ValueError('Exact truth span changed')
    return rows


def masks_for_row(root,row):
    masks=[]; union=np.zeros((row['height'],row['width']),dtype=bool)
    for region in row['regions']:
        path=confined(root,region['mask'])
        if common.digest(path)!=region['mask_sha256']: raise ValueError('Frozen visible-ink mask differs')
        with Image.open(path) as source:
            if source.mode!='L' or source.size!=(row['width'],row['height']): raise ValueError('Mask mode/size differs')
            source.load(); mask=np.asarray(source)>0
        ys,xs=np.where(mask)
        if (not len(xs) or [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]!=region['bbox']
                or int(mask.sum())!=region['pixel_count'] or np.any(union & mask)):
            raise ValueError('Mask pixels/bounds/independence differ')
        union|=mask; masks.append(mask)
    return masks


def mask_pixels(mask,bbox):
    left,top,right,bottom=bbox
    if not 0<=left<right<=mask.shape[1] or not 0<=top<bottom<=mask.shape[0]: raise ValueError('Measured box outside source')
    return int(mask[top:bottom,left:right].sum())


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/area if area else 0.


def aligned_region(regions,target_index,masks,config):
    selected=[]; mixed=0
    for region in regions:
        if region.get('source') == 'baseline-with-regional-splice':
            actual = {'bbox': region['regional_bbox'], 'text': region['regional_text'], 'score': region['score']}
            if mask_pixels(masks[target_index], actual['bbox']):
                if any(mask_pixels(mask, actual['bbox']) for i,mask in enumerate(masks) if i != target_index): mixed += 1
                else: selected.append(actual)
            elif mask_pixels(masks[target_index], region['bbox']):
                mixed += 1  # A preserved broad body transcript has no per-word model position.
        elif mask_pixels(masks[target_index],region['bbox']):
            if any(mask_pixels(mask,region['bbox']) for i,mask in enumerate(masks) if i!=target_index): mixed+=1
            else: selected.append(region)
    return joined_regions(selected,config), mixed


def body_preserving_splices(before, after, body_mask):
    # Check raw original bytes and real changed pixels without reading truth text.
    expected = {index for index, item in enumerate(before) if mask_pixels(body_mask, item['bbox'])}
    consumed = set(); splices = 0
    for item in after:
        if not mask_pixels(body_mask, item['bbox']): continue
        index = item.get('baseline_index')
        if type(index) is not int or index not in expected or index in consumed: return False, 0
        original = before[index]
        if item['bbox'] != original['bbox'] or item['score'] != original['score']: return False, 0
        if item.get('source') == 'baseline':
            if item['text'] != original['text']: return False, 0
        elif item.get('source') == 'baseline-with-regional-splice':
            proof = item.get('splice', {}); span = proof.get('original_text_span', [])
            if (len(span) != 2 or any(type(v) is not int for v in span)
                    or not 0 <= span[0] <= span[1] <= len(original['text'])
                    or proof.get('baseline_index') != index or proof.get('ordinary_prefix_suffix_preserved') is not True
                    or proof.get('preserved_prefix') != original['text'][:span[0]]
                    or proof.get('preserved_suffix') != original['text'][span[1]:]
                    or proof.get('replacement_text') != item.get('regional_text')
                    or item['text'] != original['text'][:span[0]] + item['regional_text'] + original['text'][span[1]:]
                    or mask_pixels(body_mask, item['regional_bbox'])):
                return False, 0
            splices += 1
        else: return False, 0
        consumed.add(index)
    return consumed == expected and splices > 0, splices


def summarize(root,rows,predictions,config,model_ids):
    if len(rows) != len(predictions): raise ValueError('Scene prediction count differs')
    localization=Counter(); candidates_by_source=Counter(); baseline_pairs=[]; pairs={key:[] for key in model_ids}
    body={key:Counter() for key in model_ids}; faces={key:[] for key in model_ids}; base_faces=[]
    negatives={key:Counter() for key in model_ids}; crop_pairs={key:[] for key in model_ids}
    kind_counts=Counter(); details=[]; ious=[]; coverage_values=[]
    for row,record in zip(rows,predictions):
        if record['id']!=row['id'] or record['image_sha256']!=row['sha256'] or record['pixel_sha256']!=row['pixel_sha256']:
            raise ValueError('Measured scene is not the actual inferred image')
        prediction=record['prediction']; masks=masks_for_row(root,row); kinds=[r['kind'] for r in row['regions']]
        routing=prediction['routing']['candidates']; kind_counts[row['scene_kind']]+=1
        baseline_pairs.append({'label':row['text'],'prediction':prediction['baseline_text']})
        for identity in model_ids: pairs[identity].append({'label':row['text'],'prediction':prediction['models'][identity]['text']})
        body_union=np.zeros((row['height'],row['width']),dtype=bool); graphic_union=body_union.copy()
        for region,mask in zip(row['regions'],masks):
            if region['kind']=='body': body_union|=mask
            elif region['kind']=='graphic': graphic_union|=mask
        routed_ids={key for identity in model_ids for key,views in prediction['models'][identity]['views'].items() if views}
        routed=[item for item in routing if item['candidate_id'] in routed_ids]
        eligible=[item for item in routing if item.get('route_eligible',False)]
        scene_detail={'id':row['id'],'regions':[]}
        for candidate in routing:
            candidates_by_source.update(candidate['sources'])
            swallowed=mask_pixels(body_union,candidate['bbox']); graphic=mask_pixels(graphic_union,candidate['bbox'])
            localization['candidate_count']+=1
            localization['candidates_intersecting_body']+=int(swallowed>0)
            localization['candidates_intersecting_graphic']+=int(graphic>0)
            if candidate['candidate_id'] in routed_ids:
                localization['routed_candidate_count']+=1
                localization['routed_candidates_intersecting_body']+=int(swallowed>0)
                localization['routed_candidates_intersecting_graphic']+=int(graphic>0)
        localization['eligible_candidate_count']+=len(eligible)
        for index,(truth,mask) in enumerate(zip(row['regions'],masks)):
            baseline_text,mixed=aligned_region(prediction['baseline'],index,masks,config)
            if truth['kind']=='kaomoji':
                localization['faces']+=1; base_faces.append({'label':truth['text'],'prediction':baseline_text})
                hit=[item for item in routing if mask_pixels(mask,item['bbox'])]
                complete_candidates=[item for item in hit if mask_pixels(mask,item['bbox'])==truth['pixel_count']]
                safe_candidates=[item for item in complete_candidates if not mask_pixels(body_union,item['bbox']) and not mask_pixels(graphic_union,item['bbox'])]
                best=max(safe_candidates or hit,key=lambda item:(mask_pixels(mask,item['bbox']),iou(truth['bbox'],item['bbox'])),default=None)
                covered=mask_pixels(mask,best['bbox']) if best else 0
                complete=covered==truth['pixel_count']; swallowing=mask_pixels(body_union,best['bbox']) if best else 0
                contamination=mask_pixels(graphic_union,best['bbox']) if best else 0
                localization['found_any']+=int(bool(hit));localization['missed']+=int(not hit)
                localization['complete_ink_coverage']+=int(bool(complete_candidates))
                localization['safe_complete_coverage']+=int(bool(safe_candidates))
                localization['partial_only']+=int(bool(hit) and not complete_candidates)
                localization['eligible_complete_ink_coverage']+=int(any(item['candidate_id'] in {x['candidate_id'] for x in eligible} for item in complete_candidates))
                localization['routed_complete_ink_coverage']+=int(any(item['candidate_id'] in routed_ids for item in complete_candidates))
                localization['safe_routed_complete_coverage']+=int(any(item['candidate_id'] in routed_ids for item in safe_candidates))
                localization['baseline_no_isolated_face_output']+=int(not baseline_text)
                baseline_cover=max((mask_pixels(mask,item['bbox']) for item in prediction['baseline']),default=0)
                localization['baseline_detection_missed_faces']+=int(not baseline_cover)
                if not baseline_cover:
                    localization['image_branch_recovers_baseline_detection_miss']+=int(any(any(source.startswith('image-') for source in item['sources']) for item in hit))
                coverage_values.append(covered/truth['pixel_count']); ious.append(iou(truth['bbox'],best['bbox']) if best else 0.)
                detail={'truth_id':truth['id'],'kind':truth['kind'],'bbox':truth['bbox'],'matched_candidate':best['candidate_id'] if best else None,
                        'covered_pixels':covered,'ink_pixels':truth['pixel_count'],'body_pixels_in_crop':swallowing,
                        'graphic_pixels_in_crop':contamination,'baseline_mixed_regions':mixed}
                for identity in model_ids:
                    final_text,final_mixed=aligned_region(prediction['models'][identity]['regions'],index,masks,config)
                    faces[identity].append({'label':truth['text'],'prediction':final_text})
                    if best and complete and not swallowing and not contamination:
                        views=prediction['models'][identity]['views'].get(best['candidate_id'],[])
                        raw=next((view['text'] for view in views if view['transform']=='raw'),None)
                        if raw is not None: crop_pairs[identity].append({'label':truth['text'],'prediction':raw})
                    detail[identity]={'baseline':baseline_text,'prediction':final_text,'mixed_regions':final_mixed}
                scene_detail['regions'].append(detail)
            elif truth['kind']=='body':
                localization['body_regions']+=1
                localization['body_regions_with_candidate']+=int(any(mask_pixels(mask,item['bbox']) for item in routing))
                localization['body_regions_routed']+=int(any(mask_pixels(mask,item['bbox']) for item in routed))
                for identity in model_ids:
                    final_text,final_mixed=aligned_region(prediction['models'][identity]['regions'],index,masks,config)
                    counter=body[identity]; counter['regions']+=1;counter['baseline_mixed_regions']+=mixed
                    counter['final_mixed_regions']+=final_mixed;counter['baseline_exact']+=int(baseline_text==truth['text'])
                    counter['final_exact']+=int(final_text==truth['text'])
                    counter['baseline_correct_to_wrong']+=int(baseline_text==truth['text'] and final_text!=truth['text'])
                    counter['erroneous_rewrites']+=int(final_text!=baseline_text and final_text!=truth['text'])
                    before=joined_regions([item for item in prediction['baseline'] if mask_pixels(mask,item['bbox'])],config)
                    after=joined_regions([item for item in prediction['models'][identity]['regions'] if mask_pixels(mask,item['bbox'])],config)
                    preserved, splice_count = body_preserving_splices(prediction['baseline'],
                                                prediction['models'][identity]['regions'], mask)
                    # This post-inference deletion tripwire uses a whole raw label;
                    # it never extracts a predicted span or changes model coordinates.
                    literal_deleted = bool(truth['text'] and truth['text'] in before and truth['text'] not in after)
                    counter['body_intersection_transcript_changes']+=int(before!=after)
                    counter['body_literal_deleted']+=int(literal_deleted)
                    counter['verified_body_preserving_splices']+=splice_count if preserved and not literal_deleted else 0
                    counter['new_body_intersection_errors']+=int(literal_deleted or
                                                               (before!=after and after!=truth['text'] and not preserved))
            elif truth['kind']=='graphic':
                localization['graphic_regions']+=1
                localization['graphic_regions_with_candidate']+=int(any(mask_pixels(mask,item['bbox']) for item in routing))
                localization['graphic_regions_routed']+=int(any(mask_pixels(mask,item['bbox']) for item in routed))
                for identity in model_ids:
                    final=prediction['models'][identity]['regions']
                    base_nonempty=any(item['text'] and mask_pixels(mask,item['bbox']) for item in prediction['baseline'])
                    final_nonempty=any(item['text'] and mask_pixels(mask,item['bbox']) for item in final)
                    negatives[identity]['graphic_regions']+=1; negatives[identity]['baseline_graphic_nonempty']+=int(base_nonempty)
                    negatives[identity]['final_graphic_nonempty']+=int(final_nonempty)
                    negatives[identity]['new_graphic_nonempty']+=int(final_nonempty and not base_nonempty)
        if row['scene_kind'] in ('blank','graphic-only'):
            localization['negative_scenes']+=1
            localization['negative_scenes_with_candidate']+=int(bool(routing))
            localization['negative_scenes_routed']+=int(bool(routed))
            for identity in model_ids:
                negative=negatives[identity];negative['negative_scenes']+=1
                negative['baseline_nonempty_scenes']+=int(bool(prediction['baseline_text']))
                negative['final_nonempty_scenes']+=int(bool(prediction['models'][identity]['text']))
                negative['new_nonempty_scenes']+=int(not prediction['baseline_text'] and bool(prediction['models'][identity]['text']))
                prefix='blank' if row['scene_kind']=='blank' else 'graphic_only'
                negative[prefix+'_scenes']+=1
                negative[prefix+'_baseline_nonempty']+=int(bool(prediction['baseline_text']))
                negative[prefix+'_final_nonempty']+=int(bool(prediction['models'][identity]['text']))
                negative[prefix+'_new_nonempty']+=int(not prediction['baseline_text'] and bool(prediction['models'][identity]['text']))
        details.append(scene_detail)
    result={'scenes':len(rows),'scene_kinds':dict(kind_counts),'localization':dict(localization),
            'candidate_sources':dict(candidates_by_source),
            'coverage':{'mean':float(np.mean(coverage_values)) if coverage_values else None,
                        'minimum':min(coverage_values) if coverage_values else None},
            'bbox_iou':{'mean':float(np.mean(ious)) if ious else None,'minimum':min(ious) if ious else None},
            'baseline':{'whole_scene':metrics(baseline_pairs,'prediction'),'isolated_kaomoji':metrics(base_faces,'prediction')},
            'models':{}}
    for identity in model_ids:
        protection=body[identity];negative=negatives[identity]
        result['models'][identity]={'whole_scene':metrics(pairs[identity],'prediction'),
          'isolated_kaomoji':metrics(faces[identity],'prediction'),'safe_complete_specialist_crops':metrics(crop_pairs[identity],'prediction'),
          'body':dict(protection),'negative':dict(negative),
          'quality_gates':{'no_new_body_errors':not (protection['baseline_correct_to_wrong'] or protection['erroneous_rewrites'] or protection['new_body_intersection_errors']),
                           'no_new_negative_outputs':not (negative['new_nonempty_scenes'] or negative['new_graphic_nonempty']),
                           'localization_recall_threshold':'not specified; measured only',
                           'app_acceptance':False,'production_enabled':False}}
    return result,details


def evaluate(args):
    root,output=safe_path(args.dataset),output_path(args.output)
    if output.exists(): raise ValueError('Evaluation output already exists')
    config=configuration(args.config); binding=pipeline_binding(root,args.candidate,args.baseline_models,config,args.inputs)
    usage=None
    if args.split=='heldout':
        if not args.selection: raise ValueError('Heldout requires frozen whole-scene selection before any images or truth')
        usage=claim_heldout(root,safe_path(args.selection),binding,output)
    jobs=image_jobs(root,args.split,args.inputs,allow_heldout=usage is not None)
    common.disk_guard(output); output.mkdir(parents=True)
    predictions,resources=run_worker(args,output,binding,jobs,usage)
    try:
        # Ground truth is first loaded here, after actual candidate/fusion output.
        rows=load_truth(root,args.split,jobs,allow_heldout=usage is not None)
        aggregate,details=summarize(root,rows,predictions,config,[item['id'] for item in binding['models']])
        if pipeline_binding(root,args.candidate,args.baseline_models,config,args.inputs)!=binding:
            raise ValueError('Pipeline changed before evaluation report')
        write_new(output/'alignment-private.json',details)
        outputs = {key: {'file': str(output / name), 'sha256': common.digest(output / name)}
                   for key, name in [('predictions','predictions.jsonl'), ('resources','resources.json'),
                                     ('alignment','alignment-private.json')]}
        report={'schema_version':SCHEMA,'completed_utc':common.utc(),'outputs':outputs,
                'split':args.split,'pipeline_binding':binding,'actual_cpu_inference':True,
                'independent_heldout':args.split=='heldout','heldout_usage':usage,'promote':False,'enabled':False,
                'scope':'Actual original-image baseline det/cls/rec and image-derived candidates, model-specific pixel views and fusion; no APP lifecycle acceptance',
                'ground_truth_inference_input':False,'normalization':'none; exact raw Unicode, whitespace and graphemes',
                'regional_alignment':'Visible-ink mask overlap only; a verified regional splice uses its actual automatic ROI and specialist text. Other mixed transcripts are not split using labels; unchanged body prefix/suffix bytes have no invented word boxes',
                'whole_scene_reading_order':'Declared geometry line clustering, two spaces between predicted chunks and LF between lines; regional metrics are separate',
                'summary':aggregate,'resources':resources}
        write_new(output/'evaluation.json',report)
        return report
    except BaseException as error:
        write_new(output/'measurement-failure.json', {'schema_version':SCHEMA,
                  'status':'post-inference measurement failed; no acceptance claim',
                  'error_type':type(error).__name__, 'heldout_remains_consumed':args.split=='heldout'})
        raise


def freeze_selection(args):
    report_path=output_path(args.dev_report)
    config=configuration(args.config); binding=pipeline_binding(args.dataset,args.candidate,args.baseline_models,config,args.inputs)
    final_families(binding); report=verify_dev_evidence(report_path,binding)
    if args.selected_id not in {item['id'] for item in binding['models']} or not args.reason.strip():
        raise ValueError('Frozen research selection needs a measured model identity and explicit development reason')
    budget=safe_path(args.budget_receipt); validate_budget(common.read(budget))
    value={'schema_version':SELECTION_SCHEMA,'frozen_utc':common.utc(),'pipeline_binding':binding,
           'dev_report':{'file':str(report_path),'sha256':common.digest(report_path)},
           'budget_receipt':{'file':str(budget),'sha256':common.digest(budget)},
           'selected_id':args.selected_id,'selection_reason':args.reason,
           'baseline_runtime':report['resources']['baseline_runtime'],'promote':False,'enabled':False,
           'quality_gate_snapshot':{key:item['quality_gates'] for key,item in report['summary']['models'].items()},
           'usage':'research-only-selected-for-first-heldout; no production promotion'}
    write_new(args.output,value); return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='action',required=True)
    for name in ('evaluate','_worker','freeze-selection'):
        command=commands.add_parser(name)
        for key in ('dataset','baseline-models','output'): command.add_argument('--'+key,required=True)
        command.add_argument('--candidate',action='append',default=[])
        command.add_argument('--inputs'); command.add_argument('--config')
        if name in ('evaluate','_worker'):
            command.add_argument('--split',choices=('dev','heldout'),default='dev')
        if name=='evaluate': command.add_argument('--selection')
        elif name=='_worker':
            for key in ('jobs','jobs-sha256','binding','binding-sha256'): command.add_argument('--'+key,required=True)
            command.add_argument('--usage-sha256')
        else:
            for key in ('dev-report','budget-receipt','selected-id','reason'): command.add_argument('--'+key,required=True)
    args=parser.parse_args(argv)
    if args.action=='_worker': worker(args); return 0
    value=evaluate(args) if args.action=='evaluate' else freeze_selection(args)
    print(json.dumps({'schema_version':value['schema_version'],'split':value.get('split'),'enabled':False,
                      'scenes':value.get('summary',{}).get('scenes')},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
