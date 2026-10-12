"""Build anonymous whole-scene chart summaries from complete CPU evidence.

Only reports and receipts are parsed. Bound model, source and private output
files receive opaque streaming SHA/size checks, never content interpretation.
Scene truth, images and masks are not opened. The bound inference-input index
is hashed opaquely; its job records are never interpreted. No inference runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.Documentation import plot_region_evaluation as plot

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts'
RAW_SCHEMA = 'localocr.region-research.v1'
FAMILIES = {'official-ppocrv6-tiny': 'ppocrv6-tiny', 'openocr-repsvtr-mobile': 'repsvtr-mobile'}
RAW_SCOPE = 'Actual original-image baseline det/cls/rec and image-derived candidates, model-specific pixel views and fusion; no APP lifecycle acceptance'
NORMALIZATION = 'none; exact raw Unicode, whitespace and graphemes'
ALIGNMENT = 'Visible-ink mask overlap only; a verified regional splice uses its actual automatic ROI and specialist text. Other mixed transcripts are not split using labels; unchanged body prefix/suffix bytes have no invented word boxes'
LIMITS = {'maximum_source_bytes': 20 * 1024**2, 'maximum_pixels': 50_000_000,
          'maximum_candidates': 48, 'maximum_crop_pixels': 1_000_000,
          'maximum_scene_output_bytes': 8 * 1024**2, 'scene_timeout_seconds': 90,
          'batch_timeout_seconds': 5400}
CANDIDATE_CONFIG = {
    'maximum_pixels': 50_000_000, 'maximum_baseline_regions': 4096,
    'maximum_candidates': 48, 'maximum_crop_pixels': 1_000_000,
    'maximum_components': 8192, 'maximum_bracket_scan_width': 4096,
    'contrast_threshold': 8, 'minimum_component_area': 2,
    'minimum_height': 7, 'maximum_height': 72, 'maximum_width': 1024,
    'line_overlap': .45, 'horizontal_gap_height': .58, 'padding': 3,
    'baseline_coverage': .80, 'baseline_ink_coverage': .98, 'merge_iou': .45,
    'minimum_context_blank_band': 2,
}
FUSION_CONFIG = {'minimum_agreeing_views': 2, 'maximum_output_characters': 256,
                 'require_raw_view': True, 'allow_pixel_only_insertions': True,
                 'require_extra_shape_support': True, 'protect_body_letters_numbers': True}
IMPLEMENTATION_FILES = (
    'scripts/local-ai-ocr-worker.py', 'localocr/__init__.py', 'localocr/bundles.py', 'localocr/version.py',
    'tools/RegionKaomoji/__init__.py', 'tools/RegionKaomoji/evaluate.py',
    'tools/RegionKaomoji/candidates.py', 'tools/RegionKaomoji/fusion.py',
    'tools/PretrainedOcr/__init__.py', 'tools/PretrainedOcr/common.py', 'tools/PretrainedOcr/evaluate.py',
    'tools/OcrDatasets/__init__.py', 'tools/OcrDatasets/evaluate.py',
    'tools/OcrDatasets/validate.py', 'tools/OcrDatasets/private.py',
)
OUTPUT_NAMES = {'predictions': 'predictions.jsonl', 'resources': 'resources.json', 'alignment': 'alignment-private.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_object(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def sha(value):
    plot.checksum(value, 'source SHA')
    return value


def count(value, maximum=None):
    return plot.integer(value, 'source count', maximum)


def timestamp(value):
    require(isinstance(value, str), 'Missing completed/frozen UTC receipt')
    try:
        date = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError('Invalid UTC receipt') from None
    require(date.tzinfo is not None and date.utcoffset().total_seconds() == 0
            and date <= datetime.now(timezone.utc), 'Receipt timestamp must be actual UTC')
    return date


def safe_path(path, fixture=False, source=False, output=False):
    """Inspect ancestors before following them; reject network/ADS/reparse paths."""
    require(isinstance(path, (str, Path)) and str(path), 'Missing local evidence path')
    path = Path(path)
    require(not str(path).startswith(('\\\\', '//')) and '..' not in path.parts,
            'Only local non-traversing paths are permitted')
    if os.name == 'nt':
        import ctypes
        require(not any(':' in part for part in path.parts[1:]), 'Alternate streams are forbidden')
        if path.is_absolute():
            require(ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) != 4, 'Mapped network drives are forbidden')
    path = Path(os.path.abspath(path))
    if os.name == 'nt':
        require(ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) != 4, 'Mapped network drives are forbidden')
    boundary = ROOT if source or output and not fixture else ARTIFACTS
    require(fixture or path.is_relative_to(boundary), 'Actual evidence must remain in controlled LocalOCR artifacts')
    for item in reversed((path, *path.parents)):
        if os.path.lexists(item):
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                    'Reparse points and symlinks are forbidden')
    return path


class Evidence:
    """Read each declared receipt once and stream opaque files without retaining bytes."""
    def __init__(self, fixture=False):
        self.fixture = fixture
        self.entries = []

    def capture(self, path, role, expected=None, size=None, opaque=False, source=False, maximum=32 * 1024**2):
        path = safe_path(path, self.fixture, source=source)
        require(path.is_file() and stat.S_ISREG(path.lstat().st_mode), 'Expected a regular evidence file')
        before = path.stat()
        require(0 < before.st_size <= maximum, 'Evidence byte size exceeds its declared limit')
        if size is not None:
            require(count(size) == before.st_size, 'Evidence byte size differs')
        digest = hashlib.sha256()
        blocks = [] if not opaque else None
        total = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                total += len(block)
                require(total <= maximum, 'Evidence grew beyond its declared byte limit')
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
        safe_path(path, self.fixture, source=source)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                'Evidence changed while hashing')
        actual = digest.hexdigest()
        if expected is not None:
            require(actual == sha(expected), 'Evidence SHA differs from bound source')
        value = None
        if blocks is not None:
            value = plot.strict_json(b''.join(blocks).removeprefix(b'\xef\xbb\xbf'))
            require(self.fixture or not isinstance(value, dict) or value.get('_fixture') is not True,
                    'Labelled fixtures cannot be published as actual evaluation')
        self.entries.append({'path': path, 'role': role, 'sha256': actual, 'bytes': before.st_size,
                             'mtime_ns': after.st_mtime_ns,
                             'mode': 'opaque-streaming-sha-size' if opaque else 'receipt-json'})
        return value, actual

    def receipt(self, path, role, expected=None, name=None):
        if name is not None:
            require(Path(path).name == name, 'Unexpected evidence filename')
        return self.capture(path, role, expected)

    def opaque(self, path, role, expected, size=None, source=False, maximum=512 * 1024**2):
        return self.capture(path, role, expected, size, True, source, maximum)[1]

    def stable(self):
        for entry in self.entries:
            path = safe_path(entry['path'], self.fixture, source=entry['role'] == 'implementation')
            before = path.stat()
            digest = hashlib.sha256()
            total = 0
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    total += len(block)
                    require(total <= entry['bytes'], 'Source evidence changed before anonymous output')
                    digest.update(block)
            safe_path(path, self.fixture, source=entry['role'] == 'implementation')
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                    and after.st_size == entry['bytes'] and after.st_mtime_ns == entry['mtime_ns']
                    and digest.hexdigest() == entry['sha256'],
                    'Source evidence changed before anonymous output')

    def anonymous(self):
        return [{key: item[key] for key in ('role', 'sha256', 'bytes', 'mode')} for item in self.entries]


def reference(base, value):
    require(isinstance(value, dict) and isinstance(value.get('file'), str), 'Missing bound file reference')
    sha(value.get('sha256'))
    path = Path(value['file'])
    return path if path.is_absolute() else Path(base) / path


def preprocessing(value):
    require(isinstance(value, dict) and set(value) == {'height', 'color', 'padding', 'resize', 'width', 'normalize'} and value.get('height') == 48 and value.get('color') in ('RGB', 'BGR')
            and value.get('padding') == 'right-zero-normalized' and value.get('resize') in ('PIL-bilinear', 'cv2-linear'),
            'Unsupported specialist preprocessing binding')
    width = value.get('width', {})
    require(isinstance(width, dict) and set(width) == {'mode', 'min', 'max', 'multiple', 'rounding'}, 'Unknown preprocessing width fields')
    low, high, multiple = width.get('min'), width.get('max'), width.get('multiple')
    require(type(low) is int and low > 0 and type(multiple) is int and multiple > 0 and low % multiple == 0
            and (high is None or type(high) is int and high >= low and high % multiple == 0)
            and width.get('mode') == 'dynamic' and width.get('rounding') == 'ceil-multiple', 'Invalid explicit width policy')
    norm = value.get('normalize', {})
    require(isinstance(norm, dict) and set(norm) == {'mean', 'std'}, 'Unknown preprocessing normalization fields')
    mean, std = norm.get('mean'), norm.get('std')
    require(isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 3
            and all(type(item) in (int, float) and math.isfinite(item) for item in mean + std)
            and all(item > 0 for item in std), 'Invalid explicit normalization')
    return digest_object(value)


def config(value, expected):
    require(isinstance(value, dict) and set(value) == {'schema_version', 'candidate', 'fusion', 'views', 'limits', 'join'}
            and value['schema_version'] == 'localocr.region-pipeline-config.v1'
            and value['views'] == ['raw', 'edge-pad-6'] and value['limits'] == LIMITS
            and value['join'] == {'same_line': '  ', 'different_line': '\n', 'vertical_overlap': .5},
            'Unsupported whole-scene pipeline configuration or CPU budgets')
    candidate_config, fusion = value['candidate'], value['fusion']
    require(isinstance(candidate_config, dict) and set(candidate_config) == set(CANDIDATE_CONFIG)
            and isinstance(fusion, dict) and set(fusion) == set(FUSION_CONFIG),
            'Unknown or incomplete candidate/fusion configuration')
    for key, default in CANDIDATE_CONFIG.items():
        item = candidate_config[key]
        require(type(item) in (int, float) and math.isfinite(item) and item > 0
                and (type(default) is not int or type(item) is int), 'Invalid candidate configuration type/range')
    for key in ('line_overlap', 'baseline_coverage', 'baseline_ink_coverage', 'merge_iou'):
        require(candidate_config[key] <= 1, 'Invalid candidate overlap ratio')
    for key in ('maximum_pixels', 'maximum_candidates', 'maximum_crop_pixels', 'maximum_baseline_regions',
                'maximum_components', 'maximum_bracket_scan_width'):
        require(candidate_config[key] <= CANDIDATE_CONFIG[key], 'Candidate resource ceiling cannot increase')
    require(candidate_config['minimum_context_blank_band'] >= 2
            and candidate_config['minimum_height'] <= candidate_config['maximum_height']
            and candidate_config['contrast_threshold'] <= 255, 'Invalid candidate shape/body guard')
    for key, default in FUSION_CONFIG.items():
        require(type(fusion[key]) is type(default), 'Invalid fusion configuration type')
    require(fusion['require_raw_view'] is True and fusion['require_extra_shape_support'] is True
            and fusion['protect_body_letters_numbers'] is True
            and 2 <= fusion['minimum_agreeing_views'] <= 8 and 1 <= fusion['maximum_output_characters'] <= 256,
            'Independent-view/pixel/body protections cannot be disabled')
    require(digest_object(value) == sha(expected), 'Pipeline configuration SHA differs')


def frozen(value):
    require(isinstance(value, dict) and value.get('schema_version') == 'localocr.region-scenes.freeze.v1',
            'Missing frozen original scene dataset binding')
    sha(value.get('dataset_sha256'))
    require(set(value.get('split_sha256', {})) == {'dev', 'heldout'} and value.get('counts') == plot.COUNTS
            and all(type(item) is int for entry in value['counts'].values() for item in entry.values()),
            'Require frozen development 200/120 and heldout 400/280 scenes/faces')
    for item in value['split_sha256'].values():
        sha(item)
    require(value['split_sha256']['dev'] != value['split_sha256']['heldout'], 'Development/heldout bindings overlap')
    return {'sha256': value['dataset_sha256'], 'split_sha256': value['split_sha256'], 'counts': value['counts']}


def implementation(value, evidence):
    require(isinstance(value, dict), 'Missing implementation SHA closure')
    expected = set(IMPLEMENTATION_FILES)
    require(set(value) == expected, 'Implementation source inventory differs from frozen pipeline')
    for name, expected_sha in value.items():
        evidence.opaque(ROOT / name, 'implementation', sha(expected_sha), source=True, maximum=4 * 1024**2)


def source_references(value, base, prefix='source'):
    references = {}
    if isinstance(value, dict):
        if 'file' in value:
            references[prefix] = {'path': reference(base, value), 'sha256': sha(value.get('sha256'))}
        else:
            for key, item in value.items():
                references.update(source_references(item, base, prefix + '.' + key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            references.update(source_references(item, base, prefix + '.' + str(index)))
    return references


def candidate(item, evidence):
    require(isinstance(item, dict) and item.get('family') in FAMILIES and item.get('stage') == 'finetuned',
            'Require both final finetuned specialist families')
    model_id = item.get('id')
    require(isinstance(model_id, str) and plot.ID.fullmatch(model_id) and model_id != 'production-baseline',
            'Invalid specialist identity')
    path = item.get('candidate_file')
    meta, _ = evidence.receipt(path, 'candidate-metadata', item.get('candidate_sha256'), 'candidate.json')
    require(isinstance(meta, dict) and meta.get('schema_version') == 'localocr.pretrained.candidate.v1'
            and all(meta.get(key) == item.get(key) for key in ('id', 'family', 'stage', 'preprocessing', 'decoder', 'source'))
            and meta.get('promote', False) is False and meta.get('enabled', False) is False,
            'Final candidate metadata differs from CPU binding or enables production')
    parent = Path(path).parent
    model = meta.get('model'); vocab = meta.get('vocabulary')
    require(isinstance(model, dict) and isinstance(vocab, dict), 'Missing model/vocabulary file closure')
    require(model.get('sha256') == item.get('model_sha256') and vocab.get('sha256') == item.get('vocabulary_sha256'),
            'Bound candidate model/vocabulary SHA differs')
    evidence.opaque(reference(parent, model), 'candidate-model', item['model_sha256'], item.get('model_bytes'))
    evidence.opaque(reference(parent, vocab), 'candidate-vocabulary', item['vocabulary_sha256'])
    require(count(item.get('token_count')) >= 2, 'Missing actual candidate vocabulary size')
    require(type(vocab.get('blank_id')) is int and vocab['blank_id'] == 0
            and type(vocab.get('space_id')) is int and 0 < vocab['space_id'] < item['token_count'], 'Invalid exact blank/space identity')
    decoder = item.get('decoder', {})
    require(decoder.get('type') == 'ctc' and decoder.get('repeat') == 'ctc'
            and type(decoder.get('blank_id')) is int and decoder['blank_id'] == 0
            and decoder.get('output') in ('probabilities', 'logits'), 'Unsupported bound CTC decoder')
    references = source_references(meta.get('source'), parent)
    require({'source.checkpoint', 'source.dictionary'}.issubset(references)
            and set(references) == set(item.get('source_files', {})), 'Candidate source file closure is incomplete')
    for key, value in references.items():
        bound = item['source_files'][key]
        require(safe_path(bound.get('file'), evidence.fixture) == safe_path(value['path'], evidence.fixture)
                and bound.get('sha256') == value['sha256'], 'Candidate source reference differs')
        require(value['path'].suffix.lower() in {'.json', '.txt', '.py', '.yml', '.yaml', '.zip', '.onnx', '.pth', '.pdparams', '.pdopt'}
                and value['path'].name not in {'train.json', 'dev.json', 'heldout.json', 'labels.json'},
                'Candidate sources cannot consume scene labels/images/masks')
        evidence.opaque(value['path'], 'candidate-source', value['sha256'])
    return {'id': model_id, 'family': FAMILIES[item['family']], 'stage': 'final-fusion',
            'model_sha256': sha(item['model_sha256']), 'vocabulary_sha256': sha(item['vocabulary_sha256']),
            'preprocessing_sha256': preprocessing(item['preprocessing'])}


def baseline(item, evidence):
    require(isinstance(item, dict) and item.get('id') == 'production-baseline', 'Missing production baseline binding')
    path = item.get('manifest_file')
    meta, _ = evidence.receipt(path, 'baseline-manifest', item.get('manifest_sha256'), 'manifest.json')
    require(isinstance(meta, dict) and meta.get('schema_version') == 'localocr.bundle.v1'
            and meta.get('version') == item.get('version') and meta.get('runtime') == {'rapidocr': '3.10.0', 'onnxruntime': '1.31.0'}
            and meta.get('compatibility') == {'architecture': 'ppocr-det-cls-rec-v1', 'dictionary': 'onnx-embedded-v1',
                                            'preprocessing': 'rapidocr-3.10.0-default-v1', 'postprocessing': 'localocr-symbol-reread-v1'}
            and set(item.get('models', {})) == set(meta.get('models', {})) == {'det', 'cls', 'rec'},
            'Unsupported baseline runtime/architecture or model closure')
    for key, bound in item['models'].items():
        entry = meta['models'][key]
        require(entry.get('sha256') == bound.get('sha256') and entry.get('size') == bound.get('bytes'), 'Baseline model closure differs')
        evidence.opaque(reference(Path(path).parent, entry), 'baseline-model', bound['sha256'], bound['bytes'])
    return sha(item['models']['rec']['sha256'])


def pipeline(value, evidence):
    require(isinstance(value, dict) and value.get('schema_version') == 'localocr.region-pipeline-binding.v1'
            and value.get('promote') is False and value.get('enabled') is False, 'Missing disabled research pipeline binding')
    dataset = frozen(value.get('dataset'))
    inputs = value.get('inputs')
    require(isinstance(inputs, dict) and inputs.get('dataset_sha256') == dataset['sha256']
            and inputs.get('split_sha256') == dataset['split_sha256'], 'Truth-free inference input binding differs')
    require(Path(inputs.get('file', '')).name == 'inference-inputs.json', 'Never open scene labels as inference evidence')
    freeze, _ = evidence.receipt(Path(inputs['file']).parent / 'freeze.json', 'scene-freeze', name='freeze.json')
    require(freeze == value['dataset'], 'Frozen scene receipt differs from CPU dataset/split binding')
    evidence.opaque(inputs['file'], 'truth-free-input-index', inputs.get('sha256'), maximum=8 * 1024**2)
    config(value.get('config'), value.get('config_sha256'))
    implementation(value.get('implementation'), evidence)
    raw = value.get('models')
    require(isinstance(raw, list) and len(raw) == 2, 'Require complete two-family final specialist binding')
    models = [candidate(item, evidence) for item in raw]
    require({item['family'] for item in models} == set(FAMILIES.values())
            and len({item['id'] for item in models}) == 2, 'Repeated/missing final specialist family/identity')
    models.sort(key=lambda item: plot.FAMILIES.index(item['family']))
    baseline_sha = baseline(value.get('baseline'), evidence)
    return dataset, models, baseline_sha


def metric(value, expected):
    require(isinstance(value, dict), 'Missing actual region metric')
    result = {key: count(value.get(key)) for key in ('count', 'exact', 'characters', 'character_errors')}
    plot.metric(result, 'actual face regions', expected)
    cer = value.get('cer')
    require(type(cer) in (int, float) and math.isfinite(cer)
            and math.isclose(cer, result['character_errors'] / result['characters'], abs_tol=1e-12, rel_tol=0),
            'CER ratio differs from raw character/edit counts')
    return result


def resources(report, resource, models, scenes):
    require(isinstance(resource, dict) and resource.get('pipeline_binding') == report['pipeline_binding']
            and resource.get('resources') == report.get('resources'), 'CPU resource receipt differs from final report')
    value = resource['resources']
    require(value.get('providers') == ['CPUExecutionProvider'] and type(value.get('intra_op_threads')) is int
            and value['intra_op_threads'] == 2 and type(value.get('inter_op_threads')) is int and value['inter_op_threads'] == 1
            and value.get('actual_baseline_detection') is True and type(value.get('actual_specialist_models')) is int
            and value['actual_specialist_models'] == len(models)
            and value.get('no_python_children_or_socket_io') is True and type(value.get('scene_timeout_seconds')) is int
            and value['scene_timeout_seconds'] == 90
            and value.get('onnxruntime_version') == '1.31.0' and type(value.get('warmup_scenes')) is int
            and value['warmup_scenes'] == 3 and isinstance(value.get('warm'), dict)
            and type(value['warm'].get('count')) is int and value['warm']['count'] == scenes,
            'Incomplete actual isolated CPU evaluation or wrong backend/threads')
    for key in ('load_seconds', 'first_scene_seconds', 'total_warm_seconds', 'process_wall_seconds'):
        require(type(value.get(key)) in (int, float) and math.isfinite(value[key]) and value[key] >= 0,
                'Missing actual CPU resource measurement')
    require(value['process_wall_seconds'] <= LIMITS['batch_timeout_seconds'], 'Actual CPU batch exceeded fixed timeout')
    return value


def baseline_runtime(value, evidence):
    algorithm = 'sha256(canonical JSON: sorted keys, UTF-8, ensure_ascii=false, compact separators)'
    require(isinstance(value, dict) and value.get('schema_version') == 'localocr.baseline-recognizer.v1'
            and value.get('rapidocr_version') == '3.10.0' and set(value.get('source_files', {})) == {'recognizer', 'decoder'},
            'Missing actual baseline decoder/preprocessing runtime receipt')
    vocab = value.get('vocabulary', {})
    sha(vocab.get('sha256'))
    token_count = count(vocab.get('token_count'))
    require(token_count >= 2 and type(vocab.get('blank_id')) is int and vocab['blank_id'] == 0
            and isinstance(vocab.get('blank_token'), str) and vocab.get('hash_algorithm') == algorithm,
            'Invalid actual baseline token identity/hash semantics')
    spaces = vocab.get('space_ids')
    require(isinstance(spaces, list) and spaces and all(type(item) is int and 0 < item < token_count for item in spaces)
            and len(set(spaces)) == len(spaces), 'Missing exact baseline space token identity')
    prep = value.get('preprocessing', {})
    facts = prep.get('configuration', {})
    require(prep.get('hash_algorithm') == algorithm and prep.get('sha256') == digest_object(facts)
            and isinstance(facts, dict) and facts.get('schema_version') == 'localocr.baseline-rec-preprocessing.v1'
            and facts.get('rapidocr_version') == '3.10.0' and facts.get('color') == 'BGR'
            and facts.get('resize') == 'cv2.INTER_LINEAR' and facts.get('padding') == 'right-zero-normalized'
            and facts.get('layout') == 'NCHW-float32' and facts.get('return_word_box') is False
            and facts.get('normalize') == {'scale': 255, 'mean': [.5] * 3, 'std': [.5] * 3},
            'Baseline preprocessing canonical facts/hash differ')
    shape = facts.get('rec_image_shape')
    require(isinstance(shape, list) and len(shape) == 3 and all(type(item) is int and item > 0 for item in shape)
            and shape[:2] == [3, 48] and type(facts.get('rec_batch_num')) is int and facts['rec_batch_num'] > 0,
            'Baseline actual recognition shape/batch differs')
    require(facts.get('width') == 'int(height * max(configured_width / height, maximum input aspect ratio in the recognition batch))'
            and facts.get('content_width') == 'min(batch_width, ceil(height * input_width / input_height))'
            and isinstance(facts.get('opencv_version'), str) and re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', facts['opencv_version']),
            'Unknown baseline actual width/resize implementation')
    require(facts.get('implementation') == {key: item.get('sha256') for key, item in value['source_files'].items()},
            'Baseline preprocessing source SHA differs')
    for item in value['source_files'].values():
        evidence.opaque(item.get('file'), 'baseline-runtime-source', item.get('sha256'), maximum=4 * 1024**2)
    return vocab['sha256'], sha(prep['sha256'])


def raw_report(path, split, evidence):
    report, report_sha = evidence.receipt(path, split + '-report', name='evaluation.json')
    require(isinstance(report, dict) and report.get('schema_version') == RAW_SCHEMA and report.get('split') == split
            and report.get('actual_cpu_inference') is True and report.get('independent_heldout') is (split == 'heldout')
            and report.get('promote') is False and report.get('enabled') is False
            and report.get('ground_truth_inference_input') is False and report.get('scope') == RAW_SCOPE
            and report.get('normalization') == NORMALIZATION and report.get('regional_alignment') == ALIGNMENT,
            'Require completed actual whole-scene CPU research report')
    timestamp(report.get('completed_utc'))
    require(not any((Path(path).parent / name).exists() for name in
                    ('failure.json', 'measurement-failure.json')), 'Failed evaluation cannot be published as completed')
    binding = report.get('pipeline_binding')
    dataset, identities, baseline_sha = pipeline(binding, evidence)
    aggregate = report.get('summary')
    require(isinstance(aggregate, dict) and aggregate.get('scenes') == plot.COUNTS[split]['scenes']
            and set(aggregate.get('models', {})) == {item['id'] for item in identities},
            'Incomplete whole-scene aggregate/model set')
    kinds = aggregate.get('scene_kinds')
    expected_kinds = {'same-line', 'middle-face', 'different-line', 'low-body', 'small-face', 'low-face',
                      'ordinary-brackets', 'body-graphic', 'graphic-only', 'blank'}
    require(isinstance(kinds, dict) and set(kinds) == expected_kinds
            and sum(count(value) for value in kinds.values()) == aggregate['scenes']
            and all(value > 0 for value in kinds.values()), 'Incomplete synthetic scene category coverage')
    outputs = report.get('outputs')
    require(isinstance(outputs, dict) and set(outputs) == set(OUTPUT_NAMES), 'Missing complete output SHA closure')
    resource = None
    for key, name in OUTPUT_NAMES.items():
        record = outputs[key]
        target = reference(Path(path).parent, record)
        require(safe_path(target, evidence.fixture) == safe_path(Path(path).parent / name, evidence.fixture),
                'CPU output reference escapes its completed evaluation')
        if key == 'resources':
            resource, _ = evidence.receipt(target, 'cpu-resources', record['sha256'], name)
        else:
            maximum = (LIMITS['maximum_scene_output_bytes'] + 16384) * aggregate['scenes']
            evidence.opaque(target, key + '-integrity-only', record['sha256'], maximum=maximum)
    runtime = resources(report, resource, identities, aggregate['scenes'])
    vocab_sha, prep_sha = baseline_runtime(runtime.get('baseline_runtime'), evidence)
    raw_loc = aggregate.get('localization')
    require(isinstance(raw_loc, dict), 'Missing actual discovered/routed localization counters')
    fields = {'faces', 'found_any', 'missed', 'partial_only', 'complete_ink_coverage', 'safe_complete_coverage',
              'eligible_complete_ink_coverage', 'routed_complete_ink_coverage', 'safe_routed_complete_coverage',
              'candidate_count', 'routed_candidate_count', 'routed_candidates_intersecting_body',
              'routed_candidates_intersecting_graphic', 'body_regions', 'body_regions_with_candidate',
              'body_regions_routed', 'graphic_regions', 'graphic_regions_with_candidate', 'graphic_regions_routed',
              'negative_scenes', 'negative_scenes_with_candidate', 'negative_scenes_routed'}
    # Counter omits fields whose candidate loop ran zero times. These zeroes are
    # valid only inside an otherwise complete, hash-bound CPU aggregate.
    localization = {key: count(raw_loc.get(key, 0)) for key in fields}
    localization.update(intersecting_body=count(raw_loc.get('candidates_intersecting_body', 0)),
                        intersecting_graphic=count(raw_loc.get('candidates_intersecting_graphic', 0)))
    body_fields = ('regions', 'baseline_exact', 'final_exact', 'baseline_correct_to_wrong', 'erroneous_rewrites',
                   'body_intersection_transcript_changes', 'new_body_intersection_errors')
    models = []
    for identity in identities:
        raw = aggregate['models'][identity['id']]
        require(isinstance(raw, dict) and isinstance(raw.get('body'), dict) and isinstance(raw.get('negative'), dict),
                'Missing final routed body/negative metrics')
        metric(raw.get('whole_scene'), aggregate['scenes'])
        body = {key: count(raw['body'].get(key)) for key in body_fields}
        require(count(raw['body'].get('body_literal_deleted')) <= body['new_body_intersection_errors'],
                'Literal body deletion guard exceeds reported intersection errors')
        negatives = {}
        for kind, prefix, unit in (('blank', 'blank', 'scenes'), ('graphic-only', 'graphic_only', 'scenes'),
                                   ('graphic_regions', 'graphic', 'regions')):
            total = raw['negative'].get('graphic_regions' if unit == 'regions' else prefix + '_scenes')
            negatives[kind] = {unit: count(total), **{suffix + '_nonempty': count(raw['negative'].get((suffix + '_graphic_nonempty' if unit == 'regions' else prefix + '_' + suffix + '_nonempty')))
                                                   for suffix in ('baseline', 'final', 'new')}}
        for total, suffix in (('negative_scenes', 'scenes'), ('baseline_nonempty_scenes', 'baseline_nonempty'),
                              ('final_nonempty_scenes', 'final_nonempty'), ('new_nonempty_scenes', 'new_nonempty')):
            require(count(raw['negative'].get(total)) == sum(negatives[key][suffix] for key in ('blank', 'graphic-only')),
                    'Negative scene totals differ from grouped counters')
        require(negatives['blank']['scenes'] == kinds['blank'] and negatives['graphic-only']['scenes'] == kinds['graphic-only'],
                'Negative scene counters differ from full category denominators')
        gate = raw.get('quality_gates', {})
        require(gate.get('app_acceptance') is False and gate.get('production_enabled') is False
                and gate.get('no_new_body_errors') is (not any(body[key] for key in ('baseline_correct_to_wrong', 'erroneous_rewrites', 'new_body_intersection_errors')))
                and gate.get('no_new_negative_outputs') is (not any(negatives[key]['new_nonempty'] for key in negatives)),
                'Quality gate counters disagree or claim production/APP acceptance')
        models.append({**identity, 'face_regions': metric(raw.get('isolated_kaomoji'), plot.COUNTS[split]['faces']),
                       'body': body, 'negative': negatives})
    before = aggregate.get('baseline', {})
    metric(before.get('whole_scene'), aggregate['scenes'])
    body = {key: models[0]['body'][key] if key in ('regions', 'baseline_exact') else 0 for key in body_fields}
    body['final_exact'] = body['baseline_exact']
    negative = {kind: {unit: item[unit], 'baseline_nonempty': item['baseline_nonempty'],
                       'final_nonempty': item['baseline_nonempty'], 'new_nonempty': 0}
                for kind, item in models[0]['negative'].items()
                for unit in ('regions' if kind == 'graphic_regions' else 'scenes',)}
    models.insert(0, {'id': 'production-baseline', 'family': 'production-baseline', 'stage': 'production',
                     'model_sha256': baseline_sha, 'vocabulary_sha256': vocab_sha, 'preprocessing_sha256': prep_sha,
                     'face_regions': metric(before.get('isolated_kaomoji'), plot.COUNTS[split]['faces']),
                     'body': body, 'negative': negative})
    anonymous = {'report_sha256': report_sha, 'dataset_sha256': dataset['sha256'],
                 'split_sha256': dataset['split_sha256'][split], 'pipeline_sha256': digest_object(binding),
                 'config_sha256': binding['config_sha256'], 'selection_sha256': None, 'first_use_sha256': None,
                 'evaluation_backend': plot.BACKEND, 'evaluation_scope': plot.SCOPE,
                 'runtime': {'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1},
                 'scenes': aggregate['scenes'], 'localization': localization, 'models': models}
    if split == 'dev':
        plot.validate_report(anonymous, 'dev', dataset)
    require(split != 'dev' or report.get('heldout_usage') is None, 'Development report cannot claim heldout first use')
    return report, anonymous, dataset


def budget(path, evidence, expected=None):
    require('budget' in Path(path).stem and Path(path).suffix == '.json', 'Expected an explicit budget receipt')
    value, actual = evidence.receipt(path, 'shared-gpu-budget', expected)
    require(isinstance(value, dict) and value.get('schema_version') == 'localocr.gpu-budget.v1'
            and type(value.get('maximum_seconds')) in (int, float) and math.isfinite(value['maximum_seconds'])
            and value['maximum_seconds'] == 7200 and value.get('active') is None
            and type(value.get('used_seconds')) in (int, float) and math.isfinite(value['used_seconds'])
            and 0 <= value['used_seconds'] <= 7200, 'Require closed shared 7200-second GPU budget receipt')
    return value, actual


def selection(path, dev_path, dev, dev_sha, budget_path, budget_sha, evidence):
    require(Path(path).name.startswith('evaluation') and Path(path).suffix == '.json', 'Expected a regional selection receipt')
    value, actual = evidence.receipt(path, 'region-selection')
    require(isinstance(value, dict) and value.get('schema_version') == 'localocr.region-selection.v1'
            and value.get('pipeline_binding') == dev['pipeline_binding'] and value.get('promote') is False and value.get('enabled') is False
            and value.get('selected_id') in dev['summary']['models']
            and isinstance(value.get('selection_reason'), str) and value['selection_reason'].strip()
            and value.get('baseline_runtime') == dev['resources']['baseline_runtime'],
            'Frozen selection differs from complete CPU development pipeline')
    require(timestamp(value.get('frozen_utc')) >= timestamp(dev['completed_utc']), 'Selection precedes completed development evidence')
    for key, target, expected in (('dev_report', dev_path, dev_sha), ('budget_receipt', budget_path, budget_sha)):
        record = value.get(key)
        require(safe_path(reference(Path(path).parent, record), evidence.fixture) == safe_path(target, evidence.fixture)
                and record['sha256'] == expected, 'Selection evidence SHA/path binding differs')
    require(value.get('quality_gate_snapshot') == {key: item['quality_gates'] for key, item in dev['summary']['models'].items()},
            'Selection quality gate snapshot differs from actual development')
    return value, actual


def heldout_use(dev, held, path, selection_path, selected, selected_sha, evidence):
    record = held.get('heldout_usage')
    require(isinstance(record, dict), 'Completed heldout requires frozen first-use evidence')
    usage_path = reference(Path(path).parent, record)
    split_sha = held['pipeline_binding']['dataset']['split_sha256']['heldout']
    expected_name = 'evaluation-heldout-first-use-' + split_sha + '.json'
    usage, actual = evidence.receipt(usage_path, 'heldout-first-use', record['sha256'], expected_name)
    require(isinstance(usage, dict) and usage.get('schema_version') == 'localocr.region-heldout-first-use.v1'
            and usage.get('pipeline_binding') == held['pipeline_binding']
            and usage.get('status') == 'claimed-before-reading-heldout-images-or-truth; failure remains consumed'
            and count(usage.get('pid')) > 0, 'Invalid whole-scene first-use claim')
    require(safe_path(reference(Path(usage_path).parent, usage.get('selection')), evidence.fixture) == safe_path(selection_path, evidence.fixture)
            and usage['selection']['sha256'] == selected_sha, 'First-use frozen selection SHA differs')
    root = Path(path).parent
    require(safe_path(usage.get('output'), evidence.fixture) == safe_path(root, evidence.fixture)
            and safe_path(usage.get('worker_output'), evidence.fixture) == safe_path(root / '.worker' / 'predictions.jsonl', evidence.fixture)
            and safe_path(usage.get('worker_jobs'), evidence.fixture) == safe_path(root / '.worker' / 'jobs.json', evidence.fixture),
            'Heldout first-use output/worker slot differs')
    require(timestamp(selected['frozen_utc']) <= timestamp(usage.get('claimed_utc')) <= timestamp(held['completed_utc']),
            'Heldout first use precedes freeze or follows report completion')
    worker_path = Path(usage_path).with_name(Path(usage_path).stem + '-worker.json')
    worker, _ = evidence.receipt(worker_path, 'heldout-worker-use')
    require(isinstance(worker, dict) and worker.get('schema_version') == 'localocr.region-heldout-worker-use.v1'
            and worker.get('usage_sha256') == actual and count(worker.get('pid')) > 0
            and worker.get('status') == 'consumed-before-reading-images; failed workers cannot replay'
            and timestamp(usage['claimed_utc']) <= timestamp(worker.get('claimed_utc')) <= timestamp(held['completed_utc']),
            'Missing actual first-use worker consumption receipt')
    seal_path = Path(held['pipeline_binding']['inputs']['file']).parent / 'heldout-seal.json'
    seal, _ = evidence.receipt(seal_path, 'heldout-seal', usage.get('seal_sha256'), 'heldout-seal.json')
    dataset = held['pipeline_binding']['dataset']
    require(isinstance(seal, dict) and seal.get('schema_version') == 'localocr.region-scenes.seal.v1'
            and seal.get('dataset_sha256') == dataset['dataset_sha256'] and seal.get('heldout_sha256') == split_sha
            and seal.get('inference_results_seen') is False and seal.get('requires_frozen_pipeline_selection') is True,
            'Heldout seal SHA or first-use contract differs')
    return actual


def build(dev_report, budget_receipt, heldout_report=None, selection_receipt=None, fixture=False):
    evidence = Evidence(fixture)
    dev, development, dataset = raw_report(dev_report, 'dev', evidence)
    _, budget_sha = budget(budget_receipt, evidence)
    selected, selected_sha = None, None
    if selection_receipt is not None:
        selected, selected_sha = selection(selection_receipt, dev_report, dev, development['report_sha256'],
                                           budget_receipt, budget_sha, evidence)
        development['selection_sha256'] = selected_sha
    heldout = None
    if heldout_report is not None:
        require(selection_receipt is not None, 'Heldout summary requires explicit frozen selection receipt')
        held, heldout, other = raw_report(heldout_report, 'heldout', evidence)
        require(other == dataset and held['pipeline_binding'] == dev['pipeline_binding']
                and held['resources']['baseline_runtime'] == dev['resources']['baseline_runtime'],
                'Development/heldout dataset/model/preprocessing/config/source/baseline binding differs')
        heldout['selection_sha256'] = selected_sha
        heldout['first_use_sha256'] = heldout_use(dev, held, heldout_report, selection_receipt, selected, selected_sha, evidence)
    summary = {'schema_version': plot.SCHEMA, 'fixture': fixture, 'dataset': dataset,
               'plot_font': {'sha256': plot.FONT_SHA256, 'license': 'OFL-1.1'}, 'dev': development, 'heldout': heldout}
    plot.validate(summary)
    evidence.stable()
    closure = evidence.anonymous()
    receipt = {'schema_version': 'localocr.region.summary-build-receipt.v1', 'fixture': fixture,
               'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'schema_validator_sha256': hashlib.sha256(Path(plot.__file__).read_bytes()).hexdigest(),
               'pipeline_sha256': development['pipeline_sha256'], 'budget_receipt_sha256': budget_sha,
               'baseline_vocabulary_hash_semantics': 'actual CPU decoder canonical token-list SHA from bound resource receipt; token list not read',
               'baseline_preprocessing_hash_semantics': 'verified canonical actual preprocessing facts and source SHA',
               'opaque_integrity_checks': True, 'private_output_content_parsed': False,
               'scene_truth_images_or_masks_opened': False, 'inference_job_index_content_parsed': False,
               'opaque_inference_job_index_hashed': True, 'model_execution': False,
               'source_closure_sha256': digest_object(closure), 'source_closure': closure,
               'scope': plot.SCOPE}
    return summary, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-report', required=True, type=Path)
    parser.add_argument('--budget-receipt', required=True, type=Path)
    parser.add_argument('--heldout-report', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture', action='store_true', help='Transparent tool fixture; outputs confined to artifacts')
    args = parser.parse_args(argv)
    output = safe_path(args.output, args.fixture, output=True)
    require(output.suffix == '.json', 'Anonymous summary output must be JSON')
    receipt_path = output.with_name(output.stem + '.build-receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'Fresh summary/receipt output is required')
    if args.fixture:
        require(output.is_relative_to(ARTIFACTS), 'Fixture output must remain in artifacts')
    summary, receipt = build(args.dev_report, args.budget_receipt, args.heldout_report, args.selection, args.fixture)
    raw = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    receipt['summary_sha256'] = hashlib.sha256(raw).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_path(output.parent, args.fixture, output=True)
    with output.open('xb') as stream:
        stream.write(raw)
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'fixture': args.fixture, 'development_scenes': summary['dev']['scenes'],
                      'heldout_scenes': summary['heldout']['scenes'] if summary['heldout'] else None,
                      'summary_sha256': receipt['summary_sha256']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
