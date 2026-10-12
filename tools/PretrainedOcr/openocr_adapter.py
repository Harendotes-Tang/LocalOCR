"""Offline RepSVTR Mobile research, exact codepoint CTC and verified head transfer.

Only the hash-bound official implementation is imported. No stock OpenOCR
inference helper (which downloads assets and overrides dictionaries) is used.
The research checkpoint and ONNX artifacts are never production bundle v1.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import random
import re
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

from tools.PretrainedOcr import common, quality_selection

COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
SOURCE_SHA = 'c7ffe75b07ffc50f23dd38320e5607db50babf32536e23abcffae6d2fb625334'
SOURCE_BYTES = 2047647
WEIGHT_BYTES = 65279266
WEIGHT_SHA = 'fbf8fc54a1962c500cd07683408c72476c584c4c33e242e8a706fc9b59a9c0dc'
BLANK = '<blank>'
_VERIFIED_SOURCES = {}
PREPROCESSING = {'height': 48, 'color': 'RGB',
                 'width': {'mode': 'dynamic', 'min': 16, 'max': 1024, 'multiple': 4,
                           'rounding': 'ceil-multiple'},
                 'padding': 'right-zero-normalized', 'resize': 'PIL-bilinear',
                 'normalize': {'mean': [0.5] * 3, 'std': [0.5] * 3}}


def validate_inputs(receipt):
    """Verify all local source bytes before importing or deserializing weights."""
    data = common.read(receipt)
    if data['source_commit'] != COMMIT:
        raise ValueError('Unapproved OpenOCR source commit')
    archive = next(Path(x['path']).resolve() for x in data['downloads'] if Path(x['path']).name == 'source.zip')
    if archive.stat().st_size != SOURCE_BYTES or common.digest(archive) != SOURCE_SHA:
        raise ValueError('Official source archive bytes differ')
    expected = {}
    prefix = 'OpenOCR-' + COMMIT + '/'
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir(): continue
            if not member.filename.startswith(prefix): raise ValueError('Unexpected official archive root')
            relative = member.filename[len(prefix):]
            if not relative or '\\' in relative or ':' in relative or any(part in ('', '.', '..') for part in relative.split('/')):
                raise ValueError('Unsafe official archive path')
            if relative in expected: raise ValueError('Duplicate official archive file')
            raw = bundle.read(member)
            expected[relative] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    inventory = data.get('source_file_inventory', [])
    claimed = {}
    for record in inventory:
        relative = record['path']
        if relative in claimed: raise ValueError('Duplicate source inventory record')
        claimed[relative] = {'bytes': record['bytes'], 'sha256': record['sha256']}
    if not expected or claimed != expected:
        raise ValueError('Source inventory differs from the fixed official archive')
    source = Path(data['extracted_root']).resolve()
    actual = set()
    for path in source.rglob('*'):
        if path.is_symlink() or (getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Source reparse entries are unsupported')
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            if path.suffix == '.pyc' and '__pycache__' in path.relative_to(source).parts: continue
            actual.add(relative)
    if actual != set(expected): raise ValueError('Extracted source file closure differs from fixed archive')
    for relative, record in expected.items():
        path = common.confined(source, relative)
        if path.stat().st_size != record['bytes'] or common.digest(path) != record['sha256']:
            raise ValueError('Official source bytes changed: ' + relative)
    weight = next(Path(x['path']).resolve() for x in data['downloads']
                  if Path(x['path']).name == 'openocr_repsvtr_ch.pth')
    if common.digest(weight) != WEIGHT_SHA or weight.stat().st_size != WEIGHT_BYTES:
        raise ValueError('Official checkpoint bytes differ')
    _VERIFIED_SOURCES[str(source)] = expected
    return source, weight, data


def original_tokens(source):
    # Match official UTF-8 dictionary newline codec without stripping spaces.
    raw = (Path(source) / 'tools/utils/ppocr_keys_v1.txt').read_bytes()
    chars = [line.decode('utf-8').rstrip('\r\n') for line in raw.split(b'\n') if line]
    if any(len(c) != 1 for c in chars) or len(set(chars)) != len(chars) or ' ' in chars:
        raise ValueError('Unsupported original dictionary identities')
    return [BLANK] + chars + [' ']


def encode(text, tokens):
    lookup = {c: n for n, c in enumerate(tokens) if n != 0}
    if any(c not in lookup for c in text):
        raise ValueError('Unknown exact codepoint; never skip labels')
    return [lookup[c] for c in text]


def decode(probabilities, tokens):
    import numpy as np
    results = []
    for sequence in probabilities:
        ids = sequence.argmax(axis=-1)
        text, scores, previous = [], [], None
        for pos, identity in enumerate(ids):
            identity = int(identity)
            if identity != 0 and identity != previous:
                text.append(tokens[identity]); scores.append(float(sequence[pos, identity]))
            previous = identity
        results.append((''.join(text), float(np.mean(scores)) if scores else 0.0))
    return results


def preprocess(image, policy=None):
    """Use the shared exact RGB/padding contract, including train-derived width."""
    from tools.PretrainedOcr.evaluate import prepare_image
    return prepare_image(image, policy or PREPROCESSING)


def batch_images(images, policy=None):
    import numpy as np
    arrays = [preprocess(image, policy) for image in images]
    result = np.zeros((len(arrays), 3, 48, max(a.shape[2] for a in arrays)), dtype=np.float32)
    for n, array in enumerate(arrays):
        result[n, :, :, :array.shape[2]] = array
    return result


def verify_imported_source(source):
    """Bind regular modules and official namespace packages to verified source."""
    source = Path(source).resolve()
    for name, loaded in list(sys.modules.items()):
        if name != 'openrec' and not name.startswith('openrec.'):
            continue
        filename = getattr(loaded, '__file__', None)
        locations = ([filename] if filename else []) + list(getattr(loaded, '__path__', ()))
        spec = getattr(loaded, '__spec__', None)
        origin = getattr(spec, 'origin', None)
        if origin is not None:
            locations.append(origin)
        locations.extend(getattr(spec, 'submodule_search_locations', None) or ())
        if not locations or any(not Path(path).resolve().is_relative_to(source) for path in locations):
            raise ValueError('Foreign OpenOCR implementation already imported: ' + name)


class VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile fixed official source bytes; never read or write bytecode caches."""
    def __init__(self, name, filename, source, expected):
        super().__init__(name, filename)
        self.verified_source = str(source)
        self.expected = expected

    def get_code(self, fullname):
        path = Path(self.path).resolve()
        relative = path.relative_to(Path(self.verified_source)).as_posix()
        record = self.expected.get(relative)
        raw = path.read_bytes()
        if record is None or len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('OpenOCR source changed before compile: ' + relative)
        return compile(raw, str(path), 'exec', dont_inherit=True)


class VerifiedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source, expected):
        self.source = Path(source).resolve()
        self.expected = expected

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'openrec' and not fullname.startswith('openrec.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        locations = list(spec.submodule_search_locations or ())
        if spec.origin is not None:
            locations.append(spec.origin)
        if not locations or any(not Path(item).resolve().is_relative_to(self.source) for item in locations):
            raise ValueError('Foreign OpenOCR import specification: ' + fullname)
        if spec.origin is not None:
            if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                raise ValueError('OpenOCR requires fixed Python source: ' + fullname)
            spec.loader = VerifiedSourceLoader(fullname, spec.origin, self.source, self.expected)
        return spec


def install_source_importer(source):
    source = Path(source).resolve()
    expected = _VERIFIED_SOURCES.get(str(source))
    if not expected:
        raise ValueError('OpenOCR source must be validated before import')
    for name, loaded in list(sys.modules.items()):
        if name != 'openrec' and not name.startswith('openrec.'):
            continue
        spec = getattr(loaded, '__spec__', None)
        if getattr(spec, 'origin', None) is not None:
            loader = getattr(spec, 'loader', None)
            if not isinstance(loader, VerifiedSourceLoader) or loader.verified_source != str(source):
                raise ValueError('OpenOCR module was imported without fixed-source compilation: ' + name)
    finders = [finder for finder in sys.meta_path if isinstance(finder, VerifiedSourceFinder)]
    if any(finder.source != source for finder in finders):
        raise ValueError('A different OpenOCR source importer is already active')
    if not finders:
        sys.meta_path.insert(0, VerifiedSourceFinder(source, expected))


def build_model(source, classes):
    import yaml
    source = Path(source).resolve()
    verify_imported_source(source)
    install_source_importer(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from openrec.modeling import build_model as official_build
    verify_imported_source(source)
    cfg = yaml.safe_load((source / 'configs/rec/svtrv2/repsvtr_ch.yml').read_text('utf-8'))
    if cfg['Architecture']['algorithm'] != 'SVTRv2_mobile':
        raise ValueError('Expected actual Mobile architecture')
    cfg['Architecture']['Decoder']['out_channels'] = classes
    return official_build(cfg['Architecture'])


def transplant(model, old_state, old_tokens, new_tokens):
    """Retain every backbone tensor and old classifier row by token identity."""
    import torch
    mapping = common.token_mapping(old_tokens, new_tokens)
    state = model.state_dict()
    extras = set(old_state) - set(state)
    missing = set(state) - set(old_state)
    if any(not k.endswith('.num_batches_tracked') for k in missing) or any(not k.startswith(('decoder.before_gtc.', 'decoder.gtc_head.')) for k in extras):
        raise ValueError('Checkpoint architecture keys differ')
    head_keys = {'decoder.fc.weight', 'decoder.fc.bias'}
    records = {}
    for name, tensor in state.items():
        if name in missing:
            if tensor.ndim != 0 or tensor.item() != 0:
                raise ValueError("Unexpected missing BN counter default")
            continue
        old = old_state[name]
        if name in head_keys:
            if old.shape[0] != len(old_tokens) or tensor.shape[0] != len(new_tokens) or old.shape[1:] != tensor.shape[1:]:
                raise ValueError('Classifier dimensions disagree with vocabulary')
            for old_row, new_row in mapping.items():
                tensor[new_row].copy_(old[old_row])
            records[name] = {'retained_rows': len(mapping), 'new_rows': len(new_tokens) - len(old_tokens)}
        else:
            if tensor.shape != old.shape:
                raise ValueError('Backbone shape changed: ' + name)
            state[name] = old
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    for name, old in old_state.items():
        if name in extras:
            continue
        if name in head_keys:
            if any(not torch.equal(actual[name][new], old[old_index]) for old_index, new in mapping.items()):
                raise AssertionError('Old classifier row changed')
        elif not torch.equal(actual[name], old):
            raise AssertionError('Backbone initialization changed')
    return {'heads': records, 'initialized_missing_bn_counters': sorted(missing), 'discarded_training_only_sgm': {
                name: {'shape': list(old_state[name].shape),
                       'sha256': hashlib.sha256(old_state[name].numpy().tobytes()).hexdigest()}
                for name in sorted(extras)}, 'all_backbone_tensors_identical': True,
            'all_old_rows_identical': True, 'old_to_new': mapping}


def save_checkpoint(path, model, vocabulary_sha, steps):
    import torch
    common.disk_guard(path)
    temporary = Path(str(path) + '.partial')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'vocabulary_sha256': vocabulary_sha, 'steps': steps}, temporary)
    temporary.replace(path)


def prepare(args):
    import torch
    torch.set_num_threads(1)
    source, weight, receipt = validate_inputs(args.inputs_receipt)
    common.disk_guard(args.output)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    old = original_tokens(source)
    chars = common.extended_characters(old[1:], common.train_characters(args.dataset))
    tokens = [BLANK] + chars
    rows = common.read_split(args.dataset, 'train')
    encoded = [encode(row['text'], tokens) for row in rows]
    needed = max((len(label) + sum(a == b for a, b in zip(label, label[1:])) for label in encoded), default=0)
    policy = copy.deepcopy(PREPROCESSING)
    policy['width']['min'] = max(16, needed * 8)
    policy['width']['max'] = max(1024, policy['width']['min'])
    vocabulary = {'schema_version': 'localocr.pretrained.vocabulary.v1', 'tokens': tokens,
                  'blank_id': 0, 'space_id': tokens.index(' '), 'extension_source': 'train-only'}
    common.write(output / 'vocabulary.json', vocabulary)
    model = build_model(source, len(tokens))
    state = torch.load(weight, map_location='cpu', weights_only=True)['state_dict']
    transfer = transplant(model, state, old, tokens)
    vocab_sha = common.digest(output / 'vocabulary.json')
    save_checkpoint(output / 'initial.pth', model, vocab_sha, 0)
    common.write(output / 'prepare.json', {'family': 'openocr-repsvtr-mobile', 'source_commit': COMMIT,
        'inputs_receipt': str(Path(args.inputs_receipt).resolve()), 'source_root': str(source),
        'official_checkpoint': str(weight), 'official_checkpoint_sha256': WEIGHT_SHA,
        'dictionary_sha256': common.digest(source / 'tools/utils/ppocr_keys_v1.txt'),
        'dataset_root': str(Path(args.dataset).resolve()), 'dataset_binding': common.dataset_binding(args.dataset),
        'vocabulary_sha256': vocab_sha, 'original_classes': len(old), 'classes': len(tokens),
        'transfer': transfer, 'seed': args.seed, 'preprocessing': policy,
        'train_ctc_feasibility': {'rows': len(rows), 'maximum_required_timesteps': needed,
                                  'observed_width_stride': 8, 'minimum_width': policy['width']['min']}})


def validate_warm_start(directory, info):
    directory = Path(directory)
    path = directory / 'warm-start.json'
    if not path.exists():
        return
    receipt = common.read(path)
    initial_sha = common.digest(directory / 'initial.pth')
    if initial_sha != receipt['initial_checkpoint_sha256'] or initial_sha != receipt['parent_checkpoint']['sha256']:
        raise ValueError('Warm-start immutable parent initialization changed')
    for record in receipt['parent_evidence']:
        if common.digest(common.confined(directory, record['file'])) != record['sha256']:
            raise ValueError('Warm-start parent evidence changed')
    parent_prepare = directory / 'parent-prepare.json'
    parent_vocab = directory / 'parent-vocabulary.json'
    if (directory / 'parent-run-freeze.json').exists():
        frozen = common.read(directory / 'parent-run-freeze.json')
        inventory = {item['file']: item for item in frozen['files']}
        expected_prepare = inventory['prepare.json']['sha256']
        expected_vocab = inventory['vocabulary.json']['sha256']
    else:
        trained = common.read(directory / 'parent-training.json')
        expected_prepare = trained['preparation_sha256']
        expected_vocab = trained['vocabulary_sha256']
    if common.digest(parent_prepare) != expected_prepare or common.digest(parent_vocab) != expected_vocab:
        raise ValueError('Warm-start parent preparation differs from frozen run evidence')
    parent_info = common.read(parent_prepare)
    if expected_vocab != parent_info['vocabulary_sha256'] or common.digest(directory / 'vocabulary.json') != expected_vocab:
        raise ValueError('Warm-start actual vocabulary differs from frozen parent')
    for key in ('vocabulary_sha256', 'preprocessing', 'dataset_binding'):
        if receipt[key] != parent_info[key] or info[key] != parent_info[key]:
            raise ValueError('Warm-start preparation differs from frozen parent: ' + key)


def load_prepared(directory, checkpoint='initial.pth'):
    import torch
    directory = Path(directory)
    info = common.read(directory / 'prepare.json')
    validate_warm_start(directory, info)
    source, _, _ = validate_inputs(info['inputs_receipt'])
    if common.digest(directory / 'vocabulary.json') != info['vocabulary_sha256']:
        raise ValueError('Prepared vocabulary changed')
    tokens = common.read(directory / 'vocabulary.json')['tokens']
    model = build_model(source, len(tokens))
    saved = torch.load(common.confined(directory, checkpoint), map_location='cpu', weights_only=True)
    if saved['vocabulary_sha256'] != info['vocabulary_sha256']:
        raise ValueError('Checkpoint dictionary mismatch')
    model.load_state_dict(saved['state_dict'], strict=True)
    return model, tokens, info


def read_images(dataset, rows):
    from PIL import Image
    images = []
    for row in rows:
        p = common.confined(dataset, row['image'])
        if common.digest(p) != row['sha256']:
            raise ValueError('Frozen sample image bytes differ')
        with Image.open(p) as image:
            images.append(image.convert('RGB').copy())
    return images


def exact_ctc_loss(logits, labels, negative_weight=1.0):
    """Mean per-token CTC, with empty negatives normalized by available frames."""
    import torch
    if not math.isfinite(negative_weight) or negative_weight <= 0:
        raise ValueError('Negative-target weight must be positive and finite')
    t = logits.shape[1]
    if any(len(label) + sum(a == b for a, b in zip(label, label[1:])) > t for label in labels):
        raise ValueError('CTC time axis too short for exact repeated label')
    target = torch.tensor([n for label in labels for n in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    raw = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)(
        logits.log_softmax(-1).transpose(0, 1), target,
        torch.full((len(labels),), t, dtype=torch.long), lengths)
    denominators = torch.tensor([len(label) if label else t for label in labels],
                                dtype=raw.dtype, device=raw.device)
    weights = torch.tensor([1.0 if label else negative_weight for label in labels],
                           dtype=raw.dtype, device=raw.device)
    return (raw / denominators * weights).mean()


def predict_dev(model, tokens, info, rows, batch, deadline):
    import torch
    results = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch):
            if deadline.remaining() <= 30: return None
            chunk = rows[offset:offset + batch]
            data = torch.from_numpy(batch_images(read_images(info['dataset_root'], chunk), info['preprocessing'])).cuda()
            predictions = decode(model(data).cpu().numpy(), tokens)
            results.extend({'image': row['image'], 'text': text, 'score': score}
                           for row, (text, score) in zip(chunk, predictions))
    return results


def dev_statistics(rows, predictions, original=None, quality=False):
    from tools.OcrDatasets.evaluate import distance
    kinds = {}
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        stats = kinds.setdefault(row['kind'], {'samples': 0, 'exact_correct': 0, 'edit_distance': 0,
            'reference_codepoints': 0, 'negative_false_outputs': 0,
            'original_correct_to_wrong': 0, 'erroneous_rewrites': 0})
        expected, actual = row['text'], prediction['text']
        stats['samples'] += 1; stats['exact_correct'] += int(expected == actual)
        stats['edit_distance'] += distance(expected, actual); stats['reference_codepoints'] += len(expected)
        stats['negative_false_outputs'] += int(not expected and bool(actual))
        if original is not None:
            old = original[index]['text']
            stats['original_correct_to_wrong'] += int(old == expected and actual != expected)
            stats['erroneous_rewrites'] += int(old != expected and actual != expected and old != actual)
    for value in kinds.values():
        value['exact_accuracy'] = value['exact_correct'] / value['samples']
        value['cer'] = value['edit_distance'] / value['reference_codepoints'] if value['reference_codepoints'] else None
    result = {'exact_accuracy': sum(x['exact_correct'] for x in kinds.values()) / max(1, len(rows)), 'by_kind': kinds}
    if quality: result['quality'] = quality_selection.summarize(rows, predictions, reference=original)
    return result


REGION_SAMPLING = {'face': 0.4, 'punctuation': 0.2, 'plain': 0.2, 'emoji': 0.1, 'negative': 0.1}


def sampling_kind(row):
    return row['kind'] if row['text'] else 'negative'


def sampling_order(rows, policy, rng):
    if policy == 'uniform':
        order = list(range(len(rows))); rng.shuffle(order); return order
    if policy != 'region-balanced':
        raise ValueError('Unknown training sampling policy')
    counts = {kind: 0 for kind in REGION_SAMPLING}
    for row in rows:
        kind = sampling_kind(row)
        if kind not in counts:
            raise ValueError('Unsupported region-balanced training kind: ' + kind)
        counts[kind] += 1
    if not all(counts.values()):
        raise ValueError('Region-balanced training requires all frozen training kinds')
    weights = [REGION_SAMPLING[sampling_kind(row)] / counts[sampling_kind(row)] for row in rows]
    return rng.choices(range(len(rows)), weights=weights, k=len(rows))


def development_better(candidate, incumbent, policy='legacy'):
    if policy == 'face-pareto':
        quality_selection.validate_summary(candidate.get('quality'))
        return incumbent is None or quality_selection.face_better(candidate['quality'], incumbent.get('quality'))
    if policy != 'legacy': raise ValueError('Unknown development selection policy')
    return incumbent is None or candidate['exact_accuracy'] > incumbent['exact_accuracy']


def train(args, probe=False):
    selection_policy = getattr(args, 'selection_policy', 'legacy')
    if selection_policy not in ('legacy', 'face-pareto'):
        raise ValueError('Unknown development selection policy')
    specialist = not probe and selection_policy == 'face-pareto'
    import torch
    directory = Path(args.prepared)
    if specialist and any((directory / name).exists() for name in ('training.json', 'best.pth', 'best-selection.json')):
        raise ValueError('Specialist training run must be fresh')
    torch.set_num_threads(2)
    model, tokens, info = load_prepared(directory)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    rows = common.read_split(info['dataset_root'], 'train')
    dev = [] if probe else common.read_split(info['dataset_root'], 'dev')
    for row in rows:
        encode(row['text'], tokens)
    if common.dataset_binding(info['dataset_root']) != info['dataset_binding']:
        raise ValueError('Prepared dataset changed')
    common.disk_guard(directory)
    rng_seed = getattr(args, 'seed', None)
    if rng_seed is None: rng_seed = info['seed']
    random.seed(rng_seed); torch.manual_seed(rng_seed)
    initial_sha = common.digest(directory / 'initial.pth')
    run_name = 'probe' if probe else 'last'
    if probe:
        counter = 2
        while (directory / (run_name + '.json')).exists() or (directory / (run_name + '.pth')).exists():
            run_name = 'probe-' + str(counter).zfill(3); counter += 1
    if probe:
        positive = [row for row in rows if row['text']][:args.batch]
        negative = [row for row in rows if not row['text']][:args.batch]
        if len(positive) != args.batch or len(negative) != args.batch:
            raise ValueError('Probe needs full positive and negative batches')
        rows = positive + negative
    sampling_policy = getattr(args, 'sampling_policy', 'uniform')
    if not probe: sampling_order(rows, sampling_policy, random.Random(rng_seed))
    coverage = None
    if specialist:
        fixed_source, _, _ = validate_inputs(info['inputs_receipt'])
        coverage = quality_selection.original_class_coverage(rows, original_tokens(fixed_source))
        snapshot = directory / 'quality-selection-source.py'
        if snapshot.exists(): raise ValueError('Specialist training source snapshot must be fresh')
        snapshot.write_bytes(Path(quality_selection.__file__).read_bytes())
    history = []; steps = 0; best = -1; nonzero = False; best_stats = None; initial_quality = None
    maximum_steps = 2 if probe else args.steps
    with common.GPUBudget(args.budget).run('openocr-probe' if probe else 'openocr-train',
                                         limit_seconds=min(120, getattr(args, 'seconds', 120)) if probe else args.seconds) as deadline:
        optimizer = original_model = batch = logits = loss = norm = None
        failure = None
        try:
            model = model.cuda(); model.train(); torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-5)
            original_predictions = None
            if not probe:
                source, weight, _ = validate_inputs(info['inputs_receipt'])
                original_tokens_list = original_tokens(source)
                original_model = build_model(source, len(original_tokens_list))
                transplant(original_model, torch.load(weight, map_location='cpu', weights_only=True)['state_dict'],
                           original_tokens_list, original_tokens_list)
                original_model = original_model.cuda().eval()
                original_predictions = predict_dev(original_model, original_tokens_list, info, dev, args.batch, deadline)
                if original_predictions is None: raise RuntimeError('Lease exhausted before original dev reference completed')
                common.write(directory / 'original-dev.json', {'predictions': original_predictions,
                    'metrics': dev_statistics(dev, original_predictions), 'preprocessing': info['preprocessing'],
                    'checkpoint_sha256': WEIGHT_SHA, 'dev_split_sha256': info['dataset_binding']['split_sha256']['dev']})
                original_model = None
                torch.cuda.empty_cache()
                if specialist:
                    model.eval()
                    initial_predictions = predict_dev(model, tokens, info, dev, args.batch, deadline)
                    if initial_predictions is None: raise RuntimeError('Lease exhausted before specialist initial development reference')
                    best_stats = dev_statistics(dev, initial_predictions, original_predictions, quality=True)
                    initial_quality = best_stats['quality']; best = best_stats['exact_accuracy']
                    common.write(directory / 'quality-initial-dev.json', {'metrics': best_stats, 'predictions': initial_predictions,
                        'checkpoint_sha256': initial_sha, 'dev_split_sha256': info['dataset_binding']['split_sha256']['dev']})
                    model.train()
            while deadline.remaining() > (0 if probe else 30) and steps < maximum_steps:
                order = list(range(len(rows))) if probe else sampling_order(rows, sampling_policy, random)
                run_batch = args.batch
                for start in range(0, len(order), run_batch):
                    if deadline.remaining() <= (0 if probe else 30) or steps >= maximum_steps: break
                    step_started = time.perf_counter()
                    chosen = [rows[n] for n in order[start:start + run_batch]]
                    pixels = batch_images(read_images(info['dataset_root'], chosen), info['preprocessing'])
                    probe_width = getattr(args, 'probe_width', None) if probe else None
                    if probe_width is not None:
                        import numpy as np
                        width_policy = info['preprocessing']['width']
                        if not (width_policy['min'] <= probe_width <= width_policy['max']) or probe_width % width_policy['multiple']:
                            raise ValueError('Probe width must follow the frozen train-derived policy')
                        if pixels.shape[-1] > probe_width:
                            raise ValueError('Capacity probe cannot crop content')
                        pixels = np.pad(pixels, ((0, 0), (0, 0), (0, 0), (0, probe_width - pixels.shape[-1])))
                    batch = torch.from_numpy(pixels).cuda()
                    labels = [encode(row['text'], tokens) for row in chosen]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch)
                    loss = exact_ctc_loss(logits, labels, getattr(args, 'negative_weight', 1.0))
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite CTC loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if not torch.isfinite(norm) or norm.item() == 0: raise RuntimeError('Invalid gradient')
                    nonzero = True; optimizer.step(); torch.cuda.synchronize(); steps += 1
                    history.append({'step': steps, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                                    'remaining_seconds': deadline.remaining(), 'batch_size': len(chosen),
                                    'step_seconds': time.perf_counter() - step_started,
                                    'samples_per_second': len(chosen) / max(1e-9, time.perf_counter() - step_started),
                                    'empty_targets': sum(not label for label in labels),
                                    'sample_kind_counts': {kind: sum(sampling_kind(row) == kind for row in chosen)
                                                           for kind in sorted({sampling_kind(row) for row in chosen})}})
                    if steps % 50 == 0:
                        common.disk_guard(directory); common.write(directory / 'training-progress.json', history)
                    if not probe and steps % args.eval_every == 0 and deadline.remaining() > 30:
                        model.eval()
                        predictions = predict_dev(model, tokens, info, dev, args.batch, deadline)
                        if predictions is not None:
                            stats = dev_statistics(dev, predictions, original_predictions, quality=specialist)
                            acc = stats['exact_accuracy']; history[-1]['dev_metrics'] = stats
                            if (development_better(stats, best_stats, selection_policy) if specialist else acc > best):
                                best = acc; best_stats = stats; save_checkpoint(directory / 'best.pth', model, info['vocabulary_sha256'], steps)
                                common.write(directory / 'best-dev.json', {'predictions': predictions, 'metrics': stats})
                                common.write(directory / 'best-selection.json', {'step': steps, 'dev_exact': acc,
                                    'dev_metrics': stats, 'checkpoint_sha256': common.digest(directory / 'best.pth'),
                                    'dev_split_sha256': info['dataset_binding']['split_sha256']['dev'],
                                    'selection_policy': selection_policy, 'quality': stats.get('quality'),
                                    'quality_selection_sha256': common.digest(Path(quality_selection.__file__)) if specialist else None})
                        model.train()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            save_checkpoint(directory / (run_name + '.pth'), model, info['vocabulary_sha256'], steps)
        except BaseException as error:
            # Helper traceback frames can retain CUDA tensors after a failed
            # forward/backward. Drop those references while the lease is held.
            traceback.clear_frames(error.__traceback__)
            failure = error.with_traceback(None)
            print(f'OpenOCR training failure: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
            optimizer = batch = logits = loss = norm = None
            if original_model is not None:
                original_model.cpu()
            original_model = None
            model.cpu()
            # Autograd/optimizer reference cycles are not released by refcount
            # alone; collect them before checking the lease's live allocation.
            gc.collect()
            torch.cuda.synchronize()
            before_workspace_release = torch.cuda.memory_allocated()
            # Pinned Torch 2.10 caches cuBLAS/cuBLASLt workspaces independently
            # of empty_cache; its own memory-leak tests clear this same pool.
            # https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/cuda.rst
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            released_gpu_bytes = torch.cuda.memory_allocated()
            if failure is not None or released_gpu_bytes:
                report = {'schema_version': 'localocr.openocr.failure.v1', 'steps': steps,
                    'original_failure': None if failure is None else {'type': type(failure).__name__, 'message': str(failure)},
                    'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
                    'allocated_before_workspace_release': before_workspace_release,
                    'allocated_before_lease_release': released_gpu_bytes,
                    'budget': str(Path(args.budget).resolve()), 'history': history}
                common.write(directory / (run_name + '-failure.json'), report)
                print(json.dumps(report), file=sys.stderr, flush=True)
            if released_gpu_bytes != 0:
                raise RuntimeError(f'GPU tensors remain live before lease release: {released_gpu_bytes} bytes')
        if failure is not None:
            raise failure from None
    if probe and steps != 2:
        raise RuntimeError('Probe did not finish both positive and negative CUDA backward')
    destination = directory / (run_name + '.pth')
    reloaded = torch.load(destination, map_location='cpu', weights_only=True)
    if steps == 0 or not nonzero or common.digest(destination) == initial_sha:
        raise RuntimeError('No effective pretrained weight update')
    # Verify tensor change, rather than counting metadata/serialization changes.
    initial = torch.load(directory / 'initial.pth', map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict(reloaded['state_dict'], strict=True)
    changed = [k for k, v in reloaded['state_dict'].items() if not torch.equal(initial[k], v)]
    if not changed: raise RuntimeError('No model tensor changed')
    common.write(directory / (run_name + '.json') if probe else directory / 'training.json',
        {'steps': steps, 'nonzero_gradient': nonzero, 'changed_tensors': changed,
         'history': history, 'peak_cuda_allocated': peak, 'checkpoint_sha256': common.digest(destination),
         'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
         'budget': str(Path(args.budget).resolve()), 'best_dev_exact': best,
         'loss_normalization': 'positive-target-codepoints; empty-negative-timesteps',
         'negative_target_weight': getattr(args, 'negative_weight', 1.0),
         'learning_rate': args.lr, 'training_batch_size': args.batch, 'optimizer': 'Adam', 'optimizer_reset': True,
         'sampling_policy': sampling_policy, 'target_sampling_distribution': REGION_SAMPLING if sampling_policy == 'region-balanced' else None,
         'selection_policy': selection_policy, 'original_class_coverage': coverage,
         'initial_quality': initial_quality, 'selected_quality': best_stats.get('quality') if best_stats else None,
         'specialist_checkpoint_selected': specialist and (directory / 'best.pth').exists(),
         'quality_selection_sha256': common.digest(Path(quality_selection.__file__)) if specialist else None,
         'actual_sample_kind_counts': {kind: sum(item['sample_kind_counts'].get(kind, 0) for item in history)
                                     for kind in sorted({key for item in history for key in item['sample_kind_counts']})},
         'probe_tensor_width': getattr(args, 'probe_width', None) if probe else None,
         'allocated_before_workspace_release': before_workspace_release,
         'save_reload_strict': True, 'gpu_allocated_before_lease_release': released_gpu_bytes,
         'adapter_sha256': common.digest(__file__), 'training_rng_seed': rng_seed,
         'initial_checkpoint_sha256': initial_sha,
         'preparation_sha256': common.digest(directory / 'prepare.json'),
         'vocabulary_sha256': common.digest(directory / 'vocabulary.json'),
         'warm_start_sha256': common.digest(directory / 'warm-start.json') if (directory / 'warm-start.json').exists() else None,
         'trained_samples': sum(item['batch_size'] for item in history),
         'trained_empty_negative_samples': sum(item['empty_targets'] for item in history)})


def fuse_encoder(module):
    # RepViT provides exact eval BN fusion. Retain any standalone BN, never
    # replace an unfused normalization with Identity merely to simplify export.
    for name, child in list(module.named_children()):
        if hasattr(child, 'fuse'):
            fused = child.fuse(); setattr(module, name, fused); fuse_encoder(fused)
        else:
            fuse_encoder(child)


def finetune_evidence(directory, checkpoint):
    """Bind a trained checkpoint to its real run, including interrupted runs."""
    import torch
    directory = Path(directory)
    path = common.confined(directory, checkpoint)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    initial = torch.load(directory / 'initial.pth', map_location='cpu', weights_only=True)
    if not isinstance(saved.get('steps'), int) or saved['steps'] <= 0:
        raise ValueError('Finetuned checkpoint has no actual training steps')
    if set(saved['state_dict']) != set(initial['state_dict']):
        raise ValueError('Finetuned checkpoint architecture differs')
    changed = [name for name, value in saved['state_dict'].items()
               if not torch.equal(value, initial['state_dict'][name])]
    if not changed:
        raise ValueError('Finetuned checkpoint has no actual tensor update')
    checkpoint_sha = common.digest(path)
    files = []; selection = None
    if (directory / 'run-freeze.json').exists():
        receipt = common.read(directory / 'run-freeze.json')
        inventory = {item['file']: item for item in receipt['files']}
        for name in (checkpoint, 'initial.pth', 'last-failure.json', 'best-selection.json'):
            if name not in inventory or common.digest(directory / name) != inventory[name]['sha256']:
                raise ValueError('Frozen interrupted-run evidence differs: ' + name)
        failure = common.read(directory / 'last-failure.json')
        selection = common.read(directory / 'best-selection.json')
        if receipt['completed'] or receipt['exit_code'] != 1 or failure['steps'] < saved['steps']:
            raise ValueError('Interrupted-run status differs')
        if selection['checkpoint_sha256'] != checkpoint_sha or selection['step'] != saved['steps']:
            raise ValueError('Selected checkpoint differs from actual run')
        if not any(point['step'] == saved['steps'] and 'dev_metrics' in point for point in failure['history']):
            raise ValueError('Selected step absent from run history')
        files = ['run-freeze.json', 'last-failure.json', 'best-selection.json']
        completed = False
    else:
        receipt = common.read(directory / 'training.json')
        if receipt['steps'] < saved['steps'] or not receipt['nonzero_gradient'] or not receipt['changed_tensors']:
            raise ValueError('Training receipt has no actual update')
        if checkpoint_sha != receipt['checkpoint_sha256']:
            selection = common.read(directory / 'best-selection.json')
            if selection['checkpoint_sha256'] != checkpoint_sha or selection['step'] != saved['steps']:
                raise ValueError('Checkpoint not bound to training selection')
            files.append('best-selection.json')
        if receipt['initial_checkpoint_sha256'] != common.digest(directory / 'initial.pth'):
            raise ValueError('Training initialization differs')
        files.append('training.json'); completed = True
        if receipt.get('warm_start_sha256') is not None:
            if common.digest(directory / 'warm-start.json') != receipt['warm_start_sha256']:
                raise ValueError('Warm-start parent evidence differs')
            files.append('warm-start.json')
            warm_receipt = common.read(directory / 'warm-start.json')
            for record in warm_receipt['parent_evidence']:
                if common.digest(common.confined(directory, record['file'])) != record['sha256']:
                    raise ValueError('Warm-start ancestor evidence differs')
                files.append(record['file'])
    declared = {value for value in (receipt.get('selection_policy'), (selection or {}).get('selection_policy')) if value is not None}
    if len(declared) > 1 or not declared <= {'legacy', 'face-pareto'}:
        raise ValueError('Development selection policy evidence conflicts')
    policy = next(iter(declared), 'legacy')
    if (directory / 'quality-selection-source.py').exists() and policy != 'face-pareto':
        raise ValueError('Specialist source evidence cannot use legacy selection')
    if policy == 'face-pareto':
        selection = common.read(directory / 'best-selection.json')
        initial_record = common.read(directory / 'quality-initial-dev.json')
        if selection.get('checkpoint_sha256') != checkpoint_sha or selection.get('step') != saved['steps']:
            raise ValueError('Specialist export requires the selected best checkpoint')
        source_sha = selection.get('quality_selection_sha256')
        if (common.digest(directory / 'quality-selection-source.py') != source_sha
                or common.digest(Path(quality_selection.__file__)) != source_sha
                or (completed and receipt.get('quality_selection_sha256') != source_sha)):
            raise ValueError('Specialist selection source binding differs')
        if not completed:
            for name in ('quality-selection-source.py', 'quality-initial-dev.json'):
                if name not in inventory or common.digest(directory / name) != inventory[name]['sha256']:
                    raise ValueError('Frozen specialist development evidence differs: ' + name)
        original = initial_record.get('metrics', {}).get('quality')
        if (initial_record.get('checkpoint_sha256') != common.digest(directory / 'initial.pth')
                or (completed and original != receipt.get('initial_quality'))):
            raise ValueError('Specialist actual initialization evidence differs')
        history = receipt['history'] if completed else failure['history']
        selected_point = next((point for point in history if point.get('step') == saved['steps']), None)
        point_quality = (selected_point or {}).get('dev_metrics', {}).get('quality')
        if selection.get('quality') != point_quality or selection.get('dev_metrics', {}).get('quality') != point_quality:
            raise ValueError('Specialist selected quality differs from actual training history')
        if not quality_selection.face_better(selection.get('quality'), original):
            raise ValueError('Specialist selected checkpoint did not improve its actual initialization')
        files.extend(name for name in ('quality-selection-source.py', 'quality-initial-dev.json', 'best-selection.json') if name not in files)
    elif policy != 'legacy':
        raise ValueError('Unknown development selection policy in training evidence')
    return {'steps': saved['steps'], 'changed_tensors': changed, 'run_completed': completed}, files


def export(args):
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    directory = Path(args.prepared) if args.prepared else None
    if args.original:
        prepared_info = common.read(directory / 'prepare.json') if directory else None
        receipt_path = args.inputs_receipt or prepared_info['inputs_receipt']
        source, weight, _ = validate_inputs(receipt_path)
        info = {'inputs_receipt': str(Path(receipt_path).resolve()), 'official_checkpoint': str(weight),
                'source_root': str(source), 'dictionary_sha256': common.digest(source / 'tools/utils/ppocr_keys_v1.txt'),
                'dataset_binding': prepared_info['dataset_binding'] if prepared_info else None,
                'preprocessing': prepared_info['preprocessing'] if prepared_info else PREPROCESSING}
        tokens = original_tokens(source); model = build_model(source, len(tokens))
        state = torch.load(weight, map_location='cpu', weights_only=True)['state_dict']
        transplant(model, state, tokens, tokens)
    else:
        model, tokens, info = load_prepared(directory, args.checkpoint)
    evidence, evidence_files = (None, []) if args.original or args.checkpoint == 'initial.pth' else finetune_evidence(directory, args.checkpoint)
    output = Path(args.output); common.disk_guard(output); output.mkdir(parents=True, exist_ok=False)
    model.cpu().eval(); fused = copy.deepcopy(model); fuse_encoder(fused.encoder)
    torch.set_num_threads(args.threads)
    example = torch.randn(1, 3, 48, 320)
    torch.onnx.export(fused, example, output / 'model.onnx', input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch', 3: 'width'}, 'output': {0: 'batch', 1: 'time'}},
        opset_version=17, dynamo=False)
    onnx.checker.check_model(str(output / 'model.onnx'))
    options = ort.SessionOptions(); options.intra_op_num_threads = args.threads; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    checks = []
    for batch, width in [(1, 16), (1, 64), (2, 320), (1, 1024)]:
        sample = torch.rand(batch, 3, 48, width) * 2 - 1
        with torch.inference_mode():
            native = model(sample).numpy(); deployed = fused(sample).numpy()
        actual = session.run(None, {'input': sample.numpy()})[0]
        if not np.allclose(native, deployed, atol=3e-4, rtol=3e-3): raise RuntimeError('BN fusion mismatch')
        if not np.allclose(deployed, actual, atol=3e-4, rtol=3e-3): raise RuntimeError('Native/ONNX mismatch')
        if [x[0] for x in decode(deployed, tokens)] != [x[0] for x in decode(actual, tokens)]:
            raise RuntimeError('Native/ONNX exact decode differs')
        checks.append({'batch': batch, 'width': width, 'shape': list(actual.shape),
                       'maximum_absolute_difference': float(np.max(np.abs(actual - deployed))), 'decode_identical': True})
    common.write(output / 'vocabulary.json', {'schema_version': 'localocr.pretrained.vocabulary.v1',
        'tokens': tokens, 'blank_id': 0, 'space_id': tokens.index(' ')})
    checkpoint_source = Path(info['official_checkpoint']) if args.original else common.confined(directory, args.checkpoint)
    shutil.copyfile(checkpoint_source, output / 'checkpoint.pth')
    shutil.copyfile(Path(info['source_root']) / 'tools/utils/ppocr_keys_v1.txt', output / 'pretrained-dictionary.txt')
    shutil.copyfile(info['inputs_receipt'], output / 'inputs-manifest.json')
    if directory:
        shutil.copyfile(directory / 'prepare.json', output / 'preparation.json')
    bound_source = {'commit': COMMIT, 'official_checkpoint_sha256': WEIGHT_SHA,
        'checkpoint': {'file': 'checkpoint.pth', 'sha256': common.digest(output / 'checkpoint.pth')},
        'dictionary': {'file': 'pretrained-dictionary.txt', 'sha256': common.digest(output / 'pretrained-dictionary.txt')},
        'source_manifest': {'file': 'inputs-manifest.json', 'sha256': common.digest(output / 'inputs-manifest.json')},
        'dataset_binding': info['dataset_binding']}
    if evidence is not None:
        shutil.copyfile(directory / 'initial.pth', output / 'initialization.pth')
        bound_source['initialization'] = {'file': 'initialization.pth', 'sha256': common.digest(output / 'initialization.pth')}
        bound_source['finetune'] = evidence
        for name in evidence_files:
            shutil.copyfile(directory / name, output / name)
            bound_source['run_' + name.removesuffix('.json').replace('-', '_')] = {'file': name, 'sha256': common.digest(output / name)}
    if directory:
        bound_source['preparation'] = {'file': 'preparation.json', 'sha256': common.digest(output / 'preparation.json')}
    common.write(output / 'candidate.json', {'schema_version': 'localocr.pretrained.candidate.v1',
        'id': getattr(args, 'candidate_id', None) or ('repsvtr-mobile-pretrained-develop0.0.1' if args.original else 'repsvtr-mobile-exp-20261011-001-' + args.checkpoint.removesuffix('.pth')),
        'family': 'openocr-repsvtr-mobile', 'production_compatible': False,
        'enabled': False, 'promote': False,
        'stage': 'official-pretrained' if args.original else ('expanded-head-initialization' if args.checkpoint == 'initial.pth' else 'finetuned'),
        'model': {'file': 'model.onnx', 'sha256': common.digest(output / 'model.onnx')},
        'vocabulary': {'file': 'vocabulary.json', 'sha256': common.digest(output / 'vocabulary.json'),
                       'blank_id': 0, 'space_id': tokens.index(' ')}, 'preprocessing': info['preprocessing'],
        'decoder': {'type': 'ctc', 'output': 'probabilities', 'blank_id': 0, 'repeat': 'ctc'},
        'source': bound_source,
        'export': {'adapter_sha256': common.digest(__file__), 'opset': 17, 'torch': torch.__version__, 'onnx': onnx.__version__,
                   'onnxruntime': ort.__version__, 'providers': session.get_providers(), 'checks': checks}})


def warm_start(args):
    """Create a separate run from an explicitly selected immutable parent."""
    directory = Path(args.prepared).resolve()
    checkpoint = common.confined(directory, args.checkpoint)
    if common.digest(checkpoint) != args.checkpoint_sha256:
        raise ValueError('Warm-start checkpoint differs from explicit approval hash')
    evidence, evidence_files = finetune_evidence(directory, args.checkpoint)
    _, _, info = load_prepared(directory, args.checkpoint)
    if common.dataset_binding(info['dataset_root']) != info['dataset_binding']:
        raise ValueError('Warm-start frozen dataset binding differs')
    if (directory / 'run-freeze.json').exists():
        inventory = {item['file']: item for item in common.read(directory / 'run-freeze.json')['files']}
        expected_prepare = inventory['prepare.json']['sha256']
        expected_vocab = inventory['vocabulary.json']['sha256']
    else:
        parent_training = common.read(directory / 'training.json')
        expected_prepare = parent_training['preparation_sha256']
        expected_vocab = parent_training['vocabulary_sha256']
    if common.digest(directory / 'prepare.json') != expected_prepare or common.digest(directory / 'vocabulary.json') != expected_vocab:
        raise ValueError('Parent preparation differs from frozen training evidence')
    output = Path(args.output); common.disk_guard(output); output.mkdir(parents=True, exist_ok=False)
    for name in ('prepare.json', 'vocabulary.json'):
        shutil.copyfile(directory / name, output / name)
    shutil.copyfile(checkpoint, output / 'initial.pth')
    records = []
    for name in evidence_files + ['prepare.json', 'vocabulary.json']:
        target = 'parent-' + name
        shutil.copyfile(directory / name, output / target)
        records.append({'file': target, 'sha256': common.digest(output / target)})
    common.write(output / 'warm-start.json', {'schema_version': 'localocr.openocr.warm-start.v1',
        'parent_checkpoint': {'path': str(checkpoint), 'sha256': args.checkpoint_sha256},
        'initial_checkpoint_sha256': common.digest(output / 'initial.pth'),
        'parent_finetune': evidence, 'parent_evidence': records,
        'dataset_binding': info['dataset_binding'], 'vocabulary_sha256': info['vocabulary_sha256'],
        'preprocessing': info['preprocessing'], 'adapter_sha256': common.digest(__file__),
        'training_started': False, 'heldout_read': False})
    validate_warm_start(output, info)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare'); prepare_parser.add_argument('--inputs-receipt', required=True)
    prepare_parser.add_argument('--dataset', required=True); prepare_parser.add_argument('--output', required=True)
    prepare_parser.add_argument('--seed', type=int, default=48)
    warm = sub.add_parser('warm-start'); warm.add_argument('--prepared', required=True)
    warm.add_argument('--checkpoint', required=True); warm.add_argument('--checkpoint-sha256', required=True)
    warm.add_argument('--output', required=True)
    for name in ('probe', 'train'):
        p = sub.add_parser(name); p.add_argument('--prepared', required=True); p.add_argument('--budget', required=True)
        p.add_argument('--batch', type=int, default=8); p.add_argument('--lr', type=float, default=0.0001)
        p.add_argument('--seconds', type=float, default=2700); p.add_argument('--steps', type=int, default=100000)
        p.add_argument('--eval-every', type=int, default=500); p.add_argument('--seed', type=int)
        p.add_argument('--negative-weight', type=float, default=1.0)
        p.add_argument('--sampling-policy', choices=('uniform', 'region-balanced'), default='uniform')
        p.add_argument('--selection-policy', choices=('legacy', 'face-pareto'), default='legacy')
        if name == 'probe': p.add_argument('--probe-width', type=int, help='Capacity-only normalized zero padding within frozen width policy')
    p = sub.add_parser('export'); p.add_argument('--prepared'); p.add_argument('--inputs-receipt'); p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', default='initial.pth'); p.add_argument('--original', action='store_true')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--candidate-id', help='Unique research experiment identity for later warm-start runs')
    args = parser.parse_args(argv)
    if getattr(args, 'batch', 1) < 1 or getattr(args, 'eval_every', 1) < 1 or getattr(args, 'steps', 1) < 1:
        parser.error('batch, steps and eval-every must be positive')
    if getattr(args, 'seconds', 1) <= 0 or getattr(args, 'lr', 1) <= 0:
        parser.error('duration and learning rate must be positive')
    if not math.isfinite(getattr(args, 'negative_weight', 1.0)) or getattr(args, 'negative_weight', 1.0) <= 0:
        parser.error('negative-weight must be positive and finite')
    if args.command == 'prepare': prepare(args)
    elif args.command == 'warm-start': warm_start(args)
    elif args.command == 'export':
        if args.candidate_id and (args.original or not re.fullmatch(r'repsvtr-mobile-exp-[0-9]{8}-[0-9]{3}(?:-[a-z0-9-]+)?', args.candidate_id)):
            parser.error('candidate-id must identify a research experiment, and cannot relabel official weights')
        if not args.prepared and (not args.original or not args.inputs_receipt):
            parser.error('export needs --prepared or --original with --inputs-receipt')
        export(args)
    else: train(args, probe=args.command == 'probe')


if __name__ == '__main__':
    main()
