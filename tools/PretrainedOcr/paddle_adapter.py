"""Fixed official PP-OCRv6 Tiny research, exact token expansion and CPU export.

All models remain inactive. Training only accepts frozen synthetic train/dev.
The adapter changes the official training-only pool from fixed T=40 to the
original inference pool, retaining all learned layers and parameter identities.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.util
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import time
import traceback
import unicodedata
import weakref
import zipfile
import numpy as np
from . import quality_selection
from .common import (GPUBudget, confined, dataset_binding, digest, disk_guard,
                     extended_characters, read, read_split, token_mapping,
                     train_characters, utc, write)

REPO = Path(__file__).resolve().parents[2]
ROUND = REPO / 'artifacts/pretrained-round-20261011'
INPUT = ROUND / 'inputs/paddleocr'
COMMIT = 'b03f46425e8ff4442b268ce449e3eef758146cd4'
SOURCE = INPUT / ('PaddleOCR-' + COMMIT)
ARCHIVE = INPUT / ('PaddleOCR-' + COMMIT + '.zip')
CHECKPOINT = INPUT / 'PP-OCRv6_tiny_rec_pretrained.pdparams'
CONFIG = SOURCE / 'configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml'
DICTIONARY = SOURCE / 'ppocr/utils/dict/ppocrv6_tiny_dict.txt'
BUDGET = ROUND / 'gpu-budget.json'
ARCHIVE_SHA = '449760f5d29842ab6f9cb02cbfbc1f343a5b40fbfd7105333835d72d6a5c741e'
CHECKPOINT_SHA = '960cb4aa5276e3ac235b7f671fb8c9a7c1c1423617da0f96da66a33d0ed53f84'
DICT_SHA = 'c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd'
CONFIG_SHA = '8524a08f95c024df1dbd7e33ec10ff2303733fc6f5ac6fc56b76ffdf2743ab3b'
HEAD_AXES = {'head.ctc_head.fc2.weight': ('ctc', 1),
             'head.ctc_head.fc2.bias': ('ctc', 0),
             'head.gtc_head.embedding.embedding.weight': ('nrtr', 0),
             'head.gtc_head.tgt_word_prj.weight': ('nrtr', 1)}
PREPROCESSING = {'height': 48, 'color': 'BGR',
    'width': {'mode': 'dynamic', 'min': 320, 'max': None, 'multiple': 1, 'rounding': 'ceil-multiple'},
    'padding': 'right-zero-normalized', 'normalize': {'mean': [.5]*3, 'std': [.5]*3}, 'resize': 'cv2-linear'}


def tokens(characters, head='ctc'):
    if ' ' in characters or len(set(characters)) != len(characters) or any(len(c) != 1 for c in characters):
        raise ValueError('Raw dictionary must contain unique scalar characters excluding implicit space')
    if head == 'ctc':
        return ['<blank>', *characters, ' ']
    if head == 'nrtr':
        # The original Transformer allocates one additional unused category.
        return ['<blank>', '<unk>', '<s>', '</s>', *characters, ' ', '<nrtr-unused>']
    raise ValueError('Unknown Tiny head')


def remap_array(old, initialized, mapping, axis):
    old, result = np.asarray(old), np.asarray(initialized).copy()
    if old.ndim != result.ndim or axis not in range(old.ndim):
        raise ValueError('Invalid output-head axis')
    if any(old.shape[d] != result.shape[d] for d in range(old.ndim) if d != axis):
        raise ValueError('Non-token output dimensions changed')
    if set(mapping) != set(range(old.shape[axis])) or len(set(mapping.values())) != len(mapping):
        raise ValueError('Incomplete or ambiguous token mapping')
    if any(i < 0 or i >= result.shape[axis] for i in mapping.values()):
        raise ValueError('Token mapping exceeds output')
    for source, target in mapping.items():
        a, b = [slice(None)] * old.ndim, [slice(None)] * old.ndim
        a[axis], b[axis] = source, target
        result[tuple(b)] = old[tuple(a)]
    return result


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


_SOURCE_FILES = {}
_SOURCE_TOKEN = object()


class _FixedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, expected_sha256):
        super().__init__(name, path)
        self.expected_sha256 = expected_sha256

    def get_code(self, fullname):
        # Read exact source bytes directly. Standard timestamp/size bytecode
        # caches do not prove execution of the fixed ZIP's source bytes.
        if fullname != self.name:
            raise ValueError('Official loader module identity differs')
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.expected_sha256:
            raise ValueError('Official executed source bytes differ')
        return self.source_to_code(source, self.path)

    def exec_module(self, module):
        super().exec_module(module)
        module.__localocr_source_token__ = _SOURCE_TOKEN
        module.__localocr_source_sha256__ = self.expected_sha256


class _FixedNamespaceLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__localocr_source_token__ = _SOURCE_TOKEN
        module.__localocr_source_sha256__ = None


class _FixedSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'ppocr' and not fullname.startswith('ppocr.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        expected_path = SOURCE.joinpath(*fullname.split('.'))
        if spec is None:
            raise ImportError('Official ppocr module is absent: ' + fullname)
        if spec.origin is None:
            paths = list(spec.submodule_search_locations or ())
            if not paths or any(Path(item).resolve() != expected_path for item in paths):
                raise ImportError('Official ppocr namespace resolves outside fixed source')
            spec.loader = _FixedNamespaceLoader()
        else:
            origin = Path(spec.origin).resolve()
            if origin not in (expected_path.with_suffix('.py'), expected_path / '__init__.py'):
                raise ImportError('Official ppocr module resolves outside fixed source')
            relative = origin.relative_to(SOURCE).as_posix()
            if relative not in _SOURCE_FILES:
                raise ImportError('Official ppocr module is not in fixed ZIP')
            spec.loader = _FixedSourceLoader(fullname, str(origin), _SOURCE_FILES[relative])
        return spec


def _install_fixed_imports(expected):
    global _SOURCE_FILES
    _SOURCE_FILES = dict(expected)
    if not any(isinstance(finder, _FixedSourceFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FixedSourceFinder())


def _verify_source_inventory(root, expected):
    root = Path(root).resolve()
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Official source contains a symbolic link')
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == '.pyc' and path.parent.name == '__pycache__':
            continue
        actual[relative] = path
    if set(actual) != set(expected):
        raise ValueError('Official source inventory differs; missing=' + str(sorted(set(expected) - set(actual))) +
                         '; extra=' + str(sorted(set(actual) - set(expected))))
    for relative, path in actual.items():
        if digest(path) != expected[relative]:
            raise ValueError('Official source bytes differ: ' + relative)
    return [{'file': str(actual[relative]), 'sha256': expected[relative]} for relative in sorted(expected)]


def _verify_cached_modules(source=SOURCE):
    source = Path(source).resolve()
    def location(value, is_directory=False):
        if not value:
            raise ValueError('Cached ppocr module has no source origin')
        path = Path(value).resolve()
        try:
            path.relative_to(source / 'ppocr')
        except ValueError:
            raise ValueError('Cached ppocr module origin is outside fixed official source') from None
        invalid = not path.is_dir() if is_directory else (not path.is_file() or path.suffix != '.py')
        if invalid:
            raise ValueError('Cached ppocr module source origin is invalid')
    for name, module in tuple(sys.modules.items()):
        if name != 'ppocr' and not name.startswith('ppocr.'):
            continue
        if module is None:
            raise ValueError('Cached ppocr module is incomplete')
        spec = getattr(module, '__spec__', None)
        if spec is None or spec.name != name:
            raise ValueError('Cached ppocr module specification differs')
        filename = getattr(module, '__file__', None)
        paths = list(getattr(module, '__path__', ()))
        spec_paths = list(spec.submodule_search_locations or ())
        if filename is None and spec.origin is None:
            # Several fixed official directories are Python namespace packages.
            # Their complete search path must be the matching source directory.
            expected = source.joinpath(*name.split('.'))
            if not paths or paths != spec_paths or (expected / '__init__.py').exists():
                raise ValueError('Cached ppocr namespace specification differs')
            for path in paths:
                location(path, is_directory=True)
                if Path(path).resolve() != expected:
                    raise ValueError('Cached ppocr namespace origin is outside fixed official source')
        else:
            location(filename)
            location(spec.origin)
            if Path(filename).resolve() != Path(spec.origin).resolve():
                raise ValueError('Cached ppocr module file/spec origin differs')
            for path in paths + spec_paths:
                location(path, is_directory=True)
            expected_path = source.joinpath(*name.split('.'))
            if Path(filename).resolve() not in (expected_path.with_suffix('.py'), expected_path / '__init__.py'):
                raise ValueError('Cached ppocr logical module/file identity differs')
        if getattr(module, '__localocr_source_token__', None) is not _SOURCE_TOKEN or module.__loader__ is not spec.loader:
            raise ValueError('Cached ppocr module execution is not bound to fixed source bytes')
        if filename is not None:
            relative = Path(filename).resolve().relative_to(source).as_posix()
            if not isinstance(spec.loader, _FixedSourceLoader) or getattr(module, '__localocr_source_sha256__', None) != _SOURCE_FILES.get(relative):
                raise ValueError('Cached ppocr module executed source checksum differs')
        elif not isinstance(spec.loader, _FixedNamespaceLoader):
            raise ValueError('Cached ppocr namespace execution is not fixed')


def source_audit():
    for path, expected in ((ARCHIVE, ARCHIVE_SHA), (CHECKPOINT, CHECKPOINT_SHA),
                           (DICTIONARY, DICT_SHA), (CONFIG, CONFIG_SHA)):
        if digest(path) != expected:
            raise ValueError('Fixed official input checksum differs: ' + path.name)
    expected = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        prefix = SOURCE.name + '/'
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not member.filename.startswith(prefix):
                raise ValueError('Fixed archive root differs')
            relative = member.filename[len(prefix):]
            confined(SOURCE, relative)
            if relative in expected:
                raise ValueError('Duplicate source archive member')
            expected[relative] = hashlib.sha256(archive.read(member)).hexdigest()
    files = _verify_source_inventory(SOURCE, expected)
    _install_fixed_imports(expected)
    _verify_cached_modules()
    return {'commit': COMMIT, 'archive_sha256': ARCHIVE_SHA, 'source_inventory': files,
            'python_files': [item for item in files if Path(item['file']).suffix == '.py' and '/ppocr/' in Path(item['file']).as_posix()],
            'source_files': len(files), 'source_execution': 'direct compile of ZIP-bound .py bytes; bytecode cache bypassed; cached modules require fixed-loader receipt', 'checkpoint_sha256': CHECKPOINT_SHA, 'checked_utc': utc()}


def _paddle(device='cpu'):
    import paddle
    if paddle.__version__ not in ('3.3.0', '3.1.1') or (device.startswith('gpu') and paddle.__version__ != '3.3.0'):
        raise RuntimeError('GPU research requires Paddle 3.3.0; CPU conversion permits fixed 3.1.1')
    if device.startswith('gpu') and (not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1):
        raise RuntimeError('Paddle CUDA unavailable; no CPU fallback')
    paddle.set_device(device)
    return paddle


def build_model(characters, maximum_length=64):
    import paddle
    import yaml
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    _verify_cached_modules()
    specification = importlib.util.find_spec('ppocr')
    if specification is None or Path(specification.origin).resolve() != SOURCE / 'ppocr/__init__.py':
        raise ValueError('ppocr import does not resolve to fixed official source')
    from ppocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4
    from ppocr.modeling.heads.rec_multi_head import MultiHead
    _verify_cached_modules()
    architecture = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))['Architecture']
    if architecture.get('Transform') is not None or architecture.get('Neck') is not None:
        raise ValueError('Unexpected official architecture')
    backbone, head = copy.deepcopy(architecture['Backbone']), copy.deepcopy(architecture['Head'])
    if backbone.pop('name') != 'PPLCNetV4' or backbone.get('model_size') != 'tiny' or head.pop('name') != 'MultiHead':
        raise ValueError('Not official Tiny')
    head['head_list'][1]['NRTRHead']['max_text_length'] = maximum_length + 2
    count = len(tokens(characters))
    head['out_channels_list'] = {'CTCLabelDecode': count, 'NRTRLabelDecode': count + 3}

    class DynamicTrainingBackbone(PPLCNetV4):
        def forward(self, x):
            # Identical learned layers and official inference pooling in both
            # modes. BatchNorm layers still follow train/eval mode normally.
            x = self.conv1(x)
            x = self.blocks2(x)
            x = self.blocks3(x)
            x = self.blocks4(x)
            x = self.blocks5(x)
            x = self.blocks6(x)
            return paddle.nn.functional.avg_pool2d(x, [3, 2])

    class Tiny(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.backbone = DynamicTrainingBackbone(**backbone)
            self.head = MultiHead(in_channels=self.backbone.out_channels, **head)
        def forward(self, x, data=None):
            return self.head(self.backbone(x), targets=data)
    return Tiny()


def _numpy(value):
    return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)


def load_original(model, checkpoint=CHECKPOINT):
    import paddle
    state, expected = paddle.load(str(checkpoint)), model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Parameter keys differ: ' + str(sorted(set(state) ^ set(expected))))
    for key in state:
        if tuple(state[key].shape) != tuple(expected[key].shape):
            raise ValueError('Parameter shape differs: ' + key)
    model.set_state_dict(state)
    if any(not np.array_equal(_numpy(v), _numpy(state[k])) for k, v in model.state_dict().items()):
        raise ValueError('Checkpoint load changed parameters')
    return state


def expand_pretrained(model, original, old_characters, new_characters):
    initialized = model.state_dict()
    if set(original) != set(initialized):
        raise ValueError('Expanded model keys changed')
    state, audit, preserved = {}, [], 0
    for key, target in initialized.items():
        old = _numpy(original[key])
        if key in HEAD_AXES:
            head, axis = HEAD_AXES[key]
            old_tokens, new_tokens = tokens(old_characters, head), tokens(new_characters, head)
            if old.shape[axis] != len(old_tokens) or target.shape[axis] != len(new_tokens):
                raise ValueError('Actual checkpoint/token dimensions differ: ' + key)
            mapping = token_mapping(old_tokens, new_tokens)
            state[key] = remap_array(old, _numpy(target), mapping, axis)
            audit.append({'parameter': key, 'head': head, 'axis': axis,
                'old_shape': list(old.shape), 'new_shape': list(state[key].shape),
                'preserved_tokens': len(mapping), 'added_tokens': len(new_tokens) - len(old_tokens),
                'old_sha256': array_sha(old), 'expanded_sha256': array_sha(state[key]),
                'old_space_id': old_tokens.index(' '), 'new_space_id': new_tokens.index(' '), 'all_old_slices_equal': True})
        else:
            if old.shape != tuple(target.shape):
                raise ValueError('Non-output shape changed: ' + key)
            state[key], preserved = old, preserved + 1
    model.set_state_dict(state)
    if any(not np.array_equal(_numpy(v), state[k]) for k, v in model.state_dict().items()):
        raise ValueError('Expanded checkpoint load changed parameters')
    return {'matrices': audit, 'unchanged_non_output_parameters': preserved,
            'backbone_preserved': True, 'all_loaded_parameters_equal': True}


def encode_labels(texts, characters, maximum_length):
    ctc_map = {c: i for i, c in enumerate(tokens(characters)) if len(c) == 1}
    nrtr_map = {c: i for i, c in enumerate(tokens(characters, 'nrtr')) if len(c) == 1}
    ctc = np.zeros((len(texts), max(1, maximum_length)), dtype='int64')
    nrtr = np.zeros((len(texts), maximum_length + 2), dtype='int64')
    lengths = np.zeros(len(texts), dtype='int64')
    for n, text in enumerate(texts):
        if len(text) > maximum_length or any(c not in ctc_map for c in text):
            raise ValueError('Exact label cannot be encoded losslessly')
        lengths[n] = len(text)
        ctc[n, :len(text)] = [ctc_map[c] for c in text]
        nrtr[n, :len(text) + 2] = [2, *[nrtr_map[c] for c in text], 3]
    return ctc, nrtr, lengths


def decode(probabilities, characters):
    vocabulary = tokens(characters)
    if probabilities.ndim != 3 or probabilities.shape[-1] != len(vocabulary) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid exact CTC output')
    outputs = []
    for indices in probabilities.argmax(axis=-1):
        previous, text = None, []
        for index in indices:
            index = int(index)
            if index != 0 and index != previous:
                text.append(vocabulary[index])
            previous = index
        outputs.append(''.join(text))
    return outputs


def image_batch(root, rows, preprocessing=None):
    import cv2
    preprocessing = preprocessing or PREPROCESSING
    arrays, widths = [], []
    for row in rows:
        path = confined(root, row['image'])
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (row['height'], row['width']) or digest(path) != row['sha256']:
            raise ValueError('Synthetic image checksum/dimensions differ')
        width = math.ceil(48 * image.shape[1] / image.shape[0])
        arrays.append(cv2.resize(image, (width, 48), interpolation=cv2.INTER_LINEAR))
        widths.append(width)
    batch = np.zeros((len(rows), 3, 48, max(preprocessing['width']['min'], max(widths))), dtype='float32')
    for n, image in enumerate(arrays):
        batch[n, :, :, :widths[n]] = ((image.astype('float32') / 255 - .5) / .5).transpose(2, 0, 1)
    return batch


def _official_loss():
    losses = []
    for filename, classname in (('rec_ctc_loss.py', 'CTCLoss'), ('rec_nrtr_loss.py', 'NRTRLoss')):
        relative = 'ppocr/losses/' + filename
        if relative not in _SOURCE_FILES:
            raise ValueError('Official loss is absent from fixed source audit')
        loader = _FixedSourceLoader('localocr_' + classname, str(SOURCE / relative), _SOURCE_FILES[relative])
        spec = importlib.util.spec_from_file_location(loader.name, loader.path, loader=loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        losses.append(getattr(module, classname)())
    return losses


def exact_ctc_loss(logits, labels, lengths, texts, official_loss, empty_weight=1.0):
    """Exact per-sequence CTC; empty target has the sole all-blank path.

    The native Windows CTC kernel exits the process for an empty CPU target.
    Empty negatives use the mathematical CTC blank-only log likelihood instead;
    this remains differentiable and avoids routing zero length into that kernel.
    """
    import paddle
    if not math.isfinite(empty_weight) or not 0 < empty_weight <= 64:
        raise ValueError('Empty CTC weight must be finite in (0,64]')
    time_steps = logits.shape[1]
    positives = [i for i, text in enumerate(texts) if text]
    negatives = [i for i, text in enumerate(texts) if not text]
    totals = []
    if positives:
        index = paddle.to_tensor(positives, dtype='int64')
        positive_logits = paddle.index_select(logits, index, axis=0)
        positive_labels = paddle.index_select(labels, index, axis=0).astype('int32')
        positive_lengths = paddle.index_select(lengths, index, axis=0)
        input_lengths = paddle.to_tensor([time_steps] * len(positives), dtype='int64', place=paddle.CPUPlace())
        raw = official_loss.loss_func(positive_logits.transpose([1, 0, 2]), positive_labels, input_lengths, positive_lengths)
        totals.append((raw / positive_lengths.astype(raw.dtype)).sum())
    if negatives:
        index = paddle.to_tensor(negatives, dtype='int64')
        negative_logits = paddle.index_select(logits, index, axis=0)
        negative_log_probability = paddle.logsumexp(negative_logits, axis=-1) - negative_logits[:, :, 0]
        totals.append(empty_weight * negative_log_probability.mean(axis=1).sum())
    if not totals:
        raise ValueError('Empty training batch')
    return sum(totals) / len(texts)

def train_step(model, optimizer, loss_functions, images, texts, characters, maximum_length, trace=False, empty_ctc_weight=1.0):
    import paddle
    ctc, nrtr, lengths = encode_labels(texts, characters, maximum_length)
    batch = [paddle.to_tensor(images), paddle.to_tensor(ctc), paddle.to_tensor(nrtr),
             paddle.to_tensor(lengths), paddle.ones([len(texts)], dtype='float32')]
    if trace: print(json.dumps({'probe_phase': 'before_forward', 'rows': len(texts), 'empty': texts.count('')}), flush=True)
    predicts = model(batch[0], data=batch[1:])
    if trace: print(json.dumps({'probe_phase': 'forward_complete', 'ctc_shape': list(predicts['ctc'].shape)}), flush=True)
    if any(len(t) + sum(a == b for a, b in zip(t, t[1:])) > predicts['ctc'].shape[1] for t in texts):
        raise ValueError('CTC time cannot align exact training target')
    time_steps = predicts['ctc'].shape[1]
    if trace: print(json.dumps({'probe_phase': 'before_ctc_loss'}), flush=True)
    ctc_loss = exact_ctc_loss(predicts['ctc'], batch[1], batch[3], texts, loss_functions[0], empty_ctc_weight)
    if trace: print(json.dumps({'probe_phase': 'ctc_loss_complete', 'ctc_loss': float(ctc_loss)}), flush=True)
    nrtr_loss = loss_functions[1](predicts['gtc'], [batch[0], batch[2], batch[3], batch[4]])['loss']
    loss = ctc_loss + nrtr_loss
    values = [float(loss), float(ctc_loss), float(nrtr_loss)]
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError('Non-finite real CTC/NRTR loss')
    if trace: print(json.dumps({'probe_phase': 'before_backward', 'ctc_loss': values[1], 'nrtr_loss': values[2]}), flush=True)
    loss.backward()
    if trace: print(json.dumps({'probe_phase': 'backward_complete'}), flush=True)
    gradients = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not bool(paddle.isfinite(parameter.grad).all()):
                raise RuntimeError('Non-finite gradient')
            gradients += 1
    if not gradients:
        raise RuntimeError('No real backward gradients')
    optimizer.step()
    optimizer.clear_grad()
    return {'loss': values[0], 'ctc_loss': values[1], 'nrtr_loss': values[2],
            'gradient_parameters': gradients, 'rows': len(texts), 'empty_labels': texts.count(''),
            'ctc_time_steps': int(time_steps), 'ctc_normalization': 'positive/reference-codepoints; empty-negative/output-time-steps',
            'empty_ctc_weight': empty_ctc_weight}


def configure_finetune(model, freeze_backbone=False):
    """Freeze the existing visual parameters and its running statistics together."""
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.stop_gradient = True
    set_training_mode(model, freeze_backbone)
    return {'freeze_backbone': bool(freeze_backbone),
            'backbone_parameters': len(list(model.backbone.parameters())),
            'trainable_parameters': sum(not parameter.stop_gradient for parameter in model.parameters()),
            'backbone_statistics': 'frozen eval mode' if freeze_backbone else 'original training mode'}


def set_training_mode(model, freeze_backbone=False):
    model.train()
    if freeze_backbone:
        model.backbone.eval()


def backbone_sha256(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.backbone.state_dict().items()):
        value.update(name.encode('utf-8'))
        value.update(array_sha(_numpy(tensor)).encode('ascii'))
    return value.hexdigest()


def prepare(dataset, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Prepare output must be fresh')
    disk_guard(output)
    audit = source_audit()
    paddle = _paddle('cpu')
    paddle.seed(20261011)
    np.random.seed(20261011)
    old = DICTIONARY.read_text(encoding='utf-8').splitlines()
    new = extended_characters(old, [c for c in train_characters(dataset) if c != ' '])
    rows = read_split(dataset, 'train')
    maximum = max(len(r['text']) for r in rows)
    required = max(len(r['text']) + sum(a == b for a, b in zip(r['text'], r['text'][1:])) for r in rows)
    preprocessing = copy.deepcopy(PREPROCESSING)
    preprocessing['width']['min'] = max(320, 8 * (required + 1))
    encode_labels([r['text'] for r in rows], new, maximum)
    original = build_model(old, maximum)
    state = load_original(original)
    del original
    expanded = build_model(new, maximum)
    mapping_audit = expand_pretrained(expanded, state, old, new)
    output.mkdir(parents=True)
    write(output / 'source-audit.json', audit)
    for filename, characters in (('vocabulary.json', new), ('original-vocabulary.json', old)):
        write(output / filename, {'schema_version': 'localocr.pretrained.vocabulary.v1',
              'tokens': tokens(characters), 'blank_id': 0, 'space_id': len(characters) + 1})
    (output / 'dictionary.txt').write_text('\n'.join(new) + '\n', encoding='utf-8', newline='\n')
    paddle.save(expanded.state_dict(), str(output / 'initialized.pdparams'))
    write(output / 'initialization.json', {'schema_version': 'localocr.paddle.initialization.v1',
        'source_commit': COMMIT, 'source_checkpoint_sha256': CHECKPOINT_SHA,
        'dataset_root': str(Path(dataset).resolve()), 'dataset': dataset_binding(dataset),
        'maximum_label_length': maximum, 'maximum_required_ctc_steps': required,
        'train_rows': len(rows), 'negative_rows': sum(not r['text'] for r in rows),
        'old_characters': len(old), 'new_characters': len(new), 'added_characters': len(new) - len(old),
        'mapping_audit': mapping_audit, 'preprocessing': preprocessing,
        'training_pool_change': {'original': 'adaptive_avg_pool2d[1,40]', 'adapter': 'avg_pool2d[3,2]',
                                 'same_as_official_inference': True, 'parameter_changes': 0},
        'seed': 20261011, 'numpy_seed': 20261011,
        'checkpoint': {'file': 'initialized.pdparams', 'sha256': digest(output / 'initialized.pdparams')},
        'vocabulary_sha256': digest(output / 'vocabulary.json'), 'dictionary_sha256': digest(output / 'dictionary.txt'),
        'created_utc': utc(), 'adapter_sha256': digest(__file__)})
    print(json.dumps({'prepared': str(output), 'added_characters': len(new) - len(old),
                      'maximum_label_length': maximum, 'minimum_width': preprocessing['width']['min']}), flush=True)


def _load_prepared(prepared, checkpoint=None, device='cpu', owned=None):
    prepared = Path(prepared).resolve()
    info = read(prepared / 'initialization.json')
    if digest(prepared / 'vocabulary.json') != info['vocabulary_sha256'] or digest(prepared / 'dictionary.txt') != info['dictionary_sha256']:
        raise ValueError('Prepared vocabulary bytes changed')
    if digest(confined(prepared, info['checkpoint']['file'])) != info['checkpoint']['sha256']:
        raise ValueError('Prepared initialization checkpoint bytes changed')
    characters = read(prepared / 'vocabulary.json')['tokens'][1:-1]
    _paddle(device)
    model = build_model(characters, info['maximum_label_length'])
    if owned is not None:
        _own(owned, 'models', model)
    checkpoint = Path(checkpoint) if checkpoint else prepared / 'initialized.pdparams'
    load_original(model, checkpoint)
    return model, characters, info


def _device_receipt():
    import paddle
    return {'paddle': paddle.__version__, 'compiled_cuda': paddle.version.cuda(),
            'device': paddle.get_device(), 'name': paddle.device.cuda.get_device_name(),
            'visible_devices': paddle.device.cuda.device_count()}


def _own(owned, category, value):
    owned[category].append(value)
    owned['references'].append(weakref.ref(value))
    return value


def _move_owned_to_cpu(owned):
    # Return before collecting so the loop's last optimizer/model cannot retain
    # a GPU tensor after the ownership containers have been emptied.
    for optimizer in owned['optimizers']:
        optimizer.clear_grad()
    for model in owned['models']:
        model.clear_gradients()
        model.to(device='cpu')
    owned['optimizers'].clear()
    owned['models'].clear()


def _clear_exception_frames(error):
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        pending.extend(getattr(current, 'exceptions', ()))


def _owned_execution(body, dispose, failure=None):
    """Unwind failed model/loss frames before releasing owned accelerator state."""
    owned = {'models': [], 'optimizers': [], 'references': []}
    try:
        return body(owned)
    except BaseException as error:
        try:
            if failure is not None:
                failure(error)
        finally:
            # A propagated traceback otherwise retains predictions, gradients,
            # optimizer accumulators and batch tensors until after lease exit.
            _clear_exception_frames(error)
        raise
    finally:
        dispose(owned)


def _gpu_execution(label, seconds, receipt, body, budget=None):
    receipt = Path(receipt)
    with GPUBudget(BUDGET if budget is None else budget).run(label, seconds) as deadline:
        paddle = _paddle('gpu:0')
        allocated_before = paddle.device.cuda.memory_allocated()
        def failure(error):
            report = {'schema_version': 'localocr.paddle.failure.v1', 'utc': utc(),
                      'exception': type(error).__name__, 'message': str(error),
                      'remaining_seconds': deadline.remaining(), 'adapter_sha256': digest(__file__)}
            print(json.dumps({'gpu_failure': report}), flush=True)
            write(receipt.with_suffix('.failure.json'), report)
        def dispose(owned):
            try:
                _move_owned_to_cpu(owned)
                gc.collect()
                paddle.device.synchronize()
                paddle.device.cuda.empty_cache()
                allocated_after = paddle.device.cuda.memory_allocated()
                live = sum(reference() is not None for reference in owned['references'])
                report = {'schema_version': 'localocr.paddle.cleanup.v1', 'utc': utc(),
                          'inside_gpu_lease': True, 'allocated_before_bytes': allocated_before,
                          'allocated_after_bytes': allocated_after, 'owned_objects_still_alive': live,
                          'release_confirmed': live == 0 and allocated_after <= allocated_before,
                          'remaining_seconds': deadline.remaining()}
                print(json.dumps({'gpu_cleanup': report}), flush=True)
                write(receipt.with_suffix('.cleanup.json'), report)
                if not report['release_confirmed']:
                    raise RuntimeError('Owned GPU state was not fully released')
            except BaseException as error:
                # Fail closed: never release the persistent lease while GPU
                # ownership is uncertain. Process exit preserves active lease
                # evidence for the parent's explicit dead-process audit.
                try:
                    print(json.dumps({'gpu_cleanup_failed': type(error).__name__,
                                      'message': str(error), 'utc': utc(), 'exit_code': 70}), flush=True)
                finally:
                    os._exit(70)
        return _owned_execution(lambda owned: body(deadline, owned), dispose, failure)


def probe(prepared, dataset, output, seconds=120, batch_size=8, minimum_width=None, learning_rate=.0001, freeze_backbone=False, empty_ctc_weight=1.0, warm_start=None, warm_start_receipt_sha256=None, budget=None):
    output = Path(output)
    if any(path.exists() for path in (output, output.with_suffix('.pdparams'),
                                      output.with_suffix('.failure.json'), output.with_suffix('.cleanup.json'))):
        raise ValueError('Probe output must be fresh, including failure/cleanup receipts')
    if batch_size < 1 or (minimum_width is not None and minimum_width < 1):
        raise ValueError('Probe batch/width must be positive')
    if warm_start and not warm_start_receipt_sha256:
        raise ValueError('Warm probe requires the externally frozen parent training receipt SHA')
    parent = _training_checkpoint(warm_start, prepared, warm_start_receipt_sha256) if warm_start else None
    source_audit()
    binding = dataset_binding(dataset)
    rows = read_split(dataset, 'train')
    width = lambda row: math.ceil(48 * row['width'] / row['height'])
    positive_rows = [r for r in rows if r['text']]
    required = lambda row: len(row['text']) + sum(a == b for a, b in zip(row['text'], row['text'][1:]))
    extremes = ([max(positive_rows, key=width), max(positive_rows, key=lambda row: (len(row['text']), width(row))),
                 max(positive_rows, key=lambda row: (required(row), width(row)))] if positive_rows else [])
    selected, identities = [], set()
    for row in extremes + sorted(positive_rows, key=width, reverse=True):
        if row['image'] not in identities:
            identities.add(row['image'])
            selected.append(row)
        if len(selected) == batch_size:
            break
    positives = selected
    negatives = sorted((r for r in rows if not r['text']), key=width, reverse=True)[:batch_size]
    if not positives or not negatives:
        raise ValueError('Actual positive and negative training samples required')
    def body(deadline, owned):
        import paddle
        paddle.seed(20261011)
        np.random.seed(20261011)
        model, characters, info = _load_prepared(prepared, checkpoint=warm_start, device='gpu:0', owned=owned)
        if info['dataset'] != binding:
            raise ValueError('Prepared dataset binding differs')
        policy = configure_finetune(model, freeze_backbone)
        frozen_backbone = backbone_sha256(model) if freeze_backbone else None
        optimizer = _own(owned, 'optimizers', paddle.optimizer.Adam(
            learning_rate=learning_rate, parameters=[p for p in model.parameters() if not p.stop_gradient], grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0)))
        loss = _official_loss()
        before, outcomes = array_sha(_numpy(model.head.ctc_head.fc2.weight)), []
        preprocessing = copy.deepcopy(info['preprocessing'])
        if minimum_width is not None:
            preprocessing['width']['min'] = max(preprocessing['width']['min'], minimum_width)
        if hasattr(paddle.device.cuda, 'reset_max_memory_allocated'):
            paddle.device.cuda.reset_max_memory_allocated()
        for label, batch in (('positive', positives), ('empty-negative', negatives)):
            if deadline.expired():
                raise TimeoutError('GPU probe deadline reached')
            started = time.monotonic()
            images = image_batch(dataset, batch, preprocessing)
            outcome = train_step(model, optimizer, loss, images, [r['text'] for r in batch],
                                 characters, info['maximum_label_length'], trace=True, empty_ctc_weight=empty_ctc_weight)
            outcome.update({'kind': label, 'seconds': time.monotonic() - started,
                            'input_shape': list(images.shape), 'maximum_natural_width': max(map(width, batch)),
                            'maximum_label_codepoints': max(len(r['text']) for r in batch),
                            'maximum_required_ctc_steps': max(map(required, batch))})
            outcomes.append(outcome)
        after = array_sha(_numpy(model.head.ctc_head.fc2.weight))
        if before == after:
            raise RuntimeError('Optimizer did not update actual weights')
        if freeze_backbone and backbone_sha256(model) != frozen_backbone:
            raise RuntimeError('Frozen probe backbone parameters/statistics changed')
        disk_guard(output)
        checkpoint = output.with_suffix('.pdparams')
        paddle.save(model.state_dict(), str(checkpoint))
        restored = _own(owned, 'models', build_model(characters, info['maximum_label_length']))
        load_original(restored, checkpoint)
        if array_sha(_numpy(restored.head.ctc_head.fc2.weight)) != after:
            raise RuntimeError('Checkpoint roundtrip differs')
        report = {'schema_version': 'localocr.paddle.probe.v1', 'device': _device_receipt(),
            'steps': outcomes, 'before_weight_sha256': before, 'after_weight_sha256': after,
            'actual_weights_changed': True, 'checkpoint_roundtrip': True,
            'finetune_policy': policy, 'empty_ctc_weight': empty_ctc_weight, 'learning_rate': learning_rate,
            'frozen_backbone_sha256': frozen_backbone, 'frozen_backbone_unchanged': freeze_backbone and backbone_sha256(model) == frozen_backbone,
            'warm_start': parent,
            'checkpoint_sha256': digest(checkpoint), 'created_utc': utc(),
            'remaining_seconds': deadline.remaining(), 'batch_size': len(positives),
            'selection': 'positive max-width/max-label/max-CTC extremes then widest; negative widest; train only',
            'probe_minimum_width': preprocessing['width']['min'],
            'maximum_training_natural_width': max(map(width, rows)),
            'gpu_peak_allocated_bytes': paddle.device.cuda.max_memory_allocated(),
            'rows_per_second': sum(s['rows'] for s in outcomes) / sum(s['seconds'] for s in outcomes),
            'common_sha256': digest(Path(__file__).with_name('common.py')),
            'dataset': binding, 'preparation': {'file': str(Path(prepared).resolve() / 'initialization.json'),
                'sha256': digest(Path(prepared) / 'initialization.json')}, 'adapter_sha256': digest(__file__),
            'loss_normalization': 'CTC positive/reference-codepoints; empty-negative/output-time-steps; NRTR official nonpad token mean'}
        write(output, report)
        return report
    _gpu_execution('paddle-tiny-positive-negative-probe', seconds, output, body, budget=budget)
    print(json.dumps({'probe': str(output), 'actual_weights_changed': True}), flush=True)


def _edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def native_evaluate(model, characters, root, rows, preprocessing, batch_size=16, deadline=None, quality=False, quality_reference=None, return_predictions=False):
    import paddle
    model.eval()
    correct, edits, codepoints, negatives, false_outputs = 0, 0, 0, 0, 0
    by_kind, width_groups, raw_predictions = {}, {}, {}
    for row in rows:
        width = max(preprocessing['width']['min'], math.ceil(48 * row['width'] / row['height']))
        width_groups.setdefault(width, []).append(row)
    # Equal-width batches retain exactly the single-crop evaluator padding.
    batches = [group[offset:offset + batch_size] for group in width_groups.values()
               for offset in range(0, len(group), batch_size)]
    with paddle.no_grad():
        for batch in batches:
            if deadline is not None and deadline.expired():
                raise TimeoutError('GPU deadline reached in dev evaluation')
            predictions = decode(_numpy(model(paddle.to_tensor(image_batch(root, batch, preprocessing)))), characters)
            for row, prediction in zip(batch, predictions):
                if quality: raw_predictions[row['id']] = {'id': row['id'], 'text': prediction}
                exact = int(prediction == row['text'])
                distance = _edit_distance(row['text'], prediction)
                correct += exact
                edits += distance
                codepoints += len(row['text'])
                negatives += not row['text']
                false_outputs += not row['text'] and bool(prediction)
                item = by_kind.setdefault(row['kind'], {'rows': 0, 'exact_correct': 0,
                    'edit_distance': 0, 'reference_codepoints': 0, 'negative_rows': 0, 'negative_nonempty': 0})
                item['rows'] += 1
                item['exact_correct'] += exact
                item['edit_distance'] += distance
                item['reference_codepoints'] += len(row['text'])
                item['negative_rows'] += not row['text']
                item['negative_nonempty'] += not row['text'] and bool(prediction)
    for item in by_kind.values():
        item['exact_accuracy'] = item['exact_correct'] / item['rows']
        item['cer'] = item['edit_distance'] / item['reference_codepoints'] if item['reference_codepoints'] else None
        item['negative_nonempty_rate'] = item['negative_nonempty'] / item['negative_rows'] if item['negative_rows'] else None
    result = {'rows': len(rows), 'exact_correct': correct, 'exact_accuracy': correct / len(rows),
            'edit_distance': edits, 'reference_codepoints': codepoints,
            'cer': edits / codepoints if codepoints else None, 'ignore_space': False,
            'negative_rows': negatives, 'negative_nonempty': false_outputs,
            'negative_nonempty_rate': false_outputs / negatives if negatives else None,
            'by_kind': by_kind, 'batching': 'exact-width groups; same preprocessing as single crop'}
    if quality:
        ordered = [raw_predictions[row['id']] for row in rows]
        result['quality'] = quality_selection.summarize(rows, ordered, reference=quality_reference)
        if return_predictions: result['quality_predictions'] = ordered
    return result

def train(prepared, dataset, output, seconds=2580, batch_size=16, learning_rate=.0001, seed=20261011, warm_start=None, freeze_backbone=False, empty_ctc_weight=1.0, warm_start_receipt_sha256=None, selection_policy='legacy', budget=None, eval_every_seconds=180.0):
    if selection_policy not in ('legacy', 'face-pareto'):
        raise ValueError('Unknown development selection policy')
    if isinstance(eval_every_seconds, bool) or not math.isfinite(eval_every_seconds) or eval_every_seconds <= 0:
        raise ValueError('Development evaluation frequency must be positive and finite')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Training output must be fresh; restart does not reset budget')
    if warm_start and not warm_start_receipt_sha256:
        raise ValueError('Warm start requires the externally frozen parent training receipt SHA')
    disk_guard(output)
    audit = source_audit()
    train_rows, dev_rows = read_split(dataset, 'train'), read_split(dataset, 'dev')
    binding = dataset_binding(dataset)
    prepared = Path(prepared).resolve()
    initialization = read(prepared / 'initialization.json')
    coverage = quality_selection.original_class_coverage(train_rows, read(prepared / 'original-vocabulary.json')['tokens']) if selection_policy == 'face-pareto' else None
    if initialization['dataset'] != binding:
        raise ValueError('Prepared dataset binding differs')
    start = _training_checkpoint(warm_start, prepared, warm_start_receipt_sha256) if warm_start else None
    output.mkdir(parents=True)
    write(output / 'source-audit.json', audit)
    random.seed(seed)
    np.random.seed(seed)
    def body(deadline, owned):
        history, best, steps, epochs, epochs_completed = [], None, 0, 0, 0
        seen, negative_identities, exposure = set(), set(), {}
        total_rows = 0
        selected_exposure = None
        import paddle
        paddle.seed(seed)
        model, characters, info = _load_prepared(prepared, checkpoint=warm_start, device='gpu:0', owned=owned)
        if info['dataset'] != binding:
            raise ValueError('Prepared dataset binding differs')
        policy = configure_finetune(model, freeze_backbone)
        frozen_backbone = backbone_sha256(model) if freeze_backbone else None
        optimizer = _own(owned, 'optimizers', paddle.optimizer.Adam(learning_rate=learning_rate,
                        parameters=[p for p in model.parameters() if not p.stop_gradient],
                        grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0)))
        loss = _official_loss()
        initial = array_sha(_numpy(model.head.ctc_head.fc2.weight))
        specialist = selection_policy == 'face-pareto'
        before = native_evaluate(model, characters, dataset, dev_rows, info['preprocessing'], batch_size, deadline, quality=specialist, return_predictions=specialist)
        quality_reference = before.pop('quality_predictions', None)
        if specialist: write(output / 'quality-initial-predictions.json', quality_reference)
        write(output / ('warm-start-dev.json' if warm_start else 'expanded-initial-dev.json'), before)
        best = before if warm_start or specialist else None
        selected_weight_sha = None
        last_eval, losses, negative_seen = time.monotonic(), [], 0
        while deadline.remaining() > 45:
            order = list(range(len(train_rows)))
            random.shuffle(order)
            epochs += 1
            set_training_mode(model, freeze_backbone)
            for offset in range(0, len(order), batch_size):
                if deadline.remaining() < 45:
                    break
                batch = [train_rows[i] for i in order[offset:offset + batch_size]]
                outcome = train_step(model, optimizer, loss, image_batch(dataset, batch, info['preprocessing']),
                                     [r['text'] for r in batch], characters, info['maximum_label_length'],
                                     empty_ctc_weight=empty_ctc_weight)
                losses.append(outcome['loss'])
                negative_seen += outcome['empty_labels']
                total_rows += outcome['rows']
                for row in batch:
                    seen.add(row['image'])
                    if not row['text']:
                        negative_identities.add(row['image'])
                    exposure[row['kind']] = exposure.get(row['kind'], 0) + 1
                steps += 1
                if steps % 50 == 0:
                    disk_guard(output)
                    print(json.dumps({'step': steps, 'epoch': epochs, 'loss': float(np.mean(losses[-50:])),
                                      'negative_rows_trained': negative_seen, 'remaining_seconds': deadline.remaining()}), flush=True)
                if time.monotonic() - last_eval >= eval_every_seconds and deadline.remaining() > 90:
                    metric = native_evaluate(model, characters, dataset, dev_rows, info['preprocessing'], batch_size, deadline, quality=specialist, quality_reference=quality_reference)
                    record = {'step': steps, 'epoch': epochs, 'mean_loss': float(np.mean(losses)),
                              'rows_trained': total_rows, 'distinct_train_rows_seen': len(seen),
                              'negative_rows_trained': negative_seen, 'distinct_negative_rows_seen': len(negative_identities),
                              'dev': metric, 'remaining_seconds': deadline.remaining(), 'utc': utc()}
                    history.append(record)
                    print(json.dumps({'dev': record}), flush=True)
                    if development_better(metric, best, selection_policy):
                        best = metric
                        selected_weight_sha = array_sha(_numpy(model.head.ctc_head.fc2.weight))
                        selected_exposure = {'step': steps, 'rows_trained': total_rows, 'distinct_train_rows_seen': len(seen),
                                             'negative_rows_trained': negative_seen, 'distinct_negative_rows_seen': len(negative_identities),
                                             'rows_by_kind': dict(exposure)}
                        disk_guard(output)
                        paddle.save(model.state_dict(), str(output / 'best.pdparams'))
                        write(output / 'best-dev.json', record)
                    write(output / 'history.json', history)
                    last_eval = time.monotonic()
                    set_training_mode(model, freeze_backbone)
            else:
                epochs_completed += 1
            if deadline.remaining() < 45:
                break
        changed = array_sha(_numpy(model.head.ctc_head.fc2.weight)) != initial
        if not steps or not changed or not negative_seen:
            raise RuntimeError('No effective successful updates or negative exposure')
        selected_changed = selected_weight_sha is not None and selected_weight_sha != initial
        if not warm_start and not selected_changed and selection_policy == 'legacy':
            raise RuntimeError('No effective selected first checkpoint')
        if freeze_backbone and backbone_sha256(model) != frozen_backbone:
            raise RuntimeError('Frozen backbone parameters/statistics changed')
        disk_guard(output)
        paddle.save(model.state_dict(), str(output / 'last.pdparams'))
        write(output / 'training.json', {'schema_version': 'localocr.paddle.training.v1', 'device': _device_receipt(),
            'dataset': binding, 'train_rows': len(train_rows), 'dev_rows': len(dev_rows),
            'preparation': {'file': str(prepared / 'initialization.json'), 'sha256': digest(prepared / 'initialization.json')},
            'initialized_checkpoint': info['checkpoint'], 'start_checkpoint': start or info['checkpoint'],
            'warm_start': bool(warm_start), 'optimizer_state_restored': False,
            'optimizer_reset': 'Fresh Adam parameters/accumulators; checkpoint parameters only',
            'negative_train_rows': sum(not r['text'] for r in train_rows), 'negative_rows_trained': negative_seen,
            'maximum_label_length': info['maximum_label_length'], 'training_pool_change': info['training_pool_change'],
            'batch_size': batch_size, 'learning_rate': learning_rate, 'optimizer': 'Adam', 'seed': seed,
            'finetune_policy': policy, 'empty_ctc_weight': empty_ctc_weight,
            'frozen_backbone_sha256': frozen_backbone,
            'frozen_backbone_unchanged': freeze_backbone and backbone_sha256(model) == frozen_backbone,
            'requested_seconds': seconds, 'rows_trained': total_rows, 'distinct_train_rows_seen': len(seen),
            'distinct_negative_rows_seen': len(negative_identities), 'rows_by_kind': exposure,
            'selected_training_exposure': selected_exposure, 'epochs_completed': epochs_completed,
            'common_sha256': digest(Path(__file__).with_name('common.py')),
            'source_audit': {'file': 'source-audit.json', 'sha256': digest(output / 'source-audit.json')},
            'loss_normalization': 'CTC positive/reference-codepoints; empty-negative/output-time-steps; NRTR official nonpad token mean',
            'epochs_started': epochs, 'steps': steps, 'initial_dev': before, 'best_dev': best, 'history': history,
            'actual_weights_changed': changed, 'selected_weights_changed': selected_changed,
            'selection_outcome': 'new-effective-best' if selected_changed else ('parent-retained-no-development-improvement' if warm_start else 'initial-retained-no-specialist-improvement'),
            'best_checkpoint': {'file': 'best.pdparams', 'sha256': digest(output / 'best.pdparams')} if selected_changed else None,
            'last_checkpoint': {'file': 'last.pdparams', 'sha256': digest(output / 'last.pdparams')},
            'selection': 'face exact/CER/P-S Pareto and nonincreasing negative false outputs; heldout unread' if specialist else 'dev exact_correct descending, edit_distance ascending; heldout unread',
            'selection_policy': selection_policy, 'original_class_coverage': coverage,
            'quality_selection_sha256': digest(Path(quality_selection.__file__)) if specialist else None,
            'budget': str(Path(BUDGET if budget is None else budget).resolve()), 'eval_every_seconds': eval_every_seconds,
            'remaining_seconds': deadline.remaining(), 'created_utc': utc(), 'adapter_sha256': digest(__file__)})
        return {'steps': steps, 'best_dev': best, 'new_candidate_selected': selected_changed}
    result = _gpu_execution('paddle-tiny-finetune', seconds, output / 'training.json', body, budget=budget)
    print(json.dumps({'training': str(output), **result}), flush=True)


def _reject_reparse(path, root):
    """Reject symbolic links and Windows reparse points before dereferencing."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if '..' in path.parts or not path.is_relative_to(root) or path == root:
        raise ValueError('Training evidence escapes the owned research root')
    # An ancestor junction can relocate the whole owned root before any
    # checkpoint-relative segment is reached. Inspect its native path too.
    for ancestor in reversed((root, *root.parents)):
        properties = ancestor.lstat()
        if stat.S_ISLNK(properties.st_mode) or getattr(properties, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Training evidence root contains a reparse point or symbolic link')
    cursor = root
    for segment in (None, *path.relative_to(root).parts):
        if segment is not None:
            cursor = cursor / segment
        properties = cursor.lstat()
        if stat.S_ISLNK(properties.st_mode) or getattr(properties, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Training evidence contains a reparse point or symbolic link')


def _closure_file(reference, owner, root):
    if not isinstance(reference, dict) or not isinstance(reference.get('file'), str):
        raise ValueError('Missing provenance file reference')
    target = Path(reference['file'])
    target = target if target.is_absolute() else owner / target
    _reject_reparse(target, root)
    target = target.resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError('Training provenance escapes owned research root')
    expected = reference.get('sha256')
    if not isinstance(expected, str) or len(expected) != 64 or digest(target) != expected:
        raise ValueError('Training provenance file bytes differ')
    return target


def _training_checkpoint(checkpoint, prepared, expected_receipt_sha256=None, _seen=None):
    """Validate the externally anchored selected checkpoint's complete ancestry.

    A caller-provided receipt SHA is mandatory for a warm start. It is never
    computed from the sibling receipt's claims; hashes bind observed local
    evidence, rather than authenticating an unknown publisher.
    """
    _reject_reparse(checkpoint, ROUND / 'paddle')
    _reject_reparse(Path(prepared) / 'initialization.json', ROUND / 'paddle')
    checkpoint, prepared = Path(checkpoint).resolve(), Path(prepared).resolve()
    receipt = checkpoint.parent / 'training.json'
    _reject_reparse(receipt, ROUND / 'paddle')
    seen = set() if _seen is None else _seen
    if receipt in seen or len(seen) >= 8:
        raise ValueError('Training parent chain is cyclic or too deep')
    seen.add(receipt)
    if expected_receipt_sha256 is not None and digest(receipt) != expected_receipt_sha256:
        raise ValueError('Externally anchored training receipt bytes differ')
    training, initialization = read(receipt), read(prepared / 'initialization.json')
    if training.get('schema_version') != 'localocr.paddle.training.v1' or training.get('selected_weights_changed') is not True:
        raise ValueError('Finetuned checkpoint lacks a selected changed training receipt')
    if training.get('actual_weights_changed') is not True or training.get('steps', 0) <= 0:
        raise ValueError('Finetuned receipt lacks real successful updates')
    selected = training.get('best_checkpoint')
    selected_path = _closure_file(selected, receipt.parent, ROUND / 'paddle')
    if checkpoint != selected_path:
        raise ValueError('Checkpoint is not the frozen selected best')
    prepared_manifest = _closure_file(training.get('preparation'), receipt.parent, ROUND / 'paddle')
    if prepared_manifest != prepared / 'initialization.json':
        raise ValueError('Training preparation identity differs')
    if initialization.get('source_commit') != COMMIT or initialization.get('source_checkpoint_sha256') != CHECKPOINT_SHA:
        raise ValueError('Initialization is outside the fixed official source/checkpoint')
    init_path = _closure_file(initialization.get('checkpoint'), prepared, ROUND / 'paddle')
    if training.get('initialized_checkpoint') != initialization['checkpoint']:
        raise ValueError('Training initialization weights differ')
    if training.get('dataset') != initialization['dataset'] or dataset_binding(initialization['dataset_root']) != initialization['dataset']:
        raise ValueError('Training checkpoint dataset differs')
    _reject_reparse(prepared / 'vocabulary.json', ROUND / 'paddle')
    _reject_reparse(prepared / 'dictionary.txt', ROUND / 'paddle')
    if digest(prepared / 'vocabulary.json') != initialization['vocabulary_sha256'] or digest(prepared / 'dictionary.txt') != initialization['dictionary_sha256']:
        raise ValueError('Training checkpoint vocabulary differs')
    _reject_reparse(receipt.parent / 'best-dev.json', ROUND / 'paddle')
    best_record = read(receipt.parent / 'best-dev.json')
    exposure = training.get('selected_training_exposure', {})
    if exposure.get('step', 0) <= 0 or exposure.get('step') != best_record.get('step') or best_record.get('dev') != training.get('best_dev'):
        raise ValueError('Selected checkpoint lacks effective development selection')
    audit_path = _closure_file(training.get('source_audit'), receipt.parent, ROUND / 'paddle')
    audit, fresh = read(audit_path), source_audit()
    for key in ('commit', 'archive_sha256', 'checkpoint_sha256', 'source_inventory'):
        if audit.get(key) != fresh.get(key):
            raise ValueError('Training source audit differs from fixed official inventory')
    snapshot = receipt.parent.with_name(receipt.parent.name + '-source')
    _reject_reparse(snapshot / 'paddle_adapter.py', ROUND / 'paddle')
    _reject_reparse(snapshot / 'common.py', ROUND / 'paddle')
    if digest(snapshot / 'paddle_adapter.py') != training.get('adapter_sha256') or digest(snapshot / 'common.py') != training.get('common_sha256'):
        raise ValueError('Training code differs from preserved source snapshot')
    if training.get('selection_policy') == 'face-pareto':
        _reject_reparse(snapshot / 'quality_selection.py', ROUND / 'paddle')
        if digest(snapshot / 'quality_selection.py') != training.get('quality_selection_sha256') or digest(Path(quality_selection.__file__)) != training.get('quality_selection_sha256'):
            raise ValueError('Specialist selection code differs from the preserved snapshot')
        if not quality_selection.face_better(training['best_dev']['quality'], training['initial_dev']['quality']):
            raise ValueError('Specialist selected checkpoint did not improve its actual initialization')
    ancestry = []
    if training.get('warm_start'):
        if expected_receipt_sha256 is None:
            raise ValueError('Warm selected receipt requires an external frozen SHA anchor')
        start = training.get('start_checkpoint', {})
        parent = start.get('training_receipt')
        parent_receipt = _closure_file(parent, receipt.parent, ROUND / 'paddle')
        parent_checkpoint = _closure_file(start, receipt.parent, ROUND / 'paddle')
        if parent_receipt != parent_checkpoint.parent / 'training.json':
            raise ValueError('Parent checkpoint and receipt identity differ')
        verified = _training_checkpoint(parent_checkpoint, prepared, parent['sha256'], seen)
        ancestry = [verified, *verified.get('parents', [])]
        # A new research identity requires a strict development improvement.
        parent_metric = read(parent_receipt)['best_dev']
        current_metric = training['best_dev']
        if training.get('selection_policy') != 'face-pareto' and (current_metric['exact_correct'], -current_metric['edit_distance']) <= (parent_metric['exact_correct'], -parent_metric['edit_distance']):
            raise ValueError('Warm selection did not improve its parent development baseline')
    elif training.get('start_checkpoint') != initialization['checkpoint']:
        raise ValueError('First training did not start from the bound expanded initialization')
    return {'file': str(checkpoint), 'sha256': digest(checkpoint),
            'training_receipt': {'file': str(receipt), 'sha256': digest(receipt)},
            'initialization': {'file': str(prepared_manifest), 'sha256': digest(prepared_manifest)},
            'initialized_weights': {'file': str(init_path), 'sha256': digest(init_path)},
            'source_audit': {'file': str(audit_path), 'sha256': digest(audit_path)},
            'best_dev': {'file': str(receipt.parent / 'best-dev.json'), 'sha256': digest(receipt.parent / 'best-dev.json')},
            'vocabulary': {'file': str(prepared / 'vocabulary.json'), 'sha256': digest(prepared / 'vocabulary.json')},
            'dictionary': {'file': str(prepared / 'dictionary.txt'), 'sha256': digest(prepared / 'dictionary.txt')},
            'source_snapshot': {'adapter': {'file': str(snapshot / 'paddle_adapter.py'), 'sha256': digest(snapshot / 'paddle_adapter.py')},
                                'common': {'file': str(snapshot / 'common.py'), 'sha256': digest(snapshot / 'common.py')},
                                'quality_selection': {'file': str(snapshot / 'quality_selection.py'), 'sha256': digest(snapshot / 'quality_selection.py')} if training.get('selection_policy') == 'face-pareto' else None},
            'parents': ancestry}


def _snapshot_closure(value, output):
    """Copy every already-validated effective parent dependency into this candidate."""
    if isinstance(value, list):
        return [_snapshot_closure(item, output) for item in value]
    if not isinstance(value, dict):
        return value
    copied = {key: _snapshot_closure(item, output) for key, item in value.items()}
    if 'file' in value and 'sha256' in value:
        _reject_reparse(value['file'], ROUND / 'paddle')
        source = Path(value['file']).resolve()
        if digest(source) != value['sha256']:
            raise ValueError('Validated provenance changed before snapshot')
        directory = output / 'provenance'
        directory.mkdir(exist_ok=True)
        target = directory / (value['sha256'] + '-' + source.name)
        disk_guard(target)
        if not target.exists():
            shutil.copyfile(source, target)
        if digest(target) != value['sha256']:
            raise ValueError('Owned provenance snapshot bytes differ')
        copied['file'] = target.relative_to(output).as_posix()
    return copied


def export(prepared, output, checkpoint=None, original=False, identifier=None, native_reference=None, training_receipt_sha256=None):
    import paddle
    if paddle.__version__ != '3.1.1':
        raise RuntimeError('Use isolated CPU Paddle 3.1.1 export environment for Paddle2ONNX 2.1.0 ABI')
    import paddle2onnx
    import onnx
    import onnxruntime as ort
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Export output must be fresh')
    disk_guard(output)
    audit = source_audit()
    _paddle('cpu')
    prepared = Path(prepared).resolve()
    info = read(prepared / 'initialization.json')
    training_source = _training_checkpoint(checkpoint, prepared, training_receipt_sha256) if checkpoint and not original else None
    if original:
        characters = DICTIONARY.read_text(encoding='utf-8').splitlines()
        model = build_model(characters, info['maximum_label_length'])
        load_original(model)
        actual_checkpoint, stage = CHECKPOINT, 'official-pretrained'
    else:
        model, characters, info = _load_prepared(prepared, checkpoint, 'cpu')
        actual_checkpoint = Path(checkpoint).resolve() if checkpoint else prepared / 'initialized.pdparams'
        stage = 'finetuned' if checkpoint else 'expanded-head-initialization'
    model.eval()
    reference_metadata = read(str(native_reference) + '.json')
    if reference_metadata['checkpoint_sha256'] != digest(actual_checkpoint) or reference_metadata['source_commit'] != COMMIT:
        raise ValueError('Paddle 3.3 native reference checkpoint/source differs')
    if reference_metadata['preprocessing'] != info['preprocessing'] or reference_metadata['vocabulary'] != tokens(characters):
        raise ValueError('Native reference preprocessor/vocabulary differs')
    if digest(native_reference) != reference_metadata['sha256']:
        raise ValueError('Native reference arrays changed')
    source_native = np.load(native_reference, allow_pickle=False)
    output.mkdir(parents=True)
    if not original:
        snapshot = output / 'weights.pdparams'
        shutil.copyfile(actual_checkpoint, snapshot)
        if digest(snapshot) != digest(actual_checkpoint):
            raise RuntimeError('Candidate checkpoint snapshot differs')
        actual_checkpoint = snapshot
    prefix = output / 'recognizer'
    static = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec([None, 3, 48, None], 'float32', 'x')], full_graph=True)
    paddle.jit.save(static, str(prefix))
    paddle2onnx.export(model_filename=str(prefix) + '.json', params_filename=str(prefix) + '.pdiparams',
                      save_file=str(prefix) + '.onnx', opset_version=14, auto_upgrade_opset=False,
                      enable_onnx_checker=True, export_fp16_model=False, optimize_tool='none')
    graph = onnx.load(str(prefix) + '.onnx')
    props = {item.key: item.value for item in graph.metadata_props}
    props.update({'character': '\n'.join(characters), 'localocr.source_commit': COMMIT,
                  'localocr.stage': stage, 'localocr.preprocessing': json.dumps(info['preprocessing'], sort_keys=True)})
    onnx.helper.set_model_props(graph, props)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(prefix) + '.onnx')
    if graph.opset_import[0].version != 14:
        raise RuntimeError('ONNX opset differs from fixed 14')
    shape = graph.graph.input[0].type.tensor_type.shape.dim
    if len(shape) != 4 or shape[0].HasField('dim_value') or shape[3].HasField('dim_value'):
        raise RuntimeError('ONNX batch/width is not dynamic')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(prefix) + '.onnx', sess_options=options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('CPU-only ONNX Runtime required')
    reference_cases = reference_metadata['cases']
    if stage == 'finetuned':
        policy = reference_metadata.get('final_shape_policy', {})
        if reference_metadata.get('dataset') != info['dataset'] or policy.get('minimum_width') != info['preprocessing']['width']['min']:
            raise ValueError('Final native reference lacks the bound training shape policy')
        required_shapes = {(batch, width) for batch in (1, 2) for width in (
            policy['minimum_width'], policy['maximum_training_width'], policy['capacity_width'])}
        observed_shapes = {(case['batch'], case['width']) for case in reference_cases}
        if not required_shapes.issubset(observed_shapes) or not any(case.get('type') == 'synthetic-training-crop' for case in reference_cases):
            raise ValueError('Final native reference omits real crops or required dynamic shapes')
    parity = []
    with paddle.no_grad():
        for case, case_metadata in enumerate(reference_cases):
            batch, width = case_metadata['batch'], case_metadata['width']
            values = source_native['input_' + str(case)]
            if values.dtype != np.float32 or values.shape != (batch, 3, 48, width) or not np.isfinite(values).all():
                raise ValueError('Native reference input differs from declared shape')
            native = _numpy(model(paddle.to_tensor(values)))
            converted = session.run(None, {session.get_inputs()[0].name: values})[0]
            if native.shape != converted.shape:
                raise RuntimeError('Native/ONNX shape differs')
            error = float(np.max(np.abs(native - converted)))
            reference = source_native['output_' + str(case)]
            same = decode(native, characters) == decode(converted, characters) == decode(reference, characters)
            if not np.allclose(native, converted, atol=2e-4, rtol=2e-3) or not np.allclose(reference, converted, atol=2e-4, rtol=2e-3) or not same:
                raise RuntimeError('Native/ONNX numerical or exact decode parity failed')
            parity.append({'batch': batch, 'width': width, 'type': case_metadata.get('type', 'random-shape'), 'shape': list(native.shape),
                           'maximum_absolute_error': error, 'paddle_3_3_maximum_error': float(np.max(np.abs(reference - converted))), 'exact_decode_equal': same})
    write(output / 'vocabulary.json', {'schema_version': 'localocr.pretrained.vocabulary.v1',
          'tokens': tokens(characters), 'blank_id': 0, 'space_id': len(characters) + 1})
    dictionary = output / 'dictionary.txt'
    dictionary.write_text('\n'.join(characters) + '\n', encoding='utf-8', newline='\n')
    write(output / 'source-audit.json', audit)
    candidate = {'schema_version': 'localocr.pretrained.candidate.v1',
        'id': identifier or ('ppocrv6-tiny-exp-20261011-001' if stage == 'finetuned' else 'ppocrv6-tiny-reference-20261011-' + stage), 'family': 'official-ppocrv6-tiny', 'stage': stage,
        'model': {'file': 'recognizer.onnx', 'sha256': digest(str(prefix) + '.onnx')},
        'vocabulary': {'file': 'vocabulary.json', 'sha256': digest(output / 'vocabulary.json'),
                       'blank_id': 0, 'space_id': len(characters) + 1},
        'preprocessing': info['preprocessing'],
        'decoder': {'type': 'ctc', 'output': 'probabilities', 'blank_id': 0, 'repeat': 'ctc'},
        'source': {'commit': COMMIT, 'checkpoint': {'file': str(actual_checkpoint), 'sha256': digest(actual_checkpoint)},
                   'dictionary': {'file': str(dictionary), 'sha256': digest(dictionary)},
                   'source_manifest': {'file': 'source-audit.json', 'sha256': digest(output / 'source-audit.json')},
                   'config': {'file': str(CONFIG), 'sha256': CONFIG_SHA},
                   'preparation': {'file': str(prepared / 'initialization.json'), 'sha256': digest(prepared / 'initialization.json')}},
        'export': {'paddle': paddle.__version__, 'paddle2onnx': paddle2onnx.__version__,
                   'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'opset': 14,
                   'cpu_parity': parity, 'native_reference': {'file': str(Path(native_reference).resolve()), 'sha256': digest(native_reference), 'paddle': reference_metadata['paddle']}, 'reparameterized': False},
        'production_enabled': False, 'dataset': info['dataset'], 'created_utc': utc(), 'adapter_sha256': digest(__file__)}
    if training_source is not None:
        shutil.copyfile(training_source['training_receipt']['file'], output / 'training-receipt.json')
        candidate['source']['training_receipt'] = {'file': 'training-receipt.json', 'sha256': digest(output / 'training-receipt.json')}
        candidate['source']['training_closure'] = _snapshot_closure(training_source, output)
        candidate['source']['hash_semantics'] = 'Integrity of locally frozen evidence; not publisher authentication'
        candidate['selected_weights_changed'] = True
    write(output / 'candidate.json', candidate)
    print(json.dumps({'candidate': str(output / 'candidate.json'), 'stage': stage, 'cpu_parity_cases': len(parity)}), flush=True)


def _reference_inputs(info, extended=False):
    """Deterministic shape probes plus frozen-train crops; no heldout access."""
    rng = np.random.default_rng(20261011)
    arrays, cases = [], []
    shapes = [(1, 32), (2, 160), (1, 320), (2, 640), (6, 320)]
    rows, policy = [], None
    if extended:
        root = Path(info['dataset_root'])
        if dataset_binding(root) != info['dataset']:
            raise ValueError('Reference training dataset differs from preparation')
        rows = read_split(root, 'train')
        maximum_width = max(math.ceil(48 * row['width'] / row['height']) for row in rows)
        policy = {'minimum_width': info['preprocessing']['width']['min'],
                  'maximum_training_width': maximum_width, 'capacity_width': max(1024, maximum_width),
                  'training_rows': len(rows), 'derived_from': 'frozen train only'}
        for width in (policy['minimum_width'], policy['maximum_training_width'], policy['capacity_width']):
            for batch in (1, 2):
                if (batch, width) not in shapes:
                    shapes.append((batch, width))
    for batch, width in shapes:
        arrays.append(rng.uniform(-1, 1, (batch, 3, 48, width)).astype('float32'))
        cases.append({'type': 'random-shape', 'batch': batch, 'width': width})
    if extended:
        positive = [row for row in rows if row['text']]
        required = lambda row: len(row['text']) + sum(a == b for a, b in zip(row['text'], row['text'][1:]))
        selections = [max(rows, key=lambda row: row['width'] / row['height']),
                      max(positive, key=lambda row: len(row['text'])), max(positive, key=required)]
        for kind in sorted({row['kind'] for row in rows}):
            selections.append(next(row for row in rows if row['kind'] == kind))
        for predicate in (lambda text: bool(text) and text.startswith(' '),
                          lambda text: bool(text) and text.endswith(' '),
                          lambda text: '  ' in text, lambda text: '\u200d' in text,
                          lambda text: '\ufe0f' in text,
                          lambda text: any(unicodedata.combining(character) for character in text)):
            selected = next((row for row in rows if predicate(row['text'])), None)
            if selected is not None:
                selections.append(selected)
        seen = set()
        batches = []
        for row in selections:
            if row['image'] not in seen:
                seen.add(row['image'])
                batches.append([row])
        # Two negative canvases both have the exact single-crop width 512.
        same_width = [row for row in rows if not row['text'] and
                      math.ceil(48 * row['width'] / row['height']) <= policy['minimum_width']][:2]
        if len(same_width) == 2:
            batches.append(same_width)
        for batch in batches:
            images = image_batch(root, batch, info['preprocessing'])
            arrays.append(images)
            cases.append({'type': 'synthetic-training-crop', 'batch': len(batch), 'width': images.shape[-1],
                          'split': 'train', 'kinds': [row['kind'] for row in batch],
                          'image_sha256': [row['sha256'] for row in batch],
                          'label_codepoints': [len(row['text']) for row in batch],
                          'ctc_required_steps': [required(row) for row in batch]})
    return arrays, cases, policy


def reference(prepared, output, checkpoint=None, original=False):
    output = Path(output).resolve()
    if output.exists() or Path(str(output) + '.json').exists():
        raise ValueError('Native reference output must be fresh')
    source_audit()
    disk_guard(output)
    paddle = _paddle('cpu')
    if paddle.__version__ != '3.3.0':
        raise RuntimeError('Native source reference must use training backend Paddle 3.3.0')
    prepared = Path(prepared).resolve()
    info = read(prepared / 'initialization.json')
    if original:
        characters = DICTIONARY.read_text(encoding='utf-8').splitlines()
        model = build_model(characters, info['maximum_label_length'])
        load_original(model)
        actual_checkpoint = CHECKPOINT
    else:
        model, characters, info = _load_prepared(prepared, checkpoint, 'cpu')
        actual_checkpoint = Path(checkpoint).resolve() if checkpoint else prepared / 'initialized.pdparams'
    model.eval()
    inputs, cases, final_shape_policy = _reference_inputs(info, extended=bool(checkpoint))
    arrays = {}
    with paddle.no_grad():
        for case, values in enumerate(inputs):
            arrays['input_' + str(case)] = values
            arrays['output_' + str(case)] = _numpy(model(paddle.to_tensor(values)))
            cases[case]['shape'] = list(arrays['output_' + str(case)].shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write(str(output) + '.json', {'schema_version': 'localocr.paddle.native-reference.v1',
        'paddle': paddle.__version__, 'device': paddle.get_device(), 'source_commit': COMMIT,
        'checkpoint_sha256': digest(actual_checkpoint), 'preprocessing': info['preprocessing'],
        'vocabulary': tokens(characters), 'cases': cases, 'dataset': info['dataset'],
        'final_shape_policy': final_shape_policy, 'sha256': digest(output), 'created_utc': utc()})
    print(json.dumps({'native_reference': str(output), 'cases': len(cases)}), flush=True)

def development_better(candidate, incumbent, policy='legacy'):
    if policy == 'face-pareto':
        quality_selection.validate_summary(candidate.get('quality'))
        return incumbent is None or quality_selection.face_better(candidate['quality'], incumbent.get('quality'))
    if policy != 'legacy': raise ValueError('Unknown development selection policy')
    return incumbent is None or (candidate['exact_correct'], -candidate['edit_distance']) > (incumbent['exact_correct'], -incumbent['edit_distance'])


def set_research_root(root):
    """Change owned research evidence scope while preserving fixed official inputs."""
    global ROUND, BUDGET
    target = Path(root).absolute()
    artifact_root = (REPO / 'artifacts').resolve()
    if '..' in target.parts or target.resolve() == artifact_root or not target.resolve().is_relative_to(artifact_root):
        raise ValueError('Research root must be a dedicated directory inside repository artifacts')
    for ancestor in (target, *target.parents):
        if ancestor.exists():
            properties = ancestor.lstat()
            if stat.S_ISLNK(properties.st_mode) or getattr(properties, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Research root contains a symbolic link or reparse point')
    ROUND = target.resolve(); BUDGET = ROUND / 'gpu-budget.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', help='Owned new research evidence root; fixed official inputs stay unchanged')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('reference')
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--original', action='store_true')
    for name in ('probe', 'train', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True)
        p.add_argument('--output', required=True)
        if name == 'export':
            p.add_argument('--checkpoint')
            p.add_argument('--identifier')
            p.add_argument('--native-reference', required=True)
            p.add_argument('--training-receipt-sha256')
            p.add_argument('--original', action='store_true')
        else:
            p.add_argument('--dataset', required=True)
            p.add_argument('--seconds', type=float, default=120 if name == 'probe' else 2580)
            p.add_argument('--budget', help='Internal child budget receipt; parent authorization is separate')
            p.add_argument('--batch-size', type=int, default=8 if name == 'probe' else 16)
            p.add_argument('--learning-rate', type=float, default=.0001)
            p.add_argument('--warm-start')
            p.add_argument('--warm-start-receipt-sha256')
            p.add_argument('--freeze-backbone', action='store_true')
            p.add_argument('--empty-ctc-weight', type=float, default=1.0)
            if name == 'probe':
                p.add_argument('--minimum-width', type=int)
            else:
                p.add_argument('--selection-policy', choices=('legacy', 'face-pareto'), default='legacy')
                p.add_argument('--eval-every-seconds', type=float, default=180.0)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    research_root = args.pop('research_root')
    if research_root is not None: set_research_root(research_root)
    globals()[command](**args)


if __name__ == '__main__':
    main()









