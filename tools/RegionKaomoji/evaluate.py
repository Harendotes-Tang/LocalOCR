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
from tools.PretrainedOcr.quality_evaluate import exact_metrics as metrics, scene_gate, ROUND, is_round, round_write_guard, verify_dual, validate_owned_cpu_capture
from tools.PretrainedOcr import round_profile as rp
from tools.RegionKaomoji import candidates as c, fusion as f, classifier as cls

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
SUPPORTED_VIEWS = ('raw', 'edge-pad-4', 'edge-pad-6')


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


def research_root(path):
    """Owning research root: a registered round profile, the legacy quality round, or the first region root."""
    profile=rp.profile_for(path)
    if profile is not None:return Path(profile.root)
    return ROUND if is_round(path) else EVALUATION_ROOT


def output_path(path):
    path=safe_path(path); root=safe_path(research_root(path))
    if path == root or not path.is_relative_to(root) or not path.relative_to(root).parts[0].startswith('evaluation'):
        raise ValueError('Output must belong to this research evaluation namespace')
    return path


def write_new(path, value):
    path = output_path(path)
    common.disk_guard(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    round_write_guard(path,len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_path(path.parent)
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


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


REGRESSION_SPLIT = 'heldout-regression'


def demotion_receipt(root):
    """The registered round whose demotion sidecar lists this dataset's viewed heldout, or None.

    A viewed heldout is development regression material only; it never claims
    a new first use and its outputs belong to the demoting round.
    """
    frozen = scene_binding(root)
    heldout = safe_path(root) / common.read(safe_path(root) / 'dataset.json')['splits']['heldout']['file']
    digest = common.digest(heldout)
    if digest != frozen['split_sha256']['heldout']:
        raise ValueError('Frozen heldout split changed')
    for profile in rp.ROUNDS:
        sidecar = Path(profile.root) / 'previous-heldout-development-use.json'
        if not sidecar.exists():
            continue
        value = common.read(sidecar)
        if value.get('schema') != 'localocr.development-demotion.v1' or value.get('original_rewritten') is not False:
            continue
        for record in value.get('old_datasets', []):
            if record.get('sha256') == digest and safe_path(REPO / record['file']) == heldout:
                return {'round': profile.name, 'sidecar': str(sidecar), 'sidecar_sha256': common.digest(sidecar), 'heldout_sha256': digest}
    return None


def data_split(split):
    return 'heldout' if split == REGRESSION_SPLIT else split


def image_jobs(root, split, inputs=None, allow_heldout=False):
    if split == REGRESSION_SPLIT:
        if demotion_receipt(root) is None:
            raise ValueError('Heldout regression requires a registered demotion sidecar for this viewed heldout')
        split = 'heldout'
        allow_heldout = True
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


def classifier_section(reference):
    """Normalize a classifier reference to its bound identity; None means no classifier evidence."""
    if reference is None:
        return None
    if isinstance(reference, str):
        reference = {'file': reference}
    if not isinstance(reference, dict) or not isinstance(reference.get('file'), str):
        raise ValueError('Classifier configuration needs a receipt file')
    binding = cls.classifier_binding(safe_path(reference['file']))
    for key in ('sha256', 'threshold', 'id'):
        if key in reference and reference[key] != binding[key]:
            raise ValueError('Classifier receipt differs from the frozen configuration')
    return {'file': binding['file'], 'id': binding['id'], 'sha256': binding['sha256'],
            'model_sha256': binding['model_sha256'], 'threshold': binding['threshold']}


def configuration(path=None, classifier=None):
    value = common.read(safe_path(path)) if path else {
        'schema_version': CONFIG_SCHEMA, 'candidate': c.configuration(), 'fusion': f.configuration(),
        'split': c.split_configuration(), 'classifier': None,
        'views': VIEWS, 'limits': LIMITS, 'join': {'same_line': '  ', 'different_line': '\n', 'vertical_overlap': .5}}
    if not isinstance(value, dict) or set(value) - {'split', 'classifier'} != {'schema_version', 'candidate', 'fusion', 'views', 'limits', 'join'}:
        raise ValueError('Unknown pipeline configuration')
    if classifier is not None:
        if value.get('classifier') not in (None, classifier) and value['classifier'].get('file') != str(safe_path(classifier)):
            raise ValueError('Configuration already binds a different classifier')
        value['classifier'] = classifier
    if value['schema_version'] != CONFIG_SCHEMA or value['limits'] != LIMITS:
        raise ValueError('Fixed inference budgets cannot change')
    views = value['views']
    if (not isinstance(views, list) or len(views) < 2 or len(set(views)) != len(views) or 'raw' not in views
            or any(view not in SUPPORTED_VIEWS for view in views)):
        raise ValueError('Declared views must be a unique supported set of at least two that includes the raw view')
    if value['join'] != {'same_line': '  ', 'different_line': '\n', 'vertical_overlap': .5}:
        raise ValueError('Unsupported declared output reading order')
    value = json.loads(json.dumps(value, allow_nan=False))
    # A configuration frozen before pre-recognition splitting existed declares no split section: splitting stays off.
    split = c.split_configuration(value['split']) if 'split' in value else dict(c.split_configuration(), enabled=False)
    fusion = f.configuration(value['fusion'])
    if len(views) < fusion['minimum_agreeing_views']:
        raise ValueError('Declared views cannot be fewer than the agreeing minimum')
    return {**value, 'candidate': c.configuration(value['candidate']), 'fusion': fusion, 'split': split,
            'classifier': classifier_section(value.get('classifier'))}


def split_entries(image, entries, split_config, recognize):
    """Research first pass: split production boxes at blank bands and re-recognize children.

    ``entries`` are production regions; ``recognize`` maps a list of RGB crops to
    (text, score) pairs with the production recognizer. The returned plan keeps
    every parent so the split can be undone when no child becomes a face.
    """
    array = c.rgb_image(image)
    height, width = array.shape[:2]
    validated = c.baseline_regions(entries, (width, height))
    if not split_config['enabled']:
        return [dict(item) for item in validated], {'parents': {}, 'children': {}}
    mask = c.foreground(array, c.DEFAULT_CONFIG['contrast_threshold'], c.DEFAULT_CONFIG['background_window'])
    crops, pending = [], []
    for entry in validated:
        children = c.split_children(mask, entry['bbox'], split_config)
        if not children:
            pending.append((entry, None))
            continue
        pending.append((entry, children))
        # Each child is recognized on the parent's own crop with the other children
        # painted over in the local background, so the recognizer sees the same
        # scale and margins as the production pass and reproduces its spacing.
        left, top, right, bottom = entry['bbox']
        parent = np.ascontiguousarray(array[top:bottom, left:right])
        border = np.concatenate((parent[0], parent[-1], parent[:, 0], parent[:, -1]), axis=0)
        background = np.median(border.astype(np.int16), axis=0).astype(np.uint8)
        for index, _ in enumerate(children):
            crop = parent.copy()
            for other, (ol, _, orr, _) in enumerate(children):
                if other != index:
                    crop[:, max(0, ol - left):max(0, orr - left)] = background
            crops.append(crop)
    results = list(recognize(crops)) if crops else []
    if len(results) != len(crops):
        raise ValueError('Child recognition count differs from split children')
    first_pass, plan, cursor = [], {'parents': {}, 'children': {}, 'map': {}}, 0
    for entry, children in pending:
        if children is None:
            plan['map'][str(len(first_pass))] = entry['index']
            first_pass.append({'bbox': entry['bbox'], 'text': entry['text'], 'score': entry['score']})
            continue
        parent_index, indices = entry['index'], []
        for box in children:
            text, score = results[cursor]
            cursor += 1
            if not isinstance(text, str) or not math.isfinite(float(score)):
                raise ValueError('Child recognition output invalid')
            indices.append(len(first_pass))
            plan['children'][str(len(first_pass))] = parent_index
            first_pass.append({'bbox': list(box), 'text': text, 'score': float(score)})
        plan['parents'][str(parent_index)] = {'entry': {'bbox': entry['bbox'], 'text': entry['text'], 'score': entry['score']},
                                             'children': indices}
    return first_pass, plan


def locate_parent_span(parent_text, child_text):
    """Half-open span of the child's reading inside the parent's production reading, ignoring ASCII spaces; None when not unique."""
    if not child_text or not parent_text:
        return None
    parent_positions = [index for index, char in enumerate(parent_text) if char != ' ']
    parent_compact = ''.join(parent_text[index] for index in parent_positions)
    child_compact = child_text.replace(' ', '')
    if not child_compact:
        return None
    first = parent_compact.find(child_compact)
    if first < 0 or parent_compact.find(child_compact, first + 1) >= 0:
        return None
    return parent_positions[first], parent_positions[first + len(child_compact) - 1] + 1


def locate_parent_bytes(parent_text, child_text):
    """Find the child's reading inside the parent's production reading ignoring ASCII spaces.

    Returns the parent's own raw byte span (its spaces included) when the
    space-insensitive match is unique; otherwise None. The child never
    contributes bytes: only the production reading is ever output for body.
    """
    span = locate_parent_span(parent_text, child_text)
    return None if span is None else parent_text[span[0]:span[1]]


def materialize_splits(regions, plan, config):
    """Keep a split only where fusion accepted a child and every other child maps onto the parent reading.

    Body children never introduce a new reading: each must locate uniquely in
    the production parent's text (ignoring ASCII spaces) and then carries the
    parent's own bytes. Otherwise the production parent is restored verbatim
    and the face replacement is dropped. Indices of unsplit regions are mapped
    back to production indices so body protection compares against the
    production first pass; materialized children keep their own boxes.
    """
    if not plan['parents']:
        return regions, {'materialized': 0, 'restored': 0, 'restored_unverifiable': 0, 'parent_level_splices': 0}
    owner = {int(k): v for k, v in plan['children'].items()}
    mapping = {int(k): v for k, v in plan['map'].items()}
    def region_children(region):
        indices = set(region.get('baseline_indices', []))
        if region.get('baseline_index') is not None:
            indices.add(region['baseline_index'])
        return {owner[i] for i in indices if i in owner}
    def remap(region):
        value = dict(region)
        if value.get('baseline_index') in mapping:
            value['baseline_index'] = mapping[value['baseline_index']]
            if isinstance(value.get('splice'), dict):
                value['splice'] = dict(value['splice'], baseline_index=value['baseline_index'])
        if value.get('baseline_indices'):
            value['baseline_indices'] = [mapping.get(i, i) for i in value['baseline_indices']]
        return value
    touched = set()
    for region in regions:
        if region.get('source', 'baseline') != 'baseline':
            touched.update(region_children(region))
    # A touched parent materializes only when every untouched child locates
    # uniquely inside the parent's production reading; the child then carries
    # the parent's bytes. Otherwise the parent is restored and the face dropped.
    parent_bytes, unverifiable = {}, set()
    for parent_index in sorted(touched):
        parent_text = plan['parents'][str(parent_index)]['entry']['text']
        for region in regions:
            if region.get('source', 'baseline') != 'baseline' or region_children(region) != {parent_index}:
                continue
            located = locate_parent_span(parent_text, region['text'])
            if located is None:
                unverifiable.add(parent_index)
                break
            parent_bytes[(parent_index, region['baseline_index'])] = located
    touched -= unverifiable
    # A whole-reading splice accepted on a split child is re-expressed against the production parent:
    # its prefix and suffix must locate in the parent's bytes and the parent bytes between them must
    # be one bracketed non-body fragment; the parent then carries the splice and its split is dropped.
    regions, lifted, touched, unverifiable = lift_child_splices(regions, plan, parent_bytes, owner, touched, unverifiable)
    output, restored = [], set(unverifiable)
    for region in regions:
        parents = region_children(region)
        if parents and not parents & touched:
            restored.update(parents)
            continue
        value = remap(region)
        if parents:
            value['split_parent_index'] = sorted(parents)[0]
            if value.get('source', 'baseline') == 'baseline':
                value['source'] = 'baseline-split-child'
                start, end = parent_bytes[(value['split_parent_index'], region['baseline_index'])]
                value['text'] = plan['parents'][str(value['split_parent_index'])]['entry']['text'][start:end]
                value['child_reading'] = region['text']
        output.append(value)
    output = coalesce_body_children(output, plan, parent_bytes)
    output.extend(lifted)
    for parent_index in sorted(restored):
        entry = plan['parents'][str(parent_index)]['entry']
        output.append({'bbox': entry['bbox'], 'text': entry['text'], 'score': entry['score'], 'source': 'baseline',
                       'baseline_index': parent_index, 'split_restored': True, 'split_unverifiable': parent_index in unverifiable})
    output.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    return output, {'materialized': len(touched), 'restored': len(restored), 'restored_unverifiable': len(unverifiable), 'parent_level_splices': len(lifted)}


def _bracketed_non_body(middle):
    if not middle or not any(char in c.ATOMS.bracket_chars for char in middle):
        return False
    if c.is_face_text(middle):
        return True
    outside, digits = c.ordinary_anchor_evidence(middle)
    return not digits and not outside and c.single_bracket_pair(middle)


def lift_child_splices(regions, plan, spans, owner, touched, unverifiable):
    """Turn whole-reading splices on split children into production-parent splices, or restore the parent."""
    kept, lifted = [], []
    touched, unverifiable = set(touched), set(unverifiable)
    for region in regions:
        proof = region.get('splice') if region.get('source') == 'baseline-with-regional-splice' else None
        child = region.get('baseline_index')
        if not proof or child not in owner:
            kept.append(region)
            continue
        parent_index = owner[child]
        parent = plan['parents'][str(parent_index)]
        text = parent['entry']['text']
        order = parent['children']
        position = order.index(child)
        previous_end = spans[(parent_index, order[position - 1])][1] if position > 0 and (parent_index, order[position - 1]) in spans else 0
        next_start = spans[(parent_index, order[position + 1])][0] if position + 1 < len(order) and (parent_index, order[position + 1]) in spans else len(text)
        prefix, suffix, face = proof.get('preserved_prefix', ''), proof.get('preserved_suffix', ''), proof.get('replacement_text')
        start = locate_parent_span(text, prefix)[1] if prefix and locate_parent_span(text, prefix) else (previous_end if not prefix else None)
        end = locate_parent_span(text, suffix)[0] if suffix and locate_parent_span(text, suffix) else (next_start if not suffix else None)
        if start is not None and end is not None:
            # Separator whitespace around the fragment stays in the preserved parent bytes.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
        if (parent_index not in touched or start is None or end is None or not previous_end <= start <= end <= next_start
                or not isinstance(face, str) or not face or not _bracketed_non_body(text[start:end])):
            unverifiable.add(parent_index)
            touched.discard(parent_index)
            continue
        entry = parent['entry']
        lifted.append({'bbox': entry['bbox'], 'text': text[:start] + face + text[end:], 'score': entry['score'],
                       'source': 'baseline-with-regional-splice', 'baseline_index': parent_index,
                       'regional_bbox': region['regional_bbox'], 'regional_ink_bbox': region.get('regional_ink_bbox', region['regional_bbox']),
                       'regional_text': face, 'candidate_id': region.get('candidate_id'), 'model_id': region.get('model_id'),
                       'split_parent_index': parent_index, 'lifted_from_child': child,
                       'splice': {'baseline_index': parent_index, 'original_text_span': [start, end], 'original_face_text': text[start:end],
                                  'replacement_text': face, 'preserved_prefix': text[:start], 'preserved_suffix': text[end:],
                                  'context_crops': [], 'context_records': [], 'baseline_parent_sha256': c.baseline_parent_digest(dict(entry, index=parent_index)),
                                  'ordinary_prefix_suffix_preserved': True, 'method': 'whole-reading-agreement-lifted-to-parent'}})
        touched.discard(parent_index)
        unverifiable.discard(parent_index)
    lifted_parents = {item['baseline_index'] for item in lifted}
    # Children of a lifted parent are dropped: the parent region carries every byte.
    kept = [region for region in kept if not ({owner[i] for i in region.get('baseline_indices', []) if i in owner} | ({owner[region['baseline_index']]} if region.get('baseline_index') in owner else set())) & lifted_parents]
    return kept, lifted, touched, unverifiable


def coalesce_body_children(output, plan, spans):
    """Adjacent body children of one materialized parent become one region carrying the parent's contiguous bytes.

    Splitting exists to isolate faces; it never fragments a body run, so the
    body literal (its separators included) survives byte-for-byte.
    """
    by_parent = {}
    for value in output:
        if value.get('source') == 'baseline-split-child':
            by_parent.setdefault(value['split_parent_index'], []).append(value)
    drop, merged = set(), []
    for parent_index, items in by_parent.items():
        parent = plan['parents'][str(parent_index)]
        position = {index: order for order, index in enumerate(parent['children'])}
        items.sort(key=lambda value: position[value['baseline_index']])
        runs = []
        for value in items:
            if runs and position[value['baseline_index']] == position[runs[-1][-1]['baseline_index']] + 1:
                runs[-1].append(value)
            else:
                runs.append([value])
        for run in runs:
            if len(run) < 2:
                continue
            start = spans[(parent_index, run[0]['baseline_index'])][0]
            end = spans[(parent_index, run[-1]['baseline_index'])][1]
            boxes = [value['bbox'] for value in run]
            merged.append(dict(run[0], bbox=[min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)],
                               text=parent['entry']['text'][start:end], child_reading=' '.join(value['child_reading'] for value in run),
                               merged_children=[value['baseline_index'] for value in run]))
            drop.update(id(value) for value in run)
    return [value for value in output if id(value) not in drop] + merged


def implementation_binding():
    names = ['scripts/local-ai-ocr-worker.py', 'localocr/__init__.py', 'localocr/bundles.py', 'localocr/version.py',
             'tools/RegionKaomoji/__init__.py', 'tools/RegionKaomoji/evaluate.py',
             'tools/RegionKaomoji/candidates.py', 'tools/RegionKaomoji/fusion.py', 'tools/RegionKaomoji/classifier.py',
             'tools/PretrainedOcr/kaomoji_atoms.json', 'tools/PretrainedOcr/kaomoji_grammar.py', 'tools/PretrainedOcr/round_profile.py',
             'tools/PretrainedOcr/quality_evaluate.py', 'tools/PretrainedOcr/__init__.py', 'tools/PretrainedOcr/common.py', 'tools/PretrainedOcr/evaluate.py',
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
            'classifier': config.get('classifier'),
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

    def recognize_children(self, crops):
        """Production recognizer on axis-aligned child crops (RGB arrays), batched like the worker's symbol reread."""
        from rapidocr.ch_ppocr_rec import TextRecInput
        if not crops:
            return []
        samples = [np.ascontiguousarray(np.asarray(crop)[:, :, ::-1]) for crop in crops]
        result = self.engine._load_rec_model()(TextRecInput(img=samples, return_word_box=False))
        if len(result.txts) != len(samples) or len(result.scores) != len(samples):
            raise ValueError('Actual child recognizer output count differs')
        return [(str(text), float(score)) for text, score in zip(result.txts, result.scores)]

    def first_pass(self, image, split_config):
        """Production first pass plus the research pre-recognition split of its boxes.

        The production entries are returned unchanged for comparison; children
        are recognized by the same production recognizer and then pass through
        the production symbol re-read so the only difference is the split.
        """
        entries, native = self.predict(image)
        first_pass, plan = split_entries(image, entries, split_config, self.recognize_children)
        if plan['parents']:
            from types import SimpleNamespace
            pixels = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
            # Only split children pass through the symbol re-read; unsplit production entries stay byte-identical.
            children = [first_pass[int(index)] for index in sorted(plan['children'], key=int)]
            boxes = [[[l, t], [r, t], [r, b], [l, b]] for l, t, r, b in (item['bbox'] for item in children)]
            result = SimpleNamespace(boxes=boxes, txts=[item['text'] for item in children], scores=[item['score'] for item in children])
            reread = self.worker.reread_symbols(self.engine, pixels, result)
            if len(reread.txts) == len(children):
                for item, text, score in zip(children, reread.txts, reread.scores):
                    item['text'], item['score'] = str(text), float(score)
        return entries, native, first_pass, plan


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


def predict_scene(image, baseline, models, config, classifier=None):
    """Inference boundary: receives pixels and model objects, never scene truth."""
    timings = {}; started = time.perf_counter()
    split_config = config.get('split') or dict(c.split_configuration(), enabled=False)
    if split_config['enabled'] and hasattr(baseline, 'first_pass'):
        production, native, first_pass, plan = baseline.first_pass(image, split_config)
        timings['baseline_seconds'] = time.perf_counter() - started
    else:
        production, native = baseline.predict(image); timings['baseline_seconds'] = time.perf_counter() - started
        first_pass, plan = split_entries(image, production, dict(split_config, enabled=False), lambda crops: [])
    entries = first_pass
    started = time.perf_counter(); routing = c.locate_candidates(image, entries, config['candidate'], classifier=classifier)
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
        fused = f.fuse_candidates(image, entries, routing['candidates'], records, config['fusion'], config['candidate'], contexts=contexts, classifier=classifier)
        regions, split_stats = materialize_splits(fused['regions'], plan, config)
        results[identity] = {'regions': regions, 'decisions': fused['decisions'], 'views': records,
                             'text': joined_regions(regions, config),
                             'specialist_seconds': sum(infer_times), 'fusion_seconds': time.perf_counter() - at,
                             'view_latency': percentiles(infer_times), 'accepted_count': fused['accepted_count'],
                             'split': split_stats}
    result = {'baseline': production, 'baseline_native': native, 'baseline_text': joined_regions(production, config),
              'first_pass': {'entries': entries, 'plan': plan, 'config': split_config,
                             'split_parents': len(plan['parents']), 'split_children': len(plan.get('children', {}))},
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
    return output_path(research_root(binding['inputs']['file']) / ('evaluation-heldout-first-use-' + binding['dataset']['split_sha256']['heldout'] + '.json'))


def validate_budget(budget, path=None):
    """A closed ledger whose ceiling is the owning round's declared GPU authorization (0 s for a CPU-only round)."""
    profile = rp.profile_for(path) if path is not None else None
    maximum = profile.maximum_seconds if profile is not None else 7200
    if (budget.get('schema_version') != 'localocr.gpu-budget.v1' or budget.get('active') is not None
            or budget.get('maximum_seconds') != maximum or type(budget.get('used_seconds')) not in (int, float)
            or not math.isfinite(budget['used_seconds']) or not 0 <= budget['used_seconds'] <= maximum):
        raise ValueError(f'Selection needs the completed closed budget receipt with the round ceiling of {maximum} seconds')


def final_families(binding):
    models = binding['models']
    required = rp.specialist_families(binding['inputs']['file'])
    if (len(models) != required or len({item['family'] for item in models}) != required
            or any(item['stage'] != 'finetuned' for item in models)):
        raise ValueError('Freeze requires the round\'s declared finetuned specialist families')


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
    if is_round(path):
        validate_owned_cpu_capture(actual)
        proof=actual.get('worker_ownership',{})
        verify_worker_proof(proof,proof.get('launcher_pid'),proof.get('owner_pid'))
        if actual['worker_pid']!=proof['worker_pid']:raise ValueError('Measured worker PID differs from verified chain')
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
    validate_budget(common.read(value['budget_receipt']['file']), value['budget_receipt']['file'])
    return {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'value': value}


def claim_heldout(root, selection, binding, output, dual_freeze=None):
    dual=verify_dual(dual_freeze,'scene',selection,root) if is_round(root) and dual_freeze else None
    if is_round(root) and dual is None:raise ValueError('New scene heldout requires dual freeze before access')
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
             'status': 'claimed-before-reading-heldout-images-or-truth; failure remains consumed','dual_freeze':dual}
    path = usage_path(binding); write_new(path, value)
    return {'file': str(path), 'sha256': common.digest(path)}


def owned_parent(owner, action="_worker"):
    if os.getppid() == owner: return True
    if sys.platform != 'win32': return False
    import psutil
    try:
        parent = psutil.Process(os.getppid()); command = parent.cmdline()
        return (parent.ppid() == owner and safe_path(parent.exe()) == safe_path(sys.executable)
                and command[1:4] == ['-m', 'tools.RegionKaomoji.evaluate', action])
    except (psutil.Error, OSError):
        return False


def owned_worker_proof(owner, action='_worker'):
    """Capture the live direct/venv-redirector parent, before its PID disappears."""
    if type(owner) is not int or owner<=0 or not owned_parent(owner,action):
        raise ValueError('CPU worker is not owned by the declared launcher owner')
    parent=os.getppid()
    return {'owner_pid':owner,'worker_pid':os.getpid(),'worker_parent_pid':parent,
            'launcher_pid':os.getpid() if parent==owner else parent,
            'relation':'direct-child' if parent==owner else 'windows-venv-redirector',
            'owner_chain_verified':True}


def verify_worker_proof(proof, launcher_pid, owner_pid):
    if not isinstance(proof,dict) or proof.get('owner_chain_verified') is not True:
        raise ValueError('Missing actual owned CPU parent-chain proof')
    for key in ('owner_pid','worker_pid','worker_parent_pid','launcher_pid'):
        if type(proof.get(key)) is not int or proof[key]<=0:raise ValueError('Invalid worker parent-chain PID')
    if proof['owner_pid']!=owner_pid or proof['launcher_pid']!=launcher_pid:
        raise ValueError('CPU worker launcher/owner differs')
    direct=(proof['relation']=='direct-child' and proof['worker_pid']==launcher_pid and proof['worker_parent_pid']==owner_pid)
    redirected=(sys.platform=='win32' and proof['relation']=='windows-venv-redirector'
                and proof['worker_pid']!=launcher_pid and proof['worker_parent_pid']==launcher_pid)
    if not (direct or redirected):raise ValueError('CPU worker parent-chain relation differs')
    return proof


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
    if is_round(root):
        proof=value.get('dual_freeze')
        if not proof or common.digest(proof['file'])!=proof['sha256'] or verify_dual(proof['file'],'scene',value['selection']['file'],root)!=proof:raise ValueError('Worker dual freeze differs')
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
    ownership=owned_worker_proof(args.owner_pid)
    root, output, jobs_path = safe_path(args.dataset), output_path(args.output), safe_path(args.jobs)
    if output.exists(): raise ValueError('CPU worker output already exists')
    config = configuration(args.config, classifier=getattr(args, 'classifier', None))
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
        classifier = cls.CropClassifier(config['classifier']['file']) if config.get('classifier') else None
        if classifier is not None and (classifier.binding['sha256'] != config['classifier']['sha256'] or classifier.threshold != config['classifier']['threshold']):
            raise ValueError('Loaded crop classifier differs from the bound configuration')
        verify_baseline_runtime(baseline.runtime_receipt)
        if args.split == 'heldout':
            first_use = common.read(usage_path(binding))
            frozen = common.read(first_use['selection']['file'])
            if baseline.runtime_receipt != frozen['baseline_runtime']:
                raise ValueError('Heldout actual baseline runtime differs from frozen development')
        load_seconds = time.perf_counter() - at
        first = image_for_job(root, jobs[0]); at = time.perf_counter(); predict_scene(first, baseline, models, config, classifier=classifier)
        cold = time.perf_counter() - at; timer.cancel(); timer.join()
        for _ in range(2):
            timer = threading.Timer(config['limits']['scene_timeout_seconds'], expire); timer.daemon=True; timer.start()
            predict_scene(first, baseline, models, config, classifier=classifier); timer.cancel(); timer.join()
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            for job in jobs:
                timer = threading.Timer(config['limits']['scene_timeout_seconds'], expire); timer.daemon=True; timer.start()
                image = image_for_job(root, job); at = time.perf_counter(); prediction = predict_scene(image, baseline, models, config, classifier=classifier)
                elapsed = time.perf_counter() - at; timings.append(elapsed)
                common.disk_guard(output)
                line=json.dumps({'id':job['id'],'image_sha256':job['sha256'],'pixel_sha256':job['pixel_sha256'],
                                 'prediction':prediction,'seconds':elapsed}, ensure_ascii=False, allow_nan=False) + '\n'
                round_write_guard(output,len(line.encode('utf-8')))
                stream.write(line)
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
                     'baseline_runtime':baseline.runtime_receipt,'worker_pid':os.getpid(),'worker_ownership':ownership}
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
    round_write_guard(destination,safe_path(source).stat().st_size)
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
                   '--owner-pid',str(os.getpid()),'--inputs',binding['inputs']['file'],'--split',args.split,'--baseline-models',str(safe_path(args.baseline_models)),
                   '--output',str(paths['predictions']),'--jobs',str(paths['jobs']),'--jobs-sha256',common.digest(paths['jobs']),
                   '--config',str(paths['config']),'--binding',str(paths['binding']),'--binding-sha256',common.digest(paths['binding'])]
        for path in args.candidate: command += ['--candidate',str(safe_path(path))]
        if getattr(args,'classifier',None): command += ['--classifier',str(safe_path(args.classifier))]
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
        proof=verify_worker_proof(resources['resources'].get('worker_ownership'),process.pid,os.getpid())
        if resources['resources'].get('worker_pid')!=proof['worker_pid']:raise ValueError('Actual CPU worker PID differs from its verified parent chain')
        resources['resources']['worker_exit_code']=code
        resources['resources']['capture']='successful-owned-subprocess'
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
    if split==REGRESSION_SPLIT:
        if demotion_receipt(root) is None: raise ValueError('Heldout regression truth requires a registered demotion sidecar')
        split='heldout'; allow_heldout=True
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
            if region['kind'] not in ('body','kaomoji','graphic','unsupported'): raise ValueError('Unknown scene truth region')
            if region['kind']!='graphic':
                a,b=region['span']
                if not 0<=a<b<=len(row['text']) or row['text'][a:b]!=region['text']: raise ValueError('Exact truth span changed')
            if region['kind']=='unsupported':
                # A face cut by a line wrap: counted, never scored. Each part binds its whole source face.
                source=region.get('source_face'); cut=region.get('cut')
                if (region.get('part') not in ('left','right') or not isinstance(source,str) or type(cut) is not int or not 0<cut<len(source)
                        or region['text']!=(source[:cut] if region['part']=='left' else source[cut:])):
                    raise ValueError('Unknown or incomplete unsupported truth part')
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


def negative_scene_kind(kind):
    """Scenes that carry no face at all: blank, graphic-only, or a layout family's declared '-negative' kind."""
    return kind in ('blank','graphic-only') or (isinstance(kind,str) and kind.endswith('-negative'))


def negative_scene_prefix(kind):
    return 'blank' if kind=='blank' else ('graphic_only' if kind=='graphic-only' else kind.replace('-','_'))


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
                    or mask_pixels(body_mask, item.get('regional_ink_bbox', item['regional_bbox']))):
                return False, 0
            splices += 1
        else: return False, 0
        consumed.add(index)
    return consumed == expected and splices > 0, splices


def overlap_box(item):
    """Box used for body-mask overlap: replaced faces count by their ink, not by their padded ROI."""
    if item.get('source') == 'specialist-exact-views' and item.get('ink_bbox'):
        return item['ink_bbox']
    return item['bbox']


def split_children_keep_parent_reading(overlapping, production):
    """True when every region touching a body reproduces the production reading.

    Each region must be an untouched production region or a split child whose
    bytes are a contiguous run of its production parent's text: a boundary
    change, not a new reading. Any replacement, splice or restored region, or
    an empty overlap, returns False.
    """
    if not overlapping:
        return False
    for item in overlapping:
        source = item.get('source', 'baseline')
        if source == 'baseline-split-child':
            parent = item.get('split_parent_index')
            if type(parent) is not int or not 0 <= parent < len(production):
                return False
            if not item['text'] or item['text'] not in production[parent]['text']:
                return False
        elif source == 'baseline':
            index = item.get('baseline_index')
            if type(index) is not int or not 0 <= index < len(production):
                return False
            if item['text'] != production[index]['text'] or item['bbox'] != production[index]['bbox']:
                return False
        else:
            return False
    return True


def effective_face_change(model, baseline_scene_text, mask):
    """Post-inference metric only: accepted, changed face bytes in this ink region."""
    if model['text']==baseline_scene_text:return False
    for decision in model.get('decisions',[]):
        if decision.get('accepted') is not True or not mask_pixels(mask,decision['bbox']):continue
        splice=decision.get('splice') or {}
        old=splice.get('original_face_text',decision.get('original_text'))
        new=decision.get('specialist_text')
        if isinstance(old,str) and isinstance(new,str) and old!=new:return True
    return False


def summarize(root,rows,predictions,config,model_ids):
    if len(rows) != len(predictions): raise ValueError('Scene prediction count differs')
    localization=Counter(); candidates_by_source=Counter(); baseline_pairs=[]; pairs={key:[] for key in model_ids}
    body={key:Counter() for key in model_ids}; faces={key:[] for key in model_ids}; base_faces=[]
    negatives={key:Counter() for key in model_ids}; crop_pairs={key:[] for key in model_ids}
    kind_counts=Counter(); details=[]; ious=[]; coverage_values=[]; effectiveness={key:Counter(actual_routed_candidates=0,face_wrong_to_exact=0,isolated_face_wrong_to_exact=0,face_exact_to_wrong=0) for key in model_ids}
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
        first_pass=prediction.get('first_pass',{})
        localization['split_parents']+=int(first_pass.get('split_parents',0))
        localization['split_children']+=int(first_pass.get('split_children',0))
        localization['structural_components']+=int(prediction['routing'].get('structural_components',0))
        for identity in model_ids:
            effectiveness[identity]['actual_routed_candidates']+=sum(bool(views) for views in prediction['models'][identity]['views'].values())
            split_stats=prediction['models'][identity].get('split',{})
            effectiveness[identity]['split_materialized']+=int(split_stats.get('materialized',0))
            effectiveness[identity]['split_restored']+=int(split_stats.get('restored',0))
            effectiveness[identity]['split_restored_unverifiable']+=int(split_stats.get('restored_unverifiable',0))
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
                    improved=baseline_text!=truth['text'] and final_text==truth['text']
                    effectiveness[identity]['isolated_face_wrong_to_exact']+=int(improved)
                    effectiveness[identity]['face_wrong_to_exact']+=int(improved and effective_face_change(prediction['models'][identity],prediction['baseline_text'],mask))
                    effectiveness[identity]['face_exact_to_wrong']+=int(baseline_text==truth['text'] and final_text!=truth['text'])
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
                    overlapping=[item for item in prediction['models'][identity]['regions'] if mask_pixels(mask,overlap_box(item))]
                    unchanged_reading=split_children_keep_parent_reading(overlapping,prediction['baseline'])
                    counter['erroneous_rewrites']+=int(final_text!=baseline_text and final_text!=truth['text'] and not unchanged_reading)
                    counter['split_children_with_unchanged_reading']+=int(unchanged_reading)
                    before=joined_regions([item for item in prediction['baseline'] if mask_pixels(mask,item['bbox'])],config)
                    after=joined_regions(overlapping,config)
                    preserved, splice_count = body_preserving_splices(prediction['baseline'],
                                                prediction['models'][identity]['regions'], mask)
                    # This post-inference deletion tripwire uses a whole raw label;
                    # it never extracts a predicted span or changes model coordinates.
                    literal_deleted = bool(truth['text'] and truth['text'] in before and truth['text'] not in after)
                    counter['body_intersection_transcript_changes']+=int(before!=after)
                    counter['body_literal_deleted']+=int(literal_deleted)
                    counter['verified_body_preserving_splices']+=splice_count if preserved and not literal_deleted else 0
                    counter['new_body_intersection_errors']+=int(literal_deleted or
                                                               (before!=after and after!=truth['text'] and final_text!=truth['text']
                                                                and not preserved and not unchanged_reading))
            elif truth['kind']=='unsupported':
                # Cut-face parts: discovery is reported, nothing is scored, no population includes them.
                localization['unsupported_regions']+=1
                localization['unsupported_regions_with_candidate']+=int(any(mask_pixels(mask,item['bbox']) for item in routing))
                localization['unsupported_regions_routed']+=int(any(mask_pixels(mask,item['bbox']) for item in routed))
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
        if negative_scene_kind(row['scene_kind']):
            localization['negative_scenes']+=1
            localization['negative_scenes_with_candidate']+=int(bool(routing))
            localization['negative_scenes_routed']+=int(bool(routed))
            for identity in model_ids:
                negative=negatives[identity];negative['negative_scenes']+=1
                negative['baseline_nonempty_scenes']+=int(bool(prediction['baseline_text']))
                negative['final_nonempty_scenes']+=int(bool(prediction['models'][identity]['text']))
                negative['new_nonempty_scenes']+=int(not prediction['baseline_text'] and bool(prediction['models'][identity]['text']))
                prefix=negative_scene_prefix(row['scene_kind'])
                negative[prefix+'_scenes']+=1
                negative[prefix+'_baseline_nonempty']+=int(bool(prediction['baseline_text']))
                negative[prefix+'_final_nonempty']+=int(bool(prediction['models'][identity]['text']))
                negative[prefix+'_new_nonempty']+=int(not prediction['baseline_text'] and bool(prediction['models'][identity]['text']))
        details.append(scene_detail)
    faces_total=localization.get('faces',0)
    candidate_stage={'faces':faces_total,'safe_complete_coverage':localization.get('safe_complete_coverage',0),
                     'safe_complete_coverage_ratio':(localization.get('safe_complete_coverage',0)/faces_total) if faces_total else None,
                     'complete_ink_coverage':localization.get('complete_ink_coverage',0),'partial_only':localization.get('partial_only',0),
                     'routed_candidate_count':localization.get('routed_candidate_count',0),
                     'safe_routed_complete_coverage':localization.get('safe_routed_complete_coverage',0),
                     'body_regions_routed':localization.get('body_regions_routed',0),
                     'bracket_pair_candidates':candidates_by_source.get('image-bracket-pair',0),
                     'structural_components':localization.get('structural_components',0),
                     'unsupported_regions':localization.get('unsupported_regions',0),
                     'note':'Candidate-stage development metrics; unsupported cut-face parts are counted here and excluded from every scored population.'}
    result={'scenes':len(rows),'scene_kinds':dict(kind_counts),'localization':dict(localization),
            'candidate_stage':candidate_stage,
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
          'body':dict(protection),'negative':dict(negative),'routing_effectiveness':dict(effectiveness[identity]),
          'quality_gates':{'no_new_body_errors':not (protection['baseline_correct_to_wrong'] or protection['erroneous_rewrites'] or protection['new_body_intersection_errors']),
                           'no_new_negative_outputs':not (negative['new_nonempty_scenes'] or negative['new_graphic_nonempty']),
                           'localization_recall_threshold':'not specified; measured only',
                           'app_acceptance':False,'production_enabled':False}}
        if result['baseline']['isolated_kaomoji']['count'] and protection['regions'] and negative['negative_scenes'] and negative['graphic_regions']:
            # Explicit zero counters: missing fields cannot silently waive protection.
            for key in ('baseline_correct_to_wrong','erroneous_rewrites','new_body_intersection_errors','body_literal_deleted'):result['models'][identity]['body'].setdefault(key,0)
            for key in ('new_nonempty_scenes','new_graphic_nonempty'):result['models'][identity]['negative'].setdefault(key,0)
            result['models'][identity]['research_quality_gates']=scene_gate(result,identity)
    return result,details


def evaluate(args):
    root,output=safe_path(args.dataset),output_path(args.output)
    if output.exists(): raise ValueError('Evaluation output already exists')
    config=configuration(args.config, classifier=getattr(args,'classifier',None)); binding=pipeline_binding(root,args.candidate,args.baseline_models,config,args.inputs)
    usage=None; demotion=None
    if args.split=='heldout':
        if not args.selection: raise ValueError('Heldout requires frozen whole-scene selection before any images or truth')
        usage=claim_heldout(root,safe_path(args.selection),binding,output,getattr(args,'dual_freeze',None))
    elif args.split==REGRESSION_SPLIT:
        demotion=demotion_receipt(root)
        if demotion is None: raise ValueError('Heldout regression requires a registered demotion sidecar for this viewed heldout')
        if rp.profile_for(output) is None or rp.profile_for(output).name!=demotion['round']:
            raise ValueError('Heldout regression output must belong to the demoting round')
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
                'independent_heldout':args.split=='heldout','heldout_usage':usage,'demotion':demotion,'promote':False,'enabled':False,
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
    config=configuration(args.config, classifier=getattr(args,'classifier',None)); binding=pipeline_binding(args.dataset,args.candidate,args.baseline_models,config,args.inputs)
    final_families(binding); report=verify_dev_evidence(report_path,binding)
    if args.selected_id not in {item['id'] for item in binding['models']} or not args.reason.strip():
        raise ValueError('Frozen research selection needs a measured model identity and explicit development reason')
    budget=safe_path(args.budget_receipt); validate_budget(common.read(budget), budget)
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
        command.add_argument('--inputs'); command.add_argument('--config'); command.add_argument('--classifier')
        if name in ('evaluate','_worker'):
            command.add_argument('--split',choices=('dev','heldout',REGRESSION_SPLIT),default='dev')
        if name=='evaluate':
            command.add_argument('--selection'); command.add_argument('--dual-freeze')
        elif name=='_worker':
            for key in ('jobs','jobs-sha256','binding','binding-sha256'): command.add_argument('--'+key,required=True)
            command.add_argument('--owner-pid',type=int,required=True)
            command.add_argument('--usage-sha256')
        else:
            for key in ('dev-report','budget-receipt','selected-id','reason'): command.add_argument('--'+key,required=True)
    probe=commands.add_parser('_ownership-probe',help=argparse.SUPPRESS)
    probe.add_argument('--owner-pid',type=int,required=True)
    args=parser.parse_args(argv)
    if args.action=='_ownership-probe':
        print(json.dumps({'schema_version':'localocr.cpu-owner-probe.v1','proof':owned_worker_proof(args.owner_pid,'_ownership-probe'),'model_inference':False,'GPU_used':False}));return 0
    if args.action=='_worker': worker(args); return 0
    value=evaluate(args) if args.action=='evaluate' else freeze_selection(args)
    print(json.dumps({'schema_version':value['schema_version'],'split':value.get('split'),'enabled':False,
                      'scenes':value.get('summary',{}).get('scenes')},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
