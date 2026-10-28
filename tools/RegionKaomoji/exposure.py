"""Bound, label-only cross-round exposure exclusions for new research datasets.

``build([{'dataset': old_root, 'split': 'dev'}, ...], inventory_path)`` freezes
the exact source manifests. A consumed heldout additionally requires its
``first_use`` receipt path. No image, prediction or unused heldout is opened.
Only the outer bracket pair is width-normalized for exclusion; recognition and
the grammar's partition function keep their original Unicode semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from tools.PretrainedOcr import grammar_corpus as gc, kaomoji_grammar as kg

SCHEMA = 'localocr.cross-round-exposure.v1'
TARGET_TEXT_SCOPE = 'nonempty crop labels and crop neighbours; scene whole text, faces and comment body; fixed UI chrome excluded'
CORE_POLICY = 'NFKC of outer bracket pair only; internal core codepoints unchanged'
ATOMS = kg.default_atoms()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized_core(text, key=None):
    """Core with width-normalized outer brackets; interior codepoints stay exact."""
    if key is None:
        parsed = kg.parse_face(text, ATOMS)
        if parsed is None:
            return None
        key = gc.core_key(parsed['bracket'], *parsed['core'])
    if not isinstance(key, dict) or set(key) != {'bracket', 'eye1', 'mouth', 'eye2'}:
        raise ValueError('Invalid exposure core key')
    bracket = key['bracket']
    if (not isinstance(bracket, (list, tuple)) or len(bracket) != 2
            or any(not isinstance(c, str) or len(c) != 1 for c in bracket)
            or any(not isinstance(key[k], str) for k in ('eye1', 'mouth', 'eye2'))):
        raise ValueError('Invalid exposure core key')
    return {**key, 'bracket': [unicodedata.normalize('NFKC', c) for c in bracket]}


def core_sha(text, key=None):
    """Width-sibling core digest, or None for text outside the face grammar."""
    core = normalized_core(text, key)
    return gc.core_digest(core) if core is not None else None


def split_matches(text, split, key=None):
    """Keep original partition ownership and require its width sibling to agree.

    No key is reassigned: a disagreement rejects a proposed new row. Source
    composition keys and independently parsed keys must both agree when present.
    """
    parsed = normalized_core(text)
    composed = normalized_core(text, key) if key is not None else None
    return all(gc.partition(core) == split for core in (parsed, composed) if core is not None)


def targets(row):
    """Yield only quality-target text, with any original composition key.

    Repeated fixed UI fields are protection controls, not new target vocabulary.
    Crop neighbours are included because they supply the target's body context.
    Empty labels have no exclusion identity, so blank controls remain possible.
    """
    if not isinstance(row, dict):
        raise ValueError('Exposure source rows must be objects')
    if 'text' in row:
        yield row['text'], row.get('core_key')
    for region in row.get('regions', []):
        if region.get('kind') in ('face', 'kaomoji') or region.get('kind') == 'body' and region.get('column') in (None, 'comment'):
            yield region.get('text', ''), region.get('core_key')
    neighbours = row.get('transform', {}).get('neighbours', {})
    for side in ('left', 'right'):
        if side in neighbours:
            yield neighbours[side], None


def _source(spec):
    """Validate read authorization before opening a split's label bytes."""
    if not isinstance(spec, dict) or set(spec) - {'dataset', 'split', 'first_use'} or not {'dataset', 'split'} <= set(spec):
        raise ValueError('Exposure sources need dataset, split and optional first_use')
    root, split = Path(spec['dataset']).resolve(), spec['split']
    if split not in ('train', 'dev', 'heldout'):
        raise ValueError('Exposure source split must be train, dev or heldout')
    meta_path = root / 'dataset.json'
    meta = read(meta_path)
    freeze_path = root / 'freeze.json'
    frozen = read(freeze_path)
    if (frozen.get('dataset_sha256') != sha(meta_path)
            or frozen.get('split_sha256') != {s: info.get('sha256') for s, info in meta.get('splits', {}).items()}):
        raise ValueError('Exposure source freeze or dataset metadata changed')
    info = meta.get('splits', {}).get(split)
    if not isinstance(info, dict) or not {'file', 'sha256'} <= set(info):
        raise ValueError('Exposure source split metadata missing')
    relative = Path(info['file'])
    manifest = (root / relative).resolve()
    if relative.is_absolute() or relative.drive or not manifest.is_relative_to(root):
        raise ValueError('Exposure source manifest escapes its dataset')
    if relative.stem in ('train', 'dev', 'heldout') and relative.stem != split:
        raise ValueError('Exposure source split aliases another manifest')
    for other_split, other in meta['splits'].items():
        if other_split != split and ((root / other['file']).resolve() == manifest or other['sha256'] == info['sha256']):
            raise ValueError('Exposure source split aliases another manifest')
    receipt = None
    if split == 'heldout':
        if not spec.get('first_use'):
            raise ValueError('Exposure heldout must have a bound consumed first-use receipt')
        receipt_path = Path(spec['first_use']).resolve()
        claim = read(receipt_path)
        schema = claim.get('schema_version')
        binding = (claim.get('dataset_binding') if schema == 'localocr.pretrained.heldout-first-use.v1'
                   else claim.get('pipeline_binding', {}).get('dataset') if schema == 'localocr.region-heldout-first-use.v1' else None)
        if (binding != frozen or frozen.get('dataset_sha256') != sha(meta_path)
                or frozen.get('split_sha256', {}).get(split) != info['sha256']
                or not isinstance(claim.get('status'), str)
                or not claim['status'].startswith(('consumed-before-reading', 'claimed-before-reading-heldout'))):
            raise ValueError('Exposure heldout lacks a consumed receipt for this source')
        receipt = {'file': str(receipt_path), 'sha256': sha(receipt_path)}
    actual = sha(manifest)
    if actual != info['sha256']:
        raise ValueError('Exposure source manifest SHA changed')
    rows = read(manifest)
    if not isinstance(rows, list):
        raise ValueError('Exposure source manifest must contain a row list')
    binding = {'dataset': str(root), 'split': split, 'manifest': {'file': str(manifest), 'sha256': actual},
               'dataset_metadata': {'file': str(meta_path), 'sha256': sha(meta_path)},
               'freeze': {'file': str(freeze_path), 'sha256': sha(freeze_path)}, 'rows': len(rows)}
    if receipt is not None:
        binding['first_use'] = receipt
    return binding, rows


def _collect(sources):
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError('Exposure inventory needs at least one source manifest')
    bindings, texts, cores, seen = [], set(), set(), set()
    for spec in sources:
        binding, rows = _source(spec)
        identity = binding['manifest']['file']
        if identity in seen:
            raise ValueError('Duplicate exposure source manifest')
        seen.add(identity)
        bindings.append(binding)
        for row in rows:
            for text, key in targets(row):
                if not isinstance(text, str):
                    raise ValueError('Exposure source text must be a string')
                if text:
                    texts.add(text_sha(text))
                    # Source-composed and parser-derived cores can differ for decorated faces.
                    # Keep both so neither an exact label nor a width sibling escapes the inventory.
                    for core in (core_sha(text), core_sha(text, key) if key is not None else None):
                        if core is not None:
                            cores.add(core)
    return {'schema_version': SCHEMA, 'target_text_scope': TARGET_TEXT_SCOPE, 'core_policy': CORE_POLICY,
            'sources': bindings, 'raw_text_sha256': sorted(texts), 'core_sha256': sorted(cores)}


def build(sources, output):
    """Write a fresh immutable inventory and return its absolute Path.

    Each source is ``{'dataset': path, 'split': 'train'|'dev'|'heldout'}``.
    Heldout descriptors must additionally name their existing ``first_use``.
    """
    value = _collect(sources)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write('\n')
    return output


@dataclass(frozen=True)
class Exposure:
    binding: dict
    raw_texts: frozenset
    cores: frozenset
    word_runtime: dict = field(default_factory=lambda: {'lengths': Counter(), 'single_pool_exposed': None}, compare=False, repr=False)

    def matches(self, text, key=None):
        if not text:
            return False
        return (text_sha(text) in self.raw_texts or core_sha(text) in self.cores
                or key is not None and core_sha(text, key) in self.cores)

    def assert_clear(self, row):
        for text, key in targets(row):
            if self.matches(text, key):
                raise ValueError('Generated target intersects cross-round exposure exclusions')
            if 'split' in row and not split_matches(text, row['split'], key):
                raise ValueError('Generated target lacks raw-and-width-sibling split agreement')

    def words(self, rng, requested, atoms, seen=None, strip=False):
        """Strict new word combinations, extending an exhausted request up to six atoms.

        Only the existing comment vocabulary is used. A completely exposed one-atom
        pool can be proved by enumeration and cached; larger lengths use the same
        bounded rejection strategy as the original generators. Requested lengths
        above six retain their requested size and cannot expand further.
        """
        from . import card_scenes as cs
        if self.word_runtime['single_pool_exposed'] is None:
            singletons = (*cs.CN_WORDS, *cs.EN_WORDS, *cs.PUNCTUATION, *(str(n) for n in range(1, 100)))
            self.word_runtime['single_pool_exposed'] = all(self.matches(t.strip() if strip else t) for t in singletons)
        for actual in range(requested, max(requested, 6) + 1):
            if actual == 1 and self.word_runtime['single_pool_exposed']:
                continue
            for _ in range(1000):
                text = cs.comment_words(rng, actual)
                if strip:
                    text = text.strip()
                digest = text_sha(text)
                if (text.strip() and not self.matches(text) and not cs.contains_face(text, atoms)
                        and (seen is None or digest not in seen)):
                    if seen is not None:
                        seen.add(digest)
                    self.word_runtime['lengths'][f'{requested}->{actual}'] += 1
                    return text
        raise ValueError('Original comment words exhausted after bounded length recovery')

    def word_recovery_summary(self):
        return {'maximum_recovery_words': 6, 'maximum_attempts_per_length': 1000,
                'population': 'accepted planning requests including later discarded overflow/retry plans',
                'single_word_pool_fully_exposed': self.word_runtime['single_pool_exposed'],
                'requested_to_actual_word_counts': dict(sorted(self.word_runtime['lengths'].items()))}


def load(path):
    """Re-derive the inventory from its SHA-bound, authorized source labels."""
    path = Path(path).resolve()
    value = read(path)
    if (not isinstance(value, dict) or value.get('schema_version') != SCHEMA
            or not isinstance(value.get('sources'), list)
            or any(not isinstance(s, dict) or not {'dataset', 'split'} <= set(s) for s in value['sources'])):
        raise ValueError('Unknown exposure inventory schema')
    # Check *all* bound source metadata before _collect may open any label bytes.
    # In particular a changed dev.file must never redirect us into a sealed heldout.
    for source in value['sources']:
        root = Path(source['dataset']).resolve()
        for field, expected_path in (('dataset_metadata', root / 'dataset.json'), ('freeze', root / 'freeze.json')):
            ref = source.get(field, {})
            if (not isinstance(ref, dict) or set(ref) != {'file', 'sha256'}
                    or Path(ref['file']).resolve() != expected_path or sha(expected_path) != ref['sha256']):
                raise ValueError('Exposure source metadata/freeze binding changed')
        if 'first_use' in source:
            ref = source['first_use']
            if not isinstance(ref, dict) or set(ref) != {'file', 'sha256'} or sha(ref['file']) != ref['sha256']:
                raise ValueError('Exposure source first-use binding changed')
        meta = read(root / 'dataset.json')
        info = meta.get('splits', {}).get(source['split'], {})
        ref = source.get('manifest', {})
        if (not isinstance(ref, dict) or set(ref) != {'file', 'sha256'} or 'file' not in info
                or (root / info['file']).resolve() != Path(ref['file']).resolve() or info.get('sha256') != ref['sha256']):
            raise ValueError('Exposure source manifest binding changed')
    specs = [{'dataset': s['dataset'], 'split': s['split'],
              **({'first_use': s['first_use']['file']} if 'first_use' in s else {})} for s in value.get('sources', [])]
    if _collect(specs) != value:
        raise ValueError('Exposure inventory or source binding changed')
    return Exposure({'file': str(path), 'sha256': sha(path), 'target_text_scope': TARGET_TEXT_SCOPE,
                     'core_policy': CORE_POLICY}, frozenset(value['raw_text_sha256']), frozenset(value['core_sha256']))


def load_bound(binding):
    if (not isinstance(binding, dict) or set(binding) != {'file', 'sha256', 'target_text_scope', 'core_policy'}
            or sha(binding['file']) != binding['sha256']):
        raise ValueError('Bound exposure inventory SHA changed')
    loaded = load(binding['file'])
    if loaded.binding != binding:
        raise ValueError('Bound exposure inventory policy changed')
    return loaded
