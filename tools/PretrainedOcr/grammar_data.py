"""Grammar-round crop rasters: renderable atom projection, hashed core isolation, declared effects.

Train, development and heldout share every structural form; exact labels, core
combinations and decoded pixels stay disjoint. Effects follow a declared
distribution per split (set A for train/dev, independent set B for heldout).
The dataset keeps the pretrained dataset schema so existing readers work.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
from tools.OcrDatasets.render import Unsupported, validate_effects
from .data import MixedRenderer, pixels, sha, read
from .common import write as atomic_write, utc
from . import kaomoji_grammar as kg, grammar_corpus as gc, round_profile as rp
from . import quality_data as qd
from .quality_corpus import SPLITS, expressions as v1_expressions, NEGATIVE_FORMS

GENERATOR_ID = 'localocr.grammar-data.20261013.v1'
SIZES = (20, 24, 28, 32)
SPLIT_EFFECT_SETS = {'train': 'A', 'dev': 'A', 'heldout': 'B'}
EFFECT_SETS = {
    'A': {'outline': {'probability': .35, 'widths': (1,)}, 'shadow': {'probability': .35, 'offsets': (1, 2), 'alpha': (120, 200)},
          'texture': {'probability': .30, 'amplitudes': (4, 8)}, 'blur': {'probability': .30, 'values': (.3, .5)},
          'jpeg': {'probability': .30, 'quality': (80, 95)}, 'dark': {'probability': .30}},
    'B': {'outline': {'probability': .40, 'widths': (2,)}, 'shadow': {'probability': .40, 'offsets': (2, 3), 'alpha': (90, 170)},
          'texture': {'probability': .40, 'amplitudes': (6, 12)}, 'blur': {'probability': .30, 'values': (.4, .7)},
          'jpeg': {'probability': .40, 'quality': (70, 85)}, 'dark': {'probability': .40}},
}


def previous_exclusions():
    """Labels of every earlier research dataset, including the demoted quality-round corpus."""
    seen, bindings = qd.previous_exclusions()
    folder = qd.ARTIFACTS / 'quality-optimization-20261012/data/corpus-v3'
    for name in ('train', 'dev', 'heldout'):
        path = folder / (name + '.json')
        if not path.exists():
            continue
        bindings[path.relative_to(qd.ARTIFACTS).as_posix()] = sha(path)
        for row in read(path):
            if row.get('text'):
                seen.add(gc.text_hash(row['text']))
    return seen, bindings


def renderable_atoms(renderer, atoms):
    """Atoms whose glyph clusters every registered font run can shape; exclusions are returned, never hidden."""
    renderable, excluded = set(), []
    candidates = {atom for role in kg.LIST_ROLES if role != 'gaps' for atom in getattr(atoms, role)}
    candidates.update(atoms.bracket_chars)
    for atom in sorted(candidates):
        try:
            renderer.shape(atom)
            renderable.add(atom)
        except Unsupported as error:
            excluded.append({'atom': atom, 'codepoints': [f'U+{ord(c):04X}' for c in atom], 'reason': str(error)})
    return gc.allowed_atoms(atoms, renderable), excluded


def draw_effects(split, rng):
    """Sample one declared effect configuration for a group; both views share it."""
    name = SPLIT_EFFECT_SETS[split]
    spec = EFFECT_SETS[name]
    dark = rng.random() < spec['dark']['probability']
    background = tuple(rng.randrange(28, 62) for _ in range(3)) if dark else tuple(rng.randrange(222, 253) for _ in range(3))
    ink = tuple(rng.randrange(198, 242) for _ in range(3)) if dark else tuple(rng.randrange(8, 62) for _ in range(3))
    effects = {'background': background, 'ink': ink}
    if rng.random() < spec['outline']['probability']:
        effects['outline'] = rng.choice(spec['outline']['widths'])
        effects['outline_color'] = (0, 0, 0) if dark else (96, 96, 96)
    if rng.random() < spec['shadow']['probability']:
        low, high = spec['shadow']['offsets']
        effects['shadow'] = {'dx': rng.randint(low, high), 'dy': rng.randint(low, high), 'color': (0, 0, 0),
                             'alpha': rng.randint(*spec['shadow']['alpha'])}
    if rng.random() < spec['texture']['probability']:
        effects['texture'] = {'amplitude': rng.choice(spec['texture']['amplitudes']), 'seed': rng.randrange(1_000_000)}
    blur = rng.choice(spec['blur']['values']) if rng.random() < spec['blur']['probability'] else 0.0
    jpeg = rng.randint(*spec['jpeg']['quality']) if rng.random() < spec['jpeg']['probability'] else None
    return {'effect_set': name, 'effects': validate_effects(effects), 'blur': blur, 'jpeg': jpeg, 'dark': dark}


def render_view(renderer, text, view, size, draw):
    """Render one view with declared effects; keep label whitespace advances, drop artificial margins."""
    image, region, fonts, evidence, alpha = renderer.render(text, size=size, dark=draw['dark'], blur=draw['blur'],
                                                            jpeg=draw['jpeg'], effects=draw['effects'], with_alpha=True)
    x, y, w, h = region
    pad = 3
    def advance(value):
        return sum(p.x_advance / f[5] * size for f, _, positions in renderer.shape(value) for p in positions) if value else 0
    leading = text[:len(text) - len(text.lstrip(' '))]
    trailing = text[len(text.rstrip(' ')):]
    left = max(0, int(x - pad - advance(leading)))
    right = min(image.width, int(np.ceil(x + w + pad + advance(trailing))))
    top, bottom = max(0, y - pad), min(image.height, y + h + pad)
    image = image.crop((left, top, right, bottom))
    region = [x - left, y - top, w, h]
    if view:
        padded = Image.new('RGB', (image.width + 12, image.height + 12), image.getpixel((0, 0)))
        padded.paste(image, (6, 6))
        image = padded
        region[0] += 6
        region[1] += 6
    evidence = dict(evidence, whitespace_normalization='none', view='edge-pad-6' if view else 'raw-ink-pad-3')
    return image, region, fonts, evidence


def _rows_core_digests(rows):
    return {row['core_digest'] for row in rows if row.get('kind') == 'face'}


def validate(root):
    root = Path(root).resolve()
    meta = read(root / 'dataset.json')
    if meta.get('schema_version') != 'localocr.pretrained.dataset.v1' or meta.get('generator_id') != GENERATOR_ID or meta.get('synthetic_only') is not True:
        raise ValueError('Unknown grammar dataset schema/generator')
    atoms = kg.default_atoms()
    if meta.get('atoms_binding') != kg.binding(atoms):
        raise ValueError('Dataset atoms binding differs from the canonical table')
    texts, groups, cores, pixelsets, counts, tokens = {}, {}, {}, {}, {}, set()
    for split in SPLITS:
        info = meta['splits'][split]
        relative = Path(info['file'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise ValueError('Split path escape')
        if sha(root / relative) != info['sha256']:
            raise ValueError('Split changed')
        rows = read(root / relative)
        if len(rows) != info['count']:
            raise ValueError('Split count changed')
        for row in rows:
            if row['split'] != split or row['provenance'] != {'source': 'original', 'license': 'CC0-1.0'}:
                raise ValueError('Split/provenance differs')
            if gc.text_hash(row['text']) != row['text_sha256']:
                raise ValueError('Exact Unicode label changed')
            if row['kind'] == 'face':
                signature = gc.structure_signature(row['roles'])
                if (row['recipe_expression'] != signature or row['family'] != gc.family(signature)
                        or gc.core_digest(row['core_key']) != row['core_digest'] or gc.partition(row['core_key']) != split
                        or row['core_partition'] != split or kg.parse_face(row['text'], atoms) is None):
                    raise ValueError('Face structure, core partition or grammar identity differs')
                if row['core_digest'] in cores and cores[row['core_digest']] != split:
                    raise ValueError('Cross-split core combination leakage')
                cores[row['core_digest']] = split
            elif row['kind'] in ('blank', 'graphic'):
                if row['recipe_expression'] != NEGATIVE_FORMS[split][row['kind']]:
                    raise ValueError('Negative recipe differs')
            elif row['recipe_expression'] not in v1_expressions(split):
                raise ValueError('Ordinary recipe AST differs')
            for key, inventory in ((row['group_id'], groups), (row['text_sha256'] if row['text'] else None, texts)):
                if key is not None:
                    if key in inventory and inventory[key] != split:
                        raise ValueError('Cross-split label/group leakage')
                    inventory[key] = split
            path = (root / row['image']).resolve()
            if not path.is_relative_to(root) or sha(path) != row['sha256']:
                raise ValueError('Image path/hash changed')
            with Image.open(path) as image:
                image.load()
                pixel = pixels(image)
                if image.size != (row['width'], row['height']) or pixel != row['pixel_sha256']:
                    raise ValueError('Decoded pixels changed')
            if pixel in pixelsets and pixelsets[pixel] != split:
                raise ValueError('Cross-split pixels')
            pixelsets[pixel] = split
            transform = row['transform']
            if transform.get('effect_set') != SPLIT_EFFECT_SETS[split]:
                raise ValueError('Effect set does not match the split declaration')
            validate_effects(transform.get('effects'))
            if row['text'] and (not row['font_id'] or row['rendering']['glyphs'] <= 0):
                raise ValueError('Missing genuine glyphs')
            if split == 'train':
                tokens.update(row['text'])
        count_groups = len({r['group_id'] for r in rows})
        positive = len({r['group_id'] for r in rows if r['text']})
        if count_groups != info['groups'] or positive != info['positive_groups']:
            raise ValueError('Group count differs')
        counts[split] = {'count': len(rows), 'positive_groups': positive, 'groups': count_groups}
    if sorted(tokens, key=ord) != meta['train_vocabulary']:
        raise ValueError('Vocabulary must derive from train only')
    binding = {'schema_version': 'localocr.pretrained.freeze.v1', 'dataset_sha256': sha(root / 'dataset.json'),
               'split_sha256': {s: meta['splits'][s]['sha256'] for s in SPLITS}, 'counts': counts, 'license': 'CC0-1.0'}
    if (root / 'freeze.json').exists() and read(root / 'freeze.json') != binding:
        raise ValueError('Freeze changed')
    if (root / 'heldout-seal.json').exists():
        seal = read(root / 'heldout-seal.json')
        if seal['dataset_sha256'] != binding['dataset_sha256'] or seal['heldout_sha256'] != binding['split_sha256']['heldout']:
            raise ValueError('Heldout seal differs')
    return binding


def generate(output, fonts, counts=(8000, 1000, 2000), variants=(2, 2, 2), seed=20261013):
    if len(counts) != 3 or len(variants) != 3 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (*counts, *variants)):
        raise ValueError('Positive split counts/variant counts required')
    root = qd.output_root(output)
    if rp.profile_for(root) is None:
        raise ValueError('Grammar data must be generated inside a registered research round')
    if root.exists():
        raise ValueError('Fresh output required')
    qd.storage_gate(root, 1024 ** 2)
    renderer = MixedRenderer(fonts)
    atoms = kg.default_atoms()
    allowed, excluded_atoms = renderable_atoms(renderer, atoms)
    seen, old_bindings = previous_exclusions()
    excluded_prior_count = len(seen)
    root.mkdir(parents=True)
    (root / 'images').mkdir()
    writer = qd.Writer(root)
    started = time.perf_counter()
    splits, allpixels, exclusions, tokens, summaries = {}, {}, Counter(), set(), {}
    train_counts = None
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'started_utc': utc(), 'counts': list(counts), 'seed': seed, 'completed_splits': []})
    for split_index, split in enumerate(SPLITS):
        rows = []
        rng = random.Random(seed + split_index)
        kind_counts = Counter()
        for index in range(counts[split_index]):
            for attempt in range(1000):
                item = gc.make_sample(split, index, rng, seen, atoms, allowed, train_counts)
                draw = draw_effects(split, rng)
                rendered = []
                try:
                    for view in range(variants[split_index]):
                        size = SIZES[index % len(SIZES)]
                        if item['text']:
                            image, region, used, evidence = render_view(renderer, item['text'], view, size, draw)
                            transform = {'size': size, 'view': 'edge-pad-6' if view else 'raw-ink-pad-3', 'whitespace': 'exact original advance',
                                         'effect_set': draw['effect_set'], 'effects': evidence['effects'], 'blur': draw['blur'],
                                         'jpeg': draw['jpeg'], 'dark': draw['dark']}
                        else:
                            image, negative_transform = qd.negative(split, index, view, rng)
                            region = [0, 0, image.width, image.height]
                            used = []
                            evidence = {'text_present': False}
                            transform = {**negative_transform, 'effect_set': draw['effect_set'], 'effects': {}, 'blur': 0.0, 'jpeg': None, 'dark': False}
                        pixel = pixels(image)
                        if pixel in allpixels and allpixels[pixel] != split:
                            raise Unsupported('cross-split-pixels')
                        rendered.append((image, region, used, evidence, transform, pixel))
                except Unsupported as error:
                    exclusions[str(error)] += 1
                    if item['text']:
                        seen.discard(item['text_sha256'])
                    continue
                break
            else:
                raise ValueError('Unable to render requested original group')
            for view, (image, region, used, evidence, transform, pixel) in enumerate(rendered):
                identifier = f'{split}-{index:06d}-{view}'
                path = root / 'images' / (identifier + '.png')
                writer.image(path, image)
                allpixels[pixel] = split
                rows.append(dict(item, id=identifier, split=split, image=path.relative_to(root).as_posix(), font_id=used,
                                 pixel_sha256=pixel, width=image.width, height=image.height, region=region, sha256=sha(path),
                                 provenance={'source': 'original', 'license': 'CC0-1.0'}, rendering=evidence, transform=transform))
            kind_counts[item['kind']] += 1
            if split == 'train':
                tokens.update(item['text'])
            if (index + 1) % 100 == 0:
                qd.storage_gate(root, 128 * 1024)
                atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'split': split, 'groups': index + 1, 'target': counts[split_index],
                                                      'completed_splits': list(splits), 'elapsed_seconds': time.perf_counter() - started, 'updated_utc': utc()})
            if (index + 1) % 1000 == 0:
                print(json.dumps({'split': split, 'groups': index + 1, 'elapsed_seconds': round(time.perf_counter() - started, 2)}), flush=True)
        writer.json(root / (split + '.json'), rows)
        positive = len({r['group_id'] for r in rows if r['text']})
        splits[split] = {'file': split + '.json', 'sha256': sha(root / (split + '.json')), 'count': len(rows),
                         'positive_groups': positive, 'negative_groups': counts[split_index] - positive, 'groups': counts[split_index],
                         'variants': variants[split_index]}
        summaries[split] = {'groups_by_kind': dict(kind_counts), 'images_by_kind': dict(Counter(r['kind'] for r in rows)),
                            'structures': len({r['recipe_expression'] for r in rows if r['kind'] == 'face'}),
                            'core_combinations': len(_rows_core_digests(rows)),
                            'effects': dict(Counter(key for r in rows for key in r['transform'].get('effects', {}) if key in ('outline', 'shadow', 'texture')))}
        if split == 'train':
            train_counts = gc.atom_counts([r for r in rows if r['kind'] == 'face'])
    missing = sorted(atom for role in ('eyes', 'mouths', 'cheeks', 'whiskers', 'inner_arms', 'arms_left', 'arms_right', 'tails')
                     for atom in allowed[role] if train_counts.get(atom, 0) == 0)
    missing += sorted({c for pair in allowed['brackets'] for c in pair if train_counts.get(c, 0) == 0})
    meta = {'schema_version': 'localocr.pretrained.dataset.v1', 'generator_id': GENERATOR_ID, 'synthetic_only': True,
            'private_inputs_consumed': False, 'license': 'CC0-1.0',
            'counts_semantics': 'total groups including 10 percent blank/graphic; every group retains all requested views',
            'distribution_scope': 'finite atom-table grammar with shared structural forms; hashed core combinations and pixels are split-disjoint; shared OFL fonts; no natural-screenshot accuracy claim',
            'isolation': {'exact_label': 'disjoint', 'core_combination': 'disjoint by hashed partition', 'pixels': 'disjoint',
                          'structural_forms': 'shared across splits', 'partition_buckets': {k: list(v) for k, v in gc.PARTITION_BUCKETS.items()}},
            'seed': seed, 'splits': splits, 'train_vocabulary': sorted(tokens, key=ord),
            'atoms_binding': kg.binding(atoms), 'renderable_atoms': {role: len(allowed[role]) for role in allowed},
            'excluded_atoms': excluded_atoms, 'train_atom_counts': train_counts, 'atoms_without_train_instances': missing,
            'effect_sets': EFFECT_SETS, 'split_effect_sets': SPLIT_EFFECT_SETS,
            'font_independence': False, 'atomic_emoji_glyph_independence': False,
            'fonts': [{k: v for k, v in f.items() if k not in ('file', 'license_file')} for f in renderer.fonts],
            'font_manifest_sha256': sha(fonts), 'renderer': renderer.versions,
            'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('grammar_data.py', 'grammar_corpus.py', 'quality_corpus.py', 'quality_data.py', 'kaomoji_grammar.py', 'kaomoji_atoms.json', 'data.py')},
            'renderer_sha256': sha(Path(__file__).resolve().parents[1] / 'OcrDatasets/render.py'),
            'excluded_prior_label_digests': excluded_prior_count, 'exclusion_reasons': dict(exclusions), 'summary': summaries,
            'usage': {'train': 'training', 'dev': 'development-selection', 'heldout': 'sealed-first-evaluation-after-model-and-router-selection'}}
    writer.json(root / 'dataset.json', meta)
    train = read(root / 'train.json')
    required = [len(r['text']) + sum(a == b for a, b in zip(r['text'], r['text'][1:])) for r in train]
    writer.json(root / 'train-length-statistics.json', {'source': 'train only', 'train_sha256': splits['train']['sha256'],
                'max_label_codepoints': max(len(r['text']) for r in train), 'max_ctc_required_steps': max(required), 'count': len(train),
                'brackets': {''.join(pair): sum(all(c in r['text'] for c in pair) for r in train if r['kind'] == 'face') for pair in allowed['brackets']}})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'validating-decoded-images', 'completed_splits': list(splits), 'elapsed_seconds': time.perf_counter() - started, 'updated_utc': utc()})
    binding = validate(root)
    writer.json(root / 'heldout-seal.json', {'schema_version': 'localocr.heldout-seal.v1', 'dataset_sha256': binding['dataset_sha256'],
                'heldout_sha256': binding['split_sha256']['heldout'], 'usage': 'first evaluation after model+router dual freeze',
                'labels_used_for_vocabulary': False, 'results_seen': False})
    for relative, value in old_bindings.items():
        if sha(qd.ARTIFACTS / relative) != value:
            raise ValueError('Previous data bytes changed during generation')
    writer.json(root / 'generation.json', {'schema_version': 'localocr.grammar-generation.v1', 'binding': binding, 'elapsed_seconds': time.perf_counter() - started,
                'old_inputs_sha256': old_bindings, 'old_inputs_unchanged': True, 'CPU_model_inference': False, 'GPU_used': False,
                'font_files_distributed': False, 'private_inputs_consumed': False, 'exclusions': dict(exclusions), 'excluded_atoms': excluded_atoms,
                'atoms_without_train_instances': missing, 'summary': summaries, 'storage': qd.storage_gate(root)})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'complete', 'dataset_sha256': binding['dataset_sha256'], 'elapsed_seconds': time.perf_counter() - started, 'completed_utc': utc()})
    writer.json(root / 'freeze.json', binding)
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'validate'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--fonts')
    parser.add_argument('--counts', nargs=3, type=int, default=(8000, 1000, 2000))
    parser.add_argument('--variants', nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    print(json.dumps(generate(args.dataset, args.fonts, tuple(args.counts), tuple(args.variants), args.seed) if args.action == 'generate' else validate(args.dataset), indent=2))


if __name__ == '__main__':
    main()
