"""Grammar-round mixed scenes: atom-table faces, hashed core partitions and declared ink effects.

Only the separate six-field image index is valid for inference. Coordinates,
labels, spans and masks are measurement-only truth. Faces use the development
or heldout core partition of the canonical atom table, so no scene core is
ever a training core. Backgrounds stay solid so visible-ink masks remain exact.
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
from PIL import Image, ImageDraw
from tools.OcrDatasets.render import Unsupported, validate_effects
from tools.PretrainedOcr.data import MixedRenderer
from tools.PretrainedOcr.quality_corpus import CN_WORDS, EN_WORDS
from tools.PretrainedOcr import kaomoji_grammar as kg, grammar_corpus as gc, round_profile as rp
from tools.PretrainedOcr import quality_data as qd
from tools.PretrainedOcr.grammar_data import renderable_atoms
from tools.PretrainedOcr.common import write as atomic_write, utc
from .scenes import sha, th, read, pixels, bbox, BG
from . import quality_scenes as qs

GENERATOR_ID = 'localocr.grammar-scenes.20261013.v1'
NEW_TYPES = ('cheek-face', 'arm-face', 'tail-face', 'corner-face', 'mixed-width-face', 'wide-gap-body',
             'adjacent-faces', 'outlined-face', 'shadow-body')
TYPES = qs.TYPES + NEW_TYPES
RECIPES = {
    'dev': {**qs.RECIPES['dev'],
            'cheek-face': 'body gap cheek-face', 'arm-face': 'body gap arm-face', 'tail-face': 'body newline tail-face',
            'corner-face': 'body gap corner-face', 'mixed-width-face': 'body newline mixed-width-face',
            'wide-gap-body': 'wide-gap-body gap face', 'adjacent-faces': 'body gap face narrow-gap face',
            'outlined-face': 'body gap outlined-face', 'shadow-body': 'shadow-body gap face'},
    'heldout': {**qs.RECIPES['heldout'],
                'cheek-face': 'cheek-face newline body', 'arm-face': 'arm-face gap body', 'tail-face': 'tail-face gap body gap body',
                'corner-face': 'corner-face newline body', 'mixed-width-face': 'mixed-width-face gap body',
                'wide-gap-body': 'face newline wide-gap-body', 'adjacent-faces': 'face narrow-gap face newline body',
                'outlined-face': 'outlined-face newline outlined-body', 'shadow-body': 'face gap shadow-body'},
}
SCENE_EFFECTS = {'dev': {'outline': 1, 'outline_color': (70, 70, 70), 'shadow': {'dx': 1, 'dy': 1, 'color': (0, 0, 0), 'alpha': 150}, 'random_probability': .25},
                 'heldout': {'outline': 2, 'outline_color': (40, 40, 40), 'shadow': {'dx': 2, 'dy': 2, 'color': (0, 0, 0), 'alpha': 120}, 'random_probability': .30}}
CORNER = (('「', '」'), ('『', '』'), ('｢', '｣'))
SQUARE = (('[', ']'),)
CURLY = (('{', '}'),)
FULLWIDTH = (('（', '）'), ('【', '】'), ('〈', '〉'))
JOB_KEYS = qs.JOB_KEYS


def family(split, kind):
    return 'grammar-scene-grammar:' + th(RECIPES[split][kind])


def _mixed_pairs(allowed):
    return tuple(pair for pair in allowed['brackets'] if (pair[0] in '([' ) != (pair[1] in ')]'))


def face(rng, split, seen, atoms, allowed, constraint='plain'):
    """One scene face from the split's core partition, satisfying a structural constraint."""
    scoped = dict(allowed)
    if constraint in ('corner', 'square', 'curly', 'fullwidth', 'mixed'):
        pool = {'corner': CORNER, 'square': SQUARE, 'curly': CURLY, 'fullwidth': FULLWIDTH, 'mixed': _mixed_pairs(allowed)}[constraint]
        scoped['brackets'] = tuple(pair for pair in allowed['brackets'] if pair in pool)
        if not scoped['brackets']:
            raise Unsupported('bracket-family-unrenderable')
    for _ in range(10000):
        composed = gc.compose_face(split, rng, atoms, scoped)
        roles = composed['roles']
        if constraint == 'cheek' and not (roles['cheek_left'] or roles['cheek_right']):
            continue
        if constraint == 'arm' and len(roles['arms_left']) + len(roles['arms_right']) < 2:
            continue
        if constraint == 'tail' and not roles['tails']:
            continue
        if constraint == 'plain' and (roles['tails'] or roles['arms_left'] or roles['arms_right']) and rng.random() < .5:
            continue
        digest = th(composed['text'])
        if digest in seen:
            continue
        seen.add(digest)
        return composed
    raise ValueError('Original scene expression pool exhausted')


def body(rng, split, seen, wide=False):
    for _ in range(10000):
        a, b = rng.sample(CN_WORDS, 2)
        c, d = rng.sample(EN_WORDS, 2)
        if wide:
            text = f'{a}    {c}' if split == 'dev' else f'{d}    {b}'
        else:
            text = f'{a} {c} {b}' if split == 'dev' else f'{d}/{a} {b}'
        if th(text) not in seen:
            seen.add(th(text))
            return text
    raise ValueError('Original scene body pool exhausted')


def plan(split, kind, rng, seen, atoms, allowed):
    b1 = body(rng, split, seen)
    b2 = body(rng, split, seen)
    effect = SCENE_EFFECTS[split]
    def normal(text, effects=None):
        return dict(kind='body', text=text, size=24, contrast=1., effects=effects or {})
    def facial(size=24, contrast=1., constraint='plain', effects=None):
        composed = face(rng, split, seen, atoms, allowed, constraint)
        return dict(kind='kaomoji', text=composed['text'], size=size, contrast=contrast, effects=effects or {},
                    core_key=composed['core_key'], core_digest=composed['core_digest'], core_partition=composed['core_partition'],
                    structure=composed['structure'])
    outline = {'outline': effect['outline'], 'outline_color': effect['outline_color']}
    shadow = {'shadow': effect['shadow']}
    if kind == 'same-line': return [[normal(b1), facial()]] if split == 'dev' else [[facial(), normal(b1)]]
    if kind == 'middle-face': return [[normal(b1), facial(), normal(b2)]] if split == 'dev' else [[facial(), normal(b1), facial()]]
    if kind == 'different-line': return [[normal(b1)], [facial()]] if split == 'dev' else [[facial()], [normal(b1)], [normal(b2)]]
    if kind == 'low-body':
        low = normal(b1); low['contrast'] = .14
        return [[low, facial()]] if split == 'dev' else [[facial()], [low]]
    if kind == 'small-face': return [[normal(b1), facial(12)]] if split == 'dev' else [[facial(12)], [normal(b1)]]
    if kind == 'low-face': return [[normal(b1)], [facial(20, .13)]] if split == 'dev' else [[facial(20, .13), normal(b1)]]
    if kind == 'double-face': return [[normal(b1), facial(), facial()]] if split == 'dev' else [[facial(), facial()], [normal(b1)]]
    if kind == 'tight-body': return [[normal(b1), facial()]] if split == 'dev' else [[facial(), normal(b1)]]
    if kind == 'edge-face': return [[normal(b1)], [facial()]] if split == 'dev' else [[facial()], [normal(b1)]]
    if kind == 'square-face': return [[normal(b1), facial(constraint='square')]] if split == 'dev' else [[facial(constraint='square')], [normal(b1)]]
    if kind == 'curly-face': return [[normal(b1)], [facial(constraint='curly')]] if split == 'dev' else [[facial(constraint='curly'), normal(b1)]]
    if kind == 'fullwidth-face': return [[normal(b1), facial(constraint='fullwidth')]] if split == 'dev' else [[facial(constraint='fullwidth')], [normal(b1)]]
    if kind == 'ordinary-brackets': return [[normal(b1 + ' (' + b2 + ')')]] if split == 'dev' else [[normal('[' + b1 + ']')], [normal(b2)]]
    if kind == 'letter-brackets': return [[normal(b1 + ' [o] (T)')]] if split == 'dev' else [[normal('(o) [x] ' + b1)]]
    if kind == 'ordinary-numbers':
        value = str(rng.randrange(100, 10000))
        return [[normal(b1 + ' [123] (' + value + ')')]] if split == 'dev' else [[normal('(' + value + ') [123] ' + b1)]]
    if kind == 'combining-body':
        text = 'café ' + b1 + ' ō' if split == 'dev' else 'ō/' + b1 + '/café'
        return [[normal(text), facial()]] if split == 'dev' else [[facial()], [normal(text)]]
    if kind in ('body-graphic', 'table-lines'): return [[normal(b1)]]
    if kind == 'cheek-face': return [[normal(b1), facial(constraint='cheek')]] if split == 'dev' else [[facial(constraint='cheek')], [normal(b1)]]
    if kind == 'arm-face': return [[normal(b1), facial(constraint='arm')]] if split == 'dev' else [[facial(constraint='arm'), normal(b1)]]
    if kind == 'tail-face': return [[normal(b1)], [facial(constraint='tail')]] if split == 'dev' else [[facial(constraint='tail'), normal(b1), normal(b2)]]
    if kind == 'corner-face': return [[normal(b1), facial(constraint='corner')]] if split == 'dev' else [[facial(constraint='corner')], [normal(b1)]]
    if kind == 'mixed-width-face': return [[normal(b1)], [facial(constraint='mixed')]] if split == 'dev' else [[facial(constraint='mixed'), normal(b1)]]
    if kind == 'wide-gap-body':
        wide = normal(body(rng, split, seen, wide=True))
        return [[wide, facial()]] if split == 'dev' else [[facial()], [wide]]
    if kind == 'adjacent-faces': return [[normal(b1), facial(), facial()]] if split == 'dev' else [[facial(), facial()], [normal(b1)]]
    if kind == 'outlined-face': return [[normal(b1), facial(effects=outline)]] if split == 'dev' else [[facial(effects=outline)], [normal(b1, effects=outline)]]
    if kind == 'shadow-body': return [[normal(b1, effects=shadow), facial()]] if split == 'dev' else [[facial(), normal(b1, effects=shadow)]]
    return []


def render_tile(renderer, segment, split, rng):
    effect = SCENE_EFFECTS[split]
    effects = dict(segment['effects'])
    if not effects and segment['kind'] in ('body', 'kaomoji') and rng.random() < effect['random_probability']:
        effects = {'outline': effect['outline'], 'outline_color': effect['outline_color']} if rng.random() < .5 else {'shadow': effect['shadow']}
    effects = validate_effects({**effects, 'background': BG})
    image, region, fonts, evidence, alpha = renderer.render(segment['text'], size=segment['size'], effects=effects, with_alpha=True)
    x, y, w, h = region
    pad = 3
    def advance(value):
        return sum(p.x_advance / f[5] * segment['size'] for f, _, positions in renderer.shape(value) for p in positions) if value else 0
    text = segment['text']
    leading = text[:len(text) - len(text.lstrip(' '))]
    trailing = text[len(text.rstrip(' ')):]
    left = max(0, int(x - pad - advance(leading)))
    right = min(image.width, int(np.ceil(x + w + pad + advance(trailing))))
    top, bottom = max(0, y - pad), min(image.height, y + h + pad)
    image = image.crop((left, top, right, bottom))
    array = np.asarray(image).astype(float)
    array = np.rint(np.array(BG) + (array - np.array(BG)) * segment['contrast']).astype('uint8')
    mask = Image.fromarray(np.any(array != np.array(BG), axis=2).astype('uint8') * 255)
    return Image.fromarray(array), mask, fonts, dict(evidence, effects=evidence['effects'])


def render_scene(renderer, split, kind, rng, index, seen, atoms, allowed):
    lines = plan(split, kind, rng, seen, atoms, allowed)
    gap = 0 if kind == 'tight-body' else 6 if kind == 'adjacent-faces' else 18
    tiles = []
    for line in lines:
        local = []
        for segment in line:
            tile, mask, fonts, evidence = render_tile(renderer, segment, split, rng)
            local.append((segment, tile, mask, fonts, evidence))
        tiles.append(local)
    widths = [sum(tile.width for _, tile, *_ in line) + gap * (len(line) - 1) for line in tiles]
    width = max(380, max(widths, default=220) + 96)
    height = max(170, 70 + len(tiles) * 64)
    if kind == 'graphic-only':
        width += index % 31
        height += index % 23
    if split == 'heldout' and kind in ('body-graphic', 'table-lines'):
        height = max(height, 240)
    canvas = Image.new('RGB', (width, height), BG)
    whole = ''
    regions = []
    masks = []
    for li, line in enumerate(tiles):
        if li:
            whole += '\n'
        cursor = 3 if kind == 'edge-face' and line and line[0][0]['kind'] == 'kaomoji' else 48
        for si, (segment, tile, mask, fonts, evidence) in enumerate(line):
            if si:
                whole += '  '
                cursor += gap
            start = len(whole)
            whole += segment['text']
            x = cursor
            y = 150 + li * 64 if split == 'heldout' and kind in ('body-graphic', 'table-lines') else 28 + li * 64
            if x < 0 or x + tile.width > width or y + tile.height > height:
                raise Unsupported('scene-clipping')
            canvas.paste(tile, (x, y), mask)
            globalmask = Image.new('L', canvas.size)
            globalmask.paste(mask, (x, y))
            region = {'kind': segment['kind'], 'text': segment['text'], 'span': [start, len(whole)], 'bbox': bbox(globalmask),
                      'font_ids': fonts, 'rendering': evidence, 'contrast': segment['contrast'], 'size': segment['size']}
            if segment['kind'] == 'kaomoji':
                region.update(core_key=segment['core_key'], core_digest=segment['core_digest'], core_partition=segment['core_partition'],
                              structure=segment['structure'])
            regions.append(region)
            masks.append(globalmask)
            cursor += tile.width
    if kind in ('body-graphic', 'graphic-only', 'table-lines'):
        mask = Image.new('L', canvas.size)
        draw = ImageDraw.Draw(canvas)
        md = ImageDraw.Draw(mask)
        box = [width - 170, 20, width - 24, 78] if split == 'heldout' and kind in ('body-graphic', 'table-lines') else [width - 170, height - 72, width - 24, height - 15]
        ink = (25, 40, 70)
        primitive = ('ellipse', 'triangle', 'box', 'table')[index // len(TYPES) % 4]
        if kind == 'table-lines':
            primitive = 'table'
        def both(method, *args, **kwargs):
            getattr(draw, method)(*args, fill=ink, **kwargs)
            getattr(md, method)(*args, fill=255, **kwargs)
        if primitive == 'ellipse':
            draw.ellipse(box, outline=ink, width=2); md.ellipse(box, outline=255, width=2)
        elif primitive == 'triangle':
            points = [(box[0], box[3]), ((box[0] + box[2]) // 2, box[1]), (box[2], box[3])]
            draw.polygon(points, outline=ink, width=2); md.polygon(points, outline=255, width=2)
        else:
            draw.rectangle(box, outline=ink, width=1); md.rectangle(box, outline=255, width=1)
            if primitive == 'table':
                for x in range(box[0] + 20, box[2], 20):
                    both('line', (x, box[1], x, box[3]), width=1)
                for y in range(box[1] + 16, box[3], 16):
                    both('line', (box[0], y, box[2], y), width=1)
        if split == 'dev':
            both('line', (box[0], box[1], box[2], box[3]), width=1)
        else:
            draw.arc(box, 0, 180, fill=ink, width=2); md.arc(box, 0, 180, fill=255, width=2)
        regions.append({'kind': 'graphic', 'text': '', 'span': None, 'bbox': bbox(mask), 'font_ids': [],
                        'rendering': {'text_present': False}, 'geometry': primitive})
        masks.append(mask)
    if kind == 'blank':
        bg = tuple(rng.randrange(210, 251) for _ in range(3))
        canvas = Image.new('RGB', (width + index % 31, height + index % 17), bg)
        if split == 'heldout':
            draw = ImageDraw.Draw(canvas)
            for y in range(canvas.height):
                draw.line((0, y, canvas.width - 1, y), fill=tuple(c - y // 12 for c in bg))
    return canvas, whole, regions, masks


def validate(root):
    root = Path(root).resolve()
    meta = read(root / 'dataset.json')
    if meta.get('schema_version') != 'localocr.region-scenes.v1' or meta.get('synthetic_only') is not True or meta.get('generator_id') != GENERATOR_ID:
        raise ValueError('Scene schema/synthetic declaration')
    atoms = kg.default_atoms()
    if meta.get('atoms_binding') != kg.binding(atoms):
        raise ValueError('Scene atoms binding differs from the canonical table')
    texts, micros, families, cores, pixelsets, counts = {}, {}, {}, {}, {}, {}
    for split in ('dev', 'heldout'):
        info = meta['splits'][split]
        relative = Path(info['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Split path escape')
        if sha(root / relative) != info['sha256']:
            raise ValueError('Split hash changed')
        rows = read(root / relative)
        if len(rows) != info['count']:
            raise ValueError('Scene count changed')
        for row in rows:
            if row['split'] != split or row['source_split'] != split or row['recipe_family'] != family(split, row['scene_kind']):
                raise ValueError('Scene source/AST family differs')
            if row['provenance'] != {'source': 'original', 'license': 'CC0-1.0'} or th(row['text']) != row['text_sha256']:
                raise ValueError('Label/provenance changed')
            for key, inventory in [(row['text_sha256'] if row['text'] else None, texts), (row['recipe_family'], families)]:
                if key is not None:
                    if key in inventory and inventory[key] != split:
                        raise ValueError('Cross-split label/family')
                    inventory[key] = split
            p = (root / row['image']).resolve()
            if not p.is_relative_to(root) or sha(p) != row['sha256']:
                raise ValueError('Image hash/path')
            with Image.open(p) as im:
                im.load()
                arr = np.asarray(im.convert('RGB'))
                pixel = pixels(im)
                if im.size != (row['width'], row['height']) or pixel != row['pixel_sha256']:
                    raise ValueError('Scene pixels/dimensions')
            if pixel in pixelsets:
                raise ValueError('Duplicate scene pixels')
            pixelsets[pixel] = split
            masks = []
            for region in row['regions']:
                mp = (root / region['mask']).resolve()
                if not mp.is_relative_to(root) or sha(mp) != region['mask_sha256']:
                    raise ValueError('Mask hash/path')
                with Image.open(mp) as m:
                    m.load()
                    support = np.asarray(m) > 0
                    if m.mode != 'L' or m.size != (row['width'], row['height']) or bbox(m) != region['bbox'] or int(support.sum()) != region['pixel_count']:
                        raise ValueError('Actual mask/bbox pixels differ')
                if region['kind'] != 'graphic':
                    a, b = region['span']
                    if not 0 <= a < b <= len(row['text']) or row['text'][a:b] != region['text']:
                        raise ValueError('Exact character span')
                    key = th(region['text'])
                    if key in micros and micros[key] != split:
                        raise ValueError('Micro-label source leakage')
                    micros[key] = split
                    if region['rendering']['glyphs'] <= 0 or not region['font_ids']:
                        raise ValueError('Missing genuine glyphs')
                    validate_effects(region['rendering'].get('effects'))
                    if not np.all(np.any(arr[support] != np.array(BG), axis=1)):
                        raise ValueError('No visible source image ink')
                if region['kind'] == 'kaomoji':
                    if (gc.core_digest(region['core_key']) != region['core_digest'] or gc.partition(region['core_key']) != split
                            or region['core_partition'] != split or kg.parse_face(region['text'], atoms) is None):
                        raise ValueError('Scene face core partition or grammar identity differs')
                    if region['core_digest'] in cores and cores[region['core_digest']] != split:
                        raise ValueError('Cross-split scene core leakage')
                    cores[region['core_digest']] = split
                masks.append(support)
            if masks and np.any(np.sum(masks, axis=0) > 1):
                raise ValueError('Region glyph pixels overlap')
            if row['scene_kind'] != 'blank' and not np.array_equal(np.any(masks, axis=0) if masks else np.zeros(arr.shape[:2], bool), np.any(arr != np.array(BG), axis=2)):
                raise ValueError('Truth masks do not cover all actual visible ink')
        counts[split] = {'scenes': len(rows), 'faces': sum(r['kind'] == 'kaomoji' for row in rows for r in row['regions'])}
    binding = {'schema_version': 'localocr.region-scenes.freeze.v1', 'dataset_sha256': sha(root / 'dataset.json'),
               'split_sha256': {s: meta['splits'][s]['sha256'] for s in ('dev', 'heldout')}, 'counts': counts}
    if (root / 'freeze.json').exists() and read(root / 'freeze.json') != binding:
        raise ValueError('Scene freeze differs')
    if (root / 'heldout-seal.json').exists():
        seal = read(root / 'heldout-seal.json')
        if seal['dataset_sha256'] != binding['dataset_sha256'] or seal['heldout_sha256'] != binding['split_sha256']['heldout']:
            raise ValueError('Scene seal differs')
    return binding


def generate(output, fonts, counts=(200, 400), seed=20261013):
    if len(counts) != 2 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
        raise ValueError('Positive scene counts required')
    root = qd.output_root(output)
    if rp.profile_for(root) is None:
        raise ValueError('Grammar scenes must be generated inside a registered research round')
    if root.exists():
        raise ValueError('Fresh scene output required')
    qd.storage_gate(root, 1024 ** 2)
    renderer = MixedRenderer(fonts)
    atoms = kg.default_atoms()
    allowed, excluded_atoms = renderable_atoms(renderer, atoms)
    root.mkdir(parents=True)
    (root / 'images').mkdir()
    (root / 'masks').mkdir()
    writer = qd.Writer(root)
    started = time.perf_counter()
    seen, allpixels, splits, exclusions, summary, jobs = set(), set(), {}, Counter(), {}, {}
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'started_utc': utc()})
    for split, target in zip(('dev', 'heldout'), counts):
        rng = random.Random(seed + (0 if split == 'dev' else 1))
        rows = []
        for index in range(target):
            kind = TYPES[index % len(TYPES)]
            for attempt in range(1000):
                try:
                    canvas, text, regions, masks = render_scene(renderer, split, kind, rng, index, seen, atoms, allowed)
                except Unsupported as error:
                    exclusions[str(error)] += 1
                    continue
                pixel = pixels(canvas)
                if pixel in allpixels:
                    exclusions['duplicate-pixels'] += 1
                    continue
                break
            else:
                raise ValueError('Unable to render reliable original scene')
            allpixels.add(pixel)
            identifier = f'{split}-{index:04d}'
            path = root / 'images' / (identifier + '.png')
            writer.image(path, canvas)
            for ri, (region, mask) in enumerate(zip(regions, masks)):
                mp = root / 'masks' / f'{identifier}-{ri}.png'
                writer.image(mp, mask)
                region.update(id=f'{identifier}-region-{ri}', mask=mp.relative_to(root).as_posix(), mask_sha256=sha(mp), pixel_count=int((np.asarray(mask) > 0).sum()))
            rows.append({'id': identifier, 'split': split, 'source_split': split, 'scene_kind': kind, 'recipe_family': family(split, kind),
                         'text': text, 'text_sha256': th(text), 'image': path.relative_to(root).as_posix(), 'sha256': sha(path), 'pixel_sha256': pixel,
                         'width': canvas.width, 'height': canvas.height, 'regions': regions, 'provenance': {'source': 'original', 'license': 'CC0-1.0'},
                         'baseline_omission': 'not measured; synthetic pressure only'})
            if (index + 1) % 50 == 0:
                qd.storage_gate(root, 128 * 1024)
                atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'split': split, 'scenes': index + 1, 'target': target,
                                                      'elapsed_seconds': time.perf_counter() - started, 'updated_utc': utc()})
        writer.json(root / (split + '.json'), rows)
        splits[split] = {'file': split + '.json', 'sha256': sha(root / (split + '.json')), 'count': len(rows), 'requested': target}
        jobs[split] = [{k: row[k] for k in JOB_KEYS} for row in rows]
        summary[split] = {'scene_types': dict(Counter(r['scene_kind'] for r in rows)),
                          'faces': sum(region['kind'] == 'kaomoji' for row in rows for region in row['regions']),
                          'effects': dict(Counter(key for row in rows for region in row['regions'] for key in region['rendering'].get('effects', {}) if key in ('outline', 'shadow')))}
    writer.json(root / 'dataset.json', {'schema_version': 'localocr.region-scenes.v1', 'generator_id': GENERATOR_ID, 'synthetic_only': True,
                'private_inputs_consumed': False, 'license': 'CC0-1.0', 'seed': seed, 'splits': splits, 'recipes': RECIPES,
                'scene_effects': SCENE_EFFECTS, 'atoms_binding': kg.binding(atoms), 'excluded_atoms': excluded_atoms,
                'core_partition_buckets': {k: list(v) for k, v in gc.PARTITION_BUCKETS.items()},
                'renderer': renderer.versions, 'fonts': [{k: v for k, v in f.items() if k not in ('file', 'license_file')} for f in renderer.fonts],
                'font_manifest_sha256': sha(fonts), 'font_independence': False, 'vocabulary_independence': False,
                'source_sha256': sha(__file__),
                'dependency_sha256': {name: sha(Path(__file__).resolve().parents[1] / name) for name in ['OcrDatasets/render.py', 'PretrainedOcr/quality_corpus.py', 'PretrainedOcr/grammar_corpus.py', 'PretrainedOcr/grammar_data.py', 'PretrainedOcr/kaomoji_grammar.py', 'PretrainedOcr/kaomoji_atoms.json', 'RegionKaomoji/quality_scenes.py', 'RegionKaomoji/scenes.py']},
                'bounds': 'original-image visible-ink half-open [left,top,right,bottom)',
                'spans': 'raw Python Unicode codepoint spans [start,end); reading-order separator two spaces/LF; separators are not regions',
                'exclusions': dict(exclusions), 'summary': summary,
                'usage': {'dev': 'routing/fusion/split development', 'heldout': 'first CPU evaluation after model+router+fusion selection freeze'}})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'validating-decoded-images-and-masks', 'updated_utc': utc()})
    binding = validate(root)
    writer.json(root / 'heldout-seal.json', {'schema_version': 'localocr.region-scenes.seal.v1', 'dataset_sha256': binding['dataset_sha256'],
                'heldout_sha256': binding['split_sha256']['heldout'], 'inference_results_seen': False, 'requires_frozen_pipeline_selection': True})
    writer.json(root / 'inference-inputs.json', {'schema_version': 'localocr.region-scenes.inputs.v1', 'dataset_sha256': binding['dataset_sha256'],
                'split_sha256': binding['split_sha256'], 'jobs': jobs})
    writer.json(root / 'generation.json', {'binding': binding, 'elapsed_seconds': time.perf_counter() - started, 'summary': summary, 'exclusions': dict(exclusions),
                'excluded_atoms': excluded_atoms, 'CPU_model_inference': False, 'GPU_used': False, 'private_inputs': False, 'fonts_distributed': False, 'storage': qd.storage_gate(root)})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'complete', 'completed_utc': utc(), 'dataset_sha256': binding['dataset_sha256']})
    writer.json(root / 'freeze.json', binding)
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'validate'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--fonts')
    parser.add_argument('--counts', nargs=2, type=int, default=(200, 400))
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    print(json.dumps(generate(args.dataset, args.fonts, tuple(args.counts), args.seed) if args.action == 'generate' else validate(args.dataset), indent=2))


if __name__ == '__main__':
    main()
