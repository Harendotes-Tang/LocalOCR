"""Record-card list scenes: light text on dark rounded cards with inline faces, original content only.

The layout mirrors a real record-list UI the user showed: a gold emblem, a level
label, a small square icon, a duty-like name, a timestamp, a comment that may
carry inline kaomoji (attached, spaced, standalone, wrapped onto a second line)
and a close mark on the right. Emblems and icons are original geometric shapes,
names are original word compounds; no game assets or vocabulary are used. Each
scene stores its text-free background so visible-ink masks stay exact on
multi-colour pages.
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
from tools.OcrDatasets.render import Unsupported
from tools.PretrainedOcr.data import MixedRenderer
from tools.PretrainedOcr.quality_corpus import CN_WORDS, EN_WORDS
from tools.PretrainedOcr import kaomoji_grammar as kg, grammar_corpus as gc, round_profile as rp
from tools.PretrainedOcr import quality_data as qd
from tools.PretrainedOcr.grammar_data import renderable_atoms
from tools.PretrainedOcr.common import write as atomic_write, utc
from .scenes import sha, th, read, pixels, bbox

GENERATOR_ID = 'localocr.card-scenes.20261013.v1'
JOB_KEYS = ('id', 'image', 'sha256', 'pixel_sha256', 'width', 'height')
STYLES = {
    'dev': {'page': (232, 230, 226), 'cards': [(92, 46, 44), (58, 92, 42)], 'gold': (206, 172, 88), 'text': (242, 240, 236),
            'muted': (214, 210, 200), 'icons': [(44, 150, 168), (226, 132, 58), (68, 110, 196)],
            'height': 88, 'gap': 12, 'radius': 10, 'size': 19, 'width': 1860, 'rows': 10,
            'recipe': 'card-list dev palette: two card colours, 88px rows, 19px text, two-line comments wrap at segment boundaries'},
    'heldout': {'page': (238, 236, 232), 'cards': [(84, 40, 40), (52, 84, 38), (70, 60, 44)], 'gold': (214, 180, 96), 'text': (246, 244, 240),
                'muted': (222, 216, 204), 'icons': [(52, 160, 150), (210, 118, 70), (80, 96, 188), (150, 70, 160)],
                'height': 92, 'gap': 14, 'radius': 12, 'size': 20, 'width': 1864, 'rows': 10,
                'recipe': 'card-list heldout palette: three card colours, 92px rows, 20px text, two-line comments wrap at segment boundaries'},
}
PUNCTUATION = ('，', '。', '！', '？', '...', '…', '（', '）', ';;', '.jpg', '030', '（目移', '哈哈哈', '！！！', '（。', '233')
# Each plan is a sequence of (segment kind, separator before the segment); separators are '' (attached), ' ' or '\n' (wrap).
COMMENT_PLANS = {
    'empty': (),
    'plain': (('body', ''),),
    'face-attached': (('body', ''), ('face', '')),
    'face-spaced': (('body', ''), ('face', ' ')),
    'face-only': (('face', ''),),
    'face-middle': (('body', ''), ('face', '?'), ('body', '?')),
    'two-faces': (('body', ''), ('face', ''), ('face', ' ')),
    'wrap-plain': (('body', ''), ('body', '\n')),
    'wrap-face-end': (('body', ''), ('face', ''), ('body', '\n')),
    'wrap-face-start': (('body', ''), ('face', '\n'), ('body', '?')),
}
COMMENT_WEIGHTS = {'empty': 6, 'plain': 34, 'face-attached': 14, 'face-spaced': 10, 'face-only': 7, 'face-middle': 6,
                   'two-faces': 5, 'wrap-plain': 8, 'wrap-face-end': 5, 'wrap-face-start': 5}
COLUMN_GAP = '  '
LINE_SHIFT = .62
NEGATIVE_KINDS = ('empty', 'plain', 'wrap-plain')
NEGATIVE_EVERY = 5  # every fifth scene carries no face at all: the negative-scene population for the research gates
SCENE_KINDS = ('card-list', 'card-list-negative')


def family(split):
    return 'card-scene-grammar:' + th(STYLES[split]['recipe'])


def comment_words(rng, count):
    parts = []
    for _ in range(count):
        choice = rng.random()
        if choice < .55:
            parts.append(rng.choice(CN_WORDS))
        elif choice < .75:
            parts.append(rng.choice(EN_WORDS))
        elif choice < .9:
            parts.append(rng.choice(PUNCTUATION))
        else:
            parts.append(str(rng.randrange(1, 100)))
    return ''.join(parts)


def contains_face(text, atoms):
    """True when any bracket-delimited substring parses as a face; body text must never carry one."""
    chars = atoms.bracket_chars
    positions = [index for index, char in enumerate(text) if char in chars]
    return any(kg.parse_face(text[a:b + 1], atoms, allow_arms=False) is not None
               for i, a in enumerate(positions) for b in positions[i + 1:])


def plan_comment(rng, split, seen, atoms, allowed, negative=False):
    """Ordered segments [(kind, payload, separator)] for one comment; faces come from the split's core partition."""
    kinds = [k for k in COMMENT_WEIGHTS if not negative or k in NEGATIVE_KINDS]
    kind = rng.choices(kinds, [COMMENT_WEIGHTS[k] for k in kinds])[0]
    def face():
        for _ in range(10000):
            composed = gc.compose_face(split, rng, atoms, allowed)
            digest = th(composed['text'])
            if digest not in seen:
                seen.add(digest)
                return composed
        raise ValueError('Original face pool exhausted')
    def body(count):
        for _ in range(10000):
            text = comment_words(rng, count)
            if th(text) not in seen and text.strip() and not contains_face(text, atoms):
                seen.add(th(text))
                return text
        raise ValueError('Original comment pool exhausted')
    segments = []
    for position, (segment_kind, separator) in enumerate(COMMENT_PLANS[kind]):
        if separator == '?':
            separator = rng.choice(('', ' '))
        if segment_kind == 'face':
            segments.append(('face', face(), separator))
        else:
            long = kind.startswith('wrap') and position == 0
            segments.append(('body', body(rng.randrange(12, 20) if long else rng.randrange(2, 9)), separator))
    return kind, segments


def advance(renderer, text, size):
    return sum(p.x_advance / f[5] * size for f, _, positions in renderer.shape(text) for p in positions) if text else 0


def render_text(renderer, text, size, color, background):
    """Tight tile and exact ink mask of ``text`` drawn in ``color`` on a flat ``background``."""
    tile, region, fonts, evidence, _ = renderer.render(text, size=size, effects={'ink': color, 'background': background}, with_alpha=True)
    x, y, w, h = region
    pad = 2
    leading = text[:len(text) - len(text.lstrip(' '))]
    trailing = text[len(text.rstrip(' ')):]
    left = max(0, int(x - pad - advance(renderer, leading, size)))
    right = min(tile.width, int(np.ceil(x + w + pad + advance(renderer, trailing, size))))
    top, bottom = max(0, y - pad), min(tile.height, y + h + pad)
    tile = tile.crop((left, top, right, bottom))
    array = np.asarray(tile)
    mask = Image.fromarray(np.any(array != np.array(background), axis=2).astype('uint8') * 255)
    return tile, mask, fonts, evidence


def draw_gold_glyph(canvas, mask, centre, colour, rng):
    """Original abstract gold emblem: ring, triangle, arc or crossed strokes; never an application icon."""
    cx, cy = centre
    draw, md = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
    r = rng.randrange(22, 30)
    style = rng.randrange(4)
    for d, fill in ((draw, colour), (md, 255)):
        if style == 0:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=fill, width=5)
            d.arc((cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8), 200, 340, fill=fill, width=4)
        elif style == 1:
            d.polygon([(cx - r, cy + r), (cx, cy - r), (cx + r, cy + r)], outline=fill, width=5)
            d.line((cx - r // 2, cy + r // 3, cx + r // 2, cy + r // 3), fill=fill, width=4)
        elif style == 2:
            d.arc((cx - r, cy - r, cx + r, cy + r), 20, 300, fill=fill, width=6)
            d.line((cx - r // 3, cy - r, cx + r // 2, cy + r // 2), fill=fill, width=4)
        else:
            d.line((cx - r, cy + r, cx + r // 2, cy - r), fill=fill, width=6)
            d.line((cx - r // 2, cy + r, cx + r, cy - r // 2), fill=fill, width=6)
            d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=fill, width=3)
    return style


def draw_square_icon(canvas, mask, box, colour, rng):
    draw, md = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill=colour)
    md.rounded_rectangle(box, radius=5, fill=255)
    inner = (x0 + 8, y0 + 8, x1 - 8, y1 - 8)
    if rng.random() < .5:
        draw.ellipse(inner, fill=(236, 240, 244))
    else:
        draw.polygon([(inner[0], inner[3]), ((inner[0] + inner[2]) // 2, inner[1]), (inner[2], inner[3])], fill=(236, 240, 244))


def graphic_region(mask, geometry):
    return {'kind': 'graphic', 'text': '', 'span': None, 'bbox': bbox(mask), 'font_ids': [], 'rendering': {'text_present': False, 'glyphs': 0}, 'geometry': geometry}


class Page:
    """Accumulates one scene: pastes tiles, tracks the reading-order text and the regions."""
    def __init__(self, canvas):
        self.canvas, self.whole, self.regions, self.masks = canvas, '', [], []

    def paste(self, tile, mask, x, y, kind, text, fonts, evidence, size, separator, extra=None):
        if x < 0 or y < 0 or x + tile.width > self.canvas.width or y + tile.height > self.canvas.height:
            raise Unsupported('card-row-overflow')
        self.canvas.paste(tile, (x, y), mask)
        full = Image.new('L', self.canvas.size)
        full.paste(mask, (x, y))
        if self.whole and not self.whole.endswith('\n'):
            self.whole += separator
        start = len(self.whole)
        self.whole += text
        region = {'kind': kind, 'text': text, 'span': [start, len(self.whole)], 'bbox': bbox(full), 'font_ids': fonts,
                  'rendering': evidence, 'contrast': 1., 'size': size}
        if extra:
            region.update(extra)
        self.regions.append(region)
        self.masks.append(full)

    def graphic(self, mask, geometry):
        self.regions.append(graphic_region(mask, geometry))
        self.masks.append(mask)


def layout_comment(renderer, segments, style, card, start_x, limit_x):
    """Pre-render comment tiles and assign (x, line) positions; wrap only at declared LF separators."""
    size = style['size']
    space = advance(renderer, ' ', size)
    placed, cursor, line = [], start_x, 0
    for kind, payload, separator in segments:
        text = payload['text'] if kind == 'face' else payload
        tile, mask, fonts, evidence = render_text(renderer, text, size, style['text'], card)
        if separator == '\n':
            line += 1
            cursor = start_x
        elif separator == ' ':
            cursor += space
        x = int(round(cursor))
        if x + tile.width > limit_x:
            raise Unsupported('card-row-overflow')
        placed.append((kind, payload, separator, tile, mask, fonts, evidence, x, line))
        cursor = x + tile.width
    return placed, line


def render_scene(renderer, split, rng, seen, atoms, allowed, negative=False):
    style = STYLES[split]
    width, size = style['width'], style['size']
    height = 16 + style['rows'] * (style['height'] + style['gap'])
    background = Image.new('RGB', (width, height), style['page'])
    draw = ImageDraw.Draw(background)
    cards = []
    for r in range(style['rows']):
        top = 8 + r * (style['height'] + style['gap'])
        card = rng.choice(style['cards'])
        draw.rounded_rectangle((10, top, width - 10, top + style['height']), radius=style['radius'], fill=card)
        cards.append((top, card))
    page = Page(background.copy())
    comment_kinds, emblems = Counter(), Counter()
    for top, card in cards:
        cy = top + style['height'] // 2
        if page.whole:
            page.whole += '\n'
        mask = Image.new('L', page.canvas.size)
        emblems[draw_gold_glyph(page.canvas, mask, (76, cy), style['gold'], rng)] += 1
        page.graphic(mask, 'gold-emblem')
        def column(text, x, colour=style['text'], size=size):
            tile, tmask, fonts, evidence = render_text(renderer, text, size, colour, card)
            page.paste(tile, tmask, x, cy - tile.height // 2, 'body', text, fonts, evidence, size, COLUMN_GAP)
            return x + tile.width
        cursor = column(f'Lv.{rng.randrange(10, 101)}', 146)
        icon_box = (cursor + 26, cy - 15, cursor + 56, cy + 15)
        imask = Image.new('L', page.canvas.size)
        draw_square_icon(page.canvas, imask, icon_box, rng.choice(style['icons']), rng)
        page.graphic(imask, 'square-icon')
        duty = ''.join(rng.choice(CN_WORDS) for _ in range(rng.randrange(2, 5))) + rng.choice(('', '', '！'))
        cursor = column(duty, icon_box[2] + 24)
        stamp = f'{rng.randrange(2024, 2027)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d} {rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}'
        cursor = column(stamp, cursor + 24)
        kind, segments = plan_comment(rng, split, seen, atoms, allowed, negative)
        comment_kinds[kind] += 1
        placed, lines = layout_comment(renderer, segments, style, card, cursor + 28, width - 96)
        shift = int(round(size * LINE_SHIFT)) if lines else 0
        for index, (segment_kind, payload, separator, tile, tmask, fonts, evidence, x, line) in enumerate(placed):
            centre = cy - shift if line == 0 else cy + shift
            extra = None
            if segment_kind == 'face':
                extra = {'core_key': payload['core_key'], 'core_digest': payload['core_digest'], 'core_partition': payload['core_partition'], 'structure': payload['structure']}
            text = payload['text'] if segment_kind == 'face' else payload
            page.paste(tile, tmask, x, centre - tile.height // 2, 'kaomoji' if segment_kind == 'face' else 'body', text, fonts, evidence, size,
                       COLUMN_GAP if index == 0 else separator, extra)
        close_tile, cmask, fonts, evidence = render_text(renderer, '×', size + 2, style['muted'], card)
        page.paste(close_tile, cmask, width - 64, cy - close_tile.height // 2, 'body', '×', fonts, evidence, size + 2, COLUMN_GAP)
    return page.canvas, background, page.whole, page.regions, page.masks, {'comment_kinds': dict(comment_kinds), 'emblem_styles': {str(k): v for k, v in emblems.items()},
                                                                             'scene_kind': 'card-list-negative' if negative else 'card-list'}


def validate(root):
    root = Path(root).resolve()
    meta = read(root / 'dataset.json')
    if meta.get('schema_version') != 'localocr.region-scenes.v1' or meta.get('synthetic_only') is not True or meta.get('generator_id') != GENERATOR_ID:
        raise ValueError('Scene schema/synthetic declaration')
    atoms = kg.default_atoms()
    if meta.get('atoms_binding') != kg.binding(atoms):
        raise ValueError('Scene atoms binding differs from the canonical table')
    texts, micros, cores, pixelsets, counts = {}, {}, {}, {}, {}
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
            if row['split'] != split or row['source_split'] != split or row['recipe_family'] != family(split) or row['scene_kind'] not in SCENE_KINDS:
                raise ValueError('Scene source/AST family differs')
            face_count = sum(region['kind'] == 'kaomoji' for region in row['regions'])
            if (row['scene_kind'] == 'card-list-negative') != (face_count == 0):
                raise ValueError('Negative card scenes carry no face and positive card scenes carry at least one')
            if row['provenance'] != {'source': 'original', 'license': 'CC0-1.0'} or th(row['text']) != row['text_sha256']:
                raise ValueError('Label/provenance changed')
            if row['text_sha256'] in texts and texts[row['text_sha256']] != split:
                raise ValueError('Cross-split label')
            texts[row['text_sha256']] = split
            p = (root / row['image']).resolve()
            bg = (root / row['background']).resolve()
            if not p.is_relative_to(root) or sha(p) != row['sha256'] or not bg.is_relative_to(root) or sha(bg) != row['background_sha256']:
                raise ValueError('Image/background hash/path')
            with Image.open(p) as im:
                im.load()
                arr = np.asarray(im.convert('RGB'))
                pixel = pixels(im)
                if im.size != (row['width'], row['height']) or pixel != row['pixel_sha256']:
                    raise ValueError('Scene pixels/dimensions')
            with Image.open(bg) as im:
                im.load()
                base = np.asarray(im.convert('RGB'))
            if base.shape != arr.shape:
                raise ValueError('Background dimensions differ from the scene')
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
                    if region['rendering']['glyphs'] <= 0 or not region['font_ids']:
                        raise ValueError('Missing genuine glyphs')
                if region['kind'] == 'body' and contains_face(region['text'], atoms):
                    raise ValueError('Body region carries a grammatical face')
                if region['kind'] == 'kaomoji':
                    key = th(region['text'])
                    if key in micros and micros[key] != split:
                        raise ValueError('Micro-label source leakage')
                    micros[key] = split
                    if (gc.core_digest(region['core_key']) != region['core_digest'] or gc.partition(region['core_key']) != split
                            or region['core_partition'] != split or kg.parse_face(region['text'], atoms) is None):
                        raise ValueError('Scene face core partition or grammar identity differs')
                    if region['core_digest'] in cores and cores[region['core_digest']] != split:
                        raise ValueError('Cross-split scene core leakage')
                    cores[region['core_digest']] = split
                if not np.all(np.any(arr[support] != base[support], axis=1)):
                    raise ValueError('No visible source image ink')
                masks.append(support)
            if masks and np.any(np.sum(masks, axis=0) > 1):
                raise ValueError('Region glyph pixels overlap')
            if not np.array_equal(np.any(masks, axis=0), np.any(arr != base, axis=2)):
                raise ValueError('Truth masks do not cover all actual visible ink')
        negatives = sum(row['scene_kind'] == 'card-list-negative' for row in rows)
        if not negatives:
            raise ValueError('Each split needs face-free negative card scenes')
        counts[split] = {'scenes': len(rows), 'faces': sum(r['kind'] == 'kaomoji' for row in rows for r in row['regions']), 'negative_scenes': negatives}
    binding = {'schema_version': 'localocr.region-scenes.freeze.v1', 'dataset_sha256': sha(root / 'dataset.json'),
               'split_sha256': {s: meta['splits'][s]['sha256'] for s in ('dev', 'heldout')}, 'counts': counts}
    if (root / 'freeze.json').exists() and read(root / 'freeze.json') != binding:
        raise ValueError('Scene freeze differs')
    if (root / 'heldout-seal.json').exists():
        seal = read(root / 'heldout-seal.json')
        if seal['dataset_sha256'] != binding['dataset_sha256'] or seal['heldout_sha256'] != binding['split_sha256']['heldout']:
            raise ValueError('Scene seal differs')
    return binding


def _dependency_digests():
    base = Path(__file__).resolve().parents[1]
    names = ['OcrDatasets/render.py', 'PretrainedOcr/quality_corpus.py', 'PretrainedOcr/grammar_corpus.py', 'PretrainedOcr/grammar_data.py',
             'PretrainedOcr/kaomoji_grammar.py', 'PretrainedOcr/kaomoji_atoms.json', 'RegionKaomoji/scenes.py']
    return {name: sha(base / name) for name in names}


def generate(output, fonts, counts=(60, 120), seed=20261013):
    if len(counts) != 2 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
        raise ValueError('Positive scene counts required')
    root = qd.output_root(output)
    if rp.profile_for(root) is None:
        raise ValueError('Card scenes must be generated inside a registered research round')
    if root.exists():
        raise ValueError('Fresh scene output required')
    qd.storage_gate(root, 1024 ** 2)
    renderer = MixedRenderer(fonts)
    atoms = kg.default_atoms()
    allowed, excluded_atoms = renderable_atoms(renderer, atoms)
    root.mkdir(parents=True)
    for name in ('images', 'masks', 'backgrounds'):
        (root / name).mkdir()
    writer = qd.Writer(root)
    started = time.perf_counter()
    seen, allpixels, splits, exclusions, summary, jobs = set(), set(), {}, Counter(), {}, {}
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'started_utc': utc()})
    for split, target in zip(('dev', 'heldout'), counts):
        rng = random.Random(seed + (0 if split == 'dev' else 1))
        rows, kinds, emblems = [], Counter(), Counter()
        for index in range(target):
            negative = index % NEGATIVE_EVERY == NEGATIVE_EVERY - 1
            for _ in range(200):
                try:
                    canvas, background, text, regions, masks, stats = render_scene(renderer, split, rng, seen, atoms, allowed, negative)
                except Unsupported as error:
                    exclusions[str(error)] += 1
                    continue
                pixel = pixels(canvas)
                if pixel in allpixels:
                    exclusions['duplicate-pixels'] += 1
                    continue
                break
            else:
                raise ValueError('Unable to render reliable original card scene')
            allpixels.add(pixel)
            kinds.update(stats['comment_kinds'])
            emblems.update(stats['emblem_styles'])
            identifier = f'{split}-{index:04d}'
            path = root / 'images' / (identifier + '.png')
            writer.image(path, canvas)
            bg_path = root / 'backgrounds' / (identifier + '.png')
            writer.image(bg_path, background)
            for ri, (region, mask) in enumerate(zip(regions, masks)):
                mp = root / 'masks' / f'{identifier}-{ri}.png'
                writer.image(mp, mask)
                region.update(id=f'{identifier}-region-{ri}', mask=mp.relative_to(root).as_posix(), mask_sha256=sha(mp), pixel_count=int((np.asarray(mask) > 0).sum()))
            rows.append({'id': identifier, 'split': split, 'source_split': split, 'scene_kind': stats['scene_kind'], 'recipe_family': family(split),
                         'text': text, 'text_sha256': th(text), 'image': path.relative_to(root).as_posix(), 'sha256': sha(path), 'pixel_sha256': pixel,
                         'background': bg_path.relative_to(root).as_posix(), 'background_sha256': sha(bg_path),
                         'width': canvas.width, 'height': canvas.height, 'regions': regions, 'provenance': {'source': 'original', 'license': 'CC0-1.0'},
                         'baseline_omission': 'not measured; synthetic pressure only'})
            if (index + 1) % 10 == 0:
                qd.storage_gate(root, 512 * 1024)
                atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'split': split, 'scenes': index + 1, 'target': target,
                                                      'elapsed_seconds': time.perf_counter() - started, 'updated_utc': utc()})
        writer.json(root / (split + '.json'), rows)
        splits[split] = {'file': split + '.json', 'sha256': sha(root / (split + '.json')), 'count': len(rows), 'requested': target}
        jobs[split] = [{k: row[k] for k in JOB_KEYS} for row in rows]
        summary[split] = {'comment_kinds': dict(kinds), 'emblem_styles': dict(emblems), 'negative_scenes': sum(row['scene_kind'] == 'card-list-negative' for row in rows),
                          'faces': sum(region['kind'] == 'kaomoji' for row in rows for region in row['regions']),
                          'body_regions': sum(region['kind'] == 'body' for row in rows for region in row['regions']),
                          'graphic_regions': sum(region['kind'] == 'graphic' for row in rows for region in row['regions'])}
    writer.json(root / 'dataset.json', {'schema_version': 'localocr.region-scenes.v1', 'generator_id': GENERATOR_ID, 'synthetic_only': True,
                'private_inputs_consumed': False, 'license': 'CC0-1.0', 'seed': seed, 'splits': splits,
                'recipes': {split: {kind: STYLES[split]['recipe'] for kind in SCENE_KINDS} for split in STYLES},
                'negative_scenes': {'every': NEGATIVE_EVERY, 'comment_kinds': list(NEGATIVE_KINDS), 'scene_kind': 'card-list-negative'},
                'styles': STYLES, 'comment_plans': {k: list(map(list, v)) for k, v in COMMENT_PLANS.items()}, 'comment_weights': COMMENT_WEIGHTS,
                'atoms_binding': kg.binding(atoms), 'excluded_atoms': excluded_atoms,
                'core_partition_buckets': {k: list(v) for k, v in gc.PARTITION_BUCKETS.items()},
                'renderer': renderer.versions, 'fonts': [{k: v for k, v in f.items() if k not in ('file', 'license_file')} for f in renderer.fonts],
                'font_manifest_sha256': sha(fonts), 'font_independence': False, 'vocabulary_independence': False,
                'source_sha256': sha(__file__), 'dependency_sha256': _dependency_digests(),
                'bounds': 'original-image visible-ink half-open [left,top,right,bottom)',
                'spans': 'raw Python Unicode codepoint spans [start,end); rows separated by LF, columns by two spaces, comment segments by their declared separator (none, one space, or LF for a wrapped line); separators are not regions',
                'background': 'per-scene text-free background PNG; visible ink is any pixel differing from it',
                'layout_note': 'Original geometric emblems and square icons stand in for application icons; names are original word compounds; no game assets are reproduced',
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
    parser.add_argument('--counts', nargs=2, type=int, default=(60, 120))
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    print(json.dumps(generate(args.dataset, args.fonts, tuple(args.counts), args.seed) if args.action == 'generate' else validate(args.dataset), indent=2))


if __name__ == '__main__':
    main()
