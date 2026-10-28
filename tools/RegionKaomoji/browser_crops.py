"""Browser real-font training crops (improve-browser-font-kaomoji-recognition).

The specialist and the route classifier only ever saw HarfBuzz/Noto crops. This generator renders
crop pages with the local Edge (the browser session, page script, font records and WebP round trip
of ``browser_scenes``): single-column card strips in the record-card colours, each holding one target
(a face with nearby body words, a comment phrase, punctuation, emoji, an empty strip area or a CSS
placeholder graphic). Target ink comes from a background capture and a Custom Highlight capture of
the target range; the crop is that ink box plus a 2-6 device pixel margin, cut from the full capture
or from its lossy WebP decode. Faces follow the core partition of their split, with the outer bracket
width stratified 45/45/10 inside width families and about 20% atom-table brackets outside them.
A tail-shaped right neighbour is never glued to a face.

The train split mixes about 70% browser crops with about 30% seeded, kind-stratified copies of the
grammar round's ``atoms-corpus-v1`` train rows (bytes copied, source recorded per row). Dev and
heldout are browser crops only. The dataset keeps the pretrained crop schema, so ``read_split``,
the crop evaluator, the classifier trainer and the Paddle adapter read it unchanged. Nothing is
fetched and system fonts are recorded, never copied. The sealed browser-cards-v1 heldout is never
opened: isolation from it rests on the core partition alone.
"""
from __future__ import annotations
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
import numpy as np
from PIL import Image, features
from tools.OcrDatasets.render import Unsupported
from tools.PretrainedOcr import kaomoji_grammar as kg, grammar_corpus as gc, round_profile as rp
from tools.PretrainedOcr import common, quality_data as qd
from tools.PretrainedOcr.common import write as atomic_write, utc
from tools.PretrainedOcr.data import MixedRenderer
from tools.PretrainedOcr.grammar_data import renderable_atoms
from tools.PretrainedOcr.quality_corpus import EMOJI
from . import browser_scenes as bs, card_scenes as cs
from . import exposure
from .scenes import sha, th, read, pixels, bbox

GENERATOR_ID = 'localocr.browser-crop-data.20261016.v1'
SPLITS = ('train', 'dev', 'heldout')
COUNTS = (16000, 2000, 4000)
KIND_SHARES = (('face', .60), ('plain', .20), ('punctuation', .05), ('emoji', .05), ('blank', .05), ('graphic', .05))
BROWSER_TRAIN_SHARE = .70
SOURCE = qd.ARTIFACTS / 'kaomoji-grammar-20261013/data/atoms-corpus-v1'
NONE_SHARE = .20               # faces whose outer brackets have no width family
MARGIN = (2, 6)                # device pixels around the target ink
ROWS_PER_PAGE = 24
PAGE_WIDTH = 900               # CSS pixels
FAR = '  '                     # two spaces keep non-face neighbours outside the crop margin


MAXIMUM_RETRIES = 200  # consecutive failed pages, or failed plans of one kind, before generation stops


def kind_counts(total):
    counts = {kind: int(round(total * share)) for kind, share in KIND_SHARES}
    counts['face'] += total - sum(counts.values())
    return counts


# ---------------------------------------------------------------- targets

class StratifiedFaces:
    """Faces of one split: ~20% outside width families ('none'), the rest 45/45/10 by width class."""
    def __init__(self, split, rng, atoms, allowed, pool, seen, exclusions=None):
        self.split, self.rng, self.atoms, self.allowed, self.seen = split, rng, atoms, allowed, seen
        self.exclusions = exclusions
        self.families = bs.width_families(atoms)
        self.pool = {name: [f for f in pool[split] if f['bracket_width_class'] == name
                           and (exclusions is None or not exclusions.matches(f['text'], f['core_key'])
                                and exposure.split_matches(f['text'], split, f['core_key']))]
                     for name in (*bs.WIDTH_CLASS_WEIGHTS, 'none')}

    def __call__(self):
        names = list(bs.WIDTH_CLASS_WEIGHTS)
        target = 'none' if self.rng.random() < NONE_SHARE else self.rng.choices(names, [bs.WIDTH_CLASS_WEIGHTS[n] for n in names])[0]
        if self.rng.random() < bs.COMMON_SHARE and self.pool[target]:
            return dict(self.rng.choice(self.pool[target]))
        for _ in range(50000):
            composed = gc.compose_face(self.split, self.rng, self.atoms, self.allowed)
            pair = tuple(kg.parse_face(composed['text'], self.atoms)['bracket'])
            if (target == 'none') != (pair not in self.families):
                continue
            text = composed['text'] if target == 'none' else bs.rewrite_width(composed['text'], self.atoms, self.families, target, self.rng)
            record = bs.face_record(text, self.atoms, 'atom-table-' + self.split + '-partition', composed['structure'])
            if (record['core_partition'] != self.split or th(text) in self.seen
                    or self.exclusions is not None and (self.exclusions.matches(text, record['core_key'])
                                                        or not exposure.split_matches(text, self.split, record['core_key']))):
                continue
            self.seen.add(th(text))
            return {**record, 'bracket_width_class': bs.width_class(text, self.atoms, self.families)}
        raise ValueError('Original face pool exhausted for ' + self.split)


def tail_start(face, word, atoms):
    """True when the start of ``word`` would read as a tail glued to ``face``."""
    return any(kg.parse_face(face + word[:k], atoms) is not None for k in range(1, len(word) + 1))


def words(rng, atoms, count, exclusions=None):
    if exclusions is not None:
        return exclusions.words(rng, count, atoms, strip=True)
    for _ in range(1000):
        text = cs.comment_words(rng, count)
        if (text.strip() and not cs.contains_face(text, atoms)
                and (exclusions is None or not exclusions.matches(text.strip()))):
            return text.strip()
    raise ValueError('Original comment words exhausted')


class Labels:
    """Non-face labels are unique across the whole dataset: a label already owned anywhere is never reused."""
    def __init__(self, owners, exclusions=None):
        self.owners = owners
        self.exclusions = exclusions

    def take(self, split, make, atoms):
        for _ in range(20000):
            text = make()
            if (not text or th(text) in self.owners or cs.contains_face(text, atoms)
                    or self.exclusions is not None and self.exclusions.matches(text)):
                continue
            self.owners[th(text)] = split
            return text
        raise ValueError('Original label pool exhausted for ' + split)


def covered(text, families, fonts):
    """Glyph coverage ignoring format characters (zero-width joiner, variation selectors), which draw nothing."""
    import unicodedata
    return fonts.covers(''.join(c for c in text if unicodedata.category(c) != 'Cf'), families)


def plan_target(kind, split, rng, faces, labels, atoms):
    """One row spec: (kind, left, separator_left, target, separator_right, right, extra)."""
    def body_words(count):
        extra = {'exclusions': labels.exclusions} if labels.exclusions is not None else {}
        return words(rng, atoms, count, **extra)
    if kind == 'face':
        face = faces()
        left = body_words(rng.randrange(1, 4))
        right = body_words(rng.randrange(1, 4))
        sep_left, sep_right = rng.choice(('', ' ')), rng.choice(('', ' '))
        if sep_right == '' and tail_start(face['text'], right, atoms):
            sep_right = ' '              # a tail-shaped neighbour is never glued to the face
        return {'kind': 'face', 'left': left, 'sep_left': sep_left, 'target': face['text'], 'sep_right': sep_right, 'right': right, 'face': face}
    if kind == 'plain':
        text = labels.take(split, lambda: body_words(rng.randrange(2, 7)), atoms)
    elif kind == 'punctuation':
        tokens = cs.PUNCTUATION + cs.FRAGMENTS
        text = labels.take(split, lambda: ''.join(rng.choice(tokens) for _ in range(rng.randrange(2, 5))) + rng.choice(('', str(rng.randrange(10, 1000)))), atoms)
    elif kind == 'emoji':
        text = labels.take(split, lambda: ''.join(rng.choice(EMOJI) for _ in range(rng.randrange(1, 4))), atoms)
    else:
        return {'kind': kind, 'left': body_words(rng.randrange(1, 3)), 'target': '', 'size': [rng.randrange(40, 160), rng.randrange(18, 34)],
                'geometry': rng.choice(('badge-ring', 'badge-double', 'badge-diamond', 'icon-circle', 'icon-triangle')) if kind == 'graphic' else None}
    left = body_words(rng.randrange(1, 3)) if rng.random() < .5 else ''
    right = body_words(rng.randrange(1, 3)) if rng.random() < .5 else ''
    return {'kind': kind, 'left': left, 'sep_left': FAR if left else '', 'target': text, 'sep_right': FAR if right else '', 'right': right}


# ---------------------------------------------------------------- page

def page_style(rng):
    scale = rng.choice(bs.DEVICE_SCALES)
    size = rng.choice(bs.FONT_SIZES)
    row_height = int(size * 1.6) + 8
    rows_height = ROWS_PER_PAGE * (row_height + 6) + 12
    webp = rng.random() < bs.WEBP['probability']
    palette = rng.choice(sorted(bs.PALETTES))
    return {'device_scale_factor': scale, 'viewport': [PAGE_WIDTH, rows_height], 'rows': ROWS_PER_PAGE, 'font_size': size, 'row_height': row_height,
            'palette': palette, 'colours': [list(c) for c in bs.PALETTES[palette]],
            'delivery': {'format': 'webp', 'quality': rng.randrange(bs.WEBP['quality'][0], bs.WEBP['quality'][1] + 1), 'method': bs.WEBP['method']}
            if webp else {'format': 'png', 'quality': None, 'method': None}}


def crop_page_html(style, specs):
    """Card strips; a text target is a UTF-16 range of the strip's single comment text node."""
    size, c0, c1 = style['font_size'], *style['colours']
    css = (f'html,body{{margin:0;background:#fff;overflow:hidden}}body{{font-family:{bs.FONT_FAMILY};font-size:{size}px;line-height:1.3}}'
           f'.list{{padding:6px 10px 0}}.card{{display:flex;align-items:center;height:{style["row_height"]}px;margin-bottom:6px;border-radius:10px;'
           f'padding:0 18px;color:#fff;overflow:hidden;background:linear-gradient(90deg,{bs._rgb(c0)},{bs._rgb(c1)})}}'
           f'.comment{{flex:none;white-space:pre}}.blank{{flex:none;display:inline-block;margin-left:48px}}'
           f'.badge{{flex:none;width:{size + 10}px;height:{size + 10}px;margin-right:24px;box-sizing:border-box;border:4px solid {bs._rgb(bs.GOLD)};border-radius:50%}}'
           f'.badge.double{{box-shadow:inset 0 0 0 5px transparent,inset 0 0 0 8px {bs._rgb(bs.GOLD)}}}'
           f'.badge.diamond{{width:{size}px;height:{size}px;border-radius:3px;transform:rotate(45deg)}}'
           f'.icon{{flex:none;position:relative;width:{size + 8}px;height:{size + 8}px;margin-right:24px;border-radius:5px;background:{bs._rgb(bs.ICON_COLOURS[0])}}}'
           f'.icon i{{position:absolute;left:25%;top:25%;width:50%;height:50%;background:#eef0f4}}'
           f'.icon.circle i{{border-radius:50%}}.icon.triangle i{{clip-path:polygon(50% 0,100% 100%,0 100%)}}')
    rows = []
    for r, spec in enumerate(specs):
        if spec['kind'] in ('blank', 'graphic'):
            graphic = ''
            if spec['kind'] == 'graphic':
                cls = spec['geometry'].replace('badge-ring', 'badge').replace('badge-', 'badge ').replace('icon-', 'icon ')
                graphic = f'<div class="{cls}" data-r="{r}">' + ('<i></i>' if cls.startswith('icon') else '') + '</div>'
            blank = (f'<span class="blank" data-b="{r}" style="width:{spec["size"][0]}px;height:{spec["size"][1]}px"></span>'
                     if spec['kind'] == 'blank' else '')
            rows.append(f'<div class="card">{graphic}<div class="comment" data-segs="">{html.escape(spec["left"])}</div>{blank}</div>')
            continue
        before = spec['left'] + spec['sep_left']
        start = bs._utf16(before)
        text = before + spec['target'] + spec['sep_right'] + spec['right']
        rows.append(f'<div class="card"><div class="comment" data-segs="{r}:{start}:{start + bs._utf16(spec["target"])}">{html.escape(text)}</div></div>')
    script = bs.PAGE_SCRIPT + ('window.__rects=function(){const s=window.devicePixelRatio;const box=e=>{const r=e.getBoundingClientRect();'
                               'return [r.left*s,r.top*s,r.right*s,r.bottom*s];};return {cards:Array.from(document.querySelectorAll(".card")).map(box),'
                               'comments:Array.from(document.querySelectorAll(".comment")).map(box),blanks:Object.fromEntries('
                               'Array.from(document.querySelectorAll("[data-b]")).map(e=>[e.dataset.b,box(e)]))};};')
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{css}::highlight(only){{color:#fff}}</style></head>'
            f'<body><div class="list">{"".join(rows)}</div><script>{script}</script></body></html>')


def crop_box(ink, card, rng, shape):
    left, top, right, bottom = ink
    box = [left - rng.randint(*MARGIN), top - rng.randint(*MARGIN), right + rng.randint(*MARGIN), bottom + rng.randint(*MARGIN)]
    box = [max(box[0], int(np.ceil(card[0]))), max(box[1], int(np.ceil(card[1]))), min(box[2], int(card[2])), min(box[3], int(card[3]))]
    box = [max(0, box[0]), max(0, box[1]), min(shape[1], box[2]), min(shape[0], box[3])]
    if box[0] > left or box[1] > top or box[2] < right or box[3] < bottom:
        raise Unsupported('ink-outside-card')
    return box


def render_page(session, style, specs, path, fonts, rng):
    """Crops for one page; rows that fail get ``None`` with a reason, the page itself never half-succeeds."""
    path.write_text(crop_page_html(style, specs), encoding='utf-8')
    page = bs.ScenePage(session, style, path)
    results = [None] * len(specs)
    try:
        page.state('none')
        background = page.capture()
        page.state('none')
        if not np.array_equal(background, page.capture()):
            raise Unsupported('unstable-background')
        rects = page.page.evaluate('() => window.__rects()')
        masks = {}
        for r, spec in enumerate(specs):
            if spec['kind'] == 'blank':
                continue
            page.state('element' if spec['kind'] == 'graphic' else 'range', r)
            masks[r] = np.any(page.capture() != background, axis=2)
        page.state('full')
        full = page.capture()
        page.segment()
        used = {r: page.platform_fonts(r, comment=True) for r, spec in enumerate(specs) if spec['kind'] not in ('blank', 'graphic')}
    finally:
        page.close()
    session.check()
    delivery = style['delivery']
    delivered = bs.webp_roundtrip(full, delivery['quality'], delivery['method']) if delivery['format'] == 'webp' else full
    ink_all = np.any(full != background, axis=2)
    for r, spec in enumerate(specs):
        card, comment = rects['cards'][r], rects['comments'][r]
        if comment[2] > card[2] - 10:
            results[r] = 'row-overflow'
            continue
        if spec['kind'] == 'blank':
            ink = [int(round(v)) for v in rects['blanks'][str(r)]]
            try:
                box = crop_box(ink, card, rng, full.shape)
            except Unsupported as error:
                results[r] = str(error)
                continue
            if ink_all[box[1]:box[3], box[0]:box[2]].any():
                results[r] = 'blank-has-ink'
                continue
            region = [0, 0, box[2] - box[0], box[3] - box[1]]
        else:
            if not masks[r].any():
                results[r] = 'target-without-ink'
                continue
            ink = bbox(masks[r])
            if spec['kind'] != 'graphic' and not covered(spec['target'], [f['family'] for f in used[r]], fonts):
                results[r] = 'uncovered-glyph'
                continue
            try:
                box = crop_box(ink, card, rng, full.shape)
            except Unsupported as error:
                results[r] = str(error)
                continue
            region = [ink[0] - box[0], ink[1] - box[1], ink[2] - ink[0], ink[3] - ink[1]]
        results[r] = {'image': np.ascontiguousarray(delivered[box[1]:box[3], box[0]:box[2]]), 'box': box, 'region': region,
                      'platform_fonts': used.get(r, [])}
    return results, rects


# ---------------------------------------------------------------- dataset

class ExposedSourceExhausted(ValueError):
    """No exact kind-stratified whole-group sample survives exposure exclusions."""


def copy_source_rows(writer, root, rng, counts, owners, source, exclusions=None):
    """Seeded, kind-stratified copies of whole groups from the grammar round's train split."""
    rows = common.read_split(source, 'train')
    groups = {}
    for row in rows:
        groups.setdefault(row['group_id'], []).append(row)
    by_kind = {}
    for group in groups.values():
        if exclusions is not None and any(exclusions.matches(text, key) or not exposure.split_matches(text, 'train', key)
                                           for row in group for text, key in exposure.targets(row)):
            continue
        by_kind.setdefault(group[0]['kind'], []).append(group)
    selected = []
    for kind, total in counts.items():
        pool = sorted(by_kind.get(kind, []), key=lambda g: g[0]['id'])
        if exclusions is None:
            chosen = rng.sample(pool, total // len(pool[0]))
        else:
            # Determine the whole sample before copying any bytes. A shortage can safely
            # switch this unused train contract to original browser rows without partial output.
            rng.shuffle(pool)
            reachable = {0: []}
            for group in pool:
                for size, chosen_so_far in list(reachable.items()):
                    size += len(group)
                    if size <= total and size not in reachable:
                        reachable[size] = chosen_so_far + [group]
                if total in reachable:
                    break
            if total not in reachable:
                raise ExposedSourceExhausted('Copied train source exhausted by exposure exclusions: ' + kind)
            chosen = reachable[total]
        selected.extend(chosen)
    copied = []
    for group in selected:
        for row in group:
            identifier = 'train-src-' + row['id']
            path = root / 'images' / (identifier + '.png')
            writer.bytes(path, (source / row['image']).read_bytes())
            owners.setdefault(row['text_sha256'], 'train')
            copied.append({**row, 'id': identifier, 'image': path.relative_to(root).as_posix(), 'group_id': 'src:' + row['group_id'],
                           'origin': 'noto-grammar-copy', 'source': {'dataset': _relative(source),
                                                                    'row_id': row['id'], 'sha256': row['sha256']}})
    if len(copied) != sum(counts.values()):
        raise RuntimeError('Copied grammar rows differ from the requested kind counts')
    return copied


def browser_rows(session, writer, root, split, specs, rng, fonts, exclusions, replan, started, pixel_owner):
    """Render ``specs`` page by page; a failed row is replaced by a fresh spec of the same kind."""
    queue, rows, page_index, failures = list(specs), [], 0, 0
    while queue:
        batch, queue = queue[:ROWS_PER_PAGE], queue[ROWS_PER_PAGE:]
        style = page_style(rng)
        attempt = session.temp_root / 'crop-page.html'
        try:
            results, _ = render_page(session, style, batch, attempt, fonts, rng)
        except Unsupported as error:
            exclusions['page:' + str(error)] += 1
            failures += 1
            if failures >= MAXIMUM_RETRIES:
                raise RuntimeError(f'{failures} consecutive {split} pages failed: ' + json.dumps(dict(exclusions), sort_keys=True))
            queue = batch + queue
            continue
        failures = 0
        page_name = f'{split}-page-{page_index:05d}'
        page_path = root / 'pages' / (page_name + '.html')
        writer.bytes(page_path, attempt.read_bytes())
        for r, (spec, result) in enumerate(zip(batch, results)):
            if not isinstance(result, dict):
                exclusions[result] += 1
                queue.append(replan(spec['kind']))
                continue
            image = Image.fromarray(result['image'])
            pixel = pixels(image)
            if pixel in pixel_owner:
                exclusions['duplicate-pixels'] += 1
                queue.append(replan(spec['kind']))
                continue
            pixel_owner[pixel] = split
            rows.append((spec, result, image, pixel, style, page_name, r))
        page_index += 1
        if page_index % 10 == 0:
            qd.storage_gate(root, 4 * 1024 ** 2)
            atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'split': split, 'rows': len(rows), 'target': len(specs),
                                                  'elapsed_seconds': time.perf_counter() - started, 'updated_utc': utc()})
    return rows


def row_record(root, writer, split, index, item, atoms):
    spec, result, image, pixel, style, page_name, r = item
    identifier = f'{split}-{index:06d}'
    path = root / 'images' / (identifier + '.png')
    writer.image(path, image)
    kind = spec['kind']
    record = {'id': identifier, 'split': split, 'kind': kind, 'subtype': {'emoji': 'emoji-sequence'}.get(kind, kind), 'recipe': kind,
              'text': spec['target'], 'text_sha256': th(spec['target']), 'group_id': f'{split}-browser-{index:06d}', 'image': path.relative_to(root).as_posix(),
              'sha256': sha(path), 'pixel_sha256': pixel, 'width': image.width, 'height': image.height, 'region': result['region'],
              'font_id': sorted({f['family'] for f in result['platform_fonts']}),
              'rendering': {'text_present': bool(spec['target']), 'glyphs': sum(f['glyphs'] for f in result['platform_fonts']), 'platform_fonts': result['platform_fonts']},
              'transform': {'device_scale_factor': style['device_scale_factor'], 'font_size': style['font_size'], 'palette': style['palette'],
                            'delivery': style['delivery'], 'crop_box': result['box'], 'page': page_name, 'row': r,
                            'neighbours': {'left': spec.get('left', ''), 'sep_left': spec.get('sep_left', ''), 'right': spec.get('right', ''), 'sep_right': spec.get('sep_right', '')}},
              'origin': 'browser', 'provenance': {'source': 'original', 'license': 'CC0-1.0'}}
    if kind == 'face':
        face = spec['face']
        record.update({key: face[key] for key in ('core_key', 'core_digest', 'core_partition', 'source', 'bracket_width_class')},
                      recipe_expression=face['structure'], family='browser-face:' + th(face['structure']))
    elif kind in ('blank', 'graphic'):
        record.update(recipe_expression=spec.get('geometry') or 'card-strip-blank', family='browser-negative:' + kind)
        if kind == 'graphic':
            record['geometry'] = spec['geometry']
    else:
        record.update(recipe_expression='card-' + kind, family='browser-' + kind)
    return record


def split_specs(split, total, rng, faces, labels, atoms):
    counts = kind_counts(total)
    kinds = [kind for kind, n in counts.items() for _ in range(n)]
    rng.shuffle(kinds)
    specs = []
    for kind in kinds:
        specs.append(_plan(kind, split, rng, faces, labels, atoms))
    return specs


def environment(session, fonts, rows):
    from importlib.metadata import version
    families = sorted({family for row in rows if row.get('origin') == 'browser' for family in row['font_id']})
    return {'browser': {'product': 'Microsoft Edge', 'version': session.version, 'channel': bs.DRIVER['channel'], 'headless': True,
                        'launch_args': list(bs.LAUNCH_ARGS), 'allowed_schemes': list(bs.LOCAL_SCHEMES), 'blocked_requests': len(session.blocked)},
            'driver': {**bs.DRIVER, 'installed': version('playwright')}, 'windows': platform.platform(), 'python': platform.python_version(),
            'pillow': Image.__version__, 'webp_codec': features.version('webp'),
            'platform_fonts': {family: fonts.files(family)[0] for family in families},
            'fonts_distributed': False, 'images_distributable': False}


def summarize(rows):
    browser = [r for r in rows if r.get('origin') == 'browser']
    faces = [r for r in browser if r['kind'] == 'face']
    return {'rows': len(rows), 'browser_rows': len(browser), 'copied_rows': len(rows) - len(browser),
            'kinds': dict(Counter(r['kind'] for r in rows)), 'browser_kinds': dict(Counter(r['kind'] for r in browser)),
            'bracket_width_classes': dict(Counter(r['bracket_width_class'] for r in faces)),
            'face_sources': dict(Counter(r['source'] for r in faces)),
            'device_scale_factors': {str(k): v for k, v in Counter(r['transform']['device_scale_factor'] for r in browser).items()},
            'deliveries': dict(Counter(r['transform']['delivery']['format'] for r in browser)),
            'font_sizes': {str(k): v for k, v in Counter(r['transform']['font_size'] for r in browser).items()},
            'glued_faces': sum(r['transform']['neighbours']['sep_left'] == '' or r['transform']['neighbours']['sep_right'] == '' for r in faces),
            'platform_fonts': dict(Counter(f for r in browser for f in r['font_id']))}


def generate(output, fonts, counts=COUNTS, seed=20261016, source=None, exclusions=None):
    """Render, validate, freeze and seal the browser crop dataset inside the registered round."""
    if len(counts) != 3 or any(not isinstance(n, int) or isinstance(n, bool) or n < 100 for n in counts):
        raise ValueError('Three split counts of at least 100 rows are required')
    source = SOURCE if source is None else Path(source)
    root = qd.output_root(output)
    profile = rp.profile_for(root)
    if (profile is None or profile.change not in (rp.CURRENT.change, rp.OWNED_POSITION.change, rp.FIRST_PASS_POSITION.change)
            or root.exists()):
        raise ValueError("Fresh browser crops must be generated inside the current round's registered research root")
    qd.storage_gate(root, 256 * 1024 ** 2)
    excluded = exposure.load(exclusions) if exclusions is not None else None
    renderer, atoms = MixedRenderer(fonts), kg.default_atoms()
    allowed, excluded_atoms = renderable_atoms(renderer, atoms)
    def renderable(text):
        try:
            renderer.shape(text)
        except Unsupported:
            return False
        return True
    pool = bs.common_pool(atoms, renderable, exclusions=excluded)
    root.mkdir(parents=True)
    for name in ('images', 'pages'):
        (root / name).mkdir()
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'copying-source', 'started_utc': utc()})
    writer, font_files, started = qd.Writer(root), bs.FontFiles(), time.perf_counter()
    owners, pixel_owner, seen_faces, exclusions, splits, rows_by_split = {}, {}, set(), Counter(), {}, {}
    browser_train = int(round(counts[0] * BROWSER_TRAIN_SHARE))
    train_source_fallback = False
    try:
        copied = copy_source_rows(writer, root, random.Random(seed + 100), kind_counts(counts[0] - browser_train), owners, source, exclusions=excluded)
    except ExposedSourceExhausted:
        if excluded is None:
            raise
        copied, browser_train, train_source_fallback = [], counts[0], True
    for row in copied:
        pixel_owner[row['pixel_sha256']] = 'train'
    with bs.BrowserSession(root / 'browser-temp') as session:
        for index, split in enumerate(SPLITS):
            rng = random.Random(seed + index)
            faces = StratifiedFaces(split, rng, atoms, allowed, pool, seen_faces, exclusions=excluded)
            labels = Labels(owners, exclusions=excluded)
            def replan(kind, split=split, rng=rng, faces=faces, labels=labels):
                return _plan(kind, split, rng, faces, labels, atoms)
            total = browser_train if split == 'train' else counts[index]
            items = browser_rows(session, writer, root, split, split_specs(split, total, rng, faces, labels, atoms), rng, font_files,
                                 exclusions, replan, started, pixel_owner)
            records = [row_record(root, writer, split, i, item, atoms) for i, item in enumerate(items)]
            records += copied if split == 'train' else []
            writer.json(root / (split + '.json'), records)
            rows_by_split[split] = records
            splits[split] = {'file': split + '.json', 'sha256': sha(root / (split + '.json')), 'count': len(records),
                             'groups': len({r['group_id'] for r in records}), 'positive_groups': len({r['group_id'] for r in records if r['text']}),
                             'browser_rows': sum(r.get('origin') == 'browser' for r in records), 'copied_rows': sum(r.get('origin') != 'browser' for r in records)}
        env = environment(session, font_files, [r for rows in rows_by_split.values() for r in rows])
    if env['browser']['blocked_requests']:
        raise RuntimeError('Non-local browser requests were attempted')
    summary = {split: summarize(rows) for split, rows in rows_by_split.items()}
    train_vocabulary = sorted({c for row in rows_by_split['train'] for c in row['text']}, key=ord)
    writer.json(root / 'dataset.json', {
        'schema_version': 'localocr.pretrained.dataset.v1', 'generator_id': GENERATOR_ID, 'synthetic_only': True, 'browser_rendered': True,
        'private_inputs_consumed': False, 'license': 'CC0-1.0',
        'license_note': 'Labels are original CC0 text. Browser crops are rendered with locally licensed Windows system fonts for local training and evaluation only; crops and any weights trained on them stay in ignored directories, and any release needs a separate license review.',
        'seed': seed, 'requested_counts': dict(zip(SPLITS, counts)), 'splits': splits, 'train_vocabulary': train_vocabulary, 'kind_shares': dict(KIND_SHARES),
        'browser_train_share': browser_train / counts[0] if excluded is not None else BROWSER_TRAIN_SHARE, 'none_share': NONE_SHARE, 'width_class_weights': bs.WIDTH_CLASS_WEIGHTS, 'margin_device_pixels': list(MARGIN),
        'rows_per_page': ROWS_PER_PAGE, 'device_scales': list(bs.DEVICE_SCALES), 'font_sizes': list(bs.FONT_SIZES), 'webp': bs.WEBP, 'font_family': bs.FONT_FAMILY,
        'source': {'dataset': _relative(source),
                   'freeze_sha256': sha(source / 'freeze.json'), 'train_sha256': read(source / 'dataset.json')['splits']['train']['sha256']},
        'isolation': {'faces': 'core partition per split (train 0-6, dev 7, heldout 8-9)', 'labels': 'non-empty labels disjoint across splits', 'pixels': 'disjoint across splits',
                      'browser_cards_v1': ('exposed target labels bound by cross-round inventory; no source images or predictions read'
                                           if excluded is not None else 'never opened; isolation from its sealed heldout rests on the core partition')},
        'atoms_binding': kg.binding(atoms), 'excluded_atoms': excluded_atoms, 'environment': env,
        'source_sha256': sha(__file__), 'dependency_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('browser_scenes.py', 'card_scenes.py', 'known_input.py', 'scenes.py') + (('exposure.py',) if excluded is not None else ())},
        'exclusions': dict(exclusions), 'summary': summary,
        **({'exposure_exclusions': excluded.binding,
            'exposure_split_policy': 'raw-and-width-sibling split agreement',
            'exposure_distribution': {'common_pool_available': {s: len(rows) for s, rows in pool.items()},
                                      'empty_common_class_fallback': 'original atom compositions',
                                      'train_source_exhausted': train_source_fallback,
                                      'train_used_for_training': False,
                                      'body_word_recovery': excluded.word_recovery_summary()}} if excluded is not None else {}),
        'usage': {'train': 'schema contract only; not used for training in this CPU round' if excluded is not None else 'training (specialist warm start, classifier)',
                  'dev': 'crop development selection', 'heldout': 'sealed first evaluation after the dual freeze'}})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'validating', 'updated_utc': utc()})
    binding = validate(root)
    writer.json(root / 'heldout-seal.json', {'schema_version': 'localocr.heldout-seal.v1', 'dataset_sha256': binding['dataset_sha256'],
                'heldout_sha256': binding['split_sha256']['heldout'], 'usage': 'first evaluation after model+router dual freeze',
                'labels_used_for_vocabulary': False, 'results_seen': False})
    writer.json(root / 'generation.json', {'schema_version': 'localocr.browser-crop-generation.v1', 'binding': binding, 'elapsed_seconds': time.perf_counter() - started,
                'summary': summary, 'exclusions': dict(exclusions), 'CPU_model_inference': False, 'GPU_used': False, 'blocked_requests': 0,
                'fonts_distributed': False, 'private_inputs_consumed': False, 'storage': qd.storage_gate(root)})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'complete', 'completed_utc': utc(), 'dataset_sha256': binding['dataset_sha256']})
    writer.json(root / 'freeze.json', binding)
    return binding


def _relative(path):
    path = Path(path).resolve()
    repo = qd.ARTIFACTS.parent.resolve()
    return path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path)


def _try_plan(kind, split, rng, faces, labels, atoms):
    try:
        return plan_target(kind, split, rng, faces, labels, atoms)
    except Unsupported:
        return None


def _plan(kind, split, rng, faces, labels, atoms):
    for _ in range(MAXIMUM_RETRIES):
        spec = _try_plan(kind, split, rng, faces, labels, atoms)
        if spec is not None:
            return spec
    raise RuntimeError(f'No renderable {kind} target for {split} after {MAXIMUM_RETRIES} plans')


# ---------------------------------------------------------------- validation

def _check_row(root, row, split, atoms, families, source_rows):
    if row['split'] != split or row['provenance'] != {'source': 'original', 'license': 'CC0-1.0'} or th(row['text']) != row['text_sha256']:
        raise ValueError('Split, provenance or label digest differs')
    path = (root / row['image']).resolve()
    if not path.is_relative_to(root) or sha(path) != row['sha256']:
        raise ValueError('Image path/hash changed')
    with Image.open(path) as image:
        image.load()
        if image.size != (row['width'], row['height']) or pixels(image) != row['pixel_sha256']:
            raise ValueError('Decoded pixels changed')
    if row['kind'] == 'face':
        # A copied grammar face keeps its generator's composition key; parsing a decorated face can read
        # its atoms differently, so only browser faces are re-derived by parsing.
        if row.get('origin') == 'browser':
            parsed = kg.parse_face(row['text'], atoms)
            key = None if parsed is None else gc.core_key(parsed['bracket'], *parsed['core'])
        else:
            original = source_rows.get(row['source']['row_id'], {})
            key = original.get('core_key')
        if (key is None or row['core_key'] != key or gc.core_digest(row['core_key']) != row['core_digest']
                or gc.partition(row['core_key']) != split or row['core_partition'] != split):
            raise ValueError('Face core partition or grammar identity differs')
    if row.get('origin') == 'browser':
        if row['kind'] == 'face' and bs.width_class(row['text'], atoms, families) != row['bracket_width_class']:
            raise ValueError('Bracket width class differs from the text')
        transform = row['transform']
        if transform['device_scale_factor'] not in bs.DEVICE_SCALES or transform['font_size'] not in bs.FONT_SIZES:
            raise ValueError('Undeclared page scale or size')
        if row['kind'] == 'face' and transform['neighbours']['sep_right'] == '' and tail_start(row['text'], transform['neighbours']['right'], atoms):
            raise ValueError('A tail-shaped neighbour is glued to the face')
        if row['text'] and (row['rendering']['glyphs'] <= 0 or not row['font_id']):
            raise ValueError('Missing genuine glyphs')
    else:
        original = source_rows.get(row['source']['row_id'])
        if split != 'train' or row.get('origin') != 'noto-grammar-copy' or original is None or original['sha256'] != row['sha256'] or original['text'] != row['text']:
            raise ValueError('Copied row differs from its source')


def validate(root):
    root = Path(root).resolve()
    meta = read(root / 'dataset.json')
    if (meta.get('schema_version') != 'localocr.pretrained.dataset.v1' or meta.get('generator_id') != GENERATOR_ID
            or meta.get('synthetic_only') is not True or meta['environment']['browser']['blocked_requests'] != 0):
        raise ValueError('Unknown browser crop dataset schema/generator')
    atoms = kg.default_atoms()
    excluded = exposure.load_bound(meta['exposure_exclusions']) if 'exposure_exclusions' in meta else None
    if excluded is not None:
        if meta.get('exposure_split_policy') != 'raw-and-width-sibling split agreement':
            raise ValueError('Cross-round exposure split policy differs')
        share = meta.get('browser_train_share')
        expected_share = 1.0 if meta.get('exposure_distribution', {}).get('train_source_exhausted') else round(meta['requested_counts']['train'] * BROWSER_TRAIN_SHARE) / meta['requested_counts']['train']
        if share != expected_share:
            raise ValueError('Exclusion-aware train share differs from its declared fallback')
    if meta.get('atoms_binding') != kg.binding(atoms):
        raise ValueError('Dataset atoms binding differs from the canonical table')
    source = Path(meta['source']['dataset'])
    source = source if source.is_absolute() else qd.ARTIFACTS.parent / source
    if sha(source / 'freeze.json') != meta['source']['freeze_sha256']:
        raise ValueError('Source dataset changed')
    source_rows = {row['id']: row for row in common.read_split(source, 'train')}
    families, labels, pixelset, counts, tokens = bs.width_families(atoms), {}, {}, {}, set()
    for split in SPLITS:
        info = meta['splits'][split]
        relative = Path(info['file'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts or sha(root / relative) != info['sha256']:
            raise ValueError('Split path or hash changed')
        rows = read(root / relative)
        if len(rows) != info['count'] or len(rows) != meta['requested_counts'][split] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Split count or identities changed')
        for row in rows:
            if excluded is not None:
                excluded.assert_clear(row)
            _check_row(root, row, split, atoms, families, source_rows)
            if row['text'] and labels.setdefault(row['text_sha256'], split) != split:
                raise ValueError('Cross-split label leakage')
            if pixelset.setdefault(row['pixel_sha256'], split) != split:
                raise ValueError('Cross-split pixels')
            if split == 'train':
                tokens.update(row['text'])
        browser = [r for r in rows if r.get('origin') == 'browser']
        if excluded is not None and split == 'train' and len(browser) != round(len(rows) * meta['browser_train_share']):
            raise ValueError('Actual train mix differs from exclusion-aware browser share')
        if split != 'train' and len(browser) != len(rows):
            raise ValueError('Dev and heldout must be browser crops only')
        expected = kind_counts(len(browser))
        if dict(Counter(r['kind'] for r in browser)) != {k: v for k, v in expected.items() if v}:
            raise ValueError('Browser kind shares differ from the declaration')
        counts[split] = {'count': len(rows), 'groups': len({r['group_id'] for r in rows}), 'positive_groups': len({r['group_id'] for r in rows if r['text']})}
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('action', choices=('generate', 'validate'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--fonts')
    parser.add_argument('--counts', nargs=3, type=int, default=COUNTS)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--exclusions', help='Frozen cross-round exposure inventory')
    args = parser.parse_args(argv)
    value = generate(args.dataset, args.fonts, tuple(args.counts), args.seed, exclusions=args.exclusions) if args.action == 'generate' else validate(args.dataset)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
