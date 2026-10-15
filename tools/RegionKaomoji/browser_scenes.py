"""Browser real-font record-card scenes (add-browser-rendering-evaluation).

The record list the APP reads is a web page drawn by the user's browser with Windows system fonts,
which the HarfBuzz/Noto scene renderer cannot reproduce. This generator writes an original
record-card page per scene, renders it with the locally installed Microsoft Edge through Playwright
and derives exact truth from the browser itself: a text-free background capture, one capture per
region with only that region visible, and the full capture. Nothing is fetched: only ``file:`` and
``data:`` requests are allowed and any other request fails the generation. System fonts are never
copied; the dataset records which platform fonts each region used and the files' SHA256.

Text follows the card-scene vocabulary without wrapping. Faces come from the original CC0 common
listing (plus original paste-style additions) and atom-table compositions; the outermost bracket
pair of a width family is rewritten to ASCII, fullwidth or mixed width (45/45/10) before the core
partition decides dev or heldout, because the bracket pair is part of the core key. Users may type or
paste either width, so no input method or width is assumed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import html
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
import unicodedata
import numpy as np
from PIL import Image, features
from tools.OcrDatasets.render import Unsupported
from tools.PretrainedOcr import kaomoji_grammar as kg, grammar_corpus as gc, round_profile as rp
from tools.PretrainedOcr import quality_data as qd
from tools.PretrainedOcr.common import write as atomic_write, utc
from tools.PretrainedOcr.data import MixedRenderer
from tools.PretrainedOcr.grammar_data import renderable_atoms
from . import card_scenes as cs
from .known_input import COMMON_FACES
from .scenes import sha, th, read, pixels, bbox

GENERATOR_ID = 'localocr.browser-card-scenes.20261015.v1'
DRIVER = {'package': 'playwright', 'version': '1.63.0', 'channel': 'msedge'}
LAUNCH_ARGS = ('--disable-background-networking', '--disable-component-update', '--disable-sync', '--no-pings',
               '--disable-default-apps', '--disable-extensions', '--no-first-run', '--disable-domain-reliability',
               '--disable-client-side-phishing-detection', '--host-resolver-rules=MAP * ~NOTFOUND')
LOCAL_SCHEMES = ('file:', 'data:')
JOB_KEYS = cs.JOB_KEYS
SCENE_KINDS = cs.SCENE_KINDS
DEVICE_SIZE = (1860, 981)
DEVICE_SCALES = (1.0, 1.25, 1.5)
FONT_SIZES = (16, 17, 18, 19, 20)
FONT_FAMILY = '"Microsoft YaHei", sans-serif'
WIDTH_CLASS_WEIGHTS = {'ascii': 45, 'fullwidth': 45, 'mixed': 10}
NON_FAMILY_ACCEPT = .15         # atom compositions with a bracket outside a width family are kept at this rate
COMMON_SHARE = .4               # share of faces drawn from the common paste-style listing
WEBP = {'probability': .5, 'quality': (70, 95), 'method': 4}
# Measured from the record-list screenshots already demoted to a development probe (layout and colour only).
LAYOUT = {'card_height': 84, 'gap': 13, 'top': 6, 'side': 10, 'radius': 12, 'badge': 62, 'icon': 30}
PALETTES = {'blue': ((32, 54, 93), (54, 67, 146)), 'green': ((38, 52, 37), (59, 100, 39)), 'red': ((63, 44, 44), (99, 51, 51))}
GOLD = (206, 176, 104)
ICON_COLOURS = ((44, 150, 168), (60, 120, 200), (210, 132, 64))
PAGE_COLOUR = (255, 255, 255)
# Card-scene comment plans without any wrapped, long or cut layout (comments stay on one line).
PLANS = {k: v for k, v in cs.COMMENT_PLANS_V2.items() if not any(sep in ('\n', 'long', 'split') for _, sep in v)}
PLAN_WEIGHTS = {k: cs.COMMENT_WEIGHTS_V2[k] for k in PLANS}
NEGATIVE_PLANS = ('empty', 'plain', 'fragment-tail')
NEGATIVE_EVERY = cs.NEGATIVE_EVERY
COLUMN_GAP = cs.COLUMN_GAP
# Original paste-style expressions written for this round (CC0); each parses under the atom grammar and
# renders with the registered OFL fonts, like ``COMMON_FACES``.
PASTE_FACES = (
    '(｡･ω･｡)', '(｀・ω・´)ゞ', '(*´∀`)~♥', '(｡ŏ﹏ŏ)', '(*≧ω≦)', '(≧∀≦)ゞ', '(´；ω；`)', '(T▽T)', '(⌒▽⌒)', '(´・ω・)つ',
    '(・∀・)', '(´∀｀)', '(￣▽￣)', '(￣ω￣)', '(￣ー￣)', '(´-ω-`)', '(>_<)ノ', '(^_^)v', '(・ω・)ノ', '(*´ω`*)', '(◎_◎;)',
    '(*^▽^*)', '(╥_╥)', '(ಥ_ಥ)', '(ಥ﹏ಥ)', '(｡♥‿♥｡)', '(◠‿◠)', '(◕‿◕)', '(°ロ°)', '(・_・;)', '(´Д｀)', '(;´∀`)',
    '(*´▽`)ノノ', '(´ε｀ )', 'ヾ(^▽^*)', '(～￣▽￣)～', '(｀ε´)', '(；・∀・)', '(・`ω´・)', '(^ω^)ノ', '(・ε・)', '(°ー°〃)', '(´・_・`)',
)
RECIPES = {split: f'browser record-card list {split}: Edge real-font rendering, Microsoft YaHei, DSF 1.0/1.25/1.5, '
                  f'16-20px text, blue/green/red gradient cards, single-line comments, outer bracket width 45/45/10, '
                  f'half of the scenes delivered through lossy webp' for split in ('dev', 'heldout')}


def family(split):
    return 'browser-card-scene:' + th(RECIPES[split])


# ---------------------------------------------------------------- bracket width

def width_families(atoms):
    """Declared bracket pairs grouped by NFKC form; a family needs ASCII, fullwidth and mixed members."""
    groups = {}
    for pair in atoms.brackets:
        canonical = tuple(unicodedata.normalize('NFKC', c) for c in pair)
        groups.setdefault(canonical, []).append(tuple(pair))
    families = {}
    for canonical, pairs in groups.items():
        by_class = {}
        for pair in pairs:
            by_class.setdefault(pair_class(pair), []).append(pair)
        if all(by_class.get(name) for name in WIDTH_CLASS_WEIGHTS):
            for pair in pairs:
                families[pair] = {name: sorted(by_class[name]) for name in WIDTH_CLASS_WEIGHTS}
    return families


def pair_class(pair):
    narrow = [c.isascii() for c in pair]
    return 'ascii' if all(narrow) else 'fullwidth' if not any(narrow) else 'mixed'


def width_class(text, atoms, families):
    parsed = kg.parse_face(text, atoms)
    if parsed is None:
        raise ValueError('Not a complete expression: ' + text)
    pair = tuple(parsed['bracket'])
    return pair_class(pair) if pair in families else 'none'


def rewrite_width(text, atoms, families, target, rng):
    """``text`` with its outer bracket pair replaced by a declared pair of class ``target`` in the same family."""
    parsed = kg.parse_face(text, atoms)
    pair = tuple(parsed['bracket'])
    if pair not in families:
        return text
    left, right = rng.choice(families[pair][target])
    positions = [i for i, c in enumerate(text) if c in atoms.bracket_chars]
    chars = list(text)
    chars[positions[0]], chars[positions[1]] = left, right
    rewritten = ''.join(chars)
    if kg.parse_face(rewritten, atoms) is None:
        raise Unsupported('width-rewrite-unparsed')
    return rewritten


def face_record(text, atoms, source, structure):
    parsed = kg.parse_face(text, atoms)
    eye1, mouth, eye2 = parsed['core']
    key = gc.core_key(parsed['bracket'], eye1, mouth, eye2)
    return {'text': text, 'core_key': key, 'core_digest': gc.core_digest(key), 'core_partition': gc.partition(key),
            'structure': structure, 'source': source}


def common_pool(atoms, renderable):
    """Every width variant of the common listings, keyed by the split its own core belongs to."""
    families = width_families(atoms)
    pool = {'dev': [], 'heldout': []}
    for source, texts in (('common-structure', COMMON_FACES), ('paste-structure', PASTE_FACES)):
        for base in texts:
            if kg.parse_face(base, atoms) is None or not renderable(base):
                continue
            pair = tuple(kg.parse_face(base, atoms)['bracket'])
            variants = ({c: [p for p in families[pair][c]] for c in WIDTH_CLASS_WEIGHTS} if pair in families else {'none': [pair]})
            for name, pairs in variants.items():
                for left, right in pairs:
                    positions = [i for i, c in enumerate(base) if c in atoms.bracket_chars]
                    chars = list(base)
                    chars[positions[0]], chars[positions[1]] = left, right
                    text = ''.join(chars)
                    if kg.parse_face(text, atoms) is None:
                        continue
                    record = face_record(text, atoms, source, 'common')
                    if record['core_partition'] in pool:
                        pool[record['core_partition']].append({**record, 'bracket_width_class': name})
    return pool


class FacePicker:
    """Faces of one split: common listings (may recur) and distinct atom compositions, width sampled first."""
    def __init__(self, split, rng, atoms, allowed, pool, seen):
        self.split, self.rng, self.atoms, self.allowed, self.seen = split, rng, atoms, allowed, seen
        self.families = width_families(atoms)
        self.pool = {name: [f for f in pool[split] if f['bracket_width_class'] == name] for name in (*WIDTH_CLASS_WEIGHTS, 'none')}

    def target(self):
        names = list(WIDTH_CLASS_WEIGHTS)
        return self.rng.choices(names, [WIDTH_CLASS_WEIGHTS[n] for n in names])[0]

    def __call__(self):
        target = self.target()
        if self.rng.random() < COMMON_SHARE and self.pool[target]:
            return dict(self.rng.choice(self.pool[target]))
        for _ in range(20000):
            composed = gc.compose_face(self.split, self.rng, self.atoms, self.allowed)
            pair = tuple(kg.parse_face(composed['text'], self.atoms)['bracket'])
            if pair not in self.families and self.rng.random() >= NON_FAMILY_ACCEPT:
                continue
            text = rewrite_width(composed['text'], self.atoms, self.families, target, self.rng)
            record = face_record(text, self.atoms, 'atom-table-' + self.split + '-partition', composed['structure'])
            if record['core_partition'] != self.split or th(text) in self.seen:
                continue
            self.seen.add(th(text))
            return {**record, 'bracket_width_class': width_class(text, self.atoms, self.families)}
        raise ValueError('Original face pool exhausted for ' + self.split)


def plan_comment(rng, faces, atoms, seen, negative=False, words=None):
    """Ordered (kind, payload, separator) segments of one single-line comment."""
    kinds = [k for k in PLAN_WEIGHTS if not negative or k in NEGATIVE_PLANS]
    kind = rng.choices(kinds, [PLAN_WEIGHTS[k] for k in kinds])[0]
    def body(count):
        for _ in range(10000):
            text = cs.comment_words(rng, count)
            if th(text) not in seen and text.strip() and not cs.contains_face(text, atoms):
                seen.add(th(text))
                return text
        raise ValueError('Original comment pool exhausted')
    segments = []
    for segment_kind, separator in PLANS[kind]:
        separator = rng.choice(('', ' ')) if separator == '?' else separator
        if separator == 'fragment':
            segments.append(('body', rng.choice(cs.FRAGMENTS), ''))
        elif segment_kind == 'face':
            segments.append(('face', faces(), separator))
        else:
            segments.append(('body', body(rng.randrange(2, words or 9)), separator))
    return kind, segments


# ---------------------------------------------------------------- page

def scene_style(rng):
    scale = rng.choice(DEVICE_SCALES)
    viewport = (round(DEVICE_SIZE[0] / scale), round(DEVICE_SIZE[1] / scale))
    rows = (viewport[1] - LAYOUT['top'] + LAYOUT['gap']) // (LAYOUT['card_height'] + LAYOUT['gap'])
    webp = rng.random() < WEBP['probability']
    return {'device_scale_factor': scale, 'viewport': list(viewport), 'rows': int(rows), 'font_size': rng.choice(FONT_SIZES),
            'name_weight': rng.choice((600, 700)), 'delivery': {'format': 'webp', 'quality': rng.randrange(WEBP['quality'][0], WEBP['quality'][1] + 1),
            'method': WEBP['method']} if webp else {'format': 'png', 'quality': None, 'method': None}}


def plan_row(rng, faces, atoms, seen, negative):
    palette = rng.choice(sorted(PALETTES))
    jitter = lambda colour: tuple(max(0, min(255, c + rng.randrange(-4, 5))) for c in colour)
    kind, segments = plan_comment(rng, faces, atoms, seen, negative)
    duty = ''.join(rng.choice(cs.CN_WORDS) for _ in range(rng.randrange(2, 5))) + rng.choice(('', '', '！'))
    stamp = (f'{rng.randrange(2024, 2027)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d} '
             f'{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}')
    return {'palette': palette, 'colours': [jitter(c) for c in PALETTES[palette]], 'badge': rng.choice(('ring', 'double', 'diamond')),
            'icon': [rng.choice(ICON_COLOURS), rng.choice(('circle', 'triangle'))], 'level': f'Lv.{rng.randrange(10, 101)}',
            'duty': duty, 'stamp': stamp, 'kind': kind, 'segments': segments}


def page_regions(rows):
    """Reading-order text and region list; region index ``r`` is the element's ``data-r``."""
    whole, regions = '', []
    def add(kind, text, separator, extra=None):
        nonlocal whole
        if whole and not whole.endswith('\n'):
            whole += separator
        start = len(whole)
        whole += text
        regions.append({'kind': kind, 'text': text, 'span': [start, len(whole)], **(extra or {})})
    for index, row in enumerate(rows):
        if whole:
            whole += '\n'
        regions.append({'kind': 'graphic', 'geometry': 'badge-' + row['badge'], 'row': index})
        add('body', row['level'], COLUMN_GAP, {'row': index, 'column': 'level'})
        regions.append({'kind': 'graphic', 'geometry': 'icon-' + row['icon'][1], 'row': index})
        add('body', row['duty'], COLUMN_GAP, {'row': index, 'column': 'duty'})
        add('body', row['stamp'], COLUMN_GAP, {'row': index, 'column': 'stamp'})
        for position, (kind, payload, separator) in enumerate(row['segments']):
            text = payload['text'] if kind == 'face' else payload
            extra = {'row': index, 'column': 'comment', 'segment': position}
            if kind == 'face':
                extra.update({key: payload[key] for key in ('core_key', 'core_digest', 'core_partition', 'structure', 'source', 'bracket_width_class')})
            add('kaomoji' if kind == 'face' else 'body', text, COLUMN_GAP if position == 0 else separator, extra)
        add('body', '×', COLUMN_GAP, {'row': index, 'column': 'close'})
    return whole, regions


def _rgb(colour):
    return 'rgb(%d,%d,%d)' % tuple(colour)


def page_html(style, rows):
    """Self-contained page: no external resource, every region a leaf element with ``data-r``."""
    size, l = style['font_size'], LAYOUT
    css = (f'html,body{{margin:0;background:{_rgb(PAGE_COLOUR)};overflow:hidden}}'
           f'body{{font-family:{FONT_FAMILY};font-size:{size}px;line-height:1.3}}'
           f'.list{{padding:{l["top"]}px {l["side"]}px 0}}'
           f'.card{{position:relative;display:flex;align-items:center;height:{l["card_height"]}px;margin-bottom:{l["gap"]}px;'
           f'border-radius:{l["radius"]}px;color:#fff;box-shadow:0 1px 3px rgba(0,0,0,.28)}}'
           f'.badge{{flex:none;width:{l["badge"]}px;height:{l["badge"]}px;margin:0 36px 0 25px;box-sizing:border-box;border:5px solid {_rgb(GOLD)};border-radius:50%}}'
           f'.badge.double{{box-shadow:inset 0 0 0 7px transparent,inset 0 0 0 11px {_rgb(GOLD)}}}'
           f'.badge.diamond{{width:{l["badge"] - 18}px;height:{l["badge"] - 18}px;margin:0 45px 0 34px;border-radius:4px;transform:rotate(45deg)}}'
           f'.lv{{flex:none;width:{max(64, 4 * size)}px}}'
           f'.icon{{flex:none;position:relative;width:{l["icon"]}px;height:{l["icon"]}px;margin-right:27px;border-radius:5px}}'
           f'.icon i{{position:absolute;left:8px;top:8px;width:14px;height:14px;background:#eef0f4}}'
           f'.icon.circle i{{border-radius:50%}}.icon.triangle i{{clip-path:polygon(50% 0,100% 100%,0 100%)}}'
           f'.duty{{flex:none;white-space:pre;margin-right:27px}}.stamp{{flex:none;white-space:pre;margin-right:27px}}'
           f'.comment{{flex:none;white-space:pre}}.close{{position:absolute;right:{40 + l["side"]}px;top:50%;transform:translateY(-50%);font-size:{size + 2}px;color:#e6e2da}}')
    out, r = [], 0
    for row in rows:
        c0, c1 = row['colours']
        parts = [f'<div class="card" style="background:linear-gradient(90deg,{_rgb(c0)},{_rgb(c1)})">',
                 f'<div class="badge {row["badge"]}" data-r="{r}"></div>', f'<span class="lv" data-r="{r + 1}">{html.escape(row["level"])}</span>',
                 f'<div class="icon {row["icon"][1]}" style="background:{_rgb(row["icon"][0])}" data-r="{r + 2}"><i></i></div>',
                 f'<span class="duty" data-r="{r + 3}" style="font-weight:{style["name_weight"]}">{html.escape(row["duty"])}</span>',
                 f'<span class="stamp" data-r="{r + 4}">{html.escape(row["stamp"])}</span>']
        r += 5
        # The comment is one text node, as on a real page; segments are UTF-16 ranges for the Highlight API.
        comment, segments = '', []
        for position, (kind, payload, separator) in enumerate(row['segments']):
            comment += separator if position else ''
            start = _utf16(comment)
            comment += payload['text'] if kind == 'face' else payload
            segments.append(f'{r}:{start}:{_utf16(comment)}')
            r += 1
        parts.append(f'<div class="comment" data-segs="{";".join(segments)}">{html.escape(comment)}</div>'
                     f'<span class="close" data-r="{r}">×</span></div>')
        r += 1
        out.append(''.join(parts))
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{css}::highlight(only){{color:#fff}}</style></head>'
            f'<body><div class="list">{"".join(out)}</div><script>{PAGE_SCRIPT}</script></body></html>')


def _utf16(text):
    return len(text.encode('utf-16-le')) // 2


# Page controls: 'full' shows everything as rendered for delivery; 'none' hides every region (background);
# 'element' shows one element region; 'range' paints one comment segment through the Custom Highlight API
# on a transparent comment, so the comment's single text node keeps its exact shaping and positions.
PAGE_SCRIPT = (
    'window.__regions=Array.from(document.querySelectorAll("[data-r]"));'
    'window.__comments=Array.from(document.querySelectorAll(".comment"));window.__ranges=[];'
    'for(const c of __comments){const t=c.firstChild;if(!t)continue;for(const s of (c.dataset.segs||"").split(";")){if(!s)continue;'
    'const v=s.split(":").map(Number);const g=new Range();g.setStart(t,v[1]);g.setEnd(t,v[2]);__ranges.push([v[0],g]);}}'
    'window.__state=function(mode,id){for(const e of __regions){e.style.visibility=(mode==="full"||(mode==="element"&&+e.dataset.r===id))?"visible":"hidden";}'
    'for(const c of __comments){c.style.color=mode==="full"?"":"transparent";}CSS.highlights.clear();'
    'if(mode==="range"){const h=new Highlight();for(const p of __ranges){if(p[0]===id)h.add(p[1]);}CSS.highlights.set("only",h);}'
    'return new Promise(function(res){requestAnimationFrame(function(){requestAnimationFrame(res);});});};'
    'window.__segment=function(){for(const c of __comments){const t=c.textContent;let pos=0;const nodes=[];'
    'for(const s of (c.dataset.segs||"").split(";")){if(!s)continue;const v=s.split(":").map(Number);'
    'if(v[1]>pos)nodes.push(document.createTextNode(t.slice(pos,v[1])));const sp=document.createElement("span");sp.dataset.q=v[0];'
    'sp.textContent=t.slice(v[1],v[2]);nodes.push(sp);pos=v[2];}if(pos<t.length)nodes.push(document.createTextNode(t.slice(pos)));c.replaceChildren(...nodes);}};'
    'window.__overflow=function(){const out=[];document.querySelectorAll(".card").forEach(function(card,i){'
    'const c=card.querySelector(".comment").getBoundingClientRect(),x=card.querySelector(".close").getBoundingClientRect();'
    'if(c.right>x.left-16||card.getBoundingClientRect().bottom>window.innerHeight)out.push(i);});return out;};')


# ---------------------------------------------------------------- browser

class BrowserSession:
    """Local Edge driven by Playwright; every non-local request is aborted, counted and fails the run."""
    def __init__(self, temp_root):
        self.temp_root = Path(temp_root)
        self.blocked, self._saved = [], {}
        self._manager = self.browser = None

    def __enter__(self):
        from importlib.metadata import version
        from playwright.sync_api import sync_playwright
        if version('playwright') != DRIVER['version']:
            raise RuntimeError('Pinned Playwright version required: ' + DRIVER['version'])
        # The driver creates the browser's temporary profile in its own temp directory; keep it in the round.
        self.temp_root.mkdir(parents=True, exist_ok=True)
        for key in ('TEMP', 'TMP'):
            self._saved[key] = os.environ.get(key)
            os.environ[key] = str(self.temp_root.resolve())
        try:
            self._manager = sync_playwright()
            playwright = self._manager.__enter__()
            self.browser = playwright.chromium.launch(channel=DRIVER['channel'], headless=True, args=list(LAUNCH_ARGS))
            self.version = self.browser.version
        except BaseException:
            self.__exit__(None, None, None)    # a failed launch must not leave the driver, TEMP or the profile behind
            raise
        return self

    def context(self, scale, viewport):
        context = self.browser.new_context(viewport={'width': viewport[0], 'height': viewport[1]}, device_scale_factor=scale,
                                           locale='zh-CN', service_workers='block')
        context.on('request', lambda request: None if request.url.startswith(LOCAL_SCHEMES) else self.blocked.append(request.url))
        context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(LOCAL_SCHEMES) else route.abort())
        return context

    def check(self):
        if self.blocked:
            raise RuntimeError('Non-local browser request attempted: ' + self.blocked[0][:120])

    def __exit__(self, *exc):
        try:
            try:
                if self.browser is not None:
                    self.browser.close()
            finally:
                if self._manager is not None:
                    self._manager.__exit__(*exc)
        finally:
            self.browser = self._manager = None
            for key, value in self._saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self._saved = {}
            shutil.rmtree(self.temp_root, ignore_errors=True)
        return False


class ScenePage:
    """One loaded scene page: element visibility control, device-pixel captures and platform fonts."""
    def __init__(self, session, style, path):
        self.context = session.context(style['device_scale_factor'], style['viewport'])
        try:
            self.page = self.context.new_page()
            self.page.goto(Path(path).resolve().as_uri(), wait_until='load')
            self.page.evaluate('() => document.fonts.ready.then(() => true)')
            self.cdp = self.context.new_cdp_session(self.page)
            self.cdp.send('DOM.enable')
            self.cdp.send('CSS.enable')
        except BaseException:
            self.context.close()
            raise
        scale = style['device_scale_factor']
        self.expected = (round(style['viewport'][0] * scale), round(style['viewport'][1] * scale))

    def overflow(self):
        return self.page.evaluate('() => window.__overflow()')

    def state(self, mode, index=None):
        self.page.evaluate('([mode, id]) => window.__state(mode, id)', [mode, index])

    def segment(self):
        """Split comments into marker spans; only for platform-font queries after every capture is taken."""
        self.page.evaluate('() => window.__segment()')

    def capture(self):
        # Playwright's screenshot is taken at device pixels; a bare CDP capture returns CSS pixels at DSF > 1.
        data = self.page.screenshot(type='png', scale='device', animations='disabled', caret='hide')
        with Image.open(io.BytesIO(data)) as image:
            array = np.asarray(image.convert('RGB'))
        if (array.shape[1], array.shape[0]) != self.expected:
            raise RuntimeError('Capture size differs from the device viewport')
        return array

    def platform_fonts(self, index, comment=False):
        root = self.cdp.send('DOM.getDocument', {'depth': -1})['root']['nodeId']
        selector = f'[data-q="{index}"]' if comment else f'[data-r="{index}"]'
        node = self.cdp.send('DOM.querySelector', {'nodeId': root, 'selector': selector})['nodeId']
        fonts = self.cdp.send('CSS.getPlatformFontsForNode', {'nodeId': node})['fonts']
        return [{'family': f['familyName'], 'postscript': f.get('postScriptName', ''), 'glyphs': int(f['glyphCount'])} for f in fonts]

    def close(self):
        self.context.close()


# ---------------------------------------------------------------- system fonts

FONT_REGISTRY_KEY = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts'


def _registry_fonts():
    """(registry name, file) pairs for machine and per-user fonts; empty off Windows."""
    try:
        import winreg
    except ImportError:
        return []
    found = []
    windows = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(hive, FONT_REGISTRY_KEY)
        except OSError:
            continue
        with key:
            index = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                path = Path(value) if Path(value).is_absolute() else windows / value
                if path.is_file():
                    found.append((name, path))
    return found


class FontFiles:
    """Family name -> system font files (confirmed by their name tables) and the code points they cover."""
    def __init__(self):
        self.registry = _registry_fonts()
        self.cache = {}

    def files(self, family):
        if family not in self.cache:
            from fontTools.ttLib import TTCollection, TTFont
            matches, cmap = [], set()
            for name, path in self.registry:
                labels = [part.split(' (')[0].strip() for part in name.split(' & ')]
                if not any(label == family or label.startswith(family + ' ') for label in labels):
                    continue
                fonts = TTCollection(str(path), lazy=True).fonts if path.suffix.lower() == '.ttc' else [TTFont(str(path), lazy=True)]
                hit = False
                for font in fonts:
                    names = {record.toUnicode() for record in font['name'].names if record.nameID in (1, 16)}
                    if family in names:
                        hit = True
                        cmap.update(font.getBestCmap() or {})
                if hit:
                    matches.append({'file': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
            self.cache[family] = (matches, cmap)
        return self.cache[family]

    def covers(self, text, families):
        points = set()
        for family in families:
            points |= self.files(family)[1]
        return all(ord(c) in points for c in text if not c.isspace())


# ---------------------------------------------------------------- masks

def resolve_overlaps(masks, regions):
    """Exclusive masks: ink shared by a face and an adjacent body segment of the same comment goes to the face."""
    if not np.any(np.sum(masks, axis=0) > 1):
        return masks, {}
    exclusive, contact = [m.copy() for m in masks], Counter()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            shared = masks[i] & masks[j]
            if not shared.any():
                continue
            a, b = regions[i], regions[j]
            same_comment = (a.get('column') == b.get('column') == 'comment' and a.get('row') == b.get('row')
                            and abs(a['segment'] - b['segment']) == 1)
            if not same_comment or {a['kind'], b['kind']} != {'kaomoji', 'body'}:
                raise Unsupported('region-overlap')
            face, other = (i, j) if a['kind'] == 'kaomoji' else (j, i)
            exclusive[other] &= ~shared
            contact[face] += int(shared.sum())
    if np.any(np.sum(exclusive, axis=0) > 1):
        raise Unsupported('region-overlap')
    if any(not m.any() for m in exclusive):
        raise Unsupported('region-ink-absorbed')
    return exclusive, dict(contact)


def webp_roundtrip(array, quality, method):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, 'WEBP', quality=quality, method=method)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert('RGB'))


CLIP_RADIUS = 3  # device pixels: a highlight range is clipped at its box, so a glyph's antialiased overhang can fall outside


def assign_clipped(ink, masks, regions):
    """Give ink outside every mask (overhang clipped at a comment range edge) to a comment segment within reach.

    Reach is measured against the masks as captured, so assignments never chain past ``CLIP_RADIUS``.
    A face within reach wins, otherwise the nearest segment; ink near no comment segment is a mask failure.
    Returns the completed masks and, per face, the pixels taken while another segment was also within
    reach (contact with a neighbour); a face's own overhang with no neighbour in reach is not contact.
    """
    missing = ink & ~np.any(masks, axis=0)
    captured, masks, contact = masks, [m.copy() for m in masks], Counter()
    comment = [i for i, region in enumerate(regions) if region.get('column') == 'comment']
    for y, x in zip(*np.nonzero(missing)):
        top, left = max(0, y - CLIP_RADIUS), max(0, x - CLIP_RADIUS)
        near = []
        for i in comment:
            window = captured[i][top:y + CLIP_RADIUS + 1, left:x + CLIP_RADIUS + 1]
            if window.any():
                ys, xs = np.nonzero(window)
                near.append((regions[i]['kind'] != 'kaomoji', int(np.min((ys + top - y) ** 2 + (xs + left - x) ** 2)), i))
        if not near:
            raise Unsupported('mask-union')
        winner = min(near)[2]
        masks[winner][y, x] = True
        if regions[winner]['kind'] == 'kaomoji' and len(near) > 1:
            contact[winner] += 1
    return masks, contact


def capture_scene(session, style, path, regions, fonts):
    """Background, per-region masks and the delivered capture of one page; raises Unsupported on any doubt."""
    page = ScenePage(session, style, path)
    try:
        if page.overflow():
            raise Unsupported('comment-overflow')
        page.state('none')
        background = page.capture()
        page.state('none')
        if not np.array_equal(background, page.capture()):
            raise Unsupported('unstable-background')
        masks = []
        for index, region in enumerate(regions):
            page.state('range' if region.get('column') == 'comment' else 'element', index)
            mask = np.any(page.capture() != background, axis=2)
            if not mask.any():
                raise Unsupported('region-without-ink')
            masks.append(mask)
        page.state('full')
        full = page.capture()
        page.segment()
        used = [page.platform_fonts(i, region.get('column') == 'comment') if region['kind'] != 'graphic' else []
                for i, region in enumerate(regions)]
    finally:
        page.close()
    session.check()
    ink = np.any(full != background, axis=2)
    if np.any(np.any(masks, axis=0) & ~ink):
        raise Unsupported('mask-outside-ink')
    masks, overlap_contact = resolve_overlaps(masks, regions)
    masks, clipped_contact = assign_clipped(ink, masks, regions)
    if not np.array_equal(np.any(masks, axis=0), ink):
        raise Unsupported('mask-union')
    for region, fonts_used in zip(regions, used):
        if region['kind'] != 'graphic' and not fonts.covers(region['text'], [f['family'] for f in fonts_used]):
            raise Unsupported('uncovered-glyph')
    contact = Counter(overlap_contact)
    contact.update(clipped_contact)
    return background, full, masks, dict(contact), used


# ---------------------------------------------------------------- generation

def fitting_rows(session, style, plan, attempt_path, rows_override=None):
    """Plan rows and re-plan only the comments that overflow their column (comments never wrap)."""
    count = style['rows'] if rows_override is None else min(style['rows'], rows_override)
    rows = [plan(index) for index in range(count)]
    for _ in range(6):
        attempt_path.write_text(page_html(style, rows), encoding='utf-8')
        page = ScenePage(session, style, attempt_path)
        try:
            overflowing = page.overflow()
        finally:
            page.close()
        if not overflowing:
            return rows
        for index in overflowing:
            rows[index] = plan(index)
    raise Unsupported('comment-overflow')


def region_records(regions, masks, contact, used, style):
    records = []
    for index, (region, mask) in enumerate(zip(regions, masks)):
        if region['kind'] == 'graphic':
            record = {**cs.graphic_region(mask, region['geometry']), 'row': region['row']}
        else:
            record = {**region, 'bbox': bbox(mask), 'font_ids': sorted({f['family'] for f in used[index]}),
                      'rendering': {'text_present': True, 'glyphs': sum(f['glyphs'] for f in used[index]), 'platform_fonts': used[index]},
                      'size': style['font_size'] + (2 if region.get('column') == 'close' else 0)}
            if region['kind'] == 'kaomoji':
                record['contact_pixels'] = int(contact.get(index, 0))
        records.append(record)
    return records


def render_one(session, split, rng, picker, atoms, seen, negative, fonts, exclusions, rows_override=None):
    """One reliable scene: (style, rows, text, records, background, lossless, delivered, masks, page html)."""
    attempt = session.temp_root / 'attempt.html'
    for _ in range(60):
        style = scene_style(rng)
        try:
            rows = fitting_rows(session, style, lambda _: plan_row(rng, picker, atoms, seen, negative), attempt, rows_override)
            if not negative and not any(kind == 'face' for row in rows for kind, _, _ in row['segments']):
                raise Unsupported('positive-card-scene-without-face')
            text, regions = page_regions(rows)
            page = page_html(style, rows)
            attempt.write_text(page, encoding='utf-8')
            background, full, masks, contact, used = capture_scene(session, style, attempt, regions, fonts)
        except Unsupported as error:
            exclusions[str(error)] += 1
            continue
        delivery = style['delivery']
        delivered = webp_roundtrip(full, delivery['quality'], delivery['method']) if delivery['format'] == 'webp' else full
        return style, rows, text, region_records(regions, masks, contact, used, style), background, full, delivered, masks, page
    raise ValueError('Unable to render a reliable browser card scene')


def _image(array):
    return Image.fromarray(np.ascontiguousarray(array))


def write_scene(writer, root, identifier, split, scene, family_name):
    style, rows, text, regions, background, lossless, delivered, masks, page = scene
    paths = {'page': root / 'pages' / (identifier + '.html'), 'background': root / 'backgrounds' / (identifier + '.png'),
             'lossless': root / 'lossless' / (identifier + '.png')}
    writer.bytes(paths['page'], page.encode('utf-8'))
    writer.image(paths['background'], _image(background))
    writer.image(paths['lossless'], _image(lossless))
    image = paths['lossless']
    if style['delivery']['format'] == 'webp':
        image = root / 'images' / (identifier + '.png')
        writer.image(image, _image(delivered))
    for index, (region, mask) in enumerate(zip(regions, masks)):
        path = root / 'masks' / f'{identifier}-{index}.png'
        writer.image(path, Image.fromarray(mask.astype('uint8') * 255))
        region.update(id=f'{identifier}-region-{index}', mask=path.relative_to(root).as_posix(), mask_sha256=sha(path), pixel_count=int(mask.sum()))
    rel = lambda path: path.relative_to(root).as_posix()
    with Image.open(image) as opened:
        pixel = pixels(opened)
    faces = sum(region['kind'] == 'kaomoji' for region in regions)
    return {'id': identifier, 'split': split, 'source_split': split, 'scene_kind': SCENE_KINDS[1] if not faces else SCENE_KINDS[0],
            'recipe_family': family_name, 'text': text, 'text_sha256': th(text), 'image': rel(image), 'sha256': sha(image), 'pixel_sha256': pixel,
            'lossless_image': rel(paths['lossless']), 'lossless_sha256': sha(paths['lossless']), 'background': rel(paths['background']),
            'background_sha256': sha(paths['background']), 'page': rel(paths['page']), 'page_sha256': sha(paths['page']),
            'delivery': style['delivery'], 'device_scale_factor': style['device_scale_factor'],
            'style': {key: style[key] for key in ('viewport', 'rows', 'font_size', 'name_weight')} | {'palettes': [row['palette'] for row in rows]},
            'width': int(lossless.shape[1]), 'height': int(lossless.shape[0]), 'regions': regions,
            'provenance': {'source': 'original', 'license': 'CC0-1.0'}, 'baseline_omission': 'not measured; browser real-font rendering pressure only'}


def split_summary(rows):
    faces = [region for row in rows for region in row['regions'] if region['kind'] == 'kaomoji']
    return {'scenes': len(rows), 'faces': len(faces), 'negative_scenes': sum(row['scene_kind'] == SCENE_KINDS[1] for row in rows),
            'body_regions': sum(region['kind'] == 'body' for row in rows for region in row['regions']),
            'graphic_regions': sum(region['kind'] == 'graphic' for row in rows for region in row['regions']),
            'bracket_width_classes': dict(Counter(face['bracket_width_class'] for face in faces)),
            'face_sources': dict(Counter(face['source'] for face in faces)), 'contact_faces': sum(face['contact_pixels'] > 0 for face in faces),
            'deliveries': dict(Counter(row['delivery']['format'] for row in rows)),
            'device_scale_factors': {str(k): v for k, v in Counter(row['device_scale_factor'] for row in rows).items()},
            'platform_fonts': dict(Counter(font for row in rows for region in row['regions'] for font in region['font_ids']))}


def _dependency_digests():
    base = Path(__file__).resolve().parents[1]
    names = ['RegionKaomoji/card_scenes.py', 'RegionKaomoji/known_input.py', 'RegionKaomoji/scenes.py', 'PretrainedOcr/quality_corpus.py',
             'PretrainedOcr/grammar_corpus.py', 'PretrainedOcr/grammar_data.py', 'PretrainedOcr/kaomoji_grammar.py', 'PretrainedOcr/kaomoji_atoms.json']
    return {name: sha(base / name) for name in names}


def environment(session, fonts, rows):
    from importlib.metadata import version
    families = sorted({font for row in rows for region in row['regions'] for font in region['font_ids']})
    return {'browser': {'product': 'Microsoft Edge', 'version': session.version, 'channel': DRIVER['channel'], 'headless': True,
                        'launch_args': list(LAUNCH_ARGS), 'allowed_schemes': list(LOCAL_SCHEMES), 'blocked_requests': len(session.blocked)},
            'driver': {**DRIVER, 'installed': version('playwright')}, 'windows': platform.platform(), 'python': platform.python_version(),
            'pillow': Image.__version__, 'webp_codec': features.version('webp'),
            'platform_fonts': {family: fonts.files(family)[0] for family in families},
            'fonts_distributed': False, 'images_distributable': False}


def generate(output, fonts, counts=(60, 120), seed=20261015, rows_override=None):
    """Render, freeze and seal a browser real-font card dataset inside the registered round."""
    if len(counts) != 2 or any(not isinstance(n, int) or isinstance(n, bool) or n < NEGATIVE_EVERY for n in counts):
        raise ValueError('Each split needs at least one negative scene')
    root = qd.output_root(output)
    if rp.profile_for(root) is None or root.exists():
        raise ValueError('Fresh browser scenes must be generated inside a registered research round')
    qd.storage_gate(root, 64 * 1024 ** 2)
    renderer, atoms = MixedRenderer(fonts), kg.default_atoms()
    allowed, excluded_atoms = renderable_atoms(renderer, atoms)
    def renderable(text):
        try:
            renderer.shape(text)
        except Unsupported:
            return False
        return True
    pool = common_pool(atoms, renderable)
    root.mkdir(parents=True)
    for name in ('images', 'lossless', 'backgrounds', 'masks', 'pages'):
        (root / name).mkdir()
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'started_utc': utc()})
    writer, font_files, started = qd.Writer(root), FontFiles(), time.perf_counter()
    seen, allpixels, splits, jobs, summary, exclusions, every = set(), set(), {}, {}, {}, Counter(), []
    with BrowserSession(root / 'browser-temp') as session:
        for split, target in zip(('dev', 'heldout'), counts):
            rng = random.Random(seed + (0 if split == 'dev' else 1))
            picker, rows = FacePicker(split, rng, atoms, allowed, pool, seen), []
            for index in range(target):
                negative = index % NEGATIVE_EVERY == NEGATIVE_EVERY - 1
                while True:
                    scene = render_one(session, split, rng, picker, atoms, seen, negative, font_files, exclusions, rows_override)
                    digest = pixels(_image(scene[6]))
                    if digest not in allpixels:
                        break
                    exclusions['duplicate-pixels'] += 1
                allpixels.add(digest)
                rows.append(write_scene(writer, root, f'{split}-{index:04d}', split, scene, family(split)))
                atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'rendering', 'split': split, 'scenes': index + 1, 'target': target,
                                                      'elapsed_seconds': time.perf_counter() - started, 'updated_utc': utc()})
            writer.json(root / (split + '.json'), rows)
            splits[split] = {'file': split + '.json', 'sha256': sha(root / (split + '.json')), 'count': len(rows), 'requested': target}
            jobs[split] = [{k: row[k] for k in JOB_KEYS} for row in rows]
            summary[split], every = split_summary(rows), every + rows
        env = environment(session, font_files, every)
    if env['browser']['blocked_requests']:
        raise RuntimeError('Non-local browser requests were attempted')
    writer.json(root / 'dataset.json', {
        'schema_version': 'localocr.region-scenes.v1', 'generator_id': GENERATOR_ID, 'synthetic_only': True, 'browser_rendered': True,
        'private_inputs_consumed': False, 'license': 'CC0-1.0', 'license_note': 'Content is original CC0 text; images are rendered with locally licensed system fonts and stay private.',
        'seed': seed, 'splits': splits, 'recipes': {split: {kind: RECIPES[split] for kind in SCENE_KINDS} for split in RECIPES},
        'negative_scenes': {'every': NEGATIVE_EVERY, 'comment_kinds': list(NEGATIVE_PLANS), 'scene_kind': SCENE_KINDS[1]},
        'comment_plans': {k: list(map(list, v)) for k, v in PLANS.items()}, 'comment_weights': PLAN_WEIGHTS,
        'width_class_weights': WIDTH_CLASS_WEIGHTS, 'non_family_accept': NON_FAMILY_ACCEPT, 'common_share': COMMON_SHARE,
        'paste_faces': list(PASTE_FACES), 'device_scales': list(DEVICE_SCALES), 'device_size': list(DEVICE_SIZE), 'font_sizes': list(FONT_SIZES),
        'font_family': FONT_FAMILY, 'layout': LAYOUT, 'palettes': {k: [list(c) for c in v] for k, v in PALETTES.items()}, 'webp': WEBP,
        'atoms_binding': kg.binding(atoms), 'excluded_atoms': excluded_atoms, 'core_partition_buckets': {k: list(v) for k, v in gc.PARTITION_BUCKETS.items()},
        'environment': env, 'source_sha256': sha(__file__), 'dependency_sha256': _dependency_digests(), 'font_manifest_sha256': sha(fonts),
        'bounds': 'lossless-capture visible-ink half-open [left,top,right,bottom)',
        'spans': 'raw Python Unicode codepoint spans [start,end); rows separated by LF, columns by two spaces, comment segments by their declared separator; separators are not regions',
        'background': 'per-scene capture with every region hidden and comments transparent; visible ink is any lossless pixel differing from it',
        'masks': 'element regions: capture with only that element visible; comment segments: Custom Highlight of the segment range on the single comment text node; '
                 'ink clipped at a range edge goes to a comment segment within 3 device pixels, a face first (contact_pixels)',
        'delivery': 'png scenes deliver the lossless capture; webp scenes deliver the decoded lossy webp of it with the recorded quality and method',
        'layout_note': 'Original geometric badges and icons stand in for application icons; names are original word compounds; no game assets are reproduced',
        'exclusions': dict(exclusions), 'summary': summary,
        'usage': {'dev': 'measurement of frozen configurations only', 'heldout': 'first CPU evaluation after selection and dual freeze'}})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'validating', 'updated_utc': utc()})
    binding = validate(root)
    writer.json(root / 'heldout-seal.json', {'schema_version': 'localocr.region-scenes.seal.v1', 'dataset_sha256': binding['dataset_sha256'],
                'heldout_sha256': binding['split_sha256']['heldout'], 'inference_results_seen': False, 'requires_frozen_pipeline_selection': True})
    writer.json(root / 'inference-inputs.json', {'schema_version': 'localocr.region-scenes.inputs.v1', 'dataset_sha256': binding['dataset_sha256'],
                'split_sha256': binding['split_sha256'], 'jobs': jobs})
    writer.json(root / 'generation.json', {'binding': binding, 'elapsed_seconds': time.perf_counter() - started, 'summary': summary,
                'exclusions': dict(exclusions), 'excluded_atoms': excluded_atoms, 'CPU_model_inference': False, 'GPU_used': False,
                'private_inputs': False, 'fonts_distributed': False, 'blocked_requests': 0, 'storage': qd.storage_gate(root)})
    atomic_write(root / 'progress.json', {'pid': os.getpid(), 'phase': 'complete', 'completed_utc': utc(), 'dataset_sha256': binding['dataset_sha256']})
    writer.json(root / 'freeze.json', binding)
    return binding


# ---------------------------------------------------------------- validation

def _load(path, mode=None):
    with Image.open(path) as image:
        image.load()
        if mode is not None and image.mode != mode:
            raise ValueError('Image mode differs: ' + str(path))
        return np.asarray(image.convert('RGB') if mode is None else image)


def _confined(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Dataset path escape')
    return path


def _check_files(root, row):
    for key, digest_key in (('image', 'sha256'), ('lossless_image', 'lossless_sha256'), ('background', 'background_sha256'), ('page', 'page_sha256')):
        if sha(_confined(root, row[key])) != row[digest_key]:
            raise ValueError('Scene file hash differs: ' + key)
    delivered, lossless, background = (_load(_confined(root, row[k])) for k in ('image', 'lossless_image', 'background'))
    with Image.open(_confined(root, row['image'])) as image:
        if pixels(image) != row['pixel_sha256']:
            raise ValueError('Delivered pixels differ')
    if delivered.shape != lossless.shape or lossless.shape != background.shape or (row['width'], row['height']) != (lossless.shape[1], lossless.shape[0]):
        raise ValueError('Scene dimensions differ')
    delivery = row['delivery']
    if delivery['format'] == 'png':
        if row['image'] != row['lossless_image'] or delivery['quality'] is not None:
            raise ValueError('PNG delivery must be the lossless capture')
    elif delivery['format'] != 'webp' or not WEBP['quality'][0] <= delivery['quality'] <= WEBP['quality'][1] or delivery['method'] != WEBP['method']:
        raise ValueError('Undeclared delivery')
    elif not np.array_equal(webp_roundtrip(lossless, delivery['quality'], delivery['method']), delivered):
        raise ValueError('WebP delivery is not the declared re-encoding of the lossless capture')
    if row['device_scale_factor'] not in DEVICE_SCALES:
        raise ValueError('Undeclared device scale')
    return lossless, background


def _check_region(region, row, atoms, families, split):
    if region['kind'] == 'graphic':
        return
    a, b = region['span']
    if not 0 <= a < b <= len(row['text']) or row['text'][a:b] != region['text']:
        raise ValueError('Exact character span')
    if region['rendering']['glyphs'] <= 0 or not region['font_ids']:
        raise ValueError('Missing genuine glyphs')
    if region['kind'] == 'body':
        if cs.contains_face(region['text'], atoms) or 'contact_pixels' in region:
            raise ValueError('Body region carries a face or a face-only field')
        return
    if region['kind'] != 'kaomoji':
        raise ValueError('Unknown region kind')
    parsed = kg.parse_face(region['text'], atoms)
    # The core key is re-derived from the text: the outer bracket pair belongs to it, so a key kept from
    # before a width rewrite (or edited by hand) would let a face land in the wrong partition.
    if (parsed is None or region['core_key'] != gc.core_key(parsed['bracket'], *parsed['core'])
            or gc.core_digest(region['core_key']) != region['core_digest'] or gc.partition(region['core_key']) != split
            or region['core_partition'] != split):
        raise ValueError('Scene face core partition or grammar identity differs')
    if width_class(region['text'], atoms, families) != region['bracket_width_class']:
        raise ValueError('Bracket width class differs from the text')
    if type(region.get('contact_pixels')) is not int or region['contact_pixels'] < 0:
        raise ValueError('Face contact pixels missing')


def _check_masks(root, row, lossless, background):
    ink = np.any(lossless != background, axis=2)
    union = np.zeros(ink.shape, dtype=bool)
    for region in row['regions']:
        path = _confined(root, region['mask'])
        if sha(path) != region['mask_sha256']:
            raise ValueError('Mask hash/path')
        support = _load(path, 'L') > 0
        if support.shape != ink.shape or bbox(support) != region['bbox'] or int(support.sum()) != region['pixel_count']:
            raise ValueError('Actual mask/bbox pixels differ')
        if np.any(union & support):
            raise ValueError('Region glyph pixels overlap')
        if not np.all(ink[support]):
            raise ValueError('Mask outside visible ink')
        union |= support
    if not np.array_equal(union, ink):
        raise ValueError('Truth masks do not cover all actual visible ink')


def _check_row_identity(row, split, owners, pixelsets):
    faces = [region for region in row['regions'] if region['kind'] == 'kaomoji']
    if (row['split'] != split or row['source_split'] != split or row['recipe_family'] != family(split)
            or row['scene_kind'] != (SCENE_KINDS[0] if faces else SCENE_KINDS[1])):
        raise ValueError('Scene source/family differs')
    if row['provenance'] != {'source': 'original', 'license': 'CC0-1.0'} or th(row['text']) != row['text_sha256']:
        raise ValueError('Label/provenance changed')
    if row['pixel_sha256'] in pixelsets:
        raise ValueError('Duplicate scene pixels')
    pixelsets.add(row['pixel_sha256'])
    keys = [('text', row['text_sha256'])] + [('face', th(f['text'])) for f in faces] + [('core', f['core_digest']) for f in faces]
    keys += [('body', th(r['text'])) for r in row['regions'] if r['kind'] == 'body' and r.get('column') == 'comment' and r['text'] not in cs.FRAGMENTS]
    for key in keys:
        if owners.setdefault(key, split) != split:
            raise ValueError('Cross-split leakage: ' + key[0])


def validate(root):
    root = Path(root).resolve()
    meta = read(root / 'dataset.json')
    if meta.get('schema_version') != 'localocr.region-scenes.v1' or meta.get('synthetic_only') is not True or meta.get('generator_id') != GENERATOR_ID:
        raise ValueError('Scene schema/synthetic declaration')
    atoms = kg.default_atoms()
    if meta.get('atoms_binding') != kg.binding(atoms):
        raise ValueError('Scene atoms binding differs from the canonical table')
    if meta['environment']['browser']['blocked_requests'] != 0:
        raise ValueError('Generation attempted non-local requests')
    families, owners, counts, pixelsets = width_families(atoms), {}, {}, set()
    for split in ('dev', 'heldout'):
        info = meta['splits'][split]
        relative = Path(info['file'])
        if relative.is_absolute() or '..' in relative.parts or sha(root / relative) != info['sha256']:
            raise ValueError('Split path or hash changed')
        rows = read(root / relative)
        if len(rows) != info['count']:
            raise ValueError('Scene count changed')
        for row in rows:
            _check_row_identity(row, split, owners, pixelsets)
            lossless, background = _check_files(root, row)
            for region in row['regions']:
                _check_region(region, row, atoms, families, split)
            _check_masks(root, row, lossless, background)
        negatives = sum(row['scene_kind'] == SCENE_KINDS[1] for row in rows)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('action', choices=('generate', 'validate'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--fonts')
    parser.add_argument('--counts', nargs=2, type=int, default=(60, 120))
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args(argv)
    value = generate(args.dataset, args.fonts, tuple(args.counts), args.seed) if args.action == 'generate' else validate(args.dataset)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
