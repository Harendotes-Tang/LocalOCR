"""Original CC0 exact labels with shared glyph primitives and disjoint full ASTs.

Sample identifiers are metadata. Face labels contain only their visible expression;
no decimal identifier is appended. Shared atoms and fonts are explicitly disclosed.
"""
from __future__ import annotations
import hashlib
import json
import random
from .corpus import CN_WORDS, EN_WORDS
from . import kaomoji_grammar as kg

SPLITS = ('train', 'dev', 'heldout')
# The v1 generator keeps its frozen 2026-10-12 pools, projected from the canonical atom table.
ATOM_SUBSET = 'quality-20261012-v1'
_POOLS = kg.default_atoms().subset(ATOM_SUBSET)
BRACKETS = _POOLS['brackets']
EYES = _POOLS['eyes']
MOUTHS = _POOLS['mouths']
ARMS = _POOLS['arms']
EMOJI = ('😀','😁','😂','😃','😄','😅','😆','😇','😉','😊','😍','😎','🐱','🌈','👩‍💻','👨‍🔬')
CORE = '{left}{eye1}{gap1}{mouth}{gap2}{eye2}{right}'
FACE_FORMS = {
    'train': (CORE, '{arm1}'+CORE, CORE+'{arm1}', '{arm1}'+CORE+'{arm2}', CORE+'!', CORE+'──', '{arm1}'+CORE+'!', '{arm1}'+CORE+'──'),
    'dev': ('{arm1}{arm2}'+CORE, CORE+'{arm1}{arm2}', CORE+'!!', '{arm1}'+CORE+'{arm2}{arm3}'),
    'heldout': ('{arm1}{arm2}'+CORE+'{arm3}', '{arm1}{arm2}'+CORE+'!', CORE+'{arm1}!', '{arm1}'+CORE+'{arm2}!'),
}
PLAIN_FORMS = {
    'train': ('{cn1} {cn2}', '{en1} {en2}', '{value}.{decimal}', ' café {en1} ō '),
    'dev': ('{cn1}：{cn2}', '{en1}:{en2}', '{value}/{decimal}', 'ō [{en1}] café'),
    'heldout': ('{cn1}/{cn2}', '{en2}={en1}', '{decimal}-{value}', 'café/{en1}/ō'),
}
PUNCTUATION_FORMS = {
    'train': ('({value}) {symbols}', '[{value}] {symbols}', '{{{value}}} {symbols}'),
    'dev': ('{symbols}({value})!', '{symbols}[{value}]!', '{symbols}{{{value}}}!'),
    'heldout': ('({value}):{symbols}', '[{value}]:{symbols}', '{{{value}}}:{symbols}'),
}
EMOJI_FORMS = {'train':'{emoji1}{emoji2}{emoji3}', 'dev':'{emoji1} {emoji2}', 'heldout':'{emoji1}  {emoji2}'}
NEGATIVE_FORMS = {
    'train': {'blank':'solid-rgb', 'graphic':'single(ellipse|triangle|box|sine)-solid'},
    'dev': {'blank':'vertical-linear-rgb', 'graphic':'paired(ellipse|triangle|box|sine)-gradient'},
    'heldout': {'blank':'modular-rgb-texture', 'graphic':'overlaid(ellipse|triangle|box|sine)-texture'},
}
GENERATOR_ID = 'localocr.quality-corpus.20261012.v1'


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def family(expression):
    return 'quality-grammar:'+text_hash(expression)


def kind_at(index):
    bucket = index % 20
    return 'face' if bucket < 12 else 'plain' if bucket < 16 else 'punctuation' if bucket == 16 else 'emoji' if bucket == 17 else 'blank' if bucket == 18 else 'graphic'


def make_sample(split, index, rng, seen):
    if split not in SPLITS:
        raise ValueError('Unknown split')
    kind = kind_at(index)
    if kind in ('blank','graphic'):
        expression=NEGATIVE_FORMS[split][kind]
        return {'text':'', 'text_sha256':text_hash(''), 'kind':kind, 'subtype':kind, 'recipe':kind,
                'recipe_expression':expression, 'family':family(expression), 'group_id':f'{split}-negative-{index:06d}'}
    for attempt in range(100000):
        left,right=rng.choice(BRACKETS)
        gap=lambda: rng.choices(('', ' ', '  '),(8,1,1))[0]
        args={'left':left,'right':right,'eye1':rng.choice(EYES),'eye2':rng.choice(EYES),
              'mouth':rng.choice(MOUTHS),'gap1':gap(),'gap2':gap(),
              'arm1':rng.choice(ARMS),'arm2':rng.choice(ARMS),'arm3':rng.choice(ARMS),
              'cn1':rng.choice(CN_WORDS),'cn2':rng.choice(CN_WORDS),
              'en1':rng.choice(EN_WORDS),'en2':rng.choice(EN_WORDS),
              'value':rng.randrange(10,100000),'decimal':f'{rng.randrange(100):02d}',
              'symbols':rng.choice(('!?;:', '<->/\\', '___===', '───+-')),
              'emoji1':rng.choice(EMOJI),'emoji2':rng.choice(EMOJI),'emoji3':rng.choice(EMOJI)}
        expression=rng.choice(FACE_FORMS[split] if kind=='face' else PLAIN_FORMS[split] if kind=='plain' else PUNCTUATION_FORMS[split] if kind=='punctuation' else (EMOJI_FORMS[split],))
        text=expression.format(**args)
        digest=text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        return {'text':text,'text_sha256':digest,'kind':kind,'subtype':'combining' if '́' in text else 'emoji-sequence' if kind=='emoji' else kind,
                'recipe':kind,'recipe_expression':expression,'family':family(expression),'group_id':'text:'+digest}
    raise ValueError('Original expression pool exhausted')


def samples(split,count,seed,excluded=()):
    if split not in SPLITS or not isinstance(count,int) or isinstance(count,bool) or count<0:
        raise ValueError('Unknown split/count')
    rng=random.Random(seed);seen=set(excluded)
    for index in range(count):
        yield make_sample(split,index,rng,seen)


def expressions(split):
    return set(FACE_FORMS[split]+PLAIN_FORMS[split]+PUNCTUATION_FORMS[split]+(EMOJI_FORMS[split],)+tuple(NEGATIVE_FORMS[split].values()))
