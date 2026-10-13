"""Atom-table faces with hashed core partitions and shared structural forms.

Every face is composed from the canonical atom table. The (bracket pair, eye,
mouth, eye) core is assigned to one split by a deterministic hash, so cores
never leak across train, development and heldout while every structural form
(cheeks, whiskers, arms, tails, gaps) remains available to training. Labels
are the visible expression only; identifiers stay in metadata.
"""
from __future__ import annotations
import hashlib
import json
import random

from . import kaomoji_grammar as kg
from .quality_corpus import (SPLITS, PLAIN_FORMS, PUNCTUATION_FORMS, EMOJI_FORMS, NEGATIVE_FORMS, EMOJI,
                             text_hash, kind_at)
from .corpus import CN_WORDS, EN_WORDS

GENERATOR_ID = 'localocr.grammar-corpus.20261013.v1'
PARTITION_BUCKETS = {'train': tuple(range(7)), 'dev': (7,), 'heldout': (8, 9)}
STRUCTURE_PREFIX = 'grammar-structure:'
ROLE_FIELDS = ('arms_left', 'inner_arm_left', 'cheek_left', 'whisker_left', 'gap_left', 'gap_right',
               'whisker_right', 'cheek_right', 'inner_arm_right', 'arms_right', 'tails')


def core_key(bracket, eye1, mouth, eye2):
    return {'bracket': list(bracket), 'eye1': eye1, 'mouth': mouth, 'eye2': eye2}


def core_digest(key):
    return hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def partition(key):
    """Deterministic split ownership of a core combination."""
    bucket = int(core_digest(key)[:8], 16) % 10
    return next(split for split, buckets in PARTITION_BUCKETS.items() if bucket in buckets)


def structure_signature(roles):
    """Structural form independent of the atoms chosen; shared across splits by design."""
    parts = [f'L{len(roles["arms_left"])}', f'I{int(bool(roles["inner_arm_left"]))}', f'C{int(bool(roles["cheek_left"]))}',
             f'W{int(bool(roles["whisker_left"]))}', f'G{len(roles["gap_left"])}', 'EME', f'G{len(roles["gap_right"])}',
             f'W{int(bool(roles["whisker_right"]))}', f'C{int(bool(roles["cheek_right"]))}', f'I{int(bool(roles["inner_arm_right"]))}',
             f'R{len(roles["arms_right"])}', f'T{len(roles["tails"])}']
    return '|'.join(parts)


def family(signature):
    return STRUCTURE_PREFIX + text_hash(signature)


def allowed_atoms(atoms, renderable=None):
    """Project the table onto renderable atoms; the table itself is never edited."""
    allowed = {}
    for role in ('eyes', 'mouths', 'cheeks', 'whiskers', 'inner_arms', 'arms_left', 'arms_right', 'tails'):
        items = [atom for atom in getattr(atoms, role) if renderable is None or atom in renderable]
        if not items and role in ('eyes', 'mouths'):
            raise ValueError('No renderable atoms for role ' + role)
        allowed[role] = tuple(items)
    allowed['brackets'] = tuple(pair for pair in atoms.brackets
                                if renderable is None or (pair[0] in renderable and pair[1] in renderable))
    if not allowed['brackets']:
        raise ValueError('No renderable bracket pairs')
    allowed['gaps'] = tuple(atoms.gaps)
    return allowed


def _choose_count(rng, weights):
    return rng.choices(range(len(weights)), weights)[0]


def compose_face(split, rng, atoms, allowed, train_counts=None, attempts=20000):
    """Compose one face whose core belongs to ``split``; returns text and metadata."""
    for _ in range(attempts):
        bracket = rng.choice(allowed['brackets'])
        eye1, eye2 = rng.choice(allowed['eyes']), rng.choice(allowed['eyes'])
        mouth = rng.choice(allowed['mouths'])
        key = core_key(bracket, eye1, mouth, eye2)
        if partition(key) != split:
            continue
        if train_counts is not None and split != 'train' and any(train_counts.get(atom, 0) == 0 for atom in (eye1, mouth, eye2, *bracket)):
            continue
        def pick(role, probability):
            pool = allowed[role]
            return rng.choice(pool) if pool and rng.random() < probability else ''
        def pick_many(role, weights):
            pool = allowed[role]
            return [rng.choice(pool) for _ in range(_choose_count(rng, weights))] if pool else []
        roles = {
            'arms_left': pick_many('arms_left', (60, 32, 8)),
            'inner_arm_left': pick('inner_arms', .08),
            'cheek_left': pick('cheeks', .30),
            'whisker_left': pick('whiskers', .12),
            'gap_left': rng.choices(allowed['gaps'], (80, 12, 8))[0],
            'gap_right': rng.choices(allowed['gaps'], (80, 12, 8))[0],
        }
        roles['whisker_right'] = roles['whisker_left'] if roles['whisker_left'] and rng.random() < .8 else pick('whiskers', .05)
        if roles['cheek_left']:
            roles['cheek_right'] = roles['cheek_left'] if rng.random() < .5 else pick('cheeks', .6)
        else:
            roles['cheek_right'] = pick('cheeks', .3)
        roles['inner_arm_right'] = pick('inner_arms', .05)
        roles['arms_right'] = pick_many('arms_right', (55, 37, 8))
        roles['tails'] = pick_many('tails', (60, 33, 7))
        text = (''.join(roles['arms_left']) + bracket[0] + roles['inner_arm_left'] + roles['cheek_left'] + eye1
                + roles['whisker_left'] + roles['gap_left'] + mouth + roles['gap_right'] + roles['whisker_right'] + eye2
                + roles['cheek_right'] + roles['inner_arm_right'] + bracket[1] + ''.join(roles['arms_right']) + ''.join(roles['tails']))
        if kg.parse_face(text, atoms) is None:
            continue
        signature = structure_signature(roles)
        return {'text': text, 'core_key': key, 'core_digest': core_digest(key), 'core_partition': split,
                'structure': signature, 'roles': {name: roles[name] for name in ROLE_FIELDS},
                'atoms': sorted({eye1, mouth, eye2, *bracket, *roles['arms_left'], *roles['arms_right'], *roles['tails'],
                                 *(v for v in (roles['inner_arm_left'], roles['cheek_left'], roles['whisker_left'],
                                               roles['whisker_right'], roles['cheek_right'], roles['inner_arm_right']) if v)})}
    raise ValueError('Unable to compose a face for split ' + split)


def make_sample(split, index, rng, seen, atoms, allowed, train_counts=None):
    if split not in SPLITS:
        raise ValueError('Unknown split')
    kind = kind_at(index)
    if kind in ('blank', 'graphic'):
        expression = NEGATIVE_FORMS[split][kind]
        return {'text': '', 'text_sha256': text_hash(''), 'kind': kind, 'subtype': kind, 'recipe': kind,
                'recipe_expression': expression, 'family': 'negative-grammar:' + text_hash(expression),
                'group_id': f'{split}-negative-{index:06d}'}
    for _ in range(100000):
        if kind == 'face':
            face = compose_face(split, rng, atoms, allowed, train_counts)
            text, expression, fam = face['text'], face['structure'], family(face['structure'])
            extra = {'core_key': face['core_key'], 'core_digest': face['core_digest'], 'core_partition': face['core_partition'],
                     'roles': face['roles'], 'atoms': face['atoms']}
        else:
            args = {'cn1': rng.choice(CN_WORDS), 'cn2': rng.choice(CN_WORDS), 'en1': rng.choice(EN_WORDS), 'en2': rng.choice(EN_WORDS),
                    'value': rng.randrange(10, 100000), 'decimal': f'{rng.randrange(100):02d}',
                    'symbols': rng.choice(('!?;:', '<->/\\', '___===', '───+-')),
                    'emoji1': rng.choice(EMOJI), 'emoji2': rng.choice(EMOJI), 'emoji3': rng.choice(EMOJI)}
            expression = rng.choice(PLAIN_FORMS[split] if kind == 'plain' else PUNCTUATION_FORMS[split] if kind == 'punctuation' else (EMOJI_FORMS[split],))
            text = expression.format(**args)
            fam = 'quality-grammar:' + text_hash(expression)
            extra = {}
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        subtype = 'combining' if '́' in text else 'emoji-sequence' if kind == 'emoji' else kind
        return {'text': text, 'text_sha256': digest, 'kind': kind, 'subtype': subtype, 'recipe': kind,
                'recipe_expression': expression, 'family': fam, 'group_id': 'text:' + digest, **extra}
    raise ValueError('Original expression pool exhausted')


def samples(split, count, seed, excluded=(), atoms=None, allowed=None, train_counts=None):
    if split not in SPLITS or not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError('Unknown split/count')
    atoms = kg.default_atoms() if atoms is None else atoms
    allowed = allowed_atoms(atoms) if allowed is None else allowed
    rng = random.Random(seed)
    seen = set(excluded)
    for index in range(count):
        yield make_sample(split, index, rng, seen, atoms, allowed, train_counts)


def atom_counts(rows):
    """Per-atom training image counts from face rows; label text is not re-parsed."""
    counts = {}
    for row in rows:
        for atom in row.get('atoms', ()):
            counts[atom] = counts.get(atom, 0) + 1
    return counts
