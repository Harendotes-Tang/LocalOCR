"""Atom-table corpus: hashed core partitions, shared structures and exact labels."""
from collections import Counter

import pytest

from tools.PretrainedOcr import grammar_corpus as gc, kaomoji_grammar as kg


def pools(count=300):
    atoms = kg.default_atoms()
    allowed = gc.allowed_atoms(atoms)
    seen = set()
    result = {}
    train_counts = None
    for index, split in enumerate(gc.SPLITS):
        rows = list(gc.samples(split, count, 500 + index, seen, atoms, allowed, train_counts))
        seen.update(r['text_sha256'] for r in rows if r['text'])
        if split == 'train':
            train_counts = gc.atom_counts([r for r in rows if r['kind'] == 'face'])
        result[split] = rows
    return result, train_counts


def test_partition_is_deterministic_and_covers_declared_buckets():
    key = gc.core_key(('(', ')'), 'ω', '_', 'ω')
    assert gc.partition(key) == gc.partition(dict(key)) and gc.partition(key) in gc.SPLITS
    assert gc.core_digest(key) != gc.core_digest(gc.core_key(('(', ')'), 'ω', '_', 'Д'))
    seen = Counter(gc.partition(gc.core_key(('(', ')'), eye, '_', eye)) for eye in kg.default_atoms().eyes)
    assert set(seen) <= set(gc.SPLITS) and seen['train'] >= seen['dev']


def test_cores_pixels_free_and_structures_shared_across_splits():
    rows, train_counts = pools()
    cores = {s: {r['core_digest'] for r in rows[s] if r['kind'] == 'face'} for s in rows}
    for a, b in (('train', 'dev'), ('train', 'heldout'), ('dev', 'heldout')):
        assert not cores[a] & cores[b]
        assert not {r['text_sha256'] for r in rows[a] if r['text']} & {r['text_sha256'] for r in rows[b] if r['text']}
    structures = {s: {r['recipe_expression'] for r in rows[s] if r['kind'] == 'face'} for s in rows}
    assert structures['train'] & structures['heldout'] and structures['train'] & structures['dev']
    for split, items in rows.items():
        for row in items:
            if row['kind'] == 'face':
                assert gc.partition(row['core_key']) == split == row['core_partition']
                assert row['family'] == gc.family(row['recipe_expression']) and gc.structure_signature(row['roles']) == row['recipe_expression']
                assert kg.parse_face(row['text']) is not None and row['group_id'] == 'text:' + row['text_sha256']
                assert not any(c.isdecimal() and c != '3' for c in row['text'] if c not in '٩۶')
            elif row['kind'] in ('blank', 'graphic'):
                assert row['text'] == ''
    assert Counter(r['kind'] for r in rows['train']) == {'face': 180, 'plain': 60, 'punctuation': 15, 'emoji': 15, 'blank': 15, 'graphic': 15}
    assert all(train_counts.get(atom, 0) > 0 for r in rows['heldout'] if r['kind'] == 'face' for atom in (r['core_key']['eye1'], r['core_key']['mouth'], r['core_key']['eye2']))
    assert any(row['roles']['arms_left'] and row['roles']['arms_right'] for row in rows['train'] if row['kind'] == 'face')
    assert any(row['roles']['tails'] for row in rows['heldout'] if row['kind'] == 'face')
    assert any('  ' in row['text'] for row in rows['train'] if row['kind'] == 'face')


def test_renderable_projection_restricts_atoms_without_editing_table():
    atoms = kg.default_atoms()
    allowed = gc.allowed_atoms(atoms, renderable={'(', ')', 'ω', '_', 'Д', 'つ', '!', '´', '・', 'ノ'})
    assert allowed['brackets'] == (('(', ')'),) and set(allowed['eyes']) == {'ω', 'Д'} and allowed['mouths'] == ('_', 'ω', 'Д') or set(allowed['mouths']) <= {'_', 'ω', 'Д'}
    assert '(' in atoms.bracket_chars and len(atoms.eyes) > 2
    with pytest.raises(ValueError):
        gc.allowed_atoms(atoms, renderable={'(', ')'})
    composed = gc.compose_face(gc.partition(gc.core_key(('(', ')'), 'ω', '_', 'ω')), __import__('random').Random(1), atoms, allowed)
    assert set(composed['atoms']) <= {'(', ')', 'ω', '_', 'Д', 'つ', '!', '´', '・', 'ノ'}


def test_unknown_split_and_exhaustion_fail_closed():
    with pytest.raises(ValueError):
        list(gc.samples('wrong', 1, 1))
    atoms = kg.default_atoms()
    allowed = gc.allowed_atoms(atoms, renderable={'(', ')', 'ω', '_'})
    with pytest.raises(ValueError):
        gc.compose_face('dev', __import__('random').Random(2), atoms, allowed, train_counts={'ω': 0, '_': 0, '(': 1, ')': 1}, attempts=50)
