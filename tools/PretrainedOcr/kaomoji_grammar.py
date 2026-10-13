"""Single source of kaomoji structure: atom table loading, role parsing and projections.

The JSON atom table is the canonical inventory. Corpus generation, scene
generation, candidate routing and fusion derive every bracket, eye, mouth, arm
and tail set from the loaded table; no consumer may keep a literal copy.
Parsing is a recognizer for one complete expression and never rewrites text.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata

ATOMS_FILE = Path(__file__).with_name('kaomoji_atoms.json')
SCHEMA = 'localocr.kaomoji-atoms.v1'
LIST_ROLES = ('eyes', 'mouths', 'cheeks', 'whiskers', 'inner_arms', 'arms_left', 'arms_right', 'tails', 'gaps')
INNER_ROLES = ('eyes', 'mouths', 'cheeks', 'whiskers', 'inner_arms')
# One complete bare expression between the bracket pair; optional slots may be absent.
INNER_SLOTS = (('inner_arms', True), ('cheeks', True), ('eyes', False), ('whiskers', True), ('mouths', False),
               ('whiskers', True), ('eyes', False), ('cheeks', True), ('inner_arms', True))
_CACHE = {}


@dataclass(frozen=True)
class Atoms:
    sha256: str
    brackets: tuple
    eyes: tuple
    mouths: tuple
    cheeks: tuple
    whiskers: tuple
    inner_arms: tuple
    arms_left: tuple
    arms_right: tuple
    tails: tuple
    gaps: tuple
    limits: dict
    subsets: dict

    @property
    def bracket_chars(self):
        return frozenset(char for pair in self.brackets for char in pair)

    @property
    def letters(self):
        """Category-L codepoints that may legitimately appear in a face (never digits)."""
        chars = [char for role in LIST_ROLES if role != 'gaps' for atom in getattr(self, role) for char in atom]
        chars.extend(self.bracket_chars)
        return frozenset(char for char in chars if unicodedata.category(char).startswith('L'))

    @property
    def characters(self):
        chars = set(self.bracket_chars)
        for role in LIST_ROLES:
            for atom in getattr(self, role):
                chars.update(atom)
        return frozenset(chars)

    def role_set(self, role):
        return frozenset(getattr(self, role))

    def lexicon(self, roles):
        atoms = {atom for role in roles for atom in getattr(self, role)}
        return tuple(sorted(atoms, key=lambda atom: (-len(atom), atom)))

    def subset(self, name):
        return self.subsets[name]


def _validate_role(name, values, allow_space=False):
    if not isinstance(values, list) or not values:
        raise ValueError('Atom role ' + name + ' must be a nonempty list')
    seen = set()
    for atom in values:
        if not isinstance(atom, str) or (name != 'gaps' and not atom) or atom in seen or len(atom) > 8:
            raise ValueError('Atom role ' + name + ' needs unique nonempty short strings')
        if any(unicodedata.category(char) == 'Cc' for char in atom) or '\n' in atom or '\t' in atom:
            raise ValueError('Control characters are not kaomoji atoms')
        if name == 'gaps':
            if atom.strip(' ') != '':
                raise ValueError('Gaps may only contain ASCII spaces')
        elif ' ' in atom and not allow_space:
            raise ValueError('Only tails may contain an internal space')
        elif allow_space and (atom.startswith(' ') or atom.endswith(' ')):
            raise ValueError('Tails cannot begin or end with a space')
        seen.add(atom)
    return tuple(values)


def load_atoms(path=None):
    """Load and validate the atom table; the result is cached by its exact bytes."""
    path = Path(ATOMS_FILE if path is None else path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    key = (str(path.resolve()), digest)
    if key in _CACHE:
        return _CACHE[key]
    value = json.loads(raw.decode('utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA:
        raise ValueError('Unknown kaomoji atom table schema')
    brackets = value.get('brackets')
    if not isinstance(brackets, list) or not brackets:
        raise ValueError('Atom table needs bracket pairs')
    pairs = []
    for pair in brackets:
        if (not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(c, str) or len(c) != 1 for c in pair)
                or pair[0] == pair[1] or tuple(pair) in pairs):
            raise ValueError('Bracket pairs must be two distinct single codepoints without duplicates')
        pairs.append(tuple(pair))
    roles = {role: _validate_role(role, value.get(role), allow_space=role == 'tails') for role in LIST_ROLES}
    bracket_chars = {char for pair in pairs for char in pair}
    for role, atoms in roles.items():
        if role != 'gaps' and any(char in bracket_chars for atom in atoms for char in atom):
            raise ValueError('Bracket characters cannot double as ' + role)
        if role != 'gaps' and any(char.isdecimal() for atom in atoms for char in atom if atom != '3'):
            if role != 'mouths' and role not in ('arms_left', 'arms_right', 'inner_arms'):
                raise ValueError('Digits are only permitted as the numeric mouth 3 or declared arm glyphs')
    if '' not in roles['gaps']:
        raise ValueError('Gaps must include the empty gap')
    limits = value.get('limits')
    if (not isinstance(limits, dict) or set(limits) != {'maximum_length', 'minimum_length', 'maximum_outer_arms', 'maximum_tails'}
            or any(type(limits[k]) is not int or limits[k] <= 0 for k in limits) or limits['minimum_length'] > limits['maximum_length']):
        raise ValueError('Atom limits must be positive integers')
    subsets = value.get('subsets', {})
    if not isinstance(subsets, dict):
        raise ValueError('Subsets must be a mapping')
    frozen_subsets = {}
    for name, subset in subsets.items():
        if not isinstance(subset, dict):
            raise ValueError('Subset ' + name + ' must be a mapping')
        projected = {}
        for role, items in subset.items():
            if role == 'brackets':
                chosen = [tuple(p) for p in items]
                if any(p not in pairs for p in chosen):
                    raise ValueError('Subset bracket outside the atom table')
                projected[role] = tuple(chosen)
            elif role == 'arms':
                allowed = set(roles['arms_left']) & set(roles['arms_right'])
                if any(item not in allowed for item in items):
                    raise ValueError('Subset arms must be usable on both sides')
                projected[role] = tuple(items)
            elif role in roles:
                if any(item not in roles[role] for item in items):
                    raise ValueError('Subset ' + role + ' outside the atom table')
                projected[role] = tuple(items)
            else:
                raise ValueError('Unknown subset role ' + role)
        frozen_subsets[name] = projected
    atoms = Atoms(sha256=digest, brackets=tuple(pairs), limits=dict(limits), subsets=frozen_subsets, **roles)
    _CACHE[key] = atoms
    return atoms


def default_atoms():
    return load_atoms(ATOMS_FILE)


def _tokenize(segment, lexicon, allow_space):
    tokens = []
    index = 0
    while index < len(segment):
        if segment[index] == ' ':
            if not allow_space:
                return None
            end = index
            while end < len(segment) and segment[end] == ' ':
                end += 1
            tokens.append((' ', segment[index:end]))
            index = end
            continue
        for atom in lexicon:
            if segment.startswith(atom, index):
                tokens.append(('atom', atom))
                index += len(atom)
                break
        else:
            return None
    return tokens


def _match_inner(atoms, tokens, index, slot, roles):
    while index < len(tokens) and tokens[index][0] == ' ':
        if tokens[index][1] not in atoms.gaps:
            return None
        index += 1
    if slot == len(INNER_SLOTS):
        return roles if index == len(tokens) else None
    role, optional = INNER_SLOTS[slot]
    if index < len(tokens) and tokens[index][1] in atoms.role_set(role):
        result = _match_inner(atoms, tokens, index + 1, slot + 1, roles + [(role, tokens[index][1])])
        if result is not None:
            return result
    if optional:
        return _match_inner(atoms, tokens, index, slot + 1, roles)
    return None


def _outer(atoms, prefix, suffix):
    left = _tokenize(prefix, atoms.lexicon(('arms_left',)), allow_space=False) if prefix else []
    if left is None or len(left) > atoms.limits['maximum_outer_arms']:
        return None
    right_tokens = _tokenize(suffix, atoms.lexicon(('arms_right', 'tails')), allow_space=False) if suffix else []
    if right_tokens is None:
        return None
    arms_right, tails = [], []
    arms = atoms.role_set('arms_right')
    for _, atom in right_tokens:
        if not tails and atom in arms:
            arms_right.append(atom)
        elif atom in atoms.role_set('tails'):
            tails.append(atom)
        else:
            return None
    if len(arms_right) > atoms.limits['maximum_outer_arms'] or len(tails) > atoms.limits['maximum_tails']:
        return None
    return {'arms_left': [atom for _, atom in left], 'arms_right': arms_right, 'tails': tails}


def parse_face(text, atoms=None, allow_arms=True):
    """Return the role assignment of one complete expression, or None.

    Exactly one declared bracket pair must enclose eye, mouth and eye roles.
    Surrounding whitespace is ignored; interior whitespace runs must be
    declared gaps. The three core glyphs cannot all be one ASCII character.
    """
    atoms = default_atoms() if atoms is None else atoms
    limits = atoms.limits
    if (not isinstance(text, str) or not limits['minimum_length'] <= len(text) <= limits['maximum_length']
            or any(c in text for c in '\r\n\t')):
        return None
    bracket_chars = atoms.bracket_chars
    positions = [(index, char) for index, char in enumerate(text) if char in bracket_chars]
    if len(positions) != 2 or (positions[0][1], positions[1][1]) not in atoms.brackets:
        return None
    left, right = positions[0][0], positions[1][0]
    start, end = 0, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    prefix, suffix = text[start:left], text[right + 1:end]
    if not allow_arms and (prefix or suffix):
        return None
    outer = _outer(atoms, prefix, suffix)
    if outer is None:
        return None
    tokens = _tokenize(text[left + 1:right], atoms.lexicon(INNER_ROLES), allow_space=True)
    if tokens is None:
        return None
    roles = _match_inner(atoms, tokens, 0, 0, [])
    if roles is None:
        return None
    core = [atom for role, atom in roles if role in ('eyes', 'mouths')]
    if len(core) == 3 and len(set(core)) == 1 and core[0].isascii():
        return None
    return {'bracket': (positions[0][1], positions[1][1]), 'roles': tuple(roles), **outer,
            'core': tuple(core), 'atoms_sha256': atoms.sha256}


def is_face_text(text, atoms=None, allow_arms=True):
    return parse_face(text, atoms, allow_arms) is not None


def face_letters(atoms=None):
    return (default_atoms() if atoms is None else atoms).letters


def bracket_pairs(atoms=None):
    return (default_atoms() if atoms is None else atoms).brackets


def characters(atoms=None):
    return (default_atoms() if atoms is None else atoms).characters


def binding(atoms=None):
    atoms = default_atoms() if atoms is None else atoms
    return {'schema_version': SCHEMA, 'file': ATOMS_FILE.name, 'sha256': atoms.sha256,
            'counts': {role: len(getattr(atoms, role)) for role in ('brackets',) + LIST_ROLES}}
