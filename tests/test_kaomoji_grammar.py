"""Canonical atom table: validation, role parsing and consumer projections."""
import json
from pathlib import Path
import re
import unicodedata

import pytest

from tools.PretrainedOcr import kaomoji_grammar as kg

COMMON_FACES = [
    '(´･ω･`)', '(≧▽≦)', 'ヽ(´▽`)/', '(╯°□°)╯︵ ┻━┻', '(ノ´∀`)ノ', '٩(◕‿◕｡)۶', '( ´_ゝ`)', '(^_^)v',
    '(・ω・)ノ', '(*´ω`*)', '(ಥ_ಥ)', '(T▽T)', '(｡♥‿♥｡)', '(￣▽￣)', '(〃￣ω￣〃)', '(´；ω；`)', '(⌒▽⌒)',
    '(☆▽☆)', '(◕‿◕)', 'ʕ•ᴥ•ʔ', '(ง •̀_•́)ง', '(¬‿¬)', '(눈_눈)', '(；一_一)', '(＾▽＾)', '(〜￣▽￣)〜',
    '(*≧ω≦*)', '(ノಠ益ಠ)ノ彡┻━┻', '(｡◕‿◕｡)', '(✿◠‿◠)', '(❁´◡`❁)', '(´ω｀)', '(｡ŏ﹏ŏ)', '(°ロ°)', '(⊙_⊙)',
    '(☞ﾟヮﾟ)☞', '(´-ω-`)', '(*^▽^*)', '(〒︵〒)', '(´∀｀)', '(・∀・)', '(^・ω・^ )', '(^o^)', '(ToT)', '(x_x)',
    '「ω_ω」', '『Д益Д』', '(ω_ω）', '【°▽°】', '〈T_T〉!!',
]
NOT_FACES = [
    '(ABC)', '(o)', '订单 (123)', '(T)', '[o] (T)', '(123)', '(ω_ω)(ω_ω)', '(ω_ω) (ω_ω)', 'o  (Д_Д)', '(Д_Д) o',
    '(o Д_Д)', '(Д_Д', 'Д_Д)', 'Д_Д', '(oo_oo)', '3(ω3ω)', '(ω3ω)3', '(0_0)', 'orz', '¯\\_(ツ)_/¯',
    '( ͡° ͜ʖ ͡°)', '(---)', '(ooo)', '(^^^)', '(・・)', '(;;)', '(--)', '(3_3)', '[333]', '(xax)',
    'ノノノ(ω_ω)', '(ω_ω)!!!!!!!!', '(ω_ω) ノ', '(ω_ω)\t', 'x' * 65,
]


def test_default_table_loads_with_consistent_projections():
    atoms = kg.default_atoms()
    assert atoms.sha256 == kg.binding()['sha256'] and len(atoms.sha256) == 64
    assert ('(', ')') in atoms.brackets and ('(', '）') in atoms.brackets and ('「', '」') in atoms.brackets
    assert '3' in atoms.mouths and '3' not in atoms.letters and not any(c.isdecimal() for c in atoms.letters)
    assert {'ω', 'Д', 'つ', 'ノ', '益', '皿', 'ʕ', 'ᴥ'} <= atoms.letters
    assert atoms.bracket_chars.isdisjoint(set(''.join(atoms.eyes + atoms.mouths + atoms.tails)))
    assert kg.characters(atoms) >= set('ωД´｀･‿◕╯︵￣ヽ┻')
    legacy = atoms.subset('quality-20261012-v1')
    assert legacy['brackets'][0] == ('(', ')') and len(legacy['brackets']) == 6 and len(legacy['eyes']) == 12
    assert set(legacy['arms']) <= set(atoms.arms_left) & set(atoms.arms_right)


@pytest.mark.parametrize('text', COMMON_FACES)
def test_common_real_world_structures_are_recognized(text):
    parsed = kg.parse_face(text)
    assert parsed is not None, text
    assert len(parsed['core']) == 3 and parsed['bracket'] in kg.bracket_pairs()


@pytest.mark.parametrize('text', NOT_FACES)
def test_ordinary_text_fragments_and_out_of_scope_forms_are_rejected(text):
    assert not kg.is_face_text(text), text


def test_roles_are_assigned_without_rewriting_text():
    parsed = kg.parse_face('(╯°□°)╯︵ ┻━┻')
    assert parsed['roles'][0] == ('inner_arms', '╯') and parsed['core'] == ('°', '□', '°')
    assert parsed['arms_right'] == ['╯'] and parsed['tails'] == ['︵ ┻━┻'] and parsed['arms_left'] == []
    parsed = kg.parse_face('ノノ[ω  3  ω]つ!')
    assert parsed['arms_left'] == ['ノ', 'ノ'] and parsed['arms_right'] == ['つ'] and parsed['tails'] == ['!']
    parsed = kg.parse_face('(´･ω･`)')
    assert parsed['core'][1] == 'ω' and {role for role, _ in parsed['roles']} <= {'cheeks', 'eyes', 'mouths', 'whiskers'}
    assert ''.join(atom for _, atom in parsed['roles']) == '´･ω･`'
    assert kg.is_face_text('つ(Д_Д)つ') and not kg.is_face_text('つ(Д_Д)つ', allow_arms=False)
    assert kg.is_face_text(' (ω  _  ω) ') and not kg.is_face_text('(ω   _   ω)')


def test_tampered_or_malformed_tables_fail_closed(tmp_path):
    source = json.loads(Path(kg.ATOMS_FILE).read_text(encoding='utf-8'))
    def write(changes):
        value = json.loads(json.dumps(source))
        for key, item in changes.items():
            value[key] = item
        path = tmp_path / 'atoms.json'
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return path
    for changes in ({'schema_version': 'other'}, {'eyes': ['ω', 'ω']}, {'eyes': ['(']}, {'eyes': ['5']},
                    {'gaps': [' ']}, {'tails': [' !']}, {'mouths': ['a b']}, {'brackets': [['(', '(']]},
                    {'limits': {'maximum_length': 2, 'minimum_length': 3, 'maximum_outer_arms': 2, 'maximum_tails': 2}},
                    {'subsets': {'x': {'eyes': ['☃']}}}, {'subsets': {'x': {'arms': ['۶']}}}, {'subsets': {'x': {'noses': ['ω']}}}):
        with pytest.raises(ValueError):
            kg.load_atoms(write(changes))
    path = write({'eyes': source['eyes'] + ['☃']})
    atoms = kg.load_atoms(path)
    assert '☃' in atoms.eyes and kg.is_face_text('(☃_☃)', atoms) and not kg.is_face_text('(☃_☃)')
    assert kg.load_atoms(path) is atoms


def test_consumers_project_the_table_and_keep_no_literal_copies():
    from tools.RegionKaomoji import candidates as c
    from tools.PretrainedOcr import quality_corpus as corpus
    atoms = kg.default_atoms()
    assert c.FACE_LETTERS == atoms.letters and c.BRACKETS == atoms.brackets and c.ATOMS is atoms
    assert c.is_face_text('(≧▽≦)') and not c.is_face_text('(ABC)')
    assert corpus.BRACKETS == atoms.subset('quality-20261012-v1')['brackets']
    assert corpus.EYES == atoms.subset('quality-20261012-v1')['eyes'] and corpus.ARMS == atoms.subset('quality-20261012-v1')['arms']
    for name in ('tools/RegionKaomoji/candidates.py', 'tools/RegionKaomoji/fusion.py', 'tools/PretrainedOcr/quality_corpus.py'):
        text = (Path(__file__).resolve().parents[1] / name).read_text(encoding='utf-8')
        assert not re.search(r"frozenset\('[^']*[ωД益皿]", text), name
        assert not re.search(r"tuple\('[^']*[ωД益皿]", text), name
        assert "('(', ')')" not in text, name


def test_table_characters_are_single_scalars_or_combining_sequences():
    atoms = kg.default_atoms()
    for role in kg.LIST_ROLES:
        if role == 'gaps':
            continue
        for atom in getattr(atoms, role):
            marks = [c for c in atom if unicodedata.category(c).startswith('M')]
            assert len(atom) - len(marks) == 1 or role == 'tails', (role, atom)
