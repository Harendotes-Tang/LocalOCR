"""Known-input set generation and manifests built from stored text (labels never read from pixels)."""
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.PretrainedOcr import kaomoji_grammar as kg
from tools.RegionKaomoji import known_input as ki, real_screens as rs

FONTS = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    if not FONTS.exists():
        pytest.skip('Pinned local OFL fonts absent')
    base = tmp_path_factory.mktemp('known-input')
    exclude = base / 'previous-list.json'
    exclude.write_text(json.dumps({'items': [{'text': '(´･ω･`)'}, {'text': '(≧▽≦)'}]}, ensure_ascii=False), encoding='utf-8')
    root = base / 'known-input-2'
    summary = ki.build(root, FONTS, seed=20261015, exclude=[str(exclude)])
    return root, summary, exclude


def test_generation_is_reproducible_distinct_and_excludes_previous_faces(built, tmp_path):
    root, summary, exclude = built
    listing = json.loads((root / 'input-list.json').read_text(encoding='utf-8'))
    texts = [item['text'] for item in listing['items']]
    assert summary['faces'] == len(texts) == 50 == len(set(texts)) and summary['screens'] == 14 and summary['embedded'] == 35
    assert not set(texts) & {'(´･ω･`)', '(≧▽≦)'} and all(kg.parse_face(t) is not None and len(t) <= ki.MAXIMUM_FACE_LENGTH for t in texts)
    sources = [item['source'] for item in listing['items']]
    assert sources.count('common-structure') == ki.COMMON_COUNT == 35 and sources.count('atom-table-heldout-partition') == 15
    assert all(item['core_partition'] == 'heldout' for item in listing['items'] if item['source'] != 'common-structure')
    assert {item['text'] for item in listing['items'] if item['source'] == 'common-structure'} <= set(ki.COMMON_FACES)
    script = json.loads((root / 'entry-script.json').read_text(encoding='utf-8'))
    standalone = [l for s in script['screens'] if s['section'] == 'standalone' for l in s['lines']]
    mixed = [l for s in script['screens'] if s['section'] == 'mixed' for l in s['lines']]
    assert [l['text'] for l in standalone if l['tag'] == 'face'] == texts and len(mixed) == 49
    assert sum(l['tag'] == 'mixed' for l in mixed) == 35 and all(l['face'] in l['text'] for l in mixed if l['tag'] == 'mixed')
    assert {l['variant'] for l in mixed if l['tag'] == 'mixed'} == set(ki.VARIANTS)
    assert all(kg.parse_face(l['text']) is None and not any(ch in l['text'] for ch in kg.default_atoms().bracket_chars) for l in standalone + mixed if l['tag'] == 'body')
    for screen in script['screens']:
        expected = (root / 'expected' / (Path(screen['file']).stem + '.txt')).read_text(encoding='utf-8').splitlines()
        assert expected == [l['text'] for l in screen['lines']]
    again = ki.build(tmp_path / 'again', FONTS, seed=20261015, exclude=[str(exclude)])
    assert json.loads((tmp_path / 'again' / 'input-list.json').read_text(encoding='utf-8'))['items'] == listing['items']
    with pytest.raises(ValueError):
        ki.build(root, FONTS, seed=1)


def test_embedded_faces_are_maximal_left_to_right_spans():
    assert ki.embedded_faces('今天 (´･ω･`) 顺利 (≧▽≦)') == ['(´･ω･`)', '(≧▽≦)']
    assert ki.embedded_faces('毕业了ヽ(´▽`)/撒花') == ['ヽ(´▽`)/']
    assert ki.embedded_faces('订单 (123) 完成') == [] and ki.embedded_faces('没有颜文字') == []
    assert ki.embedded_faces('（≧▽≦）') == ['（≧▽≦）']


def test_manifest_takes_stored_text_as_truth_and_counts_differences(built):
    root, _, _ = built
    script = json.loads((root / 'entry-script.json').read_text(encoding='utf-8'))
    changed_face = changed_body = None
    for screen in script['screens']:
        lines = []
        for entry in screen['lines']:
            text = entry['text']
            if entry['tag'] == 'face' and changed_face is None and '(' in text:
                text = text.replace('(', '（', 1)      # an input method switched the bracket width
                changed_face = text
            elif entry['tag'] == 'mixed' and changed_body is None and '(' in entry['face']:
                typed_face = entry['face'].replace('(', '（', 1)
                text = text.replace(entry['face'], typed_face, 1)
                changed_body = (text, typed_face)
            lines.append(text)
        (root / 'typed' / (Path(screen['file']).stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        Image.new('RGB', (64, 32), 'white').save(root / screen['file'])
    assert changed_face and changed_body
    out = root / 'manifest.json'
    summary = ki.build_manifest(root, 'unit UI', out, delivery='unit')
    assert summary['typed_differences'] == 2 and summary['mixed_lines_without_detected_face'] == 0
    loaded = rs.load_manifest(out)
    value = loaded['value']
    assert value['label_origin'].startswith('typed-via-ime') and value['typed_differences'] == 2 and value['usage'] == 'independent-evaluation-only'
    faces = [l for s in value['screens'] for l in s['lines'] if l['kind'] == 'face']
    bodies = [l for s in value['screens'] for l in s['lines'] if l['kind'] == 'body' and l.get('faces')]
    assert len(faces) == 50 and len(bodies) == 35
    assert any(l['text'] == changed_face and l['script_text'] != changed_face for l in faces)
    assert any(l['text'] == changed_body[0] and l['faces'] == [changed_body[1]] for l in bodies)
    (root / 'typed' / 'shot-001.txt').write_text('只有一行\n', encoding='utf-8')
    with pytest.raises(ValueError, match='exactly one nonempty line'):
        ki.build_manifest(root, 'unit UI', root / 'manifest-2.json')
