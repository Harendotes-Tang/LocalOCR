"""Browser real-font training crops: targets, tail separation, isolation, offline rendering and dataset readers."""
import builtins
import dataclasses
import io
import json
import random
import shutil
from pathlib import Path

import pytest

from tools.PretrainedOcr import common, grammar_corpus as gc, kaomoji_grammar as kg, round_profile as rp
from tools.RegionKaomoji import browser_crops as bc, browser_scenes as bs

REPO = Path(__file__).resolve().parents[1]
FONTS = REPO / 'artifacts/ocr-data-fonts-v1/fonts.json'
ATOMS = kg.default_atoms()


def _edge_available():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe').exists()


def _allowed():
    return gc.allowed_atoms(ATOMS, {a for role in kg.LIST_ROLES if role != 'gaps' for a in getattr(ATOMS, role)} | set(ATOMS.bracket_chars))


def test_kind_counts_follow_the_declared_shares():
    assert bc.kind_counts(16000) == {'face': 9600, 'plain': 3200, 'punctuation': 800, 'emoji': 800, 'blank': 800, 'graphic': 800}
    assert sum(bc.kind_counts(11200).values()) == 11200 and bc.kind_counts(4800)['face'] == 2880


def test_train_faces_come_from_the_train_partition_with_stratified_widths():
    pool = bs.common_pool(ATOMS, lambda text: True)
    assert pool['train'] and all(f['core_partition'] == 'train' for f in pool['train'])
    picker = bc.StratifiedFaces('train', random.Random(3), ATOMS, _allowed(), pool, set())
    faces = [picker() for _ in range(400)]
    assert all(f['core_partition'] == 'train' == gc.partition(f['core_key']) for f in faces)
    classes = {name: sum(f['bracket_width_class'] == name for f in faces) for name in ('ascii', 'fullwidth', 'mixed', 'none')}
    assert 40 < classes['none'] < 130 and classes['ascii'] > 100 and classes['fullwidth'] > 100 and classes['mixed'] > 10


def test_a_tail_shaped_right_neighbour_is_never_glued_to_a_face(monkeypatch):
    assert bc.tail_start('(◡‿◡)', '~海洋', ATOMS) and not bc.tail_start('(◡‿◡)', '海洋', ATOMS)
    face = bs.face_record('(◡‿◡)', ATOMS, 'unit', 'common')
    monkeypatch.setattr(bc, 'words', lambda rng, atoms, count: '~海洋')
    rng = random.Random(0)
    for _ in range(40):
        spec = bc.plan_target('face', 'dev', rng, lambda: face, bc.Labels({}), ATOMS)
        assert spec['sep_right'] == ' '


def test_labels_are_unique_and_never_parse_as_faces():
    owners, rng = {}, random.Random(1)
    labels = bc.Labels(owners)
    seen = {labels.take('dev', lambda: ''.join(rng.choice(('（', '）', '^', '_', '。', '！')) for _ in range(4)), ATOMS) for _ in range(200)}
    assert len(seen) == 200 and not any(kg.parse_face(t, ATOMS) for t in seen)
    taken = next(iter(seen))
    with pytest.raises(ValueError):
        bc.Labels(owners).take('heldout', lambda: taken, ATOMS)


def test_same_seed_plans_the_same_page():
    pool = bs.common_pool(ATOMS, lambda text: True)
    def plan(seed):
        rng = random.Random(seed)
        specs = bc.split_specs('dev', 30, rng, bc.StratifiedFaces('dev', rng, ATOMS, _allowed(), pool, set()), bc.Labels({}), ATOMS)
        return bc.crop_page_html(bc.page_style(random.Random(seed)), specs[:bc.ROWS_PER_PAGE])
    assert plan(5) == plan(5) and plan(5) != plan(6)
    assert 'http' not in plan(5)


@pytest.mark.skipif(not _edge_available(), reason='Playwright or the local Edge is absent')
def test_a_non_local_request_fails_the_page(tmp_path, monkeypatch):
    original = bc.crop_page_html
    monkeypatch.setattr(bc, 'crop_page_html', lambda style, specs: original(style, specs).replace('</body>', '<img src="https://example.invalid/x.png"></body>'))
    style = dict(bc.page_style(random.Random(2)), device_scale_factor=1.0)
    specs = [{'kind': 'plain', 'left': '', 'sep_left': '', 'target': '海洋', 'sep_right': '', 'right': ''}]
    with bs.BrowserSession(tmp_path / 'browser-temp') as session:
        with pytest.raises(RuntimeError, match='Non-local'):
            bc.render_page(session, style, specs, tmp_path / 'page.html', bs.FontFiles(), random.Random(0))


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    if not (_edge_available() and FONTS.exists() and bc.SOURCE.exists()):
        pytest.skip('Playwright, Edge, the OFL fonts or the grammar-round crops are absent')
    base = tmp_path_factory.mktemp('crop-round')
    if not base.resolve().is_relative_to((REPO / 'artifacts').resolve()):
        pytest.skip('Generation must run with --basetemp under artifacts/')
    profile = dataclasses.replace(rp.BROWSER_FONT, name='unit-crops', root=base / 'unit-round', previous_budget=base / 'old.json')
    # pathlib reads go through io.open, plain open() through builtins.open: watch both.
    opened, real_open, real_io_open = [], builtins.open, io.open
    def watching(real):
        def watching_open(file, *args, **kwargs):
            opened.append(str(file))
            return real(file, *args, **kwargs)
        return watching_open
    saved = rp.ROUNDS
    rp.ROUNDS, builtins.open, io.open = (profile,), watching(real_open), watching(real_io_open)
    try:
        root = profile.root / 'data' / 'crops'
        binding = bc.generate(root, FONTS, (100, 100, 100), seed=11)
    finally:
        rp.ROUNDS, builtins.open, io.open = saved, real_open, real_io_open
    return root, binding, opened


def test_generation_never_opens_the_sealed_browser_cards_heldout(tiny):
    _, _, opened = tiny
    assert opened and not any('browser-cards-v1' in path.replace('\\', '/') for path in opened)


def test_tiny_dataset_is_readable_by_the_shared_crop_readers(tiny):
    root, binding, _ = tiny
    from tools.RegionKaomoji import train_classifier as tc
    assert bc.validate(root) == binding
    assert len(common.read_split(root, 'train')) == 100 and len(common.read_split(root, 'dev')) == 100
    with pytest.raises(ValueError):
        common.read_split(root, 'heldout')
    arrays, labels = tc.load_rows(root, 'dev')
    assert len(arrays) == 100 and set(labels.tolist()) == {0, 1, 2}
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    assert meta['environment']['browser']['blocked_requests'] == 0 and meta['environment']['fonts_distributed'] is False
    train = json.loads((root / 'train.json').read_text(encoding='utf-8'))
    assert {r['origin'] for r in train} == {'browser', 'noto-grammar-copy'}
    assert not (root / 'browser-temp').exists()


def _rewrite(root, split, mutate):
    rows = json.loads((root / f'{split}.json').read_text(encoding='utf-8'))
    mutate(rows)
    (root / f'{split}.json').write_text(json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    meta['splits'][split]['sha256'] = common.digest(root / f'{split}.json')
    (root / 'dataset.json').write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    for name in ('freeze.json', 'heldout-seal.json'):
        (root / name).unlink(missing_ok=True)


def test_a_face_outside_its_split_partition_is_rejected(tiny, tmp_path):
    root, _, _ = tiny
    copy = tmp_path / 'partition'
    shutil.copytree(root, copy)
    train_face = next(r for r in json.loads((copy / 'train.json').read_text(encoding='utf-8')) if r['kind'] == 'face')
    def mutate(rows):
        face = next(r for r in rows if r['kind'] == 'face')
        for key in ('text', 'text_sha256', 'core_key', 'core_digest'):
            face[key] = train_face[key]
    _rewrite(copy, 'dev', mutate)
    with pytest.raises(ValueError):
        bc.validate(copy)


def test_cross_split_label_or_pixel_reuse_is_rejected(tiny, tmp_path):
    root, _, _ = tiny
    plain = next(r for r in json.loads((root / 'train.json').read_text(encoding='utf-8')) if r['kind'] == 'plain' and r['origin'] == 'browser')
    for field in (('text', 'text_sha256'), ('pixel_sha256',)):
        copy = tmp_path / ('reuse-' + field[0])
        shutil.copytree(root, copy)
        def mutate(rows, field=field):
            target = next(r for r in rows if r['kind'] == 'plain')
            for key in field:
                target[key] = plain[key]
        _rewrite(copy, 'dev', mutate)
        with pytest.raises(ValueError):
            bc.validate(copy)


def test_copied_faces_keep_their_source_composition_key(tiny, tmp_path):
    root, _, _ = tiny
    source = {r['id']: r for r in common.read_split(bc.SOURCE, 'train')}
    train = json.loads((root / 'train.json').read_text(encoding='utf-8'))
    copied = [r for r in train if r['kind'] == 'face' and r['origin'] == 'noto-grammar-copy']
    assert copied and all(r['core_key'] == source[r['source']['row_id']]['core_key'] for r in copied)
    other = next(r for r in copied if r['core_key'] != copied[0]['core_key'])
    tampered = tmp_path / 'copied-key'
    shutil.copytree(root, tampered)
    def mutate(rows):
        face = next(r for r in rows if r['id'] == copied[0]['id'])
        face['core_key'], face['core_digest'] = other['core_key'], other['core_digest']
    _rewrite(tampered, 'train', mutate)
    with pytest.raises(ValueError, match='core partition'):
        bc.validate(tampered)


def test_planning_and_page_retries_are_bounded(monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    monkeypatch.setattr(bc, 'MAXIMUM_RETRIES', 5)
    def unsupported(*args, **kwargs):
        raise bc.Unsupported('control')
    monkeypatch.setattr(bc, 'plan_target', unsupported)
    with pytest.raises(RuntimeError, match='after 5 plans'):
        bc._plan('face', 'dev', None, None, None, None)
    monkeypatch.setattr(bc, 'render_page', unsupported)
    session = SimpleNamespace(temp_root=REPO / 'artifacts')
    with pytest.raises(RuntimeError, match='5 consecutive dev pages failed'):
        bc.browser_rows(session, None, None, 'dev', [{'kind': 'face'}], __import__('random').Random(1), None, Counter(), None, 0.0, {})


def test_generation_refuses_a_round_other_than_the_current_change(tmp_path):
    older = rp.BROWSER_RENDERING.root / 'data' / 'never-written'
    with pytest.raises(ValueError, match="current round"):
        bc.generate(older, FONTS, (100, 100, 100), seed=1)
    assert not older.exists()
