"""Cross-round exclusions use label manifests only and preserve exact internal Unicode."""
import json
import random

import pytest

from tools.PretrainedOcr import grammar_corpus as gc, kaomoji_grammar as kg
from tools.RegionKaomoji import exposure as x
from tools.RegionKaomoji import browser_scenes as bs, browser_crops as bc


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def source(tmp_path, rows, split='dev'):
    manifest = write(tmp_path / (split + '.json'), rows)
    meta = write(tmp_path / 'dataset.json', {'schema_version': 'localocr.pretrained.dataset.v1',
          'splits': {split: {'file': manifest.name, 'sha256': x.sha(manifest)}}})
    write(tmp_path / 'freeze.json', {'dataset_sha256': x.sha(meta), 'split_sha256': {split: x.sha(manifest)}})
    return {'dataset': str(tmp_path), 'split': split}


def test_width_siblings_excluded_but_internal_unicode_is_exact(tmp_path):
    spec = source(tmp_path / 'source', [{'kind': 'face', 'text': '(ω_ω)'}, {'kind': 'plain', 'text': '正文'}])
    inventory = x.build([spec], tmp_path / 'inventory.json')
    loaded = x.load(inventory)
    assert loaded.matches('（ω_ω）') and loaded.matches('（ω_ω)') and loaded.matches('正文')
    assert not loaded.matches('(ω＿ω)') and not loaded.matches('正文 ')
    assert loaded.matches('') is False


def test_source_and_inventory_tampering_fail(tmp_path):
    spec = source(tmp_path / 'source', [{'text': '正文'}])
    path = x.build([spec], tmp_path / 'inventory.json')
    binding = x.load(path).binding
    write(tmp_path / 'source' / 'dev.json', [{'text': '变更'}])
    with pytest.raises(ValueError, match='source'):
        x.load(path)
    with pytest.raises(ValueError, match='inventory'):
        x.load_bound({**binding, 'sha256': '0' * 64})


def test_metadata_redirection_is_refused_before_any_heldout_label_bytes(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    spec = source(root, [{'text': '开发'}])
    path = x.build([spec], tmp_path / 'inventory.json')
    write(root / 'heldout.json', [{'text': '不得读取'}])
    meta = x.read(root / 'dataset.json')
    meta['splits']['dev']['file'] = 'heldout.json'
    write(root / 'dataset.json', meta)
    original_read, original_sha = x.read, x.sha
    def watched_read(path):
        assert path.name != 'heldout.json'
        return original_read(path)
    def watched_sha(path):
        assert path.name != 'heldout.json'
        return original_sha(path)
    monkeypatch.setattr(x, 'read', watched_read)
    monkeypatch.setattr(x, 'sha', watched_sha)
    with pytest.raises(ValueError, match='source'):
        x.load(path)
    with pytest.raises(ValueError, match='source'):
        x.build([spec], tmp_path / 'invalid-inventory.json')


def test_a_frozen_dev_alias_of_heldout_is_rejected_without_reading_it(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    spec = source(root, [{'text': '开发'}])
    heldout = write(root / 'heldout.json', [{'text': '不得读取'}])
    digest = x.sha(heldout)
    meta = write(root / 'dataset.json', {'splits': {'dev': {'file': 'heldout.json', 'sha256': digest}}})
    write(root / 'freeze.json', {'dataset_sha256': x.sha(meta), 'split_sha256': {'dev': digest}})
    original_read, original_sha = x.read, x.sha
    def watched_read(path):
        assert path.name != 'heldout.json'
        return original_read(path)
    def watched_sha(path):
        assert path.name != 'heldout.json'
        return original_sha(path)
    monkeypatch.setattr(x, 'read', watched_read)
    monkeypatch.setattr(x, 'sha', watched_sha)
    with pytest.raises(ValueError, match='aliases'):
        x.build([spec], tmp_path / 'invalid-inventory.json')


def test_unused_heldout_is_refused_before_opening_label_manifest(tmp_path, monkeypatch):
    spec = source(tmp_path / 'source', [{'text': '不得读取'}], 'heldout')
    original = x.read
    original_sha = x.sha
    def guarded(path):
        assert path.name != 'heldout.json'
        return original(path)
    monkeypatch.setattr(x, 'read', guarded)
    def guarded_sha(path):
        assert path.name != 'heldout.json'
        return original_sha(path)
    monkeypatch.setattr(x, 'sha', guarded_sha)
    with pytest.raises(ValueError, match='consumed'):
        x.build([spec], tmp_path / 'inventory.json')


def test_consumed_heldout_binds_receipt_and_dataset_before_labels(tmp_path):
    spec = source(tmp_path / 'source', [{'kind': 'face', 'text': '(ω_ω)'}], 'heldout')
    root = tmp_path / 'source'
    freeze = {'dataset_sha256': x.sha(root / 'dataset.json'),
              'split_sha256': {'heldout': x.sha(root / 'heldout.json')}}
    write(root / 'freeze.json', freeze)
    receipt = write(tmp_path / 'first-use.json', {'schema_version': 'localocr.pretrained.heldout-first-use.v1',
                    'dataset_binding': freeze, 'status': 'consumed-before-reading; failures remain consumed'})
    spec['first_use'] = str(receipt)
    inventory = x.build([spec], tmp_path / 'inventory.json')
    assert x.load(inventory).matches('（ω_ω）')
    write(receipt, {'schema_version': 'localocr.pretrained.heldout-first-use.v1', 'dataset_binding': {}})
    with pytest.raises(ValueError, match='source|consumed'):
        x.load(inventory)


def test_scene_target_scope_excludes_comments_but_not_fixed_chrome(tmp_path):
    spec = source(tmp_path / 'source', [{'text': '整图', 'regions': [
        {'kind': 'body', 'column': 'close', 'text': '×'},
        {'kind': 'body', 'column': 'comment', 'text': '评论'},
        {'kind': 'kaomoji', 'column': 'comment', 'text': '(ω_ω)'}]}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    assert loaded.matches('整图') and loaded.matches('评论') and loaded.matches('（ω_ω）')
    assert not loaded.matches('×')


def test_common_pool_and_both_composers_filter_seen_cores(tmp_path, monkeypatch):
    atoms = kg.default_atoms()
    spec = source(tmp_path / 'source', [{'kind': 'face', 'text': '(ω_ω)'}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    pool = bs.common_pool(atoms, lambda text: True, exclusions=loaded)
    assert not any(loaded.matches(f['text'], f['core_key']) for rows in pool.values() for f in rows)
    allowed = gc.allowed_atoms(atoms, {a for role in kg.LIST_ROLES if role != 'gaps' for a in getattr(atoms, role)} | set(atoms.bracket_chars))
    for cls in (bs.FacePicker, bc.StratifiedFaces):
        picker = cls('dev', random.Random(3), atoms, allowed, pool, set(), exclusions=loaded)
        faces = [picker() for _ in range(30)]
        assert all(not loaded.matches(f['text'], f['core_key']) and x.split_matches(f['text'], 'dev', f['core_key']) for f in faces)


def test_nonface_labels_filter_exposure_and_do_not_allow_controls_to_repeat(tmp_path):
    spec = source(tmp_path / 'source', [{'text': '！！'}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    with pytest.raises(ValueError, match='label pool'):
        bc.Labels({}, exclusions=loaded).take('dev', lambda: '！！', kg.default_atoms())


def test_optional_exclusions_absent_preserve_seeded_plans():
    atoms = kg.default_atoms()
    assert bs.common_pool(atoms, lambda text: True) == bs.common_pool(atoms, lambda text: True, exclusions=None)
    def planned(explicit):
        rng = random.Random(41)
        pool = bs.common_pool(atoms, lambda text: True)
        allowed = gc.allowed_atoms(atoms, {a for role in kg.LIST_ROLES if role != 'gaps' for a in getattr(atoms, role)} | set(atoms.bracket_chars))
        kwargs = {'exclusions': None} if explicit else {}
        picker = bs.FacePicker('dev', rng, atoms, allowed, pool, set(), **kwargs)
        return [bs.plan_comment(rng, picker, atoms, set(), **kwargs) for _ in range(8)]
    assert planned(False) == planned(True)


def test_copied_training_source_selects_whole_unexposed_groups_before_writing(tmp_path, monkeypatch):
    spec = source(tmp_path / 'old', [{'text': '旧'}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    originals = tmp_path / 'originals'
    originals.mkdir()
    records = []
    for index, text in enumerate(('旧', '新甲', '新乙')):
        image = originals / f'{index}.png'
        image.write_bytes(bytes([index]))
        records.append({'id': str(index), 'image': image.name, 'text': text, 'text_sha256': x.text_sha(text),
                        'group_id': str(index), 'kind': 'plain', 'sha256': x.sha(image)})
    monkeypatch.setattr(bc.common, 'read_split', lambda root, split: records)
    class Writer:
        def __init__(self):
            self.saved = []
        def bytes(self, path, content):
            self.saved.append((path, content))
    writer = Writer()
    copied = bc.copy_source_rows(writer, tmp_path / 'new', random.Random(3), {'plain': 2}, {}, originals, exclusions=loaded)
    assert {row['text'] for row in copied} == {'新甲', '新乙'} and len(writer.saved) == 2
    writer = Writer()
    with pytest.raises(bc.ExposedSourceExhausted):
        bc.copy_source_rows(writer, tmp_path / 'new', random.Random(3), {'plain': 2, 'punctuation': 1}, {}, originals, exclusions=loaded)
    assert not writer.saved


def test_exhausted_fixed_fragments_are_removed_before_comment_plan_sampling(tmp_path):
    spec = source(tmp_path / 'old', [{'text': t} for t in bs.cs.FRAGMENTS])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    rng = random.Random(4)
    kinds = [bs.plan_comment(rng, None, kg.default_atoms(), set(), negative=True, exclusions=loaded)[0] for _ in range(80)]
    assert kinds and 'fragment-tail' not in kinds


@pytest.mark.parametrize('module', [bs, bc])
def test_both_generators_validate_exposure_before_any_new_image_reads(tmp_path, monkeypatch, module):
    spec = source(tmp_path / 'old', [{'text': '已见'}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    root = tmp_path / 'new'
    row = {'text': '已见', 'id': 'dev-1', 'group_id': 'dev-1'}
    manifest = write(root / 'dev.json', [row])
    atoms = kg.default_atoms()
    meta = {'schema_version': 'localocr.region-scenes.v1' if module is bs else 'localocr.pretrained.dataset.v1',
            'generator_id': module.GENERATOR_ID, 'synthetic_only': True,
            'environment': {'browser': {'blocked_requests': 0}}, 'atoms_binding': kg.binding(atoms),
            'splits': {'dev': {'file': 'dev.json', 'sha256': x.sha(manifest), 'count': 1}},
            'exposure_exclusions': loaded.binding, 'exposure_split_policy': 'raw-and-width-sibling split agreement'}
    if module is bc:
        origin = tmp_path / 'origin'
        write(origin / 'freeze.json', {})
        meta.update(source={'dataset': str(origin), 'freeze_sha256': x.sha(origin / 'freeze.json')},
                    requested_counts={'train': 1, 'dev': 1}, browser_train_share=1.0,
                    exposure_distribution={'train_source_exhausted': True})
        train = write(root / 'train.json', [dict(row, id='train-1')])
        meta['splits']['train'] = {'file': 'train.json', 'sha256': x.sha(train), 'count': 1}
        monkeypatch.setattr(bc.common, 'read_split', lambda *args: [])
    write(root / 'dataset.json', meta)
    with pytest.raises(ValueError, match='exposure'):
        module.validate(root)


def test_width_sibling_split_guard_rejects_without_reassigning_the_original_core(tmp_path):
    atoms = kg.default_atoms()
    pool = bs.common_pool(atoms, lambda text: True)
    incompatible = next(f for split, rows in pool.items() for f in rows if not x.split_matches(f['text'], split, f['core_key']))
    raw_key = incompatible['core_key']
    original_owner = gc.partition(raw_key)
    assert original_owner == incompatible['core_partition']
    spec = source(tmp_path / 'old', [{'text': '旧正文'}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    with pytest.raises(ValueError, match='width-sibling split'):
        loaded.assert_clear({'text': incompatible['text'], 'core_key': raw_key, 'split': original_owner})
    assert gc.partition(raw_key) == original_owner and raw_key == incompatible['core_key']


def test_exposed_single_word_pool_uses_bounded_original_combinations_and_records_lengths(tmp_path, monkeypatch):
    tokens = (*bs.cs.CN_WORDS, *bs.cs.EN_WORDS, *bs.cs.PUNCTUATION, *(str(n) for n in range(1, 100)))
    spec = source(tmp_path / 'old', [{'text': t} for t in tokens])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    attempts = []
    def make(rng, count):
        attempts.append(count)
        return '原创组合' if count == 2 else tokens[0]
    monkeypatch.setattr(bs.cs, 'comment_words', make)
    assert bc.words(random.Random(1), kg.default_atoms(), 1, exclusions=loaded) == '原创组合'
    assert attempts == [2]
    assert loaded.word_recovery_summary()['requested_to_actual_word_counts'] == {'1->2': 1}
    assert loaded.word_recovery_summary()['single_word_pool_fully_exposed'] is True
    # Without opt-in exclusions even the exposed original one-word output stays unchanged.
    assert bc.words(random.Random(1), kg.default_atoms(), 1) == tokens[0]


def test_word_recovery_never_expands_beyond_six_or_bypasses_exclusion(tmp_path, monkeypatch):
    spec = source(tmp_path / 'old', [{'text': '已见'}])
    loaded = x.load(x.build([spec], tmp_path / 'inventory.json'))
    attempts = []
    def make(rng, count):
        attempts.append(count)
        return '已见'
    monkeypatch.setattr(bs.cs, 'comment_words', make)
    with pytest.raises(ValueError, match='bounded length recovery'):
        loaded.words(random.Random(1), 5, kg.default_atoms())
    assert set(attempts) == {5, 6} and len(attempts) == 2000
