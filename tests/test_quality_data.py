"""Exact synthetic quality corpus and verified raster isolation regressions."""
from collections import Counter
from pathlib import Path
import json
import pytest
from tools.PretrainedOcr import quality_corpus as corpus, quality_data as data


def test_no_metadata_serial_in_face_and_all_brackets_train():
    rows=list(corpus.samples('train', 1200, 31))
    faces=[r for r in rows if r['kind']=='face']
    assert len(faces)==720
    assert len({r['family'] for r in faces})>=6
    for left,right in corpus.BRACKETS:
        assert any(left in r['text'] and right in r['text'] for r in faces)
    assert all(not any(c.isdecimal() and c!='3' for c in r['text']) for r in faces)
    assert all('serial' not in r['recipe_expression'] for r in rows)
    assert any('  ' in r['text'] for r in faces)
    assert any(chr(0x0301) in r['text'] for r in rows)
    assert any(chr(0x200d) in r['text'] for r in rows)
    assert any(r['text'].startswith(' ') for r in rows if r['kind']=='plain')


def test_full_label_and_actual_recipe_family_partition():
    seen=set(); pools={}
    for index,split in enumerate(corpus.SPLITS):
        pools[split]=list(corpus.samples(split, 1600, 45+index, seen))
        seen.update(r['text_sha256'] for r in pools[split] if r['text'])
    for a,b in [('train','dev'),('train','heldout'),('dev','heldout')]:
        assert not {r['family'] for r in pools[a]} & {r['family'] for r in pools[b]}
        assert not {r['text_sha256'] for r in pools[a] if r['text']} & {r['text_sha256'] for r in pools[b] if r['text']}
    for rows in pools.values():
        for row in rows:
            assert row['family']==corpus.family(row['recipe_expression'])
            if row['text']:assert row['group_id']=='text:'+corpus.text_hash(row['text'])


def test_total_group_mixture_and_excluded_labels():
    first=list(corpus.samples('train',100,77))
    assert Counter(r['kind'] for r in first)=={'face':60,'plain':20,'punctuation':5,'emoji':5,'blank':5,'graphic':5}
    excluded={r['text_sha256'] for r in first if r['text']}
    second=list(corpus.samples('train',100,77,excluded))
    assert not excluded & {r['text_sha256'] for r in second if r['text']}
    with pytest.raises(ValueError):list(corpus.samples('wrong',1,1))


def test_disk_and_owned_quota_fail_before_write(tmp_path,monkeypatch):
    monkeypatch.setattr(data.shutil,'disk_usage',lambda _:type('D',(),{'free':8*1024**3-1})())
    with pytest.raises(ValueError,match='8GiB'):data.storage_gate(tmp_path)
    monkeypatch.setattr(data.shutil,'disk_usage',lambda _:type('D',(),{'free':9*1024**3})())
    monkeypatch.setattr(data,'OWNED_BYTE_LIMIT',1)
    (tmp_path/'payload').write_bytes(b'xx')
    with pytest.raises(ValueError,match='350MiB'):data.storage_gate(tmp_path)


def test_real_verified_font_views_preserve_space_and_exclude_notdef():
    from tools.OcrDatasets.render import Unsupported
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Pinned local OFL fonts absent')
    renderer=data.MixedRenderer(profile)
    image,region,fonts,evidence=data.render_variant(renderer,'  (o_3)  ',0)
    assert fonts and evidence['glyphs']>0 and region[0]>0
    assert image.width>region[0]+region[2]
    second,*_=data.render_variant(renderer,'  (o_3)  ',1)
    assert data.pixels(image)!=data.pixels(second)
    with pytest.raises(Unsupported):data.render_variant(renderer,chr(0x10ffff),0)


@pytest.fixture
def generated(tmp_path,monkeypatch):
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Pinned local OFL fonts absent')
    monkeypatch.setattr(data,'ARTIFACTS',tmp_path)
    root=tmp_path/'quality-optimization-test/data/fixture'
    data.generate(root,profile,(20,20,20),(2,2,2),seed=93)
    return root


def update(root,split,fn):
    path=root/(split+'.json');rows=data.read(path);fn(rows[0]);data.write(path,rows)
    meta=data.read(root/'dataset.json');meta['splits'][split]['sha256']=data.sha(path);data.write(root/'dataset.json',meta)


def test_actual_generation_exact_counts_and_adapter_read(generated):
    from tools.PretrainedOcr.common import read_split,train_characters
    assert data.validate(generated)['counts']['train']=={'count':40,'positive_groups':18,'groups':20}
    assert len(read_split(generated,'train'))==40
    assert train_characters(generated)==data.read(generated/'dataset.json')['train_vocabulary']
    with pytest.raises(ValueError,match='Heldout'):read_split(generated,'heldout')
    assert data.read(generated/'heldout-seal.json')['results_seen'] is False


def test_changed_space_label_rejected(generated):
    update(generated,'train',lambda row:row.update(text=row['text']+' '))
    with pytest.raises(ValueError,match='Unicode'):data.validate(generated)


def test_family_false_split_and_image_hash_rejected(generated):
    update(generated,'dev',lambda row:row.update(recipe_expression=corpus.FACE_FORMS['heldout'][0]))
    with pytest.raises(ValueError,match='AST'):data.validate(generated)


def test_frozen_metadata_and_seal_tamper_rejected(generated):
    seal=data.read(generated/'heldout-seal.json');seal['heldout_sha256']='0'*64;data.write(generated/'heldout-seal.json',seal)
    with pytest.raises(ValueError,match='seal'):data.validate(generated)


def test_image_byte_tamper_fails_before_decoding(generated):
    row=data.read(generated/'dev.json')[0];path=generated/row['image'];path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash'):data.validate(generated)


def test_writer_reserves_every_encoded_byte_before_file_creation(tmp_path,monkeypatch):
    state={'owned_bytes':0,'round_bytes':0,'free_bytes':9*1024**3}
    monkeypatch.setattr(data,'storage_gate',lambda *args,**kwargs:dict(state))
    monkeypatch.setattr(data,'OWNED_BYTE_LIMIT',5)
    writer=data.Writer(tmp_path);writer.bytes(tmp_path/'first',b'12345')
    with pytest.raises(ValueError,match='350MiB'):writer.bytes(tmp_path/'excess',b'x')
    assert not (tmp_path/'excess').exists()


def test_writer_global_bytes_and_external_reconciliation(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'quota_root',lambda path:Path(path))
    state={'owned_bytes':0,'round_bytes':0,'free_bytes':9*1024**3}
    monkeypatch.setattr(data,'storage_gate',lambda *args,**kwargs:dict(state))
    monkeypatch.setattr(data,'ROUND_BYTE_LIMIT',5)
    writer=data.Writer(tmp_path);writer.bytes(tmp_path/'first',b'12345')
    with pytest.raises(ValueError,match='1GiB'):writer.bytes(tmp_path/'excess',b'x')
    assert not (tmp_path/'excess').exists()
    state['round_bytes']=5;writer.reconciled-=31
    with pytest.raises(ValueError,match='1GiB'):writer.bytes(tmp_path/'external',b'x')
    assert not (tmp_path/'external').exists()


def test_live_external_write_cannot_break_global_limit(tmp_path,monkeypatch):
    root=tmp_path/'quality-optimization-test';dataset=root/'data/generator';dataset.mkdir(parents=True)
    monkeypatch.setattr(data,'ROUND_BYTE_LIMIT',10)
    writer=data.Writer(dataset);writer.bytes(dataset/'first',b'123')
    (root/'other-owner-output').write_bytes(b'1234567')
    with pytest.raises(ValueError,match='1GiB'):writer.bytes(dataset/'excess',b'x')
    assert not (dataset/'excess').exists()


def test_final_generation_receipt_failure_does_not_publish_freeze(tmp_path,monkeypatch):
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Pinned local OFL fonts absent')
    monkeypatch.setattr(data,'ARTIFACTS',tmp_path)
    original=data.Writer.json
    def fail_generation(self,path,value):
        if Path(path).name=='generation.json':raise OSError('injected final receipt failure')
        return original(self,path,value)
    monkeypatch.setattr(data.Writer,'json',fail_generation)
    root=tmp_path/'quality-optimization-test/data/incomplete'
    with pytest.raises(OSError,match='injected'):data.generate(root,profile,(20,20,20),(2,2,2))
    assert not (root/'freeze.json').exists()
    from tools.PretrainedOcr.common import dataset_binding
    with pytest.raises(FileNotFoundError):dataset_binding(root)
