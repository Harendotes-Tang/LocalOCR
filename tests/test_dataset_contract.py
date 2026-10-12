"""Independent dataset contract checks use original tiny pixel fixtures only."""
import hashlib
from pathlib import Path
import pytest
from PIL import Image
from tools.OcrDatasets import validate as contract
from tools.OcrDatasets.export import export
from tools.OcrDatasets.training import project,projection_fingerprint
from tools.OcrDatasets.evaluate import summarize

@pytest.fixture
def corpus(tmp_path,monkeypatch):
    monkeypatch.setattr(contract,'ROOT',tmp_path)
    root=tmp_path/'artifacts'/'dataset';root.mkdir(parents=True)
    meta={'schema_version':'localocr.dataset.v1','synthetic_only':True,'license':'CC0-1.0','seed':17,
          'generator':{'version':'fixture','source_sha256':'b'*64},'renderer':{'versions':{'Pillow':'fixture'}},
          'fonts':[{'id':'fixture','sha256':'a'*64,'license_sha256':'c'*64,'source_url':'https://raw.githubusercontent.com/notofonts/noto-cjk/'+('d'*40)+'/font.otf',
                    'source_commit':'d'*40,'version':'fixture','license':'OFL-1.1'}],
          'exclusions':{'count':0},'splits':{}}
    for index,split in enumerate(contract.SPLITS):
        label=('é e\u0301  ─' if split=='train' else split+'字')
        path='images/'+split+'.png';(root/'images').mkdir(exist_ok=True)
        Image.new('RGB',(120+index,32),(250-index,250,250)).save(root/path)
        row={'id':split,'group':'text:'+contract.label_digest(label),'kind':'plain','label':label,
             'label_sha256':contract.label_digest(label),'path':path,'sha256':contract.digest(root/path),
             'width':120+index,'height':32,'region':[1,1,100,20],'font':['fixture'],'transform':{},
             'provenance':{'source':'original','license':'CC0-1.0'},'usage':split,'rendering':{'glyphs':1,'bounds_basis':'transformed-raster-alpha'}}
        contract.write(root/(split+'.json'),[row]);meta['splits'][split]={'file':split+'.json','sha256':contract.digest(root/(split+'.json')),'count':1}
    contract.write(root/'dataset.json',meta);contract.write(root/'coverage.json',{'unsupported':0})
    contract.freeze(root);return root

def rewrite(root,split,update):
    rows=contract.read(root/(split+'.json'));update(rows[0]);contract.write(root/(split+'.json'),rows)
    meta=contract.read(root/'dataset.json');meta['splits'][split]['sha256']=contract.digest(root/(split+'.json'));contract.write(root/'dataset.json',meta)

def test_frozen_tamper(corpus):
    contract.validate(corpus)
    (corpus/'images/test.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Image bytes changed'):contract.validate(corpus)

def test_exact_unicode_not_normalized(corpus):
    _,rows,_=contract.validate(corpus)
    assert rows['train'][0]['label']=='é e\u0301  ─'
    rewrite(corpus,'train',lambda row:row.update(label='é é  ─'))
    with pytest.raises(ValueError,match='Exact Unicode'):contract.validate(corpus,False)

def test_cross_split_group_leakage(corpus):
    train=contract.read(corpus/'train.json')[0]
    rewrite(corpus,'test',lambda r:r.update(label=train['label'],label_sha256=train['label_sha256'],group=train['group']))
    with pytest.raises(ValueError,match='Cross-split exact text'):contract.validate(corpus,False)

def test_duplicate_pixels(corpus):
    first=corpus/'images/train.png';target=corpus/'images/test.png';target.write_bytes(first.read_bytes())
    rewrite(corpus,'test',lambda r:r.update(sha256=contract.digest(target),width=120))
    with pytest.raises(ValueError,match='pixel duplicate'):contract.validate(corpus,False)

def test_prior_label_exclusion(corpus):
    inventory=corpus.parent/'excluded.json';label=contract.read(corpus/'test.json')[0]['label_sha256']
    contract.write(inventory,{'schema_version':'localocr.exclusions.v1','label_sha256':[label]})
    meta=contract.read(corpus/'dataset.json');meta['exclusions']={'count':1,'inventory_path':str(inventory),'inventory_sha256':contract.digest(inventory)};contract.write(corpus/'dataset.json',meta)
    with pytest.raises(ValueError,match='reserved label'):contract.validate(corpus,False)

def test_selection_demotion_immutable(corpus):
    before=contract.digest(corpus/'freeze.json');receipt=corpus.parent/'selection.json'
    value=contract.selection_use(corpus,receipt,'acceptance','Used to choose symbol correction rule')
    assert value['independent_acceptance'] is False
    assert before==contract.digest(corpus/'freeze.json')
    contract.validate(corpus)
    assert contract.usage_state(corpus,contract.validate(corpus)[2])['acceptance']['independent_acceptance'] is False

def test_export_whitelist(corpus):
    (corpus/'private.png').write_bytes(b'private');(corpus/'weight.onnx').write_bytes(b'model')
    out=corpus.parent/'export';result=export(corpus,out)
    assert result['images']==1
    assert {p.name for p in out.rglob('*') if p.is_file()}=={'dataset.json','regression.json','NOTICE.txt','regression.png'}
    assert 'exclusions' not in contract.read(out/'dataset.json')

def test_training_no_heldout_alphabet(corpus):
    out=corpus.parent/'projection';project(corpus,out)
    meta=contract.read(out/'dataset.json')
    assert meta['alphabet']==sorted(set('é e\u0301  ─dev字'))
    assert not (out/'test.json').exists()
    projection_fingerprint(out)
    meta['alphabet'].append('🧬');contract.write(out/'dataset.json',meta)
    with pytest.raises(ValueError,match='Alphabet'):projection_fingerprint(out)

def test_metrics_ambiguous_negative_emoji():
    rows=[{'kind':'emoji','label':'👩\u200d💻','baseline':'👩\u200d💻','candidate':'👩','ambiguous':False},
          {'kind':'face','label':'','baseline':'?','candidate':'?', 'ambiguous':True},
          {'kind':'blank','label':'','baseline':'X','candidate':''},
          {'kind':'plain','label':'é','baseline':'é','candidate':'e\u0301'}]
    summary=summarize(rows)
    assert summary['ambiguous_excluded']==1
    assert summary['categories']['emoji']['baseline']['graphemes']==1
    assert summary['categories']['emoji']['candidate']['character_errors']==2
    assert summary['categories']['plain']['candidate']['exact']==0
    assert summary['categories']['blank']['negative_false_outputs']=={'baseline':1,'candidate':0}
    assert summary['categories']['plain']['candidate_regressions']==1


def test_unlicensed_record_rejected(corpus):
    rewrite(corpus,'test',lambda r:r.update(provenance={'source':'original','license':'unknown'}))
    with pytest.raises(ValueError,match='Non-original'):contract.validate(corpus,False)

def test_missing_emoji_color_rejected(corpus):
    rewrite(corpus,'test',lambda r:r.update(kind='emoji',rendering={'glyphs':1,'bounds_basis':'transformed-raster-alpha','composed':True,'colored_glyphs':0}))
    with pytest.raises(ValueError,match='Emoji requires'):contract.validate(corpus,False)

def test_demoted_receipt_tamper_rejected(corpus):
    contract.selection_use(corpus,None,'test','Observed test used to select threshold')
    state=contract.read(corpus/'selection-use.json');state['splits']['test']['independent_acceptance']=True;contract.write(corpus/'selection-use.json',state)
    with pytest.raises(ValueError,match='demotion receipt'):contract.validate(corpus)


def test_png_metadata_does_not_hide_pixel_leakage(corpus):
    from PIL.PngImagePlugin import PngInfo
    target=corpus/'images/test.png'
    info=PngInfo();info.add_text('unrelated','different container metadata')
    with Image.open(corpus/'images/train.png') as image:image.save(target,pnginfo=info)
    assert contract.digest(target)!=contract.digest(corpus/'images/train.png')
    rewrite(corpus,'test',lambda r:r.update(sha256=contract.digest(target),width=120))
    with pytest.raises(ValueError,match='decoded pixel duplicate'):contract.validate(corpus,False)

def test_private_extension_rejected_before_export(corpus):
    rewrite(corpus,'regression',lambda r:r.update(private_source='C:/private/screenshot.png'))
    with pytest.raises(ValueError,match='Undeclared record'):contract.validate(corpus,False)

def test_nested_private_extension_rejected(corpus):
    rewrite(corpus,'regression',lambda r:r['transform'].update(private_source='C:/private/screenshot.png'))
    with pytest.raises(ValueError,match='Undeclared record'):contract.validate(corpus,False)


def test_private_value_cannot_hide_in_transform(corpus):
    rewrite(corpus,'regression',lambda r:r['transform'].update(size='C:/private/name.png'))
    with pytest.raises(ValueError,match='numeric transform'):contract.validate(corpus,False)
