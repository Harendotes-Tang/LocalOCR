"""Data family/exact Unicode/pixel isolation regressions; no inference claim."""
import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image
from tools.PretrainedOcr.corpus import samples, text_hash
from tools.PretrainedOcr.data import validate, pixels, sha, write, disk_gate


def fixture_dataset(tmp_path):
    splits = {}; tokens=set()
    for index, split in enumerate(('train','dev','heldout')):
        image=Image.new('RGB',(30+index,20),(index+1,20,30));path=tmp_path/(split+'.png');image.save(path)
        text=('  Aa  e\u0301  ' if split=='train' else 'Aa' if split=='dev' else 'Ω')
        if split=='train':tokens.update(text)
        row={'split':split,'image':path.name,'text':text,'text_sha256':text_hash(text),'group_id':split,'family':split+':grammar','pixel_sha256':pixels(image),'width':image.width,'height':image.height,'sha256':sha(path),'provenance':{'source':'original','license':'CC0-1.0'}}
        rp=tmp_path/(split+'.json');write(rp,[row]);splits[split]={'file':rp.name,'sha256':sha(rp),'count':1,'positive_groups':1}
    write(tmp_path/'dataset.json',{'schema_version':'localocr.pretrained.dataset.v1','synthetic_only':True,'splits':splits,'train_vocabulary':sorted(tokens,key=ord)})
    return tmp_path


def change(root,split,mutate):
    p=root/(split+'.json');rows=json.loads(p.read_text(encoding='utf-8'));mutate(rows[0]);write(p,rows)
    meta=json.loads((root/'dataset.json').read_text(encoding='utf-8'));meta['splits'][split]['sha256']=sha(p);write(root/'dataset.json',meta)


def test_structural_families_and_labels_disjoint():
    pools={s:list(samples(s,200,37)) for s in ('train','dev','heldout')}
    for a,b in [('train','dev'),('train','heldout'),('dev','heldout')]:
        assert not {r['text_sha256'] for r in pools[a]} & {r['text_sha256'] for r in pools[b]}
        assert not {r['family'] for r in pools[a]} & {r['family'] for r in pools[b]}
    assert any(r['text'].startswith(' ') for r in pools['train'])
    assert any('  ' in r['text'] for r in pools['train'])
    assert any('\u0301' in r['text'] for r in pools['train'])
    assert any('\u200d' in r['text'] for r in pools['train'])


def test_prior_exact_labels_are_excluded():
    first=list(samples('train',100,11));excluded={r['text_sha256'] for r in first}
    again=list(samples('train',100,11,excluded))
    assert not excluded & {r['text_sha256'] for r in again}


def test_vocabulary_uses_only_train(tmp_path):
    root=fixture_dataset(tmp_path);assert validate(root)['counts']['heldout']['count']==1
    assert 'Ω' not in json.loads((root/'dataset.json').read_text())['train_vocabulary']


def test_cross_family_leak_rejected(tmp_path):
    root=fixture_dataset(tmp_path);change(root,'dev',lambda r:r.update(family='train:grammar'))
    with pytest.raises(ValueError,match='family'):validate(root)


def test_unicode_space_trim_rejected(tmp_path):
    root=fixture_dataset(tmp_path);change(root,'train',lambda r:r.update(text=r['text'].strip()))
    with pytest.raises(ValueError,match='Unicode'):validate(root)


def test_same_pixels_with_distinct_png_metadata_rejected(tmp_path):
    root=fixture_dataset(tmp_path)
    from PIL.PngImagePlugin import PngInfo
    with Image.open(root/'train.png') as im:
        info=PngInfo();info.add_text('metadata','different');im.save(root/'dev.png',pnginfo=info)
        pixel=pixels(im);size=im.size
    change(root,'dev',lambda r:r.update(pixel_sha256=pixel,width=size[0],height=size[1],sha256=sha(root/'dev.png')))
    with pytest.raises(ValueError,match='pixels'):validate(root)


def test_seal_binding_rejected(tmp_path):
    root=fixture_dataset(tmp_path);write(root/'heldout-seal.json',{'heldout_sha256':'wrong','dataset_sha256':sha(root/'dataset.json')})
    with pytest.raises(ValueError,match='seal'):validate(root)


def test_freeze_cannot_be_rewritten(tmp_path):
    root=fixture_dataset(tmp_path);binding=validate(root);write(root/'freeze.json',binding)
    change(root,'dev',lambda r:r.update(family='dev:new-family'))
    with pytest.raises(ValueError,match='Freeze'):validate(root)


def test_real_official_fonts_reject_missing_glyph_and_keep_space_canvas():
    from tools.OcrDatasets.render import Renderer, Unsupported
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Official local font assets not installed')
    renderer=Renderer(profile)
    with pytest.raises(Unsupported):renderer.render(chr(0x10ffff))
    image,region,used,evidence=renderer.render('  Aa  cafe\u0301  ')
    assert evidence['glyphs']>0 and used
    assert region[0]>0 and region[0]+region[2]<image.width
    image,region,used,evidence=renderer.render('👩\u200d💻🐱',emoji=True)
    assert evidence['glyphs']==2 and evidence['colored_glyphs']==2


def test_ast_identity_is_structural_not_split_prefix():
    from tools.PretrainedOcr.corpus import GRAMMARS,grammar_family
    asts={s:set(recipes.values()) for s,recipes in GRAMMARS.items()}
    for a,b in [('train','dev'),('train','heldout'),('dev','heldout')]:assert not asts[a]&asts[b]
    for split,recipes in GRAMMARS.items():
        for recipe,expr in recipes.items():assert grammar_family(split,recipe)=='grammar:'+text_hash(expr)


def test_real_mixed_emoji_uses_color_and_regular_font_runs():
    from tools.PretrainedOcr.data import MixedRenderer
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Official local fonts not installed')
    renderer=MixedRenderer(profile)
    image,region,used,evidence=renderer.render('Icon[👩\u200d💻🐱]=🌈',emoji=True)
    assert evidence['colored_glyphs']==3 and evidence['glyphs']>3
    assert 'emoji' in used and 'cjk' in used
    assert image.width>region[2]


def test_negative_recipe_structure_is_disjoint():
    from tools.PretrainedOcr.data import negative_image
    import random
    structures={};families={}
    for split in ('train','dev','heldout'):
        rows=[negative_image(split,i,random.Random(31)) for i in (0,1)]
        structures[split]={json.dumps(r[3]['structure'],sort_keys=True) for r in rows}
        families[split]={r[2] for r in rows}
        assert all(not r[3]['structure']['geometry'] for r in rows if r[1]=='blank')
    for a,b in [('train','dev'),('train','heldout'),('dev','heldout')]:
        assert not structures[a]&structures[b] and not families[a]&families[b]


def test_training_consumer_rejects_missing_synthetic_declaration(tmp_path):
    root=fixture_dataset(tmp_path)
    meta=json.loads((root/'dataset.json').read_text(encoding='utf-8'));meta.pop('synthetic_only');write(root/'dataset.json',meta)
    with pytest.raises(ValueError,match='synthetic'):validate(root)
