"""Actual OFL scene-coordinate regressions; no OCR/model inference claims."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from tools.RegionKaomoji import scenes

@pytest.fixture
def dataset(tmp_path):
 profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
 if not profile.exists():pytest.skip('Official local OFL fonts absent')
 root=tmp_path/'scenes';scenes.generate(root,profile,(10,10),seed=19);return root

def update_row(root,split,fn):
 p=root/(split+'.json');rows=scenes.read(p);fn(rows[0]);scenes.write(p,rows)
 meta=scenes.read(root/'dataset.json');meta['splits'][split]['sha256']=scenes.sha(p);scenes.write(root/'dataset.json',meta)

def test_actual_full_scene_geometry_span_and_no_overlap(dataset):
 binding=scenes.validate(dataset);assert binding['counts']['dev']['scenes']==10
 for row in scenes.read(dataset/'dev.json'):
  for region in row['regions']:
   with Image.open(dataset/region['mask']) as m:
    assert scenes.bbox(m)==region['bbox'] and m.size==(row['width'],row['height'])
    assert int((np.asarray(m)>0).sum())==region['pixel_count']
   if region['span'] is not None:
    a,b=region['span'];assert row['text'][a:b]==region['text']

def test_recipe_structures_not_split_prefix():
 dev=set(scenes.RECIPES['dev'].values());held=set(scenes.RECIPES['heldout'].values());assert not dev&held
 for split,recipes in scenes.RECIPES.items():
  for kind,expression in recipes.items():assert scenes.family(split,kind)=='scene-grammar:'+scenes.th(expression)

def test_half_open_bbox_off_by_one_fails(dataset):
 update_row(dataset,'dev',lambda row:row['regions'][0]['bbox'].__setitem__(2,row['regions'][0]['bbox'][2]-1))
 with pytest.raises(ValueError,match='bbox'):scenes.validate(dataset)

def test_unicode_span_tampering_fails(dataset):
 update_row(dataset,'dev',lambda row:row['regions'][0]['span'].__setitem__(0,1))
 with pytest.raises(ValueError,match='character span'):scenes.validate(dataset)

def test_source_family_swap_fails(dataset):
 update_row(dataset,'dev',lambda row:row.update(recipe_family=scenes.family('heldout',row['scene_kind'])))
 with pytest.raises(ValueError,match='AST'):scenes.validate(dataset)

def test_mask_padded_into_background_is_not_true_ink(dataset):
 def mutate(row):
  region=row['regions'][0];p=dataset/region['mask']
  with Image.open(p) as image:
   image.putpixel((0,0),255);image.save(p);region['bbox']=scenes.bbox(image);region['pixel_count']=int((np.asarray(image)>0).sum())
  region['mask_sha256']=scenes.sha(p)
 update_row(dataset,'dev',mutate)
 with pytest.raises(ValueError,match='source image ink'):scenes.validate(dataset)

def test_freeze_metadata_cannot_change(dataset):
 meta=scenes.read(dataset/'dataset.json');meta['seed']+=1;scenes.write(dataset/'dataset.json',meta)
 with pytest.raises(ValueError,match='freeze'):scenes.validate(dataset)

def test_blank_and_graphics_have_no_invented_text_truth(dataset):
 rows=scenes.read(dataset/'dev.json');blank=next(r for r in rows if r['scene_kind']=='blank');graphic=next(r for r in rows if r['scene_kind']=='graphic-only')
 assert blank['text']=='' and blank['regions']==[]
 assert graphic['text']=='' and all(r['span'] is None and r['text']=='' for r in graphic['regions'])
 assert all(r['baseline_omission'].startswith('not measured') for r in rows)

def test_heldout_seal_tampering_fails(dataset):
 seal=scenes.read(dataset/'heldout-seal.json');seal['heldout_sha256']='bad';scenes.write(dataset/'heldout-seal.json',seal)
 with pytest.raises(ValueError,match='seal'):scenes.validate(dataset)
