"""Actual mixed-scene geometry, label partitions and GT-free input binding."""
from pathlib import Path
import numpy as np
import pytest
from tools.RegionKaomoji import quality_scenes as scenes, scenes as old
from tools.PretrainedOcr import quality_data


def test_layout_asts_are_actual_and_disjoint():
    assert set(scenes.RECIPES['dev'])==set(scenes.RECIPES['heldout'])
    assert not set(scenes.RECIPES['dev'].values()) & set(scenes.RECIPES['heldout'].values())
    for split,recipes in scenes.RECIPES.items():
        for kind,expression in recipes.items():
            assert scenes.family(split,kind)=='quality-scene-grammar:'+scenes.th(expression)
    assert {'double-face','tight-body','edge-face','square-face','curly-face','fullwidth-face'}<=set(scenes.TYPES)


@pytest.fixture
def dataset(tmp_path,monkeypatch):
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Pinned local OFL fonts absent')
    monkeypatch.setattr(quality_data,'ARTIFACTS',tmp_path)
    root=tmp_path/'quality-optimization-test/scenes/fixture'
    scenes.generate(root,profile,(len(scenes.TYPES),len(scenes.TYPES)),seed=51)
    return root


def test_real_masks_spans_and_backward_dispatch(dataset):
    binding=old.validate(dataset)
    assert binding==scenes.validate(dataset)
    assert binding['counts']['dev']['scenes']==len(scenes.TYPES)
    rows=scenes.read(dataset/'dev.json')
    double=next(r for r in rows if r['scene_kind']=='double-face')
    assert sum(r['kind']=='kaomoji' for r in double['regions'])==2
    assert any(r['kind']=='kaomoji' and r['size']<=12 for row in rows for r in row['regions'])
    for row in rows:
        for region in row['regions']:
            if region['span'] is not None:
                a,b=region['span'];assert row['text'][a:b]==region['text']


def test_inference_index_contains_only_six_image_fields(dataset):
    index=scenes.read(dataset/'inference-inputs.json');binding=scenes.read(dataset/'freeze.json')
    assert index['dataset_sha256']==binding['dataset_sha256']
    assert all(set(job)=={'id','image','sha256','pixel_sha256','width','height'} for jobs in index['jobs'].values() for job in jobs)
    from tools.RegionKaomoji.evaluate import image_jobs
    assert len(image_jobs(dataset,'dev'))==len(scenes.TYPES)
    with pytest.raises(ValueError,match='Heldout'):image_jobs(dataset,'heldout')
    assert scenes.read(dataset/'heldout-seal.json')['inference_results_seen'] is False


def test_ordinary_numbers_and_single_letters_are_body_truth(dataset):
    rows=scenes.read(dataset/'dev.json')
    rows=[row for row in rows if row['scene_kind'] in ('ordinary-brackets','ordinary-numbers','letter-brackets')]
    assert len(rows)==3 and all(region['kind']=='body' for row in rows for region in row['regions'])


def test_span_and_mask_tampering_rejected(dataset):
    path=dataset/'dev.json';rows=scenes.read(path);rows[0]['regions'][0]['span'][0]+=1;scenes.write(path,rows)
    meta=scenes.read(dataset/'dataset.json');meta['splits']['dev']['sha256']=scenes.sha(path);scenes.write(dataset/'dataset.json',meta)
    with pytest.raises(ValueError,match='span'):scenes.validate(dataset)


def test_recipe_family_tampering_rejected(dataset):
    path=dataset/'dev.json';rows=scenes.read(path);rows[0]['recipe_family']=scenes.family('heldout',rows[0]['scene_kind']);scenes.write(path,rows)
    meta=scenes.read(dataset/'dataset.json');meta['splits']['dev']['sha256']=scenes.sha(path);scenes.write(dataset/'dataset.json',meta)
    with pytest.raises(ValueError,match='AST'):scenes.validate(dataset)


def test_repeated_graphic_primitives_have_distinct_real_pixels(tmp_path,monkeypatch):
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Pinned local OFL fonts absent')
    monkeypatch.setattr(quality_data,'ARTIFACTS',tmp_path)
    root=tmp_path/'quality-optimization-test/scenes/repeated'
    scenes.generate(root,profile,(100,100),seed=71)
    rows=scenes.read(root/'dev.json')
    graphics=[r for r in rows if r['scene_kind']=='graphic-only']
    assert len(graphics)==5 and len({r['pixel_sha256'] for r in graphics})==5
    assert scenes.validate(root)['counts']['heldout']['scenes']==100


def test_final_scene_receipt_failure_does_not_publish_freeze(tmp_path,monkeypatch):
    profile=Path(__file__).resolve().parents[1]/'artifacts/ocr-data-fonts-v1/fonts.json'
    if not profile.exists():pytest.skip('Pinned local OFL fonts absent')
    monkeypatch.setattr(quality_data,'ARTIFACTS',tmp_path)
    original=quality_data.Writer.json
    def fail_generation(self,path,value):
        if Path(path).name=='generation.json':raise OSError('injected scene receipt failure')
        return original(self,path,value)
    monkeypatch.setattr(quality_data.Writer,'json',fail_generation)
    root=tmp_path/'quality-optimization-test/scenes/incomplete'
    with pytest.raises(OSError,match='injected'):scenes.generate(root,profile,(20,20))
    assert not (root/'freeze.json').exists()
    from tools.RegionKaomoji.evaluate import scene_binding
    with pytest.raises(FileNotFoundError):scene_binding(root)
