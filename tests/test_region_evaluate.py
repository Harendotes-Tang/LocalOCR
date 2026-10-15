"""Synthetic policy fixtures and real tiny ONNX tensors, never OCR accuracy claims."""
import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import pytest
from tools.PretrainedOcr import common
from tools.RegionKaomoji import evaluate as e

GRAPH='CAg6vQIKOQoFaW1hZ2USBG1lYW4iClJlZHVjZU1lYW4qDQoEYXhlc0ABQAKgAQcqDwoIa2VlcGRpbXMYAKABAgohCgRtZWFuCgRheGlzEghleHBhbmRlZCIJVW5zcXVlZXplChwKCGV4cGFuZGVkCgR6ZXJvEgV6ZXJvcyIDTXVsCiMKBXplcm9zCgZzY29yZXMSDXByb2JhYmlsaXRpZXMiA0FkZBINcmVhbC1jcHUtdGVzdCoNCAEQBzoBAkIEYXhpcyoOEAEiBAAAAABCBHplcm8qIggBCAEIBBABIhAAAAAAAACAPwAAAAAAAAAAQgZzY29yZXNaIQoFaW1hZ2USGAoWCAESEgoDEgFOCgIIAwoCCDAKAxIBV2IlCg1wcm9iYWJpbGl0aWVzEhQKEggBEg4KAxIBTgoDEgFUCgIIBEIECgAQDQ=='


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def root(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'EVALUATION_ROOT',tmp_path/'region')
    e.EVALUATION_ROOT.mkdir()
    return tmp_path


def fixture_dataset(root):
    root.mkdir()
    image=Image.new('RGB',(80,40),'white');ImageDraw.Draw(image).rectangle((4,5,18,18),fill='black')
    image.save(root/'image.png')
    mask=Image.new('L',image.size);ImageDraw.Draw(mask).rectangle((4,5,18,18),fill=255);mask.save(root/'mask.png')
    row={'id':'scene','image':'image.png','sha256':common.digest(root/'image.png'),'pixel_sha256':e.c.image_digest(image),
         'width':80,'height':40,'scene_kind':'ordinary-brackets','recipe_family':'synthetic-unit-fixture',
         'text':' A  ','text_sha256':sha(b' A  '),'provenance':{'source':'original','license':'CC0-1.0'},
         'regions':[{'id':'body','kind':'body','text':' A  ','span':[0,4],'bbox':[4,5,19,19],
                     'mask':'mask.png','mask_sha256':common.digest(root/'mask.png'),'pixel_count':15*14}]}
    splits={}
    for split in ('dev','heldout'):
        common.write(root/(split+'.json'),[dict(row,split=split,source_split=split)])
        splits[split]={'file':split+'.json','sha256':common.digest(root/(split+'.json')),'count':1}
    common.write(root/'dataset.json',{'schema_version':'localocr.region-scenes.v1','synthetic_only':True,'splits':splits})
    frozen={'schema_version':'localocr.region-scenes.freeze.v1','dataset_sha256':common.digest(root/'dataset.json'),
            'split_sha256':{split:item['sha256'] for split,item in splits.items()},
            'counts':{'dev':{'scenes':1,'faces':0},'heldout':{'scenes':1,'faces':0}}}
    common.write(root/'freeze.json',frozen)
    common.write(root/'heldout-seal.json',{'schema_version':'localocr.region-scenes.seal.v1',
                 'dataset_sha256':frozen['dataset_sha256'],'heldout_sha256':frozen['split_sha256']['heldout'],
                 'inference_results_seen':False,'requires_frozen_pipeline_selection':True})
    job={key:row[key] for key in e.JOB_KEYS}
    common.write(root/'inference-inputs.json',{'schema_version':'localocr.region-scenes.inputs.v1',
                 'dataset_sha256':frozen['dataset_sha256'],'split_sha256':frozen['split_sha256'],
                 'jobs':{'dev':[job],'heldout':[job]}})
    return root,job


def tiny_candidate(root,identity):
    root.mkdir();(root/'model.onnx').write_bytes(base64.b64decode(GRAPH))
    (root/'source.bin').write_bytes(b'unit tensor provenance, not trained OCR weights')
    (root/'dictionary.txt').write_bytes(b'A\ne\n')
    common.write(root/'vocabulary.json',{'schema_version':'localocr.pretrained.vocabulary.v1',
                 'tokens':['<blank>','A','e',' '],'blank_id':0,'space_id':3})
    file=lambda name:{'file':name,'sha256':common.digest(root/name)}
    value={'schema_version':'localocr.pretrained.candidate.v1','id':identity,'family':identity,'stage':'finetuned',
       'model':file('model.onnx'),'vocabulary':dict(file('vocabulary.json'),blank_id=0,space_id=3),
       'preprocessing':{'height':48,'color':'RGB','width':{'mode':'dynamic','min':16,'max':1024,'multiple':4,'rounding':'ceil-multiple'},
                        'padding':'right-zero-normalized','normalize':{'mean':[.5]*3,'std':[.5]*3},'resize':'PIL-bilinear'},
       'decoder':{'type':'ctc','output':'probabilities','blank_id':0,'repeat':'ctc'},
       'source':{'checkpoint':file('source.bin'),'dictionary':file('dictionary.txt')},'enabled':False,'promote':False}
    common.write(root/'candidate.json',value)
    return root/'candidate.json'


def geometry():
    image=Image.new('RGB',(160,80),'white');draw=ImageDraw.Draw(image)
    draw.arc((20,20,34,50),90,270,fill='black',width=2);draw.arc((88,20,102,50),270,90,fill='black',width=2)
    draw.ellipse((43,31,50,38),outline='black',width=2);draw.ellipse((72,31,79,38),outline='black',width=2)
    draw.line((54,43,68,43),fill='black',width=2)
    return image


class UnitBaseline:
    """A declared unit geometry stub; production correctness is tested by live dev."""
    def predict(self,image):
        return [{'bbox':[20,20,103,51],'text':'(Д_Д)','score':.001}],{'unit_fixture':True}


def test_truth_free_index_and_default_heldout_denial(root,monkeypatch):
    dataset,job=fixture_dataset(root/'data')
    original_read=common.read
    def no_truth(path):
        assert Path(path).name not in ('dev.json','heldout.json','mask.png')
        return original_read(path)
    monkeypatch.setattr(common,'read',no_truth)
    assert e.image_jobs(dataset,'dev')==[job]
    with pytest.raises(ValueError,match='Heldout'):e.image_jobs(dataset,'heldout')
    assert e.image_for_job(dataset,job).size==(80,40)


def test_truth_or_extra_options_cannot_enter_jobs(root):
    dataset,_=fixture_dataset(root/'data');path=dataset/'inference-inputs.json';value=common.read(path)
    value['jobs']['dev'][0]['text']='oracle';common.write(path,value)
    with pytest.raises(ValueError,match='truth-bearing'):e.image_jobs(dataset,'dev')


def test_snapshot_image_dimensions_pixels_and_metadata_are_bound(root):
    dataset,job=fixture_dataset(root/'data')
    image=e.image_for_job(dataset,job);image.putpixel((0,0),(0,0,0));image.save(dataset/'image.png')
    with pytest.raises(ValueError,match='bytes'):e.image_jobs(dataset,'dev')
    with pytest.raises(ValueError,match='changed'):e.image_for_job(dataset,job)
    common.write(dataset/'dataset.json',{})
    with pytest.raises(ValueError,match='frozen'):e.scene_binding(dataset)


def test_real_specialist_cpu_views_are_separate_for_each_model(root):
    models=[e.Recognizer(tiny_candidate(root/name,name)) for name in ('one','two')]
    result=e.predict_scene(geometry(),UnitBaseline(),models,e.configuration())
    assert set(result['models'])=={'one','two'}
    for identity,model in result['models'].items():
        records=next(iter(model['views'].values()))
        assert [r['transform'] for r in records]==['raw','edge-pad-6']
        assert {r['model_id'] for r in records}=={identity}
        assert records[0]['image_sha256']!=records[1]['image_sha256']
        assert {r['text'] for r in records}=={'A'}
        assert model['regions'][0]['text']=='(Д_Д)'
        assert not model['accepted_count']


def test_ineligible_body_does_not_invoke_specialist(root):
    class OrdinaryBaseline:
        def predict(self,image):return [{'bbox':[20,20,103,51],'text':'(123)','score':.01}],{}
    class NoCall:
        binding={'id':'unit-no-call'}
        def predict(self,image):raise AssertionError('ineligible body entered specialist')
    # No measured bracket strokes: body glyphs only, so nothing routes to the specialist.
    plain=Image.new('RGB',(160,80),'white');draw=ImageDraw.Draw(plain)
    draw.rectangle((45,31,50,38),fill='black');draw.rectangle((72,31,79,38),fill='black');draw.line((54,43,68,43),fill='black',width=2)
    result=e.predict_scene(plain,OrdinaryBaseline(),[NoCall()],e.configuration())
    assert not result['models']['unit-no-call']['views']
    assert result['models']['unit-no-call']['regions'][0]['text']=='(123)'


def test_exact_unicode_and_declared_geometry_join(root):
    regions=[{'bbox':[20,10,40,20],'text':'ω  Д','score':.1},{'bbox':[1,10,10,20],'text':' A ','score':.9},
             {'bbox':[1,30,10,40],'text':'e\u0301','score':.1}]
    assert e.joined_regions(regions,e.configuration())==' A   ω  Д\ne\u0301'


def test_configuration_guards_are_not_shared_mutable_state(root):
    config=e.configuration();config['limits']=dict(config['limits'],scene_timeout_seconds=100)
    common.write(root/'config.json',config)
    with pytest.raises(ValueError,match='budgets'):e.configuration(root/'config.json')
    config=e.configuration();config['views']=['raw','raw'];common.write(root/'config.json',config)
    with pytest.raises(ValueError):e.configuration(root/'config.json')


def test_mask_alignment_keeps_mixed_transcript_unsplit(root):
    masks=[np.zeros((20,40),bool),np.zeros((20,40),bool)]
    masks[0][5:10,3:8]=True;masks[1][5:10,25:30]=True
    regions=[{'bbox':[1,3,35,13],'text':'body (Д_Д)','score':.9}]
    assert e.aligned_region(regions,0,masks,e.configuration())==('',1)
    assert e.aligned_region(regions,1,masks,e.configuration())==('',1)
    assert e.mask_pixels(masks[1],[25,5,30,10])==25
    assert e.iou([0,0,10,10],[5,0,15,10])==pytest.approx(1/3)


def binding_fixture(root,monkeypatch):
    dataset,job=fixture_dataset(root/'data')
    models=[tiny_candidate(root/name,name) for name in ('one','two')]
    monkeypatch.setattr(e,'baseline_binding',lambda path:{'id':'production-baseline','version':'fixture','manifest_sha256':'a'*64,'models':{}})
    baseline=root/'baseline';baseline.mkdir()
    config=e.configuration()
    binding=e.pipeline_binding(dataset,models,baseline,config)
    folder=e.EVALUATION_ROOT/'evaluation-fixture';folder.mkdir()
    report_path=folder/'evaluation.json'
    # State-machine fixture flags are explicit; these files prove no OCR accuracy.
    monkeypatch.setattr(e,'verify_baseline_runtime',lambda value: None)  # Only this declared state-machine fixture.
    resources={'baseline_runtime':{'unit_fixture':True},'providers':['CPUExecutionProvider'],'intra_op_threads':2,'inter_op_threads':1,
               'actual_baseline_detection':True,'actual_specialist_models':2,
               'no_python_children_or_socket_io':True,'unit_fixture':True}
    e.write_new(folder/'resources.json',{'pipeline_binding':binding,'resources':resources})
    e.write_new(folder/'alignment-private.json',[{'unit_fixture':True}])
    (folder/'predictions.jsonl').write_text('{"unit_fixture":true}\n',encoding='utf-8')
    outputs={key:{'file':str(folder/name),'sha256':common.digest(folder/name)}
             for key,name in [('predictions','predictions.jsonl'),('resources','resources.json'),('alignment','alignment-private.json')]}
    e.write_new(report_path,{'schema_version':e.SCHEMA,'split':'dev','pipeline_binding':binding,
                            'completed_utc':common.utc(),'outputs':outputs,'resources':resources,
                            'actual_cpu_inference':True,'independent_heldout':False,'promote':False,'enabled':False,
                            'summary':{'scenes':1,'models':{identity:{'quality_gates':{'unit_fixture':True}} for identity in ('one','two')}}})
    budget=root/'budget.json';common.write(budget,{'schema_version':'localocr.gpu-budget.v1',
                         'maximum_seconds':7200,'used_seconds':20.,'active':None})
    args=argparse.Namespace(dataset=str(dataset),candidate=list(map(str,models)),baseline_models=str(baseline),
           inputs=None,config=None,dev_report=str(report_path),budget_receipt=str(budget),selected_id='one',
           reason='Unit state transition, no accuracy claim',output=str(e.EVALUATION_ROOT/'evaluation-selection.json'))
    e.freeze_selection(args)
    return dataset,binding,Path(args.output),args


def test_freeze_and_atomic_first_use_refuse_replay(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch)
    assert e.verify_selection(selection,binding)['value']['selected_id']=='one'
    output=e.EVALUATION_ROOT/'evaluation-heldout'
    first=e.claim_heldout(dataset,selection,binding,output)
    assert common.read(first['file'])['pipeline_binding']==binding
    with pytest.raises(FileExistsError):e.claim_heldout(dataset,selection,binding,output)
    assert not output.exists()
    monkeypatch.setattr(e,'owned_parent',lambda owner:True)
    e.consume_heldout(dataset,first['sha256'],binding,output/'.worker/jobs.json',output/'.worker/predictions.jsonl')
    with pytest.raises(FileExistsError):
        e.consume_heldout(dataset,first['sha256'],binding,output/'.worker/jobs.json',output/'.worker/predictions.jsonl')


def test_first_use_has_one_winner_when_callers_race(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch)
    def claim(index):
        try:
            e.claim_heldout(dataset,selection,binding,e.EVALUATION_ROOT/('evaluation-heldout-'+str(index)))
            return True
        except FileExistsError:return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(claim,(0,1)))==1


def test_selection_hashes_rules_model_and_budget_fail_closed(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch)
    changed=copy.deepcopy(binding);changed['config']['candidate']['padding']+=1
    with pytest.raises(ValueError,match='differs'):e.verify_selection(selection,changed)
    Path(args.budget_receipt).write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='evidence changed'):e.verify_selection(selection,binding)
    for budget in ({'active':{}},{'used_seconds':7201},{'used_seconds':float('nan')}):
        value={'schema_version':'localocr.gpu-budget.v1','maximum_seconds':7200,'used_seconds':1.,'active':None}|budget
        with pytest.raises(ValueError):e.validate_budget(value)


def test_heldout_refusal_precedes_image_jobs_or_truth_read(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch)
    monkeypatch.setattr(e,'image_jobs',lambda *a,**k:pytest.fail('images were read'))
    monkeypatch.setattr(e,'load_truth',lambda *a,**k:pytest.fail('truth was read'))
    args.split='heldout';args.selection=None;args.output=str(e.EVALUATION_ROOT/'evaluation-denied')
    with pytest.raises(ValueError,match='before any images or truth'):e.evaluate(args)


def test_worker_claim_binds_owner_output_slot_and_unchanged_seal(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch);output=e.EVALUATION_ROOT/'evaluation-heldout'
    first=e.claim_heldout(dataset,selection,binding,output)
    monkeypatch.setattr(e,'owned_parent',lambda owner:False)
    with pytest.raises(ValueError,match='owner'):e.consume_heldout(dataset,first['sha256'],binding,output/'.worker/jobs.json',output/'.worker/predictions.jsonl')
    monkeypatch.setattr(e,'owned_parent',lambda owner:True)
    with pytest.raises(ValueError,match='slot'):e.consume_heldout(dataset,first['sha256'],binding,output/'.worker/other.json',output/'.worker/predictions.jsonl')
    seal=common.read(dataset/'heldout-seal.json');seal['inference_results_seen']=True;common.write(dataset/'heldout-seal.json',seal)
    with pytest.raises(ValueError,match='seal'):e.consume_heldout(dataset,first['sha256'],binding,output/'.worker/jobs.json',output/'.worker/predictions.jsonl')


def test_truth_mask_tampering_rejected_after_inference(root):
    dataset,job=fixture_dataset(root/'data')
    rows=e.load_truth(dataset,'dev',[job]);assert len(e.masks_for_row(dataset,rows[0]))==1
    (dataset/'mask.png').write_bytes(b'not the immutable mask')
    with pytest.raises(ValueError,match='mask'):e.masks_for_row(dataset,rows[0])
    with pytest.raises(ValueError,match='Heldout'):e.load_truth(dataset,'heldout',[job])


def test_new_body_error_metric_uses_actual_region_not_truth_substring(root):
    dataset,job=fixture_dataset(root/'data');rows=e.load_truth(dataset,'dev',[job])
    baseline=[{'bbox':[4,5,19,19],'text':' A  ','score':.5}]
    final=[{'bbox':[4,5,19,19],'text':'WRONG','score':.99}]
    predictions=[{'id':'scene','image_sha256':job['sha256'],'pixel_sha256':job['pixel_sha256'],
      'prediction':{'baseline':baseline,'baseline_text':' A  ','routing':{'candidates':[]},
                    'models':{'one':{'regions':final,'text':'WRONG','views':{}}}}}]
    report,_=e.summarize(dataset,rows,predictions,e.configuration(),['one'])
    body=report['models']['one']['body']
    assert body['regions']==1 and body['baseline_exact']==1
    assert body['baseline_correct_to_wrong']==1 and body['erroneous_rewrites']==1
    assert body['body_intersection_transcript_changes']==1 and body['new_body_intersection_errors']==1
    assert not report['models']['one']['quality_gates']['no_new_body_errors']


def test_unknown_temporary_files_are_retained_and_process_tree_is_stopped(root):
    temporary=e.EVALUATION_ROOT/'evaluation-own'/'worker';temporary.mkdir(parents=True)
    (temporary/'jobs.json').write_bytes(b'owned');(temporary/'foreign.txt').write_bytes(b'foreign')
    e.cleanup_temporary(temporary)
    assert not (temporary/'jobs.json').exists() and (temporary/'foreign.txt').read_bytes()==b'foreign'
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    time.sleep(.15);at=time.monotonic();e.stop_process(process)
    assert process.poll() is not None and time.monotonic()-at<12


def test_audit_policy_is_worker_only_and_refuses_process_and_socket(root):
    code="""
import json,socket,subprocess,sys
from tools.RegionKaomoji import evaluate as e
sys.addaudithook(e.load_worker().deny_network)
denied=[]
for name,call in [('socket',lambda:socket.socket()),('process',lambda:subprocess.Popen([sys.executable,'-c','pass']))]:
    try:call()
    except (OSError,RuntimeError):denied.append(name)
print(json.dumps(denied))
"""
    result=subprocess.run([sys.executable,'-m','tools.RegionKaomoji.evaluate','--help'],cwd=e.REPO,capture_output=True,text=True,timeout=20)
    assert result.returncode==0
    result=subprocess.run([sys.executable,'-c',code],cwd=e.REPO,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.splitlines()[-1])==['socket','process']
    # The pytest broker retains its independent ability to launch a process.
    subprocess.run([sys.executable,'-c','pass'],check=True,timeout=20)


def test_development_prediction_resources_and_model_families_are_bound(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch)
    wrong=copy.deepcopy(binding);wrong['models'][1]['family']=wrong['models'][0]['family']
    with pytest.raises(ValueError,match='families'):e.verify_selection(selection,wrong)
    outputs=common.read(args.dev_report)['outputs']
    Path(outputs['predictions']['file']).write_bytes(b'changed local predictions')
    with pytest.raises(ValueError,match='output bytes'):e.verify_selection(selection,binding)


def test_worker_preparation_failure_cleans_owned_files_and_keeps_evidence(root,monkeypatch):
    output=e.EVALUATION_ROOT/'evaluation-prepare-failure';output.mkdir()
    real=e.write_new
    def fail_config(path,value):
        if Path(path).name=='config.json':raise OSError('unit injected preparation failure')
        return real(path,value)
    monkeypatch.setattr(e,'write_new',fail_config)
    args=argparse.Namespace(split='dev')
    with pytest.raises(OSError,match='injected'):
        e.run_worker(args,output,{'config':{}},[])
    assert not (output/'.worker').exists()
    assert common.read(output/'failure.json')['error_type']=='OSError'


def test_worker_line_read_is_bounded_ordered_and_complete(root):
    path=root/'predictions.jsonl';jobs=[{'id':'one'}]
    path.write_bytes(b'{"id":"one"}\n')
    assert e.read_predictions(path,jobs)==[{'id':'one'}]
    for raw in (b'{"id":"one"}',b'{"id":"two"}\n',b'{"id":"one"}\n{"id":"one"}\n',b'x'*(e.LIMITS['maximum_scene_output_bytes']+16385)):
        path.write_bytes(raw)
        with pytest.raises(ValueError):e.read_predictions(path,jobs)


def test_regional_splice_uses_real_roi_without_splitting_truth_and_preserves_body(root):
    masks=[np.zeros((20,40),bool),np.zeros((20,40),bool)]
    masks[0][5:10,3:8]=True;masks[1][5:10,25:30]=True
    before=[{'bbox':[1,3,35,13],'text':'ABC (Д)','score':.01}]
    after=[{'bbox':[1,3,35,13],'text':'ABC (ω)','score':.01,'source':'baseline-with-regional-splice',
            'baseline_index':0,'regional_bbox':[24,4,31,11],'regional_text':'(ω)',
            'splice':{'baseline_index':0,'original_text_span':[4,7],'preserved_prefix':'ABC ',
                      'preserved_suffix':'','replacement_text':'(ω)','ordinary_prefix_suffix_preserved':True}}]
    assert e.aligned_region(before,1,masks,e.configuration())==('',1)
    assert e.aligned_region(after,1,masks,e.configuration())==('(ω)',0)
    assert e.aligned_region(after,0,masks,e.configuration())==('',1)
    assert e.body_preserving_splices(before,after,masks[0])==(True,1)
    changed=copy.deepcopy(after);changed[0]['text']='BAD (ω)'
    assert e.body_preserving_splices(before,changed,masks[0])==(False,0)
    changed=copy.deepcopy(after);changed[0]['regional_bbox']=[2,4,31,11]
    assert e.body_preserving_splices(before,changed,masks[0])==(False,0)


class UnitDecoder:
    def __init__(self,tokens):self.character=tokens
    @staticmethod
    def get_ignored_tokens():return [0]


class UnitRecognizer:
    rec_image_shape=[3,48,320]
    rec_batch_num=6
    def __init__(self,tokens):self.postprocess_op=UnitDecoder(tokens)


def test_baseline_runtime_token_hash_preserves_actual_special_order_and_unicode(root):
    tokens=['blank','e\u0301','é',' ']
    receipt=e.baseline_runtime_receipt(UnitRecognizer(tokens))
    expected={'schema_version':'localocr.baseline-decoder-tokens.v1','tokens':tokens,'blank_id':0,'space_ids':[3]}
    assert receipt['vocabulary']['sha256']==e.hash_object(expected)
    assert receipt['vocabulary']['blank_token']=='blank' and receipt['vocabulary']['space_ids']==[3]
    e.verify_baseline_runtime(receipt)
    changed=e.baseline_runtime_receipt(UnitRecognizer(['blank','é','e\u0301',' ']))
    assert changed['vocabulary']['sha256']!=receipt['vocabulary']['sha256']
    altered=copy.deepcopy(receipt);altered['preprocessing']['configuration']['color']='RGB'
    with pytest.raises(ValueError,match='runtime receipt'):e.verify_baseline_runtime(altered)


def test_occupied_temporary_directory_cannot_be_cleaned_by_failed_claim(root):
    output=e.EVALUATION_ROOT/'evaluation-occupied';temporary=output/'.worker';temporary.mkdir(parents=True)
    (temporary/'jobs.json').write_bytes(b'foreign already owned jobs')
    (temporary/'foreign.txt').write_bytes(b'preserve me')
    (temporary/'predictions.jsonl').write_bytes(b'foreign predictions')
    (temporary/'stderr.txt').write_bytes(b'foreign diagnostics')
    with pytest.raises(FileExistsError):e.run_worker(argparse.Namespace(split='dev'),output,{'config':{}},[])
    assert (temporary/'jobs.json').read_bytes()==b'foreign already owned jobs'
    assert (temporary/'foreign.txt').read_bytes()==b'preserve me'
    assert not (output/'partial-predictions.jsonl').exists()
    assert common.read(output/'failure.json')['diagnostic']==''


def test_body_literal_deletion_cannot_be_waived_by_a_declared_splice(root):
    image=Image.new('RGB',(40,20),'white');image.save(root/'image.png')
    regions=[]
    for index,(kind,text,left,right) in enumerate([('body','LEFT o',3,8),('kaomoji','(ω)',25,30)]):
        mask=np.zeros((20,40),dtype='uint8');mask[5:10,left:right]=255
        path=root/('m'+str(index)+'.png');Image.fromarray(mask).save(path)
        regions.append({'id':str(index),'kind':kind,'text':text,'bbox':[left,5,right,10],
                        'mask':path.name,'mask_sha256':common.digest(path),'pixel_count':25})
    before=[{'bbox':[1,3,35,13],'text':'LEFT o (Д)','score':.1}]
    # Declared unsafe metadata is a metric negative control, not an accepted producer claim.
    after=[{'bbox':[1,3,35,13],'text':'LEFT (ω)','score':.1,'source':'baseline-with-regional-splice',
            'baseline_index':0,'regional_bbox':[24,4,31,11],'regional_text':'(ω)',
            'splice':{'baseline_index':0,'original_text_span':[5,10],'preserved_prefix':'LEFT ',
                     'preserved_suffix':'','replacement_text':'(ω)','ordinary_prefix_suffix_preserved':True}}]
    job={'id':'literal-control','sha256':common.digest(root/'image.png'),'pixel_sha256':e.c.image_digest(image)}
    row=dict(job,width=40,height=20,scene_kind='mixed-unit',regions=regions,text='LEFT o (ω)')
    prediction={'baseline':before,'baseline_text':'LEFT o (Д)','routing':{'candidates':[]},
                'models':{'unit':{'regions':after,'text':'LEFT (ω)','views':{}}}}
    summary,_=e.summarize(root,[row],[dict(job,image_sha256=job['sha256'],prediction=prediction)],e.configuration(),['unit'])
    assert summary['models']['unit']['body']['body_literal_deleted']==1
    assert summary['models']['unit']['body']['new_body_intersection_errors']==1
    assert not summary['models']['unit']['quality_gates']['no_new_body_errors']


def test_quality_gate_snapshot_cannot_disagree_with_development_report(root,monkeypatch):
    dataset,binding,selection,args=binding_fixture(root,monkeypatch)
    value=common.read(selection);value['quality_gate_snapshot']['one']={'fabricated_pass':True}
    selection.unlink();e.write_new(selection,value)
    with pytest.raises(ValueError,match='quality gates'):e.verify_selection(selection,binding)


def test_actual_context_producer_binds_original_parent_and_fusion_refuses_stale_record(root):
    # Geometric pixels are real; OCR strings are declared unit stubs, not model accuracy.
    image=Image.new('RGB',(360,80),'white');image.paste(geometry(),(100,0))
    draw=ImageDraw.Draw(image);draw.rectangle((15,23,55,46),fill='black');draw.rectangle((245,23,289,46),fill='black')
    baseline=[{'bbox':[10,10,291,61],'text':'LEFT  (Д_Д)  RIGHT','score':.4}]
    config=e.configuration();routing=e.c.locate_candidates(image,baseline,config['candidate'])
    candidate=next(item for item in routing['candidates'] if item['route_eligible'] and item['splice_baseline_index'] is not None)
    texts={e.c.image_digest(e.f.view_image(image,declaration['bbox'],'raw')):
           'LEFT' if declaration['role']=='left' else 'RIGHT' for declaration in candidate['context_crops']}
    class DeclaredContextBaseline:
        def predict(self,source):return baseline,{'unit_fixture':True}
        def predict_context(self,source):return texts[e.c.image_digest(source)],.2
    class DeclaredFaceModel:
        binding={'id':'unit-face'}
        def predict(self,source):return '(ω_ω)',.9
    result=e.predict_scene(image,DeclaredContextBaseline(),[DeclaredFaceModel()],config)
    records=result['contexts'][candidate['candidate_id']]
    actual_parent=e.c.baseline_regions(baseline,image.size,config['candidate']['maximum_baseline_regions'])[0]
    assert candidate['baseline_parent_sha256']==e.c.baseline_parent_digest(actual_parent)
    assert len(records)==2 and result['view_budget']['context_crops']==2
    assert {record['baseline_parent_sha256'] for record in records}=={candidate['baseline_parent_sha256']}
    assert result['models']['unit-face']['accepted_count']==1
    assert result['models']['unit-face']['regions'][0]['text']=='LEFT  (ω_ω)  RIGHT'
    for missing in (False,True):
        changed=copy.deepcopy(result['contexts'])
        for record in changed[candidate['candidate_id']]:
            if missing:record.pop('baseline_parent_sha256')
            else:record['baseline_parent_sha256']='0'*64
        fused=e.f.fuse_candidates(image,baseline,result['routing']['candidates'],result['models']['unit-face']['views'],
                                  config['fusion'],config['candidate'],contexts=changed)
        assert fused['accepted_count']==0 and fused['regions'][0]['text']==baseline[0]['text']


def test_declared_views_are_a_unique_supported_set_with_the_raw_view(root):
    config=e.configuration();config['views']=['raw','edge-pad-4','edge-pad-6'];common.write(root/'config.json',config)
    assert e.configuration(root/'config.json')['views']==['raw','edge-pad-4','edge-pad-6']
    for views in (['edge-pad-4','edge-pad-6'],['raw'],['raw','edge-pad-9'],['raw','edge-pad-6','edge-pad-6'],['raw','edge-pad-4','edge-pad-6','raw']):
        config=e.configuration();config['views']=views;common.write(root/'config.json',config)
        with pytest.raises(ValueError):e.configuration(root/'config.json')
    assert e.configuration()['views']==e.VIEWS==['raw','edge-pad-6']


def test_unsupported_cut_face_parts_are_counted_but_never_scored(root):
    """A face cut by a wrap is two `unsupported` truth parts: counted in the report, outside every scored population."""
    image=Image.new('RGB',(60,30),'white');image.save(root/'image.png')
    specs=[('body','AB',3,8,5,10,{}),
           ('unsupported','(ω',20,25,5,10,{'part':'left','source_face':'(ω_ω)','cut':2}),
           ('unsupported','_ω)',10,15,15,20,{'part':'right','source_face':'(ω_ω)','cut':2})]
    regions=[]
    for index,(kind,text,left,right,top,bottom,extra) in enumerate(specs):
        mask=np.zeros((30,60),dtype='uint8');mask[top:bottom,left:right]=255
        path=root/('u'+str(index)+'.png');Image.fromarray(mask).save(path)
        regions.append({'id':str(index),'kind':kind,'text':text,'bbox':[left,top,right,bottom],
                        'mask':path.name,'mask_sha256':common.digest(path),'pixel_count':25,**extra})
    baseline=[{'bbox':[2,4,9,11],'text':'AB','score':.5},{'bbox':[19,4,26,11],'text':'(w','score':.2},{'bbox':[9,14,16,21],'text':'_w)','score':.2}]
    candidate={'candidate_id':'candidate-000','bbox':[19,4,26,11],'sources':['image-components'],'route_eligible':False,'route_reason':'x','evidence':{}}
    job={'id':'cut-face','sha256':common.digest(root/'image.png'),'pixel_sha256':e.c.image_digest(image)}
    row=dict(job,width=60,height=30,scene_kind='card-list',regions=regions,text='AB  (ω\n_ω)')
    prediction={'baseline':baseline,'baseline_text':'AB  (w\n_w)','routing':{'candidates':[candidate],'structural_components':3},
                'models':{'unit':{'regions':baseline,'text':'AB  (w\n_w)','views':{}}}}
    summary,details=e.summarize(root,[row],[dict(job,image_sha256=job['sha256'],prediction=prediction)],e.configuration(),['unit'])
    localization=summary['localization']
    assert localization['unsupported_regions']==2 and localization['unsupported_regions_with_candidate']==1
    assert localization.get('faces',0)==0 and localization['body_regions']==1
    assert summary['models']['unit']['isolated_kaomoji']['count']==0 and summary['baseline']['isolated_kaomoji']['count']==0
    assert summary['models']['unit']['body']['regions']==1
    stage=summary['candidate_stage']
    assert stage['structural_components']==3 and stage['unsupported_regions']==2 and stage['faces']==0
    assert set(stage)>={'faces','safe_complete_coverage','routed_candidate_count','safe_routed_complete_coverage','bracket_pair_candidates','structural_components','unsupported_regions'}
    assert details[0]['regions']==[] or all(item['kind']!='unsupported' for item in details[0]['regions'])


def test_truth_loader_accepts_unsupported_parts_only_with_their_source_face(root):
    dataset,job=fixture_dataset(root/'data')
    rows=common.read(dataset/'dev.json')
    mask=Image.new('L',(80,40));ImageDraw.Draw(mask).rectangle((30,5,40,18),fill=255);mask.save(dataset/'cut.png')
    part={'id':'cut','kind':'unsupported','text':'(ω','span':[0,2],'bbox':[30,5,41,19],'mask':'cut.png',
          'mask_sha256':common.digest(dataset/'cut.png'),'pixel_count':11*14,'part':'left','source_face':'(ω_ω)','cut':2}
    rows[0]['text']='(ω  '; rows[0]['text_sha256']=sha(rows[0]['text'].encode('utf-8'))
    rows[0]['regions'][0]['text']='(ω  '
    rows[0]['regions'].append(part)
    common.write(dataset/'dev.json',rows)
    meta=common.read(dataset/'dataset.json'); meta['splits']['dev']['sha256']=common.digest(dataset/'dev.json'); common.write(dataset/'dataset.json',meta)
    frozen=common.read(dataset/'freeze.json'); frozen['dataset_sha256']=common.digest(dataset/'dataset.json'); frozen['split_sha256']['dev']=meta['splits']['dev']['sha256']; common.write(dataset/'freeze.json',frozen)
    seal=common.read(dataset/'heldout-seal.json'); seal['dataset_sha256']=frozen['dataset_sha256']; common.write(dataset/'heldout-seal.json',seal)
    loaded=e.load_truth(dataset,'dev',[job])
    assert [region['kind'] for region in loaded[0]['regions']]==['body','unsupported']
    bad=copy.deepcopy(rows); bad[0]['regions'][1].pop('source_face')
    common.write(dataset/'dev.json',bad)
    meta['splits']['dev']['sha256']=common.digest(dataset/'dev.json'); common.write(dataset/'dataset.json',meta)
    frozen['dataset_sha256']=common.digest(dataset/'dataset.json'); frozen['split_sha256']['dev']=meta['splits']['dev']['sha256']; common.write(dataset/'freeze.json',frozen)
    seal['dataset_sha256']=frozen['dataset_sha256']; common.write(dataset/'heldout-seal.json',seal)
    with pytest.raises(ValueError,match='unsupported'):
        e.load_truth(dataset,'dev',[job])


def test_zero_second_round_ledger_is_a_valid_selection_budget_receipt(root,monkeypatch):
    """A CPU-only round freezes against its own closed 0 s ledger; other rounds keep the 7200 s expectation."""
    from tools.PretrainedOcr import round_profile as rp
    profile=rp.RoundProfile(name='unit-zero',change='unit',root=root/'unit-zero',maximum_bytes=64,owned_bytes=32,
                            previous_budget=root/'old.json',previous_sha256='0'*64,previous_used_seconds=1.0,
                            maximum_seconds=0.0,specialist_families=1)
    monkeypatch.setattr(rp,'ROUNDS',(profile,))
    ledger=profile.root/'gpu-budget.json'
    zero={'schema_version':'localocr.gpu-budget.v1','maximum_seconds':0.0,'used_seconds':0.0,'runs':[],'active':None}
    common.write(ledger,zero)
    e.validate_budget(zero,ledger)
    with pytest.raises(ValueError,match='budget'):
        e.validate_budget(dict(zero,maximum_seconds=7200.0),ledger)
    with pytest.raises(ValueError,match='budget'):
        e.validate_budget(zero,root/'elsewhere'/'gpu-budget.json')  # outside any round: the shared 7200 s receipt is required
    e.validate_budget({'schema_version':'localocr.gpu-budget.v1','maximum_seconds':7200,'used_seconds':5.0,'runs':[],'active':None})
    with pytest.raises(ValueError,match='budget'):
        e.validate_budget(dict(zero,used_seconds=1.0),ledger)


def test_spliced_face_inside_a_body_box_takes_the_parent_bracket_width(root):
    # Same geometry as the context-splice test; the face box is pixel-separated inside one broad baseline box.
    image=Image.new('RGB',(360,80),'white');image.paste(geometry(),(100,0))
    draw=ImageDraw.Draw(image);draw.rectangle((15,23,55,46),fill='black');draw.rectangle((245,23,289,46),fill='black')
    baseline=[{'bbox':[10,10,291,61],'text':'LEFT  (Д_Д)  RIGHT','score':.4}]
    config=e.configuration();config['fusion']['outer_bracket_width']='baseline-else-raw'
    routing=e.c.locate_candidates(image,baseline,config['candidate'])
    candidate=next(item for item in routing['candidates'] if item['route_eligible'] and item['splice_baseline_index'] is not None)
    texts={e.c.image_digest(e.f.view_image(image,declaration['bbox'],'raw')):
           'LEFT' if declaration['role']=='left' else 'RIGHT' for declaration in candidate['context_crops']}
    class DeclaredContextBaseline:
        def predict(self,source):return baseline,{'unit_fixture':True}
        def predict_context(self,source):return texts[e.c.image_digest(source)],.2
    class FullWidthFaceModel:
        binding={'id':'unit-face'}
        def predict(self,source):return '（ω_ω）',.9     # the specialist's full-width bias on real fonts
    result=e.predict_scene(image,DeclaredContextBaseline(),[FullWidthFaceModel()],config)
    model=result['models']['unit-face']
    assert model['accepted_count']==1 and model['regions'][0]['text']=='LEFT  (ω_ω)  RIGHT'
    decision=next(d for d in model['decisions'] if d['accepted'])
    assert decision['bracket_width_source']=='baseline' and decision['specialist_text']=='(ω_ω)'
    exact=e.configuration();exact_result=e.predict_scene(image,DeclaredContextBaseline(),[FullWidthFaceModel()],exact)
    assert exact_result['models']['unit-face']['regions'][0]['text']=='LEFT  （ω_ω）  RIGHT'
