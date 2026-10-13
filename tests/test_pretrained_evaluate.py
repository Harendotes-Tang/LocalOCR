"""Actual CPU tensor tests and fail-closed research selection state transitions."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest

from tools.PretrainedOcr import evaluate as e
from tools.PretrainedOcr import common as c

# Tiny real ONNX graph, built with ONNX helper/checker: ReduceMean -> Unsqueeze
# -> zero -> constant class distribution. N and T follow input N and W.
# This tests tensor/runtime contracts only; it is not OCR accuracy evidence.
GRAPH = 'CAg6vQIKOQoFaW1hZ2USBG1lYW4iClJlZHVjZU1lYW4qDQoEYXhlc0ABQAKgAQcqDwoIa2VlcGRpbXMYAKABAgohCgRtZWFuCgRheGlzEghleHBhbmRlZCIJVW5zcXVlZXplChwKCGV4cGFuZGVkCgR6ZXJvEgV6ZXJvcyIDTXVsCiMKBXplcm9zCgZzY29yZXMSDXByb2JhYmlsaXRpZXMiA0FkZBINcmVhbC1jcHUtdGVzdCoNCAEQBzoBAkIEYXhpcyoOEAEiBAAAAABCBHplcm8qIggBCAEIBBABIhAAAAAAAACAPwAAAAAAAAAAQgZzY29yZXNaIQoFaW1hZ2USGAoWCAESEgoDEgFOCgIIAwoCCDAKAxIBV2IlCg1wcm9iYWJpbGl0aWVzEhQKEggBEg4KAxIBTgoDEgFUCgIIBEIECgAQDQ=='


def candidate(root, identity='one-after', family='one', stage='finetuned'):
    root.mkdir(parents=True)
    (root / 'model.onnx').write_bytes(base64.b64decode(GRAPH))
    (root / 'checkpoint.bin').write_bytes(b'explicit unit fixture source, not a pretrained quality claim')
    (root / 'dictionary.txt').write_text('A\ne\n', encoding='utf-8')
    c.write(root / 'vocabulary.json', {'schema_version':'localocr.pretrained.vocabulary.v1',
             'tokens':['<blank>','A','e',' '], 'blank_id':0, 'space_id':3})
    file = lambda name: {'file':name, 'sha256':c.digest(root/name)}
    meta = {'schema_version':e.CANDIDATE_SCHEMA, 'id':identity, 'family':family, 'stage':stage,
            'model':file('model.onnx'), 'vocabulary':dict(file('vocabulary.json'),blank_id=0,space_id=3),
            'preprocessing':{'height':48,'color':'RGB','width':{'mode':'dynamic','min':16,'max':1024,'multiple':4,'rounding':'ceil-multiple'},
                             'padding':'right-zero-normalized','normalize':{'mean':[.5]*3,'std':[.5]*3},'resize':'PIL-bilinear'},
            'decoder':{'type':'ctc','output':'probabilities','blank_id':0,'repeat':'ctc'},
            'source':{'checkpoint':file('checkpoint.bin'),'dictionary':file('dictionary.txt')},'promote':False,'enabled':False}
    c.write(root/'candidate.json',meta)
    return root/'candidate.json'


def dataset(root):
    root.mkdir(parents=True)
    image=Image.new('RGB',(40,20),'white'); image.save(root/'image.png')
    pixel=hashlib.sha256(image.width.to_bytes(8,'big')+image.height.to_bytes(8,'big')+image.tobytes()).hexdigest()
    row={'id':'sample','text':' A  ','text_sha256':hashlib.sha256(' A  '.encode()).hexdigest(),'image':'image.png',
         'kind':'plain','subtype':'plain','family':'test-family','group_id':'group','width':40,'height':20,
         'sha256':c.digest(root/'image.png'),'pixel_sha256':pixel,'provenance':{'source':'original','license':'CC0-1.0'}}
    splits={}
    for split in ('train','dev','heldout'):
        c.write(root/(split+'.json'),[dict(row,split=split)])
        splits[split]={'file':split+'.json','sha256':c.digest(root/(split+'.json')),'count':1}
    c.write(root/'dataset.json',{'schema_version':'localocr.pretrained.dataset.v1','synthetic_only':True,'splits':splits})
    binding={'schema_version':'localocr.pretrained.freeze.v1','dataset_sha256':c.digest(root/'dataset.json'),
             'split_sha256':{s:v['sha256'] for s,v in splits.items()}}
    c.write(root/'freeze.json',binding)
    c.write(root/'heldout-seal.json',{'schema_version':'localocr.heldout-seal.v1','dataset_sha256':binding['dataset_sha256'],
             'heldout_sha256':binding['split_sha256']['heldout'],'results_seen':False,'labels_used_for_vocabulary':False})
    return root


@pytest.fixture
def artifacts():
    with tempfile.TemporaryDirectory(prefix='evaluate-test-',dir=e.ARTIFACTS) as path:
        yield Path(path)


def test_real_dynamic_cpu_and_repeat_decoder(artifacts):
    path=candidate(artifacts/'candidate')
    rec=e.Recognizer(path)
    assert rec.session.get_providers()==['CPUExecutionProvider']
    assert rec.predict(Image.new('RGB',(12,12),'white'))==('A',1.)
    assert rec.session.run(None,{rec.input_name:np.zeros((2,3,48,77),np.float32)})[0].shape==(2,77,4)
    output=np.eye(4,dtype=np.float32)[[3,3,0,3,1,1,0,1,2]]
    assert e.decode_ctc(output,rec.tokens,rec.metadata['decoder'])==('  AAe',1.)
    decoder=dict(rec.metadata['decoder'],output='logits')
    assert e.decode_ctc(output*12,rec.tokens,decoder)[0]=='  AAe'
    with pytest.raises(ValueError,match='probabilities'):
        e.decode_ctc(output*12,rec.tokens,rec.metadata['decoder'])
    with pytest.raises(ValueError,match='finite'):
        e.decode_ctc(np.full((3,4),np.nan),rec.tokens,rec.metadata['decoder'])


def test_precise_pixel_preprocessing_padding_and_color(artifacts):
    meta=c.read(candidate(artifacts/'candidate'))['preprocessing']
    image=Image.new('RGB',(1,48),(255,0,0))
    tensor=e.prepare_image(image,meta)
    assert tensor.shape==(3,48,16)
    assert np.array_equal(tensor[:,0,0],np.array([1,-1,-1],np.float32))
    assert np.all(tensor[:,:,1:]==0)
    bgr=e.prepare_image(image,dict(meta,color='BGR',resize='cv2-linear'))
    assert np.array_equal(bgr[:,0,0],[-1,-1,1])
    assert e.prepare_image(Image.new('RGB',(10000,48)),meta).shape==(3,48,1024)
    with pytest.raises(ValueError,match='width'):
        e.prepare_image(image,dict(meta,width=dict(meta['width'],max=1023)))


def test_model_dictionary_and_source_are_actual_bound_bytes(artifacts):
    path=candidate(artifacts/'candidate')
    original=e.candidate_binding(path)
    assert original['stage']=='finetuned' and original['token_count']==4
    (path.parent/'dictionary.txt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        e.candidate_binding(path)
    path=candidate(artifacts/'another')
    metadata=c.read(path);metadata['model']['file']='../candidate/model.onnx';c.write(path,metadata)
    with pytest.raises(ValueError,match='confined'):
        e.candidate_binding(path)


def test_unicode_identity_and_emoji_coverage_not_dropped():
    rows=[{'text':' A e\u0301  ','kind':'plain','subtype':'combining','family':'f','group_id':'g'},
          {'text':'👩\u200d💻','kind':'emoji','subtype':'emoji-zwj','family':'emoji','group_id':'emoji'}]
    predictions=[{'text':'A é'}, {'text':'👩💻'}]
    result=e.summary(rows,predictions,['A',' ','e','\u0301','👩','💻'])
    assert result['overall']['exact']==0
    assert result['overall']['count']==2
    assert result['emoji']['unsupported_codepoint_occurrences']==1
    assert result['emoji']['count']==1
    assert result['combining_or_joined']['count']==2
    assert result['by_kind']['plain']['character_errors']>0


def test_baseline_correct_to_wrong_erroneous_rewrite_and_negative():
    rows=[{'text':'A','kind':'plain'},{'text':'B','kind':'plain'},{'text':'','kind':'blank'},{'text':'','kind':'graphic'}]
    before=[{'text':'A'},{'text':'X'},{'text':''},{'text':'x'}]
    after=[{'text':'a'},{'text':'Y'},{'text':'Q'},{'text':''}]
    moves=e.transitions(rows,before,after)
    assert moves['by_kind']['plain']['baseline_correct_to_wrong']==1
    assert moves['by_kind']['plain']['erroneous_rewrites']==2
    assert moves['overall']['new_negative_false_outputs']==1
    assert moves['overall']['negative_false_outputs']==1


def test_private_ambiguous_excluded_from_exact_and_transitions():
    rows=[{'text':'A','kind':'plain','subtype':'manual','family':'p','group_id':'1'},
          {'text':'not-known','kind':'face','subtype':'manual','family':'f','group_id':'2','ambiguous':True}]
    before=[{'text':'A'},{'text':'not-known'}]; after=[{'text':'a'},{'text':'?'}]
    summary=e.summary(rows,after,['A','a',' '])
    assert summary['total_images']==2 and summary['ambiguous_excluded']==1
    assert summary['overall']['count']==1
    assert 'face' not in summary['by_kind']
    moves=e.transitions(rows,before,after)
    assert moves['overall']['count']==1 and moves['ambiguous_excluded']==1


def test_worker_heldout_cannot_bypass_selection(artifacts,monkeypatch):
    root=dataset(artifacts/'data')
    def never(*args,**kwargs):
        pytest.fail('heldout reader called before receipt validation')
    monkeypatch.setattr(e,'read_split',never)
    args=argparse.Namespace(dataset=root,split='heldout',private_manifest=None,heldout_usage_sha256=None,candidate='missing',baseline_models=None,output=artifacts/'never.json')
    with pytest.raises(ValueError,match='first-use'):
        e.worker(args)


def test_actual_candidate_subprocess_only_dev(artifacts):
    root=dataset(artifacts/'data'); path=candidate(artifacts/'candidate')
    # Deleting the heldout file proves the dev subprocess has no label dependency.
    (root/'heldout.json').unlink()
    result=e.run_recognizer(root,'dev',artifacts/'predictions.json',candidate=path)
    assert result['predictions']==[{'id':'sample','text':'A','score':1.}]
    resources=result['resources']
    assert resources['providers']==['CPUExecutionProvider']
    assert resources['intra_op_threads']==2 and resources['warm']['count']==1
    assert resources['cold_load_and_first_crop_seconds']>0 and resources['peak_rss_bytes']>0
    if e.sys.platform=='win32':
        assert resources['peak_private_bytes']>0 and resources['peak_commit_bytes']>0


def selection_fixture(artifacts,monkeypatch):
    root=dataset(artifacts/'data')
    paths=[candidate(artifacts/name,name,family,stage) for family in ('one','two')
           for name,stage in ((family+'-before','official-pretrained'),(family+'-after','finetuned'))]
    bindings=[e.candidate_binding(p) for p in paths]
    baseline={'id':'production-baseline','sha256':'unit-metadata-fixture'}
    monkeypatch.setattr(e,'baseline_binding',lambda path:baseline)
    # Real tiny ONNX candidate inference; the baseline substitution is explicitly
    # a unit fixture, never research baseline or quality evidence.
    rows=c.read_split(root,'dev'); frozen=c.dataset_binding(root)
    models={}; workers={}; predictions=None
    for index,(identity,binding) in enumerate([('production-baseline',baseline)]+[(b['id'],b) for b in bindings]):
        path=artifacts/f'worker-{index}.json'
        captured=e.run_recognizer(root,'dev',path,candidate=paths[max(0,index-1)])['worker_file']
        result=c.read(path)
        if index==0:
            result['inference_binding']=baseline;c.write(path,result)
        workers[identity]={**captured,'sha256':c.digest(path)}
        value={'summary':e.summary(rows,result['predictions'],result['tokens']),'resources':result['resources']}
        if predictions is None: predictions=result['predictions']
        else:value.update(transitions_from_production=e.transitions(rows,predictions,result['predictions']),stage=binding['stage'])
        models[identity]=value
    e.add_comparisons_and_gates(models)
    report={'schema_version':e.REPORT_SCHEMA,'split':'dev','independent_heldout':False,'promote':False,'enabled':False,
            'candidate_bindings':bindings,'baseline_binding':baseline,'dataset_binding':frozen,
            'models':models,'worker_results':workers,'source_inventory':e.evaluator_source_inventory()}
    report_path=artifacts/'unit-dev-metadata-report.json';c.write(report_path,report)
    budget_path=artifacts/'budget.json';c.write(budget_path,{'schema_version':'localocr.gpu-budget.v1','maximum_seconds':7200,'used_seconds':123,'active':None})
    args=argparse.Namespace(dataset=root,candidate=paths,dev_report=report_path,budget_receipt=budget_path,
                baseline_models='metadata-fixture',selected_id='one-after',reason='unit receipt hash/state validation; not inference evidence',output=artifacts/'selection.json')
    return root,paths,bindings,baseline,args


def test_selection_first_use_claim_survives_failure_and_rejects_reuse(artifacts,monkeypatch):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    receipt=e.freeze_selection(args)
    assert receipt['promote'] is False and receipt['usage']=='research-only-selected-for-evaluation'
    # No heldout labels need to be readable when the exclusive claim is created.
    (root/'heldout.json').unlink()
    claim=e.selection_for_heldout(root,args.output,bindings,baseline)
    assert c.digest(root/'heldout-first-use.json')==claim['sha256']
    e.validate_worker_heldout(root,claim['sha256'],paths[0],None)
    with pytest.raises(FileExistsError):
        e.selection_for_heldout(root,args.output,bindings,baseline)
    with pytest.raises(ValueError,match='digest'):
        e.validate_worker_heldout(root,'0'*64,paths[0],None)


@pytest.mark.parametrize('mutation',['model','dev','budget','seal','selection-time'])
def test_frozen_selection_mutation_denied_before_heldout(artifacts,monkeypatch,mutation):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    e.freeze_selection(args)
    if mutation=='model':
        (paths[0].parent/'model.onnx').write_bytes(b'mutated')
        with pytest.raises(ValueError,match='checksum'):
            e.candidate_binding(paths[0])
        return
    if mutation=='dev':
        c.write(args.dev_report,{'tampered':True})
    elif mutation=='budget':
        c.write(args.budget_receipt,{'tampered':True})
    elif mutation=='seal':
        seal=c.read(root/'heldout-seal.json');seal['results_seen']=True;c.write(root/'heldout-seal.json',seal)
    else:
        value=c.read(args.output);value['frozen_utc']='2999-01-01T00:00:00+00:00';c.write(args.output,value)
    with pytest.raises(ValueError):
        e.selection_for_heldout(root,args.output,bindings,baseline)
    assert not (root/'heldout-first-use.json').exists()


def test_freeze_rejects_unfinished_budget_and_missing_before_after(artifacts,monkeypatch):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    state=c.read(args.budget_receipt);state['active']={'pid':1};c.write(args.budget_receipt,state)
    with pytest.raises(ValueError,match='cumulative budget'):
        e.freeze_selection(args)
    state['active']=None;state['used_seconds']=7201;c.write(args.budget_receipt,state)
    with pytest.raises(ValueError,match='cumulative budget'):
        e.freeze_selection(args)
    state['used_seconds']=1;c.write(args.budget_receipt,state)
    args.candidate=paths[:2]
    report=c.read(args.dev_report);report['candidate_bindings']=bindings[:2];c.write(args.dev_report,report)
    with pytest.raises(ValueError,match='required family'):
        e.freeze_selection(args)



def test_reserved_baseline_id_and_no_output_overwrite(artifacts):
    path=candidate(artifacts/'candidate',identity='production-baseline')
    with pytest.raises(ValueError,match='identity'):
        e.candidate_binding(path)
    output=artifacts/'owned.json'
    e.write_new(output,{'first':True})
    before=output.read_bytes()
    with pytest.raises(FileExistsError):
        e.write_new(output,{'first':False})
    assert output.read_bytes()==before
    with pytest.raises(ValueError,match='artifacts'):
        e.write_new(e.REPOSITORY/'must-not-create.json',{})


def test_actual_heldout_candidate_worker_has_single_use_slot(artifacts,monkeypatch):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    e.freeze_selection(args)
    out=artifacts/'controlled-heldout'
    claim=e.selection_for_heldout(root,args.output,bindings,baseline,output_dir=out)
    output=out/'candidate-0-predictions.json'
    result=e.run_recognizer(root,'heldout',output,candidate=paths[0],first_use=claim)
    assert result['predictions'][0]['text']=='A'
    consumed=root/'heldout-first-use-workers'/'1.json'
    assert consumed.is_file()
    # Even if the owned output is removed, this model's claim cannot be reused.
    output.unlink()
    with pytest.raises(RuntimeError,match='CPU recognizer failed'):
        e.run_recognizer(root,'heldout',output,candidate=paths[0],first_use=claim)
    assert not output.exists()
    assert c.read(consumed)['usage_sha256']==claim['sha256']


def test_heldout_child_failures_still_consume_slot(artifacts,monkeypatch):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    e.freeze_selection(args)
    out=artifacts/'controlled-heldout'
    claim=e.selection_for_heldout(root,args.output,bindings,baseline,output_dir=out)
    (root/'heldout.json').write_text('changed',encoding='utf-8')
    with pytest.raises(RuntimeError,match='CPU recognizer failed'):
        e.run_recognizer(root,'heldout',out/'candidate-0-predictions.json',candidate=paths[0],first_use=claim)
    assert (root/'heldout-first-use-workers'/'1.json').is_file()
    assert not (out/'candidate-0-predictions.json').exists()



def test_seal_changed_after_global_claim_blocks_worker_before_label_read(artifacts,monkeypatch):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    e.freeze_selection(args)
    out=artifacts/'controlled-heldout'
    claim=e.selection_for_heldout(root,args.output,bindings,baseline,output_dir=out)
    seal=c.read(root/'heldout-seal.json');seal['results_seen']=True;c.write(root/'heldout-seal.json',seal)
    with pytest.raises(RuntimeError,match='CPU recognizer failed'):
        e.run_recognizer(root,'heldout',out/'candidate-0-predictions.json',candidate=paths[0],first_use=claim)
    assert not (root/'heldout-first-use-workers'/'1.json').exists()
    assert not (out/'candidate-0-predictions.json').exists()


def test_development_evidence_rejects_metadata_and_bound_worker_drift(artifacts,monkeypatch):
    import copy
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    good=c.read(args.dev_report)
    e.validate_dev_evidence(good,args.dev_report,root,bindings,baseline)
    for field in ('models','worker_results','source_inventory'):
        altered=copy.deepcopy(good);altered.pop(field)
        with pytest.raises(ValueError):e.validate_dev_evidence(altered,args.dev_report,root,bindings,baseline)
    altered=copy.deepcopy(good);altered['models']['one-after']['summary']['overall']['exact']+=1
    with pytest.raises(ValueError,match='summaries/resources'):e.validate_dev_evidence(altered,args.dev_report,root,bindings,baseline)
    worker=artifacts/good['worker_results']['one-after']['file']; original=c.read(worker)
    for change in ('cpu','source','threads','count','nan','predictions','binding','dataset'):
        value=copy.deepcopy(original)
        if change=='cpu':value['actual_cpu']=False
        elif change=='source':value['source_inventory']=[]
        elif change=='threads':value['resources']['intra_op_threads']=True
        elif change=='count':value['resources']['warm']['count']=True
        elif change=='nan':value['resources']['session_load_seconds']=float('nan')
        elif change=='predictions':value['predictions'][0]['text']='tampered'
        elif change=='binding':value['inference_binding']={}
        elif change=='dataset':value['dataset_binding']={}
        # Re-signing just the prediction file cannot make its metrics/input
        # bindings agree with the captured report.
        worker.write_text(json.dumps(value),encoding='utf-8')
        altered=copy.deepcopy(good);altered['worker_results']['one-after']['sha256']=c.digest(worker)
        with pytest.raises(ValueError):e.validate_dev_evidence(altered,args.dev_report,root,bindings,baseline)
    c.write(worker,original)
    c.write(args.dev_report,good)
    for used in (True,False,float('nan'),float('inf'),-1,7201):
        budget=c.read(args.budget_receipt);budget['used_seconds']=used
        args.budget_receipt.write_text(json.dumps(budget),encoding='utf-8')
        with pytest.raises(ValueError,match='cumulative budget'):e.freeze_selection(args)


def test_worker_report_checksum_and_current_source_drift_rejected(artifacts,monkeypatch):
    root,paths,bindings,baseline,args=selection_fixture(artifacts,monkeypatch)
    report=c.read(args.dev_report)
    worker=artifacts/report['worker_results']['one-after']['file']
    worker.write_bytes(worker.read_bytes()+b' ')
    with pytest.raises(ValueError,match='checksum'):e.validate_dev_evidence(report,args.dev_report,root,bindings,baseline)
    monkeypatch.setattr(e,'evaluator_source_inventory',lambda:[])
    with pytest.raises(ValueError,match='evidence'):e.validate_dev_evidence(report,args.dev_report,root,bindings,baseline)
