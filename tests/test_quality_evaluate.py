"""Pure quality/freezing negative controls, never inference-success evidence."""
import copy
from pathlib import Path
import os
from types import SimpleNamespace
import json
import pytest
from tools.PretrainedOcr import quality_evaluate as q

def test_write_quota_counts_nested_files_reserved_bytes_and_fresh_growth(tmp_path,monkeypatch):
 monkeypatch.setattr(q,'ROUND',tmp_path)
 monkeypatch.setattr(q,'ROUND_MAX_BYTES',16)
 monkeypatch.setattr(q.common,'disk_guard',lambda *args:None)
 (tmp_path/'models').mkdir();(tmp_path/'data').mkdir()
 (tmp_path/'models/weight.bin').write_bytes(b'x'*9)
 (tmp_path/'data/image.bin').write_bytes(b'x'*2)
 q.round_write_guard(tmp_path/'result.json',5)
 with pytest.raises(ValueError,match='quota'):q.round_write_guard(tmp_path/'result.json',6)
 (tmp_path/'data/new.bin').write_bytes(b'x'*5)
 with pytest.raises(ValueError,match='quota'):q.round_write_guard(tmp_path/'result.json',1)

def test_write_quota_reserves_actual_output_disk_space(tmp_path,monkeypatch):
 monkeypatch.setattr(q,'ROUND',tmp_path)
 monkeypatch.setattr(q.common.shutil,'disk_usage',lambda _:SimpleNamespace(free=q.common.MINIMUM_FREE_BYTES+2))
 q.round_write_guard(tmp_path/'result.json',2)
 with pytest.raises(RuntimeError,match='disk reserve'):q.round_write_guard(tmp_path/'result.json',3)

@pytest.mark.parametrize('reserved',[True,-1,1.0,None,float('nan'),float('inf')])
def test_write_quota_rejects_invalid_reserved_bytes(tmp_path,monkeypatch,reserved):
 monkeypatch.setattr(q,'ROUND',tmp_path)
 with pytest.raises(ValueError,match='nonnegative integer'):q.round_write_guard(tmp_path/'result.json',reserved)

def test_write_quota_rejects_directory_links_instead_of_traversing(tmp_path,monkeypatch):
 root=tmp_path/'round';root.mkdir();target=tmp_path/'outside';target.mkdir()
 monkeypatch.setattr(q,'ROUND',root)
 monkeypatch.setattr(q.common,'disk_guard',lambda *args:None)
 try:os.symlink(target,root/'linked',target_is_directory=True)
 except OSError:pytest.skip('Directory symlink creation unavailable on this host')
 with pytest.raises(ValueError,match='reparse'):q.round_write_guard(root/'result.json',0)

@pytest.mark.parametrize('kind',['crop','scene'])
def test_json_writer_reserves_exact_final_utf8_bytes(tmp_path,monkeypatch,kind):
 from tools.PretrainedOcr import evaluate as crop
 from tools.RegionKaomoji import evaluate as scene_evaluate
 module=crop if kind=='crop' else scene_evaluate
 monkeypatch.setattr(module,'local_output' if kind=='crop' else 'output_path',lambda p:Path(p))
 seen=[];monkeypatch.setattr(module,'round_write_guard',lambda p,n:seen.append(n))
 value={'regions':[{'text':'中文（e\u0301）','box':[1,2,3,4]} for _ in range(10)]}
 path=tmp_path/(kind+'.json');module.write_new(path,value)
 actual=path.read_bytes()
 assert seen==[len(actual)] and json.loads(actual)==value
 assert b'\r\n' not in actual and actual.endswith(b'\n')

@pytest.mark.parametrize('kind',['crop','scene'])
def test_json_writer_rejects_pretty_expansion_before_creating_file(tmp_path,monkeypatch,kind):
 from tools.PretrainedOcr import evaluate as crop
 from tools.RegionKaomoji import evaluate as scene_evaluate
 module=crop if kind=='crop' else scene_evaluate
 monkeypatch.setattr(module,'local_output' if kind=='crop' else 'output_path',lambda p:Path(p))
 value={'regions':[{'text':'中文','box':[1,2,3,4]} for _ in range(10)]}
 compact=len(json.dumps(value,ensure_ascii=False).encode('utf-8'))
 def guard(path,reserved):
  if reserved>compact+1:raise ValueError('quota')
 monkeypatch.setattr(module,'round_write_guard',guard)
 path=tmp_path/(kind+'.json')
 with pytest.raises(ValueError,match='quota'):module.write_new(path,value)
 assert not path.exists()

def metric(exact=0,cer=1.,ps=1.):return {'count':2,'exact':exact,'cer':cer,'ps_cer':ps}
def model():
 kinds={k:metric() for k in ('face','plain','punctuation','blank','graphic')};kinds['face']=metric(1,.4,.3)
 moves={k:{'baseline_correct_to_wrong':0,'erroneous_rewrites':0,'new_negative_false_outputs':0,'negative_false_outputs':0} for k in kinds}
 return {'summary':{'by_kind':kinds},'transitions_from_production':{'by_kind':moves}}
def baseline():return {'summary':{'by_kind':{'face':metric()}}}
def scene(routes=1,effective=1):
 return {'baseline':{'isolated_kaomoji':metric()},'models':{'chosen':{'isolated_kaomoji':metric(1,.4,.3),'body':{'regions':3,'baseline_correct_to_wrong':0,'erroneous_rewrites':0,'new_body_intersection_errors':0,'body_literal_deleted':0},'negative':{'negative_scenes':2,'graphic_regions':1,'new_nonempty_scenes':0,'new_graphic_nonempty':0,'baseline_graphic_nonempty':1,'final_graphic_nonempty':1},'routing_effectiveness':{'actual_routed_candidates':routes,'face_wrong_to_exact':effective}}}}

def test_ps_excludes_combining_marks_without_unicode_normalization():
 result=q.ps_metrics([{'label':'(e\u0301)','prediction':'(e)'}]);assert result['ps_errors']==0
 from tools.PretrainedOcr.evaluate import summary
 # Legacy exact/CER still observes the lost original combining mark.
 value=q.exact_metrics([{'label':'(e\u0301)','prediction':'(e)'}]);assert value['exact']==0 and value['character_errors']==1

def test_three_independent_face_improvements_required():
 assert q.face_gate(metric(),metric(1,.5,.5))['passed']
 for field in ('exact','cer','ps_cer'):
  proposed=metric(1,.5,.5);proposed[field]=metric()[field]
  assert not q.face_gate(metric(),proposed)['passed']

def test_raw_generic_failure_does_not_waive_or_confuse_routed_specialist():
 value=model();value['transitions_from_production']['by_kind']['plain']['erroneous_rewrites']=1
 result=q.raw_crop_gate(baseline(),value)
 assert result['specialist_face_eligible'] and not result['raw_generic_replacement_passed']
 assert q.scene_gate(scene(),'chosen')['passed']
 assert result['production_enabled'] is False

def test_missing_protection_fields_fail_closed():
 value=model();del value['transitions_from_production']['by_kind']['punctuation']['erroneous_rewrites']
 with pytest.raises(ValueError,match='counter'):q.raw_crop_gate(baseline(),value)
 value=scene();del value['models']['chosen']['negative']['new_graphic_nonempty']
 with pytest.raises(ValueError,match='counter'):q.scene_gate(value,'chosen')

@pytest.mark.parametrize('routes,effective',[(0,1),(1,0),(0,0)])
def test_no_routes_or_no_effective_change_is_failure(routes,effective):
 assert not q.scene_gate(scene(routes,effective),'chosen')['passed']

@pytest.mark.parametrize('section,key',[('body','baseline_correct_to_wrong'),('body','erroneous_rewrites'),('body','body_literal_deleted'),('body','new_body_intersection_errors'),('negative','new_nonempty_scenes'),('negative','new_graphic_nonempty')])
def test_each_final_protection_error_fails(section,key):
 value=scene();value['models']['chosen'][section][key]=1
 assert not q.scene_gate(value,'chosen')['passed']

def test_baseline_existing_graphic_nonempty_is_not_a_new_error():
 result=q.scene_gate(scene(),'chosen');assert result['passed'] and result['negative_new_nonempty']['new_graphic_nonempty']==0

def test_invalid_numbers_and_unequal_face_population_rejected():
 for invalid in (None,float('nan'),float('inf'),-1,True):
  value=metric(1,.3,.3);value['ps_cer']=invalid
  with pytest.raises(ValueError):q.face_gate(metric(),value)
 value=metric(1,.3,.3);value['count']=3
 with pytest.raises(ValueError,match='populations'):q.face_gate(metric(),value)

def test_new_crop_heldout_rejected_before_dataset_read(monkeypatch,tmp_path):
 from tools.PretrainedOcr import evaluate
 monkeypatch.setattr(evaluate,'is_round',lambda _:True)
 monkeypatch.setattr(evaluate,'dataset_binding',lambda _:pytest.fail('must not read dataset'))
 with pytest.raises(ValueError,match='dual freeze'):evaluate.selection_for_heldout(tmp_path,tmp_path/'selection',[],{})

def test_new_scene_heldout_rejected_before_truth_or_selection(monkeypatch,tmp_path):
 from tools.RegionKaomoji import evaluate
 monkeypatch.setattr(evaluate,'is_round',lambda _:True)
 monkeypatch.setattr(evaluate,'verify_selection',lambda *_:pytest.fail('must not read scene selection'))
 with pytest.raises(ValueError,match='dual freeze'):evaluate.claim_heldout(tmp_path,tmp_path/'selection',{},tmp_path/'out')

def test_dual_source_drift_rejected_before_development_replay(monkeypatch,tmp_path):
 from tools.PretrainedOcr import common
 p=tmp_path/'dual.json';common.write(p,{'schema_version':q.DUAL_SCHEMA,'production_enabled':False,'promote':False,'enabled':False,'source_inventory':{'old':'hash'}})
 monkeypatch.setattr(q,'is_round',lambda _:True)
 monkeypatch.setattr(q,'source_inventory',lambda:{'new':'hash'})
 monkeypatch.setattr(q,'recompute_development',lambda *_:pytest.fail('must not replay dev after source drift'))
 with pytest.raises(ValueError,match='source closure'):q.verify_dual(p,'crop',tmp_path,tmp_path)

def test_new_round_budget_rejects_child_or_old_ledger(tmp_path):
 from tools.PretrainedOcr import common
 p=tmp_path/'child-budget.json';common.write(p,{'schema_version':'localocr.gpu-budget.v1','active':None,'maximum_seconds':7200,'used_seconds':0})
 with pytest.raises(ValueError,match='sole-authority'):q.validate_round_budget(p)

def test_quality_evaluation_namespace_remains_scoped():
 from tools.RegionKaomoji import evaluate
 good=q.ROUND/'evaluation-new'/'evaluation.json';assert evaluate.output_path(good)==good.resolve()
 with pytest.raises(ValueError):evaluate.output_path(q.ROUND/'models'/'evaluation.json')


def test_actual_cpu_boolean_without_owned_successful_worker_not_proof():
    with pytest.raises(ValueError,match='owned real CPU'):q.validate_owned_cpu_capture({'actual_cpu':True,'providers':['CPUExecutionProvider'],'intra_op_threads':2,'inter_op_threads':1})
    valid={'providers':['CPUExecutionProvider'],'intra_op_threads':2,'inter_op_threads':1,'worker_pid':123,'worker_exit_code':0,'capture':'successful-owned-subprocess'}
    for key,value in [('worker_pid',True),('worker_exit_code',1),('providers',['CUDAExecutionProvider']),('capture','mock')]:
        changed=dict(valid);changed[key]=value
        with pytest.raises(ValueError):q.validate_owned_cpu_capture(changed)


def test_failed_development_cannot_write_dual_or_consume_heldout(monkeypatch,tmp_path):
    monkeypatch.setattr(q,'ROUND',tmp_path)
    monkeypatch.setattr(q,'is_round',lambda _:True)
    def reject(*_):raise ValueError('Development research quality failed')
    monkeypatch.setattr(q,'recompute_development',reject)
    output=tmp_path/'evaluation'/'dual.json'
    with pytest.raises(ValueError,match='quality failed'):
        q.freeze_dual(tmp_path/'crop-selection',tmp_path/'scene-selection',tmp_path/'crop',tmp_path/'scene',output)
    assert not output.exists() and not (tmp_path/'crop/heldout-first-use.json').exists()


def test_both_selections_must_precede_dual_freeze(monkeypatch,tmp_path):
    from tools.PretrainedOcr import common
    from datetime import datetime,timezone,timedelta
    now=datetime.now(timezone.utc);later=now-timedelta(seconds=1);earlier=now-timedelta(seconds=2)
    selections=[]
    for kind in ('crop','scene'):
        p=tmp_path/(kind+'.json');common.write(p,{'frozen_utc':later.isoformat()});selections.append(q.reference(p))
    value={'schema_version':q.DUAL_SCHEMA,'frozen_utc':earlier.isoformat(),'source_inventory':{},'production_enabled':False,'promote':False,'enabled':False,'crop_selection':selections[0],'scene_selection':selections[1]}
    p=tmp_path/'dual.json';common.write(p,value)
    monkeypatch.setattr(q,'is_round',lambda _:True);monkeypatch.setattr(q,'source_inventory',lambda:{})
    monkeypatch.setattr(q,'recompute_development',lambda *_:pytest.fail('Late selection must stop before replay'))
    with pytest.raises(ValueError,match='precede'):q.verify_dual(p,'crop',tmp_path/'crop.json',tmp_path)


def test_scene_copy_reserves_whole_file_before_any_output(monkeypatch,tmp_path):
    from tools.RegionKaomoji import evaluate as scene
    source=tmp_path/'source';source.write_bytes(b'owned predictions')
    target=tmp_path/'destination'
    monkeypatch.setattr(scene,'output_path',lambda p:Path(p))
    seen=[]
    def reject(path,additional):seen.append((path,additional));raise ValueError('quota')
    monkeypatch.setattr(scene,'round_write_guard',reject)
    with pytest.raises(ValueError,match='quota'):scene.copy_new(source,target)
    assert seen==[(target,source.stat().st_size)] and not target.exists()


def test_effective_face_change_requires_this_face_bytes_not_other_scene_change():
    import numpy as np
    from tools.RegionKaomoji.evaluate import effective_face_change
    mask=np.zeros((10,30),dtype=bool);mask[2:8,2:10]=True
    decision={'accepted':True,'bbox':[1,1,11,9],'original_text':'body (x) suffix','specialist_text':'(x)','splice':{'original_face_text':'(x)'}}
    model={'text':'different scene text','decisions':[decision]}
    assert not effective_face_change(model,'before',mask)  # Identical face, other text changed.
    decision['splice']['original_face_text']='(y)';assert effective_face_change(model,'before',mask)
    decision['bbox']=[20,1,29,9];assert not effective_face_change(model,'before',mask)


def test_real_serviceenv_venv_launcher_owns_cpu_probe():
    import os,subprocess,sys,json
    from tools.RegionKaomoji.evaluate import verify_worker_proof
    command=[sys.executable,'-B','-m','tools.RegionKaomoji.evaluate','_ownership-probe','--owner-pid',str(os.getpid())]
    # Windows launcher cmdline has -B before -m; use standard research worker prefix.
    command.remove('-B')
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    out,err=process.communicate(timeout=15)
    assert process.returncode==0,err
    record=json.loads(out);proof=verify_worker_proof(record['proof'],process.pid,os.getpid())
    assert record['model_inference'] is False and record['GPU_used'] is False
    if sys.platform=='win32':assert proof['relation'] in ('direct-child','windows-venv-redirector')


def test_real_serviceenv_probe_rejects_fabricated_owner():
    import os,subprocess,sys
    process=subprocess.run([sys.executable,'-m','tools.RegionKaomoji.evaluate','_ownership-probe','--owner-pid',str(os.getpid()+100000000)],capture_output=True,text=True,encoding='utf-8',timeout=15)
    assert process.returncode!=0 and 'not owned' in process.stderr


def test_fabricated_redirector_chain_cannot_pass_parent_validation(monkeypatch):
    from tools.RegionKaomoji import evaluate as scene
    monkeypatch.setattr(scene.sys,'platform','win32')
    proof={'owner_pid':10,'worker_pid':30,'worker_parent_pid':20,'launcher_pid':20,'relation':'windows-venv-redirector','owner_chain_verified':True}
    assert scene.verify_worker_proof(proof,20,10)==proof
    for key,value in [('owner_pid',11),('launcher_pid',21),('worker_parent_pid',10),('worker_pid',20),('worker_pid',True),('owner_chain_verified',False)]:
        altered=dict(proof);altered[key]=value
        with pytest.raises(ValueError):scene.verify_worker_proof(altered,20,10)
