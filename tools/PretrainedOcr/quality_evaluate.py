"""Fail-closed synthetic research quality and paired pre-heldout freeze.

Pure gate helpers are not evidence of inference. Freeze replays bound dev worker
outputs; existing evaluator owner/worker receipts retain their unique-use slots.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import unicodedata
from . import common

REPO=Path(__file__).resolve().parents[2]
ROUND=REPO/'artifacts/quality-optimization-20261012'
ROUND_MAX_BYTES=1024**3
OLD_BUDGET_SHA='6d4aa4827bf41f84f99a49610a17611dd07185351c6f7a749da8881d250b8497'
DUAL_SCHEMA='localocr.quality-dual-freeze.v1'

def is_round(path):return Path(path).resolve().is_relative_to(ROUND.resolve())

def round_write_guard(path,additional_bytes=0):
    if not is_round(path):return
    if type(additional_bytes) is not int or additional_bytes<0:
        raise ValueError('Reserved output bytes must be a nonnegative integer')
    common.disk_guard(path,common.MINIMUM_FREE_BYTES+additional_bytes)
    from .round_limits import _no_reparse
    _no_reparse(ROUND)
    size=0;pending=[ROUND] if ROUND.exists() else []
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                    raise ValueError('Research resource path contains a reparse point')
                if stat.S_ISDIR(info.st_mode):pending.append(entry.path)
                elif stat.S_ISREG(info.st_mode):size+=info.st_size
    _no_reparse(ROUND)
    if size+additional_bytes>ROUND_MAX_BYTES:raise ValueError('Research round exceeds1GiB output quota')

def ps_metrics(rows,field='prediction'):
    from tools.OcrDatasets.evaluate import distance
    take=lambda value:''.join(c for c in value if unicodedata.category(c)[0] in 'PS')
    chars=sum(len(take(row['label'])) for row in rows)
    errors=sum(distance(take(row['label']),take(row[field])) for row in rows)
    return {'ps_characters':chars,'ps_errors':errors,'ps_cer':errors/chars if chars else None}

def exact_metrics(rows,field='prediction'):
    from tools.OcrDatasets.evaluate import metrics
    return {**metrics(rows,field),**ps_metrics(rows,field)}

def count(value,name):
    result=value.get(name)
    if type(result) is not int or result<0:raise ValueError('Missing/noninteger quality counter:'+name)
    return result

def face_gate(baseline,candidate):
    for record in (baseline,candidate):
        if count(record,'count')<=0:raise ValueError('Face quality needs nonempty measured population')
        count(record,'exact')
        for field in ('cer','ps_cer'):
            if type(record.get(field)) not in (int,float) or not math.isfinite(record[field]) or record[field]<0:raise ValueError('Missing/nonfinite face '+field)
        if record['exact']>record['count']:raise ValueError('Invalid face exact count')
    if baseline['count']!=candidate['count']:raise ValueError('Paired face populations differ')
    changes={'exact_gain':candidate['exact']-baseline['exact'],'cer_reduction':baseline['cer']-candidate['cer'],'ps_cer_reduction':baseline['ps_cer']-candidate['ps_cer']}
    return {**changes,'passed':all(value>0 for value in changes.values())}

def raw_crop_gate(baseline,model):
    kinds=model['summary']['by_kind'];moves=model['transitions_from_production']['by_kind']
    faces=face_gate(baseline['summary']['by_kind']['face'],kinds['face'])
    ordinary={};negative={}
    for kind in ('plain','punctuation'):
        if count(kinds[kind],'count')<=0:raise ValueError('Raw generic protection population missing')
        ordinary[kind]={'correct_to_wrong':count(moves[kind],'baseline_correct_to_wrong'),'erroneous_rewrites':count(moves[kind],'erroneous_rewrites')}
    for kind in ('blank','graphic'):
        if count(kinds[kind],'count')<=0:raise ValueError('Raw negative population missing')
        negative[kind]={'new_nonempty':count(moves[kind],'new_negative_false_outputs'),'total_nonempty':count(moves[kind],'negative_false_outputs')}
    protected=all(not any(v.values()) for v in ordinary.values()) and all(v['new_nonempty']==0 for v in negative.values())
    return {'face':faces,'raw_generic_protection':ordinary,'raw_negative':negative,'raw_generic_replacement_passed':faces['passed'] and protected,'specialist_face_eligible':faces['passed'],'production_enabled':False,'promote':False}

def scene_gate(summary,identity):
    model=summary['models'][identity]
    faces=face_gate(summary['baseline']['isolated_kaomoji'],model['isolated_kaomoji'])
    body=model['body'];negative=model['negative'];routing=model['routing_effectiveness']
    if count(body,'regions')<=0 or count(negative,'negative_scenes')<=0 or count(negative,'graphic_regions')<=0:raise ValueError('Final protection population missing')
    protection={name:count(body,name) for name in ('baseline_correct_to_wrong','erroneous_rewrites','new_body_intersection_errors','body_literal_deleted')}
    negatives={name:count(negative,name) for name in ('new_nonempty_scenes','new_graphic_nonempty')}
    routes=count(routing,'actual_routed_candidates');effective=count(routing,'face_wrong_to_exact')
    passed=faces['passed'] and not any(protection.values()) and not any(negatives.values()) and routes>0 and effective>0
    return {'face':faces,'body_and_punctuation_new_errors':protection,'negative_new_nonempty':negatives,'actual_routed_candidates':routes,'effective_face_improvements':effective,'passed':passed,'research_quality_passed':passed,'production_enabled':False,'promote':False,'app_acceptance':False}

def validate_owned_cpu_capture(actual):
    if (actual.get('providers')!=['CPUExecutionProvider'] or actual.get('intra_op_threads')!=2 or actual.get('inter_op_threads')!=1 or type(actual.get('worker_pid')) is not int or actual['worker_pid']<=0 or actual.get('worker_exit_code')!=0 or actual.get('capture')!='successful-owned-subprocess'):
        raise ValueError('Requires successful owned real CPU worker evidence')


def source_inventory():
    names=sorted({p.relative_to(REPO).as_posix() for folder in ('tools','scripts','localocr') for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts})
    return {name:common.digest(REPO/name) for name in names}

def timestamp(text):
    value=datetime.fromisoformat(text)
    if value.tzinfo is None or value>datetime.now(timezone.utc):raise ValueError('Invalid elapsed freeze timestamp')
    return value

def reference(path):return {'file':str(Path(path).resolve()),'sha256':common.digest(path)}

def bound(record):
    p=Path(record['file']).resolve()
    if not is_round(p) or common.digest(p)!=record['sha256']:raise ValueError('Round evidence bytes/path differ')
    return p

def validate_round_budget(path):
    path=Path(path).resolve();value=common.read(path);binding=value.get('round_binding',{})
    if path!=ROUND/'gpu-budget.json' or value.get('schema_version')!='localocr.gpu-budget.v1' or value.get('active') is not None or value.get('maximum_seconds')!=7200 or path.with_name(path.name+'.lock').exists():raise ValueError('Requires closed sole-authority new-round ledger')
    used=value.get('used_seconds')
    if type(used) not in (int,float) or not math.isfinite(used) or not 0<=used<=7200:raise ValueError('Invalid new budget usage')
    if binding.get('schema_version')!='localocr.gpu-round-binding.v1' or binding.get('previous_sha256')!=OLD_BUDGET_SHA or not math.isclose(binding.get('previous_used_seconds',-1),7011.043364,abs_tol=.000001) or Path(binding.get('round_root','')).resolve()!=ROUND:raise ValueError('New ledger must bind approved old closed history')
    if common.digest(binding['previous_budget'])!=OLD_BUDGET_SHA:raise ValueError('Historical ledger changed')
    old=common.read(binding['previous_budget'])
    if old.get('active') is not None or Path(binding['previous_budget']).with_name(Path(binding['previous_budget']).name+'.lock').exists():raise ValueError('Historical ledger is not closed')
    runs=value.get('runs')
    if not isinstance(runs,list) or any(not isinstance(run,dict) or type(run.get('elapsed_seconds')) not in (int,float) or not math.isfinite(run['elapsed_seconds']) or run['elapsed_seconds']<0 for run in runs):raise ValueError('Invalid measured GPU intervals')
    if not math.isclose(sum(run['elapsed_seconds'] for run in runs),used,abs_tol=.000001):raise ValueError('New usage intervals differ')
    if binding.get('maximum_artifact_bytes')!=1024**3 or binding.get('minimum_free_bytes',0)<8*1024**3:raise ValueError('Round resource limits differ')
    round_write_guard(path)
    return reference(path)


def recompute_development(crop_selection,scene_selection,crop_dataset,scene_dataset):
    from . import evaluate as crop
    from tools.RegionKaomoji import evaluate as scene
    cs=common.read(crop_selection);ss=common.read(scene_selection)
    if cs['schema_version']!=crop.SELECTION_SCHEMA or ss['schema_version']!=scene.SELECTION_SCHEMA:raise ValueError('Two real frozen selections required')
    timestamp(cs['frozen_utc']);timestamp(ss['frozen_utc'])
    for value in (cs,ss):
        budget=bound(value['budget_receipt']);validate_round_budget(budget)
    if cs['promote'] is not False or ss['promote'] is not False or ss['enabled'] is not False:raise ValueError('Research selections must remain disabled')
    if cs['dataset_binding']!=common.dataset_binding(crop_dataset):raise ValueError('Crop dataset changed')
    if any(crop.candidate_binding(item['candidate_file'])!=item for item in cs['candidate_bindings']):raise ValueError('Crop model/source changed')
    candidates=[item['candidate_file'] for item in ss['pipeline_binding']['models']]
    baseline_root=Path(ss['pipeline_binding']['baseline']['manifest_file']).parent
    pipeline=scene.pipeline_binding(scene_dataset,candidates,baseline_root,ss['pipeline_binding']['config'],ss['pipeline_binding']['inputs']['file'])
    scene.verify_selection(scene_selection,pipeline)
    cdev_path=bound(cs['dev_report']);sdev_path=bound(ss['dev_report']);cdev=common.read(cdev_path)
    models=crop.validate_dev_evidence(cdev,cdev_path,crop_dataset,cs['candidate_bindings'],cs['baseline_binding'])
    sdev=scene.verify_dev_evidence(sdev_path,pipeline)
    jobs=scene.image_jobs(scene_dataset,'dev',pipeline['inputs']['file'])
    rows=scene.load_truth(scene_dataset,'dev',jobs)
    predictions=scene.read_predictions(sdev['outputs']['predictions']['file'],jobs)
    aggregate,_=scene.summarize(scene_dataset,rows,predictions,pipeline['config'],[m['id'] for m in pipeline['models']])
    if aggregate!=sdev['summary']:raise ValueError('Scene aggregate differs from actual worker output')
    if cs['selected_id']!=ss['selected_id']:raise ValueError('Both freezes must select the same specialist identity')
    identity=ss['selected_id']
    cg=raw_crop_gate(models['production-baseline'],models[identity]);sg=scene_gate(aggregate,identity)
    if not cg['specialist_face_eligible'] or not sg['passed']:raise ValueError('Development research quality failed; new heldout remains unconsumed')
    if not set(item['candidate_sha256'] for item in pipeline['models']).issubset({item['candidate_sha256'] for item in cs['candidate_bindings']}):raise ValueError('Scene specialists absent from crop freeze')
    return {'selected_id':identity,'crop':cg,'scene':sg}

def freeze_dual(crop_selection,scene_selection,crop_dataset,scene_dataset,output):
    selections=[Path(crop_selection),Path(scene_selection)];datasets=[Path(crop_dataset),Path(scene_dataset)]
    if not all(is_round(p) for p in selections+datasets+[Path(output)]):raise ValueError('Dual selection must belong to the new research round')
    # Refuse manufacturing a new freeze after either crop or scene first-use exists.
    for dataset in datasets:
        if (dataset/'heldout-first-use.json').exists():raise ValueError('New heldout already consumed')
    if any(ROUND.glob('evaluation-heldout-first-use-*.json')):raise ValueError('Scene heldout already consumed')
    gates=recompute_development(*selections,*datasets)
    value={'schema_version':DUAL_SCHEMA,'frozen_utc':common.utc(),'crop_selection':reference(selections[0]),'scene_selection':reference(selections[1]),'crop_dataset':{'path':str(datasets[0].resolve()),'binding':common.dataset_binding(datasets[0])},'scene_dataset':{'path':str(datasets[1].resolve()),'binding':common.read(datasets[1]/'freeze.json')},'source_inventory':source_inventory(),'gates':gates,'production_enabled':False,'promote':False,'enabled':False,'usage':'both selections frozen before either first new heldout; no tuning on results'}
    from .evaluate import write_new
    write_new(output,value);return value

def verify_dual(path,kind,selection,dataset):
    value=common.read(bound(reference(path)))
    if value.get('schema_version')!=DUAL_SCHEMA or value.get('production_enabled') is not False or value.get('promote') is not False or value.get('enabled') is not False or value.get('source_inventory')!=source_inventory():raise ValueError('Dual freeze source closure changed')
    when=timestamp(value['frozen_utc'])
    for k in ('crop','scene'):
        sp=bound(value[k+'_selection']);selected=common.read(sp)
        if timestamp(selected['frozen_utc'])>when:raise ValueError('Both selections must precede dual freeze')
        dp=Path(value[k+'_dataset']['path']);frozen=common.read(dp/'freeze.json')
        if frozen!=value[k+'_dataset']['binding'] or common.digest(dp/'dataset.json')!=frozen['dataset_sha256']:raise ValueError('Either dataset changed before first-use')
    if Path(selection).resolve()!=Path(value[kind+'_selection']['file']).resolve() or Path(dataset).resolve()!=Path(value[kind+'_dataset']['path']).resolve():raise ValueError('Wrong dual-freeze selection/data slot')
    gates=recompute_development(value['crop_selection']['file'],value['scene_selection']['file'],value['crop_dataset']['path'],value['scene_dataset']['path'])
    if gates!=value['gates']:raise ValueError('Development gate proof changed')
    return reference(path)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--crop-selection',required=True);p.add_argument('--scene-selection',required=True);p.add_argument('--crop-dataset',required=True);p.add_argument('--scene-dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    value=freeze_dual(a.crop_selection,a.scene_selection,a.crop_dataset,a.scene_dataset,a.output)
    print(json.dumps({'schema_version':value['schema_version'],'selected_id':value['gates']['selected_id'],'research_quality_passed':True,'production_enabled':False}))
if __name__=='__main__':main()
