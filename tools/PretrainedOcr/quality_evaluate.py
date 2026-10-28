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
from . import round_profile as rp

REPO=Path(__file__).resolve().parents[2]
ROUND=REPO/'artifacts/quality-optimization-20261012'
ROUND_MAX_BYTES=1024**3
OLD_BUDGET_SHA='6d4aa4827bf41f84f99a49610a17611dd07185351c6f7a749da8881d250b8497'
DUAL_SCHEMA='localocr.quality-dual-freeze.v1'

def round_scope(path):
    """Return (round root, byte quota) for a registered round, the legacy module round, or (None, None)."""
    profile=rp.profile_for(path)
    if profile is not None:return profile.root,profile.maximum_bytes
    if Path(path).resolve().is_relative_to(ROUND.resolve()):return ROUND,ROUND_MAX_BYTES
    return None,None

def is_round(path):return round_scope(path)[0] is not None

def round_ledger_contract(path):
    """Immutable predecessor and quota expectations for the round owning ``path``."""
    profile=rp.profile_for(path)
    if profile is not None:
        return {'root':Path(profile.root),'previous_sha256':profile.previous_sha256,'previous_used_seconds':profile.previous_used_seconds,
                'maximum_bytes':profile.maximum_bytes,'maximum_seconds':profile.maximum_seconds}
    return {'root':ROUND,'previous_sha256':OLD_BUDGET_SHA,'previous_used_seconds':7011.043364,'maximum_bytes':1024**3,'maximum_seconds':7200}

def round_write_guard(path,additional_bytes=0):
    root,limit=round_scope(path)
    if root is None:return
    if type(additional_bytes) is not int or additional_bytes<0:
        raise ValueError('Reserved output bytes must be a nonnegative integer')
    common.disk_guard(path,common.MINIMUM_FREE_BYTES+additional_bytes)
    from .round_limits import _no_reparse
    _no_reparse(root)
    size=0;pending=[root] if root.exists() else []
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                    raise ValueError('Research resource path contains a reparse point')
                if stat.S_ISDIR(info.st_mode):pending.append(entry.path)
                elif stat.S_ISREG(info.st_mode):size+=info.st_size
    _no_reparse(root)
    if size+additional_bytes>limit:raise ValueError('Research round exceeds its authorized output quota')

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
    # With tail-ambiguity pairs the gate reads the resolved counts; strict fields stay reported alongside.
    model=summary['models'][identity];tail=summary.get('tail_ambiguity')
    resolved=tail['models'][identity] if tail else None
    faces=face_gate(tail['baseline']['isolated_kaomoji'] if tail else summary['baseline']['isolated_kaomoji'],
                    resolved['isolated_kaomoji'] if tail else model['isolated_kaomoji'])
    body=resolved['body'] if tail else model['body'];negative=model['negative'];routing=model['routing_effectiveness']
    if count(body,'regions')<=0 or count(negative,'negative_scenes')<=0 or count(negative,'graphic_regions')<=0:raise ValueError('Final protection population missing')
    protection={name:count(body,name) for name in ('baseline_correct_to_wrong','erroneous_rewrites','new_body_intersection_errors','body_literal_deleted')}
    negatives={name:count(negative,name) for name in ('new_nonempty_scenes','new_graphic_nonempty')}
    routes=count(routing,'actual_routed_candidates');effective=count(resolved,'face_wrong_to_exact') if tail else count(routing,'face_wrong_to_exact')
    passed=faces['passed'] and not any(protection.values()) and not any(negatives.values()) and routes>0 and effective>0
    gate={'face':faces,'body_and_punctuation_new_errors':protection,'negative_new_nonempty':negatives,'actual_routed_candidates':routes,'effective_face_improvements':effective,'passed':passed,'research_quality_passed':passed,'production_enabled':False,'promote':False,'app_acceptance':False}
    if tail:gate['counting']={'mode':'tail-ambiguity-resolved','pairs':tail['pairs'],'resolved_pairs':resolved['resolved_pairs']}
    return gate

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
    contract=round_ledger_contract(path);root=Path(contract['root']).resolve();maximum=contract['maximum_seconds']
    if path!=root/'gpu-budget.json' or value.get('schema_version')!='localocr.gpu-budget.v1' or value.get('active') is not None or value.get('maximum_seconds')!=maximum or path.with_name(path.name+'.lock').exists():raise ValueError('Requires closed sole-authority new-round ledger')
    used=value.get('used_seconds')
    if type(used) not in (int,float) or not math.isfinite(used) or not 0<=used<=maximum:raise ValueError('Invalid new budget usage')
    if binding.get('schema_version')!='localocr.gpu-round-binding.v1' or binding.get('previous_sha256')!=contract['previous_sha256'] or not math.isclose(binding.get('previous_used_seconds',-1),contract['previous_used_seconds'],abs_tol=.000001) or Path(binding.get('round_root','')).resolve()!=root:raise ValueError('New ledger must bind approved old closed history')
    if common.digest(binding['previous_budget'])!=contract['previous_sha256']:raise ValueError('Historical ledger changed')
    old=common.read(binding['previous_budget'])
    if old.get('active') is not None or Path(binding['previous_budget']).with_name(Path(binding['previous_budget']).name+'.lock').exists():raise ValueError('Historical ledger is not closed')
    runs=value.get('runs')
    if not isinstance(runs,list) or any(not isinstance(run,dict) or type(run.get('elapsed_seconds')) not in (int,float) or not math.isfinite(run['elapsed_seconds']) or run['elapsed_seconds']<0 for run in runs):raise ValueError('Invalid measured GPU intervals')
    if not math.isclose(sum(run['elapsed_seconds'] for run in runs),used,abs_tol=.000001):raise ValueError('New usage intervals differ')
    if binding.get('maximum_artifact_bytes')!=contract['maximum_bytes'] or binding.get('minimum_free_bytes',0)<8*1024**3:raise ValueError('Round resource limits differ')
    round_write_guard(path)
    return reference(path)


def _pixel_round_slots(*paths):
    """Confine every slot when any part of a paired operation is the new round."""
    profiles = [rp.profile_for(path) if path is not None else None for path in paths]
    if rp.PIXEL_OWNED_SPANS not in profiles:
        return False
    if any(profile is not rp.PIXEL_OWNED_SPANS for profile in profiles):
        raise ValueError('Pixel-owned selections, datasets and dual output must share the new round')
    return True


def _position_round_slots(*paths):
    """The position round owns both fresh datasets and every selection slot."""
    profiles = [rp.profile_for(path) if path is not None else None for path in paths]
    if rp.OWNED_POSITION not in profiles:
        return False
    if any(profile is not rp.OWNED_POSITION for profile in profiles):
        raise ValueError('Position selections, datasets and output must share the new round')
    return True


def _position_crop_models(selection):
    """Require current-source four-model crop development, never an old freeze."""
    frozen = common.read(rp.OWNED_POSITION.root / 'validation/source-inventory-before-dev.json')
    models = frozen.get('models', {})
    expected = models.get('candidates', [])
    if (frozen.get('schema_version') != 'localocr.position-before-dev-freeze.v1'
            or frozen.get('source_inventory') != source_inventory()
            or len(expected) != 3 or len({item['id'] for item in expected}) != 3
            or selection['candidate_bindings'] != expected
            or selection['baseline_binding'] != models.get('baseline')):
        raise ValueError('Position crop development requires the current frozen four-model set')


def _first_pass_round_slots(*paths):
    """The first-pass round owns both fresh datasets and every selection slot."""
    profiles = [rp.profile_for(path) if path is not None else None for path in paths]
    if rp.FIRST_PASS_POSITION not in profiles:
        return False
    if any(profile is not rp.FIRST_PASS_POSITION for profile in profiles):
        raise ValueError('First-pass selections, datasets and output must share the new round')
    return True


def _first_pass_crop_models(selection):
    """Require current-source four-model crop development, never an old freeze."""
    frozen = common.read(rp.FIRST_PASS_POSITION.root / 'validation/source-inventory-before-dev.json')
    models = frozen.get('models', {})
    expected = models.get('candidates', [])
    if (frozen.get('schema_version') != 'localocr.first-pass-before-dev-freeze.v1'
            or frozen.get('source_inventory') != source_inventory()
            or len(expected) != 3 or len({item['id'] for item in expected}) != 3
            or selection['candidate_bindings'] != expected
            or selection['baseline_binding'] != models.get('baseline')):
        raise ValueError('First-pass crop development requires the current frozen four-model set')


def _pixel_crop_models(selection):
    """Require the new frozen three-candidate plus production-baseline crop set."""
    frozen = common.read(rp.PIXEL_OWNED_SPANS.root / 'validation/source-inventory-before-dev.json')
    models = frozen.get('models', {})
    expected = models.get('candidates', [])
    if (frozen.get('schema_version') != 'localocr.pixel-owned-before-dev-freeze.v1'
            or frozen.get('source_inventory') != source_inventory()
            or len(expected) != 3 or len({item['id'] for item in expected}) != 3
            or selection['candidate_bindings'] != expected
            or selection['baseline_binding'] != models.get('baseline')):
        raise ValueError('Pixel-owned crop development requires the current frozen four-model set')


def predecessor_crop_receipt(crop_selection,crop_dataset,scene_dataset):
    """The predecessor round's consumed crop first-use receipt bound to ``crop_selection``, or None.

    Only a CPU-only scene round (0 s GPU) that reuses the frozen specialist of a different registered round may
    build on a consumed crop heldout; the receipt must name this selection, its bytes and the crop dataset.
    """
    if _first_pass_round_slots(crop_selection, crop_dataset, scene_dataset):
        return None
    if _position_round_slots(crop_selection, crop_dataset, scene_dataset):
        return None
    if _pixel_round_slots(crop_selection, crop_dataset, scene_dataset):
        return None  # This round must generate/evaluate both fresh sides itself.
    crop_round,scene_round=rp.profile_for(crop_dataset),rp.profile_for(scene_dataset)
    receipt_path=Path(crop_dataset)/'heldout-first-use.json'
    if (not receipt_path.exists() or crop_round is None or scene_round is None or crop_round.name==scene_round.name
            or scene_round.maximum_seconds!=0 or not Path(crop_selection).resolve().is_relative_to(Path(crop_round.root).resolve())):
        return None
    receipt=common.read(receipt_path)
    if (receipt.get('schema_version')!='localocr.pretrained.heldout-first-use.v1'
            or Path(receipt.get('selection_file','')).resolve()!=Path(crop_selection).resolve()
            or receipt.get('selection_sha256')!=common.digest(crop_selection)
            or receipt.get('dataset_binding')!=common.dataset_binding(crop_dataset)):
        raise ValueError('Predecessor crop first use does not bind this crop selection and dataset')
    return {'round':crop_round.name,'heldout_first_use':reference(receipt_path)}

def recompute_development(crop_selection,scene_selection,crop_dataset,scene_dataset):
    from . import evaluate as crop
    from tools.RegionKaomoji import evaluate as scene
    first_pass_owned = _first_pass_round_slots(crop_selection, scene_selection, crop_dataset, scene_dataset)
    position_owned = _position_round_slots(crop_selection, scene_selection, crop_dataset, scene_dataset)
    pixel_owned = _pixel_round_slots(crop_selection, scene_selection, crop_dataset, scene_dataset)
    predecessor=predecessor_crop_receipt(crop_selection,crop_dataset,scene_dataset)
    cs=common.read(crop_selection);ss=common.read(scene_selection)
    if pixel_owned:
        _pixel_crop_models(cs)
    if position_owned:
        _position_crop_models(cs)
        from tools.RegionKaomoji.position_data import verify_fresh_data
        fresh_data_proof = verify_fresh_data(crop_dataset, scene_dataset)
    if first_pass_owned:
        _first_pass_crop_models(cs)
        from tools.RegionKaomoji.first_pass_data import verify_fresh_data
        fresh_data_proof = verify_fresh_data(crop_dataset, scene_dataset)
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
    models=crop.validate_dev_evidence(cdev,cdev_path,crop_dataset,cs['candidate_bindings'],cs['baseline_binding'],
                                      source_inventory=cdev.get('source_inventory') if predecessor is not None else None)
    sdev=scene.verify_dev_evidence(sdev_path,pipeline)
    jobs=scene.image_jobs(scene_dataset,'dev',pipeline['inputs']['file'])
    rows=scene.load_truth(scene_dataset,'dev',jobs)
    predictions=scene.read_predictions(sdev['outputs']['predictions']['file'],jobs)
    aggregate,_=scene.summarize(scene_dataset,rows,predictions,pipeline['config'],[m['id'] for m in pipeline['models']],tail=scene.tail_counting(sdev_path))
    if aggregate!=sdev['summary']:raise ValueError('Scene aggregate differs from actual worker output')
    if cs['selected_id']!=ss['selected_id']:raise ValueError('Both freezes must select the same specialist identity')
    identity=ss['selected_id']
    cg=raw_crop_gate(models['production-baseline'],models[identity]);sg=scene_gate(aggregate,identity)
    if not cg['specialist_face_eligible'] or not sg['passed']:raise ValueError('Development research quality failed; new heldout remains unconsumed')
    if not set(item['candidate_sha256'] for item in pipeline['models']).issubset({item['candidate_sha256'] for item in cs['candidate_bindings']}):raise ValueError('Scene specialists absent from crop freeze')
    result = {'selected_id':identity,'crop':cg,'scene':sg}
    if position_owned or first_pass_owned:
        result['fresh_data_proof'] = fresh_data_proof
    return result

# Rounds whose dual freeze must carry regression proofs, pinned by identity: scene development datasets by
# freeze dataset_sha256 and known-input probe manifests by sha256. Each pinned item needs exactly one proof.
FIXED_REGRESSIONS={
    'scene-dev':('81862891f4989f3b3d1bb87f46b2b1d22422e00c01b58c8cceafb2efd7a2cb5a',   # cards-v4
                 'ee53228e1ed1610a33b7dd2cdb4512f79c11f681aa317de342fb4f0b2bcda70c'),  # atoms-mixed-v3
    'known-input-probe':('9cc6f325259e69d4ce140cd3266ba80919ceb043346397993b299bafcad8d804',)}  # known-input-001
REQUIRED_REGRESSIONS={'browser-font-20261016':FIXED_REGRESSIONS,'bracket-pairs-20261016':FIXED_REGRESSIONS,
                     'candidate-recovery-20261018': {
                         **FIXED_REGRESSIONS,
                         'scene-dev': (*FIXED_REGRESSIONS['scene-dev'],
                                       '9576a24fbc9ee177828c500ae43e8519cc11d4d9e787773942f8c7b06a8dd5a7')}}
REQUIRED_REGRESSIONS['complete-spans-20261019'] = {
    **FIXED_REGRESSIONS,
    'scene-dev': (*FIXED_REGRESSIONS['scene-dev'],
                  '9576a24fbc9ee177828c500ae43e8519cc11d4d9e787773942f8c7b06a8dd5a7',
                  'e118e408529880f39f8e1cf0414045258343b1c64ebf8e82a9a7ffb57bc24613')}
REQUIRED_REGRESSIONS['pixel-owned-spans-20261020'] = {
    **FIXED_REGRESSIONS,
    'scene-dev': (*REQUIRED_REGRESSIONS['complete-spans-20261019']['scene-dev'],
                  '85d62779cbac00d24465b2f12635a53779d1f2932110657392b83bddd130d17e')}
REQUIRED_REGRESSIONS['ownership-position-20261022'] = {
    **FIXED_REGRESSIONS,
    'scene-dev': (*REQUIRED_REGRESSIONS['pixel-owned-spans-20261020']['scene-dev'],
                  'c7a72ed3305b0359681e276bbd24076aca5f1aaf61221aecf15c6611bdfaf6f8')}

REQUIRED_REGRESSIONS['first-pass-position-20261027'] = {
    **FIXED_REGRESSIONS,
    'scene-dev': (*REQUIRED_REGRESSIONS['ownership-position-20261022']['scene-dev'],
                  '2dd2f0e826899d52da0308176a80d48066b62a4fe495d9d25da541e9f9239b6d')}

def _same_binding(models,config_sha256,classifier,baseline,selection):
    pb=selection['pipeline_binding']
    return (sorted(m['candidate_sha256'] for m in models)==sorted(m['candidate_sha256'] for m in pb['models'])
            and config_sha256==pb['config_sha256'] and (classifier or {}).get('sha256')==(pb['config'].get('classifier') or {}).get('sha256')
            and baseline.get('manifest_sha256')==pb['baseline'].get('manifest_sha256'))

def _scene_regression_summary(path,value):
    """Re-verify a scene development report against its bound worker outputs and recompute its summary."""
    from tools.RegionKaomoji import evaluate as scene
    binding=value['pipeline_binding'];root=Path(binding['inputs']['file']).parent
    pipeline=scene.pipeline_binding(root,[m['candidate_file'] for m in binding['models']],Path(binding['baseline']['manifest_file']).parent,binding['config'],binding['inputs']['file'])
    dev=scene.verify_dev_evidence(path,pipeline)
    jobs=scene.image_jobs(root,'dev',pipeline['inputs']['file'])
    predictions=scene.read_predictions(dev['outputs']['predictions']['file'],jobs)
    aggregate,_=scene.summarize(root,scene.load_truth(root,'dev',jobs),predictions,pipeline['config'],[m['id'] for m in pipeline['models']],tail=scene.tail_counting(path))
    if aggregate!=dev['summary']:raise ValueError('Regression aggregate differs from actual worker output')
    return aggregate

def regression_proof(path,scene_selection):
    """Hard regression conditions of the selected configuration on one development report: zero new body
    errors (tail-ambiguity resolved counts when present, zero lost body lines for a known-input probe) and
    zero new negative outputs over a non-empty population. The report must bind the same specialists,
    classifier, config and baseline, and must not be the selection's own development dataset."""
    p=bound(reference(path));value=common.read(p);selection=common.read(scene_selection);identity=selection['selected_id']
    if value.get('promote') is not False or value.get('enabled') is not False:raise ValueError('Regression reports must stay disabled')
    if value.get('schema_version')=='localocr.known-input-evaluation.v1':
        if value.get('independent') is not False:raise ValueError('Known-input regression must be a development probe')
        if not _same_binding(value['models'],value['config_sha256'],value.get('classifier'),value['baseline'],selection):raise ValueError('Regression report binds a different model or configuration')
        # The probe summary is recomputed from the bound per-screen predictions and screenshots, never trusted.
        # real_screens imports this module at top level, so it is imported here.
        from tools.RegionKaomoji import real_screens
        summary=real_screens.recompute_probe(p)
        model=summary['models'][identity];lost=count(model,'body_lost');lines=count(summary,'body_lines')
        return {'report':reference(p),'kind':'known-input-probe','manifest_sha256':value['manifest']['sha256'],
                'predictions_private_sha256':value.get('predictions_private_sha256'),'body_lines':lines,'body_lost':lost,
                'passed':lines>0 and lost==0 and model['gates']['no_body_lost'] is True}
    binding=value.get('pipeline_binding') or {}
    if value.get('split')!='dev' or not _same_binding(binding.get('models',[]),binding.get('config_sha256'),binding.get('config',{}).get('classifier'),binding.get('baseline',{}),selection):
        raise ValueError('Regression report binds a different model, configuration or split')
    if binding['dataset']['dataset_sha256']==selection['pipeline_binding']['dataset']['dataset_sha256']:
        raise ValueError("The selection's own development dataset is not a regression report")
    summary=_scene_regression_summary(p,value);model=summary['models'][identity];tail=summary.get('tail_ambiguity')
    body=tail['models'][identity]['body'] if tail else model['body'];negative=model['negative']
    errors={name:count(body,name) for name in ('baseline_correct_to_wrong','erroneous_rewrites','new_body_intersection_errors','body_literal_deleted')}
    negatives={name:count(negative,name) for name in ('new_nonempty_scenes','new_graphic_nonempty')}
    regions=count(model['body'],'regions')
    return {'report':reference(p),'kind':'scene-dev','dataset_sha256':binding['dataset']['dataset_sha256'],'counting':'tail-ambiguity-resolved' if tail else 'strict',
            'body_regions':regions,'body_new_errors':errors,'negative_new_nonempty':negatives,
            'passed':regions>0 and not any(errors.values()) and not any(negatives.values())}

def regression_proofs(reports,scene_selection,output):
    proofs=[regression_proof(path,scene_selection) for path in reports]
    profile=rp.profile_for(output)
    required=REQUIRED_REGRESSIONS.get(profile.name if profile else None,{})
    for kind,pinned in required.items():
        field='dataset_sha256' if kind=='scene-dev' else 'manifest_sha256'
        if sorted(item[field] for item in proofs if item['kind']==kind)!=sorted(pinned):
            raise ValueError('This round requires exactly its pinned regression reports in the dual freeze')
    if not all(item['passed'] for item in proofs):raise ValueError('Regression hard condition failed; new heldout remains unconsumed')
    return proofs

def _complete_span_proof(reports, scene_selection, root):
    """The additional approved gates cannot be replaced by relative production gain."""
    if rp.profile_for(root) is not rp.COMPLETE_SPANS:
        if reports:
            raise ValueError('Complete-span development reports belong only to their registered round')
        return None
    if len(reports) != 4:
        raise ValueError('Complete-span paired freeze requires four strict development reports')
    from tools.RegionKaomoji import span_quality
    selection = common.read(scene_selection)
    proof = span_quality.verify_round_development(*reports, selection['selected_id'], round_root=rp.COMPLETE_SPANS.root)
    if proof['reports']['final_new']['report'] != selection['dev_report']:
        raise ValueError('Strict final development does not bind the actual scene selection')
    if not proof['passed']:
        raise ValueError('Complete-span development thresholds failed; heldouts remain sealed')
    return proof


def _pixel_joint_proof(inputs, scene_selection, root):
    """Recompute the new round's complete four-tier/repeated-latency evidence.

    The request lists immutable original report references, never caller verdicts.
    Its read-only verifier rejects quality failure before a dual freeze is written.
    Other rounds cannot use this entry to replace their original requirements.
    """
    profile = rp.profile_for(root)
    if profile is not rp.PIXEL_OWNED_SPANS:
        if inputs is not None:
            raise ValueError('Joint development inputs belong only to their registered pixel-owned round')
        return None
    if inputs is None:
        raise ValueError('Pixel-owned paired freeze requires joint development and repeated latency inputs')
    request_ref = reference(inputs)
    request_path = bound(request_ref)
    if rp.profile_for(request_path) is not profile:
        raise ValueError('Joint development request must belong to this round')
    request = common.read(request_path)
    if request.get('schema_version') != 'localocr.pixel-owned-development-inputs.v1':
        raise ValueError('Invalid joint development request schema')
    selection_ref = reference(scene_selection)
    if rp.profile_for(selection_ref['file']) is not profile:
        raise ValueError('Joint scene selection must stay in its registered round')
    selection = common.read(scene_selection)
    if request.get('selected_id') != selection['selected_id']:
        raise ValueError('Joint development model differs from scene selection')

    captured_refs = [dict(selection_ref)]

    def owned_ref(record):
        path = bound(record)
        if rp.profile_for(path) is not profile:
            raise ValueError('Joint development evidence must stay in its registered round')
        captured_refs.append(dict(record))
        return path

    manifest = owned_ref(request['configuration_manifest'])
    source_freeze = owned_ref(request['source_freeze'])
    expected_freeze = Path(profile.root).resolve() / 'validation/source-inventory-before-dev.json'
    if source_freeze != expected_freeze:
        raise ValueError('Joint development requires the original before-dev source freeze')
    frozen = common.read(source_freeze)
    manifest_value = common.read(manifest)
    if (frozen.get('schema_version') != 'localocr.pixel-owned-before-dev-freeze.v1'
            or frozen.get('source_inventory') != source_inventory()
            or request['configuration_manifest'] not in frozen.get('references', [])
            or timestamp(frozen['frozen_utc']) < timestamp(manifest_value['frozen_utc'])):
        raise ValueError('Joint source/configuration manifest differs from before-dev freeze')
    scored = {tier: {population: owned_ref(record) for population, record in pair.items()}
              for tier, pair in request['scored_reports'].items()}
    repeated = {tier: {population: [owned_ref(record) for record in records]
                       for population, records in pair.items()}
                for tier, pair in request['repeated_reports'].items()}
    for pair in scored.values():
        for path in pair.values():
            if timestamp(common.read(path)['completed_utc']) <= timestamp(frozen['frozen_utc']):
                raise ValueError('Scored report must follow the original before-dev freeze')
    from tools.RegionKaomoji import joint_quality
    proof = joint_quality.verify_round_development(
        scored, repeated, selection['selected_id'], round_root=profile.root,
        configuration_manifest=request['configuration_manifest'])
    if reference(request_path) != request_ref:
        raise ValueError('Joint development inputs changed during verification')
    if reference(source_freeze) != request['source_freeze']:
        raise ValueError('Before-dev source freeze changed during joint verification')
    for record in captured_refs:
        if reference(record['file']) != record:
            raise ValueError('Joint bound selection or original report changed during verification')
    chosen = proof['selected_tier']
    if not proof['passed'] or chosen == 'start':
        raise ValueError('Joint quality or repeated latency failed; heldouts remain sealed')
    if proof['reports'][chosen]['fresh']['report'] != selection['dev_report']:
        raise ValueError('Joint selected development does not bind the actual scene selection')
    return {'inputs': request_ref, 'proof': proof}


def _position_development_proof(inputs, scene_selection, root):
    """Recompute all five-tier/repeat originals before position first-use.

    This path cannot reuse predecessor crop, source or quality approval. The
    request holds only immutable original file references; counts and verdicts
    supplied by a caller have no authority.
    """
    profile = rp.profile_for(root)
    if profile is not rp.OWNED_POSITION:
        if inputs is not None:
            raise ValueError('Position development inputs belong only to the position round')
        return None
    if inputs is None:
        raise ValueError('Position paired freeze requires five-tier and repeated latency inputs')
    request_ref = reference(inputs)
    request_path = bound(request_ref)
    if rp.profile_for(request_path) is not profile:
        raise ValueError('Position request must belong to its round')
    request = common.read(request_path)
    if request.get('schema_version') != 'localocr.position-development-inputs.v1':
        raise ValueError('Invalid position development request schema')
    selection_ref = reference(scene_selection)
    if rp.profile_for(selection_ref['file']) is not profile:
        raise ValueError('Position scene selection must belong to its round')
    selection = common.read(scene_selection)
    if request.get('selected_id') != selection['selected_id']:
        raise ValueError('Position development model differs from selection')
    captured = [selection_ref]

    def owned(record):
        path = bound(record)
        if rp.profile_for(path) is not profile:
            raise ValueError('Position evidence must stay in its registered round')
        captured.append(dict(record))
        return path

    manifest = owned(request['configuration_manifest'])
    source_freeze = owned(request['source_freeze'])
    if source_freeze != Path(profile.root).resolve() / 'validation/source-inventory-before-dev.json':
        raise ValueError('Position proof requires the original before-dev source freeze')
    frozen, configurations = common.read(source_freeze), common.read(manifest)
    if (frozen.get('schema_version') != 'localocr.position-before-dev-freeze.v1'
            or frozen.get('source_inventory') != source_inventory()
            or request['configuration_manifest'] not in frozen.get('references', [])
            or timestamp(frozen['frozen_utc']) < timestamp(configurations['frozen_utc'])):
        raise ValueError('Position source/configuration differs from original freeze')
    scored = {tier: {population: owned(record) for population, record in pair.items()}
              for tier, pair in request['scored_reports'].items()}
    repeated = {tier: {population: [owned(record) for record in records]
                       for population, records in pair.items()}
                for tier, pair in request['repeated_reports'].items()}
    for pair in scored.values():
        for path in pair.values():
            report = common.read(path)
            if (timestamp(report['completed_utc']) <= timestamp(frozen['frozen_utc'])
                    or timestamp(report['resources']['started_utc']) < timestamp(frozen['frozen_utc'])):
                raise ValueError('Position scored report must follow source freeze')
    for pair in repeated.values():
        for paths in pair.values():
            for path in paths:
                if timestamp(common.read(path)['resources']['started_utc']) < timestamp(frozen['frozen_utc']):
                    raise ValueError('Position repeated worker must follow source freeze')
    from tools.RegionKaomoji import position_quality
    proof = position_quality.verify_round_development(scored, repeated, selection['selected_id'],
        round_root=profile.root, configuration_manifest=request['configuration_manifest'])
    for record in [request_ref, *captured]:
        if reference(record['file']) != record:
            raise ValueError('Position bound input changed during verification')
    chosen = proof['selected_tier']
    if not proof['passed'] or chosen == 'start':
        raise ValueError('Position quality or repeated latency failed; heldouts remain sealed')
    if proof['reports'][chosen]['fresh']['report'] != selection['dev_report']:
        raise ValueError('Position chosen report differs from actual scene selection')
    return {'inputs': request_ref, 'proof': proof}


def _first_pass_development_proof(inputs, scene_selection, root):
    """Recompute all four-tier/repeat originals before first-pass first-use.

    This path cannot reuse predecessor crop, source or quality approval. The
    request holds only immutable original file references; counts and verdicts
    supplied by a caller have no authority.
    """
    profile = rp.profile_for(root)
    if profile is not rp.FIRST_PASS_POSITION:
        if inputs is not None:
            raise ValueError('First-pass development inputs belong only to the first-pass round')
        return None
    if inputs is None:
        raise ValueError('First-pass paired freeze requires four-tier and repeated latency inputs')
    request_ref = reference(inputs)
    request_path = bound(request_ref)
    if rp.profile_for(request_path) is not profile:
        raise ValueError('First-pass request must belong to its round')
    request = common.read(request_path)
    if request.get('schema_version') != 'localocr.first-pass-development-inputs.v1':
        raise ValueError('Invalid position development request schema')
    selection_ref = reference(scene_selection)
    if rp.profile_for(selection_ref['file']) is not profile:
        raise ValueError('First-pass scene selection must belong to its round')
    selection = common.read(scene_selection)
    if request.get('selected_id') != selection['selected_id']:
        raise ValueError('First-pass development model differs from selection')
    captured = [selection_ref]

    def owned(record):
        path = bound(record)
        if rp.profile_for(path) is not profile:
            raise ValueError('First-pass evidence must stay in its registered round')
        captured.append(dict(record))
        return path

    manifest = owned(request['configuration_manifest'])
    source_freeze = owned(request['source_freeze'])
    if source_freeze != Path(profile.root).resolve() / 'validation/source-inventory-before-dev.json':
        raise ValueError('First-pass proof requires the original before-dev source freeze')
    frozen, configurations = common.read(source_freeze), common.read(manifest)
    if (frozen.get('schema_version') != 'localocr.first-pass-before-dev-freeze.v1'
            or frozen.get('source_inventory') != source_inventory()
            or request['configuration_manifest'] not in frozen.get('references', [])
            or timestamp(frozen['frozen_utc']) < timestamp(configurations['frozen_utc'])):
        raise ValueError('First-pass source/configuration differs from original freeze')
    scored = {tier: {population: owned(record) for population, record in pair.items()}
              for tier, pair in request['scored_reports'].items()}
    repeated = {tier: {population: [owned(record) for record in records]
                       for population, records in pair.items()}
                for tier, pair in request['repeated_reports'].items()}
    for pair in scored.values():
        for path in pair.values():
            report = common.read(path)
            if (timestamp(report['completed_utc']) <= timestamp(frozen['frozen_utc'])
                    or timestamp(report['resources']['started_utc']) < timestamp(frozen['frozen_utc'])):
                raise ValueError('First-pass scored report must follow source freeze')
    for pair in repeated.values():
        for paths in pair.values():
            for path in paths:
                if timestamp(common.read(path)['resources']['started_utc']) < timestamp(frozen['frozen_utc']):
                    raise ValueError('First-pass repeated worker must follow source freeze')
    from tools.RegionKaomoji import first_pass_quality
    proof = first_pass_quality.verify_round_development(scored, repeated, selection['selected_id'],
        round_root=profile.root, configuration_manifest=request['configuration_manifest'])
    for record in [request_ref, *captured]:
        if reference(record['file']) != record:
            raise ValueError('First-pass bound input changed during verification')
    chosen = proof['selected_tier']
    if not proof['passed'] or chosen == 'start':
        raise ValueError('First-pass quality or repeated latency failed; heldouts remain sealed')
    if proof['reports'][chosen]['fresh']['report'] != selection['dev_report']:
        raise ValueError('First-pass chosen report differs from actual scene selection')
    return {'inputs': request_ref, 'proof': proof}


def freeze_dual(crop_selection,scene_selection,crop_dataset,scene_dataset,output,regression_reports=(),development_reports=(),joint_development_inputs=None,position_development_inputs=None,first_pass_development_inputs=None):
    selections=[Path(crop_selection),Path(scene_selection)];datasets=[Path(crop_dataset),Path(scene_dataset)]
    _first_pass_round_slots(*selections, *datasets, output)
    _position_round_slots(*selections, *datasets, output)
    _pixel_round_slots(*selections, *datasets, output)
    if not all(is_round(p) for p in selections+datasets+[Path(output)]):raise ValueError('Dual selection must belong to the new research round')
    # Refuse manufacturing a new freeze after either crop or scene first-use exists. A CPU-only round
    # (0 s GPU) that reuses a predecessor round's frozen specialist binds that round's consumed crop
    # first use as provenance instead: no new crop heldout exists to protect.
    predecessor=predecessor_crop_receipt(crop_selection,datasets[0],datasets[1])
    for dataset in datasets:
        if (dataset/'heldout-first-use.json').exists():
            if dataset is datasets[0] and predecessor is not None:
                continue
            raise ValueError('New heldout already consumed')
    round_root=round_scope(datasets[1])[0] or ROUND
    if any(Path(round_root).glob('evaluation-heldout-first-use-*.json')):raise ValueError('Scene heldout already consumed')
    complete_span = _complete_span_proof(development_reports, selections[1], round_root)
    pixel_owned = _pixel_joint_proof(joint_development_inputs, selections[1], round_root)
    position_owned = _position_development_proof(position_development_inputs, selections[1], round_root)
    first_pass_owned = _first_pass_development_proof(first_pass_development_inputs, selections[1], round_root)
    gates=recompute_development(*selections,*datasets)
    regressions=regression_proofs(regression_reports,selections[1],output)
    value={'schema_version':DUAL_SCHEMA,'frozen_utc':common.utc(),'crop_selection':reference(selections[0]),'scene_selection':reference(selections[1]),'crop_dataset':{'path':str(datasets[0].resolve()),'binding':common.dataset_binding(datasets[0])},'scene_dataset':{'path':str(datasets[1].resolve()),'binding':common.read(datasets[1]/'freeze.json')},'source_inventory':source_inventory(),'gates':gates,'production_enabled':False,'promote':False,'enabled':False,'usage':'both selections frozen before either first new heldout; no tuning on results'}
    if regressions:
        value['regressions']=regressions
    if complete_span is not None:
        value['complete_span_development'] = complete_span
    if pixel_owned is not None:
        value['pixel_owned_development'] = pixel_owned
    if position_owned is not None:
        value['position_development'] = position_owned
    if first_pass_owned is not None:
        value['first_pass_development'] = first_pass_owned
    if predecessor is not None:
        value['crop_predecessor']=predecessor;value['usage']='scene selection frozen before its first new heldout; the crop side is the predecessor round\'s frozen specialist whose crop heldout that round consumed; no tuning on results'
    from .evaluate import write_new
    write_new(output,value);return value

def verify_dual(path,kind,selection,dataset):
    value=common.read(bound(reference(path)))
    if value.get('schema_version')!=DUAL_SCHEMA or value.get('production_enabled') is not False or value.get('promote') is not False or value.get('enabled') is not False or value.get('source_inventory')!=source_inventory():raise ValueError('Dual freeze source closure changed')
    _first_pass_round_slots(path, selection, dataset, value['crop_selection']['file'],
                            value['scene_selection']['file'], value.get('crop_dataset', {}).get('path'),
                            value.get('scene_dataset', {}).get('path'))
    _position_round_slots(path, selection, dataset, value['crop_selection']['file'],
                          value['scene_selection']['file'], value.get('crop_dataset', {}).get('path'),
                          value.get('scene_dataset', {}).get('path'))
    _pixel_round_slots(path, selection, dataset, value['crop_selection']['file'],
                       value['scene_selection']['file'], value.get('crop_dataset', {}).get('path'),
                       value.get('scene_dataset', {}).get('path'))
    when=timestamp(value['frozen_utc'])
    for k in ('crop','scene'):
        sp=bound(value[k+'_selection']);selected=common.read(sp)
        if timestamp(selected['frozen_utc'])>when:raise ValueError('Both selections must precede dual freeze')
        dp=Path(value[k+'_dataset']['path']);frozen=common.read(dp/'freeze.json')
        if frozen!=value[k+'_dataset']['binding'] or common.digest(dp/'dataset.json')!=frozen['dataset_sha256']:raise ValueError('Either dataset changed before first-use')
    if Path(selection).resolve()!=Path(value[kind+'_selection']['file']).resolve() or Path(dataset).resolve()!=Path(value[kind+'_dataset']['path']).resolve():raise ValueError('Wrong dual-freeze selection/data slot')
    if rp.profile_for(path) is rp.COMPLETE_SPANS:
        proof = value.get('complete_span_development', {})
        names = ('start_anchor', 'final_anchor', 'start_new', 'final_new')
        reports = [proof['reports'][name]['report']['file'] for name in names] if proof else []
        if _complete_span_proof(reports, value['scene_selection']['file'], rp.COMPLETE_SPANS.root) != proof:
            raise ValueError('Strict complete-span development proof changed')
    if rp.profile_for(path) is rp.PIXEL_OWNED_SPANS:
        proof = value.get('pixel_owned_development', {})
        inputs = bound(proof['inputs']) if proof else None
        if _pixel_joint_proof(inputs, value['scene_selection']['file'], rp.PIXEL_OWNED_SPANS.root) != proof:
            raise ValueError('Strict joint quality or repeated latency proof changed')
    if rp.profile_for(path) is rp.OWNED_POSITION:
        proof = value.get('position_development', {})
        inputs = bound(proof['inputs']) if proof else None
        if _position_development_proof(inputs, value['scene_selection']['file'], rp.OWNED_POSITION.root) != proof:
            raise ValueError('Strict position quality or repeated latency proof changed')
    if rp.profile_for(path) is rp.FIRST_PASS_POSITION:
        proof = value.get('first_pass_development', {})
        inputs = bound(proof['inputs']) if proof else None
        if _first_pass_development_proof(inputs, value['scene_selection']['file'], rp.FIRST_PASS_POSITION.root) != proof:
            raise ValueError('Strict first-pass quality or repeated latency proof changed')
    gates=recompute_development(value['crop_selection']['file'],value['scene_selection']['file'],value['crop_dataset']['path'],value['scene_dataset']['path'])
    if gates!=value['gates']:raise ValueError('Development gate proof changed')
    if 'regressions' in value or REQUIRED_REGRESSIONS.get((rp.profile_for(path).name if rp.profile_for(path) else None)):
        reports=[item['report']['file'] for item in value.get('regressions',[])]
        if regression_proofs(reports,value['scene_selection']['file'],path)!=value.get('regressions'):raise ValueError('Regression proof changed')
    return reference(path)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--crop-selection',required=True);p.add_argument('--scene-selection',required=True);p.add_argument('--crop-dataset',required=True);p.add_argument('--scene-dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--regression-report',action='append',default=[],help='Development scene report or known-input probe of the selected configuration (repeatable)')
    p.add_argument('--development-report',action='append',default=[],help='Complete-span reports in start-anchor,final-anchor,start-new,final-new order')
    p.add_argument('--joint-development-inputs',help='Pixel-owned round four-tier and three-repeat original report references')
    p.add_argument('--position-development-inputs',help='Position round five-tier and three-repeat original report references')
    p.add_argument('--first-pass-development-inputs',help='First-pass round four-tier and three-repeat original report references')
    a=p.parse_args(argv)
    value=freeze_dual(a.crop_selection,a.scene_selection,a.crop_dataset,a.scene_dataset,a.output,a.regression_report,a.development_report,a.joint_development_inputs,a.position_development_inputs,a.first_pass_development_inputs)
    print(json.dumps({'schema_version':value['schema_version'],'selected_id':value['gates']['selected_id'],'research_quality_passed':True,'production_enabled':False}))
if __name__=='__main__':main()
