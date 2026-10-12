"""Actual CPU paired comparison of an unenabled, frozen kaomoji experiment."""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path
import regex
from PIL import Image
from .validate import validate,local_output,write,digest,read,KINDS,usage_state
from .private import read_private

LEGACY=Path(__file__).resolve().parents[1]/'KaomojiOcr'

def legacy_module():
    sys.path.insert(0,str(LEGACY))
    spec=importlib.util.spec_from_file_location('localocr_frozen_comparison',LEGACY/'evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=current
    return row[-1]

def metrics(rows,field):
    import unicodedata
    syms=lambda s:''.join(c for c in s if unicodedata.category(c)[0] in 'PSM')
    chars=sum(len(r['label']) for r in rows);errors=sum(distance(r['label'],r[field]) for r in rows)
    symbols=sum(len(syms(r['label'])) for r in rows);se=sum(distance(syms(r['label']),syms(r[field])) for r in rows)
    gs=[(regex.findall(r'\X',r['label']),regex.findall(r'\X',r[field])) for r in rows]
    gc=sum(len(a) for a,b in gs);ge=sum(distance(a,b) for a,b in gs)
    return {'count':len(rows),'exact':sum(r['label']==r[field] for r in rows),'characters':chars,'character_errors':errors,
        'cer':errors/chars if chars else None,'symbol_characters':symbols,'symbol_errors':se,'symbol_cer':se/symbols if symbols else None,
        'graphemes':gc,'grapheme_errors':ge,'grapheme_cer':ge/gc if gc else None,
        'grapheme_exact':sum(a==b for a,b in gs)}

def summarize(rows):
    exact=[r for r in rows if not r.get('ambiguous',False)]; report={'total':len(rows),'ambiguous_excluded':len(rows)-len(exact),'categories':{}}
    for kind in KINDS:
        part=[r for r in exact if r['kind']==kind]
        item={f:metrics(part,f) for f in ('baseline','candidate')}
        item['candidate_regressions']=sum(r['baseline']==r['label'] and r['candidate']!=r['label'] for r in part)
        item['candidate_erroneous_rewrites']=sum(r['candidate']!=r['baseline'] and r['candidate']!=r['label'] for r in part)
        if kind in ('blank','graphic'):
            item['negative_false_outputs']={f:sum(bool(r[f]) for r in part) for f in ('baseline','candidate')}
        report['categories'][kind]=item
    return report

def frozen_candidate_authority():
    """Read the experiment display source while retaining historical authority."""
    authority=read(Path(__file__).with_name('frozen-v4.json'))
    if authority.get('schema_version')!='localocr.frozen-candidate.v1' or authority.get('identity')!='rejected-v4':
        raise ValueError('Unknown frozen candidate authority')
    if (not isinstance(authority.get('display_name'),str)
            or not re.fullmatch(r'kaomoji-exp-[0-9]{8}-[0-9]{3}',authority['display_name'])
            or authority.get('historical_marker')!='v4' or authority.get('identity_kind')!='experiment'
            or authority.get('status')!='rejected' or authority.get('enabled') is not False):
        raise ValueError('Frozen experiment must retain its rejected, unenabled identity')
    return authority


def frozen_candidate_metadata():
    """Project display metadata for new reports; this never enables a model."""
    authority=frozen_candidate_authority()
    return {'candidate_identity':authority['identity'],'candidate_display_name':authority['display_name'],
            'candidate_historical_marker':authority['historical_marker'],'candidate_identity_kind':authority['identity_kind'],
            'candidate_status':'rejected; not enabled; CPU comparison reference only',
            'candidate_enabled':False,'promote':False}


def frozen_candidate_binding(directory):
    """Bind to approved experiment bytes, not an arbitrary self-signed report."""
    directory=Path(directory)
    authority=frozen_candidate_authority()
    binding={key:digest(directory/name) for key,name in (
        ('model_sha256','kaomoji.onnx'),('alphabet_sha256','alphabet.json'),('training_sha256','training.json'))}
    if any(binding[key]!=authority[key] for key in binding):
        raise ValueError('Candidate is not the approved frozen rejected v4')
    return binding


def evaluate(args):
    root=Path(args.dataset).resolve();meta,splits,binding=validate(root);out=local_output(args.output)
    if out.exists():raise ValueError('Evaluation output already exists')
    if args.split not in ('test','regression','acceptance'):raise ValueError('Evaluate explicit heldout or regression split')
    candidate=Path(args.candidate).resolve();baseline_path=Path(args.baseline_models).resolve()
    candidate_binding=frozen_candidate_binding(candidate)
    module=legacy_module();session,alphabet,resources=module.load_candidate(candidate)
    worker_path=Path(__file__).resolve().parents[2]/'scripts/local-ai-ocr-worker.py'
    spec=importlib.util.spec_from_file_location('localocr_dataset_baseline',worker_path);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    baseline=worker.create_engine(baseline_path,baseline_path/'manifest.json')
    if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('Candidate must use CPU only')
    baseline_binding={'manifest_sha256':digest(baseline_path/'manifest.json'),'models':{role:entry['sha256'] for role,entry in read(baseline_path/'manifest.json')['models'].items()}}
    rows=splits[args.split];images=[]
    for row in rows:
        with Image.open(root/row['path']) as image:images.append(image.convert('RGB'))
    started=time.perf_counter();predictions,times=module.compare(images,rows,session,alphabet,baseline)
    private_predictions=[];private_binding=None;private_times=[]
    if args.private_manifest:
        pr,pi,private_binding=read_private(args.private_manifest)
        private_predictions,private_times=module.compare(pi,pr,session,alphabet,baseline)
    if validate(root)[2]!=binding or digest(candidate/'kaomoji.onnx')!=candidate_binding['model_sha256'] or digest(candidate/'alphabet.json')!=candidate_binding['alphabet_sha256'] or digest(candidate/'training.json')!=candidate_binding['training_sha256']:
        raise ValueError('Frozen inputs changed during evaluation')
    if args.private_manifest and read_private(args.private_manifest)[2]!=private_binding:
        raise ValueError('Private frozen inputs changed during evaluation')
    if digest(baseline_path/'manifest.json')!=baseline_binding['manifest_sha256']:
        raise ValueError('Baseline manifest changed during evaluation')
    for role,entry in read(baseline_path/'manifest.json')['models'].items():
        if digest(baseline_path/entry['file'])!=baseline_binding['models'][role]:raise ValueError('Baseline model changed during evaluation')
    report={'schema_version':'localocr.evaluation.v1','split':args.split,'dataset_binding':binding,'baseline_binding':baseline_binding,
         'candidate_binding':candidate_binding,**frozen_candidate_metadata(),
         'synthetic':summarize(predictions),'private':summarize(private_predictions),
         'evaluated_usage':'development-regression' if args.split=='regression' or args.split in usage_state(root,binding) else 'independent-heldout',
         'independent_acceptance':args.split=='acceptance' and args.split not in usage_state(root,binding),
         'private_usage':'previously seen regression evidence; not fresh independent acceptance',
         'resources':resources,'elapsed_seconds':time.perf_counter()-started,'median_candidate_crop_seconds':statistics.median(times+private_times),
         'scope':'identical manually supplied synthetic/private crops; raw recognizer CPU inference; not automatic ROI or APP lifecycle accuracy',
         'normalization':'none; exact Unicode codepoints and extended graphemes','private_source_count':len(set(private_binding['source_hashes'])) if private_binding else 0,
         'private_crop_count':len(private_predictions)}
    write(out/'synthetic-predictions.json',predictions)
    if args.private_manifest:
        write(out/'private-predictions.json',{'binding':private_binding,'rows':private_predictions})
    write(out/'evaluation.json',report)
    # This summary contains aggregate counts and model/dataset hashes only.
    return report

def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','candidate','baseline-models','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['test','regression','acceptance'],default='test');p.add_argument('--private-manifest');a=p.parse_args()
    print(json.dumps(evaluate(a),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
