"""Train/dev-only projection and explicit opt-in existing CTC trainer entry."""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from .validate import validate,local_output,write,read,digest,confined

LEGACY=Path(__file__).resolve().parents[1]/'KaomojiOcr'

def project(root,output):
    root=Path(root).resolve();_,splits,binding=validate(root);out=local_output(output)
    if out.exists(): raise ValueError('Projection destination already exists')
    out.mkdir(parents=True);alphabet=sorted({c for split in ('train','dev') for r in splits[split] for c in r['label']})
    hashes={};counts={}
    for split in ('train','dev'):
        rows=[]
        for row in splits[split]:
            target=out/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/row['path'],target)
            rows.append({k:row[k] for k in ('path','label','kind','font','sha256')})
        write(out/(split+'.json'),rows);hashes[split]=digest(out/(split+'.json'));counts[split]=len(rows)
    write(out/'dataset.json',{'schema':1,'synthetic_only':True,'private_inputs_consumed':False,'alphabet':alphabet,
          'image_rows':counts,'split_sha256':hashes,'source_freeze':binding,'usage':'train-dev-only'})
    binding=projection_fingerprint(out);write(out/'projection-freeze.json',binding);return binding

def projection_fingerprint(root):
    root=Path(root).resolve();meta=read(root/'dataset.json')
    if meta.get('usage')!='train-dev-only' or meta.get('private_inputs_consumed') is not False or meta.get('synthetic_only') is not True:
        raise ValueError('Invalid training projection declaration')
    if (root/'test.json').exists() or (root/'acceptance.json').exists() or (root/'regression.json').exists():
        raise ValueError('Heldout split must not enter training projection')
    hashes={};sets={};alphabet=set()
    for split in ('train','dev'):
        path=root/(split+'.json');rows=read(path);hashes[split]=digest(path)
        if not rows or len(rows)!=meta['image_rows'][split]: raise ValueError('Projection rows differ')
        sets[split]={r['label'] for r in rows if r['label']}
        for row in rows:
            alphabet.update(row['label'])
            if digest(confined(root,row['path']))!=row['sha256']: raise ValueError('Projection image changed')
    if hashes!=meta['split_sha256'] or sets['train']&sets['dev']: raise ValueError('Projection changed or labels overlap')
    if meta['alphabet']!=sorted(alphabet): raise ValueError('Alphabet must come exclusively from train/dev')
    return {'dataset_sha256':digest(root/'dataset.json'),'split_sha256':hashes}

def check(root):
    """One CPU forward/CTC-loss probe; no backward, optimizer or model output."""
    root=Path(root).resolve();binding=projection_fingerprint(root)
    if read(root/'projection-freeze.json')!=binding: raise ValueError('Projection differs from freeze')
    sys.path.insert(0,str(LEGACY))
    spec=importlib.util.spec_from_file_location('localocr_ctc_probe',LEGACY/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch=module.torch;torch.set_num_threads(2);torch.manual_seed(20261011)
    alphabet=read(root/'dataset.json')['alphabet'];model=module.Recognizer(len(alphabet)+1).cpu().eval()
    inputs,targets,labels=module.load_split(root,'train',alphabet)
    with torch.no_grad():
        logits=model(inputs[:1]);target=targets[0]
        loss=torch.nn.CTCLoss(blank=0,zero_infinity=False)(logits.log_softmax(-1).transpose(0,1),target,
             torch.tensor([logits.shape[1]],dtype=torch.long),torch.tensor([len(target)],dtype=torch.long))
    if not torch.isfinite(loss): raise ValueError('Training entry CPU CTC probe produced nonfinite loss')
    if projection_fingerprint(root)!=binding: raise ValueError('Projection changed during probe')
    return {'projection_binding':binding,'device':'cpu','parameters':sum(p.numel() for p in model.parameters()),
            'logit_shape':list(logits.shape),'ctc_loss':float(loss),'torch_version':torch.__version__,
            'optimizer_steps':0,'backward_calls':0,'weights_written':False,'probe_count':1}

def run(args):
    root=Path(args.dataset).resolve();binding=projection_fingerprint(root)
    if read(root/'projection-freeze.json')!=binding: raise ValueError('Projection differs from freeze')
    sys.path.insert(0,str(LEGACY))
    spec=importlib.util.spec_from_file_location('localocr_ctc_train',LEGACY/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.dataset_fingerprint=projection_fingerprint
    module.main(args)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['project','verify','check','run']);p.add_argument('--dataset',required=True);p.add_argument('--output')
    p.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='cpu');p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch',type=int,default=96);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=20261011);a=p.parse_args()
    if a.action=='project': value=project(a.dataset,a.output)
    elif a.action=='check': value=check(a.dataset)
    elif a.action=='verify':
        value=projection_fingerprint(a.dataset)
        if read(Path(a.dataset)/'projection-freeze.json')!=value: raise ValueError('Projection changed')
    else:
        if not a.output or a.epochs<1 or a.batch<1 or a.lr<=0: p.error('Training needs output and positive hyperparameters')
        run(a);return
    print(json.dumps(value,indent=2))
if __name__=='__main__':main()
