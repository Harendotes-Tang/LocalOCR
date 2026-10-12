"""Local CC0 regression export with an explicit file whitelist."""
import argparse
import json
import shutil
from pathlib import Path
from .validate import validate,local_output,write,digest

def export(root,output):
    root=Path(root).resolve(); meta,splits,binding=validate(root); out=local_output(output)
    if out.exists(): raise ValueError('Export destination already exists')
    out.mkdir(parents=True)
    rows=splits['regression']
    for row in rows:
        target=out/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/row['path'],target)
    write(out/'regression.json',rows)
    public={k:meta[k] for k in ('schema_version','seed','license','synthetic_only')}
    public['generator']={k:meta['generator'][k] for k in ('version','source_sha256','groups') if k in meta['generator']}
    public['renderer']={k:meta['renderer'][k] for k in ('pillow','harfbuzz','freetype','regex','python') if k in meta['renderer']}
    public['fonts']=[{k:f[k] for k in ('id','sha256','license_sha256','license','source_url','source_commit','version')} for f in meta['fonts']]
    public['splits']={'regression':{'file':'regression.json','count':len(rows),'sha256':digest(out/'regression.json')}}
    public['source_freeze_sha256']=digest(root/'freeze.json')
    write(out/'dataset.json',public)
    (out/'NOTICE.txt').write_text('Original text, fictional records, geometric recipes, generated images and manifests: CC0-1.0.\nhttps://creativecommons.org/publicdomain/zero/1.0/\nGenerator code: GPL-3.0-or-later, supplied separately in the source repository.\nOfficial OFL font binaries, model weights, private inputs and exclusion inventories are not included.\nRegenerate using the pinned font sources and renderer versions in dataset.json.\nThis local export does not publish or upload any data.\n',encoding='utf-8')
    allowed={'dataset.json','regression.json','NOTICE.txt'}|{r['path'] for r in rows}
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if actual!=allowed: raise ValueError('Export whitelist mismatch')
    return {'files':len(actual),'images':len(rows),'font_binaries':0,'private_inputs':0,'model_weights':0,'uploaded':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(export(a.dataset,a.output),indent=2))
if __name__=='__main__': main()
