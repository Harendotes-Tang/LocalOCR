"""Validate immutable original datasets and record explicit selection reuse."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from urllib.parse import urlsplit
from pathlib import Path
from PIL import Image

SPLITS = ('train','dev','test','regression','acceptance')
KINDS = ('plain','face','punctuation','emoji','blank','graphic')
ROW_FIELDS={'id','group','kind','label','label_sha256','path','sha256','width','height','region','font','transform','provenance','usage','rendering','recipe_id'}
TRANSFORM_FIELDS={'size','dark','scale','blur','jpeg','background','shapes'}
RENDER_FIELDS={'glyphs','colored_glyphs','composed','bounds_basis','text_present'}
ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def label_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def local_output(path):
    target=Path(path).resolve(); artifact=(ROOT/'artifacts').resolve()
    if artifact.parent!=ROOT.resolve() or artifact not in target.parents:
        raise ValueError('Output must be inside repository artifacts')
    return target

def confined(root,relative):
    if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Invalid relative image path')
    target=(root/relative).resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError('Image path escapes dataset or is missing')
    return target

def validate(root, require_frozen=True):
    root=Path(root).resolve(); meta=read(root/'dataset.json')
    if meta.get('schema_version')!='localocr.dataset.v1' or meta.get('synthetic_only') is not True or meta.get('license')!='CC0-1.0':
        raise ValueError('Dataset must declare original CC0 synthetic schema')
    fonts={f['id']:f for f in meta['fonts']}
    if not fonts or len(fonts)!=len(meta['fonts']): raise ValueError('Missing or duplicate font provenance')
    for f in fonts.values():
        if f.get('license')!='OFL-1.1' or not all(f.get(k) for k in ('sha256','license_sha256','source_url','source_commit','version')):
            raise ValueError('Incomplete OFL font provenance')
        for key in ('sha256','license_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}',f[key]): raise ValueError('Invalid font hash')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',f['id']): raise ValueError('Invalid public font identifier')
        if not re.fullmatch(r'[A-Za-z0-9 .+_-]{1,128}',f['version']): raise ValueError('Invalid public font version')
        url=urlsplit(f['source_url'])
        if url.username or url.password or url.query or url.fragment: raise ValueError('Font URL cannot contain credentials or private query')
        if url.scheme!='https' or url.hostname not in ('github.com','raw.githubusercontent.com') or not url.path.startswith(('/notofonts/','/googlefonts/')):
            raise ValueError('Font source must be official Noto repository')
        if not re.fullmatch(r'[0-9a-f]{40}',f['source_commit']) or f['source_commit'] not in url.path:
            raise ValueError('Font source must pin an official commit')
    exclusions=meta.get('exclusions',{})
    excluded=set()
    if exclusions.get('inventory_path'):
        inventory=Path(exclusions['inventory_path']).resolve()
        if (ROOT/'artifacts').resolve() not in inventory.parents:
            raise ValueError('Exclusion inventory must stay local artifacts')
        if digest(inventory)!=exclusions['inventory_sha256']: raise ValueError('Exclusion inventory changed')
        inv=read(inventory)
        if not isinstance(inv,dict) or inv.get('schema_version')!='localocr.exclusions.v1': raise ValueError('Unknown exclusion inventory')
        excluded=set(inv['label_sha256'])
        if len(excluded)!=exclusions.get('count') or any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v) for v in excluded):
            raise ValueError('Excluded label count or digest differs')
    elif exclusions.get('count',0): raise ValueError('Nonempty exclusions require local inventory')
    groups={}; texts={}; pixels={}; ids=set(); rows_by_split={}; bindings={}
    for split in SPLITS:
        item=meta['splits'][split]
        if item['file']!=split+'.json': raise ValueError('Unexpected split filename')
        path=root/item['file']; rows=read(path)
        if not rows or len(rows)!=item['count'] or digest(path)!=item['sha256']:
            raise ValueError('Split differs from manifest')
        bindings[split]=digest(path); rows_by_split[split]=rows
        for row in rows:
            if set(row)-ROW_FIELDS or set(row.get('provenance',{}))-{'source','license'} or set(row.get('transform',{}))-TRANSFORM_FIELDS or set(row.get('rendering',{}))-RENDER_FIELDS:
                raise ValueError('Undeclared record field may contain private material')
            transform=row.get('transform',{})
            for key in ('size','jpeg'):
                if key in transform and transform[key] is not None and type(transform[key]) is not int: raise ValueError('Invalid numeric transform')
            for key in ('scale','blur'):
                if key in transform and type(transform[key]) not in (int,float): raise ValueError('Invalid numeric transform')
            if 'dark' in transform and type(transform['dark']) is not bool: raise ValueError('Invalid dark transform')
            if 'background' in transform and (len(transform['background'])!=3 or any(type(v) is not int or not 0<=v<=255 for v in transform['background'])):
                raise ValueError('Invalid geometric background')
            for shape in transform.get('shapes',[]):
                if set(shape)-{'ellipse','line','color'}: raise ValueError('Undeclared geometric recipe field')
                if any(len(shape[k])!=n or any(type(v) is not int for v in shape[k]) for k,n in (('ellipse',4),('line',4),('color',3))):
                    raise ValueError('Invalid geometric recipe values')
            if row['id'] in ids: raise ValueError('Duplicate row ID')
            ids.add(row['id'])
            if row['kind'] not in KINDS or row['usage']!=split or row.get('provenance',{}).get('source')!='original' or row.get('provenance',{}).get('license')!='CC0-1.0':
                raise ValueError('Non-original or invalid usage record')
            text=row['label']
            if not isinstance(text,str) or label_digest(text)!=row['label_sha256']: raise ValueError('Exact Unicode label hash differs')
            negative=row['kind'] in ('blank','graphic')
            if negative:
                if text or not row.get('recipe_id') or row.get('group')!='negative:'+row.get('recipe_id',''): raise ValueError('Invalid negative recipe group')
            else:
                if not text or row['group']!='text:'+row['label_sha256']: raise ValueError('Invalid exact text group')
                if row['label_sha256'] in excluded: raise ValueError('Prior or reserved label reused')
                if row['label_sha256'] in texts and texts[row['label_sha256']]!=split: raise ValueError('Cross-split exact text leakage')
                texts[row['label_sha256']]=split
                if not row['font'] or any(f not in fonts for f in row['font']): raise ValueError('Undeclared font')
                evidence=row.get('rendering',{})
                if not isinstance(evidence.get('glyphs'),int) or evidence['glyphs']<=0 or evidence.get('bounds_basis')!='transformed-raster-alpha':
                    raise ValueError('Missing actual glyph rendering evidence')
                if row['kind']=='emoji' and (evidence.get('composed') is not True or evidence.get('colored_glyphs',0)<=0):
                    raise ValueError('Emoji requires actual color/composition evidence')
            if row['group'] in groups and groups[row['group']]!=split: raise ValueError('Cross-split source group leakage')
            groups[row['group']]=split
            image_path=confined(root,row['path'])
            if digest(image_path)!=row['sha256']: raise ValueError('Image bytes changed')
            with Image.open(image_path) as image:
                image.load()
                if image.size!=(row['width'],row['height']): raise ValueError('Image dimensions differ')
                rgb=image.convert('RGB')
                pixel_hash=hashlib.sha256(rgb.width.to_bytes(8,'big')+rgb.height.to_bytes(8,'big')+rgb.tobytes()).hexdigest()
                if pixel_hash in pixels and pixels[pixel_hash]!=split: raise ValueError('Cross-split decoded pixel duplicate')
                pixels[pixel_hash]=split
            x,y,w,h=row['region']
            if any(type(v) is not int for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or x+w>row['width'] or y+h>row['height']:
                raise ValueError('Invalid pixel region')
    binding={'schema_version':'localocr.freeze.v1','dataset_sha256':digest(root/'dataset.json'),'split_sha256':bindings,
             'coverage_sha256':digest(root/'coverage.json') if (root/'coverage.json').exists() else None,
             'row_count':len(ids),'group_count':len(groups),'exact_text_count':len(texts),'pixel_hash_count':len(pixels)}
    if require_frozen and read(root/'freeze.json')!=binding: raise ValueError('Frozen inventory differs')
    usage_state(root,binding)
    return meta, rows_by_split, binding

def freeze(root):
    root=local_output(root)
    if (root/'freeze.json').exists(): raise ValueError('Freeze already exists; validate without rewriting')
    _,_,binding=validate(root,False); write(root/'freeze.json',binding); return binding

def usage_state(root,binding):
    """Validate canonical demotion receipts without rewriting frozen data bytes."""
    path=Path(root)/'selection-use.json'
    if not path.exists(): return {}
    state=read(path)
    if state.get('schema_version')!='localocr.selection-state.v1' or not isinstance(state.get('splits'),dict):
        raise ValueError('Invalid selection-use state')
    for split,value in state['splits'].items():
        if split not in ('test','acceptance') or value.get('dataset_sha256')!=binding['dataset_sha256'] or value.get('split_sha256')!=binding['split_sha256'][split] or value.get('usage')!='development-regression' or value.get('independent_acceptance') is not False or not value.get('reason'):
            raise ValueError('Selection demotion receipt differs from frozen split')
    return state['splits']

def selection_use(root,receipt,split,reason):
    if split not in ('test','acceptance'): raise ValueError('Only independent splits require demotion receipt')
    root=local_output(root);_,_,binding=validate(root)
    state=usage_state(root,binding)
    if split in state: raise ValueError('Split has already been demoted')
    target=local_output(receipt) if receipt else None
    if target and target.exists(): raise ValueError('Selection receipt already exists')
    if not reason or not reason.strip(): raise ValueError('Selection use needs a reason')
    value={'schema_version':'localocr.selection-use.v1','dataset_sha256':binding['dataset_sha256'],'split':split,
           'split_sha256':binding['split_sha256'][split],'usage':'development-regression','independent_acceptance':False,'reason':reason,
           'required_next_step':'Generate a fresh independently grouped acceptance inventory'}
    state[split]=value
    write(root/'selection-use.json',{'schema_version':'localocr.selection-state.v1','splits':state})
    if target: write(target,value)
    return value

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['validate','freeze','selection-use']); p.add_argument('--dataset',required=True)
    p.add_argument('--receipt');p.add_argument('--split',choices=['test','acceptance']);p.add_argument('--reason');a=p.parse_args()
    if a.action=='freeze': result=freeze(a.dataset)
    elif a.action=='selection-use': result=selection_use(a.dataset,a.receipt,a.split,a.reason)
    else: result=validate(a.dataset)[2]
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
