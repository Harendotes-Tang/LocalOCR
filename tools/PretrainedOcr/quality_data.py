"""Local verified-font quality rasters, exact labels and sealed split identities.

Counts include negatives. All visible text is original CC0 material; code follows
repository licensing. Existing font files are verified and referenced, never copied.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import shutil
import stat
import time
import numpy as np
from PIL import Image, ImageDraw
from tools.OcrDatasets.render import Unsupported
from .data import MixedRenderer, pixels, sha, read, write
from .common import write as atomic_write, utc
from .quality_corpus import (SPLITS, BRACKETS, GENERATOR_ID, make_sample, expressions,
                             text_hash, family)
from . import round_profile as rp

ARTIFACTS=Path(__file__).resolve().parents[2]/'artifacts'
OWNED_BYTE_LIMIT=350*1024**2
ROUND_BYTE_LIMIT=1024**3
MINIMUM_FREE=8*1024**3


def round_limits(root):
    """(owned quota, round quota, owned label, round label) for a registered round or the legacy defaults."""
    profile=rp.profile_for(root)
    if profile is None:return OWNED_BYTE_LIMIT,ROUND_BYTE_LIMIT,'350MiB','1GiB'
    return profile.owned_bytes,profile.maximum_bytes,f'{profile.owned_bytes//1024**2}MiB',f'{profile.maximum_bytes//1024**3}GiB'


def tree_bytes(path):
    path=Path(path)
    if not path.exists():return 0
    total=0;pending=[str(path)]
    while pending:
        current=pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                info=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                    raise ValueError('Reparse artifact path is forbidden')
                if stat.S_ISDIR(info.st_mode):pending.append(entry.path)
                elif stat.S_ISREG(info.st_mode):total+=info.st_size
    return total


def quota_root(path):
    path=Path(path).resolve()
    profile=rp.profile_for(path)
    if profile is not None:return Path(profile.root).resolve()
    return next((p for p in (path,*path.parents) if p.name.startswith('quality-optimization-')),path)


def scoped_root(root):
    return rp.profile_for(root) is not None or Path(root).name.startswith('quality-optimization-')


def storage_gate(path,reserved=0):
    path=Path(path).resolve();probe=path
    while not probe.exists():probe=probe.parent
    free=shutil.disk_usage(probe).free
    if free-reserved<MINIMUM_FREE:raise ValueError('Disk headroom below 8GiB')
    root=quota_root(path);owned_limit,round_limit,owned_label,round_label=round_limits(root)
    owned=tree_bytes(root/'data')+tree_bytes(root/'scenes') if root!=path or scoped_root(root) else tree_bytes(path)
    if owned+reserved>owned_limit:raise ValueError('Data/scenes quota exceeds '+owned_label)
    if tree_bytes(root)+reserved>round_limit:raise ValueError('Round artifacts exceed '+round_label)
    return {'free_bytes':free,'owned_bytes':owned,'round_bytes':tree_bytes(root)}


def output_root(output):
    if str(output).startswith(('\\\\','//')):raise ValueError('Network output forbidden')
    root=Path(output).resolve()
    if not root.is_relative_to(ARTIFACTS.resolve()) or root==ARTIFACTS.resolve():raise ValueError('Output must remain in local artifacts')
    for p in (root,*root.parents):
        if p.exists():
            info=p.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('Reparse output forbidden')
    return root


def nonowned_bytes(root):
    """Read other owners' actual bytes without rescanning this owner's images."""
    root=Path(root)
    if not root.exists():return 0
    total=0
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name in ('data','scenes'):continue
            info=entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                raise ValueError('Reparse artifact path is forbidden')
            total+=tree_bytes(entry.path) if stat.S_ISDIR(info.st_mode) else info.st_size
    return total


class Writer:
    """Reserve each encoded write against owned bytes and live external bytes.

    Data/scenes are exclusive to this owner; integration writes outside those
    namespaces. Cached owned inventories plus exact encoded-byte reservations
    avoid scanning tens of thousands of immutable images. Every write refreshes
    other owners' actual bytes. Full inventories reconcile after 500 writes or
    30 seconds; the local progress receipt is statted on every write as well.
    """
    def __init__(self,root):
        self.root=Path(root);self.quota=quota_root(root);state=storage_gate(root)
        self.owned=state['owned_bytes'];self.round=state['round_bytes']
        self.scoped=scoped_root(self.quota)
        self.owned_limit,self.round_limit,self.owned_label,self.round_label=round_limits(self.quota)
        self.progress=self.root/'progress.json';self.progress_base=self.progress.stat().st_size if self.progress.exists() else 0
        self.external=nonowned_bytes(self.quota) if self.scoped else self.round-self.owned
        self.pending=0;self.writes=0;self.reconciled=time.monotonic()
    def bytes(self,path,payload):
        n=len(payload)
        if self.writes%500==0 or time.monotonic()-self.reconciled>=30:
            state=storage_gate(self.root,n);self.owned=state['owned_bytes'];self.round=state['round_bytes']
            self.pending=0;self.reconciled=time.monotonic()
            self.progress_base=self.progress.stat().st_size if self.progress.exists() else 0
        auxiliary=(self.progress.stat().st_size if self.progress.exists() else 0)-self.progress_base
        owned=self.owned+self.pending+auxiliary
        external=nonowned_bytes(self.quota) if self.scoped else self.round-self.owned
        owned_limit,round_limit,owned_label,round_label=round_limits(self.quota)
        if owned+n>owned_limit:raise ValueError('Data/scenes quota exceeds '+owned_label)
        if owned+external+n>round_limit:raise ValueError('Round artifacts exceed '+round_label)
        if shutil.disk_usage(self.root).free-n<MINIMUM_FREE:raise ValueError('Disk headroom below 8GiB')
        with Path(path).open('xb') as stream:stream.write(payload)
        self.pending+=n;self.writes+=1
    def image(self,path,image):
        buffer=BytesIO();image.save(buffer,format='PNG',compress_level=6)
        self.bytes(path,buffer.getvalue())
    def json(self,path,value):
        self.bytes(path,(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8'))


def render_variant(renderer,text,variant,size=24):
    image,region,fonts,evidence=renderer.render(text,size=size)
    x,y,w,h=region;pad=3
    # Keep label whitespace advances while discarding the renderer's artificial
    # two-em margins; no label or codepoint is trimmed or normalized.
    def advance(value):
        return sum(p.x_advance/f[5]*size for f,_,positions in renderer.shape(value) for p in positions) if value else 0
    leading=text[:len(text)-len(text.lstrip(' '))];trailing=text[len(text.rstrip(' ')):]
    left=max(0,int(x-pad-advance(leading)));right=min(image.width,int(np.ceil(x+w+pad+advance(trailing))))
    top=max(0,y-pad);bottom=min(image.height,y+h+pad)
    image=image.crop((left,top,right,bottom));region=[x-left,y-top,w,h]
    if variant:
        padded=Image.new('RGB',(image.width+12,image.height+12),image.getpixel((0,0)))
        padded.paste(image,(6,6));image=padded;region[0]+=6;region[1]+=6
    evidence=dict(evidence,whitespace_normalization='none',view='edge-pad-6' if variant else 'raw-ink-pad-3')
    return image,region,fonts,evidence


def negative(split,index,variant,rng):
    width=rng.randrange(48,240)+variant*12;height=rng.randrange(24,55)+variant*6
    background=tuple(rng.randrange(210,252) for _ in range(3))
    array=np.full((height,width,3),background,dtype=np.uint8)
    if split=='dev':
        array=np.clip(array.astype(int)-np.arange(height)[:,None,None]//3,0,255).astype('uint8')
    elif split=='heldout':
        yy,xx=np.indices((height,width));array=(array.astype(int)-((xx*17+yy*13)%4)[:,:,None]).astype('uint8')
    image=Image.fromarray(array);kind='blank' if index%20==18 else 'graphic'
    primitive=('ellipse','triangle','box','sine')[index//20%4]
    if kind=='graphic':
        draw=ImageDraw.Draw(image);box=(4,4,width-5,height-5);ink=(40,50,70)
        if primitive=='ellipse':draw.ellipse(box,outline=ink,width=1+variant)
        elif primitive=='triangle':draw.polygon([(width//2,4),(width-5,height-5),(4,height-5)],outline=ink,width=1+variant)
        elif primitive=='box':draw.rectangle(box,outline=ink,width=1+variant)
        else:draw.line([(x,round(height/2+height/4*np.sin(x*4*np.pi/width))) for x in range(4,width-4)],fill=ink,width=1+variant)
        if split=='dev':draw.line((width//2,2,width//2,height-3),fill=ink,width=1)
        if split=='heldout':draw.line((4,height-5,width-5,4),fill=ink,width=1)
    return image,{'background':background,'geometry':primitive if kind=='graphic' else None,'composition':'single' if split=='train' else 'paired' if split=='dev' else 'overlaid'}


def previous_exclusions():
    seen=set();bindings={}
    for folder in (ARTIFACTS/'pretrained-round-20261011/data',ARTIFACTS/'ocr-datasets-v1'):
        if not folder.exists():continue
        for name in ('train','dev','heldout','test','regression','acceptance'):
            path=folder/(name+'.json')
            if not path.exists():continue
            bindings[str(path.relative_to(ARTIFACTS))]=sha(path)
            for row in read(path):
                label=row.get('text',row.get('label',''))
                if label:seen.add(text_hash(label))
    exclusion=ARTIFACTS/'ocr-datasets-verification-20261011/exclusions.json'
    if exclusion.exists():
        bindings[str(exclusion.relative_to(ARTIFACTS))]=sha(exclusion)
        seen.update(read(exclusion).get('label_sha256',[]))
    return seen,bindings


def validate(root):
    root=Path(root).resolve();meta=read(root/'dataset.json')
    if meta.get('schema_version')!='localocr.pretrained.dataset.v1' or meta.get('generator_id')!=GENERATOR_ID or meta.get('synthetic_only') is not True:
        raise ValueError('Unknown quality dataset schema/generator')
    texts={};groups={};families={};pixelsets={};counts={};tokens=set()
    for split in SPLITS:
        info=meta['splits'][split];relative=Path(info['file'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:raise ValueError('Split path escape')
        if sha(root/relative)!=info['sha256']:raise ValueError('Split changed')
        rows=read(root/relative)
        if len(rows)!=info['count']:raise ValueError('Split count changed')
        for row in rows:
            if row['split']!=split or row['provenance']!={'source':'original','license':'CC0-1.0'}:raise ValueError('Split/provenance differs')
            if text_hash(row['text'])!=row['text_sha256']:raise ValueError('Exact Unicode label changed')
            if row['recipe_expression'] not in expressions(split) or row['family']!=family(row['recipe_expression']):raise ValueError('Actual grammar AST differs')
            for key,inventory in ((row['group_id'],groups),(row['family'],families),(row['text_sha256'] if row['text'] else None,texts)):
                if key is not None:
                    if key in inventory and inventory[key]!=split:raise ValueError('Cross-split label/group/family leakage')
                    inventory[key]=split
            path=(root/row['image']).resolve()
            if not path.is_relative_to(root) or sha(path)!=row['sha256']:raise ValueError('Image path/hash changed')
            with Image.open(path) as image:
                image.load();pixel=pixels(image)
                if image.size!=(row['width'],row['height']) or pixel!=row['pixel_sha256']:raise ValueError('Decoded pixels changed')
            if pixel in pixelsets and pixelsets[pixel]!=split:raise ValueError('Cross-split pixels')
            pixelsets[pixel]=split
            if row['text'] and (not row['font_id'] or row['rendering']['glyphs']<=0):raise ValueError('Missing genuine glyphs')
            if split=='train':tokens.update(row['text'])
        count_groups=len({r['group_id'] for r in rows});positive=len({r['group_id'] for r in rows if r['text']})
        if count_groups!=info['groups'] or positive!=info['positive_groups']:raise ValueError('Group count differs')
        counts[split]={'count':len(rows),'positive_groups':positive,'groups':count_groups}
    if sorted(tokens,key=ord)!=meta['train_vocabulary']:raise ValueError('Vocabulary must derive from train only')
    binding={'schema_version':'localocr.pretrained.freeze.v1','dataset_sha256':sha(root/'dataset.json'),
             'split_sha256':{s:meta['splits'][s]['sha256'] for s in SPLITS},'counts':counts,'license':'CC0-1.0'}
    if (root/'freeze.json').exists() and read(root/'freeze.json')!=binding:raise ValueError('Freeze changed')
    if (root/'heldout-seal.json').exists():
        seal=read(root/'heldout-seal.json')
        if seal['dataset_sha256']!=binding['dataset_sha256'] or seal['heldout_sha256']!=binding['split_sha256']['heldout']:raise ValueError('Heldout seal differs')
    return binding


def generate(output,fonts,counts=(8000,1000,2000),variants=(2,2,2),seed=20261012):
    if len(counts)!=3 or len(variants)!=3 or any(not isinstance(n,int) or isinstance(n,bool) or n<=0 for n in (*counts,*variants)):
        raise ValueError('Positive split counts/variant counts required')
    root=output_root(output)
    if root.exists():raise ValueError('Fresh output required')
    storage_gate(root,1024**2);renderer=MixedRenderer(fonts);seen,old_bindings=previous_exclusions()
    excluded_prior_count=len(seen)
    root.mkdir(parents=True);(root/'images').mkdir();writer=Writer(root)
    started=time.perf_counter();splits={};allpixels={};exclusions=Counter();tokens=set();summaries={}
    atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'rendering','started_utc':utc(),'counts':list(counts),'seed':seed,'completed_splits':[]})
    for split_index,split in enumerate(SPLITS):
        rows=[];rng=random.Random(seed+split_index);kind_counts=Counter()
        for index in range(counts[split_index]):
            for attempt in range(1000):
                item=make_sample(split,index,rng,seen);rendered=[]
                try:
                    for view in range(variants[split_index]):
                        size=(20,24,28,32)[index%4]
                        if item['text']:
                            image,region,used,evidence=render_variant(renderer,item['text'],view,size)
                            transform={'size':size,'view':'edge-pad-6' if view else 'raw-ink-pad-3','whitespace':'exact original advance'}
                        else:
                            image,transform=negative(split,index,view,rng);region=[0,0,image.width,image.height];used=[];evidence={'text_present':False}
                        pixel=pixels(image)
                        if pixel in allpixels and allpixels[pixel]!=split:raise Unsupported('cross-split-pixels')
                        rendered.append((image,region,used,evidence,transform,pixel))
                except Unsupported as error:exclusions[str(error)]+=1;continue
                break
            else:raise ValueError('Unable to render requested original group')
            for view,(image,region,used,evidence,transform,pixel) in enumerate(rendered):
                identifier=f'{split}-{index:06d}-{view}';path=root/'images'/(identifier+'.png')
                writer.image(path,image);allpixels[pixel]=split
                rows.append(dict(item,id=identifier,split=split,image=path.relative_to(root).as_posix(),font_id=used,
                                 pixel_sha256=pixel,width=image.width,height=image.height,region=region,sha256=sha(path),
                                 provenance={'source':'original','license':'CC0-1.0'},rendering=evidence,transform=transform))
            kind_counts[item['kind']]+=1
            if split=='train':tokens.update(item['text'])
            if (index+1)%100==0:
                storage_gate(root,128*1024)
                atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'rendering','split':split,'groups':index+1,'target':counts[split_index],'completed_splits':list(splits),'elapsed_seconds':time.perf_counter()-started,'updated_utc':utc()})
            if (index+1)%1000==0:print(json.dumps({'split':split,'groups':index+1,'elapsed_seconds':round(time.perf_counter()-started,2)}),flush=True)
        writer.json(root/(split+'.json'),rows)
        positive=len({r['group_id'] for r in rows if r['text']})
        splits[split]={'file':split+'.json','sha256':sha(root/(split+'.json')),'count':len(rows),
                       'positive_groups':positive,'negative_groups':counts[split_index]-positive,'groups':counts[split_index],
                       'variants':variants[split_index]}
        summaries[split]={'groups_by_kind':dict(kind_counts),'images_by_kind':dict(Counter(r['kind'] for r in rows)),
                          'face_families':len({r['family'] for r in rows if r['kind']=='face'}),'full_expression_families':len({r['family'] for r in rows})}
    meta={'schema_version':'localocr.pretrained.dataset.v1','generator_id':GENERATOR_ID,'synthetic_only':True,
          'private_inputs_consumed':False,'license':'CC0-1.0','counts_semantics':'total groups including 10 percent blank/graphic; every group retains all requested views',
          'distribution_scope':'finite original full-expression grammar; shared glyph atoms, vocabulary and OFL fonts; no natural-screenshot accuracy claim',
          'seed':seed,'splits':splits,'train_vocabulary':sorted(tokens,key=ord),
          'grammar_family_ast':{family(e):e for s in SPLITS for e in expressions(s)},
          'font_independence':False,'atomic_emoji_glyph_independence':False,
          'fonts':[{k:v for k,v in f.items() if k not in ('file','license_file')} for f in renderer.fonts],
          'font_manifest_sha256':sha(fonts),'renderer':renderer.versions,
          'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in ('quality_data.py','quality_corpus.py','data.py')},
          'renderer_sha256':sha(Path(__file__).resolve().parents[1]/'OcrDatasets/render.py'),
          'excluded_prior_label_digests':excluded_prior_count,
          'exclusion_reasons':dict(exclusions),'summary':summaries,
          'usage':{'train':'training','dev':'development-selection','heldout':'sealed-first-evaluation-after-model-and-router-selection'}}
    writer.json(root/'dataset.json',meta)
    train=read(root/'train.json');required=[len(r['text'])+sum(a==b for a,b in zip(r['text'],r['text'][1:])) for r in train]
    writer.json(root/'train-length-statistics.json',{'source':'train only','train_sha256':splits['train']['sha256'],
                 'max_label_codepoints':max(len(r['text']) for r in train),'max_ctc_required_steps':max(required),'count':len(train),
                 'basic_brackets':{''.join(pair):sum(all(c in r['text'] for c in pair) for r in train if r['kind']=='face') for pair in BRACKETS}})
    atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'validating-decoded-images','completed_splits':list(splits),'elapsed_seconds':time.perf_counter()-started,'updated_utc':utc()})
    binding=validate(root)
    writer.json(root/'heldout-seal.json',{'schema_version':'localocr.heldout-seal.v1','dataset_sha256':binding['dataset_sha256'],
                 'heldout_sha256':binding['split_sha256']['heldout'],'usage':'first evaluation after model+router dual freeze',
                 'labels_used_for_vocabulary':False,'results_seen':False})
    for relative,value in old_bindings.items():
        if sha(ARTIFACTS/relative)!=value:raise ValueError('Previous data bytes changed during generation')
    writer.json(root/'generation.json',{'schema_version':'localocr.quality-generation.v1','binding':binding,'elapsed_seconds':time.perf_counter()-started,
                 'old_inputs_sha256':old_bindings,'old_inputs_unchanged':True,'CPU_model_inference':False,'GPU_used':False,
                 'font_files_distributed':False,'private_inputs_consumed':False,'exclusions':dict(exclusions),'summary':summaries,
                 'storage':storage_gate(root)})
    atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'complete','dataset_sha256':binding['dataset_sha256'],'elapsed_seconds':time.perf_counter()-started,'completed_utc':utc()})
    # Freeze is the successful commit point; failed metadata/seal writes cannot publish it.
    writer.json(root/'freeze.json',binding)
    return binding


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('generate','validate'))
    parser.add_argument('--dataset',required=True);parser.add_argument('--fonts');parser.add_argument('--counts',nargs=3,type=int,default=(8000,1000,2000))
    parser.add_argument('--variants',nargs=3,type=int,default=(2,2,2));parser.add_argument('--seed',type=int,default=20261012)
    args=parser.parse_args()
    print(json.dumps(generate(args.dataset,args.fonts,args.counts,args.variants,args.seed) if args.action=='generate' else validate(args.dataset),indent=2))

if __name__=='__main__':main()
