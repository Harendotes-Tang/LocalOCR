"""Original CC0 mixed quality scenes with measurement-only visible-ink truth.

Only the separate six-field image index is valid for inference. Coordinates,
labels, spans and masks are used for evaluation after model outputs are produced.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageDraw
from tools.OcrDatasets.render import Renderer, Unsupported
from tools.PretrainedOcr.quality_corpus import (BRACKETS, make_sample, text_hash,
                                               CN_WORDS, EN_WORDS)
from tools.PretrainedOcr.quality_data import (Writer, output_root, storage_gate,
                                            render_variant)
from tools.PretrainedOcr.common import write as atomic_write, utc
from .scenes import sha, th, write, read, pixels, bbox, BG

GENERATOR_ID='localocr.quality-scenes.20261012.v1'
TYPES=('same-line','middle-face','different-line','low-body','small-face','low-face',
       'ordinary-brackets','body-graphic','graphic-only','blank','double-face',
       'tight-body','edge-face','square-face','curly-face','fullwidth-face',
       'letter-brackets','ordinary-numbers','table-lines','combining-body')
RECIPES={
 'dev':{
 'same-line':'body gap face','middle-face':'body gap face gap body',
 'different-line':'body newline face','low-body':'faint-body gap face',
 'small-face':'body gap small-face','low-face':'body newline faint-face',
 'ordinary-brackets':'body (body)','body-graphic':'body newline single-geometry',
 'graphic-only':'single(ellipse|triangle|box|table)-diagonal','blank':'solid-rgb',
 'double-face':'body gap face gap face','tight-body':'body zero-gap face',
 'edge-face':'body newline left-edge-face','square-face':'body gap square-face',
 'curly-face':'body newline curly-face','fullwidth-face':'body gap fullwidth-face',
 'letter-brackets':'body [o] (T)','ordinary-numbers':'body [123] (number)',
 'table-lines':'body newline table-grid','combining-body':'combining-body gap face'},
 'heldout':{
 'same-line':'face gap body','middle-face':'face gap body gap face',
 'different-line':'face newline body newline body','low-body':'face newline faint-body',
 'small-face':'small-face newline body','low-face':'faint-face gap body',
 'ordinary-brackets':'[body] newline body','body-graphic':'paired-geometry newline body',
 'graphic-only':'paired(ellipse|triangle|box|table)-arc','blank':'vertical-linear-rgb',
 'double-face':'face gap face newline body','tight-body':'face zero-gap body',
 'edge-face':'left-edge-face newline body','square-face':'square-face newline body',
 'curly-face':'curly-face gap body','fullwidth-face':'fullwidth-face newline body',
 'letter-brackets':'(o) [x] body','ordinary-numbers':'(number) [123] body',
 'table-lines':'table-grid newline body','combining-body':'face newline combining-body'},
}
JOB_KEYS=('id','image','sha256','pixel_sha256','width','height')


def family(split,kind):return 'quality-scene-grammar:'+th(RECIPES[split][kind])


def face(rng,split,seen,bracket=None):
    for _ in range(10000):
        item=make_sample(split,0,rng,seen);text=item['text']
        if bracket is not None:
            positions=[(i,c) for i,c in enumerate(text) if c in {ch for pair in BRACKETS for ch in pair}]
            if len(positions)!=2:raise ValueError('Generated face bracket structure')
            left,right=bracket;a,b=positions[0][0],positions[1][0]
            text=text[:a]+left+text[a+1:b]+right+text[b+1:]
        digest=th(text)
        if bracket is not None and digest in seen and text!=item['text']:continue
        seen.add(digest);return text
    raise ValueError('Original scene expression pool exhausted')


def body(rng,split,seen):
    for _ in range(10000):
        a,b=rng.sample(CN_WORDS,2);c,d=rng.sample(EN_WORDS,2)
        text=f'{a} {c} {b}' if split=='dev' else f'{d}/{a} {b}'
        if th(text) not in seen:seen.add(th(text));return text
    raise ValueError('Original scene body pool exhausted')


def plan(split,kind,rng,seen):
    b1=body(rng,split,seen);b2=body(rng,split,seen)
    normal=lambda text:dict(kind='body',text=text,size=24,contrast=1.)
    facial=lambda size=24,contrast=1.,bracket=None:dict(kind='kaomoji',text=face(rng,split,seen,bracket),size=size,contrast=contrast)
    if kind=='same-line':return [[normal(b1),facial()]] if split=='dev' else [[facial(),normal(b1)]]
    if kind=='middle-face':return [[normal(b1),facial(),normal(b2)]] if split=='dev' else [[facial(),normal(b1),facial()]]
    if kind=='different-line':return [[normal(b1)],[facial()]] if split=='dev' else [[facial()],[normal(b1)],[normal(b2)]]
    if kind=='low-body':
        low=normal(b1);low['contrast']=.14
        return [[low,facial()]] if split=='dev' else [[facial()],[low]]
    if kind=='small-face':return [[normal(b1),facial(12)]] if split=='dev' else [[facial(12)],[normal(b1)]]
    if kind=='low-face':return [[normal(b1)],[facial(20,.13)]] if split=='dev' else [[facial(20,.13),normal(b1)]]
    if kind=='double-face':return [[normal(b1),facial(),facial()]] if split=='dev' else [[facial(),facial()],[normal(b1)]]
    if kind=='tight-body':return [[normal(b1),facial()]] if split=='dev' else [[facial(),normal(b1)]]
    if kind=='edge-face':return [[normal(b1)],[facial()]] if split=='dev' else [[facial()],[normal(b1)]]
    if kind in ('square-face','curly-face','fullwidth-face'):
        bracket=BRACKETS[1] if kind=='square-face' else BRACKETS[2] if kind=='curly-face' else rng.choice(BRACKETS[3:])
        return [[normal(b1),facial(bracket=bracket)]] if split=='dev' and kind!='curly-face' else [[normal(b1)],[facial(bracket=bracket)]] if split=='dev' else [[facial(bracket=bracket)],[normal(b1)]] if kind!='curly-face' else [[facial(bracket=bracket),normal(b1)]]
    if kind=='ordinary-brackets':return [[normal(b1+' ('+b2+')')]] if split=='dev' else [[normal('['+b1+']')],[normal(b2)]]
    if kind=='letter-brackets':return [[normal(b1+' [o] (T)')]] if split=='dev' else [[normal('(o) [x] '+b1)]]
    if kind=='ordinary-numbers':
        value=str(rng.randrange(100,10000))
        return [[normal(b1+' [123] ('+value+')')]] if split=='dev' else [[normal('('+value+') [123] '+b1)]]
    if kind=='combining-body':
        text='café '+b1+' ō' if split=='dev' else 'ō/'+b1+'/café'
        return [[normal(text),facial()]] if split=='dev' else [[facial()],[normal(text)]]
    if kind in ('body-graphic','table-lines'):return [[normal(b1)]]
    return []


def render_scene(renderer,split,kind,rng,index,seen):
    lines=plan(split,kind,rng,seen);tiles=[]
    gap=0 if kind=='tight-body' else 18
    for line in lines:
        local=[]
        for segment in line:
            tile,_,fonts,evidence=render_variant(renderer,segment['text'],0,segment['size'])
            array=np.asarray(tile).astype(float)
            array=np.rint(np.array(BG)+(array-np.array(BG))*segment['contrast']).astype('uint8')
            mask=Image.fromarray(np.any(array!=np.array(BG),axis=2).astype('uint8')*255)
            local.append((segment,Image.fromarray(array),mask,fonts,evidence))
        tiles.append(local)
    widths=[sum(tile.width for _,tile,*_ in line)+gap*(len(line)-1) for line in tiles]
    width=max(380,max(widths,default=220)+96);height=max(170,70+len(tiles)*64)
    if kind=='graphic-only':width+=index%31;height+=index%23
    if split=='heldout' and kind in ('body-graphic','table-lines'):height=max(height,240)
    canvas=Image.new('RGB',(width,height),BG);whole='';regions=[];masks=[]
    for li,line in enumerate(tiles):
        if li:whole+='\n'
        cursor=3 if kind=='edge-face' and line and line[0][0]['kind']=='kaomoji' else 48
        for si,(segment,tile,mask,fonts,evidence) in enumerate(line):
            if si:whole+='  ';cursor+=gap
            start=len(whole);whole+=segment['text'];x=cursor;y=150+li*64 if split=='heldout' and kind in ('body-graphic','table-lines') else 28+li*64
            if x<0 or x+tile.width>width or y+tile.height>height:raise Unsupported('scene-clipping')
            canvas.paste(tile,(x,y),mask);globalmask=Image.new('L',canvas.size);globalmask.paste(mask,(x,y))
            regions.append({'kind':segment['kind'],'text':segment['text'],'span':[start,len(whole)],
                            'bbox':bbox(globalmask),'font_ids':fonts,'rendering':evidence,
                            'contrast':segment['contrast'],'size':segment['size']})
            masks.append(globalmask);cursor+=tile.width
    if kind in ('body-graphic','graphic-only','table-lines'):
        mask=Image.new('L',canvas.size);draw=ImageDraw.Draw(canvas);md=ImageDraw.Draw(mask)
        box=[width-170,20,width-24,78] if split=='heldout' and kind in ('body-graphic','table-lines') else [width-170,height-72,width-24,height-15];ink=(25,40,70)
        primitive=('ellipse','triangle','box','table')[index//len(TYPES)%4]
        if kind=='table-lines':primitive='table'
        def both(method,*args,**kwargs):
            getattr(draw,method)(*args,fill=ink,**kwargs);getattr(md,method)(*args,fill=255,**kwargs)
        if primitive=='ellipse':
            draw.ellipse(box,outline=ink,width=2);md.ellipse(box,outline=255,width=2)
        elif primitive=='triangle':
            points=[(box[0],box[3]),((box[0]+box[2])//2,box[1]),(box[2],box[3])]
            draw.polygon(points,outline=ink,width=2);md.polygon(points,outline=255,width=2)
        else:
            draw.rectangle(box,outline=ink,width=1);md.rectangle(box,outline=255,width=1)
            if primitive=='table':
                for x in range(box[0]+20,box[2],20):both('line',(x,box[1],x,box[3]),width=1)
                for y in range(box[1]+16,box[3],16):both('line',(box[0],y,box[2],y),width=1)
        if split=='dev':both('line',(box[0],box[1],box[2],box[3]),width=1)
        else:
            draw.arc(box,0,180,fill=ink,width=2);md.arc(box,0,180,fill=255,width=2)
        regions.append({'kind':'graphic','text':'','span':None,'bbox':bbox(mask),'font_ids':[],
                        'rendering':{'text_present':False},'geometry':primitive});masks.append(mask)
    if kind=='blank':
        bg=tuple(rng.randrange(210,251) for _ in range(3));canvas=Image.new('RGB',(width+index%31,height+index%17),bg)
        if split=='heldout':
            draw=ImageDraw.Draw(canvas)
            for y in range(canvas.height):draw.line((0,y,canvas.width-1,y),fill=tuple(c-y//12 for c in bg))
    return canvas,whole,regions,masks


def validate(root):
 root=Path(root).resolve();meta=read(root/'dataset.json');texts={};micros={};families={};pixelsets={};counts={}
 if meta.get('schema_version')!='localocr.region-scenes.v1' or meta.get('synthetic_only') is not True or meta.get('generator_id')!=GENERATOR_ID:raise ValueError('Scene schema/synthetic declaration')
 for split in ('dev','heldout'):
  info=meta['splits'][split];relative=Path(info['file'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Split path escape')
  if sha(root/relative)!=info['sha256']:raise ValueError('Split hash changed')
  rows=read(root/relative)
  if len(rows)!=info['count']:raise ValueError('Scene count changed')
  for row in rows:
   if row['split']!=split or row['source_split']!=split or row['recipe_family']!=family(split,row['scene_kind']):raise ValueError('Scene source/AST family differs')
   if row['provenance']!={'source':'original','license':'CC0-1.0'} or th(row['text'])!=row['text_sha256']:raise ValueError('Label/provenance changed')
   for key,inventory in [(row['text_sha256'] if row['text'] else None,texts),(row['recipe_family'],families)]:
    if key is not None:
     if key in inventory and inventory[key]!=split:raise ValueError('Cross-split label/family')
     inventory[key]=split
   p=(root/row['image']).resolve()
   if not p.is_relative_to(root) or sha(p)!=row['sha256']:raise ValueError('Image hash/path')
   with Image.open(p) as im:
    im.load();arr=np.asarray(im.convert('RGB'));pixel=pixels(im)
    if im.size!=(row['width'],row['height']) or pixel!=row['pixel_sha256']:raise ValueError('Scene pixels/dimensions')
   if pixel in pixelsets:raise ValueError('Duplicate scene pixels')
   pixelsets[pixel]=split
   masks=[]
   for region in row['regions']:
    mp=(root/region['mask']).resolve()
    if not mp.is_relative_to(root) or sha(mp)!=region['mask_sha256']:raise ValueError('Mask hash/path')
    with Image.open(mp) as m:
     m.load();support=np.asarray(m)>0
     if m.mode!='L' or m.size!=(row['width'],row['height']) or bbox(m)!=region['bbox'] or int(support.sum())!=region['pixel_count']:raise ValueError('Actual mask/bbox pixels differ')
    if region['kind']!='graphic':
     a,b=region['span']
     if not 0<=a<b<=len(row['text']) or row['text'][a:b]!=region['text']:raise ValueError('Exact character span')
     key=th(region['text'])
     if key in micros and micros[key]!=split:raise ValueError('Micro-label source leakage')
     micros[key]=split
     if region['rendering']['glyphs']<=0 or not region['font_ids']:raise ValueError('Missing genuine glyphs')
     if not np.all(np.any(arr[support]!=np.array(BG),axis=1)):raise ValueError('No visible source image ink')
    masks.append(support)
   # No glyph pixel is shared by another region; prevents body hidden under faces.
   if masks and np.any(np.sum(masks,axis=0)>1):raise ValueError('Region glyph pixels overlap')
   if row['scene_kind']!='blank' and not np.array_equal(np.any(masks,axis=0) if masks else np.zeros(arr.shape[:2],bool),np.any(arr!=np.array(BG),axis=2)):
    raise ValueError('Truth masks do not cover all actual visible ink')
  counts[split]={'scenes':len(rows),'faces':sum(r['kind']=='kaomoji' for row in rows for r in row['regions'])}
 binding={'schema_version':'localocr.region-scenes.freeze.v1','dataset_sha256':sha(root/'dataset.json'),'split_sha256':{s:meta['splits'][s]['sha256'] for s in ('dev','heldout')},'counts':counts}
 if (root/'freeze.json').exists() and read(root/'freeze.json')!=binding:raise ValueError('Scene freeze differs')
 if (root/'heldout-seal.json').exists():
  seal=read(root/'heldout-seal.json')
  if seal['dataset_sha256']!=binding['dataset_sha256'] or seal['heldout_sha256']!=binding['split_sha256']['heldout']:raise ValueError('Scene seal differs')
 return binding


def generate(output,fonts,counts=(200,400),seed=20261012):
    if len(counts)!=2 or any(not isinstance(n,int) or isinstance(n,bool) or n<=0 for n in counts):raise ValueError('Positive scene counts required')
    root=output_root(output)
    if root.exists():raise ValueError('Fresh scene output required')
    storage_gate(root,1024**2);renderer=Renderer(fonts)
    root.mkdir(parents=True);(root/'images').mkdir();(root/'masks').mkdir();writer=Writer(root)
    started=time.perf_counter();seen=set();allpixels=set();splits={};exclusions=Counter();summary={};jobs={}
    atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'rendering','started_utc':utc()})
    for split,target in zip(('dev','heldout'),counts):
        rng=random.Random(seed+(0 if split=='dev' else 1));rows=[]
        for index in range(target):
            kind=TYPES[index%len(TYPES)]
            for attempt in range(1000):
                try:canvas,text,regions,masks=render_scene(renderer,split,kind,rng,index,seen)
                except Unsupported as error:exclusions[str(error)]+=1;continue
                pixel=pixels(canvas)
                if pixel in allpixels:exclusions['duplicate-pixels']+=1;continue
                break
            else:raise ValueError('Unable to render reliable original scene')
            allpixels.add(pixel);identifier=f'{split}-{index:04d}';path=root/'images'/(identifier+'.png');writer.image(path,canvas)
            for ri,(region,mask) in enumerate(zip(regions,masks)):
                mp=root/'masks'/f'{identifier}-{ri}.png';writer.image(mp,mask)
                region.update(id=f'{identifier}-region-{ri}',mask=mp.relative_to(root).as_posix(),mask_sha256=sha(mp),pixel_count=int((np.asarray(mask)>0).sum()))
            rows.append({'id':identifier,'split':split,'source_split':split,'scene_kind':kind,'recipe_family':family(split,kind),
                         'text':text,'text_sha256':th(text),'image':path.relative_to(root).as_posix(),'sha256':sha(path),'pixel_sha256':pixel,
                         'width':canvas.width,'height':canvas.height,'regions':regions,'provenance':{'source':'original','license':'CC0-1.0'},
                         'baseline_omission':'not measured; synthetic pressure only'})
            if (index+1)%50==0:
                storage_gate(root,128*1024);atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'rendering','split':split,'scenes':index+1,'target':target,'elapsed_seconds':time.perf_counter()-started,'updated_utc':utc()})
        writer.json(root/(split+'.json'),rows);splits[split]={'file':split+'.json','sha256':sha(root/(split+'.json')),'count':len(rows),'requested':target}
        jobs[split]=[{k:row[k] for k in JOB_KEYS} for row in rows]
        summary[split]={'scene_types':dict(Counter(r['scene_kind'] for r in rows)),'faces':sum(region['kind']=='kaomoji' for row in rows for region in row['regions'])}
    writer.json(root/'dataset.json',{'schema_version':'localocr.region-scenes.v1','generator_id':GENERATOR_ID,'synthetic_only':True,
                'private_inputs_consumed':False,'license':'CC0-1.0','seed':seed,'splits':splits,'recipes':RECIPES,
                'renderer':renderer.versions,'fonts':[{k:v for k,v in f.items() if k not in ('file','license_file')} for f in renderer.fonts],
                'font_manifest_sha256':sha(fonts),'font_independence':False,'vocabulary_independence':False,
                'source_sha256':sha(__file__),'dependency_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in ['OcrDatasets/render.py','PretrainedOcr/quality_corpus.py','PretrainedOcr/quality_data.py','RegionKaomoji/scenes.py']},
                'bounds':'original-image visible-ink half-open [left,top,right,bottom)',
                'spans':'raw Python Unicode codepoint spans [start,end); reading-order separator two spaces/LF; separators are not regions',
                'exclusions':dict(exclusions),'summary':summary,'usage':{'dev':'routing/fusion development','heldout':'first CPU evaluation after model+router+fusion selection freeze'}})
    atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'validating-decoded-images-and-masks','updated_utc':utc()})
    binding=validate(root)
    writer.json(root/'heldout-seal.json',{'schema_version':'localocr.region-scenes.seal.v1','dataset_sha256':binding['dataset_sha256'],
                'heldout_sha256':binding['split_sha256']['heldout'],'inference_results_seen':False,'requires_frozen_pipeline_selection':True})
    writer.json(root/'inference-inputs.json',{'schema_version':'localocr.region-scenes.inputs.v1','dataset_sha256':binding['dataset_sha256'],
                'split_sha256':binding['split_sha256'],'jobs':jobs})
    writer.json(root/'generation.json',{'binding':binding,'elapsed_seconds':time.perf_counter()-started,'summary':summary,'exclusions':dict(exclusions),
                'CPU_model_inference':False,'GPU_used':False,'private_inputs':False,'fonts_distributed':False,'storage':storage_gate(root)})
    atomic_write(root/'progress.json',{'pid':os.getpid(),'phase':'complete','completed_utc':utc(),'dataset_sha256':binding['dataset_sha256']})
    # Publish the freeze only after all generation receipts, index and seal succeed.
    writer.json(root/'freeze.json',binding)
    return binding


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('generate','validate'));parser.add_argument('--dataset',required=True)
    parser.add_argument('--fonts');parser.add_argument('--counts',nargs=2,type=int,default=(200,400));parser.add_argument('--seed',type=int,default=20261012)
    args=parser.parse_args();print(json.dumps(generate(args.dataset,args.fonts,args.counts,args.seed) if args.action=='generate' else validate(args.dataset),indent=2))

if __name__=='__main__':main()
