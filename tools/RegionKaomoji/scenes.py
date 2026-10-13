"""Original CC0 mixed scene rasters and evaluation-only visible-ink truth."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image, ImageDraw
from tools.OcrDatasets.render import Renderer, Unsupported
from tools.PretrainedOcr.corpus import CN_WORDS, EN_WORDS

ARTIFACTS=Path(__file__).resolve().parents[2]/'artifacts'
TYPES=('same-line','middle-face','different-line','low-body','small-face','low-face','ordinary-brackets','body-graphic','graphic-only','blank')
BG=(246,246,242)
RECIPES={
 'dev':{'same-line':'body gap face','middle-face':'body gap face gap body','different-line':'body newline face','low-body':'faint-body gap face','small-face':'body gap small-face','low-face':'body newline faint-face','ordinary-brackets':'body (body)','body-graphic':'body newline ellipse-right','graphic-only':'ellipse-with-diagonal','blank':'solid'},
 'heldout':{'same-line':'face gap body','middle-face':'face gap body gap face','different-line':'body newline body newline face','low-body':'face newline faint-body','small-face':'small-face newline body','low-face':'faint-face gap body','ordinary-brackets':'(body) newline body','body-graphic':'triangle-left newline body','graphic-only':'triangle-with-arc','blank':'vertical-gradient'},
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def th(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pixels(image):
 rgb=image.convert('RGB');return hashlib.sha256(rgb.width.to_bytes(8,'big')+rgb.height.to_bytes(8,'big')+rgb.tobytes()).hexdigest()
def bbox(mask):
 ys,xs=np.where(np.asarray(mask)>0)
 if not len(xs):raise ValueError('Empty visible glyph mask')
 return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
def family(split,kind):return 'scene-grammar:'+th(RECIPES[split][kind])
def advance(renderer,text,size):
 return sum(p.x_advance/font[5]*size for font,_,positions in renderer.shape(text) for p in positions)

def body(rng,split,serial):
 a,b=rng.sample(CN_WORDS,2);c,d=rng.sample(EN_WORDS,2)
 return f'{a} {c} {serial} {b}' if split=='dev' else f'{serial}: {d}/{a} {b}'
def face(rng,split,seen):
 for _ in range(10000):
  e=rng.choice(['ω','Д','°','^','•','o','T']);m=rng.choice(['▽','_','∀','益','3','皿']);s=rng.choice(['',' ','  ']);arm=rng.choice(['つ','╯','ノ',''])
  text=f'{arm}({e}{s}{m}{s}{e})'+rng.choice(['','!','つ','──']) if split=='dev' else f'[{e}{s}{m}{s}{e}]'+arm+rng.choice(['!','ノ','──',''])
  if th(text) not in seen:seen.add(th(text));return text
 raise ValueError('Face grammar exhausted')

def plan(split,kind,rng,serial,seen):
 b1=body(rng,split,serial);b2=body(rng,split,serial+100000)
 normal=lambda t:dict(kind='body',text=t,size=24,contrast=1.)
 facial=lambda size=24,contrast=1.:dict(kind='kaomoji',text=face(rng,split,seen),size=size,contrast=contrast)
 if kind=='same-line':return [[normal(b1),facial()]] if split=='dev' else [[facial(),normal(b1)]]
 if kind=='middle-face':return [[normal(b1),facial(),normal(b2)]] if split=='dev' else [[facial(),normal(b1),facial()]]
 if kind=='different-line':return [[normal(b1)],[facial()]] if split=='dev' else [[normal(b1)],[normal(b2)],[facial()]]
 if kind=='low-body':
  low=normal(b1);low['contrast']=.14
  return [[low,facial()]] if split=='dev' else [[facial()],[low]]
 if kind=='small-face':return [[normal(b1),facial(12)]] if split=='dev' else [[facial(12)],[normal(b1)]]
 if kind=='low-face':return [[normal(b1)],[facial(20,.13)]] if split=='dev' else [[facial(20,.13),normal(b1)]]
 if kind=='ordinary-brackets':return [[normal(b1+' ('+b2+')')]] if split=='dev' else [[normal('('+b1+')')],[normal(b2)]]
 if kind=='body-graphic':return [[normal(b1)]]
 return []

def render_scene(renderer,split,kind,rng,serial,seen):
 lines=plan(split,kind,rng,serial,seen)
 widths=[sum(advance(renderer,s['text'],s['size']) for s in line)+18*(len(line)-1) for line in lines]
 width=max(380,int(max(widths,default=200))+160);height=max(170,80+len(lines)*70)
 if kind=='graphic-only':width+=serial%61;height+=serial%37
 canvas=Image.new('RGB',(width,height),BG);regions=[];whole='';glyphs=[]
 for li,line in enumerate(lines):
  if li:whole+='\n'
  cursor=72.
  for si,segment in enumerate(line):
   if si:whole+='  ';cursor+=18
   start=len(whole);whole+=segment['text']
   tile,_,fonts,evidence=renderer.render(segment['text'],size=segment['size'])
   arr=np.asarray(tile).astype(float);arr=np.rint(np.array(BG)+(arr-np.array(BG))*segment['contrast']).astype('uint8');tile=Image.fromarray(arr)
   local=np.any(arr!=np.array(BG),axis=2).astype('uint8')*255;localmask=Image.fromarray(local)
   x=round(cursor-segment['size']*2);y=30+li*70
   if x<0 or y+tile.height>height or x+tile.width>width:raise Unsupported('scene-clipping')
   canvas.paste(tile,(x,y),localmask)
   mask=Image.new('L',canvas.size);mask.paste(localmask,(x,y))
   regions.append({'kind':segment['kind'],'text':segment['text'],'span':[start,len(whole)],'bbox':bbox(mask),'font_ids':fonts,'rendering':evidence,'contrast':segment['contrast'],'size':segment['size']})
   glyphs.append(mask);cursor+=advance(renderer,segment['text'],segment['size'])
 if kind in ('body-graphic','graphic-only'):
  draw=ImageDraw.Draw(canvas);mask=Image.new('L',canvas.size);md=ImageDraw.Draw(mask)
  box=[width-125,height-80,width-25,height-20]
  if split=='dev':
   draw.ellipse(box,outline=(25,40,70),width=3);md.ellipse(box,outline=255,width=3)
   draw.line((box[0],box[1],box[2],box[3]),fill=(25,40,70),width=2);md.line((box[0],box[1],box[2],box[3]),fill=255,width=2)
  else:
   points=[(box[0],box[3]),((box[0]+box[2])//2,box[1]),(box[2],box[3])]
   draw.polygon(points,outline=(25,40,70),width=3);md.polygon(points,outline=255,width=3)
   draw.arc(box,0,180,fill=(25,40,70),width=2);md.arc(box,0,180,fill=255,width=2)
  regions.append({'kind':'graphic','text':'','span':None,'bbox':bbox(mask),'font_ids':[],'rendering':{'text_present':False}});glyphs.append(mask)
 if kind=='blank':
  bg=tuple(rng.randrange(210,251) for _ in range(3));canvas=Image.new('RGB',(width+serial%31,height+serial%17),bg)
  if split=='heldout':
   d=ImageDraw.Draw(canvas)
   for y in range(canvas.height):d.line((0,y,canvas.width,y),fill=tuple(max(0,c-y//12) for c in bg))
 return canvas,whole,regions,glyphs

def validate(root):
 root=Path(root).resolve();meta=read(root/'dataset.json')
 if meta.get('generator_id')=='localocr.quality-scenes.20261012.v1':
  from .quality_scenes import validate as quality_validate
  return quality_validate(root)
 if meta.get('generator_id')=='localocr.grammar-scenes.20261013.v1':
  from .grammar_scenes import validate as grammar_validate
  return grammar_validate(root)
 if meta.get('generator_id') in ('localocr.card-scenes.20261013.v1','localocr.card-scenes.20261013.v2'):
  from .card_scenes import validate as card_validate
  return card_validate(root)
 texts={};micros={};families={};pixelsets={};counts={}
 if meta.get('schema_version')!='localocr.region-scenes.v1' or meta.get('synthetic_only') is not True:raise ValueError('Scene schema/synthetic declaration')
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
 root=Path(output).resolve()
 if str(output).startswith(('\\\\','//')) or not root.is_relative_to(ARTIFACTS.resolve()) or root.exists():raise ValueError('Fresh local artifacts required')
 if shutil.disk_usage(ARTIFACTS).free<8*1024**3:raise ValueError('Disk below8GiB')
 renderer=Renderer(fonts);root.mkdir(parents=True);(root/'images').mkdir();(root/'masks').mkdir();seen=set();allpixels=set();splits={};exclusions=collections.Counter();started=time.perf_counter()
 for split,target in zip(('dev','heldout'),counts):
  rng=random.Random(seed+(0 if split=='dev' else 1));rows=[];attempt=0
  while len(rows)<target and attempt<target*3:
   index=len(rows);kind=TYPES[index%len(TYPES)];attempt+=1;serial=(1000000 if split=='dev' else 7000000)+attempt
   try:canvas,text,regions,masks=render_scene(renderer,split,kind,rng,serial,seen)
   except Unsupported as exc:exclusions[str(exc)]+=1;continue
   pixel=pixels(canvas)
   if pixel in allpixels:exclusions['duplicate-pixels']+=1;continue
   allpixels.add(pixel);identifier=f'{split}-{index:04d}';image=root/'images'/(identifier+'.png');canvas.save(image,compress_level=3)
   for ri,(region,mask) in enumerate(zip(regions,masks)):
    path=root/'masks'/f'{identifier}-{ri}.png';mask.save(path,compress_level=3);region.update(id=f'{identifier}-region-{ri}',mask=path.relative_to(root).as_posix(),mask_sha256=sha(path),pixel_count=int((np.asarray(mask)>0).sum()))
   rows.append({'id':identifier,'split':split,'source_split':split,'scene_kind':kind,'recipe_family':family(split,kind),'text':text,'text_sha256':th(text),'image':image.relative_to(root).as_posix(),'sha256':sha(image),'pixel_sha256':pixel,'width':canvas.width,'height':canvas.height,'regions':regions,'provenance':{'source':'original','license':'CC0-1.0'},'baseline_omission':'not measured; small/faint face is stress evidence only'})
  if len(rows)!=target:raise ValueError('Unable to reach requested reliable scene count')
  write(root/(split+'.json'),rows);splits[split]={'file':split+'.json','sha256':sha(root/(split+'.json')),'count':len(rows),'requested':target}
 write(root/'dataset.json',{'schema_version':'localocr.region-scenes.v1','synthetic_only':True,'private_inputs_consumed':False,'license':'CC0-1.0','seed':seed,'splits':splits,'recipes':RECIPES,'renderer':renderer.versions,'fonts':[{k:v for k,v in f.items() if k not in ('file','license_file')} for f in renderer.fonts],'font_independence':False,'vocabulary_independence':False,'bounds':'original-image visible-ink half-open [left,top,right,bottom)','spans':'Python Unicode codepoint indices [start,end); scene reading order with two spaces between same-line chunks and LF between lines','source_sha256':sha(__file__),'dependency_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in ['OcrDatasets/render.py','PretrainedOcr/corpus.py']},'exclusions':dict(exclusions),'usage':{'dev':'routing/fusion development','heldout':'first CPU evaluation after model+router+fusion selection freeze'}})
 binding=validate(root);write(root/'freeze.json',binding);write(root/'heldout-seal.json',{'schema_version':'localocr.region-scenes.seal.v1','dataset_sha256':binding['dataset_sha256'],'heldout_sha256':binding['split_sha256']['heldout'],'inference_results_seen':False,'requires_frozen_pipeline_selection':True})
 write(root/'generation.json',{'binding':binding,'elapsed_seconds':time.perf_counter()-started,'scene_types':{s:dict(collections.Counter(r['scene_kind'] for r in read(root/(s+'.json')))) for s in ('dev','heldout')},'exclusions':dict(exclusions),'CPU_model_inference':False,'GPU_used':False,'private_inputs':False,'fonts_distributed':False,'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'free_bytes':shutil.disk_usage(root).free})
 return binding

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','validate']);p.add_argument('--dataset',required=True);p.add_argument('--fonts');p.add_argument('--counts',nargs=2,type=int,default=[200,400]);a=p.parse_args();print(json.dumps(generate(a.dataset,a.fonts,a.counts) if a.action=='generate' else validate(a.dataset),indent=2))
if __name__=='__main__':main()
