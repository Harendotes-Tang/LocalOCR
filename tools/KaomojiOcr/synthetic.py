"""Synthetic generator with disjoint text splits and checked font coverage."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import string
import unicodedata
from pathlib import Path
from fontTools.ttLib import TTFont
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import numpy as np
from common import local_output,sha256,write_json,dataset_fingerprint

FACE_GLYPHS="Ддωづつ∀▽△°´｀￣̅̄•●Σ∑益皿ಥ╯╰ノヮ"
PUNCT="()（）[]{}<>~^_-;:,.!?/\\|=+*'\" "+chr(96)
HAN="普通文字没有表情测试日期等级记录副本通关结果需要确认今天很好"
ALPHABET="".join(sorted(set(string.ascii_letters+string.digits+FACE_GLYPHS+PUNCT+HAN)))
FONT_NAMES=["arial.ttf","arialbd.ttf","calibri.ttf","calibrib.ttf","segoeui.ttf","segoeuib.ttf"]
HOLDOUT_FONTS=["tahoma.ttf","tahomabd.ttf","verdana.ttf"]
FALLBACK_FONTS=["seguisym.ttf","msyh.ttc","Nirmala.ttc"]

class Renderer:
    def __init__(self,font_dir):
        self.root=Path(font_dir);self.maps={};self.cache={}
        for name in FONT_NAMES+HOLDOUT_FONTS+FALLBACK_FONTS:
            font=TTFont(str(self.root/name),fontNumber=0)
            self.maps[name]=set(font.getBestCmap());font.close()
        missing=[c for c in ALPHABET if not any(ord(c) in m for m in self.maps.values())]
        if missing:
            raise ValueError("No font coverage: "+repr(missing))

    def font(self,name,size):
        key=(name,size)
        if key not in self.cache:
            self.cache[key]=ImageFont.truetype(str(self.root/name),size,index=0)
        return self.cache[key]

    def render(self,text,primary,rng):
        size=rng.randint(14,28)
        background=rng.choice([(46,67,37),(63,44,42),(48,62,135),(235,235,235)])
        foreground=rng.randint(215,255)
        ink=(foreground,)*3 if min(background)<128 else (rng.randint(5,60),)*3
        clusters=[]
        for c in text:
            if unicodedata.combining(c) and clusters:
                clusters[-1]+=c
            else:
                clusters.append(c)
        parts=[]
        for c in clusters:
            name=next((n for n in [primary]+FALLBACK_FONTS if all(ord(g) in self.maps[n] for g in c)),None)
            if name is None:
                raise ValueError("Missing requested glyph: "+repr(c))
            parts.append((c,self.font(name,size)))
        width=sum(float(f.getlength(c)) for c,f in parts)
        image=Image.new("RGB",(max(16,round(width)+10),size*2+8),background)
        draw=ImageDraw.Draw(image);x=5.
        for c,font in parts:
            draw.text((x,size+2),c,font=font,fill=ink,anchor="ls")
            x+=float(font.getlength(c))
        if parts:
            a=np.asarray(image)
            mask=(a.min(axis=2)>170) if min(background)<128 else (a.max(axis=2)<100)
            ys,xs=np.where(mask)
            if len(xs):
                image=image.crop((max(0,int(xs.min())-3),max(0,int(ys.min())-3),
                                  min(image.width,int(xs.max())+4),min(image.height,int(ys.max())+4)))
        factor=rng.uniform(.72,1.25)
        image=image.resize((max(8,round(image.width*factor)),max(8,round(image.height*factor))),Image.Resampling.BILINEAR)
        if rng.random()<.35:
            image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.15,.6)))
        return image

def make_texts(seed,count):
    rng=random.Random(seed);texts={}
    eyes=["o","O","T","Д","д","ω","°","•","●","^","ಥ","´","｀"]
    mouths=["ω","∀","▽","_","Д","д","3","益","皿","ヮ","-"]
    while len(texts)<count:
        if rng.random()<.75:
            opening,closing=rng.choice([("(",")"),("（","）"),("[","]")])
            eye=rng.choice(eyes);sep=rng.choice([""," ","  "])
            accent=rng.choice(["","´",chr(96),"￣","̅","̄"])
            value=(rng.choice(["","つ","づ","ノ","╯","Σ","∑"])+opening+eye+accent+sep+
                   rng.choice(mouths)+sep+eye+accent+rng.choice(["",";","!"])+closing+
                   rng.choice(["","つ","づ","ノ","╰"]))
            texts[value]="face"
        else:
            value="".join(rng.choice(HAN+string.ascii_letters+string.digits+PUNCT) for _ in range(rng.randint(4,18)))
            if value!=value.strip():
                continue
            texts[value]="plain"
    return sorted(texts.items(),key=lambda p:hashlib.sha256((str(seed)+p[0]).encode()).hexdigest())

def load_corpus(path,renderer,excluded):
    if not path:
        return [],None
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema")!=1 or not isinstance(data.get("texts"),list):
        raise ValueError("Unknown public corpus")
    accepted=[];counts=dict(private_reserved=0,unsupported_font=0,invalid=0)
    for text in sorted(set(data["texts"])):
        if text in excluded:
            counts["private_reserved"]+=1;continue
        if not text or text!=text.strip() or len(text)>60 or len(text)+sum(a==b for a,b in zip(text,text[1:]))>80 or any(unicodedata.category(c)[0]=="C" for c in text):
            counts["invalid"]+=1;continue
        if any(not any(ord(c) in m for m in renderer.maps.values()) for c in text):
            counts["unsupported_font"]+=1;continue
        accepted.append((text,"face"))
    return accepted,dict(source=data["source"],corpus_sha256=sha256(path),html_sha256=data["html_sha256"],
        accepted_texts=len(accepted),exclusions=counts)

def generate(args):
    out=local_output(args.output)
    if out.exists():
        raise ValueError("Dataset output exists; choose a new directory")
    renderer=Renderer(args.fonts)
    excluded=set()
    if args.exclude_texts:
        excluded=set(json.loads(Path(args.exclude_texts).read_text(encoding="utf-8"))["texts"])
    corpus,corpus_meta=load_corpus(args.corpus,renderer,excluded)
    wanted=args.train_texts+args.dev_texts+args.test_texts
    pool=dict(corpus)
    for text,kind in make_texts(args.seed,wanted*2):
        if text not in excluded:
            pool.setdefault(text,kind)
        if len(pool)>=wanted:
            break
    if len(pool)!=wanted:
        raise ValueError("Insufficient independent synthetic text pool")
    pool=sorted(pool.items(),key=lambda p:hashlib.sha256((str(args.seed)+p[0]).encode()).hexdigest())
    alphabet="".join(sorted(set(ALPHABET+"".join(t for t,_ in corpus))))
    splits={"train":pool[:args.train_texts],
            "dev":pool[args.train_texts:args.train_texts+args.dev_texts],
            "test":pool[args.train_texts+args.dev_texts:]}
    base_frozen=None
    if args.base_dataset:
        base=Path(args.base_dataset)
        base_frozen=dataset_fingerprint(base)
        splits={}
        existing=set()
        for name in ("train","dev","test"):
            old=json.loads((base/f"{name}.json").read_text(encoding="utf-8"))
            labels={r["label"]:r["kind"] for r in old if r["label"] and r["label"] not in excluded}
            splits[name]=list(labels.items());existing.update(labels)
        additions=sorted((p for p in corpus if p[0] not in existing),key=lambda p:hashlib.sha256((str(args.seed)+p[0]).encode()).hexdigest())
        boundary1=int(len(additions)*.7);boundary2=int(len(additions)*.8)
        for name,extra in zip(("train","dev","test"),(additions[:boundary1],additions[boundary1:boundary2],additions[boundary2:])):
            splits[name].extend(extra)
    reserved_count=len(excluded)
    drill_count=0
    if args.glyph_drills:
        trained=set("".join(t for t,_ in splits["train"]))
        occupied={t for items in splits.values() for t,_ in items}|excluded
        for character in sorted(set(alphabet)-trained):
            prefix="A"
            while prefix+character+prefix in occupied:
                prefix+="A"
            text=prefix+character+prefix
            splits["train"].append((text,"plain"));occupied.add(text);drill_count+=1
    rows={};excluded={}
    corpus_labels={t for t,_ in corpus}
    for idx,(name,items) in enumerate(splits.items()):
        rng=random.Random(args.seed+idx+100);output=out/name
        output.mkdir(parents=True);records=[];excluded[name]=set()
        for text_index,(label,kind) in enumerate(items):
            for view in range((args.corpus_views if label in corpus_labels else args.views) if name=="train" else 1):
                primary=rng.choice(HOLDOUT_FONTS if name=="test" else FONT_NAMES)
                try:
                    image=renderer.render(label,primary,rng)
                except ValueError as error:
                    if "Missing requested glyph" not in str(error):
                        raise
                    excluded[name].add(label)
                    continue
                filename=f"{text_index:05d}-{view}.png";image.save(output/filename)
                records.append(dict(path=f"{name}/{filename}",label=label,kind=kind,font=primary,sha256=sha256(output/filename)))
        for i in range(96 if name=="train" else 32):
            image=Image.new("RGB",(rng.randint(25,180),rng.randint(14,32)),rng.choice(["#304520","#623534","#ffffff"]))
            kind="blank"
            if i%2:
                kind="icon"
                draw=ImageDraw.Draw(image)
                margin=min(image.size)//4
                draw.ellipse((margin,margin,image.width-margin-1,image.height-margin-1),outline="#dac781",width=2)
            filename=f"negative-{i}.png";image.save(output/filename)
            records.append(dict(path=f"{name}/{filename}",label="",kind=kind,font=None,sha256=sha256(output/filename)))
        write_json(out/f"{name}.json",records);rows[name]=len(records)
    write_json(out/"dataset.json",dict(schema=1,seed=args.seed,alphabet=alphabet,image_rows=rows,corpus=corpus_meta,base_dataset=base_frozen,
         reserved_texts=reserved_count,glyph_drill_texts=drill_count,
         unique_texts={k:len({r["label"] for r in json.loads((out/f"{k}.json").read_text(encoding="utf-8")) if r["label"]}) for k in splits},
         excluded_unrenderable={k:len(v) for k,v in excluded.items()},train_fonts=FONT_NAMES,heldout_fonts=HOLDOUT_FONTS,
         fallback_fonts=FALLBACK_FONTS,font_hashes={n:sha256(renderer.root/n) for n in renderer.maps},
         split_sha256={n:sha256(out/f"{n}.json") for n in splits},
         synthetic_only=True,private_inputs_consumed=False))
    print(rows,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--fonts",required=True)
    p.add_argument("--seed",type=int,default=20261011);p.add_argument("--train-texts",type=int,default=2400)
    p.add_argument("--dev-texts",type=int,default=240);p.add_argument("--test-texts",type=int,default=480)
    p.add_argument("--views",type=int,default=4)
    p.add_argument("--corpus");p.add_argument("--exclude-texts");p.add_argument("--base-dataset")
    p.add_argument("--corpus-views",type=int,default=12);p.add_argument("--glyph-drills",action="store_true")
    generate(p.parse_args())
