"""Supplementary rendered examples; never count these as real screenshots."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fontTools.ttLib import TTFont
from common import local_output,metrics,sha256,write_json
from compatibility import EMOJI
from evaluate import load_candidate,compare
from synthetic import Renderer,HOLDOUT_FONTS

def main(args):
    out=local_output(args.output)
    if out.exists():
        raise ValueError("Supplement output exists")
    out.mkdir(parents=True)
    texts=json.loads(Path(args.reserved_texts).read_text(encoding="utf-8"))["texts"]
    renderer=Renderer(args.fonts);images=[];rows=[];excluded=0
    for index,text in enumerate(texts):
        for view,font in enumerate(HOLDOUT_FONTS):
            try:
                image=renderer.render(text,font,random.Random(5100+index*10+view))
            except ValueError:
                excluded+=1;continue
            images.append(image);rows.append(dict(kind="face",label=text,source="user literal rendered synthetically",view=view))
    emoji_font=Path(args.fonts)/"seguiemj.ttf"
    info=TTFont(str(emoji_font));cmap=info.getBestCmap();info.close()
    font=ImageFont.truetype(str(emoji_font),48)
    emoji_excluded=0
    for index,text in enumerate(EMOJI):
        # Variation selectors / ZWJ are shaping controls rather than visible boxes.
        if any(ord(c) not in cmap and ord(c) not in (0xFE0F,0x200D) for c in text):
            emoji_excluded+=1;continue
        image=Image.new("RGB",(max(70,int(font.getlength(text))+12),74),"#623534")
        ImageDraw.Draw(image).text((6,6),text,font=font,embedded_color=True)
        ink=np.asarray(image).max(axis=2)>130
        ys,xs=np.where(ink)
        if not len(xs):
            emoji_excluded+=1;continue
        image=image.crop((max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(image.width,int(xs.max())+4),min(image.height,int(ys.max())+4)))
        images.append(image);rows.append(dict(kind="emoji",label=text,source="Segoe emoji synthetic rendering",index=index))
    for i,image in enumerate(images):
        image.save(out/f"private-render-{i:02d}.png")
    session,alphabet,resources=load_candidate(args.candidate)
    worker_path=Path(__file__).resolve().parents[2]/"scripts/local-ai-ocr-worker.py"
    spec=importlib.util.spec_from_file_location("supplement_baseline",worker_path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    baseline=worker.create_engine(Path(args.baseline_models))
    predictions,_=compare(images,rows,session,alphabet,baseline)
    report=dict(schema=1,scope="synthetic rendering only; user literal strings excluded from all model splits; no real-image accuracy claim",
        candidate_sha256=resources["model_sha256"],reserved_manifest_sha256=sha256(args.reserved_texts),
        face_font_exclusions=excluded,emoji_font_exclusions=emoji_excluded,emoji_font_sha256=sha256(emoji_font))
    for kind in ("face","emoji"):
        part=[r for r in predictions if r["kind"]==kind]
        report[kind]={field:metrics(part,field) for field in ("baseline","candidate")}
    write_json(out/"private-predictions.json",predictions);write_json(out/"summary.json",report)
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("candidate","baseline-models","reserved-texts","fonts","output"):
        p.add_argument("--"+name,required=True)
    main(p.parse_args())