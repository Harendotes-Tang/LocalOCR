"""Shared development-only CTC preprocessing and metrics; no production import."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import unicodedata
import numpy as np
from PIL import Image

HEIGHT, WIDTH = 40, 320

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")

def local_output(path):
    """Outputs belong to ignored artifacts; reject a path through a junction."""
    root=Path(__file__).resolve().parents[2]
    target=Path(path).resolve()
    artifact=(root/"artifacts").resolve()
    if artifact.parent!=root.resolve() or target==artifact or artifact not in target.parents:
        raise ValueError("Outputs must be inside this repository's artifacts directory")
    return target

def prepare(image):
    """Keep aspect ratio, pale glyph edges and spaces on a fixed CTC canvas."""
    a=np.asarray(image.convert("L"),dtype=np.float32)
    if not a.size:
        raise ValueError("Empty crop")
    if np.median([a[0,0],a[0,-1],a[-1,0],a[-1,-1]])>128:
        a=255-a
    bg,fg=np.percentile(a,[15,99])
    a=np.clip((a-bg)*(255/max(40.,fg-bg)),0,255).astype(np.uint8)
    image=Image.fromarray(a)
    scale=min((HEIGHT-4)/image.height,(WIDTH-8)/image.width)
    image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.BILINEAR)
    canvas=Image.new("L",(WIDTH,HEIGHT),0)
    canvas.paste(image,(4,(HEIGHT-image.height)//2))
    return np.asarray(canvas,dtype=np.float32)[None]/255.

def decode(logits, alphabet):
    """Greedy CTC without grammar, normalization or fixed whole-face classes."""
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim!=2 or logits.shape[1]!=len(alphabet)+1 or not np.isfinite(logits).all():
        raise ValueError("Invalid CTC output")
    z=logits-logits.max(axis=-1,keepdims=True)
    probs=np.exp(z); probs/=probs.sum(axis=-1,keepdims=True)
    text,scores,previous=[],[],-1
    for t,idx in enumerate(probs.argmax(axis=-1)):
        if idx and idx!=previous:
            text.append(alphabet[idx-1]);scores.append(float(probs[t,idx]))
        previous=idx
    confidence=math.exp(sum(math.log(max(s,1e-12)) for s in scores)/len(scores)) if scores else 0.
    return "".join(text),confidence

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=current
    return row[-1]

def symbols(text):
    return "".join(c for c in text if unicodedata.category(c)[0] in "PSM")

def metrics(rows,field):
    """Aggregate without private labels, source paths or per-example predictions."""
    errors=sum(edit_distance(r["label"],r[field]) for r in rows)
    count=sum(len(r["label"]) for r in rows)
    se=sum(edit_distance(symbols(r["label"]),symbols(r[field])) for r in rows)
    sc=sum(len(symbols(r["label"])) for r in rows)
    return dict(count=len(rows),exact=sum(r["label"]==r[field] for r in rows),
                character_errors=errors,characters=count,cer=errors/count if count else None,
                symbol_errors=se,symbol_characters=sc,symbol_cer=se/sc if sc else None)

def decide(synthetic,private,resources,real_chain_verified=False):
    """Fail closed: manual crop success does not prove production ROI safety."""
    def improves(part):
        a,b=part.get("baseline",{}),part.get("candidate",{})
        return (a.get("count",0)>0 and a.get("count")==b.get("count")
                and all(isinstance(v,(int,float)) and math.isfinite(v) for v in (a.get("cer"),b.get("cer"),a.get("symbol_cer"),b.get("symbol_cer")))
                and b.get("exact",0)>a.get("exact",0) and b.get("cer",1)<a.get("cer",0)
                and b.get("symbol_cer",1)<a.get("symbol_cer",0))
    gates={
        "synthetic_accuracy":improves(synthetic.get("face",{})),
        "real_accuracy":improves(private.get("face",{})),
        "no_real_regressions":(private.get("face_regressions")==0 and private.get("face_erroneous_changes")==0),
        "no_added_plain_errors":(synthetic.get("plain_regressions")==0 and private.get("plain_regressions")==0
             and synthetic.get("plain_erroneous_changes")==0 and private.get("plain_erroneous_changes")==0
             and synthetic.get("plain",{}).get("candidate",{}).get("count",0)>0
             and private.get("plain",{}).get("candidate",{}).get("count",0)>0),
        "negative_safety":(private.get("negative_count",0)>0 and synthetic.get("negative_count",0)>0
             and private.get("negative_false_outputs")==0 and synthetic.get("negative_false_outputs")==0),
        "model_size":0<resources.get("model_bytes",0)<=8*1024*1024,
        "cpu_budget":(resources.get("providers")==["CPUExecutionProvider"] and resources.get("threads")==2
             and resources.get("groups")==48 and 950000<=resources.get("probe_total_pixels",0)<=1000000
             and 0<resources.get("forty_eight_crops_seconds",999)<=2
             and resources.get("peak_method")=="windows_process_peak_wset"
             and 0<=resources.get("peak_extra_bytes",999999999)<=64*1024*1024),
        "production_region_and_lifecycle_evidence":bool(real_chain_verified),
    }
    return dict(promote=all(gates.values()),gates=gates,failed=[k for k,v in gates.items() if not v])

def dataset_fingerprint(root):
    """Bind labels, mapping and every rendered image; reject path escape or tamper."""
    root=Path(root).resolve()
    meta=json.loads((root/"dataset.json").read_text(encoding="utf-8"))
    if meta.get("schema")!=1 or not meta.get("synthetic_only"):
        raise ValueError("Unknown synthetic dataset")
    hashes={}
    for name in ("train","dev","test"):
        manifest=root/f"{name}.json"
        hashes[name]=sha256(manifest)
        rows=json.loads(manifest.read_text(encoding="utf-8"))
        if len(rows)!=meta["image_rows"][name] or not rows:
            raise ValueError("Incomplete dataset split")
        for row in rows:
            path=(root/row["path"]).resolve()
            if root not in path.parents or sha256(path)!=row["sha256"]:
                raise ValueError("Rendered image differs from frozen inventory")
    if hashes!=meta["split_sha256"]:
        raise ValueError("Split differs from generated dataset")
    return dict(dataset_sha256=sha256(root/"dataset.json"),split_sha256=hashes)
