"""Frozen-model crop comparison; outputs private rows and anonymous aggregates.

This is a recognizer capability test using identical crops, not a production ROI
or automatic-correction acceptance test. A crop result cannot promote a model.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import math
import time
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps
import psutil
from common import decode,decide,local_output,metrics,prepare,sha256,write_json,dataset_fingerprint

def load_candidate(directory):
    directory=Path(directory)
    meta=json.loads((directory/"alphabet.json").read_text(encoding="utf-8"))
    training=json.loads((directory/"training.json").read_text(encoding="utf-8"))
    if sha256(directory/"alphabet.json")!=training["alphabet_sha256"]:
        raise ValueError("Candidate alphabet changed")
    model=directory/"kaomoji.onnx"
    if sha256(model)!=training["model_sha256"]:
        raise ValueError("Candidate differs from frozen training report")
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    process=psutil.Process();before=process.memory_info().rss
    start=time.perf_counter()
    session=ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])
    cold=time.perf_counter()-start
    return session,meta["alphabet"],dict(model_bytes=model.stat().st_size,
        model_sha256=sha256(model),providers=session.get_providers(),threads=2,
        session_load_seconds=cold,session_rss_delta_bytes=max(0,process.memory_info().rss-before))

def read_private(path):
    manifest=json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("usage")!="independent-evaluation-only":
        raise ValueError("Private manifest lacks evaluation-only declaration")
    result=[]
    for record in manifest["records"]:
        if record["kind"] not in ("face","plain","blank","icon"):
            raise ValueError("Unknown private sample kind")
        with Image.open(record["source"]) as source:
            source.load()
            rect=record.get("rect",[0,0,source.width,source.height])
            x,y,w,h=rect
            if min(x,y)<0 or min(w,h)<=0 or x+w>source.width or y+h>source.height:
                raise ValueError("Private crop outside source")
            image=source.crop((x,y,x+w,y+h)).convert("RGB")
        if record["kind"] in ("face","plain") and not record.get("ambiguous",False) and not record.get("label"):
            raise ValueError("Reliable text samples require nonempty label")
        if record["kind"] in ("blank","icon") and record.get("label","")!="":
            raise ValueError("Negative samples require blank label")
        result.append(dict(record,image=image,source_sha256=sha256(record["source"]),rect=rect))
    return result

def compare(images,rows,session,alphabet,baseline):
    from rapidocr.ch_ppocr_rec import TextRecInput
    recognizer=baseline._load_rec_model()
    output=[];latencies=[]
    for start in range(0,len(images),16):
        batch=images[start:start+16]
        array=[np.asarray(im)[:,:,::-1].copy() for im in batch]
        recognized=recognizer(TextRecInput(img=array,return_word_box=False))
        if len(recognized.txts)!=len(batch) or len(recognized.scores)!=len(batch) or not all(math.isfinite(float(v)) for v in recognized.scores):
            raise ValueError("Baseline output count mismatch")
        for index,(image,text,score) in enumerate(zip(batch,recognized.txts,recognized.scores),start):
            ts=time.perf_counter()
            pred,confidence=decode(session.run(None,{"image":prepare(image)[None]})[0][0],alphabet)
            latencies.append(time.perf_counter()-ts)
            output.append(dict(rows[index],baseline=str(text),baseline_score=float(score),
                          candidate=pred,candidate_score=confidence))
    if len(output)!=len(rows) or len(rows)!=len(images):
        raise ValueError("Incomplete comparison")
    return output,latencies

def summarize(rows):
    exact=[r for r in rows if not r.get("ambiguous",False)]
    report={"ambiguous_count":len(rows)-len(exact)}
    for kind in ("face","plain"):
        part=[r for r in exact if r["kind"]==kind]
        report[kind]={field:metrics(part,field) for field in ("baseline","candidate")}
        report[kind+"_regressions"]=sum(r["baseline"]==r["label"] and r["candidate"]!=r["label"] for r in part)
        report[kind+"_erroneous_changes"]=sum(r["candidate"]!=r["baseline"] and r["candidate"]!=r["label"] for r in part)
    negative=[r for r in exact if r["kind"] in ("blank","icon")]
    report.update(negative_count=len(negative),negative_false_outputs=sum(bool(r["candidate"]) for r in negative))
    # Glyph features are separate from exact codepoint metrics.
    feature_rows=[r for r in rows if r.get("visible_features")]
    report["visible_glyph_features"]={field:dict(
        matched=sum(all(c in r[field] for c in r["visible_features"]) for r in feature_rows),
        count=len(feature_rows)) for field in ("baseline","candidate")}
    return report

def main(args):
    out=local_output(args.output)
    if out.exists():
        raise ValueError("Evaluation output exists")
    root=Path(args.dataset).resolve()
    training=json.loads((Path(args.candidate)/"training.json").read_text(encoding="utf-8"))
    if dataset_fingerprint(root)!={k:training[k] for k in ("dataset_sha256","split_sha256")}:
        raise ValueError("Dataset changed since training")
    rows=json.loads((root/"test.json").read_text(encoding="utf-8"))
    train_rows=json.loads((root/"train.json").read_text(encoding="utf-8"))
    dev_rows=json.loads((root/"dev.json").read_text(encoding="utf-8"))
    labels=lambda seq:{r["label"] for r in seq if r["label"]}
    if labels(rows)&(labels(train_rows)|labels(dev_rows)):
        raise ValueError("Independent test text leakage")
    images=[]
    for r in rows:
        target=(root/r["path"]).resolve()
        if root not in target.parents:
            raise ValueError("Synthetic path escape")
        with Image.open(target) as image:
            images.append(image.convert("RGB"))
    session,alphabet,resources=load_candidate(args.candidate)
    # Separate process peak includes cold model load, preparation and inference.
    def area(row):
        with Image.open(root/row["path"]) as image:
            return image.width*image.height
    probe_file=root/max(rows,key=area)["path"]
    measured=[]
    for _ in range(3):
        result=subprocess.run([sys.executable,__file__,"--resource-probe",str(args.candidate),str(probe_file)],
                              capture_output=True,text=True,check=True,timeout=30)
        measured.append(json.loads(result.stdout))
    resources.update(forty_eight_crops_seconds=max(r["seconds"] for r in measured),
        peak_extra_bytes=max(r["peak_extra_bytes"] for r in measured),
        peak_method=measured[0]["peak_method"],cold_session_plus_48_probe=True,
        resource_probes=measured,groups=48,
        probe_total_pixels=min(r["probe_total_pixels"] for r in measured))
    worker_path=Path(__file__).resolve().parents[2]/"scripts/local-ai-ocr-worker.py"
    spec=importlib.util.spec_from_file_location("baseline_worker",worker_path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    baseline=worker.create_engine(Path(args.baseline_models))
    synthetic,s_times=compare(images,rows,session,alphabet,baseline)
    private_records=read_private(args.private_manifest)
    private_images=[r.pop("image") for r in private_records]
    private,p_times=compare(private_images,private_records,session,alphabet,baseline)
    report=dict(schema=1,candidate_sha256=resources["model_sha256"],
        dataset_sha256=sha256(root/"dataset.json"),test_split_sha256=sha256(root/"test.json"),
        private_manifest_sha256=sha256(args.private_manifest),
        synthetic=summarize(synthetic),private=summarize(private),resources=resources,
        median_crop_seconds=statistics.median(s_times+p_times),
        compare_scope="same manual crop; raw recognition; no production correction",
        font_split_scope="held-out primary fonts; shared glyph fallback fonts",
        private_excluded_from_training_and_selection=True,
        production_correction_verified=False,
        actual_production_added_corrections=0)
    report["decision"]=decide(report["synthetic"],report["private"],resources)
    write_json(out/"private-predictions.json",private)
    write_json(out/"synthetic-predictions.json",synthetic)
    write_json(out/"evaluation.json",report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)

def resource_probe(candidate,image_path):
    process=psutil.Process()
    before=process.memory_info().rss
    peak_before=getattr(process.memory_info(),"peak_wset",before)
    method="windows_process_peak_wset" if hasattr(process.memory_info(),"peak_wset") else "unsupported_peak_method"
    started=time.perf_counter()
    session,alphabet,_=load_candidate(candidate)
    with Image.open(image_path) as im:
        image=im.convert("RGB")
    # Exercise almost the full existing 1M-pixel budget across 48 crops.
    scale=math.sqrt((1000000/48)/(image.width*image.height))
    width=max(1,int(image.width*scale));height=max(1,int(image.height*scale))
    while width*height>1000000//48:
        width-=1
    image=image.resize((width,height),Image.Resampling.BILINEAR)
    for _ in range(48):
        session.run(None,{"image":prepare(image)[None]})
    peak=getattr(process.memory_info(),"peak_wset",process.memory_info().rss)
    print(json.dumps(dict(seconds=time.perf_counter()-started,peak_extra_bytes=max(0,peak-before),peak_method=method,
        baseline_rss=before,peak_before=peak_before,peak_after=peak,
        groups=48,probe_total_pixels=48*image.width*image.height)))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--resource-probe":
        resource_probe(sys.argv[2],sys.argv[3])
        sys.exit(0)
    p=argparse.ArgumentParser()
    for name in ("dataset","candidate","baseline-models","private-manifest","output"):
        p.add_argument("--"+name,required=True)
    main(p.parse_args())
