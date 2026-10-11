"""Check exact codepoint representability; this is not pixel OCR accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import onnxruntime as ort
from common import local_output,sha256,write_json

EMOJI=["😀","😂","👍","❤️","🇨🇳","👨‍👩‍👧‍👦"]

def audit_texts(texts,alphabet):
    return [dict(index=i,codepoints=[f"U+{ord(c):04X}" for c in text],
        representable=all(c in alphabet for c in text),
        missing=[f"U+{ord(c):04X}" for c in text if c not in alphabet]) for i,text in enumerate(texts)]

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--baseline-model",required=True)
    p.add_argument("--reserved-texts",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();target=local_output(args.output)
    if target.exists():
        raise ValueError("Compatibility report exists")
    candidate=json.loads((Path(args.dataset)/"dataset.json").read_text(encoding="utf-8"))["alphabet"]
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(args.baseline_model,options,providers=["CPUExecutionProvider"])
    baseline=session.get_modelmeta().custom_metadata_map["character"].splitlines()
    private=json.loads(Path(args.reserved_texts).read_text(encoding="utf-8"))["texts"]
    report=dict(schema=1,scope="dictionary representability only; no pixel or production correctness claim",
        baseline_sha256=sha256(args.baseline_model),dataset_sha256=sha256(Path(args.dataset)/"dataset.json"),
        private_manifest_sha256=sha256(args.reserved_texts),
        reserved={"baseline":audit_texts(private,baseline),"candidate":audit_texts(private,candidate)},
        emoji={"baseline":audit_texts(EMOJI,baseline),"candidate":audit_texts(EMOJI,candidate)})
    write_json(target,report)
    print(json.dumps({group:{name:dict(count=len(rows),representable=sum(r['representable'] for r in rows)) for name,rows in report[group].items()} for group in ('reserved','emoji')}))