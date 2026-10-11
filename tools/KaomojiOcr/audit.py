"""Audit the pinned baseline dictionary without downloading model assets."""
import argparse
from pathlib import Path
import onnxruntime as ort
from common import local_output,sha256,write_json

TARGET="Ддωづつ∀°´｀￣̅̄•●Σ∑"

def audit(model,output):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])
    metadata=session.get_modelmeta().custom_metadata_map
    if "character" not in metadata:
        raise ValueError("Baseline has no embedded dictionary")
    chars=metadata["character"].splitlines()
    report=dict(schema=1,model_sha256=sha256(model),model_bytes=Path(model).stat().st_size,
        dictionary_entries=len(chars),coverage={c:c in chars for c in TARGET},
        missing=[c for c in TARGET if c not in chars],providers=session.get_providers())
    write_json(local_output(output),report)
    print(report["dictionary_entries"],[f"U+{ord(c):04X}" for c in report["missing"]])

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();audit(a.model,a.output)
