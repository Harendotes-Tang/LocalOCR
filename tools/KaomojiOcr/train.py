"""GPU/MPS/CPU CTC trainer; consumes declared synthetic train/dev only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from PIL import Image
from common import HEIGHT,WIDTH,decode,local_output,prepare,sha256,write_json,dataset_fingerprint

class Recognizer(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2),
             nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2),
             nn.Conv2d(64,96,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1)),
             nn.Conv2d(96,96,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1)))
        self.rnn=nn.GRU(96,64,num_layers=2,batch_first=True,bidirectional=True)
        self.head=nn.Linear(128,classes)
    def forward(self,x):
        x=self.cnn(x).mean(dim=2).transpose(1,2)
        x,_=self.rnn(x)
        return self.head(x)

def load_split(root,name,alphabet):
    rows=json.loads((root/f"{name}.json").read_text(encoding="utf-8"))
    mapping={c:i+1 for i,c in enumerate(alphabet)}
    images,targets,labels=[],[],[]
    for r in rows:
        path=(root/r["path"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Dataset path escapes root")
        with Image.open(path) as image:
            images.append(prepare(image))
        targets.append(torch.tensor([mapping[c] for c in r["label"]],dtype=torch.long))
        labels.append(r["label"])
    return torch.from_numpy(np.stack(images)),targets,labels

def main(args):
    out=local_output(args.output)
    if out.exists():
        raise ValueError("Training output exists")
    root=Path(args.dataset).resolve()
    meta=json.loads((root/"dataset.json").read_text(encoding="utf-8"))
    if not meta.get("synthetic_only") or meta.get("private_inputs_consumed") is not False:
        raise ValueError("Training requires synthetic declaration")
    frozen=dataset_fingerprint(root)
    out.mkdir(parents=True);alphabet=meta["alphabet"]
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    device=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if args.device=="auto" else args.device
    model=Recognizer(len(alphabet)+1).to(device)
    inputs,targets,labels=load_split(root,"train",alphabet)
    dev_inputs,_,dev_labels=load_split(root,"dev",alphabet)
    if (set(labels)-{""}) & (set(dev_labels)-{""}):
        raise ValueError("Train/development text leakage")
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    loss_fn=nn.CTCLoss(blank=0,zero_infinity=False)
    history=[];best=-1;started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();losses=[]
        for indices in torch.randperm(len(inputs)).split(args.batch):
            t=[targets[i] for i in indices.tolist()]
            pred=model(inputs[indices].to(device))
            loss_device="cpu" if device=="mps" else device
            loss=loss_fn(pred.log_softmax(-1).transpose(0,1).to(loss_device),torch.cat(t).to(loss_device),
                torch.full((len(t),),pred.shape[1],dtype=torch.long),
                torch.tensor([len(x) for x in t],dtype=torch.long))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(loss.item())
        model.eval();predictions=[]
        with torch.no_grad():
            for batch in dev_inputs.split(args.batch):
                predictions.extend(decode(v,alphabet)[0] for v in model(batch.to(device)).cpu().numpy())
        exact=sum(a==b for a,b in zip(predictions,dev_labels))
        row=dict(epoch=epoch+1,loss=sum(losses)/len(losses),dev_exact=exact,dev_count=len(dev_labels),seconds=time.perf_counter()-started)
        history.append(row);print(json.dumps(row),flush=True)
        if exact>best:
            best=exact;torch.save(dict(state=model.cpu().state_dict(),alphabet=alphabet),out/"best.pt");model.to(device)
    checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    model.cpu().load_state_dict(checkpoint["state"]);model.eval()
    torch.onnx.export(model,dev_inputs[:1],str(out/"kaomoji.onnx"),opset_version=17,
         input_names=["image"],output_names=["logits"],dynamo=False,external_data=False)
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(out/"kaomoji.onnx"),options,providers=["CPUExecutionProvider"])
    differences=[]
    with torch.no_grad():
        for sample in dev_inputs[:12]:
            a=model(sample[None]).numpy();b=session.run(None,{"image":sample[None].numpy()})[0]
            differences.append(float(np.max(np.abs(a-b))))
            if decode(a[0],alphabet)[0]!=decode(b[0],alphabet)[0]:
                raise ValueError("ONNX decoding differs")
    write_json(out/"alphabet.json",dict(schema=1,alphabet=alphabet,height=HEIGHT,width=WIDTH))
    if dataset_fingerprint(root)!=frozen:
        raise ValueError("Dataset changed during training")
    write_json(out/"training.json",dict(schema=1,seed=args.seed,epochs=args.epochs,batch=args.batch,lr=args.lr,
        device=device,torch_version=torch.__version__,parameters=sum(p.numel() for p in model.parameters()),
        gpu_name=torch.cuda.get_device_name(0) if device=="cuda" else None,
        cuda_version=torch.version.cuda if device=="cuda" else None,
        gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else 0,
        training_seconds=time.perf_counter()-started,history=history,selected_by="development exact string only",
        dataset_sha256=frozen["dataset_sha256"],split_sha256=frozen["split_sha256"],
        alphabet_sha256=sha256(out/"alphabet.json"),
        model_sha256=sha256(out/"kaomoji.onnx"),model_bytes=(out/"kaomoji.onnx").stat().st_size,
        onnx_max_logit_difference=max(differences),onnx_decode_equal=True,private_inputs_consumed=False,baseline_changed=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["auto","cuda","mps","cpu"],default="auto")
    p.add_argument("--epochs",type=int,default=24);p.add_argument("--batch",type=int,default=96)
    p.add_argument("--lr",type=float,default=.001);p.add_argument("--seed",type=int,default=20261011)
    main(p.parse_args())
