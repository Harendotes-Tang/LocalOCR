"""Extract text only from locally saved LDDGO symbol-list entries; no networking."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import re
import unicodedata
from common import local_output,sha256,write_json

SOURCE="https://www.lddgo.net/common/emoticons"

def extract(page):
    entries=[];rejected=0
    groups=re.findall(r'<ul\s+class="symbol_list"\s*>(.*?)</ul>',page,re.S)
    if not groups:
        raise ValueError("No declared symbol lists")
    for group in groups:
        for raw in re.findall(r'<span\s*>(.*?)<br\s*/?>',group,re.S):
            # Remove HTML layout indentation only; preserve internal Unicode spaces.
            value=html.unescape(raw.strip())
            if (not value or len(value)>60 or re.search(r'<[A-Za-z/!][^>]*>',value)
                or any(unicodedata.category(c)[0]=="C" for c in value)):
                rejected+=1;continue
            entries.append(value)
    return sorted(set(entries)),dict(raw_entries=len(entries),unique_entries=len(set(entries)),rejected=rejected)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--html",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();target=local_output(args.output)
    if target.exists():
        raise ValueError("Corpus exists")
    texts,audit=extract(Path(args.html).read_text(encoding="utf-8"))
    write_json(target,dict(schema=1,source=SOURCE,html_sha256=sha256(args.html),
        extraction="symbol_list span before br; HTML entity decode; no Unicode normalization",
        distribution="local training input only; no claim of open dataset licence",texts=texts,audit=audit))
    print(audit)