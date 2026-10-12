"""Aggregate exact Unicode/font coverage from a verified frozen dataset."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .validate import validate, read, local_output, write

def report(root, output):
    root=Path(root).resolve();meta,splits,binding=validate(root)
    coverage=read(root/'coverage.json')
    rendered={ord(c) for rows in splits.values() for row in rows for c in row['label']}
    requested=set(coverage['unicode_codepoints'])
    fonts={name:dict(Counter(font for row in rows for font in row['font'])) for name,rows in splits.items()}
    counts={name:dict(Counter(row['kind'] for row in rows)) for name,rows in splits.items()}
    result={'schema_version':'localocr.coverage-audit.v1','dataset_binding':binding,'categories':counts,
            'requested_codepoints':['U+%04X'%cp for cp in sorted(requested)],
            'rendered_codepoints':['U+%04X'%cp for cp in sorted(rendered)],
            'unrendered_codepoints':['U+%04X'%cp for cp in sorted(requested-rendered)],
            'font_record_counts':fonts,'render_exclusion_reasons':coverage['exclusion_reasons'],
            'prior_label_digest_count':coverage['excluded_prior_inventory_count'],
            'cross_split_text_leakage':0,'cross_split_source_group_leakage':0,'cross_split_decoded_pixel_duplicates':0,
            'normalization':'none; raw Unicode labels, spaces and combining sequences retained'}
    output=local_output(output)
    if output.exists():raise ValueError('Coverage report output must be fresh')
    write(output,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(report(a.dataset,a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
