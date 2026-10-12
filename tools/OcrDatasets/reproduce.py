"""Offline replay of exported original regression pixels without private inventories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from .render import Renderer, file_hash
from .validate import local_output, read, label_digest, write, confined

def reproduce(package, fonts, output):
    package = Path(package).resolve()
    meta = read(package/'dataset.json')
    row_path = package/'regression.json'
    if meta.get('license') != 'CC0-1.0' or file_hash(row_path) != meta['splits']['regression']['sha256']:
        raise ValueError('Public regression inventory changed')
    rows = read(row_path)
    if len(rows) != meta['splits']['regression']['count']:
        raise ValueError('Public regression count changed')
    for row in rows:
        if file_hash(confined(package,row['path'])) != row['sha256']:
            raise ValueError('Public regression image changed')
    renderer = Renderer(fonts)
    if any(meta['renderer'].get(k) != v for k,v in renderer.versions.items()):
        raise ValueError('Renderer versions differ from frozen package')
    if {r['id']:r['sha256'] for r in renderer.fonts} != {r['id']:r['sha256'] for r in meta['fonts']}:
        raise ValueError('Font bytes differ from frozen package')
    output = local_output(output)
    if output.exists():
        raise ValueError('Replay output must be fresh')
    output.mkdir(parents=True)
    for row in rows:
        confined(package,row['path'])
        if label_digest(row['label']) != row['label_sha256']:
            raise ValueError('Exact public label changed')
        transform=row['transform']
        if row['kind'] in ('blank','graphic'):
            if row['label']:
                raise ValueError('Negative image has a text label')
            image=Image.new('RGB',(row['width'],row['height']),tuple(transform['background']))
            draw=ImageDraw.Draw(image)
            for shape in transform['shapes']:
                draw.ellipse(shape['ellipse'],outline=tuple(shape['color']),width=3)
                draw.line(shape['line'],fill=tuple(shape['color']),width=2)
        else:
            image,bounds,used,evidence=renderer.render(row['label'],emoji=row['kind']=='emoji',**transform)
            if bounds != row['region'] or used != row['font'] or evidence != row['rendering']:
                raise ValueError('Glyph or pixel bounds differ from frozen record')
        target=output/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        image.save(target,compress_level=9)
        if file_hash(target) != row['sha256']:
            raise ValueError('Replayed pixels differ: '+row['id'])
    report={'schema_version':'localocr.reproduction.v1','images':len(rows),'identical_image_hashes':True,
            'private_inventory_consumed':False,'regression_sha256':file_hash(row_path)}
    write(output/'reproduction.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--fonts',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(reproduce(a.package,a.fonts,a.output),indent=2))
if __name__=='__main__':main()
