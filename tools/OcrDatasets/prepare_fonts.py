"""Explicit development-only official-font acquisition; generation stays offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def acquire(output: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    artifacts = (repo / 'artifacts').resolve()
    output = output.resolve()
    if artifacts.parent != repo or output == artifacts or artifacts not in output.parents:
        raise ValueError('Font output must be under repository artifacts')
    profile = json.loads(Path(__file__).with_name('font-sources.json').read_text(encoding='utf-8-sig'))
    output.mkdir(parents=True, exist_ok=True)
    for row in profile['fonts']:
        for filename, url, expected in ((row['file'], row['source_url'], row['sha256']),
                                       (row['license_file'], row['license_url'], row['license_sha256'])):
            if Path(filename).name != filename or not url.startswith('https://raw.githubusercontent.com/') or row['source_commit'] not in url:
                raise ValueError('Unpinned font source')
            target = output / filename
            if target.is_symlink() or target.resolve().parent != output:
                raise ValueError('Font target escapes local output')
            data = target.read_bytes() if target.exists() else urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Official font or license hash mismatch: '+filename)
            if filename == row['license_file'] and b'SIL OPEN FONT LICENSE' not in data:
                raise ValueError('OFL license absent')
            if not target.exists():
                target.write_bytes(data)
    (output/'fonts.json').write_text(json.dumps(profile, indent=2)+'\n', encoding='utf-8')
    return dict(fonts=len(profile['fonts']), schema_version=profile['schema_version'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(acquire(args.output)))

if __name__ == '__main__':
    main()
