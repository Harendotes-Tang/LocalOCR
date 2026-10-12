"""Offline generator: python -m tools.OcrDatasets.generate --help."""
import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from . import __version__
from .corpus import originals
from .render import Renderer, Unsupported, file_hash

SPLITS = ('train', 'dev', 'test', 'regression', 'acceptance')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def allocate(groups):
    by_kind = {}
    for group in groups:
        by_kind.setdefault(group['kind'], []).append(group)
    result = {name: [] for name in SPLITS}
    cycle = ('train', 'dev', 'test', 'regression', 'acceptance', 'train', 'train', 'test', 'regression', 'acceptance')
    for values in by_kind.values():
        for index, value in enumerate(values):
            result[cycle[index % len(cycle)]].append(value)
    return result


def generate(output, fonts, exclusions, seed=731942, groups=500):
    if any(str(value).startswith(('\\\\', '//')) for value in (output, fonts, exclusions)):
        raise ValueError('Network paths are forbidden')
    root = Path(output).resolve()
    artifacts = Path(__file__).resolve().parents[2] / 'artifacts'
    if not root.is_relative_to(artifacts.resolve()) or root.exists():
        raise ValueError('Output must be a fresh directory under artifacts')
    font_path = Path(fonts).resolve()
    if not font_path.is_relative_to(artifacts.resolve()):
        raise ValueError('Font profiles must remain local under artifacts')
    exclusion_path = Path(exclusions).resolve()
    if not exclusion_path.is_relative_to(artifacts.resolve()):
        raise ValueError('Exclusion inventory must remain local under artifacts')
    excluded = json.loads(exclusion_path.read_text(encoding='utf-8-sig'))
    if excluded.get('schema_version') != 'localocr.exclusions.v1' or not isinstance(excluded.get('label_sha256'), list):
        raise ValueError('Unknown exclusion inventory')
    hashes = excluded['label_sha256']
    if any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in hashes):
        raise ValueError('Invalid excluded exact label digest')
    renderer = Renderer(fonts)
    original = originals(seed, groups, hashes)
    split_groups = allocate(original)
    root.mkdir(parents=True)
    (root / 'images').mkdir()
    rng = random.Random(seed)
    coverage = {'schema_version': 'localocr.coverage.v1', 'requested_groups': groups, 'excluded_prior_inventory_count': len(hashes), 'render_exclusions': [], 'counts': {}, 'unicode_codepoints': sorted({ord(c) for item in original for c in item['label']})}
    splits = {}
    image_hash_split = {}
    for split in SPLITS:
        records = []
        for group in split_groups[split]:
            for variant in range(3 if split == 'train' else 2):
                transform = {'size': (24, 32, 20)[variant], 'dark': bool(variant % 2), 'scale': 1.0 if variant == 0 else 0.85, 'blur': 0.0 if variant == 0 else 0.3, 'jpeg': None if variant == 0 else 86}
                try:
                    image, bounds, used, evidence = renderer.render(group['label'], emoji=group['kind'] == 'emoji', **transform)
                except Unsupported as exc:
                    coverage['render_exclusions'].append({'group': group['group'], 'kind': group['kind'], 'variant': variant, 'reason': str(exc)})
                    continue
                identifier = f'{split}-{group["label_sha256"][:16]}-{variant}'
                path = root / 'images' / (identifier + '.png')
                image.save(path, compress_level=9)
                image_sha = file_hash(path)
                if image_sha in image_hash_split and image_hash_split[image_sha] != split:
                    path.unlink()
                    coverage['render_exclusions'].append({'group': group['group'], 'kind': group['kind'], 'variant': variant, 'reason': 'cross-split-identical-pixels'})
                    continue
                image_hash_split[image_sha] = split
                records.append(dict(group, id=identifier, path=path.relative_to(root).as_posix(), sha256=image_sha, width=image.width, height=image.height, region=bounds, font=used, transform=transform, provenance={'source': 'original', 'license': 'CC0-1.0'}, usage=split, rendering=evidence))
        for index in range(12):
            kind = 'blank' if index % 2 == 0 else 'graphic'
            recipe = f'{seed}-{split}-{index}'
            width, height = 75 + SPLITS.index(split) * 17 + index, 35 + index * 3
            background = tuple(rng.randrange(15, 240) for _ in range(3))
            image = Image.new('RGB', (width, height), background)
            shapes = []
            if kind == 'graphic':
                draw = ImageDraw.Draw(image)
                color = tuple(255 - v for v in background)
                box = [7, 7, width - 9, height - 9]
                draw.ellipse(box, outline=color, width=3)
                draw.line([8, height - 8, width - 8, 8], fill=color, width=2)
                shapes = [{'ellipse': box, 'line': [8, height - 8, width - 8, 8], 'color': color}]
            identifier = f'{split}-negative-{index}'
            path = root / 'images' / (identifier + '.png')
            image.save(path, compress_level=9)
            records.append({'id': identifier, 'group': 'negative:' + recipe, 'recipe_id': recipe, 'kind': kind, 'label': '', 'label_sha256': hashlib.sha256(b'').hexdigest(), 'path': path.relative_to(root).as_posix(), 'sha256': file_hash(path), 'width': width, 'height': height, 'region': [0, 0, width, height], 'font': [], 'transform': {'background': background, 'shapes': shapes}, 'provenance': {'source': 'original', 'license': 'CC0-1.0'}, 'usage': split, 'rendering': {'text_present': False}})
        write_json(root / (split + '.json'), records)
        splits[split] = {'file': split + '.json', 'sha256': file_hash(root / (split + '.json')), 'count': len(records)}
        coverage['counts'][split] = dict(Counter(record['kind'] for record in records))
    coverage['exclusion_reasons'] = dict(Counter(item['reason'] for item in coverage['render_exclusions']))
    write_json(root / 'coverage.json', coverage)
    public_fonts = [{key: value for key, value in font.items() if key not in ('file', 'license_file')} for font in renderer.fonts]
    for font in public_fonts:
        font.setdefault('version', font['source_commit'])
    source_digest = hashlib.sha256(b''.join((Path(__file__).parent / name).read_bytes() for name in ('__init__.py', 'corpus.py', 'render.py', 'generate.py'))).hexdigest()
    manifest = {'schema_version': 'localocr.dataset.v1', 'seed': seed, 'generator': {'version': __version__, 'source_sha256': source_digest, 'groups': groups}, 'fonts': public_fonts, 'renderer': dict(renderer.versions, python=platform.python_version()), 'splits': splits, 'exclusions': {'count': len(hashes), 'inventory_path': str(exclusion_path), 'inventory_sha256': file_hash(exclusion_path)}, 'license': 'CC0-1.0', 'synthetic_only': True}
    write_json(root / 'dataset.json', manifest)
    return manifest, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fonts', required=True, help='Local verified OFL fonts.json')
    parser.add_argument('--exclusions', required=True)
    parser.add_argument('--seed', type=int, default=731942)
    parser.add_argument('--groups', type=int, default=500)
    args = parser.parse_args()
    manifest, coverage = generate(args.output, args.fonts, args.exclusions, args.seed, args.groups)
    print(json.dumps({'splits': manifest['splits'], 'excluded': coverage['exclusion_reasons']}, indent=2))


if __name__ == '__main__':
    main()
