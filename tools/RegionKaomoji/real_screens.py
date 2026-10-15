"""Known-input real screenshot evaluation: labels are the typed strings, never read from pixels.

The manifest lists, per screenshot, the exact lines the user typed. Faces are
scored by exact region match and by containment in the reading-order text;
body lines that the production first pass already reads must survive fusion.
Results stay in the ignored round root; the public summary holds counts only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from tools.PretrainedOcr import common, round_profile as rp, quality_evaluate as qe
from tools.RegionKaomoji import evaluate as e, candidates as c, classifier as cls, fusion as f

MANIFEST_SCHEMA = 'localocr.known-input-screens.v1'
SCHEMA = 'localocr.known-input-evaluation.v1'
FIRST_USE_SCHEMA = 'localocr.known-input-first-use.v1'
PROBE_SIDECAR = 'known-input-development-use.json'
PROBE_KIND = 'known-input-screens'
KINDS = ('face', 'body')


def probe_registration(manifest_sha256):
    """The registered round whose known-input sidecar lists this manifest as a development probe, or None.

    A probed set was inspected per face; it may still be run for counts and per-face outcomes, but
    its report is never independent and it never claims a first use.
    """
    for profile in rp.ROUNDS:
        sidecar = Path(profile.root) / PROBE_SIDECAR
        if not sidecar.exists():
            continue
        value = common.read(sidecar)
        if value.get('schema') != 'localocr.development-demotion.v1' or value.get('kind') != PROBE_KIND or value.get('original_rewritten') is not False:
            continue
        for record in value.get('old_datasets', []):
            if record.get('sha256') == manifest_sha256:
                return {'round': profile.name, 'sidecar': str(sidecar), 'sidecar_sha256': common.digest(sidecar)}
    return None


def first_use_path(round_root, manifest_sha256):
    # Lives in the round's evaluation namespace like the heldout first-use receipts.
    return Path(round_root) / ('evaluation-known-input-first-use-' + manifest_sha256 + '.json')


def existing_first_use(manifest_sha256):
    for profile in rp.ROUNDS:
        path = first_use_path(profile.root, manifest_sha256)
        if path.exists():
            return path
    return None


def claim_first_use(manifest, output, config, config_file=None, dual=()):
    """Write the independent first-use receipt for ``manifest`` before any screenshot is read.

    The receipt binds the manifest bytes, the pipeline configuration, the research source closure and
    any declared dual-freeze files. A second independent run on the same manifest is refused in every
    registered round; failure after the claim leaves the manifest consumed.
    """
    profile = rp.profile_for(output)
    if profile is None:
        raise ValueError('Known-input evaluation output must belong to a registered research round')
    sha = manifest['sha256']
    if probe_registration(sha) is not None:
        raise ValueError('This known-input manifest is registered as a development probe; run it with --probe')
    prior = existing_first_use(sha)
    if prior is not None:
        raise ValueError('Known-input manifest already consumed by an independent first use: ' + str(prior))
    inventory = qe.source_inventory()
    receipt = {'schema_version': FIRST_USE_SCHEMA, 'claimed_utc': common.utc(), 'round': profile.name,
               'manifest': {'file': manifest['file'], 'sha256': sha, 'screens': len(manifest['value']['screens'])},
               'output': str(e.output_path(output)),
               'config': {'file': str(e.safe_path(config_file)) if config_file else None,
                          'sha256': common.digest(config_file) if config_file else None, 'object_sha256': e.hash_object(config)},
               'source_inventory': {'files': len(inventory), 'sha256': e.hash_object(inventory)},
               'dual_freezes': [{'file': str(e.safe_path(path)), 'sha256': common.digest(path)} for path in dual],
               'rule': 'independent evaluation only; a failure after this claim leaves the manifest consumed'}
    path = first_use_path(profile.root, sha)
    e.write_new(path, receipt)
    return {'file': str(path), 'sha256': common.digest(path)}


def load_manifest(path):
    path = e.safe_path(path)
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8'))
    if value.get('schema') != MANIFEST_SCHEMA or value.get('usage') != 'independent-evaluation-only':
        raise ValueError('Known-input manifest must declare independent evaluation usage')
    screens = value.get('screens')
    if not isinstance(screens, list) or not screens or len(screens) > 500:
        raise ValueError('Manifest needs 1..500 screenshots')
    seen = set()
    for screen in screens:
        if not isinstance(screen, dict) or not isinstance(screen.get('file'), str) or not isinstance(screen.get('lines'), list) or not screen['lines']:
            raise ValueError('Each screenshot needs a file and typed lines')
        target = e.confined(path.parent, screen['file'])
        if not target.is_file() or target.suffix.lower() != '.png' or target in seen:
            raise ValueError('Screenshot must be a unique local PNG inside the manifest directory')
        seen.add(target)
        for line in screen['lines']:
            if (not isinstance(line, dict) or line.get('kind') not in KINDS or not isinstance(line.get('text'), str)
                    or not line['text'].strip() or type(line.get('ambiguous', False)) is not bool
                    or any(ch in line['text'] for ch in '\r\n\t')):
                raise ValueError('Each line needs a kind, exact nonempty text and an optional boolean ambiguous flag')
            faces = line.get('faces', [])
            if not isinstance(faces, list) or any(not isinstance(f, str) or f not in line['text'] for f in faces):
                raise ValueError('Embedded faces must be substrings of their body line')
            if line['kind'] == 'face' and c.is_face_text(line['text']) is False:
                line['grammar_accepted'] = False
            elif line['kind'] == 'face':
                line['grammar_accepted'] = True
    return {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'value': value, 'root': path.parent}


def score_screen(screen, prediction, model_ids):
    """Per-line outcomes for one screenshot; texts stay private, only flags are aggregated."""
    baseline_regions = [r['text'] for r in prediction['baseline']]
    baseline_joined = prediction['baseline_text']
    results = []
    for line in screen['lines']:
        text = line['text']
        record = {'kind': line['kind'], 'ambiguous': bool(line.get('ambiguous', False)),
                  'grammar_accepted': line.get('grammar_accepted'), 'models': {}}
        targets = [text] if line['kind'] == 'face' else list(line.get('faces', []))
        record['baseline'] = {'exact_region': text in baseline_regions, 'contained': text in baseline_joined,
                              'faces_contained': [f in baseline_joined for f in targets]}
        projected = f.width_key(text)
        for identity in model_ids:
            model = prediction['models'][identity]
            regions = [r['text'] for r in model['regions']]
            joined = model['text']
            record['models'][identity] = {'exact_region': text in regions, 'contained': text in joined,
                                          'faces_contained': [fc in joined for fc in targets],
                                          'body_lost': line['kind'] == 'body' and text in baseline_joined and text not in joined}
            # Diagnostic only: a region equal to the typed face except for its outermost bracket width.
            record['models'][identity]['exact_except_outer_bracket_width'] = bool(
                line['kind'] == 'face' and text not in regions and any(f.width_key(r) == projected for r in regions))
        results.append(record)
    return results


def diagnostics(scored, predictions, model_ids):
    """Counts-only shape-detection and width diagnostics; never part of the scored metrics."""
    totals = {'shape_base_pairs': 0, 'shape_core_pairs': 0, 'merged_bracket_candidates': 0, 'shape_threshold': None}
    for prediction in predictions:
        shape = prediction['routing'].get('shape_detection') or {}
        totals['shape_base_pairs'] += int(shape.get('base_pairs', 0))
        totals['shape_core_pairs'] += int(shape.get('core_pairs', 0))
        totals['merged_bracket_candidates'] += int(shape.get('merged_bracket_candidates', 0))
        totals['shape_threshold'] = shape.get('threshold', totals['shape_threshold'])
    faces = [r for s in scored for r in s if r['kind'] == 'face' and not r['ambiguous']]
    totals['models'] = {identity: {'face_exact_except_outer_bracket_width': sum(r['models'][identity].get('exact_except_outer_bracket_width', False) for r in faces),
                                   'bracket_width_sources': dict(_width_sources(predictions, identity))}
                        for identity in model_ids}
    return totals


def _width_sources(predictions, identity):
    from collections import Counter
    counts = Counter()
    for prediction in predictions:
        for decision in prediction['models'][identity].get('decisions', []):
            if decision.get('accepted'):
                counts[str(decision.get('bracket_width_source'))] += 1
    return counts


def aggregate(scored, model_ids):
    faces = [r for s in scored for r in s if r['kind'] == 'face' and not r['ambiguous']]
    bodies = [r for s in scored for r in s if r['kind'] == 'body' and not r['ambiguous']]
    embedded = [(r, i) for s in scored for r in s if r['kind'] == 'body' and not r['ambiguous'] for i in range(len(r['baseline']['faces_contained']))]
    summary = {'faces': len(faces), 'faces_ambiguous_excluded': sum(r['ambiguous'] for s in scored for r in s if r['kind'] == 'face'),
               'faces_grammar_accepted': sum(bool(r['grammar_accepted']) for r in faces), 'body_lines': len(bodies),
               'embedded_faces': len(embedded),
               'baseline': {'face_exact_region': sum(r['baseline']['exact_region'] for r in faces),
                            'face_contained': sum(r['baseline']['contained'] for r in faces),
                            'body_contained': sum(r['baseline']['contained'] for r in bodies),
                            'embedded_face_contained': sum(r['baseline']['faces_contained'][i] for r, i in embedded)},
               'models': {}}
    for identity in model_ids:
        summary['models'][identity] = {
            'face_exact_region': sum(r['models'][identity]['exact_region'] for r in faces),
            'face_contained': sum(r['models'][identity]['contained'] for r in faces),
            'face_wrong_to_exact': sum((not r['baseline']['exact_region']) and r['models'][identity]['exact_region'] for r in faces),
            'face_exact_to_wrong': sum(r['baseline']['exact_region'] and not r['models'][identity]['exact_region'] for r in faces),
            'body_contained': sum(r['models'][identity]['contained'] for r in bodies),
            'body_lost': sum(r['models'][identity]['body_lost'] for r in bodies),
            'embedded_face_contained': sum(r['models'][identity]['faces_contained'][i] for r, i in embedded),
            'gates': {'no_body_lost': not any(r['models'][identity]['body_lost'] for r in bodies)}}
    return summary


def evaluate_manifest(args):
    manifest = load_manifest(args.manifest)
    output = e.output_path(args.output)
    if rp.profile_for(output) is None:
        raise ValueError('Known-input evaluation output must belong to a registered research round')
    if output.exists():
        raise ValueError('Evaluation output already exists')
    config = e.configuration(args.config, classifier=getattr(args, 'classifier', None))
    probe = bool(getattr(args, 'probe', False))
    registration = probe_registration(manifest['sha256'])
    if probe and registration is None:
        raise ValueError('Probe mode requires a registered known-input development-use sidecar for this manifest')
    # Independent runs claim the manifest before any screenshot is read; probes never claim.
    first_use = None if probe else claim_first_use(manifest, output, config, getattr(args, 'config', None), getattr(args, 'dual', None) or ())
    baseline = e.FullBaseline(e.safe_path(args.baseline_models))
    models = [e.Recognizer(e.safe_path(path)) for path in args.candidate]
    model_ids = [m.binding['id'] for m in models]
    classifier = cls.CropClassifier(config['classifier']['file']) if config.get('classifier') else None
    common.disk_guard(output)
    output.mkdir(parents=True)
    scored, timings, predictions = [], [], []
    with (output / 'predictions-private.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for screen in manifest['value']['screens']:
            path = e.confined(manifest['root'], screen['file'])
            with Image.open(path) as source:
                source.load()
                image = source.convert('RGB')
            at = time.perf_counter()
            prediction = e.predict_scene(image, baseline, models, config, classifier=classifier)
            elapsed = time.perf_counter() - at
            timings.append(elapsed)
            results = score_screen(screen, prediction, model_ids)
            scored.append(results)
            predictions.append(prediction)
            stream.write(json.dumps({'file': screen['file'], 'sha256': common.digest(path), 'seconds': elapsed,
                                     'prediction': prediction, 'lines': results}, ensure_ascii=False, allow_nan=False) + '\n')
    summary = aggregate(scored, model_ids)
    summary['split'] = {'parents': sum(prediction['first_pass']['split_parents'] for prediction in predictions)}
    summary['diagnostics'] = diagnostics(scored, predictions, model_ids)
    report = {'schema_version': SCHEMA, 'completed_utc': common.utc(), 'manifest': {'file': manifest['file'], 'sha256': manifest['sha256'],
              'screens': len(manifest['value']['screens']), 'environment': manifest['value'].get('environment'),
              'label_origin': manifest['value'].get('label_origin'), 'typed_differences': manifest['value'].get('typed_differences')},
              'baseline': baseline.binding, 'models': [m.binding for m in models], 'config_sha256': e.hash_object(config),
              'classifier': config.get('classifier'), 'summary': summary, 'timing': e.percentiles(timings),
              'independent': not probe, 'probe': registration if probe else None, 'first_use': first_use,
              'scope': ('Private real screenshots with typed-input labels; development probe of a set already inspected per face; '
                        'counts and per-face outcomes only, never an independent claim') if probe else
                       'Private real screenshots with typed-input labels; independent evaluation only; no training, selection or threshold use',
              'promote': False, 'enabled': False}
    e.write_new(output / 'evaluation.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--baseline-models', required=True)
    parser.add_argument('--candidate', action='append', default=[])
    parser.add_argument('--config')
    parser.add_argument('--classifier')
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe', action='store_true', help='Run a manifest registered as a development probe (report is not independent)')
    parser.add_argument('--dual', action='append', default=[], help='Dual-freeze file bound into the independent first-use receipt')
    args = parser.parse_args(argv)
    report = evaluate_manifest(args)
    print(json.dumps({'schema_version': report['schema_version'], 'independent': report['independent'], 'summary': report['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
