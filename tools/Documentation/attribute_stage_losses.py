"""Attribute every development face of recorded scene reports to the stage where it is lost.

Read-only recount over finished CPU reports: the frozen truth, visible-ink masks, automatic
candidates, routing decisions, specialist views and final regions already recorded in each
report. No model inference, no candidate generation, no threshold selection and no heldout
access. Categories reuse the 2026-10-19 development attribution so counts stay comparable:

  nohit > partial-only > complete-but-body/graphicunsafe > safe-unrouted
  > safe-routed-raw-wrong > safe-routed-raw-correct

A face falls into the first category that applies. "Raw correct" means the raw specialist
view of a safe, complete, routed candidate equals the truth; it is a diagnostic upper bound
on the recorded crops, never an automatic selection rule. Body protection counters are the
evaluator's own increments, recomputed per body region so the regions behind each count are
named. The private output keeps texts; the public output keeps only counts and identifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.RegionKaomoji import evaluate as scene

SCHEMA = 'localocr.stage-attribution.v1'
CATEGORIES = ('nohit', 'partial-only', 'complete-but-body/graphicunsafe',
              'safe-unrouted', 'safe-routed-raw-wrong', 'safe-routed-raw-correct')
BODY_COUNTERS = ('baseline_correct_to_wrong', 'erroneous_rewrites',
                 'new_body_intersection_errors', 'body_literal_deleted')
COVERAGE_BINS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0))


def read(path: Path) -> dict:
    def refuse(value: str) -> None:
        raise ValueError('Nonfinite JSON evidence: ' + value)
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        return json.load(stream, parse_constant=refuse)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open('x', encoding='utf-8', newline='\n') as stream:  # exclusive: never overwrite earlier evidence
        stream.write(text + '\n')


def by_id(records: list[dict], key: str = 'candidate_id') -> dict[str, dict]:
    index = {}
    for record in records:
        if record[key] in index:
            raise ValueError('Duplicate identity in recorded report: ' + str(record[key]))
        index[record[key]] = record
    return index


def coverage_bin(ratio: float) -> str:
    for low, high in COVERAGE_BINS:
        if low <= ratio < high:
            return f'[{low},{high})'
    return '1.0' if ratio >= 1.0 else 'invalid'


def categorize(truth: dict, mask: np.ndarray, candidates: dict, views: dict,
               body_union: np.ndarray, graphic_union: np.ndarray) -> dict:
    """Stage category of one face, from the recorded candidates and specialist views only."""
    hits = [item for item in candidates.values() if scene.mask_pixels(mask, item['bbox']) > 0]
    complete = [item for item in hits if scene.mask_pixels(mask, item['bbox']) == truth['pixel_count']]
    safe = [item for item in complete if not scene.mask_pixels(body_union, item['bbox'])
            and not scene.mask_pixels(graphic_union, item['bbox'])]
    routed = [item for item in safe if views.get(item['candidate_id'])]
    raw_correct = []
    for item in routed:
        raw = [view for view in views[item['candidate_id']] if view['transform'] == 'raw']
        if len(raw) != 1:
            raise ValueError('Safe routed candidate requires exactly one recorded raw view')
        if raw[0]['text'] == truth['text']:
            raw_correct.append(item)
    category = ('nohit' if not hits else 'partial-only' if not complete else
                'complete-but-body/graphicunsafe' if not safe else 'safe-unrouted' if not routed else
                'safe-routed-raw-correct' if raw_correct else 'safe-routed-raw-wrong')
    best = max(safe or hits, key=lambda item: scene.mask_pixels(mask, item['bbox']), default=None)
    unsafe_kind = None
    if category == 'complete-but-body/graphicunsafe':
        body = any(scene.mask_pixels(body_union, item['bbox']) for item in complete)
        graphic = any(scene.mask_pixels(graphic_union, item['bbox']) for item in complete)
        unsafe_kind = 'body+graphic' if body and graphic else 'body' if body else 'graphic'
    return {'category': category, 'hit_ids': [item['candidate_id'] for item in hits],
            'complete_ids': [item['candidate_id'] for item in complete],
            'safe_ids': [item['candidate_id'] for item in safe],
            'routed_ids': [item['candidate_id'] for item in routed],
            'raw_correct_ids': [item['candidate_id'] for item in raw_correct],
            'best_coverage': (scene.mask_pixels(mask, best['bbox']) / truth['pixel_count']) if best else 0.0,
            'unsafe_kind': unsafe_kind,
            'routed_any_hit': any(views.get(item['candidate_id']) for item in hits)}


def conflict_signature(model: dict, candidate_id: str) -> list[str]:
    """Recorded atomic-ownership conflicts that name this candidate, if the tier records them."""
    conflicts = model.get('split', {}).get('atomic_conflicts') or []
    return sorted({item['reason'] for item in conflicts if candidate_id in (item.get('candidate_ids') or [])})


def loss_signature(model: dict, decisions: dict, raw_correct_ids: list[str]) -> dict:
    """Why a raw-correct face still ends wrong: fusion decision, retention and ownership conflicts.

    ``fusion-rejected`` needs every raw-correct candidate's recorded decision to be not accepted. An accepted
    candidate missing from the final regions is ``accepted-then-parent-restored`` when a recorded atomic
    ownership conflict names it, otherwise ``accepted-absent-from-final-regions`` (the 10-19 absent flag).
    """
    reasons, conflicts, retained, accepted = set(), set(), False, False
    for identity in raw_correct_ids:
        if identity not in decisions:
            raise ValueError('Raw-correct candidate lacks a recorded fusion decision: ' + identity)
        decision = decisions[identity]
        if not isinstance(decision.get('accepted'), bool):
            raise ValueError('Fusion decision lacks its recorded accepted flag: ' + identity)
        reasons.add(decision['reason'])
        accepted |= decision['accepted']
        conflicts.update(conflict_signature(model, identity))
        retained |= any(region.get('candidate_id') == identity for region in model['regions'])
    kind = ('fusion-rejected' if not accepted else 'accepted-but-final-not-exact' if retained else
            'accepted-then-parent-restored' if conflicts else 'accepted-absent-from-final-regions')
    return {'kind': kind, 'decision_reasons': sorted(reasons), 'conflicts': sorted(conflicts),
            'accepted': accepted, 'retained': retained}


def face_record(row: dict, index: int, prediction: dict, identity: str, masks: list, config: dict,
                unions: tuple, candidates: dict, decisions: dict, skipped: dict) -> dict:
    truth, mask = row['regions'][index], masks[index]
    model = prediction['models'][identity]
    stage = categorize(truth, mask, candidates, model['views'], *unions)
    baseline_text, _ = scene.aligned_region(prediction['baseline'], index, masks, config)
    final_text, final_mixed = scene.aligned_region(model['regions'], index, masks, config)
    record = {'scene_id': row['id'], 'truth_id': truth['id'], 'pixel_sha256': row['pixel_sha256'], 'category': stage['category'],
              'coverage_bin': coverage_bin(stage['best_coverage']), 'unsafe_kind': stage['unsafe_kind'],
              'routed_any_hit': stage['routed_any_hit'], 'final_exact': final_text == truth['text'],
              'final_mixed_regions': final_mixed, 'baseline_exact': baseline_text == truth['text'],
              'safe_ids': stage['safe_ids'], 'routed_ids': stage['routed_ids'],
              'raw_correct_ids': stage['raw_correct_ids'], 'truth_text': truth['text'],
              'baseline_text': baseline_text, 'final_text': final_text, 'unrouted_reasons': [],
              'loss': None}
    if stage['category'] == 'safe-unrouted':
        reasons = set()
        for candidate_id in stage['safe_ids']:
            if candidate_id not in skipped:
                raise ValueError('Unrouted safe candidate lacks its recorded skip reason: ' + candidate_id)
            reasons.add(skipped[candidate_id]['reason'])
        record['unrouted_reasons'] = sorted(reasons)
    if stage['raw_correct_ids'] and not record['final_exact']:
        record['loss'] = loss_signature(model, decisions, stage['raw_correct_ids'])
    return record


def body_record(row: dict, index: int, prediction: dict, identity: str, masks: list, config: dict) -> dict | None:
    truth = row['regions'][index]
    baseline_text, mixed = scene.aligned_region(prediction['baseline'], index, masks, config)
    counter = scene.body_increments(prediction, identity, truth, index, masks, config, baseline_text, mixed)
    flagged = {key: int(counter[key]) for key in BODY_COUNTERS if counter[key]}
    if not flagged:
        return None
    final_text, _ = scene.aligned_region(prediction['models'][identity]['regions'], index, masks, config)
    return {'scene_id': row['id'], 'truth_id': truth['id'], 'counters': flagged, 'truth_text': truth['text'],
            'baseline_text': baseline_text, 'final_text': final_text,
            'transcript_changed': bool(counter['body_intersection_transcript_changes'])}


def require_development_report(report: dict, label: str) -> None:
    """Refuse anything but a finished development report; heldout reports and claims are never read."""
    if report.get('split') != 'dev' or report.get('independent_heldout') is not False or report.get('heldout_usage') is not None:
        raise ValueError('Stage attribution reads finished development reports only: ' + label)
    if report.get('actual_cpu_inference') is not True or report.get('ground_truth_inference_input') is not False:
        raise ValueError('Report is not an actual truth-free CPU development report: ' + label)


def analyze(label: str, path: Path, identity: str) -> dict:
    report = read(path)
    require_development_report(report, label)
    binding = report['pipeline_binding']
    config = binding['config']
    root = Path(binding['inputs']['file']).parent
    if digest(report['outputs']['predictions']['file']) != report['outputs']['predictions']['sha256']:
        raise ValueError('Recorded predictions changed after the report was written: ' + label)
    jobs = scene.image_jobs(root, 'dev', binding['inputs']['file'])
    rows = scene.load_truth(root, 'dev', jobs)
    predictions = scene.read_predictions(report['outputs']['predictions']['file'], jobs)
    faces, bodies, seen = [], [], set()
    for row, record in zip(rows, predictions):
        if record['id'] != row['id'] or record['image_sha256'] != row['sha256'] or record['pixel_sha256'] != row['pixel_sha256']:
            raise ValueError('Attribution row and recorded prediction differ: ' + row['id'])
        if row['id'] in seen:
            raise ValueError('Duplicate development scene identity: ' + row['id'])
        seen.add(row['id'])
        prediction = record['prediction']
        masks = scene.masks_for_row(root, row)
        body_union = np.zeros((row['height'], row['width']), dtype=bool)
        graphic_union = np.zeros_like(body_union)
        for truth, mask in zip(row['regions'], masks):
            if truth['kind'] == 'body':
                body_union |= mask
            elif truth['kind'] == 'graphic':
                graphic_union |= mask
        candidates = by_id(prediction['routing']['candidates'])
        decisions = by_id(prediction['models'][identity]['decisions'])
        skipped = by_id(prediction['view_budget']['skipped'])
        for index, truth in enumerate(row['regions']):
            if truth['kind'] == 'kaomoji':
                faces.append(face_record(row, index, prediction, identity, masks, config,
                                         (body_union, graphic_union), candidates, decisions, skipped))
            elif truth['kind'] == 'body':
                body = body_record(row, index, prediction, identity, masks, config)
                if body:
                    bodies.append(body)
    summary = report['summary']['models'][identity]
    return {'label': label, 'report': str(path), 'report_sha256': digest(path), 'faces': faces, 'bodies': bodies,
            'recorded': {'exact': summary['isolated_kaomoji']['exact'], 'faces': summary['isolated_kaomoji']['count'],
                         'body': summary['body'], 'localization': report['summary']['localization']}}


def counts(analysis: dict) -> dict:
    """Anonymous counts of one report: per-category totals, exact-by-category and loss signatures."""
    faces = analysis['faces']
    categories = Counter(dict.fromkeys(CATEGORIES, 0))
    exact = Counter(dict.fromkeys(CATEGORIES, 0))
    for face in faces:
        categories[face['category']] += 1
        exact[face['category']] += int(face['final_exact'])
    losses = [face for face in faces if face['loss']]
    partial = [face for face in faces if face['category'] == 'partial-only']
    unsafe = [face for face in faces if face['category'] == 'complete-but-body/graphicunsafe']
    unrouted = [face for face in faces if face['category'] == 'safe-unrouted']
    recorded = analysis['recorded']
    if sum(exact.values()) != recorded['exact'] or sum(categories.values()) != recorded['faces']:
        raise ValueError('Attribution totals differ from the recorded report summary: ' + analysis['label'])
    localization = recorded['localization']
    if (categories['nohit'] != localization['missed'] or categories['partial-only'] != localization['partial_only']
            or categories['safe-unrouted'] + categories['safe-routed-raw-wrong'] + categories['safe-routed-raw-correct']
            != localization['safe_complete_coverage']
            or categories['safe-routed-raw-wrong'] + categories['safe-routed-raw-correct'] != localization['safe_routed_complete_coverage']):
        raise ValueError('Attribution stages differ from the recorded localization counters: ' + analysis['label'])
    if len(losses) != categories['safe-routed-raw-correct'] - exact['safe-routed-raw-correct']:
        raise ValueError('Loss signatures do not partition the raw-correct final-wrong faces: ' + analysis['label'])
    flagged = sum((Counter(body['counters']) for body in analysis['bodies']), Counter())
    if any(flagged[key] != recorded['body'].get(key, 0) for key in BODY_COUNTERS):
        raise ValueError('Named body regions differ from the recorded body protection counters: ' + analysis['label'])
    return {'label': analysis['label'], 'report': analysis['report'], 'report_sha256': analysis['report_sha256'],
            'faces': recorded['faces'], 'final_exact': recorded['exact'],
            'stage_partition': dict(categories), 'final_exact_by_stage': dict(exact),
            'raw_correct_final_wrong': {'count': len(losses),
                                        'kinds': dict(Counter(face['loss']['kind'] for face in losses)),
                                        'decision_reasons': dict(Counter(r for face in losses for r in face['loss']['decision_reasons'])),
                                        'conflicts': dict(Counter(r for face in losses for r in face['loss']['conflicts'])),
                                        'target_ids': [[face['scene_id'], face['truth_id']] for face in losses]},
            'partial_only': {'coverage_bins': dict(Counter(face['coverage_bin'] for face in partial)),
                             'routed_partial_candidate': sum(face['routed_any_hit'] for face in partial)},
            'complete_unsafe': {'kinds': dict(Counter(face['unsafe_kind'] for face in unsafe)),
                                'routed_any_hit': sum(face['routed_any_hit'] for face in unsafe)},
            'safe_unrouted_reasons': dict(Counter(r for face in unrouted for r in face['unrouted_reasons'])),
            'baseline_exact': sum(face['baseline_exact'] for face in faces),
            'body_regions_flagged': {'count': len(analysis['bodies']),
                                     'counters': dict(flagged),
                                     'target_ids': [[body['scene_id'], body['truth_id']] for body in analysis['bodies']]},
            'recorded_body': recorded['body']}


def compare(start: dict, other: dict) -> dict:
    """Same-population transitions from the start tier to another tier, with loss and gain signatures."""
    base = {(face['scene_id'], face['truth_id']): face for face in start['faces']}
    if len(base) != len(start['faces']) or len(base) != len(other['faces']):
        raise ValueError('Tier populations differ in size or repeat a face identity')
    transitions, lost, gained = Counter(), [], []
    for face in other['faces']:
        key = (face['scene_id'], face['truth_id'])
        if key not in base:
            raise ValueError('Tier populations differ: ' + str(key))
        before = base[key]
        if before['truth_text'] != face['truth_text'] or before['pixel_sha256'] != face['pixel_sha256']:
            raise ValueError('Tier populations share an identity but not its truth or pixels: ' + str(key))
        transitions[before['category'] + ' -> ' + face['category']] += 1
        if before['final_exact'] and not face['final_exact']:
            lost.append({'scene_id': key[0], 'truth_id': key[1], 'category_before': before['category'],
                         'category_after': face['category'], 'loss': face['loss']})
        elif face['final_exact'] and not before['final_exact']:
            gained.append({'scene_id': key[0], 'truth_id': key[1], 'category_before': before['category'],
                           'category_after': face['category']})
    return {'start': start['label'], 'tier': other['label'], 'category_transitions': dict(transitions),
            'lost_targets': lost, 'gained_targets': gained,
            'lost_signatures': dict(Counter((item['loss'] or {}).get('kind') or 'not-raw-correct' for item in lost))}


def parse_reports(values: list[str]) -> list[tuple[str, Path]]:
    reports = []
    for value in values:
        label, _, path = value.partition('=')
        if not label or not path:
            raise ValueError('Report argument must be label=path: ' + value)
        reports.append((label, Path(path)))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', action='append', required=True, help='label=path of a finished evaluation.json')
    parser.add_argument('--model', required=True, help='specialist identity scored in the reports')
    parser.add_argument('--compare', action='append', default=[], help='start_label:tier_label pairs sharing one population')
    parser.add_argument('--private', required=True, type=Path, help='ignored-directory output with texts')
    parser.add_argument('--public', required=True, type=Path, help='counts-only output for the documentation')
    args = parser.parse_args()
    analyses = {label: analyze(label, path, args.model) for label, path in parse_reports(args.report)}
    comparisons = []
    for value in args.compare:
        start, _, tier = value.partition(':')
        comparisons.append(compare(analyses[start], analyses[tier]))
    generated = datetime.now(timezone.utc).isoformat()
    public = {'schema_version': SCHEMA, 'generated_utc': generated, 'model': args.model,
              'evidence_kind': 'Read-only recount of recorded development reports; no inference, no heldout access',
              'categories': list(CATEGORIES), 'reports': {label: counts(item) for label, item in analyses.items()},
              'comparisons': comparisons}
    write(args.private, dict(public, faces={label: item['faces'] for label, item in analyses.items()},
                             bodies={label: item['bodies'] for label, item in analyses.items()}))
    public['private_output'] = {'file': str(args.private), 'sha256': digest(args.private)}
    write(args.public, public)
    print(json.dumps({label: {'faces': item['faces'], 'final_exact': item['final_exact'],
                              'stage_partition': item['stage_partition'],
                              'raw_correct_final_wrong': item['raw_correct_final_wrong']['count'],
                              'bodies_flagged': item['body_regions_flagged']['count']}
                      for label, item in public['reports'].items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
