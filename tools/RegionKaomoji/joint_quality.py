"""Fail-closed four-tier CPU quality and interleaved timing research proofs.

Pure ``summarize_*`` helpers exercise gate mechanics only. ``verify_round_*``
replays the original bound development predictions and truth masks before using
those helpers. Neither API writes evidence, runs inference, reads heldout data,
or grants first use. A paired freeze must store and later recompute the returned
proof. The preceding round's ``span_quality`` contract is left unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import math

from . import span_quality as span


TIERS = ('start', 'complete-exclusive', 'local-exclusive', 'joint')
POPULATIONS = ('anchor', 'fresh')
QUALITY_SCHEMA = 'localocr.joint-development-quality.v1'
LATENCY_SCHEMA = 'localocr.joint-repeated-latency.v1'
DEVELOPMENT_SCHEMA = 'localocr.joint-development-gates.v1'
CONFIGURATION_SCHEMA = 'localocr.joint-configurations.v1'
ROUND = 'pixel-owned-spans-20261020'
SOURCE = 'tools/RegionKaomoji/joint_quality.py'
_DISABLED = dict(promote=False, enabled=False, production_enabled=False, app_acceptance=False)


def _keys(value, expected, label):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError('Exactly the predeclared ' + label + ' are required')
    return value


def _views(scored, selected_id):
    _keys(scored, TIERS, 'four tiers')
    result = {}
    for tier in TIERS:
        _keys(scored[tier], POPULATIONS, 'two development populations')
        result[tier] = {name: span._evidence(scored[tier][name], selected_id) for name in POPULATIONS}
    reference = result['start']['anchor']['common']
    if SOURCE not in reference['implementation']:
        raise ValueError('Joint quality source must be included in frozen implementation')
    if any(value['common'] != reference for pair in result.values() for value in pair.values()):
        raise ValueError('Machine/model/baseline/source/thread/view bindings differ')
    for tier, pair in result.items():
        if pair['anchor']['binding']['config_sha256'] != pair['fresh']['binding']['config_sha256']:
            raise ValueError('Cross-population full configuration bindings differ')
        for name in POPULATIONS:
            start, value = result['start'][name], pair[name]
            if (start['dataset'] != value['dataset'] or start['targets'] != value['targets']
                    or start['populations'] != value['populations']
                    or start['binding'].get('inputs') != value['binding'].get('inputs')):
                raise ValueError('Paired target/data/protection populations differ')
            if scored[tier][name]['summary']['scenes'] != 60:
                raise ValueError('Each approved timing population requires 60 warm scenes')
            if scored[tier][name]['resources']['warmup_scenes'] != 3:
                raise ValueError('The owned scene worker must exclude its three warmups')
    anchor = result['start']['anchor']
    if (anchor['dataset']['dataset_sha256'] != span.ANCHOR_DATASET_SHA256
            or anchor['dataset']['split_sha256']['dev'] != span.ANCHOR_DEV_SHA256 or anchor['count'] != 244):
        raise ValueError('Approved 244-target anchor identity differs')
    if anchor['dataset']['dataset_sha256'] == result['start']['fresh']['dataset']['dataset_sha256']:
        raise ValueError('Fresh development cannot reuse the anchor dataset')
    if len({pair['anchor']['binding']['config_sha256'] for pair in result.values()}) != len(TIERS):
        raise ValueError('Four distinct predeclared complete configurations are required')
    return result


def _quality_pair(start, candidate, *, anchor):
    lost, gained = start['correct'] - candidate['correct'], candidate['correct'] - start['correct']
    checks = {'exact_improved': candidate['exact'] > start['exact'],
              'all_start_correct_retained': not lost,
              'zero_new_body_and_negative_errors': not any(candidate['errors'].values())}
    if anchor:
        checks.update(safe_complete_at_least_100=candidate['safe'] >= 100,
                      strict_exact_at_least_30=candidate['exact'] >= 30)
    return {'count': candidate['count'], 'start_exact': start['exact'], 'final_exact': candidate['exact'],
            'exact_gain': candidate['exact'] - start['exact'], 'gained_targets': len(gained),
            'lost_targets': len(lost), 'start_safe_complete': start['safe'],
            'final_safe_complete': candidate['safe'], 'protection_population': candidate['populations'],
            'final_new_errors': candidate['errors'], 'target_set_sha256': span._hash(sorted(candidate['targets'])),
            'start_correct_targets_sha256': span._hash(sorted(start['correct'])),
            'final_correct_targets_sha256': span._hash(sorted(candidate['correct'])),
            'checks': checks, 'passed': all(checks.values())}


def summarize_quality(scored, selected_id):
    """Check all three candidate tiers independently, without a latency gate.

    ``scored`` maps all four ``TIERS`` to ``anchor``/``fresh`` already verified
    evidence views, as produced by the wrapper. Synthetic caller-supplied counts
    are suitable only for mechanism tests; they are not a freeze proof.
    """
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError('Selected specialist identity required')
    views = _views(scored, selected_id)
    tiers = {}
    for tier in TIERS[1:]:
        anchor = _quality_pair(views['start']['anchor'], views[tier]['anchor'], anchor=True)
        fresh = _quality_pair(views['start']['fresh'], views[tier]['fresh'], anchor=False)
        tiers[tier] = {'anchor': anchor, 'fresh_development': fresh,
                       'passed': anchor['passed'] and fresh['passed']}
    eligible = [tier for tier in TIERS[1:] if tiers[tier]['passed']]
    return {'schema_version': QUALITY_SCHEMA, 'selected_id': selected_id,
            'evidence_kind': 'pure quality mechanics; verified worker proof required before freeze',
            'counting': 'strict regional exact raw Unicode; no width or tail normalization',
            'tiers': tiers, 'quality_eligible_tiers': eligible, 'latency_status': 'NOT_RUN',
            'common_binding_sha256': span._hash(views['start']['anchor']['common']),
            'selected_tier': 'start', 'passed': False, **_DISABLED}


def prediction_identity(records):
    """Hash every semantic worker output, stripping only known timing fields.

    Do not remove arbitrary keys by suffix: scores, geometry, decisions, views,
    contexts, native baseline and every unknown future field remain significant.
    Resource PID/wall time lives in a separate output and is not hashed here.
    """
    if not isinstance(records, list) or not records:
        raise ValueError('Nonempty original worker predictions required')
    stable = deepcopy(records)
    for record in stable:
        span._object(record, 'worker prediction')
        record.pop('seconds', None)
        prediction = span._object(record.get('prediction'), 'scene prediction')
        prediction.pop('timings', None)
        for model in span._object(prediction.get('models'), 'prediction models').values():
            span._object(model, 'prediction model')
            for key in ('specialist_seconds', 'fusion_seconds', 'view_latency'):
                model.pop(key, None)
    return span._hash(stable)


def _quality_identity(view):
    # All recomputed metrics and original target sets, not just the final count.
    return span._hash({'summary': view['summary'], 'target_correctness': view['target_correctness']})


def _samples(view, samples):
    from tools.PretrainedOcr.evaluate import percentiles

    if (view['resources']['warmup_scenes'] != 3 or not isinstance(samples, list)
            or len(samples) != view['summary']['scenes']):
        raise ValueError('Every original warm scene must provide one latency sample')
    for sample in samples:
        span._positive(sample, 'raw warm scene latency')
    if view['resources']['warm'] != percentiles(samples):
        raise ValueError('Original warm samples differ from the bound resource percentiles')
    total = view['resources']['total_warm_seconds']
    if not math.isclose(total, sum(samples), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Original warm samples differ from total warm seconds')
    return samples


def summarize_repeated_latency(scored, tier, repetitions, selected_id, *, scored_output_sha256=None):
    """Pool all three fixed interleaved repetitions of a quality-eligible tier.

    Each population has six entries in strict chronological start1/candidate1/
    start2/candidate2/start3/candidate3 order. An entry contains a recomputed
    ``view``, all original ``seconds`` and ``output_sha256`` (semantic prediction
    identity). Only the wrapper supplies those values as real worker evidence.
    ``scored_output_sha256`` maps population to start/tier fingerprints; omitting
    it is permitted for pure fixture gate mechanics only.
    """
    from tools.PretrainedOcr import quality_evaluate as existing
    from tools.PretrainedOcr.evaluate import percentiles

    verdict = summarize_quality(scored, selected_id)
    if tier not in verdict['quality_eligible_tiers']:
        raise ValueError('Repeated latency requires a quality-eligible predeclared tier')
    _keys(repetitions, POPULATIONS, 'two repeated populations')
    evaluated = _views(scored, selected_id)
    result, all_candidate = {}, []
    for population in POPULATIONS:
        rows = repetitions[population]
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError('Exactly six interleaved repetition reports required')
        pools = {'start': [], tier: []}
        previous = max(existing.timestamp(scored[role][name]['completed_utc'])
                       for role in TIERS for name in POPULATIONS)
        for index, row in enumerate(rows):
            span._object(row, 'repeat evidence')
            view = span._object(row.get('view'), 'repeat view')
            role = 'start' if index % 2 == 0 else tier
            value = span._evidence(view, selected_id)
            reference = evaluated[role][population]
            if value['binding'] != reference['binding'] or value['common'] != reference['common']:
                raise ValueError('Interleaved repeat configuration/data/machine/source bindings differ')
            stamp = existing.timestamp(view['completed_utc'])
            if stamp <= previous:
                raise ValueError('Repeated reports must follow strict scored/interleaved chronological order')
            previous = stamp
            if _quality_identity(view) != _quality_identity(scored[role][population]):
                raise ValueError('Repeated quality outputs differ from scored outputs')
            output = span._sha(row.get('output_sha256'), 'semantic worker output SHA256')
            if scored_output_sha256 is not None:
                if output != scored_output_sha256[population][role]:
                    raise ValueError('Repeated semantic worker output drift')
            pools[role].extend(_samples(view, row.get('seconds')))
        start, candidate = percentiles(pools['start']), percentiles(pools[tier])
        ratio = candidate['p95_seconds'] / start['p95_seconds']
        if not math.isfinite(ratio):
            raise ValueError('Nonfinite repeated latency ratio')
        result[population] = {'start': start, 'candidate': candidate, 'ratio': ratio,
                              'passed': ratio <= 1.15}
        all_candidate.extend(pools[tier])
    return {'schema_version': LATENCY_SCHEMA, 'tier': tier, **result,
            'evidence_kind': 'pure repeat mechanics; verified worker proof required before freeze',
            'repetitions': 3, 'order': ['start', tier] * 3,
            'quantile': 'tools.PretrainedOcr.evaluate.percentiles (NumPy percentile 95)',
            'pooled_candidate': percentiles(all_candidate),
            'passed': all(item['passed'] for item in result.values()), **_DISABLED}


def select_tier(quality, latency):
    """Choose a unique fully eligible tier using only the declared rank order.

    This helper consumes summaries for fixture tests; the wrapper recomputes
    them. Supplying missing or extra timing tiers is an error, not implicit
    eligibility. No candidate passing both gates returns start/FAIL.
    """
    eligible = quality.get('quality_eligible_tiers')
    tiers = _keys(quality.get('tiers'), TIERS[1:], 'candidate quality tiers')
    if (quality.get('schema_version') != QUALITY_SCHEMA or not isinstance(eligible, list)
            or any(type(tiers[tier].get('passed')) is not bool for tier in TIERS[1:])
            or eligible != [tier for tier in TIERS[1:] if tiers[tier]['passed']]):
        raise ValueError('Malformed predeclared quality verdict')
    _keys(latency, eligible, 'quality-eligible latency tiers')
    ranking = []
    for tier in eligible:
        item = span._object(latency[tier], 'tier latency')
        if type(item.get('passed')) is not bool:
            raise ValueError('Explicit repeated latency verdict required')
        p95 = span._positive(span._object(item.get('pooled_candidate'), 'pooled candidate').get('p95_seconds'), 'pooled p95')
        if item['passed']:
            measured = quality['tiers'][tier]
            ranking.append(( -measured['fresh_development']['final_exact'], -measured['anchor']['final_exact'],
                             p95, TIERS.index(tier), tier))
    ranking.sort()
    return {'selected_tier': ranking[0][-1] if ranking else 'start', 'passed': bool(ranking),
            'ranked_eligible_tiers': [item[-1] for item in ranking],
            'ranking': 'fresh exact descending; anchor exact descending; both-population pooled candidate p95 ascending; declared order',
            **_DISABLED}


def _manifest(configuration_manifest, root, selected_id, existing, scene, sources):
    ref = span._object(configuration_manifest, 'pre-scoring configuration manifest reference')
    path = scene.safe_path(ref.get('file', ''))
    if path == root or not path.is_relative_to(root) or existing.reference(path) != ref:
        raise ValueError('Predeclared configuration manifest bytes/path differ')
    value = span._object(span._read(path), 'configuration manifest')
    if (value.get('schema_version') != CONFIGURATION_SCHEMA or value.get('selected_id') != selected_id
            or value.get('source_inventory') != sources):
        raise ValueError('Predeclared configuration manifest source/model identity differs')
    existing.timestamp(value['frozen_utc'])
    _keys(value.get('configurations'), TIERS, 'four manifest configurations')
    _keys(value.get('populations'), POPULATIONS, 'two manifest populations')
    return value, dict(ref)


def _root(round_root, existing, scene):
    root = scene.safe_path(round_root)
    profile = existing.rp.profile_for(root)
    if profile is None or profile.name != ROUND or root != Path(profile.root).resolve():
        raise ValueError('Joint development proof requires its registered round')
    return root


def _paths(scored, root, scene, used):
    _keys(scored, TIERS, 'four scored tiers')
    result = {}
    for tier in TIERS:
        _keys(scored[tier], POPULATIONS, 'two scored populations')
        result[tier] = {}
        for name in POPULATIONS:
            result[tier][name] = _unique_path(scored[tier][name], root, scene, used)
    return result


def _unique_path(path, root, scene, used):
    path = scene.output_path(path)
    if path in used or not path.is_relative_to(root):
        raise ValueError('Distinct original current-round report/output paths required')
    used.add(path)
    return path


def _verified(path, selected_id, manifest, tier, population, root, existing, scene, used):
    before = existing.reference(path)
    declared = span._object(span._read(path), 'original development report')
    # Reject wrong populations/splits before invoking replay or opening truth.
    if declared.get('split') != 'dev' or declared.get('independent_heldout') is not False:
        raise ValueError('Only original development reports may enter joint verification')
    binding = span._object(declared.get('pipeline_binding'), 'pipeline_binding')
    expected = manifest['populations'][population]
    if (binding['config'] != manifest['configurations'][tier]
            or binding['dataset'] != expected.get('dataset') or binding['inputs'] != expected.get('inputs')
            or binding['models'] != manifest.get('models') or binding['baseline'] != manifest.get('baseline')
            or existing.timestamp(declared['completed_utc']) <= existing.timestamp(manifest['frozen_utc'])):
        raise ValueError('Scored/repeated report differs from predeclared configuration/population/model/time')
    view, proof = span._verified_view(path, selected_id, existing, scene)
    if proof['report'] != before or view['pipeline_binding'] != binding:
        raise ValueError('Original development report changed before/during replay')
    for output in proof['outputs'].values():
        _unique_path(output['file'], root, scene, used)
    dataset_root = Path(binding['inputs']['file']).parent
    jobs = scene.image_jobs(dataset_root, 'dev', binding['inputs']['file'])
    records = scene.read_predictions(proof['outputs']['predictions']['file'], jobs)
    proof['semantic_output_sha256'] = prediction_identity(records)
    # The additional fingerprint read must not open a TOCTOU gap after replay.
    _recheck(proof, existing)
    return view, proof, [record.get('seconds') for record in records]


def _recheck(proof, existing):
    for ref in (proof['report'], *proof['outputs'].values()):
        if existing.reference(ref['file']) != ref:
            raise ValueError('Bound original report/output changed during joint verification')


def _scored(scored_reports, selected_id, root, manifest, existing, scene, used):
    paths = _paths(scored_reports, root, scene, used)
    scored, proofs = {}, {}
    for tier in TIERS:
        scored[tier], proofs[tier] = {}, {}
        for population in POPULATIONS:
            view, proof, _ = _verified(paths[tier][population], selected_id, manifest, tier, population,
                                       root, existing, scene, used)
            scored[tier][population], proofs[tier][population] = view, proof
    return scored, proofs


def _finish(sources, manifest_ref, scored_proofs, repeat_proofs, existing):
    for pair in scored_proofs.values():
        for proof in pair.values():
            _recheck(proof, existing)
    for pair in repeat_proofs.values():
        for proofs in pair.values():
            for proof in proofs:
                _recheck(proof, existing)
    if existing.source_inventory() != sources:
        raise ValueError('Sources changed during joint development verification')
    if existing.reference(manifest_ref['file']) != manifest_ref:
        raise ValueError('Configuration manifest changed during joint development verification')


def verify_round_quality(scored_reports, selected_id, *, round_root, configuration_manifest):
    """Replay all eight scored CPU reports and return quality-only eligibility.

    Arguments are exact tier/population mappings of existing evaluation paths
    plus a bound pre-scoring configuration manifest. This deterministic result
    never substitutes for the repeated latency proof required by paired freeze.
    """
    from tools.PretrainedOcr import quality_evaluate as existing
    from tools.RegionKaomoji import evaluate as scene

    root = _root(round_root, existing, scene)
    sources = existing.source_inventory()
    manifest, manifest_ref = _manifest(configuration_manifest, root, selected_id, existing, scene, sources)
    scored, proofs = _scored(scored_reports, selected_id, root, manifest, existing, scene, set())
    result = summarize_quality(scored, selected_id)
    _finish(sources, manifest_ref, proofs, {}, existing)
    return {**result, 'evidence_kind': 'recomputed bound original-pixel CPU quality only',
            'reports': proofs, 'configuration_manifest': manifest_ref, 'source_inventory': sources}


def verify_round_development(scored_reports, repeated_reports, selected_id, *, round_root, configuration_manifest):
    """Recompute scored/repeated reports and return a bindable freeze proof.

    ``repeated_reports`` must contain exactly all quality-eligible tiers, each
    with anchor/fresh lists of six distinct original reports. Empty mapping is
    required on total quality failure and yields start/FAIL, latency NOT_RUN.
    Paths and all bound prediction/resource/alignment outputs cannot be reused
    across scored reports, tiers or repetitions. All bytes and source inventory
    are checked again at completion. No caller-supplied aggregate is accepted.
    """
    from tools.PretrainedOcr import quality_evaluate as existing
    from tools.RegionKaomoji import evaluate as scene

    root = _root(round_root, existing, scene)
    sources = existing.source_inventory()
    manifest, manifest_ref = _manifest(configuration_manifest, root, selected_id, existing, scene, sources)
    used = set()
    scored, proofs = _scored(scored_reports, selected_id, root, manifest, existing, scene, used)
    quality = summarize_quality(scored, selected_id)
    eligible = quality['quality_eligible_tiers']
    _keys(repeated_reports, eligible, 'quality-eligible repeated tiers')
    latency, repeated_proofs = {}, {}
    for tier in eligible:
        _keys(repeated_reports[tier], POPULATIONS, 'two repeated populations')
        repeated, repeated_proofs[tier] = {}, {}
        signatures = {name: {role: proofs[role][name]['semantic_output_sha256'] for role in ('start', tier)}
                      for name in POPULATIONS}
        for population in POPULATIONS:
            paths = repeated_reports[tier][population]
            if not isinstance(paths, list) or len(paths) != 6:
                raise ValueError('Exactly six distinct interleaved raw reports required')
            repeated[population], repeated_proofs[tier][population] = [], []
            for index, path in enumerate(paths):
                role = 'start' if index % 2 == 0 else tier
                path = _unique_path(path, root, scene, used)
                view, proof, seconds = _verified(path, selected_id, manifest, role, population,
                                                root, existing, scene, used)
                repeated[population].append({'view': view, 'seconds': seconds,
                                              'output_sha256': proof['semantic_output_sha256']})
                repeated_proofs[tier][population].append(proof)
        latency[tier] = summarize_repeated_latency(scored, tier, repeated, selected_id,
                                                   scored_output_sha256=signatures)
        latency[tier]['evidence_kind'] = 'recomputed bound original-pixel CPU interleaved repetitions'
    selected = select_tier(quality, latency)
    _finish(sources, manifest_ref, proofs, repeated_proofs, existing)
    return {'schema_version': DEVELOPMENT_SCHEMA, 'selected_id': selected_id,
            'evidence_kind': 'recomputed bound original-pixel CPU joint development',
            'quality': quality, 'latency': latency, 'latency_status': 'MEASURED' if eligible else 'NOT_RUN',
            'reports': proofs, 'repeated_reports': repeated_proofs,
            'configuration_manifest': manifest_ref, 'source_inventory': sources, **selected}
