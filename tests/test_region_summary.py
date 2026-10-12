"""Opaque provenance closure tests; fixture counts are not OCR evaluation evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.Documentation import build_region_summary as b
from tools.Documentation import plot_region_evaluation as p
from tests.test_region_plot_schema import fixture as chart_fixture

SHA = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def opaque(path, content=b'PRIVATE_FIXTURE_CONTENT_NOT_JSON'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {'file': str(path), 'sha256': SHA(path)}


def refs(path):
    return {'file': str(path), 'sha256': SHA(path)}


def sources(tmp_path, monkeypatch):
    """Create separate fake model/source/CPU evidence, never execute a model."""
    root = tmp_path / 'repo'; artifact = root / 'artifacts'; artifact.mkdir(parents=True)
    monkeypatch.setattr(b, 'ROOT', root); monkeypatch.setattr(b, 'ARTIFACTS', artifact)
    impl = {}
    for name in b.IMPLEMENTATION_FILES:
        path = root / name
        opaque(path)
        impl[name] = SHA(path)
    example = chart_fixture()
    frozen = {'schema_version': 'localocr.region-scenes.freeze.v1', 'dataset_sha256': example['dataset']['sha256'],
              'split_sha256': example['dataset']['split_sha256'], 'counts': deepcopy(p.COUNTS)}
    inputs = artifact / 'scenes/inference-inputs.json'
    opaque(inputs)
    write(inputs.parent / 'freeze.json', frozen)
    config = {'schema_version': 'localocr.region-pipeline-config.v1', 'candidate': deepcopy(b.CANDIDATE_CONFIG),
              'fusion': deepcopy(b.FUSION_CONFIG),
              'views': ['raw', 'edge-pad-6'], 'limits': deepcopy(b.LIMITS),
              'join': {'same_line': '  ', 'different_line': '\n', 'vertical_overlap': .5}}
    preprocessing = {'height': 48, 'color': 'RGB', 'padding': 'right-zero-normalized', 'resize': 'PIL-bilinear',
                     'width': {'mode': 'dynamic', 'min': 504, 'max': 1024, 'multiple': 8, 'rounding': 'ceil-multiple'},
                     'normalize': {'mean': [.5] * 3, 'std': [.5] * 3}}
    bindings = []
    for index, (raw_family, family) in enumerate(b.FAMILIES.items()):
        base = artifact / family
        model = opaque(base / 'model.onnx'); vocabulary = opaque(base / 'vocabulary.json')
        checkpoint = opaque(base / 'checkpoint.pth'); dictionary = opaque(base / 'dictionary.txt')
        source = {'checkpoint': checkpoint, 'dictionary': dictionary}
        decoder = {'type': 'ctc', 'repeat': 'ctc', 'blank_id': 0, 'output': 'logits'}
        vocabulary.update(blank_id=0, space_id=1)
        model_id = family + '-fixture-final'
        meta = {'schema_version': 'localocr.pretrained.candidate.v1', '_fixture': True, 'id': model_id,
                'family': raw_family, 'stage': 'finetuned', 'promote': False, 'enabled': False, 'model': model,
                'vocabulary': vocabulary, 'preprocessing': preprocessing, 'decoder': decoder, 'source': source,
                'private_text': 'NEVER_EXPORT_PRIVATE_FIELD'}
        candidate = write(base / 'candidate.json', meta)
        bindings.append({'id': model_id, 'family': raw_family, 'stage': 'finetuned', 'candidate_file': str(candidate),
            'candidate_sha256': SHA(candidate), 'model_sha256': model['sha256'], 'model_bytes': (base / 'model.onnx').stat().st_size,
            'vocabulary_sha256': vocabulary['sha256'], 'token_count': 4, 'preprocessing': preprocessing, 'decoder': decoder,
            'source': source, 'source_files': {'source.checkpoint': checkpoint, 'source.dictionary': dictionary}})
    base = artifact / 'baseline'
    manifest_models = {}
    baseline_models = {}
    for role in ('det', 'cls', 'rec'):
        record = opaque(base / (role + '.onnx'), b'opaque-baseline-' + role.encode())
        size = (base / (role + '.onnx')).stat().st_size
        manifest_models[role] = {**record, 'size': size}
        baseline_models[role] = {'sha256': record['sha256'], 'bytes': size}
    manifest = write(base / 'manifest.json', {'schema_version': 'localocr.bundle.v1', '_fixture': True,
        'version': 'fixture-baseline', 'runtime': {'rapidocr': '3.10.0', 'onnxruntime': '1.31.0'}, 'models': manifest_models,
        'compatibility': {'architecture': 'ppocr-det-cls-rec-v1', 'dictionary': 'onnx-embedded-v1',
                          'preprocessing': 'rapidocr-3.10.0-default-v1', 'postprocessing': 'localocr-symbol-reread-v1'}})
    baseline = {'id': 'production-baseline', 'version': 'fixture-baseline', 'manifest_file': str(manifest),
                'manifest_sha256': SHA(manifest), 'models': baseline_models}
    binding = {'schema_version': 'localocr.region-pipeline-binding.v1', 'dataset': frozen,
               'inputs': {**refs(inputs), 'dataset_sha256': frozen['dataset_sha256'], 'split_sha256': frozen['split_sha256']},
               'baseline': baseline, 'models': bindings, 'config': config, 'config_sha256': b.digest_object(config),
               'implementation': impl, 'promote': False, 'enabled': False}
    runtime_source = {name: opaque(artifact / 'runtime' / (name + '.py')) for name in ('recognizer', 'decoder')}
    algorithm = 'sha256(canonical JSON: sorted keys, UTF-8, ensure_ascii=false, compact separators)'
    facts = {'schema_version': 'localocr.baseline-rec-preprocessing.v1', 'rec_image_shape': [3, 48, 320], 'rec_batch_num': 6,
             'color': 'BGR', 'resize': 'cv2.INTER_LINEAR', 'padding': 'right-zero-normalized', 'layout': 'NCHW-float32',
             'return_word_box': False, 'rapidocr_version': '3.10.0', 'opencv_version': '4.13.0',
             'width': 'int(height * max(configured_width / height, maximum input aspect ratio in the recognition batch))',
             'content_width': 'min(batch_width, ceil(height * input_width / input_height))',
             'normalize': {'scale': 255, 'mean': [.5] * 3, 'std': [.5] * 3},
             'implementation': {key: value['sha256'] for key, value in runtime_source.items()}}
    baseline_runtime = {'schema_version': 'localocr.baseline-recognizer.v1', 'rapidocr_version': '3.10.0',
        'source_files': runtime_source, 'vocabulary': {'sha256': '1' * 64, 'token_count': 4, 'blank_id': 0,
        'blank_token': '<blank>', 'space_ids': [3], 'hash_algorithm': algorithm},
        'preprocessing': {'sha256': b.digest_object(facts), 'configuration': facts, 'hash_algorithm': algorithm}}
    paths = {}
    for split in ('dev', 'heldout'):
        data = example[split]
        loc = deepcopy(data['localization'])
        loc['candidates_intersecting_body'] = loc.pop('intersecting_body')
        loc['candidates_intersecting_graphic'] = loc.pop('intersecting_graphic')
        models = {}
        for item in data['models'][1:]:
            negative = {}
            for kind, prefix in (('blank', 'blank'), ('graphic-only', 'graphic_only'), ('graphic_regions', 'graphic')):
                current = item['negative'][kind]
                negative['graphic_regions' if prefix == 'graphic' else prefix + '_scenes'] = current['regions' if prefix == 'graphic' else 'scenes']
                for stage in ('baseline', 'final', 'new'):
                    negative[stage + '_graphic_nonempty' if prefix == 'graphic' else prefix + '_' + stage + '_nonempty'] = current[stage + '_nonempty']
            for total, suffix in (('negative_scenes', 'scenes'), ('baseline_nonempty_scenes', 'baseline_nonempty'),
                                  ('final_nonempty_scenes', 'final_nonempty'), ('new_nonempty_scenes', 'new_nonempty')):
                negative[total] = sum(item['negative'][key][suffix] for key in ('blank', 'graphic-only'))
            body = {**item['body'], 'body_literal_deleted': 0}
            face = item['face_regions']
            models[item['id']] = {'body': body, 'negative': negative, 'whole_scene': {'count': data['scenes'], 'exact': 0, 'characters': data['scenes'] * 20,
                'character_errors': data['scenes'], 'cer': .05},
                'isolated_kaomoji': {**face, 'cer': face['character_errors'] / face['characters']},
                'safe_complete_specialist_crops': {'count': 1}, 'private_prediction': 'NEVER_EXPORT_PRIVATE_FIELD',
                'quality_gates': {'app_acceptance': False, 'production_enabled': False,
                    'no_new_body_errors': not any(body[key] for key in ('baseline_correct_to_wrong', 'erroneous_rewrites', 'new_body_intersection_errors')),
                    'no_new_negative_outputs': not any(item['negative'][key]['new_nonempty'] for key in item['negative'])}}
        runtime = {'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1,
                   'actual_baseline_detection': True, 'actual_specialist_models': 2, 'no_python_children_or_socket_io': True,
                   'scene_timeout_seconds': 90, 'onnxruntime_version': '1.31.0', 'warmup_scenes': 3,
                   'warm': {'count': data['scenes']}, 'load_seconds': .5, 'first_scene_seconds': .2,
                   'total_warm_seconds': 10., 'process_wall_seconds': 15., 'baseline_runtime': baseline_runtime}
        folder = artifact / ('evaluation-' + split)
        outputs = {key: opaque(folder / name) for key, name in b.OUTPUT_NAMES.items() if key != 'resources'}
        resource = write(folder / 'resources.json', {'pipeline_binding': binding, 'resources': runtime, '_fixture': True})
        outputs['resources'] = refs(resource)
        face = data['models'][0]['face_regions']
        report = {'schema_version': b.RAW_SCHEMA, '_fixture': True, 'completed_utc': '2026-10-10T06:00:00+00:00' if split == 'dev' else '2026-10-10T09:00:00+00:00',
            'outputs': outputs, 'split': split, 'pipeline_binding': binding, 'actual_cpu_inference': True,
            'independent_heldout': split == 'heldout', 'heldout_usage': None, 'promote': False, 'enabled': False,
            'scope': b.RAW_SCOPE, 'ground_truth_inference_input': False, 'normalization': b.NORMALIZATION, 'regional_alignment': b.ALIGNMENT,
            'resources': runtime, 'summary': {'scenes': data['scenes'], 'scene_kinds': {key: data['scenes'] // 10 for key in
                ('same-line', 'middle-face', 'different-line', 'low-body', 'small-face', 'low-face', 'ordinary-brackets', 'body-graphic', 'graphic-only', 'blank')},
                'localization': loc, 'models': models, 'baseline': {'whole_scene': {'count': data['scenes'], 'exact': 0, 'characters': data['scenes'] * 20,
                'character_errors': data['scenes'], 'cer': .05},
                    'isolated_kaomoji': {**face, 'cer': face['character_errors'] / face['characters']}}}}
        paths[split] = write(folder / 'evaluation.json', report)
    paths['budget'] = write(artifact / 'shared-budget.json', {'schema_version': 'localocr.gpu-budget.v1', '_fixture': True,
                                                           'maximum_seconds': 7200, 'active': None, 'used_seconds': 5000.})
    dev = json.loads(paths['dev'].read_text(encoding='utf-8'))
    selected = {'schema_version': 'localocr.region-selection.v1', '_fixture': True, 'frozen_utc': '2026-10-10T07:00:00+00:00',
                'pipeline_binding': binding, 'baseline_runtime': baseline_runtime, 'dev_report': refs(paths['dev']),
                'budget_receipt': refs(paths['budget']), 'selected_id': bindings[0]['id'], 'selection_reason': 'PRIVATE_REASON_NOT_EXPORT',
                'promote': False, 'enabled': False, 'quality_gate_snapshot': {key: item['quality_gates'] for key, item in dev['summary']['models'].items()}}
    paths['selection'] = write(artifact / 'evaluation-selection.json', selected)
    seal = write(inputs.parent / 'heldout-seal.json', {'schema_version': 'localocr.region-scenes.seal.v1', '_fixture': True,
        'dataset_sha256': frozen['dataset_sha256'], 'heldout_sha256': frozen['split_sha256']['heldout'],
        'inference_results_seen': False, 'requires_frozen_pipeline_selection': True})
    usage_name = 'evaluation-heldout-first-use-' + frozen['split_sha256']['heldout'] + '.json'
    paths['usage'] = write(artifact / usage_name, {'schema_version': 'localocr.region-heldout-first-use.v1', '_fixture': True,
        'claimed_utc': '2026-10-10T08:00:00+00:00', 'pid': 42, 'selection': refs(paths['selection']), 'pipeline_binding': binding,
        'seal_sha256': SHA(seal), 'output': str(paths['heldout'].parent),
        'worker_output': str(paths['heldout'].parent / '.worker/predictions.jsonl'), 'worker_jobs': str(paths['heldout'].parent / '.worker/jobs.json'),
        'status': 'claimed-before-reading-heldout-images-or-truth; failure remains consumed'})
    paths['worker'] = write(paths['usage'].with_name(paths['usage'].stem + '-worker.json'), {
        'schema_version': 'localocr.region-heldout-worker-use.v1', '_fixture': True, 'usage_sha256': SHA(paths['usage']), 'pid': 43,
        'claimed_utc': '2026-10-10T08:01:00+00:00', 'status': 'consumed-before-reading-images; failed workers cannot replay'})
    held = json.loads(paths['heldout'].read_text(encoding='utf-8'));held['heldout_usage'] = refs(paths['usage']);write(paths['heldout'], held)
    return paths


def build(paths, heldout=False):
    return b.build(paths['dev'], paths['budget'], paths['heldout'] if heldout else None,
                   paths['selection'] if heldout else None, fixture=True)


def mutate(path, change):
    value = json.loads(Path(path).read_text(encoding='utf-8'));change(value);write(Path(path), value)


def test_dev_only_and_complete_transparently_project_actual_counts(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    summary, receipt = build(paths)
    assert summary['fixture'] is True and summary['heldout'] is None
    assert summary['dev']['localization']['candidate_count'] == 300
    assert summary['dev']['models'][1]['face_regions']['exact'] == 7
    p.validate(summary)
    summary, receipt = build(paths, True)
    p.validate(summary)
    assert summary['heldout']['scenes'] == 400
    assert receipt['opaque_integrity_checks'] is True and receipt['private_output_content_parsed'] is False
    text = json.dumps([summary, receipt])
    for token in ('PRIVATE', str(tmp_path), 'candidate_file', '"prediction":'):
        assert token not in text
    assert any(entry['role'] == 'predictions-integrity-only' for entry in receipt['source_closure'])

def edit_report(path, change):
    """Update the receipt consistently, so semantic negative controls reach validation."""
    value = json.loads(path.read_text(encoding='utf-8'))
    change(value)
    resource = Path(value['outputs']['resources']['file'])
    write(resource, {'pipeline_binding': value['pipeline_binding'], 'resources': value['resources'], '_fixture': True})
    value['outputs']['resources']['sha256'] = SHA(resource)
    write(path, value)


@pytest.mark.parametrize('change', [
    lambda r: r.update(actual_cpu_inference=False),
    lambda r: r.pop('completed_utc'),
    lambda r: r.update(completed_utc='2026-10-10T06:00:00'),
    lambda r: r.update(completed_utc='2099-01-01T00:00:00+00:00'),
    lambda r: r.update(normalization='strip spaces'),
    lambda r: r.update(regional_alignment='Visible-ink mask overlap only; boxes touching multiple truth regions are mixed and their transcript is not split using labels'),
    lambda r: r.update(ground_truth_inference_input=True),
    lambda r: r.update(enabled=True),
    lambda r: r['summary'].update(scenes=199),
    lambda r: r['summary']['scene_kinds'].update(blank=19),
    lambda r: r['summary']['models'].pop(next(iter(r['summary']['models']))),
    lambda r: r['pipeline_binding']['models'].pop(),
    lambda r: r['pipeline_binding']['models'][0].update(stage='official-pretrained'),
    lambda r: r['pipeline_binding']['models'][0].update(model_bytes=999),
    lambda r: r['pipeline_binding']['models'][0].update(model_sha256='0' * 64),
    lambda r: r['pipeline_binding']['models'][0].update(vocabulary_sha256='0' * 64),
    lambda r: r['pipeline_binding']['models'][0].update(token_count=1),
    lambda r: r['pipeline_binding']['models'][0]['preprocessing']['width'].update(min=505),
    lambda r: r['pipeline_binding']['inputs'].update(dataset_sha256='0' * 64),
    lambda r: r['pipeline_binding']['inputs'].update(file='heldout.json'),
    lambda r: r['pipeline_binding']['implementation'].pop(next(iter(r['pipeline_binding']['implementation']))),
    lambda r: r['pipeline_binding'].update(config_sha256='0' * 64),
    lambda r: r['resources'].update(providers=['CUDAExecutionProvider']),
    lambda r: r['resources'].update(intra_op_threads=True),
    lambda r: r['resources'].update(inter_op_threads=2),
    lambda r: r['resources'].update(actual_specialist_models=1),
    lambda r: r['resources'].update(no_python_children_or_socket_io=False),
    lambda r: r['resources'].update(warm={'count': 199}),
    lambda r: r['resources'].update(process_wall_seconds=5401),
    lambda r: r['resources']['baseline_runtime']['vocabulary'].update(blank_id=True),
    lambda r: r['resources']['baseline_runtime']['vocabulary'].update(space_ids=[0]),
    lambda r: r['resources']['baseline_runtime']['preprocessing'].update(sha256='0' * 64),
    lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['quality_gates'].update(app_acceptance=True),
    lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['quality_gates'].update(no_new_body_errors=True),
    lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['isolated_kaomoji'].update(count=1),
    lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['isolated_kaomoji'].update(cer=.123),
    lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['body'].update(body_literal_deleted=199),
    lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['negative'].update(negative_scenes=1),
    lambda r: r['summary']['localization'].update(routed_candidate_count=301),
])
def test_incomplete_or_drifted_cpu_report_rejected(tmp_path, monkeypatch, change):
    paths = sources(tmp_path, monkeypatch)
    edit_report(paths['dev'], change)
    with pytest.raises(ValueError):
        build(paths)


@pytest.mark.parametrize('change', [
    lambda c: c['candidate'].pop('maximum_candidates'),
    lambda c: c['candidate'].update(unknown=1),
    lambda c: c['candidate'].update(maximum_candidates=49),
    lambda c: c['candidate'].update(contrast_threshold=True),
    lambda c: c['candidate'].update(line_overlap=1.1),
    lambda c: c['candidate'].update(minimum_context_blank_band=1),
    lambda c: c['fusion'].update(require_extra_shape_support=False),
    lambda c: c['fusion'].update(maximum_output_characters=257),
    lambda c: c['fusion'].update(allow_pixel_only_insertions=1),
    lambda c: c['limits'].update(scene_timeout_seconds=91),
])
def test_self_consistent_unsupported_config_rejected(tmp_path, monkeypatch, change):
    paths = sources(tmp_path, monkeypatch)
    def apply(raw):
        change(raw['pipeline_binding']['config'])
        raw['pipeline_binding']['config_sha256'] = b.digest_object(raw['pipeline_binding']['config'])
    edit_report(paths['dev'], apply)
    with pytest.raises(ValueError):
        build(paths)


@pytest.mark.parametrize('key', ['predictions', 'alignment', 'resources'])
def test_bound_outputs_are_mandatory_and_byte_exact(tmp_path, monkeypatch, key):
    paths = sources(tmp_path, monkeypatch)
    raw = json.loads(paths['dev'].read_text(encoding='utf-8'))
    target = Path(raw['outputs'][key]['file'])
    target.write_bytes(target.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA'):
        build(paths)


@pytest.mark.parametrize('name', ['failure.json', 'measurement-failure.json'])
def test_failure_marker_cannot_be_completed_evidence(tmp_path, monkeypatch, name):
    paths = sources(tmp_path, monkeypatch)
    write(paths['dev'].parent / name, {'status': 'failed'})
    with pytest.raises(ValueError, match='Failed evaluation'):
        build(paths)


@pytest.mark.parametrize('maximum', [7200, 7200.0])
def test_closed_gpu_budget_accepts_equivalent_numeric_maximum(tmp_path, monkeypatch, maximum):
    paths = sources(tmp_path, monkeypatch)
    mutate(paths['budget'], lambda r: r.update(maximum_seconds=maximum))
    summary, receipt = build(paths)
    p.validate(summary)
    assert summary['fixture'] is True and receipt['fixture'] is True


@pytest.mark.parametrize('maximum', [True, False, float('nan'), float('inf'), -float('inf'),
                                     7199, 7201, '7200', None])
def test_closed_gpu_budget_rejects_invalid_maximum(tmp_path, monkeypatch, maximum):
    paths = sources(tmp_path, monkeypatch)
    mutate(paths['budget'], lambda r: r.update(maximum_seconds=maximum))
    with pytest.raises(ValueError, match='closed shared|Nonfinite JSON number'):
        build(paths)


@pytest.mark.parametrize('change', [lambda r: r.update(active={'owner': 'fixture'}),
                                   lambda r: r.update(used_seconds=7201),
                                   lambda r: r.update(used_seconds=True)])
def test_open_or_overrun_gpu_budget_rejected(tmp_path, monkeypatch, change):
    paths = sources(tmp_path, monkeypatch); mutate(paths['budget'], change)
    with pytest.raises(ValueError, match='closed shared'):
        build(paths)


@pytest.mark.parametrize('change', [
    lambda r: r.update(frozen_utc='2026-10-10T05:00:00+00:00'),
    lambda r: r.update(selected_id='not-measured'),
    lambda r: r['dev_report'].update(sha256='0' * 64),
    lambda r: r['budget_receipt'].update(sha256='0' * 64),
    lambda r: r.update(quality_gate_snapshot={}),
    lambda r: r.update(baseline_runtime={}),
])
def test_selection_must_freeze_actual_dev_closure(tmp_path, monkeypatch, change):
    paths = sources(tmp_path, monkeypatch); mutate(paths['selection'], change)
    with pytest.raises(ValueError):
        build(paths, True)


def rebind_usage(paths, change):
    mutate(paths['usage'], change)
    mutate(paths['worker'], lambda r: r.update(usage_sha256=SHA(paths['usage'])))
    mutate(paths['heldout'], lambda r: r['heldout_usage'].update(sha256=SHA(paths['usage'])))


@pytest.mark.parametrize('change', [
    lambda r: r.update(claimed_utc='2026-10-10T06:30:00+00:00'),
    lambda r: r.update(claimed_utc='2026-10-10T10:00:00+00:00'),
    lambda r: r.update(pid=0),
    lambda r: r['selection'].update(sha256='0' * 64),
    lambda r: r.update(seal_sha256='0' * 64),
    lambda r: r.update(worker_output=r['output'] + '/other.jsonl'),
])
def test_heldout_first_use_cannot_move_or_precede_freeze(tmp_path, monkeypatch, change):
    paths = sources(tmp_path, monkeypatch); rebind_usage(paths, change)
    with pytest.raises(ValueError):
        build(paths, True)


@pytest.mark.parametrize('change', [lambda r: r.update(usage_sha256='0' * 64),
                                   lambda r: r.update(status='replayed'),
                                   lambda r: r.update(claimed_utc='2026-10-10T10:00:00+00:00')])
def test_worker_consumption_receipt_mandatory(tmp_path, monkeypatch, change):
    paths = sources(tmp_path, monkeypatch); mutate(paths['worker'], change)
    with pytest.raises(ValueError, match='worker consumption'):
        build(paths, True)


def test_heldout_requires_first_use_and_same_full_pipeline(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='explicit frozen selection'):
        b.build(paths['dev'], paths['budget'], paths['heldout'], fixture=True)
    mutate(paths['heldout'], lambda r: r.update(heldout_usage=None))
    with pytest.raises(ValueError, match='first-use'):
        build(paths, True)
    paths = sources(tmp_path / 'other', monkeypatch)
    def change(raw):
        raw['pipeline_binding']['config']['candidate']['padding'] = 4
        raw['pipeline_binding']['config_sha256'] = b.digest_object(raw['pipeline_binding']['config'])
    edit_report(paths['heldout'], change)
    with pytest.raises(ValueError, match='Development/heldout'):
        build(paths, True)


def test_closed_counter_missing_zeroes_project_as_zero_only(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    def change(raw):
        loc = raw['summary']['localization']
        for key in loc:
            if key not in ('faces', 'body_regions', 'graphic_regions', 'negative_scenes'):
                loc[key] = 0
        loc['missed'] = loc['faces']
        for key in ('candidate_count', 'routed_candidate_count', 'routed_candidates_intersecting_body',
                    'routed_candidates_intersecting_graphic'):
            loc.pop(key)
    edit_report(paths['dev'], change)
    summary, _ = build(paths)
    assert summary['dev']['localization']['routed_candidate_count'] == 0
    mutate(paths['dev'], lambda r: r['summary']['models'][next(iter(r['summary']['models']))]['body'].pop('regions'))
    with pytest.raises(ValueError):
        build(paths)


def test_opaque_inputs_never_parse_prediction_vocabulary_or_model_contents(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    parser = p.strict_json
    def inspect(data):
        assert b'PRIVATE_FIXTURE_CONTENT_NOT_JSON' not in data
        return parser(data)
    monkeypatch.setattr(p, 'strict_json', inspect)
    summary, receipt = build(paths, True)
    assert summary['dev']['models'][0]['vocabulary_sha256'] == '1' * 64
    assert summary['dev']['models'][0]['vocabulary_sha256'] != summary['dev']['models'][0]['model_sha256']
    assert receipt['scene_truth_images_or_masks_opened'] is False
    assert receipt['inference_job_index_content_parsed'] is False
    assert receipt['opaque_inference_job_index_hashed'] is True
    assert receipt['model_execution'] is False
    assert {entry['mode'] for entry in receipt['source_closure']} == {'receipt-json', 'opaque-streaming-sha-size'}


def test_fixture_cannot_be_promoted_and_output_cannot_reach_docs(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='fixtures cannot'):
        b.build(paths['dev'], paths['budget'], fixture=False)
    target = b.ROOT / 'docs/fixture.json'
    with pytest.raises(ValueError, match='Fixture output must'):
        b.main(['--dev-report', str(paths['dev']), '--budget-receipt', str(paths['budget']),
                '--fixture', '--output', str(target)])
    assert not target.exists()


def test_cli_writes_fresh_whitelisted_summary_and_receipt(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    target = b.ARTIFACTS / 'documentation-check/fixture-summary.json'
    args = ['--dev-report', str(paths['dev']), '--budget-receipt', str(paths['budget']), '--fixture', '--output', str(target)]
    assert b.main(args) == 0
    summary = p.strict_json(target.read_bytes());p.validate(summary)
    receipt = p.strict_json(target.with_name(target.stem + '.build-receipt.json').read_bytes())
    assert receipt['summary_sha256'] == SHA(target)
    assert receipt['source_closure_sha256'] == b.digest_object(receipt['source_closure'])
    for entry in receipt['source_closure']:
        assert set(entry) == {'role', 'sha256', 'bytes', 'mode'}
    before = target.read_bytes()
    repeat = target.with_name('fixture-repeat.json')
    assert b.main(args[:-1] + [str(repeat)]) == 0
    assert repeat.read_bytes() == before
    assert repeat.with_name(repeat.stem + '.build-receipt.json').read_bytes() == target.with_name(target.stem + '.build-receipt.json').read_bytes()
    with pytest.raises(ValueError, match='Fresh summary'):
        b.main(args)
    assert target.read_bytes() == before


def test_changed_source_before_output_rejected(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    original = b.Evidence.stable
    def mutate_then_verify(evidence):
        target = next(item['path'] for item in evidence.entries if item['role'] == 'candidate-model')
        target.write_bytes(target.read_bytes() + b'changed')
        original(evidence)
    monkeypatch.setattr(b.Evidence, 'stable', mutate_then_verify)
    output = b.ARTIFACTS / 'fresh-summary.json'
    with pytest.raises(ValueError, match='changed before anonymous output'):
        b.main(['--dev-report', str(paths['dev']), '--budget-receipt', str(paths['budget']),
                '--output', str(output), '--fixture'])
    assert not output.exists()


@pytest.mark.parametrize('path', ['../outside.json', '//server/share/evaluation.json'])
def test_traversal_or_network_paths_rejected(path):
    with pytest.raises(ValueError, match='local non-traversing'):
        b.safe_path(path, fixture=True)


def test_actual_artifact_boundary_enforced(tmp_path, monkeypatch):
    sources(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='controlled LocalOCR artifacts'):
        b.safe_path(b.ROOT / 'ordinary-file.json')
    with pytest.raises(ValueError, match='Alternate streams'):
        b.safe_path(str(b.ARTIFACTS / 'receipt.json') + ':alternate', fixture=True)


def test_reparse_ancestor_rejected_before_reading(tmp_path, monkeypatch):
    sources(tmp_path, monkeypatch)
    path = b.ARTIFACTS / 'unopened.json';opaque(path)
    original = Path.lstat
    def marked(self):
        if self == path.parent:
            class Info:
                st_mode = 0o040755
                st_file_attributes = 0x400
            return Info()
        return original(self)
    monkeypatch.setattr(Path, 'lstat', marked)
    with pytest.raises(ValueError, match='Reparse points'):
        b.Evidence(True).capture(path, 'must-not-open')

@pytest.mark.parametrize('suffix', ['.yml', '.png'])
def test_model_source_configuration_allowed_but_scene_images_never_consumed(tmp_path, monkeypatch, suffix):
    paths = sources(tmp_path, monkeypatch)
    raw = json.loads(paths['dev'].read_text(encoding='utf-8'))
    item = raw['pipeline_binding']['models'][0]
    meta_path = Path(item['candidate_file']);meta = json.loads(meta_path.read_text(encoding='utf-8'))
    record = opaque(meta_path.parent / ('source-config' + suffix))
    meta['source']['config'] = record;write(meta_path, meta)
    item['source'] = meta['source'];item['candidate_sha256'] = SHA(meta_path)
    item['source_files']['source.config'] = record
    evidence = b.Evidence(True)
    if suffix == '.yml':
        assert b.candidate(item, evidence)['stage'] == 'final-fusion'
    else:
        with pytest.raises(ValueError, match='scene labels/images/masks'):
            b.candidate(item, evidence)
        assert all(entry['path'] != Path(record['file']) for entry in evidence.entries)


def test_source_freeze_receipt_cannot_drift(tmp_path, monkeypatch):
    paths = sources(tmp_path, monkeypatch)
    raw = json.loads(paths['dev'].read_text(encoding='utf-8'))
    freeze = Path(raw['pipeline_binding']['inputs']['file']).parent / 'freeze.json'
    mutate(freeze, lambda r: r['split_sha256'].update(heldout='0' * 64))
    with pytest.raises(ValueError, match='Frozen scene receipt differs'):
        build(paths)


def test_private_output_hash_check_uses_bounded_streams(tmp_path, monkeypatch):
    path = tmp_path / 'opaque.bin';path.write_bytes(b'a' * (2 * 1024**2 + 3))
    sizes = []
    original = Path.open
    class Stream:
        def __init__(self, raw): self.raw = raw
        def __enter__(self): self.raw.__enter__();return self
        def __exit__(self, *args): return self.raw.__exit__(*args)
        def read(self, size): sizes.append(size);return self.raw.read(size)
    def opened(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        return Stream(raw) if self == path and args == ('rb',) else raw
    expected = SHA(path)
    monkeypatch.setattr(Path, 'open', opened)
    value, digest = b.Evidence(True).capture(path, 'predictions-integrity-only', expected, opaque=True)
    assert value is None and digest == expected and sizes and set(sizes) == {1024**2}
    with pytest.raises(ValueError, match='byte size exceeds'):
        b.Evidence(True).capture(path, 'bounded', opaque=True, maximum=2 * 1024**2)

@pytest.mark.parametrize('field', ['actual_specialist_models', 'scene_timeout_seconds', 'warmup_scenes'])
def test_integer_cpu_receipt_facts_cannot_be_floats(tmp_path, monkeypatch, field):
    paths = sources(tmp_path, monkeypatch)
    edit_report(paths['dev'], lambda r: r['resources'].update({field: float(r['resources'][field])}))
    with pytest.raises(ValueError, match='wrong backend/threads'):
        build(paths)
