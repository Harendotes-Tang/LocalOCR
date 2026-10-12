"""Anonymous-summary provenance and arithmetic tests; fixtures are not OCR evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.Documentation import build_pretrained_summary as b

COUNTS = {'plain': 1100, 'face': 400, 'punctuation': 400, 'emoji': 100, 'blank': 50, 'graphic': 50}
CHARACTERS = {'plain': 28022, 'face': 7824, 'punctuation': 7576, 'emoji': 1198, 'blank': 0, 'graphic': 0}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def frozen():
    return {'schema_version': 'localocr.pretrained.freeze.v1', 'license': 'CC0-1.0', 'dataset_sha256': 'a' * 64,
            'split_sha256': {'train': 'b' * 64, 'dev': 'c' * 64, 'heldout': 'd' * 64},
            'counts': {split: {'count': size} for split, size in (('train', 31000), ('dev', 2100), ('heldout', 4200))}}


def categories(scale=1, candidate=0):
    result = {}
    for kind, n in COUNTS.items():
        count = n * scale
        negative = kind in ('blank', 'graphic')
        exact = (10 + candidate) * scale if negative else (20 + candidate * 5) * scale
        errors = (count - exact) if negative else max(count - exact, CHARACTERS[kind] * scale // 4)
        chars = CHARACTERS[kind] * scale
        result[kind] = {'count': count, 'exact': exact, 'characters': chars, 'character_errors': errors,
                        'false_nonempty_outputs': count - exact if negative else 0, 'cer': errors / chars if chars else None}
    return result


def cpu_fixture(split='dev'):
    """Five-source receipt shape only; no network, model or image execution."""
    scale = 1 if split == 'dev' else 2
    preprocessing = lambda family: {'height': 48, 'color': 'BGR' if family == b.FAMILIES[0] else 'RGB',
                                    'width': {'min': 512 if family == b.FAMILIES[0] else 504, 'max': 1024}}
    baseline = {'id': 'production-baseline', 'version': 'fixture', 'models': {'rec': {'sha256': 'e' * 64}},
                'manifest_file': 'D:/never-export-private-path/manifest.json'}
    bindings, models = [], {}
    for index, (family, stage) in enumerate(b.SERIES):
        model_id = family + '-' + stage if index else 'production-baseline'
        values = categories(scale, index)
        models[model_id] = {'summary': {'total_images': 2100 * scale, 'ambiguous_excluded': 0, 'by_kind': values,
                            'overall': {key: sum(item[key] for item in values.values()) for key in
                                        ('count', 'exact', 'characters', 'character_errors', 'false_nonempty_outputs')}},
                            'resources': {'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1},
                            'prediction': 'NEVER EXPORT THIS PRIVATE FIXTURE STRING'}
        if index:
            raw_family = 'official-ppocrv6-tiny' if family == b.FAMILIES[0] else 'openocr-repsvtr-mobile'
            bindings.append({'id': model_id, 'family': raw_family, 'stage': stage,
                             'candidate_sha256': str(index) * 64, 'model_sha256': str(index + 1) * 64,
                             'vocabulary_sha256': 'f' * 64, 'preprocessing': preprocessing(family),
                             'candidate_file': 'D:/never-export-private-path/candidate.json'})
            models[model_id]['stage'] = stage
            models[model_id]['transitions_from_production'] = {'by_kind': {kind: {'count': COUNTS[kind] * scale,
                'baseline_correct_to_wrong': 2 * scale, 'erroneous_rewrites': 60 * scale} for kind in ('plain', 'punctuation')}}
    return {'schema_version': 'localocr.pretrained.evaluation.v1', '_fixture': True, 'split': split,
            'dataset_binding': frozen(), 'baseline_binding': baseline, 'candidate_bindings': bindings, 'models': models,
            'first_use': None if split == 'dev' else {'file': 'D:/never-open/heldout-first-use.json', 'sha256': '8' * 64, 'selection_sha256': '9' * 64},
            'promote': False, 'enabled': False, 'independent_heldout': split == 'heldout',
            'normalization': 'none; exact Unicode, case, whitespace and graphemes'}


def native(family):
    items = {}
    for kind, item in categories().items():
        if family == 'repsvtr-mobile':
            items[kind] = {'samples': item['count'], 'exact_correct': item['exact'],
                          'reference_codepoints': item['characters'], 'edit_distance': item['character_errors'],
                          'negative_false_outputs': item['false_nonempty_outputs']}
        else:
            items[kind] = {'rows': item['count'], 'exact_correct': item['exact'],
                          'reference_codepoints': item['characters'], 'edit_distance': item['character_errors'],
                          'negative_nonempty': item['false_nonempty_outputs']}
    stats = {'exact_accuracy': sum(item['exact'] for item in categories().values()) / 2100, 'by_kind': items}
    if family == 'ppocrv6-tiny':
        stats.update({key: sum(item[key] for item in items.values()) for key in
                      ('rows', 'exact_correct', 'reference_codepoints', 'edit_distance')})
        stats['ignore_space'] = False
    return stats


def inputs(root):
    dev = write(root / 'dev/evaluation.json', cpu_fixture())
    heldout = write(root / 'heldout/evaluation.json', cpu_fixture('heldout'))
    runs = []
    for family in b.FAMILIES:
        prep = ({'family': 'openocr-repsvtr-mobile', 'dataset_binding': frozen()} if family == 'repsvtr-mobile'
                else {'schema_version': 'localocr.paddle.initialization.v1', 'dataset': frozen()})
        prepfile = write(root / family / ('prepare.json' if family == 'repsvtr-mobile' else 'initialization.json'), prep)
        record = {'step': 500, 'dev_metrics' if family == 'repsvtr-mobile' else 'dev': native(family),
                  'prediction': 'NEVER EXPORT THIS PRIVATE FIXTURE STRING', 'image': 'D:/never-open/image.png'}
        curve = write(root / family / ('training-progress.json' if family == 'repsvtr-mobile' else 'history.json'), [record])
        runs.append((family, family + '-fixture-001', prepfile, curve))
    return dev, heldout, runs


def mutate(path, change):
    value = json.loads(path.read_text(encoding='utf-8')); change(value);write(path, value)


def test_dev_only_extracts_raw_counts_and_source_closures(tmp_path):
    dev, _, runs = inputs(tmp_path)
    summary, receipt = b.build(dev, runs, fixture=True)
    assert summary['heldout'] is None
    assert len(summary['training']) == 2
    assert summary['training'][1]['points'][0] == {'step': 500, 'count': 2100, 'exact': 100,
        'characters': 44620, 'character_errors': 11234}
    assert summary['training'][0]['evaluation_backend'] == 'CUDA-native'
    for item, closure in zip(summary['training'], receipt['run_closures']):
        digest = hashlib.sha256(json.dumps(closure, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
        assert item['source_sha256'] == digest
    assert receipt['dev_report_sha256'] == hashlib.sha256(dev.read_bytes()).hexdigest()


def test_heldout_uses_five_identical_model_bindings_without_opening_samples(tmp_path, monkeypatch):
    dev, held, runs = inputs(tmp_path)
    allowed = {Path(path).resolve() for path in (dev, held, *(r[2] for r in runs), *(r[3] for r in runs), Path(b.__file__))}
    original = Path.read_bytes
    def read_bytes(path):
        assert path.resolve() in allowed, 'Attempted dataset/image/model/private read'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    summary, receipt = b.build(dev, runs, held, fixture=True)
    assert len(summary['heldout']['models']) == 5
    assert summary['heldout']['evaluation_backend'] == 'ONNX Runtime CPU'
    encoded = json.dumps([summary, receipt], ensure_ascii=False)
    assert 'NEVER EXPORT' not in encoded and 'never-export' not in encoded and 'never-open' not in encoded
    assert 'prediction' not in json.dumps(summary) and 'image' not in json.dumps(summary)
    assert 'paths_exported' in encoded  # Explicit false scope flag, not input paths.


@pytest.mark.parametrize('change,expected', [
    (lambda d: d['candidate_bindings'].pop(), 'pairs'),
    (lambda d: d['candidate_bindings'][0].update(stage='finetuned'), 'family/stage'),
    (lambda d: d['candidate_bindings'][1]['preprocessing']['width'].update(min=640), 'preprocessing'),
    (lambda d: d['models']['production-baseline']['resources'].update(providers=['CUDAExecutionProvider']), 'CPU evaluator'),
    (lambda d: d['models']['production-baseline']['summary']['overall'].update(exact=101), 'category totals'),
    (lambda d: d['models']['production-baseline']['summary']['by_kind']['plain'].update(cer=.999), 'CER differs'),
    (lambda d: d.update(normalization='ignore spaces and case'), 'raw-Unicode'),
])
def test_invalid_cpu_evidence_rejected(tmp_path, change, expected):
    dev, _, runs = inputs(tmp_path);mutate(dev, change)
    with pytest.raises(ValueError, match=expected):b.build(dev, runs, fixture=True)


def test_training_dataset_mismatch_rejected(tmp_path):
    dev, _, runs = inputs(tmp_path)
    mutate(runs[0][2], lambda d:d['dataset']['split_sha256'].update(dev='f' * 64))
    with pytest.raises(ValueError, match='dataset binding differs'):b.build(dev, runs, fixture=True)


def test_incomplete_native_dev_point_rejected(tmp_path):
    dev, _, runs = inputs(tmp_path)
    mutate(runs[1][3], lambda d:d[0]['dev_metrics']['by_kind']['plain'].update(samples=1099))
    with pytest.raises(ValueError, match='denominator differs'):b.build(dev, runs, fixture=True)


def test_no_completed_dev_point_rejected(tmp_path):
    dev, _, runs = inputs(tmp_path);write(runs[1][3], [{'step':500,'loss':1}])
    with pytest.raises(ValueError, match='No completed full'):b.build(dev, runs, fixture=True)


def test_wrong_metric_shape_cannot_be_relabelled_as_another_family(tmp_path):
    dev, _, runs = inputs(tmp_path)
    mutate(runs[1][3], lambda d:d[0].update(dev=d[0].pop('dev_metrics')))
    with pytest.raises(ValueError, match='another family'):b.build(dev, runs, fixture=True)


def test_source_changed_before_output_is_rejected(tmp_path, monkeypatch):
    dev, _, runs = inputs(tmp_path)
    original = b.curve
    def change_after_capture(*args):
        result = original(*args);dev.write_bytes(dev.read_bytes()+b' ');return result
    monkeypatch.setattr(b, 'curve', change_after_capture)
    with pytest.raises(ValueError, match='changed during'):b.build(dev, runs, fixture=True)


def test_heldout_and_development_weights_must_match(tmp_path):
    dev, held, runs = inputs(tmp_path)
    mutate(held, lambda d:d['candidate_bindings'][0].update(model_sha256='f' * 64))
    with pytest.raises(ValueError, match='model bindings differ'):b.build(dev, runs, held, fixture=True)


def test_same_curve_cannot_be_counted_twice(tmp_path):
    dev, _, runs = inputs(tmp_path);duplicate=list(runs[0]);duplicate[1]='duplicate-fixture-002'
    with pytest.raises(ValueError, match='impersonate another iteration'):b.build(dev, runs+[duplicate], fixture=True)


def test_dataset_split_filename_is_rejected_before_read(tmp_path, monkeypatch):
    path=write(tmp_path/'heldout.json', {'fixture':True})
    monkeypatch.setattr(Path,'read_bytes',lambda path:pytest.fail('Dataset split was opened'))
    with pytest.raises(ValueError,match='receipt files'):b.Snapshot(path,'run-curve',fixture=True)


def test_cli_fixture_stays_in_artifacts_and_refuses_overwrite(tmp_path, monkeypatch):
    dev, _, runs=inputs(tmp_path)
    monkeypatch.setattr(b,'ARTIFACTS',tmp_path)
    monkeypatch.setattr(b,'ROOT',tmp_path.parent)
    output=tmp_path/'output/summary.json'
    arguments=['--dev-report',str(dev),'--output',str(output),'--fixture']
    for run in runs:arguments += ['--run',*(str(v) for v in run)]
    assert b.main(arguments)==0
    assert output.with_name('summary.build-receipt.json').is_file()
    with pytest.raises(ValueError,match='must be fresh'):b.main(arguments)
    arguments[arguments.index('--output')+1]=str(tmp_path.parent/'outside-fixture-summary.json')
    with pytest.raises(ValueError,match='Fixture summaries'):b.main(arguments)


def test_nonfinite_or_duplicate_fields_rejected(tmp_path):
    path=tmp_path/'evaluation.json';path.write_text('{"same":1,"same":2}',encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate JSON'):b.Snapshot(path,'dev-evaluation',fixture=True)
    path.write_text('{"fixture":NaN}',encoding='utf-8')
    with pytest.raises(ValueError,match='Nonfinite JSON'):b.Snapshot(path,'dev-evaluation',fixture=True)


def failed_openocr_run(runs):
    family, run_id, preparation, _ = runs[1]
    root=preparation.parent
    run_id='repsvtr-mobile-fixture-failed-run1'
    metrics=native(family)
    failure={'schema_version':'localocr.openocr.failure.v1','steps':24050,
             'original_failure':{'type':'PermissionError','message':'[WinError 5] D:/never-export-private-path/receipt.tmp'},
             'history':[{'step':18000,'dev_metrics':metrics},{'step':24050,'dev_metrics':metrics}]}
    curve=write(root/'last-failure.json',failure)
    selection=write(root/'best-selection.json',{'step':18000,'dev_metrics':metrics,'checkpoint_sha256':'8'*64,
                                             'dev_split_sha256':frozen()['split_sha256']['dev']})
    manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (preparation,curve,selection)]
    write(root/'run-freeze.json',{'schema_version':'localocr.openocr.run-freeze.v1','run_id':run_id,'completed':False,
          'actual_steps':24050,'selected_step':18000,'process_death_verified':True,'heldout_read':False,
          'original_failure':failure['original_failure'],'files':manifest})
    return runs[0],(family,run_id,preparation,curve)


def test_frozen_failure_keeps_real_steps_selection_and_anonymous_code(tmp_path):
    dev,_,runs=inputs(tmp_path);runs=failed_openocr_run(runs)
    summary,receipt=b.build(dev,runs,fixture=True)
    closure=receipt['run_closures'][1]
    assert closure['completed'] is False and closure['status']=='failed'
    assert closure['actual_steps']==24050 and closure['selected_step']==18000
    assert closure['failure_code']=='WinError5' and closure['failure_type']=='PermissionError'
    assert [point['step'] for point in summary['training'][1]['points']]==[18000,24050]
    assert 'never-export' not in json.dumps([summary,receipt])
    assert {'run-freeze','run-selection'}.issubset({source['role'] for source in receipt['sources']})


def test_failure_without_consistent_frozen_source_manifest_is_rejected(tmp_path):
    dev,_,runs=inputs(tmp_path);runs=failed_openocr_run(runs)
    mutate(runs[1][3].parent/'run-freeze.json',lambda d:d['files'][1].update(sha256='f'*64))
    with pytest.raises(ValueError,match='frozen source SHA differs'):b.build(dev,runs,fixture=True)


def test_failed_selection_requires_a_real_complete_development_point(tmp_path):
    dev,_,runs=inputs(tmp_path);runs=failed_openocr_run(runs)
    path=runs[1][3].parent/'best-selection.json'
    mutate(path,lambda d:d.update(step=17999))
    newsha=hashlib.sha256(path.read_bytes()).hexdigest()
    def update(d):
        d['selected_step']=17999
        for item in d['files']:
            if item['file']=='best-selection.json':item['sha256']=newsha
    mutate(path.parent/'run-freeze.json',update)
    with pytest.raises(ValueError,match='no matching completed development point'):b.build(dev,runs,fixture=True)


def test_completed_paddle_training_binds_preparation_and_effective_update(tmp_path):
    dev,_,runs=inputs(tmp_path)
    family,run_id,preparation,history=runs[0]
    value={'schema_version':'localocr.paddle.training.v1','dataset':frozen(),'dev_rows':2100,'train_rows':31000,
           'selected_weights_changed':True,'actual_weights_changed':True,
           'start_checkpoint':{'sha256':'1'*64},'last_checkpoint':{'sha256':'2'*64},
           'best_checkpoint':{'sha256':'3'*64},'selected_training_exposure':{'step':500},
           'best_dev':native('ppocrv6-tiny'),'preparation':{'file':'D:/never-open/source.json','sha256':hashlib.sha256(preparation.read_bytes()).hexdigest()},
           'steps':550,'history':json.loads(history.read_text(encoding='utf-8'))}
    file=write(history.parent/'training.json',value)
    runs[0]=(family,run_id,preparation,file)
    summary,receipt=b.build(dev,runs,fixture=True)
    assert receipt['run_closures'][0]['completed'] is True
    assert receipt['run_closures'][0]['actual_steps']==550
    mutate(file,lambda d:d.update(selected_weights_changed=False))
    with pytest.raises(ValueError,match='no-newbest'):b.build(dev,runs,fixture=True)


def test_labelled_fixture_cannot_be_published_as_actual(tmp_path,monkeypatch):
    dev,_,runs=inputs(tmp_path)
    monkeypatch.setattr(b,'ARTIFACTS',tmp_path)
    with pytest.raises(ValueError,match='fixture cannot be published as actual'):
        b.build(dev,runs,fixture=False)

def completed_paddle_run(runs, retained=False):
    family, run_id, preparation, history = runs[0]
    point = native(family)
    value = {'schema_version': 'localocr.paddle.training.v1', 'dataset': frozen(),
             'dev_rows': 2100, 'train_rows': 31000, 'actual_weights_changed': True,
             'selected_weights_changed': not retained, 'warm_start': retained,
             'selection_outcome': 'parent-retained-no-development-improvement' if retained else 'new-effective-best',
             'preparation': {'sha256': hashlib.sha256(preparation.read_bytes()).hexdigest()},
             'steps': 550, 'start_checkpoint': {'sha256': '1' * 64}, 'last_checkpoint': {'sha256': '2' * 64},
             'best_checkpoint': None if retained else {'sha256': '3' * 64},
             'selected_training_exposure': None if retained else {'step': 500},
             'initial_dev': point, 'best_dev': point, 'history': json.loads(history.read_text(encoding='utf-8'))}
    file = write(history.parent / 'training.json', value)
    return (family, run_id, preparation, file)


def completed_openocr_run(runs):
    family, run_id, preparation, history = runs[1]
    point = native(family)
    write(history.parent / 'best-selection.json', {'step': 500, 'dev_metrics': point, 'checkpoint_sha256': '5' * 64,
                                                  'dev_split_sha256': frozen()['split_sha256']['dev']})
    value = {'steps': 550, 'nonzero_gradient': True, 'save_reload_strict': True, 'changed_tensors': ['head.weight'],
             'initial_checkpoint_sha256': '6' * 64, 'checkpoint_sha256': '7' * 64,
             'preparation_sha256': hashlib.sha256(preparation.read_bytes()).hexdigest(),
             'best_dev_exact': point['exact_accuracy'], 'history': json.loads(history.read_text(encoding='utf-8'))}
    file = write(history.parent / 'training.json', value)
    return family, run_id, preparation, file


def test_completed_parent_retained_is_an_effective_curve_with_no_new_checkpoint(tmp_path):
    dev, _, runs = inputs(tmp_path)
    runs[0] = completed_paddle_run(runs, retained=True)
    summary, receipt = b.build(dev, runs, fixture=True)
    state = summary['training'][0]['run_state']
    assert state == {'status': 'completed', 'actual_steps': 550,
                     'checkpoint_selection': {'outcome': 'parent-retained', 'step': None, 'sha256': '1' * 64}}
    assert summary['training'][0]['points'][0]['step'] == 500
    closure = receipt['run_closures'][0]
    assert closure['completed'] is True and closure['last_checkpoint_sha256'] == '2' * 64
    assert 'final family selection is separate' in closure['selection_scope']


@pytest.mark.parametrize('change', [
    lambda r: r.update(actual_weights_changed=False),
    lambda r: r.update(warm_start=False),
    lambda r: r.update(selected_training_exposure={'step': 0}),
    lambda r: r.update(best_checkpoint={'sha256': '3' * 64}),
    lambda r: r.update(selection_outcome='new-effective-best'),
    lambda r: r['last_checkpoint'].update(sha256='1' * 64),
    lambda r: r['initial_dev']['by_kind']['plain'].update(exact_correct=19),
])
def test_retained_parent_requires_effective_updates_and_consistent_outcome(tmp_path, change):
    dev, _, runs = inputs(tmp_path);runs[0] = completed_paddle_run(runs, retained=True)
    mutate(runs[0][3], change)
    with pytest.raises(ValueError):b.build(dev, runs, fixture=True)


def test_completed_openocr_binds_run_best_separately_from_last_checkpoint(tmp_path):
    dev, _, runs = inputs(tmp_path);runs[1] = completed_openocr_run(runs)
    summary, receipt = b.build(dev, runs, fixture=True)
    state = summary['training'][1]['run_state']
    assert state['status'] == 'completed' and state['actual_steps'] == 550
    assert state['checkpoint_selection'] == {'outcome': 'run-best', 'step': 500, 'sha256': '5' * 64}
    assert receipt['run_closures'][1]['last_checkpoint_sha256'] == '7' * 64
    assert {'run-selection'}.issubset({source['role'] for source in receipt['sources']})


@pytest.mark.parametrize('target,change', [
    ('training.json', lambda r: r.update(preparation_sha256='0' * 64)),
    ('training.json', lambda r: r.update(best_dev_exact=.99)),
    ('best-selection.json', lambda r: r.update(step=499)),
    ('best-selection.json', lambda r: r.update(dev_split_sha256='0' * 64)),
])
def test_completed_openocr_selection_cannot_drift(tmp_path, target, change):
    dev, _, runs = inputs(tmp_path);runs[1] = completed_openocr_run(runs)
    mutate(runs[1][3].parent / target, change)
    with pytest.raises(ValueError):b.build(dev, runs, fixture=True)


def test_initialization_point_cannot_be_claimed_as_new_training_iteration(tmp_path):
    dev, _, runs = inputs(tmp_path)
    mutate(runs[0][3], lambda history: history[0].update(step=0))
    with pytest.raises(ValueError, match='initialization-only'):b.build(dev, runs, fixture=True)

def frozen_completed_openocr_run(runs):
    chosen = completed_openocr_run(runs)
    family, run_id, preparation, history = chosen
    selected = history.parent / 'best-selection.json'
    files = [{'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()} for file in (preparation, history, selected)]
    files += [{'file': name, 'sha256': digest} for name, digest in
              [('initial.pth', '6' * 64), ('last.pth', '7' * 64), ('best.pth', '5' * 64)]]
    write(history.parent / 'completed-run-freeze.json', {'schema_version': 'localocr.openocr.run-freeze.v1',
        'run_id': run_id, 'completed': True, 'exit_code': 0, 'heldout_read': False, 'actual_steps': 550,
        'selected_step': 500, 'gpu_allocated_before_lease_release': 0, 'files': files})
    return chosen


def test_completed_openocr_freeze_binds_identity_receipts_and_three_distinct_checkpoints(tmp_path):
    dev, _, runs = inputs(tmp_path);runs[1] = frozen_completed_openocr_run(runs)
    summary, receipt = b.build(dev, runs, fixture=True)
    closure = receipt['run_closures'][1]
    assert closure['run_freeze_sha256'] == hashlib.sha256((runs[1][3].parent / 'completed-run-freeze.json').read_bytes()).hexdigest()
    assert summary['training'][1]['run_state']['checkpoint_selection']['sha256'] == '5' * 64
    assert closure['last_checkpoint_sha256'] == '7' * 64


@pytest.mark.parametrize('change', [
    lambda r: r.update(run_id='renamed-source-identity'),
    lambda r: r.update(completed=False),
    lambda r: r.update(actual_steps=500),
    lambda r: r.update(selected_step=0),
    lambda r: r.update(heldout_read=True),
    lambda r: r['files'][1].update(sha256='0' * 64),
    lambda r: r['files'][-1].update(sha256='0' * 64),
])
def test_completed_openocr_freeze_cannot_be_relabelled_or_drifted(tmp_path, change):
    dev, _, runs = inputs(tmp_path);runs[1] = frozen_completed_openocr_run(runs)
    mutate(runs[1][3].parent / 'completed-run-freeze.json', change)
    with pytest.raises(ValueError):b.build(dev, runs, fixture=True)
