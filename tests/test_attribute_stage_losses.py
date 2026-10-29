"""Stage attribution over synthetic recorded structures; fixture counts are not OCR evaluation evidence."""
import numpy as np
import pytest

from tools.Documentation import attribute_stage_losses as a


def mask(box, shape=(20, 40)):
    value = np.zeros(shape, dtype=bool)
    left, top, right, bottom = box
    value[top:bottom, left:right] = True
    return value


def candidate(identity, box):
    return {'candidate_id': identity, 'bbox': list(box)}


def truth(text='(^_^)', box=(10, 5, 20, 15)):
    return {'id': 'face-1', 'kind': 'kaomoji', 'text': text, 'bbox': list(box), 'pixel_count': 100}


def view(text):
    return [{'transform': 'raw', 'text': text}, {'transform': 'edge-pad-4', 'text': text}]


@pytest.fixture
def unions():
    return mask((0, 5, 8, 15)), mask((30, 5, 38, 15))


def test_first_failing_stage_wins(unions):
    face, ink = truth(), mask((10, 5, 20, 15))
    assert a.categorize(face, ink, {}, {}, *unions)['category'] == 'nohit'
    partial = {'c': candidate('c', (12, 5, 20, 15))}
    result = a.categorize(face, ink, partial, {}, *unions)
    assert (result['category'], result['coverage_bin'] if 'coverage_bin' in result else a.coverage_bin(result['best_coverage'])) == ('partial-only', '[0.8,0.95)')
    unsafe = {'c': candidate('c', (5, 5, 20, 15))}
    result = a.categorize(face, ink, unsafe, {}, *unions)
    assert (result['category'], result['unsafe_kind']) == ('complete-but-body/graphicunsafe', 'body')
    graphic = {'c': candidate('c', (10, 5, 35, 15))}
    assert a.categorize(face, ink, graphic, {}, *unions)['unsafe_kind'] == 'graphic'
    both = {'c': candidate('c', (5, 5, 20, 15)), 'd': candidate('d', (10, 5, 35, 15))}
    assert a.categorize(face, ink, both, {}, *unions)['unsafe_kind'] == 'body+graphic'
    safe = {'c': candidate('c', (9, 4, 21, 16))}
    assert a.categorize(face, ink, safe, {}, *unions)['category'] == 'safe-unrouted'
    wrong = a.categorize(face, ink, safe, {'c': view('(^o^)')}, *unions)
    assert (wrong['category'], wrong['raw_correct_ids']) == ('safe-routed-raw-wrong', [])
    right = a.categorize(face, ink, safe, {'c': view('(^_^)')}, *unions)
    assert (right['category'], right['raw_correct_ids'], right['best_coverage']) == ('safe-routed-raw-correct', ['c'], 1.0)


def test_safe_routed_requires_exactly_one_raw_view(unions):
    face, ink = truth(), mask((10, 5, 20, 15))
    safe = {'c': candidate('c', (9, 4, 21, 16))}
    with pytest.raises(ValueError, match='exactly one recorded raw view'):
        a.categorize(face, ink, safe, {'c': [{'transform': 'edge-pad-4', 'text': '(^_^)'}]}, *unions)


def test_coverage_bins():
    assert [a.coverage_bin(v) for v in (0.0, 0.49, 0.5, 0.94, 0.95, 1.0)] == ['[0.0,0.5)', '[0.0,0.5)', '[0.5,0.8)', '[0.8,0.95)', '[0.95,1.0)', '1.0']


def decision(reason, accepted):
    return {'reason': reason, 'accepted': accepted}


def test_loss_signature_distinguishes_rejection_restore_absence_and_retention():
    rejected = a.loss_signature({'regions': [], 'split': {}}, {'c': decision('ordinary-body-region-protected', False)}, ['c'])
    assert (rejected['kind'], rejected['accepted'], rejected['retained']) == ('fusion-rejected', False, False)
    accepted = {'c': decision('exact-views-and-independent-pixels', True)}
    conflict = {'regions': [], 'split': {'atomic_conflicts': [{'reason': 'multiple-parent-face-edits', 'candidate_ids': ['c', 'd']}]}}
    restored = a.loss_signature(conflict, accepted, ['c'])
    assert (restored['kind'], restored['conflicts']) == ('accepted-then-parent-restored', ['multiple-parent-face-edits'])
    absent = a.loss_signature({'regions': [], 'split': {}}, accepted, ['c'])
    assert absent['kind'] == 'accepted-absent-from-final-regions'
    kept = a.loss_signature({'regions': [{'candidate_id': 'c'}], 'split': {}}, accepted, ['c'])
    assert (kept['kind'], kept['retained']) == ('accepted-but-final-not-exact', True)
    with pytest.raises(ValueError, match='lacks a recorded fusion decision'):
        a.loss_signature(conflict, {}, ['c'])
    with pytest.raises(ValueError, match='accepted flag'):
        a.loss_signature(conflict, {'c': {'reason': 'x'}}, ['c'])


def face(scene_id, truth_id, category, exact, loss=None, text='(^_^)', pixels='p1'):
    return {'scene_id': scene_id, 'truth_id': truth_id, 'category': category, 'final_exact': exact, 'loss': loss,
            'truth_text': text, 'pixel_sha256': pixels}


def test_compare_reports_lost_and_gained_targets():
    start = {'label': 'start', 'faces': [face('s', 'a', 'safe-routed-raw-correct', True), face('s', 'b', 'partial-only', False)]}
    loss = {'kind': 'accepted-then-parent-restored', 'decision_reasons': [], 'conflicts': ['multiple-parent-face-edits'], 'retained': False}
    tier = {'label': 'tier', 'faces': [face('s', 'a', 'safe-routed-raw-correct', False, loss), face('s', 'b', 'safe-routed-raw-correct', True)]}
    result = a.compare(start, tier)
    assert result['lost_signatures'] == {'accepted-then-parent-restored': 1}
    assert [item['truth_id'] for item in result['gained_targets']] == ['b']
    assert result['category_transitions'] == {'safe-routed-raw-correct -> safe-routed-raw-correct': 1,
                                              'partial-only -> safe-routed-raw-correct': 1}


def test_compare_refuses_populations_that_differ():
    start = {'label': 'start', 'faces': [face('s', 'a', 'nohit', False), face('s', 'b', 'nohit', False)]}
    with pytest.raises(ValueError, match='differ in size'):
        a.compare(start, {'label': 'other', 'faces': [face('s', 'a', 'nohit', False)]})
    with pytest.raises(ValueError, match='populations differ'):
        a.compare(start, {'label': 'other', 'faces': [face('s', 'a', 'nohit', False), face('s', 'zz', 'nohit', False)]})
    with pytest.raises(ValueError, match='not its truth or pixels'):
        a.compare(start, {'label': 'other', 'faces': [face('s', 'a', 'nohit', False), face('s', 'b', 'nohit', False, text='(;_;)')]})
    with pytest.raises(ValueError, match='not its truth or pixels'):
        a.compare(start, {'label': 'other', 'faces': [face('s', 'a', 'nohit', False), face('s', 'b', 'nohit', False, pixels='p2')]})
    with pytest.raises(ValueError, match='repeat a face identity'):
        a.compare({'label': 'start', 'faces': [face('s', 'a', 'nohit', False), face('s', 'a', 'nohit', False)]}, start)


def analysis(exact, missed=0, partial=0, safe=1, routed=1, bodies=(), body=None):
    faces = [dict(face('s', 'a', 'safe-routed-raw-correct', True), coverage_bin='1.0', unsafe_kind=None, routed_any_hit=True,
                  baseline_exact=False, unrouted_reasons=[])]
    recorded = {'exact': exact, 'faces': 1, 'body': body or {},
                'localization': {'missed': missed, 'partial_only': partial, 'safe_complete_coverage': safe, 'safe_routed_complete_coverage': routed}}
    return {'label': 'x', 'report': 'r', 'report_sha256': '0' * 64, 'faces': faces, 'bodies': list(bodies), 'recorded': recorded}


def test_counts_must_match_the_recorded_summary():
    result = a.counts(analysis(exact=1))
    assert (result['final_exact'], result['stage_partition']['safe-routed-raw-correct']) == (1, 1)
    with pytest.raises(ValueError, match='differ from the recorded report summary'):
        a.counts(analysis(exact=0))
    with pytest.raises(ValueError, match='differ from the recorded localization counters'):
        a.counts(analysis(exact=1, safe=0))
    with pytest.raises(ValueError, match='differ from the recorded localization counters'):
        a.counts(analysis(exact=1, routed=0))
    flagged = {'scene_id': 's', 'truth_id': 'b1', 'counters': {'new_body_intersection_errors': 1}}
    assert a.counts(analysis(exact=1, bodies=[flagged], body={'new_body_intersection_errors': 1}))['body_regions_flagged']['count'] == 1
    with pytest.raises(ValueError, match='recorded body protection counters'):
        a.counts(analysis(exact=1, bodies=[flagged], body={'new_body_intersection_errors': 0}))
    with pytest.raises(ValueError, match='recorded body protection counters'):
        a.counts(analysis(exact=1, body={'erroneous_rewrites': 1}))


def test_only_finished_development_reports_are_read():
    good = {'split': 'dev', 'independent_heldout': False, 'heldout_usage': None, 'actual_cpu_inference': True, 'ground_truth_inference_input': False}
    a.require_development_report(good, 'x')
    for bad in ({'split': 'heldout'}, {'independent_heldout': True}, {'heldout_usage': {'claim': 1}}):
        with pytest.raises(ValueError, match='development reports only'):
            a.require_development_report(dict(good, **bad), 'x')
    with pytest.raises(ValueError, match='truth-free CPU development report'):
        a.require_development_report(dict(good, ground_truth_inference_input=True), 'x')


def test_outputs_are_exclusive(tmp_path):
    target = tmp_path / 'out.json'
    a.write(target, {'a': 1})
    with pytest.raises(FileExistsError):
        a.write(target, {'a': 2})


def test_report_arguments_need_labels():
    assert a.parse_reports(['start=a.json'])[0][0] == 'start'
    with pytest.raises(ValueError, match='label=path'):
        a.parse_reports(['a.json'])
