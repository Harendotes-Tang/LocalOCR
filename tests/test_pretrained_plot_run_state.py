"""Run-state arithmetic/schema controls; test fixtures are not measurement evidence."""
from copy import deepcopy
import pytest
from tests.test_pretrained_summary import inputs, completed_paddle_run, completed_openocr_run, failed_openocr_run
from tools.Documentation import build_pretrained_summary as b
from tools.Documentation import plot_pretrained_evaluation as p


def source(tmp_path, retained=False):
    dev, _, runs = inputs(tmp_path)
    runs[0] = completed_paddle_run(runs, retained)
    runs[1] = completed_openocr_run(runs)
    return b.build(dev, runs, fixture=True)[0]


def test_completed_effective_parent_retained_displays_actual_run_state(tmp_path):
    data = source(tmp_path, True);p.validate(data)
    label = p.curve_label(data['training'][0])
    assert '完成' in label and '550' in label and '保留父权重' in label
    assert '500 步' not in label and '0 步' not in label
    label = p.curve_label(data['training'][1])
    assert '运行内选取 500 步' in label and '胜出' not in label


def test_failed_curve_has_failure_status_and_real_checkpoint_step(tmp_path):
    dev, _, runs = inputs(tmp_path);runs = failed_openocr_run(runs)
    data, _ = b.build(dev, runs, fixture=True);p.validate(data)
    label = p.curve_label(data['training'][1])
    assert '中断' in label and '24,050' in label and '18,000' in label


@pytest.mark.parametrize('change', [
    lambda r: r['run_state'].update(status='production-winner'),
    lambda r: r['run_state'].update(actual_steps=True),
    lambda r: r['run_state'].update(actual_steps=499),
    lambda r: r['run_state'].update(actual_steps=0),
    lambda r: r['run_state']['checkpoint_selection'].update(outcome='global-winner'),
    lambda r: r['run_state']['checkpoint_selection'].update(step=0),
    lambda r: r['run_state']['checkpoint_selection'].update(step=499),
    lambda r: r['run_state']['checkpoint_selection'].update(sha256='private text'),
    lambda r: r['run_state'].update(status='failed'),
    lambda r: r['run_state'].update(status='progress-snapshot'),
    lambda r: r['run_state'].update(private_prediction='never export'),
])
def test_unknown_or_false_completion_and_selection_rejected(tmp_path, change):
    data = source(tmp_path);change(data['training'][0])
    with pytest.raises(ValueError):p.validate(data)


def test_retained_parent_has_no_new_step_and_old_summaries_remain_valid(tmp_path):
    data = source(tmp_path, True)
    data['training'][0]['run_state']['checkpoint_selection']['step'] = 0
    with pytest.raises(ValueError, match='not a new optimizer'):p.validate(data)
    for run in data['training']:run.pop('run_state')
    p.validate(data)
    assert p.curve_label(data['training'][0]) == data['training'][0]['run_id']
