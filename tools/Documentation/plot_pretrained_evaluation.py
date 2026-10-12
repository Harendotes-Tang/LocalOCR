"""Render anonymous, frozen pretrained OCR aggregates as standalone SVG.

The input contains counts and hashes only: no images, sample text, predictions,
paths or corpus labels. A fixture must be declared and is visibly watermarked.
No model execution, dataset access or font download occurs in this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

SCHEMA = 'localocr.pretrained.readme-evaluation.v1'
FONT_SHA256 = '2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b'
FAMILIES = ('ppocrv6-tiny', 'repsvtr-mobile')
KINDS = ('plain', 'face', 'punctuation', 'emoji', 'blank', 'graphic')
NAMES = {'plain': '普通文字', 'face': '颜文字', 'punctuation': '符号',
         'emoji': 'Emoji', 'blank': '空白', 'graphic': '几何图形'}
FAMILY_NAMES = {'ppocrv6-tiny': 'PP-OCRv6 Tiny', 'repsvtr-mobile': 'RepSVTR Mobile'}
TEXT, GRID, BLUE, ORANGE = '#203044', '#e1e7ed', '#2369a6', '#b54e20'
SERIES = (('production-baseline', 'production'),
          ('ppocrv6-tiny', 'official-pretrained'), ('ppocrv6-tiny', 'finetuned'),
          ('repsvtr-mobile', 'official-pretrained'), ('repsvtr-mobile', 'finetuned'))
COLORS = ('#687580', '#9ec2e0', BLUE, '#e5b497', ORANGE)
LABELS = ('当前基线 · PP-OCRv6 Small', 'Tiny · 官方预训练', 'Tiny · 微调后',
          'RepSVTR · 官方预训练', 'RepSVTR · 微调后')
ID = re.compile(r'[a-z0-9][a-z0-9._-]{0,95}\Z')
SHA = re.compile(r'[a-f0-9]{64}\Z')


def keys(value, expected, name, optional=()):
    """Reject unapproved fields so labels, paths or predictions cannot leak."""
    if (not isinstance(value, dict) or not set(expected).issubset(value)
            or set(value) - set(expected) - set(optional)):
        raise ValueError(f'{name}: expected aggregate fields {sorted(expected)}')


def checksum(value, name):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f'{name}: expected lowercase SHA256')


def identifier(value, name):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f'{name}: expected non-path research identity')


def number(value, name, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: invalid nonnegative count')
    return value


def validate_metric(value, name, count=None, negative=False):
    fields = {'count', 'exact', 'characters', 'character_errors'}
    if negative is not None:
        fields.add('false_nonempty_outputs')
    keys(value, fields, name)
    total = number(value['count'], name + '.count')
    if not total or (count is not None and total != count):
        raise ValueError(f'{name}: frozen sample count differs or is empty')
    number(value['exact'], name + '.exact', total)
    number(value['characters'], name + '.characters')
    errors = number(value['character_errors'], name + '.character_errors')
    if errors < total - value['exact']:
        raise ValueError(f'{name}: errors are fewer than nonexact rows')
    if negative is not None:
        false = number(value['false_nonempty_outputs'], name + '.false_nonempty_outputs', total)
        if negative:
            if value['characters'] != 0 or false != total - value['exact']:
                raise ValueError(f'{name}: negative count disagrees with exact count')
        elif false != 0 or value['characters'] == 0:
            raise ValueError(f'{name}: positive category has invalid denominator/negative count')


def validate(data):
    """Require one frozen split and raw counts; absence never becomes zero."""
    keys(data, {'schema_version', 'fixture', 'dataset', 'plot_font', 'training', 'heldout'}, 'summary')
    if data['schema_version'] != SCHEMA or type(data['fixture']) is not bool:
        raise ValueError('Unknown schema or undeclared fixture status')
    binding = data['dataset']
    keys(binding, {'sha256', 'split_sha256', 'counts'}, 'dataset')
    checksum(binding['sha256'], 'dataset.sha256')
    keys(binding['split_sha256'], {'dev', 'heldout'}, 'dataset.split_sha256')
    for field, value in binding['split_sha256'].items():
        checksum(value, 'dataset.split_sha256.' + field)
    keys(binding['counts'], {'dev', 'heldout'}, 'dataset.counts')
    if binding['counts'] != {'dev': 2100, 'heldout': 4200}:
        raise ValueError('This round requires the frozen 2100 dev / 4200 heldout images')
    keys(data['plot_font'], {'sha256', 'license'}, 'plot_font')
    if data['plot_font'] != {'sha256': FONT_SHA256, 'license': 'OFL-1.1'}:
        raise ValueError('Use the existing pinned OFL Noto CJK font')
    if not isinstance(data['training'], list):
        raise ValueError('training: expected a list of actual run aggregates')
    identities = set()
    development_denominator = None
    for run in data['training']:
        keys(run, {'family', 'run_id', 'source_sha256', 'dataset_sha256',
                   'dev_split_sha256', 'dev_count', 'points'}, 'training run',
             optional={'evaluation_backend', 'run_state'})
        if run.get('evaluation_backend', 'CUDA-native') != 'CUDA-native':
            raise ValueError('Training development points require CUDA-native evaluation')
        if run['family'] not in FAMILIES:
            raise ValueError('Unknown training family')
        identifier(run['run_id'], 'training.run_id')
        if run['run_id'] in identities:
            raise ValueError('Repeated training run identity')
        identities.add(run['run_id'])
        checksum(run['source_sha256'], 'training.source_sha256')
        if (run['dataset_sha256'] != binding['sha256'] or
                run['dev_split_sha256'] != binding['split_sha256']['dev'] or run['dev_count'] != 2100):
            raise ValueError('Training curves must use this identical frozen development split')
        if not isinstance(run['points'], list) or not run['points']:
            raise ValueError('A curve requires at least one actual completed dev evaluation')
        previous = -1
        denominator = None
        for point in run['points']:
            keys(point, {'step', 'count', 'exact', 'characters', 'character_errors'}, 'training point')
            step = number(point['step'], 'training.step')
            if step <= previous:
                raise ValueError('Training optimizer steps must strictly increase within a run')
            previous = step
            validate_metric({key: point[key] for key in ('count', 'exact', 'characters', 'character_errors')},
                            'training point', count=2100, negative=None)
            if point['characters'] == 0 or (denominator is not None and denominator != point['characters']):
                raise ValueError('Training CER reference denominator differs')
            denominator = point['characters']
            if development_denominator is not None and development_denominator != denominator:
                raise ValueError('Paired development curves have different reference codepoint counts')
            development_denominator = denominator
        if 'run_state' in run:
            state = run['run_state']
            keys(state, {'status', 'actual_steps', 'checkpoint_selection'}, 'run_state')
            if state['status'] not in ('completed', 'failed', 'progress-snapshot'):
                raise ValueError('Unknown actual run completion status')
            actual = number(state['actual_steps'], 'run_state.actual_steps')
            if actual < previous or actual == 0:
                raise ValueError('Actual run steps are fewer than full dev evaluations or initialization only')
            selection = state['checkpoint_selection']
            keys(selection, {'outcome', 'step', 'sha256'}, 'run checkpoint selection')
            if selection['outcome'] not in ('run-best', 'parent-retained', 'selected-before-failure', 'unrecorded'):
                raise ValueError('Unknown run-local checkpoint outcome')
            if selection['outcome'] in ('run-best', 'selected-before-failure'):
                step = number(selection['step'], 'run checkpoint step', actual)
                if step == 0 or step not in {point['step'] for point in run['points']}:
                    raise ValueError('Selected run checkpoint lacks a completed development point')
                checksum(selection['sha256'], 'run checkpoint SHA')
            elif selection['outcome'] == 'parent-retained':
                if selection['step'] is not None:
                    raise ValueError('Retained parent is not a new optimizer-step checkpoint')
                checksum(selection['sha256'], 'retained parent SHA')
            elif selection['step'] is not None or selection['sha256'] is not None:
                raise ValueError('Unrecorded selection cannot invent a checkpoint')
            if ((state['status'] == 'failed') != (selection['outcome'] == 'selected-before-failure') or
                    state['status'] == 'progress-snapshot' and selection['outcome'] != 'unrecorded' or
                    state['status'] == 'completed' and selection['outcome'] not in ('run-best', 'parent-retained')):
                raise ValueError('Run completion and checkpoint-selection outcome disagree')
    heldout = data['heldout']
    if heldout is None:
        if not data['training']:
            raise ValueError('No actual development or heldout aggregates to plot')
        return
    keys(heldout, {'report_sha256', 'dataset_sha256', 'split_sha256', 'count', 'models'}, 'heldout',
         optional={'evaluation_backend'})
    if heldout.get('evaluation_backend', 'ONNX Runtime CPU') != 'ONNX Runtime CPU':
        raise ValueError('Final heldout comparison requires ONNX Runtime CPU evaluation')
    checksum(heldout['report_sha256'], 'heldout.report_sha256')
    if (heldout['dataset_sha256'] != binding['sha256'] or
            heldout['split_sha256'] != binding['split_sha256']['heldout'] or heldout['count'] != 4200):
        raise ValueError('Heldout comparison must use this identical frozen heldout split')
    if not isinstance(heldout['models'], list) or len(heldout['models']) != 5:
        raise ValueError('Heldout requires production plus both official before / finetuned after pairs')
    stages, ids, denominators = set(), set(), None
    for model in heldout['models']:
        fields = {'id', 'family', 'stage', 'model_sha256', 'by_kind'}
        production = model.get('family') == 'production-baseline' if isinstance(model, dict) else False
        if not production:
            fields.add('transitions_from_production')
        keys(model, fields, 'heldout model')
        identifier(model['id'], 'model.id')
        checksum(model['model_sha256'], 'model.model_sha256')
        stage = model['family'], model['stage']
        if stage not in SERIES or stage in stages or model['id'] in ids:
            raise ValueError('Repeated or unsupported model family/stage/identity')
        stages.add(stage)
        ids.add(model['id'])
        keys(model['by_kind'], set(KINDS), 'model.by_kind')
        category_denominators = {}
        for kind in KINDS:
            metric = model['by_kind'][kind]
            validate_metric(metric, 'model.by_kind.' + kind, negative=kind in ('blank', 'graphic'))
            category_denominators[kind] = metric['count'], metric['characters']
        if sum(item['count'] for item in model['by_kind'].values()) != 4200:
            raise ValueError('Heldout categories do not total 4200 images')
        if denominators is not None and category_denominators != denominators:
            raise ValueError('Paired models have different category counts or reference codepoints')
        denominators = category_denominators
        if not production:
            transitions = model['transitions_from_production']
            keys(transitions, {'by_kind'}, 'model.transitions_from_production')
            keys(transitions['by_kind'], {'plain', 'punctuation'}, 'transitions.by_kind')
            for kind in ('plain', 'punctuation'):
                item = transitions['by_kind'][kind]
                keys(item, {'count', 'baseline_correct_to_wrong', 'erroneous_rewrites'}, 'transition')
                if item['count'] != model['by_kind'][kind]['count']:
                    raise ValueError('Protection transition denominator differs')
                number(item['baseline_correct_to_wrong'], 'transition.correct_to_wrong', item['count'])
                number(item['erroneous_rewrites'], 'transition.erroneous_rewrites', item['count'])
    if stages != set(SERIES):
        raise ValueError('Incomplete official before / finetuned after comparison')
    baseline = next(model for model in heldout['models'] if model['family'] == 'production-baseline')
    for model in heldout['models']:
        if model is baseline:
            continue
        for kind, transition in model['transitions_from_production']['by_kind'].items():
            if transition['baseline_correct_to_wrong'] > baseline['by_kind'][kind]['exact']:
                raise ValueError('Correct-to-wrong exceeds the baseline correct count')


def style(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color=GRID, linewidth=.7)
    for edge in ('top', 'right'):
        ax.spines[edge].set_visible(False)
    for edge in ('bottom', 'left'):
        ax.spines[edge].set_color('#b7c3ce')


def finish(fig, name, output, preview, fixture):
    if fixture:
        fig.text(.5, .93, '绘图验证用匿名数据 · 非测评结果', color='#9a321b',
                 fontsize=12, ha='center', weight='bold')
    svg = output / (name + '.svg')
    fig.savefig(svg, facecolor='white', metadata={'Date': None, 'Creator': 'LocalOCR anonymous evaluation plotter'})
    if preview is not None:
        fig.savefig(preview / (name + '.png'), dpi=160, facecolor='white')
    plt.close(fig)
    # Standalone SVG may reference its own path definitions, never external data.
    document = ET.fromstring(svg.read_bytes())
    for element in document.iter():
        if element.tag.rsplit('}', 1)[-1] in ('script', 'image', 'foreignObject'):
            raise ValueError('SVG contains active or embedded source content')
        for attribute, value in element.attrib.items():
            if attribute.rsplit('}', 1)[-1] == 'href' and not value.startswith('#'):
                raise ValueError('SVG contains an external reference')
    return svg


def curve_label(run):
    state = run.get('run_state')
    if state is None:
        return run['run_id']
    status = {'completed': '完成', 'failed': '中断', 'progress-snapshot': '进度快照'}[state['status']]
    selected = state['checkpoint_selection']
    details = ('保留父权重' if selected['outcome'] == 'parent-retained' else
               '选取状态未记录' if selected['outcome'] == 'unrecorded' else
               '运行内选取 {:,} 步'.format(selected['step']))
    return '{}\n{}；{:,} 次更新；{}'.format(run['run_id'], status, state['actual_steps'], details)


def development(data, output, preview):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4), squeeze=False)
    fig.subplots_adjust(left=.085, right=.96, top=.83, bottom=.18, hspace=.48, wspace=.26)
    fig.suptitle('预训练识别器微调：GPU 开发评估', y=.985, fontsize=17)
    fig.text(.5, .89, 'CUDA 原生框架 · 同 2,100 张合成完整画布 · 原始 Unicode 与空格计分', ha='center', color=TEXT)
    for row, family in enumerate(FAMILIES):
        entries = [run for run in data['training'] if run['family'] == family]
        for column, metric in enumerate(('exact', 'cer')):
            ax = axes[row, column]
            style(ax)
            ax.set_title(FAMILY_NAMES[family], fontsize=12, pad=8)
            ax.set_xlabel('实际优化器更新步数')
            largest_step = max((run.get('run_state', {}).get('actual_steps', run['points'][-1]['step']) for run in entries), default=1)
            ax.set_xlim(0, max(1, largest_step * 1.04))
            ax.set_ylabel('整串精确匹配率（%）' if metric == 'exact' else 'CER（%）')
            if metric == 'exact':
                ax.set_ylim(0, 100)
            else:
                largest = max((point['character_errors'] / point['characters'] * 100
                               for run in entries for point in run['points']), default=1)
                ax.set_ylim(0, max(1, largest * 1.12))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            for index, run in enumerate(entries):
                values = [point['exact'] / point['count'] * 100 if metric == 'exact'
                          else point['character_errors'] / point['characters'] * 100 for point in run['points']]
                ax.plot([point['step'] for point in run['points']], values,
                        color=BLUE if row == 0 else ORANGE, linestyle=('-', '--', ':', '-.')[index % 4],
                        marker='o', markersize=3.5, linewidth=1.5, label=curve_label(run))
            if entries:
                ax.legend(loc='best', frameon=False, fontsize=9)
            else:
                ax.text(.5, .5, '尚无完成的开发集测评记录', transform=ax.transAxes,
                        ha='center', va='center', color='#596776')
    fig.text(.085, .065, '步数仅表示各次运行的实际更新次数；不同 batch 的步数不表示等量训练。CER 可能超过 100%。', fontsize=10)
    fig.text(.085, .027, '仅展示完整 GPU 开发测评点；运行内权重选取不代表最终家族胜出。ONNX Runtime CPU 测评另报。', fontsize=10)
    return finish(fig, 'pretrained-development', output, preview, data['fixture'])


def models_in_order(data):
    models = {(model['family'], model['stage']): model for model in data['heldout']['models']}
    return [models[key] for key in SERIES]


def legend(fig, bottom=.09):
    handles = [Patch(facecolor=color, edgecolor='#526473' if index in (1, 3) else color, hatch='//' if index in (1, 3) else None, label=label)
               for index, (color, label) in enumerate(zip(COLORS, LABELS))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, bottom),
               ncol=3, frameon=False, fontsize=10)


def heldout(data, output, preview, metric):
    models = models_in_order(data)
    positive = KINDS[:4]
    fig, ax = plt.subplots(figsize=(12.8, 6.7))
    fig.subplots_adjust(left=.16, right=.915, top=.82, bottom=.22)
    title = '字符错误率 CER（越低越好）' if metric == 'cer' else '整串精确匹配率（越高越好）'
    fig.suptitle('冻结保留集：' + title, fontsize=17, y=.985)
    fig.text(.5, .89, 'ONNX Runtime CPU · 同 4,200 张合成画布 · 当前基线及两候选微调前后', ha='center', color=TEXT)
    largest = 0
    for index, (model, color) in enumerate(zip(models, COLORS)):
        values = [model['by_kind'][kind]['character_errors'] / model['by_kind'][kind]['characters'] * 100
                  if metric == 'cer' else model['by_kind'][kind]['exact'] / model['by_kind'][kind]['count'] * 100
                  for kind in positive]
        largest = max(largest, *values)
        positions = [group + (index - 2) * .145 for group in range(len(positive))]
        ax.barh(positions, values, height=.125, color=color, edgecolor='#526473' if index in (1, 3) else color, linewidth=.4,
                hatch='//' if index in (1, 3) else None)
        for position, value in zip(positions, values):
            ax.text(value + 1.0, position, f'{value:.2f}', va='center', fontsize=9, color=TEXT)
    ax.set_yticks(range(len(positive)), [f"{NAMES[kind]}\n(n={models[0]['by_kind'][kind]['count']:,})" for kind in positive])
    ax.set_ylim(-.45, len(positive) - .55)
    ax.invert_yaxis()
    ax.set_xlim(0, max(10, largest * 1.15) if metric == 'cer' else 110)
    if metric == 'exact':
        ax.set_xticks((0, 20, 40, 60, 80, 100))
    ax.set_xlabel('CER（%）' if metric == 'cer' else '整串精确匹配率（%）')
    style(ax, horizontal=True)
    legend(fig, .075)
    fig.text(.07, .025, '精确 Unicode 与空格，不作归一化；同家族前后使用相同预处理。合成画布指标不代表截图端到端准确率。', fontsize=10)
    return finish(fig, 'pretrained-heldout-' + metric, output, preview, data['fixture'])


def protection(data, output, preview):
    models = models_in_order(data)
    fig = plt.figure(figsize=(12.8, 7.6))
    grid = fig.add_gridspec(2, 2, left=.085, right=.965, top=.82, bottom=.23, hspace=.68, wspace=.23)
    negative = fig.add_subplot(grid[0, :])
    regressions, rewrites = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.suptitle('冻结保留集：负样本与普通文字保护计数', fontsize=17, y=.985)
    fig.text(.5, .89, 'ONNX Runtime CPU · 同 4,200 张画布 · 普通文字与符号分别统计', ha='center', color=TEXT)
    for ax, metric, categories, candidates in ((negative, 'false_nonempty_outputs', ('blank', 'graphic'), models),
            (regressions, 'baseline_correct_to_wrong', ('plain', 'punctuation'), models[1:]),
            (rewrites, 'erroneous_rewrites', ('plain', 'punctuation'), models[1:])):
        style(ax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        maximum = 0
        for index, model in enumerate(candidates):
            series_index = index if ax is negative else index + 1
            values = [model['by_kind'][kind][metric] if ax is negative else
                      model['transitions_from_production']['by_kind'][kind][metric] for kind in categories]
            maximum = max(maximum, *values)
            positions = [group + (index - (len(candidates) - 1) / 2) * .14 for group in range(2)]
            ax.bar(positions, values, width=.12, color=COLORS[series_index], linewidth=.4,
                   edgecolor='#526473' if series_index in (1, 3) else COLORS[series_index],
                   hatch='//' if series_index in (1, 3) else None)
            for position, value in zip(positions, values):
                ax.text(position, value + max(.5, maximum * .025), str(value), ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(5, maximum * 1.23))
        ax.set_xticks(range(2), [f"{NAMES[kind]} (n={models[0]['by_kind'][kind]['count']:,})" for kind in categories])
        ax.set_ylabel('画布数量')
        ax.set_title({'false_nonempty_outputs': '负样本非空输出', 'baseline_correct_to_wrong': '相对当前基线：正确转错',
                      'erroneous_rewrites': '相对当前基线：错误改写'}[metric], fontsize=12, pad=8)
    legend(fig, .095)
    fig.text(.085, .048, '错误改写：输出不同于当前基线，且仍不等于标签。负样本计数限原始识别器，不含检测或区域选择。', fontsize=10)
    fig.text(.085, .019, '当前结果仅用于研究评估；通过这些计数不能替代真实截图、资源、许可及 APP 生命周期门禁。', fontsize=10)
    return finish(fig, 'pretrained-protection', output, preview, data['fixture'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Anonymous count/hash-only summary JSON')
    parser.add_argument('--font', required=True, type=Path, help='Locally prepared pinned NotoSansCJKsc-Regular.otf')
    parser.add_argument('--output', required=True, type=Path, help='Standalone SVG output directory')
    parser.add_argument('--preview', type=Path, help='Optional local raster preview directory')
    args = parser.parse_args(argv)
    raw = args.data.read_bytes()
    data = json.loads(raw.decode('utf-8'))
    validate(data)
    if data['fixture']:
        artifact_root = Path(__file__).resolve().parents[2] / 'artifacts'
        for destination in (args.output, args.preview):
            if destination is not None and not destination.resolve().is_relative_to(artifact_root.resolve()):
                raise ValueError('Fixture figures must remain in local artifacts, never documentation')
    if hashlib.sha256(args.font.read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('Font bytes differ from pinned OFL font')
    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 11, 'text.color': TEXT,
        'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
        'svg.fonttype': 'path', 'svg.hashsalt': 'localocr-pretrained-evaluation-v1'})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
    generated = []
    if data['training']:
        generated.append(development(data, args.output, args.preview))
    if data['heldout'] is not None:
        generated.extend(heldout(data, args.output, args.preview, metric) for metric in ('cer', 'exact'))
        generated.append(protection(data, args.output, args.preview))
    receipt = {'schema_version': 'localocr.pretrained.plot-receipt.v1', 'fixture': data['fixture'],
        'summary_sha256': hashlib.sha256(raw).hexdigest(), 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'font_sha256': FONT_SHA256, 'font_license': 'OFL-1.1', 'matplotlib': matplotlib.__version__,
        'training_evaluation_backend': 'CUDA-native' if data['training'] else None,
        'heldout_evaluation_backend': 'ONNX Runtime CPU' if data['heldout'] is not None else None,
        'figures': [{'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'bytes': path.stat().st_size} for path in generated], 'external_or_active_svg_references': 0}
    (args.output / 'plot-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'fixture': data['fixture'], 'figures': [path.name for path in generated]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
