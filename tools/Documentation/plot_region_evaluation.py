"""Plot anonymous, hash-bound whole-scene regional research aggregates offline.

Only integer counts, fixed enums and SHA-256 identities are accepted. This
command never opens scenes, images, truth regions, predictions or model files.
Development and heldout figures are separate. Crop-only measurements are invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SCHEMA = 'localocr.region.readme-evaluation.v1'
FONT_SHA256 = '2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b'
COUNTS = {'dev': {'scenes': 200, 'faces': 120}, 'heldout': {'scenes': 400, 'faces': 280}}
FAMILIES = ('production-baseline', 'ppocrv6-tiny', 'repsvtr-mobile')
NAMES = ('当前基线', 'Tiny · 最终融合', 'RepSVTR · 最终融合')
COLORS = ('#687580', '#2369a6', '#b54e20')
TEXT, GRID = '#203044', '#e1e7ed'
SHA = re.compile(r'[a-f0-9]{64}\Z')
ID = re.compile(r'[a-z0-9][a-z0-9._-]{0,95}\Z')
SCOPE = 'automatic whole-scene regional fusion; exact Unicode and whitespace'
BACKEND = 'ONNX Runtime CPU'


def keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{name}: missing or unknown aggregate fields')


def checksum(value, name):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f'{name}: expected lowercase SHA256')


def integer(value, name, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: invalid nonnegative count')
    return value


def metric(value, name, expected_count=None):
    keys(value, {'count', 'exact', 'characters', 'character_errors'}, name)
    count = integer(value['count'], name + '.count')
    if not count or (expected_count is not None and count != expected_count):
        raise ValueError(f'{name}: frozen region denominator differs')
    integer(value['exact'], name + '.exact', count)
    if integer(value['characters'], name + '.characters') == 0:
        raise ValueError(f'{name}: empty reference character denominator')
    if integer(value['character_errors'], name + '.character_errors') < count - value['exact']:
        raise ValueError(f'{name}: errors fewer than nonexact regions')


def strict_json(raw):
    """Reject duplicate fields and nonfinite numbers before schema validation."""
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON field')
            value[key] = item
        return value
    def invalid(_):
        raise ValueError('Nonfinite JSON number')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def destination(data, output, preview):
    """Keep fixture figures local rather than publishing illustrative counts."""
    if not data['fixture']:
        return
    artifacts = Path(__file__).resolve().parents[2] / 'artifacts'
    for target in (output, preview):
        if target is not None and not target.resolve().is_relative_to(artifacts.resolve()):
            raise ValueError('Fixture output must remain in artifacts, never documentation')


def style(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color=GRID, linewidth=.7)
    for edge in ('top', 'right'):
        ax.spines[edge].set_visible(False)
    for edge in ('bottom', 'left'):
        ax.spines[edge].set_color('#b7c3ce')


def finish(fig, name, output, preview, fixture):
    import matplotlib.pyplot as plt
    if fixture:
        fig.text(.5, .938, '绘图验证用匿名数据 · 非测评结果', color='#9a321b',
                 fontsize=12, ha='center', weight='bold')
    path = output / (name + '.svg')
    fig.savefig(path, facecolor='white', metadata={'Date': None, 'Creator': 'LocalOCR anonymous regional evaluation plotter'})
    if preview is not None:
        fig.savefig(preview / (name + '.png'), dpi=160, facecolor='white')
    plt.close(fig)
    for item in ET.fromstring(path.read_bytes()).iter():
        if item.tag.rsplit('}', 1)[-1] in ('script', 'image', 'foreignObject', 'text'):
            raise ValueError('SVG has active content, embedded pixels or unconverted text')
        for key, value in item.attrib.items():
            if key.rsplit('}', 1)[-1] == 'href' and not value.startswith('#'):
                raise ValueError('SVG has an external reference')
    return path


def validate_report(report, split, dataset):
    fields = {'report_sha256', 'dataset_sha256', 'split_sha256', 'pipeline_sha256',
              'selection_sha256', 'first_use_sha256', 'config_sha256',
              'evaluation_backend', 'evaluation_scope', 'runtime', 'scenes',
              'localization', 'models'}
    keys(report, fields, split)
    for field in ('report_sha256', 'pipeline_sha256', 'config_sha256'):
        checksum(report[field], split + '.' + field)
    if (report['dataset_sha256'] != dataset['sha256']
            or report['split_sha256'] != dataset['split_sha256'][split]
            or report['scenes'] != COUNTS[split]['scenes']):
        raise ValueError('Report dataset/split/count binding differs')
    if report['evaluation_backend'] != BACKEND or report['evaluation_scope'] != SCOPE:
        raise ValueError('Whole-scene final regional CPU scope is required')
    if split == 'heldout':
        checksum(report['selection_sha256'], 'heldout.selection_sha256')
        checksum(report['first_use_sha256'], 'heldout.first_use_sha256')
    elif report['first_use_sha256'] is not None:
        raise ValueError('Development report cannot claim heldout first use')
    elif report['selection_sha256'] is not None:
        checksum(report['selection_sha256'], 'dev.selection_sha256')
    keys(report['runtime'], {'providers', 'intra_op_threads', 'inter_op_threads'}, 'runtime')
    if (type(report['runtime']['intra_op_threads']) is not int or type(report['runtime']['inter_op_threads']) is not int
            or report['runtime'] != {'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2, 'inter_op_threads': 1}):
        raise ValueError('Report requires the same CPU two-thread runtime')
    loc = report['localization']
    keys(loc, {'faces', 'found_any', 'missed', 'partial_only', 'complete_ink_coverage',
               'safe_complete_coverage', 'eligible_complete_ink_coverage', 'routed_complete_ink_coverage',
               'safe_routed_complete_coverage', 'candidate_count', 'intersecting_body', 'intersecting_graphic',
               'routed_candidate_count', 'routed_candidates_intersecting_body', 'routed_candidates_intersecting_graphic',
               'body_regions', 'body_regions_with_candidate', 'body_regions_routed', 'graphic_regions',
               'graphic_regions_with_candidate', 'graphic_regions_routed', 'negative_scenes',
               'negative_scenes_with_candidate', 'negative_scenes_routed'}, 'localization')
    for field, value in loc.items():
        integer(value, 'localization.' + field)
    faces = COUNTS[split]['faces']
    if (loc['faces'] != faces or not 0 <= loc['safe_complete_coverage'] <= loc['complete_ink_coverage'] <= loc['found_any'] <= faces
            or loc['missed'] != faces - loc['found_any']
            or loc['partial_only'] != loc['found_any'] - loc['complete_ink_coverage']
            or not 0 <= loc['safe_routed_complete_coverage'] <= loc['routed_complete_ink_coverage'] <= loc['eligible_complete_ink_coverage'] <= loc['complete_ink_coverage']
            or loc['safe_routed_complete_coverage'] > loc['safe_complete_coverage']):
        raise ValueError('Face discovered/eligible/routed coverage or frozen denominator differs')
    if (max(loc['intersecting_body'], loc['intersecting_graphic']) > loc['candidate_count']
            or loc['routed_candidate_count'] > loc['candidate_count']
            or max(loc['routed_candidates_intersecting_body'], loc['routed_candidates_intersecting_graphic']) > loc['routed_candidate_count']
            or loc['routed_candidates_intersecting_body'] > loc['intersecting_body']
            or loc['routed_candidates_intersecting_graphic'] > loc['intersecting_graphic']):
        raise ValueError('Candidate routing numerator exceeds its own denominator')
    for kind, unit in (('body', 'regions'), ('graphic', 'regions'), ('negative', 'scenes')):
        total = loc[f'{kind}_{unit}']
        if not total or not 0 <= loc[f'{kind}_{unit}_routed'] <= loc[f'{kind}_{unit}_with_candidate'] <= total:
            raise ValueError('Truth-region/negative-scene routing denominator differs')
    if loc['negative_scenes'] > report['scenes']:
        raise ValueError('Negative scene count exceeds whole split')
    if loc['candidate_count'] == 0 and any(loc[key] for key in
            ('found_any', 'intersecting_body', 'intersecting_graphic', 'body_regions_with_candidate',
             'graphic_regions_with_candidate', 'negative_scenes_with_candidate')):
        raise ValueError('Discovered coverage without any candidate')
    if loc['routed_candidate_count'] == 0 and any(loc[key] for key in
            ('routed_complete_ink_coverage', 'body_regions_routed', 'graphic_regions_routed', 'negative_scenes_routed')):
        raise ValueError('Actual routing coverage without any routed candidate')
    models = report['models']
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError('Whole-scene comparison requires baseline and both final fusion pipelines')
    identities, families, denominators = set(), set(), None
    for model in models:
        keys(model, {'id', 'family', 'stage', 'model_sha256', 'vocabulary_sha256',
                     'preprocessing_sha256', 'face_regions', 'body', 'negative'}, 'model')
        if not isinstance(model['id'], str) or not ID.fullmatch(model['id']) or model['id'] in identities:
            raise ValueError('Invalid/repeated non-path model identity')
        identities.add(model['id'])
        family = model['family']
        if family not in FAMILIES or family in families:
            raise ValueError('Unknown/repeated model family')
        families.add(family)
        expected = 'production' if family == 'production-baseline' else 'final-fusion'
        if model['stage'] != expected:
            raise ValueError('Crop-only or raw specialist stage cannot substitute final fusion')
        for field in ('model_sha256', 'vocabulary_sha256', 'preprocessing_sha256'):
            checksum(model[field], 'model.' + field)
        metric(model['face_regions'], 'face_regions', faces)
        body = model['body']
        keys(body, {'regions', 'baseline_exact', 'final_exact', 'baseline_correct_to_wrong', 'erroneous_rewrites',
                    'body_intersection_transcript_changes', 'new_body_intersection_errors'}, 'body')
        for field, value in body.items():
            integer(value, 'body.' + field)
        if (body['regions'] != loc['body_regions'] or max(body['baseline_exact'], body['final_exact']) > body['regions']
                or body['baseline_correct_to_wrong'] > body['baseline_exact']
                or not body['baseline_correct_to_wrong'] <= body['erroneous_rewrites'] <= body['regions'] - body['final_exact']
                or not body['new_body_intersection_errors'] <= body['body_intersection_transcript_changes'] <= body['regions']):
            raise ValueError('Body protection numerator/denominator differs')
        if family == 'production-baseline' and (body['final_exact'] != body['baseline_exact'] or any(body[key] for key in
                ('baseline_correct_to_wrong', 'erroneous_rewrites', 'body_intersection_transcript_changes', 'new_body_intersection_errors'))):
            raise ValueError('Production baseline cannot regress relative to itself')
        negative = model['negative']
        keys(negative, {'blank', 'graphic-only', 'graphic_regions'}, 'negative')
        negative_counts, baseline_negatives = {}, {}
        for kind in ('blank', 'graphic-only', 'graphic_regions'):
            unit = 'regions' if kind == 'graphic_regions' else 'scenes'
            keys(negative[kind], {unit, 'baseline_nonempty', 'final_nonempty', 'new_nonempty'}, 'negative.' + kind)
            total = integer(negative[kind][unit], 'negative.' + kind + '.' + unit)
            if total == 0 or (unit == 'scenes' and total > report['scenes']):
                raise ValueError('Missing/out-of-range negative denominator')
            for key in ('baseline_nonempty', 'final_nonempty', 'new_nonempty'):
                integer(negative[kind][key], 'negative.' + kind + '.' + key, total)
            if (negative[kind]['new_nonempty'] > min(negative[kind]['final_nonempty'], total - negative[kind]['baseline_nonempty'])
                    or negative[kind]['final_nonempty'] > negative[kind]['baseline_nonempty'] + negative[kind]['new_nonempty']):
                raise ValueError('Negative new-nonempty transition is inconsistent')
            if family == 'production-baseline' and (negative[kind]['baseline_nonempty'] != negative[kind]['final_nonempty'] or negative[kind]['new_nonempty'] != 0):
                raise ValueError('Production negative baseline cannot change relative to itself')
            if unit == 'regions' and total != loc['graphic_regions']:
                raise ValueError('Graphic region negative denominator differs')
            negative_counts[kind] = total
            baseline_negatives[kind] = negative[kind]['baseline_nonempty']
        if sum(negative_counts[key] for key in ('blank', 'graphic-only')) != loc['negative_scenes']:
            raise ValueError('Negative scene categories differ from the localization denominator')
        current = (model['face_regions']['characters'], body['regions'], body['baseline_exact'], negative_counts, baseline_negatives)
        if denominators is not None and current != denominators:
            raise ValueError('Paired pipelines have different reference denominators/baseline counts')
        denominators = current
    if families != set(FAMILIES):
        raise ValueError('Incomplete final regional pipeline comparison')


def validate(data):
    """Keep different splits, stages, backends and denominators distinguishable."""
    keys(data, {'schema_version', 'fixture', 'dataset', 'plot_font', 'dev', 'heldout'}, 'summary')
    if data['schema_version'] != SCHEMA or type(data['fixture']) is not bool:
        raise ValueError('Unknown schema or undeclared fixture status')
    binding = data['dataset']
    keys(binding, {'sha256', 'split_sha256', 'counts'}, 'dataset')
    checksum(binding['sha256'], 'dataset.sha256')
    keys(binding['split_sha256'], {'dev', 'heldout'}, 'dataset.split_sha256')
    for split, value in binding['split_sha256'].items():
        checksum(value, 'dataset.split_sha256.' + split)
    if binding['split_sha256']['dev'] == binding['split_sha256']['heldout']:
        raise ValueError('Development and heldout split hashes must differ')
    if binding['counts'] != COUNTS:
        raise ValueError('Require frozen 200/120 development and 400/280 heldout scenes/faces')
    if data['plot_font'] != {'sha256': FONT_SHA256, 'license': 'OFL-1.1'}:
        raise ValueError('Use the existing pinned OFL Noto CJK font')
    if data['dev'] is None and data['heldout'] is None:
        raise ValueError('No completed whole-scene evaluation report')
    for split in ('dev', 'heldout'):
        if data[split] is not None:
            validate_report(data[split], split, binding)
    if data['dev'] is not None and data['heldout'] is not None:
        if data['dev']['report_sha256'] == data['heldout']['report_sha256']:
            raise ValueError('Two distinct split reports cannot have the same source bytes')
        if data['dev']['selection_sha256'] is not None and data['dev']['selection_sha256'] != data['heldout']['selection_sha256']:
            raise ValueError('Development/heldout selection receipt differs')
        for field in ('pipeline_sha256', 'config_sha256'):
            if data['dev'][field] != data['heldout'][field]:
                raise ValueError('Development/heldout frozen pipeline differs')
        fields = ('family', 'id', 'stage', 'model_sha256', 'vocabulary_sha256', 'preprocessing_sha256')
        bindings = lambda report: sorted(tuple(model[field] for field in fields) for model in report['models'])
        if bindings(data['dev']) != bindings(data['heldout']):
            raise ValueError('Development/heldout model binding differs')


def title(fig, split, topic, scenes):
    name = '开发集' if split == 'dev' else '冻结保留集'
    fig.suptitle(name + '：' + topic, y=.989, fontsize=17)
    fig.text(.5, .884, f'ONNX Runtime CPU · 同 {scenes:,} 张原创合成整图 · 自动定位与最终区域融合',
             ha='center', color=TEXT, fontsize=11)


def count_bar(ax, counts, totals, labels, colors, xlabel, percentages=True, secondary=None):
    """Annotate each rate with its own numerator and denominator; zero is absent."""
    values = [100 * count / total if total else 0 for count, total in zip(counts, totals)] if percentages else counts
    maximum = max(values, default=0)
    ax.barh(range(len(values)), values, height=.5, color=colors)
    for index, (count, total, value) in enumerate(zip(counts, totals, values)):
        label = f'{value:.2f}%  ({count:,}/{total:,})' if percentages and total else f'{count:,}/{total:,}'
        if not total:
            label = '无可评价分母'
        if secondary is not None:
            label += '\n' + secondary[index]
        ax.text(value + max(1.5, maximum * .025), index, label, va='center', fontsize=10)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(115, maximum * 1.45) if percentages else max(5, maximum * 1.5))
    ax.set_xlabel(xlabel, fontsize=10)
    style(ax, horizontal=True)


def localization(data, split, output, preview):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    report = data[split]
    loc = report['localization']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    fig.subplots_adjust(left=.115, right=.985, top=.79, bottom=.205, wspace=.55, hspace=.72)
    title(fig, split, '候选发现、真实路由与完整性', report['scenes'])
    ax = axes[0, 0]
    partitions = [loc['safe_complete_coverage'], loc['complete_ink_coverage'] - loc['safe_complete_coverage'],
                  loc['partial_only'], loc['missed']]
    labels = ('安全完整', '完整但相交邻域', '部分覆盖', '完全漏检')
    shades = ('#2369a6', '#739ab5', '#d3a253', '#b54e20')
    left = 0
    for count, label, color in zip(partitions, labels, shades):
        value = count / loc['faces'] * 100
        ax.barh([0], [value], left=left, height=.6, color=color)
        if value >= 10:
            ax.text(left + value / 2, 0, str(count), ha='center', va='center', fontsize=11,
                    color='white' if color in ('#2369a6', '#b54e20') else TEXT)
        left += value
    ax.set_xlim(0, 100)
    ax.set_ylim(-.6, 2.0)
    ax.set_yticks([])
    ax.set_xlabel(f"全部真值颜文字区域（n={loc['faces']:,}）；比例（%）", fontsize=10)
    ax.set_title('发现完整性分区（互斥且穷尽）', fontsize=12, pad=10)
    ax.legend(handles=[Patch(facecolor=color, label=f'{label}：{count}/{loc["faces"]}')
                       for color, label, count in zip(shades, labels, partitions)],
              loc='upper left', bbox_to_anchor=(-.02, 1.03), ncol=1, frameon=False, fontsize=9)
    style(ax, horizontal=True)
    fields = ('complete_ink_coverage', 'safe_complete_coverage', 'eligible_complete_ink_coverage',
              'routed_complete_ink_coverage', 'safe_routed_complete_coverage')
    ax = axes[0, 1]
    count_bar(ax, [loc[field] for field in fields], [loc['faces']] * len(fields),
              ['发现：全墨完整', '发现：安全完整', '资格：全墨完整', '运行：全墨完整', '运行：安全完整'],
              ['#687580', '#97a7b5', '#d3a253', COLORS[1], COLORS[2]], '真值颜文字区域占比（%）')
    ax.set_title('发现→资格→实际专项运行', fontsize=12, pad=10)
    ax = axes[0, 2]
    count_bar(ax, [loc['intersecting_body'], loc['intersecting_graphic'],
                   loc['routed_candidates_intersecting_body'], loc['routed_candidates_intersecting_graphic']],
              [loc['candidate_count']] * 2 + [loc['routed_candidate_count']] * 2,
              ['发现：相交正文', '发现：相交图形', '运行：相交正文', '运行：相交图形'],
              ['#9ec2e0', '#e5b497', COLORS[1], COLORS[2]], '各自候选分母占比（%）')
    ax.set_title(f"候选发现 {loc['candidate_count']:,} / 运行 {loc['routed_candidate_count']:,}", fontsize=12, pad=10)
    for ax, kind, unit, label in ((axes[1, 0], 'body', 'regions', '正文真值区域'),
                                  (axes[1, 1], 'graphic', 'regions', '图形真值区域'),
                                  (axes[1, 2], 'negative', 'scenes', '空白及仅图形整图')):
        count_bar(ax, [loc[f'{kind}_{unit}_with_candidate'], loc[f'{kind}_{unit}_routed']],
                  [loc[f'{kind}_{unit}']] * 2, ['存在相交候选', '实际专项路由'],
                  ['#97a7b5', COLORS[2]], '各类真值区域占比（%）' if unit == 'regions' else '负样本整图占比（%）')
        ax.set_title(label, fontsize=12, pad=10)
    fig.text(.075, .122, '全墨完整：至少一个候选覆盖全部颜文字墨迹；安全完整：另要求不相交正文或图形墨迹。', fontsize=10)
    fig.text(.075, .08, '发现与实际运行分别计数；正文与图形相交候选可重叠。完全漏检指所有候选均未相交该颜文字区域墨迹。', fontsize=10)
    fig.text(.075, .038, '候选误路由不等于最终融合误改；本图仅为原创整图研究证据，不含单裁片指标或 APP 验收。', fontsize=10)
    return finish(fig, 'region-' + split + '-localization', output, preview, data['fixture'])


def final_results(data, split, output, preview):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    report = data[split]
    models = [next(model for model in report['models'] if model['family'] == family) for family in FAMILIES]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    fig.subplots_adjust(left=.115, right=.985, top=.81, bottom=.225, wspace=.53, hspace=.66)
    title(fig, split, '最终区域识别与正文保护', report['scenes'])
    count_bar(axes[0, 0], [m['face_regions']['exact'] for m in models],
              [m['face_regions']['count'] for m in models], NAMES, COLORS, '真值颜文字区域精确匹配率（%）')
    axes[0, 0].set_title('最终颜文字整串精确匹配', fontsize=12, pad=9)
    count_bar(axes[0, 1], [m['face_regions']['character_errors'] for m in models],
              [m['face_regions']['characters'] for m in models], NAMES, COLORS, 'CER（%）；编辑距离 / 真值码点')
    axes[0, 1].set_title('最终颜文字字符错误率', fontsize=12, pad=9)
    candidates = models[1:]
    ax = axes[0, 2]
    entries = [(model, kind) for model in candidates for kind in ('baseline_correct_to_wrong', 'erroneous_rewrites')]
    count_bar(ax, [model['body'][kind] for model, kind in entries],
              [model['body']['baseline_exact'] if kind == 'baseline_correct_to_wrong' else model['body']['regions']
               for model, kind in entries],
              [label + '\n' + ('正确转错' if kind == 'baseline_correct_to_wrong' else '错误改写')
               for label in NAMES[1:] for kind in ('baseline_correct_to_wrong', 'erroneous_rewrites')],
              [COLORS[1], '#9ec2e0', COLORS[2], '#e5b497'], '各自明确分母占比（%）')
    ax.set_title('正文真值区域新增错误', fontsize=12, pad=9)
    entries = [(model, kind) for model in candidates for kind in ('body_intersection_transcript_changes', 'new_body_intersection_errors')]
    count_bar(axes[1, 0], [model['body'][kind] for model, kind in entries],
              [model['body']['regions'] for model, kind in entries],
              [label + '\n' + ('完整预测变化' if kind == 'body_intersection_transcript_changes' else '新增正文错误')
               for label in NAMES[1:] for kind in ('body_intersection_transcript_changes', 'new_body_intersection_errors')],
              ['#9ec2e0', COLORS[1], '#e5b497', COLORS[2]], '全部正文真值区域占比（%）')
    axes[1, 0].set_title('正文完整预测保护', fontsize=12, pad=9)
    ax = axes[1, 1]
    entries = [(model, kind) for model in models for kind in ('blank', 'graphic-only')]
    count_bar(ax, [m['negative'][kind]['final_nonempty'] for m, kind in entries],
              [m['negative'][kind]['scenes'] for m, kind in entries],
              [label + '\n' + ('空白' if kind == 'blank' else '仅图形') for label in NAMES for kind in ('blank', 'graphic-only')],
              [color for color in COLORS for _ in range(2)], '各类负样本整图占比（%）',
              secondary=[f"新增非空：{m['negative'][kind]['new_nonempty']}/{m['negative'][kind]['scenes']}" for m, kind in entries])
    ax.set_title('负样本最终非空输出', fontsize=12, pad=9)
    count_bar(axes[1, 2], [m['negative']['graphic_regions']['final_nonempty'] for m in models],
              [m['negative']['graphic_regions']['regions'] for m in models], NAMES, COLORS, '全部图形真值区域占比（%）',
              secondary=[f"新增非空：{m['negative']['graphic_regions']['new_nonempty']}/{m['negative']['graphic_regions']['regions']}" for m in models])
    axes[1, 2].set_title('图形区域最终非空输出', fontsize=12, pad=9)
    fig.legend(handles=[Patch(facecolor=color, label=name) for color, name in zip(COLORS, NAMES)],
               loc='lower center', bbox_to_anchor=(.5, .135), ncol=3, frameon=False, fontsize=11)
    fig.text(.075, .117, '整图自动候选→专项多视图→像素与正文保护→融合输出；所有真值颜文字区域纳入，不仅统计已选裁片。', fontsize=10)
    fig.text(.075, .079, '正确转错分母为基线正确正文区域；错误改写分母为全部正文区域。完整预测变化与新增正文错误另计，均以全部正文区域为分母。', fontsize=10)
    fig.text(.075, .039, '精确 Unicode 与空格，不作归一化。CER 可超过 100%；合成整图研究成绩不代表自然截图或 APP 接入验收。', fontsize=10)
    return finish(fig, 'region-' + split + '-final-results', output, preview, data['fixture'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Anonymous count/hash-only regional summary JSON')
    parser.add_argument('--font', required=True, type=Path, help='Existing pinned local OFL Noto CJK font')
    parser.add_argument('--output', required=True, type=Path, help='Standalone SVG output directory')
    parser.add_argument('--preview', type=Path, help='Optional local raster preview directory')
    args = parser.parse_args(argv)
    raw = args.data.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError('Anonymous regional summary exceeds 1 MiB')
    data = strict_json(raw)
    validate(data)
    destination(data, args.output, args.preview)
    if hashlib.sha256(args.font.read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('Font bytes differ from pinned OFL font')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 10, 'text.color': TEXT,
        'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
        'svg.fonttype': 'path', 'svg.hashsalt': 'localocr-region-evaluation-v1'})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
    generated = []
    for split in ('dev', 'heldout'):
        if data[split] is not None:
            generated.extend((localization(data, split, args.output, args.preview),
                              final_results(data, split, args.output, args.preview)))
    receipt = {'schema_version': 'localocr.region.plot-receipt.v1', 'fixture': data['fixture'],
        'summary_sha256': hashlib.sha256(raw).hexdigest(), 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'font_sha256': FONT_SHA256, 'font_license': 'OFL-1.1', 'matplotlib': matplotlib.__version__,
        'evaluation_backend': BACKEND, 'evaluation_scope': SCOPE,
        'reports': {split: {'report_sha256': data[split]['report_sha256'], 'pipeline_sha256': data[split]['pipeline_sha256'],
                           'split_sha256': data[split]['split_sha256']} for split in ('dev', 'heldout') if data[split] is not None},
        'figures': [{'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'bytes': path.stat().st_size} for path in generated], 'external_or_active_svg_references': 0}
    (args.output / 'plot-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'fixture': data['fixture'], 'figures': [path.name for path in generated]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
