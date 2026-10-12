"""Render README figures from anonymous frozen evaluation aggregates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, TEAL, TEXT = '#2369a6', '#b54e20', '#087f7b', '#203044'


def save(fig, name: str, output: Path, preview: Path | None) -> None:
    """Write standalone SVG; optional raster previews stay outside docs."""
    fig.savefig(output / f'{name}.svg', facecolor='white', metadata={'Date': None})
    if preview is not None:
        fig.savefig(preview / f'{name}.png', dpi=160, facecolor='white')
    plt.close(fig)


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e1e7ed', linewidth=0.7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color('#b7c3ce')


def development(data: dict, output: Path, preview: Path | None) -> None:
    """Separate unequal development sets; show incomplete v3 as missing data."""
    iterations = {r['historical_marker']: r for r in data['training']['iterations']}
    v1, v2, v4 = (iterations[k] for k in ('v1', 'v2', 'v4'))
    if v1['split_sha256']['dev'] != v2['split_sha256']['dev']:
        raise ValueError('v1/v2 development curves require the same frozen split')
    fig = plt.figure(figsize=(12.8, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 0.48, 1.15),
                            left=0.075, right=0.975, top=0.73, bottom=0.22, wspace=0.3)
    left, missing, right = (fig.add_subplot(grid[0, i]) for i in range(3))
    fig.suptitle('历史训练迭代：开发集整串精确匹配率', fontsize=17, y=0.98)
    fig.text(0.5, 0.90, '开发指标包含文字与 32 个负样本；不同数据集分别展示', ha='center', color=TEXT)
    for ax, entries, title in ((left, ((v1, BLUE), (v2, TEAL)), '原始开发集 · n=262'),
                               (right, ((v4, ORANGE),), '扩充开发集 · n=299')):
        style_axis(ax)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel('训练轮次（epoch）')
        ax.set_title(title, pad=12, fontsize=12)
        for entry, color in entries:
            history, best = entry['history'], entry['best']
            ratios = [r['dev_exact'] / r['dev_count'] * 100 for r in history]
            label = f"{entry['historical_marker']} · 最佳 {best['exact_percent']:.2f}%"
            ax.plot([r['epoch'] for r in history], ratios, color=color, linewidth=1.8,
                    linestyle='--' if entry['historical_marker'] == 'v1' else '-', label=label)
            ax.scatter([best['epoch']], [best['exact_percent']], color=color, s=32, zorder=4)
        ax.legend(loc='upper left', frameon=False, fontsize=10)
    left.set_xlim(1, 96)
    right.set_xlim(1, 128)
    left.set_ylabel('整串精确匹配率（%）')
    missing.axis('off')
    missing.text(0.5, 0.67, 'v3', ha='center', fontsize=15, color=TEXT)
    missing.text(0.5, 0.46, '训练未完成\n\n仅 1–8 轮日志\n无有效冻结汇总', ha='center', va='center', fontsize=11, color='#596776')
    fig.text(0.075, 0.025, 'v1 → v2：同集提升 74.05 个百分点。v4 为 kaomoji-exp-20261011-004，未启用。', fontsize=10, color=TEXT)
    save(fig, 'training-iterations', output, preview)


def heldout(data: dict, output: Path, preview: Path | None) -> None:
    """Compare recognizers within each held-out dataset using raw CER."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    fig.subplots_adjust(left=0.135, right=0.945, top=0.70, bottom=0.22, wspace=0.6)
    fig.suptitle('冻结保留集：字符错误率 CER（越低越好）', fontsize=17, y=0.98)
    fig.text(0.5, 0.895, '固定基线 vs kaomoji-exp-20261011-004（未启用）；两组冻结数据分别比较', ha='center', color=TEXT)
    historical = data['historical_heldout']['categories']
    original = data['original_dataset']['reports']['test']['categories']
    panels = ((axes[0], historical, ('plain', 'face'), '历史保留测试 · 564 张'),
              (axes[1], original, ('plain', 'face', 'punctuation', 'emoji'), '原创保留测试 · 208 张'))
    names = {'plain': '普通文字', 'face': '颜文字', 'punctuation': '符号', 'emoji': 'Emoji'}
    for ax, categories, kinds, title in panels:
        for field, offset, color in (('baseline', -0.17, BLUE), ('candidate', 0.17, ORANGE)):
            positions = [i + offset for i in range(len(kinds))]
            values = [categories[k][field]['cer'] * 100 for k in kinds]
            ax.barh(positions, values, height=0.30, color=color)
            for y, value in zip(positions, values):
                ax.text(value + 1.3, y, f'{value:.2f}', va='center', fontsize=10, color=TEXT)
        ax.set_yticks(range(len(kinds)))
        ax.set_yticklabels([f"{names[k]}\n(n={categories[k]['baseline']['count']})" for k in kinds], fontsize=10)
        ax.set_ylim(-0.55, len(kinds) - 0.45)
        ax.invert_yaxis()
        ax.set_xlim(0, 115)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.set_xlabel('CER（%）')
        ax.set_title(title, pad=12, fontsize=12)
        style_axis(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color='#e1e7ed', linewidth=0.7)
    fig.legend(handles=[Patch(color=BLUE, label='固定文字基线'),
                        Patch(color=ORANGE, label='实验候选（历史 v4，未启用）')],
               loc='lower center', bbox_to_anchor=(0.5, 0.09), ncol=2, frameon=False, fontsize=11)
    fig.text(0.075, 0.025, '原始 Unicode 与空格参与计分；图中不含负样本。指标限人工裁片识别器，不代表 APP 端到端准确率。', fontsize=10, color=TEXT)
    save(fig, 'heldout-comparison', output, preview)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'docs/evaluation/metrics.json')
    parser.add_argument('--font', type=Path, required=True, help='Locally prepared pinned OFL CJK font')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/evaluation')
    parser.add_argument('--preview', type=Path, help='Optional local raster preview directory')
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    if data['schema_version'] != 'localocr.readme-evaluation.v1':
        raise ValueError('Unknown documentation metric schema')
    if hashlib.sha256(args.font.read_bytes()).hexdigest() != data['plot_font']['sha256']:
        raise ValueError('Font bytes do not match the pinned OFL font')
    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 11, 'text.color': TEXT,
                         'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'localocr-evaluation-20261011'})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
    development(data, args.output, args.preview)
    heldout(data, args.output, args.preview)
    print('Generated anonymous development and held-out figures')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
