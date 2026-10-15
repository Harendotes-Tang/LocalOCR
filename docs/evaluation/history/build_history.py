"""Build the cross-version comparison of kaomoji recognition results (one figure, three panels).

Panel A compares pipeline versions on the same datasets (development scenes and the known-input real
screenshot set); its counts come from a local counts-only file recounted from each version's own reports.
Panels B and C list every round's own first test (whole-scene heldout and specialist crop heldout) from the
tracked anonymous summaries; those datasets differ per round, so they show baseline-relative results, not a
trend. No images, labels, row predictions, private screenshots or weights are opened; no inference; no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

PUBLIC_REL = Path("docs/evaluation/history")
COUNTS_REL = Path("artifacts/real-rendering-20261015/validation/version-comparison-counts.json")
SUMMARY_SCHEMA = "localocr.version-history.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.version-history.assets-build.v1"
MODEL = "ppocrv6-tiny-exp-20261013-001"
VERSIONS = (
    ("baseline", "生产文字基线", "#64748b"),
    ("grammar-20261013", "语法路由轮 10-13", "#93b3cb"),
    ("candidates-20261013", "候选定位轮 10-13", "#155e9b"),
    ("real-rendering-20261015", "真实渲染轮 10-15（当前）", "#b45309"),
)
DATASET_NAMES = {"atoms-mixed-v1": "原子混合 v1", "cards-v2": "记录卡片 v2", "atoms-mixed-v2": "原子混合 v2", "cards-v3": "记录卡片 v3"}
SCENE_SETS = (
    ("cards-v2", "记录卡片 v2"), ("atoms-mixed-v1", "原子混合 v1"), ("cards-v3", "记录卡片 v3"),
    ("atoms-mixed-v2", "原子混合 v2"), ("cards-v4", "记录卡片 v4"), ("atoms-mixed-v3", "原子混合 v3"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def same_data(counts: dict) -> list[dict]:
    """Rows of panel A: one dataset per row, one entry per version (missing when that version never ran on it)."""
    rows = []
    for key, label in SCENE_SETS:
        versions = counts["scene_dev"].get(key, {})
        if not versions:
            continue
        any_entry = next(iter(versions.values()))
        entry = {"key": key, "label": f"{label} dev", "faces": any_entry["faces"], "kind": "synthetic-dev",
                 "versions": {"baseline": {"exact": any_entry["baseline_exact"], "width_only": any_entry["baseline_exact_except_outer_bracket_width"]}}}
        for version, value in versions.items():
            if value["baseline_exact"] != any_entry["baseline_exact"] or value["faces"] != any_entry["faces"]:
                raise ValueError("Versions disagree on the baseline of one dataset: " + key)
            entry["versions"][version] = {"exact": value["exact"], "width_only": value["exact_except_outer_bracket_width"],
                                          "cer": value["cer"], "research_gate_passed": value["research_gate_passed"]}
        rows.append(entry)
    known = counts["known_input_001"]
    first = next(iter(known.values()))
    entry = {"key": "known-input-001", "label": "已知输入真实截图", "faces": first["faces"], "kind": "real-screens",
             "note": "14 real screenshots, 50 typed standalone faces; inspected per face on 2026-10-15, a development probe from then on",
             "versions": {"baseline": {"exact": first["baseline_exact"], "width_only": None}}}
    for version, value in known.items():
        entry["versions"][version] = {"exact": value["exact"], "width_only": value["exact_except_outer_bracket_width"],
                                      "embedded_contained": value["embedded_contained"], "embedded_faces": value["embedded_faces"],
                                      "body_lost": value["body_lost"]}
    rows.append(entry)
    return rows


def scene_firsts(docs: Path, load) -> list[dict]:
    """Panel B: each round's own whole-scene heldout first test (baseline vs research chain)."""
    region = load(docs / "region-20261012/summary.json", "round:region-20261012")
    quality = load(docs / "quality-optimization-20261012/summary.json", "round:quality-optimization-20261012")
    grammar = load(docs / "kaomoji-grammar-20261013/summary.json", "round:kaomoji-grammar-20261013")
    candidates = load(docs / "region-candidates-20261013/summary.json", "round:region-candidates-20261013")
    rendering = load(docs / "real-rendering-20261015/summary.json", "round:real-rendering-20261015")
    models = {m["id"]: m for m in region["heldout"]["models"]}
    rows = [{"round": "10-12 区域链路", "dataset": "区域整图 280 面", "faces": models["production-baseline"]["face_regions"]["count"],
             "baseline": models["production-baseline"]["face_regions"]["exact"], "chain": models["ppocrv6-tiny-exp-20261011-001"]["face_regions"]["exact"]}]
    scene = quality["scene_cpu_heldout"]
    tiny = next(m for m in scene["models"] if m["id"] == "ppocrv6-tiny-exp-20261012-001")
    rows.append({"round": "10-12 质量优化", "dataset": "质量整图 300 面", "faces": tiny["face"]["count"],
                 "baseline": scene["baseline_face"]["exact"], "chain": tiny["face"]["exact"]})
    for summary, name in ((grammar, "10-13 语法路由"), (candidates, "10-13 候选定位"), (rendering, "10-15 真实渲染")):
        for item in summary["scene"]["heldout"]:
            label = (item.get("label") or item.get("dataset")).replace(" heldout", "")
            model = item["models"][MODEL]["isolated_kaomoji"]
            rows.append({"round": name, "dataset": f"{DATASET_NAMES.get(label, label)} {model['count']} 面", "faces": model["count"],
                         "baseline": item["baseline"]["isolated_kaomoji"]["exact"], "chain": model["exact"],
                         "gate": (item["models"][MODEL].get("research_quality_gates") or {}).get("passed")})
    return rows


def crop_firsts(docs: Path, load) -> list[dict]:
    """Panel C: each round's specialist crop heldout first test (baseline vs fine-tuned specialist)."""
    history = load(docs / "kaomoji-v1-v4-20261011/metrics.json", "round:kaomoji-v1-v4-20261011")
    pretrained = load(docs / "pretrained-20261012/summary.json", "round:pretrained-20261012")
    quality = load(docs / "quality-optimization-20261012/summary.json", "crop:quality-optimization-20261012")
    grammar = load(docs / "kaomoji-grammar-20261013/summary.json", "crop:kaomoji-grammar-20261013")
    face = history["historical_heldout"]["categories"]["face"]
    rows = [{"round": "10-11 v4 专项", "dataset": "历史合成 398 张", "count": face["baseline"]["count"], "baseline": face["baseline"]["exact"], "specialist": face["candidate"]["exact"]}]
    models = {m["id"]: m for m in pretrained["heldout"]["models"]}
    rows.append({"round": "10-12 通用微调", "dataset": "通用合成 800 张", "count": models["production-baseline"]["by_kind"]["face"]["count"],
                 "baseline": models["production-baseline"]["by_kind"]["face"]["exact"], "specialist": models["ppocrv6-tiny-exp-20261011-001"]["by_kind"]["face"]["exact"]})
    crop = {m["id"]: m for m in quality["cpu_crop_heldout"]}
    rows.append({"round": "10-12 质量优化", "dataset": "质量裁片 2400 张", "count": crop["production-baseline"]["face"]["count"],
                 "baseline": crop["production-baseline"]["face"]["exact"], "specialist": crop["ppocrv6-tiny-exp-20261012-001"]["face"]["exact"]})
    heldout = grammar["crop"]["heldout"]["models"]
    rows.append({"round": "10-13 语法路由", "dataset": "原子表裁片 2400 张", "count": heldout["production-baseline"]["by_kind"]["face"]["count"],
                 "baseline": heldout["production-baseline"]["by_kind"]["face"]["exact"], "specialist": heldout[MODEL]["by_kind"]["face"]["exact"]})
    return rows


def pct(value, total):
    return 100.0 * value / total if total else 0.0


def figure(panel_a, panel_b, panel_c, public: Path, preview: Path | None = None) -> dict:
    fig = plt.figure(figsize=(14, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1), width_ratios=(1.9, 1), hspace=.42, wspace=.14)
    ax = fig.add_subplot(grid[0, :])
    width = .19
    for vi, (version, label, color) in enumerate(VERSIONS):
        for di, row in enumerate(panel_a):
            value = row["versions"].get(version)
            x = di + (vi - 1.5) * width
            if value is None:
                continue  # that version never ran on this dataset (stated in the panel title)
            exact = pct(value["exact"], row["faces"])
            ax.bar(x, exact, width, color=color, edgecolor="white", linewidth=.4)
            if value.get("width_only"):
                extra = pct(value["width_only"], row["faces"])
                ax.bar(x, extra, width, bottom=exact, facecolor="white", hatch="///", edgecolor=color, linewidth=.6)
            ax.text(x, exact + .6, f"{value['exact']}", ha="center", va="bottom", fontsize=7)
            if (value.get("width_only") or 0) >= 3:
                ax.text(x, exact + extra + .6, f"+{value['width_only']}", ha="center", va="bottom", fontsize=6.5, color="#475569")
    ax.set_xticks(range(len(panel_a)))
    ax.set_xticklabels([f"{row['label']}\n{row['faces']} 面" + ("\n（真实，非独立）" if row["kind"] == "real-screens" else "") for row in panel_a], fontsize=9)
    ax.set_ylabel("颜文字整串精确率（%）")
    ax.set_title("A · 同一数据上的版本对比（实心柱顶数字为整串正确数；斜线段为“仅外括号宽度不符”；空位表示该版本未在此数据上评测）", fontsize=10.5, loc="left")
    handles = [Patch(color=color, label=label) for _, label, color in VERSIONS] + [Patch(facecolor="white", edgecolor="#475569", hatch="///", label="仅外括号宽度不符")]
    ax.legend(handles=handles, fontsize=8, ncol=5, loc="upper left")
    ax.set_ylim(0, max(45, max(pct((v["exact"] or 0) + (v.get("width_only") or 0), r["faces"]) for r in panel_a for v in r["versions"].values()) * 1.25))
    ax.grid(axis="y", alpha=.3)
    for name, rows, key, title, axis in (("B", panel_b, "chain", "B · 各轮自动整图新保留集首测（数据各不相同，只比各自基线）", fig.add_subplot(grid[1, 0])),
                                         ("C", panel_c, "specialist", "C · 各轮专项模型裁片保留集首测（数据各不相同）", fig.add_subplot(grid[1, 1]))):
        total = "faces" if name == "B" else "count"
        xs = range(len(rows))
        base = [pct(r["baseline"], r[total]) for r in rows]
        mine = [pct(r[key], r[total]) for r in rows]
        axis.bar([x - .2 for x in xs], base, .4, color=VERSIONS[0][2], label="生产文字基线")
        axis.bar([x + .2 for x in xs], mine, .4, color="#155e9b" if name == "B" else "#15803d", label="研究链路" if name == "B" else "专项模型")
        for x, b, m, r in zip(xs, base, mine, rows):
            axis.text(x - .2, b + .8, f"{r['baseline']}", ha="center", fontsize=7)
            axis.text(x + .2, m + .8, f"{r[key]}", ha="center", fontsize=7)
            if r.get("gate") is False:
                axis.text(x + .2, m + .8 + max(mine) * .08, "门禁未过", ha="center", fontsize=7, color="#b91c1c")
        axis.set_xticks(list(xs))
        axis.set_xticklabels([r["round"].replace(" ", "\n", 1) + "\n" + r["dataset"].rsplit(" ", 2)[0] + "\n" + " ".join(r["dataset"].rsplit(" ", 2)[1:])
                              for r in rows], fontsize=7)
        axis.set_ylabel("颜文字整串精确率（%）"); axis.set_title(title, fontsize=10, loc="left"); axis.grid(axis="y", alpha=.3)
        axis.set_ylim(0, max(mine + base + [10]) * 1.3); axis.legend(fontsize=8, loc="upper left")
    fig.suptitle("历次版本颜文字识别结果对比", fontsize=13)
    path = public / "figures" / "version-comparison.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "LocalOCR evaluation"}, bbox_inches="tight")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / "version-comparison.png", dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"file": path.relative_to(public.parents[2]).as_posix(), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--counts", type=Path, default=COUNTS_REL, help="Local counts-only cross-version file (artifacts, ignored)")
    parser.add_argument("--preview", type=Path, help="Optional local raster preview directory (not a public asset)")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    public = repo / PUBLIC_REL
    docs = repo / "docs/evaluation"
    registry = read(repo / "artifacts/ocr-data-fonts-v1/fonts.json")
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = repo / "artifacts/ocr-data-fonts-v1" / cjk["file"]
    if cjk["license"] != "OFL-1.1" or digest(font_path) != cjk["sha256"]:
        raise ValueError("Chart font must be the registered OFL CJK font")
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 10, "svg.fonttype": "path", "svg.hashsalt": "localocr-version-history",
                         "axes.unicode_minus": False, "hatch.linewidth": .6})
    sources = []

    def load(path: Path, role: str) -> dict:
        sources.append({"role": role, "file": path.resolve().relative_to(repo).as_posix(), "sha256": digest(path)})
        return read(path)

    counts_path = (repo / args.counts).resolve()
    counts = load(counts_path, "version-comparison-counts")
    panel_a = same_data(counts)
    panel_b = scene_firsts(docs, load)
    panel_c = crop_firsts(docs, load)
    figures = [figure(panel_a, panel_b, panel_c, public, args.preview)]
    summary = {"schema_version": SUMMARY_SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
               "scope": "Panel A compares versions on identical datasets; panels B and C list each round's own first test on different data and are baseline-relative only. Synthetic data except the known-input set, which became a non-independent development probe on 2026-10-15.",
               "normalization": "Raw Unicode exact match; the width-only count is a diagnostic of faces equal to the truth except for the outermost bracket width, never part of the exact score.",
               "versions": [{"key": key, "label": label} for key, label, _ in VERSIONS],
               "same_data": panel_a, "scene_first_tests": panel_b, "crop_first_tests": panel_c, "figures": figures, "sources": sources,
               "font": {"id": cjk["id"], "license": cjk["license"], "sha256": cjk["sha256"]}, "promote": False, "enabled": False}
    write(public / "summary.json", summary)
    write(public / "summary.build-receipt.json", {"schema_version": RECEIPT_SCHEMA, "built_utc": summary["generated_utc"],
                                                   "summary_sha256": digest(public / "summary.json"), "figures": figures, "sources": sources,
                                                   "script_sha256": digest(Path(__file__)),
                                                   "statement": "Built from tracked anonymous round summaries and a local counts-only recount; no images, labels, row predictions, private screenshots or weights opened; no inference; no network."})
    print(json.dumps({"summary": (PUBLIC_REL / "summary.json").as_posix(), "figures": [f["file"] for f in figures]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
