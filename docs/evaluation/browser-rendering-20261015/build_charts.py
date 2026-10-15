"""Build the anonymous summary and chart of the browser-rendering-20261015 round from explicit local reports.

Inputs are completed evaluation reports, the dataset metadata and the freezes retained under
``artifacts/browser-rendering-20261015`` (ignored directory). Only aggregate counts, configuration keys,
bindings, SHA-256 identities and system-font family names are copied into the public summary; images,
pages, labels, row predictions and weights are never opened. SVG text is drawn as glyph outlines from the
registered OFL CJK font. No model inference, no network I/O.
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

PUBLIC_REL = Path("docs/evaluation/browser-rendering-20261015")
ROUND_REL = Path("artifacts/browser-rendering-20261015")
SUMMARY_SCHEMA = "localocr.browser-rendering.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.browser-rendering.assets-build.v1"
MODEL = "ppocrv6-tiny-exp-20261013-001"
CLASSES = (("ascii", "半角括号"), ("fullwidth", "全角括号"), ("mixed", "左右宽度不一"), ("none", "其他括号"))
COLORS = ("#64748b", "#155e9b", "#b45309")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def confined(repo: Path, path: Path) -> Path:
    resolved = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to((repo / ROUND_REL).resolve()):
        raise ValueError("Inputs must stay inside the round root: " + str(path))
    return resolved


def config_keys(binding: dict) -> dict:
    candidate, fusion = binding["config"]["candidate"], binding["config"]["fusion"]
    return {"shape_contrast_threshold": candidate.get("shape_contrast_threshold", candidate["contrast_threshold"]),
            "outer_bracket_width": fusion.get("outer_bracket_width", "exact"),
            "exterior_bridge_ratio": candidate.get("exterior_bridge_ratio", .45)}


def scene_report(value: dict, resources: dict, split: str, label: str) -> dict:
    if value.get("split") != split or value.get("promote") is not False or value.get("enabled") is not False or value.get("actual_cpu_inference") is not True:
        raise ValueError("Scene report split/flags differ: " + label)
    if split == "heldout" and value.get("independent_heldout") is not True:
        raise ValueError("Heldout report lacks the first-use claim: " + label)
    summary, binding = value["summary"], value["pipeline_binding"]
    model = summary["models"][MODEL]
    warm = resources["resources"]["warm"]
    return {"label": label, "split": split, "scenes": summary["scenes"], "scene_kinds": summary.get("scene_kinds"),
            "dataset_binding": binding["dataset"], "config": config_keys(binding), "config_views": binding["config"]["views"],
            "classifier": (binding["config"].get("classifier") or {}).get("id"),
            "baseline": {"isolated_kaomoji": summary["baseline"]["isolated_kaomoji"]},
            "model": {"id": MODEL, "isolated_kaomoji": model["isolated_kaomoji"], "body": model["body"], "negative": model["negative"],
                      "routing_effectiveness": model["routing_effectiveness"], "research_quality_gates": model.get("research_quality_gates")},
            "candidate_stage": summary.get("candidate_stage"), "breakdowns": summary.get("breakdowns"),
            "resources": {"warm_p50_seconds": warm["p50_seconds"], "warm_p95_seconds": warm["p95_seconds"], "scenes": warm["count"],
                          "peak_private_bytes": int(resources["resources"]["peak_private_bytes"])}}


def dataset_summary(meta: dict, generation: dict) -> dict:
    environment = meta["environment"]
    fonts = {family: [{"file": Path(f["file"]).name, "sha256": f["sha256"]} for f in files] for family, files in environment["platform_fonts"].items()}
    return {"generator_id": meta["generator_id"], "seed": meta["seed"], "splits": {s: {"count": v["count"], "sha256": v["sha256"]} for s, v in meta["splits"].items()},
            "summary": meta["summary"], "exclusions": meta["exclusions"], "width_class_weights": meta["width_class_weights"],
            "device_scales": meta["device_scales"], "font_sizes": meta["font_sizes"], "webp": meta["webp"], "font_family": meta["font_family"],
            "browser": {k: environment["browser"][k] for k in ("product", "version", "headless", "blocked_requests")},
            "driver": environment["driver"], "windows": environment["windows"], "platform_fonts": fonts,
            "fonts_distributed": environment["fonts_distributed"], "images_distributable": environment["images_distributable"],
            "generation_seconds": generation["elapsed_seconds"], "dataset_sha256": generation["binding"]["dataset_sha256"]}


CONDITIONS = (("by_delivery", "png", "PNG 无损"), ("by_delivery", "webp", "WebP 有损"), ("by_device_scale_factor", "1.0", "缩放 100%"),
              ("by_device_scale_factor", "1.25", "缩放 125%"), ("by_device_scale_factor", "1.5", "缩放 150%"))


def rates(report: dict, key: str, groups: list[tuple[str, str]]) -> tuple[list[float], list[str]]:
    values, labels = [], []
    for breakdown, name in groups:
        group = (report.get("breakdowns") or {}).get(breakdown, {}).get(name)
        exact = 0 if not group else group["baseline_exact"] if key == "baseline" else group["models"][MODEL]["exact"]
        faces = group["faces"] if group else 0
        values.append(100 * exact / faces if faces else 0.0)
        labels.append(f"{exact}/{faces}")
    return values, labels


def panel(ax, series, groups, names, title):
    width = .8 / len(series)
    for si, (label, report, key) in enumerate(series):
        values, counts = rates(report, key, groups)
        xs = [i + (si - (len(series) - 1) / 2) * width for i in range(len(groups))]
        ax.bar(xs, values, width, color=COLORS[si % len(COLORS)], label=label)
        for x, v, n in zip(xs, values, counts):
            ax.text(x, v + .3, n, ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="upper right")


def figure_classes(dev: list[dict], heldout: list[dict], public: Path) -> dict:
    """Dev: by truth bracket width and by delivery/scale; a heldout first test adds a third panel."""
    count = 3 if heldout else 2
    fig, axes = plt.subplots(1, count, figsize=(5.2 * count, 4.2), sharey=True)
    series = [("生产基线", dev[0], "baseline")] + [(r["label"], r, "model") for r in dev]
    classes = [("by_bracket_width_class", name) for name, _ in CLASSES]
    panel(axes[0], series, classes, [label for _, label in CLASSES], "开发集 · 按真值最外层括号宽度")
    panel(axes[1], series, [(b, n) for b, n, _ in CONDITIONS], [label for _, _, label in CONDITIONS], "开发集 · 按交付方式与系统缩放")
    if heldout:
        held = [("生产基线", heldout[0], "baseline")] + [(r["label"], r, "model") for r in heldout]
        panel(axes[2], held, classes, [label for _, label in CLASSES], "保留集首测 · 按括号宽度")
    axes[0].set_ylabel("颜文字整串精确（%）")
    top = max(max(rates(r, k, g)[0]) for _, r, k in series for g in (classes, [(b, n) for b, n, _ in CONDITIONS]))
    axes[0].set_ylim(0, max(5.0, top * 1.35))
    fig.suptitle("浏览器真实字体渲染集（Edge + Windows 系统字体）· 柱顶为整串正确数/颜文字数", fontsize=11); fig.tight_layout()
    return save(fig, "bracket-width-class-exact", public)


def save(fig, name: str, public: Path) -> dict:
    path = public / "figures" / (name + ".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    return {"file": str(path.relative_to(public.parents[2])).replace("\\", "/"), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dev-report", action="append", default=[], help="label=report directory (dev split)")
    parser.add_argument("--heldout-report", action="append", default=[], help="label=report directory (heldout first use)")
    parser.add_argument("--dual-freeze", type=Path, action="append", default=[])
    parser.add_argument("--gpu-budget", type=Path, required=True)
    parser.add_argument("--equivalence-log", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    public = repo / PUBLIC_REL
    registry = read(repo / "artifacts/ocr-data-fonts-v1/fonts.json")
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = repo / "artifacts/ocr-data-fonts-v1" / cjk["file"]
    if cjk["license"] != "OFL-1.1" or digest(font_path) != cjk["sha256"]:
        raise ValueError("Chart font must be the registered OFL CJK font")
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "svg.fonttype": "path", "svg.hashsalt": "localocr-browser-rendering-20261015",
                         "axes.unicode_minus": False})
    sources = []

    def load(path: Path, role: str) -> dict:
        resolved = confined(repo, path)
        sources.append({"role": role, "file": str(path).replace("\\", "/"), "sha256": digest(resolved)})
        return read(resolved)

    reports = {"dev": [], "heldout": []}
    for split, items in (("dev", args.dev_report), ("heldout", args.heldout_report)):
        for item in items:
            label, path = item.split("=", 1)
            report = load(Path(path) / "evaluation.json", f"{split}:{label}")
            resources = load(Path(path) / "resources.json", f"{split}-resources:{label}")
            reports[split].append(scene_report(report, resources, split, label))
    dataset = dataset_summary(load(args.dataset / "dataset.json", "dataset"), load(args.dataset / "generation.json", "generation"))
    duals = [{"file": str(path).replace("\\", "/"), "sha256": digest(confined(repo, path)), "frozen_utc": read(confined(repo, path)).get("frozen_utc")}
             for path in args.dual_freeze]
    budget = load(args.gpu_budget, "gpu-budget")
    equivalence = confined(repo, args.equivalence_log)
    sources.append({"role": "equivalence-probe-log", "file": str(args.equivalence_log).replace("\\", "/"), "sha256": digest(equivalence)})
    probe = json.loads(next(line for line in equivalence.read_text(encoding="utf-8").splitlines() if line.startswith("{")))
    figures = [figure_classes(reports["dev"], reports["heldout"], public)]
    summary = {"schema_version": SUMMARY_SCHEMA, "round": "browser-rendering-20261015", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Browser real-font renderings of original CC0 record-card pages (Edge + Windows system fonts); not real screenshots; no accuracy claim beyond the listed sets; candidate and classifier stay disabled.",
               "normalization": "Raw Unicode exact match and codepoint CER; no width, space or combining-mark normalization; width_only is a diagnostic count.",
               "gpu_budget": {"maximum_seconds": budget["maximum_seconds"], "used_seconds": budget["used_seconds"], "runs": len(budget.get("runs", [])),
                              "previous_sha256": budget["round_binding"]["previous_sha256"], "previous_used_seconds": budget["round_binding"]["previous_used_seconds"]},
               "equivalence_probe": {"scenes": probe["scenes"], "mismatched": probe["mismatched"]},
               "dataset": dataset, "scene": reports, "dual_freezes": duals, "figures": figures, "sources": sources,
               "font": {"id": cjk["id"], "license": cjk["license"], "sha256": cjk["sha256"]},
               "promote": False, "enabled": False, "production_enabled": False}
    write(public / "summary.json", summary)
    receipt = {"schema_version": RECEIPT_SCHEMA, "built_utc": summary["generated_utc"], "summary_sha256": digest(public / "summary.json"),
               "figures": figures, "sources": sources, "script_sha256": digest(Path(__file__)),
               "statement": "Built from explicit completed local reports; no images, pages, labels, row predictions or weights opened; no inference; no network."}
    write(public / "summary.build-receipt.json", receipt)
    print(json.dumps({"summary": str(PUBLIC_REL / "summary.json"), "figures": [f["file"] for f in figures]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
