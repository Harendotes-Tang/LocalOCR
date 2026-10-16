"""Build the anonymous summary and charts of the browser-font-20261016 round from explicit local reports.

Inputs are completed evaluation reports, dataset metadata, receipts and selections retained under
``artifacts/browser-font-20261016`` (ignored directory) plus the previous round's recorded browser-cards-v1
development reports used as reference. Only aggregate counts, configuration keys, bindings, SHA-256 identities
and system-font family names are copied into the public summary; images, pages, labels, row predictions and
weights are never opened. SVG text is drawn as glyph outlines from the registered OFL CJK font. No model
inference, no network I/O.
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

PUBLIC_REL = Path("docs/evaluation/browser-font-20261016")
ROUND_REL = Path("artifacts/browser-font-20261016")
REFERENCE_REL = Path("artifacts/browser-rendering-20261015")
SUMMARY_SCHEMA = "localocr.browser-font.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.browser-font.assets-build.v1"
MODEL = "ppocrv6-tiny-exp-20261016-001"
PREVIOUS_MODEL = "ppocrv6-tiny-exp-20261013-001"
CLASSES = (("ascii", "半角括号"), ("fullwidth", "全角括号"), ("mixed", "左右宽度不一"), ("none", "其他括号"))
CONDITIONS = (("by_delivery", "png", "PNG 无损"), ("by_delivery", "webp", "WebP 有损"), ("by_device_scale_factor", "1.0", "缩放 100%"),
              ("by_device_scale_factor", "1.25", "缩放 125%"), ("by_device_scale_factor", "1.5", "缩放 150%"))
KINDS = (("face", "颜文字"), ("plain", "普通文字"), ("punctuation", "标点"), ("emoji", "emoji"))
COLORS = ("#64748b", "#93b3cb", "#b45309", "#15803d")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def confined(repo: Path, path: Path) -> Path:
    resolved = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if not any(resolved.is_relative_to((repo / root).resolve()) for root in (ROUND_REL, REFERENCE_REL)):
        raise ValueError("Inputs must stay inside the round or reference root: " + str(path))
    return resolved


def config_keys(binding: dict) -> dict:
    candidate, fusion = binding["config"]["candidate"], binding["config"]["fusion"]
    return {"shape_contrast_threshold": candidate.get("shape_contrast_threshold", candidate["contrast_threshold"]),
            "outer_bracket_width": fusion.get("outer_bracket_width", "exact"), "exterior_bridge_ratio": candidate.get("exterior_bridge_ratio", .45),
            "low_mark_guard": candidate.get("low_mark_guard"), "small_text_shape": candidate.get("small_text_shape")}


def scene_report(value: dict, resources: dict, label: str) -> dict:
    if value.get("split") != "dev" or value.get("promote") is not False or value.get("enabled") is not False or value.get("actual_cpu_inference") is not True:
        raise ValueError("Scene report split/flags differ: " + label)
    summary, binding = value["summary"], value["pipeline_binding"]
    identity = next(iter(summary["models"]))
    model = summary["models"][identity]
    tail = summary.get("tail_ambiguity")
    warm = resources["resources"]["warm"]
    return {"label": label, "model_id": identity, "scenes": summary["scenes"], "dataset_binding": binding["dataset"], "config": config_keys(binding),
            "classifier": (binding["config"].get("classifier") or {}).get("id"),
            "baseline": {"isolated_kaomoji": summary["baseline"]["isolated_kaomoji"]},
            "model": {"isolated_kaomoji": model["isolated_kaomoji"], "body": model["body"], "negative": model["negative"],
                      "routing_effectiveness": model["routing_effectiveness"], "research_quality_gates": model.get("research_quality_gates")},
            "tail_ambiguity": ({"pairs": tail["pairs"], "baseline": tail["baseline"], "model": tail["models"][identity]} if tail else None),
            "candidate_stage": summary.get("candidate_stage"), "breakdowns": summary.get("breakdowns"),
            "resources": {"warm_p50_seconds": warm["p50_seconds"], "warm_p95_seconds": warm["p95_seconds"], "scenes": warm["count"],
                          "peak_private_bytes": int(resources["resources"]["peak_private_bytes"])}}


def probe_report(value: dict, label: str, repo: Path) -> dict:
    summary = value["summary"]
    model = summary["models"][MODEL]
    probe = dict(value.get("probe") or {})
    if probe.get("sidecar"):  # keep the repository-relative identity only; no machine paths in public summaries
        probe["sidecar"] = Path(probe["sidecar"]).resolve().relative_to(repo).as_posix()
    return {"label": label, "independent": value.get("independent"), "probe": probe, "faces": summary["faces"],
            "body_lines": summary["body_lines"], "embedded_faces": summary["embedded_faces"],
            "baseline": {k: summary["baseline"][k] for k in ("face_exact_region", "body_contained", "embedded_face_contained")},
            "model": {k: model[k] for k in ("face_exact_region", "face_wrong_to_exact", "face_exact_to_wrong", "body_contained", "body_lost", "embedded_face_contained")},
            "no_body_lost": model["gates"]["no_body_lost"],
            "width_only": (summary.get("diagnostics") or {}).get("models", {}).get(MODEL, {}).get("face_exact_except_outer_bracket_width"),
            "note": "known-input-001 is a development probe inspected per face, not independent real validation; private screenshots stay local."}


def crop_report(value: dict, label: str) -> dict:
    if value.get("split") != "dev" or value.get("promote") is not False:
        raise ValueError("Crop report split/flags differ: " + label)
    models = {}
    for identity, model in value["models"].items():
        summary = model["summary"]
        models[identity] = {"overall": {k: summary["overall"][k] for k in ("count", "exact", "cer", "ps_cer")},
                            "by_kind": {kind: {k: group.get(k) for k in ("count", "exact", "cer")} for kind, group in summary["by_kind"].items()},
                            "research_quality_gates": model.get("research_quality_gates"), "stage": model.get("stage")}
    return {"label": label, "dataset_binding": value["dataset_binding"], "models": models}


def crop_dataset(meta: dict, generation: dict) -> dict:
    environment = meta["environment"]
    fonts = {family: [{"file": Path(f["file"]).name, "sha256": f["sha256"]} for f in files] for family, files in environment["platform_fonts"].items()}
    return {"generator_id": meta["generator_id"], "seed": meta["seed"], "splits": {s: {k: v[k] for k in ("count", "sha256", "browser_rows", "copied_rows")} for s, v in meta["splits"].items()},
            "summary": meta["summary"], "exclusions": meta["exclusions"], "kind_shares": meta["kind_shares"], "browser_train_share": meta["browser_train_share"],
            "none_share": meta["none_share"], "width_class_weights": meta["width_class_weights"], "device_scales": meta["device_scales"],
            "font_sizes": meta["font_sizes"], "webp": meta["webp"], "font_family": meta["font_family"], "rows_per_page": meta["rows_per_page"],
            "source": {"dataset": meta["source"]["dataset"], "freeze_sha256": meta["source"]["freeze_sha256"]},
            "browser": {k: environment["browser"][k] for k in ("product", "version", "headless", "blocked_requests")},
            "platform_fonts": fonts, "fonts_distributed": environment["fonts_distributed"], "images_distributable": environment["images_distributable"],
            "generation_seconds": generation["elapsed_seconds"], "dataset_sha256": generation["binding"]["dataset_sha256"]}


def breakdown_rates(report: dict, key: str, groups) -> tuple[list[float], list[str]]:
    values, labels = [], []
    for breakdown, name in groups:
        group = (report.get("breakdowns") or {}).get(breakdown, {}).get(name)
        exact = 0 if not group else group["baseline_exact"] if key == "baseline" else group["models"][report["model_id"]]["exact"]
        faces = group["faces"] if group else 0
        values.append(100 * exact / faces if faces else 0.0)
        labels.append(f"{exact}/{faces}")
    return values, labels


def bars(ax, series, names, title, colors=None, ncol=1):
    width = .8 / len(series)
    for si, (label, values, counts) in enumerate(series):
        xs = [i + (si - (len(series) - 1) / 2) * width for i in range(len(names))]
        ax.bar(xs, values, width, color=(colors or COLORS)[si % len(colors or COLORS)], label=label)
        for x, v, n in zip(xs, values, counts):
            ax.text(x, v + .4, n, ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="upper left", ncol=ncol)


def figure_scene(dev: list[dict], public: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    classes = [("by_bracket_width_class", name) for name, _ in CLASSES]
    conditions = [(b, n) for b, n, _ in CONDITIONS]
    for ax, groups, names, title in ((axes[0], classes, [label for _, label in CLASSES], "按真值最外层括号宽度"),
                                     (axes[1], conditions, [label for _, _, label in CONDITIONS], "按交付方式与系统缩放")):
        series = [("生产基线", *breakdown_rates(dev[0], "baseline", groups))] + [(r["label"], *breakdown_rates(r, "model", groups)) for r in dev]
        bars(ax, series, names, title)
    axes[0].set_ylabel("颜文字整串精确（%）")
    top = max(max(breakdown_rates(r, "model", g)[0]) for r in dev for g in (classes, conditions))
    axes[0].set_ylim(0, max(10.0, top * 1.45))
    fig.suptitle("browser-cards-v1 开发集（60 张 / 255 面，浏览器真实字体渲染，非真实截图）· 柱顶为整串正确数/颜文字数", fontsize=11)
    fig.tight_layout()
    return save(fig, "browser-cards-dev-breakdowns", public)


def figure_crops(crops: list[dict], public: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    order = (("production-baseline", "生产基线", COLORS[0]), ("ppocrv6-tiny-reference-20261013-official-pretrained", "官方 Tiny 预训练", "#cbd5e1"),
             (PREVIOUS_MODEL, "上一专项 20261013-001", COLORS[2]), (MODEL, "新专项 20261016-001", COLORS[3]))
    for ax, report, title in zip(axes, crops, ("浏览器裁片 dev（选点依据）", "语法轮 Noto 裁片 dev（仅报告）")):
        series, colors = [], []
        for identity, label, color in order:
            if identity not in report["models"]:
                continue
            kinds = report["models"][identity]["by_kind"]
            series.append((label, [100 * kinds[k]["exact"] / kinds[k]["count"] for k, _ in KINDS], [f"{kinds[k]['exact']}/{kinds[k]['count']}" for k, _ in KINDS]))
            colors.append(color)
        bars(ax, series, [label for _, label in KINDS], title, colors=colors, ncol=2)
    axes[0].set_ylabel("整串精确（%）"); axes[0].set_ylim(0, 140)
    fig.suptitle("专项裁片开发评测 · 柱顶为整串正确数/样本数", fontsize=11)
    fig.tight_layout()
    return save(fig, "crop-dev-by-kind", public)


def save(fig, name: str, public: Path) -> dict:
    path = public / "figures" / (name + ".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    return {"file": path.relative_to(public.parents[2]).as_posix(), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    public = repo / PUBLIC_REL
    registry = read(repo / "artifacts/ocr-data-fonts-v1/fonts.json")
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = repo / "artifacts/ocr-data-fonts-v1" / cjk["file"]
    if cjk["license"] != "OFL-1.1" or digest(font_path) != cjk["sha256"]:
        raise ValueError("Chart font must be the registered OFL CJK font")
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "svg.fonttype": "path", "svg.hashsalt": "localocr-browser-font-20261016",
                         "axes.unicode_minus": False})
    sources = []

    def load(path: Path, role: str) -> dict:
        resolved = confined(repo, path)
        sources.append({"role": role, "file": path.as_posix(), "sha256": digest(resolved)})
        return read(resolved)

    r, ref = ROUND_REL, REFERENCE_REL
    def scene(label, directory):
        return scene_report(load(directory / "evaluation.json", "scene-dev:" + label), load(directory / "resources.json", "scene-resources:" + label), label)
    reference = [scene("候选定位轮配置 t008-exact（上一轮 003）", ref / "evaluation/scene-browser-cards-dev-003"),
                 scene("真实渲染轮冻结配置 t080-baseline（上一轮 004）", ref / "evaluation/scene-browser-cards-dev-004")]
    tiers = [scene("第 1 档 exact+t080+b045（最终）", r / "evaluation/scene-browser-cards-dev-005"),
             scene("第 2 档 +低位守卫", r / "evaluation/scene-browser-cards-dev-003"),
             scene("第 3 档 +小字号形状", r / "evaluation/scene-browser-cards-dev-004")]
    regressions = {"cards-v4 dev": scene("cards-v4 dev", r / "evaluation/scene-cards-v4-dev-001"),
                   "atoms-mixed-v3 dev": scene("atoms-mixed-v3 dev", r / "evaluation/scene-mixed-v3-dev-001"),
                   "known-input-001 probe": probe_report(load(r / "evaluation/known-input-probe-001/evaluation.json", "probe:known-input-001"), "known-input-001 probe", repo)}
    crops = [crop_report(load(r / "evaluation/crop-dev-002/evaluation.json", "crop-dev:browser-crops-v1"), "browser-crops-v1 dev"),
             crop_report(load(r / "evaluation/crop-grammar-dev-001/evaluation.json", "crop-dev:atoms-corpus-v1"), "atoms-corpus-v1 dev (report item)")]
    dataset = crop_dataset(load(r / "data/browser-crops-v1/dataset.json", "crop-dataset"), load(r / "data/browser-crops-v1/generation.json", "crop-generation"))
    budget = load(r / "gpu-budget.json", "gpu-budget")
    be6 = load(r / "validation/be6-receipt.json", "be6-receipt")
    be5 = load(r / "validation/be5-receipt.json", "be5-receipt")
    be4 = load(r / "validation/be4-receipt.json", "be4-receipt")
    int1 = load(r / "validation/int1-receipt.json", "int1-receipt")
    int2 = load(r / "validation/int2-receipt.json", "int2-receipt")
    crop_selection = load(r / "evaluation/crop-selection-001.json", "crop-selection")
    scene_selection = load(r / "evaluation/scene-selection-browser-cards-001.json", "scene-selection")
    figures = [figure_scene([*reference, tiers[0]], public), figure_crops(crops, public)]
    summary = {"schema_version": SUMMARY_SCHEMA, "round": "browser-font-20261016", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Browser real-font renderings (Edge + Windows system fonts) for crops and record-card scenes; not real screenshots; synthetic regressions and a non-independent known-input probe; candidate and classifier stay disabled; dual freeze refused, heldouts sealed.",
               "normalization": "Raw Unicode exact match and codepoint CER; no width, space or combining-mark normalization; width_only is a diagnostic count; tail_ambiguity counts are listed beside the strict fields.",
               "gpu_budget": {"maximum_seconds": budget["maximum_seconds"], "used_seconds": budget["used_seconds"], "runs": [{k: run[k] for k in ("label", "elapsed_seconds", "status")} for run in budget["runs"]],
                              "previous_sha256": budget["round_binding"]["previous_sha256"], "previous_used_seconds": budget["round_binding"]["previous_used_seconds"]},
               "equivalence_probe": {"after_be4": be4["equivalence_probe"]["result"]["mismatched"], "after_revision_six": be4["revision_six"]["equivalence_probe"]["result"]["mismatched"],
                                     "scenes": be4["equivalence_probe"]["result"]["scenes"]},
               "crop_dataset": dataset,
               "specialist": {"id": MODEL, "start": "continuation of " + PREVIOUS_MODEL, "parameters": be6["training"]["parameters"], "steps": be6["training"]["steps"],
                              "selected_step": be6["training"]["selected_training_exposure"]["step"], "initial_dev": be6["training"]["initial_dev"], "best_dev": be6["training"]["best_dev"],
                              "vocabulary_expansion": be4["continuation"]["result"]["added_characters"], "onnx_sha256": be6["export"]["onnx_sha256"],
                              "cpu_parity_max_abs_error": be6["export"]["cpu_parity_max_abs_error"]},
               "classifier": {"id": be5["classifier"]["id"], "threshold": be5["dev"]["threshold"], "dev": be5["dev"], "onnx_sha256": be5["classifier"]["onnx_sha256"]},
               "crop_dev": crops, "crop_selection": {"selected_id": crop_selection["selected_id"], "candidates": [c["id"] for c in crop_selection["candidate_bindings"]]},
               "scene_dev": {"reference_previous_round": reference, "tiers": tiers, "final_config": int1["dev_matrix"]["final_config"],
                             "remaining_body_error": int1["dev_matrix"]["remaining_body_error"]},
               "regressions": regressions, "scene_selection": {"selected_id": scene_selection["selected_id"]},
               "dual_freeze": {k: int1["dual_freeze"][k] for k in ("status", "error", "dual_file_written")},
               "heldouts": {"browser_cards_v1": int2["browser_cards_v1_heldout"], "browser_crops_v1": int2["browser_crops_v1_heldout"],
                            "first_use_receipts": int2["first_use_receipts"], "heldout_reports": int2["heldout_reports"]},
               "figures": figures, "sources": sources, "font": {"id": cjk["id"], "license": cjk["license"], "sha256": cjk["sha256"]},
               "promote": False, "enabled": False, "production_enabled": False}
    write(public / "summary.json", summary)
    receipt = {"schema_version": RECEIPT_SCHEMA, "built_utc": summary["generated_utc"], "summary_sha256": digest(public / "summary.json"),
               "figures": figures, "sources": sources, "script_sha256": digest(Path(__file__)),
               "statement": "Built from explicit completed local reports and receipts; no images, pages, labels, row predictions or weights opened; no inference; no network."}
    write(public / "summary.build-receipt.json", receipt)
    print(json.dumps({"summary": (PUBLIC_REL / "summary.json").as_posix(), "figures": [f["file"] for f in figures]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
