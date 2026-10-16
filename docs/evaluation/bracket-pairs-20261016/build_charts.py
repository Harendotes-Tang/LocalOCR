"""Build the anonymous summary and charts of the bracket-pairs-20261016 round from explicit local reports.

Inputs are completed evaluation reports, receipts, selections and the dual freeze retained under
``artifacts/bracket-pairs-20261016`` (ignored directory), plus the browser-font round's recorded development
reports used as the starting point. Only aggregate counts, configuration keys and SHA-256 identities are copied
into the public summary; images, pages, labels, row predictions, private screenshots and weights are never opened.
SVG text is drawn as glyph outlines from the registered OFL CJK font. No model inference, no network I/O.
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

PUBLIC_REL = Path("docs/evaluation/bracket-pairs-20261016")
ROUND_REL = Path("artifacts/bracket-pairs-20261016")
REFERENCE_REL = Path("artifacts/browser-font-20261016")
SUMMARY_SCHEMA = "localocr.bracket-pairs.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.bracket-pairs.assets-build.v1"
MODEL = "ppocrv6-tiny-exp-20261016-001"
CLASSES = (("ascii", "半角括号"), ("fullwidth", "全角括号"), ("mixed", "左右宽度不一"), ("none", "其他括号"))
CONDITIONS = (("by_delivery", "png", "PNG 无损"), ("by_delivery", "webp", "WebP 有损"), ("by_device_scale_factor", "1.0", "缩放 100%"),
              ("by_device_scale_factor", "1.25", "缩放 125%"), ("by_device_scale_factor", "1.5", "缩放 150%"))
STAGES = (("missed", "漏检"), ("partial", "部分候选"), ("complete-unsafe", "完整但不安全"), ("safe-not-routed", "安全但未路由"),
          ("routed-wrong", "路由后读错"), ("routed-exact", "路由后读对"))
STAGE_COLORS = ("#cbd5e1", "#94a3b8", "#f59e0b", "#93b3cb", "#b91c1c", "#15803d")
COLORS = ("#64748b", "#15803d", "#155e9b", "#7c3aed")
BODY = ("baseline_correct_to_wrong", "erroneous_rewrites", "new_body_intersection_errors", "body_literal_deleted")


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
            "pair_selection": candidate.get("pair_selection"), "dense_guard": candidate.get("dense_guard"),
            "small_text_shape": candidate.get("small_text_shape"), "low_mark_guard": candidate.get("low_mark_guard")}


def scene_report(value: dict, resources: dict, label: str, split: str = "dev") -> dict:
    if value.get("split") != split or value.get("promote") is not False or value.get("enabled") is not False or value.get("actual_cpu_inference") is not True:
        raise ValueError("Scene report split/flags differ: " + label)
    summary, binding = value["summary"], value["pipeline_binding"]
    identity = next(iter(summary["models"]))
    model = summary["models"][identity]
    tail = summary.get("tail_ambiguity")
    warm = resources["resources"]["warm"]
    return {"label": label, "split": split, "model_id": identity, "scenes": summary["scenes"], "dataset_binding": binding["dataset"],
            "config": config_keys(binding), "classifier": (binding["config"].get("classifier") or {}).get("id"),
            "baseline": {"isolated_kaomoji": summary["baseline"]["isolated_kaomoji"]},
            "model": {"isolated_kaomoji": model["isolated_kaomoji"], "body": model["body"], "negative": model["negative"],
                      "routing_effectiveness": model["routing_effectiveness"], "research_quality_gates": model.get("research_quality_gates")},
            "tail_ambiguity": ({"pairs": tail["pairs"], "baseline": tail["baseline"], "model": tail["models"][identity]} if tail else None),
            "candidate_stage": summary.get("candidate_stage"), "breakdowns": summary.get("breakdowns"),
            "resources": {"warm_p50_seconds": warm["p50_seconds"], "warm_p95_seconds": warm["p95_seconds"], "scenes": warm["count"],
                          "peak_private_bytes": int(resources["resources"]["peak_private_bytes"])}}


def probe_report(value: dict, label: str) -> dict:
    summary = value["summary"]
    model = summary["models"][MODEL]
    return {"label": label, "independent": value.get("independent"), "faces": summary["faces"], "body_lines": summary["body_lines"],
            "embedded_faces": summary["embedded_faces"], "predictions_private_sha256": value.get("predictions_private_sha256"),
            "baseline": {k: summary["baseline"][k] for k in ("face_exact_region", "body_contained", "embedded_face_contained")},
            "model": {k: model[k] for k in ("face_exact_region", "face_wrong_to_exact", "face_exact_to_wrong", "body_contained", "body_lost", "embedded_face_contained")},
            "no_body_lost": model["gates"]["no_body_lost"],
            "width_only": (summary.get("diagnostics") or {}).get("models", {}).get(MODEL, {}).get("face_exact_except_outer_bracket_width"),
            "note": "known-input-001 is a development probe inspected per face, not independent real validation; private screenshots stay local."}


def crop_report(value: dict, label: str, split: str) -> dict:
    if value.get("split") != split or value.get("promote") is not False:
        raise ValueError("Crop report split/flags differ: " + label)
    models = {}
    for identity, model in value["models"].items():
        summary = model["summary"]
        moves = (model.get("transitions_from_production") or {}).get("by_kind", {})
        models[identity] = {"overall": {k: summary["overall"][k] for k in ("count", "exact", "cer", "ps_cer")},
                            "by_kind": {kind: {k: group.get(k) for k in ("count", "exact", "cer")} for kind, group in summary["by_kind"].items()},
                            "transitions_from_production": {kind: {k: t.get(k) for k in ("baseline_correct_to_wrong", "erroneous_rewrites", "new_negative_false_outputs")}
                                                            for kind, t in moves.items()} or None,
                            "research_quality_gates": model.get("research_quality_gates"), "stage": model.get("stage")}
    return {"label": label, "split": split, "dataset_binding": value["dataset_binding"], "models": models}


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


def figure_breakdowns(reports: list[dict], name: str, title: str, public: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    classes = [("by_bracket_width_class", key) for key, _ in CLASSES]
    conditions = [(b, n) for b, n, _ in CONDITIONS]
    for ax, groups, names, panel in ((axes[0], classes, [label for _, label in CLASSES], "按真值最外层括号宽度"),
                                     (axes[1], conditions, [label for _, _, label in CONDITIONS], "按交付方式与系统缩放")):
        series = [("生产基线", *breakdown_rates(reports[0], "baseline", groups))] + [(r["label"], *breakdown_rates(r, "model", groups)) for r in reports]
        bars(ax, series, names, panel)
    axes[0].set_ylabel("颜文字整串精确（%）")
    top = max(max(breakdown_rates(r, "model", g)[0]) for r in reports for g in (classes, conditions))
    axes[0].set_ylim(0, max(10.0, top * 1.45))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save(fig, name, public)


def figure_stages(before: dict, after: dict, public: Path) -> dict:
    scales = ("1.0", "1.25", "1.5")
    fig, ax = plt.subplots(figsize=(12, 4.4))
    positions, labels = [], []
    for si, scale in enumerate(scales):
        for vi, (diag, label) in enumerate(((before, "上一轮最终"), (after, "本轮冻结"))):
            x = si * 2.6 + vi
            row = diag["stages_by_device_scale"][scale]
            bottom = 0.0
            for (stage, _), color in zip(STAGES, STAGE_COLORS):
                value = 100 * row.get(stage, 0) / row["faces"]
                ax.bar(x, value, .8, bottom=bottom, color=color, edgecolor="white", linewidth=.4)
                if row.get(stage, 0):
                    ax.text(x, bottom + value / 2, str(row[stage]), ha="center", va="center", fontsize=7)
                bottom += value
            positions.append(x)
            labels.append(f"缩放 {int(float(scale) * 100)}%\n{label}\n（{row['faces']} 面，整串对 {row.get('exact', 0)}）")
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("颜文字占比（%）"); ax.set_ylim(0, 100)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS], labels=[label for _, label in STAGES], fontsize=8, ncol=6,
              loc="upper center", bbox_to_anchor=(.5, -.28))
    ax.set_title("browser-cards-v1 开发集各缩放档的候选阶段分解（只作诊断，不回流门禁）", fontsize=10)
    fig.tight_layout()
    return save(fig, "browser-cards-dev-stages", public)


def figure_heldout(scene: dict, crop: dict, widths: dict, public: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": (1.6, 1, 1)})
    conditions = [(b, n) for b, n, _ in CONDITIONS]
    series = [("生产基线", *breakdown_rates(scene, "baseline", conditions)), ("本轮冻结配置", *breakdown_rates(scene, "model", conditions))]
    bars(axes[0], series, [label for _, _, label in CONDITIONS], "browser-cards-v1 保留集（120 张 / 481 面）", colors=(COLORS[0], COLORS[3]))
    axes[0].set_ylabel("颜文字整串精确（%）"); axes[0].set_ylim(0, 45)
    for ax, key, title in ((axes[1], "browser_cards_v1_heldout", "卡片保留集 · 宽度同构分项"), (axes[2], "browser_crops_v1_heldout", "裁片保留集 · 宽度同构分项")):
        groups = widths[key]
        names = ("sibling", "non_sibling")
        base = [100 * groups[n]["production-baseline_exact"] / groups[n]["faces"] for n in names]
        mine = [100 * groups[n][MODEL + "_exact"] / groups[n]["faces"] for n in names]
        bars(ax, [("生产基线", base, [f"{groups[n]['production-baseline_exact']}/{groups[n]['faces']}" for n in names]),
                  ("专项 20261016-001", mine, [f"{groups[n][MODEL + '_exact']}/{groups[n]['faces']}" for n in names])],
             ["宽度同构", "非同构"], title, colors=(COLORS[0], COLORS[3]))
        ax.set_ylim(0, 125)
    fig.suptitle("保留集首测（双冻结之后各一次；浏览器真实字体渲染，非真实截图）· 柱顶为整串正确数/颜文字数", fontsize=11)
    fig.tight_layout()
    return save(fig, "heldout-first-use", public)


PREVIEW: list[Path] = []  # optional local raster preview directory (not a public asset)


def save(fig, name: str, public: Path) -> dict:
    path = public / "figures" / (name + ".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    for preview in PREVIEW:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / (name + ".png"), dpi=100, facecolor="white")
    plt.close(fig)
    return {"file": path.relative_to(public.parents[2]).as_posix(), "sha256": digest(path)}


def strip_paths(value):
    """Drop machine paths from copied receipt fragments; SHA identities stay."""
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items() if k not in ("file", "sidecar", "candidate_file", "manifest_file")}
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preview", type=Path, help="Optional local raster preview directory (not a public asset)")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if args.preview is not None:
        PREVIEW.append(args.preview.resolve())
    public = repo / PUBLIC_REL
    registry = read(repo / "artifacts/ocr-data-fonts-v1/fonts.json")
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = repo / "artifacts/ocr-data-fonts-v1" / cjk["file"]
    if cjk["license"] != "OFL-1.1" or digest(font_path) != cjk["sha256"]:
        raise ValueError("Chart font must be the registered OFL CJK font")
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "svg.fonttype": "path", "svg.hashsalt": "localocr-bracket-pairs-20261016",
                         "axes.unicode_minus": False})
    sources = []

    def load(path: Path, role: str) -> dict:
        resolved = confined(repo, path)
        sources.append({"role": role, "file": path.as_posix(), "sha256": digest(resolved)})
        return read(resolved)

    r, ref = ROUND_REL, REFERENCE_REL

    def scene(label, directory, split="dev"):
        return scene_report(load(directory / "evaluation.json", f"scene-{split}:" + label), load(directory / "resources.json", f"scene-resources:{label}"), label, split)

    start = scene("上一轮最终配置（browser-font dev-005）", ref / "evaluation/scene-browser-cards-dev-005")
    tiers = [scene("第 1 档 +pair_selection", r / "evaluation/scene-browser-cards-dev-001"),
             scene("第 2 档 +dense_guard 1.8（冻结）", r / "evaluation/scene-browser-cards-dev-002"),
             scene("第 3 档 +small_text_shape 22/2", r / "evaluation/scene-browser-cards-dev-003")]
    regressions = {"cards-v4 dev": scene("cards-v4 dev", r / "evaluation/scene-cards-v4-dev-002"),
                   "atoms-mixed-v3 dev": scene("atoms-mixed-v3 dev", r / "evaluation/scene-mixed-v3-dev-002"),
                   "known-input-001 probe": probe_report(load(r / "evaluation/known-input-probe-002/evaluation.json", "probe:known-input-001"), "known-input-001 probe")}
    previous_regressions = {"cards-v4 dev": scene("cards-v4 dev（上一轮）", ref / "evaluation/scene-cards-v4-dev-001"),
                            "atoms-mixed-v3 dev": scene("atoms-mixed-v3 dev（上一轮）", ref / "evaluation/scene-mixed-v3-dev-001"),
                            "known-input-001 probe": probe_report(load(ref / "evaluation/known-input-probe-001/evaluation.json", "probe:known-input-001-previous"), "known-input-001 probe（上一轮）")}
    crop_dev = crop_report(load(r / "evaluation/crop-dev-001/evaluation.json", "crop-dev:browser-crops-v1"), "browser-crops-v1 dev（本轮重跑）", "dev")
    crop_heldout = crop_report(load(r / "evaluation/crop-heldout-001/evaluation.json", "crop-heldout:browser-crops-v1"), "browser-crops-v1 heldout（首测）", "heldout")
    scene_heldout = scene("browser-cards-v1 保留集（首测）", r / "evaluation/scene-browser-cards-heldout-001", "heldout")
    budget = load(r / "gpu-budget.json", "gpu-budget")
    be2 = load(r / "validation/be2-receipt.json", "be2-receipt")
    be4 = load(r / "validation/be4-receipt.json", "be4-receipt")
    be5 = load(r / "validation/be5-receipt.json", "be5-receipt")
    int1 = load(r / "validation/int1-receipt.json", "int1-receipt")
    int2 = load(r / "validation/int2-receipt.json", "int2-receipt")
    int3 = load(r / "validation/int3-receipt.json", "int3-receipt")
    widths = load(r / "validation/width-siblings.json", "width-siblings")
    diagnostics = {tag: load(r / f"validation/pair-diagnostics-{tag}.json", "pair-diagnostics:" + tag) for tag in ("bf-dev-005", "dev-001", "dev-002", "dev-003")}
    figures = [figure_breakdowns([start, tiers[1]], "browser-cards-dev-breakdowns",
                                 "browser-cards-v1 开发集（60 张 / 255 面，浏览器真实字体渲染，非真实截图）· 柱顶为整串正确数/颜文字数", public),
               figure_stages(diagnostics["bf-dev-005"], diagnostics["dev-002"], public),
               figure_heldout(scene_heldout, crop_heldout, widths, public)]
    summary = {"schema_version": SUMMARY_SCHEMA, "round": "bracket-pairs-20261016", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Browser real-font renderings (Edge + Windows system fonts) for crops and record-card scenes; not real screenshots; synthetic regressions and a non-independent known-input probe; no training; specialist, classifier and configuration stay disabled.",
               "normalization": "Raw Unicode exact match and codepoint CER; no width, space or combining-mark normalization; width_only is a diagnostic count; tail_ambiguity counts are listed beside the strict fields.",
               "gpu_budget": {"maximum_seconds": budget["maximum_seconds"], "used_seconds": budget["used_seconds"], "runs": len(budget["runs"]),
                              "previous_sha256": budget["round_binding"]["previous_sha256"], "previous_used_seconds": budget["round_binding"]["previous_used_seconds"]},
               "keys": {"pair_selection": be2["code"]["keys"]["pair_selection"], "dense_guard": be2["code"]["keys"]["dense_guard"],
                        "pair_family": be2["code"]["pair_family"], "filter_order": be2["code"]["filter_order"]},
               "dense_guard_measurement": {k: be2["dense_guard"][k] for k in ("maximum_width_ratio", "upper_bound", "noto", "edge", "edge_by_polarity")},
               "atom_projection": {k: be2["projection"][k] for k in ("shape_threshold", "declared_pairs", "noto_unsupported_pairs", "noto_cross_family",
                                                                     "edge_samples", "edge_cross_family_by_scale")},
               "dense_guard_risk": be2["risk"],
               "tests_and_review": {"full_suite": [t["summary"] for t in be4["full_tests"]], "review_verdict": be4["review"]["verdict"]},
               "equivalence_probe": {k: be5["equivalence_probe"][k] for k in ("reports", "scenes", "mismatched", "per_report")},
               "source_inventory": {k: be5["source_inventory"][k] for k in ("files", "object_sha256", "sha256")},
               "scene_dev": {"start": start, "tiers": tiers, "decisions": int1["matrix"]["decisions"], "final": strip_paths(int1["matrix"]["final"]),
                             "dev_0040": int1["dev_0040"]},
               "diagnostics": {tag: {k: value[k] for k in ("pairs", "stages_by_device_scale", "face_crops_by_device_scale", "body_inside_threshold_pixels")}
                               for tag, value in diagnostics.items()},
               "regressions": {"current": regressions, "previous_round": previous_regressions, "hard_conditions_passed": int1["regressions"]["hard_conditions_passed"]},
               "crop_dev": crop_dev,
               "freezes": {"crop_selected_id": int2["crop_selection"]["selected_id"], "crop_budget_receipt_sha256": int2["crop_selection"]["budget_receipt"]["sha256"],
                           "scene_selected_id": int2["scene_selection"]["selected_id"], "scene_config_sha256": int2["scene_selection"]["config_sha256"],
                           "dual_freeze_sha256": int2["dual_freeze"]["sha256"], "dual_gates": int2["dual_freeze"]["gates"],
                           "regression_proofs": strip_paths(int2["dual_freeze"]["regressions"]),
                           "source_inventory_equals_be5_snapshot": int2["dual_freeze"]["source_inventory_equals_be5_snapshot"]},
               "heldouts": {"browser_crops_v1": crop_heldout, "browser_cards_v1": scene_heldout,
                            "claims_sha256": {"browser_crops_v1": int3["browser_crops_v1_heldout"]["claim"]["sha256"],
                                              "browser_cards_v1": int3["browser_cards_v1_heldout"]["claim"]["sha256"]},
                            "width_siblings": {k: widths[k] for k in ("rule", "lineage_train_core_keys", "browser_crops_v1_heldout", "browser_cards_v1_heldout")}},
               "figures": figures, "sources": sources, "font": {"id": cjk["id"], "license": cjk["license"], "sha256": cjk["sha256"]},
               "promote": False, "enabled": False, "production_enabled": False}
    write(public / "summary.json", summary)
    receipt = {"schema_version": RECEIPT_SCHEMA, "built_utc": summary["generated_utc"], "summary_sha256": digest(public / "summary.json"),
               "figures": figures, "sources": sources, "script_sha256": digest(Path(__file__)),
               "statement": "Built from explicit completed local reports and receipts; no images, pages, labels, row predictions, private screenshots or weights opened; no inference; no network."}
    write(public / "summary.build-receipt.json", receipt)
    print(json.dumps({"summary": (PUBLIC_REL / "summary.json").as_posix(), "figures": [f["file"] for f in figures]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
