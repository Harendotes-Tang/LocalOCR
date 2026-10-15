"""Build the anonymous summary and charts of the real-rendering-20261015 round from explicit local reports.

Inputs are completed evaluation reports retained under ``artifacts/real-rendering-20261015`` and the
comparison baselines of the previous round under ``artifacts/region-candidates-20261013`` (ignored
directories). Only aggregate counts, configuration keys, dataset/model bindings and SHA-256 identities are
copied into the public summary; images, labels, row predictions, private screenshots and weights are
never opened. SVG text is rendered as glyph outlines from a locally obtained OFL font. No model
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

PUBLIC_REL = Path("docs/evaluation/real-rendering-20261015")
ROUND_REL = Path("artifacts/real-rendering-20261015")
PREVIOUS_REL = Path("artifacts/region-candidates-20261013")
SUMMARY_SCHEMA = "localocr.real-rendering.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.real-rendering.assets-build.v1"
MODEL = "ppocrv6-tiny-exp-20261013-001"
MODEL_LABELS = {"production-baseline": "生产文字基线", MODEL: "Tiny · 冻结专项（exp-20261013-001）"}
COLORS = ("#64748b", "#93b3cb", "#155e9b", "#b45309", "#15803d", "#7c3aed")
MODE_LABELS = {"exact": "逐字节一致", "baseline-else-raw": "宽度取基线", "raw": "宽度取原始视图"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def confined(repo: Path, path: Path) -> Path:
    resolved = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    roots = [(repo / ROUND_REL).resolve(), (repo / PREVIOUS_REL).resolve()]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Reports must stay inside the round roots: " + str(path))
    return resolved


def config_keys(binding: dict) -> dict:
    candidate, fusion = binding["config"]["candidate"], binding["config"]["fusion"]
    return {"shape_contrast_threshold": candidate.get("shape_contrast_threshold", candidate["contrast_threshold"]),
            "outer_bracket_width": fusion.get("outer_bracket_width", "exact"),
            "exterior_bridge_ratio": candidate.get("exterior_bridge_ratio", .45)}


def scene_report(value: dict, split: str, label: str) -> dict:
    if value.get("split") != split or value.get("promote") is not False or value.get("enabled") is not False or value.get("actual_cpu_inference") is not True:
        raise ValueError("Scene report split/flags differ: " + label)
    if split == "heldout" and value.get("independent_heldout") is not True:
        raise ValueError("Scene heldout report lacks the first-use claim: " + label)
    summary = value["summary"]
    models = {}
    for identity, model in summary["models"].items():
        models[identity] = {"label": MODEL_LABELS.get(identity, identity), "isolated_kaomoji": model["isolated_kaomoji"],
                            "body": model["body"], "negative": model["negative"], "routing_effectiveness": model["routing_effectiveness"],
                            "research_quality_gates": model.get("research_quality_gates")}
    binding = value["pipeline_binding"]
    localization = summary["localization"]
    return {"label": label, "split": split, "scenes": summary["scenes"], "scene_kinds": summary.get("scene_kinds"),
            "dataset_binding": binding["dataset"], "config_views": binding["config"]["views"], "config": config_keys(binding),
            "classifier": (binding["config"].get("classifier") or {}).get("id"),
            "baseline": {"isolated_kaomoji": summary["baseline"]["isolated_kaomoji"]}, "models": models,
            "localization": localization, "candidate_stage": summary.get("candidate_stage"),
            "candidate_sources": summary.get("candidate_sources"),
            "resources": {key: value["resources"].get(key) for key in ("load_seconds", "first_scene_seconds", "warm", "total_warm_seconds",
                                                                        "peak_rss_bytes", "intra_op_threads", "inter_op_threads")}}


def relativize(value, repo: Path):
    """Repo-relative, forward-slash copies of any absolute local paths inside a receipt reference."""
    if isinstance(value, dict):
        return {k: relativize(v, repo) for k, v in value.items()}
    if isinstance(value, str) and (value.startswith(str(repo)) or value.replace("/", "\\").startswith(str(repo))):
        return str(Path(value).resolve().relative_to(repo)).replace("\\", "/")
    return value


def real_report(value: dict, label: str, repo: Path | None = None) -> dict:
    summary = value["summary"]
    repo = repo or Path.cwd().resolve()
    return {"label": label, "independent": value.get("independent"), "screens": value["manifest"]["screens"],
            "environment": value["manifest"].get("environment"), "label_origin": value["manifest"].get("label_origin"),
            "typed_differences": value["manifest"].get("typed_differences"), "config_sha256": value.get("config_sha256"),
            "faces": summary["faces"], "faces_ambiguous_excluded": summary["faces_ambiguous_excluded"], "embedded_faces": summary.get("embedded_faces"),
            "body_lines": summary["body_lines"], "baseline": summary["baseline"], "models": summary["models"],
            "diagnostics": summary.get("diagnostics"), "timing": value.get("timing"), "scope": value.get("scope"),
            "first_use": relativize(value.get("first_use"), repo), "probe": relativize(value.get("probe"), repo)}


def save(fig, name: str, public: Path) -> dict:
    path = public / "figures" / f"{name}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "LocalOCR evaluation"})
    plt.close(fig)
    return {"file": str(path.relative_to(public.parents[1])).replace("\\", "/"), "sha256": digest(path)}


def figure_threshold(reports: list[dict], probes: list[dict], public: Path) -> dict:
    """Shape threshold sweep (exact width mode): core-mask pairs and safe complete coverage per dev set, probe pairs."""
    groups: dict[str, list[dict]] = {}
    for report in reports:
        if report["config"]["outer_bracket_width"] == "exact" and report["config"]["exterior_bridge_ratio"] == .45:
            groups.setdefault(report["label"], []).append(report)
    fig, axes = plt.subplots(1, len(groups) + 1, figsize=(3.6 * (len(groups) + 1), 3.6))
    axes = list(axes)
    for ax, (label, items) in zip(axes, groups.items()):
        items = sorted(items, key=lambda r: r["config"]["shape_contrast_threshold"])
        x = [str(r["config"]["shape_contrast_threshold"]) for r in items]
        faces = items[0]["localization"]["faces"]
        ax.plot(x, [100 * r["localization"]["safe_complete_coverage"] / faces for r in items], marker="o", color=COLORS[2], label="安全完整覆盖（%）")
        ax.plot(x, [100 * r["models"][MODEL]["isolated_kaomoji"]["exact"] / r["models"][MODEL]["isolated_kaomoji"]["count"] for r in items], marker="s", color=COLORS[3], label="整串精确（%）")
        ax.set_title(f"{label}（{faces} 面）", fontsize=10); ax.set_xlabel("核心墨迹阈值"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    ax = axes[-1]
    probes = sorted([p for p in probes if p.get("diagnostics")], key=lambda p: p["diagnostics"].get("shape_threshold") or 0)
    if probes:
        x = [str(p["diagnostics"].get("shape_threshold")) for p in probes]
        ax.bar(x, [p["diagnostics"]["shape_core_pairs"] for p in probes], color=COLORS[1], label="核心掩码括号对")
        ax.bar(x, [p["diagnostics"]["shape_base_pairs"] for p in probes], color=COLORS[0], label="基础掩码括号对")
        ax.set_title("真实探针 · 括号对数", fontsize=10); ax.set_xlabel("核心墨迹阈值"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.suptitle("形状检测核心阈值 · 开发集与真实探针", fontsize=11); fig.tight_layout()
    return save(fig, "shape-threshold-sweep", public)


def figure_width_modes(reports: list[dict], probes: list[dict], public: Path) -> dict:
    """Outer-bracket width modes at one threshold: face exact on dev sets and on the real probe."""
    labels = sorted({r["label"] for r in reports})
    modes = ["exact", "baseline-else-raw", "raw"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    width = .26
    columns = labels + ["真实探针（known-input-001）"]
    for mi, mode in enumerate(modes):
        values = []
        for label in labels:
            match = [r for r in reports if r["label"] == label and r["config"]["outer_bracket_width"] == mode]
            model = match[0]["models"][MODEL]["isolated_kaomoji"] if match else None
            values.append(100 * model["exact"] / model["count"] if model else 0)
        probe = [p for p in probes if p.get("mode") == mode]
        values.append(100 * probe[0]["models"][MODEL]["face_exact_region"] / probe[0]["faces"] if probe else 0)
        xs = [i + (mi - 1) * width for i in range(len(columns))]
        ax.bar(xs, values, width, color=COLORS[2 + mi], label=MODE_LABELS[mode])
        for x, v in zip(xs, values):
            ax.text(x, v + .5, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(columns))); ax.set_xticklabels(columns, fontsize=9); ax.set_ylabel("颜文字整串精确（%）")
    ax.set_title("外括号宽度裁决模式 · 同一阈值", fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    return save(fig, "bracket-width-modes", public)


def figure_scene(reports: list[dict], split: str, public: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels, base, final = [], [], []
    for report in reports:
        model = report["models"][MODEL]
        labels.append(f"{report['label']}（{report['scenes']} 张）")
        base.append(100 * report["baseline"]["isolated_kaomoji"]["exact"] / report["baseline"]["isolated_kaomoji"]["count"])
        final.append(100 * model["isolated_kaomoji"]["exact"] / model["isolated_kaomoji"]["count"])
    x = range(len(labels)); width = .36
    ax.bar([i - width / 2 for i in x], base, width, color=COLORS[0], label="生产基线")
    ax.bar([i + width / 2 for i in x], final, width, color=COLORS[2], label="基线 + 冻结 Tiny 三视图融合")
    for i, (b, f) in enumerate(zip(base, final)):
        ax.text(i - width / 2, b + .5, f"{b:.2f}%", ha="center", fontsize=9); ax.text(i + width / 2, f + .5, f"{f:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel("颜文字整串准确率（%）"); ax.set_ylim(0, max(final + base + [10]) * 1.35)
    ax.set_title("自动整图链路 · " + ("新保留集首测" if split == "heldout" else "开发集")); ax.legend(fontsize=9); fig.tight_layout()
    return save(fig, f"scene-{split}-face-exact-accuracy", public)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dev-report", action="append", default=[], help="label=report (dev split); the config keys are read from the report")
    parser.add_argument("--heldout-report", action="append", default=[], help="label=report")
    parser.add_argument("--probe-report", action="append", default=[], help="mode=label=path; known-input-001 development probes (never independent)")
    parser.add_argument("--known-input-report", action="append", default=[], help="label=path; independent known-input first use")
    parser.add_argument("--dual-freeze", type=Path, action="append", default=[])
    parser.add_argument("--gpu-budget", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    public = repo / PUBLIC_REL
    registry = read(repo / "artifacts/ocr-data-fonts-v1/fonts.json")
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = repo / "artifacts/ocr-data-fonts-v1" / cjk["file"]
    if cjk["license"] != "OFL-1.1" or digest(font_path) != cjk["sha256"]:
        raise ValueError("Chart font must be the registered OFL CJK font")
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "svg.fonttype": "path", "svg.hashsalt": "localocr-real-rendering-20261015",
                         "axes.unicode_minus": False})
    sources = []

    def load(path: Path, role: str) -> dict:
        resolved = confined(repo, path)
        sources.append({"role": role, "file": str(path).replace("\\", "/"), "sha256": digest(resolved)})
        return read(resolved)

    dev_reports, heldout_reports, probes, known = [], [], [], []
    for item in args.dev_report:
        label, path = item.split("=", 1)
        dev_reports.append(scene_report(load(Path(path), f"dev:{label}"), "dev", label))
    for item in args.heldout_report:
        label, path = item.split("=", 1)
        heldout_reports.append(scene_report(load(Path(path), f"heldout:{label}"), "heldout", label))
    for item in args.probe_report:
        mode, label, path = item.split("=", 2)
        value = load(Path(path), f"probe:{label}")
        if value.get("independent") is not False:
            raise ValueError("Probe reports must be marked not independent: " + label)
        probes.append({"mode": mode, **real_report(value, label, repo)})
    for item in args.known_input_report:
        label, path = item.split("=", 1)
        value = load(Path(path), f"known-input:{label}")
        if value.get("independent") is not True or not value.get("first_use"):
            raise ValueError("Known-input reports must carry the independent first-use claim: " + label)
        known.append(real_report(value, label, repo))
    duals = [{"file": str(path).replace("\\", "/"), "sha256": digest(confined(repo, path)), **{k: v for k, v in read(confined(repo, path)).items() if k in ("schema_version", "frozen_utc")}}
             for path in args.dual_freeze]
    budget = load(args.gpu_budget, "gpu-budget")
    figures = []
    if any(r["config"]["outer_bracket_width"] == "exact" for r in dev_reports):
        figures.append(figure_threshold(dev_reports, probes, public))
    if len({r["config"]["outer_bracket_width"] for r in dev_reports}) > 1:
        figures.append(figure_width_modes(dev_reports, probes, public))
    if heldout_reports:
        figures.append(figure_scene(heldout_reports, "heldout", public))
    summary = {"schema_version": SUMMARY_SCHEMA, "round": "real-rendering-20261015", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Synthetic CC0 scenes and counts-only private real screenshots; no accuracy claim beyond the listed sets; candidate and classifier stay disabled.",
               "normalization": "Raw Unicode exact match and codepoint CER; no width, space or combining-mark normalization; the outer-bracket width rule changes output bytes, never the metric; exact_except_outer_bracket_width is a diagnostic count.",
               "gpu_budget": {"maximum_seconds": budget["maximum_seconds"], "used_seconds": budget["used_seconds"], "runs": len(budget.get("runs", [])),
                              "previous_sha256": budget["round_binding"]["previous_sha256"], "previous_used_seconds": budget["round_binding"]["previous_used_seconds"]},
               "scene": {"dev": dev_reports, "heldout": heldout_reports},
               "real_screens": {"probes": probes, "known_input": known}, "dual_freezes": duals, "figures": figures, "sources": sources,
               "font": {"id": cjk["id"], "license": cjk["license"], "sha256": cjk["sha256"]},
               "promote": False, "enabled": False, "production_enabled": False}
    write(public / "summary.json", summary)
    receipt = {"schema_version": RECEIPT_SCHEMA, "built_utc": summary["generated_utc"], "summary_sha256": digest(public / "summary.json"),
               "figures": figures, "sources": sources, "script_sha256": digest(Path(__file__)),
               "statement": "Built from explicit completed local reports; no images, labels, row predictions, private screenshots or weights opened; no inference; no network."}
    write(public / "summary.build-receipt.json", receipt)
    print(json.dumps({"summary": str(PUBLIC_REL / "summary.json"), "figures": [f["file"] for f in figures]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
