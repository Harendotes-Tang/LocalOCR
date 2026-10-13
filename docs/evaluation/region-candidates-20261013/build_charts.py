"""Build the anonymous summary and charts of the region-candidates-20261013 round from explicit local reports.

Inputs are completed evaluation reports retained under ``artifacts/region-candidates-20261013`` and the
comparison baselines of the previous round under ``artifacts/kaomoji-grammar-20261013`` (ignored
directories). Only aggregate counts, dataset/model bindings and SHA-256 identities are copied into the
public summary; images, labels, row predictions, private screenshots and weights are never opened. SVG
text is rendered as glyph outlines from a locally obtained OFL font. No model inference, no network I/O.
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

PUBLIC_REL = Path("docs/evaluation/region-candidates-20261013")
ROUND_REL = Path("artifacts/region-candidates-20261013")
PREVIOUS_REL = Path("artifacts/kaomoji-grammar-20261013")
SUMMARY_SCHEMA = "localocr.region-candidates.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.region-candidates.assets-build.v1"
MODEL_LABELS = {"production-baseline": "生产文字基线", "ppocrv6-tiny-exp-20261013-001": "Tiny · 冻结专项（exp-20261013-001）"}
COLORS = ("#64748b", "#93b3cb", "#155e9b", "#b45309", "#15803d")
FUNNEL_KEYS = ("faces", "found_any", "complete_ink_coverage", "safe_complete_coverage", "safe_routed_complete_coverage",
               "routed_candidate_count", "body_regions_routed", "structural_components")


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
            "dataset_binding": binding["dataset"], "config_views": binding["config"]["views"],
            "exterior_bridge_ratio": binding["config"]["candidate"].get("exterior_bridge_ratio"),
            "classifier": (binding["config"].get("classifier") or {}).get("id"),
            "baseline": {"isolated_kaomoji": summary["baseline"]["isolated_kaomoji"]}, "models": models,
            "localization": localization, "candidate_stage": summary.get("candidate_stage"),
            "candidate_sources": summary.get("candidate_sources"),
            "resources": {key: value["resources"].get(key) for key in ("load_seconds", "first_scene_seconds", "warm", "total_warm_seconds",
                                                                        "peak_rss_bytes", "intra_op_threads", "inter_op_threads")}}


def real_screens_report(value: dict) -> dict:
    summary = value["summary"]
    return {"screens": value["manifest"]["screens"], "environment": value["manifest"].get("environment"),
            "faces": summary["faces"], "faces_ambiguous_excluded": summary["faces_ambiguous_excluded"],
            "body_lines": summary["body_lines"], "baseline": summary["baseline"], "models": summary["models"],
            "pixel_branch": value.get("pixel_branch"), "timing": value.get("timing"), "scope": value.get("scope")}


def save(fig, name: str, public: Path) -> dict:
    path = public / "figures" / f"{name}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "LocalOCR evaluation"})
    plt.close(fig)
    return {"file": str(path.relative_to(public.parents[1])).replace("\\", "/"), "sha256": digest(path)}


def figure_funnel(pairs: list[tuple[dict, dict]], public: Path) -> dict:
    """Candidate-stage funnel per dev dataset: previous final code versus this round's final rules."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.8), sharey=False)
    axes = list(axes) if len(pairs) > 1 else [axes]
    keys = ("complete_ink_coverage", "safe_complete_coverage", "safe_routed_complete_coverage")
    names = ("完整覆盖", "安全完整覆盖", "安全完整且送专项")
    for ax, (before, after) in zip(axes, pairs):
        faces = after["localization"]["faces"]
        x = range(len(keys)); width = .36
        b = [100 * before["localization"].get(k, 0) / faces for k in keys]
        a = [100 * after["localization"].get(k, 0) / faces for k in keys]
        ax.bar([i - width / 2 for i in x], b, width, color=COLORS[0], label="上轮最终代码")
        ax.bar([i + width / 2 for i in x], a, width, color=COLORS[2], label="本轮候选规则")
        for i, (u, v) in enumerate(zip(b, a)):
            ax.text(i - width / 2, u + .8, f"{u:.0f}%", ha="center", fontsize=8); ax.text(i + width / 2, v + .8, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 105)
        ax.set_title(f"{after['label']}（{faces} 面）", fontsize=10); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("占真值颜文字数（%）"); axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("候选阶段漏斗 · 开发集", fontsize=11); fig.tight_layout()
    return save(fig, "candidate-stage-dev-funnel", public)


def figure_scene(reports: list[dict], split: str, public: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels, base, final = [], [], []
    for report in reports:
        identity = next(iter(report["models"]))
        model = report["models"][identity]
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
    parser.add_argument("--dev-pair", action="append", default=[], help="label=previous_report=current_report (dev split, same dataset)")
    parser.add_argument("--dev-report", action="append", default=[], help="label=report for new datasets without a previous-round baseline")
    parser.add_argument("--heldout-report", action="append", default=[], help="label=report")
    parser.add_argument("--real-screens-report", action="append", default=[], help="label=path; counts-only real-screenshot reports (probe and known-input)")
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
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "svg.fonttype": "path", "svg.hashsalt": "localocr-region-candidates-20261013",
                         "axes.unicode_minus": False})
    sources = []

    def load(path: Path, role: str) -> dict:
        resolved = confined(repo, path)
        sources.append({"role": role, "file": str(path).replace("\\", "/"), "sha256": digest(resolved)})
        return read(resolved)

    pairs, dev_reports, heldout_reports = [], [], []
    for item in args.dev_pair:
        label, previous, current = item.split("=", 2)
        pairs.append((scene_report(load(Path(previous), f"dev-previous:{label}"), "dev", label),
                      scene_report(load(Path(current), f"dev-current:{label}"), "dev", label)))
    for item in args.dev_report:
        label, path = item.split("=", 1)
        dev_reports.append(scene_report(load(Path(path), f"dev:{label}"), "dev", label))
    for item in args.heldout_report:
        label, path = item.split("=", 1)
        heldout_reports.append(scene_report(load(Path(path), f"heldout:{label}"), "heldout", label))
    real = {}
    for item in args.real_screens_report:
        label, path = item.split("=", 1)
        value = load(Path(path), "real-screens:" + label)
        counts = confined(repo, Path(path)).parent / "pixel-branch.json"
        if counts.exists():
            value["pixel_branch"] = {k: v for k, v in load(counts.relative_to(repo), "real-screens-pixel-branch:" + label).items() if k != "per_screen"}
        real[label] = real_screens_report(value)
    duals = [{"file": str(path).replace("\\", "/"), "sha256": digest(confined(repo, path)), **{k: v for k, v in read(confined(repo, path)).items() if k in ("schema_version", "frozen_utc")}}
             for path in args.dual_freeze]
    budget = load(args.gpu_budget, "gpu-budget")
    figures = []
    if pairs:
        figures.append(figure_funnel(pairs, public))
    if dev_reports + [after for _, after in pairs]:
        figures.append(figure_scene([after for _, after in pairs] + dev_reports, "dev", public))
    if heldout_reports:
        figures.append(figure_scene(heldout_reports, "heldout", public))
    summary = {"schema_version": SUMMARY_SCHEMA, "round": "region-candidates-20261013", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Synthetic CC0 scenes only; counts and bindings; no real-screenshot accuracy claim; candidate and classifier stay disabled.",
               "normalization": "Raw Unicode exact match and codepoint CER; no width, space or combining-mark normalization; unsupported cut-face parts are counted and never scored.",
               "gpu_budget": {"maximum_seconds": budget["maximum_seconds"], "used_seconds": budget["used_seconds"], "runs": len(budget.get("runs", [])),
                              "previous_sha256": budget["round_binding"]["previous_sha256"], "previous_used_seconds": budget["round_binding"]["previous_used_seconds"]},
               "scene": {"dev_pairs": [{"label": after["label"], "previous": before, "current": after} for before, after in pairs],
                         "dev": dev_reports, "heldout": heldout_reports},
               "real_screens": real or None, "dual_freezes": duals, "figures": figures, "sources": sources,
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
