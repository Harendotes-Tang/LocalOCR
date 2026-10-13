"""Build the anonymous summary and charts of the kaomoji-grammar-20261013 round from explicit local reports.

Inputs are completed evaluation reports retained under ``artifacts/kaomoji-grammar-20261013``
(an ignored directory). Only aggregate counts, dataset/model bindings and SHA-256
identities are copied into the public summary; images, labels, row predictions, private
screenshots and weights are never opened. SVG text is rendered as glyph outlines from a
locally obtained OFL font. The script performs no model inference and no network I/O.
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

PUBLIC_REL = Path("docs/evaluation/kaomoji-grammar-20261013")
ROUND_REL = Path("artifacts/kaomoji-grammar-20261013")
SUMMARY_SCHEMA = "localocr.kaomoji-grammar.anonymous-evaluation.v1"
RECEIPT_SCHEMA = "localocr.kaomoji-grammar.assets-build.v1"
MODEL_LABELS = {
    "production-baseline": "生产文字基线",
    "ppocrv6-tiny-reference-20261013-official-pretrained": "Tiny · 官方预训练",
    "ppocrv6-tiny-exp-20261013-001": "Tiny · 本轮微调（exp-20261013-001）",
}
COLORS = ("#64748b", "#93b3cb", "#155e9b", "#b45309", "#15803d")


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
        raise ValueError("Reports must stay inside the round root: " + str(path))
    return resolved


def kind_metrics(block: dict) -> dict:
    return {"count": block["count"], "exact": block["exact"], "cer": block.get("cer"), "ps_cer": block.get("ps_cer"),
            "false_nonempty_outputs": block.get("false_nonempty_outputs")}


def crop_report(value: dict, split: str) -> dict:
    if value.get("split") != split or value.get("promote") is not False or value.get("enabled") is not False:
        raise ValueError("Crop report split/flags differ")
    if split == "heldout" and (value.get("independent_heldout") is not True or not value.get("first_use")):
        raise ValueError("Crop heldout report lacks the first-use claim")
    models = {}
    for identity, model in value["models"].items():
        summary = model["summary"]
        models[identity] = {"label": MODEL_LABELS.get(identity, identity), "stage": model.get("stage"),
                            "overall": kind_metrics(summary["overall"]),
                            "by_kind": {kind: kind_metrics(block) for kind, block in summary["by_kind"].items()},
                            "research_quality_gates": model.get("research_quality_gates")}
    return {"split": split, "dataset_binding": value["dataset_binding"], "models": models}


def scene_report(value: dict, split: str, dataset_name: str) -> dict:
    if value.get("split") != split or value.get("promote") is not False or value.get("enabled") is not False or value.get("actual_cpu_inference") is not True:
        raise ValueError("Scene report split/flags differ")
    if split == "heldout" and value.get("independent_heldout") is not True:
        raise ValueError("Scene heldout report lacks the first-use claim")
    summary = value["summary"]
    models = {}
    for identity, model in summary["models"].items():
        models[identity] = {"label": MODEL_LABELS.get(identity, identity), "isolated_kaomoji": model["isolated_kaomoji"],
                            "body": model["body"], "negative": model["negative"], "routing_effectiveness": model["routing_effectiveness"],
                            "research_quality_gates": model.get("research_quality_gates")}
    binding = value["pipeline_binding"]
    return {"split": split, "dataset": dataset_name, "scenes": summary["scenes"], "scene_kinds": summary.get("scene_kinds"),
            "dataset_binding": binding["dataset"], "config_views": binding["config"]["views"],
            "split_configuration": binding["config"].get("split"), "classifier": (binding["config"].get("classifier") or {}).get("id"),
            "baseline": {"isolated_kaomoji": summary["baseline"]["isolated_kaomoji"]}, "models": models,
            "localization": summary["localization"]}


def training_curve(training: dict, best: dict) -> dict:
    points = [{"step": item["step"], "face_exact": item["dev"]["by_kind"]["face"]["exact_correct"],
               "face_cer": item["dev"]["by_kind"]["face"]["cer"], "negative_nonempty": item["dev"]["negative_nonempty"],
               "plain_exact": item["dev"]["by_kind"]["plain"]["exact_correct"]} for item in training["history"]]
    initial = training["initial_dev"]["by_kind"]
    return {"device": training["device"]["device"], "requested_seconds": training["requested_seconds"], "steps": training["steps"],
            "epochs_completed": training["epochs_completed"], "selection_policy": training["selection_policy"],
            "initial": {"face_exact": initial["face"]["exact_correct"], "plain_exact": initial["plain"]["exact_correct"],
                        "negative_nonempty": training["initial_dev"]["negative_nonempty"]},
            "selected": {"step": best["step"], "face_exact": best["dev"]["by_kind"]["face"]["exact_correct"],
                         "face_cer": best["dev"]["by_kind"]["face"]["cer"], "plain_exact": best["dev"]["by_kind"]["plain"]["exact_correct"],
                         "negative_nonempty": best["dev"]["negative_nonempty"]},
            "dev_rows": training["dev_rows"], "history": points}


def real_screens_report(value: dict) -> dict:
    summary = value["summary"]
    return {"screens": value["manifest"]["screens"], "environment": value["manifest"].get("environment"),
            "faces": summary["faces"], "faces_ambiguous_excluded": summary["faces_ambiguous_excluded"],
            "body_lines": summary["body_lines"], "embedded_faces": summary["embedded_faces"],
            "baseline": summary["baseline"], "models": summary["models"], "split_parents": summary.get("split", {}).get("parents"),
            "timing": value.get("timing"), "scope": value.get("scope")}


def save(fig, name: str, public: Path) -> dict:
    path = public / "figures" / f"{name}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "LocalOCR evaluation"})
    plt.close(fig)
    return {"file": str(path.relative_to(public.parents[1])).replace("\\", "/"), "sha256": digest(path)}


def figure_training(curve: dict, public: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    steps = [p["step"] for p in curve["history"]]
    ax.plot(steps, [p["face_exact"] for p in curve["history"]], marker="o", color=COLORS[2], label="开发集颜文字整串正确数（/1200）")
    ax.plot(steps, [p["negative_nonempty"] for p in curve["history"]], marker="s", color=COLORS[3], label="负样本误出（/200）")
    ax.axvline(curve["selected"]["step"], color=COLORS[4], linestyle="--", label=f"选定步 {curve['selected']['step']}")
    ax.set_xlabel("训练步"); ax.set_ylabel("计数"); ax.set_title("Tiny 原生训练开发曲线（face-pareto 选点）")
    ax.legend(loc="center right", fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
    return save(fig, "native-tiny-development", public)


def figure_crop(report: dict, split: str, public: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    names, values = [], []
    for identity, model in report["models"].items():
        face = model["by_kind"]["face"]
        names.append(model["label"]); values.append(100 * face["exact"] / face["count"])
    bars = ax.barh(names, values, color=COLORS[:len(names)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + .5, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105); ax.set_xlabel("颜文字整串准确率（%）"); ax.invert_yaxis()
    ax.set_title("裁片 CPU 评测 · " + ("新保留集首测" if split == "heldout" else "开发集")); fig.tight_layout()
    return save(fig, f"cpu-{split}-face-exact-accuracy", public)


def figure_scene(reports: list[dict], split: str, public: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels, base, final = [], [], []
    for report in reports:
        identity = next(iter(report["models"]))
        model = report["models"][identity]
        labels.append(f"{report['dataset']}（{report['scenes']} 张）")
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


def figure_scene_iterations(iterations: list[dict], public: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = [item["label"] for item in iterations]
    mixed = [item["mixed_face_exact_percent"] for item in iterations]
    cards = [item["cards_face_exact_percent"] for item in iterations]
    x = range(len(labels)); width = .36
    ax.bar([i - width / 2 for i in x], mixed, width, color=COLORS[2], label="原子混合整图 dev（164 面）")
    ax.bar([i + width / 2 for i in x], [v if v is not None else 0 for v in cards], width, color=COLORS[3], label="记录卡片整图 dev（269 面）")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("颜文字整串准确率（%）")
    ax.set_title("整图开发迭代（同一冻结 Tiny，仅路由/融合/配置变化）"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    return save(fig, "scene-dev-iterations", public)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--best-dev", type=Path, required=True)
    parser.add_argument("--crop-dev-report", type=Path, required=True)
    parser.add_argument("--crop-heldout-report", type=Path, required=True)
    parser.add_argument("--scene-dev-report", type=Path, action="append", default=[], help="name=path pairs, dataset name first")
    parser.add_argument("--scene-heldout-report", type=Path, action="append", default=[], help="name=path pairs")
    parser.add_argument("--iteration", action="append", default=[], help="label=mixed_percent[/cards_percent] in measured order")
    parser.add_argument("--real-screens-report", type=Path)
    parser.add_argument("--dual-freeze", type=Path, action="append", default=[])
    parser.add_argument("--gpu-budget", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(); public = repo / PUBLIC_REL
    registry = read(repo / "artifacts/ocr-data-fonts-v1/fonts.json")
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = repo / "artifacts/ocr-data-fonts-v1" / cjk["file"]
    if cjk["license"] != "OFL-1.1" or digest(font_path) != cjk["sha256"]:
        raise ValueError("Chart font must be the registered OFL CJK font")
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "svg.fonttype": "path", "svg.hashsalt": "localocr-kaomoji-grammar-20261013",
                         "axes.unicode_minus": False})
    sources = {}

    def load(role: str, path: Path) -> dict:
        resolved = confined(repo, path); sources[role] = digest(resolved); return read(resolved)

    training = load("tiny_training", args.training); best = load("tiny_selected", args.best_dev)
    crop_dev = crop_report(load("crop_cpu_development", args.crop_dev_report), "dev")
    crop_heldout = crop_report(load("crop_cpu_heldout", args.crop_heldout_report), "heldout")

    def pairs(items, split):
        reports = []
        for item in items:
            name, _, path = str(item).partition("=")
            reports.append(scene_report(load(f"scene_{split}_{name}", Path(path)), split, name))
        return reports

    scene_dev = pairs(args.scene_dev_report, "dev"); scene_heldout = pairs(args.scene_heldout_report, "heldout")
    duals = [load(f"dual_freeze_{index:02}", path) for index, path in enumerate(args.dual_freeze, 1)]
    budget = load("gpu_budget", args.gpu_budget)
    real = real_screens_report(load("real_screens", args.real_screens_report)) if args.real_screens_report else None
    iterations = []
    for item in args.iteration:
        label, _, values = item.partition("="); mixed, _, cards = values.partition("/")
        iterations.append({"label": label, "mixed_face_exact_percent": float(mixed), "cards_face_exact_percent": float(cards) if cards else None})
    curve = training_curve(training, best)
    figures = [figure_training(curve, public), figure_crop(crop_dev, "dev", public), figure_crop(crop_heldout, "heldout", public),
               figure_scene(scene_dev, "dev", public), figure_scene(scene_heldout, "heldout", public)]
    if iterations:
        figures.append(figure_scene_iterations(iterations, public))
    sources["chart_builder_source"] = digest(Path(__file__).resolve())
    summary = {"schema_version": SUMMARY_SCHEMA, "round": "kaomoji-grammar-20261013", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "production_enabled": False, "promote": False, "enabled": False,
               "scope": "Synthetic combination-isolated crops, atom-mixed scenes and record-card layout scenes; CPU evaluation; private real screenshots reported separately and independently; no APP acceptance",
               "normalization": "none; exact Unicode, spaces and combining characters",
               "training": curve, "gpu_budget": {"maximum_seconds": budget["maximum_seconds"], "used_seconds": budget["used_seconds"],
                                                  "runs": [{"label": r["label"], "status": r["status"], "elapsed_seconds": r["elapsed_seconds"]} for r in budget["runs"]]},
               "crop": {"dev": crop_dev, "heldout": crop_heldout}, "scene": {"dev": scene_dev, "heldout": scene_heldout},
               "scene_dev_iterations": iterations, "real_screens": real,
               "dual_freezes": [{"selected_id": d["gates"]["selected_id"], "frozen_utc": d["frozen_utc"], "scene_dataset_sha256": d["scene_dataset"]["binding"]["dataset_sha256"],
                                 "crop_dataset_sha256": d["crop_dataset"]["binding"]["dataset_sha256"], "research_quality_passed": d["gates"]["scene"]["passed"] and d["gates"]["crop"]["specialist_face_eligible"]} for d in duals],
               "sources": [{"role": role, "sha256": value} for role, value in sorted(sources.items())], "figures": figures}
    write(public / "summary.json", summary)
    receipt = {"schema_version": RECEIPT_SCHEMA, "builder_sha256": sources["chart_builder_source"], "matplotlib_version": matplotlib.__version__,
               "font": {"id": cjk["id"], "license": cjk["license"], "sha256": cjk["sha256"], "source_commit": cjk.get("source_commit")},
               "summary_sha256": digest(public / "summary.json"), "figures": figures, "generated_utc": summary["generated_utc"]}
    write(public / "summary.build-receipt.json", receipt)
    print(json.dumps({"summary": str(public / "summary.json"), "figures": len(figures)}))


if __name__ == "__main__":
    main()
