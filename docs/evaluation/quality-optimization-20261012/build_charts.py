"""Build anonymous charts from explicit completed, locally retained receipts.

Training inputs are fixed to this round; crop and scene reports must be explicit.
New heldout report statistics require both explicit completed
results and a matching dual-freeze first-use chain. Image, label, row-prediction
and weight files are never opened. SVG paths contain glyph outlines; the OFL
font files stay local. The script performs no model inference or network I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter


PUBLIC_REL = Path("docs/evaluation/quality-optimization-20261012")
ROUND_REL = Path("artifacts/quality-optimization-20261012")
MODEL_ORDER = (
    ("production-baseline", "生产基线", "baseline"),
    ("ppocrv6-tiny-reference-20261012-official-pretrained", "Tiny · 官方预训练", "official-pretrained"),
    ("ppocrv6-tiny-exp-20261012-001", "Tiny · 选定微调", "finetuned"),
    ("repsvtr-mobile-exp-20261012-000-initial", "RepSVTR · 扩字典初始化", "expanded-head-initialization"),
    ("repsvtr-mobile-exp-20261012-001", "RepSVTR · 选定微调", "finetuned"),
)
COLORS = ("#64748b", "#93b3cb", "#155e9b", "#d1aa87", "#9b4f16")
LOCALIZATION_FIELDS = (
    "candidate_count", "candidates_intersecting_body", "candidates_intersecting_graphic", "eligible_candidate_count",
    "body_regions", "body_regions_with_candidate", "body_regions_routed", "faces", "found_any", "missed",
    "complete_ink_coverage", "safe_complete_coverage", "partial_only", "eligible_complete_ink_coverage",
    "routed_complete_ink_coverage", "safe_routed_complete_coverage", "baseline_no_isolated_face_output",
    "baseline_detection_missed_faces", "routed_candidate_count", "routed_candidates_intersecting_body",
    "routed_candidates_intersecting_graphic", "graphic_regions", "graphic_regions_with_candidate", "graphic_regions_routed",
    "negative_scenes", "negative_scenes_with_candidate", "negative_scenes_routed",
)
BODY_FIELDS = (
    "regions", "baseline_mixed_regions", "final_mixed_regions", "baseline_exact", "final_exact",
    "baseline_correct_to_wrong", "erroneous_rewrites", "body_intersection_transcript_changes", "body_literal_deleted",
    "verified_body_preserving_splices", "new_body_intersection_errors",
)
NEGATIVE_FIELDS = (
    "graphic_regions", "baseline_graphic_nonempty", "final_graphic_nonempty", "new_graphic_nonempty",
    "negative_scenes", "baseline_nonempty_scenes", "final_nonempty_scenes", "new_nonempty_scenes",
    "graphic_only_scenes", "graphic_only_baseline_nonempty", "graphic_only_final_nonempty", "graphic_only_new_nonempty",
    "blank_scenes", "blank_baseline_nonempty", "blank_final_nonempty", "blank_new_nonempty",
)
ROUTING_FIELDS = ("actual_routed_candidates", "face_wrong_to_exact", "isolated_face_wrong_to_exact", "face_exact_to_wrong")
RESOURCE_FIELDS = ("peak_rss_bytes", "peak_private_bytes", "intra_op_threads", "inter_op_threads", "onnxruntime_version")
ALLOCATION_BYTES = 6 * 1024 * 1024
MINIMUM_FREE_BYTES = 8 * 1024 * 1024 * 1024
OUTPUT_ROOTS: tuple[Path, ...] = ()
GENERATED_SVGS: list[Path] = []


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}")
    return value


def configure_outputs(repo: Path) -> None:
    global OUTPUT_ROOTS
    OUTPUT_ROOTS = ((repo / PUBLIC_REL).resolve(), (repo / ROUND_REL / "reporting").resolve())


def write_bytes(path: Path, payload: bytes) -> None:
    """Reserve actual serialized bytes within the owned allocation before writing."""
    target = path.resolve()
    if not OUTPUT_ROOTS or not any(target.is_relative_to(root) for root in OUTPUT_ROOTS):
        raise ValueError("Output escaped the two reporting-owned directories")
    current = sum(item.stat().st_size for root in OUTPUT_ROOTS for item in root.rglob("*") if item.is_file())
    previous = target.stat().st_size if target.is_file() else 0
    additional = len(payload) - previous
    if current + additional > ALLOCATION_BYTES:
        raise ValueError(f"Reporting allocation would exceed {ALLOCATION_BYTES} bytes")
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free - max(additional, 0) < MINIMUM_FREE_BYTES:
        raise ValueError("Output would cross the confirmed 8 GiB free-space floor")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write(path: Path, value: dict) -> None:
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def integer_fields(value: dict, fields: tuple[str, ...]) -> dict:
    result = {key: value[key] for key in fields}
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise ValueError("Expected nonnegative integer aggregates")
    return result


def sha256_identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 identity")
    return value


def model_metadata(binding: dict) -> dict:
    result = {key: sha256_identity(binding[key]) for key in ("candidate_sha256", "model_sha256", "vocabulary_sha256")}
    result["model_bytes"] = binding["model_bytes"]
    if type(result["model_bytes"]) is not int or result["model_bytes"] <= 0:
        raise ValueError("Model byte size must be a positive integer")
    return result


def dataset_metadata(binding: dict, *, scene: bool) -> dict:
    """Copy only fixed dataset hashes and integer population counts."""
    fields = ("scenes", "faces") if scene else ("count", "groups", "positive_groups")
    splits = ("dev", "heldout") if scene else ("train", "dev", "heldout")
    result = {"dataset_sha256": binding["dataset_sha256"],
              "split_sha256": {split: binding["split_sha256"][split] for split in splits},
              "counts": {split: integer_fields(binding["counts"][split], fields) for split in splits}}
    for sha in (result["dataset_sha256"], *result["split_sha256"].values()):
        sha256_identity(sha)
    return result


def resource_metrics(resources: dict, *, scene: bool) -> tuple[dict, dict]:
    duration = "load_seconds" if scene else "session_load_seconds"
    fields = (*RESOURCE_FIELDS, duration, "process_wall_seconds")
    result = {key: resources[key] for key in fields}
    warm = {key: resources["warm"][key] for key in ("count", "p50_seconds", "p95_seconds")}
    if (any(type(result[key]) is not int or result[key] <= 0 for key in ("peak_rss_bytes", "peak_private_bytes"))
            or any(type(result[key]) is not int or result[key] <= 0 for key in ("intra_op_threads", "inter_op_threads"))
            or type(warm["count"]) is not int or warm["count"] <= 0
            or not isinstance(result["onnxruntime_version"], str)
            or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?", result["onnxruntime_version"])):
        raise ValueError("CPU resource counts/version are invalid")
    seconds = (result[duration], result["process_wall_seconds"], warm["p50_seconds"], warm["p95_seconds"])
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in seconds)
            or result["process_wall_seconds"] <= 0 or warm["p50_seconds"] <= 0
            or warm["p95_seconds"] < warm["p50_seconds"]):
        raise ValueError("CPU latency/wall durations are invalid")
    return result, warm


def budget_metadata(value: dict) -> dict:
    result = {key: value[key] for key in ("maximum_seconds", "used_seconds")}
    if (any(type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
            for seconds in result.values()) or result["maximum_seconds"] != 7200
            or result["used_seconds"] > result["maximum_seconds"]):
        raise ValueError("GPU budget snapshot differs from this round's confirmed 7200-second ceiling")
    return result


def crop_frozen_closure(value: dict, selection: dict, development: dict, dual: dict) -> None:
    """Check the measured models and source list against the frozen development proof."""
    for field in ("candidate_bindings", "baseline_binding"):
        if value[field] != selection[field] or development[field] != selection[field]:
            raise ValueError(f"Completed crop {field} differs from the frozen selection")
    inventory = value["source_inventory"]
    if not isinstance(inventory, list) or not inventory or inventory != development["source_inventory"]:
        raise ValueError("Completed crop source inventory differs from frozen development")
    seen = set()
    for item in inventory:
        name = item["file"]
        if (not isinstance(name, str) or name in seen or "\\" in name or Path(name).is_absolute()
                or Path(name).drive or ".." in Path(name).parts
                or type(item["size"]) is not int or item["size"] <= 0):
            raise ValueError("Frozen crop source inventory contains invalid paths/sizes")
        sha = sha256_identity(item["sha256"])
        if dual["source_inventory"].get(name) != sha:
            raise ValueError("Crop worker source differs from the dual freeze")
        seen.add(name)


def canonical_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def confined_report(repo: Path, round_root: Path, path: Path) -> Path:
    result = (path if path.is_absolute() else repo / path).resolve()
    if result.name != "evaluation.json" or not result.is_relative_to((round_root / "evaluation").resolve()):
        raise ValueError("A completed evaluation.json must remain inside this round's evaluation directory")
    return result


def validate_report(value: dict, kind: str, split: str) -> None:
    expected = "localocr.pretrained.evaluation.v1" if kind == "crop" else "localocr.region-research.v1"
    if (value["schema_version"] != expected or value["split"] != split
            or value["independent_heldout"] is not (split == "heldout")
            or value["enabled"] is not False or value["promote"] is not False):
        raise ValueError("Evaluation schema/split/research status differs")
    if kind == "crop":
        for model_id, model in value["models"].items():
            resources, worker = model["resources"], value["worker_results"][model_id]
            if (resources["providers"] != ["CPUExecutionProvider"] or resources["intra_op_threads"] != 2
                    or resources["inter_op_threads"] != 1 or worker["actual_cpu"] is not True
                    or worker["capture"] != "successful-owned-subprocess-stdout"
                    or worker["source_inventory"] != value["source_inventory"]
                    or type(worker["worker_pid"]) is not int or worker["worker_pid"] <= 0):
                raise ValueError("Crop report lacks consistent completed CPU worker evidence")
        if split == "dev" and value["first_use"] is not None:
            raise ValueError("Development report must not claim heldout first use")
    else:
        resources, binding = value["resources"], value["pipeline_binding"]
        if (value["actual_cpu_inference"] is not True or value["ground_truth_inference_input"] is not False
                or resources["providers"] != ["CPUExecutionProvider"] or resources["intra_op_threads"] != 2
                or resources["inter_op_threads"] != 1 or resources["worker_exit_code"] != 0
                or resources["capture"] != "successful-owned-subprocess"
                or resources["worker_ownership"]["owner_chain_verified"] is not True
                or type(resources["worker_pid"]) is not int or resources["worker_pid"] <= 0
                or resources["worker_pid"] != resources["worker_ownership"]["worker_pid"]
                or resources["actual_baseline_detection"] is not True
                or resources["actual_specialist_models"] != len(binding["models"])
                or binding["promote"] is not False or binding["enabled"] is not False):
            raise ValueError("Scene report lacks consistent actual CPU/owned completion evidence")
        if split == "dev" and value["heldout_usage"] is not None:
            raise ValueError("Development scene report must not claim heldout first use")


def heldout_use(value: dict, kind: str, round_root: Path, files: dict, hashes: dict) -> dict:
    """Check explicit completed-result first-use metadata; never open image/label/prediction files."""
    def referenced(role: str, reference: dict) -> tuple[Path, dict]:
        path = Path(reference["file"]).resolve()
        if not path.is_relative_to(round_root.resolve()) or path.suffix != ".json":
            raise ValueError("First-use metadata escaped this research round")
        sha = digest(path)
        if not re.fullmatch("[0-9a-f]{64}", reference["sha256"]) or sha != reference["sha256"]:
            raise ValueError("First-use metadata hash changed")
        role = f"{kind}_heldout_{role}"
        files[role], hashes[role] = path, sha
        return path, read(path)

    def stamp(text: str) -> datetime:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            raise ValueError("First-use metadata needs timezone-aware timestamps")
        return moment

    ref = value["first_use" if kind == "crop" else "heldout_usage"]
    if not isinstance(ref, dict):
        raise ValueError("A heldout report requires an actual first-use reference")
    _, claim = referenced("first_use", ref)
    expected = "localocr.pretrained.heldout-first-use.v1" if kind == "crop" else "localocr.region-heldout-first-use.v1"
    if claim["schema_version"] != expected:
        raise ValueError("Wrong heldout first-use schema")
    selection_ref = {"file": claim["selection_file"], "sha256": claim["selection_sha256"]} if kind == "crop" else claim["selection"]
    if kind == "crop" and (ref["selection_sha256"] != selection_ref["sha256"] or claim["dataset_binding"] != value["dataset_binding"]):
        raise ValueError("Crop first-use selection/data differ")
    if kind == "scene" and claim["pipeline_binding"] != value["pipeline_binding"]:
        raise ValueError("Scene first-use pipeline differs")
    selection_path, selection = referenced("selection", selection_ref)
    _, dual = referenced("dual_freeze", claim["dual_freeze"])
    selection_schema = "localocr.pretrained.selection.v1" if kind == "crop" else "localocr.region-selection.v1"
    if (selection["schema_version"] != selection_schema or selection["promote"] is not False
            or (kind == "crop" and selection["dataset_binding"] != value["dataset_binding"])
            or (kind == "scene" and selection["pipeline_binding"] != value["pipeline_binding"])):
        raise ValueError("The first-use selection differs from the completed heldout result")
    if (dual["schema_version"] != "localocr.quality-dual-freeze.v1" or dual["production_enabled"] is not False
            or dual["promote"] is not False or dual["enabled"] is not False
            or Path(dual[f"{kind}_selection"]["file"]).resolve() != selection_path
            or dual[f"{kind}_selection"]["sha256"] != selection_ref["sha256"]
            or selection["selected_id"] != dual["gates"]["selected_id"]
            or dual["gates"]["crop"]["specialist_face_eligible"] is not True
            or dual["gates"]["scene"]["passed"] is not True):
        raise ValueError("Heldout first-use dual selection differs")
    if selection["selected_id"] not in (MODEL_ORDER[2][0], MODEL_ORDER[4][0]):
        raise ValueError("The selected identity must be one of this round's evaluated finetuned models")
    if kind == "crop":
        development_path, development = referenced("development_report", selection["dev_report"])
        if (development_path.name != "evaluation.json"
                or not development_path.is_relative_to((round_root / "evaluation").resolve())):
            raise ValueError("Frozen crop development proof escaped completed evaluation reports")
        validate_report(development, "crop", "dev")
        crop_frozen_closure(value, selection, development, dual)
    completed = value["evaluated_utc" if kind == "crop" else "completed_utc"]
    if not stamp(selection["frozen_utc"]) <= stamp(dual["frozen_utc"]) <= stamp(claim["claimed_utc"]) <= stamp(completed):
        raise ValueError("Heldout selection/dual/first-use/completion chronology differs")
    return {"sha256": ref["sha256"], "selection_sha256": selection_ref["sha256"],
            "dual_freeze_sha256": claim["dual_freeze"]["sha256"], "claimed_utc": claim["claimed_utc"],
            "dual_frozen_utc": dual["frozen_utc"], "selected_id": selection["selected_id"]}


def counts(value: dict, *, native: bool) -> dict:
    """Whitelist integer aggregates and derive rates without copying row data."""
    result = {key: value[key] for key in ("count", "exact", "characters", "character_errors")}
    result["ps_characters"] = value["symbol_characters" if native else "ps_characters"]
    result["ps_errors"] = value["symbol_errors" if native else "ps_errors"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in result.values()):
        raise ValueError("Metrics must contain nonnegative integer counts")
    if not result["count"] or not result["characters"] or not result["ps_characters"]:
        raise ValueError("Face metric denominators must be positive")
    if result["exact"] > result["count"]:
        raise ValueError("Exact count exceeds sample count")
    for name, numerator, denominator, input_key in (
        ("exact_accuracy", "exact", "count", None if not native else "exact_accuracy"),
        ("cer", "character_errors", "characters", "cer"),
        ("ps_cer", "ps_errors", "ps_characters", "symbol_cer" if native else "ps_cer"),
    ):
        result[name] = result[numerator] / result[denominator]
        if input_key and not math.isclose(result[name], value[input_key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Rate disagrees with integer counts: {name}")
    return result


def native_curve(training: dict, family: str, selected_record: dict) -> dict:
    if training["selection_policy"] != "face-pareto":
        raise ValueError("Unrecognized native checkpoint selection policy")
    tiny = family == "tiny"
    dev_key = "dev" if tiny else "dev_metrics"
    initial = training["initial_dev"]["quality"] if tiny else training["initial_quality"]
    selected = training["best_dev"]["quality"] if tiny else training["selected_quality"]
    initial_face = counts(initial["by_kind"]["face"], native=True)
    points = [{"step": 0, **initial_face}]
    for record in training["history"]:
        if dev_key in record:
            points.append({"step": record["step"], **counts(record[dev_key]["quality"]["by_kind"]["face"], native=True)})
    steps = [point["step"] for point in points]
    if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(set(steps)):
        raise ValueError("Development evaluation steps must be unique and ordered")
    selected_step = selected_record["step"]
    if type(selected_step) is not int or selected_step <= 0:
        raise ValueError("Selected native checkpoint step must be a positive integer")
    selected_point = next(point for point in points if point["step"] == selected_step)
    expected_selected = counts(selected["by_kind"]["face"], native=True)
    if {k: v for k, v in selected_point.items() if k != "step"} != expected_selected:
        raise ValueError("Selected checkpoint differs from its recorded evaluation")
    for point in points:
        if any(point[key] != initial_face[key] for key in ("count", "characters", "ps_characters")):
            raise ValueError("Development denominators changed during the run")
    peak = max(points, key=lambda point: point["exact"])
    batch = training["batch_size" if tiny else "training_batch_size"]
    if (type(training["steps"]) is not int or training["steps"] <= 0 or type(batch) is not int or batch <= 0
            or type(training["learning_rate"]) not in (int, float) or not math.isfinite(training["learning_rate"])
            or training["learning_rate"] <= 0 or max(steps) > training["steps"]):
        raise ValueError("Native training aggregates are invalid")
    return {
        "framework": "Paddle native" if tiny else "PyTorch native",
        "run": "short-run-001",
        "selection_policy": training["selection_policy"],
        "selection_includes_nonincreasing_empty_label_false_outputs": True,
        "updates_completed": training["steps"],
        "batch_size": batch,
        "learning_rate": training["learning_rate"],
        "points": points,
        "selected": selected_point,
        "observed_exact_peak": peak,
        "peak_is_selected": peak["step"] == selected_step,
    }


def crop_rows(report: dict) -> list[dict]:
    rows = []
    bindings = {binding["id"]: binding for binding in report["candidate_bindings"]}
    for model_id, display_name, stage in MODEL_ORDER:
        model = report["models"][model_id]
        if model_id != "production-baseline" and (model["stage"] != stage or bindings[model_id]["stage"] != stage):
            raise ValueError("A CPU model's declared before/after stage differs from its bound candidate")
        resources, warm = resource_metrics(model["resources"], scene=False)
        row = {"id": model_id, "display_name": display_name, "stage": stage,
               "face": counts(model["summary"]["by_kind"]["face"], native=False),
               "resources": resources, "warm_latency_seconds": warm}
        if model_id != "production-baseline":
            row["model_binding"] = model_metadata(bindings[model_id])
            gate = model["research_quality_gates"]
            ordinary = {kind: integer_fields(gate["raw_generic_protection"][kind], ("correct_to_wrong", "erroneous_rewrites"))
                        for kind in ("plain", "punctuation")}
            negatives = {kind: integer_fields(gate["raw_negative"][kind], ("new_nonempty", "total_nonempty"))
                         for kind in ("blank", "graphic")}
            row["raw_generic_protection"] = ordinary
            row["raw_negative_protection"] = negatives
            row["raw_generic_replacement_passed"] = gate["raw_generic_replacement_passed"]
            row["specialist_face_eligible"] = gate["specialist_face_eligible"]
            baseline = rows[0]["face"]
            face_gate = {"exact_gain": row["face"]["exact"] - baseline["exact"],
                         "cer_reduction": baseline["cer"] - row["face"]["cer"],
                         "ps_cer_reduction": baseline["ps_cer"] - row["face"]["ps_cer"]}
            face_gate["passed"] = all(face_gate[key] > 0 for key in ("exact_gain", "cer_reduction", "ps_cer_reduction"))
            raw_pass = face_gate["passed"] and all(not any(value.values()) for value in ordinary.values())
            raw_pass = raw_pass and all(value["new_nonempty"] == 0 for value in negatives.values())
            if (type(row["raw_generic_replacement_passed"]) is not bool or type(row["specialist_face_eligible"]) is not bool
                    or gate["production_enabled"] is not False or gate["promote"] is not False
                    or gate["face"] != face_gate or row["specialist_face_eligible"] is not face_gate["passed"]
                    or row["raw_generic_replacement_passed"] is not raw_pass):
                raise ValueError("Raw crop research status differs")
        rows.append(row)
    reference = rows[0]["face"]
    if any(any(row["face"][key] != reference[key] for key in ("count", "characters", "ps_characters")) for row in rows):
        raise ValueError("CPU crop comparison reference denominators differ")
    return rows


def scene_summary(report: dict, iteration: str) -> dict:
    summary, binding = report["summary"], report["pipeline_binding"]
    baseline = counts(summary["baseline"]["isolated_kaomoji"], native=False)
    localization = integer_fields(summary["localization"], LOCALIZATION_FIELDS)
    if type(summary["scenes"]) is not int or summary["scenes"] <= 0 or baseline["count"] != localization["faces"]:
        raise ValueError("Scene/face denominators differ")
    if localization["found_any"] + localization["missed"] != localization["faces"]:
        raise ValueError("Localization found/missed counts differ")
    bindings = {model["id"]: model for model in binding["models"]}
    models = []
    for model_id, display_name, _ in (MODEL_ORDER[2], MODEL_ORDER[4]):
        model = summary["models"][model_id]
        face = counts(model["isolated_kaomoji"], native=False)
        if any(face[key] != baseline[key] for key in ("count", "characters", "ps_characters")):
            raise ValueError("Final scene face reference denominators differ")
        body, negative = integer_fields(model["body"], BODY_FIELDS), integer_fields(model["negative"], NEGATIVE_FIELDS)
        if not body["regions"] or not negative["negative_scenes"] or not negative["graphic_regions"]:
            raise ValueError("Final scene protection populations are missing")
        routing = integer_fields(model["routing_effectiveness"], ROUTING_FIELDS)
        raw_gate = model["research_quality_gates"]
        body_guard = {key: body[key] for key in ("baseline_correct_to_wrong", "erroneous_rewrites", "new_body_intersection_errors", "body_literal_deleted")}
        negative_guard = {key: negative[key] for key in ("new_nonempty_scenes", "new_graphic_nonempty")}
        face_gate = {"exact_gain": face["exact"] - baseline["exact"], "cer_reduction": baseline["cer"] - face["cer"],
                     "ps_cer_reduction": baseline["ps_cer"] - face["ps_cer"]}
        face_gate["passed"] = all(face_gate[key] > 0 for key in ("exact_gain", "cer_reduction", "ps_cer_reduction"))
        passed = (face_gate["passed"] and not any(body_guard.values()) and not any(negative_guard.values())
                  and routing["actual_routed_candidates"] > 0 and routing["face_wrong_to_exact"] > 0)
        if (raw_gate["face"] != face_gate or raw_gate["body_and_punctuation_new_errors"] != body_guard
                or raw_gate["negative_new_nonempty"] != negative_guard
                or raw_gate["actual_routed_candidates"] != routing["actual_routed_candidates"]
                or raw_gate["effective_face_improvements"] != routing["face_wrong_to_exact"]
                or raw_gate["passed"] is not passed or raw_gate["research_quality_passed"] is not passed
                or raw_gate["production_enabled"] is not False or raw_gate["promote"] is not False
                or raw_gate["app_acceptance"] is not False):
            raise ValueError("Scene research gates differ from reported integer aggregates")
        models.append({"id": model_id, "display_name": display_name, "face": face, "body": body, "negative": negative,
                       "routing": routing, "safe_complete_specialist_crop_counts": integer_fields(model["safe_complete_specialist_crops"], ("count", "exact")),
                       "research_gates": {"face": face_gate, "body_and_punctuation_new_errors": body_guard,
                                          "negative_new_nonempty": negative_guard, "passed": passed,
                                          "effective_face_improvements": routing["face_wrong_to_exact"],
                                          "production_enabled": False, "app_acceptance": False},
                       "model_binding": model_metadata(bindings[model_id])})
    resources, warm = resource_metrics(report["resources"], scene=True)
    return {"iteration": iteration, "split": report["split"], "scenes": summary["scenes"],
            "dataset_binding": dataset_metadata(binding["dataset"], scene=True), "pipeline_binding_canonical_sha256": canonical_sha(binding),
            "config_sha256": sha256_identity(binding["config_sha256"]), "implementation_canonical_sha256": canonical_sha(binding["implementation"]),
            "actual_cpu_inference": True, "production_enabled": False, "app_acceptance": False,
            "localization": localization, "baseline_face": baseline, "models": models,
            "resources": resources, "warm_latency_seconds": warm}


def save_figure(fig, name: str, public: Path, rendering: Path) -> None:
    try:
        for path, format_name, options in (
            (public / "figures" / f"{name}.svg", "svg", {"metadata": {"Date": None, "Creator": "LocalOCR evaluation"}}),
            (rendering / f"{name}-normal.png", "png", {"dpi": 110}),
            # Preserve the original aspect ratio at a 375 px Markdown image width.
            (rendering / f"{name}-375px.png", "png", {"dpi": 375 / fig.get_figwidth()}),
        ):
            buffer = BytesIO()
            fig.savefig(buffer, format=format_name, **options)
            write_bytes(path, buffer.getvalue())
            if format_name == "svg":
                GENERATED_SVGS.append(path)
    finally:
        plt.close(fig)


def curve_figure(data: dict, family: str, public: Path, rendering: Path) -> None:
    color = COLORS[2 if family == "tiny" else 4]
    title = "Tiny" if family == "tiny" else "RepSVTR Mobile"
    points, selected, peak = data["points"], data["selected"], data["observed_exact_peak"]
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 8.4), sharex=True)
    fig.subplots_adjust(left=.17, right=.96, top=.82, bottom=.14, hspace=.27)
    fig.suptitle(f"{title}：本轮短训练开发过程", x=.08, ha="left", y=.98, fontsize=17)
    fig.text(.08, .925, f"原生框架 · 合成颜文字 {selected['count']:,} 裁片 · 每次评估记录", fontsize=12)
    for ax, metric, label in zip(axes, ("exact_accuracy", "cer", "ps_cer"),
                                 ("整串准确率（%）", "CER（%，对数轴）", "P/S CER（%，对数轴）")):
        ax.plot([point["step"] for point in points], [point[metric] * 100 for point in points],
                color=color, marker="o", markersize=3, linewidth=1.8, label="开发集观测")
        ax.scatter(selected["step"], selected[metric] * 100, marker="*", s=160,
                   color="#111827", zorder=5, label="选定检查点")
        ax.set_ylabel(label, fontsize=12)
        ax.grid(axis="y", color="#d9e1e9", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "exact_accuracy":
            ax.set_ylim(0, 103)
            ax.scatter(peak["step"], peak[metric] * 100, marker="D", s=40,
                       facecolors="white", edgecolors=color, linewidths=1.5, zorder=6, label="整串观测峰值")
        elif all(point[metric] > 0 for point in points):
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
        ax.text(.98, .93 if metric != "exact_accuracy" else .48,
                f"选定 {selected[metric] * 100:.2f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, color="#111827",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.07, .897),
               ncol=3, fontsize=9, frameon=False, handlelength=1.6, columnspacing=1.2)
    axes[-1].set_xlabel("优化器更新步数", fontsize=13)
    fig.text(.08, .057, f"选定 {selected['exact']}/{selected['count']}（第 {selected['step']:,} 步）；"
             f"观测峰值 {peak['exact']}/{peak['count']}（第 {peak['step']:,} 步）。", fontsize=10)
    fig.text(.08, .026, "选点同时约束颜文字 CER、P/S CER 和负样本；本图不证明最终融合保护。", fontsize=9)
    save_figure(fig, f"native-{family}-development", public, rendering)


def cpu_figure(rows: list[dict], metric: str, public: Path, rendering: Path, split: str = "dev") -> None:
    population = "同一开发集" if split == "dev" else "同一新保留集"
    titles = {"exact_accuracy": f"{population} CPU 裁片：整串准确率", "cer": f"{population} CPU 裁片：CER",
              "ps_cer": f"{population} CPU 裁片：严格 P/S CER"}
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    fig.subplots_adjust(left=.37, right=.87, top=.77, bottom=.26)
    values = [row["face"][metric] * 100 for row in rows]
    bars = ax.barh(range(len(rows)), values, color=COLORS, height=.58)
    # The two before states also use hatching, so their distinction survives monochrome output.
    for index in (1, 3):
        bars[index].set_hatch("///")
        bars[index].set_edgecolor("#596775")
    ax.set_yticks(range(len(rows)), [row["display_name"] for row in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 105 if metric == "exact_accuracy" else max(values) * 1.12)
    ax.grid(axis="x", color="#d9e1e9", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    for i, value in enumerate(values):
        ax.text(value + ax.get_xlim()[1] * .02, i, f"{value:.2f}%", va="center", fontsize=11)
    ax.set_xlabel("准确率（%，越高越好）" if metric == "exact_accuracy" else "错误率（%，越低越好）", fontsize=12)
    fig.suptitle(titles[metric], x=.06, y=.975, ha="left", fontsize=16)
    fig.text(.06, .86, f"合成颜文字 {rows[0]['face']['count']:,} 裁片 · ONNX Runtime CPU · 线程 2:1", fontsize=11)
    if metric == "exact_accuracy":
        footer = "整串计数：" + "、".join(str(row["face"]["exact"]) for row in rows)
        footer += f" / {rows[0]['face']['count']}；顺序与图中一致。"
    else:
        footer = "CER 使用 Unicode 码点；P/S CER 仅取 Unicode P、S 类，排除组合标记。"
    fig.text(.06, .075, footer, fontsize=9)
    fig.text(.06, .035, "直接识别裁片；未经自动定位、最终融合、真实截图或 APP 生命周期验收。", fontsize=9)
    save_figure(fig, f"cpu-{split}-face-{metric.replace('_', '-')}", public, rendering)


def scene_metric_figure(reports: list[dict], metric: str, public: Path, rendering: Path, split: str = "dev") -> None:
    rows = []
    for report in reports:
        rows.append((f"{report['iteration']} · 基线", report["baseline_face"], COLORS[0]))
        for index, model in enumerate(report["models"]):
            name = "Tiny" if index == 0 else "RepSVTR"
            rows.append((f"{report['iteration']} · {name}", model["face"], COLORS[2 if index == 0 else 4]))
    fig, ax = plt.subplots(figsize=(6.3, 4.5 + .26 * len(reports)))
    fig.subplots_adjust(left=.30, right=.86, top=.77, bottom=.24)
    values = [face[metric] * 100 for _, face, _ in rows]
    ax.barh(range(len(rows)), values, color=[color for _, _, color in rows], height=.57)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows], fontsize=11)
    ax.invert_yaxis(); ax.set_xlim(0, max(max(values) * 1.16, 1))
    ax.grid(axis="x", color="#d9e1e9", linewidth=.7); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0, pad=8)
    for index, value in enumerate(values):
        text = f"{value:.2f}%"
        if metric == "exact_accuracy": text += f" ({rows[index][1]['exact']})"
        ax.text(value + ax.get_xlim()[1] * .02, index, text, va="center", fontsize=10)
    ax.set_xlabel("准确率（%，越高越好）" if metric == "exact_accuracy" else "错误率（%，越低越好）", fontsize=12)
    population = "开发整图" if split == "dev" else "新保留整图"
    name = {"exact_accuracy": "颜文字整串准确率", "cer": "颜文字 CER", "ps_cer": "颜文字严格 P/S CER"}[metric]
    fig.suptitle(f"自动链路：{population} {name}", x=.06, y=.975, ha="left", fontsize=15)
    fig.text(.06, .87, f"每次 {reports[0]['scenes']} 张整图 · {reports[0]['baseline_face']['count']} 个颜文字区域 · 最终融合文本", fontsize=11)
    fig.text(.06, .08, "开发迭代使用同一分区；裁片、整图及新保留集指标分别报告。" if split == "dev" else "模型与路由冻结后的首次测评；保留结果不用于调整参数。", fontsize=9)
    fig.text(.06, .038, "区域对齐依据真实可见像素；不代表私人真实截图或 APP 生命周期验收。", fontsize=9)
    save_figure(fig, f"scene-{split}-face-{metric.replace('_', '-')}", public, rendering)


def scene_localization_figure(reports: list[dict], public: Path, rendering: Path, split: str = "dev") -> None:
    fields = (("found_any", "发现任意候选"), ("complete_ink_coverage", "完整覆盖像素"),
              ("safe_complete_coverage", "安全完整覆盖"), ("eligible_complete_ink_coverage", "完整候选符合路由资格"),
              ("safe_routed_complete_coverage", "安全完整且实际路由"))
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    fig.subplots_adjust(left=.39, right=.91, top=.78, bottom=.21)
    width = .7 / len(reports)
    for iteration, report in enumerate(reports):
        offset = (iteration - (len(reports) - 1) / 2) * width
        positions = [index + offset for index in range(len(fields))]
        values = [report["localization"][field] for field, _ in fields]
        ax.barh(positions, values, height=width * .88, label=report["iteration"],
                color=["#91abc1", "#155e9b", "#64748b"][iteration % 3])
        for position, value in zip(positions, values):
            ax.text(value + 2, position, str(value), va="center", fontsize=10)
    ax.set_yticks(range(len(fields)), [label for _, label in fields], fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, reports[0]["baseline_face"]["count"] * 1.13)
    ax.grid(axis="x", color="#d9e1e9", linewidth=.7); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("颜文字区域数", fontsize=12)
    population = "开发整图" if split == "dev" else "新保留整图"
    fig.suptitle(f"{population}：定位覆盖与路由资格", x=.06, y=.98, ha="left", fontsize=15)
    fig.text(.06, .875, f"{reports[0]['scenes']} 张整图 · {reports[0]['baseline_face']['count']} 个颜文字区域", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06, .83), ncol=3, frameon=False, fontsize=9)
    fig.text(.06, .076, "计数单位均为颜文字区域；路由资格增加不等同于最终准确率提高。", fontsize=9)
    fig.text(.06, .035, "完整像素覆盖、安全覆盖、推理资格和实际路由分别统计。", fontsize=9)
    save_figure(fig, f"scene-{split}-localization", public, rendering)


def scene_routing_table(reports: list[dict], public: Path, rendering: Path, split: str = "dev") -> None:
    rows = [(report, model) for report in reports for model in report["models"]]
    fig, ax = plt.subplots(figsize=(6.3, 3.2 + .35 * len(rows)))
    fig.subplots_adjust(left=.04, right=.98, top=.78, bottom=.27); ax.axis("off")
    cells = []
    for report, model in rows:
        name = "Tiny" if model["id"] == MODEL_ORDER[2][0] else "RepSVTR"
        cells.append([f"{report['iteration']} · {name}", model["routing"]["actual_routed_candidates"],
                      model["routing"]["face_wrong_to_exact"], model["routing"]["face_exact_to_wrong"],
                      model["body"]["baseline_correct_to_wrong"], model["body"]["erroneous_rewrites"],
                      model["negative"]["new_nonempty_scenes"]])
    table = ax.table(cellText=cells, colLabels=["链路", "实际推理\n候选", "有效改善\n区域", "颜文\n正确转错", "正文\n正确转错", "正文\n错误改写", "负场景\n新增非空"],
                     cellLoc="center", colWidths=[.27, .12, .12, .12, .12, .12, .13], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(11.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e1e9"); cell.set_linewidth(.7)
        cell.set_facecolor("#eef3f7" if row == 0 else "white")
        if row > 0 and col in (3, 4, 5, 6) and int(cells[row - 1][col]) > 0:
            cell.set_text_props(color="#9b4f16", weight="bold")
    population = "开发整图" if split == "dev" else "新保留整图"
    fig.suptitle(f"{population}：实际推理、有效改善与融合保护", x=.06, y=.98, ha="left", fontsize=14)
    fig.text(.06, .88, "候选推理次数与改善区域数量使用各自单位；同时披露颜文字损失。", fontsize=10)
    fig.text(.06, .14, "正文保护还核验交叠错误与正文删除；所有原始计数见匿名 summary.json。", fontsize=9)
    fig.text(.06, .095, "实际推理候选数不等于融合接受数；局部接受可保留原区域前后缀。", fontsize=9)
    fig.text(.06, .05, "本表不宣称全部接受区域均完整，也不代表 APP 接入资格。", fontsize=9)
    save_figure(fig, f"scene-{split}-routing-protection", public, rendering)


def resource_figure(values: list[dict], public: Path, rendering: Path, *, scene: bool, split: str = "dev") -> None:
    """Plot measured worker resources with inference and whole-process scopes distinct."""
    labels = [value["iteration"] if scene else value["display_name"] for value in values]
    panels = 3 if scene else 2
    fig, axes = plt.subplots(panels, 1, figsize=(6.3, 7.7 if scene else 6.7))
    fig.subplots_adjust(left=.35 if not scene else .22, right=.85, top=.80,
                        bottom=.21 if not scene else .18, hspace=.72)
    positions = list(range(len(values)))
    for ax in axes:
        ax.invert_yaxis(); ax.grid(axis="x", color="#d9e1e9", linewidth=.7); ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks(positions, labels, fontsize=10); ax.tick_params(axis="y", length=0, pad=7)
    latency = axes[0]
    for key, offset, color, label in (("p50_seconds", -.17, "#155e9b", "p50"), ("p95_seconds", .17, "#93b3cb", "p95")):
        numbers = [value["warm_latency_seconds"][key] * 1000 for value in values]
        latency.barh([position + offset for position in positions], numbers, height=.29, color=color, label=label)
        for position, number in zip(positions, numbers):
            latency.text(number + max(numbers) * .025, position + offset, f"{number:.2f}", va="center", fontsize=9)
    latency.set_xlim(0, max(value["warm_latency_seconds"]["p95_seconds"] * 1000 for value in values) * 1.24)
    latency.set_xlabel("每张整图暖推理延迟（ms）" if scene else "每个裁片暖推理延迟（ms）", fontsize=11)
    latency.legend(loc="lower right", bbox_to_anchor=(1, 1.02), frameon=False, ncol=2, fontsize=9)
    if scene:
        wall = axes[1]
        numbers = [value["resources"]["process_wall_seconds"] for value in values]
        wall.barh(positions, numbers, height=.56, color="#64748b")
        wall.set_xlim(0, max(numbers) * 1.22)
        for position, number in zip(positions, numbers):
            wall.text(number + max(numbers) * .025, position, f"{number:.1f}", va="center", fontsize=9)
        wall.set_xlabel("完整 worker 进程墙钟时间（s）", fontsize=11)
    memory = axes[-1]
    for key, offset, color, label in (("peak_rss_bytes", -.17, "#64748b", "RSS"), ("peak_private_bytes", .17, "#c3ced8", "Private")):
        numbers = [value["resources"][key] / 1024 / 1024 for value in values]
        memory.barh([position + offset for position in positions], numbers, height=.29, color=color, label=label)
        for position, number in zip(positions, numbers):
            memory.text(number + max(numbers) * .025, position + offset, f"{number:.1f}", va="center", fontsize=9)
    memory.set_xlim(0, max(value["resources"][key] / 1024 / 1024 for value in values
                           for key in ("peak_rss_bytes", "peak_private_bytes")) * 1.24)
    memory.set_xlabel("worker 进程峰值内存（MiB）", fontsize=11)
    memory.legend(loc="lower right", bbox_to_anchor=(1, 1.02), frameon=False, ncol=2, fontsize=9)
    population = "开发" if split == "dev" else "新保留"
    scope = "整图链路" if scene else "裁片识别"
    fig.suptitle(f"CPU 资源：{population}{scope}", x=.06, y=.98, ha="left", fontsize=16)
    fig.text(.06, .90, "ONNX Runtime · CPUExecutionProvider · 线程 2:1 · 实际 worker 测量", fontsize=10)
    if scene:
        fig.text(.06, .072, "墙钟包含完整测评与库存检查；暖推理延迟使用单张整图范围。", fontsize=9)
        fig.text(.06, .035, "峰值覆盖该 worker 的全部会话与辅助处理，不能解释为单模型内存。", fontsize=9)
    else:
        fig.text(.06, .072, "延迟包括预处理与解码；基线含三个会话，其余各含一个会话。", fontsize=9)
        fig.text(.06, .035, "峰值为进程测量；不同数据分区与系统负载分别报告，不作硬件推广。", fontsize=9)
    save_figure(fig, f"{'scene' if scene else 'cpu'}-{split}-resources", public, rendering)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--crop-dev-report", type=Path, required=True,
                        help="Explicit completed development evaluation JSON")
    parser.add_argument("--scene-dev-report", type=Path, action="append", default=[],
                        help="Explicit completed scene development report; repeat in measured iteration order")
    parser.add_argument("--crop-heldout-report", type=Path,
                        help="Explicit authorized completed first new heldout result; requires scene counterpart")
    parser.add_argument("--scene-heldout-report", type=Path,
                        help="Explicit authorized completed first new heldout result; requires crop counterpart")
    parser.add_argument("--validate-only", action="store_true", help="Check metadata and aggregates without writing charts")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    round_root, public = repo / ROUND_REL, repo / PUBLIC_REL
    crop_report = confined_report(repo, round_root, args.crop_dev_report)
    if bool(args.crop_heldout_report) != bool(args.scene_heldout_report):
        raise ValueError("New heldout statistics require both explicit completed crop and scene reports")
    rendering = round_root / "reporting" / "rendering"
    files = {
        "chart_builder_source": Path(__file__).resolve(),
        "tiny_training": round_root / "models/paddle/run-001/training.json",
        "tiny_selected": round_root / "models/paddle/run-001/best-dev.json",
        "repsvtr_training": round_root / "models/openocr/run-001/training.json",
        "repsvtr_selected": round_root / "models/openocr/run-001/best-selection.json",
        "crop_cpu_development": crop_report,
        "gpu_budget_snapshot": round_root / "gpu-budget.json",
        "font_registry": repo / "artifacts/ocr-data-fonts-v1/fonts.json",
        "native_metric_definition": repo / "tools/PretrainedOcr/quality_selection.py",
    }
    for index, path in enumerate(args.scene_dev_report, 1):
        files[f"scene_cpu_development_{index:03}"] = confined_report(repo, round_root, path)
    if args.crop_heldout_report:
        files["crop_cpu_heldout"] = confined_report(repo, round_root, args.crop_heldout_report)
        files["scene_cpu_heldout"] = confined_report(repo, round_root, args.scene_heldout_report)
    source_hashes = {role: digest(path) for role, path in files.items()}
    inputs = {role: read(path) for role, path in files.items() if path.suffix == ".json"}
    report = inputs["crop_cpu_development"]
    validate_report(report, "crop", "dev")
    registry = inputs["font_registry"]
    cjk = next(record for record in registry["fonts"] if record["id"] == "cjk")
    font_path = files["font_registry"].parent / cjk["file"]
    if (cjk["license"] != "OFL-1.1" or digest(font_path) != sha256_identity(cjk["sha256"])
            or not isinstance(cjk["source_commit"], str) or not re.fullmatch("[0-9a-f]{40}", cjk["source_commit"])):
        raise ValueError("Local OFL font integrity failed")
    curves = {family: native_curve(inputs[f"{family}_training"], family, inputs[f"{family}_selected"])
              for family in ("tiny", "repsvtr")}
    rows = crop_rows(report)
    scenes = []
    for index, path in enumerate(args.scene_dev_report, 1):
        value = inputs[f"scene_cpu_development_{index:03}"]
        validate_report(value, "scene", "dev")
        phase = re.fullmatch(r"scene-dev-(\d{3})", Path(path).parent.name)
        if not phase:
            raise ValueError("Scene development report must retain its measured scene-dev-NNN identity")
        scenes.append(scene_summary(value, "DEV" + phase.group(1)))
    if scenes and any(value["dataset_binding"] != scenes[0]["dataset_binding"] or value["scenes"] != scenes[0]["scenes"] for value in scenes):
        raise ValueError("Scene development iteration comparison requires the same frozen dataset")
    heldout = {}
    if args.crop_heldout_report:
        cvalue, svalue = inputs["crop_cpu_heldout"], inputs["scene_cpu_heldout"]
        validate_report(cvalue, "crop", "heldout"); validate_report(svalue, "scene", "heldout")
        cp = heldout_use(cvalue, "crop", round_root, files, source_hashes)
        sp = heldout_use(svalue, "scene", round_root, files, source_hashes)
        if cp["dual_freeze_sha256"] != sp["dual_freeze_sha256"] or cp["selected_id"] != sp["selected_id"]:
            raise ValueError("The two first new heldout results differ in dual freeze or selected identity")
        if cvalue["dataset_binding"] != report["dataset_binding"]:
            raise ValueError("Crop development/heldout dataset bindings differ")
        held_scene = scene_summary(svalue, "HELD001")
        if scenes and held_scene["dataset_binding"] != scenes[-1]["dataset_binding"]:
            raise ValueError("Scene development/heldout dataset bindings differ")
        heldout = {"cpu_crop_heldout": crop_rows(cvalue), "scene_cpu_heldout": held_scene,
                   "heldout_first_use": {"crop": cp, "scene": sp}}
    sources = [{"role": role, "sha256": source_hashes[role]} for role in files]
    budget = inputs["gpu_budget_snapshot"]
    summary = {
        "schema_version": "localocr.quality-optimization.anonymous-evaluation.v1",
        "round": "quality-optimization-20261012", "split": "separate-dev-and-heldout" if heldout else "dev",
        "heldout_results_included": bool(heldout),
        "production_enabled": False, "promote": False,
        "scope": "Synthetic native training observations, crop CPU evaluation and explicit original-image CPU pipeline evaluation; populations remain separate; no APP acceptance",
        "normalization": "none; exact Unicode and whitespace",
        "ps_definition": "Unicode general categories P and S only; native symbol_cer is remapped explicitly to ps_cer",
        "dataset_binding": dataset_metadata(report["dataset_binding"], scene=False),
        "sources": sources, "native_short_runs": curves, "cpu_crop_development": rows,
        "scene_cpu_development": scenes, **heldout,
        "budget_snapshot": budget_metadata(budget),
        "notes": ["Every curve point is a recorded development evaluation, not an interpolated checkpoint result.",
                  "Observed exact peak is independent of Pareto checkpoint selection and is not the exported model result.",
                  "RepSVTR before means expanded-head initialization, not the original official unmodified recognizer.",
                  "Native RepSVTR selected evaluation and exported CPU results differ; multi-case export checks do not establish full-dataset identity.",
                  "Error rates may exceed 100 percent when insertion errors exceed reference length.",
                  "No previous-round heldout results are joined to these development curves.",
                  "Model-specific inferred candidates, eligible complete face regions, effective improvements and fusion acceptance have distinct meanings.",
                  "A valid regional edit may retain prefix/suffix bytes of the existing region; accepted edits are not all claimed to cover a complete face.",
                  "Face exact-to-wrong losses are disclosed separately from unchanged ordinary-text and negative guards."]}
    if any(digest(path) != source_hashes[role] for role, path in files.items()):
        raise ValueError("A source changed while its aggregates were being read; rerun after the writer stops")
    if args.validate_only:
        print(json.dumps({"validated": True, "development_scene_reports": len(scenes),
                          "heldout_reports_explicitly_included": bool(heldout), "production_enabled": False}))
        return
    configure_outputs(repo)
    public.mkdir(parents=True, exist_ok=True); (public / "figures").mkdir(exist_ok=True); rendering.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname=str(font_path)); matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 12, "svg.fonttype": "path",
                         "svg.hashsalt": "localocr-quality-20261012", "axes.unicode_minus": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    public_builder = public / "build_charts.py"
    builder_bytes = files["chart_builder_source"].read_bytes()
    if hashlib.sha256(builder_bytes).hexdigest() != source_hashes["chart_builder_source"]:
        raise ValueError("The chart builder source changed before public source publication")
    if public_builder.resolve() != Path(__file__).resolve(): write_bytes(public_builder, builder_bytes)
    write(public / "summary.json", summary)
    GENERATED_SVGS.clear()
    for family, curve in curves.items():
        curve_figure(curve, family, public, rendering)
    for metric in ("exact_accuracy", "cer", "ps_cer"):
        cpu_figure(rows, metric, public, rendering)
        if scenes: scene_metric_figure(scenes, metric, public, rendering)
        if heldout:
            cpu_figure(heldout["cpu_crop_heldout"], metric, public, rendering, "heldout")
            scene_metric_figure([heldout["scene_cpu_heldout"]], metric, public, rendering, "heldout")
    if scenes:
        scene_localization_figure(scenes, public, rendering); scene_routing_table(scenes, public, rendering)
        resource_figure(scenes, public, rendering, scene=True)
    resource_figure(rows, public, rendering, scene=False)
    if heldout:
        scene_localization_figure([heldout["scene_cpu_heldout"]], public, rendering, "heldout")
        scene_routing_table([heldout["scene_cpu_heldout"]], public, rendering, "heldout")
        resource_figure([heldout["scene_cpu_heldout"]], public, rendering, scene=True, split="heldout")
        resource_figure(heldout["cpu_crop_heldout"], public, rendering, scene=False, split="heldout")
    outputs = [{"file": str(path.relative_to(public)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": digest(path)}
               for path in [public / "summary.json", public_builder, *sorted(set(GENERATED_SVGS))]]
    if any(digest(path) != source_hashes[role] for role, path in files.items()):
        raise ValueError("A source changed during chart rendering; no completion receipt may be published")
    receipt = {"schema_version": "localocr.quality-optimization.assets-build.v1", "fixture": False,
               "builder_sha256": source_hashes["chart_builder_source"], "matplotlib_version": matplotlib.__version__,
               "font": {key: cjk[key] for key in ("id", "license", "sha256", "source_commit")},
               "sources": sources, "outputs": outputs,
               "rendered_sizes": {"normal_png_dpi": 110, "narrow_png_width_pixels": 375},
               "privacy": "Aggregate counts and hashes only; no images, labels, predictions, weights, or font files copied"}
    write(public / "summary.build-receipt.json", receipt)
    write(round_root / "reporting" / "assets-build-receipt.json", receipt)
    generated_bytes = sum(path.stat().st_size for directory in (public, round_root / "reporting") for path in directory.rglob("*") if path.is_file())
    if generated_bytes > ALLOCATION_BYTES:
        raise ValueError(f"Reporting allocation exceeded: {generated_bytes}")
    print(json.dumps({"generated_bytes": generated_bytes, "public_summary_sha256": digest(public / "summary.json"),
                      "figure_count": sum(path["file"].endswith(".svg") for path in outputs),
                      "selected_exact": {family: curve["selected"]["exact"] for family, curve in curves.items()},
                      "observed_exact_peak": {family: curve["observed_exact_peak"]["exact"] for family, curve in curves.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
