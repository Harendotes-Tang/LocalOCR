"""Build a pinned Windows CPU-only, one-directory OCR payload.

This is a developer build utility. Network access is never needed by the frozen
runtime; obtaining the pinned build packages is an explicit separate operation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_unused_video_runtime(output):
    """Remove OpenCV's optional video DLL; preserve all still-image dependencies."""
    root = output.resolve(strict=True)
    candidates = [path for path in output.rglob('*') if path.is_file() and 'ffmpeg' in path.name.lower()]
    removed = []
    # Check every candidate before deleting anything; unfamiliar video files
    # require investigation rather than silent removal of a possible dependency.
    for path in candidates:
        relative = path.relative_to(output).as_posix()
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.parent != output / '_internal' / 'cv2'
                or not re.fullmatch(r'opencv_videoio_ffmpeg\d+(?:_\d+)?\.dll', path.name, re.I)):
            raise RuntimeError(f'Unexpected FFmpeg payload: {relative}')
        removed.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    for path in candidates:
        path.unlink()
    return removed


def build(args):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("Windows x64 Python 3.11 is required for this Windows executable")
    import struct
    if struct.calcsize("P") != 8:
        raise RuntimeError("A 64-bit build interpreter is required")
    repo = Path(__file__).resolve().parents[2]
    worker = repo / "scripts" / "local-ai-ocr-worker.py"
    requirements = Path(__file__).with_name("requirements-build.txt")
    pins = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            pins[name] = version
            if importlib.metadata.version(name) != version:
                raise RuntimeError(f"Build dependency version mismatch: {name}")
    source_models, output, build_root = (Path(args.models), Path(args.output), Path(args.build_dir))
    if not all(p.is_absolute() for p in (source_models, output, build_root)) or output.exists():
        raise RuntimeError("Use absolute paths and a new output directory")
    # Reuse the runtime's checksum contract before any model is copied.
    spec = importlib.util.spec_from_file_location("mr_offline_worker", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_models(source_models)
    import rapidocr
    rapid_root = Path(rapidocr.__file__).parent
    dist = build_root / "dist"
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--noupx",
               "--name", "local-ai-ocr", "--distpath", str(dist),
               "--workpath", str(build_root / "work"), "--specpath", str(build_root),
               "--hidden-import", "rapidocr.main", "--hidden-import", "rapidocr.inference_engine.onnxruntime"]
    for name in ("config.yaml", "default_models.yaml"):
        command.extend(["--add-data", f"{rapid_root / name};rapidocr"])
    for name in ("pytorch", "openvino", "paddle", "tensorrt", "mnn"):
        command.extend(["--exclude-module", f"rapidocr.inference_engine.{name}"])
    command.append(str(worker))
    subprocess.run(command, env=environment, check=True)
    shutil.copytree(dist / "local-ai-ocr", output)
    excluded_video = remove_unused_video_runtime(output)
    (output / "models").mkdir()
    for name in module.MODELS:
        shutil.copyfile(source_models / name, output / "models" / name)
    license_dir = output / "licenses"
    license_dir.mkdir()
    components = []
    # Include the license/notice texts for every pinned runtime distribution;
    # build tooling is also recorded, since the PyInstaller bootloader is shipped.
    for name, version in pins.items():
        distribution = importlib.metadata.distribution(name)
        texts = []
        for relative in distribution.files or []:
            filename = str(relative).replace("\\", "/")
            if re.search(r"(?:^|/)(?:LICENSE(?:[^/]*)?|LICENCE(?:[^/]*)?|COPYING(?:[^/]*)?|NOTICE(?:[^/]*)?|ThirdPartyNotices\.txt|MODEL_LICENSES\.md)$", filename, re.I):
                path = Path(distribution.locate_file(relative))
                if not path.is_file() or path.suffix.lower() in {".py", ".pyc"}:
                    continue
                target_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name + "__" + filename)
                shutil.copyfile(path, license_dir / target_name)
                texts.append("licenses/" + target_name)
        components.append({"name": name, "version": version,
                           "licence": distribution.metadata.get("License-Expression") or distribution.metadata.get("License") or "See included license text",
                           "texts": texts, "distribution_root": str(distribution.locate_file(""))})
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise RuntimeError("CPython distribution license is missing")
    shutil.copyfile(python_license, license_dir / "Python-LICENSE.txt")
    components.append({"name": "CPython", "version": sys.version.split()[0], "licence": "PSF-2.0 and bundled third-party notices", "texts": ["licenses/Python-LICENSE.txt"]})
    # Repository-curated notices include CPython/OpenSSL/Expat/xz/zlib and any
    # wheel notices missing from distribution metadata. They are governed by
    # the deployment manifest, independently of this newly built executable.
    manifest_path = repo / "docs" / "licenses" / "ocr" / "dependency-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema_version") == 2:
            for file in manifest.get("files", []):
                if not file.get("path", "").startswith("licenses/") or file.get("origin") != "repository":
                    continue
                source = (repo / file["source_path"]).resolve()
                target = (output / file["path"]).resolve()
                if not source.is_relative_to(repo) or not target.is_relative_to(license_dir):
                    raise RuntimeError("License manifest path leaves its permitted directory")
                if source.stat().st_size != file["bytes"] or digest(source) != file["sha256"]:
                    raise RuntimeError("Repository-curated license does not match its manifest")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    # Verify the output as an end-user process, with no build/host Python on PATH.
    minimal_env = os.environ.copy()
    minimal_env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32") + os.pathsep + os.environ["SystemRoot"]
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        minimal_env.pop(name, None)
    executable = output / "local-ai-ocr.exe"
    result = subprocess.run([str(executable), "--self-test", "--models", str(output / "models")],
                            env=minimal_env, capture_output=True, timeout=30, check=True)
    if result.stdout.strip() != b"Local CPU OCR models and dependencies verified.":
        raise RuntimeError("Unexpected frozen helper self-test output")
    files = [{"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(output.rglob("*")) if path.is_file()]
    inventory = {"schema_version": 1, "worker_source_sha256": digest(worker), "requirements_sha256": digest(requirements),
                 "excluded_optional_video_payload": excluded_video,
                 "python": sys.version, "platform": sys.platform, "self_test_system_only_path": True,
                 "model_contract": {name: {"bytes": values[0], "sha256": values[1]} for name, values in module.MODELS.items()},
                 "components": components, "files": files, "total_bytes": sum(file["bytes"] for file in files)}
    inventory_path = output.parent / (output.name + "-build-inventory.json")
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "inventory": str(inventory_path), "bytes": inventory["total_bytes"], "files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-dir", required=True)
    build(parser.parse_args())
