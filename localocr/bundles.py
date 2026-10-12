"""Strict offline model packages; activation never runs package-provided code.

Directory packages are immutable once installed. SHA256 establishes integrity,
not trusted provenance; import ONNX models only from a source you trust.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

SCHEMA = "localocr.bundle.v1"
RUNTIME = {"rapidocr": "3.10.0", "onnxruntime": "1.31.0"}
COMPATIBILITY = {
    "architecture": "ppocr-det-cls-rec-v1",
    "dictionary": "onnx-embedded-v1",
    "preprocessing": "rapidocr-3.10.0-default-v1",
    "postprocessing": "localocr-symbol-reread-v1",
}
BASELINE_MODELS = {
    "det": ("PP-OCRv6_det_small.onnx", 9929594, "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "rec": ("PP-OCRv6_rec_small.onnx", 21234383, "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    "cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", 585532, "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
}
VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
MAX_LICENSE_BYTES = 1024 * 1024
WORKER = Path(__file__).resolve().parents[1] / "scripts" / "local-ai-ocr-worker.py"

@dataclass(frozen=True)
class Bundle:
    root: Path
    version: str
    manifest_path: Path
    manifest: dict


def _version(value):
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value) or value in {".", ".."} or value.endswith(".") or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        raise ValueError("Invalid bundle version")
    return value


def _plain(path: Path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Symlinks and reparse points are forbidden")
    return info


def require_local_path(path: Path):
    """Reject UNC/mapped remote paths before callers create runtime state."""
    if path.anchor.startswith("\\\\"):
        raise ValueError("Network filesystem paths are forbidden")
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) == 4:
            raise ValueError("Mapped network drives are forbidden")


def _directory(path: Path, allow_missing=False):
    path = Path(os.path.abspath(path))
    require_local_path(path)
    # Inspect from the root down: a descendant lookup through a junction can
    # already access its target, so reject the junction before that lookup.
    for parent in reversed([path, *path.parents]):
        try:
            info = _plain(parent)
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Expected a directory")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(path: Path, limit=64 * 1024):
    if not stat.S_ISREG(_plain(path).st_mode) or path.stat().st_size > limit:
        raise ValueError("Invalid JSON file")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Unexpected manifest fields")


def _relative(value):
    # Deliberately restrict to a flat ASCII package: no drive, ADS or path traversal.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("Unsafe bundle filename")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        raise ValueError("Reserved bundle filename")
    return value


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(root) -> Bundle:
    root = Path(root).absolute()
    _directory(root)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    _keys(manifest, {"schema_version", "version", "runtime", "compatibility", "models", "licenses"})
    version = _version(manifest["version"])
    if manifest["schema_version"] != SCHEMA or manifest["runtime"] != RUNTIME or manifest["compatibility"] != COMPATIBILITY:
        raise ValueError("Unsupported bundle schema/runtime/compatibility")
    _keys(manifest["models"], {"det", "cls", "rec"})
    if not isinstance(manifest["licenses"], list) or not 1 <= len(manifest["licenses"]) <= 16:
        raise ValueError("Model license declarations are required")
    declared = {"manifest.json"}
    total = 0
    licensed_roles = set()
    for kind, entry in [("model", e) for e in manifest["models"].values()] + [("license", e) for e in manifest["licenses"]]:
        _keys(entry, {"file", "size", "sha256"} if kind == "model" else {"file", "size", "sha256", "license", "models"})
        if kind == "license":
            if entry["license"] != "Apache-2.0" or not isinstance(entry["models"], list) or not entry["models"] or any(role not in {"det", "cls", "rec"} for role in entry["models"]):
                raise ValueError("Unsupported or incomplete model license declaration")
            licensed_roles.update(entry["models"])
        name = _relative(entry["file"])
        if name.casefold() in {n.casefold() for n in declared}:
            raise ValueError("Duplicate package filename")
        suffix = Path(name).suffix.lower()
        if (kind == "model" and suffix != ".onnx") or (kind == "license" and suffix not in {".txt", ".md"}):
            raise ValueError("Unsupported package asset")
        size = entry["size"]
        if type(size) is not int or not 0 < size <= (MAX_MODEL_BYTES if kind == "model" else MAX_LICENSE_BYTES):
            raise ValueError("Invalid asset size")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid asset hash")
        path = root / name
        info = _plain(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size or _digest(path) != digest:
            raise ValueError("Bundle asset integrity failure")
        if kind == "license" and not path.read_text(encoding="utf-8").strip():
            raise ValueError("Empty model license")
        declared.add(name)
        total += size
    if licensed_roles != {"det", "cls", "rec"}:
        raise ValueError("Every model must have a license declaration")
    if total > MAX_PACKAGE_BYTES:
        raise ValueError("Package exceeds limit")
    if {p.name for p in root.iterdir()} != declared:
        raise ValueError("Undeclared package assets")
    return Bundle(root, version, manifest_path, manifest)


def verify_runtime():
    from importlib.metadata import version
    for package, required in RUNTIME.items():
        if version(package) != required:
            raise ValueError("Installed OCR runtime is incompatible")


def probe_bundle(bundle: Bundle, timeout=90):
    """Run actual CPU loading and all-role execution in a network-denying child, bounded to 90 s."""
    if not 0 < timeout <= 90:
        raise ValueError("Probe timeout exceeds limit")
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).with_name("local-ai-ocr.exe")
        _directory(executable.parent)
        if not stat.S_ISREG(_plain(executable).st_mode):
            raise ValueError("Missing standalone OCR executable")
        command = [str(executable)]
    else:
        command = [sys.executable, str(WORKER)]
    command.extend(["--self-test", "--models", str(bundle.root),
                    "--manifest", str(bundle.manifest_path)])
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    except subprocess.TimeoutExpired as error:
        raise ValueError("CPU model probe timed out") from error
    if completed.returncode != 0:
        raise ValueError("CPU model probe failed")


@contextmanager
def _lock(store):
    _directory(store, allow_missing=True)
    store.mkdir(parents=True, exist_ok=True)
    _directory(store)
    lock = store / ".bundle.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Bundle store is busy; inspect stale locks after a crashed installer") from error
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def _write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def create_baseline(source_models, output, version) -> Bundle:
    """Copy only hash-frozen baseline ONNX files and their checked-in license notice."""
    source, output = Path(source_models).absolute(), Path(output).absolute()
    require_local_path(source)
    require_local_path(output)
    _version(version)
    _directory(source)
    _directory(output, allow_missing=True)
    if output.exists():
        raise ValueError("Baseline destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _directory(output.parent)
    stage = Path(tempfile.mkdtemp(prefix=".baseline-", dir=output.parent))
    try:
        models = {}
        for role, (name, size, digest) in BASELINE_MODELS.items():
            path = source / name
            if not stat.S_ISREG(_plain(path).st_mode) or path.stat().st_size != size or _digest(path) != digest:
                raise ValueError("Frozen baseline model integrity failure")
            shutil.copyfile(path, stage / name)
            models[role] = {"file": name, "size": size, "sha256": digest}
        notice = WORKER.parents[1] / "docs/licenses/ocr/local-ai/rapidocr__rapidocr-3.10.0.dist-info_licenses_MODEL_LICENSES.md"
        shutil.copyfile(notice, stage / "MODEL_LICENSES.md")
        licenses = [{"file": "MODEL_LICENSES.md", "size": (stage / "MODEL_LICENSES.md").stat().st_size,
                     "sha256": _digest(stage / "MODEL_LICENSES.md"), "license": "Apache-2.0", "models": ["det", "cls", "rec"]}]
        license_text = WORKER.parents[1] / "docs/licenses/ocr/local-ai/rapidocr__rapidocr-3.10.0.dist-info_licenses_LICENSE"
        shutil.copyfile(license_text, stage / "APACHE_LICENSE.txt")
        licenses.append({"file": "APACHE_LICENSE.txt", "size": (stage / "APACHE_LICENSE.txt").stat().st_size,
                         "sha256": _digest(stage / "APACHE_LICENSE.txt"), "license": "Apache-2.0", "models": ["det", "cls", "rec"]})
        _write_json(stage / "manifest.json", {"schema_version": SCHEMA, "version": version,
                    "runtime": RUNTIME, "compatibility": COMPATIBILITY, "models": models, "licenses": licenses})
        validate_bundle(stage)
        os.rename(stage, output)
        return validate_bundle(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def install_bundle(source, store) -> Bundle:
    source, store = Path(source).absolute(), Path(store).absolute()
    require_local_path(source)
    require_local_path(store)
    original = validate_bundle(source)
    with _lock(store):
        versions = store / "versions"
        _directory(versions, allow_missing=True)
        versions.mkdir(exist_ok=True)
        _directory(versions)
        destination = versions / original.version
        _directory(destination, allow_missing=True)
        if destination.exists():
            raise ValueError("Version is already installed")
        stage = Path(tempfile.mkdtemp(prefix=".install-", dir=store))
        try:
            for file in source.iterdir():
                shutil.copyfile(file, stage / file.name)
            staged = validate_bundle(stage)
            if staged.manifest != original.manifest:
                raise ValueError("Source package changed during import")
            probe_bundle(staged)
            os.rename(stage, destination)
            return validate_bundle(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def activate_bundle(store, version) -> Bundle:
    store = Path(store).absolute()
    _version(version)
    with _lock(store):
        bundle = validate_bundle(store / "versions" / version)
        if bundle.version != version:
            raise ValueError("Installed version identity mismatch")
        probe_bundle(bundle)
        temporary = store / (".active-" + uuid.uuid4().hex + ".json")
        try:
            _write_json(temporary, {"schema_version": "localocr.active.v1", "version": version})
            os.replace(temporary, store / "active.json")
        finally:
            temporary.unlink(missing_ok=True)
        return bundle


def load_active(store) -> Bundle:
    store = Path(store).absolute()
    _directory(store)
    pointer = read_json(store / "active.json")
    _keys(pointer, {"schema_version", "version"})
    if pointer["schema_version"] != "localocr.active.v1":
        raise ValueError("Unsupported active pointer")
    version = _version(pointer["version"])
    bundle = validate_bundle(store / "versions" / version)
    if bundle.version != version:
        raise ValueError("Active version identity mismatch")
    return bundle
