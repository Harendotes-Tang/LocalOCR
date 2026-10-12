"""Build audited Windows x64 CPU entrypoints from a dedicated pinned environment.

Developer operation only: never fetch dependencies or models while building.
The sidecar inventory hashes every shipped file without a recursive self-hash.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import stat
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
FLATBUFFERS_LICENSE_SHA256 = 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
MSVCP_SHA256 = '639342ea9a67c0009122238ce070a8257e2e04d367d627509fec29f8442afb42'
MSVC_LICENSE_SHA256 = 'fcd3649a1ea1ed04d849521aca4950ca3db98b3a69f542b51d94cf71f4f17b64'
MSVC_LICENSE_DOCX_SHA256 = '08651651a7602fc7c0e2763de0fde1ff9f868df2780597cd1775ee9d6441c783'
MSVC_LICENSE_SOURCE = 'https://visualstudio.microsoft.com/wp-content/uploads/2025/10/Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx'
FLATBUFFERS_LICENSE_SOURCE = 'https://raw.githubusercontent.com/google/flatbuffers/7e163021e59cca4f8e1e35a7c828b5c6b7915953/LICENSE'
FORBIDDEN = re.compile(r"(?:^|/)(?:torch|torchvision|torchaudio|tensorflow|fastapi|uvicorn|starlette|fonts?|models?|datasets?|candidate-v4)(?:/|$)|\.(?:onnx|pt|pth|ttf|otf|ttc)$", re.I)
CONTRACT_FILES = ('runtime-v1.md', 'model-bundle-v1.md', 'api-v1.md', 'versioning.md')
LICENSE_NAME = re.compile(r"(?:^|/)(?:LICENSE[^/]*|LICENCE[^/]*|COPYING[^/]*|NOTICE[^/]*|ThirdPartyNotices\.txt|MODEL_LICENSES\.md)$", re.I)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def merge_tree(source, target):
    """Merge two frozen closures only when shared relative paths have equal bytes."""
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Unexpected symlink in frozen closure')
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        destination = target / relative
        if destination.exists():
            if digest(destination) != digest(path):
                raise RuntimeError(f'Conflicting shared runtime file: {relative}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

def audit_payload(root):
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Payload symlinks are forbidden')
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if FORBIDDEN.search(relative):
            raise RuntimeError(f'Non-runtime payload forbidden: {relative}')
        files.append(path)
    return files

def minimal_environment():
    env = os.environ.copy()
    windows = env.get('SystemRoot') or env.get('SYSTEMROOT') or env.get('WINDIR')
    if not windows:
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetWindowsDirectoryW(buffer, len(buffer)):
            raise RuntimeError('Cannot resolve Windows system directory')
        windows = buffer.value
    env['SystemRoot'] = windows
    env['WINDIR'] = windows
    env['PATH'] = str(Path(windows) / 'System32') + os.pathsep + windows
    for name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP'):
        env.pop(name, None)
    return env

def collect_licenses(output, pins, notices=None):
    target = output / 'licenses'
    target.mkdir()
    components = {}
    index = {}
    for name, version in pins.items():
        dist = metadata.distribution(name)
        texts = []
        for relative in dist.files or []:
            original = Path(dist.locate_file(relative))
            if original.is_file():
                index.setdefault(digest(original), []).append({'component': name, 'source': f'https://pypi.org/project/{name}/{version}/', 'source_file': str(relative).replace('\\', '/')})
            filename = str(relative).replace('\\', '/')
            if not LICENSE_NAME.search(filename) or not original.is_file() or original.suffix.lower() in ('.py', '.pyc'):
                continue
            destination = target / re.sub(r'[^A-Za-z0-9_.-]', '_', name + '__' + filename)
            shutil.copyfile(original, destination)
            texts.append(destination.relative_to(output).as_posix())
        if not texts and name == 'flatbuffers':
            external = Path(notices or REPO / 'artifacts/runtime-build-v1/upstream-notices') / 'flatbuffers-25.12.19-LICENSE.txt'
            if not external.is_file() or digest(external) != FLATBUFFERS_LICENSE_SHA256:
                raise RuntimeError('Prepare the pinned upstream FlatBuffers license; build performs no downloads')
            destination = target / external.name
            shutil.copyfile(external, destination)
            texts.append(destination.relative_to(output).as_posix())
            index.setdefault(digest(external), []).append({'component': name, 'source': FLATBUFFERS_LICENSE_SOURCE, 'source_file': external.name})
        components[name] = {'version': version, 'license': dist.metadata.get('License-Expression') or dist.metadata.get('License') or 'See complete included license texts', 'source': f'https://pypi.org/project/{name}/{version}/', 'texts': texts, 'distribution_metadata': {str(rel).replace('\\', '/'): {'sha256': digest(Path(dist.locate_file(rel))), 'bytes': Path(dist.locate_file(rel)).stat().st_size} for rel in (dist.files or []) if Path(str(rel)).name in {'METADATA', 'WHEEL', 'RECORD'} and Path(dist.locate_file(rel)).is_file()}}
    old = json.loads((REPO / 'docs/licenses/ocr/dependency-manifest.json').read_text(encoding='utf-8-sig'))
    for item in old['files']:
        if item.get('origin') != 'repository' or not item['path'].startswith('licenses/'):
            continue
        source = (REPO / item['source_path']).resolve()
        if not source.is_relative_to(REPO) or digest(source) != item['sha256'] or source.stat().st_size != item['bytes']:
            raise RuntimeError('Curated notice hash mismatch')
        destination = output / item['path']
        if not destination.resolve().is_relative_to(target.resolve()):
            raise RuntimeError('Curated notice leaves license folder')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        index.setdefault(digest(source), []).append({'component': 'LocalOCR', 'source': 'hash-checked repository curated license', 'source_file': source.relative_to(REPO).as_posix()})
    for item in old['components']:
        name = item['id']
        if name in ('openssl', 'xz-liblzma', 'zlib', 'microsoft-runtime'):
            texts = ['licenses/' + text for text in item['texts']]
            if not all((output / text).is_file() for text in texts):
                raise RuntimeError(f'Missing native notice for {name}')
            components[name] = {'version': item['version'], 'license': item['licence'], 'source': item.get('source', 'hash-checked curated upstream notice'), 'texts': texts}
    upstream = Path(notices or REPO / 'artifacts/runtime-build-v1/upstream-notices')
    for name, sha in [('Microsoft-Visual-C-V14-2026-LICENSE.txt', MSVC_LICENSE_SHA256), ('Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx', MSVC_LICENSE_DOCX_SHA256)]:
        original = upstream / name
        if digest(original) != sha:
            raise RuntimeError('Prepare the hash-bound full Microsoft V14 license before offline build')
        shutil.copyfile(original, target / name)
        components['microsoft-runtime']['texts'].append('licenses/' + name)
        index.setdefault(sha, []).append({'component': 'microsoft-runtime', 'source': MSVC_LICENSE_SOURCE, 'source_file': name})
    components['microsoft-runtime']['source'] = 'Actual hash-pinned signed Microsoft DLL; V14 runtime publisher license document'
    components['microsoft-runtime']['actual_msvcp_version'] = '14.51.36231.0'
    components['microsoft-runtime']['license_document_source'] = MSVC_LICENSE_SOURCE
    index.setdefault(MSVCP_SHA256, []).append({'component': 'microsoft-runtime', 'source': 'C:/Windows/System32/msvcp140.dll; valid Microsoft Authenticode checked during preparation', 'source_file': 'msvcp140.dll', 'version': '14.51.36231.0', 'publisher': 'Microsoft Windows Software Compatibility Publisher', 'publisher_certificate_thumbprint': '81915C173D7FFCBF49EAA8CF7594696B29A035E1'})
    index.setdefault('456aebccb449fcba35f1109c259ba0aa10923e660f4de812a490d35f37f54fc9', []).append({'component': 'microsoft-runtime', 'source': 'C:/Windows/System32/MSVCP140_1.dll; valid Microsoft Authenticode checked during preparation', 'source_file': 'MSVCP140_1.dll', 'version': '14.51.36231.0', 'publisher': 'Microsoft Windows Software Compatibility Publisher', 'publisher_certificate_thumbprint': 'EC5F0D7EE2327688384B4FDF5D7633553A0D055F'})
    components['flatbuffers']['license_text_source'] = FLATBUFFERS_LICENSE_SOURCE
    components['flatbuffers']['license_text_sha256'] = FLATBUFFERS_LICENSE_SHA256
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    shutil.copyfile(python_license, target / 'Python-LICENSE.txt')
    shutil.copyfile(REPO / 'LICENSE', target / 'LocalOCR-GPL-3.0.txt')
    components['CPython'] = {'version': sys.version.split()[0], 'license': 'PSF-2.0 and bundled third-party licenses', 'source': 'https://www.python.org/downloads/release/python-3119/', 'texts': ['licenses/Python-LICENSE.txt', 'licenses/Python-3.11.9-license.rst', 'licenses/Python-3.11.9-expat-COPYING.txt']}
    import pyexpat
    components['expat'] = {'version': pyexpat.EXPAT_VERSION, 'license': 'MIT', 'source': 'CPython bundled Expat', 'texts': ['licenses/Python-3.11.9-expat-COPYING.txt']}
    from localocr.version import SOFTWARE_VERSION
    components['LocalOCR'] = {'version': SOFTWARE_VERSION, 'license': 'GPL-3.0-or-later', 'source': 'local repository source snapshot', 'texts': ['licenses/LocalOCR-GPL-3.0.txt']}
    # CPython DLLs, extension modules and standard-library inputs are not wheels.
    base = Path(sys.base_prefix)
    for folder in (base, base / 'DLLs', base / 'Lib'):
        paths = folder.iterdir() if folder == base else folder.rglob('*')
        for path in paths:
            if path.is_file() and 'site-packages' not in path.parts:
                index.setdefault(digest(path), []).append({'component': 'CPython', 'source': components['CPython']['source'], 'source_file': path.relative_to(base).as_posix()})
    for name, item in components.items():
        if not item['texts']:
            raise RuntimeError(f'Missing complete license texts for {name}')
    return components, index

def repository_import_closure(files, repo=REPO):
    """Follow statically named repository imports, including imports inside build().

    Namespace packages need no manufactured __init__.py. A named repository
    module missing from the checkout is an error, not an external dependency.
    Dynamic worker loading remains an explicit source_snapshot input.
    """
    seen = set()
    pending = [Path(path) for path in files if Path(path).suffix == '.py']
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise RuntimeError(f'Missing repository build dependency: {path}')
        seen.add(path)
        relative = path.relative_to(repo)
        package = list(relative.with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise RuntimeError(f'Invalid relative repository import in {relative}')
                    names = ['.'.join(package[:len(package) - node.level + 1] + (node.module or '').split('.')).strip('.')]
                else:
                    names = [node.module or '']
            for name in names:
                if name.split('.')[0] not in {'localocr', 'tools'}:
                    continue
                target = repo.joinpath(*name.split('.'))
                module = target.with_suffix('.py')
                initializer = target / '__init__.py'
                if module.is_file():
                    pending.append(module)
                elif initializer.is_file():
                    pending.append(initializer)
                elif not target.is_dir():
                    raise RuntimeError(f'Missing repository build dependency: {name}')
    return seen


def source_files():
    files = [REPO / 'tools/LocalAiOcr/requirements-build.txt', REPO / 'tools/LocalAiOcr/build_runtime.py', REPO / 'tools/LocalAiOcr/build_worker.py', REPO / 'tools/LocalAiOcr/package_runtime.py', REPO / 'LICENSE', REPO / 'scripts/build-localocr-runtime.ps1', *[(REPO / 'docs/contracts' / name) for name in CONTRACT_FILES]]
    for directory in ('localocr', 'scripts'):
        files.extend((REPO / directory).glob('*.py'))
    manifest_path = REPO / 'docs/licenses/ocr/dependency-manifest.json'
    files.append(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    for item in manifest['files']:
        if item.get('origin') == 'repository' and item['path'].startswith('licenses/'):
            path = (REPO / item['source_path']).resolve()
            if not path.is_relative_to(REPO) or not path.is_file():
                raise RuntimeError('Missing repository license build dependency')
            files.append(path)
    # The fixed baseline resource paths are read by the model manager at runtime.
    files.extend((REPO / 'docs/licenses/ocr/local-ai' / name) for name in ('rapidocr__rapidocr-3.10.0.dist-info_licenses_LICENSE', 'rapidocr__rapidocr-3.10.0.dist-info_licenses_MODEL_LICENSES.md'))
    files.extend(repository_import_closure(files))
    return sorted(set(files))


def checked_archive_file(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative or any(part in {'.', '..'} for part in relative.split('/')):
        raise RuntimeError('Invalid archive source path')
    pure = PurePosixPath(relative)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if pure.is_absolute() or pure.as_posix() != relative or any(part.endswith(('.', ' ')) or any(c in part for c in '<>\"|?*') or part.split('.')[0].upper() in reserved for part in pure.parts):
        raise RuntimeError('Invalid archive source path')
    path = REPO
    try:
        for part in pure.parts:
            path = path / part
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise RuntimeError('Archive source reparse points are forbidden')
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError('Archive source must be a file')
    except FileNotFoundError as error:
        raise RuntimeError(f'Missing archive source file: {relative}') from error
    return path


def archive_source_snapshot():
    path = checked_archive_file('provenance/source-snapshot.json')
    if path.stat().st_size > 16 * 1024 * 1024:
        raise RuntimeError('Archive provenance is too large')
    def unique(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise RuntimeError('Duplicate archive provenance key')
            values[key] = value
        return values
    document = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)
    if not isinstance(document, dict) or set(document) != {'schema_version', 'source_snapshot'} or document['schema_version'] != 'localocr.source-snapshot.v1':
        raise RuntimeError('Unsupported archive source provenance schema')
    snapshot = document['source_snapshot']
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('base_commit'), str) or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', snapshot.get('base_commit', '')) or type(snapshot.get('working_tree_dirty')) is not bool:
        raise RuntimeError('Invalid archive source commit/dirty provenance')
    records = snapshot.get('files')
    if not isinstance(records, list) or not records or len(records) > 10000:
        raise RuntimeError('Invalid archive source file inventory')
    declared = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {'path', 'bytes', 'sha256'} or type(record['bytes']) is not int or record['bytes'] < 0 or not isinstance(record['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', record['sha256']):
            raise RuntimeError('Invalid archive source file record')
        source = checked_archive_file(record['path'])
        if record['path'].casefold() in declared:
            raise RuntimeError('Duplicate archive source file path')
        declared.add(record['path'].casefold())
        if source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise RuntimeError(f'Archive source hash/size mismatch: {record["path"]}')
    required = {item.relative_to(REPO).as_posix().casefold() for item in source_files()}
    if not required <= declared:
        raise RuntimeError('Archive inventory omits required build dependencies')
    return {'source_kind': 'archive-snapshot', 'original_source_kind': snapshot.get('source_kind', 'unspecified'), 'base_commit': snapshot['base_commit'], 'working_tree_dirty': snapshot['working_tree_dirty'], 'files': records}


def source_snapshot():
    # An extracted ZIP can live inside someone else's Git checkout. Never ask
    # Git to search its ancestors when this source root has no own .git entry.
    if not os.path.lexists(REPO / '.git'):
        return archive_source_snapshot()
    top = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=REPO, text=True).strip()).resolve()
    if top != REPO:
        raise RuntimeError('Git source root does not match this build source root')
    return {'source_kind': 'git-checkout', 'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), 'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()), 'files': [{'path': p.relative_to(REPO).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in source_files()]}



def frozen_inputs(work, names):
    """Record transformed Python archive inputs from fresh PyInstaller TOCs."""
    collected = {}
    generated = []
    def visit(value):
        if isinstance(value, (list, tuple)):
            if len(value) == 3 and isinstance(value[1], str) and isinstance(value[2], str) and value[2] in {'PYMODULE', 'PYSOURCE', 'PYMODULE-1', 'PYMODULE-2', 'BINARY', 'EXTENSION', 'DATA'}:
                path = Path(value[1])
                if path.is_file():
                    collected[str(path)] = {'name': value[0], 'type': value[2], 'source_file': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
            else:
                for child in value:
                    visit(child)
    for name in names:
        for toc in sorted((work / name / name).glob('*.toc')):
            visit(ast.literal_eval(toc.read_text(encoding='utf-8')))
            generated.append({'path': str(toc), 'sha256': digest(toc), 'bytes': toc.stat().st_size})
        spec = work / (name + '.spec')
        generated.append({'path': str(spec), 'sha256': digest(spec), 'bytes': spec.stat().st_size})
    import PyInstaller
    bootloader = Path(PyInstaller.__file__).parent / 'bootloader/Windows-64bit-intel/run.exe'
    return {'source_inputs': sorted(collected.values(), key=lambda item: item['source_file']), 'build_metadata': generated, 'console_bootloader': {'source_file': str(bootloader), 'bytes': bootloader.stat().st_size, 'sha256': digest(bootloader), 'component': 'pyinstaller'}}


def validate_build_paths(output, work):
    from localocr.bundles import require_local_path, _directory
    if not output.is_absolute() or not work.is_absolute():
        raise RuntimeError('Use absolute paths and a new output directory')
    for path in (output, work):
        require_local_path(path)
        _directory(path, allow_missing=True)
    if output.exists():
        raise RuntimeError('Use a new output directory')


def verify_frozen_versions(output, names):
    from localocr.version import VERSION_TEXT
    version_checks = []
    for name in names:
        executable = str(output / f'{name}.exe')
        subprocess.run([executable, '--help'], env=minimal_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        result = subprocess.run([executable, '--version'], env=minimal_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        if result.stdout.decode('utf-8').replace('\r\n', '\n') != VERSION_TEXT + '\n':
            raise RuntimeError(f'Frozen software version mismatch: {name}')
        version_checks.append({'entrypoint': name + '.exe', 'stdout': VERSION_TEXT + '\n', 'system_only_path': True})
    return version_checks


def build(args):
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8:
        raise RuntimeError('Build requires Windows x64 CPython 3.11')
    requirements = REPO / 'tools/LocalAiOcr/requirements-build.txt'
    pins = {}
    for line in requirements.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, version = line.split('==')
            if metadata.version(name) != version:
                raise RuntimeError(f'Pinned dependency mismatch: {name}')
            pins[name] = version
    output, work = Path(args.output), Path(args.build_dir)
    validate_build_paths(output, work)
    from localocr.bundles import _directory
    work.mkdir(parents=True, exist_ok=True)
    _directory(work)
    import rapidocr
    rapid_root = Path(rapidocr.__file__).parent
    names = {'local-ai-ocr': 'local-ai-ocr-entry.py', 'localocr-models': 'localocr-models-entry.py'}
    before_snapshot = source_snapshot()
    for name, entry in names.items():
        command = [sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', '--onedir', '--noupx', '--name', name, '--distpath', str(work / 'dist'), '--workpath', str(work / name), '--specpath', str(work), '--paths', str(REPO)]
        for module in ('localocr.bundles', 'localocr.runtime_entry', 'localocr.runtime_job', 'rapidocr.main', 'rapidocr.inference_engine.onnxruntime', 'onnxruntime', 'cv2', 'numpy', 'PIL.Image'):
            command += ['--hidden-import', module]
        for dist in ('rapidocr', 'onnxruntime'):
            command += ['--copy-metadata', dist]
        for module in ('torch', 'torchvision', 'torchaudio', 'tensorflow', 'fastapi', 'uvicorn', 'starlette', 'IPython', 'matplotlib', 'scipy', 'pandas', 'pytest', 'rapidocr.inference_engine.pytorch', 'rapidocr.inference_engine.openvino', 'rapidocr.inference_engine.paddle', 'rapidocr.inference_engine.tensorrt', 'rapidocr.inference_engine.mnn'):
            command += ['--exclude-module', module]
        for filename in ('config.yaml', 'default_models.yaml'):
            command += ['--add-data', f'{rapid_root / filename};rapidocr']
        command += ['--add-data', f'{REPO / "scripts/local-ai-ocr-worker.py"};scripts']
        for filename in ('rapidocr__rapidocr-3.10.0.dist-info_licenses_LICENSE', 'rapidocr__rapidocr-3.10.0.dist-info_licenses_MODEL_LICENSES.md'):
            command += ['--add-data', f'{REPO / "docs/licenses/ocr/local-ai" / filename};docs/licenses/ocr/local-ai']
        command.append(str(REPO / 'scripts' / entry))
        env = minimal_environment()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + sys.base_prefix + os.pathsep + env['PATH']
        env['PYTHONHASHSEED'] = '0'
        with (work / f'{name}-build.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    _directory(output, allow_missing=True)
    if source_snapshot()['files'] != before_snapshot['files']:
        raise RuntimeError('Build inputs changed while freezing; repeat from a stable source snapshot')
    output.mkdir()
    _directory(output)
    for name in names:
        merge_tree(work / 'dist' / name, output)
    from tools.LocalAiOcr.build_worker import remove_unused_video_runtime
    removed = remove_unused_video_runtime(output)
    components, origins = collect_licenses(output, pins, args.notices)
    for name in CONTRACT_FILES:
        source = REPO / 'docs/contracts' / name
        destination = output / 'docs/contracts' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        origins.setdefault(digest(source), []).append({'component': 'LocalOCR', 'source': 'repository stable contract', 'source_file': source.relative_to(REPO).as_posix()})
    for record in before_snapshot['files']:
        origins.setdefault(record['sha256'], []).append({'component': 'LocalOCR', 'source': 'exact repository source snapshot', 'source_file': record['path']})
    snapshot = before_snapshot
    files = []
    for path in audit_payload(output):
        relative = path.relative_to(output).as_posix()
        sha = digest(path)
        source = origins.get(sha)
        if path.suffix.lower() in ('.dll', '.pyd') and not source:
            raise RuntimeError(f'Unknown binary origin: {relative}')
        if not source:
            if relative in ('local-ai-ocr.exe', 'localocr-models.exe'):
                source = [{'component': 'PyInstaller+LocalOCR', 'source': 'PyInstaller bootloader, Python modules and audited source snapshot', 'source_file': f'{path.stem}.spec'}]
            elif relative.startswith('licenses/'):
                source = [{'component': 'license-notice', 'source': 'complete installed distribution or hash-checked curated notice', 'source_file': relative}]
            else:
                source = [{'component': 'PyInstaller+runtime', 'source': 'generated/copy output from pinned build closure; source snapshot and distribution inventory', 'source_file': relative}]
        native_licenses = set()
        lower = path.name.lower()
        if 'vcruntime' in lower or 'msvcp' in lower:
            native_licenses.add('microsoft-runtime')
        if any(part in lower for part in ('libcrypto', 'libssl', '_ssl', '_hashlib')):
            native_licenses.add('openssl')
        if '_lzma' in lower:
            native_licenses.add('xz-liblzma')
        if any(part in lower for part in ('pyexpat', '_elementtree')):
            native_licenses.add('expat')
        files.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha, 'origins': source, 'license_components': sorted(native_licenses | ({x['component'] for x in source if x['component'] in components} or (set(components) if path.suffix.lower() in ('.exe', '.zip') else {'LocalOCR', 'CPython'})))})
    from localocr.version import SOFTWARE_VERSION, artifact_names
    canonical_names = artifact_names()
    version_checks = verify_frozen_versions(output, names)
    from localocr.bundles import RUNTIME, COMPATIBILITY, SCHEMA
    inventory = {'schema_version': 'localocr.runtime-inventory.v1', 'platform': 'Windows x64', 'software_version': SOFTWARE_VERSION, 'artifact_names': canonical_names, 'system_only_path_version_checks': version_checks, 'entrypoints': [name + '.exe' for name in names], 'python': sys.version, 'runtime': RUNTIME, 'compatibility': COMPATIBILITY, 'bundle_schema': SCHEMA, 'result_schema': 'localocr.result.v1', 'pins': pins, 'source_snapshot': snapshot, 'frozen_archive_inputs': frozen_inputs(work, names), 'components': components, 'excluded_optional_video_payload': removed, 'system_only_path_help_passed': True, 'files': files, 'total_bytes': sum(x['bytes'] for x in files)}
    sidecar_name = canonical_names['inventory'] if output.name == canonical_names['runtime_dir'] else output.name + '-inventory.json'
    sidecar = output.with_name(sidecar_name)
    sidecar.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'inventory': str(sidecar), 'files': len(files), 'bytes': inventory['total_bytes']}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--notices', help='Prepared pinned upstream license directory; never downloaded by build')
    build(parser.parse_args())
