"""Prepare hash-pinned official research inputs from already-local files.

This module never downloads, imports archive code or deserializes weights. New
receipts describe preparation only: no dataset, environment, GPU lease, running
round or production model is changed. SHA-256 establishes the selected bytes,
not a redistribution permission or a general claim that source code is safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tools.PretrainedOcr import common

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / 'artifacts'
CHUNK = 1024 * 1024
MAX_MEMBERS = 5000
MAX_EXPANDED = 256 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str
    url: str


@dataclass(frozen=True)
class Family:
    key: str
    commit: str
    prefix: str
    archive: Asset
    weight: Asset
    files: int
    license_size: int
    license_sha256: str
    preserved: tuple


FAMILIES = (
    Family('openocr', '1ccfc6ee6161f7133b192e16af3a9265273997ba',
           'OpenOCR-1ccfc6ee6161f7133b192e16af3a9265273997ba',
           Asset('source.zip', 2047647,
                 'c7ffe75b07ffc50f23dd38320e5607db50babf32536e23abcffae6d2fb625334',
                 'https://github.com/Topdu/OpenOCR/archive/1ccfc6ee6161f7133b192e16af3a9265273997ba.zip'),
           Asset('openocr_repsvtr_ch.pth', 65279266,
                 'fbf8fc54a1962c500cd07683408c72476c584c4c33e242e8a706fc9b59a9c0dc',
                 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth'),
           418, 11357, 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
           ('LICENSE', 'tools/utils/ppocr_keys_v1.txt',
            'configs/rec/svtrv2/repsvtr_ch.yml', 'docs/finetune_rec.md', 'tools/utils/ckpt.py')),
    Family('paddleocr', 'b03f46425e8ff4442b268ce449e3eef758146cd4',
           'PaddleOCR-b03f46425e8ff4442b268ce449e3eef758146cd4',
           Asset('PaddleOCR-b03f46425e8ff4442b268ce449e3eef758146cd4.zip', 118686571,
                 '449760f5d29842ab6f9cb02cbfbc1f343a5b40fbfd7105333835d72d6a5c741e',
                 'https://codeload.github.com/PaddlePaddle/PaddleOCR/zip/b03f46425e8ff4442b268ce449e3eef758146cd4'),
           Asset('PP-OCRv6_tiny_rec_pretrained.pdparams', 71528759,
                 '960cb4aa5276e3ac235b7f671fb8c9a7c1c1423617da0f96da66a33d0ed53f84',
                 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_rec_pretrained.pdparams'),
           2488, 11376, '3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e',
           ('LICENSE', 'configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml',
            'ppocr/utils/dict/ppocrv6_tiny_dict.txt')),
)


def _reparse(path):
    value = path.lstat()
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, 'st_file_attributes', 0) & 0x400)


def _component(name):
    if not name or name in ('.', '..') or name[-1:] in (' ', '.'):
        raise ValueError('Unsafe Windows path component')
    if any(ord(c) < 32 or c in '<>:"/\\|?*' for c in name):
        raise ValueError('Unsafe Windows path character')
    base = name.split('.')[0].rstrip(' ').upper()
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$', 'CONIN$', 'CONOUT$'}
    reserved.update(prefix + n for prefix in ('COM', 'LPT') for n in '123456789\u00b9\u00b2\u00b3')
    if base in reserved:
        raise ValueError('Reserved Windows path')


def _local_path(path):
    # Inspect the lexical path before resolve(), which would hide a junction.
    path = Path(path)
    if str(path).startswith(('\\\\', '//')) or '..' in path.parts:
        raise ValueError('Only local paths without traversal are supported')
    absolute = Path(os.path.abspath(path))
    for part in absolute.parts[1:]:
        _component(part)
    for parent in reversed((absolute, *absolute.parents)):
        if os.path.lexists(parent) and _reparse(parent):
            raise ValueError('Reparse paths are unsupported')
    return absolute


def _file(path):
    path = _local_path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Input must be a regular local file')
    return path


def _hash_stream(stream, sink=None, limit=None):
    digest = hashlib.sha256()
    count = 0
    for chunk in iter(lambda: stream.read(CHUNK), b''):
        digest.update(chunk)
        count += len(chunk)
        if limit is not None and count > limit:
            raise ValueError('Input exceeds its pinned byte size')
        if sink is not None:
            sink.write(chunk)
    return count, digest.hexdigest()


def _matches(actual, asset):
    if actual != (asset.size, asset.sha256):
        raise ValueError('Fixed official input bytes differ: ' + asset.name)


def _verify_asset(path, asset):
    path = _file(path)
    if path.stat().st_size != asset.size:
        raise ValueError('Fixed official input size differs: ' + asset.name)
    with path.open('rb') as stream:
        _matches(_hash_stream(stream, limit=asset.size), asset)
    return path


def _member_paths(bundle, family):
    members = bundle.infolist()
    if len(members) > MAX_MEMBERS:
        raise ValueError('Official archive has too many entries')
    entries, explicit, aliases, files = {}, set(), {}, []
    expanded = 0
    for member in members:
        raw = member.orig_filename
        if raw != member.filename or '\\' in raw or ':' in raw or not raw.startswith(family.prefix + '/'):
            raise ValueError('Unsafe official archive path or root')
        is_dir = member.is_dir()
        name = raw[:-1] if is_dir else raw
        parts = name.split('/')
        for part in parts:
            _component(part)
        mode = member.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFDIR if is_dir else stat.S_IFREG):
            raise ValueError('Archive special files are unsupported')
        if member.external_attr & 0x400 or member.flag_bits & 1:
            raise ValueError('Archive reparse or encrypted entries are unsupported')
        for index in range(1, len(parts) + 1):
            parent = '/'.join(parts[:index])
            folded = unicodedata.normalize('NFC', parent).casefold()
            if folded in aliases and aliases[folded] != parent:
                raise ValueError('Archive Windows path alias')
            aliases[folded] = parent
            wanted = 'dir' if index < len(parts) or is_dir else 'file'
            if parent in entries and entries[parent] != wanted:
                raise ValueError('Archive file/directory collision')
            entries[parent] = wanted
        if name in explicit:
            raise ValueError('Duplicate official archive entry')
        explicit.add(name)
        if not is_dir:
            relative = '/'.join(parts[1:])
            if not relative:
                raise ValueError('Archive root must be a directory')
            expanded += member.file_size
            if expanded > MAX_EXPANDED or member.file_size < 0:
                raise ValueError('Official archive expansion exceeds limit')
            files.append((relative, member))
    if len(files) != family.files:
        raise ValueError('Official source file count differs')
    return files


class OwnedOutput:
    """Fresh output with cleanup restricted to entries created by this instance.

    Foreign entries appearing during a failure are preserved. Cleanup checks
    every lexical ancestor again and never traverses a reparse point.
    """
    def __init__(self, path):
        self.root = _local_path(path)
        allowed = _local_path(ARTIFACTS).resolve()
        if self.root == allowed or not self.root.resolve().is_relative_to(allowed):
            raise ValueError('Output must be a new directory inside repository artifacts')
        if os.path.lexists(self.root):
            raise FileExistsError('Output already exists; overwrite is prohibited')
        common.disk_guard(self.root)
        self.files, self.dirs = [], []
        # Ancestors are outside this output's ownership and are never removed.
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _local_path(self.root.parent)
        self.root.mkdir()
        self.dirs.append((self.root, self.root.stat().st_dev, self.root.stat().st_ino))

    def mkdir(self, path):
        path = self.checked(path)
        missing = []
        while not path.exists():
            missing.append(path)
            path = path.parent
        for directory in reversed(missing):
            self.checked(directory).mkdir()
            self.dirs.append((directory, directory.stat().st_dev, directory.stat().st_ino))

    def checked(self, path):
        path = _local_path(path)
        if path != self.root and not path.resolve().is_relative_to(self.root):
            raise ValueError('Owned output path escapes its root')
        return path

    def open(self, path, expected_bytes=0):
        path = self.checked(path)
        common.disk_guard(self.root, minimum=common.MINIMUM_FREE_BYTES + expected_bytes)
        self.mkdir(path.parent)
        stream = path.open('xb')
        identity = os.fstat(stream.fileno())
        self.files.append((path, identity.st_dev, identity.st_ino))
        return stream

    def json(self, name, value):
        raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        with self.open(self.root / name, len(raw)) as stream:
            stream.write(raw)

    def cleanup(self):
        for path, device, inode in reversed(self.files):
            try:
                path = self.checked(path)
                value = path.lstat()
                if stat.S_ISREG(value.st_mode) and (value.st_dev, value.st_ino) == (device, inode):
                    path.unlink()
            except (OSError, ValueError):
                pass
        for path, device, inode in reversed(self.dirs):
            try:
                path = self.checked(path)
                value = path.lstat()
                if (value.st_dev, value.st_ino) == (device, inode):
                    path.rmdir()
            except (OSError, ValueError):
                pass


def _archive(path, family, owner=None, destination=None):
    path = _file(path)
    inventory = []
    with path.open('rb') as stream:
        _matches(_hash_stream(stream, limit=family.archive.size), family.archive)
        stream.seek(0)
        with zipfile.ZipFile(stream) as bundle:
            files = _member_paths(bundle, family)
            for relative, member in files:
                with bundle.open(member) as source:
                    if owner is None:
                        size, sha = _hash_stream(source, limit=member.file_size)
                    else:
                        with owner.open(destination / relative, member.file_size) as sink:
                            size, sha = _hash_stream(source, sink, limit=member.file_size)
                if size != member.file_size:
                    raise ValueError('Archive member size differs')
                inventory.append({'path': relative, 'bytes': size, 'sha256': sha})
        # Detect an external input changed in-place while its members were read.
        stream.seek(0)
        _matches(_hash_stream(stream, limit=family.archive.size), family.archive)
    by_path = {record['path']: record for record in inventory}
    license_record = by_path.get('LICENSE')
    if license_record != {'path': 'LICENSE', 'bytes': family.license_size,
                          'sha256': family.license_sha256}:
        raise ValueError('Full official source license differs')
    if any(name not in by_path for name in family.preserved):
        raise ValueError('Required official source asset missing')
    return sorted(inventory, key=lambda item: item['path'])


def _copy(path, asset, owner, destination):
    path = _file(path)
    with path.open('rb') as source, owner.open(destination, asset.size) as sink:
        _matches(_hash_stream(source, sink, limit=asset.size), asset)


def _record(asset, path=None):
    item = {'url': asset.url, 'bytes': asset.size, 'sha256': asset.sha256}
    if path is not None:
        item['path'] = str(path)
    return item


def _summary(family, inventory):
    return {'source_commit': family.commit,
            'source_archive': _record(family.archive),
            'training_checkpoint': _record(family.weight),
            'extracted_files': len(inventory),
            'expanded_bytes': sum(item['bytes'] for item in inventory),
            'source_license': {'identifier': 'Apache-2.0', 'bytes': family.license_size,
                               'sha256': family.license_sha256},
            'source_file_inventory': inventory,
            'third_party_source_executed': False, 'weight_deserialized': False,
            'training_started': False, 'runtime_baseline_unchanged': True}


def _receipt(family, folder, source, inventory):
    result = _summary(family, inventory)
    result.update(schema_version='localocr.' + ('openocr' if family.key == 'openocr' else 'ppocr') + '-inputs.v1',
                  status='prepared-inputs-only', extracted_root=str(source),
                  source_scope='complete hash-pinned official source archive; never executed by preparer')
    archive = folder / family.archive.name
    weight = folder / family.weight.name
    result['downloads'] = [_record(family.archive, archive), _record(family.weight, weight)]
    result['source_archive']['file'] = str(archive)
    result['training_checkpoint']['file'] = str(weight)
    by_path = {item['path']: item for item in inventory}
    result['preserved_assets'] = [by_path[name] for name in family.preserved]
    if family.key == 'openocr':
        result.update(release_tag='develop0.0.1',
                      release_commit='8c21766aff5a859a6029df28ac5084fd29c55576')
    else:
        result.update(source_tag='v3.7.0', fixed_source_inventory=inventory,
                      adapter_layout='inputs/paddleocr/PaddleOCR-' + family.commit,
                      adapter_fixed_round_unchanged=True,
                      adapter_note='Preparation in a new round does not switch paddle_adapter.ROUND; '
                                   'no running input, environment or adapter configuration is changed.')
        result['source_license']['file'] = str(folder / 'PaddleOCR-SOURCE-LICENSE-APACHE-2.0.txt')
    result['license_scope'] = 'Full source license preserved; preparation grants no weight redistribution permission.'
    return result


def process(paths, output, inspect_only=False):
    """Verify four pinned inputs; inspect or copy/extract into a fresh owned root."""
    if set(paths) != {family.key for family in FAMILIES}:
        raise ValueError('Exactly the two fixed input families are required')
    verified, inventories = {}, {}
    for family in FAMILIES:
        archive, weight = paths[family.key]
        archive = _verify_asset(archive, family.archive)
        weight = _verify_asset(weight, family.weight)
        inventories[family.key] = _archive(archive, family)
        verified[family.key] = (archive, weight)
    owner = OwnedOutput(output)
    try:
        results = {}
        for family in FAMILIES:
            archive, weight = verified[family.key]
            if inspect_only:
                results[family.key] = _summary(family, inventories[family.key])
                continue
            folder = owner.root / 'inputs' / family.key
            target = folder / family.archive.name
            _copy(archive, family.archive, owner, target)
            _copy(weight, family.weight, owner, folder / family.weight.name)
            source = folder / ('source' if family.key == 'openocr' else family.prefix)
            owner.mkdir(source)
            inventory = _archive(target, family, owner, source)
            if inventory != inventories[family.key]:
                raise ValueError('Copied official source inventory differs')
            if family.key == 'paddleocr':
                with (source / 'LICENSE').open('rb') as license_stream, owner.open(
                        folder / 'PaddleOCR-SOURCE-LICENSE-APACHE-2.0.txt', family.license_size) as sink:
                    _matches(_hash_stream(license_stream, sink, limit=family.license_size),
                             Asset('LICENSE', family.license_size, family.license_sha256, ''))
            receipt = _receipt(family, folder, source, inventory)
            name = 'research/' + ('openocr' if family.key == 'openocr' else 'ppocr') + '-inputs.json'
            owner.json(name, receipt)
            results[family.key] = {'receipt': name, 'receipt_sha256': common.digest(owner.root / name),
                                  'source_commit': family.commit, 'extracted_files': len(inventory)}
        # Inspect also rechecks original assets immediately before its receipt.
        for family in FAMILIES:
            for path, asset in zip(verified[family.key], (family.archive, family.weight)):
                _verify_asset(path, asset)
        summary = {'schema_version': 'localocr.pretrained.input-preparation.v1',
                   'created_at_utc': common.utc(), 'mode': 'inspect' if inspect_only else 'prepare',
                   'offline_only': True, 'families': results, 'minimum_free_bytes': common.MINIMUM_FREE_BYTES,
                   'existing_inputs_modified': False, 'datasets_consumed': False,
                   'third_party_source_executed': False, 'weight_deserialized': False,
                   'gpu_used': False, 'production_model_changed': False,
                   'paddle_fixed_round_changed': False}
        owner.json('inspection.json' if inspect_only else 'inputs.json', summary)
        return summary
    except BaseException:
        owner.cleanup()
        raise


def _receipt_path(value):
    path = Path(value)
    return _file(path if path.is_absolute() else REPO / path)


def _verify_tree(root, inventory):
    root = _local_path(root)
    expected = {item['path']: item for item in inventory}
    actual = {}
    def visit(folder):
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.path)
                if _reparse(path):
                    raise ValueError('Extracted source reparse entry')
                if entry.is_dir(follow_symlinks=False):
                    _component(path.name)
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    relative = path.relative_to(root).as_posix()
                    if path.suffix == '.pyc' and '__pycache__' in path.relative_to(root).parts:
                        continue
                    _component(path.name)
                    if relative not in expected or path.stat().st_size != expected[relative]['bytes']:
                        raise ValueError('Extracted source closure or size differs')
                    with path.open('rb') as stream:
                        size, sha = _hash_stream(stream, limit=expected[relative]['bytes'])
                    actual[relative] = {'path': relative, 'bytes': size, 'sha256': sha}
                else:
                    raise ValueError('Extracted source special entry')
    visit(root)
    if actual != expected:
        raise ValueError('Extracted source closure or bytes differ')


def check_existing(openocr_receipt, paddle_receipt):
    """Read-only verification; never creates or replaces an existing receipt."""
    summaries = {}
    for family, receipt_path in zip(FAMILIES, (openocr_receipt, paddle_receipt)):
        data = common.read(_file(receipt_path))
        if data.get('source_commit') != family.commit:
            raise ValueError('Unapproved source commit in receipt')
        if family.key == 'openocr':
            downloads = data['downloads']
            archive = next(x['path'] for x in downloads if Path(x['path']).name == family.archive.name)
            weight = next(x['path'] for x in downloads if Path(x['path']).name == family.weight.name)
            source = data['extracted_root']
        else:
            archive, weight = data['source_archive']['file'], data['training_checkpoint']['file']
            source = data.get('extracted_root') or str(_receipt_path(archive).parent / family.prefix)
        archive, weight = _receipt_path(archive), _receipt_path(weight)
        _verify_asset(weight, family.weight)
        inventory = _archive(archive, family)
        # Legacy Paddle receipt has only a partial fixed_source_inventory. Its
        # full closure is still derived from the pinned archive, never trusted.
        claimed = data.get('source_file_inventory')
        if family.key == 'openocr' and claimed is None:
            raise ValueError('OpenOCR receipt requires the full source inventory')
        if claimed is not None:
            mapping = {}
            for item in claimed:
                if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'} or item['path'] in mapping:
                    raise ValueError('Duplicate or malformed receipt inventory')
                mapping[item['path']] = item
            if mapping != {item['path']: item for item in inventory}:
                raise ValueError('Receipt inventory differs from pinned archive')
        _verify_tree(source, inventory)
        if family.key == 'paddleocr':
            license_path = _receipt_path(data['source_license']['file'])
            _verify_asset(license_path, Asset('LICENSE', family.license_size, family.license_sha256, ''))
        summaries[family.key] = {'source_commit': family.commit, 'source_archive_sha256': family.archive.sha256,
                                'weight_sha256': family.weight.sha256, 'source_files': len(inventory)}
    return {'schema_version': 'localocr.pretrained.input-check.v1', 'verified': True,
            'read_only': True, 'families': summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'inspect'):
        command = actions.add_parser(name, help='Fresh exclusive output; inspect writes anonymous metadata only')
        for key in ('openocr-archive', 'openocr-weight', 'paddle-archive', 'paddle-weight'):
            command.add_argument('--' + key, required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
    command = actions.add_parser('check-existing', help='Verify existing sources and receipts without rewriting')
    command.add_argument('--openocr-receipt', required=True, type=Path)
    command.add_argument('--paddle-receipt', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == 'check-existing':
        summary = check_existing(args.openocr_receipt, args.paddle_receipt)
    else:
        paths = {'openocr': (args.openocr_archive, args.openocr_weight),
                 'paddleocr': (args.paddle_archive, args.paddle_weight)}
        summary = process(paths, args.output, inspect_only=args.action == 'inspect')
    print(json.dumps({'verified': True, 'mode': args.action,
                      'families': {key: value.get('source_commit') for key, value in summary['families'].items()}},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
