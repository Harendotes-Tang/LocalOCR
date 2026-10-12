"""Tiny offline ZIP fixtures; these tests never load a real model or source tree."""
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tools.PretrainedOcr import inputs, openocr_adapter


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class InputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.paths, families = {}, []
        for family in inputs.FAMILIES:
            archive = self.root / (family.key + '.zip')
            license_raw = b'fixture full license\n'
            entries = {name: (license_raw if name == 'LICENSE' else b'fixture\n') for name in family.preserved}
            # Executing archive modules would produce a clear failure. They are
            # ordinary inert bytes for this preparer, including __init__.py.
            entries['__init__.py'] = b'raise RuntimeError("third-party code executed")\n'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(family.prefix + '/', b'')
                for name, raw in entries.items():
                    bundle.writestr(family.prefix + '/' + name, raw)
            weight = self.root / (family.key + '.weight')
            weight.write_bytes(b'inert, not a serialized checkpoint\n')
            selected = replace(family, archive=replace(family.archive, size=archive.stat().st_size,
                                                       sha256=digest(archive.read_bytes())),
                               weight=replace(family.weight, size=weight.stat().st_size,
                                              sha256=digest(weight.read_bytes())),
                               files=len(entries), license_size=len(license_raw),
                               license_sha256=digest(license_raw))
            families.append(selected)
            self.paths[family.key] = (archive, weight)
        self.families = tuple(families)
        self.patches = [patch.object(inputs, 'ARTIFACTS', self.artifacts),
                        patch.object(inputs, 'REPO', self.root),
                        patch.object(inputs, 'FAMILIES', self.families),
                        patch.object(inputs.common, 'disk_guard', return_value=100 * 1024**3)]
        for item in self.patches:
            item.start()
        self.addCleanup(self.temporary.cleanup)
        for item in self.patches:
            self.addCleanup(item.stop)

    def prepare(self, name='prepared'):
        output = self.artifacts / name
        inputs.process(self.paths, output)
        return output

    def alter_zip(self, records):
        family = self.families[0]
        archive = self.paths['openocr'][0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(archive, 'w') as bundle:
                for name, raw, mode in records:
                    info = zipfile.ZipInfo(name)
                    if mode is not None:
                        info.create_system = 3
                        info.external_attr = mode
                    bundle.writestr(info, raw)
        # Windows ZipInfo rewrites backslashes when writing; mutate the actual
        # serialized names so this negative control tests the incoming bytes.
        serialized = archive.read_bytes()
        for name, _, _ in records:
            if '\\' in name:
                serialized = serialized.replace(name.replace('\\', '/').encode(), name.encode())
        archive.write_bytes(serialized)
        family = replace(family, archive=replace(family.archive, size=archive.stat().st_size,
                                                 sha256=digest(archive.read_bytes())),
                         files=sum(not name.endswith('/') for name, _, _ in records))
        self.families = (family, self.families[1])
        inputs.FAMILIES = self.families
        return family

    def valid_records(self):
        family = self.families[0]
        with zipfile.ZipFile(self.paths['openocr'][0]) as bundle:
            return [(x.filename, bundle.read(x), x.external_attr) for x in bundle.infolist()]

    def test_prepare_receipt_consumed_by_actual_local_openocr_validator(self):
        output = self.prepare()
        family = self.families[0]
        receipt = output / 'research/openocr-inputs.json'
        with patch.multiple(openocr_adapter, COMMIT=family.commit, SOURCE_SHA=family.archive.sha256,
                            SOURCE_BYTES=family.archive.size, WEIGHT_SHA=family.weight.sha256,
                            WEIGHT_BYTES=family.weight.size):
            source, weight, metadata = openocr_adapter.validate_inputs(receipt)
        self.assertEqual(source, output / 'inputs/openocr/source')
        self.assertEqual(weight.read_bytes(), self.paths['openocr'][1].read_bytes())
        self.assertEqual(len(metadata['source_file_inventory']), family.files)
        check = inputs.check_existing(receipt, output / 'research/ppocr-inputs.json')
        self.assertTrue(check['verified'])
        metadata = json.loads(receipt.read_text('utf-8'))
        metadata['source_file_inventory'].reverse()
        receipt.write_text(json.dumps(metadata), encoding='utf-8')
        self.assertTrue(inputs.check_existing(receipt, output / 'research/ppocr-inputs.json')['verified'])
        metadata['source_file_inventory'].append(metadata['source_file_inventory'][0])
        receipt.write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaises(ValueError):
            inputs.check_existing(receipt, output / 'research/ppocr-inputs.json')
        paddle = json.loads((output / 'research/ppocr-inputs.json').read_text('utf-8'))
        self.assertTrue(paddle['adapter_fixed_round_unchanged'])
        self.assertFalse(paddle['third_party_source_executed'])
        self.assertEqual((output / 'inputs/paddleocr/PaddleOCR-SOURCE-LICENSE-APACHE-2.0.txt').read_bytes(),
                         b'fixture full license\n')

    def test_prepare_and_inspect_have_no_network_or_process_launch(self):
        with patch('socket.socket', side_effect=AssertionError('network attempted')), \
                patch('subprocess.Popen', side_effect=AssertionError('process attempted')):
            self.prepare('no-network-prepare')
            inputs.process(self.paths, self.artifacts / 'no-network-inspect', inspect_only=True)

    def test_inspection_has_inventory_but_no_extraction_or_external_paths(self):
        output = self.artifacts / 'inspection'
        result = inputs.process(self.paths, output, inspect_only=True)
        self.assertEqual({p.name for p in output.iterdir()}, {'inspection.json'})
        self.assertEqual(result['families']['openocr']['extracted_files'], self.families[0].files)
        raw = (output / 'inspection.json').read_text('utf-8')
        self.assertNotIn(str(self.root), raw)
        self.assertFalse(result['datasets_consumed'])
        self.assertFalse(result['gpu_used'])

    def test_existing_output_is_never_overwritten(self):
        output = self.artifacts / 'existing'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            inputs.process(self.paths, output)
        self.assertEqual(marker.read_bytes(), b'keep')
        self.assertEqual(list(output.iterdir()), [marker])

    def test_output_outside_artifacts_or_artifacts_itself_rejected(self):
        for output in (self.root / 'elsewhere', self.artifacts):
            with self.subTest(output=output), self.assertRaises(ValueError):
                inputs.process(self.paths, output)
        self.assertFalse((self.root / 'elsewhere').exists())

    def test_unsafe_zip_paths_and_case_aliases_rejected_before_output(self):
        original = self.valid_records()
        prefix = self.families[0].prefix + '/'
        cases = [('traversal', [(prefix + '../escape', b'a', None)]),
                 ('absolute', [('/escape', b'a', None)]),
                 ('drive', [(prefix + 'C:/escape', b'a', None)]),
                 ('backslash', [(prefix + r'a\b', b'a', None)]),
                 ('ADS', [(prefix + 'file:stream', b'a', None)]),
                 ('reserved', [(prefix + 'CoM\u00b9.txt', b'a', None)]),
                 ('device-space-before-extension', [(prefix + 'NUL .txt', b'a', None)]),
                 ('trailingdot', [(prefix + 'a.', b'a', None)]),
                 ('trailingspace', [(prefix + 'a ', b'a', None)]),
                 ('emptycomponent', [(prefix + 'a//b', b'a', None)]),
                 ('control', [(prefix + 'a\x01b', b'a', None)]),
                 ('casealias', [(prefix + 'extra/A', b'a', None), (prefix + 'EXTRA/b', b'b', None)]),
                 ('duplicate', [(prefix + 'extra', b'a', None), (prefix + 'extra', b'b', None)]),
                 ('filedir', [(prefix + 'extra', b'a', None), (prefix + 'extra/b', b'b', None)]),
                 ('symlink', [(prefix + 'link', b'target', (stat.S_IFLNK | 0o777) << 16)]),
                 ('reparse', [(prefix + 'link', b'target', ((stat.S_IFREG | 0o600) << 16) | 0x400)]),
                 ('fifo', [(prefix + 'fifo', b'', (stat.S_IFIFO | 0o600) << 16)])]
        for name, extra in cases:
            with self.subTest(name=name):
                self.alter_zip(original + extra)
                output = self.artifacts / name
                with self.assertRaises(ValueError):
                    inputs.process(self.paths, output)
                self.assertFalse(output.exists())

    def test_fixed_archive_and_weight_hashes_are_not_receipt_supplied(self):
        for key, slot in (('openocr', 0), ('openocr', 1), ('paddleocr', 0), ('paddleocr', 1)):
            with self.subTest(key=key, slot=slot):
                path = self.paths[key][slot]
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaises(ValueError):
                    inputs.process(self.paths, self.artifacts / 'rejected')
                path.write_bytes(original)
        self.assertFalse((self.artifacts / 'rejected').exists())

    def test_missing_openocr_inventory_rejected_but_legacy_paddle_inventory_can_be_derived(self):
        output = self.prepare()
        receipt = output / 'research/openocr-inputs.json'
        paddle = output / 'research/ppocr-inputs.json'
        metadata = json.loads(paddle.read_text('utf-8'))
        del metadata['source_file_inventory']
        paddle.write_text(json.dumps(metadata), encoding='utf-8')
        self.assertTrue(inputs.check_existing(receipt, paddle)['verified'])
        metadata = json.loads(receipt.read_text('utf-8'))
        del metadata['source_file_inventory']
        receipt.write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'full source inventory'):
            inputs.check_existing(receipt, paddle)

    def test_modified_extracted_source_and_receipt_fail_readonly_check(self):
        output = self.prepare()
        receipt = output / 'research/openocr-inputs.json'
        paddle = output / 'research/ppocr-inputs.json'
        source = output / 'inputs/openocr/source/LICENSE'
        old = source.read_bytes()
        source.write_bytes(b'corrupted')
        with self.assertRaises(ValueError):
            inputs.check_existing(receipt, paddle)
        source.write_bytes(old)
        metadata = json.loads(receipt.read_text('utf-8'))
        metadata['source_file_inventory'][0]['sha256'] = '0' * 64
        receipt.write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaises(ValueError):
            inputs.check_existing(receipt, paddle)

    def test_missing_or_altered_license_not_accepted(self):
        records = self.valid_records()
        changed = [(name, b'bad-license' if name.endswith('/LICENSE') else raw, mode)
                   for name, raw, mode in records]
        self.alter_zip(changed)
        with self.assertRaisesRegex(ValueError, 'license'):
            inputs.process(self.paths, self.artifacts / 'badlicense')
        self.assertFalse((self.artifacts / 'badlicense').exists())

    def test_declared_expansion_and_member_limits_rejected(self):
        with patch.object(inputs, 'MAX_MEMBERS', 1), self.assertRaises(ValueError):
            inputs.process(self.paths, self.artifacts / 'many')
        with patch.object(inputs, 'MAX_EXPANDED', 1), self.assertRaises(ValueError):
            inputs.process(self.paths, self.artifacts / 'large')
        self.assertFalse((self.artifacts / 'many').exists())
        self.assertFalse((self.artifacts / 'large').exists())

    def test_disk_guard_failure_before_and_during_writes_cleans_owned_files(self):
        output = self.artifacts / 'disk-low'
        with patch.object(inputs.common, 'disk_guard', side_effect=RuntimeError('reserve')), self.assertRaises(RuntimeError):
            inputs.process(self.paths, output)
        self.assertFalse(output.exists())
        calls = 0
        def guard(path, minimum=inputs.common.MINIMUM_FREE_BYTES):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError('reserve during extraction')
            return minimum
        with patch.object(inputs.common, 'disk_guard', side_effect=guard), self.assertRaises(RuntimeError):
            inputs.process(self.paths, output)
        self.assertFalse(output.exists())

    def test_failed_copy_cleans_only_own_output_and_preserves_foreign_entry(self):
        output = self.artifacts / 'interrupted'
        original_copy = inputs._copy
        def interrupted(path, asset, owner, destination):
            original_copy(path, asset, owner, destination)
            if asset.name == 'openocr_repsvtr_ch.pth':
                (owner.root / 'foreign.txt').write_bytes(b'foreign')
                raise RuntimeError('interrupted')
        with patch.object(inputs, '_copy', side_effect=interrupted), self.assertRaises(RuntimeError):
            inputs.process(self.paths, output)
        self.assertEqual([p.name for p in output.iterdir()], ['foreign.txt'])
        self.assertEqual((output / 'foreign.txt').read_bytes(), b'foreign')
        self.assertTrue(self.paths['openocr'][0].exists())

    def test_license_copy_rejects_truncation_and_same_size_changes_after_extraction(self):
        original_open = inputs.OwnedOutput.open
        for case in ('truncated', 'same-size'):
            with self.subTest(case=case):
                def changed(owner, path, expected_bytes=0):
                    path = Path(path)
                    if path.name == 'PaddleOCR-SOURCE-LICENSE-APACHE-2.0.txt':
                        source = path.parent / self.families[1].prefix / 'LICENSE'
                        raw = source.read_bytes()
                        source.write_bytes(b'' if case == 'truncated' else b'x' * len(raw))
                    return original_open(owner, path, expected_bytes)
                output = self.artifacts / ('license-' + case)
                with patch.object(inputs.OwnedOutput, 'open', new=changed), self.assertRaisesRegex(ValueError, 'LICENSE'):
                    inputs.process(self.paths, output)
                self.assertFalse(output.exists())

    def test_pinned_size_limit_rejects_growth_before_extra_bytes_are_written(self):
        import io
        owner = inputs.OwnedOutput(self.artifacts / 'grew-during-copy')
        path = owner.root / 'bounded-copy.bin'
        with patch.object(inputs, 'CHUNK', 2):
            with owner.open(path, expected_bytes=3) as sink:
                with self.assertRaisesRegex(ValueError, 'pinned byte size'):
                    inputs._hash_stream(io.BytesIO(b'123456789'), sink, limit=3)
        self.assertEqual(path.read_bytes(), b'12')
        owner.cleanup()
        self.assertFalse(owner.root.exists())

    def test_cleanup_preserves_foreign_replacement_of_owned_path(self):
        output = self.artifacts / 'replaced'
        owner = inputs.OwnedOutput(output)
        path = output / 'owned.txt'
        with owner.open(path) as stream:
            stream.write(b'owned')
        path.rename(output / 'moved-by-other.txt')
        path.write_bytes(b'replacement by other')
        owner.cleanup()
        self.assertEqual(path.read_bytes(), b'replacement by other')
        self.assertEqual((output / 'moved-by-other.txt').read_bytes(), b'owned')

    def test_nul_filename_crc_corruption_and_encryption_rejected(self):
        import struct
        original = self.valid_records()
        prefix = self.families[0].prefix + '/'
        self.alter_zip(original + [(prefix + 'aZb', b'payloadunique', None)])
        archive = self.paths['openocr'][0]
        for case in ('nul', 'crc', 'encrypted'):
            with self.subTest(case=case):
                self.alter_zip(original + [(prefix + 'aZb', b'payloadunique', None)])
                raw = bytearray(archive.read_bytes())
                if case == 'nul':
                    raw = raw.replace((prefix + 'aZb').encode(), (prefix + 'a\x00b').encode())
                elif case == 'crc':
                    raw = raw.replace(b'payloadunique', b'payloadWRONG!')
                else:
                    # Set encryption on matching local and central headers.
                    for signature, offset in ((b'PK\x03\x04', 6), (b'PK\x01\x02', 8)):
                        start = 0
                        while True:
                            start = raw.find(signature, start)
                            if start < 0:
                                break
                            flags = struct.unpack_from('<H', raw, start + offset)[0]
                            struct.pack_into('<H', raw, start + offset, flags | 1)
                            start += 4
                archive.write_bytes(raw)
                selected = replace(self.families[0], archive=replace(self.families[0].archive,
                                   size=len(raw), sha256=digest(raw)))
                inputs.FAMILIES = (selected, self.families[1])
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    inputs.process(self.paths, self.artifacts / ('bad-' + case))
                self.assertFalse((self.artifacts / ('bad-' + case)).exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows junction test')
    def test_native_windows_junction_rejected_for_output_and_existing_source(self):
        target = self.root / 'junction-target'
        target.mkdir()
        link = self.artifacts / 'junction'
        environment = dict(os.environ, LOCAL_OCR_TEST_JUNCTION=str(link), LOCAL_OCR_TEST_TARGET=str(target))
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                                 'New-Item -ItemType Junction -Path $env:LOCAL_OCR_TEST_JUNCTION '
                                 '-Target $env:LOCAL_OCR_TEST_TARGET | Out-Null'],
                                env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(inputs._reparse(link))
        with self.assertRaises(ValueError):
            inputs.process(self.paths, link / 'new-output')
        self.assertEqual(list(target.iterdir()), [])
        output = self.prepare()
        source = output / 'inputs/openocr/source'
        source_link = source / 'extra-junction'
        environment['LOCAL_OCR_TEST_JUNCTION'] = str(source_link)
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                                 'New-Item -ItemType Junction -Path $env:LOCAL_OCR_TEST_JUNCTION '
                                 '-Target $env:LOCAL_OCR_TEST_TARGET | Out-Null'],
                                env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(ValueError):
            inputs.check_existing(output / 'research/openocr-inputs.json', output / 'research/ppocr-inputs.json')
        self.assertEqual(list(target.iterdir()), [])

    def test_output_input_and_tree_reparse_points_rejected(self):
        link = self.artifacts / 'link'
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError as error:
            self.skipTest('Symlink privilege unavailable: ' + str(error))
        with self.assertRaises(ValueError):
            inputs.process(self.paths, link / 'out')
        alias = self.root / 'alias.zip'
        alias.symlink_to(self.paths['openocr'][0])
        paths = dict(self.paths)
        paths['openocr'] = (alias, self.paths['openocr'][1])
        with self.assertRaises(ValueError):
            inputs.process(paths, self.artifacts / 'input-link')
        output = self.prepare()
        source = output / 'inputs/openocr/source'
        (source / 'linked.py').symlink_to(self.paths['openocr'][1])
        with self.assertRaises(ValueError):
            inputs.check_existing(output / 'research/openocr-inputs.json', output / 'research/ppocr-inputs.json')

    def test_cli_help_runs_in_subprocess_without_onnx_paddle_torch(self):
        result = subprocess.run([sys.executable, '-m', 'tools.PretrainedOcr.inputs', '--help'],
                                cwd=Path(inputs.__file__).resolve().parents[2], capture_output=True,
                                text=True, timeout=20, check=True)
        self.assertIn('check-existing', result.stdout)
        self.assertIn('inspect', result.stdout)
        code = ('import sys; from tools.PretrainedOcr import inputs; '
                'assert not set(sys.modules).intersection({"torch", "paddle", "onnxruntime", "openrec", "ppocr"})')
        subprocess.run([sys.executable, '-c', code], cwd=Path(inputs.__file__).resolve().parents[2],
                       capture_output=True, text=True, timeout=20, check=True)


if __name__ == '__main__':
    unittest.main()
