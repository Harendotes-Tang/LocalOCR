"""Installer safety and atomic activation tests; CPU probes are tested separately."""
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from localocr import bundles


def _varint(value):
    encoded = bytearray()
    while value > 127:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _protobuf_fields(data):
    """Read just the ONNX protobuf wire format without requiring an ONNX package."""
    offset = 0
    def read_integer():
        nonlocal offset
        value, shift = 0, 0
        while True:
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7
    while offset < len(data):
        key = read_integer()
        wire = key & 7
        prefix = _varint(key)
        if wire == 0:
            payload = _varint(read_integer())
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            payload = data[offset:offset+size]
            offset += size
        elif wire == 2:
            size = read_integer()
            payload = data[offset:offset+size]
            offset += size
        else:
            raise AssertionError("Unexpected ONNX protobuf wire type")
        yield key >> 3, wire, prefix, payload


def _replace_nested(data, route, replacement):
    """Replace one nested field while preserving every other graph byte/value."""
    number, wanted_index = route[0]
    seen = 0
    found = False
    encoded = bytearray()
    for field, wire, prefix, payload in _protobuf_fields(data):
        if field == number:
            if seen == wanted_index:
                assert wire == 2
                payload = _replace_nested(payload, route[1:], replacement) if len(route) > 1 else replacement
                found = True
            seen += 1
        encoded.extend(prefix)
        if wire == 2:
            encoded.extend(_varint(len(payload)))
        encoded.extend(payload)
    assert found, "Frozen model tensor metadata route changed"
    return bytes(encoded)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.store = self.root / "store"
        self.manifest = {"schema_version": bundles.SCHEMA, "version": "test-v1", "runtime": bundles.RUNTIME,
                         "compatibility": bundles.COMPATIBILITY, "models": {}, "licenses": []}
        for role in ("det", "cls", "rec"):
            name = role + ".onnx"
            self.manifest["models"][role] = self.asset(name, b"not-a-real-model-" + role.encode())
        license_entry = self.asset("MODEL_LICENSES.md", b"Test fixture model notice; never a real license proof.")
        license_entry.update(license="Apache-2.0", models=["det", "cls", "rec"])
        self.manifest["licenses"] = [license_entry]
        self.save()

    def asset(self, name, data):
        (self.source / name).write_bytes(data)
        return {"file": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def save(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_validation_keeps_version_and_paths(self):
        bundle = bundles.validate_bundle(self.source)
        self.assertEqual(bundle.version, "test-v1")
        self.assertEqual(bundle.manifest_path, self.source / "manifest.json")

    def test_unsafe_version_asset_runtime_and_undeclared_code_fail(self):
        for field, value in [("version", "../outside"), ("runtime", {"rapidocr": "0"})]:
            old = self.manifest[field]
            self.manifest[field] = value
            self.save()
            with self.assertRaises(ValueError):
                bundles.validate_bundle(self.source)
            self.manifest[field] = old
        self.save()
        (self.source / "evil.py").write_text("pass")
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)

    def test_corrupt_model_and_unlicensed_role_fail(self):
        (self.source / "det.onnx").write_bytes(b"corruption")
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)
        self.manifest["models"]["det"] = self.asset("det.onnx", b"fixture")
        self.manifest["licenses"][0]["models"] = ["det"]
        self.save()
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)

    def test_unsafe_asset_path_and_executable_license_are_rejected(self):
        self.manifest["models"]["det"]["file"] = "../det.onnx"
        self.save()
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)
        self.manifest["models"]["det"]["file"] = "det.onnx"
        self.manifest["licenses"][0]["file"] = "notice.py"
        self.save()
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)

    def test_declared_symlink_is_rejected(self):
        path = self.source / "det.onnx"
        target = self.root / "det-target.onnx"
        path.rename(target)
        try:
            path.symlink_to(target)
        except OSError:
            target.rename(path)
            self.skipTest("Symlink creation requires permission on this Windows host")
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)

    def test_duplicate_json_key_fails(self):
        path = self.source / "manifest.json"
        text = path.read_text()
        path.write_text(text.replace('"version":', '"version":"extra","version":'), encoding="utf-8")
        with self.assertRaises(ValueError):
            bundles.validate_bundle(self.source)

    def test_probe_failure_keeps_store_without_active_pointer_and_cleans_stage(self):
        with patch.object(bundles, "probe_bundle", side_effect=ValueError("probe failure")):
            with self.assertRaises(ValueError):
                bundles.install_bundle(self.source, self.store)
        self.assertFalse((self.store / "active.json").exists())
        self.assertEqual(list((self.store / "versions").iterdir()), [])
        self.assertFalse(list(self.store.glob(".install-*")))
        self.assertFalse((self.store / ".bundle.lock").exists())

    def test_activation_failure_preserves_pointer_snapshot_and_rollback(self):
        with patch.object(bundles, "probe_bundle"):
            bundles.install_bundle(self.source, self.store)
            first = bundles.activate_bundle(self.store, "test-v1")
            self.manifest["version"] = "test-v2"
            self.save()
            bundles.install_bundle(self.source, self.store)
            bundles.activate_bundle(self.store, "test-v2")
            self.assertEqual(first.version, "test-v1")
            self.assertTrue(first.root.exists())
        pointer = (self.store / "active.json").read_bytes()
        with patch.object(bundles, "probe_bundle", side_effect=ValueError("failure")):
            with self.assertRaises(ValueError):
                bundles.activate_bundle(self.store, "test-v1")
        self.assertEqual((self.store / "active.json").read_bytes(), pointer)
        with patch.object(bundles, "probe_bundle"):
            bundles.activate_bundle(self.store, "test-v1")
        self.assertEqual(bundles.load_active(self.store).version, "test-v1")

    def test_existing_version_is_never_replaced(self):
        with patch.object(bundles, "probe_bundle"):
            bundles.install_bundle(self.source, self.store)
            with self.assertRaises(ValueError):
                bundles.install_bundle(self.source, self.store)

    def test_installer_lock_blocks_mutation(self):
        self.store.mkdir()
        (self.store / ".bundle.lock").write_text("123")
        with self.assertRaises(ValueError):
            bundles.install_bundle(self.source, self.store)
        self.assertEqual((self.store / ".bundle.lock").read_text(), "123")

    def test_real_probe_rejects_loadable_cls_with_wrong_preprocessing_height(self):
        baseline = Path(__file__).resolve().parents[1] / "artifacts/baseline-package"
        if not baseline.is_dir():
            self.skipTest("Create the local hash-frozen baseline package before real-model regressions")
        valid = bundles.validate_bundle(baseline)
        mutated = self.root / "mutated-baseline"
        shutil.copytree(valid.root, mutated)
        manifest = bundles.read_json(mutated / "manifest.json")
        entry = manifest["models"]["cls"]
        model = mutated / entry["file"]
        # Model.graph -> Graph.input -> ValueInfo.type -> TensorType -> shape -> dimension[2].
        # Model tensors remain unchanged: ORT accepts the metadata but 48-high input fails.
        altered = _replace_nested(model.read_bytes(), [(7, 0), (11, 0), (2, 0), (1, 0), (2, 0), (1, 2)], b"\x08\x20")
        model.write_bytes(altered)
        entry["size"] = len(altered)
        entry["sha256"] = hashlib.sha256(altered).hexdigest()
        (mutated / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # Integrity still validates; execution compatibility must reject the model.
        package = bundles.validate_bundle(mutated)
        with self.assertRaisesRegex(ValueError, "CPU model probe failed"):
            bundles.probe_bundle(package)

    def test_real_probe_rejects_fake_onnx(self):
        with self.assertRaises(ValueError):
            bundles.probe_bundle(bundles.validate_bundle(self.source), timeout=10)


if __name__ == "__main__":
    unittest.main()
