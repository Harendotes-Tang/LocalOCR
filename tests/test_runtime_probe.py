"""Frozen model probes use the fixed sibling executable, never a host interpreter."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from localocr import bundles


def test_frozen_probe_uses_sibling(tmp_path):
    manager = tmp_path / 'localocr-models.exe'
    worker = tmp_path / 'local-ai-ocr.exe'
    manager.write_bytes(b'entry')
    worker.write_bytes(b'worker')
    bundle = bundles.Bundle(tmp_path, 'test', tmp_path / 'manifest.json', {})
    with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', str(manager)), patch.object(bundles.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
        bundles.probe_bundle(bundle)
    assert run.call_args.args[0] == [str(worker), '--self-test', '--models', str(tmp_path), '--manifest', str(bundle.manifest_path)]
    assert run.call_args.kwargs['timeout'] == 90


def test_missing_frozen_worker_fails_before_spawn(tmp_path):
    bundle = bundles.Bundle(tmp_path, 'test', tmp_path / 'manifest.json', {})
    with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', str(tmp_path / 'localocr-models.exe')), patch.object(bundles.subprocess, 'run') as run:
        try:
            bundles.probe_bundle(bundle)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing fixed worker accepted')
    run.assert_not_called()
