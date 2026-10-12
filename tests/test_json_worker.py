"""Lossless region serialization regression tests; real inference is separate."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

WORKER = Path(__file__).resolve().parents[1] / "scripts/local-ai-ocr-worker.py"
spec = importlib.util.spec_from_file_location("json_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class JsonWorkerTests(unittest.TestCase):
    def result(self, text=" \te\u0301 Ａ\r\n ", score=0.912345678901234):
        return SimpleNamespace(boxes=[[[-0.25, 2.1], [20.1, 2.0], [20.0, 9.5], [0.0, 9.5]]], txts=[text], scores=[score])

    def test_unicode_whitespace_fractional_geometry_and_raw_score(self):
        result = self.result()
        value = json.loads(worker.serialize_json_result(result, 20, 10, "baseline-v1"))
        self.assertEqual(value["schema_version"], "localocr.result.v1")
        self.assertEqual(value["model_version"], "baseline-v1")
        self.assertEqual(value["image"], {"width": 20, "height": 10})
        region = value["regions"][0]
        self.assertEqual(region["text"], result.txts[0])
        self.assertEqual(region["score"], result.scores[0])
        self.assertEqual(region["box"], result.boxes[0])
        self.assertEqual(region["rect"], {"x": 0, "y": 2, "width": 20, "height": 8})
        self.assertNotIn("characters", region)

    def test_empty_result_has_no_invented_regions(self):
        value = json.loads(worker.serialize_json_result(SimpleNamespace(boxes=None), 10, 20, "v1"))
        self.assertEqual(value["regions"], [])

    def test_nonfinite_score_and_mismatched_vectors_fail(self):
        result = self.result(score=float("nan"))
        with self.assertRaises(ValueError):
            worker.serialize_json_result(result, 20, 10, "v1")
        result = self.result()
        result.txts = []
        with self.assertRaises(ValueError):
            worker.serialize_json_result(result, 20, 10, "v1")

    def test_output_limit_counts_utf8_bytes(self):
        with self.assertRaises(ValueError):
            worker.serialize_json_result(self.result("字" * (3*1024*1024)), 20, 10, "v1")

    def test_offline_child_refuses_socket_creation_tcp_udp_and_dns(self):
        # Pre-hook sockets exercise operations themselves; all destinations are loopback.
        # Hook remains process-local so HTTP frontend networking is unaffected.
        code = r"""
import importlib.util, socket, sys
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
tcp = socket.socket()
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sys.addaudithook(worker.deny_network)
assert isinstance(socket.gethostname(), str)  # Local metadata used by Windows platform.system().
operations = [lambda: socket.socket(),
              lambda: tcp.connect(('127.0.0.1', 9)),
              lambda: tcp.bind(('127.0.0.1', 0)),
              lambda: udp.sendto(b'offline-test', ('127.0.0.1', 9)),
              lambda: socket.getaddrinfo('localhost', 9),
              lambda: socket.gethostbyname('localhost')]
if hasattr(udp, 'sendmsg'):
    operations.append(lambda: udp.sendmsg([b'offline-test'], [], 0, ('127.0.0.1', 9)))
for operation in operations:
    try:
        operation()
    except RuntimeError as error:
        assert str(error) == 'Offline worker forbids network access'
    else:
        raise AssertionError('Network operation was not refused')
tcp.close()
udp.close()
"""
        subprocess.run([sys.executable, "-c", code, str(WORKER)], check=True, timeout=10)

    def test_reread_voting_preserves_selected_string(self):
        selected = "  (●_●)\t"
        self.assertEqual(worker.choose_reread("(•_•)", [(selected, .9), (selected, .8)])[0], selected)


if __name__ == "__main__":
    unittest.main()
