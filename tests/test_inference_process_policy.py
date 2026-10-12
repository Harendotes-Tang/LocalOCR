"""Fresh-process checks for the inference core's network/process audit boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/local-ai-ocr-worker.py"
BUILD_PYTHON = ROOT / "artifacts/runtime-build-env/Scripts/python.exe"
PROCESS_ERROR = "Offline worker forbids process execution"
PRELUDE = r"""
import importlib.util, json, os, pathlib, socket, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("policy_worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
exec(compile(pathlib.Path(sys.argv[1]).read_bytes(), sys.argv[1], "exec"), worker.__dict__)
"""


def run_child(code, tmp_path, *args, python=None):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run(
        [str(python or sys.executable), "-B", "-I", "-c", PRELUDE + code,
         str(WORKER), *map(str, args)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("frontend", ["subprocess", "system", "spawn"])
def test_real_process_frontends_are_refused_before_marker_execution(tmp_path, frontend):
    marker = tmp_path / "unexpected-child.txt"
    result = run_child(r"""
import shlex
sys.addaudithook(worker.deny_network)
marker = pathlib.Path(sys.argv[3])
command = [sys.executable, "-B", "-I", "-c",
           "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('executed')"]
try:
    if sys.argv[2] == "subprocess":
        subprocess.run(command, check=True)
    elif sys.argv[2] == "system":
        os.system(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command))
    else:
        os.spawnv(os.P_WAIT, sys.executable, command)
except OSError as error:
    assert str(error) == "Offline worker forbids process execution", str(error)
else:
    raise AssertionError("Process frontend was not refused")
assert not marker.exists()
print(json.dumps({"denied": sys.argv[2], "marker_exists": marker.exists()}))
""", tmp_path, frontend, marker)
    assert result == {"denied": frontend, "marker_exists": False}
    assert not marker.exists()


def test_remaining_audited_execution_events_are_refused_on_every_platform(tmp_path):
    # Portable audit emission also covers APIs absent on the test OS. Native
    # frontends above and the CreateProcess guard below establish execution order.
    result = run_child(r"""
sys.addaudithook(worker.deny_network)
events = ("os.posix_spawn", "os.exec", "os.fork", "os.forkpty",
          "os.startfile", "os.startfile/2")
for event in events:
    try:
        sys.audit(event, "process-policy-probe")
    except OSError as error:
        assert str(error) == "Offline worker forbids process execution"
    else:
        raise AssertionError("Execution event was not refused: " + event)
print(json.dumps({"denied": list(events)}))
""", tmp_path)
    assert len(result["denied"]) == 6


def test_core_installs_policy_before_validation_and_native_process_creation(tmp_path):
    result = run_child(r"""
native_calls = []
if os.name == "nt":
    import _winapi
    def unexpected_create(*args, **kwargs):
        native_calls.append("CreateProcess")
        raise AssertionError("Execution reached native CreateProcess")
    _winapi.CreateProcess = unexpected_create
else:
    def unexpected_create(*args, **kwargs):
        native_calls.append("fork_exec")
        raise AssertionError("Execution reached native fork_exec")
    subprocess._fork_exec = unexpected_create
try:
    worker.create_engine(pathlib.Path(sys.argv[2]))
except ValueError as error:
    assert str(error) == "OCR model is missing or incomplete"
else:
    raise AssertionError("Missing models unexpectedly passed validation")
try:
    subprocess.Popen([sys.executable, "-B", "-I", "-c", "pass"])
except OSError as error:
    assert str(error) == "Offline worker forbids process execution"
else:
    raise AssertionError("Core did not install its process policy")
assert not native_calls
print(json.dumps({"native_calls": native_calls, "denied": True}))
""", tmp_path, tmp_path / "absent-models")
    assert result == {"native_calls": [], "denied": True}


def test_fresh_real_dependencies_import_with_native_platform_fallback(tmp_path):
    python = BUILD_PYTHON if BUILD_PYTHON.is_file() else Path(sys.executable)
    result = run_child(r"""
attempts, native_calls, network_events = [], [], []
def observe(event, args):
    if event == "subprocess.Popen":
        attempts.append({"executable": str(args[0]), "command": str(args[1])})
    if event.startswith("socket."):
        network_events.append(event)
sys.addaudithook(observe)
sys.addaudithook(worker.deny_network)
if os.name == "nt":
    import _winapi
    def unexpected_create(*args, **kwargs):
        native_calls.append("CreateProcess")
        raise AssertionError("Dependency import reached native CreateProcess")
    _winapi.CreateProcess = unexpected_create
else:
    def unexpected_create(*args, **kwargs):
        native_calls.append("fork_exec")
        raise AssertionError("Dependency import reached native fork_exec")
    subprocess._fork_exec = unexpected_create
import cv2
from rapidocr import RapidOCR
import onnxruntime, platform
assert RapidOCR.__module__ == "rapidocr.main"
assert onnxruntime.__version__ == "1.31.0"
if sys.platform == "win32":
    native_version = tuple(sys.getwindowsversion().platform_version)
    assert tuple(map(int, platform.version().split("."))) == native_version
    # CPython 3.11 performs three denied ver probes, then reads native metadata.
    if sys.version_info[:2] == (3, 11):
        assert len(attempts) == 3, attempts
        assert [entry["command"].split(' /c "', 1)[1] for entry in attempts] == [
            'ver"', 'command /c ver"', 'cmd /c ver"']
else:
    native_version = os.uname().version if hasattr(os, "uname") else None
    if native_version is not None:
        assert platform.version() == native_version
assert not native_calls
# This observer records attempts before the rejecting hook. A dependency may
# catch a denied socket creation; only the local hostname query is permitted.
try:
    socket.socket()
except RuntimeError as error:
    assert str(error) == "Offline worker forbids network access"
else:
    raise AssertionError("Dependency import disabled the network policy")
assert isinstance(socket.gethostname(), str)
print(json.dumps({"platform": platform.system(), "version": platform.version(),
                  "native_version": native_version, "attempts": attempts,
                  "native_calls": native_calls, "network_events": network_events}))
""", tmp_path, python=python)
    assert result["native_calls"] == []
    assert result["platform"]


def test_existing_socket_denials_and_local_hostname_permission_are_preserved(tmp_path):
    result = run_child(r"""
tcp = socket.socket()
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sys.addaudithook(worker.deny_network)
assert isinstance(socket.gethostname(), str)
actions = [lambda: socket.socket(),
           lambda: tcp.connect(("127.0.0.1", 9)),
           lambda: tcp.bind(("127.0.0.1", 0)),
           lambda: udp.sendto(b"process-policy-probe", ("127.0.0.1", 9)),
           lambda: socket.getaddrinfo("localhost", 9),
           lambda: socket.gethostbyname("localhost")]
if hasattr(udp, "sendmsg"):
    actions.append(lambda: udp.sendmsg([b"process-policy-probe"], [], 0, ("127.0.0.1", 9)))
for action in actions:
    try:
        action()
    except RuntimeError as error:
        assert str(error) == "Offline worker forbids network access"
    else:
        raise AssertionError("Network operation was not refused")
tcp.close()
udp.close()
print(json.dumps({"denied": len(actions), "hostname_allowed": True}))
""", tmp_path)
    assert result["denied"] >= 6
    assert result["hostname_allowed"] is True
