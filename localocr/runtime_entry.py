"""Fixed standalone broker for the existing offline OCR implementation."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import secrets
from pathlib import Path
import subprocess
import sys
import time
from .runtime_job import establish_job
from .version import VERSION_TEXT

MAX_INPUT = 20 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
MAX_SECONDS = 90.0


def worker_path():
    root = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    return root / "scripts" / "local-ai-ocr-worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("localocr_fixed_worker", worker_path())
    module = importlib.util.module_from_spec(spec)
    # Source is an audited runtime resource in a possibly read-only installation.
    # Compile those exact bytes directly: SourceFileLoader.exec_module would
    # consult/create __pycache__, adding unaudited installed files.
    code = compile(worker_path().read_bytes(), str(worker_path()), "exec")
    exec(code, module.__dict__)
    return module


def parser():
    result = argparse.ArgumentParser(description="Offline CPU OCR; fixed local files and bounded lifetime")
    result.add_argument("--version", action="version", version=VERSION_TEXT)
    result.add_argument("--input")
    result.add_argument("--output")
    result.add_argument("--owned-png", action="store_true", help="Caller-validated lossless PNG copy; original source must already satisfy 20 MiB/50M pixels")
    result.add_argument("--models", required=True)
    result.add_argument("--manifest")
    result.add_argument("--model-version")
    result.add_argument("--format", choices=("json", "tsv"), default="tsv")
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--network-audit-self-test", action="store_true")
    result.add_argument("--timeout", type=float, default=MAX_SECONDS)
    result.add_argument("--_runtime-core", action="store_true", help=argparse.SUPPRESS)
    return result


def local_path(value, directory=False, new=False):
    from .bundles import require_local_path, _directory, _plain
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Absolute local paths are required")
    # Remote roots must be rejected before even parent/path existence checks.
    require_local_path(path)
    _directory(path if directory else path.parent)
    if not directory:
        if new:
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ValueError("Output already exists")
        else:
            import stat
            if not stat.S_ISREG(_plain(path).st_mode):
                raise ValueError("Expected a regular file")
    return path


def validate(args):
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= MAX_SECONDS:
        raise ValueError("Timeout must be positive and at most 90 seconds")
    models = local_path(args.models, directory=True)
    if args.manifest:
        manifest = local_path(args.manifest)
        if manifest != models / "manifest.json":
            raise ValueError("Manifest must belong to the supplied models directory")
    elif args.model_version:
        raise ValueError("Model version requires a manifest")
    if args.self_test or args.network_audit_self_test:
        if args.owned_png:
            raise ValueError("Owned PNG mode requires an image, not self-test")
        if args.input or args.output:
            raise ValueError("Self-test does not accept input/output")
        return None, False
    if not args.input or not args.output:
        raise ValueError("Input and output are required")
    source = local_path(args.input)
    output = local_path(args.output, new=True) if args.format == "json" else Path(args.output)
    if args.format == "tsv":
        from .bundles import require_local_path, _directory
        if not output.is_absolute():
            raise ValueError("Absolute local paths are required")
        require_local_path(output)
        _directory(output.parent)
        try:
            output.lstat()
        except FileNotFoundError:
            pass
        else:
            local_path(str(output))
    if source == output:
        raise ValueError("Input and output must be separate")
    limit = load_worker().MAX_OWNED_PNG_BYTES if args.owned_png else MAX_INPUT
    if not 0 < source.stat().st_size <= limit:
        raise ValueError("Owned PNG exceeds worker byte limit or is empty" if args.owned_png else "Input exceeds 20 MiB or is empty")
    partial = output.with_name(output.name + ".partial")
    local_path(str(partial), new=True)
    return output, not output.exists()


def authenticate_core():
    """Only the same fixed broker may invoke the private computation mode.

    The broker sends its nonce on an owned anonymous stdin pipe. Windows also
    verifies the kernel-reported parent PID and its exact executable image;
    merely passing the private CLI flag or environment fields is insufficient.
    """
    token = os.environ.pop("LOCALOCR_CORE_NONCE", "")
    parent = os.environ.pop("LOCALOCR_CORE_PARENT", "")
    if len(token) != 64 or not parent.isdecimal():
        raise ValueError("Private core requires its supervising broker")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class BASIC(ctypes.Structure):
            _fields_ = [("Reserved1", ctypes.c_void_p), ("PebBaseAddress", ctypes.c_void_p),
                        ("Reserved2", ctypes.c_void_p * 2), ("UniqueProcessId", ctypes.c_size_t),
                        ("InheritedFromUniqueProcessId", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        native = ctypes.WinDLL("ntdll")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        native.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
        info = BASIC()
        if native.NtQueryInformationProcess(kernel.GetCurrentProcess(), 0, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ValueError("Cannot verify supervising broker")
        actual_parent = info.InheritedFromUniqueProcessId
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, actual_parent)
        if not handle:
            raise ValueError("Cannot inspect supervising broker")
        try:
            if actual_parent != int(parent):
                # A source venv launcher creates one native intermediary; the
                # frozen EXE never accepts that exception.
                ancestor = BASIC()
                if getattr(sys, "frozen", False) or native.NtQueryInformationProcess(handle, 0, ctypes.byref(ancestor), ctypes.sizeof(ancestor), None) or ancestor.InheritedFromUniqueProcessId != int(parent):
                    raise ValueError("Private core parent identity mismatch")
            name = ctypes.create_unicode_buffer(32768)
            length = wintypes.DWORD(len(name))
            if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
                raise ValueError("Cannot inspect supervising broker executable")
            if os.path.normcase(os.path.realpath(name.value)) != os.path.normcase(os.path.realpath(sys.executable)):
                raise ValueError("Private core requires the same fixed executable broker")
        finally:
            kernel.CloseHandle(handle)
    elif os.getppid() != int(parent):
        raise ValueError("Private core parent identity mismatch")
    if sys.stdin.buffer.readline(66) != token.encode("ascii") + b"\n":
        raise ValueError("Private core handshake failed")


def child_command(argv):
    if getattr(sys, "frozen", False):
        prefix = [sys.executable]
    else:
        prefix = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "local-ai-ocr-entry.py")]
    return [*prefix, "--_runtime-core", *argv]


def network_self_test(worker, args):
    # Pre-existing sockets let us exercise audited connect/sendto/bind, in
    # addition to denial of socket creation after engine audit installation.
    import socket
    tcp = socket.socket()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    engine = worker.create_engine(Path(args.models), Path(args.manifest) if args.manifest else None)
    worker.probe_engine(engine)
    actions = {
        "creation": lambda: socket.socket(),
        "dns": lambda: socket.getaddrinfo("localhost", 9),
        "connect": lambda: tcp.connect(("127.0.0.1", 9)),
        "sendto": lambda: udp.sendto(b"denial probe", ("127.0.0.1", 9)),
        "bind": lambda: udp.bind(("127.0.0.1", 0)),
    }
    denied = []
    try:
        for name, action in actions.items():
            try:
                action()
            except RuntimeError as error:
                if str(error) != "Offline worker forbids network access":
                    raise
                denied.append(name)
            else:
                raise ValueError("Network audit did not deny " + name)
    finally:
        tcp.close()
        udp.close()
    print(json.dumps({"schema_version": "localocr.network-audit.v1", "denied": denied, "cpu_probe": True}))


def verify_png_chunks(source):
    """Stream every chunk CRC, including IEND, and require an exact PNG end."""
    import zlib
    with source.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid owned PNG integrity: signature")
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Invalid owned PNG integrity: missing chunk or IEND")
            length = int.from_bytes(header[:4], "big")
            kind = header[4:]
            checksum = zlib.crc32(kind)
            remaining = length
            while remaining:
                block = stream.read(min(remaining, 65536))
                if not block:
                    raise ValueError("Invalid owned PNG integrity: truncated chunk")
                checksum = zlib.crc32(block, checksum)
                remaining -= len(block)
            expected = stream.read(4)
            if len(expected) != 4 or (checksum & 0xffffffff) != int.from_bytes(expected, "big"):
                raise ValueError("Invalid owned PNG integrity: chunk CRC")
            if kind == b"IEND":
                if length != 0 or stream.read(1):
                    raise ValueError("Invalid owned PNG integrity: trailing data or nonempty IEND")
                return


def verify_owned_png(source, worker):
    """Validate the copy inside the supervised core, before CPU model loading.

    The marker cannot prove source provenance. Header, single frame, pixels and
    PNG checksums are verified here; the existing worker then decodes all pixels.
    """
    from PIL import Image
    try:
        with Image.open(source, formats=["PNG"]) as image:
            width, height = image.size
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Owned input must be a single-frame PNG")
            if width <= 0 or height <= 0 or width * height > worker.MAX_PIXELS:
                raise ValueError("Owned PNG exceeds pixel limit")
            verify_png_chunks(source)
            image.verify()
    except (OSError, SyntaxError) as error:
        raise ValueError("Invalid owned PNG format or integrity") from error


def core(args, argv):
    validate(args)
    worker = load_worker()
    if args.owned_png:
        verify_owned_png(Path(args.input), worker)
    if args.network_audit_self_test:
        network_self_test(worker, args)
        return 0
    for name in ("serialize_result", "serialize_json_result"):
        original = getattr(worker, name)
        def bounded(*values, _original=original, **kw):
            payload = _original(*values, **kw)
            if len(payload) > MAX_OUTPUT:
                raise ValueError("OCR output exceeds 8 MiB")
            return payload
        setattr(worker, name, bounded)
    forwarded = []
    for name in ("input", "output", "models", "manifest", "model_version", "format"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(["--" + name.replace("_", "-"), str(value)])
    if args.self_test:
        forwarded.append("--self-test")
    sys.argv = [str(worker_path()), *forwarded]
    worker.main()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--version"]:
        print(VERSION_TEXT)
        return 0
    started = time.monotonic()
    args = parser().parse_args(argv)
    output = None
    owns_output = False
    process = None
    claim = None
    staged = None
    success = False
    try:
        if args._runtime_core:
            authenticate_core()
            return core(args, argv)
        output, owns_output = validate(args)
        establish_job()
        child_argv = list(argv)
        if output is not None:
            # Exclusive private workspace is also the same-output claim. The
            # core never opens caller-named output/partial paths.
            proposed_claim = output.with_name(output.name + ".runtime-claim")
            proposed_claim.mkdir()
            claim = proposed_claim
            staged = claim / "result"
            for index, value in enumerate(child_argv):
                if value == "--output":
                    child_argv[index + 1] = str(staged)
                    break
                if value.startswith("--output="):
                    child_argv[index] = "--output=" + str(staged)
                    break
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "2"
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        nonce = secrets.token_hex(32)
        environment["LOCALOCR_CORE_NONCE"] = nonce
        environment["LOCALOCR_CORE_PARENT"] = str(os.getpid())
        remaining = args.timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired("validation", args.timeout)
        process = subprocess.Popen(child_command(child_argv), env=environment, close_fds=True,
                                   stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr,
                                   creationflags=0x08000000 if os.name == "nt" else 0)
        process.stdin.write(nonce.encode("ascii") + b"\n")
        process.stdin.close()
        result = process.wait(timeout=max(0.001, args.timeout - (time.monotonic() - started)))
        if result:
            return result
        if staged is not None:
            if not staged.is_file() or staged.stat().st_size > MAX_OUTPUT:
                raise ValueError("Missing or oversized OCR output")
            if args.format == "json":
                # Atomic no-overwrite publication; a concurrently created
                # unrelated output remains untouched even on failure.
                if os.name == "nt":
                    os.rename(staged, output)  # Windows fails if destination exists.
                else:
                    os.link(staged, output)
                    staged.unlink()
            else:
                os.replace(staged, output)
        success = True
        return 0
    except subprocess.TimeoutExpired:
        print("Local OCR deadline exceeded.", file=sys.stderr)
        return 124
    except (ValueError, OSError, RuntimeError) as error:
        print("Local OCR failed: " + str(error), file=sys.stderr)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if staged is not None:
            staged.with_name(staged.name + ".partial").unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
        if claim is not None:
            claim.rmdir()


def models_main():
    if sys.argv[1:] == ["--version"]:
        print(VERSION_TEXT)
        return
    establish_job()
    from .bundle_cli import main as bundle_main
    bundle_main()
