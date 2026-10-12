"""Authenticated loopback frontend owning a bounded offline worker lifecycle."""
from __future__ import annotations
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect

BODY_LIMIT = 28 * 1024 * 1024
IMAGE_LIMIT = 20 * 1024 * 1024
PIXEL_LIMIT = 50_000_000
RESULT_LIMIT = 8 * 1024 * 1024

class ApiError(Exception):
    def __init__(self, status, message, code='invalid_request'):
        self.status, self.message, self.code = status, message, code

def _error(status, message, code='invalid_request'):
    return JSONResponse({'error': {'message': message, 'type': code, 'code': code}}, status_code=status)

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result

def _json(raw):
    def constant(value):
        raise ValueError('Non-finite JSON')
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=constant)

def _request_image(data):
    if not isinstance(data, dict) or set(data) - {'model', 'messages', 'stream'}:
        raise ApiError(400, 'Only model, messages and stream=false are supported')
    if not isinstance(data.get('model'), str) or not data['model']:
        raise ApiError(400, 'model is required')
    if 'stream' in data and data['stream'] is not False:
        raise ApiError(400, 'Only stream=false is supported')
    messages = data.get('messages')
    if not isinstance(messages, list) or len(messages) != 1:
        raise ApiError(400, 'Exactly one user message is required')
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] != 'user':
        raise ApiError(400, 'Only one user message with content is supported')
    content = message['content']
    if not isinstance(content, list) or not 1 <= len(content) <= 2:
        raise ApiError(400, 'Content requires one image and optional advisory text')
    images, texts = [], 0
    for item in content:
        if not isinstance(item, dict):
            raise ApiError(400, 'Invalid content part')
        if item.get('type') == 'text' and set(item) == {'type', 'text'} and isinstance(item['text'], str):
            texts += 1
        elif item.get('type') == 'image_url' and set(item) == {'type', 'image_url'}:
            image = item['image_url']
            if not isinstance(image, dict) or set(image) != {'url'} or not isinstance(image['url'], str):
                raise ApiError(400, 'image_url requires only a data URI url')
            images.append(image['url'])
        else:
            raise ApiError(400, 'Unsupported content part or option')
    if len(images) != 1 or texts > 1:
        raise ApiError(400, 'Exactly one image and at most one text are supported')
    uri = images[0]
    prefixes = {'data:image/png;base64,': 'PNG', 'data:image/jpeg;base64,': 'JPEG'}
    prefix = next((value for value in prefixes if uri.startswith(value)), None)
    if prefix is None:
        raise ApiError(400, 'Only Base64 PNG/JPEG data URIs are supported')
    encoded = uri[len(prefix):]
    if len(encoded) > ((IMAGE_LIMIT + 2) // 3) * 4:
        raise ApiError(413, 'Decoded image exceeds 20 MiB')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ApiError(400, 'Invalid Base64 image') from None
    if len(raw) > IMAGE_LIMIT:
        raise ApiError(413, 'Decoded image exceeds 20 MiB')
    return data['model'], raw, prefixes[prefix]


class StoreOwner:
    """OS lock survives crashes without stale-file takeover races."""
    def __init__(self, store):
        self.path, self.file = Path(store) / 'service-owner.lock', None
    def acquire(self):
        from localocr.bundles import _directory, _plain
        _directory(self.path.parent, allow_missing=True)
        try:
            _plain(self.path)
        except FileNotFoundError:
            pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise RuntimeError('This model store already has a service owner') from None
    def release(self):
        if self.file:
            self.file.close()
            self.file = None

def _token(path):
    path = Path(path).absolute()
    from localocr.bundles import require_local_path, _directory, _plain
    require_local_path(path)
    _directory(path.parent, allow_missing=True)
    try:
        _plain(path)
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        token = path.read_text(encoding='ascii').strip()
    else:
        token = secrets.token_urlsafe(32)
        with os.fdopen(descriptor, 'w', encoding='ascii') as file:
            file.write(token + '\n')
    if len(token) < 32 or len(token) > 256 or any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in token):
        raise RuntimeError('Invalid local access token file')
    return token

async def _stop(process):
    if process is not None:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()

def create_app(store, *, token_path=None, timeout=90.0, worker_command=None, prepare_command=None):
    """Test-only worker injection is not exposed through the service CLI."""
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 90:
        raise ValueError('timeout must be greater than zero and at most 90 seconds')
    from localocr.bundles import require_local_path, _directory
    store = Path(store).absolute()
    require_local_path(store)
    _directory(store, allow_missing=True)
    store = store.resolve()
    token_path = Path(token_path).absolute() if token_path else store / 'access-token'
    require_local_path(token_path)
    _directory(token_path.parent, allow_missing=True)
    owner = StoreOwner(store)
    state = {'busy': False, 'process': None, 'token': None, 'stopping': False}
    @asynccontextmanager
    async def lifespan(app):
        owner.acquire()
        try:
            state['token'] = _token(token_path)
            yield
        finally:
            state['stopping'] = True
            await _stop(state['process'])
            owner.release()
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ocr = state
    class AccessGuard:
        # Pure ASGI preserves the original receive/disconnect channel. HTTP
        # middleware implemented with BaseHTTPMiddleware can consume it early.
        def __init__(self, app):
            self.app = app
        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http':
                return await self.app(scope, receive, send)
            headers = Headers(scope=scope)
            hosts = headers.getlist('host')
            host = hosts[0] if len(hosts) == 1 else ''
            pieces = host.split(':')
            response = None
            if pieces[0] != '127.0.0.1' or len(pieces) > 2 or (len(pieces) == 2 and (len(pieces[1]) > 5 or not pieces[1].isascii() or not pieces[1].isdigit() or not 1 <= int(pieces[1]) <= 65535)):
                response = _error(400, 'Host must be 127.0.0.1')
            origins = headers.getlist('origin')
            if response is None and origins and (len(origins) != 1 or origins[0] != 'http://' + host):
                response = _error(403, 'Foreign Origin is forbidden', 'access_denied')
            auth = headers.getlist('authorization')
            expected = 'Bearer ' + (state['token'] or '')
            if response is None and (len(auth) != 1 or not hmac.compare_digest(auth[0].encode(), expected.encode())):
                response = _error(401, 'Valid local Bearer token required', 'authentication_error')
            if response is not None:
                return await response(scope, receive, send)
            return await self.app(scope, receive, send)
    app.add_middleware(AccessGuard)
    def active():
        from localocr.bundles import load_active
        try:
            return load_active(store)
        except (OSError, ValueError):
            raise ApiError(503, 'No valid active model bundle', 'model_unavailable') from None
    @app.get('/v1/models')
    async def models():
        try:
            bundle = await asyncio.to_thread(active)
            return {'object': 'list', 'data': [{'id': bundle.version, 'object': 'model', 'created': 0, 'owned_by': 'localocr'}]}
        except ApiError as error:
            return _error(error.status, error.message, error.code)
    async def prepare(request, deadline, function, *args):
        # Hashing/decoding must not block disconnect/busy checks. The owning
        # coroutine waits for its bounded thread before deleting any temp file.
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            while not task.done():
                if state['stopping'] or request.state.disconnected.is_set():
                    raise ApiError(499, 'Client disconnected', 'request_cancelled')
                if time.monotonic() >= deadline:
                    raise ApiError(504, 'CPU inference deadline exceeded', 'inference_timeout')
                await asyncio.sleep(0.02)
            if time.monotonic() >= deadline:
                raise ApiError(504, 'CPU inference deadline exceeded', 'inference_timeout')
            return task.result()
        finally:
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    async def watch_disconnect(request):
        # One dedicated receiver after the full body avoids cancelled receive
        # polling swallowing events on ASGI implementations with checkpoints.
        while True:
            message = await request.receive()
            if message['type'] == 'http.disconnect':
                request.state.disconnected.set()
                return

    async def read_body(request):
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > BODY_LIMIT:
                raise ApiError(413, 'Request body exceeds 28 MiB')
            body.extend(chunk)
        return body

    async def run_owned(request, deadline, command, result_path, limit=RESULT_LIMIT):
        try:
            process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError:
            raise ApiError(502, 'Owned CPU process could not start', 'worker_error') from None
        state['process'] = process
        try:
            while process.returncode is None:
                if state['stopping'] or request.state.disconnected.is_set():
                    raise ApiError(499, 'Client disconnected', 'request_cancelled')
                if time.monotonic() >= deadline:
                    raise ApiError(504, 'CPU inference deadline exceeded', 'inference_timeout')
                if result_path.exists() and result_path.stat().st_size > limit:
                    raise ApiError(502, 'Owned process result exceeds limit', 'worker_error')
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.02)
                except asyncio.TimeoutError:
                    pass
            if time.monotonic() >= deadline:
                raise ApiError(504, 'CPU inference deadline exceeded', 'inference_timeout')
            return process.returncode
        finally:
            await _stop(process)
            state['process'] = None

    @app.post('/v1/chat/completions')
    async def completion(request: Request):
        acquired = False
        watcher = None
        try:
            if state['busy'] or state['stopping']:
                raise ApiError(429, 'An inference request is already running', 'service_busy')
            state['busy'] = acquired = True
            deadline = time.monotonic() + timeout
            if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/json':
                raise ApiError(415, 'Content-Type must be application/json')
            lengths = request.headers.getlist('content-length')
            if lengths:
                if len(lengths) != 1 or len(lengths[0]) > 9 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise ApiError(400, 'Invalid Content-Length')
                if int(lengths[0]) > BODY_LIMIT:
                    raise ApiError(413, 'Request body exceeds 28 MiB')
            try:
                body = await asyncio.wait_for(read_body(request), timeout=max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                raise ApiError(504, 'Request upload deadline exceeded', 'inference_timeout') from None
            except ClientDisconnect:
                raise ApiError(499, 'Client disconnected', 'request_cancelled') from None
            request.state.disconnected = asyncio.Event()
            watcher = asyncio.create_task(watch_disconnect(request))
            try:
                data = await prepare(request, deadline, _json, body)
            except (ValueError, UnicodeError, RecursionError):
                raise ApiError(400, 'Malformed or duplicate-key JSON') from None
            model, raw, image_format = await prepare(request, deadline, _request_image, data)
            bundle = await prepare(request, deadline, active)
            if model != bundle.version:
                raise ApiError(404, 'Requested model is not active', 'model_not_found')
            with tempfile.TemporaryDirectory(prefix='request-', dir=store) as temporary:
                temporary = Path(temporary)
                source, owned, metadata, output = [temporary / name for name in ('source.bin', 'input.png', 'image.json', 'result.json')]
                source.write_bytes(raw)
                command = prepare_command(source, owned, metadata, image_format) if prepare_command else [
                    sys.executable, '-m', 'localocr.prepare', '--input', str(source), '--output', str(owned),
                    '--metadata', str(metadata), '--format', image_format]
                code = await run_owned(request, deadline, command, metadata, limit=4096)
                if not metadata.is_file() or metadata.stat().st_size > 4096:
                    raise ApiError(502, 'Image decoder result missing or oversized', 'worker_error')
                try:
                    dimensions = _json(metadata.read_bytes())
                    if code != 0:
                        error = dimensions['error']
                        if error['status'] not in (400, 413) or not isinstance(error['message'], str):
                            raise ValueError('Invalid decoder error')
                        raise ApiError(error['status'], error['message'])
                    if set(dimensions) != {'width', 'height'} or any(type(dimensions[key]) is not int or dimensions[key] <= 0 for key in dimensions) or dimensions['width'] * dimensions['height'] > PIXEL_LIMIT or not owned.is_file():
                        raise ValueError('Invalid image decoder result')
                except (ValueError, UnicodeError, KeyError, TypeError, RecursionError):
                    raise ApiError(502, 'Invalid image decoder result', 'worker_error') from None
                command = worker_command(owned, output, bundle) if worker_command else [
                    sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'local-ai-ocr-worker.py'),
                    '--input', str(owned), '--output', str(output), '--models', str(bundle.root),
                    '--manifest', str(bundle.manifest_path), '--format', 'json', '--model-version', bundle.version]
                if await run_owned(request, deadline, command, output) != 0:
                    raise ApiError(502, 'CPU worker failed', 'worker_error')
                if not output.is_file() or output.stat().st_size > RESULT_LIMIT:
                    raise ApiError(502, 'Missing or oversized worker result', 'worker_error')
                try:
                    result = _json(output.read_bytes())
                    if result['schema_version'] != 'localocr.result.v1' or result['model_version'] != bundle.version or result['image'] != dimensions or not isinstance(result['regions'], list) or not isinstance(result['score_semantics'], str):
                        raise ValueError('Invalid result contract')
                    content = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                except (ValueError, UnicodeError, KeyError, TypeError, RecursionError):
                    raise ApiError(502, 'Invalid worker JSON result', 'worker_error') from None
                return {'id': 'chatcmpl-' + secrets.token_hex(12), 'object': 'chat.completion', 'created': int(time.time()),
                        'model': bundle.version, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}]}
        except ApiError as error:
            return _error(error.status, error.message, error.code)
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            if acquired:
                state['busy'] = False
    return app
