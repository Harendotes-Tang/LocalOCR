"""Protocol boundaries; injected workers prove lifecycle, not OCR quality."""
import base64
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from localocr.service import ApiError, StoreOwner, _request_image, create_app
from localocr.prepare import ImageError, save_png as _save_png


def payload():
    stream = io.BytesIO()
    Image.new('RGB', (40, 20), 'white').save(stream, 'PNG')
    return {'model': 'baseline-test', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}}]}]}


@pytest.fixture
def client(tmp_path):
    bundle = SimpleNamespace(root=tmp_path, version='baseline-test', manifest_path=tmp_path / 'manifest.json')
    def command(source, output, bundle):
        result = {'schema_version': 'localocr.result.v1', 'model_version': bundle.version,
                  'image': {'width': 40, 'height': 20}, 'regions': [], 'score_semantics': 'uncalibrated'}
        return [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2], encoding="utf-8")', str(output), json.dumps(result)]
    app = create_app(tmp_path, worker_command=command)
    with patch('localocr.bundles.load_active', return_value=bundle), TestClient(app, base_url='http://127.0.0.1') as test:
        test.headers['Authorization'] = 'Bearer ' + (tmp_path / 'access-token').read_text().strip()
        yield test, tmp_path, app


def test_models_and_json_result(client):
    test, store, app = client
    assert test.get('/v1/models').json()['data'][0]['id'] == 'baseline-test'
    response = test.post('/v1/chat/completions', json=payload())
    assert response.status_code == 200, response.text
    result = json.loads(response.json()['choices'][0]['message']['content'])
    assert result['image'] == {'width': 40, 'height': 20}
    assert not list(store.glob('request-*'))
    assert app.state.ocr['process'] is None
    assert not app.state.ocr['busy']


@pytest.mark.parametrize('headers,status', [
    ({'Authorization': 'wrong'}, 401), ({'Host': 'evil.example'}, 400),
    ({'Origin': 'https://evil.example'}, 403), ({'Origin': 'null'}, 403),
    ({'Host': '127.0.0.1:99999'}, 400)])
def test_guards(client, headers, status):
    assert client[0].get('/v1/models', headers=headers).status_code == status


def test_docs_not_exposed(client):
    for path in ('/docs', '/redoc', '/openapi.json'):
        assert client[0].get(path).status_code == 404


@pytest.mark.parametrize('mutation', [
    lambda data: data.update(stream=True), lambda data: data.update(temperature=0),
    lambda data: data.update(messages=[]),
    lambda data: data['messages'][0]['content'][0]['image_url'].update(url='https://example.com/image.png'),
    lambda data: data['messages'][0]['content'][0]['image_url'].update(detail='high'),
    lambda data: data['messages'][0]['content'].append(data['messages'][0]['content'][0]),
    lambda data: data['messages'][0]['content'][0]['image_url'].update(url='data:image/png;base64,!!!!')])
def test_unsupported_request(client, mutation):
    data = payload()
    mutation(data)
    assert client[0].post('/v1/chat/completions', json=data).status_code == 400
    assert not list(client[1].glob('request-*'))


def test_duplicate_keys(client):
    response = client[0].post('/v1/chat/completions', content='{"model":"a","model":"b"}', headers={'Content-Type': 'application/json'})
    assert response.status_code == 400


def test_body_and_busy_limits(client):
    test, _, app = client
    assert test.post('/v1/chat/completions', content=b'{}', headers={'Content-Type': 'application/json', 'Content-Length': str(28 * 1024 * 1024 + 1)}).status_code == 413
    app.state.ocr['busy'] = True
    assert test.post('/v1/chat/completions', json=payload()).status_code == 429
    app.state.ocr['busy'] = False


def test_format_and_pixels(tmp_path):
    _, raw, _ = _request_image(payload())
    with pytest.raises(ImageError) as error:
        _save_png(raw, 'JPEG', tmp_path / 'image.png')
    assert error.value.status == 400
    with patch('localocr.prepare.PIXEL_LIMIT', 799):
        with pytest.raises(ImageError) as error:
            _save_png(raw, 'PNG', tmp_path / 'image.png')
    assert error.value.status == 413


def test_decoded_limit():
    with patch('localocr.service.IMAGE_LIMIT', 8):
        with pytest.raises(ApiError) as error:
            _request_image(payload())
    assert error.value.status == 413


def test_owner_excludes_second_instance(tmp_path):
    first, second = StoreOwner(tmp_path), StoreOwner(tmp_path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


@pytest.mark.parametrize('worker,status', [
    ('import time; time.sleep(60)', 504),
    ('import sys; sys.exit(5)', 502),
    ('from pathlib import Path; import sys; Path(sys.argv[1]).write_text("bad json")', 502),
    ('from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b"x" * (8*1024*1024+1))', 502)])
def test_failed_workers_cleanup(tmp_path, worker, status):
    bundle = SimpleNamespace(root=tmp_path, version='baseline-test', manifest_path=tmp_path / 'manifest.json')
    app = create_app(tmp_path, timeout=0.8, worker_command=lambda source, output, bundle: [sys.executable, '-c', worker, str(output)])
    with patch('localocr.bundles.load_active', return_value=bundle), TestClient(app, base_url='http://127.0.0.1') as test:
        test.headers['Authorization'] = 'Bearer ' + (tmp_path / 'access-token').read_text().strip()
        assert test.post('/v1/chat/completions', json=payload()).status_code == status
        assert app.state.ocr['process'] is None
        assert not app.state.ocr['busy']
        assert not list(tmp_path.glob('request-*'))

@pytest.mark.parametrize('timeout', [0, -1, 91, True, float('nan'), float('inf')])
def test_timeout_configuration(tmp_path, timeout):
    with pytest.raises(ValueError):
        create_app(tmp_path, timeout=timeout)


def test_busy_is_checked_before_body_parse(client):
    test, _, app = client
    app.state.ocr['busy'] = True
    try:
        response = test.post('/v1/chat/completions', content='not json', headers={'Content-Type': 'application/json'})
        assert response.status_code == 429
    finally:
        app.state.ocr['busy'] = False


def test_preparation_timeout_kills_decoder(tmp_path):
    import time
    bundle = SimpleNamespace(root=tmp_path, version='baseline-test', manifest_path=tmp_path / 'manifest.json')
    app = create_app(tmp_path, timeout=0.2,
                     prepare_command=lambda *args: [sys.executable, '-c', 'import time; time.sleep(60)'],
                     worker_command=lambda *args: pytest.fail('worker must not start'))
    with patch('localocr.bundles.load_active', return_value=bundle), TestClient(app, base_url='http://127.0.0.1') as test:
        test.headers['Authorization'] = 'Bearer ' + (tmp_path / 'access-token').read_text().strip()
        started = time.monotonic()
        assert test.post('/v1/chat/completions', json=payload()).status_code == 504
        assert time.monotonic() - started < 2
        assert app.state.ocr['process'] is None
    assert not list(tmp_path.glob('request-*'))
    assert not app.state.ocr['busy']


def test_decoder_process_does_not_block_busy_response(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor
    bundle = SimpleNamespace(root=tmp_path, version='baseline-test', manifest_path=tmp_path / 'manifest.json')
    def decoder(source, owned, metadata, image_format):
        return [sys.executable, '-c', 'import time,runpy; time.sleep(.3); runpy.run_module("localocr.prepare",run_name="__main__")',
                '--input', str(source), '--output', str(owned), '--metadata', str(metadata), '--format', image_format]
    result = {'schema_version': 'localocr.result.v1', 'model_version': bundle.version,
              'image': {'width': 40, 'height': 20}, 'regions': [], 'score_semantics': 'uncalibrated'}
    def worker(source, output, bundle):
        return [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])', str(output), json.dumps(result)]
    app = create_app(tmp_path, prepare_command=decoder, worker_command=worker)
    with patch('localocr.bundles.load_active', return_value=bundle), TestClient(app, base_url='http://127.0.0.1') as test, ThreadPoolExecutor(max_workers=1) as pool:
        test.headers['Authorization'] = 'Bearer ' + (tmp_path / 'access-token').read_text().strip()
        pending = pool.submit(test.post, '/v1/chat/completions', json=payload())
        for _ in range(100):
            if app.state.ocr['process'] is not None:
                break
            time.sleep(.01)
        assert app.state.ocr['process'] is not None
        assert test.post('/v1/chat/completions', content='malformed', headers={'Content-Type': 'application/json'}).status_code == 429
        assert pending.result(timeout=3).status_code == 200


def test_asgi_disconnect_reaps_owned_worker(tmp_path):
    import asyncio
    async def exercise():
        bundle = SimpleNamespace(root=tmp_path, version='baseline-test', manifest_path=tmp_path / 'manifest.json')
        app = create_app(tmp_path, worker_command=lambda *args: [sys.executable, '-c', 'import time; time.sleep(60)'])
        queue, messages = asyncio.Queue(), []
        async def receive():
            return await queue.get()
        async def send(message):
            messages.append(message)
        with patch('localocr.bundles.load_active', return_value=bundle):
            async with app.router.lifespan_context(app):
                token = (tmp_path / 'access-token').read_text().strip()
                scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                         'scheme': 'http', 'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
                         'query_string': b'', 'root_path': '', 'server': ('127.0.0.1', 8765), 'client': ('127.0.0.1', 12345),
                         'headers': [(b'host', b'127.0.0.1'), (b'authorization', ('Bearer ' + token).encode()), (b'content-type', b'application/json')]}
                await queue.put({'type': 'http.request', 'body': json.dumps(payload()).encode(), 'more_body': False})
                task = asyncio.create_task(app(scope, receive, send))
                for _ in range(100):
                    if app.state.ocr['process'] is not None:
                        break
                    await asyncio.sleep(0.01)
                process = app.state.ocr['process']
                assert process is not None
                await queue.put({'type': 'http.disconnect'})
                await asyncio.wait_for(task, 2)
                assert process.returncode is not None
                assert app.state.ocr['process'] is None
                assert not app.state.ocr['busy']
                assert not list(tmp_path.glob('request-*'))
                assert messages[0]['status'] == 499
    asyncio.run(exercise())


def test_asgi_slow_upload_deadline(tmp_path):
    import asyncio
    async def exercise():
        app = create_app(tmp_path, timeout=0.05)
        messages = []
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            messages.append(message)
        async with app.router.lifespan_context(app):
            token = (tmp_path / 'access-token').read_text().strip()
            scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                     'scheme': 'http', 'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
                     'query_string': b'', 'root_path': '', 'server': ('127.0.0.1', 8765), 'client': ('127.0.0.1', 12345),
                     'headers': [(b'host', b'127.0.0.1'), (b'authorization', ('Bearer ' + token).encode()), (b'content-type', b'application/json')]}
            await asyncio.wait_for(app(scope, receive, send), 1)
            assert messages[0]['status'] == 504
            assert not app.state.ocr['busy']
    asyncio.run(exercise())

def test_decoder_format_mismatch_returns_explicit_error(client):
    data = payload()
    data['messages'][0]['content'][0]['image_url']['url'] = data['messages'][0]['content'][0]['image_url']['url'].replace('data:image/png;', 'data:image/jpeg;')
    response = client[0].post('/v1/chat/completions', json=data)
    assert response.status_code == 400
    assert not list(client[1].glob('request-*'))


def test_decoder_rejects_animated_png(client):
    stream = io.BytesIO()
    first = Image.new('RGB', (40, 20), 'white')
    first.save(stream, format='PNG', save_all=True, append_images=[Image.new('RGB', (40, 20), 'black')], duration=100, loop=0)
    data = payload()
    data['messages'][0]['content'][0]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    response = client[0].post('/v1/chat/completions', json=data)
    assert response.status_code == 400
    assert not list(client[1].glob('request-*'))


def test_decoder_malformed_image_returns_explicit_error(client):
    data = payload()
    data['messages'][0]['content'][0]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(b'not an image').decode()
    assert client[0].post('/v1/chat/completions', json=data).status_code == 400
    assert not list(client[1].glob('request-*'))


def test_decoder_network_denial_in_subprocess():
    import subprocess
    cases = ['import socket; socket.socket()', 'import socket; socket.getaddrinfo("localhost", 80)',
             'import subprocess; subprocess.Popen(["whoami"])']
    for case in cases:
        code = 'import sys; from localocr.prepare import deny_network; sys.addaudithook(deny_network); ' + case
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'Image decoder networking and process execution are disabled' in result.stderr


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows UNC path semantics')
def test_unc_store_rejected_before_io():
    with pytest.raises(ValueError, match='Network filesystem'):
        create_app(Path(r"\\invalid.example.invalid\share\ocr"))
