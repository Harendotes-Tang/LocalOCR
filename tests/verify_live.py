"""Real local CPU/SDK and owned lifecycle acceptance; all generated data stays local.

Run with the service interpreter and an already activated trusted baseline store.
This verifies protocol and lifecycle, not screenshot accuracy or APP acceptance.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import httpx
import psutil
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI, DefaultHttpxClient

ROOT = Path(__file__).resolve().parents[1]


def wait_until(check, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError('Timed out waiting for verification condition')


class RunningServer:
    def __init__(self, store, output, port, timeout=90):
        self.store, self.output, self.port = store, output, port
        self.stop = output / ('stop-' + uuid.uuid4().hex)
        self.log = (output / ('server-' + str(port) + '.log')).open('wb')
        # Stop file belongs only to this verification harness; no HTTP admin endpoint.
        code = """import pathlib,threading,time,uvicorn
from localocr.service import create_app
store,stop,port,timeout=__import__('sys').argv[1:]
server=uvicorn.Server(uvicorn.Config(create_app(pathlib.Path(store),timeout=float(timeout)),host='127.0.0.1',port=int(port),log_level='warning',timeout_graceful_shutdown=1))
def watch():
 while not pathlib.Path(stop).exists(): time.sleep(.02)
 server.should_exit=True
threading.Thread(target=watch,daemon=True).start()
server.run()
"""
        self.process = subprocess.Popen([sys.executable, '-c', code, str(store), str(self.stop), str(port), str(timeout)], cwd=ROOT, stdout=self.log, stderr=self.log)
        self.base = f'http://127.0.0.1:{port}'
        self.client = httpx.Client(base_url=self.base, timeout=100, trust_env=False)
        def ready():
            if self.process.poll() is not None:
                raise AssertionError('Service exited; inspect local server log')
            try:
                return self.client.get('/v1/models').status_code == 401
            except httpx.HTTPError:
                return False
        wait_until(ready)
        self.token = (store / 'access-token').read_text().strip()
        self.headers = {'Authorization': 'Bearer ' + self.token}

    def children(self):
        if self.process.poll() is not None:
            return []
        return [p for p in psutil.Process(self.process.pid).children(recursive=True) if p.is_running() and any(arg.endswith('local-ai-ocr-worker.py') or arg == 'localocr.prepare' for arg in p.cmdline())]

    def close(self):
        self.stop.touch()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise AssertionError('Service failed graceful shutdown')
        finally:
            self.client.close()
            self.log.close()
            self.stop.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=18765)
    args = parser.parse_args()
    store, output = args.store.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'scope': 'real CPU/SDK local acceptance; synthetic image, not screenshot accuracy', 'checks': {}}
    image = Image.new('RGB', (700, 130), 'white')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 46)
    ImageDraw.Draw(image).text((25, 30), 'HELLO OCR 123', fill='black', font=font)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    uri = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')
    server = RunningServer(store, output, args.port)
    body = None
    try:
        sdk = OpenAI(base_url=server.base + '/v1', api_key=server.token, http_client=DefaultHttpxClient(trust_env=False), max_retries=0, timeout=95)
        listed = sdk.models.list()
        model = listed.data[0].id
        body = {'model': model, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': uri}}]}], 'stream': False}
        started = time.monotonic()
        completion = sdk.chat.completions.create(**body)
        result = json.loads(completion.choices[0].message.content)
        assert result['schema_version'] == 'localocr.result.v1'
        assert result['image'] == {'width': 700, 'height': 130}
        assert result['regions'] and 'HELLO' in ' '.join(r['text'] for r in result['regions'])
        evidence['checks']['sdk_real_cpu'] = {'model': model, 'elapsed_seconds': time.monotonic()-started, 'result': result}
        sdk.close()
        evidence['checks']['missing_auth'] = server.client.get('/v1/models').status_code
        assert evidence['checks']['missing_auth'] == 401
        for name, headers in [('foreign_host', {**server.headers, 'Host': 'evil.example'}), ('foreign_origin', {**server.headers, 'Origin': 'https://evil.example'})]:
            status = server.client.get('/v1/models', headers=headers).status_code
            assert status in (400, 403)
            evidence['checks'][name] = status
        for name, mutation in [('streaming', {'stream': True}), ('unsupported', {'temperature': 0})]:
            status = server.client.post('/v1/chat/completions', json={**body, **mutation}, headers=server.headers).status_code
            assert status == 400
            evidence['checks'][name] = status
        remote = json.loads(json.dumps(body))
        remote['messages'][0]['content'][0]['image_url']['url'] = 'https://example.invalid/image.png'
        assert server.client.post('/v1/chat/completions', json=remote, headers=server.headers).status_code == 400
        evidence['checks']['remote_url_rejected'] = True
        duplicate = '{"model":"x","model":"y","messages":[]}'
        assert server.client.post('/v1/chat/completions', content=duplicate, headers={**server.headers, "Content-Type": "application/json"}).status_code == 400
        evidence['checks']['duplicate_json_rejected'] = True
        # Send a full valid request then sever TCP while a real worker is alive.
        payload = json.dumps(body).encode()
        connection = socket.create_connection(('127.0.0.1', args.port))
        request = (f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{args.port}\r\nAuthorization: Bearer {server.token}\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n').encode() + payload
        connection.sendall(request)
        workers = wait_until(lambda: [p for p in server.children() if any(arg.endswith('local-ai-ocr-worker.py') for arg in p.cmdline())])
        pids = [p.pid for p in workers]
        busy = server.client.post('/v1/chat/completions', json=body, headers=server.headers)
        assert busy.status_code == 429
        evidence['checks']['busy_real_worker'] = 429
        connection.close()
        wait_until(lambda: not server.children())
        assert not any(psutil.pid_exists(pid) for pid in pids)
        evidence['checks']['disconnect_worker_reaped'] = True
        # Gracefully stop with another real worker in progress.
        connection = socket.create_connection(('127.0.0.1', args.port))
        connection.sendall(request)
        shutdown_pids = [p.pid for p in wait_until(lambda: [p for p in server.children() if any(arg.endswith('local-ai-ocr-worker.py') for arg in p.cmdline())])]
        server.close()
        connection.close()
        assert not any(psutil.pid_exists(pid) for pid in shutdown_pids)
        evidence['checks']['shutdown_worker_reaped'] = True
    finally:
        if server.process.poll() is None:
            server.close()
    timeout_server = RunningServer(store, output, args.port+1, timeout=.05)
    try:
        response = timeout_server.client.post('/v1/chat/completions', json=body, headers=timeout_server.headers)
        assert response.status_code == 504, response.text
        wait_until(lambda: not timeout_server.children())
        evidence['checks']['owned_process_deadline'] = 504
    finally:
        timeout_server.close()
    evidence['checks']['no_worker_survivors'] = True
    (output/'live-verification.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': list(evidence['checks']), 'evidence': str(output/'live-verification.json')}, indent=2))


if __name__ == '__main__':
    main()
