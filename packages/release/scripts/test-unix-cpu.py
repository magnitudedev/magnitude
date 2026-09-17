"""Consume a final CPU archive on its native Unix host and run real inference."""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import pathlib
import platform
import signal
import subprocess
import struct
import tarfile
import time
import urllib.parse
import urllib.request
import uuid
import zlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-directory', type=pathlib.Path, required=True)
parser.add_argument('--output-directory', type=pathlib.Path, required=True)
parser.add_argument('--model', choices=['lfm2.5-2.6b:gguf:q4', 'qwen3.5-4b:gguf:q4'], default='lfm2.5-2.6b:gguf:q4')
args = parser.parse_args()
assert platform.system() == 'Darwin', 'This consumer currently validates macOS CPU artifacts'
host = {'arm64': 'darwin-arm64', 'x86_64': 'darwin-x64'}[platform.machine()]
root = args.output_directory.resolve()
root.mkdir(parents=True, exist_ok=False)
results = []
def record(name, detail):
    results.append({'test': name, 'detail': detail})
    (root / 'results.json').write_text(json.dumps(results, indent=2))
    print('PASS', name, flush=True)
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

metadata = json.loads((args.artifact_directory / f'icn-base-{host}.artifact.json').read_text())
assert metadata['id'] == f'icn-base-{host}' and metadata['host'] == host
assert metadata['kind'] == 'icn-base' and metadata['backend'] == 'cpu'
assert metadata['filename'] == f'magnitude-icn-base-{host}.tar.gz'
archive = args.artifact_directory / metadata['filename']
assert archive.stat().st_size == metadata['bytes'] and digest(archive) == metadata['sha256']
record('artifact-integrity', metadata)
installation = root / 'installation'
installation.mkdir()
with tarfile.open(archive) as files:
    files.extractall(installation, filter='data')
declaration = dict(schemaVersion=1, backend='cpu', nativeBuild=metadata['nativeBuild'], backendModuleAbi=metadata['backendModuleAbi'])
(installation / 'installation.json').write_text(json.dumps(declaration))
binary = installation / 'bin/magnitude-inference'
architecture = subprocess.check_output(['/usr/bin/lipo', '-archs', str(binary)], text=True).strip()
assert architecture == platform.machine(), architecture
record('native-architecture', {'host': host, 'architecture': architecture})

model = args.model
catalog_path = '/api/v1/catalog/models/' + urllib.parse.quote(model, safe='')

token = uuid.uuid4().hex
origin = 'http://127.0.0.1:18843'
def request(path, body=None, raw=False):
    req = urllib.request.Request(origin + path, data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        value = response.read().decode()
    return value if raw else (json.loads(value) if value else None)
def payload(text, tokens=2048):
    return dict(model=model, messages=[dict(role='user', content=text)], temperature=0, seed=42, max_tokens=tokens, reasoning_effort='none')
def generate(text):
    response = request('/v1/chat/completions', payload(text))
    assert (response['choices'][0]['message']['content'] or '').strip(), response
    assert response['usage']['completion_tokens'] > 0
    assert response['choices'][0]['finish_reason'] in ('stop', 'length')
    return response

environment = {key: value for key, value in os.environ.items() if not key.startswith(('DYLD_', 'LD_'))}
environment.update(MAGNITUDE_ICN_AUTH_TOKEN=token, RUST_LOG='info')
with (root / 'server.log').open('w') as log:
    process = subprocess.Popen([str(binary), 'serve', '--bind', '127.0.0.1:18843', '--instance-id', 'cpu-acceptance-' + uuid.uuid4().hex,
        '--installation', str(installation / 'installation.json'), '--model-store', str(root / 'models'),
        '--cache-root', str(root / 'cache')],
        env=environment, stdout=log, stderr=log, start_new_session=True)
    try:
        for _ in range(120):
            assert process.poll() is None, 'Engine exited before readiness; inspect server.log'
            try:
                health = request('/health')
                break
            except OSError:
                time.sleep(1)
        else:
            raise AssertionError('Engine did not become healthy')
        record('health', health)
        hardware = request('/api/v1/hardware')
        assert hardware['native_build'] == metadata['nativeBuild']
        assert hardware['enabled_backends'] == ['cpu']
        record('hardware-identity', hardware)
        catalog_model = request(catalog_path)
        assert catalog_model['id'] == model
        assert catalog_model['localState']['_tag'] == 'NotInstalled'
        assert catalog_model['desired']['profile']['contextLength'] == 64000
        record('catalog-model', catalog_model)
        admission = request(catalog_path + '/install', {})
        assert admission['_tag'] == 'Admitted', admission
        record('catalog-install-admission', admission)
        operation_path = '/api/v1/catalog/installations/' + admission['operationId']
        for _ in range(1200):
            operation = request(operation_path)
            assert operation['modelId'] == model
            state = operation['state']['_tag']
            if state == 'Completed':
                break
            if state in ('Failed', 'Cancelled'):
                record('catalog-install-failure', operation)
                raise AssertionError(operation)
            time.sleep(1)
        else:
            raise AssertionError('Catalog installation did not complete')
        record('catalog-install-completed', operation)
        installed = request(catalog_path)
        assert installed['localState']['_tag'] == 'Installed'
        assert installed['localState']['effective']['_tag'] == 'Ready', installed
        assert installed['localState']['installation']['ownership'] == 'Magnitude'
        assert installed['localState']['updateState']['_tag'] == 'Current'
        record('catalog-installed', installed)
        # Independently check the content-addressed bytes fetched by the product downloader.
        blobs = sorted(blob for blob in (root / 'models' / 'hub').glob('models--*/blobs/lfs-sha256-*') if blob.suffix != '.integrity')
        assert len(blobs) == 2, 'Catalog target and its required draft or projector must both be installed'
        integrity = []
        for blob in blobs:
            assert not blob.name.endswith('.incomplete'), blob
            actual = digest(blob)
            assert actual == blob.name.removeprefix('lfs-sha256-'), blob
            integrity.append({'file': str(blob.relative_to(root)), 'bytes': blob.stat().st_size, 'sha256': actual})
        record('catalog-material-integrity', integrity)
        assert request(catalog_path + '/install', {})['_tag'] == 'Current'
        record('catalog-install-idempotent', {'model': model})
        for _ in range(120):
            models = request('/v1/models')
            if model in [item['id'] for item in models['data']]:
                break
            time.sleep(1)
        else:
            raise AssertionError('Model was not discovered')
        response = generate('What is the capital of France? Answer in one sentence.')
        assert 'Paris' in response['choices'][0]['message']['content']
        record('baseline', response)
        instances = request('/api/v1/instances')
        ready = [item for item in instances['instances'] if item['modelId'] == model and item['lifecycle']['_tag'] == 'Ready']
        assert len(ready) == 1
        assert sum(domain['modelBytes'] for domain in ready[0]['lifecycle']['allocation']['memoryDomains']) > 0
        record('resident-allocation', instances)
        if model.startswith('qwen3.5-4b:'):
            # Deterministic lossless RGB fixture: image understanding must use the shipped projector.
            def png_chunk(kind, data):
                return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
            pixels = (b'\x00' + b'\xff\x00\x00' * 96) * 96
            png = b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', struct.pack('>IIBBBBB', 96, 96, 8, 2, 0, 0, 0)) + png_chunk(b'IDAT', zlib.compress(pixels)) + png_chunk(b'IEND', b'')
            body = payload('What color fills this image? Answer with the color name.')
            body['messages'][0]['content'] = [
                {'type': 'text', 'text': 'What color fills this image? Answer with the color name.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
            ]
            image_response = request('/v1/chat/completions', body)
            assert 'red' in (image_response['choices'][0]['message']['content'] or '').lower(), image_response
            assert image_response['usage']['completion_tokens'] > 0
            assert image_response['choices'][0]['finish_reason'] in ('stop', 'length')
            record('catalog-projector-image-inference', image_response)
        for index in range(4):
            record(f'repeat-{index}', generate('What is the capital of France? Answer in one sentence.'))
        body = payload('What is the capital of France? Answer in one sentence.')
        body['stream'] = True
        stream = request('/v1/chat/completions', body, raw=True)
        assert 'data: [DONE]' in stream
        chunks = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: {')]
        assert ''.join(choice['delta'].get('content') or '' for chunk in chunks for choice in chunk.get('choices', []))
        record('streaming', stream)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            record('concurrent', list(pool.map(generate, ['What is the capital of France? Answer in one sentence.', 'What is the capital of Italy? Answer in one sentence.', 'What is two plus two? Answer in one sentence.'])))
        response = generate('The gardener waters the apple trees every morning. ' * 200 + 'What fruit grows on these trees? Answer in one sentence.')
        assert response['usage']['prompt_tokens'] >= 1000
        record('long-prefill', response)
        request('/api/v1/instances/' + ready[0]['id'] + '/stop', {})
        assert not any(item['id'] == ready[0]['id'] and item['lifecycle']['_tag'] == 'Ready' for item in request('/api/v1/instances')['instances'])
        record('unload', ready[0]['id'])
        record('reload', generate('What is the capital of France?'))
        record('final-health', request('/health'))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        # The fixture owns this entire group, including any workers left after parent exit.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        (root / 'cleanup.json').write_text(json.dumps({'parentPid': process.pid, 'returncode': process.returncode}))
