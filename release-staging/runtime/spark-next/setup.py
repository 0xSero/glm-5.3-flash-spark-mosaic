"""Pinned checkpoint setup and bounded serving acceptance, inside Docker only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

from launch_controls import controls
from prepare_native_mtp_view import prepare

ROOT = Path(__file__).parent
MODEL = Path('/model')
STATE = Path('/state')
LOCK = json.loads((ROOT / 'model-lock.json').read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def contained(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe relative checkpoint path: {name}')
    result = root / p
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Checkpoint symlink escapes model root: {name}')
    return result


def verify(root, lock):
    expected = {r['path']: r for r in lock['files']}
    actual = {str(p.relative_to(root)) for p in root.rglob('*.safetensors')}
    if len(expected) != lock['weight_file_count'] or actual != set(expected):
        raise ValueError('Weight file set differs from pinned checkpoint')
    for name, sha in lock['metadata_sha256'].items():
        if digest(contained(root, name)) != sha:
            raise ValueError(f'Metadata hash mismatch: {name}')
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    if len(index) != lock['indexed_tensor_count'] or set(index.values()) != set(expected):
        raise ValueError('Pinned weight/index closure failed')
    seen = set()
    for i, (name, row) in enumerate(expected.items(), 1):
        path = contained(root, name)
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Weight size/hash mismatch: {name}')
        with path.open('rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError(f'Invalid safetensors header size: {name}')
            header = json.loads(f.read(header_size))
        offset = 0
        for key, tensor in sorted(((k, v) for k, v in header.items() if k != '__metadata__'), key=lambda x: x[1]['data_offsets'][0]):
            start, end = tensor['data_offsets']
            if index.get(key) != name or key in seen or start != offset or end < start:
                raise ValueError(f'Tensor/index/offset mismatch: {key}')
            seen.add(key)
            offset = end
        if offset + header_size + 8 != row['bytes']:
            raise ValueError(f'Tensor payload/file closure failed: {name}')
        print(f'Verified weight {i}/{len(expected)}: {name}', flush=True)
    if seen != set(index):
        raise ValueError('Indexed tensor is absent from checkpoint headers')
    return {'weight_files': len(expected), 'tensor_count': len(seen), 'weight_bytes': sum(r['bytes'] for r in expected.values()), 'manifest_sha256': lock['metadata_sha256']['EXL3_MANIFEST.json'], 'weight_stats': {name: {'bytes': (root / name).stat().st_size, 'mtime_ns': (root / name).stat().st_mtime_ns} for name in expected}}


def stage(download):
    if download:
        from huggingface_hub import snapshot_download
        # Resumes only the pinned public checkpoint. Never writes another repo.
        snapshot_download(repo_id=LOCK['repo'], revision=LOCK['revision'], local_dir=str(MODEL), max_workers=2, allow_patterns=[r['path'] for r in LOCK['files']] + list(LOCK['metadata_sha256']))
    receipt = verify(MODEL, LOCK)
    # A new temporary view is compared with any existing view; source is readonly.
    temp_parent = Path(tempfile.mkdtemp(prefix='mtp-prepare-', dir=STATE))
    try:
        candidate = temp_parent / 'mtp'
        prepare(MODEL, candidate, '/model', MODEL / 'config.json')
        target = STATE / 'mtp'
        if target.exists():
            if {p.name for p in target.iterdir()} != {p.name for p in candidate.iterdir()}:
                raise ValueError('Existing native MTP view differs; use a new state directory')
            for src in candidate.iterdir():
                dst = target / src.name
                matches = dst.is_symlink() and os.readlink(src) == os.readlink(dst) if src.is_symlink() else not dst.is_symlink() and dst.read_bytes() == src.read_bytes()
                if not matches:
                    raise ValueError(f'Existing native MTP view differs: {src.name}')
        else:
            candidate.rename(target)
    finally:
        shutil.rmtree(temp_parent)
    receipt.update(repo=LOCK['repo'], revision=LOCK['revision'], verified_at=time.time(), state='HASH_AND_INDEX_CLOSURE_PASSED', native_mtp_view=view_identity(STATE / 'mtp'))
    (STATE / 'checkpoint-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


def view_identity(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Native MTP view must be a generated directory')
    result = {}
    for file in sorted(path.iterdir()):
        if file.is_symlink():
            result[file.name] = {'symlink': os.readlink(file)}
        elif file.is_file():
            result[file.name] = {'sha256': digest(file)}
        else:
            raise ValueError(f'Unexpected native MTP view entry: {file.name}')
    return result


def verify_view(path, expected):
    if view_identity(path) != expected:
        raise RuntimeError('Native MTP view changed after setup; use a fresh state directory')


def gpu_check():
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip().splitlines()
    if len(names) != 1 or 'GB10' not in names[0]:
        raise RuntimeError(f'Expected one visible DGX Spark GB10 GPU, got {names}')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if any(line.strip().isdigit() for line in jobs.splitlines()):
        raise RuntimeError('GPU has an existing compute job. No job was stopped.')


def runtime_args():
    c = controls(int(os.getenv('MTP_DEPTH', '1')), int(os.getenv('ACTIVE_SLOTS', '1')), int(os.getenv('PREFILL_CHUNK', '2048')))
    fraction = float(os.getenv('MEMORY_FRACTION', '.93'))
    if not 0 < fraction <= .95:
        raise ValueError('MEMORY_FRACTION must be greater than zero and at most0.95')
    c['speculative_config']['model'] = '/state/mtp'
    args = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model', '--served-model-name', 'glm-5.3-flash', '--host', os.getenv('BIND_HOST', '127.0.0.1'), '--port', '18080', '--tensor-parallel-size', '1', '--decode-context-parallel-size', '1', '--no-enable-expert-parallel', '--quantization', 'exl3', '--load-format', 'safetensors', '--dtype', 'bfloat16', '--kv-cache-dtype', 'fp8_ds_mla', '--block-size', '256', '--gpu-memory-utilization', str(fraction), '--max-model-len', '262144', '--max-num-seqs', str(c['max_num_seqs']), '--max-num-batched-tokens', str(c['max_num_batched_tokens']), '--attention-backend', 'B12X', '--additional-config', '{"kda_prefill_backend":"b12x"}', '--enable-chunked-prefill', '--no-enable-prefix-caching', '--mm-processor-cache-gb', '0.1', '--compilation-config', json.dumps(c['compilation_config']), '--generation-config', 'vllm', '--reasoning-parser', 'glm47', '--tool-call-parser', 'glm47', '--enable-auto-tool-choice', '--trust-remote-code', '--speculative-config', json.dumps(c['speculative_config'])]
    return args, c


def http(path, payload=None, timeout=10):
    req = urllib.request.Request('http://127.0.0.1:18080' + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def counters(timeout=10):
    text = http('/metrics', timeout=timeout)
    result = {}
    for name in ('vllm:spec_decode_num_draft_tokens_total', 'vllm:spec_decode_num_accepted_tokens_total'):
        result[name] = sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name + '{') or line.startswith(name + ' '))
    return result


def wait_mtp_counters(before, timeout=15):
    deadline = time.monotonic() + timeout
    delta = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Native MTP metrics did not show finite 0 < accepted <= drafted within {timeout}s: {delta}')
        try:
            after = counters(timeout=min(1, remaining))
            delta = {k: after[k] - before[k] for k in before}
            drafted = delta['vllm:spec_decode_num_draft_tokens_total']
            accepted = delta['vllm:spec_decode_num_accepted_tokens_total']
            if math.isfinite(drafted) and math.isfinite(accepted) and 0 < accepted <= drafted:
                return delta
        except (OSError, TimeoutError):
            pass
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def check_bind_port(host, port=18080):
    # This reserves the endpoint briefly before launch; it never probes an
    # unrelated API. The engine must still acquire the port after release.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reservation.bind((host, port))
        reservation.listen(1)


def serve():
    ready = STATE / 'fresh-text.json'
    ready.unlink(missing_ok=True)
    gpu_check()
    if not (STATE / 'checkpoint-verification.json').is_file():
        raise RuntimeError('Run pinned checkpoint setup first')
    receipt = json.loads((STATE / 'checkpoint-verification.json').read_text())
    if receipt['manifest_sha256'] != LOCK['metadata_sha256']['EXL3_MANIFEST.json']:
        raise RuntimeError('Prepared checkpoint receipt differs from pinned model')
    for name, sha in LOCK['metadata_sha256'].items():
        if digest(contained(MODEL, name)) != sha:
            raise RuntimeError(f'Metadata changed after setup: {name}')
    for row in LOCK['files']:
        stat = contained(MODEL, row['path']).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != receipt['weight_stats'][row['path']]:
            raise RuntimeError(f'Weights changed after setup; rerun verification: {row["path"]}')
    verify_view(STATE / 'mtp', receipt.get('native_mtp_view'))
    args, config = runtime_args()
    (STATE / 'launch.json').write_text(json.dumps({'argv': args, 'controls': config, 'acceptance': 'pending'}, indent=2) + '\n')
    check_bind_port(args[args.index('--host') + 1])
    child = subprocess.Popen(args)
    def stop(signum, frame):
        child.send_signal(signum)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + int(os.getenv('READINESS_TIMEOUT', '1800'))
        while True:
            if child.poll() is not None:
                raise RuntimeError(f'Engine exited before readiness: {child.returncode}')
            try:
                http('/health')
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Engine readiness timed out; inspect container logs')
                time.sleep(5)
        before = counters()
        payload = {'model': 'glm-5.3-flash', 'messages': [{'role': 'user', 'content': 'What is 6 multiplied by 7? Answer only the number.'}], 'temperature': 0, 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': True}}
        response = json.loads(http('/v1/chat/completions', payload, timeout=180))
        choice = response['choices'][0]
        if choice['message']['content'].strip() != '42' or choice['finish_reason'] != 'stop':
            raise RuntimeError(f'Fresh text check failed: {response}')
        delta = wait_mtp_counters(before)
        ready.write_text(json.dumps({'fresh_text_passed': True, 'native_mtp_counters': delta, 'request': payload, 'response': response, 'vision_image_verified': False, 'vision_video_verified': False, 'long_context_verified': False}, indent=2) + '\n')
        print('READY: fresh text and native MTP counters passed. Image/video, long context and performance require separate acceptance.', flush=True)
        raise SystemExit(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'gpu-check', 'serve', 'health', 'print-config'))
    parser.add_argument('--download', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare':
        stage(a.download)
    elif a.command == 'gpu-check':
        gpu_check()
    elif a.command == 'serve':
        serve()
    elif a.command == 'health':
        if not (STATE / 'fresh-text.json').is_file():
            raise SystemExit(1)
        http('/health')
    else:
        print(json.dumps(runtime_args(), indent=2))
