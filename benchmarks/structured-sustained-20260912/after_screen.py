"""One-shot sequential handoff; never infer terminal state from a stale status file."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).parent
old = root.parent / 'structured-decode-20260912'
expected = '95fa3d37087adf09ff2b3ed9518d234577b11bdbd2d4bddc416509cec03e9f42'


def save(value):
    (root / 'handoff.json').write_text(json.dumps(value, indent=2) + '\n')


while True:
    try:
        os.kill(1125784, 0)
        time.sleep(10)
    except ProcessLookupError:
        break
status = json.loads((old / 'status.json').read_text())
if status.get('phase') != 'completed':
    save({'phase': 'not_started_prior_test_failed', 'prior_status': status})
    raise SystemExit(1)
identity = json.loads(subprocess.check_output(['docker', 'inspect', expected]))[0]
assert identity['Id'] == expected and identity['State']['Running'] and not identity['State']['OOMKilled']
save({'phase': 'starting', 'container_id': expected, 'image_id': identity['Image']})
with (root / 'probe.log').open('w') as log:
    child = subprocess.Popen(['python3', str(root / 'probe.py')], stdout=log, stderr=subprocess.STDOUT, cwd=root)
    save({'phase': 'running', 'pid': child.pid, 'container_id': expected, 'image_id': identity['Image']})
    code = child.wait()
save({'phase': 'terminal', 'exit_code': code, 'status': json.loads((root / 'status.json').read_text())})
