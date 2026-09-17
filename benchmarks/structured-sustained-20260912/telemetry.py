"""Read-only telemetry tied to one existing benchmark process; no GPU setting changes."""
import datetime
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).parent
pid = json.loads((root / 'handoff.json').read_text())['pid']
fields = 'temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used,memory.total'
deadline = time.monotonic() + 14400
with (root / 'telemetry.jsonl').open('x') as out:
    while time.monotonic() < deadline:
        proc = Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or str(root / 'probe.py').encode() not in proc.read_bytes():
            break
        started = time.monotonic()
        p = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        out.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'monotonic': started, 'sample_seconds': time.monotonic() - started,
            'gpu_fields': fields, 'gpu_csv': p.stdout.strip(), 'exit_code': p.returncode,
            'mem_available': mem['MemAvailable'].strip(), 'swap_free': mem['SwapFree'].strip(),
            'loadavg': Path('/proc/loadavg').read_text().strip()}) + '\n')
        out.flush()
        time.sleep(max(0, 2 - (time.monotonic() - started)))
