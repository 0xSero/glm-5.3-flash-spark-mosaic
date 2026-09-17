#!/usr/bin/env python3
"""Preserve completed layer timings before the evaluator rotates its state slots."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

PROBE = '''import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
r={}
for name in ["evaluation-identity.json","quality-report.json","variant-stream/progress.json"]:
 f=p/name
 if f.exists():r[name]=json.loads(f.read_text())
r["manifests"]=[json.loads(f.read_text()) for f in (p/"variant-stream").glob("slot-*/manifest.json")]
print(json.dumps(r))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result-root', default='/release/results/original-q3-control')
    args = parser.parse_args()
    result = {'container': args.container, 'layers': {}}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['container'] != args.container:
            raise ValueError('monitor output belongs to another container')
    while True:
        info = json.loads(subprocess.check_output(['docker', 'inspect', args.container]))[0]
        result['container_id'] = info['Id']
        result['container_state'] = info['State']
        result['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if info['State']['Running']:
            probe = subprocess.run(['docker', 'exec', args.container, 'python3', '-c', PROBE, args.result_root], capture_output=True, text=True)
            if probe.returncode == 0:
                data = json.loads(probe.stdout)
                for manifest in data.pop('manifests'):
                    if manifest.get('state') == 'COMPLETE':
                        result['layers'][str(manifest['layer'])] = manifest
                result['latest'] = data
            else:
                result['last_probe_error'] = probe.stderr[-2000:]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        temporary.replace(args.output)
        if not info['State']['Running']:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
