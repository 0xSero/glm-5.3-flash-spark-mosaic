"""Replay the exact successful baseline prompts after depth/runtime admission."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'b12x-runtime/mtp-depth-sweep'))
import admission
import counters
from common import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', default='1024,16384')
    a = p.parse_args()
    receipt = admission.verify(a.run)
    if receipt['max_num_seqs'] != 1:
        raise ValueError('This exact-baseline replay is C1 only')
    plan = json.loads((a.run / 'plan.json').read_text())
    targets = [int(n) for n in a.targets.split(',')]
    manifest = json.loads((a.fixture / 'manifest.json').read_text())
    sources = ('probe.py', 'run.py', 'timing.py', 'native_metrics.py')
    for name in sources:
        if sha(a.fixture / name) != manifest['files'][name]:
            raise ValueError('Baseline source changed: ' + name)
    fixtures = {}
    verified = {r['file']: r['sha256'] for r in json.loads((a.fixture / 'verified.json').read_text())['records']}
    for target in targets:
        name = f'{target}-measured1.json'
        if sha(a.fixture / name) != verified[name]:
            raise ValueError('Baseline result changed: ' + name)
        f = json.loads((a.fixture / name).read_text())
        if not f['row']['success'] or f['row']['finish_reason'] != 'stop':
            raise ValueError('Baseline fixture was not accepted')
        fixtures[target] = f
    a.output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        shutil.copy2(a.fixture / name, a.output / name)
    write(a.output / 'manifest.json', manifest)
    write(a.output / 'replay-source.json', {'admission_sha256': sha(a.run / 'admission.json'),
        'wrapper_sha256': sha(__file__), 'baseline_sha256': {str(n): verified[f'{n}-measured1.json'] for n in targets},
        'scope': 'C1 exact baseline payloads; warmups excluded, no general quality claim'})
    sys.path.insert(0, str(a.output))
    import probe
    probe.ROOT = a.output
    # Reuse the immutable streaming/timing/semantic implementation. Inputs are
    # copied from accepted raw payloads, so no tokenizer/padding changes arise.
    probe.prepare = lambda target: (fixtures[target]['payload']['messages'][0]['content'], fixtures[target]['actual_prompt_tokens'])
    for target in targets:
        for repeat in ('warmup1', 'warmup2', 'measured1', 'measured2'):
            admission.verify(a.run)
            before = counters.snapshot(probe.BASE + '/metrics', probe.MODEL, plan['depth'], plan['container_name'])
            write(a.output / 'status.json', {'state': 'RUNNING', 'target': target, 'repeat': repeat})
            probe.cell(target, repeat)
            after = counters.snapshot(probe.BASE + '/metrics', probe.MODEL, plan['depth'], plan['container_name'])
            admission.verify(a.run)
            path = a.output / f'{target}-{repeat}.json'
            value = json.loads(path.read_text())
            # Retain raw output even if payload/counter validation fails.
            write(a.output / f'positions-{target}-{repeat}.json', {'before': before, 'after': after})
            if value['payload'] != fixtures[target]['payload']:
                raise ValueError('Exact baseline request payload mismatch')
            delta = counters.compare(before, after)
            write(a.output / f'position-delta-{target}-{repeat}.json', delta)
    write(a.output / 'status.json', {'state': 'COMPLETED_REQUIRES_INDEPENDENT_VERIFICATION'})


if __name__ == '__main__':
    main()
