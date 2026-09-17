"""Isolated full draft memory comparison; never starts a server."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=('fp8', 'nvfp4'), required=True)
p.add_argument('--attempt', type=int, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
model = root.parents[1] / 'quality/unpruned-next-candidate/projection/model'
mtp = root / 'native-mtp-view'
image = 'sha256:368820997e1146e9d7843367478b53ce18db708e79861f7ea269860e9a1bda4b'
name = f'glm53-full-draft-{a.mode}-load-{a.attempt}'


def read(*cmd):
    return subprocess.check_output(cmd, text=True, timeout=60)


assert not read('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader').strip()
mem = {row.split(':')[0]: int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable'] >= 64*1024**3, mem['MemAvailable']
assert model.is_dir() and mtp.is_dir()
out = root / f'full-{a.mode}-attempt{a.attempt}'
out.mkdir(exist_ok=False)
source = out / 'source'
source.mkdir()
for script in ('full_load.py', 'policy.py', 'native-parameter-baseline.json', 'launch_full_load.py'):
    shutil.copy2(root / script, source / script)
cid = read('docker', 'run', '-d', '--name', name, '--network', 'none', '--gpus', 'all',
    '--cpus', '4', '--memory', '64g', '--memory-swap', '64g', '--shm-size', '1g',
    '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2',
    '-e', f'GLM53_DRAFT_PROBE_MODE={a.mode}',
    '-e', f'GLM53_MTP_EXPERT_FP8={int(a.mode=="fp8")}',
    '-e', 'VLLM_B12X_MOE_FP4_FORCE_A16=1', '-e', 'VLLM_MXFP8_LM_HEAD=0',
    '-e', 'VLLM_MTP_NVFP4_LM_HEAD=0', '-e', 'VLLM_USE_V2_MODEL_RUNNER=1',
    '-e', 'SAFETENSORS_LOAD_DEVICE=cuda:0', '-e', 'SAFETENSORS_DROP_PAGE_CACHE=1',
    '-e', 'HF_HUB_OFFLINE=1', '-e', 'PYTHONUNBUFFERED=1',
    '-v', f'{model}:/model:ro', '-v', f'{mtp}:/mtp:ro',
    '-v', f'{source}:/proof:ro', '-v', f'{out}:/out',
    '--entrypoint', 'python3', image, '/proof/full_load.py').strip()
(out/'launch.json').write_text(json.dumps({'container_id': cid, 'image_id': image,
    'mode': a.mode, 'MemAvailable_before': mem['MemAvailable'], 'state': 'RUNNING_NOT_ACCEPTED'}, indent=2)+'\n')
(out/'start-inspect.json').write_text(read('docker', 'inspect', cid))
print(cid)
