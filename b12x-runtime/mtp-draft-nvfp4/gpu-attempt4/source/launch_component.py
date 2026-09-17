"""Launch one bounded native-expert probe after the exact quality container exits."""
import json
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parents[1] / 'quality/unpruned-next-candidate/projection/model'
IMAGE = 'sha256:368820997e1146e9d7843367478b53ce18db708e79861f7ea269860e9a1bda4b'
QUALITY = '2e0bda63e337c4247105ac3990471781c0cd8f2bfe0e75001d5b74fc5e1e5f67'
NAME = 'glm53-mtp-nvfp4-component-attempt4'


def read(*args):
    return subprocess.check_output(args, text=True, timeout=60)


state = json.loads(read('docker', 'inspect', QUALITY))[0]['State']
assert state['Status'] == 'exited' and state['ExitCode'] == 0 and not state['OOMKilled'], state
processes = read('nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader')
assert not processes.strip(), processes
assert MODEL.is_dir()
out = ROOT / 'gpu-attempt4'
out.mkdir(exist_ok=False)
source = out / 'source'
source.mkdir()
for script in ('policy.py', 'cpu_scope.py', 'gpu_component.py', 'launch_component.py'):
    shutil.copy2(ROOT / script, source / script)
cpu = subprocess.run(['docker', 'run', '--name', NAME + '-cpu', '--network', 'none',
                      '--cpus', '1', '--memory', '4g', '--memory-swap', '4g', '--gpus', 'all',
                      '-e', 'CUDA_VISIBLE_DEVICES=', '-v', f'{source}:/proof:ro',
                      '--entrypoint', 'python3', IMAGE, '/proof/cpu_scope.py'],
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
(out / 'cpu-scope.log').write_text(cpu.stdout)
(out / 'cpu-inspect.json').write_text(read('docker', 'inspect', NAME + '-cpu'))
assert cpu.returncode == 0, cpu.stdout
assert not read('nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader').strip()
cid = read('docker', 'run', '-d', '--name', NAME, '--restart', 'no', '--network', 'none',
           '--cpus', '4', '--memory', '16g', '--memory-swap', '16g', '--gpus', 'all',
           '--shm-size', '1g', '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2',
           '-e', 'VLLM_B12X_MOE_FP4_FORCE_A16=1', '-e', 'GLM53_MTP_EXPERT_FP8=0',
           '-e', 'VLLM_MTP_NVFP4_LM_HEAD=0', '-e', 'VLLM_MXFP8_LM_HEAD=0',
           '-e', 'OMP_NUM_THREADS=4', '-e', 'PYTHONUNBUFFERED=1',
           '-v', f'{source}:/proof:ro', '-v', f'{MODEL}:/model:ro', '-v', f'{out}:/out',
           '--entrypoint', 'python3', IMAGE, '/proof/gpu_component.py').strip()
(out / 'launch.json').write_text(json.dumps({'container_id': cid, 'image_id': IMAGE,
    'quality_terminal_state': state, 'compute_processes_before': processes,
    'state': 'RUNNING_NOT_ACCEPTED'}, indent=2) + '\n')
(out / 'start-inspect.json').write_text(read('docker', 'inspect', cid))
print(cid)
