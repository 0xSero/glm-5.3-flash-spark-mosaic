"""Prepare an isolated sparse288 proof attempt; launching requires parent GPU release."""
import argparse,json,subprocess,shutil,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
IMAGE='sha256:368820997e1146e9d7843367478b53ce18db708e79861f7ea269860e9a1bda4b'
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);a=p.parse_args()
release=json.loads((W/'ROOT_GPU_RELEASE.json').read_text());assert release['state']=='ROOT_RELEASED_FULL288_SPARSE_GPU_PROOF'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU owned'
mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem>=64*2**30 and shutil.disk_usage(W).free>=10*2**30
out=W/f'attempt{a.attempt}';out.mkdir(exist_ok=False);src=out/'source';src.mkdir()
for f in ('full_sparse.py','full_load.py','policy.py','glm_mtp_expert_precision.py','native-parameter-baseline.json','reference_packed.py','packed_routes.py','SOURCE_FREEZE.json','launch.py'):
 shutil.copyfile(W/f,src/f)
(src/'EXECUTION_SHA256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in src.iterdir()},indent=2)+'\n')
model=ROOT/'quality/unpruned-next-candidate/projection/model';mtp=ROOT/'b12x-runtime/mtp-draft-nvfp4/native-mtp-view'
assert model.is_dir() and mtp.is_dir()
cmd=['docker','run','-d','--name',f'glm53-full288-sparse-proof-{a.attempt}','--network','none','--gpus','all','--cpus','4','--memory','64g','--memory-swap','64g','--shm-size','1g']
for env in ['GLM53_DRAFT_PROBE_MODE=nvfp4','GLM53_MTP_EXPERT_NVFP4=1','GLM53_MTP_EXPERT_FP8=0','VLLM_B12X_MOE_FP4_FORCE_A16=1','VLLM_MXFP8_LM_HEAD=0','VLLM_MTP_NVFP4_LM_HEAD=0','VLLM_USE_V2_MODEL_RUNNER=1','SAFETENSORS_LOAD_DEVICE=cuda:0','SAFETENSORS_DROP_PAGE_CACHE=1','HF_HUB_OFFLINE=1','PYTHONUNBUFFERED=1']:cmd+=['-e',env]
cmd+=['-v',f'{model}:/model:ro','-v',f'{mtp}:/mtp:ro','-v',f'{src}:/proof:ro','-v',f'{out}:/out','--entrypoint','python3',IMAGE,'/proof/full_sparse.py']
cid=subprocess.check_output(cmd,text=True).strip();(out/'launch.json').write_text(json.dumps({'cid':cid,'command':cmd,'MemAvailable_before':mem},indent=2)+'\n');print(cid,flush=True)
subprocess.check_output(['docker','wait',cid]);(out/'terminal-inspect.json').write_bytes(subprocess.check_output(['docker','inspect',cid]))
with (out/'terminal.log').open('wb') as f:subprocess.run(['docker','logs',cid],stdout=f,stderr=subprocess.STDOUT,check=True)
i=json.loads((out/'terminal-inspect.json').read_text())[0];assert i['State']['ExitCode']==0 and not i['State']['OOMKilled'],'failed; evidence preserved'
r=json.loads((out/'full-sparse.json').read_text());assert r['state']=='FULL288_SPARSE_MOE_REFERENCE_GRAPH_PASS_SERVING_PENDING' and len(r['cells'])==10
(out/'TERMINAL_RECEIPT.json').write_text(json.dumps({'state':'COMPONENT_PROOF_COMPLETED_SERVING_PENDING','cid':cid,'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}},indent=2)+'\n')
